#!/usr/bin/env julia
#
# Phase 0 microbenchmark: Multi-RHS sparse solve scaling via cuDSS.
#
# Question: Is "1 factorization + M triangular solves" faster than
# "M independent factorizations + M independent solves" (cuDSS uniform batch)?
#
# Tests both the reduced Schur complement S (~4000 for ACTIVSg2000) and
# the full monolithic Jacobian (~9694).

using GradPower, CUDA, CUDSS, SparseArrays, Printf, LinearAlgebra
using CUDA.CUSPARSE

const ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

# ── Build the test system ──
ps = GradPower.from_psse("examples/ACTIVSg2000.raw", "examples/ACTIVSg2000.dyr")
GradPower.build_network!(ps); GradPower.runpf!(ps)
for d in ps.dynamic.devices
    if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
end
dp = GradPower.DynamicProblem(ps)
GradPower.initialize_dynamics!(dp, ps)
GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 0.1, 0.2))

# ── Helper: get a realistic sparse matrix (S or J) with numeric values ──
# Use DynamicProblem's preallocated Jacobian for correct structure.
dyn = ps.dynamic::GradPower.PowerSystemDynamics
L = dyn.layout::GradPower.SimulationLayout
dt = 1.0/120.0
sys_dim = length(dp.zvec)

# Build J as a proper SparseMatrixCSC
J_cpu = GradPower.preallocate_jacobian(ps)
z0 = copy(dp.zvec)
u0 = copy(dp.uvec)
p0 = copy(dp.pvec)
f0 = zeros(Float64, sys_dim)
ybus = ps.network.ybus_real

# Evaluate residual and Jacobian
GradPower.rhs_fun!(f0, z0, u0, p0, ps)
GradPower.rhs_jac!(J_cpu, z0, u0, p0, ps)

# Apply backward Euler scaling
diff_dim = dyn.diff_dim
J_rows = rowvals(J_cpu)
for col in 1:sys_dim
    for nz_idx in nzrange(J_cpu, col)
        row = J_rows[nz_idx]
        if row <= diff_dim && row == col
            J_cpu.nzval[nz_idx] = 1.0 - dt * J_cpu.nzval[nz_idx]
        elseif row <= diff_dim
            J_cpu.nzval[nz_idx] = -dt * J_cpu.nzval[nz_idx]
        end
    end
end

# Build S via SchurWorkspace
sw = GradPower.SchurWorkspace(ps)
n_red = length(sw.reduced_idx)

# Assemble S with real numeric values
ct = dyn.clusters::GradPower.ClusterTable
net_ptr = dyn.diff_dim + dyn.alg_dim
GradPower.assemble_schur!(sw, J_cpu, ct, net_ptr)
S_cpu = copy(sw.S)

println("System dimensions:")
@printf("  Full Jacobian:  %d × %d, nnz = %d\n", sys_dim, sys_dim, nnz(J_cpu))
@printf("  Reduced S:      %d × %d, nnz = %d\n", n_red, n_red, nnz(S_cpu))
println()
flush(stdout)

# ── Benchmark function: single-factor multi-RHS solve ──
function bench_multi_rhs(A_csc::SparseMatrixCSC, label::String, M_values::Vector{Int};
                         n_warmup=3, n_repeat=20)
    n = size(A_csc, 1)
    A_csr = CuSparseMatrixCSR(A_csc)

    # Create solver and do analysis + factorization once
    solver = CudssSolver(A_csr, "G", 'F')
    x_tmp = CUDA.zeros(Float64, n)
    b_tmp = CUDA.zeros(Float64, n)
    cudss("analysis", solver, x_tmp, b_tmp)
    cudss("factorization", solver, x_tmp, b_tmp; asynchronous=false)

    println("── $label: Single factorization, multi-RHS solve ──")
    @printf("  %-6s  %10s  %10s  %10s  %10s\n",
            "M", "Solve (ms)", "per-RHS", "Factor+Solve", "per-RHS")
    flush(stdout)

    # Also measure factorization time once for reference
    CUDA.synchronize()
    t_factor = 0.0
    for _ in 1:n_repeat
        CUDA.synchronize()
        t0 = time_ns()
        cudss("refactorization", solver, x_tmp, b_tmp; asynchronous=false)
        CUDA.synchronize()
        t_factor += time_ns() - t0
    end
    t_factor_ms = t_factor / (1e6 * n_repeat)
    @printf("  Factor: %.3f ms (single system, for reference)\n", t_factor_ms)
    flush(stdout)

    results = Dict{Int, NamedTuple{(:solve_ms, :per_rhs_ms, :factor_ms), Tuple{Float64,Float64,Float64}}}()

    for M in M_values
        B = CUDA.rand(Float64, n, M)
        X = CUDA.zeros(Float64, n, M)
        CUDA.synchronize()

        # Warmup
        for _ in 1:n_warmup
            cudss("solve", solver, X, B; asynchronous=false)
            CUDA.synchronize()
        end

        # Timed runs
        t_solve = 0.0
        for _ in 1:n_repeat
            CUDA.synchronize()
            t0 = time_ns()
            cudss("solve", solver, X, B; asynchronous=false)
            CUDA.synchronize()
            t_solve += time_ns() - t0
        end
        solve_ms = t_solve / (1e6 * n_repeat)
        per_rhs = solve_ms / M
        total = t_factor_ms + solve_ms
        total_per_rhs = total / M

        results[M] = (solve_ms=solve_ms, per_rhs_ms=per_rhs, factor_ms=t_factor_ms)

        @printf("  %-6d  %10.3f  %10.4f  %10.3f  %10.4f\n",
                M, solve_ms, per_rhs, total, total_per_rhs)
        flush(stdout)
    end
    println()
    flush(stdout)

    return results
end

# ── Benchmark function: cuDSS uniform batch (M independent factorizations) ──
function bench_ubatch(A_csc::SparseMatrixCSC, label::String, M_values::Vector{Int};
                      n_warmup=3, n_repeat=20)
    n = size(A_csc, 1)
    nnz_A = nnz(A_csc)

    # CSC → CSR permutation (computed once)
    idx_csc = SparseMatrixCSC(n, n, copy(A_csc.colptr), copy(rowvals(A_csc)),
                               collect(Float64, 1:nnz_A))
    idx_csr = copy(idx_csc')
    perm = Int32.(idx_csr.nzval)

    A_csr_template = CuSparseMatrixCSR(A_csc)

    println("── $label: cuDSS uniform batch (M independent systems) ──")
    @printf("  %-6s  %10s  %10s  %10s  %10s\n",
            "M", "Factor(ms)", "Solve(ms)", "Total(ms)", "per-scen")
    flush(stdout)

    results = Dict{Int, NamedTuple{(:factor_ms, :solve_ms, :total_ms, :per_scen_ms), Tuple{Float64,Float64,Float64,Float64}}}()

    for M in M_values
        # Build batched nzval: (nnz, M) matrix
        nzval_batched = CUDA.zeros(Float64, nnz_A, M)
        perm_gpu = CuVector(perm)

        # Fill each column with the CSR-permuted values
        nzval_csc = CuVector(A_csc.nzval)
        for m in 1:M
            # Apply CSC→CSR permutation: csr_nz[i] = csc_nz[perm[i]]
            nzval_col = view(nzval_batched, :, m)
            # Simple gather on CPU, upload
            csr_vals = A_csc.nzval[Int.(idx_csr.nzval)]
            copyto!(nzval_col, CuVector(csr_vals))
        end

        # Create batched solver
        solver_b = CudssSolver(A_csr_template.rowPtr, A_csr_template.colVal,
                                nzval_batched, "G", 'F')
        cudss_set(solver_b, "ubatch_size", M)

        sol_buf = CUDA.zeros(Float64, n, M)
        rhs_buf = CUDA.rand(Float64, n, M)

        sol_wrap = CudssMatrix(Float64, n; nbatch=M)
        cudss_update(sol_wrap, sol_buf)
        rhs_wrap = CudssMatrix(Float64, n; nbatch=M)
        cudss_update(rhs_wrap, rhs_buf)

        cudss("analysis", solver_b, sol_wrap, rhs_wrap)
        CUDA.synchronize()

        # Warmup
        for _ in 1:n_warmup
            cudss("factorization", solver_b, sol_wrap, rhs_wrap; asynchronous=false)
            cudss("solve", solver_b, sol_wrap, rhs_wrap; asynchronous=false)
            CUDA.synchronize()
        end

        # Timed factorization
        t_factor = 0.0
        for _ in 1:n_repeat
            CUDA.synchronize()
            t0 = time_ns()
            cudss("refactorization", solver_b, sol_wrap, rhs_wrap; asynchronous=false)
            CUDA.synchronize()
            t_factor += time_ns() - t0
        end

        # Timed solve
        t_solve = 0.0
        for _ in 1:n_repeat
            CUDA.synchronize()
            t0 = time_ns()
            cudss("solve", solver_b, sol_wrap, rhs_wrap; asynchronous=false)
            CUDA.synchronize()
            t_solve += time_ns() - t0
        end

        factor_ms = t_factor / (1e6 * n_repeat)
        solve_ms = t_solve / (1e6 * n_repeat)
        total_ms = factor_ms + solve_ms
        per_scen = total_ms / M

        results[M] = (factor_ms=factor_ms, solve_ms=solve_ms, total_ms=total_ms, per_scen_ms=per_scen)

        @printf("  %-6d  %10.3f  %10.3f  %10.3f  %10.4f\n",
                M, factor_ms, solve_ms, total_ms, per_scen)
        flush(stdout)
    end
    println()
    flush(stdout)

    return results
end

# ── Run benchmarks ──
M_values = [1, 2, 4, 8, 16, 32, 64, 128]

println("=" ^ 70)
println("PHASE 0: Multi-RHS Sparse Solve Scaling Benchmark")
println("=" ^ 70)
println()
flush(stdout)

# 1. Reduced system S
r_multi_S = bench_multi_rhs(S_cpu, "Reduced S ($(n_red)×$(n_red))", M_values)
r_batch_S = bench_ubatch(S_cpu, "Reduced S ($(n_red)×$(n_red))", M_values)

# 2. Full monolithic Jacobian
r_multi_J = bench_multi_rhs(J_cpu, "Full J ($(sys_dim)×$(sys_dim))", M_values)
r_batch_J = bench_ubatch(J_cpu, "Full J ($(sys_dim)×$(sys_dim))", M_values)

# ── Summary comparison ──
println("=" ^ 70)
println("SUMMARY: Speedup of shared-factor multi-RHS over uniform batch")
println("=" ^ 70)
println()

for (label, r_multi, r_batch) in [
    ("Reduced S", r_multi_S, r_batch_S),
    ("Full J", r_multi_J, r_batch_J),
]
    println("── $label ──")
    @printf("  %-6s  %12s  %12s  %10s\n",
            "M", "Batch(ms)", "Multi-RHS(ms)", "Speedup")
    for M in M_values
        batch_total = r_batch[M].total_ms
        multi_total = r_multi[M].factor_ms + r_multi[M].solve_ms
        speedup = batch_total / multi_total
        @printf("  %-6d  %12.3f  %12.3f  %10.1fx\n",
                M, batch_total, multi_total, speedup)
    end
    println()
end
flush(stdout)

# ── Go/No-Go Decision ──
println("=" ^ 70)
println("GO/NO-GO DECISION")
println("=" ^ 70)
# The key metric: at M=64, is multi-RHS solve < 3ms for S?
if haskey(r_multi_S, 64)
    solve_64 = r_multi_S[64].solve_ms
    factor_S = r_multi_S[64].factor_ms
    total = factor_S + solve_64
    @printf("\nReduced S at M=64:\n")
    @printf("  Factorization: %.3f ms (paid once per refresh, amortized over ~720 Newton iters)\n", factor_S)
    @printf("  Multi-RHS solve: %.3f ms (paid every Newton iteration)\n", solve_64)
    @printf("  Total per Newton iter: %.3f ms (factor amortized: %.4f ms + solve: %.3f ms)\n",
            factor_S/720 + solve_64, factor_S/720, solve_64)
    println()
    if solve_64 < 3.0
        println("  ✓ GO — Multi-RHS solve is fast enough for the shared-factor architecture")
    elseif solve_64 < 10.0
        println("  ~ MARGINAL — Multi-RHS solve is borderline; proceed with caution")
    else
        println("  ✗ NO-GO — Multi-RHS solve is too slow; consider CUDA streams approach")
    end
end
println()
flush(stdout)
