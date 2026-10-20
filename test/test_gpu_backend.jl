# Unit tests for GPU backend (phase 14b D5).
# All GPU tests guarded by CUDA.functional() — skip gracefully on CPU-only machines.
# The sparse direct solver tests run once per available GPU solver
# (CUDSS.jl and/or SparseDirectSolver.jl). If CUDA or both solvers are
# unavailable, all tests are skipped.

using LinearAlgebra, SparseArrays

const HAS_CUDA_DEVICE = try
    @eval using CUDA
    @eval using CUDA.CUSPARSE
    CUDA.functional()
catch
    false
end

const GPU_SOLVERS = GradPower.AbstractDirectSolverBackend[]
if HAS_CUDA_DEVICE
    try
        @eval using CUDSS
        push!(GPU_SOLVERS, CUDSSBackend())
    catch err
        @info "CUDSS.jl not available — skipping CUDSSBackend GPU tests" err
    end
    try
        @eval using SparseDirectSolver
        push!(GPU_SOLVERS, SparseDirectSolverBackend())
    catch err
        @info "SparseDirectSolver.jl not available — skipping SparseDirectSolverBackend GPU tests" err
    end
end

const HAS_CUDA = HAS_CUDA_DEVICE && !isempty(GPU_SOLVERS)

if !HAS_CUDA
    @info "No functional CUDA device or no GPU direct solver available — skipping GPU backend tests"
end

@testset "gpu_backend" begin

@testset "CuArray GpuBatchedLayout construction" begin
    HAS_CUDA || return

    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)
    M = 4
    gbl = ext.GpuBatchedLayout(dp, ps, M)

    @test gbl.M == M
    @test gbl.sys_dim == length(dp.zvec)
    @test gbl.z isa CuMatrix{Float64}
    @test size(gbl.z) == (M, gbl.sys_dim)
    @test gbl.p isa CuMatrix{Float64}
    @test gbl.u isa CuMatrix{Float64}
    @test gbl.f isa CuMatrix{Float64}

    # Each scenario matches single-scenario IC
    z_cpu = Array(gbl.z)
    p_cpu = Array(gbl.p)
    u_cpu = Array(gbl.u)
    for m in 1:M
        @test z_cpu[m, :] == dp.zvec
        @test p_cpu[m, :] == dp.pvec
        @test u_cpu[m, :] == dp.uvec
    end
end

@testset "GPU direct-solver preconditioner matches KLU ($(nameof(typeof(ls))))" for ls in GPU_SOLVERS
    HAS_CUDA || return

    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)

    Y = ps.network.ybus_real
    n = size(Y, 1)

    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

    # Use a well-conditioned RHS: b = Y * x_true where x_true is random
    x_true = randn(n)
    b = Y * x_true

    # KLU reference
    klu_fact = GradPower.KLU.klu(Y)
    y_cpu = similar(b)
    ldiv!(y_cpu, klu_fact, b)

    # GPU direct solver
    prec = ext.CuDSSPreconditioner(Y; linear_solver = ls)
    b_gpu = CuVector(b)
    y_gpu = CUDA.zeros(Float64, n)
    ldiv!(y_gpu, prec, b_gpu)
    y_from_gpu = Array(y_gpu)

    # Both should have small backward error (relative residual).
    residual_klu = norm(Y * y_cpu - b) / norm(b)
    residual_gpu = norm(Y * y_from_gpu - b) / norm(b)

    @test residual_klu <= 1e-10
    @test residual_gpu <= 1e-10
end

@testset "Batched cuBLAS LU matches CPU" begin
    HAS_CUDA || return

    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

    w_k = 5
    n_clusters = 3
    M = 4
    batch = n_clusters * M

    # Random diagonally-dominant dense systems
    A_cpu = [randn(w_k, w_k) + 10.0 * I for _ in 1:batch]
    B_cpu = [randn(w_k, 2) for _ in 1:batch]

    # CPU reference: LU + solve
    x_ref = similar.(B_cpu)
    for b in 1:batch
        F = lu(A_cpu[b])
        x_ref[b] = F \ B_cpu[b]
    end

    # GPU path
    glu = ext.GpuBatchedLU(w_k, n_clusters, M)
    # Pack A and B into 3D arrays
    A_packed = zeros(Float64, w_k, w_k, batch)
    for b in 1:batch; A_packed[:, :, b] .= A_cpu[b]; end
    copyto!(glu.A_packed, CuArray(A_packed))

    B_packed = zeros(Float64, w_k, 2, batch)
    for b in 1:batch; B_packed[:, :, b] .= B_cpu[b]; end
    copyto!(glu.B_packed, CuArray(B_packed))

    ext.gpu_batched_lu_factor!(glu)
    ext.gpu_batched_lu_solve!(glu)

    B_result = Array(glu.B_packed)
    max_err = 0.0
    for b in 1:batch
        err = maximum(abs, B_result[:, :, b] - x_ref[b])
        max_err = max(max_err, err)
    end
    @test max_err <= 1e-12
end

@testset "GPU Schur operator matches CPU assembled S" begin
    HAS_CUDA || return

    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    ct = dyn.clusters::GradPower.ClusterTable
    net_ptr = dyn.diff_dim + dyn.alg_dim
    dt = 1.0 / 120.0

    # Build SchurWorkspace, assemble S, then build GPU operator from J
    sw = GradPower.SchurWorkspace(ps)
    J0 = GradPower.preallocate_jacobian(ps)
    GradPower.beuler_jac_batched!(J0, dp.zvec, dp.uvec, dp.pvec,
                                   dyn, ps.network.ybus_real,
                                   dyn.layout::GradPower.SimulationLayout,
                                   dyn.diff_dim, dt)
    GradPower.assemble_schur!(sw, J0, ct, net_ptr)

    # GPU operator constructed from J (re-extracts A_k/B_k/C_k independently)
    op = ext.GpuSchurOperator(sw, J0, ct, net_ptr)
    n_red = length(sw.reduced_idx)

    @test size(op) == (n_red, n_red)
    @test eltype(op) == Float64

    # Compare against CPU-assembled S * x for multiple random vectors
    for _ in 1:5
        x = randn(n_red)
        y_ref = zeros(n_red)
        mul!(y_ref, sw.S, x)

        x_d = CuVector(x)
        y_d = CUDA.zeros(Float64, n_red)
        mul!(y_d, op, x_d)
        y_gpu = Array(y_d)

        @test maximum(abs, y_gpu - y_ref) <= 1e-12
    end
end

# Devices on one bus write the same Jacobian / Schur entries; the GPU kernels
# must accumulate them without a data race (ACTIVSg cases have hundreds of
# such buses). testdata/ieee9_multiload adds loads 5/'2', 5/'3', 8/'2' to
# ieee9_v33 and splits generator 2 into two machines (two clusters on bus 2).
@testset "GPU with several devices per bus matches CPU ($(nameof(typeof(ls))))" for ls in GPU_SOLVERS
    HAS_CUDA || return
    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

    ps = from_psse(joinpath(@__DIR__, "testdata", "ieee9_multiload.raw"),
                   joinpath(@__DIR__, "testdata", "ieee9_multiload_gov.dyr"))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    zt = ps.dynamic.layout.zipload
    @test length(unique(zt.bus)) < zt.n
    cl_bus = [c.bus for c in ps.dynamic.clusters.clusters]
    @test length(unique(cl_bus)) < length(cl_bus)

    # Jacobian at a perturbed state
    dyn = ps.dynamic; L = dyn.layout; dt = 1/120
    zp = dp.zvec .+ 1e-3 .* sin.(1:length(dp.zvec))
    J = GradPower.preallocate_jacobian(ps)
    GradPower.beuler_jac!(J, zp, dp.zvec, dp.uvec, dp.pvec, ps, dyn.diff_dim, dt)
    M = 3
    gbl = ext.GpuBatchedLayout(dp, ps, M; linear_solver = ls)
    copyto!(gbl.zold, CuMatrix(repeat(permutedims(dp.zvec), M)))
    copyto!(gbl.z, CuMatrix(repeat(permutedims(zp), M)))
    ext._beuler_jac_all_scenarios_gpu!(gbl, dyn, L, dt)
    Jg = Array(gbl.J_nzval)
    for m in 1:M
        @test maximum(abs.(Jg[m, :] .- nonzeros(J))) < 1e-12
    end

    # Trajectories of both exact GPU paths vs CPU full Newton
    z0 = copy(dp.zvec)
    add_event!(ps, GradPower.ContingencyEvent(4, 0.02, 0.1, 0.2))
    _, tr_cpu = integrate!(dp, ps, 0.5; jacobian_reuse = false, predictor = false)
    for f in (ext.integrate_gpu_cudss!, ext.integrate_gpu_schur_cudss!)
        dp.zvec .= z0
        g1 = ext.GpuBatchedLayout(dp, ps, 1; linear_solver = ls)
        _, tr = f(g1, ps, 0.5)
        @test maximum(abs.(tr[1] .- tr_cpu)) < 1e-6
    end
    empty!(ps.dynamic.events)
end

@testset "KA CUDA kernel dispatch — single-scenario residual" begin
    HAS_CUDA || return

    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

    ps = from_psse(joinpath(@__DIR__, "..", "examples", "2bus.raw"),
                   joinpath(@__DIR__, "..", "examples", "2bus.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    L = dyn.layout::GradPower.SimulationLayout

    # CPU reference
    f_ref = zeros(length(dp.zvec))
    GradPower._rhs_fun_batched!(f_ref, dp.zvec, dp.uvec, dp.pvec,
                                 dyn, ps.network.ybus_real, L)

    # GPU path: build GpuBatchedLayout with M=1, evaluate residual
    gbl = ext.GpuBatchedLayout(dp, ps, 1)
    ext._residual_all_scenarios_gpu!(gbl, dyn, L)
    f_gpu = Array(gbl.f)

    diff = maximum(abs, f_gpu[1, :] - f_ref)
    @test diff <= 1e-12
end

# -----------------------------------------------------------------------
# GPU integration tests — trajectory accuracy against CPU reference
# -----------------------------------------------------------------------

@testset "integrate_gpu_cudss! accuracy ($(nameof(typeof(ls))))" for ls in GPU_SOLVERS
    HAS_CUDA || return
    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 0.1, 0.2))

    # CPU batched reference
    bl = GradPower.BatchedLayout(dp, ps, 1)
    tvec_cpu, trajs_cpu = GradPower.integrate_batched!(bl, ps, 1.0; dt=1.0/120.0)

    for M in [1, 4]
        gbl = ext.GpuBatchedLayout(dp, ps, M; linear_solver = ls)
        tvec_gpu, trajs_gpu = ext.integrate_gpu_cudss!(gbl, ps, 1.0; dt=1.0/120.0)
        for m in 1:M
            err = maximum(abs.(trajs_gpu[m] .- trajs_cpu[1]))
            @test err < 1e-5
        end
    end
end

# Batches larger than MAX_UBATCH are split over several direct solvers (cuDSS
# 0.8 uniform batches of >= 16 are an order of magnitude slower). Scenarios
# differ (scaled loads) so a mix-up of chunk columns would show.
@testset "Chunked batch solver: M > MAX_UBATCH matches single runs ($(nameof(typeof(ls))))" for ls in GPU_SOLVERS
    HAS_CUDA || return
    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(4, 0.02, 0.1, 0.2))
    z0 = copy(dp.zvec); p0 = copy(dp.pvec)
    zl = ps.dynamic.layout.zipload
    function scaled_p(m)
        p = copy(p0)
        for k in 1:zl.n, o in (0, 1, 7, 8)   # pl, ql, yreal, yimag
            p[Int(zl.par_ptr[k]) + o] *= 1 + 0.02 * (m - 1)
        end
        p
    end

    M = ext.MAX_UBATCH + 5
    nchunks = cld(M, ext._ubatch_chunk(ls, M))   # 2 for CUDSS, 1 otherwise
    gbl = ext.GpuBatchedLayout(dp, ps, M; linear_solver = ls)
    @test length(gbl.cudss_solver_batched.solvers) == nchunks
    @test length(gbl.cudss_S_solver_batched.solvers) == nchunks
    for f in (ext.integrate_gpu_cudss!, ext.integrate_gpu_schur_cudss!)
        dp.zvec .= z0
        gb = ext.GpuBatchedLayout(dp, ps, M; linear_solver = ls)
        copyto!(gb.p, CuMatrix(permutedims(reduce(hcat, scaled_p(m) for m in 1:M))))
        _, trb = f(gb, ps, 0.5)
        @test maximum(abs.(trb[M] .- trb[1])) > 1e-3     # scenarios really differ
        for m in (1, ext.MAX_UBATCH, ext.MAX_UBATCH + 1, M)  # both sides of the chunk edge
            g1 = ext.GpuBatchedLayout(dp, ps, 1; linear_solver = ls)
            copyto!(g1.p, CuMatrix(permutedims(scaled_p(m))))
            _, tr1 = f(g1, ps, 0.5)
            @test maximum(abs.(trb[m] .- tr1[1])) < 1e-8
        end
    end
    empty!(ps.dynamic.events)
end

@testset "integrate_gpu_schur_cudss! accuracy ($(nameof(typeof(ls))))" for ls in GPU_SOLVERS
    HAS_CUDA || return
    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 0.1, 0.2))

    bl = GradPower.BatchedLayout(dp, ps, 1)
    tvec_cpu, trajs_cpu = GradPower.integrate_batched!(bl, ps, 1.0; dt=1.0/120.0)

    for M in [1, 4]
        gbl = ext.GpuBatchedLayout(dp, ps, M; linear_solver = ls)
        tvec_gpu, trajs_gpu = ext.integrate_gpu_schur_cudss!(gbl, ps, 1.0; dt=1.0/120.0)
        for m in 1:M
            err = maximum(abs.(trajs_gpu[m] .- trajs_cpu[1]))
            @test err < 1e-5
        end
    end
end

@testset "integrate_gpu_shared! accuracy ($(nameof(typeof(ls))))" for ls in GPU_SOLVERS
    HAS_CUDA || return
    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 0.1, 0.2))

    # Exact Schur GPU as reference (not CPU integrate! which uses different norm)
    gbl_ref = ext.GpuBatchedLayout(dp, ps, 1; linear_solver = ls)
    tvec_ref, trajs_ref = ext.integrate_gpu_schur_cudss!(gbl_ref, ps, 1.0; dt=1.0/120.0)

    for M in [1, 4]
        gbl = ext.GpuBatchedLayout(dp, ps, M; linear_solver = ls)
        tvec_gpu, trajs_gpu = ext.integrate_gpu_shared!(gbl, ps, 1.0; dt=1.0/120.0)
        for m in 1:M
            err = maximum(abs.(trajs_gpu[m] .- trajs_ref[1]))
            @test err < 1e-4
        end
    end
end

@testset "integrate_gpu_shared_multi! per-scenario faults ($(nameof(typeof(ls))))" for ls in GPU_SOLVERS
    HAS_CUDA || return
    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    M = 4
    fault_buses = [1, 2, 3, 4]

    # GPU single-scenario references per fault bus
    ref_trajs = []
    for bus in fault_buses
        empty!(ps.dynamic.events)
        GradPower.add_event!(ps, GradPower.ContingencyEvent(bus, 0.02, 0.1, 0.2))
        gbl_one = ext.GpuBatchedLayout(dp, ps, 1; linear_solver = ls)
        _, tr = ext.integrate_gpu_shared!(gbl_one, ps, 1.0; dt=1.0/120.0)
        push!(ref_trajs, tr[1])
    end

    # GPU: all 4 faults in one batched call
    empty!(ps.dynamic.events)
    gbl = ext.GpuBatchedLayout(dp, ps, M; linear_solver = ls)
    tvec_gpu, trajs_gpu = ext.integrate_gpu_shared_multi!(
        gbl, ps, 1.0, fault_buses, fill(0.02, M), 0.1, 0.2; dt=1.0/120.0)

    for m in 1:M
        err = maximum(abs.(trajs_gpu[m] .- ref_trajs[m]))
        @test err < 1e-6
    end
end

# Modified Newton + predictor on the exact GPU paths (item 1.6). Every
# combination of `jacobian_reuse`/`predictor` must match CPU full Newton per
# scenario; scenarios differ (scaled loads) so the global refactorization
# decision is exercised with unequal convergence. Integrating the same layout
# twice checks the persistent solver/descriptor state.
@testset "GPU Jacobian reuse and predictor ($(nameof(typeof(ls))))" for ls in GPU_SOLVERS
    HAS_CUDA || return
    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(4, 0.02, 0.1, 0.2))
    z0 = copy(dp.zvec); p0 = copy(dp.pvec)
    zl = ps.dynamic.layout.zipload
    function scaled_p(m)
        p = copy(p0)
        for k in 1:zl.n, o in (0, 1, 7, 8)   # pl, ql, yreal, yimag
            p[Int(zl.par_ptr[k]) + o] *= 1 + 0.05 * (m - 1)
        end
        p
    end
    M = 3
    refs = map(1:M) do m
        dp.zvec .= z0; dp.pvec .= scaled_p(m)
        integrate!(dp, ps, 0.5; jacobian_reuse = false, predictor = false, newton_tol = 1e-11)[2]
    end
    dp.zvec .= z0; dp.pvec .= p0

    for integ in (ext.integrate_gpu_cudss!, ext.integrate_gpu_schur_cudss!),
        jr in (false, true), pr in (false, true)
        gbl = ext.GpuBatchedLayout(dp, ps, M; linear_solver = ls)
        pm = CuMatrix(permutedims(reduce(hcat, scaled_p(m) for m in 1:M)))
        z0m = copy(gbl.z)
        for rep in 1:2                      # second call reuses the layout
            copyto!(gbl.z, z0m); copyto!(gbl.p, pm)
            _, tr = integ(gbl, ps, 0.5; jacobian_reuse = jr, predictor = pr)
            for m in 1:M
                @test maximum(abs.(tr[m] .- refs[m])) < 1e-6
            end
        end
    end

    # Fallback: a reused factorization that cannot converge in the allowed
    # iterations must be retried with full Newton (from the same start).
    dyn = ps.dynamic; L = dyn.layout
    empty!(dyn.events)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(4, 1e-3, 0.1, 0.2))
    gbl = ext.GpuBatchedLayout(dp, ps, 1; linear_solver = ls)
    gbl.z .+= 1e-3
    reuse = GradPower.JacobianReuse(true)
    @test ext._newton_step_cudss_gpu!(gbl, dyn, L, 0.0; reuse = reuse, zstart = similar(gbl.z))
    GradPower.activate!(dyn.events[1])           # stiff fault: factorization is stale
    CUDA.@allowscalar gbl.event_status[1] = true
    z1 = copy(gbl.z)
    @test !ext._newton_loop_cudss_gpu!(gbl, dyn, L, 0.0, reuse, true, 2, 1e-10)  # stale J alone fails
    copyto!(gbl.z, z1)
    @test ext._newton_step_cudss_gpu!(gbl, dyn, L, 0.0; itermax = 2, reuse = reuse,
                                      zstart = similar(gbl.z))                   # retry succeeds
    GradPower.deactivate!(dyn.events[1])
    empty!(ps.dynamic.events)
end

@testset "CUDSSBackend and SparseDirectSolverBackend agree" begin
    (HAS_CUDA && length(GPU_SOLVERS) == 2) || return
    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 0.1, 0.2))

    @test GradPower.default_gpu_backend() isa CUDSSBackend
    # batch-size-dependent default: SparseDirectSolver for large batches
    @test GradPower.default_gpu_backend(GradPower.SDS_MIN_BATCH - 1) isa CUDSSBackend
    @test GradPower.default_gpu_backend(GradPower.SDS_MIN_BATCH) isa SparseDirectSolverBackend

    for integ in (ext.integrate_gpu_cudss!, ext.integrate_gpu_schur_cudss!), M in (1, 4)
        gbl_c = ext.GpuBatchedLayout(dp, ps, M; linear_solver = CUDSSBackend())
        gbl_s = ext.GpuBatchedLayout(dp, ps, M; linear_solver = SparseDirectSolverBackend())
        @test gbl_s.linear_solver isa SparseDirectSolverBackend
        _, trajs_c = integ(gbl_c, ps, 1.0; dt=1.0/120.0)
        _, trajs_s = integ(gbl_s, ps, 1.0; dt=1.0/120.0)
        for m in 1:M
            @test maximum(abs, trajs_s[m] .- trajs_c[m]) < 1e-6
        end
    end

    # Large batch: the default layout uses SparseDirectSolver and agrees with
    # cuDSS (the factored flag drives factorization -> refactorization).
    M = GradPower.SDS_MIN_BATCH
    gbl_d = ext.GpuBatchedLayout(dp, ps, M)
    @test gbl_d.linear_solver isa SparseDirectSolverBackend
    gbl_c = ext.GpuBatchedLayout(dp, ps, M; linear_solver = CUDSSBackend())
    _, trajs_d = ext.integrate_gpu_schur_cudss!(gbl_d, ps, 0.4)
    _, trajs_c = ext.integrate_gpu_schur_cudss!(gbl_c, ps, 0.4)
    @test gbl_d.cudss_S_solver_batched.factored[]
    @test maximum(maximum(abs, trajs_d[m] .- trajs_c[m]) for m in 1:M) < 1e-6
end

@testset "GPU integrators refuse CPU-only events" begin
    ps = from_psse(joinpath(@__DIR__, "..", "examples", "ieee9_v33.raw"),
                   joinpath(@__DIR__, "..", "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps); GradPower.initialize_dynamics!(dp, ps)
    ext = Base.get_extension(GradPower, :GradPowerCUDAExt)
    gbl = ext.GpuBatchedLayout(dp, ps, 2)
    li = findfirst(d -> d.dtype isa GradPower.ZIPLoad, ps.dynamic.devices)
    add_load_step_event!(ps, LoadStepEvent(li, 0.1, 1.2))
    @test_throws ArgumentError ext.integrate_gpu_cudss!(gbl, ps, 0.2)
    @test_throws ArgumentError ext.integrate_gpu_schur_cudss!(gbl, ps, 0.2)
    @test_throws ArgumentError ext.integrate_gpu_shared_multi!(gbl, ps, 0.2, [4, 5],
                                                               [0.02, 0.02], 0.05, 0.1)
    empty!(ps.dynamic.load_step_events)
    GradPower.add_trip_event!(ps, GradPower.create_trip_line_event(ps, 4, 5, 0.05))
    @test_throws ArgumentError ext.integrate_gpu_cudss!(gbl, ps, 0.2)
    empty!(ps.dynamic.trip_events)
    add_event!(ps, ContingencyEvent(4, 0.01, 0.05, 0.1; xfault = 0.05))   # reactive fault
    @test_throws ArgumentError ext.integrate_gpu_cudss!(gbl, ps, 0.2)
    @test_throws ArgumentError ext.integrate_gpu_shared!(gbl, ps, 0.2)
    empty!(ps.dynamic.events)
end

end # @testset "gpu_backend"
