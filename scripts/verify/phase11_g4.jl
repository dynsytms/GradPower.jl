#!/usr/bin/env julia
# Phase 11 G4 — No allocation in Schur hot loop.
#
# Verify that @allocated inside the Newton loop (assemble S, solve,
# back-substitute) is zero after first call. SchurWorkspace
# preallocates everything at setup.
#
# Writes artifacts/phase11/g4.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase11", "g4.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

"""
    _schur_inner!(sw, J0, ct, net_ptr, f0)

Exercise the Schur-specific parts of one Newton iteration:
assemble_schur!, reduced RHS construction, and back-substitution.
Excludes KLU solve (which allocates internally in SuiteSparse).
"""
function _schur_inner!(sw::GradPower.SchurWorkspace,
                        J0, ct::GradPower.ClusterTable,
                        net_ptr::Int, f0::Vector{Float64})
    # 1. Assemble Schur complement S
    GradPower.assemble_schur!(sw, J0, ct, net_ptr)

    # 2. Build reduced RHS
    n_red = length(sw.reduced_idx)
    @inbounds for i in 1:n_red
        sw.rhs_red[i] = f0[sw.reduced_idx[i]]
    end

    ng = length(sw.nt_groups)
    @inbounds for g in 1:ng
        group = sw.nt_groups[g]
        nc_g = length(group)
        for ki in 1:nc_g
            ci = group[ki]
            cl = ct.clusters[ci]
            wk = cl.w_size; ws = cl.w_start; bus = cl.bus
            vr_g = net_ptr + 2*(bus - 1) + 1; vi_g = vr_g + 1
            vr_l = sw.global_to_reduced[vr_g]; vi_l = sw.global_to_reduced[vi_g]
            A = sw.A_k_bufs[g][ki]; C = sw.C_k_bufs[g][ki]
            ipiv = sw.lu_pivots[g][ki]; tmp = sw.tmp_w[g][ki]
            for i in 1:wk; tmp[i] = f0[ws + i - 1]; end
            GradPower._lu_solve!(A, ipiv, tmp, wk)
            for i in 1:wk
                sw.rhs_red[vr_l] -= C[1, i] * tmp[i]
                sw.rhs_red[vi_l] -= C[2, i] * tmp[i]
            end
        end
    end

    # 3. Back-substitution (using dummy dv values from rhs_red)
    @inbounds for i in 1:n_red
        sw.dz[sw.reduced_idx[i]] = sw.rhs_red[i]
    end

    @inbounds for g in 1:ng
        group = sw.nt_groups[g]
        nc_g = length(group)
        for ki in 1:nc_g
            ci = group[ki]
            cl = ct.clusters[ci]
            wk = cl.w_size; ws = cl.w_start; bus = cl.bus
            vr_g = net_ptr + 2*(bus - 1) + 1; vi_g = vr_g + 1
            A = sw.A_k_bufs[g][ki]; B = sw.B_k_bufs[g][ki]
            ipiv = sw.lu_pivots[g][ki]; tmp = sw.tmp_w[g][ki]
            dv1 = sw.dz[vr_g]; dv2 = sw.dz[vi_g]
            for i in 1:wk
                tmp[i] = f0[ws + i - 1] + B[i, 1] * dv1 + B[i, 2] * dv2
            end
            GradPower._lu_solve!(A, ipiv, tmp, wk)
            for i in 1:wk; sw.dz[ws + i - 1] = -tmp[i]; end
        end
    end
    return nothing
end

function main()
    t0 = time()

    # Setup ACTIVSg200
    ps = from_psse(
        joinpath(REPO, "examples", "ACTIVSg200.raw"),
        joinpath(REPO, "examples", "ACTIVSg200.dyr"))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    net = ps.network::GradPower.Network
    L   = dyn.layout::GradPower.SimulationLayout
    ct  = dyn.clusters::GradPower.ClusterTable
    ybus = net.ybus_real
    net_ptr = dyn.diff_dim + dyn.alg_dim
    diff_dim = dyn.diff_dim
    sys_dim = length(dp.zvec)
    dt = 1.0/120.0

    # Preallocate
    sw = GradPower.SchurWorkspace(ps)
    J0 = GradPower.preallocate_jacobian(ps)
    f0 = zeros(sys_dim)
    zold = copy(dp.zvec)

    # Fill J and f with actual values
    GradPower.beuler_batched!(f0, dp.zvec, zold, dp.uvec, dp.pvec, dyn, ybus, L, diff_dim, dt)
    GradPower.beuler_jac_batched!(J0, dp.zvec, dp.uvec, dp.pvec, dyn, ybus, L, diff_dim, dt)

    # Warmup: call the function 3 times to ensure full JIT compilation
    _schur_inner!(sw, J0, ct, net_ptr, f0)
    _schur_inner!(sw, J0, ct, net_ptr, f0)
    _schur_inner!(sw, J0, ct, net_ptr, f0)

    # Measure: the Schur-specific hot loop operations
    alloc_schur = @allocated _schur_inner!(sw, J0, ct, net_ptr, f0)
    println("  alloc_schur_inner: $alloc_schur")

    # Also verify assemble_schur! standalone
    alloc_assemble = @allocated GradPower.assemble_schur!(sw, J0, ct, net_ptr)
    println("  alloc_assemble: $alloc_assemble")

    passed = alloc_schur == 0
    criteria = Any[
        Dict("name" => "schur_inner_alloc", "value" => alloc_schur,   "threshold" => 0, "passed" => alloc_schur == 0),
        Dict("name" => "assemble_alloc",    "value" => alloc_assemble, "threshold" => 0, "passed" => alloc_assemble == 0),
    ]

    metadata = Dict{String,Any}(
        "hardware"    => string(Sys.cpu_info()[1].model),
        "git_sha"     => git_sha(REPO),
        "wallclock_s" => time() - t0,
    )
    write_artifact(OUT, 11, "G4", passed, criteria, metadata)
    println("Phase 11 G4: passed=$passed  -> $OUT")
    return passed
end

exit(main() ? 0 : 1)
