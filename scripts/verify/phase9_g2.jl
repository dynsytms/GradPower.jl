#!/usr/bin/env julia
# Phase 9 G2 -- Jacobian FD check
#
# Same 2-bus case as G1 (Genrou + SEXS + IEEEST). Checks analytic Jacobian
# against finite-difference at z0 and at a mid-trajectory point (t ~ 0.5 s
# after a fault).
#
# Pass: max |J_analytic - J_fd| / max(|J_fd|_max, 1) <= 1e-6 for each check.
# Uses the same relative error formula as phase3_g2.jl.
#
# Writes artifacts/phase9/g2.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower
using FiniteDiff
using SparseArrays

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase9", "g2.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function jac_fd_check(ps, dp, z, label)
    n = length(z)
    u = copy(dp.uvec)
    p = copy(dp.pvec)

    # Analytic Jacobian
    J = GradPower.preallocate_jacobian(ps)
    fill!(J.nzval, 0.0)
    GradPower.rhs_jac!(J, z, u, p, ps)
    J_a = Matrix(J)

    # FD Jacobian
    function fun(z_in::AbstractVector)
        f = zeros(n)
        GradPower.rhs_fun!(f, z_in, copy(u), p, ps)
        return f
    end
    J_fd = FiniteDiff.finite_difference_jacobian(fun, z)

    diff = J_a .- J_fd
    rel = maximum(abs, diff) / max(maximum(abs, J_fd), 1.0)
    println("  $label: rel_err=$rel, max_abs_diff=$(maximum(abs, diff))")
    return rel
end

function main()
    t_start = time()
    criteria = Any[]
    all_passed = true
    tol = 1e-6

    # Load case
    ps = GradPower.from_psse(
        joinpath(REPO, "examples", "2bus.raw"),
        joinpath(REPO, "examples", "2bus_IEEEST.dyr"))
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    # Check at z0
    err_z0 = jac_fd_check(ps, dp, copy(dp.zvec), "z0")
    ok_z0 = err_z0 <= tol
    push!(criteria, Dict("name" => "jac_fd_z0",
                         "value" => err_z0, "threshold" => tol, "passed" => ok_z0))
    if !ok_z0; all_passed = false; end

    # Integrate with fault to get a mid-trajectory point
    fault_bus_int = ps.busmap[1]
    GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus_int, 0.02, 0.1, 0.2))
    tvec, traj = GradPower.integrate!(dp, ps, 1.0; dt=1.0/120.0)

    # Mid-trajectory: t ~ 0.5 s -> step 60
    step = min(60, size(traj, 2))
    z_mid = traj[:, step]

    # Deactivate fault events for the FD check
    for ev in ps.dynamic.events
        GradPower.deactivate!(ev)
    end

    err_mid = jac_fd_check(ps, dp, z_mid, "mid-traj")
    ok_mid = err_mid <= tol
    push!(criteria, Dict("name" => "jac_fd_mid",
                         "value" => err_mid, "threshold" => tol, "passed" => ok_mid))
    if !ok_mid; all_passed = false; end

    metadata = Dict{String,Any}(
        "hardware" => string(Sys.cpu_info()[1].model),
        "git_sha"  => git_sha(REPO),
        "wallclock_s" => time() - t_start,
    )
    write_artifact(OUT, 9, "G2", all_passed, criteria, metadata)
    println("\nPhase 9 G2: passed=$all_passed  -> $OUT")
    return all_passed
end

exit(main() ? 0 : 1)
