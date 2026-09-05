#!/usr/bin/env julia
# Phase 11 G1 — Schur matches monolithic KLU (small case).
#
# Runs IEEE-9 fault case with both monolithic and Schur Newton solvers.
# Pass criterion: max absolute trajectory difference <= 1e-12.
#
# Writes artifacts/phase11/g1.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase11", "g1.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function run_ieee9(solver::Symbol)
    ps = from_psse(
        joinpath(REPO, "examples", "ieee9_v33.raw"),
        joinpath(REPO, "examples", "ieee9bus_gov.dyr"))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(7, 0.02, 0.2, 0.3))
    # Use tight Newton tolerance (1e-12) so FP rounding differences
    # between Schur and monolithic paths stay below 1e-12 over many steps.
    tvec, traj = GradPower.integrate!(dp, ps, 5.0; dt=1.0/120.0, solver=solver, newton_tol=1e-12)
    # Deactivate events for clean reuse
    for ev in ps.dynamic.events; GradPower.deactivate!(ev); end
    return tvec, traj
end

function main()
    t0 = time()
    criteria = Any[]

    println("  Running IEEE-9 monolithic...")
    tvec_m, traj_m = run_ieee9(:monolithic)
    println("  Running IEEE-9 Schur...")
    tvec_s, traj_s = run_ieee9(:schur)

    max_diff = maximum(abs, traj_m .- traj_s)
    println("  Max abs diff: $max_diff")

    threshold = 1e-12
    passed = max_diff <= threshold

    push!(criteria, Dict("name" => "ieee9_schur_vs_monolithic",
                         "value" => max_diff, "threshold" => threshold, "passed" => passed))

    metadata = Dict{String,Any}(
        "hardware"    => string(Sys.cpu_info()[1].model),
        "git_sha"     => git_sha(REPO),
        "wallclock_s" => time() - t0,
    )
    write_artifact(OUT, 11, "G1", passed, criteria, metadata)
    println("Phase 11 G1: passed=$passed  -> $OUT")
    return passed
end

exit(main() ? 0 : 1)
