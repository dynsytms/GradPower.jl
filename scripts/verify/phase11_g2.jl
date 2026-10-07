#!/usr/bin/env julia
# Phase 11 G2 — Schur matches monolithic KLU (large cases).
#
# Runs ACTIVSg200 and ACTIVSg2000 fault cases with both solvers.
# Pass criterion: max absolute trajectory difference <= 1e-12 on both.
# No SingularException on any case.
#
# Writes artifacts/phase11/g2.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase11", "g2.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function run_case(raw, dyr, fault_bus, solver::Symbol; tend=1.0)
    ps = from_psse(joinpath(REPO, "examples", raw), joinpath(REPO, "examples", dyr))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus, 0.02, 0.1, 0.2))
    # Use tight Newton tolerance (1e-12) so the per-step FP rounding
    # differences between Schur and monolithic paths stay below the
    # 1e-12 trajectory threshold over many timesteps.
    tvec, traj = GradPower.integrate!(dp, ps, tend; dt=1.0/120.0, solver=solver, newton_tol=1e-12)
    for ev in ps.dynamic.events; GradPower.deactivate!(ev); end
    return tvec, traj
end

function main()
    t0 = time()
    criteria = Any[]
    all_pass = true
    threshold = 1e-12

    for (name, raw, dyr, fbus) in [
        ("activs200", "ACTIVSg200.raw", "ACTIVSg200.dyr", 1),
        ("activs2000", "ACTIVSg2000.raw", "ACTIVSg2000.dyr", 1),
    ]
        println("  $name monolithic...")
        tvec_m, traj_m = run_case(raw, dyr, fbus, :monolithic)
        println("  $name Schur...")
        tvec_s, traj_s = run_case(raw, dyr, fbus, :schur)

        max_diff = maximum(abs, traj_m .- traj_s)
        passed = max_diff <= threshold
        println("  $name max_diff=$max_diff passed=$passed")

        push!(criteria, Dict("name" => "$(name)_schur_vs_monolithic",
                             "value" => max_diff, "threshold" => threshold, "passed" => passed))
        all_pass &= passed
    end

    metadata = Dict{String,Any}(
        "hardware"    => string(Sys.cpu_info()[1].model),
        "git_sha"     => git_sha(REPO),
        "wallclock_s" => time() - t0,
    )
    write_artifact(OUT, 11, "G2", all_pass, criteria, metadata)
    println("Phase 11 G2: passed=$all_pass  -> $OUT")
    return all_pass
end

exit(main() ? 0 : 1)
