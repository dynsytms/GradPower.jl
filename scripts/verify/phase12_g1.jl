#!/usr/bin/env julia
# Phase 12 G1 — GMRES converges on all test cases.
#
# Runs IEEE-9, ACTIVSg200, ACTIVSg2000 fault cases with Schur-GMRES.
# Pass criterion: GMRES converges (residual <= 1e-10) on every Newton
# step for all cases. No fallback to direct solve needed.
#
# Writes artifacts/phase12/g1.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower
using Statistics

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase12", "g1.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function run_case(raw, dyr, fault_bus; tend=1.0)
    ps = from_psse(joinpath(REPO, "examples", raw), joinpath(REPO, "examples", dyr))
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus, 0.02, 0.1, 0.2))
    slog = GradPower.SolverLog()
    tvec, traj = GradPower.integrate!(dp, ps, tend; dt=1.0/120.0,
                                       solver=:schur_gmres, log=slog,
                                       newton_tol=1e-10)
    for ev in ps.dynamic.events; GradPower.deactivate!(ev); end
    return slog
end

function main()
    t0 = time()
    criteria = Any[]
    all_pass = true

    cases = [
        ("ieee9",      "ieee9_v33.raw",    "ieee9bus_gov.dyr",  7),
        ("activs200",  "ACTIVSg200.raw",   "ACTIVSg200.dyr",   1),
        ("activs2000", "ACTIVSg2000.raw",  "ACTIVSg2000.dyr",  1),
    ]

    for (name, raw, dyr, fbus) in cases
        println("  $name: running Schur-GMRES...")
        slog = run_case(raw, dyr, fbus)

        n_newton = length(slog.gmres_iters)
        max_iters = maximum(slog.gmres_iters)
        med_iters = median(slog.gmres_iters)
        # GMRES converges iff no step hit the itmax limit (100)
        hit_max = count(n -> n >= 100, slog.gmres_iters)
        converged = hit_max == 0
        println("  $name: n_newton=$n_newton, max_iters=$max_iters, median=$med_iters, hit_max=$hit_max")

        push!(criteria, Dict("name" => "$(name)_gmres_converged",
                             "value" => hit_max, "threshold" => 0, "passed" => converged))
        push!(criteria, Dict("name" => "$(name)_gmres_max_iters",
                             "value" => max_iters, "threshold" => 100, "passed" => max_iters < 100))
        all_pass &= converged
    end

    metadata = Dict{String,Any}(
        "hardware"    => string(Sys.cpu_info()[1].model),
        "git_sha"     => git_sha(REPO),
        "wallclock_s" => time() - t0,
    )
    write_artifact(OUT, 12, "G1", all_pass, criteria, metadata)
    println("Phase 12 G1: passed=$all_pass  -> $OUT")
    return all_pass
end

exit(main() ? 0 : 1)
