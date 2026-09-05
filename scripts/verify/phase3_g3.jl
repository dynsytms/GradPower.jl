#!/usr/bin/env julia
# Phase 3 G3 — Fault trajectory with ESDC1A is finite (no NaN, no divergence).

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const RAW  = joinpath(REPO, "examples", "2bus.raw")
const DYR  = joinpath(REPO, "examples", "2bus_ESDC1A.dyr")
const OUT  = joinpath(REPO, "artifacts", "phase3", "g3.json")
const MAX_ABS = 1.0e6  # generous physical bound; anything past this is "diverged"
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function main()
    t0 = time()
    ps = from_psse(RAW, DYR)
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for dev in ps.dynamic.devices
        if dev.dtype isa GradPower.ZIPLoad; dev.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps); GradPower.initialize_dynamics!(dp, ps)
    GradPower.add_event!(ps, GradPower.ContingencyEvent(1, 0.02, 0.2, 0.3))
    tvec, traj = GradPower.integrate!(dp, ps, 5.0; dt=1.0/120.0, verbose=false)

    finite_ok = all(isfinite, traj)
    bounded_ok = maximum(abs, traj) < MAX_ABS
    passed = finite_ok && bounded_ok

    criteria = Any[
        Dict("name" => "all_finite", "value" => finite_ok ? 1 : 0,
             "threshold" => 1, "passed" => finite_ok),
        Dict("name" => "bounded", "value" => maximum(abs, traj),
             "threshold" => MAX_ABS, "passed" => bounded_ok),
    ]
    metadata = Dict{String,Any}("wallclock_s" => time() - t0,
                                 "n_steps" => size(traj, 2),
                                 "git_sha" => git_sha(REPO))
    write_artifact(OUT, 3, "G3", passed, criteria, metadata)
    println("Phase 3 G3: passed=$passed  finite=$finite_ok  max_abs=$(maximum(abs, traj))  -> $OUT")
    return passed
end

exit(main() ? 0 : 1)
