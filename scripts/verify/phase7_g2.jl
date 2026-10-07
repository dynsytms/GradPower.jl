#!/usr/bin/env julia
# Phase 7 G2 — Integration works.
#
# BE integration on IEEE-9 unaffected (AD was not on the integration path).
#
# Writes artifacts/phase7/g2.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase7", "g2.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function main()
    t_start = time()
    ok = false
    detail = ""
    try
        raw = joinpath(REPO, "examples", "ieee9_v33.raw")
        dyr = joinpath(REPO, "examples", "ieee9bus.dyr")
        ps = GradPower.from_psse(raw, dyr)
        GradPower.build_network!(ps)
        GradPower.runpf!(ps)
        # Set ZIPLoad alpha to match reference
        for dev in ps.dynamic.devices
            if dev.dtype isa GradPower.ZIPLoad
                dev.dtype.α = 0.5
            end
        end
        dp = GradPower.DynamicProblem(ps)
        GradPower.initialize_dynamics!(dp, ps)
        fault_bus_int = ps.busmap[7]
        GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus_int, 0.02, 0.2, 0.3))
        tvec, traj = GradPower.integrate!(dp, ps, 5.0; dt=1.0/120.0)

        nsteps = length(tvec) - 1
        ok = nsteps > 0 && size(traj, 2) == length(tvec)
        detail = "nsteps=$nsteps, traj_cols=$(size(traj, 2))"
    catch e
        detail = sprint(showerror, e)
    end

    criteria = [Dict("name" => "ieee9_integration", "value" => ok,
                     "threshold" => true, "passed" => ok)]
    metadata = Dict{String,Any}(
        "hardware" => string(Sys.cpu_info()[1].model),
        "git_sha"  => git_sha(REPO),
        "wallclock_s" => time() - t_start,
    )
    write_artifact(OUT, 7, "G2", ok, criteria, metadata)
    println("Phase 7 G2: passed=$ok  -> $OUT  $detail")
    return ok
end

exit(main() ? 0 : 1)
