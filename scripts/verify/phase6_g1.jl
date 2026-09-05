#!/usr/bin/env julia
# Phase 6 G1 — All cases run.
#
# Loads and integrates 2-bus, IEEE-9, IEEE-39, ACTIVSg200, ACTIVSg2000.
# Pass criterion: every case completes without crash.
#
# Writes artifacts/phase6/g1.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase6", "g1.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

struct RunCase
    name::String
    raw::String
    dyr::String
    fault_bus_ext::Int
    rfault::Float64
    ton::Float64
    toff::Float64
    dt::Float64
    tend::Float64
    set_zipload_alpha::Bool
end

const CASES = [
    RunCase("2bus_genrou",
        joinpath(REPO, "examples", "2bus.raw"),
        joinpath(REPO, "examples", "2bus.dyr"),
        1, 0.02, 0.1, 0.2, 1.0/120.0, 2.0, false),
    RunCase("ieee9_nogov",
        joinpath(REPO, "examples", "ieee9_v33.raw"),
        joinpath(REPO, "examples", "ieee9bus.dyr"),
        7, 0.02, 0.2, 0.3, 1.0/120.0, 5.0, true),
    RunCase("ieee39_gov",
        joinpath(REPO, "examples", "IEEE39.raw"),
        joinpath(REPO, "examples", "IEEE39_gov.dyr"),
        30, 0.02, 0.1, 0.2, 1.0/120.0, 2.0, true),
    RunCase("activs200",
        joinpath(REPO, "examples", "ACTIVSg200.raw"),
        joinpath(REPO, "examples", "ACTIVSg200.dyr"),
        15, 0.02, 0.2, 0.3, 1.0/120.0, 2.0, true),
    RunCase("activs2000",
        joinpath(REPO, "examples", "ACTIVSg2000.raw"),
        joinpath(REPO, "examples", "ACTIVSg2000.dyr"),
        1001, 0.02, 0.2, 0.3, 1.0/120.0, 2.0, true),
]

function run_case(c::RunCase)
    ps = GradPower.from_psse(c.raw, c.dyr)
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    if c.set_zipload_alpha
        for dev in ps.dynamic.devices
            if dev.dtype isa GradPower.ZIPLoad
                dev.dtype.α = 0.5
            end
        end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    fault_bus_int = ps.busmap[c.fault_bus_ext]
    GradPower.add_event!(ps, GradPower.ContingencyEvent(fault_bus_int, c.rfault, c.ton, c.toff))
    tvec, traj = GradPower.integrate!(dp, ps, c.tend; dt=c.dt)
    nbus = length(ps.buses)
    sys_dim = ps.dynamic.diff_dim + ps.dynamic.alg_dim + 2*nbus
    return (nsteps=length(tvec)-1, sys_dim=sys_dim)
end

function main()
    t_start = time()
    criteria = Any[]
    all_passed = true

    for c in CASES
        print("  $(c.name) ... ")
        flush(stdout)
        ok = false
        detail = ""
        try
            info = run_case(c)
            ok = true
            detail = "nsteps=$(info.nsteps), dim=$(info.sys_dim)"
        catch e
            detail = sprint(showerror, e)
        end
        push!(criteria, Dict("name" => "$(c.name)_completes",
                             "value" => ok, "threshold" => true, "passed" => ok))
        if !ok; all_passed = false; end
        println(ok ? "PASS" : "FAIL", "  ", detail)
    end

    metadata = Dict{String,Any}(
        "hardware" => string(Sys.cpu_info()[1].model),
        "git_sha"  => git_sha(REPO),
        "wallclock_s" => time() - t_start,
    )
    write_artifact(OUT, 6, "G1", all_passed, criteria, metadata)
    println("\nPhase 6 G1: passed=$all_passed  -> $OUT")
    return all_passed
end

exit(main() ? 0 : 1)
