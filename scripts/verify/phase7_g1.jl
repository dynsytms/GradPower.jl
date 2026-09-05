#!/usr/bin/env julia
# Phase 7 G1 — Module loads.
#
# `using GradPower` succeeds after removing src/ad.jl and src/sensitivities.jl.
#
# Writes artifacts/phase7/g1.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase7", "g1.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function main()
    t_start = time()
    ok = false
    detail = ""
    try
        @eval using GradPower
        ok = true
        detail = "module loaded"
    catch e
        detail = sprint(showerror, e)
    end

    criteria = [Dict("name" => "module_loads", "value" => ok,
                     "threshold" => true, "passed" => ok)]
    metadata = Dict{String,Any}(
        "hardware" => string(Sys.cpu_info()[1].model),
        "git_sha"  => git_sha(REPO),
        "wallclock_s" => time() - t_start,
    )
    write_artifact(OUT, 7, "G1", ok, criteria, metadata)
    println("Phase 7 G1: passed=$ok  -> $OUT")
    return ok
end

exit(main() ? 0 : 1)
