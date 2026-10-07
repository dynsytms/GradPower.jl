#!/usr/bin/env julia
# Phase 2 G2 — No behavior change.
#
# Same PSS/E parity check as Phase 1 G1: 2-bus, IEEE-9, IEEE-39 generator
# speed trajectories vs. the frozen uqgrid references. Phase 2 only adds
# a passive registry of device contracts; the integrator math must be
# bit-identical to Phase 1.
#
# Implementation: shell out to scripts/verify/phase1_g1.jl, then copy the
# resulting artifact into artifacts/phase2/g2.json with the phase/gate
# fields relabeled. Reuses the Phase 1 case set as the single source of
# truth so the two gates cannot drift apart.

using Dates

const REPO     = abspath(joinpath(@__DIR__, "..", ".."))
const P1_OUT   = joinpath(REPO, "artifacts", "phase1", "g1.json")
const OUT_PATH = joinpath(REPO, "artifacts", "phase2", "g2.json")
mkpath(dirname(OUT_PATH))

const JULIA = Base.julia_cmd()
const P1_SCRIPT = joinpath(REPO, "scripts", "verify", "phase1_g1.jl")

function main()
    t_start = time()
    rc = run(ignorestatus(`$JULIA $P1_SCRIPT`))
    ok = success(rc)

    if !isfile(P1_OUT)
        @error "Phase 1 G1 artifact not found at $P1_OUT — cannot relabel."
        return false
    end

    # Read JSON as raw text and rewrite the phase/gate header keys; the
    # rest of the body (criteria, metadata) is identical and re-attesting
    # it under the Phase 2 label is exactly the gate's purpose.
    txt = read(P1_OUT, String)
    txt = replace(txt, "\"phase\": 1" => "\"phase\": 2")
    txt = replace(txt, "\"gate\": \"G1\"" => "\"gate\": \"G2\"")
    open(OUT_PATH, "w") do io; write(io, txt); end

    println("\nPhase 2 G2: passed=$ok  artifact=$OUT_PATH  wallclock=", round(time()-t_start, digits=2), "s")
    return ok
end

exit(main() ? 0 : 1)
