#!/usr/bin/env julia
# Phase 3 G5 — Regression suite unconditional gate.
#
# Wraps scripts/regression/run_all.jl. The Phase-3 ESDC1A kernel touches
# src/kernels, src/coupling.jl, src/exciters.jl — all in the regression
# trigger list (docs/plan/conventions.md). This gate ensures the existing
# canonical cases still pass after the new kernel lands.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

const REPO    = abspath(joinpath(@__DIR__, "..", ".."))
const SCRIPT  = joinpath(REPO, "scripts", "regression", "run_all.jl")
const SUITE_OUT = joinpath(REPO, "artifacts", "phase0", "g3.json")
const OUT     = joinpath(REPO, "artifacts", "phase3", "g5.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function main()
    t0 = time()
    rc = run(ignorestatus(`$(Base.julia_cmd()) --project=$REPO $SCRIPT`))
    suite_passed = success(rc)

    if !isfile(SUITE_OUT)
        @error "regression aggregate artifact not found at $SUITE_OUT"
        return false
    end
    # Re-emit under the phase 3 / G5 label.
    txt = read(SUITE_OUT, String)
    txt = replace(txt, "\"phase\": 0" => "\"phase\": 3"; count=1)
    txt = replace(txt, "\"gate\": \"G3\"" => "\"gate\": \"G5\""; count=1)
    open(OUT, "w") do io; write(io, txt); end

    println("Phase 3 G5: passed=$suite_passed  wallclock=", round(time()-t0, digits=2),
            "s  -> $OUT")
    return suite_passed
end

exit(main() ? 0 : 1)
