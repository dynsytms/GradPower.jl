#!/usr/bin/env julia
# Regression: ACTIVSg2000 (GENROU-only dyr) vs uqgrid, power-flow Q limits enforced.
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using GradPower
include(joinpath(@__DIR__, "common.jl"))

passed, path = run_reference_regression(
    "compare_activs2000",
    joinpath(REPO_ROOT, "uqgrid", "data", "ACTIVSg2000.raw"),
    joinpath(REPO_ROOT, "uqgrid", "data", "ACTIVSg2000_genrou_only.dyr"),
    "activs2000.npz";
    fault_bus_external=1001, rfault=0.02, ton=0.2, toff=0.3,
    dt=1/120, tend=2.0, zipload_alpha=0.5, tolerance=5e-9,
)
println("compare_activs2000: passed=$passed -> $path")
exit(passed ? 0 : 1)
