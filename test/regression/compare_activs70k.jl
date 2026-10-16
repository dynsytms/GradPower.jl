#!/usr/bin/env julia
# Regression: ACTIVSg70k (GENROU-only dyr) vs uqgrid, power-flow Q limits enforced.
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using GradPower
include(joinpath(@__DIR__, "common.jl"))

passed, path = run_reference_regression(
    "compare_activs70k",
    joinpath(REPO_ROOT, "uqgrid", "data", "ACTIVSg70k.raw"),
    joinpath(REPO_ROOT, "uqgrid", "benchmarks", "phase3_generated_dyrs", "ACTIVSg70k_genrou_only.dyr"),
    "activs70k.npz";
    fault_bus_external=3, rfault=0.02, ton=0.2, toff=0.3,
    dt=1/120, tend=1.0, zipload_alpha=0.5, tolerance=5e-9,
)
println("compare_activs70k: passed=$passed -> $path")
exit(passed ? 0 : 1)
