#!/usr/bin/env julia
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using GradPower
include(joinpath(@__DIR__, "common.jl"))

passed, path = run_reference_regression(
    "compare_ieee9",
    joinpath(REPO_ROOT, "examples", "ieee9_v33.raw"),
    joinpath(REPO_ROOT, "examples", "ieee9bus.dyr"),
    "ieee9_nogov.npz";
    fault_bus_external=7, rfault=0.02, ton=0.2, toff=0.3,
    dt=1/120, tend=5.0, zipload_alpha=0.5, tolerance=5e-9,
)
println("compare_ieee9: passed=$passed -> $path")
exit(passed ? 0 : 1)
