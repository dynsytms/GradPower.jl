#!/usr/bin/env julia
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using GradPower
include(joinpath(@__DIR__, "common.jl"))

passed, path = run_reference_regression(
    "compare_sexs_2bus",
    joinpath(REPO_ROOT, "examples", "2bus.raw"),
    joinpath(REPO_ROOT, "examples", "2bus_SEXS.dyr"),
    "2bus_sexs.npz";
    fault_bus_external=1, rfault=0.02, ton=0.2, toff=0.3,
    dt=1/120, tend=5.0, zipload_alpha=0.5, tolerance=5e-9,
)
println("compare_sexs_2bus: passed=$passed -> $path")
exit(passed ? 0 : 1)
