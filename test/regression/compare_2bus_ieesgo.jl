#!/usr/bin/env julia
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using GradPower
include(joinpath(@__DIR__, "common.jl"))

passed, path = run_reference_regression(
    "compare_2bus_ieesgo",
    joinpath(REPO_ROOT, "examples", "2bus.raw"),
    joinpath(REPO_ROOT, "examples", "2bus_IEESGO.dyr"),
    "2bus_ieesgo.npz";
    fault_bus_external=1, rfault=0.02, ton=0.1, toff=0.2,
    dt=1/120, tend=2.0, zipload_alpha=1.0, tolerance=5e-9,
)
println("compare_2bus_ieesgo: passed=$passed -> $path")
exit(passed ? 0 : 1)
