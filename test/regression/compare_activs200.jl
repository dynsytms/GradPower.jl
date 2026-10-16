#!/usr/bin/env julia
using Pkg; Pkg.activate(joinpath(@__DIR__, "..", ".."))
using GradPower
include(joinpath(@__DIR__, "common.jl"))

passed, path = run_reference_regression(
    "compare_activs200",
    joinpath(REPO_ROOT, "uqgrid", "data", "ACTIVSg200.raw"),
    joinpath(REPO_ROOT, "uqgrid", "data", "ACTIVSg200.dyr"),
    "activs200.npz";
    fault_bus_external=15, rfault=0.02, ton=0.2, toff=0.3,
    dt=1/120, tend=5.0, zipload_alpha=0.5, tolerance=5e-9,
    add_static_gen_stubs=false,
)
println("compare_activs200: passed=$passed -> $path")
exit(passed ? 0 : 1)
