#!/usr/bin/env julia
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using GradPower
using NPZ

include(joinpath(@__DIR__, "..", "test", "regression", "common.jl"))

const RAW = joinpath(REPO_ROOT, "examples", "2bus.raw")
const DYR = joinpath(REPO_ROOT, "examples", "2bus_ESDC1A.dyr")
const OUTPUT = joinpath(REPO_ROOT, "examples", "refs", "2bus_esdc1a_reduced.npz")

ps, dp = setup_regression_case(RAW, DYR; zipload_alpha=0.5)
z0 = copy(dp.zvec)
GradPower.add_event!(ps, GradPower.ContingencyEvent(ps.busmap[1], 0.02, 0.2, 0.3))
tvec, history = GradPower.integrate!(dp, ps, 5.0; dt=1/120, verbose=false)

metadata = Dict{String,Any}(
    "schema_version" => 1,
    "state_index_base" => 1,
    "case" => "2bus_esdc1a_reduced",
    "producer" => "GradPower",
    "model_revision" => "reduced-three-state",
    "git_sha" => git_sha(),
    "source_files" => Dict(
        "raw" => Dict("path" => relpath(RAW, REPO_ROOT), "sha256" => sha256_file(RAW)),
        "dyr" => Dict("path" => relpath(DYR, REPO_ROOT), "sha256" => sha256_file(DYR)),
    ),
    "state_manifest" => build_state_manifest(ps),
    "integration" => Dict("method" => "backward_euler", "dt" => 1/120,
                           "tend" => 5.0, "ton" => 0.2, "toff" => 0.3),
)
npzwrite(OUTPUT, Dict(
    "tvec" => tvec,
    "history" => history,
    "z0" => z0,
    "metadata_json" => Vector{UInt8}(codeunits(json_value(metadata))),
))
println("Saved $OUTPUT with history size $(size(history))")
