#!/usr/bin/env julia
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using GradPower
using SHA

include(joinpath(@__DIR__, "..", "test", "regression", "common.jl"))

const CASES = ("200", "500", "2000")
const OUTPUT = joinpath(REPO_ROOT, "artifacts", "phase0", "dyr_coverage.json")

function main()
    cases = Dict{String,Any}()
    for case in CASES
        raw = joinpath(REPO_ROOT, "uqgrid", "data", "ACTIVSg$case.raw")
        dyr = joinpath(REPO_ROOT, "uqgrid", "data", "ACTIVSg$case.dyr")
        report = analyze_dyr_coverage(raw, dyr)
        counts = coverage_counts(report)
        by_model = Dict{String,Any}()
        for model in sort(collect(keys(report.by_source_model)))
            c = report.by_source_model[model]
            by_model[model] = Dict(
                "total" => c.total, "active" => c.active, "inactive" => c.inactive,
                "native" => c.native, "redirected" => c.redirected,
                "unsupported" => c.unsupported, "unmatched" => c.unmatched,
                "duplicate" => c.duplicate,
            )
        end
        cases["ACTIVSg$case"] = Dict(
            "raw" => relpath(raw, REPO_ROOT), "raw_sha256" => sha256_file(raw),
            "dyr" => relpath(dyr, REPO_ROOT), "dyr_sha256" => sha256_file(dyr),
            "counts" => Dict(String(k) => v for (k, v) in counts),
            "native_coverage" => native_coverage(report),
            "active_generators_without_machine" => length(report.active_generators_without_machine),
            "by_source_model" => by_model,
        )
    end
    output = Dict{String,Any}(
        "schema_version" => 1,
        "policy" => "current GradPower DEVICE_TYPE_MAP; no redirects",
        "native_models" => sort(collect(native_dyr_models())),
        "git_sha" => git_sha(),
        "cases" => cases,
    )
    mkpath(dirname(OUTPUT))
    open(OUTPUT, "w") do io
        write(io, pretty_json(json_value(output)), "\n")
    end
    println("Wrote $OUTPUT")
end

main()
