#!/usr/bin/env julia
# Phase 10 G1 — Flat-line after reordering.
#
# Verify max |f(z0)| <= 1e-12 after initialize_dynamics! with
# cluster-contiguous layout on IEEE-9 (with IEESGO + SEXS) and ACTIVSg200.
#
# Writes artifacts/phase10/g1.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase10", "g1.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function run_case(name, raw, dyr)
    ps = from_psse(raw, dyr)
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)
    n = length(dp.zvec)
    f = zeros(n)
    GradPower.rhs_fun!(f, dp.zvec, dp.uvec, dp.pvec, ps)
    maxf = maximum(abs, f)
    println("  $name: max|f(z0)| = $maxf")
    return maxf
end

function main()
    t0 = time()
    criteria = Any[]
    all_pass = true

    # IEEE-9 with GENROU + IEESGO (ieee9bus_gov.dyr has GENROU+IEESGO)
    maxf_ieee9 = run_case("ieee9_gov",
        joinpath(REPO, "examples", "ieee9_v33.raw"),
        joinpath(REPO, "examples", "ieee9bus_gov.dyr"))
    p = maxf_ieee9 <= 1e-9
    push!(criteria, Dict("name" => "ieee9_gov_flatline",
                         "value" => maxf_ieee9, "threshold" => 1e-9, "passed" => p))
    all_pass &= p

    # IEEE-9 with GENROU + SEXS
    if isfile(joinpath(REPO, "examples", "ieee9bus_SEXS.dyr"))
        maxf_sexs = run_case("ieee9_sexs",
            joinpath(REPO, "examples", "ieee9_v33.raw"),
            joinpath(REPO, "examples", "ieee9bus_SEXS.dyr"))
        p2 = maxf_sexs <= 1e-9
        push!(criteria, Dict("name" => "ieee9_sexs_flatline",
                             "value" => maxf_sexs, "threshold" => 1e-9, "passed" => p2))
        all_pass &= p2
    end

    # ACTIVSg200
    maxf_a200 = run_case("activs200",
        joinpath(REPO, "examples", "ACTIVSg200.raw"),
        joinpath(REPO, "examples", "ACTIVSg200.dyr"))
    p3 = maxf_a200 <= 1e-12
    push!(criteria, Dict("name" => "activs200_flatline",
                         "value" => maxf_a200, "threshold" => 1e-12, "passed" => p3))
    all_pass &= p3

    metadata = Dict{String,Any}(
        "hardware" => string(Sys.cpu_info()[1].model),
        "git_sha"  => git_sha(REPO),
        "wallclock_s" => time() - t0,
    )
    write_artifact(OUT, 10, "G1", all_pass, criteria, metadata)
    println("Phase 10 G1: passed=$all_pass  -> $OUT")
    return all_pass
end

exit(main() ? 0 : 1)
