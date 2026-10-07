#!/usr/bin/env julia
# Phase 3 G1 — Flat-line / initialization residual for ESDC1A.
#
# Verify: 2bus_ESDC1A (GENROU + ESDC1A) initializes with f(z0) ≈ 0.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const RAW  = joinpath(REPO, "examples", "2bus.raw")
const DYR  = joinpath(REPO, "examples", "2bus_ESDC1A.dyr")
const OUT  = joinpath(REPO, "artifacts", "phase3", "g1.json")
const TOL  = 1.0e-9
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function main()
    t0 = time()
    ps = from_psse(RAW, DYR)
    GradPower.build_network!(ps)
    GradPower.runpf!(ps)
    for dev in ps.dynamic.devices
        if dev.dtype isa GradPower.ZIPLoad; dev.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    f = zero(dp.zvec)
    GradPower.rhs_fun!(f, dp.zvec, dp.uvec, dp.pvec, ps)
    maxres = maximum(abs, f)

    passed = maxres < TOL
    criteria = Any[
        Dict("name" => "max_init_residual", "value" => maxres,
             "threshold" => TOL, "passed" => passed),
    ]
    metadata = Dict{String,Any}("wallclock_s" => time() - t0,
                                 "n_diff" => ps.dynamic.diff_dim,
                                 "n_alg" => ps.dynamic.alg_dim,
                                 "git_sha" => git_sha(REPO))
    write_artifact(OUT, 3, "G1", passed, criteria, metadata)
    println("Phase 3 G1: passed=$passed  max_residual=$maxres  -> $OUT")
    return passed
end

exit(main() ? 0 : 1)
