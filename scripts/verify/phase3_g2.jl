#!/usr/bin/env julia
# Phase 3 G2 — Jacobian vs finite-difference for ESDC1A case at z0.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower
using FiniteDiff
using SparseArrays
using LinearAlgebra

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const RAW  = joinpath(REPO, "examples", "2bus.raw")
const DYR  = joinpath(REPO, "examples", "2bus_ESDC1A.dyr")
const OUT  = joinpath(REPO, "artifacts", "phase3", "g2.json")
const TOL  = 1.0e-6
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

function main()
    t0 = time()
    ps = from_psse(RAW, DYR)
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for dev in ps.dynamic.devices
        if dev.dtype isa GradPower.ZIPLoad; dev.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps); GradPower.initialize_dynamics!(dp, ps)
    z0 = copy(dp.zvec); u = dp.uvec; p = dp.pvec

    # Analytic Jacobian
    J = GradPower.preallocate_jacobian(ps)
    fill!(J.nzval, 0.0)
    GradPower.rhs_jac!(J, z0, u, p, ps)
    Jdense = Matrix(J)

    # FD Jacobian — wrap rhs_fun! and also refresh u from z via uvec routing.
    n = length(z0)
    f_workspace = zeros(n)
    function fun(z::AbstractVector)
        out = zeros(n)
        GradPower.rhs_fun!(out, z, u, p, ps)
        return out
    end
    Jfd = FiniteDiff.finite_difference_jacobian(fun, z0)

    diff = Jdense .- Jfd
    rel = maximum(abs, diff) / max(maximum(abs, Jfd), 1.0)
    passed = rel < TOL

    criteria = Any[
        Dict("name" => "max_rel_jac_err", "value" => rel,
             "threshold" => TOL, "passed" => passed),
    ]
    metadata = Dict{String,Any}("wallclock_s" => time() - t0,
                                 "n" => n, "git_sha" => git_sha(REPO),
                                 "max_abs_diff" => maximum(abs, diff))
    write_artifact(OUT, 3, "G2", passed, criteria, metadata)
    println("Phase 3 G2: passed=$passed  rel_err=$rel  -> $OUT")
    return passed
end

exit(main() ? 0 : 1)
