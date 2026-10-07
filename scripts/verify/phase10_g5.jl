#!/usr/bin/env julia
# Phase 10 G5 — No allocation regression.
#
# Same as Phase 5 G1: measures @allocated for residual, Jacobian, and
# KLU solve on a single Newton iteration. GradPower-side primitives
# must allocate zero bytes.
#
# Writes artifacts/phase10/g5.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower
using LinearAlgebra
using SparseArrays
using KLU

const REPO = abspath(joinpath(@__DIR__, "..", ".."))
const OUT  = joinpath(REPO, "artifacts", "phase10", "g5.json")
mkpath(dirname(OUT))

include(joinpath(@__DIR__, "_phase3_common.jl"))

const KLU_UPSTREAM_BOUND = 32

function measure_iter_allocs(f, z, zold, u, p, dyn, ybus, L, J, fact, dx, dt, diff_dim)
    a_res = @allocated GradPower.beuler_batched!(f, z, zold, u, p, dyn, ybus, L, diff_dim, dt)
    a_jac = @allocated GradPower.beuler_jac_batched!(J, z, u, p, dyn, ybus, L, diff_dim, dt)
    a_fact = @allocated klu!(fact, J)
    a_solve = @allocated ldiv!(dx, fact, f)
    return a_res, a_jac, a_fact, a_solve
end

function main()
    t_start = time()

    raw = joinpath(REPO, "examples", "ACTIVSg200.raw")
    dyr = joinpath(REPO, "examples", "ACTIVSg200.dyr")
    ps = from_psse(raw, dyr)
    GradPower.build_network!(ps); GradPower.runpf!(ps)
    for d in ps.dynamic.devices
        if d.dtype isa GradPower.ZIPLoad; d.dtype.α = 0.5; end
    end
    dp = GradPower.DynamicProblem(ps)
    GradPower.initialize_dynamics!(dp, ps)

    dt = 1.0/120.0
    n = length(dp.zvec)
    f0 = zeros(Float64, n)
    J0 = GradPower.preallocate_jacobian(ps)
    zold = copy(dp.zvec)
    z = copy(dp.zvec)

    dyn = ps.dynamic::GradPower.PowerSystemDynamics
    net = ps.network::GradPower.Network
    L = dyn.layout::GradPower.SimulationLayout
    diff_dim = dyn.diff_dim

    # Warm up
    GradPower.beuler_batched!(f0, z, zold, dp.uvec, dp.pvec, dyn, net.ybus_real, L, diff_dim, dt)
    GradPower.beuler_jac_batched!(J0, z, dp.uvec, dp.pvec, dyn, net.ybus_real, L, diff_dim, dt)
    fact = klu(J0)
    fact.common.scale = 0; fact.common.btf = 0
    fact.common.ordering = 1; fact.common.tol = 1e-3
    klu!(fact, J0)
    dx = zeros(Float64, n)
    ldiv!(dx, fact, f0)

    # Warm the probe
    measure_iter_allocs(f0, z, zold, dp.uvec, dp.pvec, dyn, net.ybus_real, L,
                        J0, fact, dx, dt, diff_dim)
    a_res, a_jac, a_fact, a_solve = measure_iter_allocs(
        f0, z, zold, dp.uvec, dp.pvec, dyn, net.ybus_real, L,
        J0, fact, dx, dt, diff_dim)

    a_klu_total = a_fact + a_solve

    pass_res = a_res == 0
    pass_jac = a_jac == 0
    pass_klu = a_klu_total <= KLU_UPSTREAM_BOUND
    passed = pass_res && pass_jac && pass_klu

    criteria = Any[
        Dict("name" => "residual_alloc_bytes",
             "value" => a_res, "threshold" => 0, "passed" => pass_res),
        Dict("name" => "jacobian_alloc_bytes",
             "value" => a_jac, "threshold" => 0, "passed" => pass_jac),
        Dict("name" => "klu_alloc_bytes",
             "value" => a_klu_total, "threshold" => KLU_UPSTREAM_BOUND,
             "passed" => pass_klu),
    ]

    wall = time() - t_start
    metadata = Dict{String,Any}(
        "hardware" => string(Sys.cpu_info()[1].model),
        "git_sha" => git_sha(REPO),
        "wallclock_s" => wall,
        "case" => "ACTIVSg200",
        "n_states" => n,
    )
    write_artifact(OUT, 10, "G5", passed, criteria, metadata)
    println("Phase 10 G5: passed=$passed  res=$a_res jac=$a_jac klu=$a_klu_total  -> $OUT")
    return passed
end

exit(main() ? 0 : 1)
