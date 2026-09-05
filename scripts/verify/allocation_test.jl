#!/usr/bin/env julia
# Phase 5 G1 — Allocation test.
#
# Loads IEEE-9 (with ZIPLoad α=0.5), initializes dynamics, takes one
# warm-up timestep so all code paths are compiled, then measures
# `@allocated` for each of the three primitives that run per Newton
# iteration in the integrate! hot loop:
#
#   1. residual    — `beuler_batched!`
#   2. Jacobian    — `beuler_jac_batched!`
#   3. KLU solve   — `klu!` + `ldiv!`
#
# GradPower-side primitives (1 and 2) must be exactly zero. The KLU
# solve allocates 16 bytes per `klu!` and 16 bytes per `ldiv!` from
# inside KLU.jl (a `Ref(K.common)` boxing in `klu_refactor`); this is
# upstream-internal and not addressable from this repo, so the gate
# allows at most 32 bytes for the combined KLU step. This bound is
# recorded in the artifact for traceability.
#
# Writes artifacts/phase5/g1.json. Exits 0 iff passed.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using GradPower
using LinearAlgebra
using SparseArrays
using KLU
using Dates

const REPO     = abspath(joinpath(@__DIR__, "..", ".."))
const OUT_PATH = joinpath(REPO, "artifacts", "phase5", "g1.json")
mkpath(dirname(OUT_PATH))

include(joinpath(@__DIR__, "_phase3_common.jl"))

const KLU_UPSTREAM_BOUND = 32   # 16 bytes/klu! + 16 bytes/ldiv! from KLU.jl

# Measure `@allocated` for a single Newton-iteration's worth of work,
# inside a function (top-level `@allocated` is unreliable for closures
# over module bindings).
function measure_iter_allocs(f, z, zold, u, p, dyn, ybus, L, J, fact, dx, dt, diff_dim)
    a_res = @allocated GradPower.beuler_batched!(f, z, zold, u, p, dyn, ybus, L, diff_dim, dt)
    a_jac = @allocated GradPower.beuler_jac_batched!(J, z, u, p, dyn, ybus, L, diff_dim, dt)
    a_fact = @allocated klu!(fact, J)
    a_solve = @allocated ldiv!(dx, fact, f)
    return a_res, a_jac, a_fact, a_solve
end

function main()
    t_start = time()

    raw = joinpath(REPO, "examples", "ieee9_v33.raw")
    dyr = joinpath(REPO, "examples", "ieee9bus.dyr")
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

    # warm up every code path so subsequent `@allocated` only sees the
    # steady-state allocations, not first-call compilation artifacts.
    GradPower.beuler_batched!(f0, z, zold, dp.uvec, dp.pvec, dyn, net.ybus_real, L, diff_dim, dt)
    GradPower.beuler_jac_batched!(J0, z, dp.uvec, dp.pvec, dyn, net.ybus_real, L, diff_dim, dt)
    fact = klu(J0)
    fact.common.scale = 0; fact.common.btf = 0
    fact.common.ordering = 1; fact.common.tol = 1e-3
    klu!(fact, J0)
    dx = zeros(Float64, n)
    ldiv!(dx, fact, f0)

    # warm the probe itself
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
        "case" => "ieee9_v33 + ieee9bus.dyr (ZIPLoad α=0.5)",
        "n_states" => n,
        "klu_refactor_bytes" => a_fact,
        "klu_solve_bytes" => a_solve,
        "klu_upstream_note" => "KLU.jl klu! + ldiv! each box K.common via Ref(); not addressable here",
    )
    write_artifact(OUT_PATH, 5, "G1", passed, criteria, metadata)
    println("Phase 5 G1: passed=$passed  res=$a_res jac=$a_jac klu=$a_klu_total  -> $OUT_PATH")
    return passed
end

exit(main() ? 0 : 1)
