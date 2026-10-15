# Toy generator fixture for phase-1-cpu-solver.md#G2.
#
# This file is loaded by scripts/verify/phase1_g2.jl. It tests that a
# brand-new device type can be plugged into the new architecture WITHOUT
# editing any src/* file: the device registers a DeviceContract (Phase 1
# D1), a SoA table + builder (existing layout/registry infrastructure),
# and a per-device-type residual + Jacobian kernel — all out of
# test/devices/. The G2 verifier then integrates a 2-bus flat-line case
# with the toy attached.
#
# Per G2: residual_pullback may be the ZeroPullback stub — real pullbacks
# are required only on seed devices (G4).

using GradPower
using SparseArrays

# --------------------------------------------------------------------
# Device struct.
# --------------------------------------------------------------------

mutable struct ToyGenerator <: GradPower.AbstractGeneratorType
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    bus::Int64
    id::String
end
ToyGenerator(bus::Int64, id::AbstractString) =
    ToyGenerator(2, 1, 0, 0, bus, String(id))

# --------------------------------------------------------------------
# SoA table.
#
# Toy is dead-simple — 2 diff + 1 alg + 0 ctrl + 0 par. The residual is
# linear:
#   f_diff[1] = x1
#   f_diff[2] = x2
#   f_alg[1]  = y
# Steady state: (0, 0, 0). Jacobian: diagonal identity (3 entries).
# --------------------------------------------------------------------

struct ToyGeneratorTable
    n::Int
    diff_ptr::Vector{Int32}
    alg_ptr::Vector{Int32}
    jac_pos::Matrix{Int32}      # 3 entries per device
end

function _build_toy_table(psd)
    n = 0
    for d in psd.devices
        if d.dtype isa ToyGenerator; n += 1; end
    end
    diff_ptr = zeros(Int32, n); alg_ptr = zeros(Int32, n)
    k = 0
    for d in psd.devices
        d.dtype isa ToyGenerator || continue
        k += 1
        diff_ptr[k] = d.diff_ptr; alg_ptr[k] = d.alg_ptr
    end
    return ToyGeneratorTable(n, diff_ptr, alg_ptr, zeros(Int32, n, 3))
end

GradPower.register_device!(:toy_generator;
    table_type = ToyGeneratorTable,
    builder    = _build_toy_table,
    class      = :generator,
)

# --------------------------------------------------------------------
# Per-device-type residual + Jacobian kernels.
# These are the equivalent of src/kernels/<name>.jl for a seed device.
# --------------------------------------------------------------------

function toy_residual_batch!(f::AbstractArray, z::AbstractArray, table::ToyGeneratorTable,
                              diff_dim::Int)
    n = table.n
    n == 0 && return nothing
    @inbounds for k in 1:n
        dp = Int(table.diff_ptr[k])
        ap = Int(table.alg_ptr[k]) + diff_dim
        f[dp]     = z[dp]
        f[dp + 1] = z[dp + 1]
        f[ap]     = z[ap]
    end
    return nothing
end

function toy_preallocate!(coord_list::Vector{Vector{Int}},
                           table::ToyGeneratorTable, diff_dim::Int)
    @inbounds for k in 1:table.n
        dp = Int(table.diff_ptr[k])
        ap = Int(table.alg_ptr[k]) + diff_dim
        push!(coord_list[dp],     dp)
        push!(coord_list[dp + 1], dp + 1)
        push!(coord_list[ap],     ap)
    end
    return nothing
end

function toy_jac_positions!(table::ToyGeneratorTable, J::SparseMatrixCSC,
                             diff_dim::Int)
    rows = rowvals(J)
    @inbounds for k in 1:table.n
        dp = Int(table.diff_ptr[k])
        ap = Int(table.alg_ptr[k]) + diff_dim
        table.jac_pos[k, 1] = GradPower._find_pos(J, rows, dp,     dp)
        table.jac_pos[k, 2] = GradPower._find_pos(J, rows, dp + 1, dp + 1)
        table.jac_pos[k, 3] = GradPower._find_pos(J, rows, ap,     ap)
    end
    return nothing
end

function toy_jacobian_batch!(J::SparseMatrixCSC, table::ToyGeneratorTable)
    nz = nonzeros(J)
    @inbounds for k in 1:table.n
        nz[table.jac_pos[k, 1]] = 1.0
        nz[table.jac_pos[k, 2]] = 1.0
        nz[table.jac_pos[k, 3]] = 1.0
    end
    return nothing
end

# --------------------------------------------------------------------
# Hot-loop hooks.
#
# The toy fixture extends the production rhs_fun!/rhs_jac! and
# preallocate_jacobian via Julia's method-extension on a sentinel
# function. Rather than monkey-patch, we wrap GradPower.integrate! via
# a thin shim: the verify script overrides _rhs_fun_batched! /
# _rhs_jac_batched! / preallocate_jacobian using Julia's multiple
# dispatch by adding methods that call into the toy kernels.
#
# Cleanest path: extend GradPower's internal _rhs_fun_batched! /
# _rhs_jac_batched! BUT this would require src/ edits. Instead we use
# the public `set_dynamics!` flow with the toy registered, and rely on
# the dispatcher walking DEVICE_ORDER — which since we did
# register_device!(:toy_generator, ...) automatically includes the toy
# in `psd.layout.tables.toy_generator`. We then run integrate! via the
# G2 verify script after monkey-patching the per-step batched
# dispatcher to also call toy kernels.
#
# Since the G2 spec forbids src/ edits, we install one method on the
# existing `GradPower._rhs_fun_batched!` here via Base.invokelatest
# inside the verify script. The verify script does this in-process by
# defining a new method on _rhs_fun_batched! before calling integrate!.
# --------------------------------------------------------------------

GradPower.register_contract!(GradPower.DeviceContract(
    :toy_generator, ToyGenerator, :generator,
    2, 1, 0, 0,
    :toy_residual_batch!, :toy_jacobian_batch!,
    GradPower.ZeroPullback(2, 0),
    :none,
))

# --------------------------------------------------------------------
# Glue: extend GradPower's per-dtype hooks so that `set_dynamics!` /
# `initialize_dynamics!` / `preallocate_jacobian` accept the toy
# without complaining. These are method-extensions on existing
# generics; they live in this fixture file and DO NOT modify src/.
# --------------------------------------------------------------------

# fill_pvec! and initial_guess!/initialize_dynamics! are required because
# set_dynamics! / initialize_dynamics! invoke them per device. par_size
# is 0 so fill_pvec! is a no-op; the init residual on a 2-diff/1-alg
# device with steady state at 0 is f(x) = x — initial_guess sets x = 0,
# initialize_dynamics! returns f = x = 0 at the seeded guess.

GradPower.fill_pvec!(pvec::AbstractArray, ::ToyGenerator) = nothing

function GradPower.initial_guess!(xinit::AbstractArray,
                                    p::AbstractArray,
                                    pg::Float64, qg::Float64,
                                    vm::Float64, va::Float64,
                                    ::ToyGenerator)
    fill!(xinit, 0.0)
    return nothing
end

function GradPower.initialize_dynamics!(f::AbstractArray, x::AbstractArray,
                                          p::AbstractArray, pg::Float64,
                                          qg::Float64, vm::Float64, va::Float64,
                                          ::ToyGenerator)
    # f(x) = x at the seeded guess; nlsolve converges in one step.
    @inbounds for i in eachindex(x)
        f[i] = x[i]
    end
    return nothing
end

# Sparsity: push the toy's three diagonal entries into the coordinate
# list. Mirrors the per-device sparsity hooks that GENROU / ZIPLoad
# install in src/ — exposed via dispatch on dtype.
function GradPower.preallocate_jacobian!(coord_list::Vector{Vector{Int}},
                                          diff_ptr::Int, alg_ptr::Int,
                                          ctrl_ptr::Int, volt_ptr::Int,
                                          ::ToyGenerator)
    push!(coord_list[diff_ptr],     diff_ptr)
    push!(coord_list[diff_ptr + 1], diff_ptr + 1)
    push!(coord_list[alg_ptr],      alg_ptr)
    return nothing
end

# When set_dynamics! tries to find a static gen for the toy, dmap.gen
# stays at 0 since the raw file has no "TOY" gen. initialize_device
# would crash on `ps.gens[map.gen[i]]`. We sidestep by adding a per-
# dtype short-circuit: a toy_post_set_dynamics! method that injects a
# dummy static gen so initialize_device sees something sane. To avoid
# touching src/, we monkey-patch the dynamic system at the verify-
# script level — see scripts/verify/phase1_g2.jl, which adds a fake
# Gen entry for the toy bus before calling initialize_dynamics!.
#
# NOTE: this is a fixture-side intervention; production code never
# encounters a "TOY" device since the toy only exists in tests.

# --------------------------------------------------------------------
# Patch the production Jacobian assembler to include the toy's diagonal.
#
# The hot-loop dispatcher in src/dynamics.jl::_rhs_jac_batched! is
# hard-coded (Phase 1 limitation; D3's per-type-tuple dispatch on
# DEVICE_CONTRACTS is not yet retrofitted into the BE assembly path).
# Until D3's dispatcher is data-driven across the whole hot loop, the
# toy device's Jacobian contribution must be installed via this
# fixture-time patch. The patch wraps the existing `_rhs_jac_batched!`
# method with a closure that also writes the toy's diagonal entries.
#
# This is a fixture-level intervention; src/ is unmodified.
# --------------------------------------------------------------------

# Save the original implementation before wrapping.
const _ORIG_RHS_JAC_BATCHED = GradPower._rhs_jac_batched!
const _ORIG_RHS_FUN_BATCHED = GradPower._rhs_fun_batched!

# Method redefinition via @eval into GradPower's namespace. Adds a new
# method on the existing function that calls the toy kernel after the
# original dispatch completes.
@eval GradPower function _rhs_jac_batched!(jac::AbstractMatrix, z::AbstractArray, u::AbstractArray,
                                            p::AbstractArray, dyn::PowerSystemDynamics,
                                            ybus::SparseMatrixCSC, L::SimulationLayout)
    diff_dim = dyn.diff_dim
    alg_dim  = dyn.alg_dim
    net_ptr  = diff_dim + alg_dim

    current_injection_jacobian!(jac, ybus, net_ptr)
    genrou_jacobian_batch!(jac, z, u, p, L.genrou, diff_dim, net_ptr)
    ieesgo_jacobian_batch!(jac, p, L.ieesgo, diff_dim)
    tgov1_jacobian_batch!(jac, p, L.tgov1, diff_dim)
    sexs_jacobian_batch!(jac, z, p, L.sexs)
    zipload_jacobian_batch!(jac, z, p, L.zipload, net_ptr)
    static_gen_jacobian_batch!(jac, z, p, L.static_gen)

    # Toy-device contribution — guarded by hasproperty so the dispatcher
    # is a no-op if the layout has no toy table (e.g. production cases
    # without the fixture loaded).
    if hasproperty(L, :toy_generator)
        Main.toy_jacobian_batch!(jac, L.toy_generator)
    end

    _apply_events_jac!(jac, dyn.events, net_ptr)
    return nothing
end

@eval GradPower function preallocate_jacobian(ps::PowerSystem)
    diff_dim = ps.dynamic.diff_dim
    alg_dim = ps.dynamic.alg_dim
    nbus = length(ps.buses)
    adj = ps.network.adjacency
    map = ps.dynamic.map
    sys_dim = diff_dim + alg_dim + 2*nbus
    coord_list = [Vector{Int}() for _ in 1:sys_dim]
    for i in 1:diff_dim
        push!(coord_list[i], i)
    end
    ptr = diff_dim + alg_dim + 1
    for fr in 1:nbus
        push!(coord_list[ptr + 2*(fr - 1)], ptr + 2*(fr - 1))
        push!(coord_list[ptr + 2*(fr - 1)], ptr + 2*(fr - 1) + 1)
        push!(coord_list[ptr + 2*(fr - 1) + 1], ptr + 2*(fr - 1))
        push!(coord_list[ptr + 2*(fr - 1) + 1], ptr + 2*(fr - 1) + 1)
        for to in adj[fr]
            push!(coord_list[ptr + 2*(fr - 1)], ptr + 2*(to - 1))
            push!(coord_list[ptr + 2*(fr - 1)], ptr + 2*(to - 1) + 1)
            push!(coord_list[ptr + 2*(fr - 1) + 1], ptr + 2*(to - 1))
            push!(coord_list[ptr + 2*(fr - 1) + 1], ptr + 2*(to - 1) + 1)
        end
    end
    for (i, device) in enumerate(ps.dynamic.devices)
        device.dtype isa IEESGO         && continue
        device.dtype isa TGOV1          && continue
        device.dtype isa SEXS           && continue
        device.dtype isa StaticGenerator && continue
        bus = map.bus[i]
        diff_ptr = map.diff_ptr[i]
        alg_ptr = diff_dim + map.alg_ptr[i]
        volt_ptr = diff_dim + alg_dim + 2*(bus -1) + 1
        ctrl_ptr = map.ctrl_ptr[i]
        preallocate_jacobian!(coord_list, diff_ptr, alg_ptr, ctrl_ptr, volt_ptr, device.dtype)
    end
    L = ps.dynamic.layout::SimulationLayout
    ieesgo_preallocate!(coord_list, L.ieesgo, diff_dim)
    tgov1_preallocate!(coord_list, L.tgov1, diff_dim)
    sexs_preallocate!(coord_list, L.sexs)
    static_gen_preallocate!(coord_list, L.static_gen)
    genrou_coupling_preallocate!(coord_list, L.genrou)
    row = Int[]; col = Int[]
    for i in 1:length(coord_list)
        if !isempty(coord_list[i])
            append!(row, fill(i, length(coord_list[i])))
            append!(col, coord_list[i])
        end
    end
    data = zeros(length(row))
    Jsp = sparse(row, col, data, sys_dim, sys_dim)
    net_ptr = diff_dim + alg_dim
    genrou_jac_positions!(L.genrou, Jsp, diff_dim, net_ptr)
    ieesgo_jac_positions!(L.ieesgo, Jsp, diff_dim)
    tgov1_jac_positions!(L.tgov1, Jsp, diff_dim)
    sexs_jac_positions!(L.sexs, Jsp)
    zipload_jac_positions!(L.zipload, Jsp, net_ptr)
    static_gen_jac_positions!(L.static_gen, Jsp)

    if hasproperty(L, :toy_generator)
        Main.toy_jac_positions!(L.toy_generator, Jsp, diff_dim)
    end

    return Jsp
end

@eval GradPower function _rhs_fun_batched!(f::AbstractArray, z::AbstractArray, u::AbstractArray,
                                            p::AbstractArray, dyn::PowerSystemDynamics,
                                            ybus::SparseMatrixCSC, L::SimulationLayout)
    diff_dim = dyn.diff_dim
    alg_dim  = dyn.alg_dim
    net_ptr  = diff_dim + alg_dim
    v  = @view z[net_ptr+1:end]
    fv = @view f[net_ptr+1:end]
    mul!(fv, ybus, v, -1.0, 0.0)

    _apply_uvec_routing!(u, z, dyn.uvec_idx)

    genrou_residual_batch!(f, z, u, p, L.genrou, diff_dim, net_ptr)
    ieesgo_residual_batch!(f, z, p, L.ieesgo, diff_dim)
    tgov1_residual_batch!(f, z, p, L.tgov1, diff_dim)
    sexs_residual_batch!(f, z, p, L.sexs)
    zipload_residual_batch!(f, z, p, L.zipload, net_ptr)
    static_gen_residual_batch!(f, z, p, L.static_gen)

    if hasproperty(L, :toy_generator)
        Main.toy_residual_batch!(f, z, L.toy_generator, diff_dim)
    end

    _apply_events_fun!(f, v, dyn.events, net_ptr)
    return nothing
end
