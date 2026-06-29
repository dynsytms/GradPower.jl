# Per-device-type SoA tables. The hot loop in src/dynamics.jl consumes
# these tables INSTEAD of iterating over psd.devices.
#
# SimulationLayout holds a NamedTuple of concrete table types so adding
# a new device class is one registry entry + one tables/*.jl file, with
# no edits to layout.jl or rhs_fun!. The hot-loop dispatcher walks the
# NamedTuple via @generated for full specialization.
#
# This file is included BEFORE PowerSystemDynamics is defined (so its
# `layout::Union{Nothing,SimulationLayout}` field is well-typed). The
# builder functions below intentionally do NOT type-annotate `psd` so
# that PowerSystemDynamics does not need to exist at this point.

# ------------------------------------------------------------------------
# Device registry
#
# Build-time metadata only. The hot loop NEVER reads from this Dict —
# it reads the NamedTuple stored in SimulationLayout.tables, which is
# concrete and fully specialized. The Dict exists so set_dynamics! can
# enumerate device types in a stable order and call each builder.
#
# Per-device-type files (src/tables/*.jl, and later src/coupling.jl)
# call `register_device!` at module load time once their table struct
# and builder function are in scope.
# ------------------------------------------------------------------------

const DEVICE_REGISTRY = Dict{Symbol, NamedTuple}()
const DEVICE_ORDER    = Symbol[]

"""
    register_device!(name::Symbol; table_type, builder, class)

Register a device type with the layout registry. `name` is the symbol
used as the NamedTuple key in `SimulationLayout.tables`. `builder` is
a function `(psd) -> Table` that scans `psd.devices` and emits the SoA
table for this device type. `class` is one of `:generator`, `:governor`,
`:exciter`, `:stabilizer`, `:load` — used by the coupling graph to
decide which devices attach to which.

Subsequent registrations under the same `name` overwrite the previous
entry but preserve the insertion order in `DEVICE_ORDER` — useful when
src/tables/*.jl files are re-included during development.
"""
function register_device!(name::Symbol; table_type, builder, class::Symbol)
    if !haskey(DEVICE_REGISTRY, name)
        push!(DEVICE_ORDER, name)
    end
    DEVICE_REGISTRY[name] = (table_type=table_type, builder=builder, class=class)
    return nothing
end

# ------------------------------------------------------------------------
# Per-device-type SoA table structs.
#
# Each is a plain immutable struct of `Vector{T}` fields. Adding a new
# device type means adding a new struct here + a builder in src/tables/
# + one registration. The hot loop sees the concrete type through the
# NamedTuple, so specialization is preserved per kernel.
# ------------------------------------------------------------------------

struct GenrouTable
    n::Int
    # global pointers
    bus::Vector{Int32}
    diff_ptr::Vector{Int32}
    alg_ptr::Vector{Int32}
    ctrl_ptr::Vector{Int32}
    par_ptr::Vector{Int32}
    # SoA parameters
    x_d::Vector{Float64};  x_q::Vector{Float64};   x_dp::Vector{Float64}
    x_qp::Vector{Float64}; x_ddp::Vector{Float64}; xl::Vector{Float64}
    H::Vector{Float64};    D::Vector{Float64}
    T_d0p::Vector{Float64};  T_q0p::Vector{Float64}
    T_d0dp::Vector{Float64}; T_q0dp::Vector{Float64}
    S1::Vector{Float64};     S2::Vector{Float64}
    # control coupling resolved by set_dynamics!
    has_gov::Vector{Bool};  pm_idx::Vector{Int32}
    has_exc::Vector{Bool};  efd_idx::Vector{Int32}
    # Precomputed J.nzval positions, filled by preallocate_jacobian.
    jac_pos::Matrix{Int32}
    # Per-device online flag (mutable vector so integrate! can flip it)
    online::Vector{Bool}
end

struct IEESGOTable
    n::Int
    bus::Vector{Int32}
    diff_ptr::Vector{Int32}
    alg_ptr::Vector{Int32}
    ctrl_ptr::Vector{Int32}
    par_ptr::Vector{Int32}
    T1::Vector{Float64}; T2::Vector{Float64}; T3::Vector{Float64}
    T4::Vector{Float64}; T5::Vector{Float64}; T6::Vector{Float64}
    K1::Vector{Float64}; K2::Vector{Float64}; K3::Vector{Float64}
    pmax::Vector{Float64}; pmin::Vector{Float64}
    pref::Vector{Float64}
    w_idx::Vector{Int32}
    jac_pos::Matrix{Int32}
    online::Vector{Bool}
end

struct TGOV1Table
    n::Int
    bus::Vector{Int32}
    diff_ptr::Vector{Int32}
    alg_ptr::Vector{Int32}
    ctrl_ptr::Vector{Int32}
    par_ptr::Vector{Int32}
    R::Vector{Float64};    T1::Vector{Float64}
    VMAX::Vector{Float64}; VMIN::Vector{Float64}
    T2::Vector{Float64};   T3::Vector{Float64}
    DT::Vector{Float64}
    pref::Vector{Float64}
    w_idx::Vector{Int32}
    jac_pos::Matrix{Int32}
    online::Vector{Bool}
end

struct SEXSTable
    n::Int
    bus::Vector{Int32}
    diff_ptr::Vector{Int32}
    ctrl_ptr::Vector{Int32}
    par_ptr::Vector{Int32}
    TA_TB::Vector{Float64}; TB::Vector{Float64}
    K::Vector{Float64};      TE::Vector{Float64}
    EMIN::Vector{Float64};   EMAX::Vector{Float64}
    vref::Vector{Float64}
    vr_idx::Vector{Int32}
    vs_idx::Vector{Int32}    # PSS v_s z-index; 0 = no PSS attached
    jac_pos::Matrix{Int32}
    online::Vector{Bool}
end

struct ESDC1ATable
    n::Int
    bus::Vector{Int32}
    diff_ptr::Vector{Int32}
    par_ptr::Vector{Int32}
    Ka::Vector{Float64}; Ta::Vector{Float64}; Kf::Vector{Float64}
    Tf::Vector{Float64}; Ke::Vector{Float64}; Te::Vector{Float64}
    Tr::Vector{Float64}; Ae::Vector{Float64}; Be::Vector{Float64}
    vref::Vector{Float64}
    vr_idx::Vector{Int32}
    vs_idx::Vector{Int32}    # PSS v_s z-index; 0 = no PSS attached
    jac_pos::Matrix{Int32}
    online::Vector{Bool}
end

struct IEEESTTable
    n::Int
    bus::Vector{Int32}
    diff_ptr::Vector{Int32}
    alg_ptr::Vector{Int32}
    ctrl_ptr::Vector{Int32}
    par_ptr::Vector{Int32}
    A1::Vector{Float64}; A2::Vector{Float64}
    A3::Vector{Float64}; A4::Vector{Float64}
    A5::Vector{Float64}; A6::Vector{Float64}
    T1::Vector{Float64}; T2::Vector{Float64}
    T3::Vector{Float64}; T4::Vector{Float64}
    T5::Vector{Float64}; T6::Vector{Float64}
    KS::Vector{Float64}
    w_idx::Vector{Int32}
    jac_pos::Matrix{Int32}
    online::Vector{Bool}
end

struct ZIPLoadTable
    n::Int
    bus::Vector{Int32}
    par_ptr::Vector{Int32}
    pinj::Vector{Float64}; qinj::Vector{Float64}
    α::Vector{Float64}; β::Vector{Float64}; γ::Vector{Float64}
    weight::Vector{Float64}; v0mag::Vector{Float64}
    yreal::Vector{Float64}; yimag::Vector{Float64}
    jac_pos::Matrix{Int32}
    online::Vector{Bool}
end

# ------------------------------------------------------------------------
# SimulationLayout — NamedTuple of concrete tables.
#
# Performance guardrails:
#   - `tables` must be a NamedTuple of CONCRETE table types so Julia
#     specializes per-field; no boxing, no dynamic dispatch in hot loop.
#   - Never store as Vector{AbstractTable} or Dict{Symbol,Any}.
#   - The hot-loop dispatcher uses @generated to unroll the iteration
#     so the compiler sees every kernel call site concretely.
#
# Back-compat: existing code reads `layout.genrou`, `layout.ieesgo`, etc.
# A `getproperty` overload forwards those names to `getfield(layout.tables, name)`.
# ------------------------------------------------------------------------

struct SimulationLayout{T<:NamedTuple}
    tables::T
    # Network admittance entries' positions in J.nzval
    net_jac_pos::Vector{Int32}
end

# Forward unknown property accesses to `tables`. `:tables` and `:net_jac_pos`
# resolve to the underlying fields; anything else (e.g. `:genrou`) goes
# through `getproperty(tables, name)` which is itself fully specialized
# because `tables` is a concrete NamedTuple.
@inline function Base.getproperty(L::SimulationLayout, name::Symbol)
    if name === :tables
        return getfield(L, :tables)
    elseif name === :net_jac_pos
        return getfield(L, :net_jac_pos)
    else
        return getproperty(getfield(L, :tables), name)
    end
end

Base.propertynames(L::SimulationLayout) =
    (:tables, :net_jac_pos, propertynames(getfield(L, :tables))...)

# ------------------------------------------------------------------------
# build_layout! — driver
#
# Iterates DEVICE_ORDER (stable across runs), calls each registered
# builder against `psd`, and packs results into a NamedTuple. This runs
# once per `set_dynamics!` call; the hot loop reads the NamedTuple
# millions of times via the @generated dispatcher.
# ------------------------------------------------------------------------

"""
    build_layout!(psd) -> SimulationLayout

Scans psd.devices and builds one SoA table per registered device type.
Called as the final step of `set_dynamics!`.

The `psd` parameter is a `PowerSystemDynamics`; type annotation is
omitted because this file is included before that struct is defined.
"""
function build_layout!(psd)
    # Build a Vector of (Symbol, Table) pairs in registration order, then
    # construct the NamedTuple. Doing it this way (vs. comprehension into
    # NamedTuple) avoids relying on Dict iteration order — DEVICE_ORDER
    # is the authoritative ordering.
    pairs = Pair{Symbol,Any}[]
    for name in DEVICE_ORDER
        entry = DEVICE_REGISTRY[name]
        push!(pairs, name => entry.builder(psd))
    end
    tables = NamedTuple(pairs)
    net_jac_pos = Int32[]  # populated by preallocate_jacobian
    return SimulationLayout(tables, net_jac_pos)
end

# ------------------------------------------------------------------------
# Stub dispatcher — used only by the test/test_layout.jl
# type-stability + zero-allocation gate.
#
# `_dispatch_count_n(tables)` returns the total device count across all
# tables, computed by visiting each table once. The visit is unrolled by
# @generated so the compiler sees a concrete call to `_n_of` for each
# table type — no Any, no Union, no allocation.
# ------------------------------------------------------------------------

@inline _n_of(t) = t.n  # every Table struct has an `n::Int` field

@generated function _dispatch_count_n(tables::NamedTuple{names}) where {names}
    body = Expr(:block)
    sym = gensym(:acc)
    push!(body.args, :($sym = 0))
    for n in names
        push!(body.args, :($sym += _n_of(tables.$n)))
    end
    push!(body.args, :(return $sym))
    return body
end
