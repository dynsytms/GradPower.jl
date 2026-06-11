# Phase 1 of ROADMAP.md: per-device-type SoA tables.
# The hot loop in src/dynamics.jl consumes these tables INSTEAD of iterating
# over psd.devices. This phase defines the contract; Phase 2 rewrites the
# hot loop to use it.
#
# This file is included BEFORE PowerSystemDynamics is defined (so its
# `layout::Union{Nothing,SimulationLayout}` field is well-typed). The
# builder functions below intentionally do NOT type-annotate `psd` so
# that PowerSystemDynamics does not need to exist at this point.

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
    # control coupling resolved by set_dynamics!
    has_gov::Vector{Bool};  pm_idx::Vector{Int32}
    has_exc::Vector{Bool};  efd_idx::Vector{Int32}
    # Phase 2: precomputed J.nzval positions, filled by preallocate_jacobian.
    # In Phase 1 we allocate zeros(Int32, n, 0) (empty second dim).
    jac_pos::Matrix{Int32}
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
    # control coupling
    w_idx::Vector{Int32}     # global z-index of the generator's `w` state this governor reads
    # Phase 2: precomputed J.nzval positions, filled by preallocate_jacobian.
    jac_pos::Matrix{Int32}
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
    # Phase 2: precomputed J.nzval positions, filled by preallocate_jacobian.
    jac_pos::Matrix{Int32}
end

struct ZIPLoadTable
    n::Int
    bus::Vector{Int32}
    par_ptr::Vector{Int32}
    pinj::Vector{Float64}; qinj::Vector{Float64}
    α::Vector{Float64}; β::Vector{Float64}; γ::Vector{Float64}
    weight::Vector{Float64}; v0mag::Vector{Float64}
    yreal::Vector{Float64}; yimag::Vector{Float64}
    # Phase 2: precomputed J.nzval positions, filled by preallocate_jacobian.
    jac_pos::Matrix{Int32}
end

struct SimulationLayout
    genrou::GenrouTable
    ieesgo::IEESGOTable
    esdc1a::ESDC1ATable
    zipload::ZIPLoadTable
    # Phase 2: network admittance entries' positions in J.nzval
    net_jac_pos::Vector{Int32}
end

"""
    build_layout!(psd) -> SimulationLayout

Scans psd.devices and builds one SoA table per device type. Called as the
final step of `set_dynamics!`. The old heterogeneous device loop in
src/dynamics.jl is unchanged in Phase 1; this layout is built but not used.

The `psd` parameter is a `PowerSystemDynamics`; type annotation is omitted
because this file is included before that struct is defined.
"""
function build_layout!(psd)
    genrou = build_genrou_table(psd)
    ieesgo = build_ieesgo_table(psd)
    esdc1a = build_esdc1a_table(psd)
    zipload = build_zipload_table(psd)
    net_jac_pos = Int32[]  # Phase 2 will populate
    return SimulationLayout(genrou, ieesgo, esdc1a, zipload, net_jac_pos)
end

# ------------------------------------------------------------------------
# Per-type builders — STUBS for Phase 1 contract.
# Agents A1.1–A1.4 will replace these with real implementations.
# Each stub must return an empty table (n=0) of the correct type, so the
# layout is buildable on any case during Phase 1 even if a sibling agent
# hasn't landed yet.
# ------------------------------------------------------------------------

function build_genrou_table(psd)
    # Real implementation lives in src/tables/genrou.jl (included from
    # src/GradPower.jl after dynamics.jl so the `Genrou` type is in scope).
    # Kept as a thin delegator here so layout.jl stays small.
    return _build_genrou_table_impl(psd)
end

function build_ieesgo_table(psd)
    # Real implementation lives in src/tables/ieesgo.jl (included from
    # src/GradPower.jl after dynamics.jl so the `IEESGO` type is in scope).
    # Kept as a thin delegator here so layout.jl stays small.
    return _build_ieesgo_table_impl(psd)
end

function build_esdc1a_table(psd)
    # Real implementation lives in src/tables/esdc1a.jl (included from
    # src/GradPower.jl). The builder duck-types on the dtype's typename so
    # it works whether or not src/exciters.jl (currently untracked) has
    # been wired into the module.
    return _build_esdc1a_table_impl(psd)
end

function build_zipload_table(psd)
    # Real implementation lives in src/tables/zipload.jl (included from
    # src/GradPower.jl after dynamics.jl so the `ZIPLoad` type is in scope).
    return _build_zipload_table_impl(psd)
end
