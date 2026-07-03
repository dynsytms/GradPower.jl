# SoA table for StaticGenerator (aggregated per-bus stand-in for active
# static generators without a dynamic machine model). Mirrors the pattern
# of tables/zipload.jl but per-device size differs by `bus_type`.

# Jacobian slot layout (per device); unused slots remain 0 for bus types
# that don't write them:
#   slot 1: (vr, vr)        — all bus types
#   slot 2: (vr, vi)        — all
#   slot 3: (vi, vr)        — all
#   slot 4: (vi, vi)        — all
#   slot 5: (vr, ap)        — PV, SLACK
#   slot 6: (vi, ap)        — PV, SLACK
#   slot 7: (vr, ap+1)      — SLACK only
#   slot 8: (vi, ap+1)      — SLACK only
#   slot 9: (ap, vr)        — PV, SLACK
#   slot 10: (ap, vi)       — PV only
#   slot 11: (ap+1, vi)     — SLACK only
const STATIC_GEN_JAC_NENTRIES = 11

const J_SG_VR_VR  = 1
const J_SG_VR_VI  = 2
const J_SG_VI_VR  = 3
const J_SG_VI_VI  = 4
const J_SG_VR_AP  = 5
const J_SG_VI_AP  = 6
const J_SG_VR_AP1 = 7
const J_SG_VI_AP1 = 8
const J_SG_AP_VR  = 9
const J_SG_AP_VI  = 10
const J_SG_AP1_VI = 11

struct StaticGenTable
    n::Int
    bus::Vector{Int32}        # external bus number on build; remapped to internal index by fix_static_gen_bus_idx!
    bus_type::Vector{Int8}    # 1=PQ, 2=PV, 3=SLACK
    alg_ptr::Vector{Int32}    # 0 if bus_type==1 (no alg state)
    par_ptr::Vector{Int32}
    # parameter mirror — refreshed post-init from device.dtype
    p::Vector{Float64}
    q::Vector{Float64}
    vset::Vector{Float64}
    aset::Vector{Float64}
    # net_ptr offset of vr for this device's bus (filled during build_layout!
    # using the external-bus index; remapped in fix_static_gen_bus_idx!)
    vr_idx::Vector{Int32}
    jac_pos::Matrix{Int32}
    online::Vector{Bool}
end

function _build_static_gen_table_impl(psd)
    n = 0
    for device in psd.devices
        device.dtype isa StaticGenerator && (n += 1)
    end

    bus      = Vector{Int32}(undef, n)
    bus_type = Vector{Int8}(undef, n)
    alg_ptr  = Vector{Int32}(undef, n)
    par_ptr  = Vector{Int32}(undef, n)
    p        = zeros(Float64, n)
    q        = zeros(Float64, n)
    vset     = Vector{Float64}(undef, n)
    aset     = Vector{Float64}(undef, n)
    vr_idx   = Vector{Int32}(undef, n)
    jac_pos  = zeros(Int32, n, STATIC_GEN_JAC_NENTRIES)

    diff_dim = psd.diff_dim
    alg_dim_total = psd.alg_dim
    net_ptr  = diff_dim + alg_dim_total

    k = 0
    for device in psd.devices
        device.dtype isa StaticGenerator || continue
        k += 1
        sg = device.dtype
        bus[k]      = Int32(sg.bus)
        bus_type[k] = Int8(sg.bus_type)
        alg_ptr[k]  = sg.alg_size == 0 ? Int32(0) : Int32(diff_dim + device.alg_ptr)
        par_ptr[k]  = Int32(device.par_ptr)
        vset[k]     = sg.vset
        aset[k]     = sg.aset
        # Provisional — fix_static_gen_bus_idx! corrects after ps.busmap is in scope.
        vr_idx[k]   = Int32(net_ptr + 2*(Int(sg.bus) - 1) + 1)
    end

    online = fill(true, n)
    return StaticGenTable(n, bus, bus_type, alg_ptr, par_ptr,
                          p, q, vset, aset, vr_idx, jac_pos, online)
end

# Remap bus (external PSS/E number) → internal 1-based index and rebuild
# vr_idx using ps.busmap. Mirrors fix_genrou_bus_idx! / fix_sexs_vr_idx!.
function fix_static_gen_bus_idx!(psd, ps)
    table = psd.layout.static_gen
    table.n == 0 && return nothing
    net_ptr = psd.diff_dim + psd.alg_dim
    k = 0
    for device in psd.devices
        device.dtype isa StaticGenerator || continue
        k += 1
        internal_bus = ps.busmap[Int(table.bus[k])]
        table.bus[k] = Int32(internal_bus)
        table.vr_idx[k] = Int32(net_ptr + 2*(internal_bus - 1) + 1)
    end
    return nothing
end

# Post-init refresh: copy p0, q0, vset, aset from device structs to table.
function refresh_static_gen_table!(psd)
    table = psd.layout.static_gen
    table.n == 0 && return nothing
    k = 0
    for device in psd.devices
        device.dtype isa StaticGenerator || continue
        k += 1
        sg = device.dtype
        table.p[k]    = sg.p0
        table.q[k]    = sg.q0
        table.vset[k] = sg.vset
        table.aset[k] = sg.aset
    end
    return nothing
end

register_device!(:static_gen;
    table_type = StaticGenTable,
    builder    = _build_static_gen_table_impl,
    class      = :generator)
