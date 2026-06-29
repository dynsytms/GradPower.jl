# SEXS SoA table builder. Mirror of tables/ieesgo.jl, but the cross-device
# coupling is "exciter reads vm from a bus" rather than "governor reads w
# from a generator".

function _build_sexs_table_impl(psd)
    n = 0
    for device in psd.devices
        if device.dtype isa SEXS
            n += 1
        end
    end

    bus      = Vector{Int32}(undef, n)
    diff_ptr = Vector{Int32}(undef, n)
    ctrl_ptr = Vector{Int32}(undef, n)
    par_ptr  = Vector{Int32}(undef, n)

    TA_TB = Vector{Float64}(undef, n)
    TB    = Vector{Float64}(undef, n)
    K     = Vector{Float64}(undef, n)
    TE    = Vector{Float64}(undef, n)
    EMIN  = Vector{Float64}(undef, n)
    EMAX  = Vector{Float64}(undef, n)
    vref  = zeros(Float64, n)
    vr_idx = Vector{Int32}(undef, n)
    vs_idx = zeros(Int32, n)          # PSS v_s z-index; 0 = no PSS attached

    jac_pos = zeros(Int32, n, SEXS_JAC_NENTRIES)

    diff_dim = psd.diff_dim
    alg_dim  = psd.alg_dim
    net_ptr  = diff_dim + alg_dim

    k = 0
    for device in psd.devices
        device.dtype isa SEXS || continue
        k += 1
        exc = device.dtype

        bus[k]      = Int32(exc.bus)
        diff_ptr[k] = Int32(device.diff_ptr)
        ctrl_ptr[k] = Int32(device.ctrl_ptr)
        par_ptr[k]  = Int32(device.par_ptr)

        TA_TB[k] = exc.TA_TB
        TB[k]    = exc.TB
        K[k]     = exc.K
        TE[k]    = exc.TE
        EMIN[k]  = exc.EMIN
        EMAX[k]  = exc.EMAX

        # Map the exciter's `.bus` (raw PSS/E bus number) to the internal
        # 1-based bus index, then compute vr's global z-index.
        internal_bus = haskey(psd.layout === nothing ? Dict{Int,Int}() : Dict{Int,Int}(), 0) ? 0 : Int(exc.bus)
        # psd doesn't carry busmap; we resolve via ps.busmap at refresh time.
        vr_idx[k] = Int32(net_ptr + 2*(internal_bus - 1) + 1)
    end

    online = fill(true, n)
    return SEXSTable(n, bus, diff_ptr, ctrl_ptr, par_ptr,
        TA_TB, TB, K, TE, EMIN, EMAX, vref, vr_idx, vs_idx, jac_pos, online)
end

# After set_dynamics! finishes, refresh vr_idx using ps.busmap (which maps
# raw bus number → internal 1-based index). The build pass above can't see
# ps; this is called from set_dynamics! once ps is wired.
function fix_sexs_vr_idx!(psd, ps)
    table = psd.layout.sexs
    table.n == 0 && return nothing
    net_ptr = psd.diff_dim + psd.alg_dim
    k = 0
    for device in psd.devices
        device.dtype isa SEXS || continue
        k += 1
        internal_bus = ps.busmap[Int(table.bus[k])]
        table.vr_idx[k] = Int32(net_ptr + 2*(internal_bus - 1) + 1)
    end
    return nothing
end

function refresh_sexs_table!(psd)
    table = psd.layout.sexs
    table.n == 0 && return nothing
    k = 0
    for device in psd.devices
        device.dtype isa SEXS || continue
        k += 1
        table.vref[k] = device.dtype.vref
    end
    @assert k == table.n
    return nothing
end

register_device!(:sexs;
    table_type = SEXSTable,
    builder    = _build_sexs_table_impl,
    class      = :exciter)
