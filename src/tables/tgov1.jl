# TGOV1 SoA table builder. Mirrors tables/ieesgo.jl.

function _build_tgov1_table_impl(psd)
    n = 0
    for device in psd.devices
        if device.dtype isa TGOV1
            n += 1
        end
    end

    bus      = Vector{Int32}(undef, n)
    diff_ptr = Vector{Int32}(undef, n)
    alg_ptr  = Vector{Int32}(undef, n)
    ctrl_ptr = Vector{Int32}(undef, n)
    par_ptr  = Vector{Int32}(undef, n)

    R    = Vector{Float64}(undef, n)
    T1   = Vector{Float64}(undef, n)
    VMAX = Vector{Float64}(undef, n)
    VMIN = Vector{Float64}(undef, n)
    T2   = Vector{Float64}(undef, n)
    T3   = Vector{Float64}(undef, n)
    DT   = Vector{Float64}(undef, n)
    pref = zeros(Float64, n)

    w_idx = Vector{Int32}(undef, n)
    jac_pos = zeros(Int32, n, TGOV1_JAC_NENTRIES)

    uvec = psd.uvec_idx
    k = 0
    for device in psd.devices
        device.dtype isa TGOV1 || continue
        k += 1
        gov = device.dtype

        bus[k]      = Int32(gov.bus)
        diff_ptr[k] = Int32(device.diff_ptr)
        alg_ptr[k]  = Int32(device.alg_ptr)
        ctrl_ptr[k] = Int32(device.ctrl_ptr)
        par_ptr[k]  = Int32(device.par_ptr)

        R[k]    = gov.R
        T1[k]   = gov.T1
        VMAX[k] = gov.VMAX
        VMIN[k] = gov.VMIN
        T2[k]   = gov.T2
        T3[k]   = gov.T3
        DT[k]   = gov.DT

        w_routed = uvec[device.ctrl_ptr]
        w_idx[k] = Int32(w_routed)
    end

    online = fill(true, n)
    return TGOV1Table(n, bus, diff_ptr, alg_ptr, ctrl_ptr, par_ptr,
        R, T1, VMAX, VMIN, T2, T3, DT, pref, w_idx, jac_pos, online)
end

function refresh_tgov1_table!(psd)
    table = psd.layout.tgov1
    table.n == 0 && return nothing
    k = 0
    for device in psd.devices
        device.dtype isa TGOV1 || continue
        k += 1
        table.pref[k] = device.dtype.pref
    end
    @assert k == table.n
    return nothing
end

register_device!(:tgov1;
    table_type = TGOV1Table,
    builder    = _build_tgov1_table_impl,
    class      = :governor)
