# IEEEG1 SoA table builder. Mirrors tables/tgov1.jl.
#
# Time constants are floored at IEEEG1_TMIN HERE, once, so the hot
# residual/Jacobian kernels never branch on T == 0 (which matters for the
# GPU path).

function _build_ieeeg1_table_impl(psd)
    n = 0
    for device in psd.devices
        if device.dtype isa IEEEG1
            n += 1
        end
    end

    bus      = Vector{Int32}(undef, n)
    diff_ptr = Vector{Int32}(undef, n)
    alg_ptr  = Vector{Int32}(undef, n)
    ctrl_ptr = Vector{Int32}(undef, n)
    par_ptr  = Vector{Int32}(undef, n)

    K  = Vector{Float64}(undef, n)
    T1 = Vector{Float64}(undef, n)
    T2 = Vector{Float64}(undef, n)
    T3 = Vector{Float64}(undef, n)
    T4 = Vector{Float64}(undef, n)
    T5 = Vector{Float64}(undef, n)
    T6 = Vector{Float64}(undef, n)
    T7 = Vector{Float64}(undef, n)
    K1 = Vector{Float64}(undef, n)
    K3 = Vector{Float64}(undef, n)
    K5 = Vector{Float64}(undef, n)
    K7 = Vector{Float64}(undef, n)
    pref = zeros(Float64, n)

    w_idx   = Vector{Int32}(undef, n)
    jac_pos = zeros(Int32, n, IEEEG1_JAC_NENTRIES)

    uvec = psd.uvec_idx
    k = 0
    for device in psd.devices
        device.dtype isa IEEEG1 || continue
        k += 1
        gov = device.dtype

        bus[k]      = Int32(gov.bus)
        diff_ptr[k] = Int32(device.diff_ptr)
        alg_ptr[k]  = Int32(device.alg_ptr)
        ctrl_ptr[k] = Int32(device.ctrl_ptr)
        par_ptr[k]  = Int32(device.par_ptr)

        K[k]  = gov.K
        T1[k] = max(gov.T1, IEEEG1_TMIN)
        T2[k] = gov.T2
        T3[k] = max(gov.T3, IEEEG1_TMIN)
        T4[k] = max(gov.T4, IEEEG1_TMIN)
        T5[k] = max(gov.T5, IEEEG1_TMIN)
        T6[k] = max(gov.T6, IEEEG1_TMIN)
        T7[k] = max(gov.T7, IEEEG1_TMIN)
        K1[k] = gov.K1
        K3[k] = gov.K3
        K5[k] = gov.K5
        K7[k] = gov.K7

        w_idx[k] = Int32(uvec[device.ctrl_ptr])
    end

    online = fill(true, n)
    return IEEEG1Table(n, bus, diff_ptr, alg_ptr, ctrl_ptr, par_ptr,
        K, T1, T2, T3, T4, T5, T6, T7, K1, K3, K5, K7,
        pref, w_idx, jac_pos, online)
end

function refresh_ieeeg1_table!(psd)
    table = psd.layout.ieeeg1
    table.n == 0 && return nothing
    k = 0
    for device in psd.devices
        device.dtype isa IEEEG1 || continue
        k += 1
        table.pref[k] = device.dtype.pref
    end
    @assert k == table.n
    return nothing
end

register_device!(:ieeeg1;
    table_type = IEEEG1Table,
    builder    = _build_ieeeg1_table_impl,
    class      = :governor)
