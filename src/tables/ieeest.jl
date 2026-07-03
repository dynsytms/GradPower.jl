# IEEEST SoA table builder.

is_ieeest(d) = nameof(typeof(d.dtype)) === :IEEEST

function _build_ieeest_table_impl(psd)
    n = 0
    for device in psd.devices
        if is_ieeest(device)
            n += 1
        end
    end

    bus      = Vector{Int32}(undef, n)
    diff_ptr = Vector{Int32}(undef, n)
    alg_ptr  = Vector{Int32}(undef, n)
    ctrl_ptr = Vector{Int32}(undef, n)
    par_ptr  = Vector{Int32}(undef, n)

    A1  = Vector{Float64}(undef, n)
    A2  = Vector{Float64}(undef, n)
    A3  = Vector{Float64}(undef, n)
    A4  = Vector{Float64}(undef, n)
    A5  = Vector{Float64}(undef, n)
    A6  = Vector{Float64}(undef, n)
    T1  = Vector{Float64}(undef, n)
    T2  = Vector{Float64}(undef, n)
    T3  = Vector{Float64}(undef, n)
    T4  = Vector{Float64}(undef, n)
    T5  = Vector{Float64}(undef, n)
    T6  = Vector{Float64}(undef, n)
    KS  = Vector{Float64}(undef, n)

    # omega z-index — resolved by wire_controls! or fixup
    w_idx = zeros(Int32, n)

    jac_pos = zeros(Int32, n, IEEEST_JAC_NENTRIES)

    diff_dim = psd.diff_dim
    alg_dim  = psd.alg_dim

    k = 0
    for device in psd.devices
        is_ieeest(device) || continue
        k += 1
        pss = device.dtype

        bus[k]      = Int32(getfield(pss, :bus))
        diff_ptr[k] = Int32(device.diff_ptr)
        alg_ptr[k]  = Int32(device.alg_ptr)
        ctrl_ptr[k] = Int32(device.ctrl_ptr)
        par_ptr[k]  = Int32(device.par_ptr)

        A1[k]  = getfield(pss, :A1)
        A2[k]  = getfield(pss, :A2)
        A3[k]  = getfield(pss, :A3)
        A4[k]  = getfield(pss, :A4)
        A5[k]  = getfield(pss, :A5)
        A6[k]  = getfield(pss, :A6)
        T1[k]  = getfield(pss, :T1)
        T2[k]  = getfield(pss, :T2)
        T3[k]  = getfield(pss, :T3)
        T4[k]  = getfield(pss, :T4)
        T5[k]  = getfield(pss, :T5)
        T6[k]  = getfield(pss, :T6)
        KS[k]  = getfield(pss, :KS)
    end

    online = fill(true, n)
    return IEEESTTable(n, bus, diff_ptr, alg_ptr, ctrl_ptr, par_ptr,
        A1, A2, A3, A4, A5, A6, T1, T2, T3, T4, T5, T6, KS,
        w_idx, jac_pos, online)
end

# Post-layout fixup: populate w_idx on IEEEST table from uvec_idx,
# and vs_idx on exciter tables (SEXS, ESDC1A) from IEEEST alg states.
function fix_ieeest_wiring!(psd, ps)
    L = psd.layout
    ieeest_tbl = L.ieeest
    ieeest_tbl.n == 0 && return nothing

    diff_dim = psd.diff_dim

    # 1. Populate w_idx from uvec_idx routing.
    k = 0
    for device in psd.devices
        is_ieeest(device) || continue
        k += 1
        ctrl_ptr = device.ctrl_ptr
        # uvec_idx[ctrl_ptr] was set by wire_controls! to point at
        # the generator's w z-index.
        w_z = psd.uvec_idx[ctrl_ptr]
        ieeest_tbl.w_idx[k] = Int32(w_z)
    end

    # 2. Populate vs_idx on exciter tables.
    # For each IEEEST, find the exciter it attaches to (same bus, id)
    # and set that exciter's vs_idx to the IEEEST's alg state z-index.
    for (i, device) in enumerate(psd.devices)
        is_ieeest(device) || continue
        pss = device.dtype
        pss_bus = pss.bus
        pss_id  = _normalize_id(pss.id)
        # The IEEEST's v_s is its first (and only) alg state.
        vs_z = diff_dim + device.alg_ptr

        # Find matching SEXS exciter
        sexs_k = 0
        for (j, d) in enumerate(psd.devices)
            if d.dtype isa SEXS
                sexs_k += 1
                if d.dtype.bus == pss_bus && _normalize_id(d.dtype.id) == pss_id
                    L.sexs.vs_idx[sexs_k] = Int32(vs_z)
                    break
                end
            end
        end

        # Find matching ESDC1A exciter
        esdc1a_k = 0
        for (j, d) in enumerate(psd.devices)
            if is_esdc1a(d)
                esdc1a_k += 1
                if getfield(d.dtype, :bus) == pss_bus &&
                   _normalize_id(getfield(d.dtype, :id)) == pss_id
                    L.esdc1a.vs_idx[esdc1a_k] = Int32(vs_z)
                    break
                end
            end
        end
    end
    return nothing
end


# IEEEST has no init-derived parameters, but we provide a refresh stub
# for consistency with other device tables.
function refresh_ieeest_table!(psd)
    return nothing
end

register_device!(:ieeest;
    table_type = IEEESTTable,
    builder    = _build_ieeest_table_impl,
    class      = :stabilizer)
