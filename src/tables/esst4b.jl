# ESST4B SoA table builder. Mirrors tables/sexs.jl (an exciter reads its
# terminal voltage through a bus index baked into the table, not a ctrl slot)
# with the zero-time-constant flooring discipline of tables/ieeeg1.jl.
#
# TR/TA are floored at ESST4B_TMIN and KP guarded by `_esst4b_kp_eff` HERE,
# once, so the hot residual/Jacobian kernels never branch on T == 0 or
# KP == 0 (which matters for the GPU path). See the model header in
# src/exciters.jl for why zero time constants appear in real .dyr data.

function _build_esst4b_table_impl(psd)
    n = 0
    for device in psd.devices
        if device.dtype isa ESST4B
            n += 1
        end
    end

    bus      = Vector{Int32}(undef, n)
    diff_ptr = Vector{Int32}(undef, n)
    par_ptr  = Vector{Int32}(undef, n)

    TR  = Vector{Float64}(undef, n)
    KPR = Vector{Float64}(undef, n)
    KIR = Vector{Float64}(undef, n)
    TA  = Vector{Float64}(undef, n)
    KPM = Vector{Float64}(undef, n)
    KIM = Vector{Float64}(undef, n)
    KG  = Vector{Float64}(undef, n)
    KP  = Vector{Float64}(undef, n)

    VRMAX  = Vector{Float64}(undef, n)
    VRMIN  = Vector{Float64}(undef, n)
    VMMAX  = Vector{Float64}(undef, n)
    VMMIN  = Vector{Float64}(undef, n)
    KI     = Vector{Float64}(undef, n)
    VBMAX  = Vector{Float64}(undef, n)
    KC     = Vector{Float64}(undef, n)
    XL     = Vector{Float64}(undef, n)
    THETAP = Vector{Float64}(undef, n)

    vref = zeros(Float64, n)
    vb0  = zeros(Float64, n)
    # vr_idx is resolved by `fix_esst4b_vr_idx!` once ps.busmap is available.
    vr_idx = zeros(Int32, n)
    vs_idx = zeros(Int32, n)          # PSS v_s z-index; 0 = no PSS attached

    jac_pos = zeros(Int32, n, ESST4B_JAC_NENTRIES)

    k = 0
    for device in psd.devices
        device.dtype isa ESST4B || continue
        k += 1
        exc = device.dtype

        bus[k]      = Int32(exc.bus)
        diff_ptr[k] = Int32(device.diff_ptr)
        par_ptr[k]  = Int32(device.par_ptr)

        TR[k]  = max(exc.TR, ESST4B_TMIN)
        KPR[k] = exc.KPR
        KIR[k] = exc.KIR
        TA[k]  = max(exc.TA, ESST4B_TMIN)
        KPM[k] = exc.KPM
        KIM[k] = exc.KIM
        KG[k]  = exc.KG
        KP[k]  = _esst4b_kp_eff(exc.KP)

        VRMAX[k]  = exc.VRMAX
        VRMIN[k]  = exc.VRMIN
        VMMAX[k]  = exc.VMMAX
        VMMIN[k]  = exc.VMMIN
        KI[k]     = exc.KI
        VBMAX[k]  = exc.VBMAX
        KC[k]     = exc.KC
        XL[k]     = exc.XL
        THETAP[k] = exc.THETAP
    end

    online = fill(true, n)
    return ESST4BTable(n, bus, diff_ptr, par_ptr,
        TR, KPR, KIR, TA, KPM, KIM, KG, KP,
        VRMAX, VRMIN, VMMAX, VMMIN, KI, VBMAX, KC, XL, THETAP,
        vref, vb0, vr_idx, vs_idx, jac_pos, online)
end

# After set_dynamics! finishes, resolve vr_idx using ps.busmap (which maps
# raw PSS/E bus number -> internal 1-based index). The build pass above
# can't see ps; this is called from set_dynamics! once ps is wired.
function fix_esst4b_vr_idx!(psd, ps)
    table = psd.layout.esst4b
    table.n == 0 && return nothing
    net_ptr = psd.diff_dim + psd.alg_dim
    k = 0
    for device in psd.devices
        device.dtype isa ESST4B || continue
        k += 1
        internal_bus = ps.busmap[Int(table.bus[k])]
        table.vr_idx[k] = Int32(net_ptr + 2*(internal_bus - 1) + 1)
    end
    return nothing
end

# Re-snapshot the init-derived vref / vb0 from the device structs into the
# SoA table after init.
function refresh_esst4b_table!(psd)
    table = psd.layout.esst4b
    table.n == 0 && return nothing
    k = 0
    for device in psd.devices
        device.dtype isa ESST4B || continue
        k += 1
        table.vref[k] = device.dtype.vref
        table.vb0[k]  = device.dtype.vb0
    end
    @assert k == table.n
    return nothing
end

register_device!(:esst4b;
    table_type = ESST4BTable,
    builder    = _build_esst4b_table_impl,
    class      = :exciter)
