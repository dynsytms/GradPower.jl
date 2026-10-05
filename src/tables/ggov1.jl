# GGOV1 SoA table builder. Mirrors tables/ieeeg1.jl.
#
# ⚠️ Governor (PID) path only — the load limiter, the acceleration
# limiter and the low-value select are NOT modelled. See the long scope
# note above the GGOV1 struct in src/governors.jl.
#
# Parameter layout provenance
# ---------------------------
# The PSS/E CON order was derived from the data rather than from memory:
# a per-field min/max/zero census over all 367 GGOV1 records in
# cases/ACTIVSg2000_dynamics.dyr gives 35 CONs after (bus, 'GGOV1', id),
# and the physically-recognisable values land exactly where the
# PowerWorld-exported GGOV1 CON list puts them:
#
#   field  name     range over 367 records
#     1    Rselect  1 (all)                 -- electrical-power droop
#     2    Flag     1 (all)
#     3    R        0.04 .. 0.05            <- droop, textbook 4-5 %
#     4    Tpelec   1.0 (all)
#     5/6  maxerr   +0.05 / -0.05 (all)     <- speed-error clamp
#     7    Kpgov    1.02 .. 27.6
#     8    Kigov    0.055 .. 7.71
#     9    Kdgov    0 (all)                 <- derivative unused in data
#    10    Tdgov    1.0 (all)
#    11/12 Vmax/Vmin 1.0 / 0.15 (all)       <- valve stroke limits
#    13    Tact     0.017 .. 0.5
#    14    Kturb    1.008 .. 2.48           <- ~= 1/(1-Wfnl)
#    15    Wfnl     0.10 .. 0.20            <- no-load fuel flow
#    16    Tb       0.1 .. 0.5
#    17    Tc       0 (all)
#    18    Teng     0 (all)
#    19    Tfload   3.0 (all)               -- load limiter (unused)
#    20    Kpload   1 .. 2                  -- load limiter (unused)
#    21    Kiload   0.2 .. 1.0              -- load limiter (unused)
#    22    Ldref    1.0 (all)               -- load limiter (unused)
#    23    Dm       0 (all)
#    24/25 Ropen/Rclose  +/-0.1 .. +/-1     <- rate limits (unused)
#    26    Kimw     0 (all)
#    27    Aset     0.01 (all)              -- accel limiter (unused)
#    28    Ka       10 (all)                -- accel limiter (unused)
#    29    Ta       0.1 (all)               -- accel limiter (unused)
#    30    Trate    1.56 .. 1118.4          <- MW turbine rating
#    31    db       0 (all)
#    32/33 Tsa/Tsb  4 / 5 (all)
#    34/35 Rup/Rdown +99 / -99 (all)
#
# Two independent confirmations of the alignment:
#   * Kturb ~= 1/(1 - Wfnl) holds record by record, which is the defining
#     relation of the GGOV1 turbine gain / no-load-fuel pair.
#   * field 30 equals the generator's MBASE in ACTIVSg2000.RAW for all
#     367 matched (bus, id) pairs — Trate is the turbine MW rating.
#
# Time constants are floored at GGOV1_TMIN HERE, once, so the hot
# residual/Jacobian kernels never branch on T == 0 (matters for the GPU
# path).

function _build_ggov1_table_impl(psd)
    n = 0
    for device in psd.devices
        if device.dtype isa GGOV1
            n += 1
        end
    end

    bus      = Vector{Int32}(undef, n)
    diff_ptr = Vector{Int32}(undef, n)
    alg_ptr  = Vector{Int32}(undef, n)
    ctrl_ptr = Vector{Int32}(undef, n)
    par_ptr  = Vector{Int32}(undef, n)

    R      = Vector{Float64}(undef, n)
    Tpelec = Vector{Float64}(undef, n)
    Kpgov  = Vector{Float64}(undef, n)
    Kigov  = Vector{Float64}(undef, n)
    Kdgov  = Vector{Float64}(undef, n)
    Tdgov  = Vector{Float64}(undef, n)
    Tact   = Vector{Float64}(undef, n)
    Kturb  = Vector{Float64}(undef, n)
    Wfnl   = Vector{Float64}(undef, n)
    Tb     = Vector{Float64}(undef, n)
    Tc     = Vector{Float64}(undef, n)
    Dm     = Vector{Float64}(undef, n)
    pref   = zeros(Float64, n)

    w_idx   = Vector{Int32}(undef, n)
    jac_pos = zeros(Int32, n, GGOV1_JAC_NENTRIES)

    uvec = psd.uvec_idx
    k = 0
    for device in psd.devices
        device.dtype isa GGOV1 || continue
        k += 1
        gov = device.dtype

        bus[k]      = Int32(gov.bus)
        diff_ptr[k] = Int32(device.diff_ptr)
        alg_ptr[k]  = Int32(device.alg_ptr)
        ctrl_ptr[k] = Int32(device.ctrl_ptr)
        par_ptr[k]  = Int32(device.par_ptr)

        R[k]      = gov.R
        Tpelec[k] = max(gov.Tpelec, GGOV1_TMIN)
        Kpgov[k]  = gov.Kpgov
        Kigov[k]  = gov.Kigov
        Kdgov[k]  = gov.Kdgov
        Tdgov[k]  = max(gov.Tdgov, GGOV1_TMIN)
        Tact[k]   = max(gov.Tact, GGOV1_TMIN)
        Kturb[k]  = gov.Kturb
        Wfnl[k]   = gov.Wfnl
        Tb[k]     = max(gov.Tb, GGOV1_TMIN)
        Tc[k]     = gov.Tc
        Dm[k]     = gov.Dm

        w_idx[k] = Int32(uvec[device.ctrl_ptr])
    end

    online = fill(true, n)
    return GGOV1Table(n, bus, diff_ptr, alg_ptr, ctrl_ptr, par_ptr,
        R, Tpelec, Kpgov, Kigov, Kdgov, Tdgov, Tact, Kturb, Wfnl, Tb, Tc, Dm,
        pref, w_idx, jac_pos, online)
end

function refresh_ggov1_table!(psd)
    table = psd.layout.ggov1
    table.n == 0 && return nothing
    k = 0
    for device in psd.devices
        device.dtype isa GGOV1 || continue
        k += 1
        table.pref[k] = device.dtype.pref
    end
    @assert k == table.n
    return nothing
end

register_device!(:ggov1;
    table_type = GGOV1Table,
    builder    = _build_ggov1_table_impl,
    class      = :governor)
