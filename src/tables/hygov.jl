# HYGOV SoA table builder. Mirrors tables/ggov1.jl.
#
# Parameter layout provenance
# ---------------------------
# Derived from the data, not from memory: a per-field min/max/zero census
# over all 25 HYGOV records in cases/ACTIVSg2000_dynamics.dyr (every record
# has 15 tokens => 12 CONs after bus, 'HYGOV', id):
#
#   field  name   range over 25 records     reading
#     1    R      0.04 .. 0.07  (3 values)  permanent droop, textbook 4-7 %
#     2    r      0.105 .. 0.961 (25 vals)  temporary droop, per-unit tuned
#     3    Tr     1.61 .. 8.83   (25 vals)  governor reset time, per-unit tuned
#     4    Tf     0.05 .. 0.4               filter time constant (typ. 0.05)
#     5    Tg     0.33 .. 0.5               gate servo time constant (typ. 0.5)
#     6    VELM   0.07 .. 0.2               gate velocity limit (typ. 0.2)
#     7    GMAX   1 (all)                   max gate
#     8    GMIN   0 (all)                   min gate
#     9    TW     0.30 .. 1.95   (25 vals)  water time constant (typ. 1-2 s)
#    10    At     1.2 .. 1.25               turbine gain (typ. 1.2)
#    11    Dturb  0 .. 1 (11 zeros)         turbine damping (typ. 0-0.5)
#    12    qNL    0.07 .. 0.0799            no-load flow (typ. 0.08)
#
# This matches the PSS/E HYGOV CON list (R r Tr Tf Tg VELM GMAX GMIN TW At
# Dturb qNL). Independent confirmation: fields 2/3/9 are the only ones with
# 25 distinct values, and field 3 obeys the classic hydro temporary-droop
# tuning  Tr = [5 - 0.5 (TW - 1)] TW  against field 9 record by record
# (e.g. bus 3105: TW = 1.2091 -> 5.9191 = field 3), which pins Tr and TW;
# r = [2.3 - 0.15 (TW - 1)] TW / M then fixes field 2. GMAX/GMIN are
# exactly 1/0.
#
# Divisors are floored at HYGOV_TMIN HERE and in fill_pvec!, so the hot
# residual/Jacobian kernels never branch on a zero (matters for the GPU).

function _build_hygov_table_impl(psd)
    n = 0
    for device in psd.devices
        device.dtype isa HYGOV && (n += 1)
    end

    bus      = Vector{Int32}(undef, n)
    diff_ptr = Vector{Int32}(undef, n)
    alg_ptr  = Vector{Int32}(undef, n)
    ctrl_ptr = Vector{Int32}(undef, n)
    par_ptr  = Vector{Int32}(undef, n)

    R     = Vector{Float64}(undef, n)
    r     = Vector{Float64}(undef, n)
    Tr    = Vector{Float64}(undef, n)
    Tf    = Vector{Float64}(undef, n)
    Tg    = Vector{Float64}(undef, n)
    Tw    = Vector{Float64}(undef, n)
    At    = Vector{Float64}(undef, n)
    Dturb = Vector{Float64}(undef, n)
    qNL   = Vector{Float64}(undef, n)
    pref  = zeros(Float64, n)

    w_idx   = Vector{Int32}(undef, n)
    jac_pos = zeros(Int32, n, HYGOV_JAC_NENTRIES)

    uvec = psd.uvec_idx
    k = 0
    for device in psd.devices
        device.dtype isa HYGOV || continue
        k += 1
        gov = device.dtype

        bus[k]      = Int32(gov.bus)
        diff_ptr[k] = Int32(device.diff_ptr)
        alg_ptr[k]  = Int32(device.alg_ptr)
        ctrl_ptr[k] = Int32(device.ctrl_ptr)
        par_ptr[k]  = Int32(device.par_ptr)

        R[k]     = gov.R
        r[k]     = max(gov.r,  HYGOV_TMIN)
        Tr[k]    = max(gov.Tr, HYGOV_TMIN)
        Tf[k]    = max(gov.Tf, HYGOV_TMIN)
        Tg[k]    = max(gov.Tg, HYGOV_TMIN)
        Tw[k]    = max(gov.TW, HYGOV_TMIN)
        At[k]    = gov.At
        Dturb[k] = gov.Dturb
        qNL[k]   = gov.qNL

        w_idx[k] = Int32(uvec[device.ctrl_ptr])
    end

    online = fill(true, n)
    return HYGOVTable(n, bus, diff_ptr, alg_ptr, ctrl_ptr, par_ptr,
        R, r, Tr, Tf, Tg, Tw, At, Dturb, qNL,
        pref, w_idx, jac_pos, online)
end

function refresh_hygov_table!(psd)
    table = psd.layout.hygov
    table.n == 0 && return nothing
    k = 0
    for device in psd.devices
        device.dtype isa HYGOV || continue
        k += 1
        table.pref[k] = device.dtype.pref
    end
    @assert k == table.n
    return nothing
end

register_device!(:hygov;
    table_type = HYGOVTable,
    builder    = _build_hygov_table_impl,
    class      = :governor)
