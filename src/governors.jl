mutable struct IEESGO <: AbstractGovernorType
    # representation
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    # topology
    bus::Int64
    id::String
    # parameters
    T1::Float64
    T2::Float64
    T3::Float64
    T4::Float64
    T5::Float64
    T6::Float64
    K1::Float64
    K2::Float64
    K3::Float64
    pmax::Float64
    pmin::Float64
    # initialization-derived parameter — set by initialize_dynamics! from the
    # generator's post-PF Pe. Used by the residual kernel as `pref` in the
    # SatP term. Stored in slot 12 of pvec so the kernel can read SoA.
    pref::Float64
end

function IEESGO(bus, id, T1, T2, T3, T4, T5, T6, K1, K2, K3, pmax, pmin)
    # par_size is 12 now (11 parsed + pref filled during init). pref starts at
    # 0; `initialize_dynamics!` writes the converged value back.
    governor = IEESGO(5, 1, 1, 12, bus, id, T1, T2, T3, T4, T5, T6, K1, K2, K3, pmax, pmin, 0.0)
    return governor
end

function from_data_fields(::Type{IEESGO}, fields::Vector{SubString{String}})
    bus = parse(Int64, fields[1])
    id = String(fields[3])
    T1 = parse(Float64, fields[4])
    T2 = parse(Float64, fields[5])
    T3 = parse(Float64, fields[6])
    T4 = parse(Float64, fields[7])
    T5 = parse(Float64, fields[8])
    T6 = parse(Float64, fields[9])
    K1 = parse(Float64, fields[10])
    K2 = parse(Float64, fields[11])
    K3 = parse(Float64, fields[12])
    pmax = parse(Float64, fields[13])
    pmin = parse(Float64, fields[14])
    IEESGO(bus, id, T1, T2, T3, T4, T5, T6, K1, K2, K3, pmax, pmin)
end

function fill_pvec!(pvec::AbstractArray, dtype::IEESGO)
    pvec[1] = dtype.T1
    pvec[2] = dtype.T2
    pvec[3] = dtype.T3
    pvec[4] = dtype.T4
    pvec[5] = dtype.T5
    pvec[6] = dtype.T6
    pvec[7] = dtype.K1
    pvec[8] = dtype.K2
    pvec[9] = dtype.K3
    pvec[10] = dtype.pmax
    pvec[11] = dtype.pmin
    pvec[12] = dtype.pref
end

function get_device_name(dtype::IEESGO)
    return "IEESGO"
end

function initialize_dynamics!(
        f::AbstractArray,
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::IEESGO
)
    T1 = pvec[1]
    T2 = pvec[2]
    T3 = pvec[3]
    T4 = pvec[4]
    T5 = pvec[5]
    T6 = pvec[6]
    K1 = pvec[7]
    K2 = pvec[8]
    K3 = pvec[9]
    pmax = pvec[10]
    pmin = pvec[11]

    # Unknowns: 5 diff states + 1 alg state (p_m) + 1 ctrl/init unknown (pref).
    PF0 = x0[1]
    PLL = x0[2]
    TP1 = x0[3]
    TP2 = x0[4]
    TP3 = x0[5]
    p_m = x0[6]
    pref = x0[7]

    # At steady state generator speed deviation w = 0.
    w = 0.0

    f[1] = (1.0/T1)*(K1*w - PF0)
    f[2] = (1/T3)*((1.0 - (T2/T3))*PF0 - PLL)
    SatP = pref - (T2/T3)*PF0 - PLL
    f[3] = (1/T4)*(SatP - TP1)
    f[4] = (1/T5)*(K2*TP1 - TP2)
    f[5] = (1/T6)*(K3*TP2 - TP3)
    # algebraic governor output equation: p_m = blend of TP1, TP2, TP3
    f[6] = TP1*(1 - K2) + TP2*(1 - K3) + TP3 - p_m
    # initialization constraint: governor output must equal generator electrical power
    f[7] = p_m - pg
    return nothing
end

function initial_guess!(
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::IEESGO
    )
    K2 = pvec[8]
    K3 = pvec[9]
    pref = pg
    x0[1] = 0.0            # PF0
    x0[2] = 0.0            # PLL
    x0[3] = pref           # TP1
    x0[4] = K2*pref        # TP2
    x0[5] = K2*K3*pref     # TP3
    x0[6] = pg             # p_m
    x0[7] = pref           # pref
    return nothing
end

# ===========================
# TGOV1
# ===========================

mutable struct TGOV1 <: AbstractGovernorType
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    bus::Int64
    id::String
    R::Float64
    T1::Float64
    VMAX::Float64
    VMIN::Float64
    T2::Float64
    T3::Float64
    DT::Float64
    # initialization-derived parameter; filled by initialize_dynamics! and
    # mirrored into pvec slot 8 so the kernel can read it from SoA.
    pref::Float64
end

function TGOV1(bus, id, R, T1, VMAX, VMIN, T2, T3, DT)
    TGOV1(2, 1, 1, 8, bus, id, R, T1, VMAX, VMIN, T2, T3, DT, 0.0)
end

function from_data_fields(::Type{TGOV1}, fields::Vector{SubString{String}})
    bus = parse(Int64, fields[1])
    id = String(fields[3])
    R = parse(Float64, fields[4])
    T1 = parse(Float64, fields[5])
    VMAX = parse(Float64, fields[6])
    VMIN = parse(Float64, fields[7])
    T2 = parse(Float64, fields[8])
    T3 = parse(Float64, fields[9])
    DT = parse(Float64, fields[10])
    TGOV1(bus, id, R, T1, VMAX, VMIN, T2, T3, DT)
end

function fill_pvec!(pvec::AbstractArray, dtype::TGOV1)
    pvec[1] = dtype.R
    pvec[2] = dtype.T1
    pvec[3] = dtype.VMAX
    pvec[4] = dtype.VMIN
    pvec[5] = dtype.T2
    pvec[6] = dtype.T3
    pvec[7] = dtype.DT
    pvec[8] = dtype.pref
end

# Apply mbase/sbase scaling: R/VMAX/VMIN scale by sbase/mbase and DT by
# mbase/sbase. With `ratio = mbase/sbase` here, that means dividing
# R/VMAX/VMIN by ratio and multiplying DT by ratio.
function set_ratio!(dtype::TGOV1, ratio::Float64)
    dtype.R    = dtype.R    / ratio
    dtype.VMAX = dtype.VMAX / ratio
    dtype.VMIN = dtype.VMIN / ratio
    dtype.DT   = dtype.DT   * ratio
end

function get_device_name(dtype::TGOV1)
    return "TGOV1"
end

function initialize_dynamics!(
        f::AbstractArray,
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::TGOV1
)
    R = pvec[1]
    T1 = pvec[2]
    T2 = pvec[5]
    T3 = pvec[6]
    DT = pvec[7]

    # Unknowns: 2 diff (x1, x2) + 1 alg (p_m) + 1 ctrl/init (pref).
    x1 = x0[1]
    x2 = x0[2]
    p_m = x0[3]
    pref = x0[4]

    w = 0.0  # steady state

    f[1] = (-x1 + (1.0 - T2/T3)*x2) / T3
    f[2] = ((pref - w)/R - x2) / T1
    f[3] = x1 + (T2/T3)*x2 - DT*w - p_m
    # init constraint: governor output equals generator electrical power
    f[4] = p_m - pg
    return nothing
end

function initial_guess!(
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::TGOV1
    )
    R = pvec[1]
    T2 = pvec[5]
    T3 = pvec[6]
    x2 = pg
    x1 = (1.0 - T2/T3)*x2
    x0[1] = x1
    x0[2] = x2
    x0[3] = pg          # p_m
    x0[4] = R*pg        # pref
    return nothing
end

# ===========================
# IEEEG1 — IEEE Type 1 speed-governing system (steam turbine)
# ===========================
#
# Structure (unsaturated — Uo/Uc rate limits and Pmax/Pmin position limits
# are parsed but NOT applied, matching the TGOV1/IEESGO convention in this
# package):
#
#   e   = pref - K*w                      speed error through droop gain K
#   x1' = ((1 - T2/T1)*e - x1) / T1       lead-lag (1 + sT2)/(1 + sT1)
#   y1  = x1 + (T2/T1)*e
#   x2' = (y1 - x2) / T3                  valve servo
#   x3' = (x2 - x3) / T4                  turbine stage 1
#   x4' = (x3 - x4) / T5                  turbine stage 2
#   x5' = (x4 - x5) / T6                  turbine stage 3
#   x6' = (x5 - x6) / T7                  turbine stage 4
#   p_m = K1*x3 + K3*x4 + K5*x5 + K7*x6   HP shaft output
#
# K2/K4/K6/K8 are the second-shaft (LP) fractions; this package models the
# single-shaft output only, and in the ACTIVSg2000 data they are uniformly 0.
#
# Zero time constants are common in real .dyr data (T1 = 0 in 22 of 43
# ACTIVSg2000 records; T7 = 0 in all of them). A zero-T stage is an
# algebraic pass-through. It is handled by flooring the time constant at
# IEEEG1_TMIN when the SoA table is built, NOT by branching in the kernel —
# the residual/Jacobian stay branch-free so the same code runs on the GPU.
# Backward Euler is L-stable, so a lag far faster than dt settles to the
# algebraic limit without ringing.

const IEEEG1_TMIN = 1.0e-3

mutable struct IEEEG1 <: AbstractGovernorType
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    bus::Int64
    id::String
    # parameters (PSS/E order)
    JBUS::Float64
    M::Float64
    K::Float64
    T1::Float64
    T2::Float64
    T3::Float64
    Uo::Float64
    Uc::Float64
    Pmax::Float64
    Pmin::Float64
    T4::Float64
    K1::Float64
    K2::Float64
    T5::Float64
    K3::Float64
    K4::Float64
    T6::Float64
    K5::Float64
    K6::Float64
    T7::Float64
    K7::Float64
    K8::Float64
    # initialization-derived; mirrored into pvec slot 23 for the kernel
    pref::Float64
end

function IEEEG1(bus, id, JBUS, M, K, T1, T2, T3, Uo, Uc, Pmax, Pmin,
                T4, K1, K2, T5, K3, K4, T6, K5, K6, T7, K7, K8)
    # 6 diff + 1 alg (p_m) + 1 ctrl (w) ; 22 parsed params + pref = 23
    IEEEG1(6, 1, 1, 23, bus, id, JBUS, M, K, T1, T2, T3, Uo, Uc, Pmax, Pmin,
           T4, K1, K2, T5, K3, K4, T6, K5, K6, T7, K7, K8, 0.0)
end

function from_data_fields(::Type{IEEEG1}, fields::Vector{SubString{String}})
    bus = parse(Int64, fields[1])
    id  = String(fields[3])
    v(i) = parse(Float64, fields[i])
    IEEEG1(bus, id,
           v(4),  v(5),  v(6),  v(7),  v(8),  v(9),  v(10), v(11),
           v(12), v(13), v(14), v(15), v(16), v(17), v(18), v(19),
           v(20), v(21), v(22), v(23), v(24), v(25))
end

# Time constants are stored FLOORED at IEEEG1_TMIN so every consumer
# (residual, Jacobian, initialization) sees the same effective parameter and
# no kernel has to branch on T == 0.
function fill_pvec!(pvec::AbstractArray, dtype::IEEEG1)
    pvec[1]  = dtype.JBUS; pvec[2]  = dtype.M;    pvec[3]  = dtype.K
    pvec[4]  = max(dtype.T1, IEEEG1_TMIN)
    pvec[5]  = dtype.T2
    pvec[6]  = max(dtype.T3, IEEEG1_TMIN)
    pvec[7]  = dtype.Uo;   pvec[8]  = dtype.Uc;   pvec[9]  = dtype.Pmax
    pvec[10] = dtype.Pmin
    pvec[11] = max(dtype.T4, IEEEG1_TMIN)
    pvec[12] = dtype.K1
    pvec[13] = dtype.K2
    pvec[14] = max(dtype.T5, IEEEG1_TMIN)
    pvec[15] = dtype.K3
    pvec[16] = dtype.K4
    pvec[17] = max(dtype.T6, IEEEG1_TMIN)
    pvec[18] = dtype.K5
    pvec[19] = dtype.K6
    pvec[20] = max(dtype.T7, IEEEG1_TMIN)
    pvec[21] = dtype.K7
    pvec[22] = dtype.K8;   pvec[23] = dtype.pref
end

get_device_name(dtype::IEEEG1) = "IEEEG1"

# Pmax/Pmin are on the machine base; K (droop gain) is a per-unit gain that
# scales inversely with base, matching TGOV1's treatment of R.
function set_ratio!(dtype::IEEEG1, ratio::Float64)
    dtype.K    = dtype.K    * ratio
    dtype.Pmax = dtype.Pmax / ratio
    dtype.Pmin = dtype.Pmin / ratio
end

# Total HP-shaft turbine gain. Guarded so a degenerate all-zero record does
# not produce a division by zero during initialization.
@inline function _ieeeg1_ksum(K1, K3, K5, K7)
    s = K1 + K3 + K5 + K7
    return s == 0.0 ? 1.0 : s
end

function initialize_dynamics!(
        f::AbstractArray,
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::IEEEG1
)
    K  = pvec[3]
    T1 = max(pvec[4], IEEEG1_TMIN)
    T2 = pvec[5]
    T3 = max(pvec[6], IEEEG1_TMIN)
    T4 = max(pvec[11], IEEEG1_TMIN)
    K1 = pvec[12]
    T5 = max(pvec[14], IEEEG1_TMIN)
    K3 = pvec[15]
    T6 = max(pvec[17], IEEEG1_TMIN)
    K5 = pvec[18]
    T7 = max(pvec[20], IEEEG1_TMIN)
    K7 = pvec[21]

    x1 = x0[1]; x2 = x0[2]; x3 = x0[3]
    x4 = x0[4]; x5 = x0[5]; x6 = x0[6]
    p_m  = x0[7]
    pref = x0[8]

    w = 0.0  # steady state
    e = pref - K*w
    t2_t1 = T2 / T1
    y1 = x1 + t2_t1 * e

    f[1] = ((1.0 - t2_t1)*e - x1) / T1
    f[2] = (y1 - x2) / T3
    f[3] = (x2 - x3) / T4
    f[4] = (x3 - x4) / T5
    f[5] = (x4 - x5) / T6
    f[6] = (x5 - x6) / T7
    f[7] = K1*x3 + K3*x4 + K5*x5 + K7*x6 - p_m
    # init constraint: turbine output equals generator electrical power
    f[8] = p_m - pg
    return nothing
end

function initial_guess!(
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::IEEEG1
    )
    T1 = max(pvec[4], IEEEG1_TMIN)
    T2 = pvec[5]
    K1 = pvec[12]; K3 = pvec[15]; K5 = pvec[18]; K7 = pvec[21]
    ksum = _ieeeg1_ksum(K1, K3, K5, K7)
    pref = pg / ksum
    # At steady state every stage settles to the lead-lag input `pref`.
    x0[1] = (1.0 - T2/T1) * pref
    x0[2] = pref
    x0[3] = pref
    x0[4] = pref
    x0[5] = pref
    x0[6] = pref
    x0[7] = pg      # p_m
    x0[8] = pref
    return nothing
end

# IEEEG1: pref is the 8th unknown (6 diff + 1 alg + 1 ctrl). Mirror it into
# pvec slot 23 so the batched kernels see the converged value. Without this
# method pref stayed 0 and IEEEG1 was NOT in equilibrium at t = 0 (its first
# diff row carried a residual of -x1/T1).
function extract_init_params!(dtype::IEEEG1, sol_zero::AbstractArray,
                              p::AbstractArray, par_ptr::Int)
    dtype.pref = sol_zero[8]
    p[par_ptr + 22] = dtype.pref   # slot 23, zero-based offset 22
    return nothing
end

# ===========================
# GGOV1 — GE general-purpose governor / turbine
# ===========================
#
# ⚠️ SCOPE: GOVERNOR (PID) PATH ONLY — THIS IS NOT FULL GGOV1.
#
# The complete GGOV1 combines three control paths with a **low-value
# select**:
#     fsr = min(fsr_gov, fsr_load_limiter, fsr_accel_limiter)
# `min` is non-differentiable and would fight the analytic Jacobian used
# by this package's Newton solver. The load limiter and the acceleration
# limiter *are* limiters, and this package's established convention is
# that limiters are parsed but never applied (TGOV1 stores VMAX/VMIN and
# ignores them, SEXS ignores EMIN/EMAX, IEEEG1 ignores Uo/Uc/Pmax/Pmin).
#
# Accordingly this implementation models the **unsaturated governor PID
# path only** and omits:
#   * the load limiter   (Tfload, Kpload, Kiload, Ldref)
#   * the accel limiter  (Aset, Ka, Ta)
#   * the low-value select
#   * the MW / power controller (Kimw)
#   * every limit and rate limit (maxerr/minerr, Vmax/Vmin, Ropen/Rclose,
#     Rup/Rdown)
#   * the speed deadband `db` and the temperature-detection lead-lag
#     (Tsa/Tsb) and the diesel transport lag `Teng`
#   * the `Flag` fuel-flow-vs-speed multiplier (fuel is taken as
#     independent of speed, i.e. Flag treated as 0). Keeping it out makes
#     the whole model LINEAR, so the Jacobian is state-independent, which
#     matches the IEEEG1/TGOV1 kernel shape and keeps the GPU path simple.
#   * `Trate` — the turbine MW rating. Power quantities are put on MBASE
#     via `set_ratio!`. (In ACTIVSg2000, Trate == MBASE for all 367
#     records, so this is exact there.)
# All of those parameters ARE parsed and stored in pvec, so a future
# saturated variant can pick them up without a data-layout change.
#
# Modelled chain (5 diff states + 1 alg output):
#
#   x_pelec' = (p_m - x_pelec) / Tpelec        power transducer lag
#   err      = pref - w - R * x_pelec          droop summing junction
#   x_igov'  = Kigov * err                     PI integral state
#   x_dgov'  = ((Kdgov/Tdgov)*err - x_dgov)/Tdgov   derivative washout
#   fsr      = Kpgov*err + x_igov + ((Kdgov/Tdgov)*err - x_dgov)
#   x_act'   = (fsr - x_act) / Tact            actuator lag; x_act = valve
#   q        = Kturb * (x_act - Wfnl)          turbine gain, no-load fuel
#   x_tb'    = ((1 - Tc/Tb)*q - x_tb) / Tb     turbine lead-lag (1+sTc)/(1+sTb)
#   p_m      = x_tb + (Tc/Tb)*q - Dm*w         mechanical power out
#
# DROOP FEEDBACK SUBSTITUTION: all 367 ACTIVSg2000 records set
# Rselect = 1, i.e. the droop feedback is the *measured electrical*
# power. This package's coupling layer only exposes the generator's `w`
# to a governor (`consumes_signals` supports `state_kind = :w` only);
# GENROU's Pe is not a state. The transducer is therefore fed the
# governor's own mechanical output `p_m`. Steady state is identical
# (Pm == Pe at equilibrium) so the droop characteristic dP/dw = -1/R is
# exact; only the transient differs, and Tpelec = 1 s in this data heavily
# filters the difference anyway. This is equivalent to Rselect = -1
# (droop off governor output).
#
# Zero time constants are floored at GGOV1_TMIN in `fill_pvec!` and in
# the SoA table builder — NOT by branching in the kernel, so the `_one!`
# leaf functions stay branch-free for the GPU.
#
# PSS/E / PowerWorld CON order after (bus, 'GGOV1', id), verified against
# all 367 ACTIVSg2000 records (see the file header of src/tables/ggov1.jl):
#   1 Rselect  2 Flag    3 R      4 Tpelec  5 maxerr  6 minerr  7 Kpgov
#   8 Kigov    9 Kdgov  10 Tdgov 11 Vmax   12 Vmin   13 Tact   14 Kturb
#  15 Wfnl    16 Tb     17 Tc    18 Teng   19 Tfload 20 Kpload 21 Kiload
#  22 Ldref   23 Dm     24 Ropen 25 Rclose 26 Kimw   27 Aset   28 Ka
#  29 Ta      30 Trate  31 db    32 Tsa    33 Tsb    34 Rup    35 Rdown
# (Note: 35 CONs, not 36 — the PowerWorld-exported flavour omits Pmwset.)

const GGOV1_TMIN = 1.0e-3

mutable struct GGOV1 <: AbstractGovernorType
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    bus::Int64
    id::String
    # parameters (PSS/E CON order)
    Rselect::Float64
    Flag::Float64
    R::Float64
    Tpelec::Float64
    maxerr::Float64
    minerr::Float64
    Kpgov::Float64
    Kigov::Float64
    Kdgov::Float64
    Tdgov::Float64
    Vmax::Float64
    Vmin::Float64
    Tact::Float64
    Kturb::Float64
    Wfnl::Float64
    Tb::Float64
    Tc::Float64
    Teng::Float64
    Tfload::Float64
    Kpload::Float64
    Kiload::Float64
    Ldref::Float64
    Dm::Float64
    Ropen::Float64
    Rclose::Float64
    Kimw::Float64
    Aset::Float64
    Ka::Float64
    Ta::Float64
    Trate::Float64
    db::Float64
    Tsa::Float64
    Tsb::Float64
    Rup::Float64
    Rdown::Float64
    # initialization-derived; mirrored into pvec slot 36 for the kernel
    pref::Float64
end

function GGOV1(bus, id, Rselect, Flag, R, Tpelec, maxerr, minerr, Kpgov,
               Kigov, Kdgov, Tdgov, Vmax, Vmin, Tact, Kturb, Wfnl, Tb, Tc,
               Teng, Tfload, Kpload, Kiload, Ldref, Dm, Ropen, Rclose,
               Kimw, Aset, Ka, Ta, Trate, db, Tsa, Tsb, Rup, Rdown)
    # 5 diff + 1 alg (p_m) + 1 ctrl (w); 35 parsed params + pref = 36
    GGOV1(5, 1, 1, 36, bus, id, Rselect, Flag, R, Tpelec, maxerr, minerr,
          Kpgov, Kigov, Kdgov, Tdgov, Vmax, Vmin, Tact, Kturb, Wfnl, Tb, Tc,
          Teng, Tfload, Kpload, Kiload, Ldref, Dm, Ropen, Rclose, Kimw,
          Aset, Ka, Ta, Trate, db, Tsa, Tsb, Rup, Rdown, 0.0)
end

function from_data_fields(::Type{GGOV1}, fields::Vector{SubString{String}})
    bus = parse(Int64, fields[1])
    id  = String(fields[3])
    v(i) = parse(Float64, fields[i])
    GGOV1(bus, id,
          v(4),  v(5),  v(6),  v(7),  v(8),  v(9),  v(10), v(11), v(12),
          v(13), v(14), v(15), v(16), v(17), v(18), v(19), v(20), v(21),
          v(22), v(23), v(24), v(25), v(26), v(27), v(28), v(29), v(30),
          v(31), v(32), v(33), v(34), v(35), v(36), v(37), v(38))
end

# Time constants that appear in a denominator (Tpelec, Tdgov, Tact, Tb)
# are stored FLOORED at GGOV1_TMIN so every consumer (residual, Jacobian,
# initialization) sees the same effective parameter and no kernel has to
# branch on T == 0. Tc is a pure numerator (lead) term, so zero is fine.
function fill_pvec!(pvec::AbstractArray, dtype::GGOV1)
    pvec[1]  = dtype.Rselect
    pvec[2]  = dtype.Flag
    pvec[3]  = dtype.R
    pvec[4]  = max(dtype.Tpelec, GGOV1_TMIN)
    pvec[5]  = dtype.maxerr
    pvec[6]  = dtype.minerr
    pvec[7]  = dtype.Kpgov
    pvec[8]  = dtype.Kigov
    pvec[9]  = dtype.Kdgov
    pvec[10] = max(dtype.Tdgov, GGOV1_TMIN)
    pvec[11] = dtype.Vmax
    pvec[12] = dtype.Vmin
    pvec[13] = max(dtype.Tact, GGOV1_TMIN)
    pvec[14] = dtype.Kturb
    pvec[15] = dtype.Wfnl
    pvec[16] = max(dtype.Tb, GGOV1_TMIN)
    pvec[17] = dtype.Tc
    pvec[18] = dtype.Teng
    pvec[19] = dtype.Tfload
    pvec[20] = dtype.Kpload
    pvec[21] = dtype.Kiload
    pvec[22] = dtype.Ldref
    pvec[23] = dtype.Dm
    pvec[24] = dtype.Ropen
    pvec[25] = dtype.Rclose
    pvec[26] = dtype.Kimw
    pvec[27] = dtype.Aset
    pvec[28] = dtype.Ka
    pvec[29] = dtype.Ta
    pvec[30] = dtype.Trate
    pvec[31] = dtype.db
    pvec[32] = dtype.Tsa
    pvec[33] = dtype.Tsb
    pvec[34] = dtype.Rup
    pvec[35] = dtype.Rdown
    pvec[36] = dtype.pref
end

get_device_name(dtype::GGOV1) = "GGOV1"

# Machine-base → system-base conversion, `ratio = mbase/sbase`.
#
# The governor error `err` and the valve stroke `x_act` are dimensionless,
# so the PID gains (Kpgov/Kigov/Kdgov) need no scaling. Power-valued
# quantities do:
#   * R is a droop in (pu speed)/(pu power on MBASE)  → divide by ratio,
#     exactly as TGOV1 does, so dP_sys/dw = -ratio/R.
#   * Kturb maps valve stroke to MBASE power          → multiply by ratio,
#     so `p_m` comes out in system pu and the loop gain Kpgov*Kturb*R is
#     base-invariant.
#   * Dm is a damping power per pu speed              → multiply by ratio.
# Wfnl is a valve stroke (not a power) and is left alone. Vmax/Vmin,
# Ropen/Rclose, Rup/Rdown, Ldref are limits on stroke/load and unused.
function set_ratio!(dtype::GGOV1, ratio::Float64)
    dtype.R     = dtype.R     / ratio
    dtype.Kturb = dtype.Kturb * ratio
    dtype.Dm    = dtype.Dm    * ratio
end

# Guard against a degenerate all-zero record producing a division by zero
# in the closed-form initial guess.
@inline _ggov1_kturb(K) = K == 0.0 ? 1.0 : K

function initialize_dynamics!(
        f::AbstractArray,
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::GGOV1
)
    R      = pvec[3]
    Tpelec = max(pvec[4],  GGOV1_TMIN)
    Kpgov  = pvec[7]
    Kdgov  = pvec[9]
    Tdgov  = max(pvec[10], GGOV1_TMIN)
    Tact   = max(pvec[13], GGOV1_TMIN)
    Kturb  = pvec[14]
    Wfnl   = pvec[15]
    Tb     = max(pvec[16], GGOV1_TMIN)
    Tc     = pvec[17]
    Dm     = pvec[23]

    # Unknowns: 5 diff + 1 alg (p_m) + 1 ctrl/init unknown (pref).
    x_pelec = x0[1]
    x_igov  = x0[2]
    x_dgov  = x0[3]
    x_act   = x0[4]
    x_tb    = x0[5]
    p_m     = x0[6]
    pref    = x0[7]

    w = 0.0  # steady state

    err  = pref - w - R * x_pelec
    kd   = Kdgov / Tdgov
    dgov = kd * err - x_dgov
    fsr  = Kpgov * err + x_igov + dgov
    q    = Kturb * (x_act - Wfnl)
    tcb  = Tc / Tb

    f[1] = (p_m - x_pelec) / Tpelec
    # NOTE: the dynamic row is `Kigov*err`; the equilibrium condition is
    # `err == 0` whenever Kigov != 0. Writing the condition directly keeps
    # the init Jacobian non-singular even for a (nonphysical) Kigov == 0
    # record, and yields the identical solution otherwise.
    f[2] = err
    f[3] = (kd * err - x_dgov) / Tdgov
    f[4] = (fsr - x_act) / Tact
    f[5] = ((1.0 - tcb) * q - x_tb) / Tb
    f[6] = x_tb + tcb * q - Dm * w - p_m
    # init constraint: governor output equals generator electrical power
    f[7] = p_m - pg
    return nothing
end

function initial_guess!(
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::GGOV1
    )
    R     = pvec[3]
    Kturb = _ggov1_kturb(pvec[14])
    Wfnl  = pvec[15]
    Tb    = max(pvec[16], GGOV1_TMIN)
    Tc    = pvec[17]

    # Exact closed-form equilibrium: err = 0, valve holds pg, integrator
    # holds the valve command, derivative washout is empty.
    valve = pg / Kturb + Wfnl
    x0[1] = pg                      # x_pelec
    x0[2] = valve                   # x_igov
    x0[3] = 0.0                     # x_dgov
    x0[4] = valve                   # x_act
    x0[5] = (1.0 - Tc / Tb) * pg    # x_tb
    x0[6] = pg                      # p_m
    x0[7] = R * pg                  # pref
    return nothing
end

# pref is the 7th unknown (5 diff + 1 alg + 1 ctrl). Mirror it into pvec
# slot 36 so the batched kernels reading the SoA snapshot see it.
function extract_init_params!(dtype::GGOV1, sol_zero::AbstractArray,
                              p::AbstractArray, par_ptr::Int)
    dtype.pref = sol_zero[7]
    p[par_ptr + 35] = dtype.pref   # slot 36, zero-based offset 35
    return nothing
end

# ===========================
# HYGOV — PSS/E hydro turbine-governor
# ===========================
#
# Chain (speed-deviation input w = Genrou's w state; states x1..x4, output p_m):
#
#   x1' = (pref - w - R*c - x1) / Tf           speed-error filter
#   c   = x1/r + x2                            PI "desired gate",
#   x2' = x1 / (r*Tr)                             (1 + s Tr)/(r Tr s)
#   g'  = (c - g) / Tg                         gate servo; x3 = g (gate)
#   h   = (q/g)^2                              turbine head
#   q'  = (1 - h) / Tw                         penstock water column; x4 = q
#   p_m = At*h*(q - qNL) - Dturb*g*w           turbine mechanical power
#
# The head/flow relation (flow ∝ gate·√head, i.e. h = (q/g)^2) and the
# At·h·(q − qNL) power are the hydro nonlinearity and are modelled exactly,
# so the Jacobian is state-dependent (like ESST4B/IEEEST, unlike
# TGOV1/IEEEG1/GGOV1).
#
# NOT modelled (package convention: limits parsed, not applied): the gate
# velocity limit VELM and the gate position limits GMAX/GMIN. PSS/E's
# permanent-droop feedback R is taken off the desired gate c (as in the
# PSS/E block diagram), so R needs no MBASE rescaling; At and Dturb are
# powers on MBASE and are rescaled by `set_ratio!`.
#
# Zero time constants (Tf, Tr, Tg, Tw) and a zero temporary droop r (a
# divisor) are floored at HYGOV_TMIN in `fill_pvec!` and the table builder;
# the `_one!` kernels never branch on them.
#
# CON order after (bus, 'HYGOV', id) — derived from a per-field census of
# all 25 ACTIVSg2000 records (15 tokens each => 12 CONs), see the header of
# src/tables/hygov.jl:
#   1 R   2 r   3 Tr   4 Tf   5 Tg   6 VELM   7 GMAX   8 GMIN
#   9 TW  10 At  11 Dturb  12 qNL
#
# Equilibrium (w = 0, given the generator's pg on system base):
#   h = 1, q = g = pg/At + qNL, c = g, x1 = 0, x2 = c, pref = R*c.
# pref is the init-derived unknown (6th) mirrored into pvec slot 13.

const HYGOV_TMIN = 1.0e-3

mutable struct HYGOV <: AbstractGovernorType
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    bus::Int64
    id::String
    # parameters (PSS/E CON order)
    R::Float64
    r::Float64
    Tr::Float64
    Tf::Float64
    Tg::Float64
    VELM::Float64
    GMAX::Float64
    GMIN::Float64
    TW::Float64
    At::Float64
    Dturb::Float64
    qNL::Float64
    # initialization-derived; mirrored into pvec slot 13 for the kernel
    pref::Float64
end

function HYGOV(bus, id, R, r, Tr, Tf, Tg, VELM, GMAX, GMIN, TW, At, Dturb, qNL)
    # 4 diff + 1 alg (p_m) + 1 ctrl (w); 12 parsed params + pref = 13
    HYGOV(4, 1, 1, 13, bus, id, R, r, Tr, Tf, Tg, VELM, GMAX, GMIN, TW,
          At, Dturb, qNL, 0.0)
end

function from_data_fields(::Type{HYGOV}, fields::Vector{SubString{String}})
    bus = parse(Int64, fields[1])
    id  = String(fields[3])
    v(i) = parse(Float64, fields[i])
    HYGOV(bus, id, v(4), v(5), v(6), v(7), v(8), v(9), v(10), v(11),
          v(12), v(13), v(14), v(15))
end

# Divisors (r, Tr, Tf, Tg, TW) are stored FLOORED so every consumer
# (residual, Jacobian, init) sees the same effective parameter.
function fill_pvec!(pvec::AbstractArray, dtype::HYGOV)
    pvec[1]  = dtype.R
    pvec[2]  = max(dtype.r,  HYGOV_TMIN)
    pvec[3]  = max(dtype.Tr, HYGOV_TMIN)
    pvec[4]  = max(dtype.Tf, HYGOV_TMIN)
    pvec[5]  = max(dtype.Tg, HYGOV_TMIN)
    pvec[6]  = dtype.VELM
    pvec[7]  = dtype.GMAX
    pvec[8]  = dtype.GMIN
    pvec[9]  = max(dtype.TW, HYGOV_TMIN)
    pvec[10] = dtype.At
    pvec[11] = dtype.Dturb
    pvec[12] = dtype.qNL
    pvec[13] = dtype.pref
end

get_device_name(dtype::HYGOV) = "HYGOV"

# Machine-base → system-base (ratio = mbase/sbase). Gate, flow, head and
# the droop feedback (on gate) are dimensionless; only the power-valued
# At (turbine gain, MBASE power per unit flow) and Dturb (damping power
# per unit speed per unit gate) are rescaled.
function set_ratio!(dtype::HYGOV, ratio::Float64)
    dtype.At    = dtype.At    * ratio
    dtype.Dturb = dtype.Dturb * ratio
end

@inline _hygov_at(At) = At == 0.0 ? 1.0 : At

function initialize_dynamics!(
        f::AbstractArray,
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::HYGOV
)
    R   = pvec[1]
    r   = max(pvec[2], HYGOV_TMIN)
    Tf  = max(pvec[4], HYGOV_TMIN)
    Tg  = max(pvec[5], HYGOV_TMIN)
    Tw  = max(pvec[9], HYGOV_TMIN)
    At  = pvec[10]
    qNL = pvec[12]

    # Unknowns: 4 diff + 1 alg (p_m) + 1 ctrl/init unknown (pref).
    x1   = x0[1]
    x2   = x0[2]
    g    = x0[3]
    q    = x0[4]
    p_m  = x0[5]
    pref = x0[6]

    w = 0.0  # steady state
    c = x1 / r + x2
    h = (q / g)^2

    f[1] = (pref - w - R * c - x1) / Tf
    # dynamic row is x1/(r*Tr); the equilibrium condition is x1 == 0
    f[2] = x1
    f[3] = (c - g) / Tg
    f[4] = (1.0 - h) / Tw
    f[5] = At * h * (q - qNL) - p_m
    # init constraint: governor output equals generator electrical power
    f[6] = p_m - pg
    return nothing
end

function initial_guess!(
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::HYGOV
    )
    R   = pvec[1]
    At  = _hygov_at(pvec[10])
    qNL = pvec[12]
    # Exact closed-form equilibrium (h = 1, q = g = c).
    g = pg / At + qNL
    x0[1] = 0.0      # x1 (filtered speed error)
    x0[2] = g        # x2 (PI integrator) = c
    x0[3] = g        # gate
    x0[4] = g        # flow
    x0[5] = pg       # p_m
    x0[6] = R * g    # pref
    return nothing
end

# pref is the 6th unknown (4 diff + 1 alg + 1 ctrl). It is the ONLY
# init-derived parameter (gate/flow/integrator are states and land in z
# directly). Mirror it into pvec slot 13 so the batched kernels see it.
function extract_init_params!(dtype::HYGOV, sol_zero::AbstractArray,
                              p::AbstractArray, par_ptr::Int)
    dtype.pref = sol_zero[6]
    p[par_ptr + 12] = dtype.pref   # slot 13, zero-based offset 12
    return nothing
end
