# Additional PSS/E exciter models — IEEET1, EXPIC1, EXAC2, EXAC1, ESAC1A,
# SCRX, ESAC6A — sharing one integration pattern:
#
#   * every model subtypes `AbstractStdExciter` (below). That one abstract
#     type drives the generic plumbing: the Genrou e_fd0 stash before the
#     controller init (src/dynamics.jl), the per-device preallocate skip list,
#     the generic SoA table `StdExcTable{M}` (src/tables/std_exciters.jl), the
#     bus -> voltage-index fixup, the IEEEST v_s wiring, the cluster remap and
#     `_set_table_online!`.
#   * each model reads its terminal voltage through `vr_idx` and an optional
#     PSS output through `vs_idx` (0 = no PSS), exactly like SEXS/ESDC1A, and
#     produces e_fd from a DIFFERENTIAL state that `wire_controls!` routes into
#     Genrou ctrl slot 0 (`source_kind = :diff_at`).
#   * zero time constants are FLOORED in `fill_pvec!` (which also produces the
#     table's parameter snapshot), never branched on in the `_one!` kernels.
#   * limiters are PARSED BUT NOT APPLIED (package convention).
#   * the matched Genrou's post-power-flow e_fd0 is stashed in `dtype.efd0`
#     before the controller init; `initial_guess!` builds the whole
#     equilibrium in closed form from it, and `extract_init_params!` mirrors
#     the init-derived parameters (vref, ...) into both the struct and pvec.
#
# Saturation, where a model has it, is the PSS/E quadratic
#     VX(x) = x·SE(x) = B·max(x − A, 0)²
# with (A, B) fitted through (E1, SE1), (E2, SE2) by `_esdc1a_sat_coefficients`
# (src/exciters.jl). It is C¹ (value and slope are both zero at x = A), so
# Newton with the analytic Jacobian is fine, and `max` compiles to a select,
# so the kernels stay branch-free.

abstract type AbstractStdExciter <: AbstractExciterType end

@inline _std_exc_vx(x, A, B) = (d = max(x - A, 0.0); B * d * d)

# PSS/E rectifier regulation function FEX(IN). Only ever evaluated at
# INITIALIZATION (outside the kernels), so the piecewise form is harmless.
function _psse_fex(IN::Float64)
    IN <= 0.0   && return 1.0
    IN <= 0.433 && return 1.0 - 0.577 * IN
    IN < 0.75   && return sqrt(0.75 - IN^2)
    IN <= 1.0   && return 1.732 * (1.0 - IN)
    return 0.0
end

# Stash hook used by the initialize_dynamics! driver (src/dynamics.jl).
_set_efd0!(dtype::AbstractStdExciter, e_fd0::Float64) = (dtype.efd0 = e_fd0; nothing)

# ST/AC gains are per unit on the field-voltage base, not the MVA base.
set_ratio!(dtype::AbstractStdExciter, ratio::Float64) = nothing

# =============================================================================
# IEEET1 — IEEE Type 1 (1968) DC-commutator excitation system
# =============================================================================
#
# PSS/E record order (14 values after BUS / 'IEEET1' / ID), derived from the
# ACTIVSg2000 data (all 23 records have exactly 14 values):
#   TR KA TA VRMAX VRMIN KE TE KF TF SWITCH E1 SE(E1) E2 SE(E2)
#
# Census over the 23 IEEET1 records in ACTIVSg2000_dynamics.dyr
# (min / max / #zeros):
#   TR     0      / 0.06   / 16      KA   25     / 400    / 0
#   TA     0.05   / 0.2    / 0       VRMAX 1     / 7      / 0
#   VRMIN  -7     / -1     / 0       KE   0      / 0      / 23  <-- ALL zero
#   TE     0.2094 / 0.9707 / 0       KF   0.0341 / 0.1188 / 0
#   TF     0.35   / 1      / 0       SWITCH 0    / 0      / 23
#   E1     1.749  / 4.138  / 0       SE1  0.0161 / 0.1778 / 0
#   E2     2.332  / 5.517  / 0       SE2  0.2661 / 0.4475 / 0
# All in physically plausible ranges (KA/TA a rotating-amplifier regulator,
# TE a DC exciter, E2 > E1 with SE2 > SE1).
#
# MODEL. States: 4 diff, 0 alg, 0 ctrl.
#   x[0] = vc    sensed voltage         vc'  = (vt − vc) / TR
#   x[1] = vr    regulator output       vr'  = (KA·(vref − vc + vs − vf) − vr) / TA
#   x[2] = xf    rate-feedback state    xf'  = −((KF/TF)·e_fd + xf) / TF
#   x[3] = e_fd  exciter output         e_fd'= (vr − KE·e_fd − VX(e_fd)) / TE
# with vf = xf + (KF/TF)·e_fd  (= sKF/(1+sTF)·e_fd), VX the quadratic
# saturation above, and vs the IEEEST output (0 when no PSS is attached).
#
# pvec layout (13): TR KA TA VRMAX VRMIN KE_eff TE KF TF SWITCH A B vref

const IEEET1_TMIN = 1.0e-3

mutable struct IEEET1 <: AbstractStdExciter
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    bus::Int64
    id::String
    TR::Float64
    KA::Float64
    TA::Float64
    VRMAX::Float64
    VRMIN::Float64
    KE::Float64
    TE::Float64
    KF::Float64
    TF::Float64
    SWITCH::Float64
    E1::Float64
    SE1::Float64
    E2::Float64
    SE2::Float64
    # derived / init-derived
    sat_a::Float64
    sat_b::Float64
    ke_eff::Float64
    vref::Float64
    efd0::Float64      # scratch: matched Genrou's post-PF e_fd0
end

function IEEET1(bus, id, TR, KA, TA, VRMAX, VRMIN, KE, TE, KF, TF, SWITCH,
                E1, SE1, E2, SE2)
    sat_a, sat_b = _esdc1a_sat_coefficients(Float64(E1), Float64(SE1),
                                            Float64(E2), Float64(SE2))
    IEEET1(4, 0, 0, 13, bus, id, TR, KA, TA, VRMAX, VRMIN, KE, TE, KF, TF,
           SWITCH, E1, SE1, E2, SE2, sat_a, sat_b, KE, 0.0, 0.0)
end

function from_data_fields(::Type{IEEET1}, fields::Vector{SubString{String}})
    bus = parse(Int64, fields[1])
    id  = String(fields[3])
    v(i) = parse(Float64, fields[i])
    IEEET1(bus, id, v(4), v(5), v(6), v(7), v(8), v(9), v(10), v(11), v(12),
           v(13), v(14), v(15), v(16), v(17))
end

get_device_name(::IEEET1) = "IEEET1"

function fill_pvec!(pvec::AbstractArray, d::IEEET1)
    pvec[1]  = max(d.TR, IEEET1_TMIN)
    pvec[2]  = d.KA
    pvec[3]  = max(d.TA, IEEET1_TMIN)
    pvec[4]  = d.VRMAX
    pvec[5]  = d.VRMIN
    pvec[6]  = d.ke_eff          # = KE unless KE == 0 (then init-derived)
    pvec[7]  = max(d.TE, IEEET1_TMIN)
    pvec[8]  = d.KF
    pvec[9]  = max(d.TF, IEEET1_TMIN)
    pvec[10] = d.SWITCH
    pvec[11] = d.sat_a
    pvec[12] = d.sat_b
    pvec[13] = d.vref
    return nothing
end

function initial_guess!(x0::AbstractArray, pvec::AbstractArray,
                        pg::Float64, qg::Float64, vm::Float64, va::Float64,
                        d::IEEET1)
    KA = pvec[2]; KF = pvec[8]; TF = pvec[9]
    A  = pvec[11]; B = pvec[12]
    e0 = d.efd0
    vx0 = _std_exc_vx(e0, A, B)
    ke  = d.KE == 0.0 ? (e0 != 0.0 ? -vx0 / e0 : 0.0) : d.KE
    vr0 = ke * e0 + vx0
    xf0 = -((KF / TF) * e0)
    vref = vm + vr0 / KA
    x0[1] = vm
    x0[2] = KA * (vref - vm + 0.0 - xf0 - (KF / TF) * e0)
    x0[3] = xf0
    x0[4] = e0
    pvec[6]  = ke
    pvec[13] = vref
    d.ke_eff = ke
    d.vref   = vref
    return nothing
end

function initialize_dynamics!(f::AbstractArray, x::AbstractArray, pvec::AbstractArray,
                              pg::Float64, qg::Float64, vm::Float64, va::Float64,
                              d::IEEET1)
    TR = pvec[1]; KA = pvec[2]; TA = pvec[3]; KE = pvec[6]; TE = pvec[7]
    KF = pvec[8]; TF = pvec[9]; A = pvec[11]; B = pvec[12]; vref = pvec[13]
    vc = x[1]; vr = x[2]; xf = x[3]; efd = x[4]
    kf = KF / TF
    f[1] = (vm - vc) / TR
    f[2] = (KA * (vref - vc - xf - kf * efd) - vr) / TA
    f[3] = -(kf * efd + xf) / TF
    f[4] = (vr - KE * efd - _std_exc_vx(efd, A, B)) / TE
    return nothing
end

function extract_init_params!(d::IEEET1, sol_zero::AbstractArray, p::AbstractArray, par_ptr::Int)
    p[par_ptr + 5]  = d.ke_eff     # slot 6
    p[par_ptr + 12] = d.vref       # slot 13
    return nothing
end

# =============================================================================
# EXPIC1 — proportional/integral excitation system (PSS/E EXPIC1)
# =============================================================================
#
# PSS/E record order (24 values after BUS / 'EXPIC1' / ID), derived from the
# ACTIVSg2000 data (all 61 records have exactly 24 values):
#   TR KA TA1 VR1 VR2 TA2 TA3 TA4 VRMAX VRMIN KF TF1 TF2 EFDMAX EFDMIN
#   KE TE E1 SE(E1) E2 SE(E2) KP KI KC
#
# Census over the 61 EXPIC1 records in ACTIVSg2000_dynamics.dyr
# (min / max / #zeros):
#   TR    0    / 0     / 61   KA    3     / 4     / 0    TA1  1  / 1  / 0
#   VR1   1    / 1     / 0    VR2  -0.87  / -0.87 / 0    TA2  0  / 0  / 61
#   TA3   0    / 0     / 61   TA4   0     / 0     / 61   VRMAX 1 / 1  / 0
#   VRMIN -0.87/ -0.87 / 0    KF    0     / 0     / 61   TF1  1  / 1  / 0
#   TF2   0    / 0     / 61   EFDMAX 6.45 / 8.286 / 0    EFDMIN 0 / 0 / 61
#   KE    0    / 0     / 61   TE    0     / 0     / 61   E1   1  / 1  / 0
#   SE1   0    / 0     / 61   E2    1.2   / 1.2   / 0    SE2  0  / 0  / 61
#   KP    5.06 / 6.97  / 0    KI    0     / 0     / 61   KC   0.08 / 0.08 / 0
# i.e. every record is a bus-fed static exciter: PI regulator (KA/s + KA·TA1)
# driving a potential-source bridge (KP ≈ 6, KC = 0.08), no rotating exciter
# (TE = KE = 0), no rate feedback (KF = 0), unity lead-lag (TA2..TA4 = 0).
# EFDMAX ≈ KP·VRMAX, consistent with that reading.
#
# MODEL. States: 7 diff, 0 alg, 0 ctrl.
#   err  = vref − vc + vs − vf                     (vs = PSS output, 0 if none)
#   va   = xi + KA·TA1·err                         (PI output, KA(1+sTA1)/s)
#   out1 = xll + (TA2/TA3)·(va − xll)              (lead-lag (1+sTA2)/(1+sTA3))
#   x[0] = vc    vc'   = (vt − vc)/TR
#   x[1] = xi    xi'   = KA·err                    (PI integrator; no diagonal)
#   x[2] = xll   xll'  = (va − xll)/TA3
#   x[3] = vr    vr'   = (out1 − vr)/TA4           (lag 1/(1+sTA4))
#   x[4] = e_fd  e_fd' = (vr·VB0 − KE·e_fd − VX(e_fd))/TE
#   x[5] = y1    y1'   = (e_fd − y1)/TF1           (rate feedback, stage 1)
#   x[6] = vf    vf'   = (KF·(e_fd − y1)/TF1 − vf)/TF2
#                        (vf = sKF/((1+sTF1)(1+sTF2)) · e_fd)
#
# Initialization: the PI integrator makes the init Jacobian singular (the
# equilibrium is a one-parameter family pinned only by e_fd = e_fd0), so, as
# for ESST4B, `initial_guess!` builds an equilibrium whose residual is EXACTLY
# zero in floating point (e_fd is taken as vr0·VB0 in the static case) and
# nlsolve accepts it at iteration 0.
#
# pvec layout (28):
#   1 TR  2 KA  3 TA1  4 VR1  5 VR2  6 TA2  7 TA3  8 TA4  9 VRMAX 10 VRMIN
#  11 KF 12 TF1 13 TF2 14 EFDMAX 15 EFDMIN 16 KE_eff 17 TE_eff 18 E1 19 SE1
#  20 E2 21 SE2 22 KP_eff 23 KI 24 KC 25 A 26 B 27 vref 28 VB0

const EXPIC1_TMIN = 1.0e-3

mutable struct EXPIC1 <: AbstractStdExciter
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    bus::Int64
    id::String
    TR::Float64
    KA::Float64
    TA1::Float64
    VR1::Float64
    VR2::Float64
    TA2::Float64
    TA3::Float64
    TA4::Float64
    VRMAX::Float64
    VRMIN::Float64
    KF::Float64
    TF1::Float64
    TF2::Float64
    EFDMAX::Float64
    EFDMIN::Float64
    KE::Float64
    TE::Float64
    E1::Float64
    SE1::Float64
    E2::Float64
    SE2::Float64
    KP::Float64
    KI::Float64
    KC::Float64
    # derived / init-derived
    sat_a::Float64
    sat_b::Float64
    vref::Float64
    vb0::Float64
    efd0::Float64
end

function EXPIC1(bus, id, TR, KA, TA1, VR1, VR2, TA2, TA3, TA4, VRMAX, VRMIN,
                KF, TF1, TF2, EFDMAX, EFDMIN, KE, TE, E1, SE1, E2, SE2, KP, KI, KC)
    sat_a, sat_b = _esdc1a_sat_coefficients(Float64(E1), Float64(SE1),
                                            Float64(E2), Float64(SE2))
    EXPIC1(7, 0, 0, 28, bus, id, TR, KA, TA1, VR1, VR2, TA2, TA3, TA4, VRMAX,
           VRMIN, KF, TF1, TF2, EFDMAX, EFDMIN, KE, TE, E1, SE1, E2, SE2,
           KP, KI, KC, sat_a, sat_b, 0.0, 0.0, 0.0)
end

function from_data_fields(::Type{EXPIC1}, fields::Vector{SubString{String}})
    bus = parse(Int64, fields[1])
    id  = String(fields[3])
    v(i) = parse(Float64, fields[i])
    EXPIC1(bus, id, (v(i) for i in 4:27)...)
end

get_device_name(::EXPIC1) = "EXPIC1"

_expic1_static(d::EXPIC1) = d.TE <= 0.0

function fill_pvec!(pvec::AbstractArray, d::EXPIC1)
    st = _expic1_static(d)
    pvec[1]  = max(d.TR, EXPIC1_TMIN)
    pvec[2]  = d.KA
    pvec[3]  = d.TA1
    pvec[4]  = d.VR1
    pvec[5]  = d.VR2
    pvec[6]  = d.TA2
    pvec[7]  = max(d.TA3, EXPIC1_TMIN)
    pvec[8]  = max(d.TA4, EXPIC1_TMIN)
    pvec[9]  = d.VRMAX
    pvec[10] = d.VRMIN
    pvec[11] = d.KF
    pvec[12] = max(d.TF1, EXPIC1_TMIN)
    pvec[13] = max(d.TF2, EXPIC1_TMIN)
    pvec[14] = d.EFDMAX
    pvec[15] = d.EFDMIN
    pvec[16] = st ? 1.0 : d.KE
    pvec[17] = max(d.TE, EXPIC1_TMIN)
    pvec[18] = d.E1
    pvec[19] = d.SE1
    pvec[20] = d.E2
    pvec[21] = d.SE2
    pvec[22] = d.KP <= 0.0 ? 1.0 : d.KP
    pvec[23] = d.KI
    pvec[24] = d.KC
    pvec[25] = st ? 0.0 : d.sat_a
    pvec[26] = st ? 0.0 : d.sat_b
    pvec[27] = d.vref
    pvec[28] = d.vb0
    return nothing
end

function initial_guess!(x0::AbstractArray, pvec::AbstractArray,
                        pg::Float64, qg::Float64, vm::Float64, va::Float64,
                        d::EXPIC1)
    KE = pvec[16]; KP = pvec[22]; KC = pvec[24]; A = pvec[25]; B = pvec[26]
    e0  = d.efd0
    ve0 = KP * vm
    vb0 = ve0 * _psse_fex(KC * e0 / ve0)
    vb0 <= 0.0 && (vb0 = ve0)
    vr0 = (KE * e0 + _std_exc_vx(e0, A, B)) / vb0
    efd = (KE == 1.0 && B == 0.0) ? vr0 * vb0 : e0
    x0[1] = vm      # vc
    x0[2] = vr0     # xi  (err = 0 ⇒ va = xi = vr0)
    x0[3] = vr0     # xll
    x0[4] = vr0     # vr
    x0[5] = efd
    x0[6] = efd     # y1
    x0[7] = 0.0     # vf
    pvec[27] = vm   # vref = vc0 (vs = vf = 0 at equilibrium)
    pvec[28] = vb0
    d.vref = vm
    d.vb0  = vb0
    return nothing
end

function initialize_dynamics!(f::AbstractArray, x::AbstractArray, pvec::AbstractArray,
                              pg::Float64, qg::Float64, vm::Float64, va::Float64,
                              d::EXPIC1)
    TR = pvec[1]; KA = pvec[2]; TA1 = pvec[3]; TA2 = pvec[6]; TA3 = pvec[7]
    TA4 = pvec[8]; KF = pvec[11]; TF1 = pvec[12]; TF2 = pvec[13]
    KE = pvec[16]; TE = pvec[17]; A = pvec[25]; B = pvec[26]
    vref = pvec[27]; vb0 = pvec[28]
    vc = x[1]; xi = x[2]; xll = x[3]; vr = x[4]; efd = x[5]; y1 = x[6]; vf = x[7]
    err  = vref - vc + 0.0 - vf
    vpi  = xi + KA * TA1 * err
    out1 = xll + (TA2 / TA3) * (vpi - xll)
    f[1] = (vm - vc) / TR
    f[2] = KA * err
    f[3] = (vpi - xll) / TA3
    f[4] = (out1 - vr) / TA4
    f[5] = (vr * vb0 - KE * efd - _std_exc_vx(efd, A, B)) / TE
    f[6] = (efd - y1) / TF1
    f[7] = (KF * (efd - y1) / TF1 - vf) / TF2
    return nothing
end

function extract_init_params!(d::EXPIC1, sol_zero::AbstractArray, p::AbstractArray, par_ptr::Int)
    p[par_ptr + 26] = d.vref       # slot 27
    p[par_ptr + 27] = d.vb0        # slot 28
    return nothing
end

# =============================================================================
# EXAC2 — IEEE Type AC2 high-initial-response alternator-rectifier exciter
# =============================================================================
#
# PSS/E record order (23 values after BUS / 'EXAC2' / ID), derived from the
# ACTIVSg2000 data (all 38 records have exactly 23 values):
#   TR TB TC KA TA VAMAX VAMIN KB VRMAX VRMIN TE KL KH KF TF KC KD KE VLR
#   E1 SE(E1) E2 SE(E2)
#
# Census over the 38 EXAC2 records in ACTIVSg2000_dynamics.dyr
# (min / max / #zeros):
#   TR    0      / 0.05   / 27   TB   1   / 1   / 0    TC   1    / 1    / 0
#   KA    400    / 1000   / 0    TA   0.05/ 0.05/ 0    VAMAX 7.72 / 100  / 0
#   VAMIN -100   / -7.72  / 0    KB   1   / 1   / 0    VRMAX 21.2 / 29.7 / 0
#   VRMIN -29.7  / -21.2  / 0    TE   0.55/ 1.3 / 0    KL   4    / 4    / 0
#   KH    0      / 0      / 38   KF   0.0135 / 0.49 / 0  TF 1   / 1    / 0
#   KC    0      / 0.1    / 12   KD   0   / 1.6 / 12   KE   1    / 1    / 0
#   VLR   10     / 14     / 0    E1   3.05/ 3.87/ 0    SE1  0.0004 / 1.091 / 0
#   E2    4.07   / 5.17   / 0    SE2  0.0102 / 1.136 / 0
# Plausible for an AC2 alternator exciter (KA·KB ~ 400-1000, TE ~ 1 s,
# VRMAX ~ 25 with VLR/KL as the field-current limiter). Note 11 of the 38
# records have SE(E1) ≈ 0.9-1.09 and SE(E2) only slightly larger; the fitted
# quadratic knee is then NEGATIVE (A ≈ −2.8 … −3.3), i.e. saturation is
# already active at VE = 0. That is what the PSS/E quadratic gives for these
# numbers, and it is used as fitted.
#
# MODEL. States: 5 diff, 0 alg, 0 ctrl.
#   VE   = kfex·e_fd                       (exciter alternator voltage, below)
#   VFE  = KE·VE + VX(VE) + KD·IFD,  IFD ≈ e_fd   (see 2)
#   vf   = xf + (KF/TF)·VFE                (= sKF/(1+sTF)·VFE)
#   u    = vref − vc + vs − vf              (vs = PSS output, 0 if none)
#   out  = xll + (TC/TB)·(u − xll)          (lead-lag (1+sTC)/(1+sTB))
#   VR   = KB·(va − KH·VFE)
#   x[0] = vc    vc'   = (vt − vc)/TR
#   x[1] = xll   xll'  = (u − xll)/TB
#   x[2] = va    va'   = (KA·out − va)/TA
#   x[3] = e_fd  e_fd' = (VR − VFE)/(TE·kfex)   (TE·VE' = VR − VFE)
#   x[4] = xf    xf'   = −((KF/TF)·VFE + xf)/TF
#
# pvec layout (27):
#   1 TR  2 TB  3 TC_eff  4 KA  5 TA  6 VAMAX  7 VAMIN  8 KB  9 VRMAX
#  10 VRMIN 11 TE 12 KL 13 KH 14 KF 15 TF 16 KC 17 KD 18 KE 19 VLR
#  20 E1 21 SE1 22 E2 23 SE2 24 A 25 B 26 kfex 27 vref

const EXAC2_TMIN = 1.0e-3

# EXAC2, EXAC1 and ESAC1A share ONE parameter layout (the EXAC2 pvec layout
# below) and one set of kernels (src/kernels/exac2.jl): an AC1-type exciter is
# the AC2 structure with KB = 1 and KH = 0 (and no LV gate, which is a limiter
# and not applied anyway). Initialization is shared through this abstract type.
abstract type AbstractACExciter <: AbstractStdExciter end

mutable struct EXAC2 <: AbstractACExciter
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    bus::Int64
    id::String
    TR::Float64
    TB::Float64
    TC::Float64
    KA::Float64
    TA::Float64
    VAMAX::Float64
    VAMIN::Float64
    KB::Float64
    VRMAX::Float64
    VRMIN::Float64
    TE::Float64
    KL::Float64
    KH::Float64
    KF::Float64
    TF::Float64
    KC::Float64
    KD::Float64
    KE::Float64
    VLR::Float64
    E1::Float64
    SE1::Float64
    E2::Float64
    SE2::Float64
    # derived / init-derived
    sat_a::Float64
    sat_b::Float64
    vref::Float64
    efd0::Float64      # scratch: matched Genrou's post-PF e_fd0
end

function EXAC2(bus, id, TR, TB, TC, KA, TA, VAMAX, VAMIN, KB, VRMAX, VRMIN,
               TE, KL, KH, KF, TF, KC, KD, KE, VLR, E1, SE1, E2, SE2)
    sat_a, sat_b = _esdc1a_sat_coefficients(Float64(E1), Float64(SE1),
                                            Float64(E2), Float64(SE2))
    EXAC2(5, 0, 0, 27, bus, id, TR, TB, TC, KA, TA, VAMAX, VAMIN, KB, VRMAX,
          VRMIN, TE, KL, KH, KF, TF, KC, KD, KE, VLR, E1, SE1, E2, SE2,
          sat_a, sat_b, 0.0, 0.0)
end

function from_data_fields(::Type{EXAC2}, fields::Vector{SubString{String}})
    bus = parse(Int64, fields[1])
    id  = String(fields[3])
    v(i) = parse(Float64, fields[i])
    d = EXAC2(bus, id, (v(i) for i in 4:26)...)
    d.KC > 0.577 && @warn "EXAC2 at bus $bus id $id: KC = $(d.KC) > 0.577, " *
        "outside FEX mode 1; the e_fd = VE/(1+0.577·KC) rectifier form is used anyway"
    return d
end

get_device_name(::EXAC2) = "EXAC2"

function fill_pvec!(pvec::AbstractArray, d::EXAC2)
    TB = max(d.TB, EXAC2_TMIN)
    pvec[1]  = max(d.TR, EXAC2_TMIN)
    pvec[2]  = TB
    pvec[3]  = d.TB <= 0.0 ? TB : d.TC       # TB = 0: lead-lag bypassed
    pvec[4]  = d.KA == 0.0 ? 1.0 : d.KA
    pvec[5]  = max(d.TA, EXAC2_TMIN)
    pvec[6]  = d.VAMAX           # parsed, not applied
    pvec[7]  = d.VAMIN           # parsed, not applied
    pvec[8]  = d.KB == 0.0 ? 1.0 : d.KB
    pvec[9]  = d.VRMAX           # parsed, not applied
    pvec[10] = d.VRMIN           # parsed, not applied
    pvec[11] = max(d.TE, EXAC2_TMIN)
    pvec[12] = d.KL              # parsed, not applied (LV gate)
    pvec[13] = d.KH
    pvec[14] = d.KF
    pvec[15] = max(d.TF, EXAC2_TMIN)
    pvec[16] = d.KC              # enters only through kfex (slot 26)
    pvec[17] = d.KD
    pvec[18] = d.KE
    pvec[19] = d.VLR             # parsed, not applied (LV gate)
    pvec[20] = d.E1
    pvec[21] = d.SE1
    pvec[22] = d.E2
    pvec[23] = d.SE2
    pvec[24] = d.sat_a
    pvec[25] = d.sat_b
    pvec[26] = 1.0 + 0.577 * d.KC
    pvec[27] = d.vref
    return nothing
end

@inline _exac2_vfe(efd, KE, KD, A, B, kfex) = KE * (kfex * efd) + _std_exc_vx(kfex * efd, A, B) + KD * efd

function initial_guess!(x0::AbstractArray, pvec::AbstractArray,
                        pg::Float64, qg::Float64, vm::Float64, va::Float64,
                        d::AbstractACExciter)
    KA = pvec[4]; KB = pvec[8]; KH = pvec[13]; KF = pvec[14]; TF = pvec[15]
    KD = pvec[17]; KE = pvec[18]; A = pvec[24]; B = pvec[25]; kfex = pvec[26]
    e0   = d.efd0
    vfe0 = _exac2_vfe(e0, KE, KD, A, B, kfex)
    u0   = (vfe0 / KB + KH * vfe0) / KA    # VR0 = VFE0; unity-DC-gain lead-lag
    vref = vm + u0                         # vf0 = 0
    # Rebuild u exactly as the residual evaluates it, and derive xll/va from
    # that value, so the stiff rows (gain KA/TA ~ 2e4) are EXACTLY zero at the
    # guess; otherwise roundoff times KA/TA exceeds nlsolve's ftol = 1e-12 and
    # it stalls at ~1e-11 (reported as "did not converge"). The KB row then
    # carries only O(1e-16) roundoff.
    u  = vref - vm + 0.0
    x0[1] = vm
    x0[2] = u
    x0[3] = KA * u
    x0[4] = e0
    x0[5] = -((KF / TF) * vfe0)
    pvec[27] = vref
    d.vref   = vref
    return nothing
end

function initialize_dynamics!(f::AbstractArray, x::AbstractArray, pvec::AbstractArray,
                              pg::Float64, qg::Float64, vm::Float64, va::Float64,
                              d::AbstractACExciter)
    TR = pvec[1]; TB = pvec[2]; TC = pvec[3]; KA = pvec[4]; TA = pvec[5]
    KB = pvec[8]; TE = pvec[11]; KH = pvec[13]; KF = pvec[14]; TF = pvec[15]
    KD = pvec[17]; KE = pvec[18]; A = pvec[24]; B = pvec[25]; kfex = pvec[26]
    vref = pvec[27]
    vc = x[1]; xll = x[2]; vA = x[3]; efd = x[4]; xf = x[5]
    kf  = KF / TF
    vfe = _exac2_vfe(efd, KE, KD, A, B, kfex)
    u   = vref - vc + 0.0 - (xf + kf * vfe)
    out = xll + (TC / TB) * (u - xll)
    f[1] = (vm - vc) / TR
    f[2] = (u - xll) / TB
    f[3] = (KA * out - vA) / TA
    f[4] = (KB * (vA - KH * vfe) - vfe) / (TE * kfex)
    f[5] = -(kf * vfe + xf) / TF
    return nothing
end

function extract_init_params!(d::AbstractACExciter, sol_zero::AbstractArray, p::AbstractArray, par_ptr::Int)
    p[par_ptr + 26] = d.vref       # slot 27
    return nothing
end

# =============================================================================
# EXAC1 — IEEE Type AC1 alternator-rectifier exciter (PSS/E EXAC1)
# ESAC1A — IEEE 421.5-2005 Type AC1A (PSS/E ESAC1A)
# =============================================================================
#
# Both are run through the EXAC2 kernels with KB = 1, KH = 0 (see
# AbstractACExciter above); everything in the EXAC2 header (IFD ≈ e_fd, the
# exact FEX-mode-1 form e_fd = VE/(1 + 0.577·KC), quadratic saturation on VE,
# limiters parsed but not applied, zero-T floors, TB = 0 ⇒ lead-lag bypassed)
# applies unchanged.
#
# PSS/E record orders, derived from ACTIVSg2000 (all EXAC1 records have 17
# values, all ESAC1A records 19):
#   EXAC1 : TR TB TC KA TA VRMAX VRMIN TE KF TF KC KD KE E1 SE1 E2 SE2
#   ESAC1A: TR TB TC KA TA VAMAX VAMIN TE KF TF KC KD KE E1 SE1 E2 SE2 VRMAX VRMIN
#
# Census, 6 EXAC1 records (min / max / #zeros):
#   TR 0/0.05/5   TB 0/0/6   TC 0/0/6   KA 400/400/0   TA 0.05/0.1/0
#   VRMAX 5.25/12.2/0  VRMIN -12.2/-5.25/0  TE 0.327/1.1/0  KF 0.02/0.08/0
#   TF 1/1/0  KC 0.205/0.293/0  KD 0.5/0.5/0  KE 1/1/0
#   E1 1.74/4.09/0  SE1 0.030/0.091/0  E2 2.32/5.46/0  SE2 0.151/0.457/0
# Census, 4 ESAC1A records:
#   TR 0/0.025/3  TB 0/0/4  TC 0/0/4  KA 230/326/0  TA 0.05/0.1/0
#   VAMAX 15.2/50/0  VAMIN -50/-15.2/0  TE 0.936/1.086/0  KF 0.0109/0.0243/0
#   TF 1/1/0  KC 0.2/0.2/0  KD 0.4/1/0  KE 1/1/0  E1 2.27/5.6/0
#   SE1 0.0615/0.251/0  E2 3.02/7.47/0  SE2 0.553/2.26/0  VRMAX 9/9/0
#   VRMIN -9/-9/0
# All plausible for AC1 exciters. TB = TC = 0 everywhere (no lead-lag). KC up
# to 0.293 keeps IN = KC/(1 + 0.577·KC) ≤ 0.25, inside FEX mode 1.
# EXAC1 has a single regulator limit pair (VRMAX/VRMIN, on the KA stage); it
# is stored in both the VA and VR slots. Neither is applied.

mutable struct EXAC1 <: AbstractACExciter
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    bus::Int64
    id::String
    TR::Float64
    TB::Float64
    TC::Float64
    KA::Float64
    TA::Float64
    VRMAX::Float64
    VRMIN::Float64
    TE::Float64
    KF::Float64
    TF::Float64
    KC::Float64
    KD::Float64
    KE::Float64
    E1::Float64
    SE1::Float64
    E2::Float64
    SE2::Float64
    sat_a::Float64
    sat_b::Float64
    vref::Float64
    efd0::Float64
end

function EXAC1(bus, id, TR, TB, TC, KA, TA, VRMAX, VRMIN, TE, KF, TF, KC, KD, KE,
               E1, SE1, E2, SE2)
    sat_a, sat_b = _esdc1a_sat_coefficients(Float64(E1), Float64(SE1),
                                            Float64(E2), Float64(SE2))
    EXAC1(5, 0, 0, 27, bus, id, TR, TB, TC, KA, TA, VRMAX, VRMIN, TE, KF, TF,
          KC, KD, KE, E1, SE1, E2, SE2, sat_a, sat_b, 0.0, 0.0)
end

mutable struct ESAC1A <: AbstractACExciter
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    bus::Int64
    id::String
    TR::Float64
    TB::Float64
    TC::Float64
    KA::Float64
    TA::Float64
    VAMAX::Float64
    VAMIN::Float64
    TE::Float64
    KF::Float64
    TF::Float64
    KC::Float64
    KD::Float64
    KE::Float64
    E1::Float64
    SE1::Float64
    E2::Float64
    SE2::Float64
    VRMAX::Float64
    VRMIN::Float64
    sat_a::Float64
    sat_b::Float64
    vref::Float64
    efd0::Float64
end

function ESAC1A(bus, id, TR, TB, TC, KA, TA, VAMAX, VAMIN, TE, KF, TF, KC, KD, KE,
                E1, SE1, E2, SE2, VRMAX, VRMIN)
    sat_a, sat_b = _esdc1a_sat_coefficients(Float64(E1), Float64(SE1),
                                            Float64(E2), Float64(SE2))
    ESAC1A(5, 0, 0, 27, bus, id, TR, TB, TC, KA, TA, VAMAX, VAMIN, TE, KF, TF,
           KC, KD, KE, E1, SE1, E2, SE2, VRMAX, VRMIN, sat_a, sat_b, 0.0, 0.0)
end

function _ac_kc_check(name, d)
    d.KC > 0.577 && @warn "$name at bus $(d.bus) id $(d.id): KC = $(d.KC) > 0.577, " *
        "outside FEX mode 1; the e_fd = VE/(1+0.577·KC) rectifier form is used anyway"
    return d
end

function from_data_fields(::Type{EXAC1}, fields::Vector{SubString{String}})
    v(i) = parse(Float64, fields[i])
    _ac_kc_check("EXAC1", EXAC1(parse(Int64, fields[1]), String(fields[3]),
                                (v(i) for i in 4:20)...))
end

function from_data_fields(::Type{ESAC1A}, fields::Vector{SubString{String}})
    v(i) = parse(Float64, fields[i])
    _ac_kc_check("ESAC1A", ESAC1A(parse(Int64, fields[1]), String(fields[3]),
                                  (v(i) for i in 4:22)...))
end

get_device_name(::EXAC1)  = "EXAC1"
get_device_name(::ESAC1A) = "ESAC1A"

# Shared AC1 fill into the EXAC2 layout. VA/VR limits are passed explicitly.
function _fill_ac1_pvec!(pvec, d, VAMAX, VAMIN, VRMAX, VRMIN)
    TB = max(d.TB, EXAC2_TMIN)
    pvec[1]  = max(d.TR, EXAC2_TMIN)
    pvec[2]  = TB
    pvec[3]  = d.TB <= 0.0 ? TB : d.TC       # TB = 0: lead-lag bypassed
    pvec[4]  = d.KA == 0.0 ? 1.0 : d.KA
    pvec[5]  = max(d.TA, EXAC2_TMIN)
    pvec[6]  = VAMAX             # parsed, not applied
    pvec[7]  = VAMIN             # parsed, not applied
    pvec[8]  = 1.0               # KB (AC1: no second regulator stage)
    pvec[9]  = VRMAX             # parsed, not applied
    pvec[10] = VRMIN             # parsed, not applied
    pvec[11] = max(d.TE, EXAC2_TMIN)
    pvec[12] = 0.0               # KL (no LV gate in AC1)
    pvec[13] = 0.0               # KH
    pvec[14] = d.KF
    pvec[15] = max(d.TF, EXAC2_TMIN)
    pvec[16] = d.KC
    pvec[17] = d.KD
    pvec[18] = d.KE
    pvec[19] = 0.0               # VLR (no LV gate in AC1)
    pvec[20] = d.E1
    pvec[21] = d.SE1
    pvec[22] = d.E2
    pvec[23] = d.SE2
    pvec[24] = d.sat_a
    pvec[25] = d.sat_b
    pvec[26] = 1.0 + 0.577 * d.KC
    pvec[27] = d.vref
    return nothing
end

fill_pvec!(pvec::AbstractArray, d::EXAC1)  = _fill_ac1_pvec!(pvec, d, d.VRMAX, d.VRMIN, d.VRMAX, d.VRMIN)
fill_pvec!(pvec::AbstractArray, d::ESAC1A) = _fill_ac1_pvec!(pvec, d, d.VAMAX, d.VAMIN, d.VRMAX, d.VRMIN)

# =============================================================================
# SCRX — bus- or solid-fed SCR bridge excitation system (PSS/E SCRX)
# =============================================================================
#
# PSS/E record order (8 values after BUS / 'SCRX' / ID), derived from the
# ACTIVSg2000 data (all 5 records have exactly 8 values):
#   TA/TB TB K TE EMIN EMAX CSWITCH rc/rfd
#
# Census over the 5 SCRX records (all identical):
#   TA/TB 0.1   TB 10.01   K 100   TE 0 (all)   EMIN -4   EMAX 5
#   CSWITCH 0 (all: bus-fed)   rc/rfd 0 (all)
# Plausible: a high-gain static exciter with a 10 s transient gain reduction
# (lag-lead, TA/TB = 0.1) and an EMAX of 5 pu.
#
# MODEL. States: 2 diff, 0 alg, 0 ctrl.
#   err  = vref − vt + vs            (SCRX has no voltage transducer lag)
#   x[0] = x1    x1'   = ((1 − TA/TB)·err − x1)/TB     (lead-lag (1+sTA)/(1+sTB))
#   x[1] = e_fd  e_fd' = (K·VT0·(x1 + (TA/TB)·err) − e_fd)/TE
#
# pvec layout (10): TA/TB TB K TE EMIN EMAX CSWITCH rc/rfd vref VT0

const SCRX_TMIN = 1.0e-3

mutable struct SCRX <: AbstractStdExciter
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    bus::Int64
    id::String
    TA_TB::Float64
    TB::Float64
    K::Float64
    TE::Float64
    EMIN::Float64
    EMAX::Float64
    CSWITCH::Float64
    RCRFD::Float64
    vref::Float64
    vt0::Float64
    efd0::Float64      # scratch: matched Genrou's post-PF e_fd0
end

SCRX(bus, id, TA_TB, TB, K, TE, EMIN, EMAX, CSWITCH, RCRFD) =
    SCRX(2, 0, 0, 10, bus, id, TA_TB, TB, K, TE, EMIN, EMAX, CSWITCH, RCRFD,
         0.0, 0.0, 0.0)

function from_data_fields(::Type{SCRX}, fields::Vector{SubString{String}})
    v(i) = parse(Float64, fields[i])
    SCRX(parse(Int64, fields[1]), String(fields[3]), (v(i) for i in 4:11)...)
end

get_device_name(::SCRX) = "SCRX"

function fill_pvec!(pvec::AbstractArray, d::SCRX)
    pvec[1]  = d.TA_TB
    pvec[2]  = max(d.TB, SCRX_TMIN)
    pvec[3]  = d.K == 0.0 ? 1.0 : d.K      # K = 0 would make init impossible
    pvec[4]  = max(d.TE, SCRX_TMIN)
    pvec[5]  = d.EMIN            # parsed, not applied
    pvec[6]  = d.EMAX            # parsed, not applied
    pvec[7]  = d.CSWITCH
    pvec[8]  = d.RCRFD           # parsed, not applied
    pvec[9]  = d.vref
    pvec[10] = d.vt0
    return nothing
end

function initial_guess!(x0::AbstractArray, pvec::AbstractArray,
                        pg::Float64, qg::Float64, vm::Float64, va::Float64,
                        d::SCRX)
    a = pvec[1]; K = pvec[3]
    vt0  = pvec[7] == 0.0 ? vm : 1.0     # CSWITCH = 0: bus-fed
    e0   = d.efd0
    vref = vm + e0 / (K * vt0)
    # Rebuild everything with the residual's own expressions so the init
    # residual is exactly zero (the TE row is scaled by 1/TE = 1e3).
    err = vref - vm + 0.0
    x1  = (1.0 - a) * err
    x0[1] = x1
    x0[2] = K * vt0 * (x1 + a * err)
    pvec[9]  = vref
    pvec[10] = vt0
    d.vref = vref
    d.vt0  = vt0
    return nothing
end

function initialize_dynamics!(f::AbstractArray, x::AbstractArray, pvec::AbstractArray,
                              pg::Float64, qg::Float64, vm::Float64, va::Float64,
                              d::SCRX)
    a = pvec[1]; TB = pvec[2]; K = pvec[3]; TE = pvec[4]
    vref = pvec[9]; vt0 = pvec[10]
    x1 = x[1]; efd = x[2]
    err = vref - vm + 0.0
    f[1] = ((1.0 - a) * err - x1) / TB
    f[2] = (K * vt0 * (x1 + a * err) - efd) / TE
    return nothing
end

function extract_init_params!(d::SCRX, sol_zero::AbstractArray, p::AbstractArray, par_ptr::Int)
    p[par_ptr + 8] = d.vref        # slot 9
    p[par_ptr + 9] = d.vt0         # slot 10
    return nothing
end

# =============================================================================
# ESAC6A — IEEE 421.5-2005 Type AC6A alternator-rectifier exciter (PSS/E ESAC6A)
# =============================================================================
#
# PSS/E record order (23 values after BUS / 'ESAC6A' / ID), derived from the
# ACTIVSg2000 data (all 7 records have exactly 23 values):
#   TR KA TA TK TB TC VAMAX VAMIN VRMAX VRMIN TE VFELIM KH VHMAX TH TJ KC KD KE
#   E1 SE(E1) E2 SE(E2)
#
# Census over the 7 ESAC6A records (min / max / #zeros):
#   TR 0/0.05/2   KA 448/585/0   TA 10.3/23.7/0   TK 3/5/0   TB 0.125/0.23/0
#   TC 0.9/0.9/0  VAMAX 9.08/9.08/0  VAMIN -9.08/-9.08/0  VRMAX 9.08  VRMIN -9.08
#   TE 0.5/1/0    VFELIM 0/100/6  KH 0/0/7  VHMAX 1/100/0  TH 0/1/5  TJ 0/0.25/5
#   KC 0/0.225/5  KD 0/22.5/5  KE 1/1/0  E1 4.60/5.84/0  SE1 0.0156/0.142/0
#   E2 6.14/7.78/0  SE2 0.104/0.945/0
# Plausible for AC6A (high-gain lag-lead regulator, KA ~ 500 with TA ~ 15 s and
# TK ~ 4 s) EXCEPT KD = 22.5 in 2 records (buses 4030, 6110): typical KD is
# 0.1-2. It is used as given, but those two exciters need VR0 ≈ 24·e_fd0 at
# equilibrium, well above VAMAX = 9.08 (a limit this package does not apply);
# the value looks like a data-entry error (2.25?).
#
# MODEL. States: 4 diff, 0 alg, 0 ctrl.
#   u    = vref − vc + vs
#   y    = xk + (TK/TA)·(KA·u − xk)          (KA(1+sTK)/(1+sTA))
#   VR   = xll + (TC/TB)·(y − xll)          ((1+sTC)/(1+sTB))
#   VE   = kfex·e_fd ;  VFE = KE·VE + VX(VE) + KD·e_fd
#   x[0] = vc    vc'   = (vt − vc)/TR
#   x[1] = xk    xk'   = (KA·u − xk)/TA
#   x[2] = xll   xll'  = (y − xll)/TB
#   x[3] = e_fd  e_fd' = (VR − VFE)/(TE·kfex)
#
#
# pvec layout (27):
#   1 TR 2 KA 3 TA 4 TK 5 TB 6 TC_eff 7 VAMAX 8 VAMIN 9 VRMAX 10 VRMIN 11 TE
#  12 VFELIM 13 KH 14 VHMAX 15 TH 16 TJ 17 KC 18 KD 19 KE 20 E1 21 SE1 22 E2
#  23 SE2 24 A 25 B 26 kfex 27 vref

const ESAC6A_TMIN = 1.0e-3

mutable struct ESAC6A <: AbstractStdExciter
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    bus::Int64
    id::String
    TR::Float64
    KA::Float64
    TA::Float64
    TK::Float64
    TB::Float64
    TC::Float64
    VAMAX::Float64
    VAMIN::Float64
    VRMAX::Float64
    VRMIN::Float64
    TE::Float64
    VFELIM::Float64
    KH::Float64
    VHMAX::Float64
    TH::Float64
    TJ::Float64
    KC::Float64
    KD::Float64
    KE::Float64
    E1::Float64
    SE1::Float64
    E2::Float64
    SE2::Float64
    sat_a::Float64
    sat_b::Float64
    vref::Float64
    efd0::Float64      # scratch: matched Genrou's post-PF e_fd0
end

function ESAC6A(bus, id, TR, KA, TA, TK, TB, TC, VAMAX, VAMIN, VRMAX, VRMIN, TE,
                VFELIM, KH, VHMAX, TH, TJ, KC, KD, KE, E1, SE1, E2, SE2)
    sat_a, sat_b = _esdc1a_sat_coefficients(Float64(E1), Float64(SE1),
                                            Float64(E2), Float64(SE2))
    ESAC6A(4, 0, 0, 27, bus, id, TR, KA, TA, TK, TB, TC, VAMAX, VAMIN, VRMAX,
           VRMIN, TE, VFELIM, KH, VHMAX, TH, TJ, KC, KD, KE, E1, SE1, E2, SE2,
           sat_a, sat_b, 0.0, 0.0)
end

function from_data_fields(::Type{ESAC6A}, fields::Vector{SubString{String}})
    v(i) = parse(Float64, fields[i])
    d = ESAC6A(parse(Int64, fields[1]), String(fields[3]), (v(i) for i in 4:26)...)
    d.KH != 0.0 && @warn "ESAC6A at bus $(d.bus) id $(d.id): KH = $(d.KH) ≠ 0, but the " *
        "VH field-current-limiter path is not modelled (treated as KH = 0)"
    return _ac_kc_check("ESAC6A", d)
end

get_device_name(::ESAC6A) = "ESAC6A"

function fill_pvec!(pvec::AbstractArray, d::ESAC6A)
    TB = max(d.TB, ESAC6A_TMIN)
    pvec[1]  = max(d.TR, ESAC6A_TMIN)
    pvec[2]  = d.KA == 0.0 ? 1.0 : d.KA
    pvec[3]  = max(d.TA, ESAC6A_TMIN)
    pvec[4]  = d.TK
    pvec[5]  = TB
    pvec[6]  = d.TB <= 0.0 ? TB : d.TC       # TB = 0: lead-lag bypassed
    pvec[7]  = d.VAMAX           # parsed, not applied
    pvec[8]  = d.VAMIN           # parsed, not applied
    pvec[9]  = d.VRMAX           # parsed, not applied
    pvec[10] = d.VRMIN           # parsed, not applied
    pvec[11] = max(d.TE, ESAC6A_TMIN)
    pvec[12] = d.VFELIM          # parsed, not applied (VH path)
    pvec[13] = d.KH              # parsed, not applied (VH path)
    pvec[14] = d.VHMAX           # parsed, not applied (VH path)
    pvec[15] = d.TH              # parsed, not applied (VH path)
    pvec[16] = d.TJ              # parsed, not applied (VH path)
    pvec[17] = d.KC              # enters only through kfex (slot 26)
    pvec[18] = d.KD
    pvec[19] = d.KE
    pvec[20] = d.E1
    pvec[21] = d.SE1
    pvec[22] = d.E2
    pvec[23] = d.SE2
    pvec[24] = d.sat_a
    pvec[25] = d.sat_b
    pvec[26] = 1.0 + 0.577 * d.KC
    pvec[27] = d.vref
    return nothing
end

function initial_guess!(x0::AbstractArray, pvec::AbstractArray,
                        pg::Float64, qg::Float64, vm::Float64, va::Float64,
                        d::ESAC6A)
    KA = pvec[2]; KD = pvec[18]; KE = pvec[19]; A = pvec[24]; B = pvec[25]
    kfex = pvec[26]
    e0   = d.efd0
    vfe0 = _exac2_vfe(e0, KE, KD, A, B, kfex)
    vref = vm + vfe0 / KA               # VR0 = VFE0; both lead-lags unity DC gain
    u    = vref - vm + 0.0              # the residual's own expression (exact rows)
    x0[1] = vm
    x0[2] = KA * u
    x0[3] = KA * u
    x0[4] = e0
    pvec[27] = vref
    d.vref   = vref
    return nothing
end

function initialize_dynamics!(f::AbstractArray, x::AbstractArray, pvec::AbstractArray,
                              pg::Float64, qg::Float64, vm::Float64, va::Float64,
                              d::ESAC6A)
    TR = pvec[1]; KA = pvec[2]; TA = pvec[3]; TK = pvec[4]; TB = pvec[5]
    TC = pvec[6]; TE = pvec[11]; KD = pvec[18]; KE = pvec[19]
    A = pvec[24]; B = pvec[25]; kfex = pvec[26]; vref = pvec[27]
    vc = x[1]; xk = x[2]; xll = x[3]; efd = x[4]
    u   = vref - vc + 0.0
    y   = xk + (TK / TA) * (KA * u - xk)
    vr  = xll + (TC / TB) * (y - xll)
    vfe = _exac2_vfe(efd, KE, KD, A, B, kfex)
    f[1] = (vm - vc) / TR
    f[2] = (KA * u - xk) / TA
    f[3] = (y - xll) / TB
    f[4] = (vr - vfe) / (TE * kfex)
    return nothing
end

function extract_init_params!(d::ESAC6A, sol_zero::AbstractArray, p::AbstractArray, par_ptr::Int)
    p[par_ptr + 26] = d.vref       # slot 27
    return nothing
end
