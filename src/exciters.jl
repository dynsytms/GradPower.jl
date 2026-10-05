abstract type AbstractExciterType <: AbstractGenControlType end

mutable struct ESDC1A <: AbstractExciterType
    # representation
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    # topology
    bus::Int64
    id::String
    # parameters
    Ka::Float64
    Ta::Float64
    Kf::Float64
    Tf::Float64
    Ke::Float64
    Te::Float64
    Tr::Float64
    Ae::Float64
    Be::Float64
    # runtime fields set during initialization
    vref::Float64
end

function ESDC1A(bus, id, Ka, Ta, Kf, Tf, Ke, Te, Tr, Ae, Be)
    exciter = ESDC1A(3, 0, 0, 10, bus, id, Ka, Ta, Kf, Tf, Ke, Te, Tr, Ae, Be, 0.0)
    return exciter
end

"""
    _esdc1a_sat_coefficients(E1, SE1, E2, SE2) -> (sat_a, sat_b)

Quadratic saturation: Se(e_fd) = sat_b * (e_fd - sat_a)^2 for e_fd > sat_a,
zero otherwise. Coefficients chosen so Se(E1)=SE1·E1, Se(E2)=SE2·E2.
Returns (0.0, 0.0) when any of the inputs is non-positive (no saturation).
Returns (0.0, 0.0) when saturation data is absent.
"""
function _esdc1a_sat_coefficients(E1::Float64, SE1::Float64, E2::Float64, SE2::Float64)
    if E1 <= 0.0 || E2 <= 0.0 || SE1 <= 0.0 || SE2 <= 0.0
        return 0.0, 0.0
    end
    a = sqrt(SE1*E1 / (SE2*E2))
    if a == 1.0
        return 0.0, 0.0
    end
    sat_a = E2 - (E1 - E2)/(a - 1.0)
    sat_b = SE2*E2 * (a - 1.0)^2 / (E1 - E2)^2
    return sat_a, sat_b
end

function from_data_fields(::Type{ESDC1A}, fields::Vector{SubString{String}})
    bus = parse(Int64, fields[1])
    id = String(fields[3])

    # PSS/E ESDC1A record fields (after bus, type, id):
    #   Tr, Ka, Ta, Tb, Tc, Vrmax, Vrmin, Ke, Te, Kf, Tf, Sw, E1, SE1, E2, SE2
    # The reduced model in this kernel ignores Tb, Tc, Vrmax, Vrmin, Sw and
    # the Tr first-order filter on vm. Saturation is the quadratic form with
    # sat_a/sat_b precomputed from (E1, SE1, E2, SE2) — stored in the
    # struct's Ae/Be slots (legacy field names).
    Tr  = parse(Float64, fields[4])
    Ka  = parse(Float64, fields[5])
    Ta  = parse(Float64, fields[6])
    Ke  = parse(Float64, fields[11])
    Te  = parse(Float64, fields[12])
    Kf  = parse(Float64, fields[13])
    Tf  = parse(Float64, fields[14])
    E1  = parse(Float64, fields[16])
    SE1 = parse(Float64, fields[17])
    E2  = parse(Float64, fields[18])
    SE2 = parse(Float64, fields[19])
    sat_a, sat_b = _esdc1a_sat_coefficients(E1, SE1, E2, SE2)
    ESDC1A(bus, id, Ka, Ta, Kf, Tf, Ke, Te, Tr, sat_a, sat_b)
end

function fill_pvec!(pvec::AbstractArray, dtype::ESDC1A)
    pvec[1] = dtype.Ka
    pvec[2] = dtype.Ta
    pvec[3] = dtype.Kf
    pvec[4] = dtype.Tf
    pvec[5] = dtype.Ke
    pvec[6] = dtype.Te
    pvec[7] = dtype.Tr
    pvec[8] = dtype.Ae
    pvec[9] = dtype.Be
    pvec[10] = dtype.vref
end

function init_exciter!(
        xdiff::AbstractArray,
        pvec::AbstractArray,
        e_fd0::Float64,
        vm::Float64,
        dtype::ESDC1A
)
    Ka = pvec[1]
    Kf = pvec[3]
    Tf = pvec[4]
    Ke = pvec[5]
    sat_a = pvec[8]
    sat_b = pvec[9]

    sat = (sat_b == 0.0 || e_fd0 <= sat_a) ? 0.0 : sat_b * (e_fd0 - sat_a)^2
    vr1 = Ke*e_fd0 + sat
    vr2 = -(Kf/Tf)*e_fd0
    vref = vm + vr1/Ka

    xdiff[1] = vr1
    xdiff[2] = vr2
    xdiff[3] = e_fd0
    dtype.vref = vref
    return nothing
end

# Standard initialization-path hooks (called by `initialize_device`).
# These mirror SEXS's flow: `initial_guess!` writes both the state
# guesses and `dtype.vref` (the init-derived parameter); the residual
# returned by `initialize_dynamics!` is identically zero at the guess
# so nlsolve converges in 0 iterations. `extract_init_params!`
# (defined in src/dynamics.jl) mirrors the converged vref into the
# parameter vector.
function initial_guess!(
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::ESDC1A
)
    Ka = pvec[1]
    Kf = pvec[3]
    Tf = pvec[4]
    Ke = pvec[5]
    sat_a = pvec[8]
    sat_b = pvec[9]
    # pre-init: `dtype.vref` carries the matched Genrou's post-PF e_fd0,
    # stashed by initialize_dynamics! before this controller is initialized.
    e_fd0 = dtype.vref

    sat = (sat_b == 0.0 || e_fd0 <= sat_a) ? 0.0 : sat_b * (e_fd0 - sat_a)^2
    vr1 = Ke*e_fd0 + sat
    vr2 = -(Kf/Tf)*e_fd0
    vref = vm + vr1/Ka

    x0[1] = vr1
    x0[2] = vr2
    x0[3] = e_fd0
    # Mirror vref into pvec slot 10 now so `initialize_dynamics!` sees it.
    pvec[10] = vref
    # Store vref on the struct so `extract_init_params!(::ESDC1A)` and
    # `refresh_esdc1a_table!` can pick it up.
    dtype.vref = vref
    return nothing
end

function initialize_dynamics!(
        f::AbstractArray,
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::ESDC1A
)
    Ka = pvec[1]
    Ta = pvec[2]
    Kf = pvec[3]
    Tf = pvec[4]
    Ke = pvec[5]
    Te = pvec[6]
    sat_a = pvec[8]
    sat_b = pvec[9]
    vref = pvec[10]

    vr1  = x0[1]
    vr2  = x0[2]
    e_fd = x0[3]

    sat = (sat_b == 0.0 || e_fd <= sat_a) ? 0.0 : sat_b * (e_fd - sat_a)^2
    f[1] = (Ka*(vref - vm - vr2 - (Kf/Tf)*e_fd) - vr1) / Ta
    f[2] = -((Kf/Tf)*e_fd + vr2) / Tf
    f[3] = (vr1 - Ke*e_fd - sat) / Te
    return nothing
end

function cinject!(
        f::AbstractArray,
        x::AbstractArray,
        y::AbstractArray,
        u::AbstractArray,
        p::AbstractArray,
        v::AbstractArray,
        dtype::ESDC1A
)
    return nothing
end

function rhs_fun!(
        f_diff::AbstractArray,
        f_alg::AbstractArray,
        x::AbstractArray,
        y::AbstractArray,
        u::AbstractArray,
        p::AbstractArray,
        v::AbstractArray,
        dtype::ESDC1A
)
    Ka = p[1]
    Ta = p[2]
    Kf = p[3]
    Tf = p[4]
    Ke = p[5]
    Te = p[6]
    sat_a = p[8]
    sat_b = p[9]
    vref = p[10]

    vr1 = x[1]
    vr2 = x[2]
    e_fd = x[3]
    vm = hypot(v[1], v[2])
    sat = (sat_b == 0.0 || e_fd <= sat_a) ? 0.0 : sat_b * (e_fd - sat_a)^2

    f_diff[1] = (Ka*(vref - vm - vr2 - (Kf/Tf)*e_fd) - vr1)/Ta
    f_diff[2] = -((Kf/Tf)*e_fd + vr2)/Tf
    f_diff[3] = (vr1 - Ke*e_fd - sat)/Te
end

function preallocate_jacobian!(
    coord_list::Vector{Vector{Int}},
    diff_ptr::Int,
    alg_ptr::Int,
    ctrl_ptr::Int,
    volt_ptr::Int,
    dtype::ESDC1A
)
    dp = diff_ptr
    vp = volt_ptr

    vr1 = dp
    vr2 = dp + 1
    e_fd = dp + 2
    vr = vp
    vi = vp + 1

    append!(coord_list[dp], [vr1, vr2, e_fd, vr, vi])
    append!(coord_list[dp + 1], [vr2, e_fd])
    append!(coord_list[dp + 2], [vr1, e_fd, vr, vi])
end

function rhs_jac!(
    jac::AbstractMatrix,
    x::AbstractArray,
    y::AbstractArray,
    u::AbstractArray,
    p::AbstractArray,
    v::AbstractArray,
    idx_dev::Vector{Int},
    dtype::ESDC1A
)
    dp = idx_dev[1]
    dev = idx_dev[3]
    bus = idx_dev[5]

    Ka = p[1]
    Ta = p[2]
    Kf = p[3]
    Tf = p[4]
    Ke = p[5]
    Te = p[6]
    sat_a = p[8]
    sat_b = p[9]

    e_fd = x[3]
    vr = v[1]
    vi = v[2]
    vm = hypot(vr, vi)
    dvm_dvr = vr/vm
    dvm_dvi = vi/vm

    vr1_idx = dp
    vr2_idx = dp + 1
    e_fd_idx = dp + 2
    vr_idx = dev + 2*(bus - 1) + 1
    vi_idx = vr_idx + 1

    row = dp
    jac[row, vr1_idx] = -1.0/Ta
    jac[row, vr2_idx] = -Ka/Ta
    jac[row, e_fd_idx] = -Ka*Kf/(Ta*Tf)
    jac[row, vr_idx] = -Ka/Ta*dvm_dvr
    jac[row, vi_idx] = -Ka/Ta*dvm_dvi

    row = dp + 1
    jac[row, vr2_idx] = -1.0/Tf
    jac[row, e_fd_idx] = -Kf/(Tf^2)

    row = dp + 2
    dsat = (sat_b == 0.0 || e_fd <= sat_a) ? 0.0 : 2.0*sat_b*(e_fd - sat_a)
    jac[row, vr1_idx] = 1.0/Te
    jac[row, e_fd_idx] = -(Ke + dsat)/Te
end

# ===========================
# SEXS — Simplified Excitation System
# ===========================
#
# States: 2 diff (x1, e_fd). No alg.
# Residual (vref is initialization-derived parameter):
#   F[dp+0] = (-x1 + (1 - TA_TB)·(vref - vm)) / TB
#   F[dp+1] = (-e_fd + K·(x1 + TA_TB·(vref - vm))) / TE
# Init: vref = e_fd0/K + vm,  x1 = (1 - TA_TB)·(vref - vm)

mutable struct SEXS <: AbstractExciterType
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
    vref::Float64
end

function SEXS(bus, id, TA_TB, TB, K, TE, EMIN, EMAX)
    # ctrl_size=1 reserves a slot for the init-only `vref` unknown; not
    # wired at runtime (kernel reads vr/vi directly via table.vr_idx).
    SEXS(2, 0, 1, 7, bus, id, TA_TB, TB, K, TE, EMIN, EMAX, 0.0)
end

function from_data_fields(::Type{SEXS}, fields::Vector{SubString{String}})
    bus = parse(Int64, fields[1])
    id = String(fields[3])
    TA_TB = parse(Float64, fields[4])
    TB = parse(Float64, fields[5])
    K = parse(Float64, fields[6])
    TE = parse(Float64, fields[7])
    EMIN = parse(Float64, fields[8])
    EMAX = parse(Float64, fields[9])
    SEXS(bus, id, TA_TB, TB, K, TE, EMIN, EMAX)
end

function fill_pvec!(pvec::AbstractArray, dtype::SEXS)
    pvec[1] = dtype.TA_TB
    pvec[2] = dtype.TB
    pvec[3] = dtype.K
    pvec[4] = dtype.TE
    pvec[5] = dtype.EMIN
    pvec[6] = dtype.EMAX
    pvec[7] = dtype.vref
end

function get_device_name(dtype::SEXS)
    return "SEXS"
end

function initialize_dynamics!(
        f::AbstractArray,
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::SEXS
)
    TA_TB = pvec[1]
    TB = pvec[2]
    K = pvec[3]
    TE = pvec[4]

    # Unknowns at init: 2 diff (x1, e_fd) + 1 init unknown (vref).
    # (No alg states; ctrl_size=1 reserves the vref slot in xinit.)
    x1   = x0[1]
    e_fd = x0[2]
    vref = x0[3]

    # SS residual: w drops; vm is the steady PF magnitude.
    f[1] = (-x1 + (1.0 - TA_TB)*(vref - vm)) / TB
    f[2] = (-e_fd + K*(x1 + TA_TB*(vref - vm))) / TE
    # Init constraint: e_fd must match the matched Genrou's post-PF e_fd0
    # (stashed in dtype.vref by `set_dynamics!` before this call).
    f[3] = e_fd - dtype.vref
    return nothing
end

function initial_guess!(
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::SEXS
    )
    TA_TB = pvec[1]
    K = pvec[3]
    e_fd0 = dtype.vref   # pre-init: holds e_fd0; post-init: holds vref
    vref_guess = e_fd0/K + vm
    x0[1] = (1.0 - TA_TB)*(vref_guess - vm)
    x0[2] = e_fd0
    x0[3] = vref_guess
    return nothing
end

# ===========================
# ESST4B — IEEE Type ST4B static (bus-fed potential-source) excitation system
# ===========================
#
# PSS/E record order (17 values after BUS / 'ESST4B' / ID):
#   TR KPR KIR VRMAX VRMIN TA KPM KIM VMMAX VMMIN KG KP KI VBMAX KC XL THETAP
#
# ---------------------------------------------------------------------------
# WHAT IS MODELLED
# ---------------------------------------------------------------------------
# States: 4 diff, 0 alg, 0 ctrl.
#   x[0] = vc    sensed terminal voltage    vc'  = (vt - vc) / TR
#   x[1] = xr    outer PI integrator        xr'  = KIR * err
#   x[2] = xm    inner PI integrator        xm'  = KIM * (vr_pi - KG*e_fd)
#   x[3] = e_fd  field voltage              e_fd'= (vm_pi*VB0 - e_fd) / TA
# with
#   err   = vref - vc + vs              (vs = PSS output, 0 when no PSS)
#   vr_pi = xr + KPR*err                (outer PI output, "VA")
#   vm_pi = xm + KPM*(vr_pi - KG*e_fd)  (inner PI output, "VM")
#   VB0   = KP * |Vt(0)|                (bridge voltage FROZEN at its
#                                        initialization value — see (2))
#
# Two init-derived parameters: vref (pvec slot 18) and VB0 (pvec slot 19).
#
# ---------------------------------------------------------------------------
# DELIBERATE SIMPLIFICATIONS — read this before trusting the model
# ---------------------------------------------------------------------------
# 1. LIMITERS ARE PARSED BUT NOT APPLIED. VRMAX/VRMIN (outer PI), VMMAX/VMMIN
#    (inner PI) and VBMAX (bridge) are stored on the struct and in the SoA
#    table, and are never used by the residual or the Jacobian. This is the
#    package-wide convention (TGOV1 stores VMAX/VMIN unused; SEXS ignores
#    EMIN/EMAX). The unsaturated linear system is what is integrated.
#
# 2. THE POTENTIAL-SOURCE BRIDGE VOLTAGE VB IS A CONSTANT GAIN, FROZEN AT
#    ITS INITIALIZATION VALUE  VB0 = KP·|Vt(0)| .
#    The full ST4B bridge is
#        VE = |KP∠THETAP · Vt + j(KI + KP·XL) · It|
#        IN = KC · IFD / VE,   VB = VE · FEX(IN),  VB ≤ VBMAX
#    and it varies with the terminal voltage and field current. Here it is
#    evaluated ONCE, at the power-flow operating point, with:
#      (a) no current compounding (KI, XL, THETAP ignored). In ACTIVSg2000
#          all 278 ESST4B records have KI = XL = THETAP = 0, so
#          VE = KP·|Vt| exactly at t = 0 — no approximation for this data.
#      (b) no rectifier loading, FEX ≡ 1 (KC ignored). FEX needs the machine
#          field current IFD, which Genrou does not expose, and is piecewise
#          (non-differentiable). For ACTIVSg2000, KC ∈ [0.040, 0.115] and
#          KP ∈ [1.0, 7.6], so IN ≈ KC·IFD/(KP·|Vt|) ≈ 0.02 for IFD ~ 1-2 pu,
#          giving FEX ≈ 0.99 — a ~1% static gain error.
#    WHAT IS LOST: the time variation of VB. A real ST4B is fed from the
#    generator terminal, so its ceiling collapses with |Vt| during a nearby
#    fault. This model keeps full field-forcing capability through a fault.
#
#    WHY VB IS NOT LEFT LIVE (VB = KP·|Vt(t)|). That form was implemented and
#    tested first; it is smooth and its analytic Jacobian matched finite
#    differences. But |Vt| then feeds e_fd twice with opposite signs: through
#    the regulator (gain −KPM·KPR·KP·|Vt|, stabilizing) and directly through
#    the bridge (gain +e_fd/|Vt|, destabilizing). When
#        e_fd0 / |Vt| > KPM·KPR·KP·|Vt|
#    the direct path wins and the loop is locally unstable. The real device
#    is saved by VMMAX/VBMAX, which this package does not apply (point 1).
#    In ACTIVSg2000, 3 of the 212 active ESST4B machines are in that regime
#    (the worst has e_fd0 = 52 pu); with a live VB their e_fd grows without
#    bound (to ~1e33 within 0.8 s of a bus fault) and the Newton solve fails
#    with a singular Jacobian, even though the other 209 are well damped.
#    Freezing VB removes the destabilizing path for every device, uniformly,
#    and without a data-dependent switch in the kernel.
#
# 3. TA IS APPLIED AS A LAG ON e_fd RATHER THAN ON THE INNER PI OUTPUT.
#    IEEE 421.5 ST4B places 1/(1+sTA) between the inner PI and the ×VB
#    multiplier. Here the lag is placed after the multiplier so that e_fd is
#    itself a differential state — required because the coupling graph feeds
#    Genrou's e_fd ctrl slot from a diff state (`source_kind = :diff_at`).
#    With VB constant (point 2) the two orderings are EXACTLY equivalent.
#
# ---------------------------------------------------------------------------
# ZERO TIME CONSTANTS
# ---------------------------------------------------------------------------
# In ACTIVSg2000 TR = 0 and TA = 0 in ALL 278 records. A zero-T stage is an
# algebraic pass-through. As with IEEEG1, it is handled by FLOORING the time
# constant at ESST4B_TMIN in `fill_pvec!` and in the SoA table builder, never
# by branching inside the kernel — the residual/Jacobian must stay
# branch-free so the same `_one!` leaves run on the GPU. Backward Euler is
# L-stable, so a lag far faster than dt settles to the algebraic limit
# without ringing.
#
# KP is likewise guarded (`_esst4b_kp_eff`): VB0 = KP·|Vt| would be zero for
# KP = 0, leaving the exciter unable to produce any field voltage and the
# initialization unsolvable. KP = 0 falls back to unity. All ACTIVSg2000
# records have KP ∈ [1.0, 7.6].
#
# KIM = 0 in all 278 records, i.e. the inner loop is pure gain KPM. That is
# NOT floored: xm then simply holds the steady-state bias e_fd0/VB0 while the
# outer PI does the regulating through the KPM path. `initial_guess!` is
# constructed so the initialization residual is exactly zero in IEEE754, so
# nlsolve converges in zero iterations and never has to factor the (then
# rank-deficient) initialization Jacobian.

const ESST4B_TMIN = 1.0e-3

# ---------------------------------------------------------------------------
# PSS INPUT (v_s): WIRED BY DEFAULT (flag below; set false to opt out).
# ---------------------------------------------------------------------------
# The residual/Jacobian support a PSS input through the table's `vs_idx`
# column (err = vref - vc + vs), and those Jacobian slots are
# finite-difference verified. `fix_ieeest_wiring!` connects an IEEEST to
# an ESST4B only when this flag is true.
#
# The flag once defaulted to false: wired PSS made ACTIVSg2000 diverge until
# the IEEEST output limit and zero-denominator floor were fixed
# (src/stabilizers.jl). Set
# `GradPower.ESST4B_WIRE_PSS[] = false` BEFORE `from_psse` to opt out;
# IEEEST devices are then still built and integrated, only their v_s is
# not routed into the ESST4B.
const ESST4B_WIRE_PSS = Ref(true)

mutable struct ESST4B <: AbstractExciterType
    # representation
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    # topology
    bus::Int64
    id::String
    # parameters (PSS/E order)
    TR::Float64
    KPR::Float64
    KIR::Float64
    VRMAX::Float64
    VRMIN::Float64
    TA::Float64
    KPM::Float64
    KIM::Float64
    VMMAX::Float64
    VMMIN::Float64
    KG::Float64
    KP::Float64
    KI::Float64
    VBMAX::Float64
    KC::Float64
    XL::Float64
    THETAP::Float64
    # initialization-derived; mirrored into pvec slot 18.
    # Also used as SCRATCH before the controller solve: `initialize_dynamics!`
    # (the driver in src/dynamics.jl) stashes the matched Genrou's post-PF
    # e_fd0 here, exactly as it does for SEXS/ESDC1A.
    vref::Float64
    # initialization-derived frozen bridge voltage KP·|Vt(0)|; pvec slot 19.
    vb0::Float64
end

function ESST4B(bus, id, TR, KPR, KIR, VRMAX, VRMIN, TA, KPM, KIM,
                VMMAX, VMMIN, KG, KP, KI, VBMAX, KC, XL, THETAP)
    # 4 diff + 0 alg + 0 ctrl ; 17 parsed params + vref + vb0 = 19
    ESST4B(4, 0, 0, 19, bus, id, TR, KPR, KIR, VRMAX, VRMIN, TA, KPM, KIM,
           VMMAX, VMMIN, KG, KP, KI, VBMAX, KC, XL, THETAP, 0.0, 0.0)
end

# Effective potential-source gain. KP = 0 would make VB0 ≡ 0 (no field
# voltage possible, singular initialization); fall back to unity gain.
# Applied OUTSIDE the kernel (fill_pvec! + table builder).
@inline _esst4b_kp_eff(KP::Float64) = KP <= 0.0 ? 1.0 : KP

function from_data_fields(::Type{ESST4B}, fields::Vector{SubString{String}})
    bus = parse(Int64, fields[1])
    id  = String(fields[3])
    v(i) = parse(Float64, fields[i])
    # fields[4..20] = TR KPR KIR VRMAX VRMIN TA KPM KIM VMMAX VMMIN
    #                 KG KP KI VBMAX KC XL THETAP
    ESST4B(bus, id,
           v(4),  v(5),  v(6),  v(7),  v(8),  v(9),  v(10), v(11),
           v(12), v(13), v(14), v(15), v(16), v(17), v(18), v(19), v(20))
end

# TR/TA are stored FLOORED at ESST4B_TMIN and KP guarded, so every consumer
# (residual, Jacobian, initialization) sees the same effective parameter and
# no kernel has to branch.
function fill_pvec!(pvec::AbstractArray, dtype::ESST4B)
    pvec[1]  = max(dtype.TR, ESST4B_TMIN)
    pvec[2]  = dtype.KPR
    pvec[3]  = dtype.KIR
    pvec[4]  = dtype.VRMAX      # parsed, not applied
    pvec[5]  = dtype.VRMIN      # parsed, not applied
    pvec[6]  = max(dtype.TA, ESST4B_TMIN)
    pvec[7]  = dtype.KPM
    pvec[8]  = dtype.KIM
    pvec[9]  = dtype.VMMAX      # parsed, not applied
    pvec[10] = dtype.VMMIN      # parsed, not applied
    pvec[11] = dtype.KG
    pvec[12] = _esst4b_kp_eff(dtype.KP)
    pvec[13] = dtype.KI         # parsed, not applied (compounding dropped)
    pvec[14] = dtype.VBMAX      # parsed, not applied
    pvec[15] = dtype.KC         # parsed, not applied (FEX ≡ 1)
    pvec[16] = dtype.XL         # parsed, not applied (compounding dropped)
    pvec[17] = dtype.THETAP     # parsed, not applied (compounding dropped)
    pvec[18] = dtype.vref
    pvec[19] = dtype.vb0
end

get_device_name(dtype::ESST4B) = "ESST4B"

# ST4B gains are per-unit on the generator terminal-voltage / field-voltage
# base, not on MVA base, so nothing rescales with mbase/baseMVA. Explicit
# no-op (the AbstractDeviceType fallback would do the same) to document that
# this was considered rather than forgotten.
set_ratio!(dtype::ESST4B, ratio::Float64) = nothing

function initial_guess!(
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::ESST4B
    )
    KG = pvec[11]
    KP = pvec[12]
    # pre-init: `dtype.vref` carries the matched Genrou's post-PF e_fd0,
    # stashed by the initialize_dynamics! driver before this controller runs.
    e_fd0 = dtype.vref

    vb0 = KP * vm          # frozen bridge voltage (simplification (2))
    xm  = e_fd0 / vb0
    # Round-trip e_fd through xm*vb0 so that the residual below is EXACTLY
    # zero in floating point (see the KIM note in the header): with KIM = 0
    # the initialization Jacobian is rank deficient, and nlsolve must be able
    # to declare convergence at iteration 0 without ever factoring it.
    e_fd = xm * vb0
    xr   = KG * e_fd
    vc   = vm
    vref = vm

    x0[1] = vc
    x0[2] = xr
    x0[3] = xm
    x0[4] = e_fd
    # Mirror the init-derived parameters into pvec now so
    # `initialize_dynamics!` sees them.
    pvec[18] = vref
    pvec[19] = vb0
    # Store them on the struct so `refresh_esst4b_table!` can pick them up.
    dtype.vref = vref
    dtype.vb0  = vb0
    return nothing
end

function initialize_dynamics!(
        f::AbstractArray,
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::ESST4B
)
    TR   = pvec[1]
    KPR  = pvec[2]
    KIR  = pvec[3]
    TA   = pvec[6]
    KPM  = pvec[7]
    KIM  = pvec[8]
    KG   = pvec[11]
    vref = pvec[18]
    vb0  = pvec[19]

    vc   = x0[1]
    xr   = x0[2]
    xm   = x0[3]
    e_fd = x0[4]

    # Steady state: terminal voltage is the power-flow magnitude, no PSS
    # contribution (v_s = 0 at equilibrium by construction of IEEEST).
    err   = vref - vc
    vr_pi = xr + KPR * err
    vm_pi = xm + KPM * (vr_pi - KG * e_fd)

    f[1] = (vm - vc) / TR
    f[2] = KIR * err
    f[3] = KIM * (vr_pi - KG * e_fd)
    f[4] = (vm_pi * vb0 - e_fd) / TA
    return nothing
end
