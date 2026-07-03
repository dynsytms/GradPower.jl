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
