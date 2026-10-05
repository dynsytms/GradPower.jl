# IEEEST Power System Stabilizer (PSS/E).
#
# Transfer function (MODE/ICS = 1: input = rotor speed deviation w):
#
#            1 + A5 s + A6 s^2                1 + T1 s   1 + T3 s        T5 s
#   G(s) = ------------------------------ . -------- . -------- . KS . --------
#          (1 + A1 s + A2 s^2)(1 + A3 s + A4 s^2)   1 + T2 s   1 + T4 s      1 + T6 s
#
# followed by the output limit [LSMIN, LSMAX]
#
#   sig = w  (Genrou's w state is the speed deviation)
#
#   F1 = N/D1:   A2 * ds0 = sig - s1 - A1*s0
#                ds1 = s0
#                y1 = s1 + A5*s0 + (A6/A2)*(sig - s1 - A1*s0)   [inline]
#   F2 = 1/D2:   A4 * ds2 = y1 - s3 - A3*s2
#                ds3 = s2
#                y2 = s3                                        [inline]
#   LL1:         T2 * ds4 = y2 - s4
#                y3 = s4 + (T1/T2)*(y2 - s4)                    [inline]
#   LL2:         T4 * ds5 = y3 - s5
#                y4 = s5 + (T3/T4)*(y3 - s5)                    [inline]
#   Washout:     T6 * ds6 = KS*y4 - s6
#                x   = (T5/T6)*(KS*y4 - s6)                     [inline]
#   Limit:       v_s = c + h*tanh(x/h - atanh(c/h))             [alg eq]
#
# VCU/VCL (the terminal-voltage-based output cutoff) are parsed and stored
# but NOT applied: they are zero (= disabled) in every ACTIVSg2000 record.

abstract type AbstractStabilizerType <: AbstractGenControlType end

# Floor for zero time constants (see from_data_fields).
const IEEEST_EPS_TC = 0.001
# Half-width used for "no output limit" (see header / fill_pvec!).
const IEEEST_NOLIMIT = 1.0e3

# Realize a second-order denominator 1 + a*s + b*s^2 with b == 0 (b is a
# divisor in the kernel) as a nearby proper second-order one:
#   a == 0, b == 0 : 1                -> (1 + eps s)^2
#   a != 0, b == 0 : 1 + a s          -> (1 + a s)(1 + eps s)
# Both add only real poles at -1/eps (critically damped, no resonance).
function _ieeest_floor_den(a::Float64, b::Float64)
    eps = IEEEST_EPS_TC
    b != 0.0 && return (a, b)
    a == 0.0 && return (2eps, eps^2)
    return (a + eps, a * eps)
end

# Effective (LSMAX, LSMIN) written into pvec. Any pair that does not
# strictly bracket 0 (incl. the PSS/E "no limit" LSMAX = LSMIN = 0) becomes
# +/-IEEEST_NOLIMIT, so the kernel never branches on it.
function ieeest_effective_limits(LSMAX::Float64, LSMIN::Float64)
    (LSMAX > 0.0 && LSMIN < 0.0) && return (LSMAX, LSMIN)
    if !(LSMAX == 0.0 && LSMIN == 0.0)
        @warn "IEEEST: output limits LSMAX=$LSMAX, LSMIN=$LSMIN do not bracket 0; treating as unlimited" maxlog=1
    end
    return (IEEEST_NOLIMIT, -IEEEST_NOLIMIT)
end

# Smooth output saturation (branch-free; GPU-safe). Returns (v_s, dv_s/dx).
@inline function _ieeest_sat(x, LSMAX, LSMIN)
    c = 0.5 * (LSMAX + LSMIN)
    h = 0.5 * (LSMAX - LSMIN)
    t = tanh(x / h - atanh(c / h))
    return c + h * t, 1.0 - t * t
end

mutable struct IEEEST <: AbstractStabilizerType
    # representation
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    # topology
    bus::Int64
    id::String
    # parameters
    MODE::Int64
    BUSR::Int64
    A1::Float64
    A2::Float64
    A3::Float64
    A4::Float64
    A5::Float64
    A6::Float64
    T1::Float64
    T2::Float64
    T3::Float64
    T4::Float64
    T5::Float64
    T6::Float64
    KS::Float64
    LSMAX::Float64
    LSMIN::Float64
    VCU::Float64
    VCL::Float64
end

function IEEEST(bus, id, MODE, BUSR, A1, A2, A3, A4, A5, A6,
                T1, T2, T3, T4, T5, T6, KS, LSMAX, LSMIN, VCU, VCL)
    # 7 diff, 1 alg (v_s), 1 ctrl (omega), 19 params
    IEEEST(7, 1, 1, 19, bus, id, MODE, BUSR,
           A1, A2, A3, A4, A5, A6, T1, T2, T3, T4, T5, T6, KS,
           LSMAX, LSMIN, VCU, VCL)
end

function from_data_fields(::Type{IEEEST}, fields::Vector{SubString{String}})
    # PSS/E DYR field order:
    # BUS, 'IEEEST', ID, MODE, BUSR, A1, A2, A3, A4, A5, A6,
    # T1, T2, T3, T4, T5, T6, KS, LSMAX, LSMIN, VCU, VCL /
    bus  = parse(Int64, fields[1])
    id   = String(fields[3])
    MODE = parse(Int64, fields[4])
    BUSR = parse(Int64, fields[5])
    A1   = parse(Float64, fields[6])
    A2   = parse(Float64, fields[7])
    A3   = parse(Float64, fields[8])
    A4   = parse(Float64, fields[9])
    A5   = parse(Float64, fields[10])
    A6   = parse(Float64, fields[11])
    T1   = parse(Float64, fields[12])
    T2   = parse(Float64, fields[13])
    T3   = parse(Float64, fields[14])
    T4   = parse(Float64, fields[15])
    T5   = parse(Float64, fields[16])
    T6   = parse(Float64, fields[17])
    KS   = parse(Float64, fields[18])
    LSMAX = parse(Float64, fields[19])
    LSMIN = parse(Float64, fields[20])
    VCU   = parse(Float64, fields[21])
    VCL   = parse(Float64, fields[22])

    # Only MODE=1 (rotor speed deviation) is supported.
    MODE != 1 && error("IEEEST MODE=$MODE not supported; only MODE=1 (rotor speed deviation) is implemented.")

    # Clamp zero time constants T1..T6: a zero pair means "bypass
    # this lead-lag block" (unity gain). We replace zeros with a value large
    # enough to avoid introducing stiffness (1/eps must stay comparable to
    # 1/dt, not 1e6) but small enough to act as a passthrough.
    _EPS_TC = IEEEST_EPS_TC
    # The numerator N(s) = 1 + A5 s + A6 s^2 is realized together with the
    # (A1, A2) denominator (see header). The two denominators commute, so
    # if only the (A3, A4) one has an s^2 term, swap them.
    if A2 == 0.0 && A4 != 0.0
        A1, A2, A3, A4 = A3, A4, A1, A2
    end
    # A2/A4 are divisors: floor a zero as a critically damped fast pole
    # pair, not coefficient-by-coefficient (see _ieeest_floor_den).
    A1, A2 = _ieeest_floor_den(A1, A2)
    A3, A4 = _ieeest_floor_den(A3, A4)
    # A5/A6 are numerator coefficients, never divisors: zeros are kept.
    T1 = T1 == 0.0 ? _EPS_TC : T1
    T2 = T2 == 0.0 ? _EPS_TC : T2
    T3 = T3 == 0.0 ? _EPS_TC : T3
    T4 = T4 == 0.0 ? _EPS_TC : T4
    T5 = T5 == 0.0 ? _EPS_TC : T5
    T6 = T6 == 0.0 ? _EPS_TC : T6

    IEEEST(bus, id, MODE, BUSR, A1, A2, A3, A4, A5, A6,
           T1, T2, T3, T4, T5, T6, KS, LSMAX, LSMIN, VCU, VCL)
end

function fill_pvec!(pvec::AbstractArray, dtype::IEEEST)
    pvec[1]  = dtype.A1
    pvec[2]  = dtype.A2
    pvec[3]  = dtype.A3
    pvec[4]  = dtype.A4
    pvec[5]  = dtype.A5
    pvec[6]  = dtype.A6
    pvec[7]  = dtype.T1
    pvec[8]  = dtype.T2
    pvec[9]  = dtype.T3
    pvec[10] = dtype.T4
    pvec[11] = dtype.T5
    pvec[12] = dtype.T6
    pvec[13] = dtype.KS
    lsmax, lsmin = ieeest_effective_limits(dtype.LSMAX, dtype.LSMIN)
    pvec[14] = lsmax
    pvec[15] = lsmin
    pvec[16] = dtype.VCU
    pvec[17] = dtype.VCL
    pvec[18] = Float64(dtype.MODE)
    pvec[19] = Float64(dtype.BUSR)
end

function get_device_name(dtype::IEEEST)
    return "IEEEST"
end

function initial_guess!(
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::IEEEST
)
    # At steady state, w (speed deviation) = 0, so sig = 0. All blocks
    # have zero input, so all states are zero and v_s = 0.
    # x0 layout: 7 diff + 1 alg + 1 ctrl = 9 slots, all zero.
    fill!(x0, 0.0)
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
        dtype::IEEEST
)
    # At steady state with omega=1, sig=0, all derivatives are zero and v_s=0.
    # Unknowns: x0[1:7] diff, x0[8] alg (v_s), x0[9] ctrl (omega).
    # All should be zero at init (no residual).
    A1 = pvec[1]; A2 = pvec[2]; A3 = pvec[3]; A4 = pvec[4]
    A5 = pvec[5]; A6 = pvec[6]
    T1 = pvec[7]; T2 = pvec[8]; T3 = pvec[9]; T4 = pvec[10]
    T5 = pvec[11]; T6 = pvec[12]; KS = pvec[13]

    s0 = x0[1]; s1 = x0[2]; s2 = x0[3]; s3 = x0[4]
    s4 = x0[5]; s5 = x0[6]; s6 = x0[7]
    vs = x0[8]
    # ctrl slot: w (speed deviation, = omega - 1). At steady state, w = 0.
    w = x0[9]

    sig = w

    # F1 = N/D1 (inline y1)
    q1 = sig - s1 - A1*s0
    f[1] = q1 / A2
    f[2] = s0
    y1 = s1 + A5*s0 + (A6/A2)*q1

    # F2 = 1/D2 (y2 = s3)
    f[3] = (y1 - s3 - A3*s2) / A4
    f[4] = s2
    y2 = s3

    # LL1: LeadLag (inline y3)
    f[5] = (y2 - s4) / T2
    y3 = s4 + (T1/T2)*(y2 - s4)

    # LL2: LeadLag (inline y4)
    f[6] = (y3 - s5) / T4
    y4 = s5 + (T3/T4)*(y3 - s5)

    # Washout
    f[7] = (KS*y4 - s6) / T6

    # Alg: v_s = sat((T5/T6)*(KS*y4 - s6))  (sat(0) = 0)
    vsat, _ = _ieeest_sat((T5/T6)*(KS*y4 - s6), pvec[14], pvec[15])
    f[8] = vs - vsat

    # Ctrl: w must equal 0 at init (speed deviation = 0 at SS)
    f[9] = w
    return nothing
end
