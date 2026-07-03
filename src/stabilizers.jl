# IEEEST Power System Stabilizer.
#
# Signal chain (all time constants nonzero, limits disabled):
#
#   input (omega-1) -> Lag2ndOrd(K=1, T1=A1, T2=A2)
#                   -> LeadLag2ndOrd(T1=A3, T2=A4, T3=A5, T4=A6)
#                   -> LeadLag(T1=T1, T2=T2)
#                   -> LeadLag(T3=T3, T4=T4)
#                   -> Gain(KS)
#                   -> Washout(T=T6, K=T5)
#                   -> v_s (output to exciter)
#
# Diff states (7):
#   s0 = F1_x   (Lag2ndOrd internal state)
#   s1 = F1_y   (Lag2ndOrd output)
#   s2 = F2_x1  (LeadLag2ndOrd state 1)
#   s3 = F2_x2  (LeadLag2ndOrd state 2)
#   s4 = LL1_x  (LeadLag 1 state)
#   s5 = LL2_x  (LeadLag 2 state)
#   s6 = WO_x   (Washout state)
#
# Alg states (1):
#   y0 = v_s    (PSS output, read by exciter via vs_idx)
#
# Ctrl slots (1):
#   u[0] = omega (generator rotor speed, from generator diff state w)
#
# Parameters (19):
#   A1, A2, A3, A4, A5, A6, T1, T2, T3, T4, T5, T6, KS,
#   LSMAX, LSMIN, VCU, VCL, MODE, BUSR
#
# Residual equations (derived from IEEEST transfer-function block diagram):
#
#   sig = omega - 1  (MODE=1)
#
#   F1 (Lag2ndOrd, K=1, T1=A1, T2=A2):
#     A2 * ds0 = sig - s1 - A1*s0
#     ds1 = s0
#
#   F2 (LeadLag2ndOrd, T1=A3, T2=A4, T3=A5, T4=A6):
#     A4 * ds2 = s1 - s3 - A3*s2
#     ds3 = s2
#     y2 = s3 + A5*s2 + (A6/A4)*(s1 - s3 - A3*s2)   [inline algebraic]
#
#   LL1 (LeadLag, T1=T1, T2=T2):
#     T2 * ds4 = y2 - s4
#     y3 = s4 + (T1/T2)*(y2 - s4)                    [inline algebraic]
#
#   LL2 (LeadLag, T1=T3, T2=T4):
#     T4 * ds5 = y3 - s5
#     y4 = s5 + (T3/T4)*(y3 - s5)                    [inline algebraic]
#
#   Washout (T=T6, K=T5):
#     T6 * ds6 = KS*y4 - s6
#     v_s = (T5/T6)*(KS*y4 - s6)                     [alg equation]
#
# Limits (LSMAX/LSMIN, VCU/VCL) parsed and stored but NOT enforced.
# Limit enforcement deferred to a future phase.

abstract type AbstractStabilizerType <: AbstractGenControlType end

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

    # Clamp zero time constants: a zero pair (e.g. A3=A4=0) means "bypass
    # this lead-lag block" (unity gain). We replace zeros with a value large
    # enough to avoid introducing stiffness (1/eps must stay comparable to
    # 1/dt, not 1e6) but small enough to act as a passthrough.
    _EPS_TC = 0.001
    A1 = A1 == 0.0 ? _EPS_TC : A1
    A2 = A2 == 0.0 ? _EPS_TC : A2
    A3 = A3 == 0.0 ? _EPS_TC : A3
    A4 = A4 == 0.0 ? _EPS_TC : A4
    A5 = A5 == 0.0 ? _EPS_TC : A5
    A6 = A6 == 0.0 ? _EPS_TC : A6
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
    pvec[14] = dtype.LSMAX
    pvec[15] = dtype.LSMIN
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

    # F1: Lag2ndOrd
    f[1] = (sig - s1 - A1*s0) / A2
    f[2] = s0

    # F2: LeadLag2ndOrd (inline y2)
    f[3] = (s1 - s3 - A3*s2) / A4
    f[4] = s2
    y2 = s3 + A5*s2 + (A6/A4)*(s1 - s3 - A3*s2)

    # LL1: LeadLag (inline y3)
    f[5] = (y2 - s4) / T2
    y3 = s4 + (T1/T2)*(y2 - s4)

    # LL2: LeadLag (inline y4)
    f[6] = (y3 - s5) / T4
    y4 = s5 + (T3/T4)*(y3 - s5)

    # Washout
    f[7] = (KS*y4 - s6) / T6

    # Alg: v_s = (T5/T6)*(KS*y4 - s6)
    f[8] = vs - (T5/T6)*(KS*y4 - s6)

    # Ctrl: w must equal 0 at init (speed deviation = 0 at SS)
    f[9] = w
    return nothing
end
