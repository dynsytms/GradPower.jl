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
