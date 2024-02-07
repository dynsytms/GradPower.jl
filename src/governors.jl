mutable struct IEESGO <: AbstractGovernorType
    # representation
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    # topology
    bus::Int64
    id::Int64
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
end

function IEESGO(bus, id, T1, T2, T3, T4, T5, T6, K1, K2, K3, pmax, pmin)
    governor = IEESGO(5, 1, 0, 11, bus, id, T1, T2, T3, T4, T5, T6, K1, K2, K3, pmax, pmin)
    return governor
end

function from_data_fields(::Type{IEESGO}, fields::Vector{SubString{String}})
    bus = parse(Int64, fields[1])
    id = parse(Int64, fields[3])
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
        w::Float64,
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

    PF0 = x0[1]
    PLL = x0[2]
    TP1 = x0[3]
    TP2 = x0[4]
    TP3 = x0[5]
    pref = x0[6]

    F[1] = (1.0/T1)*(K1*w - PF0)
    F[2] = (1/T3)*((1.0 - (T2/T3))*PF0 - PLL)
    SatP = pref - (T2/T3)*PF0 - PLL
    F[3] = (1/T4)*(SatP - TP1)
    F[4] = (1/T5)*(K2*TP1 - TP2)
    F[5] = (1/T6)*(K3*TP2 - TP3)
    F[6] = TP1*(1 - K2) + TP2*(1 - K3) + TP3 - pg
    return nothing
end
