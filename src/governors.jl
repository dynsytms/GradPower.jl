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
