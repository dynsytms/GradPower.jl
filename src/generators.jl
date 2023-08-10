struct Genrou <: AbstractGeneratorType
    # representation
    diff_size::Int64
    alg_size::Int64
    par_size::Int64
    # topology
    bus::Int64
    id::Int64
    # parameters
    x_d::Float64
    x_q::Float64
    x_dp::Float64
    x_qp::Float64
    x_ddp::Float64
    xl::Float64
    H::Float64
    D::Float64
    T_d0p::Float64
    T_q0p::Float64
    T_d0dp::Float64
    T_q0dp::Float64
end

function Genrou(bus, id, x_d, x_q, x_dp, x_qp, x_ddp, xl, H, D, T_d0p, T_q0p, T_d0dp, T_q0dp)
    gen = Genrou(12, 6, 4, bus, id, x_d, x_q, x_dp, x_qp, x_ddp, xl, H, D, T_d0p, T_q0p, T_d0dp, T_q0dp)
    return gen
end

function from_data_fields(::Type{Genrou}, fields::Vector{SubString{String}})
    # Parse fields into Genrou constructor
    bus = parse(Int64, fields[1])
    id = parse(Int64, fields[3])
    T_d0p = parse(Float64, fields[4])
    T_d0dp = parse(Float64, fields[5])
    T_q0p = parse(Float64, fields[6])
    T_q0dp = parse(Float64, fields[7])
    H = parse(Float64, fields[8])
    D = parse(Float64, fields[9])
    x_d = parse(Float64, fields[10])
    x_q = parse(Float64, fields[11])
    x_dp = parse(Float64, fields[12])
    x_qp = parse(Float64, fields[13])
    x_ddp = parse(Float64, fields[14])
    xl = parse(Float64, fields[15])

    # The last two fields are optional, check if they are provided before parsing
    S1 = length(fields) >= 17 ? parse(Float64, fields[16]) : 0.0
    S2 = length(fields) >= 18 ? parse(Float64, fields[17]) : 0.0

    Genrou(bus, id, x_d, x_q, x_dp, x_qp, x_ddp, xl, H, D, T_d0p, T_q0p, T_d0dp, T_q0dp)
end
