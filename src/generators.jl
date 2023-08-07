struct Genrou <: AbstractDeviceType
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
