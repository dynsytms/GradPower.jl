mutable struct Genrou <: AbstractGeneratorType
    # representation
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
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
    gen = Genrou(6, 4, 2, 12, bus, id, x_d, x_q, x_dp, x_qp, x_ddp, xl, H, D, T_d0p, T_q0p, T_d0dp, T_q0dp)
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

function fill_pvec!(pvec::AbstractArray, dtype::Genrou)
    pvec[1] = dtype.x_d
    pvec[2] = dtype.x_q
    pvec[3] = dtype.x_dp
    pvec[4] = dtype.x_qp
    pvec[5] = dtype.x_ddp
    pvec[6] = dtype.xl
    pvec[7] = dtype.H
    pvec[8] = dtype.D
    pvec[9] = dtype.T_d0p
    pvec[10] = dtype.T_q0p
    pvec[11] = dtype.T_d0dp
    pvec[12] = dtype.T_q0dp
end

function initial_guess!(x0::AbstractArray, pvec::AbstractArray, p::Float64, q::Float64, vm::Float64, va::Float64, dtype::Genrou)
    x_d = pvec[1]
    x_q = pvec[2]
    x_dp = pvec[3]
    x_qp = pvec[4]
    x_ddp = pvec[5]
    xl = pvec[6]
    H = pvec[7]
    D = pvec[8]
    T_d0p = pvec[9]
    T_q0p = pvec[10]
    T_d0dp = pvec[11]
    T_q0dp = pvec[12]

    vt  = vm * cos(va) + 1im * vm * sin(va)
    ig = (p - 1im*q) / conj(vt)
    delta = angle(vt + (1im*x_q)*ig)

    v_d = vm * sin(delta - va)
    v_q = vm * cos(delta - va)
    i_d = (p*v_d + q*v_q) / (v_d^2 + v_q^2)
    i_q = (p*v_q - q*v_d) / (v_d^2 + v_q^2)

    phi_d = v_q
    phi_q = -v_d

    e_dp = (-x_qp)*i_q - phi_q
    e_qp = x_dp*i_d + phi_d

    phi_1d =  e_qp - (x_dp - xl)*i_d
    phi_2q =  -e_dp - (x_qp - xl)*i_q

    e_fd = e_qp + (x_d - x_dp)*i_d
    p_m = p

    x0[1] = e_qp
    x0[2] = e_dp
    x0[3] = phi_1d
    x0[4] = phi_2q
    x0[5] = 0.0
    x0[6] = delta
    x0[7] = v_q
    x0[8] = v_d
    x0[9] = i_q
    x0[10] = i_d
    x0[11] = e_fd
    x0[12] = p_m
end

function initialize_dynamics!(
        f::AbstractArray,
        x0::AbstractArray,
        pvec::AbstractArray,
        pg::Float64,
        qg::Float64,
        vm::Float64,
        va::Float64,
        dtype::Genrou
)

    x_d = pvec[1]
    x_q = pvec[2]
    x_dp = pvec[3]
    x_qp = pvec[4]
    x_ddp = pvec[5]
    xl = pvec[6]
    H = pvec[7]
    D = pvec[8]
    T_d0p = pvec[9]
    T_q0p = pvec[10]
    T_d0dp = pvec[11]
    T_q0dp = pvec[12]
    x_qdp = x_ddp

    e_qp = x0[1]
    e_dp = x0[2]
    phi_1d = x0[3]
    phi_2q = x0[4]
    w = x0[5]
    delta = x0[6]
    v_q = x0[7]
    v_d = x0[8]
    i_q = x0[9]
    i_d = x0[10]
    e_fd = x0[11]
    p_m = x0[12]
    
    # Generator dynamics
    psi_de = (x_ddp - xl) / (x_dp - xl) * e_qp + 
             (x_dp - x_ddp) / (x_dp - xl) * phi_1d

    psi_qe = -(x_ddp - xl) / (x_qp - xl) * e_dp + 
             (x_qp - x_ddp) / (x_qp - xl) * phi_2q

    # Machine states
    f[1] = (-e_qp + e_fd - (i_d - (-x_ddp + x_dp) * (-e_qp + i_d * 
           (x_dp - xl) + phi_1d) / ((x_dp - xl)^2.0)) * (x_d - x_dp)) / T_d0p
    f[2] = (-e_dp + (i_q - (-x_qdp + x_qp) * 
           (e_dp + i_q * (x_qp - xl) + phi_2q) / ((x_qp - xl)^2.0)) * 
           (x_q - x_qp)) / T_q0p
    f[3] = (e_qp - i_d * (x_dp - xl) - phi_1d) / T_d0dp
    f[4] = (-e_dp - i_q * (x_qp - xl) - phi_2q) / T_q0dp

    f[5] = (p_m - psi_de * i_q + psi_qe * i_d) / (2.0 * H)
    f[6] = 2.0 * π * 60.0 * w

    # Stator currents
    f[7] = i_d - ((x_ddp - xl) / (x_dp - xl) * e_qp + 
           (x_dp - x_ddp) / (x_dp - xl) * phi_1d - v_q) / x_ddp
    f[8] = i_q - (-(x_qdp - xl) / (x_qp - xl) * e_dp + 
           (x_qp - x_qdp) / (x_qp - xl) * phi_2q + v_d) / x_qdp

    # Stator voltage
    f[9] = v_d - vm * sin(delta - va)
    f[10] = v_q - vm * cos(delta - va)

    # Stator additional equations
    f[11] = v_d * i_d + v_q * i_q - pg
    f[12] = v_q * i_d - v_d * i_q - qg
end
