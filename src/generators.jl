mutable struct Genrou <: AbstractGeneratorType
    # representation
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    # topology
    bus::Int64
    id::String
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

function GenericGenerator(bus, id)
    gen = Genrou(bus, id, 1.9266, 1.8442, 0.3812, 0.5469, 0.2889, 0.2443, 50.0, 0.0, 7.729, 0.859, 0.047, 0.068)
    return gen
end

function from_data_fields(::Type{Genrou}, fields::Vector{SubString{String}})
    # Parse fields into Genrou constructor
    bus = parse(Int64, fields[1])
    id = String(fields[3])
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

function get_device_name(dtype::Genrou)
    return "Genrou"
end

function get_bus(dtype::Genrou)
    return dtype.bus
end

function get_param_names(dtype::Genrou)
    return ["x_d", "x_q", "x_dp", "x_qp", "x_ddp", "xl", "H", "D", "T_d0p", "T_q0p", "T_d0dp", "T_q0dp"]
end

function get_diff_names(dtype::Genrou)
    return ["delta", "omega", "e_dp", "e_qp", "phi_1d", "phi_2q"]
end

function get_alg_names(dtype::Genrou)
    return ["i_d", "i_q"]
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

function cinject!(
        f::AbstractArray,
        x::AbstractArray,
        y::AbstractArray,
        u::AbstractArray,
        p::AbstractArray,
        v::AbstractArray,
        dtype::Genrou
)
    @inbounds begin
        v_q = y[1]
        v_d = y[2]
        i_q = y[3]
        i_d = y[4]
        delta = x[6]
        #sd, cd = sincos(delta)
        sd = sin(delta)
        cd = cos(delta)

        f[1] += sd*i_d + cd*i_q
        f[2] += -cd*i_d + sd*i_q
    end
end

function rhs_fun!(
        f_diff::AbstractArray,
        f_alg::AbstractArray,
        x::AbstractArray,
        y::AbstractArray,
        u::AbstractArray,
        p::AbstractArray,
        v::AbstractArray,
        dtype::Genrou
)
    @inbounds begin
        # parameters
        x_d = p[1]
        x_q = p[2]
        x_dp = p[3]
        x_qp = p[4]
        x_ddp = p[5]
        xl = p[6]
        H = p[7]
        D = p[8]
        T_d0p = p[9]
        T_q0p = p[10]
        T_d0dp = p[11]
        T_q0dp = p[12]
        x_qdp = x_ddp

        # states
        e_qp = x[1]
        e_dp = x[2]
        phi_1d = x[3]
        phi_2q = x[4]
        w = x[5]
        delta = x[6]
        v_q = y[1]
        v_d = y[2]
        i_q = y[3]
        i_d = y[4]

        # control
        e_fd = u[1]
        p_m = u[2]

        # voltage
        vr = v[1]
        vi = v[2]

        tmech = (p_m - D*w)/(1.0 + w)

        # auxiliary variables
        psi_de = (x_ddp - xl)/(x_dp - xl)*e_qp +
                (x_dp - x_ddp)/(x_dp - xl)*phi_1d

        psi_qe = -(x_ddp - xl)/(x_qp - xl)*e_dp +
                (x_qp - x_ddp)/(x_qp - xl)*phi_2q

        # equations
        f_diff[1] = (-e_qp + e_fd - (i_d - (-x_ddp + x_dp)*(-e_qp + i_d*(x_dp - xl)
                    + phi_1d)/((x_dp - xl)^2))*(x_d - x_dp))/T_d0p
        f_diff[2] = (-e_dp + (i_q - (-x_qdp + x_qp)*( e_dp + i_q*(x_qp - xl)
                    + phi_2q)/((x_qp - xl)^2))*(x_q - x_qp))/T_q0p
        f_diff[3] = ( e_qp - i_d*(x_dp - xl) - phi_1d)/T_d0dp
        f_diff[4] = (-e_dp - i_q*(x_qp - xl) - phi_2q)/T_q0dp
        f_diff[5] = (tmech - psi_de*i_q + psi_qe*i_d)/(2.0*H)
        f_diff[6] = 2.0*π*60.0*w

        # Stator currents
        f_alg[1] = i_d - ((x_ddp - xl)/(x_dp - xl)*e_qp +
                (x_dp - x_ddp)/(x_dp - xl)*phi_1d - v_q)/x_ddp
        f_alg[2] = i_q - (-(x_qdp - xl)/(x_qp - xl)*e_dp +
                (x_qp - x_qdp)/(x_qp - xl)*phi_2q + v_d)/x_qdp
        
        sd, cd = sincos(delta)
        f_alg[3] = v_d - (vr*sd - vi*cd)
        f_alg[4] = v_q - (vr*cd + vi*sd)
    end
end

function rhs_alg!(
        f_alg::AbstractArray,
        x::AbstractArray,
        y::AbstractArray,
        u::AbstractArray,
        p::AbstractArray,
        v::AbstractArray,
        dtype::Genrou
)
    @inbounds begin
        # parameters
        x_d = p[1]
        x_q = p[2]
        x_dp = p[3]
        x_qp = p[4]
        x_ddp = p[5]
        xl = p[6]
        H = p[7]
        D = p[8]
        T_d0p = p[9]
        T_q0p = p[10]
        T_d0dp = p[11]
        T_q0dp = p[12]
        x_qdp = x_ddp

        # states
        e_qp = x[1]
        e_dp = x[2]
        phi_1d = x[3]
        phi_2q = x[4]
        w = x[5]
        delta = x[6]
        v_q = y[1]
        v_d = y[2]
        i_q = y[3]
        i_d = y[4]

        # control
        e_fd = u[1]
        p_m = u[2]

        # voltage
        vr = v[1]
        vi = v[2]

        tmech = (p_m - D*w)/(1.0 + w)

        # auxiliary variables
        psi_de = (x_ddp - xl)/(x_dp - xl)*e_qp +
                (x_dp - x_ddp)/(x_dp - xl)*phi_1d

        psi_qe = -(x_ddp - xl)/(x_qp - xl)*e_dp +
                (x_qp - x_ddp)/(x_qp - xl)*phi_2q

        # Stator currents
        f_alg[1] = i_d - ((x_ddp - xl)/(x_dp - xl)*e_qp +
                (x_dp - x_ddp)/(x_dp - xl)*phi_1d - v_q)/x_ddp
        f_alg[2] = i_q - (-(x_qdp - xl)/(x_qp - xl)*e_dp +
                (x_qp - x_qdp)/(x_qp - xl)*phi_2q + v_d)/x_qdp
        sd, cd = sincos(delta)
        f_alg[3] = v_d - (vr*sd - vi*cd)
        f_alg[4] = v_q - (vr*cd + vi*sd)
    end
end

function rhs_diff!(
        f_diff::AbstractArray,
        x::AbstractArray,
        y::AbstractArray,
        u::AbstractArray,
        p::AbstractArray,
        v::AbstractArray,
        dtype::Genrou
)
    @inbounds begin
        # parameters
        x_d = p[1]
        x_q = p[2]
        x_dp = p[3]
        x_qp = p[4]
        x_ddp = p[5]
        xl = p[6]
        H = p[7]
        D = p[8]
        T_d0p = p[9]
        T_q0p = p[10]
        T_d0dp = p[11]
        T_q0dp = p[12]
        x_qdp = x_ddp

        # states
        e_qp = x[1]
        e_dp = x[2]
        phi_1d = x[3]
        phi_2q = x[4]
        w = x[5]
        delta = x[6]
        v_q = y[1]
        v_d = y[2]
        i_q = y[3]
        i_d = y[4]

        # control
        e_fd = u[1]
        p_m = u[2]

        # voltage
        vr = v[1]
        vi = v[2]

        tmech = (p_m - D*w)/(1.0 + w)

        # auxiliary variables
        psi_de = (x_ddp - xl)/(x_dp - xl)*e_qp +
                (x_dp - x_ddp)/(x_dp - xl)*phi_1d

        psi_qe = -(x_ddp - xl)/(x_qp - xl)*e_dp +
                (x_qp - x_ddp)/(x_qp - xl)*phi_2q

        # equations
        f_diff[1] = (-e_qp + e_fd - (i_d - (-x_ddp + x_dp)*(-e_qp + i_d*(x_dp - xl)
                    + phi_1d)/((x_dp - xl)^2))*(x_d - x_dp))/T_d0p
        f_diff[2] = (-e_dp + (i_q - (-x_qdp + x_qp)*( e_dp + i_q*(x_qp - xl)
                    + phi_2q)/((x_qp - xl)^2))*(x_q - x_qp))/T_q0p
        f_diff[3] = ( e_qp - i_d*(x_dp - xl) - phi_1d)/T_d0dp
        f_diff[4] = (-e_dp - i_q*(x_qp - xl) - phi_2q)/T_q0dp
        f_diff[5] = (tmech - psi_de*i_q + psi_qe*i_d)/(2.0*H)
        f_diff[6] = 2.0*π*60.0*w
    end
end

function preallocate_jacobian!(
    coord_list::Vector{Vector{Int}},
    diff_ptr::Int,
    alg_ptr::Int,
    ctrl_ptr::Int,
    volt_ptr::Int,
    dtype::Genrou
)
    dp = diff_ptr
    ap = alg_ptr
    vp = volt_ptr

    e_qp = dp
    e_dp = dp + 1
    phi_1d = dp + 2
    phi_2q = dp + 3
    w = dp + 4
    delta = dp + 5

    v_q = ap
    v_d = ap + 1
    i_q = ap + 2
    i_d = ap + 3
    vr, vi = vp, vp + 1

    exciter = false
    governor = false

    # First row
    row = dp
    cols = exciter ? [e_qp, phi_1d, ctrl_ptr, i_d] : [e_qp, phi_1d, i_d]
    append!(coord_list[row], cols)

    # Second row
    row = dp + 1
    cols = [e_dp, phi_2q, i_q]
    append!(coord_list[row], cols)

    # Third row
    row = dp + 2
    cols = [e_qp, phi_1d, i_d]
    append!(coord_list[row], cols)

    # Fourth row
    row = dp + 3
    cols = [e_dp, phi_2q, i_q]
    append!(coord_list[row], cols)

    # Fifth row
    row = dp + 4
    cols = governor ? [e_qp, e_dp, phi_1d, phi_2q, ctrl_ptr, i_q, i_d, w] : [e_qp, e_dp, phi_1d, phi_2q, i_q, i_d, w]
    append!(coord_list[row], cols)

    # Sixth row
    row = dp + 5
    cols = [w]
    append!(coord_list[row], cols)

    # Algebraic part
    row = ap
    cols = [e_qp, phi_1d, v_q, i_d]
    append!(coord_list[row], cols)

    row = ap + 1
    cols = [e_dp, phi_2q, v_d, i_q]
    append!(coord_list[row], cols)

    row = ap + 2
    cols = [delta, v_d, vr, vi]
    append!(coord_list[row], cols)

    row = ap + 3
    cols = [delta, v_q, vr, vi]
    append!(coord_list[row], cols)

    row = vp
    cols = [delta, i_q, i_d]
    append!(coord_list[row], cols)

    row = vp + 1
    cols = [delta, i_q, i_d]
    append!(coord_list[row], cols)
end

function rhs_jac!(
    jac::AbstractMatrix,
    x::AbstractArray,
    y::AbstractArray,
    u::AbstractArray,
    p::AbstractArray,
    v::AbstractArray,
    idx_dev::Vector{Int},
    dtype::Genrou
    )

        dp = idx_dev[1]
        ap = idx_dev[2]
        dev = idx_dev[3]
        pp = idx_dev[4]
        bus = idx_dev[5]

        # parameters
        x_d = p[1]
        x_q = p[2]
        x_dp = p[3]
        x_qp = p[4]
        x_ddp = p[5]
        xl = p[6]
        H = p[7]
        D = p[8]
        T_d0p = p[9]
        T_q0p = p[10]
        T_d0dp = p[11]
        T_q0dp = p[12]
        x_qdp = x_ddp

        # states
        e_qp = x[1]
        e_dp = x[2]
        phi_1d = x[3]
        phi_2q = x[4]
        w = x[5]
        delta = x[6]
        v_q = y[1]
        v_d = y[2]
        i_q = y[3]
        i_d = y[4]

        # control
        e_fd = u[1]
        p_m = u[2]

        # voltage
        vr = v[1]
        vi = v[2]
    
        # indexes
        e_qp_idx = dp + 0
        e_dp_idx = dp + 1
        phi_1d_idx = dp + 2
        phi_2q_idx = dp + 3
        w_idx = dp + 4
        delta_idx = dp + 5
        v_q_idx = ap + 0
        v_d_idx = ap + 1
        i_q_idx = ap + 2
        i_d_idx = ap + 3
        vr_idx = dev + 2*(bus - 1) + 1
        vi_idx = dev + 2*(bus - 1) + 2

    psi_de = (x_ddp - xl) / (x_dp - xl) * e_qp + 
             (x_dp - x_ddp) / (x_dp - xl) * phi_1d

    psi_qe = -(x_ddp - xl) / (x_qp - xl) * e_dp + 
             (x_qp - x_ddp) / (x_qp - xl) * phi_2q

    row = 0
    col = 0
    val = 0.0

    # row 1
    row = dp
    
    col = e_qp_idx
    val = (-(x_d - x_dp)*(-x_ddp + x_dp)*(x_dp - xl)^(-2.0) - 1)/T_d0p
    jac[row, col] = val

    col = phi_1d_idx
    val = (x_d - x_dp)*(-x_ddp + x_dp)*(x_dp - xl)^(-2.0)/T_d0p
    jac[row, col] = val
    
    if false
        col = efd_idx
        val = 1/T_d0p
        jac[row, col] = val
        col = i_d_idx
        val = -(x_d - x_dp)*(-(-x_ddp + x_dp)*(x_dp - xl)^(-1.0) + 1)/T_d0p
        jac[row, col] = val
    else
        col = i_d_idx
        val = -(x_d - x_dp)*(-(-x_ddp + x_dp)*(x_dp - xl)^(-1.0) + 1)/T_d0p
        jac[row, col] = val
    end

    # second row
    row = dp + 1
    col = e_dp_idx
    val = (-(x_q - x_qp)*(-x_qdp + x_qp)*(x_qp - xl)^(-2.0) - 1)/T_q0p
    jac[row, col] = val
    col = phi_2q_idx
    val = -(x_q - x_qp)*(-x_qdp + x_qp)*(x_qp - xl)^(-2.0)/T_q0p
    jac[row, col] = val
    col = i_q_idx
    val = (x_q - x_qp)*(-(-x_qdp + x_qp)*(x_qp - xl)^(-1.0) + 1)/T_q0p
    jac[row, col] = val

    # Third row
    row = dp + 2
    col = e_qp_idx
    val = 1.0 / T_d0dp
    jac[row, col] = val
    col = phi_1d_idx
    val = -1.0 / T_d0dp
    jac[row, col] = val
    col = i_d_idx
    val = (-x_dp + xl) / T_d0dp
    jac[row, col] = val

    # Fourth row
    row = dp + 3
    col = e_dp_idx
    val = -1.0 / T_q0dp
    jac[row, col] = val
    col = phi_2q_idx
    val = -1.0 / T_q0dp
    jac[row, col] = val
    col = i_q_idx
    val = (-x_qp + xl) / T_q0dp
    jac[row, col] = val

    # Fifth row
    row = dp + 4
    col = e_qp_idx
    val = -0.5 * i_q * (x_ddp - xl) / (H * (x_dp - xl))
    jac[row, col] = val
    col = e_dp_idx
    val = 0.5 * i_d * (-x_ddp + xl) / (H * (x_qp - xl))
    jac[row, col] = val
    col = phi_1d_idx
    val = -0.5 * i_q * (-x_ddp + x_dp) / (H * (x_dp - xl))
    jac[row, col] = val
    col = phi_2q_idx
    val = 0.5 * i_d * (-x_ddp + x_qp) / (H * (x_qp - xl))
    jac[row, col] = val
    col = w_idx
    val = 0.5 * (-D / (w + 1.0) - (-D * w + p_m) / (w + 1.0)^2.0) / H
    jac[row, col] = val

    if false # Replace with appropriate condition
        col = i_q_idx
        val = 0.5 * (-e_qp * (x_ddp - xl) / (x_dp - xl) - phi_1d * (-x_ddp + x_dp) / (x_dp - xl)) / H
        jac[row, col] = val

        col = i_d_idx
        val = 0.5 * (e_dp * (-x_ddp + xl) / (x_qp - xl) + phi_2q * (-x_ddp + x_qp) / (x_qp - xl)) / H
        jac[row, col] = val

        col = pm_idx
        val = 0.5 / (H * (w + 1))
        jac[row, col] = val
    else
        col = i_q_idx
        val = 0.5 * (-e_qp * (x_ddp - xl) / (x_dp - xl) - phi_1d * (-x_ddp + x_dp) / (x_dp - xl)) / H
        jac[row, col] = val

        col = i_d_idx
        val = 0.5 * (e_dp * (-x_ddp + xl) / (x_qp - xl) + phi_2q * (-x_ddp + x_qp) / (x_qp - xl)) / H
        jac[row, col] = val
    end

    # Sixth row
    row = dp + 5
    col = w_idx
    val = 120.0 * π # Using π for Pi in Julia
    jac[row, col] = val

    # Algebraic first
    row = ap
    col = e_qp_idx
    val = -(x_ddp - xl) / (x_ddp * (x_dp - xl))
    jac[row, col] = val

    col = phi_1d_idx
    val = -(-x_ddp + x_dp) / (x_ddp * (x_dp - xl))
    jac[row, col] = val

    col = v_q_idx
    val = 1 / x_ddp
    jac[row, col] = val

    col = i_d_idx
    val = 1.0
    jac[row, col] = val

    # Algebraic second
    row = ap + 1
    col = e_dp_idx
    val = -(-x_qdp + xl) / (x_qdp * (x_qp - xl))
    jac[row, col] = val

    col = phi_2q_idx
    val = -(-x_qdp + x_qp) / (x_qdp * (x_qp - xl))
    jac[row, col] = val

    col = v_d_idx
    val = -1 / x_qdp
    jac[row, col] = val

    col = i_q_idx
    val = 1.0
    jac[row, col] = val

    # Algebraic third
    row = ap + 2
    col = delta_idx
    val = -vr * cos(delta) - vi * sin(delta)
    jac[row, col] = val

    col = v_d_idx
    val = 1.0
    jac[row, col] = val

    col = vr_idx
    val = -sin(delta)
    jac[row, col] = val

    col = vi_idx
    val = cos(delta)
    jac[row, col] = val

    # Algebraic fourth
    row = ap + 3
    col = delta_idx
    val = vr * sin(delta) - vi * cos(delta)
    jac[row, col] = val

    col = v_q_idx
    val = 1.0
    jac[row, col] = val

    col = vr_idx
    val = -cos(delta)
    jac[row, col] = val

    col = vi_idx
    val = -sin(delta)
    jac[row, col] = val

    # Power injection
    row = dev + 2 * (bus - 1) + 1
    col = delta_idx
    val = i_d * cos(delta) - i_q * sin(delta)
    jac[row, col] = val

    col = i_q_idx
    val = cos(delta)
    jac[row, col] = val

    col = i_d_idx
    val = sin(delta)
    jac[row, col] = val

    # Power injection
    row = dev + 2 * (bus - 1) + 2
    col = delta_idx
    val = i_d * sin(delta) + i_q * cos(delta)
    jac[row, col] = val

    col = i_q_idx
    val = sin(delta)
    jac[row, col] = val

    col = i_d_idx
    val = -cos(delta)
    jac[row, col] = val
end
