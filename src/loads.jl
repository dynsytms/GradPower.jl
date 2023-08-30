mutable struct ZIPLoad <: AbstractLoadType
    # representation
    diff_size::Int64
    alg_size::Int64
    ctrl_size::Int64
    par_size::Int64
    # topology
    bus::Int64
    id::String
    # parameters
    pinj::Float64
    qinj::Float64
    α::Float64
    β::Float64
    γ::Float64
    weight::Float64
    v0mag::Float64
    yreal::Float64
    yimag::Float64
end

function ZIPLoad(bus, id, pinj, qinj, α, β, γ, weight, v0mag, yreal, yimag)
    load = ZIPLoad(0, 0, 0, 9, bus, id, pinj, qinj, α, β, γ, weight, v0mag, yreal, yimag)
    return load
end

function fill_pvec!(pvec::AbstractArray, dtype::ZIPLoad)
    pvec[1] = dtype.pinj
    pvec[2] = dtype.qinj
    pvec[3] = dtype.α
    pvec[4] = dtype.β
    pvec[5] = dtype.γ
    pvec[6] = dtype.weight
    pvec[7] = dtype.v0mag
    pvec[8] = dtype.yreal
    pvec[9] = dtype.yimag
end

function cinject!(
        f::AbstractArray,
        x::AbstractArray,
        y::AbstractArray,
        u::AbstractArray,
        p::AbstractArray,
        v::AbstractArray,
        dtype::ZIPLoad
)
    @inbounds begin
        vr = v[1]
        vi = v[2]

        pl = p[1]
        ql = p[2]
        α = p[3]
        β = p[4]
        γ = p[5]
        yload_real = p[8]
        yload_imag = p[9]
        yload_real = α*yload_real
        yload_imag = α*yload_imag

        vm2 = vr*vr + vi*vi
        vm2_tld = 0.2

        f[1] -= vr*yload_real - vi*yload_imag
        f[2] -= vr*yload_imag + vi*yload_real

        if vm2 > vm2_tld
            f[1] -= (1-α)*(pl*vr - ql*vi)/vm2
            f[2] -= (1-α)*(ql*vr + pl*vi)/vm2
        else
            f[1] -= (1-α)*(pl*vr - ql*vi)/vm2_tld
            f[2] -= (1-α)*(ql*vr + pl*vi)/vm2_tld
        end
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
        dtype::ZIPLoad
)
end

function preallocate_jacobian!(
        coord_list::Vector{Vector{Int}},
        diff_ptr::Int,
        alg_ptr::Int,
        ctrl_ptr::Int,
        volt_ptr::Int,
        dtype::ZIPLoad
)
    return nothing
end

function rhs_jac!(
    jac::AbstractMatrix,
    x::AbstractArray,
    y::AbstractArray,
    u::AbstractArray,
    p::AbstractArray,
    v::AbstractArray,
    idx_dev::Vector{Int},
    dtype::ZIPLoad
    )

    dp = idx_dev[1]
    ap = idx_dev[2]
    dev = idx_dev[3]
    pp = idx_dev[4]
    bus = idx_dev[5]

    vr = v[1]
    vi = v[2]

    pl = p[1]
    ql = p[2]
    α = p[3]
    β = p[4]
    γ = p[5]
    yload_real = p[8]
    yload_imag = p[9]

    vm = sqrt(vr^2 + vi^2)
    va = atan(vi, vr)

    row1 = dev + 2 * (bus - 1) + 1
    row2 = dev + 2 * (bus - 1) + 2
    col1 = dev + 2 * (bus - 1) + 1
    col2 = dev + 2 * (bus - 1) + 2

    # Constant admittance contribution
    vm2 = vr^2 + vi^2
    vm2_tld = 0.2

    val1 = -α * yload_real
    val2 = α * yload_imag
    jac[row1, col1] += val1
    jac[row1, col2] += val2

    val1 = -α * yload_imag
    val2 = -α * yload_real
    jac[row2, col1] += val1
    jac[row2, col2] += val2

    # Constant power contribution
    row = dev + 2 * (bus - 1) + 1
    if vm2 > vm2_tld
        val1 = (1-α) * ((ql * vr + pl * vi) * 2 * vr - ql * vm2) / vm2^2
        val2 = (1-α) * ((ql * vr + pl * vi) * 2 * vi - pl * vm2) / vm2^2
        jac[row1, col1] += val1
        jac[row1, col2] += val2
    else
        val1 = (1-α) * (-ql) / vm2_tld
        val2 = (1-α) * (-pl) / vm2_tld
        jac[row1, col1] += val1
        jac[row1, col2] += val2
    end

    row = dev + 2 * (bus - 1) + 2
    if vm2 > vm2_tld
        val1 = (1-α) * ((pl * vr - ql * vi) * 2 * vr - pl * vm2) / vm2^2
        val2 = (1-α) * ((pl * vr - ql * vi) * 2 * vi + ql * vm2) / vm2^2
        jac[row2, col1] += val1
        jac[row2, col2] += val2
    else
        val1 = (1-α) * (-pl) / vm2_tld
        val2 = (1-α) * ql / vm2_tld
        jac[row2, col1] += val1
        jac[row2, col2] += val2
    end
end
