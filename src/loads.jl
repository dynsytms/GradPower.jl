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

function get_device_name(dtype::ZIPLoad)
    return "ZIPLoad"
end

function get_bus(dtype::ZIPLoad)
    return dtype.bus
end

function get_param_names(dtype::ZIPLoad)
    return ["pinj", "qinj", "α", "β", "γ", "weight", "v0mag", "yreal", "yimag"]
end

function get_diff_names(dtype::ZIPLoad)
    return []
end

function get_alg_names(dtype::ZIPLoad)
    return []
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
