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
