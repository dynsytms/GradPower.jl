struct ZIPLoad <: AbstractDeviceType
    # representation
    diff_size::Int64
    alg_size::Int64
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
    load = ZIPLoad(0, 0, 9, bus, id, pinj, qinj, α, β, γ, weight, v0mag, yreal, yimag)
    return load
end
