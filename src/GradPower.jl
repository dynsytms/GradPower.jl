module GradPower

using LinearAlgebra
using SparseArrays

struct Bus
    i::Int64
    id::String
    type::Int64
    baseKV::Float64
    v0m::Float64
    v0a::Float64
end

struct Gen
    bus::Int64
    psch::Float64
    qsch::Float64
    mbase::Float64
end

struct Load
    bus::Int64
    pd::Float64
    qd::Float64
end

struct Branch
    fr::Int
    to::Int
    r::Float64
    x::Float64
    sh::Float64
    tap::Float64
    shift::Float64
end

struct Shunt
    bus::Int64
    gsh::Float64
    bsh::Float64
end

mutable struct Network
    adjacency::Vector{Vector{Int}}
    ybus::Matrix
end

mutable struct PowerSystem
    baseMVA::Float64
    buses::Array{Bus,1}
    gens::Array{Gen,1}
    loads::Array{Load,1}
    branches::Array{Branch,1}
    shunts::Array{Shunt,1}
    busmap::Dict{Int,Int}
    network::Union{Nothing,Network}
end

function PowerSystem(baseMVA, buses, gens, loads, branches, shunts, busmap)
    ps = PowerSystem(baseMVA, buses, gens, loads, branches, shunts, busmap, nothing)
    return ps
end

include("parse.jl")
include("network.jl")

end # module GradPower
