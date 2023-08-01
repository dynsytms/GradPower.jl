module GradPower

# numerics
using LinearAlgebra
using SparseArrays
using NLsolve

# profiling
using TimerOutputs

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
    ybus::SparseMatrixCSC{ComplexF64,Int64}
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
    profiler::TimerOutput
end

function PowerSystem(baseMVA, buses, gens, loads, branches, shunts, busmap)
    ps = PowerSystem(baseMVA, buses, gens, loads, branches, shunts, busmap, nothing, TimerOutput())
    return ps
end

# Power Flow and static analysis

struct PowerFlowSolution
    volt::AbstractArray
    sinj::AbstractArray
end

# Dynamics

abstract type AbstractDeviceType end
struct DynamicDevice
    dtype::AbstractDeviceType
    diff_ptr::Int64
    alg_ptr::Int64
    par_ptr::Int64
end

mutable struct PowerSystemDynamics
    devices::Vector{DynamicDevice}
    num_devices::Int64
    diff_size::Int64
    alg_size::Int64
    par_size::Int64
end

function PowerSystemDynamics()
    psd = PowerSystemDynamics(Vector{DynamicDevice}(), 0, 0, 0, 0)
    return psd
end

"""
    add_device!(psd::PowerSystemDynamics, dtype::AbstractDeviceType, bus::Int64)

Add a device to the dynamic system. The device type `dtype` must be a subtype of `AbstractDeviceType`.
"""
function add_device!(psd::PowerSystemDynamics, dtype::AbstractDeviceType)
    diff_ptr = psd.diff_size + 1
    alg_ptr = psd.alg_size + 1
    par_ptr = psd.par_size + 1
    push!(psd.devices, DynamicDevice(dtype, diff_ptr, alg_ptr, par_ptr))
    psd.num_devices += 1
    psd.diff_size += dtype.diff_size
    psd.alg_size += dtype.alg_size
    psd.par_size += dtype.par_size
end


# Include files. functionality.
include("parse.jl")
include("numerics.jl")
include("network.jl")
include("pflow.jl")
include("dynamics.jl")

# Include files. devices.
include("generators.jl")

# Exports
export Bus, Gen, Load, Branch, Shunt, PowerSystem
export build_network!
export runpf
export DynamicDevice, PowerSystemDynamics
export add_device!
export Genrou

end # module GradPower
