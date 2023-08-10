module GradPower

# numerics
using LinearAlgebra
using SparseArrays
using NLsolve

# profiling
using TimerOutputs

# misc
using Printf

mutable struct Bus
    i::Int64
    id::String
    type::Int64
    baseKV::Float64
    v0m::Float64
    v0a::Float64
end

mutable struct Gen
    bus::Int64
    id::String
    psch::Float64
    qsch::Float64
    mbase::Float64
end

mutable struct Load
    bus::Int64
    id::String
    pd::Float64
    qd::Float64
end

mutable struct Branch
    fr::Int
    to::Int
    id::String
    r::Float64
    x::Float64
    sh::Float64
    tap::Float64
    shift::Float64
end

mutable struct Shunt
    bus::Int64
    id::String
    gsh::Float64
    bsh::Float64
end

mutable struct Network
    adjacency::Vector{Vector{Int}}
    ybus::SparseMatrixCSC{ComplexF64,Int64}
end

abstract type AbstractDeviceType end
abstract type AbstractGeneratorType <: AbstractDeviceType end
abstract type AbstractLoadType <: AbstractDeviceType end

struct DynamicDevice
    dtype::AbstractDeviceType
    diff_ptr::Int64
    alg_ptr::Int64
    par_ptr::Int64
end


struct DynamicMap
    bus::Vector{Int64}
    gen::Vector{Int64}
    load::Vector{Int64}
end

function DynamicMap(ndevices::Int64)
    dm = DynamicMap(zeros(Int64, ndevices), zeros(Int64, ndevices), zeros(Int64, ndevices))
    return dm
end

mutable struct PowerSystemDynamics
    devices::Vector{DynamicDevice}
    num_devices::Int64
    diff_size::Int64
    alg_size::Int64
    par_size::Int64
    map::Union{Nothing,DynamicMap}
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
    dynamic::Union{Nothing,PowerSystemDynamics}
    profiler::TimerOutput
end

function PowerSystem(baseMVA, buses, gens, loads, branches, shunts, busmap)
    ps = PowerSystem(baseMVA, buses, gens, loads, branches, shunts, busmap, nothing, nothing, TimerOutput())
    return ps
end

struct PowerFlowSolution
    volt::AbstractArray
    sinj::AbstractArray
end

function PowerSystemDynamics()
    psd = PowerSystemDynamics(Vector{DynamicDevice}(), 0, 0, 0, 0, nothing)
    return psd
end

function PowerSystemDynamics(psse_dyr_file::String)
    psd = PowerSystemDynamics()
    dyr_data = read_psse_dyr(psse_dyr_file)
    psse_devices = create_device_vector(dyr_data)
    for device in psse_devices
        add_device!(psd, device)
    end
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

function find_gen(ps::PowerSystem, bus::Int64)
    for (i, gen) in enumerate(ps.gens)
        if gen.bus == bus
            return i
        end
    end
    return 0
end


function set_dynamics!(ps::PowerSystem, psd::PowerSystemDynamics; add_loads::Bool=true)

    # create dynamic map.
    dmap = DynamicMap(psd.num_devices + length(ps.loads))

    # iterate dynamic vector and assign buses.
    for (i, device) in enumerate(psd.devices)
        dmap.bus[i] = ps.busmap[device.dtype.bus]

        # if device is a generator, find the corresponding generator in the static system.
        if device.dtype isa AbstractGeneratorType
            dmap.gen[i] = find_gen(ps, dmap.bus[i])
            if dmap.gen[i] == 0
                @warn "Generator not found for dynamic device $i"
            end
        end
    end

    if add_loads
        for (i, load) in enumerate(ps.loads)
            dmap.load[psd.num_devices + 1] = i
            dmap.bus[psd.num_devices + 1] = load.bus
            add_device!(psd, ZIPLoad(load.bus, load.id, load.pd, load.qd, 1.0, 0.0, 0.0, 1.0, ps.buses[load.bus].v0m, 0.0, 0.0))
        end
    end
    psd.map = dmap
    ps.dynamic = psd
end


# Include files. functionality.
include("numerics.jl")
include("network.jl")
include("pflow.jl")
include("dynamics.jl")

# Include files. devices.
include("generators.jl")
include("loads.jl")

# Include files. parsers.
include("parse.jl")

# Display functions.
include("display.jl")

# Exports
export Bus, Gen, Load, Branch, Shunt, PowerSystem
export build_network!
export runpf
export DynamicDevice, PowerSystemDynamics
export add_device!
export Genrou
export from_data_field

end # module GradPower
