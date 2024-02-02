module GradPower

import Base: show

# numerics
using LinearAlgebra
using SparseArrays
using NLsolve
using KLU
using ForwardDiff
using UnicodePlots
using Statistics

import NLPModels
import MadNLP

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
    status::Bool
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
    ybus_real::SparseMatrixCSC{Float64,Int64}
end

abstract type AbstractDeviceType end
abstract type AbstractGeneratorType <: AbstractDeviceType end
abstract type AbstractLoadType <: AbstractDeviceType end

struct DynamicDevice
    dtype::AbstractDeviceType
    diff_ptr::Int64
    alg_ptr::Int64
    ctrl_ptr::Int64
    par_ptr::Int64
end

struct DynamicMap
    bus::Vector{Int64}
    gen::Vector{Int64}
    load::Vector{Int64}
    # pointers
    diff_ptr::Vector{Int64}
    alg_ptr::Vector{Int64}
    ctrl_ptr::Vector{Int64}
    par_ptr::Vector{Int64}
    # sizes
    diff_size::Vector{Int64}
    alg_size::Vector{Int64}
    ctrl_size::Vector{Int64}
    par_size::Vector{Int64}
end

function DynamicMap(ndevices::Int64)
    dm = DynamicMap(zeros(Int64, ndevices),
                    zeros(Int64, ndevices),
                    zeros(Int64, ndevices),
                    zeros(Int64, ndevices),
                    zeros(Int64, ndevices),
                    zeros(Int64, ndevices),
                    zeros(Int64, ndevices),
                    zeros(Int64, ndevices),
                    zeros(Int64, ndevices),
                    zeros(Int64, ndevices),
                    zeros(Int64, ndevices))
    return dm
end

mutable struct ContingencyEvent
    bus::Int64
    status::Bool
    rfault::Float64
    ton::Float64
    toff::Float64
end

function ContingencyEvent(bus::Int64, rfault::Float64, ton::Float64, toff::Float64)
    ce = ContingencyEvent(bus, false, rfault, ton, toff)
    return ce
end

mutable struct DynamicProblem
    zvec::AbstractArray
    uvec::AbstractArray
    pvec::AbstractArray
end

mutable struct PowerSystemDynamics
    devices::Vector{DynamicDevice}
    num_devices::Int64
    diff_dim::Int64
    alg_dim::Int64
    ctrl_dim::Int64
    par_dim::Int64
    map::Union{Nothing,DynamicMap}
    events::Union{Nothing, Vector{ContingencyEvent}}
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
    psd = PowerSystemDynamics(Vector{DynamicDevice}(), 0, 0, 0, 0, 0, nothing, Vector{ContingencyEvent}())
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
    diff_ptr = psd.diff_dim + 1
    alg_ptr = psd.alg_dim + 1
    ctrl_ptr = psd.ctrl_dim + 1
    par_ptr = psd.par_dim + 1
    push!(psd.devices, DynamicDevice(dtype, diff_ptr, alg_ptr, ctrl_ptr, par_ptr))
    psd.num_devices += 1
    psd.diff_dim += dtype.diff_size
    psd.alg_dim += dtype.alg_size
    psd.ctrl_dim += dtype.ctrl_size
    psd.par_dim += dtype.par_size
end

function find_gen(ps::PowerSystem, bus::Int64, gen_id::String)
    # remove spaces and appostrophes from gen_id.
    gen_id = replace(gen_id, " " => "")
    gen_id = replace(gen_id, "'" => "")
    for (i, gen) in enumerate(ps.gens)
            gen_id_gen = replace(gen.id, " " => "")
            gen_id_gen = replace(gen_id_gen, "'" => "")
        if gen.bus == bus && gen_id_gen == gen_id
            #println("Gen bus: ", gen.bus, " gen id: ", gen_id_gen, " gen_id: ", gen_id)
            return i
        end
    end
    return 0
end

function set_dynamics!(ps::PowerSystem, psd::PowerSystemDynamics; add_loads::Bool=true)

    # number of dynamic devices of generator type.
    num_gen_devices = 0
    for device in psd.devices
        if device.dtype isa AbstractGeneratorType
            num_gen_devices += 1
        end
    end

    # create dynamic map.
    dmap = DynamicMap(psd.num_devices - num_gen_devices + length(ps.gens) + length(ps.loads))
    #dmap = DynamicMap(psd.num_devices + length(ps.loads))

    matched_gens = []

    # iterate dynamic vector and assign buses.
    for (i, device) in enumerate(psd.devices)
        dmap.bus[i] = ps.busmap[device.dtype.bus]

        # if device is a generator, find the corresponding generator in the static system.
        if device.dtype isa AbstractGeneratorType
            gen_id = device.dtype.id
            dmap.gen[i] = find_gen(ps, dmap.bus[i], gen_id)
            if dmap.gen[i] == 0
                @warn "Generator not found for dynamic device $i"
            else
                push!(matched_gens, dmap.gen[i])
            end
        end
    end

    # check if all static generators have a corresponding dynamic generator. if not,
    # we will add static negative loads.
    if length(matched_gens) != length(ps.gens)
        @warn "Not all static generators have a corresponding dynamic generator. Adding negative loads."
        for (i, gen) in enumerate(ps.gens)
            if !(i in matched_gens)
                dmap.bus[psd.num_devices + 1] = gen.bus
                #dmap.load[psd.num_devices + 1] = 0
                #add_device!(psd, ZIPLoad(gen.bus, gen.id, -gen.psch, -gen.qsch, 0.0, 0.0, 0.0, 1.0, ps.buses[gen.bus].v0m, 0.0, 0.0))
                dmap.gen[psd.num_devices + 1] = i
                add_device!(psd, GenericGenerator(gen.bus, gen.id))
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

    # iterate dynamic vector and assign pointers and sizes.
    for (i, device) in enumerate(psd.devices)
        dmap.diff_ptr[i] = device.diff_ptr
        dmap.alg_ptr[i] = device.alg_ptr
        dmap.ctrl_ptr[i] = device.ctrl_ptr
        dmap.par_ptr[i] = device.par_ptr

        dmap.diff_size[i] = device.dtype.diff_size
        dmap.alg_size[i] = device.dtype.alg_size
        dmap.ctrl_size[i] = device.dtype.ctrl_size
        dmap.par_size[i] = device.dtype.par_size
    end

    psd.map = dmap
    ps.dynamic = psd
end

function DynamicProblem(ps::PowerSystem)
    @assert ps.dynamic != nothing "Dynamic system not initialized."
    @assert ps.dynamic.map != nothing "Dynamic map not initialized."
    dp = DynamicProblem(zeros(Float64, ps.dynamic.diff_dim + ps.dynamic.alg_dim + 2*length(ps.buses)),
                        zeros(Float64, ps.dynamic.ctrl_dim),
                        zeros(Float64, ps.dynamic.par_dim))
end

function add_event!(psd::PowerSystemDynamics, event::ContingencyEvent)
    push!(psd.events, event)
end

function add_event!(ps::PowerSystem, event::ContingencyEvent)
    if ps.dynamic == nothing
        @warn "No dynamics data found. Did not register event"
        return
    end
    add_event!(ps.dynamic, event)
end

function activate!(event::ContingencyEvent)
    event.status = true
end

function deactivate!(event::ContingencyEvent)
    event.status = false
end

# Include files. functionality.
include("utils.jl")
include("numerics.jl")
include("ad.jl")
include("network.jl")
include("pflow.jl")
include("dynamics.jl")
include("sensitivities.jl")

# Optimization model.
include("nlp.jl")

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
export from_psse

end # module GradPower
