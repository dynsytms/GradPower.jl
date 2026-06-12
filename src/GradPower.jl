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
abstract type AbstractGenControlType <: AbstractDeviceType end
abstract type AbstractGovernorType <: AbstractGenControlType end

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

# Phase 1 SoA layout (per-device-type tables). Included here so the
# `layout` field of PowerSystemDynamics below can be typed against it.
include("layout.jl")

mutable struct PowerSystemDynamics
    devices::Vector{DynamicDevice}
    num_devices::Int64
    diff_dim::Int64
    alg_dim::Int64
    ctrl_dim::Int64
    par_dim::Int64
    map::Union{Nothing,DynamicMap}
    uvec_idx::Union{Nothing,Vector{Int64}}
    events::Vector{ContingencyEvent}
    layout::Union{Nothing,SimulationLayout}
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
    psd = PowerSystemDynamics(Vector{DynamicDevice}(), 0, 0, 0, 0, 0, nothing, nothing, Vector{ContingencyEvent}(), nothing)
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
    mvbase = ps.baseMVA

    # create dynamic map.
    dmap = DynamicMap(psd.num_devices - num_gen_devices + length(ps.gens) + length(ps.loads))

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
                set_ratio!(device.dtype, ps.gens[dmap.gen[i]].mbase/mvbase)
                push!(matched_gens, dmap.gen[i])
            end
        end
    end

    # add controllers. We iterate dynamic devices again.

    # check if all static generators have a corresponding dynamic generator. if not,
    # we will add static negative loads.
    if length(matched_gens) != length(ps.gens)
        @warn "Not all static generators have a corresponding dynamic generator. Adding negative loads."
        for (i, gen) in enumerate(ps.gens)
            if !(i in matched_gens)
                dmap.bus[psd.num_devices + 1] = gen.bus
                dmap.gen[psd.num_devices + 1] = i
                add_device!(psd, GenericGenerator(gen.bus, gen.id))
            end
        end
    end

    # by default, add static loads to the dynamic system as ZIP loads.
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

    # create vector of control indexes. By default, no control is 0.
    psd.uvec_idx = zeros(Int64, psd.ctrl_dim)

    # Phase 1.5b: trait-driven wiring replaces the prior hand-coded
    # governor block. Each controller type declares attaches_to /
    # produces_signals / consumes_signals in src/coupling.jl (or in its
    # own kernel file once Phase 3 adds more types). This also fixes the
    # off-by-one bug from Phase 1 (p_m used to land in the e_fd slot).
    wire_controls!(psd, dmap)

    # Apply mbase/sbase scaling to controllers that need it (e.g. TGOV1's
    # R/VMAX/VMIN/DT). uqgrid does this in its parser. `set_ratio!` has
    # a no-op default; only devices whose parameters live on machine base
    # override (defined alongside the device struct).
    for (i, device) in enumerate(psd.devices)
        device.dtype isa AbstractGenControlType || continue
        gen_idx = dmap.gen[i]
        gen_idx == 0 && continue
        ratio = ps.gens[gen_idx].mbase / mvbase
        set_ratio!(device.dtype, ratio)
    end

    psd.map = dmap
    ps.dynamic = psd

    # Phase 1: build SoA layout tables. Phase 2 will switch the hot loop in
    # dynamics.jl to consume these; for now they coexist with the old
    # heterogeneous device loop and are not used at runtime.
    psd.layout = build_layout!(psd)

    # SEXS needs ps.busmap to resolve its bus into a global voltage index;
    # fix that up after the layout build.
    fix_sexs_vr_idx!(psd, ps)
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

# Phase 1 SoA table builders. Included AFTER dynamics.jl so device structs
# (Genrou, IEESGO, ZIPLoad, ...) are in scope. Each file defines an
# `_build_*_table_impl(psd)` function and registers itself with
# DEVICE_REGISTRY (Phase 1.5).
include("tables/genrou.jl")
include("tables/ieesgo.jl")
include("tables/tgov1.jl")
include("tables/sexs.jl")
include("tables/esdc1a.jl")
include("tables/zipload.jl")

# Phase 1.5b: device coupling graph. Defines `attaches_to`,
# `produces_signals`, `consumes_signals` traits + `wire_controls!`
# replacing the hand-coded governor wiring in set_dynamics!.
# Must be included AFTER tables/*.jl since the trait definitions
# reference concrete device types (Genrou, IEESGO).
include("coupling.jl")

# Phase 2.1: batched per-device-type kernels. Each kernel file owns
# `<dev>_residual_batch!`, `<dev>_jacobian_batch!`, and
# `<dev>_jac_positions!` (the position-precomputation helper that
# `preallocate_jacobian` calls once after building the sparsity pattern).
# Phase 2 cuts over `rhs_fun!`/`rhs_jac!` in dynamics.jl to call these
# instead of the heterogeneous loop. Until that cutover, both paths
# coexist and the parity test in test/test_parity.jl asserts agreement.
include("kernels/genrou.jl")
include("kernels/ieesgo.jl")
include("kernels/tgov1.jl")
include("kernels/sexs.jl")
include("kernels/zipload.jl")

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
