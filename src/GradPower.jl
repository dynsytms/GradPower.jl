module GradPower

import Base: show

# numerics
using LinearAlgebra
using SparseArrays
using NLsolve
using KLU
using ForwardDiff
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

# SoA layout (per-device-type tables). Included here so the
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

function PowerSystemDynamics(psse_dyr_file::String;
                              active_gen_keys::Union{Nothing,Set{Tuple{Int64,String}}}=nothing)
    psd = PowerSystemDynamics()
    dyr_data = read_psse_dyr(psse_dyr_file)
    psse_devices = create_device_vector(dyr_data; active_gen_keys=active_gen_keys)
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

function set_dynamics!(ps::PowerSystem, psd::PowerSystemDynamics;
                        add_loads::Bool=true,
                        add_static_gen_stubs::Bool=true)

    # number of dynamic devices of generator type.
    num_gen_devices = 0
    for device in psd.devices
        if device.dtype isa AbstractGeneratorType
            num_gen_devices += 1
        end
    end
    mvbase = ps.baseMVA

    # create dynamic map. Upper-bound the size: every dynamic device kept,
    # plus one stand-in per static gen (covers the worst case where every
    # dynamic gen is orphaned from its static counterpart), plus one ZIPLoad
    # per static load. Unused trailing slots stay zero and are ignored.
    dmap = DynamicMap(psd.num_devices + length(ps.gens) + length(ps.loads))

    matched_gens = Int[]
    orphan_dyn = Tuple{Int64,String}[]

    # iterate dynamic vector and assign buses.
    for (i, device) in enumerate(psd.devices)
        dmap.bus[i] = ps.busmap[device.dtype.bus]

        # if device is a generator, find the corresponding generator in the static system.
        if device.dtype isa AbstractGeneratorType
            gen_id = device.dtype.id
            dmap.gen[i] = find_gen(ps, dmap.bus[i], gen_id)
            if dmap.gen[i] == 0
                push!(orphan_dyn, (device.dtype.bus, gen_id))
            else
                set_ratio!(device.dtype, ps.gens[dmap.gen[i]].mbase/mvbase)
                push!(matched_gens, dmap.gen[i])
            end
        end
    end

    # Summary warning instead of one-per-device — large cases (70k) can have
    # thousands of dynamic rows without an active static gen counterpart.
    if !isempty(orphan_dyn)
        @warn "$(length(orphan_dyn)) dynamic generator(s) had no active static gen at the same (bus, id); first few: $(first(orphan_dyn, 5))"
    end

    # add controllers. We iterate dynamic devices again.

    # For every active static gen not covered by a dynamic machine model,
    # inject a StaticGenerator (constant pinj + PV/SLACK voltage regulation).
    # Aggregate one StaticGenerator per bus across all unmatched gens on that bus.
    if add_static_gen_stubs && length(matched_gens) != length(ps.gens)
        matched_set = Set(matched_gens)
        by_bus = Dict{Int64,Vector{Int64}}()
        for (i, gen) in enumerate(ps.gens)
            i in matched_set && continue
            push!(get!(by_bus, gen.bus, Int64[]), Int64(i))
        end
        n_aggregated = sum(length, values(by_bus); init=0)
        @info "Adding $(length(by_bus)) StaticGenerator(s) aggregating $(n_aggregated) static gen(s) without a dynamic machine model."
        for (bus_internal, gen_idxs) in by_bus
            bt = Int64(ps.buses[bus_internal].type)
            # vset: take from any of the aggregated gens' static voltage setpoint
            # (already stored on ps.buses[bus].v0m by raw_to_grad's PV/SLACK
            # write-back). All gens on a bus share that bus's setpoint.
            vset = ps.buses[bus_internal].v0m
            aset = ps.buses[bus_internal].v0a
            # StaticGenerator stores its bus as the EXTERNAL PSS/E number so the
            # build_layout! path mirrors Genrou/SEXS; fix_static_gen_bus_idx!
            # remaps to internal index post-build.
            ext_bus = ps.buses[bus_internal].i
            sg = StaticGenerator(Int64(ext_bus), bt, gen_idxs, vset, aset)
            slot = psd.num_devices + 1
            dmap.bus[slot] = bus_internal
            # dmap.gen[slot]: first aggregated index — used as a representative
            # so init can recover totals via the device's own gen_idxs vector.
            dmap.gen[slot] = gen_idxs[1]
            add_device!(psd, sg)
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

    # Trait-driven wiring: each controller type declares attaches_to /
    # produces_signals / consumes_signals in src/coupling.jl (or in its
    # own kernel file).
    wire_controls!(psd, dmap)

    # Apply mbase/sbase scaling to controllers that need it (e.g. TGOV1's
    # R/VMAX/VMIN/DT). `set_ratio!` has a no-op default; only devices whose
    # parameters live on machine base override (defined alongside the device struct).
    for (i, device) in enumerate(psd.devices)
        device.dtype isa AbstractGenControlType || continue
        gen_idx = dmap.gen[i]
        gen_idx == 0 && continue
        ratio = ps.gens[gen_idx].mbase / mvbase
        set_ratio!(device.dtype, ratio)
    end

    psd.map = dmap
    ps.dynamic = psd

    # Build SoA layout tables consumed by the hot loop in dynamics.jl.
    psd.layout = build_layout!(psd)

    # SEXS needs ps.busmap to resolve its bus into a global voltage index;
    # fix that up after the layout build. Genrou.bus also lives in the table
    # as the external PSSE bus number and must be remapped to internal index.
    fix_genrou_bus_idx!(psd, ps)
    fix_sexs_vr_idx!(psd, ps)
    fix_static_gen_bus_idx!(psd, ps)
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

# SoA table builders. Included AFTER dynamics.jl so device structs
# (Genrou, IEESGO, ZIPLoad, ...) are in scope. Each file defines an
# `_build_*_table_impl(psd)` function and registers itself with
# DEVICE_REGISTRY.
include("tables/genrou.jl")
include("tables/ieesgo.jl")
include("tables/tgov1.jl")
include("tables/sexs.jl")
include("tables/esdc1a.jl")
include("tables/zipload.jl")
include("tables/static_gen.jl")

# Device coupling graph. Defines `attaches_to`, `produces_signals`,
# `consumes_signals` traits + `wire_controls!`. Must be included AFTER
# tables/*.jl since the trait definitions reference concrete device
# types (Genrou, IEESGO).
include("coupling.jl")

# Batched per-device-type kernels. Each kernel file owns
# `<dev>_residual_batch!`, `<dev>_jacobian_batch!`, and
# `<dev>_jac_positions!` (the position-precomputation helper that
# `preallocate_jacobian` calls once after building the sparsity pattern).
# `rhs_fun!`/`rhs_jac!` in dynamics.jl call these.
include("kernels/genrou.jl")
include("kernels/ieesgo.jl")
include("kernels/tgov1.jl")
include("kernels/sexs.jl")
include("kernels/zipload.jl")
include("kernels/static_gen.jl")

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
