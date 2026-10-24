module GradPower

import Base: show

# numerics
using LinearAlgebra
using SparseArrays
using NLsolve
using KLU
using Krylov

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
    # Reactive limits, pu on the SYSTEM base (not mbase). Used only by the
    # power flow's PV->PQ switching; +/-Inf means "unlimited".
    qmax::Float64
    qmin::Float64
end

# Backward-compatible constructor: no reactive limits.
Gen(bus, id, psch, qsch, mbase, status) =
    Gen(bus, id, psch, qsch, mbase, status, Inf, -Inf)

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

mutable struct DynamicDevice
    dtype::AbstractDeviceType
    diff_ptr::Int64
    alg_ptr::Int64
    ctrl_ptr::Int64
    par_ptr::Int64
    online::Bool
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

"""
    ContingencyEvent(bus, rfault, ton, toff; xfault=0.0)

Shunt fault at internal bus `bus`, active for `ton <= t < toff`, with fault
impedance `rfault + j xfault` (pu), i.e. admittance `y = 1/(rfault + j xfault)`
to ground. `xfault = 0` (default) is the resistive fault `y = 1/rfault`.
`rfault >= 0` and `xfault` must be finite and not both zero: a bolted fault
(zero impedance) is not representable; use a small impedance instead.
The GPU integrators support `xfault = 0` only.
"""
mutable struct ContingencyEvent
    bus::Int64
    status::Bool
    rfault::Float64
    ton::Float64
    toff::Float64
    xfault::Float64
end

function ContingencyEvent(bus::Int64, rfault::Float64, ton::Float64, toff::Float64;
                          xfault::Float64=0.0)
    (isfinite(rfault) && isfinite(xfault) && rfault >= 0.0) ||
        throw(ArgumentError("fault impedance must be finite with rfault >= 0 (got $rfault + j$xfault)"))
    (rfault == 0.0 && xfault == 0.0) &&
        throw(ArgumentError("bolted fault (rfault = xfault = 0) is not supported: use a small impedance"))
    return ContingencyEvent(bus, false, rfault, ton, toff, xfault)
end

# Fault admittance (g, b) = 1/(rfault + j xfault). The resistive case keeps
# 1/rfault exactly (previous arithmetic).
@inline function fault_admittance(ev::ContingencyEvent)
    ev.xfault == 0.0 && return (1.0 / ev.rfault, 0.0)
    d = ev.rfault * ev.rfault + ev.xfault * ev.xfault
    return (ev.rfault / d, -ev.xfault / d)
end

"""
    DisconnectDeviceEvent

Disconnect a device at a specified time by setting its `online` flag to false.
The device's batch residual/Jacobian kernels skip it when `online == false`.

Fields:
- `device_idx`: index into `psd.devices`
- `ton`: time at which the device is disconnected
"""
struct DisconnectDeviceEvent
    device_idx::Int64
    ton::Float64
end

"""
    TripLineEvent

Trip a transmission line at `ton` by subtracting its admittance from
`ybus_real`. If `toff < Inf`, the line is reclosed at `toff` by adding
the admittance back.

Fields:
- `from_bus`, `to_bus`: internal bus indices
- `ton`: time at which the line is tripped
- `toff`: time at which the line is reclosed (`Inf` = permanent trip)
- `yff`, `yft`, `ytf`, `ytt`: precomputed complex admittance contributions
"""
struct TripLineEvent
    from_bus::Int64
    to_bus::Int64
    ton::Float64
    toff::Float64
    yff::ComplexF64
    yft::ComplexF64
    ytf::ComplexF64
    ytt::ComplexF64
    branch::Int64     # index into ps.branches (0 = not known)
end
TripLineEvent(fr, to, ton, toff, yff, yft, ytf, ytt) =
    TripLineEvent(fr, to, ton, toff, yff, yft, ytf, ytt, 0)

"""
    LoadStepEvent(device_idx, t, scale)

At time `t`, set the ZIPLoad `psd.devices[device_idx]` to `scale` times its
INITIAL value (the `p` it had when the integration started): constant-power
`P`, `Q` and the constant-impedance `G`, `B` all scale, so the total load at
nominal voltage scales by `scale`. Steps do not compound: a later step on the
same load with `scale = 1.0` restores it. `t` must be at least one step
(`t >= dt`); the parameters are restored when the integration returns. Build
with [`create_load_step_event`](@ref) (external bus number + load id).
"""
struct LoadStepEvent
    device_idx::Int64
    t::Float64
    scale::Float64
end

mutable struct DynamicProblem
    zvec::AbstractArray
    uvec::AbstractArray
    pvec::AbstractArray
end

struct FixedStateLimit
    table_index::Int
    state_index::Int
    input_index::Int
    diagonal_position::Int
    input_position::Int
    lower_parameter_index::Int
    upper_parameter_index::Int
    device_type::Symbol
    bus::Int
    device_id::String
end

struct LimitEvent
    device_type::Symbol
    bus::Int
    device_id::String
    state_index::Int
    side::Symbol
    action::Symbol
    time::Float64
    bound::Float64
    state_value::Float64
end

mutable struct LimitWorkspace
    method::Symbol
    mu::Float64
    tolerance::Float64
    descriptors::Vector{FixedStateLimit}
    online::Vector{Bool}
    modes::Vector{UInt8} # 0=free, 1=lower, 2=upper
    events::Vector{LimitEvent}
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
    disconnect_events::Vector{DisconnectDeviceEvent}
    trip_events::Vector{TripLineEvent}
    load_step_events::Vector{LoadStepEvent}
    layout::Union{Nothing,SimulationLayout}
    clusters::Any  # Union{Nothing,ClusterTable} — can't use ClusterTable here (defined later in clusters.jl)
    diff_indices::Union{Nothing,Vector{Int}}  # z-positions of differential states after cluster reordering
    is_diff::Union{Nothing,BitVector}  # precomputed mask: is_diff[i] = true iff z[i] is a diff state
    # z-rows (diff and alg) of offline devices. Their residual is zeroed and
    # alg rows get a unit Jacobian diagonal, so the states stay frozen at
    # their value when the device was disconnected.
    offline_diff_rows::Vector{Int}
    offline_alg_rows::Vector{Int}
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
    converged::Bool
end

"""
    SolverLog

Lightweight event timer for the solver, modeled after PETSc PetscLogEvent.
Tracks per-event call count and cumulative wall-clock time (in nanoseconds).
Opt-in: pass to `integrate!` via the `log` kwarg; when `nothing`, zero overhead.
"""
mutable struct SolverLog
    residual_count::Int
    residual_ns::UInt64
    jacobian_count::Int
    jacobian_ns::UInt64
    lsolve_factor_count::Int
    lsolve_factor_ns::UInt64
    lsolve_solve_count::Int
    lsolve_solve_ns::UInt64
    ybus_mul_count::Int
    ybus_mul_ns::UInt64
    init_ns::UInt64
    parse_ns::UInt64
    gmres_iters::Vector{Int}  # per-Newton-step GMRES iteration counts
end

function SolverLog()
    SolverLog(0, UInt64(0), 0, UInt64(0), 0, UInt64(0), 0, UInt64(0),
              0, UInt64(0), UInt64(0), UInt64(0), Int[])
end

function PowerSystemDynamics()
    psd = PowerSystemDynamics(Vector{DynamicDevice}(), 0, 0, 0, 0, 0, nothing, nothing, Vector{ContingencyEvent}(), Vector{DisconnectDeviceEvent}(), Vector{TripLineEvent}(), LoadStepEvent[], nothing, nothing, nothing, nothing, Int[], Int[])
    return psd
end

function PowerSystemDynamics(psse_dyr_file::String;
                              active_gen_keys::Union{Nothing,Set{Tuple{Int64,String}}}=nothing,
                              surrogates::Bool=false)
    psd = PowerSystemDynamics()
    dyr_data = read_psse_dyr(psse_dyr_file)
    psse_devices = create_device_vector(dyr_data; active_gen_keys=active_gen_keys,
                                        surrogates=surrogates)
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
    push!(psd.devices, DynamicDevice(dtype, diff_ptr, alg_ptr, ctrl_ptr, par_ptr, true))
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
    fix_esdc1a_vr_idx!(psd, ps)
    fix_ieeest_wiring!(psd, ps)
    fix_static_gen_bus_idx!(psd, ps)

    # Build device clusters and reorder state vector to cluster-contiguous
    # layout: z = (w_1, w_2, ..., w_Nc, v). This must happen AFTER all
    # fix_* calls (bus indices resolved) and AFTER build_layout! (SoA
    # tables populated). The reordering updates diff_ptr/alg_ptr on both
    # DynamicDevice and SoA tables, plus uvec_idx entries.
    ct = build_clusters!(psd, dmap)
    reorder_state!(psd, ct)
    psd.clusters = ct

    # Refresh DynamicMap pointers to match the reordered layout.
    for (i, device) in enumerate(psd.devices)
        dmap.diff_ptr[i] = device.diff_ptr
        dmap.alg_ptr[i]  = device.alg_ptr
    end
end

"""
    DynamicProblem(sys)

Allocate the state `zvec`, control `uvec` and parameter `pvec` vectors of the
dynamic model of `sys`. Fill them with [`initialize_dynamics!`](@ref).
"""
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

"""
    add_event!(sys, ev::ContingencyEvent)

Register a bus fault on `sys`; it is applied by every subsequent integration
until removed from `sys.dynamic.events`.
"""
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

function add_disconnect_event!(psd::PowerSystemDynamics, event::DisconnectDeviceEvent)
    push!(psd.disconnect_events, event)
end

"""
    add_disconnect_event!(sys, ev::DisconnectDeviceEvent)

Register a device disconnection. [`add_generator_trip!`](@ref) and
[`add_load_trip!`](@ref) build these from PSS/E bus numbers and ids.
"""
function add_disconnect_event!(ps::PowerSystem, event::DisconnectDeviceEvent)
    if ps.dynamic === nothing
        @warn "No dynamics data found. Did not register disconnect event"
        return
    end
    add_disconnect_event!(ps.dynamic, event)
end

function add_trip_event!(psd::PowerSystemDynamics, event::TripLineEvent)
    # Tripping a branch twice would subtract its admittance twice.
    event.branch != 0 && any(e -> e.branch == event.branch, psd.trip_events) &&
        throw(ArgumentError("branch $(event.branch) already has a trip event"))
    push!(psd.trip_events, event)
end

"""
    add_trip_event!(sys, ev::TripLineEvent)

Register a line trip (build it with [`create_trip_line_event`](@ref)).
"""
function add_trip_event!(ps::PowerSystem, event::TripLineEvent)
    if ps.dynamic === nothing
        @warn "No dynamics data found. Did not register trip event"
        return
    end
    add_trip_event!(ps.dynamic, event)
end

"""
    add_load_step_event!(sys, ev::LoadStepEvent)

Register a load step (build it with [`create_load_step_event`](@ref)). The
device must be a ZIP load and `scale` finite and `>= 0`.
"""
function add_load_step_event!(ps::PowerSystem, event::LoadStepEvent)
    ps.dynamic === nothing && throw(ArgumentError("no dynamics data: cannot register a load step"))
    dev = ps.dynamic.devices[event.device_idx]
    dev.dtype isa ZIPLoad ||
        throw(ArgumentError("LoadStepEvent device $(event.device_idx) is a $(typeof(dev.dtype)), not a ZIPLoad"))
    isfinite(event.scale) && event.scale >= 0.0 ||
        throw(ArgumentError("LoadStepEvent scale must be finite and >= 0, got $(event.scale)"))
    push!(ps.dynamic.load_step_events, event)
end

"""
    find_load_device(ps, bus_ext, id) -> device index

Index in `ps.dynamic.devices` of the ZIPLoad with PSS/E bus number `bus_ext`
and load id `id` (compared after stripping quotes/blanks, as for machine ids).
Errors if there is none or more than one.
"""
function find_load_device(ps::PowerSystem, bus_ext::Int, id::AbstractString)
    haskey(ps.busmap, bus_ext) || throw(ArgumentError("bus $bus_ext not in the case"))
    b = ps.busmap[bus_ext]
    key = _normalize_id(id)
    hits = findall(d -> d.dtype isa ZIPLoad && d.dtype.bus == b && _normalize_id(d.dtype.id) == key,
                   ps.dynamic.devices)
    isempty(hits) && throw(ArgumentError("no load with id \"$key\" at bus $bus_ext"))
    length(hits) == 1 || throw(ArgumentError("load id \"$key\" at bus $bus_ext is ambiguous ($(length(hits)) matches)"))
    return hits[1]
end

"""
    create_load_step_event(ps, bus_ext, id, t, scale) -> LoadStepEvent

Load step on the load `id` at PSS/E bus `bus_ext`: at time `t` it becomes
`scale` times its initial value (e.g. `1.1` = +10 %, `0.0` = shed).
"""
create_load_step_event(ps::PowerSystem, bus_ext::Int, id::AbstractString, t::Real, scale::Real) =
    LoadStepEvent(find_load_device(ps, bus_ext, id), Float64(t), Float64(scale))

"""
    add_load_trip!(ps, bus_ext, id, t) -> DisconnectDeviceEvent

Remove (permanently disconnect) the load `id` at PSS/E bus `bus_ext` at `t`.
"""
function add_load_trip!(ps::PowerSystem, bus_ext::Int, id::AbstractString, t::Real)
    ev = DisconnectDeviceEvent(find_load_device(ps, bus_ext, id), Float64(t))
    add_disconnect_event!(ps, ev)
    return ev
end

"""
    generator_devices(ps, bus_ext, id) -> Vector{Int}

Indices in `ps.dynamic.devices` that model the generator `id` at PSS/E bus
`bus_ext`: its machine model and every controller (governor, exciter,
stabilizer) attached to it (same bus and id, as used to wire them). A
generator without a machine model is represented by the bus's
`StaticGenerator`, returned only if it stands for this generator alone.
"""
function generator_devices(ps::PowerSystem, bus_ext::Int, id::AbstractString)
    haskey(ps.busmap, bus_ext) || throw(ArgumentError("bus $bus_ext not in the case"))
    dyn = ps.dynamic::PowerSystemDynamics
    key = _normalize_id(id)
    match(d) = (d.dtype isa AbstractGeneratorType || d.dtype isa AbstractGenControlType) &&
               !(d.dtype isa StaticGenerator) &&
               d.dtype.bus == bus_ext && _normalize_id(d.dtype.id) == key
    idx = findall(match, dyn.devices)
    machines = filter(i -> dyn.devices[i].dtype isa AbstractGeneratorType, idx)
    length(machines) > 1 &&
        throw(ArgumentError("generator \"$key\" at bus $bus_ext has $(length(machines)) machine models"))
    length(machines) == 1 && return idx
    isempty(idx) || throw(ArgumentError("generator \"$key\" at bus $bus_ext has controllers but no machine model"))
    # No machine model: the static stand-in, if it represents only this gen.
    g = find_gen(ps, ps.busmap[bus_ext], String(id))
    g == 0 && throw(ArgumentError("no generator \"$key\" at bus $bus_ext"))
    sg = findfirst(d -> d.dtype isa StaticGenerator && g in d.dtype.gen_idxs, dyn.devices)
    sg === nothing && throw(ArgumentError("generator \"$key\" at bus $bus_ext has no dynamic representation"))
    dt = dyn.devices[sg].dtype
    length(dt.gen_idxs) == 1 ||
        throw(ArgumentError("generator \"$key\" at bus $bus_ext is aggregated with $(length(dt.gen_idxs) - 1) " *
                            "other static generator(s); removing it alone is not supported"))
    dt.bus_type == 3 &&
        throw(ArgumentError("generator \"$key\" at bus $bus_ext is the slack static generator; removing it is not supported"))
    return [sg]
end

"""
    add_generator_trip!(ps, bus_ext, id, t) -> Vector{DisconnectDeviceEvent}

Remove the generator `id` at PSS/E bus `bus_ext` at `t`: its machine and all
its controllers go offline together (see [`generator_devices`](@ref)); their
states are frozen and the machine stops injecting current.
"""
function add_generator_trip!(ps::PowerSystem, bus_ext::Int, id::AbstractString, t::Real)
    evs = [DisconnectDeviceEvent(i, Float64(t)) for i in generator_devices(ps, bus_ext, id)]
    foreach(ev -> add_disconnect_event!(ps, ev), evs)
    return evs
end

# Circuit ids are stored as parsed (e.g. "\"'1 '\""): compare without
# quotes and blanks.
_normalize_ckt(s::AbstractString) = _normalize_id(replace(s, "\"" => ""))

"""
    find_branch(ps, from_bus_ext, to_bus_ext; ckt=nothing) -> branch index

Index in `ps.branches` of the branch (line or two-winding transformer)
between PSS/E buses `from_bus_ext` and `to_bus_ext`, in either direction,
with circuit id `ckt` (quotes/blanks ignored). Without `ckt` the branch must
be unique: parallel circuits raise an `ArgumentError` listing their ids.
"""
function find_branch(ps::PowerSystem, from_bus_ext::Int, to_bus_ext::Int; ckt=nothing)
    for b in (from_bus_ext, to_bus_ext)
        haskey(ps.busmap, b) || throw(ArgumentError("bus $b not in the case"))
    end
    f = ps.busmap[from_bus_ext]; t = ps.busmap[to_bus_ext]
    hits = findall(b -> (b.fr == f && b.to == t) || (b.fr == t && b.to == f), ps.branches)
    isempty(hits) && throw(ArgumentError("no branch between buses $from_bus_ext and $to_bus_ext"))
    if ckt === nothing
        length(hits) == 1 && return hits[1]
        ids = join((_normalize_ckt(ps.branches[i].id) for i in hits), ", ")
        throw(ArgumentError("$(length(hits)) parallel branches between buses $from_bus_ext and " *
                            "$to_bus_ext (circuits $ids): pass ckt"))
    end
    key = _normalize_ckt(string(ckt))
    sel = filter(i -> _normalize_ckt(ps.branches[i].id) == key, hits)
    isempty(sel) && throw(ArgumentError("no circuit \"$key\" between buses $from_bus_ext and $to_bus_ext"))
    length(sel) == 1 || throw(ArgumentError("circuit \"$key\" between buses $from_bus_ext and $to_bus_ext is ambiguous"))
    return sel[1]
end

"""
    create_trip_line_event(ps, from_bus_ext, to_bus_ext, ton; toff=Inf, ckt=nothing) -> TripLineEvent

Create a TripLineEvent (line removal) by looking up the branch between
`from_bus_ext` and `to_bus_ext` (external PSS/E bus numbers, circuit `ckt`;
see [`find_branch`](@ref)) and precomputing its admittance contribution. The
line is tripped at `ton` and reclosed at `toff` (default `Inf` = permanent).
"""
function create_trip_line_event(ps::PowerSystem, from_bus_ext::Int, to_bus_ext::Int, ton::Float64;
                                toff::Float64=Inf, ckt=nothing)
    bidx = find_branch(ps, from_bus_ext, to_bus_ext; ckt=ckt)
    br = ps.branches[bidx]

    # Compute admittance contributions (same convention as create_ybus_complex)
    tap = br.tap
    shift = br.shift
    if tap > 0.0
        tpsh = tap * exp(im * π / 180.0 * shift)
    else
        tpsh = 1.0 + 0.0im
        tap = 1.0
    end
    y = 1.0 / (br.r + im * br.x)

    fr = br.fr
    to = br.to

    yff = y / (tap * tap) + im * 0.5 * br.sh
    ytt = y + (im * 0.5 * br.sh) / (tap * tap)
    yft = -y / conj(tpsh)
    ytf = -y / tpsh

    return TripLineEvent(fr, to, ton, toff, yff, yft, ytf, ytt, bidx)
end

"""
    _apply_trip_line!(ybus_real, event::TripLineEvent)

Subtract the tripped line's admittance contribution from `ybus_real` in-place.
Uses the same realify convention as `network.jl`.
"""
function _apply_trip_line!(ybus_real::SparseMatrixCSC, event::TripLineEvent)
    fr = event.from_bus
    to = event.to_bus

    # Subtract yff (from-from block)
    _subtract_complex_from_realified!(ybus_real, fr, fr, event.yff)
    # Subtract yft (from-to block)
    _subtract_complex_from_realified!(ybus_real, fr, to, event.yft)
    # Subtract ytf (to-from block)
    _subtract_complex_from_realified!(ybus_real, to, fr, event.ytf)
    # Subtract ytt (to-to block)
    _subtract_complex_from_realified!(ybus_real, to, to, event.ytt)
    return nothing
end

function _apply_reclose_line!(ybus_real::SparseMatrixCSC, event::TripLineEvent)
    fr = event.from_bus
    to = event.to_bus
    _add_complex_to_realified!(ybus_real, fr, fr, event.yff)
    _add_complex_to_realified!(ybus_real, fr, to, event.yft)
    _add_complex_to_realified!(ybus_real, to, fr, event.ytf)
    _add_complex_to_realified!(ybus_real, to, to, event.ytt)
    return nothing
end

function _add_complex_to_realified!(ybus_real::SparseMatrixCSC,
                                     bus_row::Int, bus_col::Int,
                                     yval::ComplexF64)
    zr = real(yval)
    zi = imag(yval)
    r1 = 2 * bus_row - 1
    r2 = 2 * bus_row
    c1 = 2 * bus_col - 1
    c2 = 2 * bus_col
    ybus_real[r1, c1] += zr
    ybus_real[r2, c2] += zr
    ybus_real[r1, c2] += -zi
    ybus_real[r2, c1] += zi
    return nothing
end

"""
    _subtract_complex_from_realified!(ybus_real, bus_row, bus_col, yval)

Subtract a complex admittance entry from the realified Ybus matrix at the
(bus_row, bus_col) 2x2 block. The realify convention is:
  [real(y), -imag(y); imag(y), real(y)]
"""
function _subtract_complex_from_realified!(ybus_real::SparseMatrixCSC,
                                            bus_row::Int, bus_col::Int,
                                            yval::ComplexF64)
    zr = real(yval)
    zi = imag(yval)
    r1 = 2 * bus_row - 1
    r2 = 2 * bus_row
    c1 = 2 * bus_col - 1
    c2 = 2 * bus_col

    # (r1, c1) -= zr
    ybus_real[r1, c1] -= zr
    # (r2, c2) -= zr
    ybus_real[r2, c2] -= zr
    # (r1, c2) -= -zi  =>  += zi
    ybus_real[r1, c2] += zi
    # (r2, c1) -= zi
    ybus_real[r2, c1] -= zi
    return nothing
end

# Include files. functionality.
include("utils.jl")
include("direct_solver.jl")
include("numerics.jl")
include("recorder.jl")
include("results.jl")
include("network.jl")
include("pflow.jl")
include("cpf/statistics.jl")
include("cpf/formulation.jl")
include("cpf/palc.jl")
include("cpf/safeguards.jl")
include("cpf/events.jl")
include("cpf/results.jl")
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
include("tables/ieeest.jl")
include("tables/static_gen.jl")

# Device coupling graph. Defines `attaches_to`, `produces_signals`,
# `consumes_signals` traits + `wire_controls!`. Must be included AFTER
# tables/*.jl since the trait definitions reference concrete device
# types (Genrou, IEESGO).
include("coupling.jl")

# Cluster infrastructure: DeviceCluster, ClusterTable, build_clusters!,
# reorder_state!, A_k extraction helpers. Must be AFTER coupling.jl
# (uses _normalize_id, attaches_to traits) and device type definitions.
include("clusters.jl")

# Batched per-device-type kernels. Each kernel file owns
# `<dev>_residual_batch!`, `<dev>_jacobian_batch!`, and
# `<dev>_jac_positions!` (the position-precomputation helper that
# `preallocate_jacobian` calls once after building the sparsity pattern).
# `rhs_fun!`/`rhs_jac!` in dynamics.jl call these.
include("kernels/genrou.jl")
include("kernels/ieesgo.jl")
include("kernels/tgov1.jl")
include("kernels/sexs.jl")
include("kernels/esdc1a.jl")
include("kernels/zipload.jl")
include("kernels/ieeest.jl")
include("kernels/static_gen.jl")

# Fixed-state limit declarations and backward-Euler row transformations.
include("limits.jl")

# KernelAbstractions wrappers + injection buffer dispatch.
include("kernels/ka_wrappers.jl")

# Schur complement reduction. Must be AFTER clusters.jl (cluster types)
# and kernels/*.jl (preallocate_jacobian references kernel preallocators).
include("schur.jl")

# Device online/offline helpers. Must be AFTER device type definitions
# and kernel includes so Genrou, IEESGO, etc. are in scope.
"""
    _set_device_offline!(psd, device_idx)

Set a device's `online` flag to false on both the DynamicDevice struct and
the corresponding entry in the SoA table. The batch kernels skip devices
whose `online` entry is false.
"""
function _set_device_offline!(psd::PowerSystemDynamics, device_idx::Int)
    device = psd.devices[device_idx]
    device.online || return nothing
    device.online = false
    append!(psd.offline_diff_rows, device.diff_ptr:device.diff_ptr+device.dtype.diff_size-1)
    alg0 = psd.diff_dim + device.alg_ptr
    append!(psd.offline_alg_rows, alg0:alg0+device.dtype.alg_size-1)

    # Find the device's table-local index and set online[k] = false.
    dtype = device.dtype
    L = psd.layout::SimulationLayout
    k = 0
    for (i, d) in enumerate(psd.devices)
        typeof(d.dtype) == typeof(dtype) || continue
        k += 1
        if i == device_idx
            _set_table_online!(L, dtype, k, false)
            return nothing
        end
    end
    error("Device $device_idx not found in its table")
end

# Dispatch helper to set online[k] on the correct table.
_set_table_online!(L::SimulationLayout, ::Genrou,           k::Int, v::Bool) = (L.genrou.online[k] = v; nothing)
_set_table_online!(L::SimulationLayout, ::IEESGO,           k::Int, v::Bool) = (L.ieesgo.online[k] = v; nothing)
_set_table_online!(L::SimulationLayout, ::TGOV1,            k::Int, v::Bool) = (L.tgov1.online[k] = v; nothing)
_set_table_online!(L::SimulationLayout, ::SEXS,             k::Int, v::Bool) = (L.sexs.online[k] = v; nothing)
_set_table_online!(L::SimulationLayout, ::ESDC1A,           k::Int, v::Bool) = (L.esdc1a.online[k] = v; nothing)
_set_table_online!(L::SimulationLayout, ::IEEEST,           k::Int, v::Bool) = (L.ieeest.online[k] = v; nothing)
_set_table_online!(L::SimulationLayout, ::ZIPLoad,          k::Int, v::Bool) = (L.zipload.online[k] = v; nothing)
_set_table_online!(L::SimulationLayout, ::StaticGenerator,  k::Int, v::Bool) = (L.static_gen.online[k] = v; nothing)

# Device contract registry. Included AFTER kernels/*.jl so device structs
# (Genrou, IEESGO, TGOV1, SEXS, ESDC1A, ZIPLoad, StaticGenerator) are in
# scope for the seed registrations.
include("devices/registry.jl")
include("devices/seeds.jl")

# 2D scenario-indexed batched layout + batched CPU integration.
include("batched.jl")

# Optional optimization entry point (implementation in GradPowerNLPExt).
include("nlp.jl")

# Include files. parsers.
include("parse.jl")

# Display functions.
include("queries.jl")
include("interactions.jl")
include("plotting.jl")
include("display.jl")

# Exports
export Bus, Gen, Load, Branch, Shunt, PowerSystem
export build_network!
export runpf, runpf!
export LoadingDirection, runcpf, ContinuationResult, ContinuationPoint, ContinuationEvent
export ContinuationStats
export DynamicDevice, PowerSystemDynamics
export DynamicProblem, ContingencyEvent, DisconnectDeviceEvent, TripLineEvent, LoadStepEvent
export add_load_step_event!, create_load_step_event, find_load_device
export add_load_trip!, add_generator_trip!, generator_devices, find_branch
export add_device!
export add_event!, add_disconnect_event!, add_trip_event!, create_trip_line_event
export initialize_dynamics!, integrate!
export SimulationResult, StateChannel, state_channels, state, speed, bus_voltage
export bus_index, get_bus, find_generator, find_load, find_devices, find_device
export state_index, parameter_index, get_parameter, set_parameter!
export system_summary, display_parameters, get_bus_info, get_index_info
export set_load!, set_generator!
export plot_states, plot_voltage, plot_speed, plot_network, animate_network
export plot_cpf, animate_cpf
export FixedStateLimit, LimitEvent, LimitWorkspace
export set_dynamics!
export from_psse
export get_device_name, get_diff_names, get_alg_names, get_param_names
export DyrRecordCoverage, DyrModelCoverage, DyrCoverageReport
export analyze_dyr_coverage, native_dyr_models, coverage_counts, coverage_by_model, native_coverage
export SolverLog
export AbstractDirectSolverBackend, KLUBackend, CUDSSBackend, SparseDirectSolverBackend

end # module GradPower
