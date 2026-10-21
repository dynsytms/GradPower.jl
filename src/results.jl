"""
    StateChannel

Identity of one saved raw state row. `index` is its original simulation row,
not a persistent identity: use `(model, bus, device_id, name)` across layouts.
`bus` is an external PSS/E number, `kind` is `:differential` or `:algebraic`,
and `online` records the device status at the start of the simulation.
"""
struct StateChannel
    index::Int
    model::Symbol
    bus::Int
    device_id::String
    name::Symbol
    kind::Symbol
    online::Bool
end

function _result_device_id(ps, dtype)
    if dtype isa StaticGenerator
        ids = sort!([_normalize_id(ps.gens[j].id) for j in dtype.gen_idxs])
        return length(ids) == 1 ? only(ids) : "[" * join(repr.(ids), ",") * "]"
    end
    return _normalize_id(dtype.id)
end

"""
    state_channels(ps; save_idxs=:) -> Vector{StateChannel}

Describe raw state rows in recording order, including network `:bus` states
`:vr` and `:vi`. Duplicate/subset indices follow the recorder exactly.
Static generator aggregates use a sorted, quoted list of member IDs as their
device ID (a single member uses its own ID).
"""
function state_channels(ps::PowerSystem; save_idxs=Colon())
    dyn = ps.dynamic
    dyn === nothing && throw(ArgumentError("system has no dynamics"))
    ndev = dyn.diff_dim + dyn.alg_dim
    channels = Vector{StateChannel}(undef, ndev + 2length(ps.buses))
    for (i, dev) in enumerate(dyn.devices)
        dtype = dev.dtype
        model = Symbol(lowercase(get_device_name(dtype)))
        bus = ps.buses[dyn.map.bus[i]].i
        id = _result_device_id(ps, dtype)
        for (kind, names, start) in (
            (:differential, get_diff_names(dtype), dev.diff_ptr),
            (:algebraic, get_alg_names(dtype), dyn.diff_dim + dev.alg_ptr))
            for (j, name) in enumerate(names)
                row = start + j - 1
                channels[row] = StateChannel(row, model, bus, id, Symbol(name), kind, dev.online)
            end
        end
    end
    for (i, bus) in enumerate(ps.buses), (j, name) in enumerate((:vr, :vi))
        row = ndev + 2(i - 1) + j
        channels[row] = StateChannel(row, :bus, bus.i, "", name, :algebraic, true)
    end
    idxs = _resolve_save_idxs(save_idxs, length(channels))
    return channels[idxs]
end

"""
    SimulationResult(times, trajectories, channels; scenario_ids, status, events,
                     metadata, storage)
    integrate!(dp, ps, tf; result=true, ...)

Named simulation output. Integrators retain their tuple return by default;
`result=true` opts in. `trajectories` is a matrix for a single simulation or a
vector of matrices for batched simulations. Rows follow `channels`, columns
follow `times`. Arrays are not copied. `storage` is an optional caller-owned
storage handle; no file ownership or automatic closing is implied.

`status` is one symbol per scenario (`:success` or `:terminated` for the built-in
CPU integrators). Failed solves still throw and do not return partial results.
`events` describes applied scheduled events, not merely requested events.
Tuple destructuring (`times, trajectories = result`) is also supported.
"""
struct SimulationResult{T,A,I,E,M,S}
    times::T
    trajectories::A
    channels::Vector{StateChannel}
    scenario_ids::I
    status::Vector{Symbol}
    events::E
    metadata::M
    storage::S
end

function SimulationResult(times::AbstractVector, trajectories,
                          channels::AbstractVector{StateChannel};
                          scenario_ids=nothing, status=nothing, events=NamedTuple[],
                          metadata=(;), storage=nothing)
    mats = trajectories isa AbstractMatrix ? (trajectories,) : trajectories
    isempty(mats) && throw(ArgumentError("at least one scenario is required"))
    all(a -> a isa AbstractMatrix && size(a) == (length(channels), length(times)), mats) ||
        throw(DimensionMismatch("trajectory rows/columns must match channels/times"))
    ids = scenario_ids === nothing ? collect(1:length(mats)) : collect(scenario_ids)
    length(ids) == length(mats) || throw(DimensionMismatch("scenario_ids must match scenarios"))
    allunique(ids) || throw(ArgumentError("scenario_ids must be unique"))
    statuses = status === nothing ? fill(:unknown, length(mats)) : collect(Symbol, status)
    length(statuses) == length(mats) || throw(DimensionMismatch("status must match scenarios"))
    return SimulationResult(times, trajectories, collect(channels), ids, statuses,
                            events, metadata, storage)
end

Base.length(::SimulationResult) = 2
Base.iterate(r::SimulationResult, i::Int=1) =
    i == 1 ? (r.times, 2) : i == 2 ? (r.trajectories, 3) : nothing

function _scenario_trajectory(r::SimulationResult, scenario)
    m = findfirst(isequal(scenario), r.scenario_ids)
    m === nothing && throw(ArgumentError("scenario $(repr(scenario)) not recorded"))
    return r.trajectories isa AbstractMatrix ? r.trajectories : r.trajectories[m]
end

"""
    state(result, model, bus, device_id, name; scenario=first(result.scenario_ids))
    state(result, model, k, name; scenario=first(result.scenario_ids))

View of a raw saved state. Explicit identity uses external `bus` and normalized
PSS/E `device_id`. The short form selects the `k`th recorded device of `model`,
sorted by `(bus, device_id)`, not by solver layout. Model symbols are lowercase;
state names retain their declared case. Missing rows raise `ArgumentError`.
"""
function state(r::SimulationResult, model::Symbol, bus::Integer,
               device_id::AbstractString, name::Symbol; scenario=first(r.scenario_ids))
    key = (model, bus, device_id, name)
    rows = findall(c -> c.model == model && c.bus == bus && c.name == name &&
                  (c.device_id == device_id || c.device_id == _normalize_id(device_id)),
                  r.channels)
    isempty(rows) && throw(ArgumentError("state $key was not recorded"))
    all(i -> r.channels[i].index == r.channels[first(rows)].index, rows) ||
        throw(ArgumentError("state identity $key is ambiguous"))
    # Duplicate save_idxs name the same state; use its first recorded copy.
    return view(_scenario_trajectory(r, scenario), first(rows), :)
end

function state(r::SimulationResult, model::Symbol, k::Integer, name::Symbol;
               scenario=first(r.scenario_ids))
    devices = sort!(unique([(c.bus, c.device_id) for c in r.channels if c.model == model]))
    1 <= k <= length(devices) || throw(ArgumentError("no recorded $model device $k"))
    bus, id = devices[k]
    return state(r, model, bus, id, name; scenario)
end

"""
    speed(result, k; model=:genrou, scenario=first(result.scenario_ids))
    speed(result, bus, device_id; model=:genrou, scenario=first(result.scenario_ids))

View of raw `:w`: per-unit speed deviation, not absolute speed or Hz.
"""
speed(r::SimulationResult, k::Integer; model::Symbol=:genrou, kwargs...) =
    state(r, model, k, :w; kwargs...)
speed(r::SimulationResult, bus::Integer, id::AbstractString; model::Symbol=:genrou, kwargs...) =
    state(r, model, bus, id, :w; kwargs...)

"""
    bus_voltage(result, external_bus; component=:complex, scenario=first(result.scenario_ids))

Network voltage in pu. `:real` and `:imag` return raw views; `:complex`,
`:magnitude` and `:angle` allocate derived vectors (angle in radians).
Both raw components must have been recorded for derived output.
"""
function bus_voltage(r::SimulationResult, bus::Integer; component::Symbol=:complex,
                     scenario=first(r.scenario_ids))
    component in (:real, :imag, :complex, :magnitude, :angle) ||
        throw(ArgumentError("unknown voltage component $component"))
    component === :real && return state(r, :bus, bus, "", :vr; scenario)
    component === :imag && return state(r, :bus, bus, "", :vi; scenario)
    vr = state(r, :bus, bus, "", :vr; scenario)
    vi = state(r, :bus, bus, "", :vi; scenario)
    component === :magnitude && return hypot.(vr, vi)
    component === :angle && return atan.(vi, vr)
    return complex.(vr, vi)
end

# Capture metadata before callbacks or events can mutate the system. The event
# schedule is quantized exactly as in the fixed-step integrators.
function _result_context(ps, save_idxs, M, scenario_ids, dt, metadata;
                         system_events::Bool=true)
    ids = scenario_ids === nothing ? collect(1:M) : collect(scenario_ids)
    length(ids) == M || throw(DimensionMismatch("scenario_ids must match scenarios"))
    allunique(ids) || throw(ArgumentError("scenario_ids must be unique"))
    dyn = ps.dynamic
    events = NamedTuple[]
    for ev in (system_events ? dyn.events : ())
        details = (bus=ps.buses[ev.bus].i, rfault=ev.rfault, xfault=ev.xfault)
        push!(events, (step=round(Int, ev.ton / dt), action=:fault_on, details))
        push!(events, (step=round(Int, ev.toff / dt), action=:fault_off, details))
    end
    for ev in (system_events ? dyn.disconnect_events : ())
        dev = dyn.devices[ev.device_idx]
        push!(events, (step=round(Int, ev.ton / dt), action=:disconnect,
                       details=(bus=ps.buses[dyn.map.bus[ev.device_idx]].i,
                                model=Symbol(lowercase(get_device_name(dev.dtype))),
                                device_id=_result_device_id(ps, dev.dtype))))
    end
    for ev in (system_events ? dyn.trip_events : ())
        details = (from_bus=ps.buses[ev.from_bus].i, to_bus=ps.buses[ev.to_bus].i,
                   branch=ev.branch)
        push!(events, (step=round(Int, ev.ton / dt), action=:trip, details))
        isfinite(ev.toff) &&
            push!(events, (step=round(Int, ev.toff / dt), action=:reclose, details))
    end
    for ev in (system_events ? dyn.load_step_events : ())
        push!(events, (step=round(Int, ev.t / dt), action=:load_step,
                       details=(bus=ps.buses[dyn.map.bus[ev.device_idx]].i,
                                device_id=_normalize_id(dyn.devices[ev.device_idx].dtype.id),
                                scale=ev.scale)))
    end
    sort!(events, by=e -> e.step)
    return (; channels=state_channels(ps; save_idxs), ids, events, dt, metadata,
            applied_steps=Int[])
end

_mark_result_event!(::Nothing, step) = nothing
_mark_result_event!(context, step) = push!(context.applied_steps, step)

function _result_output(output, context; status=:success, final_time,
                        limit_events=LimitEvent[], limit_event_start::Int=0)
    context === nothing && return output
    events = [(action=e.action, time=e.step * context.dt, details=e.details)
              for e in context.events if e.step in context.applied_steps]
    metadata = merge(context.metadata,
                     (; final_time, sample_phase=:before_event_resolve,
                        limit_events=limit_events[limit_event_start+1:end]))
    return SimulationResult(output..., context.channels; scenario_ids=context.ids,
                            status=fill(status, length(context.ids)), events, metadata)
end
