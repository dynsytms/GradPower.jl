_plot_backend_error() = throw(ArgumentError(
    "Plotting requires an optional Makie backend; install CairoMakie and run `using CairoMakie` first."))

"""
    plot_states(result, selectors; scenarios=nothing, figure_kwargs=(;), axis_kwargs=(;))

Plot selected raw states, one axis per state. Each selector is a `StateChannel`
or `(model, external_bus, device_id, state_name)`. `scenarios` selects recorded
scenario IDs, not ordinal positions; `nothing` selects all. Requires loading
an optional Makie backend, e.g. `using CairoMakie`. Returns a Makie Figure.
"""
plot_states(args...; kwargs...) = _plot_backend_error()

"""
    plot_voltage(result; buses=nothing, scenarios=nothing, envelope=false,
                 component=:magnitude, figure_kwargs=(;), axis_kwargs=(;))

Plot recorded network voltages at external bus numbers. Components are
`:magnitude`, `:real`, `:imag` (pu) and `:angle` (radians).
With `envelope=true`, show a pointwise min/max band and arithmetic mean across
the selected scenario IDs, separately for each bus, not a confidence interval.
Defaults select all recorded buses/scenarios. Missing required rows are errors.
Requires an optional Makie backend; returns a Makie Figure.
"""
plot_voltage(args...; kwargs...) = _plot_backend_error()

"""
    plot_cpf(result; buses=nothing, figure_kwargs=(;), axis_kwargs=(;))

Plot voltage magnitude versus loading factor in accepted continuation order.
`buses` selects recorded external bus numbers; defaults to all recorded buses.
Control exchanges and a verified terminal maximum are marked. Requires an
optional Makie backend. Returns a Figure. This is not a time-domain plot.
"""
plot_cpf(args...; kwargs...) = _plot_backend_error()

"""
    animate_cpf(path, result; buses=nothing, framerate=12,
                figure_kwargs=(;), axis_kwargs=(;))

Animate accepted continuation points in tracing order, one point per frame.
Axes remain fixed over the recorded branch. Playback speed is not physical
simulation time; steps are not equally spaced in load or arclength. Requires
an optional Makie backend and at least one recorded bus/point. Returns `path`.
"""
animate_cpf(args...; kwargs...) = _plot_backend_error()

function _plot_cpf_indices(r::ContinuationResult, buses)
    isempty(r.points) && throw(ArgumentError("continuation result has no accepted points"))
    selected = buses === nothing ? r.bus_numbers : buses
    selected isa AbstractVector || throw(ArgumentError("buses must be a vector of recorded external numbers"))
    isempty(selected) && throw(ArgumentError("select at least one recorded bus"))
    indices = Int[]
    for bus in selected
        bus isa Integer && !(bus isa Bool) || throw(ArgumentError("bus numbers must be integers"))
        i = findfirst(==(bus),r.bus_numbers)
        i === nothing && throw(ArgumentError("bus $bus was not recorded"))
        i in indices && throw(ArgumentError("duplicate bus $bus"))
        push!(indices,i)
    end
    return indices
end

"""
    plot_speed(result; generators=nothing, scenarios=nothing, envelope=false,
               unit=:pu, nominal_frequency=60.0, model=:genrou,
               figure_kwargs=(;), axis_kwargs=(;))

Plot raw speed deviation `w`, selecting machines by `(external_bus, device_id)`.
`:pu` is per-unit deviation; `:hz` multiplies it by `nominal_frequency` and is
frequency deviation in Hz, not absolute frequency. Defaults select all recorded
machines/scenarios. Envelopes are pointwise min/max plus arithmetic mean.
Requires an optional Makie backend; returns a Makie Figure.
"""
plot_speed(args...; kwargs...) = _plot_backend_error()

"""
    plot_network(sys, result=nothing; scenario=nothing, frame=nothing,
                 positions=nothing, quantity=:voltage, labels=true,
                 colorrange=nothing, colormap=:viridis,
                 figure_kwargs=(;), axis_kwargs=(;))

Plot static branch connectivity colored by voltage magnitude (pu) or `:angle`
(radians). Without a result, use the system's PF voltages. Otherwise use a
saved frame index (last by default) and scenario ID (first by default).
`positions` maps every external bus number to an `(x,y)` pair; the default is
a deterministic circle ordered by external number, not a geographic one-line.
Branch outages are not inferred from event history. Reads only the selected
frame's voltage entries. Requires a Makie backend; returns a Makie Figure.
"""
plot_network(args...; kwargs...) = _plot_backend_error()

"""
    animate_network(path, sys, result; scenario=nothing, frames=nothing,
                    framerate=20, positions=nothing, quantity=:voltage,
                    labels=true, colorrange=nothing, colormap=:viridis,
                    figure_kwargs=(;), axis_kwargs=(;))

Record saved network voltage frames using an optional Makie backend. Frames
are strictly increasing saved-column indices, defaulting to all; playback
`framerate` must be a positive integer and does not resample simulation time.
The color range is fixed over the selected frames. Topology/positions follow
`plot_network`; no event outages are inferred. Returns `path`. File
format/encoder support comes from Makie.
"""
animate_network(args...; kwargs...) = _plot_backend_error()

function _plot_times(r::SimulationResult)
    isempty(r.times) && throw(ArgumentError("cannot plot a result without saved times"))
    all(t -> t isa Real && isfinite(t), r.times) ||
        throw(ArgumentError("saved times must be finite real numbers"))
    all(i -> r.times[i] > r.times[i-1], 2:length(r.times)) ||
        throw(ArgumentError("saved times must be strictly increasing"))
    return r.times
end

function _plot_scenarios(r, scenarios)
    scenarios === nothing && return collect(r.scenario_ids)
    (scenarios isa AbstractString || scenarios isa Symbol || scenarios isa Number) &&
        throw(ArgumentError("scenarios must be a collection of recorded IDs"))
    ids = collect(scenarios)
    isempty(ids) && throw(ArgumentError("select at least one scenario"))
    allunique(ids) || throw(ArgumentError("scenario selection contains duplicates"))
    for id in ids
        any(isequal(id), r.scenario_ids) ||
            throw(ArgumentError("scenario $(repr(id)) was not recorded"))
    end
    return ids
end

function _plot_values(values)
    all(x -> x isa Real && isfinite(x), values) ||
        throw(ArgumentError("selected plot values must be finite real numbers"))
    return values
end

function _plot_model(model)
    key = _query_model(model)
    key isa Symbol || throw(ArgumentError("plot selectors require a model name, not a type"))
    return key
end

function _plot_selector(selector)
    selector isa StateChannel &&
        return (selector.model, selector.bus, selector.device_id, selector.name)
    selector isa Tuple && length(selector) == 4 &&
        selector[2] isa Integer && selector[3] isa AbstractString ||
        throw(ArgumentError("state selector must be a StateChannel or (model, bus, id, name)"))
    return (_plot_model(selector[1]), Int(selector[2]), String(selector[3]),
            _query_name(selector[4]))
end

function _plot_state_units(model, name)
    model == :genrou && name == :w && return "Speed deviation (pu)"
    model == :genrou && name == :delta && return "Rotor angle (rad)"
    model == :bus && name in (:vr, :vi) && return "Voltage (pu)"
    return "State value (model units)"
end

function _plot_states_data(r, selectors; scenarios=nothing)
    _plot_times(r)
    ids = _plot_scenarios(r, scenarios)
    single = selectors isa StateChannel ||
             (selectors isa Tuple && length(selectors) == 4 &&
              (first(selectors) isa Symbol || first(selectors) isa AbstractString))
    chosen = single ? [selectors] : collect(selectors)
    isempty(chosen) && throw(ArgumentError("select at least one state"))
    return map(chosen) do selector
        model, bus, id, name = _plot_selector(selector)
        series = [(scenario=s, values=_plot_values(state(r, model, bus, id, name; scenario=s)))
                  for s in ids]
        label = "$model bus $bus id $(repr(id)): $name"
        return (; label, ylabel=_plot_state_units(model, name), series)
    end
end

function _plot_curves(r, entities, ids, getter, envelope)
    envelope isa Bool || throw(ArgumentError("envelope must be a Bool"))
    curves = NamedTuple[]
    for (identity, label) in entities
        if envelope
            initial = _plot_values(getter(identity, first(ids)))
            lower, upper, mean_values = Float64.(initial), Float64.(initial), Float64.(initial)
            for (n, scenario) in enumerate(Iterators.drop(ids, 1))
                values = _plot_values(getter(identity, scenario))
                for j in eachindex(values)
                    value = Float64(values[j])
                    lower[j] = min(lower[j], value)
                    upper[j] = max(upper[j], value)
                    mean_values[j] += (value - mean_values[j]) / (n + 1)
                end
            end
            _plot_values(mean_values)
            push!(curves, (; label, values=mean_values, lower, upper))
        else
            for scenario in ids
                values = _plot_values(getter(identity, scenario))
                push!(curves, (label="$label, scenario $(repr(scenario))",
                               values, lower=nothing, upper=nothing))
            end
        end
    end
    return curves
end

function _plot_buses(r, buses)
    chosen = buses === nothing ?
        sort!(unique(c.bus for c in r.channels if c.model == :bus)) :
        buses isa Integer ? [buses] : collect(buses)
    isempty(chosen) && throw(ArgumentError("select at least one recorded bus"))
    all(b -> b isa Integer, chosen) || throw(ArgumentError("buses must be external integers"))
    allunique(chosen) || throw(ArgumentError("bus selection contains duplicates"))
    return Int.(chosen)
end

function _plot_voltage_data(r; buses=nothing, scenarios=nothing,
                            envelope=false, component=:magnitude)
    _plot_times(r)
    component in (:magnitude, :real, :imag, :angle) ||
        throw(ArgumentError("voltage component must be :magnitude, :real, :imag, or :angle"))
    ids = _plot_scenarios(r, scenarios)
    entities = [(b, "Bus $b") for b in _plot_buses(r, buses)]
    getter = (bus, scenario) -> bus_voltage(r, bus; component, scenario)
    ylabel = component == :angle ? "Voltage angle (rad)" :
             component == :magnitude ? "Voltage magnitude (pu)" :
             component == :real ? "Real voltage (pu)" : "Imaginary voltage (pu)"
    return (; ylabel, curves=_plot_curves(r, entities, ids, getter, envelope))
end

function _plot_generators(r, model, generators)
    single = generators isa Tuple && length(generators) == 2 &&
             first(generators) isa Integer
    chosen = generators === nothing ?
        sort!(unique((c.bus, c.device_id) for c in r.channels if c.model == model && c.name == :w)) :
        single ? [generators] : collect(generators)
    isempty(chosen) && throw(ArgumentError("select at least one recorded machine"))
    all(g -> g isa Tuple && length(g) == 2 && g[1] isa Integer &&
             g[2] isa AbstractString, chosen) ||
        throw(ArgumentError("generators must be (external_bus, device_id) pairs"))
    keys = [(Int(g[1]), _normalize_id(g[2])) for g in chosen]
    allunique(keys) || throw(ArgumentError("generator selection contains duplicates"))
    return keys
end

function _plot_speed_data(r; generators=nothing, scenarios=nothing, envelope=false,
                          unit=:pu, nominal_frequency=60.0, model=:genrou)
    _plot_times(r)
    unit in (:pu, :hz) || throw(ArgumentError("speed unit must be :pu or :hz"))
    nominal_frequency isa Real && isfinite(nominal_frequency) && nominal_frequency > 0 ||
        throw(ArgumentError("nominal_frequency must be finite and positive"))
    key = _plot_model(model)
    ids = _plot_scenarios(r, scenarios)
    entities = [(g, "$key bus $(g[1]) id $(repr(g[2]))")
                for g in _plot_generators(r, key, generators)]
    getter = function (g, scenario)
        values = speed(r, g...; model=key, scenario)
        return unit == :pu ? values : nominal_frequency .* values
    end
    ylabel = unit == :pu ? "Speed deviation (pu)" : "Frequency deviation (Hz)"
    return (; ylabel, curves=_plot_curves(r, entities, ids, getter, envelope))
end

function _plot_positions(buses, positions)
    n = length(buses)
    if positions === nothing
        order = sortperm(buses)
        points = Vector{Tuple{Float64,Float64}}(undef, n)
        for (rank, i) in enumerate(order)
            angle = 2pi * (rank - 1) / n
            points[i] = n == 1 ? (0.0, 0.0) : (cos(angle), sin(angle))
        end
        return points
    end
    positions isa AbstractDict ||
        throw(ArgumentError("positions must map external bus numbers to (x,y)"))
    return map(buses) do bus
        haskey(positions, bus) || throw(ArgumentError("no position for external bus $bus"))
        point = positions[bus]
        (point isa Tuple || point isa AbstractVector) && length(point) == 2 &&
            all(x -> x isa Real && isfinite(x), point) ||
            throw(ArgumentError("position for bus $bus must be a finite (x,y) pair"))
        return (Float64(point[1]), Float64(point[2]))
    end
end

function _plot_frame(r, frame)
    frame === nothing && return length(r.times)
    frame isa Integer && !(frame isa Bool) && 1 <= frame <= length(r.times) ||
        throw(ArgumentError("frame must be a saved-column index between 1 and $(length(r.times))"))
    return Int(frame)
end

function _plot_network_data(ps, r=nothing; kwargs...)
    r === nothing || _plot_times(r)
    return _plot_network_frame_data(ps, r; kwargs...)
end

# Animation validates its time grid once before repeatedly reading frames.
function _plot_network_frame_data(ps, r=nothing; scenario=nothing, frame=nothing,
                                  positions=nothing, quantity=:voltage)
    quantity in (:voltage, :angle) || throw(ArgumentError("quantity must be :voltage or :angle"))
    buses = [b.i for b in ps.buses]
    isempty(buses) && throw(ArgumentError("cannot plot an empty network"))
    points = _plot_positions(buses, positions)
    edges = [(b.fr, b.to) for b in ps.branches]
    all(e -> all(i -> 1 <= i <= length(buses), e), edges) ||
        throw(ArgumentError("branch endpoint is not an internal bus index"))
    label = quantity == :voltage ? "Voltage magnitude (pu)" : "Voltage angle (rad)"
    if r === nothing
        (scenario === nothing && frame === nothing) ||
            throw(ArgumentError("scenario/frame require a SimulationResult"))
        values = [quantity == :voltage ? b.v0m : b.v0a for b in ps.buses]
        return (; buses, positions=points, edges, values=_plot_values(values), label, time=nothing)
    end
    k = _plot_frame(r, frame)
    id = scenario === nothing ? first(r.scenario_ids) : scenario
    trajectory = _scenario_trajectory(r, id)
    # Resolve saved-row positions, not original solver indices. Duplicate saves
    # are valid only when they identify the same original state.
    rows = Dict{Tuple{Int,Symbol},Int}()
    for (i, c) in enumerate(r.channels)
        c.model == :bus && c.device_id == "" && c.name in (:vr, :vi) || continue
        key = (c.bus, c.name)
        if haskey(rows, key)
            r.channels[rows[key]].index == c.index ||
                throw(ArgumentError("network state identity $key is ambiguous"))
        else
            rows[key] = i
        end
    end
    values = map(buses) do bus
        vr_row = get(rows, (bus, :vr), 0)
        vi_row = get(rows, (bus, :vi), 0)
        vr_row > 0 && vi_row > 0 ||
            throw(ArgumentError("both voltage components at external bus $bus must be recorded"))
        vr, vi = trajectory[vr_row, k], trajectory[vi_row, k]
        _plot_values((vr, vi))
        return quantity == :voltage ? Float64(hypot(vr, vi)) : Float64(atan(vi, vr))
    end
    return (; buses, positions=points, edges, values, label, time=Float64(r.times[k]))
end

function _plot_animation_frames(r, frames, framerate)
    _plot_times(r)
    framerate isa Integer && !(framerate isa Bool) && framerate > 0 ||
        throw(ArgumentError("framerate must be a positive integer"))
    chosen = frames === nothing ? collect(eachindex(r.times)) : collect(frames)
    isempty(chosen) && throw(ArgumentError("select at least one animation frame"))
    all(frame -> frame isa Integer && !(frame isa Bool), chosen) ||
        throw(ArgumentError("animation frames must be saved-column integer indices"))
    indices = [_plot_frame(r, frame) for frame in chosen]
    all(i -> indices[i] > indices[i-1], 2:length(indices)) ||
        throw(ArgumentError("animation frame indices must be strictly increasing"))
    return indices
end
