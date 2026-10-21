function _query_numbering(numbering)
    numbering in (:external, :internal) && numbering isa Symbol ||
        throw(ArgumentError("numbering must be :external or :internal"))
    return numbering
end

"""
    bus_index(ps, number; numbering=:external)

Resolve an external bus number or explicit internal index to an internal index.
Missing buses and invalid numbering modes raise `ArgumentError`.
"""
function bus_index(ps::PowerSystem, number::Integer; numbering=:external)
    _query_numbering(numbering)
    if numbering === :internal
        1 <= number <= length(ps.buses) ||
            throw(ArgumentError("internal bus index $number is out of range"))
        return Int(number)
    end
    haskey(ps.busmap, number) || throw(ArgumentError("bus $number not in the case"))
    return ps.busmap[number]
end

"""
    get_bus(ps, number; numbering=:external) -> Bus

Return the bus selected by external number or explicit internal index.
"""
get_bus(ps::PowerSystem, number::Integer; numbering=:external) =
    ps.buses[bus_index(ps, number; numbering)]

function _query_unique(indices, description)
    isempty(indices) && throw(ArgumentError("no matching $description"))
    length(indices) == 1 ||
        throw(ArgumentError("$description is ambiguous ($(length(indices)) matches)"))
    return only(indices)
end

function _query_static(rows, ps, bus, id, numbering, description)
    b = bus_index(ps, bus; numbering)
    key = _normalize_id(id)
    return _query_unique(findall(r -> r.bus == b && _normalize_id(r.id) == key, rows),
                         "$description $(repr(key)) at bus $(ps.buses[b].i)")
end

"""
    find_generator(ps, bus, id; numbering=:external)

Return the unique static generator row index, comparing normalized PSS/E IDs.
Missing or ambiguous identities raise `ArgumentError`.
"""
find_generator(ps::PowerSystem, bus::Integer, id::AbstractString; numbering=:external) =
    _query_static(ps.gens, ps, bus, id, numbering, "generator")

"""
    find_load(ps, bus, id; numbering=:external)

Return the unique static load row index, comparing normalized PSS/E IDs.
Missing or ambiguous identities raise `ArgumentError`.
"""
find_load(ps::PowerSystem, bus::Integer, id::AbstractString; numbering=:external) =
    _query_static(ps.loads, ps, bus, id, numbering, "load")

function _query_model(model)
    if model isa Symbol || model isa AbstractString
        name = Symbol(lowercase(String(model)))
        name === :static_gen && return :staticgenerator
        name === :gensal && return :genrou
        return name
    elseif model isa Type && model <: AbstractDeviceType
        return model
    end
    throw(ArgumentError("model must be a Symbol, string, or AbstractDeviceType subtype"))
end

_query_model_matches(dtype, model::Symbol) =
    Symbol(lowercase(get_device_name(dtype))) === model
_query_model_matches(dtype, model::Type) = dtype isa model

# Preserve aggregate IDs verbatim: normalizing their quoted member list loses identity.
_query_id_matches(actual, requested) =
    actual == requested || actual == _normalize_id(requested)

"""
    find_devices(ps; model=nothing, bus=nothing, id=nothing,
                 numbering=:external, online=nothing)

Return dynamic device indices matching all supplied filters. Models are
case-insensitive names (including `:gensal` and `:static_gen` aliases) or device
types. IDs follow `state_channels`, including static-generator aggregate IDs.
"""
function find_devices(ps::PowerSystem; model=nothing, bus=nothing, id=nothing,
                      numbering=:external, online=nothing)
    _query_numbering(numbering)
    selector = model === nothing ? nothing : _query_model(model)
    b = bus === nothing ? nothing : bus_index(ps, bus; numbering)
    id === nothing || id isa AbstractString ||
        throw(ArgumentError("id must be a string or nothing"))
    online === nothing || online isa Bool ||
        throw(ArgumentError("online must be a Bool or nothing"))
    dyn = ps.dynamic
    dyn === nothing && return Int[]
    dyn.map === nothing && throw(ArgumentError("dynamic bus map is not initialized"))
    return findall(eachindex(dyn.devices)) do i
        dev = dyn.devices[i]
        (selector === nothing || _query_model_matches(dev.dtype, selector)) &&
        (b === nothing || dyn.map.bus[i] == b) &&
        (id === nothing || _query_id_matches(_result_device_id(ps, dev.dtype), id)) &&
        (online === nothing || dev.online == online)
    end
end

"""
    find_device(ps, model, bus, id; numbering=:external)

Return a unique dynamic device index; missing or ambiguous matches are errors.
"""
function find_device(ps::PowerSystem, model, bus::Integer, id::AbstractString;
                     numbering=:external)
    return _query_unique(find_devices(ps; model, bus, id, numbering), "dynamic device")
end

function _query_name(name)
    name isa Symbol || name isa AbstractString ||
        throw(ArgumentError("name must be a Symbol or string"))
    return Symbol(name)
end

"""
    state_index(ps, model, bus, id, name; numbering=:external)

Return the unique current `StateChannel.index`. State names are case-sensitive;
network voltages use model `:bus`, an empty ID, and `:vr` or `:vi`.
"""
function state_index(ps::PowerSystem, model, bus::Integer, id::AbstractString, name;
                     numbering=:external)
    b = bus_index(ps, bus; numbering)
    selector = _query_model(model)
    key = _query_name(name)
    models = if selector === :bus
        Set([:bus])
    else
        indices = find_devices(ps; model, bus, id, numbering)
        Set(Symbol(lowercase(get_device_name(ps.dynamic.devices[i].dtype))) for i in indices)
    end
    channels = state_channels(ps)
    hits = filter(c -> c.model in models && c.bus == ps.buses[b].i &&
                       _query_id_matches(c.device_id, id) && c.name === key, channels)
    return _query_unique([c.index for c in hits], "state $(repr(key))")
end

"""
    parameter_index(ps, device_index, name)
    parameter_index(ps, model, bus, id, name; numbering=:external)

Return the current parameter-vector index. Parameter names are case-sensitive
Symbols or strings; missing or ambiguous names raise `ArgumentError`.
"""
function parameter_index(ps::PowerSystem, device_index::Integer, name)
    dyn = ps.dynamic
    dyn === nothing && throw(ArgumentError("system has no dynamics"))
    dev = dyn.devices[device_index]
    key = _query_name(name)
    offsets = findall(n -> Symbol(n) === key, get_param_names(dev.dtype))
    return dev.par_ptr + _query_unique(offsets, "parameter $(repr(key))") - 1
end

function parameter_index(ps::PowerSystem, model, bus::Integer, id::AbstractString, name;
                         numbering=:external)
    return parameter_index(ps, find_device(ps, model, bus, id; numbering), name)
end

"""
    system_summary(ps; section=:buses) -> Vector{NamedTuple}

Return rows for `:buses`, `:generators`, `:loads`, or dynamic `:devices`.
Rows contain `index`, external `bus`, and `id` plus section-specific data.
Device IDs are normalized; bus names are preserved. Device rows include model,
online status, and current vector pointers.
"""
function system_summary(ps::PowerSystem; section=:buses)
    if section === :buses
        return NamedTuple[(index=i, bus=b.i, id=b.id, type=b.type, baseKV=b.baseKV,
                           v0m=b.v0m, v0a=b.v0a) for (i, b) in enumerate(ps.buses)]
    elseif section === :generators
        return NamedTuple[(index=i, bus=ps.buses[g.bus].i, id=_normalize_id(g.id),
                           psch=g.psch, qsch=g.qsch, mbase=g.mbase, status=g.status,
                           qmin=g.qmin, qmax=g.qmax) for (i, g) in enumerate(ps.gens)]
    elseif section === :loads
        return NamedTuple[(index=i, bus=ps.buses[l.bus].i, id=_normalize_id(l.id),
                           pd=l.pd, qd=l.qd) for (i, l) in enumerate(ps.loads)]
    elseif section === :devices
        dyn = ps.dynamic
        dyn === nothing && return NamedTuple[]
        dyn.map === nothing && throw(ArgumentError("dynamic bus map is not initialized"))
        return NamedTuple[(index=i, model=Symbol(lowercase(get_device_name(d.dtype))),
                           bus=ps.buses[dyn.map.bus[i]].i, id=_result_device_id(ps, d.dtype),
                           online=d.online, diff_ptr=d.diff_ptr, alg_ptr=d.alg_ptr,
                           ctrl_ptr=d.ctrl_ptr, par_ptr=d.par_ptr)
                          for (i, d) in enumerate(dyn.devices)]
    end
    throw(ArgumentError("section must be :buses, :generators, :loads, or :devices"))
end
