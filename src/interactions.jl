"""
    get_parameter(sys, device_index, name)
    get_parameter(dp, sys, device_index, name)
    get_parameter([dp,] sys, model, bus, id, name; numbering=:external)

Read a named device parameter, using the device configuration or, when `dp`
is supplied, its live parameter vector. Names are case-sensitive and match
`get_param_names`. Values use the model's internal, system-base units.
"""
function get_parameter(ps::PowerSystem, device_index::Integer, name)
    index = parameter_index(ps, device_index, name)
    dev = ps.dynamic.devices[device_index]
    values = zeros(dev.dtype.par_size)
    fill_pvec!(values, dev.dtype)
    return values[index - dev.par_ptr + 1]
end

function get_parameter(dp::DynamicProblem, ps::PowerSystem, device_index::Integer, name)
    return dp.pvec[parameter_index(ps, device_index, name)]
end

function get_parameter(ps::PowerSystem, model, bus::Integer, id::AbstractString, name;
                       numbering=:external)
    return get_parameter(ps, find_device(ps, model, bus, id; numbering), name)
end

function get_parameter(dp::DynamicProblem, ps::PowerSystem, model, bus::Integer,
                       id::AbstractString, name; numbering=:external)
    return get_parameter(dp, ps, find_device(ps, model, bus, id; numbering), name)
end

_parameter_field(::AbstractDeviceType, name::Symbol) = name
_parameter_field(::ESDC1A, name::Symbol) =
    name == :sat_a ? :Ae : name == :sat_b ? :Be : name

_parameter_readonly(::AbstractDeviceType, name::Symbol) = true
_parameter_readonly(::Genrou, name::Symbol) = false
_parameter_readonly(::IEESGO, name::Symbol) = name == :pref
_parameter_readonly(::TGOV1, name::Symbol) = name == :pref
_parameter_readonly(::SEXS, name::Symbol) = name == :vref
_parameter_readonly(::ESDC1A, name::Symbol) = name == :vref
_parameter_readonly(::IEEEST, name::Symbol) = name in (:MODE, :BUSR)
_parameter_readonly(::ZIPLoad, name::Symbol) =
    name in (:pinj, :qinj, :v0mag, :yreal, :yimag)

function _parameter_snapshot(ps::PowerSystem, dev::DynamicDevice, name::Symbol)
    field = _parameter_field(dev.dtype, name)
    for table in ps.dynamic.layout.tables
        row = findfirst(==(dev.par_ptr), table.par_ptr)
        row === nothing && continue
        # Parameters without a table mirror are carried in pvec only.
        return (hasproperty(table, field) ? getproperty(table, field) : nothing, row)
    end
    throw(ArgumentError("no parameter table for $(get_device_name(dev.dtype))"))
end

function _validate_parameter(dtype, field, value)
    isfinite(value) || throw(ArgumentError("parameter value must be finite"))
    # Preserve the TGOV1 parser's invariants before touching any store.
    # Other models' stored, unenforced limits may legitimately be equal.
    if dtype isa TGOV1
        field in (:R, :T1, :T3) && value == 0 &&
            throw(ArgumentError("TGOV1 $field must be non-zero"))
        lower = field == :VMIN ? value : dtype.VMIN
        upper = field == :VMAX ? value : dtype.VMAX
        lower < upper || throw(ArgumentError("VMIN must be less than VMAX"))
    end
    return nothing
end

"""
    set_parameter!(dp, sys, device_index, name, value)
    set_parameter!(dp, sys, model, bus, id, name, value; numbering=:external)

Edit a native device parameter **between runs**. Synchronizes the device
struct, `dp.pvec`, and its CPU table snapshot. The edit survives subsequent
`initialize_dynamics!` calls. Names and units match `get_parameter`.

Initialization-derived parameters (controller references, static-generator
setpoints and load admittances), load powers, and wiring selectors are
read-only. Use `set_load!` / `set_generator!` for operating-point changes.
No state is reset: call `initialize_dynamics!` to obtain a new equilibrium
when needed. Do not call during integration or on a GPU/batched problem.
Other problems or GPU/batch layouts sharing `sys` must be recreated.
"""
function set_parameter!(dp::DynamicProblem, ps::PowerSystem, device_index::Integer,
                        name, value::Real)
    index = parameter_index(ps, device_index, name)
    dp.pvec isa Vector{Float64} ||
        throw(ArgumentError("parameter editing requires a CPU DynamicProblem"))
    length(dp.pvec) == ps.dynamic.par_dim ||
        throw(DimensionMismatch("parameter vector does not match the system"))
    dev = ps.dynamic.devices[device_index]
    key = Symbol(name)
    _parameter_readonly(dev.dtype, key) &&
        throw(ArgumentError("$key is derived, operating-point dependent, or structural; cannot edit it with set_parameter!"))
    field = _parameter_field(dev.dtype, key)
    converted = convert(fieldtype(typeof(dev.dtype), field), value)
    _validate_parameter(dev.dtype, field, converted)
    column, row = _parameter_snapshot(ps, dev, key)
    setfield!(dev.dtype, field, converted)
    dp.pvec[index] = converted
    column === nothing || (column[row] = converted)
    return converted
end

function set_parameter!(dp::DynamicProblem, ps::PowerSystem, model, bus::Integer,
                        id::AbstractString, name, value::Real; numbering=:external)
    return set_parameter!(dp, ps, find_device(ps, model, bus, id; numbering), name, value)
end

function _finite_schedule(value, name)
    converted = Float64(value)
    isfinite(converted) || throw(ArgumentError("$name must be finite"))
    return converted
end

"""
    set_load!(sys, bus, id; pd=nothing, qd=nothing, numbering=:external)

Set static load power in pu on the system base **before power flow**. Updates
any mapped ZIPLoad configuration and table as well. Does not change topology.
After editing, run `runpf!`, construct a new `DynamicProblem`, and initialize
it. Existing problems and GPU/batch layouts are stale and must not be reused.
Returns the edited static `Load`.
"""
function set_load!(ps::PowerSystem, bus::Integer, id::AbstractString;
                   pd=nothing, qd=nothing, numbering=:external)
    index = find_load(ps, bus, id; numbering)
    load = ps.loads[index]
    p = pd === nothing ? load.pd : _finite_schedule(pd, :pd)
    q = qd === nothing ? load.qd : _finite_schedule(qd, :qd)
    updates = Tuple[]
    if ps.dynamic !== nothing
        for (i, dev) in enumerate(ps.dynamic.devices)
            ps.dynamic.map.load[i] == index || continue
            dev.dtype isa ZIPLoad || continue
            v = dev.dtype.v0mag
            isfinite(v) && v > 0 ||
                throw(ArgumentError("load reference voltage must be finite and positive"))
            values = (pinj=p, qinj=q, yreal=p/v^2, yimag=q/v^2)
            for (name, value) in pairs(values)
                column, row = _parameter_snapshot(ps, dev, name)
                push!(updates, (dev.dtype, name, value, column, row))
            end
        end
    end
    load.pd = p
    load.qd = q
    for (dtype, name, value, column, row) in updates
        setfield!(dtype, name, value)
        column[row] = value
    end
    return load
end

"""
    set_generator!(sys, bus, id; psch=nothing, qsch=nothing,
                   qmin=nothing, qmax=nothing, numbering=:external)

Edit a static generator schedule / reactive limits in system-base pu, before
power flow. Infinite reactive limits are allowed. Bus type, status, machine
base and network topology are not editable here. After editing, run `runpf!`,
construct a new `DynamicProblem`, and initialize it; discard existing problems
and GPU/batch layouts. Returns the edited static `Gen`.
"""
function set_generator!(ps::PowerSystem, bus::Integer, id::AbstractString;
                        psch=nothing, qsch=nothing, qmin=nothing, qmax=nothing,
                        numbering=:external)
    gen = ps.gens[find_generator(ps, bus, id; numbering)]
    p = psch === nothing ? gen.psch : _finite_schedule(psch, :psch)
    q = qsch === nothing ? gen.qsch : _finite_schedule(qsch, :qsch)
    lower = qmin === nothing ? gen.qmin : Float64(qmin)
    upper = qmax === nothing ? gen.qmax : Float64(qmax)
    (!isnan(lower) && !isnan(upper) && lower <= upper &&
     lower != Inf && upper != -Inf) ||
        throw(ArgumentError("reactive limits must satisfy -Inf <= qmin <= qmax <= Inf"))
    gen.psch = p
    gen.qsch = q
    gen.qmin = lower
    gen.qmax = upper
    return gen
end
