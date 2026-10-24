# Queries and parameter edits

Use external PSS/E bus numbers by default. Select `numbering=:internal`
explicitly when passing an internal bus index; there is no automatic fallback.
Device IDs accept surrounding quotes and blanks, so `"1"` and `" '1' "` select
the same ordinary device. Model names are case-insensitive Symbols or strings
(or device types); state and parameter names are case-sensitive.

## Find devices and vector indices

This example uses the bundled IEEE9 case:

```@example queries
using GradPower
ex = joinpath(pkgdir(GradPower), "examples")
ps = from_psse(joinpath(ex, "ieee9_v33.raw"),
               joinpath(ex, "ieee9bus_gov.dyr"))
build_network!(ps)
runpf!(ps)
dp = DynamicProblem(ps)
initialize_dynamics!(dp, ps)

b = bus_index(ps, 1)                         # internal bus index
bus = get_bus(ps, b; numbering=:internal)     # Bus object
gen = find_generator(ps, 1, "1")             # index into ps.gens
load = find_load(ps, 5, "1")                 # index into ps.loads
machines = find_devices(ps; model=:genrou, online=true)
device = find_device(ps, :genrou, 1, "1")     # dynamic device index
iw = state_index(ps, :genrou, 1, "1", :w)
ivr = state_index(ps, :bus, 1, "", :vr)
ip = parameter_index(ps, device, :H)
@assert ip == parameter_index(ps, :genrou, 1, "1", :H)
@assert get_index_info(iw, ps) isa StateChannel
(speed=dp.zvec[iw], voltage_real=dp.zvec[ivr])
```

`find_generator`, `find_load`, and `find_device` require exactly one match;
missing or ambiguous identities raise `ArgumentError`. `find_devices` returns
all matching dynamic indices (possibly empty), with optional `model`, `bus`,
`id`, and `online` filters. Static row indices and dynamic device indices are
different index spaces.

`state_index` returns the actual global `z` index, not a device-local offset
or a row in a selectively recorded trajectory. Network states use model
`:bus`, ID `""`, and name `:vr` or `:vi`. `parameter_index` returns the global
`pvec` index. Resolve indices again after rebuilding or changing the layout.

## Inspect the system

[`system_summary`](@ref) returns a vector of NamedTuple rows rather than printed
text. Choose `section=:buses` (default), `:generators`, `:loads`, or `:devices`.
Rows include an index, external bus number, identity, and section-specific data;
device rows also include model, online status, and vector pointers.

```@example queries
system_summary(ps; section=:devices)
```

For text output, use `get_bus_info(1, ps)` or `display_parameters(ps)`.
Both accept an IO destination first: `get_bus_info(io, 1, ps)` and
`display_parameters(io, ps)`. `get_bus_info` takes an external bus number.
`get_index_info(iw, ps)` returns a `StateChannel` without printing;
`get_index_info(io, iw, ps)` also prints it.

## Edit native parameters between runs

[`get_parameter`](@ref) without `dp` reads the device configuration; with `dp`
it reads that problem's live parameter vector. Both accept a dynamic device
index or the `(model, bus, id, name)` selector:

```@example queries
h = get_parameter(ps, device, :H)
@assert h == get_parameter(dp, ps, :genrou, 1, "1", :H)
set_parameter!(dp, ps, :genrou, 1, "1", :H, 1.05h)
@assert get_parameter(dp, ps, device, :H) == 1.05h
initialize_dynamics!(dp, ps)
@assert get_parameter(ps, device, :H) == 1.05h
t, trajectory = integrate!(dp, ps, 0.1)
nothing # hide
```

[`set_parameter!`](@ref) accepts finite native parameter values and synchronizes
the device configuration (`dtype`), the selected problem's `pvec` entry, and the
matching CPU struct-of-arrays table field. Edits persist through
`initialize_dynamics!`. Attached parameter values already use the model's
internal, system-base conventions, **not RAW/DYR machine-base values**; do not
apply a second base conversion.

The setter does **not** reset state. Reinitialize when a new equilibrium is
desired. Initialization-derived `pref`/`vref`, all static-generator parameters,
ZIPLoad `pinj`, `qinj`, `v0mag`, `yreal`, `yimag`, and IEEEST `MODE`/`BUSR` are
read-only here. ESDC1A saturation parameters use the supported names
`sat_a` and `sat_b`. Inspect names with `get_param_names` or `display_parameters`.

!!! warning "CPU, between runs only"
    Never call these setters during integration, including from callbacks.
    `set_parameter!` is for a CPU `DynamicProblem`, not GPU or batched problems.
    Other `DynamicProblem` instances and GPU/batch layouts sharing `ps` are
    not synchronized: recreate them after an edit.

## Change the operating point before power flow

Use [`set_load!`](@ref) for static `pd`/`qd`; it also synchronizes mapped ZIPLoad
configurations and tables. Use [`set_generator!`](@ref) for static
`psch`/`qsch` schedules and `qmin`/`qmax` reactive limits. Omitted keywords
leave values unchanged. Powers and limits are in pu on the system base;
reactive limits may be infinite. `qd` retains the load struct's sign convention:
negative `qd` means positive reactive demand. Generator `qsch` is positive for
reactive injection.

Continuing the example, discard the old problem and solve the new operating
point before allocating dynamics:

```@example queries
pd = ps.loads[load].pd
psch = ps.gens[gen].psch
set_load!(ps, 5, "1"; pd=1.01pd)
set_generator!(ps, 1, "1"; psch=psch, qmin=-10.0, qmax=10.0)
runpf!(ps)
dp = DynamicProblem(ps)                     # new problem, not the old dp
initialize_dynamics!(dp, ps)
nothing # hide
```

These are **pre-power-flow edits only**, not events or mid-run controls.
After either setter, run `runpf!`, construct a **new** `DynamicProblem`, and
initialize it; recreate any other problems or GPU/batch layouts sharing `ps`.
Neither setter edits topology, bus type, status, or machine base (`mbase`).
Use the [event API](../tutorials/events.md) for supported scheduled disturbances.
