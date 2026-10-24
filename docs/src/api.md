# API reference

## Building a system

```@docs
from_psse
build_network!
runpf!
runpf
analyze_dyr_coverage
```

## Continuation power flow

`runcpf` traces steady-state AC operating points on CPU without mutating the
input system. Directions use internal bus indices, while the `buses` recording
selection and result metadata use external bus numbers. Inspect `status` and
`maximum`: a sampled maximum or an exhausted step budget does not verify a
loading margin. See the [continuation manual](guides/continuation.md) for
worked examples, and the docstrings below for injection/dispatch signs and
units. `plot_cpf` and `animate_cpf` require a loaded Makie backend, such as
`CairoMakie`; the continuation solver itself does not.

```@docs
LoadingDirection
runcpf
ContinuationResult
ContinuationPoint
ContinuationEvent
ContinuationStats
plot_cpf
animate_cpf
```

## Queries and inspection

See [Queries and parameter edits](guides/queries.md) for identity conventions
and an IEEE9 example. Bus selectors default to external numbering; internal
indices require `numbering=:internal`.

```@docs
bus_index
get_bus
find_generator
find_load
find_devices
find_device
state_index
parameter_index
system_summary
display_parameters
get_bus_info
get_index_info
```

## Parameter and operating-point edits

Parameter edits are CPU-only and between runs, never during integration.
Operating-point edits require power flow followed by a new dynamic problem
and initialization; existing problems and GPU/batch layouts must be recreated.

```@docs
get_parameter
set_parameter!
set_load!
set_generator!
```

## Dynamic problem

```@docs
DynamicProblem
initialize_dynamics!
integrate!
GradPower.integrate_batched!
SimulationResult
StateChannel
state_channels
state
speed
bus_voltage
GradPower.BatchedLayout
GradPower.gen_speeds
```

## Optional plotting

Load a Makie backend such as `CairoMakie` before or after `GradPower` to enable
these methods; Makie is not a core dependency. See
[Plotting simulation results](guides/plotting.md) for runnable illustrated
examples, units, scenario envelopes, and network-layout limitations.

```@docs
plot_states
plot_voltage
plot_speed
plot_network
animate_network
```

## Events

```@docs
ContingencyEvent
add_event!
TripLineEvent
create_trip_line_event
find_branch
add_trip_event!
DisconnectDeviceEvent
add_disconnect_event!
add_generator_trip!
generator_devices
add_load_trip!
LoadStepEvent
create_load_step_event
add_load_step_event!
find_load_device
```

## Solvers and diagnostics

```@docs
KLUBackend
CUDSSBackend
SparseDirectSolverBackend
GradPower.JacobianReuse
SolverLog
GradPower.NewtonConvergenceError
```

## Optional NLP Model

```@docs
GradPower.DynamicNLP
```

Install and load `NLPModels` and `MadNLP` before constructing the model. The
constructor returns an `NLPModels.AbstractNLPModel` implemented in
`GradPowerNLPExt`; `GradPower.DynamicNLP` itself is a function, not a type.
Only forward objective evaluation is supported. Gradients still raise an
explicit error, so gradient-based optimization is not currently available.
