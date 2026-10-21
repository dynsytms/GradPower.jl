# API reference

## Building a system

```@docs
from_psse
build_network!
runpf!
runpf
analyze_dyr_coverage
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
