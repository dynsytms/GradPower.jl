# First simulation

Every GradPower study follows the same five steps:

```
 .raw + .dyr ──► from_psse ──► build_network! ──► runpf! ──► DynamicProblem
                                                              initialize_dynamics!
                                                                     │
                       add_event! (faults, trips, ...) ──────────────┤
                                                                     ▼
                                                     integrate!  ──►  (t, traj)
```

We go through them on the WSCC 9-bus system: three GENROU machines, each with an
IEESGO steam governor, and three loads.

## 1. Read the case

```@example first
using GradPower, CairoMakie
ex = joinpath(pkgdir(GradPower), "examples")

sys = from_psse(joinpath(ex, "ieee9_v33.raw"), joinpath(ex, "ieee9bus_gov.dyr"))
```

`from_psse` parses the network (buses, lines, transformers, generators, loads) and
attaches a dynamic model to each device. Generators found in the `.dyr` get their
machine and controller models. Loads become ZIP loads, constant impedance by
default. Generators without a dynamic record become `StaticGenerator` stand-ins
that hold their bus voltage.

## 2. Build the network and solve the power flow

```@example first
build_network!(sys)          # admittance matrix Y
runpf!(sys)                  # Newton power flow (with generator Q limits)
[(b.i, round(b.v0m, digits = 4), round(rad2deg(b.v0a), digits = 2)) for b in sys.buses]
```

The table lists the PSS/E bus number, the voltage magnitude (pu) and the angle
(degrees) of the operating point.

## 3. Initialize the dynamics

```@example first
dp = DynamicProblem(sys)
initialize_dynamics!(dp, sys)
length(dp.zvec)              # number of unknowns of the DAE
```

`initialize_dynamics!` computes every machine's internal state, every
controller's setpoint and every load's admittance so that the system sits
**exactly** at the power-flow equilibrium. A simulation without any event stays
flat:

```@example first
t, traj = integrate!(dp, sys, 2.0)
maximum(abs.(traj .- traj[:, 1]))     # drift over 2 s
```

## 4. Schedule a fault

[`ContingencyEvent`](@ref) is a shunt fault at an (internal) bus index.
`sys.busmap` converts PSS/E bus numbers to internal indices.

```@example first
add_event!(sys, ContingencyEvent(sys.busmap[7], 0.02, 0.1, 0.2))  # r = 0.02 pu, 0.1 s → 0.2 s
```

## 5. Integrate

```@example first
dp.zvec .= traj[:, 1]                 # restart from the equilibrium
t, traj = integrate!(dp, sys, 5.0)    # dt = 1/120 s by default
size(traj)
```

`traj[:, j]` is the full state at time `t[j]`. Plot the machine speeds and every
bus voltage:

```@example first
net = sys.dynamic.diff_dim + sys.dynamic.alg_dim       # voltages come last
vm(b) = hypot.(traj[net + 2b - 1, :], traj[net + 2b, :])

fig = Figure(size = (800, 520))
ax1 = Axis(fig[1, 1], ylabel = "speed deviation (pu)")
for (k, i) in enumerate(GradPower.gen_speeds(sys))
    lines!(ax1, t, traj[i, :], label = "gen $k")
end
axislegend(ax1)
ax2 = Axis(fig[2, 1], xlabel = "time (s)", ylabel = "|V| (pu)")
for b in eachindex(sys.buses)
    lines!(ax2, t, vm(b), label = "bus $(sys.buses[b].i)")
end
fig[2, 2] = Legend(fig, ax2, framevisible = false)
fig
```

During the fault, voltages collapse near bus 7. The machines accelerate, swing
against each other after the fault is cleared, and the governors bring the
frequency back.

!!! tip "Running it again"
    `integrate!` leaves `sys` as it found it: faults are switched off, tripped
    lines restored, parameters reset. To repeat a study, reset `dp.zvec` to the
    initial state (or call `initialize_dynamics!` again), and clear or replace
    the events in `sys.dynamic.events`.

## Variations to try

```julia
integrate!(dp, sys, 5.0; dt = 1/240)                     # smaller step
integrate!(dp, sys, 5.0; integrator = :crank_nicolson)   # 2nd-order method
integrate!(dp, sys, 5.0; solver = :schur)                # Schur-reduced Newton
integrate!(dp, sys, 5.0; save_every = 4)                 # store every 4th step
```

Each option is explained in the guides.
