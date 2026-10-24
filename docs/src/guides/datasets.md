# Generating training data

GradPower ships a pipeline that generates **transient-stability datasets**: many
fault simulations over many operating points, with a stability label for each,
written to HDF5. It lives in `scripts/`:

| file | role |
|:--|:--|
| `scripts/generate_dynamics.jl` | the driver: reads a TOML sweep, shards across MPI/SLURM ranks, writes one HDF5 per rank |
| `scripts/dynstab_io.jl` | the building blocks: case loading, load scaling, simulation, labels, HDF5 layout |
| `scripts/sweeps/*.toml` | sweep definitions |
| `scripts/tutorial_generate_data.jl` | a commented single-process walkthrough |

```sh
julia --project=scripts -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'   # once
julia --project=scripts scripts/generate_dynamics.jl scripts/sweeps/example.toml --dry-run
julia --project=scripts scripts/generate_dynamics.jl scripts/sweeps/example.toml
```

## One sample

A sample is one simulation: an operating point (multiplicative load scale), a fault (bus,
resistance, duration) and the response. By default `simulate_sample` records
**all state rows** in `Float64`: device differential and algebraic states,
plus real and imaginary voltage at every bus. `downsample` affects only the
saved time columns. The output includes actual saved times and a row schema
with external bus, model, device ID, state name, kind, and initial online flag.

Use `output_mode=:derived_channels` explicitly for only bus voltage magnitude
and angle and machine rotor angle and speed in `Float32`. The default is
`output_mode=:all_states`, including when no option is supplied.

The labels are accumulated from **every** step by a step callback.

```@example data
using GradPower, CairoMakie
include(joinpath(pkgdir(GradPower), "scripts", "dynstab_io.jl"))

case = Dict("raw" => "examples/IEEE39.raw", "dyr" => "examples/IEEE39_gov.dyr")
sys, base = build_system(case)
dprob, residual = initialized_problem(sys)
z0 = copy(dprob.zvec)

function sample(bus, r, duration; kw...)
    empty!(sys.dynamic.events)
    GradPower.add_event!(sys, ContingencyEvent(sys.busmap[bus], r, 0.2, 0.2 + duration))
    dprob.zvec .= z0
    simulate_sample(dprob, sys, 5.0; dt = 1/120, downsample = 2, kw...)
end

ch, m = sample(16, 0.005, 0.1)
(keys(ch)..., ), size(ch["states"]), m["stable"], round(m["max_angle_sep_deg"], digits = 1)
```

## Stable or not?

With the default thresholds, the label is `stable = true` when all hold:
- the spread of rotor-angle changes **relative to the initial state** stays
  strictly below 180° over the simulated window;
- the maximum spread in the final 20% of that window is at most half the
  maximum spread over the whole run;
- the recorded metrics remain finite.

This is a finite-horizon screening label, not a proof of asymptotic or
small-signal stability. Sweeping fault duration illustrates its transition:

```@example data
durations = [0.10, 0.20, 0.30, 0.35, 0.40, 0.50]
fig = Figure(size = (900, 360))
ax = Axis(fig[1, 1], xlabel = "time (s)", ylabel = "max rotor angle separation (deg)",
          yscale = Makie.pseudolog10, yticks = [0, 10, 100, 1000, 10000],
          title = "fault at bus 16, r = 0.005 pu")
for d in durations
    ch, m = sample(16, 0.005, d; output_mode=:derived_channels)
    delta = ch["gen_delta"]
    sep = vec(maximum(delta .- delta[:, 1], dims = 1) .- minimum(delta .- delta[:, 1], dims = 1))
    t = ch["time"]
    lines!(ax, t, rad2deg.(sep), label = "$(Int(1000d)) ms → " * (m["stable"] ? "stable" : "UNSTABLE"),
           linestyle = m["stable"] ? :solid : :dash)
end
axislegend(ax, position = :lt)
fig
```

In this example, the shorter faults produce bounded swings over the observation
window, while the longer faults produce large angle separation. The sweep
brackets a transition under the chosen model, time step, horizon, and labeling
rule. Estimating a critical clearing time requires refining that bracket and
checking sensitivity to these numerical and modeling choices.

## A small sweep

The same loop over buses and durations gives a picture of the case's stability
boundary. With `early_stop_sep`, unstable samples are cut as soon as they pass
360°, which saves most of their cost without changing any label:

```@example data
buses = [2, 4, 8, 16, 21, 26, 29]
grid = falses(length(buses), length(durations))
tstop = zeros(length(buses), length(durations))
for (i, b) in enumerate(buses), (j, d) in enumerate(durations)
    _, m = sample(b, 0.005, d; early_stop_sep = deg2rad(360))
    grid[i, j] = m["stable"]; tstop[i, j] = m["t_stop"]
end
fig = Figure(size = (800, 380))
ax = Axis(fig[1, 1], xlabel = "fault duration (ms)", ylabel = "faulted bus",
          xticks = (1:length(durations), string.(Int.(1000 .* durations))),
          yticks = (1:length(buses), string.(buses)),
          title = "r = 0.005 pu: stable (green) / unstable (red); number = simulated seconds")
heatmap!(ax, 1:length(durations), 1:length(buses), Float64.(permutedims(grid)),
         colormap = [:tomato, :mediumseagreen], colorrange = (0, 1))
for i in eachindex(buses), j in eachindex(durations)
    text!(ax, j, i, text = string(round(tstop[i, j], digits = 1)), align = (:center, :center))
end
fig
```

Every unstable cell stopped 1–2 s into the run instead of simulating the full
5 s. The boundary moves with the location: a fault at bus 2 can last 100 ms
longer than one at bus 16.

## The sweep file

A sweep is the cross product of loading draws × fault buses × resistances ×
durations:

```toml
output_root = "./dynstab_out"

[sim]
t_final    = 10.0
dt         = 0.008333333333333333   # 1/120 s
downsample = 2                      # store every 2nd step
output_mode = "all_states"          # default, even when omitted
# output_mode = "derived_channels" # explicit compact Float32 output
early_stop_deg = 360.0              # optional: stop unstable runs early

[load]
distribution = "uniform"            # lambda ~ U(low, high)
low = 0.95
high = 1.05
n_samples = 10                      # operating points
seed = 20240917

[fault]
buses     = "all"                   # or a list of PSS/E bus numbers
r_fault   = [0.01, 0.05, 0.2]
t_on      = 0.2
durations = [0.10, 0.20, 0.25, 0.30, 0.35]

[label]
angle_sep_threshold_deg = 180.0
tail_fraction           = 0.2
decay_ratio             = 0.5

[[cases]]
name = "ieee39"
raw  = "examples/IEEE39.raw"
dyr  = "examples/IEEE39_gov.dyr"
```

For each λ draw, the power flow and initialization run once and are shared by
all the faults below it. Before using an operating point, the driver also
screens its response to a small speed perturbation (`check_self_stability`).
Passing that finite-duration check is not a mathematical stability proof.
The dataset load scale is multiplicative (1 is the base case), unlike the
additive CPF parameter whose base case is λ = 0.

## HDF5 layout

```
<case>_rank0000.h5
├── attributes: schema_version=4, output_mode, case_name, baseMVA, dt, dt_eff, t_final, downsample, ...
├── grid/            topology, once per file: bus_id, bus_type, branch_index, gen_bus, ...
├── state_schema/    index, model, bus, device_id, name, kind, online (all_states only)
└── samples/
    ├── 000001/      time [T], states [length(z) × T] (Float64, all_states default)
    │                attributes: load_scale, fault_bus_ext, r_fault, t_on, t_off, clearing_time,
    │                            stable, max_angle_sep_deg, max_freq_dev, tail_peak_ratio, ...
    │                            (+ terminated_early, t_stop, stop_reason with early stop)
    └── ...
```

`/grid` holds what a graph neural network needs: the nodes are buses, the edges
are `branch_index`, and machines are attached through `gen_bus`.

In `derived_channels` mode, each sample contains `bus_vm`, `bus_va`,
`gen_delta`, and `gen_omega` instead of `states`, and no `/state_schema` is
written. Both modes store `time[T]`; use it rather than reconstructing times
from the requested horizon, particularly with early stopping.

Schema entries describe rows in recording order. Use
`(model, bus, device_id, name)` to identify a raw state across layouts, not the
solver row `index`. See `scripts/README_datagen.md` for a readback example.

## Pitfalls

These are documented in detail in `scripts/README_datagen.md`. The most
important:

- **The horizon must resolve the first swing.** Large systems peak late; use
  10 s for ACTIVSg2000.
- **Clearing times must straddle the critical clearing time**, or every sample
  lands in one class.
- **Fault severity need not be monotonic in `r`.** In a fixed Thevenin
  equivalent ``Z_{th}=R_{th}+\mathrm{i}X_{th}``, a purely resistive shunt
  absorbs maximum active power at ``r=|Z_{th}|``, not as ``r\to0``.
  Maximum absorbed power is not necessarily the worst transient-stability
  disturbance; the network and machine dynamics also matter.
