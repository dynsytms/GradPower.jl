# Transient-stability data generation

Three inputs define one sample: **loading** (λ, sampled from a distribution),
**fault location** (bus), and **fault admittance** (`r_fault`, applied as a
shunt `y = 1/r_fault`).

## Start here

```sh
julia --project=scripts -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'   # once
julia --project=scripts scripts/tutorial_generate_data.jl
```

`scripts/tutorial_generate_data.jl` is a commented single-process walkthrough of
the whole loop. Read it before touching the production driver.

## Generating a dataset

```sh
julia --project=scripts scripts/generate_dynamics.jl scripts/sweeps/activs2000.toml --dry-run
julia --project=scripts scripts/generate_dynamics.jl scripts/sweeps/activs2000.toml
```

Always `--dry-run` first: it prints the scenario count and sampled λ range
without simulating. On Polaris:

```sh
SWEEP=scripts/sweeps/activs2000.toml qsub -V -q debug -l select=1 \
    -l walltime=01:00:00 scripts/hpc/gen-polaris.sh
```

Ranks shard scenarios round-robin and each writes
`<output_root>/<case>/<case>_rank<RRRR>.h5`.

## Config

```toml
[sim]
output_mode = "all_states"  # default, even when omitted
downsample = 1             # save every integration step; affects columns only
```

`all_states` exports **every row of the solver's `z` vector**: all device
differential states, all device algebraic states, and both rectangular voltage
components at every bus. Values remain Float64. To export only the compact
Float32 voltage magnitude/angle and machine angle/speed channels, explicitly
set `[sim] output_mode = "derived_channels"`. No state rows are silently
discarded by the default mode.

```toml
[load]                      # lambda ~ Uniform(low, high); base case is the mean
distribution = "uniform"    #   when low + high == 2
low = 0.95
high = 1.05
n_samples = 10              # number of operating points drawn
seed = 20240917             # reproducible from the config alone
```

One λ draw is shared by the whole fault cross product beneath it, so the power
flow and dynamics initialization cost is paid `n_samples` times per case, not
once per simulation. Scenario count = `n_samples × n_buses × n_r_fault ×
n_durations`.

Cost reference: ~3.3 s per 5 s ACTIVSg2000 simulation on one core, ≈35k
simulations per Polaris node-hour. The compact `derived_channels` output is
≈1.2 MB per ACTIVSg2000 sample at `downsample = 2`; the default `all_states`
output is larger.

**Memory.** `all_states` retains all `z` rows at every `downsample`-th step.
Only the explicit `derived_channels` mode restricts stored rows to bus
`(vr, vi)` and machine `omega`/`delta` before deriving the compact output.
Both use `integrate!(...; save_idxs, save_every)`. Labels are accumulated over
every step through `step_callback` (`DynstabRecorder`), so they are computed
at full resolution in either mode. See `simulate_sample` in `dynstab_io.jl`;
`julia --project=scripts scripts/test_dynstab_recorder.jl` checks exact
full-trajectory equivalence and HDF5 roundtrips.

**Early stop (optional).** `[sim] early_stop_deg = 360.0` ends a simulation
once the rotor-angle separation, relative to the initial operating point,
exceeds that angle (it must be `>= angle_sep_threshold_deg`, so the `stable`
label cannot change). Stopped samples are shorter (`T` attribute) and carry
`terminated_early = 1`, `t_stop`, `stop_reason = "angle_separation"`; their
`tail_peak_ratio` is NaN and `settled = 0`. Off by default. `time` always
contains the actual saved times; its final entry may precede an off-stride
`t_stop`.

## Output

Schema version **4** adds `output_mode` to file/sample attributes and an
explicit Float64 `time[T]` dataset to every `/samples/NNNNNN`.
`/grid` holds static topology once per file (`bus_id`, `bus_type`,
`branch_index`, `gen_bus`, …).

In the default **`all_states`** mode each sample holds Float64
`states[length(z), T]`, in solver row order. `/state_schema` holds one column
per field of `GradPower.state_channels(sys; save_idxs=:)`, shared by the
samples in that file:

| Column | Stored Type | Meaning |
| --- | --- | --- |
| `index` | Int | 1-based original `z` row |
| `model` | String | Device model, or `"bus"` for network voltages |
| `bus` | Int | External PSS/E bus number |
| `device_id` | String | Stable device identifier; empty for network voltages |
| `name` | String | Raw state name, including `"vr"` and `"vi"` |
| `kind` | String | `"differential"` or `"algebraic"` |
| `online` | Bool | Initial device online status |

The entry at schema position `r` describes row `r` of `states`. Identify a
state across layouts using `(model, bus, device_id, name)`, not `index`.
All fields are ordinary HDF5 arrays, not Julia object serialization.

The opt-in **`derived_channels`** mode instead holds Float32 `bus_vm`,
`bus_va` `[n_bus, T]` and `gen_delta`, `gen_omega` `[n_gen, T]`; it does
not write `states` or `/state_schema`. These are derived channels, not a
complete raw-state representation.

Both modes include sample attributes `load_scale`, `fault_bus_ext`,
`r_fault`, `clearing_time`, `stable`, `max_angle_sep_deg`, `max_freq_dev`, …
Consumers of schema v3 must explicitly opt in to the compact mode to retain
those channel names, and should use the stored `time` rather than inferring
times from a requested horizon.

For example, reconstruct a named result from an `all_states` sample without
a Julia-specific serialization format:

```julia
using GradPower, HDF5
result = h5open("dataset.h5", "r") do fid
    t = read(fid["samples/000001/time"])
    z = read(fid["samples/000001/states"])
    schema = read(fid["state_schema"])
    channels = StateChannel[
        StateChannel(schema["index"][i], Symbol(schema["model"][i]),
                     schema["bus"][i], schema["device_id"][i],
                     Symbol(schema["name"][i]), Symbol(schema["kind"][i]),
                     schema["online"][i])
        for i in eachindex(schema["index"])
    ]
    SimulationResult(t, z, channels)
end
# For a dataset with GENROU machines:
dw = speed(result, 1)
delta = state(result, :genrou, 1, :delta)
bus = first(c.bus for c in result.channels if c.model === :bus)
vm = bus_voltage(result, bus; component=:magnitude)
```

The arrays remain usable after the file closes. This reconstructs state
channels and times, not solver status or event history; the constructor's
status defaults to `:unknown`.

## Three things that will bite you

**`r_fault` severity is not monotonic.** The fault is purely resistive, so the
power it absorbs is `V²/r` and peaks where `r` matches the local Thevenin
impedance — not as `r → 0`. On IEEE9 at bus 7 the worst `r_fault` is near 0.1,
and 0.02 is *milder* than 0.05. Sweep on a log grid and find the peak for your
case before fixing a range.

**Separation is measured relative to the pre-fault state.** A large grid has a
big standing angle spread at rest (ACTIVSg2000 sits at ~197° with no fault), so
an absolute threshold would label everything unstable. `initial_sep_deg`
reports the standing spread; `max_angle_sep_deg` is the deviation from it.

**`stable` is a screen, not a certificate.** Unstable trajectories are chaotic,
so near the boundary the label is dt-sensitive — refining dt does not make an
unstable run converge. Do a dt-refinement check on a few near-boundary cases.

## Preflight

Before each operating point the driver asserts the initialization residual and
runs `check_self_stability`: a 1e-6 speed kick with no fault must not grow. A
case that fails is one where labels would reflect an unstable mode rather than
the fault. Disable it only deliberately.

## Labels: read this before training on them

`stable = (peak separation < 180 deg) AND (the swing decayed to <= 0.5 of its
peak by the end of the window)`. Both conditions separate the two populations
with orders of magnitude of margin on measured data; requiring both guards
against a bounded-but-still-growing swing.

**`t_final` must be long enough for the first swing to resolve.** This is the
easiest way to silently corrupt the dataset. ACTIVSg2000's first swing peaks at
1.5-5 s; at `t_final = 2.5` s the settling test mislabelled 90% of the
"unstable" samples, and reruns at 10 s returned *identical* separations with
the label flipped. IEEE39 settles in ~1 s and showed none of this, which is why
it is not obvious from small cases. Use 10 s for ACTIVSg2000.

**Clearing times must straddle the critical clearing time** or every sample
lands in one class. On IEEE39, `[0.05, 0.1, 0.15]` is 100% stable while the CCT
is 0.16-0.34 s depending on fault location.

**`max_angle_sep_deg` is bimodal** — roughly 2-45 deg for stable and
39000-57000 deg for runaways, because nothing trips a machine that goes over
the top. Fine as a classification label, poor as a regression target.

`tail_peak_ratio` is stored on every sample, so labels can be recomputed
offline under a different rule without re-simulating.

## ACTIVSg2000

The ACTIVSg2000 sweep requires power-flow reactive limits and
`check_self_stability = true`; both are enabled in
`scripts/sweeps/activs2000.toml`.
