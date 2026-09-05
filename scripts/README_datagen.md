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
simulations per Polaris node-hour. Output ≈1.2 MB per ACTIVSg2000 sample at
`downsample = 2`.

## Output

`/grid` holds static topology once per file (`bus_id`, `bus_type`,
`branch_index`, `gen_bus`, …). Each `/samples/NNNNNN` holds
`bus_vm`, `bus_va` `[n_bus, T]` and `gen_delta`, `gen_omega` `[n_gen, T]`, all
Float32, plus attributes: `load_scale`, `fault_bus_ext`, `r_fault`,
`clearing_time`, `stable`, `max_angle_sep_deg`, `max_freq_dev`, …

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

## ACTIVSg2000

Runs, and matches uqgrid to ~1% — but is **dynamically unstable**, so its
labels carry almost no information about the fault. Three codes agree:
GradPower and uqgrid both run away (identically, with limits on or off), and
ANDES eigenvalue analysis finds 293 modes with `Re > 0`, max `Re = +5.94`.
Limiters are not the missing piece.

It also needs `surrogates = true`, which comes in two tiers: 440 records copy
uqgrid's own redirects (comparable), 418 are GradPower-local stand-ins with no
oracle (not valid for validation). Read the header of
`scripts/sweeps/activs2000.toml` before generating from it.
