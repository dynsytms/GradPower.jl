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

**Ready to use, provided the power flow enforces generator reactive limits** —
which it does by default (`enforce_q_limits` in the case table).

The case itself was never the problem; GradPower's power flow was. Without Q
limits, 200 of 432 generators solve outside their nameplate QT/QB, 20 machines
end up past their pull-out angle, and a 1e-6 speed kick with no fault grows to
147.7° in 5 s — every label would describe that runaway rather than the fault.
With limits, 199 buses switch PV→PQ, no generator is outside its limits, and
the same kick decays to 0.0018°.

Validated against uqgrid on the same case, models and fault: identical PV→PQ
active set (199 buses, all 2000 bus types agree), bus voltages to 6.7e-15,
reactive dispatch to 2.3e-13, and machine speeds to 1.2e-14 elementwise over
the whole trajectory.

Keep `check_self_stability = true` — it is the guard that caught this. Full
analysis in `docs/activsg2000-diagnosis.md`.

**Input strength, measured.** Fault location is the strongest of the three
inputs, clearing time next, loading the weakest — λ=0.95 and λ=1.05 gave the
same 79.2% stable fraction and the same unstable buses. The feasible λ range is
~0.70–1.08; above 1.08 the dynamic initialization fails. The sweep uses
0.85–1.08 for operating-point diversity, not because it is expected to move the
labels much.
