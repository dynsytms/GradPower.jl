# uqgrid regression suite

Compares GradPower trajectories against the Python reference simulator
[uqgrid](https://github.com/dmaldona/uqgrid). Not run by `Pkg.test()` or CI:
it needs a local uqgrid checkout and generated reference files, neither of
which is committed.

## Setup

1. Clone uqgrid into `./uqgrid` (the repo root). The ACTIVSg cases read their
   `.raw`/`.dyr` from `uqgrid/data/` and `uqgrid/benchmarks/`.
2. Generate the references (written to `examples/refs/*.npz`, git-ignored):

   ```sh
   for f in scripts/gen_ref_*.py; do python3 "$f"; done
   ```

   Each `.npz` embeds `metadata_json` with the uqgrid commit, source-file
   hashes and the integration settings.
3. Freeze them for the comparison scripts:

   ```sh
   julia +1.13 --project scripts/verify/phase0_g2.jl   # copies to artifacts/phase0/references/
   ```

## Run

```sh
julia +1.13 scripts/regression/run_all.jl          # whole suite
julia +1.13 test/regression/compare_activs70k.jl   # one case
```

Each script writes `artifacts/regression/<case>.json` and exits 0 iff it passed.

## Conventions

- **Power-flow Q limits are enforced** on both sides (`runpf!` default and
  `ReferenceCase.enforce_q_limits=True`). Without them ACTIVSg70k starts some
  machines at a ~180 degree load angle; one slips a pole after the fault and the
  trajectory becomes sensitive to solver round-off.
- **Event samples are masked.** At `ton`/`toff`, uqgrid stores the post-event
  algebraic projection while GradPower stores the state before the event
  re-solve, so those two samples are excluded (`compare_trajectories` mask).
- References must be regenerated whenever a model changes on either side
  (e.g. two-axis GENROU saturation, Aug 2026). A stale reference shows up as a
  large `initial_state_error`.
