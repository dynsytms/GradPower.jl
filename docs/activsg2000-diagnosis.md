# ACTIVSg2000: why it does not produce usable transient-stability data

Status: **not usable as shipped** for a fault-response study. The instability is
in the case, not in GradPower. ANDES cannot run the case at all.

## Summary

| question | answer |
|---|---|
| Does GradPower run it? | Yes. PF solves, init residual 1.3e-09, no-fault drift 1.8e-10. |
| Is the equilibrium stable? | No. A 1e-8 speed kick with no fault grows to ~140° in 5 s. |
| Is that a GradPower bug? | No. uqgrid reproduces it on the same 334-machine set, to ~1%. |
| Is it missing limiters? | No. uqgrid with limits on vs off: 347.898° vs 346.983°. |
| Does ANDES corroborate? | It cannot run the case. No verdict either way. |

## Evidence that the case is unstable

GradPower and uqgrid, same active machine set, bus 1001, ton 0.2, toff 0.3,
tend 2, ZIP alpha 0.5:

| r_fault | GradPower | uqgrid |
|---|---|---|
| 0.02 | sep 343.6° / \|w\| 5.105e-2 | sep 347.0° / \|w\| 5.100e-2 |
| 1000 | sep 196.8° / \|w\| 5.125e-2 | sep 210.0° / \|w\| 5.121e-2 |

Agreement is ~1% on separation and ~0.1% on frequency. Both run away, and a
fault five orders of magnitude weaker gives the same peak |w| — the mode is
fault-independent, which is exactly why labels from this case carry no
information about the fault.

The growth rate is λ ≈ 3/s. That independently explains the "flat" no-fault
run: round-off at 1e-16 grows to ~3e-10 in 5 s, and the measured drift is
1.8e-10.

The runaway is a property of the equations, not the integrator: refining dt
16× (1/120 → 1/1920) moves the result from 140.33° to 139.32°. Response to
perturbation size is non-monotonic (1e-10 → 346°, 1e-8 → 134°), i.e. chaotic.

## What was ruled out

None of these change the outcome (all stay in 139–147°):

- controllers — surrogates on/off, no PSS, no exciters, GENROU only
- inertia — flooring H at 1.0 or 3.0 (30 machines have H < 1; min H = 0.0258)
- damping — sweeping GENROU D from 0 to 20
- load model — ZIP alpha 0.0 / 0.5 / 1.0
- limiters — tested in uqgrid, which implements them
- generator MBASE defect — see below

GradPower's own initialization is physically sane: e_qp has median 0.96 and
max 4.53 across 334 machines.

## Genuine data defect found: generators dispatched above MBASE

16 in-service generators in `ACTIVSg2000.raw` have `PG > MBASE`, most at
~2.47–2.50×. Straight from the raw:

| bus | PG (MW) | MBASE (MVA) | PG/MBASE |
|---|---|---|---|
| 6266 | 4.500 | 1.800 | 2.500 |
| 6268 | 4.500 | 1.800 | 2.500 |
| 3105 | 44.500 | 18.000 | 2.472 |

They carry 2356 MW of 68728 MW dispatched (3.4%). This is a real inconsistency
worth reporting upstream, and it is the direct cause of the extreme field
voltages ANDES reports. **It is not the cause of the instability**: rebuilding
the raw with `MBASE = PG/0.85` for all 16 leaves the self-kick at 140.2° versus
140.3°.

## Why ANDES cannot be used here

ANDES 1.10.0 implements every model natively (278 ESST4B, 43 IEEEG1, 25 HYGOV,
38 EXAC2, …), so it should be the ideal third opinion. It is not:

1. Initialization never converges. It prints `Initialization failed` while
   `TDS.initialized` still returns `True` — do not trust that flag.
2. Four EXAC2 devices (45, 66, 67, 75) fail iterative initialization, reaching
   `IN = -3.1e+26` and `VHA = -1.3e+17`.
3. Upstream of that, 14 GENROU are reported needing vf of 5.7–52 pu against a
   typical limit of 5 — the exciters cannot deliver it, so they diverge.
4. `TDS.run()` terminates at **t = -0.033 s**: the simulation never starts.
5. Disabling the 4 bad EXAC2 gets it to t = 1.28 s, then it blows up with
   |w-1| = 7052. Disabling all EXAC2 and all ESST4B still fails at t = 1.20 s.

Because initialization never converged, ANDES starts off-equilibrium, so its
divergence cannot be used as evidence of instability — the argument would be
circular. An earlier eigenvalue result (293 modes with Re > 0) was withdrawn:
it was computed at that broken point, and disabling 38 EXAC2 devices left the
spectrum bit-identical, so the experiment was not controlled.

Getting a verdict from ANDES means repairing the case data first — the exciter
limits, or the operating point that demands vf = 52.

Reproduce: `/tmp` scripts are transient; the checks above are
`check_self_stability` in `scripts/dynstab_io.jl`, plus uqgrid via
`uqgrid.IntegrationConfig(..., enforce_dynamic_limits=True/False)`.

## If you need this case to yield usable labels

The instability survives every controller and parameter intervention tried, so
the remaining candidates are the operating point itself (`ACTIVSg2000.raw`) and
the machine data, not the controller coverage. Implementing ESST4B natively
(plan_enhance.md Task 4, 278 records) is worth doing on its own merits but,
given that removing exciters entirely changes nothing, should not be expected
to fix this.
