# ACTIVSg2000: the case is fine; our power flow was wrong

**Root cause: `runpf!` did not enforce generator reactive-power limits.**
ACTIVSg2000 is a usable case. Every "the case is unstable" claim in the first
revision of this document was wrong and is corrected below.

**Status: fixed.** `runpf!` now takes `enforce_q_limits`, and the data-generation
pipeline turns it on by default. See "The fix" at the end for what landed and
how it was validated.

## The one-line result

uqgrid, ACTIVSg2000, negligible fault (r=1000), tend 2 s, ZIP alpha 0.5:

| `enforce_q_limits` | max angle separation | max \|omega\| |
|---|---|---|
| `False` | 209.981 deg | 5.1207e-2 |
| `True`  | **0.001 deg** | **3.6724e-7** |

Same case, same models, same fault. Enforcing PF Q limits is the difference
between a runaway and a flat, stable trajectory.

## Causal chain, each step measured

1. **No Q-limit enforcement in the power flow.** GradPower's `runpf!` has no
   PV→PQ switching. `plan_enhance.md` already lists this as missing
   ("Static generator: PF Q-limit behavior — GradPower: no PF Q limits";
   Phase 6, "Power-flow reactive limits"). uqgrid defaults to
   `enforce_q_limits=False`, so both codes had the same gap.

2. **200 of 432 generators end up outside their limits.** Worst cases:

   | bus | Q solved (MVAr) | QT | QB |
   |---|---|---|---|
   | 3048 | +878.76 | 51.33 | -34.70 |
   | 7406 | +763.67 | 32.12 | -7.00 |
   | 8155 | -639.95 | 81.36 | -21.30 |
   | 1079 | -398.00 | 48.61 | -10.60 |
   | 7400 | -278.81 | 36.65 | -7.99 |

   Bus 7400 absorbs 279 MVAr against an 8 MVAr floor.

3. **Machines absorbing that much Q are driven past pull-out.** The internal
   angle `delta - V_angle` exceeds 90 deg on 20 machines, worst 164 deg. The
   offenders are exactly the buses above (7400 at 163, 1079 at 164, 8155 at 162).

4. **Past 90 deg, dP/ddelta is negative** — negative synchronizing torque, so
   the equilibrium is unstable by construction. GradPower's reduced-system
   spectrum: **18 eigenvalues with Re > 0, max Re = +10.06, all at f = 0 Hz**
   (non-oscillatory, as pull-out implies). Controls IEEE39_gov and ACTIVSg200:
   **0 unstable eigenvalues**. (These came from a dedicated reduction; a
   quick dense Schur written later produced nonsense on all three cases,
   including the stable controls, so treat any eigenvalue figure as valid only
   with its own reduction shown. The self-kick test below needs no reduction
   and says the same thing.)

5. **The link is monotonic.** Capping q-axis reactance moves machines back
   inside 90 deg and removes unstable modes almost one-for-one:

   | xq cap | machines >90 deg | Re>0 | max Re | 5 s self-kick |
   |---|---|---|---|---|
   | as-is | 20 | 18 | 10.06 | 140.3 deg |
   | 2.0 | 16 | 15 | 8.61 | 123.7 deg |
   | 1.0 | 15 | 13 | 5.12 | 93.9 deg |
   | 0.5 | 7 | 7 | 1.69 | 0.003 deg (stable) |

## Why every earlier hypothesis failed

All of these leave the self-kick at 139-147 deg, because none of them touch the
operating point: controllers (surrogates on/off, no PSS, no exciters, GENROU
only), inertia (H floored at 1.0/3.0), damping (D swept 0-20), load model (ZIP
alpha 0/0.5/1.0), dynamic limiters (tested in uqgrid), StaticGenerator stubs
(ACTIVSg200 given 21 stubs stays stable at 0.0015 deg), and the MBASE defect
below. De-loading made it *worse* (74 unstable modes at lambda=0.1), which is
the tell: re-solving an unlimited power flow at light load pushes even more
machines into absorbing reactive power.

## Secondary finding: generators dispatched above MBASE

16 in-service generators in `ACTIVSg2000.raw` have `PG > MBASE`, most at
~2.47-2.50x (bus 6266 and 6268: 4.5 MW on a 1.8 MVA base; bus 3105: 44.5 MW on
18.0 MVA). They carry 3.4% of dispatch. Worth reporting upstream, and it is why
ANDES reports field voltages up to 52 pu — but it is **not** the instability:
rebuilding the raw with `MBASE = PG/0.85` leaves the self-kick at 140.2 vs
140.3 deg.

## ANDES

ANDES 1.10.0 cannot run this case: `TDS.run()` terminates at t = -0.033 s, four
EXAC2 devices fail iterative initialization with internals at 1e26, and
`TDS.initialized` returns `True` while ANDES prints "Initialization failed" —
do not trust that flag. This is consistent with the Q-limit story (machines at
extreme angles demand impossible excitation), but since ANDES never reaches a
converged equilibrium it cannot serve as independent evidence. An eigenvalue
result quoted earlier (293 modes, Re > 0) was withdrawn: computed at that
broken point, and disabling 38 EXAC2 devices left it bit-identical, so the
experiment was not controlled.

## The fix

`runpf!(psys; enforce_q_limits=true)` runs the standard outer loop: solve,
convert any PV bus past its aggregate limit to PQ with its generators pinned at
their own bounds, repeat until nothing switches. Conversion is one-way, as in
MATPOWER's default `enforce_q_lims = 1`; allowing pinned buses back to PV cycles
on this case and exits on the iteration cap with buses still violating.

Two things had to change alongside it, both found by validation rather than by
reading the code:

* **Reactive dispatch across co-located generators.** The even split is wrong
  at the machine level: bus 7422 has one 3.5 MVAr unit beside four 57 MVAr
  units, and an even split hands the small one 0.41 pu against a 0.035 pu
  ceiling — while the bus total sits comfortably inside its aggregate limits.
  The dispatch is now the projection of the even split onto the box, which is
  what uqgrid does and which reduces to the old even split exactly when no
  limit binds.

* **`StaticGenerator` stubs on converted buses.** 67 of the 199 converted buses
  carry a stub, and it kept holding `vr^2 + vi^2 = vset^2` with unlimited
  reactive power through the transient — reintroducing the very behaviour the
  limits remove. Stubs on a converted bus now switch to `q = q0`. Their
  `vset`/`aset` are also re-read from the solved voltage instead of the
  parse-time setpoint (a no-op without enforcement, since a PV bus's magnitude
  is not a power-flow variable; 0.08 pu of initialization residual with it).

### Effect on ACTIVSg2000

| | no Q limits | Q limits |
|---|---|---|
| generators outside QT/QB | 200 / 432 | **0 / 432** |
| buses switched PV→PQ | 0 | 199 |
| machines past 90 deg pull-out | 20 (worst 164 deg) | **0** (worst 66.6 deg) |
| 5 s self-kick (1e-6, no fault) | 147.72 deg | **0.0018 deg** |
| initialization residual | 1.3e-9 | 1.7e-11 |

Other cases: IEEE9 has no binding limits and is bit-unchanged; IEEE39 pins 3
buses; ACTIVSg200 pins 4; ACTIVSg70k pins 4063 of 8107 generators' buses with 0
residual violations. All were stable before and after — ACTIVSg2000 is the case
where the operating point was actually load-bearing.

### Validation against uqgrid

Same case, same `.dyr`, same fault (bus 1001, r=0.01, 0.2-0.3 s), ZIP alpha 0.5,
both codes with `enforce_q_limits` on:

| quantity | agreement |
|---|---|
| PV→PQ active set | identical — 199 buses, all 2000 bus types match |
| bus voltage magnitude / angle | 6.7e-15 / 1.4e-14 |
| generator reactive dispatch | 2.3e-13 |
| machine speeds, 314 machines x 241 steps | **1.2e-14** elementwise |

The two implementations were written independently from the same standard
algorithm; the trajectory agreement is to round-off.

Note this is agreement between two codes, not against measured behaviour or a
commercial tool. ANDES could not run the case (below), so it is not a third
opinion.

### Caveat that remains

A de-regulated `StaticGenerator` stub holds its reactive output fixed for the
whole transient. A real machine at a limit would re-enter regulation if the
voltage recovered. Modelling that needs the dynamic Q-limit logic that
`plan_enhance.md` tracks separately; it is not part of this change.

Implementing ESST4B natively does not help here, and neither do limiters:
removing exciters entirely changes nothing, because the defect was in the
operating point, not the dynamics.
