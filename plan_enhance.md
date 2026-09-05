# GradPower Capability Enhancement Plan

## 1. Objective

Bring GradPower toward behavioral parity with the Python simulator in `uqgrid/` for dynamic limits and supported device models, while preserving GradPower's architecture:

- current-injection DAE with `z = [device states | network voltages]`;
- explicit `uvec` routing and differentiable `pvec` parameters;
- cluster-contiguous state ordering;
- concrete struct-of-arrays (SoA) tables and generated hot-loop dispatch;
- analytic sparse Jacobians with cached `jac_pos` entries;
- backward Euler with monolithic KLU as the enhancement target;
- scenario-major batched CPU/GPU execution;
- fixed-size device layouts and concrete kernels suitable for batched execution.

This is an implementation plan only. It does not propose copying uqgrid code or adopting uqgrid's internal architecture. uqgrid is the equation, initialization, and trajectory oracle unless a task identifies and documents a uqgrid limitation.

## 2. Scope And Ground Rules

### 2.1 In scope

- Hard differential-state limits, output clamps, selectors, anti-windup, rate limits, and voltage-dependent bounds.
- Fidelity fixes for existing GradPower models.
- Native models present in uqgrid but missing in GradPower.
- DYR parsing, initialization, residuals, analytic Jacobians, coupling, SoA kernels, clustering, and monolithic backend parity for each model.
- Repeatable Julia-versus-uqgrid reference generation and comparisons.
- Dynamic-model coverage reports for the ACTIVS cases.
- Power-flow reactive limits as a separate track because they are required for matching uqgrid initialization on limited cases.

### 2.2 Explicitly not counted as parity

- uqgrid redirects `GGOV1 -> TGOV1`, `EXPIC1 -> SEXS`, `SCRX -> SEXS`, and `ESAC6A -> SEXS` are compatibility surrogates, not implementations of the source equations. GradPower must not report those source models as natively supported if it adopts equivalent redirects.
- CIM5BL and other induction-motor models are out of scope for this plan.
- `uqgrid/andes_code/` and `uqgrid/uqgrid/models/classical_model.py` are not native uqgrid DYR support.
- Deadbands are not in uqgrid's native production models and are not part of this parity effort.
- New integrators, forced-oscillation injectors, PMU emulation, relays, and protection systems are deferred unless separately requested.

### 2.3 Source-of-truth policy

Use current source and tests, not historical planning text, to determine uqgrid behavior. In particular, `uqgrid/PLAN_LIMITS_CONTROLLERS.md` is partially stale because several listed future items are now implemented.

For each behavior, record the oracle in the implementation PR or task notes:

1. uqgrid model source and test.
2. PSS/E/standard documentation if uqgrid is ambiguous.
3. Any intentional GradPower deviation and its architectural reason.

## 3. Current Capability Assessment

### 3.1 Model matrix

| Capability | uqgrid | GradPower | Planned action |
|---|---|---|---|
| GENROU | Native | Native | Keep as regression baseline; reconcile metadata only if needed |
| GENSAL | GENROU specialization | Parsed into GENROU variant | Verify exact parameter mapping; no separate kernel unless behavior differs |
| IEESGO | Native; no PMAX/PMIN enforcement | Native; bounds parsed but ignored | Confirm desired semantics before enabling; do not claim uqgrid-limit parity because uqgrid also leaves these unlimited |
| TGOV1 | Native; hard bound on `x2` | Native; bounds ignored | First fixed-state limiter pilot |
| GAST | Native | Missing | Add after limiter pilot |
| HYGOV | Native; position and rate limits | Missing | Add after fixed and rate-limit primitives |
| IEEEG1 | Native; optional second output, position/rate limits | Missing | Extend coupling, then add |
| SEXS | Native; hard bound on `e_fd` | Native; bounds ignored | Add after TGOV1 limiter pilot |
| ESDC1A | Full five-state model; `vr` limit | Reduced three-state model | Replace reduced implementation with full behavior and migrate references |
| ESDC2A | ESDC1A with voltage-scaled bounds | Missing | Add after full ESDC1A and moving bounds |
| IEEET1 | Native | Missing | Add using DC-exciter shared building blocks |
| EXAC1 | Native; algebraic `e_fd`, rectifier and machine coupling | Missing | Add after shared rectifier/field-current helpers |
| EXAC2 | Native; selectors and two limit layers | Missing | Add after EXAC1 and selector coverage |
| ESAC1A | Native; rectifier and two limit layers | Missing | Add after EXAC1 shared helpers |
| ESST4B | Native; local PI clamps and anti-windup | Missing | High priority due to ACTIVSg2000 prevalence |
| IEEEST | Native MODE 1/local; output clamp and voltage gate | MODE 1; limits disabled | Complete existing model first |
| Static impedance/power load | Two-component alpha blend | Equivalent reduced blend | Preserve behavior; clarify unused GradPower beta/gamma fields |
| CIM5BL | Partial/legacy polar implementation | Missing | Out of scope; do not implement |
| Static generator | PF Q-limit behavior plus dynamic stub | Dynamic stub plus PF Q limits (Phase 6 done) | Dynamic (in-transient) Q limits still absent |

### 3.2 Limiter matrix

| Limit shape | uqgrid examples | GradPower status | GradPower target |
|---|---|---|---|
| Fixed differential-state bound | TGOV1, SEXS, GAST, ESDC1A, IEEET1, EXAC1, EXAC2, ESAC1A, HYGOV, IEEEG1 | Missing | Shared metadata with uqgrid-compatible active-set and experimental complementarity enforcement |
| Voltage-scaled state bound | ESDC2A | Missing | Bound function with voltage derivatives in residual/Jacobian |
| Rate clamp | HYGOV, IEEEG1 | Missing | Piecewise local residual with branch-consistent Jacobian |
| Output clamp and voltage gate | IEEEST | Parsed but ignored | Piecewise algebraic output with branch-consistent Jacobian |
| PI clamp with directional anti-windup | ESST4B | Missing | Model-local piecewise kernel |
| Selector/minimum gate | GAST, EXAC2 | Missing | Deterministic tie convention and branch-consistent Jacobian |
| Saturation curve | GENROU, ESDC1A family, AC exciters | GENROU and reduced ESDC1A only | Reuse tested scalar helpers; preserve exact model equations |
| Power-flow Q active set | Static generators | Missing | PV-to-PQ switching before dynamic initialization |

### 3.3 Architectural differences that affect parity

- uqgrid enforces common hard limits at the integration layer by projection and active-set row replacement. GradPower's documented direction in `docs/plan/phase-16-complementarity.md` is device-local Fischer-Burmeister complementarity.
- uqgrid has generator blend states (`p_m0`, `e_fd0`, `p_m_out`, `e_fd_out`). GradPower routes controller outputs directly through `uvec_idx`; comparison maps must omit uqgrid-only blend states.
- uqgrid inserts off-grid event times. GradPower rounds events to steps and performs a `dt=0` algebraic resolve. Every reference must record whether samples are stored before or after a step and before or after an event; align by physical timestamp and event phase before applying any event-window mask.
- GradPower reorders device states into clusters. Any cached generator, controller, limiter, or secondary-output index must be remapped by `reorder_state!` and `_update_table_pointers!`.
- GradPower production behavior is in SoA kernels. A legacy per-device residual alone is not an implementation.
- Existing measurements do not establish Schur as the fastest general solver. CPU Schur-direct is generally slightly slower than monolithic KLU, and Y-preconditioned Schur-GMRES degrades at large system sizes. New limit and model work therefore targets monolithic backward Euler. Existing Schur paths must keep working for unaffected models, but extending new capabilities to Schur is deferred.

## 4. Required Validation Protocol

Every limiter or model task must use the same gates. Start with a 2-bus fixture; do not begin validation on an ACTIVS case.

### Gate A: parser and layout

- Parse raw DYR fields into a test-side source record before either simulator converts them. Compare that record separately from each simulator's effective object/table/parameter values, recording the conversion formula and machine/system bases.
- Assert device counts, type, bus/ID attachment, sizes, pointer ranges, table values, and contract registration.
- Assert unsupported, redirected, inactive, and unmatched records are reported distinctly.

### Gate B: initialization

- Compare overlapping GradPower and uqgrid initial states, controller references (`pref`, `vref`), effective bounds, and algebraic outputs.
- Require `maximum(abs, rhs_fun!(f, z0, u, p, ps)) <= 1e-9`.
- Verify limit initialization under both policies selected in Phase 1: strict rejection and explicit adjustment with diagnostics.

### Gate C: no-fault flat line

- Integrate at least 1 second with `dt = 1/120` and no event.
- Require maximum state drift `<= 1e-6` for the focused 2-bus fixture unless a tighter existing test applies.
- Before wiring the analytic kernel, evaluate initialization and residuals directly. If integration is needed at this stage, use a temporary finite-difference Newton test harness; production `integrate!` requires the analytic Jacobian.

### Gate D: Jacobian

- Compare the analytic residual Jacobian with `FiniteDiff.finite_difference_jacobian` at initialization.
- Repeat at a perturbed state and at a mid-fault state away from a switching surface.
- For piecewise models, test each branch and one-sided behavior at the switching surface; do not use centered finite differences exactly at a kink.
- Include network-voltage, PSS, generator-state, secondary-generator, and limiter-variable columns where applicable.
- Default maximum absolute error: `1e-5`; tighter model-specific tolerances may be retained.

### Gate E: inactive-limit parity

- Generate uqgrid and GradPower trajectories with limits disabled or made unreachable.
- Require agreement with the pre-feature GradPower baseline and uqgrid on mapped states.
- This proves the limiter machinery does not perturb ordinary operation.

### Gate F: active-limit behavior

- Use a tight-limit fixture that forces activation. A test with no activation is invalid.
- Require matching activation side, activation/release time within one `dt`, bound value within the nonlinear tolerance, state feasibility, and mapped post-release trajectory within Gate G tolerances.
- Require bound violation no larger than the nonlinear solve tolerance plus a documented complementarity smoothing allowance.
- Exercise lower and upper bounds separately.

### Gate G: fault trajectory

- Match RAW/DYR, load alpha, fault bus, resistance, `ton`, `toff`, `dt`, and `tend`.
- Prefer grid-aligned event times. Align pre-step/post-step and pre-event/post-event sample semantics first; only then exclude a documented one-sample window on each side of an event if the integrators cannot represent the same event phase.
- Initial acceptance tolerances: rotor states and voltage magnitude `1e-3`; controller states `5e-3`. Do not loosen without documenting whether the difference is model, limit formulation, or event timing.

### Gate H: architecture and backends

- Use monolithic backward Euler as the correctness reference and first implementation target.
- Run CPU loop versus KernelAbstractions lockstep checks for residual and Jacobian.
- Require zero hot-loop allocations after warm-up.
- Add CUDA table copies and GPU lockstep only after CPU semantics pass. A model may land CPU-first only if parser diagnostics clearly label GPU use unsupported rather than silently omitting it.
- Benchmark monolithic batched cuDSS over batch size and limit-activity distributions. Measure divergent active sets and the larger complementarity system rather than assuming either method is GPU-friendly.
- Run existing Schur regression tests on unaffected models to prevent accidental regressions, but do not add Schur support or performance gates for the new models and limit methods in this plan.

### Gate I: parameter sensitivities

- For differentiable model parameters, compare tangent/adjoint gradients with centered finite differences on an inactive-limit trajectory.
- For active limits, compare smoothed `mu > 0` gradients with finite differences and explicitly classify exact `mu = 0` gradients as generalized, unsupported, or separately validated.
- Include existing non-default sensitivity tests such as `test/test_tlm_genrou_param.jl` in the relevant verification command rather than assuming `Pkg.test()` runs them.

## 5. Phase 0: Freeze Baselines And Build The Comparison Harness

Status: implemented on 2026-08-31. Baseline results and known comparison gaps are recorded in `docs/validation/phase0_baseline.md`; generated machine-readable artifacts remain under `artifacts/phase0/`.

### Task 0.1: Record repository baselines

- Run `Pkg.test()` and record pass/fail/skip counts without fixing unrelated failures.
- Inventory and run representative existing scripts: `scratch_debug_genrou.jl`, `scratch_debug_tgov1_2bus.jl`, `scratch_compare_tgov1_2bus.jl`, `test/regression/compare_*.jl`, and the current `scripts/regression/run_all.jl` runner after preparing frozen references.
- Record Python environment, uqgrid commit/tree hash, Julia version, package manifest hash, and case-file SHA-256 values in generated reference metadata.
- Verify existing unlimited GENROU, GENSAL, TGOV1, SEXS, IEESGO, and ESDC1A comparisons before changing dimensions or equations.

### Task 0.2: Consolidate reference generation

- Generalize the pattern in `scripts/gen_ref_2bus_esdc1a.py` into a small shared Python helper used by model-specific scripts.
- Keep one explicit script per case/model so references remain reproducible and reviewable.
- Save `tvec`, `history`, state names or an index manifest, pointers, effective parameters/bounds, limit diagnostics, case settings, and source hashes.
- Do not rely on positional comments alone. Emit a machine-readable state manifest because GradPower and uqgrid dimensions differ.
- Preserve existing `.npz` files under `examples/refs`; create new names instead of silently overwriting baselines when a model's equations change.
- Extend the existing frozen-reference layout under `artifacts/phase0/references` when it is available or generated; do not create a competing baseline format.

### Task 0.3: Consolidate Julia comparison logic

- Extract reusable setup, mapped-state comparison, event masking, and JSON reporting from `scratch_compare_*` and `test/regression/common.jl`.
- Keep `scratch_debug_*` scripts as focused interactive diagnostics, but make regression scripts assert and exit nonzero.
- Add a comparison result containing maximum absolute error, time and state of first/worst divergence, initial-state error, flat-line drift, and limit activation summary.
- Add all new model comparisons to `scripts/regression/run_all.jl` only after their references are frozen.

### Task 0.4: Add capability coverage reporting

- Add a GradPower DYR coverage report analogous to `uqgrid/scripts/validation/dyr_coverage.py`.
- Classify each record as native, redirected, unsupported, unmatched, inactive, or duplicate.
- Generate reports for ACTIVSg200, ACTIVSg500, and ACTIVSg2000 and compare counts with `uqgrid/docs/validation/activsg_current_coverage.json` and the current uqgrid parser.
- Make coverage percentage a planning metric, not a correctness metric: redirected records do not count as native equation coverage.

### Phase 0 exit criteria

- Existing behavior has frozen, reproducible references.
- A new model can add one Python generator script and one Julia comparison script without duplicating index-alignment logic.
- State mapping is name-based or manifest-based.
- The coverage report explains exactly which active records GradPower skips.

## 6. Phase 1: Implement Comparable Active-Set And Complementarity Limits

### Task 1.1: Write the behavioral contract before code

- Reconcile `docs/plan/phase-16-complementarity.md` with uqgrid behavior in `uqgrid/uqgrid/simulation/dynamic_limits.py`.
- Define one shared limit declaration and two enforcement strategies selected at integration setup: `limit_method=:active_set` and `limit_method=:complementarity`. `:none` disables enforcement for baseline comparisons.
- Treat `:active_set` as the uqgrid-parity and correctness baseline. For implicit BE, replace an active differential row with the exact bound equation, retain the discarded free residual for release logic, and iterate the active set until stable as uqgrid does.
- Treat `:complementarity` as an experimental GradPower-native path. It may use Fischer-Burmeister equations and optional smoothing, but must be compared against the active-set result on the same time grid and tolerances.
- Share bound evaluation, source/effective parameters, initialization validation, state identity, and diagnostics between methods. Do not share solver-specific mutable state or force both methods through one residual shape.
- Specify signs and equations for a limited differential state. The formulation must block outward motion at a bound while allowing immediate inward release; a simple algebraic clamp of the state reference is insufficient.
- Write the exact continuous and backward-Euler equations, residual rows, Jacobian entries, and differential/algebraic classification. One candidate is `xdot = f + lambda_lower - lambda_upper` with `0 <= x-lower perpendicular lambda_lower >= 0` and `0 <= upper-x perpendicular lambda_upper >= 0`; validate the signs against upper/lower release tests before adopting it.
- Specify deterministic generalized derivatives at `(a,b) = (0,0)` and at selector/rate-clamp ties.
- Specify `mu=0` forward behavior and smoothed `mu>0` behavior without changing the default unlimited model.
- Match uqgrid's initialization policy per model: the parser default is `adjust`; HYGOV and IEEEG1 honor the selected `adjust`/`strict` policy, while current uqgrid forces adjustment for ESDC1A/ESDC2A. Preserve source and effective bounds and emit adjustment diagnostics. Also expose `strict` as an explicit validation mode so invalid or surprising source data can be audited without silent mutation.
- Define whether limits are enabled by parser records by default. Match uqgrid per model unless project requirements deliberately override it.
- Define a nonsingular event policy for GradPower's `dt=0` post-event solve. At `dt=0`, a BE differential row can lose multiplier dependence; options must be tested explicitly, such as preserving the converged active set while solving only algebraic/network rows.

### Task 1.2: Prototype fixed bounds on TGOV1 `x2`

- Use uqgrid `GovTGOV1.bounded_state_metadata` and its tests as the oracle: the limited state is `x2`, not lead-lag state `x1`.
- Add a tight two-bus TGOV1 fixture to both simulators with upper and lower activation scenarios.
- Implement uqgrid-compatible active-set TGOV1 first in monolithic BE. This path should preserve the existing state dimension and sparse pattern by replacing the active BE row and Jacobian row at solve time.
- Validate active-set initialization, upper/lower activation, release, and uqgrid trajectory parity before implementing complementarity.
- Implement complementarity as a second TGOV1 strategy. Update `TGOV1` dimensions, table data, cluster metadata, residual/Jacobian structure, and workspaces only for the complementarity layout; do not impose multiplier unknowns on `:none` or `:active_set` runs.
- Compare active-set and complementarity results as `dt -> 0` and, for complementarity, `mu -> 0`. Record trajectory error, activation/release timing, nonlinear iterations, and bound violation.
- Keep the first complementarity implementation in monolithic BE. Defer Schur integration until separate benchmarks establish a need.
- Validate Gates A-I against uqgrid native BE with limits enabled.

### Task 1.3: Add generic fixed-bound metadata without runtime abstraction overhead

- Introduce a build-time declaration describing device type, state offset, lower/upper parameter offsets, and enabled flag.
- Resolve declarations to concrete table columns and global indices during layout construction.
- Keep metadata out of the hot loop; generated dispatch must still see concrete table types.
- Preallocate active-set modes, complementarity workspaces, and diagnostics separately. No per-step dictionaries or heap allocation in kernels.
- Add validation for finite, ordered, non-degenerate bounds and state-in-bound initialization.

### Task 1.4: Solver and event integration

- Thread `limit_method` and the complementarity-only smoothing parameter through `integrate!`, residual evaluation, Jacobian evaluation, and supported batched paths.
- Define active-set stability and complementarity residual convergence criteria for monolithic BE.
- Apply the selected nonsingular limit policy after bus-fault application/removal and during the `dt=0` algebraic resolve.
- Emit compact activation/release diagnostics comparable to uqgrid fields: model, bus/ID, state, side, time, bound, action, and nonlinear iterations.
- Add failure messages that identify the device and bound rather than returning only Newton failure.

### Task 1.5: Compare the two limit methods

- Run the same TGOV1 tight-limit matrix with `:active_set` and `:complementarity` using monolithic BE.
- Compare both methods to uqgrid and to each other for inactive limits, upper/lower activation, release, fault transitions, and decreasing `dt`.
- Sweep complementarity `mu` over `{0, 1e-6, 1e-4, 1e-2}` and record feasibility error, trajectory bias, Newton iterations, failures, and runtime.
- Promote neither method to the default solely from one 2-bus result. Use active-set as the initial default if exact uqgrid parity is the priority; use complementarity only when its robustness, differentiability, or batched performance is demonstrated.

### Task 1.6: Monolithic backend proof

- Verify both methods with loop and KernelAbstractions CPU kernels where applicable.
- Add CUDA table fields and kernels only after CPU semantics pass. Compare GPU and CPU trajectories for each supported method.
- Verify batched active-set scenarios can have different modes without shared mutable state or branch-state leakage. Measure the effect of divergent active sets on GPU execution.
- Evaluate smoothed complementarity as a potentially more uniform GPU path, but measure the cost of extra unknowns and nonlinear iterations.
- Benchmark inactive and active limits on ACTIVSg200 using monolithic batched cuDSS. Record throughput by batch size, memory, iterations, and allocations.
- Run existing Schur tests on old model combinations as regression coverage only; new limiter/model support need not work with Schur in this plan.

### Phase 1 exit criteria

- TGOV1 upper/lower activation side matches uqgrid, activation/release times differ by at most one `dt`, and mapped trajectories satisfy Gate G tolerances.
- Inactive-limit trajectories match the old GradPower baseline.
- Active-set and complementarity results converge toward each other under documented `dt`/`mu` refinement, or their remaining semantic difference is explained and bounded.
- Every enabled solver/backend agrees within configured tolerances; unsupported combinations fail explicitly.
- The implementation pattern is reusable without dynamic dispatch in the hot loop.

## 7. Phase 2: Complete Limits On Existing Models

Implement each task separately and run Gates A-I before proceeding.

### Task 2.1: SEXS hard field-voltage bound

- Apply fixed-state limiting to differential state `e_fd` using `EMIN/EMAX`.
- Compare with `uqgrid/uqgrid/models/sexs_imp.py`, `test_sexs_jacobian.py`, and dynamic-limit tests.
- Add tight upper and lower limit references based on `examples/2bus_SEXS.dyr`.
- Verify PSS input remains in the error signal and the limited `e_fd` is what GENROU receives through `uvec_idx`.

### Task 2.2: IEEEST output clamp and voltage gate

- Match `uqgrid/uqgrid/models/ieeest_imp.py`: clamp `v_s` to `LSMIN/LSMAX`; force output to zero outside `VCL < |V| < VCU` using uqgrid's zero-value conventions.
- This is an algebraic output clamp/gate, not a limited differential state; implement branch-consistent residual and Jacobian without adding unnecessary complementarity states.
- Match uqgrid parser validation: require `MODE == 1`, require `BUSR == 0`, require positive `A2` and `T6`, and enforce paired denominator bypass rules.
- Restore exact bypass semantics for `A3 == A4 == 0`, `T1 == T2 == 0`, and `T3 == T4 == 0` instead of replacing zeros with `0.001`.
- Test unclamped, upper, lower, high-voltage gate, low-voltage gate, each bypass, and `T5 == 0` as a zero washout-gain case.
- Compare the exciter input as well as PSS internal states to catch wiring errors.

### Task 2.3: Resolve IEESGO PMAX/PMIN semantics

- uqgrid's current `GovIEESGO` does not enforce PMAX/PMIN and does not pack them in its parameter vector, while GradPower parses and stores them.
- First compare both implementations with tight source bounds and confirm they remain inactive/unlimited.
- Decide from the PSS/E definition and project requirements whether GradPower should intentionally exceed uqgrid here.
- If parity is the goal, leave IEESGO unlimited and document the bounds as unsupported.
- If physical limits are required, specify whether the bound applies to the pre-turbine `SatP` algebraic signal or mechanical output, add a dedicated reference from a third simulator or hand-derived test, and do not label uqgrid as the oracle for this task.

### Task 2.4: Limit initialization policy and diagnostics

- Reproduce uqgrid's effective parser policy per model, with `adjust` as the default where uqgrid adjusts bounds and `strict` available explicitly for validation.
- Preserve original DYR bounds separately from effective bounds if adjustment is selected.
- Add tests for inverted, non-finite, degenerate, and equilibrium-excluding bounds.
- Include source/effective bounds and adjustment status in comparison artifacts.

### Phase 2 exit criteria

- Existing GradPower controllers enforce every behavior that current uqgrid enforces for the same model.
- IEESGO's intentional status is explicit and tested.
- No model silently changes a source limit during initialization.

## 8. Phase 3: Replace Reduced ESDC1A And Add The DC Family

### Task 3.1: Freeze and retire the reduced ESDC1A baseline

- Before changing GradPower, generate and freeze a GradPower-produced three-state trajectory and state manifest as `2bus_esdc1a_reduced.npz`.
- Preserve the existing uqgrid reference unchanged with its provenance; generate a new full-model uqgrid reference if its source revision or equations do not match current uqgrid.
- Document that GradPower currently omits `Tb/Tc`, `Vrmax/Vrmin`, `Sw`, the `Tr` transducer, and full washout state behavior.
- Add a migration note because changing from three to five differential states changes state indices and cluster sizes.

### Task 3.2: Implement full ESDC1A unlimited equations

- Match uqgrid's states `[vt, ll, vr, e_fd, wf]` and equations in `esdc1a_imp.py`.
- Parse and retain all source parameters, including bypass semantics for `Tr == 0` and `Tb == 0` and the requirement `Tc == 0` when `Tb == 0`. Retain `Sw` for source fidelity but mark it unused by current uqgrid equations.
- Preserve quadratic saturation and its analytic derivative.
- Update exciter output routing from old offset 2 to new `e_fd` offset 3.
- Update table, contract, cluster pointers, PSS coupling, kernels, Jacobian pattern, and all state maps.
- Compare initialization and unlimited fault trajectories to a newly generated uqgrid reference.

### Task 3.3: Add ESDC1A regulator bounds

- Limit `vr`, not `e_fd`, with `Vrmin/effective Vrmax`.
- Match uqgrid's `Vrmax == 0 -> 999` convention unless standards evidence requires another value.
- Exercise strict and adjustment initialization policies.
- Verify saturation and hard-limit activation independently so one does not mask the other.

### Task 3.4: Add voltage-dependent bounds and ESDC2A

- Implement a bound-function interface whose value and derivatives can depend on local bus voltage.
- Add ESDC2A as the ESDC1A equations with `Vrmin*|Vt| <= vr <= Vrmax*|Vt|`.
- Include `vr`, `vi` derivatives in the bound/complementarity Jacobian and sparsity pattern.
- Test moving-bound behavior during a fault, including feasibility while voltage falls and recovers.
- Compare against `uqgrid/uqgrid/models/esdc2a_imp.py` and `tests/test_esdc_models.py`.

### Task 3.5: Add IEEET1

- Reuse only validated scalar saturation and limiter helpers; keep the model kernel explicit.
- Match optional `Tr` state/bypass, `vr` bounds, washout, PSS input, and saturation. Retain `Switch` for source fidelity but mark it unused by current uqgrid equations; implementing switch-dependent physics requires a non-uqgrid oracle.
- Prefer a fixed state dimension if it simplifies SoA/GPU layout; if a bypass state is retained, set a well-defined residual row and verify it does not create a singular cluster block.
- Compare with `uqgrid/uqgrid/models/ieeet1_imp.py` and `tests/test_ieeet1.py`.

### Phase 3 exit criteria

- ESDC1A state and trajectory parity uses the full model.
- ESDC2A moving bounds work in all CPU solvers.
- IEEET1 shares tested primitives without creating a generic block-diagram interpreter.

## 9. Phase 4: Add Governor Models

### Task 4.1: GAST unlimited model and selector

- Parse `R,T1,T2,T3,AT,KT,VMAX,VMIN,DT` with correct machine/system-base conversion.
- Add states `[x1,x2,x3]`, algebraic `p_m`, speed input, and `pref` initialization.
- Implement `min((pref-w)/R, AT + KT*(AT-x3))` with uqgrid's tie convention.
- Add branch-specific Jacobian tests for demand and temperature branches.
- Compare against `uqgrid/data/2bus_GAST.dyr` and `gast_imp.py` before enabling bounds.

### Task 4.2: GAST position bound

- Limit `x1` to `VMIN/VMAX` using the fixed-bound infrastructure.
- Use `uqgrid/data/2bus_GAST_tight.dyr` to prove upper/lower activation and release.
- On an ACTIVSg500 model subset with fixed/frozen operating-point inputs, confirm the six active GAST records are now native in GradPower's coverage report. Defer full-case initialization and trajectory parity until Phase 6 power-flow Q limits are complete.

### Task 4.3: Add reusable local rate-clamp semantics

- Define a branch-stable scalar rate clamp returning value and derivative multiplier.
- Specify derivatives as zero on clipped branches and document equality behavior.
- Test lower, interior, upper, and exact-bound cases independently of a device.
- Keep this as an inline scalar helper usable from CPU/KA/GPU kernels, not a runtime object hierarchy.

### Task 4.4: HYGOV unlimited hydraulic model

- Parse and base-convert `R,r,Tr,Tf,Tg,VELM,GMAX,GMIN,Tw,At,DT,qNL` and define the GradPower equivalent of uqgrid's positive `g_floor`.
- Add states `[LG,gtpos,g,q]`, algebraic `p_m`, speed input, and initialization `q0=p_m0/At+qNL`.
- Implement the hydraulic head and mechanical-power equations first with rate and position limits disabled.
- Test both branches of `g_eff=max(g,g_floor)` and their Jacobians.
- Compare against `uqgrid/data/2bus_HYGOV.dyr`.

### Task 4.5: HYGOV rate and position limits

- Apply local `[-VELM,+VELM]` clipping to `d(gtpos)/dt` and fixed-state bounds `GMIN/GMAX` to `gtpos`.
- Verify interaction at a position bound: outward rate is blocked, inward movement releases.
- Compare activation and trajectories with `2bus_HYGOV_tight.dyr`.
- On an ACTIVSg500 model subset with fixed/frozen operating-point inputs, confirm the 35 active HYGOV records plus GAST materially improve native coverage. Defer full-case initialization and trajectory parity until Phase 6.

### Task 4.6: Extend coupling for IEEEG1 secondary output

- Extend `produces_signals` and cluster construction to support one controller feeding two generators by `(BUS2,ID2)`.
- Route the secondary algebraic output explicitly in the monolithic Jacobian and control map. Do not redesign clusters or Schur extraction for this feature.
- Mark IEEEG1 secondary-output cases unsupported by Schur until a separate solver project addresses cross-cluster coupling.
- Reject missing/inactive secondary targets with a precise parser diagnostic.

### Task 4.7: IEEEG1 equations, bypasses, and outputs

- Implement six differential states, one or two algebraic outputs, coefficient normalization, optional `T1` bypass, and zero turbine-stage time-constant behavior.
- Preserve source and effective `K1..K8` values in diagnostics.
- Validate primary-only and dual-output initialization, including inconsistent initial mechanical powers.
- Compare against `uqgrid/uqgrid/models/ieeeg1_imp.py` and `tests/test_ieeeg1.py`.

### Task 4.8: IEEEG1 rate and position limits

- Apply `[UC,UO]` valve-rate clipping and `[PMIN,PMAX]` valve-position limiting.
- Test strict/adjust initialization, both rate branches, both position bounds, and release.
- Compare with `uqgrid/data/2bus_IEEEG1.dyr` and a new tight fixture if the existing case does not activate every branch.

### Phase 4 exit criteria

- GAST and HYGOV match uqgrid on focused fixtures and fixed-operating-point ACTIVSg500 model subsets; full-case parity is gated by Phase 6.
- IEEEG1 works for primary-only and secondary-output topology.
- Cross-generator secondary output is represented explicitly in the monolithic Jacobian and fails clearly if a Schur solver is requested.

## 10. Phase 5: Add AC And Static Exciter Families

### Task 5.1: Build and test shared scalar physics helpers

- Implement commutating-rectifier value/slope functions used by EXAC1, EXAC2, ESAC1A, and ESST4B.
- Implement generator field-current/saturation coupling from GENROU states with analytic derivatives.
- Test every rectifier interval boundary (`0`, `0.433`, `0.75`, `1.0`) with one-sided derivatives.
- Keep helpers scalar, allocation-free, and GPU-compatible; do not create a generic block graph.

### Task 5.2: EXAC1

- Add states `[vt,ll,vr,ve,wf]` and algebraic `e_fd`.
- Extend exciter coupling so `produces_signals` can route an algebraic output, not only a differential offset.
- Implement transducer/lead-lag/washout bypasses, machine field current, exciter saturation, rectifier, `vr` bound, and initialization Newton solve.
- Add generator-state and generator-algebraic cross-couplings to sparsity and `jac_pos`.
- Compare with `exac1_imp.py` and `tests/test_exac1.py` on initialization, branch tests, and fault trajectories.

### Task 5.3: ESAC1A

- Reuse validated field-current, saturation, and rectifier helpers.
- Implement shared `va` state bound plus internal `VRMIN/VRMAX` clamp.
- Preserve strict initialization semantics for both limit pairs.
- Compare with `esac1a_imp.py` and `tests/test_esac1a.py`.

### Task 5.4: EXAC2

- Implement `va` state bounds, internal `VRMIN/VRMAX` clamp, `VLR` selector, field-current/rectifier coupling, and initialization adjustments.
- Test both high/low selector branches, both hard-bound sides, internal clamp branches, and adjusted `VLRx` behavior.
- Compare with `exac2_imp.py` and `tests/test_exac2.py`.

### Task 5.5: ESST4B

- Prioritize this model because the uqgrid ACTIVSg2000 coverage fixture contains 212 active ESST4B records.
- Add states `[v_sensed,xi_r,v_lag,xi_m]` and algebraic `e_fd`.
- Implement outer and inner PI clamps with directional anti-windup, optional `TR/TA` bypasses, `VGMAX`, potential-source calculation, rectifier, and `VBMAX`.
- Keep these limits model-local as in uqgrid; do not force PI anti-windup into the fixed-state complementarity abstraction.
- Test every clamp, anti-windup release direction, source ceiling, bypass, and rectifier branch.
- Compare with `esst4b_imp.py` and `tests/test_esst4b.py`, then run an ACTIVSg2000 subset containing ESST4B devices.

### Phase 5 exit criteria

- Algebraic-output exciters route correctly to GENROU.
- All machine-coupled Jacobian columns pass finite differences.
- ESST4B scales to a representative multi-device subset without allocation or solver regressions.

## 11. Phase 6: Power-Flow Reactive Limits

**Status: implemented** (`runpf!(...; enforce_q_limits)`, default off for the
library, on in the data-generation pipeline). Tasks 6.1-6.3 are done and both
exit criteria are met on ACTIVSg2000: GradPower and uqgrid select an identical
active set (199 buses, all 2000 bus types agree), and the dynamic
initialization residual is 1.7e-11. Trajectory agreement is 1.2e-14 elementwise
over 314 machines x 241 steps. Tests: `test/test_pflow_qlimits.jl`. Analysis:
`docs/activsg2000-diagnosis.md`.

One deviation from Task 6.2 as written: the active set is applied by mutating
`bus.type` and `gen.qsch` in place rather than preserving the source objects and
recording the effective mode in the solution. That keeps `runpf` itself
untouched, but it means a caller re-solving the same system at a new operating
point has to restore the original bus types first, which
`scripts/dynstab_io.jl`'s `apply_load_scale!` now does explicitly. Worth
revisiting if the power flow ever needs to be re-entrant.

Not covered here: a machine pinned at a reactive bound stays pinned for the
whole transient. Re-entering regulation when the voltage recovers is dynamic
Q-limit behaviour, tracked separately.

This is separate from dynamic hard limits. It changes the operating point and therefore every dynamic initialization comparison.
Execute this track after Phase 3 and before any full ACTIVSg500 or ACTIVSg2000 initialization/trajectory parity gate. Focused cases may proceed earlier only when their Q limits are demonstrably inactive.

### Task 6.1: Retain generator capabilities

- Propagate already-parsed PSS/E and MATPOWER capability fields into `Gen` storage, preserving units, status, and per-generator identity.
- Preserve bounds through aggregation and static-generator construction.
- Add parser tests for multiple generators at one bus and base units.

### Task 6.2: Add PV-to-PQ active-set power flow

- Use uqgrid `simulation/pflow.py` as behavioral reference for non-slack PV buses.
- Compute aggregate bus Q bounds, detect violations, fix Q at the active bound, and re-solve with the bus treated as PQ.
- Preserve original bus type and schedules in source objects; store effective solve mode in the solution.
- Define bounded reactive sharing among multiple generators and test both aggregate and individual limits.

### Task 6.3: Propagate the limited operating point

- Initialize `StaticGenerator` and dynamic devices from the limited PF result.
- Verify voltage setpoint residuals are not imposed on buses switched to PQ.
- Compare bus types, voltages, generator Q, and dynamic `z0` against uqgrid Q-limit fixtures.

### Phase 6 exit criteria

- GradPower and uqgrid select the same active Q bounds and effective PV/PQ buses.
- Dynamic initialization remains at machine-precision residual after switching.

## 12. Phase 7: Compatibility Redirects And Large-Case Closure

### Task 7.1: Define redirect policy

- Decide explicitly whether to adopt uqgrid's compatibility redirects for GGOV1, EXPIC1, SCRX, and ESAC6A.
- If adopted, store `source_model`, `effective_model`, ignored source parameters, and reason in parser diagnostics.
- Warn once per source model with counts, not once per device on large cases.
- Never count redirects as native model coverage or silently use them in scientific validation.

### Task 7.2: ACTIVSg500 closure

- Regenerate coverage and require native handling of GENROU, SEXS, TGOV1, GAST, HYGOV, and supported IEEEST records.
- Compare PF, `z0`, flat-line drift, first divergence, limiter activations, and fault trajectories.
- Use `scratch_first_divergence.jl` style diagnostics when aggregate error fails.

### Task 7.3: ACTIVSg2000 staged closure

- Validate model-family subsets before the full case: DC exciters, AC exciters, ESST4B, governors, and IEEEST.
- Do not use redirected GGOV1/EXPIC1/SCRX/ESAC6A records to mask missing native equations.
- Compare coverage counts before comparing trajectories; a trajectory comparison is not meaningful if controller populations differ.
- Run monolithic CPU first, then monolithic batched/GPU.

### Task 7.4: Performance and memory acceptance

- Benchmark unlimited baseline, limits present but inactive, and limits active.
- Record residual/Jacobian kernel time, Newton iterations, memory, and allocations.
- Confirm added table fields and complementarity rows do not regress scenarios with no limited devices.
- Add representative GPU batch tests with different limit activity per scenario.

### Phase 7 exit criteria

- Coverage reports, state manifests, and trajectory artifacts agree on which models are being simulated.
- Large-case failures identify a model family and first divergent state rather than only a global norm.
- Performance impact is measured and accepted explicitly.

## 13. Per-Model Implementation Checklist

Use this checklist for every new native device. A task is not complete if any applicable item is missing.

- Add the device struct and exact DYR field parser with validation.
- Add machine/system-base conversion tests against uqgrid effective parameters.
- Define `diff_size`, `alg_size`, `ctrl_size`, and `par_size`.
- Add `fill_pvec!`, `initial_guess!`, `initialize_dynamics!`, and `extract_init_params!` where needed.
- Add a concrete SoA table, builder, and `register_device!` entry.
- Add a `DeviceContract` and aliases only for mathematically equivalent models.
- Add coupling traits and special routing only when the existing trait vocabulary is insufficient.
- Add cluster symbol handling and pointer remapping for every cached state index.
- Add residual kernel, analytic Jacobian kernel, sparsity preallocation, and `jac_pos` population.
- Add online/offline behavior and ensure disconnect does not leave stale controller signals.
- Add differential-index bookkeeping for backward Euler.
- Add focused parser, initialization, flat-line, branch, Jacobian, and trajectory tests.
- Add a uqgrid reference generator and a Julia comparison script with a state manifest.
- Define stable differential, algebraic, control, and parameter names and emit `(model, bus, id, local_name, global_index)` entries after cluster reordering.
- Add CPU loop/KA lockstep and zero-allocation tests.
- Add CUDA table transfer and GPU tests before declaring GPU support.
- Add the model to coverage reports and the regression runner.

## 14. Recommended Delivery Order

The order minimizes architectural risk and maximizes useful case coverage:

1. Comparison harness and coverage reporting.
2. Fixed-state limiter contract using TGOV1.
3. SEXS and IEEEST existing-model limits.
4. Full ESDC1A, then ESDC2A and IEEET1.
5. Power-flow Q limits before any full ACTIVS initialization/trajectory parity claim.
6. GAST, then HYGOV, using focused fixtures and model subsets first.
7. IEEEG1 coupling extension and model.
8. Shared AC-exciter physics, then EXAC1 and ESAC1A/EXAC2.
9. ESST4B, followed by ACTIVSg2000 subset validation.
10. Optional compatibility redirects and full large-case closure.

Do not implement all model parsers first and defer kernels/tests. Complete one vertical slice at a time so every merged model is executable and comparable.

## 15. Resolved Decisions

### Decision 1: Complementarity versus uqgrid active-set rows

Use both behind one shared limit declaration. Implement uqgrid-compatible active-set row replacement first as the behavioral baseline, then implement Fischer-Burmeister complementarity as an experimental alternative. Compare both with monolithic BE. Select defaults by measured correctness, robustness, differentiability, and batched throughput rather than requiring one method to replace the other.

### Decision 2: State dimensions for bypasses and limits

Use fixed dimensions per model. For a bypassed state that must remain frozen, set its continuous residual to `f_i = 0`; backward Euler then gives `z_i - zold_i = 0`, and the existing BE Jacobian transformation naturally produces a row with zero off-diagonals and `1` on the diagonal. For a bypass that must transmit an input, use a constraint such as `z_i - input = 0` and retain the `-d(input)/dz` entries; replacing that row with diagonal `1` alone would be incorrect. Apply row replacement after ordinary residual/Jacobian assembly, preallocate every possible bypass coupling, and test rank plus finite differences for each bypass configuration.

### Decision 3: Initialization adjustment policy

Follow uqgrid per model so reference trajectories start from the same effective system. Use `adjust` as the normal parser default where uqgrid adjusts limits, preserve source and effective bounds, and emit diagnostics for every adjustment. Also provide explicit `strict` mode for audits and tests; it rejects an equilibrium outside enabled bounds. Never mutate source fields without retaining them in metadata.

### Decision 4: IEESGO bounds

Match uqgrid: parse and retain IEESGO PMAX/PMIN for source fidelity, but do not enforce them. Mark these bounds as unsupported in capability diagnostics. Any future physical implementation requires a separate standards-based task and a non-uqgrid oracle.

### Decision 5: Schur scope

Do not extend Schur for the new limits and models in this plan. Use monolithic KLU as the CPU correctness and production path and monolithic batched cuDSS as the initial GPU path. Keep existing Schur tests as regression coverage for existing model combinations. IEEEG1 secondary output is implemented as an explicit monolithic cross-generator coupling and rejected when a Schur solver is requested.

### Decision 6: CIM5 scope

CIM5BL and other induction-motor models are excluded from this plan. Coverage reporting must continue to identify CIM5BL records as unsupported; no parser redirect or static-load substitution may claim native parity.

## 16. Final Acceptance Criteria

The enhancement program is complete when:

- GradPower's capability report distinguishes native, redirected, unsupported, inactive, and unmatched DYR records.
- Every claimed native model passes parser, initialization, flat-line, branch, finite-difference Jacobian, and fault-trajectory gates against uqgrid or a documented alternate oracle.
- Every claimed limiter has a test that proves activation and release; inactive-limit tests preserve prior behavior.
- Existing and new models work through concrete SoA kernels and analytic sparse Jacobians, not fallback object dispatch.
- Monolithic is the correctness reference; CPU/KA and advertised monolithic GPU combinations match it within tolerance. Existing Schur combinations retain their prior regression coverage, while new model/limit combinations reject Schur explicitly.
- Differentiable parameters pass tangent/adjoint versus finite-difference gates under the documented inactive and smoothed-active limit regimes.
- Large-case comparisons start from equivalent model populations and power-flow operating points.
- Reference generation is reproducible from scripts and includes state manifests, effective parameters, diagnostics, and source hashes.
- Redirects and partial models are never reported as full native parity.
