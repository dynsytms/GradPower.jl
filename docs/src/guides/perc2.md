# Data-center loads (PERC2)

[`PERC2Load`](@ref) represents a large power-electronic load, such as a data
center, as one aggregate device at a bus. It implements PERC2 (Power
Electronic Reconnecting and Ceasing, version 2), a positive-sequence load
model endorsed by the NERC Load Modeling Working Group in July 2026. The
parameter names follow the model's 6 flags and 57 constants, so a `.dyr`
record maps onto [`PERC2Params`](@ref) field by field.

Reference: NERC, *Data Center Load Modeling Technical Reference*, July 2026,
<https://www.nerc.com/globalassets/who-we-are/standing-committees/rstc/data_center_load_modeling_technical_reference_r3_clean.pdf>.

## What the model represents

The load draws an active current `Ip` and a reactive current `Iq` from its
bus. Both are computed from the filtered bus voltage `Vtf` and the filtered
local frequency, in five steps:

1. **Demand.** The active power request is the initial loading `Ldfact` plus a
   frequency droop and an optional voltage washout, passed through an optional
   lead-lag. The reactive request is built the same way, without droop. The
   requests are converted to currents with the voltage exponents `nP` and `nQ`:
   ``I_p = P_\text{ctrl}\,(V_{tf}/V_0)^{n_P} / V_{tf}``. With `nP = 0` the load
   draws constant power.
2. **Current limits.** `Ip` is limited to `[Ipmin, Ipmax]` and `Iq` to
   `[Iqmin, Iqmax]`, and the total current to `Imax`, with priority to the
   active current.
3. **Cease and reconnect.** Cease checks compare `Vtf` (or the filtered
   frequency) with thresholds. When a check stays active for its delay, the
   load drops a fraction `fcease` of its demand. When the voltage returns to
   the reconnect window for long enough, the load ramps back to a fraction
   `frecon` of the ceased part.
4. **FracOn.** The online fraction `FracOn` multiplies both currents. It is 1
   in normal operation, `1 - fcease` while ceased, and
   `(1 - fcease) + frecon·fcease` after a full reconnection. With the default
   `fcease = 0.9` and `frecon = 0.5`, a ceased load keeps 10 % of its demand
   and returns to 55 %.
5. **Output lag.** A first-order lag with time constant `Tt` smooths the
   currents sent to the network.

There are three kinds of cease checks:

| level | checks | condition | reconnect |
|:--|:--|:--|:--|
| momentary | `VLM`, `VHM` | `Vtf < VLcease` or `Vtf > VHcease` for `Tcease` | window `(VLrecon, VHrecon)` for `Trecon`, ramp `Tramp·Ldfact` |
| long, low voltage | `VLC1` … `VLC4` | `Vtf < VLCi` for `TVLCi` | same window for `TLreconL`, ramp `TLrampL·Ldfact` |
| long, high voltage | `VHC1` … `VHC4` | `Vtf > VHCi` for `TVHCi` | same window for `TLreconH`, ramp `TLrampH·Ldfact` |
| frequency | `FLC1`, `FHC1` | `f < FLC1` or `f > FHC1` (Hz) for `TFLC1`, `TFHC1` | never |

Each group is enabled by its flag. A cease decision takes effect `Tdelay`
seconds after a check has timed out. While the load is ceased, a check of a
higher level can still fire. It does not change `FracOn`, but it replaces the
reconnect settings by those of the higher level. A frequency cease is
therefore permanent for the rest of the run.

The default [`PERC2Params`](@ref) enable only the momentary low-voltage check
(`VLMflag`), which reproduces the behaviour of the first PERC version.

## Parameters

`PERC2Params` is a keyword structure: `PERC2Params(VLcease = 0.7, Tcease = 0.02)`
changes two settings and keeps the defaults for the others. Units:

- currents and powers are per unit on the model base
  ``M_\text{base} = P_0 / L_\text{dfact}``, where `P0` is the initial active
  power of the PERC2 part. With `Ldfact = 0.8`, `Ipmax = 1.0` lets the load
  draw 1.25 `P0` of active current at 1 pu voltage;
- voltages are bus per unit;
- frequencies are Hz, except the droop deadband, which is per unit of `fnom`;
- times are seconds.

**Flags** (`true` enables the check group):

| field | default | enables |
|:--|:--|:--|
| `VLMflag` | `true` | momentary low-voltage cease |
| `VHMflag` | `false` | momentary high-voltage cease |
| `VLLflag` | `false` | long low-voltage cease (`VLC1`–`VLC4`) |
| `VLHflag` | `false` | long high-voltage cease (`VHC1`–`VHC4`) |
| `FLflag` | `false` | low-frequency cease (`FLC1`) |
| `FHflag` | `false` | high-frequency cease (`FHC1`) |

**Base, filters and droop:**

| field | default | unit | meaning |
|:--|:--|:--|:--|
| `Ldfact` | 0.8 | — | initial loading, `P0 / Mbase` |
| `QPratio` | 0.0 | — | initial reactive power of the dynamic path, as a ratio to `P0` (see [Reactive power and `QPratio`](@ref perc2-qpratio)) |
| `Tt` | 0.02 | s | output current lag |
| `Tv` | 0.05 | s | voltage measurement filter |
| `Tf` | 0.02 | s | frequency measurement filter |
| `dbfl` | 0.0 | pu of `fnom` | lower edge of the droop deadband |
| `dbfh` | 0.0 | pu of `fnom` | upper edge of the droop deadband |
| `Kdroop` | 8.0 | pu / pu | active power change per unit frequency deviation outside the deadband |

**Washouts and lead-lags** (inactive by default):

| field | default | unit | meaning |
|:--|:--|:--|:--|
| `Kvp`, `Tvp` | 0.0, 0.05 | pu·s, s | voltage washout ``sK_{vp}/(1+sT_{vp})`` added to the active request |
| `Kvq`, `Tvq` | 0.0, 0.1 | pu·s, s | voltage washout added to the reactive request |
| `Tap`, `Tbp` | 1.0, 1.0 | s | lead-lag ``(1+sT_{ap})/(1+sT_{bp})`` on the active request |
| `Taq`, `Tbq` | 1.0, 1.0 | s | lead-lag on the reactive request |

A washout with `K = 0` and a lead-lag with `Ta = Tb` or `Tb = 0` are bypassed.
The washouts act on the raw bus voltage. A voltage step ``\Delta V`` gives a
power response ``(K/T)\,\Delta V\,e^{-t/T}``, so a voltage drop lowers the
demand.

**Voltage exponents and current limits:**

| field | default | unit | meaning |
|:--|:--|:--|:--|
| `nP`, `nQ` | 0.0, 0.0 | — | voltage exponents of the active and reactive power |
| `Ipmax`, `Ipmin` | 1.0, 0.0 | pu | active current limits |
| `Iqmax`, `Iqmin` | 0.7, −0.7 | pu | reactive current limits (positive: inductive) |
| `Imax` | 1.5 | pu | total current limit |

**Momentary cease and reconnect:**

| field | default | unit | meaning |
|:--|:--|:--|:--|
| `fcease` | 0.9 | — | fraction of the load that ceases |
| `VLcease`, `VHcease` | 0.5, 1.2 | pu | cease thresholds |
| `Tcease` | 0.05 | s | time a momentary check must stay active |
| `Tdelay` | 0.0 | s | delay between a timed-out check and the cease (all levels) |
| `frecon` | 0.5 | — | fraction of the ceased part that reconnects |
| `VLrecon`, `VHrecon` | 0.6, 1.1 | pu | reconnect window (all voltage levels) |
| `Tramp` | 1.25 | s | ramp time at `Ldfact = 1`; the ramp lasts `Tramp·Ldfact` |
| `Trecon` | 0.05 | s | time in the window before the ramp starts |

**Long low-voltage cease:**

| field | default | unit | meaning |
|:--|:--|:--|:--|
| `TLrampL`, `TLreconL` | 1.25, 0.05 | s | ramp and reconnect times after a long low-voltage cease |
| `VLC1`, `TVLC1` | 0.9, 2.0 | pu, s | point 1 |
| `VLC2`, `TVLC2` | 0.8, 0.5 | pu, s | point 2 |
| `VLC3`, `TVLC3` | 0.0, 0.0 | pu, s | point 3 (off by default: `Vtf < 0` never holds) |
| `VLC4`, `TVLC4` | 0.5, 0.15 | pu, s | point 4 |

**Long high-voltage cease:**

| field | default | unit | meaning |
|:--|:--|:--|:--|
| `TLrampH`, `TLreconH` | 1.25, 0.05 | s | ramp and reconnect times after a long high-voltage cease |
| `VHC1`, `TVHC1` | 1.1, 1.0 | pu, s | point 1 |
| `VHC2`, `TVHC2` | 1.15, 0.5 | pu, s | point 2 |
| `VHC3`, `TVHC3` | 99.0, 0.0 | pu, s | point 3 (off by default) |
| `VHC4`, `TVHC4` | 1.2, 0.16 | pu, s | point 4 |

**Frequency cease:**

| field | default | unit | meaning |
|:--|:--|:--|:--|
| `FLC1`, `TFLC1` | 58.0, 100.0 | Hz, s | low-frequency threshold and time |
| `FHC1`, `TFHC1` | 62.0, 100.0 | Hz, s | high-frequency threshold and time |

**GradPower options** (not part of the `.dyr` record):

| field | default | meaning |
|:--|:--|:--|
| `fnom` | 60.0 | nominal frequency (Hz); all PERC2 loads in a system must use the same value |
| `tt_floor_steps` | 2 | `Tt` is raised to `tt_floor_steps·dt` |
| `use_qpratio` | `true` | `false` puts all of the power-flow reactive power on the dynamic path and ignores `QPratio` |

Appendix H of the NERC reference gives example settings. Parameters for a
study must come from facility data.

## Model conventions

The published specification leaves some details open, and some inputs can be
inconsistent. GradPower makes the following choices.

- **Reconnect window.** Reconnection requires
  `VLrecon < Vtf < VHrecon` (strict inequalities) without interruption for the
  reconnect time. `VLrecon` is raised to `VLcease` if it is lower, and
  `VHrecon` is lowered to `VHcease` if it is higher, whether or not `VHMflag`
  is set.
- **Ramps.** A cease during a ramp interrupts it: `FracOn` drops back to
  `1 - fcease`. All check timers restart when a ramp starts.
- **Frequency cease.** It never reconnects.
- **Frequency input at low voltage.** While the bus voltage is below 0.7 pu,
  the frequency input is set to nominal, so the filtered frequency and the
  droop term relax towards zero instead of following an angle estimate taken
  during the fault.
- **Current limits.** The active limit is `min(Ipmax, |Imax|)`. The reactive
  limits are `Iqmax` and `Iqmin`, further restricted to
  ``\pm\sqrt{I_\text{max}^2 - I_p^2}``, where `Ip` is the limited active
  current. A negative `Imax` is used as `|Imax|`.
- **Defaults for out-of-range inputs.** `Ldfact ≤ 0` is replaced by 0.8.
  `fcease` is clamped to [0, 1], and a negative `frecon` is treated as 0.
- **Output lag floor.** `Tt` is raised to `tt_floor_steps·dt` (2 steps by
  default) so the lag stays resolved by the time step. Whether the floor was
  applied is recorded in the `tt_floor_applied` field of the log and of
  [`perc2_states`](@ref).
- **Long point 3.** `VLC3` and `VHC3` are timed by their own `TVLC3` and
  `TVHC3`.
- **Voltage floor.** The bus voltage is floored at 1e-3 pu before it enters
  the model, so a bolted fault gives finite currents.
- **Initial limits.** If the current limits bind at the power-flow voltage, the
  load starts at the limited current and draws less than `P0 + jQ0`. The
  initial state is then not an equilibrium.

`PERC2Params` are checked when the load is added. `Imax = 0` is rejected
because it would zero every current limit. Non-finite values, negative time
constants, `Ipmax < Ipmin`, `Iqmax < Iqmin`, a lead-lag with `Ta > 0` and
`Tb = 0`, and a washout with `K ≠ 0` and `T ≤ 0` are rejected as well.

### [Reactive power and `QPratio`](@id perc2-qpratio)

The power flow gives the PERC2 part an initial reactive power `Q0`. With the
default `use_qpratio = true`, the dynamic reactive path starts at
`Mvarinit = QPratio·P0`. The remainder is drawn as a constant current

```math
I_{q,\text{extra}} = \frac{Q_0 - \text{QPratio}\cdot P_0}{V_0}
\qquad \text{(system base)}.
```

`Iqextra` bypasses `nQ`, the reactive washout and lead-lag, and the current
limits, but it is multiplied by `FracOn` and passes through the `Tt` lag: it
ceases and reconnects with the load. With the default `QPratio = 0`, all of
`Q0` is `Iqextra`. With `QPratio = Q0/P0`, or with `use_qpratio = false`,
`Iqextra = 0` and all of `Q0` follows the dynamic path.

## Numerical treatment

`PERC2Load` has no differential or algebraic states in `z`. During each Newton
solve, the network sees the consumed current

```math
I = (I_p - j I_q)\,\frac{V}{|V|},
```

where `Ip` and `Iq` are held fixed in the parameter vector. The residual and
its analytic Jacobian depend on the bus voltage only through the direction
``V/|V|``. Below ``|V| = 10^{-6}`` pu the direction of the last accepted step
is used.

The filters, limits and cease/reconnect logic advance once per **accepted**
step, from the converged bus voltage and frequency at time `t_k`. The new
currents are then held during the step from `t_k` to `t_{k+1}`, so the load
lags the network by one step. The filters, washouts, lead-lags and the output
lag use the exact exponential update ``x \leftarrow x + (u - x)(1 - e^{-dt/T})``
with the input held over the step. When a held current changes by more than
1e-6 pu, the next solve refactors the Jacobian.

This partitioning has three consequences.

- **Integrator order.** With `integrator = :crank_nicolson`, the machines and
  controllers keep the second-order formula. The PERC2 internal dynamics are
  discrete updates that are first order in time. In the fault example below,
  backward Euler and Crank–Nicolson give the same cease and reconnect times
  and differ in the PERC2 power by at most 1.3e-3 pu.
- **Events.** The algebraic re-solve after an event (a `dt = 0` solve) does
  not advance the PERC2 logic.
- **Local frequency.** The bus frequency is estimated from the rate of change
  of the bus voltage angle over the last accepted step,
  ``f = f_\text{nom} + \Delta\theta / (2\pi\,\Delta t)``. It lags the true
  frequency by about `dt/2`. After a switching event, the angle reference is
  reset to the post-event angle, so the angle jump of the event is not read as
  a frequency deviation. Below 1e-3 pu the last estimate is held.

Every call to `integrate!` restarts the PERC2 logic from the initial operating
point at the first time of the run.

## Supported features

| feature | status |
|:--|:--|
| `integrate!` with `solver = :monolithic`, `:backward_euler` and `:crank_nicolson` | supported |
| `limit_method = :active_set` / `:complementarity` | runs; these methods act on TGOV1 valve limits only, the PERC2 limits are internal to the model |
| bus faults, line trips, generator trips, load steps | supported |
| tripping the PERC2 part | `DisconnectDeviceEvent(find_device(sys, :perc2load, bus, id), t)` |
| `add_load_trip!`, `LoadStepEvent`, `set_load!` on a split load | act on the static remainder only |
| [`GradPower.integrate_batched!`](@ref) on CPU, threaded or not | supported: one `PERC2Params` shared by all scenarios; `P0`, `Q0`, `V0` and the initial state per scenario |
| queries (`find_devices`, `find_device`, `get_parameter`, `parameter_index`) | supported; all PERC2 parameters are read-only for `set_parameter!` |
| `solver = :schur`, `:schur_gmres` | `ArgumentError` |
| GPU integrators, `CUDSSBackend`, KernelAbstractions residuals | `ArgumentError` |
| `GradPower.DynamicNLP` (optimization, sensitivities) | `ArgumentError` |

## Adding a PERC2 load to a case

[`add_perc2_load!`](@ref) replaces a fraction of an existing static load with a
PERC2 load. The remainder stays in the static ZIP load, and the power flow is
unchanged, so the split system starts in equilibrium. Call it after `runpf!`
and before creating the `DynamicProblem`.

We replace 80 % of the load at bus 5 of the WSCC 9-bus case and apply a
4.5-cycle fault at bus 4. The same fault is simulated twice: with the default
settings, and with a facility that ceases below 0.7 pu after 20 ms.

```@example perc2
using GradPower, CairoMakie
ex = joinpath(pkgdir(GradPower), "examples")

function run_perc2(params; tf = 4.0, integrator = :backward_euler)
    sys = from_psse(joinpath(ex, "ieee9_v33.raw"), joinpath(ex, "ieee9bus_gov.dyr"))
    build_network!(sys); runpf!(sys)
    add_perc2_load!(sys, 5, "1"; fraction = 0.8, params = params)
    dp = DynamicProblem(sys); initialize_dynamics!(dp, sys)
    add_event!(sys, ContingencyEvent(sys.busmap[4], 0.02, 1.0, 1.075))
    log = NamedTuple[]
    integrate!(dp, sys, tf; integrator, perc2_log = log)
    return sys, log
end

sys_a, log_a = run_perc2(PERC2Params())
sys_b, log_b = run_perc2(PERC2Params(VLcease = 0.7, VLrecon = 0.8, Tcease = 0.02))
perc2_states(sys_b)[1].events
```

`perc2_log` receives one named tuple per PERC2 load at the initial time and
after every accepted step. Its fields include the time `t`, the bus voltage
`vm`, the frequency estimate `f_hz`, `Vtf`, `FracOn`, the consumed power `P`
and `Q` (system-base pu), `mode` (0 monitoring, 1 ceased, 2 ramping) and
`level` (−1 none, 0 momentary, 1 long, 2 frequency). [`perc2_states`](@ref)
returns the same quantities at the end of the run, with the event history
`(kind, time, check)`.

With the default settings, the filtered voltage stays below 0.5 pu for less
than `Tcease = 50` ms, so the load rides through. With `VLcease = 0.7`, the
load ceases 58 ms after the fault starts, reconnects once `Vtf` has stayed
above 0.8 pu for 50 ms, and ramps to `FracOn = 0.55` in `1.25 × 0.8 = 1` s:

```@example perc2
fig = Figure(size = (900, 330))
ax1 = Axis(fig[1, 1], xlabel = "time (s)", ylabel = "bus 5 voltage (pu)")
ax2 = Axis(fig[1, 2], xlabel = "time (s)", ylabel = "PERC2 active power (pu)")
for (lg, lab) in ((log_a, "default"), (log_b, "VLcease = 0.7"))
    lines!(ax1, [e.t for e in lg], [e.vm for e in lg], label = lab)
    lines!(ax2, [e.t for e in lg], [e.P for e in lg], label = lab)
end
axislegend(ax2, position = :rb)
fig
```

The PERC2 part is 100 MW (`P0 = 1.0` pu). In the default run, its power
briefly exceeds `P0` after clearing: the load draws constant power through
`Ip = P/Vtf`, and `Vtf` lags the recovering bus voltage, so `V·Ip`
overshoots until `Vtf` catches up. In the ceasing run,
`FracOn` ends at 0.55, but the power settles near 0.62 pu. The load
reduction raises the system frequency, and the droop `Kdroop` increases the
demand of the reconnected part.

## Loading from a `.dyr` file

`from_psse` reads bus-scope PERC2 records: a `USRLODC` (or `USRLOD`) record
with model name `PERCBL2U`. The load fraction comes from the matching
`MAINBL10` record with the same bus and load id. Without a `MAINBL10` record,
the PERC2 part replaces the whole load, with a warning. The bundled file
`examples/ieee9_perc2.dyr` contains the machines of the 9-bus case and a
default PERC2 component for 80 % of the load at bus 5:

```@example perc2
sys = from_psse(joinpath(ex, "ieee9_v33.raw"), joinpath(ex, "ieee9_perc2.dyr"))
build_network!(sys); runpf!(sys)
dp = DynamicProblem(sys); initialize_dynamics!(dp, sys)
add_event!(sys, ContingencyEvent(sys.busmap[5], 0.01, 1.0, 1.075))
integrate!(dp, sys, 3.0)
st = perc2_states(sys)[1]
(st.bus, st.id, st.FracOn, st.events)
```

The `LSTCBL` record describes the static remainder. GradPower keeps that
remainder as the original ZIP load and reports that the `LSTCBL` exponents are
not modelled. [`analyze_dyr_coverage`](@ref) lists `PERCBL2U` and `MAINBL10`
as native and `LSTCBL` as unsupported.

## Many scenarios

[`GradPower.integrate_batched!`](@ref) runs one PERC2 logic and one frequency
estimator per scenario. The `PERC2Params` are shared; the operating point
(`P0`, `Q0`, `V0` in the parameter row) and the initial state can differ. Build
the `BatchedLayout` after `add_perc2_load!` and `initialize_dynamics!`. Here
the PERC2 power is scaled by 1.0, 0.9 and 0.8 (the parameter-only scaling
starts each scenario away from equilibrium, as in
[Many scenarios at once](../tutorials/batched.md)):

```@example perc2
sys = from_psse(joinpath(ex, "ieee9_v33.raw"), joinpath(ex, "ieee9bus_gov.dyr"))
build_network!(sys); runpf!(sys)
dev = add_perc2_load!(sys, 5, "1"; fraction = 0.8)
dp = DynamicProblem(sys); initialize_dynamics!(dp, sys)
add_event!(sys, ContingencyEvent(sys.busmap[5], 0.01, 1.0, 1.075))

bl = GradPower.BatchedLayout(dp, sys, 3)
iP0 = parameter_index(sys, dev, :P0)
for m in 1:3
    bl.p[m, iP0] *= 1 - 0.1 * (m - 1)
end
t, trajs = GradPower.integrate_batched!(bl, sys, 3.0)
[(m, perc2_states(sys; scenario = m)[1].P) for m in 1:3]
```

## Verification

The tests check the model at three levels.

- **Specification.** Driven by 24 prescribed voltage and frequency profiles,
  GradPower's PERC2 block reproduces trajectories from an independent
  implementation of the published PERC2 specification. The cease and
  reconnect events are identical, and the filtered voltage, power and `FracOn`
  differ by at most 4.4e-16.
- **Closed form.** Step responses of the washouts and lead-lags, the `Tt`
  floor and the ramp duration `Tramp·Ldfact` are compared with their analytic
  values.
- **Network.** A flat run equals the system without PERC2. The Jacobian
  matches finite differences. Disconnecting a PERC2 load that replaces a whole
  load gives the trajectory of tripping the static load to within 1e-6. Batched runs
  equal single runs to 1e-10, and `.dyr` import equals `add_perc2_load!` with
  the same settings.

## Limitations

- The model represents the grid-facing behaviour of the facility. UPS battery
  energy and converter control dynamics are not modelled.
- The parameters describe a specific facility. The defaults and the example
  settings in Appendix H of the NERC reference are illustrations, not data for
  a particular site.
- Only bus-scope `.dyr` records (`PERCBL2U`) are imported. Owner, zone and
  area records raise `ArgumentError`.
- The static remainder of a split load stays a ZIP load. `LSTCBL` exponents
  are not modelled.
