# PERC2 reference trajectories

Reference data for `test/test_perc2_core.jl`: 24 trajectories from an
independent implementation of the published PERC2 specification, each driven
by a prescribed piecewise-constant bus voltage and frequency (no network).

## Files

- `cases.json`: the time step `dt` and, for every case, its description, the
  PERC2 parameters (`cons`: the six ICON flags and the CONs by name), the
  voltage/frequency breakpoints `[t, V, f]`, the end time `t_end`, the expected
  events (`trip_time`, `reconnect_start`, `reconnect_end`; `null` when the
  event does not occur) and the event history.
- `<case>.csv`: one trajectory per case, columns
  - `t`: time (s)
  - `V`: bus voltage magnitude input (pu)
  - `P`: active power drawn, per unit of the initial power `P0`
  - `FracOn`: fraction of the load connected
  - `f`: bus frequency input (Hz)
  - `Vtf`: filtered voltage (pu)

## Conventions

- `dt = 1/240` s.
- Row `k` is the output of the model call at `t_k = k·dt`, made with the
  input `V(t_k)`, `f(t_k)`. The input at `t` is the value of the last
  breakpoint with time `<= t`.
- Each case starts at `P0 = 1`, `Q0 = 0` and `V0 = V(0)`.
