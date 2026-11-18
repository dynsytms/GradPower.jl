# PSS/E validation

GradPower's dynamic models are checked against trajectories exported from
PSS/E 35. The same comparison runs in the test suite
(`test/psse_reference/`), so the results on this page are recomputed with
the current code every time the documentation is built.

## Setup

| | |
|:--|:--|
| reference | PSS/E 35.0.0 RC5, trajectories exported at 1/1920 s |
| network | IEEE 9-bus (WSCC), `test/psse_reference/data/ieee9_v33.raw` |
| machines | three `GENROU`, no exciter or governor (`ieee9_ctrl.dyr`) |
| power flow | reactive limits not enforced, as in the PSS/E run |
| GradPower | `mech_input = :power`, Crank–Nicolson, step 1/1920 s, 10 s |

`mech_input = :power` makes the swing equation divide the mechanical input
by speed, as PSS/E does (see [Supported models](../guides/models.md)). With
the default `:torque` form, these cases differ from PSS/E by 18–30 % in
speed, because nothing damps the frequency drift when there is no governor.

| case | event |
|:--|:--|
| `i9_ctrl_trip57` | line 5–7 opened at 0.2 s |
| `i9_ctrl_trip89` | line 8–9 opened at 0.2 s |
| `i9_ctrl_flt7` | fault at bus 7 (r = 0.01 pu), applied at 0.2 s, cleared at 0.3 s |

## Metrics

Each metric is the largest difference over the 10 s run:

- **angle**: change in each machine's angle relative to machine 1, as % of
  PSS/E's largest swing;
- **speed**: change in speed deviation, as % of PSS/E's largest change;
- **voltage**: bus voltage magnitude after the event, in pu.

Speed and voltage leave out 0.05 s around each switching instant, where the
two programs resolve the discontinuity differently.

```@setup psse
using GradPower, CairoMakie, Printf, Markdown
const REF = joinpath(pkgdir(GradPower), "test", "psse_reference")
include(joinpath(REF, "harness.jl"))
using .PSSEReference

cases = ["i9_ctrl_trip57", "i9_ctrl_trip89", "i9_ctrl_flt7"]
runs = Dict(map(cases) do tag
    case, ref = load_case(tag)
    gp = run_gradpower(case)
    tag => (; case, ref, gp, m = compare_runs(gp, ref, case["event"]))
end)

function compare_plot(r)
    fig = Figure(size = (900, 620))
    ref, gp = r.ref, r.gp
    rel(d) = rad2deg.(d[:, 2:end] .- d[:, 1])
    ax1 = Axis(fig[1, 1]; ylabel = "angle vs machine 1 (deg)")
    ax2 = Axis(fig[2, 1]; ylabel = "speed deviation (pu)")
    ax3 = Axis(fig[3, 1]; ylabel = "|V| (pu)", xlabel = "time (s)")
    colors = Makie.wong_colors()
    for (j, (bus, id)) in enumerate(ref.gens[2:end])
        lines!(ax1, ref.t, rel(ref.delta)[:, j]; color = colors[j + 1], label = "machine $bus, PSS/E")
        lines!(ax1, gp.t, rel(gp.delta)[:, j]; color = :black, linestyle = :dash, linewidth = 1)
    end
    for (j, (bus, id)) in enumerate(ref.gens)
        lines!(ax2, ref.t, ref.omega[:, j]; color = colors[j], label = "machine $bus, PSS/E")
        lines!(ax2, gp.t, gp.omega[:, j]; color = :black, linestyle = :dash, linewidth = 1)
    end
    for bus in (5, 7, 8)
        jr = findfirst(==(bus), ref.vm_bus); jg = findfirst(==(bus), gp.vm_bus)
        lines!(ax3, ref.t, ref.vm[:, jr]; color = colors[bus - 3], label = "bus $bus, PSS/E")
        lines!(ax3, gp.t, gp.vm[:, jg]; color = :black, linestyle = :dash, linewidth = 1)
    end
    for (i, ax) in enumerate((ax1, ax2, ax3))
        lines!(ax, [NaN], [NaN]; color = :black, linestyle = :dash, linewidth = 1, label = "GradPower")
        Legend(fig[i, 2], ax; framevisible = false, labelsize = 11, rowgap = 0)
    end
    linkxaxes!(ax1, ax2, ax3)
    return fig
end
```

## Results

```@example psse
rows = [@sprintf("| `%s` | %.2f %% | %.2f %% | %.1e |", tag, runs[tag].m["angle_pct"],
                 runs[tag].m["speed_pct"], runs[tag].m["v_err"]) for tag in cases]
Markdown.parse(join(["| case | angle | speed | voltage (pu) |", "|:--|--:|--:|--:|", rows...], "\n"))
```

The test suite fails if any case exceeds angle 0.5 %, speed 0.25 % or
voltage 5e-4 pu. In the plots, colored lines are PSS/E and dashed black
lines are GradPower.

### Line 5–7 trip

```@example psse
compare_plot(runs["i9_ctrl_trip57"])
```

### Line 8–9 trip

```@example psse
compare_plot(runs["i9_ctrl_trip89"])
```

### Fault at bus 7

```@example psse
compare_plot(runs["i9_ctrl_flt7"])
```
