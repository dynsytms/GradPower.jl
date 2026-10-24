# Supported models and input files

## Input files

| format | read with | contains |
|:--|:--|:--|
| PSS/E `.raw` (v33) | `from_psse(raw, dyr)` | buses, loads, fixed shunts, generators, lines, 2- and 3-winding transformers |
| PSS/E `.dyr` | `from_psse(raw, dyr)` | machine and controller models |
| MATPOWER `.m` | `GradPower.read_matpower_case(file)` | static network (power flow only) |

`from_psse(raw, nothing)` reads a network without dynamics, for power flow alone.

## Device models

| PSS/E model | kind | differential / algebraic states |
|:--|:--|:--|
| `GENROU` | round-rotor synchronous machine (with saturation) | 6 / 4 |
| `GENSAL` | salient-pole machine (mapped onto GENROU with `x_q' = x_d'`, `T_q0' = T_d0'`) | 6 / 4 |
| `IEESGO` | steam turbine-governor | 5 / 1 |
| `TGOV1` | steam turbine-governor (with optional valve limits) | 2 / 1 |
| `SEXS` | simplified excitation system | 2 / 0 |
| `ESDC1A` | DC exciter | 3 / 0 |
| `IEEEST` | power system stabilizer | 7 / 1 |
| loads | `ZIPLoad`: impedance (default) / power mixture; no separate current component | 0 / 0 |
| generators without `.dyr` record | `StaticGenerator`: PQ injection, PV regulation, or fixed reference voltage, according to bus type | 0 / 0–2 |

Controllers are attached to their machine by **bus and id**, as in PSS/E.
`StaticGenerator` aggregates unmatched online generators at a bus. A PV stand-in
stops regulating if power flow converts its bus to PQ at a reactive limit.

## Checking what a `.dyr` file uses

Real `.dyr` files often contain models GradPower does not implement.
[`analyze_dyr_coverage`](@ref) reports, record by record, which ones are
native, unsupported or without a matching generator:

```@example models
using GradPower
ex  = joinpath(pkgdir(GradPower), "examples")
rep = analyze_dyr_coverage(joinpath(ex, "IEEE39.raw"), joinpath(ex, "IEEE39_gov.dyr"))
coverage_counts(rep)
```

By default unsupported records are skipped with a warning, and their machine
keeps running without that controller. `from_psse(raw, dyr; surrogates = true)`
instead maps common unsupported governors and exciters (GGOV1, HYGOV, IEEEG1,
EXAC*, ESST*, ...) onto TGOV1/SEXS stand-ins. A surrogate lets a large case run
end to end. It is not the original model, and results obtained with
surrogates must not be presented as a validation of it.

## Bundled cases

The `examples/` folder contains small cases used throughout this
documentation and the tests:

| case | buses | machines | `.dyr` variants |
|:--|--:|--:|:--|
| `2bus.raw` | 2 | 1 | `2bus.dyr`, `_IEESGO`, `_TGOV1`, `_SEXS`, `_ESDC1A`, `_IEEEST` |
| `ieee9_v33.raw` (WSCC 9-bus) | 9 | 3 | `ieee9bus.dyr`, `_gov`, `_SEXS`, `_TGOV1` |
| `IEEE39.raw` (New England) | 39 | 10 | `IEEE39.dyr`, `IEEE39_gov.dyr` |

Larger synthetic grids (including ACTIVSg200 to ACTIVSg70k from the Texas A&M
electric-grid test-case repository) have been used in studies, but are not
bundled. Check dynamic-model coverage and initialization for each case; this
is not a claim that every original model or control is supported.
