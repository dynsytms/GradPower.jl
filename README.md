# GradPower.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://dynsytms.github.io/GradPower.jl/dev/)
[![Test](https://github.com/dynsytms/GradPower.jl/actions/workflows/test.yml/badge.svg)](https://github.com/dynsytms/GradPower.jl/actions/workflows/test.yml)

**Documentation:** <https://dynsytms.github.io/GradPower.jl/dev/>. It covers
tutorials, the event types, integrators, GPU use, dataset generation and the
API reference.

See [Queries and parameter edits](docs/src/guides/queries.md) for device lookup
and the lifecycle for changing parameters or operating points.

See [Plotting simulation results](docs/src/guides/plotting.md) for optional
CairoMakie figures, scenario envelopes, and animated network voltages.

A Julia package for **GPU-accelerated power-system dynamic simulation**.

`GradPower.jl` parses standard PSS/E input files (`.raw` + `.dyr`), solves the
power flow, and integrates the differential-algebraic equations of a
multi-machine power system through fault contingencies. Simulations can run on
CPU or NVIDIA GPU, and multiple contingency scenarios can be batched in a single
call.

## Features

- Read PSS/E `.raw` (network) and `.dyr` (dynamic) files; MATPOWER `.m` files
  for the static network.
- Sparse Newton power flow on the resulting system.
- Backward-Euler or Crank–Nicolson DAE integration through scheduled events:
  resistive/impedance bus faults, line trips (with circuit ids), generator
  trips (with their controllers), load trips and relative load steps.
- Selective recording (`save_every`, `save_idxs`, `Float32`), step callbacks
  and early termination on loss of synchronism.
- Struct-of-arrays state layout and per-device-type batched residual /
  Jacobian kernels via KernelAbstractions.jl (CPU and GPU backends).
- **GPU acceleration** via a CUDA extension: residual, Jacobian assembly,
  and sparse direct solve (cuDSS or SparseDirectSolver.jl) run entirely on
  the GPU with zero host-device transfers per Newton iteration.
- **Switchable sparse direct solver**: KLU (default on CPU), CUDSS.jl
  (GPU), or SparseDirectSolver.jl (CPU and GPU), selected with the
  `linear_solver` keyword.
- **Batched multi-scenario simulation**: integrate M independent contingency
  scenarios in one call, sharing the Jacobian sparsity pattern.
- **Schur-complement reduction**: optional block-elimination solver that
  factors per-generator dense blocks with batched cuBLAS and solves only the
  reduced network system with the GPU sparse direct solver.
- Supported device models: GENROU, GENSAL, IEESGO, TGOV1, SEXS, ESDC1A,
  IEEEST (PSS), ZIPLoad, StaticGenerator.

## Installation

`GradPower.jl` is not yet registered. Install from the repository:

```julia
julia> ]
pkg> add https://github.com/dynsytms/GradPower.jl
```

Or, for local development:

```julia
pkg> dev https://github.com/dynsytms/GradPower.jl
```

## Quick start

### CPU single-scenario

A complete fault-on-bus simulation on the bundled 2-bus case (GENROU
machine with an IEESGO governor):

```julia
using GradPower

# 1. Parse PSS/E network + dynamic data.
sys = from_psse("examples/2bus.raw", "examples/2bus_IEESGO.dyr")

# 2. Build the admittance matrix and solve the power flow. Generator reactive
#    limits are enforced (PV -> PQ switching); pass `enforce_q_limits=false`
#    to hold every PV voltage regardless of the reactive power it takes.
build_network!(sys)
runpf!(sys)

# 3. Allocate the dynamic problem and initialize states from the PF solution.
dp = DynamicProblem(sys)
initialize_dynamics!(dp, sys)

# 4. Schedule a 3-phase fault on bus 2: r_fault = 0.2 pu, t = 0.2 s to 0.3 s.
add_event!(sys, ContingencyEvent(2, 0.2, 0.2, 0.3))

# 5. Integrate to t_final = 10.0 s.
tvec, traj = integrate!(dp, sys, 10.0)
```

For named state access, opt into a result without changing the default tuple API:

```julia
res = integrate!(dp, sys, 10.0; result=true)
w = speed(res, 1)                          # raw pu speed deviation, view
v = bus_voltage(res, sys.buses[1].i)        # complex voltage at external bus
```

Data generation defaults to `output_mode = "all_states"`: every differential,
algebraic, and network-voltage state row. Compact derived channels require
`output_mode = "derived_channels"`; see the dataset guide.

`traj` is a `(system_size x nsteps)` matrix laid out as
`[diff_states; alg_states; v_re/v_im per bus]`. The rotor speed of
generator `k` is at `traj[L.genrou.diff_ptr[k] + 4, :]` (the `w`
state). Bus voltage magnitude on bus `b`:

```julia
voff = sys.dynamic.diff_dim + sys.dynamic.alg_dim
vre  = traj[voff + 2*(b-1) + 1, :]
vim  = traj[voff + 2*(b-1) + 2, :]
vm   = sqrt.(vre.^2 .+ vim.^2)
```

### CPU batched multi-scenario

Simulate `M` independent contingency scenarios in one call. Each scenario
shares the same system and Jacobian sparsity but can have different faults:

```julia
using GradPower

sys = from_psse("examples/ieee9_v33.raw", "examples/ieee9bus_gov.dyr")
build_network!(sys)
runpf!(sys)

dp = DynamicProblem(sys)
initialize_dynamics!(dp, sys)
add_event!(sys, ContingencyEvent(1, 0.02, 0.1, 0.2))

M = 8
bl = GradPower.BatchedLayout(dp, sys, M)
# Make scenarios differ by editing bl.z[m, :] (initial state) or
# bl.p[m, :] (parameters), e.g. scale a load's constant-power P per scenario.
tvec, trajs = GradPower.integrate_batched!(bl, sys, 1.0; dt=1.0/120.0)
# trajs[m] is the trajectory matrix for scenario m
```

Scenarios run in parallel on `Threads.nthreads()` threads (start Julia with
`-t auto`); each one gets exactly the trajectory `integrate!` gives for the
same initial state and parameters.

### Choosing the sparse direct solver

Every Newton solve goes through a sparse direct solver chosen with the
`linear_solver` keyword:

| Backend | Where | Requires |
| --- | --- | --- |
| `KLUBackend()` | CPU (default) | nothing (KLU.jl is a dependency) |
| `CUDSSBackend()` | GPU | `using CUDA, CUDSS` |
| `SparseDirectSolverBackend()` | CPU and GPU | `using SparseDirectSolver` (plus `using CUDA` on the GPU) |

SparseDirectSolver.jl is an optional, unregistered dependency. Install it with
`Pkg.add(url="https://github.com/exanauts/SparseDirectSolver.jl")`.

```julia
using GradPower, SparseDirectSolver

tvec, traj = integrate!(dp, sys, 10.0; linear_solver=SparseDirectSolverBackend())
tvec, trajs = GradPower.integrate_batched!(bl, sys, 1.0;
                                           linear_solver=SparseDirectSolverBackend())
```

`integrate!` accepts `linear_solver` with every `solver=` choice
(`:monolithic`, `:schur`, `:schur_gmres`).

### Newton options

By default `integrate!` keeps the Jacobian factorization across Newton
iterations and time steps and refactors only when convergence slows, at
events, or after a failed solve (`jacobian_reuse=true`), and starts each step
from a linear extrapolation of the previous two (`predictor=true`). Both
converge to the same `newton_tol`; on ACTIVSg2000 they cut factorizations from
~1200 to ~10 over 5 s and run about 4x faster. Pass
`jacobian_reuse=false, predictor=false` for full Newton from the previous
state. They apply to `:monolithic` and `:schur`.

### GPU acceleration (NVIDIA)

With CUDA.jl and a GPU sparse direct solver installed (CUDSS.jl or
SparseDirectSolver.jl), the CUDA extension loads automatically. All
residual/Jacobian assembly and the sparse direct solve run on the GPU:

```julia
using GradPower, CUDA, CUDSS

sys = from_psse("examples/ieee9_v33.raw", "examples/ieee9bus_gov.dyr")
build_network!(sys)
runpf!(sys)

dp = DynamicProblem(sys)
initialize_dynamics!(dp, sys)
add_event!(sys, ContingencyEvent(1, 0.02, 0.1, 0.2))

# Build GPU layout for M scenarios
ext = Base.get_extension(GradPower, :GradPowerCUDAExt)
M   = 16
gbl = ext.GpuBatchedLayout(dp, sys, M)

# Integrate on GPU — monolithic cuDSS solver
tvec, trajs = ext.integrate_gpu_cudss!(gbl, sys, 1.0; dt=1.0/120.0)

# Or use the Schur-complement solver (faster for large systems)
tvec, trajs = ext.integrate_gpu_schur_cudss!(gbl, sys, 1.0; dt=1.0/120.0)
```

`GpuBatchedLayout` picks the GPU solver with
`linear_solver=CUDSSBackend()` or `linear_solver=SparseDirectSolverBackend()`.
Without the keyword it uses CUDSS when CUDSS.jl is loaded, else
SparseDirectSolver.jl. The `integrate_gpu_*` functions keep their names
whichever solver the layout holds:

```julia
using GradPower, CUDA, SparseDirectSolver

gbl = ext.GpuBatchedLayout(dp, sys, M; linear_solver=SparseDirectSolverBackend())
tvec, trajs = ext.integrate_gpu_cudss!(gbl, sys, 1.0; dt=1.0/120.0)
```

Measured on a Quadro GV100 vs a 4× Xeon Gold 6140 (one core per scenario),
ACTIVSg70k (216k unknowns), 1 s with a bus fault, both with Jacobian reuse:
one scenario takes 0.9 s on the GPU (`integrate_gpu_schur_cudss!`) vs 9.3 s
with CPU `integrate!` (KLU); 16 scenarios take 18 s vs 54 s with
`integrate_batched!` on 16 threads. For smaller systems run in batches the
threaded CPU path is usually faster (ACTIVSg2000, 64 scenarios: 3.8 s GPU vs
2.6 s CPU on 64 threads). GPU batch memory grows with M (ACTIVSg70k: about
0.33 GiB per scenario).

### Optional NLP model

Install and load `NLPModels` and `MadNLP` to enable
`GradPower.DynamicNLP(sys, dp, tfinal, lvar, uvar)`. Neither package is required
for simulation. The constructor is now a function returning a model whose
concrete type lives in `GradPowerNLPExt`, rather than a type in `GradPower`.
Forward objective evaluation works; gradients and gradient-based optimization
remain unavailable. See `examples/nlp.jl` for a runnable objective evaluation.

The unused internal `newton`, `compare_matrix`, `find_outliers` helpers and the
`newton_step!` debug keyword `jac_verify` have been removed. FiniteDiff and NPZ
remain test dependencies, not core dependencies. Verification and timing
scripts use `scripts/Project.toml`, which declares their NPZ, FiniteDiff and
Statistics dependencies. Set it up once with
`julia --project=scripts -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'`.
ForwardDiff is still pulled in transitively by NLsolve for device initialization.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/`        | Package source (CPU kernels, dynamics, parsing, Schur complement) |
| `ext/`        | CUDA extension (GPU kernels, batched GPU layout) and sparse direct solver extensions (CUDSS.jl, SparseDirectSolver.jl) |
| `examples/`   | PSS/E `.raw` / `.dyr` files (2-bus through 70k-bus) |
| `test/`       | Unit tests (`julia --project -e 'using Pkg; Pkg.test()'`) |
| `docs/`       | Design proposal and notes |

## License

MIT. See [LICENSE](LICENSE).
