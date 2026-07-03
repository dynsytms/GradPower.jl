# GradPower.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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
- Backward-Euler DAE integration through scheduled bus-fault contingencies.
- Struct-of-arrays state layout and per-device-type batched residual /
  Jacobian kernels via KernelAbstractions.jl (CPU and GPU backends).
- **GPU acceleration** via a CUDA extension: residual, Jacobian assembly,
  and sparse direct solve (cuDSS) run entirely on the GPU with zero
  host-device transfers per Newton iteration.
- **Batched multi-scenario simulation**: integrate M independent contingency
  scenarios in one call, sharing the Jacobian sparsity pattern.
- **Schur-complement reduction**: optional block-elimination solver that
  factors per-generator dense blocks with batched cuBLAS and solves only the
  reduced network system with cuDSS.
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

# 2. Build the admittance matrix and solve the power flow.
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
tvec, trajs = GradPower.integrate_batched!(bl, sys, 1.0; dt=1.0/120.0)
# trajs[m] is the trajectory matrix for scenario m
```

### GPU acceleration (NVIDIA)

With CUDA.jl and CUDSS.jl installed, the CUDA extension loads automatically.
All residual/Jacobian assembly and the sparse direct solve run on the GPU:

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

On a Quadro GV100, the GPU path achieves ~11x speedup over CPU KLU for
a 70,000-bus system (216k unknowns).

## Repository layout

| Path | Contents |
| --- | --- |
| `src/`        | Package source (CPU kernels, dynamics, parsing, Schur complement) |
| `ext/`        | CUDA extension (GPU kernels, cuDSS solver, batched GPU layout) |
| `examples/`   | PSS/E `.raw` / `.dyr` files (2-bus through 70k-bus) |
| `test/`       | Unit tests (`julia --project -e 'using Pkg; Pkg.test()'`) |
| `docs/`       | Design proposal and notes |

## License

MIT. See [LICENSE](LICENSE).
