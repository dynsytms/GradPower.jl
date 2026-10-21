# Linear solvers and GPU

Almost all of the time in a simulation goes into Newton's linear systems. On the
CPU they are solved with a sparse direct method. On an NVIDIA GPU the whole
Newton iteration can run on the device: residual and Jacobian kernels, sparse
factorization and solve.

## Choosing the sparse direct solver

Every Newton solve goes through the solver passed as `linear_solver`:

| backend | where | requires |
|:--|:--|:--|
| `KLUBackend()` | CPU (default) | nothing |
| `SparseDirectSolverBackend()` | CPU and GPU | `using SparseDirectSolver` |
| `CUDSSBackend()` | GPU | `using CUDA, CUDSS` |

```julia
using GradPower, SparseDirectSolver
t, traj = integrate!(dp, sys, 10.0; linear_solver = SparseDirectSolverBackend())
```

`linear_solver` works with every `solver=` choice and with
`integrate_batched!`. SparseDirectSolver.jl is not registered. Install it with
`Pkg.add(url = "https://github.com/exanauts/SparseDirectSolver.jl")`.

## Running on the GPU

Loading `CUDA` and a GPU solver activates GradPower's CUDA extension. A batch of
`M` scenarios is copied to the device once, integrated there, and the
trajectories come back at the end:

```julia
using GradPower, CUDA, CUDSS
ext = Base.get_extension(GradPower, :GradPowerCUDAExt)

sys = from_psse("examples/ieee9_v33.raw", "examples/ieee9bus_gov.dyr")
build_network!(sys); runpf!(sys)
dp = DynamicProblem(sys); initialize_dynamics!(dp, sys)
add_event!(sys, ContingencyEvent(sys.busmap[7], 0.02, 0.1, 0.2))

gbl = ext.GpuBatchedLayout(dp, sys, 16)                # 16 scenarios on the device
t, trajs = ext.integrate_gpu_cudss!(gbl, sys, 1.0)     # monolithic Newton
t, trajs = ext.integrate_gpu_schur_cudss!(gbl, sys, 1.0) # Schur-reduced Newton
```

Without `linear_solver`, `GpuBatchedLayout` chooses the solver from the batch
size:
- below 128 scenarios: cuDSS;
- from 128 scenarios: SparseDirectSolver, if it is loaded.

SparseDirectSolver's uniform batch keeps getting cheaper per system up to about
1,000 systems. cuDSS has to be split into chunks of 15, and its cost per system
is flat beyond that. On ACTIVSg2000 the full GPU Schur integration ties at 64
scenarios, and SparseDirectSolver is 1.3× faster at 256.

Like `BatchedLayout`, `GpuBatchedLayout` holds per-scenario states and
parameters (`gbl.z`, `gbl.p`). Scenarios with **different fault buses or
impedances** share one factorization through low-rank updates:

```julia
t, trajs = ext.integrate_gpu_shared_multi!(gbl, sys, 1.0,
                fault_buses, rfaults, 0.1, 0.2)   # one (bus, r) per scenario
```

### Storing less

The GPU integrators take the same output options as `integrate!`:
`save_every`, `save_idxs` and `save_eltype`. Only the selected rows of the
selected steps are kept on the device, and they are downloaded once at the
end:

```julia
net  = sys.dynamic.diff_dim + sys.dynamic.alg_dim
rows = collect(net + 1 : net + 2length(sys.buses))        # bus voltages only
t, V = ext.integrate_gpu_schur_cudss!(gbl, sys, 1.0;
           save_idxs = rows, save_every = 2, save_eltype = Float32)
```

On ACTIVSg2000 (1 s, 256 scenarios) this takes 3.4 s instead of 6.6 s for the
full trajectory. Device memory drops from 5.3 to 3.7 GiB, and the output from
1.9 GB to 0.29 GB.

The GPU integrators support resistive bus faults (`ContingencyEvent` without
`xfault`). If other events are registered (trips, load steps, impedance faults),
they raise an `ArgumentError` rather than silently ignoring them; use the CPU
integrators for those.

## When is the GPU faster?

Measured on one Quadro GV100 and a 4× Xeon Gold 6140 server, 1 s with a bus fault,
Jacobian reuse on:

| case (unknowns) | scenarios | GPU | CPU (threads) |
|:--|--:|--:|--:|
| ACTIVSg2000 (7.5 k) | 1 | 0.15 s | 0.14 s |
| ACTIVSg2000 | 16 | 0.84 s | 1.5 s |
| ACTIVSg2000 | 64 | 3.5 s | 2.6 s (64) |
| ACTIVSg70k (217 k) | 1 | 0.94 s | 9.3 s |
| ACTIVSg70k | 16 | 18 s | 54 s (16) |
| ACTIVSg70k | 64 | 93 s | 162 s (64) |

As a rule of thumb:
- **Large networks** benefit most: one 70 k-bus simulation is about 10× faster.
- **Small and medium networks** in big batches usually run as fast or faster on
  a many-core CPU with `integrate_batched!`.
- **GPU memory** grows with the batch, about 0.33 GiB per ACTIVSg70k scenario.

## Schur reduction

With `solver = :schur` (CPU) or `integrate_gpu_schur_cudss!` (GPU), each Newton
system is solved by first eliminating every device cluster. A cluster is a
machine with its controllers, and its block is small and dense. What remains
is the network-sized system, which has the same sparsity as the admittance
matrix. The dense blocks are independent, so they batch well on the GPU. See
[How it works](../theory.md) for the algebra.
