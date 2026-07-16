# GradPower.jl — GPU Batched Parallelization Report

## Objective

Accelerate power-system dynamic simulation for surrogate model training on Argonne's Polaris HPC. The target workload is $10^5$–$10^6$ contingency scenarios across varying operating conditions on systems up to 70,000 buses. This requires batched simulation (M independent scenarios in one call) with GPU acceleration.

## Mathematical Formulation

### DAE System

A power system with $n_b$ buses, $n_g$ generators, and associated controllers is modeled as a semi-explicit differential-algebraic equation (DAE):

$$\dot{x} = f(x, y, v, u, p)$$
$$0 = g(x, y, v, u, p)$$
$$0 = h(x, v) - Y_{\text{bus}} v$$

where:

- $x \in \mathbb{R}^{n_d}$ — differential states (rotor fluxes, speeds, angles, governor/exciter states)
- $y \in \mathbb{R}^{n_a}$ — algebraic states (stator currents and voltages in dq-frame)
- $v \in \mathbb{R}^{2 n_b}$ — network voltages (real and imaginary per bus)
- $u \in \mathbb{R}^{n_u}$ — control signals routed between devices (e.g., $\omega$ from generator to governor, $E_{fd}$ from exciter to generator, $V_s$ from PSS to exciter)
- $p \in \mathbb{R}^{n_p}$ — device parameters (machine reactances, time constants, governor gains, etc.)
- $Y_{\text{bus}} \in \mathbb{C}^{n_b \times n_b}$ — bus admittance matrix (sparse, topology-dependent)

The composite state vector is $z = [x; y; v] \in \mathbb{R}^n$ where $n = n_d + n_a + 2n_b$.

### Device Models

Each device type contributes rows to $f$, $g$, and $h$. The GENROU synchronous generator (the most complex) has 6 differential and 4 algebraic states:

**Differential states** ($x_k$ for generator $k$):

$$\dot{e}'_q = \frac{1}{T'_{d0}} \left[ E_{fd} - e'_q - (x_d - x'_d)\left(i_d - \frac{(x'_d - x''_d)}{(x'_d - x_l)^2}(-e'_q + i_d(x'_d - x_l) + \phi_{1d})\right) - S_e \psi_{de} \right]$$

$$\dot{e}'_d = \frac{1}{T'_{q0}} \left[ -e'_d + (x_q - x'_q)\left(i_q - \frac{(x'_q - x''_q)}{(x'_q - x_l)^2}(e'_d + i_q(x'_q - x_l) + \phi_{2q})\right) \right]$$

$$\dot{\phi}_{1d} = \frac{1}{T''_{d0}} (e'_q - i_d(x'_d - x_l) - \phi_{1d})$$

$$\dot{\phi}_{2q} = \frac{1}{T''_{q0}} (-e'_d - i_q(x'_q - x_l) - \phi_{2q})$$

$$\dot{\omega} = \frac{1}{2H} (P_m - D\omega - \psi_{de} i_q + \psi_{qe} i_d)$$

$$\dot{\delta} = 2\pi f_0 \omega$$

where $\psi_{de}$, $\psi_{qe}$ are subtransient flux linkages and $S_e(\psi)$ is the quadratic saturation function.

**Algebraic states** ($y_k$): stator current balance and Park transformation:

$$0 = i_d - \frac{\psi_{de} - v_q}{x''_d}, \quad 0 = i_q - \frac{\psi_{qe} + v_d}{x''_q}$$

$$0 = v_d - (v_r \sin\delta - v_i \cos\delta), \quad 0 = v_q - (v_r \cos\delta + v_i \sin\delta)$$

**Network injection**: each generator injects current into its bus:

$$h_k(x_k, v) = \begin{bmatrix} i_d \sin\delta + i_q \cos\delta \\ -i_d \cos\delta + i_q \sin\delta \end{bmatrix}$$

The network equation becomes $\sum_k h_k - Y_{\text{bus}} v = 0$ (KCL at each bus).

Controllers (IEESGO, TGOV1, SEXS, ESDC1A, IEEEST) add their own differential and algebraic states following the same pattern. Control signals are routed between devices via an index table (`uvec_idx`): $u[j] = z[\text{uvec\_idx}[j]]$.

### Device Dimensions

| Device | Diff states | Alg states | Parameters | Jacobian entries |
|--------|------------|------------|------------|-----------------|
| GENROU | 6 ($e'_q, e'_d, \phi_{1d}, \phi_{2q}, \omega, \delta$) | 4 ($v_q, v_d, i_q, i_d$) | 14 | 46 |
| GENSAL | 5 | 4 | 12 | ~40 |
| IEESGO | 5 ($P_{F0}, P_{LL}, T_{P1}, T_{P2}, T_{P3}$) | 1 ($P_m$) | 12 | 18 |
| TGOV1 | 2 | 1 | 7 | 9 |
| SEXS | 2 | 0 | 5 | 8 |
| ESDC1A | 3 | 0 | 8 | 11 |
| IEEEST | 7 | 1 | 14 | ~30 |
| ZIPLoad | 0 | 0 | 3 | 4 |
| StaticGen | 0 | 0 | 2 | 4 |

### Backward Euler Discretization

At each timestep $t_{k+1} = t_k + \Delta t$, the backward Euler method converts the DAE to a nonlinear system. Define $F(z_{k+1})$:

$$F_i(z) = \begin{cases} z_i - z_i^{\text{old}} - \Delta t \cdot f_i(z, u, p) & i \in \text{differential indices} \\ g_i(z, u, p) & i \in \text{algebraic indices} \\ h_i(z) - (Y_{\text{bus}} v)_i & i \in \text{network indices} \end{cases}$$

This is solved by Newton's method: given $z^{(0)} = z_k$, iterate:

$$J(z^{(\nu)}) \, \Delta z = -F(z^{(\nu)}), \qquad z^{(\nu+1)} = z^{(\nu)} + \Delta z$$

until $\|F(z^{(\nu)})\| < \varepsilon$. The Jacobian $J = \partial F / \partial z$ is sparse with structure determined by the device models and network topology.

### Jacobian Structure

The Jacobian has a natural $3 \times 3$ block structure reflecting the state partition $z = [x; y; v]$:

$$J = \begin{bmatrix} I - \Delta t \frac{\partial f}{\partial x} & -\Delta t \frac{\partial f}{\partial y} & -\Delta t \frac{\partial f}{\partial v} \\ \frac{\partial g}{\partial x} & \frac{\partial g}{\partial y} & \frac{\partial g}{\partial v} \\ \frac{\partial h}{\partial x} & 0 & \frac{\partial h}{\partial v} - Y_{\text{bus}} \end{bmatrix}$$

Each device contributes a dense diagonal block (its own states) and sparse off-diagonal entries (coupling to bus voltages). The network block $\frac{\partial h}{\partial v} - Y_{\text{bus}}$ has the sparsity of $Y_{\text{bus}}$ plus per-generator entries.

**Sparsity pattern**: for a system with $n_g$ generators on $n_b$ buses, the Jacobian is mostly block-diagonal with coupling only through the network. This structure is invariant across scenarios — only the numeric values change.

### Schur-Complement Reduction

The Jacobian can be partitioned into per-device "working" variables $w$ (differential + algebraic states) and "reduced" variables $r$ (network voltages):

$$\begin{bmatrix} A & B \\ C & D \end{bmatrix} \begin{bmatrix} \Delta w \\ \Delta r \end{bmatrix} = \begin{bmatrix} f_w \\ f_r \end{bmatrix}$$

where:
- $A = \text{blkdiag}(A_1, \ldots, A_{n_g})$ — block-diagonal, each $A_k$ is a dense $(n_{d,k} + n_{a,k}) \times (n_{d,k} + n_{a,k})$ matrix (e.g., $10 \times 10$ for GENROU)
- $B$ — sparse coupling from device states to network ($\partial g / \partial v$, Park transform)
- $C$ — sparse coupling from network to device states ($\partial h / \partial x$, current injections)
- $D = \frac{\partial h}{\partial v} - Y_{\text{bus}}$ — sparse network block

The Schur complement is $S = D - C A^{-1} B$. The reduced system $S \, \Delta r = f_r - C A^{-1} f_w$ has dimension $2 n_b$ (much smaller than $n$). Back-substitution gives $\Delta w = A^{-1}(f_w - B \, \Delta r)$.

**Computational advantage**:
- $A^{-1}$ is block-diagonal → each $A_k^{-1}$ is a tiny dense solve ($\leq 10 \times 10$), trivially parallelizable via cuBLAS batched LU.
- $S$ has dimension $2 n_b$ vs full system dimension $n = n_d + n_a + 2 n_b$. For ACTIVSg2000: $|S| = 4{,}000$ vs $n = 9{,}694$.
- If topology is shared across scenarios, $Y_{\text{bus}}$ and the sparsity of $S$ are identical — only numeric values of $A_k$, $B_k$, $C_k$ change.

### Batched Multi-Scenario Simulation

For $M$ independent contingency scenarios sharing the same network topology but potentially different fault locations and device parameter perturbations:

$$F^{(m)}(z^{(m)}) = 0, \quad m = 1, \ldots, M$$

Each scenario has its own state vector $z^{(m)}$ but shares:
- The Jacobian sparsity pattern (same topology → same nonzero positions)
- The $Y_{\text{bus}}$ structure (same admittances, perturbed only at fault buses)
- All device model code

The batched state layout stores $z$ as an $(M \times n)$ matrix. GPU kernels launch with `ndrange = (n_{\text{devices}}, M)`, computing one device × one scenario per thread. The sparse factorization is the only component that cannot trivially parallelize across $M$ — this is the bottleneck identified in our profiling.

### Contingency Events and Bus Faults

A three-phase bus fault at bus $b$ with fault impedance $r_f$ is modeled by adding $1/r_f$ to the diagonal of $Y_{\text{bus}}$ at bus $b$ during the fault window $[t_{\text{on}}, t_{\text{off}}]$. This modifies the network equation:

$$Y_{\text{bus}}^{\text{faulted}} = Y_{\text{bus}} + \frac{1}{r_f} e_b e_b^T$$

where $e_b$ is the unit vector for bus $b$. This is a rank-1 update — exploitable via the Sherman-Morrison formula for the Schur complement factorization:

$$(S + u v^T)^{-1} = S^{-1} - \frac{S^{-1} u v^T S^{-1}}{1 + v^T S^{-1} u}$$

This would allow factoring $S$ once and updating per-scenario fault perturbations at the cost of a single matrix-vector product per scenario, rather than a full refactorization.

### KLU Symbolic Reuse (CPU Path)

On CPU, the sparse LU factorization uses KLU (SuiteSparse). KLU separates symbolic analysis (fill-reducing ordering, symbolic factorization) from numeric factorization. Since the sparsity pattern is invariant across Newton iterations and timesteps:

- `klu(J)` — full symbolic + numeric factorization (first call only)
- `klu!(fact, J)` — numeric refactorization reusing symbolic analysis (subsequent calls)

If `klu!` encounters a zero pivot (`SingularException`), we fall back to a fresh `klu(J)`. This is rare but can happen after large state jumps (e.g., fault clearing).

### cuDSS GPU Factorization

On GPU, NVIDIA's cuDSS library provides the sparse direct solve. The same symbolic/numeric separation applies:

- `cudss("analysis", ...)` — symbolic analysis (once at construction)
- `cudss("factorization", ...)` — first numeric factorization
- `cudss("refactorization", ...)` — subsequent numeric updates (reuses symbolic)
- `cudss("solve", ...)` — triangular solve

cuDSS supports **uniform batching**: `cudss_set(solver, "ubatch_size", M)` configures a single solver to factorize and solve $M$ independent systems sharing the same sparsity pattern. The numeric values are provided as a `CuMatrix(nnz, M)` where each column holds one system's nonzero values.

## Starting Point

GradPower.jl had a working GPU backend for single-scenario integration via a CUDA extension (`GradPowerCUDAExt`). The batched infrastructure existed but all GPU kernels and the sparse solve looped sequentially over scenarios (`for m in 1:M`). The codebase supported 9 device models (GENROU, GENSAL, IEESGO, TGOV1, SEXS, ESDC1A, IEEEST, ZIPLoad, StaticGen) using KernelAbstractions.jl for portable CPU/GPU kernels.

## Work Completed

### Phase 1: PR Preparation and Test Coverage

Before parallelization, the codebase was audited and hardened for production:

- **Dead code removal**: deleted 3 unused GPU kernels, 6 dead struct fields, and the abandoned `_newton_step_cudss_batched_gpu!` function.
- **Allocation elimination**: pre-allocated `v_buf`/`fv_buf` in `GpuBatchedLayout` to eliminate per-call allocations in the residual hot loop.
- **KLU symbolic reuse** (CPU): changed `newton_step!` to reuse KLU symbolic analysis across Newton iterations via `klu!()` with `SingularException` fallback. Measured 2–5x CPU speedup.
- **Exports and API**: added missing public exports (`runpf`, `DynamicProblem`, `integrate!`, etc.).
- **TODO cleanup**: resolved all TODOs in `dynamics.jl`, `pflow.jl` to descriptive comments.
- **Version bump**: 0.1.1 → 0.2.0.
- **Test coverage**: wrote 4 new test files (455 new assertions, 592 → 1047 total):
  - `test_fd_jacobian.jl` — finite-difference Jacobian validation for 7 device kernels + 2 full-system cases
  - `test_integrate.jl` — integration, flat-line, trajectory reproducibility, event handling
  - `test_device_kernels.jl` — TGOV1, SEXS, ESDC1A, IEEEST, StaticGen kernel acceptance tests
  - `test_coupling.jl` — control signal routing traits and wiring for all controller types
- **README rewrite** with CPU single-scenario, CPU batched, and GPU quick-start examples.
- **.gitignore** created.

### Phase 2: GPU Kernel Parallelization (Workflow 1)

Eliminated sequential `for m in 1:M` loops in all GPU kernels via an automated 6-step workflow. Each step was executed by an independent agent, verified against CPU baseline trajectories (M=1,4,16), and committed.

| Step | Description | Kernels Modified | Commit |
|------|-------------|-----------------|--------|
| 1 | 2D residual kernels | 8 (genrou, ieesgo, tgov1, sexs, esdc1a, ieeest, zipload, static_gen) | `b893df1` |
| 2 | 2D Jacobian kernels | 8 (same device types) | `7ab353e` |
| 3 | 2D utility kernels | 9 (uvec routing, events, injection reduce, beuler scaling, snapshot) | `ac821b3` |
| 4 | Batched Ybus SpMV | 1 (replaced per-scenario `CUSPARSE.mv!` loop with 2D KA kernel) | `aa19cd6` |
| 5 | cuDSS uniform batched solve | 4 new kernels + Newton step rewrite | `980c4bc` |
| 6 | Benchmark | — | — |

**Kernel transform pattern** (applied mechanically to all ~25 kernels):
```julia
# Before: 1D launch, sequential M loop
@kernel function foo_batched_ka!(f, z, ..., @Const(M))
    k = @index(Global)
    for m in 1:M
        # body
    end
end
# ndrange = n_devices

# After: 2D launch, M in the index
@kernel function foo_batched_ka!(f, z, ...)
    k, m = @index(Global, NTuple)
    # body (unchanged)
end
# ndrange = (n_devices, M)
```

**cuDSS uniform batched solve** (Step 5): replaced the per-scenario cuDSS factorize+solve loop with CUDSS.jl's `ubatch_size` API. One `CudssSolver` with `cudss_set(solver, "ubatch_size", M)` handles all M systems sharing the same sparsity pattern in a single `cudss("factorization")` / `cudss("solve")` call.

### Phase 3: Schur-Complement Parallelization (Workflow 2)

The Schur-complement solver decomposes the Jacobian into dense per-device blocks (A) and a sparse reduced network system (S). The hypothesis was that factoring the smaller S via cuDSS batched would avoid the cost cliff seen with the full-system batched factorization.

| Step | Description | Commit |
|------|-------------|--------|
| 1 | 2D Schur gather kernels (A, B, C, fwk) | `21d6c6d` |
| 2 | Batch S assembly and RHS across M | `7570a43` |
| 3 | cuDSS uniform batch on reduced S + batched scatter/backsub | `3665550` |
| 4 | Benchmark | — |

## Benchmark Results

### Baseline: Sequential GPU (before parallelization)

ACTIVSg2000 (2,000 buses, 9,694 unknowns), 1s fault simulation, dt=1/120s.

| M | CPU Batched | GPU cuDSS (sequential) | GPU Speedup |
|---|---|---|---|
| 1 | 1.58s | 1.47s | 1.1x |
| 4 | 6.74s | 4.56s | 1.5x |
| 16 | 28.5s | 18.1s | 1.6x |
| 64 | 119.6s | 69.0s | 1.7x |

### After Parallelization: Per-Phase Profiling

Julia-level profiling of a single Newton iteration (ACTIVSg2000, monolithic cuDSS path):

| Phase | M=1 | M=4 | M=16 | M=64 |
|-------|-----|-----|------|------|
| Residual assembly | 0.37 ms | 0.26 ms | 0.58 ms | 0.61 ms |
| Jacobian assembly | 0.26 ms | 0.21 ms | 0.43 ms | 0.52 ms |
| CSC→CSR permutation | 0.02 ms | 0.03 ms | 0.07 ms | 0.21 ms |
| **cuDSS factorization** | **1.11 ms** | **2.07 ms** | **47.00 ms** | **43.54 ms** |
| Build RHS | 0.04 ms* | 0.04 ms | 0.10 ms | 0.10 ms |
| **cuDSS solve** | **0.59 ms** | **0.36 ms** | **23.02 ms** | **22.26 ms** |
| State update | 0.03 ms* | 0.03 ms | 0.10 ms | 0.14 ms |
| **Total** | **2.42 ms** | **3.00 ms** | **71.31 ms** | **67.38 ms** |
| cuDSS % of total | 70% | 81% | **98%** | **98%** |

*M=1 shows JIT anomalies in Build RHS/Update z; M=4 is the clean baseline.

**Key finding**: the 2D kernel parallelization works perfectly — residual and Jacobian assembly barely grow from M=1 to M=64 (sub-millisecond). The bottleneck is cuDSS batched factorization, which jumps 23x from M=4 to M=16 and then plateaus.

### Monolithic cuDSS vs Schur (both parallelized)

| M | Monolithic cuDSS | Schur cuDSS | Schur overhead |
|---|---|---|---|
| 1 | 1.4s | 7.0s | 5.1x slower |
| 4 | 2.6s | 11.2s | 4.2x slower |
| 16 | 51.5s | 131.2s | 2.5x slower |
| 64 | 53.9s | 140.7s | 2.6x slower |

The Schur path is slower at all batch sizes. cuDSS batched factorization has the same M=16 cliff on the reduced S (~4,000 unknowns) as on the full system (~9,700). The additional kernel launch overhead from the Schur decomposition (gather A/B/C, cuBLAS batched LU, S assembly, backsub) adds ~2.5x cost that isn't recovered from the smaller S factorization.

### Prior 70k-bus result (single scenario, before batched parallelization)

| System | Unknowns | CPU KLU | GPU cuDSS | Speedup |
|--------|----------|---------|-----------|---------|
| ACTIVSg70k | 216,698 | 43.5s | 5.3s | 8.3x |

## Conclusions

1. **2D kernel parallelization is complete and effective.** All ~25 GPU kernels now launch with `ndrange=(n_devices, M)`, and residual/Jacobian assembly cost is essentially M-independent. This is the right architecture.

2. **cuDSS uniform batched factorization has a fixed overhead cliff at M≈16.** This appears to be internal to NVIDIA's cuDSS library — it is not proportional to system size (happens at both 4k and 10k unknowns) and cannot be fixed in user code. From M=16 to M=64, wall time is essentially constant, so throughput does scale in that regime.

3. **The Schur decomposition does not help** at the ACTIVSg2000 scale. The reduced system S is still large enough to hit the cuDSS batched cliff, and the decomposition overhead (many kernel launches, cuBLAS batched LU, scatter/backsub) outweighs the savings. The Schur approach may become favorable at 70k+ buses where S reduction ratio is larger and the full-system factorization is more expensive.

4. **Best current throughput**: monolithic cuDSS batched achieves **1.2 scen/s at M=64** on ACTIVSg2000 with a Quadro GV100. Projected on Polaris A100: ~1.8 scen/s per GPU, ~7 scen/s per node (4x A100), ~**25,000 scen/hour on 1 node**.

## Possible Next Steps

- **CUDA streams**: dispatch M independent cuDSS solvers on separate CUDA streams instead of using the uniform batch API. This bypasses the batched overhead entirely — each stream runs an independent factorization.
- **Target 70k-bus benchmarks**: the GPU advantage is 8.3x at single-scenario; batched parallelization should amplify this since the kernel work (which now scales with M) is a smaller fraction of total cost.
- **Sherman-Morrison fault updates**: for same-topology scenarios differing only in fault location, update S via rank-1 perturbation instead of refactoring. Cost: one matrix-vector product per scenario instead of a full factorization.
- **Lockstep Newton convergence**: synchronize convergence checks across scenarios to eliminate per-scenario branching and maintain batch coherence.

## Code Artifacts

All changes are in `ext/GradPowerCUDAExt/GradPowerCUDAExt.jl` on the `batched_gpu` branch.

| Commit | Description |
|--------|-------------|
| `b893df1` | 2D residual kernels (8 kernels) |
| `7ab353e` | 2D Jacobian kernels (8 kernels) |
| `ac821b3` | 2D utility kernels (9 kernels) |
| `aa19cd6` | Batched Ybus SpMV (replaced CUSPARSE.mv! loop) |
| `980c4bc` | cuDSS uniform batched solve (monolithic path) |
| `21d6c6d` | 2D Schur gather kernels |
| `7570a43` | Batched S assembly and RHS |
| `3665550` | cuDSS uniform batch on reduced S |

All commits verified against CPU baseline trajectories at M=1, 4, 16 with error < 1e-6. The existing 1,047 CPU tests remain passing.
