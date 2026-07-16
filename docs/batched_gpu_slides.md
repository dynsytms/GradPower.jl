---
marp: true
theme: default
paginate: true
math: mathjax
style: |
  section { font-size: 24px; }
  table { font-size: 20px; margin: 0 auto; }
  h1 { font-size: 36px; }
  h2 { font-size: 28px; }
  .columns { display: flex; gap: 2em; }
  .columns > div { flex: 1; }
---

# GradPower.jl — GPU Batched Dynamic Simulation Engine for Surrogate Model Training

**Goal:** Generate $10^5$–$10^6$ contingency simulation scenarios on Argonne's Polaris HPC to train surrogate models for power system dynamics.

**What we built:**

- **Batched simulation engine** — $M$ independent contingency scenarios in a single GPU call, each with a different fault bus and impedance
- **Shared-factorization Schur solver** — one sparse factorization shared across all $M$ scenarios, with per-scenario Woodbury rank-2 corrections for bus faults
- **GPU-resident pipeline** — residual, Jacobian, device elimination, and multi-RHS triangular solve run entirely on GPU with zero host–device transfers per Newton iteration
- **9 device models** (GENROU, GENSAL, IEESGO, TGOV1, SEXS, ESDC1A, IEEEST, ZIPLoad, StaticGen) — portable CPU/GPU kernels via KernelAbstractions.jl

---

# Architecture: Shared-Factor Schur with Woodbury Corrections

<div class="columns">
<div>

**Per-timestep Newton solve (backward Euler):**

1. **Residual assembly** — 2D batched kernels: all $M$ × all devices in parallel
2. **Jacobian assembly** — same 2D pattern, per-scenario fault applied via kernel
3. **Device elimination** — extract $A_k$, factor via cuBLAS batched LU, solve $A_k^{-1} B_k$
4. **Build reduced RHS** — gather + accumulate $C_k A_k^{-1} f_{wk}$ for all $M$
5. **Multi-RHS solve** — one `cuDSS("solve")` call: $P \Delta v = b$ with $(n_\text{red}, M)$ dense RHS
6. **Woodbury correction** — per-scenario rank-2 update: $\Delta v_m \leftarrow P_m^{-1} b_m$
7. **Back-substitute** — recover $\Delta w_m = A_m^{-1}(f_{wm} - B_m \Delta v_m)$

**P is factored once** at the start. No per-scenario, per-iteration factorization.

</div>
<div>

**Key design decisions:**

| Feature | Benefit |
|---|---|
| Shared sparsity | One symbolic analysis for all $M$ |
| Schur reduction | Eliminate device vars, solve smaller $S$ |
| Frozen reference $P$ | Factor once, solve $M$ RHS per iteration |
| Woodbury rank-2 | Exact fault topology correction, no refactor |
| Per-scenario faults | Each of $M$ scenarios gets its own contingency |
| Pre-allocated buffers | Zero allocations in the hot loop |
| 2D KA kernels | `ndrange=(n_devices, M)` — all work parallel |

**Result:** The linear solve — previously 98% of iteration time — becomes a sub-millisecond multi-RHS triangular solve instead of $M$ independent factorizations.

</div>
</div>

---

# Throughput Results and Polaris Projection

**Per-scenario faults, 1 s simulation, $\Delta t = 1/120$ s, Quadro GV100 (32 GB)**

<div class="columns">
<div>

**ACTIVSg200** (200 buses, 970 unknowns)

| $M$ | GPU time | Throughput |
|:---:|:---:|:---:|
| 16 | 0.8 s | 19.8 scen/s |
| 64 | 1.3 s | 49.6 scen/s |
| 128 | 2.1 s | 61.1 scen/s |
| 256 | 2.3 s | **109.5 scen/s** |

**ACTIVSg2000** (2,000 buses, 7,547 unknowns)

| $M$ | GPU time | Throughput |
|:---:|:---:|:---:|
| 16 | 5.4 s | 3.0 scen/s |
| 64 | 9.7 s | 6.6 scen/s |
| 128 | 15.9 s | 8.1 scen/s |
| 256 | 30.8 s | **8.3 scen/s** |

</div>
<div>

**ACTIVSg70k** (70,000 buses, 216,698 unknowns)

| $M$ | GPU time | Throughput |
|:---:|:---:|:---:|
| 1 | 8.0 s | 0.13 scen/s |
| 4 | 17.8 s | 0.22 scen/s |
| 16 | 86.8 s | **0.18 scen/s** |

> GPU throughput scales sub-linearly with $M$ — the shared factorization cost is fixed.

**Projected throughput on Polaris** (A100 ≈ 1.5× GV100):

| System | 1 A100 | 1 node (4× A100) | 10 nodes |
|---|:---:|:---:|:---:|
| 2k-bus | 12 sc/s | 50 sc/s | **1.8M sc/day** |
| 70k-bus | 0.3 sc/s | 1.3 sc/s | **110k sc/day** |

</div>
</div>
