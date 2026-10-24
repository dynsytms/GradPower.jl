# How it works

This page summarizes the model and the numerical method. Each section
points to where it is implemented.

## Steady-state continuation

Continuation is separate from DAE time integration. It solves
``F(x,\lambda)=0`` along a loading direction, adding a weighted arclength
constraint so a simple voltage nose can be traversed even when the ordinary
power-flow Jacobian is singular. Reactive voltage control uses exact projection
complementarity with two-way limit recovery. A nonsmooth control-induced
maximum is distinguished from a smooth fold by one-sided tangents.

See the [illustrated continuation manual](guides/continuation.md#6.-How-the-method-works)
for AC balances, control inequalities, augmented Newton equations, and examples.
The implementation is in `src/cpf/`; optional rendering lives in the plotting
extension. This analysis does not imply time-domain or small-signal stability.

## The DAE

A power system with machines, controllers and loads is a semi-explicit
differential-algebraic system:

```math
\begin{aligned}
\dot x &= f(x, y, v)            && \text{machine and controller dynamics} \\
0      &= g(x, y, v)            && \text{device algebraic equations (stator, governor outputs, ...)} \\
0      &= Y v - I(x, y, v)      && \text{network: current balance at every bus}
\end{aligned}
```

- `x`: differential states, for example the rotor angle ``\delta``, speed
  ``\omega``, and internal EMFs ``e_q', e_d'`` of each machine.
- `y`: algebraic device states.
- `v`: bus voltages in rectangular form ``v_b = v_{r,b} + j v_{i,b}``.
- ``Y``: the network admittance matrix.

The network equation is written in **current-injection** form: each device
injects a current ``I_b(x, y, v)`` at its bus, and Kirchhoff's law gives
``Y v = I``. An impedance-plus-power load injects
``-y_L v_b - \overline{S_P/v_b}``, where ``y_L`` is its impedance-part
admittance and ``S_P`` its constant-power demand. In the implemented `ZIPLoad`,
these contributions are weighted by ``\alpha`` and ``1-\alpha``; a separate
constant-current component is not implemented. The power-current denominator
is clipped below ``|v_b|^2=0.2`` for numerical protection. A fault with nonzero
impedance ``r + jx`` injects ``-v_b / (r + jx)``.

## Initialization

`runpf!` solves the steady state ``Y v = \overline{S / v}`` at given
generation and load. It enforces generator reactive limits by switching PV
buses to PQ. `initialize_dynamics!` then inverts each device's equations at
that operating point. It finds machine internal states that inject exactly
the power-flow current, and controller setpoints (governor reference, exciter
voltage reference) that hold them there. For a consistent, converged
initialization, the equilibrium residuals are small and an undisturbed
simulation should remain close to that state. This is a numerical consistency
check, not a proof of stability; an unstable equilibrium can amplify roundoff
or initialization error.

## Time stepping

With step ``h``, backward Euler (BE) and Crank–Nicolson (CN) both lead, at
every step, to one nonlinear system in ``z_{k+1} = (x, y, v)_{k+1}``:

```math
F(z) = \begin{bmatrix} x - a - c\, f(z) \\ g(z) \\ Y v - I(z) \end{bmatrix} = 0,
\qquad
(a, c) = \begin{cases} (x_k,\ h) & \text{BE} \\ (x_k + \tfrac h2 f(z_k),\ \tfrac h2) & \text{CN} \end{cases}
```

Both methods share the same Newton machinery and only differ in the anchor
``a`` and the coefficient ``c``. The Jacobian is

```math
J = \begin{bmatrix} I - c\,f_x & -c\,f_y & -c\,f_v \\ g_x & g_y & g_v \\ -I_x & -I_y & Y - I_v \end{bmatrix}.
```

Events (a fault switching on, a line tripping, ...) change ``F``. They are
applied between steps, followed by a re-solve of the algebraic equations
(``c = 0``), so the next step starts from a consistent state. This handles
the jump in voltages at a fault.

## Newton's method and its cost

Each step runs Newton iterations ``J \Delta z = -F(z)``. For large networks the
sparse LU factorization of ``J`` dominates. Two cheap savings:
- **Modified Newton**: reuse the factorization across iterations and steps
  while convergence is fast (the default requires about a tenfold residual
  reduction), subject to a stale-iteration limit and refreshes at events.
- **Predictor**: start from the extrapolation ``2z_k - z_{k-1}``.

The savings depend on the case, step size, events, and solver. Use `SolverLog`
to measure residual evaluations and factorizations for a particular study.

## Structure: clusters and Schur reduction

Order the unknowns by **cluster**: each machine together with its governor,
exciter and stabilizer, and finally the bus voltages. A cluster only touches
the voltage of its own bus, so ``J`` has an arrow shape:

```math
J = \begin{bmatrix} A_1 & & & B_1 \\ & \ddots & & \vdots \\ & & A_N & B_N \\ C_1 & \cdots & C_N & D \end{bmatrix},
\qquad
S = D - \sum_{k} C_k A_k^{-1} B_k .
```

- **The blocks:** each ``A_k`` is small and dense. ``B_k`` and ``C_k`` couple the
  cluster to its bus.
- **Elimination:** when the ``A_k`` blocks are invertible, eliminating them
  leaves the **Schur complement** ``S``. Each correction is confined to the
  associated bus's diagonal voltage block, so it adds no interbus coupling.
  Clusters with singular local blocks, such as some static-generator
  constraints, stay in the reduced system; it is not always a voltage-only
  matrix with exactly the sparsity of ``Y``.
- **One Newton step then costs:**
  1. ``N`` independent small dense LUs, which batch perfectly on a GPU;
  2. one sparse LU of a network-sized matrix;
  3. back-substitution.

This is `solver = :schur` on the CPU and `integrate_gpu_schur_cudss!` on the GPU.
`:schur_gmres` instead solves the reduced system iteratively. Its preconditioner
uses the network admittance plus retained constraint rows, is reused between
steps, and is refreshed after topology changes.

## Kernels and layout

Device equations are not written per device instance. Each device *type*
stores its instances as a structure of arrays (a column per parameter). One
kernel evaluates the residual, and one the Jacobian entries, for all
instances of that type. The same kernels run on the CPU and, through
KernelAbstractions.jl, on the GPU. The Jacobian's sparsity pattern is built
once, and each kernel writes into precomputed positions of its nonzeros, so
assembly allocates nothing.

## Where to look in the source

| topic | file |
|:--|:--|
| power flow | `src/pflow.jl` |
| continuation, reactive-control events, physical results | `src/cpf/` |
| residual, Jacobian, time loop, events | `src/dynamics.jl` |
| Newton, Jacobian reuse | `src/numerics.jl` |
| clusters, Schur reduction | `src/clusters.jl`, `src/schur.jl` |
| device kernels / tables | `src/kernels/`, `src/tables/` |
| batched CPU | `src/batched.jl` |
| GPU | `ext/GradPowerCUDAExt/` |
