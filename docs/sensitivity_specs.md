# Historical Sensitivity Acceptance Criteria

The inactive legacy sensitivity tests and examples were removed in cleanup
item 4.4 because they called APIs removed with `src/ad.jl` and
`src/sensitivities.jl`. These are specifications for future tests, not
currently supported or validated functionality. `Pkg.test()` does not validate
adjoint or tangent-linear sensitivities; `DynamicNLP` supports only forward
objective evaluation.

## Finite-Difference Comparisons

Unless stated otherwise, use the bundled 2-bus case, a fault on internal bus 2
from 0.2 to 0.3 seconds, backward Euler, and `dt = 1/120`. Reset states,
parameters and event registrations before each evaluation.

| Target | Historical setup | Historical tolerance |
| --- | --- | --- |
| Initial-state TLM | `rfault=0.2`, final time 1 s, all-ones state direction; centered FD step `1e-6` | `rtol=1e-6` |
| Parameter TLM | Same case, zero initial-state direction, all-ones parameter direction; centered FD step `1e-4` | `rtol=1e-3` |
| Terminal adjoint | `rfault=0.02`, final time 4.8 s; terminal state row 4, initial-state and parameter gradients against FiniteDiff | `rtol=1e-6` |
| Running-objective adjoint | Same case, zero terminal seed; right-endpoint quadrature of summed squared rotor speed deviations | `rtol=1e-4` |
| Governor-coupled TLM | `2bus_IEESGO.dyr`, `rfault=0.02`, final time 1 s, direction in first GENROU's `H`; centered FD step `1e-3` | `rtol=1e-3` |

The last test covers a machine parameter through governor coupling, not
IEESGO parameter derivatives. Parameter perturbations in the old tests were
applied after initialization: initialization derivatives were not covered.
The old residual-AD example compared an all-ones parameter JVP against finite
differences; it did not validate a VJP. Resolve parameter positions through
device metadata rather than the obsolete 12-parameter GENROU stride.

## Event-Offset Regression

For the running objective above, differentiate with respect to the first
GENROU's `H` (offset `+6` from its parameter pointer). Use centered differences
with `h = 1e-5 * max(abs(H), 1)`. Require relative error below `1e-6`,
normalizing by `max(abs(gradient_fd), 1e-12)`.

The historical test described faulted forward steps
`[step_on+1, step_off]` and reverse trajectory indices
`[step_on+2, step_off+1]`. Its header reported relative errors near `1.5e-9`
with the `+2` offset and `0.15` after changing it to `+1`. These measurements
have not been rerun and the offsets are not a complete adjoint algorithm.

Current integration records a pre-projection event sample, then performs a
`dt=0` algebraic projection before the next step. A new discrete adjoint must
differentiate that projection and the actual DAE residual, including its
differential/algebraic row scaling. Modified Newton factors are not generally
the accepted-state Jacobian: reassemble exact Jacobians for reverse solves.
Add independent finite differences and timestep-refinement checks before
treating any historical tolerance as an acceptance gate.
