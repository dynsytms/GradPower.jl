"""
    ContinuationStats()

Opt-in cumulative CPU work counters/timers for `runcpf(...; stats=log)`.
Fields ending in `_ns` are elapsed nanoseconds. Setup, residual, Jacobian,
factorization, triangular solve, and fold verification timers are separate;
they do not include every allocation, event search, or result-recording cost.
`factorizations` counts fresh KLU attempts, not successful solves or symbolic
analyses separately. Reusing a log accumulates work across calls. Rejected
steps are reported separately in `ContinuationResult`. Timings include
compilation unless the caller warms the same workload first.
"""
Base.@kwdef mutable struct ContinuationStats
    setups::Int = 0
    setup_ns::UInt64 = 0
    residuals::Int = 0
    residual_ns::UInt64 = 0
    jacobians::Int = 0
    jacobian_ns::UInt64 = 0
    factorizations::Int = 0
    factorization_ns::UInt64 = 0
    solves::Int = 0
    solve_ns::UInt64 = 0
    fold_checks::Int = 0
    fold_ns::UInt64 = 0
end
