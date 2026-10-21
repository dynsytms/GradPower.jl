# numerical algorithms

using LinearAlgebra

"""Thrown by the integrators when a Newton solve does not converge within its iteration limit."""
struct NewtonConvergenceError <: Exception
    t::Float64
    msg::String
end
Base.showerror(io::IO, e::NewtonConvergenceError) =
    print(io, "NewtonConvergenceError at t = ", e.t, " s: ", e.msg)

"""
    _time_grid(tf, dt) -> (nsteps, tvec)

Fixed-step grid used by all integrators: `nsteps = round(tf/dt)` steps and
`tvec[k] = (k-1)*dt`, so `length(tvec) == nsteps + 1` always matches the
trajectory. The final time is `nsteps*dt`, which differs from `tf` when
`tf` is not a multiple of `dt`.
"""
function _time_grid(tf::Real, dt::Real)
    nsteps = Int(round(tf / dt))
    return nsteps, collect(range(0.0, step=Float64(dt), length=nsteps + 1))
end

# Throw `NewtonConvergenceError` unless `success`. `what` describes the solve.
@inline check_newton(success::Bool, t::Real, what::AbstractString) =
    success || throw(NewtonConvergenceError(Float64(t), what))

"""
    JacobianReuse(enabled; rho=0.1, max_stale=6)

Modified-Newton state shared by the Newton solves of one integration.

When `enabled`, a Newton iteration reuses the factorization of the last
Jacobian, across iterations and time steps, and refactors only if

- there is no factorization for the current `dt` (first step; event
  re-solves at `dt = 0` and the step after them),
- the residual norm fell by less than `1/rho` since the previous iteration,
- `max_stale` iterations of the current solve have used one factorization.

A solve that fails with reuse is retried from its initial guess with a fresh
Jacobian every iteration (full Newton), so reuse cannot make a step fail that
full Newton solves. `dt` is the step size of the current factorization (`NaN`
when there is none).
"""
mutable struct JacobianReuse
    enabled::Bool
    rho::Float64
    max_stale::Int
    dt::Float64
end
JacobianReuse(enabled::Bool; rho::Float64=0.1, max_stale::Int=6) =
    JacobianReuse(enabled, rho, max_stale, NaN)

# Decide whether iteration `iter` of a solve with step `dt` refactors.
@inline _needs_refactor(r::JacobianReuse, use_reuse::Bool, dt::Float64, stale::Int,
                        norm_f::Float64, norm_prev::Float64) =
    !use_reuse || r.dt != dt || stale >= r.max_stale || norm_f > r.rho * norm_prev

function newton_step!(
    z0::AbstractVector,
    f0::AbstractVector,
    J0::AbstractMatrix,
    fact::Base.RefValue,
    zold::AbstractVector,
    u::AbstractVector,
    p::AbstractVector,
    sys::PowerSystem,
    dt::Float64;
    itermax::Int=50,
    tol::Float64=1e-9,
    verbose::Bool=false,
    dx::Union{Nothing,AbstractVector}=nothing,
    zwork::Union{Nothing,AbstractVector}=nothing,
    log::Union{Nothing,SolverLog}=nothing,
    newton_norm::Symbol=:inf,
    limits::Union{Nothing,LimitWorkspace}=nothing,
    reuse::JacobianReuse=JacobianReuse(false),
)
    z_buf = zwork === nothing ? similar(z0) : zwork
    dx_buf = dx === nothing ? zeros(length(z0)) : dx
    success = _newton_loop!(z_buf, z0, f0, J0, fact, zold, u, p, sys, dt, reuse,
                            reuse.enabled, itermax, tol, verbose,
                            dx_buf, log, newton_norm, limits)
    if !success && reuse.enabled
        # Retry from the same initial guess with full Newton.
        success = _newton_loop!(z_buf, z0, f0, J0, fact, zold, u, p, sys, dt, reuse,
                                false, itermax, tol, verbose,
                                dx_buf, log, newton_norm, limits)
    end
    z0 .= z_buf
    return success
end

function _newton_loop!(z_buf, z0, f0, J0, fact, zold, u, p, sys::PowerSystem, dt::Float64,
                       reuse::JacobianReuse, use_reuse::Bool, itermax::Int, tol::Float64,
                       verbose::Bool, dx_buf, log, newton_norm::Symbol,
                       limits)
    success = false
    verbose && @printf("   Iter     Residual norm\n")
    copyto!(z_buf, z0)
    stale = 0
    norm_prev = Inf
    # Pre-extract Union-typed handles once — see comment on
    # `beuler_batched!` / `beuler_jac_batched!` for the rationale.
    dyn  = sys.dynamic::PowerSystemDynamics
    net  = sys.network::Network
    L    = dyn.layout::SimulationLayout
    diff_dim = dyn.diff_dim
    for i = 1:itermax
        # Evaluate the right-hand side
        if log !== nothing
            _t0 = time_ns()
            beuler_batched!(f0, z_buf, zold, u, p, dyn, net.ybus_real, L, diff_dim, dt, log, limits)
            log.residual_ns += time_ns() - _t0
            log.residual_count += 1
        else
            beuler_batched!(f0, z_buf, zold, u, p, dyn, net.ybus_real, L, diff_dim, dt, nothing, limits)
        end
        if newton_norm === :l2
            norm_f = norm(f0, 2)
            tol_eff = tol * sqrt(Float64(length(f0)))
        else
            norm_f = norm(f0, Inf)
            tol_eff = tol
        end
        verbose && @printf("   %2d     %.6e\n", i-1, norm_f)
        if norm_f < tol_eff
            success = true
            break
        end

        if _needs_refactor(reuse, use_reuse, dt, stale, norm_f, norm_prev)
        # Evaluate the Jacobian
        if log !== nothing
            _t0 = time_ns()
            beuler_jac_batched!(J0, z_buf, u, p, dyn, net.ybus_real, L, diff_dim, dt, zold, limits)
            log.jacobian_ns += time_ns() - _t0
            log.jacobian_count += 1
        else
            beuler_jac_batched!(J0, z_buf, u, p, dyn, net.ybus_real, L, diff_dim, dt, zold, limits)
        end

        # Numeric re-factorization: reuse the symbolic analysis of the
        # caller-provided `fact` (created once in `integrate!` by
        # `ds_solver`). `ds_factorize!` may return a new handle (KLU's
        # fresh-factorization fallback), so it is stored back into `fact`.
        if log !== nothing
            _t0 = time_ns()
            fact[] = ds_factorize!(fact[], J0)
            log.lsolve_factor_ns += time_ns() - _t0
            log.lsolve_factor_count += 1
        else
            fact[] = ds_factorize!(fact[], J0)
        end
        reuse.dt = dt
        stale = 0
        end

        if log !== nothing
            _t0 = time_ns()
            ldiv!(dx_buf, fact[], f0)
            log.lsolve_solve_ns += time_ns() - _t0
            log.lsolve_solve_count += 1
        else
            ldiv!(dx_buf, fact[], f0)
        end
        stale += 1
        norm_prev = norm_f

        # Update the state in place (no temporary).
        @inbounds for k in eachindex(z_buf)
            z_buf[k] -= dx_buf[k]
        end
    end
    return success
end
