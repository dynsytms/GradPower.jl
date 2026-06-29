# numerical algorithms

using LinearAlgebra

"""
    newton(x0, J0, rhs_fun!, jac_fun!; itermax=100, tol=1e-12, verbose=false)

Use the Newton-Raphson method to solve a system of nonlinear equations.

# Arguments
- `x0::AbstractVector{T}`: Initial guess.
- `J0::AbstractMatrix{T}`: Initial Jacobian.
- `rhs_fun!::Function`: Function to evaluate the right-hand side of the system.
- `jac_fun!::Function`: Function to evaluate the Jacobian.

# Keywords
- `itermax::Int=100`: Maximum number of iterations.
- `tol::T=1e-6`: Convergence tolerance.
- `verbose::Bool=false`: If `true`, print the iteration

# Returns
- `x::AbstractVector{T}`: The approximate solution.
- `success::Bool`: `true` if the method converged within `itermax` steps, `false` otherwise.

"""
function newton(
    x0::AbstractVector{T}, 
    J0::AbstractMatrix{T}, 
    rhs_fun!::Function, 
    jac_fun!::Function;
    itermax::Int=50, 
    tol::T=1e-12, 
    verbose::Bool=false
) where T <: Number
    x = x0
    f = zeros(T, length(x0))
    success = false
    verbose && @printf("   Iter     Residual inf-norm\n")
    for i = 1:itermax
        rhs_fun!(f, x)
        norm_f = norm(f, Inf)
        verbose && @printf("   %2d     %.6e\n", i-1, norm_f)
        if norm_f < tol
            success = true
            break
        end
        jac_fun!(J0, x)
        dx = -J0 \ f
        x += dx
    end
    return x, success
end


function newton_step!(
    z0::AbstractVector,
    f0::AbstractVector,
    J0::AbstractMatrix,
    fact,
    zold::AbstractVector,
    u::AbstractVector,
    p::AbstractVector,
    sys::PowerSystem,
    dt::Float64;
    itermax::Int=50,
    tol::Float64=1e-9,
    verbose::Bool=false,
    jac_verify::Bool=false,
    dx::Union{Nothing,AbstractVector}=nothing,
    zwork::Union{Nothing,AbstractVector}=nothing,
    log::Union{Nothing,SolverLog}=nothing,
)

    jac_verify = false
    # Initialize
    success = false
    verbose && @printf("   Iter     Residual inf-norm\n")
    # Reuse caller-provided scratch when supplied; otherwise allocate
    # once per Newton call (legacy path for callers that do not pass
    # scratch).
    dx_buf = dx === nothing ? zeros(length(z0)) : dx
    z_buf  = zwork === nothing ? similar(z0) : zwork
    copyto!(z_buf, z0)
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
            beuler_batched!(f0, z_buf, zold, u, p, dyn, net.ybus_real, L, diff_dim, dt, log)
            log.residual_ns += time_ns() - _t0
            log.residual_count += 1
        else
            beuler_batched!(f0, z_buf, zold, u, p, dyn, net.ybus_real, L, diff_dim, dt)
        end
        norm_f = norm(f0, Inf)
        verbose && @printf("   %2d     %.6e\n", i-1, norm_f)
        if norm_f < tol
            success = true
            break
        end

        # Evaluate the Jacobian
        if log !== nothing
            _t0 = time_ns()
            beuler_jac_batched!(J0, z_buf, u, p, dyn, net.ybus_real, L, diff_dim, dt)
            log.jacobian_ns += time_ns() - _t0
            log.jacobian_count += 1
        else
            beuler_jac_batched!(J0, z_buf, u, p, dyn, net.ybus_real, L, diff_dim, dt)
        end

        # verify jacobian
        if jac_verify
            #@assert size(J0, 1) <= 100
            @warn "Jacobian verification"
            function ff(zz)
                ftmp = zeros(length(zz))
                beuler!(ftmp, zz, zold, u, p, sys, diff_dim, dt)
                return ftmp
            end
            Jfd = FiniteDiff.finite_difference_jacobian(ff, z_buf)
            valid = compare_matrix(Array(J0), Jfd)
            @assert valid "Jacobian verification failed"
            @assert false "Jacobian verification passed"
        end

        # Solve. First iter does full symbolic+numeric factor (refresh
        # ordering — dt may have changed, or z drifted enough to invalidate
        # the prior pivot order). Subsequent iters reuse the symbolic and
        # only re-do numeric, which is ~3× cheaper. The iter-1 `klu` call
        # allocates inside KLU.jl and is the single per-Newton allocation
        # the integrate! loop cannot avoid without a forked solver; all
        # later iterations of the inner loop are zero-alloc on our side.
        if log !== nothing
            _t0 = time_ns()
            if i == 1
                fact = klu(J0)
            else
                klu!(fact, J0)
            end
            log.lsolve_factor_ns += time_ns() - _t0
            log.lsolve_factor_count += 1

            _t0 = time_ns()
            ldiv!(dx_buf, fact, f0)
            log.lsolve_solve_ns += time_ns() - _t0
            log.lsolve_solve_count += 1
        else
            if i == 1
                fact = klu(J0)
            else
                klu!(fact, J0)
            end
            ldiv!(dx_buf, fact, f0)
        end

        # Update the state in place (no temporary).
        @inbounds for k in eachindex(z_buf)
            z_buf[k] -= dx_buf[k]
        end
    end
    z0 .= z_buf
    return success
end
