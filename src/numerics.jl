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
    jac_verify::Bool=false
)
    
    # Initialize
    success = false
    verbose && @printf("   Iter     Residual inf-norm\n")
    dx = zeros(length(z0))
    z = copy(z0)
    for i = 1:itermax
        # Evaluate the right-hand side
        beuler!(f0, z, zold, u, p, sys, sys.dynamic.diff_dim, dt)
        norm_f = norm(f0, Inf)
        verbose && @printf("   %2d     %.6e\n", i-1, norm_f)
        if norm_f < tol
            success = true
            break
        end
        
        # Evaluate the Jacobian
        beuler_jac!(J0, z, zold, u, p, sys, sys.dynamic.diff_dim, dt)

        # verify jacobian
        if jac_verify
            @assert size(J0, 1) <= 100
            @warn "Jacobian verification"
            function ff(z)
                f = zeros(length(z))
                beuler!(f, z, zold, u, p, sys, sys.dynamic.diff_dim, dt)
                return f
            end
            Jfd = FiniteDiff.finite_difference_jacobian(ff, z)
            valid = compare_matrix(Array(J0), Jfd)
            @assert valid "Jacobian verification failed"
        end
        
        # Solve the linear system
        #fact = klu(J0)
        klu!(fact, J0)
        ldiv!(dx,fact,f0)

        # Update the state
        z -= dx
    end
    z0 .= z
    return success
end
