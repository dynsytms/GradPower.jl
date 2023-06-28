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

