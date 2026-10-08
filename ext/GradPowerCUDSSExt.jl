module GradPowerCUDSSExt

# GradPower direct-solver interface (src/direct_solver.jl) for NVIDIA cuDSS
# through CUDSS.jl. GPU only: the matrices are CuSparseMatrixCSR / raw CSR
# device arrays, the right-hand sides CuArrays or CudssMatrix wrappers.

using CUDSS
using GradPower
using GradPower: CUDSSBackend
using SparseArrays

GradPower.ds_solver(::CUDSSBackend, rowptr, colval, nzval, structure::String, view::Char) =
    CudssSolver(rowptr, colval, nzval, structure, view)

GradPower.ds_solver(::CUDSSBackend, A, structure::String, view::Char) =
    CudssSolver(A, structure, view)

GradPower.ds_solver(::CUDSSBackend, ::SparseMatrixCSC; kwargs...) =
    throw(ArgumentError("CUDSSBackend is GPU only; use KLUBackend() or SparseDirectSolverBackend() for CPU matrices"))

GradPower.ds_set!(solver::CudssSolver, key::String, value) = cudss_set(solver, key, value)

GradPower.ds_matrix(::CUDSSBackend, ::Type{T}, n::Integer; nbatch::Integer = 1) where {T} =
    CudssMatrix(T, n; nbatch)

GradPower.ds_update!(matrix::CudssMatrix, buf) = cudss_update(matrix, buf)
GradPower.ds_update!(solver::CudssSolver, A) = cudss_update(solver, A)

GradPower.ds_execute!(phase::String, solver::CudssSolver, x, b; asynchronous::Bool = false) =
    cudss(phase, solver, x, b; asynchronous)

end # module
