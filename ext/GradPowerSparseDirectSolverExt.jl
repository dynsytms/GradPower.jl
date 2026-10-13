module GradPowerSparseDirectSolverExt

# GradPower direct-solver interface (src/direct_solver.jl) for
# SparseDirectSolver.jl. One code path for every KernelAbstractions backend:
# the solver runs where its arrays live (SparseMatrixCSC / Vector on the CPU,
# device arrays on a GPU once the GPU package, e.g. CUDA.jl, is loaded).

using SparseDirectSolver
using SparseDirectSolver: DirectSolver, MatrixDescriptor, execute!, setparam!, update!
using GradPower
using GradPower: SparseDirectSolverBackend
using LinearAlgebra
using SparseArrays

const SDSBackend = SparseDirectSolverBackend

# ---- handle-level API (cuDSS phases) ------------------------------------

GradPower.ds_solver(::SDSBackend, rowptr, colval, nzval, structure::String, view::Char) =
    DirectSolver(rowptr, colval, nzval, structure, view)

GradPower.ds_solver(::SDSBackend, A, structure::String, view::Char) =
    DirectSolver(A, structure, view)

GradPower.ds_set!(solver::DirectSolver, key::String, value) = setparam!(solver, key, value)

GradPower.ds_matrix(::SDSBackend, ::Type{T}, n::Integer; nbatch::Integer = 1) where {T} =
    MatrixDescriptor(T, n; nbatch)

GradPower.ds_update!(desc::MatrixDescriptor, buf) = update!(desc, buf)
GradPower.ds_update!(solver::DirectSolver, A) = update!(solver, A)

GradPower.ds_execute!(phase::String, solver::DirectSolver, x, b; asynchronous::Bool = false) =
    execute!(phase, solver, x, b; asynchronous)

# ---- factorization API (CPU integrators) --------------------------------

function GradPower.ds_solver(::SDSBackend, A::SparseMatrixCSC; kwargs...)
    solver = DirectSolver(A, "G", 'F')
    execute!("analysis", solver, nothing, nothing)
    execute!("factorization", solver, nothing, nothing; asynchronous = false)
    return solver
end

function GradPower.ds_factorize!(solver::DirectSolver, A::SparseMatrixCSC)
    update!(solver, A)
    phase = solver.fresh_factorization ? "factorization" : "refactorization"
    execute!(phase, solver, nothing, nothing; asynchronous = false)
    return solver
end

end # module
