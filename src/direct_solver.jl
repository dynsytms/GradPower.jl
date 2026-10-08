# -----------------------------------------------------------------------
# Sparse direct solver backends
#
# GradPower's Newton solves go through a small interface so the sparse
# direct solver can be switched:
#   * KLUBackend                 — CPU, KLU.jl (direct dependency, default on CPU)
#   * CUDSSBackend               — GPU, CUDSS.jl (weak dependency, ext/GradPowerCUDSSExt.jl)
#   * SparseDirectSolverBackend  — CPU or GPU, SparseDirectSolver.jl
#                                  (weak dependency, ext/GradPowerSparseDirectSolverExt.jl)
#
# The handle-level functions (`ds_solver` with raw CSR arrays, `ds_set!`,
# `ds_matrix`, `ds_update!`, `ds_execute!`) follow the cuDSS phase API and
# are used by the GPU extension. The CPU code uses `ds_solver(b, A)` and
# `ds_factorize!(fact, A)` on `SparseMatrixCSC` and solves with `ldiv!`.
# -----------------------------------------------------------------------

"""
    AbstractDirectSolverBackend

Selects the sparse direct solver used by the Newton iterations. Pass a concrete
backend as `linear_solver = ...` to [`integrate!`](@ref), `integrate_batched!`,
or the GPU `GpuBatchedLayout` constructor.
"""
abstract type AbstractDirectSolverBackend end

"""
    KLUBackend()

CPU sparse LU from KLU.jl. Default for the CPU integrators.
"""
struct KLUBackend <: AbstractDirectSolverBackend end

"""
    CUDSSBackend()

NVIDIA cuDSS through CUDSS.jl (GPU only). Requires `using CUDA, CUDSS`.
"""
struct CUDSSBackend <: AbstractDirectSolverBackend end

"""
    SparseDirectSolverBackend()

SparseDirectSolver.jl, on whichever KernelAbstractions backend the matrix lives
(CPU arrays or GPU arrays). Requires `using SparseDirectSolver`; GPU use also
needs the GPU package (e.g. `using CUDA`).
"""
struct SparseDirectSolverBackend <: AbstractDirectSolverBackend end

"""
    ds_solver(backend, A::SparseMatrixCSC; tuned = false) -> fact

Create a solver handle for the general (unsymmetric) matrix `A`, run the
symbolic analysis and the numeric factorization. Solve with `ldiv!`.
`tuned = true` applies the KLU settings of the monolithic integrator
(no scaling, no BTF, COLAMD, pivot tolerance 1e-3); other backends ignore it.

    ds_solver(backend, rowptr, colval, nzval, structure, view) -> solver
    ds_solver(backend, A_csr, structure, view) -> solver

Create a cuDSS-style handle on CSR data (no phase is run). `nzval` may be an
`nnz × nbatch` matrix for a uniform batch.
"""
function ds_solver end

"""
    ds_factorize!(fact, A::SparseMatrixCSC) -> fact

Numeric (re)factorization of `A`, which has the sparsity pattern `fact` was
built on. Returns the handle to use afterwards (it may be a new object).
"""
function ds_factorize! end

"""
    ds_set!(solver, key::String, value)

Set a cuDSS-style solver option, e.g. `"ubatch_size"`.
"""
function ds_set! end

"""
    ds_matrix(backend, T, n; nbatch = 1) -> wrapper

Dense right-hand-side/solution wrapper of `n` rows for a uniform batch of
`nbatch` systems. Point it at data with [`ds_update!`](@ref).
"""
function ds_matrix end

"""
    ds_update!(wrapper, buf)
    ds_update!(solver, A_csr)

Point a dense wrapper at `buf`, or point a solver at new matrix values with the
same sparsity pattern (refactorize afterwards).
"""
function ds_update! end

"""
    ds_execute!(phase::String, solver, x, b; asynchronous = false)

Run a cuDSS phase (`"analysis"`, `"factorization"`, `"refactorization"`,
`"solve"`) on `solver`.
"""
function ds_execute! end

# -----------------------------------------------------------------------
# KLU backend (CPU)
# -----------------------------------------------------------------------

function ds_solver(::KLUBackend, A::SparseMatrixCSC; tuned::Bool = false)
    fact = klu(A)
    tuned || return fact
    fact.common.scale = 0
    fact.common.btf = 0
    fact.common.ordering = 1
    fact.common.tol = 1e-3
    return fact
end

ds_solver(::KLUBackend, args...; kwargs...) =
    throw(ArgumentError("KLUBackend only supports CPU SparseMatrixCSC matrices; use CUDSSBackend() or SparseDirectSolverBackend() for GPU/batched solves"))

# klu! with a fallback to a fresh klu() when the pivot sequence chosen at
# analysis time hits a zero pivot (rare; happens after large state jumps).
function ds_factorize!(fact::KLU.KLUFactorization, A::SparseMatrixCSC)
    try
        klu!(fact, A)
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        fact = klu(A)
    end
    return fact
end

# -----------------------------------------------------------------------
# Missing-extension fallbacks
# -----------------------------------------------------------------------

ds_solver(::CUDSSBackend, args...; kwargs...) =
    error("CUDSSBackend requires CUDSS.jl: run `using CUDA, CUDSS` first")
ds_solver(::SparseDirectSolverBackend, args...; kwargs...) =
    error("SparseDirectSolverBackend requires SparseDirectSolver.jl: run `using SparseDirectSolver` first")
ds_matrix(::CUDSSBackend, args...; kwargs...) =
    error("CUDSSBackend requires CUDSS.jl: run `using CUDA, CUDSS` first")
ds_matrix(::SparseDirectSolverBackend, args...; kwargs...) =
    error("SparseDirectSolverBackend requires SparseDirectSolver.jl: run `using SparseDirectSolver` first")

"""
    default_gpu_backend() -> AbstractDirectSolverBackend

`CUDSSBackend()` when CUDSS.jl is loaded, else `SparseDirectSolverBackend()`
when SparseDirectSolver.jl is loaded, else an error.
"""
function default_gpu_backend()
    Base.get_extension(@__MODULE__, :GradPowerCUDSSExt) !== nothing && return CUDSSBackend()
    Base.get_extension(@__MODULE__, :GradPowerSparseDirectSolverExt) !== nothing && return SparseDirectSolverBackend()
    error("no GPU sparse direct solver loaded: run `using CUDSS` or `using SparseDirectSolver`")
end
