# GPU portability audit (oneAPI / Aurora readiness), GradPower side

Date: 2026-10-20. Scope: everything in `ext/GradPowerCUDAExt/` *outside* the
sparse direct solver, which is SparseDirectSolver.jl's job (its oneAPI
extension is T31 there).

## Already portable
- **Device kernels** (residual, Jacobian, Schur gather/scatter/assembly,
  events, snapshots): 42 KernelAbstractions `@kernel`s. On a oneAPI backend
  only the `CUDABackend()` launch argument changes (16 call sites).
- **Leaf device code** (`src/kernels/*.jl`, `_<dev>_residual_one!`, ...) is
  generic Julia shared with the CPU path.
- **Sparse direct solver calls** go through `GradPower.ds_*`, so a oneAPI
  build of SparseDirectSolver plugs in with no GradPower change.

## CUDA-specific (to port)
| what | where | count | oneAPI replacement |
|---|---|---|---|
| array types in struct fields / allocs | `GpuBatchedLayout` (121 fields), `CuMatrix`/`CuVector`/`CUDA.zeros` | ~310 | parametrize on the array type (`AT{Float64}`), allocate with `KernelAbstractions.zeros(backend, ...)` |
| `CuSparseMatrixCSR` | Ybus (residual SpMV is our own kernel; only the CSR arrays are used), cuDSS shells | 12 | plain `rowPtr`/`colVal`/`nzVal` device vectors; SDS takes raw CSR arrays |
| batched dense LU of machine blocks | `CUBLAS.getrf_strided_batched!`, `getrs_strided_batched!` | 1 + 4 | oneMKL `getrf_batch`/`getrs_batch` via oneAPI.jl, or a KA kernel (blocks are <= ~20x20) |
| norms / scaling | `CUBLAS.nrm2`, `CUBLAS.scal!` | 5 + 3 | `norm(x)` / `x .*= a` (GPUArrays generic) |
| `CUDA.@allowscalar` | event status writes | 5 | `copyto!` of a host vector |
| cuDSS-only paths | `CuDSSPreconditioner`, `integrate_gpu_shared*` (Woodbury on a shared cuDSS factor) | - | keep CUDA-only, or route through `ds_*` |

## Plan
1. Parametrize `GpuBatchedLayout` on backend + array type and move the
   integrators to `GradPower` proper (`src/gpu/`), with the CUDA extension
   providing only the backend, the dense batched LU and the vendor solver.
2. Replace CUBLAS nrm2/scal! with generic array ops (no measurable cost).
3. Batched dense LU: KA kernel fallback (measured needed only if oneMKL batch
   is slower) + vendor hook.
4. oneAPI extension = backend + oneMKL batched LU + `oneArray` CSR adapters.
Steps 1-2 are testable here on CUDA (results must be unchanged); 3-4 need an
Intel GPU (Sunspot/Aurora).
EOF
mkdir -p docs/notes 2>/dev/null; true