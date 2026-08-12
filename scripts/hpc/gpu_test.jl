# Run GradPower's GPU test suite (test/test_gpu_backend.jl) against CUDA devices
# Driven by scripts/hpc/gpu-test-polaris.sh, runs standalone like:
#
#   julia --project=$GP_ENV scripts/hpc/gpu_test.jl
#
# Expected on an A100: `gpu_backend | 49  49`.

using Pkg
haskey(ENV, "GP_ENV") && Pkg.activate(ENV["GP_ENV"])

using CUDA

println("=== CUDA.functional() = ", CUDA.functional(),
        " ; runtime = ", CUDA.functional() ? string(CUDA.runtime_version()) : "n/a", " ===")
CUDA.functional() || error("no functional CUDA device — run this on a compute node")
CUDA.versioninfo()

using CUDSS
using GradPower, Test, LinearAlgebra, SparseArrays

ext = Base.get_extension(GradPower, :GradPowerCUDAExt)
ext === nothing && error("GradPowerCUDAExt failed to load (CUDSS/CUDA not visible to GradPower?)")

repo = get(ENV, "GRADPOWER_DIR", abspath(joinpath(@__DIR__, "..", "..")))
include(joinpath(repo, "test", "test_gpu_backend.jl"))
