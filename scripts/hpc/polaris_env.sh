#!/bin/bash
# Environment variables for GradPower.jl GPU (CUDA + cuDSS) runs on Polaris.
#
# this file is sourced by  `polaris_bootstrap.sh` during env setup, you shouldn't have to call this directly
#
# Overridable env vars:
#   GP_ENV        Julia project holding GradPower + CUDA + CUDSS (default ~/gpenv)
#   CUDSS_ROOT    unpacked cuDSS redistributable   (default ~/cudss/libcudss-...)
#   CUDA_LOCAL    local NVIDIA HPC SDK CUDA prefix (default 25.5 / 12.9)
#

export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export NO_PROXY="localhost,127.0.0.1,.alcf.anl.gov"

# Julia
module use /soft/modulefiles 2>/dev/null || true
module load julia/1.12 2>/dev/null || module load julia 2>/dev/null || true
export JULIA="${JULIA:-julia}"
export GP_ENV="${GP_ENV:-$HOME/gpenv}"

# cuDSS
export CUDSS_VERSION="${CUDSS_VERSION:-0.8.0.10_cuda12}"
export CUDSS_ROOT="${CUDSS_ROOT:-$HOME/cudss/libcudss-linux-x86_64-${CUDSS_VERSION}-archive}"
export JULIA_CUDSS_LIBRARY_PATH="$CUDSS_ROOT/lib"

# cuBLAS/cuSPARSE
CUDA_LOCAL="${CUDA_LOCAL:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.5}"
export LD_LIBRARY_PATH="$JULIA_CUDSS_LIBRARY_PATH:$CUDA_LOCAL/cuda/12.9/lib64:$CUDA_LOCAL/math_libs/12.9/lib64:${LD_LIBRARY_PATH:-}"

if [[ -f "$GP_ENV/LocalPreferences.toml" ]] && grep -q "version" "$GP_ENV/LocalPreferences.toml" 2>/dev/null; then
    echo "WARNING: $GP_ENV/LocalPreferences.toml may pin a CUDA artifact runtime;" >&2
    echo "         delete it if you hit 'Failure artifact: CUDA_Runtime'." >&2
fi

if [[ ! -f "$JULIA_CUDSS_LIBRARY_PATH/libcudss.so" ]]; then
    echo "NOTE: no libcudss.so under $JULIA_CUDSS_LIBRARY_PATH yet." >&2
    echo "      Source polaris_bootstrap.sh instead of this file and it will fetch it." >&2
fi
