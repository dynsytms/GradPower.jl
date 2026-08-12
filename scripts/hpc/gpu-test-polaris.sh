#!/bin/bash
# GPU backend test suite (test/test_gpu_backend.jl) on ALCF Polaris (A100).
# Expected result: `gpu_backend | 49  49`.
#
# Submit from the GradPower.jl repo root:
#   qsub -A <alloc> scripts/hpc/gpu-test-polaris.sh
#
#PBS -A GridFM
#PBS -N gp-gpu-test
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l filesystems=home
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -j oe

set -euo pipefail
cd "${PBS_O_WORKDIR:-$PWD}"

export GRADPOWER_DIR="${GRADPOWER_DIR:-$PWD}"

# env + idempotent provisioning; aborts the job (set -e) if it cannot complete
set +u
source scripts/hpc/polaris_bootstrap.sh
set -u

echo "== node: $(hostname) =="
nvidia-smi -L || true
echo

"${JULIA:-julia}" --project="$GP_ENV" scripts/hpc/gpu_test.jl
