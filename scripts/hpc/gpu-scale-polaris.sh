#!/bin/bash
# Multi-node / multi-GPU data-parallel scaling run on ALCF Polaris.

# Submit from the GradPower.jl repo root:
#   qsub -A <alloc> scripts/hpc/gpu-scale-polaris.sh                     # 2 nodes
#   qsub -A <alloc> -l select=8 -q prod scripts/hpc/gpu-scale-polaris.sh # 8 nodes
#
# Env knobs (pass via `qsub -v`):
#   GP_CASE     ieee9 | ieee39                   [ieee39]
#   GP_M        scenarios per GPU                [512]
#   GP_METHODS  schur_cudss | shared | cpu       [schur_cudss]
#   GP_SECONDS  sustained load window per rank   [15]
#   GP_REF      single-GPU reference scen/s for the efficiency column
#               (omit to use the fastest rank in this run)
#
#PBS -A GridFM
#PBS -N gp-gpu-scale
#PBS -q debug-scaling
#PBS -l select=2:system=polaris
#PBS -l filesystems=home
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -j oe

set -euo pipefail
cd "${PBS_O_WORKDIR:-$PWD}"

export GRADPOWER_DIR="${GRADPOWER_DIR:-$PWD}"

set +u
source scripts/hpc/polaris_bootstrap.sh
set -u
export GP_CASE="${GP_CASE:-ieee39}"
export GP_M="${GP_M:-512}"
export GP_METHODS="${GP_METHODS:-schur_cudss}"
GP_SECONDS="${GP_SECONDS:-15}"
export JULIA_NUM_THREADS=1

# Polaris: 4 A100 + 32 CPU cores per node run at 4 ranks/node, 8 cores per rank.
PPN=4
DEPTH=8
NODES=$(wc -l < "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || echo 1)
NRANKS=$(( NODES * PPN ))

# results dirs are split by rank and need to be merged after data generation
JOBID="${PBS_JOBID:-local}"; JOBID="${JOBID%%.*}"     # job name 7300792.polaris-pbs-01 has rank output dir: 7300792
OUTDIR="${BENCH_JSON_DIR:-$GRADPOWER_DIR/benchmarks/results}/scale_${JOBID}"
mkdir -p "$OUTDIR"

echo "== $NODES node(s) x $PPN GPU = $NRANKS ranks; case=$GP_CASE M/GPU=$GP_M window=${GP_SECONDS}s =="
cat "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || true
echo

# warm start the depot once so N ranks don't race on precompilation.
"${JULIA:-julia}" --project="$GP_ENV" -e 'using CUDA, CUDSS, GradPower' >/dev/null

mpiexec -n "$NRANKS" --ppn "$PPN" --depth="$DEPTH" --cpu-bind depth \
    "${JULIA:-julia}" --project="$GP_ENV" scripts/hpc/gpu_bench_mpi.jl \
        --case "$GP_CASE" --M "$GP_M" --method "$GP_METHODS" \
        --seconds "$GP_SECONDS" --out-dir "$OUTDIR"

echo
"${JULIA:-julia}" --project="$GP_ENV" scripts/hpc/gpu_scale_report.jl "$OUTDIR" \
    ${GP_REF:+--ref "$GP_REF"}
echo
echo "== per-rank JSON in $OUTDIR =="
