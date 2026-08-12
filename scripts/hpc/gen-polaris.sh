#!/bin/bash
# Dynamic-stability data generation on ALCF Polaris, cpu-only run script see gpu-scale-polaris for GPU run.
#
# runs one MPI rank per core. generate_dynamics.jl reads rank/size from PMI_RANK/PMI_SIZE and round-robins shards
#
# Submit from the GradPower.jl repo root:
#   qsub -V -q debug -l select=1 -l walltime=01:00:00 job_submission_scripts/gen-polaris.sh
# Note: change the GridFM allocation to a current one when that expires
#PBS -A GridFM
#PBS -N dynstab-gen
#PBS -l filesystems=home:eagle:grand
#PBS -j oe

set -euo pipefail
cd "${PBS_O_WORKDIR:-$PWD}"

SWEEP="${SWEEP:-scripts/sweeps/example.toml}"
export DYNSTAB_OUTPUT_ROOT="${ROOT:-/eagle/GridFM/datasets}/dynstab/DynStab/raw"
export JULIA_NUM_THREADS=1

module use /soft/modulefiles 2>/dev/null || true
module load julia 2>/dev/null || true
JULIA="${JULIA:-julia}"

# ranks = nodes * 32 cores.
NODES=$(wc -l < "$PBS_NODEFILE")
PPN=32
NRANKS=$(( NODES * PPN ))

# if you see errors about julia availability, run this once on your login node:
#   julia --project=job_submission_scripts -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'

echo "Generating: sweep=$SWEEP -> $DYNSTAB_OUTPUT_ROOT with $NRANKS ranks ($NODES node(s))"
mpiexec -n "$NRANKS" --ppn "$PPN" --depth=1 --cpu-bind depth \
    "$JULIA" --project=scripts scripts/generate_dynamics.jl "$SWEEP"
