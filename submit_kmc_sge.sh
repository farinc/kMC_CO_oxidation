#!/bin/bash
#
# SGE + OpenMPI submission script for sweeps/mpi.py.
#
# One-time setup on the cluster (run once from the repo root):
#   module load openmpi          # or whatever module exposes OpenMPI on your site
#   uv sync --extra mpi          # builds .venv/, compiles mpi4py against that OpenMPI
#
# Submit with:
#   qsub jobs/submit_kmc_sge.sh --L 24 --out case1
#
# Everything after the script name is forwarded to sweeps/mpi.py, so any of
# its flags (--L, --beta-min/max/step, --khop-scale, --seed, --tmax,
# --max-steps, --sample-interval, --out) can be set at submit time.
#
#$ -N kmc_co_ox
#$ -cwd
#$ -j y
#$ -o logs/
#$ -pe mpi 24

mkdir -p logs

source .venv/bin/activate

module load openmpi/4.1.2

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

mpirun -np 1 python -m sweeps.mpi "$@"
