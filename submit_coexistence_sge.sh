#!/bin/bash
#$ -N coexist_slepc
#$ -cwd
#$ -j y
#$ -pe mpi 24

module purge
unset OMPI_MCA_btl
export OMPI_MCA_btl=self,vader,tcp

module load gcc/10.3.0
module load openmpi/4.1.2

source .venv/bin/activate

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# One-time setup in the venv (NOT done automatically here):
#   pip install mpi4py
#   pip install petsc petsc4py slepc slepc4py
# The pip petsc build takes a while; to get MUMPS (needed for the
# shift-invert factorization on big tiles) build with:
#   PETSC_CONFIGURE_OPTIONS="--download-mumps --download-scalapack" \
#       pip install --no-binary petsc petsc
# If the cluster provides PETSc/SLEPc modules instead, load them and set
# PETSC_DIR/SLEPC_DIR before pip-installing only petsc4py/slepc4py.

# Everything after the script's own flags is forwarded to PETSc/SLEPc,
# e.g.: ./submit_coexistence_sge.sh --sites 12 -eps_monitor
mpirun -np 24 python exploration/find_coexistence_mpi.py "$@"
