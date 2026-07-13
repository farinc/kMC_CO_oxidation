"""
MPI-parallel beta sweep: splits the (beta, init) kMC cases round-robin
across MPI ranks, gathers the results on rank 0 and writes
{out}_kmc_sweep.csv. Requires the `mpi` extra (mpi4py) and an MPI launcher.

Usage:
    mpirun -np 4 uv run python -m sweeps.mpi
    mpirun -np 4 uv run python -m sweeps.mpi --L 24 --out case1
"""

from mpi4py import MPI

from sweeps._common import (assemble, build_argparser, build_betas,
                            build_tasks, params_from_args, run_task,
                            save_sweep_csv)


def run_sweep(betas, params, seed, comm=None):
    """Run every (beta, init) task round-robin across MPI ranks.

    Returns the {out}_kmc_sweep.csv dict on rank 0, None on every other rank
    (mirrors comm.gather's root-only result). Works with COMM_WORLD's
    default single-rank world too, e.g. when called outside of mpirun.
    """
    comm = comm or MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    tasks = build_tasks(betas, seed)
    local_tasks = [t for t in tasks if t[0] % size == rank]
    local_results = [run_task(task, params, verbose_prefix=f"[rank {rank}] ")
                     for task in local_tasks]

    gathered = comm.gather(local_results, root=0)

    if rank != 0:
        return None
    results = [r for part in gathered for r in part]
    return assemble(betas, results)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ap = build_argparser(__doc__.splitlines()[1])
    args = ap.parse_args()

    params = params_from_args(args)
    betas = build_betas(args.beta_min, args.beta_max, args.beta_step)
    sweep = run_sweep(betas, params, args.seed, comm=comm)

    if rank == 0:
        save_sweep_csv(betas, sweep, args.L, f"{args.out}_kmc_sweep.csv")
        print(f"Data written to '{args.out}_kmc_sweep.csv'.")


if __name__ == "__main__":
    main()
