"""
Serial beta sweep: runs every (beta, init) kMC case one after another on a
single core and writes {out}_kmc_sweep.csv.

Usage:
    uv run python -m sweeps.linear
    uv run python -m sweeps.linear --L 24 --out case1
"""

from sweeps._common import (assemble, build_argparser, build_betas,
                            build_tasks, params_from_args, run_task,
                            save_sweep_csv)


def run_sweep(betas, params, seed):
    """Run every (beta, init) task serially. Returns the {out}_kmc_sweep.csv dict."""
    tasks = build_tasks(betas, seed)
    results = [run_task(task, params) for task in tasks]
    return assemble(betas, results)


def main():
    ap = build_argparser(__doc__.splitlines()[1])
    args = ap.parse_args()

    params = params_from_args(args)
    betas = build_betas(args.beta_min, args.beta_max, args.beta_step)
    sweep = run_sweep(betas, params, args.seed)

    save_sweep_csv(betas, sweep, args.L, f"{args.out}_kmc_sweep.csv")
    print(f"Data written to '{args.out}_kmc_sweep.csv'.")


if __name__ == "__main__":
    main()
