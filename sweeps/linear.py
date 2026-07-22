"""
Serial beta sweep: runs every (beta, init) kMC case and the matching ME-MKM /
SLEPc coexistence analysis on a single core, writing {out}_kmc_sweep.csv (kMC +
ME-MKM coverages and the basin log-ratio) and {out}_coexistence.csv (the
transition rates at each coexistence point beta*).

Usage:
    uv run python -m sweeps.linear
    uv run python -m sweeps.linear --L 24 --sites 8 --out case1
    uv run python -m sweeps.linear --no-coexistence          # kMC only
"""

from sweeps._common import (assemble, build_argparser, build_betas,
                            build_tasks, build_tile, delta_scale_of,
                            maybe_plot_coexistence, maybe_plot_sweep,
                            params_from_args,
                            run_coexistence, run_task, save_coexistence_csv,
                            save_sweep_csv)


def run_sweep(betas, params, seed, delta_scale=0.0):
    """Run every (beta, init) task serially. Returns the {out}_kmc_sweep.csv dict."""
    tasks = build_tasks(betas, seed)
    results = [run_task(task, params, delta_scale=delta_scale)
              for task in tasks]
    return assemble(betas, results)


def main():
    ap = build_argparser(__doc__.splitlines()[1])
    args, _ = ap.parse_known_args()          # let any PETSc/SLEPc options pass

    params = params_from_args(args)
    betas = build_betas(args.beta_min, args.beta_max, args.beta_step)
    dscale = delta_scale_of(args)
    sweep = run_sweep(betas, params, args.seed, delta_scale=dscale)

    if not args.no_coexistence:
        tile = build_tile(args)
        print("ME-MKM / SLEPc coexistence phase")
        cols, rows, arrays = run_coexistence(betas, tile, args, comm=None)
        sweep.update(cols)
        coex_path = args.coexistence_out or f"{args.out}_coexistence.csv"
        if save_coexistence_csv(rows, coex_path):
            print(f"Coexistence data written to '{coex_path}'.")
        if args.plot:
            maybe_plot_coexistence(cols, rows, arrays, betas, args.out)

    save_sweep_csv(betas, sweep, args.L, f"{args.out}_kmc_sweep.csv",
                   delta_scale=dscale)
    print(f"Data written to '{args.out}_kmc_sweep.csv'.")

    if args.plot:
        maybe_plot_sweep(sweep, betas, args, dscale)


if __name__ == "__main__":
    main()
