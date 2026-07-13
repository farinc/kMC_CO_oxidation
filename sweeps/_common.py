"""Shared task list, single-task runner and CSV writer for the beta sweeps.

Both sweeps/linear.py (serial) and sweeps/mpi.py (MPI-parallel) build the
same (idx, beta, init, seed) task list and combine per-task results into the
same {out}_kmc_sweep.csv shape, so the two drivers stay interchangeable.
"""

import argparse

import numpy as np
import pandas as pd

from kmc_co_oxidation import KMCParams, run_kmc

TAGS = ("empty", "full")


def build_betas(beta_min, beta_max, beta_step):
    return np.arange(beta_min, beta_max + 0.5 * beta_step, beta_step)


def build_tasks(betas, seed):
    """List of (idx, beta, init_tag, seed), two entries per beta."""
    tasks = []
    for i, b in enumerate(betas):
        for j, tag in enumerate(TAGS):
            tasks.append((i, float(b), tag, seed + 2 * i + j))
    return tasks


def run_task(task, params, verbose_prefix=""):
    """Run one (idx, beta, init, seed) task, returning (idx, tag, steady coverages)."""
    idx, beta, tag, seed = task
    res = run_kmc(beta, init=tag, params=params, seed=seed)
    note = " [absorbing state]" if res.stuck else ""
    print(f"{verbose_prefix}beta={beta:5.2f} init={tag:5s}  "
          f"theta_CO={res.steady_co:.3f} theta_O={res.steady_o:.3f}  "
          f"({res.steps:.3e} events){note}", flush=True)
    return idx, tag, res.steady_empty, res.steady_co, res.steady_o


def assemble(betas, results):
    """Combine run_task() outputs into the {out}_kmc_sweep.csv dict shape."""
    n = len(betas)
    out = {key: np.full(n, np.nan) for key in
           ("e_empty", "co_empty", "o_empty", "e_full", "co_full", "o_full")}
    for idx, tag, e, co, o in results:
        out[f"e_{tag}"][idx] = e
        out[f"co_{tag}"][idx] = co
        out[f"o_{tag}"][idx] = o
    return out


def save_sweep_csv(betas, sweep, L, path):
    df = pd.DataFrame({"beta": betas, "L": L, **sweep})
    df.to_csv(path, index=False)


def build_argparser(description):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--L", type=int, default=16, help="lattice edge length")
    ap.add_argument("--tmax", type=float, default=30.0, help="kMC time limit, s")
    ap.add_argument("--max-steps", type=int, default=1_000_000_000,
                    help="event limit (whichever hits first)")
    ap.add_argument("--beta-min", type=float, default=0.0)
    ap.add_argument("--beta-max", type=float, default=10.0)
    ap.add_argument("--beta-step", type=float, default=0.4)
    ap.add_argument("--khop-scale", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-interval", type=int, default=10_000)
    ap.add_argument("--out", default="co_oxidation", help="output file prefix")
    return ap


def params_from_args(args):
    return KMCParams(L=args.L, t_max=args.tmax, max_steps=args.max_steps,
                     khop_scale=args.khop_scale, seed=args.seed,
                     sample_interval=args.sample_interval)
