"""
Plot: render the bifurcation and rate figures for a beta sweep.

Reads a {out}_kmc_sweep.csv file (as written by sweeps/linear.py or
sweeps/mpi.py), computes the matching mean-field branches (Fig. 3) and
rate curves (Fig. 4) on the fly via meanfield.branches/meanfield.rates,
and writes {out}_bifurcation.png and {out}_rates.png next to it. Fig. 3 /
Fig. 4 style plots of Tian & Rangarajan (2021). delta = beta * 1e-4
throughout, matching the sweeps' --delta-scale-beta flag.

Usage:
    uv run python -m co_oxidation.plotting co_oxidation_kmc_sweep.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import meanfield

sns.set_theme(style="whitegrid")

DELTA_SCALE = 1e-4
MODELS = ("mf", "ea")
MODEL_LABELS = {"mf": "MF-MK", "ea": "Ea-MK"}
MODEL_COLORS = {"mf": None, "ea": "green"}

def branch_dataframe(betas_fine):
    """Branches of both models in the shape plot.plot_bifurcation expects."""
    frames = []
    for model in MODELS:
        rows = {"stable_hi": [], "stable_lo": [], "unstable": []}
        for b in betas_fine:
            hi, lo, un = meanfield.branches(
                [b], model=model, delta=b * DELTA_SCALE)
            rows["stable_hi"].append(hi[0])
            rows["stable_lo"].append(lo[0])
            rows["unstable"].append(un[0])
        for branch, arr in rows.items():
            arr = np.asarray(arr)
            frames.append(pd.DataFrame({
                "model": model, "branch": branch, "beta": betas_fine,
                "theta_empty": arr[:, 0], "theta_co": arr[:, 1],
                "theta_o": arr[:, 2],
            }))
    return pd.concat(frames, ignore_index=True)


def rates_dataframe(beta, theta_o):
    """Fig. 4 rate curves vs theta_CO at fixed beta and theta_O."""
    theta_co = np.linspace(1e-4, 1.0 - theta_o - 1e-4, 200)
    frames = []
    for model in MODELS:
        _, r_des_co, r_ads_o, r_oxi, _ = meanfield.rates(
            theta_co, theta_o, beta, model=model, delta=beta * DELTA_SCALE)
        frames.append(pd.DataFrame({
            "model": model, "beta": beta, "theta_o": theta_o,
            "theta_co": theta_co, "r_oxi": r_oxi, "r_ads_o": r_ads_o,
            "r_des_co": np.broadcast_to(r_des_co, theta_co.shape),
        }))
    return pd.concat(frames, ignore_index=True)


def plot_bifurcation(branches, sweep, path):
    """Fig. 3 style plot: theta_CO and theta_O vs beta."""
    L = int(sweep["L"].iloc[0])
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    for ax, col, ylabel in ((axes[0], "theta_co", r"$\theta_{CO}$"),
                            (axes[1], "theta_o", r"$\theta_O$")):
        for model in ("mf", "ea"):
            mdf = branches[branches["model"] == model]
            label = MODEL_LABELS[model]
            hi = mdf[mdf["branch"] == "stable_hi"].sort_values("beta")
            lo = mdf[mdf["branch"] == "stable_lo"].sort_values("beta")
            un = mdf[mdf["branch"] == "unstable"].sort_values("beta")
            line, = ax.plot(hi["beta"], hi[col], "-", color=MODEL_COLORS[model],
                            label=f"{label} stable")
            ax.plot(lo["beta"], lo[col], "-", color=line.get_color())
            ax.plot(un["beta"], un[col], "--", color=line.get_color(),
                    label=f"{label} unstable")
        key = "co" if col == "theta_co" else "o"
        ax.scatter(sweep["beta"], sweep[f"{key}_full"], marker="o", zorder=5,
                  label="kMC (CO-covered start)")
        ax.scatter(sweep["beta"], sweep[f"{key}_empty"], marker="s", zorder=5,
                  label="kMC (empty start)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=8, ncol=3, loc="lower center",
                  bbox_to_anchor=(0.5, 1.02))
    axes[-1].set_xlabel(r"$\beta$ (O$_2$ impingement rate, s$^{-1}$)")
    fig.suptitle(f"CO oxidation bifurcation diagram (L={L})", y=1.08)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_rates(rates, path):
    """Fig. 4 style plot: log10(rate) vs theta_CO at fixed theta_O and beta."""
    beta = rates["beta"].iloc[0]
    theta_o = rates["theta_o"].iloc[0]
    mf = rates[rates["model"] == "mf"].sort_values("theta_co")
    ea = rates[rates["model"] == "ea"].sort_values("theta_co")
    fig, ax = plt.subplots(figsize=(7, 5))
    for col, label in (("r_oxi", "CO oxidation"), ("r_ads_o", "O2 adsorption"),
                       ("r_des_co", "CO desorption")):
        ax.plot(mf["theta_co"], np.log10(mf[col]), "-", label=f"{label} (MF-MK)")
        ax.plot(ea["theta_co"], np.log10(ea[col]), "--", color="green",
               label=f"{label} (Ea-MK)")
    ax.set_xlabel(r"$\theta_{CO}$")
    ax.set_ylabel("log10(rate)")
    ax.legend(fontsize=8, ncol=3, loc="lower center",
             bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(rf"Rate comparison at $\beta$={beta}, $\theta_O$={theta_o}",
                y=1.1)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(traj, path):
    """Coverage-vs-time trajectories from empty and CO-covered starts."""
    beta = traj["beta"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, init, title in ((axes[0], "empty", "empty start"),
                            (axes[1], "full", "CO-covered start")):
        tdf = traj[traj["init"] == init].sort_values("t")
        ax.plot(tdf["t"], tdf["theta_CO"], label=r"$\theta_{CO}$")
        ax.plot(tdf["t"], tdf["theta_O"], label=r"$\theta_O$")
        ax.plot(tdf["t"], tdf["theta_empty"], label=r"$\theta_*$")
        ax.set_xlabel("t (s)")
        ax.set_title(title)
    axes[0].set_ylabel("coverage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(rf"Coverage trajectories at $\beta$={beta}", y=1.08)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("csv", help="path to a {out}_kmc_sweep.csv file")
    ap.add_argument("--beta-fine-step", type=float, default=0.05,
                    help="beta grid step for the mean-field branches")
    ap.add_argument("--rates-beta", type=float, default=4.0,
                    help="fixed beta for the Fig. 4 rate plot")
    ap.add_argument("--rates-theta-o", type=float, default=0.01,
                    help="fixed theta_O for the Fig. 4 rate plot")
    args = ap.parse_args()

    p = Path(args.csv)
    dir = p.parents[0]

    sweep = pd.read_csv(args.csv).dropna()
    stem = p.stem.replace("_kmc_sweep", "")

    betas_fine = np.arange(sweep["beta"].min(),
                           sweep["beta"].max() + 0.5 * args.beta_fine_step,
                           args.beta_fine_step)
    branches = branch_dataframe(betas_fine)

    plot_bifurcation(branches, sweep, f"{dir}/{stem}_bifurcation.png")
    print(f"wrote {stem}_bifurcation.png")

    rates = rates_dataframe(args.rates_beta, args.rates_theta_o)
    plot_rates(rates, f"{dir}/{stem}_rates.png")
    print(f"wrote {stem}_rates.png")

if __name__ == "__main__":
    main()