"""
Plot: render figures from the CSVs written by the beta sweeps and dev tests.

Reads {out}_meanfield_rates.csv, {out}_meanfield_branches.csv and
{out}_kmc_sweep.csv (and {out}_trajectory.csv if present) and draws the
Fig. 3 / Fig. 4 style plots of Tian & Rangarajan (2021). Deliberately kept
free of any kMC/mean-field computation -- restyling a figure never requires
rerunning a simulation. Dev-only tooling, used by tests/test_bistability.py.

Usage:
    uv run python -m tests.support.plot --out co_oxidation
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

MODEL_LABELS = {"mf": "MF-MK", "ea": "Ea-MK"}
MODEL_COLORS = {"mf": None, "ea": "green"}


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
    ap.add_argument("--out", default="co_oxidation",
                    help="input/output file prefix, must match the data run")
    args = ap.parse_args()

    rates = pd.read_csv(f"{args.out}_meanfield_rates.csv")
    plot_rates(rates, f"{args.out}_rates.png")

    branches = pd.read_csv(f"{args.out}_meanfield_branches.csv")
    sweep = pd.read_csv(f"{args.out}_kmc_sweep.csv")
    plot_bifurcation(branches, sweep, f"{args.out}_bifurcation.png")

    traj_path = Path(f"{args.out}_trajectory.csv")
    if traj_path.exists():
        plot_trajectory(pd.read_csv(traj_path), f"{args.out}_trajectory.png")


if __name__ == "__main__":
    main()
