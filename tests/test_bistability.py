"""
Test to make sure the bifurcation diagram (Tian & Rangarajan 2021, Fig. 3)
can be reproduced. Runs the full default beta sweep (same range as the
paper reproduction: beta 0..10, L=16), compares the kMC branches against
the mean-field steady states, and renders an inspection figure. Generalizes
test_kmc.py's single-beta test_bistability_branches into the full beta-
dependent transition. Both the serial (sweeps.linear) and MPI (sweeps.mpi)
drivers are used in the test.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from kmc_co_oxidation import KMCParams
from sweeps._common import build_betas
from tests.support import meanfield, plot

BETA_MIN, BETA_MAX, BETA_STEP = 0.0, 10.0, 0.4
L = 16
SEED = 100
OUTPUT_DIR = Path("tests/outputs")


def _params():
    return KMCParams(L=L, t_max=30.0, khop_scale=1000.0, sample_interval=10_000)


def _branch_dataframe(betas_fine, hi, lo):
    frames = []
    for branch, arr in (("stable_hi", hi), ("stable_lo", lo)):
        frames.append(pd.DataFrame({
            "model": "mf", "branch": branch, "beta": betas_fine,
            "theta_empty": arr[:, 0], "theta_co": arr[:, 1], "theta_o": arr[:, 2],
        }))
    return pd.concat(frames, ignore_index=True)


def _check_and_plot(betas, sweep, figure_name):
    # Inside the bistable window the two initial conditions must diverge:
    # full-CO start stays CO-rich (branch I), empty start stays CO-poor
    # (branch II).
    mid = int(np.argmin(np.abs(betas - 4.0)))
    assert sweep["co_full"][mid] > sweep["co_empty"][mid] + 0.2

    betas_fine = build_betas(BETA_MIN, BETA_MAX, 0.05)
    hi, lo, _ = meanfield.branches(betas_fine, alpha=1.6, gamma=1e-3, kr=1.0,
                                   delta=0.0, model="mf")
    assert np.any(~np.isnan(hi[:, 1]))
    assert np.any(~np.isnan(lo[:, 1]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    branches_df = _branch_dataframe(betas_fine, hi, lo)
    sweep_df = pd.DataFrame({"beta": betas, "L": L, **sweep})
    figure_path = OUTPUT_DIR / figure_name
    plot.plot_bifurcation(branches_df, sweep_df, figure_path)
    assert figure_path.exists()


def test_bistability_transition_linear():
    from sweeps import linear

    betas = build_betas(BETA_MIN, BETA_MAX, BETA_STEP)
    sweep = linear.run_sweep(betas, _params(), SEED)
    _check_and_plot(betas, sweep, "bifurcation_linear.png")


def test_bistability_transition_mpi():
    pytest.importorskip("mpi4py")
    from sweeps import mpi

    betas = build_betas(BETA_MIN, BETA_MAX, BETA_STEP)
    sweep = mpi.run_sweep(betas, _params(), SEED)
    _check_and_plot(betas, sweep, "bifurcation_mpi.png")
