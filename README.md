# kMC CO oxidation

Rejection-free **n-fold (BKL) kinetic Monte Carlo** for CO oxidation on a
periodic square lattice, following the model of
[Tian & Rangarajan, *J. Phys. Chem. C* 2021, 125, 20275](https://doi.org/10.1021/acs.jpcc.1c04495)
and the algorithms reviewed by
[Chatterjee & Vlachos, *J. Comput.-Aided Mater. Des.* 2007, 14, 253](https://doi.org/10.1007/s10820-006-9042-9)
(n-fold method sec. 6.3, linear search sec. 6.1.1, local updates sec. 6.4).

## Model

Site states: empty, CO\*, O\*. Elementary steps: CO adsorption/desorption,
dissociative O2 adsorption on adjacent empty pairs, CO\*+O\* -> CO2, and fast
CO\*/O\* nearest-neighbour hops. The only lateral interaction is a CO-CO
nearest-neighbour repulsion (eps = 8.368 kJ/mol) entering rates through BEP
relations (omega = 1 for desorption/reaction, 1/2 for hops). The system is
bistable in the O2 impingement rate beta, which mean-field kinetics misses.

## Algorithm

Because the only interaction is a nearest-neighbour pair term, every event
rate belongs to one of **20 discrete classes** (neighbour counts 0-4). Each
kMC step selects a class by linear search over 20 cumulative weights, then a
uniform random member — rejection-free, no null events. After an event, only
the events within graph distance 2 of the changed sites are re-classified
(**local update**), so the cost per event is O(1), independent of lattice
size. Class membership uses swap-with-last lists for O(1) add/remove.

Versus the previous direct-kMC implementation (full rate-matrix rebuild every
event, O(N) twice per event): ~4x faster at L=16, ~15x at L=32, ~60x at L=64,
with a flat ~4.5e5 events/s on one core.

## Layout

| Path | Contents |
|---|---|
| `src/kmc_co_oxidation/kmc.py` | numba n-fold kMC core + `KMCParams` / `run_kmc` API |
| `src/kmc_co_oxidation/cli.py` | single-run CLI (`kmc-run` entry point): one `(L, beta, init, seed, ...)` -> one run |
| `sweeps/linear.py` | serial beta x `{empty, full}` sweep, writes `{out}_kmc_sweep.csv` |
| `sweeps/mpi.py` | same sweep, MPI-parallel (mpi4py); what `jobs/submit_kmc_sge.sh` launches |
| `tests/support/meanfield.py` | MF steady states via one quartic + `np.roots` (all branches, stability from the Jacobian), RK4 transients; dev-only, used by `tests/test_bistability.py` |
| `tests/support/plot.py` | reads the sweep/MF CSVs and renders the PNG figures; dev-only, no simulation code |
| `tests/test_kmc.py` | rate oracle, local-update vs full-rebuild invariant, Langmuir limit, single-beta bistability, MF checks |
| `tests/test_bistability.py` | dev-gated: reproduces the paper's Fig. 3 bifurcation diagram across a beta sweep, checks it against mean-field, renders an inspection figure |
| `jobs/submit_kmc_sge.sh` | SGE + OpenMPI submission script for `sweeps/mpi.py` |

## Usage

```sh
uv sync                                    # installs the package + dev group
uv run pytest                              # test suite (incl. bistability reproduction)
uv run kmc-run --L 16 --beta 5.0 --init full   # one kMC run
uv run python -m sweeps.linear             # default sweep: L=16, beta 0..10, t_max 30 s -> CSV
uv run python -m sweeps.linear --L 24 --out case1
```

`sweeps/linear.py` and `sweeps/mpi.py` write `{out}_kmc_sweep.csv` (default
prefix `co_oxidation`). The mean-field comparison CSVs and bifurcation plot
live in `tests/test_bistability.py`, which uses `tests/support/meanfield.py`
and `tests/support/plot.py`.

Runs stop at `--tmax` (kMC time) or `--max-steps` events, whichever comes
first. Steady-state coverages are time-weighted averages over the second half
of the run. `khop = khop_scale * max(beta, alpha)` mimics the fast-diffusion
limit (default 1000, i.e. three orders of magnitude, per the paper).

```python
from kmc_co_oxidation import run_kmc
res = run_kmc(beta=5.0, init="full", L=32, t_max=30.0, seed=1)
print(res.steady_co, res.steady_o, res.steps)
```

Note: a lattice with no CO and no empty sites is an absorbing state
(O-poisoned, total rate zero); `KMCResult.stuck` flags it.

## Running on an HPC cluster (SGE + OpenMPI)

`sweeps/mpi.py` parallelizes the beta sweep's ~52 independent `(beta, init)`
kMC runs across MPI ranks (round-robin, gathered on rank 0). One-time setup
and submission:

```sh
module load openmpi        # whatever module exposes OpenMPI on your site
uv sync --extra mpi         # builds .venv, compiles mpi4py against that OpenMPI
mkdir -p logs
qsub jobs/submit_kmc_sge.sh --L 24 --out case1
```

Edit the `-pe <PE_NAME> <N>` line in `jobs/submit_kmc_sge.sh` before
submitting -- the parallel-environment name is cluster-specific (commonly
`orte`, `mpi` or `openmpi`; check with `qconf -spl`). Everything after the
script name on the `qsub` command line is forwarded to `sweeps/mpi.py`.
`sweeps/linear.py` remains available for a plain single-core run (no
OpenMPI/mpi4py needed).
