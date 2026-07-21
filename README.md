# CO oxidation
Features a number of kMC, mean field, and (soon) ME-MKM kinetic models for CO oxidation following the model of [Tian & Rangarajan, *J. Phys. Chem. C* 2021, 125, 20275](https://doi.org/10.1021/acs.jpcc.1c04495)

## Model
$$\begin{aligned}
    \mathrm{CO(g)} + \ast &\xrightleftarrows[\gamma]{\alpha} \mathrm{CO}^\ast \\
    \mathrm{O_2(g)} + \ast + \ast &\xrightleftarrows[\delta]{\beta} \mathrm{O}^\ast + \mathrm{O}^\ast \\
    \mathrm{CO}^\ast + \mathrm{O}^\ast &\xrightarrow{k_r} \mathrm{CO_2(g)} + \ast + \ast \\
    \mathrm{CO}^\ast + \ast &\xleftrightharpoons[]{k_\text{hop}} \ast + \mathrm{CO}^\ast \\
    \mathrm{O}^\ast + \ast &\xleftrightharpoons[]{k_\text{hop}} \ast + \mathrm{O}^\ast \\
\end{aligned}$$
Note the differences in arrows. The first two equation are competitive rates whereas the last two diffusion equations follow detailed balance. Hence whole system does not obey detailed balence; particularly because of the reaction. The model only differs from Tian & Rangarajan by the addition of a desorption step for $\mathrm{O}^\ast$. The only real reason for this is providing a model that features ergodicity more realistic conditions. In genereal oxygen binds strongly to most catalyst surfaces so $\delta \ll \beta$ and can be set to zero to restore the model of Tian & Rangarajan. Only $\mathrm{CO}^\ast$ has repuslive lateral interactions in this model. For diffusion, `khop = khop_scale * max(beta, alpha)` mimics the fast-diffusion limit (the default is 1000, i.e. three orders of magnitude, per the paper).
## kMC Algorithm
The kinetic Monte Carlo method is a rejection-free n-fold (BKL) following the algorithms reviewed by [Chatterjee & Vlachos, *J. Comput.-Aided Mater. Des.* 2007, 14, 253](https://doi.org/10.1007/s10820-006-9042-9) (n-fold method sec. 6.3, linear search sec. 6.1.1, local updates sec. 6.4) on a periodic square lattice. The n-fold works well here because the only interaction is nearest-neighbour, allowing every event rate to belong to one of 20 discrete classes (neighbour counts 0-4). Each kMC step selects a class by linear search over 20 cumulative weights, then a uniform random member — rejection-free, no null events. After an event, only the events within graph distance 2 of the changed sites are re-classified (local update), so the cost per event is O(1), independent of lattice size. Class membership uses swap-with-last lists for O(1) add/remove.

Versus the previous direct-kMC implementation (full rate-matrix rebuild every event, O(N) twice per event): ~4x faster at L=16, ~15x at L=32, ~60x at L=64.

## Development

I would highly recommed to use `git` when making changes to the project. Install [Git Bash](https://git-scm.com/install/windows) then also [Github Desktop](https://desktop.github.com/download/). Git acts as like a code journal, it comes in handy when dealing with complicated projects where you should be concise of every detail you have made to the project and to easily reverse changes if they dont work. I would also [setup SSH keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) for your desktop and on the cluster so you can pull/push to the code repository without manaully syncing individual files back and forth.

The manual approach is using the command line interface (CLI) such as CMD, GitBash, etc. This is definitly needed on the cluster.
```sh
git clone https://github.com/farinc/kMC_CO_oxidation
git checkout MFPT
```
Github Desktop can do most of the `git` actions, so on your laptop its a bit easier to get started.

If this is something you would rather not do then skip and download the project as a zip file and carry on.

To actually have the intended python enviroment, download and install `uv` [by Astral](https://docs.astral.sh/uv/getting-started/installation/). If you plan to run this on the cluster then install it there too (its a linux machine) for your user. If you have a conda enviroment active I would deactivate it before setup.

```sh 
# Run those to get a more recent clib for Python on the cluster BEFORE running the uv command.
module load gcc/10.3.0
module load openmpi/4.1.2

# installs the package + dev group
uv sync                                        
```

Once that completes your ready to code! IDE's and the code editor VS Code is aware of python enivroments and will activate them for you to run files and code hints related to the dependencies.

## Usage
### On Laptop Development
```sh
uv run kmc-run --L 16 --beta 5.0 --init full   # one kMC run
uv run python -m sweeps.linear                 # default sweep: L=16, beta 0..10 s^{-1}, t_max 30 s
uv run python -m sweeps.linear --L 24 --out case1

uv run pytest                                  # test suite for development purposes
```

`sweeps/linear.py` and `sweeps/mpi.py` write `{out}_kmc_sweep.csv` (default
prefix `co_oxidation`). The mean-field comparison CSVs and bifurcation plot
live in `tests/test_bistability.py`, which uses `tests/support/meanfield.py`
and `tests/support/plot.py`.

Runs stop at `--tmax` (kMC time) or `--max-steps` events, whichever comes first. Steady-state coverages are time-weighted averages over the second half of the run.

Note that `sweeps/linear.py` remains available for a plain single-core run. If done on a laptop limit the sweep to a few beta cause the runs are not run in parallel and take considerable time.

### Running on an HPC cluster

`sweeps/mpi.py` parallelizes the beta sweep's ~52 independent `(beta, (empty, full))` 
configurations using OpenMPI and runs kMC across MPI ranks (round-robin, gathered on rank 0). 
```sh
qsub submit_kmc_sge.sh --L 24 --out case1
```
Everything after the script name on the `qsub` command line is forwarded to `sweeps/mpi.py`. This is considerably faster than using the linear sweep. 

## Using as a Dependency
Since this is a `uv` library you can use this as a dependency in other projects:
```sh
uv init
uv add "co_oxidation @ git+https://github.com/farinc/CO-Oxidation.git"
``` 

Then one can use it like a model
```python
from kmc_co_oxidation import 
```