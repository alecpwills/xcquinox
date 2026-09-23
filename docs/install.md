# Installation

## From the repository

```bash
git clone https://github.com/alecpwills/xcquinox.git
cd xcquinox
pip install .
```

An editable install (`pip install -e .`) is the usual form for development. Python 3.12 or
newer is required (numpy 2.5 and scipy 1.18 admit nothing older); the tests run on 3.12 and
3.13.

The install brings the runtime stack with it, so nothing else has to be pinned by hand:
`ase`, `equinox`, `jax`, `matplotlib`, `numpy`, `optax`, `pyscf`, `pyscfad`, `pyyaml`,
`scipy` and `tqdm`.

## The versions the results were produced with

The floors in `pyproject.toml` are the pins of `requirements.txt`, the parity file: the
current stack around pyscfad 0.3.4, with jax 0.10.2, pyscf 2.14.0, numpy 2.5.3, scipy
1.18.1, equinox 0.13.8 and optax 0.2.8 on python 3.12, the one coherent set the resolver
reaches from pyscfad 0.3.4 (2026-09-21) and the stack the v8 program runs on. The parity
file pins that set exactly (`pip install -r requirements.txt -e .` reproduces it); an
install from the declaration alone resolves to it today and to whatever newer releases
satisfy the floors later. The v7 campaigns ran on the previous stack, jax 0.7.0, pyscf 2.11.0, pyscfad 0.1.11
and numpy 2.3.

No upper bound is written into the declaration: the floors are one resolved set, and the
bounds a library states about its own dependencies (pyscfad's `jax<0.11`) are the resolver's
to enforce. The two previous caps are gone: pyscf was held below 2.12 because the
closed-shell record's bound had been measured against 2.11.0, and pyscfad
below 0.2 because 0.3.4 no longer carries the `define_xc_` that PySCF's mean-field method
reaches; the development history records the move and what it changed for each.

## Running the tests

```bash
pip install ".[test]"
pytest                     # the whole suite, slow items included
pytest -m "not slow and not oracle"   # the fast loop (the oracle set runs on the cluster per architecture)
```

The test extra adds what the repository's own suites need beyond the runtime stack:
`pytest`, `pytest-cov`, `nbformat`, `nbclient`, `ipykernel` (the kernel the end-to-end notebook
runs execute in), `pandas`, `pillow` and `brokenaxes`. With no
path argument pytest collects the five test directories the packaging file declares: the
package's two test trees, the tools, the notebooks and the cluster jobs. The suite is
the core of the package: one test per behaviour of the live code (about 1,400 tests),
the guards that have fired, and the packaging, environment and
repository rules. Two marked sets stay out of the routine run: `slow` (the end-to-end
notebook executions and the long SCF trainings) and `oracle` (the per-architecture
spin-scaling oracles the cluster's workflow matrix runs one architecture at a time).

One process running the whole fast suite reaches the kernel's memory-mapping ceiling
(`vm.max_map_count`, 65530 on a stock Linux kernel) under jaxlib 0.10.2 and aborts inside
XLA's compiler: the differentiated-SCF tests add mappings a process never gives back. On a
machine where the ceiling can be raised (`sudo sysctl -w vm.max_map_count=262144`, which the
workflow does on its hosted runner) the one-process form above works; elsewhere the suite
runs in ten interpreters, each ending well below the ceiling:

```bash
python -m tools.run_fast_suite            # the fast suite in chunks, one log per chunk
python -m tools.run_fast_suite --dry-run  # the ten command lines
```

The chunks partition the tracked test modules, which `tools/test_run_fast_suite.py` holds; a
test module added to the tree must be assigned to a chunk there.

## The cluster harness

The install provides a console script:

```bash
xcquinox-cluster --help
xcquinox-cluster prepare hpcjobs/configs/dfs_step7.dfs6311_grid3_v7g1_size.yaml
```

The same commands are reachable as `python -m xcquinox.pipeline.cluster`, which is the form
the job scripts use. The cluster side of the workflow, from preparing a run directory to
pulling the results back, is in the [user guide](user_guide.md) and the
[pull-and-figures runbook](pipeline/pull_and_figures.md).

## The cluster environment

The jobs run in the parity environment at
`/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j0102`, built on a short milan queue
by `hpcjobs/build_parity_env.sbatch`: conda supplies python 3.12, pip installs
`requirements.txt` and the package from the checkout, and the job validates by printing every
version, running a pyscfad SCF and importing the package. Every pin is a PyPI wheel on the
nodes' glibc 2.34 (Rocky Linux 9.6, measured 2026-09-22 on the login node and on a compute
node), so nothing is compiled there; the file carries the test extra, so the cluster's
regression job runs in the same environment. The job states no version of its own, the
prefix and the job's steps are held by `hpcjobs/test_build_parity_env.py`, and section 3 of
`hpcjobs/SEAWULF_RUNBOOK.md` walks through the build and its verification.

## Building the documentation

```bash
pip install ".[docs]"
sphinx-build -W -b html docs docs/_build/html
```

`-W` turns warnings into errors, which is how the site is kept from drifting: a page that is
not in the table of contents, or a reference that does not resolve, fails the build.
