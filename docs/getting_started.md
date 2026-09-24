# Getting started

## Install

```bash
git clone https://github.com/alecpwills/xcquinox.git
cd xcquinox
pip install .
```

Python 3.12 or newer. `requirements.txt` pins the exact releases the campaigns run on;
`pip install -r requirements.txt -e .` reproduces that environment. The tests run with
`pip install .[test]` and `pytest -m "not slow and not oracle"`.

## A campaign is one configuration file

The cluster harness, `xcquinox.pipeline.cluster`, drives a campaign from one YAML file. The
current campaign is `hpcjobs/configs/dfs_step8.v8_dfs_allsc.yaml`. Its blocks:

- `sweep`: the axes of the grid, one training cell per combination: the architectures, the
  loss, the subset-selection metric, the subset sizes and the solver.
- `solvers`: the named SCF solvers: the mode, the cycle count, the density mixer and the
  weighting of the SCF-trajectory loss.
- `hyperparams`: the optimizer and its schedule, the gradient clip, the channel weights, the
  validation split and early stopping, the checkpoint interval, and the seed axis of the arm
  (`seed_mix_atomic`, `respect_sc_flag`, `nonsc_weight`).
- `inputs`: the basis, the grid level, density fitting, the seed density (`seed_xc`), the
  subset ledger, the reference directories and the output root.
- `pretrain`: the pretraining set (`dfs_set`, `pool_atoms`, `atoms`), the parent density, the
  cloning objective and its schedule, the exchange footing and the data directory.
- `cluster`: the partition, the wall of each stage, the array throttles, the retry
  partitions, the conda environment and the mail directives.
- `model`: the parent anchor, the descriptor coordinates and the uniform-gas gate.
- `fidelity`: the tolerances of the pretraining-fidelity certificate and whether it gates
  training.
- The top-level switches: `domain_profile`, `use_polarized_correlation`, `inline_eval`,
  `eval_coldstart`.

## The job graph

```bash
python -m xcquinox.pipeline.cluster submit hpcjobs/configs/dfs_step8.v8_dfs_allsc.yaml \
    --partition extended-96core --max-nodes 4 --submit
```

`submit` creates a run directory under `inputs.output_root`, renders one sbatch script per
stage and submits the stages with SLURM dependencies:

1. `datagen`, a single job. It writes the pretraining data file under `pretrain.data_dir`
   for the run's identity and leaves an existing file in place when its manifest matches.
2. `pretrain`, an array with one task per distinct architecture, after `datagen`. Each task
   fits the exchange and correlation networks to the parent functional and writes the
   fidelity certificate beside the checkpoint.
3. `preflight`, a single job after the pretrain array: the compile smoke and the cold-start
   convergence census over the training species.
4. `train`, an array with one task per cell, after both the pretrain array and the preflight.
   Each task trains its cell through the differentiable SCF and, under `inline_eval`, runs
   the held-out evaluation at the end of the same task; otherwise an `eval` array follows it,
   task for task.
5. `benchmark_refs`, the reference calculations of the held-out pools.

Without `--submit` the command is a dry run: the run directory and the rendered scripts are
written and nothing is submitted. `--partition` is required. `--max-nodes` caps the nodes an
array stage holds at once.

## Following a run

```bash
python -m xcquinox.pipeline.cluster status <run_dir>
python -m xcquinox.pipeline.cluster resubmit <run_dir> --submit
python -m xcquinox.pipeline.cluster list-runs
```

`status` reads the job records of a run directory. `resubmit` classifies the failed tasks,
out of memory or timeout, and submits them again on the retry partitions the configuration
names. `list-runs` lists the run directories.

## Pulling results

```bash
python -m xcquinox.pipeline.cluster pull auto --category dfs_step8
```

`pull` copies a run's summaries, certificates and per-reaction results from the cluster to the
workstation through the packaged rsync filter.

## The cluster checkout

The cluster runs a git clone of this repository. Code reaches it by `git pull` after a merge,
never by copying files. The environment is `requirements.txt` installed by
`hpcjobs/build_parity_env.sbatch` after conda supplies the interpreter. Every job carries the
mail directives of the `cluster` block.
