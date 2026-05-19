# hpcjobs — running the `xcquinox.alec.cluster` HPC training harness

This directory is a **job-staging area** for the SLURM training harness
(`xcquinox/alec/cluster/`). It is deliberately kept *outside* the importable
`xcquinox/` package.

```
hpcjobs/
  README.md            this file
  .gitignore           keeps run outputs + filled-in configs out of git
  configs/
    step7.yaml          copy-me template for the step-7 40-spec grid
  runs/                 default local output_root (run dirs land in runs/run_<ts>/)
    .gitkeep
```

The harness submits the step-7 grid as a **4-stage SLURM job graph**:

1. **pretrain array** — one task per *distinct architecture*, submitted up
   front. Each task builds a `PretrainSpec` and writes an `xnet.eqx` +
   `cnet.eqx` pair into `<pretrain_root>/<arch>/`.
2. **preflight** (`afterok:pretrain`) — a single job that (re)computes the
   per-species CCSD external refs (skip-if-cached) and materializes one
   `TrainingSpec` per grid cell.
3. **train array** (`afterok:pretrain:preflight` — gated on *both*) — one task
   per spec; each task uses `<pretrain_root>/<arch>/` as its pretrained
   checkpoint.
4. **eval array** (`aftercorr:train`) — one task per spec, paired index-for-
   index with the train array.

---

## 0. Prerequisites

- The `xcquinox` package importable in a conda env (the env you train in).
- To submit for real: a clone of this repo **on the SLURM cluster** (e.g. SeaWulf)
  and a shell on a **login node** — `sbatch` only exists there. You can dry-run
  anywhere.
- **The subset ledger must be pre-staged.** The harness *consumes* the existing
  `subset_index_log.json` (see "How the harness uses the subset ledger" below);
  it does not create it. Stage a copy of
  `notebooks/checkpoints_step7/subset_index_log.json` — and the per-spec
  `subset.traj` files alongside it — onto shared cluster storage and point
  `inputs.subset_ledger_path` at it before submitting.
- **The pretrain `data_dir` must be pre-staged.** The pretrain stage builds its
  `PretrainSpec` from the training data under `pretrain.data_dir`; that
  directory must exist on shared storage before submitting.

---

## 1. Copy the config and fill in the placeholders

`configs/step7.yaml` is a template. Copy it to a personal, git-ignored file:

```bash
cp hpcjobs/configs/step7.yaml hpcjobs/configs/step7.local.yaml
```

(`*.local.yaml` is git-ignored — see `.gitignore`.) Then edit it: every value
marked `CHANGE_ME` must be replaced. They are:

| field | what it is |
|---|---|
| `cluster.conda_profile` | absolute path to the conda profile script the job `source`s, e.g. `…/miniconda3/etc/profile.d/conda.sh`. Leave empty only if `conda` is already on `PATH`. |
| `cluster.conda_env` | the conda env name to `conda activate` (default `xcquinox`). |
| `cluster.account` | your SLURM allocation/account name. |
| `cluster.mail_user` / `cluster.mail_type` | your email + SLURM mail events (`NONE`/`BEGIN`/`END`/`FAIL`/`ALL`). |
| `inputs.output_root` | run-output root. **Must be on shared scratch** (`/gpfs/scratch/...`), never `$HOME` and never inside the repo — run dirs hold large `model.eqx` checkpoints. |
| `inputs.external_refs_dir` | shared dir of per-species CCSD `<name>.npz` reference files. These ARE (re)computable by the preflight (skip-if-cached) — point it at a writable shared-FS path; it does not need to pre-exist. |
| `inputs.subset_ledger_path` | shared path to the **existing** `subset_index_log.json` from the offline subset-selection pre-process. A **consumed, read-only input** — it must already exist on shared storage when you submit (see below). |
| `inputs.basis` / `inputs.grid_level` | step-7 physical constants (basis set, DFT grid level). Leave at the template values. |
| `pretrain.data_dir` | shared dir of training data the pretrain stage builds its `PretrainSpec` from. Must be staged before submitting. |
| `pretrain.pretrain_root` | shared root the pretrain stage writes each arch's `xnet.eqx`/`cnet.eqx` into (`<pretrain_root>/<arch>/`). A harness **product**, not a pre-staged input — point it at a writable shared-FS path. |

The `pretrain:` section also carries the pretraining hyperparameters
(`n_steps`, `lr_start`, `lr_end`, `lr_decay_start`, `grad_clip`, `seed`,
`loss_weighting`); the template defaults mirror the step-7 notebook's
pretraining cell — leave them unless you have a reason to change them.

Every `inputs.*` and `pretrain.*` path must be **absolute** and on a filesystem
that resolves identically on the login node and every compute node.

**SeaWulf resource defaults** are already set in the `cluster:` block
(`short-96core-shared`, `-c 24`, `--mem=96G`, `--time=02:00:00`,
`array_throttle: 4` — the etiquette cap; preflight + pretrain on
`long-96core-shared`). The `cluster:` block also carries optional **pretrain
resource knobs** — `pretrain_partition`, `pretrain_time`, `pretrain_mem`,
`pretrain_cpus_per_task`, `pretrain_throttle`. Each falls back to the
train-array resource when unset; `pretrain_throttle` unset means every distinct
architecture pretrains concurrently. Adjust if your queues differ.

### How the harness uses the subset ledger

Subset selection is **not** a harness step — it is a finished offline
pre-process. The harness **consumes** the existing ledger
(`subset_index_log.json`) plus the per-spec `subset.traj` files alongside it. It
does **not** run subset selection, descriptor extraction, or reference-histogram
building, and there are no descriptor/refhist caches to stage.

The only thing the preflight job *computes* is the per-species CCSD external
refs (skip-if-cached into `inputs.external_refs_dir`); it then materializes one
`TrainingSpec` per grid cell from the consumed ledger. The subset ledger
therefore must already exist on shared storage before you submit.

---

## 2. Dry-run locally to sanity-check (no cluster needed)

```bash
python -m xcquinox.alec.cluster submit hpcjobs/configs/step7.local.yaml \
    --run-root "$(pwd)/hpcjobs"
```

This loads + validates the config, creates `hpcjobs/runs/run_<UTC-timestamp>/`,
renders the four `scripts/*.sbatch` files (pretrain, preflight, train, eval),
and **prints the `sbatch` commands it *would* run** — it makes no SLURM call.
Inspect `hpcjobs/runs/run_<ts>/scripts/` to confirm the rendered scripts look
right. Advisory `UserWarning`s about non-existent `/gpfs/...` (or `CHANGE_ME`)
paths are expected when those paths only exist on the cluster — the preflight
job is authoritative.

(A one-line `RuntimeWarning` from `python -m` about `__main__` in `sys.modules`
is a benign CPython quirk — ignore it.)

---

## 3. Submit on the cluster

On a SeaWulf **login node**, from the repo root, with the `xcquinox` env active:

```bash
python -m xcquinox.alec.cluster submit hpcjobs/configs/step7.local.yaml --submit
```

Without `--run-root` the run dir is created under `inputs.output_root`. `submit`
prints the run-dir path and the four submitted job IDs. **`--submit` is
required for a real submission — without it you get a dry-run.** The harness
submits pretrain → preflight (`afterok:pretrain`) → train
(`afterok:pretrain:preflight`) → eval (`aftercorr:train`) and records the job
IDs in `<run_dir>/jobs.json`.

The run directory ends up containing:

```
<output_root>/runs/run_<UTC-timestamp>/
  resolved_config.yaml      the exact resolved config (provenance)
  scripts/{pretrain,preflight,train_array,eval_array}.sbatch
  specs/spec_0000.spec …    one materialized TrainingSpec per array task   (preflight writes)
  manifest.json             idx → GridCell → spec file → hash               (preflight writes)
  checkpoints/spec_0000/ …  model.eqx, losses, eval/, eval_df.csv           (train/eval write)
  jobs.json                 submitted-job records (pretrain/preflight/train/eval)
  logs/{pretrain,preflight,train,eval}_*.out
  submit_commands.txt       every sbatch invocation, timestamped
```

(The pretrained `xnet.eqx`/`cnet.eqx` pairs are written by the pretrain stage
into `<pretrain_root>/<arch>/`, not into the run dir.)

---

## 4. Monitor and recover

All commands take the run-dir path. Recovery commands default to dry-run; pass
`--submit` to act for real.

```bash
# Per-index train/eval status (+ a pretrain checkpoint-presence line) and an
# actionable remedy line:
python -m xcquinox.alec.cluster status        <run_dir>

# Re-run failed TRAIN tasks (OOM/timeout retried, deterministic skipped):
python -m xcquinox.alec.cluster resubmit      <run_dir> --submit

# Recover a failed/timed-out PRETRAIN or PREFLIGHT (re-runs the whole graph):
python -m xcquinox.alec.cluster resubmit-preflight <run_dir> --submit

# Rebuild a corrupt/missing manifest.json (non-destructive):
python -m xcquinox.alec.cluster repair-manifest    <run_dir>
```

---

## Notes

- **`output_root` on scratch, not the repo.** Putting run dirs in `$HOME` or the
  repo will blow quotas and slow networked-FS directory walks. The local
  `hpcjobs/runs/` is for dry-runs and local smoke tests only.
- **BH76 mode.** `bh76_mode: reaction_energy` (the default) is fully functional.
  `barrier_height` is wired but gated — it needs transition-state geometries
  staged in `dfs_pool.py` and will raise a clear error until then.
- **Golden faithfulness test.** `test_cluster_spec_golden.py` skips until you
  capture the snapshot once: `python scripts/capture_notebook_spec_snapshot.py`.
- **`prepare` subcommand.** `python -m xcquinox.alec.cluster prepare <grid>`
  stages inputs without SLURM: it builds the training-point pool, validates the
  existing subset ledger, and pre-warms the per-species CCSD external refs
  (skip-if-cached). The CCSD `precompute_all` is heavy, so `prepare` refuses to
  run it on a login node — use `submit` (whose preflight runs the precompute on
  a compute node) or an interactive `salloc`. Pass `--no-recompute-refs` to skip
  the precompute and make `prepare` a light ledger-only check that is safe on a
  login node.
