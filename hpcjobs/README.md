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

The harness submits the step-7 grid as a 3-stage SLURM job graph:
**preflight** (build inputs + materialize specs) → **train array** (one task per
spec, `afterok` on preflight) → **eval array** (`aftercorr` on the train array).

---

## 0. Prerequisites

- The `xcquinox` package importable in a conda env (the env you train in).
- To submit for real: a clone of this repo **on the SLURM cluster** (e.g. SeaWulf)
  and a shell on a **login node** — `sbatch` only exists there. You can dry-run
  anywhere.

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
| `inputs.pretrain_checkpoint` | shared dir holding the pretrained xnet/cnet (the step-6 pretrain output). Must be staged before submitting. |
| `inputs.external_refs_dir`, `inputs.descriptor_cache`, `inputs.refhist_cache`, `inputs.subset_ledger_path` | shared input-cache locations. In the **default (regenerate) mode** the preflight job *creates* these — point them at writable shared-FS paths; they do not need to pre-exist. |

Every `inputs.*` path must be **absolute** and on a filesystem that resolves
identically on the login node and every compute node.

**SeaWulf resource defaults** are already set in the `cluster:` block
(`short-96core-shared`, `-c 24`, `--mem=96G`, `--time=02:00:00`,
`array_throttle: 4` — the etiquette cap; preflight on `long-96core-shared`).
Adjust if your queues differ.

### regenerate vs. reuse

The default config has no `mode` key → the preflight runs in **regenerate**
mode: it does the CCSD/OEP precompute, descriptor extraction, histogram build,
and subset selection itself, on a compute node. You only need to pre-stage
`pretrain_checkpoint`. (To instead reuse pre-computed inputs, stage
`external_refs/`, the caches, and the ledger on shared storage and add a
top-level `mode: reuse` to the config — the preflight then validates rather
than regenerates.)

---

## 2. Dry-run locally to sanity-check (no cluster needed)

```bash
python -m xcquinox.alec.cluster submit hpcjobs/configs/step7.local.yaml \
    --run-root "$(pwd)/hpcjobs"
```

This loads + validates the config, creates `hpcjobs/runs/run_<UTC-timestamp>/`,
renders the three `scripts/*.sbatch`, and **prints the `sbatch` commands it
*would* run** — it makes no SLURM call. Inspect
`hpcjobs/runs/run_<ts>/scripts/` to confirm the rendered scripts look right.
Advisory `UserWarning`s about non-existent `/gpfs/...` paths are expected when
those paths only exist on the cluster — the preflight job is authoritative.

(A one-line `RuntimeWarning` from `python -m` about `__main__` in `sys.modules`
is a benign CPython quirk — ignore it.)

---

## 3. Submit on the cluster

On a SeaWulf **login node**, from the repo root, with the `xcquinox` env active:

```bash
python -m xcquinox.alec.cluster submit hpcjobs/configs/step7.local.yaml --submit
```

Without `--run-root` the run dir is created under `inputs.output_root`. `submit`
prints the run-dir path and the three submitted job IDs. **`--submit` is
required for a real submission — without it you get a dry-run.** The harness
submits preflight → train (`afterok`) → eval (`aftercorr`) and records the job
IDs in `<run_dir>/jobs.json`.

The run directory ends up containing:

```
<output_root>/runs/run_<UTC-timestamp>/
  resolved_config.yaml      the exact resolved config (provenance)
  scripts/{preflight,train_array,eval_array}.sbatch
  specs/spec_0000.spec …    one materialized TrainingSpec per array task   (preflight writes)
  manifest.json             idx → GridCell → spec file → hash               (preflight writes)
  subset_ledger.json        provenance copy of the canonical ledger         (preflight writes)
  checkpoints/spec_0000/ …  model.eqx, losses, eval/, eval_df.csv           (train/eval write)
  jobs.json                 submitted-job records
  logs/{preflight,train,eval}_*.out
  submit_commands.txt       every sbatch invocation, timestamped
```

---

## 4. Monitor and recover

All commands take the run-dir path. Recovery commands default to dry-run; pass
`--submit` to act for real.

```bash
# Per-index train/eval status + an actionable remedy line:
python -m xcquinox.alec.cluster status        <run_dir>

# Re-run failed TRAIN tasks (OOM/timeout retried, deterministic skipped):
python -m xcquinox.alec.cluster resubmit      <run_dir> --submit

# Recover a failed/timed-out PREFLIGHT (re-runs the whole graph):
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
- **`prepare` subcommand.** `python -m xcquinox.alec.cluster prepare <grid>
  [--regenerate]` stages inputs without SLURM; regenerate mode refuses to run on
  a login node (heavy CCSD) — use `submit` (whose preflight runs it on a compute
  node) or an interactive `salloc`.
