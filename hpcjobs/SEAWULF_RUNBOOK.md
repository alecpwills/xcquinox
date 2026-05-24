# SeaWulf step-7 runbook (awills @ FernandezGroup)

Personalized companion to `hpcjobs/README.md`. All paths are resolved for:

| env var | value |
|---|---|
| `$HOME`    | `/gpfs/home/awills` |
| `$GROUP`   | `/gpfs/projects/FernandezGroup` |
| `$SCRATCH` | `/gpfs/scratch/awills` |
| repo root  | `$GROUP/Alec/xcquinox` = `/gpfs/projects/FernandezGroup/Alec/xcquinox` |

The repo has already been rsynced to `$GROUP/Alec/xcquinox` and `~/xcquinox` is a
symlink to that path. This runbook starts from that point.

In bash commands below, `$GROUP` / `$SCRATCH` / `$HOME` are used where convenient.
In `step7.local.yaml`, **always use the canonical absolute path** — env vars are
not expanded inside YAML strings and the harness wants absolute paths that
resolve identically on every node.

---

## 1. Sanity-check the copy

```bash
ls $GROUP/Alec/xcquinox/hpcjobs/configs/step7.yaml
ls $GROUP/Alec/xcquinox/notebooks/checkpoints_step7/subset_index_log.json
ls $GROUP/Alec/xcquinox/notebooks/checkpoints_step7/*/subset.traj | head
ls -ld $SCRATCH    # confirm scratch dir exists and is writable
```

If any of those fail, fix before continuing.

---

## 2. Create the scratch output directories

```bash
mkdir -p $SCRATCH/xcquinox_runs
mkdir -p $SCRATCH/external_refs
mkdir -p $SCRATCH/pretrain
```

---

## 3. Build the conda env in the group dir

`$HOME` is too small. Install the env under `$GROUP` so it's persistent and
visible from every compute node.

```bash
conda create --prefix $GROUP/Alec/conda_envs/xcquinox python=3.11 -y
conda activate $GROUP/Alec/conda_envs/xcquinox
cd $GROUP/Alec/xcquinox
pip install -e .
```

Verify the install and record the conda profile path you'll need for the YAML:

```bash
python -c "import xcquinox; print(xcquinox.__file__)"
# expect: /gpfs/projects/FernandezGroup/Alec/xcquinox/xcquinox/__init__.py
python -c "from xcquinox.alec import cluster; print('cluster ok')"
echo "$(conda info --base)/etc/profile.d/conda.sh"
# ↑ save this output — it goes in cluster.conda_profile in step 5
```

---

## 4. Verify visibility from a compute node

Cheap insurance before queuing 40 jobs:

```bash
salloc -p debug-40core -c 4 --mem=8G -t 00:15:00
# once you land on the compute node:
ls $GROUP/Alec/xcquinox/
ls $SCRATCH/
conda activate $GROUP/Alec/conda_envs/xcquinox
python -c "import xcquinox; from xcquinox.alec import cluster; print('ok')"
exit
```

All four checks should succeed. If the import fails on the compute node but
worked on the login node, fix the env / module-load mismatch now.

---

## 5. Fill in the config

```bash
cd $GROUP/Alec/xcquinox
cp hpcjobs/configs/step7.yaml hpcjobs/configs/step7.local.yaml
$EDITOR hpcjobs/configs/step7.local.yaml
```

Set every `CHANGE_ME` to the values in this table. **Use the absolute paths
exactly as written** — do not use `$GROUP` / `$SCRATCH` env vars in the YAML.

| field | value |
|---|---|
| `cluster.conda_profile` | `/gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh` (user's personal miniconda install under `$GROUP`, confirmed via `conda info --base`) |
| `cluster.conda_env` | `/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox` |
| `cluster.account` | **leave blank** (`""`). SeaWulf routes your jobs to your default Fernandez allocation based on user identity — no explicit `--account=` needed. The harness omits the `#SBATCH --account=` line entirely when this is empty (`submit.py:_optional_sbatch_line`). |
| `cluster.mail_user` | `alec.wills@stonybrook.edu` |
| `cluster.mail_type` | `BEGIN,END,FAIL` (SLURM's keyword for job-start is `BEGIN`, not `START`) |
| `inputs.output_root` | `/gpfs/scratch/awills/xcquinox_runs` |
| `inputs.external_refs_dir` | `/gpfs/scratch/awills/external_refs` |
| `inputs.subset_ledger_path` | `/gpfs/projects/FernandezGroup/Alec/xcquinox/notebooks/checkpoints_step7/subset_index_log.json` |
| `pretrain.data_dir` | `/gpfs/projects/FernandezGroup/Alec/xcquinox/notebooks/checkpoints_step6/pretrain_data` (contains `pretrain_data.npz` — matches step-7 notebook cells 469/552) |
| `pretrain.pretrain_root` | `/gpfs/scratch/awills/pretrain` |

Leave `inputs.basis`, `inputs.grid_level`, the `cluster:` resource defaults, and
the `pretrain:` hyperparameters at template values. The SeaWulf etiquette
throttle (`array_throttle: 4`) is already set in the template.

**Partition is NOT in the config** — it is set on the CLI at submit time via the
required `--partition` flag (and the optional per-stage
`--{train,eval,preflight,pretrain}-partition` overrides). The config's
`partition:` is intentionally empty so a submission never silently lands on a
queue that only exists on one login-node set. See steps 7–8.

> **Login-node ↔ queue coupling.** The `*-96core-shared` queues live on the
> `milan1`/`milan2` SLURM instance; the `*-28core` queues live on
> `login1`/`login2`. `sbatch` only accepts partitions that exist on the instance
> you submit from. So: choose your `--partition` values to match the login node
> you're on. The examples below assume a **milan** login node.

---

## 6. Capture the golden-snapshot test fixture (one-time)

Lock the harness against the current notebook spec layout. After this,
`test_cluster_spec_golden.py` enforces parity instead of skipping.

```bash
cd $GROUP/Alec/xcquinox
python scripts/capture_notebook_spec_snapshot.py
```

---

## 7. Dry-run on the login node

From a **milan** login node (so the `*-96core-shared` queues resolve):

```bash
cd $GROUP/Alec/xcquinox
python -m xcquinox.alec.cluster submit hpcjobs/configs/step7.local.yaml \
    --run-root "$(pwd)/hpcjobs" \
    --partition long-96core-shared \
    --max-nodes 3
```

**Use a long-wall queue for all stages.** The config now sets train, preflight,
and pretrain to an **8 h** wall (a 2 h train wall previously killed 15/40 specs
mid-training). The *short* queues cap at **4 h**, so the whole graph goes on a
queue that allows 8 h — `long-96core-shared` is simplest (`--partition` is the
base for all four stages). On `login1`/`login2`, use `long-28core` instead.
SeaWulf queue max-walls: `short-* 4 h`, `medium-* 12 h`, `long-28core 2 days`,
`long-96core-shared` (longer still). To shorten/lengthen a stage ad-hoc, add
`--time` (all stages) or `--{train,eval,preflight,pretrain}-time`.

**Allocation = whole node per task.** Every stage defaults to `exclusive`
(`#SBATCH --nodes=1 --exclusive`, **no `--mem`**), so each training task owns a
full node's RAM — required since training peaks near 90 GB and 4 sliced tasks
would OOM a 256 GB node. You never set `mem`. `--max-nodes N` is then the number
of *nodes running at once* (it sets the array throttle; 1 task = 1 node);
`--max-nodes 3` keeps you at the shared-queue etiquette cap. Per-stage
`--{train,eval,pretrain}-max-nodes` override it.

Inspect `hpcjobs/runs/run_<UTC-timestamp>/scripts/{pretrain,preflight,train_array,eval_array}.sbatch`:

- No `CHANGE_ME` left anywhere.
- All four scripts on `long-96core-shared` (the single `--partition` base).
- `train_array`/`pretrain`/`preflight` show `#SBATCH --time=08:00:00`;
  `eval_array` shows `02:00:00`.
- Every script carries `#SBATCH --nodes=1` + `#SBATCH --exclusive` and **no**
  `#SBATCH --mem` line (whole-node allocation).
- The train script runs `exec python -m …_train_task …` (the `exec` lets the
  worker receive the wall-clock grace signal so a timeout is recorded + auto-
  recoverable by `resubmit`).
- Train/eval arrays carry `--array=0-N%3` (the `--max-nodes 3` cap).
- Conda activation lines reference `/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox`.

Advisory `UserWarning`s about non-existent paths during a login-node dry-run are
expected — the preflight job is authoritative on the compute side.

A one-line `RuntimeWarning` from `python -m` about `__main__` in `sys.modules` is
a benign CPython quirk — ignore it.

---

## 8. Real submit

```bash
cd $GROUP/Alec/xcquinox
python -m xcquinox.alec.cluster submit hpcjobs/configs/step7.local.yaml --submit \
    --partition long-96core-shared \
    --max-nodes 3
```

The resolved partitions, node caps **and** walls are baked into
`<run_dir>/resolved_config.yaml`, so the recovery commands (`resubmit`,
`resubmit-preflight`) reuse them automatically — you do **not** repeat
`--partition` / `--max-nodes` / `--time` on those.

**Timeout recovery is now automatic.** If a train task still hits its wall, the
worker records a `failure.json{classification: "timeout"}` (via the `exec`'d
SIGTERM handler), and `resubmit … --submit` re-runs it — rerouted to the
`timeout_retry_partition`/`timeout_retry_time` knobs if you set them in the
config (otherwise on the same resources). No more manual `sbatch` overrides.

Prints the run-dir path (under `/gpfs/scratch/awills/xcquinox_runs/runs/run_<UTC-timestamp>/`)
and four job IDs (pretrain, preflight, train, eval). Dependencies are wired
automatically:

```
pretrain → preflight (afterok:pretrain) → train (afterok:pretrain:preflight) → eval (aftercorr:train)
```

Records land in `<run_dir>/jobs.json`. Save the run-dir path for step 9.

---

## 9. Monitor and recover

```bash
RUN_DIR=/gpfs/scratch/awills/xcquinox_runs/runs/run_<UTC-timestamp>   # paste real path

# per-index train/eval status (+ pretrain checkpoint-presence line) + remedy:
python -m xcquinox.alec.cluster status              "$RUN_DIR"

# re-run failed TRAIN tasks (OOM/timeout retried, deterministic skipped):
python -m xcquinox.alec.cluster resubmit            "$RUN_DIR" --submit

# recover a failed PRETRAIN or PREFLIGHT (re-runs the dependent graph):
python -m xcquinox.alec.cluster resubmit-preflight  "$RUN_DIR" --submit

# rebuild a corrupt/missing manifest.json (non-destructive):
python -m xcquinox.alec.cluster repair-manifest     "$RUN_DIR"
```

Standard SLURM commands:

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

---

## Run-dir contents (for reference)

```
/gpfs/scratch/awills/xcquinox_runs/runs/run_<UTC-timestamp>/
  resolved_config.yaml                                 the exact resolved config (provenance)
  scripts/{pretrain,preflight,train_array,eval_array}.sbatch
  specs/spec_0000.spec …                               one materialized TrainingSpec per array task (preflight writes)
  manifest.json                                        idx → GridCell → spec file → hash (preflight writes)
  checkpoints/spec_0000/ …                             model.eqx, losses, eval/, eval_df.csv (train/eval write)
  jobs.json                                            submitted-job records (pretrain/preflight/train/eval)
  logs/{pretrain,preflight,train,eval}_*.out
  submit_commands.txt                                  every sbatch invocation, timestamped
```

Pretrained `xnet.eqx`/`cnet.eqx` pairs live at `/gpfs/scratch/awills/pretrain/<arch>/`,
not under the run dir.

---

## Resolved values (no remaining unknowns)

- **`pretrain.data_dir`**: `/gpfs/projects/FernandezGroup/Alec/xcquinox/notebooks/checkpoints_step6/pretrain_data`
  — directory holding `pretrain_data.npz`. Matches what the step-7 notebook uses
  (cells 469 and 552 of `gga_training_example-step7.ipynb`), which is what the
  pretrain stage's `run_pretrain` loads via
  `os.path.join(spec.data_dir, "pretrain_data.npz")` (see
  `xcquinox/alec/pretrain.py:225`). Bundled with the rsync; verify on cluster:
  `ls $GROUP/Alec/xcquinox/notebooks/checkpoints_step6/pretrain_data/pretrain_data.npz`.
- **`cluster.account`**: leave blank. The user confirmed they have never needed
  to pass `--account=` in their normal `sbatch` workflow on SeaWulf — their
  default allocation (Fernandez group) is bound to the `awills` user identity
  and is selected automatically. The harness already treats `account` as
  optional (`ClusterResources.account` defaults to `""` in
  `xcquinox/alec/cluster/grid_config.py:203`; the `_optional_sbatch_line` helper
  in `submit.py:81` omits the `#SBATCH --account=` directive entirely when the
  field is blank), so a blank value produces a fully valid sbatch script.
