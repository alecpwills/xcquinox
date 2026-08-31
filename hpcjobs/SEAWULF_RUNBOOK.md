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
In `step7.local.yaml`, **always use the canonical absolute path** -- env vars are
not expanded inside YAML strings and the harness wants absolute paths that
resolve identically on every node.

---

## 1. Sanity-check the copy

```bash
ls $GROUP/Alec/xcquinox/hpcjobs/configs/step7.yaml
# C4-03: the ledger now lives under an alpha-mode subdir. Pick the mode you
# intend to train (alpha_off = the GGA-faithful selection that drops the
# meta-GGA tau descriptor; alpha_on = the default alpha-weighted selection):
ls $GROUP/Alec/xcquinox/notebooks/checkpoints_step7/alpha_off/subset_index_log.json
ls $GROUP/Alec/xcquinox/notebooks/checkpoints_step7/alpha_off/*/subset.traj | head
ls -ld $SCRATCH    # confirm scratch dir exists and is writable
```

If any of those fail, fix before continuing.

> **C4-03 provenance.** The step-7 notebook regenerates the ledger in two
> modes via `STEP7_IGNORE_ALPHA`: `alpha_on` (default, `descriptor_weights=None`)
> and `alpha_off` (`descriptor_weights={'alpha': 0.0}`), each under its own
> `checkpoints_step7/<mode>/` root so both coexist. Set
> `inputs.subset_ledger_path` to the mode you are training. Since the GGA
> network never consumes alpha, `alpha_off` matches the descriptors the model
> can actually see; `alpha_on` is kept for the comparison experiment.

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
# ↑ save this output -- it goes in cluster.conda_profile in step 5
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
exactly as written** -- do not use `$GROUP` / `$SCRATCH` env vars in the YAML.

| field | value |
|---|---|
| `cluster.conda_profile` | `/gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh` (the miniconda install under `$GROUP`, confirmed via `conda info --base`) |
| `cluster.conda_env` | `/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox` |
| `cluster.account` | **leave blank** (`""`). SeaWulf routes your jobs to your default Fernandez allocation based on user identity -- no explicit `--account=` needed. The harness omits the `#SBATCH --account=` line entirely when this is empty (`submit.py:_optional_sbatch_line`). |
| `cluster.mail_user` | `alec.wills@stonybrook.edu` |
| `cluster.mail_type` | `BEGIN,END,FAIL` (SLURM's keyword for job-start is `BEGIN`, not `START`) |
| `inputs.output_root` | `/gpfs/scratch/awills/xcquinox_runs` |
| `inputs.external_refs_dir` | `/gpfs/scratch/awills/external_refs` |
| `inputs.subset_ledger_path` | `/gpfs/projects/FernandezGroup/Alec/xcquinox/notebooks/checkpoints_step7/<alpha_on\|alpha_off>/subset_index_log.json` (C4-03: choose the alpha-mode subdir you are training) |
| `pretrain.data_dir` | `/gpfs/projects/FernandezGroup/Alec/xcquinox/notebooks/checkpoints_step6/pretrain_data` (contains `pretrain_data.npz` -- matches step-7 notebook cells 469/552) |
| `pretrain.pretrain_root` | `/gpfs/scratch/awills/pretrain` |

Leave `inputs.basis`, `inputs.grid_level`, the `cluster:` resource defaults, and
the `pretrain:` hyperparameters at template values. The SeaWulf etiquette
throttle (`array_throttle: 4`) is already set in the template.

**Partition is NOT in the config** -- it is set on the CLI at submit time via the
required `--partition` flag (and the optional per-stage
`--{train,eval,preflight,pretrain}-partition` overrides). The config's
`partition:` is intentionally empty so a submission never silently lands on a
queue that only exists on one login-node set. See steps 7-8.

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
queue that allows 8 h -- `long-96core-shared` is simplest (`--partition` is the
base for all four stages). On `login1`/`login2`, use `long-28core` instead.
SeaWulf queue max-walls: `short-* 4 h`, `medium-* 12 h`, `long-28core 2 days`; every `long-*` QOS (40- and 96-core, shared included) caps MaxWall at 48 h, and the `extended-*` partitions carry 7-day caps (sacctmgr, 2026-08-27). To shorten/lengthen a stage ad-hoc, add
`--time` (all stages) or `--{train,eval,preflight,pretrain}-time`.

**Allocation = whole node per task.** Every stage defaults to `exclusive`
(`#SBATCH --nodes=1 --exclusive`, **no `--mem`**), so each training task owns a
full node's RAM -- required since training peaks near 90 GB and 4 sliced tasks
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
- The train script runs `exec python -m ..._train_task ...` (the `exec` lets the
  worker receive the wall-clock grace signal so a timeout is recorded + auto-
  recoverable by `resubmit`).
- Train/eval arrays carry `--array=0-N%3` (the `--max-nodes 3` cap).
- Conda activation lines reference `/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox`.

Advisory `UserWarning`s about non-existent paths during a login-node dry-run are
expected -- the preflight job is authoritative on the compute side.

A one-line `RuntimeWarning` from `python -m` about `__main__` in `sys.modules` is
a benign CPython quirk -- ignore it.

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
`resubmit-preflight`) reuse them automatically -- you do **not** repeat
`--partition` / `--max-nodes` / `--time` on those.

**Timeout recovery is now automatic.** If a train task still hits its wall, the
worker records a `failure.json{classification: "timeout"}` (via the `exec`'d
SIGTERM handler), and `resubmit ... --submit` re-runs it -- rerouted to the
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

## 10. Pull results back to your laptop

> **dfs_step7 figures:** for the end-to-end "pull a training + regenerate the
> figures" recipe (exact `pull` categories per run + the `make_ablation_arch_figure.py
> --suite` command), see [`notebooks/analysis/RUNBOOK_pull_and_figures.md`](../notebooks/analysis/RUNBOOK_pull_and_figures.md).

The `pull` subcommand wraps `rsync` with a packaged filter that knows the
harness layout. The `list-runs` subcommand discovers what's on the cluster.

> **Migration note.** Pre-2026-05-29, `XCQUINOX_CLUSTER_REMOTE_ROOT`
> defaulted to `/gpfs/scratch/awills/xcquinox_runs/runs` (a specific
> single-series subdir that doesn't actually exist in the canonical layout).
> The default is now `/gpfs/scratch/awills/xcquinox_runs` -- the base scratch
> directory -- and a new `--category` flag selects which experiment-series
> subdir (`alpha_off/runs`, `polarized/alpha_on`, ...) the run dirs live in.
> If your shell currently has `export XCQUINOX_CLUSTER_REMOTE_ROOT=.../runs`,
> drop the `/runs` tail and add `--category runs` to your pull invocations.

### Layout this assumes

```
$XCQUINOX_CLUSTER_REMOTE_ROOT/    # /gpfs/scratch/awills/xcquinox_runs by default
  alpha_off/runs/run_<UTC>Z/...
  alpha_on/runs/run_<UTC>Z/...
  polarized/alpha_on/run_<UTC>Z/...
  polarized/alpha_off/run_<UTC>Z/...
```

`pull --category <segment>` joins the segment onto the remote root and
expects `run_<UTC>Z` dirs directly inside. The local destination mirrors the
same category layout so two different categories with the same stamp cannot
collide.

### Two profiles

- **`summaries`** (default) -- manifest, resolved config, every per-spec
  `eval_df.csv` / `failure.json` / `losses.npy` / `eval/per_molecule.json`,
  pretrain metadata + loss curves + fidelity certificate, and (since
  2026-08-30) the network weights the figures forward-evaluate: per spec
  `model.eqx` / `model_val_best.eqx` with their `.class.json` records, per
  arch `xnet.eqx` / `cnet.eqx` and the val-best pair under `xnet/` / `cnet/`.
  **No `logs/`**, and of the `*.eqx` tier no `model_best.eqx`, no
  `resume_*.eqx`, no `*.gen<N>`, no pretrain `xc.eqx.<step>` snapshots.
  Typically **< 100 MB / 40-spec run** (the weights add a few MB: the
  checkpoints measured here run 11.8-129 KB each), so this stays the right
  default for driving `analyze.collect_results()`, the notebook figures and
  the enhancement-factor figures.
- **`full`** -- mirrors the run dir, `logs/` included. **~11-12 MB / 40-spec
  run** of artifacts for the current `deep_combined_attn` arch (each
  `model.eqx` is ~126 KB; xnet+cnet pretrain nets ~64 KB each) plus the log
  tree. Use this when you need the excluded tier -- `model_best.eqx`, the
  resume set, the pretrain trajectory snapshots -- or the SLURM logs to
  diagnose a failed run off-cluster. The full profile is small enough that
  surgical `--specs` filtering is rarely necessary; pull whole categories in
  seconds.

### One-time setup

```bash
# ~/.ssh/config -- alias + ControlMaster so repeat rsyncs do not re-handshake
cat >> ~/.ssh/config <<'EOF'
Host seawulf
    HostName login.seawulf.stonybrook.edu
    User awills
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
EOF

# ~/.bashrc -- optional, overrides the built-in defaults
export XCQUINOX_CLUSTER_HOST=seawulf
# (XCQUINOX_CLUSTER_REMOTE_ROOT default = /gpfs/scratch/awills/xcquinox_runs)
# Set if you want a sticky default category for this shell:
# export XCQUINOX_CLUSTER_CATEGORY=alpha_off/runs
# Set if you want results staged somewhere other than the default:
# export XCQUINOX_CLUSTER_LOCAL_ROOT=$HOME/Documents/Research/xcquinox-results/runs
```

Bash aliases like `alias seawulf="ssh awills@login2.seawulf.stonybrook.edu"`
are **not** usable for the `XCQUINOX_CLUSTER_HOST` value -- the harness
`subprocess.run(["ssh", host, ...])` bypasses bash so aliases are invisible.
Use an `~/.ssh/config` Host alias (above) or set the env var to a literal
`user@hostname` string (e.g. your existing `$swpath`).

### Discover what's on the cluster

```bash
python -m xcquinox.alec.cluster list-runs
# Example output:
#   remote_root: /gpfs/scratch/awills/xcquinox_runs (host=seawulf)
#
#   alpha_off/runs/  (3 runs)
#     run_20260528T143052Z
#     run_20260530T100000Z
#     run_20260601T120000Z   <- latest
#
#   alpha_on/runs/  (1 run)
#     run_20260529T090000Z   <- latest
#
#   polarized/alpha_on/  (1 run)
#     run_20260527T143052Z   <- latest

# Tune the search depth (default 5 -- sufficient for the current layout, where
# the deepest run lives at polarized/<axis>/runs/run_<UTC>Z = depth 4). Bump
# if you nest categories further:
python -m xcquinox.alec.cluster list-runs --depth 7
```

### Daily commands

```bash
# Latest sweep in the alpha_off series, summaries only, < 100 MB
python -m xcquinox.alec.cluster pull latest --category alpha_off/runs

# Same for alpha_on or polarized:
python -m xcquinox.alec.cluster pull latest --category alpha_on/runs
python -m xcquinox.alec.cluster pull latest --category polarized/alpha_on

# A specific run by stamp:
python -m xcquinox.alec.cluster pull run_20260528T143052Z --category alpha_off/runs

# Preview without transferring:
python -m xcquinox.alec.cluster pull latest --category alpha_off/runs --dry-run

# When you need the actual trained models for re-evaluation:
python -m xcquinox.alec.cluster pull latest --category alpha_off/runs --profile full

# Then analyze locally -- the local tree mirrors the category:
python -m xcquinox.alec.cluster results \
    ~/Documents/Research/xcquinox-results/runs/alpha_off/runs/run_20260601T120000Z
```

`pull latest` resolves the newest `run_<UTC>Z` under
`<remote-root>/<category>/` via `ssh <host> ls -1tr` (filtering out stray
non-run-dir entries). The summaries profile is small enough that you can
re-run `pull latest --category ...` ad hoc to refresh as eval tasks finish --
rsync's `--partial` flag makes resumes cheap.

### Why error output is clean

SeaWulf's SSH server prints a multi-line compliance banner to stderr on
every connection. On a failing command (e.g. wrong `--remote-root`), the
harness used to dump the banner *and* the real error together; now the
error formatter shows only the last 3 non-blank lines of stderr -- the real
`ls:`/`find:` error always lives at the tail, so the banner gets dropped
from view without anything important being suppressed.

### Where the filter rules live

The filter files at `xcquinox/alec/cluster/filters/{summaries,full}.filter`
are exercised on every test run by the canary tests in
`xcquinox/alec/tests/test_cluster_sync.py`. Any future change to the
harness output layout that forgets to update the filter (or the category
plumbing) fails CI loudly instead of silently dropping artifacts at pull
time.

### 10.5 Local test-set re-evaluation -- because cluster eval is in-sample only

> **Known harness gap.** The cluster eval array (`_eval_one_spec.py`)
> evaluates every trained network **only on the molecules it was trained
> on**: `_eval_one_spec.py:282` calls `build_test_spec(...)` with no
> `holdout_molecules`, and `spec_builder.py:545-556` then silently defaults
> to `eval_molecules = training_spec.molecules` while emitting a
> `RuntimeWarning` to eval-job stderr. The grid-config schema
> (`grid_config.py:InputPaths`) has **no field** for a held-out pool, so
> the operator cannot opt in from the YAML. **Every `eval_df.csv` row is
> labeled `set=training_subset`; there is no `test_set` row anywhere.**
>
> This is therefore a measure of **training fit quality**, not held-out
> generalization. Large MAE in those rows indicates **training failure**
> (the optimizer couldn't fit even its own training data), not poor
> generalization. The proper harness fix -- adding `holdout_molecules` to
> the schema and plumbing it through to the eval worker -- is tracked as a
> future PR; meanwhile, the recommended workflow is:

#### Workflow

1. **Identify which specs are worth re-evaluating** using
   `make_cluster_pulls_figure.py`'s Fig 1 (training diagnostics):

   ```bash
   python notebooks/analysis/make_cluster_pulls_figure.py
   ```

   The right panel of Fig 1 is a scatter of training-subset MAE vs final
   training loss. Specs in the **lower-left** cluster (low loss, low MAE)
   converged cleanly and are the ones worth re-evaluating on a held-out
   pool. Specs in the shaded **upper-right failure region** (MAE > 5
   kcal/mol or final loss > 5e-3) were not adequately trained -- local
   re-eval would just confirm "the network can't represent anything", so
   skip them.

2. **Pull the checkpoints.** With the current `deep_combined_attn` arch
   `model.eqx` files at ~126 KB each, a full-mirror pull across all 3 ready
   categories is only **~34 MB** -- small enough to pull everything in one
   shell loop:

   ```bash
   python -m xcquinox.alec.cluster pull auto --days 0 --profile full --yes
   ```

   (`pull auto` discovers every run in one ssh shot and pulls the selection
   in one rsync over the same authenticated connection; `--days 0` lifts the
   30-day activity horizon and `--yes` confirms a >15-run batch. The
   per-category loop it replaces:)

   ```bash
   for cat in alpha_on/runs alpha_off/runs polarized/alpha_off/runs; do
     python -m xcquinox.alec.cluster pull latest --category "$cat" --profile full
   done
   ```

   If you want surgical control (e.g. for a future larger arch where this
   becomes >GB scale), pass `--specs N,M,K` to filter:

   ```bash
   python -m xcquinox.alec.cluster pull latest \
       --category alpha_on/runs --profile full --specs 0,1,21,5,9
   ```

   `--specs` accepts a comma-separated list of integer indices. Combined
   with `--profile full` (which keeps `*.eqx` files), only
   `checkpoints/spec_{0000,0001,0021,0005,0009}/model.eqx` will land
   locally, alongside the manifest, resolved config, and pretrain
   checkpoint.

   Use `--dry-run` to preview the rsync argv and estimated transfer
   size:

   ```bash
   python -m xcquinox.alec.cluster pull latest --category alpha_on/runs \
       --profile full --specs 0,1,21,5,9 --dry-run
   ```

3. **Run the local re-eval script.** The recommended invocation
   auto-discovers every pulled category under `--local-root` and runs
   `model.eqx`-bearing specs across all of them in a single process so the
   PBE precompute amortizes:

   ```bash
   python notebooks/analysis/local_reeval.py --auto
   ```

   Per-spec failures (NaN convergence, etc.) are logged but do not abort
   the batch -- a final summary table reports `n_ok / n_total` per
   category. To background a long run:

   ```bash
   nohup python notebooks/analysis/local_reeval.py --auto \
       > /tmp/local_reeval.log 2>&1 &
   ```

   You can still drive a single run dir + specific specs if you prefer:

   ```bash
   python notebooks/analysis/local_reeval.py \
       ~/Documents/Research/xcquinox-results/runs/alpha_on/runs/run_<UTC>Z \
       --specs 0,1,21
   ```

   Defaults: held-out pool = **BH76 + W4-11 combined**; **loose mode**
   (every reaction is kept; any training-set overlap is recorded in the
   output `note` column). Rationale: H is in every training set as a
   Dick regularization anchor -- dropping every reaction with H overlap
   would discard the *entire* BH76 pool. Similarly, when H2O appears in
   the training set (e.g. spec_0 with subset_size=1), evaluating its
   atomization energy AE(H2O) = E(H2O) − 2·E(H) − E(O) is exactly the
   verification we want (does the model recover the right AE?), not a
   contaminated metric. Pass `--strict` to opt into the old
   strictly-disjoint behavior if you ever need it.

   The script writes two files per spec, sitting alongside the cluster's
   eval outputs so the figure pipeline can consume them later:

   - `<run_dir>/checkpoints/spec_<NNNN>/local_test_set.csv` -- one row per
     pool (BH76, W4-11) plus a combined `held_out_combined` row, with
     columns `set, mae_nn_kcalmol, mae_pbe_kcalmol, delta_nn_minus_pbe,
     n_reactions, n_dropped_overlap, note`. The PBE MAE is computed on the
     SAME reactions using the by-product `E_pbe` from each species'
     precompute (~free) -- gives a direct apples-to-apples NN-vs-PBE
     comparison on the curated subset. The PBE numbers should reproduce
     the published values (BH76 ≈ 8.08, W4-11 ≈ 10.45 kcal/mol on this
     curation); they're a sanity-check on the pool builders.
   - `<run_dir>/checkpoints/spec_<NNNN>/eval/local_per_molecule.json` --
     one record per pool species (including those in the training set,
     flagged via `from_training_subset: bool` so downstream plotters can
     split). Schema-compatible with the cluster's `per_molecule.json`.
   - `<run_dir>/checkpoints/spec_<NNNN>/eval/local_per_reaction.json` --
     one record per held-out reaction (16: 6 BH76 + 10 W4-11) carrying
     paired NN/PBE predicted ΔE and absolute errors plus an
     `in_sample_overlap` list of any training-set species the reaction
     touches. Drives the per-reaction figures (per-pool breakdown,
     NN−PBE grid heatmap, per-reaction ranking).
   - `<run_dir>/checkpoints/spec_<NNNN>/eval/local_subset_descriptors.json`
     -- per-molecule grid-weighted means of every DMStatistics + Cusp
     descriptor feature column across the spec's training subset, plus
     summary stats (mean/std/min/max/range across the subset). Written
     by `python notebooks/analysis/extract_subset_descriptors.py --auto`
     (~3 min, ~30 unique training molecules precomputed once). Drives
     Fig 10 (descriptor range vs held-out accuracy) and Fig 15 (per-
     subset descriptor distributions, colored by subset_size, overlaid
     against the largest-subset reference, separated by metric).

When ≥2 categories are populated, `make_cluster_pulls_figure.py` also
renders **Fig 16 -- cross-category NN vs PBE**: strip plot of per-spec
held-out NN MAE per (category × metric × solver) plus a stacked histogram
of NN−PBE delta per category. Use this to read off the
alpha-mode and polarization effects at a glance once
`polarized/alpha_on` finishes on the cluster.

   The expensive part is **`fixed_density_total_energy` per molecule**
   (one PBE SCF + grid build + NN forward). On a laptop this is ~few
   seconds per molecule; a ~24-species held-out pool re-eval per spec is
   minutes, not hours.

   **Polarized networks auto-detected.** `local_reeval.py` reads
   `training_spec.arch.use_polarized_correlation` and prints
   `[polarized (UKS for open-shell)]` vs `[unpolarized (RKS)]` at model
   load time. The pool builders set every open-shell atom's
   `MoleculeSpec.spin` to its NIST ground-state value, so
   `precompute_fixed_density_data` runs the right PBE branch automatically
   and `fixed_density_total_energy` routes through `split_exc_energy_uks`
   for polarized models. **No flag needed when you re-eval the
   polarized/* checkpoints** -- once they finish on the cluster, the same
   `--auto` invocation evaluates them with the correct UKS path.

#### Long-term fix (future work)

A follow-up harness PR should:

- Add an `inputs.holdout_molecules_path` field to the grid config schema
  (`xcquinox/alec/cluster/grid_config.py:InputPaths`).
- Pipe it through `build_test_spec` so the eval worker writes a SECOND
  row (`set=test_set`) to `eval_df.csv` whenever holdout is configured.
- Promote the silent `RuntimeWarning` to a hard error at config-submit
  time (in `cmd_submit`) so future operators cannot accidentally run an
  in-sample-only sweep.

Until that PR lands, the local-reeval workflow above is the source of
truth for held-out MAE numbers in any paper/report.

---

## Run-dir contents (for reference)

```
/gpfs/scratch/awills/xcquinox_runs/runs/run_<UTC-timestamp>/
  resolved_config.yaml                                 the exact resolved config (provenance)
  scripts/{pretrain,preflight,train_array,eval_array}.sbatch
  specs/spec_0000.spec ...                               one materialized TrainingSpec per array task (preflight writes)
  manifest.json                                        idx → GridCell → spec file → hash (preflight writes)
  checkpoints/spec_0000/ ...                             model.eqx, losses, eval/, eval_df.csv (train/eval write)
  jobs.json                                            submitted-job records (pretrain/preflight/train/eval)
  logs/{pretrain,preflight,train,eval}_*.out
  submit_commands.txt                                  every sbatch invocation, timestamped
```

Pretrained `xnet.eqx`/`cnet.eqx` pairs live at `/gpfs/scratch/awills/pretrain/<run_id>/<arch>/`,
not under the run dir.

---

## Resolved values (no remaining unknowns)

- **`pretrain.data_dir`**: `/gpfs/projects/FernandezGroup/Alec/xcquinox/notebooks/checkpoints_step6/pretrain_data`
  -- directory holding `pretrain_data.npz`. Matches what the step-7 notebook uses
  (cells 469 and 552 of `gga_training_example-step7.ipynb`), which is what the
  pretrain stage's `run_pretrain` loads via
  `os.path.join(spec.data_dir, "pretrain_data.npz")` (see
  `xcquinox/alec/pretrain.py:225`). Bundled with the rsync; verify on cluster:
  `ls $GROUP/Alec/xcquinox/notebooks/checkpoints_step6/pretrain_data/pretrain_data.npz`.
- **`cluster.account`**: leave blank. This account has not been needed
  to pass `--account=` in the normal `sbatch` workflow on SeaWulf -- the
  default allocation (Fernandez group) is bound to the `awills` user identity
  and is selected automatically. The harness already treats `account` as
  optional (`ClusterResources.account` defaults to `""` in
  `xcquinox/alec/cluster/grid_config.py:203`; the `_optional_sbatch_line` helper
  in `submit.py:81` omits the `#SBATCH --account=` directive entirely when the
  field is blank), so a blank value produces a fully valid sbatch script.
