# Runbook: pull cluster trainings + regenerate figures (dfs_step7)

**Purpose:** a step-by-step, copy-paste runbook -- you can follow it mechanically. Run the blocks in order; each step says how to verify it
worked before moving on. The only step that can require a *code* change is the
`ARCH_ORDER` guard (Step 2, "If it fails") -- that one, ask the author.

All commands run **locally** (your laptop/workstation), in the conda env that has
`xcquinox` importable (e.g. `conda activate xcq`), from the repo root
`~/Documents/Research/xcquinox`. The pull uses `ssh`/`rsync` to SeaWulf; you do
**not** log in to the cluster for any of this.

---

## TL;DR (the whole thing)

```bash
conda activate xcq
cd ~/Documents/Research/xcquinox
export XCQUINOX_CLUSTER_HOST=seawulf          # your ~/.ssh/config Host alias

# 1. PULL the runs you want figures for (latest run per category):
python -m xcquinox.alec.cluster pull latest --category dfs_step7/svp_grid2_v3/runs
python -m xcquinox.alec.cluster pull latest --category dfs_step7/svp_grid2_v3_full25/runs
python -m xcquinox.alec.cluster pull latest --category dfs_step7/dfs6311_grid3_v3/runs

# 2. REGENERATE figures (per-run sets + cross comparison):
python notebooks/analysis/make_ablation_arch_figure.py --suite \
    --domain dfs_step7 --bases svp_grid2_v3,svp_grid2_v3_full25,dfs6311_grid3_v3 \
    --outroot notebooks/analysis
# -> notebooks/analysis/figures_dfs_step7_svp_v3/  (+ _val_best)
#    notebooks/analysis/figures_dfs_step7_svp_v3_full25/  (+ _val_best)
#    notebooks/analysis/figures_dfs_step7_dfs6311_grid3_v3/  (+ _val_best)
#    notebooks/analysis/figures_dfs_step7_basis_comparison/  (+ _val_best)
```

That's the common case. Details, other comparisons, verification, and
troubleshooting below.

---

## 0. One-time local setup

1. **conda env** with `xcquinox` importable: `conda activate xcq` (the same env
   you run the tests in). Verify: `python -c "import xcquinox; print('ok')"`.
2. **ssh alias** (so rsync reuses one connection). In `~/.ssh/config`:
   ```
   Host seawulf
       HostName login.seawulf.stonybrook.edu
       User awills
       ControlMaster auto
       ControlPath ~/.ssh/cm-%r@%h:%p
       ControlPersist 10m
   ```
   then `export XCQUINOX_CLUSTER_HOST=seawulf` (add to `~/.bashrc`). If you skip
   this, pass `--host login.seawulf.stonybrook.edu` to every `pull`/`list-runs`.
3. **Defaults you do NOT need to set** (the harness already uses them):
   - remote scratch root `XCQUINOX_CLUSTER_REMOTE_ROOT` = `/gpfs/scratch/awills/xcquinox_runs`
   - local results root `XCQUINOX_CLUSTER_LOCAL_ROOT` = `~/Documents/Research/xcquinox-results/runs`
   The figure script's `--results-root` defaults to that SAME local root, so a
   pull lands exactly where the figures look. Keep them aligned.

---

## 1. Pull the updated trainings (cluster -> local)

### The runs and their `--category` (subdir under the remote scratch root)

| Run (what it is) | `--category` | local landing dir | figure basis alias |
|---|---|---|---|
| **v3** 3x16 + adamw-WD + val/early-stop, full_3 SCF (the decoupling A/B) | `dfs_step7/svp_grid2_v3/runs` | `.../runs/dfs_step7/svp_grid2_v3/runs/run_*` | `svp_v3` |
| **full25** same as v3 but 25-cycle SCF (ceiling fix) | `dfs_step7/svp_grid2_v3_full25/runs` | `.../dfs_step7/svp_grid2_v3_full25/runs/run_*` | `svp_v3_full25` |
| **4x32 baseline** (the over-capacity reference, `run_20260611T022820Z`) | `dfs_step7/svp_grid2/runs` | `.../dfs_step7/svp_grid2/runs/run_*` | `svp` |
| **tzvpd+DF** (def2-tzvpd + density fitting) | `dfs_step7/tzvpd_grid2_df/runs` | `.../dfs_step7/tzvpd_grid2_df/runs/run_*` | `tzvpd_df` |
| **dfs6311 v3** 6-311++G(3df,2pd) grid-3 + DF (the DFS-paper basis, production) | `dfs_step7/dfs6311_grid3_v3/runs` | `.../dfs_step7/dfs6311_grid3_v3/runs/run_*` | `dfs6311_grid3_v3` |

> The "basis" is just the directory name under `dfs_step7/`. The figure script
> turns it into an output-dir alias by deleting `_grid2` (`svp_grid2_v3` ->
> `svp_v3`); `_grid3` names pass through unchanged (`dfs6311_grid3_v3` keeps its
> full name, matching the on-disk `figures_dfs_step7_dfs6311_grid3_v3*` dirs).
> Pull whichever runs you want to plot.

### Pull command

```bash
python -m xcquinox.alec.cluster pull latest --category <CATEGORY>
```
- `latest` resolves to the newest `run_<UTC>Z` under that category on the cluster.
  To pull a specific run, replace `latest` with the run id, e.g.
  `pull run_20260622T111908Z --category dfs_step7/svp_grid2_v3/runs`.
- **Profile (default is correct for figures):** the default `--profile summaries`
  pulls all the JSON/npy the figures read -- `eval_holdout/**`,
  `eval_holdout_best/**`, `eval_holdout_val_best/**`, `eval/per_molecule.json`, `losses.npy`,
  `train_metadata.json`, `resolved_config.yaml` -- and **skips** the big
  `model.eqx` weights. Only add `--profile full` if you need the weights for a
  LOCAL re-eval (Step 3); figures do **not** need it.
- Add `--dry-run` first if you want to see what rsync would transfer.

### See what's on the cluster before pulling

```bash
python -m xcquinox.alec.cluster list-runs --category dfs_step7/svp_grid2_v3/runs
```

### Verify the pull worked (figures need held-out eval JSON)

```bash
ls ~/Documents/Research/xcquinox-results/runs/dfs_step7/svp_grid2_v3/runs/*/checkpoints/spec_*/eval_holdout/per_reaction.json | head
```
If that lists files, you have held-out coverage and figures will render. If it's
empty, the run hasn't produced held-out eval yet (train_eval still running, or
you pulled too early) -- wait and re-pull. (A still-training run pulls fine; the
figures just draw the not-yet-evaluated cells as hatched/incomplete, never crash.)

---

## 2. Regenerate the figures

One command renders **every figure family** for the runs you name. It reads only
the pulled JSON -- no SCF, no model weights, fast (seconds-minutes).

```bash
python notebooks/analysis/make_ablation_arch_figure.py --suite \
    --domain dfs_step7 \
    --bases <comma-separated basis subdirs> \
    --outroot notebooks/analysis
```
- `--results-root` defaults to `~/Documents/Research/xcquinox-results/runs` (where
  pulls land) -- usually omit it.
- Each `--bases` entry must already be pulled (Step 1), else you get
  `FileNotFoundError: no run_* dir under ...`.
- It uses the **newest** pulled `run_*` per basis.

### What it writes (under `--outroot`, i.e. `notebooks/analysis/`)

Per basis (two parallel sets -- final-checkpoint and val-best):
- `figures_dfs_step7_<alias>/`           -- final-step eval (`eval_holdout/`)
- `figures_dfs_step7_<alias>_val_best/`  -- val-best eval (`eval_holdout_val_best/`, the held-out-validation-best checkpoint), only if that data was pulled

Cross comparison (only when **>= 2** bases are given AND both have eval coverage):
- `figures_dfs_step7_basis_comparison/`       (+ `_val_best`)

Each per-basis dir contains: the arch-aware ablation set (parity, MAE-by-arch,
arch×subset heatmap, MAE-vs-subset), held-out energy/density figures, the five
parity-layout variants, per-run size-consistency / training-loss diagnostics,
and the DFS Eq. 21 combined energy-density figure + per-cell CSV
(`ablation_combined_energy_density.png` / `.csv`; held-out only, rendered when
the pulled `eval_holdout*/per_molecule.json` carries the NN + PBE density
columns -- skipped with a console note otherwise, in which case a stale file
from an earlier render may persist).

Each per-basis dir also gets two overview composites plus a standalone density
trend: `ablation_density_energy_overview.png` (per-pool + 2-subset WTMAD-2
bars over the NN-vs-PBE density parity, the iso-ED decomposition, and the ED
headline; rendered whenever the held-out density figure renders, with
placeholder panels when the ED anchors are missing),
`ablation_holdout_density_per_arch.png` (the per-arch held-out density trend
vs subset_size as its own figure; same gate), and
`ablation_insample_overview.png` (in-sample AE + density; always rendered;
final-checkpoint data, so its panels are identical in the final and val-best
dirs). The per-channel 3x3 `ablation_density_energy_3x3.png` rides the same
held-out-density gate (WTMAD-2 / density parity / ED as columns
BH76 | W4-11 | combined, each channel's ED gamma self-calibrated from its own
PBE anchors, with `ablation_density_energy_3x3.csv` alongside), and the
enriched combined-channel standalone `ablation_ed_decomposition.png` (iso-ED
contour family, beats-PBE shading, per-arch subset trajectories) rides the
stricter ED-anchor gate of the ED figure. The held-out figures carry
a dataset footer line stating what the held-out eval is (live name-dedup
reaction counts per pool + density species coverage); the energy figures
carry the reactions clause too, and the full-pool PBE/SCAN baselines in the
grey footers are labeled as full-pool. Every label on the density/energy
figures and every ED CSV column is decoded in `README_density_figures.md`,
and the held-out set's exact constituents (every test/validation reaction by
name, the density species, the atoms skipped) are enumerated in
`HOLDOUT_SET.md` (both in this directory).

### The specific comparisons you'll want

```bash
# (a) the two new clean runs (v3 full_3 vs full25 25-cycle SCF):
python notebooks/analysis/make_ablation_arch_figure.py --suite \
    --domain dfs_step7 --bases svp_grid2_v3,svp_grid2_v3_full25 --outroot notebooks/analysis

# (b) the decoupling A/B (3x16 v3 vs the 4x32 baseline) -- pull svp_grid2 first:
python notebooks/analysis/make_ablation_arch_figure.py --suite \
    --domain dfs_step7 --bases svp_grid2,svp_grid2_v3 --outroot notebooks/analysis

# (c) everything you've pulled, all per-run sets in one call:
python notebooks/analysis/make_ablation_arch_figure.py --suite \
    --domain dfs_step7 --bases svp_grid2,svp_grid2_v3,svp_grid2_v3_full25,tzvpd_grid2_df \
    --outroot notebooks/analysis
```

> NOTE: per-basis dirs are uniquely named (by alias) so they accumulate across
> calls, but `figures_dfs_step7_basis_comparison/` is a **shared** dir -- the most
> recent `--suite` call's comparison overwrites it. For a focused 2-way
> comparison figure, run that pair on its own (as in (a)/(b)).

> CONVENIENCE: `notebooks/analysis/regen_dfs_step7_basis_comparison.py` is a thin
> guarded launcher for the svp-vs-tzvpd comparison specifically -- it refuses with
> a clear message until BOTH `svp_grid2` and `tzvpd_grid2_df` are pulled, then
> calls the command above. Use it only for that svp-vs-tzvpd pair.

### If it fails

- **`FileNotFoundError: no run_* dir under .../<basis>/runs`** -- you named a basis
  you haven't pulled. Pull it (Step 1) or drop it from `--bases`.
- **`... has archs not in ARCH_ORDER [..]`** (raises, does not draw) -- a run
  contains an architecture the figures don't know how to order/color. This needs
  a **code change**: add the arch name to `ARCH_ORDER` (and a color in
  `ARCH_COLOR`) near the top of `make_ablation_arch_figure.py`. The `deep_*_3x16`
  twins are already registered; you'd only hit this for a brand-new arch. If
  unsure, ask the author rather than guess.
- **"only one basis with eval coverage -- skipping the basis-comparison set"** --
  expected when you pass one basis (or the 2nd basis has no `eval_holdout/` yet).
  Per-run figures still render; pull the other basis to get the comparison.

---

## 3. (Optional) Local re-eval -- only if you need to re-score weights locally

The figures above use the **cluster-side** held-out eval (`eval_holdout/`), so you
normally never re-eval locally. If you do need to (e.g. re-score `model.eqx` with
a changed eval setting):
1. Pull with weights: `pull <run> --category <cat> --profile full` (adds
   `model.eqx`; large). Narrow to specific specs with `--specs 0,3,7`.
2. Run `python notebooks/analysis/reeval_holdout_fixed.py ...` (see its `--help`).
   This is CPU-heavy (runs SCF) -- prefer a background run, and it does not need a code change.

---

## Quick reference

| Task | Command |
|---|---|
| list cluster runs | `python -m xcquinox.alec.cluster list-runs --category dfs_step7/<basis>/runs` |
| pull (figures) | `python -m xcquinox.alec.cluster pull latest --category dfs_step7/<basis>/runs` |
| pull (weights too) | `... pull latest --category ... --profile full` |
| verify pull | `ls .../runs/dfs_step7/<basis>/runs/*/checkpoints/spec_*/eval_holdout/per_reaction.json` |
| figures | `python notebooks/analysis/make_ablation_arch_figure.py --suite --domain dfs_step7 --bases <...> --outroot notebooks/analysis` |

Figures are regenerated artifacts -- the `figures_*` dirs are not version
controlled. Re-run the suite any time after a fresh pull.
