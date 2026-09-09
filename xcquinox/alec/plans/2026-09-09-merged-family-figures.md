# Plan: one figure per quantity over every finished cell, named by what the architecture is (2026-09-09)

## Why

The three v7 groups (G1 "size", G2a "GGA families", G2 "meta-GGA families") are cluster runs,
not physical axes. The registry (`xcquinox/alec/config.py:508, 573-576`) builds `medium` and
`deep_3x16` as the same network (depth 3, width 16, the same inputs, the same parameter count);
the only setting that acts on `deep_3x16` is `zero_init_final_layer`, which zeroes the last
layer's weight and bias at construction so that pre-training starts from the LDA ($F = 1$)
instead of a Glorot initialization (`networks.py:287-295, 560-570`). Nothing is frozen and
nothing is appended; after pre-training both are PBE clones that differ only in their fitted
weights, and their held-out verdicts agree at every shared subset size. Every figure, table and
document that shows "medium" and "deep_3x16" as two architectures in two groups therefore
shows one architecture twice, and the names say nothing about the difference that exists.

Decisions taken (2026-09-09): the names say what the network is; the run split disappears from
the figures and the documents; every quantity is drawn once with every finished cell on it; new
runs (the meta-GGA group, the 25-cycle and dpyscf-parity arms, later campaigns) enter the same
merged view by an entry in one list, never by a new figure set.

## Naming

Storage keys (registry entries, run directories, `train_metadata.json`, the running arms) do
not change: the arms train `medium` on the cluster now and every pulled result is filed under
the stored key. A display layer maps the stored key to the name shown, and registry aliases let
new configuration files use the shown names directly.

| stored key | displayed name | expanded key (legend and table text) |
|---|---|---|
| medium | deep_3x16 | 3 x 16, Glorot initialization |
| medium_attn | deep_attn_3x16 | 3 x 16, 4 attention heads, Glorot initialization |
| deep_3x16 | deep0_3x16 | 3 x 16, last layer zeroed: pre-training starts at the LDA |
| deep_attn_3x16 | deep0_attn_3x16 | 3 x 16, 4 heads, last layer zeroed |
| deep_cusp_3x16 | deep0_cusp_3x16 | 3 x 16, cusp descriptors, last layer zeroed |
| deep_cusp_mgga_3x16 | deep0_cusp_mgga_3x16 | meta-GGA (SCAN parent), cusp, last layer zeroed |
| deep_mgga_3x16, deep_mgga_attn_3x16, deep_rung35_mgga_3x16, deep_rung35ms_mgga_3x16 | deep0_ + the rest | meta-GGA variants, last layer zeroed |
| shallow | shallow_2x8 | 2 x 8, Glorot initialization |
| shallow_attn | shallow_attn_2x8 | 2 x 8, 2 heads, Glorot initialization |

`deep0` marks the zero-initialized last layer (the pre-training starts from the LDA); `deep`
without it is the default (Glorot start). The expanded key is printed in every legend and
table caption, so a reader never has to know the map.

A run may carry a protocol tag that is appended to the displayed name of its cells
(`deep_3x16 [25 cycles]`, `deep_3x16 [dpyscf parity]`) so that the arms, which train the
same architecture under a different protocol, are distinguishable on the same axis.

## Item 1: arch display names

Files: `notebooks/analysis/arch_style.py` (the single source of truth for order and colour),
`xcquinox/alec/config.py` (aliases), the suite and the standalone figure scripts (labels),
their tests, `README_density_figures.md`, `RUNBOOK_pull_and_figures.md`.

- `arch_style.py`: `DISPLAY_NAME[stored] -> shown`, `EXPANDED_KEY[shown] -> text`,
  `display_name(stored, protocol=None)`, `expanded_key(stored)`; `ARCH_ORDER` and
  `ARCH_COLOR` keyed by the stored key as now, with the order rewritten around the real axes
  (initialization, attention, cusp, rung); a stored key without a display entry is shown as
  is and listed by a test as a gap.
- `config.py`: `ARCH_ALIASES = {"deep_3x16_glorot": "medium", ...}` is wrong (it would
  collide with the existing key `deep_3x16`); the aliases are therefore the shown names that
  are NOT existing keys (`deep0_3x16`, `deep0_attn_3x16`, `deep0_cusp_3x16`,
  `deep0_cusp_mgga_3x16`, ..., `shallow_2x8`, `shallow_attn_2x8`) resolving to their stored
  configuration, and the two names that collide (`deep_3x16`, `deep_attn_3x16`) keep resolving
  to the zero-init configurations until the next campaign's registry, which the plan of that
  campaign renames outright; `lookup_architecture(name)` resolves aliases; `describe()` records
  both names.
- Every label the suite writes (legends, footers, titles, CSV `arch` column keeps the stored
  key and gains `arch_display`), `enhancement_factors.py`, `trained_fx_fc.py`,
  `pretrain_fx_fc.py` (whose titles also change from "pretrained corrections" to "pre-trained
  networks against the parent: differences"), `plot_pretraining_curves.py`,
  `plot_certificate_summary.py`, `plot_subset_jsd.py` (no arch labels), the tail tables.
- Tests: the map covers every registry key; `display_name` round-trips; every figure label
  in a rendered fixture set uses the displayed name and the expanded key; the alias resolves
  to the same `ArchitectureConfig`; the holdover guard refuses a bare stored key in a legend.
- Regenerate the tracked figure sets (they are replaced by the merged set in item 2, so this
  regeneration is the per-run sets only for the record of the rename).

## Item 2: merged family view

Files: `notebooks/analysis/merge_family_runs.py` (new, from `merge_v4_arms.py`),
`notebooks/analysis/family_runs.yaml` (new, the list), the suite entry point, the runbook.

- `family_runs.yaml`: the runs that form the merged view, one entry per run: the category and
  run id (or `latest`), the architectures to take (default: every architecture the manifest
  names), the protocol tag (default none), and whether the run's pre-training directory is
  the canonical one for its architectures. New runs are added by one entry; the meta-GGA
  run and the two arms are listed now with their tags, and enter the figures as their cells
  land.
- `merge_family_runs.py`: composes `~/.../runs/dfs_step7/family_view/<stamp>/` from the list
  as `merge_v4_arms.py` does for the v4 arms: renumbered `checkpoints/spec_XXXX` symlinks,
  a composed `manifest.json` whose cells carry the stored key, the displayed name and the
  protocol tag, `pretrain/<arch>` symlinks from the canonical run, the certificates carried
  with their status, `resolved_config.yaml` copied from the first run with the merged sweep
  written in; refuses a run whose identity (basis, grid, density fitting, solver) differs
  from the first, unless the entry carries a protocol tag, in which case the difference is
  recorded on the cells.
- The suite runs once on the view: `figures_dfs_step7_v7_family/` (+ `_val_best`,
  `_excl_tail`), every figure family present exactly once, the architecture axis ordered by
  `ARCH_ORDER`; the pre-training figures (`plot_pretraining_curves.py`, `pretrain_fx_fc.py`,
  `plot_certificate_summary.py`) run on the view's `pretrain/` and write into
  `figures_dfs_step7_v7_family_pretrain/`; the per-run sets stop being regenerated and are
  removed from the tree (history keeps them).
- The runbook's standard refresh becomes pull, merge, suite, in three commands; the merge
  prints the cell inventory per run and refuses a listed run that is absent locally.
- Tests: a fixture of two synthetic runs merges to one view with the right cell count, the
  renumbering is stable across re-runs, a protocol-tagged run's cells carry the tag in the
  manifest and in the rendered legend, a run with a different identity and no tag is
  refused, every finished cell of every listed run appears in the merged CSVs (the future
  guard: a listed run whose cells are missing from the view fails the test), and the
  documents' figure references point only at the family directories (item 3's guard).

## Item 3: documents on the merged set

Files: the report parts (assembled outside the tree), `REPORT_v7_2026-09-09.md`,
`SUMMARY_v7_2026-09-09.md`, `SLIDES_v7_2026-09-09_frames.tex` (both decks), `md_to_tex.py`
unchanged, HISTORY.

- Every figure reference moves to the family directories; each figure family appears once
  with all cells; the per-cell tables are single tables over all cells (with the displayed
  name, the expanded key in the caption, the protocol tag where present, and the $\Delta$ED
  column with the winners marked); the "G1 / G2a" framing becomes one provenance line in the
  scope table; the architecture section and slide are written around the real axes
  (initialization, attention, cusp, rung), with the replicate agreement of deep_3x16 and
  deep0_3x16 stated as such; the pre-training figure titles are the corrected ones.
- The reading sections are re-derived from the merged CSVs (no per-group sentences), every
  number CSV-backed as before.
- A build guard: the document builds fail if a figure path outside the family directories
  is referenced.

## Order and evidence

Item 1 before item 2 (the merged view's labels use the map); item 3 last. Each item under the
delivery gates: claims from executed commands, RED tests by the review model, GREEN, a
mutation battery, an independent review, HISTORY, commit. The cluster is not touched: no
running arm changes, no configuration file of a running run changes.

## What changes for future results

- A new run is added to `family_runs.yaml`; the merge, the suite and the document builds
  pick it up; a test fails if a listed run's finished cells are missing from the view.
- A new architecture gets a display entry and an expanded key in `arch_style.py`; a test
  fails if a registry key has none.
- The next campaign's registry uses the shown names as stored keys, and the two colliding
  names (`deep_3x16`, `deep_attn_3x16`) are renamed there; this plan does not rename stored
  keys because the running arms and every pulled result are filed under them.

## Data at the time of the plan (pull of 2026-09-09, full profiles)

Counted locally from the manifests and the `eval_holdout_val_best/per_reaction.json` files:

- G1 `run_20260902T145245Z`: 20 cells with a validation-best evaluation (18 in the tracked
  figure CSVs); new: medium_attn at 12 and 15 points.
- G2a `run_20260902T145247Z`: 13 cells (10 tracked); new: deep_3x16 at 26 points (the column
  is complete) and deep_attn_3x16 at 1 and 2 points (the first cells of that architecture).
- Meta-GGA `run_20260902T145250Z`: still one certificate (deep_cusp_mgga_3x16, FAIL), no cells.
- Arms `v7g1_c25` and `v7g1_dfsparity` (`run_20260908T153856Z`, `run_20260908T153908Z`): the
  four copied G1 certificates each, no cells yet.
- No `eval_holdout_converged` channel in any run yet.
- The full profile adds the model weights, class records and logs per cell; the figures read
  nothing new from them, but the enhancement-factor scripts no longer depend on a partial pull.

The five new cells enter every figure and table through the merged view of item 2; item 3
re-derives every CSV-backed sentence of the documents from the merged CSVs, so the counts
(33 cells, 21 architectures-by-size in the deep family, the first deep0_attn_3x16 cells) and
the verdicts are recomputed rather than edited. The tracked per-run figure sets are not
regenerated for these cells; they are replaced.
