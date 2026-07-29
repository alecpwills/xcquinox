# Density and combined energy-density figures -- reader's guide (dfs_step7 ablation suite)

This note decodes every panel, label, marker, and footer band of the density-related
figures the ablation suite writes into `figures_dfs_step7_<alias>/` and
`figures_dfs_step7_<alias>_val_best/`:

| File | One-line purpose |
|---|---|
| `ablation_insample_density_ccsd.png` | Training-set density FIT vs CCSD (not generalization) |
| `ablation_holdout_density_ccsd.png` | Held-out density GENERALIZATION vs CCSD, NN vs PBE |
| `ablation_combined_energy_density.png` | DFS Eq. 21 combined energy-density metric ED, per cell |
| `ablation_combined_energy_density.csv` | Machine-readable per-cell ED table (both energy legs) |
| `ablation_density_energy_overview.png` | One-canvas held-out story: per-pool WTMAD-2 + density + ED |
| `ablation_insample_overview.png` | One-canvas in-sample story: AE + density (training fit) |

The intended use mirrors the multimode figure glossary: open the figure on one screen and
this file on the other. The `figures_*` directories are regenerated outputs and are never
version-controlled; this guide and the producing script
(`notebooks/analysis/make_ablation_arch_figure.py`) are the durable artifacts. Line anchors
below refer to that script unless another file is named, at the revision that introduced the
overview composites; `grep -n` for the symbol name if drift is suspected.

## 1. Where the numbers come from

Every quantity is read from the JSON a cluster pull already delivers -- no SCF runs locally,
no model weights are opened.

| Source file (per spec dir) | Feeds | Notes |
|---|---|---|
| `eval/per_molecule.json` | All IN-SAMPLE panels | Final-checkpoint eval of the trained molecules. There is no val-best variant of this file, so in-sample panels are identical across the two output dirs (only the title's checkpoint stamp differs). Carries `AE_error_kcalmol`, `AE_ref_kcalmol`, `density_rmse`, `density_l1`, `density_rmse_pbe`, `density_l1_pbe`, `ref_density_method`; it has NO PBE AE column. |
| `eval_holdout*/per_reaction.json` | All held-out ENERGY panels | One row per benchmark reaction (`pool` = `bh76` or `w411`) with `abs_error_nn_kcalmol`, `abs_error_pbe_kcalmol`, `reaction_energy_ref_kcalmol`. The three variants `eval_holdout/`, `eval_holdout_best/`, `eval_holdout_val_best/` hold the final-step, train-best, and validation-best checkpoints' evals. |
| `eval_holdout*/per_molecule.json` | All held-out DENSITY panels | Per benchmark species: `density_rmse`, `density_l1` (NN vs CCSD) and `density_rmse_pbe`, `density_l1_pbe` (model-free PBE vs CCSD on the same grid). Atoms carry None by design (skipped as `atomic_system`, `xcquinox/alec/evaluation.py:203`). |
| `pbe_density_errors.json` (run level) | Optional PBE density anchor | Written only by `reeval_holdout_fixed.py --pbe-density-only`; takes precedence over the inline PBE columns when present (`_pbe_density_map`, :1836). Absent on ordinary pulls. |

Reference densities are CCSD (not CCSD(T)) benchmark references generated at the SAME basis
and grid as the run (`xcquinox.alec.benchmark_refs`); the PBE density channel is model-free
and identical for every spec of a run. The density error scalar is computed cluster-side at
eval time (`DensityRMSEMetric`, `xcquinox/alec/evaluation.py:185`):

    density_rmse = sqrt( sum_i w_i (rho_NN(r_i) - rho_ref(r_i))^2 / sum_i w_i )
    density_l1   =       sum_i w_i |rho_NN(r_i) - rho_ref(r_i)|   / sum_i w_i

with w_i the DFT quadrature weights (`evaluation.py:241-242`; PBE twin at :152-180). This is
a grid-weight-AVERAGED error -- deliberately NOT the per-electron L1 of the DFS Letter's
Eq. 20 and NOT the N_e^2-normalized form used inside the training loss (see Sec. 2.3).

Checkpoint stamps: figure suptitles end in `final-step` (from `eval_holdout/`) or `val-best`
(from `eval_holdout_val_best/`), mapped by `_ckpt_label` (:3755).

## 2. The metrics

### 2.1 Combined reaction MAE

Plain mean of |reaction-energy error| (kcal/mol) over the held-out reactions of a cell
(`reaction_mae_by_arch_subset`, :434). The PBE twin deduplicates rows by reaction name
before averaging, because the PBE error is spec-invariant and would otherwise be counted
once per spec (`pbe_reaction_mae_baseline`, :1900; `_dedup_rows_by_name`, :1625).

### 2.2 2-subset WTMAD-2 and its single-pool reduction

The suite's WTMAD-2 follows the GMTKN55 Eq. 14 form with the global scale
56.84 kcal/mol (`_GMTKN55_SCALE`, :1622; Goerigk 2017):

    WTMAD-2 = (56.84 / N_total) * sum_pools N_pool * MAD_pool / mean|dE_ref|_pool

computed over the two pools present here, BH76 barrier heights and W4-11 atomization
energies (`_wtmad2_over_pools`, :1639; per-cell map `wtmad2_by_arch_subset`, :1658; pooled
PBE baseline with name-dedup `wtmad2_pbe_baseline`, :1683). Two caveats are stamped wherever
these numbers appear:

- it is a LABELED reweighting over 2 subsets, NOT the full 55-subset GMTKN55 WTMAD-2;
- restricted to ONE pool the sum collapses, since N_pool = N_total, to
  `56.84 * MAD_pool / mean|dE_ref|_pool` -- a scaled relative error, not a reweighting.
  The per-pool panels of the held-out overview are exactly this reduction and say so in
  their titles. Only the 2-subset panel balances BH76 against W4-11.

### 2.3 The DFS combined energy-density metric ED

Source: Dick and Fernandez-Serra, "Highly accurate and constrained density functional
obtained with differentiable programming", Phys. Rev. B 104, L161109 (2021). The Letter
proposes (Eq. 21, p. L161109-5) the harmonic mean of an energy error and a density error
rescaled to an energy:

    ED = 2 / ( 1/E + 1/(gamma * D) )        [kcal/mol]

with, in the Letter, E = WTMAD-2 on diet-GMTKN55-150 (its Eq. 19), D = the per-electron L1
density error eps_|n| = E[(1/N_e) INT |n - n_ref|] (its Eq. 20), and
gamma = 1084.87 kcal/mol, the slope of a zero-intercept regression of WTMAD-2 on eps_|n|
across six nonempirical functionals (PW91, PBE, TPSS, revTPSS, SCAN, PBE0; R^2 = 0.87).

The suite's implementation (`combined_ed_by_cell`, :1922; section note :1791-1817) keeps the
Eq. 21 form with three documented deviations, each stamped on the figures:

1. **gamma is self-calibrated, per energy leg, from the pooled PBE anchors:**
   `gamma = E_PBE / D_PBE`, so `ED_PBE == E_PBE` by construction and every cell shares the
   PBE kcal/mol scale. The Letter's 1084.87 kcal/mol is dimensionally tied to its
   per-electron L1 density units and would be wrong against the grid-weight-averaged RMSE
   stored by the eval pipeline. Consequence: ED here is a relative-to-PBE score; its values
   are comparable across cells of one figure, not across runs with different PBE baselines.
2. **Energy legs:** the headline leg is the 2-subset WTMAD-2 (Sec. 2.2), and a second leg
   uses the combined reaction MAE. The Spearman rank correlation between the two legs' ED
   values over the shared cells is printed in the figure's note band -- the ranking, not the
   absolute number, is the claim, and the two legs agreeing shows it is leg-independent.
3. **Density leg:** the grid-weight-averaged RMSE of Sec. 1, not Eq. 20. The Letter's own
   SI Sec. VI reports the findings "largely independent of the density error metric chosen",
   and its L2 variant (SI Eq. 8) correlates best with WTMAD-2 (R^2 = 0.98); gamma absorbs
   the unit change.

Supporting rules, all fail-loud: cells lacking a finite value in either leg are excluded and
named in the note band (`_ed_exclusion_note`, :1960); the PBE density anchor deduplicates
molecules across specs and uses finite rows only (`pbe_density_baseline`, :1855); a
divergence between the anchor's molecule set and the NN density union is stamped as a
warning rather than silently averaged (`_pbe_anchor_coverage_warning`, :1871); per-cell
density species sets that differ from the pooled union are named
(`_density_cell_coverage_warning`, :1986). The same arithmetic, on identical inputs, matches
`combined_energy_density` in `notebooks/dfs_selfconsistent_density/dfs_demo.py`
(reimplemented in the figure script; the notebook module's import chain is too heavy for a
plotting-only script).

## 3. Shared visual vocabulary

| Visual | Meaning |
|---|---|
| Bar/line color | Architecture, fixed palette from `arch_style.py` (`ARCH_COLOR`); legends are rung-ordered |
| Black dashed horizontal line | PBE baseline of that panel's metric (energy panels; ED panels) |
| Grey dashed line | PBE-vs-CCSD DENSITY baseline (pool-mean line in held-out density; per-subset line in in-sample density) |
| Green triangle-down | "beats PBE": the value sits strictly below that panel's PBE line (`_beats_pbe_marks`, :414) |
| Grey `x` marker | PBE value (per molecule in strips; the PBE point in the ED decomposition) |
| `n=...` annotations | Number of species behind that mean point |
| Italic line under the title | The panel-family caveat (what the metric is and is not) |
| Red band above the footer | Coverage/exclusion warnings: untrained archs, excluded cells, set divergences, the leg-agreement Spearman rho |
| Grey footer line | Data provenance (which JSON, which references, which normalization) |

(Footer stamping: `_stamp_parity_footer`, :1325.)

## 4. Figure register

### 4.1 `ablation_insample_density_ccsd.png` (`plot_insample_density_ccsd`, :2387)

Training-set density FIT -- the direct diagnostic of the 20*rho density term the functionals
were trained with. IN-SAMPLE only; not generalization; final checkpoint always.

| Panel | Content |
|---|---|
| Left | Per-arch mean `density_rmse` vs training subset_size (log y), `n=` species counts, grey dashed PBE-vs-CCSD line over the same subsets (present when the model-free PBE columns exist) |
| Right | Per-molecule strip: every (spec, molecule) point, arch-jittered; one grey `x` per molecule = PBE-vs-CCSD |

### 4.2 `ablation_holdout_density_ccsd.png` (`plot_holdout_density_ccsd`, :2497)

Held-out density GENERALIZATION on the W4-11+BH76 benchmark species (~198 with finite
density channels; atoms excluded).

| Panel | Content |
|---|---|
| Left | Per-arch mean held-out `density_rmse` vs subset_size (log y), grey dashed PBE pool-mean line (its label prints the pool-mean value) |
| Right | Per-species NN-vs-PBE parity, log-log, dotted diagonal: a point BELOW the diagonal means the NN density is closer to CCSD than PBE is for that species. Falls back to a PBE-only sorted strip when no NN density exists (a PBE-only re-eval) |

### 4.3 `ablation_combined_energy_density.png` (`plot_combined_energy_density`, :2594)

The DFS Eq. 21 ED figure, NN vs PBE, held-out.

| Panel | Content |
|---|---|
| (a) | Headline ED (2-subset WTMAD-2 leg) vs subset_size per arch, log y; dashed PBE at `ED_PBE == E_PBE`; green beats-PBE marks; the self-calibrated gamma printed lower-left |
| (b) | Per-cell decomposition in the (E, gamma*D) plane, log-log: dotted y=x is the self-calibration locus (PBE sits on it exactly, grey `x`); thin grey iso-ED harmonic contours at 0.5x, 1x, 2x the PBE ED; points below the locus are density-limited, above it energy-limited; small digits = subset_size |
| (c) | Secondary ED with the reaction-MAE leg (its own gamma) -- the leg-independence check; renders a grey placeholder when the MAE anchors are unavailable |

### 4.4 `ablation_combined_energy_density.csv` (`write_combined_ed_csv`, :2044)

One row per (energy leg, arch, subset_size); legs are `wtmad2` and `mae`. Columns
(`_ED_CSV_FIELDS`, :2038):

| Column | Meaning |
|---|---|
| `leg` | Energy leg: `wtmad2` (headline) or `mae` |
| `arch` | Architecture (ARCH_ORDER-sorted within each leg) |
| `subset_size` | Training subset size of the cell |
| `n_reactions` | Finite-NN reaction rows in the cell behind E (equals the reaction count under the current one-spec-per-cell layout) |
| `n_density_species` | Finite-NN density rows in the cell behind D (equals the species count under the same layout) |
| `E_kcalmol` | Cell energy error (2-subset WTMAD-2 or combined reaction MAE) |
| `D_rmse` | Cell mean held-out density RMSE vs CCSD |
| `gamma` | Self-calibrated rescale, `E_pbe_kcalmol / D_pbe_rmse` (one value per leg) |
| `gammaD_kcalmol` | `gamma * D_rmse`, the density leg on the energy scale |
| `ED_kcalmol` | `2 / (1/E + 1/(gamma*D))` |
| `E_pbe_kcalmol` | Pooled PBE energy anchor (name-dedup) |
| `D_pbe_rmse` | Pooled PBE density anchor (molecule-dedup, finite rows only) |
| `ED_pbe_kcalmol` | PBE's ED; equals `E_pbe_kcalmol` by construction |
| `beats_pbe` | `True` iff `ED_kcalmol < ED_pbe_kcalmol` |

### 4.5 `ablation_density_energy_overview.png` (`plot_density_energy_overview`, :2695)

The one-canvas held-out story; rendered whenever the held-out density figure renders. Same
panel bodies as the dedicated figures (shared ax-level helpers), so the views cannot drift.

| Panel | Content |
|---|---|
| (A) | WTMAD-2, BH76 only -- the one-bucket reduction of Sec. 2.2 (title says so); own pool-filtered PBE dashed line |
| (B) | WTMAD-2, W4-11 only -- same reduction, same conventions |
| (C) | The genuine 2-subset WTMAD-2 per (arch, subset_size) |
| (D) | Held-out density RMSE vs subset_size (= left panel of 4.2) |
| (E) | Per-species NN-vs-PBE density parity (= right panel of 4.2) |
| (F) | The ED headline (= panel (a) of 4.3); degrades to a grey "ED unavailable" placeholder when the ED anchors are missing |

No SCAN lines anywhere on this figure: a SCAN energy cache exists only for the combined MAE,
and no SCAN WTMAD-2 or SCAN density cache exists.

### 4.6 `ablation_insample_overview.png` (`plot_insample_overview`, :2764)

The one-canvas in-sample (training-fit) story; always rendered. Three stamped disclosures:
final checkpoint only (`eval/` has no val-best variant, so the panels are identical in the
two output dirs and only the title stamp differs); no PBE AE baseline exists in-sample
(`per_molecule.json` has no PBE AE column); hence no in-sample ED (no PBE energy anchor to
self-calibrate gamma).

| Panel | Content |
|---|---|
| (A) | In-sample AE MAE per (arch, subset_size) bars, NN only -- no PBE line by construction. The near-zero subset_size-1 bars are real: a one-molecule training set fits its own AE |
| (B) | Per-molecule \|AE error\| strip (log y), arch-jittered (`_insample_ae_strip_panel`, :2359) |
| (C) | In-sample density RMSE vs subset_size (= left panel of 4.1, PBE dashed line included) |
| (D) | Per-molecule density strip with grey PBE `x` (= right panel of 4.1) |

## 5. Regeneration

One suite invocation refreshes every figure above for the pulled bases, both checkpoint
variants; see `RUNBOOK_pull_and_figures.md` for the pull commands and the canonical

    python notebooks/analysis/make_ablation_arch_figure.py --suite \
        --domain dfs_step7 --bases <comma-separated basis subdirs> \
        --outroot notebooks/analysis

Figures gated on missing inputs are skipped with a console note; a stale file from an
earlier render then persists on disk -- the console line is the truth about what was
refreshed.

## 6. Reading order and permitted claims

Start with the held-out overview (4.5): panels A-C say whether a cell beats PBE on energies
per pool and jointly; D-E say the same for densities; F says it in the Letter's joint
metric. Beats-PBE statements in the DFS sense belong to the ED cells (4.3/4.4, `beats_pbe`
column) and to the WTMAD-2 panels; in-sample figures (4.1, 4.6) support statements about
training fit only, never about generalization. Cross-basis ED values are relative to each
basis's own PBE baseline (Sec. 2.3, deviation 1) -- compare rankings, not raw ED, across
bases.
