# Density and combined energy-density figures -- reader's guide (dfs_step7 ablation suite)

This note decodes every panel, label, marker, and footer band of the density-related
figures the ablation suite writes into `figures_dfs_step7_<alias>/` and
`figures_dfs_step7_<alias>_val_best/`:

| File | One-line purpose |
|---|---|
| `ablation_insample_density_ccsd.png` | Training-set density FIT vs CCSD (not generalization) |
| `ablation_holdout_density_ccsd.png` | Held-out density GENERALIZATION vs CCSD, NN vs PBE (trend + parity) |
| `ablation_holdout_density_per_arch.png` | The per-arch held-out density trend alone (standalone) |
| `ablation_combined_energy_density.png` | DFS Eq. 21 combined energy-density metric ED, per cell |
| `ablation_combined_energy_density.csv` | Machine-readable per-cell ED table (all energy legs) |
| `ablation_combined_energy_density_dfs_units.png` | The ED parity in DFS units: Eq. 20 eps density leg, gamma fixed (published / own-axes fit); eps pulls only |
| `ablation_density_energy_overview.png` | One-canvas held-out story: per-pool WTMAD-2 + density parity + iso-ED + ED |
| `ablation_density_energy_overview_dfs_units.png` | The overview with panels D/E/F in DFS units (eps parity, ED under the operative gamma stamped in-panel); eps pulls only |
| `ablation_density_energy_3x3.png` (+ `.csv`) | Per-channel 3x3, ALL BARS: WTMAD-2 / density RMSE / ED for BH76, W4-11, combined |
| `ablation_density_energy_3x3_dfs_units.png` (+ `.csv`) | The all-bars 3x3 in DFS units: eps density bars + per-channel ED bars under ONE shared gamma (stamped in-panel); eps pulls only |
| `ablation_density_parity_by_channel.png` | Per-species NN-vs-PBE density-RMSE parity by channel (one shared frame) -- the 3x3's former parity row |
| `ablation_density_parity_by_channel_dfs_units.png` | The per-channel parity in eps units (shared frame); eps pulls only |
| `ablation_ed_decomposition.png` | The iso-ED decomposition as its own enriched canvas |
| `ablation_ed_decomposition_dfs_units.png` | The enriched decomposition under the operative DFS-units gamma (Eq. 20 eps units, stamped in-panel); eps pulls only |
| `ablation_insample_overview.png` | One-canvas in-sample story: AE + density (training fit) |

The intended use mirrors the multimode figure glossary: open the figure on one screen and
this file on the other. The `figures_*` directories are regenerated outputs and are never
version-controlled; this guide and the producing script
(`notebooks/analysis/make_ablation_arch_figure.py`) are the durable artifacts. Line anchors
below refer to that script unless another file is named, at the revision that
extended the dataset line to the energy figures; `grep -n` for the symbol name if drift is suspected.

## 1. Where the numbers come from

Every quantity is read from the JSON a cluster pull already delivers -- no SCF runs locally,
no model weights are opened.

| Source file (per spec dir) | Feeds | Notes |
|---|---|---|
| `eval/per_molecule.json` | All IN-SAMPLE panels | Final-checkpoint eval of the trained molecules. There is no val-best variant of this file, so in-sample panels are identical across the two output dirs (only the title's checkpoint stamp differs). Carries `AE_error_kcalmol`, `AE_ref_kcalmol`, `density_rmse`, `density_l1`, `density_rmse_pbe`, `density_l1_pbe`, `ref_density_method`; it has NO PBE AE column. |
| `eval_holdout*/per_reaction.json` | All held-out ENERGY panels | One row per evaluated benchmark reaction (`pool` = `bh76` or `w411`) with `abs_error_nn_kcalmol`, `abs_error_pbe_kcalmol`, `reaction_energy_ref_kcalmol`. These are the reactions the eval wrote -- the run's TEST slice, not the full canonical pool; the dataset footer line (Sec. 3) carries the live name-deduplicated counts. The three variants `eval_holdout/`, `eval_holdout_best/`, `eval_holdout_val_best/` hold the final-step, train-best, and validation-best checkpoints' evals. |
| `eval_holdout*/per_molecule.json` | All held-out DENSITY panels | Per benchmark species: `density_rmse`, `density_l1` (NN vs CCSD) and `density_rmse_pbe`, `density_l1_pbe` (model-free PBE vs CCSD on the same grid). Atoms carry None by design (skipped as `atomic_system`, `xcquinox/alec/evaluation.py:203`). |
| `pbe_density_errors.json` (run level) | Optional PBE density anchor | Written only by `reeval_holdout_fixed.py --pbe-density-only`; takes precedence over the inline PBE columns when present (`_pbe_density_map`, :1869). Absent on ordinary pulls. |

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
(from `eval_holdout_val_best/`), mapped by `_ckpt_label` (:4330).

## 2. The metrics

### 2.1 Combined reaction MAE

Plain mean of |reaction-energy error| (kcal/mol) over the held-out reactions of a cell
(`reaction_mae_by_arch_subset`, :439). The PBE twin deduplicates rows by reaction name
before averaging, because the PBE error is spec-invariant and would otherwise be counted
once per spec (`pbe_reaction_mae_baseline`, :1939; `_dedup_rows_by_name`, :1649).

### 2.2 2-subset WTMAD-2 and its single-pool reduction

The suite's WTMAD-2 follows the GMTKN55 Eq. 14 form with the global scale
56.84 kcal/mol (`_GMTKN55_SCALE`, :1646; Goerigk 2017):

    WTMAD-2 = (56.84 / N_total) * sum_pools N_pool * MAD_pool / mean|dE_ref|_pool

computed over the two pools present here, BH76 barrier heights and W4-11 atomization
energies (`_wtmad2_over_pools`, :1663; per-cell map `wtmad2_by_arch_subset`, :1682; pooled
PBE baseline with name-dedup `wtmad2_pbe_baseline`, :1707). Two caveats are stamped wherever
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

The suite's implementation (`combined_ed_by_cell`, :1961; section note :1821-1847) keeps the
Eq. 21 form with three documented deviations, each stamped on the figures:

1. **gamma is self-calibrated, per energy leg, from the pooled PBE anchors:**
   `gamma = E_PBE / D_PBE`, so `ED_PBE == E_PBE` by construction and every cell shares the
   PBE kcal/mol scale. What gamma IS: inside each cell's ED, E and D are the NETWORK'S OWN
   aggregate errors vs the references -- gamma is the conversion slope that maps density
   error onto the energy axis, and it must come from OUTSIDE the scored functional
   (gamma = E_NN/D_NN would give gamma*D_NN = E_NN and collapse ED to the pure energy
   error for every network). In the Letter, gamma is the zero-intercept regression slope
   of WTMAD-2 on eps across six nonempirical functionals ("the energy error (WTMAD-2), a
   fictional nonempirical functional with density error eps_n would exhibit"); here the
   calibration set is collapsed to one functional, PBE (a zero-intercept line through one
   point has slope E_PBE/D_PBE). The Letter's 1084.87 kcal/mol is dimensionally tied to
   its per-electron L1 density units and would be wrong against the grid-weight-averaged
   RMSE stored by the eval pipeline. Consequence: ED here is a relative-to-PBE score; its
   values are comparable across cells of one figure, not across runs with different PBE
   baselines.
2. **Energy legs:** the headline leg is the 2-subset WTMAD-2 (Sec. 2.2), and a second leg
   uses the combined reaction MAE. The Spearman rank correlation between the two legs' ED
   values over the shared cells is printed in the figure's note band -- the ranking, not the
   absolute number, is the claim, and the two legs agreeing shows it is leg-independent.
3. **Density leg:** the grid-weight-averaged RMSE of Sec. 1, not Eq. 20. The Letter's own
   SI Sec. VI reports the findings "largely independent of the density error metric chosen",
   and its L2 variant (SI Eq. 8) correlates best with WTMAD-2 (R^2 = 0.98); gamma absorbs
   the unit change.

**Closing deviations 1 and 3 (DFS-units legs).** The eval also emits the exact Eq. 20
per-species term, `density_eps_l1 = sum_i(w_i |rho - rho_ref|_i) / N_e` with
`N_e = sum_i(w_i rho_ref_i)` (plus the PBE twin and the `n_electrons` / `grid_weight_sum`
bookkeeping; `density_eps_terms` in `xcquinox/alec/evaluation.py`). When a pull carries
those columns, the ED CSV gains a `wtmad2_eps_gamma_dfs` leg: same WTMAD-2 energy cells,
D = per-cell mean eps, and gamma FIXED at the Letter's published 1084.87 kcal/mol -- now
dimensionally valid because the density error is in the Letter's own units. Because that
gamma is external, `ED_PBE != E_PBE` on this leg: PBE lands off the y=x locus by exactly
its displacement from the Letter's cross-functional trend. A second conditional leg,
`wtmad2_eps_gamma_fit`, repeats the construction with gamma refit the DFS way on OUR axes
-- the zero-intercept regression of WTMAD-2 on eps across the same six nonempirical
functionals, computed from the offline calibration cache
(`precompute_nonempirical_pool.py`; `nonempirical_gamma` / `gamma_zero_intercept` in the
figure module). The fitted slope is not expected to equal 1084.87 (different reaction set,
basis, grid, and reference level); the published constant transplants the Letter's scale,
the fitted one reproduces its procedure. Both legs are strictly additive -- pulls without
the eps columns produce byte-identical CSVs, and the self-calibrated legs above remain the
headline for relative-to-PBE claims.

The DFS-units legs also render as figures -- four `_dfs_units` twins, one per ED surface:
`ablation_combined_energy_density_dfs_units.png` (panel (a) the ED lines under the
published gamma, (b) the (E, gamma*D) decomposition, (c) the own-axes-fit leg when the
calibration cache sits in the run dir, a labeled placeholder otherwise),
`ablation_ed_decomposition_dfs_units.png` (the enriched decomposition under the operative
gamma), `ablation_density_energy_overview_dfs_units.png` (the held-out overview with panel
D's parity in eps units and E/F under the operative gamma, stamped in-panel), and
`ablation_density_energy_3x3_dfs_units.png` + `.csv` (the per-channel 3x3 with row 2 in
eps units and row 3's ED under ONE shared gamma -- so, unlike the self-calibrated
original whose per-channel gammas forbid it, EDs DO compare across the
BH76 | W4-11 | combined columns. The OPERATIVE gamma is the own-axes six-functional fit
whenever the calibration cache resolves next to the run dir -- the calibration performed
on this data's axes -- with the Letter's published 1084.87 only as the fallback; the
value and its source are stamped top-right in each ED panel, and the CSV carries BOTH
families, `<channel>_wtmad2_eps_gamma_dfs` (published) and
`<channel>_wtmad2_eps_gamma_fit` (own-axes, cache present). The twin is ALL BARS: row 2 =
per-channel cell-mean epsilon_|n| bars (PBE dashed at the channel anchor), row 3 = the
combined-metric bars; its caveat line 1 spells out the single-pool "one-bucket"
reduction, 56.84*MAD_pool/mean|dE_ref|_pool, and line 2 the metric and density-error
equations in the paper's notation. The per-species parity view lives in
`ablation_density_parity_by_channel_dfs_units.png` -- three channel panels in ONE shared
square frame (own-data envelopes remain on the single-panel parity figures)).
The panel bodies are the shared `gamma_mode`-aware ones, so the self-calibration claims
(ED_PBE = E_PBE, PBE-on-y=x) never appear on these figures; the gamma stamp reads
"(fixed, external)" and the caveats state the Eq. 20 units and the gamma source. The eps
coverage disclosures (partial-backfill cell listing, eps-anchor-vs-union, eps
cell-species homogeneity) are stamped into the note band of ALL four figures in addition
to the console -- on a partially-covered pull the missing cells are named on the figure
itself. Pulls without the eps columns skip the twins with a console line carrying the
suite's standard stale-file warning (a `_dfs_units` file left by a prior eps render
persists).

Supporting rules, all fail-loud: cells lacking a finite value in either leg are excluded and
named in the note band (`_ed_exclusion_note`, :2166); the PBE density anchor deduplicates
molecules across specs and uses finite rows only (`pbe_density_baseline`, :1889); a
divergence between the anchor's molecule set and the NN density union is stamped as a
warning rather than silently averaged (`_pbe_anchor_coverage_warning`, :1907); per-cell
density species sets that differ from the pooled union are named
(`_density_cell_coverage_warning`, :2192). The same arithmetic, on identical inputs, matches
`combined_energy_density` in `notebooks/dfs_selfconsistent_density/dfs_demo.py`
(reimplemented in the figure script; the notebook module's import chain is too heavy for a
plotting-only script).

## 3. Shared visual vocabulary

| Visual | Meaning |
|---|---|
| Bar/line color | Architecture, fixed palette from `arch_style.py` (`ARCH_COLOR`); legends are rung-ordered |
| Black dashed horizontal line | PBE baseline of that panel's metric (energy panels; ED panels) |
| Grey dashed line | PBE-vs-CCSD DENSITY baseline (pool-mean line in held-out density; per-subset line in in-sample density) |
| Green triangle-down | "beats PBE": the value sits strictly below that panel's PBE line (`_beats_pbe_marks`, :419) |
| Grey `x` marker | PBE value (per molecule in strips; the PBE point in the ED decomposition) |
| `n=...` annotations | Number of species behind that mean point |
| Italic line under the title | The panel-family caveat (what the metric is and is not) |
| Small grey line under the caveat | The DATASET line: what the held-out eval is, with live counts -- name-deduplicated reactions per pool and density-species coverage (`_holdout_eval_note`, :2244). Density/ED figures carry the full line; energy figures carry the reactions clause (as a dedicated line on the stamper-based figures, appended to the grey provenance on the five bespoke-footer figures). The name-by-name expansion of this line is `HOLDOUT_SET.md` (this directory) |
| Red band above the footer | Coverage/exclusion warnings: untrained archs, excluded cells, set divergences, the leg-agreement Spearman rho |
| Grey footer line | Data provenance (which JSON, which references, which normalization) |

(Footer stamping: `_stamp_parity_footer`, :1332.)

## 4. Figure register

### 4.1 `ablation_insample_density_ccsd.png` (`plot_insample_density_ccsd`, :2694)

Training-set density FIT -- the direct diagnostic of the 20*rho density term the functionals
were trained with. IN-SAMPLE only; not generalization; final checkpoint always.

| Panel | Content |
|---|---|
| Left | Per-arch mean `density_rmse` vs training subset_size (log y), `n=` species counts, grey dashed PBE-vs-CCSD line over the same subsets (present when the model-free PBE columns exist) |
| Right | Per-molecule strip: every (spec, molecule) point, arch-jittered; one grey `x` per molecule = PBE-vs-CCSD |

### 4.2 `ablation_holdout_density_ccsd.png` (`plot_holdout_density_ccsd`, :2811)

Held-out density GENERALIZATION on the W4-11+BH76 benchmark species (198 with finite
density channels on the current pulls; atoms excluded).

| Panel | Content |
|---|---|
| Left | Per-arch mean held-out `density_rmse` vs subset_size (log y), grey dashed PBE pool-mean line (its label prints the pool-mean value). Also shipped standalone as 4.3 |
| Right | Per-species NN-vs-PBE parity, log-log, dotted diagonal: a point BELOW the diagonal means the NN density is closer to CCSD than PBE is for that species. Falls back to a PBE-only sorted strip when no NN density exists (a PBE-only re-eval) |

### 4.3 `ablation_holdout_density_per_arch.png` (`plot_holdout_density_per_arch`, :2853)

The left panel of 4.2 promoted to its own single-panel figure (same panel body, same PBE
pool-mean baseline, same caveat) after the held-out overview swapped this slot for the
parity and iso-ED decomposition panels. Use it when the per-arch density TREND vs
subset_size is the point; use 4.2's parity panel when the per-species NN-vs-PBE comparison
is the point.

### 4.4 `ablation_combined_energy_density.png` (`plot_combined_energy_density`, :3100)

The DFS Eq. 21 ED figure, NN vs PBE, held-out.

| Panel | Content |
|---|---|
| (a) | Headline ED (2-subset WTMAD-2 leg) vs subset_size per arch, log y; dashed PBE at `ED_PBE == E_PBE`; green beats-PBE marks; the self-calibrated gamma printed lower-left |
| (b) | Per-cell decomposition in the (E, gamma*D) plane, log-log (`_ed_decomposition_panel`): dotted y=x is the self-calibration locus (PBE sits on it exactly, grey `x`); thin grey iso-ED harmonic contours at 0.5x, 1x, 2x the PBE ED; points below the locus are density-limited, above it energy-limited; small digits = subset_size |
| (c) | Secondary ED with the reaction-MAE leg (its own gamma) -- the leg-independence check; renders a grey placeholder when the MAE anchors are unavailable |

### 4.5 `ablation_combined_energy_density.csv` (`write_combined_ed_csv`, :2341)

One row per (energy leg, arch, subset_size); legs are `wtmad2` and `mae`, plus -- only
when the pull carries the Eq. 20 eps columns (Sec. 2.3, "Closing deviations 1 and 3") --
`wtmad2_eps_gamma_dfs` (D = per-cell mean `density_eps_l1`, gamma = 1084.87 fixed) and,
when the nonempirical calibration cache sits in the run dir, `wtmad2_eps_gamma_fit`
(gamma = the own-axes six-functional regression slope). On those two legs the `D_rmse` /
`D_pbe_rmse` columns carry eps values (per-electron L1, not RMSE), `gamma` is the fixed
slope rather than `E_pbe/D_pbe`, and `ED_pbe_kcalmol` generally differs from
`E_pbe_kcalmol`. Columns (`_ED_CSV_FIELDS`, :2335):

| Column | Meaning |
|---|---|
| `leg` | Energy leg: `wtmad2` (headline), `mae`, or -- eps columns present -- `wtmad2_eps_gamma_dfs` / `wtmad2_eps_gamma_fit` |
| `arch` | Architecture (ARCH_ORDER-sorted within each leg) |
| `subset_size` | Training subset size of the cell |
| `n_reactions` | Finite-NN reaction rows in the cell behind E (equals the reaction count under the current one-spec-per-cell layout) |
| `n_density_species` | Finite-NN density rows in the cell behind D (counted on the leg's own channel: RMSE rows, or eps rows on the DFS-units legs) |
| `E_kcalmol` | Cell energy error (2-subset WTMAD-2 or combined reaction MAE) |
| `D_rmse` | Cell mean held-out density error vs CCSD: grid-weighted RMSE on the self-calibrated legs, per-electron L1 eps on the DFS-units legs |
| `gamma` | The leg's rescale slope: `E_pbe_kcalmol / D_pbe_rmse` (self-calibrated legs) or the fixed external slope (1084.87 / the own-axes fit) |
| `gammaD_kcalmol` | `gamma * D_rmse`, the density leg on the energy scale |
| `ED_kcalmol` | `2 / (1/E + 1/(gamma*D))` |
| `E_pbe_kcalmol` | Pooled PBE energy anchor (name-dedup) |
| `D_pbe_rmse` | Pooled PBE density anchor (molecule-dedup, finite rows only; eps units on the DFS-units legs) |
| `ED_pbe_kcalmol` | PBE's ED; equals `E_pbe_kcalmol` by construction on the self-calibrated legs, generally differs on the DFS-units legs |
| `beats_pbe` | `True` iff `ED_kcalmol < ED_pbe_kcalmol` |

### 4.6 `ablation_density_energy_overview.png` (`plot_density_energy_overview`, :3162)

The one-canvas held-out story -- energy above, the energy-density TRADE below; rendered
whenever the held-out density figure renders. Same panel bodies as the dedicated figures
(shared ax-level helpers), so the views cannot drift.

| Panel | Content |
|---|---|
| (A) | WTMAD-2, BH76 only -- the one-bucket reduction of Sec. 2.2 (title says so); own pool-filtered PBE dashed line |
| (B) | WTMAD-2, W4-11 only -- same reduction, same conventions |
| (C) | The genuine 2-subset WTMAD-2 per (arch, subset_size) |
| (D) | Per-species NN-vs-PBE density parity (= right panel of 4.2) |
| (E) | The per-cell (E, gamma*D) iso-ED decomposition (= panel (b) of 4.4); grey "ED decomposition unavailable" placeholder when the ED anchors are missing |
| (F) | The ED headline (= panel (a) of 4.4); same placeholder degradation |

The per-arch density trend that formerly occupied (D) lives in 4.3 (and in the left panel
of 4.2). No SCAN lines anywhere on this figure: a SCAN energy cache exists only for the
combined MAE, and no SCAN WTMAD-2 or SCAN density cache exists.

### 4.7 `ablation_density_energy_3x3.png` + `.csv` (`plot_density_energy_3x3`, :3250)

The per-channel held-out story: one column per channel (BH76 | W4-11 | combined), rendered
whenever the held-out density figure renders.

| Row | Content |
|---|---|
| 1 (A/B/C) | WTMAD-2 bars per (arch, subset_size): A/B are the one-bucket reduction of Sec. 2.2 (titles say so), C the genuine 2-subset form -- the overview's energy row |
| 2 (D/E/F) | Density-error BARS per (arch, subset_size): the cell-mean error on the figure's density channel (RMSE here; eps on the DFS-units twin) restricted to that channel's species, PBE dashed at the channel's deduplicated per-molecule anchor, beats-PBE marks = lower error than PBE. Species-channel membership comes from the reactions' reactants+products (`_species_pools`); overlap species contribute to BOTH channels (stated in the caveat). The bar heights equal the ED legs' D column in the companion CSV |
| 3 (G/H/I) | The DFS Eq. 21 combined metric per channel as BARS (`channel_ed_summaries`): the energy leg is that channel's WTMAD-2 form, the density leg that channel's species, and the panel-title "own gamma" means gamma = E_PBE/D_PBE computed from THAT CHANNEL'S OWN PBE anchors (value stamped in each panel) -- so PBE's value == E_PBE per channel and the metric never compares across channels (the DFS-units twin's SHARED gamma does). Grey placeholder when a channel's anchors are missing |

The whole figure is bar charts; the per-species parity view lives in
`ablation_density_parity_by_channel[_dfs_units].png` (one shared square frame across the
three channel panels, so the channels are directly comparable).

The companion `ablation_density_energy_3x3.csv` reuses the Sec. 4.5 schema with legs
`bh76_wtmad2` / `w411_wtmad2` / `combined_wtmad2` and per-channel `n_reactions` /
`n_density_species` counts (each channel counts only its own pool's rows/species).
Older pulls whose `per_reaction.json` predates the species lists (no
`reactants`/`products`) cannot map species to channels: the single-pool columns then
render placeholders/empty panels while the combined column stays intact.

Figure text uses the DFS paper's notation: the combined metric is the calligraphic ED
(with the |n| subscript on the eps-leg figures, matching the Letter's Eq. 21 / Table I),
and the per-electron density error is epsilon_|n| with its defining integral shown in the
eps caveats (Eq. 20). CSV column names keep the ASCII schema of Sec. 4.5.

### 4.8 `ablation_ed_decomposition.png` (`plot_ed_decomposition`, :3077)

The iso-ED decomposition promoted to its own enriched canvas
(`_ed_decomposition_rich_panel`, :2997), WTMAD-2 leg, same `combined_ed_by_cell` summary as
the ED figure's headline:

| Visual | Meaning |
|---|---|
| Labeled grey curves | Iso-ED contour family at 0.25x, 0.5x, 0.75x, 1x, 1.5x, 2x, 3x the PBE ED; each label sits where the curve crosses the y=x locus (where E = gamma*D = ED) |
| Light green shading | The beats-PBE region, ED < ED of PBE -- including the whole strip E < ED_PBE/2, where no density error can push the harmonic mean back up to PBE's ED |
| Thin colored lines | Per-arch trajectories through the cells in subset_size order (digits = subset_size) |
| Dotted diagonal | The y=x self-calibration locus; the black `x` is PBE, on it by construction |

### 4.9 `ablation_insample_overview.png` (`plot_insample_overview`, :3337)

The one-canvas in-sample (training-fit) story; always rendered. Three stamped disclosures:
final checkpoint only (`eval/` has no val-best variant, so the panels are identical in the
two output dirs and only the title stamp differs); no PBE AE baseline exists in-sample
(`per_molecule.json` has no PBE AE column); hence no in-sample ED (no PBE energy anchor to
self-calibrate gamma).

| Panel | Content |
|---|---|
| (A) | In-sample AE MAE per (arch, subset_size) bars, NN only -- no PBE line by construction. The near-zero subset_size-1 bars are real: a one-molecule training set fits its own AE |
| (B) | Per-molecule \|AE error\| strip (log y), arch-jittered (`_insample_ae_strip_panel`, :2666) |
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

Start with the held-out overview (4.6): panels A-C say whether a cell beats PBE on energies
per pool and jointly; D shows the per-species density comparison; E-F say it in the
Letter's joint metric, with E separating energy-limited from density-limited cells. For the
per-channel question -- does a cell win on barriers, atomizations, or both, on energies AND
densities? -- use the 3x3 (4.7); for the richest single view of the energy-density trade,
the standalone decomposition (4.8). The per-arch density trend is 4.3. Beats-PBE statements
in the DFS sense belong to the ED cells (4.4/4.5 and the per-channel 4.7 CSV,
`beats_pbe` column) and to the WTMAD-2 panels; per-channel EDs compare within a channel
only (each has its own gamma). In-sample figures (4.1, 4.9) support statements about
training fit only, never about generalization. Cross-basis ED values are relative to each
basis's own PBE baseline (Sec. 2.3, deviation 1) -- compare rankings, not raw ED, across
bases.
