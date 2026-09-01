# Pretraining evolution across the campaign generations (v4, v5, v6)

This document records how the pretraining stage of the dfs6311 campaigns changed from the
v4 generation (unanchored point-wise fits to the parent's enhancement factors), through v5
(per-rung self-consistent seeding around an unchanged pretraining scheme), to v6 (parent-anchored
networks with a machine-checked fidelity certificate), together with the measured findings that
forced each change and the first quantitative comparison of the completed v6 GGA cells against
the corrected v4 GGA record. Every number carries its source: an entry of
`xcquinox/alec/HISTORY.md` (cited as `HISTORY <date>`), a code definition (cited as
`file:symbol` or `file:line`), a figure CSV under `notebooks/analysis/figures_*`, or a pulled
cluster artifact (cited relative to the local results tree `xcquinox-results/runs/`, e.g.
`dfs_step7/dfs6311_grid3_v6g1_size/runs/run_20260827T163330Z/...`). Values re-derived here were evaluated directly from the
named artifacts and committed implementations. Coverage is stated per number; the campaign is live and several sets below are
partial. Current as of 2026-09-01.

The functional family under study descends from Dick and Fernandez-Serra, Phys. Rev. B 104,
L161109 (2021) ("DFS" below): 3x16 multilayer perceptrons returning exchange and correlation
enhancement factors over an LDA baseline, trained through a differentiable self-consistent
field. The meta-GGA rung uses the iso-orbital indicator of SCAN (Sun, Ruzsinszky and Perdew,
Phys. Rev. Lett. 115, 036402 (2015)).

Figures are embedded at the point their subject is introduced, each with the reading it
supports. All figure paths are relative to `notebooks/analysis/`, where this file lives and
where the document is rendered.

---

## 1. The campaign in one page

This section fixes the vocabulary the rest of the document uses. A reader who knows density
functional theory but not this codebase needs it; a reader who knows the codebase can skip to
Section 2.

### 1.1 Architecture, cell, spec

An **architecture** is a named entry of the registry `xcquinox/alec/config.py` `ARCHITECTURES`
(31 entries, `config.py:505-684`). It fixes the network shape (hidden layers x width, with or
without attention) and the **descriptor set** -- the extra per-grid-point columns the two MLPs
read beyond the semilocal variables (Section 3).

A **grid cell** is the swept unit: `GridCell` (`cluster/grid_config.py:60-72`) carries five
fields, `(arch, loss, metric, subset_size, solver)`, and the grid is their Cartesian product in
that fixed axis order, the position in the expansion being the SLURM array task index
(`grid_config.py:1656-1673`). In every dfs6311 campaign configuration three of the five axes are
singletons -- `loss: L5_gradnorm_vxc_step7`, `metric: jsd`, `solver: full_3` -- so a cell is in
practice exactly one **(architecture, training-subset-size)** pair, and the grid arithmetic is
stated per file (e.g. `hpcjobs/configs/dfs_step7.dfs6311_grid3_v6g1_size.yaml:105-110`,
"4 x 1 x 1 x 11 x 1 = 44 cells").

A **spec** is the on-disk materialization of one cell: a pickled `TrainingSpec`
(`config.py:961` onward) written as `spec_<index>.spec` with a SHA-256 recorded in the run's
`manifest.json` (`cluster/materialize.py:123-268`), and a checkpoint directory
`<run>/checkpoints/spec_<index>/`. What does **not** vary per spec is as important as what does:
the random seed is a single run-level value (`HyperParams.seed = 42`, `grid_config.py:168`,
threaded verbatim at `cluster/spec_builder.py:631`), as are the basis and grid level
(`grid_config.py:211-212`). There is no replicate axis anywhere in the harness, so **every point
on every curve below is a single training run**, and cell-to-cell scatter at fixed architecture
carries seed variance that the campaign does not resolve. What *is* derived per spec is the SCF
seed density: under `inputs.seed_xc: auto` the source is resolved from the architecture's rung
(`spec_builder.py:571-577`), converged PBE for GGA architectures and converged SCAN for
meta-GGA ones (Section 5).

### 1.2 The training pool and the subset-size axis

Training points are drawn from the DFS Letter's own training pool, transcribed verbatim from its
supplementary material (`xcquinox/alec/dfs_pool.py:1-12`, "SI Sec. II 'Training Data'"): 21
atomization energies from G2/97 (10 linear closed-shell, 3 linear open-shell, 8 non-linear), 3
BH76 reaction barriers, 2 IP13 ionization potentials, and 2 atomic-density references (H, Li)
that are always present rather than swept. The 26 selectable points are assembled by
`training_points.build_dfs_pool_points` (`training_points.py:312-316`).

**Subset size (ss) counts training points, not molecules and not reactions.** The axis is
`[1, 2, 3, 4, 5, 6, 7, 12, 15, 18, 26]` in all thirteen dfs6311 configurations (e.g.
`dfs_step7.dfs6311_grid3_v6g1_size.yaml:129`), the largest equal to the whole pool. The
composition at each size is fixed once, in the committed ledger
`notebooks/checkpoints_step7/alpha_on/subset_index_log.json`, and reused by every generation --
which is what makes the subset axis comparable across the campaign lineage
(`CAMPAIGN_V6.md:695-700`). At `ss = 12`, for instance, the ledger's `jsd/12` entry holds 7
atomization energies, 3 BH76 barriers and 2 ionization potentials; at `ss = 26` it holds all 21,
3 and 2. Selection itself is a pre-process, not part of the harness: `subset_selection.select_subset`
(`subset_selection.py:472-487`) enumerates every size-$r$ combination and minimizes the
Jensen-Shannon divergence between the subset's and the full pool's histograms over
$(\rho^{1/3}, s, \alpha)$; the harness consumes the resulting ledger by point *name*, never by
index (`spec_builder.py:38-43`). The number of molecules per spec is larger than ss and varies,
being the union of every chosen point's reactants, products and atomic anchors
(`spec_builder.py:502-511`).

### 1.3 The held-out pools and their slices

Two GMTKN55 subsets are held out (`notebooks/analysis/HOLDOUT_SET.md:317-324`):

* **BH76** -- forward barrier heights (Goerigk, Hansen, Bauer, Ehrlich, Najibi and Grimme,
  Phys. Chem. Chem. Phys. 19, 32184 (2017)), 76 reaction entries.
* **W4-11** -- zero-point-exclusive non-relativistic atomization energies (Karton, Daon and
  Martin, Chem. Phys. Lett. 510, 165 (2011)), 140 entries.

Their union is 216 reaction entries over 214 unique species, with 17 species shared
(`full_benchmark_pools.py:516-527`). Two distinct multiplicities reduce that count and should
not be conflated. Four BH76 entries are *listed twice under the same name* --
`bh76_fch3fcomp_to_fch3fts`, `bh76_clch3clcomp_to_clch3clts`, `bh76_h_H2_to_RKT06` and
`bh76_C5H8_to_RKT22` -- so 216 entries carry **212 distinct names**. Separately, several
barriers appear under permuted reactant order, so the 216 entries carry only **208 distinct
physical identities** (both counts re-derived by enumerating `load_full_held_out_pools()` and
`reaction_identity_key`).

Three slices matter, and the document keeps them apart:

1. **Validation.** A deterministic 20% partition -- **34 names, 35 entries** (15 BH76, 20
   W4-11) -- keyed on the
   reaction's *physical identity* (sorted case-folded reactant and product tuples), not its name,
   so permuted-name twins land on the same side (`eval_holdout.split_held_out`,
   `eval_holdout.py:174-198`; `reaction_identity_key`, `:158-166`). It drives in-loop early stop
   and validation-best checkpoint selection and is withheld from every reported metric.
2. **Test ("strict").** The complement, **178 names covering 181 entries** (61 BH76, 120
   W4-11), minus each cell's own
   *verbatim supervised* reactions. The exclusion is verbatim, not species-level: a held-out
   reaction that merely contains a trained molecule stays, because it is a generalization target
   (`eval_holdout.trained_reaction_exclusion`, `:237-292`; `_finalize_holdout_outputs`,
   `:1304-1330`). Bridging the two naming vocabularies -- ASE Hill formulas in the training pool
   (`HO`, `CHN`, `H3N`) against GMTKN55 names in the benchmark pools (`oh`, `hcn`, `nh3`) -- is
   an explicit identity layer keyed on `(composition, charge, spin)`
   (`species_matching.py:1-24`), without which a strict filter keeps trained molecules'
   reactions in the "held-out" set.
3. **In-sample.** The cell's own training subset, reported separately.

Because the exclusion is per cell, slices are nearly but not exactly uniform across subset
sizes, and they can only shrink from the 61 / 120 / 181 the split itself defines: over the 27
completed v6 G1 cells the strict slices hold 61 BH76, 111--120 W4-11 and
172--181 combined reactions; over the 54 v4 GGA cells, 43--50, 97--113 and 145--163. Every
"beats PBE" verdict below is therefore **within-cell** -- the NN mean absolute error against the
PBE mean absolute error on exactly the reactions the network was scored on -- never a
cross-campaign row-for-row comparison.

The density channel is per species rather than per reaction: each evaluated non-atomic species
carries `density_rmse` (network against CCSD at the run's own basis and grid) and
`density_rmse_pbe` (the model-free PBE twin). Atomic species are skipped by design, since
the atomization anchors make lone-atom densities redundant, leaving **199 density species** of
the 214 evaluated (counted from the `density_rmse` column of a completed spec's
`per_molecule.json`; `HOLDOUT_SET.md:304-313`).

Two conventions used throughout are worth naming here rather than at first use.
**WTMAD-2** is GMTKN55's weighted total mean absolute deviation, the benchmark's own
cross-subset aggregate: each subset's mean absolute error is rescaled by the ratio of the
average absolute reference value across all subsets to that subset's own, so that subsets whose
reference energies are intrinsically large do not dominate a single number (Goerigk et al. 2017,
cited via the repo record at `notebooks/analysis/README_density_figures.md`). It is reported in
kcal/mol and is **not** the same quantity as the plain mean absolute error of Section 9.
The **orientation lock** is a traceless-quadrupole bias of strength $3\times 10^{-5}$
(dimensionless) added to the core Hamiltonian of every reference and evaluation SCF; it removes
the arbitrary spatial orientation a degenerate open-p-shell atom's converged density would
otherwise pick at random, which is what makes the atomic references reproducible across runs and
machines (HISTORY 2026-07-05).

### 1.4 The four evaluation channels

Every completed spec is evaluated four times. All four are dispatched from `_run_held_out_eval`
in `cluster/_eval_one_spec.py`; the channel-to-checkpoint map is stated verbatim at
`hpcjobs/reeval_c2_patch.py:177-186`.

| channel | checkpoint | what selects it | SCF protocol |
|---|---|---|---|
| `eval_holdout` (final) | `model.eqx` | the last training step, written unconditionally (`train.py:915`) | the spec's own `full_3` solver: FULL mode, 3 cycles, `REASSEMBLE` features (every density-matrix-dependent descriptor is recomputed from the live density matrix each SCF cycle, rather than held at its entry value), decaying-linear mixer, `conv_tol` $10^{-6}$; seed = converged parent density |
| `eval_holdout_best` (best-loss) | `model_best.eqx` | minimum trailing-mean **training** loss over one epoch (`train._BestModelTracker`, `train.py:537-543`) | as above |
| `eval_holdout_val_best` (validation-best) | `model_val_best.eqx` | minimum held-out **validation** metric, checked every 25 steps with patience 5 (`train._BestValidationTracker`, `train.py:566-575`) | as above |
| `eval_holdout_coldstart` | `model.eqx` | the same final weights under a different SCF | functional-free `minao` seed, 25 cycles (the Letter's step count), `conv_tol` $10^{-12}$ so the latched freeze never masks the trajectory (`eval_holdout.py:49-75`) |

Two of these carry standing caveats. The **best-loss** channel minimizes a quantity that keeps
falling as a network overfits, so on overfit-prone architectures it selects the *most* overfit
snapshot; it was measured worse on held-out data than even the final step for the v3
`deep_combined` specs (99.98 against 70.62 kcal/mol at spec_0024) and is no longer plotted
(HISTORY 2026-06-28; `make_ablation_arch_figure.py:7531`). The **cold-start** channel is a
trajectory diagnostic, not a converged-evaluation replica: the manual solver has linear mixing
only, so its 25 recorded cycles show whether a functional is walking toward a fixed point rather
than certifying that it arrived (HISTORY 2026-08-14). It exists only from v5 onward
(`grid_config.py:567-572`). Everything headline in this document is the **validation-best**
channel, which is what the figure suite plots (`make_ablation_arch_figure.py:8706-8707`).

### 1.5 Generations and runs

The table below is the roster the rest of the document refers to by short name. Three of those
names recur often enough to fix here: **v3** is the earliest sweep at DFS reference parity and
supplies the oldest comparison curves; **G1** is the v6 anchored size ladder
(`dfs6311_grid3_v6g1_size`), the only v6 group with trained cells on disk and therefore the
source of every v6 held-out number below; and **`merged_v4_arms`** is not a campaign but a
derived directory that merges the completed v4 and v4gga arms onto one comparison slice, which
is what the "merged" figure sets are built from (Sections 9.4 and 10.3).

One recurring failure mode is also named once here. The **open NaN-gradient defect** is an
unresolved condition in which a training cell's loss gradient becomes non-finite mid-run and the
cell terminates without writing a usable checkpoint; it is what left the G1 `medium` ss=26 cell
absent from the tables below, and it is open rather than diagnosed (HISTORY 2026-08-31).

| generation | architectures | cells | distinguishing feature |
|---|---|---|---|
| v3 | 8 (GGA + first meta-GGA forms) | 88 | first sweep at DFS reference parity (6-311++G(3df,2pd), grid 3); closes as the pre-$V_{xc}$-correction record |
| v4 | 3 meta-GGA | 33 | meta-GGA arm of the corrected-$V_{xc}$ re-sweep |
| v4gga | 6 GGA | 66 | GGA arm of the same re-sweep, same methodology, separate output root |
| v4mgga2 | 2 meta-GGA stacking forms | 22 | the two registry completions added 2026-08-10 |
| v5 | 3 meta-GGA | 33 | SCAN-seeded retrain of the v4 meta-GGA arm; a controlled seed A/B, the v4 rows remaining as the PBE-seeded control |
| v5mgga2 | 2 meta-GGA | 22 | the same seed A/B on the stacking arm |
| v6 G1 (`v6g1_size`) | shallow, shallow_attn, medium, medium_attn | 44 | anchored size ladder: does capacity limit anything at fixed inputs |
| v6 G2a (`v6g2a_families_core`) | deep_3x16, deep_attn_3x16, deep_cusp_3x16 | 33 | anchored GGA core trio |
| v6 G2b (`v6g2b_families_rung35`) | the three rung-3.5 GGA forms | 33 | anchored rung-3.5 trio |
| v6 G2 meta-GGA (`v6g2_families_mgga`) | five meta-GGA family forms | 55 | SCAN-parented, SCAN-seeded parity forms |
| v6 G3 (`v6g3_dm`) | deep_dm_3x16, deep_combined_3x16, deep_combined_attn_3x16 | 33 | the density-matrix indicators in their repaired two-column form |
| v6 G4 (`v6g4_ablations`) | deep, deep_attn | 22 | depth 4 / width 32 against the production depth 3 / width 16 |

v6 totals 220 cells over 20 of the 31 registry architectures (`CAMPAIGN_V6.md:934-940`). The
file `dfs_step7.dfs6311_grid3_v6.yaml`, covering all 31 architectures at 341 cells, is the
statement of the method and is not submitted (`CAMPAIGN_V6.md:934-940`).

### 1.6 From artifacts to figures

Each held-out channel writes three data files plus a provenance stamp
(`eval_holdout._finalize_holdout_outputs`, `eval_holdout.py:1379-1387`):

* `test_set.csv` -- the per-spec mean-absolute-error summary, one row per pool plus one
  combined, with `n_reactions`, `n_dropped_overlap` and `n_dropped_nan`
  (`eval_holdout.write_test_set_csv`, `:875-964`). **The tables of Sections 9.2 and 9.4 are
  derived from these files.**
* `per_reaction.json` -- one record per surviving reaction: reference energy, network and PBE
  reaction energies and absolute errors, and the annotated `in_sample_overlap` list.
* `per_molecule.json` -- one record per evaluated species: energies, the six density-error
  columns (`density_rmse`, `density_l1` and `density_eps_l1`, each with its `_pbe` twin),
  SCF convergence and cycle count.
* `eval_metadata.json` -- channel name, checkpoint filename, cold-start flag, the serialized
  solver configuration, and the species slice (`_eval_one_spec.py:471-489`).

The **figures** are built by `notebooks/analysis/make_ablation_arch_figure.py` and read the two
JSON files, not the CSV: `collect_holdout_reaction_rows` (`:274-386`) reconstructs each spec's
full test slice from the per-species energies of `per_molecule.json` and re-applies the verbatim
and validation exclusions itself, falling back to `per_reaction.json` only for pulls that predate
the energy columns. `collect_holdout_density_rows` (`:4010`) reads the density columns. The
enhancement-factor figures come from two further modules, `pretrain_fx_fc.py` and
`trained_fx_fc.py`, which load the stored network weights through the production certified-model
builder and evaluate them on a fixed $(r_s, s)$ grid. Each of those two modules writes **one**
long-form CSV per figure directory -- `pretrain_fx_fc_curves.csv` and
`trained_fx_fc_curves.csv`, one row per plotted grid point across all architectures and cells --
so every plotted enhancement-factor number is recoverable without re-running the code. The
`ablation_*` suite is different: only some of its figures have a same-stem CSV, and the three
`ablation_*` figures embedded below (Section 9.2) have none, so their captions quote the
per-spec artifacts they are built from instead.

---

## 2. The common frame: networks, bounded maps, parent conventions

All three generations share the network form (`xcquinox/alec/networks.py`). Per grid point the
exchange network reads the reduced gradient $s = |\nabla\rho| / (2 k_F \rho)$ of one spin
channel posed on its doubled density, and the correlation network reads the total density with
the spin polarization $\zeta$; descriptor-carrying architectures append extra columns (cusp,
rung-3.5 occupancies, the meta-GGA indicator; Section 3). The raw MLP output is gated by a
UEG-recovery prefactor ($\tanh^2 s$ at the GGA level; the DFS Eq. 12 form
$(x_2 + \tanh^2 x_3)$ at the meta-GGA level) and passed through a bounded squash with a static
limit (`networks._AlecLOB`, `networks.py:65`):

$$L(x) = \Lambda\,\sigma\!\left(x - \ln(\Lambda - 1)\right) - 1,$$

with $\sigma$ the logistic function; $\Lambda = 1.804 = 1 + \kappa_{\mathrm{PBE}}$ for the
GGA exchange ceiling; $\Lambda = 1.174$, the DFS exchange ceiling, for a meta-GGA exchange
network -- hardcoded in `networks.create_network_pair` (line 751), bypassing the registry's
1.804, and touched exactly by the parent, since SCAN's $F_x$ reaches exactly 1.174 at
$s = 0$, $\alpha = 0$, where the pre-image clamps; and $\Lambda = 2.0$ for the correlation
squash, whose purpose is non-negativity of $F_c$ (the DFS Eq. 13 transform), not a
Lieb-Oxford bound (`networks._AlecLOB` docstring). Writing
$T(g)$ for the gated network term on the descriptor row $g$:

* **Unanchored** (v3/v4/v5): $F = 1 + L(T(g))$. With the zero-initialized final layer the
  networks start at $T(g) = 0$, i.e. $F \equiv 1$ (the LDA limit), and pretraining moves the
  curve toward the parent.
* **Anchored** (v6): $F = 1 + L(z_{\mathrm{parent}} + T(g))$ with
  $z_{\mathrm{parent}} = L^{-1}(F_{\mathrm{parent}} - 1)$, so $T(g) = 0$ returns the parent
  exactly (Section 6.1).

![Bounded map and its pre-image. Left: $F = 1 + L(z)$ for the three limits in use, with the
pre-image clamp at $z = \pm 40$ marked. Right: the inverse $z = \ln[(\Lambda-1)F/(\Lambda-F)]$,
annotated with the distance from each bound inside which the clamp
binds.](figures_report_pretraining/bounded_map.png)

Both panels are drawn by calling `networks._AlecLOB` and `parents.lob_preimage` themselves
(`report_equation_figures.make_bounded_map`), so the figure cannot drift from the code. The
left panel is the constraint: whatever the MLP emits, $F$ stays inside $(0, \Lambda)$, and every
curve passes through $F = 1$ at $z = 0$ -- the property that makes a zero-initialized final layer
the LDA limit for an unanchored network and the parent for an anchored one. The right panel is
the price: the map is exponentially flat near both bounds, so a parent within
$\Lambda(\Lambda-1)e^{-40}$ of the ceiling or $\Lambda e^{-40}/(\Lambda-1)$ of the floor cannot
be represented by a finite pre-image and is clamped instead. The figure's own bind thresholds,
computed from that closed form and then checked against `lob_preimage` by requiring the clamp to
return exactly $\pm 40$, read (from `bounded_map.csv`, panel `c_bind`)
$8.678\times 10^{-19}$ / $2.866\times 10^{-17}$ at $\Lambda = 1.174$,
$6.162\times 10^{-18}$ / $9.532\times 10^{-18}$ at 1.804, and
$8.497\times 10^{-18}$ at both bounds of $\Lambda = 2.0$. **Conclusion:** the clamp is inert at
any physical parent value; the flatness it reflects is not, and Section 8.2 is about exactly
that flatness.

The parent is fixed per rung: PBE (Perdew, Burke and Ernzerhof, Phys. Rev. Lett. 77, 3865
(1996)) for GGA-rung architectures, SCAN for meta-GGA ones
(`parents.parent_for_arch`; `cluster/fidelity.resolve_parent` reads the same predicate).
Exchange is posed per spin channel on the doubled density, following the exact spin scaling
$E_x[\rho_\alpha, \rho_\beta] = (E_x[2\rho_\alpha] + E_x[2\rho_\beta])/2$ (Oliver and Perdew,
Phys. Rev. A 20, 397 (1979)); correlation is posed on the total density with $\zeta$, its
target ratio formed against the polarized PW92 baseline (Perdew and Wang, Phys. Rev. B 45,
13244 (1992)) the model itself multiplies (`parents.py` module docstring).

![Parent enhancement factors at libxc constants. Left: exchange, PBE against SCAN at
$\alpha = 0$ and $\alpha = 1$, with the two ceilings marked. Right: correlation relative to the
PW92 baseline at three densities, $\zeta = 0$.](figures_report_pretraining/parent_enhancement.png)

These are the curves every "distance from the parent" statement in this document is measured
against, evaluated through `parents.pbe_fx` / `pbe_fc` / `scan_fx` / `scan_fc` at libxc's own
constants. Read the left panel for the two exchange conventions: PBE rises monotonically from
$F_x(0) = 1$ toward its ceiling (`parent_enhancement.csv` gives $F_x = 1.72976$ at $s = 6$,
still below 1.804), SCAN at $\alpha = 1$ likewise starts at 1, while SCAN at $\alpha = 0$ --
the single-orbital branch -- *starts at its ceiling*, exactly 1.174. Read the right panel for
the correlation asymmetry that Section 8.2 turns on: PBE correlation is switched off by the
gradient, falling from $F_c \approx 1$ at $s = 0$ to $1.532\times 10^{-3}$ at $s = 6$ for
$r_s = 2$ (and to $7.56\times 10^{-4}$ at $r_s = 0.5$), whereas SCAN's single-orbital branch
retains a finite floor, $0.18019$ at the same point. **Conclusion:** the two parents differ
qualitatively in the large-gradient correlation limit, and the anchored parameterization's
sensitivity is proportional to the parent's own value there -- so the same anchor costs more
against PBE than against SCAN.

The correlation baseline itself carries a coordinate singularity worth stating, because it is
clipped rather than removed:

![PW92 spin interpolation $f(\zeta)$ and its curvature. Left: the interpolation, with the
production clip at $|\zeta| = 1 - 10^{-6}$ marked. Right: $f''(\zeta)$, analytic against a
central difference, on a log scale, with the pole resolved in the
inset.](figures_report_pretraining/zeta_pole.png)

$f(\zeta) = [(1+\zeta)^{4/3} + (1-\zeta)^{4/3} - 2]/(2^{4/3}-2)$ is the PW92 spin interpolation
(Perdew and Wang, eq. 9). It is smooth, but its second derivative diverges as
$|\zeta| \to 1$: `zeta_pole.csv` records $f''(0) = 1.7099209342$ -- reproducing
`parents._PW_MOD_FZ20` exactly, which the figure asserts as an oracle -- rising to
$8.550\times 10^{3}$ at the clip $|\zeta| = 1 - 10^{-6}$. A fully polarized spin channel
therefore has an unbounded correlation second derivative, and taking one through a full SCF
returns non-finite values; the production path clips $\zeta$ at $1 - 10^{-6}$
(`oneshot._ZETA_BOUNDARY_EPS`). The written $f$ and its analytic $f''$ are not independent
assertions: the figure reconstructs `parents._pw92_mod_eps` from them through the repository's
own $G(r_s)$ parametrizations and raises if the agreement exceeds $10^{-15}$
(`report_equation_figures._verify_f_spin`), and the drawn central difference tracks the analytic
form to $2.3\times 10^{-4}$ relative over $|\zeta| \leq 0.99$ -- the figure's own stated
agreement, dominated by the stencil's truncation error where the curvature is steepest; over the
flat interior $|\zeta| \leq 0.5$ the same comparison closes to $2.6\times 10^{-7}$.
**Conclusion:** the clip is a
numerical necessity of the baseline, not of the network, and it is the reason an anchored
correlation network must be polarization-aware (Section 6.1).

---

## 3. The descriptors

Beyond $(\rho, \sigma)$ for exchange and $(\rho, \zeta, \sigma)$ for correlation, an
architecture may append **descriptor** columns to both MLP inputs. Each descriptor is a
registered class in `xcquinox/alec/descriptors.py` implementing a `compute` method that returns
an $(N_{\mathrm{grid}}, n_{\mathrm{features}})$ block, and
`assemble_descriptor_features` (`descriptors.py:475-494`) concatenates the blocks left to right
in the architecture's declaration order. Column order and width are identical whether the block
is evaluated on the physical (total) density, which the correlation term consumes, or on the
symmetric doubled density $\mathrm{diag}(P_\sigma, P_\sigma)$ of one spin channel, which the
exact exchange spin scaling requires.

The five registered descriptors follow. Ordering within an architecture is not canonical -- the
`deep_combined` family declares `("dm_statistics", "cusp")` while every rung-3.5 and meta-GGA
family declares cusp first -- so the input-column layout differs between families and cannot be
assumed alphabetical.

### 3.1 `cusp` -- nuclear cusp proximity

`CuspDescriptor` (`descriptors.py:226`, `compute` at `:257`) supplies two bounded features per
grid point:

$$c_0(\mathbf{r}) = \exp\!\left(-2 Z_{\mathrm{near}}\, r_{\min}\right) \in [0, 1],
\qquad
c_1(\mathbf{r}) = \tanh\!\left[\tfrac{1}{5}\,\ln\!\left(\sum_A \frac{Z_A}{r_A} + 10^{-12}\right)\right]
\in (-1, 1),$$

with $Z_{\mathrm{near}}$ the charge of the nearest nucleus, $r_{\min}$ the distance to it, the
sum in $c_1$ over all nuclei, and the $10^{-12}$ guarding the logarithm at a vanishing weighted
sum (`features.py:329`). Under `log_transform=False` the logarithm is dropped and
$c_1 = \tanh(\tfrac{1}{5}\sum_A Z_A/r_A)$; the two are semantically distinct checkpoint
families, and the trained checkpoint's class record now states
`descriptor_log_transform` and refuses a load whose record and skeleton disagree on it
(`checkpoint_class.require_matching_log_transform`); a record written before the field
existed -- every cluster record at the time of writing -- is accepted exactly as
before, the check firing only when both sides state the flag. Measured consequence of
a silent cross-flag load, the case the record now refuses: the cusp $c_1$ column moves
by 0.51 on its bounded $(-1, 1)$ range at 0.3--4 bohr from an oxygen nucleus.

Which form a run uses is not a descriptor-local setting, and the routing matters later. The
architecture-level `descriptor_log_transform` reaches **two** distinct places. One is the MLP
coordinate transform of Section 3.7, which the DFS coordinate set bypasses entirely. The other
is this logarithm: `ArchitectureConfig.materialize_descriptors` (`config.py:351-368`) copies the
arch-level flag into the cusp descriptor's own `log_transform` kwarg whenever the feature spec
does not set it explicitly. That second path is live under *every* coordinate set, DFS included,
so a flag that is inert for a descriptor-free architecture is not inert for a cusp-carrying one.
Over a whole converged PBE H2O density at def2-svp (30,632 grid points) the two settings place
$c_1$ at $-0.026$ to $0.981$ and at $0.174$ to $1.000$ respectively, differing by up to $0.534$
point-wise, while $c_0$ is bitwise identical. Section 9.1 turns on exactly this distinction.

The rationale is the electron-nucleus cusp. Kato (*Commun. Pure Appl. Math.* 10, 151 (1957))
fixes the wavefunction condition $(\partial\langle\psi\rangle/\partial r)|_{r=0} = -Z\psi(0)$;
the corresponding spherically averaged density relation
$(\partial\langle\rho\rangle/\partial r)|_{r=0} = -2Z\rho(0)$ is due to Steiner (*J. Chem. Phys.*
39, 2365 (1963)). Column $c_0$ approximates the resulting Slater envelope $\exp(-2Zr)$ rather
than enforcing the condition; the docstring states this explicitly, and the feature should be
described as a proximity heuristic motivated by the cusp, not as a cusp constraint. Column $c_1$
is a log-compressed nuclear-attraction weight in the DFS convention, the $1/5$ scaling chosen on
a dynamic-range argument (`xcquinox.features.compute_cusp_descriptor`). Both are pure geometry:
they do not depend on the density matrix, and they are exact under the doubled-spin
substitution, which is why the cusp architectures' pretraining offsets of Section 4.3 are *not*
spin-scaling defects.

**Carried by:** `deep_cusp`, `deep_cusp_attn`, `deep_cusp_3x16` (cusp alone); the four
`deep_combined*` forms; both `deep_rung35*` forms and `deep_rung35ms_3x16`; and **three of the
five** meta-GGA forms (`deep_cusp_mgga_3x16`, `deep_rung35_mgga_3x16`,
`deep_rung35ms_mgga_3x16`; `deep_mgga_3x16` and `deep_mgga_attn_3x16` carry the indicator
alone).

### 3.2 `dm_statistics` -- global density-matrix indicators

`DMStatisticsDescriptor` (`descriptors.py:262`, `compute` at `:315`) supplies two features
(`features.compute_dm_features_array`, `features.py:109-181`):

$$\mathrm{idem} = \frac{\lVert PSP - P\rVert_F^2}{N_{\mathrm{pair}} + 10^{-12}},
\qquad
\mathrm{offdiag} = \frac{\lVert D - \mathrm{diag}(D)\rVert_F}{\mathrm{Tr}\,D + 10^{-12}}.$$

Two details of the first are load-bearing. The operand is the **pair** density matrix
$P = D/2$, normalized by the **electron-pair** count $N_{\mathrm{pair}} = \mathrm{Tr}(PS) = N_e/2$,
which is what makes $PSP = P$ the idempotency condition at all; on an open-shell UKS row the
feature is instead the per-spin mean $\tfrac{1}{2}[\mathrm{idem}(P_\alpha) + \mathrm{idem}(P_\beta)]$
(`features.py:113-127`). And the norm is **squared**: $\lVert X\rVert_F$ is not differentiable
at $X = 0$, which is identically where every converged density sits, and autodiff there returned
$-2.08\times 10^{-3}$ against a finite difference of $+4.5\times 10^{-9}$; the squared form is a
polynomial with the same zero set and restores agreement to $6.5\times 10^{-15}$.

Only the first feature vanishes on a single determinant. `idempotency_error` is zero to
round-off at any exact Hartree-Fock or Kohn-Sham reference and grows with departure from a
single Slater determinant, but `off_diag_norm` does **not** -- it measures AO-basis
off-diagonality, which a single determinant has in abundance. Measured on a converged
closed-shell PBE H2O reference (def2-svp), whose $\lVert PSP - P\rVert_F$ is
$1.15\times 10^{-15}$: `idempotency_error` reads $2.65\times 10^{-31}$ while `off_diag_norm`
reads **0.2714**. The second feature is a bonding-structure indicator, not a correlation
indicator that vanishes at the mean-field limit.

Two properties must be stated with the definition. First, these are **global, per-molecule
scalars**, `jnp.tile`d identically to every grid point and fed into a per-point (semilocal)
enhancement factor -- so the exchange-correlation energy density at a point in fragment A shifts
when a distant fragment B is added. The size-consistency and locality caveat is recorded in the
class docstring (`:288-296`) and remains open; the rung-3.5 descriptors below are the leak-free
members of the same family. Second, a third feature, `dm_entropy`, was **removed on 2026-08-06**
(width 3 to 2): it had no usable gradient at any converged density, because the physical-bounds
clip put every natural occupation on a boundary and, without the clip, the eigenvector
derivatives of `eigh` are ill-defined on the degenerate occupation spectrum of an idempotent
density matrix. No spectral invariant can replace it -- for a single determinant the eigenvalues
of $DS$ are exactly $\{2, \ldots, 2, 0, \ldots, 0\}$, so any function of the spectrum alone is
constant on the idempotent manifold. Removing it took the energy/potential finite-difference
residual of the dm_statistics architectures from $1.04\times 10^{-2}$ to $2.1\times 10^{-10}$
under the committed test's own ordering, the dead gradient having dominated it
(`descriptors.py:275-286`; `features.py:130-155`;
`notebooks/analysis/DM_DESCRIPTOR_SPEC.md`).

**Carried by:** `deep_dm`, `deep_dm_attn`, `deep_dm_3x16` (alone) and the four `deep_combined*`
forms (with cusp). No meta-GGA architecture carries it.

### 3.3 `rung35` -- localized density-matrix occupancy

`DMRung35Descriptor` (`descriptors.py:320`, `compute` at `:366`; kernel
`xcquinox/alec/rung35.py:96`) is the bounded local occupancy of Janesko's unified rung-3.5 /
DFT+U formalism (arXiv:2206.07118, Eqs. 12-13; M11plus, Verma et al., *J. Chem. Theory Comput.*
15, 4804 (2019)):

$$n_\sigma(\mathbf{r}_m) = \sum_i \left|\langle \psi_{i\sigma} \,|\, \phi^G_{\mathbf{r}_m}\rangle\right|^2
= A(\mathbf{r}_m)^{\mathsf T} P^\sigma A(\mathbf{r}_m) \in [0, 1],$$

with an $L^2$-normalized Gaussian projector centred at the grid point,
$\phi^G_{\mathbf{r}_m}(\mathbf{r}) = (2\alpha/\pi)^{3/4}\exp(-\alpha|\mathbf{r}-\mathbf{r}_m|^2)$,
and the projected-AO overlap vector
$A_\mu(\mathbf{r}_m) = \langle \chi_\mu | \phi^G_{\mathbf{r}_m}\rangle$. The two features are
the $\alpha$- and $\beta$-spin occupancies, and they feed **both** networks: the M11plus rung-3.5
ingredient is a correlation ingredient, so the C-net is a first-class consumer, and the X-net
receives it equally through the shared extras mechanism.

The construction is what makes the descriptor tractable. $A_\mu$ depends only on the basis, the
grid and the *fixed* width $\alpha$ -- not on the density matrix -- so it is a precomputed
constant (a plain PySCF overlap integral) that is never differentiated, and the occupancy is a
single einsum, linear and differentiable in the *live* density matrix. The default width is
`DEFAULT_RUNG35_ALPHA = 0.2` $a_0^{-2}$ (`rung35.py:39`), which the repository states as
grounded at the M11plus kernel scale $d^2 = 5\,a_0^2$; that attribution is carried by the
repository itself and is still to be confirmed against the library copy of the paper
(`CAMPAIGN_V6.md:229-230`), so it is reported here as the repo record rather than as a read of
Verma et al. Boundedness in $[0,1]$ follows from Bessel's inequality -- $P^\sigma$ is
positive semidefinite, and orthonormal occupied orbitals against a normalized projector give the
upper bound -- so the feature is NaN-safe by construction. Unlike `dm_statistics` this is a
genuine per-grid-point contraction of the non-local one-particle density matrix, hence
size-intensive and leak-free, and it is self-consistent: a functional of the live density
matrix, recomputed each SCF cycle under the `REASSEMBLE` policy, never a static reference. It
carries no $\tau$ and is therefore not a meta-GGA -- its own rung, between meta-GGA and hybrid.

**Carried by:** `deep_rung35_3x16` and `deep_rung35_attn_3x16` (with cusp),
`deep_rung35only_3x16` (alone), and `deep_rung35_mgga_3x16` (with cusp and the indicator).

### 3.4 `rung35_multishell` -- the radial generalization

`DMRung35MultishellDescriptor` (`descriptors.py:371`, `compute` at `:428`; kernel
`rung35.py:156`) evaluates the same occupancy at several projector widths,

$$n_\sigma(\mathbf{r}; w) = A_w(\mathbf{r})^{\mathsf T} P^\sigma A_w(\mathbf{r}) \in [0, 1],$$

with `DEFAULT_RUNG35_MULTISHELL_ALPHAS = (0.05, 0.2, 0.8)` $a_0^{-2}$ (`rung35.py:130`) --
the M11plus scale bracketed by a factor of about four either side, so the set gives a coarse
radial profile of the one-particle density matrix around each point. The feature count is
$2\,\times$ the number of widths, six by default, ordered **alpha-major then spin**; setting a
single width reproduces `rung35` bitwise, and the constructor enforces the count relation
(`:413-418`).

This is the radial part of the localized density-matrix projection used by NeuralXC (Dick and
Fernandez-Serra, *Nat. Commun.* 11, 3509 (2020)) and carried in the DFS reference
implementation, which projects the density matrix onto a localized basis and contracts the
coefficients into rotationally invariant per-shell norms. The stated limitation belongs with the
definition: `fakemol_for_charges` builds s-type projectors only, so only the $l = 0$ channels
exist, and with one $m$ per shell the invariant $\sqrt{\sum_m c_{nlm}^2}$ collapses to the
occupancy itself. Angular channels require solid-harmonic fakemols and are not implemented, so
this should not be described as "the DFS descriptor" (`descriptors.py:383-389`).

**Carried by:** `deep_rung35ms_3x16` (with cusp) and `deep_rung35ms_mgga_3x16` (with cusp and
the indicator).

### 3.5 `metagga` -- the SCAN iso-orbital indicator

`MetaGGAAlphaDescriptor` (`descriptors.py:433`, `compute` at `:471`) supplies one feature, the
iso-orbital indicator introduced by SCAN (Sun, Ruzsinszky and Perdew, Phys. Rev. Lett. 115,
036402 (2015), Eq. 2) and reused by DFS (Eq. 6):

$$\alpha = \frac{\tau - \tau_W}{\tau_{\mathrm{unif}}},
\qquad
\tau(\mathbf{r}) = \tfrac{1}{2}\sum_{\mu\nu} P_{\mu\nu}\,\nabla\chi_\mu \cdot \nabla\chi_\nu,
\qquad
\tau_W = \frac{|\nabla\rho|^2}{8\rho},
\qquad
\tau_{\mathrm{unif}} = \tfrac{3}{10}(3\pi^2)^{2/3}\rho^{5/3},$$

with $\alpha = 1$ the uniform gas and $\alpha = 0$ a single orbital. The kinetic-energy density
is a *linear* contraction of the live density matrix against AO gradients already on the grid
(`metagga.compute_tau_from_dm`, `metagga.py:115`), so -- exactly like the rung-3.5 occupancy --
the descriptor is self-consistent, differentiable through the SCF, and needs no new integrals,
no Laplacian and no `deriv=2`.

The stored column is not $\alpha$ raw but

$$\min\!\big(p(\alpha_{\mathrm{raw}}),\, 100\big),
\qquad
p(x) = \tfrac{1}{2}\left(x + \sqrt{x^2 + w^2}\right), \quad w = 10^{-5},$$

with $p$ the smooth positive part (`metagga.smooth_positive_part`, `:142`;
`_ALPHA_SMOOTHING_WIDTH = 1e-5` at `:104`) and the ceiling `_ALPHA_MAX = 100` at `:61`. The
lower bound $\alpha \geq 0$ is the von Weizsacker inequality, exact on every positive
semidefinite density matrix, so a negative raw value is rounding; the smoothing exists because a
hard clip $\max(\alpha_{\mathrm{raw}}, 0)$ has a one-sided derivative exactly on the manifold
every one-electron spin channel occupies. Both regularizations are derived in Section 6.2 and
Section 6.5 and are not repeated here.

![Iso-orbital indicator: smooth floor and hard ceiling. Left: the stored $\alpha$ against
$\tau/\tau_{\mathrm{unif}}$ at $\rho = 1$ and $s = 1$, with the single-orbital and uniform-gas
points marked and the smooth floor resolved in the inset. Right: the same on log axes, showing
the ceiling truncating the low-density
tail.](figures_report_pretraining/alpha_indicator.png)

Both panels call `metagga.compute_alpha` directly. Read the left panel for the two physical
anchors: at $\tau = \tau_W$ the indicator goes to its smoothed floor rather than to zero
(`alpha_indicator.csv` gives $5\times 10^{-6} = w/2$ at raw indicator 0), and at
$\tau = \tau_W + \tau_{\mathrm{unif}}$ it reads $1.000000000025$ -- exactly $1 + w^2/4$, the
smoothing's own second-order offset at the uniform gas, which is the scale at which the
regularization is visible at all. Read the right panel for the ceiling: over the four decades of
raw indicator the panel draws ($0.1$ to $1000$), the stored column tracks the raw value for the
first three -- to $\alpha = 98.5$ -- and then saturates, 151 of the panel's 601 points sitting
at the ceiling (`alpha_indicator.csv`, panel `b_ceiling`). **Conclusion:** the column is the
raw indicator everywhere the physics is resolved and departs from it only in two regimes -- one
below the numerical noise floor of the $\tau - \tau_W$ cancellation, one on the low-density tail
where the indicator diverges. Section 6.5 prices both.

![Smooth positive part at $w = 10^{-5}$. Left: $p(x)$ against the hard clip, with the excess
over the clip and its $w^2/4|x|$ asymptote in the inset. Right: the exactness of the stored-column
inversion, against a first-order conditioning
scale.](figures_report_pretraining/smooth_positive_part.png)

The left panel shows what the smoothing does: $p$ equals $\max(x, 0)$ to within $w^2/4|x|$ for
$|x| \gg w$, sits at $w/2$ with slope $1/2$ at the origin, is strictly positive everywhere, and
satisfies $p(x) - p(-x) = x$ to one unit in the last place, so a central difference across zero
reproduces its derivative. The right panel is what makes the anchored construction of Section 6 work at all:
the stored column can be inverted, $x = p - w^2/(4p)$
(`metagga.invert_smooth_positive_part`, `:155`), and the round trip closes to
$2.778\times 10^{-19}$ absolute over the plotted grid ($8.470\times 10^{-15}$ relative), from
`smooth_positive_part.csv`. **Conclusion:** below the ceiling the smoothing is invertible to
round-off, so an anchored network recovers the exact indicator its parent needs; at the ceiling
it is not, and that single fact is the whole content of Section 6.5.

One documented deviation belongs here. DFS feeds its network the log-transformed coordinate
$x_3 = \ln((\alpha+1)/2)$ (its Eq. 10); in the legacy coordinate family the *raw* clamped column
is the MLP input and $x_3$ enters only through the UEG-recovery gate. Under the DFS coordinate
family (Section 3.7) the network input is $x_3$ itself. The deviation is recorded, not silently
carried (`descriptors.py:450-452`).

**Carried by:** all five meta-GGA architectures.

### 3.6 The architecture-to-descriptor map

The registry holds 31 architectures. Grouped by descriptor set, with `config.py` line numbers
(re-read from the current file):

| descriptor set | count | architectures (line) |
|---|---|---|
| none | 12 | `shallow` (506), `shallow_attn` (507), `medium` (508), `medium_attn` (509), `deep` (514), `deep_attn` (518), `deep_notransform` (560), `deep_notransform_attn` (564), `deep_3x16` (573), `deep_attn_3x16` (577), `deep_notransform_3x16` (675), `deep_notransform_attn_3x16` (679) |
| `("cusp",)` | 3 | `deep_cusp` (523), `deep_cusp_attn` (528), `deep_cusp_3x16` (582) |
| `("dm_statistics",)` | 3 | `deep_dm` (534), `deep_dm_attn` (539), `deep_dm_3x16` (587) |
| `("dm_statistics", "cusp")` | 4 | `deep_combined` (545), `deep_combined_attn` (550), `deep_combined_3x16` (592), `deep_combined_attn_3x16` (597) |
| `("cusp", "rung35")` | 2 | `deep_rung35_3x16` (609), `deep_rung35_attn_3x16` (614) |
| `("cusp", "rung35_multishell")` | 1 | `deep_rung35ms_3x16` (624) |
| `("rung35",)` | 1 | `deep_rung35only_3x16` (628) |
| meta-GGA-bearing | 5 | `deep_mgga_3x16` (641) `("metagga",)`; `deep_mgga_attn_3x16` (646) `("metagga",)`; `deep_rung35_mgga_3x16` (652) `("cusp", "rung35", "metagga")`; `deep_cusp_mgga_3x16` (664) `("cusp", "metagga")`; `deep_rung35ms_mgga_3x16` (669) `("cusp", "rung35_multishell", "metagga")` |

The twelve descriptor-free entries span the whole shape axis, which is what makes the size
ladder of Section 9.1 a controlled comparison: `shallow` and `shallow_attn` are 2x8, `medium`,
`medium_attn`, `deep_3x16` and `deep_attn_3x16` are 3x16, and `deep` and `deep_attn` are 4x32,
with attention heads 2 for `shallow_attn` and 4 for every other attention form. One registry
inconsistency is worth recording because it looks material and is not: `deep_rung35ms_3x16`
(line 624) is the only `from_spec` entry that omits `dm_entropy_intensive=True` and so takes the
default `False`, while its meta-GGA sibling at line 669 sets it. The asymmetry is inert for a
stronger reason than descriptor coverage: **the flag has no functional consumer anywhere in the
package.** A quote-agnostic sweep returns 48 occurrences, and every one is a declaration
(`config.py:131`), a type validation (`:208`), a `from_spec` pass-through (`:386`, `:458`), a
registry literal, or a test; `DMStatisticsDescriptor` accepts no `intensive` argument at all,
and the only hits outside `config.py` and the test tree are two comment strings in notebook
tooling. The field survives solely in the serialized architecture record through `describe()`,
so it can change a spec's SHA-256 digest and nothing else. It is a vestige of the removed
`dm_entropy` feature (Section 3.2), and the naming asymmetry is an editing artifact.

### 3.7 The coordinate transforms

Which coordinates the MLPs read is a per-architecture, per-run switch, and the two settings are
separate checkpoint families. The exchange network branches at `networks.py:274-281` and the
correlation network at `:556-583`.

Under `descriptor_coordinates: dfs` (every v6 run; `model.descriptor_coordinates: dfs` in the
YAML, recorded in each trained checkpoint's class record) the inputs are DFS's own, implemented
at `networks.py:27-50`:

$$x_s = \left(1 - e^{-s^2}\right)\ln(s + 1),
\qquad
x_\alpha = \ln\!\left(\frac{\alpha + 1}{2}\right),$$
$$x_0 = \ln\!\left(\rho^{1/3} + 10^{-5}\right),
\qquad
x_1 = \ln\!\left[\tfrac{1}{2}\left((1+\zeta)^{4/3} + (1-\zeta)^{4/3}\right)\right].$$

These are DFS Eqs. 9, 10, 7 and 8 in turn: $x_s$ is `_dfs_log_transform` (`:30`), $x_\alpha$ is
`_dfs_indicator_coordinate` (`:37`), and $x_0$ and $x_1$ are inlined in the correlation network
at `:574` and `:575`. Note that $x_1$ as written -- the *logged* spin-scale factor -- is DFS
Eq. 8; Eq. 4 is the same factor unlogged, which is the form the legacy family feeds (below). The
in-code comment at `networks.py:563` labels the logged form "eq. 4", which is the narrower
citation.
The offset $10^{-5}$ inside the density logarithm is DFS's own (`_DFS_LOG_EPS`, `:27`, from
dpyscfl `net.py` line 39, `self.loge = 1e-5`). The exchange MLP reads $x_s$ alone at the GGA
level and $x_s$ with
$x_\alpha$ at the meta-GGA level; the correlation MLP reads $(x_0, x_1, x_s)$ plus extras, with
$x_\alpha$ substituted for the raw indicator column among the extras on the meta-GGA rung. The
$x_\alpha$ substitution operates on the **raw** indicator recovered from the stored column by
`networks._raw_indicator` (`:44`), which is why the invertibility measured in Section 3.5 is
load-bearing.

A third documented deviation sits at the $x_s$ line itself (`networks.py:576`), beside the
Eq. 7 deviation of the legacy C-net and the Eq. 10 deviation of the legacy indicator input: the
reduced gradient entering $x_s$ is that of the **total** density, with no spin rescaling. The
reference implementation's unpolarized branch divides $s$ by
$(1+\zeta)^{2/3} + (1-\zeta)^{2/3}$ before the transform, and its own source marks that line as
inherited from xcdiff rather than from dpyscfl (`net.py:202`, comment "line below in xcdiff, not
dpyscfl"); the repository does not apply it, and records the choice at `networks.py:565-566`.
The epsilon asymmetry between the coordinates is *not* a deviation: the reference applies its
`loge` offset only inside the density logarithms (`net.py:187-190`, `:226-229`) and writes the
gradient coordinate as `log(descr3 + 1)` with no offset (`net.py:198`, `:204`), exactly as
`_dfs_log_transform` does.

Under the legacy family (`descriptor_log_transform: true`, every run through v5) the exchange
network applies the same $\{1 - e^{-x^2}\}\ln(x+1)$ form to $s$, but the correlation network
carries a **documented deviation from DFS Eq. 7**, stated at `networks.py:540-548`: its density
feature is $r_s$ rather than DFS's $\rho^{1/3}$, and it is passed through the *s*-style
transform (the reduced-gradient form of Eq. 9) rather than the plain logarithm Eq. 7 applies to
the density variable. The deviation is recorded and not changed, since a plain-log density form
would invalidate every existing checkpoint. In that family $\zeta$ enters through the bounded
feature $x_1 = \tfrac{1}{2}[(1+\zeta)^{4/3} + (1-\zeta)^{4/3}]$ without the outer logarithm,
which equals 1 at $\zeta = 0$ so that a closed-shell call sees the unpolarized input. Under
`descriptor_log_transform: false` (the `deep_notransform*` entries) both variables are fed raw.

This distinction is not cosmetic for the cross-campaign comparison of Section 9: the anchored v6
`medium` cells and the unanchored v4 `deep_3x16` cells differ operatively by the anchor **plus**
the legacy-to-dfs coordinate change, and by nothing else (Section 9.1).

---

## 4. v4: unanchored pretraining to parent enhancement values

### 4.1 Protocol

The v4 pretraining fitted each network pair, per architecture, to the parent's enhancement
factors point-wise on stored reference rows:

* **Data.** Self-consistent PBE atomic densities of the training pool's elements -- for
  the v4 arms H, Li, C, N, O, F and Na at their ground-state spins, with He dropped as
  absent from the production basis (`hpcjobs/configs/dfs_step7.dfs6311_grid3_v4gga.yaml`,
  `pretrain.atoms`; the library default `pretrain_data_gen.DEFAULT_PRETRAIN_ATOMS` is the
  smaller H/He/O/N set) -- on their Becke grids, with libxc's parent values stored as
  clipped ratio-minus-one targets. For architectures whose descriptor set is
  exactly the meta-GGA indicator, a synthetic regular $(r_s, s, \alpha)$ mesh
  ($7 \times 8 \times 10$ nodes; `pretrain_data_gen.MESH_RS = (0.1, 0.3, 0.7, 1.5, 3.0, 5.0,
  10.0)`, `MESH_S = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)`, `MESH_ALPHA = (0.0, 0.1,
  0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)`) was added on 2026-08-10 at a stated
  `MESH_WEIGHT_FRACTION = 0.3` share of the loss weight, each node realized as a physical
  $(\rho, \sigma, \tau)$ triple ($\rho = 3/(4\pi r_s^3)$, $\sigma = (2 s k_F \rho)^2$,
  $\tau = \alpha\,\tau_{\mathrm{unif}} + \tau_W$) and evaluated through the same libxc calls
  as the atomic rows. The mesh exists because SCAN is three-dimensional in
  $(r_s, s, \alpha)$ and the atomic manifolds leave the $\alpha$ axis underdetermined: the
  pretrained-only meta-GGA network scored 42.65 kcal/mol on held-out reactions against SCAN's
  own 4.45 before the mesh, and the meta-GGA C-net was measured up to 0.457 from SCAN away
  from $\alpha = 1$ (HISTORY 2026-08-10; `pretrain.py` lines 1275-1277, the mesh-append
  rationale). Geometry-bearing
  descriptors (cusp, rung-3.5) cannot be defined at a geometry-free mesh node, so those
  meta-GGA stacks pretrained atoms-only with the caveat on record.
* **Objective.** Integration-weighted mean-squared enhancement-factor residual (quadrature
  weight times $|\rho\,\epsilon^{\mathrm{LDA}}|$), 2500 steps
  (`dfs_step7/dfs6311_grid3_v4gga/runs/run_20260810T202813Z/pretrain/deep_3x16/pretrain_metadata.json`:
  `pretrain_steps = 2500`, `parent_anchor` absent). No per-system energy term, no held-out
  validation, no acceptance gate of any kind.
* **Target policy.** Each architecture pretrained to the best functional its inputs can
  exactly represent: PBE for the GGA and rung-3.5 forms (faithful at $\leq 0.013$ in $F$),
  SCAN for the indicator-bearing forms (HISTORY 2026-08-10, the v4 sweep definition).

![The synthetic SCAN pretraining mesh. Left: the $(s, \alpha)$ projection of all 560 nodes,
coloured by $r_s$ and dodged in $s$ for visibility. Right: the ten $\alpha$ nodes against the
stored column's ceiling.](figures_report_pretraining/dfs_mesh.png)

The mesh is enumerated from `pretrain_data_gen.MESH_RS`, `MESH_S` and `MESH_ALPHA` themselves;
`dfs_mesh.csv` records the 560 nodes ($7 \times 8 \times 10$) and their coordinates. Read the
left panel for what the mesh buys: the atomic rows of a real molecule trace a one-dimensional
manifold through $(r_s, s, \alpha)$, whereas the mesh covers the space as a product grid, so the
$\alpha$ axis is determined independently of the density profile that generated it. Read the
right panel for what it does not cover: the largest $\alpha$ node is 5, while the stored column
runs to `_ALPHA_MAX = 100`, leaving a factor of 20 in $\alpha$ unsampled. **Conclusion:** the
mesh regularizes the indicator dependence over the physically populated range and says nothing
about the low-density tail -- which is exactly why the mesh contributes nothing to the anchored
pretraining floor of Section 6.5, where the entire residual is carried by capped tail rows.

### 4.2 What the handoff actually delivered

Measured from the pretrained checkpoints of the v4 GGA arm
(`figures_dfs_step7_dfs6311_grid3_v4gga_val_best/pretrain_fx_fc_curves.csv`): the
pretrained exchange curves sit $\max|\Delta F_x| = 0.039$ (deep_3x16) to
$0.090$ (deep_rung35_3x16) from PBE over the plotted grid; the meta-GGA correlation nets sit
up to $0.49$--$0.52$ from SCAN
(`figures_dfs_step7_dfs6311_grid3_{v4,v5}_val_best/pretrain_fx_fc_curves.csv`,
`.../v5mgga2/pretrain_fx_fc_curves.csv`). The final point-wise exchange loss of the v4
deep_3x16 pretrain was $4.6\times 10^{-5}$ (same metadata file) -- for comparison, the v6
anchored runs *start* at $2.7\times 10^{-32}$ (Section 6.5).

![Unanchored (v4gga) pretrained enhancement factors against PBE: the difference panels for all
six GGA architectures, $F_x$ against $s$ and $F_c$ against $s$ at three
densities.](figures_dfs_step7_dfs6311_grid3_v4gga_val_best/pretrain_fx_fc_delta_all.png)

Each curve is a trained-from-flat network minus its parent, evaluated through the production
model builder against `parents.pbe_fx` / `pbe_fc`. Read it as the state in which each
architecture *entered* its fine-tune. The per-architecture worst deviations, recomputed from
`pretrain_fx_fc_curves.csv`, are $|\Delta F_x|$ of $3.899\times 10^{-2}$ (deep_3x16),
$4.259\times 10^{-2}$ (deep_attn_3x16), $5.659\times 10^{-2}$ (deep_rung35ms_3x16),
$6.287\times 10^{-2}$ (deep_cusp_3x16), $8.503\times 10^{-2}$ (deep_rung35_attn_3x16) and
$8.983\times 10^{-2}$ (deep_rung35_3x16), with the correlation channel reaching
$1.517\times 10^{-1}$ (deep_rung35ms_3x16). **Conclusion:** a 2500-step point-wise fit leaves
the descriptor-carrying architectures two to four times further from the parent than the
descriptor-free ones, and the ordering of the curve metric does not predict the energy offsets
of the next paragraph. The per-architecture panels of the same set
(`pretrain_fx_fc_deep_*.png` in that directory) show each pair separately; they carry no
information the difference panels omit and are not reproduced here.

In energy units, the 2026-08-20 probe (frozen parent densities, production evaluation path,
every value re-derived independently; HISTORY 2026-08-20, "Pretraining does not deliver the
parent") measured atomization-energy offsets from the parent on H2O / N2 / CH4 at pretraining
handoff of $-2.5 / -4.2 / -2.4$ kcal/mol (deep_3x16) and $-2.3 / -4.1 / -3.1$
(deep_attn_3x16) for the descriptor-free networks, against $-13.2 / -4.2 / -25.7$
(deep_cusp), $-13.5 / -3.5 / -29.1$ (deep_rung35), $-29.5 / -20.4 / -56.1$
(deep_rung35_attn) and $-22.0 / -30.9 / -42.8$ (deep_rung35ms). The separation is in the
**worst system per architecture**, not uniformly system by system: each descriptor-carrying
architecture entered its 200-epoch fine-tune 25.7 to 56.1 kcal/mol from its parent on its worst
of the three, against 4.1--4.2 for the two descriptor-free controls. Individual systems do
overlap -- deep_rung35's N2 offset of 3.5 sits inside the controls' own 2.3--4.2 span -- so the
claim is about the worst case an architecture carries into training, on the reaction
type it would be scored on. The pretraining loss did not see it -- the architecture with the
lowest exchange residual carried the largest offset -- because a point-wise objective does not
control the integral (HISTORY 2026-08-20; `SPEC_pretrain_fidelity_program.md` Section 2;
`NOTES_v5_mgga_vs_scan.md:192-211`).

### 4.3 The two defects behind the offsets

Two distinct mechanisms, both surfaced on 2026-08-20, ended the v4/v5 record for the affected
architectures (HISTORY 2026-08-20, Phase 39):

1. **The open-shell spin-scaling defect (meta-GGA architectures).** The production UKS
   exchange applied the Oliver-Perdew scaling by doubling $\rho$ and quadrupling $\sigma$ per
   spin channel but passed the *total-density* iso-orbital indicator unchanged into both
   channels, in the energy and in the SCF potential -- so every open-shell species of every
   meta-GGA architecture was trained and evaluated on a functional different from its own
   closed-shell definition. The pretrained meta-GGA pair over-bound H2O / N2 / CH4 by 30.5 /
   55.9 / 20.8 kcal/mol relative to SCAN; transforming only the indicator
   ($\alpha_\sigma$ from $2\rho_\sigma, 4\sigma_{\sigma\sigma}, 2\tau_\sigma$, which libxc's
   spin-polarized SCAN satisfies to $<10^{-12}$ Ha) recovered 75, 86 and 63 percent of the effect on H2O, N2 and CH4 respectively (fractions from NOTES_v5_mgga_vs_scan.md; the residual offsets -7.6 / -7.9 / -7.6 kcal/mol). The
   superseded two-block evaluation costs $-30.1$ kcal/mol on the O atom alone for SCAN
   exchange, and O anchors atomization energies throughout the pools. A secondary
   contribution came from the pretraining rows themselves: open-shell atoms stored
   spin-resolved SCAN targets against total-density inputs (+1.0 / +7.3 / $-2.0$ kcal/mol of
   the offsets). (HISTORY 2026-08-20; the repair is Phase 43.)
2. **The pretraining-fidelity gap (GGA descriptor carriers).** The cusp and rung-3.5 offsets
   are *not* spin-scaling defects -- the cusp feature is pure geometry and exact under the
   doubled-spin substitution (Section 3.1), and the rung-3.5 occupancy has no in-domain
   doubled-spin evaluation. They decompose into an H-atom pretraining error shared by every cusp-carrying
   network (+13.7 mHa against +0.8) multiplied by the hydrogen count, plus molecular
   extrapolation of density-matrix features never constrained by the atoms-plus-mesh
   pretraining set (HISTORY 2026-08-20).

No campaign stage had ever compared a pretrained network with its parent on atoms or
atomization energies; the offsets were larger than the architecture differences the campaigns
existed to resolve, and this held while the campaigns ran. Every in-flight training was
cancelled on 2026-08-20 and the v4/v5 descriptor-architecture record retired (HISTORY
2026-08-20).

---

## 5. v5: per-rung self-consistent seeding

v5 changed where the truncated training SCF *starts*, not how the networks are pretrained
(HISTORY Phase 37, 2026-08-14):

* **What changed.** `precompute_fixed_density_data` gained a `dm_seed` channel; meta-GGA-family
  architectures train and evaluate their truncated (3-cycle) SCF from a converged SCAN density
  matrix instead of converged PBE, matching each rung's own baseline (the DFS reference work
  prepares per-campaign converged PBE-or-SCAN baselines the same way). A fourth held-out
  evaluation channel (`eval_holdout_coldstart`: functional-free minao seed, 25 cycles,
  `conv_tol` $10^{-12}$) was added as a trajectory diagnostic (Section 1.4). The v5 arm YAMLs are
  byte-derived from v4 with exactly five deltas (seed source, seed cache, coldstart flag,
  output root, eval wall); losses, solver, subsets, hyperparameters and the pretrain block are
  untouched, and the shared pretrain data file could not regenerate, so the v5 pretrained
  pairs replicate the v4 protocol and land at the same distances from the parent (the v4 and
  v5 meta-GGA pretrain curve CSVs carry identical deviations,
  `figures_dfs_step7_dfs6311_grid3_{v4,v5}_val_best/pretrain_fx_fc_curves.csv`).
* **What it fixed.** The density-footing asymmetry: previously every rung's truncated SCF was
  seeded from PBE's fixed point while SCAN comparison numbers used converged SCAN, a handicap
  concentrated on the beyond-GGA arms.
* **What it did not fix.** Neither defect of Section 4.3. The 2026-08-20 read of the
  SCAN-seeded cells against SCAN (HISTORY 2026-08-20; `notebooks/analysis/NOTES_v5_mgga_vs_scan.md`)
  found the cells reproducing SCAN's held-out accuracy at subset sizes 2--5
  ($E/E_{\mathrm{SCAN}}$ 0.94--1.01) with one of 21 (leg, subset) cells beating SCAN, and
  identified the limiting factor as the fine-tuning regime itself: both parents' BH76 error is
  ~90% bias (PBE $-7.5$, SCAN $-6.0$ kcal/mol of MAE 7.7 / 6.4); fine-tuning removes the bias
  and injects reaction-level scatter nearly uncorrelated with the parent's residual
  (root-mean-square difference from SCAN 8.6--11.0 kcal/mol, correlation of the two error
  vectors 0.21--0.33 at subset sizes 2--5, `NOTES_v5_mgga_vs_scan.md:68-72`) -- a property of
  the recipe, not of the seeding. The same read exposed that the cells never
  started at SCAN at all (Sections 4.2-4.3), which retired the v5 meta-GGA record as a
  quantitative result. Its trained enhancement-factor curves
  (`figures_dfs_step7_dfs6311_grid3_v5_val_best/trained_fx_fc_*`) document a retired record
  and are kept for the development narrative.

![Unanchored meta-GGA pretrained factors against SCAN, v4/v5 arm: `deep_mgga_3x16`,
`deep_mgga_attn_3x16` and `deep_rung35_mgga_3x16`, with the exchange and correlation panels at
the SCAN indicator slices $\alpha = 0$ and
$\alpha = 1$.](figures_dfs_step7_dfs6311_grid3_v4_val_best/pretrain_fx_fc_delta_all.png)

The v4 and v5 pretrained networks are the same objects -- the shared pretraining data file could
not regenerate between the arms -- and the two directories' `pretrain_fx_fc_curves.csv` carry
bit-identical deviations, which is the check that establishes it: $\max|\Delta F_c|$ reads
$9.4367\times 10^{-2}$ (deep_mgga_3x16), $4.6973\times 10^{-2}$ (deep_mgga_attn_3x16) and
$4.8980\times 10^{-1}$ (deep_rung35_mgga_3x16) in *both* files, with $\max|\Delta F_x|$ at
$2.0487\times 10^{-2}$, $1.7658\times 10^{-2}$ and $6.7213\times 10^{-2}$. The v5 set's own
difference panel is therefore not reproduced here: it is the same figure. **Conclusion:** the
correlation channel of the rung-3.5-stacked meta-GGA form is half an enhancement factor away
from SCAN at handoff -- five times the worst GGA-arm exchange deviation of Section 4.2 -- and no
part of the v5 change touched it.

![Unanchored meta-GGA pretrained factors, the v5mgga2 stacking arm: `deep_cusp_mgga_3x16` and
`deep_rung35ms_mgga_3x16` against
SCAN.](figures_dfs_step7_dfs6311_grid3_v5mgga2/pretrain_fx_fc_delta_all.png)

This arm never produced trained weights: no `model*.eqx` file exists anywhere under
`dfs_step7/dfs6311_grid3_v5mgga2/` in the pulled tree, and the directory accordingly carries
`pretrain_fx_fc_*` files and no `trained_fx_fc_*`. It is a pretraining-only record, retained
because it completes the measurement of how far the legacy protocol landed from SCAN on the two
most heavily stacked forms: from `pretrain_fx_fc_curves.csv`, $\max|\Delta F_c|$ is
$5.2084\times 10^{-1}$ (deep_cusp_mgga_3x16) and $4.8408\times 10^{-1}$
(deep_rung35ms_mgga_3x16), the worst values recorded anywhere in the campaign, with the exchange
channel at $9.6451\times 10^{-2}$ and $7.5367\times 10^{-2}$. **Conclusion:** the legacy
pretraining error grew with descriptor stacking in the correlation channel specifically, and the
worst case is a network that reproduces barely half of its parent's correlation enhancement
before a single fine-tuning step is taken.

---

## 6. v6: the parent-anchored pretraining protocol

v6 replaces "pretrain toward the parent and hope" with "construct the parent, then certify".
Four coupled changes: the anchored network class, the exact open-shell footing with a smoothed
indicator, a pretraining protocol with parent energies and validation, and a per-architecture
fidelity certificate that gates every campaign stage.

### 6.1 The anchored construction

With `parent_anchor` set (every v6 configuration; `hpcjobs/configs/dfs_step7.dfs6311_grid3_v6*.yaml`,
`model.parent_anchor: true`), the gated network term enters in the *pre-image* of the bounded
map at the parent's value (`networks.AlecGGA_XNet._core`, `AlecGGA_CNet._core`):

$$F(g) = 1 + L\!\left(z_{\mathrm{parent}} + T(g)\right), \qquad
z_{\mathrm{parent}} = \ln\!\left[\frac{(\Lambda - 1)\,F_{\mathrm{parent}}}{\Lambda - F_{\mathrm{parent}}}\right]
\;\text{clamped to } [-40, 40],$$

with $F_{\mathrm{parent}}$ evaluated on the row's own physical inputs by the JAX parent
implementations (`parents.pbe_fx` / `pbe_fc` / `scan_fx` / `scan_fc`, every constant the value
libxc 7.0.0 carries) and the pre-image `parents.lob_preimage` (`parents.py:633`). The clamp at
$z_{\max} = 40$
binds only where the parent approaches a bound of $(0, \Lambda)$: within
$\Lambda(\Lambda - 1)e^{-40}$ of the ceiling ($8.5\times 10^{-18}$ at $\Lambda = 2$,
$6.2\times 10^{-18}$ at 1.804, $8.7\times 10^{-19}$ at 1.174) and within
$\Lambda e^{-40}/(\Lambda - 1)$ of the floor
($8.5\times 10^{-18}$ at $\Lambda = 2$, $9.53\times 10^{-18}$ at 1.804,
$2.87\times 10^{-17}$ at 1.174); both logarithm arguments
are floored at the smallest normal float so a parent at or past a bound by round-off clamps
instead of returning NaN (`parents.lob_preimage` docstring, as corrected). The construction guarantees
$F \in (0, \Lambda)$ for every $T(g)$ and returns $F_{\mathrm{parent}}$ exactly at
$T(g) = 0$.

That identity is exact to one unit in the last place, and the qualification matters where the
ceiling is 1.174. Measured through the committed classes: $F(0) = 1$ bitwise at
$\Lambda = 1.804$ and $\Lambda = 2.0$, and $F(0) = 0.9999999999999999 = 1 - \varepsilon/2$ at
$\Lambda = 1.174$, where $\varepsilon$ is the double-precision epsilon -- a single rounding in
$\exp(\ln(\Lambda - 1))$. The full pre-image round trip $F \to z \to F$, taken through
`_AlecLOB` in the production form $F = 1 + L(z)$ over
$F \in [10^{-3}, \Lambda - 10^{-3}]$, closes to a few units in the last place, and the
measured maximum is a sampling statement: a uniform $2\times10^{6}$-point grid reads
$4.44\times 10^{-16}$ at $\Lambda = 1.804$ (worst near $F = 1.32$),
$2.78\times 10^{-16}$ at 1.174 and $2.22\times 10^{-16}$ at 2.0, while dense sampling
near the worst region at 1.174 reaches $3.33\times 10^{-16}$ -- per-point round-off
extremes grow with sampling density, so these are observed lower bounds on the true
worst, all of order one to two ulp of $F$. Part of the residual is the $1 + L$
addition itself: comparing $L(z)$ against $F - 1$ directly, without re-adding the
one, tightens 1.174 to $2.22\times 10^{-16}$ on the same grid. These are the floors
every "reproduces its parent to $2.2\times 10^{-16}$" statement below rests on, not
fitted agreements.

`create_network_pair` forces `zero_init_final_layer` for an anchored pair, so both networks
equal the parent at initialization by construction. Measured: the anchored meta-GGA
architectures return $F_x$ = `scan_fx` within $2.8\times 10^{-16}$ and $F_c$ = `scan_fc`
within $2.2\times 10^{-16}$ on 31,550 exchange and 15,790 correlation rows of OH and H2O
(HISTORY 2026-08-25); a freshly built anchored pair reproduces the parent curves under
$10^{-10}$ ($F_x$) and $10^{-8}$ ($F_c$) on the figure grid where an unanchored build differs
by more than $10^{-2}$ (HISTORY 2026-08-30, the pretrained-figure identity pins). An anchored
correlation network must be polarization-aware ($\zeta$-blind construction refused by name,
`networks.py:462`), because the parent correlation is divided by the model's $\zeta$-dependent
PW92 baseline whose curvature pole is the subject of Section 2's third figure.

The v6 runs use the DFS descriptor coordinates (`model.descriptor_coordinates: dfs`; the
trained-model class records state it, e.g.
`dfs_step7/dfs6311_grid3_v6g1_size/runs/run_20260827T163330Z/checkpoints/spec_0009/model_val_best.eqx.class.json`):
the exchange MLP reads $x_s = (1 - e^{-s^2})\ln(s + 1)$ of the doubled channel and, on the
meta-GGA rung, $x_\alpha = \ln((\alpha + 1)/2)$ of the raw indicator -- the network inputs of
DFS Eqs. 9-10, defined in Section 3.7.

### 6.2 Exact open-shell footing and the smoothed indicator

Every density-matrix-derived descriptor is now evaluated on the symmetric doubled density of
its own spin channel, $\mathrm{diag}(P_\sigma, P_\sigma)$ -- in the precompute, the SCF loop of
both solver backends, the energy, the potential, the losses and the pretraining rows -- so the
open-shell exchange is the same functional the closed-shell path defines. With libxc SCAN in
place of the network, the three-block energy reproduces PySCF's spin-polarized SCAN exchange
to $1.8\times 10^{-15}$ Ha on O and OH, and the assembled potential is the finite-difference
derivative of the energy to $1.0\times 10^{-10}$ Ha worst case; closed-shell paths are bitwise
unchanged (HISTORY Phase 43, 2026-08-20 to 2026-08-24).

The meta-GGA indicator (`metagga.compute_alpha`, `metagga.py:163`) is
$\alpha = (\tau - \tau_W)/\tau_{\mathrm{unif}}$, stored as
$\min(p(\alpha_{\mathrm{raw}}), 100)$ with $p$ the smooth positive part

$$p(x) = \tfrac{1}{2}\left(x + \sqrt{x^2 + w^2}\right), \qquad w = 10^{-5},$$

which replaced a hard lower clip whose derivative was discontinuous exactly on the manifold
every one-electron spin channel occupies (`metagga.smooth_positive_part`, `:142`;
`_ALPHA_SMOOTHING_WIDTH = 1e-5` at `:104`, `_ALPHA_MAX = 100` at `:61`). The width is anchored
to measurement, not chosen: it must dominate the floating-point residue of $\tau - \tau_W$ on
one-orbital channels (worst on-domain residue $1.3$--$3.7\times 10^{-6}$, margin 2.7--7.7x), and
its SCAN energy cost is $1.17\times 10^{-7}$ Ha on the H atom, linear in the width,
$8.5\times 10^{3}$ below the certificate's free-atom tolerance (`metagga.py` width commentary;
HISTORY 2026-08-24). What the clip cost is on record: with it, the beta-channel
feature-response term of Li's Fock matrix moved by 0.93 Ha under a $10^{-14}$ relative change of
the density matrix, and H's by $4.2\times 10^{-3}$ Ha; with the smooth positive part the same
probe moves H's by $3.6\times 10^{-12}$ Ha (`metagga.py:181-199`). The definition string
`metagga.ALPHA_DEFINITION` (`:112`) is part of the pretraining-data manifest identity, so a file
computed under another definition is stale by construction.

### 6.3 The pretraining set and objective

* **Systems.** The DFS pretraining set (eight free atoms and 22 G2/97 molecules, committed as
  package data at `xcquinox/alec/data/dfs_pretrain_set.json`, its bytes pinned by the
  SHA-256 in `xcquinox/alec/tests/test_dfs_pretrain_set.py`, line 189) at the rung's
  level -- the meta-GGA variant of the DFS protocol drops H2 and N2
  (`cluster/fidelity.dfs_level_for_parent`) -- merged with every free atom of the BH76 /
  W4-11 pools at the pools' own charges and spins, de-duplicated by geometry
  (`pretrain.resolve_pretrain_systems`; the G1 run records 38 systems and 1.39M exchange /
  1.21M correlation rows, `run_20260827T163330Z/pretrain/medium/pretrain_metadata.json`).
  This set is distinct from the 26-point *training* pool of Section 1.2 and from the DFS
  Letter's own 21-molecule training set; conflating them is easy and the three are separate
  objects (`xcquinox/alec/dfs_pretrain_set.py:1` against `dfs_pool.py:1-12`).
  Rows are built on the *parent's* reference density -- PBE or SCAN by rung
  (`inputs.parent_density: auto`) -- under the training orientation lock ($3\times 10^{-5}$),
  with reference SCFs converged through a two-stage DIIS-then-second-order ladder that
  refuses to write an unconverged record (HISTORY Phase 40; the second stage starts from the
  best point of the DIIS trajectory since 2026-08-30, which is what let the SCAN reference of
  the Li atom at 6-311++G(3df,2pd) converge and the meta-GGA datagen complete, HISTORY
  2026-08-31).
* **Exchange footing.** `exchange_footing: spin_channel`: each open-shell exchange row is
  posed per spin channel at $(2\rho_\sigma, 4\sigma_{\sigma\sigma})$ with descriptor blocks
  of the doubled density, reproducing libxc spin-polarized PBE exchange to
  $3.6\times 10^{-12}$ Ha on O (HISTORY Phase 43). The historical `total` footing is what
  every run through v5 used.
* **Objective.** The point-wise integration-weighted residual, plus an optional per-system
  energy term
  $w_E\,\frac{1}{N_{\mathrm{sys}}}\sum_s\big(\sum_{i\in s} w_i\,\epsilon^{\mathrm{LDA}}_i F^{\mathrm{NN}}_i - E_s\big)^2$
  holding the network to the parent's own energies on the same quadrature
  (`pretrain._PretrainLoss`; targets close on PySCF to $4.9\times 10^{-13}$--$4.5\times
  10^{-11}$ Ha, HISTORY Phase 42). Under the anchor the shipped campaigns run
  `energy_term_weight: 0.0` as the exact statement of the objective -- both terms are zero to
  round-off at initialization -- while the term remains *measured* on the saved network
  (recorded `energy_term_max_abs_dE_mHa` 2.6e-3 for G1 medium, 3.5e-3 for deep_mgga). For
  an *unanchored* run the same configuration is refused, because the point-wise objective was
  measured unable to deliver the parent (2.3--56.1 kcal/mol of atomization offset,
  `SPEC_pretrain_fidelity_program.md` Section 2) and an energy-weight sweep then measured
  that no weight closes the gap either (`SPEC_parent_anchor.md` Section 2; both cited in
  the G1 YAML's objective block).
* **Validation.** A seeded 20% of the multi-nucleus systems is withheld (never an atom, since
  every pool atom anchors an atomization energy), scored every 50 steps with patience 10, and
  the best-validation network is the one written; a run whose validation values are all
  non-finite raises instead of returning the untrained network (HISTORY Phase 42). The G1
  `medium` record: held-out systems CO2, HCl, C4H6, C3H8; 2500 steps run; best at step 2500
  (`pretrain_metadata.json`, `validation` block).
* **Mesh.** The synthetic $(r_s, s, \alpha)$ mesh is kept, as a regularizer at the 0.3 share,
  for meta-GGA architectures whose descriptors a mesh node can define: in the v6 meta-GGA
  group, `deep_mgga_3x16` and `deep_mgga_attn_3x16` carry it (`pretrain_mesh: true`,
  `mesh_loss_share_x` 0.3) while the geometry-bearing stacks do not; the GGA groups run
  without it (`pretrain_metadata.json` of the respective runs).

### 6.4 The per-architecture fidelity certificate

`xcquinox/alec/cluster/fidelity.py` accepts a pretrained pair only when it reproduces its
parent in energy units, on the parent's own self-consistent density, through the production
energy path, at the run's SCF identity:

$$\Delta E_{xc} = E_{xc}^{\mathrm{NN}}[\rho_{\mathrm{parent}}] - E_{xc}^{\mathrm{parent}}[\rho_{\mathrm{parent}}],
\qquad
\Delta AE(\mathrm{mol}) = \Delta E_{xc}(\mathrm{mol}) - \sum_{\mathrm{atoms}} n_{\mathrm{atom}}\,\Delta E_{xc}(\mathrm{atom}),$$

with PASS requiring $\max_{\mathrm{atoms}} |\Delta E_{xc}| \leq$ `tol_atom` $= 1.0$ mHa AND
$\max |\Delta AE| \leq$ `tol_AE` $= 1.0$ kcal/mol (the program's binding decision; a tolerance
above 2.0 requires a written override reason). The oracle set is every free atom of the BH76 /
W4-11 pools, the DFS pretraining set at the rung's level, H2O / N2 / CH4 at the *pool*
geometries for every rung (so the three headline offsets are one physical quantity across
architectures), and ground-state Li and Na -- 39 systems for a GGA rung (16 atoms + 23
atomizations), 38 for a meta-GGA one (16 + 22: the meta-GGA DFS protocol drops H2 and N2
while the fixed oracle pool restores N2) (`fidelity.build_oracle_set`,
`dfs_level_for_parent`; the pulled certificates' `summary` blocks). The parent's $E_{xc}$ per
system is computed three independent ways (point-wise libxc on the stored grid, PySCF numint
on a fresh grid, the reference SCF's own accumulated value), with any pairwise disagreement
above $10^{-6}$ Ha a named failure (worst recorded spread at the production identity:
`max_parent_grid_diff_Ha` $3.04\times 10^{-9}$ over the four G1 certificates,
$2.28\times 10^{-9}$ over the five meta-GGA ones); degenerate open-p-shell atoms are measured on orientation-locked references; a
non-finite value is a named failure, never a silent pass. With the parent itself presented
behind the model interface the certificate is an identity to $3.6\times 10^{-15}$ Ha (PBE)
and $2.0\times 10^{-10}$ Ha (SCAN) on the O atom (HISTORY Phase 40).

The verdict, every per-system number, the run identity, the SHA-256 digests of the two network
files and the installed code version are written to
`<run>/pretrain/<arch>/fidelity_certificate.json`. Enforcement has two layers: the on-node
gates (pretrain task exit code, train task, preflight, in-process model builder) honour a
recorded waiver for workflow smoke tests; the record layers (`validate_run`, the cross-arm
merge, the figure suite) require PASS unconditionally, so a waived run can never become a
quantitative result (HISTORY Phase 40).

**The gate has fired on real submissions.** The size ladder was first submitted unanchored:
its certificates read FAIL at 0.79--3.01 mHa / 1.63--11.3 kcal/mol (worst: shallow), against
the 1.0 / 1.0 gate
(`dfs_step7/dfs6311_grid3_v6g1_size/runs/run_20260827T123919Z/pretrain/*/fidelity_certificate.json`,
`enforced: true`, `parent_anchor: false`; first-step pretrain losses 0.008--0.012). The
anchored resubmission passes at $7.2$--$8.5\times 10^{-4}$ mHa / $1.9$--$3.9\times 10^{-3}$
kcal/mol (`run_20260827T163330Z`, same files). The DM-descriptor group shows the same pair for two of its
three architectures: unanchored FAIL at 5.76 mHa / 11.7 kcal/mol (deep_combined) and 5.87 /
11.4 (deep_dm), anchored PASS at $1.2$--$5.1\times 10^{-3}$ mHa for all three
(`dfs_step7/dfs6311_grid3_v6g3_dm/runs/run_20260827T124112Z` and `...T163335Z`;
deep_combined_attn has no unanchored control -- that run wrote no certificate for it). The five
meta-GGA family architectures pretrained to PASS certificates at production identity
(6-311++G(3df,2pd) / grid 3, lock $3\times 10^{-5}$), worst `max_atom` $5.15\times 10^{-3}$
mHa (deep_mgga_attn) and worst `max_dAE` $2.5\times 10^{-3}$ kcal/mol (deep_mgga), over 38
systems each
(`dfs_step7/dfs6311_grid3_v6g2_families_mgga/runs/run_20260831T011905Z/pretrain/*/fidelity_certificate.json`;
HISTORY 2026-08-31). Twelve anchored PASS certificates exist across the pulled v6 groups (4
G1 + 5 meta-GGA + 3 DM).

### 6.5 The measured pretraining floors, and the SCAN-parent floor derivation

Because the anchored start *is* the parent, the first-step point-wise pretraining loss is a
direct measurement of how exactly the stored targets and the anchored parent agree:

| group (parent) | architectures | first-step exchange loss |
|---|---|---|
| G1 size ladder (PBE) | shallow, shallow_attn, medium, medium_attn | $2.72\times 10^{-32}$ (all four) |
| G2 meta-GGA (SCAN), mesh-carrying | deep_mgga, deep_mgga_attn | $3.02\times 10^{-14}$ |
| G2 meta-GGA (SCAN), mesh-free | deep_cusp_mgga, deep_rung35_mgga, deep_rung35ms_mgga | $4.31\times 10^{-14}$ |

(`.../pretrain/<arch>/losses_x.npy`, step 1, of `run_20260827T163330Z` and
`run_20260831T011905Z`.) The PBE-parent floor is the double-precision identity: residuals
of order $10^{-16}$ in $F$, squared. The 18-orders-of-magnitude gap to the SCAN-parent
floor is carried entirely by the indicator ceiling, `metagga._ALPHA_MAX` $= 100$ -- not by
the smoothing (HISTORY 2026-08-31 (erratum), which supersedes the first recorded
derivation). The stored column is $\min(p(\alpha_{\mathrm{raw}}), 100)$, and the anchored
network recovers the raw indicator from it exactly below the ceiling
(`networks._raw_indicator` / `metagga.invert_smooth_positive_part`; Section 3.5's right-hand
panel is that inversion measured): a stored
$p(0) = w/2 = 5\times 10^{-6}$ inverts to $\alpha = 0.0$ to round-off, the end-to-end
anchored exchange MSE on the committed mesh block is $7.62\times 10^{-32}$, and the H atom
-- one-orbital on every row, the column at its smoothed floor everywhere -- prices at
$2.85\times 10^{-32}$ (HISTORY 2026-08-31 (erratum)). At the ceiling the inversion cannot
act: the column reads 100 while the exact indicator on the capped rows is unrecoverable.
The capped population is the low-density tail: on the O atom 2492 rows are capped, with
$\alpha_{\mathrm{exact}}$ spanning $\sim 1.0\times 10^{2}$ to $\sim 7.0\times 10^{6}$
(median 555) at $\rho$ from $10^{-10}$ to $8.6\times 10^{-5}$ (median
$6.1\times 10^{-8}$), 90% of the capped weight sitting at $\alpha = 108$--$2475$.

![Indicator ceiling: the SCAN exchange residual on a capped row. Each curve is
$|F_x^{\mathrm{SCAN}}(s,\alpha) - F_x^{\mathrm{SCAN}}(s,100)|$ against the exact indicator, at
three reduced gradients.](figures_report_pretraining/alpha_ceiling.png)

This is the price of the ceiling in the quantity the pretraining loss measures. Read it as
follows: a grid row whose exact indicator exceeds 100 is evaluated at 100, and the vertical axis
is the resulting error in the parent's own exchange enhancement factor. The residual is zero by
construction at the cap and rises to a saturation value as $\alpha \to \infty$;
`alpha_ceiling.csv` gives that saturation as $1.7365\times 10^{-3}$ at $s = 0$,
$1.2421\times 10^{-3}$ at $s = 1$ and $1.0008\times 10^{-3}$ at $s = 4$, against
$F_x(s=0, \alpha=100) = 0.78598$ -- so at most about 0.2 percent relative. SCAN's switching
function has nearly but not exactly saturated at the cap: per capped row of the O atom the
median $|\Delta F|$ against the exact-$\tau$ libxc target is $2.55\times 10^{-4}$ and the worst
is $5.70\times 10^{-4}$. **Conclusion:** the ceiling is not a bug and not a precision
regression; it is a stated, bounded truncation whose $F$-space cost is at most about 0.2 percent
relative ($1.74\times 10^{-3}$ absolute in $F_x$, at $s = 0$) and whose energy cost the
certificate gates at 194 times inside the atomic tolerance.

The capped rows carry 100.0% of the weighted exchange MSE on the O
atom ($1.2593\times 10^{-13}$ on one converged reference path, $1.2190\times 10^{-13}$ on a
second, the capped share 1.000000 on both) and on H2O ($3.07\times 10^{-14}$), the uncapped
remainder pricing at $2.7\times 10^{-29}$ and 0.0 (HISTORY 2026-08-31 (erratum, as
amended)).
The synthetic mesh contributes nothing ($\leq 6\times 10^{-29}$; its $\alpha$ nodes stop at
5, below the ceiling -- `pretrain_data_gen.py` line 1070, `MESH_ALPHA`, and the right-hand panel
of the mesh figure in Section 4.1), so the
mesh-carrying floor is the mesh-free floor scaled by exactly the atomic share of the loss
weight: $3.0167\times 10^{-14} / 4.3096\times 10^{-14} = 0.7000000000000004$ (the ratio
from `losses_x.npy` step 1 of the two file variants). A hypothetical in which the parent reads the smoothed column
directly, the inversion removed, prices the mesh block at $1.90\times 10^{-14}$, worst
$|\Delta F| = 5.6\times 10^{-7}$ at $(r_s = 0.1, s = 0, \alpha = 0)$ (the coordinates as
the superseded derivation recorded them) -- numerically adjacent to the measured floors,
which is what that derivation mistook for the mechanism -- but it was never the run's code
path (HISTORY 2026-08-31 (erratum)).
`parents.scan_fx` itself agrees with libxc SCAN to $4.9\times 10^{-15}$ max over
$\rho \in [10^{-6}, 10^{2}]$, $s \leq 8$, $\alpha \leq 10$ (HISTORY 2026-08-31), so the
floor is not an implementation error of the parent. Nothing is repaired because nothing is
broken: the ceiling is the recorded energy-faithfulness bound on an indicator whose raw
value grows without bound on the low-density tail (`metagga.py`, HISTORY Phase 17) -- it
zeroes the indicator's derivative only at capped points, and the separate tail-gradient
freeze that once accompanied it was removed on 2026-08-06 for misreporting the derivative
of an energy ingredient (`metagga.py` lines 55-60; HISTORY 2026-08-06). Its energy
consequences -- $8.8\times 10^{-8}$ Ha on the N atom's exchange for the ceiling and
$1.2\times 10^{-7}$ Ha on the H atom for the smoothing -- are stated floors of the SCAN
oracle four orders under the 1.0 mHa free-atom gate ($1.14\times 10^{4}$x and
$8.3\times 10^{3}$x; `SPEC_parent_anchor.md` Section 3.1), with the ceiling's $F_x$ effect
$1.8\times 10^{-3}$ absolute in $F_x$ on the capped rows (about 0.2 percent relative, as
priced above); and the certificates gate the whole
start's consequence at 194x inside the atomic tolerance (worst meta-GGA `max_atom`
$5.15\times 10^{-3}$ mHa, deep_mgga_attn, against 1.0 mHa). The corrected derivation is recorded
so the $10^{-14}$ start reads as the priced consequence of a documented bound rather than
as a precision regression.

### 6.6 Pretrained enhancement factors

The pretrained curves are now a published, per-generation artifact
(`notebooks/analysis/pretrain_fx_fc.py`, loading through the production certified-model
builder; baselines are `parents.pbe_fx` / `pbe_fc` -- the anchor's own parent at libxc
constants, because a rounded-constant analytic helper differs by up to $4.553\times 10^{-6}$
and would read as a spurious learned correction under the anchor; HISTORY 2026-08-30):

* v6 G1 (anchored, PBE parent), after the 2500-step pretrain: $\max|\Delta F_x| =
  8.7\times 10^{-7}$ (shallow), $4.0\times 10^{-6}$ (shallow_attn), $4.2\times 10^{-6}$
  (medium_attn), $9.2\times 10^{-6}$ (medium); the correlation channel spans
  $1.1\times 10^{-6}$ (medium) to $2.2\times 10^{-5}$ (shallow_attn)
  (`figures_dfs_step7_dfs6311_grid3_v6g1_size_val_best/pretrain_fx_fc_curves.csv`).
* v6 G2 meta-GGA (anchored, SCAN parent): all five architectures within
  $8.1\times 10^{-7}$ to $1.3\times 10^{-5}$ of SCAN over both channels
  (`figures_dfs_step7_dfs6311_grid3_v6g2_families_mgga/pretrain_fx_fc_curves.csv`; per-channel
  worst values from the same file: $F_x$ $8.1\times 10^{-7}$ (deep_cusp_mgga) to
  $1.3\times 10^{-5}$ (deep_mgga), $F_c$ $5.4\times 10^{-6}$ to $1.1\times 10^{-5}$).
* The legacy (unanchored) pretrains, same figure pipeline: $\max|\Delta F_x|$ 0.039--0.090
  (v4gga GGA arms), meta-GGA $F_c$ up to 0.49--0.52
  (`figures_dfs_step7_dfs6311_grid3_{v4gga,v4,v5}_val_best/pretrain_fx_fc_curves.csv`,
  `.../v5mgga2/...`).

![Anchored (v6 G1) pretrained enhancement factors against PBE, all four size-ladder
architectures. Note the vertical
scale.](figures_dfs_step7_dfs6311_grid3_v6g1_size_val_best/pretrain_fx_fc_delta_all.png)

This is the same figure, from the same module, as the v4gga panel of Section 4.2 -- and the only
thing that has changed is the axis. The worst deviation anywhere on this figure is
$2.1957\times 10^{-5}$ (`shallow_attn`, correlation channel, from
`pretrain_fx_fc_curves.csv`), against $1.517\times 10^{-1}$ on the v4gga panel: a factor of
about 7000. **Conclusion:** whatever the anchored networks later learn, they do not begin by
having to relearn their parent, and the residual structure visible here is the 2500-step
pretrain's own drift away from an exact start rather than an approach toward one.

![Anchored (v6 G2) meta-GGA pretrained factors against SCAN, all five family architectures, at
the indicator slices $\alpha = 0$ and
$\alpha = 1$.](figures_dfs_step7_dfs6311_grid3_v6g2_families_mgga/pretrain_fx_fc_delta_all.png)

The meta-GGA counterpart, against the harder parent. From `pretrain_fx_fc_curves.csv` the worst
exchange deviations are $8.094\times 10^{-7}$ (deep_cusp_mgga_3x16) to
$1.2901\times 10^{-5}$ (deep_mgga_3x16) and the worst correlation deviations
$5.385\times 10^{-6}$ to $1.1197\times 10^{-5}$. Compare the v5mgga2 panel of Section 5, whose
correlation channel reaches $5.2\times 10^{-1}$ on the same two stacked architectures.
**Conclusion:** the anchor removes the descriptor-stacking dependence of the pretraining error
entirely -- the five architectures now agree with SCAN to within a factor of 16 of each other,
where the legacy protocol spread them over two orders of magnitude and made the most
information-rich architectures the least faithful.

The anchor bought pretraining fidelity of four orders of magnitude in the curve metric
(HISTORY 2026-08-31).

---

## 7. The comparison figure sets

Per-generation enhancement-factor sets, all rendered by the same two modules
(`notebooks/analysis/pretrain_fx_fc.py`, `trained_fx_fc.py`) against the same parent
baselines:

| set | architectures | contents |
|---|---|---|
| `figures_dfs_step7_dfs6311_grid3_v3_val_best` | deep, deep_attn, deep_cusp (3x16) | pretrain + trained |
| `figures_dfs_step7_dfs6311_grid3_v4gga_val_best` | six GGA forms | pretrain + trained |
| `figures_dfs_step7_dfs6311_grid3_v4_val_best` | deep_mgga, deep_mgga_attn (+ rung35_mgga pretrain) | pretrain + trained (retired record) |
| `figures_dfs_step7_dfs6311_grid3_v5_val_best` | deep_mgga (+ trio pretrain) | pretrain + trained (retired record) |
| `figures_dfs_step7_dfs6311_grid3_v5mgga2` | deep_cusp_mgga, deep_rung35ms_mgga | pretrain only |
| `figures_dfs_step7_dfs6311_grid3_v6g1_size` | shallow(_attn), medium(_attn) | pretrain + `anchored_vs_unanchored_fx_fc.png` |
| `figures_dfs_step7_dfs6311_grid3_v6g1_size_val_best` | shallow(_attn), medium(_attn) | pretrain + trained + `anchored_vs_unanchored_fx_fc.png` |
| `figures_dfs_step7_dfs6311_grid3_v6g2_families_mgga` | five meta-GGA family forms | pretrain only (training pending) |

Two of these are omitted from the embedded figures above on measured grounds. The v3 pretrained
deltas are the v4gga ones for the three shared architectures (deep_3x16 $F_x$
$3.8989\times 10^{-2}$ against $3.8990\times 10^{-2}$; deep_cusp_3x16 $6.2870\times 10^{-2}$ in
both files), and the v5 pretrained deltas are bit-identical to v4's, as Section 5 shows. Neither
adds a curve the embedded panels do not already carry.

`anchored_vs_unanchored_fx_fc.png` (in both v6g1 directories) overlays the anchored G1
corrections on the unanchored v3 and v4gga records. Its provenance chain to the published v4
merged sets
(`figures_dfs6311_v4_merged_val_best{,_gga}`) was verified by execution: symlink resolution
into `run_20260810T202813Z`, byte-equal evaluation files, weights older than their evaluations
in all 54 specs, single-generation job records (HISTORY 2026-08-31). The long-form data behind
every curve statement below is `trained_fx_fc_curves.csv` / `pretrain_fx_fc_curves.csv` in the
respective directories, with the evaluation channel recorded per row (`eval_channel`,
validation-best with a labelled final-checkpoint fallback).

![Optimized (validation-best) enhancement-factor differences from PBE, unanchored v4gga, one
cell per architecture at its largest completed subset
size.](figures_dfs_step7_dfs6311_grid3_v4gga_val_best/trained_fx_fc_delta_best.png)

![Optimized (validation-best) enhancement-factor differences from PBE, anchored v6 G1, one cell
per architecture at its largest completed subset
size.](figures_dfs_step7_dfs6311_grid3_v6g1_size_val_best/trained_fx_fc_delta_best.png)

These two are the trained counterparts of the pretrained panels of Sections 4.2 and 6.6, and are
read as a pair. Both draw the *end* of training rather than the start, each architecture at the
largest cell it has completed, so the vertical scales are comparable (order $10^{-1}$) even
though the starting scales differed by four orders of magnitude. From
`trained_fx_fc_curves.csv`, the exchange corrections at subset size 18 span
$[-0.0441, +0.0908]$ for unanchored deep_3x16 and $[-0.0226, +0.1616]$ for anchored medium; the
correlation corrections at $r_s = 2$ span $[0, +0.916]$ and $[0, +0.431]$ respectively.
**Conclusion:** the anchored networks do move, and by an amount comparable to the unanchored
ones in exchange; the divergence is in correlation, where the anchored maximum is less than half
the unanchored one, and the next figure isolates why.

![Anchored against unanchored corrections, pretrained start and optimized end. Rows: pretrained
(top) and optimized at subset size 18 (bottom). Columns: $F_x$ against $s$ (left) and $F_c$ at
$r_s = 2$ (right). Four series: v3 and v4gga `deep_3x16` unanchored (dashed, different dash
periods), v6 `medium` and `medium_attn` anchored
(solid).](figures_dfs_step7_dfs6311_grid3_v6g1_size_val_best/anchored_vs_unanchored_fx_fc.png)

This figure is the document's central comparison, and it is rebuilt by a committed module
(`anchored_vs_unanchored_fx_fc.py`) from the committed long-form curve CSVs rather than by an ad
hoc script, with a footer that states its own coverage from the rows it read. The subset size 18
representative is the largest cell every drawn generation has reached, enforced by an internal
consistency test; the footer currently reports v6 trained coverage of 27 cells (medium 10,
medium_attn 10, shallow 7), all on the validation-best channel, which matches the tally of
Section 9.2 exactly.

Read the top row first: the two unanchored pretrained curves lie on one another at
$3.9\times 10^{-2}$ in $F_x$ and $1.4\times 10^{-2}$ in $F_c$ -- the same protocol and seed --
while the two anchored ones are invisible at this scale, at $9.2\times 10^{-6}$ and
$6.0\times 10^{-7}$ (all from `anchored_vs_unanchored_fx_fc.csv`). Then read the bottom row.
The **exchange** panel shows one family: every optimized curve has a bond-region dip and a
positive bump peaking between $s = 2.35$ and $s = 2.7$, of height $+0.0908$ (unanchored
deep_3x16), $+0.1181$ (medium_attn) and $+0.1616$ (medium), with the v3 cell an order of
magnitude smaller at $+0.0026$. The **correlation** panel shows two families: the unanchored
correction grows monotonically into large $s$, reaching $+0.916$ at $s = 6$, while both anchored
corrections peak near $s = 1.2$ ($+0.4314$ and $+0.3545$) and then collapse, to $+0.0099$ and
$+0.0048$ at $s = 6$. **Conclusion:** training discovers the same exchange physics from either
start, and does not discover the same correlation physics. Section 8.2 identifies the mechanism
as a property of the anchored parameterization rather than of the optimizer.

---

## 8. Measured pros and cons of the anchor

### 8.1 What the anchor buys

1. **Exact parent start.** First-step losses at the identity floor (Section 6.5); pretrained
   curves $10^{-7}$--$10^{-5}$ from the parent against 0.04--0.5 for the legacy protocol
   (Section 6.6); the certificate passes at initialization by construction, and a 2500-step
   pretrain cannot lose it (all twelve pulled anchored certificates PASS, Section 6.4).
2. **Certificate-gated fidelity.** The silent handoff class of Section 4.2 -- worst-system
   offsets of 25.7 to 56.1 kcal/mol per descriptor-carrying architecture, against 4.1--4.2 for
   the descriptor-free controls -- is
   structurally excluded: an architecture that does not reproduce its parent cannot reach the
   train stage (measured FAIL-then-PASS pairs on the same architectures, Section 6.4).
3. **A convergent exchange family.** Where anchored and unanchored campaigns can be compared
   at the same capacity, the *trained* exchange corrections agree in form: a bond-region dip
   near $s = 0.7$--$1.1$ ($-0.016$ to $-0.044$ over the four cells) and a positive bump
   peaking near $s = 2.4$--$2.7$,
   of height $+0.087$ (shallow, ss=7, its largest completed cell), $+0.091$ (unanchored
   deep_3x16, ss=18), $+0.118$ (medium_attn, ss=18), $+0.162$ (medium, ss=18); at $s = 3$ the
   band is $+0.083$ to $+0.156$ (recomputed from
   `trained_fx_fc_curves.csv` of `v4gga_val_best` and `v6g1_size_val_best`; HISTORY 2026-08-31
   records the band as +0.07 to +0.16 near $s = 3$ at 18-cell coverage, where `shallow`
   reached only ss=5 and read $+0.079$; the unanchored attention twin peaks higher, $+0.210$).
   Training discovers the same exchange physics from either start -- the optimized networks move
   off the parent by up to 0.16 in $F_x$ (HISTORY 2026-08-31).

### 8.2 The cost: pre-image sensitivity suppression

The anchored correction enters through the bounded map at the parent's pre-image, so its
parameter sensitivity carries the factor

$$L'(z_{\mathrm{parent}}) = F_{\mathrm{parent}}\left(1 - \frac{F_{\mathrm{parent}}}{\Lambda}\right),$$

which vanishes as the parent approaches either bound of $(0, \Lambda)$.

![Pre-image sensitivity of the anchored correction. Left: exchange, $L'(z_{\mathrm{parent}})$
against $s$ for the PBE parent at $\Lambda = 1.804$ and the SCAN parent at $\Lambda = 1.174$ for
both indicator slices, with the SCAN $\alpha = 0$ case on a log inset. Right: the correlation
mirror at $r_s = 2$, $\zeta = 0$, showing $L'$ tracking $F_c$ down to
zero.](figures_report_pretraining/preimage_sensitivity.png)

Every value on this figure is obtained by differentiating the committed `_AlecLOB` class with
`jax.grad` at the pre-image `parents.lob_preimage` returns, so it is the gradient prefactor the
optimizer actually sees. Read the left panel as the exchange story:
`preimage_sensitivity.csv` gives $L' = 0.4456762749$ at $s = 0$ for the PBE parent, falling to
$0.0072655692$ by $s = 20$ -- a factor of 61 across the plotted range, but never zero. The SCAN
$\alpha = 0$ curve is the degenerate case: its parent *sits on* the ceiling at $s = 0$, so the
pre-image clamps at $+40$ and $L'$ is **exactly 0.0** there, rising to $0.2599$ by $s = 20$;
the anchored meta-GGA exchange network is untrainable at the single-orbital small-gradient
point, by construction. Read the right panel as the correlation mirror: as $F_c \to 0$ the
factor $L' \to F_c$, and the two run together down the plot -- $0.5000000000$ at $s = 0$,
$0.0764106655$ at $s = 2$, $0.0015311495$ at $s = 6$, against $F_c$ values of
$0.9999979292$, $0.0795769076$ and $0.0015323235$. **Conclusion:** the anchored
parameterization suppresses trainability exactly where the parent vanishes, and for PBE
correlation that region is the large-gradient tail -- a factor of 327 less gradient at $s = 6$
than at $s = 0$. HISTORY 2026-08-31 records the exchange endpoints rounded, as 0.45 and 0.007.

The measured consequence, on ss=18 validation-best cells at $r_s = 2$
(`trained_fx_fc_curves.csv` of `v4gga_val_best` and `v6g1_size_val_best`; HISTORY
2026-08-31): the *unanchored* v4gga deep_3x16 correlation correction grows into large $s$ --
$+0.79$ at $s = 2$ to $+0.92$ at $s = 6$ -- built entirely by training from a pretrained
start that sits within about $\pm 0.014$ of the parent at $r_s = 2$
(`v4gga_val_best/pretrain_fx_fc_curves.csv`); the *anchored* medium correction collapses past
$s = 2$: $+0.29$ at $s = 2$ to $+0.010$ at $s = 6$ (medium_attn: $+0.21 \to +0.005$). Both
anchored cells still build sizable corrections where the pre-image leaves trainability
(peaks $+0.43$ and $+0.35$ near $s = 1.2$).

### 8.3 The BH76 barrier bias

The unanchored campaigns used precisely that large-$s$ correlation freedom to remove the
parent's systematic barrier bias. PBE's BH76 error is almost pure bias ($-6.6$ to $-7.5$
kcal/mol mean signed error on the two slices below; $|\mathrm{bias}|/\mathrm{MAE} = 0.97$ for
PBE and 0.93 for SCAN on the full v5 held-out pool,
`NOTES_v5_mgga_vs_scan.md:79-89`). Measured mean signed BH76 errors on the
strict held-out slice, validation-best channel (recomputed from `per_reaction.json` of the
respective specs):

| cell | mean signed BH76 error (NN) | PBE same slice | slice |
|---|---|---|---|
| unanchored deep_3x16, ss=12 (`merged_v4_arms/checkpoints/spec_0007`) | $-0.20$ | $-6.62$ | 50 reactions |
| anchored medium, ss=12 (`v6g1_size .../spec_0007`) | $-7.75$ | $-7.47$ | 61 reactions |
| anchored medium, ss=18 | $-4.41$ | $-7.47$ | 61 reactions |
| anchored medium_attn, ss=12 | $-0.81$ | $-7.47$ | 61 reactions |

The recorded comparison pair is the first two rows (HISTORY 2026-08-31): the unanchored
network removed the bias, the anchored medium cell kept it essentially at the parent's value.
The last two rows, added here from the same files, show the suppression is not absolute --
the anchored attention twin at the same subset size removed most of the bias (through other
channels; its correlation still collapses at large $s$, Section 8.2) while paying for it in
scatter (BH76 MAE 10.47 against PBE's 7.73, Section 9.2). Whether anchored cells can
reproduce the unanchored barrier improvement is the campaign's live question (Section 12).

**Unanchored cons, for the same ledger:** no parent fidelity at handoff (the worst-system
offsets of 25.7 to 56.1 kcal/mol per descriptor-carrying architecture in Section 4.2, against
4.1--4.2 for the controls, and invisible to the point-wise loss); every correction, physical or not,
must be created by training from a flat start, so the pretrained state carries none of the
parent's structure into the fine-tune beyond the $\leq 0.09$ curve proximity; and there is no
gate -- the v4 record carried its offsets into production for the whole life of both running arms.

---

## 9. Current numbers: completed v6 GGA cells against the corrected v4/v5 GGA record

### 9.1 What is being compared

* **v6 G1** (`dfs6311_grid3_v6g1_size`, run `run_20260827T163330Z`): the anchored size
  ladder -- shallow, shallow_attn (2x8) and medium, medium_attn (3x16, the production
  width). Three registry fields separate `medium` from `deep_3x16`
  (`descriptor_log_transform`, `zero_init_final_layer`, `dm_entropy_intensive`), and under
  the v6 model block the two named flags are inert for this descriptor-free pair
  specifically -- `parent_anchor` forces the zero-initialized final layer in
  `networks.create_network_pair`, and `descriptor_coordinates: dfs` takes the branch of
  both `_core` paths that never applies the log transform to the MLP coordinates
  (Section 3.7); the scope matters, because for a cusp-carrying architecture the
  flag stays live under every coordinate set through
  `ArchitectureConfig.materialize_descriptors`, which injects it into the cusp
  descriptor's own logarithm (a measured 0.51 shift of the bounded $c_1$ column;
  Section 3.1) -- while the third field has no functional consumer in the package
  at all and reaches only the spec digest (Section 3.6); the operative cross-campaign difference between the
  anchored medium cells and the unanchored v4 deep_3x16 cells is therefore the anchor plus
  the legacy-to-dfs coordinate change alone (HISTORY 2026-08-31 (erratum), superseding the
  2026-08-30 transform/initialization reading). 44 cells (4 architectures x 11 subset
  sizes), evaluated at the validation-best checkpoint over strict held-out slices.
* **v4gga merged validation-best** (`merged_v4_arms`, evaluations symlinked into
  `run_20260810T202813Z`): the corrected v4 GGA record -- after the 2026-08-13 strict-holdout
  validity pass (the 2026-08-13 entries under the HISTORY Phase 36 header), the 2026-08-18
  full-slice comparator anchors, and the NaN-species backfill (HISTORY Phase 38) -- 54 of
  66 cells with validation-best evaluations (11 each for
  deep_3x16, deep_attn, deep_cusp, deep_rung35; 7 rung35ms; 3 rung35_attn). The v5 arms
  contribute no GGA cells (their GGA rows are the v4 rows by design, Section 5).

The strict slices are cell-matched but not identical across campaigns: each cell is scored
against its own run's held-out slice with the cell's own PBE anchor (over the 54 v4 GGA cells:
BH76 43--50, W4-11 97--113, combined 145--163 reactions; over the 27 v6 cells:
61 / 111--120 / 172--181; per-spec `test_set.csv` `n_reactions` and `per_reaction.json`). Beats verdicts below are therefore within-cell (NN MAE < the same slice's PBE
MAE), never cross-campaign row-for-row.

### 9.2 v6 G1 at current coverage: 27 of 44 cells

From the 27 `eval_holdout_val_best/test_set.csv` files present (medium 10 of 11, medium_attn
10 of 11, shallow 7 of 11, shallow_attn 0; the medium ss=26 cell failed on the open
NaN-gradient defect, HISTORY 2026-08-31), all values kcal/mol, NN / cell-matched PBE:

| arch | ss=1 | 2 | 3 | 4 | 5 | 6 | 7 | 12 | 15 | 18 |
|---|---|---|---|---|---|---|---|---|---|---|
| **W4-11** medium | 6.48/13.57 | 11.38/13.47 | 10.86/13.47 | 11.38/13.55 | 9.19/13.47 | 10.32/13.58 | 8.50/13.48 | 12.25/13.41 | 9.54/13.36 | 7.84/13.08 |
| medium_attn | 5.99/13.57 | 11.03/13.47 | 9.04/13.47 | 11.56/13.55 | 9.54/13.47 | 9.50/13.58 | 7.04/13.48 | 9.13/13.41 | 8.06/13.36 | 6.83/13.08 |
| shallow | 10.48/13.57 | 10.16/13.47 | 8.74/13.47 | 8.97/13.55 | 8.56/13.47 | 8.67/13.58 | 10.41/13.48 | -- | -- | -- |
| **BH76** medium | 8.03/7.73 | 9.71/7.73 | 9.96/7.73 | 10.47/7.73 | 6.51/7.73 | 10.22/7.73 | 6.51/7.73 | 10.24/7.73 | 8.13/7.73 | 6.92/7.73 |
| medium_attn | 9.66/7.73 | 7.45/7.73 | 9.01/7.73 | 11.30/7.73 | 8.72/7.73 | 8.95/7.73 | 7.61/7.73 | 10.47/7.73 | 10.75/7.73 | 10.57/7.73 |
| shallow | 7.34/7.73 | 7.84/7.73 | 7.84/7.73 | 6.46/7.73 | 6.47/7.73 | 7.22/7.73 | 8.67/7.73 | -- | -- | -- |
| **combined** medium | 7.00/11.59 | 10.82/11.53 | 10.56/11.53 | 11.07/11.58 | 8.29/11.53 | 10.29/11.60 | 7.82/11.52 | 11.55/11.43 | 9.05/11.38 | 7.51/11.18 |
| medium_attn | 7.23/11.59 | 9.83/11.53 | 9.03/11.53 | 11.47/11.58 | 9.27/11.53 | 9.32/11.60 | 7.23/11.52 | 9.60/11.43 | 9.00/11.38 | 8.16/11.18 |
| shallow | 9.42/11.59 | 9.38/11.53 | 8.43/11.53 | 8.12/11.58 | 7.86/11.53 | 8.18/11.60 | 9.82/11.52 | -- | -- | -- |

Headline, at 27-cell coverage:

* **W4-11: beaten in 27 of 27 cells** -- NN 5.99--12.25 against cell-matched PBE 13.08--13.58,
  a ratio of 0.441 (medium_attn, ss=1) to 0.914 (medium, ss=12).
* **Combined pool: 26 of 27** -- the one miss is medium ss=12 (11.55 against 11.43, ratio
  1.011).
* **BH76: 9 of 27**, best 6.46 (shallow, ss=4) against 7.73; the other beats are medium
  ss=5/7/18 (6.51, 6.51, 6.92), medium_attn ss=2/7 (7.45, 7.61) and shallow ss=1/5/6 (7.34,
  6.47, 7.22). The worst cell is medium_attn ss=4 at 11.30, a ratio of 1.463.

The 27-cell state supersedes the 25-cell one published on 2026-08-31 in two independent ways.
Two `shallow` cells completed (ss=6 and ss=7), and seven cells were repaired for a reference
defect (Section 9.3) -- `medium_attn` ss=15 and ss=18 and all five then-completed `shallow`
cells -- which moved their W4-11 and combined rows and their PBE anchors, though not their BH76
rows. The verdict counts moved from 25/25, 24/25 and 8/25 to 27/27, 26/27 and 9/27; the one
combined-pool miss is the same cell in both. The first published set, 18 cells, read 18 of 18,
17 of 18 and 5 of 18 (HISTORY 2026-08-31).

![Learning curves: held-out reaction-energy MAE against training subset size, one line per
architecture, with the full-pool PBE reference dashed. Left panel held-out, right panel
in-sample.](figures_dfs_step7_dfs6311_grid3_v6g1_size_val_best/ablation_mae_vs_subset.png)

This is the combined-pool row of the table above, drawn. Read the left panel against the dashed
PBE line: every one of the 27 anchored cells except `medium` at ss=12 sits below it (26 of 27,
derived from the `test_set.csv` files above; combined NN 7.00--11.55 against PBE 11.18--11.60).
Read the *shape* rather than the level, because there is one training run per point and no
replicates (Section 1.1): the curves are not monotone in subset size, the best combined cells
being at the smallest sizes (`medium` ss=1 at 7.00) and at the largest (`medium` ss=18 at 7.51),
with a pronounced worsening through ss=2 to ss=6. Read the right panel as the control: in-sample
error falls with subset size as it must, so the non-monotone held-out curve is a generalization
property and not a fitting failure. **Conclusion:** at this coverage the anchored ladder shows no
capacity effect that the noise of single runs would not explain, and the 2x8 `shallow`
architecture is not distinguishable from the 3x16 pair on the combined pool.

![NN-against-PBE cell grid: the ratio of held-out reaction-energy MAE, architecture by subset
size, diverging about 1.0. Blue is better than PBE; missing cells are
hatched.](figures_dfs_step7_dfs6311_grid3_v6g1_size_val_best/ablation_arch_subset_heatmap_vs_pbe.png)

The same data as a grid, and the most compact statement of coverage. Each cell's PBE mean
absolute error is computed on exactly the reactions its network was scored on, so every ratio is
like-for-like. Over the 27 present cells the combined-pool ratio runs from 0.604 (`medium`,
ss=1) to 1.011 (`medium`, ss=12); the hatched cells are the 17 not yet complete -- all eleven
`shallow_attn`, four `shallow` sizes, and the ss=26 cells of both 3x16 architectures.
**Conclusion:** the grid reads almost uniformly blue, which is the headline result of this
section, and its hatching is the honest statement of what remains: no `shallow_attn` cell exists
at all, so the size ladder's attention comparison is currently one-sided.

![Held-out parity split by reaction class: W4-11 atomization, BH76 barriers, and the combined
total. Top row coloured by architecture at each one's largest cell, with PBE as grey crosses;
bottom row every cell coloured by training subset
size.](figures_dfs_step7_dfs6311_grid3_v6g1_size_val_best/ablation_parity_by_class.png)

Of the eight parity variants the suite renders, this is the one embedded, because the document's
central finding is a *class* asymmetry -- W4-11 beaten everywhere, BH76 rarely -- and this is the
only variant that separates the classes with per-class axis limits. The others are marginals
over the same points: `ablation_parity.png` and `ablation_ae_parity.png` collapse to two panels
(all reactions, and W4-11 alone), `ablation_parity_marginal_2x2.png` splits arch against subset
without splitting class, and `ablation_parity_arch_cols.png`, `_facet_subset`,
`_grid_by_subset` and `_errbars_by_subset` re-facet the identical scatter by one further
variable each. None carries a class distinction this one lacks.

Read the axis ranges first, since they are why the classes must be separated: on `medium` ss=18
the W4-11 references span 39.0 to 1007.9 kcal/mol while the BH76 references span $-12.3$ to
104.8 (from `per_reaction.json` of spec_0009), so on shared axes the barrier column would be a
point. Read each column against its own diagonal. The W4-11 column shows the network tracking
atomization energies over three decades with visible improvement on PBE. The BH76 column shows
the scatter that the MAE table quantifies: on the same cell the network's barrier predictions
run $-24.5$ to 87.8 kcal/mol against references of $-12.3$ to 104.8, i.e. it both over- and
under-shoots. **Conclusion:** the anchored networks have learned atomization energetics and have
not learned barrier heights; the combined column is dominated by W4-11 simply because that pool
contributes about twice as many reactions per slice.

### 9.3 The c2 reference-branch incident and its repair

One species in the pool required a repair between the 25-cell and 27-cell states, and the
episode is worth recording because it is a class of defect that internal consistency cannot
catch.

C2 is a multireference dimer, and its PBE SCF at 6-311++G(3df,2pd) / grid 3 under the
$3\times 10^{-5}$ orientation lock never converges in DIIS -- 100 cycles oscillating between two
SCF solutions with a trajectory spread of $1.204\times 10^{-1}$ Ha. The trajectory-best rescue
introduced for the Li atom then flipped C2's reference from $-75.8167407121$ Ha (internally
stable) to the internally unstable solution $50.10$ kcal/mol above it, converged at
$-75.7368945310$ Ha and stamped into the affected evaluations as $-75.7368945258$ Ha (the two
differ at the eleventh decimal, the stamped value being the one the patch tool reads back as
`from_E_pbe`), because PySCF's
second-order SCF ingests a starting density by aufbau re-occupation of its Fock matrix and C2's
ground solution is **non-aufbau in its own Fock** (an occupied orbital $2.35\times 10^{-4}$ Ha
above a virtual). Any density start near the crossing therefore lands on either branch
draw-dependently; four of ten local draws of the pre-fix code stamped the higher branch, as did
seven pulled evaluations of `run_20260827T163330Z` (HISTORY 2026-08-31). The cross-spec
reference guard, established on a different 24 mHa incident (HISTORY Phase 38), excluded c2 from
the pooled figure baselines while the spread stood.

The mechanism is closed at its source: the DIIS recorder now also keeps the trajectory's
lowest-energy point as an orbital pair, and any converged solution sitting more than
$10^{-4}$ Ha above that minimum is rerun from the exact determinant -- immune to re-occupation
-- with a standing excess refused rather than stamped (`data._converge_reference_scf`). The
tolerance is anchored to four measured same-basin excesses ($-2.97\times 10^{-7}$ Ha for
Li/SCAN, $-4.09\times 10^{-6}$ for C2/PBE, $+8.26\times 10^{-7}$ for the O/PBE rescue and
$+8.38\times 10^{-6}$ for the S/SCAN DIIS endpoint over its own trajectory minimum), sitting
about 12 times above the largest of those and nearly three decades below the
$+7.984\times 10^{-2}$ Ha wrong-branch signal (`data.py:414-429`).

The already-written evaluations were repaired surgically rather than re-run:
`hpcjobs/reeval_c2_patch.py` audits every channel against the two measured branch anchors,
recomputes the C2 reference once through the branch-checked rescue, recomputes the C2-derived
network quantities per PBE-seeded channel, and rewrites only the C2 entries of
`per_molecule.json`, the `w411_c2_atomization` row of `per_reaction.json`, the containing slice
rows of `test_set.csv` and a `reference_patch` stamp in `eval_metadata.json`, with the CSV and
JSON reconstruction proven byte-identical on all 99 channels that existed when the tool ran
before any patch logic executed (HISTORY 2026-09-01). Those 99 were the 25 then-complete specs'
$25 \times 4 = 100$ channels less one: `spec_0026` had not yet written a cold-start artifact.
The run now carries 108, which is $27 \times 4$ -- the two `shallow` cells of Section 9.2
contributing 8 channels and `spec_0026`'s completed cold-start the ninth. No training subset
ever contained c2 -- the reaction
`w411_c2_atomization` sits in the strict held-out slice of every completed cell with an empty
`in_sample_overlap` -- so no trained checkpoint was affected.

The state as pulled, verified by execution over the 27 completed specs: the seven repaired
specs (0019, 0020 and 0022-0026) carry a `reference_patch` stamp on three channels each --
`eval_holdout`, `eval_holdout_val_best` and `eval_holdout_coldstart` -- recording
`from_E_pbe` $= -75.7368945258$ Ha and `to_E_pbe` $= -75.8167407121$ Ha. On those three channels
the C2 atomization's PBE error now reads $-3.5457$ kcal/mol in all 27 cells, against
$-53.6499$ kcal/mol before, so the cross-spec reference guard is silent and the species has
rejoined every pooled PBE baseline. **The `eval_holdout_best` channel is not repaired**: 7 of
its 27 cells still carry $-53.6499$, pending a checkpoint fetch that must follow the push of the
patched channels back to the cluster. That channel is therefore excluded from every table in
this document -- which costs nothing, since it is the channel the figure suite stopped plotting
for independent reasons (Section 1.4).

### 9.4 The corrected v4gga validation-best record, per architecture (54 cells)

Within-cell beats over the same three slices (recomputed from
`merged_v4_arms/checkpoints/spec_*/eval_holdout_val_best/test_set.csv`; cell-matched PBE
anchors: W4-11 13.60--14.00, BH76 7.42, combined 11.59--11.96):

| arch | cells | W4-11 | BH76 | combined |
|---|---|---|---|---|
| deep_3x16 | 11 | 11/11 | 11/11 | 11/11 |
| deep_attn_3x16 | 11 | 9/11 | 3/11 | 9/11 |
| deep_cusp_3x16 | 11 | 5/11 | 6/11 | 5/11 |
| deep_rung35_3x16 | 11 | 2/11 | 7/11 | 2/11 |
| deep_rung35_attn_3x16 | 3 | 1/3 | 0/3 | 1/3 |
| deep_rung35ms_3x16 | 7 | 0/7 | 0/7 | 0/7 |
| total | 54 | 28/54 | 27/54 | 28/54 |

Best cells: W4-11 4.51 (deep_attn, ss=26), BH76 4.14 (deep_3x16, ss=2), combined 4.79
(deep_attn, ss=26). The unanchored deep_3x16 curve beats PBE on *every* slice at *every*
subset size -- BH76 4.14--6.52, W4-11 7.12--11.41, combined 6.84--9.74.

The merged directory has since acquired 7 `deep_mgga_3x16` cells (3/7 on W4-11, 4/7 on BH76,
3/7 combined), taking it to 61 cells; those are a meta-GGA record and are excluded from the GGA
table above and from every GGA comparison in this document. Note also that c2 is *absent* from
the strict slice of 54 of the merged cells and present with a PBE error of $-3.5638$ kcal/mol in
the remaining 7, so the v4 record is not affected by the incident of Section 9.3.

### 9.5 Reading

At matched capacity (medium pair against deep_3x16 pair), the anchored cells reproduce the
unanchored campaigns' W4-11 gains -- and do so uniformly (27 of 27 against 28 of 54; the
unanchored total is diluted by the retired-fidelity descriptor arms, whose pretraining never
delivered their parent, Section 4.2). On BH76 the picture inverts: the unanchored deep_3x16
beats PBE in all 11 cells (best 4.14) where the anchored cells beat it in 9 of 27 (best
6.46), and the signed decomposition (Section 8.3) locates the difference in the parent's
barrier bias, which the unanchored large-$s$ correlation freedom removed and the anchored
parameterization largely retains. Whether that is the anchor's price or a removable training
artifact is exactly what the queued anchored deep_3x16 group measures (Section 12). The
comparison carries the coverage caveats stated: 27 of 44 G1 cells, no `shallow_attn` cell at
all, one run per cell, and slices cell-matched within each campaign but not identical across
campaigns.

---

## 10. Energy and density together: the DFS-unit comparison

Everything above scores energies. The DFS Letter's central claim, however, is about the
*density*: that a functional trained through a differentiable SCF improves the self-consistent
density and not only the energy. The figure suite therefore carries a second family of views in
which energy and density errors are combined on one axis, and this section reads them across
generations.

### 10.1 The unit convention

Two quantities and one conversion define it.

**The density error.** The Letter's per-electron $L^1$ density error (its Eq. 20) is implemented
in `xcquinox/alec/evaluation.py:187-209` as

$$\varepsilon_{|n|} = \frac{\sum_i w_i\,\left|\rho_i - \rho^{\mathrm{ref}}_i\right|}{\sum_i w_i\,\rho^{\mathrm{ref}}_i},$$

the numerator and the electron count $N_e$ taken on the *same* quadrature so that grid
truncation partially cancels, and $N_e$ formed from the **reference** density so the measure is
charge-correct for ions (the vendored dpyscf instead counts neutral-atom $Z$). It is
dimensionless -- electrons per electron -- and is emitted per species as `density_eps_l1`
alongside a model-free PBE twin `density_eps_l1_pbe`. It is *not* the same quantity as the
suite's other density column, `density_rmse`, which is the grid-weight-averaged root mean square
$\sqrt{\sum_i w_i (\Delta\rho_i)^2 / \sum_i w_i}$ (`evaluation.py:283-291`); the two differ by
about a factor of 40 in magnitude on this basis and grid, and mixing them is a units error. The
equation number is the repository's own, verified against the Letter's PDF when the figure
notation was fixed and stated as such (HISTORY 2026-08-03); the PDF is not in the repository, so
the provenance of the equation numbering is the repo record.

**The combined metric.** The Letter's Eq. 21 combines an energy error $E$ (WTMAD-2, kcal/mol)
and a density error $D$ by a harmonic mean,

$$\mathcal{ED} = \frac{2}{\dfrac{1}{E} + \dfrac{1}{\gamma D}},$$

so that a functional must be good on both to score well.

**The conversion slope.** $\gamma$ carries units of kcal/mol per unit of $\varepsilon_{|n|}$. The
Letter fixes it as the zero-intercept regression of WTMAD-2 on $\varepsilon_{|n|}$ across six
nonempirical functionals (PW91, PBE, TPSS, revTPSS, SCAN, PBE0; its Fig. 3), obtaining
$\gamma = 1084.87$ kcal/mol on its own diet-GMTKN55 axes. That published value is the only
$\gamma$ hardcoded in the repository (`make_ablation_arch_figure.py:4375-4381`, `_DFS_GAMMA_KCAL`).
The same procedure re-run on this project's basis, grid, reference set and reaction list yields
an **own-axes** slope of $\gamma = 1158.336985911894$ kcal/mol, fitted by
$\gamma = \sum \varepsilon W / \sum \varepsilon^2$ over the same six functionals
(`make_ablation_arch_figure.gamma_zero_intercept` and `nonempirical_gamma`,
`:4394-4404` and `:4506-4583`; the calibration data from
`notebooks/analysis/precompute_nonempirical_pool.py`). That value was independently re-derived
from the calibration JSON and the canonical reaction list and agreed with the pipeline to one
unit in the last place, over all 216 reactions computable for all six functionals
(HISTORY 2026-08-02).

Which slope a given figure actually plots is decided at render time -- the own-axes fit when its
calibration cache resolves, the published slope otherwise -- and each panel stamps the value and
source it used (`make_ablation_arch_figure.py:8153-8158`). **Verified by execution over all six
DFS-units 3x3 CSVs read for this section: only the v3 figure sets carry the own-axes legs at
1158.336985911894; the v4gga, v4, v5, v6 G1 and merged-v4-GGA sets embedded below all carry
$\gamma = 1084.87$**, the Letter's published value, because the calibration cache does not
resolve for those runs. Every $\mathcal{ED}$ number quoted in this section is therefore at the
published slope, and cross-generation comparisons of $\mathcal{ED}$ below are internally
consistent for that reason.

One further distinction separates the two 3x3 variants. In the plain-units figure, $\gamma$ is
*self-calibrated per channel* as $E_{\mathrm{PBE}}/D_{\mathrm{PBE}}$ from that channel's own
anchors, so $\mathcal{ED}$ of PBE equals $E_{\mathrm{PBE}}$ by construction and the three
columns cannot be compared with one another. In the DFS-units figure one external $\gamma$ is
shared by all three columns, so $\mathcal{ED}_{|n|}$ *does* compare across columns and
$\mathcal{ED}_{|n|}$ of PBE does not equal $E_{\mathrm{PBE}}$
(`make_ablation_arch_figure.py:6203-6224`, the two caveat strings).

### 10.2 The anchored generation

![Per-channel held-out story in DFS units, anchored v6 G1. Columns: BH76, W4-11, combined. Row
1: WTMAD-2 per cell (kcal/mol). Row 2: cell-mean $\varepsilon_{|n|}$ against CCSD. Row 3: the
combined metric $\mathcal{ED}_{|n|}$ at one shared $\gamma$. PBE dashed in every
panel.](figures_dfs_step7_dfs6311_grid3_v6g1_size_val_best/ablation_density_energy_3x3_dfs_units.png)

Read down a column, not across a row: row 1 is what the network did to the energy, row 2 what it
did to the density, row 3 the combination. From
`ablation_density_energy_3x3_dfs_units.csv` (27 cells per column, $\gamma = 1084.87$):

| channel | WTMAD-2, NN / PBE | $\varepsilon_{\lvert n\rvert}$, NN / PBE | $\mathcal{ED}_{\lvert n\rvert}$, NN / PBE | energy beats | density beats | $\mathcal{ED}$ beats |
|---|---|---|---|---|---|---|
| BH76 | 20.19--42.26 / 24.85 | 0.00920--0.01607 / 0.00946 | 13.63--23.39 / 14.53 | 10/27 | 2/27 | 6/27 |
| W4-11 | 1.08--2.13 / 2.55 | 0.00885--0.01302 / 0.00893 | 1.95--3.66 / 4.03 | 27/27 | 2/27 | 27/27 |
| combined | 7.96--15.82 / 10.12 | 0.00912--0.01427 / 0.00921 | 9.12--14.39 / 10.06 | 13/27 | 1/27 | 8/27 |

The finding is in row 2. **The anchored cells improve the energy without improving the
density**: on the combined channel exactly one of 27 cells has a smaller per-electron density
error than PBE, and the best cell's $\varepsilon_{|n|}$ ($0.00912$) is 1 percent below PBE's
$0.00921$. The same asymmetry on W4-11, where the best-$\varepsilon_{|n|}$ cell is `medium` at
ss=1: its density error is 0.86 percent better than PBE's ($0.008855$ against $0.008932$) while
its WTMAD-2 is **53.56 percent** better ($1.1821$ against $2.5455$), or 52.25 percent on the
plain mean absolute error of Section 9.2. The energy is
beaten in every W4-11 cell and the density in two. **Conclusion:** at this coverage the anchored
networks are energy-fitted functionals whose self-consistent densities are indistinguishable
from -- and usually slightly worse than -- their parent's. That is precisely the failure mode the
DFS protocol's density-weighted loss exists to prevent, and it is a live finding, not a settled
one: the loss carries the Letter's density weight of 20 against reaction weight 1
(`LOSS_PRIMER.md:42-56`, verified to $2.2\times 10^{-16}$ over all 1400 optimizer updates), so
the density channel is not being ignored by the objective. Note also that WTMAD-2 here is the
one-bucket reduction on BH76 and W4-11 separately and the genuine two-subset form on the
combined column, so the row-1 numbers are not the mean absolute errors of Section 9.2 and should
not be read against them.

One presentational difference from Section 9 is worth stating, because the two "beats PBE"
counts are not the same test. Section 9 compares each cell against **its own slice's** PBE
anchor, so the anchor varies by cell; the `beats_pbe` column of these CSVs compares against a
single **pooled** PBE anchor per channel, which is why the PBE column above is one number rather
than a range. The two agree on the qualitative picture and need not agree cell for cell.

### 10.3 The unanchored generations

![Per-channel held-out story in DFS units, unanchored v4gga, six architectures at 54
cells.](figures_dfs_step7_dfs6311_grid3_v4gga_val_best/ablation_density_energy_3x3_dfs_units.png)

The same three rows for the unanchored GGA arm, per-arm slice. From that directory's
`ablation_density_energy_3x3_dfs_units.csv`, at the same $\gamma = 1084.87$: BH76 WTMAD-2 spans
11.60--543.47 against PBE's 21.23, $\varepsilon_{|n|}$ spans 0.00759--0.04746 against 0.00943,
and $\mathcal{ED}_{|n|}$ spans 11.60--94.07 against 13.81; the beat counts are 27/54 on energy,
**27/54 on density**, and 31/54 on the combined metric. The W4-11 column reads 27/54, 15/54 and
28/54 on the same three legs, the combined column 25/54, 18/54 and 27/54.

Two things separate this from the anchored panel. The **range** is enormous -- three of the
retired-fidelity descriptor architectures produce WTMAD-2 values in the hundreds and density
errors five times PBE's, which is what a network that started 25.7 to 56.1 kcal/mol from its
parent on its worst system and then diverged looks like. But the **density beat rate is an order of magnitude higher**:
27 of 54 BH76 cells improve the per-electron density error where 2 of 27 anchored cells do, and
the best unanchored density error, $0.00759$, is 20 percent below PBE's where the best anchored
one is 3 percent below. **Conclusion:** the unanchored parameterization moved the density and
the anchored one has not. Read together with Section 8.2 this is consistent: the large-gradient
correlation freedom the anchor suppresses is exactly the region that distinguishes one
self-consistent density from another, and the anchored networks have been buying their energy
improvements in the small-gradient region where the pre-image leaves them trainable.

![Per-channel held-out story in DFS units, the merged v4 GGA record on the union-removed
comparison slice, 47 cells.](figures_dfs6311_v4_merged_val_best_gga/ablation_density_energy_3x3_dfs_units.png)

This third panel is embedded because it is not a redundant view of the previous one: it is the
same architectures scored on a **different slice** -- the cross-arm merged slice, which removes
the union of both arms' validation reactions rather than each arm's own, and which drops the
diverged rung-3.5-multishell cells. It is the cleaner statement of what the unanchored protocol
achieved when it worked. From its CSV: BH76 WTMAD-2 8.92--41.81 against PBE's 16.48,
$\varepsilon_{|n|}$ 0.00681--0.01338 against 0.00903, $\mathcal{ED}_{|n|}$ 9.64--19.04 against
12.29, with **43 of 47 cells beating PBE on the combined metric, 41 of 47 on energy and 30 of 47
on density**. No single architecture wins every leg: the best WTMAD-2 is `deep_3x16` at ss=6
(8.92084), the best $\varepsilon_{|n|}$ is `deep_cusp_3x16` at ss=26 (0.006805), and the best
$\mathcal{ED}_{|n|}$ is `deep_attn_3x16` at ss=18 (9.63922) -- the descriptor architecture takes
the density leg while the descriptor-free one takes the energy leg. **Conclusion:** on a slice where
the failed architectures do not dominate the range, the unanchored protocol improved the density
in about two thirds of its BH76 cells -- a genuine density result of the kind the Letter reports,
and the specific thing the anchored generation has not yet reproduced. It also carries the
caveat that makes it not a clean comparison target: those same cells' pretrained networks were
25.7 to 56.1 kcal/mol from their parents on their worst systems, so their density improvement is
not attributable to a faithful starting point.

### 10.4 The plain-units view, and why one is enough

![Per-channel held-out story in plain units, anchored v6 G1: row 2 is the grid-weight-averaged
density RMSE and row 3 uses a per-channel self-calibrated
$\gamma$.](figures_dfs_step7_dfs6311_grid3_v6g1_size_val_best/ablation_density_energy_3x3.png)

The plain-units twin is the same data under a different density measure and a different
conversion. Row 1 is byte-for-byte the DFS-units row 1. Row 2 plots `density_rmse` instead of
$\varepsilon_{|n|}$: from `ablation_density_energy_3x3.csv` the combined-channel values span
$2.075\times 10^{-4}$ to $3.172\times 10^{-4}$ against PBE's $2.298\times 10^{-4}$, where the
same cells' $\varepsilon_{|n|}$ span 0.00912 to 0.01427 -- the two measures differ by a factor
of about 40 and, being an $L^2$ and an $L^1$ measure on different normalizations, they are not
interconvertible by a constant. Row 3 uses the self-calibrated slope, which the CSV records as
$\gamma = 120154.3$ (BH76), $10656.7$ (W4-11) and $44039.3$ (combined) kcal/mol per unit RMSE --
by construction $E_{\mathrm{PBE}}/D_{\mathrm{PBE}}$ per channel, which is why PBE's
$\mathcal{ED}$ equals its WTMAD-2 exactly in every column (24.85, 2.55, 10.12) and why these
columns must not be compared with one another.

**Conclusion, and the redundancy judgment:** the plain-units view reproduces the qualitative
finding of Section 10.2 -- the anchored cells improve energy and not density -- under an
independent density norm, which is the reason to show it once. It adds nothing beyond that, and
its self-calibrated $\gamma$ makes it strictly less useful for cross-channel and
cross-generation comparison, so the plain-units twins of the other generations, the `_logy`
variants of all of them, and the `ablation_density_energy_overview*`,
`ablation_ed_decomposition*` and `ablation_density_parity_by_channel*` families (which
re-decompose the same three quantities per species and per architecture) are not reproduced
here.

---

## 11. The generation comparison

Every cell of the following table is sourced in the section named beside it.

| | v4 / v4gga | v5 / v5mgga2 | v6 |
|---|---|---|---|
| **Pretraining objective** | integration-weighted point-wise $F$ residual, 2500 steps; no energy term, no validation, no gate (Sec. 4.1) | unchanged from v4 -- the shared pretrain data file could not regenerate, so the pretrained pairs are the v4 objects (Sec. 5) | same residual plus an optional per-system parent-energy term, run at weight 0.0 under the anchor because both terms are zero at initialization; 20% seeded validation with patience, best-validation network written (Sec. 6.3) |
| **SCF seeding** | converged PBE for every rung | converged PBE for GGA, converged SCAN for meta-GGA (`seed_xc: auto`); a fourth cold-start channel added (Sec. 5, Sec. 1.4) | as v5 |
| **Anchor** | none: $F = 1 + L(T(g))$, start at the LDA limit $F \equiv 1$ (Sec. 2) | none | $F = 1 + L(z_{\mathrm{parent}} + T(g))$; the parent returned exactly at $T = 0$, to one unit in the last place at $\Lambda = 1.174$ and bitwise at 1.804 and 2.0 (Sec. 6.1) |
| **Coordinates** | legacy: $s$ through $\{1-e^{-x^2}\}\ln(x+1)$, C-net density feature $r_s$ through the same $s$-style transform -- a documented deviation from DFS Eq. 7 (Sec. 3.7) | as v4 | DFS: $x_s$ (Eq. 9), $x_0 = \ln(\rho^{1/3}+10^{-5})$ (Eq. 7), $x_1$ (Eq. 4), $x_\alpha$ (Eq. 10) (Sec. 3.7) |
| **Open-shell footing** | total-density descriptors passed into both spin channels -- the meta-GGA spin-scaling defect (Sec. 4.3) | unrepaired | symmetric doubled density $\mathrm{diag}(P_\sigma,P_\sigma)$ everywhere; libxc spin-polarized SCAN exchange reproduced to $1.8\times 10^{-15}$ Ha on O and OH (Sec. 6.2) |
| **Acceptance gate** | none | none | per-architecture fidelity certificate: PASS requires $\max$ atomic $\lvert\Delta E_{xc}\rvert \leq 1.0$ mHa and $\max\lvert\Delta AE\rvert \leq 1.0$ kcal/mol on 38--39 systems through the production energy path; enforced unconditionally by every record layer (Sec. 6.4) |
| **First-step pretrain loss** | $1.591407\times 10^{-2}$, identical across all six GGA architectures (the v4gga run's own `pretrain/*/losses_x.npy`); the unanchored G1 control submission reads 0.008--0.012 (Sec. 6.4) | as v4 | $2.72\times 10^{-32}$ (PBE parent); $3.02$--$4.31\times 10^{-14}$ (SCAN parent, the `_ALPHA_MAX` ceiling, Sec. 6.5) |
| **Handoff fidelity, curve metric** | $\max\lvert\Delta F_x\rvert$ 0.039--0.090 (GGA); meta-GGA $\max\lvert\Delta F_c\rvert$ up to 0.49 (Sec. 4.2, Sec. 5) | identical to v4 by construction; the stacking arm reaches 0.52 (Sec. 5) | $8.7\times 10^{-7}$--$9.2\times 10^{-6}$ in $F_x$ (G1); $8.1\times 10^{-7}$--$1.3\times 10^{-5}$ over both channels (meta-GGA) (Sec. 6.6) |
| **Handoff fidelity, energy** | atomization offsets $-2.3$ to $-56.1$ kcal/mol on H2O/N2/CH4; worst-system offset per descriptor-carrying architecture 25.7--56.1 kcal/mol against 4.1--4.2 for the descriptor-free controls, individual systems overlapping (Sec. 4.2) | unmeasured until 2026-08-20, then the same (Sec. 5) | certificates PASS at $7.2\times 10^{-4}$--$5.15\times 10^{-3}$ mHa atomic and $1.9\times 10^{-3}$--$2.5\times 10^{-3}$ kcal/mol on atomization (Sec. 6.4) |
| **Held-out headline (GGA, validation-best, within-cell)** | W4-11 28/54, BH76 27/54, combined 28/54; best cells 4.51 / 4.14 / 4.79 kcal/mol (Sec. 9.4) | no GGA cells (its GGA rows are the v4 rows) | W4-11 27/27, BH76 9/27, combined 26/27 at 27 of 44 cells; best cells 5.99 / 6.46 / 7.00 kcal/mol (Sec. 9.2) |
| **Held-out density (DFS units, combined channel)** | $\varepsilon_{\lvert n\rvert}$ beats PBE in 18/54 on the per-arm slice and 18/47 on the merged slice (both from the `ablation_density_energy_3x3_dfs_units.csv` of the respective directory); BH76 30/47 on the merged slice (Sec. 10.3) | -- | 1/27 (Sec. 10.2) |
| **Status of the record** | retired for every descriptor-carrying architecture; the descriptor-free `deep_3x16` and `deep_attn_3x16` rows stand (Sec. 4.3) | meta-GGA record retired as a quantitative result; v5mgga2 never trained (Sec. 5) | live |

---

## 12. Open questions the next results answer

1. **The controlled anchored-vs-unanchored test.** The G2a core trio (deep_3x16,
   deep_attn_3x16, deep_cusp_3x16 under the full v6 protocol with `parent_anchor: true`) is
   queued behind the draining G1 group (HISTORY 2026-08-30, the trio split; HISTORY
   2026-08-31). It meets the strongest unanchored record at the registry identity itself: the
   G1 medium pair already realizes the deep_3x16 capacity with both differing flags inert
   for that descriptor-free pair (Section 9.1), so the trio's deep_3x16 and deep_attn cells
   test the anchored result's
   reproducibility at the registry names themselves, and its deep_cusp cells extend the
   comparison to a descriptor form G1 does not carry; the two group files share an
   identical subset axis including ss=26 (`dfs_step7.dfs6311_grid3_v6g1_size.yaml` line
   129, `...v6g2a_families_core.yaml` line 122), so G1's ss=26 cells are pending rather
   than absent (item 4). The BH76 signed bias (Section 8.3) is the
   discriminating observable; the G1 spread ($-7.75$ at
   medium/ss=12 against $-0.81$ at medium_attn/ss=12) says the outcome is not foreclosed.
2. **The density question.** The anchored G1 cells improve energies almost everywhere and the
   per-electron density error almost nowhere -- 1 of 27 combined-channel cells against 18 of 47
   for the unanchored merged record (Section 10). Two candidate explanations are separable by
   the queued groups: that the pre-image suppression of large-$s$ correlation (Section 8.2)
   removes the freedom a density improvement needs, in which case the anchored deep_cusp and
   rung-3.5 cells will show the same pattern; or that the unanchored density improvements were
   an artifact of networks that started far from their parent and had large corrections to
   build, in which case they should not survive the anchored re-run of the same architectures.
3. **The meta-GGA trained factors.** The five anchored meta-GGA family architectures hold
   PASS certificates at production identity and pretrained curves within
   $8.1\times 10^{-7}$--$1.3\times 10^{-5}$ of SCAN (Sections 6.4-6.6); their training cells
   are the pending half of `figures_dfs_step7_dfs6311_grid3_v6g2_families_mgga`. They answer
   whether the anchored fine-tune preserves SCAN-level held-out accuracy where the v5
   SCAN-seeded (but mis-footed, unanchored) cells only matched it at subset sizes 2--5
   (Section 5), and whether the correlation-collapse mechanism of Section 8.2 repeats against
   the SCAN parent -- whose correlation, unlike PBE's, retains a finite floor at large $s$
   (Section 2), so the suppression should be materially weaker.
4. **G1 completion.** 17 G1 cells remain (all 11 shallow_attn cells, the four remaining
   shallow sizes, and the ss=26 cells of medium and medium_attn -- the medium ss=26 cell
   needs the open NaN-gradient defect closed, HISTORY 2026-08-31), after
   which the medium-pair-vs-anchored-deep-pair comparison of item 1 -- registry names
   differing in three fields, all inert under the v6 model block for the descriptor-free
   pair (Section 9.1) -- can be
   read at full coverage. Until then the size ladder's attention comparison is one-sided:
   no `shallow_attn` cell exists. Also pending is the checkpoint fetch that lets the
   `eval_holdout_best` channel be repaired for the seven c2-affected specs (Section 9.3).

---

## 13. Sources

**Primary literature.**

* S. Dick and M. Fernandez-Serra, *Phys. Rev. B* **104**, L161109 (2021) -- the DFS Letter.
  Cited for the network coordinates (Eqs. 4, 7, 9, 10), the UEG-recovery gate and correlation
  transform (Eqs. 12, 13), the density-error measure (Eq. 20) and the combined metric (Eq. 21),
  the conversion slope $\gamma = 1084.87$ kcal/mol (Fig. 3), the loss weights
  $\lambda_{RE} = 1$, $\lambda_n = 20$, $\lambda_E = 0.01$ (after Eq. 18), the 25-cycle SCF with
  trajectory weights $w_j = ((j-10)/15)^2$ (Eqs. 15-16), and the 21-molecule G2/97 training set
  (SI Sec. II). Equation numbers and the quoted weight and cycle-count statements are carried by
  the repository record -- HISTORY 2026-07-29 (with its 2026-08-03 erratum on the pipeline's own
  weighting, `HISTORY.md:643`), `notebooks/analysis/LOSS_PRIMER.md:42-47`
  and `:237-245`, `xcquinox/alec/dfs_pool.py:1-12` -- which transcribes them from the PDF; the
  PDF itself is not in the repository, so the provenance of the equation numbering is the repo
  record rather than an independent reading here.
* J. Sun, A. Ruzsinszky and J. P. Perdew, *Phys. Rev. Lett.* **115**, 036402 (2015) -- SCAN. The
  iso-orbital indicator (its Eq. 2), the exchange ceiling $h_x^0 = 1.174$
  (`parents.py:282`, `SCAN_H0X`), and the $\alpha = 0$ / $\alpha = 1$ slice convention of its
  Fig. 1, which the enhancement-factor figures follow.
* J. P. Perdew, K. Burke and M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) -- PBE, the
  GGA-rung parent; $\kappa = 0.804$ sets $\Lambda = 1.804$.
* J. P. Perdew and Y. Wang, *Phys. Rev. B* **45**, 13244 (1992) -- PW92, the correlation
  baseline; eqs. 8-9 for $\epsilon_c(r_s, \zeta)$ and the spin interpolation $f(\zeta)$.
* G. L. Oliver and J. P. Perdew, *Phys. Rev. A* **20**, 397 (1979) -- the exact exchange spin
  scaling $E_x[\rho_\alpha, \rho_\beta] = (E_x[2\rho_\alpha] + E_x[2\rho_\beta])/2$.
* T. Kato, *Commun. Pure Appl. Math.* **10**, 151 (1957) and F. Steiner, *J. Chem. Phys.* **39**,
  2365 (1963) -- the wavefunction and density cusp conditions behind the `cusp` descriptor.
* B. G. Janesko, arXiv:2206.07118, Eqs. 12-13, and P. Verma et al., *J. Chem. Theory Comput.*
  **15**, 4804 (2019) -- the rung-3.5 occupancy. Both identifiers are carried by the repository
  (`rung35.py:3-8`, `descriptors.py:321-324`); the attribution of the default projector width to
  the M11plus kernel scale $d^2 = 5\,a_0^2$ is the repo record's own and is still to be confirmed
  against the library copy (`CAMPAIGN_V6.md:229-230`), so it is not asserted here as a read of
  the paper.
* S. Dick and M. Fernandez-Serra, *Nat. Commun.* **11**, 3509 (2020) -- NeuralXC, the localized
  density-matrix projection the multishell descriptor implements the radial part of.
* L. Goerigk, A. Hansen, C. Bauer, S. Ehrlich, A. Najibi and S. Grimme, *Phys. Chem. Chem. Phys.*
  **19**, 32184 (2017) -- GMTKN55, the source of the BH76 slice.
* A. Karton, S. Daon and J. M. L. Martin, *Chem. Phys. Lett.* **510**, 165 (2011) -- W4-11.

**Repository records cited as such.** `xcquinox/alec/HISTORY.md` (the canonical development
record; entries cited by date), `xcquinox/alec/CAMPAIGN_V6.md`, `xcquinox/alec/SPEC_parent_anchor.md`,
`xcquinox/alec/SPEC_pretrain_fidelity_program.md`, `notebooks/analysis/LOSS_PRIMER.md`,
`notebooks/analysis/HOLDOUT_SET.md`, `notebooks/analysis/NOTES_v5_mgga_vs_scan.md`,
`notebooks/analysis/DM_DESCRIPTOR_SPEC.md`, `notebooks/analysis/README_density_figures.md`.

**Figure-generating code.** `notebooks/analysis/report_equation_figures.py` (the
governing-equation figures, drawn by calling the repository's own functions),
`notebooks/analysis/pretrain_fx_fc.py` and `trained_fx_fc.py` (the per-generation
enhancement-factor sets), `notebooks/analysis/anchored_vs_unanchored_fx_fc.py` (the
cross-generation 2x2), and `notebooks/analysis/make_ablation_arch_figure.py` (the `ablation_*`
suite). The equation-figure and enhancement-factor modules write a long-form CSV for every
figure they render; the `ablation_*` suite does so only for some, and the three `ablation_*`
figures embedded here have none. Every number quoted in a caption above was read either from
that figure's own CSV or, where none exists, from the per-spec artifacts named in the caption.
