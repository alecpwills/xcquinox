# Pretraining evolution across the campaign generations (v4, v5, v6)

This document records how the pretraining stage of the dfs6311 campaigns changed from the
v4 generation (unanchored point-wise fits to the parent's enhancement factors), through v5
(per-rung self-consistent seeding around an unchanged pretraining scheme), to v6 (parent-anchored
networks with a machine-checked fidelity certificate), together with the measured findings that
forced each change and the first quantitative comparison of the completed v6 GGA cells against
the corrected v4 GGA record. Every number carries its source: an entry of
`xcquinox/alec/HISTORY.md` (cited as `HISTORY <date>`), a code definition (cited as
`file:symbol`), a figure CSV under `notebooks/analysis/figures_*`, or a pulled cluster artifact
(cited relative to the local results tree `xcquinox-results/runs/`, e.g.
`dfs_step7/dfs6311_grid3_v6g1_size/runs/run_20260827T163330Z/...`). Values re-derived here were evaluated directly from the
named artifacts and committed implementations. Coverage is stated per number; the campaign is live and several sets below are
partial. Current as of 2026-08-31.

The functional family under study descends from Dick and Fernandez-Serra, Phys. Rev. B 104,
L161109 (2021) ("DFS" below): 3x16 multilayer perceptrons returning exchange and correlation
enhancement factors over an LDA baseline, trained through a differentiable self-consistent
field. The meta-GGA rung uses the iso-orbital indicator of SCAN (Sun, Ruzsinszky and Perdew,
Phys. Rev. Lett. 115, 036402 (2015)).

---

## 1. The common frame: networks, bounded maps, parent conventions

All three generations share the network form (`xcquinox/alec/networks.py`). Per grid point the
exchange network reads the reduced gradient $s = |\nabla\rho| / (2 k_F \rho)$ of one spin
channel posed on its doubled density, and the correlation network reads the total density with
the spin polarization $\zeta$; descriptor-carrying architectures append extra columns (cusp,
rung-3.5 occupancies, the meta-GGA indicator). The raw MLP output is gated by a UEG-recovery
prefactor ($\tanh^2 s$ at the GGA level; the DFS Eq. 12 form $(x_2 + \tanh^2 x_3)$ at the
meta-GGA level) and passed through a bounded squash with a static limit
(`networks._AlecLOB`):

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
  exactly (Section 4.1).

The parent is fixed per rung: PBE (Perdew, Burke and Ernzerhof, Phys. Rev. Lett. 77, 3865
(1996)) for GGA-rung architectures, SCAN for meta-GGA ones
(`parents.parent_for_arch`; `cluster/fidelity.resolve_parent` reads the same predicate).
Exchange is posed per spin channel on the doubled density, following the exact spin scaling
$E_x[\rho_\alpha, \rho_\beta] = (E_x[2\rho_\alpha] + E_x[2\rho_\beta])/2$ (Oliver and Perdew,
Phys. Rev. A 20, 397 (1979)); correlation is posed on the total density with $\zeta$, its
target ratio formed against the polarized PW92 baseline (Perdew and Wang, Phys. Rev. B 45,
13244 (1992)) the model itself multiplies (`parents.py` module docstring).

---

## 2. v4: unanchored pretraining to parent enhancement values

### 2.1 Protocol

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

### 2.2 What the handoff actually delivered

Measured from the pretrained checkpoints of the v4 GGA arm
(`figures_dfs_step7_dfs6311_grid3_v4gga_val_best/pretrain_fx_fc_curves.csv`): the
pretrained exchange curves sit $\max|\Delta F_x| = 0.039$ (deep_3x16) to
$0.090$ (deep_rung35_3x16) from PBE over the plotted grid; the meta-GGA correlation nets sit
up to $0.49$--$0.52$ from SCAN
(`figures_dfs_step7_dfs6311_grid3_{v4,v5}_val_best/pretrain_fx_fc_curves.csv`,
`.../v5mgga2/pretrain_fx_fc_curves.csv`). The final point-wise exchange loss of the v4
deep_3x16 pretrain was $4.6\times 10^{-5}$ (same metadata file) -- for comparison, the v6
anchored runs *start* at $2.7\times 10^{-32}$ (Section 4.5).

In energy units, the 2026-08-20 probe (frozen parent densities, production evaluation path,
every value re-derived independently; HISTORY 2026-08-20, "Pretraining does not deliver the
parent") measured atomization-energy offsets from the parent on H2O / N2 / CH4 at pretraining
handoff of $-2.5 / -4.2 / -2.4$ kcal/mol (deep_3x16) and $-2.3 / -4.1 / -3.1$
(deep_attn_3x16) for the descriptor-free networks, against $-13.2 / -4.2 / -25.7$
(deep_cusp), $-13.5 / -3.5 / -29.1$ (deep_rung35), $-29.5 / -20.4 / -56.1$
(deep_rung35_attn) and $-22.0 / -30.9 / -42.8$ (deep_rung35ms): every descriptor-carrying
architecture entered its 200-epoch fine-tune 13--56 kcal/mol from its parent on the reaction
type it would be scored on. The pretraining loss did not see it -- the architecture with the
lowest exchange residual carried the largest offset -- because a point-wise objective does not
control the integral (HISTORY 2026-08-20; `SPEC_pretrain_fidelity_program.md` Section 2).

### 2.3 The two defects behind the offsets

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
   doubled-spin substitution, and the rung-3.5 occupancy has no in-domain doubled-spin
   evaluation. They decompose into an H-atom pretraining error shared by every cusp-carrying
   network (+13.7 mHa against +0.8) multiplied by the hydrogen count, plus molecular
   extrapolation of density-matrix features never constrained by the atoms-plus-mesh
   pretraining set (HISTORY 2026-08-20).

No campaign stage had ever compared a pretrained network with its parent on atoms or
atomization energies; the offsets were larger than the architecture differences the campaigns
existed to resolve, and this held while the campaigns ran. Every in-flight training was
cancelled on 2026-08-20 and the v4/v5 descriptor-architecture record retired (HISTORY
2026-08-20).

---

## 3. v5: per-rung self-consistent seeding

v5 changed where the truncated training SCF *starts*, not how the networks are pretrained
(HISTORY Phase 37, 2026-08-14):

* **What changed.** `precompute_fixed_density_data` gained a `dm_seed` channel; meta-GGA-family
  architectures train and evaluate their truncated (3-cycle) SCF from a converged SCAN density
  matrix instead of converged PBE, matching each rung's own baseline (the DFS reference work
  prepares per-campaign converged PBE-or-SCAN baselines the same way). A fourth held-out
  evaluation channel (`eval_holdout_coldstart`: functional-free minao seed, 25 cycles,
  `conv_tol` $10^{-12}$) was added as a trajectory diagnostic. The v5 arm YAMLs are
  byte-derived from v4 with exactly five deltas (seed source, seed cache, coldstart flag,
  output root, eval wall); losses, solver, subsets, hyperparameters and the pretrain block are
  untouched, and the shared pretrain data file could not regenerate, so the v5 pretrained
  pairs replicate the v4 protocol and land at the same distances from the parent (the v4 and
  v5 meta-GGA pretrain curve CSVs carry identical deviations,
  `figures_dfs_step7_dfs6311_grid3_{v4,v5}_val_best/pretrain_fx_fc_curves.csv`).
* **What it fixed.** The density-footing asymmetry: previously every rung's truncated SCF was
  seeded from PBE's fixed point while SCAN comparison numbers used converged SCAN, a handicap
  concentrated on the beyond-GGA arms.
* **What it did not fix.** Neither defect of Section 2.3. The 2026-08-20 read of the
  SCAN-seeded cells against SCAN (HISTORY 2026-08-20; `notebooks/analysis/NOTES_v5_mgga_vs_scan.md`)
  found the cells reproducing SCAN's held-out accuracy at subset sizes 2--5
  ($E/E_{\mathrm{SCAN}}$ 0.94--1.01) with one of 21 (leg, subset) cells beating SCAN, and
  identified the limiting factor as the fine-tuning regime itself: both parents' BH76 error is
  ~90% bias (PBE $-7.5$, SCAN $-6.0$ kcal/mol of MAE 7.7 / 6.4); fine-tuning removes the bias
  and injects reaction-level scatter nearly uncorrelated with the parent's residual -- a
  property of the recipe, not of the seeding. The same read exposed that the cells never
  started at SCAN at all (Sections 2.2-2.3), which retired the v5 meta-GGA record as a
  quantitative result. Its trained enhancement-factor curves
  (`figures_dfs_step7_dfs6311_grid3_v5_val_best/trained_fx_fc_*`) document a retired record
  and are kept for the development narrative.

---

## 4. v6: the parent-anchored pretraining protocol

v6 replaces "pretrain toward the parent and hope" with "construct the parent, then certify".
Four coupled changes: the anchored network class, the exact open-shell footing with a smoothed
indicator, a pretraining protocol with parent energies and validation, and a per-architecture
fidelity certificate that gates every campaign stage.

### 4.1 The anchored construction

With `parent_anchor` set (every v6 configuration; `hpcjobs/configs/dfs_step7.dfs6311_grid3_v6*.yaml`,
`model.parent_anchor: true`), the gated network term enters in the *pre-image* of the bounded
map at the parent's value (`networks.AlecGGA_XNet._core`, `AlecGGA_CNet._core`):

$$F(g) = 1 + L\!\left(z_{\mathrm{parent}} + T(g)\right), \qquad
z_{\mathrm{parent}} = \ln\!\left[\frac{(\Lambda - 1)\,F_{\mathrm{parent}}}{\Lambda - F_{\mathrm{parent}}}\right]
\;\text{clamped to } [-40, 40],$$

with $F_{\mathrm{parent}}$ evaluated on the row's own physical inputs by the JAX parent
implementations (`parents.pbe_fx` / `pbe_fc` / `scan_fx` / `scan_fc`, every constant the value
libxc 7.0.0 carries) and the pre-image `parents.lob_preimage`. The clamp at $z_{\max} = 40$
binds only where the parent approaches a bound of $(0, \Lambda)$: within
$\Lambda(\Lambda - 1)e^{-40}$ of the ceiling ($8.5\times 10^{-18}$ at $\Lambda = 2$,
$6.2\times 10^{-18}$ at 1.804) and within $\Lambda e^{-40}/(\Lambda - 1)$ of the floor
($9.53\times 10^{-18}$ at 1.804, $2.87\times 10^{-17}$ at 1.174); both logarithm arguments
are floored at the smallest normal float so a parent at or past a bound by round-off clamps
instead of returning NaN (`parents.lob_preimage` docstring, as corrected). The construction guarantees
$F \in (0, \Lambda)$ for every $T(g)$ and returns $F_{\mathrm{parent}}$ exactly at
$T(g) = 0$.

`create_network_pair` forces `zero_init_final_layer` for an anchored pair, so both networks
equal the parent at initialization by construction. Measured: the anchored meta-GGA
architectures return $F_x = $ `scan_fx` within $2.8\times 10^{-16}$ and $F_c = $ `scan_fc`
within $2.2\times 10^{-16}$ on 31,550 exchange and 15,790 correlation rows of OH and H2O
(HISTORY 2026-08-25); a freshly built anchored pair reproduces the parent curves under
$10^{-10}$ ($F_x$) and $10^{-8}$ ($F_c$) on the figure grid where an unanchored build differs
by more than $10^{-2}$ (HISTORY 2026-08-30, the pretrained-figure identity pins). An anchored
correlation network must be polarization-aware ($\zeta$-blind construction refused by name),
because the parent correlation is divided by the model's $\zeta$-dependent PW92 baseline
(`networks.create_network_pair`).

The v6 runs use the DFS descriptor coordinates (`model.descriptor_coordinates: dfs`; the
trained-model class records state it, e.g.
`dfs_step7/dfs6311_grid3_v6g1_size/runs/run_20260827T163330Z/checkpoints/spec_0009/model_val_best.eqx.class.json`):
the exchange MLP reads $x_s = (1 - e^{-s^2})\ln(s + 1)$ of the doubled channel and, on the
meta-GGA rung, $x_\alpha = \ln((\alpha + 1)/2)$ of the raw indicator -- the network inputs of
DFS Eqs. 9-10.

### 4.2 Exact open-shell footing and the smoothed indicator

Every density-matrix-derived descriptor is now evaluated on the symmetric doubled density of
its own spin channel, $\mathrm{diag}(P_\sigma, P_\sigma)$ -- in the precompute, the SCF loop of
both solver backends, the energy, the potential, the losses and the pretraining rows -- so the
open-shell exchange is the same functional the closed-shell path defines. With libxc SCAN in
place of the network, the three-block energy reproduces PySCF's spin-polarized SCAN exchange
to $1.8\times 10^{-15}$ Ha on O and OH, and the assembled potential is the finite-difference
derivative of the energy to $1.0\times 10^{-10}$ Ha worst case; closed-shell paths are bitwise
unchanged (HISTORY Phase 43, 2026-08-20 to 2026-08-24).

The meta-GGA indicator (`metagga.compute_alpha`) is
$\alpha = (\tau - \tau_W)/\tau_{\mathrm{unif}}$, stored as
$\min(p(\alpha_{\mathrm{raw}}), 100)$ with $p$ the smooth positive part

$$p(x) = \tfrac{1}{2}\left(x + \sqrt{x^2 + w^2}\right), \qquad w = 10^{-5},$$

which replaced a hard lower clip whose derivative was discontinuous exactly on the manifold
every one-electron spin channel occupies (`metagga.smooth_positive_part`;
`_ALPHA_SMOOTHING_WIDTH = 1e-5`, `_ALPHA_MAX = 100`). The width is anchored to measurement,
not chosen: it must dominate the floating-point residue of $\tau - \tau_W$ on one-orbital
channels (worst on-domain residue $1.3$--$3.7\times 10^{-6}$, margin 2.7--7.7x), and its SCAN
energy cost is $1.17\times 10^{-7}$ Ha on the H atom, linear in the width, $8.5\times 10^{3}$
below the certificate's free-atom tolerance (`metagga.py` width commentary; HISTORY
2026-08-24). The definition string `metagga.ALPHA_DEFINITION` is part of the pretraining-data
manifest identity, so a file computed under another definition is stale by construction.

### 4.3 The pretraining set and objective

* **Systems.** The DFS pretraining set (eight free atoms and 22 G2/97 molecules, committed as
  package data at `xcquinox/alec/data/dfs_pretrain_set.json`, its bytes pinned by the
  SHA-256 in `xcquinox/alec/tests/test_dfs_pretrain_set.py`, line 189) at the rung's
  level -- the meta-GGA variant of the DFS protocol drops H2 and N2
  (`cluster/fidelity.dfs_level_for_parent`) -- merged with every free atom of the BH76 /
  W4-11 pools at the pools' own charges and spins, de-duplicated by geometry
  (`pretrain.resolve_pretrain_systems`; the G1 run records 38 systems and 1.39M exchange /
  1.21M correlation rows, `run_20260827T163330Z/pretrain/medium/pretrain_metadata.json`).
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

### 4.4 The per-architecture fidelity certificate

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

### 4.5 The measured pretraining floors, and the SCAN-parent floor derivation

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
(`networks._raw_indicator` / `metagga.invert_smooth_positive_part`): a stored
$p(0) = w/2 = 5\times 10^{-6}$ inverts to $\alpha = 0.0$ to round-off, the end-to-end
anchored exchange MSE on the committed mesh block is $7.62\times 10^{-32}$, and the H atom
-- one-orbital on every row, the column at its smoothed floor everywhere -- prices at
$2.85\times 10^{-32}$ (HISTORY 2026-08-31 (erratum)). At the ceiling the inversion cannot
act: the column reads 100 while the exact indicator on the capped rows is unrecoverable.
The capped population is the low-density tail: on the O atom 2492 rows are capped, with
$\alpha_{\mathrm{exact}}$ spanning $\sim 1.0\times 10^{2}$ to $\sim 7.0\times 10^{6}$
(median 555) at $\rho$ from $10^{-10}$ to $8.6\times 10^{-5}$ (median
$6.1\times 10^{-8}$), 90% of the capped weight sitting at $\alpha = 108$--$2475$. SCAN's
switching function has nearly but not exactly saturated there: the parent at $\alpha = 100$
sits $|\Delta F|$ of median $2.55\times 10^{-4}$ and at most $5.70\times 10^{-4}$ from the
exact-$\tau$ libxc target per capped row, the ceiling residual saturating at
$1.74\times 10^{-3}$ at $s = 0$ (`parents.scan_fx` at $\alpha = 100$ against its
large-$\alpha$ limit). The capped rows carry 100.0% of the weighted exchange MSE on the O
atom ($1.2593\times 10^{-13}$ on one converged reference path, $1.2190\times 10^{-13}$ on a
second, the capped share 1.000000 on both) and on H2O ($3.07\times 10^{-14}$), the uncapped
remainder pricing at $2.7\times 10^{-29}$ and 0.0 (HISTORY 2026-08-31 (erratum, as
amended)).
The synthetic mesh contributes nothing ($\leq 6\times 10^{-29}$; its $\alpha$ nodes stop at
5, below the ceiling -- `pretrain_data_gen.py` line 1070, `MESH_ALPHA`), so the
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
$1.8\times 10^{-3}$ relative on the capped rows; and the certificates gate the whole
start's consequence at 194x inside the atomic tolerance (worst meta-GGA `max_atom`
$5.15\times 10^{-3}$ mHa, deep_mgga_attn, against 1.0 mHa). The corrected derivation is recorded
so the $10^{-14}$ start reads as the priced consequence of a documented bound rather than
as a precision regression.

### 4.6 Pretrained enhancement factors

The pretrained curves are now a published, per-generation artifact
(`notebooks/analysis/pretrain_fx_fc.py`, loading through the production certified-model
builder; baselines are `parents.pbe_fx` / `pbe_fc` -- the anchor's own parent at libxc
constants, because a rounded-constant analytic helper differs by up to $4.553\times 10^{-6}$
and would read as a spurious learned correction under the anchor; HISTORY 2026-08-30):

* v6 G1 (anchored, PBE parent), after the 2500-step pretrain: $\max|\Delta F_x| =
  8.7\times 10^{-7}$ (shallow), $4.0\times 10^{-6}$ (shallow_attn), $4.2\times 10^{-6}$
  (medium_attn), $9.2\times 10^{-6}$ (medium)
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

The anchor bought pretraining fidelity of four orders of magnitude in the curve metric
(HISTORY 2026-08-31).

---

## 5. The comparison figure sets

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

`anchored_vs_unanchored_fx_fc.png` (in both v6g1 directories) overlays the anchored G1
corrections on the unanchored v4 record. Its provenance chain to the published v4 merged sets
(`figures_dfs6311_v4_merged_val_best{,_gga}`) was verified by execution: symlink resolution
into `run_20260810T202813Z`, byte-equal evaluation files, weights older than their evaluations
in all 54 specs, single-generation job records (HISTORY 2026-08-31). The long-form data behind
every curve statement below is `trained_fx_fc_curves.csv` / `pretrain_fx_fc_curves.csv` in the
respective directories, with the evaluation channel recorded per row (`eval_channel`,
validation-best with a labelled final-checkpoint fallback).

---

## 6. Measured pros and cons of the anchor

### 6.1 What the anchor buys

1. **Exact parent start.** First-step losses at the identity floor (Section 4.5); pretrained
   curves $10^{-7}$--$10^{-5}$ from the parent against 0.04--0.5 for the legacy protocol
   (Section 4.6); the certificate passes at initialization by construction, and a 2500-step
   pretrain cannot lose it (all twelve pulled anchored certificates PASS, Section 4.4).
2. **Certificate-gated fidelity.** The 13--56 kcal/mol silent handoff class of Section 2.2 is
   structurally excluded: an architecture that does not reproduce its parent cannot reach the
   train stage (measured FAIL-then-PASS pairs on the same architectures, Section 4.4).
3. **A convergent exchange family.** Where anchored and unanchored campaigns can be compared
   at the same capacity, the *trained* exchange corrections agree in form: a bond-region dip
   near $s = 0.7$--$1.1$ ($-0.013$ to $-0.044$ over the four cells) and a positive bump
   peaking near $s = 2.4$--$2.7$,
   of height $+0.079$ (shallow, ss=5, its largest completed cell), $+0.091$ (unanchored
   deep_3x16, ss=18), $+0.118$ (medium_attn, ss=18), $+0.162$ (medium, ss=18); the recorded
   band is $+0.07$ to $+0.16$ near $s = 3$ (HISTORY 2026-08-31; recomputed from
   `trained_fx_fc_curves.csv` of `v4gga_val_best` and `v6g1_size_val_best`; the unanchored
   attention twin peaks higher, $+0.21$). Training discovers the same exchange physics from
   either start -- the optimized networks move off the parent by up to 0.16 in $F_x$ (HISTORY
   2026-08-31).

### 6.2 The cost: pre-image sensitivity suppression

The anchored correction enters through the bounded map at the parent's pre-image, so its
parameter sensitivity carries the factor

$$L'(z_{\mathrm{parent}}) = F_{\mathrm{parent}}\left(1 - \frac{F_{\mathrm{parent}}}{\Lambda}\right),$$

which vanishes as the parent approaches either bound of $(0, \Lambda)$. Measured on the
exchange ceiling ($\Lambda = 1.804$, PBE parent): $L' = 0.446$ at $s = 0$ falling to $0.0073$
by $s = 20$ (recomputed from `parents.pbe_fx` + `lob_preimage`; HISTORY 2026-08-31 records
0.45 and 0.007). The correlation floor is the mirror case: at $r_s = 2$, $\zeta = 0$,
$F_c^{\mathrm{PBE}}$ falls from 1.0 at $s = 0$ to $1.5\times 10^{-3}$ at $s = 6$, and
$L'(z_{\mathrm{parent}}) \approx F_c$ there ($0.50 \to 0.076 \to 0.0015$ at $s = 0, 2, 6$) --
the parameterization suppresses trainability exactly where PBE correlation vanishes, which is
the large-$s$ region.

The measured consequence, on ss=18 validation-best cells at $r_s = 2$
(`trained_fx_fc_curves.csv` of `v4gga_val_best` and `v6g1_size_val_best`; HISTORY
2026-08-31): the *unanchored* v4gga deep_3x16 correlation correction grows into large $s$ --
$+0.79$ at $s = 2$ to $+0.92$ at $s = 6$ -- built entirely by training from a pretrained
start that sits within about $\pm 0.014$ of the parent at $r_s = 2$
(`v4gga_val_best/pretrain_fx_fc_curves.csv`); the *anchored* medium correction collapses past
$s = 2$: $+0.29$ at $s = 2$ to $+0.010$ at $s = 6$ (medium_attn: $+0.21 \to +0.005$). Both
anchored cells still build sizable corrections where the pre-image leaves trainability
(peaks $+0.43$ and $+0.35$ near $s = 1.2$).

### 6.3 The BH76 barrier bias

The unanchored campaigns used precisely that large-$s$ correlation freedom to remove the
parent's systematic barrier bias. PBE's BH76 error is almost pure bias ($-6.6$ to $-7.5$
kcal/mol mean signed error on the two slices below). Measured mean signed BH76 errors on the
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
channels; its correlation still collapses at large $s$, Section 6.2) while paying for it in
scatter (BH76 MAE 10.47 against PBE's 7.73, Section 7.2). Whether anchored cells can
reproduce the unanchored barrier improvement is the campaign's live question (Section 8).

**Unanchored cons, for the same ledger:** no parent fidelity at handoff (the 13--56 kcal/mol
offsets of Section 2.2, invisible to the point-wise loss); every correction, physical or not,
must be created by training from a flat start, so the pretrained state carries none of the
parent's structure into the fine-tune beyond the $\leq 0.09$ curve proximity; and there is no
gate -- the v4 record carried its offsets into production for the whole life of both running arms.

---

## 7. Current numbers: completed v6 GGA cells against the corrected v4/v5 GGA record

### 7.1 What is being compared

* **v6 G1** (`dfs6311_grid3_v6g1_size`, run `run_20260827T163330Z`): the anchored size
  ladder -- shallow, shallow_attn (2x8) and medium, medium_attn (3x16, the production
  width). Three registry fields separate `medium` from `deep_3x16`
  (`descriptor_log_transform`, `zero_init_final_layer`, `dm_entropy_intensive`), and under
  the v6 model block the two named flags are inert -- `parent_anchor` forces the
  zero-initialized final layer in `networks.create_network_pair`, and
  `descriptor_coordinates: dfs` takes the branch of both `_core` paths that never applies
  the log transform -- while the third touches only DM-statistics descriptors, which
  neither architecture carries; the operative cross-campaign difference between the
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
  contribute no GGA cells (their GGA rows are the v4 rows by design, Section 3).

The strict slices are cell-matched but not identical across campaigns: each cell is scored
against its own run's held-out slice with the cell's own PBE anchor (over the 54 v4 cells:
BH76 43--50, W4-11 97--113, combined 145--163 reactions; over the 25 v6 cells:
61 / 111--120 / 172--181; per-spec `test_set.csv` `n_reactions` and `per_reaction.json`). Beats verdicts below are therefore within-cell (NN MAE < the same slice's PBE
MAE), never cross-campaign row-for-row.

### 7.2 v6 G1 at current coverage: 25 of 44 cells

From the 25 `eval_holdout_val_best/test_set.csv` files present (medium 10 of 11, medium_attn
10 of 11, shallow 5 of 11, shallow_attn 0; the medium ss=26 cell failed on the open
NaN-gradient defect, HISTORY 2026-08-31), all values kcal/mol, NN / cell-matched PBE:

| arch | ss=1 | 2 | 3 | 4 | 5 | 6 | 7 | 12 | 15 | 18 |
|---|---|---|---|---|---|---|---|---|---|---|
| **W4-11** medium | 6.48/13.57 | 11.38/13.47 | 10.86/13.47 | 11.38/13.55 | 9.19/13.47 | 10.32/13.58 | 8.50/13.48 | 12.25/13.41 | 9.54/13.36 | 7.84/13.08 |
| medium_attn | 5.99/13.57 | 11.03/13.47 | 9.04/13.47 | 11.56/13.55 | 9.54/13.47 | 9.50/13.58 | 7.04/13.48 | 9.13/13.41 | 8.16/13.80 | 6.94/13.53 |
| shallow | 10.56/13.99 | 10.12/13.89 | 8.70/13.89 | 9.03/13.97 | 8.62/13.89 | -- | -- | -- | -- | -- |
| **BH76** medium | 8.03/7.73 | 9.71/7.73 | 9.96/7.73 | 10.47/7.73 | 6.51/7.73 | 10.22/7.73 | 6.51/7.73 | 10.24/7.73 | 8.13/7.73 | 6.92/7.73 |
| medium_attn | 9.66/7.73 | 7.45/7.73 | 9.01/7.73 | 11.30/7.73 | 8.72/7.73 | 8.95/7.73 | 7.61/7.73 | 10.47/7.73 | 10.75/7.73 | 10.57/7.73 |
| shallow | 7.34/7.73 | 7.84/7.73 | 7.84/7.73 | 6.46/7.73 | 6.47/7.73 | -- | -- | -- | -- | -- |
| **combined** medium | 7.00/11.59 | 10.82/11.53 | 10.56/11.53 | 11.07/11.58 | 8.29/11.53 | 10.29/11.60 | 7.82/11.52 | 11.55/11.43 | 9.05/11.38 | 7.51/11.18 |
| medium_attn | 7.23/11.59 | 9.83/11.53 | 9.03/11.53 | 11.47/11.58 | 9.27/11.53 | 9.32/11.60 | 7.23/11.52 | 9.60/11.43 | 9.07/11.67 | 8.23/11.47 |
| shallow | 9.47/11.87 | 9.36/11.81 | 8.41/11.81 | 8.16/11.86 | 7.89/11.81 | -- | -- | -- | -- | -- |

Headline, at 25-cell coverage:

* **W4-11: beaten in 25 of 25 cells** -- with the c2 reaction excluded, NN 5.91--12.18
  against cell-matched PBE 13.17--13.66; as scored, c2 included, NN 5.99--12.25 against
  PBE 13.08--13.99; every verdict is identical either way (the c2 note below). (The first
  published set, 18 cells (medium 10, medium_attn 8), read 18 of 18 at 5.99--12.25 against
  13.1--13.6; HISTORY 2026-08-31.)
* **Combined pool: 24 of 25** -- the one miss is medium ss=12 (11.55 vs 11.43). (18-cell
  state: 17 of 18, same single miss.)
* **BH76: 8 of 25**, best 6.46 (shallow, ss=4) against 7.73; the other beats are medium
  ss=5/7/18 (6.51, 6.51, 6.92), medium_attn ss=2/7 (7.45, 7.61), shallow ss=1/5 (7.34, 6.47).
  (18-cell state: 5 of 18 with best 6.51.)

The c2 note: the per-eval PBE SCF of the multireference C2 dimer splits into two solutions
across the 25 specs ($-75.816741$ Ha, internally stable, in 18 cells; $-75.736895$ Ha in 7:
medium_attn ss=15/18 and the five shallow cells; the 25 `per_molecule.json` files), and the
$>10^{-4}$-Ha cross-spec spread makes the reference guard exclude c2 from the pooled figure
baselines (the guard and the multi-solution class were established on a different, 24 mHa
incident; HISTORY Phase 38). HISTORY 2026-08-31 traces the flip to the reference-SCF
rescue, rules the higher branch internally unstable ($+50.10$ kcal/mol) and queues the
seven evaluations for re-evaluation; on the pulled evaluations the c2 atomization's PBE
error reads $-53.6499$ kcal/mol on the 7 affected cells against $-3.5457$ on the 18
(`per_reaction.json`, the cells named through `manifest.json`). The reaction
(`w411_c2_atomization`) sits in the strict W4-11 slice of every completed cell
(`per_reaction.json`, empty `in_sample_overlap`), so the affected cells' table rows above
carry it as scored; excluding it, the cell-matched W4-11 PBE anchors span 13.17--13.66 and
every beat verdict of this section is unchanged (25/25 W4-11, 24/25 combined, 8/25 BH76).

### 7.3 The corrected v4gga validation-best record, per architecture (54 cells)

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

### 7.4 Reading

At matched capacity (medium pair vs deep_3x16 pair), the anchored cells reproduce the
unanchored campaigns' W4-11 gains -- and do so uniformly (25 of 25 against 28 of 54; the
unanchored total is diluted by the retired-fidelity descriptor arms, whose pretraining never
delivered their parent, Section 2.2). On BH76 the picture inverts: the unanchored deep_3x16
beats PBE in all 11 cells (best 4.14) where the anchored cells beat it in 8 of 25 (best
6.46), and the signed decomposition (Section 6.3) locates the difference in the parent's
barrier bias, which the unanchored large-$s$ correlation freedom removed and the anchored
parameterization largely retains. Whether that is the anchor's price or a removable training
artifact is exactly what the queued anchored deep_3x16 group measures (Section 8). The
comparison carries the coverage caveats stated: 25 of 44 G1 cells, slices cell-matched within
each campaign but not identical across campaigns.

---

## 8. Open questions the next results answer

1. **The controlled anchored-vs-unanchored test.** The G2a core trio (deep_3x16,
   deep_attn_3x16, deep_cusp_3x16 under the full v6 protocol with `parent_anchor: true`) is
   queued behind the draining G1 group (HISTORY 2026-08-30, the trio split; HISTORY
   2026-08-31). It meets the strongest unanchored record at the registry identity itself: the
   G1 medium pair already realizes the deep_3x16 capacity with both differing flags inert
   (Section 7.1), so the trio's deep_3x16 and deep_attn cells test the anchored result's
   reproducibility at the registry names themselves, and its deep_cusp cells extend the
   comparison to a descriptor form G1 does not carry; the two group files share an
   identical subset axis including ss=26 (`dfs_step7.dfs6311_grid3_v6g1_size.yaml` line
   129, `...v6g2a_families_core.yaml` line 122), so G1's ss=26 cells are pending rather
   than absent (item 3). The BH76 signed bias (Section 6.3) is the
   discriminating observable; the G1 spread ($-7.75$ at
   medium/ss=12 against $-0.81$ at medium_attn/ss=12) says the outcome is not foreclosed.
2. **The meta-GGA trained factors.** The five anchored meta-GGA family architectures hold
   PASS certificates at production identity and pretrained curves within
   $8.1\times 10^{-7}$--$1.3\times 10^{-5}$ of SCAN (Sections 4.4-4.6); their training cells
   are the pending half of `figures_dfs_step7_dfs6311_grid3_v6g2_families_mgga`. They answer
   whether the anchored fine-tune preserves SCAN-level held-out accuracy where the v5
   SCAN-seeded (but mis-footed, unanchored) cells only matched it at subset sizes 2--5
   (Section 3), and whether the correlation-collapse mechanism of Section 6.2 repeats against
   the SCAN parent, whose correlation approaches the same zero bound at large $s$.
3. **G1 completion.** 19 G1 cells remain (all 11 shallow_attn cells, the six remaining
   shallow sizes, and the ss=26 cells of medium and medium_attn -- the medium ss=26 cell
   needs the open NaN-gradient defect closed, HISTORY 2026-08-31), after
   which the medium-pair-vs-anchored-deep-pair comparison of item 1 -- registry names
   differing in three fields, all inert under the v6 model block (Section 7.1) -- can be
   read at full coverage, alongside the re-evaluation of the seven c2-affected cells
   and the pooled-baseline re-inclusion of c2 (Section 7.2).
