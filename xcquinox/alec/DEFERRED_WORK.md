# xcquinox/alec -- deferred work

`HISTORY.md` records what was done and why. This file records what was
consciously NOT done: work items evaluated and set aside, each with the
measurement or constraint that forced the deferral, what is already known (so
paid-for numbers are not re-derived), and the trigger that should reopen it.
Entries are removed when executed, with the closing change referenced in
`HISTORY.md`.

---

## 1. Global bond-order / delocalization descriptors (dropped 2026-08-06)

**What:** two candidate `dm_statistics`-style global descriptors, a mean Mayer
bond order per atom (`B_AB = sum_{mu in A, nu in B} (PS)_{mu nu} (PS)_{nu mu}`;
Mayer, *Chem. Phys. Lett.* **97**, 270 (1983)) and an interatomic
delocalization fraction (inter-atomic share of `||PS||_F^2`), both per spin.

**Why dropped:** measured non-size-consistency -- both are per-system averages,
so a distant fragment moves the value at every grid point inside the fragment
of interest: CO+H2 shifts the Mayer mean by a factor 0.6928, H2O+H2 by 0.9099,
a three-fragment composite by 1.2242 (delocalization: 1.1719 / 1.2072 /
1.0656). Both are also identically zero on every single-atom system, with zero
gradient and zero assembled V_xc -- and atoms are the atomization-energy
references, so the pair encodes a molecule-versus-atom label. This is the same
global-scalar leak class as the removed `dm_entropy`. Additionally the
atom-slice implementation cannot run inside the training kernel
(`ConcretizationTypeError` under `filter_jit` for array-typed slices; a static
tuple keys the compile by value) and the `(n_atom, 2)` slice array does not
collapse to a common shape under `pad_group_to_common_shape`, reintroducing the
per-molecule compile split of the 2026-07-17 mapping-exhaustion incident.

**Already known (do not re-derive):** a same-atom boolean MASK formulation --
`(n_ao, n_ao)` mask padded as a zero AO block, `n_atoms` traced like `nocc`,
`mayer = 0.5 * sum(M * M.T * (1 - mask)) / n_at`, intra-block sums via
`sum(M * M * mask)` -- reproduces the slice form to 2.78e-17, traces under
`filter_jit`, and is exactly padding-neutral. Any future LOCAL reformulation of
a bonding/covalency ingredient should start from that construction. The plain
per-spin quantity implemented in the screen is exactly HALF Mayer's published
spin-resolved bond order (verified against the known values H2 = 1.0, N2 = 3.0,
CO = 2.594, H2O = 1.936); any revival must match the cited equation or state
the deviation.

**Trigger:** a design for a per-grid-point (not per-system) bonding descriptor;
size consistency is the acceptance gate, tested with non-identical fragment
pairs -- identical-fragment pairs cannot fail it.

## 2. Angular channels for `rung35_multishell`

**What:** the shipped descriptor is the RADIAL (`l = 0`) part of the
NeuralXC-style localized density-matrix projection: `fakemol_for_charges`
builds s-type projectors only, and with one m per shell the rotational
invariant `sqrt(sum_m c_{nlm}^2)` collapses to the occupancy itself.

**Why deferred:** `l > 0` channels need solid-harmonic fakemols (or an explicit
auxiliary-basis overlap build), a new integral path rather than a parameter
change.

**Already known:** the per-shell-norm contraction and its differentiability
argument carry over unchanged; the descriptor must not be described as "the
DFS descriptor" until the angular channels exist.

**Trigger:** evidence that the radial profile alone under-resolves bonding
anisotropy (e.g. no improvement on systems where the single-width descriptor
also fails), or a need for parity with the reference implementation.

## 3. A local replacement for the removed `dm_entropy`

**What:** a correlation-indicator feature to replace `dm_entropy`, which was
removed because its gradient is ill-defined at every converged density.

**Why deferred:** an impossibility result rules out the obvious candidates
wholesale. For a single determinant the eigenvalues of `DS` are exactly
{2,...,2,0,...,0}, so ANY function of the spectrum alone depends only on
`N_occ` and is constant on the idempotent manifold -- measured:
`Tr[(DS)^n]/N = 2^(n-1)` for H2, N2 and CO alike, and the participation ratio
returns `N_occ`. A useful replacement must probe the eigenvectors (spatial and
bonding structure), not the spectrum.

**Already known:** the candidate screen and criteria are in
`notebooks/analysis/DM_DESCRIPTOR_SPEC.md`; the local projector family
(`rung35`, `rung35_multishell`) already supplies leak-free local
density-matrix information, so the bar for a new GLOBAL scalar is high.
The retained `x_6` (idempotency defect) shares the dead-on-manifold property:
on the single-determinant KS densities this pipeline evaluates it is zero in
value AND gradient (measured 2.7e-31 in precompute, 2.9e-13 mid-SCF, gradient
6.9e-17 at convergence -- twelve orders below the sibling `x_7`). The
distinction from the removed entropy is that this zero is exact and smooth (a
squared norm at its own minimum), not an autodiff artifact; but any
replacement design should treat the whole `dm_statistics` block, since `x_7`
is currently its only live channel.

**Trigger:** a candidate that varies between N2 and CO at fixed electron count,
is size-intensive under the non-identical-fragment test, and has an exact
gradient at an idempotent density matrix.

## 4. Pointwise correlation-energy-density loss (Local Energy Loss)

**What:** replace or augment the total-energy training loss with a pointwise
real-space correlation-energy-density loss, targets from kappa-regularized MP2
within the Moller-Plesset adiabatic connection (Polak, Zhao and Vuckovic,
*Nat. Commun.* **16**, 11306 (2025), DOI 10.1038/s41467-025-66450-z).

**Why deferred:** a loss-architecture change orthogonal to the descriptor and
potential work; needs its own design record (target generation in PySCF, grid
alignment, weighting against the existing channels).

**Already known:** the claimed benefits -- each molecule's single scalar
becomes thousands of grid targets, spatial error cancellation is penalized,
unphysical dissociation curves from global losses avoided -- are consonant with
this project's per-grid-point descriptor design; the citation above is
verified against the publisher record (the article NUMBER is 11306; an earlier
note carried the DOI suffix in its place).

**Trigger:** the next loss-design iteration, or evidence that total-energy
supervision is the binding constraint on functional quality.

## 5. PYSCFAD `get_veff` override for DM-dependent descriptors

**What:** the pyscfad backend RAISES for density-matrix-dependent descriptors
outside FROZEN policy, because its libxc-style per-point `eval_xc` callback
returns only `(exc, vrho, vsigma)` and cannot carry the global
`sum_g w_g (de/df)_g . df_g/dP` term. The fix is a `get_veff` override:
accumulate the term across grid blocks in the AO basis, symmetrize, add to the
numint result.

**Why deferred:** the design is specified in the raising function's docstring
(`solver_pyscfad._reject_dm_dependent_descriptors`); the production sweep uses
the MANUAL backend, so nothing running needs it.

**Trigger:** a pyscfad-backend requirement for any `_mgga` / `_rung35` / `_dm`
architecture under REASSEMBLE.

## 6. Implicit-function-theorem SCF gradients

**What:** replace reverse-mode differentiation of the unrolled `lax.scan` SCF
with implicit (fixed-point) differentiation.

**Why deferred:** it changes the cost profile of every heavy item at once and
interacts with the deliberately-unconverged `full_3` production solver (an IFT
gradient assumes a fixed point; `full_3` stops after 3 cycles by design).

**Already known:** `full_25` reverse-mode grad-of-grad exhausts a 30 GB
workstation without `scf_grad_checkpoint` and costs 1.6-5.6 GB with it, so
checkpointing already contains the memory problem the IFT route would solve;
the unrolled 25-cycle diffuse-basis H atom does not converge and its
unconverged energy wanders under multithreaded BLAS (single-thread runs are
bit-identical), so any IFT comparison must pin one BLAS thread.

**Trigger:** solver depths beyond `full_25`, or a convergence-to-fixed-point
training scheme where the IFT assumptions actually hold.

## 7. Exact-constraint inventory (beyond those already enforced)

**What:** additional exact constraints as architectural maps or losses.

**Already enforced** (constrained output construction): the Lieb-Oxford bound
`0 <= F_x <= 1.804`; the UEG limit `F_x = F_c = 1` at `s = 0` via
`I_a(0) = 0`; uniform density scaling and the spin-scaling relation via the
`x~2`-only dependence of `F_x`.

**Open:** the Gell-Mann-Brueckner high-density correlation limit (distinct
from the `s -> 0` UEG limit); the `-1/r` asymptotics of `v_x` and the
`-alpha/2r^4` tail of `v_c` (now cheap to check, since the corrected potential
is the true derivative and the autodiff `v_xc` is trustworthy); fractional
charge/spin piecewise linearity (needs fractional-electron reference systems);
the Sham-Schlueter consistency condition (needs a differentiable self-energy;
out of reach of the current stack).

**Trigger:** per constraint, the next functional-design iteration; the
potential-asymptotics loss is the cheapest first candidate.

## 8. FRG-DFT-to-ML data handoff

**What:** using functional-renormalization-group DFT correlation-energy tables
(Yokota and Naito line) as fit-free training anchors.

**Why deferred:** FRG-DFT for real electrons is LDA/LSDA-level only, so a
naive data handoff reproduces LDA; the survey-derived numbers (a 65,536-point
correlation-energy table; 2.0%/9.9%/20% deviations from Monte Carlo across
densities) are UNVERIFIED against the primary sources.

**Trigger:** verification of the primary sources plus a use case where a
QMC-independent anchor materially changes a conclusion.

## 9. Unverified survey citations

**What:** a batch of literature claims collected 2026-08-06 from a research
survey, of which eight were checked: the physics and arithmetic held in every
checked case, but three carried material citation errors (an attribution to a
paper that does not contain the attributed claims; a DOI suffix reported as an
article number; a one-sided account of the DM21 fractional-electron
controversy that omitted the authors' published Response,
DOI 10.1126/science.abq4282).

**Already known:** the verified subset is recorded in
`reports_local/latex/references.bib` with per-entry notes; everything else from
that survey is unverified until checked against the publisher record.

**Trigger:** any use of a survey-derived claim in a manuscript or docstring.

## 10. `loss_vxc` remains on frozen precompute features

**What:** `losses._vxc_term` and `oneshot._uks_spin_resolved_vxc` still
assemble V_xc from features evaluated at `dm_pbe` (frozen), so the training
channel compares a frozen-feature potential against the OEP reference while
the SCF now builds the live-feature potential.

**Why deferred:** each site is the exact derivative of its own frozen-feature
energy, so nothing is inconsistent internally; changing the definition makes
every historical `loss_vxc` value incomparable.

**Trigger:** the retraining pass that the corrected potential already requires
for the affected architectures -- change both together and re-baseline.

**Decision 2026-08-10:** the v4 re-sweep (`dfs6311_grid3_v4`) fired this
trigger and the change was deliberately NOT taken: redefining `loss_vxc` in
the same run would confound the cross-run A/B, and the channel's DEFINITION
is identical on both sides, which is what the A/B needs. NOTE (corrected
same day): the original justification quoted a 0.1-3.6% composite share;
that figure was the median over ALL updates, dominated by the ~25% of
updates where the channel is identically zero. Measured over vxc-ACTIVE
updates on the v3 `aux_log.pkl` (47 specs, 94,825 live rows), the per-spec
median share reaches 48.15% on the `subset_size = 1` cells (spec_0000
48.15%, spec_0022 46.74%, spec_0011 46.52%) and 19/47 specs exceed 3.6% --
`loss_vxc` is a first-order term at the ablation curves' leftmost points.
The deferral therefore stands on definition-consistency, not on smallness,
and the frozen definition's influence must be kept in mind when reading the
small-subset cells. The trigger MOVES to the re-baselining re-sweep.

## 11. `uks_zeta` gradient freeze on the deep negative-density tail

**What:** `oneshot.uks_zeta` retains a `stop_gradient` where
`rho_tot <= 1e-12` (the Phase-17 guard against `0*inf` on grid-tail noise that
drives the total density non-positive). A `stop_gradient` inside an energy
ingredient misreports the derivative wherever it is live -- the same class as
the two defects already removed -- but this one is live only below the
`1e-10` network tail mask, and toggling it changed the measured
energy/potential residual by exactly nothing on the systems tested. The
finite-difference suite cannot see it by construction (guard-straddling grid
points are excluded).

**Trigger:** any diffuse-basis system where grid-tail quadrature noise drives
`rho_tot` non-positive AT points the network tail mask keeps live
(`rho > 1e-10`); or the fallback design of a smooth energy-level damping for
the zeta boundary, at which point this guard should be replaced in the same
change and `split_exc_energy_uks` / `compute_vc_polarized_per_spin` must move
together.

## 12. Single-copy work must never live only in /tmp

**What:** a working rule, recorded after the (s, alpha) pretrain-mesh
implementation was parked as /tmp backups during a commit separation and a
session restart wiped the directory. The work was rebuilt by replaying the
recorded edit operations from the session transcript
(`scratch/mesh_recovery/`, see its README), at the cost of a recovery pass
that a durable parking spot would have made unnecessary.

**Rule:** in-progress work separated out of a commit is parked under
`scratch/` (untracked, real disk) or committed work-in-progress to a side
branch -- never only under /tmp.

## 13. SCF convergence freeze: a theta-dependent branch with no DFS counterpart

**What:** `solver_manual.py` freezes the SCF state (`jnp.where(already, ...)`)
once the per-cycle |dE| < `conv_tol` = 1e-6 Ha. dpyscf runs all 25 cycles
unconditionally, so the branch is a deviation, and it makes the loss a
piecewise function of theta near the threshold.

**Why deferred:** removing or retuning it mid-arc would change the solver
behavior every completed run trained with, breaking cross-run comparability;
the sweep pins `device: cpu`, where the loss and gradient are reproducible.

**Already known:** fires on 22.9% of v3 molecule-instances (155/678), 9.4%
before the last cycle (13 at cycle 1, 51 at cycle 2); most frequent: Li, H,
Li+, HLi. A local probe sat 14-17% inside the tolerance. The density
channel's value is backend-sensitive at the ~1e-4 relative level (CPU vs
CUDA), while the energy channels reproduce bit-identically.

**Trigger:** the next re-baselining sweep -- either raise `conv_tol` well
above the per-cycle step or run all cycles unconditionally (DFS-faithful),
and add a padding-neutrality fixture that sits NEAR the tolerance (the
committed fixtures all sit far from it).

## 14. H/Li atomic-density supervision (SI Sec. II)

**What:** the Letter's SI states H and Li atomic electron densities were
calculated and included; dpyscf routes them through density+energy losses.
This pipeline's free-atom anchor groups carry no density/V_xc references --
their density channels iterate over nothing (verified in production
`aux_log.pkl`: `anchor:H`/`anchor:Li` have `loss_rho != 0` in 0 of 200 rows).

**Why deferred:** adding references changes the training composition for
every run; it belongs to a re-baselining sweep, not a mid-arc patch.

**Trigger:** the same re-baselining sweep as item 13; the reference
generator already produces atomic densities for the benchmark side.

## 15. Full-composition finite-difference coverage for rung-3.5 and open-shell groups

**What:** the 2026-08-10 workflow review FD-checked the complete production
loss composition (tail + DF + lock + polarized + per_molecule) against
central differences for `deep_mgga_3x16` (2.0e-8..3.0e-8 over three random
directions; de-fused equal to fused at machine precision). The same
composition on `deep_rung35_3x16` and on an open-shell UKS group exceeded
the local resource budget (the OH group peaked at 12.5 GB and timed out).

**Already known:** both families pass the committed
`test_training_gradient_consistency.py` FD checks under a simplified solver
config, and the rung-3.5 production training STEP descends with finite
nonzero gradients (executed in the v4 batch review).

**Trigger:** run the review's `scratch/review_train/m1a_full_loss_fd.py`
harness for those two cases on a cluster node (RAM is the binding
constraint, not correctness).

## 16. GGA-arm c2 PBE eval reference regeneration (drift detected 2026-08-12)

**What:** the v4 GGA arm (`dfs6311_grid3_v4gga`, `run_20260810T202813Z`)
carries a drifted c2 PBE reference energy in every completed spec's held-out
eval: E_pbe(c2) = -75.757329256 Ha, vs -75.816711949 Ha in both the meta-GGA
arm and the trusted post-repair v3 run (+37.26 kcal/mol; within-arm spread
<= 3.6e-12 Ha). The c2 non-convergence pathology of the plain PBE kernel is
the established mechanism (`hpcjobs/dfs6311_c2_ref_probe.py`); the drifted
value entered whatever per-run PBE cache the arm's eval consumed. Affected:
the arm's per_reaction/test_set PBE comparison columns on c2-containing
reactions and the arm-local PBE baselines (NN energies untouched).

**Already known:** no current cell's beats-PBE verdict flips (nearest cell
0.58 kcal/mol from the affected band). The figure layer excludes c2 from the
cross-arm reference baselines with a printed warning
(`_first_pbe_energies` consistency check, 1e-4 Ha tolerance), and both the
PBE and SCAN reference legs skip its reactions symmetrically, so merged
figures are deterministic in the meantime.

**Trigger:** when train array 2116743 drains, locate the arm's PBE eval
cache for c2, delete it TOGETHER WITH its `_intermediates/` entries
(deletion alone re-drifts; see the probe script header), re-run the affected
specs' held-out evals, and confirm the figure-layer disagreement warning no
longer fires on a fresh pull.

## 17. Cluster-side strict-holdout repair deployment + re-eval (found 2026-08-13)

**RESOLVED 2026-08-15.** Deployment (user rsync 2026-08-13) + refinalize job
2120119 (COMPLETED 0:0, 21 + 57 channels rewritten with one-time backups,
verified on the artifacts) + the closing parity probe on the repulled v4gga
run: **54 parity, 0 stale-rule, 0 value-mismatch, max |delta| = 0.0 on every
row** across 27 specs x 2 channels -- including the 8 specs evaluated
natively under the deployed rule after the refinalize, proving cluster
writes and the local reconstruction are one path. Nothing remains open.

**What:** the cluster-side strict held-out filter is name-based and blind to
the training-vs-pool species naming split (training uses ASE Hill formulas
from the DFS pool builder -- `CHN`, `H3N`, `HO`, `CH2` -- while the
benchmark pool names the same molecules `hcn`, `nh3`, `oh`, `ch2-trip`), so
trained molecules' reactions and density species remained inside the
"held-out" per_reaction/per_molecule rows of every affected spec (e.g.
`w411_hcn_atomization` in every cell whose subset trains `CHN`). Two
further set defects: the four BH76 barriers duplicated in the pool under
permuted-reactant names sit one copy in the validation slice and one in the
test slice (validation-best selection saw those four test barriers), and
the `in_training_subset` per-molecule flag is false for the same
naming-mismatch species.

**Already done (local, 2026-08-13; superseded the species-level rule the
same day):** held-out exclusion is now by VERBATIM supervised reaction
(canonical identities with geometric isomer classes,
`xcquinox/alec/species_matching.py`; `eval_holdout.trained_reaction_exclusion`
+ identity drops in `_finalize_holdout_outputs`; recorded-validation
exclusion by identity). The figure layer reconstructs each spec's full test
slice from its per-species energies over the canonical pool with the same
exclusions, so locally rendered figures already carry the verbatim-rule
slices; density means keep the species-level exclusion (trained densities
are verbatim training targets) and the c2 consistency guard.

**Deployment done (user rsync, 2026-08-13):** eval_holdout.py,
species_matching.py, train.py, cluster/_eval_one_spec.py,
cluster/_holdout_parallel.py are live on the cluster; in-flight and pending
array tasks eval under the verbatim rule from that moment.

**Remaining trigger -- refinalize the stale-rule specs:** specs whose eval
completed BEFORE the deployment carry species-strict artifacts. MARKED SET
(parity probe, 2026-08-13): v4gga spec_0000..0017 and v4mgga spec_0000..0006
(all 50 pulled spec-channels report stale-rule, 0 value mismatches) plus the
cluster-only v4gga spec_0018 (completed between the last pull and the
deployment). Remedy: `sbatch hpcjobs/refinalize_verbatim_holdout.sbatch`
(no SCF; rewrites per_reaction.json/test_set.csv from the existing
per-species energies with one-time *.pre_verbatim.* backups; idempotent, so
running it over whole run dirs is safe and its report is the ground-truth
stale list). Safe alongside the running arrays (touches only completed
specs' dirs) or at drain. Afterwards: re-pull and run
`python notebooks/analysis/verify_holdout_parity.py <pulled run dirs>` --
the closing state is all `parity`. Local figures already use the
reconstructed verbatim slices either way.

## 18. B-regime seed-blend campaign (Letter-faithful training seed)

WHAT: a controlled comparison arm training gga/rung35/mgga archs from the
Letter's randomized seed blend, dm_in = (1-m)*dm_seed + m*dm_minao with
m ~ U(0.5, 1.0) redrawn per molecule visit, per the campaign decision
record (phase 2 of the 2026-08-14 A/B). NOT a knob on the current
protocol: at max_cycles=3 the tail-weighted loss puts 80% of its mass on
cycle 3 (production weight power 2.0) with zero on cycle 1, so a
majority-minao blend would score near-minao transients.
KNOWN: the package couples blend + max_cycles ~15-25 + scf_grad_checkpoint
+ dm_minao as a second mol_data key (padding/cache-key entries) + a
per-visit PRNG step index threaded train loop -> loss -> solver + the
recorded 15-25-cycle mgga SCF oscillation gate + ~8x SCF walltime on
cells already measured at 42 h + a seed_blend SolverConfig field with
describe()/round-trip coverage.
TRIGGER: after the v5 pure-seed arms land and the mgga oscillation gate
resolves (arm-1 deep_mgga ss12/15/18 verdicts).

## 19. SCAN-seeded rung-3.5 control arm

WHAT: a small follow-up submission training the three pure rung-3.5 archs
(deep_rung35 / deep_rung35_attn / deep_rung35ms, 3x16) with seed_xc
forced to scan, against their PBE-seeded v4 rows -- the direct seed A/B
for the rung whose baseline assignment is ambiguous (GGA functional form,
beyond-GGA information content; the figures grade them against SCAN).
KNOWN: one YAML (copy the v4gga arm, restrict the arch axis, seed_xc:
scan, fresh output root); the rungs.seed_xc_for_arch "beyond_gga_scan"
policy already exists for the auto route.
Also KNOWN: the merged figure view will REFUSE this arm twice over -- its
resolved seed diverges from the phase-1 rung-baseline policy the view
validates, and its (arch, subset_size) cells duplicate the v4gga rows.
That is correct: the merged view is single-protocol by construction; the
control comparison belongs in per-arm / protocol-comparison figures (or,
if a merged multi-protocol view is ever wanted, a per-arm-base policy map
passing "beyond_gga_scan" into seed_xc_for_arch).
TRIGGER: user's call after the v5 mgga arms report.

## 20. DIIS mixer for a converged cold-start eval

WHAT: the eval_holdout_coldstart channel is a trajectory diagnostic
(linear/decaying-linear mixing, fixed 25 cycles); the Letter's benchmark
protocol it approximates is a minao cold start run to CONVERGENCE under
PySCF DIIS. A DIIS mixer in the manual solver (v2+ scope per the solver
design record) would close that gap and make the channel a
converged-eval replica.
KNOWN: CRITERION_REGISTRY has only the energy-delta criterion; a DM-RMS
criterion and a real early-exit (lax.while_loop or forward-only break)
would land with it.
TRIGGER: if the cold-start diagnostic proves informative enough to
promote into a headline comparison.

## 21. dm_target collocation experiment for the vxc channel

WHAT: the L5 loss_vxc term evaluates V_xc^NN at the PBE density for all
arms (kept deliberately in the 2026-08-14 protocol change; the Wu-Yang
vxc_ref retains a PBE pairing in its weakly constrained directions). The
stronger uniform alternative: collocate at dm_target (the CCSD density,
already in mol_data) for ALL arms -- zero bias for an exact functional,
one footing everywhere. Changes an active loss channel for every arch,
so it is its own controlled experiment, not a rider.
KNOWN: RKS branch reads stored rho_grid/sigma_grid (PBE) and would need
the dm_target contraction; UKS already contracts from a dm. The
descriptor-feature and feature-response parts of the potential must move
with the density or the evaluation point is inconsistent (the round-1
plan-review finding).
TRIGGER: after the v5 arms establish the seeding effect in isolation.

## 22. Consolidate the redundant per-molecule metric SCFs

WHAT: under FULL mode the eval metric stack re-runs the identical SCF up
to 4x per molecule (total_energy, atomization_energy, density_rmse,
scf_convergence each call run_scf); the coldstart channel multiplies the
cost. One shared SCFResult per (molecule, solver_config) handed to all
metrics would cut eval wall-time ~4x.
KNOWN: compile is cached, execution repeats; the metric ABC's
solver_config seam is where a memo would live.
TRIGGER: next eval-side refactor window, or if v5 eval walltime becomes
the binding constraint.

## 23. Diagnostic scripts must mirror the spec seed when replaying v5 specs

WHAT: hpcjobs/dfs6311_nan_verify.py, hpcjobs/dfs6311_nan_isolate.py, and
notebooks/analysis/multimode_constraint_eval.py construct fresh
SolverConfig objects (seed_source defaults 'pbe') and precompute without
seed threading; replaying a v5 SCAN-seeded spec through them silently
evaluates a different protocol. local_reeval already mirrors the spec
(fixed 2026-08-14).
KNOWN: each needs the same two-line threading the production call sites
got (seed fields from the spec's solver_config into precompute).
TRIGGER: first v5-spec replay through any of these tools.
