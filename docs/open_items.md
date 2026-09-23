# Open items

This file records what was
consciously NOT done: work items evaluated and set aside, each with the
measurement or constraint that forced the deferral, what is already known (so
paid-for numbers are not re-derived), and the trigger that should reopen it.
Entries are removed when executed.

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
the descriptor specification (the screen itself is in no tracked file); the local projector family
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
`docs/references.bib` with per-entry notes; everything else from
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
implementation was parked as /tmp backups during a commit separation and the
machine cleared /tmp. The work was rebuilt from the recorded edit operations
(`scratch/mesh_recovery/`, see its README), at the cost of a recovery pass
that a durable parking spot would have made unnecessary.

**Rule:** in-progress work separated out of a commit is parked under
`scratch/` (untracked, real disk) or committed work-in-progress to a side
branch -- never only under /tmp.

## 13. SCF convergence freeze: a theta-dependent branch with no DFS counterpart

**2026-09-08:** switchable. `SolverConfig.freeze_on_convergence` (named solver key
`freeze_on_convergence`, default true) keeps the branch for every existing solver block;
the dpyscf-parity arm (`dfs_step7.dfs6311_grid3_v7g1_dfsparity.yaml`, `full_25` with
`freeze_on_convergence: false`) runs every cycle as dpyscf's loop does, the converged
flag still latching for the record. Closed for that arm; open as stated below for the
running arms until their re-baselining.

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
`xcquinox/pipeline/species_matching.py`; `eval_holdout.trained_reaction_exclusion`
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
`python tools/analysis/verify_holdout_parity.py <pulled run dirs>` --
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
tools/analysis/multimode_constraint_eval.py construct fresh
SolverConfig objects (seed_source defaults 'pbe') and precompute without
seed threading; replaying a v5 SCAN-seeded spec through them silently
evaluates a different protocol. local_reeval already mirrors the spec
(fixed 2026-08-14).
KNOWN: each needs the same two-line threading the production call sites
got (seed fields from the spec's solver_config into precompute).
TRIGGER: first v5-spec replay through any of these tools.

## 24. Per-architecture tier-1 worker cap for the held-out eval (found 2026-08-20)

WHAT: the held-out eval ladder starts every architecture at 40 workers x 1 thread per node;
the DM-projector architectures (rung35ms: `(3, N_grid, nao)` projector stack, rung35_attn)
lose 153 / 68 tier-1 shards to worker death and, before the Phase 39 repair, 147 / 70 species
to swallowed per-species exceptions, against 0-5 for every other architecture.
WHY DEFERRED: the per-worker RSS of those architectures at the production identity has not
been measured on the cluster (the task logs carry no per-shard memory line; training in the
same array task peaked at 98.8 GB RSS on a 40-core --mem=0 node); the cap should be set from
`sacct` MaxRSS / `seff` on array 2116743 tasks 50 and 55-61 against tasks 0-10, not guessed.
KNOWN: `cluster.eval_workers` (null today) already caps the top of the ladder; the Phase 39
re-queue means a too-high cap now costs time (tier-2 retries), not records.
TRIGGER: the sacct numbers for 2116743, or the next submission of a DM-projector arm.

## 25. Meta-GGA open-shell exchange: the alpha feature is not spin-scaled in the production path (found 2026-08-20)

WHAT: `oneshot.split_exc_energy_uks` (:493-495) and `_uks_spin_resolved_vxc` (:764) implement
the exact exchange spin-scaling relation (Oliver and Perdew, Phys. Rev. A 20, 397 (1979)) by
doubling rho and quadrupling sigma per spin channel while passing the descriptor feature vector
unchanged into both channel evaluations; the P2-02 note documents that as an approximation for
context features without a doubled-spin transform. The meta-GGA alpha feature HAS an exact
transform, alpha_sigma = alpha(2 rho_sigma, 4 sigma_sigma, 2 tau_sigma) (libxc's spin-polarized
SCAN exchange equals two unpolarized evaluations at those ingredients to <1e-12 Ha on the O-atom
grid), so for the meta-GGA architectures the production energy and potential of every
open-shell species are evaluated at a feature the functional never sees for polarized
densities. Measured 2026-08-20 on frozen PBE densities: the pretrained deep_mgga_3x16 over-binds
H2O / N2 / CH4 by 30.5 / 55.9 / 20.8 kcal/mol relative to SCAN; transforming alpha per spin
channel and nothing else leaves -7.6 / -7.9 / -7.6 (atomic exchange offsets Li / C / N / O from
+7.6 / +19.2 / +32.9 / +28.2 mHa to +2.7 / -1.8 / -5.4 / -8.3 mHa). Secondary: the open-shell
pretraining rows store spin-resolved SCAN targets against total-density inputs
(`pretrain_data_gen._atom_columns`); undoing the substitution there changes the offsets by
+1.0 / +7.3 / -2.0 kcal/mol only. The GGA architectures without descriptors are exempt
(deep_3x16 vs PBE: -2.5 / -4.2 / -2.4 kcal/mol); whether the rung-3.5 per-spin occupancy
features need an analogous transform is an open question (the occupancy is linear in the
spin density matrix).
WHY DEFERRED: the correction needs the per-spin kinetic-energy density on the grid in both the
energy and the Fock-build paths (and in the descriptor reassembly of the manual solver), plus
the pretraining rows posed per spin channel; it changes the trained and evaluated energies of
every open-shell species for the meta-GGA architectures and therefore invalidates the v5
meta-GGA cells (trained and evaluated under the frozen-alpha scaling) and the v5 pretrain
checkpoints. A campaign decision with the v5 results in hand, not a patch under a running array.
KNOWN: acceptance oracle = libxc SCAN spin=1 reproduced by the spin-scaled evaluation to
numerical precision; acceptance test = the Section 5 table of
the v5 meta-GGA note of 2026-08 collapsing to the corrected values (-7.6 / -7.9 /
-7.6 kcal/mol or better); probe + independent re-derivation run in about a minute locally
(scratch/probe_pretrain_vs_scan.py, scratch/mgga_spin_scaling_check/indep.py).
TRIGGER: the decision to re-run the meta-GGA arms, or any new meta-GGA training at the
production identity.
AFFECTED SPECS (to be retrained after the correction; every architecture carrying the
meta-GGA alpha descriptor, i.e. every cell trained and evaluated under the frozen-alpha
spin scaling):
- dfs6311_grid3_v5 / run_20260815T034818Z: spec_0000-0010 deep_mgga_3x16 (0000-0006
  evaluated 2026-08-20, 0007-0010 in flight), spec_0011-0021 deep_mgga_attn_3x16 and
  spec_0022-0032 deep_rung35_mgga_3x16 (queued/in flight on array 2120759) -- all 33;
- dfs6311_grid3_v5mgga2 / run_20260815T034822Z: spec_0000-0010 deep_cusp_mgga_3x16 and
  spec_0011-0021 deep_rung35ms_mgga_3x16 (array 2120764) -- all 22;
- their five pretrain checkpoints (pretrain/<arch>/ in both runs) and the pretrain data
  set (open-shell rows re-posed); the SCAN seed cache is unaffected (PySCF SCAN SCFs).
Not affected by THIS defect: the GGA-rung arms (dfs6311_grid3_v4gga, array 2116743).
deep_3x16 and deep_attn_3x16 carry no descriptor feature (exact spin scaling; 2.3-4.2
kcal/mol pretrain offset vs PBE). deep_cusp / deep_rung35 / deep_rung35_attn / deep_rung35ms:
the cusp feature is pure geometry (exact under spin scaling) and the rung-3.5 occupancy has no
in-domain doubled-spin evaluation (values up to 1.9 against its [0, 1] bound; energies 40-80x
worse), so their frozen-feature convention stands; their 13-56 kcal/mol pretrain offsets are a
pretraining-protocol gap, recorded as #26.

## 26. Descriptor-carrying architectures do not start fine-tuning at their parent functional (found 2026-08-20)

WHAT: on frozen PBE densities at the production identity, the v4gga pretrained networks
deviate from PBE in atomization energy (H2O / N2 / CH4, kcal/mol; every value re-derived
independently to 0.001): deep_3x16 -2.5 / -4.2 / -2.4 and deep_attn_3x16 -2.3 / -4.1 / -3.1
(descriptor-free, acceptable); deep_cusp_3x16 -13.2 / -4.2 / -25.7; deep_rung35_3x16
-13.5 / -3.5 / -29.1; deep_rung35_attn_3x16 -29.5 / -20.4 / -56.1; deep_rung35ms_3x16
-22.0 / -30.9 / -42.8. Two causes, neither a spin-scaling defect: (a) the H-atom
pretraining error of every cusp-carrying network (+13.7 mHa against +0.8 for the
descriptor-free control), multiplied by the hydrogen count; (b) molecular extrapolation of
density-matrix features that the atoms-plus-mesh pretraining set never constrained (the
molecular offsets of rung35_attn and rung35ms flip sign, -6.5 / -0.3 / -12.8 and
-7.3 / -12.8 / -14.1 mHa). The pretraining loss is blind to both: deep_rung35_attn_3x16 has
the lowest exchange residual of the six (2.1e-6) and the largest offset. The meta-GGA
pretrains carry the same gap on top of the alpha defect (#25): -7.6 / -7.9 / -7.6 kcal/mol
remain after the alpha correction.
WHY DEFERRED: a remedy changes the pretraining protocol (molecular rows with density-matrix
features in the pretraining set, an explicit H-atom / per-atom energy term or reweighting,
or a parent-reproduction constraint during pretraining) and therefore the starting point of
every descriptor-carrying cell; it is part of the campaign decision in #25 and must be taken
together with the preflight gate that measures it.
KNOWN: acceptance test = the dAE table above within a stated tolerance (order 1 kcal/mol
on AEs, the descriptor-free level) for every architecture; the probe runs in under a minute
per architecture (scratch/probe_pretrain_gga_rungs.py, scratch/mgga_spin_scaling_check/indep2.py).
TRIGGER: the next pretraining of any descriptor-carrying architecture; the preflight gate.

## 28. Production-basis energy/potential check for the three-block UKS Fock (2026-08-23)

**WHAT:** the spin-scaling spec's oracle O2 asks for a central-difference check of the assembled
UKS Fock matrices against the three-block energy on H, Li, N and O with every descriptor active,
"extended from Li/def2-svp to the production basis". All four species are covered at def2-svp;
since #27 closed the probe there is the rotation path of `_uks_fd_path` -- a rank-preserving
orbital rotation of every populated channel -- run over the 124 harness cells of 31
architectures x {H, Li, N, O}, plus Li probed at its own fixed point along a constrained and an
unconstrained direction. The
repetition at the production identity (6-311++G(3df,2pd), grid level 3) EXISTS as a slow-marked
test, `test_solv01_split_xc.test_fd_consistency_uks_polarized_production_identity` (`-m slow`,
the O atom, every architecture); what is deferred is RUNNING it. The figures on its record
(1.5e-9 to 7.5e-8, the mask keeping 99.6% of the grid) were taken with the linear displacement
and the indicator's clip-state straddle rows, before the smooth positive part of #27 and the
move onto the rotation path, so the case ships with numbers that no longer describe the probe it
now carries.

**WHY DEFERRED:** cost, and the wrong machine for it. Each direction is two full three-block
energy evaluations -- three descriptor blocks, three feature-response contractions -- plus one
SCF cycle to capture the Fock pair on its way to the eigensolver. At def2-svp / grid level 1 with
nao 14 the N check measures 4-8 s per architecture and the Li fixed-point check 39 s on four CPU
threads; the production basis raises nao by roughly a factor of four and the grid by an order,
which puts the same battery in the tens of minutes to hours and into the class of run this
repository submits rather than runs on the workstation.

**KNOWN (do not re-derive):** the current def2-svp record is the rotation-path run at grid level
2 -- 1.8e-10 to 6.6e-8 relative over the 124 harness cells against `_TOL_UKS = 5e-7`, the
guard-straddle mask removing zero points in every cell (re-measured 2026-08-24 on
`deep_mgga_3x16`, `deep_3x16` and `deep_rung35_mgga_3x16` x {H, Li, N, O}: 1.9e-10 to 3.3e-8,
`dropped_mass` exactly 0.0 and `kept_points` exactly 1.0 in all twelve) -- and, on the solver's
own Fock pair at grid level 1, 2.9e-11 (H), 2.4e-10 (Li), 9.8e-11 (N) and 5.5e-11 (O) at the
1e-5 step. The defect signal these bounds must stay below is the superseded two-block potential
against the same three-block energy: 1.4e-4, 3.6e-5 and 7.1e-5 on O. Kept as the record of the
earlier probe (linear displacement, clip-state straddle rows, grid level 1) and NOT as figures
to compare a current run against: 3.8e-13 to 6.6e-11 (N, element directions, 1e-5 step, three
architectures), 4.1e-11 to 4.6e-10 (O, random symmetric direction, 1e-6 step), 7.8e-10 (H) and
8.7e-8 (Li) along the one-orbital rotation at the 1e-5 step, and 5.0e-10 for the alpha direction
at Li's fixed point. The clip-state straddle that drove the direction choice then -- a random
step changing the clip state of 431 (h = 1e-6) to 710 (h = 1e-5) of N's 4098 resolved beta
points, reading as a 6.0e-5 relative "failure" that was the clip's one-sided derivative and not
a potential defect -- cannot recur: the indicator has no lower clip to straddle. Two things must
still be derived at the larger basis rather than carried over: the FD step (1e-5 to 1e-6 is
right at def2-svp; a wider basis is worse conditioned) and the choice of direction (entry 30).

**TRIGGER:** run the shipped slow-marked case on the cluster, on its own (`-m slow`) or as a cell
of the workflow matrix, before the next production UKS campaign on the meta-GGA rungs, and
whenever the three-block potential or the feature-response contraction is changed. The production
residuals must be re-measured along the rotation path and the case's docstring restated from that
run. Whether the N atom rejoins the production case is entry 30's question and is not settled
here: the rotation path removes the clip-boundary pathology that excluded N, but the indicator's
tail response still rules out any direction that leaves the cone of positive semidefinite
matrices there, and no rotation-path measurement of N at the production identity has been taken.

## 30. The iso-orbital indicator's tail response makes the meta-GGA Fock hyper-sensitive off the SCF manifold (found 2026-08-24)

**WHAT:** `alpha = (tau - tau_W)/tau_unif` divides by `n^{5/3}`, so its derivative with respect
to the density matrix, `d alpha / dP ~ |grad chi_i . grad chi_j| / n^{5/3}`, becomes large in
the density tail wherever a diffuse basis function is large relative to the occupied orbital.
The response is PEAKED ON A SHELL and is not a monotone power law in the density, so the
shorthand `d alpha / dP ~ n^{-5/3}` states a scaling the data does not show. Measured by a JVP
of the block's raw indicator along a unit diagonal element of the most diffuse basis function
(Li beta, `deep_mgga_3x16`, def2-svp, grid 1), `max |d alpha_raw / dP|` per density band:
8.00e4 above 2 rho_beta = 1e-4 (1710 points), 2.22e6 on 1e-6 to 1e-4 (776), 1.26e8 on 1e-8 to
1e-6 (1140), 4.07e11 on 1e-9 to 1e-8 (728) and 1.15e1 below 1e-9 (510) -- ten orders BELOW the
peak in the deepest tail, a log-log fit against 2 rho giving slope -0.43, not -5/3. The
amplification therefore lives on the 2 rho_beta ~ 1e-9 to 1e-8 shell and nowhere else. On that
same record the raw indicator moves under a 1e-14 relative change of the density matrix by
4e-4 on the outermost radial shell (rho_beta = 1.0e-9, 898 points), by 1.25e-6 at 2 rho_sigma =
1e-8 to 1e-7, 1.05e-8 at 1e-6 to 1e-4 and 9.4e-10 above 1e-4; each of those 898 points
contributes 2.9e-3 Ha to one element of the feature-response Fock term (0.57 Ha over the shell)
while the energy they carry is 1e-8 Ha at most (their weight times the indicator's ceiling).
Consequences, all measured: (i) that channel's Fock matrix moves by 0.37 Ha in its
virtual-virtual block under the 1e-14 probe with the smooth positive part of entry 27 in place
(0.93 with the hard clip; 0.2-0.5 at every width from 1e-9 to 1e-5 -- no width in indicator
units can be large against 4e-4 without an alpha floor of the same order); (ii) a
finite-difference probe of the Fock against the energy along a linear symmetric displacement
of a RANK-ONE channel is not a derivative estimate at any usable step: a 1e-6 step moves the
raw indicator by 1e3 to 1e5 on those points, beyond the ceiling, and the residual
reads 5.2e-2 (Li) and 6.0e-6 (N) flat from the 1e-5 to the 1e-7 step, on the hard clip and on
the smoothing alike, while a density cut at rho_sigma > 1e-8 on the probe only takes Li to
2.5e-3 because the response is still 1e6 to 1e8 a decade or two above that. Leaving the cone of
positive semidefinite matrices is NOT the operative condition, measured channel by channel on
Li at the reference density with one fixed random symmetric direction: displacing the ALPHA
channel alone takes it out of the cone (minimum eigenvalue of the displaced block -2.2e-7) and
still reproduces dE/dP to 1.24e-10, 5.37e-11 and 4.03e-10 at the 1e-5, 1e-6 and 1e-7 steps,
while displacing the BETA channel alone -- rank one, raw indicator identically zero, block
indicator saturating the ceiling along the probe (max alpha = 100.0 and max |d alpha| = 100.0
at every step) -- reads 7.40e-1, 7.29e-1 and 7.26e-1, flat in the step; both channels together
read 1.60e-1, 1.51e-1 and 1.48e-1. The operative condition is the rank-one boundary together
with the ceiling crossing, not cone departure; (iii) H is exempt
(its basis has no function more diffuse than its orbital; the same probe reads 3.8e-10 at the
1e-7 step), O is exempt (every block's raw indicator is >= 6.6e-4 on the resolved grid), and
N's beta channel (1s and 2s, one-orbital-like in its tail) is the multi-electron case.

**WHAT IT DOES NOT AFFECT:** the raw indicator is stationary along every rank-preserving
rotation of a one-orbital block, so the response annihilates that block's occupied orbital
exactly (measured: F_beta c moves by 2e-13 while the matrix moves by 0.37 Ha) and the fixed
point of the manual loop does not depend on the rounding (H and Li reproduce from seeds 1e-14
apart to 0.0 and 1.9e-14 Ha, the Li figure of the order of the seed separation itself). Along
the SCF's own manifold -- random orbital rotations of every populated channel, positive
semidefinite at every step -- the Fock pair reproduces the energy to 2.9e-11 (H), 2.4e-10 (Li),
9.8e-11 (N) and 5.5e-11 (O) of the probe's absolute-contribution scale at the 1e-5 step with no
mask and no gate, which is the tangent direction the Roothaan step needs. That restriction is a
probe-design constraint and not a hole in the potential: the loop assembles its features and
its Fock at the MIXER output, a convex combination of two rank-`nocc` projectors, so Li's beta
block is rank TWO where the energy is actually evaluated (raw indicator median 0.87 at a mixing
of 0.05 and 0.36 at 0.20, against 3e-16 at the aufbau matrix), and an UNRESTRICTED linear
symmetric probe of both channels at that rank-raised base point reproduces dE/dP to 3.27e-10,
5.41e-11 and 2.33e-9 (mixing 0.05) and 1.58e-10, 4.29e-10 and 2.29e-11 (mixing 0.20) at the
1e-5, 1e-6 and 1e-7 steps -- 5.4e-11 at best, the floor the rotation path itself reaches. The
rotation path is thus the geometry that keeps a probe off the rank-one boundary, not the only
density the loop visits. The energy is unaffected (the tail's integrand mass is nil).

**WHY DEFERRED:** the only remedy is a change of the descriptor's tail behaviour in the ENERGY --
a smooth damping of the indicator's dependence on the density matrix below a density scale
(the network already masks 2 rho_sigma <= 1e-10 with a hard step), or a bounded reformulation
of the shell-peaked amplification above -- which redefines the descriptor for every meta-GGA
checkpoint and pretraining file and needs an explicit decision; a rescaling of the width in
indicator units cannot do it (the width would have to reach the alpha scale of the tail
response, 1e-2 or more, against a physical floor that must stay below 1e-5), and a
stop-gradient is excluded by the 2026-08-06 rule that the derivative must be the derivative of
the energy.

**KNOWN:** the probe geometries that are valid -- rank-preserving rotations (`_rotation_path` in
`test_spin_scaling_solver_manual.py`, `_uks_fd_path` in `test_solv01_split_xc.py`), and a linear
symmetric displacement at a rank-raised base point -- and the one that is not (a linear
symmetric displacement of a RANK-ONE channel; rank deficiency on its own is harmless, as the
alpha channel above shows), with the numbers above; the per-point decomposition script pattern
(contribution of each grid point to one Fock
element via a JVP of the block closure along a unit matrix direction) that located the shell.
Also known, and the second route by which this can reach training: the indeterminacy is already
in `d(Fock)/d(theta)` itself, through `feature_response_vxc` and BEFORE any eigensolver. The
parameter gradient of the assembled beta Fock, `d/d theta <F_beta, M>` for a fixed random `M`
(Li, `deep_mgga_3x16`, def2-svp, grid 1), moves by 6.4e-3 relative under a 1e-14
rank-preserving change of the density matrix, and by 8.3e-1 under an unrestricted one, where an
architecture carrying no indicator column (`deep_rung35_3x16`) moves by 2.6e-14. The smooth
positive part of entry 27 IMPROVES that quantity by 264x over the hard clip (1.70 -> 6.4e-3)
and it falls as 1/w exactly (6.4e-4 at w = 1e-4, 6.5e-2 at 1e-6), so nothing here counts
against the smoothing.
What is NOT known: whether that indeterminacy reaches a trained model -- through
`feature_response_vxc`'s own parameter gradient, or through the differentiable SCF's
eigensolver derivative (the training gradient of a FULL-mode meta-GGA fit
on a one-orbital channel visits densities within 1e-14 of rank one only in its last cycles) --
to be measured on Li with two seeds 1e-14 apart before the meta-GGA rungs are trained in FULL
mode; and the size of the response at the production identity, where `d alpha / d sigma`
reaches 2.2e31 (entry 27's measurement) and the outermost shells are diffuse.

**FIRED 2026-09-04 (in part):** the v7 meta-GGA campaign's FULL-mode training found the
defect this entry did not anticipate -- the DFS coordinate ln((alpha + 1)/2) undefined on the
non-positive intermediate density the step-0 mixing coefficient 1.3 produces (alpha_raw < -1 in
the doubled-channel tail) -- and closed it by flooring the indicator at alpha = -1/2 at the column
level (networks._domain_indicator, _dfs_indicator_coordinate; test_metagga_indicator_domain): below every physical row, residues included, so the smooth
positive part's differentiability through alpha = 0 that this entry measured is preserved. The determinism question
stated here is untouched and still open.

**TRIGGER:** a FULL-mode meta-GGA training campaign on the open-shell atoms (measure the
gradient's determinism first), or any finite-difference check of the meta-GGA potential that
must probe directions off the SCF manifold.


## 31. The train array's thread pools keep the allocation on an unmeasured premise (2026-08-25)

**WHAT:** every process that drives PySCF in its own address space now caps its OpenMP and
OpenBLAS pools at `parallel.PYSCF_POOL_THREADS_MAX` (8) from the allocation -- the stage
templates for datagen, pretrain, preflight, benchmark refs and the evaluation array, the
standalone job scripts, and the workflow matrix's stage environment -- because the two pools,
sized to the allocation, spin-wait between the small dense operations of an SCF loop
(workflow-matrix job 2134488: about ten minutes per molecule at 40 threads on a 40-core node
against 8 s at four). The train array's template is the one exemption: it still exports the
allocation to both pools, on the premise that its per-step work is XLA's and its PySCF work is
confined to inputs the preflight precomputed.

**WHY deferred:** the premise is a statement about where a train step's time goes, not about
which libraries load -- `solver_pyscfad.py` and `df_jk.py` import PySCF and sit on the
training path (integrals built once per system) -- and settling it needs one timing of a
production train step at 8 against 40 threads per pool on a node, which is cluster work.

**KNOWN:** the inline train-and-evaluate template already runs its pools at `SLURM_CPUS_ON_NODE / 12`
(3 on a 40-core node, 8 on a 96-core node) with XLA's own pool taking the rest, and the v4 and
v5 campaigns trained under that setting; the array template's allocation-sized pools have
never been compared against it on the same cell.

**TRIGGER:** before the v6 groups are submitted on the array template, or at the first
group's first train cell: time one cell's first hundred steps at the two settings from the
same checkpoint; if the capped setting is not slower, cap the train array too and retire the
exemption in `test_render_thread_caps_present_every_template`.

## 33. Deploy the T1-key remedy to the cluster and re-evaluate the two refused cells (deferred 2026-09-15)

**WHAT:** the remedy (the reference loader accepts `t1_diagnostic`; a
reference the loader cannot read stops a held-out pass; an evaluation that precomputed no
species is refused before any table is written) is committed locally and not yet on the
cluster. Two evaluated cells carry NaN tables from the
refusal (size run `run_20260902T145245Z` spec 42, deep_attn_2x8 at 18; 25-cycle run
`run_20260908T153856Z` spec 2, deep_3x16 [25 cycles] at 15); their channel directories are
set aside locally under `checkpoints/spec_XXXX/refused_t1key_2026-09-15/` so the figures
hold the evaluated cells only.

**WHY:** the user chose on 2026-09-15 to verify and publish the pulled results first and to
re-run the two cells in a later job.

**KNOWN:** every task whose held-out evaluation starts before the deploy lands writes the same
NaN tables (the four tasks running at the pull and everything queued behind them); the sweep
line below lists every such task from the cluster's logs, and each one listed needs one more
repair submission (the same command with its run directory and index). The repair job's
default wall (6 h at 20 workers) is short for the 25-cycle cell (8.3 h scaled from its
measured passes); the submissions carry 12 h and the campaign's own 40 workers.

**TRIGGER:** the next cluster session. The commands, from the repository root:

```bash
rsync -av xcquinox/pipeline/data.py xcquinox/pipeline/eval_holdout.py xcquinox/pipeline/benchmark_refs.py "$swpath":/gpfs/projects/FernandezGroup/Alec/xcquinox/xcquinox/pipeline/
rsync -av xcquinox/pipeline/cluster/_eval_one_spec.py xcquinox/pipeline/cluster/_holdout_parallel.py "$swpath":/gpfs/projects/FernandezGroup/Alec/xcquinox/xcquinox/pipeline/cluster/
ssh "$swpath" bash -l <<'EOS'
grep -c t1_diagnostic /gpfs/projects/FernandezGroup/Alec/xcquinox/xcquinox/pipeline/data.py
grep -c 'FAILED (external_data' /gpfs/scratch/awills/xcquinox_runs/dfs_step7/dfs6311_grid3_v7g*/runs/run_*/logs/train_eval_*.out | grep -v ':0$'
sacct -j 2166308,2166309 -X -o JobID%14,State,ExitCode,Elapsed,Start,End
squeue -u awills
EOS
ssh "$swpath" bash -l <<'EOS'
cd /gpfs/scratch/awills || exit 1
grep -q t1_diagnostic /gpfs/projects/FernandezGroup/Alec/xcquinox/xcquinox/pipeline/data.py || { echo "deploy not on the cluster"; exit 1; }
sbatch --time=12:00:00 --export=ALL,REEVAL_TASKSET_CPUS=40,REEVAL_RUN_DIR=/gpfs/scratch/awills/xcquinox_runs/dfs_step7/dfs6311_grid3_v7g1_size/runs/run_20260902T145245Z,REEVAL_SPEC_IDX=42 /gpfs/projects/FernandezGroup/Alec/xcquinox/hpcjobs/reeval_holdout_spec.sbatch
sbatch --time=12:00:00 --export=ALL,REEVAL_TASKSET_CPUS=40,REEVAL_RUN_DIR=/gpfs/scratch/awills/xcquinox_runs/dfs_step7/dfs6311_grid3_v7g1_c25/runs/run_20260908T153856Z,REEVAL_SPEC_IDX=2 /gpfs/projects/FernandezGroup/Alec/xcquinox/hpcjobs/reeval_holdout_spec.sbatch
squeue -u awills
EOS
```

After the repair jobs end: pull the two runs, remove the local `refused_t1key_2026-09-15/`
directories, re-merge and re-render.


---

## 34. The notebooks pretrain without writing the fidelity certificate the training stage requires

**WHAT:** the training gate refuses a pretrain checkpoint that carries no `fidelity_certificate.json`.
The anchor-transfer notebook writes the certificate for every pretrained pair through
`fidelity.fidelity_certificate` at the harness layout the gate judges, with `enforce: false` and the
reason on the record, since its short pretraining cannot meet the tolerance; the other notebooks
pretraining cells do not, so their training cells cannot run end to end.

**Remedy:** the remaining builders run the certificate on the checkpoints they write, as the cluster
harness does (`cluster/fidelity.py`), and the end-to-end notebook tests then cover every notebook.

**Why deferred:** a change to the notebooks training flow with its own tests and review.

---


---

## 35. The training-side references carry no PBE twin, so their in-sample PBE density baseline is absent

**WHAT:** the model-free PBE-against-CCSD density baseline reads the reference calculation's own
PBE density (`rho_pbe_grid`, written beside `rho_ref_grid` by `xcquinox.pipeline.benchmark_refs`)
and never the locally recomputed density of the run's own SCF, a second PBE twin from another
calculation (the two measured 0.39 percent apart on c2). The OEP references the training side
reads (`inputs.external_refs_dir`, written by `external_refs.run_oep_cascade`) carry no such twin,
so every in-sample record built from them reports `density_rmse_pbe`, `density_l1_pbe` and
`density_eps_l1_pbe` as absent; the held-out channels, whose references come from the benchmark
writer, keep the baseline. Every reader of the three columns tolerates the absence (the in-sample
mean becomes nan, the tables and figures skip the column).

**Remedy:** the OEP writer stores `rho_pbe_grid` from the reference SCF payload on the reference
grid, as the benchmark writer does, and a backfill on the `t1_backfill` pattern adds the key to
the existing references from their cached SCF payloads without a new SCF.

**Why deferred:** a change to the reference file format on the training side, with its own tests
and review; the v8 campaigns report the held-out baseline.
