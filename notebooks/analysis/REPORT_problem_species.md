# Problem species and numerical artifacts: a compendium

This document collects the numerically difficult species and the physical (as opposed to
purely infrastructural) artifacts encountered while training and evaluating neural
exchange-correlation functionals in this repository, for use in the paper's methods and
appendix material. Each section states the symptom with its magnitude, the root cause
with the governing relations, the detection signal that now fires on a recurrence, the
remedy the pipeline applies, and the status. Provenance is given inline in a compact
form: `HISTORY <date>` denotes the dated entry in `xcquinox/alec/HISTORY.md`; all other
paths are repository-relative and name the log, docstring, or analysis document in which
the quoted number was measured or is pinned. Production identity, unless stated
otherwise, means 6-311++G(3df,2pd), grid level 3, density fitting on, orientation-lock
strength $3\times10^{-5}$.

---

## 1. C2: a two-configuration SCF landscape

**Symptom.** The C2 dimer is the only species of the 214-species benchmark pool whose
plain PBE kernel fails to converge (HISTORY 2026-07-26). At the held-out evaluation
identity (RKS/PBE, 6-311++G(3df,2pd), grid 3, lock $3\times10^{-5}$) DIIS oscillates for
all 100 cycles between two SCF configurations with a trajectory energy spread of
$1.204\times10^{-1}$ Ha, 73 of 100 cycles in the lower basin, ending at an arbitrary
phase of the oscillation (final cycle $E=-75.7175899504$)
(scratch/v6_diag/repro_c2_pbe_branch.log).

**Root cause.** Two distinct SCF solutions exist, consistent with the multireference
character of the C2 ground state: a second-order (SOSCF) solve converges either to
$E=-75.8167407121$ Ha (internally stable; externally unstable toward a UKS symmetry
breaking, as is the higher solution) or to $E=-75.7368945310$ Ha (internally unstable),
split by $7.984\times10^{-2}$ Ha $=50.10$ kcal/mol
(scratch/v6_diag/repro_c2_pbe_branch.log).

![C2 at the held-out evaluation identity (RKS/PBE, 6-311++G(3df,2pd), grid 3): the
100-cycle DIIS trajectory oscillating between the two converged SCF solutions, with the
lowest-energy (cycle 12) and lowest-gradient (cycle 25) trajectory points
marked.](figures_report_pretraining/c2_diis_trajectory.png)

The left axis carries the total energy of each DIIS cycle and the logarithmic right axis
the orbital gradient norm; the dashed horizontals are the two converged second-order
solutions, -75.8167407121 Ha (internally stable) and -75.7368945310 Ha (internally
unstable), and the dotted horizontal is their midpoint at -75.776818 Ha, taken as the
dividing line for counting rather than as a physical separatrix. The trajectory crosses
that line repeatedly and never settles: 73 of the 100 cycles lie below it, the energies
span $1.2043\times10^{-1}$ Ha, and the phase the cycle cap happens to fall on -- here the
upper branch -- carries no information about which solution the species has. Both markers
lie on the lower branch: the lowest-energy point at cycle 12, -75.8167361296 Ha, sitting
$4.58\times10^{-6}$ Ha above this draw's converged solution -- the same separation the
acceptance check under Detection reports as $-4.09\times10^{-6}$ Ha on a different draw,
that check signing the excess as converged minus trajectory minimum, so the two figures
differ in sign convention and not in direction -- and the lowest-gradient point at cycle
25, where $|g|$ falls to $3.177\times10^{-3}$ a.u.; these are the two seeds the remedy
below distinguishes, the first for the orbital-pair rerun and the second for the
second-order stage. What the figure establishes is the landscape the next paragraph's
aufbau flip rides on: a rescue must take its seed from a trajectory that visits both
basins, and the quality of that seed does not by itself decide which one is reached.

**The dm0-ingestion aufbau flip.** Which solution a density-seeded second-order solve
reaches is not controlled by the quality of the seed: PySCF's SOSCF ingests `dm0` by
diagonalizing $F[\mathrm{dm0}]$ and re-occupying by aufbau, and the ground solution of
C2 is non-aufbau in its own Fock. Seeding SOSCF with the trajectory's lowest-energy
density ($E=-75.8167361$) converges $0.0798$ Ha uphill onto the internally unstable
higher solution, and even the converged ground solution's own exact density flips to the
higher branch ($E=-75.7368945380$); seeding from the same point's orbital pair instead
converges to the ground solution in 2 macro-iterations (xcquinox/alec/data.py,
`_converge_reference_scf` docstring; scratch/v6_diag/repro_c2_pbe_mo_start.log). Which
branch a density-seeded start lands on is draw-dependent -- the unconverged oscillation
is chaotic; local draws stayed on the ground branch while 7 cluster evaluations stamped
the higher one (data.py docstring).

**Detection.** Two independent gates. (i) At reference generation, a converged solution
-- a second-stage rescue and a DIIS-converged endpoint alike -- is accepted only within
a branch tolerance of $10^{-4}$ Ha of the DIIS trajectory's own minimum energy. The
measured same-basin excesses at equilibrium identities are $-2.97\times10^{-7}$ Ha
(Li/SCAN), $-4.09\times10^{-6}$ Ha (C2/PBE), $+8.26\times10^{-7}$ Ha (O/PBE rescue) and
$+8.38\times10^{-6}$ Ha (S/SCAN, a DIIS-converged endpoint over its own trajectory
minimum), so the tolerance sits roughly $12\times$ above the largest of them and nearly
three decades below the measured inter-branch gap of $7.984\times10^{-2}$ Ha; a sweep of
every species in the current data-generation pools (28 atom runs plus the rescue
identities) found none above $+8.4\times10^{-6}$ Ha (data.py,
`_REFERENCE_SCF_BRANCH_TOL`). (ii) At the figure layer, any species whose model-free
$E_\mathrm{PBE}$ spreads across specs by more than $10^{-4}$ Ha (SCF reconvergence noise
is $\sim2.5\times10^{-6}$ Ha) is excluded from every pooled reference baseline with a
printed warning naming the species and values (HISTORY 2026-08-12).

**Remedy.** The reference SCF runs DIIS first, then a second-order stage seeded from the
lowest-gradient point of the DIIS trajectory; a converged solution exceeding the branch
tolerance -- from either stage -- is rerun from the minimum-energy point's recorded
orbital pair (immune to the aufbau re-occupation), the lower converged solution is kept,
and a persisting excess is refused with the refusal naming the rerun's own outcome. The
residual exposure of the acceptance check is a false refusal, never a wrong recorded
value: at a near-degenerate geometry (C2 stretched to $r=1.60$ A) the rescue's converged
endpoint is itself reproducible only to $1.31\times10^{-4}$ Ha across draws, so such a
species can refuse on an unlucky draw (data.py, `_REFERENCE_SCF_BRANCH_TOL`). The
production path converges equilibrium C2 in 105 total cycles
(`diis+newton`) to $E=-75.8167407121$ with a stamped orbital gradient of
$1.8$--$2.2\times10^{-6}$ (scratch/v6_diag/verify_c2_branch_fix.log).

**Earlier episodes.**

- *Reference-grid drift* (HISTORY 2026-07-26). PySCF density-prunes the DFT grid once,
  at the first SCF cycle, on the density held at that moment
  (`small_rho_cutoff` $=10^{-7}$). The tiered convergence escalation rebuilt a fresh
  mean field seeded `dm0=make_rdm1()`, whose grid was pruned on the non-converged
  50-cycle DIIS density; the stored reference grid therefore depended on the
  generation-time trajectory. C2's reference carried 26840 points (26840/26848/26856
  across DF/lock configurations, basin-sensitive between the two solutions,
  $E=-75.8168$ against $-75.7158$ for the escalated draw) against the 26568 points every
  consumer computes from the guess-pruned grid; the eval-side loader's shape gate then
  dropped the species from every completed eval. Controls: N2 (26616) and F2 (26568)
  converge on the plain kernel and match exactly. Remedy: the stored grid is pinned to
  the FIRST kernel attempt regardless of which tier converges, and the density-fitting
  identity is stamped into the cache (`density_fit_used`); deleting the final `.npz`
  alone re-drifts, so the intermediates must be deleted with it. The regenerated
  reference reads 26568 points (probe: hpcjobs/dfs6311_c2_ref_probe.py).
- *Cross-arm reference drift* (HISTORY 2026-08-12, 2026-08-17). The same class
  resurfaced across two concurrently evaluated sweep arms: $E_\mathrm{PBE}(\mathrm{C2})
  =-75.816711949$ Ha in one arm against $-75.757329256$ Ha in the other -- a 37.26
  kcal/mol disagreement against $\le3.6\times10^{-12}$ Ha spread within each arm -- and,
  within one arm, a single degraded evaluation whose 2-thread retry re-converged the C2
  PBE SCF to a third value ($-75.781328$, 24 mHa off). The cross-spec-spread guard
  excludes the species from pooled baselines; no beats-baseline verdict moved.
- *Benchmark gate refusal* (HISTORY 2026-08-18). During a local recomputation of
  transiently lost per-species energies, every patch was gated on the locally converged
  PBE energy matching the recorded one within $10^{-7}$ Ha (two decades above the
  $\sim10^{-9}$ conv-tol reproducibility bound, five below the multi-solution signal).
  The gate fired exactly once: C2 converged $5.93\times10^{-2}$ Ha away from the
  recorded solution and the patch was refused rather than computed from the wrong
  density.

**Status.** Closed at the reference-generation layer (branch-stable rescue, pinned
grid). The only C2 benchmark reaction (`w411_c2_atomization`) sits in the STRICT
held-out slice of every completed v6 cell (its `in_sample_overlap` empty in all 116
`per_reaction.json` records -- the 29 completed specs, four evaluation channels each);
the seven branch-affected evaluations were repaired in place by a gated patch tool rather
than re-evaluated wholesale, in two rounds (the standard channels, then the best-loss
channel after its checkpoint fetch), after which the audit reads 116 channels, 0 wrong,
the cross-spec reference guard is silent and C2 rejoins the pooled PBE baselines on all
four evaluation channels; the two cells completed after the repaired reference layer was
deployed to the cluster arrived with the correct branch natively, with no patching
(HISTORY 2026-09-01); the beats-baseline verdicts are unchanged with and without the
species. In the earlier
(v4-era) split the reaction sat in the validation slice. The drifted GGA-arm evaluation
reference is scheduled for regeneration at array drain
(xcquinox/alec/DEFERRED_WORK.md item 16).

---

## 2. Li: reference-SCF fragility of a diffuse-basis alkali

**Symptom.** The v6 meta-GGA data-generation stage refused on the Li atom: the SCAN
reference SCF at 6-311++G(3df,2pd)/grid 3 under the $3\times10^{-5}$ lock reaches the
solution basin early (cycle 5: $E=-7.478697644723$, $|g|=7.5\times10^{-4}$), the DIIS
extrapolation then throws the density into an unphysical state at cycle 7
($E\approx-5.04$) and wanders chaotically to the 100-cycle cap (endpoint $-4.07$ in one
draw, $-3.85$ in another; the pre-explosion basin numbers reproduce across draws)
(HISTORY 2026-08-30; scratch/v6_diag/repro_li_scan.log).

**Root cause.** A DIIS explosion after the basin is reached leaves the trajectory's
final point as the worst possible start for a rescue: a second-order solve seeded from
the endpoint runs all 50 macro-iterations unconverged (stall gradients
$1.2\times10^{-3}$ to $4\times10^{-3}$ across draws). The failure requires SCAN together
with the diffuse `++` set -- PBE converges in 5 cycles at the same identity, SCAN at
def2-svp in 7, and Na (the protocol's other alkali) in 7 at baseline. Level shifts (0.25
/ 0.5), damping (0.5), a finer grid (level 5), and removing the lock were each measured
and none converges it (scratch/v6_diag/repro_li_scan.log,
scratch/v6_diag/repro_li_scan_fixes.log).

**Remedy and status.** The second-order stage now starts from the lowest-gradient
density the DIIS trajectory visited, recorded by a callback: from that point SOSCF
converges Li in 2 macro-iterations to $E=-7.4786979415$; the production path stamps
`diis+newton`, 102 total cycles, gradient $5.97\times10^{-6}$ against the
$\sqrt{10^{-9}}=3.16\times10^{-5}$ criterion (scratch/v6_diag/verify_li_fix.log). The
landed value was checked independently: SOSCF from the minao, atom, and huckel guesses
agrees to a $3.8\times10^{-11}$ Ha spread, with an aufbau occupation, $S^2=0.750004$,
internally stable; DIIS-converged records are bitwise unchanged, and of 48
production-setting rescues 45 are no-ops and the 3 that differ improve (HISTORY
2026-08-30). With the rescue in place the full polarized SCAN species set at the
production identity generated cleanly (datagen wall 6:00; HISTORY 2026-08-31).

**Earlier episode: poisoned-dm0 escalation** (HISTORY 2026-07-05, Phase 19). The same
atom failed differently in the SCAN baseline generator: plain DIIS diverged to a garbage
density ($E\sim-4.9$ against the true $-7.478$), and the tiered escalation then seeded
every tier with that garbage as `dm0`, so a dm0-seeded second-order solve inherited it
and stalled -- while the same solver from a fresh minao guess converged cleanly
($E=-7.4785$). The escalation gained fresh-guess tiers that ignore `dm0`; grid
refinement was shown irrelevant (grid 3/4 identical).

---

## 3. OH, CH, NO: orientation degeneracy of $^2\Pi$ radicals

**Symptom.** For an orbitally degenerate $^2\Pi$ ground state, the singly occupied
$\pi$ hole can occupy either degenerate component, so the single-determinant density on
a fixed grid is orientation-arbitrary while the energy is invariant. Measured on OH
(UKS/PBE), the model-free density RMSE against a fixed CCSD reference across fresh
processes read $9.18\times10^{-4}$, $2.74\times10^{-3}$, $2.66\times10^{-3}$,
$1.94\times10^{-3}$ -- each reproducible within its process, all different across
processes, all with correct $\int\rho=9$ and $n_{occ}^{\alpha/\beta}=5/4$; with
`OMP_NUM_THREADS=1` the value is reproducible across processes ($2.333\times10^{-3}$
twice). The scramble is threaded-BLAS non-associativity tipping the near-degenerate SCF
between components (HISTORY 2026-07-01). At production scale the pooled def2-svp record
shows the model-free PBE density error bit-identical across every spec for all 20
closed-shell species and varying for exactly three -- HO $22.96\times$, CH $20.06\times$,
NO $7.99\times$ (max/min over 208/114/50 evaluations) -- at essentially constant energy
(PBE total-energy spans of $1.1\times10^{-6}$, $8.2\times10^{-8}$,
$4.4\times10^{-8}$ Ha) (notebooks/analysis/DENSITY_DIAGNOSIS.md Sec. 3).

**Consequence for training.** With CH and NO in the training set, the density channel of
the loss was owned by an error no functional can close. The measure throughout is the
reference publication's per-electron density error,
$\epsilon_n=N_e^{-1}\int|\rho-\rho_{ref}|$, with $N_e$ the quadrature integral of the
reference density (Dick and Fernandez-Serra, Phys. Rev. B 104, L161109 (2021), eq. (20);
xcquinox/alec/evaluation.py). CH carries an Eq.-20
per-electron density error of $1.55\times10^{-1}$ against its stored reference (NO
$7.78\times10^{-2}$) where the population median is $8.4\times10^{-3}$, and model-free
PBE reproduces CH's error to 0.4% (NN/PBE $=1.004$); the two species own 68--98% of the
channel wherever present, and the channel is flat across 200 epochs (median
first-to-last ratio 0.998) while the energy channels fall by factors up to
$2\times10^{5}$ (DENSITY_DIAGNOSIS.md; HISTORY 2026-08-03). 4 of 114 CH
draws land at or below the closed-shell population median (smallest $0.61\times$), which
a genuine functional-versus-reference gap cannot produce (a fixed functional against a
fixed reference is one number) and component mismatch does.

**Detection.** The discriminating measurement is a MODEL-FREE error that varies across
otherwise identical evaluations at constant energy: same energy, different density is a
degenerate manifold. A cross-model or cross-spec spread of the PBE density reference
above 2--5% relative now warns on the figures and excludes the species from pooled
density anchors (HISTORY 2026-07-01, 2026-08-13). The degenerate-state class extends
beyond the trio on the pre-lock def2-svp record: 12 of 198 species' PBE density
references varied across spec evaluations, led by the BH76 transition state RKT14 and
bn($^3\Pi$)/ClO/NO/OF/OH/CH (HISTORY 2026-07-29).

**Remedy: the orientation lock.** A small, fixed, deterministic, traceless
anisotropic-quadrupole operator $M=\sum_{ij}W_{ij}\langle\chi_\mu|r_i
r_j|\chi_\nu\rangle$ ($W$ symmetric, $\mathrm{Tr}\,W=0$, three distinct eigenvalues,
generic principal axes, built about the nuclear-charge centroid) is added to $h_{core}$
as $\lambda M$ identically in every path that produces a density for the degenerate
radicals: the CCSD/HF reference SCFs, the PBE seed, and both differentiable-SCF backends
(HISTORY 2026-07-02; xcquinox/alec/orientation_lock.py). Tracelessness is load-bearing:
the first-order shift $\lambda\,\mathrm{Tr}(MD)$ is $\sim0$ for a near-isotropic
density, so the lock splits the degenerate $\pi$ pair without materially shifting
energies. The calibrated strength $\lambda=3\times10^{-5}$ induces a $\pi$ splitting of
$\sim10^{-6}$--$10^{-5}$ Ha, about four orders above float64/BLAS noise, while shifting
a closed-shell PBE total energy by $<0.1$ kcal/mol (HISTORY 2026-07-02;
orientation_lock.py header).

**Cache-identity episodes.** The lock is only as good as its reach. Three successive
gaps put locked SCFs against unlocked references for exactly these radicals: the
training-reference generation path omitted the strength (HISTORY 2026-07-11); the final
per-species reference cache was lock-blind -- keyed only on (name, basis) -- and
silently reused unlocked files (HISTORY 2026-07-14); and CH and NO were confirmed
unstamped mid-campaign (file dates predating the cache fix) and were relocked mid-run
with the spec partition recorded (before/after boundary specs 0000-0023 / 0024-0087)
(HISTORY 2026-08-03). The cache-currency predicates now compare the recorded lock
against the requested identity, a legacy file that records none reading as 0.0 -- a hit
for an unlocked run and a miss for a locked one, in either direction (HISTORY
2026-08-24; external_refs.py:1748-1752).

**Residual physics.** The lock delivered reproducibility, not agreement: under it the
production run has zero cross-spec scatter, but only HO landed on the component its
stored reference used ($1.36\times10^{-4}$, ordinary), while CH ($1.58\times10^{-3}$)
and NO ($2.57\times10^{-3}$) sit on a different member of the manifold than their
references (DENSITY_DIAGNOSIS.md Sec. 3). The reference publication met the same wall on
the same class and scaled its density-loss weight $\lambda_n$ by 0.01 for CH and OH; the
corresponding per-species weighting here targets CH and NO (in this pipeline OH is the
member that landed correctly), and is recorded as the ranked remedy rather than applied
mid-campaign (HISTORY 2026-08-03). Related: the open-shell radicals HO, CN and NO (HO
and NO $^2\Pi$; CN $X\,^2\Sigma^+$) also basin-hop in the Wu-Yang OEP inversion used
for $V_{xc}$ references, handled with an
inner-SCF level shift and a looser UKS tolerance ($10^{-2}$ against $2\times10^{-3}$
RKS) (HISTORY 2026-05-02, Phase 5).

**Status.** Lock productionized across every density-producing path and stamped into
every cache identity; component agreement (as opposed to reproducibility) remains an
open, documented limitation with the per-species weighting as the recorded remedy.

---

## 4. Open-shell free atoms (O, N, F): degenerate p-shells and exact spin scaling

Free atoms anchor every atomization energy in the training pools, so free-atom defects
enter every reported AE.

**Two-block spin-scaling defect.** Exchange spin scaling
$$E_x[n_\alpha,n_\beta]=\tfrac12\left(E_x[2n_\alpha]+E_x[2n_\beta]\right)$$
(Oliver and Perdew, Phys. Rev. A 20, 397 (1979)) is exact only when each channel is
evaluated on the spin-unpolarized system it names. The production open-shell exchange
passed the TOTAL-density iso-orbital indicator unchanged into both channels, making the
open-shell functional a different functional from the closed-shell one: with libxc SCAN
in place of the network, the superseded two-block evaluation errs by $-47.9$ mHa
($-30.1$ kcal/mol) on the O atom, $-40.5$ mHa on N, $-26.8$ mHa on OH at fixed density,
and 48.5 mHa (30.5 kcal/mol) self-consistently on O; the pretrained meta-GGA
consequently over-bound H2O/N2/CH4 by 30.5/55.9/20.8 kcal/mol relative to SCAN, of which
transforming the indicator alone recovers 63--86% (residual offsets $-7.6/-7.9/-7.6$
kcal/mol; fractions recomputed from those offsets, the smallest on CH4 at 63.46%)
(HISTORY 2026-08-20, Phase 39;
2026-08-23, Phase 43; notebooks/analysis/NOTES_v5_mgga_vs_scan.md). The remedy computes
every density-matrix-derived descriptor of channel $\sigma$ on the symmetric doubled
density $\mathrm{diag}(P_\sigma,P_\sigma)$ -- for the indicator,
$\alpha_\sigma=\alpha(2\rho_\sigma,4\sigma_{\sigma\sigma},2\tau_\sigma)$ -- with
correlation kept on the total density and $\zeta$; the three-block energy reproduces
PySCF's spin-polarized (`spin=1`) SCAN exchange to $1.8\times10^{-15}$ Ha on O and OH
(the GGA path reproduces `spin=1` PBE to the same $1.8\times10^{-15}$ Ha), and the
assembled potential is the finite-difference derivative of that energy
to $1.0\times10^{-10}$ Ha worst case (HISTORY Phase 43). The discriminating oracle was
one libxc `spin=1` call; a re-derivation sharing the code's frozen-feature assumption
would have passed.

**Orientation dependence of open p-shells.** An open p shell holding 1, 2, 4, or 5
electrons is a P term whose hole converges to an arbitrary orientation; the exact
functional is orientation-invariant but a quadrature on a fixed grid is not, and the
meta-GGA feels it: the SCAN $E_{xc}$ of the free O atom spreads by order 0.1 mHa between
independent unconstrained SCFs at def2-svp/grid 3 (0.26 mHa over one triple of runs,
0.084 over another; 0.21 mHa for F at sto-3g/grid 1) against $1.6\times10^{-3}$ mHa for PBE -- a meaningful
fraction of the 1.0 mHa certificate tolerance, decided by which orientation the SCF
happened to reach. The fidelity certificate named here and below is the per-architecture
pretraining gate: each pretrained network's $E_{xc}$ is evaluated on the parent
functional's own self-consistent densities for the free atoms of the BH76/W4-11 pools,
the molecular differences are folded into atomization-energy offsets, and the
architecture passes only when the worst free-atom error is at most 1.0 mHa and the worst
atomization offset at most 1.0 kcal/mol (xcquinox/alec/cluster/fidelity.py).
Certificate reference densities of such atoms are therefore built
under the orientation lock, after which two independent locked runs agree to
$3.4\times10^{-11}$ mHa; spherical atoms are exempt (HISTORY 2026-08-23, Phase 40). The
pretraining-data generator likewise refuses a spatially degenerate open-shell atom below
grid level 3 or at lock 0 without a recorded waiver: at grid level 1 four locked draws
of the O atom spread $3\times10^{-3}$ in $\rho$, 0.64 in the iso-orbital indicator, and
$3.7\times10^{-6}$ Ha in the stored exchange energy while agreeing to $9\times10^{-10}$
Ha in total energy (grid 3 reproduces to $3\times10^{-11}$) -- two materially different
files could otherwise carry one manifest identity (HISTORY 2026-08-24, Phase 42).

**Free-atom SCF stalls.** The locked PBE O atom at def2-svp/grid 1 exhausted PySCF's
default 50-cycle cap in 2 of 3 attempts, and the unlocked SCAN O atom stalled in 2 of 12
processes from the minao guess (one at a final gradient $8.98\times10^{-5}$ against the
$3.2\times10^{-5}$ criterion), rescued by the second-order stage in 106 total cycles
(HISTORY Phase 40; data.py `_converge_reference_scf` docstring). Tightening the
reference `conv_tol` from $10^{-9}$ to $10^{-10}$ was measured and rejected: at
$10^{-10}$ two processes wander 37 and 42 cycles along the lock's weakly broken flat
direction and land on different densities ($1.4\times10^{-3}$ relative in $\rho$ on 94%
of the grid) -- the lock's reproducibility is calibrated at PySCF's default (data.py,
`_REFERENCE_SCF_CONV_TOL` comment). On the O atom the two SCF backends, cross-seeded
from the same PBE density, reach two members of the degenerate manifold
$9.7\times10^{-5}$ Ha apart -- a property of the manifold, not a backend defect (HISTORY
2026-08-23, Phase 43).

---

## 5. The iso-orbital indicator in density tails

The meta-GGA rung enters through the SCAN/DFS iso-orbital indicator
$$\alpha=\frac{\tau-\tau_W}{\tau_{unif}},\qquad
\tau_W=\frac{|\nabla n|^2}{8n},\qquad
\tau_{unif}=\tfrac{3}{10}\,(3\pi^2)^{2/3}\,n^{5/3},$$
whose division by $\tau_{unif}\propto n^{5/3}$ (and the resulting
$\mathrm{d}\alpha/\mathrm{d}\sigma\sim n^{-8/3}$ sensitivity) concentrates every
pathology in low-density regions.

![The stored iso-orbital indicator produced by `metagga.compute_alpha`, drawn against
$\tau/\tau_{unif}$ at unit density and reduced gradient $s$ = 1: (a) the linear regime
with the smooth floor inset; (b) logarithmic axes, on which the stored value departs from
the raw ratio at the ceiling.](figures_report_pretraining/alpha_indicator.png)

Panel (a) fixes the density and the reduced gradient so that the ratio
$\tau_W/\tau_{unif}$ is 1.6667, the abscissa at which the raw indicator vanishes; the
stored indicator follows the raw ratio to within $5\times10^{-6}$ across the whole panel
and passes through unity at $\tau=\tau_W+\tau_{unif}$, the uniform-electron-gas
point. The inset resolves the immediate vicinity of $\tau=\tau_W$ on a logarithmic
ordinate: the stored value passes through the floor value $p(0)=\delta/2 =
5\times10^{-6}$ of Sec. 5.2 exactly at $\tau=\tau_W$ and continues smoothly to
$4.95\times10^{-7}$ at the panel's left edge, where the raw indicator is negative --
off the SCF manifold, on which $\tau \ge \tau_W$ holds and the floor is the
smallest stored value. Panel (b) repeats the map on logarithmic axes out to a
raw ratio of 1000: stored and raw coincide until the raw value reaches 100, past which
the stored curve is flat at the ceiling of Sec. 5.1 (`metagga._ALPHA_MAX` = 100), with
151 of that panel's 601 sampled points sitting on it and the largest raw value compressed
tenfold. For the physical raw indicator ($\tau \ge \tau_W$) the two bounds give the
domain of every indicator the network sees, $[5\times10^{-6},100]$, and they are what
keeps the division by $\tau_{unif}$ finite in the tail.

**5.1 Diffuse-tail ill-conditioning.** On the diffuse `++` basis the Li 2s tail reaches
$\rho=5.4\times10^{-13}$ at grid 2; with only a value floor in place, $\alpha$ reached
values $\sim10^{28}$ (overflowing the forward SCF energy) and gradients
$\mathrm{d}\alpha/\mathrm{d}\sigma\sim10^{59}$, with the XC-kernel second derivative
$\sim4.2\times10^{43}$ at a point whose bare quadrature weight is 295, not zero -- the
"negligible because $\sim0$ integration weight" argument holds for the energy (weight
$w\rho$) and fails for the kernel the gradient uses (weight $w$). The signature was a
meta-GGA training loss of NaN at step 0 on the `bh76:HLi` group; the GGA and rung-3.5
architectures, which divide by no density power, were immune (HISTORY 2026-07-04, Phase
17). The remedy caps the value ($\alpha\in[0,100]$, above which the DFS/SCAN gate has
saturated: $\tanh^2(\ln(101/2))=0.998$) with the same cap applied to precomputed and
live values. A tail gradient freeze shipped at the same time was later found to be a
defect in its own right -- a potential that is not the derivative of its energy -- and
was removed once the feature-response term was assembled exactly; removing it takes
$\max|\mathrm{d}\alpha/\mathrm{d}\sigma|$ from $1.15\times10^{14}$ to
$2.20\times10^{31}$ on Li at the production basis, yet the full-SCF training gradient
remains finite including at the original failing configuration (25 cycles, pretrained
weights, H/Li/LiH), and the meta-GGA architectures land at the descriptor-free control's
finite-difference floor ($2.1\times10^{-10}$ RKS) (HISTORY 2026-08-06).

*Measurement caveat banked with that change:* the diffuse-basis H atom does not converge
in 25 SCF cycles, and under multithreaded BLAS its unconverged energy and
training-gradient magnitude wander between identical runs -- four repeats spanned 60 mHa
(38 kcal/mol) in energy and eight orders in $|g|_{max}$ ($5.7\times10^{-2}$ to
$1.5\times10^{7}$) -- while `OMP_NUM_THREADS=1` is bit-identical across runs. A single
run on an unconverged open-shell diffuse-basis system is not evidence of anything; such
measurements are repeated or pinned to one BLAS thread (HISTORY 2026-08-06).

**5.2 One-orbital spin channels (H $\alpha$, Li $\beta$; the H atom's total density).**
A spin channel holding one electron is a single orbital, so $\tau=\tau_W$ identically on
the SCF manifold and the raw indicator is the rounding residue of that cancellation --
at most $6.6\times10^{-10}$ on every grid point with $2\rho_\sigma>10^{-8}$ across the
three tested identities (xcquinox/alec/metagga.py, `_ALPHA_SMOOTHING_WIDTH` comment). A
hard lower clip $\max(\alpha_{raw},0)$ made the derivative one-sided exactly on that
manifold, and autodiff returned whichever side the rounding selected: the
$\beta$-channel feature-response term of Li's Fock matrix, itself of magnitude 1.13 Ha,
moved by 0.93 Ha under a $10^{-14}$ relative change of the density matrix (H's by
$4.2\times10^{-3}$ Ha) (metagga.py, `compute_alpha` docstring; HISTORY 2026-08-23,
Phase 43). The cure replaces the clip with a $C^\infty$ positive part
$$p_\delta(x)=\frac{x+\sqrt{x^2+\delta^2}}{2},\qquad
\alpha=\min\!\big(p_\delta(\alpha_{raw}),100\big),$$
with width $\delta=10^{-5}$ in indicator units (equivalently $10^{-5}\,\tau_{unif}$ in
kinetic-energy-density units, so the construction is invariant under uniform density
scaling); the smoothed indicator then ranges over $[\delta/2,100]=[5\times10^{-6},100]$.
The width is anchored to measurement, not chosen: it clears the largest
on-domain rounding residue (draw-dependent, $1.3$--$3.7\times10^{-6}$ in the worst band)
by $2.7$--$7.7\times$, and its energy cost -- the H atom sits at the floor $\delta/2$ on
every point -- is $+1.17\times10^{-7}$ Ha of SCAN exchange (Li $\beta$:
$+3.1\times10^{-7}$), linear in the width and $8.5\times10^{3}$ below the 1.0 mHa
certificate tolerance (metagga.py; HISTORY 2026-08-24). Under the same $10^{-14}$ probe
the H Fock now moves $3.6\times10^{-12}$ Ha, and the H and Li fixed points reproduce
across perturbed seeds to 0.0 and $1.9\times10^{-14}$ Ha; the occupancy-keyed solver
gate and the oracle's straddle mask, each a place where the implementation and its check
had agreed to look away from the same point, were retired. The indicator definition is
recorded in the pretraining-data identity (`ALPHA_DEFINITION`), so a file computed under
the clip is stale for a run under the smooth part. The smoothing's own role is the one
measured above -- the one-orbital Fock response, cured from 0.93 Ha to
$3.6\times10^{-12}$ Ha -- and it is not the source of the pretraining floor. SCAN-parent
pretraining starts at a loss floor of $3\times10^{-14}$ where PBE parents floor at
$2.7\times10^{-32}$, and that floor belongs to the indicator's CEILING: the anchored
network inverts the stored smoothed column exactly before the parent reads it, so a
stored $p(0)=5\times10^{-6}$ recovers $\alpha_{raw}$ to round-off and the end-to-end
anchored exchange loss on the committed mesh is $7.6179\times10^{-32}$; the mesh-carrying
and mesh-free floors stand in the ratio 0.7000000000000004, exactly the 0.7 atomic share
of the loss weighting, so the mesh block contributes at most $6\times10^{-29}$; and the H
atom, one-orbital on every row, floors at $2.85\times10^{-32}$. What the inversion cannot
undo is the cap at `metagga._ALPHA_MAX` = 100: the capped low-density tail rows, whose
exact indicator spans $\sim10^{2}$ to $\sim7\times10^{6}$, carry 100.0 percent of the
weighted exchange MSE on the O atom and on H2O, departing from the exact-$\tau$ libxc
targets by a median $2.55\times10^{-4}$ and at most $5.70\times10^{-4}$ in enhancement
factor. The first reading recorded here -- the floor attributed to the
smoothed-column/exact-$\tau$ asymmetry, with the $\alpha$ = 0 mesh nodes reproducing it --
priced a computation ($1.9\times10^{-14}$) that the run's code path never performs, and
its agreement in decade with the measured floor was coincidental; it is superseded
(HISTORY 2026-08-31, erratum). Nothing is repaired because nothing is broken: the ceiling
is the documented energy-faithfulness bound, and the certificates gate its energy
consequence $194\times$ inside the 1.0 mHa threshold.

![The smooth positive part at width $w=10^{-5}$ (the $\delta$ of this section): (a) $p$
against the hard clip $\max(x,0)$, with the excess over that clip inset; (b) the absolute
error of the round trip through `metagga.invert_smooth_positive_part` against the
conditioning scale of the inversion.](figures_report_pretraining/smooth_positive_part.png)

Panel (a) draws $p_\delta$ (solid) against the hard clip it replaced (dashed) over the
band $|x|\le5\times10^{-5}$ of raw indicator values: the two are indistinguishable away
from the origin and separate only where the clip's derivative is one-sided, the marked
value at the origin being the floor $p_\delta(0)=w/2=5\times10^{-6}$, at which the slope
is 1/2 rather than 0 or 1. The inset follows the excess $p_\delta-\max(x,0)$ down from
that floor onto its own $w^2/4|x|$ asymptote; at the edge of the band the excess is
$4.951\times10^{-7}$ against the asymptote's $5\times10^{-7}$, so the smoothing is
confined to a band of a few widths: the relative distortion of $x$ is 0.99 percent at
five widths and falls to 0.01 percent by fifty. Panel (b) is the round
trip that keeps a stored indicator column readable: $p_\delta$ followed by its exact
inverse returns $x$ with an absolute error of at most $2.78\times10^{-19}$ over the grid
($8.47\times10^{-15}$ relative), 430 of the 1001 grid points returning bit-exactly, and
the largest excursion above the plotted conditioning scale
$\varepsilon\max(|x|,w)(1+w^2/4p^2)$ is $1.01\times$ its value. The inversion is
therefore
exact to the floating-point representation, which is what allows the raw $\tau$ to be
recovered from a column stored under a recorded width.

**5.3 Residual tail response.** What the smoothing does not change is the indicator's
response amplification in the density tail, which is peaked on a shell rather than a
power law: on Li's $\beta$ channel $\max|\mathrm{d}\alpha_{raw}/\mathrm{d}P|$ climbs
from $8.0\times10^{4}$ above $2\rho_\beta=10^{-4}$ to $4.07\times10^{11}$ on the
$2\rho_\beta\in[10^{-9},10^{-8}]$ band (728 points) and falls to $1.2\times10^{1}$
below it -- a log-log slope of $-0.43$ against $2\rho_\beta$, not a power of the
density. The outermost radial shell ($\rho_\beta=1.0\times10^{-9}$, i.e.
$2\rho_\beta=2\times10^{-9}$; 898 points) carries $2.9\times10^{-3}$ Ha per point in
one element of the feature-response Fock term (0.57 Ha over the shell), and a
$10^{-14}$ relative change of the density matrix still moves that channel's
virtual-virtual Fock block by 0.37 Ha through the smoothed derivative. The response annihilates the
occupied orbital of a one-orbital channel exactly, so the SCF fixed point is unaffected;
off-manifold probes are not derivative estimates there (metagga.py docstring;
xcquinox/alec/DEFERRED_WORK.md item 30). Separately, the corrected (exact-derivative)
potential destabilizes the plain meta-GGA SCF at long horizons on two tail-dominated
configurations: `deep_mgga` on H and LiH oscillates under the production mixer at cycle
caps 15--25 (late-step residuals $2.2\times10^{-1}$ and $6.7\times10^{-2}$ Ha against
$1.1\times10^{-10}$ and $1.2\times10^{-8}$ for the prior, self-inconsistent assembly; H
swings through $-0.2747$ Ha at cap 15) while every rung-3.5-stacked configuration is
stable; production training and evaluation run 3 cycles and never enter the regime, and
the recorded remedy path for long-horizon runs is energy-side damping or a DIIS mixer,
never a reinstated gradient freeze (HISTORY 2026-08-10).

---

## 6. Spin-polarized correlation at $\zeta\to\pm1$

**Symptom.** With the spin-polarized correlation channel enabled, entire training runs
went to NaN on pools containing fully polarized species (free H and Li,
$\rho_\beta=0$, $\zeta=\pm1$) (HISTORY 2026-06-03).

**Root cause.** Two distinct mechanisms. (i) The PW92 spin interpolation is built from
$(1\pm\zeta)^{4/3}$ terms whose second derivatives pair with the SAME sign,
$$\frac{\mathrm{d}^2}{\mathrm{d}\zeta^2}\,(1\pm\zeta)^{4/3}
=\tfrac{4}{9}\,(1\pm\zeta)^{-2/3},$$
so $f''(\zeta)$ diverges at $\zeta=\mp1$, where the corresponding factor vanishes -- a
genuine curvature pole at full polarization (Perdew and Wang, Phys. Rev. B 45, 13244
(1992), eqs. (8)-(9)). The full SCF differentiates $v_c$ (itself a first
derivative of $E_c$) a second time, so the exact boundary produces a NaN training
gradient on every fully polarized species. The $(\rho_\alpha-\rho_\beta)/\rho$ chain is
a separate mechanism, not a pole at the boundary: its second derivative is finite at
$|\zeta|=1$ and diverges only as $\rho^{-2}$ where the total density vanishes, which is
what the density floor below guards. (ii) Diffuse-basis tail quadrature noise can drive
the total density slightly negative mid-SCF ($\sim1300$ tail points under a pretrained
network at cycle 0; oneshot.py, `_RHO_TOT_FLOOR` comment); a floor of $10^{-300}$ then
squares to $10^{-600}$, which underflows to zero in the potential's forward-mode
quotient rule, giving $\infty$, and the saturated clip's zero derivative turns it into
$0\cdot\infty=$ NaN. The energy path is forward-only and stayed finite -- hence the
discriminating signature of finite energy with a NaN potential (HISTORY 2026-07-04,
Phase 17).

**Remedy.** One shared helper (`oneshot.uks_zeta`) applies, identically in the energy,
the per-spin potential, and the feature-derivative accumulation:
$\zeta=\mathrm{clip}\big((\rho_\alpha-\rho_\beta)/\max(\rho,10^{-12}),\,
-1+10^{-6},\,1-10^{-6}\big)$, with the gradient frozen on the non-physical
$\rho\le10^{-12}$ tail. The clip's forward bias is
$O(\epsilon\cdot\mathrm{d}E_c/\mathrm{d}\zeta)\sim10^{-8}$ Ha; the invariant that the
guards match across paths (so $v_c$ remains the exact gradient of $E_c$) is held by the
single definition and a test rather than by parallel comments (oneshot.py).

![The PW92 spin interpolation $f(\zeta)$ and its curvature: (a) $f$ over the full
polarization range with the production clip at $|\zeta|=1-10^{-6}$ marked; (b) $f''$ on a
logarithmic ordinate, analytic against a central difference, with the approach to the
pole inset.](figures_report_pretraining/zeta_pole.png)

Panel (a) shows that $f$ itself is unremarkable at the boundary: it is bounded, rises
smoothly to unity at full polarization, and at the clip already reads 0.999996787688811,
so excluding the last $10^{-6}$ of polarization costs $3.2\times10^{-6}$ of the
interpolation's range. Panel (a)'s annotation records the unpolarized curvature 1.7099209342
(`parents._PW_MOD_FZ20`, reproduced exactly). Panel (b) carries the failure: the
analytic curvature climbs from that value to 8550.14 at the clip, a factor of 5000,
with the inset showing the $(1-\zeta)^{-2/3}$ divergence up to the clip, the drawn
data terminating on the dashed clip line. The open circles are a central
second difference of the written $f$, tracking the analytic curvature to
$2.3\times10^{-4}$ relative for $|\zeta|$ up to 0.99, so the pole drawn here is a
property of the interpolation and not of the differencing. The dashed verticals are the
clip: outside them lies the region in which the second derivative the full SCF takes of
$v_c$ returns a non-finite training gradient, and the clip's purpose is to keep the
quadrature from ever evaluating a point there.

**Status.** Closed; fail-loud non-finite guards run in every training loop, and the
gradient-level sweep described in Sec. 8 closes the one-step blind spot this class
originally exploited.

---

## 7. C2H2 and the open non-finite-gradient class

**History.** C2H2 (with HCN and C2H4) drove the earliest degenerate-eigenvalue gradient
failure in the project: reverse-mode differentiation through `eigh` at the degenerate
$\pi$ shells produced spurious large gradients, cured by rebuilding the RKS density
matrix in occupation-mask form $2\,(C\odot occ)\,C^T$ rather than through an
eigendecomposition of the density (HISTORY 2026-04-27, Phase 3).

**The open class.** Two v6 incidents share the finite-loss/non-finite-gradient
signature of Sec. 8 and remain open. (i) On the first group (G1, the size ladder), the
`medium` subset-size-26 cell failed on the open non-finite-gradient defect, and the
published G1 figure set carries 29 of 44 cells with that cell excluded; the record does
not name the failing training group of that cell (HISTORY 2026-08-31; the set regenerated
at 29 cells on the repaired evaluations, HISTORY 2026-09-01). The twin cell on the
attention architecture -- `medium_attn` at the same subset size 26 -- has since completed
and evaluated cleanly, so the defect is specific to the failed cell's own trajectory
rather than a property of the largest subset. (ii) The
DM-carrying group's (G3) preflight compile smoke (run_20260827T163335Z, preflight job
2138042, spec 21 = `deep_combined_attn_3x16` at subset size 26) aborted at per-molecule
step 27 on the group `bh76:C2H2` with a FINITE loss of 18.419290403706498 and 36 of 36
gradient leaves non-finite (first leaf `.xnet.net.layers[0].weight`, 80 NaN elements,
0 Inf) (scratch/v6_diag/g3_dm/logs/compile_smoke_probe.out).

**Reproduction attempts on the preflight incident.** Local replays of the aborting
group (`bh76:C2H2` with the injected `anchor:H` regularizer) on the same architecture
(`deep_combined_attn_3x16`) at the anchored v6 model class with shape padding returned
finite losses ($8.5\times10^{-6}$--$1.4\times10^{-4}$) and 36/36 finite gradient leaves
at def2-svp/grid 1, 6-31++G**/grid 3, and the production identity, from both fresh and
pretrained starts and with the density and $V_{xc}$ channels active
(scratch/v6_diag/repro_c2h2_grad_attn_svp.log, repro_c2h2_grad_attn_631ppgss_g3.log,
repro_c2h2_grad_attn_prod_g3.log, repro_c2h2_grad_attn_prod_g3_pretrained.log,
repro_c2h2_grad_V2_channels.log). The defect therefore does not reproduce off-cluster
at these identities; a cluster-side decomposition probe is queued, and no root cause is
asserted here.

**Status.** Open at the time of writing; the G1 cell is withheld from the published
figures, and recovery of both incidents is deferred until the probe reports.

---

## 8. Degenerate-eigenvalue backward amplification: the symmetry-breaking shift

**Symptom.** A production training replay aborted at step 4 on the group
`bh76:OH+N2_to_H+N2O` with a FINITE loss ($1.085617\times10^{-2}$, every component
finite) and all 36 gradient leaves NaN; the same state evaluated unpadded, in the same
process, was finite, with a 0.219% forward-loss shift -- orders beyond round-off unless
amplified (HISTORY 2026-07-19 to 2026-07-20, Phases 31-32).

**Root cause.** The `eigh` reverse-mode rule carries $1/(\lambda_i-\lambda_j)$ factors,
and the eigenvector backward amplifies matrix-level round-off $\epsilon$ by
$\epsilon/\mathrm{gap}^2$. The graded symmetry-breaking diagonal that floors
symmetry-exact degeneracies was sized at $10^{-8}$ against forward accumulation noise,
ignoring the backward amplification: at $10^{-8}$ the ratio is
$\epsilon/\mathrm{gap}^2=2.2$ -- order unity -- so graph-order round-off differences
introduced by the shape-padding pass turned the backward pass into non-finite garbage on
the padded graph while the unpadded evaluation of the identical state stayed finite. A
five-way single-seam on-cluster ablation isolated the constant: raising
`SYM_BREAK_SHIFT` to $10^{-6}$ alone cured the failing epoch (step-4 loss unchanged to
all seven printed digits), the combined-eigh toggle containing it also cured -- a
second, independent confirmation -- and the leading density-tail candidate and two
other seams left the NaN in place (HISTORY 2026-07-20, Phase 33).

**Remedy.** `SYM_BREAK_SHIFT` $=10^{-6}$, centered in the admissible window
$[4.7\times10^{-7},3\times10^{-6}]$ whose lower wall is $\epsilon/\mathrm{gap}^2\le
10^{-3}$ and whose upper walls are the $3\times10^{-5}$ orientation-lock splitting
(which the quasi-random diagonal must not rival) and the Weyl bound (measured maximum
eigenvalue motion $9.99\times10^{-7}$). Reference caches produced under the old constant
carry at most $\sim10^{-6}$ Ha inconsistency. The enabling instrument is the
gradient-level abort guard: the loss-only guard had a one-step blind spot (a finite-loss
/ non-finite-gradient step passes, corrupts every weight, and aborts one step late on
the NEXT group, misattributing the origin -- the `ae:CO` group was exonerated exactly
this way); every training loop now sweeps the gradient pytree before the optimizer
update and names the first non-finite parameter path (HISTORY 2026-07-19, Phase 30).

**Status.** Closed; the constant is pinned by a two-sided sizing test, and the same
finite-loss/NaN-gradient detection instrument is what flagged the open Sec. 7 class.

---

## 9. Na2 and relative-error denominator collapse

**Symptom and cause.** Three incidents of one arithmetic class, in which a relative
error is formed against a target near zero. (i) Na2's atomization-energy error was
inflated $1340\times$ by a relative-AE loss with no floor on the reference magnitude
(HISTORY 2026-06-10). (ii) BH76 reactant species (HO, CH3, N2, F2) entered specs with
placeholder targets of 0.0, collapsing the relative-error denominator to $10^{-8}$ so a
$\sim0.5$ Ha residual blew up to $\sim2.5\times10^{7}$ (HISTORY 2026-05-10, Phase 6).
(iii) A `Li+` spec processed after neutral Li overwrote the shared atom anchor index,
training the cation toward the neutral anchor -- a $\sim5$ eV ionization-potential bias
(HISTORY 2026-05-10).

**Remedy and status.** An AE target floor; neutral-only (H, Li) atom anchors; auxiliary
reaction species excluded from the AE channel (`aux_only_names`); and the
reaction-form AE (`ae_as_reactions`), which scores atomization against the network's own
self-consistent atom energies instead of fixed anchors. One follow-on defect is part of
this record: the resolved-config serializer silently dropped `ae_as_reactions`, so three
run generations trained the fixed-anchor form their source configuration had turned off;
the serializer now round-trips every field by construction (HISTORY 2026-08-10). Closed.

---

## 10. H2 and N2: protocol exclusions at the meta-GGA level

The meta-GGA variant of the DFS pretraining protocol drops H2 and N2, giving 28 systems
against the GGA level's 30 (xcquinox/alec/dfs_pretrain_set.py, `MGGA_EXCLUDED`). The
fidelity-certificate oracle pool restores N2 through its fixed benchmark systems, so a
SCAN-parent architecture certifies over 38 systems where a PBE-parent one certifies over
39 -- one fewer atomization -- with the level selected per parent by
`fidelity.dfs_level_for_parent` (xcquinox/alec/cluster/fidelity.py; HISTORY 2026-08-31).
The five meta-GGA family architectures pass that certificate at the production identity
with worst free-atom error $5.2\times10^{-3}$ mHa and worst atomization error
$2.5\times10^{-3}$ kcal/mol against the 1.0/1.0 gate (HISTORY 2026-08-31). Closed by
construction; recorded because a 38-versus-39 system count in the certificates is
designed, not a coverage defect.

---

## 11. The geometry-units incident (angstrom read as Bohr)

**Symptom and cause.** The held-out geometry reader divided already-angstrom coordinates
by the Bohr-per-angstrom factor, shrinking every held-out molecule by $\sim1.89\times$
and garbaging all BH76/W4-11 reaction energies for the networks AND the PBE baseline
alike; the corrected BH76 PBE MAE is 11.82 kcal/mol where the corrupted pipeline
reported 182 (HISTORY 2026-05-31). Training was never affected (its geometries come from
a separate source).

**Detection and remedy.** Coordinates are read as angstrom and the JSON caches were
rebuilt; a Hartree-units guard now checks every frozen reference reaction and
ionization energy at loss construction, raising above a 10 Ha sanity ceiling and
catching the $\sim627\times$ kcal/mol-vs-Hartree unit-error class before a loss is
built (HISTORY 2026-06-02; xcquinox/alec/losses.py:996-1008). Closed.

---

## 12. Per-species OEP overrides

**Symptom.** The Wu-Yang optimized-effective-potential inversion that generates
$V_{xc}$ references converges to the default density-error tolerance for most of the
pool but plateaus above it for a small set of species.

**Remedy.** Plateau detection (objective and density-error trend deques) terminates a
stuck inversion rather than burning the iteration budget, and a per-species override
table carries bespoke tiers discovered by a tuning sweep: Be, C+, F2, F2O, HF, HS, N2O,
and O3 (minimum achieved density errors $4.63\times10^{-3}$, $1.40\times10^{-2}$,
$9.43\times10^{-3}$, $4.84\times10^{-3}$, $4.13\times10^{-3}$, $1.19\times10^{-2}$,
$4.70\times10^{-3}$, $9.22\times10^{-3}$, each accepted at $1.7\times$ its own minimum),
plus a manual CF4 entry (plateau $2.486\times10^{-3}$, just above the $2\times10^{-3}$
RKS default) (xcquinox/alec/external_refs.py, `_PER_SPECIES_OEP_OVERRIDES`; HISTORY
2026-05-06, Phase 6). The intermediates cache is keyed on the grid level so a grid
change cannot silently reuse stale references. Closed; the override tiers' literature
annotations remain flagged for pre-publication verification
(external_refs.py:838-843).

---

## 13. SCF-trajectory instability under trained functionals

**Symptom.** With the SCF depth raised from 3 to 25 cycles (`full_25`), held-out
reaction MAE collapsed to 75--110 kcal/mol against 13--19 for the 3-cycle runs and
$\sim15$ for PBE, uniformly across matched specs (HISTORY 2026-06-24, Phase 11).

**Root cause.** Training and evaluation both scored ONLY the final SCF cycle's energy.
For hard species the NN-driven SCF, given 25 cycles, left the stable PBE-initialized
basin and the constant linear mixer could not damp it: the captured per-cycle traces
show a clean period-2 oscillation in the tail (`t-hooo` steps 18--24 alternate
-223.40/-223.75/-223.32/-223.72/-223.31/-223.71/-223.30, a peak-to-peak swing of 0.45 Ha)
or a still-drifting endpoint (`s4-c2v` ends 10.3 Ha,
$\sim6500$ kcal/mol, off PBE). The final-step energy is then an arbitrary oscillation
phase. Splitting held-out reactions by SCF convergence proves the mechanism: converged
reactions score MAE $\sim$ 12--28 kcal/mol, non-converged 140--485, with 26--59
non-converged species per spec (HISTORY Phase 11).

**Remedy.** The DFS trajectory supervision was ported verbatim from the reference
implementation: a tail-weighted energy loss (the last $\min(N,10)$ cycles with
quadratically rising weights, generalized to any cycle count) and a step-decaying mixer
$\alpha_{mix}=0.3^{step}+0.3$, plus tail-weighted reporting. After the fix the 25-cycle
run is on par with the 3-cycle one (validation-best held-out median 15.5 against 16.4
kcal/mol, mean 28.07 against 23.69, PBE $\sim14.9$) -- a partial, in-progress comparison
at the time of measurement, the 25-cycle run having completed 23 specs against 73
(HISTORY 2026-07-02).

**Related artifacts.** (i) The garbage W4-11 atomization energies of the early held-out
record were a separate undertraining verdict -- SCF divergence of undertrained networks
scored verbatim -- and not an evaluation bug; three genuine train/held-out overlap leaks
were repaired independently of it (HISTORY 2026-06-04). (ii) Genuine three-cycle SCF
divergences scored verbatim persist at low rates and track the descriptor stack:
first-cycle energy residuals above 0.1 Ha on 6.4% of rows for the multishell rung-3.5
architecture, 3.5% for the attention rung-3.5 form, 0.4--0.9% elsewhere (HISTORY 2026-08-20,
Phase 39). Status: closed for the trajectory-supervision class; the per-architecture
divergence rates are disclosed on the figures rather than repaired.

---

## 14. Bounded-map saturation of the anchored parameterization

**Context.** No architecture pretrained to its parent functional under the point-wise
protocol: on frozen parent densities, the worst-system atomization offsets from the
parent ran 25.7 to 56.1 kcal/mol per descriptor-carrying architecture against 4.1--4.2
for the descriptor-free controls (per-system magnitudes overlap at the low end -- one
descriptor-carrying cell reads 3.5 kcal/mol against a control range of 2.3--4.2; worst
values per architecture recomputed from the recorded H2O/N2/CH4 offset table), driven by a shared H-atom pretraining error ($+13.7$ mHa against $+0.8$)
multiplied by the hydrogen count and by molecular extrapolation of features the
atoms-plus-mesh pretraining set never constrained (HISTORY 2026-08-20, Phase 39). The
remedy anchors every network to its parent at initialization: the enhancement factor is
built as
$$F=1+L\big(z_{parent}+g_\theta\big),\qquad
L(x)=\Lambda\,\sigma\!\big(x-\ln(\Lambda-1)\big)-1,\qquad
z_{parent}=\ln\!\frac{(\Lambda-1)\,F_{parent}}{\Lambda-F_{parent}},$$
with $\sigma$ the logistic function, $\Lambda$ the bound of the squash (1.804 for
GGA-rung exchange, 1.174 for meta-GGA exchange -- the DFS ceiling, networks.py:751 --
and 2.0 for the correlation non-negativity map), $z_{parent}$ clamped to
$\pm40$, and the trainable output $g_\theta$ zero-initialized so $F$ equals the parent
at initialization to round-off (xcquinox/alec/networks.py `_AlecLOB`;
xcquinox/alec/parents.py `lob_preimage`). Anchored pretrains then sit
$8.7\times10^{-7}$--$9.2\times10^{-6}$ from the parent in $\max|\Delta F_x|$ where the
legacy pretrains sat 0.039--0.090 (HISTORY 2026-08-30, 2026-08-31).

**The artifact.** The anchored correction inherits the map's saturation: the trainable
term carries $L'(z_{parent})$ as a prefactor, measured 0.45 at $s=0$ falling to 0.007 by
$s=20$ where the parent exchange approaches its ceiling, with the correlation floor the
mirror case. The parameterization therefore suppresses trainability exactly where the
parent approaches a bound of $(0,\Lambda)$, and the strongest form occurs at the
meta-GGA exchange ceiling: SCAN's exchange enhancement attains the bound in its
single-orbital limit ($F_x=1.174$ at $\alpha=0$ as $s\to0$), so $F_{parent}$ sits AT
$\Lambda=1.174$, the pre-image clamps at $z_{max}=40$, and the trainable prefactor is
zero in float64 -- the map returns the parent to within $\Lambda e^{-40}$ and the
network cannot move it there (parents.py, `lob_preimage` docstring). The measured
consequence, on subset-size-18 validation-best cells at $r_s=2$: exchange corrections
of every generation
converge to one family ($+0.07$ to $+0.16$ bump peaking near $s=3$), while correlation
corrections diverge -- the unanchored networks grow a large-$s$ correlation correction
($+0.79$ to $+0.92$ for $s\ge2$, built from a flat $\pm0.01$ pretrained start) that
removes PBE's systematic barrier bias (BH76 mean signed error $-0.20$ for the
unanchored deep_3x16 at subset size 12, against PBE's $-6.6$ to $-7.5$ kcal/mol),
whereas the anchored network's correction collapses past $s=2$ ($+0.29$ at $s=2$ to
$+0.01$ at $s=6$) and retains the parent's bias ($-7.75$ for the anchored medium; the
two signed-error cells are not subset-matched)
(HISTORY 2026-08-31;
notebooks/analysis/figures_dfs_step7_dfs6311_grid3_v6g1_size_val_best/anchored_vs_unanchored_fx_fc.png,
pretrain_fx_fc_delta_all.png, trained_fx_fc_delta_best.png in the same directory). On
the first anchored group's published cells the pattern was W4-11 beaten against the
cell's own PBE anchor in 18 of 18 cells (5.99--12.25 against 13.1--13.6 kcal/mol), the
combined pool in 17 of 18, BH76 in 5 of 18 (best 6.51 against 7.73) -- the first
published state, at 18 cells (HISTORY 2026-08-31). On the regenerated 27-cell set over
the repaired evaluations the pattern holds and strengthens: W4-11 beaten in 27 of 27
cells (the same 5.99--12.25 against 13.1--13.6 kcal/mol), the combined pool in 26 of 27,
BH76 in 9 of 27 (best 6.46 against 7.73), each cell scored on its own strict held-out
slice (HISTORY 2026-09-01; the run's per-spec
`eval_holdout_val_best/test_set.csv`).

**Status.** Open as a design question rather than a defect: the anchor bought four
orders of magnitude in pretraining fidelity, and whether it costs the barrier physics is
under controlled test (an anchored run of the identical unanchored architecture is
queued; HISTORY 2026-08-31).

---

## 15. Cross-cutting reproducibility constraints

Three process-level effects repeatedly masqueraded as, or interacted with, the physics
above; each is now pinned.

- **Threaded-BLAS non-associativity and degeneracy.** Reduction order is not associative
  in floating point, and a degenerate or unconverged system amplifies last-bit
  differences to observable scale: the OH component scramble (Sec. 3), the H-atom
  wander (Sec. 5.1), and the O-atom reference PBE SCF differing by $2.2\times10^{-7}$ Ha
  between two identical processes above one BLAS thread (HISTORY 2026-08-23, Phase 43).
  Bitwise claims are made only under pinned thread counts.
- **Process-memory-dependent quadrature.** PySCF sizes its XC grid loop, its
  incore-versus-direct integral path, and its density-fitting block sums from the memory
  the process has left, so the same specification returned energies and densities
  differing at the $10^{-13}$ level with process history (the def2-svp O atom: 1 against
  54 grid blocks, the production-basis O atom spanning two; C5H8: 4, 8, or 222 auxiliary
  blocks at 888 auxiliary functions; methane at 288 auxiliary functions: two different
  Hartree-Fock energies across memory histories). Every reference mean field now pins
  the grid block size (12544 points), the integral path (incore iff $n_{ao}\le211$), and
  a fixed 240-vector auxiliary blocking, after which records are bitwise equal across
  processes holding 0.7 GB to 3.6 GiB at one thread
  (HISTORY 2026-08-24, Phase 43). A reference whose quadrature order depends on process
  memory is not a reference; the pins enter no cache identity, so existing caches stay
  valid.
- **The returned-density convergence ceiling.** PySCF's SCF converges on the gradient of
  one Fock form, then runs one extra diagonalization and returns THAT density, accepted
  under a looser rule -- so the density a record carries is one step past the converged
  point, bounded by $3\times$ the gradient bar rather than the bar. The singlet-CH2 SCAN
  record rebuilt its orbital gradient at $3.237\times10^{-5}$ against the
  $3.16\times10^{-5}$ bar on a record stamped converged after 7 DIIS cycles (a bent-CH2
  PBE control reaches $2.26\times$ the bar). The generator therefore holds a record
  first to its own stamped gradient at $10^{-6}$ relative (record integrity: a density
  and Fock pieces from different SCFs are refused whatever their gradient) and then to
  PySCF's own $3\times$ ceiling (pretrain_data_gen.`_GRADIENT_CHECK_MARGIN` $=3.0$),
  rather than to a bar PySCF does not hold its returned density to (HISTORY 2026-08-25;
  data.py `_REFERENCE_SCF_CONV_TOL` comment).

---

## 16. Summary table

| Species / system | Artifact | Magnitude | Detection | Remedy |
|---|---|---|---|---|
| C2 | Two SCF configurations; non-convergent oscillating DIIS; aufbau flip on dm0 ingestion | 50.10 kcal/mol inter-branch gap; 0.12 Ha DIIS spread | Branch tolerance ($10^{-4}$ Ha) against trajectory minimum; cross-spec $E_\mathrm{PBE}$ spread $>10^{-4}$ Ha | Orbital-pair-seeded second-order rescue; refusal over silent recording |
| C2 (reference cache) | Trajectory-dependent grid pruning | 26840 vs 26568 grid points | Loader shape gate | First-attempt grid pinned; DF identity stamped; intermediates deleted with the final file |
| Li | SCAN/diffuse-basis DIIS basin escape | Endpoint 3.4 Ha above the solution | Two-stage non-convergence refusal | Second-order solve from the best-gradient trajectory point (102 cycles total) |
| Li ($\beta$), H | One-orbital channel: $\tau=\tau_W$, indicator a rounding residue; hard-clip derivative | 0.93 Ha Fock response under a $10^{-14}$ density change (0.37 Ha residual after smoothing) | $10^{-14}$ density-matrix perturbation probe of the Fock response | Smooth positive part, $\delta=10^{-5}\tau_{unif}$; cost $1.2\times10^{-7}$ Ha |
| Li 2s tail (meta-GGA) | Iso-orbital indicator ill-conditioning at $\rho\sim10^{-13}$ | $\mathrm{d}\alpha/\mathrm{d}\sigma\sim10^{59}$; kernel $4\times10^{43}$ | Step-0 NaN loss | Value cap at 100 (smoothed domain $[5\times10^{-6},100]$); exact feature-response term (no gradient freeze) |
| H, Li (free atoms) | $f''(\zeta)\to\infty$ at $\zeta=\pm1$; negative-tail underflow in the potential JVP | All-NaN training runs | Finite energy with NaN potential | $\zeta$ clip $\pm(1-10^{-6})$; $\rho$ floor $10^{-12}$; shared guard helper |
| OH, CH, NO | $^2\Pi$ orientation degeneracy; reference/SCF component mismatch | Up to $23\times$ model-free density-error spread; CH $\epsilon_n=0.155$ vs median 0.0084 | Model-free error varying at constant energy | Traceless-quadrupole $h_{core}$ lock ($3\times10^{-5}$); per-species density weighting recorded |
| O, N, F (atoms) | Degenerate p-shell orientation; two-block spin-scaling defect | 0.1 mHa SCAN $E_{xc}$ spread; 30.1 kcal/mol on O (two-block) | Cross-process record spread; libxc `spin=1` oracle | Orientation-locked references; per-channel doubled-density descriptor blocks |
| C2H2 (+ HCN, C2H4) | Degenerate $\pi$ eigh gradients (closed); open non-finite-gradient class | G1 cell lost (group unnamed); G3 preflight abort on `bh76:C2H2` at finite loss 18.42 | Gradient-leaf sweep before the optimizer update | Occupation-mask DM rebuild; cluster probe queued (open) |
| BH76 OH+N2 group | eigh backward amplification $\epsilon/\mathrm{gap}^2$ | All 36 gradient leaves NaN at finite loss | Gradient sweep; single-seam ablation | `SYM_BREAK_SHIFT` $10^{-8}\to10^{-6}$ (window $[4.7\times10^{-7},3\times10^{-6}]$) |
| Na2 (+ aux species, Li+) | Relative-error denominator collapse | $1340\times$ AE error; $2.5\times10^{7}$ residual; $\sim5$ eV IP bias | Loss magnitude | AE floor; neutral anchors; aux exclusion; reaction-form AE |
| H2, N2 | Meta-GGA protocol exclusion | 28 vs 30 pretraining systems | Certificate system count | Oracle pool restores N2 (38 vs 39 certified systems) |
| All held-out | Angstrom read as Bohr ($1.89\times$ shrinkage) | BH76 PBE MAE 182 vs 11.82 kcal/mol | $>10$ Ha ceiling on frozen reference reaction/IP energies at loss construction | Unit fix; caches rebuilt |
| Be, C+, F2, F2O, HF, HS, N2O, O3, CF4 | OEP plateau above default tolerance | Density-error plateaus $2.5\times10^{-3}$--$1.4\times10^{-2}$ | Plateau detection | Per-species override tiers ($1.7\times$ own minimum) |
| t-hooo, s4-c2v | Trained-functional 25-cycle SCF oscillation/drift | Period-2, 0.45 Ha peak-to-peak; 10.3 Ha drift | Per-cycle energy trace; convergence-split MAE | DFS tail-weighted loss + decaying mixer |
| CH2 (singlet) | Returned density one step past convergence | Rebuilt gradient $1.02\times$ the bar on a stamped-converged record | Generator gradient check | Gradient stamp; integrity at $10^{-6}$ relative; $3\times$ ceiling |
| Anchored networks | Bounded-map pre-image saturation | $L'$: 0.45 at $s=0$ to 0.007 at $s=20$; BH76 signed error $-7.75$ vs $-0.20$ | $F_x/F_c$ difference curves vs parent | Controlled anchored-vs-unanchored test in progress |
