# The SCAN-seeded meta-GGA arm against SCAN: held-out analysis of the 2026-08-20 pull

Scope: the first seven cells of the v5 campaign (`dfs_step7/dfs6311_grid3_v5`,
run `run_20260815T034818Z`, `deep_mgga_3x16` at training subset sizes 1-7, SCAN-seeded
full_3 SCF, final and validation-best checkpoints), read against the SCAN and PBE
comparators on the same held-out reactions, with the GGA arm (`dfs6311_grid3_v4gga`,
`deep_3x16`) as the reference case in which fine-tuning does improve on its parent.
The question addressed is why the meta-GGA cells sit at SCAN rather than above it.

## 1. Data and slices

- NN energies and PBE comparator legs: `checkpoints/spec_00NN/eval_holdout*/per_reaction.json`
  (reaction-level signed errors `error_nn_kcalmol`, `error_pbe_kcalmol` against the GMTKN55
  references; W4-11 rows are own-atom atomization reactions, `ae_as_reactions`).
- SCAN comparator leg: `scan_pool_energies_6-311++G_3df_2pd_.json` (converged SCAN total
  energies at the production identity), combined with each reaction's stoichiometric
  coefficients; the figure pipeline uses the same cache for its SCAN lines.
- Slices differ between figure sets and must not be mixed. The merged cross-arm view removes
  the union of both arms' validation slices (75 reaction names) and scores 38 BH76 + 94 W4-11
  reactions at subset size 1; the v5 per-arm figures score 58 + 120; the v4gga per-arm figures
  48 + 113. SCAN's BH76 one-bucket WTMAD-2 is 12.1 kcal/mol on the merged slice and 18.1 on
  the v5 per-arm slice. Cross-arm statements below use the merged view; the decomposition in
  Section 3 uses the full held-out pool of each run (61 BH76, 118-120 W4-11 reactions), which
  gives the same picture as the merged-like slice (stated where relevant).

## 2. Headline (merged view, final checkpoints)

BH76 one-bucket WTMAD-2 and held-out density RMSE, each as the ratio to the architecture's
own-rung reference (PBE for GGA-rung architectures, SCAN for the meta-GGA), mean over the
evaluated subset sizes:

| architecture | cells | E / E_ref (BH76 WTMAD-2) | D / D_ref (BH76 species) | cells beating the reference (BH76 column; figure criterion = combined energy-density ED) |
|---|---|---|---|---|
| deep_3x16 (GGA, PBE ref) | 11 | 0.67 | 1.11 | 11 / 11 |
| deep_rung35_3x16 (PBE ref) | 11 | 0.82 | 0.88 | 11 / 11 |
| deep_mgga_3x16 (SCAN ref) | 7 | 1.33 (0.94-1.01 at ss 2-5; 1.6-2.1 at ss 1, 6, 7) | 1.19 (1.01-1.08 at ss 2-5) | 1 / 7 (over all three legs: 1 of 21 cells) |

On energy alone the meta-GGA is below SCAN in 3 of the 7 BH76 cells (ss 3-5, by 1.5-6%) and
deep_rung35_3x16 at ss 6 is above PBE (1.03) although it beats PBE on the combined metric; the
last column follows the figure pipeline's criterion. At subset sizes 2-5 the meta-GGA reproduces
SCAN's held-out accuracy on both energy and density; at 1, 6 and 7 it is worse than SCAN (the validation-best checkpoints recover part
of the gap at 1 and 6: BH76 23.4 vs 29.7 and 24.4 vs 31.4 kcal/mol on the per-arm slice).
Final training losses are 1.6e-8 to 1.1e-4 (the minimum recorded loss of each cell 5.8e-14 to
7.3e-11), i.e. the training molecules are fit far below the target accuracy; validation MAE at
the selected epoch is 6.7-20 kcal/mol; cell ss1 stopped early at epoch 150 with its best
validation at epoch 25 and ss6 at 175 with its best at epoch 50, so their final checkpoints
are 126 epochs past the validation optimum.

## 3. Signed-error decomposition

Mean absolute error, signed mean (bias) and standard deviation of the signed error, kcal/mol,
on the full held-out pool of the v5 run (61 BH76 reactions, 118-120 W4-11 reactions), final
checkpoints. The PBE and SCAN rows are the comparators on the same reactions (W4-11 comparators
on the 120-reaction set; every row has finite NN, PBE and SCAN legs).

| functional | BH76 MAE | bias | sd | W4-11 MAE | bias | sd |
|---|---|---|---|---|---|---|
| PBE | 7.73 | -7.47 | 6.49 | 13.5 | +11.4 | 12.4 |
| SCAN | 6.39 | -5.97 | 5.65 | 3.87 | -2.16 | 4.77 |
| NN meta-GGA, ss 1 | 10.20 | -9.81 | 8.98 | 26.8 | +21.1 | 26.0 |
| NN meta-GGA, ss 2 | 6.18 | -0.12 | 8.70 | 9.29 | +3.76 | 17.8 |
| NN meta-GGA, ss 3 | 5.75 | -3.18 | 8.07 | 19.1 | +17.3 | 13.6 |
| NN meta-GGA, ss 4 | 5.70 | -2.61 | 7.85 | 13.0 | +10.3 | 13.3 |
| NN meta-GGA, ss 5 | 5.97 | -1.44 | 8.12 | 7.83 | +0.62 | 15.3 |
| NN meta-GGA, ss 6 | 10.80 | -10.65 | 16.5 | 17.6 | +14.0 | 20.1 |
| NN meta-GGA, ss 7 | 7.80 | +4.63 | 7.68 | 17.4 | -15.5 | 22.8 |

Per-reaction relation to SCAN on BH76 (ss 2-5): root-mean-square difference NN minus SCAN
8.6-11.0 kcal/mol, correlation of the two error vectors 0.21-0.33, slope of NN error on SCAN
error 0.32-0.47. On the merged-like slice (43 BH76 reactions) the same cells give NN MAE
4.98-5.31 against SCAN 5.39, with NN bias -0.6 to -3.6 against SCAN -4.9 and NN sd 6.1-7.0
against SCAN 5.4.

The GGA arm on the v4gga run (50 BH76 reactions; PBE MAE 7.42, bias -6.62, sd 7.09) gives, for
deep_3x16 at ss 3 and 7, NN MAE 4.56 and 4.96 with bias -0.25 and -0.70 and sd 5.6 and 7.3.

## 4. Reading

1. The error of both parent functionals on the held-out barrier heights is dominated by a
   systematic underestimation: |bias| / MAE = 0.97 for PBE and 0.93 for SCAN. Fine-tuning on
   a handful of atomization and BH76 reaction-energy points removes that bias in both arms (NN biases of order
   -0.1 to -3 kcal/mol at ss 2-5) and adds reaction-level scatter that is nearly uncorrelated
   with the parent's residual error (sd 5.65 -> 7.8-8.7 for the meta-GGA; correlation with
   SCAN 0.21-0.33). The trained network is therefore not SCAN with a small correction but a
   different functional whose barrier-height errors happen to have SCAN's magnitude. For PBE
   the removable bias (7.5 kcal/mol) exceeds the injected scatter and the GGA arm nets a gain
   at every subset size on the merged-slice WTMAD-2 (11 of 11 cells; on the plain MAE of the
   full 50-reaction pool four cells, ss 1, 2, 5 and 6, still sit above PBE); for SCAN (bias 6.0, sd 5.7) the exchange is even, and the MAE stays
   at 5.7-6.2 against 6.4.
2. On W4-11, the reaction type the cells are trained on, the meta-GGA is two to seven times
   worse than SCAN (sd 13-26 against 4.8) with cell-dependent biases of -15.5 to +21.1
   kcal/mol (+0.6 to +17.3 over ss 2-5). Atomization energies of held-out molecules (other elements, larger systems)
   extrapolate poorly from at most seven fitted reactions plus the exact-total-energy atomic
   anchors (H -0.5, Li -7.4781, O -75.0673 Ha), which oblige the functional to absorb the
   basis-set incompleteness of 6-311++G(3df,2pd) into its atomic energies.
3. The training regime interpolates the training set: the loss (`L5_gradnorm_vxc_step7`,
   per-molecule updates, fixed channel weights AE 1 / BH76 1 / IP13 1 / vxc 1 / rho 20,
   weight decay 1e-4, 200 epochs) has no channel anchoring the network to its parent
   functional, and the training losses (final 1.6e-8 to 1.1e-4, minima 5.8e-14 to 7.3e-11) show
   the training molecules are fit far below the target accuracy. Validation-best selection (34
   unique validation reactions) limits the damage
   in the two cells that drift late but does not create an improvement over SCAN in any cell.
4. The density channel holds: at ss 2-5 the held-out density error is 1.0-1.1 times SCAN's
   on the merged slice, so the failure to improve is an energy-channel property.
5. Footing caveats that belong in the methods text rather than in this explanation: the vxc
   channel compares a (rho, sigma)-only network potential (`compute_vxc_nn`) with a
   multiplicative OEP potential obtained from the CCSD density, while the tau/density-matrix
   feature response enters the SCF Fock matrix through `feature_response_vxc` but not the vxc
   loss; the evaluation SCF is the three-cycle truncation from the SCAN seed
   (`cycles_run` 3, `scf_converged` False on essentially every row).

## 5. Pretrain fidelity: the "SCAN" starting point is 20-56 kcal/mol from SCAN on atomization energies, and why

The pretraining fit of the meta-GGA correlation network is six times worse in
integration-weighted mean-squared residual than the GGA one (cnet 4.5e-4 against 7.6e-5;
xnet 4.4e-5 against 4.6e-5; `pretrain_metadata.json` of the two runs). The energy-space
consequence was measured directly (2026-08-20): the pretrained networks of both arms were
evaluated on frozen PBE densities at the production identity (6-311++G(3df,2pd), grid level 3;
MoleculeSpecs from the v5 spec files; models assembled as the train task assembles them from
`xnet.eqx`/`cnet.eqx`; energies through `fixed_density_total_energy`, i.e. the production
footing with Oliver-Perdew spin scaling for open shells) and compared with SCAN (meta-GGA) and
PBE (GGA control) on the same density and grid (PySCF `NumInt.nr_rks`/`nr_uks`; the two routes
to the PBE exchange-correlation energy agree to 1.9e-10 Ha, the two SCAN routes to 2.0e-10 Ha).

dE_xc = E_xc^NN[rho_PBE] - E_xc^parent[rho_PBE], mHa (kcal/mol in parentheses):

| species | meta-GGA net - SCAN | GGA net - PBE (control) |
|---|---|---|
| H (2S=1) | -0.42 (-0.27) | +0.82 (+0.51) |
| Li (2S=1) | +7.11 (+4.46) | +6.13 (+3.84) |
| C (2S=2) | +22.95 (+14.40) | +12.01 (+7.54) |
| N (2S=3) | +33.63 (+21.10) | +11.54 (+7.24) |
| O (2S=2) | +27.80 (+17.45) | +10.33 (+6.48) |
| H2O | -21.64 (-13.58) | +7.99 (+5.01) |
| N2 | -21.85 (-13.71) | +16.40 (+10.29) |
| CH4 | -11.85 (-7.43) | +11.39 (+7.15) |

Implied atomization-energy offset of the pretrained network relative to its parent
(a negative value is over-binding), kcal/mol:

| | meta-GGA net vs SCAN | GGA net vs PBE |
|---|---|---|
| H2O | -30.5 | -2.5 |
| N2 | -55.9 | -4.2 |
| CH4 | -20.8 | -2.4 |

Both networks miss their parent's exchange-correlation energy by 10-30 mHa per species, of
order 0.1-0.3% of E_xc; for the GGA the miss has one sign and a near-constant per-electron
size, so it cancels in atomization energies to 2-4 kcal/mol, whereas for the meta-GGA it flips
sign between the open-shell atoms (+7 to +34 mHa) and the closed-shell molecules (-12 to -22
mHa) and accumulates to 20-56 kcal/mol. The numbers were re-derived independently (a second
script sharing no function with the probe: its own PBE SCF, AO values and rho / grad / tau
from PySCF's `eval_rho`, the Oliver-Perdew spin combination by hand through the model's
`eval_ex` / `eval_ec`, SCAN from pointwise `eval_xc` quadrature) and agree to 0.013 mHa on every
species and to 0.01 kcal/mol on every dAE; the same density matrix and bit-identical grid
weights feed both sides, `E_pbe = E_non_xc + E_xc_pbe` holds to 1e-14 Ha, the atomic spin
states are the production ones, and the loaded weights differ from a fresh initialization by
0.52 Ha in E_xc(O). Open-shell O and C atom PBE solutions are orientationally degenerate
(successive SCFs differ in the density matrix by up to 0.18 in a matrix element and in
E_xc^SCAN by up to 7e-5 Ha), which limits the atom entries to ~0.1 mHa unless the orientation
lock is applied; the signal is 28 mHa.

The cause is in the production evaluation of open-shell exchange, not in the network. The
exact exchange spin-scaling relation E_x[rho_a, rho_b] = (E_x[2 rho_a] + E_x[2 rho_b]) / 2
(Oliver and Perdew, Phys. Rev. A 20, 397 (1979)) is implemented in `split_exc_energy_uks`
(`oneshot.py:493-495`) by doubling rho and quadrupling sigma for each spin channel, but the
descriptor feature vector -- here the meta-GGA alpha = (tau - tau_W)/tau_unif of the TOTAL
density -- is passed unchanged into both channel evaluations (the P2-02 note at
`oneshot.py:475-483` records this as an approximation for context features that have no
doubled-spin transform). Alpha does have one: alpha_sigma = alpha(2 rho_sigma, 4 sigma_sigma,
2 tau_sigma), and libxc's own spin-polarized SCAN exchange equals two spin-unpolarized
evaluations at exactly those doubled-spin ingredients to better than 1e-12 Ha on the O-atom
grid. The production path therefore evaluates the meta-GGA exchange network at a feature the
functional was never meant to see for polarized densities; the ratio alpha_sigma / alpha_total
on the O atom runs from 0.44 (first percentile) through 0.96 (median) to 1.99 (99th), so the
error is neither small nor a constant rescaling. Correcting only this term -- alpha transformed
per spin channel, everything else unchanged -- moves the atomic exchange offsets from +7.6 /
+19.2 / +32.9 / +28.2 mHa (Li / C / N / O) to +2.7 / -1.8 / -5.4 / -8.3 mHa, the same sign as
the molecules, and the atomization-energy offsets from -30.5 / -55.9 / -20.8 to -7.6 / -7.9 /
-7.6 kcal/mol (75%, 86% and 63% of the effect). The same frozen-feature spin scaling sits in
the SCF potential (`_uks_spin_resolved_vxc`, `oneshot.py:764`, used by every solver backend and
by the vxc loss), so it acts on the training and the evaluation of every open-shell species in
the meta-GGA arm, not only on this probe. A secondary, smaller contribution comes from the
pretraining rows themselves: `pretrain_data_gen._atom_columns` stores the spin-resolved SCAN
targets of open-shell atoms (`:129-132`) against total-density inputs rho, sigma, alpha
(`:94-96`, `:142-153`); undoing the doubled-spin substitution entirely changes the dAE values by
+1.0 / +7.3 / -2.0 kcal/mol only. The descriptor-free GGA networks are exempt from this defect by
construction -- `deep_3x16` and `deep_attn_3x16` carry no descriptor feature, so their spin
scaling is exact -- which is why they sit at 2-4 kcal/mol (deep_attn -2.3 / -4.1 / -3.1);
after the alpha correction the meta-GGA residual (-7.6) is three times theirs, not twelve.

The descriptor-carrying GGA-rung networks of the v4gga run were probed the same way against
PBE (production footing, frozen PBE densities, every number re-derived independently to
0.001 kcal/mol). dAE (H2O / N2 / CH4, kcal/mol): deep_cusp_3x16 -13.2 / -4.2 / -25.7;
deep_rung35_3x16 -13.5 / -3.5 / -29.1; deep_rung35_attn_3x16 -29.5 / -20.4 / -56.1;
deep_rung35ms_3x16 -22.0 / -30.9 / -42.8. The symptom is the meta-GGA's; the causes are not.
The cusp descriptor is pure geometry (`exp(-2 Z r_min)` and `tanh(log(sum_A Z_A / r_A) / 5)`),
so passing it unchanged into both spin channels is exact; its offset is a pretraining-fit
failure on the H atom (+13.7 mHa against +0.8 for the descriptor-free network), multiplied by
the hydrogen count of each molecule (for CH4: 8.8 + 4 x 13.7 = 63.5 mHa of atomic offset
against 22.6 for the molecule). The rung-3.5 occupancies admit a doubled-spin-density
evaluation formally (the occupancy is linear in the spin density matrix) but it leaves the
descriptor's domain -- values up to 1.9 against the [0, 1] Bessel bound the pretraining
saw -- and makes the energies 40-80 times worse, so the frozen-feature convention is the
only defensible one for them; their offsets combine the same H-atom error (these
architectures also carry the cusp descriptor) with molecular extrapolation of density-matrix
features that an atoms-plus-mesh pretraining never constrained (the molecular offsets flip
sign for rung35_attn and rung35ms: -6.5 / -0.3 / -12.8 and -7.3 / -12.8 / -14.1 mHa). The
pretraining loss does not see any of this: deep_rung35_attn_3x16 has the lowest exchange
pretraining residual of the six architectures (2.1e-6, twenty times below the control) and
the largest offset.

Consequence for the campaign question: the meta-GGA cells do not start at SCAN. They start
from a functional that over-binds by 20-56 kcal/mol on the very reaction type they are then
fine-tuned on, and the 200-epoch fine-tune on at most seven atomization reactions has to undo
that offset before it can improve on anything; the cell-dependent W4-11 biases of Section 3
(-15.5 to +21.1 kcal/mol; +0.6 to +17.3 over ss 2-5) are the residue of that correction, and the injected scatter is in part
the functional being reshaped far from its parent. SCAN seeding of the SCF (the v5 change)
fixed the footing of the density; the footing of the functional for open-shell species is a
separate defect in the spin-scaling of the alpha feature (DEFERRED #25), and no
descriptor-carrying architecture of either arm starts its fine-tuning at its parent
functional (DEFERRED #26).

## 6. Implications for a SCAN-beating recipe (proposals, not results)

- First, the open-shell exchange footing: transform the meta-GGA alpha feature per spin
  channel in `split_exc_energy_uks` and `_uks_spin_resolved_vxc` (alpha_sigma from 2 rho_sigma,
  4 sigma_sigma, 2 tau_sigma, which needs the per-spin kinetic-energy density on the grid), with
  libxc's spin-polarized SCAN as the acceptance oracle (two unpolarized evaluations at the
  doubled-spin ingredients must reproduce it to numerical precision), and pose the open-shell
  pretraining rows at the same inputs. This changes the energy and the potential of every
  open-shell species for the meta-GGA architectures and therefore invalidates the v5 meta-GGA
  cells (trained and evaluated under the frozen-alpha scaling); it is a campaign decision, not a
  patch.
- A parent-anchor regularizer (penalizing the departure of the network's enhancement factors
  from SCAN's on the training densities, or a replay of the pretraining set during
  fine-tuning) would trade part of the bias removal for most of the injected scatter.
- Validation-best selection should be the default checkpoint for the meta-GGA cells; the
  final-checkpoint curve at ss 1 and 6 is a late-training artefact.
- Training reactions of the held-out type (barrier heights) are the only route to correcting
  SCAN's barrier-height error without extrapolating from atomization energies.

## 7. Reproduction

- Sections 2-4: `per_reaction.json` of each channel (signed errors), the SCAN energy cache and
  the reaction stoichiometry; the merged-like slice removes the names listed in both arms'
  `validation/val_reactions.json`. Bias = mean signed error, sd = population standard
  deviation of the signed error, MAE = mean absolute error; the NN-minus-SCAN RMS and
  correlation are over the reactions with a finite NN leg.
- Section 5: the probe script is a scratch tool (`scratch/probe_pretrain_vs_scan.py`, not
  tracked); its recipe is the one stated in Section 5 and it runs in under a minute on a
  workstation (eight species, PBE SCFs included).
