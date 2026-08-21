# Pretraining-fidelity program: requirement, findings, design, gate, verification

Status: design decisions taken 2026-08-21 (Section 7); implementation in progress. Supersedes the
campaign practice of every run before it.

## 1. Requirement (binding)

Pretraining must reproduce the baseline method for every architecture: the pretrained
exchange and correlation networks, evaluated through the production code path (energy and
potential, RKS and UKS) on the production identity (6-311++G(3df,2pd), grid level 3, density
fitting), must reproduce the parent functional (PBE for GGA-rung architectures, SCAN for
meta-GGA architectures) on atoms, molecules and atomization energies within a stated
tolerance. No campaign stage may start, and no result may enter the figure pipeline, without
a machine-checked certificate that this holds for every architecture in the sweep. The DFS
pretraining set may be used in its entirety to meet the requirement.

## 2. What was found (2026-08-20; all numbers independently re-derived)

Atomization-energy offset of the pretrained networks from their parent on frozen parent
densities, production footing, kcal/mol (H2O / N2 / CH4):

| architecture | parent | offset | cause |
|---|---|---|---|
| deep_3x16 | PBE | -2.5 / -4.2 / -2.4 | pretraining fit |
| deep_attn_3x16 | PBE | -2.3 / -4.1 / -3.1 | pretraining fit |
| deep_cusp_3x16 | PBE | -13.2 / -4.2 / -25.7 | H-atom pretraining error (+13.7 mHa) x H count |
| deep_rung35_3x16 | PBE | -13.5 / -3.5 / -29.1 | H-atom error + molecular extrapolation of DM features |
| deep_rung35_attn_3x16 | PBE | -29.5 / -20.4 / -56.1 | same, amplified by attention |
| deep_rung35ms_3x16 | PBE | -22.0 / -30.9 / -42.8 | same |
| deep_mgga_3x16 | SCAN | -30.5 / -55.9 / -20.8 | alpha frozen at the total density in the UKS exchange (exact transform exists; correcting it alone gives -7.6 / -7.9 / -7.6) + pretraining fit |

Three defects, in order of origin:

D1. Open-shell exchange with descriptor features. `oneshot.split_exc_energy_uks` and
`oneshot._uks_spin_resolved_vxc` apply E_x[rho_a, rho_b] = (E_x[2 rho_a] + E_x[2 rho_b]) / 2
(Oliver and Perdew, Phys. Rev. A 20, 397 (1979)) by doubling rho and quadrupling sigma per
channel but evaluate every descriptor feature at the physical density and pass the same
feature block into both channels. Introduced with the UKS Fock (`d15ba4de4`, 2026-04-18);
documented as an approximation with no magnitude (`e9a805d0e`, 2026-05-24), whose test
`test_split_energy_openshell_passes_same_features_both_exchange_terms` pins the defect as the
contract. The exact doubled-spin system is the spin-unpolarized density matrix
diag(P_sigma, P_sigma); its features are well defined for every density-matrix descriptor:
alpha_sigma = alpha(2 rho_sigma, 4 sigma_sigma, 2 tau_sigma) (verified against libxc SCAN
spin=1 to <1e-12 Ha), rung-3.5 occupancies [n_sigma, n_sigma] (inside the [0, 1] bound), DM
statistics of diag(P_sigma, P_sigma); the cusp feature is geometry-only and unchanged.

D2. Pretraining footing and coverage. `pretrain_data_gen._atom_columns` stores open-shell
atoms with spin-resolved parent targets against total-density inputs; the set is seven
atoms plus a synthetic mesh, so density-matrix features never see a molecular environment and
the H atom -- one electron, fully polarized -- is fit to +13.7 mHa by every cusp-carrying
network. The integration-weighted loss is blind to both (the architecture with the lowest
exchange residual, deep_rung35_attn_3x16 at 2.1e-6, has the largest offset).

D3. No gate. The preflight validates specs, the subset ledger, references and a compile smoke;
the pretrain stage records losses; nothing compares a pretrained network with its parent
(`cluster/_preflight.py` docstring: "the preflight does not pre-stage or validate it").

## 3. Design

### 3.1 Exact spin scaling for every density-matrix feature (D1)

Every UKS exchange evaluation receives the feature block of the symmetric doubled density
diag(P_sigma, P_sigma) for its channel:

- `descriptors.assemble_descriptor_features` gains a per-channel form: each Descriptor
  exposes `compute_for_spin_channel(mol_data_or_dm, sigma)` returning its features for
  diag(P_sigma, P_sigma): MetaGGAAlphaDescriptor -> alpha(2 rho_sigma, 4 sigma_sigma,
  2 tau_sigma) (new per-spin tau from `ao_grad` and `dm[sigma]`); DMRung35 / Multishell ->
  the channel occupancy in both spin slots; DMStatistics -> statistics of diag(P_sigma,
  P_sigma); Cusp -> unchanged.
- Energy: `split_exc_energy_uks(model, ..., features_a, features_b, features_tot)`; exchange
  at (2 rho_sigma, 4 sigma_sigma, features_sigma); correlation on the total density with
  the total-density features (unchanged). `fixed_density_total_energy`,
  `solver_manual._compute_total_energy_uks` build the three blocks.
- Potential: `_uks_spin_resolved_vxc` and the manual-solver loop (`_features_for`,
  `_vx_nn_spin`, `_feature_response_uks`) evaluate each channel at its own block; the
  feature response differentiates each channel's P -> f_sigma(P) map; `solver_pyscfad`'s
  UKS branch and `_reassemble_features` follow.
- Closed shells: rho_a = rho_b gives identical blocks, so RKS and every closed-shell UKS
  number is unchanged byte for byte (pinned by test against the archived tree).
- The PBE anchor (`_nn_fx_local_uks`, zero extras) is retired or aligned; it is off in
  production.

Oracles (tests, executed in CI and in the preflight certificate):
- O1 Parent reproduction of the code path: with the network replaced by the parent's own
  enhancement factors (libxc PBE / SCAN evaluated at the library's inputs, features
  ignored), the library's UKS energy equals libxc spin=1 on open-shell atoms to 1e-10 Ha;
  with features present the per-channel inputs equal the libxc spin-polarized ingredients.
- O2 Potential = derivative of the energy: central-difference check of the assembled UKS
  Fock matrices against the energy on H, Li, N, O with every descriptor active (the existing
  FD test, re-pointed at the new energy and extended from Li/def2-svp to the production
  basis).
- O3 Closed-shell byte identity against the archived tree for every architecture.
- O4 H atom: one orbital, alpha identically 0, rung-3.5 occupancy of a doubled single
  orbital; the network's exchange energy equals the spin-scaled unpolarized evaluation.

### 3.2 Pretraining that delivers the parent (D2)

- Footing: open-shell rows are posed per spin channel at (2 rho_sigma, 4 sigma_sigma,
  features of diag(P_sigma, P_sigma)) with the parent's spin-unpolarized enhancement
  factors at those inputs as targets (this is what the exact spin scaling evaluates).
  Correlation rows keep the total density with zeta (polarized cnet).
- Coverage: the DFS pretraining set in its entirety (inventory in Section 6), generated at
  the production identity with the parent functional's self-consistent densities, plus every
  atom of the BH76 / W4-11 pools; the synthetic mesh is kept as a regularizer only.
- Weighting: the loss carries an explicit per-system energy term (E_xc^NN - E_xc^parent on
  each system, Hartree) beside the point-wise enhancement-factor term, so a network cannot
  lower the point-wise residual while missing a system's energy; the H atom is one system
  among many with a term of its own.
- Acceptance inside the pretrain stage: after training, the energy-space check of Section
  3.3 runs on the pretrain node; failure exits non-zero and blocks the campaign.

### 3.3 The gate: per-architecture physics certificate (D3)

A new preflight step `fidelity_certificate(cfg, run_dir)` runs once per architecture after
the pretrain stage and before the train array:

1. Loads the pretrained networks through the production model builder.
2. Builds the oracle set: every atom of the pools (all open-shell), the DFS pretraining
   molecules, and a fixed molecule set spanning the pool's elements, on frozen parent
   densities at the production identity (PBE for GGA-rung, SCAN for meta-GGA).
3. Computes E_xc^NN - E_xc^parent per system (production footing, energy path) and the
   implied atomization-energy offsets; runs O1-O4 on the installed code.
4. PASS iff max |dE_xc| per atom <= tol_atom and max |dAE| <= tol_AE and O1-O4 pass.
   Proposed: tol_AE = 1.0 kcal/mol, tol_atom = 1.0 mHa (configurable in the YAML but
   never above 2.0 / 2.0 without an explicit `fidelity.override_reason`).
5. Writes `<run_dir>/pretrain/<arch>/fidelity_certificate.json` (inputs, every number,
   tolerances, code hash, verdict).

Enforcement: the train task refuses to start without a PASS certificate for its architecture;
`merge_v4_arms` and the figure loaders refuse a run whose architectures lack one; the
certificate's table is rendered into the figure provenance footer.

### 3.4 Per-architecture workflow verification (before any YAML is rendered)

For every architecture in `ARCHITECTURES`: datagen -> pretrain -> certificate -> train (a
handful of steps on two cells) -> eval, at a small basis locally and at the production
identity for the certificate step, with the oracles O1-O4 inside; results recorded in
HISTORY as the baseline matrix every later change is measured against.

### 3.5 Campaign v6

Every arm is resubmitted under the corrected code and pretraining (the descriptor-free
architectures do not meet tol_AE = 1.0 either at 2.3-4.2 kcal/mol). Retired: v4gga
descriptor arms, v5, v5mgga2, and the earlier v3/v4 records as quantitative results; they
remain as the documented failure record.

## 4. Verification of this program

- Every code change: tests first (RED shown against the archived tree), oracle tests
  executed, `py_compile`, and TWO independent reviews per commit, each required to execute
  the oracles themselves (one on the physics, one on the workflow end to end).
- The rendered sbatch scripts and YAMLs for v6 get the same two reviews before the
  submission commands are handed over.

## 5. Sequence

1. Spin-scaling fix (3.1) with oracles; commit; two reviews.
2. Pretraining footing + data set + energy term (3.2); commit; two reviews.
3. Certificate + enforcement (3.3); commit; two reviews.
4. Workflow matrix (3.4); HISTORY baseline.
5. v6 YAMLs + rendered scripts; two reviews; handover.

## 6. DFS pretraining protocol (replicated verbatim)

Filled from the vendored dpyscf source inventory before sign-off.

## 7. Decisions (2026-08-21)

- Certificate tolerance: tol_AE = 1.0 kcal/mol on atomization energies and tol_atom = 1.0 mHa
  on atomic E_xc, for every architecture; no override without `fidelity.override_reason`.
- Spin scaling: the symmetric doubled density diag(P_sigma, P_sigma) defines the per-channel
  feature block for EVERY density-matrix descriptor (alpha, rung-3.5 single and multishell,
  DM statistics); the cusp feature is geometry-only and unchanged; pretraining rows are posed
  on the same footing.
- Campaign v6 resubmits every architecture under the corrected code, the new pretraining and
  the certificate; the descriptor-free architectures are included (2.3-4.2 kcal/mol today).
- Pretraining set: the DFS pretraining set in its entirety, plus every atom of the BH76 /
  W4-11 pools (open shells per spin channel), plus the synthetic mesh as a regularizer, with
  a per-system energy term in the loss beside the point-wise enhancement-factor term.
