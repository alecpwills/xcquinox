# Parent-anchored enhancement factors: requirement, findings, design, verification

Status: design for implementation (2026-08-25, revised the same day after the physical
review recorded in `.superpowers/sdd/2026-08-25-parent-anchor/review-spec-report.md`).
Follows `SPEC_pretrain_fidelity_program.md` (the certificate and the pretraining protocol it
binds) and supersedes the energy-weight sweep as the means of reaching the certificate for
anchored architectures.

## 1. Requirement (binding)

Every architecture's functional equals its parent -- PBE for the GGA rungs, SCAN for the
meta-GGA rungs (`fidelity.resolve_parent`) -- at initialization, pointwise in both
enhancement factors on every row the model integrates, so that the pretraining-fidelity
certificate (1.0 mHa per free atom, 1.0 kcal/mol per atomization energy) holds by
construction and training starts from the parent rather than from an approximation of it.
Unanchored architectures keep today's model byte for byte: the closed-shell fixtures and
the recorded campaigns (v4, v5) stay valid as they are.

## 2. What was found (2026-08-25, energy-weight sweep job 2134963, def2-svp / grid level 3, 1000 steps)

The sweep was the program's measurement of the one open value, the weight of the per-system
energy term. Its first twelve rows settle the question before it ends:

- deep_3x16 at w_E = 0, 0.1, 1, 10, 100: rms per-system XC error 4.25, 4.04, 4.02, 3.55,
  2.19 mHa; maximum over free atoms 6.86, 3.96, 4.54, 4.87, 3.24 mHa (gate 0.5); maximum
  atomization-energy error 3.12, 1.68, 1.74, 6.81, 8.00 kcal/mol (gate 0.5). The term does
  what it was written to do on the per-system energies and the atom gate, and the AE gate
  does not follow it. deep_cusp_3x16 sits at 23 to 27 kcal/mol at every weight.
- The residual is the correlation network's: the pointwise pretraining losses are
  5.0e-7 (exchange) against 3.4e-4 (correlation), and at w_E = 0 the per-system correlation
  errors are 11.9 (Na2), 11.3 (AlCl3), 9.9 (F2), 6.2 (Na), 5.4 (F-), 5.4 (Cl2) mHa, all of
  one sign, against exchange errors of at most 4.3 mHa. At w_E = 100 the term redistributes
  them (both signs, 2 to 4 mHa) and cannot remove them: the energy term re-weights a fit the
  network does not make. The measurement (`_energy_term_inputs`, `_system_energy_targets`)
  is the row quadrature on both sides, so the residual is the fit and nothing else.
- The atomization gate is structurally far tighter than the atom gate: for an eleven-atom
  molecule, 0.5 kcal/mol is 0.8 mHa over twelve per-system errors, about 0.07 mHa each,
  seven times under the atom gate. A pointwise fit at loss 5e-7 leaves per-system energies
  1e-4 relative (4 mHa) off; the gate asks for 2e-6.

Conclusion: no weight brings a pointwise fit of the parent to the certificate on this data.
Anchoring the model to its parent removes the requirement instead of tuning it. The
correlation network's fit quality is investigated separately (Section 6); it is not on the
path to the campaign once the anchor holds.

## 3. Design

### 3.1 Parent enhancement factors in JAX (`xcquinox/alec/parents.py`, new)

Pure-JAX, per-point functions of the physical inputs the networks already receive, on the
model's own conventions (`models.AlecGGAModel._exc_pieces`), with every constant taken as
libxc defines it (libxc 4.3.4 is the oracle; the paper values rounded to five figures
are 2.6e-6 relative off it at s = 1 and are not used):

- Exchange, per spin channel on the doubled density (Oliver and Perdew, PRA 20, 397 (1979);
  the model's `spin_channel` footing, the row's `(rho_x, sigma_x)` being the doubled channel
  quantities): `F_x^parent(rho, sigma[, alpha])` with `s = |grad rho| / (2 k_F rho)`,
  `k_F = (3 pi^2 rho)^(1/3)`. PBE: `F_x = 1 + kappa - kappa / (1 + mu s^2 / kappa)`,
  kappa = 0.804, mu = beta pi^2 / 3 with beta = 0.06672455060314922 (Perdew, Burke,
  Ernzerhof, PRL 77, 3865 (1996), eq. 14, at libxc's constants). SCAN: `F_x(s, alpha)` of
  Sun, Ruzsinszky and Perdew, PRL 115, 036402 (2015) and its supplemental material (the
  pieces h1x, h0x = 1.174, gx, the switching function fx(alpha) with its two-branch form
  at alpha = 1 written so that both branches are finite under differentiation), with any
  regularization libxc applies to the indicator reproduced, since libxc is the oracle.
- The meta-GGA indicator: the network's row carries the smoothed, capped indicator
  (`metagga.compute_alpha`: the smooth positive part of width 1e-5 and the ceiling
  `_ALPHA_MAX = 100`), which is the manifest's `ALPHA_DEFINITION`. The parent is evaluated
  on the raw indicator recovered from the row by `metagga.invert_smooth_positive_part`
  below the ceiling, and at the ceiling above it, where SCAN's switching function has
  saturated (the difference is 1.8e-3 relative in F_x on the rows above the ceiling and
  8.8e-8 Ha on the N atom's exchange energy; the smoothing itself contributes 1.2e-7 Ha on
  the H atom). Both are stated floors of the SCAN oracle, four orders under the certificate.
  The parent's derivative with respect to the indicator is taken through the same smoothed
  quantity the network differentiates, so the potential inherits the model's regularization
  (deferred item 30) rather than the raw indicator's response.
- Correlation, on the total density, relative to the model's own baseline: the model's
  correlation energy density is `rho eps_c^base F_c` with `eps_c^base` the polarized PW92
  (an anchored correlation network is polarization-aware by construction: the zeta-blind
  case is refused when the pair is built, so the unpolarized-baseline branch does not
  exist for the anchored class);
  the pretraining data divides its `Fc` targets by the polarized PW92 for open-shell rows
  (`pretrain_data_gen.py`, the polarized branch), and every v6 configuration runs
  `use_polarized_correlation: true`, so for the campaign the two agree and `zeta` is already
  in the correlation row. `F_c^parent = eps_c^parent(rs, zeta, t[, alpha]) / eps_c^base(rs, zeta)`,
  so that `rho eps_c^base F_c^parent` is the parent's correlation energy density exactly.
  PBE: `eps_c^PBE = eps_c^PW92(rs, zeta) + H`,
  `H = gamma phi^3 ln[1 + (beta/gamma) t^2 (1 + A t^2) / (1 + A t^2 + A^2 t^4)]`,
  `A = (beta/gamma) / (exp(-eps_c^PW92 / (gamma phi^3)) - 1)`,
  `phi = [(1+zeta)^(2/3) + (1-zeta)^(2/3)] / 2`, `t = |grad rho| / (2 phi k_s rho)`,
  `k_s = (4 k_F / pi)^(1/2)`, beta as above, gamma = (1 - ln 2) / pi^2 (PBE 1996,
  eqs. 3 to 8; PW92: Perdew and Wang, PRB 45, 13244 (1992), the repository's
  `pw92c_polarized_scalar`). SCAN: the correlation of PRL 115, 036402 and its supplement
  (eps_c^0, eps_c^1, H0, H1, fc(alpha), the constants as libxc carries them).
  An anchored architecture whose correlation network is zeta-blind is refused at
  construction: the data's targets and the model's baseline would disagree for open shells
  (measured 14.9 mHa on the N atom's correlation term), and no v6 architecture is built so.
- Every quantity the parent needs is a physical input present in every row the network
  sees: `rho`, `sigma`, the row's `zeta` for the correlation net, and the row's indicator
  for the meta-GGA nets. The parent reads the raw row quantities, whatever coordinates the
  MLP is fed (Section 3.7); the anchor itself changes no input width.

Oracle (executed, part of the test suite, Section 4): pointwise agreement with libxc
through pyscf (`dft.libxc.eval_xc`, the route the certificate's parent energies take) for
PBE and SCAN, exchange and correlation, on a grid of `(rs, s, zeta, alpha)` covering the
model's domain (`rho > 1e-10`, the network's tail threshold; libxc screens below about
1e-12 and the model clamps below 1e-10, so the tails are compared by their integrated
energy, Section 4 V2), `zeta = +-1`, `alpha = 0`, `alpha` at the ceiling, and on the stored
grids of real molecules; and first derivatives with respect to `rho`, `sigma` and the
indicator against libxc's `deriv=1` output, since the SCF potential is the autodiff of the
model's energy density and the parent term is inside it.

### 3.2 The anchored transform

The live transform is in the networks' `_core` (`networks.py`): the network's output is
gated by `tanh^2(s)` (the UEG damping; the meta-GGA nets add their indicator gate) and
squashed by `_AlecLOB(limit)`,

    F = 1 + L(x),   L(x) = limit * sigmoid(x - ln(limit - 1)) - 1,

so that `F` lies in `(0, limit)` with `F = 1` at `x = 0`: limit 1.804 for the GGA
exchange nets, 1.174 for the meta-GGA exchange nets, 2.0 for the correlation nets (a
non-negativity squash, not a bound). The constraint classes of `constraints.py` are
carried by no registered architecture and are not touched.

The anchor adds the network's gated output in the pre-image of that map, at the point
where the map returns the parent:

    z_parent = L^-1(F_parent - 1) = ln[(limit - 1) F_parent / (limit - F_parent)],
    F = 1 + L(z_parent + gated),   gated = tanh^2(s) * net(x)  (as today),

with `z_parent` clamped to `[-Z_MAX, Z_MAX]`, Z_MAX = 40, for the rows where the parent
sits at a bound (SCAN's exchange within an ulp of 1.174 on an alpha = 0 sweep of the N
atom; SCAN's correlation at zeta = +-1): there `F` is the parent to within
`limit e^(-40)` and the network cannot move it, which is the parent's own limit and not a
degeneracy of the transform. Properties, each pinned by a test (Section 4): `gated = 0`
gives `F = F_parent` to round-off; `F` stays in `(0, limit)` for every `gated`; at
`F_parent = 1` the form is today's transform term for term (`z_parent = 0`), so an
unanchored network is unchanged bitwise; `s -> 0` gives `F -> F_parent(s -> 0)` through the
existing gate (the UEG value 1 for PBE, the functional's own small-`s` value at the local
indicator for SCAN, which is the parent's behaviour and not a violation); the slope
`dF/d(gated) = L'(z_parent)` is largest where the parent is mid-range and vanishes at the
bounds. `ScalingSymmetric` is not carried by any registered architecture.

The transform lives inside the networks' forward because pretraining applies the networks
directly to packed rows; the model's `eval_*` paths reach the same code. A network carries
`parent: str | None` as a static field (`"pbe"`, `"scan"`, or none); `None` is today's
network exactly. `eval_core` (the unconstrained value used by the constraint report) reports
`1 + L(z_parent + gated)` as the constrained value and `F_parent + gated` as the raw one.

### 3.3 Initialization

Anchored architectures run with `zero_init_final_layer = True` whatever the registry
entry says (`shallow`, `shallow_attn`, `medium`, `medium_attn` carry False): the final
layer's weight and bias are zero, `net(x) = 0` at every point on both the plain and the
attention paths (the attention block precedes the final layer), `gated = 0`,
`F = F_parent`. The gradient of the final layer is the penultimate activation times the
loss gradient, non-zero, so training proceeds from the parent; the layers before it
receive no gradient until the final layer moves, which is the standard behaviour of a
zero-initialized output layer. The pyscfad backend tests pin `zero_init_final_layer=False`
for the unanchored nets they exercise; anchored nets are covered by their own tests.

### 3.4 Configuration and identity

`ArchitectureConfig.parent_anchor: bool = False`. Anchoring is a property of the model
class, not of the run: the v6 group configurations state it for every architecture through
one sweep-level switch (`model.parent_anchor: true` in the grid configuration, refused when
an architecture's parent cannot be resolved or its correlation network is zeta-blind),
applied when the networks are created (`create_network_pair`) and carried into
`from_arch`, which resolves the parent by rung. It is part of the architecture identity
everywhere the architecture is identified: the resolved configuration, the pretraining
manifest and the training specs, and the certificate's architecture description. The
networks' parameters are unchanged in shape, so a checkpoint's leaves do not reveal the
anchor state; the loader therefore reads it from the configuration the checkpoint was
written under (the run's resolved configuration, the pretrain directory's manifest) and
refuses to load networks recorded under one anchor state into a model of the other.

### 3.5 Pretraining, certificate, the weight placeholder

The pretraining protocol is unchanged. At initialization the pointwise loss is the
difference between the JAX parent and libxc's on the stored rows (Section 3.1's oracle
bounds it, at round-off for PBE and at the indicator floors for SCAN), the per-system
energy term is that difference integrated, and the certificate reads the parent against
itself: it passes at initialization (measured on the design's numerics: -1.3e-8 mHa on the
N atom under the polarized baseline), and a pretrain from there converges where it starts,
which the V4 test states as the certificate holding after 50 and after 2500 steps. The
energy-term weight is no longer the value that decides whether the certificate can be met:
the refusal of the weight-zero placeholder in the campaign configurations applies to
unanchored configurations only, and an anchored configuration states its weight (0.0 is
exact) without a sweep.

### 3.6 Training and evaluation

Training is unchanged: the loss gradient flows through the network; the SCF potential is
the autodiff of the model's energy density and includes the parent's derivatives by
construction (Section 3.1's derivative oracle covers them, at the smoothed indicator for
SCAN). Below the network's tail threshold (`rho <= 1e-10`) the model returns `F = 1` as it
does today while the parent's own value there may differ; the energy in those tails is
1.2e-12 Ha on the N atom and 1.9e-12 Ha on Ne, nine orders under the certificate, and the
oracle compares the tails by that integrated energy rather than pointwise. Evaluation and
the figure pipeline are unchanged. The anchored model is a different model class from the
unanchored one: v6 results are read against the parent they start from, and the
closed-shell fixtures, recorded for the unanchored class, stay the unanchored class's;
anchored architectures are covered by the oracles of Section 4.

### 3.7 Descriptor coordinates (the DFS set)

The v6 model class feeds its networks the coordinates of Dick and Fernandez-Serra (PRB
104, L161109 (2021)), read off the vendored source (`dpyscfl/net.py`, `get_descriptors`,
the dpyscfl branch): the density coordinate `x0 = ln(rho^(1/3) + 1e-5)` on the total
density, for the correlation net only (Eq. 7) -- the exchange net receives no density
coordinate (the source's `X_L(n_input=1, use=[1])` selects the reduced-gradient column
alone, and with it the meta-GGA indicator column), so its enhancement factor is invariant
under uniform density scaling, `rho -> lambda^3 rho`, `sigma -> lambda^8 sigma` at fixed
`s`, bitwise in the implementation; the spin coordinate of the correlation net
`x1 = ln(0.5 [(1 + zeta)^(4/3) + (1 - zeta)^(4/3)])`; the reduced gradient
`x_s = (1 - e^(-s^2)) ln(s + 1)` (Eq. 9) with `s` from the same density the net
integrates over, no zeta rescaling (that line is xcdiff's, not the paper's); and, for the
meta-GGA nets, the indicator coordinate `ln((alpha + 1)/2)` (Eqs. 10, 12) of the raw
indicator reconstructed from the row (Section 3.1; DFS's indicator carries no smoothing),
as the MLP input rather than the raw clamped indicator (the smoothed column would differ by
up to 3e-11 relative in F_x). The extra descriptors an architecture carries
(cusp, density-matrix, rung 3.5) follow unchanged, and the UEG gates stay on the raw
quantities. The reason is measured (`cnet-diagnosis-report.md`): today's correlation net
passes `r_s` through the reduced-gradient transform, which is cubic at the origin and
folds the high-density cores -- `r_s` in [0.05, 0.5], 44 percent of the correlation
weight -- into two percent of the input axis; `ln r_s` alone improved the correlation fit
2.3 times with exchange bit-identical, while tripling the width, changing the weighting or
the schedule did not. The choice is stated by `ArchitectureConfig.descriptor_coordinates`
(`"legacy"`, today's inputs bitwise, for every recorded architecture; `"dfs"` for v6) and
switched for a run by `model.descriptor_coordinates`. It changes the networks' input
widths and so invalidates no recorded checkpoint by accident: a recorded network is
`"legacy"` and loads only into a `"legacy"` model.

### 3.8 Sequence

PBE first: it unblocks the four GGA groups, which the DN-node maintenance holds until
08/26 08:00 in any case. SCAN second: it unblocks the meta-GGA group. Each step ships with
its oracles and an executing review before the next begins. The parents are implemented in
the repository and validated against libxc; `jax_xc` (libxc's functionals generated for
JAX) was considered and set aside: a new dependency in the cluster environment, for no
gain over an in-repository implementation held to the same oracle.

## 4. Verification (executed, in the test suite)

- V1 parents vs libxc pointwise (PBE and SCAN, x and c) on the model's domain
  (`rho > 1e-10`): PBE at round-off -- `F_x` <= 1e-15 relative, `F_c` <= 1e-13 absolute
  and <= 1e-12 relative where `F_c > 1e-3` (in the far tail `F_c` falls to 1e-12 and the
  relative measure of a quantity at round-off is unbounded: the measured worst is 6.4e-3
  relative at `rho = 1.7e-10`, `s = 1.6e3`, `F_c = 2.2e-12`), and the correlation energy on
  the quadrature <= 1e-15 Ha; SCAN within the stated indicator floors, with the
  raw-indicator reconstruction exact below the ceiling; on the `(rs, s, zeta, alpha)` grid
  of Section 3.1 and on stored molecular grids.
- V2 first derivatives vs libxc `deriv=1`: <= 1e-8 relative where the derivative is above
  round-off (SCAN at the smoothed indicator, stated); the model's `eval_exc` at `gated = 0`
  integrated against the certificate's three parent routes on the certificate's systems:
  <= 1e-8 Ha for PBE, within the indicator floors for SCAN, the tails' 1e-12 Ha included.
- V3 `gated = 0` through the live forward gives `F = F^parent` to round-off for every
  registered architecture at its parent, including rows where the parent sits at a bound
  (the clamp), and `parent = None` gives today's forward bitwise (the closed-shell fixtures
  and the O3 oracle on the unanchored nets, unchanged).
- V4 the fidelity certificate PASSes at initialization for every architecture at
  def2-svp / grid level 1 and at the workflow matrix's identity, from an untrained
  checkpoint written by the pretrain stage's own writer, with `max_atom` and `max_dAE` at
  the oracle floors; it still passes after a 50-step pretrain.
- V5 O1 to O4 (spin scaling, `test_spin_scaling_oracles`) hold on anchored models; O1's
  parent comparison is stated as the identity it becomes.
- V6 the workflow matrix runs an anchored architecture end to end (a 50-step pretrain,
  3-step train cells, evaluation, validation) clean.
- V7 the loader refuses an anchor-state mismatch read from the configuration and the
  manifest; an anchored configuration with a zeta-blind correlation network is refused at
  construction.
- V8 the local pretraining board (`scripts/pretrain_board_local.py`,
  `tests/test_pretrain_board_local.py`, slow-marked): every registered architecture,
  anchored with the DFS coordinates, pretrained on this workstation on one small dataset
  (sto-3g, grid level 3 -- below level 3 the rows of the spatially degenerate free atoms
  O and N are one arbitrary member of the P-term manifold, locked draws at level 1
  differing by of order unity in the iso-orbital indicator, and the generator refuses to
  write a file whose manifest would record an identity it does not have -- polarized, a
  system list with an
  open shell and a closed-shell molecule) for a short schedule, its per-system XC errors
  against the parent at the
  oracle floor at initialization and under the certificate's tolerances afterwards, in
  one table; beside it the unanchored DFS-coordinate fit's losses and errors, pinned to
  their measured values, as the record of what the coordinates alone deliver. Nothing is
  submitted to the cluster before the board is green.

## 5. Decisions (2026-08-25)

Direction from the user: implement the anchor so the campaign can start, then investigate
the correlation network's fit. Taken here, after the physical review: the correction enters in the pre-image of the live bounded map rather than
as a bounded additive term, because the additive form is degenerate wherever the parent
sits at a bound (SCAN exchange at 1.174, correlation at zero) and the pre-image form keeps
today's map, its bounds and the unanchored model unchanged; the correlation parent is
divided by the model's own baseline, polarized in every v6 configuration, and a zeta-blind
anchored correlation network is refused rather than supported; the SCAN parent is evaluated
at the row's indicator (reconstructed raw below the ceiling), the potential taken through
the smoothed quantity; libxc's constants, not the papers' rounded ones.

## 6. Open, recorded

- The correlation network's fit (Section 2) was traced to its density coordinate and is
  answered by Section 3.7; whether the DFS coordinates close the residue on the `s` axis as
  well is read off the board (V8).
- Whether the atomization gate should be stated per atom for reporting. Not needed under
  the anchor.
