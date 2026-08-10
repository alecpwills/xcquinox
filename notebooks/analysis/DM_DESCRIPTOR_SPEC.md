# Density-matrix descriptors: why `dm_entropy` cannot be repaired, and what can replace it

Design spec with its decision record. Sections 1-5 are the study as evaluated; section 6.0
records what was actually shipped, dropped, and removed on 2026-08-06 -- where they differ, 6.0
governs. Every number below was produced by `notebooks/analysis/dm_descriptor_study.py` and is
reproducible by running it.

## 1. The problem

`features.compute_dm_features` exposes three global scalars, tiled to every grid point. Two are
sound (`idempotency_error`, repaired 2026-08-06; `off_diag_norm`, guarded the same day). The third,
`dm_entropy`, has no usable gradient at any converged density: the natural occupations of `DS` are
clipped onto their bounds (every natural occupation sits ON a clip boundary -- at 2.0 or at/below 1e-12 to within round-off (the exact count at 2.0 varies between 2 and 5 with summation order)), so
autodiff returns exactly 0.0 against a finite difference of +5.97e-02. Removing the clip does not
help -- 22 of 23 eigenvalue gaps are below 1e-10 and `eigh` eigenvector derivatives carry
`1/(lambda_i - lambda_j)`, so the gradient is ill-defined at ANY idempotent density matrix.

## 2. An impossibility result that rules out a whole class

The participation ratio `PR = (Tr[DS])^2 / Tr[(DS)^2]` was proposed as an `eigh`-free replacement
and rejected because it equals `N_occ` exactly on the idempotent manifold. That rejection
generalizes, and the generalization is the useful part of this study:

> For a single determinant the eigenvalues of `DS` are exactly `{2, ..., 2, 0, ..., 0}` (RKS).
> Therefore ANY function of the SPECTRUM of `DS` alone is a function of `N_occ` only, hence
> CONSTANT on the idempotent manifold at fixed electron count -- and every converged SCF density
> lies on that manifold.

Measured, at fixed electron count, on systems with genuinely different bonding:

| candidate | H2 @0.74 A | H2 @2.00 A | N2 (14 e) | CO (14 e) |
|---|---|---|---|---|
| `Tr[(DS)^3]/N` | 4.000000 | 4.000000 | 4.000000 | 4.000000 |
| `Tr[(DS)^4]/N` | 8.000000 | 8.000000 | 8.000000 | 8.000000 |
| participation ratio | 1.000000 | 1.000000 | 7.000000 | 7.000000 |

`Tr[(DS)^n]/N` returns `2^(n-1)` for every system, as the result predicts. **This retracts an
earlier suggestion in this project's own planning notes that `Tr[(DS)^n]` at `n >= 3` might serve
as a replacement: it cannot, for the same reason the participation ratio cannot.** N2 and CO are
the sharp test -- same 14 electrons, different bonding, identical value.

A replacement must therefore probe the EIGENVECTORS -- the spatial and bonding structure -- not the
spectrum.

## 3. What DFS actually did

Two distinct things, and conflating them would be a mistake.

**The published functional is semilocal.** The Letter (Dick and Fernandez-Serra, *Phys. Rev. B*
**104**, L161109 (2021)) builds `F_xc` from `(r_s, zeta, alpha, x^2)`. It uses no density-matrix
descriptor. Its SI confirms the pretraining protocol this project mirrors: "For the exchange
functional, we augmented this data by evaluating the enhancement factor on a regular grid in
parameter space (s and alpha)."

**Their code carries a density-matrix branch inherited from NeuralXC.** In the vendored
`ogdpyscf/net.py` the nonlocal model is

```python
def forward(self, dm, ml_ovlp):
    coeff = contract('ij,ijlk->lk', dm, ml_ovlp)
    return self.ml_net(coeff)
```

with rotational invariance restored by a per-shell norm (`Symmetrizer.forward`):

```python
return torch.sqrt(contract('ij,...j', self.M, torch.pow(input, 2)))
```

where `M` is a shell-membership matrix. So the descriptor is: project the density matrix onto a
localized atom-centred auxiliary basis to get coefficients `c_{n l m}` per atom, then contract
within each angular-momentum shell, `d_{n l} = sqrt(sum_m c_{n l m}^2)`. That is the standard
rotationally-invariant contraction. The module imports `neuralxc` directly, placing it in the
lineage of Dick and Fernandez-Serra, *Nat. Commun.* **11**, 3509 (2020).

Two properties matter for us: the coefficients are LINEAR in the density matrix, so the descriptor
is differentiable by construction and needs no eigendecomposition; and it is PER-ATOM, so it never
becomes a molecule-identifying global scalar.

**This project already implements that idea.** `DMRung35Descriptor` contracts the live density
matrix against a fixed localized Gaussian projector to give a bounded per-grid-point occupancy, and
its docstring already states the objection to the `dm_statistics` design: "GLOBAL per-molecule
scalars tiled to every grid point -- a molecule-identity leak". The rung-3.5 descriptor is the
local, leak-free member of the same family.

## 4. Candidate menu

Screened on the three criteria. "Varies" is the N2-vs-CO test at fixed electron count;
"intensive" compares one H2 against two H2 molecules 100 A apart; "gradient" is autodiff against a
central finite difference at a converged H2O density. The gradient column is a per-run noise
floor: its digits move by orders of magnitude between runs with BLAS summation order and thread
count (repeated draws of the spectral row alone spanned 4.4e-13 to 1.7e-10; a single-threaded
re-run gave 2.6e-11 / 3.2e-12 / 6.4e-10 for the last three rows), so the recorded bound is what
every observed draw satisfies, not the digits of any one draw. The candidate values themselves
reproduce bit-for-bit.

| candidate | varies | intensive | gradient | eigh-free |
|---|---|---|---|---|
| `Tr[(DS)^n]/N`, any n | **no** (2^(n-1) always) | yes | <1e-9 | yes |
| participation ratio | **no** (7.000000 both) | **no** (2.000x) | <1e-9 | yes |
| Mayer bond order per atom | yes (1.376927 vs 1.278410) | yes (1.0000) | <1e-9 | yes |
| charge dispersion | yes (0.000000 vs 0.139931) | **0/0 on homonuclear** | <1e-8 | yes |
| interatomic delocalization | yes (0.272999 vs 0.251254) | yes (1.0000) | <1e-8 | yes |

Honest limitation of the screen: the H2 bond-stretch pair discriminates NOTHING -- every candidate
is constant across it, because symmetry pins the bond order of a homonuclear diatomic irrespective
of bond length. Only the N2/CO pair has power here, so the "varies" column rests on a single
contrast and should be widened (isomer pairs, a torsion scan) before anything is adopted.

**Recommended, in order.**

1. **DFS/NeuralXC localized projection (baseline).** `d_{n l} = sqrt(sum_m c_{n l m}^2)` with
   `c = contract(dm, ml_ovlp)`. Linear in the density matrix, per-atom, rotationally invariant,
   already validated in the literature it comes from, and directly comparable against the reference
   implementation on disk. Needs the same zero-argument guard the two repaired features now carry:
   the per-shell norm is a Frobenius norm and is non-differentiable where a whole shell vector
   vanishes.
2. **Mayer bond order per atom.** `B_AB = sum_{mu in A} sum_{nu in B} (PS)_{mu nu} (PS)_{nu mu}`,
   normalized by atom count (Mayer, *Chem. Phys. Lett.* **97**, 270 (1983)). Polynomial in `P`,
   cleanly intensive (1.0000), exact gradient, and it encodes bonding topology that `(rho, sigma,
   alpha)` does not carry. The cheapest genuinely new information in the list.
3. **Interatomic delocalization fraction.** Share of `||PS||_F^2` carried by inter-atomic blocks.
   Same algebraic class as Mayer, a direct covalency measure, intensive by construction.

**Rejected:** every spectral invariant (section 2); charge dispersion as written, since it is `0/0`
for any homonuclear system and identically zero by symmetry for many others.

**Evaluated but not recommended here:** SPAHM (Fabrizio, Briling and Corminboeuf, *Digital
Discovery* **1**, 286 (2022)) builds representations from the EIGENVALUES of guess Hamiltonians,
which reintroduces exactly the degeneracy-differentiability problem that killed `dm_entropy` --
attractive as a representation, wrong for a quantity that must be differentiated through an SCF.
The rotationally-invariant density representation of Margraf and Reuter (*Nat. Commun.* **12**, 344
(2021)) uses a SOAP-style power spectrum of the density and is differentiable, but it is a
substantially larger construction than the projection already available in the rung-3.5 path, and
would duplicate it.

## 5. Open questions before implementation

- Whether a global density-matrix scalar is wanted at all, given that the rung-3.5 descriptor
  already supplies local density-matrix information leak-free. The strongest reading of section 3
  is that `dm_entropy` should be removed and the family extended locally instead.
- Any replacement changes the descriptor definition and therefore invalidates checkpoints trained
  on the current values (`descriptors.py` records this as requiring sign-off). Affected on disk:
  the `deep_dm` / `deep_combined` families under `notebooks/checkpoints_v3b/`.
- The "varies" screen needs more than the single N2/CO contrast before adoption.

## 6.0 Decision record (2026-08-06 -- outcomes of the sections below)

The plan below was evaluated by execution before implementation; the outcomes
supersede the recommendations where they differ.

* `rung35_multishell` SHIPPED, as the radial (`l = 0`) generalization only
  (registry name `rung35_multishell`, architecture `deep_rung35ms_3x16`).
  Verified bitwise-identical to the single-width descriptor at
  `alphas=(0.2,)`, alpha-major column order, bounded, leak-free under the
  size-consistency test, padding-neutral, and at the energy/potential
  finite-difference floor on both spin paths.
* `mayer_bond_order` and `interatomic_delocalization` DROPPED, not shipped
  behind flags: measurement disqualified the design outright. Both are
  per-system averages and fail size consistency on every non-identical
  fragment pair (Mayer mean scaled by 0.6928 for CO+H2, 0.9099 for H2O+H2,
  1.2242 for a three-fragment composite), both are identically zero on every
  single-atom system (a molecule-versus-atom label, since atoms are the
  atomization references), and the atom-slice implementation cannot run under
  the training jit. A default-OFF flag does not make a non-size-consistent
  global descriptor correct; it postpones the failure. The salvageable pieces
  (a traceable, padding-neutral same-atom mask formulation; the published-form
  factor-of-two note) are recorded in `xcquinox/alec/DEFERRED_WORK.md` item 1.
* `dm_entropy` REMOVED (`dm_statistics` width 3 -> 2), per section 2's
  impossibility argument; the removal also repaired the `dm_statistics`
  architectures' energy/potential consistency (1.04e-02 -> 2.05e-10 under the
  committed test's parametrized ordering; 5.2e-03 in a fresh process), since
  the dead gradient had been dominating the residual.
* The discrimination screen (section 4's "varies" column) proved NECESSARY BUT
  NOT SUFFICIENT as a gate: it cannot detect the size-consistency failure that
  disqualified the globals. Size consistency on non-identical fragment pairs
  is the binding acceptance test for any global descriptor; the
  identical-fragment case cannot fail it.

## 6. Implementation plan

### 6.1 Shape, and why the DFS descriptor must be localized to grid points

NeuralXC projects the density matrix onto an ATOM-centred auxiliary basis and feeds the per-atom
coefficients to a per-atom energy model, which is then summed. This project's architecture is
different: the networks emit a per-grid-point enhancement factor, and every descriptor is a column
evaluated at each grid point. Porting the atom-centred form directly would force a choice between
tiling a molecular aggregate to every grid point (the leak `DMRung35Descriptor` was created to
avoid) and inventing a grid-point-to-atom assignment.

The faithful port instead moves the projector to the grid point, which is what the rung-3.5
descriptor already does with a single s-type Gaussian. The DFS/NeuralXC construction is then
recovered by generalizing the projector set:

    single s-type projector, 1 occupancy per spin        <- shipped rung-3.5
    n_radial x l_max projector set, per-shell invariants <- DFS/NeuralXC form

with the same invariant contraction DFS use, `d_{n l}^sigma = sqrt(sum_m (c_{n l m}^sigma)^2)`,
where `c^sigma = A^T P^sigma A` per channel. The shipped descriptor is exactly the
`n_radial = 1, l_max = 0` member, so the generalization is backward compatible by construction and
that must be asserted, not assumed.

### 6.2 Descriptors to add

All three are SPIN-RESOLVED. Every feature is emitted per spin channel (alpha, beta), matching the
existing rung-3.5 convention, so the descriptor carries polarization rather than averaging it away.
For RKS the two channels are equal by construction and the closed-shell reduction must be pinned by
a test.

| name | shape | definition | source |
|---|---|---|---|
| `rung35_multishell` | per grid point, per spin, per (n, l) | `sqrt(sum_m (A_{nlm}^T P^sigma A_{nlm})^2)` | NeuralXC / DFS; Janesko rung-3.5 |
| `mayer_bond_order` | global, per spin | `sum_{A<B} sum_{mu in A, nu in B} (P^sigma S)_{mu nu} (P^sigma S)_{nu mu} / n_atoms` | Mayer 1983 |
| `interatomic_delocalization` | global, per spin | inter-atomic share of `||P^sigma S||_F^2` | this work; same algebraic class |

`dm_entropy` is removed in the same change. Its stated purpose -- a correlation indicator -- is
already served by `idempotency_error`, which is the quantity that actually vanishes for a single
determinant; the existing docstring says so.

Standing caveat for the two GLOBAL descriptors: intensivity (verified, ratio 1.0000 on H2 versus
two H2 100 A apart) makes them defensible where the size-extensive `dm_entropy` was not, but a
molecular scalar broadcast to every grid point remains a weaker form of the same objection. They
ship behind a flag, default OFF, and are A/B tested rather than assumed safe. `rung35_multishell`
carries no such caveat because it is local.

### 6.3 Flags

- Registry entries `rung35_multishell`, `mayer_bond_order`, `interatomic_delocalization` via the
  existing `@register_descriptor` mechanism, so they are selectable through
  `ArchitectureConfig` `FeatureSpec` names with no new plumbing.
- `rung35_multishell` takes `n_radial` and `l_max` as static kwargs; `(1, 0)` must reproduce the
  shipped `rung35` column byte-for-byte.
- `DMStatisticsDescriptor` gains an explicit feature-selection tuple so `dm_entropy` can be dropped
  without changing the other two columns, and so the removal is visible in the spec rather than
  implicit.
- Sweep exposure: new architecture entries in the registry (`deep_rung35ms_3x16`,
  `deep_mayer_3x16`, ...) selectable from the sweep YAML `arch` axis. No change to the running
  array; new cells only.
- Every new descriptor is default OFF. No existing architecture changes shape.

### 6.4 Tests -- NaN is the acceptance criterion, not an afterthought

Today's failures were all NaN or gradient defects that a value-only test could not see, so the
suite is built around them. For EVERY new descriptor, parametrized over RKS and polarized UKS:

1. **Value finite** on the degenerate cases that actually broke things: `nao == 1` (H, He in
   sto-3g -- the zero off-diagonal block that produced a NaN Fock this session), a fully
   spin-polarized system with an EMPTY beta channel (H atom, spin=1), a homonuclear diatomic (zero
   charge dispersion, the `0/0` that disqualified that candidate), and a diffuse basis where the
   density tail underflows.
2. **Gradient finite**: `jax.grad` of the feature w.r.t. the density matrix, on all of the above.
   A finite value with a NaN gradient is the failure mode that hid in `off_diag_norm`.
3. **Assembled V_xc finite**: the feature must survive `feature_response_vxc`, since that is the
   path that converts a latent feature-map defect into a NaN Fock matrix.
4. **Training-step gradient finite AND correct**: `d/d(parameters)` of the assembled V_xc against a
   finite difference. This is the check that caught the `stop_gradient` defect, which was invisible
   in the forward value and in the density-matrix gradient alike.
5. **Every norm guarded at its zero argument.** Any `sqrt` or Frobenius norm introduced must use
   the masked-square-root pattern now in `features.compute_dm_features`, and a test must exercise
   the exact zero-argument input rather than a near-zero one.
6. **Closed-shell reduction**: alpha and beta columns identical for an RKS density.
7. **Size consistency**: the descriptor of a molecule must be unchanged by the presence of a
   distant copy -- the test that certified the rung-3.5 descriptor leak-free, applied to each new
   one. This is what makes the intensivity claim a property of the code and not of a spreadsheet.
8. **Backward compatibility**: `rung35_multishell(n_radial=1, l_max=0)` equals the shipped
   `rung35` column.
9. **Discrimination**: the descriptor must actually differ between N2 and CO at fixed electron
   count. Without this a descriptor could pass every finiteness test while being constant and
   useless -- which is precisely how `dm_entropy` and the participation ratio failed.

### 6.5 Sequencing

1. Widen the section-4 screen beyond the single N2/CO contrast (isomer pairs, a torsion scan)
   before committing to `mayer_bond_order` / `interatomic_delocalization`.
2. Implement `rung35_multishell` first: it is local, leak-free, backward compatible, and needs no
   new justification beyond the sources already in the repo.
3. Implement the two global descriptors behind their flag, with the size-consistency test as the
   gate.
4. Remove `dm_entropy`. Checkpoint-invalidating for the `deep_dm` / `deep_combined` families under
   `notebooks/checkpoints_v3b/`; requires sign-off per `descriptors.py`.

## Sources

- I. Mayer, "Charge, bond order and valence in the ab initio SCF theory", *Chem. Phys. Lett.*
  **97**, 270 (1983).
- S. Dick and M. Fernandez-Serra, "Machine learning accurate exchange and correlation functionals
  of the electronic density", *Nat. Commun.* **11**, 3509 (2020).
- S. Dick and M. Fernandez-Serra, "Highly accurate and constrained density functional obtained with
  differentiable programming", *Phys. Rev. B* **104**, L161109 (2021), and its SI.
- A. Fabrizio, K. R. Briling and C. Corminboeuf, "SPAHM: the spectrum of approximated Hamiltonian
  matrices representations", *Digital Discovery* **1**, 286 (2022).
- J. T. Margraf and K. Reuter, "Pure non-local machine-learned density functional theory for
  electron correlation", *Nat. Commun.* **12**, 344 (2021).
- B. G. Janesko, arXiv:2206.07118 (rung-3.5), and Verma et al., *J. Chem. Theory Comput.* **15**,
  4804 (2019) (M11plus) -- the basis of the existing local descriptor.
