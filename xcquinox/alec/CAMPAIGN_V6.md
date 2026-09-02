# Campaign v6 -- design

Scope: the descriptors every v6 network consumes, the material they are pretrained and
trained on, and the ordered cell groups the campaign is submitted in. Every quantity below
is either quoted from the code at a stated `file:line` or produced by one of the checks
recorded in Section 4.

v6 exists because the pretrained networks of v4gga, v5 and v5mgga2 did not reproduce their
parent functional: measured atomization-energy offsets ran from 2.3-4.2 kcal/mol on the
descriptor-free architectures to 13.2-56.1 kcal/mol on the descriptor-carrying ones
(`SPEC_pretrain_fidelity_program.md` Section 2), which is larger than the architecture
differences the campaign exists to resolve. Three things change in the method: the
open-shell exchange footing (Section 1.2), the pretraining set and objective (Section 2.1
and 2.2), and an enforced per-architecture fidelity certificate (Section 2.3). Those runs
are retired as quantitative results and remain as the documented failure record.


## 1. Descriptors

### 1.0 What feeds a network

Both networks are per-grid-point multilayer perceptrons whose output is an enhancement
factor multiplying a uniform-electron-gas energy density
(`models.py:171-172`):

    ex_density = rho_safe * ex_lda * Fx
    ec_density = rho_safe * ec_base * Fc

The exchange network's input width is `in_size = 1 + n_extra_features`
(`networks.py:119`): one semilocal column plus the concatenated descriptor block. The
correlation network's is `in_size = 2 + (1 if use_spin_polarization else 0) +
n_extra_features` (`networks.py:291`): two semilocal columns, the spin-polarization column
when the architecture opts in, then the same descriptor block. The descriptor block is
assembled left to right in declaration order by
`descriptors.assemble_descriptor_features` (`descriptors.py:475-494`), so a descriptor is
consumed identically by exchange and correlation -- complete X/C parity, with the one
exception recorded in Section 1.2.

The v6 configurations set `use_polarized_correlation: true` at run level
(`hpcjobs/configs/dfs_step7.dfs6311_grid3_v6.yaml`, key `use_polarized_correlation`), so
every correlation network in this campaign carries the spin-polarization column.

Five descriptors are registered (executed check C7): `cusp` (2 features, geometry-only),
`dm_statistics` (2, density-matrix dependent), `rung35` (2, density-matrix dependent),
`rung35_multishell` (6, density-matrix dependent), `metagga` (1, density-matrix
dependent).

### 1.1 The base semilocal inputs, and the transform `notransform` ablates

**Definition as computed.** Exchange takes the PBE reduced density gradient
(`networks.py:153-154`):

    k_F = (3 * jnp.pi**2 * rho) ** (1 / 3)
    s = jnp.sqrt(sigma) / (2 * k_F * rho)

Correlation takes the Wigner-Seitz radius beside the same `s` (`networks.py:323-325`):

    rs = (3 / (4 * jnp.pi * rho)) ** (1 / 3)
    k_F = (3 * jnp.pi**2 * rho) ** (1 / 3)
    s = jnp.sqrt(sigma) / (2 * k_F * rho)

With `descriptor_log_transform = True` the MLP receives compressed forms rather than the
raw variables. Exchange (`networks.py:161-164`):

    if self.descriptor_log_transform:
        s_mlp = (1.0 - jnp.exp(-s * s)) * jnp.log(s + 1.0)
    else:
        s_mlp = s

Correlation applies the same map to both of its columns (`networks.py:339-344`):

    if self.descriptor_log_transform:
        rs_mlp = (1.0 - jnp.exp(-rs * rs)) * jnp.log(rs + 1.0)
        s_mlp = (1.0 - jnp.exp(-s * s)) * jnp.log(s + 1.0)
    else:
        rs_mlp = rs
        s_mlp = s

The flag also selects the cusp descriptor's column-1 compression, injected at
`config.py:320-321`.

Independently of the flag, the uniform-gas recovery gate multiplying the MLP output is
always built from the RAW `s` on the GGA rung (`networks.py:187`):

    tanhterm = jnp.tanh(s) ** 2

so `F -> 1` as `s -> 0` whatever the input transform is. The transform is a feature
conditioning choice; the gate is a structural constraint.

The spin-polarization column, present on every v6 correlation network, is the bounded
Dick and Fernandez-Serra form (`networks.py:351-354`):

    zeta_c = jnp.clip(zeta, -1.0, 1.0)
    x1 = jnp.atleast_1d(
        0.5 * ((1.0 + zeta_c) ** (4 / 3) + (1.0 - zeta_c) ** (4 / 3))
    ).flatten()

which equals 1 at `zeta = 0`, recovering the unpolarized input.

**Physical justification.** `s` is the dimensionless measure of density inhomogeneity on
which every GGA exchange functional is built, and `rs` is the density variable of the
uniform-gas correlation energy. The compression exists because `s` is unbounded on
molecular grids: the log form keeps the first-layer activation in a usable range without
altering the gate, and the same argument motivates the `/5` scaling inside the cusp
descriptor (`features.py:294-303`). The `x1` form is bounded on `zeta` in `[-1, 1]` and
reduces to the unpolarized input at `zeta = 0`, so a closed-shell call is unchanged by
turning polarization on.

**Source.** Dick and Fernandez-Serra, Phys. Rev. B 104, L161109 (2021) -- the transform is
that Letter's reduced-gradient form (Eq. 9) and the polarization feature its Eq. 4, both
cited at `networks.py:243-246` and `networks.py:284-288`. Two deviations are documented in
the code and are not changed here: the correlation density feature is `rs` passed through
the reduced-gradient transform, whereas the Letter's Eq. 7 applies a plain log to
`x0 = n^(1/3)` (`networks.py:330-338`); and on the meta-GGA rung the MLP receives the raw
clamped alpha rather than the Letter's `x3` (Section 1.9).

**Where it enters.** Both. In pretraining the columns are `rho_all` / `sigma_all` for the
total-density block and `rho_x` / `sigma_x` for the per-channel exchange block
(`pretrain.py:890` and the schema at `pretrain_data_gen.py:1490-1533`); in training they
are the density and gradient invariant of the live SCF density.

**What `notransform` ablates.** The four `deep_notransform*` architectures are the
`descriptor_log_transform = False` twins of their transformed siblings and carry no
descriptors at all (executed check C4); the ablation is therefore exactly the substitution
of raw `s` for `s_mlp` in exchange and raw `(rs, s)` in correlation. Those four
architectures are NOT in the v6 sweep (Section 3.5).

### 1.2 The spin footing: doubled spin-channel density for exchange, total density for correlation

This is not a descriptor but the footing every density-matrix descriptor is evaluated on,
and it is the correction v6 is built around.

Exact exchange spin scaling, `E_x[n_a, n_b] = (E_x[2 n_a] + E_x[2 n_b]) / 2` (Oliver and
Perdew, Phys. Rev. A 20, 397 (1979)), refers each spin channel to the fictitious
spin-unpolarized system whose two spin blocks both hold `P_sigma`. That system is built by
`descriptors.doubled_spin_dm` (`descriptors.py:193-222`) and has total density
`2 rho_sigma`, gradient invariant `4 sigma_sigma_sigma` and kinetic-energy density
`2 tau_sigma`. Each descriptor's per-channel block is returned by
`Descriptor.compute_for_spin_channel` (`descriptors.py:119-148`), which reads the
per-channel precompute keys (`*_features_a` / `*_features_b`) for a descriptor declaring
`density_matrix_dependent = True` and falls through to the shared block otherwise.

Correlation is spin-interpolated rather than spin-scaled (von Barth and Hedin, J. Phys. C
5, 1629 (1972); Perdew and Wang, Phys. Rev. B 45, 13244 (1992), both cited at
`models.py:147-150`), so it stays on the total density with `zeta`, and the correlation
network never reads the per-channel block (refused at `pretrain.py:886-890`).

Before v6 the UKS exchange doubled `rho` and quadrupled `sigma` per channel but evaluated
every descriptor at the physical density and passed the same block into both channels
(`SPEC_pretrain_fidelity_program.md` Section 2, defect D1). Every descriptor entry below
therefore states which density its features are taken on.

### 1.3 `cusp` -- nuclear-cusp proximity

**Definition as computed.** Two columns. Column 0 (`features.py:268`):

    cusp_factor = jnp.exp(-2 * Z_nearest * r_min)

with `r_min` the distance to the nearest nucleus and `Z_nearest` its charge. Column 1 from
the Coulomb-like weighted sum (`features.py:271`):

    weighted_Z_sum = jnp.sum(nuclear_charges[None, :] / distances, axis=1)

compressed and bounded (`features.py:328-334`):

    if log_transform:
        log_weighted_Z = jnp.log(features['weighted_Z_sum'] + 1e-12)
        weighted_Z_bounded = jnp.tanh(log_weighted_Z / 5.0)
    else:
        weighted_Z_bounded = jnp.tanh(features['weighted_Z_sum'] / 5.0)

    return jnp.stack([features['cusp_factor'], weighted_Z_bounded], axis=1)

Column 0 lies in `[0, 1]`, column 1 in `(-1, 1)`.

**Physical justification.** Kato's cusp condition fixes the wavefunction slope at a
nucleus; the corresponding spherically averaged density relation is
`(d<rho>/dr)|_{r=0} = -2 Z rho(0)`, so the density envelope near a nucleus decays as
`exp(-2 Z r)`. Column 0 is that envelope used as a proximity feature -- a heuristic, not an
enforced condition, and the docstring says so (`descriptors.py:230-240`). Column 1 supplies
the network with a smooth nuclear-attraction-like weight; the `/5` scaling is anchored to a
measured failure: the raw `log_weighted_Z` spans about 14 units on physical grids and
saturated `F_x` at about 1.4 on the cusp-carrying architectures (`features.py:294-303`).

**Source.** Kato, Commun. Pure Appl. Math. 10, 151 (1957) for the wavefunction cusp;
Steiner, J. Chem. Phys. 39, 2365 (1963) for the density-form relation; both carried in the
descriptor docstring at `descriptors.py:233-237`. The `log_transform` branch is labelled
the Dick and Fernandez-Serra XCDiff convention (`descriptors.py:241-245`).

**Where it enters, and on which density.** Both pretraining and training. It is the ONLY
geometry-only descriptor: `density_matrix_dependent = False` and `spin_mol_keys = ()`
(executed check C7), so its per-channel exchange block is the shared block and it is
unchanged by the doubled-spin construction (`descriptors.py:126-130`;
`SPEC_pretrain_fidelity_program.md` Section 3.1, "Cusp -> unchanged"). Pretraining columns:
`cusp_all` and `cusp_x`, both computed with `log_transform=True` to match training
(`pretrain_data_gen.py:817-825`). Training key: `mol_data["cusp_features"]`
(`descriptors.py:255-258`).

### 1.4 `rung35` -- localized density-matrix occupancy

**Definition as computed.** A normalized Gaussian projector
`phi^G_{r_m}(r) = (2 alpha / pi)^(3/4) exp(-alpha |r - r_m|^2)` is placed at each grid
point, and the projected-AO overlap `A_mu(r_m) = <chi_mu | phi^G_{r_m}>` is built once as a
plain PySCF overlap integral (`rung35.py:42-93`). The feature is the contraction of the
one-particle density matrix against that projector (`rung35.py:122`):

    return jnp.einsum("gm,smn,gn->gs", A, dm_spin, A)    # (N, 2)

that is, `n_sigma(r) = A(r)^T P^sigma A(r)`, two columns (alpha and beta occupancy). The
projector width defaults to `DEFAULT_RUNG35_ALPHA = 0.2` a0^-2 (`rung35.py:39`, executed
check C7).

**Physical justification.** This is a genuine per-grid-point contraction of the NON-LOCAL
Kohn-Sham one-particle density matrix, so it carries information no semilocal ingredient
does, while remaining size-intensive: adding a distant fragment does not change the
occupancy near a point, which is precisely the property `dm_statistics` lacks (Section 1.7).
It is bounded in `[0, 1]` by Bessel's inequality -- `P^sigma` is positive semidefinite so
the value is non-negative, and orthonormal occupied orbitals against a normalized projector
bound it above -- hence NaN-safe by construction (`rung35.py:28-30`). Because `A` is
density-independent, the occupancy is LINEAR in the live density matrix: differentiable
through the SCF, no eigendecomposition, no degeneracy hazard, and self-consistent under the
`REASSEMBLE` feature policy the v6 solver sets. It carries no kinetic-energy density, so it
is not a meta-GGA; it is its own rung between meta-GGA and hybrid.

**Source.** Janesko, arXiv:2206.07118, Eq. 12-13, and M11plus (Verma et al., J. Chem.
Theory Comput. 15, 4804 (2019)), both cited in the module docstring (`rung35.py:3-8`) and
the descriptor docstring (`descriptors.py:321-324`). The default width is stated as
grounded at the M11plus rung-3.5 kernel scale `d^2 = 5 a0^2` (`rung35.py:37-38`). The
arXiv identifier and the M11plus reference are carried by the repository itself; the
literature attribution is to be confirmed against the library copy.

**Where it enters, and on which density.** Both. `density_matrix_dependent = True`, so the
exchange channels read the doubled-spin block: the channel occupancy appears in both spin
slots as `[n_sigma, n_sigma]`, still inside the Bessel bound (`descriptors.py:203-205`).
Pretraining columns: `rung35_all` (total density) and `rung35_x` (per channel, from
`doubled_spin_dm`, `pretrain_data_gen.py:470-472`). Training keys:
`mol_data["rung35_features"]` and the per-channel `rung35_features_a` /
`rung35_features_b` (executed check C7); the live-DM kernel is
`DMRung35Descriptor.compute_from_dm` (`descriptors.py:356-364`).

### 1.5 `rung35_multishell` -- the radial generalization

**Definition as computed.** The same contraction at several projector widths
(`rung35.py:195-196`):

    occ = jnp.einsum("agm,smn,agn->ags", A, dm_spin, A)
    return jnp.transpose(occ, (1, 0, 2)).reshape(A.shape[1], -1)

giving `n_sigma(r; w) = A_w(r)^T P^sigma A_w(r)` for each width, with column order
ALPHA-MAJOR then spin. Widths default to
`DEFAULT_RUNG35_MULTISHELL_ALPHAS = (0.05, 0.2, 0.8)` (`rung35.py:130`, executed check C7),
spanning the M11plus kernel scale by a factor of about four either side, so
`n_features = 2 * 3 = 6` (executed check C7). Setting `alphas` to a single width reproduces
the single-width descriptor bitwise (`descriptors.py:392-393`).

**Physical justification.** A single projector width probes the density matrix at one
length scale; the set gives a coarse RADIAL profile of the one-particle density matrix
around each grid point. All the properties of Section 1.4 carry over unchanged -- linear in
the density matrix, bounded by the same Bessel argument, differentiable through the SCF.

**Source.** The construction is the radial part of the localized density-matrix projection
used by NeuralXC (Dick and Fernandez-Serra, Nat. Commun. 11, 3509 (2020)) and carried in
the DFS reference implementation, which projects the density matrix onto a localized basis
and contracts the coefficients into rotationally invariant per-shell norms
(`rung35.py:159-168`; `descriptors.py:375-381`). A LIMITATION is recorded at both sites and
is repeated here because the name invites the stronger claim: `fakemol_for_charges` builds
s-type projectors only, so this is the `l = 0` channel; with one `m` per shell the
invariant `sqrt(sum_m c_{nlm}^2)` collapses to the occupancy itself. Angular channels
require solid-harmonic fakemols and are not implemented, so this must not be described as
"the DFS descriptor" (`descriptors.py:383-389`).

**Where it enters, and on which density.** Both, on the same footing as `rung35`:
`density_matrix_dependent = True`, per-channel keys `rung35ms_features_a` /
`rung35ms_features_b`, shared key `rung35ms_features` (executed check C7). Pretraining
columns `rung35ms_all` and `rung35ms_x` (`pretrain_data_gen.py:473-477`).

### 1.6 The rung-3.5-only form

`deep_rung35only_3x16` carries `descriptors = ('rung35',)` alone, without the cusp columns
its siblings pair it with (executed check C4). It exists to separate the localized
occupancy's contribution from the cusp feature's within the rung-3.5 families. It is a
registry architecture, not a distinct descriptor family, and it is NOT in the v6 sweep
(Section 3.5).

### 1.7 `dm_statistics` -- density-matrix correlation indicators, in the fixed bounded form

**Definition as computed.** Two columns. Column 0, the squared Frobenius departure from
single-determinant idempotency, normalized by the electron count (`features.py:109-111`):

    def _idempotency_sq(d, n):
        x = d @ S @ d - d
        return jnp.sum(x * x) / (n + 1e-12)

evaluated per spin channel and averaged for a spin-resolved density matrix
(`features.py:117-118`), or on `D/2` for a closed shell (`features.py:125-127`). Column 1,
the off-diagonal Frobenius norm normalized by the trace, with a masked square root so the
derivative at an identically zero off-diagonal block is defined as 0 rather than `0/0`
(`features.py:174-178`):

    off_diag_nonzero = off_diag_sq > 0.0
    safe_off_diag_sq = jnp.where(off_diag_nonzero, off_diag_sq, 1.0)
    off_diag_norm = jnp.where(
        off_diag_nonzero, jnp.sqrt(safe_off_diag_sq), 0.0
    ) / (jnp.trace(dm) + 1e-12)

The two are packed in that order (`features.py:221-224`) and tiled to every grid point
(`descriptors.py:312-313`).

**WHAT WAS REMOVED, AND WHY.** A third column, `dm_entropy`, was deleted on 2026-08-06,
taking the descriptor width from 3 to 2 (`descriptors.py:275-286`; HISTORY entry
2026-08-06). Two reasons, both measured. First, it was the size-extensive label leak the
2026-05-29 forensic review identified: in its original form the entropy behaved as
`ln(N_occ)`, encoding molecule identity and size, and broadcasting it to every grid point
gave the network a memorization handle on a tiny training pool (HISTORY 2026-05-29
`4def50d14`, and the 2026-06-28 diagnosis of the `deep_combined*` held-out collapse). The
2026-05-29 `dm_entropy_intensive` toggle fixed the channel's MAGNITUDE but not its
LOCALITY. Second, it had no usable gradient at any converged density: the physical-bounds
clip put every natural occupation on a boundary, so autodiff returned exactly 0.0 against a
finite difference of +5.97e-02, and removing the clip was worse because `eigh` eigenvector
derivatives carry `1/(lam_i - lam_j)` on an occupation spectrum that is degenerate for any
idempotent density matrix. An entire class is ruled out with it: for a single determinant
the eigenvalues of `DS` are exactly `{2,...,2,0,...,0}`, so any function of the spectrum
alone is constant on the idempotent manifold. Removing it also took the `dm_statistics`
architectures' energy/potential finite-difference residual from 1.04e-02 to 2.1e-10 under
the committed test's own parametrized ordering. The screening of candidate replacements is
in `notebooks/analysis/DM_DESCRIPTOR_SPEC.md`. The `ArchitectureConfig.dm_entropy_intensive`
field is kept but inert, because the live cluster array's pickled spec files carry it
(`config.py:315-319`).

**Physical justification of what remains.** Both surviving columns grow with departure from
a single Slater determinant, so they are correlation indicators: about zero for a
Hartree-Fock or Kohn-Sham reference, growing as the density matrix acquires
non-idempotency and off-diagonal weight.

**CAVEAT, unchanged and still open.** These are GLOBAL, molecule-level scalars tiled
identically to every grid point and fed into a per-point (semilocal) enhancement factor, so
the XC energy density at a point in fragment A shifts when a distant fragment B is added
(`descriptors.py:288-296`). `rung35` and `rung35_multishell` are the leak-free members of
this family. Making the global form defensible requires an architecture change and is
recorded in `DEFERRED_WORK.md`.

**Source.** The idempotency and off-diagonal indicators are stated in the code rather than
attributed to a paper; the source cited here is the code construct itself
(`features.py:66-184`, `descriptors.py:261-316`), and any literature attribution is to be
confirmed against the library copy. The removal rationale and its measurements are in
HISTORY 2026-08-06 and `notebooks/analysis/DM_DESCRIPTOR_SPEC.md`.

**Where it enters, and on which density.** Both. `density_matrix_dependent = True`
(executed check C7), so the exchange channels read the statistics of
`diag(P_sigma, P_sigma)` (`descriptors.py:205-206`; built at
`pretrain_data_gen.py:467-469`). Pretraining columns `dm_all` and `dm_x`; training keys
`dm_features` and `dm_features_a` / `dm_features_b`. A width gate refuses a stale
pretraining file carrying the old 3-column layout (`pretrain.py:914-928`).

### 1.8 The combined form

`deep_combined*` carries `descriptors = ('dm_statistics', 'cusp')`, four extra columns
(executed check C4). It is the pairing the improvement history records as swept since v3 (the shipped v3 configuration itself predates the record) and is retained so
the DM indicators are measured beside the geometry feature rather than alone. Column order
follows declaration order, `dm` before `cusp`, which is pinned by test
(`tests/test_descriptors.py:70-73`).

### 1.9 `metagga` -- the iso-orbital indicator alpha

**Definition as computed.** The total positive kinetic-energy density is a linear
contraction of the live density matrix against the AO gradients already on the grid
(`metagga.py:139`):

    return 0.5 * jnp.einsum("dgi,ij,dgj->g", ag, p_total, ag)

The indicator's ingredients (`metagga.py:239-241`):

    rho_safe = jnp.maximum(rho, _RHO_FLOOR)
    tau_w = sigma / (8.0 * rho_safe)
    tau_unif = (3.0 / 10.0) * (3.0 * jnp.pi**2) ** (2.0 / 3.0) * rho_safe ** (5.0 / 3.0)

and the returned column (`metagga.py:272-274`):

    alpha_raw = (tau - tau_w) / jnp.maximum(tau_unif, _RHO_FLOOR)
    return jnp.minimum(smooth_positive_part(alpha_raw, _ALPHA_SMOOTHING_WIDTH),
                       _ALPHA_MAX)

with the smooth positive part (`metagga.py:152`):

    return 0.5 * (x + jnp.sqrt(x * x + width * width))

at `_ALPHA_SMOOTHING_WIDTH = 1e-5` (`metagga.py:104`) and `_ALPHA_MAX = 100.0`
(`metagga.py:61`). The identity string recorded in every pretraining-data manifest is built
from the width so the two cannot drift apart (`metagga.py:112`):

    ALPHA_DEFINITION: str = f"smooth_positive_part:width={_ALPHA_SMOOTHING_WIDTH:.0e}"

which evaluates to `'smooth_positive_part:width=1e-05'` (executed check C7). A file whose
alpha rows were written under another definition is stale for a run at this one, exactly as
a file built at another basis or orientation lock is.

**Why a smooth positive part rather than a clip.** The lower bound `alpha >= 0` is the von
Weizsacker inequality, exact on every positive semidefinite density matrix, so a negative
raw value is rounding. A hard clip `max(alpha_raw, 0)` made the derivative one-sided
exactly where `tau = tau_W` identically -- a one-orbital spin channel -- and autodiff
returned whichever side the rounding selected: Li's beta-channel feature-response term moved
by 0.93 Ha under a 1e-14 relative change of the density matrix, and H's by 4.2e-3 Ha. With
the smooth form the same probe moves H's by 3.6e-12 Ha and the H atom's Fock pair
reproduces a central difference of the energy to 6.2e-10 relative
(`metagga.py:186-200`). The width is anchored, not tuned: the rounding residue of the raw
indicator on a one-orbital channel is at most 6.6e-10 wherever `2 rho_sigma > 1e-8` across
three basis/grid identities; the largest change of the SCAN exchange energy the smoothing
induces is +1.17e-7 Ha (H) and +3.1e-7 Ha (Li beta channel), linear in the width; and the
certificate's atomic tolerance is 1.0 mHa, 8.5e3 above that
(`metagga.py:64-104`, `metagga.py:217-233`). `smooth_positive_part(0, 1e-5) = 5e-06`,
exactly `width/2` (executed check C7).

**Physical justification.** `alpha = 1` for the uniform electron gas (`tau_W = 0`,
`tau = tau_unif`), `alpha = 0` for a single orbital (`tau = tau_W`); it therefore
distinguishes iso-orbital, slowly varying and overlap regions, which no GGA ingredient can.
It is a rung-3 ingredient obtained without new integrals, without the Laplacian and without
`deriv=2`, because the AO gradients are already computed for the reduced gradient `s`
(`metagga.py:16-22`). Being linear in the live density matrix, it is self-consistent and
differentiable through the SCF, exactly like the rung-3.5 occupancy.

**Source.** Introduced by SCAN: Sun, Ruzsinszky and Perdew, Phys. Rev. Lett. 115, 036402
(2015), Eq. 2; reused by DFS: Dick and Fernandez-Serra, Phys. Rev. B 104, L161109 (2021),
Eq. 6. Both cited at `metagga.py:3-6` and `descriptors.py:434-436`.

**What the flag does beyond supplying a column.** For an architecture with
`meta_gga = True` the alpha column additionally drives the DFS uniform-gas recovery gate in
place of `tanh(s)^2` (`networks.py:183-185`):

    x2 = (1.0 - jnp.exp(-s * s)) * jnp.log(s + 1.0)
    x3 = jnp.log((alpha + 1.0) / 2.0)
    tanhterm = jnp.atleast_1d(x2).flatten() + jnp.tanh(x3) ** 2

the same expression in the correlation network (`networks.py:377-379`), and the exchange
Lieb-Oxford ceiling moves from the PBE value 1.804 to the DFS value 1.174
(`networks.py:461`). Two deviations are documented and not changed: the MLP receives the
RAW clamped alpha where DFS feeds the log-transformed `x3` (`networks.py:176-181`), and the
correlation gate reuses the exchange form where DFS applies `tanh` to `x2`
(`networks.py:367-375`). `ArchitectureConfig.from_spec` refuses `meta_gga=True` without the
`metagga` descriptor (`config.py:383-388`), and `is_meta_gga` reads the descriptor list as
the ONE rung predicate (`config.py:251-276`) -- the two statements used to be asked
separately, and an architecture carrying the descriptor without the flag was fitted to PBE
targets on the SCAN density, measured 24.0 mHa per system off its parent.

**Where it enters, and on which density.** Both. `density_matrix_dependent = True`, and the
per-channel exchange block is `alpha(2 rho_sigma, 4 sigma_sigma, 2 tau_sigma)`
(`descriptors.py:202-204`; built at `pretrain_data_gen.py:454-456`). Pretraining columns
`metagga_all`, `metagga_x` and `metagga_mesh`; training keys `metagga_features` and
`metagga_features_a` / `metagga_features_b` (executed check C7). The live-DM kernel is
`MetaGGAAlphaDescriptor.compute_from_dm` (`descriptors.py:461-469`).

### 1.10 Attention -- an architecture change, not a descriptor

The attention variants add NO input column. `n_extra_features` is identical between an
attention architecture and its non-attention twin (executed check C1: `deep_attn_3x16` and
`deep_3x16` both report `n_extra=0`, `n_input=2`; `deep_rung35_attn_3x16` and
`deep_rung35_3x16` both report `n_extra=4`, `n_input=6`). What changes is the forward pass:
a scaled dot-product multi-head self-attention block is inserted after the FIRST hidden
layer and only there (`networks.py:189-197`, and identically for correlation at
`networks.py:383-391`):

                if i == 0:
                    x = self.attention(x)

The block treats the `nodes` hidden units of a single grid point as `num_heads` tokens of
dimension `nodes // num_heads` and attends ACROSS THE HEAD AXIS, with a Pre-LayerNorm
residual (`net.py:50-80`). Head counts in this registry are 4 everywhere except
`shallow_attn`, which uses 2 (executed check C1). Construction refuses `nodes` not
divisible by `num_heads` (`networks.py:104-108`).

**Source.** Vaswani et al. 2017, Sections 3.2.1-3.2.2, for the attention and multi-head
forms; Xiong et al. 2020, Section 3, for the Pre-LayerNorm convention. Both cited at
`net.py:60-66`.

### 1.11 Per-descriptor summary

Feature counts and keys are from executed check C7; architecture membership from C1.

| descriptor | features | DM-dependent | pretraining columns | training keys | exchange footing |
|---|---|---|---|---|---|
| base `s` (X), `rs, s` (C) | 1 / 2 | no | `rho_all`, `sigma_all`, `rho_x`, `sigma_x` | live rho, sigma | doubled: `2 rho_sigma`, `4 sigma_sigma` |
| `zeta` (C only) | 1 | no | `zeta_all` | live zeta | not applicable (correlation is on the total density) |
| `cusp` | 2 | no | `cusp_all`, `cusp_x` | `cusp_features` | shared block (geometry-only) |
| `dm_statistics` | 2 | yes | `dm_all`, `dm_x` | `dm_features`, `dm_features_a/b` | `diag(P_sigma, P_sigma)` |
| `rung35` | 2 | yes | `rung35_all`, `rung35_x` | `rung35_features`, `rung35_features_a/b` | `[n_sigma, n_sigma]` |
| `rung35_multishell` | 6 | yes | `rung35ms_all`, `rung35ms_x` | `rung35ms_features`, `rung35ms_features_a/b` | per width, `[n_sigma, n_sigma]` |
| `metagga` | 1 | yes | `metagga_all`, `metagga_x`, `metagga_mesh` | `metagga_features`, `metagga_features_a/b` | `alpha(2 rho_s, 4 sigma_s, 2 tau_s)` |

Input widths per architecture, from `n_input_features` (executed check C1): 2 for the
descriptor-free forms, 4 for `cusp`-only / `dm`-only / `rung35`-only, 5 for
`cusp + metagga`, 6 for `dm + cusp` and for `cusp + rung35`, 7 for
`cusp + rung35 + metagga`, 10 for `cusp + rung35_multishell`, 11 for
`cusp + rung35_multishell + metagga`. These are the registry values; the correlation
network adds one column for `zeta` under the run-level
`use_polarized_correlation: true`.


## 2. Training material

### 2.1 The pretraining set

**The DFS inventory.** Eight free atoms with explicit spins -- P (2S=3), N (3), H (1),
Li (1), O (2), Cl (1), Al (1), S (2) -- plus 22 molecules of the Haunschild and Klopper
G2/97 set (Theor. Chem. Acc. 131, 1112 (2012)) taken at trajectory indices
`[2, 113, 25, 18, 11, 17, 114, 121, 101, 0, 20, 26, 29, 67, 28, 110, 125, 10, 115, 89, 105, 50]`
(executed check C5), every molecule run as a closed shell -- including O2 and CH2, which are
open-shell species physically: the protocol poses them at 2S = 0 and the targets follow
(`dfs_pretrain_set.py:1-16`). The meta-GGA variant of the protocol drops H2 and N2
(`MGGA_EXCLUDED`, `dfs_pretrain_set.py:31`), giving 28 systems against 30. Executed check
C5 confirms: level `gga` = 30 systems (8 atoms + 22 molecules), level `mgga` = 28 (8 + 20).
Geometries are committed package data (`data/dfs_pretrain_set.json`) rather than read from
the ASE trajectory, so the compute nodes and the certificate resolve byte-identical
geometries (`dfs_pretrain_set.py:12-16`).

**The pool atoms.** Every single-atom species of the merged BH76 and W4-11 pools, read from
the committed pool JSON rather than transcribed: fourteen distinct (symbol, charge, 2S)
triples -- the twelve neutral elements Al, B, Be, C, Cl, F, H, N, O, P, S, Si at their
Hund's-rule ground-state spins, plus the closed-shell anions F- and Cl- that are BH76
reactants (`pretrain_data_gen.py:213-239`; executed check C6 enumerates all fourteen).

**The composition rule.** `resolve_pretrain_systems` orders the DFS inventory first, then
the pool atoms, then the explicit `atoms` list, first occurrence of a
(geometry, charge, spin) winning (`pretrain_data_gen.py:334-364`). The v6 configuration sets
`dfs_set: true`, `pool_atoms: true` and an explicit atoms block
`{H: 1, Li: 1, C: 2, N: 3, O: 2, F: 1, Na: 1}`. Under the two inventories that list is
almost entirely redundant and contributes exactly ONE system neither supplies: the Na atom,
which Na2's atomization energy rests on. Executed check C6 resolves the set to 38 systems on
the PBE parent and 36 on the SCAN parent (whose DFS variant drops H2 and N2) -- the same
counts the configuration states. He is absent and must stay absent: PySCF has no He in
6-311++G(3df,2pd).

**The parent density.** `parent_density: auto` resolves to the architecture's rung
baseline through `ArchitectureConfig.is_meta_gga` -- PBE for a GGA-rung architecture, SCAN
for a meta-GGA one (`pretrain_data_gen.py:259-307`; executed check C6 confirms
`deep_3x16 -> pbe`, `deep_mgga_3x16 -> scan`, `deep_rung35ms_mgga_3x16 -> scan`,
`deep_rung35only_3x16 -> pbe`). It is the same predicate `inputs.seed_xc: auto` uses, so a
meta-GGA architecture is pretrained on, and certified against, the functional its own SCF is
seeded from. The set is generated at the production identity: 6-311++G(3df,2pd), grid level
3, density fitting (`dfs_pretrain_set.py:97-99`; executed check C5 reports the
`MoleculeSpec` defaults as `6-311++G(3df,2pd)` and grid level 3).

**The footing.** `exchange_footing: spin_channel`. Each open shell's exchange rows are posed
per spin channel at `(2 rho_sigma, 4 sigma_sigma_sigma, features of diag(P_sigma, P_sigma))`
with the parent's SPIN-UNPOLARIZED enhancement factor at those inputs as the target -- the
`eval_xc(..., spin=0)` call on the doubled density, not the spin-polarized call on the
physical one (`pretrain_data_gen.py:379-479`). Each such row carries HALF the grid weight,
because `E_x = (E_x[2 rho_a] + E_x[2 rho_b]) / 2`, so summing both channels reproduces the
parent's open-shell exchange energy exactly (`pretrain_data_gen.py:392-397`). Correlation is
untouched: the total density with `zeta`.

**The two data blocks.** A pretraining file carries a total-density block (`<stem>_all`),
which supplies the correlation rows always and the exchange rows under the historical
footing; a per-channel exchange block (`<stem>_x`), present only on the `spin_channel`
footing; and the synthetic mesh block (`<stem>_mesh`). The stems are declared at
`pretrain_data_gen.py:1490-1533`. `_assemble_pretrain_descriptors` selects the block by
suffix and refuses `for_cnet` with any suffix but `_all`, because correlation is
spin-interpolated rather than spin-scaled (`pretrain.py:849-890`).

**The mesh regularizer.** A synthetic `(r_s, s, alpha)` mesh over
`MESH_RS = (0.1, 0.3, 0.7, 1.5, 3.0, 5.0, 10.0)`,
`MESH_S = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)` and
`MESH_ALPHA = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)`
(`pretrain_data_gen.py:1068-1070`), each node realized as a physical `(rho, sigma, tau)`
triple so its alpha column is produced by the same `compute_alpha` the SCF sees
(`pretrain_data_gen.py:1104-1109`). Its share of the total integration weight is
`MESH_WEIGHT_FRACTION = 0.3` (`pretrain_data_gen.py:1076`; executed check C6; and the v6
configuration states `mesh_fraction: 0.3` rather than inheriting it). The share is a
deliberate choice, not an emergent one: the atomic rows carry physical quadrature weights
and the mesh rows carry none, and pushing the mesh's synthesized densities through the
`|rho eps_LDA|` factor was measured to hand the mesh about 0.99997 of the loss
(`pretrain.py:947-952`). It exists because SCAN's `F_c` is three-dimensional in
`(r_s, s, alpha)` and the atomic grids leave the alpha axis underdetermined -- the meta-GGA
correlation network was measured at up to 0.457 from SCAN away from `alpha = 1`, where the
GGA correlation network sits within 0.013 of PBE (`pretrain.py:1271-1278`).

A DESCRIPTOR GATE restricts it: a mesh node is a synthetic triple with no geometry, so it
cannot define cusp or projection columns, and appending it for an architecture that consumes
those would teach the network that fabricated descriptor values pair with real SCAN targets.
The mesh is appended only for an architecture whose descriptor set is exactly `(metagga,)`
-- `deep_mgga_3x16` and `deep_mgga_attn_3x16`; the other three meta-GGA forms keep the
atoms-and-molecules seed and the run logs that fact (`pretrain.py:1280-1320`).

### 2.2 The pretraining objective

The point-wise term is an integration-weighted mean of squared enhancement-factor
residuals, with per-point weight `|rho eps_LDA| w_grid` -- the FIRST power of the energy
density magnitude times the quadrature weight (`pretrain.py:87-150`). Beside it sits a
per-system energy term (`pretrain.py:163-167`):

    w_E * (1 / N_sys) sum_s ( sum_{i in s} w_i e_LDA_i F^NN_i - E_s )^2

in Hartree squared, with `E_s` the parent's own value of the same quadrature. It exists
because the point-wise residual alone does not bound a system's energy: across seven
architectures the one with the LOWEST exchange residual carried the LARGEST
atomization-energy offset from its parent (`pretrain.py:169-172`;
`SPEC_pretrain_fidelity_program.md` Section 2, defect D2). The mean over systems rather than
the sum keeps the term's magnitude independent of how many systems the file holds. Rows
belonging to no system -- the mesh -- carry zero weight and a sink segment index that is
dropped.

At exactly `energy_term_weight = 0.0` the term is not small, it is not evaluated at all
(`pretrain.py:178-180`, implemented at `pretrain.py:228-238`), which is the pre-protocol objective. For an
unanchored run `validate_grid_semantics` rejects the combination `dfs_set: true` +
`fidelity.enforce: true` + `energy_term_weight: 0.0`, so such a submission stops on the
login node rather than after the datagen job and the pretrainings. The weight was to be
measured, not derived: `hpcjobs/probe_pretrain_energy_weight.py` sweeps it and recommends
the smallest weight at which every architecture clears both halves of the certificate with
margin (`margin_fraction = 0.5`) without any point-wise loss rising by more than a factor of
3 from its own weight-zero value. The sweep (job 2134963) found no such weight: the
atomization gate does not follow the term, the residual being the correlation network's fit
(`SPEC_parent_anchor.md` Section 2). The v6 configuration therefore anchors every
architecture to its parent (`model: {parent_anchor: true}`, Section 3.2 of that
specification): the networks equal the parent at initialization, the certificate holds by
construction, and `energy_term_weight: 0.0` is the exact statement of the objective, which
the semantic check accepts for an anchored run.

Validation replaces the DFS protocol's hand interruption: a seeded 20% of the
multi-nucleus systems is withheld and scored every 50 steps, training stops after 10
validations with no improvement and the best weights are kept
(`validation_fraction: 0.2`, `validation_seed: 0`, `validate_every: 50`, `patience: 10`).
At `n_steps: 2500` that is 50 checks, so patience 10 is far from degenerate.

**Deviations from the DFS protocol, adopted deliberately.** Recorded at
`SPEC_pretrain_fidelity_program.md` Section 6: (1) the set is the DFS 30 (28 for the
meta-GGA) plus every pool atom, at the production identity, on the parent functional's own
self-consistent density -- DFS used grid level 1, basis 6-311++G** and PBE densities for
BOTH rungs; (2) the exchange footing is DFS's per-channel doubled-spin form, extended to
every density-matrix feature through `diag(P_sigma, P_sigma)`; (3) the objective is
integration-weighted and carries the per-system energy term, against DFS's unweighted
`MSELoss` with no per-system term; (4) acceptance is the certificate below with a hard
threshold, against DFS's unthresholded printout.

### 2.3 The certificate

For one architecture the certificate evaluates, through the production energy path on the
parent's own self-consistent density at the run's identity,

    dE_xc = E_xc^NN[rho_parent] - E_xc^parent[rho_parent]

for every free atom of the pools, the DFS pretraining molecules and three common molecules
from the pools, folding the molecular differences against the free atoms into

    dAE(mol) = dE_xc(mol) - sum_atoms n_atom * dE_xc(atom)

(`cluster/fidelity.py:1-20`). PASS requires `max |dE_xc|` over the free atoms at or below
`tol_atom` AND `max |dAE|` at or below `tol_AE`, with finite measurements, converged
references and both parent-route agreements; the spin-scaling oracles O1-O4 are exercised
by CI and the offline workflow matrix, NOT by the per-checkpoint certificate driver. The v6 tolerances are the program's binding values, `tol_AE = 1.0` kcal/mol
and `tol_atom = 1.0` mHa, with `override_reason: null` and `enforce: true` (executed check
C5 reads the block back as `{'tol_AE': 1.0, 'tol_atom': 1.0, 'override_reason': None,
'enforce': True}`). The parent's `E_xc` on each record is computed three independent ways --
point-wise libxc on the stored grid, PySCF numint on a fresh grid, and the reference SCF's
own accumulated value -- and a disagreement above tolerance, a non-converged reference SCF,
a non-finite measurement, or an unevaluable system each FAIL by name
(`cluster/fidelity.py:20-28`). Enforcement has two layers: the on-node gates honour the
recorded `enforced` flag, while `validate_run`, the cross-arm merge and the figure loaders
require PASS unconditionally, so a non-enforcing run can never become a quantitative result
(`cluster/fidelity.py:38-52`).

### 2.4 The training pools and the subset ledger

**The 26-point pool.** `build_dfs_pool_points()` returns 26 points: 21 atomization
energies, 3 BH76 reactions and 2 IP13 ionization potentials (executed check C8, which also
lists the names). Under `ae_as_reactions: true` the 21 AE points are re-posed as reactions
and the pool reads 24 `bh76` + 2 `ip13` (executed check C8) -- the SAME 26 points under
identical names, so a name-keyed ledger resolves the same subsets either way
(`training_points.py:129-160`). The pool is transcribed from Dick and Fernandez-Serra
2021 SI Section II (`dfs_pool.py:1-20`).

**Own-atom atomization energies.** With `ae_as_reactions: true` an atomization energy
becomes `AE = sum_Z n_Z E_NN(Z) - E_NN(mol)`, trained through the reaction channel with the
network's OWN atom energies rather than against the fixed table
(`training_points.py:129-141`). This is the form dpyscf uses, and the dfs_step7 forensics
traced the Na2 blowup to the fixed-anchor relative AE form. What survives of the tabulated
anchors at runtime is the `w_atomic = 0.01` regularizer on the network's free-atom
energies, scoped to H and Li -- the Letter's own atomic-density set
(`training_points.py:35-43`; `LOSS_PRIMER.md` Section 1.2).

**Reference provenance.** Of the 21 AE entries, 19 carry Haunschild and Klopper, J. Chem.
Phys. 136, 164102 (2012), Table I, column `E_ref,non-rel` -- CCSD(T)(F12)/cc-pVQZ-F12 with
higher-excitation and core/core-valence corrections, converted from kJ/mol; the remaining
two, H2O and C2H2, carry the W4-11 values of Karton, Daon and Martin, Chem. Phys. Lett. 510,
165 (2011), which is DFS reference [29], taken from `data/w411_full_pool.json`
(executed check C9: `{'Haunschild2012': 19, 'W4-11 (Karton 2011': 2}`). The three BH76
reactions carry Zheng, Zhao and Truhlar, J. Chem. Theory Comput. 5, 808 (2009)
(NHTBH38/08 entries 1 and 5, HTBH38/08 entries 19-20), also in GMTKN55-BH76; each record
holds both a `barrier_ref` and a `reaction_energy_ref`, and `bh76_mode: reaction_energy`
selects the latter -- a documented deviation from the Letter, which used the barrier heights
(`training_points.py:194-206`). The two IP13 pairs carry NIST Atomic Spectra Database first
ionization energies (executed check C9). The atomic-energy table used for anchoring and for
single-atom target placeholders is Chakravorty, Gwaltney, Davidson, Parpia and Froese
Fischer, Phys. Rev. A 47, 3649 (1993), Table XI, with H set to the exact hydrogenic -1/2 Ha
(`cluster/domain.py:28-59`). Density and V_xc references are CCSD, generated by the
pipeline in `external_refs.py` (SCF, then CCSD density, then OEP inversion for V_xc) and,
for the held-out pool, by the density-only variant in `benchmark_refs.py`.

**The subset ledger.** The v6 subset-size axis is `[1, 2, 3, 4, 5, 6, 7, 12, 15, 18, 26]`,
eleven sizes, the largest equal to the whole 26-point pool. The committed ledger at
`notebooks/checkpoints_step7/alpha_on/subset_index_log.json` carries exactly those eleven
sizes under the `jsd` metric the sweep selects, and `jsd/26` holds 26 points spanning all
three kinds (executed check C10). The same ledger has served v3, v4 and v5, which is what
makes the subset axis comparable across the campaign lineage.

### 2.5 The training loss

The sweep selects `L5_gradnorm_vxc_step7`, a five-channel loss with target kinds
`("AE", "BH76", "IP13", "vxc", "rho")`. The per-update objective is

    L = 1 * loss_AE + 1 * loss_BH76 + 1 * loss_IP13 + 1 * loss_vxc + 20 * loss_rho

with FIXED channel weights from `_DEFAULT_CHANNEL_WEIGHTS`. `notebooks/analysis/LOSS_PRIMER.md`
is the canonical statement of this and is the source cited here; it records the executed
proof on production artifacts that the logged loss equals that combination to machine
precision. This is the Letter's weight STRUCTURE: Dick and Fernandez-Serra, PRB 104,
L161109 (2021), after Eq. 18, "We set the weights to lambda_RE = 1, lambda_n = 20, and
lambda_E = 0.01" -- the reaction-energy channels carry 1 and the density channel 20. There
is no separate total-energy channel, so `lambda_E = 0.01` has no direct analog; its closest
relative is the `w_atomic = 0.01` anchor regularizer inside `loss_AE`
(`LOSS_PRIMER.md` Section 1 and 1.2).

GradNorm is present in the codebase but DORMANT on every dfs_step7 run: the YAML sets no
`update_scheme`, so the `per_molecule` default applies, and `validate_every: 25` REQUIRES
it; the per-molecule loop is dispatched before the balancer is consulted and rebuilds the
loss per group with the `vxc_weight` and `density_weight` pre-scales forced to 1, so those
two YAML knobs are inert (`LOSS_PRIMER.md` Section 1.1). The density channel is the
Letter's `L_n` up to one documented detail: dpyscf normalizes per spin channel by
`N_sigma^2` while this code carries a spin-summed density and uses the total `N_e^2`
(`density_per_electron: true`).

With `scf_loss_use_tail: true`, `scf_loss_tail: 10` and `scf_loss_weight_power: 2.0` on the
`full_3` solver, each energy residual is scored on a weighted window of the three-cycle SCF
trajectory with squared step weights `[0, 0.0625, 1]` -- the first cycle carries exactly
zero weight. The tail applies to the energy channels only; `loss_vxc` and `loss_rho` use the
final density (`LOSS_PRIMER.md` Section 3). The same solver block pairs the tail loss with
the DFS step-decaying mixer `alpha = 0.3**step + 0.3` (`mixer_kwargs: {base: 0.3, floor: 0.3}`).

### 2.6 Held-out evaluation channels

`inline_eval: true` runs evaluation inside the train task. Four held-out passes are made,
each writing its own subdirectory (`cluster/_eval_one_spec.py:620-676`):

1. `eval_holdout` -- the final checkpoint `model.eqx`.
2. `eval_holdout_best` -- `model_best.eqx`, the lowest trailing-mean training loss.
3. `eval_holdout_val_best` -- `model_val_best.eqx`, the minimum in-loop validation
   snapshot, which is the best-generalizing model.
4. `eval_holdout_coldstart` -- the final checkpoint under the cold-start trajectory
   diagnostic (`eval_coldstart: true`): a functional-free `minao` seed, 25 cycles and
   `conv_tol = 1e-12`, so every cycle executes and is recorded; 25 is the Letter's SCF step
   count (`eval_holdout.py:49-76`).

The in-sample `eval_df.csv` is written beside them and remains the authoritative success
signal for the array task; a held-out failure is not fatal.

The held-out pool is the union of BH76 and W4-11: 216 reactions over 214 unique species
(79 BH76 + 152 W4-11, 17 overlapping), evaluated at the run's own basis and grid
(`full_benchmark_pools.py:521-527`). `held_out_strict: true` requires a reaction to have no
species in common with the training subset.


## 3. Cell groups

Every group is a standalone submission with its own configuration file, its own run
directory, its own pretraining-data root and its own certificate gate; none waits on
another. The subset-size axis is the same eleven values throughout, so a group of `n`
architectures is `n x 11` cells. The order is cheapest-and-most-diagnostic first, and within
a group the GGA-rung architectures are submitted before the meta-GGA parity forms where
those exist. The six files, in submission order (executed check C11 read the
five-file ladder back; the 2026-08-30 split of the families file into two trios
postdates that record):

| order | file, under `hpcjobs/configs/` | rung | archs | cells |
|---|---|---|---|---|
| 1 | `dfs_step7.dfs6311_grid3_v6g1_size.yaml` | GGA | 4 | 44 |
| 2 | `dfs_step7.dfs6311_grid3_v6g2a_families_core.yaml` | GGA | 3 | 33 |
| 3 | `dfs_step7.dfs6311_grid3_v6g2b_families_rung35.yaml` | GGA | 3 | 33 |
| 4 | `dfs_step7.dfs6311_grid3_v6g2_families_mgga.yaml` | meta-GGA | 5 | 55 |
| 5 | `dfs_step7.dfs6311_grid3_v6g3_dm.yaml` | GGA | 3 | 33 |
| 6 | `dfs_step7.dfs6311_grid3_v6g4_ablations.yaml` | GGA | 2 | 22 |

Each group's train array is gated on a PASS fidelity certificate for every architecture on
its axis: the certificate runs once per architecture on the pretrain node after the networks
are written, the pretrain task exits non-zero on FAIL, and the train array's `afterok`
dependency never releases (`cluster/fidelity.py:38-52`). All six files carry
`fidelity: {tol_AE: 1.0, tol_atom: 1.0, override_reason: null, enforce: true}` and all six
carry `model: {parent_anchor: true, descriptor_coordinates: dfs}` with
`pretrain.energy_term_weight: 0.0`, exact under the anchor (executed check C11 recorded the
weight as the placeholder it then was). All six groups submit as they ship: the GGA-rung
groups anchored to PBE, the meta-GGA group to SCAN (`parents.scan_fx` / `scan_fc` at the
installed libxc's constants and regularizations; `SPEC_parent_anchor.md` Sections 3.1 and
3.8). The preflight additionally sweeps every swept
architecture's certificate with no exemption before the array is submitted.

Registry-wide, the 31 registered architectures split 26 GGA-rung to 5 meta-GGA-rung through
`ArchitectureConfig.is_meta_gga` (executed check C1). The campaign sweeps 20 of them --
15 GGA-rung and 5 meta-GGA-rung -- in the groups below, for 220 cells; the remaining
11 are listed in Section 3.5 (executed check C2 verifies that the group lists are
pairwise disjoint, that their union with the exclusion list is exactly the registry, and
that the cell counts sum as stated).

### 3.1 Group 1 -- size ablations (4 architectures, 44 cells)

File: `hpcjobs/configs/dfs_step7.dfs6311_grid3_v6g1_size.yaml`.
Axis: `medium`, `medium_attn`, `shallow`, `shallow_attn`.

Shapes, from executed check C1: `medium` and `medium_attn` are depth 3 / width 16 (the
latter with 4 heads); `shallow` and `shallow_attn` are depth 2 / width 8 (the latter with
2 heads). All four are descriptor-free with `descriptor_log_transform = False`, so they
carry the raw semilocal inputs of Section 1.1 and nothing else.

**Question answered.** Whether network capacity, at fixed inputs, is a limiting factor at
all, and whether attention helps at capacities below the production width. Because these are
the smallest and cheapest cells in the campaign, running them first is also the cheapest
sanity ladder for the whole pipeline: certificate, submission, resume, evaluation and the
figure loaders are all exercised at minimum cost before an expensive group is committed.

**No meta-GGA counterparts exist in the registry.** There is no `medium_mgga` or
`shallow_mgga` entry (executed check C2: the group carries 0 meta-GGA architectures, and the
five meta-GGA forms are all depth-3 / width-16 `deep_*` names). The size ablation is
therefore GGA-only by construction, not by choice.

### 3.2 Group 2 -- the six production families and their meta-GGA parity forms (11 architectures, 121 cells; submitted as 33 + 33 + 55)

GGA-rung, submitted as two trios (split 2026-08-30: the QOS submit cap counts
100 entries per user and a 75-entry submission cannot sit beside a draining
group, while each 33-cell trio is 39 entries and submits independently).
The core trio (3 architectures, 33 cells), file
`hpcjobs/configs/dfs_step7.dfs6311_grid3_v6g2a_families_core.yaml`: `deep_3x16`,
`deep_attn_3x16`, `deep_cusp_3x16`. The rung-3.5 trio (3 architectures, 33
cells), file `hpcjobs/configs/dfs_step7.dfs6311_grid3_v6g2b_families_rung35.yaml`:
`deep_rung35_3x16`, `deep_rung35_attn_3x16`, `deep_rung35ms_3x16`.

Meta-GGA parity forms, submitted second (5 architectures, 55 cells), file
`hpcjobs/configs/dfs_step7.dfs6311_grid3_v6g2_families_mgga.yaml`: `deep_mgga_3x16`,
`deep_mgga_attn_3x16`, `deep_cusp_mgga_3x16`, `deep_rung35_mgga_3x16`,
`deep_rung35ms_mgga_3x16`.

**Question answered.** This is the campaign's headline: at one fixed capacity (depth 3,
width 16 -- the DFS network shape), what does each descriptor family buy, and does adding
the rung-3 iso-orbital indicator on top of it buy more? Holding depth and width fixed is
what makes the comparison a descriptor comparison rather than a capacity comparison. It is
ordered second because it is the group the paper's conclusions rest on, and it should run
before anything whose result would only qualify it.

**The parity map**, matched on depth, width, attention, head count, input transform and the
non-`metagga` descriptor tuple (executed check C3): `deep_3x16 -> deep_mgga_3x16`,
`deep_attn_3x16 -> deep_mgga_attn_3x16`, `deep_cusp_3x16 -> deep_cusp_mgga_3x16`,
`deep_rung35_3x16 -> deep_rung35_mgga_3x16`, `deep_rung35ms_3x16 -> deep_rung35ms_mgga_3x16`.
`deep_rung35_attn_3x16` has NO meta-GGA parity form in the registry, which is why the group
is 6 + 5 and not 6 + 6. All five meta-GGA forms are consumed by the map.

**Rung consequences.** All five meta-GGA forms resolve to the SCAN parent for pretraining
and to the SCAN SCF seed (executed check C1); the six GGA forms resolve to PBE for both. Two
of the five -- `deep_mgga_3x16` and `deep_mgga_attn_3x16` -- take the synthetic
`(r_s, s, alpha)` mesh; the other three consume descriptors a geometry-free mesh node
cannot define and pretrain on the atomic and molecular grids alone (Section 2.1).

### 3.3 Group 3 -- the DM-inclusive production-width forms (3 architectures, 33 cells)

File: `hpcjobs/configs/dfs_step7.dfs6311_grid3_v6g3_dm.yaml`.
Axis: `deep_dm_3x16`, `deep_combined_3x16`, `deep_combined_attn_3x16`.

**Question answered.** Whether the density-matrix indicators, in their REPAIRED two-column
form, still carry the held-out collapse that the three-column form produced. The 2026-06-28
diagnosis attributed that collapse to the global, molecule-identity `dm_entropy` channel
(Section 1.7); it was deleted on 2026-08-06, and no campaign has swept these architectures
since. The group is placed AFTER the production families because that is the order the
evidence requires: the descriptor was repaired on the basis of a measured defect, and the
repaired form has to be measured against a headline result that already exists rather than
alongside one that does not. The size-consistency caveat of Section 1.7 still applies to all
three, and is why they are reported as a separate group rather than folded into Group 2.

**No meta-GGA counterparts exist in the registry.** There is no `deep_dm_mgga` or
`deep_combined_mgga` entry (executed check C2: the group carries 0 meta-GGA architectures).

### 3.4 Group 4 -- width and depth at the baseline inputs (2 architectures, 22 cells)

File: `hpcjobs/configs/dfs_step7.dfs6311_grid3_v6g4_ablations.yaml`.
Axis: `deep`, `deep_attn`.

**Question answered.** Whether the depth-4 / width-32 shape helps or hurts relative to the
depth-3 / width-16 production shape, with and without attention, at the BASELINE inputs. The
pairing is exact: `deep` and `deep_attn` differ from `deep_3x16` and `deep_attn_3x16` only in
depth and width -- same empty descriptor tuple, same `descriptor_log_transform = True`, same
4 attention heads (executed check C1). The 2026-06-20 review that introduced the `_3x16`
twins recorded the motivation: the 4x32 networks carry about 3.3k parameters against the DFS
shape's about 0.6k, and were judged to overfit the 26-point pool (`config.py:485-489`). The
group is ordered last because it qualifies the production result rather than establishing
it, and because it is the most expensive per cell: the five depth-4 attention architectures
were the reason the wall was raised, and of them only `deep_attn` remains here (of the ten
4x32 forms, `deep` and `deep_attn` remain).

### 3.5 Excluded architectures

Eleven registry architectures are NOT swept in v6 (executed check C2 lists them with their
descriptor sets):

- `deep_cusp`, `deep_cusp_attn` -- depth-4 / width-32 twins of `deep_cusp_3x16`.
- `deep_dm`, `deep_dm_attn` -- depth-4 / width-32 twins of `deep_dm_3x16`.
- `deep_combined`, `deep_combined_attn` -- depth-4 / width-32 twins of the Group 3 forms.
- `deep_notransform`, `deep_notransform_3x16`, `deep_notransform_attn`,
  `deep_notransform_attn_3x16` -- the input-transform ablation of Section 1.1.
- `deep_rung35only_3x16` -- the rung-3.5-without-cusp ablation of Section 1.6.

Rationale: the width and depth question is answered on the baseline inputs by Group 4, so
descriptor-carrying width twins add cost without adding a distinct question; and the
transform and rung ablations are not part of this campaign's argument. Their descriptor
families are all covered in Section 1 -- `dm_statistics` and the combined form are swept at
production width in Group 3, `cusp` in Group 2, `rung35` in Group 2 -- so nothing in the
descriptor inventory goes unmeasured. Any of the eleven rejoins a later campaign by a single
line on the architecture axis; no code change is required.

JUDGMENT CALL, flagged for the operator. The exclusion list is a scoping decision, not a
result. Three of its members would answer questions the campaign does not otherwise ask:
`deep_notransform_3x16` is the only clean measurement of the input transform at the
production shape, `deep_rung35only_3x16` is the only separation of the localized occupancy
from the cusp columns it is always paired with, and the six descriptor-carrying width twins
are the only test of whether the capacity conclusion of Group 4 transfers to
descriptor-carrying networks. Re-cutting the list is a one-line change to the group's
architecture axis and costs 11 x 11 = 121 additional cells if taken in full.

### 3.6 Totals

| group | architectures | cells |
|---|---|---|
| G1 size ablations | 4 (4 GGA) | 44 |
| G2 production families, GGA | 6 | 66 |
| G2 production families, meta-GGA parity | 5 | 55 |
| G3 DM-inclusive, production width | 3 (3 GGA) | 33 |
| G4 width and depth at baseline inputs | 2 (2 GGA) | 22 |
| **campaign total** | **20 (15 GGA + 5 meta-GGA)** | **220** |

The suite's pins confirm `44 + 33 + 33 + 55 + 33 + 22 = 220` and that the six group lists are (executed check C2 established the pre-split five-file equivalent)
pairwise disjoint, that the swept set carries 5 meta-GGA and 15 GGA architectures, and that
the swept set together with the eleven exclusions is exactly `sorted(ARCHITECTURES)`.

NOTE ON THE WHOLE-REGISTRY FILE. `hpcjobs/configs/dfs_step7.dfs6311_grid3_v6.yaml`
enumerates ALL 31 registry architectures on its `sweep.arch` axis, giving 31 x 11 = 341
cells (executed check C5), and is pinned against `sorted(ARCHITECTURES)` by
`test_v6_sweeps_every_registry_architecture` in `xcquinox/alec/tests/test_cluster_examples.py`
(cited by name rather than by line: that file is under active edit). It is
the statement of the METHOD, not the submission, and is not submitted at all: the five group
files above are, and together they run 220 of its 341 cells. Executed check C11 confirms that
the five group axes are disjoint, that their union is 20 distinct architectures, and that the
eleven names absent from that union are exactly the exclusion list of Section 3.5.

A leaf-by-leaf comparison of each group file against the whole-registry file (executed check
C12) measures how far "one protocol" actually goes. Three fields differ everywhere and are
the grouping itself: `sweep.arch`, `inputs.output_root` and `pretrain.data_dir`, the last two
giving every group its own run directory and its own pretraining-data root. Every other field
is identical -- the whole run identity, both solver blocks, every hyperparameter, the three
reference-cache roots, the subset ledger, the seeding protocol, the pretraining protocol
block, the eval flags and the fidelity block.

TWO FIELDS ARE THE EXCEPTION, AND THE FIVE FILES DISAGREE AMONG THEMSELVES. Four of them --
`v6g1_size`, `v6g2_families`, `v6g3_dm` and `v6g4_ablations` -- set
`cluster.oom_retry_partition: extended-96core` and `cluster.timeout_retry_partition: extended-96core` (retargeted 2026-08-30: long-96core's QOS caps MaxWall at 48 h and rejects both the 96 h escalation and the 72 h campaign replay),
which the whole-registry file leaves unset and `v6g2_families_mgga` also leaves unset
(executed check C12, updated after the group files landed: 6 differing leaves for those four
-- the two retry partitions, the 48 h wall against the reference's 72 h, the axis and the two
roots -- and 3 for the meta-GGA group). The asymmetry is the pinned design, not a
disagreement: the four GGA groups mirror the v4gga arm's live retry routing (a 40-core submit
whose out-of-memory cells, and whose wall-killed cells that had not yet written a resume
checkpoint, re-route onto the larger 96-core class at the longer wall), while the meta-GGA
group mirrors the v5 arms' absence of the keys because it already submits on that class, and
the test suite asserts each group matches its own historical arm and that the two arms
differ. The retry partition must remain valid on the login instance the job is submitted
from.

The 48 h wall the four GGA groups ship is below the 48.1-50.1 h reading of the measured
attention cell, deliberately: those cells wall-kill and are recovered by
`python -m xcquinox.alec.cluster resubmit <run_dir> --submit`, which relaunches each one
from its WS5 checkpoint into a second window of the SAME wall on the SAME partition, in the
SAME run directory. The checkpoint takes precedence over the wall-kill record in the
harness's failure classifier, so a checkpointed cell is continued rather than escalated;
the retry keys above apply to the other case, a cell killed before its first checkpoint,
which restarts from step zero and therefore needs the longer wall. Nothing requeues
unattended -- no `--requeue` is rendered -- and a fresh `submit` is the wrong recovery: it
opens a new run directory that cannot see the checkpoints.


## 4. Cross-checks executed

All run with `OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu` under
`/home/awills/anaconda3/envs/xcq/bin/python`; scripts and logs under
`scratch/campaign_doc/`.

- **C1** `check_registry.py` / `check_registry.log` -- `len(ARCHITECTURES) = 31`; the
  `is_meta_gga` split is 5 meta-GGA (`deep_cusp_mgga_3x16`, `deep_mgga_3x16`,
  `deep_mgga_attn_3x16`, `deep_rung35_mgga_3x16`, `deep_rung35ms_mgga_3x16`) to 26 GGA; and
  the per-architecture table of depth, width, attention, head count,
  `descriptor_log_transform`, `meta_gga`, descriptor tuple, `n_extra_features`,
  `n_input_features`, rung and SCF seed quoted throughout Sections 1 and 3.
- **C2** `check_groups.py` / `check_groups.log` -- group sizes 4 / 6 / 5 / 3 / 2 and cell
  counts `44 + 66 + 55 + 33 + 22 = 220`; 20 swept architectures, 5 meta-GGA and 15 GGA; all
  ten pairwise group intersections empty; swept union exclusions equals the registry with
  nothing missing and nothing extra; and the eleven excluded architectures with their
  descriptor sets.
- **C3** `check_parity.py` / `check_parity.log` -- the Group 2 parity map; the single
  unmatched GGA family `deep_rung35_attn_3x16`; all five meta-GGA forms consumed.
- **C4** contained in C1 -- descriptor tuples per architecture, including
  `deep_notransform*` carrying `()` with `logT=False` and `deep_rung35only_3x16` carrying
  `('rung35',)`.
- **C5** `check_data.py` / `check_data.log` -- DFS inventory counts (gga: 30 = 8 + 22;
  mgga: 28 = 8 + 20), `MGGA_EXCLUDED = ('H2', 'N2')`, the G2/97 index list, the
  `MoleculeSpec` production defaults, and the v6 configuration read back:
  `subset_size = [1, 2, 3, 4, 5, 6, 7, 12, 15, 18, 26]` (11 values), 31 architectures on the
  shipped axis, 341 cells, `mesh_fraction = 0.3`, `dfs_set/pool_atoms/parent_density/
  exchange_footing = True/True/auto/spin_channel`, the fidelity block, the production
  identity, and the run-level flags.
- **C6** `check_pretrain_set.py` / `check_pretrain_set.log` -- the fourteen pool atoms with
  charges and spins; `MESH_WEIGHT_FRACTION = 0.3`; the resolved v6 pretraining set at 38
  systems on the PBE parent and 36 on the SCAN parent, with the member lists; and
  `resolve_parent_density` on four representative architectures.
- **C7** `check_misc.py` / `check_misc.log` -- `ALPHA_DEFINITION =
  'smooth_positive_part:width=1e-05'`, `_ALPHA_SMOOTHING_WIDTH = 1e-05`,
  `_ALPHA_MAX = 100.0`, `smooth_positive_part(0, 1e-5) = 5e-06`,
  `DEFAULT_RUNG35_ALPHA = 0.2`, `DEFAULT_RUNG35_MULTISHELL_ALPHAS = (0.05, 0.2, 0.8)`, and
  the five registered descriptors with their feature counts, DM-dependence flags and
  precompute keys.
- **C8** `check_pool.py` / `check_pool.log` -- the 26-point pool at 21 AE + 3 BH76 + 2 IP13,
  the same 26 as 24 BH76 + 2 IP13 under `ae_as_reactions`, the point names, and
  `DICK_ATOM_REGULARIZER_SYMS = ('H', 'Li')`.
- **C9** `check_ae_src.py` / `check_ae_src.log` -- AE reference provenance
  (19 Haunschild2012, 2 W4-11 Karton 2011), the three BH76 records with both reference kinds
  and their Zheng/Zhao/Truhlar sources, and the two IP13 records with their NIST sources.
- **C10** `check_ledger.py` / `check_ledger.log` -- the committed ledger's `jsd` sizes equal
  the v6 subset axis, the largest equals the 26-point pool size, and `jsd/26` holds 26 points
  spanning all three kinds.
- **C11** `check_group_yamls.py` / `check_group_yamls.log` -- each of the six group files
  loaded and read back: architecture axis, subset axis, cell count, output root, fidelity
  block and `energy_term_weight`. Every axis matches the group list of Section 3 exactly;
  each carries the same eleven subset sizes; cells are 44 / 66 / 55 / 33 / 22 summing to 220;
  the union is 20 distinct architectures with no overlap between files; all five carry
  `{'tol_AE': 1.0, 'tol_atom': 1.0, 'override_reason': None, 'enforce': True}` and
  `energy_term_weight = 0.0`; each has its own `output_root`; and the eleven names absent
  from the union are exactly the exclusion list of Section 3.5.
- **C12** `check_group_diff.py` / `check_group_diff.log` -- each group file flattened to
  leaf keys and compared against the whole-registry file: `v6g1_size`, `v6g2_families`,
  `v6g3_dm` and `v6g4_ablations` differ in 6 of 115 leaves, `v6g2_families_mgga` in 3 of 113.
  The three common differences are `sweep.arch`, `inputs.output_root` and
  `pretrain.data_dir`; the three extra ones, present on the four GGA groups only, are
  `cluster.oom_retry_partition` and `cluster.timeout_retry_partition`, both set to
  `extended-96core` (the 7-day-cap class; long-96core until the 2026-08-30 retarget) where the whole-registry file and the meta-GGA group leave them unset, and
  `cluster.time`, the 40-core class's recorded 48 h wall against the reference's 72 h.
