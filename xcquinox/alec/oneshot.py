"""xcquinox.alec.oneshot — fast pure-JAX one-shot prediction.

Implements THE SPEC §6.3:
  - fixed_density_total_energy (A/D1 losses, evaluation metrics)
  - oneshot_dm_prediction_fast (B/C/D2/D3 losses)
  - oneshot_grid_density (C/D3 grid losses, DensityRMSEMetric)
  - oneshot_total_energy (Harris diagnostic — research only)
  - compute_exc_nn, compute_vxc_nn (internal helpers)
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from xcquinox.alec.descriptors import assemble_descriptor_features

# §6.3: module-level constants for numerical regularization.
#
# DEGENERACY_REG: uniform shift. Used on the overlap matrix before Cholesky
# decomposition (conditions S so L = cholesky(S + εI) is stable for
# near-singular basis sets).
DEGENERACY_REG = 1e-10

# SYM_BREAK_SHIFT: magnitude of the NON-UNIFORM diagonal perturbation added
# to the transformed Fock matrix before ``jnp.linalg.eigh``. A uniform
# shift (DEGENERACY_REG * I) does NOT break eigenvalue degeneracies — it
# raises every eigenvalue by the same amount. For molecules with linear
# symmetry (D∞h, e.g. C2H2, HCN, C2H4) the π MO pairs have exactly equal
# energies, and ``eigh``'s reverse-mode derivative uses 1 / (λ_i - λ_j)
# which returns NaN at exact degeneracies. Without this non-uniform
# shift, any loss that differentiates through ``oneshot_dm_prediction_fast``
# (e.g. B_atomization_plus_dm) produces an all-NaN gradient on these
# systems, poisoning the optimizer at step 0.
#
# Size: 1e-8 is comfortably above float64 accumulation noise (~1e-13
# relative) and thousands of orders of magnitude below any physical
# energy scale. Forward output of eigh is shifted by ≤ 1e-8 per MO
# energy — negligible for DM / density predictions.
SYM_BREAK_SHIFT = 1e-8


def _sym_break_diag(nao: int, dtype) -> jnp.ndarray:
    """Small deterministic non-uniform diagonal to break eigenvalue degeneracies.

    Uses sin(i * φ) where φ is the golden ratio — irrational, so no two
    rows collide by periodicity. Values span roughly [-1, 1] scaled by
    SYM_BREAK_SHIFT. Output is fully deterministic in ``nao`` alone, so
    forward results are reproducible across runs (no PRNG state needed).
    """
    idx = jnp.arange(nao, dtype=dtype)
    return SYM_BREAK_SHIFT * jnp.sin(idx * 1.618033988749895)


@eqx.filter_jit
def compute_exc_nn(model, rho, sigma, features, grid_weights):
    """Integrate NN XC energy density: E_xc^NN = sum(weights * exc).

    model.eval_exc returns rho * epsilon_xc, so NO extra rho factor here.
    Returns a JAX scalar (jit/grad-safe).

    JIT-cached: keyed on (model architecture pytree structure, input shapes).
    Calling this with a different model instance of the same architecture
    skips re-tracing — critical for the eval sweep that loads 72
    checkpoints sharing two architectures.
    """
    exc = model.eval_exc(rho, sigma, features)
    return jnp.sum(exc * grid_weights)


def compute_vxc_nn(
    model,
    rho,
    sigma,
    features,
    ao_grid,
    grid_weights,
    nabla_rho=None,
    ao_grad=None,
) -> jnp.ndarray:
    """Wrapper: issue the LDA-fallback warning at call time (so it fires on
    every misuse, not just the first JIT trace), then dispatch to the
    JIT-compiled core. The split keeps the warning side-effect outside
    the cached trace boundary -- otherwise eqx.filter_jit suppresses it
    after the first call."""
    if nabla_rho is None or ao_grad is None:
        import warnings
        warnings.warn(
            "compute_vxc_nn called without nabla_rho/ao_grad: returning "
            "LDA-only V_xc (v_sigma term dropped). Correct only for LDA NNs.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _compute_vxc_nn_lda(model, rho, sigma, features, ao_grid, grid_weights)
    return _compute_vxc_nn_gga(
        model, rho, sigma, features, ao_grid, grid_weights, nabla_rho, ao_grad,
    )


@eqx.filter_jit
def _compute_vxc_nn_lda(model, rho, sigma, features, ao_grid, grid_weights):
    return _compute_vxc_nn_core(
        model, rho, sigma, features, ao_grid, grid_weights,
        nabla_rho=None, ao_grad=None,
    )


@eqx.filter_jit
def _compute_vxc_nn_gga(model, rho, sigma, features, ao_grid, grid_weights,
                        nabla_rho, ao_grad):
    return _compute_vxc_nn_core(
        model, rho, sigma, features, ao_grid, grid_weights,
        nabla_rho=nabla_rho, ao_grad=ao_grad,
    )


def _compute_vxc_nn_core(
    model,
    rho,
    sigma,
    features,
    ao_grid,
    grid_weights,
    nabla_rho=None,
    ao_grad=None,
) -> jnp.ndarray:
    """Assemble NN XC potential matrix V_xc via per-point forward-mode jvp.

    For a GGA E_xc[rho, sigma = |nabla rho|^2], the V_xc matrix element is
        V_xc_ij = integral phi_i v_rho phi_j dr
                + 2 * integral v_sigma nabla_rho . nabla(phi_i phi_j) dr
    The factor of 2 comes from dE_xc/d(nabla_rho) = 2 v_sigma * nabla_rho
    (because sigma = |nabla rho|^2). Expanding nabla(phi_i phi_j) =
    phi_j nabla(phi_i) + phi_i nabla(phi_j) and defining
        A_ij = sum_g w_g v_sigma(g) [nabla_rho(g) . nabla(phi_i)(g)] phi_j(g)
    gives the symmetric form
        V_sigma = 2 * (A + A.T).

    Parameters
    ----------
    rho : (n_grid,)
    sigma : (n_grid,)
    features : (n_grid, n_features) or (n_grid, 0)
    ao_grid : (n_grid, n_ao)
    grid_weights : (n_grid,)
    nabla_rho : (n_grid, 3), optional. If ``None``, the v_sigma term is
        omitted and a warning is issued — this is correct only for LDA NNs.
    ao_grad : (3, n_grid, n_ao) or (4, n_grid, n_ao), optional. If 4 leading
        dims, interpreted as ``eval_ao(..., deriv=1)`` and ``ao_grad[1:4]``
        is used. If ``None``, the v_sigma term is omitted.

    Returns
    -------
    V_xc : (n_ao, n_ao), symmetric.
    """
    def exc_single_point(r, s, f):
        return model.eval_exc_scalar(r, s, f)

    # Sanitize JVP inputs at low-density points. The networks' reduced-gradient
    # transform uses sqrt(sigma) whose derivative diverges at sigma=0, and a
    # spin channel with zero occupations (e.g., beta channel of an H atom with
    # spin=1) produces rho=sigma=0 everywhere. The forward value is fine (all
    # bounded networks), but JVP through sqrt(0) gives NaN, which propagates
    # through the SCF Fock matrix and produces NaN total_energy after training
    # perturbs the NN into that regime. Replace the JVP inputs at tail points
    # with safe (rho=1, sigma=1) defaults; the output is masked to zero below,
    # so the V_xc matrix contribution from these points is exactly zero.
    _V_RHO_THRESHOLD = 1e-10
    safe_rho = jnp.where(rho > _V_RHO_THRESHOLD, rho, jnp.ones_like(rho))
    safe_sigma = jnp.where(rho > _V_RHO_THRESHOLD, sigma, jnp.ones_like(sigma))

    # Per-point JVPs: tangent on rho and then on sigma
    v_rho, v_sigma = jax.vmap(
        lambda r, s, f: (
            jax.jvp(
                exc_single_point,
                (r, s, f),
                (jnp.ones_like(r), jnp.zeros_like(s), jnp.zeros_like(f)),
            )[1],
            jax.jvp(
                exc_single_point,
                (r, s, f),
                (jnp.zeros_like(r), jnp.ones_like(s), jnp.zeros_like(f)),
            )[1],
        )
    )(safe_rho, safe_sigma, features)

    # Mask JVP outputs to zero at tail points (physically negligible
    # contribution AND keeps gradients finite at rho/sigma = 0).
    v_rho = jnp.where(rho > _V_RHO_THRESHOLD, v_rho, 0.0)
    v_sigma = jnp.where(rho > _V_RHO_THRESHOLD, v_sigma, 0.0)

    # LDA-like contribution: V_rho_ij = sum_g w_g v_rho(g) phi_i(g) phi_j(g).
    V_rho = jnp.einsum("g,gi,gj->ij", grid_weights * v_rho, ao_grid, ao_grid)

    if nabla_rho is None or ao_grad is None:
        # LDA-only fallback: warning is issued at the wrapper boundary so
        # it fires on every misuse rather than only on first JIT trace.
        return V_rho

    # Accept either the (4, n_grid, n_ao) eval_ao(deriv=1) layout or the
    # (3, n_grid, n_ao) "derivatives-only" layout. This keeps callers
    # flexible: run_manual_scf/_vxc_term can pass mol_data["ao_grid_deriv"]
    # directly without an extra slice step.
    ao_grad_xyz = ao_grad[1:4] if ao_grad.shape[0] == 4 else ao_grad

    # (nabla_rho . nabla phi_i)(g) contracting cartesian axis d.
    # nabla_rho: (n_grid, 3); ao_grad_xyz: (3, n_grid, n_ao) -> (n_grid, n_ao).
    nabla_rho_dot_ao_grad = jnp.einsum("gd,dgi->gi", nabla_rho, ao_grad_xyz)

    # A_ij = sum_g w_g v_sigma(g) [nabla_rho . nabla phi_i](g) phi_j(g).
    # v_sigma already masked above at rho < threshold, so tail contribution
    # is exactly zero.
    A_matrix = jnp.einsum(
        "g,gi,gj->ij",
        grid_weights * v_sigma,
        nabla_rho_dot_ao_grad,
        ao_grid,
    )
    # V_sigma from d/dD_ij integral v_sigma |nabla rho|^2 dr = 2 v_sigma nabla_rho . nabla(phi_i phi_j).
    V_sigma = 2.0 * (A_matrix + A_matrix.T)
    return V_rho + V_sigma


def fixed_density_total_energy(model, mol_data) -> float:
    """Total energy with NN XC on frozen PBE density. No Roothaan step.

    E_total = E_non_xc + E_xc^NN[rho_PBE]
    Used by A, D1 losses and all energy-based evaluation metrics.
    """
    features = assemble_descriptor_features(model.descriptors, mol_data)
    exc_integrated = compute_exc_nn(
        model,
        mol_data["rho_grid"],
        mol_data["sigma_grid"],
        features,
        mol_data["grid_weights"],
    )
    return mol_data["E_non_xc"] + exc_integrated


def _uks_spin_resolved_vxc(model, mol_data, features):
    """Build spin-resolved V_xc^NN_a, V_xc^NN_b via the spin-scaled approximation.

    Spin-scaling relation (widely used with RKS XC functionals):
        E_xc^UKS[rho_a, rho_b] ~= (E_xc^RKS[2*rho_a] + E_xc^RKS[2*rho_b]) / 2

    Taking the functional derivative w.r.t. the alpha DM gives (for a GGA)
        V_xc_a_ij = v_rho^RKS(2 rho_a, 4 sigma_aa) integral phi_i phi_j dr
                  + 4 v_sigma^RKS(2 rho_a, 4 sigma_aa) integral
                        nabla_rho_a . nabla(phi_i phi_j) dr
    which is exactly what ``compute_vxc_nn`` produces when called with
    (2 rho_a, 4 sigma_aa, 2 nabla_rho_a) — the factor-of-2 in the sigma term
    absorbs the 2*nabla_rho_a scaling and the remaining factor of 2 from the
    scaled sigma. The beta channel is symmetric. This keeps alpha != beta V_xc
    for open-shell systems so the NN can learn spin polarization.
    """
    dm_pbe = mol_data["dm_pbe"]  # (2, nao, nao)
    ao_grid = mol_data["ao_grid"]
    ao_grid_deriv = mol_data["ao_grid_deriv"]
    grid_weights = mol_data["grid_weights"]

    # Per-spin rho and nabla_rho from spin-resolved PBE DM.
    ao_xyz = ao_grid_deriv[1:4]  # (3, n_grid, n_ao)
    rho_a = jnp.einsum("ij,gi,gj->g", dm_pbe[0], ao_grid, ao_grid)
    rho_b = jnp.einsum("ij,gi,gj->g", dm_pbe[1], ao_grid, ao_grid)
    nabla_rho_a = 2.0 * jnp.einsum("ij,dgi,gj->gd", dm_pbe[0], ao_xyz, ao_grid)
    nabla_rho_b = 2.0 * jnp.einsum("ij,dgi,gj->gd", dm_pbe[1], ao_xyz, ao_grid)
    sigma_aa = jnp.sum(nabla_rho_a * nabla_rho_a, axis=1)
    sigma_bb = jnp.sum(nabla_rho_b * nabla_rho_b, axis=1)

    vxc_nn_a = compute_vxc_nn(
        model,
        2.0 * rho_a,
        4.0 * sigma_aa,
        features,
        ao_grid,
        grid_weights,
        nabla_rho=2.0 * nabla_rho_a,
        ao_grad=ao_grid_deriv,
    )
    vxc_nn_b = compute_vxc_nn(
        model,
        2.0 * rho_b,
        4.0 * sigma_bb,
        features,
        ao_grid,
        grid_weights,
        nabla_rho=2.0 * nabla_rho_b,
        ao_grad=ao_grid_deriv,
    )
    return vxc_nn_a, vxc_nn_b


def oneshot_dm_prediction_fast(model, mol_data, solver_config=None) -> jnp.ndarray:
    """Fixed-J Roothaan one-shot DM prediction.

    Builds F_NN = h_core + J[D_PBE] + V_xc^NN(rho_PBE), solves the
    generalized eigenvalue problem F C = S C eps via Cholesky transform,
    and returns the predicted density matrix.

    Returns:
      RKS: shape (n_ao, n_ao)
      UKS: shape (2, n_ao, n_ao)
    """
    if solver_config is not None:
        from xcquinox.alec.solver import run_scf
        return run_scf(solver_config, model, mol_data).density_matrix
    features = assemble_descriptor_features(model.descriptors, mol_data)

    h_core = mol_data["h_core"]
    j_pbe = mol_data["j_matrix"]
    s_matrix = mol_data["s_matrix"]
    nao = s_matrix.shape[0]

    # Cholesky decomposition of regularized overlap
    overlap_reg = s_matrix + DEGENERACY_REG * jnp.eye(nao)
    L = jnp.linalg.cholesky(overlap_reg)
    L_inv = jnp.linalg.inv(L)

    if mol_data["is_unrestricted"]:
        nocc_a = mol_data["nocc_a"]
        nocc_b = mol_data["nocc_b"]

        # Spin-resolved V_xc^NN (spin-scaled approximation; see helper docstring).
        vxc_nn_a, vxc_nn_b = _uks_spin_resolved_vxc(model, mol_data, features)

        # UKS: j_pbe has shape (2, n_ao, n_ao); J_total = J[dm_a] + J[dm_b]
        # enters both spin Fock matrices identically (Coulomb is spin-blind).
        j_total = j_pbe[0] + j_pbe[1]
        fock_a = h_core + j_total + vxc_nn_a
        fock_b = h_core + j_total + vxc_nn_b

        # Transform to orthogonal basis. The uniform DEGENERACY_REG * I
        # shift alone does NOT resolve eigenvalue degeneracies (every
        # eigenvalue moves by the same amount), which breaks the eigh VJP
        # on linear-symmetry molecules. Adding a small non-uniform diag
        # (_sym_break_diag) resolves exact degeneracies so 1/(λ_i - λ_j)
        # in the reverse-mode derivative stays finite. See SYM_BREAK_SHIFT
        # block comment for full rationale.
        # R2-C M4 audit fix: dropped dead `+ DEGENERACY_REG * jnp.eye(nao)`
        # uniform shift (commutes through eigh; doesn't break degeneracies).
        # Only the non-uniform _sym_break_diag does work.
        _sb = jnp.diag(_sym_break_diag(nao, fock_a.dtype))
        fock_orth_a = L_inv @ fock_a @ L_inv.T + _sb
        fock_orth_b = L_inv @ fock_b @ L_inv.T + _sb

        # Eigendecomposition (JAX-native: preserves grad flow through the solver).
        _, mo_coeff_orth_a = jnp.linalg.eigh(fock_orth_a)
        _, mo_coeff_orth_b = jnp.linalg.eigh(fock_orth_b)

        # Back-transform
        mo_coeff_a = L_inv.T @ mo_coeff_orth_a
        mo_coeff_b = L_inv.T @ mo_coeff_orth_b

        # Density matrices (no factor of 2 for UKS).
        # R2-C N1 audit fix: occupation-mask form (C * occ) @ C.T matches
        # the RKS path's gradient stability under multi-cycle eigh on
        # degenerate-eigenvalue Fock matrices (e.g. linear-symmetry mols
        # C2H2 / HCN / C2H4). Pre-fix slice form C[:, :nocc] @ C[:, :nocc].T
        # produced 0*NaN=NaN through reverse-mode at exact p-orbital
        # degeneracies.
        occ_a = (jnp.arange(nao) < nocc_a).astype(mo_coeff_a.dtype)
        occ_b = (jnp.arange(nao) < nocc_b).astype(mo_coeff_b.dtype)
        dm_a = (mo_coeff_a * occ_a) @ mo_coeff_a.T
        dm_b = (mo_coeff_b * occ_b) @ mo_coeff_b.T
        dm_pred = jnp.stack([dm_a, dm_b])
    else:
        nocc = mol_data["nocc"]

        # RKS path: single spin-blind V_xc^NN evaluated on the total density.
        vxc_nn = compute_vxc_nn(
            model,
            mol_data["rho_grid"],
            mol_data["sigma_grid"],
            features,
            mol_data["ao_grid"],
            mol_data["grid_weights"],
            nabla_rho=mol_data.get("nabla_rho_grid"),
            ao_grad=mol_data.get("ao_grid_deriv"),
        )

        # RKS: j_pbe has shape (n_ao, n_ao)
        fock = h_core + j_pbe + vxc_nn

        # Transform to orthogonal basis. Only the non-uniform
        # _sym_break_diag perturbation does work — a uniform
        # DEGENERACY_REG * I shift commutes through eigh and leaves
        # eigenvalue gaps unchanged (M4 audit fix: dropped uniform
        # term as dead weight). See SYM_BREAK_SHIFT block comment for
        # full rationale on the non-uniform shift.
        fock_orth = (L_inv @ fock @ L_inv.T
                     + jnp.diag(_sym_break_diag(nao, fock.dtype)))

        # Eigendecomposition
        _, mo_coeff_orth = jnp.linalg.eigh(fock_orth)

        # Back-transform
        mo_coeff = L_inv.T @ mo_coeff_orth

        # Density matrix (factor of 2 for RKS double occupation).
        # Use occupation-mask form to match the UKS path's gradient
        # stability under multi-cycle eigh on degenerate-eigenvalue
        # Fock matrices (H2 audit fix).
        occ = (jnp.arange(nao) < nocc).astype(mo_coeff.dtype)
        dm_pred = 2.0 * (mo_coeff * occ) @ mo_coeff.T

    return dm_pred


def oneshot_grid_density(model, mol_data, solver_config=None) -> jnp.ndarray:
    """Run oneshot DM prediction, then compute grid density.

    Returns spin-summed density of shape (n_points,) for both RKS and UKS.
    """
    if solver_config is not None:
        from xcquinox.alec.solver import run_scf
        D_total = run_scf(solver_config, model, mol_data).density_matrix
        if mol_data["is_unrestricted"]:
            D_total = D_total[0] + D_total[1]
        ao = mol_data["ao_grid"]
        return jnp.einsum("ij,gi,gj->g", D_total, ao, ao)
    D_NN = oneshot_dm_prediction_fast(model, mol_data)
    ao = mol_data["ao_grid"]

    if mol_data["is_unrestricted"]:
        D_total = D_NN[0] + D_NN[1]
    else:
        D_total = D_NN

    return jnp.einsum("ij,gi,gj->g", D_total, ao, ao)


def oneshot_total_energy(model, mol_data) -> float:
    """Harris functional energy at the NN-predicted density.

    E_harris = Tr[D_NN * (h_core + J[D_PBE] + V_xc^PBE)]
             - 0.5 * Tr[D_PBE * J[D_PBE]]
             - (Tr[D_PBE * V_xc^PBE] - E_xc^PBE)
             + e_nuc

    Research diagnostic only — NOT used by any loss or metric.
    """
    D_NN = oneshot_dm_prediction_fast(model, mol_data)

    h_core = mol_data["h_core"]
    j_pbe = mol_data["j_matrix"]
    vxc_pbe = mol_data["vxc_pbe"]
    dm_pbe = mol_data["dm_pbe"]
    E_xc_pbe = mol_data["E_xc_pbe"]
    e_nuc = mol_data["e_nuc"]

    if mol_data["is_unrestricted"]:
        # UKS: sum traces over both spin channels
        # j_pbe: (2, n_ao, n_ao), dm_pbe: (2, n_ao, n_ao), vxc_pbe: (2, n_ao, n_ao)
        # D_NN: (2, n_ao, n_ao)
        j_total = j_pbe[0] + j_pbe[1]

        # Tr[D_NN * (h_core + J_total + vxc_pbe)] summed over spins
        F_pbe_a = h_core + j_total + vxc_pbe[0]
        F_pbe_b = h_core + j_total + vxc_pbe[1]
        term1 = jnp.trace(D_NN[0] @ F_pbe_a) + jnp.trace(D_NN[1] @ F_pbe_b)

        # Double-counting: 0.5 * Tr[D_PBE * J[D_PBE]]
        term2 = 0.5 * (jnp.trace(dm_pbe[0] @ j_total) + jnp.trace(dm_pbe[1] @ j_total))

        # XC double-counting: Tr[D_PBE * V_xc^PBE] - E_xc^PBE
        term3 = (jnp.trace(dm_pbe[0] @ vxc_pbe[0]) + jnp.trace(dm_pbe[1] @ vxc_pbe[1])) - E_xc_pbe
    else:
        # RKS
        F_pbe = h_core + j_pbe + vxc_pbe
        term1 = jnp.trace(D_NN @ F_pbe)
        term2 = 0.5 * jnp.trace(dm_pbe @ j_pbe)
        term3 = jnp.trace(dm_pbe @ vxc_pbe) - E_xc_pbe

    return term1 - term2 - term3 + e_nuc


def _nn_fx_local_uks(model, rho_alpha: jnp.ndarray,
                    rho_beta: jnp.ndarray,
                    s: jnp.ndarray) -> jnp.ndarray:
    """Evaluate model.xnet as UKS F_x on synthetic (rho_alpha, rho_beta, s) points.

    Step-6 PBE-anchor helper. AlecGGA_XNet.__call__ is a single-grid-point
    evaluator taking a 1-D input tensor ``[rho, sigma, *extras]`` and
    returning a scalar F_x. We therefore vmap over N sample points.

    Spin-scaled UKS approximation (matches ``_uks_spin_resolved_vxc`` at
    SCF time):

        F_x_UKS(ra, rb, s) = 0.5 * (F_x_RKS(2*ra, sigma_aa_eff)
                                   + F_x_RKS(2*rb, sigma_bb_eff))

    where ``sigma_sigma_eff = (1 +/- zeta)**2 * sigma_tot``,
    ``zeta = (ra-rb)/(ra+rb)``, and ``sigma_tot = (2*kF(rho_tot)*s*rho_tot^(4/3))^2``.
    This is the SAME per-spin effective sigma that
    ``_uks_spin_resolved_vxc`` feeds into ``compute_vxc_nn`` during SCF:
    nabla_rho_sigma = (1 +/- zeta)/2 * nabla_rho_tot spatially, so
    ``4 * sigma_sigma_sigma = (1 +/- zeta)**2 * sigma_tot`` — exactly
    ``sigma_sigma_eff`` above.

    Uses zero extras (no descriptor features). The anchor probes the bare
    functional form at synthetic (rho, s) points — no molecular grid
    visits them, so there is no physical descriptor value to feed in.
    """
    n_extra = model.xnet.n_extra_features
    kF_tot = (3.0 * jnp.pi ** 2) ** (1.0 / 3.0)

    rho_tot = rho_alpha + rho_beta
    sigma_tot = (
        2.0 * kF_tot * s
        * jnp.clip(rho_tot, 1e-30, None) ** (4.0 / 3.0)
    ) ** 2
    zeta = jnp.where(
        rho_tot > 0,
        (rho_alpha - rho_beta) / jnp.clip(rho_tot, 1e-30, None),
        0.0,
    )
    sigma_aa_eff = (1.0 + zeta) ** 2 * sigma_tot
    sigma_bb_eff = (1.0 - zeta) ** 2 * sigma_tot

    def _fx_one(rho_spin_doubled, sigma_spin_eff):
        extras = jnp.zeros(n_extra, dtype=rho_spin_doubled.dtype)
        inputs = jnp.concatenate([
            jnp.atleast_1d(rho_spin_doubled),
            jnp.atleast_1d(sigma_spin_eff),
            extras,
        ])
        return model.xnet(inputs)

    fx_a = jax.vmap(_fx_one)(2.0 * rho_alpha, sigma_aa_eff)
    fx_b = jax.vmap(_fx_one)(2.0 * rho_beta, sigma_bb_eff)
    return 0.5 * (fx_a + fx_b)
