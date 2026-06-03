"""xcquinox.alec.solver_manual: Manual JAX-native SCF backend.

Implements run_manual_scf via an unrolled jax.lax.scan over cycles.
Fully differentiable end-to-end through eqx.filter_value_and_grad.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from xcquinox.alec.solver import (
    SolverConfig,
    SolverMode,
    FeaturePolicy,
    SCFResult,
    LinearMixer,
    EnergyConvergence,
    MixerState,
    _oneshot_result,
    _contract_dm_to_grid_with_nabla,
    _reassemble_features,
)

# Shared regularization constants live in ``solver`` so manual + oneshot
# paths cannot silently diverge.
from xcquinox.alec.solver import (
    DEGENERACY_REG,
    SYM_BREAK_SHIFT,
    _sym_break_diag as _symmetry_breaking_perturbation,
)


def _diagonalize_roothaan(F: jnp.ndarray, S: jnp.ndarray, nocc: int) -> jnp.ndarray:
    """Cholesky-transform Fock, eigh, rebuild DM. Restricted (factor of 2).

    Uses the occupation-mask form ``(C * occ) @ C.T`` rather than slicing
    ``C[:, :nocc] @ C[:, :nocc].T``. Both are algebraically equivalent
    but the slice form's reverse-mode gradient through multi-cycle
    ``lax.scan`` (eigh on Fock with degenerate p-orbital eigenvalues)
    can produce NaN that propagates via 0*NaN=NaN. The mask form avoids
    that pathway and matches the UKS path's convention.
    """
    nao = S.shape[0]
    S_reg = S + DEGENERACY_REG * jnp.eye(nao)
    L = jnp.linalg.cholesky(S_reg)
    L_inv = jnp.linalg.inv(L)
    # The tiny diagonal perturbation lifts forward eigenvalue degeneracy so the
    # reverse-mode eigh gradient (which carries 1/(lambda_i - lambda_j)) stays
    # finite at degenerate p-orbital shells, the NaN-gradient bug this guards.
    # It is intentionally in the FORWARD eigh (that is what lifts the
    # degeneracy); it therefore biases the converged DM/energy, but only by
    # O(1e-8), measured ~2e-10 in E worst-case, far below the default
    # conv_tol=1e-6. A "gradient-only" form would not lift the forward
    # degeneracy and so would not fix the gradient.
    F_orth = L_inv @ F @ L_inv.T + jnp.diag(_symmetry_breaking_perturbation(nao, F.dtype))
    _, C_orth = jnp.linalg.eigh(F_orth)
    C = L_inv.T @ C_orth
    occ = (jnp.arange(nao) < nocc).astype(F.dtype)
    return 2.0 * (C * occ) @ C.T


def _diagonalize_roothaan_unrestricted(
    F: jnp.ndarray, S: jnp.ndarray, nocc: int,
) -> jnp.ndarray:
    """Cholesky-transform Fock, eigh, rebuild one-spin DM. No factor of 2.

    When ``nocc == 0`` (beta channel of a UKS atom with one unpaired electron,
    e.g., H spin=1) we bypass eigh entirely and return a zero DM. Otherwise
    use an occupation-mask ``(C * occ) @ C.T`` instead of ``C[:, :nocc] @
    C[:, :nocc].T``. The slice form is algebraically equivalent but its
    reverse-mode gradient through multi-cycle ``lax.scan`` (multi-cycle
    eigh on a Fock with degenerate p-orbital eigenvalues) produces NaN
    that then pollutes everything via ``0 * NaN = NaN`` in IEEE arithmetic.
    """
    nao = S.shape[0]
    # Static Python branch (nocc is a Python int, traced once at jit time).
    if nocc == 0:
        return jnp.zeros((nao, nao), dtype=jnp.result_type(F, S))
    S_reg = S + DEGENERACY_REG * jnp.eye(nao)
    L = jnp.linalg.cholesky(S_reg)
    L_inv = jnp.linalg.inv(L)
    # The tiny diagonal perturbation lifts forward eigenvalue degeneracy so the
    # reverse-mode eigh gradient (which carries 1/(lambda_i - lambda_j)) stays
    # finite at degenerate p-orbital shells, the NaN-gradient bug this guards.
    # It is intentionally in the FORWARD eigh (that is what lifts the
    # degeneracy); it therefore biases the converged DM/energy, but only by
    # O(1e-8), measured ~2e-10 in E worst-case, far below the default
    # conv_tol=1e-6. A "gradient-only" form would not lift the forward
    # degeneracy and so would not fix the gradient.
    F_orth = L_inv @ F @ L_inv.T + jnp.diag(_symmetry_breaking_perturbation(nao, F.dtype))
    _, C_orth = jnp.linalg.eigh(F_orth)
    C = L_inv.T @ C_orth
    # Occupation-mask rebuild (diag(occ) weights each orbital by 1 or 0).
    occ = (jnp.arange(nao) < nocc).astype(F.dtype)
    return (C * occ) @ C.T


def _compute_j_matrix(D: jnp.ndarray, eri: jnp.ndarray) -> jnp.ndarray:
    """Build J[D] = sum_kl eri[ij,kl] * D[kl] from a 4-index s1 ERI tensor."""
    return jnp.einsum("ijkl,kl->ij", eri, D)


def _resolve_coulomb(config: SolverConfig, mol_data: dict):
    """Return a function ``D -> J``. Uses the density-fitted 3-index ``cderi``
    contraction when ``SolverConfig.density_fit`` is set and ``mol_data['cderi']``
    is present, else the full-ERI contraction. Both are JAX einsums on
    NN-independent precomputed tensors -> the SCF stays differentiable."""
    use_df = bool(getattr(config, "density_fit", False)) and \
        (mol_data.get("cderi") is not None)
    if use_df:
        from xcquinox.alec.df_jk import compute_j_df
        cderi = mol_data["cderi"]
        return lambda D: compute_j_df(D, cderi)
    eri = mol_data["eri"]
    return lambda D: _compute_j_matrix(D, eri)


def _compute_total_energy(
    model,
    D: jnp.ndarray,
    rho: jnp.ndarray,
    sigma: jnp.ndarray,
    features: jnp.ndarray,
    grid_weights: jnp.ndarray,
    h_core: jnp.ndarray,
    J: jnp.ndarray,
    e_nuc: jnp.ndarray,
) -> jnp.ndarray:
    """E_total = e_nuc + Tr[h·D] + 0.5·Tr[J·D] + ∫ exc_nn dw.

    Section 5.2 of the spec proves this reduces to fixed_density_total_energy
    when D=D_PBE and J=J[D_PBE]. The ONESHOT fast-path bypasses this helper
    for byte-identical regression; it's only invoked when max_cycles > 0.
    """
    E_xc_nn = jnp.sum(grid_weights * model.eval_exc(rho, sigma, features))
    E_one = jnp.einsum("ij,ij->", h_core, D)
    E_coul = 0.5 * jnp.einsum("ij,ij->", J, D)
    return e_nuc + E_one + E_coul + E_xc_nn


def _compute_total_energy_uks(
    model,
    D_a: jnp.ndarray,
    D_b: jnp.ndarray,
    rho_a: jnp.ndarray,
    rho_b: jnp.ndarray,
    sigma_aa: jnp.ndarray,
    sigma_bb: jnp.ndarray,
    sigma_tot: jnp.ndarray,
    features: jnp.ndarray,
    grid_weights: jnp.ndarray,
    h_core: jnp.ndarray,
    J_total: jnp.ndarray,
    e_nuc: jnp.ndarray,
) -> jnp.ndarray:
    """UKS total energy with the SOLV-01 SPLIT NN XC.

    E_total = e_nuc + Tr[h·(D_a+D_b)] + 0.5·Tr[J_total·(D_a+D_b)] + E_xc^NN

    E_xc^NN = 0.5 * sum_g w_g [eps_x(2·rho_a, 4·sigma_aa)
                              + eps_x(2·rho_b, 4·sigma_bb)]
            +       sum_g w_g  eps_c(rho_tot, sigma_tot)

    Only EXCHANGE obeys the spin-scaling relation (Oliver & Perdew, Phys.
    Rev. A 20, 397 (1979)). CORRELATION is evaluated ONCE on the TOTAL
    density (zeta=0), because the model's correlation baseline
    ``pw92c_unpolarized_scalar`` is spin-unpolarized (von Barth & Hedin,
    J. Phys. C 5, 1629 (1972); PW92, Phys. Rev. B 45, 13244 (1992)).

    The functional derivative of this E_xc w.r.t. D_s is exactly the split
    V_xc that ``_vxc_nn_spin`` builds (vx spin-scaled per spin + shared vc
    on the total density), which the finite-difference consistency test
    guards. ``sigma_tot`` must be |nabla rho_tot|^2 = sigma_aa + 2 sigma_ab
    + sigma_bb computed by the caller from nabla_rho_a + nabla_rho_b.

    LIMITATION (descriptor features), P2-02: the exchange spin-scaling is
    EXACT only for a feature-free (rho, sigma) F_x. With descriptor features
    active, the same molecular features feed both doubled-spin exchange evals,
    so the open-shell relation is an approximation (closed-shell -> RKS stays
    exact). See ``oneshot.split_exc_energy_uks`` for the full discussion.

    ``split_exc_energy_uks`` evaluates correlation with the real
    zeta-dependent PW92 baseline when ``cnet.use_spin_polarization`` is set
    (Dick & Fernandez-Serra, PRB 104 L161109 (2021)); flag False keeps the
    zeta=0 total-density correlation. The SCF Fock build below matches.
    """
    from xcquinox.alec.oneshot import split_exc_energy_uks

    D_tot = D_a + D_b
    E_one = jnp.einsum("ij,ij->", h_core, D_tot)
    E_coul = 0.5 * jnp.einsum("ij,ij->", J_total, D_tot)
    E_xc_nn = split_exc_energy_uks(
        model, rho_a, rho_b, sigma_aa, sigma_bb, sigma_tot,
        features, grid_weights,
    )
    return e_nuc + E_one + E_coul + E_xc_nn


def _build_mixer(config: SolverConfig):
    """Instantiate mixer from config via the MIXER_REGISTRY.

    Looks up ``config.mixer_name`` in ``MIXER_REGISTRY`` and instantiates
    the class with kwargs from ``config.mixer_kwargs``. New mixer types
    only need to subclass ``Mixer`` with a unique ``registry_name`` and
    use ``@register_mixer``: no edits to this function are required.

    Note: ``config.mixer_kwargs`` is typed as ``tuple[tuple[str, float],
    ...]`` (frozen + hashable for jit cache keys); kwargs are converted
    to a dict and passed to ``__init__``. Mixers that need non-float
    kwargs should validate types in their own ``__init__``.
    """
    from xcquinox.alec.solver import MIXER_REGISTRY
    cls = MIXER_REGISTRY.get(config.mixer_name)
    if cls is None:
        raise NotImplementedError(
            f"mixer {config.mixer_name!r} not registered; available: "
            f"{sorted(MIXER_REGISTRY)}"
        )
    kwargs = {k: v for k, v in config.mixer_kwargs}
    return cls(**kwargs)


def _build_criterion(config: SolverConfig):
    """Instantiate convergence criterion from config via CRITERION_REGISTRY.

    Mirrors the ``_build_mixer`` registry-driven pattern: any subclass
    of ``ConvergenceCriterion`` decorated with ``@register_criterion``
    is dispatched here automatically, so adding a criterion needs no
    edits to this function.
    """
    from xcquinox.alec.solver import CRITERION_REGISTRY
    cls = CRITERION_REGISTRY.get(config.convergence_name)
    if cls is None:
        raise NotImplementedError(
            f"criterion {config.convergence_name!r} not registered; "
            f"available: {sorted(CRITERION_REGISTRY)}"
        )
    return cls(tol=config.conv_tol)


class SCFCycleState(NamedTuple):
    density_matrix: jnp.ndarray
    energy: jnp.ndarray
    mixer_state: MixerState
    converged: jnp.ndarray
    cycles_run: jnp.ndarray


def run_manual_scf(config: SolverConfig, model, mol_data: dict) -> SCFResult:
    if config.mode == SolverMode.ONESHOT:
        return _oneshot_result(model, mol_data)
    if bool(mol_data.get("is_unrestricted", False)):
        return _run_manual_scf_uks(config, model, mol_data)
    return _run_manual_scf_rks(config, model, mol_data)


def _run_manual_scf_rks(config: SolverConfig, model, mol_data: dict) -> SCFResult:
    from xcquinox.alec.descriptors import assemble_descriptor_features
    from xcquinox.alec.oneshot import compute_vxc_nn

    policy = config.effective_feature_policy
    mode = config.mode

    D0 = mol_data["dm_pbe"]
    h_core = mol_data["h_core"]
    S = mol_data["s_matrix"]
    nocc = int(mol_data["nocc"])
    ao_grid = mol_data["ao_grid"]
    ao_grid_deriv = mol_data["ao_grid_deriv"]
    grid_weights = mol_data["grid_weights"]
    e_nuc = jnp.asarray(mol_data["e_nuc"])
    J_pinned = mol_data["j_matrix"]
    cusp_cached = mol_data.get("cusp_features")
    s_matrix = mol_data["s_matrix"]

    features_initial = assemble_descriptor_features(model.descriptors, mol_data)

    def _features_and_rho(D):
        if policy == FeaturePolicy.FROZEN:
            return (
                features_initial,
                mol_data["rho_grid"],
                mol_data["sigma_grid"],
                mol_data["nabla_rho_grid"],
            )
        rho_d, nabla_rho_d, sigma_d = _contract_dm_to_grid_with_nabla(
            D, ao_grid_deriv,
        )
        if not model.descriptors:
            # No descriptors to reassemble; reuse the correctly-shaped
            # empty features from the initial precompute.
            return features_initial, rho_d, sigma_d, nabla_rho_d
        feats = _reassemble_features(
            descriptors=model.descriptors,
            dm=D,
            s_matrix=s_matrix,
            cusp_features=cusp_cached,
            n_grid=grid_weights.shape[0],
        )
        return feats, rho_d, sigma_d, nabla_rho_d

    _coulomb = _resolve_coulomb(config, mol_data)

    def _j_for_cycle(D):
        if mode == SolverMode.FIXED_J:
            return J_pinned
        return _coulomb(D)

    features_0, rho_0, sigma_0, _nabla_rho_0 = _features_and_rho(D0)
    J_0 = _j_for_cycle(D0)
    E0 = _compute_total_energy(
        model, D0, rho_0, sigma_0, features_0,
        grid_weights, h_core, J_0, e_nuc,
    )

    mixer = _build_mixer(config)
    criterion = _build_criterion(config)

    init_state = SCFCycleState(
        density_matrix=D0,
        energy=E0,
        mixer_state=mixer.init_state(S.shape[0]),
        converged=jnp.bool_(False),
        cycles_run=jnp.int32(0),
    )

    def body(state, _):
        D_cur = state.density_matrix
        features, rho_cur, sigma_cur, nabla_rho_cur = _features_and_rho(D_cur)
        vxc_nn = compute_vxc_nn(
            model, rho_cur, sigma_cur, features, ao_grid, grid_weights,
            nabla_rho=nabla_rho_cur, ao_grad=ao_grid_deriv,
        )
        J_cycle = _j_for_cycle(D_cur)
        F = h_core + J_cycle + vxc_nn
        D_new = _diagonalize_roothaan(F, S, nocc)
        new_mixer_state, D_mixed = mixer.step(state.mixer_state, D_cur, D_new)
        # Compute E_new as a consistent functional of D_mixed: recompute
        # features / rho / sigma from D_mixed, and rebuild J from D_mixed
        # (FIXED_J mode preserves J_pinned via _j_for_cycle).
        features_mix, rho_mix, sigma_mix, _nabla_rho_mix = _features_and_rho(D_mixed)
        J_mix = _j_for_cycle(D_mixed)
        E_new = _compute_total_energy(
            model, D_mixed, rho_mix, sigma_mix, features_mix,
            grid_weights, h_core, J_mix, e_nuc,
        )
        is_conv = criterion.is_converged_from_energies(state.energy, E_new)
        already = state.converged
        D_out = jnp.where(already, state.density_matrix, D_mixed)
        E_out = jnp.where(already, state.energy, E_new)
        cycles_inc = jnp.where(already, state.cycles_run, state.cycles_run + jnp.int32(1))
        # Freeze the mixer state once the scan has entered the ``already``
        # branch: otherwise emitting ``new_mixer_state`` would advance
        # ``step_index`` (and any history-tracking fields, e.g. DIIS Fock
        # buffers) on every post-convergence cycle. The leaf-wise ``where``
        # keeps the frozen mixer state pytree-shape-compatible with the new one.
        frozen_mixer_state = jax.tree_util.tree_map(
            lambda old, new: jnp.where(already, old, new),
            state.mixer_state, new_mixer_state,
        )
        next_state = SCFCycleState(
            density_matrix=D_out,
            energy=E_out,
            mixer_state=frozen_mixer_state,
            converged=already | is_conv,
            cycles_run=cycles_inc,
        )
        return next_state, E_out

    # Optional activation checkpointing of the per-cycle body: trades recompute
    # for a smaller reverse-mode tape. Off -> scan_body IS body (byte-identical).
    scan_body = jax.checkpoint(body) if config.scf_grad_checkpoint else body
    final_state, energy_trace = jax.lax.scan(scan_body, init_state, None, length=config.max_cycles)

    if policy == FeaturePolicy.FROZEN:
        features_final = features_initial
    else:
        features_final, _, _, _ = _features_and_rho(final_state.density_matrix)

    return SCFResult(
        density_matrix=final_state.density_matrix,
        total_energy=final_state.energy,
        cycles_run=final_state.cycles_run,
        converged=final_state.converged,
        features_used=features_final,
        energy_trace=energy_trace,
    )


def _run_manual_scf_uks(config: SolverConfig, model, mol_data: dict) -> SCFResult:
    """UKS SCF: spin-resolved Fock matrices with spin-scaled V_xc^NN.

    Per-cycle:
      1. rho_a/rho_b, nabla_rho_a/nabla_rho_b, sigma_aa/sigma_bb from (D_a, D_b)
      2. features assembled from total density (D_tot = D_a + D_b)
      3. V_xc^NN_s via the Task 10 spin-scaling relation
         (compute_vxc_nn called with (2 rho_s, 4 sigma_ss, 2 nabla_rho_s))
      4. J_total = J[D_a] + J[D_b] (FULL) or pinned (FIXED_J)
      5. F_s = h_core + J_total + V_xc^NN_s, diag per spin, rebuild D_s (no factor of 2)
      6. Stack (D_a, D_b) and apply the linear mixer elementwise
      7. Energy from D_mixed using ``_compute_total_energy_uks`` (spin-scaled XC)
    """
    from xcquinox.alec.descriptors import assemble_descriptor_features
    from xcquinox.alec.oneshot import compute_vxc_nn, compute_vc_polarized_per_spin

    policy = config.effective_feature_policy
    mode = config.mode

    D0 = mol_data["dm_pbe"]  # (2, nao, nao)
    h_core = mol_data["h_core"]
    S = mol_data["s_matrix"]
    nocc_a = int(mol_data["nocc_a"])
    nocc_b = int(mol_data["nocc_b"])
    ao_grid = mol_data["ao_grid"]
    ao_grid_deriv = mol_data["ao_grid_deriv"]
    ao_xyz = ao_grid_deriv[1:4]  # (3, n_grid, n_ao)
    grid_weights = mol_data["grid_weights"]
    e_nuc = jnp.asarray(mol_data["e_nuc"])
    j_pbe = mol_data["j_matrix"]  # UKS: (2, nao, nao)
    J_pinned_total = j_pbe[0] + j_pbe[1]
    cusp_cached = mol_data.get("cusp_features")
    s_matrix = mol_data["s_matrix"]

    features_initial = assemble_descriptor_features(model.descriptors, mol_data)

    def _spin_resolved_rho(D_ab):
        D_a = D_ab[0]
        D_b = D_ab[1]
        rho_a = jnp.einsum("ij,gi,gj->g", D_a, ao_grid, ao_grid)
        rho_b = jnp.einsum("ij,gi,gj->g", D_b, ao_grid, ao_grid)
        nabla_rho_a = 2.0 * jnp.einsum("ij,dgi,gj->gd", D_a, ao_xyz, ao_grid)
        nabla_rho_b = 2.0 * jnp.einsum("ij,dgi,gj->gd", D_b, ao_xyz, ao_grid)
        sigma_aa = jnp.sum(nabla_rho_a * nabla_rho_a, axis=1)
        sigma_bb = jnp.sum(nabla_rho_b * nabla_rho_b, axis=1)
        # total-density gradient for the correlation piece (zeta=0).
        # nabla_rho_tot = nabla_rho_a + nabla_rho_b, sigma_tot = |nabla_rho_tot|^2
        # = sigma_aa + 2 sigma_ab + sigma_bb.
        nabla_rho_tot = nabla_rho_a + nabla_rho_b
        sigma_tot = jnp.sum(nabla_rho_tot * nabla_rho_tot, axis=1)
        return (rho_a, rho_b, nabla_rho_a, nabla_rho_b,
                sigma_aa, sigma_bb, nabla_rho_tot, sigma_tot)

    def _features_for(D_ab):
        """Assemble descriptor features from the current DM pair.

        CuspDescriptor is geometry-only. DMStatisticsDescriptor receives
        the SPIN-RESOLVED 3-D DM (Pople-Nesbet 1954: D_sigma S D_sigma =
        D_sigma per spin) so the per-spin idempotency-projector branch
        of compute_dm_features fires. Summing alpha+beta into a 2-D total
        DM would instead route UKS through the RKS branch and produce a
        non-zero physically-meaningless idempotency_error. FROZEN policy
        reuses the initial features.
        """
        if policy == FeaturePolicy.FROZEN:
            return features_initial
        if not model.descriptors:
            return features_initial
        return _reassemble_features(
            descriptors=model.descriptors,
            dm=D_ab,                        # 3-D spin-resolved
            s_matrix=s_matrix,
            cusp_features=cusp_cached,
            n_grid=grid_weights.shape[0],
        )

    def _vx_nn_spin(features, rho_s, sigma_ss, nabla_rho_s):
        """EXCHANGE-only spin-scaled V_x^NN for a single spin channel.

        Functional derivative of 0.5 (E_x[2 rho_a] + E_x[2 rho_b]) w.r.t. the
        spin DM (Oliver & Perdew, Phys. Rev. A 20, 397 (1979)). The shared
        correlation potential vc (computed once on the total density) is
        added separately by the caller.
        """
        return compute_vxc_nn(
            model,
            2.0 * rho_s,
            4.0 * sigma_ss,
            features,
            ao_grid,
            grid_weights,
            nabla_rho=2.0 * nabla_rho_s,
            ao_grad=ao_grid_deriv,
            part="x",
        )

    def _vc_nn_total(features, rho_tot, sigma_tot, nabla_rho_tot):
        """CORRELATION V_c^NN on the TOTAL density, computed ONCE.

        Correlation does not obey the exchange spin-scaling relation; it is
        evaluated on rho_tot (zeta=0, since the baseline
        ``pw92c_unpolarized_scalar`` is spin-unpolarized, von Barth & Hedin,
        J. Phys. C 5, 1629 (1972); PW92, Phys. Rev. B 45, 13244 (1992)).
        Because delta rho_tot / delta rho_a = delta rho_tot / delta rho_b = 1,
        this SAME matrix enters BOTH spin Fock matrices. This is the
        ``use_spin_polarization=False`` fast path; the zeta-dependent per-spin
        path uses ``compute_vc_polarized_per_spin`` in the SCF body.
        """
        return compute_vxc_nn(
            model,
            rho_tot,
            sigma_tot,
            features,
            ao_grid,
            grid_weights,
            nabla_rho=nabla_rho_tot,
            ao_grad=ao_grid_deriv,
            part="c",
        )

    _coulomb = _resolve_coulomb(config, mol_data)

    def _j_total_for_cycle(D_ab):
        if mode == SolverMode.FIXED_J:
            return J_pinned_total
        return _coulomb(D_ab[0]) + _coulomb(D_ab[1])

    # Initial energy at D0 = D_PBE.
    (rho_a0, rho_b0, nabla_rho_a0, nabla_rho_b0,
     sigma_aa0, sigma_bb0, _nabla_tot0, sigma_tot0) = _spin_resolved_rho(D0)
    features_0 = _features_for(D0)
    J_total_0 = _j_total_for_cycle(D0)
    E0 = _compute_total_energy_uks(
        model, D0[0], D0[1], rho_a0, rho_b0, sigma_aa0, sigma_bb0, sigma_tot0,
        features_0, grid_weights, h_core, J_total_0, e_nuc,
    )

    mixer = _build_mixer(config)
    criterion = _build_criterion(config)

    init_state = SCFCycleState(
        density_matrix=D0,
        energy=E0,
        mixer_state=mixer.init_state(S.shape[0]),
        converged=jnp.bool_(False),
        cycles_run=jnp.int32(0),
    )

    def body(state, _):
        D_cur = state.density_matrix  # (2, nao, nao)
        (rho_a, rho_b, nabla_rho_a, nabla_rho_b,
         sigma_aa, sigma_bb, nabla_rho_tot, sigma_tot) = _spin_resolved_rho(D_cur)
        features = _features_for(D_cur)
        # V_xc^s = vx_s (exchange spin-scaled) + correlation.
        # a spin-polarization-aware cnet makes correlation PER-SPIN
        # (zeta couples the channels); otherwise vc is shared (the fast path).
        vx_a = _vx_nn_spin(features, rho_a, sigma_aa, nabla_rho_a)
        vx_b = _vx_nn_spin(features, rho_b, sigma_bb, nabla_rho_b)
        if getattr(model.cnet, "use_spin_polarization", False):
            vc_a, vc_b = compute_vc_polarized_per_spin(
                model, rho_a, rho_b, sigma_tot, features, ao_grid,
                grid_weights, nabla_rho_tot, ao_grid_deriv,
            )
            vxc_nn_a = vx_a + vc_a
            vxc_nn_b = vx_b + vc_b
        else:
            vc = _vc_nn_total(features, rho_a + rho_b, sigma_tot, nabla_rho_tot)
            vxc_nn_a = vx_a + vc
            vxc_nn_b = vx_b + vc
        j_total = _j_total_for_cycle(D_cur)
        fock_a = h_core + j_total + vxc_nn_a
        fock_b = h_core + j_total + vxc_nn_b
        dm_new_a = _diagonalize_roothaan_unrestricted(fock_a, S, nocc_a)
        dm_new_b = _diagonalize_roothaan_unrestricted(fock_b, S, nocc_b)
        D_new = jnp.stack([dm_new_a, dm_new_b], axis=0)
        # LinearMixer is elementwise linear; applying it to (2, nao, nao)
        # mixes alpha and beta channels independently with the same alpha.
        new_mixer_state, D_mixed = mixer.step(state.mixer_state, D_cur, D_new)
        # Consistency: recompute energy from D_mixed (same principle as the
        # RKS path, avoids a hybrid D_cur/D_mixed energy).
        (rho_a_m, rho_b_m, _nra_m, _nrb_m,
         sig_aa_m, sig_bb_m, _ntot_m, sig_tot_m) = _spin_resolved_rho(D_mixed)
        features_m = _features_for(D_mixed)
        j_total_m = _j_total_for_cycle(D_mixed)
        E_new = _compute_total_energy_uks(
            model, D_mixed[0], D_mixed[1], rho_a_m, rho_b_m, sig_aa_m, sig_bb_m,
            sig_tot_m, features_m, grid_weights, h_core, j_total_m, e_nuc,
        )
        is_conv = criterion.is_converged_from_energies(state.energy, E_new)
        already = state.converged
        D_out = jnp.where(already, state.density_matrix, D_mixed)
        E_out = jnp.where(already, state.energy, E_new)
        cycles_inc = jnp.where(already, state.cycles_run, state.cycles_run + jnp.int32(1))
        # Freeze mixer state on ``already`` (see _run_manual_scf_rks for
        # full rationale).
        frozen_mixer_state = jax.tree_util.tree_map(
            lambda old, new: jnp.where(already, old, new),
            state.mixer_state, new_mixer_state,
        )
        next_state = SCFCycleState(
            density_matrix=D_out,
            energy=E_out,
            mixer_state=frozen_mixer_state,
            converged=already | is_conv,
            cycles_run=cycles_inc,
        )
        return next_state, E_out

    # Optional activation checkpointing of the per-cycle body: trades recompute
    # for a smaller reverse-mode tape. Off -> scan_body IS body (byte-identical).
    scan_body = jax.checkpoint(body) if config.scf_grad_checkpoint else body
    final_state, energy_trace = jax.lax.scan(scan_body, init_state, None, length=config.max_cycles)

    features_final = _features_for(final_state.density_matrix)

    return SCFResult(
        density_matrix=final_state.density_matrix,
        total_energy=final_state.energy,
        cycles_run=final_state.cycles_run,
        converged=final_state.converged,
        features_used=features_final,
        energy_trace=energy_trace,
    )
