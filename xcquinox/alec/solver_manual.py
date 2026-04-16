"""xcquinox.alec.solver_manual — Manual JAX-native SCF backend.

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
    _contract_dm_to_grid,
    _reassemble_features,
)

DEGENERACY_REG = 1e-10


def _diagonalize_roothaan(F: jnp.ndarray, S: jnp.ndarray, nocc: int) -> jnp.ndarray:
    """Cholesky-transform Fock, eigh, rebuild DM. Restricted (factor of 2)."""
    nao = S.shape[0]
    S_reg = S + DEGENERACY_REG * jnp.eye(nao)
    L = jnp.linalg.cholesky(S_reg)
    L_inv = jnp.linalg.inv(L)
    F_orth = L_inv @ F @ L_inv.T + DEGENERACY_REG * jnp.eye(nao)
    _, C_orth = jnp.linalg.eigh(F_orth)
    C = L_inv.T @ C_orth
    C_occ = C[:, :nocc]
    return 2.0 * C_occ @ C_occ.T


def _compute_j_matrix(D: jnp.ndarray, eri: jnp.ndarray) -> jnp.ndarray:
    """Build J[D] = sum_kl eri[ij,kl] * D[kl] from a 4-index s1 ERI tensor."""
    return jnp.einsum("ijkl,kl->ij", eri, D)


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


def _build_mixer(config: SolverConfig):
    """Instantiate mixer from config. Currently only 'linear' is supported."""
    if config.mixer_name == "linear":
        kwargs = dict(config.mixer_kwargs)
        return LinearMixer(alpha=float(kwargs.get("alpha", 0.5)))
    raise NotImplementedError(f"mixer {config.mixer_name!r} not yet implemented")


def _build_criterion(config: SolverConfig):
    """Instantiate convergence criterion from config."""
    if config.convergence_name == "energy":
        return EnergyConvergence(tol=config.conv_tol)
    raise NotImplementedError(f"criterion {config.convergence_name!r} not yet implemented")


class SCFCycleState(NamedTuple):
    density_matrix: jnp.ndarray
    energy: jnp.ndarray
    mixer_state: MixerState
    converged: jnp.ndarray
    cycles_run: jnp.ndarray


def run_manual_scf(config: SolverConfig, model, mol_data: dict) -> SCFResult:
    from xcquinox.alec.descriptors import assemble_descriptor_features
    from xcquinox.alec.oneshot import compute_vxc_nn

    if config.mode == SolverMode.ONESHOT:
        return _oneshot_result(model, mol_data)

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
            )
        rho_d, sigma_d = _contract_dm_to_grid(D, ao_grid_deriv)
        feats = _reassemble_features(
            descriptors=model.descriptors,
            dm=D,
            s_matrix=s_matrix,
            cusp_features=cusp_cached,
        )
        return feats, rho_d, sigma_d

    def _j_for_cycle(D):
        if mode == SolverMode.FIXED_J:
            return J_pinned
        return _compute_j_matrix(D, mol_data["eri"])

    features_0, rho_0, sigma_0 = _features_and_rho(D0)
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
        features, rho_cur, sigma_cur = _features_and_rho(D_cur)
        vxc_nn = compute_vxc_nn(
            model, rho_cur, sigma_cur, features, ao_grid, grid_weights,
        )
        J_cycle = _j_for_cycle(D_cur)
        F = h_core + J_cycle + vxc_nn
        D_new = _diagonalize_roothaan(F, S, nocc)
        new_mixer_state, D_mixed = mixer.step(state.mixer_state, D_cur, D_new)
        E_new = _compute_total_energy(
            model, D_mixed, rho_cur, sigma_cur, features,
            grid_weights, h_core, J_cycle, e_nuc,
        )
        is_conv = criterion.is_converged_from_energies(state.energy, E_new)
        already = state.converged
        D_out = jnp.where(already, state.density_matrix, D_mixed)
        E_out = jnp.where(already, state.energy, E_new)
        cycles_inc = jnp.where(already, state.cycles_run, state.cycles_run + jnp.int32(1))
        next_state = SCFCycleState(
            density_matrix=D_out,
            energy=E_out,
            mixer_state=new_mixer_state,
            converged=already | is_conv,
            cycles_run=cycles_inc,
        )
        return next_state, None

    final_state, _ = jax.lax.scan(body, init_state, None, length=config.max_cycles)

    if policy == FeaturePolicy.FROZEN:
        features_final = features_initial
    else:
        features_final, _, _ = _features_and_rho(final_state.density_matrix)

    return SCFResult(
        density_matrix=final_state.density_matrix,
        total_energy=final_state.energy,
        cycles_run=final_state.cycles_run,
        converged=final_state.converged,
        features_used=features_final,
    )
