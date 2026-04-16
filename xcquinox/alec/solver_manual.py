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


def run_manual_scf(config: SolverConfig, model, mol_data: dict) -> SCFResult:
    """Entry point for the manual backend."""
    if config.mode == SolverMode.ONESHOT:
        return _oneshot_result(model, mol_data)
    raise NotImplementedError(
        f"run_manual_scf only implements ONESHOT in this task; "
        f"FIXED_J/FULL are added in Task 5.3-5.4"
    )
