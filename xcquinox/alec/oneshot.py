"""xcquinox.alec.oneshot — fast pure-JAX one-shot prediction.

Implements THE SPEC §6.3:
  - fixed_density_total_energy (A/D1 losses, evaluation metrics)
  - oneshot_dm_prediction_fast (B/C/D2/D3 losses)
  - oneshot_grid_density (C/D3 grid losses, DensityRMSEMetric)
  - oneshot_total_energy (Harris diagnostic — research only)
  - compute_exc_nn, compute_vxc_nn (internal helpers)
"""

import jax
import jax.numpy as jnp

from xcquinox.alec.descriptors import assemble_descriptor_features

# §6.3: module-level constant for numerical regularization
DEGENERACY_REG = 1e-10


def compute_exc_nn(model, rho, sigma, features, grid_weights):
    """Integrate NN XC energy density: E_xc^NN = sum(weights * exc).

    model.eval_exc returns rho * epsilon_xc, so NO extra rho factor here.
    Returns a JAX scalar (jit/grad-safe).
    """
    exc = model.eval_exc(rho, sigma, features)
    return jnp.sum(exc * grid_weights)


def compute_vxc_nn(model, rho, sigma, features, ao_grid, grid_weights) -> jnp.ndarray:
    """Assemble NN XC potential matrix V_xc via per-point forward-mode jvp.

    Returns shape (n_ao, n_ao). LDA-like approximation (v_sigma discarded).
    """
    def exc_single_point(r, s, f):
        return model.eval_exc_scalar(r, s, f)

    # Per-point jvp: tangent on rho only
    v_rho = jax.vmap(
        lambda r, s, f: jax.jvp(
            exc_single_point,
            (r, s, f),
            (jnp.ones_like(r), jnp.zeros_like(s), jnp.zeros_like(f)),
        )[1]
    )(rho, sigma, features)

    # Assemble Fock-matrix form: V_xc_ij = sum_g v_rho[g] * ao[g,i] * ao[g,j] * w[g]
    return jnp.einsum("g,gi,gj,g->ij", v_rho, ao_grid, ao_grid, grid_weights)


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
    vxc_nn = compute_vxc_nn(
        model,
        mol_data["rho_grid"],
        mol_data["sigma_grid"],
        features,
        mol_data["ao_grid"],
        mol_data["grid_weights"],
    )

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

        # UKS: j_pbe has shape (2, n_ao, n_ao)
        # Build per-spin Fock matrices
        fock_a = h_core + j_pbe[0] + j_pbe[1] + vxc_nn
        fock_b = h_core + j_pbe[0] + j_pbe[1] + vxc_nn

        # Transform to orthogonal basis
        fock_orth_a = L_inv @ fock_a @ L_inv.T + DEGENERACY_REG * jnp.eye(nao)
        fock_orth_b = L_inv @ fock_b @ L_inv.T + DEGENERACY_REG * jnp.eye(nao)

        # Eigendecomposition
        _, mo_coeff_orth_a = jnp.linalg.eigh(fock_orth_a)
        _, mo_coeff_orth_b = jnp.linalg.eigh(fock_orth_b)

        # Back-transform
        mo_coeff_a = L_inv.T @ mo_coeff_orth_a
        mo_coeff_b = L_inv.T @ mo_coeff_orth_b

        # Density matrices (no factor of 2 for UKS)
        C_occ_a = mo_coeff_a[:, :nocc_a]
        C_occ_b = mo_coeff_b[:, :nocc_b]
        dm_a = C_occ_a @ C_occ_a.T
        dm_b = C_occ_b @ C_occ_b.T
        dm_pred = jnp.stack([dm_a, dm_b])
    else:
        nocc = mol_data["nocc"]

        # RKS: j_pbe has shape (n_ao, n_ao)
        fock = h_core + j_pbe + vxc_nn

        # Transform to orthogonal basis
        fock_orth = L_inv @ fock @ L_inv.T + DEGENERACY_REG * jnp.eye(nao)

        # Eigendecomposition
        _, mo_coeff_orth = jnp.linalg.eigh(fock_orth)

        # Back-transform
        mo_coeff = L_inv.T @ mo_coeff_orth

        # Density matrix (factor of 2 for RKS double occupation)
        C_occ = mo_coeff[:, :nocc]
        dm_pred = 2.0 * C_occ @ C_occ.T

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
