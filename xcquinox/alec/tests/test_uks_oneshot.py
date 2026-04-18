"""Tests for UKS (spin-polarized) oneshot DM/energy.

Regression tests for Task 10 of the alec physics-fixes plan: UKS oneshot must
produce spin-resolved Fock matrices (V_xc^NN_a != V_xc^NN_b for open-shell)
so the predicted alpha/beta density matrices meaningfully differ.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import xcquinox.alec as alec
from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.oneshot import oneshot_dm_prediction_fast


def _build_o_atom_uks():
    spec = MoleculeSpec(
        name="O", atom="O 0 0 0", basis="sto-3g",
        charge=0, spin=2, atom_composition=(("O", 1),), grid_level=1,
    )
    md = precompute_fixed_density_data(spec)
    arch = alec.get_architecture("deep")
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    model = alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
    return spec, md, model


def _extract_dm(result):
    """Accept either raw-DM return or dict-wrapped return."""
    if isinstance(result, dict):
        return np.asarray(result["dm_predicted"])
    return np.asarray(result)


def test_uks_oneshot_produces_spin_resolved_dm():
    """For O atom (spin=2), oneshot must yield (2, nao, nao) DM that is NOT
    spin-identical (the two channels must differ meaningfully)."""
    spec, md, model = _build_o_atom_uks()
    result = oneshot_dm_prediction_fast(model, md)
    dm = _extract_dm(result)
    assert dm.ndim == 3, f"expected (2, nao, nao), got {dm.shape}"
    assert dm.shape[0] == 2
    diff = float(np.max(np.abs(dm[0] - dm[1])))
    assert diff > 1e-4, f"UKS DMs are suspiciously equal: max_diff={diff}"


def test_uks_oneshot_total_electrons_correct():
    """Trace(S @ DM_s) should equal n_electrons for each spin channel
    (n_alpha=5, n_beta=3 for O atom)."""
    spec, md, model = _build_o_atom_uks()
    result = oneshot_dm_prediction_fast(model, md)
    dm = _extract_dm(result)
    s = np.asarray(md["s_matrix"])
    n_a = float(np.trace(s @ dm[0]))
    n_b = float(np.trace(s @ dm[1]))
    assert abs(n_a - 5.0) < 0.1, f"n_alpha = {n_a}, expected ~5"
    assert abs(n_b - 3.0) < 0.1, f"n_beta = {n_b}, expected ~3"
    assert abs(n_a + n_b - 8.0) < 0.1, f"total = {n_a + n_b}, expected 8"


def test_uks_oneshot_fock_alpha_differs_from_beta():
    """Regression test for the Task 10 bug: previously Fock_a == Fock_b because
    the same (spin-blind) V_xc^NN was added to both spin channels, differing
    only via nocc slicing. After the fix, V_xc^NN_a and V_xc^NN_b are
    spin-resolved (built from 2*rho_s via the spin-scaling trick), so
    Fock_a != Fock_b for open-shell systems.

    We reconstruct the two Fock matrices the same way oneshot_dm_prediction_fast
    does internally, and assert they differ meaningfully."""
    from xcquinox.alec.oneshot import _uks_spin_resolved_vxc
    from xcquinox.alec.descriptors import assemble_descriptor_features
    spec, md, model = _build_o_atom_uks()
    features = assemble_descriptor_features(model.descriptors, md)
    vxc_a, vxc_b = _uks_spin_resolved_vxc(model, md, features)
    h_core = np.asarray(md["h_core"])
    j = np.asarray(md["j_matrix"])
    j_total = j[0] + j[1]
    fock_a = h_core + j_total + np.asarray(vxc_a)
    fock_b = h_core + j_total + np.asarray(vxc_b)
    diff = float(np.max(np.abs(fock_a - fock_b)))
    assert diff > 1e-6, (
        f"Fock_a and Fock_b are identical (max|diff|={diff:.2e}); "
        f"V_xc^NN is not spin-resolved (Task 10 bug)."
    )


def test_uks_oneshot_fock_spin_resolved():
    """Directly verify that V_xc^NN differs between alpha and beta channels
    when rho_a != rho_b. This is the physics fix: spin-scaled UKS V_xc."""
    spec, md, model = _build_o_atom_uks()
    # Compute per-spin rho and sigma via the same recipe oneshot uses internally.
    dm_pbe = np.asarray(md["dm_pbe"])
    ao_grid = np.asarray(md["ao_grid"])
    assert dm_pbe.ndim == 3 and dm_pbe.shape[0] == 2
    rho_a = np.einsum("ij,gi,gj->g", dm_pbe[0], ao_grid, ao_grid)
    rho_b = np.einsum("ij,gi,gj->g", dm_pbe[1], ao_grid, ao_grid)
    # Oxygen is open-shell; alpha and beta densities must differ.
    assert float(np.max(np.abs(rho_a - rho_b))) > 1e-3

    from xcquinox.alec.oneshot import compute_vxc_nn
    from xcquinox.alec.descriptors import assemble_descriptor_features
    features = assemble_descriptor_features(model.descriptors, md)
    ao_grid_deriv = md["ao_grid_deriv"]
    grid_weights = md["grid_weights"]
    ao_xyz = np.asarray(ao_grid_deriv)[1:4]
    nabla_rho_a = 2.0 * np.einsum("ij,dgi,gj->gd", dm_pbe[0], ao_xyz, ao_grid)
    nabla_rho_b = 2.0 * np.einsum("ij,dgi,gj->gd", dm_pbe[1], ao_xyz, ao_grid)
    sigma_aa = np.sum(nabla_rho_a * nabla_rho_a, axis=1)
    sigma_bb = np.sum(nabla_rho_b * nabla_rho_b, axis=1)

    vxc_a = compute_vxc_nn(
        model, jnp.asarray(2.0 * rho_a), jnp.asarray(2.0 * sigma_aa),
        features, ao_grid, grid_weights,
        nabla_rho=jnp.asarray(2.0 * nabla_rho_a), ao_grad=ao_grid_deriv,
    )
    vxc_b = compute_vxc_nn(
        model, jnp.asarray(2.0 * rho_b), jnp.asarray(2.0 * sigma_bb),
        features, ao_grid, grid_weights,
        nabla_rho=jnp.asarray(2.0 * nabla_rho_b), ao_grad=ao_grid_deriv,
    )
    diff = float(np.max(np.abs(np.asarray(vxc_a) - np.asarray(vxc_b))))
    assert diff > 1e-6, f"spin-scaled V_xc must differ for open-shell O: {diff}"
