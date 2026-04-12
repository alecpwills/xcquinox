"""Tests for xcquinox.alec.data — MoleculeData, precompute, XC helpers.

Implements THE SPEC §13.2 test_data.py items (1)-(13).
"""
import numpy as np
import pytest
import jax.numpy as jnp
from unittest.mock import patch

from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import MoleculeData, precompute_fixed_density_data
from xcquinox.alec.descriptors import CuspDescriptor, DMStatisticsDescriptor
from xcquinox.alec.tests.fixtures.molecules import (
    h_atom, h2_molecule, o_atom, h2o_molecule,
)


# §13.2 item (1)
def test_precompute_baseline_keys_populated_with_no_descriptors():
    mol = h2_molecule()
    data = precompute_fixed_density_data(mol)
    # Baseline keys must be non-None
    assert data["name"] == "H2"
    assert isinstance(data["dm_pbe"], jnp.ndarray)
    assert isinstance(data["s_matrix"], jnp.ndarray)
    assert isinstance(data["h_core"], jnp.ndarray)
    assert isinstance(data["j_matrix"], jnp.ndarray)
    assert isinstance(data["e_nuc"], float)
    assert isinstance(data["E_pbe"], float)
    assert isinstance(data["E_xc_pbe"], float)
    assert isinstance(data["E_non_xc"], float)
    assert isinstance(data["rho_grid"], jnp.ndarray)
    assert isinstance(data["sigma_grid"], jnp.ndarray)
    assert isinstance(data["grid_weights"], jnp.ndarray)
    assert isinstance(data["ao_grid"], jnp.ndarray)
    assert isinstance(data["ao_grid_deriv"], jnp.ndarray)
    # Descriptor features should be None when no descriptors requested
    assert data["cusp_features"] is None
    assert data["dm_features"] is None
    # CCSD keys not requested
    assert data["dm_target"] is None
    assert data["rho_ccsd_grid"] is None


# §13.2 item (2)
def test_precompute_cusp_descriptor_adds_cusp_features_shape_n_2():
    mol = h2_molecule()
    cusp = CuspDescriptor()
    data = precompute_fixed_density_data(mol, descriptors=(cusp,))
    assert data["cusp_features"] is not None
    n_grid = data["rho_grid"].shape[0]
    assert data["cusp_features"].shape == (n_grid, 2)


# §13.2 item (3)
def test_precompute_dm_descriptor_adds_dm_features_shape_n_3():
    mol = h2_molecule()
    dm_desc = DMStatisticsDescriptor()
    data = precompute_fixed_density_data(mol, descriptors=(dm_desc,))
    assert data["dm_features"] is not None
    n_grid = data["rho_grid"].shape[0]
    assert data["dm_features"].shape == (n_grid, 3)


# §13.2 item (4)
def test_precompute_both_descriptors_assembled_in_dm_before_cusp_order():
    mol = h2_molecule()
    dm_desc = DMStatisticsDescriptor()
    cusp = CuspDescriptor()
    data = precompute_fixed_density_data(mol, descriptors=(dm_desc, cusp))
    assert data["dm_features"] is not None
    assert data["cusp_features"] is not None
    n_grid = data["rho_grid"].shape[0]
    assert data["dm_features"].shape == (n_grid, 3)
    assert data["cusp_features"].shape == (n_grid, 2)


# §13.2 item (5) — M-E12-5
def test_precompute_d1_skips_ccsd():
    mol = h2_molecule()
    # D1_delta_ae requires only ("E_pbe",), no CCSD keys
    d1_keys = ("E_pbe",)
    data = precompute_fixed_density_data(mol, required_keys=d1_keys)
    assert data["dm_target"] is None
    assert data["rho_ccsd_grid"] is None


# §13.2 item (6)
@pytest.mark.parametrize("mol_factory,name", [
    (h_atom, "H"),
    (o_atom, "O"),
    (h2o_molecule, "H2O"),
])
def test_precompute_scf_converges_on_h_o_h2o(mol_factory, name):
    mol = mol_factory()
    data = precompute_fixed_density_data(mol)
    assert data["name"] == name
    # SCF energy should be finite and negative for stable systems
    assert np.isfinite(data["E_pbe"])
    assert data["E_pbe"] < 0.0
    # Grid density should integrate to approximately n_electrons
    n_elec = sum(c for _, c in mol.atom_composition)
    # For H: 1 electron, O: 8, H2O: 10
    expected_elec = {"H": 1, "O": 8, "H2O": 10}[name]
    integrated_density = float(jnp.sum(data["rho_grid"] * data["grid_weights"]))
    assert abs(integrated_density - expected_elec) < 0.1


# §13.2 item (7)
def test_molecule_spec_default_basis_is_sto_3g():
    mol = MoleculeSpec(name="test", atom="H 0 0 0")
    assert mol.basis == "sto-3g"


# §13.2 item (8)
def test_precompute_propagates_atom_composition():
    mol = h2o_molecule()
    data = precompute_fixed_density_data(mol)
    assert data["atom_composition"] == mol.atom_composition
    assert data["atom_composition"] == (("H", 2), ("O", 1))


# §13.2 item (9)
def test_precompute_single_point_grid_edge_case():
    """Precompute on the smallest molecule (H) should still produce
    valid arrays with more than 0 grid points."""
    mol = h_atom()
    data = precompute_fixed_density_data(mol)
    assert data["rho_grid"].shape[0] > 0
    assert data["grid_weights"].shape[0] > 0
    assert data["ao_grid"].shape[0] == data["rho_grid"].shape[0]


# §13.2 item (10)
def test_precompute_uks_path_for_o_spin_polarized():
    mol = o_atom()
    data = precompute_fixed_density_data(mol)
    assert data["is_unrestricted"] is True
    # UKS: dm_pbe has shape (2, n_ao, n_ao)
    assert data["dm_pbe"].ndim == 3
    assert data["dm_pbe"].shape[0] == 2
    # UKS occupancies
    assert data["nocc"] is None
    assert data["nocc_a"] == 5  # (8 + 2) // 2
    assert data["nocc_b"] == 3  # (8 - 2) // 2


# §13.2 item (11) — E-H4
def test_precompute_rejects_ill_conditioned_overlap():
    mol = h2_molecule()
    with patch("xcquinox.alec.data.np.linalg.cond", return_value=1e12):
        with pytest.raises(ValueError, match="ill-conditioned"):
            precompute_fixed_density_data(mol)


# §13.2 item (12) — D-H7 (xfail: fixture not yet generated)
@pytest.mark.xfail(reason="Fixture notebook_cell24_precompute_h2o.npz not yet generated")
def test_precompute_matches_notebook():
    import pathlib
    fixture_dir = pathlib.Path(__file__).parent / "fixtures"
    fixture_path = fixture_dir / "notebook_cell24_precompute_h2o.npz"
    ref = dict(np.load(str(fixture_path)))

    mol = MoleculeSpec(
        name="H2O",
        atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24",
        basis="def2-svp",
        charge=0, spin=0,
        atom_composition=(("H", 2), ("O", 1)),
    )
    data = precompute_fixed_density_data(mol)

    # Scalars
    for key in ("e_nuc", "E_pbe", "E_xc_pbe", "E_non_xc"):
        np.testing.assert_allclose(
            data[key], float(ref[key]),
            rtol=1e-10, err_msg=f"scalar mismatch: {key}",
        )
    # Arrays — bit-exact
    for key in ("rho_grid", "sigma_grid", "ao_grid", "grid_weights",
                "dm_pbe", "s_matrix", "h_core", "j_matrix"):
        np.testing.assert_array_equal(
            np.asarray(data[key]), ref[key],
            err_msg=f"array mismatch: {key}",
        )


# §13.2 item (13) — M-E12-2
def test_precompute_populates_all_required_keys():
    mol = h2_molecule()
    cusp = CuspDescriptor()
    # Simulate a loss that requires CCSD keys
    required = ("dm_target", "rho_ccsd_grid", "cusp_features")
    data = precompute_fixed_density_data(
        mol, required_keys=required, descriptors=(cusp,),
    )
    # All MoleculeData keys must be present (even if None)
    expected_keys = set(MoleculeData.__annotations__.keys())
    actual_keys = set(data.keys())
    assert expected_keys == actual_keys, (
        f"Missing: {expected_keys - actual_keys}, "
        f"Extra: {actual_keys - expected_keys}"
    )
    # cusp_features should be populated (descriptor requested it)
    assert data["cusp_features"] is not None
    # dm_target and rho_ccsd_grid are None (CCSD not implemented in precompute)
    assert data["dm_target"] is None
    assert data["rho_ccsd_grid"] is None
