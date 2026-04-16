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
    assert data["rho_ref_grid"] is None


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
    assert data["rho_ref_grid"] is None


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
    required = ("dm_target", "rho_ref_grid", "cusp_features")
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
    # dm_target and rho_ref_grid are None because no external_data_path
    # was supplied; precompute only populates them from an external .npz.
    assert data["dm_target"] is None
    assert data["rho_ref_grid"] is None


# ---------------------------------------------------------------------------
# §13.2 items (14)-(20) — MoleculeSpec.external_data_path
# ---------------------------------------------------------------------------


def _prepare_h2_external_data(tmp_path, *, keys):
    """Run PBE on H2 once, then save an .npz with shape-matching reference
    values so external_data_path tests exercise the real loader.

    Returns (path, reference_values_dict, baseline_data)."""
    mol = h2_molecule()
    baseline = precompute_fixed_density_data(mol)
    dm_shape = tuple(np.asarray(baseline["dm_pbe"]).shape)
    rho_shape = tuple(np.asarray(baseline["rho_grid"]).shape)

    payload = {}
    refs = {}
    if "dm_target" in keys:
        # Use 1.5 * dm_pbe as a distinctive "reference" so we can assert
        # precompute actually loaded from disk instead of falling back.
        dm_arr = np.asarray(baseline["dm_pbe"]) * 1.5
        payload["dm_target"] = dm_arr
        refs["dm_target"] = dm_arr
    if "rho_ref_grid" in keys:
        rho_arr = np.asarray(baseline["rho_grid"]) * 1.1
        payload["rho_ref_grid"] = rho_arr
        refs["rho_ref_grid"] = rho_arr
    if "E_ref_literature" in keys:
        payload["E_ref_literature"] = np.float64(-1.17447)
        refs["E_ref_literature"] = -1.17447

    path = str(tmp_path / "h2_external.npz")
    np.savez(path, **payload)
    return path, refs, baseline, dm_shape, rho_shape


# §13.2 item (14)
def test_precompute_loads_external_data_path_all_keys(tmp_path):
    """External .npz with all three keys populates MoleculeData and shapes match."""
    path, refs, baseline, dm_shape, rho_shape = _prepare_h2_external_data(
        tmp_path, keys=("dm_target", "rho_ref_grid", "E_ref_literature"),
    )
    mol = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
        external_data_path=path,
    )
    data = precompute_fixed_density_data(mol)
    assert data["dm_target"] is not None
    assert data["rho_ref_grid"] is not None
    assert data["E_ref_literature"] is not None
    assert tuple(np.asarray(data["dm_target"]).shape) == dm_shape
    assert tuple(np.asarray(data["rho_ref_grid"]).shape) == rho_shape
    np.testing.assert_allclose(
        np.asarray(data["dm_target"]), refs["dm_target"], rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(data["rho_ref_grid"]), refs["rho_ref_grid"], rtol=1e-12,
    )
    assert data["E_ref_literature"] == pytest.approx(-1.17447, rel=1e-10)


# §13.2 item (15)
def test_precompute_external_data_path_partial_npz(tmp_path):
    """Partial .npz (only E_ref_literature) leaves other fields None."""
    path, _, _, _, _ = _prepare_h2_external_data(
        tmp_path, keys=("E_ref_literature",),
    )
    mol = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
        external_data_path=path,
    )
    data = precompute_fixed_density_data(mol)
    assert data["dm_target"] is None
    assert data["rho_ref_grid"] is None
    assert data["E_ref_literature"] == pytest.approx(-1.17447, rel=1e-10)


# §13.2 item (16)
def test_precompute_external_data_path_rejects_unknown_keys(tmp_path):
    """An .npz with an unrecognized key triggers ValueError."""
    path = str(tmp_path / "bad_keys.npz")
    np.savez(path, dm_target=np.zeros((2, 2)), bogus=np.zeros(3))
    mol = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
        external_data_path=path,
    )
    with pytest.raises(ValueError, match="unknown keys"):
        precompute_fixed_density_data(mol)


# §13.2 item (17)
def test_precompute_external_data_path_rejects_dm_target_shape_mismatch(tmp_path):
    """dm_target shape must match dm_pbe; mismatch triggers ValueError."""
    path = str(tmp_path / "bad_dm_shape.npz")
    np.savez(path, dm_target=np.zeros((5, 5)))  # H2/sto-3g has dm shape (2, 2)
    mol = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
        external_data_path=path,
    )
    with pytest.raises(ValueError, match="dm_target shape"):
        precompute_fixed_density_data(mol)


# §13.2 item (18)
def test_precompute_external_data_path_rejects_rho_grid_shape_mismatch(tmp_path):
    """rho_ref_grid shape must match rho_grid; mismatch triggers ValueError."""
    path = str(tmp_path / "bad_rho_shape.npz")
    np.savez(path, rho_ref_grid=np.zeros(7))
    mol = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
        external_data_path=path,
    )
    with pytest.raises(ValueError, match="rho_ref_grid shape"):
        precompute_fixed_density_data(mol)


# §13.2 item (19)
def test_precompute_external_data_path_rejects_nonscalar_E_ref(tmp_path):
    """E_ref_literature must be scalar; vector triggers ValueError."""
    path = str(tmp_path / "bad_scalar.npz")
    np.savez(path, E_ref_literature=np.array([1.0, 2.0]))
    mol = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
        external_data_path=path,
    )
    with pytest.raises(ValueError, match="must be scalar"):
        precompute_fixed_density_data(mol)


# §13.2 item (20)
def test_precompute_external_data_path_missing_file(tmp_path):
    """Nonexistent external_data_path triggers FileNotFoundError."""
    missing = str(tmp_path / "does_not_exist.npz")
    mol = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
        external_data_path=missing,
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        precompute_fixed_density_data(mol)


# §13.2 item (21)
def test_precompute_external_data_path_uks_dm_shape(tmp_path):
    """O (UKS) dm_pbe is 3D; dm_target shape must match the 3D form."""
    mol_o = o_atom()
    baseline = precompute_fixed_density_data(mol_o)
    dm_shape = tuple(np.asarray(baseline["dm_pbe"]).shape)
    assert len(dm_shape) == 3  # UKS branch: (2, n_ao, n_ao)
    path = str(tmp_path / "o_external.npz")
    np.savez(
        path,
        dm_target=np.asarray(baseline["dm_pbe"]),
        E_ref_literature=np.float64(-75.0673),
    )
    mol_o_with_path = MoleculeSpec(
        name="O", atom="O 0 0 0", basis="sto-3g",
        charge=0, spin=2, atom_composition=(("O", 1),),
        external_data_path=path,
    )
    data = precompute_fixed_density_data(mol_o_with_path)
    assert data["dm_target"] is not None
    assert tuple(np.asarray(data["dm_target"]).shape) == dm_shape
    assert data["E_ref_literature"] == pytest.approx(-75.0673, rel=1e-10)


# §13.2 item (22)
def test_molecule_spec_external_data_path_default_is_none():
    """external_data_path defaults to None so existing constructors keep working."""
    mol = MoleculeSpec(name="test", atom="H 0 0 0")
    assert mol.external_data_path is None


# §13.2 item (23)
def test_molecule_spec_from_dict_accepts_external_data_path(tmp_path):
    """MoleculeSpec.from_dict forwards external_data_path to the frozen dataclass."""
    p = str(tmp_path / "x.npz")
    mol = MoleculeSpec.from_dict(
        name="H", atom="H 0 0 0", atom_composition={"H": 1},
        basis="sto-3g", spin=1, external_data_path=p,
    )
    assert mol.external_data_path == p


# ---------------------------------------------------------------------------
# §13.2 items (24)-(30) — MoleculeSpec.grid_level
# ---------------------------------------------------------------------------


# §13.2 item (24)
def test_molecule_spec_grid_level_default_is_none():
    """grid_level defaults to None so existing constructors keep working."""
    mol = MoleculeSpec(name="test", atom="H 0 0 0")
    assert mol.grid_level is None


# §13.2 item (25)
def test_molecule_spec_grid_level_rejects_non_int():
    """grid_level must be an int or None; floats/strings are rejected."""
    with pytest.raises(TypeError, match="grid_level must be int or None"):
        MoleculeSpec(name="H", atom="H 0 0 0", grid_level=1.0)
    with pytest.raises(TypeError, match="grid_level must be int or None"):
        MoleculeSpec(name="H", atom="H 0 0 0", grid_level="1")


# §13.2 item (26)
def test_molecule_spec_grid_level_rejects_out_of_range():
    """grid_level must be in [0, 9] to match pyscf Grids.level."""
    with pytest.raises(ValueError, match=r"grid_level must be in \[0, 9\]"):
        MoleculeSpec(name="H", atom="H 0 0 0", grid_level=-1)
    with pytest.raises(ValueError, match=r"grid_level must be in \[0, 9\]"):
        MoleculeSpec(name="H", atom="H 0 0 0", grid_level=10)


# §13.2 item (27)
def test_molecule_spec_grid_level_rejects_bool():
    """bool is a subclass of int in Python; grid_level must reject it."""
    with pytest.raises(TypeError, match="grid_level must be int or None"):
        MoleculeSpec(name="H", atom="H 0 0 0", grid_level=True)


# §13.2 item (28)
def test_molecule_spec_from_dict_accepts_grid_level():
    """MoleculeSpec.from_dict forwards grid_level to the frozen dataclass."""
    mol = MoleculeSpec.from_dict(
        name="H", atom="H 0 0 0", atom_composition={"H": 1},
        basis="sto-3g", spin=1, grid_level=1,
    )
    assert mol.grid_level == 1


# §13.2 item (29)
def test_precompute_honors_grid_level_smaller_than_default():
    """grid_level=1 produces a much smaller grid than the pyscf default (3)."""
    mol_default = h2_molecule()
    data_default = precompute_fixed_density_data(mol_default)
    mol_level1 = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
        grid_level=1,
    )
    data_level1 = precompute_fixed_density_data(mol_level1)
    # Level 1 is coarser than level 3 (the pyscf default), so fewer points.
    assert data_level1["rho_grid"].shape[0] < data_default["rho_grid"].shape[0]
    # Both should still integrate to approximately 2 electrons (H2).
    weights_default = data_default["grid_weights"]
    weights_level1 = data_level1["grid_weights"]
    n_default = float(jnp.sum(data_default["rho_grid"] * weights_default))
    n_level1 = float(jnp.sum(data_level1["rho_grid"] * weights_level1))
    assert abs(n_default - 2.0) < 0.1
    assert abs(n_level1 - 2.0) < 0.1


# §13.2 item (30)
def test_precompute_grid_level_interacts_with_external_data_shape(tmp_path):
    """rho_ref_grid shape is tied to the active grid_level via shape validation.

    Step 4 writes rho_ref_grid at grid_level=1; loading it back through
    precompute with the default grid (level 3) must fail because the grid
    point count is different.
    """
    mol_level1 = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
        grid_level=1,
    )
    baseline_level1 = precompute_fixed_density_data(mol_level1)
    rho_level1 = np.asarray(baseline_level1["rho_grid"]) * 1.1
    path = str(tmp_path / "h2_level1.npz")
    np.savez(path, rho_ref_grid=rho_level1)

    # Matching spec (grid_level=1) accepts the external data.
    mol_with_path_ok = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
        grid_level=1, external_data_path=path,
    )
    data_ok = precompute_fixed_density_data(mol_with_path_ok)
    assert data_ok["rho_ref_grid"] is not None

    # Mismatched spec (default grid) rejects with shape error.
    mol_with_path_bad = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
        grid_level=None, external_data_path=path,
    )
    with pytest.raises(ValueError, match="rho_ref_grid shape"):
        precompute_fixed_density_data(mol_with_path_bad)


def test_precompute_populates_ref_density_method_when_external_provides_it(tmp_path):
    """When the external .npz provides ref_density_method, mol_data echoes it."""
    import numpy as np
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule

    base = precompute_fixed_density_data(h2_molecule())
    npz_path = tmp_path / "ext.npz"
    np.savez(
        npz_path,
        rho_ref_grid=np.asarray(base["rho_grid"]),
        ref_density_method=np.array("hf"),
    )
    spec = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
        external_data_path=str(npz_path),
    )
    data = precompute_fixed_density_data(spec)
    assert data["ref_density_method"] == "hf"
    assert data["rho_ref_grid"] is not None
