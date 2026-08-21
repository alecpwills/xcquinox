"""Tests for xcquinox.alec.data: MoleculeData, precompute, XC helpers.

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


# §13.2 item (3). Width 3 -> 2 on 2026-08-06: dm_entropy removed (no usable
# gradient at any converged density).
def test_precompute_dm_descriptor_adds_dm_features_shape_n_2():
    mol = h2_molecule()
    dm_desc = DMStatisticsDescriptor()
    data = precompute_fixed_density_data(mol, descriptors=(dm_desc,))
    assert data["dm_features"] is not None
    n_grid = data["rho_grid"].shape[0]
    assert data["dm_features"].shape == (n_grid, 2)


# §13.2 item (4)
def test_precompute_both_descriptors_assembled_in_dm_before_cusp_order():
    mol = h2_molecule()
    dm_desc = DMStatisticsDescriptor()
    cusp = CuspDescriptor()
    data = precompute_fixed_density_data(mol, descriptors=(dm_desc, cusp))
    assert data["dm_features"] is not None
    assert data["cusp_features"] is not None
    n_grid = data["rho_grid"].shape[0]
    assert data["dm_features"].shape == (n_grid, 2)
    assert data["cusp_features"].shape == (n_grid, 2)


# §13.2 item (5), M-E12-5
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


# §13.2 item (11), E-H4
def test_precompute_rejects_ill_conditioned_overlap():
    mol = h2_molecule()
    with patch("xcquinox.alec.data.np.linalg.cond", return_value=1e12):
        with pytest.raises(ValueError, match="ill-conditioned"):
            precompute_fixed_density_data(mol)


# §13.2 item (12), D-H7 (xfail: fixture not yet generated)
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
    # Arrays, bit-exact
    for key in ("rho_grid", "sigma_grid", "ao_grid", "grid_weights",
                "dm_pbe", "s_matrix", "h_core", "j_matrix"):
        np.testing.assert_array_equal(
            np.asarray(data[key]), ref[key],
            err_msg=f"array mismatch: {key}",
        )


# §13.2 item (13), M-E12-2
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
# §13.2 items (14)-(20), MoleculeSpec.external_data_path
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
# §13.2 items (24)-(30), MoleculeSpec.grid_level
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


def test_precompute_populates_eri_when_requested():
    """When required_keys includes 'eri', precompute stashes the 4-index ERI tensor."""
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule

    data = precompute_fixed_density_data(h2_molecule(), required_keys=("eri",))
    assert "eri" in data
    nao = data["h_core"].shape[0]
    assert data["eri"].shape == (nao, nao, nao, nao)


def test_precompute_eri_absent_by_default():
    """Without 'eri' in required_keys, data['eri'] is None."""
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule

    data = precompute_fixed_density_data(h2_molecule())
    assert data.get("eri") is None


def test_precompute_loads_vxc_ref_from_external_npz(tmp_path):
    """vxc_ref in external .npz is loaded and shape-validated against vxc_pbe."""
    mol = h2_molecule()
    baseline = precompute_fixed_density_data(mol)
    vxc_shape = tuple(np.asarray(baseline["vxc_pbe"]).shape)
    vxc_ref_arr = np.random.default_rng(0).standard_normal(vxc_shape)
    path = str(tmp_path / "with_vxc_ref.npz")
    np.savez(path, vxc_ref=vxc_ref_arr)
    mol_with_path = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
        external_data_path=path,
    )
    data = precompute_fixed_density_data(mol_with_path)
    assert data["vxc_ref"] is not None
    np.testing.assert_allclose(
        np.asarray(data["vxc_ref"]), vxc_ref_arr, rtol=1e-10,
    )


def test_precompute_rejects_vxc_ref_shape_mismatch(tmp_path):
    """vxc_ref shape must match vxc_pbe; mismatch triggers ValueError."""
    path = str(tmp_path / "bad_vxc_shape.npz")
    np.savez(path, vxc_ref=np.zeros((5, 5)))
    mol = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
        external_data_path=path,
    )
    with pytest.raises(ValueError, match="vxc_ref shape"):
        precompute_fixed_density_data(mol)


def test_precompute_vxc_ref_none_when_absent():
    """Without external_data_path, vxc_ref is None."""
    mol = h2_molecule()
    data = precompute_fixed_density_data(mol)
    assert data["vxc_ref"] is None


def test_vxc_pbe_uks_oxygen_atom_magnitude_reasonable():
    """For UKS O atom, vxc_pbe magnitude should be O(1) Hartree, not O(10).
    The old bug (subtracting per-spin J instead of J_total) gave ~7 Ha errors
    from J[dm_other] leakage into the supposedly-V_xc matrix.
    """
    import numpy as np
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.config import MoleculeSpec
    spec = MoleculeSpec(
        name="O", atom="O 0 0 0", basis="sto-3g",
        charge=0, spin=2, atom_composition=(("O", 1),), grid_level=1,
    )
    md = precompute_fixed_density_data(spec, required_keys=("vxc_pbe",))
    vxc = md["vxc_pbe"]
    assert vxc.shape == (2, 5, 5), f"unexpected shape: {vxc.shape}"
    # UKS V_xc matrix magnitude should be < 5 Hartree (O atom in sto-3g),
    # after correct J_total subtraction.
    max_abs = float(np.max(np.abs(vxc)))
    assert max_abs < 5.0, (
        f"|vxc_pbe|_max = {max_abs:.3f} Ha -- too large; "
        f"likely the old J-per-spin bug is back"
    )


def test_vxc_pbe_uks_matches_direct_pyscf_vxc():
    """UKS vxc_pbe should equal veff - J_total when evaluated at the same DM.

    Uses the DM from precompute_fixed_density_data to avoid SCF-convergence
    drift between independent pyscf runs. Tests the J-handling formula,
    not the SCF algorithm.
    """
    import numpy as np
    from pyscf import gto, dft
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.config import MoleculeSpec

    spec = MoleculeSpec(
        name="O", atom="O 0 0 0", basis="sto-3g",
        charge=0, spin=2, atom_composition=(("O", 1),), grid_level=1,
    )
    md = precompute_fixed_density_data(spec, required_keys=("vxc_pbe",))
    # Use the UKS DM actually stored by precompute (shape (2, nao, nao))
    dm = np.asarray(md["dm_pbe"])
    assert dm.shape == (2, 5, 5), f"expected UKS dm_pbe, got {dm.shape}"

    # Now evaluate V_xc directly at this same DM
    mol = gto.M(atom="O 0 0 0", basis="sto-3g", spin=2, verbose=0)
    mf = dft.UKS(mol); mf.xc = "pbe"; mf.grids.level = 1; mf.build()
    veff = np.asarray(mf.get_veff(mol, dm))
    j_per_spin = np.asarray(mf.get_j(mol, dm))
    j_total = j_per_spin.sum(axis=0)
    vxc_direct = veff - j_total[np.newaxis, ...]

    max_diff = float(np.max(np.abs(np.asarray(md["vxc_pbe"]) - vxc_direct)))
    assert max_diff < 1e-8, (
        f"vxc_pbe does not match veff - J_total at the same DM: "
        f"max diff = {max_diff:.3e}"
    )


def test_e_xc_pbe_uks_matches_pyscf_veff_exc():
    """E_xc_pbe for UKS O atom should match mf.get_veff(...).exc."""
    import numpy as np
    from pyscf import gto, dft
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.config import MoleculeSpec

    spec = MoleculeSpec(
        name="O", atom="O 0 0 0", basis="sto-3g",
        charge=0, spin=2, atom_composition=(("O", 1),), grid_level=1,
    )
    md = precompute_fixed_density_data(
        spec, required_keys=("E_xc_pbe", "vxc_pbe"))

    # Use same DM precompute used
    dm = np.asarray(md["dm_pbe"])
    mol = gto.M(atom="O 0 0 0", basis="sto-3g", spin=2, verbose=0)
    mf = dft.UKS(mol); mf.xc = "pbe"; mf.grids.level = 1; mf.build()
    veff = mf.get_veff(mol, dm)
    e_xc_pyscf = float(veff.exc)

    assert abs(float(md["E_xc_pbe"]) - e_xc_pyscf) < 1e-6, (
        f"E_xc_pbe mismatch: md={md['E_xc_pbe']:.6f}, pyscf={e_xc_pyscf:.6f}, "
        f"diff={abs(md['E_xc_pbe'] - e_xc_pyscf):.3e}"
    )


# ---------------------------------------------------------------------------
# Precompute cache (2026-04-26 perf fix)
# ---------------------------------------------------------------------------

def test_precompute_cache_returns_same_object_on_second_call():
    """The process-level precompute cache must short-circuit a second call
    on the same MoleculeSpec. Without caching, a 72-spec eval sweep over
    5 molecules pays the full PBE SCF cost 72 times per molecule -- the
    primary cause of multi-hour eval runs the cache fixes.
    """
    from xcquinox.alec.data import (
        clear_precompute_cache, precompute_fixed_density_data,
    )
    clear_precompute_cache()
    mol = h2o_molecule()
    a = precompute_fixed_density_data(mol)
    b = precompute_fixed_density_data(mol)
    # Identity check: cache hit returns the SAME MoleculeData object,
    # not a structurally-equal recomputed copy.
    assert a is b


def test_precompute_cache_skip_when_disabled():
    """Disabling the cache must produce a fresh result on every call so
    callers that mutate external_data on disk get the latest precompute.
    """
    from xcquinox.alec.data import (
        clear_precompute_cache, precompute_fixed_density_data,
        set_precompute_cache_enabled,
    )
    clear_precompute_cache()
    set_precompute_cache_enabled(False)
    try:
        mol = h2o_molecule()
        a = precompute_fixed_density_data(mol)
        b = precompute_fixed_density_data(mol)
        assert a is not b
    finally:
        set_precompute_cache_enabled(True)
        clear_precompute_cache()


def test_precompute_cache_keys_on_required_keys_and_descriptors():
    """Different required_keys / descriptor sets must NOT collide in the
    cache -- a precompute requested with descriptors must have those
    descriptor outputs populated."""
    from xcquinox.alec.data import (
        clear_precompute_cache, precompute_fixed_density_data,
    )
    from xcquinox.alec.descriptors import CuspDescriptor
    clear_precompute_cache()
    mol = h2o_molecule()
    bare = precompute_fixed_density_data(mol)
    with_cusp = precompute_fixed_density_data(
        mol,
        required_keys=("cusp_features",),
        descriptors=(CuspDescriptor(),),
    )
    assert bare["cusp_features"] is None
    assert with_cusp["cusp_features"] is not None
    assert bare is not with_cusp


# ---------------------------------------------------------------------------
# grid_level_used provenance in external .npz is asserted against the
# resolved grid_level in _load_external_data.
# ---------------------------------------------------------------------------


def test_load_external_data_accepts_grid_level_used_key():
    """grid_level_used is an allowed key (CFG-03 provenance)."""
    from xcquinox.alec.data import _ALLOWED_EXTERNAL_KEYS
    assert "grid_level_used" in _ALLOWED_EXTERNAL_KEYS


def test_precompute_external_grid_level_match_ok(tmp_path):
    """When grid_level_used matches the resolved grid_level, load succeeds."""
    mol0 = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),), grid_level=1,
    )
    baseline = precompute_fixed_density_data(mol0)
    rho_arr = np.asarray(baseline["rho_grid"])
    path = str(tmp_path / "h2_glm.npz")
    np.savez(path, rho_ref_grid=rho_arr, grid_level_used=np.array(1))
    mol = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),), grid_level=1,
        external_data_path=path,
    )
    data = precompute_fixed_density_data(mol)
    assert data["rho_ref_grid"] is not None


def test_precompute_external_grid_level_mismatch_raises(tmp_path):
    """grid_level_used != resolved grid_level raises."""
    mol0 = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),), grid_level=1,
    )
    baseline = precompute_fixed_density_data(mol0)
    rho_arr = np.asarray(baseline["rho_grid"])
    path = str(tmp_path / "h2_glmis.npz")
    # File claims it was generated at grid_level=3 but consumer resolves to 1.
    np.savez(path, rho_ref_grid=rho_arr, grid_level_used=np.array(3))
    mol = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),), grid_level=1,
        external_data_path=path,
    )
    with pytest.raises(ValueError, match="grid_level"):
        precompute_fixed_density_data(mol)


def test_load_external_data_grid_level_used_function_direct(tmp_path):
    """Direct _load_external_data call: matching grid_level passes, mismatch
    raises. Exercises the loader contract without a full precompute."""
    from xcquinox.alec.data import _load_external_data
    path = str(tmp_path / "direct.npz")
    np.savez(path, grid_level_used=np.array(2))
    # Match: no raise.
    _load_external_data(
        path, dm_pbe_shape=(2, 2), rho_pbe_shape=(5,),
        vxc_pbe_shape=(2, 2), mol_name="H2", grid_level=2,
    )
    # Mismatch: raise.
    with pytest.raises(ValueError, match="grid_level"):
        _load_external_data(
            path, dm_pbe_shape=(2, 2), rho_pbe_shape=(5,),
            vxc_pbe_shape=(2, 2), mol_name="H2", grid_level=1,
        )


def test_load_external_data_guards_orientation_lock_mismatch(tmp_path):
    """A reference that RECORDS orientation_lock_strength must match the
    consumer's configured lock, else raise -- the load-time backstop for the
    cache-key gap that let an unlocked reference train against a locked
    functional (the degenerate OH/CH/NO radical density fix). Fires only when the
    ref carries the key and the consumer passes a lock; a None consumer or a
    legacy keyless ref does not raise."""
    from xcquinox.alec.data import _load_external_data
    path = str(tmp_path / "ol.npz")
    np.savez(path, orientation_lock_strength=np.array(3e-5))

    def _load(ol):
        return _load_external_data(
            path, dm_pbe_shape=(2, 2), rho_pbe_shape=(5,),
            vxc_pbe_shape=(2, 2), mol_name="OH", grid_level=1,
            orientation_lock_strength=ol)

    _load(3e-5)                                   # match -> no raise
    with pytest.raises(ValueError, match="orientation_lock"):
        _load(0.0)                                # unlocked consumer, locked ref
    # None consumer -> guard skipped (backward-compat for direct callers).
    _load_external_data(path, dm_pbe_shape=(2, 2), rho_pbe_shape=(5,),
                        vxc_pbe_shape=(2, 2), mol_name="OH", grid_level=1)
    # Legacy ref WITHOUT the key -> conservative guard does not fire.
    legacy = str(tmp_path / "legacy.npz")
    np.savez(legacy, grid_level_used=np.array(1))
    _load_external_data(legacy, dm_pbe_shape=(2, 2), rho_pbe_shape=(5,),
                        vxc_pbe_shape=(2, 2), mol_name="OH", grid_level=1,
                        orientation_lock_strength=3e-5)


# ---------------------------------------------------------------------------
# density-only benchmark reference npz (xcquinox.alec.benchmark_refs contract)
# ---------------------------------------------------------------------------

def test_precompute_loads_benchmark_density_only_npz(tmp_path):
    """The benchmark generator writes {rho_ref_grid, ref_density_method,
    grid_level_used, basis_used} and NOTHING else (no vxc_ref/dm_target --
    the OEP stage is a TRAINING-refs requirement). This must load cleanly
    with rho populated and the OEP keys None, and the grid_level identity
    gate must stay loud."""
    import dataclasses

    base = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),), grid_level=1,
    )
    baseline = precompute_fixed_density_data(base)
    rho_shape = tuple(np.asarray(baseline["rho_grid"]).shape)

    path = str(tmp_path / "H2.npz")
    np.savez_compressed(path, rho_ref_grid=np.full(rho_shape, 0.5),
                        ref_density_method=np.array("ccsd"),
                        grid_level_used=np.array(1),
                        basis_used=np.array("sto-3g"))
    data = precompute_fixed_density_data(
        dataclasses.replace(base, external_data_path=path))
    assert data["rho_ref_grid"] is not None
    assert data["ref_density_method"] == "ccsd"
    assert data["dm_target"] is None
    assert data["vxc_ref"] is None

    # a reference generated on a different grid must be rejected loudly
    bad = str(tmp_path / "H2_bad_grid.npz")
    np.savez_compressed(bad, rho_ref_grid=np.full(rho_shape, 0.5),
                        ref_density_method=np.array("ccsd"),
                        grid_level_used=np.array(2),
                        basis_used=np.array("sto-3g"))
    with pytest.raises(ValueError, match="grid_level=2"):
        precompute_fixed_density_data(
            dataclasses.replace(base, external_data_path=bad))

    # full benchmark contract incl the generator-side PBE density + weights
    # (informational; shape-gated, never returned into MoleculeData)
    full = str(tmp_path / "H2_full.npz")
    np.savez_compressed(full, rho_ref_grid=np.full(rho_shape, 0.5),
                        rho_pbe_grid=np.full(rho_shape, 0.4),
                        grid_weights=np.full(rho_shape, 0.1),
                        ref_density_method=np.array("ccsd"),
                        grid_level_used=np.array(1),
                        basis_used=np.array("sto-3g"))
    data_full = precompute_fixed_density_data(
        dataclasses.replace(base, external_data_path=full))
    assert data_full["rho_ref_grid"] is not None
    assert "rho_pbe_grid" not in data_full       # informational only

    bad_pbe = str(tmp_path / "H2_bad_pbe.npz")
    np.savez_compressed(bad_pbe, rho_ref_grid=np.full(rho_shape, 0.5),
                        rho_pbe_grid=np.zeros(3),
                        ref_density_method=np.array("ccsd"),
                        grid_level_used=np.array(1),
                        basis_used=np.array("sto-3g"))
    with pytest.raises(ValueError, match="rho_pbe_grid shape"):
        precompute_fixed_density_data(
            dataclasses.replace(base, external_data_path=bad_pbe))


def test_load_external_data_accepts_density_fit_used_key(tmp_path):
    """density_fit_used is an allowed, informational key: a stamped reference
    loads without an unknown-key rejection. The DF identity itself is enforced
    at generation by benchmark_refs._benchmark_npz_is_complete."""
    from xcquinox.alec.data import _ALLOWED_EXTERNAL_KEYS, _load_external_data
    assert "density_fit_used" in _ALLOWED_EXTERNAL_KEYS
    path = str(tmp_path / "dfstamp.npz")
    np.savez(path, rho_ref_grid=np.zeros(5), density_fit_used=np.array(True))
    got = _load_external_data(
        path, dm_pbe_shape=(2, 2), rho_pbe_shape=(5,),
        vxc_pbe_shape=(2, 2), mol_name="H2",
    )
    assert got[1] is not None  # rho_ref_grid loaded


# --------------------------------------------------------------------------- #
# dm_seed supply layer (per-rung SCF seeding)
# --------------------------------------------------------------------------- #
import os as _os

from xcquinox.alec.data import clear_precompute_cache


def _seed_env(monkeypatch, *, cache_dir=None, allow=False):
    if cache_dir is None:
        monkeypatch.delenv("XCQUINOX_SEED_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("XCQUINOX_SEED_CACHE_DIR", str(cache_dir))
    if allow:
        monkeypatch.setenv("XCQUINOX_SEED_ALLOW_GENERATE", "1")
    else:
        monkeypatch.delenv("XCQUINOX_SEED_ALLOW_GENERATE", raising=False)


def test_dm_seed_default_pbe_is_alias_of_dm_pbe():
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule
    clear_precompute_cache()
    md = precompute_fixed_density_data(h2_molecule())
    assert md["dm_seed"] is md["dm_pbe"]


def test_dm_seed_minao_differs_from_converged_and_leaves_rest_alone():
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule
    clear_precompute_cache()
    base = precompute_fixed_density_data(h2_molecule())
    cold = precompute_fixed_density_data(h2_molecule(), seed_source="minao")
    assert cold["dm_seed"].shape == cold["dm_pbe"].shape
    assert not np.allclose(np.asarray(cold["dm_seed"]),
                           np.asarray(cold["dm_pbe"]))
    # grid + anchors untouched by the seed choice. base and cold come from
    # two INDEPENDENT SCF runs of the same inputs, so the assertion is tight
    # tolerance, not bit-equality (which would ride the last-bit BLAS jitter
    # of separate runs; the within-record alias pins stay exact elsewhere).
    assert np.allclose(np.asarray(cold["grid_weights"]),
                       np.asarray(base["grid_weights"]))
    assert cold["E_pbe"] == pytest.approx(base["E_pbe"], abs=1e-10)
    assert np.allclose(np.asarray(cold["dm_pbe"]),
                       np.asarray(base["dm_pbe"]), rtol=0, atol=1e-10)


def test_dm_seed_minao_uks_shape():
    from xcquinox.alec.tests.fixtures.molecules import o_atom
    clear_precompute_cache()
    cold = precompute_fixed_density_data(o_atom(), seed_source="minao")
    assert np.asarray(cold["dm_seed"]).ndim == 3  # (2, nao, nao)
    assert cold["dm_seed"].shape == cold["dm_pbe"].shape


def test_dm_seed_scan_requires_cache_dir(monkeypatch):
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule
    clear_precompute_cache()
    _seed_env(monkeypatch)
    with pytest.raises(RuntimeError, match="seed"):
        precompute_fixed_density_data(h2_molecule(), seed_source="scan")


def test_dm_seed_scan_missing_cache_fails_loud_without_generate(
        tmp_path, monkeypatch):
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule
    clear_precompute_cache()
    _seed_env(monkeypatch)
    with pytest.raises(RuntimeError, match="H2"):
        precompute_fixed_density_data(
            h2_molecule(), seed_source="scan",
            seed_cache_dir=str(tmp_path), seed_allow_generate=False)


def test_dm_seed_scan_generate_gated_on_env(tmp_path, monkeypatch):
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule
    clear_precompute_cache()
    # allow-generate kwarg WITHOUT the env flag still refuses
    _seed_env(monkeypatch)
    with pytest.raises(RuntimeError):
        precompute_fixed_density_data(
            h2_molecule(), seed_source="scan",
            seed_cache_dir=str(tmp_path), seed_allow_generate=True)
    # with the env flag it generates, caches, and seeds from the cached
    # SCAN dm. (H2/sto-3g has ONE doubly-occupied symmetric MO, so the
    # converged SCAN and PBE dms are numerically identical there -- the
    # discriminating property is cache provenance, not numeric difference.)
    _seed_env(monkeypatch, allow=True)
    clear_precompute_cache()
    md = precompute_fixed_density_data(
        h2_molecule(), seed_source="scan",
        seed_cache_dir=str(tmp_path), seed_allow_generate=True)
    assert md["dm_seed"].shape == md["dm_pbe"].shape
    assert md["dm_seed"] is not md["dm_pbe"]  # loaded, not aliased
    cached = list((tmp_path / "_intermediates").glob("*_xcscan_scf.npz"))
    assert len(cached) == 1
    with np.load(cached[0]) as npz:
        assert np.allclose(np.asarray(md["dm_seed"]), npz["dm"])
    # and a second run LOADS (no generate flag needed once cached)
    clear_precompute_cache()
    _seed_env(monkeypatch)
    md2 = precompute_fixed_density_data(
        h2_molecule(), seed_source="scan", seed_cache_dir=str(tmp_path))
    assert np.allclose(np.asarray(md2["dm_seed"]), np.asarray(md["dm_seed"]))


def test_seed_cache_file_is_geometry_qualified():
    """Same species NAME at different geometries resolves to DIFFERENT cache
    files (the training-vs-pool twin problem: filename-only identity cannot
    host G2/97 H2O and BH76 H2O in one directory); identical geometry, name,
    charge, and spin resolve to the SAME file regardless of the atom
    string's formatting precision."""
    from xcquinox.alec.data import seed_cache_file
    a = MoleculeSpec(name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
                     charge=0, spin=0, atom_composition=(("H", 2),))
    b = MoleculeSpec(name="H2", atom="H 0 0 0; H 0 0 1.40", basis="sto-3g",
                     charge=0, spin=0, atom_composition=(("H", 2),))
    # same geometry, sloppier formatting -- canonicalization absorbs it
    c = MoleculeSpec(name="H2",
                     atom="H 0.0 0.0 0.0; H 0.000000 0.0 0.7400000",
                     basis="sto-3g", charge=0, spin=0,
                     atom_composition=(("H", 2),))
    fa = seed_cache_file(a, seed_cache_dir="/x")
    fb = seed_cache_file(b, seed_cache_dir="/x")
    fc = seed_cache_file(c, seed_cache_dir="/x")
    assert fa != fb
    assert fa == fc
    # charge/spin participate in the identity
    cation = MoleculeSpec(name="H2", atom="H 0 0 0; H 0 0 0.74",
                          basis="sto-3g", charge=1, spin=1,
                          atom_composition=(("H", 2),))
    assert seed_cache_file(cation, seed_cache_dir="/x") != fa


def test_dm_seed_scan_geometry_twins_get_distinct_files(tmp_path,
                                                        monkeypatch):
    """Same-name species at two geometries coexist in one cache dir: each
    generates and loads its OWN seed, no cross-contamination."""
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule
    clear_precompute_cache()
    _seed_env(monkeypatch, allow=True)
    md_a = precompute_fixed_density_data(
        h2_molecule(), seed_source="scan",
        seed_cache_dir=str(tmp_path), seed_allow_generate=True)
    stretched = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 1.40", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),))
    clear_precompute_cache()
    md_b = precompute_fixed_density_data(
        stretched, seed_source="scan",
        seed_cache_dir=str(tmp_path), seed_allow_generate=True)
    files = sorted((tmp_path / "_intermediates").glob("*_xcscan_scf.npz"))
    assert len(files) == 2
    assert not np.allclose(np.asarray(md_a["dm_seed"]),
                           np.asarray(md_b["dm_seed"]))


def test_dm_seed_scan_fingerprint_belt_still_rejects_tampered_cache(
        tmp_path, monkeypatch):
    """The overlap fingerprint remains the belt behind the filename
    identity: a wrong-S npz under the CORRECT qualified name is refused."""
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule
    from xcquinox.alec.data import seed_cache_file
    clear_precompute_cache()
    _seed_env(monkeypatch, allow=True)
    spec = h2_molecule()
    precompute_fixed_density_data(
        spec, seed_source="scan", seed_cache_dir=str(tmp_path),
        seed_allow_generate=True)
    path = seed_cache_file(spec, seed_cache_dir=str(tmp_path))
    with np.load(path) as npz:
        payload = {k: npz[k] for k in npz.files}
    payload["S"] = payload["S"] + 1e-3
    np.savez_compressed(path, **payload)
    clear_precompute_cache()
    _seed_env(monkeypatch)
    with pytest.raises(RuntimeError, match="fingerprint"):
        precompute_fixed_density_data(
            spec, seed_source="scan", seed_cache_dir=str(tmp_path))


def test_dm_seed_scan_uks_loads_spin_resolved(tmp_path, monkeypatch):
    from xcquinox.alec.tests.fixtures.molecules import o_atom
    clear_precompute_cache()
    _seed_env(monkeypatch, allow=True)
    md = precompute_fixed_density_data(
        o_atom(), seed_source="scan",
        seed_cache_dir=str(tmp_path), seed_allow_generate=True)
    assert np.asarray(md["dm_seed"]).ndim == 3
    assert md["dm_seed"].shape == md["dm_pbe"].shape


def test_precompute_memo_distinguishes_seed_source():
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule
    clear_precompute_cache()
    warm = precompute_fixed_density_data(h2_molecule())
    cold = precompute_fixed_density_data(h2_molecule(), seed_source="minao")
    # a seed-blind memo key would hand back the warm record here
    assert cold["dm_seed"] is not cold["dm_pbe"]
    assert warm["dm_seed"] is warm["dm_pbe"]


def test_seed_geometry_tag_rejects_malformed_tokens():
    """A token that is not 'Sym x y z' must raise, never silently hash a
    partial geometry (a newline-joined string would otherwise alias its
    first atom)."""
    from xcquinox.alec.data import seed_geometry_tag
    with pytest.raises(ValueError, match="malformed"):
        seed_geometry_tag("H 0 0 0\nH 0 0 0.74", 0, 0)
    with pytest.raises(ValueError, match="malformed"):
        seed_geometry_tag("H 0 0", 0, 0)


# ---------------------------------------------------------------------------
# reference_xc: the functional whose self-consistent density the record holds
# ---------------------------------------------------------------------------
# Reproducibility note, measured on this machine before these tests were
# written: two INDEPENDENT SCF runs of the same closed-shell system agree to
# ~5e-14 Ha in energy but only to ~5e-8 in the dimensionless meta-GGA alpha (a
# ratio that amplifies round-off in sigma), and two runs of a DEGENERATE
# open-shell radical (OH) can converge to different orientations of the singly
# occupied pi orbital, differing by O(100) in sigma_grid point-wise. So the
# "unchanged default" pin below is an OBJECT-IDENTITY pin through the memo
# cache plus the untouched existing suite, not a bitwise comparison of two
# separate SCF runs, which no SCF in this library would pass.

_H2O_ATOM = ("O 0.0000000000 0.0000000000 0.0000000000; "
             "H 0.0000000000 0.7570000000 0.5870000000; "
             "H 0.0000000000 -0.7570000000 0.5870000000")


def _h2o_spec():
    from xcquinox.alec.config import MoleculeSpec
    return MoleculeSpec(name="H2O_refxc", atom=_H2O_ATOM, basis="sto-3g",
                        charge=0, spin=0,
                        atom_composition=(("H", 2), ("O", 1)), grid_level=1)


def _oh_spec():
    from xcquinox.alec.config import MoleculeSpec
    return MoleculeSpec(name="OH_refxc", atom="O 0 0 0; H 0 0 0.97",
                        basis="sto-3g", charge=0, spin=1,
                        atom_composition=(("H", 1), ("O", 1)), grid_level=1)


def test_reference_xc_defaults_to_pbe_and_is_recorded():
    """The record states which functional's density it holds, so a consumer
    can assert it instead of assuming PBE."""
    from xcquinox.alec.data import precompute_fixed_density_data
    md = precompute_fixed_density_data(_h2o_spec())
    assert md["reference_xc"] == "pbe"


def test_explicit_pbe_is_the_same_record_as_the_default():
    """`reference_xc="pbe"` and the default are ONE cache entry and one SCF:
    the default path is unchanged, and no consumer silently pays for a second
    reference SCF by spelling the default out."""
    from xcquinox.alec.data import (clear_precompute_cache,
                                    precompute_fixed_density_data,
                                    set_precompute_cache_enabled)
    set_precompute_cache_enabled(True)
    clear_precompute_cache()
    spec = _h2o_spec()
    a = precompute_fixed_density_data(spec)
    b = precompute_fixed_density_data(spec, reference_xc="pbe")
    assert a is b


def test_reference_xc_scan_reproduces_a_standalone_pyscf_scan_scf():
    """The record's total energy IS the reference functional's SCF energy: a
    SCAN record must reproduce a plain PySCF SCAN calculation of the same
    molecule on the same grid."""
    import numpy as np
    from pyscf import dft, gto
    from xcquinox.alec.data import (clear_precompute_cache,
                                    precompute_fixed_density_data)
    clear_precompute_cache()
    spec = _h2o_spec()
    md = precompute_fixed_density_data(spec, reference_xc="scan")
    assert md["reference_xc"] == "scan"

    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=spec.charge,
                spin=spec.spin, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "scan"
    mf.grids.level = spec.grid_level
    mf.kernel()
    assert mf.converged
    assert abs(float(md["E_pbe"]) - float(mf.e_tot)) < 1e-8
    assert np.allclose(np.asarray(md["dm_pbe"]), np.asarray(mf.make_rdm1()),
                       atol=1e-7)


def test_reference_xc_scan_moves_the_density_and_the_energy():
    """A SCAN record is not a relabelled PBE record. H2O/sto-3g has real
    variational freedom (5 occupied orbitals in a 7-function basis); H2 and the
    H atom do NOT -- their densities are fixed by symmetry and normalization,
    so they cannot serve as this pin."""
    import numpy as np
    from xcquinox.alec.data import (clear_precompute_cache,
                                    precompute_fixed_density_data)
    clear_precompute_cache()
    spec = _h2o_spec()
    pbe = precompute_fixed_density_data(spec, reference_xc="pbe")
    scan = precompute_fixed_density_data(spec, reference_xc="scan")
    assert np.max(np.abs(np.asarray(pbe["dm_pbe"])
                         - np.asarray(scan["dm_pbe"]))) > 1e-4
    assert np.max(np.abs(np.asarray(pbe["rho_grid"])
                         - np.asarray(scan["rho_grid"]))) > 1e-5
    assert abs(float(pbe["E_pbe"]) - float(scan["E_pbe"])) > 1e-3


def test_reference_xc_scan_rebuilds_every_grid_quantity_from_that_density():
    """Every grid quantity in the record is a contraction of the record's own
    density matrix with its own AO table -- for any reference functional."""
    import numpy as np
    from xcquinox.alec.data import (clear_precompute_cache,
                                    precompute_fixed_density_data)
    clear_precompute_cache()
    md = precompute_fixed_density_data(_h2o_spec(), reference_xc="scan")
    ao = np.asarray(md["ao_grid_deriv"])
    dm = np.asarray(md["dm_pbe"])
    dm_tot = dm if dm.ndim == 2 else dm[0] + dm[1]
    rho = np.einsum("pi,ij,pj->p", ao[0], dm_tot, ao[0])
    gx = 2 * np.einsum("pi,ij,pj->p", ao[1], dm_tot, ao[0])
    gy = 2 * np.einsum("pi,ij,pj->p", ao[2], dm_tot, ao[0])
    gz = 2 * np.einsum("pi,ij,pj->p", ao[3], dm_tot, ao[0])
    assert np.allclose(np.asarray(md["rho_grid"]), rho, atol=1e-12)
    assert np.allclose(np.asarray(md["sigma_grid"]),
                       gx ** 2 + gy ** 2 + gz ** 2, atol=1e-10)
    # E_non_xc is the reference SCF's total minus its own XC energy.
    assert abs(float(md["E_non_xc"])
               - (float(md["E_pbe"]) - float(md["E_xc_pbe"]))) < 1e-12


def test_reference_xc_scan_populates_the_per_spin_blocks_by_the_same_path():
    """The per-spin-channel blocks follow the reference density with no
    special-casing: they are built from the record's own density matrix in the
    one open-shell branch, whatever functional produced it."""
    import numpy as np
    from xcquinox.alec.config import ArchitectureConfig
    from xcquinox.alec.data import (clear_precompute_cache,
                                    precompute_fixed_density_data)
    arch = ArchitectureConfig.from_spec(
        "refxc_probe", 3, 16,
        descriptors=["cusp", "dm_statistics", "rung35",
                     "rung35_multishell", "metagga"],
        meta_gga=True)
    desc = arch.materialize_descriptors()
    req = tuple(sorted({k for d in desc for k in d.required_mol_keys}))
    clear_precompute_cache()
    md = precompute_fixed_density_data(_oh_spec(), required_keys=req,
                                       descriptors=desc, reference_xc="scan")
    assert md["reference_xc"] == "scan"
    for key in ("dm_features_a", "dm_features_b",
                "rung35_features_a", "rung35_features_b",
                "rung35ms_features_a", "rung35ms_features_b",
                "metagga_features_a", "metagga_features_b",
                "tau_spin_a", "tau_spin_b"):
        assert md[key] is not None, key
        assert np.all(np.isfinite(np.asarray(md[key]))), key
    # The per-spin tau contracts the record's OWN spin-resolved density matrix.
    from xcquinox.alec.metagga import compute_tau_from_dm
    import jax.numpy as jnp
    dm = jnp.asarray(md["dm_pbe"])
    for slot, key in ((0, "tau_spin_a"), (1, "tau_spin_b")):
        want = compute_tau_from_dm(md["ao_grid_deriv"][1:4], dm[slot])
        assert np.allclose(np.asarray(md[key]), np.asarray(want), atol=1e-12)


def test_cache_key_separates_reference_xc():
    from xcquinox.alec.data import _precompute_cache_key
    spec = _h2o_spec()
    a = _precompute_cache_key(spec, (), (), None, 0.0, "pbe", None, False,
                              reference_xc="pbe")
    b = _precompute_cache_key(spec, (), (), None, 0.0, "pbe", None, False,
                              reference_xc="scan")
    assert a != b


def test_cache_never_hands_a_pbe_record_to_a_scan_caller():
    """The failure a reference_xc-blind cache key would cause: a SCAN
    certificate silently measured against the PBE density."""
    from xcquinox.alec.data import (clear_precompute_cache,
                                    precompute_fixed_density_data,
                                    set_precompute_cache_enabled)
    set_precompute_cache_enabled(True)
    clear_precompute_cache()
    spec = _h2o_spec()
    pbe = precompute_fixed_density_data(spec, reference_xc="pbe")
    scan = precompute_fixed_density_data(spec, reference_xc="scan")
    assert scan is not pbe
    assert scan["reference_xc"] == "scan"
    assert abs(float(pbe["E_pbe"]) - float(scan["E_pbe"])) > 1e-3


def test_reference_xc_must_be_a_non_empty_string():
    import pytest
    from xcquinox.alec.data import precompute_fixed_density_data
    with pytest.raises(ValueError, match="reference_xc"):
        precompute_fixed_density_data(_h2o_spec(), reference_xc="")


def _h2_spec():
    from xcquinox.alec.config import MoleculeSpec
    return MoleculeSpec(name="H2_refxc", atom="H 0 0 0; H 0 0 0.74",
                        basis="sto-3g", charge=0, spin=0,
                        atom_composition=(("H", 2),), grid_level=1)


def test_reference_xc_moves_the_per_spin_block_of_an_open_shell_record():
    """The per-spin blocks are contractions of the record's OWN density
    matrix, so a SCAN record carries SCAN's alpha_sigma, not PBE's.

    Measured on OH / sto-3g / grid 1 (standalone PySCF densities, library
    descriptor definitions): max|alpha_a(SCAN) - alpha_a(PBE)| = 5.88, against
    5.98e-6 between two independent PBE runs of the same radical -- six orders
    of magnitude between the signal and the run-to-run floor, so the 1e-2
    threshold below cannot be met by convergence noise. The ALPHA channel is
    used, not beta: OH is a degenerate 2-Pi radical whose beta channel carries
    the pi hole, and two independent PBE runs of it were measured 9.48 apart in
    alpha_b (an orientation of the singly occupied pi, not a moved density).
    The exact leg below is the load-bearing one either way: it pins the block
    to alpha of the doubled density diag(P_a, P_a) built from the record's own
    density matrix, which no orientation can satisfy accidentally.
    """
    import numpy as np
    import jax.numpy as jnp
    from xcquinox.alec.data import (clear_precompute_cache,
                                    precompute_fixed_density_data)
    from xcquinox.alec.descriptors import MetaGGAAlphaDescriptor
    from xcquinox.alec.metagga import compute_alpha, compute_tau_from_dm
    desc = (MetaGGAAlphaDescriptor(),)
    req = tuple(sorted({k for d in desc for k in d.required_mol_keys}))
    clear_precompute_cache()
    spec = _oh_spec()
    pbe = precompute_fixed_density_data(spec, required_keys=req,
                                        descriptors=desc, reference_xc="pbe")
    scan = precompute_fixed_density_data(spec, required_keys=req,
                                         descriptors=desc, reference_xc="scan")
    a_pbe = np.asarray(pbe["metagga_features_a"])
    a_scan = np.asarray(scan["metagga_features_a"])
    assert a_pbe.shape == a_scan.shape
    assert np.max(np.abs(a_scan - a_pbe)) > 1e-2

    # alpha_a of the SCAN record IS alpha of diag(P_a, P_a) for the record's
    # own density matrix: rho -> 2 rho_a, sigma -> 4 sigma_aa, tau -> 2 tau_a.
    ao = np.asarray(scan["ao_grid_deriv"])
    dm_a = np.asarray(scan["dm_pbe"])[0]
    r = np.einsum("pi,ij,pj->p", ao[0], dm_a, ao[0])
    gx = 2 * np.einsum("pi,ij,pj->p", ao[1], dm_a, ao[0])
    gy = 2 * np.einsum("pi,ij,pj->p", ao[2], dm_a, ao[0])
    gz = 2 * np.einsum("pi,ij,pj->p", ao[3], dm_a, ao[0])
    tau_a = compute_tau_from_dm(jnp.asarray(ao[1:4]), jnp.asarray(dm_a))
    want = np.asarray(compute_alpha(
        jnp.asarray(2.0 * r), jnp.asarray(4.0 * (gx ** 2 + gy ** 2 + gz ** 2)),
        2.0 * tau_a)).reshape(-1, 1)
    assert np.allclose(a_scan, want, rtol=0, atol=1e-10)
    # and the stored per-spin tau is the same contraction, undoubled.
    assert np.allclose(np.asarray(scan["tau_spin_a"]), np.asarray(tau_a),
                       rtol=0, atol=1e-12)


def test_reference_xc_leaves_a_symmetry_fixed_closed_shell_density_alone():
    """H2 / sto-3g has one doubly occupied MO fixed by symmetry and
    normalization, so it has no variational freedom: PBE and SCAN converge to
    the SAME density, and only the energy evaluated on it moves. Measured with
    standalone PySCF (grid level 1): max|dm_scan - dm_pbe| = 0.0 and
    max|rho_scan - rho_pbe| = 0.0 exactly, against |E_pbe - E_scan| =
    5.392555e-03 Ha. This is the counterpart of the H2O pin: reference_xc must
    change the record only where the physics changes it, which is why H2 (and
    the H atom) cannot serve as the density-moved test.
    """
    import numpy as np
    from xcquinox.alec.data import (clear_precompute_cache,
                                    precompute_fixed_density_data)
    clear_precompute_cache()
    spec = _h2_spec()
    pbe = precompute_fixed_density_data(spec, reference_xc="pbe")
    scan = precompute_fixed_density_data(spec, reference_xc="scan")
    assert np.max(np.abs(np.asarray(pbe["dm_pbe"])
                         - np.asarray(scan["dm_pbe"]))) < 1e-12
    assert np.max(np.abs(np.asarray(pbe["rho_grid"])
                         - np.asarray(scan["rho_grid"]))) < 1e-12
    assert np.max(np.abs(np.asarray(pbe["sigma_grid"])
                         - np.asarray(scan["sigma_grid"]))) < 1e-12
    assert abs(float(pbe["E_pbe"]) - float(scan["E_pbe"])) > 1e-3


def test_reference_xc_scan_xc_energy_matches_a_point_wise_meta_gga_evaluation():
    """The meta-GGA arm of the reference XC energy reads the value pyscf
    accumulated on the grid; that value must be the same quantity the
    point-wise route returns, evaluated with the kinetic-energy density the
    GGA row set cannot carry. Measured on H2O / sto-3g / grid 1: the two agree
    to 0.0 Ha (bitwise), and the GGA row set is refused outright by eval_xc for
    a meta-GGA (ValueError: cannot reshape ... into shape (1,5,N)), which is
    why the arm exists.
    """
    import numpy as np
    from pyscf import dft, gto
    from xcquinox.alec.data import (clear_precompute_cache,
                                    precompute_fixed_density_data)
    clear_precompute_cache()
    spec = _h2o_spec()
    md = precompute_fixed_density_data(spec, reference_xc="scan")

    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=spec.charge,
                spin=spec.spin, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "scan"
    mf.grids.level = spec.grid_level
    mf.kernel()
    assert mf.converged
    dm = mf.make_rdm1()
    ao = mf._numint.eval_ao(mol, mf.grids.coords, deriv=1)
    rho_m = mf._numint.eval_rho(mol, ao, dm, xctype="MGGA", with_lapl=False)
    # libxc's meta-GGA row set is (rho, dx, dy, dz, lapl, tau); SCAN does not
    # use the Laplacian, so the unused row is zero.
    rho6 = np.vstack([rho_m[:4], np.zeros_like(rho_m[0]), rho_m[4]])
    exc, _, _, _ = mf._numint.eval_xc("scan", rho6, spin=0)
    rho_grid = np.einsum("pi,ij,pj->p", ao[0], dm, ao[0])
    e_xc_pointwise = float(np.sum(rho_grid * exc * mf.grids.weights))
    assert abs(float(md["E_xc_pbe"]) - e_xc_pointwise) < 1e-9
    assert abs(float(md["E_non_xc"])
               - (float(md["E_pbe"]) - e_xc_pointwise)) < 1e-9


def test_reference_xc_refuses_a_hybrid_functional():
    """A hybrid's exact-exchange piece is not in the semilocal XC energy pyscf
    reports (measured: libxc.hybrid_coeff('b3lyp') = 0.2, ('pbe0') = 0.25,
    ('pbe') = 0.0), so E_xc would omit it and E_non_xc would absorb it -- the
    trained functional would then be fitted on top of a hidden exact-exchange
    term. The reference is restricted to pure functionals, of which the two the
    program uses (pbe, scan) are examples."""
    import pytest
    from xcquinox.alec.data import precompute_fixed_density_data
    with pytest.raises(ValueError, match="hybrid"):
        precompute_fixed_density_data(_h2o_spec(), reference_xc="b3lyp")


def _li_spec():
    from xcquinox.alec.config import MoleculeSpec
    return MoleculeSpec(name="Li_refxc", atom="Li 0 0 0", basis="sto-3g",
                        charge=0, spin=1, atom_composition=(("Li", 1),),
                        grid_level=1)


def test_reference_xc_scan_uks_reproduces_a_standalone_pyscf_scan_scf():
    """The open-shell branch takes the reference XC energy from the SCF's own
    veff, so it follows reference_xc with no further dispatch -- the property
    the certificate's ATOMIC E_xc numbers rest on. The pin uses the Li atom
    (2-S, non-degenerate) rather than a 2-Pi radical: OH's pi degeneracy is
    split only by the integration grid, and two independent SCAN runs of it
    were measured 1.4e-5 Ha apart, which would make the threshold a statement
    about the grid rather than about the reference functional. A
    reference_xc-blind precompute returns the PBE energy, 1.97e-2 Ha away.
    """
    import numpy as np
    from pyscf import dft, gto
    from xcquinox.alec.data import (clear_precompute_cache,
                                    precompute_fixed_density_data)
    clear_precompute_cache()
    spec = _li_spec()
    md = precompute_fixed_density_data(spec, reference_xc="scan")

    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=spec.charge,
                spin=spec.spin, verbose=0)
    mf = dft.UKS(mol)
    mf.xc = "scan"
    mf.grids.level = spec.grid_level
    mf.kernel()
    assert mf.converged
    assert np.asarray(md["dm_pbe"]).ndim == 3
    assert abs(float(md["E_pbe"]) - float(mf.e_tot)) < 1e-8
    veff = mf.get_veff(mol, mf.make_rdm1())
    assert abs(float(md["E_xc_pbe"]) - float(veff.exc)) < 1e-8
    assert abs(float(md["E_non_xc"])
               - (float(mf.e_tot) - float(veff.exc))) < 1e-8


def test_reference_xc_lda_uses_its_own_rung_row_set():
    """The closed-shell point-wise arm builds the density row set libxc
    demands for the reference functional's RUNG. An LDA reference fed the
    4-row GGA set is refused outright (measured: ValueError, cannot reshape
    array of size 36352 into shape (1,1,9088) on H2O/sto-3g/grid 1), so the
    row set follows libxc.xc_type rather than a fixed 'GGA'. For a GGA
    reference -- the whole training pipeline -- xc_type returns 'GGA' and the
    argument is unchanged."""
    import numpy as np
    from pyscf import dft, gto
    from xcquinox.alec.data import (clear_precompute_cache,
                                    precompute_fixed_density_data)
    clear_precompute_cache()
    spec = _h2o_spec()
    md = precompute_fixed_density_data(spec, reference_xc="lda,vwn")
    assert md["reference_xc"] == "lda,vwn"

    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=spec.charge,
                spin=spec.spin, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "lda,vwn"
    mf.grids.level = spec.grid_level
    mf.kernel()
    assert mf.converged
    veff = mf.get_veff(mol, mf.make_rdm1())
    assert abs(float(md["E_pbe"]) - float(mf.e_tot)) < 1e-8
    # the point-wise arm must land on pyscf's own accumulated XC energy
    assert abs(float(md["E_xc_pbe"]) - float(veff.exc)) < 1e-8
    assert np.allclose(np.asarray(md["dm_pbe"]), np.asarray(mf.make_rdm1()),
                       atol=1e-7)
