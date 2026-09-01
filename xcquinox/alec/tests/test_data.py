"""Tests for xcquinox.alec.data: MoleculeData, precompute, XC helpers.

Implements THE SPEC §13.2 test_data.py items (1)-(13).
"""
import numpy as np
import pytest
import jax.numpy as jnp
from unittest.mock import patch

from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec import data as data_mod
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

    Both records are built under the orientation lock: OH is orbitally
    degenerate, and the unlocked SCAN reference walked its full 150 cycles
    without reaching 1e-9 in two independent runs under concurrent machine
    load (thread-order noise moves the degenerate component the SCF chases);
    the lock selects one component deterministically, which is its purpose,
    and the locked reference converges in every run.
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
                                        descriptors=desc, reference_xc="pbe",
                                        orientation_lock_strength=3.0e-5)
    scan = precompute_fixed_density_data(spec, required_keys=req,
                                         orientation_lock_strength=3.0e-5,
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
    standalone PySCF (grid level 1): max|dm_scan - dm_pbe| is at round-off,
    0.0 to 2.2e-16 across repeated measurements (two ulp of the density
    matrix's 0.60 maximum), and max|rho_scan - rho_pbe| = 0.0, against
    |E_pbe - E_scan| = 5.392555e-03 Ha; the 1e-12 bounds below are round-off
    bounds, not equalities. This is the counterpart of the H2O pin:
    reference_xc must change the record only where the physics changes it,
    which is why H2 (and the H atom) cannot serve as the density-moved test.
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
    to round-off -- 1.8e-15 Ha, one ulp of |E_xc| = 9.4 Ha, with repeated
    measurements spanning 0 to 2 ulp -- against the 1e-9 bound asserted here,
    and the GGA row set is refused outright by eval_xc for a meta-GGA
    (ValueError: cannot reshape ... into shape (1,5,N)), which is why the arm
    exists.
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


# ---------------------------------------------------------------------------
# reference_xc: canonical spelling, up-front validation, and the convergence
# of the reference SCF
# ---------------------------------------------------------------------------
# libxc's parser is case- and whitespace-insensitive and resolves synonyms:
# measured with pyscf 2.11, "scan", "SCAN", "Scan", " scan" and "scan,scan" all
# parse to ((0, 0, 0), ((263, 1), (267, 1))), and "pbe", "PBE", "pbe,pbe" and
# "gga_x_pbe,gga_c_pbe" to ((0, 0, 0), ((101, 1), (130, 1))), while "blyp"
# parses to ((106, 1), (131, 1)). Spellings libxc treats as one functional are
# one functional here: one SCF, one memo entry, one recorded name -- the short
# spelling the consumers compare against with ``==``.

def test_reference_xc_spellings_share_one_scf_one_entry_and_one_name():
    """"SCAN" lands in the "scan" entry (the same object, so the same SCF), and
    the record reads "scan" whatever the caller typed. Before the
    canonicalization each spelling ran its own SCF and stored its own entry,
    and ``md["reference_xc"] == "scan"`` was False for the "SCAN" record."""
    from xcquinox.alec.data import (_PRECOMPUTE_CACHE, clear_precompute_cache,
                                    precompute_fixed_density_data,
                                    set_precompute_cache_enabled)
    set_precompute_cache_enabled(True)
    clear_precompute_cache()
    spec = _h2o_spec()
    base = precompute_fixed_density_data(spec, reference_xc="scan")
    for spelling in ("SCAN", "Scan", " scan", "scan ", "scan,scan"):
        md = precompute_fixed_density_data(spec, reference_xc=spelling)
        assert md is base, spelling
        assert md["reference_xc"] == "scan", spelling
        assert md["mol_metadata"]["reference_xc"] == "scan", spelling
    assert len(_PRECOMPUTE_CACHE) == 1


def test_reference_xc_canonical_form_follows_libxc_parse_identity():
    """A spelling canonicalizes to one of the program's two reference names
    exactly when libxc parses it to the same functional -- the same
    (hybrid, alpha, omega) triple and the same (functional id, factor) list --
    so "pbe,pbe" IS "pbe" and shares the default record, while a functional
    outside the two keeps its lower-cased, stripped form and "blyp" is not
    mistaken for PBE."""
    from xcquinox.alec.data import (canonical_reference_xc,
                                    clear_precompute_cache,
                                    precompute_fixed_density_data,
                                    set_precompute_cache_enabled)
    assert canonical_reference_xc("pbe") == "pbe"
    assert canonical_reference_xc("PBE") == "pbe"
    assert canonical_reference_xc("pbe,pbe") == "pbe"
    assert canonical_reference_xc("gga_x_pbe,gga_c_pbe") == "pbe"
    assert canonical_reference_xc("scan,scan") == "scan"
    assert canonical_reference_xc(" Scan ") == "scan"
    assert canonical_reference_xc("LDA,VWN") == "lda,vwn"
    assert canonical_reference_xc("blyp") == "blyp"
    assert canonical_reference_xc("B3LYP") == "b3lyp"
    set_precompute_cache_enabled(True)
    clear_precompute_cache()
    spec = _h2o_spec()
    default = precompute_fixed_density_data(spec)
    twin = precompute_fixed_density_data(spec, reference_xc="pbe,pbe")
    assert twin is default
    assert twin["reference_xc"] == "pbe"


def test_cache_key_is_canonical_in_reference_xc():
    """The memo key sees the canonical name even when it is called directly,
    so no spelling of one functional can own a second entry."""
    from xcquinox.alec.data import _precompute_cache_key
    args = (_h2o_spec(), (), (), None, 0.0, "pbe", None, False)
    assert (_precompute_cache_key(*args, reference_xc="SCAN")
            == _precompute_cache_key(*args, reference_xc="scan"))
    assert (_precompute_cache_key(*args, reference_xc="pbe,pbe")
            == _precompute_cache_key(*args, reference_xc="pbe"))
    assert (_precompute_cache_key(*args, reference_xc="scan")
            != _precompute_cache_key(*args, reference_xc="pbe"))


def test_reference_xc_unknown_functional_is_refused_up_front():
    """An unknown name is a bad argument and is reported as one -- a
    ValueError naming ``reference_xc`` and the string, before any SCF -- rather
    than as libxc's KeyError ("LibXCFunctional: name 'X' not found.") out of
    the hybrid-coefficient lookup several frames down."""
    from xcquinox.alec.data import (canonical_reference_xc,
                                    precompute_fixed_density_data)
    with pytest.raises(ValueError, match="reference_xc") as info:
        precompute_fixed_density_data(_h2o_spec(),
                                      reference_xc="notafunctional")
    assert not isinstance(info.value, KeyError)
    assert "'notafunctional'" in str(info.value)
    with pytest.raises(ValueError, match="reference_xc"):
        canonical_reference_xc("pbex")
    with pytest.raises(ValueError, match="reference_xc"):
        canonical_reference_xc(None)


def test_reference_xc_refuses_a_non_local_correlation_functional():
    """pyscf books the VV10 non-local correlation energy inside ``veff.exc``,
    where no point-wise semilocal consumer of the record can see it: measured
    on H2O / sto-3g / grid 1, ``veff.exc`` for b97m-v sits 4.3e-2 Ha from the
    semilocal numint value of the same functional on the same density
    (1.3e-2 Ha for scan_vv10). "scan-vv10" is not SCAN-VV10 to pyscf's parser
    at all: the hyphen is a subtraction, giving SCAN minus the VV10 semilocal
    part (total energy -65.83 Ha against SCAN's -75.29 Ha). ``libxc.is_nlc``
    flags all three, and the reference is refused the way a hybrid is."""
    from xcquinox.alec.data import precompute_fixed_density_data
    for name in ("b97m-v", "scan-vv10", "scan_vv10"):
        with pytest.raises(ValueError, match="non-local correlation"):
            precompute_fixed_density_data(_h2o_spec(), reference_xc=name)


def test_reference_scf_convergence_is_stamped_in_the_metadata():
    """The record states that its reference SCF converged, and in how many
    cycles, beside the functional's canonical name -- in ``mol_metadata``,
    the part of the record the certificate reads. The cycle count is pinned to
    an independent pyscf run of the same recipe, not to a literal."""
    from pyscf import dft, gto
    from xcquinox.alec.data import precompute_fixed_density_data
    spec = _h2o_spec()
    md = precompute_fixed_density_data(spec, reference_xc="SCAN")
    meta = md["mol_metadata"]
    assert meta["reference_xc"] == "scan"
    assert meta["reference_scf_converged"] is True
    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=spec.charge,
                spin=spec.spin, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "scan"
    mf.grids.level = spec.grid_level
    mf.conv_tol = data_mod._REFERENCE_SCF_CONV_TOL   # the recipe's own bar
    mf.kernel()
    assert mf.converged
    assert isinstance(meta["reference_scf_cycles"], int)
    assert meta["reference_scf_cycles"] >= 1
    assert meta["reference_scf_cycles"] == int(mf.cycles)
    # DIIS converged, so the second-order stage never ran.
    assert meta["reference_scf_solver"] == "diis"
    assert meta["reference_scf_conv_tol"] == data_mod._REFERENCE_SCF_CONV_TOL


def test_reference_scf_tolerance_is_pyscfs_and_the_generators_ceiling_is_pyscfs():
    """The reference SCF converges at pyscf's default (the tolerance the
    orientation lock's reproducibility was calibrated at; a decade tighter
    the locked O atom lands on different densities in different processes),
    and the pretraining-data generator holds a record's rebuilt plain-Fock
    gradient to three times pyscf's gradient criterion -- the ceiling pyscf's
    extra cycle applies to the density it returns. The tolerance reaches
    both SCF stages: the DIIS object, and the second-order wrapper when the
    DIIS stage is cut short."""
    from pyscf import dft, gto, scf
    from xcquinox.alec import pretrain_data_gen as pdg
    assert data_mod._REFERENCE_SCF_CONV_TOL == scf.hf.SCF.conv_tol == 1e-9
    assert pdg._GRADIENT_CHECK_MARGIN == 3.0
    spec = _h2o_spec()
    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=spec.charge,
                spin=spec.spin, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "scan"
    mf.grids.level = spec.grid_level
    mf.conv_tol = 1e-4   # anything but the constant: the driver must set it
    out, cycles, solver = data_mod._converge_reference_scf(mf)
    assert solver == "diis" and out.conv_tol == data_mod._REFERENCE_SCF_CONV_TOL
    mf2 = dft.RKS(mol)
    mf2.xc = "scan"
    mf2.grids.level = spec.grid_level
    mf2.conv_tol = 1e-4
    with patch.object(data_mod, "_REFERENCE_SCF_MAX_CYCLE", 2):
        out2, cycles2, solver2 = data_mod._converge_reference_scf(mf2)
    assert solver2 == "diis+newton"
    assert out2.conv_tol == data_mod._REFERENCE_SCF_CONV_TOL
    assert out2.converged


def test_locked_ch2_scan_record_passes_the_generators_gradient_check():
    """Singlet CH2 / SCAN / def2-svp / grid level 3 under the 3e-5 orientation
    lock: pyscf converges the SCF in 7 DIIS cycles on the gradient of its
    extrapolated Fock, while the plain-Fock gradient of the stored density
    rebuilds at 3.237e-5, 1.02 times pyscf's bar -- the record the
    energy-weight sweep's data generation stopped on (job 2134711) under a
    bar of 1.0 times. It is accepted under the doubled bar, well inside it,
    the record's stamp names the tolerance, and the unlocked record rebuilds
    near pyscf's own 7.1e-7."""
    from pyscf import scf
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.pretrain_data_gen import (
        _GRADIENT_CHECK_MARGIN, _require_sane_density, _scf_gradient_norm)
    spec = MoleculeSpec(
        name="CH2_singlet", basis="def2-svp", charge=0, spin=0, grid_level=3,
        atom=("C 0.0000000000 0.0000000000 0.1799180000; "
              "H 0.0000000000 0.8554750000 -0.5397540000; "
              "H 0.0000000000 -0.8554750000 -0.5397540000"),
        atom_composition=(("C", 1), ("H", 2)))
    bar = float(np.sqrt(scf.hf.SCF.conv_tol))
    clear_precompute_cache()
    locked = precompute_fixed_density_data(spec, reference_xc="scan",
                                           orientation_lock_strength=3e-5)
    meta = locked["mol_metadata"]
    assert meta["reference_scf_converged"] is True
    assert meta["reference_scf_conv_tol"] == 1e-9
    g_locked = _scf_gradient_norm(locked)
    # Above pyscf's bar (the case), under pyscf's ceiling for what it returns.
    assert bar < g_locked < 1.5 * bar, (g_locked, bar)
    assert g_locked < _GRADIENT_CHECK_MARGIN * bar
    # The stamp is pyscf's own get_grad on the same density; the rebuild
    # reproduces it (measured 9.2e-11 relative; held to 1e-6).
    assert abs(g_locked - meta["reference_scf_gradient"]) <= 1e-6 * g_locked
    # The generator's own check accepts the record as it stands.
    _require_sane_density(locked, spec, "scan", spec.basis, spec.grid_level,
                          n_electrons=8)
    clear_precompute_cache()
    unlocked = precompute_fixed_density_data(spec, reference_xc="scan",
                                             orientation_lock_strength=0.0)
    g_unlocked = _scf_gradient_norm(unlocked)
    assert g_unlocked < bar / 10, (g_unlocked, bar)


def test_bent_ch2_pbe_record_sits_between_two_and_three_times_pyscfs_bar():
    """Singlet CH2 bent to 1.44 A / 102 degrees, PBE, def2-svp, grid level 3,
    locked: pyscf converges it in 7 cycles and returns a density whose
    plain-Fock gradient is 2.26 times its bar -- inside the 3 times its extra
    cycle accepts, outside a ceiling of 2 times. The generator accepts it."""
    from pyscf import scf
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.pretrain_data_gen import (
        _GRADIENT_CHECK_MARGIN, _require_sane_density, _scf_gradient_norm)
    r, ang = 1.44, 102.0
    h = r * np.sin(np.radians(ang / 2))
    z = -r * np.cos(np.radians(ang / 2))
    spec = MoleculeSpec(
        name="CH2_bent", basis="def2-svp", charge=0, spin=0, grid_level=3,
        atom=f"C 0 0 0; H 0 {h:.6f} {z:.6f}; H 0 {-h:.6f} {z:.6f}",
        atom_composition=(("C", 1), ("H", 2)))
    bar = float(np.sqrt(scf.hf.SCF.conv_tol))
    clear_precompute_cache()
    md = precompute_fixed_density_data(spec, reference_xc="pbe",
                                       orientation_lock_strength=3e-5)
    assert md["mol_metadata"]["reference_scf_converged"] is True
    g = _scf_gradient_norm(md)
    assert 2.0 * bar < g < _GRADIENT_CHECK_MARGIN * bar, (g / bar,)
    _require_sane_density(md, spec, "pbe", spec.basis, spec.grid_level,
                          n_electrons=8)


def test_a_record_without_the_gradient_stamp_is_refused():
    """A record carrying its convergence stamp but no gradient stamp -- the
    shape of a record from before the stamp -- is refused rather than let
    past the integrity check, as an absent convergence stamp is."""
    from xcquinox.alec.pretrain_data_gen import _require_sane_density
    spec = _h2o_spec()
    clear_precompute_cache()
    md = dict(precompute_fixed_density_data(spec, reference_xc="pbe"))
    meta = dict(md["mol_metadata"])
    assert meta["reference_scf_converged"] is True
    del meta["reference_scf_gradient"]
    md["mol_metadata"] = meta
    with pytest.raises(RuntimeError, match="reference_scf_gradient"):
        _require_sane_density(md, spec, "pbe", spec.basis, spec.grid_level,
                              n_electrons=10)


def test_a_record_whose_pieces_do_not_belong_to_one_scf_is_refused():
    """The integrity half of the gradient check: a record whose stored
    density is not the one its Fock pieces were built from rebuilds a gradient
    unrelated to the stamped one and is refused for that, whatever the
    gradient's size."""
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.pretrain_data_gen import _require_sane_density
    spec = _h2o_spec()
    clear_precompute_cache()
    md = dict(precompute_fixed_density_data(spec, reference_xc="pbe"))
    dm = np.asarray(md["dm_pbe"]).copy()
    dm[0, 0] *= 1.001
    md["dm_pbe"] = dm
    with pytest.raises(RuntimeError, match="do not belong to one SCF"):
        _require_sane_density(md, spec, "pbe", spec.basis, spec.grid_level,
                              n_electrons=10)


def test_non_converged_reference_scf_is_refused_not_recorded(monkeypatch):
    """Every field of the record is a property of the SELF-CONSISTENT density
    of the reference functional; an SCF stopped short of it is nobody's
    density. Measured on H2O / sto-3g / grid 1 with SCAN stopped after one
    cycle: the total energy is +7.2e-2 Ha off the converged value, the density
    matrix 0.315 off at its maximum, and ``mf.converged`` is False -- a record
    that was written silently before this check. No caller of the precompute
    runs a deliberately short reference SCF (the pretrain-systems tests build
    their short-SCF records outside it), so the refusal is unconditional and
    nothing is memoized. Both stages are driven to their caps here: one DIIS
    cycle, then one second-order macro-iteration (H2O / SCAN needs six DIIS
    cycles, or four second-order macro-iterations from the one-cycle
    density, measured), so the total cycle count the refusal reports is 2."""
    import xcquinox.alec.data as data_mod
    from xcquinox.alec.data import (_PRECOMPUTE_CACHE,
                                    ReferenceSCFNotConverged,
                                    precompute_fixed_density_data,
                                    set_precompute_cache_enabled)
    set_precompute_cache_enabled(True)
    monkeypatch.setattr(data_mod, "_REFERENCE_SCF_MAX_CYCLE", 1)
    monkeypatch.setattr(data_mod, "_REFERENCE_SCF_NEWTON_MAX_CYCLE", 1)
    with pytest.raises(ReferenceSCFNotConverged) as info:
        precompute_fixed_density_data(_h2o_spec(), reference_xc="scan")
    assert isinstance(info.value, RuntimeError)
    msg = str(info.value)
    for needle in ("'H2O_refxc'", "scan", "cycles=2", "converged=False",
                   "max_cycle=1"):
        assert needle in msg, needle
    assert info.value.cycles == 2
    assert len(_PRECOMPUTE_CACHE) == 0


def test_reference_scf_second_stage_converges_a_stalled_diis_run(monkeypatch):
    """A DIIS run that reaches its cycle cap unconverged is not refused
    outright: the second-order solver (pyscf SOSCF, the same |g| < sqrt(conv_tol)
    and dE < conv_tol criterion) is started from the DIIS end point, and the
    record is written only if THAT converges. The DIIS cap is lowered to two
    cycles here so the stage is exercised deterministically on H2O / SCAN
    (six DIIS cycles to converge, measured). Measured agreement between the
    second-order solution from a two-cycle DIIS density and the DIIS-converged
    one: 8.5e-14 Ha in the energy (1e-11 from a three-cycle density) and
    6.2e-7 in the density matrix (6.0e-6 from three cycles) -- both stationary
    points of the same functional within the criterion's slack. The real
    stall this stage exists for is the orientation-locked PBE O atom at
    def2-SVP / grid level 1, where DIIS from the minao guess failed in 2 of 3
    attempts at 50 cycles and 1 of 3 at 100 (|g| 3.2e-4 to 6.2e-4), and the
    second-order stage converged every time (7 macro-iterations; 4e-9 Ha from
    the DIIS-converged energy on that draw, 2.3e-8 to 9.8e-7 Ha over five
    further rescued draws -- the flat-direction slack the 3.2e-5 gradient
    criterion leaves)."""
    import xcquinox.alec.data as data_mod
    from pyscf import dft, gto
    from xcquinox.alec.data import precompute_fixed_density_data
    monkeypatch.setattr(data_mod, "_REFERENCE_SCF_MAX_CYCLE", 2)
    spec = _h2o_spec()
    md = precompute_fixed_density_data(spec, reference_xc="scan")
    meta = md["mol_metadata"]
    assert meta["reference_scf_converged"] is True
    assert meta["reference_scf_solver"] == "diis+newton"
    assert 2 < meta["reference_scf_cycles"] <= 2 + data_mod._REFERENCE_SCF_NEWTON_MAX_CYCLE

    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=spec.charge,
                spin=spec.spin, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "scan"
    mf.grids.level = spec.grid_level
    mf.kernel()
    assert mf.converged and mf.cycles > 2
    assert abs(float(md["E_pbe"]) - float(mf.e_tot)) < 1e-9
    assert np.max(np.abs(np.asarray(md["dm_pbe"]) - mf.make_rdm1())) < 1e-5
    # The record is still assembled from the converged object: E_non_xc is
    # its total minus its own XC energy, and the grid is the one DIIS built.
    assert abs(float(md["E_non_xc"])
               - (float(md["E_pbe"]) - float(md["E_xc_pbe"]))) < 1e-12
    assert np.asarray(md["grid_weights"]).shape == mf.grids.weights.shape
    assert np.array_equal(np.asarray(md["grid_weights"]), mf.grids.weights)


def test_locked_oxygen_scan_reference_converges_and_matches_pyscf():
    """The certificate's own recipe for a degenerate free atom: the SCAN UKS
    reference of the O atom (3P) at def2-SVP / grid level 1 under the
    orientation lock. Measured here: DIIS from the minao guess converged in
    14 cycles in 9 of 9 runs (three processes, three precompute calls each)
    to -74.9739766967 Ha, reproducible to 1e-10 Ha; the PBE-seeded start, by
    contrast, converged to a stationary point 1.7e-4 Ha higher in 2 of 3
    processes, and the second-order solver from the minao guess to one 8e-5 Ha
    higher -- which is why the reference SCF is started from the minao guess
    and the second stage only ever starts from the DIIS end point. The oracle
    below is the same protocol written with pyscf primitives: DIIS, then SOSCF
    from the DIIS end point if DIIS stalls; the 1e-8 Ha bound is 100x the
    measured run-to-run spread and 10x pyscf's conv_tol."""
    from pyscf import dft, gto
    from xcquinox.alec.config import MoleculeSpec
    import xcquinox.alec.data as data_mod
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.orientation_lock import (DEFAULT_STRENGTH,
                                                orientation_lock_bias)
    spec = MoleculeSpec(name="O_refxc_locked", atom="O 0 0 0", basis="def2-svp",
                        charge=0, spin=2, atom_composition=(("O", 1),),
                        grid_level=1)
    lock = float(DEFAULT_STRENGTH)
    md = precompute_fixed_density_data(spec, reference_xc="scan",
                                       orientation_lock_strength=lock)
    meta = md["mol_metadata"]
    assert meta["reference_xc"] == "scan"
    assert meta["reference_scf_converged"] is True
    assert meta["reference_scf_solver"] in ("diis", "diis+newton")

    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=spec.charge,
                spin=spec.spin, verbose=0)
    mf = dft.UKS(mol)
    mf.xc = "scan"
    mf.grids.level = spec.grid_level
    locked = np.asarray(mf.get_hcore()) + orientation_lock_bias(mol, lock)
    mf.get_hcore = lambda *a, **k: locked
    mf.max_cycle = data_mod._REFERENCE_SCF_MAX_CYCLE
    mf.kernel()
    if not mf.converged:
        so = mf.newton()
        so.kernel(dm0=mf.make_rdm1())
        assert so.converged
        mf = so
    assert abs(float(md["E_pbe"]) - float(mf.e_tot)) < 1e-8
    assert np.asarray(md["dm_pbe"]).ndim == 3


def test_one_electron_reference_forced_through_second_stage_is_refused(
        monkeypatch):
    """pyscf's SOSCF cannot represent an orbital-rotation step for an
    UNRESTRICTED system whose rotation space is empty in both channels: the H
    atom in a minimal basis has one AO per channel (alpha 1 occupied, 0
    virtual; beta 0 occupied, 1 virtual; zero occupied-virtual pairs), the
    packed rotation vector degenerates to a scalar, and newton_ah.rotate_mo
    dies with TypeError: 'float' object is not subscriptable (measured).
    Forced through the second stage (both caps at 1), the precompute must
    refuse with ReferenceSCFNotConverged naming the stage, the system and the
    cycle count -- never surface pyscf's TypeError. The guard is the measured
    crash class, not the electron count: H at def2-svp (same nelec (1, 0),
    four alpha pairs) runs the stage, and the restricted zero-pair case (He
    at sto-3g) converges in it."""
    import xcquinox.alec.data as data_mod
    from xcquinox.alec.data import (_PRECOMPUTE_CACHE,
                                    ReferenceSCFNotConverged,
                                    precompute_fixed_density_data,
                                    set_precompute_cache_enabled)
    from xcquinox.alec.tests.fixtures.molecules import h_atom
    set_precompute_cache_enabled(True)
    monkeypatch.setattr(data_mod, "_REFERENCE_SCF_MAX_CYCLE", 1)
    monkeypatch.setattr(data_mod, "_REFERENCE_SCF_NEWTON_MAX_CYCLE", 1)
    with pytest.raises(ReferenceSCFNotConverged) as info:
        precompute_fixed_density_data(h_atom(), reference_xc="scan")
    assert not isinstance(info.value, TypeError)
    msg = str(info.value)
    for needle in ("'H'", "scan", "second-order", "cycles=1"):
        assert needle in msg, needle
    assert info.value.cycles == 1
    assert len(_PRECOMPUTE_CACHE) == 0


def test_second_stage_exception_is_wrapped_into_the_refusal(monkeypatch):
    """Whatever pyscf raises inside the second-order stage surfaces as the
    convergence refusal carrying the original error text and the DIIS cycle
    count -- never as the bare exception, which would reach the certificate
    as a stray TypeError instead of a refusal it can record per system. The
    stage entry point (scf.hf.SCF.newton resolves soscf.newton_ah.newton at
    call time) is replaced here with one that raises, which stands in for any
    in-stage pyscf failure."""
    import xcquinox.alec.data as data_mod
    from pyscf.soscf import newton_ah
    from xcquinox.alec.data import (ReferenceSCFNotConverged,
                                    precompute_fixed_density_data)
    monkeypatch.setattr(data_mod, "_REFERENCE_SCF_MAX_CYCLE", 2)

    def _raise(mf):
        raise ValueError("synthetic second-order failure")

    monkeypatch.setattr(newton_ah, "newton", _raise)
    with pytest.raises(ReferenceSCFNotConverged) as info:
        precompute_fixed_density_data(_h2o_spec(), reference_xc="scan")
    msg = str(info.value)
    for needle in ("'H2O_refxc'", "second-order", "ValueError",
                   "synthetic second-order failure", "cycles=2"):
        assert needle in msg, needle
    assert info.value.cycles == 2


def test_locked_oxygen_fallback_rescues_a_cut_diis_stage(monkeypatch):
    """The second-order stage must deliver the certificate's locked O atom
    when DIIS runs out, and must not be caught by the rotation-space guard
    (the O atom has occupied-virtual pairs in both channels). With the DIIS
    stage cut to 5 cycles (14 are needed, measured), the record converges
    through "diis+newton" in 3 macro-iterations -- 8 cycles in total -- and
    lands on the same locked solution: total energy 2.3e-11 Ha from the
    full-DIIS run, in 3 of 3 measured attempts. Green before the round-2
    guard by construction: it pins what the guard must not break."""
    import xcquinox.alec.data as data_mod
    from pyscf import dft, gto
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.orientation_lock import (DEFAULT_STRENGTH,
                                                orientation_lock_bias)
    monkeypatch.setattr(data_mod, "_REFERENCE_SCF_MAX_CYCLE", 5)
    spec = MoleculeSpec(name="O_refxc_fallback", atom="O 0 0 0",
                        basis="def2-svp", charge=0, spin=2,
                        atom_composition=(("O", 1),), grid_level=1)
    lock = float(DEFAULT_STRENGTH)
    md = precompute_fixed_density_data(spec, reference_xc="scan",
                                       orientation_lock_strength=lock)
    meta = md["mol_metadata"]
    assert meta["reference_scf_converged"] is True
    assert meta["reference_scf_solver"] == "diis+newton"
    assert 5 < meta["reference_scf_cycles"] \
        <= 5 + data_mod._REFERENCE_SCF_NEWTON_MAX_CYCLE

    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=spec.charge,
                spin=spec.spin, verbose=0)
    mf = dft.UKS(mol)
    mf.xc = "scan"
    mf.grids.level = spec.grid_level
    locked = np.asarray(mf.get_hcore()) + orientation_lock_bias(mol, lock)
    mf.get_hcore = lambda *a, **k: locked
    mf.kernel()
    assert mf.converged
    assert abs(float(md["E_pbe"]) - float(mf.e_tot)) < 1e-8


# --------------------------------------------------------------------------- #
# Second-order rescue from the trajectory-best DIIS density
# --------------------------------------------------------------------------- #

def _locked_li_mean_field(xc):
    """The mean-field object ``precompute_fixed_density_data`` builds for the
    Li atom of the v6 pretraining set at the production identity: UKS (2S = 1),
    ``6-311++G(3df,2pd)``, grid level 3, ``h_core`` biased by the 3e-5
    orientation lock before the first kernel call, integral path and XC block
    size pinned. Written here in the same order as the precompute so the
    driver under test sees exactly the object the data generation hands it."""
    from pyscf import dft, gto
    from xcquinox.alec.orientation_lock import orientation_lock_bias
    from xcquinox.alec.pyscf_determinism import pin_reference_scf
    mol = gto.M(atom="Li 0 0 0", basis="6-311++G(3df,2pd)", charge=0, spin=1,
                verbose=0)
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.grids.level = 3
    locked = np.asarray(mf.get_hcore()) + orientation_lock_bias(mol, 3e-5)
    mf.get_hcore = lambda *a, **k: locked
    pin_reference_scf(mf)
    return mf


def test_li_scan_reference_is_rescued_from_the_best_diis_point():
    """The Li atom at SCAN / 6-311++G(3df,2pd) / grid level 3 under the 3e-5
    orientation lock -- the system the v6 meta-GGA data generation refused
    (job 2138034, ReferenceSCFNotConverged). DIIS reaches the solution basin
    at cycle 5 (E=-7.478697644723, |g|=7.5e-4), the extrapolation then throws
    the density to an unphysical state (E~-4.07 at |g|~1.0) and stays there to
    the cycle cap. Started from that end point -- the start the second stage
    used before the trajectory-best rescue -- it stalls at |g|~4e-3 for all 50
    macro-iterations and the driver returns unconverged, which is the state
    that refused the record. Started from the lowest-gradient density it
    converges in 2 macro-iterations: 102 cycles in total, E=-7.4786979415
    (measured, reproduced to 1.7e-11 Ha through the full precompute), whose
    plain-Fock orbital gradient is 5.97e-6, a factor 5.3 under pyscf's
    sqrt(1e-9) bar. The returned energy is checked against the physical basin
    as well as against the anchor: the pre-fix end point sits 3.4 Ha above it.
    """
    from pyscf import scf
    mf = _locked_li_mean_field("scan")
    # A caller-set callback stands in as a probe: the driver takes ownership of
    # mf.callback for the DIIS stage, so this one must never fire, and the
    # recorder that replaces it must be detached before the return.
    probe_calls = []
    mf.callback = lambda envs: probe_calls.append(int(envs.get("cycle", -1)))

    out, cycles, solver = data_mod._converge_reference_scf(mf)

    assert out.converged is True
    assert solver == "diis+newton"
    # Band 2e-6: the pinned value plus the documented flat-direction slack
    # a rescued endpoint may move within (2.3e-8 to 9.8e-7 Ha), with margin;
    # three independent convergence routes agree on the value to 3.8e-11 Ha.
    assert abs(float(out.e_tot) - (-7.4786979415)) <= 2e-6, float(out.e_tot)
    # The physical basin, not the unphysical DIIS end point of the defect.
    assert float(out.e_tot) < -7.4
    # The DIIS cap plus the macro-iterations of the rescue (2 measured).
    assert data_mod._REFERENCE_SCF_MAX_CYCLE < cycles \
        <= data_mod._REFERENCE_SCF_MAX_CYCLE \
        + data_mod._REFERENCE_SCF_NEWTON_MAX_CYCLE
    # pyscf's own criterion on the returned density, not just its flag.
    g = float(np.linalg.norm(out.get_grad(out.mo_coeff, out.mo_occ)))
    assert g < float(np.sqrt(scf.hf.SCF.conv_tol)), g
    # Callback hygiene on the rescue path: the recorder and its density copy
    # are gone from the DIIS object; the caller's probe fired through the
    # recorder's chain on every DIIS cycle and is restored.
    assert len(probe_calls) == data_mod._REFERENCE_SCF_MAX_CYCLE
    assert getattr(mf, "callback", None) is not None
    assert getattr(mf.callback, "__name__", "") != "_record_best"
    # The returned SOSCF wrapper carries only the stage's macro-iteration
    # counter (a list of ints); no density copy and no recorder reach it.
    cb = getattr(out, "callback", None)
    assert getattr(cb, "__name__", "") != "_record_best"
    cells = [c.cell_contents
             for c in (getattr(cb, "__closure__", None) or ())]
    assert not any(isinstance(v, np.ndarray) for v in cells), cells
    assert not any(isinstance(v, dict) and "dm" in v for v in cells), cells


def test_li_pbe_diis_converged_path_is_unchanged_by_the_recorder():
    """The same atom, basis, grid and lock under PBE, where DIIS converges on
    its own: recording the trajectory must not perturb the first stage or its
    early return. Measured before and after the recorder: 5 DIIS cycles,
    solver "diis", E=-7.4600641060 (3e-11 Ha through the full precompute),
    |g|=1.39e-7. The object returned is the DIIS object itself, and it comes
    back with its callback cleared -- before the recorder existed a
    caller-installed callback survived the call and was invoked on every
    cycle."""
    mf = _locked_li_mean_field("pbe")
    probe_calls = []
    mf.callback = lambda envs: probe_calls.append(int(envs.get("cycle", -1)))

    out, cycles, solver = data_mod._converge_reference_scf(mf)

    assert solver == "diis"
    assert out is mf and out.converged is True
    assert abs(float(out.e_tot) - (-7.4600641060)) <= 1e-8, float(out.e_tot)
    assert 3 <= cycles <= 8, cycles          # 5 measured
    # The caller's probe fired once per DIIS cycle (chained) and is restored.
    assert len(probe_calls) == cycles
    assert getattr(out, "callback", None) is not None
    assert getattr(out.callback, "__name__", "") != "_record_best"


class _TrajectoryStubSCF:
    """A stand-in for the DIIS object of ``_converge_reference_scf``.

    It reproduces the parts of the pyscf contract the driver uses: the kernel
    invokes ``callback(locals())`` once per cycle when one is callable (as
    ``scf.hf.kernel`` does), the envs carry ``mf``, ``cycle``, ``norm_gorb``,
    ``mo_coeff`` and ``mo_occ``, and ``make_rdm1`` builds the density from the
    orbitals it is given, or from the end point when called with none. Each
    cycle's orbitals are a distinct rotation, so the trajectory's densities are
    distinguishable and the density handed to the second stage identifies the
    cycle it came from.
    """

    def __init__(self, gradients, fire_callback=True):
        self.gradients = list(gradients)
        self.fire_callback = fire_callback
        self.callback = None
        self.converged = False
        self.cycles = len(self.gradients)
        self.max_cycle = None
        self.conv_tol = None
        # UKS-shaped occupancies with one occupied-virtual pair per channel,
        # so the empty-rotation-space refusal does not fire.
        self.mo_occ = np.array([[1.0, 0.0], [1.0, 0.0]])
        self.mo_coeff = self.orbitals(len(self.gradients) - 1)
        self.second_order = _SecondOrderStub()

    @staticmethod
    def orbitals(cycle):
        theta = 0.1 * (cycle + 1)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        return np.array([rot, rot])

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff, mo_occ = self.mo_coeff, self.mo_occ
        return np.array([(c[:, o > 0] * o[o > 0]) @ c[:, o > 0].T
                         for c, o in zip(np.asarray(mo_coeff),
                                         np.asarray(mo_occ))])

    def kernel(self):
        for cycle, gorb in enumerate(self.gradients):
            if not self.fire_callback:
                continue
            if callable(self.callback):
                self.callback({"mf": self, "cycle": cycle,
                               "norm_gorb": gorb,
                               "mo_coeff": self.orbitals(cycle),
                               "mo_occ": self.mo_occ})

    def newton(self):
        return self.second_order


class _SecondOrderStub:
    """The second-order stage as the driver drives it: it records the ``dm0``
    it is started from, reports one macro-iteration through its callback, and
    converges."""

    def __init__(self):
        self.dm0 = None
        self.callback = None
        self.converged = False
        self.max_cycle = None
        self.conv_tol = None
        self.e_tot = -1.0

    def kernel(self, dm0=None):
        self.dm0 = np.array(dm0)
        self.converged = True
        if callable(self.callback):
            self.callback({"imacro": 0})
        return self.e_tot


def test_second_stage_starts_from_the_lowest_gradient_density_of_the_trajectory():
    """The selection rule, pinned deterministically. A DIIS trajectory shaped
    like the Li case -- the gradient falls to its minimum at an intermediate
    cycle and the extrapolation then leaves the basin -- must hand the second
    stage the density of the lowest-gradient cycle, not the last one, which is
    what the stage was started from before the rescue. The minimum is repeated
    at the final cycle here, so the strict comparison is pinned too: the first
    cycle to reach it wins and the end point does not displace it on a tie.
    The fallback is pinned in the same place: a first stage that never invokes
    the callback (nothing recorded) falls back to the end-point density,
    reproducing the earlier behavior exactly. A callback the caller had
    installed keeps firing through the recorder (chained) and is restored
    afterwards. The real-path anchor of this rule is the Li atom above, where
    the two starts differ by 3.4 Ha in the converged result and by
    convergence itself."""
    gradients = [1.0, 5.0e-2, 7.5e-4, 0.9, 7.5e-4]
    stub = _TrajectoryStubSCF(gradients)
    probe_calls = []
    probe = lambda envs: probe_calls.append(envs)  # noqa: E731
    stub.callback = probe

    out, cycles, solver = data_mod._converge_reference_scf(stub)

    assert solver == "diis+newton" and out is stub.second_order
    best = stub.make_rdm1(stub.orbitals(2), stub.mo_occ)
    end = stub.make_rdm1()
    assert np.allclose(stub.second_order.dm0, best), stub.second_order.dm0
    assert not np.allclose(stub.second_order.dm0, end)
    # The cycle is identified uniquely: no other cycle's density matches.
    for cycle in range(len(gradients)):
        matches = bool(np.allclose(stub.second_order.dm0,
                                   stub.make_rdm1(stub.orbitals(cycle),
                                                  stub.mo_occ)))
        assert matches == (cycle == 2), cycle
    # The DIIS cycles plus the one macro-iteration the stage reported.
    assert cycles == len(gradients) + 1
    # The caller's callback fired on every recorded cycle through the
    # recorder's chain and is restored afterwards.
    assert len(probe_calls) == len(gradients)
    assert stub.callback is probe

    silent = _TrajectoryStubSCF(gradients, fire_callback=False)
    out2, _, solver2 = data_mod._converge_reference_scf(silent)
    assert solver2 == "diis+newton"
    assert np.allclose(silent.second_order.dm0, silent.make_rdm1())


def test_recorder_is_restored_when_the_kernel_raises():
    """A raise inside the first-stage kernel must not leave the recorder on
    the caller's object: the recorder closes over a full density copy, and
    before the try/finally the caller's mf kept it installed on this path
    (measured by the closure review's sabotage probe)."""

    class _RaisingStub(_TrajectoryStubSCF):
        def kernel(self):
            for cycle, gorb in enumerate(self.gradients[:2]):
                if callable(self.callback):
                    self.callback({
                        "mf": self, "cycle": cycle, "norm_gorb": gorb,
                        "mo_coeff": self.orbitals(cycle),
                        "mo_occ": self.mo_occ,
                    })
            raise RuntimeError("mid-kernel failure")

    stub = _RaisingStub([1.0, 0.5, 0.1])
    probe_calls = []
    probe = lambda envs: probe_calls.append(int(envs["cycle"]))  # noqa: E731
    stub.callback = probe
    with pytest.raises(RuntimeError, match="mid-kernel failure"):
        data_mod._converge_reference_scf(stub)
    assert stub.callback is probe
    assert probe_calls == [0, 1]


def test_the_best_point_recorder_is_detached_before_the_rescue_returns(
        monkeypatch):
    """Neither object the rescue path touches may leave the driver holding a
    density copy: the recorder is a closure over a full density matrix, and
    the DIIS object outlives the call (the precompute reads its grids and
    integrals afterwards, and the record is memoized). The cheap forced rescue
    of the module's H2O / SCAN identity is used -- the DIIS cap cut to two
    cycles, five cycles in total, E=-75.2917089278 measured -- so the property
    is pinned without paying for the Li identity, where the same assertions
    are made on the real stall. After the call the DIIS object's callback is
    the caller's own (the recorder chains to it while the stage runs and
    restores it before returning), and the returned second-order wrapper
    carries only its own macro-iteration counter, whose closure holds a list
    of ints and no array."""
    from pyscf import dft, gto
    monkeypatch.setattr(data_mod, "_REFERENCE_SCF_MAX_CYCLE", 2)
    spec = _h2o_spec()
    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=spec.charge,
                spin=spec.spin, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "scan"
    mf.grids.level = spec.grid_level
    probe_calls = []
    probe = lambda envs: probe_calls.append(int(envs.get("cycle", -1)))  # noqa: E731
    mf.callback = probe

    out, cycles, solver = data_mod._converge_reference_scf(mf)

    assert solver == "diis+newton" and out.converged is True
    assert out is not mf
    assert 2 < cycles <= 2 + data_mod._REFERENCE_SCF_NEWTON_MAX_CYCLE  # 5
    # The caller's callback fired during the DIIS stage and is restored.
    assert len(probe_calls) == 2
    assert getattr(mf, "callback", None) is probe
    for obj in (mf, out):
        cb = getattr(obj, "callback", None)
        assert getattr(cb, "__name__", "") != "_record_best"
        cells = [c.cell_contents
                 for c in (getattr(cb, "__closure__", None) or ())]
        assert not any(isinstance(v, np.ndarray) for v in cells), cells
        assert not any(isinstance(v, dict) and "dm" in v for v in cells), cells


# --------------------------------------------------------------------------- #
# Branch acceptance of the second-order rescue (the C2 case)
# --------------------------------------------------------------------------- #

class _BistableTrajectoryStubSCF(_TrajectoryStubSCF):
    """A trajectory over TWO SCF branches, shaped like the C2 / PBE case.

    The envs carry ``e_tot`` (as ``scf.hf.kernel``'s locals do), so the
    driver can see the trajectory's minimum-energy point. The second-order
    stubs model the measured dm0-ingestion discontinuity: a stage started
    from a DENSITY converges onto the HIGHER branch (pyscf re-occupies
    Fock(dm0) by aufbau, the step that flips C2), while a stage started from
    an ORBITAL PAIR converges onto the branch of that determinant -- the
    lower one for the trajectory's minimum-energy point.
    """

    E_HIGH = -0.2
    E_LOW = -0.7

    def __init__(self, gradients, energies, retry_converges=True,
                 retry_e_tot=None):
        super().__init__(gradients)
        self.energies = list(energies)
        self.retry_converges = retry_converges
        self.retry_e_tot = self.E_LOW if retry_e_tot is None else retry_e_tot
        self.newton_calls = []

    def kernel(self):
        for cycle, (gorb, e) in enumerate(zip(self.gradients,
                                              self.energies)):
            if callable(self.callback):
                self.callback({"mf": self, "cycle": cycle,
                               "norm_gorb": gorb, "e_tot": e,
                               "mo_coeff": self.orbitals(cycle),
                               "mo_occ": self.mo_occ})

    def newton(self):
        so = _BistableSecondOrderStub(self)
        self.newton_calls.append(so)
        return so


class _BistableSecondOrderStub:
    """Records how it was started; converges HIGH from a density and onto
    the trajectory's ``retry_e_tot`` from an orbital pair."""

    def __init__(self, traj):
        self.traj = traj
        self.callback = None
        self.converged = False
        self.max_cycle = None
        self.conv_tol = None
        self.e_tot = None
        self.start = None

    def kernel(self, dm0=None, mo_coeff=None, mo_occ=None):
        if dm0 is not None:
            self.start = ("dm0", np.array(dm0))
            self.converged = True
            self.e_tot = self.traj.E_HIGH
        else:
            self.start = ("mo", np.array(mo_coeff), np.array(mo_occ))
            self.converged = bool(self.traj.retry_converges)
            self.e_tot = self.traj.retry_e_tot
        if callable(self.callback):
            self.callback({"imacro": 0})
        return self.e_tot


def test_wrong_branch_rescue_is_rerun_from_the_lowest_energy_trajectory_point():
    """A converged rescue ABOVE the DIIS trajectory's minimum energy has
    converged onto a higher stationary point than one the trajectory already
    visited (every trajectory energy is the energy of an aufbau determinant,
    a variational upper bound of its own basin's minimum) and must be rerun
    from the minimum-energy point's ORBITAL PAIR, keeping the lower converged
    solution. The real-path anchor is C2 / PBE / 6-311++G(3df,2pd) / grid 3
    under the 3e-5 lock, where the two converged branches are 0.0798461811 Ha
    (50.1042 kcal/mol) apart (the acceptance check's own excess, higher
    branch over the trajectory minimum, reads 0.0798415986 Ha there), the
    rescue's dm0 start lands on either branch
    draw-dependently (the ground solution is non-aufbau in its own Fock, so
    even its exact density re-occupies onto the higher branch), and SOSCF
    from the lowest-energy point's orbitals converges to the ground branch
    in 2 macro-iterations (E=-75.8167407121, measured)."""
    # lowest |g| at cycle 1 (the rescue's dm0 start), lowest E at cycle 2.
    stub = _BistableTrajectoryStubSCF(gradients=[1.0, 3e-3, 5e-2, 0.9],
                                      energies=[-0.1, -0.55, -0.6, -0.3])

    out, cycles, solver = data_mod._converge_reference_scf(stub)

    assert solver == "diis+newton"
    # Two second-order stages ran: the dm0 rescue (which landed on the
    # higher branch) and the orbital-pair rerun from the lowest-E point.
    assert len(stub.newton_calls) == 2, len(stub.newton_calls)
    first, second = stub.newton_calls
    assert first.start[0] == "dm0"
    assert np.allclose(first.start[1],
                       stub.make_rdm1(stub.orbitals(1), stub.mo_occ))
    assert second.start[0] == "mo"
    assert np.allclose(second.start[1], stub.orbitals(2))
    assert np.allclose(second.start[2], stub.mo_occ)
    # The LOWER converged solution is the one returned.
    assert out is second
    assert float(out.e_tot) == pytest.approx(stub.E_LOW)
    # DIIS cycles plus one macro-iteration per second-order stage.
    assert cycles == len(stub.gradients) + 2, cycles


@pytest.mark.parametrize("retry_converges,retry_e_tot", [
    (False, _BistableTrajectoryStubSCF.E_LOW),   # rerun does not converge
    (True, _BistableTrajectoryStubSCF.E_HIGH),   # rerun converges, not lower
])
def test_wrong_branch_rescue_that_cannot_reach_a_lower_solution_is_refused(
        retry_converges, retry_e_tot):
    """When the rerun cannot produce a lower converged solution, the excess
    over the trajectory minimum still stands and the record is REFUSED: a
    converged flag on the higher SCF branch would be silently wrong by the
    inter-branch gap (50.10 kcal/mol on C2), which is exactly the defect a
    refusal makes loud. Both terminal sub-cases: a rerun that does not
    converge, and one that converges without going lower."""
    stub = _BistableTrajectoryStubSCF(gradients=[1.0, 3e-3, 5e-2, 0.9],
                                      energies=[-0.1, -0.55, -0.6, -0.3],
                                      retry_converges=retry_converges,
                                      retry_e_tot=retry_e_tot)
    with pytest.raises(data_mod.ReferenceSCFNotConverged,
                       match="stationary point"):
        data_mod._converge_reference_scf(stub)
    assert len(stub.newton_calls) == 2, len(stub.newton_calls)


class _ConvergedUphillDIISStub(_BistableTrajectoryStubSCF):
    """DIIS CONVERGES -- onto a stationary point above a determinant its own
    trajectory visited. The variational argument is identical to the rescue
    case: every trajectory energy is an aufbau determinant's energy, so a
    converged endpoint above the trajectory minimum by more than the branch
    tolerance is a higher stationary point (measured on S / SCAN /
    6-311++G(3df,2pd) / grid 3: a DIIS-converged endpoint +8.38e-6 Ha above
    its own trajectory minimum -- below tolerance; the stub places the
    excess above it)."""

    def kernel(self):
        super().kernel()
        self.converged = True
        self.e_tot = self.E_HIGH
        self.cycles = len(self.gradients)


def test_converged_diis_above_its_own_trajectory_minimum_is_rerun():
    """A converged DIIS endpoint above the trajectory's minimum-energy point
    by more than the branch tolerance is rerun from that point's ORBITAL
    PAIR directly (no dm0 stage -- the aufbau re-occupation hazard is the
    thing being avoided), and the lower converged solution is returned."""
    stub = _ConvergedUphillDIISStub(gradients=[1.0, 3e-3, 5e-2, 0.9],
                                    energies=[-0.1, -0.55, -0.6, -0.3])

    out, cycles, solver = data_mod._converge_reference_scf(stub)

    assert solver == "diis+newton"
    assert len(stub.newton_calls) == 1, len(stub.newton_calls)
    (rerun,) = stub.newton_calls
    assert rerun.start[0] == "mo"
    assert np.allclose(rerun.start[1], stub.orbitals(2))
    assert out is rerun
    assert float(out.e_tot) == pytest.approx(stub.E_LOW)
    assert cycles == len(stub.gradients) + 1, cycles


@pytest.mark.parametrize("retry_converges,retry_e_tot,fragment", [
    (False, _BistableTrajectoryStubSCF.E_LOW, "did not converge within"),
    (True, _BistableTrajectoryStubSCF.E_HIGH, "converged no lower"),
])
def test_converged_diis_uphill_that_cannot_go_lower_is_refused(
        retry_converges, retry_e_tot, fragment):
    """When the orbital-pair rerun of a converged-but-uphill DIIS endpoint
    cannot produce a lower converged solution, the record is REFUSED, and
    the message states which terminal case stood (non-convergence within
    the macro-iteration budget, or convergence no lower)."""
    stub = _ConvergedUphillDIISStub(gradients=[1.0, 3e-3, 5e-2, 0.9],
                                    energies=[-0.1, -0.55, -0.6, -0.3],
                                    retry_converges=retry_converges,
                                    retry_e_tot=retry_e_tot)
    with pytest.raises(data_mod.ReferenceSCFNotConverged, match=fragment):
        data_mod._converge_reference_scf(stub)
    assert len(stub.newton_calls) == 1, len(stub.newton_calls)


def test_c2_pbe_reference_lands_on_the_ground_scf_branch():
    """C2 at PBE / 6-311++G(3df,2pd) / grid level 3 under the 3e-5 lock --
    the held-out evaluation identity whose reference flipped on the cluster.
    DIIS oscillates unconverged between the two SCF configurations of C2 for
    all 100 cycles; the converged branches sit at E=-75.8167407121
    (internally stable) and E=-75.7368945310 (internally unstable),
    0.0798461811 Ha = 50.1042 kcal/mol apart (the excess over the
    trajectory's minimum-energy point, the quantity the acceptance check
    measures, reads 0.0798415986 Ha). Which branch the dm0-ingested
    rescue lands on is draw-dependent (measured: 4 of 10 draws of the
    pre-rescue code at this identity landed the higher branch locally, and
    seven pulled evaluations of run_20260827T163330Z stamped it, the
    cross-spec reference guard's outlier set), so the acceptance check must
    pin the returned solution to
    the ground branch on every draw. Band 2e-6 as in the Li rescue test:
    the pinned value plus the documented flat-direction slack with margin."""
    from pyscf import dft, gto
    from xcquinox.alec.orientation_lock import orientation_lock_bias
    from xcquinox.alec.pyscf_determinism import pin_reference_scf
    mol = gto.M(
        atom=("C 0.6199999559 0.0000000000 0.0000000000; "
              "C -0.6199999559 0.0000000000 0.0000000000"),
        basis="6-311++G(3df,2pd)", charge=0, spin=0, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 3
    locked = np.asarray(mf.get_hcore()) + orientation_lock_bias(mol, 3e-5)
    mf.get_hcore = lambda *a, **k: locked
    pin_reference_scf(mf)

    out, cycles, solver = data_mod._converge_reference_scf(mf)

    assert out.converged is True
    assert solver == "diis+newton"
    assert abs(float(out.e_tot) - (-75.8167407121)) <= 2e-6, float(out.e_tot)
    # Not the internally unstable higher branch of the defect.
    assert float(out.e_tot) < -75.8, float(out.e_tot)
    assert data_mod._REFERENCE_SCF_MAX_CYCLE < cycles <= (
        data_mod._REFERENCE_SCF_MAX_CYCLE
        + 2 * data_mod._REFERENCE_SCF_NEWTON_MAX_CYCLE), cycles
