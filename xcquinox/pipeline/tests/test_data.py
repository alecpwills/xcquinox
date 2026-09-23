"""Tests for xcquinox.pipeline.data: MoleculeData, precompute, XC helpers.

Implements THE SPEC §13.2 test_data.py items (1)-(13).
"""
import numpy as np
import pytest
import jax.numpy as jnp
from unittest.mock import patch

from xcquinox.pipeline.config import MoleculeSpec
from xcquinox.pipeline import data as data_mod
from xcquinox.pipeline.data import MoleculeData, precompute_fixed_density_data
from xcquinox.pipeline.descriptors import CuspDescriptor, DMStatisticsDescriptor
from xcquinox.pipeline.tests.fixtures.molecules import (
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


# §13.2 item (3). Width 3 -> 2 on 2026-08-06: dm_entropy removed (no usable
# gradient at any converged density).


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


# §13.2 item (8)


# §13.2 item (9)


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
    with patch("xcquinox.pipeline.data.np.linalg.cond", return_value=1e12):
        with pytest.raises(ValueError, match="ill-conditioned"):
            precompute_fixed_density_data(mol)


# §13.2 item (12), D-H7 (xfail: fixture not yet generated)


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


# §13.2 item (19)


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


# §13.2 item (22)


# §13.2 item (23)


# ---------------------------------------------------------------------------
# §13.2 items (24)-(30), MoleculeSpec.grid_level
# ---------------------------------------------------------------------------


# §13.2 item (24)


# §13.2 item (25)
def test_molecule_spec_grid_level_rejects_non_int():
    """grid_level must be an int or None; floats/strings are rejected."""
    with pytest.raises(TypeError, match="grid_level must be int or None"):
        MoleculeSpec(name="H", atom="H 0 0 0", grid_level=1.0)
    with pytest.raises(TypeError, match="grid_level must be int or None"):
        MoleculeSpec(name="H", atom="H 0 0 0", grid_level="1")


# §13.2 item (26)


# §13.2 item (27)


# §13.2 item (28)


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


def test_precompute_populates_eri_when_requested():
    """When required_keys includes 'eri', precompute stashes the 4-index ERI tensor."""
    from xcquinox.pipeline.data import precompute_fixed_density_data
    from xcquinox.pipeline.tests.fixtures.molecules import h2_molecule

    data = precompute_fixed_density_data(h2_molecule(), required_keys=("eri",))
    assert "eri" in data
    nao = data["h_core"].shape[0]
    assert data["eri"].shape == (nao, nao, nao, nao)


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


def test_vxc_pbe_uks_matches_direct_pyscf_vxc():
    """UKS vxc_pbe should equal veff - J_total when evaluated at the same DM.

    Uses the DM from precompute_fixed_density_data to avoid SCF-convergence
    drift between independent pyscf runs. Tests the J-handling formula,
    not the SCF algorithm.
    """
    import numpy as np
    from pyscf import gto, dft
    from xcquinox.pipeline.data import precompute_fixed_density_data
    from xcquinox.pipeline.config import MoleculeSpec

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
    from xcquinox.pipeline.data import precompute_fixed_density_data
    from xcquinox.pipeline.config import MoleculeSpec

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


def test_precompute_cache_keys_on_required_keys_and_descriptors():
    """Different required_keys / descriptor sets must NOT collide in the
    cache -- a precompute requested with descriptors must have those
    descriptor outputs populated."""
    from xcquinox.pipeline.data import (
        clear_precompute_cache, precompute_fixed_density_data,
    )
    from xcquinox.pipeline.descriptors import CuspDescriptor
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


def test_load_external_data_guards_orientation_lock_mismatch(tmp_path):
    """A reference that RECORDS orientation_lock_strength must match the
    consumer's configured lock, else raise -- the load-time backstop for the
    cache-key gap that let an unlocked reference train against a locked
    functional (the degenerate OH/CH/NO radical density fix). Fires only when the
    ref carries the key and the consumer passes a lock; a None consumer or a
    legacy keyless ref does not raise."""
    from xcquinox.pipeline.data import _load_external_data
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
# density-only benchmark reference npz (xcquinox.pipeline.benchmark_refs contract)
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


# --------------------------------------------------------------------------- #
# dm_seed supply layer (per-rung SCF seeding)
# --------------------------------------------------------------------------- #

from xcquinox.pipeline.data import clear_precompute_cache


def _seed_env(monkeypatch, *, cache_dir=None, allow=False):
    if cache_dir is None:
        monkeypatch.delenv("XCQUINOX_SEED_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("XCQUINOX_SEED_CACHE_DIR", str(cache_dir))
    if allow:
        monkeypatch.setenv("XCQUINOX_SEED_ALLOW_GENERATE", "1")
    else:
        monkeypatch.delenv("XCQUINOX_SEED_ALLOW_GENERATE", raising=False)


def test_dm_seed_minao_differs_from_converged_and_leaves_rest_alone():
    from xcquinox.pipeline.tests.fixtures.molecules import h2_molecule
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


def test_dm_seed_scan_missing_cache_fails_loud_without_generate(
        tmp_path, monkeypatch):
    from xcquinox.pipeline.tests.fixtures.molecules import h2_molecule
    clear_precompute_cache()
    _seed_env(monkeypatch)
    with pytest.raises(RuntimeError, match="H2"):
        precompute_fixed_density_data(
            h2_molecule(), seed_source="scan",
            seed_cache_dir=str(tmp_path), seed_allow_generate=False)


def test_dm_seed_scan_fingerprint_belt_still_rejects_tampered_cache(
        tmp_path, monkeypatch):
    """The overlap fingerprint remains the belt behind the filename
    identity: a wrong-S npz under the CORRECT qualified name is refused."""
    from xcquinox.pipeline.tests.fixtures.molecules import h2_molecule
    from xcquinox.pipeline.data import seed_cache_file
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
    from xcquinox.pipeline.config import MoleculeSpec
    return MoleculeSpec(name="H2O_refxc", atom=_H2O_ATOM, basis="sto-3g",
                        charge=0, spin=0,
                        atom_composition=(("H", 2), ("O", 1)), grid_level=1)


def test_reference_xc_scan_reproduces_a_standalone_pyscf_scan_scf():
    """The record's total energy IS the reference functional's SCF energy: a
    SCAN record must reproduce a plain PySCF SCAN calculation of the same
    molecule on the same grid."""
    import numpy as np
    from pyscf import dft, gto
    from xcquinox.pipeline.data import (clear_precompute_cache,
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


def test_reference_xc_scan_rebuilds_every_grid_quantity_from_that_density():
    """Every grid quantity in the record is a contraction of the record's own
    density matrix with its own AO table -- for any reference functional."""
    import numpy as np
    from xcquinox.pipeline.data import (clear_precompute_cache,
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


def test_cache_never_hands_a_pbe_record_to_a_scan_caller():
    """The failure a reference_xc-blind cache key would cause: a SCAN
    certificate silently measured against the PBE density."""
    from xcquinox.pipeline.data import (clear_precompute_cache,
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


def _h2_spec():
    from xcquinox.pipeline.config import MoleculeSpec
    return MoleculeSpec(name="H2_refxc", atom="H 0 0 0; H 0 0 0.74",
                        basis="sto-3g", charge=0, spin=0,
                        atom_composition=(("H", 2),), grid_level=1)


def test_reference_xc_refuses_a_hybrid_functional():
    """A hybrid's exact-exchange piece is not in the semilocal XC energy pyscf
    reports (measured: libxc.hybrid_coeff('b3lyp') = 0.2, ('pbe0') = 0.25,
    ('pbe') = 0.0), so E_xc would omit it and E_non_xc would absorb it -- the
    trained functional would then be fitted on top of a hidden exact-exchange
    term. The reference is restricted to pure functionals, of which the two the
    program uses (pbe, scan) are examples."""
    import pytest
    from xcquinox.pipeline.data import precompute_fixed_density_data
    with pytest.raises(ValueError, match="hybrid"):
        precompute_fixed_density_data(_h2o_spec(), reference_xc="b3lyp")


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


def test_reference_scf_convergence_is_stamped_in_the_metadata():
    """The record states that its reference SCF converged, and in how many
    cycles, beside the functional's canonical name -- in ``mol_metadata``,
    the part of the record the certificate reads. The cycle count is pinned to
    an independent pyscf run of the same recipe, not to a literal."""
    from pyscf import dft, gto
    from xcquinox.pipeline.data import precompute_fixed_density_data
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
    from xcquinox.pipeline.config import MoleculeSpec
    from xcquinox.pipeline.pretrain_data_gen import (
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
    import xcquinox.pipeline.data as data_mod
    from xcquinox.pipeline.data import (_PRECOMPUTE_CACHE,
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


# --------------------------------------------------------------------------- #
# Second-order rescue from the trajectory-best DIIS density
# --------------------------------------------------------------------------- #


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
    from xcquinox.pipeline.orientation_lock import orientation_lock_bias
    from xcquinox.pipeline.pyscf_determinism import pin_reference_scf
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


# --------------------------------------------------------------------------- #
# dm_minao: the atomic (superposition-of-atomic-densities) seed carried BESIDE
# the PBE seed, for the per-update seed mixture of DFS SI Sec. III A.
# --------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------
# The benchmark references' T1 key and the loader's schema error
# ---------------------------------------------------------------------------


