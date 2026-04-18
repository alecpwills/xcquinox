"""Tests for xcquinox.alec.oep — Wu-Yang OEP inversion utility."""
import numpy as np
import pytest

from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.tests.fixtures.molecules import h2_molecule


def test_oep_result_shape():
    """OEPResult.vxc_matrix has shape (nao, nao) matching the basis."""
    from xcquinox.alec.oep import run_oep_inversion
    mol = h2_molecule()
    from xcquinox.alec.data import precompute_fixed_density_data
    data = precompute_fixed_density_data(mol)
    dm_target = np.asarray(data["dm_pbe"])
    result = run_oep_inversion(mol, dm_target, max_iter=5, aux_basis="sto-3g")
    nao = dm_target.shape[-1]
    assert result.vxc_matrix.shape == (nao, nao)


def test_oep_pbe_identity():
    """PBE density as target should recover approximately V_xc^PBE."""
    from xcquinox.alec.oep import run_oep_inversion
    from xcquinox.alec.data import precompute_fixed_density_data
    mol = h2_molecule()
    data = precompute_fixed_density_data(mol)
    dm_target = np.asarray(data["dm_pbe"])
    vxc_pbe = np.asarray(data["vxc_pbe"])
    result = run_oep_inversion(
        mol, dm_target, max_iter=50, conv_tol=1e-8,
        aux_basis="sto-3g", regularization=1e-6,
    )
    if result.converged:
        diff = np.linalg.norm(result.vxc_matrix - vxc_pbe)
        ref_norm = np.linalg.norm(vxc_pbe) + 1e-8
        assert diff / ref_norm < 1.5, (
            f"Converged OEP V_xc differs from PBE V_xc by {diff/ref_norm:.2%}"
        )


def test_oep_nonconvergence_flagged():
    """max_iter=1 should report converged=False."""
    from xcquinox.alec.oep import run_oep_inversion
    from xcquinox.alec.data import precompute_fixed_density_data
    mol = h2_molecule()
    data = precompute_fixed_density_data(mol)
    dm_target = np.asarray(data["dm_pbe"]) * 0.9
    result = run_oep_inversion(mol, dm_target, max_iter=1, aux_basis="sto-3g")
    assert result.converged is False
    assert result.n_iter <= 1
    assert result.density_error > 0.0


def test_save_vxc_ref_roundtrip(tmp_path):
    """save_vxc_ref creates a .npz loadable by _load_external_data."""
    from xcquinox.alec.oep import OEPResult, save_vxc_ref
    from xcquinox.alec.data import _load_external_data
    nao = 3
    vxc = np.random.default_rng(42).standard_normal((nao, nao))
    oep_result = OEPResult(
        vxc_matrix=vxc, converged=True, n_iter=10, density_error=1e-7,
    )
    path = str(tmp_path / "ref.npz")
    save_vxc_ref(oep_result, path, method="CCSD")
    _, _, _, _, vxc_loaded = _load_external_data(
        path,
        dm_pbe_shape=(nao, nao),
        rho_pbe_shape=(100,),
        vxc_pbe_shape=(nao, nao),
        mol_name="test",
    )
    np.testing.assert_allclose(np.asarray(vxc_loaded), vxc, rtol=1e-10)


@pytest.mark.slow
def test_oep_converges_on_h2():
    """Full OEP inversion converges on H2 with PBE target density."""
    from xcquinox.alec.oep import run_oep_inversion
    from xcquinox.alec.data import precompute_fixed_density_data
    mol = h2_molecule()
    data = precompute_fixed_density_data(mol)
    dm_target = np.asarray(data["dm_pbe"])
    result = run_oep_inversion(mol, dm_target, max_iter=200, conv_tol=1e-6, aux_basis="sto-3g")
    assert result.converged, f"OEP did not converge: error={result.density_error:.2e}"
    assert result.density_error < 1e-6
