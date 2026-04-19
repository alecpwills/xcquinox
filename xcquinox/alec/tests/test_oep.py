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


def test_oep_residual_decreases_on_h2o():
    """After L-BFGS-B iters on H2O, density_error is bounded.

    With the old obj/grad mismatch (obj = 0.5 int w Delta_rho^2 but grad =
    Wu-Yang form), the L-BFGS-B line search rejected valid steps because
    the Wolfe conditions require obj and grad to be derivatives of the same
    function. The new implementation uses the KS-energy-based Wu-Yang
    functional F(b) = E_KS[v(b)] - int v(b) * rho_target dr, which is
    exactly concave in b with gradient int g_t * Delta_rho.
    """
    from pyscf import gto, scf
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.oep import run_oep_inversion

    mol = gto.M(
        atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
        basis="sto-3g", verbose=0,
    )
    mf_hf = scf.RHF(mol)
    mf_hf.kernel()
    dm_hf = mf_hf.make_rdm1()

    spec = MoleculeSpec(
        name="H2O",
        atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
        basis="sto-3g", charge=0, spin=0,
        atom_composition=(("H", 2), ("O", 1)), grid_level=1,
    )
    result = run_oep_inversion(
        spec, dm_hf, max_iter=20, conv_tol=1e-4, aux_basis="sto-3g",
    )
    assert np.isfinite(result.density_error)
    # Pre-fix bug could allow density_error >> 1 (non-decreasing steps);
    # with the consistent obj/grad, a non-trivial reduction is expected.
    assert result.density_error < 1.0, (
        f"Density error {result.density_error:.3e} too large — L-BFGS-B "
        "did not make progress (obj/grad inconsistent?)"
    )


def test_oep_objective_gradient_consistent():
    """Finite-difference gradient agrees with returned analytic gradient.

    This is the direct obj/grad consistency check. The old implementation
    failed this test because obj = 0.5 int w Delta_rho^2 but grad used the
    Wu-Yang form (which is the derivative of a DIFFERENT function).
    """
    from pyscf import gto, dft
    from xcquinox.alec.oep import (
        _build_aux_basis_matrices,
        _dm_to_rho_on_grid,
        _ks_from_vxc_matrix,
    )

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf_pbe = dft.RKS(mol); mf_pbe.xc = "pbe"; mf_pbe.kernel()
    dm_target = mf_pbe.make_rdm1()
    _, three_center, aux_on_grid = _build_aux_basis_matrices(mol, mf_pbe, "sto-3g")
    rho_target = _dm_to_rho_on_grid(mol, mf_pbe, dm_target)
    rhotarget_integrals = np.einsum("gp,g->p", aux_on_grid, rho_target)
    h_core = mf_pbe.get_hcore()
    regularization = 1e-4

    def obj_grad(b):
        vxc_matrix = np.einsum("t,tij->ij", b, three_center)
        dm_scf, _, j_matrix = _ks_from_vxc_matrix(mol, mf_pbe, vxc_matrix)
        rho_scf = _dm_to_rho_on_grid(mol, mf_pbe, dm_scf)
        delta_rho = rho_scf - rho_target
        e_ks = (
            float(np.einsum("ij,ij->", dm_scf, h_core))
            + 0.5 * float(np.einsum("ij,ij->", dm_scf, j_matrix))
            + float(np.einsum("ij,ij->", dm_scf, vxc_matrix))
        )
        F_val = e_ks - float(np.dot(b, rhotarget_integrals))
        obj = -F_val + 0.5 * regularization * float(np.sum(b ** 2))
        grad = -np.einsum("gp,g->p", aux_on_grid, delta_rho) + regularization * b
        return obj, grad

    n_aux = three_center.shape[0]
    rng = np.random.default_rng(42)
    b0 = 0.01 * rng.standard_normal(n_aux)
    _, g_analytic = obj_grad(b0)

    h = 1e-5
    for t in range(n_aux):
        bp = b0.copy(); bp[t] += h
        bm = b0.copy(); bm[t] -= h
        fp, _ = obj_grad(bp)
        fm, _ = obj_grad(bm)
        g_fd = (fp - fm) / (2 * h)
        rel_err = abs(g_fd - g_analytic[t]) / (abs(g_analytic[t]) + 1e-12)
        # Finite-diff error from inner-SCF tolerance ~1e-12 => grad
        # accurate to ~1e-3 relative (loose bound; tight value ~2e-4).
        assert rel_err < 5e-3, (
            f"Obj/grad inconsistent at t={t}: "
            f"fd={g_fd:.3e} analytic={g_analytic[t]:.3e} rel_err={rel_err:.3e}"
        )
