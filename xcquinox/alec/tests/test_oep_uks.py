"""UKS OEP inversion tests.

Notes on open-shell tests:
    The UKS Wu-Yang inversion needs a non-degenerate ground state for the
    Hellmann-Feynman theorem to produce a smooth F(b). The O atom in sto-3g
    (spin=2, p-shell) has three equivalent spin-broken UHF solutions
    (p_x/p_y/p_z occupation) that SCF can jump between under small b
    perturbations, so density-error and FD gradient tests use the Li atom
    instead (spin=1, 1s^2 alpha / 1s^1 beta, no p-shell degeneracy). The
    shape/runs smoke test still exercises the O atom per the task plan.

    For real production species in the dfs_ae pool (HO X²Π, CN X²Π,
    NO X²Π, NO2 X²A1), the basin-hopping problem is dealt with via the
    ``level_shift`` kwarg on ``run_oep_inversion``. ``run_oep_cascade``
    in ``external_refs.py`` automatically sets ``level_shift=0.5`` for
    UKS species (``spec.spin > 0``).
"""
import numpy as np
import pytest

from pyscf import gto, scf
from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.oep import run_oep_inversion


def test_oep_uks_on_o_atom_runs():
    """OEP inversion on a UKS target DM (O atom, spin=2) must return a
    (2, nao, nao) vxc matrix and converged/density_error fields.

    No convergence requirement — only that the code runs end-to-end with a
    UKS-shaped DM and produces a (2, nao, nao) V_xc.
    """
    mol = gto.M(atom="O 0 0 0", basis="sto-3g", spin=2, verbose=0)
    mf = scf.UHF(mol)
    mf.kernel()
    dm_target = mf.make_rdm1()  # shape (2, nao, nao)

    spec = MoleculeSpec(
        name="O", atom="O 0 0 0", basis="sto-3g",
        charge=0, spin=2, atom_composition=(("O", 1),), grid_level=1,
    )
    result = run_oep_inversion(spec, dm_target, max_iter=30, conv_tol=1e-3)
    assert result.vxc_matrix.shape == (2, 5, 5), (
        f"expected (2, 5, 5), got {result.vxc_matrix.shape}"
    )
    # density_error should be finite
    assert np.isfinite(result.density_error)


def test_oep_uks_reproduces_target_density():
    """After OEP inversion on a UKS target DM (Li atom), the SCF density
    from the inverted V_xc should approximate the target density.

    Li is chosen instead of O because Li's 1s^2 alpha / 1s^1 beta structure
    has no p-shell occupation degeneracy, so the SCF stays in a single
    basin as the OEP coefficients b are varied.
    """
    mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1, verbose=0)
    mf = scf.UHF(mol)
    mf.kernel()
    dm_target = mf.make_rdm1()

    spec = MoleculeSpec(
        name="Li", atom="Li 0 0 0", basis="sto-3g",
        charge=0, spin=1, atom_composition=(("Li", 1),), grid_level=1,
    )
    result = run_oep_inversion(spec, dm_target, max_iter=80, conv_tol=1e-4)
    # The inverted V_xc should give density close to target.
    # Loose bound: check real progress, not full convergence.
    assert result.density_error < 0.1, (
        f"density_error = {result.density_error:.3e}, expected < 0.1"
    )


def test_oep_uks_objective_gradient_consistent():
    """Finite-difference gradient agrees with the analytic Wu-Yang gradient
    for the UKS functional. Uses Li atom for the same reason as above."""
    from xcquinox.alec.oep import (
        _build_aux_basis_matrices,
        _ks_from_vxc_matrix,
    )
    from pyscf import dft

    mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1, verbose=0)
    mf_pbe = dft.UKS(mol)
    mf_pbe.xc = "pbe"
    mf_pbe.kernel()
    dm_target = mf_pbe.make_rdm1()  # (2, nao, nao)

    _, three_center, aux_on_grid, _S_aux = _build_aux_basis_matrices(mol, mf_pbe, "sto-3g")
    n_aux = three_center.shape[0]
    ao = mf_pbe._numint.eval_ao(mol, mf_pbe.grids.coords)
    rho_target_a = np.einsum("pi,ij,pj->p", ao, dm_target[0], ao)
    rho_target_b = np.einsum("pi,ij,pj->p", ao, dm_target[1], ao)
    rhotarget_a_integrals = np.einsum("gp,g->p", aux_on_grid, rho_target_a)
    rhotarget_b_integrals = np.einsum("gp,g->p", aux_on_grid, rho_target_b)
    h_core = mf_pbe.get_hcore()
    regularization = 1e-4

    # Warm-start seed: use the PBE DM as a fixed seed so all FD calls hit
    # the same SCF basin.
    dm_seed = dm_target

    def obj_grad(b):
        b_a = b[:n_aux]
        b_b = b[n_aux:]
        vxc_a = np.einsum("t,tij->ij", b_a, three_center)
        vxc_b = np.einsum("t,tij->ij", b_b, three_center)
        vxc_matrix = np.stack([vxc_a, vxc_b], axis=0)
        dm_scf, _, j_matrix, _ = _ks_from_vxc_matrix(
            mol, mf_pbe, vxc_matrix, dm0=dm_seed,
        )
        rho_scf_a = np.einsum("pi,ij,pj->p", ao, dm_scf[0], ao)
        rho_scf_b = np.einsum("pi,ij,pj->p", ao, dm_scf[1], ao)
        delta_a = rho_scf_a - rho_target_a
        delta_b = rho_scf_b - rho_target_b
        # j_matrix is per-spin J; J_total = j[0] + j[1]
        j_total = j_matrix[0] + j_matrix[1]
        dm_total = dm_scf[0] + dm_scf[1]
        # E_KS = Tr(D_tot h) + 0.5 Tr(D_tot J_tot) + sum_s Tr(D_s V_xc_s)
        e_ks = (
            float(np.einsum("ij,ij->", dm_total, h_core))
            + 0.5 * float(np.einsum("ij,ij->", dm_total, j_total))
            + float(np.einsum("ij,ij->", dm_scf[0], vxc_a))
            + float(np.einsum("ij,ij->", dm_scf[1], vxc_b))
        )
        F_val = (
            e_ks
            - float(np.dot(b_a, rhotarget_a_integrals))
            - float(np.dot(b_b, rhotarget_b_integrals))
        )
        obj = -F_val + 0.5 * regularization * float(np.sum(b ** 2))
        grad_a = -np.einsum("gp,g->p", aux_on_grid, delta_a) + regularization * b_a
        grad_b = -np.einsum("gp,g->p", aux_on_grid, delta_b) + regularization * b_b
        grad = np.concatenate([grad_a, grad_b])
        return obj, grad

    rng = np.random.default_rng(42)
    b0 = 0.01 * rng.standard_normal(2 * n_aux)
    _, g_analytic = obj_grad(b0)

    h = 1e-5
    # Check first 3 entries of each spin channel for speed
    for t in list(range(3)) + list(range(n_aux, n_aux + 3)):
        bp = b0.copy(); bp[t] += h
        bm = b0.copy(); bm[t] -= h
        fp, _ = obj_grad(bp)
        fm, _ = obj_grad(bm)
        g_fd = (fp - fm) / (2 * h)
        rel_err = abs(g_fd - g_analytic[t]) / (abs(g_analytic[t]) + 1e-12)
        assert rel_err < 5e-3, (
            f"Obj/grad inconsistent at t={t}: "
            f"fd={g_fd:.3e} analytic={g_analytic[t]:.3e} rel_err={rel_err:.3e}"
        )


def test_run_oep_inversion_accepts_level_shift_kwarg():
    """Fast structural test: ``level_shift`` kwarg accepted and threads
    through to the inner SCF without raising. Uses Li/sto-3g (the
    non-degenerate UKS testbed already validated above) so this test
    does not gate on absolute density_error -- only that the kwarg is
    accepted, the result has finite density_error, and the kwarg
    forwarding is wired correctly through ``run_oep_inversion`` ->
    ``_ks_from_vxc_matrix`` -> ``_ks_from_vxc_matrix_uhf``.

    The actual basin-stabilizing effect of ``level_shift`` is verified
    end-to-end via ``scripts/smoke_preflight_uks_oep.py`` on HO
    (the X²Π radical that drove the fix).
    """
    mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1, verbose=0)
    mf = scf.UHF(mol)
    mf.kernel()
    dm_target = mf.make_rdm1()

    spec = MoleculeSpec(
        name="Li", atom="Li 0 0 0", basis="sto-3g",
        charge=0, spin=1, atom_composition=(("Li", 1),), grid_level=1,
    )
    result = run_oep_inversion(
        spec, dm_target, max_iter=20, conv_tol=1e-3, level_shift=0.5,
    )
    assert np.isfinite(result.density_error), (
        f"density_error={result.density_error} not finite with level_shift=0.5"
    )
    # vxc_matrix has the right UKS shape regardless of convergence
    assert result.vxc_matrix.ndim == 3 and result.vxc_matrix.shape[0] == 2


def test_ks_from_vxc_matrix_uhf_sets_level_shift_attr():
    """Fast unit test: ``_ks_from_vxc_matrix_uhf(level_shift=X)`` sets the
    attribute on the inner ``mf_fixed`` object. Exercises the kwarg
    plumbing without running the full L-BFGS-B optimization.

    Verifies the wiring by monkey-patching ``pyscf.scf.UHF`` (a factory
    function, not a class) with a wrapper that captures
    ``mf.level_shift`` at ``kernel()`` time.
    """
    from pyscf import scf as _scf
    from xcquinox.alec.oep import _ks_from_vxc_matrix_uhf

    mol = gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1, verbose=0)
    mf_outer = _scf.UHF(mol)
    mf_outer.kernel()

    captured = {"level_shift_seen": None}
    real_UHF = _scf.UHF

    def _spy_UHF(*args, **kwargs):
        mf = real_UHF(*args, **kwargs)
        original_kernel = mf.kernel
        def _spy_kernel(*a, **kw):
            captured["level_shift_seen"] = mf.level_shift
            return original_kernel(*a, **kw)
        mf.kernel = _spy_kernel
        return mf

    nao = mol.nao
    vxc_matrix = np.zeros((2, nao, nao))
    try:
        _scf.UHF = _spy_UHF
        _ks_from_vxc_matrix_uhf(
            mol, mf_outer, vxc_matrix, level_shift=0.5,
        )
    finally:
        _scf.UHF = real_UHF
    assert captured["level_shift_seen"] == 0.5, (
        f"expected mf_fixed.level_shift=0.5, "
        f"got {captured['level_shift_seen']!r}"
    )

    # Default (level_shift=0.0) must NOT touch the attribute (preserve
    # pre-fix behavior where mf_fixed.level_shift stays at PySCF default 0).
    captured["level_shift_seen"] = None
    try:
        _scf.UHF = _spy_UHF
        _ks_from_vxc_matrix_uhf(mol, mf_outer, vxc_matrix)
    finally:
        _scf.UHF = real_UHF
    assert captured["level_shift_seen"] == 0.0, (
        f"default level_shift must remain PySCF default 0.0, "
        f"got {captured['level_shift_seen']!r}"
    )
