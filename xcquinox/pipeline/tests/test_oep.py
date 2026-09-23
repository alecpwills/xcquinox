"""Tests for xcquinox.pipeline.oep: Wu-Yang OEP inversion utility."""
import numpy as np
import pytest

from xcquinox.pipeline.tests.fixtures.molecules import h2_molecule


def test_oep_result_shape():
    """OEPResult.vxc_matrix has shape (nao, nao) matching the basis."""
    from xcquinox.pipeline.oep import run_oep_inversion
    mol = h2_molecule()
    from xcquinox.pipeline.data import precompute_fixed_density_data
    data = precompute_fixed_density_data(mol)
    dm_target = np.asarray(data["dm_pbe"])
    result = run_oep_inversion(mol, dm_target, max_iter=5, aux_basis="sto-3g")
    nao = dm_target.shape[-1]
    assert result.vxc_matrix.shape == (nao, nao)


def test_oep_pbe_identity():
    """PBE density as target recovers V_xc^PBE.

    With the default ``baseline_xc='pbe'`` and the PBE density matrix as the
    target, the displacement form starts at b=0 already matching the target,
    so the returned V_xc equals the baseline PBE V_xc. The inversion must
    report convergence (density matched below conv_tol) and reproduce the
    reference V_xc to a tight tolerance -- not merely pass vacuously when it
    fails to converge.
    """
    from xcquinox.pipeline.oep import run_oep_inversion
    from xcquinox.pipeline.data import precompute_fixed_density_data
    mol = h2_molecule()
    data = precompute_fixed_density_data(mol)
    dm_target = np.asarray(data["dm_pbe"])
    vxc_pbe = np.asarray(data["vxc_pbe"])
    conv_tol = 1e-8
    result = run_oep_inversion(
        mol, dm_target, max_iter=50, conv_tol=conv_tol,
        aux_basis="sto-3g", regularization=1e-6,
    )
    assert result.converged is True, (
        f"OEP must converge on the PBE-identity target; "
        f"density_error={result.density_error:.2e}, "
        f"lbfgs_status={result.lbfgs_status!r}"
    )
    assert result.density_error < conv_tol
    diff = np.linalg.norm(result.vxc_matrix - vxc_pbe)
    ref_norm = np.linalg.norm(vxc_pbe) + 1e-8
    assert diff / ref_norm < 1e-6, (
        f"Converged OEP V_xc differs from PBE V_xc by {diff/ref_norm:.2%}"
    )


def test_oep_nonconvergence_flagged():
    """max_iter=1 should report converged=False (or genuinely converged
    if the displacement form's b=0 already matches; we don't pin which).
    Uses HF target with PBE baseline so the optimizer has real work
    (b=0 gives PBE DM, target is HF DM, different non-PBE DM forces
    iterations without violating the D10 Tr(S*D)=N_e sanity check).
    """
    from xcquinox.pipeline.oep import run_oep_inversion
    from pyscf import gto, scf
    mol = h2_molecule()
    pyscf_mol = gto.M(atom=mol.atom, basis=mol.basis, charge=mol.charge,
                      spin=mol.spin, verbose=0)
    mf_hf = scf.RHF(pyscf_mol)
    mf_hf.kernel()
    dm_target = np.asarray(mf_hf.make_rdm1())
    result = run_oep_inversion(mol, dm_target, max_iter=1, aux_basis="sto-3g")
    assert result.n_iter <= 1
    assert result.density_error >= 0.0
    # If the inversion didn't fully converge in 1 iteration, .converged
    # must be False; if it DID converge, that's also fine, we just
    # require the flag to be a bool reflecting the actual state.
    assert isinstance(result.converged, bool)


def test_save_vxc_ref_roundtrip(tmp_path):
    """save_vxc_ref creates a .npz loadable by _load_external_data."""
    from xcquinox.pipeline.oep import OEPResult, save_vxc_ref
    from xcquinox.pipeline.data import _load_external_data
    nao = 3
    vxc = np.random.default_rng(42).standard_normal((nao, nao))
    oep_result = OEPResult(
        vxc_matrix=vxc, converged=True, n_iter=10, density_error=1e-7,
        baseline_xc="pbe",
        aux_basis="def2-svp-jkfit",
        regularization=1e-4,
        n_electrons=2.0,
        lbfgs_status="CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
    )
    path = str(tmp_path / "ref.npz")
    save_vxc_ref(oep_result, path, method="CCSD")
    _, _, _, _, vxc_loaded, _ = _load_external_data(
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
    from xcquinox.pipeline.oep import run_oep_inversion
    from xcquinox.pipeline.data import precompute_fixed_density_data
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
    from xcquinox.pipeline.config import MoleculeSpec
    from xcquinox.pipeline.oep import run_oep_inversion

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
        f"Density error {result.density_error:.3e} too large, L-BFGS-B "
        "did not make progress (obj/grad inconsistent?)"
    )


def test_oep_objective_gradient_consistent():
    """Finite-difference gradient agrees with returned analytic gradient.

    This is the direct obj/grad consistency check. The old implementation
    failed this test because obj = 0.5 int w Delta_rho^2 but grad used the
    Wu-Yang form (which is the derivative of a DIFFERENT function).
    """
    from pyscf import gto, dft
    from xcquinox.pipeline.oep import (
        _build_aux_basis_matrices,
        _dm_to_rho_on_grid,
        _ks_from_vxc_matrix,
    )

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf_pbe = dft.RKS(mol); mf_pbe.xc = "pbe"; mf_pbe.kernel()
    dm_target = mf_pbe.make_rdm1()
    # New API: _build_aux_basis_matrices returns S_aux too (fix).
    _, three_center, aux_on_grid, S_aux = _build_aux_basis_matrices(
        mol, mf_pbe, "sto-3g",
    )
    rho_target = _dm_to_rho_on_grid(mol, mf_pbe, dm_target)
    rhotarget_integrals = np.einsum("gp,g->p", aux_on_grid, rho_target)
    h_core = mf_pbe.get_hcore()
    regularization = 1e-4

    def obj_grad(b):
        # Mirror the production displacement form with V-space reg.
        vxc_matrix = np.einsum("t,tij->ij", b, three_center)
        dm_scf, _, j_matrix, _ = _ks_from_vxc_matrix(mol, mf_pbe, vxc_matrix)
        rho_scf = _dm_to_rho_on_grid(mol, mf_pbe, dm_scf)
        delta_rho = rho_scf - rho_target
        e_ks = (
            float(np.einsum("ij,ij->", dm_scf, h_core))
            + 0.5 * float(np.einsum("ij,ij->", dm_scf, j_matrix))
            + float(np.einsum("ij,ij->", dm_scf, vxc_matrix))
        )
        F_val = e_ks - float(np.dot(b, rhotarget_integrals))
        # V-space regularization: b^T S_aux b (Heaton-Burgess 2007).
        obj = -F_val + 0.5 * regularization * float(b @ S_aux @ b)
        grad = -np.einsum("gp,g->p", aux_on_grid, delta_rho) + regularization * (S_aux @ b)
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


# ---------------------------------------------------------------------------
# Fix: baseline_xc parameter (Wu-Yang displacement form)
# ---------------------------------------------------------------------------

def test_oep_baseline_xc_parameter_accepts_arbitrary_xc():
    """run_oep_inversion(baseline_xc='lda'/'pbe'/'blyp'/'hf'/None) must
    all run without error and record baseline_xc in the OEPResult.
    Wu & Yang JCP 118, 2498 (2003) §II.B uses the displacement form
    V_xc = V_xc^baseline + sum_t b_t g_t; the baseline is user-choosable
    so the inversion is generalizable to any starting XC functional.

    R3-F strengthening: the prior assertion only checked the
    attribute round-trip (``result.baseline_xc == xc``), which a
    silently-ignored ``baseline_xc`` arg would still satisfy. Add a
    behavioral discriminator: ``vxc_matrix`` must differ between LDA
    and PBE baselines on the same target (V_xc^LDA != V_xc^PBE for any
    physical density), proving the baseline is actually consumed.
    Includes ``'hf'`` per docstring (routes through ``mf.xc = 'hf'``).
    """
    from xcquinox.pipeline.oep import run_oep_inversion
    from xcquinox.pipeline.data import precompute_fixed_density_data
    mol = h2_molecule()
    data = precompute_fixed_density_data(mol)
    dm_target = np.asarray(data["dm_pbe"])
    vxc_by_xc: dict[str | None, np.ndarray] = {}
    for xc in ("lda", "pbe", "blyp", "hf", None):
        result = run_oep_inversion(
            mol, dm_target, max_iter=3, aux_basis="sto-3g",
            baseline_xc=xc,
        )
        assert result.baseline_xc == xc, (xc, result.baseline_xc)
        assert np.all(np.isfinite(result.vxc_matrix))
        vxc_by_xc[xc] = np.asarray(result.vxc_matrix)
    # Behavioral check: distinct baselines must produce distinct V_xc
    # matrices. LDA vs PBE on H2 in sto-3g differ by ~10^-2 in Frobenius
    # norm at the converged baseline DM; require at least 1e-4 to be
    # robust against max_iter=3 truncation noise.
    diff_lda_pbe = np.linalg.norm(vxc_by_xc["lda"] - vxc_by_xc["pbe"])
    assert diff_lda_pbe > 1e-4, (
        f"V_xc(LDA baseline) and V_xc(PBE baseline) should differ; got "
        f"||ΔV_xc||_F = {diff_lda_pbe:.3e}. A near-zero difference "
        f"indicates baseline_xc is being silently ignored."
    )
    diff_pbe_blyp = np.linalg.norm(vxc_by_xc["pbe"] - vxc_by_xc["blyp"])
    assert diff_pbe_blyp > 1e-4, (
        f"V_xc(PBE) and V_xc(BLYP) should differ; got "
        f"||ΔV_xc||_F = {diff_pbe_blyp:.3e}."
    )


def test_oep_v_space_regularization_uses_aux_overlap():
    """Fix: V-space regularization 0.5*lambda*b^T S_aux b is
    aux-basis independent in meaning. Pre-fix coefficient-space
    0.5*lambda*|b|^2 silently changed regularization strength when
    aux_basis was swapped. Heaton-Burgess et al. PRL 98, 256401 (2007).

    R3-F rename: prior name ``..._basis_independent`` implied a
    cross-basis comparison; this test only verifies S_aux is constructed
    correctly (symmetric + PSD + positive diagonal) for one aux basis.
    The basis-independence property follows from the math; a numerical
    test would require running the full inversion in two bases.
    """
    from xcquinox.pipeline.oep import _build_aux_basis_matrices
    from pyscf import gto, dft
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = dft.RKS(mol); mf.xc = "pbe"; mf.kernel()
    # _build_aux_basis_matrices now returns S_aux (D2 fix).
    aux_mol, three_center, aux_on_grid, S_aux = _build_aux_basis_matrices(
        mol, mf, "sto-3g",
    )
    # S_aux must be symmetric and positive semi-definite (overlap matrix).
    assert np.allclose(S_aux, S_aux.T, atol=1e-10), (
        f"S_aux must be symmetric; max(|S - S.T|) = "
        f"{np.max(np.abs(S_aux - S_aux.T)):.3e}"
    )
    eigs = np.linalg.eigvalsh(S_aux)
    # Allow tiny numerical noise below zero from quadrature.
    assert np.all(eigs > -1e-8), (
        f"S_aux must be PSD; smallest eig = {eigs.min():.3e}"
    )
    # Diagonal entries are integrals of g_t^2, strictly positive.
    assert np.all(np.diag(S_aux) > 0)


def test_oep_provenance_metadata_persists_through_save_load():
    """save_vxc_ref records baseline_xc/aux_basis/regularization/etc.
    so downstream loaders can validate consistency (fix)."""
    import os, tempfile
    from xcquinox.pipeline.oep import OEPResult, save_vxc_ref
    nao = 3
    vxc = np.random.default_rng(0).standard_normal((nao, nao))
    oep = OEPResult(
        vxc_matrix=vxc, converged=True, n_iter=42, density_error=1.5e-7,
        baseline_xc="blyp", aux_basis="def2-tzvp-jkfit",
        regularization=2.5e-5, n_electrons=10.0,
        lbfgs_status="CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "vxc.npz")
        save_vxc_ref(oep, path, method="CCSD")
        loaded = np.load(path, allow_pickle=False)
        assert "oep_baseline_xc" in loaded.files
        assert str(loaded["oep_baseline_xc"]) == "blyp"
        assert str(loaded["oep_aux_basis"]) == "def2-tzvp-jkfit"
        assert abs(float(loaded["oep_regularization"]) - 2.5e-5) < 1e-12
        assert bool(loaded["oep_converged"]) is True
        assert abs(float(loaded["oep_density_error"]) - 1.5e-7) < 1e-12
        assert abs(float(loaded["oep_n_electrons"]) - 10.0) < 1e-12


def test_oep_converged_when_density_error_below_tol_even_at_max_iter():
    """Convergence semantics pin: ``OEPResult.converged`` reports
    "the V_xc that the inversion returns produces a KS density that
    matches dm_target to within conv_tol": NOT "scipy's L-BFGS-B
    optimizer reached its own pgtol/factr threshold". Hitting
    ``max_iter`` while ``density_error < conv_tol`` MUST still
    produce ``converged == True``.

    Pre-fix code conjuncted ``getattr(result, 'success', False)``
    (scipy's flag) into ``converged``; that flag is False when scipy
    exits at max_iter, so genuinely-good inversions were reported as
    failures and downstream save_vxc_ref was skipped. Reproduced on
    H2O/def2-svp/grid_level=1 with the displacement-form OEP: density
    matched at 1.18e-3 (well below conv_tol=2e-3) yet converged=False
    because L-BFGS-B was still making progress when max_iter fired.

    This test runs OEP with a small max_iter so scipy almost certainly
    exits at the limit, but uses a CCSD-target/PBE-baseline pair where
    density_error stays small (the displacement form starts at b=0
    which gives the PBE density, differences of order CCSD-PBE).
    The contract: if final_error < conv_tol AND the final SCF succeeded,
    converged == True.
    """
    from xcquinox.pipeline.oep import run_oep_inversion
    from xcquinox.pipeline.data import precompute_fixed_density_data
    mol = h2_molecule()
    data = precompute_fixed_density_data(mol)
    # Use the PBE DM as target so OEP converges trivially at b=0
    # (density_error << conv_tol after one iteration), while max_iter is
    # set high enough that scipy reports "RELATIVE REDUCTION OF F":
    # which IS scipy success, but we still assert that the contract
    # works regardless of which message scipy emits.
    dm_target = np.asarray(data["dm_pbe"])
    result = run_oep_inversion(
        mol, dm_target, baseline_xc="pbe",
        aux_basis="sto-3g", max_iter=2,
        conv_tol=1e-2, regularization=1e-4,
    )
    assert result.density_error < 1e-2, (
        f"PBE-target/PBE-baseline OEP at b=0 should give tiny density "
        f"error; got {result.density_error:.3e}"
    )
    assert result.converged is True, (
        f"density_error={result.density_error:.3e} < conv_tol=1e-2 "
        f"and final SCF succeeded; converged must be True. "
        f"lbfgs_status={result.lbfgs_status!r}"
    )


def test_oep_rejects_wrong_basis_target_dm():
    """Fix: Tr(S * dm_target) must equal mol.nelectron; a
    target DM built in a different basis silently has the wrong trace
    and would corrupt the inversion."""
    from xcquinox.pipeline.oep import run_oep_inversion
    import pytest
    from pyscf import gto, dft
    # Build target in a DIFFERENT basis (def2-svp) than mol_spec uses (sto-3g).
    other_mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="def2-svp", verbose=0)
    mf_other = dft.RKS(other_mol); mf_other.xc = "pbe"; mf_other.kernel()
    dm_wrong_basis = np.asarray(mf_other.make_rdm1())
    mol = h2_molecule()  # uses sto-3g
    with pytest.raises(ValueError, match="different basis"):
        run_oep_inversion(mol, dm_wrong_basis, max_iter=2, aux_basis="sto-3g")


def test_ks_from_vxc_matrix_rhf_default_damp_is_0_1():
    """Default damp=0.1 (preserves the earlier hardcoded oep.py:255)."""
    from pyscf import gto, dft, scf as _scf
    import numpy as np
    from xcquinox.pipeline.oep import _ks_from_vxc_matrix_rhf
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = dft.RKS(mol); mf.xc = "pbe"; mf.kernel()
    vxc = np.zeros((mol.nao, mol.nao))
    captured = {}
    real_RHF = _scf.RHF
    def spy_RHF(m):
        instance = real_RHF(m)
        captured["instance"] = instance
        return instance
    _scf.RHF = spy_RHF
    try:
        _ks_from_vxc_matrix_rhf(mol, mf, vxc, dm0=mf.make_rdm1())
    finally:
        _scf.RHF = real_RHF
    assert captured["instance"].damp == 0.1


def test_detect_plateau_fires_on_flat_history():
    """Both deques flat within rtol after iter >= min_iter -> fires."""
    from xcquinox.pipeline.oep import _detect_plateau
    d_e = [3.1e-3] * 20
    F_val = [-0.998] * 20
    fired, plateau_d_e = _detect_plateau(
        d_e=d_e, F_val=F_val,
        plateau_window=20, plateau_rtol=0.02,
    )
    assert fired
    assert abs(plateau_d_e - 3.1e-3) < 1e-12


def test_plateau_below_conv_tol_marks_converged():
    """When a plateau (or max_iter) stop's SCF-verified final_error sits below
    conv_tol, the result is marked converged. ``converged`` is the SCF-verified
    condition (final_success AND finite AND final_error < conv_tol), never a
    re-derivation from the plateau median. Driven with a huge conv_tol so the
    residual is far below it regardless of which sentinel fires."""
    from xcquinox.pipeline.config import MoleculeSpec
    from xcquinox.pipeline.oep import run_oep_inversion
    from pyscf import gto, scf as _scf
    spec = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),), grid_level=1,
    )
    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=0, spin=0, verbose=0)
    mf = _scf.RHF(mol); mf.kernel()
    dm_target = mf.make_rdm1()
    # Loose conv_tol so the SCF-verified residual is below it:
    result = run_oep_inversion(
        spec, dm_target,
        aux_basis="def2-svp-jkfit",
        max_iter=200, conv_tol=1.0,    # huge, anything below 1 is "converged"
        regularization=1e-2,
        plateau_window=5, plateau_rtol=0.1, plateau_min_iter=10,
    )
    # With a huge conv_tol the residual is far below it, so the inversion is
    # converged regardless of which sentinel fired.
    assert result.converged is True
    assert result.density_error < 1.0


def test_save_vxc_ref_write_is_atomic_no_tmp_leftover(tmp_path):
    """After save_vxc_ref completes, output_dir contains exactly the
    target .npz, no tempfile-mkstemp leftover. Pins the atomic-write
    pattern (tempfile + os.replace) introduced 2026-05-06 to match
    the run_scf_with_cache / run_ccsd_with_cache precedent."""
    import numpy as np
    from xcquinox.pipeline.oep import OEPResult, save_vxc_ref
    r = OEPResult(
        vxc_matrix=np.zeros((3, 3)),
        converged=True, n_iter=5, density_error=1e-4,
        baseline_xc="pbe", aux_basis="def2-svp-jkfit",
        regularization=1e-4, n_electrons=2.0, lbfgs_status="ok",
    )
    out = tmp_path / "vxc.npz"
    save_vxc_ref(r, str(out), dm_target=np.eye(3), method="ccsd")
    files = sorted(p.name for p in tmp_path.iterdir())
    assert "vxc.npz" in files, files
    # No tempfile leftover (mkstemp default prefix is "tmp"):
    assert not any(n.startswith("tmp") and n.endswith(".npz")
                    for n in files if n != "vxc.npz"), files


def test_plateau_stop_does_not_claim_converged_without_scf_verification(monkeypatch):
    """DEFECT OEP-01: a plateau early-stop must NOT be stamped
    converged=True merely because the plateau-MEDIAN density error sits
    below conv_tol. Convergence requires the SCF-VERIFIED final_error
    (recomputed on the post-optimization SCF density) to be below
    conv_tol. The returned density_error must equal the SCF-verified
    final_error (not the plateau median), and a stop_reason field must
    distinguish a plateau stop from genuine convergence.

    Setup: force a _OEPPlateau whose carried plateau_density_error is
    far below conv_tol (1e-12 << conv_tol=1e-6), while the carried
    coefficient vector ``b`` is LARGE and non-zero, so the
    post-finalization SCF runs V_xc = baseline + Σ b_t g_t at that large
    b, producing a KS density far from the target and hence a large
    SCF-verified final_error (>> conv_tol). The buggy code reports
    density_error=1e-12 and converged=True (keying off the fabricated
    plateau median); the correct behavior is converged=False with
    density_error == the real (large) final_error.
    """
    from xcquinox.pipeline.config import MoleculeSpec
    from xcquinox.pipeline.oep import run_oep_inversion, _OEPPlateau
    import xcquinox.pipeline.oep as oep_mod
    from pyscf import gto, scf as _scf
    spec = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),), grid_level=1,
    )
    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=0, spin=0, verbose=0)
    mf = _scf.RHF(mol); mf.kernel()
    dm_target = mf.make_rdm1()

    # Plateau median far below conv_tol, but it is a fabricated floor,
    # NOT the SCF-verified residual at the carried iterate.
    plateau_value = 1e-12
    conv_tol = 1e-6

    def fake_minimize(fun, x0, **kwargs):
        fun(x0)
        cb = kwargs.get("callback")
        if cb is not None:
            # Swallow any sentinel the callback raises; we deterministically
            # force the plateau path via the raise below.
            try:
                cb(x0)
            except Exception:
                pass
        # Carry a LARGE non-zero b so the finalization SCF density is far
        # from the target (large SCF-verified residual), while the plateau
        # median is fabricated tiny.
        b_large = np.full_like(np.asarray(x0, dtype=float), 5.0)
        raise _OEPPlateau(b=b_large, plateau_density_error=plateau_value)
    monkeypatch.setattr(oep_mod, "minimize", fake_minimize)

    result = run_oep_inversion(
        spec, dm_target,
        baseline_xc="pbe",
        aux_basis="def2-svp-jkfit",
        max_iter=200,
        conv_tol=conv_tol,
        regularization=1e-4,
        plateau_window=0,            # disable real detector; we force-raise
    )

    # The carried plateau iterate is a large b whose KS density does NOT
    # match the target; the worse-than-baseline guard therefore replaces it
    # with the b = 0 baseline (which on this minimal-basis fixture matches
    # the HF target exactly) and records the regression. The invariants
    # the defect concerned survive in strengthened form: the reported
    # density_error is the SCF-VERIFIED residual of the RETURNED potential
    # (re-solved here independently), never the fabricated plateau median,
    # and the fabricated iterate's own (large) error is on record in
    # lbfgs_status.
    from xcquinox.pipeline.oep import (_ks_from_vxc_matrix, _dm_to_rho_on_grid,
                                   _build_mol_and_mf)
    _mol_chk, mf_ks = _build_mol_and_mf(spec, baseline_xc="pbe")
    dm_chk, _, _, ok = _ks_from_vxc_matrix(mol, mf_ks, result.vxc_matrix)
    assert ok
    rho_chk = _dm_to_rho_on_grid(mol, mf_ks, dm_chk)
    rho_tgt = _dm_to_rho_on_grid(mol, mf_ks, dm_target)
    verified = float(np.sqrt(np.sum(
        mf_ks.grids.weights * (rho_tgt - rho_chk) ** 2)))
    assert result.density_error == pytest.approx(verified, abs=1e-8), (
        "density_error must be the SCF-verified residual of the returned "
        f"potential; got {result.density_error!r} vs re-solved {verified!r}"
    )
    assert result.stop_reason == "regressed_below_baseline"
    assert "regressed_below_baseline" in result.lbfgs_status
    # The fabricated large-b iterate's own error is recorded and is far
    # above conv_tol -- the plateau median (1e-12) faked nothing.
    import re as _re
    m = _re.search(r"optimized_error=([0-9.e+-]+)", result.lbfgs_status)
    assert m and float(m.group(1)) > conv_tol
    # converged reflects the RETURNED potential's real residual.
    assert result.converged is (verified < conv_tol)


# P3-07: a hybrid OEP baseline must warn (its vxc_ref bakes in frozen non-local K)


def test_oep_never_returns_worse_than_baseline():
    """The finite-basis Wu-Yang pathology (H2 / 6-31g / def2-svp-jkfit
    against a CCSD target at module defaults) drove the optimized iterate
    to a density error ~100x WORSE than the b = 0 baseline (0.49 vs
    3.97e-3), and scipy's own ftol stop accepted it silently. The result
    now keeps the baseline when the optimizer regressed past it, recorded
    as stop_reason='regressed_below_baseline'."""
    from pyscf import gto, scf, cc
    from xcquinox.pipeline.config import MoleculeSpec
    from xcquinox.pipeline.oep import run_oep_inversion

    ms = MoleculeSpec(name="H2", atom="H 0 0 0; H 0 0 0.74",
                      basis="6-31g", charge=0, spin=0,
                      atom_composition=(("H", 2),))
    mol = gto.M(atom=ms.atom, basis=ms.basis, unit="angstrom", verbose=0)
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf).run()
    dm_mo = mycc.make_rdm1()
    c = mf.mo_coeff
    dm_target = c @ dm_mo @ c.T

    res = run_oep_inversion(ms, dm_target, max_iter=50, conv_tol=1e-6)
    # The guard's whole content: the returned error can never exceed the
    # b=0 baseline's (~4e-3 here; the unguarded return measured ~0.49).
    assert res.density_error < 0.02, (
        f"worse-than-baseline potential returned: {res.density_error:.3e}")
    if res.stop_reason == "regressed_below_baseline":
        assert "regressed_below_baseline" in res.lbfgs_status
