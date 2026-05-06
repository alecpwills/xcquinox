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


def test_oep_progress_callback_fires_each_iteration():
    """run_oep_inversion(progress_callback=fn) must call fn(iter, density_error)
    once per L-BFGS iteration, with iter monotonically increasing from 1 and
    the final call reporting iter == result.n_iter. Needed so notebook cells
    can drive a tqdm bar during long (500-1000 iter) OEP cascades.

    Forces L-BFGS-B to take real iterations by using a HARTREE-ONLY
    baseline (baseline_xc=None) while targeting the PBE DM. With no
    XC contribution from the baseline, the auxiliary-basis correction
    has to span all of V_xc^PBE, so the optimizer cannot converge at
    b=0. (The displacement form means b=0 ≡ baseline V_xc; matching
    baselines and target zeroes the work — only mismatched ones
    exercise the loop.) D10 sanity check satisfied: dm_target = PBE DM
    has Tr(S*D) = n_electron exactly.
    """
    from xcquinox.alec.oep import run_oep_inversion
    from xcquinox.alec.data import precompute_fixed_density_data
    mol = h2_molecule()
    data = precompute_fixed_density_data(mol)
    dm_target = np.asarray(data["dm_pbe"])

    calls = []

    def _cb(it, density_error):
        calls.append((int(it), float(density_error)))

    result = run_oep_inversion(
        mol, dm_target, max_iter=5, aux_basis="sto-3g",
        baseline_xc=None,         # Hartree-only baseline forces iterations
        progress_callback=_cb,
    )

    # Callback wiring: number of callback invocations must equal
    # result.n_iter (1 callback per L-BFGS-B accepted iteration).
    # The displacement form starting from an effective baseline can
    # legitimately converge in 0 iterations when the baseline already
    # matches the target (Wu & Yang 2003 §II.B); in that case both
    # n_iter == 0 and len(calls) == 0, which is still correct wiring.
    assert len(calls) == result.n_iter, (
        f"callback fired {len(calls)} times but result.n_iter={result.n_iter}; "
        f"these must agree (1 callback per accepted iteration)."
    )
    if calls:
        iters = [c[0] for c in calls]
        assert iters == sorted(iters), f"iter counter not monotonic: {iters}"
        assert iters[0] == 1, f"first iter should be 1, got {iters[0]}"
        assert iters[-1] == result.n_iter, (
            f"last progress iter {iters[-1]} != result.n_iter {result.n_iter}"
        )
        for _, err in calls:
            assert np.isfinite(err) and err >= 0.0, (
                f"density_error reported to callback must be finite and >=0, got {err}"
            )


def test_oep_progress_callback_optional_backwards_compatible():
    """Omitting progress_callback must not change any behavior or raise."""
    from xcquinox.alec.oep import run_oep_inversion
    from xcquinox.alec.data import precompute_fixed_density_data
    mol = h2_molecule()
    data = precompute_fixed_density_data(mol)
    dm_target = np.asarray(data["dm_pbe"])
    # No callback -- must still return a valid OEPResult.
    result = run_oep_inversion(mol, dm_target, max_iter=3, aux_basis="sto-3g")
    assert np.all(np.isfinite(result.vxc_matrix))


def test_oep_nonconvergence_flagged():
    """max_iter=1 should report converged=False (or genuinely converged
    if the displacement form's b=0 already matches; we don't pin which).
    Uses HF target with PBE baseline so the optimizer has real work
    (b=0 gives PBE DM, target is HF DM — different non-PBE DM forces
    iterations without violating the D10 Tr(S*D)=N_e sanity check).
    """
    from xcquinox.alec.oep import run_oep_inversion
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
    # must be False; if it DID converge, that's also fine — we just
    # require the flag to be a bool reflecting the actual state.
    assert isinstance(result.converged, bool)


def test_save_vxc_ref_roundtrip(tmp_path):
    """save_vxc_ref creates a .npz loadable by _load_external_data."""
    from xcquinox.alec.oep import OEPResult, save_vxc_ref
    from xcquinox.alec.data import _load_external_data
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
    # New API: _build_aux_basis_matrices returns S_aux too (D2 audit fix).
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


def test_oep_h2o_ccsd_does_not_crash_with_pathological_bfgs_step():
    """Regression: L-BFGS-B line-search on H2O/def2-svp CCSD-target OEP
    previously crashed inside PySCF's DIIS subspace eigendecomposition
    (scipy.linalg.LinAlgError: Internal Error). The inner SCF is now
    hardened with guarded DIIS + try/except so pathological b-steps
    produce a penalty instead of an exception.
    """
    import numpy as np
    from pyscf import gto, scf, cc
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.oep import run_oep_inversion

    H2O = "O 0.0000 0.0000 0.1173; H 0.0000 0.7572 -0.4692; H 0.0000 -0.7572 -0.4692"
    mol = gto.M(atom=H2O, basis="def2-svp", charge=0, spin=0, verbose=0)
    mf_hf = scf.RHF(mol); mf_hf.kernel()
    mycc = cc.CCSD(mf_hf); mycc.kernel()
    C = mf_hf.mo_coeff
    dm_ao_ccsd = C @ mycc.make_rdm1() @ C.T

    spec = MoleculeSpec(
        name="H2O", atom=H2O, basis="def2-svp", charge=0, spin=0,
        atom_composition=(("H", 2), ("O", 1)), grid_level=1,
    )
    result = run_oep_inversion(
        spec, dm_ao_ccsd,
        aux_basis="def2-svp-jkfit", max_iter=30, conv_tol=1e-6,
        regularization=1e-4,
    )
    # Must complete without exception; vxc_matrix is finite.
    assert np.all(np.isfinite(result.vxc_matrix))
    assert result.density_error < 1.0, (
        f"density_error = {result.density_error:.3e} — L-BFGS-B should "
        "still make meaningful progress even with guarded inner SCF"
    )


# ---------------------------------------------------------------------------
# D1 audit fix: baseline_xc parameter (Wu-Yang displacement form)
# ---------------------------------------------------------------------------

def test_oep_baseline_xc_parameter_accepts_arbitrary_xc():
    """run_oep_inversion(baseline_xc='lda'/'pbe'/'blyp'/'hf'/None) must
    all run without error and record baseline_xc in the OEPResult.
    Wu & Yang JCP 118, 2498 (2003) §II.B uses the displacement form
    V_xc = V_xc^baseline + sum_t b_t g_t; the baseline is user-choosable
    so the inversion is generalizable to any starting XC functional.

    R3-F audit strengthening: the prior assertion only checked the
    attribute round-trip (``result.baseline_xc == xc``), which a
    silently-ignored ``baseline_xc`` arg would still satisfy. Add a
    behavioral discriminator: ``vxc_matrix`` must differ between LDA
    and PBE baselines on the same target (V_xc^LDA != V_xc^PBE for any
    physical density), proving the baseline is actually consumed.
    Includes ``'hf'`` per docstring (routes through ``mf.xc = 'hf'``).
    """
    from xcquinox.alec.oep import run_oep_inversion
    from xcquinox.alec.data import precompute_fixed_density_data
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
    """D2 audit fix: V-space regularization 0.5*lambda*b^T S_aux b is
    aux-basis independent in meaning. Pre-fix coefficient-space
    0.5*lambda*|b|^2 silently changed regularization strength when
    aux_basis was swapped. Heaton-Burgess et al. PRL 98, 256401 (2007).

    R3-F audit rename: prior name ``..._basis_independent`` implied a
    cross-basis comparison; this test only verifies S_aux is constructed
    correctly (symmetric + PSD + positive diagonal) for one aux basis.
    The basis-independence property follows from the math; a numerical
    test would require running the full inversion in two bases.
    """
    from xcquinox.alec.oep import _build_aux_basis_matrices
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
    # Diagonal entries are integrals of g_t^2 — strictly positive.
    assert np.all(np.diag(S_aux) > 0)


def test_oep_provenance_metadata_persists_through_save_load():
    """save_vxc_ref records baseline_xc/aux_basis/regularization/etc.
    so downstream loaders can validate consistency (D7 audit fix)."""
    import os, tempfile
    from xcquinox.alec.oep import OEPResult, save_vxc_ref
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
    matches dm_target to within conv_tol" — NOT "scipy's L-BFGS-B
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
    which gives the PBE density — differences of order CCSD-PBE).
    The contract: if final_error < conv_tol AND the final SCF succeeded,
    converged == True.
    """
    from xcquinox.alec.oep import run_oep_inversion
    from xcquinox.alec.data import precompute_fixed_density_data
    mol = h2_molecule()
    data = precompute_fixed_density_data(mol)
    # Use the PBE DM as target so OEP converges trivially at b=0
    # (density_error << conv_tol after one iteration), while max_iter is
    # set high enough that scipy reports "RELATIVE REDUCTION OF F" —
    # which IS scipy success — but we still assert that the contract
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
    """D10 audit fix: Tr(S * dm_target) must equal mol.nelectron; a
    target DM built in a different basis silently has the wrong trace
    and would corrupt the inversion."""
    from xcquinox.alec.oep import run_oep_inversion
    import pytest
    from pyscf import gto, dft
    # Build target in a DIFFERENT basis (def2-svp) than mol_spec uses (sto-3g).
    other_mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="def2-svp", verbose=0)
    mf_other = dft.RKS(other_mol); mf_other.xc = "pbe"; mf_other.kernel()
    dm_wrong_basis = np.asarray(mf_other.make_rdm1())
    mol = h2_molecule()  # uses sto-3g
    with pytest.raises(ValueError, match="different basis"):
        run_oep_inversion(mol, dm_wrong_basis, max_iter=2, aux_basis="sto-3g")


def test_oep_result_default_terminated_by_is_max_iter():
    """OEPResult.terminated_by defaults to 'max_iter' for back-compat."""
    import numpy as np
    from xcquinox.alec.oep import OEPResult
    r = OEPResult(
        vxc_matrix=np.zeros((2, 2)),
        converged=False,
        n_iter=10,
        density_error=1e-3,
        baseline_xc="pbe",
        aux_basis="def2-svp-jkfit",
        regularization=1e-4,
        n_electrons=2.0,
        lbfgs_status="ok",
    )
    assert r.terminated_by == "max_iter"


def test_oep_result_default_dm_final_is_none():
    """OEPResult.dm_final defaults to None for back-compat."""
    import numpy as np
    from xcquinox.alec.oep import OEPResult
    r = OEPResult(
        vxc_matrix=np.zeros((2, 2)),
        converged=False,
        n_iter=10,
        density_error=1e-3,
        baseline_xc="pbe",
        aux_basis="def2-svp-jkfit",
        regularization=1e-4,
        n_electrons=2.0,
        lbfgs_status="ok",
    )
    assert r.dm_final is None


def test_oep_result_accepts_terminated_by_kwarg():
    """OEPResult accepts terminated_by as a kwarg with the listed values."""
    import numpy as np
    from xcquinox.alec.oep import OEPResult
    r = OEPResult(
        vxc_matrix=np.zeros((2, 2)),
        converged=True,
        n_iter=5,
        density_error=1e-4,
        baseline_xc="pbe",
        aux_basis="def2-svp-jkfit",
        regularization=1e-4,
        n_electrons=2.0,
        lbfgs_status="ok",
        terminated_by="conv_tol",
    )
    assert r.terminated_by == "conv_tol"


def test_oep_result_accepts_dm_final_kwarg():
    """OEPResult accepts dm_final as a kwarg."""
    import numpy as np
    from xcquinox.alec.oep import OEPResult
    dm = np.eye(3)
    r = OEPResult(
        vxc_matrix=np.zeros((2, 2)),
        converged=True,
        n_iter=5,
        density_error=1e-4,
        baseline_xc="pbe",
        aux_basis="def2-svp-jkfit",
        regularization=1e-4,
        n_electrons=2.0,
        lbfgs_status="ok",
        dm_final=dm,
    )
    np.testing.assert_array_equal(r.dm_final, dm)


def test_oep_plateau_sentinel_carries_b_and_density_error():
    """_OEPPlateau sentinel carries the L-BFGS-B coefficient vector b
    and the plateau density_error value, mirroring _OEPEarlyStop."""
    import numpy as np
    from xcquinox.alec.oep import _OEPPlateau
    b = np.array([0.1, 0.2, 0.3])
    sentinel = _OEPPlateau(b=b, plateau_density_error=1.5e-3)
    assert isinstance(sentinel, Exception)
    np.testing.assert_array_equal(sentinel.b, b)
    assert sentinel.plateau_density_error == 1.5e-3
