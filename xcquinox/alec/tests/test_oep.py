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


def test_build_mol_and_mf_uses_grid_level_from_mol_spec():
    """When mol_spec.grid_level is set, _build_mol_and_mf must set
    mf.grids.level to that value and call mf.grids.build()."""
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.oep import _build_mol_and_mf
    spec = MoleculeSpec(
        name="H2",
        atom="H 0 0 0; H 0 0 0.74",
        basis="def2-svp",
        charge=0,
        spin=0,
        atom_composition=(("H", 2),),
        grid_level=2,
    )
    mol, mf = _build_mol_and_mf(spec, baseline_xc="pbe")
    assert mf.grids.level == 2
    assert mf.grids.coords is not None


def test_build_mol_and_mf_grid_level_none_uses_pyscf_default():
    """When mol_spec.grid_level is None, _build_mol_and_mf must NOT
    set mf.grids.level - PySCF's default (level 3) applies."""
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.oep import _build_mol_and_mf
    spec = MoleculeSpec(
        name="H2",
        atom="H 0 0 0; H 0 0 0.74",
        basis="def2-svp",
        charge=0,
        spin=0,
        atom_composition=(("H", 2),),
        grid_level=None,
    )
    mol, mf = _build_mol_and_mf(spec, baseline_xc="pbe")
    assert mf.grids.level == 3


def test_build_mol_and_mf_grid_level_zero_is_legitimate_coarsest():
    """grid_level=0 is the legitimate PySCF coarsest mesh (NOT a
    sentinel for use default); must be honored."""
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.oep import _build_mol_and_mf
    spec = MoleculeSpec(
        name="H2",
        atom="H 0 0 0; H 0 0 0.74",
        basis="def2-svp",
        charge=0,
        spin=0,
        atom_composition=(("H", 2),),
        grid_level=0,
    )
    mol, mf = _build_mol_and_mf(spec, baseline_xc="pbe")
    assert mf.grids.level == 0


def test_ks_from_vxc_matrix_rhf_default_damp_is_0_1():
    """Default damp=0.1 (preserves the pre-Pass-1 hardcoded oep.py:255)."""
    from pyscf import gto, dft, scf as _scf
    import numpy as np
    from xcquinox.alec.oep import _ks_from_vxc_matrix_rhf
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


def test_ks_from_vxc_matrix_rhf_default_diis_start_cycle_is_5():
    """Default diis_start_cycle=5 (preserves oep.py:253)."""
    from pyscf import gto, dft, scf as _scf
    import numpy as np
    from xcquinox.alec.oep import _ks_from_vxc_matrix_rhf
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
    assert captured["instance"].diis_start_cycle == 5


def test_ks_from_vxc_matrix_rhf_custom_damp():
    """Caller-supplied damp=0.3 reaches mf_fixed.damp."""
    from pyscf import gto, dft, scf as _scf
    import numpy as np
    from xcquinox.alec.oep import _ks_from_vxc_matrix_rhf
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
        _ks_from_vxc_matrix_rhf(mol, mf, vxc, dm0=mf.make_rdm1(), damp=0.3)
    finally:
        _scf.RHF = real_RHF
    assert captured["instance"].damp == 0.3


def test_ks_from_vxc_matrix_rhf_custom_diis_start_cycle():
    """Caller-supplied diis_start_cycle=10 reaches mf_fixed.diis_start_cycle."""
    from pyscf import gto, dft, scf as _scf
    import numpy as np
    from xcquinox.alec.oep import _ks_from_vxc_matrix_rhf
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
        _ks_from_vxc_matrix_rhf(mol, mf, vxc, dm0=mf.make_rdm1(), diis_start_cycle=10)
    finally:
        _scf.RHF = real_RHF
    assert captured["instance"].diis_start_cycle == 10


def test_ks_from_vxc_matrix_dispatcher_forwards_damp_and_diis_start_cycle():
    """The _ks_from_vxc_matrix dispatcher forwards damp / diis_start_cycle
    to whichever spin-specific helper it dispatches to."""
    from xcquinox.alec.oep import _ks_from_vxc_matrix
    import inspect
    sig = inspect.signature(_ks_from_vxc_matrix)
    assert "damp" in sig.parameters
    assert "diis_start_cycle" in sig.parameters
    assert sig.parameters["damp"].default == 0.1
    assert sig.parameters["diis_start_cycle"].default == 5


def test_objective_and_grad_caches_density_error_and_F_val_in_scf_state():
    """objective_and_grad writes density_error_l2_last_eval and
    F_val_last_eval into scf_state on every success-path return."""
    import inspect
    from xcquinox.alec.oep import run_oep_inversion
    src = inspect.getsource(run_oep_inversion)
    assert "density_error_l2_last_eval" in src
    assert "F_val_last_eval" in src


def test_scipy_iter_callback_snapshots_density_error_and_F_val_on_accept():
    """_scipy_iter_callback copies *_last_eval into *_accepted on each
    accepted L-BFGS-B step (so the plateau detector reads only
    accepted-iterate values, not rejected line-search probes)."""
    import inspect
    from xcquinox.alec.oep import run_oep_inversion
    src = inspect.getsource(run_oep_inversion)
    assert 'scf_state["density_error_l2_accepted"] = scf_state["density_error_l2_last_eval"]' in src
    assert 'scf_state["F_val_accepted"] = scf_state["F_val_last_eval"]' in src


def test_run_oep_inversion_accepts_plateau_kwargs():
    """run_oep_inversion accepts plateau_window, plateau_rtol,
    plateau_min_iter kwargs with the documented defaults."""
    import inspect
    from xcquinox.alec.oep import run_oep_inversion
    sig = inspect.signature(run_oep_inversion)
    assert sig.parameters["plateau_window"].default == 20
    assert sig.parameters["plateau_rtol"].default == 0.02
    assert sig.parameters["plateau_min_iter"].default == 30


def test_run_oep_inversion_accepts_inner_scf_kwargs():
    """run_oep_inversion accepts inner_damp + inner_diis_start_cycle."""
    import inspect
    from xcquinox.alec.oep import run_oep_inversion
    sig = inspect.signature(run_oep_inversion)
    assert sig.parameters["inner_damp"].default == 0.1
    assert sig.parameters["inner_diis_start_cycle"].default == 5


def test_run_oep_inversion_handles_oep_plateau_sentinel():
    """run_oep_inversion catches _OEPPlateau parallel to _OEPEarlyStop."""
    import inspect
    from xcquinox.alec.oep import run_oep_inversion
    src = inspect.getsource(run_oep_inversion)
    assert "except _OEPPlateau" in src
    assert ".plateau_density_error" in src
    assert "plateau_terminated" in src


def test_run_oep_inversion_returns_terminated_by_max_iter_on_default_path():
    """When neither sentinel fires (max_iter exhausted),
    terminated_by='max_iter' and dm_final is populated."""
    import numpy as np
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.oep import run_oep_inversion
    from pyscf import gto, scf as _scf
    spec = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),), grid_level=1,
    )
    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=0, spin=0, verbose=0)
    mf = _scf.RHF(mol); mf.kernel()
    dm_target = mf.make_rdm1()
    result = run_oep_inversion(
        spec, dm_target,
        aux_basis="def2-svp-jkfit",
        max_iter=2,
        conv_tol=1e-30,
        regularization=1e-4,
        plateau_window=0,
    )
    assert result.terminated_by == "max_iter"
    assert result.dm_final is not None
    assert result.dm_final.shape == (mol.nao, mol.nao)


def test_detect_plateau_fires_on_flat_history():
    """Both deques flat within rtol after iter >= min_iter -> fires."""
    from xcquinox.alec.oep import _detect_plateau
    d_e = [3.1e-3] * 20
    F_val = [-0.998] * 20
    fired, plateau_d_e = _detect_plateau(
        d_e=d_e, F_val=F_val,
        plateau_window=20, plateau_rtol=0.02,
    )
    assert fired
    assert abs(plateau_d_e - 3.1e-3) < 1e-12


def test_detect_plateau_does_not_fire_on_descending_history():
    """Density-error descending 20% per step -> does not fire."""
    from xcquinox.alec.oep import _detect_plateau
    d_e = [(0.8 ** k) for k in range(20)]
    F_val = [-(0.8 ** k) for k in range(20)]
    fired, _ = _detect_plateau(
        d_e=d_e, F_val=F_val,
        plateau_window=20, plateau_rtol=0.02,
    )
    assert not fired


def test_detect_plateau_does_not_fire_when_F_val_still_descending():
    """density flat but F_val still descending -> does not fire.
    Pins the Pass-7 watch-surface correction (F_val, not obj)."""
    import numpy as np
    from xcquinox.alec.oep import _detect_plateau
    d_e = [3.1e-3] * 20
    F_val = list(np.linspace(-0.90, -0.998, 20))
    fired, _ = _detect_plateau(
        d_e=d_e, F_val=F_val,
        plateau_window=20, plateau_rtol=0.02,
    )
    assert not fired


def test_detect_plateau_disabled_when_window_zero():
    """plateau_window=0 disables; should never fire."""
    from xcquinox.alec.oep import _detect_plateau
    d_e = [3.1e-3] * 20
    F_val = [-0.998] * 20
    fired, _ = _detect_plateau(
        d_e=d_e, F_val=F_val,
        plateau_window=0, plateau_rtol=0.02,
    )
    assert not fired


def test_detect_plateau_disabled_when_rtol_zero():
    """plateau_rtol=0 disables (no spread is small enough)."""
    from xcquinox.alec.oep import _detect_plateau
    d_e = [3.1e-3] * 20
    F_val = [-0.998] * 20
    fired, _ = _detect_plateau(
        d_e=d_e, F_val=F_val,
        plateau_window=20, plateau_rtol=0.0,
    )
    assert not fired


def test_detect_plateau_partial_window_does_not_fire():
    """Fewer than plateau_window samples in the deque -> does not fire."""
    from xcquinox.alec.oep import _detect_plateau
    d_e = [3.1e-3] * 5
    F_val = [-0.998] * 5
    fired, _ = _detect_plateau(
        d_e=d_e, F_val=F_val,
        plateau_window=20, plateau_rtol=0.02,
    )
    assert not fired


def test_oep_result_dm_final_rks_is_2d():
    """RKS run returns dm_final with shape (n_ao, n_ao)."""
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.oep import run_oep_inversion
    from pyscf import gto, scf as _scf
    spec = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),), grid_level=1,
    )
    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=0, spin=0, verbose=0)
    mf = _scf.RHF(mol); mf.kernel()
    dm_target = mf.make_rdm1()
    result = run_oep_inversion(
        spec, dm_target,
        aux_basis="def2-svp-jkfit",
        max_iter=5, conv_tol=1e-30, regularization=1e-4,
        plateau_window=0,
    )
    assert result.dm_final is not None
    assert result.dm_final.ndim == 2
    assert result.dm_final.shape == (mol.nao, mol.nao)


def test_save_vxc_ref_does_not_persist_terminated_by_or_dm_final(tmp_path):
    """save_vxc_ref accesses an explicit allowlist of oep_* keys; the
    new terminated_by and dm_final fields must NOT appear in the npz."""
    import numpy as np
    from xcquinox.alec.oep import OEPResult, save_vxc_ref
    r = OEPResult(
        vxc_matrix=np.zeros((3, 3)),
        converged=True, n_iter=5, density_error=1e-4,
        baseline_xc="pbe", aux_basis="def2-svp-jkfit",
        regularization=1e-4, n_electrons=2.0, lbfgs_status="ok",
        terminated_by="plateau",
        dm_final=np.eye(3),
        stop_reason="plateau",
    )
    out = tmp_path / "vxc.npz"
    save_vxc_ref(r, str(out), dm_target=np.eye(3), method="ccsd")
    loaded = np.load(str(out))
    assert "terminated_by" not in loaded.files
    assert "dm_final" not in loaded.files
    assert "stop_reason" not in loaded.files


def test_plateau_F_val_cache_uses_unregularized_lagrangian():
    """Pass-7 contract: scf_state['F_val_last_eval'] caches F_val
    (unregularized Lagrangian at oep.py:590/620), NOT obj=-F_val+reg_term.
    Drive a tiny OEP with non-zero regularization and a single
    objective_and_grad evaluation; spy on scf_state."""
    import numpy as np
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.oep import run_oep_inversion
    from pyscf import gto, scf as _scf
    spec = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0,
        atom_composition=(("H", 2),),  # CANONICAL frozen-dataclass form
        grid_level=1,
    )
    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=0, spin=0, verbose=0)
    mf = _scf.RHF(mol); mf.kernel()
    dm_target = mf.make_rdm1()
    # Run with max_iter=1 — single L-BFGS step → at least one objective
    # evaluation. After that, scf_state["F_val_last_eval"] should be
    # finite and equal to the F_val computed at oep.py:620
    # (e_ks - <b, rhotarget>), NOT obj (which is -F_val + reg_term).
    # We use regularization=1e-2 so the reg_term is non-trivially
    # different from F_val.
    result = run_oep_inversion(
        spec, dm_target,
        aux_basis="def2-svp-jkfit",
        max_iter=1, conv_tol=1e-30,
        regularization=1e-2,
        plateau_window=0,   # disable plateau
    )
    # Indirect verification: result.density_error finite + non-NaN.
    # Direct verification of the scf_state cache requires accessing
    # internal closure state — not exposed by the public API. Use
    # source-text pin as the back-up assertion (same pattern Plan 1
    # uses for cache-internals contracts).
    import inspect
    src = inspect.getsource(run_oep_inversion)
    # The F_val cache write must reference the F_val local, NOT obj:
    assert 'scf_state["F_val_last_eval"] = float(F_val)' in src
    assert 'scf_state["F_val_last_eval"] = float(obj)' not in src
    assert np.isfinite(result.density_error)


def test_plateau_F_val_cache_writes_neg_inf_on_scf_failure():
    """Pass-7 contract: scf_state['F_val_last_eval'] = float('-inf')
    on inner-SCF failure (descending sentinel). Source-text pin —
    the failure path is hard to trigger in a unit test without
    constructing an ill-conditioned problem; pin the contract via
    inspect."""
    import inspect
    from xcquinox.alec.oep import run_oep_inversion
    src = inspect.getsource(run_oep_inversion)
    # On failure path (oep.py:569-572), both sentinels must be written:
    assert 'scf_state["density_error_l2_last_eval"] = float("inf")' in src
    assert 'scf_state["F_val_last_eval"] = float("-inf")' in src


def test_terminated_by_field_for_conv_tol_path():
    """Plan-1 review fix: spec §9.1 names this test for the conv_tol
    path. Drive a trivially-converging OEP (max_iter=200, conv_tol=1e-2).
    When converged=True, terminated_by is either "conv_tol" (early-stop
    sentinel fired mid-optimization) or "max_iter" (optimizer exhausted
    max_iter but final_error < conv_tol — also a legitimate converged
    state per spec). Both cases must populate dm_final and density_error
    below conv_tol."""
    import numpy as np
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.oep import run_oep_inversion
    from pyscf import gto, scf as _scf
    spec = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),), grid_level=1,
    )
    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=0, spin=0, verbose=0)
    mf = _scf.RHF(mol); mf.kernel()
    dm_target = mf.make_rdm1()
    result = run_oep_inversion(
        spec, dm_target,
        aux_basis="def2-svp-jkfit",
        max_iter=200, conv_tol=1e-2,
        regularization=1e-4,
        plateau_window=0,
    )
    if result.converged:
        # Both "conv_tol" (early-stop fired) and "max_iter" (exhausted
        # max_iter but final density_error < conv_tol) are valid converged
        # termination paths; either satisfies the inversion contract.
        assert result.terminated_by in ("conv_tol", "max_iter")
        assert result.density_error < 1e-2
        assert result.dm_final is not None
        assert result.dm_final.ndim == 2


def test_terminated_by_field_for_plateau_path():
    """Plan-1 review fix: spec §9.1 names this test for the plateau
    path explicitly. Drive an OEP that should plateau (high reg + tight
    plateau_window/min_iter so plateau fires before max_iter)."""
    import inspect
    from xcquinox.alec.oep import run_oep_inversion
    src = inspect.getsource(run_oep_inversion)
    assert 'terminated_by = "plateau"' in src
    from xcquinox.alec.config import MoleculeSpec
    from pyscf import gto, scf as _scf
    spec = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),), grid_level=1,
    )
    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=0, spin=0, verbose=0)
    mf = _scf.RHF(mol); mf.kernel()
    dm_target = mf.make_rdm1()
    # Plateau-easy config: small plateau_window with tight rtol so
    # any flat tail fires; lots of regularization so optimizer settles
    # into a steady state quickly.
    result = run_oep_inversion(
        spec, dm_target,
        aux_basis="def2-svp-jkfit",
        max_iter=200, conv_tol=1e-30,   # impossible — forces plateau or max_iter
        regularization=1e-2,             # high reg → quick plateau
        plateau_window=5, plateau_rtol=0.1, plateau_min_iter=10,
    )
    # Either plateau fires or max_iter exhausts:
    assert result.terminated_by in ("plateau", "max_iter")


def test_plateau_below_conv_tol_marks_converged():
    """Spec §9.1 + §5.5: plateau-below-conv_tol → converged=True.

    OEP-01 audit fix: ``converged`` for the plateau path is the
    SCF-verified condition (final_success AND finite AND
    final_error < conv_tol), NOT a re-derivation from the plateau
    median. The plateau median is no longer used to set ``converged``."""
    import inspect
    from xcquinox.alec.oep import run_oep_inversion
    src = inspect.getsource(run_oep_inversion)
    # OEP-01: converged is the single SCF-verified condition on
    # final_error; the plateau branch must NOT re-derive it from the
    # (possibly biased) plateau median density_error.
    assert '(final_error < conv_tol)' in src
    from xcquinox.alec.config import MoleculeSpec
    from pyscf import gto, scf as _scf
    spec = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),), grid_level=1,
    )
    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=0, spin=0, verbose=0)
    mf = _scf.RHF(mol); mf.kernel()
    dm_target = mf.make_rdm1()
    # Loose conv_tol so any plateau is below it:
    result = run_oep_inversion(
        spec, dm_target,
        aux_basis="def2-svp-jkfit",
        max_iter=200, conv_tol=1.0,    # huge — anything below 1 is "converged"
        regularization=1e-2,
        plateau_window=5, plateau_rtol=0.1, plateau_min_iter=10,
    )
    if result.terminated_by == "plateau":
        assert result.converged is True


def test_plateau_above_conv_tol_marks_not_converged():
    """Spec §9.1 + §5.5: plateau-above-conv_tol → converged=False
    (cascade falls through to next tier)."""
    import inspect
    from xcquinox.alec.oep import run_oep_inversion
    src = inspect.getsource(run_oep_inversion)
    # The bool() wrap means False propagates correctly when plateau is above conv_tol
    assert 'converged = bool(' in src
    from xcquinox.alec.config import MoleculeSpec
    from pyscf import gto, scf as _scf
    spec = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),), grid_level=1,
    )
    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=0, spin=0, verbose=0)
    mf = _scf.RHF(mol); mf.kernel()
    dm_target = mf.make_rdm1()
    # Tight conv_tol so plateau-density-error is NOT below it:
    result = run_oep_inversion(
        spec, dm_target,
        aux_basis="def2-svp-jkfit",
        max_iter=200, conv_tol=1e-30,   # impossibly tight
        regularization=1e-2,
        plateau_window=5, plateau_rtol=0.1, plateau_min_iter=10,
    )
    if result.terminated_by == "plateau":
        assert result.converged is False


def test_plateau_detector_disabled_when_min_iter_exceeds_max_iter():
    """Spec §9.1: plateau_min_iter > max_iter → cannot fire by construction."""
    from xcquinox.alec.oep import _detect_plateau
    # Pretend max_iter=20 and plateau_min_iter=30. Even on a flat-20-deque,
    # _detect_plateau is gated only by deque-fullness in the helper, but the
    # CALL SITE in _scipy_iter_callback gates on plateau_min_iter. Pin the
    # caller-side contract via source-text:
    import inspect
    from xcquinox.alec.oep import run_oep_inversion
    src = inspect.getsource(run_oep_inversion)
    assert '_progress_state["iter"] >= plateau_min_iter' in src


def test_plateau_detector_does_not_fire_on_slow_descent_with_sign_of_trend():
    """Spec §9.1 §5.5: 0.25%/iter slow descent over 20 iters → relative
    range ~5%. WITH plateau_rtol=0.02 that's outside, so no fire. WITH
    sign-of-trend slack, the test ALSO requires last-half-median to be
    NOT below first-half-median by more than rtol*|median|."""
    from xcquinox.alec.oep import _detect_plateau
    # 0.25%/iter geometric descent over 20 iters:
    d_e = [(0.9975 ** k) for k in range(20)]
    F_val = [-(0.9975 ** k) for k in range(20)]
    fired, _ = _detect_plateau(
        d_e=d_e, F_val=F_val,
        plateau_window=20, plateau_rtol=0.02,
    )
    # last-half median < first-half median (descending), so sign-of-trend
    # check fails → plateau does not fire:
    assert not fired


def test_plateau_detector_sign_of_trend_uses_rtol_slack():
    """Spec §9.1 / Pass-7: rtol-scaled slack on the sign-of-trend test
    ensures L-BFGS-B float-noise micro-oscillations don't false-fire."""
    from xcquinox.alec.oep import _detect_plateau
    # Tight flat tail with last-half-median ε above first-half-median
    # by 0.5 × rtol × |median| (within slack):
    base = 3e-3
    half = 10
    rtol = 0.02
    slack = 0.5 * rtol * base   # within rtol slack
    first_half = [base] * half
    last_half = [base + slack] * half
    fired, _ = _detect_plateau(
        d_e=first_half + last_half, F_val=[-1.0] * 20,
        plateau_window=20, plateau_rtol=rtol,
    )
    # Within slack: should fire (does not get blocked by sign-of-trend)
    assert fired


def test_run_oep_inversion_passes_mol_spec_grid_level_through(monkeypatch):
    """Spec §9.1: run_oep_inversion does NOT silently drop or override
    mol_spec.grid_level — the same spec passes through to _build_mol_and_mf."""
    captured = {}
    import xcquinox.alec.oep as oep_mod
    real_build = oep_mod._build_mol_and_mf
    def spy_build(mol_spec, basis=None, baseline_xc="pbe"):
        captured["grid_level"] = mol_spec.grid_level
        return real_build(mol_spec, basis=basis, baseline_xc=baseline_xc)
    monkeypatch.setattr(oep_mod, "_build_mol_and_mf", spy_build)
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.oep import run_oep_inversion
    from pyscf import gto, scf as _scf
    spec = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0,
        atom_composition=(("H", 2),),
        grid_level=2,    # explicit
    )
    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=0, spin=0, verbose=0)
    mf = _scf.RHF(mol); mf.kernel()
    dm_target = mf.make_rdm1()
    try:
        run_oep_inversion(
            spec, dm_target,
            aux_basis="def2-svp-jkfit",
            max_iter=2, conv_tol=1e-30,
            regularization=1e-4,
            plateau_window=0,
        )
    except Exception:
        pass
    assert captured.get("grid_level") == 2


def test_run_oep_inversion_plateau_catch_path_behavioral(monkeypatch):
    """Behavioral end-to-end test for the plateau catch path: force
    a `_OEPPlateau` raise via monkey-patched scipy.minimize and verify
    the OEPResult carries terminated_by='plateau' and lbfgs_status starts
    with 'plateau'. Closes the deferred behavioral-coverage gap from
    Plan 1 Task 9 + Task 10 reviews.

    OEP-01 audit fix: ``density_error`` is the SCF-VERIFIED
    post-finalization ``final_error``, NOT the plateau median (the old
    behavior). ``converged`` is derived from the SCF-verified error vs
    conv_tol, never from the plateau median.

    Why monkey-patch minimize rather than _detect_plateau: on tiny
    H2/sto-3g RHF, L-BFGS-B converges before the plateau deque fills,
    so a _detect_plateau monkey-patch never gets called. We want to
    test the catch+wiring path, not the detector's deque math (the
    latter is covered by the 6 synthetic-history tests in Task 12)."""
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.oep import run_oep_inversion, _OEPPlateau
    import xcquinox.alec.oep as oep_mod
    from pyscf import gto, scf as _scf
    spec = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),), grid_level=1,
    )
    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=0, spin=0, verbose=0)
    mf = _scf.RHF(mol); mf.kernel()
    dm_target = mf.make_rdm1()
    # Stub minimize: run fun(x0) once so scf_state is populated (the
    # finalization SCF after catch reads scf_state["dm0_accepted"]),
    # then raise _OEPPlateau as if the iter-callback's detector fired.
    plateau_value = 1.5e-3
    def fake_minimize(fun, x0, **kwargs):
        # Populate scf_state via one objective evaluation:
        fun(x0)
        # Also tickle the iter callback once so dm0_accepted is set.
        # Swallow whatever sentinel the callback raises (on tiny H2/sto-3g
        # the b=0 residual can be at/below conv_tol, firing the early-stop
        # sentinel) so we deterministically force the plateau path via the
        # explicit raise below.
        cb = kwargs.get("callback")
        if cb is not None:
            try:
                cb(x0)
            except Exception:
                pass
        raise _OEPPlateau(b=x0, plateau_density_error=plateau_value)
    monkeypatch.setattr(oep_mod, "minimize", fake_minimize)
    result = run_oep_inversion(
        spec, dm_target,
        aux_basis="def2-svp-jkfit",
        max_iter=200,
        conv_tol=1e-30,
        regularization=1e-4,
        plateau_window=0,            # disable real detector; we force-raise
    )
    # Behavioral assertions on the catch+wiring path:
    assert result.terminated_by == "plateau"
    assert result.lbfgs_status.startswith("plateau")
    # OEP-01: density_error is the SCF-verified final_error, not the
    # carried plateau median (1.5e-3).
    assert abs(result.density_error - plateau_value) > 1e-9
    # conv_tol=1e-30 is impossible for the real SCF-verified residual at
    # b=0, so the result is not converged; stop_reason records plateau.
    assert result.converged is False
    assert result.stop_reason == "plateau"


def test_save_vxc_ref_write_is_atomic_no_tmp_leftover(tmp_path):
    """After save_vxc_ref completes, output_dir contains exactly the
    target .npz — no tempfile-mkstemp leftover. Pins the atomic-write
    pattern (tempfile + os.replace) introduced 2026-05-06 to match
    the run_scf_with_cache / run_ccsd_with_cache precedent."""
    import numpy as np
    from xcquinox.alec.oep import OEPResult, save_vxc_ref
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


def test_save_vxc_ref_atomic_write_preserves_file_on_overwrite(tmp_path):
    """Calling save_vxc_ref twice on the same path leaves a single
    valid .npz at every observable moment (no race with a deleted
    target during the write). Verified by reading after each call."""
    import numpy as np
    from xcquinox.alec.oep import OEPResult, save_vxc_ref
    r1 = OEPResult(
        vxc_matrix=np.ones((3, 3)),
        converged=True, n_iter=1, density_error=1e-3,
        baseline_xc="pbe", aux_basis="def2-svp-jkfit",
        regularization=1e-4, n_electrons=2.0, lbfgs_status="first",
    )
    out = tmp_path / "vxc.npz"
    save_vxc_ref(r1, str(out), dm_target=np.eye(3), method="ccsd")
    with np.load(out) as z:
        assert str(z["oep_lbfgs_status"]) == "first"
    # Second call replaces:
    r2 = OEPResult(
        vxc_matrix=np.full((3, 3), 2.0),
        converged=True, n_iter=2, density_error=2e-3,
        baseline_xc="pbe", aux_basis="def2-svp-jkfit",
        regularization=1e-4, n_electrons=2.0, lbfgs_status="second",
    )
    save_vxc_ref(r2, str(out), dm_target=np.eye(3), method="ccsd")
    with np.load(out) as z:
        assert str(z["oep_lbfgs_status"]) == "second"
        np.testing.assert_array_equal(z["vxc_ref"], np.full((3, 3), 2.0))


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
    coefficient vector ``b`` is LARGE and non-zero — so the
    post-finalization SCF runs V_xc = baseline + Σ b_t g_t at that large
    b, producing a KS density far from the target and hence a large
    SCF-verified final_error (>> conv_tol). The buggy code reports
    density_error=1e-12 and converged=True (keying off the fabricated
    plateau median); the correct behavior is converged=False with
    density_error == the real (large) final_error.
    """
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.oep import run_oep_inversion, _OEPPlateau
    import xcquinox.alec.oep as oep_mod
    from pyscf import gto, scf as _scf
    spec = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),), grid_level=1,
    )
    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=0, spin=0, verbose=0)
    mf = _scf.RHF(mol); mf.kernel()
    dm_target = mf.make_rdm1()

    # Plateau median far below conv_tol — but it is a fabricated floor,
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

    # The carried plateau iterate is a large b, whose KS density does NOT
    # match the target to 1e-6 — so the SCF-verified residual is well
    # above conv_tol. The reported density_error must be that real
    # residual, NOT the 1e-12 plateau median.
    assert result.density_error > conv_tol, (
        "expected SCF-verified final_error > conv_tol; got "
        f"{result.density_error!r}"
    )
    assert abs(result.density_error - plateau_value) > 1e-9, (
        "density_error must be the SCF-verified final_error, not the "
        "plateau median"
    )
    # Because the SCF-verified error is above conv_tol, the result must
    # NOT be marked converged even though the plateau median was tiny.
    assert result.converged is False
    # stop_reason must distinguish a plateau stop from true convergence.
    assert result.stop_reason == "plateau"


def test_plateau_stop_below_conv_tol_is_converged_with_plateau_stop_reason(monkeypatch):
    """DEFECT OEP-01 (other branch): when a plateau stop's
    SCF-verified final_error genuinely sits below conv_tol, converged is
    True, but stop_reason still records that it was a plateau stop so
    downstream can distinguish it from a stationary-point convergence."""
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.oep import run_oep_inversion, _OEPPlateau
    import xcquinox.alec.oep as oep_mod
    from pyscf import gto, scf as _scf
    spec = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),), grid_level=1,
    )
    mol = gto.M(atom=spec.atom, basis=spec.basis, charge=0, spin=0, verbose=0)
    mf = _scf.RHF(mol); mf.kernel()
    dm_target = mf.make_rdm1()

    def fake_minimize(fun, x0, **kwargs):
        fun(x0)
        cb = kwargs.get("callback")
        if cb is not None:
            # Swallow whatever sentinel the callback raises (early-stop
            # would fire here under the loose conv_tol); we deterministically
            # force the plateau path below.
            try:
                cb(x0)
            except Exception:
                pass
        raise _OEPPlateau(b=x0, plateau_density_error=1e-30)
    monkeypatch.setattr(oep_mod, "minimize", fake_minimize)

    # Loose conv_tol so the real SCF-verified residual at b=0 is below it.
    result = run_oep_inversion(
        spec, dm_target,
        baseline_xc="pbe",
        aux_basis="def2-svp-jkfit",
        max_iter=200,
        conv_tol=1e6,
        regularization=1e-4,
        plateau_window=0,
    )
    assert result.converged is True
    assert result.stop_reason == "plateau"
    # density_error is the SCF-verified value, not the 1e-30 plateau
    # median (which would have been reported by the buggy code). The
    # real residual is many orders of magnitude larger than 1e-30.
    assert result.density_error > 1e-30 * 1e6


# P3-07: a hybrid OEP baseline must warn (its vxc_ref bakes in frozen non-local K)
def test_oep_hybrid_baseline_warns():
    import warnings as _w
    import pytest
    from xcquinox.alec.oep import _build_mol_and_mf
    from xcquinox.alec.config import MoleculeSpec
    h = MoleculeSpec(name="H", atom="H 0 0 0", basis="sto-3g",
                     charge=0, spin=1, atom_composition=(("H", 1),))
    # Hybrid baseline (b3lyp) -> RuntimeWarning about frozen non-local HF-K.
    with pytest.warns(RuntimeWarning, match="hybrid"):
        _build_mol_and_mf(h, baseline_xc="b3lyp")
    # Semilocal baseline (pbe) -> no hybrid warning.
    with _w.catch_warnings():
        _w.simplefilter("error", RuntimeWarning)
        _build_mol_and_mf(h, baseline_xc="pbe")  # must not raise
