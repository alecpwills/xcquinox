"""Tests for ``xcquinox.features.compute_dm_features``.

Lives under alec/tests/ so the existing CI alec-test step picks it up
(per the new step added to .github/workflows/CI.yaml). The functions
are in the legacy library; the alec subpackage consumes them via
``descriptors.DMStatisticsDescriptor.compute_from_dm`` and the
``data.precompute_fixed_density_data`` path.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from xcquinox.features import compute_dm_features, compute_dm_features_array


def _build_clean_rks_dm(nao: int, nocc: int, seed: int = 0):
    """Build a clean closed-shell RKS DM in a synthetic non-orthogonal AO basis.

    Returns (D, S) where D = 2 C_occ C_occ^T and C is an eigenbasis of a
    random symmetric Fock under metric S. The relation D S D = 2 D
    (Szabo & Ostlund 1996 §3.4.2 eq. (3.144)) holds exactly by
    construction.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((nao, nao))
    S = A @ A.T + 2.0 * np.eye(nao)
    S = S / S[0, 0]                       # normalize for stability
    L = np.linalg.cholesky(S)
    Linv = np.linalg.inv(L)
    F = rng.standard_normal((nao, nao))
    F = (F + F.T) / 2.0
    F_orth = Linv @ F @ Linv.T
    _, U = np.linalg.eigh(F_orth)
    C = Linv.T @ U
    C_occ = C[:, :nocc]
    D = 2.0 * C_occ @ C_occ.T
    return jnp.array(D), jnp.array(S)


def test_idempotency_error_zero_for_clean_rks_dm():
    """Pin: a clean closed-shell KS DM (D = 2P with PSP = P) gives
    idempotency_error == 0 to machine precision. The pre-fix formula
    `Tr(D - DSD)/Tr(DS)` gave ≈ -1 (a near-constant useless feature).
    """
    D, S = _build_clean_rks_dm(nao=8, nocc=3, seed=0)
    out = compute_dm_features(D, S)
    err = float(out["idempotency_error"])
    assert abs(err) < 1e-6, (
        f"clean RKS DM should give idempotency_error ≈ 0; got {err}. "
        "If this is ~ -1 or ~ 0.307, the spin-orbital normalization "
        "(D_norm = D/2 for RKS) was reverted."
    )


def test_idempotency_error_zero_for_clean_uks_dm():
    """Pin: clean UKS DM (D_α S D_α = D_α; same for β) gives 0.
    Spin-resolved Frobenius norm averaged across spins."""
    D_a, S = _build_clean_rks_dm(nao=8, nocc=3, seed=1)
    D_b, _ = _build_clean_rks_dm(nao=8, nocc=2, seed=2)
    # halve each (RKS used factor-of-2; UKS spin-orbital DM has no factor)
    D_a = D_a / 2.0
    D_b = D_b / 2.0
    # Build the same S basis for both (necessary for clean idempotency).
    # Just re-build D_b in the basis from the seed=1 S:
    rng = np.random.default_rng(2)
    A = rng.standard_normal((8, 8))
    F2 = (A + A.T) / 2.0
    L = np.linalg.cholesky(np.asarray(S))
    Linv = np.linalg.inv(L)
    F_orth = Linv @ F2 @ Linv.T
    _, U = np.linalg.eigh(F_orth)
    C = Linv.T @ U
    D_b = jnp.array(C[:, :2] @ C[:, :2].T)

    D = jnp.stack([D_a, D_b], axis=0)
    out = compute_dm_features(D, S)
    err = float(out["idempotency_error"])
    assert abs(err) < 1e-5, (
        f"clean UKS DM should give idempotency_error ≈ 0; got {err}."
    )


def test_idempotency_error_nonzero_for_correlated_dm():
    """Sanity: a non-idempotent (correlated) DM gives idempotency_error > 0."""
    D, S = _build_clean_rks_dm(nao=8, nocc=3, seed=0)
    rng = np.random.default_rng(42)
    perturb = rng.standard_normal((8, 8))
    perturb = 0.05 * (perturb + perturb.T)
    D_corr = D + jnp.array(perturb)
    out = compute_dm_features(D_corr, S)
    err = float(out["idempotency_error"])
    assert err > 1e-3, (
        f"perturbed (non-idempotent) DM should give a clearly nonzero "
        f"idempotency_error; got {err}"
    )


def test_dm_features_array_shape_and_finite():
    """compute_dm_features_array returns a length-2 finite jnp array.

    Width went 3 -> 2 on 2026-08-06 with the removal of dm_entropy, which had
    no usable gradient at any converged density. This count sets the network
    input width for any dm_statistics architecture, so it is pinned here.
    """
    D, S = _build_clean_rks_dm(nao=6, nocc=2, seed=0)
    arr = compute_dm_features_array(D, S)
    assert arr.shape == (2,), arr.shape
    assert jnp.all(jnp.isfinite(arr)), arr


def test_dm_features_has_no_entropy_key():
    """dm_entropy is gone from the dict as well as the array.

    Guards the halfway state where the array shrinks but the dict still
    computes an eigendecomposition nothing consumes.
    """
    D, S = _build_clean_rks_dm(nao=6, nocc=2, seed=0)
    out = compute_dm_features(D, S)
    assert "dm_entropy" not in out, sorted(out)
    assert sorted(out) == ["idempotency_error", "off_diag_norm", "trace"]


def test_n_elec_trace_matches_density_matrix():
    """trace = Tr(DS) should equal the electron count (here 2 * nocc for RKS)."""
    nocc = 3
    D, S = _build_clean_rks_dm(nao=8, nocc=nocc, seed=0)
    out = compute_dm_features(D, S)
    n_elec = float(out["trace"])
    assert abs(n_elec - 2 * nocc) < 1e-6, n_elec


# ---------------------------------------------------------------------------
# natural-orbital occupations must be eig(D @ S) (not eig(S^{-1} D)).
# compute_dm_natural_occupations is retained after the dm_entropy removal:
# it is a correct, tested primitive, and any future replacement descriptor
# built on the occupation spectrum will need it.
# ---------------------------------------------------------------------------

from xcquinox.features import compute_dm_natural_occupations


def test_natural_occupations_equal_eig_DS():
    """the natural-orbital occupations must equal sorted(eig(D @ S)) for a
    NON-identity overlap S.

    Pre-fix code used the Lowdin transform S^{-1/2} D S^{-1/2}, whose
    eigenvalues are eig(S^{-1} D), NOT the natural occupations. The
    correct symmetric transform is S^{1/2} D S^{1/2}, whose spectrum
    equals eig(D S).
    """
    D, S = _build_clean_rks_dm(nao=8, nocc=3, seed=0)
    occ = np.asarray(compute_dm_natural_occupations(D, S))
    # Reference natural occupations: eigenvalues of D @ S.
    ref = np.linalg.eigvals(np.asarray(D) @ np.asarray(S))
    ref = np.sort(np.real(ref))
    occ_sorted = np.sort(occ)
    np.testing.assert_allclose(occ_sorted, ref, atol=1e-6)
def test_cusp_descriptor_log_transform_off_skips_log():
    """``compute_cusp_descriptor(log_transform=False)`` feeds the raw
    weighted-Z value through tanh(·/5); ``=True`` applies the Dick XCDiff
    log-compress first. The two outputs MUST differ except in the trivial
    Z_sum << 1 limit (where both reduce to ~ 0)."""
    from xcquinox.features import compute_cusp_descriptor
    # 2 grid points near a single Z=8 (oxygen) nucleus, in atomic units.
    grid = jnp.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]])
    nuc = jnp.array([[0.0, 0.0, 0.0]])
    Z = jnp.array([8.0])
    raw = compute_cusp_descriptor(grid, nuc, Z, log_transform=False)
    logd = compute_cusp_descriptor(grid, nuc, Z, log_transform=True)
    # Column 0 (cusp_factor) is identical, log_transform only gates col 1.
    assert jnp.allclose(raw[:, 0], logd[:, 0]), (raw[:, 0], logd[:, 0])
    # Column 1 differs because of the log-compress.
    assert not jnp.allclose(raw[:, 1], logd[:, 1], atol=1e-3), (
        raw[:, 1], logd[:, 1]
    )
# ---------------------------------------------------------------------------
# Fix: precompute_fixed_density_data passes spin-resolved DM
# ---------------------------------------------------------------------------

import pytest


@pytest.mark.slow
def test_precompute_passes_spin_resolved_dm_for_uks():
    """precompute_fixed_density_data must route the 3-D spin-resolved
    DM (not the spin-summed total) into compute_dm_features for UKS
    molecules so the per-spin idempotency-projector branch fires
    (Pople-Nesbet 1954: D_sigma S D_sigma = D_sigma).

    Fix: pre-fix `precompute_fixed_density_data` summed
    alpha+beta into `dm_pbe_tot` and forced UKS molecules through the
    RKS branch, producing a non-zero physically-meaningless
    idempotency_error on every open-shell molecule.
    """
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.descriptors import DMStatisticsDescriptor

    # Open-shell radical (CH triplet, spin=2 on a small basis is fast).
    mol = MoleculeSpec.from_dict(
        name="CH", atom="C 0 0 0; H 0 0 1.12",
        basis="sto-3g", charge=0, spin=1,
        atom_composition={"C": 1, "H": 1},
    )
    descriptors = (DMStatisticsDescriptor(),)
    data = precompute_fixed_density_data(
        mol, descriptors=descriptors,
        required_keys=("dm_features",),
    )
    dm_features = data.get("dm_features")
    assert dm_features is not None
    # idempotency_error column should be ~0 for a clean UKS DM (the
    # SCF-converged PBE DM for CH triplet IS a single-determinant KS
    # reference for which D_sigma S D_sigma = D_sigma per spin holds
    # exactly). Pre-fix code gave |err| ~ 1; post-fix gives < 1e-6.
    import jax.numpy as jnp
    idem_err = float(jnp.asarray(dm_features)[0, 0])
    assert abs(idem_err) < 1e-5, (
        f"clean UKS CH triplet should give idempotency_error ~ 0; "
        f"got {idem_err}. Pre-fix code routes UKS through the RKS "
        f"branch and gives |err| ~ 1."
    )


# spin-polarized PW92 correlation baseline, VERIFIED vs libxc.
# (The polarized branch of utils.pw92c was dead code, nspin hardcoded to 1, so
# pw92c_polarized_scalar is the first exercised polarized PW92; verify it
# against an independent reference.)
def test_pw92c_polarized_reduces_to_unpolarized_at_zeta0():
    from xcquinox.utils import pw92c_polarized_scalar, pw92c_unpolarized_scalar
    for rho in [1e-3, 1e-2, 0.1, 1.0, 5.0, 50.0]:
        pol = float(pw92c_polarized_scalar(rho / 2.0, rho / 2.0))
        unp = float(pw92c_unpolarized_scalar(rho))
        assert abs(pol - unp) < 1e-12, (rho, pol, unp)


def test_pw92c_polarized_matches_libxc_lda_c_pw():
    import numpy as np
    from pyscf.dft import libxc
    from xcquinox.utils import pw92c_polarized_scalar
    cases = [(0.5, 0.5), (0.8, 0.2), (1.0, 0.0), (0.3, 0.1),
             (2.0, 1.0), (0.05, 0.02), (10.0, 3.0)]
    for ra, rb in cases:
        ours = float(pw92c_polarized_scalar(ra, rb))
        exc, _vxc, _fxc, _kxc = libxc.eval_xc(
            "LDA_C_PW", (np.array([ra]), np.array([rb])), spin=1)
        assert abs(ours - float(exc[0])) < 1e-8, (ra, rb, ours, float(exc[0]))


def test_pw92c_polarized_finite_value_and_grad_at_extremes():
    """P2-03 (review LOW fix): forward value AND reverse-mode gradient must be
    finite at vanishing/zero density and at full polarization (zeta=+-1), so the
    polarized baseline is safe to differentiate through the SCF."""
    import jax
    from xcquinox.utils import pw92c_polarized_scalar
    f = lambda a, b: pw92c_polarized_scalar(a, b)
    for a, b in [(0.0, 0.0), (1e-300, 1e-300), (1e-200, 0.0),
                 (0.5, 0.0), (1.0, 1.0), (3.0, 0.0)]:
        val = float(f(a, b))
        ga = float(jax.grad(f, 0)(float(a), float(b)))
        gb = float(jax.grad(f, 1)(float(a), float(b)))
        assert np.isfinite(val) and np.isfinite(ga) and np.isfinite(gb), (a, b, val, ga, gb)


def test_pw92c_polarized_second_derivative_finite_at_full_polarization():
    """The second derivative of eps_c w.r.t. spin density must stay finite at
    full spin polarization (one spin density 0, zeta=+-1). The FULL SCF
    differentiates v_xc (itself a first derivative of E_xc) a second time, so an
    inf/NaN second derivative here breaks the training gradient of every
    fully-spin-polarized species (free atoms H, Li, ...). PW92's spin
    interpolation f(zeta) ~ (1+-zeta)**(4/3) has d2/dzeta2 ~ (1-+zeta)**(-2/3)
    -> inf at |zeta|=1 unless the interpolation base is floored. The first-order
    gradient (checked above) is finite there and hid this. Perdew & Wang
    PRB 45, 13244 (1992), eqs (8)-(9)."""
    import jax
    from xcquinox.utils import pw92c_polarized_scalar

    def d2_wrt_beta(ra, rb):
        inner = lambda x: jnp.sum(pw92c_polarized_scalar(jnp.array([ra]), x))
        outer = lambda b: jnp.sum(jax.grad(inner)(b))
        return float(jax.grad(outer)(jnp.array([rb])).sum())

    def d2_wrt_alpha(ra, rb):
        inner = lambda x: jnp.sum(pw92c_polarized_scalar(x, jnp.array([rb])))
        outer = lambda a: jnp.sum(jax.grad(inner)(a))
        return float(jax.grad(outer)(jnp.array([ra])).sum())

    # zeta -> +1 (rho_beta -> 0): the free-atom majority-spin boundary.
    for ra, rb in [(0.5, 0.0), (3.0, 0.0), (0.05, 0.0)]:
        assert np.isfinite(d2_wrt_beta(ra, rb)), ("d2/drho_beta2", ra, rb)
    # zeta -> -1 (rho_alpha -> 0): the symmetric boundary.
    for ra, rb in [(0.0, 0.5), (0.0, 3.0)]:
        assert np.isfinite(d2_wrt_alpha(ra, rb)), ("d2/drho_alpha2", ra, rb)
