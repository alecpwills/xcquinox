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
    """compute_dm_features_array returns a length-3 finite jnp array."""
    D, S = _build_clean_rks_dm(nao=6, nocc=2, seed=0)
    arr = compute_dm_features_array(D, S)
    assert arr.shape == (3,), arr.shape
    assert jnp.all(jnp.isfinite(arr)), arr


def test_n_elec_trace_matches_density_matrix():
    """trace = Tr(DS) should equal the electron count (here 2 * nocc for RKS)."""
    nocc = 3
    D, S = _build_clean_rks_dm(nao=8, nocc=nocc, seed=0)
    out = compute_dm_features(D, S)
    n_elec = float(out["trace"])
    assert abs(n_elec - 2 * nocc) < 1e-6, n_elec
