"""Sizing and mechanism of the eigh symmetry-breaking diagonal.

``jnp.linalg.eigh``'s reverse-mode rule carries 1/(lambda_i - lambda_j)
factors, so exactly degenerate pairs (linear-symmetry pi MOs, atomic p
shells) yield non-finite or round-off-dominated gradients unless the
transformed Fock matrix is perturbed. ``_sym_break_diag`` supplies that
perturbation with magnitude ``SYM_BREAK_SHIFT``, bounded on two sides:

- LOWER: the backward pass amplifies matrix-level round-off eps by
  eps/gap^2 in the eigenvector channel. Graph-order changes (kernel
  fusion, the shape-padding pass) inject eps ~ machine epsilon on O(1)
  Fock elements, so gap^2 must exceed eps by a safety margin or the
  backward degrades to garbage and non-finite values. At 1e-8 the ratio
  eps/gap^2 is O(1) -- the regime in which production training at
  6-311++G(3df,2pd) produced all-leaf NaN gradients on the padded graph
  (group bh76:OH+N2_to_H+N2O) while the unpadded evaluation of the same
  state stayed finite. At 1e-6 the ratio is ~2e-4, and the same padded
  replay completes its epoch with the step loss unchanged to seven digits.
- UPPER: the shift must stay an order below the smallest deliberate
  splitting it may not disturb (the 3e-5 orientation-lock splitting of
  the OH 2Pi pair), and by Weyl's inequality it displaces every
  eigenvalue by at most its own magnitude (<= 1e-6 Ha forward bias).

The admissible window is roughly [7e-7, 3e-6]; 1e-6 sits inside it and
was validated by the on-cluster A/B replay pair recorded in
xcquinox/alec/HISTORY.md (Phase 33).
"""
import jax
import jax.numpy as jnp
import numpy as np

from xcquinox.alec.solver import SYM_BREAK_SHIFT, _sym_break_diag

_EPS64 = float(jnp.finfo(jnp.float64).eps)
# Orientation-lock strength (orientation_lock.py; production configs pin
# inputs.orientation_lock_strength = 3e-5): the smallest deliberate
# splitting the quasi-random diagonal must not rival.
_LOCK_SPLIT = 3e-5


def _degenerate_fock(nao: int = 4) -> jnp.ndarray:
    """Non-diagonal symmetric matrix with an exactly degenerate lowest pair."""
    rng = np.random.RandomState(7)
    q, _ = np.linalg.qr(rng.standard_normal((nao, nao)))
    return jnp.asarray(q @ np.diag([1.0, 1.0, 2.0, 3.0]) @ q.T)


def _grad_of_degenerate_eigvec_element(shift_diag):
    """Gradient of a degenerate-pair eigenvector element, optionally shifted."""
    F = _degenerate_fock()

    def f(m):
        if shift_diag is not None:
            m = m + jnp.diag(shift_diag)
        _, v = jnp.linalg.eigh(m)
        return v[0, 0] ** 2

    return jax.grad(f)(F)


def test_shift_value_is_pinned():
    """The production value; the window bounds below justify it."""
    assert SYM_BREAK_SHIFT == 1e-6


def test_backward_roundoff_amplification_is_bounded():
    """eps/gap^2 must sit >= 3 orders below unity.

    Round-off eps on O(1) matrix elements reaches the eigenvector backward
    amplified by 1/gap^2; if the ratio approaches unity, graph-order
    round-off (fusion, padding) becomes O(1) gradient corruption. The
    former value 1e-8 gives eps/gap^2 ~ 2.2 and fails this bound.
    """
    assert _EPS64 / SYM_BREAK_SHIFT**2 <= 1e-3


def test_shift_stays_below_orientation_lock():
    """Degenerate-pair selection must stay with the physical lock splitting."""
    assert SYM_BREAK_SHIFT <= _LOCK_SPLIT / 10.0


def test_shift_forward_bias_bounded_by_weyl():
    """Eigenvalue displacement of the perturbation is <= its magnitude."""
    F = _degenerate_fock()
    d = _sym_break_diag(F.shape[0], F.dtype)
    w0 = jnp.linalg.eigvalsh(F)
    w1 = jnp.linalg.eigvalsh(F + jnp.diag(d))
    assert float(jnp.max(jnp.abs(w1 - w0))) <= float(SYM_BREAK_SHIFT) * (1 + 1e-12)


def test_sym_break_diag_splits_exact_degeneracy():
    """The graded diagonal opens the degenerate gap to O(SYM_BREAK_SHIFT)."""
    F = _degenerate_fock()
    d = _sym_break_diag(F.shape[0], F.dtype)
    w = jnp.linalg.eigvalsh(F + jnp.diag(d))
    gap = float(w[1] - w[0])
    # |sin(0) - sin(phi)| ~ 0.999 for the lowest pair; first-order
    # perturbation projects it through the degenerate subspace, so a
    # conservative floor of 0.1 x shift is asserted.
    assert gap >= 0.1 * float(SYM_BREAK_SHIFT)


def test_eigh_gradient_degrades_at_exact_degeneracy():
    """Documents the pathology: unshifted degenerate eigh has no usable grad.

    Depending on how the LAPACK eigenvalues round, the 1/(dlambda) factors
    are inf/NaN (bit-equal pair) or ~1e16 round-off garbage (ULP-split
    pair); both are unusable and both count as the failure mode.
    """
    g = _grad_of_degenerate_eigvec_element(None)
    finite = bool(jnp.all(jnp.isfinite(g)))
    assert (not finite) or float(jnp.max(jnp.abs(g))) > 1e10


def test_sym_break_diag_restores_bounded_gradient():
    """With the shift the same gradient is finite and O(1/SYM_BREAK_SHIFT)."""
    F = _degenerate_fock()
    d = _sym_break_diag(F.shape[0], F.dtype)
    g = _grad_of_degenerate_eigvec_element(d)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) <= 1e8
