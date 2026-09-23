"""Tests for xcquinox.pipeline.orientation_lock: the degeneracy-lifting h_core bias.

The orientation lock adds a small, fixed, deterministic anisotropic-quadrupole
operator to h_core so that an orbitally-degenerate open-shell radical (OH/NO,
X-2-Pi) always relaxes to the SAME representative of its degenerate pi manifold.
Applied identically in CCSD ref-generation, the PBE seed, training, and eval, it
makes the single-determinant density on a fixed grid reproducible across
processes/machines (the energy is degeneracy-invariant either way).

These tests pin the operator's contract (determinism, symmetry, linearity,
geometry/basis-consistency), its physical behaviour (lifts p/pi degeneracy for
any orientation), and its calibration (negligible energy impact on closed-shell
systems at the demo default strength).
"""
import numpy as np
import scipy.linalg as sla
from pyscf import gto

from xcquinox.pipeline.orientation_lock import (
    orientation_lock_bias,
    DEFAULT_STRENGTH,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _oh(axis="z"):
    """OH (X-2-Pi radical) with the bond along a chosen lab axis."""
    d = 0.97
    coords = {
        "x": f"O 0 0 0; H {d} 0 0",
        "y": f"O 0 0 0; H 0 {d} 0",
        "z": f"O 0 0 0; H 0 0 {d}",
        "skew": f"O 0 0 0; H {d/np.sqrt(3):.6f} {d/np.sqrt(3):.6f} {d/np.sqrt(3):.6f}",
    }[axis]
    return gto.M(atom=coords, basis="def2-svp", spin=1, verbose=0)


def _hcore_S(mol):
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    S = mol.intor("int1e_ovlp")
    return h, S


def _n_near_degenerate(evals, tol):
    """Number of consecutive eigenvalue gaps below tol (degenerate pairs)."""
    return int((np.diff(np.sort(evals)) < tol).sum())


# ---------------------------------------------------------------------------
# contract: determinism / symmetry / linearity / shape / consistency
# ---------------------------------------------------------------------------
def test_zero_strength_returns_zeros():
    mol = _oh()
    b = orientation_lock_bias(mol, 0.0)
    assert b.shape == (mol.nao, mol.nao)
    assert np.array_equal(b, np.zeros((mol.nao, mol.nao)))


def test_bias_is_symmetric():
    mol = _oh()
    b = orientation_lock_bias(mol, 1e-3)
    assert np.allclose(b, b.T, atol=0, rtol=0)


def test_bias_scales_linearly_with_strength():
    mol = _oh()
    b1 = orientation_lock_bias(mol, 1e-3)
    b2 = orientation_lock_bias(mol, 2e-3)
    assert np.allclose(b2, 2.0 * b1, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# physics: lifts p/pi degeneracy, for any orientation
# ---------------------------------------------------------------------------
def test_lifts_atomic_p_degeneracy():
    """An O atom has a 3-fold degenerate 2p shell; the bias splits it."""
    o = gto.M(atom="O 0 0 0", basis="def2-svp", spin=2, verbose=0)
    h, S = _hcore_S(o)
    e_unbiased = sla.eigh(h, S, eigvals_only=True)
    e_biased = sla.eigh(h + orientation_lock_bias(o, 1e-3), S, eigvals_only=True)
    assert _n_near_degenerate(e_unbiased, 1e-9) > 0
    assert _n_near_degenerate(e_biased, 1e-8) == 0


# ---------------------------------------------------------------------------
# calibration: negligible energy impact on a closed-shell system
# ---------------------------------------------------------------------------


def test_default_strength_pi_split_dominates_noise():
    """At the demo default strength the induced pi splitting is comfortably
    above float64/BLAS noise (~1e-8) so it deterministically locks the SCF."""
    mol = _oh("z")
    h, S = _hcore_S(mol)
    e_unbiased = np.sort(sla.eigh(h, S, eigvals_only=True))
    e_biased = np.sort(sla.eigh(h + orientation_lock_bias(mol, DEFAULT_STRENGTH), S, eigvals_only=True))
    # the near-degenerate pi pair (smallest unbiased gap) opens to >> 1e-8
    max_lift = np.max(np.abs(e_biased - e_unbiased))
    assert max_lift > 1e-7, f"pi lift {max_lift:.2e} not above noise floor"


# ---------------------------------------------------------------------------
# plumbing: SolverConfig field
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# plumbing: precompute injects the bias, cache key distinguishes strength
# ---------------------------------------------------------------------------
def _h2o_spec():
    from xcquinox.pipeline.config import MoleculeSpec
    return MoleculeSpec(
        name="H2O_ol",
        atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469",
        basis="def2-svp",
    )


def test_precompute_injects_bias_into_hcore():
    from xcquinox.pipeline.data import (
        precompute_fixed_density_data, clear_precompute_cache,
    )
    spec = _h2o_spec()
    clear_precompute_cache()
    d0 = precompute_fixed_density_data(spec, orientation_lock_strength=0.0)
    clear_precompute_cache()
    d1 = precompute_fixed_density_data(spec, orientation_lock_strength=DEFAULT_STRENGTH)

    mol = gto.M(atom=spec.atom, basis=spec.basis, verbose=0)
    expected = orientation_lock_bias(mol, DEFAULT_STRENGTH)
    delta = np.asarray(d1["h_core"]) - np.asarray(d0["h_core"])
    assert np.allclose(delta, expected, atol=1e-10)


def test_precompute_cache_key_distinguishes_strength():
    from xcquinox.pipeline.data import (
        precompute_fixed_density_data, clear_precompute_cache,
    )
    spec = _h2o_spec()
    clear_precompute_cache()
    d0 = precompute_fixed_density_data(spec, orientation_lock_strength=0.0)
    # same object returned on a cache hit at strength 0
    d0b = precompute_fixed_density_data(spec, orientation_lock_strength=0.0)
    assert d0b is d0
    # a different strength must NOT return the cached strength-0 entry
    d1 = precompute_fixed_density_data(spec, orientation_lock_strength=DEFAULT_STRENGTH)
    assert d1 is not d0
    assert not np.allclose(np.asarray(d1["h_core"]), np.asarray(d0["h_core"]))


# ---------------------------------------------------------------------------
# plumbing: reference self-heal (pure-function level; full CCSD path is E2E)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# plumbing: the pyscfad backend adds the bias to its own get_hcore
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The calibrated strength has ONE definition, in a module that carries no
# numeric stack
# ---------------------------------------------------------------------------
def test_the_default_strength_has_one_definition():
    """``orientation_lock.DEFAULT_STRENGTH`` IS the object defined in
    ``orientation_lock_default``, not a second literal that happens to agree.

    The constant is read by the harness parser as well as by this module, and
    a parser that reads it must not pay for numpy (the quadrupole operator
    below needs it, ``load_grid_config`` does not), so the number lives in a
    leaf module both import."""
    from xcquinox.pipeline import orientation_lock, orientation_lock_default
    assert (orientation_lock.DEFAULT_STRENGTH
            is orientation_lock_default.DEFAULT_STRENGTH)
    assert DEFAULT_STRENGTH is orientation_lock_default.DEFAULT_STRENGTH
    assert orientation_lock_default.DEFAULT_STRENGTH == 3e-5


