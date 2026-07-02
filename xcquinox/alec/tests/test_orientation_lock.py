"""Tests for xcquinox.alec.orientation_lock: the degeneracy-lifting h_core bias.

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
import pytest
import scipy.linalg as sla
from pyscf import gto, dft

from xcquinox.alec.orientation_lock import (
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


def test_bias_shape_is_nao_by_nao():
    mol = _oh()
    b = orientation_lock_bias(mol, 1e-3)
    assert b.shape == (mol.nao, mol.nao)


def test_bias_is_symmetric():
    mol = _oh()
    b = orientation_lock_bias(mol, 1e-3)
    assert np.allclose(b, b.T, atol=0, rtol=0)


def test_bias_is_deterministic():
    mol = _oh()
    b1 = orientation_lock_bias(mol, 1e-3)
    b2 = orientation_lock_bias(mol, 1e-3)
    # bit-for-bit identical: the whole point is cross-call reproducibility
    assert np.array_equal(b1, b2)


def test_bias_scales_linearly_with_strength():
    mol = _oh()
    b1 = orientation_lock_bias(mol, 1e-3)
    b2 = orientation_lock_bias(mol, 2e-3)
    assert np.allclose(b2, 2.0 * b1, rtol=1e-12, atol=0)


def test_bias_identical_for_independently_built_identical_mols():
    """Ref-gen and precompute build separate Mole objects from the same
    geometry+basis; the operator MUST be identical so they lock the same
    degenerate component."""
    m1 = _oh("z")
    m2 = _oh("z")
    b1 = orientation_lock_bias(m1, 1e-3)
    b2 = orientation_lock_bias(m2, 1e-3)
    assert np.array_equal(b1, b2)


def test_does_not_mutate_mol_common_origin():
    """The helper must not leave a common-origin set on mol (would silently
    change later dipole/quadrupole integrals in the same pipeline)."""
    mol = _oh()
    r_before = mol.intor("int1e_r").copy()
    orientation_lock_bias(mol, 1e-3)
    r_after = mol.intor("int1e_r").copy()
    assert np.array_equal(r_before, r_after)


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


@pytest.mark.parametrize("axis", ["x", "y", "z", "skew"])
def test_lifts_pi_degeneracy_for_any_orientation(axis):
    """OH's h_core pi degeneracy must be lifted regardless of bond orientation
    (a purely x^2-z^2 operator would vanish on a skew axis; the generic
    traceless quadrupole must not)."""
    mol = _oh(axis)
    h, S = _hcore_S(mol)
    e_unbiased = sla.eigh(h, S, eigvals_only=True)
    e_biased = sla.eigh(h + orientation_lock_bias(mol, 1e-3), S, eigvals_only=True)
    assert _n_near_degenerate(e_unbiased, 1e-9) > 0
    assert _n_near_degenerate(e_biased, 1e-8) == 0


# ---------------------------------------------------------------------------
# calibration: negligible energy impact on a closed-shell system
# ---------------------------------------------------------------------------
def test_default_strength_energy_shift_negligible_closed_shell():
    """At the demo default strength, a closed-shell PBE energy shifts by
    < 0.1 kcal/mol (traceless operator -> ~zero first-order shift)."""
    w = gto.M(
        atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469",
        basis="def2-svp", verbose=0,
    )
    base_h = w.intor("int1e_kin") + w.intor("int1e_nuc")
    bias = orientation_lock_bias(w, DEFAULT_STRENGTH)

    mf = dft.RKS(w); mf.xc = "pbe"; e_ref = mf.kernel()
    mf2 = dft.RKS(w); mf2.xc = "pbe"
    mf2.get_hcore = lambda *a, **k: base_h + bias
    e_lock = mf2.kernel()

    dE_kcal = abs(e_lock - e_ref) * 627.5094740631
    assert dE_kcal < 0.1, f"closed-shell energy shift {dE_kcal:.4f} kcal/mol too large"


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


def test_locked_oh_scf_converges_and_conserves_electrons():
    """Sanity: the bias does not break OH's UKS SCF."""
    oh = _oh("z")
    base_h = oh.intor("int1e_kin") + oh.intor("int1e_nuc")
    bias = orientation_lock_bias(oh, DEFAULT_STRENGTH)
    mf = dft.UKS(oh); mf.xc = "pbe"
    mf.get_hcore = lambda *a, **k: base_h + bias
    mf.kernel()
    assert mf.converged
    assert oh.nelectron == 9


# ---------------------------------------------------------------------------
# plumbing: SolverConfig field
# ---------------------------------------------------------------------------
def test_solver_config_default_strength_is_zero():
    from xcquinox.alec.solver import SolverConfig
    assert SolverConfig().orientation_lock_strength == 0.0


def test_solver_config_describe_includes_strength():
    from xcquinox.alec.solver import SolverConfig
    d = SolverConfig(orientation_lock_strength=3e-5).describe()
    assert d["orientation_lock_strength"] == 3e-5


def test_solver_config_negative_strength_raises():
    from xcquinox.alec.solver import SolverConfig
    with pytest.raises(ValueError):
        SolverConfig(orientation_lock_strength=-1e-4)


# ---------------------------------------------------------------------------
# plumbing: precompute injects the bias, cache key distinguishes strength
# ---------------------------------------------------------------------------
def _h2o_spec():
    from xcquinox.alec.config import MoleculeSpec
    return MoleculeSpec(
        name="H2O_ol",
        atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469",
        basis="def2-svp",
    )


def test_precompute_injects_bias_into_hcore():
    from xcquinox.alec.data import (
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
    from xcquinox.alec.data import (
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


def test_precompute_stashes_bias_matrix_for_pyscfad():
    from xcquinox.alec.data import (
        precompute_fixed_density_data, clear_precompute_cache,
    )
    spec = _h2o_spec()
    clear_precompute_cache()
    d1 = precompute_fixed_density_data(spec, orientation_lock_strength=DEFAULT_STRENGTH)
    stashed = d1["mol_metadata"].get("orientation_lock_bias")
    mol = gto.M(atom=spec.atom, basis=spec.basis, verbose=0)
    assert stashed is not None
    assert np.allclose(np.asarray(stashed), orientation_lock_bias(mol, DEFAULT_STRENGTH))


# ---------------------------------------------------------------------------
# plumbing: reference self-heal (pure-function level; full CCSD path is E2E)
# ---------------------------------------------------------------------------
def test_intermediate_cache_name_tags_strength():
    from xcquinox.alec.external_refs import _intermediate_cache_name
    off = _intermediate_cache_name("HO", grid_level=2, basis="def2-svp",
                                   density_fit=False, kind="scf")
    on = _intermediate_cache_name("HO", grid_level=2, basis="def2-svp",
                                  density_fit=False, kind="scf",
                                  orientation_lock_strength=3e-5)
    # strength=0 (default) must be byte-identical to the pre-lock filename
    off2 = _intermediate_cache_name("HO", grid_level=2, basis="def2-svp",
                                    density_fit=False, kind="scf",
                                    orientation_lock_strength=0.0)
    assert off == off2
    assert on != off


def test_benchmark_npz_complete_checks_strength(tmp_path):
    from xcquinox.alec.benchmark_refs import (
        _benchmark_npz_is_complete, _atomic_savez,
    )
    p = tmp_path / "HO.npz"

    def _write(strength_field):
        arrs = dict(
            rho_ref_grid=np.ones(3), rho_pbe_grid=np.ones(3),
            grid_weights=np.ones(3), ref_density_method=np.array("ccsd"),
            grid_level_used=np.array(2), basis_used=np.array("def2-svp"),
        )
        if strength_field is not None:
            arrs["orientation_lock_strength"] = np.array(float(strength_field))
        _atomic_savez(p, **arrs)

    # legacy npz (no strength field) matches only a strength-0 request
    _write(None)
    assert _benchmark_npz_is_complete(p, basis="def2-svp", grid_level=2,
                                      orientation_lock_strength=0.0)
    assert not _benchmark_npz_is_complete(p, basis="def2-svp", grid_level=2,
                                          orientation_lock_strength=3e-5)
    # a locked npz matches its own strength, not a different one
    _write(3e-5)
    assert _benchmark_npz_is_complete(p, basis="def2-svp", grid_level=2,
                                      orientation_lock_strength=3e-5)
    assert not _benchmark_npz_is_complete(p, basis="def2-svp", grid_level=2,
                                          orientation_lock_strength=0.0)


# ---------------------------------------------------------------------------
# plumbing: the pyscfad backend adds the bias to its own get_hcore
# ---------------------------------------------------------------------------
def test_pyscfad_backend_adds_bias_to_get_hcore():
    pytest.importorskip("pyscfad")
    # conftest enables jax_enable_x64, so get_hcore's float64 integrals compare cleanly.
    from xcquinox.alec.data import (
        precompute_fixed_density_data, clear_precompute_cache,
    )
    from xcquinox.alec import solver_pyscfad as sp

    spec = _h2o_spec()
    clear_precompute_cache()
    md0 = precompute_fixed_density_data(spec, orientation_lock_strength=0.0)
    clear_precompute_cache()
    md1 = precompute_fixed_density_data(spec, orientation_lock_strength=DEFAULT_STRENGTH)

    mf0 = sp._build_pyscfad_mf(sp._rebuild_mol_from_mol_data(md0), md0)
    mf1 = sp._build_pyscfad_mf(sp._rebuild_mol_from_mol_data(md1), md1)
    delta = np.asarray(mf1.get_hcore()) - np.asarray(mf0.get_hcore())
    bias = np.asarray(md1["mol_metadata"]["orientation_lock_bias"])
    assert np.allclose(delta, bias, atol=1e-8)
