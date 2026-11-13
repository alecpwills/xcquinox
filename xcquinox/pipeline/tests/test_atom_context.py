"""Contracts of the frozen atom-context environment descriptor.

``compute_atom_context_descriptor`` (xcquinox.features) mixes frozen
per-element embeddings under distance-decay attention inside a cutoff and
folds the environment vectors onto the grid; ``AtomContextDescriptor``
registers the columns and ``deep_atomctx_3x16`` consumes them. The tests
below pin the bounds, invariances and cutoff exactness of the kernel, the
agreement of the two feature-production routes (data.py precompute and the
pyscfad per-point recompute), and the registry/cache interface contracts
the package relies on.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from xcquinox.features import compute_atom_context_descriptor
from xcquinox.pipeline.config import ARCHITECTURES, ArchitectureConfig
from xcquinox.pipeline.data import _atom_context_features_on_demand, _precompute_cache_key
from xcquinox.pipeline.descriptors import AtomContextDescriptor, make_descriptor
from xcquinox.pipeline.solver_pyscfad import _reassemble_features_on_grid

# An H2O-like geometry in Bohr (O-H ~ 1.81 a0) and charges.
_H2O_COORDS = jnp.array([[0.0, 0.0, 0.0],
                         [0.0, 0.0, 1.81],
                         [0.0, 1.65, -0.50]])
_H2O_Z = jnp.array([8, 1, 1])


def _random_system(seed: int, n_atoms: int, n_grid: int):
    """Deterministic synthetic (coords, charges, grid); the grid spread exceeds
    rc so some rows sit beyond rc of every atom."""
    rng = jax.random.PRNGKey(seed)
    coords = jax.random.uniform(rng, (n_atoms, 3), minval=-6.0, maxval=6.0)
    charges = jax.random.randint(jax.random.fold_in(rng, 1), (n_atoms,),
                                 minval=1, maxval=18)
    grid = jax.random.uniform(jax.random.fold_in(rng, 2), (n_grid, 3),
                              minval=-13.0, maxval=13.0)
    return coords, charges, grid


def test_boundedness_all_columns_within_unit_interval():
    """Every column of the (n_grid, 4) output lies in (-1, 1) -- a convex
    combination of tanh-bounded embeddings through both the attention
    normalization and the fold-in normalization -- and is finite, including a
    grid point placed exactly ON a nucleus (r_gA = 0, floored by the kernel)
    and one beyond rc of every atom (empty fold-in sum: exact zero, never a
    0/0 NaN)."""
    coords, charges, grid = _random_system(0, 5, 200)
    # a row on a nucleus (r = 0 -> the u_A = Z/r division) and one far row
    grid = jnp.concatenate([grid, coords[:1] + 0.0,
                            jnp.array([[40.0, -40.0, 40.0]])], axis=0)
    feat = compute_atom_context_descriptor(grid, coords, charges)
    assert feat.shape == (grid.shape[0], 4)
    assert bool(jnp.all(jnp.isfinite(feat)))
    assert bool(jnp.all(jnp.abs(feat) <= 1.0))


def test_permutation_invariance_atoms_reordered():
    """Permuting atoms (coords and charges together) leaves every column
    identical: embeddings are indexed by Z, not atom position, and both sums
    over atoms are symmetric."""
    coords, charges, grid = _random_system(1, 6, 150)
    perm = jnp.array([3, 0, 5, 1, 4, 2])
    f_ref = compute_atom_context_descriptor(grid, coords, charges)
    f_perm = compute_atom_context_descriptor(grid, coords[perm], charges[perm])
    np.testing.assert_allclose(np.asarray(f_perm), np.asarray(f_ref),
                               atol=1e-12, rtol=0.0)


def test_translation_invariance_rigid_shift():
    """A rigid translation of atoms and grid together leaves every column
    identical: the kernel reads distances only."""
    coords, charges, grid = _random_system(2, 4, 150)
    shift = jnp.array([1.7, -2.3, 0.9])
    f_ref = compute_atom_context_descriptor(grid, coords, charges)
    f_shift = compute_atom_context_descriptor(grid + shift, coords + shift,
                                              charges)
    np.testing.assert_allclose(np.asarray(f_shift), np.asarray(f_ref),
                               atol=1e-10, rtol=0.0)


def test_rotation_invariance_rigid_rotation():
    """A rigid rotation of atoms and grid together leaves every column
    identical: rotation-invariant by distance-only construction."""
    coords, charges, grid = _random_system(3, 4, 150)
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    angle = 0.7
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    rot = jnp.asarray(np.eye(3) + np.sin(angle) * k
                      + (1.0 - np.cos(angle)) * (k @ k))
    f_ref = compute_atom_context_descriptor(grid, coords, charges)
    f_rot = compute_atom_context_descriptor(grid @ rot.T, coords @ rot.T,
                                            charges)
    np.testing.assert_allclose(np.asarray(f_rot), np.asarray(f_ref),
                               atol=1e-9, rtol=0.0)


def test_cutoff_exactness_and_size_consistency():
    """A fragment displaced beyond 2*rc changes the columns only at the matmul
    reduction's rounding (the far atom's attention and fold-in weights are the
    written zeros of jnp.where); a grid point beyond rc of every atom carries
    exactly 0.0 columns with no 0/0 NaN; and an isolated atom's rows all equal
    its own embedding, the size-consistency limit."""
    grid = jax.random.uniform(jax.random.PRNGKey(4), (120, 3),
                              minval=-4.0, maxval=4.0)
    far = _H2O_COORDS + jnp.array([25.0, 0.0, 0.0])   # 2.5 * rc away
    f_alone = compute_atom_context_descriptor(grid, _H2O_COORDS, _H2O_Z)
    f_with_far = compute_atom_context_descriptor(
        grid, jnp.concatenate([_H2O_COORDS, far], axis=0),
        jnp.concatenate([_H2O_Z, jnp.array([8, 1, 1])], axis=0))
    # The weights are bitwise zero beyond rc, so the columns agree to the
    # reduction's re-grouping only: a K=6 reduction re-groups the same
    # nonzero terms a K=3 one sums, measured at one ulp (2.2e-16). An actual
    # unmasked leak shifts columns by >= 1e-5: the far atom's unmasked weight
    # at its 19.7-Bohr nearest grid row would be 2.6e-3 under convex weights,
    # ten orders above this tolerance.
    d_far = jnp.abs(f_alone - f_with_far)
    assert bool(jnp.all(d_far <= 1e-13))

    far_point = jnp.array([[0.0, 0.0, 30.0]])            # > rc from every atom
    f_far = compute_atom_context_descriptor(far_point, _H2O_COORDS, _H2O_Z)
    assert bool(jnp.all(jnp.isfinite(f_far)))
    assert bool(jnp.all(f_far == 0.0))                    # columns exactly zero

    iso_grid = jnp.array([[0.0, 0.0, 0.5], [0.0, 0.0, 2.0], [1.0, 1.0, 1.0]])
    f_iso = compute_atom_context_descriptor(
        iso_grid, jnp.array([[0.0, 0.0, 0.0]]), jnp.array([6]))
    # One atom: no coupling is possible, but each row's (u @ e) / den
    # re-rounds through its own 6/r scale, so rows can differ by one ulp
    # (measured 2.2e-16); any real environment effect is impossible here.
    assert bool(jnp.all(jnp.abs(f_iso - f_iso[:1]) <= 1e-15))


def test_data_path_pyscfad_path_parity():
    """The data.py on-demand route (through its seam, the helper the precompute
    block calls) and the solver_pyscfad dispatch route agree with the kernel
    driven at the instance's (cutoff, seed) -- so an instance parameterized
    away from the defaults cannot drift from a hardcoded default in either
    path."""
    import pyscfad.gto as pyscfad_gto
    mol = pyscfad_gto.Mole()
    mol.atom = "O 0 0 0; H 0 0 1.81; H 0 1.65 -0.50"
    mol.basis = "sto-3g"
    mol.verbose = 0
    mol.build()

    grid = jax.random.uniform(jax.random.PRNGKey(5), (64, 3),
                              minval=-4.0, maxval=4.0)
    desc = AtomContextDescriptor(cutoff=8.0, seed=7)

    # The reference: the kernel with the instance's kwargs spelled out.
    ref = np.asarray(compute_atom_context_descriptor(
        grid, jnp.asarray(mol.atom_coords()), jnp.asarray(mol.atom_charges()),
        cutoff=float(desc.cutoff), seed=int(desc.seed)))

    via_data = np.asarray(_atom_context_features_on_demand(mol, grid, (desc,)))

    nao = int(mol.nao)
    via_pyscfad = np.asarray(_reassemble_features_on_grid(
        (desc,), jnp.zeros((nao, nao)), jnp.zeros((nao, nao)), grid, mol))

    assert via_data.shape == (grid.shape[0], 4)
    assert via_pyscfad.shape == (grid.shape[0], 4)
    np.testing.assert_allclose(via_data, ref, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(via_pyscfad, ref, atol=1e-12, rtol=0.0)


def test_registry_wiring_atom_context_and_deep_atomctx_3x16():
    """make_descriptor("atom_context") yields the 4-column geometry-only
    descriptor with declared bounds ((-1, 1),) * 4 and required mol key
    "atom_context_features"; the architecture registry carries
    deep_atomctx_3x16 built from descriptors=["atom_context"] with the
    DERIVED n_extra_features == 4 and extra_feature_bounds all (-1, 1) -- no
    per-descriptor hard-coded widths anywhere in the chain."""
    d = make_descriptor("atom_context")
    assert isinstance(d, AtomContextDescriptor)
    assert d.n_features == 4
    assert d.column_bounds == ((-1.0, 1.0),) * 4
    assert d.density_matrix_dependent is False
    assert d.required_mol_keys == ("atom_context_features",)

    arch = ARCHITECTURES["deep_atomctx_3x16"]
    assert [s.name for s in arch.descriptors] == ["atom_context"]
    assert arch.n_extra_features == 4
    assert arch.extra_feature_bounds == ((-1.0, 1.0),) * 4


def test_cache_key_distinguishes_cutoff_and_seed():
    """_precompute_cache_key separates two parameterizations of the one
    AtomContextDescriptor class along both the cutoff axis and the seed axis
    (one class, two parameterizations must not collide), while equal
    parameterizations keep one key."""
    from xcquinox.pipeline.config import MoleculeSpec

    spec = MoleculeSpec(name="h2_ctx", atom="H 0 0 0; H 0 0 1.4",
                        atom_composition=(("H", 2),))

    def key(cutoff, seed):
        return _precompute_cache_key(
            spec, ("atom_context_features",),
            (AtomContextDescriptor(cutoff=cutoff, seed=seed),))

    assert key(8.0, 7) != key(10.0, 7)     # cutoff axis
    assert key(8.0, 7) != key(8.0, 42)     # seed axis
    assert key(8.0, 7) == key(8.0, 7)      # stable for one parameterization


def test_duplicate_descriptor_class_in_one_architecture_is_refused():
    """mol_data keys are per descriptor class (the precompute stores one array
    under the first instance's key) while the pyscfad dispatch recomputes per
    instance, so two parameterizations of one class in ONE architecture would
    silently disagree between the training and eval routes; the materializer
    refuses the second spec of an already-materialized name."""
    arch = ArchitectureConfig.from_spec(
        "duplicate_ctx", 3, 16,
        descriptors=["atom_context",
                     ("atom_context", {"cutoff": 8.0})])
    with pytest.raises(ValueError, match="atom_context"):
        arch.n_extra_features  # the choke point every consumer reads
