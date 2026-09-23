import numpy as np
import pytest
import jax
import jax.numpy as jnp


# §13.2 item (1): DESCRIPTOR_REGISTRY contains both built-ins


# §13.2 item (2): make_descriptor registry roundtrip
def test_make_descriptor_roundtrip():
    from xcquinox.pipeline.descriptors import make_descriptor, CuspDescriptor, DMStatisticsDescriptor
    d_cusp = make_descriptor("cusp")
    d_dm = make_descriptor("dm_statistics")
    assert isinstance(d_cusp, CuspDescriptor)
    assert isinstance(d_dm, DMStatisticsDescriptor)


# §13.2 item (3): CuspDescriptor.n_features == 2


# §13.2 item (4): DMStatisticsDescriptor.n_features == 2


# §13.2 item (5): assemble_descriptor_features with empty tuple returns (N, 0)


# §13.2 item (6): assemble_descriptor_features with single descriptor returns correct shape


# §13.2 item (7): assemble two descriptors concatenates left-to-right
def test_assemble_two_descriptors_concatenates():
    from xcquinox.pipeline.descriptors import assemble_descriptor_features, make_descriptor
    descriptors = (make_descriptor("dm_statistics"), make_descriptor("cusp"))
    mol_data = {
        "dm_features": jnp.ones((5, 3)),
        "cusp_features": jnp.ones((5, 2)) * 2.0,
    }
    out = assemble_descriptor_features(descriptors, mol_data)
    assert out.shape == (5, 5)
    assert jnp.allclose(out[:, :3], 1.0)
    assert jnp.allclose(out[:, 3:], 2.0)


# §13.2 item (8): dm BEFORE cusp invariant in ARCHITECTURES
def test_dm_before_cusp_invariant():
    from xcquinox.pipeline.config import ARCHITECTURES
    for name, cfg in ARCHITECTURES.items():
        descr_names = [d.name for d in cfg.descriptors]
        if "dm_statistics" in descr_names and "cusp" in descr_names:
            assert descr_names.index("dm_statistics") < descr_names.index("cusp"), (
                f"Architecture {name!r} lists cusp before dm_statistics"
            )


# §13.2 item (9): make_descriptor raises KeyError on unknown name


# §13.2 item (10): list_descriptors returns sorted list


# §13.2 item (11): CuspDescriptor.compute is differentiable
def test_cusp_compute_is_differentiable():
    from xcquinox.pipeline.descriptors import make_descriptor
    d = make_descriptor("cusp")

    def scalar(x):
        return jnp.sum(d.compute({"cusp_features": x}))

    grad_fn = jax.grad(scalar)
    g = grad_fn(jnp.ones((5, 2)))
    assert g.shape == (5, 2)
    assert jnp.all(jnp.isfinite(g))


# §13.2 item (12): DMStatisticsDescriptor.compute is differentiable
def test_dm_statistics_compute_is_differentiable():
    from xcquinox.pipeline.descriptors import make_descriptor
    d = make_descriptor("dm_statistics")

    def scalar(x):
        return jnp.sum(d.compute({"dm_features": x}))

    grad_fn = jax.grad(scalar)
    g = grad_fn(jnp.ones((5, 3)))
    assert g.shape == (5, 3)
    assert jnp.all(jnp.isfinite(g))


# §13.2 item (13): D-H1 __post_init__ rejects jax.Array on float field


# §13.2 item (14): D-H3 rejects non-static field at registration time


def test_dm_statistics_compute_from_dm_matches_precomputed():
    """compute_from_dm should produce the same tiled features as the
    precompute path for identical (dm, S) inputs."""
    from xcquinox.pipeline.descriptors import DMStatisticsDescriptor
    from xcquinox.pipeline.data import precompute_fixed_density_data
    from xcquinox.pipeline.tests.fixtures.molecules import h2_molecule

    desc = DMStatisticsDescriptor()
    data = precompute_fixed_density_data(
        h2_molecule(), descriptors=(desc,),
    )
    n_grid = data["rho_grid"].shape[0]

    features_kernel = desc.compute_from_dm(
        dm=data["dm_pbe"], s_matrix=data["s_matrix"], n_grid=n_grid,
    )
    features_precomp = data["dm_features"]
    assert features_kernel.shape == features_precomp.shape
    np.testing.assert_allclose(
        np.asarray(features_kernel),
        np.asarray(features_precomp),
        atol=1e-12, rtol=0.0,
    )


# ---------------------------------------------------------------------------
# Per-spin-channel feature blocks: the symmetric doubled density
# diag(P_sigma, P_sigma) (Oliver and Perdew, Phys. Rev. A 20, 397 (1979)).
# ---------------------------------------------------------------------------

def test_doubled_spin_dm_places_the_channel_in_both_slots():
    from xcquinox.pipeline.descriptors import doubled_spin_dm
    rng = np.random.default_rng(20260821)
    p = jnp.asarray(rng.standard_normal((2, 4, 4)))
    for s in (0, 1):
        d = doubled_spin_dm(p, s)
        assert d.shape == (2, 4, 4)
        assert bool(jnp.all(d[0] == p[s]))
        assert bool(jnp.all(d[1] == p[s]))


def test_doubled_spin_dm_refuses_a_total_density_matrix():
    from xcquinox.pipeline.descriptors import doubled_spin_dm
    with pytest.raises(ValueError, match="spin-resolved"):
        doubled_spin_dm(jnp.zeros((4, 4)), 0)


def test_cusp_per_channel_block_equals_the_shared_block():
    from xcquinox.pipeline.descriptors import CuspDescriptor
    d = CuspDescriptor()
    mol_data = {"cusp_features": jnp.arange(6.0).reshape(3, 2),
                "rho_grid": jnp.ones(3)}
    for s in (0, 1):
        got = d.compute_for_spin_channel(mol_data, s)
        assert bool(jnp.all(got == mol_data["cusp_features"]))


def test_rung35_per_channel_block_reads_its_own_spin_key():
    from xcquinox.pipeline.descriptors import DMRung35Descriptor
    d = DMRung35Descriptor()
    mol_data = {"rung35_features": jnp.zeros((3, 2)),
                "rung35_features_a": jnp.full((3, 2), 0.25),
                "rung35_features_b": jnp.full((3, 2), 0.75),
                "rho_grid": jnp.ones(3)}
    assert float(d.compute_for_spin_channel(mol_data, 0)[0, 0]) == 0.25
    assert float(d.compute_for_spin_channel(mol_data, 1)[0, 0]) == 0.75


def test_per_channel_block_refuses_an_absent_spin_key():
    from xcquinox.pipeline.descriptors import DMRung35Descriptor
    d = DMRung35Descriptor()
    with pytest.raises(KeyError, match="rung35_features_a"):
        d.compute_for_spin_channel(
            {"rung35_features": jnp.zeros((3, 2)), "rung35_features_a": None}, 0)


def test_assemble_descriptor_features_spin_channel_preserves_column_order():
    from xcquinox.pipeline.descriptors import (
        assemble_descriptor_features, CuspDescriptor, DMRung35Descriptor)
    descriptors = (CuspDescriptor(), DMRung35Descriptor())
    mol_data = {
        "rho_grid": jnp.ones(3),
        "cusp_features": jnp.full((3, 2), 7.0),
        "rung35_features": jnp.zeros((3, 2)),
        "rung35_features_a": jnp.full((3, 2), 0.25),
        "rung35_features_b": jnp.full((3, 2), 0.75),
    }
    out = assemble_descriptor_features(descriptors, mol_data, spin_channel=0)
    assert out.shape == (3, 4)
    assert bool(jnp.all(out[:, :2] == 7.0))
    assert bool(jnp.all(out[:, 2:] == 0.25))


# ---------------------------------------------------------------------------
# Density-matrix dependence is declared, not inferred from the key tuple.
# ---------------------------------------------------------------------------

def test_density_matrix_dependence_flags_match_the_descriptor_family():
    from xcquinox.pipeline.descriptors import (
        CuspDescriptor, DMStatisticsDescriptor, DMRung35Descriptor,
        DMRung35MultishellDescriptor, MetaGGAAlphaDescriptor)
    assert CuspDescriptor.density_matrix_dependent is False
    for cls in (DMStatisticsDescriptor, DMRung35Descriptor,
                DMRung35MultishellDescriptor, MetaGGAAlphaDescriptor):
        assert cls.density_matrix_dependent is True, cls.__name__
        assert len(cls.spin_mol_keys) == 2, cls.__name__


# ---------------------------------------------------------------------------
# What the descriptor kernels return when handed diag(P_sigma, P_sigma).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def o_atom_mol():
    """O atom (sto-3g) from the shared fixture spec, with no SCF."""
    from pyscf import gto
    from xcquinox.pipeline.tests.fixtures.molecules import o_atom
    spec = o_atom()
    return gto.M(atom=spec.atom, basis=spec.basis, charge=spec.charge,
                 spin=spec.spin, verbose=0)


@pytest.fixture(scope="module")
def o_atom_uks(o_atom_mol):
    """O atom (sto-3g, UKS/PBE, grid level 1): a spin-resolved DM with
    P_alpha != P_beta, so a per-channel claim cannot pass by symmetry."""
    from pyscf import dft
    mf = dft.UKS(o_atom_mol)
    mf.xc = "pbe"
    mf.grids.level = 1
    mf.kernel()
    dm = jnp.asarray(mf.make_rdm1())
    assert float(jnp.abs(dm[0] - dm[1]).max()) > 1e-3
    return o_atom_mol, mf.grids.coords, dm


def test_doubled_dm_rung35_occupancy_carries_one_channel_in_both_slots(o_atom_uks):
    """n(diag(P_s, P_s)) = [n_s, n_s], the rung-3.5 ingredient of the
    spin-unpolarized system the Oliver-Perdew relation refers to."""
    from xcquinox.pipeline.descriptors import doubled_spin_dm
    from xcquinox.pipeline.rung35 import (compute_projected_ao,
                                      compute_rung35_occupancy)
    mol, coords, dm = o_atom_uks
    proj = compute_projected_ao(mol, coords)
    occ_phys = np.asarray(compute_rung35_occupancy(proj, dm))
    for s in (0, 1):
        occ_d = np.asarray(compute_rung35_occupancy(proj, doubled_spin_dm(dm, s)))
        assert occ_d.shape == occ_phys.shape
        # measured deviation 2.22e-16 in both channels, on 4328 grid points
        np.testing.assert_allclose(occ_d[:, 0], occ_phys[:, s], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(occ_d[:, 1], occ_phys[:, s], rtol=0.0, atol=1e-12)
        # Bessel bound preserved. At this projector width the occupancy is
        # dominated by the 1s core and the measured ranges reproduced to the
        # digits shown across four initial guesses (minao, 1e, atom, huckel):
        # [3.96e-04, 7.49e-01] alpha and [1.28e-04, 7.49e-01] beta.
        assert occ_d.min() >= 0.0 and occ_d.max() <= 1.0


