import numpy as np
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import ClassVar


# §13.2 item (1): DESCRIPTOR_REGISTRY contains both built-ins
def test_descriptor_registry_has_two_builtins():
    from xcquinox.alec.descriptors import DESCRIPTOR_REGISTRY
    assert "cusp" in DESCRIPTOR_REGISTRY
    assert "dm_statistics" in DESCRIPTOR_REGISTRY


# §13.2 item (2): make_descriptor registry roundtrip
def test_make_descriptor_roundtrip():
    from xcquinox.alec.descriptors import make_descriptor, CuspDescriptor, DMStatisticsDescriptor
    d_cusp = make_descriptor("cusp")
    d_dm = make_descriptor("dm_statistics")
    assert isinstance(d_cusp, CuspDescriptor)
    assert isinstance(d_dm, DMStatisticsDescriptor)


# §13.2 item (3): CuspDescriptor.n_features == 2
def test_cusp_n_features_value():
    from xcquinox.alec.descriptors import CuspDescriptor
    assert CuspDescriptor().n_features == 2


# §13.2 item (4): DMStatisticsDescriptor.n_features == 2
def test_dm_statistics_n_features_value():
    from xcquinox.alec.descriptors import DMStatisticsDescriptor
    # 3 -> 2 on 2026-08-06: dm_entropy removed (no usable gradient at any
    # converged density). This count sets the network input width.
    assert DMStatisticsDescriptor().n_features == 2


# §13.2 item (5): assemble_descriptor_features with empty tuple returns (N, 0)
def test_assemble_empty_returns_zero_width():
    from xcquinox.alec.descriptors import assemble_descriptor_features
    mol_data = {"rho_grid": jnp.ones((7,))}
    out = assemble_descriptor_features((), mol_data)
    assert out.shape == (7, 0)


# §13.2 item (6): assemble_descriptor_features with single descriptor returns correct shape
def test_assemble_single_descriptor_shape():
    from xcquinox.alec.descriptors import assemble_descriptor_features, make_descriptor
    d = make_descriptor("cusp")
    mol_data = {"cusp_features": jnp.ones((7, 2))}
    out = assemble_descriptor_features((d,), mol_data)
    assert out.shape == (7, 2)


# §13.2 item (7): assemble two descriptors concatenates left-to-right
def test_assemble_two_descriptors_concatenates():
    from xcquinox.alec.descriptors import assemble_descriptor_features, make_descriptor
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
    from xcquinox.alec.config import ARCHITECTURES
    for name, cfg in ARCHITECTURES.items():
        descr_names = [d.name for d in cfg.descriptors]
        if "dm_statistics" in descr_names and "cusp" in descr_names:
            assert descr_names.index("dm_statistics") < descr_names.index("cusp"), (
                f"Architecture {name!r} lists cusp before dm_statistics"
            )


# §13.2 item (9): make_descriptor raises KeyError on unknown name
def test_make_descriptor_unknown():
    from xcquinox.alec.descriptors import make_descriptor
    with pytest.raises(KeyError):
        make_descriptor("not-a-descriptor")


# §13.2 item (10): list_descriptors returns sorted list
def test_list_descriptors_sorted():
    from xcquinox.alec.descriptors import list_descriptors
    names = list_descriptors()
    assert names == sorted(names)


# §13.2 item (11): CuspDescriptor.compute is differentiable
def test_cusp_compute_is_differentiable():
    from xcquinox.alec.descriptors import make_descriptor
    d = make_descriptor("cusp")

    def scalar(x):
        return jnp.sum(d.compute({"cusp_features": x}))

    grad_fn = jax.grad(scalar)
    g = grad_fn(jnp.ones((5, 2)))
    assert g.shape == (5, 2)
    assert jnp.all(jnp.isfinite(g))


# §13.2 item (12): DMStatisticsDescriptor.compute is differentiable
def test_dm_statistics_compute_is_differentiable():
    from xcquinox.alec.descriptors import make_descriptor
    d = make_descriptor("dm_statistics")

    def scalar(x):
        return jnp.sum(d.compute({"dm_features": x}))

    grad_fn = jax.grad(scalar)
    g = grad_fn(jnp.ones((5, 3)))
    assert g.shape == (5, 3)
    assert jnp.all(jnp.isfinite(g))


# §13.2 item (13): D-H1 __post_init__ rejects jax.Array on float field
def test_descriptor_post_init_rejects_jax_array_on_float_field():
    from xcquinox.alec.descriptors import (
        DESCRIPTOR_REGISTRY, Descriptor, register_descriptor, make_descriptor,
    )

    try:
        @register_descriptor("_fake_float_kwarg")
        class _FakeFloatKwargDescriptor(Descriptor):
            n_features: int = eqx.field(default=2, static=True)
            scale: float = eqx.field(default=1.0, static=True)
            required_mol_keys: ClassVar[tuple[str, ...]] = ()

            def compute(self, mol_data):
                return jnp.zeros((1, 2))

        with pytest.raises(TypeError, match="scale"):
            make_descriptor("_fake_float_kwarg", scale=jnp.array(1.0))
    finally:
        DESCRIPTOR_REGISTRY.pop("_fake_float_kwarg", None)


# §13.2 item (14): D-H3 rejects non-static field at registration time
def test_register_descriptor_rejects_non_static_field():
    from xcquinox.alec.descriptors import (
        DESCRIPTOR_REGISTRY, Descriptor, register_descriptor,
    )

    try:
        with pytest.raises(TypeError, match="static"):
            @register_descriptor("bad")
            class BadDescriptor(Descriptor):
                n_features: int = eqx.field(default=2, static=True)
                required_mol_keys: ClassVar[tuple[str, ...]] = ()
                trainable: jnp.ndarray = eqx.field(default_factory=lambda: jnp.zeros(3))

                def compute(self, mol_data):
                    return jnp.zeros((1, 2))
    finally:
        DESCRIPTOR_REGISTRY.pop("bad", None)


def test_dm_statistics_compute_from_dm_matches_precomputed():
    """compute_from_dm should produce the same tiled features as the
    precompute path for identical (dm, S) inputs."""
    from xcquinox.alec.descriptors import DMStatisticsDescriptor
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule

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


def test_dm_statistics_compute_from_dm_output_shape():
    """compute_from_dm returns (n_grid, 2) tiled features."""
    from xcquinox.alec.descriptors import DMStatisticsDescriptor

    desc = DMStatisticsDescriptor()
    dm = jnp.eye(2) * 0.5
    s = jnp.eye(2)
    features = desc.compute_from_dm(dm=dm, s_matrix=s, n_grid=17)
    assert features.shape == (17, 2)


# ---------------------------------------------------------------------------
# Per-spin-channel feature blocks: the symmetric doubled density
# diag(P_sigma, P_sigma) (Oliver and Perdew, Phys. Rev. A 20, 397 (1979)).
# ---------------------------------------------------------------------------

def test_doubled_spin_dm_places_the_channel_in_both_slots():
    from xcquinox.alec.descriptors import doubled_spin_dm
    rng = np.random.default_rng(20260821)
    p = jnp.asarray(rng.standard_normal((2, 4, 4)))
    for s in (0, 1):
        d = doubled_spin_dm(p, s)
        assert d.shape == (2, 4, 4)
        assert bool(jnp.all(d[0] == p[s]))
        assert bool(jnp.all(d[1] == p[s]))


def test_doubled_spin_dm_refuses_a_total_density_matrix():
    from xcquinox.alec.descriptors import doubled_spin_dm
    with pytest.raises(ValueError, match="spin-resolved"):
        doubled_spin_dm(jnp.zeros((4, 4)), 0)


def test_doubled_spin_dm_refuses_an_out_of_range_channel():
    from xcquinox.alec.descriptors import doubled_spin_dm
    with pytest.raises(ValueError, match="spin_channel"):
        doubled_spin_dm(jnp.zeros((2, 4, 4)), 2)


def test_cusp_per_channel_block_equals_the_shared_block():
    from xcquinox.alec.descriptors import CuspDescriptor
    d = CuspDescriptor()
    mol_data = {"cusp_features": jnp.arange(6.0).reshape(3, 2),
                "rho_grid": jnp.ones(3)}
    for s in (0, 1):
        got = d.compute_for_spin_channel(mol_data, s)
        assert bool(jnp.all(got == mol_data["cusp_features"]))


def test_rung35_per_channel_block_reads_its_own_spin_key():
    from xcquinox.alec.descriptors import DMRung35Descriptor
    d = DMRung35Descriptor()
    mol_data = {"rung35_features": jnp.zeros((3, 2)),
                "rung35_features_a": jnp.full((3, 2), 0.25),
                "rung35_features_b": jnp.full((3, 2), 0.75),
                "rho_grid": jnp.ones(3)}
    assert float(d.compute_for_spin_channel(mol_data, 0)[0, 0]) == 0.25
    assert float(d.compute_for_spin_channel(mol_data, 1)[0, 0]) == 0.75


def test_metagga_and_dm_statistics_declare_their_spin_keys():
    from xcquinox.alec.descriptors import (
        DMStatisticsDescriptor, DMRung35MultishellDescriptor,
        MetaGGAAlphaDescriptor, CuspDescriptor)
    assert DMStatisticsDescriptor.spin_mol_keys == (
        "dm_features_a", "dm_features_b")
    assert DMRung35MultishellDescriptor.spin_mol_keys == (
        "rung35ms_features_a", "rung35ms_features_b")
    assert MetaGGAAlphaDescriptor.spin_mol_keys == (
        "metagga_features_a", "metagga_features_b")
    assert CuspDescriptor.spin_mol_keys == ()


def test_per_channel_block_refuses_an_absent_spin_key():
    from xcquinox.alec.descriptors import DMRung35Descriptor
    d = DMRung35Descriptor()
    with pytest.raises(KeyError, match="rung35_features_a"):
        d.compute_for_spin_channel(
            {"rung35_features": jnp.zeros((3, 2)), "rung35_features_a": None}, 0)


def test_assemble_descriptor_features_spin_channel_preserves_column_order():
    from xcquinox.alec.descriptors import (
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


def test_assemble_descriptor_features_defaults_to_the_total_block():
    from xcquinox.alec.descriptors import (
        assemble_descriptor_features, DMRung35Descriptor)
    mol_data = {
        "rho_grid": jnp.ones(3),
        "rung35_features": jnp.full((3, 2), 0.5),
        "rung35_features_a": jnp.full((3, 2), 0.25),
        "rung35_features_b": jnp.full((3, 2), 0.75),
    }
    out = assemble_descriptor_features((DMRung35Descriptor(),), mol_data)
    assert bool(jnp.all(out == 0.5))


def test_assemble_descriptor_features_empty_descriptors_ignores_spin_channel():
    from xcquinox.alec.descriptors import assemble_descriptor_features
    mol_data = {"rho_grid": jnp.ones(5)}
    assert assemble_descriptor_features((), mol_data, spin_channel=1).shape == (5, 0)


def test_doubled_spin_dm_refuses_a_boolean_channel():
    # True satisfies `in (0, 1)` yet indexes an array as a mask: before the
    # isinstance guard, doubled_spin_dm(p, True) returned shape (2, 1, 2, 2, 2)
    # and doubled_spin_dm(p, False) shape (2, 0, 2, 2, 2) on a (2, 2, 2) input,
    # neither of which is the (2, nao, nao) return contract.
    from xcquinox.alec.descriptors import doubled_spin_dm, DMRung35Descriptor
    for bad in (True, False):
        with pytest.raises(ValueError, match="spin_channel"):
            doubled_spin_dm(jnp.zeros((2, 2, 2)), bad)
        with pytest.raises(ValueError, match="spin_channel"):
            DMRung35Descriptor().compute_for_spin_channel(
                {"rung35_features_a": jnp.zeros((3, 2))}, bad)
