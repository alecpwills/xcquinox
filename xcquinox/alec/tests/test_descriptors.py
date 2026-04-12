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


# §13.2 item (4): DMStatisticsDescriptor.n_features == 3
def test_dm_statistics_n_features_value():
    from xcquinox.alec.descriptors import DMStatisticsDescriptor
    assert DMStatisticsDescriptor().n_features == 3


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
