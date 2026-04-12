import io
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx


# §13.2 items (1)-(4): AlecGGA_XNet forward pass over n_extra_features range
@pytest.mark.parametrize("n_extra,input_width", [
    (0, 2), (2, 4), (3, 5), (5, 7),
])
def test_xnet_forward_pass(n_extra, input_width):
    from xcquinox.alec.networks import AlecGGA_XNet
    xnet = AlecGGA_XNet(
        n_extra_features=n_extra, depth=2, nodes=8, seed=0,
    )
    inputs = jnp.ones(input_width)
    out = xnet(inputs)
    assert out.shape == (), f"expected scalar, got {out.shape}"
    assert jnp.isfinite(out)


# §13.2 item (5): AlecGGA_CNet forward pass on n_extra_features=0
def test_cnet_forward_pass_baseline():
    from xcquinox.alec.networks import AlecGGA_CNet
    cnet = AlecGGA_CNet(
        n_extra_features=0, depth=2, nodes=8, seed=0,
    )
    inputs = jnp.array([1.0, 0.5])
    out = cnet(inputs)
    assert out.shape == (), f"expected scalar, got {out.shape}"
    assert jnp.isfinite(out)


# §13.2 item (6): self-attention toggle produces different output
def test_attention_toggle_changes_output():
    from xcquinox.alec.networks import AlecGGA_XNet
    xnet_no_attn = AlecGGA_XNet(
        n_extra_features=0, depth=2, nodes=8, seed=42,
        use_self_attention=False,
    )
    xnet_attn = AlecGGA_XNet(
        n_extra_features=0, depth=2, nodes=8, seed=42,
        use_self_attention=True,
    )
    inputs = jnp.array([1.0, 0.5])
    out_no = xnet_no_attn(inputs)
    out_yes = xnet_attn(inputs)
    assert not jnp.allclose(out_no, out_yes), (
        "attention toggle must produce a different output"
    )


# §13.2 item (7): create_network_pair seed reproducibility
def test_create_network_pair_seed_reproducibility():
    from xcquinox.alec.networks import create_network_pair
    from xcquinox.alec.config import get_architecture
    arch = get_architecture("shallow")
    xnet1, cnet1 = create_network_pair(arch, seed=99)
    xnet2, cnet2 = create_network_pair(arch, seed=99)
    buf1, buf2 = io.BytesIO(), io.BytesIO()
    buf3, buf4 = io.BytesIO(), io.BytesIO()
    eqx.tree_serialise_leaves(buf1, xnet1)
    eqx.tree_serialise_leaves(buf2, xnet2)
    eqx.tree_serialise_leaves(buf3, cnet1)
    eqx.tree_serialise_leaves(buf4, cnet2)
    assert buf1.getvalue() == buf2.getvalue(), "xnet not reproducible"
    assert buf3.getvalue() == buf4.getvalue(), "cnet not reproducible"


# §13.2 item (8): output shape is scalar per sample
def test_output_shape_is_scalar():
    from xcquinox.alec.networks import AlecGGA_XNet, AlecGGA_CNet
    xnet = AlecGGA_XNet(n_extra_features=0, depth=2, nodes=8, seed=0)
    cnet = AlecGGA_CNet(n_extra_features=0, depth=2, nodes=8, seed=0)
    inputs = jnp.array([1.0, 0.5])
    assert xnet(inputs).ndim == 0
    assert cnet(inputs).ndim == 0


# §13.2 item (9): serialise/deserialise roundtrip
def test_serialise_deserialise_roundtrip():
    from xcquinox.alec.networks import AlecGGA_XNet
    xnet = AlecGGA_XNet(n_extra_features=2, depth=2, nodes=8, seed=7)
    inputs = jnp.array([1.0, 0.5, 0.1, 0.2])
    out_before = xnet(inputs)
    buf = io.BytesIO()
    eqx.tree_serialise_leaves(buf, xnet)
    buf.seek(0)
    skeleton = AlecGGA_XNet(n_extra_features=2, depth=2, nodes=8, seed=0)
    loaded = eqx.tree_deserialise_leaves(buf, skeleton)
    out_after = loaded(inputs)
    np.testing.assert_array_equal(np.asarray(out_before), np.asarray(out_after))


# §13.2 item (10): lower_rho_cutoff clamps tiny densities
def test_lower_rho_cutoff_clamps():
    from xcquinox.alec.networks import AlecGGA_XNet
    xnet = AlecGGA_XNet(
        n_extra_features=0, depth=2, nodes=8, seed=0,
        lower_rho_cutoff=1e-6,
    )
    tiny_rho = jnp.array([1e-30, 0.5])
    out = xnet(tiny_rho)
    assert jnp.isfinite(out), "lower_rho_cutoff should prevent NaN/inf"


# §13.2 item (11): jax.grad through AlecGGA_XNet
def test_xnet_grad_finite_nonzero():
    from xcquinox.alec.networks import AlecGGA_XNet
    xnet = AlecGGA_XNet(n_extra_features=0, depth=2, nodes=8, seed=42)
    inputs = jnp.array([1.0, 0.5])

    @eqx.filter_value_and_grad
    def loss_fn(model):
        return model(inputs)

    _, grads = loss_fn(xnet)
    leaves = jax.tree_util.tree_leaves(grads)
    array_leaves = [l for l in leaves if isinstance(l, jnp.ndarray)]
    assert len(array_leaves) > 0
    assert all(jnp.all(jnp.isfinite(l)) for l in array_leaves)
    assert any(jnp.any(l != 0) for l in array_leaves), "at least one grad must be nonzero"


# §13.2 item (12): jax.grad through AlecGGA_CNet
def test_cnet_grad_finite_nonzero():
    from xcquinox.alec.networks import AlecGGA_CNet
    cnet = AlecGGA_CNet(n_extra_features=0, depth=2, nodes=8, seed=42)
    inputs = jnp.array([1.0, 0.5])

    @eqx.filter_value_and_grad
    def loss_fn(model):
        return model(inputs)

    _, grads = loss_fn(cnet)
    leaves = jax.tree_util.tree_leaves(grads)
    array_leaves = [l for l in leaves if isinstance(l, jnp.ndarray)]
    assert len(array_leaves) > 0
    assert all(jnp.all(jnp.isfinite(l)) for l in array_leaves)
    assert any(jnp.any(l != 0) for l in array_leaves), "at least one grad must be nonzero"


# §13.2 item (13): E-H2 _AlecLOB matches library LOB bit-exactly
def test_alec_lob_matches_library_lob():
    from xcquinox.alec.networks import _AlecLOB
    from xcquinox.net import LOB
    alec_lob = _AlecLOB(limit=1.804)
    lib_lob = LOB(limit=1.804)
    grid = jnp.linspace(-5.0, 5.0, 200)
    alec_out = jax.vmap(alec_lob)(grid)
    lib_out = jax.vmap(lib_lob)(grid)
    np.testing.assert_array_equal(
        np.asarray(alec_out), np.asarray(lib_out),
        err_msg="_AlecLOB must match library LOB bit-exactly",
    )


# §13.2 item (14): E-H2 AlecGGA_XNet matches library GGA_FxNet_extended
def test_alec_xnet_matches_library():
    from xcquinox.alec.networks import AlecGGA_XNet
    from xcquinox.net import GGA_FxNet_extended
    alec = AlecGGA_XNet(
        n_extra_features=0, depth=3, nodes=16, seed=42,
        use_self_attention=False,
    )
    lib = GGA_FxNet_extended(
        depth=3, nodes=16, seed=42,
        use_self_attention=False, use_cusp=False, use_dm_features=False,
    )
    grid_rho = jnp.linspace(0.01, 2.0, 50)
    grid_sigma = jnp.linspace(0.01, 1.0, 50)
    inputs = jnp.stack([grid_rho, grid_sigma], axis=1)
    alec_out = jax.vmap(alec)(inputs)
    lib_out = jax.vmap(lib)(inputs)
    np.testing.assert_array_equal(
        np.asarray(alec_out), np.asarray(lib_out),
        err_msg="AlecGGA_XNet must match GGA_FxNet_extended bit-exactly",
    )
