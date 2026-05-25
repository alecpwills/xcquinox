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


# Lieb-Oxford bound: F_x(s) <= 1 + kappa = 1.804 globally
# (Lieb & Oxford IJQC 19, 427, 1981; Perdew/Burke/Ernzerhof PRL 77,
# 3865, 1996, eq. 14). The _AlecLOB clamp is the architectural
# enforcement; gradient descent must NOT be able to lift F_x above
# 1.804 regardless of input or network parameters.
def test_lob_ceiling_holds_for_extreme_pre_clamp_inputs():
    """``1 + _AlecLOB(x) = 1.804 * sigmoid(x - log(0.804))`` is bounded
    in [0, 1.804] by the sigmoid range. Pin this contract: even with
    arbitrarily large positive pre-clamp activations the output stays
    at or below 1.804.

    This is the LOB enforcement that lets every trained ``model.eqx``
    in the unweighted sweep respect F_x <= 1.804 (verified empirically
    across 90 / 90 specs by
    ``notebooks/analysis/audit_lob_enforcement.py``). A regression of
    the clamp form (e.g. accidentally dropping the sigmoid) would
    fail this test.
    """
    from xcquinox.alec.networks import _AlecLOB
    lob = _AlecLOB(limit=1.804)
    huge_positive = jnp.array([10.0, 100.0, 1e6, 1e9])
    huge_negative = jnp.array([-10.0, -100.0, -1e6, -1e9])
    fx_pos = 1.0 + jax.vmap(lob)(huge_positive)  # F_x = 1 + lobf(gated)
    fx_neg = 1.0 + jax.vmap(lob)(huge_negative)
    assert float(fx_pos.max()) <= 1.804 + 1e-9, (
        f"LOB UPPER bound violated: 1 + lobf(huge_positive) reached "
        f"{float(fx_pos.max())}, expected <= 1.804. The Lieb-Oxford "
        f"theorem (Lieb & Oxford 1981; PBE 1996 eq. 14) sets F_x(s) "
        f"<= 1.804 globally; a regression here breaks the central "
        f"physical guarantee of the alec exchange network."
    )
    assert float(fx_pos.min()) >= 1.0 - 1e-9, (
        f"At very large positive pre-clamp activation, sigmoid -> 1, "
        f"so 1 + lobf(x) = 1 + 0.804 = 1.804; got {float(fx_pos.min())}."
    )
    # Lower bound from the symmetric sigmoid: 1 + lobf(-inf) -> 0.
    assert float(fx_neg.min()) >= 0.0 - 1e-9
    assert float(fx_neg.max()) <= 1.0 + 1e-9


def test_lob_limit_is_static_field_not_trainable():
    """The Lieb-Oxford ceiling 1.804 is a physical constant; it must
    NOT be a JAX leaf that gradient descent could mutate.
    ``_AlecLOB.limit`` is declared ``eqx.field(static=True)`` so
    eqx.partition / eqx.is_array filters out the limit from the
    trainable pytree.
    """
    from xcquinox.alec.networks import _AlecLOB
    lob = _AlecLOB(limit=1.804)
    arrays, _static = eqx.partition(lob, eqx.is_array)
    # arrays should have no leaves (limit is static, not an array).
    leaves = jax.tree_util.tree_leaves(arrays)
    assert len(leaves) == 0, (
        f"_AlecLOB.limit must be eqx.field(static=True); found "
        f"{len(leaves)} trainable leaves. Gradient descent would "
        f"otherwise be able to relax the LOB ceiling."
    )


def test_xnet_fx_at_s_zero_equals_one():
    """UEG limit (Slater 1951; PBE 1996 §3): F_x(s=0) = 1 exactly.
    The tanh(s)^2 gate in AlecGGA_XNet ensures this regardless of
    network parameters.
    """
    from xcquinox.alec.networks import AlecGGA_XNet
    xnet = AlecGGA_XNet(
        n_extra_features=0, depth=3, nodes=16, seed=42,
        use_self_attention=False, lob_lim=1.804,
    )
    # Inputs: rho > 0, sigma = 0 implies s = 0.
    # The network input layout is [rho, sigma, *extras] for n_extra_features=0.
    fx = xnet(jnp.array([1.0, 0.0]))
    assert abs(float(fx) - 1.0) < 1e-10, (
        f"F_x(s=0) must equal 1 (UEG limit, Slater 1951; PBE 1996 §3); "
        f"got {float(fx)}. The tanh(s)^2 gate is supposed to zero out "
        f"any deviation at s=0; a regression here breaks the UEG limit."
    )


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


# ---------------------------------------------------------------------------
# Self-attention integration tests (spec §Tests 15-19)
# ---------------------------------------------------------------------------

def test_alec_xnet_uses_real_attention():
    """Test 15: outputs differ AND grads on the FIRST MLP layer differ
    when use_self_attention is toggled — not just attention weights.
    """
    from xcquinox.alec.networks import AlecGGA_XNet

    no_attn = AlecGGA_XNet(
        n_extra_features=0, depth=2, nodes=8, num_heads=1,
        use_self_attention=False, seed=42,
    )
    yes_attn = AlecGGA_XNet(
        n_extra_features=0, depth=2, nodes=8, num_heads=2,
        use_self_attention=True, seed=42,
    )
    x = jnp.array([1.0, 0.5])
    target = jnp.array(1.2)

    out_no = no_attn(x)
    out_yes = yes_attn(x)
    assert not jnp.allclose(out_no, out_yes), (
        "attention toggle must change output"
    )

    def loss(m, x, t):
        return (m(x) - t) ** 2
    g_no = eqx.filter_grad(loss)(no_attn, x, target)
    g_yes = eqx.filter_grad(loss)(yes_attn, x, target)

    # First MLP layer weights should receive measurably different grads
    # because the residual flow through attention changes upstream signal.
    g_no_l0 = jnp.asarray(g_no.net.layers[0].weight)
    g_yes_l0 = jnp.asarray(g_yes.net.layers[0].weight)
    assert not jnp.allclose(g_no_l0, g_yes_l0, atol=1e-6), (
        "MLP layer 0 grads should differ between attn-on and attn-off"
    )


def test_alec_xnet_num_heads_propagates():
    """Test 16: AlecGGA_XNet(num_heads=4) -> xnet.attention.num_heads==4,
    head_dim == nodes // 4.
    """
    from xcquinox.alec.networks import AlecGGA_XNet
    xnet = AlecGGA_XNet(
        n_extra_features=0, depth=2, nodes=32, num_heads=4,
        use_self_attention=True, seed=0,
    )
    assert xnet.num_heads == 4
    assert xnet.attention.num_heads == 4
    assert xnet.attention.head_dim == 32 // 4


def test_alec_create_network_pair_reads_arch_num_heads():
    """Test 17: create_network_pair reads arch.num_heads and forwards."""
    from xcquinox.alec.config import ArchitectureConfig
    from xcquinox.alec.networks import create_network_pair
    arch = ArchitectureConfig(
        name="t", depth=2, nodes=8, attention=True, num_heads=2,
    )
    xnet, cnet = create_network_pair(arch, seed=0)
    assert xnet.attention.num_heads == 2
    assert cnet.attention.num_heads == 2


def test_alec_arch_validation_rejects_bad_divisibility():
    """Test 18: nodes=8 + num_heads=3 raises in __post_init__."""
    from xcquinox.alec.config import ArchitectureConfig
    with pytest.raises(ValueError, match="divisible by num_heads"):
        ArchitectureConfig(
            name="bad", depth=2, nodes=8, attention=True, num_heads=3,
        )


def test_alec_xnet_attention_grad_end_to_end():
    """Test 19: full-network grad reaches grad.attention.query_proj.weight."""
    from xcquinox.alec.networks import AlecGGA_XNet
    xnet = AlecGGA_XNet(
        n_extra_features=0, depth=2, nodes=8, num_heads=2,
        use_self_attention=True, seed=0,
    )
    x = jnp.array([1.0, 0.5])
    target = jnp.array(1.2)

    def loss(m, x, t):
        return (m(x) - t) ** 2
    grad = eqx.filter_grad(loss)(xnet, x, target)
    g_q = jnp.asarray(grad.attention.query_proj.weight)
    assert jnp.all(jnp.isfinite(g_q))
    assert jnp.linalg.norm(g_q) > 0.0


# P2-03: gated spin-polarization (zeta) input on the correlation network
def test_cnet_spin_polarization_input_width_and_sensitivity():
    import jax.numpy as jnp
    from xcquinox.alec.networks import AlecGGA_CNet
    # flag off (default): in_size = 2 + n_extra, byte-identical behavior
    c_off = AlecGGA_CNet(n_extra_features=3, depth=2, nodes=8, seed=0)
    assert c_off.use_spin_polarization is False
    assert c_off.net.in_size == 2 + 3
    # flag on: in_size = 3 + n_extra (extra slot for x1)
    c_on = AlecGGA_CNet(n_extra_features=3, depth=2, nodes=8, seed=0,
                        use_spin_polarization=True)
    assert c_on.use_spin_polarization is True
    assert c_on.net.in_size == 3 + 3
    # zeta (inputs[2]) must change the polarized output; extras shift to [3:]
    base = jnp.array([0.7, 0.2, 0.0, 0.1, 0.2, 0.3])   # [rho, sigma, zeta, *3 extras]
    f0 = float(c_on(base))
    fz = float(c_on(base.at[2].set(0.8)))
    assert abs(f0 - fz) > 1e-7, (f0, fz)


def test_arch_polarized_correlation_flag_and_width():
    from xcquinox.alec.config import get_architecture, ArchitectureConfig
    import xcquinox.alec as alec
    a0 = get_architecture("deep_combined_attn")
    # Polarized arch is built via from_spec (kept OUT of the canonical registry).
    a1 = ArchitectureConfig.from_spec(
        "deep_combined_attn_polc", 4, 32, attention=True, num_heads=4,
        descriptors=["dm_statistics", "cusp"], use_polarized_correlation=True)
    assert a0.use_polarized_correlation is False
    assert a1.use_polarized_correlation is True
    assert a1.n_input_features == a0.n_input_features + 1
    _, c0 = alec.create_network_pair(a0, seed=0)
    _, c1 = alec.create_network_pair(a1, seed=0)
    assert c0.use_spin_polarization is False and c1.use_spin_polarization is True
    assert c1.net.in_size == c0.net.in_size + 1


# ---------------------------------------------------------------------------
# Intrinsic constraint enforcement (constraints baked into the network forward)
# ---------------------------------------------------------------------------

def test_network_constraints_default_is_noop():
    """An unconstrained network's __call__ is byte-identical to its eval_core —
    protects the default (constraints=()) path."""
    from xcquinox.alec.networks import AlecGGA_XNet, AlecGGA_CNet
    xnet = AlecGGA_XNet(n_extra_features=0, depth=2, nodes=8, seed=0)
    cnet = AlecGGA_CNet(n_extra_features=0, depth=2, nodes=8, seed=0)
    xin = jnp.array([0.5, 0.2])
    cin = jnp.array([0.5, 0.2])
    assert jnp.array_equal(xnet(xin), xnet.eval_core(xin))
    assert jnp.array_equal(cnet(cin), cnet.eval_core(cin))


def test_xnet_lieb_oxford_constraint_bounds_forward():
    """A network carrying the Lieb-Oxford constraint (built-in lob disabled)
    keeps F_x in (0, mu) for every input — the constraint is enforced in the
    forward pass, so pretraining/training/eval all see the bound."""
    from xcquinox.alec.networks import AlecGGA_XNet
    from xcquinox.alec.constraints import LiebOxfordBound
    mu = 1.804
    xnet = AlecGGA_XNet(
        n_extra_features=0, depth=2, nodes=8, seed=3,
        lob_lim=None, constraints=(LiebOxfordBound(),))
    rng = np.random.default_rng(0)
    for _ in range(64):
        inp = jnp.array([abs(rng.normal()) + 1e-3, abs(rng.normal())])
        F = float(xnet(inp))
        assert 0.0 < F < mu + 1e-6
    # Exactly equals 1 + I_mu(raw - 1) over the unconstrained core.
    inp = jnp.array([0.4, 0.3])
    raw = float(xnet.eval_core(inp))
    expected = 1.0 + (mu / (1.0 + (mu - 1.0) * np.exp(-(raw - 1.0))) - 1.0)
    assert abs(float(xnet(inp)) - expected) < 1e-6


def test_create_network_pair_bakes_arch_constraints():
    """create_network_pair materializes the arch's constraints into the
    networks (the source of truth for intrinsic enforcement)."""
    from xcquinox.alec.networks import create_network_pair
    from xcquinox.alec.config import ArchitectureConfig
    arch = ArchitectureConfig.from_spec(
        "t", 2, 8, x_constraints=["lieb_oxford"],
        c_constraints=["non_negative_correlation"])
    xnet, cnet = create_network_pair(arch, seed=0)
    assert [c.registry_name for c in xnet.constraints] == ["lieb_oxford"]
    assert [c.registry_name for c in cnet.constraints] == ["non_negative_correlation"]
    assert xnet.lobf is None  # lieb_oxford disables the built-in LOB wrap


def test_constrained_network_serialization_roundtrip_leaffree_constraints():
    """Constraints are static (no array leaves), so adding the ``constraints``
    field does not change the serialized leaf stream: a checkpoint saved with
    constraints=() deserializes correctly into a constrained skeleton (weights
    transfer; the constraint comes from the skeleton). Both share the same
    non-constraint static config (lob_lim=None), isolating the constraints field
    as the only difference."""
    from xcquinox.alec.networks import AlecGGA_XNet
    from xcquinox.alec.constraints import LiebOxfordBound
    plain = AlecGGA_XNet(n_extra_features=0, depth=2, nodes=8, seed=5,
                         lob_lim=None)  # constraints=() default
    buf = io.BytesIO()
    eqx.tree_serialise_leaves(buf, plain)
    buf.seek(0)
    constrained_skel = AlecGGA_XNet(
        n_extra_features=0, depth=2, nodes=8, seed=0,
        lob_lim=None, constraints=(LiebOxfordBound(),))
    loaded = eqx.tree_deserialise_leaves(buf, constrained_skel)
    inp = jnp.array([0.4, 0.3])
    # Weights transferred -> identical unconstrained core; the skeleton's
    # constraint is present and applied in __call__.
    assert len(loaded.constraints) == 1
    assert abs(float(loaded.eval_core(inp)) - float(plain.eval_core(inp))) < 1e-6
    assert float(loaded(inp)) < 1.804 + 1e-6  # constraint actually enforced
