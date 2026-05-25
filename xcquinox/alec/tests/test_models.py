import io
import dataclasses
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from xcquinox.alec.config import ArchitectureConfig, FeatureSpec, _FrozenDict
from xcquinox.alec.models import AlecGGAModel


def _make_arch(**overrides):
    defaults = dict(name="t", depth=2, nodes=8, attention=False,
                    descriptors=(), x_constraints=(), c_constraints=(),
                    double_lob_clamp_allowed=False)
    defaults.update(overrides)
    return ArchitectureConfig(**defaults)


# §13.2 item (1)
def test_from_arch_zero_descriptors():
    arch = _make_arch()
    model = AlecGGAModel.from_arch(arch, seed=0)
    assert len(model.descriptors) == 0
    assert model.xnet.n_extra_features == 0
    assert model.cnet.n_extra_features == 0


# §13.2 item (2)
def test_from_arch_one_descriptor():
    arch = ArchitectureConfig.from_spec("t", 2, 8, descriptors=["cusp"])
    model = AlecGGAModel.from_arch(arch, seed=0)
    assert model.xnet.n_extra_features == 2


# §13.2 item (3)
def test_from_arch_two_descriptors():
    arch = ArchitectureConfig.from_spec("t", 2, 8,
                                        descriptors=["dm_statistics", "cusp"])
    model = AlecGGAModel.from_arch(arch, seed=0)
    assert model.xnet.n_extra_features == 5


# §13.2 item (4)
def test_from_arch_with_x_constraints():
    arch = ArchitectureConfig.from_spec("t", 2, 8, x_constraints=["lieb_oxford"])
    model = AlecGGAModel.from_arch(arch, seed=0)
    assert model.xnet.lob_lim is None  # C-H1
    assert len(model.x_constraints) == 1


# §13.2 item (5)
def test_from_arch_without_x_constraints():
    arch = _make_arch()
    model = AlecGGAModel.from_arch(arch, seed=0)
    assert model.xnet.lob_lim == 1.804
    assert len(model.x_constraints) == 0


# §13.2 item (6)
def test_from_arch_with_c_constraints():
    arch = ArchitectureConfig.from_spec("t", 2, 8,
                                        c_constraints=["non_negative_correlation"])
    model = AlecGGAModel.from_arch(arch, seed=0)
    assert len(model.c_constraints) == 1
    assert model.cnet.lob_lim == 2.0  # cnet default, NOT from arch


# §13.2 item (7)
def test_from_arch_without_c_constraints():
    arch = _make_arch()
    model = AlecGGAModel.from_arch(arch, seed=0)
    assert len(model.c_constraints) == 0
    assert model.cnet.lob_lim == 2.0


def _synth_inputs(n=16, n_extra=0):
    rho = jnp.linspace(0.1, 2.0, n)
    sigma = jnp.linspace(0.01, 1.0, n)
    features = jnp.ones((n, n_extra)) * 0.1 if n_extra > 0 else jnp.zeros((n, 0))
    return rho, sigma, features


# §13.2 item (8)
def test_eval_fx_returns_correct_shape():
    arch = ArchitectureConfig.from_spec("t", 2, 8, descriptors=["dm_statistics", "cusp"])
    model = AlecGGAModel.from_arch(arch, seed=0)
    rho, sigma, features = _synth_inputs(16, 5)
    assert model.eval_Fx(rho, sigma, features).shape == (16,)


# §13.2 item (9)
def test_eval_fc_returns_correct_shape():
    arch = ArchitectureConfig.from_spec("t", 2, 8, descriptors=["dm_statistics", "cusp"])
    model = AlecGGAModel.from_arch(arch, seed=0)
    rho, sigma, features = _synth_inputs(16, 5)
    assert model.eval_Fc(rho, sigma, features).shape == (16,)


# §13.2 item (10)
def test_eval_exc_returns_rho_times_ex_fx_plus_ec_fc():
    from xcquinox.utils import lda_x, pw92c_unpolarized_scalar
    arch = _make_arch()
    model = AlecGGAModel.from_arch(arch, seed=0)
    rho, sigma, features = _synth_inputs(16, 0)
    exc = model.eval_exc(rho, sigma, features)
    Fx = model.eval_Fx(rho, sigma, features)
    Fc = model.eval_Fc(rho, sigma, features)
    rho_safe = jnp.maximum(rho, model.rho_cutoff)
    # SOLV-01: eval_exc is now the exact sum of the exchange-only and
    # correlation-only pieces (rho*ex*Fx) + (rho*ec*Fc). This differs from
    # the pre-SOLV-01 rho*(ex*Fx + ec*Fc) grouping by at most ~1 ULP of
    # floating-point reassociation; the split form is the new contract so
    # that eval_exc == eval_ex + eval_ec holds exactly.
    expected = rho_safe * lda_x(rho_safe) * Fx + rho_safe * pw92c_unpolarized_scalar(rho_safe) * Fc
    np.testing.assert_array_equal(np.asarray(exc), np.asarray(expected))


# §13.2 item (11)
def test_serialization_roundtrip_preserves_eval_exc_bitwise():
    arch = _make_arch()
    model = AlecGGAModel.from_arch(arch, seed=7)
    rho, sigma, features = _synth_inputs(8, 0)
    out_before = model.eval_exc(rho, sigma, features)
    buf = io.BytesIO()
    eqx.tree_serialise_leaves(buf, model)
    buf.seek(0)
    skeleton = AlecGGAModel.from_arch(arch, seed=0)
    loaded = eqx.tree_deserialise_leaves(buf, skeleton)
    out_after = loaded.eval_exc(rho, sigma, features)
    np.testing.assert_array_equal(np.asarray(out_before), np.asarray(out_after))


# §13.2 item (12)
def test_constraint_report_returns_nested_dict():
    arch = ArchitectureConfig.from_spec(
        "t", 2, 8,
        x_constraints=["lieb_oxford"],
        c_constraints=["non_negative_correlation"],
    )
    model = AlecGGAModel.from_arch(arch, seed=0)
    rho = jnp.array([0.5, 1.0, 2.0])
    sigma = jnp.array([0.2, 0.5, 1.5])
    features = jnp.zeros((3, model.xnet.n_extra_features))
    report = model.constraint_report(rho, sigma, features)
    assert "x" in report and "c" in report
    assert "lieb_oxford" in report["x"]
    assert set(report["x"]["lieb_oxford"].keys()) == {"max", "mean", "l2"}
    assert "non_negative_correlation" in report["c"]


# §13.2 item (13)
def test_jax_grad_eval_fx_returns_finite_gradients():
    arch = _make_arch()
    model = AlecGGAModel.from_arch(arch, seed=42)
    rho, sigma, features = _synth_inputs(4, 0)
    trainable, static = eqx.partition(model, eqx.is_array)

    @jax.grad
    def loss_fn(params):
        m = eqx.combine(params, static)
        return m.eval_Fx(rho, sigma, features).sum()

    grads = loss_fn(trainable)
    leaves = jax.tree_util.tree_leaves(grads)
    array_leaves = [l for l in leaves if isinstance(l, jnp.ndarray)]
    assert all(jnp.all(jnp.isfinite(l)) for l in array_leaves)
    assert any(jnp.any(l != 0) for l in array_leaves)


# §13.2 item (14)
def test_jax_grad_eval_fc_returns_finite_gradients():
    arch = _make_arch()
    model = AlecGGAModel.from_arch(arch, seed=42)
    rho, sigma, features = _synth_inputs(4, 0)
    trainable, static = eqx.partition(model, eqx.is_array)

    @jax.grad
    def loss_fn(params):
        m = eqx.combine(params, static)
        return m.eval_Fc(rho, sigma, features).sum()

    grads = loss_fn(trainable)
    leaves = jax.tree_util.tree_leaves(grads)
    array_leaves = [l for l in leaves if isinstance(l, jnp.ndarray)]
    assert all(jnp.all(jnp.isfinite(l)) for l in array_leaves)
    assert any(jnp.any(l != 0) for l in array_leaves)


# §13.2 item (15)
def test_jax_grad_eval_exc_returns_finite_nonzero_gradients():
    arch = _make_arch()
    model = AlecGGAModel.from_arch(arch, seed=42)
    rho, sigma, features = _synth_inputs(4, 0)
    trainable, static = eqx.partition(model, eqx.is_array)

    @jax.grad
    def loss_fn(params):
        m = eqx.combine(params, static)
        return m.eval_exc(rho, sigma, features).sum()

    grads = loss_fn(trainable)
    leaves = jax.tree_util.tree_leaves(grads)
    array_leaves = [l for l in leaves if isinstance(l, jnp.ndarray)]
    assert all(jnp.all(jnp.isfinite(l)) for l in array_leaves)
    assert any(jnp.any(l != 0) for l in array_leaves)


# §13.2 item (16)
def test_jit_retrace_only_on_static_metadata_change():
    arch = _make_arch()
    model = AlecGGAModel.from_arch(arch, seed=0)
    rho_a, sigma_a, features = _synth_inputs(4, 0)
    rho_b = rho_a + 0.01
    sigma_b = sigma_a + 0.01
    trace_count = [0]

    @eqx.filter_jit
    def _jitted_eval_exc(m, r, s, f):
        trace_count[0] += 1
        return m.eval_exc(r, s, f)

    _jitted_eval_exc(model, rho_a, sigma_a, features).block_until_ready()
    _jitted_eval_exc(model, rho_b, sigma_b, features).block_until_ready()
    assert trace_count[0] == 1, "same model + same-shape inputs must hit cache"

    # Constraints now live on the networks (intrinsic enforcement), so the
    # model no longer takes x_constraints/c_constraints kwargs. Changing a
    # different static field (rho_cutoff) still invalidates the jit cache.
    model2 = AlecGGAModel(
        xnet=model.xnet, cnet=model.cnet,
        descriptors=model.descriptors,
        rho_cutoff=1e-12,
    )
    _jitted_eval_exc(model2, rho_a, sigma_a, features).block_until_ready()
    assert trace_count[0] == 2, "static-field change must invalidate cache"


# §13.2 item (17)
def test_from_spec_dict_kwargs_roundtrip():
    arch_a = ArchitectureConfig.from_spec("t", 2, 8, descriptors=[("cusp", {})])
    arch_b = ArchitectureConfig(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(FeatureSpec.of("cusp"),),
        x_constraints=(), c_constraints=(), double_lob_clamp_allowed=False,
    )
    model_a = AlecGGAModel.from_arch(arch_a, seed=0)
    model_b = AlecGGAModel.from_arch(arch_b, seed=0)
    rho, sigma, features = _synth_inputs(4, 2)
    out_a = model_a.eval_exc(rho, sigma, features)
    out_b = model_b.eval_exc(rho, sigma, features)
    np.testing.assert_array_equal(np.asarray(out_a), np.asarray(out_b))


# §13.2 item (18)
def test_rho_cutoff_clamps_not_zeros():
    """``rho_cutoff`` clamps the LDA pre-factor without zeroing it out
    (so V_xc tail evaluations stay numerically well-behaved). The test
    uses inputs above ``_NN_TAIL_THRESHOLD`` (1e-10) so the explicit
    rho_cutoff is the only clamp in play; below that threshold,
    ``eval_exc`` masks F_x = F_c = 1 to keep gradients finite on
    open-shell atoms (F-H test-quality audit fix: pre-fix test put
    inputs at the threshold and called eval_Fx/eval_Fc directly which
    bypass the mask, causing index-by-index disagreement)."""
    from xcquinox.utils import lda_x, pw92c_unpolarized_scalar
    arch = _make_arch()
    cutoff = 1e-6
    model = AlecGGAModel.from_arch(arch, seed=0, rho_cutoff=cutoff)
    # All values strictly > 1e-10 so the tail-mask in eval_exc never
    # fires; the only clamp is rho_cutoff = 1e-6 (clamps 2e-10, 1e-8,
    # 1e-7 up to 1e-6).
    rho = jnp.array([2e-10, 1e-8, 1e-7, 1.0])
    sigma = jnp.array([0.01, 0.01, 0.01, 0.5])
    features = jnp.zeros((4, 0))
    exc = model.eval_exc(rho, sigma, features)
    rho_safe = jnp.maximum(rho, cutoff)
    Fx = model.eval_Fx(rho, sigma, features)
    Fc = model.eval_Fc(rho, sigma, features)
    # SOLV-01 split grouping (see test_eval_exc_returns_rho_times_ex_fx_plus_ec_fc).
    expected = rho_safe * lda_x(rho_safe) * Fx + rho_safe * pw92c_unpolarized_scalar(rho_safe) * Fc
    np.testing.assert_array_equal(np.asarray(exc), np.asarray(expected))
    # The clamped points must be non-zero (small but positive).
    assert jnp.all(jnp.abs(exc[:3]) > 0), "clamped rho must produce non-zero exc"


# SOLV-01 Test A: exchange/correlation split is exact at the model level.
def test_eval_exc_equals_eval_ex_plus_eval_ec_batched():
    """SOLV-01: ``eval_exc == eval_ex + eval_ec`` pointwise (batched path).

    The split is required so the UKS energy can spin-scale the exchange
    piece (Oliver & Perdew, PRA 20, 397 (1979)) while evaluating the
    correlation piece on the TOTAL density (von Barth & Hedin, J. Phys. C
    5, 1629 (1972); PW92, PRB 45, 13244 (1992)). The split must be exact
    so energy and potential code paths stay mutually consistent.
    """
    for cfg in [
        dict(x_constraints=[], c_constraints=[]),
        dict(x_constraints=["lieb_oxford"], c_constraints=["non_negative_correlation"]),
    ]:
        arch = ArchitectureConfig.from_spec("t", 2, 8, **cfg)
        model = AlecGGAModel.from_arch(arch, seed=0)
        rho, sigma, features = _synth_inputs(16, 0)
        exc = model.eval_exc(rho, sigma, features)
        ex = model.eval_ex(rho, sigma, features)
        ec = model.eval_ec(rho, sigma, features)
        np.testing.assert_array_equal(np.asarray(exc), np.asarray(ex + ec))


def test_eval_exc_scalar_equals_ex_plus_ec_scalar():
    """SOLV-01: scalar split matches scalar combined, point by point.

    Also covers the tail region (rho below ``_NN_TAIL_THRESHOLD``) so the
    identical-masking requirement is exercised.
    """
    arch = ArchitectureConfig.from_spec("t", 2, 8,
                                        x_constraints=["lieb_oxford"],
                                        c_constraints=["non_negative_correlation"])
    model = AlecGGAModel.from_arch(arch, seed=0)
    rho_pts = jnp.array([1e-12, 0.1, 0.5, 1.0, 2.0])  # first is a tail point
    sigma_pts = jnp.array([0.0, 0.01, 0.1, 0.5, 1.0])
    features = jnp.zeros((5, 0))
    for i in range(rho_pts.shape[0]):
        exc = model.eval_exc_scalar(rho_pts[i], sigma_pts[i], features[i])
        ex = model.eval_ex_scalar(rho_pts[i], sigma_pts[i], features[i])
        ec = model.eval_ec_scalar(rho_pts[i], sigma_pts[i], features[i])
        np.testing.assert_array_equal(np.asarray(exc), np.asarray(ex + ec))


def test_eval_ex_ec_batched_match_scalar():
    """SOLV-01: batched eval_ex/eval_ec match their scalar counterparts."""
    arch = ArchitectureConfig.from_spec("t", 2, 8,
                                        x_constraints=["lieb_oxford"],
                                        c_constraints=["non_negative_correlation"])
    model = AlecGGAModel.from_arch(arch, seed=0)
    rho = jnp.array([0.1, 0.5, 1.0, 2.0])
    sigma = jnp.array([0.01, 0.1, 0.5, 1.0])
    features = jnp.zeros((4, 0))
    ex_b = model.eval_ex(rho, sigma, features)
    ec_b = model.eval_ec(rho, sigma, features)
    for i in range(4):
        np.testing.assert_array_equal(
            np.asarray(model.eval_ex_scalar(rho[i], sigma[i], features[i])),
            np.asarray(ex_b[i]),
        )
        np.testing.assert_array_equal(
            np.asarray(model.eval_ec_scalar(rho[i], sigma[i], features[i])),
            np.asarray(ec_b[i]),
        )


# §13.2 item (19)
def test_eval_exc_scalar_matches_constrained_eval_exc():
    configs = [
        dict(x_constraints=[], c_constraints=[]),
        dict(x_constraints=["lieb_oxford"], c_constraints=[]),
        dict(x_constraints=[], c_constraints=["ueg_limit"]),
        dict(x_constraints=["lieb_oxford"], c_constraints=["non_negative_correlation"]),
    ]
    for cfg in configs:
        arch = ArchitectureConfig.from_spec("t", 2, 8, **cfg)
        model = AlecGGAModel.from_arch(arch, seed=0)
        rho_pts = jnp.array([0.1, 0.5, 1.0, 2.0])
        sigma_pts = jnp.array([0.01, 0.1, 0.5, 1.0])
        features = jnp.zeros((4, 0))
        batched = model.eval_exc(rho_pts, sigma_pts, features)
        for i in range(4):
            scalar = model.eval_exc_scalar(
                rho_pts[i], sigma_pts[i], features[i],
            )
            np.testing.assert_array_equal(
                np.asarray(scalar), np.asarray(batched[i]),
                err_msg=f"scalar/batched mismatch at point {i} with cfg={cfg}",
            )


# P2-03: model-level zeta threading for spin-polarization-aware correlation
def _build_polc_model(seed=0):
    import xcquinox.alec as alec
    from xcquinox.alec.config import ArchitectureConfig
    arch = ArchitectureConfig.from_spec(
        "polc_test", 4, 32, attention=True, num_heads=4,
        descriptors=["dm_statistics", "cusp"], use_polarized_correlation=True)
    x, c = alec.create_network_pair(arch, seed=seed)
    return alec.AlecGGAModel.from_arch(arch, xnet=x, cnet=c)


def test_unpolarized_model_ignores_zeta():
    """An unpolarized cnet (default) must ignore zeta entirely (no regression)."""
    import xcquinox.alec as alec
    from xcquinox.alec.config import get_architecture
    arch = get_architecture("deep_combined_attn")
    x, c = alec.create_network_pair(arch, seed=0)
    m = alec.AlecGGAModel.from_arch(arch, xnet=x, cnet=c)
    n = 5
    rho = jnp.linspace(0.1, 1.0, n); sig = jnp.linspace(0.01, 0.5, n)
    feats = jnp.asarray(np.random.default_rng(0).standard_normal((n, 5)))
    assert jnp.allclose(m.eval_ec(rho, sig, feats, zeta=0.0),
                        m.eval_ec(rho, sig, feats, zeta=0.7))


def test_polarized_model_split_exact_and_zeta_sensitive():
    """Polarized model: eval_exc == eval_ex + eval_ec (batched zeta array AND
    scalar), and eval_ec genuinely depends on zeta."""
    m = _build_polc_model()
    n = 5
    rho = jnp.linspace(0.1, 1.0, n); sig = jnp.linspace(0.01, 0.5, n)
    feats = jnp.asarray(np.random.default_rng(1).standard_normal((n, 5)))
    zeta = jnp.linspace(0.0, 0.8, n)
    exc = m.eval_exc(rho, sig, feats, zeta=zeta)
    split = m.eval_ex(rho, sig, feats) + m.eval_ec(rho, sig, feats, zeta=zeta)
    assert jnp.allclose(exc, split), float(jnp.max(jnp.abs(exc - split)))
    # scalar split exact
    exc_s = float(m.eval_exc_scalar(rho[0], sig[0], feats[0], zeta=0.3))
    split_s = float(m.eval_ex_scalar(rho[0], sig[0], feats[0])
                    + m.eval_ec_scalar(rho[0], sig[0], feats[0], zeta=0.3))
    assert abs(exc_s - split_s) < 1e-12
    # zeta sensitivity
    ec0 = m.eval_ec(rho, sig, feats, zeta=0.0 * rho)
    ec5 = m.eval_ec(rho, sig, feats, zeta=0.0 * rho + 0.5)
    assert float(jnp.max(jnp.abs(ec0 - ec5))) > 1e-8


def test_polarized_baseline_reduces_to_unpolarized_at_zeta0():
    """At zeta=0 the model's correlation baseline equals the unpolarized PW92,
    so a polarized model's eval_ec(zeta=0) uses the same baseline (only the
    learned cnet differs)."""
    from xcquinox.utils import pw92c_unpolarized_scalar
    m = _build_polc_model()
    rho = jnp.linspace(0.1, 1.0, 5)
    base_pol0 = m._ec_baseline(rho, jnp.zeros_like(rho))
    assert jnp.allclose(base_pol0, pw92c_unpolarized_scalar(rho), atol=1e-12)


# Intrinsic-constraint relocation: the model delegates constraint enforcement to
# the networks. eval_Fx/eval_Fc must equal explicit composition of the arch's
# constraints over the network's UNCONSTRAINED core — i.e. the relocation is a
# behavior-preserving move of the same _compose_constraints chain.
def test_eval_fx_equals_explicit_composition_over_core():
    from xcquinox.alec.constraints import _compose_constraints
    arch = ArchitectureConfig.from_spec("t", 2, 8, x_constraints=["lieb_oxford"])
    model = AlecGGAModel.from_arch(arch, seed=0)
    rho = jnp.array([0.05, 0.2, 0.7, 1.5])
    sigma = jnp.array([0.01, 0.1, 0.3, 0.8])
    feats = jnp.zeros((4, 0))

    def base(r, s, f):
        return jax.vmap(lambda rr, ss, ff: model.xnet._core(rr, ss, ff))(r, s, f)

    expected = _compose_constraints(base, model.x_constraints)(rho, sigma, feats)
    got = model.eval_Fx(rho, sigma, feats)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected),
                               rtol=1e-6, atol=1e-7)


def test_eval_fc_equals_explicit_composition_over_core():
    from xcquinox.alec.constraints import _compose_constraints
    arch = ArchitectureConfig.from_spec(
        "t", 2, 8, c_constraints=["non_negative_correlation"])
    model = AlecGGAModel.from_arch(arch, seed=0)
    rho = jnp.array([0.05, 0.2, 0.7, 1.5])
    sigma = jnp.array([0.01, 0.1, 0.3, 0.8])
    feats = jnp.zeros((4, 0))

    def base(r, s, f):
        return jax.vmap(lambda rr, ss, ff: model.cnet._core(rr, ss, ff, 0.0))(r, s, f)

    expected = _compose_constraints(base, model.c_constraints)(rho, sigma, feats)
    got = model.eval_Fc(rho, sigma, feats)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected),
                               rtol=1e-6, atol=1e-7)


def test_constraints_are_sourced_from_networks():
    arch = ArchitectureConfig.from_spec(
        "t", 2, 8, x_constraints=["lieb_oxford"],
        c_constraints=["non_negative_correlation"])
    model = AlecGGAModel.from_arch(arch, seed=0)
    assert [c.registry_name for c in model.x_constraints] == ["lieb_oxford"]
    assert [c.registry_name for c in model.c_constraints] == ["non_negative_correlation"]
    # The model's constraint view IS the networks' constraints.
    assert model.x_constraints is model.xnet.constraints
    assert model.c_constraints is model.cnet.constraints
