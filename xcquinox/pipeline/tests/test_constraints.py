from typing import ClassVar

import jax
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# 16 forward tests: 4 constraints x 4 test kinds (a/b/c/d).
# ---------------------------------------------------------------------------

# LiebOxfordBound (a), registry roundtrip


# LiebOxfordBound (b), point evaluation
def test_lieb_oxford_point_evaluation():
    from xcquinox.pipeline.constraints import LiebOxfordBound
    c = LiebOxfordBound()
    rho = jnp.array([1.0, 1.0, 1.0])
    sigma = jnp.array([1.0, 1.0, 1.0])
    feats = jnp.zeros((3, 0))

    def inner_unit(r, s, f):
        return jnp.array([1.0, 1.0, 1.0])
    out_unit = c(inner_unit, rho, sigma, feats)
    assert jnp.allclose(out_unit, 1.0, atol=1e-12)

    def inner_varied(r, s, f):
        return jnp.array([2.0, 0.5, 1.5])
    out = c(inner_varied, rho, sigma, feats)
    # REFPHYS-02: one-sided I_mu transform (Dick 2021 eq. 11), not symmetric tanh.
    mu = 1.804
    F_raw = jnp.array([2.0, 0.5, 1.5])
    expected = 1.0 + (mu / (1.0 + (mu - 1.0) * jnp.exp(-(F_raw - 1.0))) - 1.0)
    assert jnp.allclose(out, expected, atol=1e-12)


# LiebOxfordBound (b2), REFPHYS-02: lower bound is the physical 0, not 0.196.


# LiebOxfordBound (c), differentiability
def test_lieb_oxford_grad_finite():
    from xcquinox.pipeline.constraints import LiebOxfordBound
    c = LiebOxfordBound()

    def scalar_wrapped(x):
        def inner(r, s, f):
            return jnp.array([x])
        return c(inner, jnp.ones((1,)), jnp.ones((1,)), jnp.zeros((1, 0)))[0]

    g = jax.grad(scalar_wrapped)(0.5)
    assert jnp.isfinite(g)


# LiebOxfordBound (d), composition with trivial inner_fn
def test_lieb_oxford_composes_with_trivial_inner_fn():
    from xcquinox.pipeline.constraints import LiebOxfordBound
    c = LiebOxfordBound()

    def inner(r, s, f):
        return jnp.ones_like(r)
    out = c(inner, jnp.ones((3,)), jnp.ones((3,)), jnp.zeros((3, 0)))
    assert jnp.allclose(out, 1.0, atol=1e-12)


# UEGLimit (a), registry roundtrip


# UEGLimit (b), point evaluation
def test_ueg_limit_point_evaluation():
    from xcquinox.pipeline.constraints import UEGLimit
    c = UEGLimit(damping=1.0)
    rho = jnp.array([1.0, 1.0])
    sigma = jnp.array([1.0, 1.0])
    feats = jnp.zeros((2, 0))

    def inner(r, s, f):
        return jnp.array([2.0, 0.5])

    out = c(inner, rho, sigma, feats)
    rho_safe = jnp.maximum(rho, 1e-8)
    k_F = (3.0 * jnp.pi ** 2 * rho_safe) ** (1.0 / 3.0)
    s2 = (jnp.sqrt(sigma) / (2.0 * k_F * rho_safe)) ** 2
    gate = 1.0 - jnp.exp(-s2)
    expected = 1.0 + (jnp.array([2.0, 0.5]) - 1.0) * gate
    assert jnp.allclose(out, expected, atol=1e-12)


# UEGLimit (c), differentiability
def test_ueg_limit_grad_finite():
    from xcquinox.pipeline.constraints import UEGLimit
    c = UEGLimit()

    def scalar_wrapped(x):
        def inner(r, s, f):
            return jnp.array([x])
        return c(inner, jnp.ones((1,)), jnp.ones((1,)), jnp.zeros((1, 0)))[0]

    g = jax.grad(scalar_wrapped)(1.5)
    assert jnp.isfinite(g)


# UEGLimit (d), composition with trivial inner_fn


# NonNegativeCorrelation (a), registry roundtrip


# NonNegativeCorrelation (b), point evaluation
def test_non_negative_correlation_point_evaluation():
    from xcquinox.pipeline.constraints import NonNegativeCorrelation
    c = NonNegativeCorrelation()

    def inner_unit(r, s, f):
        return jnp.array([1.0, 1.0, 1.0])
    out_unit = c(inner_unit, jnp.ones((3,)), jnp.ones((3,)), jnp.zeros((3, 0)))
    assert jnp.allclose(out_unit, 1.0, atol=1e-12)


# NonNegativeCorrelation (c), differentiability


# NonNegativeCorrelation (d), composition + asymptotic / fixed-point checks


# ScalingSymmetric (a), registry roundtrip


# ScalingSymmetric (b), point evaluation
def test_scaling_symmetric_point_evaluation():
    from xcquinox.pipeline.constraints import ScalingSymmetric
    c = ScalingSymmetric(rho_ref=1.0)

    captured = {}

    def inner(r, s, f):
        captured["rho"] = r
        captured["sigma"] = s
        return jnp.ones_like(r)

    rho = jnp.array([2.0, 2.0])
    sigma = jnp.array([1.0, 1.0])
    c(inner, rho, sigma, jnp.zeros((2, 0)))
    expected_rho = jnp.ones_like(rho)
    expected_s2 = sigma / (rho ** (8.0 / 3.0))
    expected_sigma = expected_s2 * (1.0 ** (8.0 / 3.0))
    assert jnp.allclose(captured["rho"], expected_rho, atol=1e-12)
    assert jnp.allclose(captured["sigma"], expected_sigma, atol=1e-12)


# ScalingSymmetric (c), differentiability
def test_scaling_symmetric_grad_finite():
    from xcquinox.pipeline.constraints import ScalingSymmetric
    c = ScalingSymmetric()

    def scalar_wrapped(x):
        def inner(r, s, f):
            return jnp.array([x * r[0]])
        return c(inner, jnp.ones((1,)), jnp.ones((1,)), jnp.zeros((1, 0)))[0]

    g = jax.grad(scalar_wrapped)(0.5)
    assert jnp.isfinite(g)


# ScalingSymmetric (d), composition with trivial inner_fn


# ---------------------------------------------------------------------------
# 24 additional tests
# ---------------------------------------------------------------------------

# (i): _compose_constraints empty tuple returns base fn unchanged


# (ii): composition order matches innermost->outermost semantics
def test_compose_constraints_innermost_outermost_order():
    from xcquinox.pipeline.constraints import Constraint, _compose_constraints

    call_log = []

    class _TagA(Constraint):
        registry_name: ClassVar[str] = ""
        def __call__(self, inner_fn, rho, sigma, features):
            call_log.append("A")
            return inner_fn(rho, sigma, features)

    class _TagB(Constraint):
        registry_name: ClassVar[str] = ""
        def __call__(self, inner_fn, rho, sigma, features):
            call_log.append("B")
            return inner_fn(rho, sigma, features)

    def base(r, s, f):
        call_log.append("base")
        return jnp.ones_like(r)

    composed = _compose_constraints(base, (_TagA(), _TagB()))
    composed(jnp.ones((1,)), jnp.ones((1,)), jnp.zeros((1, 0)))
    assert call_log == ["B", "A", "base"]


# (iii): differentiability through composed chain of all 4 constraints
def test_compose_four_constraint_chain_jax_grad_finite():
    from xcquinox.pipeline.constraints import (
        LiebOxfordBound, NonNegativeCorrelation, ScalingSymmetric, UEGLimit,
        _compose_constraints,
    )

    def scalar_wrapped(x):
        def base(r, s, f):
            return x * jnp.ones_like(r)
        chain = _compose_constraints(
            base,
            (LiebOxfordBound(), UEGLimit(), NonNegativeCorrelation(), ScalingSymmetric()),
        )
        return chain(jnp.ones((1,)), jnp.ones((1,)), jnp.zeros((1, 0)))[0]

    g = jax.grad(scalar_wrapped)(0.7)
    assert jnp.isfinite(g)


# (iv): LiebOxfordBound UEG fixed point + saturation
def test_lieb_oxford_preserves_ueg_fixed_point_and_saturates():
    from xcquinox.pipeline.constraints import LiebOxfordBound
    c = LiebOxfordBound(mu=1.804)

    def inner_one(r, s, f):
        return jnp.ones_like(r)
    out1 = c(inner_one, jnp.ones((1,)), jnp.ones((1,)), jnp.zeros((1, 0)))
    assert jnp.allclose(out1, 1.0, atol=1e-14)

    def inner_100(r, s, f):
        return 100.0 * jnp.ones_like(r)
    out100 = c(inner_100, jnp.ones((1,)), jnp.ones((1,)), jnp.zeros((1, 0)))
    assert jnp.abs(out100 - 1.804) < 1e-10


# (v): UEGLimit at sigma=0 returns exactly F=1


# (vi): NonNegativeCorrelation strictly positive


# (vii): ScalingSymmetric invariance for (rho, sigma)-only features
def test_scaling_symmetric_invariance_rho_sigma_only():
    from xcquinox.pipeline.constraints import ScalingSymmetric
    c = ScalingSymmetric(rho_ref=1.0)

    def inner(r, s, f):
        return r * s

    lam = 5.0
    feats = jnp.zeros((2, 0))
    out_base = c(inner, jnp.array([2.0, 2.0]), jnp.array([1.0, 1.0]), feats)
    out_scaled = c(inner, lam * jnp.array([2.0, 2.0]),
                   (lam ** (8 / 3)) * jnp.array([1.0, 1.0]), feats)
    assert jnp.allclose(out_base, out_scaled, atol=1e-10)


# (viii): ScalingSymmetric partial invariance with rho-dependent descriptors


# (ix): make_constraint raises KeyError on unknown name
def test_make_constraint_unknown_raises_key_error():
    from xcquinox.pipeline.constraints import make_constraint
    with pytest.raises(KeyError):
        make_constraint("does_not_exist")


# (x): CONSTRAINT_REGISTRY contains all 4 built-ins


# (xi): list_constraints returns sorted list


# (xii): Constraint.violation returns zero for identity-preserving inputs


# (xiii): Constraint.is_satisfied returns True/False correctly


# (xiv): Constraint.describe formatting


# (xv): constraint_report aggregates per-constraint stats


# (xvi): D-H1, constructing a constraint with a jax.Array scalar raises


# (xvii): D-H3, registering a subclass with a non-static trainable field raises


# (xviii): H-E12-5, no double-clamp when LOB is registered under x_constraints


# (xix): H-E12-6, opt-in double clamp narrows F range


# (xx): H-E12-7, UEGLimit's internal s^2 matches the network's KS formula


# (xxi): H-E12-8, ScalingSymmetric on c_constraints raises by default


# (xxii): H-E12-9, allow_scaling_symmetric_on_c=True emits RuntimeWarning


# (xxiii): REFPHYS-02, linear response near the UEG fixed point has slope
# (mu-1)/mu, matching the in-network _AlecLOB squash (I_mu is algebraically
# identical to limit*sigmoid(x-log(limit-1))-1). The previous symmetric tanh
# had unit slope, which did NOT match the production network squash.
def test_lieb_oxford_linear_response_near_ueg():
    from xcquinox.pipeline.constraints import LiebOxfordBound
    mu = 1.804
    c = LiebOxfordBound(mu=mu)
    slope = (mu - 1.0) / mu  # I_mu'(0)

    for eps in (1e-6, 1e-4, 1e-2):
        def inner(r, s, f, _eps=eps):
            return (1.0 + _eps) * jnp.ones_like(r)
        out = c(inner, jnp.ones((1,)), jnp.ones((1,)), jnp.zeros((1, 0)))
        # F(1)=1 exactly; first-order response is slope*eps, second order O(eps^2).
        assert jnp.abs(out[0] - (1.0 + slope * eps)) < 10.0 * eps ** 2 + 1e-15
