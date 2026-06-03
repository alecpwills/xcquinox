import warnings
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# 16 forward tests: 4 constraints x 4 test kinds (a/b/c/d).
# ---------------------------------------------------------------------------

# LiebOxfordBound (a), registry roundtrip
def test_lieb_oxford_registry_roundtrip():
    from xcquinox.alec.constraints import CONSTRAINT_REGISTRY, LiebOxfordBound, make_constraint
    assert CONSTRAINT_REGISTRY["lieb_oxford"] is LiebOxfordBound
    c = make_constraint("lieb_oxford")
    assert isinstance(c, LiebOxfordBound)
    assert c.mu == 1.804


# LiebOxfordBound (b), point evaluation
def test_lieb_oxford_point_evaluation():
    from xcquinox.alec.constraints import LiebOxfordBound
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
def test_lieb_oxford_lower_bound_is_zero_not_0p196():
    from xcquinox.alec.constraints import LiebOxfordBound
    c = LiebOxfordBound()

    # A strongly negative raw enhancement must be driven toward F_x = 0, the
    # physical floor, NOT clamped at the old symmetric-tanh artefact 0.196.
    def inner_very_negative(r, s, f):
        return -10.0 * jnp.ones_like(r)
    out = c(inner_very_negative, jnp.ones((3,)), jnp.ones((3,)), jnp.zeros((3, 0)))
    assert jnp.all(out > 0.0)            # never negative
    assert jnp.all(out < 0.05)           # below the old 0.196 floor, heading to 0
    # And the upper Lieb-Oxford ceiling (mu = 1.804) is still respected.
    def inner_large(r, s, f):
        return 100.0 * jnp.ones_like(r)
    hi = c(inner_large, jnp.ones((1,)), jnp.ones((1,)), jnp.zeros((1, 0)))
    assert jnp.all(hi <= 1.804 + 1e-9) and jnp.all(hi > 1.803)


# LiebOxfordBound (c), differentiability
def test_lieb_oxford_grad_finite():
    from xcquinox.alec.constraints import LiebOxfordBound
    c = LiebOxfordBound()

    def scalar_wrapped(x):
        def inner(r, s, f):
            return jnp.array([x])
        return c(inner, jnp.ones((1,)), jnp.ones((1,)), jnp.zeros((1, 0)))[0]

    g = jax.grad(scalar_wrapped)(0.5)
    assert jnp.isfinite(g)


# LiebOxfordBound (d), composition with trivial inner_fn
def test_lieb_oxford_composes_with_trivial_inner_fn():
    from xcquinox.alec.constraints import LiebOxfordBound
    c = LiebOxfordBound()

    def inner(r, s, f):
        return jnp.ones_like(r)
    out = c(inner, jnp.ones((3,)), jnp.ones((3,)), jnp.zeros((3, 0)))
    assert jnp.allclose(out, 1.0, atol=1e-12)


# UEGLimit (a), registry roundtrip
def test_ueg_limit_registry_roundtrip():
    from xcquinox.alec.constraints import CONSTRAINT_REGISTRY, UEGLimit, make_constraint
    assert CONSTRAINT_REGISTRY["ueg_limit"] is UEGLimit
    c = make_constraint("ueg_limit")
    assert isinstance(c, UEGLimit)
    assert c.damping == 1.0
    assert c.rho_eps == 1e-8


# UEGLimit (b), point evaluation
def test_ueg_limit_point_evaluation():
    from xcquinox.alec.constraints import UEGLimit
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
    from xcquinox.alec.constraints import UEGLimit
    c = UEGLimit()

    def scalar_wrapped(x):
        def inner(r, s, f):
            return jnp.array([x])
        return c(inner, jnp.ones((1,)), jnp.ones((1,)), jnp.zeros((1, 0)))[0]

    g = jax.grad(scalar_wrapped)(1.5)
    assert jnp.isfinite(g)


# UEGLimit (d), composition with trivial inner_fn
def test_ueg_limit_composes_with_trivial_inner_fn():
    from xcquinox.alec.constraints import UEGLimit
    c = UEGLimit()

    def inner(r, s, f):
        return 2.0 * jnp.ones_like(r)
    out = c(inner, jnp.ones((3,)), jnp.ones((3,)), jnp.zeros((3, 0)))
    assert jnp.all(out > 1.0)
    assert jnp.all(out < 2.0)


# NonNegativeCorrelation (a), registry roundtrip
def test_non_negative_correlation_registry_roundtrip():
    from xcquinox.alec.constraints import (
        CONSTRAINT_REGISTRY, NonNegativeCorrelation, make_constraint,
    )
    assert CONSTRAINT_REGISTRY["non_negative_correlation"] is NonNegativeCorrelation
    c = make_constraint("non_negative_correlation")
    assert isinstance(c, NonNegativeCorrelation)


# NonNegativeCorrelation (b), point evaluation
def test_non_negative_correlation_point_evaluation():
    from xcquinox.alec.constraints import NonNegativeCorrelation
    c = NonNegativeCorrelation()

    def inner_unit(r, s, f):
        return jnp.array([1.0, 1.0, 1.0])
    out_unit = c(inner_unit, jnp.ones((3,)), jnp.ones((3,)), jnp.zeros((3, 0)))
    assert jnp.allclose(out_unit, 1.0, atol=1e-12)


# NonNegativeCorrelation (c), differentiability
def test_non_negative_correlation_grad_finite():
    from xcquinox.alec.constraints import NonNegativeCorrelation
    c = NonNegativeCorrelation()

    def scalar_wrapped(x):
        def inner(r, s, f):
            return jnp.array([x])
        return c(inner, jnp.ones((1,)), jnp.ones((1,)), jnp.zeros((1, 0)))[0]

    g = jax.grad(scalar_wrapped)(-2.0)
    assert jnp.isfinite(g)


# NonNegativeCorrelation (d), composition + asymptotic / fixed-point checks
def test_non_negative_correlation_composes_with_trivial_inner_fn():
    """The corrected NonNegativeCorrelation uses
    `softplus(F_raw - 1 + log(e - 1))` so that:
      * F_raw =  1 -> F_c = 1 (PBE fixed point preserved)
      * F_raw = -10 -> F_c -> 0 (Levy-Perdew non-positive correlation:
        E_c = ε_c^LDA · F_c ≤ 0 with ε_c^LDA ≤ 0 demands F_c ≥ 0)
      * monotone increasing in F_raw
    """
    from xcquinox.alec.constraints import NonNegativeCorrelation
    c = NonNegativeCorrelation()

    def inner_raw(F_raw_value):
        return lambda r, s, f: F_raw_value * jnp.ones_like(r)

    # Floor at 0 (not 1 - log 2): F_raw = -10 should give a value
    # very close to 0 (≪ 1e-3), the prior broken implementation
    # asymptoted at 1 - log(2) ≈ 0.307.
    out_floor = c(inner_raw(-10.0), jnp.ones((3,)), jnp.ones((3,)), jnp.zeros((3, 0)))
    assert jnp.all(out_floor >= 0.0), out_floor
    assert jnp.all(out_floor < 1e-3), (
        f"floor must be near zero (Levy-Perdew F_c >= 0); got {out_floor}"
    )

    # Fixed point: F_raw = 1 -> F_c = 1 (PBE preserved).
    out_fixed = c(inner_raw(1.0), jnp.ones((3,)), jnp.ones((3,)), jnp.zeros((3, 0)))
    assert jnp.allclose(out_fixed, 1.0, atol=1e-5), out_fixed

    # Monotone: F_raw = 5 should give a value > F_raw = 1.
    out_high = c(inner_raw(5.0), jnp.ones((3,)), jnp.ones((3,)), jnp.zeros((3, 0)))
    assert jnp.all(out_high > out_fixed), (out_high, out_fixed)


# ScalingSymmetric (a), registry roundtrip
def test_scaling_symmetric_registry_roundtrip():
    from xcquinox.alec.constraints import CONSTRAINT_REGISTRY, ScalingSymmetric, make_constraint
    assert CONSTRAINT_REGISTRY["scaling_symmetric"] is ScalingSymmetric
    c = make_constraint("scaling_symmetric")
    assert isinstance(c, ScalingSymmetric)
    assert c.rho_ref == 1.0
    assert c.rho_eps == 1e-8


# ScalingSymmetric (b), point evaluation
def test_scaling_symmetric_point_evaluation():
    from xcquinox.alec.constraints import ScalingSymmetric
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
    from xcquinox.alec.constraints import ScalingSymmetric
    c = ScalingSymmetric()

    def scalar_wrapped(x):
        def inner(r, s, f):
            return jnp.array([x * r[0]])
        return c(inner, jnp.ones((1,)), jnp.ones((1,)), jnp.zeros((1, 0)))[0]

    g = jax.grad(scalar_wrapped)(0.5)
    assert jnp.isfinite(g)


# ScalingSymmetric (d), composition with trivial inner_fn
def test_scaling_symmetric_composes_with_trivial_inner_fn():
    from xcquinox.alec.constraints import ScalingSymmetric
    c = ScalingSymmetric()

    def inner(r, s, f):
        return 1.0 + 0.0 * r
    out = c(inner, jnp.array([0.5, 2.0]), jnp.array([0.3, 1.2]), jnp.zeros((2, 0)))
    assert jnp.allclose(out, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 24 additional tests
# ---------------------------------------------------------------------------

# (i): _compose_constraints empty tuple returns base fn unchanged
def test_compose_constraints_empty_tuple_returns_base_fn():
    from xcquinox.alec.constraints import _compose_constraints

    def base(r, s, f):
        return 3.14 * jnp.ones_like(r)

    composed = _compose_constraints(base, ())
    out = composed(jnp.ones((2,)), jnp.ones((2,)), jnp.zeros((2, 0)))
    assert jnp.allclose(out, 3.14, atol=1e-12)


# (ii): composition order matches innermost->outermost semantics
def test_compose_constraints_innermost_outermost_order():
    from xcquinox.alec.constraints import Constraint, _compose_constraints

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
    from xcquinox.alec.constraints import (
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
    from xcquinox.alec.constraints import LiebOxfordBound
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
def test_ueg_limit_at_sigma_zero_returns_one():
    from xcquinox.alec.constraints import UEGLimit
    c = UEGLimit(damping=1.0)

    def inner(r, s, f):
        return 2.5 * jnp.ones_like(r)

    out = c(inner, jnp.ones((3,)), jnp.zeros((3,)), jnp.zeros((3, 0)))
    assert jnp.allclose(out, 1.0, atol=1e-14)


# (vi): NonNegativeCorrelation strictly positive
def test_non_negative_correlation_strictly_positive():
    from xcquinox.alec.constraints import NonNegativeCorrelation
    c = NonNegativeCorrelation()
    probe = jnp.array([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])

    def inner(r, s, f):
        return probe

    out = c(inner, jnp.ones_like(probe), jnp.ones_like(probe),
            jnp.zeros((probe.shape[0], 0)))
    assert jnp.all(out > 0.0)


# (vii): ScalingSymmetric invariance for (rho, sigma)-only features
def test_scaling_symmetric_invariance_rho_sigma_only():
    from xcquinox.alec.constraints import ScalingSymmetric
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
def test_scaling_symmetric_partial_invariance_with_rho_dependent_descriptors():
    from xcquinox.alec.constraints import ScalingSymmetric
    c = ScalingSymmetric(rho_ref=1.0)

    def inner(r, s, f):
        return r * s + f[:, 0]

    rho = jnp.array([2.0, 2.0])
    sigma = jnp.array([1.0, 1.0])
    feats_base = jnp.array([[0.1], [0.1]])
    feats_scaled = jnp.array([[0.4], [0.4]])
    out_base = c(inner, rho, sigma, feats_base)
    out_scaled = c(inner, rho, sigma, feats_scaled)
    assert not jnp.allclose(out_base, out_scaled, atol=1e-6)


# (ix): make_constraint raises KeyError on unknown name
def test_make_constraint_unknown_raises_key_error():
    from xcquinox.alec.constraints import make_constraint
    with pytest.raises(KeyError):
        make_constraint("does_not_exist")


# (x): CONSTRAINT_REGISTRY contains all 4 built-ins
def test_constraint_registry_has_four_builtins():
    from xcquinox.alec.constraints import CONSTRAINT_REGISTRY
    expected = {"lieb_oxford", "ueg_limit", "non_negative_correlation", "scaling_symmetric"}
    assert expected.issubset(set(CONSTRAINT_REGISTRY.keys()))


# (xi): list_constraints returns sorted list
def test_list_constraints_sorted():
    from xcquinox.alec.constraints import list_constraints
    names = list_constraints()
    assert names == sorted(names)
    assert "lieb_oxford" in names
    assert "ueg_limit" in names
    assert "non_negative_correlation" in names
    assert "scaling_symmetric" in names


# (xii): Constraint.violation returns zero for identity-preserving inputs
def test_constraint_violation_zero_for_identity_preserving():
    from xcquinox.alec.constraints import LiebOxfordBound
    c = LiebOxfordBound()

    def inner(r, s, f):
        return jnp.ones_like(r)

    v = c.violation(inner, jnp.ones((3,)), jnp.ones((3,)), jnp.zeros((3, 0)))
    assert jnp.allclose(v, 0.0, atol=1e-14)


# (xiii): Constraint.is_satisfied returns True/False correctly
def test_constraint_is_satisfied_true_false():
    from xcquinox.alec.constraints import LiebOxfordBound
    c = LiebOxfordBound()

    def inner_ok(r, s, f):
        return jnp.ones_like(r)

    def inner_bad(r, s, f):
        return 10.0 * jnp.ones_like(r)

    assert c.is_satisfied(inner_ok, jnp.ones((3,)), jnp.ones((3,)), jnp.zeros((3, 0))) is True
    assert c.is_satisfied(inner_bad, jnp.ones((3,)), jnp.ones((3,)), jnp.zeros((3, 0))) is False


# (xiv): Constraint.describe formatting
def test_constraint_describe_formatting():
    from xcquinox.alec.constraints import LiebOxfordBound, UEGLimit
    assert LiebOxfordBound().describe() == "LiebOxfordBound(lieb_oxford)"
    assert UEGLimit().describe() == "UEGLimit(ueg_limit)"


# (xv): constraint_report aggregates per-constraint stats
def test_constraint_report_aggregates_per_constraint_stats():
    from xcquinox.alec.config import ArchitectureConfig
    from xcquinox.alec.models import AlecGGAModel
    arch = ArchitectureConfig.from_spec(
        "test_cr", 2, 8,
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


# (xvi): D-H1, constructing a constraint with a jax.Array scalar raises
@pytest.mark.parametrize(
    "ctor,kwargs,offender",
    [
        ("LiebOxfordBound", {"mu": jnp.array(1.804)}, "mu"),
        ("UEGLimit", {"damping": jnp.array(1.0)}, "damping"),
        ("ScalingSymmetric", {"rho_ref": jnp.array(1.0)}, "rho_ref"),
    ],
)
def test_constraint_rejects_jax_scalars(ctor, kwargs, offender):
    from xcquinox.alec import constraints as _constraints_module
    klass = getattr(_constraints_module, ctor)
    with pytest.raises(TypeError, match=offender):
        klass(**kwargs)


# (xvii): D-H3, registering a subclass with a non-static trainable field raises
def test_constraint_register_rejects_trainable_field():
    from xcquinox.alec.constraints import (
        CONSTRAINT_REGISTRY, Constraint, register_constraint,
    )
    try:
        with pytest.raises(TypeError, match="static"):
            @register_constraint("bad")
            class BadConstraint(Constraint):
                trainable: jnp.ndarray = eqx.field(default_factory=lambda: jnp.zeros(3))
                def __call__(self, inner_fn, rho, sigma, features):
                    return inner_fn(rho, sigma, features)
    finally:
        CONSTRAINT_REGISTRY.pop("bad", None)


# (xviii): H-E12-5, no double-clamp when LOB is registered under x_constraints
def test_lieb_oxford_no_double_clamp():
    from xcquinox.alec.config import ArchitectureConfig
    from xcquinox.alec.models import AlecGGAModel
    arch = ArchitectureConfig.from_spec(
        "test_no_dclamp", 2, 8, x_constraints=["lieb_oxford"],
    )
    assert arch.resolved_xnet_lob_lim is None
    model = AlecGGAModel.from_arch(arch, seed=0)
    assert model.xnet.lob_lim is None
    F_raw_grid = jnp.linspace(-5.0, 5.0, 21)
    from xcquinox.alec.constraints import LiebOxfordBound
    c = LiebOxfordBound()
    def inner(r, s, f):
        return F_raw_grid
    constraint_only = c(inner, jnp.ones_like(F_raw_grid), jnp.ones_like(F_raw_grid),
                        jnp.zeros((F_raw_grid.shape[0], 0)))
    # REFPHYS-02: physical floor is 0 (not the old symmetric-tanh artefact 0.196).
    assert jnp.all(constraint_only >= -1e-10)
    assert jnp.all(constraint_only <= 1.804 + 1e-10)


# (xix): H-E12-6, opt-in double clamp narrows F range
def test_lieb_oxford_opt_in_double_clamp():
    from xcquinox.alec.config import ArchitectureConfig
    from xcquinox.alec.models import AlecGGAModel
    with pytest.warns(RuntimeWarning):
        arch = ArchitectureConfig.from_spec(
            "test_opt_dclamp", 2, 8,
            x_constraints=["lieb_oxford"],
            allow_double_lob_clamp=True,
        )
    assert arch.resolved_xnet_lob_lim == 1.804
    assert arch.double_lob_clamp_allowed is True
    model = AlecGGAModel.from_arch(arch, seed=0)
    assert model.xnet.lob_lim == 1.804


# (xx): H-E12-7, UEGLimit's internal s^2 matches the network's KS formula
def test_ueg_limit_matches_network_s():
    from xcquinox.alec.constraints import UEGLimit
    damping = 1.0
    c = UEGLimit(damping=damping)
    rho = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])
    sigma = jnp.array([0.01, 0.1, 1.0, 10.0, 100.0])

    def inner(r, s, f):
        return 2.0 * jnp.ones_like(r)

    out = c(inner, rho, sigma, jnp.zeros((5, 0)))
    gate = (out - 1.0) / 1.0
    s2_recovered = -jnp.log(1.0 - gate) / damping

    rho_safe = jnp.maximum(rho, 1e-8)
    k_F = (3.0 * jnp.pi ** 2 * rho_safe) ** (1.0 / 3.0)
    s2_ks = (jnp.sqrt(sigma) / (2.0 * k_F * rho_safe)) ** 2
    assert jnp.allclose(s2_recovered, s2_ks, atol=1e-13)


# (xxi): H-E12-8, ScalingSymmetric on c_constraints raises by default
def test_scaling_symmetric_c_raises():
    import xcquinox.alec.constraints  # noqa: F401
    from xcquinox.alec.config import ArchitectureConfig
    with pytest.raises(ValueError, match="ScalingSymmetric|scaling_symmetric"):
        ArchitectureConfig.from_spec(
            "test_ssc", 2, 8,
            c_constraints=["scaling_symmetric"],
        )


# (xxii): H-E12-9, allow_scaling_symmetric_on_c=True emits RuntimeWarning
def test_scaling_symmetric_c_allow_override():
    import xcquinox.alec.constraints  # noqa: F401
    from xcquinox.alec.config import ArchitectureConfig
    with pytest.warns(RuntimeWarning, match="correlation|rs"):
        arch = ArchitectureConfig.from_spec(
            "test_ssc_ok", 2, 8,
            c_constraints=["scaling_symmetric"],
            allow_scaling_symmetric_on_c=True,
        )
    assert any(s.name == "scaling_symmetric" for s in arch.c_constraints)


# (xxiii): REFPHYS-02, LiebOxfordBound lower asymptote is the physical 0.
def test_lieb_oxford_lower_asymptote_is_zero():
    from xcquinox.alec.constraints import LiebOxfordBound
    c = LiebOxfordBound(mu=1.804)

    def inner(r, s, f):
        return -100.0 * jnp.ones_like(r)

    out = c(inner, jnp.ones((1,)), jnp.ones((1,)), jnp.zeros((1, 0)))
    # One-sided I_mu squash (Dick 2021 eq. 11) floors F_x at 0, the physical
    # bound (eps_x = eps_x^LDA * F_x <= 0). The old symmetric tanh wrongly
    # asymptoted to 2 - mu = 0.196. (At this extreme the floor is reached to
    # machine precision, so 0 is inclusive.)
    assert jnp.all(out >= 0.0)
    assert jnp.abs(out[0]) < 1e-10


# (xxiv): REFPHYS-02, linear response near the UEG fixed point has slope
# (mu-1)/mu, matching the in-network _AlecLOB squash (I_mu is algebraically
# identical to limit*sigmoid(x-log(limit-1))-1). The previous symmetric tanh
# had unit slope, which did NOT match the production network squash.
def test_lieb_oxford_linear_response_near_ueg():
    from xcquinox.alec.constraints import LiebOxfordBound
    mu = 1.804
    c = LiebOxfordBound(mu=mu)
    slope = (mu - 1.0) / mu  # I_mu'(0)

    for eps in (1e-6, 1e-4, 1e-2):
        def inner(r, s, f, _eps=eps):
            return (1.0 + _eps) * jnp.ones_like(r)
        out = c(inner, jnp.ones((1,)), jnp.ones((1,)), jnp.zeros((1, 0)))
        # F(1)=1 exactly; first-order response is slope*eps, second order O(eps^2).
        assert jnp.abs(out[0] - (1.0 + slope * eps)) < 10.0 * eps ** 2 + 1e-15
