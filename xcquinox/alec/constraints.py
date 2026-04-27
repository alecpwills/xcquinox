"""xcquinox.alec.constraints — Constraint ABC, registry, and 4 concrete constraints.

Implements THE SPEC §4: registry-driven constraint composition for enforcing
physical properties on network enhancement factors.
"""
import abc
import dataclasses
from typing import Callable, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp


CONSTRAINT_REGISTRY: dict[str, type["Constraint"]] = {}


def register_constraint(name: str):
    """Class decorator registering a Constraint subclass under `name`."""
    def _decorator(cls):
        if name in CONSTRAINT_REGISTRY:
            raise ValueError(f"Constraint {name!r} is already registered in CONSTRAINT_REGISTRY")
        for f in dataclasses.fields(cls):
            if not f.metadata.get("static", False):
                raise TypeError(
                    f"Constraint subclass {cls.__name__} has non-static instance field "
                    f"{f.name!r}. All Constraint fields must be declared with "
                    f"eqx.field(static=True) so the instance is a pure pytree leaf "
                    f"with zero traced data. See §4.2 register_constraint docstring."
                )
        cls.registry_name = name
        CONSTRAINT_REGISTRY[name] = cls
        return cls
    return _decorator


def make_constraint(name: str, **kwargs) -> "Constraint":
    """Look up CONSTRAINT_REGISTRY[name] and instantiate with kwargs."""
    return CONSTRAINT_REGISTRY[name](**kwargs)


def list_constraints() -> list[str]:
    """Return sorted list of registered constraint names."""
    return sorted(CONSTRAINT_REGISTRY.keys())


class Constraint(eqx.Module, abc.ABC):
    """Wraps a network callable (rho, sigma, features) -> F to enforce a
    physical property. Differentiable, jit-safe, composable."""
    registry_name: ClassVar[str] = ""

    @abc.abstractmethod
    def __call__(self, inner_fn: Callable, rho, sigma, features) -> jnp.ndarray:
        """Return the constrained F."""

    def violation(self, inner_fn, rho, sigma, features) -> jnp.ndarray:
        F_raw = inner_fn(rho, sigma, features)
        F_con = self(inner_fn, rho, sigma, features)
        return jnp.abs(F_raw - F_con)

    def is_satisfied(self, inner_fn, rho, sigma, features, tol=1e-6) -> bool:
        return bool(jnp.all(self.violation(inner_fn, rho, sigma, features) < tol))

    def describe(self) -> str:
        return f"{type(self).__name__}({self.registry_name})"


@register_constraint("lieb_oxford")
class LiebOxfordBound(Constraint):
    """Smooth two-sided clamp on the exchange enhancement factor.

    Functional form ``F = 1 + (mu - 1) * tanh((F_raw - 1)/(mu - 1))``,
    range ``[2 - mu, mu] = [0.196, 1.804]`` for the default mu=1.804.

    The upper bound mu = 1 + kappa is the Lieb-Oxford bound on F_x
    (Lieb & Oxford, *Int. J. Quantum Chem.* **19**, 427 (1981); PBE
    convention with kappa = 0.804 — Perdew, Burke, Ernzerhof, *Phys.
    Rev. Lett.* **77**, 3865 (1996), §3 eq. (14)).

    The incidental lower bound 2 - mu = 0.196 is a side effect of the
    symmetric tanh smoothing. It is physically reasonable because F_x
    must be non-negative (ε_x^total = ε_x^LDA · F_x ≤ 0 with
    ε_x^LDA ≤ 0 demands F_x ≥ 0); 0.196 is well above 0.
    """
    registry_name: ClassVar[str] = "lieb_oxford"
    mu: float = eqx.field(default=1.804, static=True)

    def __post_init__(self):
        if not isinstance(self.mu, (int, float)) or isinstance(self.mu, bool):
            raise TypeError(
                f"LiebOxfordBound.mu must be a plain Python int/float (static field), "
                f"got {type(self.mu).__name__}. JAX arrays and numpy scalars are rejected "
                f"to avoid static cache key hashing errors inside eqx.filter_jit."
            )
        if self.mu <= 1.0:
            raise ValueError(
                f"LiebOxfordBound requires mu > 1 (the Lieb-Oxford constant is 1.804); "
                f"got mu={self.mu}."
            )

    def __call__(self, inner_fn, rho, sigma, features):
        F_raw = inner_fn(rho, sigma, features)
        max_delta = self.mu - 1.0
        return 1.0 + max_delta * jnp.tanh((F_raw - 1.0) / max_delta)


@register_constraint("ueg_limit")
class UEGLimit(Constraint):
    """Damped (F - 1) at small reduced gradient s, enforcing F -> 1 as sigma -> 0."""
    registry_name: ClassVar[str] = "ueg_limit"
    damping: float = eqx.field(default=1.0, static=True)
    rho_eps: float = eqx.field(default=1e-8, static=True)

    def __post_init__(self):
        for name, val in (("damping", self.damping), ("rho_eps", self.rho_eps)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(
                    f"UEGLimit.{name} must be a plain Python int/float (static field), "
                    f"got {type(val).__name__}."
                )
        if self.damping <= 0.0:
            raise ValueError(f"UEGLimit requires damping > 0, got {self.damping}")
        if self.rho_eps <= 0.0:
            raise ValueError(f"UEGLimit requires rho_eps > 0, got {self.rho_eps}")

    def __call__(self, inner_fn, rho, sigma, features):
        F_raw = inner_fn(rho, sigma, features)
        rho_safe = jnp.maximum(rho, self.rho_eps)
        k_F = (3.0 * jnp.pi ** 2 * rho_safe) ** (1.0 / 3.0)
        s_net = jnp.sqrt(jnp.maximum(sigma, 0.0)) / (2.0 * k_F * rho_safe)
        s2 = s_net ** 2
        gate = 1.0 - jnp.exp(-self.damping * s2)
        return 1.0 + (F_raw - 1.0) * gate


@register_constraint("non_negative_correlation")
class NonNegativeCorrelation(Constraint):
    """Softplus-clamped Fc, enforcing Fc >= 0 with PBE fixed point F=1.

    The shifted-softplus form ``softplus(x + log(e - 1))`` satisfies
    f(0) = log(1 + e^{log(e-1)}) = log(1 + (e-1)) = log(e) = 1. After
    subtracting 1 and re-adding 1 we get a function g with g(1) = 1 and
    g(F_raw → -∞) → 0 (true zero, not 1 - log 2 ≈ 0.307).

    This enforces the Levy-Perdew non-positive correlation energy
    constraint (E_c[ρ] ≤ 0): since ε_c^LDA(ρ) ≤ 0 for all ρ in the
    PW92 parametrization (Perdew & Wang 1992 Phys. Rev. B 45 13244),
    F_c ≥ 0 ensures ε_c^total = ε_c^LDA · F_c ≤ 0 pointwise.
    """
    registry_name: ClassVar[str] = "non_negative_correlation"

    def __call__(self, inner_fn, rho, sigma, features):
        F_raw = inner_fn(rho, sigma, features)
        # softplus(x + log(e - 1)) maps 0 -> 1 and -∞ -> 0, so the +1
        # offset preserves PBE fixed point and the floor is exactly 0.
        shift = jnp.log(jnp.expm1(1.0))  # log(e - 1)
        return jax.nn.softplus((F_raw - 1.0) + shift)


@register_constraint("scaling_symmetric")
class ScalingSymmetric(Constraint):
    """Enforce uniform-coordinate scaling symmetry by evaluating at fixed rho_ref."""
    registry_name: ClassVar[str] = "scaling_symmetric"
    rho_ref: float = eqx.field(default=1.0, static=True)
    rho_eps: float = eqx.field(default=1e-8, static=True)

    def __post_init__(self):
        for name, val in (("rho_ref", self.rho_ref), ("rho_eps", self.rho_eps)):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(
                    f"ScalingSymmetric.{name} must be a plain Python int/float (static field), "
                    f"got {type(val).__name__}."
                )
        if self.rho_ref <= 0.0:
            raise ValueError(f"ScalingSymmetric requires rho_ref > 0, got {self.rho_ref}")
        if self.rho_eps <= 0.0:
            raise ValueError(f"ScalingSymmetric requires rho_eps > 0, got {self.rho_eps}")

    def __call__(self, inner_fn, rho, sigma, features):
        s2 = sigma / (jnp.maximum(rho, self.rho_eps) ** (8.0 / 3.0))
        rho_new = jnp.full_like(rho, self.rho_ref)
        sigma_new = s2 * (self.rho_ref ** (8.0 / 3.0))
        return inner_fn(rho_new, sigma_new, features)


def _compose_constraints(base_fn, constraints):
    """constraints[0] is innermost (wraps base_fn directly).
    constraints[-1] is outermost (applied last)."""
    current = base_fn
    for c in constraints:
        prev = current
        def wrapped(r, s, f, _c=c, _prev=prev):
            return _c(_prev, r, s, f)
        current = wrapped
    return current
