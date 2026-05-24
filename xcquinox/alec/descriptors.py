"""xcquinox.alec.descriptors — Descriptor ABC, registry, and concrete descriptors.

Implements THE SPEC §3: registry-driven descriptor composition for additional
network input features beyond (rho, sigma).
"""
import abc
import dataclasses

import equinox as eqx
import jax.numpy as jnp
from typing import ClassVar


class Descriptor(eqx.Module, abc.ABC):
    """Base class for all descriptors. Subclasses provide extra input features."""
    registry_name: ClassVar[str] = ""
    required_mol_keys: ClassVar[tuple[str, ...]] = ()
    n_features: int = eqx.field(static=True)

    @abc.abstractmethod
    def compute(self, mol_data: dict) -> jnp.ndarray:
        """Return descriptor features, shape (N, n_features) where N = grid size."""

    def describe(self) -> str:
        return f"{type(self).__name__}({self.registry_name}, n={self.n_features})"

    def __post_init__(self):
        _PRIMITIVE_TYPES = (int, float, bool, str)
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            ann = f.type
            if isinstance(ann, type) and issubclass(ann, _PRIMITIVE_TYPES):
                if isinstance(value, jnp.ndarray) or type(value).__module__.startswith("jax"):
                    raise TypeError(
                        f"{type(self).__name__}.{f.name} is declared as {ann.__name__} "
                        f"but received a jax.Array value; pass a plain Python {ann.__name__} instead."
                    )


DESCRIPTOR_REGISTRY: dict[str, type[Descriptor]] = {}


def register_descriptor(name: str):
    """Class decorator registering a Descriptor subclass under `name`."""
    def wrapper(cls):
        if name in DESCRIPTOR_REGISTRY:
            raise ValueError(f"Descriptor name {name!r} already registered")
        for f in dataclasses.fields(cls):
            if not f.metadata.get("static"):
                raise TypeError(
                    f"{cls.__name__}.{f.name} must be declared with "
                    f"eqx.field(..., static=True)"
                )
        cls.registry_name = name
        DESCRIPTOR_REGISTRY[name] = cls
        return cls
    return wrapper


def make_descriptor(name: str, **kwargs) -> Descriptor:
    """Look up DESCRIPTOR_REGISTRY[name] and instantiate with kwargs."""
    return DESCRIPTOR_REGISTRY[name](**kwargs)


def list_descriptors() -> list[str]:
    """Return sorted list of registered descriptor names."""
    return sorted(DESCRIPTOR_REGISTRY.keys())


@register_descriptor("cusp")
class CuspDescriptor(Descriptor):
    """Nuclear cusp proximity (2 features per grid point).

    Both features are bounded for network-friendly inputs:
      * Column 0 ``cusp_factor = exp(-2 Z_nearest r_min)`` ∈ [0, 1], where
        Z_nearest is the charge of the nearest nucleus and r_min is the
        distance to it. This is a heuristic proximity feature *motivated by*
        the Kato electron-nucleus cusp condition (Kato, *Commun. Pure Appl.
        Math.* **10**, 151 (1957)), whose exact statement is
        ``(∂⟨ρ⟩/∂r)|_{r=0} = -2Z·ρ(0)`` on the spherically-averaged density;
        the exponential decay here approximates the resulting Slater-like
        ``exp(-Z r)`` envelope rather than enforcing the condition exactly.
      * Column 1 ``tanh(log(Σ_A Z_A / r_A) / 5)`` ∈ (-1, 1). Encodes a
        scaled log of the total nuclear-attraction-like weight.

    See ``xcquinox.features.compute_cusp_descriptor`` for the precise
    definitions and the dynamic-range argument for the /5 scaling.
    """
    n_features: int = eqx.field(default=2, static=True)
    required_mol_keys: ClassVar[tuple[str, ...]] = ("cusp_features",)

    def compute(self, mol_data):
        return mol_data["cusp_features"]


@register_descriptor("dm_statistics")
class DMStatisticsDescriptor(Descriptor):
    """Density-matrix correlation indicators. 3 features.

    The 3 features (see ``xcquinox.features.compute_dm_features_array``) are,
    in order:
      0. ``idempotency_error`` — normalized Frobenius deviation from the
         single-determinant idempotency condition; ~0 for an HF/KS reference,
         growing with correlation.
      1. ``dm_entropy`` — fractional-occupation (idempotency-deviation) entropy
         of the natural-orbital occupations: per-spin-orbital binary entropy
         summed over natural orbitals. ~0 for a single determinant
         (occupations in {0, 2}) and larger for fractional natural occupations,
         so larger => more correlated. (DESC-07: this is *not* a size-dependent
         Shannon entropy of normalized occupations.)
      2. ``off_diag_norm`` — Frobenius norm of the off-diagonal AO density-matrix
         block, normalized by the trace.

    All three are designed so larger magnitude indicates stronger departure
    from a single Slater determinant.
    """
    n_features: int = eqx.field(default=3, static=True)
    required_mol_keys: ClassVar[tuple[str, ...]] = ("dm_features",)

    @staticmethod
    def compute_from_dm(dm: jnp.ndarray, s_matrix: jnp.ndarray,
                        n_grid: int) -> jnp.ndarray:
        """Pure kernel: compute 3-feature vector from (dm, S) and tile to grid.

        Mirrors the precompute path in data.py:229-234 but accepts a live DM
        so the SCF REASSEMBLE policy can recompute features per cycle.
        """
        from xcquinox.features import compute_dm_features_array
        global_features = compute_dm_features_array(dm, s_matrix)
        return jnp.tile(global_features, (n_grid, 1))

    def compute(self, mol_data):
        return mol_data["dm_features"]


def assemble_descriptor_features(descriptors: tuple[Descriptor, ...],
                                 mol_data: dict) -> jnp.ndarray:
    """Concatenate descriptor outputs left-to-right in declaration order."""
    if not descriptors:
        return jnp.zeros((mol_data["rho_grid"].shape[0], 0))
    return jnp.concatenate([d.compute(mol_data) for d in descriptors], axis=1)
