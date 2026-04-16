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
    """Nuclear cusp proximity. 2 features: proximity weight, log distance."""
    n_features: int = eqx.field(default=2, static=True)
    required_mol_keys: ClassVar[tuple[str, ...]] = ("cusp_features",)

    def compute(self, mol_data):
        return mol_data["cusp_features"]


@register_descriptor("dm_statistics")
class DMStatisticsDescriptor(Descriptor):
    """Density-matrix correlation indicators. 3 features."""
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
