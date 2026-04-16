"""xcquinox.alec.solver — SolverConfig, ABCs, and SCF dispatcher.

Implements the SCF solver toggle design:
docs/superpowers/specs/2026-04-14-alec-scf-solver-and-ref-density-rename-design.md

This module intentionally contains no backend SCF logic. Backends are
lazily imported inside `run_scf` so that `import xcquinox.alec.solver`
never pulls `pyscfad` for users who only need the manual path.
"""
from enum import Enum


class SolverBackend(str, Enum):
    MANUAL = "manual"
    PYSCFAD = "pyscfad"


class SolverMode(str, Enum):
    ONESHOT = "oneshot"
    FIXED_J = "fixed_j"
    FULL = "full"


class FeaturePolicy(str, Enum):
    FROZEN = "frozen"
    REASSEMBLE = "reassemble"


from dataclasses import dataclass, field


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for SCF solver selection and behavior.

    Frozen + hashable so JAX jit caches work. mixer_kwargs is a
    tuple-of-tuples to stay hashable.
    """
    backend: SolverBackend = SolverBackend.MANUAL
    mode: SolverMode = SolverMode.ONESHOT
    max_cycles: int = 0
    conv_tol: float = 1e-6
    feature_policy: FeaturePolicy | None = None
    mixer_name: str = "linear"
    mixer_kwargs: tuple[tuple[str, float], ...] = (("alpha", 0.5),)
    convergence_name: str = "energy"

    def __post_init__(self):
        if self.max_cycles < 0:
            raise ValueError(f"max_cycles must be >= 0, got {self.max_cycles}")
        if self.conv_tol <= 0:
            raise ValueError(f"conv_tol must be > 0, got {self.conv_tol}")
        if self.mode == SolverMode.ONESHOT and self.max_cycles != 0:
            raise ValueError(
                f"oneshot mode requires max_cycles=0, got {self.max_cycles}"
            )
        if self.mode != SolverMode.ONESHOT and self.max_cycles == 0:
            raise ValueError(
                f"non-oneshot modes require max_cycles > 0, got 0 with mode={self.mode}"
            )

    @property
    def effective_feature_policy(self) -> FeaturePolicy:
        if self.feature_policy is not None:
            return self.feature_policy
        if self.mode == SolverMode.FIXED_J:
            return FeaturePolicy.FROZEN
        return FeaturePolicy.REASSEMBLE

    def describe(self) -> dict:
        """JSON-serializable form for test_metadata.json logging."""
        return {
            "backend": self.backend.value,
            "mode": self.mode.value,
            "max_cycles": self.max_cycles,
            "conv_tol": self.conv_tol,
            "feature_policy": self.effective_feature_policy.value,
            "mixer_name": self.mixer_name,
            "mixer_kwargs": dict(self.mixer_kwargs),
            "convergence_name": self.convergence_name,
        }


import abc
from typing import NamedTuple
import jax.numpy as jnp


class MixerState(NamedTuple):
    """Base mixer state. Subclasses may extend via their own NamedTuple."""
    step_index: jnp.ndarray  # int32 scalar


class Mixer(abc.ABC):
    registry_name: str = ""

    @abc.abstractmethod
    def init_state(self, nao: int) -> MixerState:
        ...

    @abc.abstractmethod
    def step(self, state: MixerState, D_in: jnp.ndarray,
             D_out: jnp.ndarray) -> tuple[MixerState, jnp.ndarray]:
        """Returns (new_state, D_mixed)."""


class LinearMixer(Mixer):
    registry_name = "linear"

    def __init__(self, alpha: float = 0.5):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    def init_state(self, nao: int) -> MixerState:
        return MixerState(step_index=jnp.int32(0))

    def step(self, state, D_in, D_out):
        D_mixed = self.alpha * D_out + (1.0 - self.alpha) * D_in
        new_state = MixerState(step_index=state.step_index + jnp.int32(1))
        return new_state, D_mixed
