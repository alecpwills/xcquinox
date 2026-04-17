"""xcquinox.alec.balancing — LossMetric enum and BalancingConfig hierarchy.

Implements the multi-task loss balancing design spec §3.
"""
from dataclasses import dataclass
from enum import Enum


class LossMetric(str, Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


@dataclass(frozen=True)
class BalancingConfig:
    """Base config. Represents static weighting (current behavior)."""

    def describe(self) -> dict:
        return {"strategy": "static"}


@dataclass(frozen=True)
class LossNormConfig(BalancingConfig):
    """Normalize each loss component by its step-0 value."""

    def describe(self) -> dict:
        return {"strategy": "loss_norm"}


@dataclass(frozen=True)
class TwoPhaseConfig(BalancingConfig):
    """Phase 1: energy-only. Phase 2: compound loss with fresh optimizer."""
    phase1_steps: int
    phase1_loss: str = "A_atomization"

    def __post_init__(self):
        if self.phase1_steps < 1:
            raise ValueError(f"phase1_steps must be >= 1, got {self.phase1_steps}")

    def describe(self) -> dict:
        return {
            "strategy": "two_phase",
            "phase1_steps": self.phase1_steps,
            "phase1_loss": self.phase1_loss,
        }


@dataclass(frozen=True)
class GradNormConfig(BalancingConfig):
    """Full GradNorm (Chen et al. 2018) with learned per-task weights."""
    alpha: float = 1.5
    weight_lr: float = 0.025

    def __post_init__(self):
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if self.weight_lr <= 0:
            raise ValueError(f"weight_lr must be > 0, got {self.weight_lr}")

    def describe(self) -> dict:
        return {
            "strategy": "gradnorm",
            "alpha": self.alpha,
            "weight_lr": self.weight_lr,
        }
