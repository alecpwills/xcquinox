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
