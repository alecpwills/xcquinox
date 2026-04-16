"""Tests for xcquinox.alec.solver — SolverConfig, Mixer, ConvergenceCriterion."""
import json
import pytest
import jax.numpy as jnp

from xcquinox.alec.solver import (
    SolverBackend,
    SolverMode,
    FeaturePolicy,
)


def test_solver_backend_values():
    assert SolverBackend.MANUAL.value == "manual"
    assert SolverBackend.PYSCFAD.value == "pyscfad"


def test_solver_mode_values():
    assert SolverMode.ONESHOT.value == "oneshot"
    assert SolverMode.FIXED_J.value == "fixed_j"
    assert SolverMode.FULL.value == "full"


def test_feature_policy_values():
    assert FeaturePolicy.FROZEN.value == "frozen"
    assert FeaturePolicy.REASSEMBLE.value == "reassemble"


def test_enums_are_json_serializable():
    assert json.dumps(SolverBackend.MANUAL.value) == '"manual"'
    assert json.dumps(SolverMode.FIXED_J.value) == '"fixed_j"'
    assert json.dumps(FeaturePolicy.REASSEMBLE.value) == '"reassemble"'
