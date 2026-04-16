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


from xcquinox.alec.solver import SolverConfig


def test_solver_config_default_is_oneshot_manual():
    cfg = SolverConfig()
    assert cfg.backend == SolverBackend.MANUAL
    assert cfg.mode == SolverMode.ONESHOT
    assert cfg.max_cycles == 0
    assert cfg.conv_tol == 1e-6


def test_solver_config_rejects_negative_cycles():
    with pytest.raises(ValueError, match="max_cycles must be >= 0"):
        SolverConfig(max_cycles=-1)


def test_solver_config_rejects_oneshot_with_cycles():
    with pytest.raises(ValueError, match="oneshot mode requires max_cycles=0"):
        SolverConfig(mode=SolverMode.ONESHOT, max_cycles=3)


def test_solver_config_rejects_nononeshot_with_zero_cycles():
    with pytest.raises(ValueError, match="non-oneshot modes require"):
        SolverConfig(mode=SolverMode.FIXED_J, max_cycles=0)


def test_solver_config_rejects_nonpositive_tol():
    with pytest.raises(ValueError, match="conv_tol must be > 0"):
        SolverConfig(conv_tol=0)


def test_solver_config_is_hashable():
    cfg = SolverConfig()
    hash(cfg)
    d = {cfg: "value"}
    assert d[SolverConfig()] == "value"


def test_solver_config_describe_is_json_serializable():
    cfg = SolverConfig(mode=SolverMode.FIXED_J, max_cycles=5)
    described = cfg.describe()
    assert json.dumps(described)


def test_effective_feature_policy_fixed_j_is_frozen():
    cfg = SolverConfig(mode=SolverMode.FIXED_J, max_cycles=5)
    assert cfg.effective_feature_policy == FeaturePolicy.FROZEN


def test_effective_feature_policy_full_is_reassemble():
    cfg = SolverConfig(mode=SolverMode.FULL, max_cycles=5)
    assert cfg.effective_feature_policy == FeaturePolicy.REASSEMBLE


def test_effective_feature_policy_honors_explicit_override():
    cfg = SolverConfig(
        mode=SolverMode.FIXED_J, max_cycles=5,
        feature_policy=FeaturePolicy.REASSEMBLE,
    )
    assert cfg.effective_feature_policy == FeaturePolicy.REASSEMBLE
