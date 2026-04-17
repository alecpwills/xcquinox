"""Tests for xcquinox.alec.balancing — LossMetric, BalancingConfig hierarchy."""
import pytest
from dataclasses import FrozenInstanceError


def test_loss_metric_enum_values():
    from xcquinox.alec.balancing import LossMetric
    assert LossMetric.ABSOLUTE.value == "absolute"
    assert LossMetric.RELATIVE.value == "relative"


def test_balancing_config_describe():
    from xcquinox.alec.balancing import BalancingConfig
    cfg = BalancingConfig()
    assert cfg.describe() == {"strategy": "static"}


def test_balancing_config_frozen():
    from xcquinox.alec.balancing import BalancingConfig
    cfg = BalancingConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.strategy = "other"


def test_lossnorm_config_describe():
    from xcquinox.alec.balancing import LossNormConfig
    cfg = LossNormConfig()
    assert cfg.describe() == {"strategy": "loss_norm"}


def test_twophase_config_describe():
    from xcquinox.alec.balancing import TwoPhaseConfig
    cfg = TwoPhaseConfig(phase1_steps=100)
    d = cfg.describe()
    assert d == {"strategy": "two_phase", "phase1_steps": 100, "phase1_loss": "A_atomization"}


def test_twophase_validation_phase1_steps():
    from xcquinox.alec.balancing import TwoPhaseConfig
    with pytest.raises(ValueError, match="phase1_steps must be >= 1"):
        TwoPhaseConfig(phase1_steps=0)


def test_gradnorm_config_describe():
    from xcquinox.alec.balancing import GradNormConfig
    cfg = GradNormConfig(alpha=2.0, weight_lr=0.01)
    d = cfg.describe()
    assert d == {"strategy": "gradnorm", "alpha": 2.0, "weight_lr": 0.01}


def test_gradnorm_validation_alpha():
    from xcquinox.alec.balancing import GradNormConfig
    with pytest.raises(ValueError, match="alpha must be > 0"):
        GradNormConfig(alpha=0.0)


def test_gradnorm_validation_weight_lr():
    from xcquinox.alec.balancing import GradNormConfig
    with pytest.raises(ValueError, match="weight_lr must be > 0"):
        GradNormConfig(weight_lr=-0.01)


def test_lossnorm_config_frozen():
    from xcquinox.alec.balancing import LossNormConfig
    cfg = LossNormConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.strategy = "other"


def test_twophase_config_frozen():
    from xcquinox.alec.balancing import TwoPhaseConfig
    cfg = TwoPhaseConfig(phase1_steps=50)
    with pytest.raises(FrozenInstanceError):
        cfg.phase1_steps = 100


def test_gradnorm_config_frozen():
    from xcquinox.alec.balancing import GradNormConfig
    cfg = GradNormConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.alpha = 3.0
