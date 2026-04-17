"""Tests for xcquinox.alec.balancing — LossMetric, BalancingConfig hierarchy."""
import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from xcquinox.alec.config import ArchitectureConfig, MoleculeSpec, TrainingSpec
from xcquinox.alec.tests.fixtures.molecules import h_atom, o_atom, h2o_molecule


def _make_arch(**overrides):
    defaults = dict(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    defaults.update(overrides)
    return ArchitectureConfig(**defaults)


def _make_balancing_spec(balancing, loss_name="B_atomization_plus_dm", n_steps=5, **kw):
    tmpdir = tempfile.mkdtemp()
    ckdir = os.path.join(tmpdir, "ckpt")
    h, o, h2o = h_atom(), o_atom(), h2o_molecule()
    return TrainingSpec(
        arch=_make_arch(),
        molecules=(h, o, h2o),
        targets=(("H", -0.5), ("H2O", 0.3), ("O", -74.8)),
        atom_energies=(("H", -0.5), ("O", -74.8)),
        loss_name=loss_name,
        n_steps=n_steps,
        lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.0, grad_clip=1.0,
        checkpoint_dir=ckdir, seed=42,
        balancing=balancing,
        **kw,
    )


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


@pytest.mark.slow
def test_lossnorm_runs_and_produces_artifacts():
    """LossNorm training runs and produces standard artifacts."""
    from xcquinox.alec.balancing import LossNormConfig
    from xcquinox.alec.train import run_training
    spec = _make_balancing_spec(LossNormConfig(), n_steps=3)
    metadata = run_training(spec)
    assert os.path.isfile(os.path.join(spec.checkpoint_dir, "model.eqx"))
    assert os.path.isfile(os.path.join(spec.checkpoint_dir, "losses.npy"))
    assert metadata["balancing"] == {"strategy": "loss_norm"}
    assert metadata["loss_metric"] == "absolute"


@pytest.mark.slow
def test_lossnorm_balancing_info_in_aux_log():
    """LossNorm aux_log entries contain balancing_info with effective_weights."""
    from xcquinox.alec.balancing import LossNormConfig
    from xcquinox.alec.train import run_training
    import pickle as pkl  # noqa: S403
    spec = _make_balancing_spec(LossNormConfig(), n_steps=3)
    run_training(spec)
    aux_path = os.path.join(spec.checkpoint_dir, "aux_log.pkl")
    with open(aux_path, "rb") as f:
        aux_log = pkl.load(f)  # noqa: S301
    assert len(aux_log) == 3
    for entry in aux_log:
        assert "balancing_info" in entry
        assert entry["balancing_info"]["strategy"] == "loss_norm"
        assert "effective_weights" in entry["balancing_info"]


@pytest.mark.slow
def test_twophase_runs_and_produces_artifacts():
    """TwoPhase training runs and produces standard artifacts."""
    from xcquinox.alec.balancing import TwoPhaseConfig
    from xcquinox.alec.train import run_training
    spec = _make_balancing_spec(
        TwoPhaseConfig(phase1_steps=2), n_steps=5)
    metadata = run_training(spec)
    assert os.path.isfile(os.path.join(spec.checkpoint_dir, "model.eqx"))
    assert metadata["balancing"]["strategy"] == "two_phase"


@pytest.mark.slow
def test_twophase_phase_transition_in_aux_log():
    """TwoPhase aux_log shows phase=1 then phase=2."""
    from xcquinox.alec.balancing import TwoPhaseConfig
    from xcquinox.alec.train import run_training
    import pickle as pkl  # noqa: S403
    spec = _make_balancing_spec(
        TwoPhaseConfig(phase1_steps=2), n_steps=5)
    run_training(spec)
    aux_path = os.path.join(spec.checkpoint_dir, "aux_log.pkl")
    with open(aux_path, "rb") as f:
        aux_log = pkl.load(f)  # noqa: S301
    assert len(aux_log) == 5
    phases = [e["balancing_info"]["phase"] for e in aux_log]
    assert phases == [1, 1, 2, 2, 2]
