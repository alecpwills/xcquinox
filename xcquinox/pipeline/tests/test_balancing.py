"""Tests for xcquinox.pipeline.balancing: LossMetric, BalancingConfig hierarchy."""
import os
import tempfile

import pytest
from dataclasses import FrozenInstanceError

from xcquinox.pipeline.config import ArchitectureConfig, TrainingSpec
from xcquinox.pipeline.tests.fixtures.molecules import h_atom, o_atom, h2o_molecule


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
    from xcquinox.pipeline.balancing import LossMetric
    assert LossMetric.ABSOLUTE.value == "absolute"
    assert LossMetric.RELATIVE.value == "relative"


def test_balancing_config_frozen():
    from xcquinox.pipeline.balancing import BalancingConfig
    cfg = BalancingConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.strategy = "other"


def test_twophase_validation_phase1_steps():
    from xcquinox.pipeline.balancing import TwoPhaseConfig
    with pytest.raises(ValueError, match="phase1_steps must be >= 1"):
        TwoPhaseConfig(phase1_steps=0)


def test_gradnorm_validation_alpha():
    from xcquinox.pipeline.balancing import GradNormConfig
    with pytest.raises(ValueError, match="alpha must be > 0"):
        GradNormConfig(alpha=0.0)


@pytest.mark.slow
def test_lossnorm_runs_and_produces_artifacts():
    """LossNorm training runs and produces standard artifacts."""
    from xcquinox.pipeline.balancing import LossNormConfig
    from xcquinox.pipeline.train import run_training
    spec = _make_balancing_spec(LossNormConfig(), n_steps=3)
    metadata = run_training(spec)
    assert os.path.isfile(os.path.join(spec.checkpoint_dir, "model.eqx"))
    assert os.path.isfile(os.path.join(spec.checkpoint_dir, "losses.npy"))
    assert metadata["balancing"] == {"strategy": "loss_norm"}
    assert metadata["loss_metric"] == "absolute"


@pytest.mark.slow
def test_twophase_phase_transition_in_aux_log():
    """TwoPhase aux_log shows phase=1 then phase=2."""
    from xcquinox.pipeline.balancing import TwoPhaseConfig
    from xcquinox.pipeline.train import run_training
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


@pytest.mark.slow
def test_gradnorm_weights_adapt():
    """After training, GradNorm learned weights differ from initial equal weights."""
    from xcquinox.pipeline.balancing import GradNormConfig
    from xcquinox.pipeline.train import run_training
    import pickle as pkl  # noqa: S403
    spec = _make_balancing_spec(GradNormConfig(alpha=1.5, weight_lr=0.025), n_steps=5)
    run_training(spec)
    aux_path = os.path.join(spec.checkpoint_dir, "aux_log.pkl")
    with open(aux_path, "rb") as f:
        aux_log = pkl.load(f)  # noqa: S301
    first_weights = aux_log[0]["balancing_info"]["effective_weights"]
    last_weights = aux_log[-1]["balancing_info"]["effective_weights"]
    changed = any(
        abs(first_weights[k] - last_weights[k]) > 1e-6
        for k in first_weights
    )
    assert changed, "GradNorm weights did not adapt during training"


@pytest.mark.slow
@pytest.mark.parametrize("loss_metric", ["absolute", "relative"])
def test_metadata_records_loss_metric(loss_metric):
    """train_metadata.json includes correct loss_metric field."""
    import json
    from xcquinox.pipeline.train import run_training
    spec = _make_balancing_spec(None, n_steps=3, loss_metric=loss_metric)
    run_training(spec)
    md_path = os.path.join(spec.checkpoint_dir, "train_metadata.json")
    with open(md_path) as f:
        md = json.load(f)
    assert md["loss_metric"] == loss_metric


