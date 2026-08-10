"""Tests for the pretrained-network held-out eval driver.

The eval protocol itself is not tested here -- the driver deliberately calls
``_eval_one_spec._run_held_out_eval``, the sweep's own eval function, so the
protocol is covered wherever that is. What IS pinned is the one thing this
script does on its own: build the spec's architecture and load the PRETRAIN
weights into it. A silent failure there would evaluate a randomly-initialized
network and report the result as "the pretrained net scores X", which is the
worst possible outcome for a diagnostic whose entire purpose is to localize a
divergence.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_RUN = (Path.home() / "Documents/Research/xcquinox-results/runs/dfs_step7"
        / "dfs6311_grid3_v3/runs/run_20260728T140018Z")
_ARCH = "deep_mgga_3x16"
_SPEC_IDX = 34


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


ph = _load("dfs6311_pretrained_holdout")


def _training_spec():
    from xcquinox.alec.cluster._eval_one_spec import (
        _load_spec, _read_width, _spec_path)
    if not (_RUN / "manifest.json").is_file():
        pytest.skip("dfs6311 run not pulled locally")
    return _load_spec(_spec_path(str(_RUN), _SPEC_IDX, _read_width(str(_RUN))))


def test_missing_pretrain_checkpoints_raise_rather_than_return_a_random_net():
    """The failure mode that matters: an absent checkpoint must NOT silently
    yield a freshly-initialized model whose held-out score would be reported as
    the pretrained net's."""
    spec = _training_spec()
    with pytest.raises(FileNotFoundError, match="pretrain checkpoint missing"):
        ph.build_pretrained_model(spec, _RUN / "pretrain" / "no_such_arch")


def test_only_the_cnet_missing_still_raises(tmp_path):
    """Both subnets are required. A dir carrying xnet.eqx alone must not load
    half the weights and leave the C-net at its random init."""
    import shutil
    spec = _training_spec()
    src = _RUN / "pretrain" / _ARCH
    if not (src / "xnet.eqx").is_file():
        pytest.skip("pretrain checkpoints not pulled")
    half = tmp_path / "half"
    half.mkdir()
    shutil.copy(src / "xnet.eqx", half / "xnet.eqx")
    with pytest.raises(FileNotFoundError, match="cnet.eqx"):
        ph.build_pretrained_model(spec, half)


def test_pretrained_weights_actually_reach_the_model():
    """A mutant whose ``tree_at`` no-ops (or loads into the wrong subnet)
    returns the RANDOM skeleton. Pin that the loaded model differs from a fresh
    one, and that it reproduces the pretrained F_x measured independently in
    notebooks/analysis/mgga_diagnosis_evidence.py."""
    import numpy as np
    import jax.numpy as jnp
    from xcquinox.alec.models import AlecGGAModel

    spec = _training_spec()
    if not (_RUN / "pretrain" / _ARCH / "xnet.eqx").is_file():
        pytest.skip("pretrain checkpoints not pulled")
    loaded = ph.build_pretrained_model(spec, _RUN / "pretrain" / _ARCH)
    fresh = AlecGGAModel.from_arch(spec.arch, seed=0)

    def fx(model, s, alpha, rho=1.0):
        k_f = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
        sigma = (s * 2.0 * k_f * rho) ** 2
        return float(np.asarray(model.eval_Fx(
            jnp.asarray([rho]), jnp.asarray([sigma]),
            jnp.asarray(np.array([[alpha]]))))[0])

    # the weights changed the function
    assert fx(loaded, 1.0, 1.0) != pytest.approx(fx(fresh, 1.0, 1.0), abs=1e-6)
    # and they are the SCAN-clone weights: these three values were measured
    # independently from the same checkpoints by the diagnosis script.
    assert fx(loaded, 0.0, 0.0) == pytest.approx(1.172, abs=2e-3)
    assert fx(loaded, 0.0, 1.0) == pytest.approx(1.000, abs=2e-3)
    assert fx(loaded, 4.0, 100.0) == pytest.approx(0.662, abs=2e-3)


def test_driver_reuses_the_sweeps_own_eval_entry_point():
    """Protocol identity is the whole argument for this comparison being
    meaningful, so the driver must not grow its own eval path."""
    src = (_HERE / "dfs6311_pretrained_holdout.py").read_text()
    assert "_run_held_out_eval" in src
    # it must not reimplement the pool assembly or the val/test slicing
    assert "load_full_held_out_pools" not in src
    assert "_test_slice_reactions" not in src


def test_output_subdir_is_distinct_from_the_trained_evals():
    """Writing into eval_holdout/ would overwrite the trained cell's result --
    the very number this is compared against."""
    import inspect
    sig = inspect.signature(ph.main)
    src = (_HERE / "dfs6311_pretrained_holdout.py").read_text()
    assert 'default="eval_holdout_pretrained"' in src
    assert 'model_pretrained.eqx' in src      # never overwrites model.eqx
    assert sig is not None
