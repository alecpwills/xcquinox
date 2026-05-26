"""Tests for xcquinox.alec.cluster._pretrain.

The heavy ``run_pretrain`` call is monkeypatched at the ``_run_pretrain`` seam
so NO real pretraining / JAX compute is ever spawned. A synthetic ``run_dir``
(a minimal ``resolved_config.yaml``) is built per-test in a tmp directory.

Coverage:
  - ``_pretrain.main`` loads ``resolved_config.yaml``, selects the correct
    arch for a given ``arch_idx``, and builds a ``PretrainSpec`` with the
    right ``checkpoint_dir`` = ``<run_dir>/pretrain/<arch>/`` and every
    ``cfg.pretrain`` field threaded through.
  - out-of-range ``arch_idx`` fails fast (non-zero exit, clear message).
  - the silent-no-checkpoint guard: mocked ``_run_pretrain`` "succeeds" but
    writes nothing -> worker exits non-zero.
  - mocked ``_run_pretrain`` writes ``xnet.eqx`` + ``cnet.eqx`` -> exit 0.
  - the throttled progress callback emits at least one
    ``[harness pretrain arch=`` line over a multi-step run and does not crash
    on a zero-step run.
  - the ``pretrain.sbatch.tmpl`` template is a valid ``string.Template`` and
    renders with no leftover ``${...}``, and the rendered body invokes
    ``python -m xcquinox.alec.cluster._pretrain``.
"""
import importlib.resources
import json
import os
import sys
from string import Template

import pytest

from xcquinox.alec.cluster import _pretrain as pt
from xcquinox.alec.config import PretrainSpec, get_architecture


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _config_dict(archs=("medium", "shallow", "deep"),
                  data_dir="/tmp/pretrain_data"):
    """A complete-but-minimal resolved config; ``load_grid_config`` needs every
    section, but only ``sweep.arch`` and ``pretrain`` are read by the worker."""
    return {
        "sweep": {
            "arch": list(archs),
            "loss": ["l2"],
            "metric": ["l2"],
            "subset_size": [1],
            "solver": ["oneshot"],
        },
        "solvers": {
            "oneshot": {"mode": "oneshot", "max_cycles": 1},
        },
        "hyperparams": {
            "n_steps": 1,
            "lr_start": 1e-3,
            "lr_end": 1e-4,
            "lr_decay_start": 0.5,
            "grad_clip": 1.0,
            "gradnorm_alpha": 1.0,
            "vxc_weight": 1.0,
            "density_weight": 1.0,
        },
        "inputs": {
            "external_refs_dir": "/tmp/refs",
            "subset_ledger_path": "/tmp/ledger.json",
            "basis": "def2-svp",
            "grid_level": 1,
            "output_root": "/tmp/out",
        },
        "pretrain": {
            "data_dir": data_dir,
            "n_steps": 777,
            "lr_start": 5e-2,
            "lr_end": 3e-5,
            "lr_decay_start": 0.25,
            "grad_clip": 0.5,
            "seed": 17,
            "loss_weighting": "integration",
        },
        "cluster": {
            "partition": "short",
            "time": "01:00:00",
            "mem": "8G",
            "cpus_per_task": 1,
            "array_throttle": 1,
            "eval_array_throttle": 1,
            "max_concurrent_tasks": 10,
        },
        "domain_profile": "dfs_step7",
    }


def _write_config(run_dir, cfg=None):
    """Write resolved_config.yaml (JSON fallback) and return its path."""
    cfg = cfg or _config_dict()
    path = os.path.join(run_dir, "resolved_config.yaml")
    try:
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f)
    except ImportError:  # pragma: no cover -- env-dependent
        path = os.path.join(run_dir, "resolved_config.json")
        with open(path, "w") as f:
            json.dump(cfg, f)
    return path


@pytest.fixture
def run_dir(tmp_path):
    """A run dir whose config points data_dir at a tmp-scoped path so checkpoint
    artifacts never leak between tests. Pretrain checkpoints land under the run
    dir itself (<run_dir>/pretrain/<arch>)."""
    d = tmp_path / "run"
    d.mkdir()
    data_dir = tmp_path / "pretrain_data"
    data_dir.mkdir()
    _write_config(
        str(d),
        _config_dict(data_dir=str(data_dir)),
    )
    return str(d)


# ---------------------------------------------------------------------------
# distinct-arch derivation + spec construction
# ---------------------------------------------------------------------------

def test_distinct_archs_is_sorted_and_deduped(run_dir):
    from xcquinox.alec.cluster.grid_config import load_grid_config
    cfg = load_grid_config(
        os.path.join(run_dir, "resolved_config.yaml")
        if os.path.isfile(os.path.join(run_dir, "resolved_config.yaml"))
        else os.path.join(run_dir, "resolved_config.json")
    )
    # _config_dict declares ("medium", "shallow", "deep") -> sorted-deduped.
    assert pt._distinct_archs(cfg) == ["deep", "medium", "shallow"]


def test_main_builds_pretrain_spec_with_correct_checkpoint_dir(run_dir, monkeypatch):
    captured = {}

    def fake_run_pretrain(spec, progress_callback=None):
        captured["spec"] = spec
        # emulate run_pretrain writing both checkpoints.
        os.makedirs(spec.checkpoint_dir, exist_ok=True)
        open(os.path.join(spec.checkpoint_dir, "xnet.eqx"), "wb").close()
        open(os.path.join(spec.checkpoint_dir, "cnet.eqx"), "wb").close()
        return {"arch_name": spec.arch.name}

    monkeypatch.setattr(pt, "_run_pretrain", fake_run_pretrain)

    # arch_idx 1 -> sorted distinct archs ["deep","medium","shallow"][1] = "medium".
    rc = pt.main([run_dir, "1"])
    assert rc == 0

    from xcquinox.alec.cluster.grid_config import load_grid_config
    cfg_path = os.path.join(run_dir, "resolved_config.yaml")
    if not os.path.isfile(cfg_path):
        cfg_path = os.path.join(run_dir, "resolved_config.json")
    cfg = load_grid_config(cfg_path)

    spec = captured["spec"]
    assert isinstance(spec, PretrainSpec)
    assert spec.arch.name == "medium"
    # checkpoint_dir == <run_dir>/pretrain/<arch>/ (run-scoped so two runs
    # pretraining the same arch don't collide; run_dir is unique per submission).
    assert spec.checkpoint_dir == os.path.join(
        os.path.abspath(run_dir), "pretrain", "medium"
    )
    # every cfg.pretrain field threaded through.
    assert spec.data_dir == cfg.pretrain.data_dir
    assert spec.n_steps == 777
    assert spec.lr_start == 5e-2
    assert spec.lr_end == 3e-5
    assert spec.lr_decay_start == 0.25
    assert spec.grad_clip == 0.5
    assert spec.seed == 17
    assert spec.loss_weighting == "integration"


def test_main_selects_arch_by_index(run_dir, monkeypatch):
    seen = []

    def fake_run_pretrain(spec, progress_callback=None):
        seen.append(spec.arch.name)
        os.makedirs(spec.checkpoint_dir, exist_ok=True)
        open(os.path.join(spec.checkpoint_dir, "xnet.eqx"), "wb").close()
        open(os.path.join(spec.checkpoint_dir, "cnet.eqx"), "wb").close()
        return {}

    monkeypatch.setattr(pt, "_run_pretrain", fake_run_pretrain)

    for idx, expect in enumerate(["deep", "medium", "shallow"]):
        seen.clear()
        assert pt.main([run_dir, str(idx)]) == 0
        assert seen == [expect]


# ---------------------------------------------------------------------------
# out-of-range arch_idx
# ---------------------------------------------------------------------------

def test_out_of_range_arch_idx_fails_fast(run_dir, monkeypatch):
    monkeypatch.setattr(
        pt, "_run_pretrain",
        lambda *a, **k: pytest.fail("_run_pretrain ran for an out-of-range idx"))
    # 3 distinct archs -> valid indices 0..2; 3 is out of range.
    assert pt.main([run_dir, "3"]) != 0


def test_negative_arch_idx_fails_fast(run_dir, monkeypatch):
    monkeypatch.setattr(
        pt, "_run_pretrain",
        lambda *a, **k: pytest.fail("_run_pretrain ran for a negative idx"))
    assert pt.main([run_dir, "-1"]) != 0


# ---------------------------------------------------------------------------
# silent-no-checkpoint guard
# ---------------------------------------------------------------------------

def test_silent_no_checkpoint_exits_nonzero(run_dir, monkeypatch):
    # _run_pretrain "succeeds" (returns normally) but writes nothing.
    monkeypatch.setattr(pt, "_run_pretrain", lambda spec, progress_callback=None: {})
    assert pt.main([run_dir, "0"]) != 0


def test_partial_checkpoint_exits_nonzero(run_dir, monkeypatch):
    # Only xnet.eqx written -> still a failure (cnet.eqx missing).
    def fake_run_pretrain(spec, progress_callback=None):
        os.makedirs(spec.checkpoint_dir, exist_ok=True)
        open(os.path.join(spec.checkpoint_dir, "xnet.eqx"), "wb").close()
        return {}

    monkeypatch.setattr(pt, "_run_pretrain", fake_run_pretrain)
    assert pt.main([run_dir, "0"]) != 0


def test_both_checkpoints_written_exits_zero(run_dir, monkeypatch):
    def fake_run_pretrain(spec, progress_callback=None):
        os.makedirs(spec.checkpoint_dir, exist_ok=True)
        open(os.path.join(spec.checkpoint_dir, "xnet.eqx"), "wb").close()
        open(os.path.join(spec.checkpoint_dir, "cnet.eqx"), "wb").close()
        return {}

    monkeypatch.setattr(pt, "_run_pretrain", fake_run_pretrain)
    assert pt.main([run_dir, "0"]) == 0


def test_run_pretrain_exception_exits_nonzero(run_dir, monkeypatch):
    def boom(spec, progress_callback=None):
        raise RuntimeError("pretrain blew up")

    monkeypatch.setattr(pt, "_run_pretrain", boom)
    assert pt.main([run_dir, "0"]) != 0


# ---------------------------------------------------------------------------
# missing config
# ---------------------------------------------------------------------------

def test_missing_resolved_config_exits_nonzero(tmp_path, monkeypatch):
    d = tmp_path / "emptyrun"
    d.mkdir()
    monkeypatch.setattr(
        pt, "_run_pretrain", lambda *a, **k: pytest.fail("ran"))
    assert pt.main([str(d), "0"]) != 0


# ---------------------------------------------------------------------------
# JAX env routing
# ---------------------------------------------------------------------------

def test_route_jax_env_sets_x64(monkeypatch):
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    pt._route_jax_env()
    assert os.environ["JAX_ENABLE_X64"] == "1"


# ---------------------------------------------------------------------------
# throttled progress callback
# ---------------------------------------------------------------------------

def test_progress_callback_emits_harness_line(capsys):
    cb = pt._make_progress_callback("medium")
    for step in range(1, 251):
        cb({"arch": "medium", "phase": "X", "step": step, "total": 250,
            "loss": 0.01, "timestamp": 0.0})
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines()
             if ln.startswith("[harness pretrain arch=medium]")]
    assert len(lines) >= 1
    assert all("phase=X" in ln and "loss=" in ln for ln in lines)


def test_progress_callback_zero_step_run_does_not_crash(capsys):
    cb = pt._make_progress_callback("shallow")
    # A zero-step run: callback is simply never invoked -- must not crash and
    # must produce no harness progress line.
    out = capsys.readouterr().out
    assert "[harness pretrain arch=" not in out
    # And invoking it once at step 0 (defensive) also must not crash.
    cb({"arch": "shallow", "phase": "X", "step": 0, "total": 0, "loss": 0.0,
        "timestamp": 0.0})


def test_progress_callback_emits_on_phase_switch(capsys):
    cb = pt._make_progress_callback("deep")
    cb({"phase": "X", "step": 1, "total": 10, "loss": 0.1, "timestamp": 0.0})
    cb({"phase": "C", "step": 1, "total": 10, "loss": 0.2, "timestamp": 0.0})
    out = capsys.readouterr().out
    assert "phase=X" in out
    assert "phase=C" in out


# ---------------------------------------------------------------------------
# pretrain.sbatch.tmpl template
# ---------------------------------------------------------------------------

def _template_text():
    res = (
        importlib.resources.files("xcquinox.alec.cluster")
        / "templates" / "pretrain.sbatch.tmpl"
    )
    return res.read_text(encoding="utf-8")


def test_pretrain_template_renders_with_no_leftover_placeholders():
    text = _template_text()
    mapping = {
        "JOB_NAME": "xcq_pretrain",
        "PARTITION": "short",
        "TIME": "04:00:00",
        "ALLOC_LINES": "#SBATCH --nodes=1\n#SBATCH --exclusive\n",
        "MEM_LINE": "",
        "CPUS_PER_TASK": 4,
        "ARRAY_MAX": 2,
        "THROTTLE": 3,
        "RUN_DIR": "/scratch/run",
        "CONDA_ACTIVATION": "conda activate xcq",
        "MAIL_USER_LINE": "",
        "MAIL_TYPE_LINE": "",
        "ACCOUNT_LINE": "",
    }
    rendered = Template(text).substitute(mapping)
    # No leftover harness placeholder name survives. ``string.Template`` turns
    # ``$$`` into a literal ``$`` (so ``$${SLURM_ARRAY_TASK_ID}`` -> the bash
    # token ``${SLURM_ARRAY_TASK_ID}``); the meaningful check is that every
    # mapping KEY was substituted, leaving no ``${KEY}`` token behind.
    for key in mapping:
        assert "${" + key + "}" not in rendered
    # The rendered body invokes the pretrain worker module.
    assert "python -m xcquinox.alec.cluster._pretrain" in rendered
    # SLURM array directive present + per-array log path.
    assert "#SBATCH --array=0-2%3" in rendered
    assert "logs/pretrain_%A_%a.out" in rendered
    # SLURM's own ${SLURM_ARRAY_TASK_ID} survives string.Template ($$ -> $).
    assert "${SLURM_ARRAY_TASK_ID}" in rendered
    assert "$SLURM_CPUS_PER_TASK" in rendered


def test_pretrain_template_is_valid_string_template():
    # Template construction + identifier scan must not raise.
    tmpl = Template(_template_text())
    # Every identifier is a plain placeholder name (string.Template accepts it).
    assert isinstance(tmpl.template, str)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_pretrain_arch_polarized_when_flag_set(tmp_path, monkeypatch):
    """The pretrain stage rebuilds its arch spin-polarization-aware when the run
    config sets use_polarized_correlation, so the pretrained checkpoint matches
    the (polarized) training arch."""
    captured = {}

    def fake_run_pretrain(spec, progress_callback=None):
        captured["spec"] = spec
        os.makedirs(spec.checkpoint_dir, exist_ok=True)
        open(os.path.join(spec.checkpoint_dir, "xnet.eqx"), "wb").close()
        open(os.path.join(spec.checkpoint_dir, "cnet.eqx"), "wb").close()
        return {"arch_name": spec.arch.name}

    monkeypatch.setattr(pt, "_run_pretrain", fake_run_pretrain)
    d = tmp_path / "run"
    d.mkdir()
    data_dir = tmp_path / "pretrain_data"
    data_dir.mkdir()
    cd = _config_dict(data_dir=str(data_dir))
    cd["use_polarized_correlation"] = True
    _write_config(str(d), cd)

    assert pt.main([str(d), "1"]) == 0
    assert captured["spec"].arch.use_polarized_correlation is True
