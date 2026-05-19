"""Tests for xcquinox.alec.cluster._train_task.

The ``_run_worker`` seam is monkeypatched to return canned ``(rc, text)`` so
no real training subprocess is ever spawned. A synthetic ``run_dir`` (a
minimal ``manifest.json`` for the pad ``width`` + a stub spec file) is built
per-test in a tmp directory. The four classification outcomes, the throttled
progress emission, the zero-JSON-progress (import-crash) path, and the
SIGTERM handler are all exercised.
"""
import json
import os
import signal
import subprocess
import sys

import pytest

from xcquinox.alec.cluster import _train_task as tt


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _write_manifest(run_dir, width=4, n_specs=4):
    payload = {
        "xcquinox_version": "test",
        "python_version": "3.x",
        "width": width,
        "n_specs": n_specs,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(payload, f)


def _write_spec(run_dir, idx, width=4):
    specs_dir = os.path.join(run_dir, "specs")
    os.makedirs(specs_dir, exist_ok=True)
    path = os.path.join(specs_dir, f"spec_{idx:0{width}d}.spec")
    with open(path, "wb") as f:
        f.write(b"stub-spec")
    return path


def _write_model(run_dir, idx, width=4):
    d = os.path.join(run_dir, "checkpoints", f"spec_{idx:0{width}d}")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "model.eqx"), "wb").close()


def _read_failure(run_dir, idx, width=4):
    path = os.path.join(
        run_dir, "checkpoints", f"spec_{idx:0{width}d}", "failure.json")
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    _write_manifest(str(d))
    _write_spec(str(d), 0)
    return str(d)


# ---------------------------------------------------------------------------
# Four-way classification (via the _run_worker seam)
# ---------------------------------------------------------------------------

def test_success_rc0_with_model(run_dir, monkeypatch):
    def fake_worker(spec_path, device):
        _write_model(run_dir, 0)
        return 0, "ok"
    monkeypatch.setattr(tt, "_run_worker", fake_worker)
    assert tt.main([run_dir, "0"]) == 0
    # No failure.json written on success.
    assert not os.path.exists(os.path.join(
        run_dir, "checkpoints", "spec_0000", "failure.json"))


def test_rc0_no_model_is_deterministic_failure(run_dir, monkeypatch):
    monkeypatch.setattr(tt, "_run_worker", lambda s, d: (0, "no checkpoint"))
    assert tt.main([run_dir, "0"]) != 0
    failure = _read_failure(run_dir, 0)
    assert failure["classification"] == "deterministic"
    assert failure["rc"] == 0


def test_rc_nonzero_with_model_is_benign_success(run_dir, monkeypatch):
    def fake_worker(spec_path, device):
        _write_model(run_dir, 0)
        return 139, "segfault on teardown"
    monkeypatch.setattr(tt, "_run_worker", fake_worker)
    # model.eqx exists -> teardown anomaly, NOT a failure.
    assert tt.main([run_dir, "0"]) == 0
    assert not os.path.exists(os.path.join(
        run_dir, "checkpoints", "spec_0000", "failure.json"))


def test_rc_nonzero_with_model_unknown_code_still_success(run_dir, monkeypatch):
    def fake_worker(spec_path, device):
        _write_model(run_dir, 0)
        return 1, "weird teardown"
    monkeypatch.setattr(tt, "_run_worker", fake_worker)
    assert tt.main([run_dir, "0"]) == 0


def test_rc_nonzero_no_model_oom(run_dir, monkeypatch):
    monkeypatch.setattr(
        tt, "_run_worker",
        lambda s, d: (1, "jaxlib RESOURCE_EXHAUSTED: out of GPU memory"))
    assert tt.main([run_dir, "0"]) != 0
    failure = _read_failure(run_dir, 0)
    assert failure["classification"] == "oom"
    assert "RESOURCE_EXHAUSTED" in failure["log_excerpt"]


def test_rc_nonzero_no_model_sigkill_is_oom(run_dir, monkeypatch):
    # rc -9 (SIGKILL) with no marker -> still OOM by exit code alone.
    monkeypatch.setattr(tt, "_run_worker", lambda s, d: (-9, "no output"))
    assert tt.main([run_dir, "0"]) != 0
    assert _read_failure(run_dir, 0)["classification"] == "oom"


def test_rc_nonzero_no_model_deterministic(run_dir, monkeypatch):
    monkeypatch.setattr(
        tt, "_run_worker",
        lambda s, d: (1, "Traceback ... ValueError: bad spec"))
    assert tt.main([run_dir, "0"]) != 0
    failure = _read_failure(run_dir, 0)
    assert failure["classification"] == "deterministic"
    assert failure["rc"] == 1


def test_missing_spec_file_is_deterministic_failure(run_dir, monkeypatch):
    os.remove(os.path.join(run_dir, "specs", "spec_0000.spec"))
    # _run_worker should never be reached.
    monkeypatch.setattr(tt, "_run_worker", lambda s, d: pytest.fail("ran"))
    assert tt.main([run_dir, "0"]) == 2
    assert _read_failure(run_dir, 0)["classification"] == "deterministic"


# ---------------------------------------------------------------------------
# _looks_like_gpu_oom
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,rc", [
    ("cuSolver internal error", None),
    ("CUDA_ERROR_OUT_OF_MEMORY", 1),
    ("nothing", -9),
    ("nothing", 137),
])
def test_looks_like_gpu_oom_positive(text, rc):
    assert tt._looks_like_gpu_oom(text, rc) is True


@pytest.mark.parametrize("text,rc", [
    ("ordinary ValueError traceback", 1),
    ("", None),
    ("", -11),
])
def test_looks_like_gpu_oom_negative(text, rc):
    assert tt._looks_like_gpu_oom(text, rc) is False


# ---------------------------------------------------------------------------
# _run_worker — throttled progress + zero-progress robustness
# ---------------------------------------------------------------------------

def _fake_popen_factory(lines, rc):
    """Build a fake subprocess.Popen class yielding ``lines`` then exit ``rc``."""
    class _FakeStdout:
        def __init__(self, items):
            self._items = list(items)

        def __iter__(self):
            return iter(self._items)

    class _FakePopen:
        last_kwargs = None

        def __init__(self, cmd, **kwargs):
            type(self).last_kwargs = kwargs
            self.cmd = cmd
            self.stdout = _FakeStdout(lines)
            self._rc = rc

        def poll(self):
            return self._rc

        def wait(self):
            return self._rc

        def terminate(self):
            pass

    return _FakePopen


def test_run_worker_emits_throttled_progress(monkeypatch):
    # 1200 step lines -> with _THROTTLE_STEPS=500, expect the first step plus
    # ~2 more + the final = at least 2 emitted [harness idx= lines.
    step_lines = [
        json.dumps({"kind": "step", "step": i, "total": 1200, "loss": 0.1})
        + "\n"
        for i in range(1, 1201)
    ]
    lines = [json.dumps({"kind": "init"}) + "\n"] + step_lines \
        + [json.dumps({"kind": "done", "elapsed_s": 1.0}) + "\n"]
    monkeypatch.setattr(
        subprocess, "Popen", _fake_popen_factory(lines, rc=0))

    emitted = []
    monkeypatch.setattr(tt, "_PROGRESS_SINK", emitted.append)
    rc, tail = tt._run_worker("/tmp/x.spec", "auto")
    assert rc == 0
    assert len(emitted) >= 2
    assert all("step" in line and "loss=" in line for line in emitted)


def test_run_worker_zero_json_progress_does_not_crash(monkeypatch):
    # An import-time crash: only non-JSON traceback lines, no JSON progress.
    lines = [
        "Traceback (most recent call last):\n",
        '  File "x.py", line 1, in <module>\n',
        "ImportError: cannot import name 'foo'\n",
    ]
    monkeypatch.setattr(
        subprocess, "Popen", _fake_popen_factory(lines, rc=1))
    emitted = []
    monkeypatch.setattr(tt, "_PROGRESS_SINK", emitted.append)
    rc, tail = tt._run_worker("/tmp/x.spec", "auto")
    assert rc == 1
    assert emitted == []  # no progress line for a zero-JSON worker
    assert "ImportError" in tail  # tail captured the traceback


def test_run_worker_spawns_with_env_none_and_no_progress_flag(monkeypatch):
    popen_cls = _fake_popen_factory(
        [json.dumps({"kind": "done"}) + "\n"], rc=0)
    monkeypatch.setattr(subprocess, "Popen", popen_cls)
    monkeypatch.setattr(tt, "_PROGRESS_SINK", lambda m: None)
    tt._run_worker("/tmp/x.spec", "gpu")
    # env=None -> full inheritance so sbatch thread-cap vars reach the worker.
    assert popen_cls.last_kwargs["env"] is None
    # --no-progress must NOT be passed (we need the JSON progress stream).


def test_run_worker_bounded_tail(monkeypatch):
    many = [f"line {i}\n" for i in range(5000)]
    monkeypatch.setattr(subprocess, "Popen", _fake_popen_factory(many, rc=1))
    monkeypatch.setattr(tt, "_PROGRESS_SINK", lambda m: None)
    rc, tail = tt._run_worker("/tmp/x.spec", "auto")
    assert rc == 1
    assert len(tail.splitlines()) <= tt._TAIL_MAX_LINES
    assert len(tail) <= tt._TAIL_MAX_CHARS


# ---------------------------------------------------------------------------
# SIGTERM handler
# ---------------------------------------------------------------------------

def test_write_signal_failure_writes_correct_failure_json(run_dir):
    tt._write_signal_failure(run_dir, 0, rc=-15)
    failure = _read_failure(run_dir, 0)
    assert failure["classification"] == "killed_by_signal"
    assert failure["rc"] == -15
    assert "SIGTERM" in failure["log_excerpt"]


def test_install_sigterm_handler_is_registered(run_dir):
    original = signal.getsignal(signal.SIGTERM)
    try:
        handler = tt._install_sigterm_handler(run_dir, 0)
        assert signal.getsignal(signal.SIGTERM) is handler
    finally:
        signal.signal(signal.SIGTERM, original)


def test_main_installs_sigterm_handler(run_dir, monkeypatch):
    original = signal.getsignal(signal.SIGTERM)
    try:
        def fake_worker(spec_path, device):
            # The handler must already be installed by the time the worker
            # runs — confirm it is no longer the original.
            assert signal.getsignal(signal.SIGTERM) is not original
            _write_model(run_dir, 0)
            return 0, "ok"
        monkeypatch.setattr(tt, "_run_worker", fake_worker)
        assert tt.main([run_dir, "0"]) == 0
    finally:
        signal.signal(signal.SIGTERM, original)


def test_write_failure_json_is_atomic_and_leaves_no_tmp(run_dir):
    d = os.path.join(run_dir, "checkpoints", "spec_0000")
    tt._write_failure_json(d, {"classification": "deterministic", "rc": 1})
    leftovers = [f for f in os.listdir(d) if f.startswith(".mktmp_")]
    assert leftovers == []
    assert os.path.exists(os.path.join(d, "failure.json"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
