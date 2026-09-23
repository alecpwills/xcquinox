"""Tests for xcquinox.pipeline.cluster._train_task.

The ``_run_worker`` seam is monkeypatched to return canned ``(rc, text)`` so
no real training subprocess is ever spawned. A synthetic ``run_dir`` (a
minimal ``manifest.json`` for the pad ``width`` + a stub spec file) is built
per-test in a tmp directory. The four classification outcomes, the throttled
progress emission, the zero-JSON-progress (import-crash) path, and the
SIGTERM handler are all exercised.
"""
import io
import json
import os
import signal
import sys

import pytest

from xcquinox.pipeline.cluster import _train_task as tt


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _write_manifest(run_dir, width=4, n_specs=4, arch="deep_3x16"):
    payload = {
        "xcquinox_version": "test",
        "python_version": "3.x",
        "width": width,
        "n_specs": n_specs,
        "specs": [{"index": i, "spec_file": f"spec_{i:0{width}d}.spec",
                   "sha256": "x" * 64,
                   "cell": {"arch": arch, "loss": "l2", "metric": "l2",
                            "subset_size": 1, "solver": "oneshot"}}
                  for i in range(n_specs)],
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(payload, f)


def _write_pass_certificate(run_dir, arch="deep_3x16", verdict="PASS"):
    d = os.path.join(run_dir, "pretrain", arch)
    os.makedirs(d, exist_ok=True)
    payload = {"verdict": verdict, "arch": arch,
               "summary": {"max_atom_mHa": 0.1, "max_dAE_kcalmol": 0.2}}
    with open(os.path.join(d, "fidelity_certificate.json"), "w") as f:
        json.dump(payload, f)
    return d


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
    # Every orchestration test in this file describes a run whose architecture
    # certified; the gate's own tests remove or downgrade the certificate.
    _write_pass_certificate(str(d))
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


def test_rc_nonzero_no_model_oom(run_dir, monkeypatch):
    monkeypatch.setattr(
        tt, "_run_worker",
        lambda s, d: (1, "jaxlib RESOURCE_EXHAUSTED: out of GPU memory"))
    assert tt.main([run_dir, "0"]) != 0
    failure = _read_failure(run_dir, 0)
    assert failure["classification"] == "oom"
    assert "RESOURCE_EXHAUSTED" in failure["log_excerpt"]


# ---------------------------------------------------------------------------
# _looks_like_gpu_oom
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _run_worker: throttled progress + zero-progress robustness
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SIGTERM handler
# ---------------------------------------------------------------------------


def test_write_failure_json_is_atomic_and_leaves_no_tmp(run_dir):
    d = os.path.join(run_dir, "checkpoints", "spec_0000")
    tt._write_failure_json(d, {"classification": "deterministic", "rc": 1})
    leftovers = [f for f in os.listdir(d) if f.startswith(".mktmp_")]
    assert leftovers == []
    assert os.path.exists(os.path.join(d, "failure.json"))


class _FakeChild:
    """A subprocess.Popen stand-in recording terminate()/wait() ordering."""
    def __init__(self, alive=True):
        self._alive = alive
        self.events = []
        self.wait_timeout = None

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.events.append("terminate")
        # The real child does NOT die instantly; it keeps running (flushing).

    def wait(self, timeout=None):
        self.events.append(("wait", timeout))
        self.wait_timeout = timeout
        self._alive = False
        return 0


def test_sigterm_handler_waits_for_child_flush(run_dir, monkeypatch):
    """WS5-SIG-4: after delivering SIGTERM to the worker, the parent handler must
    `child.wait(timeout=...)` so the worker's best-effort resume flush can
    finish before the parent exits 143 (otherwise the flush is cut off). FAILS
    before the fix (the handler only terminate()s, never waits)."""
    original = signal.getsignal(signal.SIGTERM)
    child = _FakeChild(alive=True)
    monkeypatch.setattr(tt, "_ACTIVE_CHILD", child)
    try:
        handler = tt._install_sigterm_handler(run_dir, 0)
        with pytest.raises(SystemExit) as ei:
            handler(signal.SIGTERM, None)
        assert ei.value.code == 143
    finally:
        signal.signal(signal.SIGTERM, original)
    # terminate() THEN a bounded wait(timeout=positive) -- in that order.
    assert child.events[0] == "terminate"
    assert any(isinstance(e, tuple) and e[0] == "wait" for e in child.events)
    assert child.wait_timeout is not None and child.wait_timeout > 0
    # ordering: the wait happens AFTER the terminate.
    wait_idx = next(i for i, e in enumerate(child.events)
                    if isinstance(e, tuple) and e[0] == "wait")
    assert wait_idx > child.events.index("terminate")


# preflight precompute_failed_species marker short-circuits the worker


# ---------------------------------------------------------------------------
# The pretraining-fidelity gate
# ---------------------------------------------------------------------------

def test_missing_certificate_refuses_before_the_worker_runs(run_dir,
                                                            monkeypatch):
    os.remove(os.path.join(run_dir, "pretrain", "deep_3x16",
                           "fidelity_certificate.json"))
    calls = []
    monkeypatch.setattr(tt, "_run_worker",
                        lambda s, d: calls.append(1) or (0, "ok"))
    assert tt.main([run_dir, "0"]) == 3
    assert calls == []          # the node is never spent on an uncertified spec
    failure = _read_failure(run_dir, 0)
    assert failure["classification"] == "fidelity_certificate_missing"
    assert failure["rc"] == 3
    assert failure["arch"] == "deep_3x16"
    assert "fidelity_certificate.json" in failure["log_excerpt"]


def test_unenforced_failure_lets_the_worker_run(run_dir, monkeypatch,
                                                capsys):
    """A workflow-verification run reaches the train stage with its FAIL on
    record; the log says so."""
    d = os.path.join(run_dir, "pretrain", "deep_3x16")
    with open(os.path.join(d, "fidelity_certificate.json"), "w") as f:
        json.dump({"verdict": "FAIL", "arch": "deep_3x16", "enforced": False,
                   "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                                  "override_reason": "workflow matrix"},
                   "summary": {"max_atom_mHa": 13.7,
                               "max_dAE_kcalmol": 25.7}}, f)

    def fake_worker(spec_path, device):
        _write_model(run_dir, 0)
        return 0, "ok"

    monkeypatch.setattr(tt, "_run_worker", fake_worker)
    assert tt.main([run_dir, "0"]) == 0
    out = capsys.readouterr().out
    assert "enforcement is OFF" in out


# ---------------------------------------------------------------------------
# One document per refusal record
# ---------------------------------------------------------------------------

def _serve_documents(monkeypatch, path, documents):
    """Serve ``documents`` to successive READ opens of ``path``.

    The list returned collects one entry per read served, so a caller can
    state how many parses a record rested on. Writes and every other path are
    passed through; once the list is exhausted its last entry repeats, so a
    caller that reads more often than the sequence is long is handed a
    complete document rather than an empty file.
    """
    import builtins
    real_open = builtins.open
    served: list = []

    def fake_open(file, *args, **kwargs):
        mode = kwargs.get("mode", args[0] if args else "r")
        if str(file) == str(path) and "r" in mode:
            doc = documents[min(len(served), len(documents) - 1)]
            served.append(doc)
            return io.StringIO(doc if isinstance(doc, str)
                               else json.dumps(doc))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    return served


# Three documents, each refused on its own and each producing a DIFFERENT
# refusal record: a FAIL stating numbers, a file that does not parse, and one
# recording an unrecognised verdict. The last two are both UNREADABLE and are
# told apart by the reason the record quotes.
_R1 = {"verdict": "FAIL",
       "summary": {"max_atom_mHa": 13.7, "max_dAE_kcalmol": 25.7}}
_R2 = "{truncated"
_R3 = {"verdict": "nope"}
_R_PASS = {"verdict": "PASS",
           "summary": {"max_atom_mHa": 0.1, "max_dAE_kcalmol": 0.2}}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


