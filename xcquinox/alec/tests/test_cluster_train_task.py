"""Tests for xcquinox.alec.cluster._train_task.

The ``_run_worker`` seam is monkeypatched to return canned ``(rc, text)`` so
no real training subprocess is ever spawned. A synthetic ``run_dir`` (a
minimal ``manifest.json`` for the pad ``width`` + a stub spec file) is built
per-test in a tmp directory. The four classification outcomes, the throttled
progress emission, the zero-JSON-progress (import-crash) path, and the
SIGTERM handler are all exercised.
"""
import io
import itertools
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


def test_rc_nonzero_no_model_cpu_compile_oom(run_dir, monkeypatch):
    # A large-basis XLA/LLVM CPU *compile* OOM: std::bad_alloc / "Cannot allocate
    # memory" in the tail, SIGABRT (-6), no model.eqx. Must classify as "oom" so
    # `resubmit` retries it on a bigger-memory partition. Regression: these were
    # mislabeled "deterministic" and dropped (6-311++G(3df,2pd)+grid3 runs).
    monkeypatch.setattr(
        tt, "_run_worker",
        lambda s, d: (-6, "[Compiling module jit__step]\n"
                          "LLVM compilation error: Cannot allocate memory\n"
                          "terminate called after throwing an instance of "
                          "'std::bad_alloc'\n  what():  std::bad_alloc"))
    assert tt.main([run_dir, "0"]) != 0
    failure = _read_failure(run_dir, 0)
    assert failure["classification"] == "oom"
    assert "std::bad_alloc" in failure["log_excerpt"]


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
    # CPU-host OOM: a large-basis XLA/LLVM compile exhausts node RAM.
    ("terminate called after throwing an instance of 'std::bad_alloc'", None),
    ("LLVM compilation error: Cannot allocate memory", 1),
    ("LLVM ERROR: pthread_create failed: Resource temporarily unavailable", None),
    # SIGABRT (std::bad_alloc -> abort()) recognized by exit code alone.
    ("no output", -6),
    ("no output", 134),
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
# _run_worker: throttled progress + zero-progress robustness
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
        last_cmd = None

        def __init__(self, cmd, **kwargs):
            type(self).last_kwargs = kwargs
            type(self).last_cmd = cmd
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
    # 1200 step lines: with _THROTTLE_STEPS=500 the worker emits the first
    # step, then one per 500-step stride, plus the final -- a small bounded
    # count, NOT one line per step. Both bounds are asserted so a throttling
    # regression that emits every step is caught, not only an under-emit.
    n_steps = 1200
    step_lines = [
        json.dumps({"kind": "step", "step": i, "total": n_steps, "loss": 0.1})
        + "\n"
        for i in range(1, n_steps + 1)
    ]
    lines = [json.dumps({"kind": "init"}) + "\n"] + step_lines \
        + [json.dumps({"kind": "done", "elapsed_s": 1.0}) + "\n"]
    monkeypatch.setattr(
        subprocess, "Popen", _fake_popen_factory(lines, rc=0))

    emitted = []
    monkeypatch.setattr(tt, "_PROGRESS_SINK", emitted.append)
    rc, tail = tt._run_worker("/tmp/x.spec", "auto")
    assert rc == 0
    # Lower bound: at least the first + final heartbeat.
    assert len(emitted) >= 2
    # Upper bound tied to the throttle stride (first + one per stride + final);
    # a per-step regression would emit ~n_steps, far above this cap.
    max_emissions = n_steps // tt._THROTTLE_STEPS + 3
    assert len(emitted) <= max_emissions, (
        f"throttle must cap emissions near {max_emissions} for {n_steps} steps "
        f"at _THROTTLE_STEPS={tt._THROTTLE_STEPS}, got {len(emitted)} "
        f"(an un-throttled worker would emit ~{n_steps})"
    )
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
    # --no-progress must NOT be passed: the throttled SLURM heartbeat depends on
    # the worker's JSON progress stream, so the flag's absence is asserted.
    assert "--no-progress" not in popen_cls.last_cmd


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
            # runs, confirm it is no longer the original.
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


def test_sigterm_handler_wait_timeout_is_survivable(run_dir, monkeypatch):
    """WS5-SIG-4: a child that overruns the bounded wait (TimeoutExpired) must NOT
    crash the handler -- it still records the failure and exits 143."""
    original = signal.getsignal(signal.SIGTERM)

    class _SlowChild(_FakeChild):
        def wait(self, timeout=None):
            self.events.append(("wait", timeout))
            self.wait_timeout = timeout
            raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)

    child = _SlowChild(alive=True)
    monkeypatch.setattr(tt, "_ACTIVE_CHILD", child)
    try:
        handler = tt._install_sigterm_handler(run_dir, 0)
        with pytest.raises(SystemExit) as ei:
            handler(signal.SIGTERM, None)
        assert ei.value.code == 143
    finally:
        signal.signal(signal.SIGTERM, original)
    # The failure.json is still written despite the wait timing out.
    failure = _read_failure(run_dir, 0)
    assert failure["classification"] == "killed_by_signal"


# preflight precompute_failed_species marker short-circuits the worker
def test_precompute_failed_species_marker_short_circuits(run_dir, monkeypatch):
    """A spec the preflight already marked ``precompute_failed_species`` must NOT
    run the worker (it would burn an exclusive node) and must NOT overwrite the
    precise preflight diagnosis."""
    _write_spec(run_dir, 0)
    ckpt = os.path.join(run_dir, "checkpoints", "spec_0000")
    os.makedirs(ckpt, exist_ok=True)
    marker = {
        "classification": "precompute_failed_species",
        "species": ["N2O"],
        "detail": "preflight marker, keep verbatim",
    }
    with open(os.path.join(ckpt, "failure.json"), "w") as f:
        json.dump(marker, f)

    def boom(spec_path, device):  # noqa: ARG001
        raise AssertionError("worker must not run for a pre-marked spec")
    monkeypatch.setattr(tt, "_run_worker", boom)

    rc = tt.main([run_dir, "0"])
    assert rc == 0
    preserved = _read_failure(run_dir, 0)
    assert preserved["classification"] == "precompute_failed_species"
    assert preserved["species"] == ["N2O"]
    assert preserved["detail"] == "preflight marker, keep verbatim"


def test_run_worker_heartbeat_includes_rss_when_present(monkeypatch):
    """Step lines carrying rss_gb surface it in the throttled heartbeat."""
    n_steps = 3
    lines = [
        json.dumps({"kind": "step", "step": i, "total": n_steps,
                    "loss": 0.1, "rss_gb": 12.34, "hwm_gb": 13.0}) + "\n"
        for i in range(1, n_steps + 1)
    ]
    monkeypatch.setattr(subprocess, "Popen", _fake_popen_factory(lines, rc=0))
    emitted = []
    monkeypatch.setattr(tt, "_PROGRESS_SINK", emitted.append)
    rc, _tail = tt._run_worker("/tmp/x.spec", "auto")
    assert rc == 0
    assert any("rss=12.3G" in line for line in emitted)


def test_run_worker_heartbeat_omits_rss_when_absent(monkeypatch):
    """Legacy step lines without rss_gb emit heartbeats with no rss field."""
    n_steps = 3
    lines = [
        json.dumps({"kind": "step", "step": i, "total": n_steps, "loss": 0.1})
        + "\n"
        for i in range(1, n_steps + 1)
    ]
    monkeypatch.setattr(subprocess, "Popen", _fake_popen_factory(lines, rc=0))
    emitted = []
    monkeypatch.setattr(tt, "_PROGRESS_SINK", emitted.append)
    rc, _tail = tt._run_worker("/tmp/x.spec", "auto")
    assert rc == 0
    assert emitted and not any("rss=" in line for line in emitted)


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


def test_failed_certificate_refuses_with_its_own_classification(run_dir,
                                                                monkeypatch):
    _write_pass_certificate(run_dir, verdict="FAIL")
    monkeypatch.setattr(tt, "_run_worker", lambda s, d: (0, "ok"))
    assert tt.main([run_dir, "0"]) == 3
    failure = _read_failure(run_dir, 0)
    assert failure["classification"] == "fidelity_certificate_failed"
    assert failure["arch"] == "deep_3x16"


def test_unreadable_certificate_is_treated_as_missing(run_dir, monkeypatch,
                                                      capsys):
    """A file that states no usable verdict leaves the spec uncertified.

    The classification vocabulary has two values, and everything that is not a
    literal FAIL joins the absent case: there is no verdict to act on either
    way. The record layer's own word for the state is carried into the log and
    the failure record, so a truncated certificate is not reported in the
    language of a deleted one.
    """
    path = os.path.join(run_dir, "pretrain", "deep_3x16",
                        "fidelity_certificate.json")
    with open(path, "w") as f:
        f.write("{truncated")
    monkeypatch.setattr(tt, "_run_worker", lambda s, d: (0, "ok"))
    assert tt.main([run_dir, "0"]) == 3
    failure = _read_failure(run_dir, 0)
    assert failure["classification"] == "fidelity_certificate_missing"
    assert failure["certificate_status"] == "UNREADABLE"
    assert "UNREADABLE" in capsys.readouterr().out


def test_a_verdict_less_certificate_is_refused_as_unreadable(run_dir,
                                                             monkeypatch):
    """A certificate recording no recognised verdict cannot be waived.

    FAIL is the only status ``enforced: false`` releases, so a schema-less
    payload carrying that flag must not be read as a FAIL.
    """
    d = os.path.join(run_dir, "pretrain", "deep_3x16")
    with open(os.path.join(d, "fidelity_certificate.json"), "w") as f:
        json.dump({"arch": "deep_3x16", "enforced": False,
                   "tolerances": {"override_reason": "workflow matrix"}}, f)
    monkeypatch.setattr(tt, "_run_worker", lambda s, d: (0, "ok"))
    assert tt.main([run_dir, "0"]) == 3
    failure = _read_failure(run_dir, 0)
    assert failure["classification"] == "fidelity_certificate_missing"
    assert failure["certificate_status"] == "UNREADABLE"


@pytest.mark.parametrize("reason", (None, "", "   ", False, 0))
def test_a_waiver_that_states_no_reason_does_not_release_the_train_task(
        run_dir, monkeypatch, reason):
    """Disabling the on-node gates requires a written reason on the record.

    ``enforced: false`` alone is not a waiver: the reason is prose, refused
    rather than coerced, since ``str(False)`` is the non-empty string 'False'.
    """
    d = os.path.join(run_dir, "pretrain", "deep_3x16")
    with open(os.path.join(d, "fidelity_certificate.json"), "w") as f:
        json.dump({"verdict": "FAIL", "arch": "deep_3x16", "enforced": False,
                   "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                                  "override_reason": reason},
                   "summary": {"max_atom_mHa": 13.7,
                               "max_dAE_kcalmol": 25.7}}, f)
    monkeypatch.setattr(tt, "_run_worker", lambda s, d: (0, "ok"))
    assert tt.main([run_dir, "0"]) == 3
    assert _read_failure(run_dir, 0)["classification"] == \
        "fidelity_certificate_failed"


def test_manifest_without_a_cell_arch_is_refused_not_waved_through(tmp_path,
                                                                   monkeypatch):
    """A manifest with no arch for this index makes the certificate
    unresolvable; an unresolvable certificate is a refusal, never a pass."""
    d = tmp_path / "run"
    d.mkdir()
    with open(d / "manifest.json", "w") as f:
        json.dump({"width": 4, "n_specs": 1}, f)
    _write_spec(str(d), 0)
    monkeypatch.setattr(tt, "_run_worker", lambda s, dev: (0, "ok"))
    assert tt.main([str(d), "0"]) == 3
    failure = _read_failure(str(d), 0)
    assert failure["classification"] == "fidelity_certificate_missing"
    assert failure["arch"] is None


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


def test_unenforced_but_MISSING_certificate_is_still_refused(run_dir,
                                                             monkeypatch):
    """Enforcement can only be waived by a certificate that exists to record
    the waiver; an absent one waives nothing."""
    os.remove(os.path.join(run_dir, "pretrain", "deep_3x16",
                           "fidelity_certificate.json"))
    monkeypatch.setattr(tt, "_run_worker", lambda s, d: (0, "ok"))
    assert tt.main([run_dir, "0"]) == 3


def test_passing_certificate_lets_the_worker_run(run_dir, monkeypatch):
    def fake_worker(spec_path, device):
        _write_model(run_dir, 0)
        return 0, "ok"
    monkeypatch.setattr(tt, "_run_worker", fake_worker)
    assert tt.main([run_dir, "0"]) == 0


def test_precompute_failed_species_marker_still_wins(run_dir, monkeypatch):
    """The preflight's precise diagnosis is preserved: it exits 0 BEFORE the
    fidelity gate, so a spec already marked unbuildable is not relabelled."""
    ck = os.path.join(run_dir, "checkpoints", "spec_0000")
    os.makedirs(ck, exist_ok=True)
    with open(os.path.join(ck, "failure.json"), "w") as f:
        json.dump({"classification": "precompute_failed_species", "rc": 0}, f)
    os.remove(os.path.join(run_dir, "pretrain", "deep_3x16",
                           "fidelity_certificate.json"))
    monkeypatch.setattr(tt, "_run_worker", lambda s, d: (0, "ok"))
    assert tt.main([run_dir, "0"]) == 0
    assert _read_failure(run_dir, 0)["classification"] == \
        "precompute_failed_species"


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


def _refusal_record(run_dir, monkeypatch, documents):
    """``(rc, failure record, documents served)`` for one ``main`` call."""
    path = os.path.join(run_dir, "pretrain", "deep_3x16",
                        "fidelity_certificate.json")
    served = _serve_documents(monkeypatch, path, list(documents))
    monkeypatch.setattr(tt, "_run_worker", lambda s, d: (0, "ok"))
    try:
        rc = tt.main([run_dir, "0"])
    finally:
        monkeypatch.undo()
    return rc, _read_failure(run_dir, 0), served


def test_the_refusal_record_describes_one_certificate(run_dir, monkeypatch):
    """A refusal record corresponds to a document, not to a sequence of them.

    The gate decided on one parse and the classification was taken from a
    second, so a certificate rewritten between the two opens assembled the
    record out of both files: the verdict and the excerpt of the document the
    gate read beside the status of the document the classifier read. Three
    documents, each refused on its own and each producing a different record,
    must reproduce the record of the FIRST -- on one read -- in every order
    they are served in.
    """
    documents = (_R1, _R2, _R3)
    alone = []
    for doc in documents:
        rc, record, served = _refusal_record(run_dir, monkeypatch, [doc])
        assert rc == 3
        assert len(served) == 1, (doc, served)
        alone.append(record)
    assert len({json.dumps(r, sort_keys=True) for r in alone}) == 3, alone
    for order in itertools.permutations(range(len(documents))):
        rc, record, served = _refusal_record(
            run_dir, monkeypatch, [documents[i] for i in order])
        assert rc == 3
        assert len(served) == 1, (order, served)
        assert record == alone[order[0]], (order, record)


def test_a_certificate_rewritten_to_pass_cannot_relabel_the_refusal(
        run_dir, monkeypatch):
    """The status the record states is the one the refusal was decided on.

    With a FAIL read by the gate and a PASS served to every later read, the
    record stated ``certificate_status: "PASS"`` under the absent-certificate
    classification with an excerpt naming the FAIL -- a record no single
    document produces, and one that reads as a passing certificate refused
    for being missing.
    """
    rc, record, served = _refusal_record(run_dir, monkeypatch,
                                         [_R1, _R_PASS])
    assert rc == 3
    assert len(served) == 1, served
    assert record["classification"] == "fidelity_certificate_failed"
    assert record["certificate_status"] == "FAIL"
    assert "13.7" in record["log_excerpt"]


def test_read_cell_arch_resolves_the_index(run_dir):
    assert tt._read_cell_arch(run_dir, 0) == "deep_3x16"
    assert tt._read_cell_arch(run_dir, 99) is None


def test_certificate_classifications_are_deterministic_not_retryable():
    """A blind resubmit cannot make an absent or failed certificate pass, so
    neither classification may enter the retry set."""
    from xcquinox.alec.cluster.__main__ import _RETRYABLE
    assert "fidelity_certificate_missing" not in _RETRYABLE
    assert "fidelity_certificate_failed" not in _RETRYABLE


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_missing_manifest_refuses_with_exit_2_and_the_cause(tmp_path, capsys):
    """The structural refusal must actually EXECUTE: a run dir without a
    readable manifest.json exits 2 with the cause named on stdout. The
    first-landed version of this path crashed on its own log call
    (TypeError: _log() missing 'message'), mapping to the unhandled exit 1
    and burning attempt_cap resubmits -- a refusal path that had never run
    once."""
    empty = tmp_path / "no_manifest_run"
    empty.mkdir()
    rc = tt.main([str(empty), "0"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "cannot read manifest width" in out
    assert "repair-manifest" in out

    corrupt = tmp_path / "bad_manifest_run"
    corrupt.mkdir()
    (corrupt / "manifest.json").write_text("{not json")
    rc2 = tt.main([str(corrupt), "3"])
    assert rc2 == 2
    assert "cannot read manifest width" in capsys.readouterr().out
