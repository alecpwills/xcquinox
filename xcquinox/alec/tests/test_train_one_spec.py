"""Tests for the ``python -m xcquinox.alec._train_one_spec`` CLI worker.

The worker must be able to route JAX to CPU before any JAX import so the
parent notebook can gracefully fall back when GPU training OOMs on small
hardware (the step6 72-spec sweep hits ~7 GB peak on 8 GB GPUs).
"""
import dataclasses
import json
import os
import subprocess
import sys
import tempfile


def _base_env():
    """Return an env dict that mimics a notebook launch, i.e. free of the
    conftest.py session-autouse JAX_PLATFORMS=cpu leak. Without this scrub
    the CPU-routing tests would pass vacuously because the subprocess would
    inherit cpu routing from the pytest session itself."""
    env = dict(os.environ)
    env.pop("JAX_PLATFORMS", None)
    return env


def _run_worker(args, timeout=60, env_overrides=None):
    """Invoke the CLI worker in a subprocess and return (returncode, stdout)."""
    env = _base_env()
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-m", "xcquinox.alec._train_one_spec", *args],
        capture_output=True, text=True, timeout=timeout, env=env,
    )
    return proc.returncode, proc.stdout


def _parse_init_line(stdout):
    """Return the first {"kind": "init"} JSON message emitted at worker start."""
    for line in stdout.splitlines():
        if line.startswith('{"kind": "init"'):
            return json.loads(line)
    return None


def test_parent_env_jax_platforms_cpu_routes_worker_to_cpu(tmp_path):
    """The parent-set JAX_PLATFORMS=cpu env var must actually land the
    worker on CPU. Necessary because `python -m xcquinox.alec._train_one_spec`
    imports xcquinox.alec's package __init__ BEFORE main() runs, which in
    turn imports jax.numpy via descriptors.py: too late for any in-process
    env fiddling inside main() to take effect. The only reliable switch is
    the env at subprocess-launch time.

    NOTE: this test scrubs JAX_PLATFORMS from the inherited environment
    before re-setting it to 'cpu' so the pytest-session conftest cannot
    supply a false positive."""
    bogus = tmp_path / "no_such.spec"
    rc, stdout = _run_worker(
        [str(bogus), "--device=cpu", "--no-progress"],
        env_overrides={"JAX_PLATFORMS": "cpu"},
    )
    init = _parse_init_line(stdout)
    assert init is not None, (
        f"worker did not emit init JSON line; stdout={stdout!r}"
    )
    assert init["jax_platform"] == "cpu", (
        f"expected jax_platform=cpu, got {init!r}. Parent must set "
        f"JAX_PLATFORMS=cpu in the subprocess env; --device=cpu alone is "
        f"too late because the package-level jax import runs first."
    )
    assert init["requested_device"] == "cpu"


def test_device_cpu_flag_alone_is_insufficient_when_gpu_visible(tmp_path):
    """Regression test for the actual bug the user hit. With no env-level
    routing (parent env has no JAX_PLATFORMS), the --device=cpu CLI flag is
    too late to redirect JAX because `python -m xcquinox.alec._train_one_spec`
    imports xcquinox.alec's __init__ (which pulls in jax) BEFORE main()
    runs. This test documents that `--device=cpu` alone reaches GPU when a
    GPU is present, so the parent cell 17 MUST supply the env override."""
    import importlib.util
    has_gpu_jax = False
    try:
        import subprocess as _sp
        probe = _sp.run(
            [sys.executable, "-c", "import jax; print(jax.default_backend())"],
            capture_output=True, text=True, timeout=30,
            env={k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"},
        )
        has_gpu_jax = probe.stdout.strip() == "gpu"
    except Exception:
        pass
    if not has_gpu_jax:
        import pytest as _pytest
        _pytest.skip("no GPU-capable JAX; regression only manifests on GPU hosts")

    bogus = tmp_path / "no_such.spec"
    # Run with env scrubbed of JAX_PLATFORMS. --device=cpu alone SHOULD fail
    # to route because xcquinox.alec's __init__ has already imported jax.
    rc, stdout = _run_worker(
        [str(bogus), "--device=cpu", "--no-progress"],
        env_overrides=None,  # No JAX_PLATFORMS set by parent
    )
    init = _parse_init_line(stdout)
    assert init is not None
    # On a GPU host, --device=cpu without parent env override lands on GPU.
    # This is the bug. The notebook cell must set env to avoid it.
    assert init["jax_platform"] == "gpu", (
        f"regression documented: on a GPU host, --device=cpu without parent "
        f"env override is expected to route to GPU (init={init!r}). If this "
        f"assertion now fails, the worker's late JAX_PLATFORMS setting has "
        f"somehow become effective -- update the notebook logic accordingly."
    )


def test_device_auto_is_default(tmp_path):
    """Omitting --device defaults to 'auto', which leaves JAX_PLATFORMS
    unset so JAX picks whatever backend is available (typically GPU)."""
    bogus = tmp_path / "no_such.spec"
    rc, stdout = _run_worker([str(bogus), "--no-progress"])
    init = _parse_init_line(stdout)
    assert init is not None, f"init line missing; stdout={stdout!r}"
    assert init["requested_device"] == "auto"


def test_device_invalid_value_is_rejected(tmp_path):
    """argparse choices must reject unknown device strings."""
    bogus = tmp_path / "no_such.spec"
    rc, stdout = _run_worker([str(bogus), "--device=tpu"], timeout=30)
    # argparse exits with code 2 on invalid choice.
    assert rc == 2, f"expected rc=2 for invalid --device, got {rc}"


def test_pad_group_flag_is_accepted(tmp_path):
    """--pad-group is a recognized CLI flag: it turns on the standalone padding
    pass for the loaded spec so an existing spec can be smoke-probed with padding
    without rebuilding it. Verified via arg-parsing -- with a bogus spec the worker
    still emits its init line (args parsed) before failing to load the spec; an
    unknown flag would make argparse exit 2 with no init line."""
    bogus = tmp_path / "no_such.spec"
    rc, stdout = _run_worker(
        [str(bogus), "--device=cpu", "--pad-group", "--smoke", "--no-progress"],
        env_overrides={"JAX_PLATFORMS": "cpu"},
    )
    init = _parse_init_line(stdout)
    assert init is not None, f"--pad-group not accepted (no init line); stdout={stdout!r}"
    assert init["requested_device"] == "cpu"


def test_worker_enables_jax_x64_by_default(tmp_path):
    """The training subprocess MUST run with ``jax_enable_x64=True``.

    Every other entry point in the codebase enables float64 (notebook
    cell 0, conftest.py, workers/{pretrain,train,test}_worker.py); this
    worker was the only entry point that silently inherited JAX's
    float32 default, producing degraded convergence on long training
    runs (loss values legitimately below 1e-7 lose precision in fp32,
    and gradients of such small losses are at machine epsilon).

    The fix sets ``JAX_ENABLE_X64=1`` as an env var BEFORE the first
    ``import jax`` (the only universally reliable switch, the
    ``jax.config.update`` path can be too late when third-party
    importers like equinox or pyscfad have already cached defaults),
    plus a defensive ``jax.config.update("jax_enable_x64", True)``
    after the import.

    This test verifies the contract end-to-end: launch the worker WITH
    NO x64 hint in the parent environment and assert the init JSON
    payload reports ``jax_enable_x64=True``. A regression of either
    the env-var setdefault or the post-import update would fail this
    test even on systems where one of the two paths works alone.
    """
    bogus = tmp_path / "no_such.spec"
    env_overrides = {}
    # Scrub any inherited JAX_ENABLE_X64 so we test the worker's own
    # default-setting logic, not parent-process leakage.
    env = _base_env()
    env.pop("JAX_ENABLE_X64", None)
    proc = subprocess.run(
        [sys.executable, "-m", "xcquinox.alec._train_one_spec",
         str(bogus), "--no-progress"],
        capture_output=True, text=True, timeout=60, env=env,
    )
    init = _parse_init_line(proc.stdout)
    assert init is not None, (
        f"worker did not emit init JSON line; stdout={proc.stdout!r}"
    )
    assert init.get("jax_enable_x64") is True, (
        f"worker MUST enable float64 by default; init={init!r}. "
        f"A regression in _train_one_spec.main() to drop the "
        f"``os.environ.setdefault('JAX_ENABLE_X64', '1')`` line OR the "
        f"post-import ``jax.config.update('jax_enable_x64', True)`` "
        f"call would let JAX fall back to float32, silently degrading "
        f"convergence on long training runs."
    )


# ---------------------------------------------------------------------------
# WS5 (2026-06-20): SIGTERM flush. The worker installs a handler that, on the
# SLURM wall-clock pre-kill SIGTERM, calls the per_molecule loop's registered
# resume flusher (best-effort) before exiting 143, so an in-flight epoch is
# checkpointed even between periodic writes.
# ---------------------------------------------------------------------------

def test_flush_on_signal_calls_registered_flusher():
    """_flush_on_signal invokes the resume flusher registered in train."""
    from xcquinox.alec import _train_one_spec as worker
    from xcquinox.alec import train as train_mod
    train_mod._clear_resume_flusher()
    calls = []
    train_mod._register_resume_flusher(lambda: calls.append("flushed"))
    try:
        worker._flush_on_signal()
        assert calls == ["flushed"]
    finally:
        train_mod._clear_resume_flusher()


def test_flush_on_signal_is_best_effort_when_flusher_raises():
    """A failing flush must NOT propagate (periodic checkpoints are the primary
    net); _flush_on_signal swallows the error."""
    from xcquinox.alec import _train_one_spec as worker
    from xcquinox.alec import train as train_mod
    train_mod._clear_resume_flusher()

    def _boom():
        raise RuntimeError("disk full")
    train_mod._register_resume_flusher(_boom)
    try:
        worker._flush_on_signal()   # must not raise
    finally:
        train_mod._clear_resume_flusher()


def test_flush_on_signal_noop_when_no_flusher_registered():
    """With no flusher registered (checkpoint_every=0 / not in the loop yet),
    _flush_on_signal is a silent no-op."""
    from xcquinox.alec import _train_one_spec as worker
    from xcquinox.alec import train as train_mod
    train_mod._clear_resume_flusher()
    worker._flush_on_signal()        # must not raise


def test_sigterm_handler_flushes_then_exits_143():
    """The installed SIGTERM handler flushes the registered resume checkpoint
    and exits 143 (128+15)."""
    import signal
    import pytest
    from xcquinox.alec import _train_one_spec as worker
    from xcquinox.alec import train as train_mod
    train_mod._clear_resume_flusher()
    calls = []
    train_mod._register_resume_flusher(lambda: calls.append(1))
    handler = worker._install_sigterm_flush_handler()
    assert signal.getsignal(signal.SIGTERM) is handler
    try:
        with pytest.raises(SystemExit) as ei:
            handler(signal.SIGTERM, None)
        assert ei.value.code == 143
        assert calls == [1]          # flushed before exiting
    finally:
        train_mod._clear_resume_flusher()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def test_progress_callback_includes_rss(capsys):
    """The per-step JSON progress payload carries the worker's current RSS and
    high-water mark so the parent heartbeat can surface live memory."""
    import math as _math

    from xcquinox.alec import _train_one_spec as worker_mod

    worker_mod._progress_callback(
        {"arch": "a", "phase": "train", "step": 1, "total": 2, "loss": 0.5})
    line = capsys.readouterr().out.strip()
    payload = json.loads(line)
    assert payload["kind"] == "step"
    assert isinstance(payload["rss_gb"], float)
    assert isinstance(payload["hwm_gb"], float)
    if sys.platform.startswith("linux"):
        assert _math.isfinite(payload["rss_gb"]) and payload["rss_gb"] > 0.0
        assert payload["hwm_gb"] >= payload["rss_gb"]


# ---------------------------------------------------------------------------
# --smoke leaves no throwaway checkpoint directory behind
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _SmokeProbeSpec:
    """A replaceable stand-in for ``TrainingSpec``, carrying only the two
    fields ``--smoke`` overrides.

    The property under test is the DISPOSAL of the throwaway checkpoint
    directory, not the training: a real spec would put a one-epoch compile of
    every per-molecule kernel between the directory being made and the exit,
    which measures nothing the disposal depends on. The two fields are the
    ones ``main`` requires of the loaded spec (``n_steps`` and
    ``checkpoint_dir``, both checked there against
    ``dataclasses.fields``), so this reaches the smoke branch by the same
    route a real spec does.
    """

    n_steps: int = 5
    checkpoint_dir: str = ""


def _write_probe_spec(tmp_path):
    """Serialize a :class:`_SmokeProbeSpec` the way the parent writes a spec."""
    import pickle

    path = tmp_path / "probe.spec"
    with open(path, "wb") as f:
        pickle.dump(_SmokeProbeSpec(), f)
    return path


def _parse_smoke_line(stdout):
    """The ``{"kind": "smoke"}`` line, which names the throwaway directory."""
    for line in stdout.splitlines():
        if line.startswith('{"kind": "smoke"'):
            return json.loads(line)
    return None


def test_smoke_temp_dir_is_removed_when_main_returns(tmp_path, monkeypatch):
    """``--smoke`` redirects the checkpoints into a temp dir that must not leak.

    The directory holds a one-epoch checkpoint of whatever cell was probed and
    is created once per compile-smoke run (``hpcjobs/dfs6311_smoke_vma.sbatch``
    is the live call site). Disposal has to be part of ``main``'s own control
    flow rather than a teardown handler, because the worker leaves through
    ``os._exit`` (see ``xcquinox/alec/cluster/_exit.py``), which runs none.
    """
    import xcquinox.alec as alec
    from xcquinox.alec import _train_one_spec as worker_mod

    # Both halves are needed in-process. ``tempfile.gettempdir`` caches the
    # root it resolves on first use in ``tempfile.tempdir``, so TMPDIR alone
    # redirects nothing once anything in the session has already made a
    # temporary file: the directory lands in the real /tmp, and a containment
    # assertion against ``tmp_path`` then reads an empty glob of a path
    # nothing was ever written to. Measured: the smoke directory was created
    # at /tmp/xcq_smoke_* with TMPDIR set to this test's own directory.
    # ``monkeypatch`` restores the cached value afterwards.
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    seen = {}

    def fake_run_training(spec, progress_callback=None):
        # The directory must still be there while the training runs -- that is
        # what it is for -- so the disposal is measured after main returns.
        seen["checkpoint_dir"] = spec.checkpoint_dir
        seen["present_during_training"] = os.path.isdir(spec.checkpoint_dir)
        seen["n_steps"] = spec.n_steps

    monkeypatch.setattr(alec, "run_training", fake_run_training)
    rc = worker_mod.main([str(_write_probe_spec(tmp_path)),
                          "--smoke", "--no-progress"])

    assert rc == 0
    assert seen["present_during_training"] is True
    assert seen["n_steps"] == 1
    # Under the test's own directory, so the containment assertion below is a
    # statement about the directory the worker actually made rather than about
    # an empty glob of a path nothing was ever written to.
    assert seen["checkpoint_dir"].startswith(str(tmp_path)), (
        f"the smoke directory landed outside the test's own directory: "
        f"{seen['checkpoint_dir']}")
    assert not os.path.exists(seen["checkpoint_dir"]), (
        f"--smoke left {seen['checkpoint_dir']} on disk")
    assert not sorted(tmp_path.glob("xcq_smoke_*"))


def test_smoke_temp_dir_is_removed_through_the_hard_exit(tmp_path):
    """The same disposal in the process the sbatch script actually launches.

    ``python -m xcquinox.alec._train_one_spec`` leaves through the shared hard
    exit, so a disposal registered with ``atexit`` never runs: measured on this
    launch, the directory survives the worker. The probe spec fails inside
    ``run_training``, which is the harder path -- the disposal owes the
    directory on a failure too, and the exception still reaches the caller
    with the interpreter's own status 1.
    """
    rc, stdout = _run_worker(
        [str(_write_probe_spec(tmp_path)), "--smoke", "--no-progress",
         "--device=cpu"],
        timeout=180,
        env_overrides={"JAX_PLATFORMS": "cpu", "TMPDIR": str(tmp_path)},
    )
    smoke = _parse_smoke_line(stdout)
    assert smoke is not None, f"no smoke line; stdout={stdout!r}"
    assert rc == 1, f"expected the escaping exception's status 1, got {rc}"
    assert not os.path.exists(smoke["checkpoint_dir"]), (
        f"--smoke left {smoke['checkpoint_dir']} on disk")
    assert not sorted(tmp_path.glob("xcq_smoke_*"))
