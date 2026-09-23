"""Tests for the ``python -m xcquinox.pipeline._train_one_spec`` CLI worker.

The worker must be able to route JAX to CPU before any JAX import so the
parent notebook can gracefully fall back when GPU training OOMs on small
hardware (the step6 72-spec sweep hits ~7 GB peak on 8 GB GPUs).
"""
import dataclasses
import json
import os
import subprocess
import sys


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
        [sys.executable, "-m", "xcquinox.pipeline._train_one_spec", *args],
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
    worker on CPU. Necessary because `python -m xcquinox.pipeline._train_one_spec`
    imports xcquinox.pipeline's package __init__ BEFORE main() runs, which in
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
        [sys.executable, "-m", "xcquinox.pipeline._train_one_spec",
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


def test_sigterm_handler_flushes_then_exits_143():
    """The installed SIGTERM handler flushes the registered resume checkpoint
    and exits 143 (128+15)."""
    import signal
    import pytest
    from xcquinox.pipeline import _train_one_spec as worker
    from xcquinox.pipeline import train as train_mod
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


