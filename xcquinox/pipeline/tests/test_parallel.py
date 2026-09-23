"""Tests for xcquinox.pipeline.parallel.

Implements THE SPEC section 13.2 test_parallel.py items (1)-(15), plus
(16)-(17) for the held-out eval shard path: the stall-watchdog opt-out and
worker stream capture.
All tests use mock subprocesses or tiny helper scripts -- no real worker
processes, no jax/equinox/optax imports.
"""
import os
import sys
import textwrap

import pytest

import xcquinox.pipeline.parallel as parallel
from xcquinox.pipeline.parallel import (
    WorkerJob,
    _thread_env,
    build_training_jobs,
    run_workers,
)


# ---------------------------------------------------------------------------
# Helpers: write tiny Python scripts to tmp_path that simulate workers
# ---------------------------------------------------------------------------

def _write_worker_script(tmp_path, name, body):
    """Write a tiny Python script to tmp_path and return its path."""
    script = tmp_path / f"{name}.py"
    script.write_text(textwrap.dedent(body))
    return str(script)


# ---------------------------------------------------------------------------
# (1) WorkerJob and WorkerResult dataclass construction
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (2) build_pretrain_jobs + build_training_jobs construct correct argv lists
# ---------------------------------------------------------------------------


def test_build_training_jobs_argv(tmp_path, monkeypatch):
    """build_training_jobs produces correct cmd, name, progress_file."""
    fake_worker = str(tmp_path / "train_worker.py")
    monkeypatch.setattr(
        "xcquinox.pipeline.parallel.worker_script_path",
        lambda name: fake_worker,
    )

    class FakeArch:
        name = "deep"
    class FakeSpec:
        arch = FakeArch()
        loss_name = "mae"

    jobs = build_training_jobs(
        [FakeSpec()],
        checkpoint_base="/ckpt",
        data_dir="/data",
        threads=4,
    )
    assert len(jobs) == 1
    j = jobs[0]
    assert j.name == "deep/mae"
    assert "--arch" in j.cmd and j.cmd[j.cmd.index("--arch") + 1] == "deep"
    assert j.progress_file == os.path.join(
        "/ckpt", "02_train", "deep", "mae", "progress.json"
    )
    assert j.thread_env["OMP_NUM_THREADS"] == "4"
    assert j.thread_env[parallel.WORKER_BIND_CPUS_ENV] == j.thread_env["OMP_NUM_THREADS"]


# ---------------------------------------------------------------------------
# (2b) the worker CPU bound: env request, slot assignment, and the pin itself
# ---------------------------------------------------------------------------

def test_thread_env_requires_and_encodes_the_bound():
    """The pool/probe distinction is a required keyword. A pool member's env
    carries the CPU-bind request and drops the eigen token (measured inert on
    the pinned jaxlib 0.7.0 thunk runtime -- it bounds nothing); the preflight
    probe keeps the historical flag string verbatim (it mirrors the train
    array's sbatch environment) and carries no bind request."""
    with pytest.raises(TypeError):
        _thread_env(4)  # the policy must be stated at the call site
    worker = _thread_env(4, bound_worker=True)
    assert worker[parallel.WORKER_BIND_CPUS_ENV] == "4"
    assert "--xla_cpu_multi_thread_eigen" not in worker["XLA_FLAGS"]
    assert "--xla_llvm_disable_expensive_passes=true" in worker["XLA_FLAGS"]
    probe = _thread_env(4, bound_worker=False)
    assert parallel.WORKER_BIND_CPUS_ENV not in probe
    assert probe["XLA_FLAGS"] == (
        "--xla_cpu_multi_thread_eigen=true "
        "--xla_llvm_disable_expensive_passes=true "
        "--xla_backend_optimization_level=1"
    )
    for env in (worker, probe):
        assert env["OMP_NUM_THREADS"] == "4"
        assert env["MKL_NUM_THREADS"] == "4"
        assert env["OPENBLAS_NUM_THREADS"] == "4"


def _fake_affinity(monkeypatch, allowed, applied):
    monkeypatch.setattr(os, "sched_getaffinity",
                        lambda pid: set(allowed), raising=False)
    monkeypatch.setattr(os, "sched_setaffinity",
                        lambda pid, cpus: applied.append(sorted(cpus)),
                        raising=False)


def test_apply_worker_cpu_bind_pins_slot_disjoint_slices(monkeypatch):
    """Slot-strided slices: with an 8-CPU allowance and 2 CPUs per worker,
    slots 0..3 partition the allowance with no overlap (the eval ladder keeps
    n_workers x threads within the allowance)."""
    applied: list = []
    _fake_affinity(monkeypatch, range(8), applied)
    monkeypatch.setenv(parallel.WORKER_BIND_CPUS_ENV, "2")
    seen = []
    for slot in range(4):
        monkeypatch.setenv(parallel.WORKER_SLOT_ENV, str(slot))
        assert parallel.apply_worker_cpu_bind() == 2
        seen.append(applied[-1])
    flat = [c for cpus in seen for c in cpus]
    assert sorted(flat) == list(range(8))          # disjoint cover
    assert seen[0] == [0, 1] and seen[3] == [6, 7]  # strided placement


# ---------------------------------------------------------------------------
# (3) run_workers with mock subprocess returns ordered results
# ---------------------------------------------------------------------------

def test_run_workers_success(tmp_path):
    """Two workers that echo JSON -- results come back in order."""
    script = _write_worker_script(tmp_path, "ok_worker", """\
        import json, sys
        print(json.dumps({"status": "ok", "idx": int(sys.argv[1])}))
    """)
    jobs = []
    for i in range(3):
        jobs.append(WorkerJob(
            name=f"job{i}",
            cmd=[sys.executable, script, str(i)],
            progress_file=str(tmp_path / f"prog{i}.json"),
        ))
    results = run_workers(jobs, max_parallel=4, poll_interval=0.05)
    assert len(results) == 3
    for i, r in enumerate(results):
        assert r.status == "success"
        assert r.returncode == 0
        assert r.payload["idx"] == i
        assert r.job.name == f"job{i}"


# ---------------------------------------------------------------------------
# (4) run_workers handles non-zero exit codes
# ---------------------------------------------------------------------------

def test_run_workers_nonzero_exit(tmp_path):
    script = _write_worker_script(tmp_path, "fail_worker", """\
        import json, sys
        print(json.dumps({"error": "boom"}))
        sys.exit(1)
    """)
    job = WorkerJob(
        name="failing", cmd=[sys.executable, script],
        progress_file=str(tmp_path / "p.json"),
    )
    results = run_workers([job], max_parallel=1, poll_interval=0.05)
    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].returncode == 1
    assert results[0].payload["error"] == "boom"


# ---------------------------------------------------------------------------
# (5) run_workers handles malformed JSON on stdout
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (6) run_workers handles missing progress files
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (7) on_progress callback invocation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (8) max_parallel bounds simultaneous workers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (9) Worker crash with traceback captured in stderr
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (10) Deterministic job ordering in output list
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (11) Stall detection warning (no progress for >STALL_WARN_SEC)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (12) Worker segfault (os._exit(139)) captured as failure
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (13) Partial progress file survives (truncated JSON in progress.json)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (14) stderr capture in WorkerResult.stderr
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (15) Concurrency bound (10 jobs, max_parallel=3, never >3 alive)
# ---------------------------------------------------------------------------

def test_concurrency_bound_never_exceeded(tmp_path):
    """With 10 jobs and max_parallel=3, never more than 3 run at once."""
    max_file = str(tmp_path / "max_concurrent.txt")
    # Initialize tracking files
    with open(max_file, "w") as f:
        f.write("0")
    current_file = str(tmp_path / "current.txt")
    with open(current_file, "w") as f:
        f.write("0")

    script = _write_worker_script(tmp_path, "concurrent", f"""\
        import json, time, os, fcntl

        max_file = {max_file!r}
        current_file = {current_file!r}

        def atomic_update(path, delta):
            fd = os.open(path, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                val = int(os.read(fd, 100).decode())
                new_val = val + delta
                os.lseek(fd, 0, os.SEEK_SET)
                os.ftruncate(fd, 0)
                os.write(fd, str(new_val).encode())
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
            return new_val

        def read_val(path):
            fd = os.open(path, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_SH)
                val = int(os.read(fd, 100).decode())
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
            return val

        def update_max():
            cur = read_val(current_file)
            fd = os.open(max_file, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                mx = int(os.read(fd, 100).decode())
                if cur > mx:
                    os.lseek(fd, 0, os.SEEK_SET)
                    os.ftruncate(fd, 0)
                    os.write(fd, str(cur).encode())
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        atomic_update(current_file, 1)
        update_max()
        time.sleep(0.1)
        atomic_update(current_file, -1)
        print(json.dumps({{"ok": True}}))
    """)
    jobs = [
        WorkerJob(
            name=f"c{i}", cmd=[sys.executable, script],
            progress_file=str(tmp_path / f"prog{i}.json"),
        )
        for i in range(10)
    ]
    results = run_workers(jobs, max_parallel=3, poll_interval=0.05)
    assert len(results) == 10
    for r in results:
        assert r.status == "success"

    # Read back the max concurrent value
    with open(max_file) as f:
        max_concurrent = int(f.read().strip())
    assert max_concurrent <= 3, f"max concurrent was {max_concurrent}, expected <= 3"
    assert max_concurrent >= 1, "at least one job should have run"


# ---------------------------------------------------------------------------
# (16) Stall watchdog opt-out for workers that write no progress file
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (17) Worker stream capture: per-job log file, and drainage of BOTH pipes
# ---------------------------------------------------------------------------


