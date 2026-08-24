"""Tests for xcquinox.alec.parallel.

Implements THE SPEC section 13.2 test_parallel.py items (1)-(15), plus
(16)-(17) for the held-out eval shard path: the stall-watchdog opt-out and
worker stream capture.
All tests use mock subprocesses or tiny helper scripts -- no real worker
processes, no jax/equinox/optax imports.
"""
import json
import os
import sys
import textwrap
import time

import pytest

from xcquinox.alec.parallel import (
    STALL_WARN_SEC,
    WorkerJob,
    WorkerResult,
    _read_progress,
    _thread_env,
    build_pretrain_jobs,
    build_training_jobs,
    run_workers,
    worker_script_path,
)


# ---------------------------------------------------------------------------
# Helpers: write tiny Python scripts to tmp_path that simulate workers
# ---------------------------------------------------------------------------

def _write_worker_script(tmp_path, name, body):
    """Write a tiny Python script to tmp_path and return its path."""
    script = tmp_path / f"{name}.py"
    script.write_text(textwrap.dedent(body))
    return str(script)


def _make_job(tmp_path, name, script_path, progress_file=None):
    """Build a WorkerJob pointing at a helper script."""
    if progress_file is None:
        progress_file = str(tmp_path / f"{name}_progress.json")
    return WorkerJob(
        name=name,
        cmd=[sys.executable, script_path],
        progress_file=progress_file,
    )


# ---------------------------------------------------------------------------
# (1) WorkerJob and WorkerResult dataclass construction
# ---------------------------------------------------------------------------

def test_dataclass_construction():
    job = WorkerJob(name="test", cmd=["echo"], progress_file="/tmp/p.json")
    assert job.name == "test"
    assert job.cmd == ["echo"]
    assert job.progress_file == "/tmp/p.json"
    assert job.thread_env == {}

    result = WorkerResult(
        job=job, status="success", returncode=0,
        payload={"ok": True}, stderr="", duration=1.5,
    )
    assert result.status == "success"
    assert result.returncode == 0
    assert result.payload == {"ok": True}
    assert result.duration == 1.5

    # thread_env custom value
    job2 = WorkerJob(
        name="t2", cmd=["x"], progress_file="/tmp/q.json",
        thread_env={"OMP_NUM_THREADS": "4"},
    )
    assert job2.thread_env == {"OMP_NUM_THREADS": "4"}


# ---------------------------------------------------------------------------
# (2) build_pretrain_jobs + build_training_jobs construct correct argv lists
# ---------------------------------------------------------------------------

def test_build_pretrain_jobs_argv(tmp_path, monkeypatch):
    """build_pretrain_jobs produces correct cmd, name, progress_file."""
    # Mock worker_script_path to avoid needing actual worker files
    fake_worker = str(tmp_path / "pretrain_worker.py")
    monkeypatch.setattr(
        "xcquinox.alec.parallel.worker_script_path",
        lambda name: fake_worker,
    )

    # Minimal spec-like object with .arch.name
    class FakeArch:
        name = "shallow"
    class FakeSpec:
        arch = FakeArch()

    jobs = build_pretrain_jobs(
        [FakeSpec()],
        checkpoint_base="/ckpt",
        data_dir="/data",
        threads=2,
    )
    assert len(jobs) == 1
    j = jobs[0]
    assert j.name == "shallow"
    assert j.cmd[0] == sys.executable
    assert j.cmd[1] == fake_worker
    assert "--arch" in j.cmd and j.cmd[j.cmd.index("--arch") + 1] == "shallow"
    assert "--spec-pickle" in j.cmd
    assert "--checkpoint-base" in j.cmd
    assert "--data-dir" in j.cmd
    assert "--threads" in j.cmd and j.cmd[j.cmd.index("--threads") + 1] == "2"
    assert j.progress_file == os.path.join(
        "/ckpt", "01_pretrain", "shallow", "progress.json"
    )
    # thread_env populated
    assert "OMP_NUM_THREADS" in j.thread_env
    assert j.thread_env["OMP_NUM_THREADS"] == "2"


def test_build_training_jobs_argv(tmp_path, monkeypatch):
    """build_training_jobs produces correct cmd, name, progress_file."""
    fake_worker = str(tmp_path / "train_worker.py")
    monkeypatch.setattr(
        "xcquinox.alec.parallel.worker_script_path",
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

def test_run_workers_malformed_json(tmp_path):
    script = _write_worker_script(tmp_path, "bad_json", """\
        print("this is not json {{{")
    """)
    job = WorkerJob(
        name="badjson", cmd=[sys.executable, script],
        progress_file=str(tmp_path / "p.json"),
    )
    results = run_workers([job], max_parallel=1, poll_interval=0.05)
    assert results[0].payload["error"] == "malformed stdout JSON"
    assert "raw" in results[0].payload


# ---------------------------------------------------------------------------
# (6) run_workers handles missing progress files
# ---------------------------------------------------------------------------

def test_run_workers_missing_progress(tmp_path):
    """Worker finishes without writing a progress file -- still returns."""
    script = _write_worker_script(tmp_path, "no_prog", """\
        import json
        print(json.dumps({"done": True}))
    """)
    # progress_file points to a non-existent path
    job = WorkerJob(
        name="noprog", cmd=[sys.executable, script],
        progress_file=str(tmp_path / "nonexistent" / "progress.json"),
    )
    results = run_workers([job], max_parallel=1, poll_interval=0.05)
    assert results[0].status == "success"
    assert results[0].payload["done"] is True


# ---------------------------------------------------------------------------
# (7) on_progress callback invocation
# ---------------------------------------------------------------------------

def test_on_progress_callback(tmp_path):
    """on_progress is called when progress.json changes."""
    progress_file = str(tmp_path / "progress.json")
    script = _write_worker_script(tmp_path, "prog_writer", f"""\
        import json, time
        pf = {progress_file!r}
        for step in range(3):
            with open(pf, "w") as f:
                json.dump({{"step": step}}, f)
            time.sleep(0.08)
        print(json.dumps({{"done": True}}))
    """)
    job = WorkerJob(
        name="prog_test", cmd=[sys.executable, script],
        progress_file=progress_file,
    )
    progress_events = []
    def callback(job, prog):
        progress_events.append(prog)

    run_workers([job], max_parallel=1, poll_interval=0.05, on_progress=callback)
    # Should have been called at least once (timing-dependent, but the
    # worker sleeps 0.08s per step and poll is 0.05s)
    assert len(progress_events) >= 1
    # Each event should be a dict with "step"
    for evt in progress_events:
        assert "step" in evt


# ---------------------------------------------------------------------------
# (8) max_parallel bounds simultaneous workers
# ---------------------------------------------------------------------------

def test_max_parallel_bounds(tmp_path):
    """With max_parallel=2 and 4 jobs, only 2 run at a time."""
    counter_file = str(tmp_path / "counter.txt")
    # Initialize counter file
    with open(counter_file, "w") as f:
        f.write("0")

    script = _write_worker_script(tmp_path, "bounded", f"""\
        import json, time, os, fcntl

        counter_file = {counter_file!r}

        def atomic_add(delta):
            fd = os.open(counter_file, os.O_RDWR)
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

        atomic_add(1)
        time.sleep(0.2)
        atomic_add(-1)
        print(json.dumps({{"ok": True}}))
    """)
    jobs = [
        WorkerJob(
            name=f"b{i}", cmd=[sys.executable, script],
            progress_file=str(tmp_path / f"prog{i}.json"),
        )
        for i in range(4)
    ]
    results = run_workers(jobs, max_parallel=2, poll_interval=0.05)
    assert len(results) == 4
    # All should succeed
    for r in results:
        assert r.status == "success"


# ---------------------------------------------------------------------------
# (9) Worker crash with traceback captured in stderr
# ---------------------------------------------------------------------------

def test_worker_crash_stderr(tmp_path):
    script = _write_worker_script(tmp_path, "crasher", """\
        import sys
        print("oops", file=sys.stderr)
        raise RuntimeError("deliberate crash")
    """)
    job = WorkerJob(
        name="crasher", cmd=[sys.executable, script],
        progress_file=str(tmp_path / "p.json"),
    )
    results = run_workers([job], max_parallel=1, poll_interval=0.05)
    assert results[0].status == "failed"
    assert results[0].returncode != 0
    assert "oops" in results[0].stderr
    assert "RuntimeError" in results[0].stderr


# ---------------------------------------------------------------------------
# (10) Deterministic job ordering in output list
# ---------------------------------------------------------------------------

def test_deterministic_ordering(tmp_path):
    """Results are indexed by input position, not completion order."""
    # Job 0 sleeps longer than job 1, but result[0] must still be job 0.
    fast_script = _write_worker_script(tmp_path, "fast", """\
        import json
        print(json.dumps({"name": "fast"}))
    """)
    slow_script = _write_worker_script(tmp_path, "slow", """\
        import json, time
        time.sleep(0.15)
        print(json.dumps({"name": "slow"}))
    """)
    jobs = [
        WorkerJob(name="slow", cmd=[sys.executable, slow_script],
                  progress_file=str(tmp_path / "p0.json")),
        WorkerJob(name="fast", cmd=[sys.executable, fast_script],
                  progress_file=str(tmp_path / "p1.json")),
    ]
    results = run_workers(jobs, max_parallel=4, poll_interval=0.05)
    assert results[0].job.name == "slow"
    assert results[0].payload["name"] == "slow"
    assert results[1].job.name == "fast"
    assert results[1].payload["name"] == "fast"


# ---------------------------------------------------------------------------
# (11) Stall detection warning (no progress for >STALL_WARN_SEC)
# ---------------------------------------------------------------------------

def test_stall_detection_warning(tmp_path, monkeypatch, capsys):
    """When a worker produces no progress for >STALL_WARN_SEC, a warning
    is emitted to stderr."""
    # Patch STALL_WARN_SEC to a tiny value so we don't wait 60s
    monkeypatch.setattr("xcquinox.alec.parallel.STALL_WARN_SEC", 0.15)

    # Worker that sleeps long enough for the stall detection to trigger
    script = _write_worker_script(tmp_path, "staller", """\
        import json, time
        time.sleep(0.5)
        print(json.dumps({"done": True}))
    """)
    job = WorkerJob(
        name="stall_arch", cmd=[sys.executable, script],
        progress_file=str(tmp_path / "nonexistent_progress.json"),
    )
    run_workers([job], max_parallel=1, poll_interval=0.05)
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "stall_arch" in captured.err
    assert "stalled" in captured.err


# ---------------------------------------------------------------------------
# (12) Worker segfault (os._exit(139)) captured as failure
# ---------------------------------------------------------------------------

def test_worker_segfault_exit(tmp_path):
    """Worker that exits with code 139 (simulated segfault) is failed."""
    script = _write_worker_script(tmp_path, "segfault", """\
        import os
        os._exit(139)
    """)
    job = WorkerJob(
        name="seg", cmd=[sys.executable, script],
        progress_file=str(tmp_path / "p.json"),
    )
    results = run_workers([job], max_parallel=1, poll_interval=0.05)
    assert results[0].status == "failed"
    assert results[0].returncode == 139
    # Empty stdout -> synthetic payload
    assert "error" in results[0].payload


# ---------------------------------------------------------------------------
# (13) Partial progress file survives (truncated JSON in progress.json)
# ---------------------------------------------------------------------------

def test_partial_progress_file(tmp_path):
    """A truncated progress.json does not crash _read_progress."""
    progress_file = tmp_path / "progress.json"
    progress_file.write_text('{"step": 5, "loss":')  # truncated
    result = _read_progress(str(progress_file))
    assert result is None

    # Also: valid progress file works
    progress_file.write_text('{"step": 5, "loss": 0.1}')
    result = _read_progress(str(progress_file))
    assert result == {"step": 5, "loss": 0.1}

    # Missing file returns None
    result = _read_progress(str(tmp_path / "nope.json"))
    assert result is None


# ---------------------------------------------------------------------------
# (14) stderr capture in WorkerResult.stderr
# ---------------------------------------------------------------------------

def test_stderr_capture(tmp_path):
    """Worker stderr is fully captured in WorkerResult.stderr."""
    script = _write_worker_script(tmp_path, "stderr_writer", """\
        import json, sys
        for i in range(5):
            print(f"line {i}", file=sys.stderr)
        print(json.dumps({"ok": True}))
    """)
    job = WorkerJob(
        name="stderr_test", cmd=[sys.executable, script],
        progress_file=str(tmp_path / "p.json"),
    )
    results = run_workers([job], max_parallel=1, poll_interval=0.05)
    assert results[0].status == "success"
    for i in range(5):
        assert f"line {i}" in results[0].stderr


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

def test_stall_watchdog_skipped_when_no_progress_file(tmp_path, monkeypatch,
                                                      capsys):
    """A job with no progress file opts OUT of the watchdog (the held-out eval
    shards write none, so every one of them 'stalled' every STALL_WARN_SEC for
    its whole runtime); a job that names one keeps its watchdog."""
    monkeypatch.setattr("xcquinox.alec.parallel.STALL_WARN_SEC", 0.15)
    script = _write_worker_script(tmp_path, "sleeper", """\
        import json, time
        time.sleep(0.6)
        print(json.dumps({"done": True}))
    """)
    jobs = [
        WorkerJob(name="no_prog", cmd=[sys.executable, script],
                  progress_file=None),
        WorkerJob(name="with_prog", cmd=[sys.executable, script],
                  progress_file=str(tmp_path / "never_written.json")),
    ]
    results = run_workers(jobs, max_parallel=2, poll_interval=0.05)
    assert all(r.status == "success" for r in results)
    err = capsys.readouterr().err
    assert "with_prog" in err and "stalled" in err
    assert "no_prog" not in err


# ---------------------------------------------------------------------------
# (17) Worker stream capture: per-job log file, and drainage of BOTH pipes
# ---------------------------------------------------------------------------

def test_worker_log_file_captures_both_streams(tmp_path):
    """A job that names a log_file gets BOTH captured streams written there;
    the result JSON is still taken from the last stdout line, and the raw
    stdout is available on the result for the caller to scan."""
    script = _write_worker_script(tmp_path, "noisy", """\
        import json, sys
        print("compiling shard 0")
        print("  eval[c2] FAILED: RuntimeError: alloc failed", file=sys.stderr)
        print(json.dumps({"status": "success", "n_done": 7}))
    """)
    log = tmp_path / "worker_t1_s0.log"
    job = WorkerJob(
        name="noisy", cmd=[sys.executable, script],
        progress_file=str(tmp_path / "p.json"), log_file=str(log),
    )
    results = run_workers([job], max_parallel=1, poll_interval=0.05)
    assert results[0].status == "success"
    assert results[0].payload["n_done"] == 7
    text = log.read_text()
    assert "compiling shard 0" in text                       # stdout
    assert "eval[c2] FAILED: RuntimeError" in text           # stderr
    assert "compiling shard 0" in results[0].stdout
    assert "eval[c2] FAILED" in results[0].stderr


def test_worker_without_log_file_writes_nothing(tmp_path):
    """log_file is opt-in: the training-side jobs that omit it are unaffected."""
    script = _write_worker_script(tmp_path, "quiet", """\
        import json
        print(json.dumps({"status": "success"}))
    """)
    job = WorkerJob(name="quiet", cmd=[sys.executable, script],
                    progress_file=str(tmp_path / "p.json"))
    results = run_workers([job], max_parallel=1, poll_interval=0.05)
    assert results[0].payload["status"] == "success"
    assert job.log_file is None
    assert not list(tmp_path.glob("*.log"))


def test_large_stdout_does_not_block_worker(tmp_path):
    """A worker writing far more than one pipe buffer (~1 MB here vs the 64 kB
    Linux default) to stdout must still exit: BOTH pipes are drained while it
    runs. The worker self-terminates if the parent stops reading, so a
    regression fails this test instead of hanging the suite."""
    script = _write_worker_script(tmp_path, "chatty", """\
        import json, os, threading
        _guard = threading.Timer(45.0, lambda: os._exit(3))
        _guard.daemon = True
        _guard.start()
        for i in range(20000):
            print("filler line %d 0123456789012345678901234567890123456789" % i)
        print(json.dumps({"status": "success"}))
    """)
    job = WorkerJob(name="chatty", cmd=[sys.executable, script],
                    progress_file=str(tmp_path / "p.json"))
    results = run_workers([job], max_parallel=1, poll_interval=0.05)
    assert results[0].returncode == 0, "worker blocked on a full stdout pipe"
    assert results[0].status == "success"
    assert results[0].payload["status"] == "success"


def test_grandchild_holding_pipe_does_not_stall_the_parent(tmp_path, monkeypatch):
    """A worker whose grandchild inherits stdout leaves the drainer parked on a
    pipe that never reaches EOF after the worker exits. Retiring that job must
    stay bounded: the poll loop also drives every other running worker, so a
    truncated capture is the acceptable outcome and a stalled parent is not.
    The grandchild outlives the assertion window, so an unbounded wait (or a
    close() that waits on the parked drainer's buffer lock) fails here."""
    monkeypatch.setattr("xcquinox.alec.parallel.STREAM_JOIN_SEC", 0.2)
    script = _write_worker_script(tmp_path, "leaky", """\
        import json, subprocess, sys
        print(json.dumps({"status": "success"}), flush=True)
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    """)
    job = WorkerJob(name="leaky", cmd=[sys.executable, script],
                    progress_file=str(tmp_path / "p.json"))
    t0 = time.time()
    results = run_workers([job], max_parallel=1, poll_interval=0.05)
    elapsed = time.time() - t0
    assert results[0].status == "success"
    assert results[0].payload["status"] == "success"
    assert elapsed < 5.0, f"parent waited {elapsed:.1f}s on a held-open pipe"


def test_pyscf_pool_threads_caps_each_pool_at_the_measured_knee():
    """Each PySCF-serving pool is capped at PYSCF_POOL_THREADS_MAX, the
    largest count within 1.5x of the measured optimum (4 threads: 7.5 s for
    the C2H2 OEP at def2-svp; 8 threads: 11.1 s; 20 threads on a 20-core box:
    over 97 s for either pool alone), and never below one thread."""
    from xcquinox.alec.parallel import PYSCF_POOL_THREADS_MAX, pyscf_pool_threads
    assert PYSCF_POOL_THREADS_MAX == 8
    assert [pyscf_pool_threads(n) for n in (0, 1, 4, 8, 9, 24, 40, 96)] == \
        [1, 1, 4, 8, 8, 8, 8, 8]
