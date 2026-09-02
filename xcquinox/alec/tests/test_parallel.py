"""Tests for xcquinox.alec.parallel.

Implements THE SPEC section 13.2 test_parallel.py items (1)-(15), plus
(16)-(17) for the held-out eval shard path: the stall-watchdog opt-out and
worker stream capture.
All tests use mock subprocesses or tiny helper scripts -- no real worker
processes, no jax/equinox/optax imports.
"""
import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

import xcquinox.alec.parallel as parallel
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
    # thread_env populated; a pool member runs single-thread Eigen
    assert "OMP_NUM_THREADS" in j.thread_env
    assert j.thread_env["OMP_NUM_THREADS"] == "2"
    assert j.thread_env[parallel.WORKER_BIND_CPUS_ENV] == j.thread_env["OMP_NUM_THREADS"]


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


def test_bound_policy_at_call_sites():
    """The one launcher that must stay unbound is the preflight compile-smoke
    probe (representative of the train array); every pool launcher binds."""
    import xcquinox.alec.cluster._holdout_parallel as hp
    import xcquinox.alec.cluster._preflight as pf
    from xcquinox.alec.tests._source_scan import code_only
    assert "bound_worker=True" in code_only(hp)
    assert "bound_worker=False" in code_only(pf)
    assert "bound_worker=True" not in code_only(pf)


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


def test_apply_worker_cpu_bind_noop_cases(monkeypatch):
    """Unbound when: no request, no slot, or a budget covering the whole
    allowance (nothing to bound). No sched_setaffinity call is made."""
    applied: list = []
    _fake_affinity(monkeypatch, range(4), applied)
    monkeypatch.delenv(parallel.WORKER_BIND_CPUS_ENV, raising=False)
    monkeypatch.delenv(parallel.WORKER_SLOT_ENV, raising=False)
    assert parallel.apply_worker_cpu_bind() is None
    monkeypatch.setenv(parallel.WORKER_BIND_CPUS_ENV, "2")
    assert parallel.apply_worker_cpu_bind() is None  # no slot
    monkeypatch.setenv(parallel.WORKER_SLOT_ENV, "0")
    monkeypatch.setenv(parallel.WORKER_BIND_CPUS_ENV, "4")
    assert parallel.apply_worker_cpu_bind() is None  # budget == allowance
    assert applied == []


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"),
                    reason="platform without sched_setaffinity")
@pytest.mark.skipif(hasattr(os, "sched_getaffinity")
                    and len(os.sched_getaffinity(0)) < 3,
                    reason="needs at least 3 allowed CPUs")
@pytest.mark.skipif(not os.path.isdir("/proc/self/task"),
                    reason="needs /proc task introspection")
def test_worker_thread_pool_is_bounded_by_the_bind():
    """BEHAVIORAL: a subprocess that binds the way the workers do (path-local
    ``_cpu_bind`` import BEFORE the first JAX import) has EVERY OS thread of
    its process confined to the slice after the JAX backend spins up --
    checked over /proc/self/task, the quantity that stayed unbounded when the
    bind was reached through the package import (the package __init__ stands
    up the JAX backend and its ~40-thread pool before the pin; measured
    694-990 percent load at budget 1 in that arrangement)."""
    workers_dir = os.path.join(
        os.path.dirname(parallel.__file__), "workers")
    code = (
        "import os, sys\n"
        f"sys.path.insert(0, {workers_dir!r})\n"
        "import _cpu_bind\n"
        "n = _cpu_bind.apply()\n"
        "import jax.numpy as jnp\n"
        "x = jnp.ones((256, 256))\n"
        "(x @ x).block_until_ready()\n"
        "slice_ = os.sched_getaffinity(0)\n"
        "def cpus(spec):\n"
        "    out = set()\n"
        "    for part in spec.split(','):\n"
        "        a, _, b = part.partition('-')\n"
        "        out.update(range(int(a), int(b or a) + 1))\n"
        "    return out\n"
        "bad = 0\n"
        "n_tasks = 0\n"
        "for tid in os.listdir('/proc/self/task'):\n"
        "    with open(f'/proc/self/task/{tid}/status') as fh:\n"
        "        for line in fh:\n"
        "            if line.startswith('Cpus_allowed_list'):\n"
        "                n_tasks += 1\n"
        "                if not cpus(line.split(':')[1].strip()) <= slice_:\n"
        "                    bad += 1\n"
        "print(n, len(slice_), n_tasks, bad)\n"
    )
    env = dict(os.environ)
    env.update(_thread_env(2, bound_worker=True))
    env[parallel.WORKER_SLOT_ENV] = "0"
    env["JAX_PLATFORMS"] = "cpu"
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, check=True,
                         timeout=120)
    n_pinned, n_affinity, n_tasks, n_bad = out.stdout.split()
    assert n_pinned == "2" and n_affinity == "2", out.stdout
    assert int(n_tasks) >= 2, out.stdout          # the pool actually spun up
    assert n_bad == "0", (
        f"{n_bad} of {n_tasks} OS threads escaped the 2-CPU slice: the JAX "
        "pool predates the pin")


def test_parent_side_delegator_matches_the_worker_module(monkeypatch):
    """parallel.apply_worker_cpu_bind delegates to workers/_cpu_bind.py by
    file path, and the two modules' env-variable names are pinned equal."""
    import importlib.util
    path = os.path.join(os.path.dirname(parallel.__file__), "workers",
                        "_cpu_bind.py")
    spec = importlib.util.spec_from_file_location("_cpu_bind_pin_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.WORKER_BIND_CPUS_ENV == parallel.WORKER_BIND_CPUS_ENV
    assert mod.WORKER_SLOT_ENV == parallel.WORKER_SLOT_ENV
    applied: list = []
    _fake_affinity(monkeypatch, range(8), applied)
    monkeypatch.setenv(parallel.WORKER_BIND_CPUS_ENV, "2")
    monkeypatch.setenv(parallel.WORKER_SLOT_ENV, "1")
    assert parallel.apply_worker_cpu_bind() == 2
    assert applied[-1] == [2, 3]


def test_apply_worker_cpu_bind_wrap_overlap_is_the_documented_degradation(
        monkeypatch):
    """A slot pushed past the allowance wraps by modulo (documented, and
    unreachable from the shipped ladders, which keep workers x threads within
    the allowance): slot 2 at budget 2 on a 4-CPU allowance lands back on the
    first slice rather than raising."""
    applied: list = []
    _fake_affinity(monkeypatch, range(4), applied)
    monkeypatch.setenv(parallel.WORKER_BIND_CPUS_ENV, "2")
    monkeypatch.setenv(parallel.WORKER_SLOT_ENV, "2")
    assert parallel.apply_worker_cpu_bind() == 2
    assert applied[-1] == [0, 1]


def test_run_workers_slot_gating_reads_the_job_not_the_inherited_env(
        tmp_path, monkeypatch):
    """A job WITHOUT the bind request consumes no slot even when the parent
    process environment happens to carry the variable (the gate reads
    job.thread_env, not the merged env)."""
    monkeypatch.setenv(parallel.WORKER_BIND_CPUS_ENV, "1")
    script = tmp_path / "noslot_echo.py"
    script.write_text(textwrap.dedent(f"""\
        import json, os
        print(json.dumps({{'slot': os.environ.get('{parallel.WORKER_SLOT_ENV}')}}))
    """))
    jobs = [WorkerJob(name="nobind", cmd=[sys.executable, str(script)],
                      progress_file=None, thread_env={})]
    results = run_workers(jobs, max_parallel=2, poll_interval=0.1)
    assert results[0].payload["slot"] is None


def test_run_workers_stamps_and_recycles_cpu_slots(tmp_path):
    """run_workers hands each bound worker a distinct pool slot and recycles
    slots as workers finish: 3 jobs through a 2-slot pool see slots {0,1}
    only, and the two concurrent workers never share one."""
    script = tmp_path / "slot_echo.py"
    script.write_text(textwrap.dedent(f"""\
        import json, os, time
        time.sleep(0.4)
        print(json.dumps({{'slot': os.environ.get('{parallel.WORKER_SLOT_ENV}')}}))
    """))
    env = _thread_env(1, bound_worker=True)
    jobs = [WorkerJob(name=f"j{i}", cmd=[sys.executable, str(script)],
                      progress_file=None, thread_env=dict(env))
            for i in range(3)]
    results = run_workers(jobs, max_parallel=2, poll_interval=0.1)
    slots = [r.payload["slot"] for r in results]
    assert all(s is not None for s in slots), slots
    assert set(slots) <= {"0", "1"}, slots
    assert set(slots[:2]) == {"0", "1"}, slots  # concurrent pair distinct


def test_worker_scripts_bind_before_jax():
    """Every worker script binds through the path-local ``_cpu_bind`` module
    (never through ``parallel``, whose package import stands up the JAX pool
    before the pin) in its preamble, and no longer carries the inert eigen
    token."""
    base = os.path.join(os.path.dirname(parallel.__file__), "workers")
    for name in ("eval_holdout_worker.py", "train_worker.py",
                 "test_worker.py", "pretrain_worker.py"):
        with open(os.path.join(base, name)) as fh:
            src = fh.read()
        assert "_cpu_bind.apply()" in src, name
        assert "import _cpu_bind" in src, name
        assert "from xcquinox.alec.parallel import apply_worker_cpu_bind" \
            not in src, name
        assert "--xla_cpu_multi_thread_eigen" not in src, name


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


def test_failed_spawn_does_not_orphan_pending_jobs(tmp_path):
    """A replacement job whose binary cannot spawn must not end the
    replenishment chain: at max_parallel=1 with [good, unspawnable, good],
    the third job must still run and all three results return."""
    ok = _write_worker_script(
        tmp_path, "ok", """
        import json
        print(json.dumps({"ok": True}))
        """)
    jobs = [
        _make_job(tmp_path, "good1", ok),
        WorkerJob(name="bad", cmd=["/nonexistent/binary/xyz"],
                  progress_file=str(tmp_path / "bad_progress.json")),
        _make_job(tmp_path, "good2", ok),
    ]
    results = run_workers(jobs, max_parallel=1, poll_interval=0.05)
    assert len(results) == 3
    assert results[0].status == "success"
    assert results[1].status == "failed"
    assert "failed to spawn" in results[1].payload.get("error", "")
    assert results[2] is not None and results[2].status == "success", (
        "the job queued behind a failed spawn was orphaned")
