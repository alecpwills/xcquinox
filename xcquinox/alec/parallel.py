"""xcquinox.alec.parallel -- WorkerJob, WorkerResult, run_workers.

Parallel orchestration module for launching worker subprocesses.
Implements THE SPEC section 10.2 (full parallel.py listing), section 10.4
(progress file schema), and section 10.5 (argv contract).

NOTE: This is a launcher module; it has NO jax / equinox / optax imports.
Every import below is stdlib, so the module can be imported freely from
the parent process before XLA_FLAGS is written into the child's environment.
"""
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field


# Per-worker thread_env field on WorkerJob: each worker may need a
# different thread count. run_workers merges job.thread_env into the
# child's environment before subprocess.Popen, so the child's first
# `import jax` sees the correct XLA_FLAGS / OMP_NUM_THREADS etc.
# The parent process NEVER mutates its own os.environ.
@dataclass
class WorkerJob:
    name: str
    cmd: list[str]
    progress_file: str
    thread_env: dict[str, str] = field(default_factory=dict)


@dataclass
class WorkerResult:
    job: WorkerJob
    status: str           # "success" | "failed"
    returncode: int
    payload: dict         # parsed JSON from stdout (or synthetic error)
    stderr: str
    duration: float


def worker_script_path(name: str) -> str:
    """Return the absolute path to an alec worker script, for use as
    argv[1] in a subprocess.Popen([sys.executable, script_path, ...])
    launch. Stdlib-only -- does NOT import the worker module itself,
    just resolves its path via importlib.util.find_spec.
    Accepts 'pretrain_worker' / 'train_worker' / 'test_worker'.
    """
    spec = importlib.util.find_spec(f"xcquinox.alec.workers.{name}")
    if spec is None or spec.origin is None:
        raise FileNotFoundError(
            f"could not locate xcquinox.alec.workers.{name}"
        )
    return spec.origin


def _read_progress(progress_file: str) -> dict | None:
    """Defensive reader for a worker's progress.json. Returns the parsed
    dict on success, or None on any expected failure mode (missing file,
    torn read, NFS flicker).
    """
    try:
        with open(progress_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    except OSError:
        return None


# Stall warning threshold in seconds. Exposed at module level so tests
# can monkeypatch it to a small value without waiting 60s.
STALL_WARN_SEC = 60.0


def run_workers(
    jobs: list[WorkerJob],
    max_parallel: int = 4,
    poll_interval: float = 2.0,
    on_progress=None,
) -> list[WorkerResult]:
    """Launch worker subprocesses with bounded parallelism.

    Returns a list of WorkerResult in the SAME ORDER as the input jobs
    list (not in completion order) so downstream bookkeeping is
    deterministic.

    Algorithm:
      1. Fixed-size slot dict mapping job-index to slot state.
      2. Background stderr drainer thread per worker.
      3. Poll loop: read progress, check proc.poll(), parse stdout JSON.
      4. Handle: crash, malformed JSON, empty stdout, missing progress,
         stall detection (STALL_WARN_SEC warning).
      5. Popen failure recorded as synthetic WorkerResult.
    """
    results: list[WorkerResult | None] = [None] * len(jobs)
    pending: deque[tuple[int, WorkerJob]] = deque(enumerate(jobs))
    running: dict[int, dict] = {}

    def _drain_stderr(proc, lines: list[str]) -> None:
        """Background drainer: append each stderr line as it arrives."""
        try:
            for line in proc.stderr:
                lines.append(line)
        except (ValueError, OSError):
            pass

    def _start(idx: int, job: WorkerJob) -> None:
        env = os.environ.copy()
        env.update(job.thread_env)
        try:
            proc = subprocess.Popen(
                job.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        except (FileNotFoundError, PermissionError, OSError) as e:
            results[idx] = WorkerResult(
                job=job,
                status="failed",
                returncode=-1,
                payload={"error": f"failed to spawn worker: {e!r}"},
                stderr="",
                duration=0.0,
            )
            return
        stderr_lines: list[str] = []
        drainer = threading.Thread(
            target=_drain_stderr, args=(proc, stderr_lines), daemon=True
        )
        drainer.start()
        running[idx] = {
            "proc": proc,
            "job": job,
            "start": time.time(),
            "last_progress": None,
            "last_progress_time": time.time(),
            "stderr_lines": stderr_lines,
            "stderr_thread": drainer,
        }

    # Seed the pool.
    while pending and len(running) < max_parallel:
        idx, job = pending.popleft()
        _start(idx, job)

    while running:
        time.sleep(poll_interval)
        finished_indices = []
        for idx, slot in list(running.items()):
            job = slot["job"]
            # Progress poll.
            prog = _read_progress(job.progress_file)
            if prog is not None and prog != slot["last_progress"]:
                slot["last_progress"] = prog
                slot["last_progress_time"] = time.time()
                if on_progress is not None:
                    on_progress(job, prog)
            elif time.time() - slot["last_progress_time"] > STALL_WARN_SEC:
                print(
                    f"WARNING: {job.name} stalled "
                    f"(no progress update for {STALL_WARN_SEC:.0f}s)",
                    file=sys.stderr,
                )
                slot["last_progress_time"] = time.time()  # debounce
            # Completion poll.
            if slot["proc"].poll() is not None:
                stdout, _ = slot["proc"].communicate()
                slot["stderr_thread"].join(timeout=1.0)
                stderr_joined = "".join(slot["stderr_lines"])
                stdout_stripped = stdout.strip() if stdout else ""
                if stdout_stripped:
                    try:
                        payload = json.loads(stdout_stripped.splitlines()[-1])
                    except json.JSONDecodeError:
                        payload = {
                            "error": "malformed stdout JSON",
                            "raw": stdout_stripped[-500:],
                        }
                else:
                    payload = {
                        "error": (
                            f"worker exited with no stdout "
                            f"(returncode={slot['proc'].returncode})"
                        ),
                        "stderr_tail": stderr_joined[-500:] if stderr_joined else "",
                    }
                results[idx] = WorkerResult(
                    job=job,
                    status="success" if slot["proc"].returncode == 0 else "failed",
                    returncode=slot["proc"].returncode,
                    payload=payload,
                    stderr=stderr_joined,
                    duration=time.time() - slot["start"],
                )
                finished_indices.append(idx)
        for idx in finished_indices:
            del running[idx]
            if pending:
                next_idx, next_job = pending.popleft()
                _start(next_idx, next_job)

    # Strict index-ordered return.
    return [results[i] for i in range(len(jobs))]


def _thread_env(threads: int) -> dict[str, str]:
    """Build the XLA_FLAGS + BLAS thread count dict that every worker
    inherits as part of its subprocess environment.
    """
    return {
        "XLA_FLAGS": (
            f"--xla_cpu_multi_thread_eigen=true "
            f"intra_op_parallelism_threads={threads}"
        ),
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
    }


def detect_available_cpus() -> int:
    """Queue-agnostic count of CPUs THIS process may actually use.

    Prefers the scheduler affinity mask (``os.sched_getaffinity``) which
    respects the SLURM/cgroup cpuset, whether the node is ``exclusive`` (the
    whole node) or a ``shared`` slice, so the eval parallelism adapts to
    whatever partition the job lands on rather than the static ``cpus_per_task``
    request. Falls back to ``SLURM_CPUS_PER_TASK`` then the machine core count.
    Always returns >= 1."""
    try:
        n = len(os.sched_getaffinity(0))
        if n >= 1:
            return n
    except (AttributeError, OSError):
        pass
    env = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if env.isdigit() and int(env) >= 1:
        return int(env)
    return max(1, os.cpu_count() or 1)


def eval_worker_ladder(total_cpus: int, top: int | None = None
                       ) -> list[tuple[int, int]]:
    """Descending ``(n_workers, threads_per_worker)`` tiers for the held-out
    eval's adaptive degradation.

    Starts at ``top`` (or ``total_cpus`` if unset, capped at ``total_cpus``) and
    halves the worker count twice, three tiers max, giving each worker
    ``total_cpus // n`` BLAS threads so the node stays fully utilized at every
    tier. Tiers with ``n <= 1`` are dropped (those mean "serial", handled
    separately by the caller), so a single core / ``top=1`` yields an empty
    ladder (serial only). Example: ``total_cpus=24 -> [(24,1),(12,2),(6,4)]``."""
    base = total_cpus if top is None else min(top, total_cpus)
    tiers: list[tuple[int, int]] = []
    seen: set[int] = set()
    for n in (base, base // 2, base // 4):
        if n > 1 and n not in seen:
            seen.add(n)
            tiers.append((n, max(1, total_cpus // n)))
    return tiers


def build_pretrain_jobs(
    specs,
    *,
    checkpoint_base: str,
    data_dir: str,
    threads: int,
) -> list[WorkerJob]:
    """Build one WorkerJob per pre-built PretrainSpec.

    Each job's cmd uses the direct-file launch path
    (python <abs_path_to_pretrain_worker.py>) -- NOT python -m.
    The caller MUST have already written each spec to
    <checkpoint_base>/01_pretrain/<arch.name>/pretrain_spec.pkl
    BEFORE calling this function.
    """
    worker_py = worker_script_path("pretrain_worker")
    env = _thread_env(threads)
    built_jobs = []
    for spec in specs:
        arch_name = spec.arch.name
        pretrain_dir = os.path.join(checkpoint_base, "01_pretrain", arch_name)
        spec_path = os.path.join(pretrain_dir, "pretrain_spec.pkl")
        progress_file = os.path.join(pretrain_dir, "progress.json")
        cmd = [
            sys.executable, worker_py,
            "--arch", arch_name,
            "--spec-pickle", spec_path,
            "--checkpoint-base", checkpoint_base,
            "--data-dir", data_dir,
            "--threads", str(threads),
        ]
        built_jobs.append(WorkerJob(
            name=arch_name,
            cmd=cmd,
            progress_file=progress_file,
            thread_env=dict(env),
        ))
    return built_jobs


def build_training_jobs(
    specs,
    *,
    checkpoint_base: str,
    data_dir: str,
    threads: int,
) -> list[WorkerJob]:
    """Build one WorkerJob per pre-built TrainingSpec.

    Each job name is <arch>/<loss_name>, matching the on-disk checkpoint
    layout at <checkpoint_base>/02_train/<arch>/<loss_name>/.
    The caller MUST have already written each spec to
    <checkpoint_base>/02_train/<arch>/<loss_name>/train_spec.pkl
    BEFORE calling this function.
    """
    worker_py = worker_script_path("train_worker")
    env = _thread_env(threads)
    built_jobs = []
    for spec in specs:
        arch_name = spec.arch.name
        loss_name = spec.loss_name
        train_dir = os.path.join(checkpoint_base, "02_train", arch_name, loss_name)
        spec_path = os.path.join(train_dir, "train_spec.pkl")
        progress_file = os.path.join(train_dir, "progress.json")
        cmd = [
            sys.executable, worker_py,
            "--arch", arch_name,
            "--spec-pickle", spec_path,
            "--checkpoint-base", checkpoint_base,
            "--data-dir", data_dir,
            "--threads", str(threads),
        ]
        built_jobs.append(WorkerJob(
            name=f"{arch_name}/{loss_name}",
            cmd=cmd,
            progress_file=progress_file,
            thread_env=dict(env),
        ))
    return built_jobs
