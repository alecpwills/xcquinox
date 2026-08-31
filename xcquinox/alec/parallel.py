"""xcquinox.alec.parallel -- WorkerJob, WorkerResult, run_workers.

Parallel orchestration module for launching worker subprocesses: the
WorkerJob/WorkerResult data model, the progress-file schema, and the
worker argv contract.

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
    # Path the worker updates as it advances, polled for the progress callback
    # and the stall watchdog. None for workers that write none (the held-out
    # eval shards), which opts them out of both.
    progress_file: str | None
    thread_env: dict[str, str] = field(default_factory=dict)
    # Optional path for this worker's captured stdout+stderr, written once it
    # exits. A worker's own diagnostics otherwise die with the pipe: run_workers
    # keeps only the parsed result line, so a per-species failure inside an
    # otherwise-successful worker leaves no trace on disk.
    log_file: str | None = None


@dataclass
class WorkerResult:
    job: WorkerJob
    status: str           # "success" | "failed"
    returncode: int
    payload: dict         # parsed JSON from stdout (or synthetic error)
    stderr: str
    duration: float
    stdout: str = ""      # raw stdout (result line + whatever the worker logged)


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

# Bound on how long a finished worker's stream drainers are waited on. The
# pipes are at EOF the moment the worker exits, so this is only reached when a
# grandchild inherited one and holds it open; the wait must stay bounded
# because the poll loop that reaches it also drives every other running worker.
STREAM_JOIN_SEC = 1.0


def _write_worker_log(path: str | None, name: str,
                      stdout: str, stderr: str) -> None:
    """Persist one worker's captured streams to ``path`` (no-op when unset).

    The two pipes are separate, so the streams are written in labelled blocks
    rather than interleaved. A log-write failure is reported but never fails
    the job the log describes.
    """
    if not path:
        return
    try:
        with open(path, "w") as f:
            f.write(f"# worker: {name}\n# ---- stdout ----\n")
            f.write(stdout if stdout.endswith("\n") or not stdout
                    else stdout + "\n")
            f.write("# ---- stderr ----\n")
            f.write(stderr if stderr.endswith("\n") or not stderr
                    else stderr + "\n")
    except OSError as e:
        print(f"WARNING: could not write worker log {path}: {e}",
              file=sys.stderr)


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
      2. Background stdout AND stderr drainer threads per worker.
      3. Poll loop: read progress, check proc.poll(), parse stdout JSON.
      4. Handle: crash, malformed JSON, empty stdout, missing progress,
         stall detection (STALL_WARN_SEC warning).
      5. Popen failure recorded as synthetic WorkerResult.
      6. Both captured streams written to ``job.log_file`` when it is set.
    """
    results: list[WorkerResult | None] = [None] * len(jobs)
    pending: deque[tuple[int, WorkerJob]] = deque(enumerate(jobs))
    running: dict[int, dict] = {}
    # Pool slots for the worker CPU bind: a job whose thread_env carries
    # WORKER_BIND_CPUS_ENV gets WORKER_SLOT_ENV stamped from this pool, so
    # apply_worker_cpu_bind() in the worker pins slot-disjoint CPU slices.
    # Slots are recycled as workers finish (popped lowest-first so a
    # non-full pool stays packed at the low CPUs).
    free_cpu_slots: list[int] = list(range(max_parallel))[::-1]

    def _drain(stream, lines: list[str]) -> None:
        """Background drainer: append each line as it arrives. BOTH pipes are
        drained while the worker runs -- a worker that fills either 64 kB pipe
        buffer blocks in write() and never exits otherwise."""
        try:
            for line in stream:
                lines.append(line)
        except (ValueError, OSError):
            pass

    def _start(idx: int, job: WorkerJob) -> None:
        env = os.environ.copy()
        # Bind hygiene: a stale bind request or slot inherited from the
        # PARENT's environment must never reach a child that did not ask for
        # one (measured: two unbound jobs under an inherited slot both pinned
        # to the same slice). The job's own thread_env is the only source.
        env.pop(WORKER_BIND_CPUS_ENV, None)
        env.pop(WORKER_SLOT_ENV, None)
        env.update(job.thread_env)
        cpu_slot = None
        if WORKER_BIND_CPUS_ENV in job.thread_env and free_cpu_slots:
            cpu_slot = free_cpu_slots.pop()
            env[WORKER_SLOT_ENV] = str(cpu_slot)
        try:
            proc = subprocess.Popen(
                job.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        except (FileNotFoundError, PermissionError, OSError) as e:
            if cpu_slot is not None:
                free_cpu_slots.append(cpu_slot)
            results[idx] = WorkerResult(
                job=job,
                status="failed",
                returncode=-1,
                payload={"error": f"failed to spawn worker: {e!r}"},
                stderr="",
                duration=0.0,
            )
            return
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        out_drainer = threading.Thread(
            target=_drain, args=(proc.stdout, stdout_lines), daemon=True
        )
        err_drainer = threading.Thread(
            target=_drain, args=(proc.stderr, stderr_lines), daemon=True
        )
        out_drainer.start()
        err_drainer.start()
        running[idx] = {
            "proc": proc,
            "job": job,
            "cpu_slot": cpu_slot,
            "start": time.time(),
            "last_progress": None,
            "last_progress_time": time.time(),
            "stdout_lines": stdout_lines,
            "stderr_lines": stderr_lines,
            "stdout_thread": out_drainer,
            "stderr_thread": err_drainer,
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
            # Progress poll. A job with no progress_file opts OUT of both the
            # poll and the watchdog: for a worker that never writes one the
            # warning fires every STALL_WARN_SEC for the worker's whole
            # lifetime and says nothing about its health.
            if job.progress_file:
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
                # Drain to EOF. The worker has exited, so both pipes normally
                # see EOF at once; the bound covers the case where a grandchild
                # inherited a pipe and holds it open, which would otherwise
                # park a drainer (and, with an unbounded join, this poll loop
                # and every other running worker with it). A drainer left alive
                # costs a truncated capture, never a stalled parent.
                for thread in (slot["stdout_thread"], slot["stderr_thread"]):
                    thread.join(timeout=STREAM_JOIN_SEC)
                # Snapshot before joining the text: a drainer that outlived the
                # bound above is still appending to these lists.
                stdout = "".join(list(slot["stdout_lines"]))
                stderr_joined = "".join(list(slot["stderr_lines"]))
                for thread, stream in ((slot["stdout_thread"],
                                        slot["proc"].stdout),
                                       (slot["stderr_thread"],
                                        slot["proc"].stderr)):
                    if thread.is_alive():
                        # close() waits on the buffered-reader lock the parked
                        # drainer holds, so it would block the parent for as
                        # long as the pipe stays open (measured: still blocked
                        # after 3 s, returning only when the grandchild exited).
                        # The fd is released when the parked drainer thread
                        # exits -- once the grandchild lets go of the pipe --
                        # and drops its reference to the stream (measured: no
                        # pipe fd outlives that point).
                        continue
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass
                _write_worker_log(job.log_file, job.name, stdout,
                                  stderr_joined)
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
                    stdout=stdout,
                )
                finished_indices.append(idx)
        for idx in finished_indices:
            if running[idx].get("cpu_slot") is not None:
                free_cpu_slots.append(running[idx]["cpu_slot"])
            del running[idx]
            if pending:
                next_idx, next_job = pending.popleft()
                _start(next_idx, next_job)

    # Strict index-ordered return.
    return [results[i] for i in range(len(jobs))]


#: Environment keys of the worker CPU bind: the pool launcher writes the
#: per-worker thread budget and run_workers stamps the pool slot; the worker
#: applies both through apply_worker_cpu_bind() before its first JAX import.
WORKER_BIND_CPUS_ENV = "XCQUINOX_WORKER_BIND_CPUS"
WORKER_SLOT_ENV = "XCQUINOX_WORKER_SLOT"


def apply_worker_cpu_bind() -> int | None:
    """Pin this pool worker's CPU affinity BEFORE the first JAX import.

    The logic lives in ``workers/_cpu_bind.py`` (stdlib-only, importable by
    the path-launched workers WITHOUT executing the package __init__, which
    stands up the JAX CPU backend and its thread pool -- a pin applied after
    that binds the calling thread only; measured: 41-43 threads already
    alive, a one-CPU worker at 694-990 percent load, against 100 / 195 / 334
    percent at budgets 1 / 2 / 4 with the pin first). This parent-side
    delegator loads that file by path for the same reason and exists for the
    API and its tests; a WORKER must ``import _cpu_bind`` directly, before
    anything else.

    Affinity is what bounds XLA's CPU intra-op pool: TSL sizes it from
    NumSchedulableCPUs, i.e. sched_getaffinity (the ``--xla_cpu_multi_thread_
    eigen`` flag is measured inert on the pinned jaxlib 0.7.0 thunk runtime).
    Returns the number of CPUs pinned, or None when unbound.
    """
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "workers", "_cpu_bind.py")
    spec = importlib.util.spec_from_file_location("_xcq_cpu_bind", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.apply()


def _thread_env(threads: int, *, bound_worker: bool) -> dict[str, str]:
    """Build the env dict a worker subprocess inherits: XLA compile trims,
    BLAS thread caps, and -- for pool members -- the CPU-bind request.

    ``bound_worker`` is required because the two callers need opposite
    things. A POOL MEMBER (``True``: the held-out eval shards, the local
    pretrain/train job builders) must be confined to its share of the node:
    the OMP/MKL/OPENBLAS variables bound only the BLAS pools, while XLA's
    CPU intra-op pool sizes to the node's schedulable CPUs regardless, so a
    tier of N single-BLAS-thread workers still carries N node-wide pools
    that spin-wait between the small dense operations of an SCF loop --
    measured: the inline train-eval task 2138032_15 ran its held-out eval
    tier (40 workers x 1 BLAS thread on a 40-core node) at a 442 percent
    node load (HPC operations report, 2026-08-29), one worker alone
    measuring 4.2-4.8x at OMP_NUM_THREADS=1, and the workflow-matrix job
    2134488 paid the same spin-wait class at about ten minutes per molecule
    against 8 s. The bound that works is the CPU affinity the worker
    applies to itself through :func:`apply_worker_cpu_bind` (TSL sizes the
    pool from sched_getaffinity), requested here via
    :data:`WORKER_BIND_CPUS_ENV`; the ``--xla_cpu_multi_thread_eigen``
    token is dropped from the pool env because it is measured inert on the
    pinned jaxlib 0.7.0 thunk runtime -- the same reason the mis-prefixed
    ``intra_op_parallelism_threads=<n>`` token went earlier.

    The PREFLIGHT COMPILE-SMOKE PROBE (``False``) keeps the historical flag
    string verbatim and carries no bind request: it exists to mirror the
    train array's own sbatch environment (which exports that flag string)
    and must stay representative of it, byte for byte.
    """
    if bound_worker:
        xla_flags = ("--xla_llvm_disable_expensive_passes=true "
                     "--xla_backend_optimization_level=1")
    else:
        xla_flags = ("--xla_cpu_multi_thread_eigen=true "
                     "--xla_llvm_disable_expensive_passes=true "
                     "--xla_backend_optimization_level=1")
    env = {
        # Compile-memory trims (results-neutral: they cut LLVM codegen peak
        # RSS and time for large-basis kernels).
        "XLA_FLAGS": xla_flags,
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
    }
    if bound_worker:
        env[WORKER_BIND_CPUS_ENV] = str(threads)
    return env


# The thread budget of each pool that serves PySCF -- its own OpenMP pool
# (integrals, numint) and numpy's OpenBLAS pthreads pool -- in a process whose
# PySCF work is a loop of small dense operations: a reference build (SCF, CCSD,
# then an OEP inversion whose L-BFGS-B outer loop runs an inner SCF per
# iteration; nao is 38 for C2H2 at def2-svp), the fidelity certificate's parent
# routes, the per-atom SCF loop of the pretraining data. Both pools spin-wait
# between operations, so sized to the core count either one alone makes such a
# loop crawl and the two together stall it. Measured on a 20-core workstation,
# C2H2 at def2-svp / grid level 1, (OpenMP threads, BLAS threads) -> SCF, CCSD,
# OEP wall:
#   (1, 1)    0.3 s   0.4 s   10.9 s
#   (4, 4)    0.2 s   0.4 s    7.5 s   <- the optimum
#   (8, 8)    0.9 s   0.9 s   11.1 s
#   (10, 10)  1.1 s   1.7 s   14.4 s
#   (1, 20)   0.9 s   2.3 s   60.1 s
#   (20, 1)   5.3 s  18.1 s   > 97 s  (killed at 120 s)
#   (20, 20)  6.0 s  18.0 s   > 216 s (killed at 240 s)
# and on a 40-core cluster node at (40, 40) the same build took 739.5 s for
# C2H2, 489.2 s for C2H4, 377.7 s for C2H6 and 56.2 s for CH3, while the atoms
# and the radicals CH and CH2, whose OEP converges in a few iterations, stayed
# under 9 s (workflow-matrix job 2134488, preflight stage, per-species run
# log): the closed-shell molecules run hundreds of L-BFGS-B iterations and pay
# the per-iteration overhead in full. The cap
# is the largest count within 1.5x of the measured optimum on the OEP wall
# (11.1 s against 7.5 s; 10 threads is 1.9x); a larger basis gains parallel
# CCSD(T) from it without approaching the core count of any node in use (28,
# 40 and 96). The shell form in the job templates treats an allocation that is
# not a non-negative integer as unset (the cap) and floors at one thread.
PYSCF_POOL_THREADS_MAX = 8


def pyscf_pool_threads(allocation: int) -> int:
    """Threads for each of the two PySCF-serving pools of a process holding
    ``allocation`` CPUs: ``min(allocation, PYSCF_POOL_THREADS_MAX)``, never
    below 1. The measurements are on :data:`PYSCF_POOL_THREADS_MAX`."""
    return max(1, min(int(allocation), PYSCF_POOL_THREADS_MAX))


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
    env = _thread_env(threads, bound_worker=True)
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
    env = _thread_env(threads, bound_worker=True)
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
