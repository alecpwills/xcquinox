"""xcquinox.alec.cluster.job_tracking: SLURM job submission log + outcome reduction.

The HPC harness submits SLURM job arrays (a pretrain array, then a preflight
job, then a train array, then an eval array) and must be able to recover
from partial failure: the ``resubmit`` / ``status`` commands need to know, per
grid index, whether the training/eval task succeeded or failed and, if it
failed: why.

This module owns three things:

  - ``jobs.json``: an append-only submission log living at
    ``<run_dir>/jobs.json``. Append-only because the log is the audit trail of
    every array we ever submitted; ``resubmit`` only ever marks old records
    superseded, it never deletes them.
  - ``_run_slurm``: the single subprocess seam through which EVERY SLURM
    command runs. Centralizing it (a) gives one place to apply timeouts and
    retry policy and (b) gives tests one thing to monkeypatch. ``submit.py``
    (a later task) imports and reuses this seam.
  - ``reduce_outcomes``: the per-index outcome map that drives ``resubmit``.

Non-obvious rules, documented at their implementation site below:
  - never-glob: ``reduce_outcomes`` iterates the manifest's index range,
    never ``glob("checkpoints/spec_*")``: a stale higher-index directory from
    a larger prior grid must be structurally uncountable.
  - disk-first: an index's outcome is taken from on-disk evidence
    (``model.eqx`` / ``failure.json``) BEFORE any ``sacct`` query, disk
    evidence is authoritative and cheap; ``sacct`` is a fallback only.
  - per-kind generation: the monotonic ``generation`` counter is *per
    ``kind``*: train generation 0 and eval generation 0 are independent.
  - query-retry-but-not-mutating: query verbs (``sacct``/``squeue``) are
    safe to retry on transient failure; mutating verbs (``sbatch``/``scancel``)
    are NEVER retried, a retried ``sbatch`` could double-submit an array.
  - short-circuit: the first ``SlurmTransientError`` while querying any
    generation aborts the whole reduction (the controller is unreachable;
    there is no point hammering it for every remaining generation).
"""
from datetime import datetime, timezone
import json
import os
import subprocess
import tempfile
import time


# Per-call subprocess timeout. SLURM client commands that hang past this are
# treated as a transient failure (for query verbs) or a hard error (mutating).
_SLURM_TIMEOUT_S = 30.0

# Query verbs are read-only and safe to retry; mutating verbs are not.
_QUERY_VERBS = frozenset({"sacct", "squeue"})
_MUTATING_VERBS = frozenset({"sbatch", "scancel"})

_VALID_KINDS = frozenset({"datagen", "pretrain", "preflight", "train", "eval",
                          "benchmark_refs"})

_JOBS_FILENAME = "jobs.json"
_MANIFEST_FILENAME = "manifest.json"
_TMP_PREFIX = ".mktmp_"


# ---------------------------------------------------------------------------
# SLURM subprocess seam
# ---------------------------------------------------------------------------

class SlurmTransientError(Exception):
    """A SLURM query command failed transiently (after exhausting retries).

    Raised only for query verbs (``sacct``/``squeue``), it signals "the
    controller is currently unreachable", which callers treat as a reason to
    stop and surface the problem rather than mis-classify a job's outcome.
    """


def _run_slurm(cmd: list[str], *, retries: int = 3) -> subprocess.CompletedProcess:
    """Run a single SLURM command, the one seam for ALL SLURM subprocesses.

    Every ``sbatch`` / ``sacct`` / ``squeue`` / ``scancel`` invocation in the
    harness goes through here, so this is the single place that owns timeout
    and retry policy.

    Policy:
      - A per-call subprocess timeout of 30 s is always applied.
      - Query verbs (``sacct``, ``squeue``) are read-only, so a non-zero
        exit (or a timeout) is assumed transient: retry up to ``retries`` times
        with exponential backoff (2 s, 4 s, 8 s, capped at 8 s). If it still
        fails, raise :class:`SlurmTransientError`.
      - Mutating verbs (``sbatch``, ``scancel``) are NEVER retried: a
        retried ``sbatch`` could submit the job array twice. A non-zero exit
        propagates immediately as ``CalledProcessError``; a timeout propagates
        as ``TimeoutExpired``.

    The verb is detected from ``cmd[0]`` (its basename, so an absolute path
    such as ``/usr/bin/sbatch`` is handled).

    Returns:
        The completed process on success (exit code 0).
    """
    if not cmd:
        raise ValueError("_run_slurm: cmd must be a non-empty list")
    verb = os.path.basename(cmd[0])

    if verb in _MUTATING_VERBS:
        # Mutating verb: run exactly once. check=True turns a non-zero exit
        # into CalledProcessError; a hang turns into TimeoutExpired. Either
        # way it propagates, we must NOT retry a job-submitting command.
        return subprocess.run(
            cmd, capture_output=True, text=True, check=True,
            timeout=_SLURM_TIMEOUT_S,
        )

    if verb not in _QUERY_VERBS:
        raise ValueError(
            f"_run_slurm: unrecognized SLURM verb {verb!r} (expected one of "
            f"{sorted(_QUERY_VERBS | _MUTATING_VERBS)})"
        )

    # Query verb: retry with exponential backoff on any non-zero exit / hang.
    last_err: Exception | None = None
    attempts = max(1, retries)
    for attempt in range(attempts):
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=_SLURM_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            last_err = exc
        else:
            if proc.returncode == 0:
                return proc
            last_err = subprocess.CalledProcessError(
                proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr,
            )
        # Backoff before the next attempt (skip the wait after the last one).
        if attempt < attempts - 1:
            time.sleep(min(8.0, 2.0 * (2 ** attempt)))

    raise SlurmTransientError(
        f"SLURM query {verb!r} failed after {attempts} attempt(s): {last_err}"
    ) from last_err


# ---------------------------------------------------------------------------
# jobs.json: append-only submission log
# ---------------------------------------------------------------------------

def _jobs_path(run_dir: str) -> str:
    return os.path.join(run_dir, _JOBS_FILENAME)


def _is_real_array_job_id(value) -> bool:
    """True iff ``value`` is a usable SLURM array-job id (non-empty str/int)."""
    if value is None:
        return False
    if isinstance(value, bool):  # bool is an int subclass, reject it explicitly
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        return value.strip() != ""
    return False


def _write_jobs_atomic(records: list[dict], run_dir: str) -> None:
    """Atomically rewrite ``jobs.json`` (mkstemp + os.replace).

    Used by both append and ``mark_superseded`` so a crash mid-write can never
    leave a truncated submission log that a later ``status`` would mis-parse.
    """
    path = _jobs_path(run_dir)
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_name = tempfile.mkstemp(prefix=_TMP_PREFIX, dir=out_dir)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(records, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_job_records(run_dir: str) -> list[dict]:
    """Read every submission record from ``<run_dir>/jobs.json``.

    Returns an empty list if the file does not exist (no jobs submitted yet).

    Raises:
        ValueError: if any record lacks a real numeric/string ``array_job_id``
: a record without one is unusable for ``sacct`` recovery, so the
            log is treated as corrupt rather than silently skipping it.
    """
    path = _jobs_path(run_dir)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(
            f"read_job_records: {path} is not a JSON list "
            f"(got {type(records).__name__})"
        )
    for rec in records:
        if not _is_real_array_job_id(rec.get("array_job_id")):
            raise ValueError(
                f"read_job_records: record in {path} has a missing/invalid "
                f"array_job_id ({rec.get('array_job_id')!r}); the submission "
                "log is corrupt"
            )
    return records


def append_job_record(
    run_dir: str, kind: str, array_job_id, indices: list[int],
) -> dict:
    """Append a submission record to ``<run_dir>/jobs.json`` and return it.

    The log is append-only: this only ever adds a record. The new record's
    ``generation`` is ``1 + max(generation of existing records of the same
    kind)`` (default base ``-1`` -> first record of a kind gets generation 0).
    ``generation`` is per-``kind`` so train and eval counters are independent.

    Args:
        run_dir: the run directory (created if absent).
        kind: one of ``pretrain`` / ``preflight`` / ``train`` / ``eval``.
        array_job_id: the SLURM array-job id ``sbatch`` returned. Must be a
            real non-empty value, an empty/None id is rejected because a
            record carrying it could never be recovered via ``sacct``.
        indices: the grid indices this array covers.

    Returns:
        The newly appended record dict.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"append_job_record: kind must be one of {sorted(_VALID_KINDS)}, "
            f"got {kind!r}"
        )
    if not _is_real_array_job_id(array_job_id):
        raise ValueError(
            "append_job_record: array_job_id must be a real non-empty "
            f"SLURM job id, got {array_job_id!r}. Refusing to log a record "
            "that could never be recovered via sacct."
        )

    os.makedirs(run_dir, exist_ok=True)
    records = read_job_records(run_dir)

    # Per-kind monotonic generation counter.
    prev_max = max(
        (r["generation"] for r in records if r.get("kind") == kind),
        default=-1,
    )
    record = {
        "kind": kind,
        "generation": prev_max + 1,
        "array_job_id": str(array_job_id),
        "indices": list(indices),
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "superseded": False,
    }
    records.append(record)
    _write_jobs_atomic(records, run_dir)
    return record


def mark_superseded(run_dir: str, kind: str, generation: int) -> None:
    """Flag every record of ``(kind, generation)`` as superseded.

    Called by ``resubmit`` before it submits a fresh array: the old array's
    record stays in the append-only log (audit trail) but ``superseded=True``
    excludes it from ``reduce_outcomes``'s ``sacct`` fallback. Rewrites
    ``jobs.json`` atomically.
    """
    records = read_job_records(run_dir)
    touched = False
    for rec in records:
        if rec.get("kind") == kind and rec.get("generation") == generation:
            rec["superseded"] = True
            touched = True
    if touched:
        _write_jobs_atomic(records, run_dir)


# ---------------------------------------------------------------------------
# reduce_outcomes: per-index train/eval outcome map
# ---------------------------------------------------------------------------

def _read_manifest(run_dir: str) -> dict:
    """Load ``manifest.json``; raise a clear error if it is missing."""
    path = os.path.join(run_dir, _MANIFEST_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"reduce_outcomes: no {_MANIFEST_FILENAME} in {run_dir}; the run "
            "directory has not been materialized"
        )
    with open(path) as f:
        return json.load(f)


def _spec_dir(run_dir: str, idx: int, width: int) -> str:
    """Per-spec checkpoint directory ``checkpoints/spec_<idx zero-padded>``."""
    return os.path.join(run_dir, "checkpoints", f"spec_{idx:0{width}d}")


def _disk_outcome(spec_dir: str):
    """Outcome from on-disk evidence in ``spec_dir``, or None if no evidence.

    ``model.eqx`` is checked BEFORE ``failure.json``: if a task produced a
    model it succeeded, regardless of any stale ``failure.json`` left by an
    earlier attempt, success evidence wins.
    """
    if os.path.exists(os.path.join(spec_dir, "model.eqx")):
        return "success"
    failure_path = os.path.join(spec_dir, "failure.json")
    if os.path.exists(failure_path):
        try:
            with open(failure_path) as f:
                failure = json.load(f)
        except (json.JSONDecodeError, OSError):
            return "failure_json_unreadable"
        classification = failure.get("classification")
        return classification if classification else "failure_unclassified"
    return None


def _classify_sacct_state(state: str, exit_code: str) -> str:
    """Map a SLURM task ``State`` (+ ``ExitCode``) to a harness outcome string.

    - ``OUT_OF_MEMORY``, or a ``CANCELLED`` whose exit-code signal looks
      OOM-ish      -> ``"oom"``
    - ``TIMEOUT``  -> ``"timeout"``
    - ``COMPLETED`` -> ``"success"`` (disk evidence should normally have
      caught this already; kept for completeness)
    - ``CANCELLED`` (not OOM) / ``FAILED`` / ``NODE_FAIL`` / empty / unknown
                   -> ``"dependency_never_satisfied"``: for a newest
      non-superseded generation this almost always means the train -> eval
      dependency never cleared (the upstream array was cancelled), so the
      task never ran.
    """
    norm = (state or "").strip().upper()
    # SLURM appends a reason in parentheses, e.g. "CANCELLED by 0".
    head = norm.split()[0] if norm else ""

    if head == "OUT_OF_MEMORY":
        return "oom"
    if head == "TIMEOUT":
        return "timeout"
    if head == "COMPLETED":
        return "success"
    if head == "CANCELLED":
        # A cgroup OOM-kill frequently surfaces as CANCELLED with a 0:125 /
        # 0:9-style exit code; treat an OOM-ish signal as oom, else as a
        # never-satisfied dependency.
        sig = ""
        if exit_code and ":" in str(exit_code):
            sig = str(exit_code).split(":", 1)[1].strip()
        if sig in {"125", "137", "9"}:
            return "oom"
        return "dependency_never_satisfied"
    # FAILED / NODE_FAIL / empty State / anything unrecognized.
    return "dependency_never_satisfied"


def _parse_sacct(stdout: str) -> dict[int, tuple[str, str]]:
    """Parse ``sacct`` pipe-delimited output into ``{task_index: (State, ExitCode)}``.

    Expects rows of ``JobID|State|ExitCode`` produced by
    ``sacct --parsable2 --noheader --format=JobID,State,ExitCode``. Only the
    array-task rows (``<arrayJobId>_<idx>``) are kept; the array's container
    row and any ``.batch`` / ``.extern`` step rows are skipped.
    """
    outcomes: dict[int, tuple[str, str]] = {}
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split("|")
        if len(fields) < 3:
            continue
        job_id, state, exit_code = fields[0], fields[1], fields[2]
        # Step rows look like "12345_0.batch": skip them.
        if "." in job_id:
            continue
        # Array-task rows look like "<arrayJobId>_<taskIndex>".
        if "_" not in job_id:
            continue
        task_part = job_id.rsplit("_", 1)[1]
        # "12345_[3-7]" pending-range rows are not concrete tasks, skip.
        if not task_part.isdigit():
            continue
        outcomes[int(task_part)] = (state, exit_code)
    return outcomes


def _query_sacct(array_job_id: str) -> dict[int, tuple[str, str]]:
    """Run one ``sacct`` for ``array_job_id`` and parse its task rows.

    Always uses an explicit ``--jobs=<id>`` filter, never a time-range scan,
    so the query is cheap and cannot pick up unrelated jobs.
    """
    proc = _run_slurm([
        "sacct",
        f"--jobs={array_job_id}",
        "--parsable2",
        "--noheader",
        "--format=JobID,State,ExitCode",
    ])
    return _parse_sacct(proc.stdout)


def reduce_outcomes(run_dir: str, kind: str) -> dict[int, str]:
    """Compute the per-index outcome map for ``kind`` ∈ {``train``, ``eval``}.

    The result maps every grid index ``0 .. n_specs-1`` to an outcome string.

    Resolution order, per index:
      1. Disk evidence (authoritative, cheap): ``model.eqx`` -> 
         ``"success"``; else ``failure.json`` -> its ``classification``.
      2. **``sacct`` fallback** (only for indices with no disk evidence): for
         the newest non-superseded generation of this ``kind``, one
         ``sacct --jobs=<id>`` is run and the task's State/ExitCode mapped.
         If a generation's ``sacct`` returns nothing (accounting purged) the
         next-older non-superseded generation is consulted; if every one is
         empty the outcome is ``"unknown_sacct_purged"``.

    The index range comes from ``manifest.json`` (``n_specs`` + ``width``),
    it is NEVER derived by globbing ``checkpoints/spec_*``: a stale higher-index
    directory from a larger prior grid must be structurally uncountable.

    At most one ``sacct`` call is made per non-superseded generation, and
    superseded-generation records are ignored entirely. The FIRST
    :class:`SlurmTransientError` raised by any generation's ``sacct`` aborts
    the whole reduction and re-raises, the controller is unreachable, so
    there is no point querying the remaining generations.
    """
    if kind not in ("train", "eval"):
        raise ValueError(
            f"reduce_outcomes: kind must be 'train' or 'eval', got {kind!r}"
        )

    manifest = _read_manifest(run_dir)
    n_specs = int(manifest["n_specs"])
    width = int(manifest["width"])

    outcomes: dict[int, str] = {}
    needs_sacct: list[int] = []
    for idx in range(n_specs):
        disk = _disk_outcome(_spec_dir(run_dir, idx, width))
        if disk is not None:
            outcomes[idx] = disk
        else:
            needs_sacct.append(idx)

    if not needs_sacct:
        return outcomes

    # Non-superseded generations of this kind, newest generation first.
    records = read_job_records(run_dir)
    live = sorted(
        (r for r in records
         if r.get("kind") == kind and not r.get("superseded", False)),
        key=lambda r: r["generation"],
        reverse=True,
    )

    # Consult each live generation in turn (newest wins). One sacct per
    # generation. A SlurmTransientError short-circuits the whole loop.
    pending = set(needs_sacct)
    for rec in live:
        if not pending:
            break
        # Let SlurmTransientError propagate, do NOT keep querying further
        # generations; the controller is unreachable.
        task_states = _query_sacct(rec["array_job_id"])
        if not task_states:
            # Accounting purged for this job id, fall through to an older
            # generation; if none resolves these, they become purged below.
            continue
        for idx in list(pending):
            if idx in task_states:
                state, exit_code = task_states[idx]
                outcomes[idx] = _classify_sacct_state(state, exit_code)
                pending.discard(idx)

    # Anything still pending: no live generation's sacct had a row for it
    # (accounting purged, or the index was never in any array).
    for idx in pending:
        outcomes[idx] = "unknown_sacct_purged"

    return outcomes
