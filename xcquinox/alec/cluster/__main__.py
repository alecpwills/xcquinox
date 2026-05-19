"""xcquinox.alec.cluster.__main__ — CLI dispatch for the HPC training harness.

Invoked as ``python -m xcquinox.alec.cluster <subcommand> ...``. This module is
purely the operator-facing front-end: it parses arguments, owns the run-dir
lock, and wires the already-built ``cluster/`` modules together. Six
subcommands:

  - ``prepare``             — stage input artifacts (CCSD refs, ledger, ...).
  - ``submit``              — create a fresh run dir + submit the 4-stage graph.
  - ``status``              — read-only per-index outcome report.
  - ``resubmit``            — recover FAILED TRAIN tasks (preflight succeeded).
  - ``resubmit-preflight``  — recover a FAILED/timed-out pretrain/preflight.
  - ``repair-manifest``     — rebuild a corrupt/missing ``manifest.json``.

Design rules (enforced below at their use sites):

  - **Dry-run is the default** for every submitting subcommand; real SLURM
    submission happens only with ``--submit``.
  - **Every SLURM subprocess goes through ``job_tracking._run_slurm``** — the
    one seam tests monkeypatch. ``submit_jobs`` already routes through it; the
    sparse-array resubmit paths call it directly for ``sbatch``/``scancel``.
  - ``main`` only dispatches; each subcommand's logic is in its own function;
    small shared helpers (lock, failed-index scan, artifact archive, sparse
    array string) are factored out.
  - **Login-node guard**: ``prepare`` runs the heavy CCSD external-refs
    precompute and is refused on a login node (absent ``$SLURM_JOB_ID``)
    unless ``--no-recompute-refs`` is passed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import sys
import tempfile
from datetime import datetime, timezone

from xcquinox.alec.cluster import job_tracking
from xcquinox.alec.cluster.grid_config import (
    expand_grid,
    load_grid_config,
    validate_grid_semantics,
)
from xcquinox.alec.cluster.domain import get_domain_profile
from xcquinox.alec.cluster.inputs import prepare_inputs
from xcquinox.alec.cluster.submit import submit_jobs
from xcquinox.alec.cluster.materialize import write_manifest


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Default cap on per-index resubmit attempts. An index that has already been
# retried this many times is reported but not re-submitted again.
_ATTEMPT_CAP = 3

# A held .harness.lock whose PID is dead AND whose file is older than this many
# seconds is considered stale and may be reclaimed (with a logged warning).
_LOCK_STALE_AGE_S = 6 * 3600

_LOCK_FILENAME = ".harness.lock"
_ATTEMPTS_FILENAME = "attempts.json"
_MANIFEST_FILENAME = "manifest.json"
_RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"
_TMP_PREFIX = ".mktmp_"


# ---------------------------------------------------------------------------
# Small generic helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Print a single operator-facing line (flushed)."""
    print(msg, flush=True)


def _on_login_node() -> bool:
    """True iff this process is NOT inside a SLURM allocation.

    SLURM sets ``$SLURM_JOB_ID`` for every job step (batch *and* interactive
    ``salloc``). Its absence means we are on a login node, where heavy compute
    is forbidden by cluster fair-use policy.
    """
    return not os.environ.get("SLURM_JOB_ID")


def _utc_stamp() -> str:
    """Filesystem-safe UTC timestamp ``YYYYmmddTHHMMSSZ`` for run-dir names."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json_atomic(payload, path: str) -> None:
    """Atomically write ``payload`` as pretty JSON (mkstemp + os.replace)."""
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_name = tempfile.mkstemp(prefix=_TMP_PREFIX, dir=out_dir)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)


# ---------------------------------------------------------------------------
# GridConfig <-> dict serialization (resolved_config.yaml round-trip)
# ---------------------------------------------------------------------------

def _config_to_raw_dict(cfg) -> dict:
    """Serialize a :class:`GridConfig` to a plain dict.

    The result is byte-for-byte loadable by :func:`load_grid_config`: every
    swept axis is a list, ``solvers`` is a name->dict mapping, and the nested
    sections (``hyperparams``/``inputs``/``cluster``) are plain dicts. This is
    what ``submit`` writes to ``resolved_config.yaml`` so the preflight (and a
    later ``repair-manifest``) can reconstruct the exact grid.
    """
    sweep = cfg.sweep
    raw = {
        "sweep": {
            "arch": list(sweep.arch),
            "loss": list(sweep.loss),
            "metric": list(sweep.metric),
            "subset_size": [int(s) for s in sweep.subset_size],
            "solver": list(sweep.solver),
        },
        "solvers": {
            name: dataclasses.asdict(sv) for name, sv in cfg.solvers.items()
        },
        "hyperparams": dataclasses.asdict(cfg.hyperparams),
        "inputs": dataclasses.asdict(cfg.inputs),
        "pretrain": dataclasses.asdict(cfg.pretrain),
        "cluster": dataclasses.asdict(cfg.cluster),
        "domain_profile": cfg.domain_profile,
        "on_precompute_failure": cfg.on_precompute_failure,
        "bh76_mode": cfg.bh76_mode,
    }
    return raw


def _write_resolved_config(cfg, run_dir: str) -> str:
    """Write ``<run_dir>/resolved_config.yaml`` — a round-trippable GridConfig.

    YAML is used (lazy ``import yaml``, matching ``load_grid_config``) so the
    file is human-auditable; the preflight reads it back with ``load_grid_config``.
    """
    raw = _config_to_raw_dict(cfg)
    path = os.path.join(run_dir, _RESOLVED_CONFIG_FILENAME)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "writing resolved_config.yaml requires PyYAML — "
            "install it with `pip install pyyaml`"
        ) from exc
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_name = tempfile.mkstemp(prefix=_TMP_PREFIX, dir=out_dir,
                                    suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=True)
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


# ---------------------------------------------------------------------------
# .harness.lock — run-dir mutual exclusion
# ---------------------------------------------------------------------------

class HarnessLockError(RuntimeError):
    """Raised when the run-dir ``.harness.lock`` is held by a live process."""


def _pid_is_alive(pid: int) -> bool:
    """True iff ``pid`` names a live process on THIS host (os.kill(pid, 0))."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but is owned by another user — still "alive".
        return True
    return True


def _lock_is_stale(info: dict, lock_path: str) -> bool:
    """Decide whether an existing lock may be reclaimed.

    A lock is stale iff EITHER the recorded PID is dead (only trustworthy when
    the lock was taken on THIS host — a PID number is meaningless across hosts)
    OR the lock file is older than :data:`_LOCK_STALE_AGE_S`. The age fallback
    covers a lock left by a process that died on a different node.
    """
    hostname = socket.gethostname()
    same_host = info.get("hostname") == hostname
    pid = int(info.get("pid", -1) or -1)
    if same_host and not _pid_is_alive(pid):
        return True
    try:
        age = datetime.now().timestamp() - os.path.getmtime(lock_path)
    except OSError:
        return False
    return age > _LOCK_STALE_AGE_S


def acquire_lock(run_dir: str, *, force: bool = False) -> str:
    """Acquire the ``<run_dir>/.harness.lock`` PID file; return its path.

    The lock file holds ``{pid, hostname, started_utc}``. On a collision the
    holder's liveness is checked: a stale lock (dead PID on the same host, or a
    file older than :data:`_LOCK_STALE_AGE_S`) is reclaimed with a logged
    warning. ``force=True`` reclaims unconditionally. A genuinely live holder
    raises :class:`HarnessLockError`.
    """
    lock_path = os.path.join(run_dir, _LOCK_FILENAME)
    payload = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        # Inspect the holder.
        info = {}
        try:
            with open(lock_path) as f:
                info = json.load(f)
        except (OSError, json.JSONDecodeError):
            info = {}
        if force:
            _log(f"WARNING: --force: reclaiming .harness.lock held by "
                 f"{info.get('hostname','?')}:{info.get('pid','?')}")
        elif _lock_is_stale(info, lock_path):
            _log(f"WARNING: reclaiming stale .harness.lock "
                 f"(holder {info.get('hostname','?')}:{info.get('pid','?')}, "
                 f"started {info.get('started_utc','?')})")
        else:
            raise HarnessLockError(
                f"{lock_path} is held by a live process "
                f"({info.get('hostname','?')}:{info.get('pid','?')}, started "
                f"{info.get('started_utc','?')}). Pass --force to override if "
                "you are certain no other harness command is running."
            )
        # Reclaim: overwrite in place.
        with open(lock_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return lock_path
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return lock_path


def release_lock(lock_path: str) -> None:
    """Best-effort removal of an acquired lock file."""
    try:
        os.unlink(lock_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Run-dir introspection helpers
# ---------------------------------------------------------------------------

def _read_manifest(run_dir: str) -> dict:
    """Load ``manifest.json``; raise FileNotFoundError if absent."""
    path = os.path.join(run_dir, _MANIFEST_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no {_MANIFEST_FILENAME} in {run_dir}"
        )
    with open(path) as f:
        return json.load(f)


def _try_read_manifest(run_dir: str):
    """Load ``manifest.json``, or None if it is missing/corrupt."""
    try:
        return _read_manifest(run_dir)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _spec_dir(run_dir: str, idx: int, width: int) -> str:
    """Per-spec checkpoint directory ``checkpoints/spec_<idx zero-padded>``."""
    return os.path.join(run_dir, "checkpoints", f"spec_{idx:0{width}d}")


def _scan_train_evidence(run_dir: str, n: int, width: int) -> list[int]:
    """Indices whose ``checkpoints/spec_<i>/`` shows a train task ran.

    "Ran" = a ``model.eqx`` OR a ``failure.json`` exists. This on-disk scan is
    the authoritative check ``resubmit-preflight`` uses to refuse a recovery
    when training has already started.
    """
    out = []
    for idx in range(n):
        d = _spec_dir(run_dir, idx, width)
        if (os.path.exists(os.path.join(d, "model.eqx"))
                or os.path.exists(os.path.join(d, "failure.json"))):
            out.append(idx)
    return out


def _failed_train_indices(run_dir: str, width: int, outcomes: dict) -> list[int]:
    """Train indices with NO ``model.eqx`` — candidates for resubmit recovery.

    An index counts as failed iff its checkpoint dir has no ``model.eqx`` (a
    produced model is the sole success signal); its ``outcomes`` entry then
    classifies *why*.
    """
    failed = []
    for idx in sorted(outcomes):
        d = _spec_dir(run_dir, idx, width)
        if not os.path.exists(os.path.join(d, "model.eqx")):
            failed.append(idx)
    return failed


def _read_failure_json(run_dir: str, idx: int, width: int):
    """Load ``checkpoints/spec_<i>/failure.json``, or None if absent/corrupt."""
    path = os.path.join(_spec_dir(run_dir, idx, width), "failure.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Failure classification (resubmit)
# ---------------------------------------------------------------------------

# Outcome string -> retry class. "oom"/"timeout" are retryable with possibly
# rerouted resources; everything else is deterministic (a code/config bug a
# blind retry will not fix).
_RETRYABLE = {"oom", "timeout"}


def _classify_failure(run_dir: str, idx: int, width: int, outcomes: dict) -> str:
    """Classify why train index ``idx`` failed: 'oom' / 'timeout' / 'deterministic'.

    Disk-first: a ``failure.json`` ``classification`` wins. Absent that, the
    ``reduce_outcomes`` value (its own ``sacct`` fallback) is used. Any value
    not in :data:`_RETRYABLE` collapses to ``'deterministic'``.
    """
    failure = _read_failure_json(run_dir, idx, width)
    if failure is not None:
        cls = (failure.get("classification") or "").strip().lower()
        if cls in _RETRYABLE:
            return cls
        return "deterministic"
    # No failure.json — fall back to the sacct-derived outcome.
    outcome = (outcomes.get(idx) or "").strip().lower()
    if outcome in _RETRYABLE:
        return outcome
    return "deterministic"


# ---------------------------------------------------------------------------
# Sparse-array helpers (resubmit)
# ---------------------------------------------------------------------------

def _sparse_array_spec(indices: list[int], throttle: int | None = None) -> str:
    """Build a SLURM ``--array`` value from an explicit index list.

    ``[3, 7, 9]`` -> ``"3,7,9"`` (plus ``"%T"`` if a throttle is given). Indices
    are sorted+deduplicated so the train and eval arrays are byte-identical.
    """
    uniq = sorted(set(int(i) for i in indices))
    body = ",".join(str(i) for i in uniq)
    if throttle is not None:
        body = f"{body}%{int(throttle)}"
    return body


def _train_generation(run_dir: str, idx: int) -> int:
    """The monotonic train ``generation`` covering index ``idx``.

    The newest non-superseded train record whose ``indices`` include ``idx``;
    its ``generation`` is the ``<g>`` suffix used to archive stale artifacts.
    Defaults to 0 if no record covers the index.
    """
    try:
        records = job_tracking.read_job_records(run_dir)
    except (FileNotFoundError, ValueError):
        return 0
    gens = [
        r["generation"] for r in records
        if r.get("kind") == "train" and not r.get("superseded", False)
        and idx in r.get("indices", [])
    ]
    return max(gens) if gens else 0


def _archive_stale_artifacts(run_dir: str, idx: int, width: int, gen: int) -> list[str]:
    """Rename an index's stale newest-generation artifacts to ``*.gen<g>``.

    Archives ``checkpoints/spec_<i>/{model.eqx,failure.json}`` and the per-spec
    ``eval/`` dir + ``eval_df.csv`` to a ``.gen<g>`` sibling. Refuses (raises)
    if a ``.gen<g>`` target already exists — that would clobber an older
    archive. Returns the list of archived paths (for the operator report).
    """
    spec_dir = _spec_dir(run_dir, idx, width)
    archived = []
    targets = [
        os.path.join(spec_dir, "model.eqx"),
        os.path.join(spec_dir, "failure.json"),
        os.path.join(spec_dir, "eval"),
        os.path.join(spec_dir, "eval_df.csv"),
    ]
    for src in targets:
        if not os.path.exists(src):
            continue
        dst = f"{src}.gen{gen}"
        if os.path.exists(dst):
            raise RuntimeError(
                f"archive target {dst!r} already exists; refusing to clobber "
                f"an existing gen-{gen} archive for spec index {idx}"
            )
        os.rename(src, dst)
        archived.append(dst)
    return archived


# ---------------------------------------------------------------------------
# attempts.json — per-index resubmit attempt counter
# ---------------------------------------------------------------------------

def _read_attempts(run_dir: str) -> dict:
    """Load ``attempts.json`` ({str(idx): count}), or {} if absent/corrupt."""
    path = os.path.join(run_dir, _ATTEMPTS_FILENAME)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_attempts(run_dir: str, attempts: dict) -> None:
    """Atomically write ``attempts.json``."""
    _write_json_atomic(attempts, os.path.join(run_dir, _ATTEMPTS_FILENAME))


# ---------------------------------------------------------------------------
# Spec content-hash re-verification
# ---------------------------------------------------------------------------

def _verify_spec_hashes(run_dir: str, manifest: dict, indices: list[int]) -> None:
    """Re-hash each spec file in ``indices`` and assert it matches the manifest.

    A resubmit re-runs the SAME ``specs/`` against a fresh array, so a spec
    file whose bytes drifted from the manifest record would silently train the
    wrong thing. Raises on the first mismatch.
    """
    from xcquinox.alec.cluster.materialize import _sha256_file

    by_index = {e["index"]: e for e in manifest.get("specs", [])}
    for idx in indices:
        entry = by_index.get(idx)
        if entry is None:
            raise RuntimeError(
                f"manifest has no record for spec index {idx}"
            )
        spec_path = os.path.join(run_dir, "specs", entry["spec_file"])
        if not os.path.exists(spec_path):
            raise RuntimeError(
                f"spec file {spec_path!r} for index {idx} is missing"
            )
        actual = _sha256_file(spec_path)
        expected = entry.get("sha256")
        if expected is not None and actual != expected:
            raise RuntimeError(
                f"spec file for index {idx} has content hash {actual} but the "
                f"manifest records {expected}; the specs/ directory drifted — "
                "refusing to resubmit against a mutated spec"
            )


# ---------------------------------------------------------------------------
# Job-id extraction
# ---------------------------------------------------------------------------

def _parse_job_id(proc) -> str:
    """Extract the array-job id from an ``sbatch --parsable`` CompletedProcess."""
    return proc.stdout.strip().split(";")[0].split()[0]


# ===========================================================================
# Subcommand: prepare
# ===========================================================================

def cmd_prepare(args) -> int:
    """``prepare`` — stage harness input artifacts.

    ``prepare`` builds the training-point pool, validates the existing subset
    ledger, and (by default) pre-warms the per-species CCSD external refs via
    ``prepare_inputs`` (skip-if-cached). The CCSD precompute is heavy compute,
    so it is refused on a login node (absent ``$SLURM_JOB_ID``): the operator
    should either use ``submit`` (whose preflight runs the precompute on a
    compute node) or an interactive ``salloc``. ``--no-recompute-refs`` skips
    the precompute entirely so ``prepare`` can validate the ledger cheaply on a
    login node.
    """
    cfg = load_grid_config(args.grid)
    recompute_refs = not args.no_recompute_refs

    if recompute_refs and _on_login_node():
        _log(
            "ERROR: `prepare` runs the heavy CCSD external-refs precompute and "
            "must NOT run on a login node (no $SLURM_JOB_ID detected).\n"
            "  - Use `python -m xcquinox.alec.cluster submit <grid>` — its "
            "preflight job runs the precompute on a compute node, or\n"
            "  - request an interactive node with `salloc` first, then re-run "
            "this command inside the allocation, or\n"
            "  - pass `--no-recompute-refs` to validate the ledger only "
            "(no precompute) — use this only when the refs are already staged."
        )
        return 2

    mode = "validate-only" if args.no_recompute_refs else "with refs precompute"
    _log(f"prepare: staging inputs ({mode}) from {args.grid}")
    staged = prepare_inputs(cfg, recompute_refs=recompute_refs)
    n_entries = len(staged.subset_ledger.get("entries", {}))
    _log(
        f"prepare: OK — pool of {len(staged.points)} training points, "
        f"{n_entries} subset-ledger entr{'y' if n_entries == 1 else 'ies'}; "
        f"ledger at {cfg.inputs.subset_ledger_path}"
    )
    return 0


# ===========================================================================
# Subcommand: submit
# ===========================================================================

def _make_run_dir(root: str) -> str:
    """Create a fresh, never-before-used run dir under ``<root>/runs/``.

    Name is ``run_<UTC-timestamp>`` with a ``_<counter>`` suffix appended on a
    same-second collision. ``os.makedirs(exist_ok=False)`` guarantees the dir
    is brand-new (so we never overwrite a prior run's artifacts).
    """
    runs_root = os.path.join(root, "runs")
    os.makedirs(runs_root, exist_ok=True)
    base = f"run_{_utc_stamp()}"
    candidate = os.path.join(runs_root, base)
    counter = 1
    while True:
        try:
            os.makedirs(candidate, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = os.path.join(runs_root, f"{base}_{counter}")
            counter += 1


def cmd_submit(args) -> int:
    """``submit`` — create a fresh run dir and (dry-run by default) submit.

    Loads + semantically validates the grid, creates a timestamped run dir,
    writes ``resolved_config.yaml`` + ``scripts/`` + ``logs/``, then calls
    ``submit_jobs`` (dry-run unless ``--submit``) which renders + submits the
    4-stage pretrain → preflight → train → eval graph.
    """
    cfg = load_grid_config(args.grid)
    domain = get_domain_profile(cfg.domain_profile)
    validate_grid_semantics(cfg, domain)

    root = args.run_root or cfg.inputs.output_root
    run_dir = _make_run_dir(root)
    os.makedirs(os.path.join(run_dir, "scripts"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    _write_resolved_config(cfg, run_dir)
    _log(f"submit: created run dir {run_dir}")

    result = submit_jobs(cfg, run_dir, submit=args.submit, force=args.force)

    if result.get("dry_run", True):
        _log(f"submit: DRY-RUN ({result['n_specs']} specs, array "
             f"0-{result['array_max']}, {result['n_archs']} distinct arch(s), "
             f"pretrain array 0-{result['pretrain_array_max']}, "
             f"device={result['device']}). "
             "No SLURM call was made; pass --submit to submit for real.")
        for line in result.get("commands", []):
            _log(f"  would run: {line}")
    else:
        ids = result.get("job_ids", {})
        _log(f"submit: SUBMITTED ({result['n_specs']} specs, "
             f"{result['n_archs']} distinct arch(s)) — "
             f"pretrain={ids.get('pretrain')} "
             f"preflight={ids.get('preflight')} train={ids.get('train')} "
             f"eval={ids.get('eval')}")
    _log(f"submit: run dir = {run_dir}")
    return 0


# ===========================================================================
# Subcommand: status
# ===========================================================================

def _pretrain_status(run_dir: str) -> str | None:
    """Lightweight pretrain-stage status line, or None if it cannot be checked.

    Pretrain is a small up-front stage — it gets no per-index
    ``reduce_outcomes``. The check is purely on-disk: for each distinct
    architecture in the resolved config, the pretrain worker writes
    ``xnet.eqx`` + ``cnet.eqx`` into ``<pretrain_root>/<arch>/``. We report how
    many of those checkpoint pairs are present.
    """
    cfg_path = os.path.join(run_dir, _RESOLVED_CONFIG_FILENAME)
    if not os.path.exists(cfg_path):
        return None
    try:
        cfg = load_grid_config(cfg_path)
    except Exception:
        return None
    archs = sorted(set(cfg.sweep.arch))
    root = cfg.pretrain.pretrain_root
    done = 0
    for arch in archs:
        d = os.path.join(root, arch)
        if (os.path.exists(os.path.join(d, "xnet.eqx"))
                and os.path.exists(os.path.join(d, "cnet.eqx"))):
            done += 1
    return f"{done}/{len(archs)} architecture checkpoint pair(s) present"


def cmd_status(args) -> int:
    """``status`` — read-only per-index outcome report (no lock taken).

    Aggregates ``train`` and ``eval`` outcomes via ``reduce_outcomes`` across
    all non-superseded ``jobs.json`` generations, diffs against the manifest's
    ``n_specs``, and prints counts + an actionable remedy line. The pretrain
    stage gets a lightweight on-disk checkpoint-presence check (no per-index
    reduction — pretrain is a handful of jobs). A
    :class:`SlurmTransientError` is reported, not crashed on.
    """
    run_dir = os.path.abspath(args.run_dir)
    manifest = _try_read_manifest(run_dir)
    if manifest is None:
        _log(f"status: {run_dir}/manifest.json is missing or corrupt.")
        _log("  remedy: run `repair-manifest <run_dir>` to rebuild it.")
        return 1

    n_specs = int(manifest["n_specs"])
    _log(f"status: run dir {run_dir} — manifest records {n_specs} spec(s).")

    pt_status = _pretrain_status(run_dir)
    if pt_status is not None:
        _log(f"  pretrain: {pt_status}")

    try:
        train = job_tracking.reduce_outcomes(run_dir, "train")
        ev = job_tracking.reduce_outcomes(run_dir, "eval")
    except job_tracking.SlurmTransientError:
        _log("status: SLURM controller unreachable — retry. "
             "(on-disk evidence below may be partial)")
        return 1

    def _tally(outcomes: dict) -> dict:
        counts: dict = {}
        for v in outcomes.values():
            counts[v] = counts.get(v, 0) + 1
        return counts

    train_counts = _tally(train)
    eval_counts = _tally(ev)

    train_success = sum(1 for v in train.values() if v == "success")
    train_failed = sum(
        1 for v in train.values()
        if v in ("oom", "timeout") or v.startswith("failure")
    )
    train_never = sum(
        1 for v in train.values()
        if v in ("dependency_never_satisfied", "unknown_sacct_purged")
    )
    eval_never = sum(
        1 for v in ev.values()
        if v in ("dependency_never_satisfied", "unknown_sacct_purged")
    )

    _log("  train: " + ", ".join(
        f"{k}={v}" for k, v in sorted(train_counts.items())) or "  train: (none)")
    _log(f"    success={train_success}  failed={train_failed}  "
         f"never-ran={train_never}")
    _log("  eval:  " + ", ".join(
        f"{k}={v}" for k, v in sorted(eval_counts.items())) or "  eval: (none)")
    _log(f"    never-scheduled={eval_never}")

    # diff vs manifest — covered indices should equal n_specs.
    if len(train) != n_specs:
        _log(f"  WARNING: reduce_outcomes covered {len(train)} train indices "
             f"but manifest records {n_specs} — manifest may be inconsistent.")

    # Remedy line.
    preflight_dead = (
        train_never == n_specs and train_success == 0 and train_failed == 0
    )
    if manifest is None or len(train) != n_specs:
        _log("  remedy: `repair-manifest <run_dir>` — manifest is "
             "corrupt/inconsistent.")
    elif preflight_dead:
        _log("  remedy: `resubmit-preflight <run_dir>` — the preflight job "
             "appears to have failed (no train task ran).")
    elif train_failed > 0:
        _log("  remedy: `resubmit <run_dir>` — re-run the failed train "
             "task(s).")
    else:
        _log("  remedy: none — no failed train tasks detected.")
    return 0


# ===========================================================================
# Subcommand: resubmit
# ===========================================================================

def cmd_resubmit(args) -> int:
    """``resubmit`` — recover FAILED TRAIN tasks (preflight already succeeded).

    See the module/plan docstring for the full contract. Summary:
      - scan for train indices with no ``model.eqx``;
      - classify each (failure.json, else sacct fallback);
      - keep retryable (oom/timeout) indices below the attempt cap;
      - re-verify spec hashes, archive stale ``*.gen<g>`` artifacts;
      - submit a sparse train array + a byte-identical sparse eval array
        (``aftercorr``), dry-run unless ``--submit``;
      - best-effort rollback if the eval ``sbatch`` fails post-train.
    """
    run_dir = os.path.abspath(args.run_dir)
    manifest = _try_read_manifest(run_dir)
    if manifest is None:
        _log(f"resubmit: {run_dir}/manifest.json is missing/corrupt — "
             "run `repair-manifest` first.")
        return 1
    n_specs = int(manifest["n_specs"])
    width = int(manifest["width"])

    lock_path = None
    try:
        lock_path = acquire_lock(run_dir, force=args.force)
    except HarnessLockError as exc:
        _log(f"resubmit: {exc}")
        return 1

    try:
        try:
            outcomes = job_tracking.reduce_outcomes(run_dir, "train")
        except job_tracking.SlurmTransientError:
            _log("resubmit: SLURM controller unreachable — retry.")
            return 1

        failed = _failed_train_indices(run_dir, width, outcomes)
        if not failed:
            _log("resubmit: no failed train tasks — nothing to do.")
            return 0

        attempts = _read_attempts(run_dir)
        cap = args.attempt_cap

        retry: list[int] = []
        skipped_det: list[int] = []
        skipped_cap: list[int] = []
        defaulted: list[int] = []      # got default knobs (no retry partition)
        classes: dict = {}
        for idx in failed:
            cls = _classify_failure(run_dir, idx, width, outcomes)
            classes[idx] = cls
            if cls == "deterministic":
                skipped_det.append(idx)
                continue
            done = int(attempts.get(str(idx), 0))
            if done >= cap:
                skipped_cap.append(idx)
                continue
            retry.append(idx)

        # Resolve retry-knob routing from resolved_config.yaml.
        cfg = load_grid_config(
            os.path.join(run_dir, _RESOLVED_CONFIG_FILENAME)
        )
        cl = cfg.cluster
        for idx in retry:
            cls = classes[idx]
            if cls == "oom" and not cl.oom_retry_partition and not cl.oom_retry_mem:
                defaulted.append(idx)
            if cls == "timeout" and not cl.timeout_retry_partition \
                    and not cl.timeout_retry_time:
                defaulted.append(idx)

        _log(f"resubmit: {len(failed)} failed train task(s): "
             f"retry={sorted(retry)} skip-deterministic={sorted(skipped_det)} "
             f"skip-attempt-cap={sorted(skipped_cap)}")
        for idx in sorted(skipped_det):
            _log(f"  index {idx}: deterministic failure "
                 f"({classes[idx]}) — NOT retried; inspect failure.json.")
        if defaulted:
            _log(f"  indices {sorted(set(defaulted))} use DEFAULT "
                 "partition/resources — no dedicated retry knob configured.")

        if not retry:
            _log("resubmit: no retryable indices below the attempt cap.")
            return 0

        # Re-verify spec content hashes before reusing specs/.
        _verify_spec_hashes(run_dir, manifest, retry)

        train_array = _sparse_array_spec(retry, cl.array_throttle)
        eval_array = _sparse_array_spec(retry, cl.eval_array_throttle)
        # aftercorr requires byte-identical index lists (throttle aside).
        train_idx_body = train_array.split("%", 1)[0]
        eval_idx_body = eval_array.split("%", 1)[0]
        assert train_idx_body == eval_idx_body, (
            f"resubmit: train array indices {train_idx_body!r} != eval array "
            f"indices {eval_idx_body!r}"
        )

        train_script = os.path.join(run_dir, "scripts", "train_array.sbatch")
        eval_script = os.path.join(run_dir, "scripts", "eval_array.sbatch")

        if not args.submit:
            _log(f"resubmit: DRY-RUN — would archive stale artifacts for "
                 f"{sorted(retry)} then submit:")
            _log(f"  sbatch --parsable --array={train_array} {train_script}")
            _log(f"  sbatch --parsable --array={eval_array} "
                 f"--dependency=aftercorr:<TRAIN_ID> {eval_script}")
            _log("resubmit: no SLURM call made; pass --submit to submit.")
            return 0

        # --- real submission --------------------------------------------------
        # Archive stale artifacts for every retried index first.
        for idx in retry:
            gen = _train_generation(run_dir, idx)
            archived = _archive_stale_artifacts(run_dir, idx, width, gen)
            if archived:
                _log(f"  index {idx}: archived {len(archived)} artifact(s) "
                     f"-> *.gen{gen}")

        # Submit the sparse train array.
        train_cmd = [
            "sbatch", "--parsable", f"--array={train_array}", train_script,
        ]
        proc = job_tracking._run_slurm(train_cmd)
        train_id = _parse_job_id(proc)

        # Submit the matching sparse eval array (aftercorr on the new train).
        eval_cmd = [
            "sbatch", "--parsable", f"--array={eval_array}",
            f"--dependency=aftercorr:{train_id}", eval_script,
        ]
        try:
            proc = job_tracking._run_slurm(eval_cmd)
            eval_id = _parse_job_id(proc)
        except Exception as exc:
            # Best-effort rollback: cancel the just-submitted train array.
            _log(f"resubmit: eval sbatch failed ({exc}); rolling back the "
                 f"train array {train_id} via scancel.")
            try:
                job_tracking._run_slurm(["scancel", str(train_id)])
            except Exception:
                _log(f"resubmit: WARNING scancel of train {train_id} also "
                     "failed — that array may be orphaned; cancel it manually.")
            _log("resubmit: nothing appended to jobs.json.")
            return 1

        # Both sbatch calls succeeded — now record + bump attempts.
        job_tracking.append_job_record(run_dir, "train", train_id, retry)
        job_tracking.append_job_record(run_dir, "eval", eval_id, retry)
        for idx in retry:
            attempts[str(idx)] = int(attempts.get(str(idx), 0)) + 1
        _write_attempts(run_dir, attempts)

        _log(f"resubmit: SUBMITTED sparse arrays — train={train_id} "
             f"eval={eval_id} for indices {sorted(retry)}.")
        return 0
    finally:
        if lock_path is not None:
            release_lock(lock_path)


# ===========================================================================
# Subcommand: resubmit-preflight
# ===========================================================================

def cmd_resubmit_preflight(args) -> int:
    """``resubmit-preflight`` — recover a FAILED/timed-out pretrain/preflight.

    Refuses unless the run is genuinely pretrain/preflight-stuck (see the
    plan): a complete manifest from a non-superseded preflight generation, OR
    any on-disk train evidence, both block the recovery. Re-submits the whole
    pretrain->preflight->train->eval graph; only after all four new ``sbatch``
    calls succeed does it ``scancel`` the old pretrain/train/eval arrays and
    ``mark_superseded``.
    """
    run_dir = os.path.abspath(args.run_dir)

    cfg_path = os.path.join(run_dir, _RESOLVED_CONFIG_FILENAME)
    if not os.path.exists(cfg_path):
        _log(f"resubmit-preflight: {cfg_path} not found — cannot reconstruct "
             "the grid. Use a fresh run dir (`submit`).")
        return 1
    cfg = load_grid_config(cfg_path)
    n_cells = len(expand_grid(cfg))

    lock_path = None
    try:
        lock_path = acquire_lock(run_dir, force=args.force)
    except HarnessLockError as exc:
        _log(f"resubmit-preflight: {exc}")
        return 1

    try:
        manifest = _try_read_manifest(run_dir)

        # Refusal 1: a complete manifest means the preflight SUCCEEDED.
        if manifest is not None:
            man_n = int(manifest.get("n_specs", -1))
            man_specs = manifest.get("specs", [])
            if man_n == n_cells and len(man_specs) == n_cells:
                _log("resubmit-preflight: REFUSING — manifest.json records a "
                     f"complete {n_cells}-cell materialization; the preflight "
                     "succeeded. Use `resubmit` to recover failed train "
                     "tasks instead.")
                return 1
            # Refusal 2: grid changed -> a fresh run dir is required.
            if man_n != n_cells:
                _log(f"resubmit-preflight: REFUSING — manifest records "
                     f"n_specs={man_n} but the grid now expands to {n_cells} "
                     "cells. A changed grid must use a fresh run dir "
                     "(`submit`).")
                return 1

        # Refusal 3: any on-disk train evidence -> a train task already ran.
        # This is authoritative — an empty attempts.json is NOT sufficient.
        scan_width = (
            int(manifest["width"]) if manifest is not None
            else max(4, len(str(n_cells - 1)))
        )
        evidence = _scan_train_evidence(run_dir, n_cells, scan_width)
        if evidence:
            _log("resubmit-preflight: REFUSING — found train-task evidence "
                 f"(model.eqx/failure.json) for indices {evidence}. A train "
                 "task has run; use `resubmit` instead.")
            return 1

        # Identify the old train/eval arrays to supersede + scancel.
        try:
            records = job_tracking.read_job_records(run_dir)
        except (FileNotFoundError, ValueError):
            records = []

        def _newest_live(kind):
            live = [r for r in records if r.get("kind") == kind
                    and not r.get("superseded", False)]
            if not live:
                return None
            return max(live, key=lambda r: r["generation"])

        old_pretrain = _newest_live("pretrain")
        old_train = _newest_live("train")
        old_eval = _newest_live("eval")

        if not args.submit:
            _log("resubmit-preflight: DRY-RUN — would re-submit the full "
                 "pretrain->preflight->train->eval graph via "
                 "submit_jobs(force=True), then scancel + mark_superseded the "
                 "old pretrain/train/eval arrays.")
            if old_pretrain:
                _log(f"  old pretrain array {old_pretrain['array_job_id']} "
                     f"(gen {old_pretrain['generation']}) would be cancelled.")
            if old_train:
                _log(f"  old train array {old_train['array_job_id']} "
                     f"(gen {old_train['generation']}) would be cancelled.")
            if old_eval:
                _log(f"  old eval array {old_eval['array_job_id']} "
                     f"(gen {old_eval['generation']}) would be cancelled.")
            _log("resubmit-preflight: no SLURM call made; pass --submit.")
            return 0

        # --- real re-submission ----------------------------------------------
        # submit_jobs does its own best-effort scancel rollback if any of its
        # four sbatch calls is rejected; force=True bypasses the live-jobs
        # guard (the old graph is intentionally still recorded here).
        result = submit_jobs(cfg, run_dir, submit=True, force=True)
        new_ids = result.get("job_ids", {})
        _log(f"resubmit-preflight: re-submitted graph — "
             f"pretrain={new_ids.get('pretrain')} "
             f"preflight={new_ids.get('preflight')} "
             f"train={new_ids.get('train')} eval={new_ids.get('eval')}.")

        # All four new sbatch calls succeeded (submit_jobs would have raised
        # otherwise). NOW, and only now: scancel old pretrain/train/eval ->
        # mark_superseded. A scancel failure aborts before mark_superseded
        # so a superseded generation always has a live successor.
        scancel_ok = True
        orphans = []
        for rec in (old_pretrain, old_train, old_eval):
            if rec is None:
                continue
            try:
                job_tracking._run_slurm(["scancel", str(rec["array_job_id"])])
            except Exception as exc:
                scancel_ok = False
                orphans.append(rec["array_job_id"])
                _log(f"resubmit-preflight: WARNING scancel of "
                     f"{rec['kind']} array {rec['array_job_id']} failed "
                     f"({exc}).")

        if not scancel_ok:
            _log("resubmit-preflight: a scancel failed — SKIPPING "
                 f"mark_superseded. Orphaned old array id(s): {orphans}. "
                 "Cancel them manually; the old jobs.json records were left "
                 "un-superseded on purpose.")
            return 1

        # scancel succeeded for every old array — mark them superseded.
        for rec in (old_pretrain, old_train, old_eval):
            if rec is None:
                continue
            job_tracking.mark_superseded(run_dir, rec["kind"],
                                         rec["generation"])
        _log("resubmit-preflight: old pretrain/train/eval arrays cancelled "
             "and marked superseded.")
        return 0
    finally:
        if lock_path is not None:
            release_lock(lock_path)


# ===========================================================================
# Subcommand: repair-manifest
# ===========================================================================

def cmd_repair_manifest(args) -> int:
    """``repair-manifest`` — rebuild a corrupt OR missing ``manifest.json``.

    Never reads the old manifest (absent and corrupt are identical). Rebuilds
    from ``resolved_config.yaml`` (the deterministic idx->GridCell map) plus the
    on-disk ``specs/*.spec`` files (re-hashed). Asserts the spec count equals N,
    cross-checks the pad width, and rewrites ONLY ``manifest.json`` atomically.
    """
    run_dir = os.path.abspath(args.run_dir)

    cfg_path = os.path.join(run_dir, _RESOLVED_CONFIG_FILENAME)
    if not os.path.exists(cfg_path):
        _log(f"repair-manifest: {cfg_path} not found — the resolved config is "
             "the only source of truth for the grid. It is unrecoverable; "
             "start a fresh run dir with `submit`.")
        return 1
    try:
        cfg = load_grid_config(cfg_path)
    except Exception as exc:
        # ANY failure to parse/build the resolved config (ValueError, a YAML
        # ParserError, an OSError, ...) means it is unrecoverable — the grid
        # cannot be reconstructed, so direct the user to a fresh run dir.
        _log(f"repair-manifest: {cfg_path} is unrecoverable ({exc}); start a "
             "fresh run dir with `submit`.")
        return 1

    lock_path = None
    try:
        lock_path = acquire_lock(run_dir, force=args.force)
    except HarnessLockError as exc:
        _log(f"repair-manifest: {exc}")
        return 1

    try:
        cells = expand_grid(cfg)
        n = len(cells)

        specs_dir = os.path.join(run_dir, "specs")
        if not os.path.isdir(specs_dir):
            _log(f"repair-manifest: {specs_dir} does not exist — no specs to "
                 "rebuild from. The preflight never materialized this run; "
                 "use `resubmit-preflight` or a fresh run dir.")
            return 1

        # Discover spec files; derive the on-disk pad width from filenames.
        spec_files = sorted(
            f for f in os.listdir(specs_dir)
            if f.startswith("spec_") and f.endswith(".spec")
        )
        if len(spec_files) != n:
            _log(f"repair-manifest: found {len(spec_files)} spec file(s) in "
                 f"{specs_dir} but the grid expands to {n} cell(s). The spec "
                 "set is incomplete — cannot rebuild a trustworthy manifest. "
                 "Use `resubmit-preflight` to re-materialize.")
            return 1

        # Pad width from the actual filenames; cross-check vs the formula.
        core = spec_files[0][len("spec_"):-len(".spec")]
        disk_width = len(core)
        expected_width = max(4, len(str(n - 1))) if n > 0 else 4
        if disk_width != expected_width:
            _log(f"repair-manifest: WARNING on-disk pad width {disk_width} "
                 f"!= computed width {expected_width}; using the on-disk "
                 "width to match the actual files.")
        width = disk_width

        # Build the deterministic idx->path list, then write_manifest.
        from xcquinox.alec.cluster.materialize import _spec_filename

        paths = []
        for idx in range(n):
            fname = _spec_filename(idx, width)
            full = os.path.join(specs_dir, fname)
            if not os.path.exists(full):
                _log(f"repair-manifest: expected spec file {fname!r} for "
                     f"index {idx} is missing from {specs_dir}. Cannot rebuild.")
                return 1
            paths.append(full)

        manifest_path = write_manifest(cells, paths, run_dir)
        _log(f"repair-manifest: rebuilt {manifest_path} — {n} spec(s), "
             f"pad width {width}. model.eqx / jobs.json / attempts.json / "
             "checkpoints/ were left untouched.")
        return 0
    finally:
        if lock_path is not None:
            release_lock(lock_path)


# ===========================================================================
# argparse wiring
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser with all six subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m xcquinox.alec.cluster",
        description="HPC (SLURM) training-harness CLI for xcquinox.alec.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser(
        "prepare", help="stage harness input artifacts (CCSD refs, ledger)")
    p_prepare.add_argument("grid", help="path to the grid config (.yaml/.json)")
    p_prepare.add_argument(
        "--no-recompute-refs", action="store_true",
        help="skip the heavy CCSD external-refs precompute and only validate "
             "the subset ledger (cheap; safe on a login node)")
    p_prepare.set_defaults(func=cmd_prepare)

    p_submit = sub.add_parser(
        "submit", help="create a run dir and submit the 3-stage job graph")
    p_submit.add_argument("grid", help="path to the grid config (.yaml/.json)")
    p_submit.add_argument(
        "--submit", action="store_true",
        help="actually submit to SLURM (default: dry-run)")
    p_submit.add_argument(
        "--force", action="store_true",
        help="override the double-submit guard")
    p_submit.add_argument(
        "--run-root", default=None,
        help="root for runs/ (default: cfg.inputs.output_root)")
    p_submit.set_defaults(func=cmd_submit)

    p_status = sub.add_parser(
        "status", help="read-only per-index outcome report")
    p_status.add_argument("run_dir", help="the run directory")
    p_status.set_defaults(func=cmd_status)

    p_resub = sub.add_parser(
        "resubmit", help="recover failed TRAIN tasks (sparse arrays)")
    p_resub.add_argument("run_dir", help="the run directory")
    p_resub.add_argument(
        "--submit", action="store_true",
        help="actually submit (default: dry-run)")
    p_resub.add_argument(
        "--force", action="store_true",
        help="reclaim a held .harness.lock")
    p_resub.add_argument(
        "--attempt-cap", type=int, default=_ATTEMPT_CAP,
        help=f"max resubmit attempts per index (default {_ATTEMPT_CAP})")
    p_resub.set_defaults(func=cmd_resubmit)

    p_rspf = sub.add_parser(
        "resubmit-preflight", help="recover a failed/timed-out preflight job")
    p_rspf.add_argument("run_dir", help="the run directory")
    p_rspf.add_argument(
        "--submit", action="store_true",
        help="actually submit (default: dry-run)")
    p_rspf.add_argument(
        "--force", action="store_true",
        help="reclaim a held .harness.lock")
    p_rspf.set_defaults(func=cmd_resubmit_preflight)

    p_repair = sub.add_parser(
        "repair-manifest", help="rebuild a corrupt/missing manifest.json")
    p_repair.add_argument("run_dir", help="the run directory")
    p_repair.add_argument(
        "--force", action="store_true",
        help="reclaim a held .harness.lock")
    p_repair.set_defaults(func=cmd_repair_manifest)

    return parser


def main(argv=None) -> int:
    """Parse ``argv`` and dispatch to the selected subcommand.

    Returns the subcommand's integer exit code. ``main`` only dispatches;
    every subcommand's logic lives in its own ``cmd_*`` function.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    sys.exit(main())
