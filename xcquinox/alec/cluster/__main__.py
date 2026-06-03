"""xcquinox.alec.cluster.__main__: CLI dispatch for the HPC training harness.

Invoked as ``python -m xcquinox.alec.cluster <subcommand> ...``. This module is
purely the operator-facing front-end: it parses arguments, owns the run-dir
lock, and wires the already-built ``cluster/`` modules together. Subcommands:

  - ``prepare``: stage input artifacts (CCSD refs, ledger, ...).
  - ``submit``: create a fresh run dir + submit the 4-stage graph.
  - ``submit-eval``: submit the deferred eval array for an existing run.
  - ``status``: read-only per-index outcome report.
  - ``results``: aggregate per-spec eval metrics (MAE etc.).
  - ``resubmit``: recover FAILED TRAIN tasks (preflight succeeded).
  - ``resubmit-preflight``: recover a FAILED/timed-out pretrain/preflight.
  - ``repair-manifest``: rebuild a corrupt/missing ``manifest.json``.
  - ``pull``: rsync a run dir from the cluster back to local
    for post-processing. Category-aware (``--category alpha_off/runs``).
  - ``list-runs``: discover ``run_<UTC>Z`` dirs under
    ``--remote-root``, grouped by category. See
    ``xcquinox/alec/cluster/sync.py`` for both.

Design rules (enforced below at their use sites):

  - Dry-run is the default for every submitting subcommand; real SLURM
    submission happens only with ``--submit``.
  - Every SLURM subprocess goes through ``job_tracking._run_slurm``, the
    one seam tests monkeypatch. ``submit_jobs`` already routes through it; the
    sparse-array resubmit paths call it directly for ``sbatch``/``scancel``.
  - ``main`` only dispatches; each subcommand's logic is in its own function;
    small shared helpers (lock, failed-index scan, artifact archive, sparse
    array string) are factored out.
  - Login-node guard: ``prepare`` runs the heavy CCSD external-refs
    precompute and is refused on a login node (absent ``$SLURM_JOB_ID``)
    unless ``--no-recompute-refs`` is passed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from xcquinox.alec.cluster import analyze
from xcquinox.alec.cluster import job_tracking
from xcquinox.alec.cluster import sync as _sync
from xcquinox.alec.cluster.grid_config import (
    expand_grid,
    load_grid_config,
    pretrain_checkpoint_dir,
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

    SLURM sets ``$SLURM_JOB_ID`` for every job step (batch and interactive
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
        "use_polarized_correlation": cfg.use_polarized_correlation,
        "held_out_strict": cfg.held_out_strict,
        "defer_eval": cfg.defer_eval,
        # inline_eval MUST round-trip: load_grid_config reads it
        # (raw.get("inline_eval", ...)), and recovery/resubmit paths re-load
        # resolved_config.yaml: omitting it silently reverts an inline-eval run
        # to a separate eval array.
        "inline_eval": cfg.inline_eval,
    }
    return raw


def _write_resolved_config(cfg, run_dir: str) -> str:
    """Write ``<run_dir>/resolved_config.yaml``: a round-trippable GridConfig.

    YAML is used (lazy ``import yaml``, matching ``load_grid_config``) so the
    file is human-auditable; the preflight reads it back with ``load_grid_config``.
    """
    raw = _config_to_raw_dict(cfg)
    path = os.path.join(run_dir, _RESOLVED_CONFIG_FILENAME)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "writing resolved_config.yaml requires PyYAML, "
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
# .harness.lock: run-dir mutual exclusion
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
        # The process exists but is owned by another user, still "alive".
        return True
    return True


def _lock_is_stale(info: dict, lock_path: str) -> bool:
    """Decide whether an existing lock may be reclaimed.

    A lock is stale iff EITHER the recorded PID is dead (only trustworthy when
    the lock was taken on THIS host, a PID number is meaningless across hosts)
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
    """Train indices with NO ``model.eqx``: candidates for resubmit recovery.

    An index counts as failed iff its checkpoint dir has no ``model.eqx`` (a
    produced model is the sole success signal); its ``outcomes`` entry then
    classifies why.
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

# failure.json classifications that ARE retryable but under a different name.
# A wall-clock pre-kill grace SIGTERM is recorded by ``_train_task`` as
# "killed_by_signal": it is a timeout in all but name (the job ran out of
# wall), so route it like a timeout (longer-wall ``timeout_retry`` resources).
_FAILURE_CLASS_ALIASES = {"killed_by_signal": "timeout"}


def _classify_failure(run_dir: str, idx: int, width: int, outcomes: dict) -> str:
    """Classify why train index ``idx`` failed: 'oom' / 'timeout' / 'deterministic'.

    Disk-first: a ``failure.json`` ``classification`` wins. Absent that, the
    ``reduce_outcomes`` value (its own ``sacct`` fallback) is used. Any value
    not in :data:`_RETRYABLE` collapses to ``'deterministic'``.
    """
    failure = _read_failure_json(run_dir, idx, width)
    if failure is not None:
        cls = (failure.get("classification") or "").strip().lower()
        cls = _FAILURE_CLASS_ALIASES.get(cls, cls)
        if cls in _RETRYABLE:
            return cls
        return "deterministic"
    # No failure.json: fall back to the sacct-derived outcome.
    outcome = (outcomes.get(idx) or "").strip().lower()
    if outcome in _RETRYABLE:
        return outcome
    return "deterministic"


# ---------------------------------------------------------------------------
# Sparse-array helpers (resubmit)
# ---------------------------------------------------------------------------

def _retry_resource_flags(cls: str, cl) -> list[str]:
    """sbatch resource-override flags for resubmitting a ``cls`` failure.

    These override the rendered train script's baked ``#SBATCH`` directives so
    a retry actually gets different resources:
      - ``oom``     -> ``--partition=oom_retry_partition`` / ``--mem=oom_retry_mem``
      - ``timeout`` -> ``--partition=timeout_retry_partition`` / ``--time=timeout_retry_time``
    Each flag is emitted only when its knob is set; an unset class falls back to
    the script's defaults (returns ``[]``).
    """
    flags: list[str] = []
    if cls == "oom":
        if cl.oom_retry_partition:
            flags.append(f"--partition={cl.oom_retry_partition}")
        if cl.oom_retry_mem:
            flags.append(f"--mem={cl.oom_retry_mem}")
        if getattr(cl, "oom_retry_force_cpu", False):
            # Force the retry onto the CPU: release the GPU and make JAX ignore
            # any still-visible device. Without this the retry resubmits the
            # gpu-rendered script and re-OOMs on the same GPU (CW2-M1).
            flags.append("--gres=gpu:0")
            flags.append("--export=ALL,JAX_PLATFORMS=cpu")
    elif cls == "timeout":
        if cl.timeout_retry_partition:
            flags.append(f"--partition={cl.timeout_retry_partition}")
        if cl.timeout_retry_time:
            flags.append(f"--time={cl.timeout_retry_time}")
    return flags


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
    if a ``.gen<g>`` target already exists, that would clobber an older
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
# attempts.json: per-index resubmit attempt counter
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
                f"manifest records {expected}; the specs/ directory drifted, "
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
    """``prepare``: stage harness input artifacts.

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
            "  - Use `python -m xcquinox.alec.cluster submit <grid>`: its "
            "preflight job runs the precompute on a compute node, or\n"
            "  - request an interactive node with `salloc` first, then re-run "
            "this command inside the allocation, or\n"
            "  - pass `--no-recompute-refs` to validate the ledger only "
            "(no precompute), use this only when the refs are already staged."
        )
        return 2

    mode = "validate-only" if args.no_recompute_refs else "with refs precompute"
    _log(f"prepare: staging inputs ({mode}) from {args.grid}")
    staged = prepare_inputs(cfg, recompute_refs=recompute_refs)
    n_entries = len(staged.subset_ledger)
    _log(
        f"prepare: OK, pool of {len(staged.points)} training points, "
        f"{n_entries} subset-ledger entr{'y' if n_entries == 1 else 'ies'}; "
        f"ledger at {cfg.inputs.subset_ledger_path}"
    )
    return 0


# ===========================================================================
# Subcommand: submit
# ===========================================================================

def _apply_partition_overrides(cfg, args):
    """Return a copy of ``cfg`` with CLI-resolved partitions on every stage.

    The submit CLI is the sole source of partitions (the config carries no
    default). ``--partition`` is the required base; the optional per-stage
    ``--{train,eval,preflight,pretrain}-partition`` flags override it for one
    stage and otherwise fall back to the base. The resolved values are written
    onto ``cfg.cluster`` so they (a) feed ``render_sbatch`` for this submission
    and (b) round-trip into ``resolved_config.yaml``, so recovery commands,
    which re-render from that file, inherit the same partitions with no flag.

    Render-time mapping (see ``submit.render_sbatch``): the TRAIN stage uses
    ``cl.partition`` directly; EVAL/PREFLIGHT/PRETRAIN use
    ``cl.<stage>_partition or cl.partition``. Setting each per-stage field to a
    non-empty resolved value makes that fallback a no-op, so every stage lands
    on exactly its resolved partition.
    """
    base = args.partition
    train_p = args.train_partition or base
    eval_p = args.eval_partition or base
    preflight_p = args.preflight_partition or base
    pretrain_p = args.pretrain_partition or base
    cluster = dataclasses.replace(
        cfg.cluster,
        partition=train_p,
        eval_partition=eval_p,
        preflight_partition=preflight_p,
        pretrain_partition=pretrain_p,
    )
    return dataclasses.replace(cfg, cluster=cluster)


def _apply_step_overrides(cfg, args):
    """Return a copy of ``cfg`` with CLI-resolved step counts.

    ``--n-steps`` overrides the per-spec training-optimization steps
    (``hyperparams.n_steps``); ``--pretrain-n-steps`` overrides the pretraining
    steps (``pretrain.n_steps``). Each unset flag leaves the config value, so
    omitting both is a no-op. The resolved counts ride into
    ``resolved_config.yaml`` and are consumed by the preflight (which
    materializes the TrainingSpecs) and the pretrain stage.
    """
    if args.n_steps is not None:
        cfg = dataclasses.replace(
            cfg, hyperparams=dataclasses.replace(
                cfg.hyperparams, n_steps=args.n_steps))
    if args.pretrain_n_steps is not None:
        cfg = dataclasses.replace(
            cfg, pretrain=dataclasses.replace(
                cfg.pretrain, n_steps=args.pretrain_n_steps))
    return cfg


def _apply_polarized_override(cfg, args):
    """Return a copy of ``cfg`` with spin-polarized correlation enabled when the
    ``--polarized`` flag is set.

    ``use_polarized_correlation=True`` rides into ``resolved_config.yaml`` and is
    read by the preflight/spec-builder and pretrain stages, which rebuild every
    architecture spin-polarization-aware. Unset leaves the config value (default
    False), so omitting the flag is byte-identical to today.
    """
    if getattr(args, "polarized", False):
        cfg = dataclasses.replace(cfg, use_polarized_correlation=True)
    return cfg


def _apply_defer_eval_override(cfg, args):
    """Return a copy of ``cfg`` with deferred-eval mode enabled when the
    ``--defer-eval`` flag is set.

    ``defer_eval=True`` rides into ``resolved_config.yaml`` and is read by
    ``submit_jobs``: the eval array is launched (afterany on train) only after
    the train array terminates, shrinking the per-run queued-job footprint.
    Unset leaves the config value (default False), so omitting the flag is
    byte-identical to today.
    """
    if getattr(args, "defer_eval", False):
        cfg = dataclasses.replace(cfg, defer_eval=True)
    return cfg


def _apply_inline_eval_override(cfg, args):
    """Return a copy of ``cfg`` with inline-eval mode enabled when the
    ``--inline-eval`` flag is set.

    ``inline_eval=True`` rides into ``resolved_config.yaml`` and is read by
    ``submit_jobs``: each train array task runs its own eval inline (same
    SLURM task) instead of submitting a separate eval array. Mutually
    exclusive with ``--defer-eval`` (inline eval is the OPPOSITE of a
    deferred eval).
    """
    if getattr(args, "inline_eval", False):
        if getattr(cfg, "defer_eval", False) or getattr(args, "defer_eval", False):
            raise ValueError(
                "submit: --inline-eval is mutually exclusive with "
                "--defer-eval / defer_eval=true. Inline-eval runs eval "
                "inside each train SLURM task; defer-eval submits eval "
                "as a SEPARATE deferred array. Pick one."
            )
        cfg = dataclasses.replace(cfg, inline_eval=True)
    return cfg


def _apply_time_overrides(cfg, args):
    """Return a copy of ``cfg`` with CLI-resolved per-stage wall times.

    ``--time`` is the base wall for every stage; the per-stage
    ``--{train,eval,preflight,pretrain}-time`` flags override it. A stage with
    neither its own flag nor a base keeps the config's value (so omitting all
    five is a no-op). Mirrors :func:`_apply_partition_overrides`: TRAIN uses
    ``cl.time`` directly; the others render ``cl.<stage>_time or cl.time``, so
    each resolved per-stage time is written explicitly to make the override a
    no-fallback. The resolved times ride into ``resolved_config.yaml``.
    """
    base = args.time
    train_t = args.train_time or base
    eval_t = args.eval_time or base
    preflight_t = args.preflight_time or base
    pretrain_t = args.pretrain_time or base
    changes = {}
    if train_t is not None:
        changes["time"] = train_t
    if eval_t is not None:
        changes["eval_time"] = eval_t
    if preflight_t is not None:
        changes["preflight_time"] = preflight_t
    if pretrain_t is not None:
        changes["pretrain_time"] = pretrain_t
    if not changes:
        return cfg
    cluster = dataclasses.replace(cfg.cluster, **changes)
    return dataclasses.replace(cfg, cluster=cluster)


def _apply_max_nodes_overrides(cfg, args):
    """Return a copy of ``cfg`` with CLI-resolved per-stage array throttles.

    With each array task booking a whole node ("exclusive"), the SLURM array
    throttle IS the number of nodes running concurrently. ``--max-nodes`` is the
    base cap for every array stage; ``--{train,eval,pretrain}-max-nodes``
    override it per stage. Any value left unset (no flag and no base) keeps the
    config's existing throttle, so omitting all four is a no-op. (Preflight is a
    single job, not an array, it has no throttle.) The resolved throttles ride
    into ``resolved_config.yaml`` so recovery commands reuse them.
    """
    base = args.max_nodes
    train = args.train_max_nodes if args.train_max_nodes is not None else base
    eval_ = args.eval_max_nodes if args.eval_max_nodes is not None else base
    pretrain = (args.pretrain_max_nodes
                if args.pretrain_max_nodes is not None else base)
    changes = {}
    if train is not None:
        changes["array_throttle"] = train
    if eval_ is not None:
        changes["eval_array_throttle"] = eval_
    if pretrain is not None:
        changes["pretrain_throttle"] = pretrain
    if not changes:
        return cfg
    cluster = dataclasses.replace(cfg.cluster, **changes)
    return dataclasses.replace(cfg, cluster=cluster)


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
    """``submit``: create a fresh run dir and (dry-run by default) submit.

    Loads + semantically validates the grid, creates a timestamped run dir,
    writes ``resolved_config.yaml`` + ``scripts/`` + ``logs/``, then calls
    ``submit_jobs`` (dry-run unless ``--submit``) which renders + submits the
    4-stage pretrain -> preflight -> train -> eval graph.
    """
    cfg = load_grid_config(args.grid)
    cfg = _apply_partition_overrides(cfg, args)
    cfg = _apply_max_nodes_overrides(cfg, args)
    cfg = _apply_time_overrides(cfg, args)
    cfg = _apply_step_overrides(cfg, args)
    cfg = _apply_polarized_override(cfg, args)
    cfg = _apply_defer_eval_override(cfg, args)
    cfg = _apply_inline_eval_override(cfg, args)
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
        if result.get("defer_eval"):
            eval_part = (f"eval_launcher={ids.get('eval_launcher')} "
                         "(eval array deferred until train terminates)")
        else:
            eval_part = f"eval={ids.get('eval')}"
        _log(f"submit: SUBMITTED ({result['n_specs']} specs, "
             f"{result['n_archs']} distinct arch(s)), "
             f"datagen={ids.get('datagen')} "
             f"pretrain={ids.get('pretrain')} "
             f"preflight={ids.get('preflight')} train={ids.get('train')} "
             f"{eval_part}")
        if result.get("manual_eval_command"):
            _log("submit: if the launcher cannot submit from a compute node, "
                 f"run after train finishes: {result['manual_eval_command']}")
    _log(f"submit: run dir = {run_dir}")
    return 0


def cmd_submit_eval(args) -> int:
    """``submit-eval``: submit the (deferred) eval array for a run.

    Reads the run's ``jobs.json`` + ``resolved_config.yaml`` and submits the
    eval array (``aftercorr`` on the train array), recording it. Idempotent: a
    no-op if an eval array is already recorded (unless ``--force``). This is what
    the deferred-eval launcher job runs, and the manual fallback to run from a
    login node if the launcher cannot submit from a compute node.
    """
    from xcquinox.alec.cluster._submit_eval import submit_deferred_eval
    try:
        result = submit_deferred_eval(args.run_dir, force=args.force)
    except RuntimeError as exc:
        _log(str(exc))
        return 1
    if result["submitted"]:
        _log(f"submit-eval: eval array {result['eval_id']} submitted "
             f"(aftercorr:{result['train_id']})")
    else:
        _log(f"submit-eval: no-op ({result['reason']}); eval array already "
             f"recorded as {result['eval_id']}")
    return 0


# ===========================================================================
# Subcommand: status
# ===========================================================================

def _pretrain_status(run_dir: str) -> str | None:
    """Lightweight pretrain-stage status line, or None if it cannot be checked.

    Pretrain is a small up-front stage, it gets no per-index
    ``reduce_outcomes``. The check is purely on-disk: for each distinct
    architecture in the resolved config, the pretrain worker writes
    ``xnet.eqx`` + ``cnet.eqx`` into the RUN-SCOPED
    ``<run_dir>/pretrain/<arch>/`` (see ``pretrain_checkpoint_dir``). We
    report how many of those checkpoint pairs are present. The path MUST be
    derived through the same helper the pretrain worker uses, or this check
    looks in the wrong directory and reports a false ``0/N``.
    """
    cfg_path = os.path.join(run_dir, _RESOLVED_CONFIG_FILENAME)
    if not os.path.exists(cfg_path):
        return None
    try:
        cfg = load_grid_config(cfg_path)
    except Exception:
        return None
    archs = sorted(set(cfg.sweep.arch))
    done = 0
    for arch in archs:
        d = pretrain_checkpoint_dir(run_dir, arch)
        if (os.path.exists(os.path.join(d, "xnet.eqx"))
                and os.path.exists(os.path.join(d, "cnet.eqx"))):
            done += 1
    return f"{done}/{len(archs)} architecture checkpoint pair(s) present"


def cmd_status(args) -> int:
    """``status``: read-only per-index outcome report (no lock taken).

    Aggregates ``train`` and ``eval`` outcomes via ``reduce_outcomes`` across
    all non-superseded ``jobs.json`` generations, diffs against the manifest's
    ``n_specs``, and prints counts + an actionable remedy line. The pretrain
    stage gets a lightweight on-disk checkpoint-presence check (no per-index
    reduction: pretrain is a handful of jobs). A
    :class:`SlurmTransientError` is reported, not crashed on.
    """
    run_dir = os.path.abspath(args.run_dir)
    manifest = _try_read_manifest(run_dir)
    if manifest is None:
        _log(f"status: {run_dir}/manifest.json is missing or corrupt.")
        _log("  remedy: run `repair-manifest <run_dir>` to rebuild it.")
        return 1

    n_specs = int(manifest["n_specs"])
    _log(f"status: run dir {run_dir}, manifest records {n_specs} spec(s).")

    pt_status = _pretrain_status(run_dir)
    if pt_status is not None:
        _log(f"  pretrain: {pt_status}")

    try:
        train = job_tracking.reduce_outcomes(run_dir, "train")
        ev = job_tracking.reduce_outcomes(run_dir, "eval")
    except job_tracking.SlurmTransientError:
        _log("status: SLURM controller unreachable, retry. "
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

    # diff vs manifest, covered indices should equal n_specs.
    if len(train) != n_specs:
        _log(f"  WARNING: reduce_outcomes covered {len(train)} train indices "
             f"but manifest records {n_specs}, manifest may be inconsistent.")

    # Remedy line.
    preflight_dead = (
        train_never == n_specs and train_success == 0 and train_failed == 0
    )
    if manifest is None or len(train) != n_specs:
        _log("  remedy: `repair-manifest <run_dir>`: manifest is "
             "corrupt/inconsistent.")
    elif preflight_dead:
        _log("  remedy: `resubmit-preflight <run_dir>`: the preflight job "
             "appears to have failed (no train task ran).")
    elif train_failed > 0:
        _log("  remedy: `resubmit <run_dir>`: re-run the failed train "
             "task(s).")
    else:
        _log("  remedy: none, no failed train tasks detected.")
    return 0


# ===========================================================================
# Subcommand: results
# ===========================================================================

def cmd_results(args) -> int:
    """``results``: aggregate per-spec eval metrics (read-only).

    Joins each finished ``eval_df.csv`` with its grid cell from
    ``manifest.json``, prints a per-spec table + a summary (MAE stats over the
    COMPLETE specs only, incomplete spec dirs are shown but excluded from the
    statistics), and optionally writes a CSV (``--csv``) and a MAE-vs-subset_size
    plot (``--plot``). Takes no lock; safe to re-run as results trickle in.
    """
    run_dir = os.path.abspath(args.run_dir)

    # --- per-spec per-molecule drill-down ----------------------------------
    if args.spec is not None:
        try:
            pm = analyze.load_per_molecule(run_dir, args.spec)
        except FileNotFoundError as exc:
            _log(f"results: {exc}")
            return 1
        if pm is None:
            _log(f"results: spec {args.spec} has no eval/per_molecule.json: "
                 "its eval has not completed (see `results <run_dir>` for its "
                 "status).")
            return 1
        _log(f"results: per-molecule AE for spec {args.spec} "
             "(worst |error| first):")
        _log(analyze.format_per_molecule_table(pm))
        return 0

    # --- cross-spec worst molecules ----------------------------------------
    if args.worst is not None:
        try:
            worst = analyze.worst_molecules(run_dir, args.worst)
        except FileNotFoundError as exc:
            _log(f"results: {exc}")
            return 1
        if not worst:
            _log("results: no per-molecule eval data found yet.")
            return 0
        _log(f"results: {len(worst)} worst molecule-instances by "
             "|AE_error_kcalmol|:")
        _log(analyze.format_worst_table(worst))
        return 0

    # --- default: grid-level table + summary -------------------------------
    try:
        rows = analyze.collect_results(run_dir)
    except FileNotFoundError as exc:
        _log(f"results: {exc}")
        _log("  the manifest is written by the preflight job, wait for it to "
             "finish (check `status <run_dir>`), then re-run.")
        return 1

    summary = analyze.summarize(rows)
    _log(analyze.format_table(rows, summary))

    if args.csv:
        analyze.write_csv(rows, args.csv)
        _log(f"results: wrote CSV -> {args.csv}")
    if args.plot:
        if summary["n_complete"] == 0:
            _log("results: --plot skipped, no completed evals to plot yet.")
        else:
            try:
                analyze.plot_mae_vs_subset(rows, args.plot)
                _log(f"results: wrote plot -> {args.plot}")
            except ImportError as exc:
                _log(f"results: --plot failed, {exc}")
                return 1
    return 0


# ===========================================================================
# Subcommand: resubmit
# ===========================================================================

def cmd_resubmit(args) -> int:
    """``resubmit``: recover FAILED TRAIN tasks (preflight already succeeded).

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
        _log(f"resubmit: {run_dir}/manifest.json is missing/corrupt, "
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
            _log("resubmit: SLURM controller unreachable, retry.")
            return 1

        failed = _failed_train_indices(run_dir, width, outcomes)
        if not failed:
            _log("resubmit: no failed train tasks, nothing to do.")
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

        # Group retryable indices by failure class so each class can be
        # resubmitted with its OWN resource overrides (oom -> bigger mem /
        # oom_retry partition; timeout -> longer wall / timeout_retry partition).
        # A single mixed array could not carry per-class resources.
        retry_by_class: dict[str, list[int]] = {}
        for idx in retry:
            retry_by_class.setdefault(classes[idx], []).append(idx)
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
                 f"({classes[idx]}): NOT retried; inspect failure.json.")
        if defaulted:
            _log(f"  indices {sorted(set(defaulted))} use DEFAULT "
                 "partition/resources: no dedicated retry knob configured.")

        if not retry:
            _log("resubmit: no retryable indices below the attempt cap.")
            return 0

        # Re-verify spec content hashes before reusing specs/.
        _verify_spec_hashes(run_dir, manifest, retry)

        train_script = os.path.join(run_dir, "scripts", "train_array.sbatch")
        eval_script = os.path.join(run_dir, "scripts", "eval_array.sbatch")

        if not args.submit:
            _log(f"resubmit: DRY-RUN, would archive stale artifacts for "
                 f"{sorted(retry)} then submit per failure class:")
            for cls in sorted(retry_by_class):
                idxs = sorted(retry_by_class[cls])
                ta = _sparse_array_spec(idxs, cl.array_throttle)
                ea = _sparse_array_spec(idxs, cl.eval_array_throttle)
                ov = " ".join(_retry_resource_flags(cls, cl))
                _log(f"  [{cls}] sbatch --parsable --array={ta} {ov} "
                     f"{train_script}".replace("  ", " "))
                _log(f"  [{cls}] sbatch --parsable --array={ea} "
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

        # Submit one sparse train+eval pair PER failure class, each with that
        # class's resource overrides. A class whose eval sbatch fails rolls back
        # only its own train array; classes already submitted are left intact
        # (they are independent and correctly recorded).
        for cls in sorted(retry_by_class):
            idxs = sorted(retry_by_class[cls])
            train_array = _sparse_array_spec(idxs, cl.array_throttle)
            eval_array = _sparse_array_spec(idxs, cl.eval_array_throttle)
            # aftercorr requires byte-identical index lists (throttle aside).
            assert train_array.split("%", 1)[0] == eval_array.split("%", 1)[0]
            overrides = _retry_resource_flags(cls, cl)

            # Train carries the class's retry resource overrides (sbatch CLI
            # flags override the script's #SBATCH directives). Eval does NOT,
            # eval is light and keeps its own (default) resources.
            train_cmd = [
                "sbatch", "--parsable", f"--array={train_array}",
                *overrides, train_script,
            ]
            proc = job_tracking._run_slurm(train_cmd)
            train_id = _parse_job_id(proc)

            eval_cmd = [
                "sbatch", "--parsable", f"--array={eval_array}",
                f"--dependency=aftercorr:{train_id}", eval_script,
            ]
            try:
                proc = job_tracking._run_slurm(eval_cmd)
                eval_id = _parse_job_id(proc)
            except Exception as exc:
                _log(f"resubmit: [{cls}] eval sbatch failed ({exc}); rolling "
                     f"back train array {train_id} via scancel.")
                try:
                    job_tracking._run_slurm(["scancel", str(train_id)])
                except Exception:
                    _log(f"resubmit: WARNING scancel of train {train_id} also "
                         "failed: that array may be orphaned; cancel it "
                         "manually.")
                _log(f"resubmit: [{cls}] not recorded in jobs.json.")
                return 1

            job_tracking.append_job_record(run_dir, "train", train_id, idxs)
            job_tracking.append_job_record(run_dir, "eval", eval_id, idxs)
            for idx in idxs:
                attempts[str(idx)] = int(attempts.get(str(idx), 0)) + 1
            _write_attempts(run_dir, attempts)
            _log(f"resubmit: [{cls}] SUBMITTED train={train_id} eval={eval_id} "
                 f"for indices {idxs}"
                 + (f" with overrides {overrides}" if overrides else ""))
        return 0
    finally:
        if lock_path is not None:
            release_lock(lock_path)


# ===========================================================================
# Subcommand: resubmit-preflight
# ===========================================================================

def cmd_resubmit_preflight(args) -> int:
    """``resubmit-preflight``: recover a FAILED/timed-out pretrain/preflight.

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
        _log(f"resubmit-preflight: {cfg_path} not found, cannot reconstruct "
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
                _log("resubmit-preflight: REFUSING, manifest.json records a "
                     f"complete {n_cells}-cell materialization; the preflight "
                     "succeeded. Use `resubmit` to recover failed train "
                     "tasks instead.")
                return 1
            # Refusal 2: grid changed -> a fresh run dir is required.
            if man_n != n_cells:
                _log(f"resubmit-preflight: REFUSING, manifest records "
                     f"n_specs={man_n} but the grid now expands to {n_cells} "
                     "cells. A changed grid must use a fresh run dir "
                     "(`submit`).")
                return 1

        # Refusal 3: any on-disk train evidence -> a train task already ran.
        # This is authoritative, an empty attempts.json is NOT sufficient.
        scan_width = (
            int(manifest["width"]) if manifest is not None
            else max(4, len(str(n_cells - 1)))
        )
        evidence = _scan_train_evidence(run_dir, n_cells, scan_width)
        if evidence:
            _log("resubmit-preflight: REFUSING, found train-task evidence "
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

        old_datagen = _newest_live("datagen")
        old_pretrain = _newest_live("pretrain")
        old_train = _newest_live("train")
        old_eval = _newest_live("eval")

        if not args.submit:
            _log("resubmit-preflight: DRY-RUN, would re-submit the full "
                 "datagen->pretrain->preflight->train->eval graph via "
                 "submit_jobs(force=True), then scancel + mark_superseded the "
                 "old datagen/pretrain/train/eval arrays.")
            if old_datagen:
                _log(f"  old datagen job {old_datagen['array_job_id']} "
                     f"(gen {old_datagen['generation']}) would be cancelled.")
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
        _log(f"resubmit-preflight: re-submitted graph, "
             f"datagen={new_ids.get('datagen')} "
             f"pretrain={new_ids.get('pretrain')} "
             f"preflight={new_ids.get('preflight')} "
             f"train={new_ids.get('train')} eval={new_ids.get('eval')}.")

        # All new sbatch calls succeeded (submit_jobs would have raised
        # otherwise). NOW, and only now: scancel old datagen/pretrain/train/eval
        # -> mark_superseded. A scancel failure aborts before mark_superseded
        # so a superseded generation always has a live successor.
        scancel_ok = True
        orphans = []
        for rec in (old_datagen, old_pretrain, old_train, old_eval):
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
            _log("resubmit-preflight: a scancel failed, SKIPPING "
                 f"mark_superseded. Orphaned old array id(s): {orphans}. "
                 "Cancel them manually; the old jobs.json records were left "
                 "un-superseded on purpose.")
            return 1

        # scancel succeeded for every old array, mark them superseded.
        for rec in (old_datagen, old_pretrain, old_train, old_eval):
            if rec is None:
                continue
            job_tracking.mark_superseded(run_dir, rec["kind"],
                                         rec["generation"])
        _log("resubmit-preflight: old datagen/pretrain/train/eval arrays "
             "cancelled and marked superseded.")
        return 0
    finally:
        if lock_path is not None:
            release_lock(lock_path)


# ===========================================================================
# Subcommand: repair-manifest
# ===========================================================================

def cmd_repair_manifest(args) -> int:
    """``repair-manifest``: rebuild a corrupt OR missing ``manifest.json``.

    Never reads the old manifest (absent and corrupt are identical). Rebuilds
    from ``resolved_config.yaml`` (the deterministic idx->GridCell map) plus the
    on-disk ``specs/*.spec`` files (re-hashed). Asserts the spec count equals N,
    cross-checks the pad width, and rewrites ONLY ``manifest.json`` atomically.
    """
    run_dir = os.path.abspath(args.run_dir)

    cfg_path = os.path.join(run_dir, _RESOLVED_CONFIG_FILENAME)
    if not os.path.exists(cfg_path):
        _log(f"repair-manifest: {cfg_path} not found, the resolved config is "
             "the only source of truth for the grid. It is unrecoverable; "
             "start a fresh run dir with `submit`.")
        return 1
    try:
        cfg = load_grid_config(cfg_path)
    except Exception as exc:
        # ANY failure to parse/build the resolved config (ValueError, a YAML
        # ParserError, an OSError, ...) means it is unrecoverable, the grid
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
            _log(f"repair-manifest: {specs_dir} does not exist, no specs to "
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
                 "set is incomplete, cannot rebuild a trustworthy manifest. "
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
        _log(f"repair-manifest: rebuilt {manifest_path}, {n} spec(s), "
             f"pad width {width}. model.eqx / jobs.json / attempts.json / "
             "checkpoints/ were left untouched.")
        return 0
    finally:
        if lock_path is not None:
            release_lock(lock_path)


# ===========================================================================
# Subcommand: pull (rsync a run dir from the cluster back to local)
# ===========================================================================

# Default connection knobs for the SeaWulf (Stony Brook) deployment. Override
# with the env vars below or the matching ``--host`` / ``--remote-root`` /
# ``--local-root`` / ``--category`` CLI flags. Documented in
# ``hpcjobs/SEAWULF_RUNBOOK.md`` Section 10.
#
# IMPORTANT: semantics: ``_PULL_DEFAULT_REMOTE_ROOT`` is the *base scratch
# directory* holding all xcquinox sweep runs, with experiment-series subdirs
# (``alpha_off/runs``, ``alpha_on/runs``, ``polarized/alpha_on``, ...) below
# it. Pull's ``--category`` flag selects which subdir to look in. Pre-v2 docs
# said this default ended in ``/runs``: anyone who set
# XCQUINOX_CLUSTER_REMOTE_ROOT explicitly should drop the ``/runs`` tail and
# pass ``--category runs`` if they want the old single-series behavior.
_PULL_DEFAULT_HOST = "login.seawulf.stonybrook.edu"
_PULL_DEFAULT_REMOTE_ROOT = "/gpfs/scratch/awills/xcquinox_runs"


def _pull_default_local_root() -> str:
    """Default destination for ``pull`` when neither flag nor env var is set."""
    return str(Path.home() / "Documents/Research/xcquinox-results/runs")


def _make_ssh_lines(host: str):
    """Factory for an SSH-wrapping ``ssh_runner`` matching
    :func:`sync.resolve_run_id` / :func:`sync.discover_runs`. Shared by
    ``cmd_pull`` and ``cmd_list_runs`` so the subprocess invocation does not
    diverge between them. Raises :class:`subprocess.CalledProcessError` on
    nonzero exit, the caller is expected to format its stderr via
    :func:`sync.format_ssh_stderr_tail` to strip the SBU banner.
    """
    def _runner(argv):
        completed = subprocess.run(
            ["ssh", host, *argv],
            check=True, capture_output=True, text=True,
        )
        return completed.stdout.splitlines()
    return _runner


def cmd_pull(args) -> int:
    """``pull``: rsync a sweep run dir from the cluster back to local.

    Resolves the run id (``"latest"`` -> newest ``run_<UTC>Z`` under
    ``<remote-root>/<category>`` via ``ssh ls -1tr``), creates
    ``<local-root>/<category>/<run_id>/`` locally (mirroring the remote
    layout to keep categories from colliding), then invokes ``rsync`` with
    the packaged filter for the chosen profile.

    Exit code is rsync's exit code; on success the local destination path is
    printed so the caller can pipe it to e.g. ``python -m xcquinox.alec.cluster
    results "$(...)"``.
    """
    host = args.host
    remote_root = args.remote_root.rstrip("/")
    local_root = args.local_root.rstrip("/")
    category = args.category.strip("/")

    # --specs 0,1,5 -> [0, 1, 5]; empty/blank entries are skipped; any
    # non-numeric entry aborts with a clear error before we hit rsync.
    spec_indices: list[int] = []
    if getattr(args, "specs", None):
        for tok in args.specs.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                spec_indices.append(int(tok))
            except ValueError:
                _log(f"pull: --specs entries must be integers, got {tok!r}")
                return 1

    ssh_runner = _make_ssh_lines(host)

    try:
        run_id = _sync.resolve_run_id(
            args.run_id, ssh_runner=ssh_runner,
            remote_root=remote_root, category=category,
        )
    except ValueError as exc:
        _log(f"pull: {exc}")
        return 1
    except subprocess.CalledProcessError as exc:
        _log(f"pull: ssh failed while resolving run_id "
             f"(rc={exc.returncode}): "
             f"{_sync.format_ssh_stderr_tail(exc.stderr or '')}")
        return 1

    # Mirror the category layout locally so two different categories with
    # the same stamp do not stomp on each other.
    local_dest = Path(local_root)
    if category:
        local_dest = local_dest / category
    local_dest = local_dest / run_id
    local_dest.mkdir(parents=True, exist_ok=True)

    argv = _sync.build_rsync_command(
        host=host, remote_root=remote_root, local_root=local_root,
        run_id=run_id, category=category,
        profile=args.profile, dry_run=args.dry_run,
        spec_indices=tuple(spec_indices),
    )
    _log(f"pull: running: {' '.join(argv)}")
    rc = subprocess.run(argv).returncode
    if rc == 0:
        if args.dry_run:
            _log(f"pull: dry-run complete (nothing transferred); "
                 f"local dest would be {local_dest}")
        else:
            _log(f"pull: synced -> {local_dest}")
    else:
        _log(f"pull: rsync exited rc={rc}")
    return rc


# ===========================================================================
# Subcommand: list-runs (discover what's under XCQUINOX_CLUSTER_REMOTE_ROOT)
# ===========================================================================

def cmd_list_runs(args) -> int:
    """``list-runs``: discover ``run_<UTC>Z`` dirs under ``--remote-root``.

    Read-only; issues a single ``find -prune`` over SSH and groups results by
    category (relative parent directory). Use this to figure out what to
    pass to ``pull --category`` when your sweep layout has experiment-series
    subdirs (``alpha_off/runs``, ``polarized/alpha_on``, ...).
    """
    host = args.host
    remote_root = args.remote_root.rstrip("/")
    ssh_runner = _make_ssh_lines(host)

    try:
        groups = _sync.discover_runs(
            ssh_runner=ssh_runner, remote_root=remote_root,
            max_depth=args.depth,
        )
    except subprocess.CalledProcessError as exc:
        _log(f"list-runs: ssh failed (rc={exc.returncode}): "
             f"{_sync.format_ssh_stderr_tail(exc.stderr or '')}")
        return 1

    _log(f"remote_root: {remote_root} (host={host})")
    _log("")
    if not groups:
        _log(f"(no run_<UTC>Z dirs found under {remote_root!r} "
             f"within depth {args.depth}; "
             f"try `--depth {args.depth + 2}` if your layout nests deeper)")
        return 0

    # Categories sorted lexicographically; the unnested ("") category first.
    for cat in sorted(groups.keys()):
        run_ids = groups[cat]
        label = (cat + "/") if cat else "(root)/"
        _log(f"{label}  ({len(run_ids)} run{'s' if len(run_ids) != 1 else ''})")
        for i, rid in enumerate(run_ids):
            tag = "   <- latest" if i == len(run_ids) - 1 else ""
            _log(f"  {rid}{tag}")
        _log("")
    _log(f"(searched to depth {args.depth} below {remote_root}; "
         f"pass `--depth N` to go deeper)")
    return 0


# ===========================================================================
# argparse wiring
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser with all the harness subcommands."""
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
        "submit", help="create a run dir and submit the 4-stage job graph")
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
    p_submit.add_argument(
        "--partition", required=True,
        help="SLURM partition for the whole 4-stage graph (REQUIRED, the "
             "config carries no partition default, so a submission never "
             "silently lands on a login-node-specific queue). The per-stage "
             "--{train,eval,preflight,pretrain}-partition flags override this "
             "base for an individual stage.")
    p_submit.add_argument(
        "--train-partition", default=None,
        help="override the partition for the TRAIN array (default: --partition)")
    p_submit.add_argument(
        "--eval-partition", default=None,
        help="override the partition for the EVAL array (default: --partition)")
    p_submit.add_argument(
        "--preflight-partition", default=None,
        help="override the partition for the PREFLIGHT job (default: --partition)")
    p_submit.add_argument(
        "--pretrain-partition", default=None,
        help="override the partition for the PRETRAIN array (default: --partition)")
    p_submit.add_argument(
        "--max-nodes", type=int, default=None,
        help="simultaneous-node cap for every array stage (sets the SLURM "
             "array throttle: with one whole node per task, this IS the number "
             "of nodes running at once). Unset -> the config's throttle values. "
             "Per-stage --{train,eval,pretrain}-max-nodes override this base.")
    p_submit.add_argument(
        "--train-max-nodes", type=int, default=None,
        help="override the simultaneous-node cap for the TRAIN array "
             "(default: --max-nodes, else config array_throttle)")
    p_submit.add_argument(
        "--eval-max-nodes", type=int, default=None,
        help="override the simultaneous-node cap for the EVAL array "
             "(default: --max-nodes, else config eval_array_throttle)")
    p_submit.add_argument(
        "--pretrain-max-nodes", type=int, default=None,
        help="override the simultaneous-node cap for the PRETRAIN array "
             "(default: --max-nodes, else config pretrain_throttle)")
    p_submit.add_argument(
        "--time", default=None,
        help="SLURM wall-clock limit (HH:MM:SS or D-HH:MM:SS) for every stage "
             "(base). Unset -> the config's per-stage times. Per-stage "
             "--{train,eval,preflight,pretrain}-time override this base.")
    p_submit.add_argument(
        "--train-time", default=None,
        help="override the wall for the TRAIN array (default: --time, else config)")
    p_submit.add_argument(
        "--eval-time", default=None,
        help="override the wall for the EVAL array (default: --time, else config)")
    p_submit.add_argument(
        "--preflight-time", default=None,
        help="override the wall for the PREFLIGHT job (default: --time, else config)")
    p_submit.add_argument(
        "--pretrain-time", default=None,
        help="override the wall for the PRETRAIN array (default: --time, else config)")
    p_submit.add_argument(
        "--n-steps", type=int, default=None,
        help="override the per-spec training-optimization step count "
             "(hyperparams.n_steps); unset -> the config value")
    p_submit.add_argument(
        "--pretrain-n-steps", type=int, default=None,
        help="override the pretraining step count (pretrain.n_steps); "
             "unset -> the config value")
    p_submit.add_argument(
        "--polarized", action="store_true",
        help="activate spin-polarized correlation on the networks for this run "
             "(use_polarized_correlation=True): the cnet becomes "
             "spin-polarization-aware and the UKS energy path uses the "
             "zeta-dependent PW92c baseline. Default off (unpolarized).")
    p_submit.add_argument(
        "--defer-eval", action="store_true", dest="defer_eval",
        help="deferred-eval mode (defer_eval=True): do NOT queue the eval array "
             "up front. A tiny launcher job (afterany on train) submits the eval "
             "array only after the train array terminates, shrinking the per-run "
             "queued-job footprint (helps under SLURM per-user submit caps). "
             "Default off (eval queued with the rest).")
    p_submit.add_argument(
        "--inline-eval", action="store_true", dest="inline_eval",
        help="inline-eval mode (inline_eval=True): each train array task runs "
             "its own eval immediately after training (in the same SLURM "
             "task), instead of submitting a separate eval array. Yields a "
             "3-stage graph (pretrain -> preflight -> train+eval inline) and "
             "eliminates the inter-stage queue gap. Mutually exclusive with "
             "--defer-eval. Default off.")
    p_submit.set_defaults(func=cmd_submit)

    p_submit_eval = sub.add_parser(
        "submit-eval",
        help="submit the (deferred) eval array for a run; manual fallback if the "
             "launcher job cannot sbatch from a compute node")
    p_submit_eval.add_argument("run_dir", help="the run directory")
    p_submit_eval.add_argument(
        "--force", action="store_true",
        help="submit even if an eval array is already recorded")
    p_submit_eval.set_defaults(func=cmd_submit_eval)

    p_status = sub.add_parser(
        "status", help="read-only per-index outcome report")
    p_status.add_argument("run_dir", help="the run directory")
    p_status.set_defaults(func=cmd_status)

    p_results = sub.add_parser(
        "results", help="aggregate per-spec eval metrics (MAE etc.)")
    p_results.add_argument("run_dir", help="the run directory")
    p_results.add_argument(
        "--csv", default=None,
        help="also write the joined per-spec rows to this CSV path")
    p_results.add_argument(
        "--plot", default=None,
        help="also write a MAE-vs-subset_size plot (PNG) to this path")
    p_results.add_argument(
        "--spec", type=int, default=None,
        help="instead of the grid table, show the per-molecule AE breakdown "
             "for this spec index (worst |error| first)")
    p_results.add_argument(
        "--worst", type=int, default=None, metavar="N",
        help="instead of the grid table, show the N worst molecule-instances "
             "by |AE_error| across all evaluated specs")
    p_results.set_defaults(func=cmd_results)

    p_pull = sub.add_parser(
        "pull",
        help="rsync a sweep run dir from the cluster back to local "
             "for post-processing (default profile: summaries, < 100 MB)")
    p_pull.add_argument(
        "run_id",
        help="run id to pull: a UTC stamp 'run_YYYYmmddTHHMMSSZ' or the "
             "literal 'latest' (resolved via `ssh <host> ls -1tr <remote-root>`)")
    p_pull.add_argument(
        "--profile", choices=list(_sync.VALID_PROFILES), default="summaries",
        help="which artifacts to pull. 'summaries' (default) skips every "
             "*.eqx and the logs/ tree (<100 MB / 40-spec run); 'full' mirrors "
             "the run dir minus logs/ (tens of GB / 40-spec run)")
    p_pull.add_argument(
        "--category",
        default=os.environ.get("XCQUINOX_CLUSTER_CATEGORY", ""),
        help="path segment under --remote-root that holds run_<UTC>Z dirs, "
             "e.g. 'alpha_off/runs', 'polarized/alpha_on'. Mirrors locally "
             "under --local-root so categories cannot collide. Empty (the "
             "default) looks directly under --remote-root. "
             "(default: $XCQUINOX_CLUSTER_CATEGORY else empty). "
             "Use `list-runs` to discover what's available.")
    p_pull.add_argument(
        "--host",
        default=os.environ.get("XCQUINOX_CLUSTER_HOST", _PULL_DEFAULT_HOST),
        help="SSH host (default: $XCQUINOX_CLUSTER_HOST else "
             f"'{_PULL_DEFAULT_HOST}'). Use a ~/.ssh/config alias for "
             "ControlMaster reuse.")
    p_pull.add_argument(
        "--remote-root",
        default=os.environ.get(
            "XCQUINOX_CLUSTER_REMOTE_ROOT", _PULL_DEFAULT_REMOTE_ROOT),
        help="base scratch directory on the cluster (default: "
             f"$XCQUINOX_CLUSTER_REMOTE_ROOT else '{_PULL_DEFAULT_REMOTE_ROOT}'). "
             "Run dirs live under <remote-root>/<category>/run_<UTC>Z.")
    p_pull.add_argument(
        "--local-root",
        default=os.environ.get(
            "XCQUINOX_CLUSTER_LOCAL_ROOT", _pull_default_local_root()),
        help="local directory under which '<run_id>/' is created "
             "(default: $XCQUINOX_CLUSTER_LOCAL_ROOT else "
             "~/Documents/Research/xcquinox-results/runs)")
    p_pull.add_argument(
        "--specs", default=None,
        help="comma-separated spec indices to restrict the pull to "
             "(e.g. '0,1,21'). When set, only checkpoints/spec_<NNNN>/ "
             "matching one of these indices are transferred; every other "
             "spec_* dir is excluded. Combine with --profile full to "
             "surgically pull just the model.eqx files for those specs "
             "(use for the local test-set re-evaluation workflow described "
             "in hpcjobs/SEAWULF_RUNBOOK.md §10.5).")
    p_pull.add_argument(
        "--dry-run", action="store_true",
        help="pass --dry-run to rsync: report what would transfer, copy nothing")
    p_pull.set_defaults(func=cmd_pull)

    p_list_runs = sub.add_parser(
        "list-runs",
        help="discover run_<UTC>Z dirs under --remote-root, grouped by "
             "category (read-only; one ssh `find -prune` call)")
    p_list_runs.add_argument(
        "--host",
        default=os.environ.get("XCQUINOX_CLUSTER_HOST", _PULL_DEFAULT_HOST),
        help="SSH host (default: $XCQUINOX_CLUSTER_HOST else "
             f"'{_PULL_DEFAULT_HOST}')")
    p_list_runs.add_argument(
        "--remote-root",
        default=os.environ.get(
            "XCQUINOX_CLUSTER_REMOTE_ROOT", _PULL_DEFAULT_REMOTE_ROOT),
        help=f"base scratch directory (default: $XCQUINOX_CLUSTER_REMOTE_ROOT "
             f"else '{_PULL_DEFAULT_REMOTE_ROOT}')")
    p_list_runs.add_argument(
        "--depth", type=int, default=5,
        help="maximum dir levels to descend below --remote-root (default 5; "
             "the deepest current layout is polarized/<axis>/runs/run_<UTC>Z "
             "at depth 4, bump if your layout nests further)")
    p_list_runs.set_defaults(func=cmd_list_runs)

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
