"""xcquinox.alec.cluster.submit — render sbatch scripts + submit the job graph.

The HPC harness submits a **3-stage SLURM job graph**:

    preflight (single job)
        |  --dependency=afterok:<preflight>
        v
    train array  (--array=0-N-1%train_throttle)
        |  --dependency=aftercorr:<train_array>
        v
    eval  array  (--array=0-N-1%eval_throttle)

The array size ``N = len(expand_grid(cfg))`` is known at submit time from the
config alone — no preflight result is needed to size the arrays. ``aftercorr``
requires the train and eval arrays to share an *identical index range* (only
the ``%throttle`` suffix may differ); :func:`submit_jobs` asserts that.

This module owns two things:

  - :func:`render_sbatch` — pure: fills a ``string.Template`` from ``cfg`` +
    ``run_dir``. The CPU-vs-GPU train template is picked from
    ``cfg.cluster.device``; eval is always CPU.
  - :func:`submit_jobs` — the orchestration. **It DEFAULTS TO DRY-RUN**: with
    ``submit=False`` it writes the rendered scripts and a ``submit_commands.txt``
    record but calls neither ``sbatch`` nor writes ``jobs.json``. Only
    ``submit=True`` actually submits, and it does best-effort ``scancel``
    rollback if any ``sbatch`` in the graph is rejected.

Every SLURM subprocess goes through ``job_tracking._run_slurm`` — the single
seam tests monkeypatch.
"""
from datetime import datetime, timezone
from string import Template
import importlib.resources
import os

from xcquinox.alec.cluster.grid_config import expand_grid
from xcquinox.alec.cluster import job_tracking


# Wall-clock grace window (seconds) between SLURM's pre-timeout SIGTERM and the
# hard SIGKILL — gives ``_train_task`` time to checkpoint/classify on timeout.
_SIGTERM_GRACE_S = 120

# kind -> template filename (the GPU train variant is resolved separately).
_TEMPLATE_FILES = {
    "preflight": "preflight.sbatch.tmpl",
    "train_cpu": "train_array_cpu.sbatch.tmpl",
    "train_gpu": "train_array_gpu.sbatch.tmpl",
    "eval": "eval_array.sbatch.tmpl",
}

_COMMANDS_FILENAME = "submit_commands.txt"


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def _load_template_text(filename: str) -> str:
    """Read a packaged ``.sbatch.tmpl`` via ``importlib.resources``.

    The ``templates/`` directory is *data*, not a package (it has no
    ``__init__.py``); ``importlib.resources.files`` traverses it fine.
    """
    res = (
        importlib.resources.files("xcquinox.alec.cluster")
        / "templates" / filename
    )
    return res.read_text(encoding="utf-8")


def _optional_sbatch_line(directive: str, value: str) -> str:
    """Render an optional ``#SBATCH`` directive line, or '' if value is blank.

    The directive is emitted as a whole line (with a trailing newline) so a
    blank value leaves NO dangling ``#SBATCH`` token in the script — keeps the
    rendered file clean for shellcheck.
    """
    value = (value or "").strip()
    if not value:
        return ""
    return f"#SBATCH --{directive}={value}\n"


def _train_template_kind(cfg) -> str:
    """'train_gpu' if cfg requests a GPU device, else 'train_cpu'."""
    device = (cfg.cluster.device or "cpu").strip().lower()
    return "train_gpu" if device == "gpu" else "train_cpu"


def render_sbatch(kind: str, cfg, run_dir: str, array_max=None) -> str:
    """Render the sbatch script for ``kind`` ∈ {preflight, train, eval}.

    Args:
        kind: ``"preflight"``, ``"train"`` or ``"eval"``. For ``"train"`` the
            CPU-vs-GPU template is chosen from ``cfg.cluster.device``.
        cfg: a :class:`~xcquinox.alec.cluster.grid_config.GridConfig`.
        run_dir: the run directory (absolute; ``logs/`` lives under it).
        array_max: the largest array index (``N-1``). Required for the array
            kinds (``train``/``eval``); ignored for ``preflight``.

    Returns:
        The fully substituted sbatch script text.
    """
    cl = cfg.cluster
    run_dir = os.path.abspath(run_dir)

    if kind == "preflight":
        template_kind = "preflight"
    elif kind == "train":
        template_kind = _train_template_kind(cfg)
    elif kind == "eval":
        template_kind = "eval"
    else:
        raise ValueError(
            f"render_sbatch: kind must be 'preflight', 'train' or 'eval', "
            f"got {kind!r}"
        )

    text = _load_template_text(_TEMPLATE_FILES[template_kind])

    # Per-kind partition / time fall back to the main cluster values.
    if kind == "preflight":
        partition = cl.preflight_partition or cl.partition
        time = cl.preflight_time or cl.time
    elif kind == "eval":
        partition = cl.eval_partition or cl.partition
        time = cl.eval_time or cl.time
    else:  # train
        partition = cl.partition
        time = cl.time

    mapping = {
        "JOB_NAME": f"xcq_{kind}",
        "PARTITION": partition,
        "TIME": time,
        "MEM": cl.mem,
        "CPUS_PER_TASK": cl.cpus_per_task,
        "RUN_DIR": run_dir,
        "CONDA_PROFILE": cl.conda_profile,
        "CONDA_ENV": cl.conda_env,
        "MAIL_USER_LINE": _optional_sbatch_line("mail-user", cl.mail_user),
        "MAIL_TYPE_LINE": _optional_sbatch_line("mail-type", cl.mail_type),
        "ACCOUNT_LINE": _optional_sbatch_line("account", cl.account),
    }

    if kind in ("train", "eval"):
        if array_max is None:
            raise ValueError(
                f"render_sbatch: array_max is required for kind {kind!r}"
            )
        throttle = (
            cl.array_throttle if kind == "train" else cl.eval_array_throttle
        )
        mapping["ARRAY_MAX"] = int(array_max)
        mapping["THROTTLE"] = int(throttle)

    if kind == "train":
        mapping["SIGTERM_GRACE"] = _SIGTERM_GRACE_S
        if template_kind == "train_gpu":
            mapping["GPUS_PER_TASK"] = int(cl.gpus_per_task)

    # .substitute (not safe_substitute): a missing placeholder is a bug we
    # want to surface loudly, not silently leave a ``${TOKEN}`` in a script.
    return Template(text).substitute(mapping)


# ---------------------------------------------------------------------------
# submit_commands.txt — human-readable audit trail
# ---------------------------------------------------------------------------

def _append_commands(run_dir: str, tag: str, lines: list[str]) -> None:
    """Append ``sbatch`` command lines to ``<run_dir>/submit_commands.txt``.

    ``tag`` is ``"dry-run"`` or ``"submit"``; each block is timestamped so the
    file is a chronological audit trail of every (would-be) submission.
    """
    path = os.path.join(run_dir, _COMMANDS_FILENAME)
    stamp = datetime.now(timezone.utc).isoformat()
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"# [{tag}] {stamp}\n")
        for line in lines:
            f.write(line + "\n")
        f.write("\n")


# ---------------------------------------------------------------------------
# Array-range guard
# ---------------------------------------------------------------------------

def _array_range(script_text: str) -> str:
    """Extract the ``0-MAX`` index range from a script's ``#SBATCH --array``.

    The ``%throttle`` suffix is stripped — ``aftercorr`` only cares that the
    train and eval arrays span the *same indices*, not that they run at the
    same concurrency.
    """
    for line in script_text.splitlines():
        line = line.strip()
        if line.startswith("#SBATCH --array="):
            spec = line.split("=", 1)[1]
            return spec.split("%", 1)[0]
    raise ValueError("script has no '#SBATCH --array=' directive")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _has_live_jobs(run_dir: str) -> bool:
    """True iff ``<run_dir>/jobs.json`` already records non-superseded jobs."""
    try:
        records = job_tracking.read_job_records(run_dir)
    except (FileNotFoundError, ValueError):
        return False
    return any(not r.get("superseded", False) for r in records)


def submit_jobs(cfg, run_dir: str, *, submit: bool = False,
                force: bool = False) -> dict:
    """Render the 3-stage sbatch graph and (optionally) submit it.

    **Defaults to dry-run** (``submit=False``): writes the rendered scripts and
    a ``submit_commands.txt`` audit record, but calls neither ``sbatch`` nor
    writes ``jobs.json``.

    Control flow:
      1. ``N = len(expand_grid(cfg))``; ``array_max = N-1``.
      2. Ensure ``run_dir`` + its ``logs/`` and ``scripts/`` subdirs exist.
      3. Render preflight, train (cpu or gpu) and eval scripts into
         ``<run_dir>/scripts/``.
      4. Assert the train and eval ``--array`` index ranges are identical
         (``aftercorr`` requires it).
      5. Dry-run: write scripts + ``submit_commands.txt`` (``[dry-run]`` tag);
         return a descriptor dict; do NOT touch SLURM or ``jobs.json``.
      6. Real run (``submit=True``): a double-submit guard requires ``force``
         if ``jobs.json`` already has live records. Submit preflight → train
         (``afterok``) → eval (``aftercorr``); record each via
         ``append_job_record``. If any ``sbatch`` is rejected mid-graph,
         ``scancel`` the ids already returned in THIS call, append no partial
         records, and re-raise.

    Returns:
        A dict describing what was (or would be) submitted: ``n_specs``,
        ``array_max``, ``device``, the script paths, the ``sbatch`` command
        lines, ``dry_run`` flag, and (real runs only) the job ids.
    """
    run_dir = os.path.abspath(run_dir)
    cells = expand_grid(cfg)
    n_specs = len(cells)
    if n_specs == 0:
        raise ValueError(
            "submit_jobs: grid expands to 0 cells — nothing to submit"
        )
    array_max = n_specs - 1

    scripts_dir = os.path.join(run_dir, "scripts")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # --- render --------------------------------------------------------------
    preflight_text = render_sbatch("preflight", cfg, run_dir)
    train_text = render_sbatch("train", cfg, run_dir, array_max=array_max)
    eval_text = render_sbatch("eval", cfg, run_dir, array_max=array_max)

    # aftercorr requires identical index ranges (throttle may differ).
    train_range = _array_range(train_text)
    eval_range = _array_range(eval_text)
    if train_range != eval_range:
        raise AssertionError(
            f"submit_jobs: train array range {train_range!r} != eval array "
            f"range {eval_range!r}; --dependency=aftercorr requires identical "
            "index ranges"
        )

    preflight_path = os.path.join(scripts_dir, "preflight.sbatch")
    train_path = os.path.join(scripts_dir, "train_array.sbatch")
    eval_path = os.path.join(scripts_dir, "eval_array.sbatch")
    for path, text in (
        (preflight_path, preflight_text),
        (train_path, train_text),
        (eval_path, eval_text),
    ):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    device = (cfg.cluster.device or "cpu").strip().lower()
    result = {
        "run_dir": run_dir,
        "n_specs": n_specs,
        "array_max": array_max,
        "device": device,
        "scripts": {
            "preflight": preflight_path,
            "train": train_path,
            "eval": eval_path,
        },
    }

    # --- dry-run -------------------------------------------------------------
    if not submit:
        cmds = [
            f"sbatch --parsable {preflight_path}",
            f"sbatch --parsable --dependency=afterok:<PREFLIGHT_ID> {train_path}",
            f"sbatch --parsable --dependency=aftercorr:<TRAIN_ID> {eval_path}",
        ]
        _append_commands(run_dir, "dry-run", cmds)
        result["dry_run"] = True
        result["commands"] = cmds
        return result

    # --- real submission -----------------------------------------------------
    # Double-submit guard: refuse to re-submit over live jobs without --force.
    if _has_live_jobs(run_dir) and not force:
        raise RuntimeError(
            f"submit_jobs: {run_dir}/jobs.json already records live (non-"
            "superseded) jobs; refusing to submit again. Pass force=True to "
            "override (e.g. after manually cancelling the prior graph)."
        )

    submitted_ids: list[str] = []  # ids returned in THIS call — for rollback.
    issued_cmds: list[str] = []
    try:
        # 1. preflight
        preflight_cmd = ["sbatch", "--parsable", preflight_path]
        issued_cmds.append(" ".join(preflight_cmd))
        proc = job_tracking._run_slurm(preflight_cmd)
        preflight_id = proc.stdout.strip().split(";")[0].split()[0]
        submitted_ids.append(preflight_id)

        # 2. train array — afterok on preflight
        train_cmd = [
            "sbatch", "--parsable",
            f"--dependency=afterok:{preflight_id}", train_path,
        ]
        issued_cmds.append(" ".join(train_cmd))
        proc = job_tracking._run_slurm(train_cmd)
        train_id = proc.stdout.strip().split(";")[0].split()[0]
        submitted_ids.append(train_id)

        # 3. eval array — aftercorr on the train array
        eval_cmd = [
            "sbatch", "--parsable",
            f"--dependency=aftercorr:{train_id}", eval_path,
        ]
        issued_cmds.append(" ".join(eval_cmd))
        proc = job_tracking._run_slurm(eval_cmd)
        eval_id = proc.stdout.strip().split(";")[0].split()[0]
        submitted_ids.append(eval_id)
    except Exception as exc:
        # Best-effort rollback: cancel everything submitted in THIS call.
        for jid in submitted_ids:
            try:
                job_tracking._run_slurm(["scancel", str(jid)])
            except Exception:
                # Rollback is best-effort: a failed scancel must not mask the
                # original submission error.
                pass
        raise RuntimeError(
            f"submit_jobs: SLURM rejected a job mid-graph ({exc}); rolled back "
            f"{len(submitted_ids)} already-submitted job(s) via scancel. No "
            "records were written to jobs.json."
        ) from exc

    # All three accepted — now (and only now) write the append-only records.
    indices = list(range(n_specs))
    job_tracking.append_job_record(run_dir, "preflight", preflight_id, [0])
    job_tracking.append_job_record(run_dir, "train", train_id, indices)
    job_tracking.append_job_record(run_dir, "eval", eval_id, indices)

    _append_commands(run_dir, "submit", issued_cmds)

    result["dry_run"] = False
    result["commands"] = issued_cmds
    result["job_ids"] = {
        "preflight": preflight_id,
        "train": train_id,
        "eval": eval_id,
    }
    return result
