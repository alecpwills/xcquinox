"""xcquinox.alec.cluster.submit: render sbatch scripts + submit the job graph.

The HPC harness submits a **4-stage SLURM job graph**:

    pretrain array  (--array=0-A-1%pretrain_throttle, A = distinct archs)
        |  --dependency=afterok:<pretrain>
        v
    preflight (single job)
        |  --dependency=afterok:<pretrain>:<preflight>  (train gated on BOTH)
        v
    train array  (--array=0-N-1%train_throttle)
        |  --dependency=aftercorr:<train_array>
        v
    eval  array  (--array=0-N-1%eval_throttle)

The train/eval array size ``N = len(expand_grid(cfg))`` is known at submit
time from the config alone, no preflight result is needed to size the arrays.
The pretrain array size ``A = len(_canon_axis(cfg.sweep.arch))`` is the distinct
architecture count (the same de-dup ``expand_grid`` applies to the arch axis).
``aftercorr`` requires the train and eval arrays to share an *identical index
range* (only the ``%throttle`` suffix may differ); :func:`submit_jobs` asserts
that. The pretrain array's range is independent (over archs) and is NOT part
of that assertion.

This module owns two things:

  - :func:`render_sbatch`: pure: fills a ``string.Template`` from ``cfg`` +
    ``run_dir``. The CPU-vs-GPU train template is picked from
    ``cfg.cluster.device``; eval is always CPU.
  - :func:`submit_jobs`: the orchestration. It DEFAULTS TO DRY-RUN: with
    ``submit=False`` it writes the rendered scripts and a ``submit_commands.txt``
    record but calls neither ``sbatch`` nor writes ``jobs.json``. Only
    ``submit=True`` actually submits, and it does best-effort ``scancel``
    rollback if any ``sbatch`` in the graph is rejected.

Every SLURM subprocess goes through ``job_tracking._run_slurm``: the single
seam tests monkeypatch.
"""
from datetime import datetime, timezone
from string import Template
import importlib.resources
import os

from xcquinox.alec.cluster.grid_config import expand_grid, _canon_axis
from xcquinox.alec.cluster import job_tracking


# Wall-clock grace window (seconds) between SLURM's pre-timeout SIGTERM and the
# hard SIGKILL, gives ``_train_task`` time to checkpoint/classify on timeout.
_SIGTERM_GRACE_S = 120

# Wall time for the deferred-eval launcher job, it only renders + sbatches the
# eval array and records it, a few seconds of work; 15 minutes is ample margin.
_EVAL_LAUNCHER_TIME = "00:15:00"

# kind -> template filename (the GPU train variant is resolved separately).
_TEMPLATE_FILES = {
    "datagen": "datagen.sbatch.tmpl",
    "pretrain": "pretrain.sbatch.tmpl",
    "preflight": "preflight.sbatch.tmpl",
    "train_cpu": "train_array_cpu.sbatch.tmpl",
    "train_gpu": "train_array_gpu.sbatch.tmpl",
    "eval": "eval_array.sbatch.tmpl",
    "eval_launcher": "eval_launcher.sbatch.tmpl",
    # 2026-05-29 inline-eval mode: train then eval in the SAME SLURM task.
    # No GPU variant today, the inline path is CPU-first; GPU inline support
    # can be added later if the cluster device is gpu.
    "train_eval_inline_cpu": "train_eval_inline_cpu.sbatch.tmpl",
    # Hold-out benchmark reference densities (CCSD+PBE, no OEP): one
    # standalone resumable job, submitted only when
    # cfg.inputs.benchmark_refs_dir is set; starts after train BEGINS.
    "benchmark_refs": "benchmark_refs.sbatch.tmpl",
}

_COMMANDS_FILENAME = "submit_commands.txt"


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def _load_template_text(filename: str) -> str:
    """Read a packaged ``.sbatch.tmpl`` via ``importlib.resources``.

    The ``templates/`` directory is data, not a package (it has no
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
    blank value leaves NO dangling ``#SBATCH`` token in the script, keeps the
    rendered file clean for shellcheck.
    """
    value = (value or "").strip()
    if not value:
        return ""
    return f"#SBATCH --{directive}={value}\n"


def _conda_activation_block(conda_profile: str, conda_env: str) -> str:
    """Render the conda-activation block as a single whole-line placeholder.

    A blank ``conda_profile`` must NEVER emit a bare ``source`` line, under the
    template's ``set -euo pipefail`` that is broken bash. When ``conda_profile``
    is set we ``source`` it before ``conda activate``; when it is empty we skip
    the ``source`` entirely and assume ``conda`` is already on ``PATH``.
    """
    conda_profile = (conda_profile or "").strip()
    conda_env = (conda_env or "").strip()
    activate = f"conda activate {conda_env}"
    if conda_profile:
        return f"source {conda_profile}\n{activate}"
    return activate


def _train_template_kind(cfg) -> str:
    """'train_gpu' if cfg requests a GPU device, else 'train_cpu'."""
    device = (cfg.cluster.device or "cpu").strip().lower()
    return "train_gpu" if device == "gpu" else "train_cpu"


def render_sbatch(kind: str, cfg, run_dir: str, array_max=None) -> str:
    """Render the sbatch script for ``kind`` ∈ {pretrain, preflight, train, eval}.

    Args:
        kind: ``"pretrain"``, ``"preflight"``, ``"train"`` or ``"eval"``. For
            ``"train"`` the CPU-vs-GPU template is chosen from
            ``cfg.cluster.device``.
        cfg: a :class:`~xcquinox.alec.cluster.grid_config.GridConfig`.
        run_dir: the run directory (absolute; ``logs/`` lives under it).
        array_max: the largest array index. Required for the array kinds
            (``pretrain``/``train``/``eval``), for ``pretrain`` it is
            ``A-1`` (A = distinct architecture count); for ``train``/``eval``
            it is ``N-1`` (N = grid cell count). Ignored for ``preflight``.

    Returns:
        The fully substituted sbatch script text.
    """
    cl = cfg.cluster
    run_dir = os.path.abspath(run_dir)

    if kind == "datagen":
        template_kind = "datagen"
    elif kind == "pretrain":
        template_kind = "pretrain"
    elif kind == "preflight":
        template_kind = "preflight"
    elif kind == "train":
        template_kind = _train_template_kind(cfg)
    elif kind == "eval":
        template_kind = "eval"
    elif kind == "eval_launcher":
        template_kind = "eval_launcher"
    elif kind == "train_eval_inline":
        # 2026-05-29: inline-eval mode. Today only the CPU variant exists.
        device = (cfg.cluster.device or "cpu").strip().lower()
        if device == "gpu":
            raise NotImplementedError(
                "render_sbatch: inline_eval is not yet implemented for "
                "device='gpu'. Use device='cpu' or omit --inline-eval."
            )
        template_kind = "train_eval_inline_cpu"
    elif kind == "benchmark_refs":
        template_kind = "benchmark_refs"
    else:
        raise ValueError(
            f"render_sbatch: kind must be 'pretrain', 'preflight', 'train', "
            f"'eval', 'eval_launcher', 'train_eval_inline' or "
            f"'benchmark_refs', got {kind!r}"
        )

    text = _load_template_text(_TEMPLATE_FILES[template_kind])

    # Per-kind partition / time / mem / cpus fall back to the train-array
    # cluster values when the per-stage knob is unset.
    if kind == "datagen":
        # Datagen is a single front-stage job; knobs fall back to pretrain's,
        # then to the train-array cluster values.
        partition = cl.datagen_partition or cl.pretrain_partition or cl.partition
        time = cl.datagen_time or cl.pretrain_time or cl.time
        mem = cl.datagen_mem or cl.pretrain_mem or cl.mem
        cpus = (cl.datagen_cpus_per_task or cl.pretrain_cpus_per_task
                or cl.cpus_per_task)
    elif kind == "pretrain":
        partition = cl.pretrain_partition or cl.partition
        time = cl.pretrain_time or cl.time
        mem = cl.pretrain_mem or cl.mem
        cpus = cl.pretrain_cpus_per_task or cl.cpus_per_task
    elif kind == "preflight":
        partition = cl.preflight_partition or cl.partition
        time = cl.preflight_time or cl.time
        mem = cl.mem
        cpus = cl.cpus_per_task
    elif kind == "eval":
        partition = cl.eval_partition or cl.partition
        time = cl.eval_time or cl.time
        mem = cl.mem
        cpus = cl.cpus_per_task
    elif kind == "eval_launcher":
        # A trivial submit-only job: share a node (never book one exclusively),
        # 1 cpu, fixed short wall. Runs on the eval partition by default.
        partition = cl.eval_partition or cl.partition
        time = _EVAL_LAUNCHER_TIME
        mem = ""
        cpus = 1
    elif kind == "benchmark_refs":
        # Single CPU job over the full hold-out pool; preflight-style
        # fallbacks (it is the same flavor of reference generation).
        partition = (cl.benchmark_refs_partition or cl.preflight_partition
                     or cl.partition)
        time = cl.benchmark_refs_time or cl.preflight_time or cl.time
        mem = cl.mem
        cpus = cl.cpus_per_task
    else:  # train
        partition = cl.partition
        time = cl.time
        mem = cl.mem
        cpus = cl.cpus_per_task

    # Per-stage node-allocation mode. "exclusive" books a whole node per array
    # task (--nodes=1 --exclusive, NO --mem, the task owns all the node's RAM,
    # which is what memory-heavy training needs); "shared" requests a cpu/mem
    # slice so several tasks co-tenant a node (--mem emitted only when set;
    # otherwise SLURM applies the partition default-mem-per-cpu).
    # The launcher has no per-stage allocation field, it always shares a node
    # (booking a whole node for a few seconds of `sbatch` would be wasteful and
    # itself counts against the per-user job budget the launcher exists to save).
    # The inline-eval kind aliases to the train allocation (it IS a train task
    # that runs eval at the end).
    if kind == "eval_launcher":
        allocation = "shared"
    elif kind == "train_eval_inline":
        allocation = getattr(cl, "train_allocation")
    else:
        allocation = getattr(cl, f"{kind}_allocation")
    if allocation == "exclusive":
        alloc_lines = "#SBATCH --nodes=1\n#SBATCH --exclusive\n"
        mem_line = ""
    else:  # "shared": validated in validate_grid_semantics
        alloc_lines = ""
        mem_line = _optional_sbatch_line("mem", mem)

    mapping = {
        "JOB_NAME": f"xcq_{kind}",
        "PARTITION": partition,
        "TIME": time,
        "ALLOC_LINES": alloc_lines,
        "MEM_LINE": mem_line,
        "CPUS_PER_TASK": cpus,
        "RUN_DIR": run_dir,
        "CONDA_ACTIVATION": _conda_activation_block(
            cl.conda_profile, cl.conda_env
        ),
        "MAIL_USER_LINE": _optional_sbatch_line("mail-user", cl.mail_user),
        "MAIL_TYPE_LINE": _optional_sbatch_line("mail-type", cl.mail_type),
        "ACCOUNT_LINE": _optional_sbatch_line("account", cl.account),
    }

    # Hold-out benchmark refs wiring. The eval-running kinds export
    # XCQUINOX_BENCH_REFS_DIR (the pools loader's env hook) so the held-out
    # eval picks up whatever reference npz files exist at eval time; empty
    # line when the feature is off.
    bench_dir = getattr(cfg.inputs, "benchmark_refs_dir", None)
    mapping["BENCH_REFS_ENV_LINE"] = (
        f"export XCQUINOX_BENCH_REFS_DIR={bench_dir}\n" if bench_dir else "")
    if kind == "benchmark_refs":
        if not bench_dir:
            raise ValueError(
                "render_sbatch: kind='benchmark_refs' requires "
                "cfg.inputs.benchmark_refs_dir to be set"
            )
        mapping["BENCH_REFS_DIR"] = bench_dir
        mapping["BASIS"] = cfg.inputs.basis
        mapping["GRID_LEVEL"] = int(cfg.inputs.grid_level)
        df_flags = ""
        if cfg.inputs.density_fit:
            df_flags = " --density-fit"
            if cfg.inputs.auxbasis:
                df_flags += f" --auxbasis {cfg.inputs.auxbasis}"
        mapping["BENCH_DF_FLAGS"] = df_flags

    if kind in ("pretrain", "train", "eval", "train_eval_inline"):
        if array_max is None:
            raise ValueError(
                f"render_sbatch: array_max is required for kind {kind!r}"
            )
        if kind == "train" or kind == "train_eval_inline":
            # Inline-eval array uses the same throttle as train (it IS a train
            # task that also runs eval at the end).
            throttle = cl.array_throttle
        elif kind == "eval":
            throttle = cl.eval_array_throttle
        else:  # pretrain
            # (E) pretrain_throttle None -> run every distinct arch
            # concurrently (the pretrain array is a handful of jobs).
            throttle = (
                cl.pretrain_throttle if cl.pretrain_throttle is not None
                else int(array_max) + 1
            )
        mapping["ARRAY_MAX"] = int(array_max)
        mapping["THROTTLE"] = int(throttle)

    if kind == "train" or kind == "train_eval_inline":
        mapping["SIGTERM_GRACE"] = _SIGTERM_GRACE_S
        if template_kind == "train_gpu":
            mapping["GPUS_PER_TASK"] = int(cl.gpus_per_task)

    # .substitute (not safe_substitute): a missing placeholder is a bug we
    # want to surface loudly, not silently leave a ``${TOKEN}`` in a script.
    return Template(text).substitute(mapping)


# ---------------------------------------------------------------------------
# submit_commands.txt: human-readable audit trail
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

    The ``%throttle`` suffix is stripped, ``aftercorr`` only cares that the
    train and eval arrays span the same indices, not that they run at the
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
                force: bool = False, defer_eval=None,
                inline_eval=None) -> dict:
    """Render the 4-stage sbatch graph and (optionally) submit it.

    Defaults to dry-run (``submit=False``): writes the rendered scripts and
    a ``submit_commands.txt`` audit record, but calls neither ``sbatch`` nor
    writes ``jobs.json``.

    Control flow:
      1. ``N = len(expand_grid(cfg))``; train/eval ``array_max = N-1``.
         ``A = len(_canon_axis(cfg.sweep.arch))``; pretrain ``array_max = A-1``.
      2. Ensure ``run_dir`` + its ``logs/`` and ``scripts/`` subdirs exist.
      3. Render pretrain, preflight, train (cpu or gpu) and eval scripts into
         ``<run_dir>/scripts/``.
      4. Assert the train and eval ``--array`` index ranges are identical
         (``aftercorr`` requires it). The pretrain array range is independent
         (over archs) and is NOT part of that assertion.
      5. Dry-run: write scripts + ``submit_commands.txt`` (``[dry-run]`` tag);
         return a descriptor dict; do NOT touch SLURM or ``jobs.json``.
      6. Real run (``submit=True``): a double-submit guard requires ``force``
         if ``jobs.json`` already has live records. Submit pretrain -> preflight
         (``afterok:pretrain``) -> train (``afterok:pretrain:preflight``: the
         train array is gated on BOTH) -> eval (``aftercorr:train``); record
         each via ``append_job_record``. If any ``sbatch`` is rejected
         mid-graph, ``scancel`` the ids already returned in THIS call, append
         no partial records, and re-raise.

    Deferred-eval mode (``cfg.defer_eval`` True, or ``defer_eval=True``): the
    eval array is NOT submitted up front. Instead a tiny launcher job is
    submitted ``afterany:train``; when train terminates the launcher submits
    the eval array (``aftercorr:train``, identical gating) and records it. This
    shrinks the per-run queued-job footprint (pretrain+preflight+train+launcher
    instead of +eval array). Only pretrain/preflight/train records are written
    here; the eval record is written later by the launcher (or by a manual
    ``submit-eval`` run). ``defer_eval=None`` (the default) reads ``cfg.defer_eval``.

    Returns:
        A dict describing what was (or would be) submitted: ``n_specs``,
        ``n_archs``, ``array_max``, ``pretrain_array_max``, ``device``, the
        script paths, the ``sbatch`` command lines, ``dry_run`` flag, and
        (real runs only) the job ids.
    """
    run_dir = os.path.abspath(run_dir)
    # defer_eval=None reads the config; an explicit bool overrides it.
    defer = getattr(cfg, "defer_eval", False) if defer_eval is None \
        else bool(defer_eval)
    # inline_eval=None reads the config; an explicit bool overrides it.
    inline = getattr(cfg, "inline_eval", False) if inline_eval is None \
        else bool(inline_eval)
    if defer and inline:
        raise ValueError(
            "submit_jobs: defer_eval and inline_eval are mutually exclusive "
            "(inline eval runs in the SAME SLURM task as train; defer eval "
            "submits a SEPARATE deferred eval array). Pick one."
        )
    cells = expand_grid(cfg)
    n_specs = len(cells)
    if n_specs == 0:
        raise ValueError(
            "submit_jobs: grid expands to 0 cells, nothing to submit"
        )
    array_max = n_specs - 1

    n_archs = len(_canon_axis(cfg.sweep.arch))
    if n_archs == 0:
        raise ValueError(
            "submit_jobs: arch sweep axis is empty, nothing to pretrain"
        )
    pretrain_array_max = n_archs - 1

    scripts_dir = os.path.join(run_dir, "scripts")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # --- render --------------------------------------------------------------
    datagen_text = render_sbatch("datagen", cfg, run_dir)
    pretrain_text = render_sbatch("pretrain", cfg, run_dir,
                                  array_max=pretrain_array_max)
    preflight_text = render_sbatch("preflight", cfg, run_dir)
    if inline:
        # Single combined train+eval array; no separate eval submission.
        train_text = render_sbatch("train_eval_inline", cfg, run_dir,
                                    array_max=array_max)
        eval_text = None  # No separate eval array in inline mode.
    else:
        train_text = render_sbatch("train", cfg, run_dir, array_max=array_max)
        eval_text = render_sbatch("eval", cfg, run_dir, array_max=array_max)

        # aftercorr requires identical index ranges (throttle may differ).
        # The pretrain array range is independent (over archs), NOT checked
        # here. Inline mode has no separate eval array, so no range check.
        train_range = _array_range(train_text)
        eval_range = _array_range(eval_text)
        if train_range != eval_range:
            raise AssertionError(
                f"submit_jobs: train array range {train_range!r} != eval array "
                f"range {eval_range!r}; --dependency=aftercorr requires identical "
                "index ranges"
            )

    datagen_path = os.path.join(scripts_dir, "datagen.sbatch")
    pretrain_path = os.path.join(scripts_dir, "pretrain.sbatch")
    preflight_path = os.path.join(scripts_dir, "preflight.sbatch")
    train_path = os.path.join(
        scripts_dir,
        "train_eval_inline.sbatch" if inline else "train_array.sbatch")
    eval_path = os.path.join(scripts_dir, "eval_array.sbatch")
    scripts_to_write = [
        (datagen_path, datagen_text),
        (pretrain_path, pretrain_text),
        (preflight_path, preflight_text),
        (train_path, train_text),
    ]
    if not inline:
        scripts_to_write.append((eval_path, eval_text))
    # In deferred mode the launcher job (submitted afterany:train) re-uses the
    # eval_array.sbatch above and is itself a tiny single-task script.
    launcher_path = os.path.join(scripts_dir, "eval_launcher.sbatch")
    if defer:
        launcher_text = render_sbatch("eval_launcher", cfg, run_dir)
        scripts_to_write.append((launcher_path, launcher_text))
    # Hold-out benchmark refs: one standalone resumable job, rendered only
    # when configured (inputs.benchmark_refs_dir).
    bench_refs_dir = getattr(cfg.inputs, "benchmark_refs_dir", None)
    bench_path = os.path.join(scripts_dir, "benchmark_refs.sbatch")
    if bench_refs_dir:
        bench_text = render_sbatch("benchmark_refs", cfg, run_dir)
        scripts_to_write.append((bench_path, bench_text))
    for path, text in scripts_to_write:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    device = (cfg.cluster.device or "cpu").strip().lower()
    result = {
        "run_dir": run_dir,
        "n_specs": n_specs,
        "n_archs": n_archs,
        "array_max": array_max,
        "pretrain_array_max": pretrain_array_max,
        "device": device,
        "scripts": {
            "datagen": datagen_path,
            "pretrain": pretrain_path,
            "preflight": preflight_path,
            "train": train_path,
        },
        "defer_eval": defer,
        "inline_eval": inline,
    }
    if not inline:
        result["scripts"]["eval"] = eval_path
    if defer:
        result["scripts"]["eval_launcher"] = launcher_path
    if bench_refs_dir:
        result["scripts"]["benchmark_refs"] = bench_path

    # Manual fallback: if the launcher can't submit (compute nodes barred from
    # sbatch), run this from a login node once the train array finishes.
    manual_eval_cmd = (
        f"python -m xcquinox.alec.cluster submit-eval {run_dir}"
    )
    if defer:
        result["manual_eval_command"] = manual_eval_cmd

    # --- dry-run -------------------------------------------------------------
    if not submit:
        cmds = [
            f"sbatch --parsable {datagen_path}",
            f"sbatch --parsable --dependency=afterok:<DATAGEN_ID> "
            f"{pretrain_path}",
            f"sbatch --parsable --dependency=afterok:<PRETRAIN_ID> "
            f"{preflight_path}",
            f"sbatch --parsable "
            f"--dependency=afterok:<PRETRAIN_ID>:<PREFLIGHT_ID> {train_path}",
        ]
        if inline:
            cmds.append(
                f"# inline-eval mode: each train array task runs "
                "_eval_one_spec at the end of its SLURM task. "
                "No separate eval array is submitted."
            )
        elif defer:
            cmds.append(
                f"sbatch --parsable --dependency=afterany:<TRAIN_ID> "
                f"{launcher_path}"
            )
            cmds.append(
                f"# (launcher then runs) {manual_eval_cmd}  "
                f"# submits: sbatch --dependency=aftercorr:<TRAIN_ID> {eval_path}"
            )
        else:
            cmds.append(
                f"sbatch --parsable --dependency=aftercorr:<TRAIN_ID> "
                f"{eval_path}"
            )
        if bench_refs_dir:
            # 'after' (not afterok): starts once the train array has BEGUN,
            # running in parallel with training.
            cmds.append(
                f"sbatch --parsable --dependency=after:<TRAIN_ID> "
                f"{bench_path}"
            )
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

    submitted_ids: list[str] = []  # ids returned in THIS call, for rollback.
    issued_cmds: list[str] = []
    try:
        # 1. datagen, FIRST stage, NO dependency. Generates the pretrain-data
        # file(s) every swept arch needs before pretrain (afterok:datagen) runs.
        datagen_cmd = ["sbatch", "--parsable", datagen_path]
        issued_cmds.append(" ".join(datagen_cmd))
        proc = job_tracking._run_slurm(datagen_cmd)
        datagen_id = proc.stdout.strip().split(";")[0].split()[0]
        submitted_ids.append(datagen_id)

        # 2. pretrain array (one task per distinct architecture), afterok on datagen
        pretrain_cmd = ["sbatch", "--parsable",
                        f"--dependency=afterok:{datagen_id}", pretrain_path]
        issued_cmds.append(" ".join(pretrain_cmd))
        proc = job_tracking._run_slurm(pretrain_cmd)
        pretrain_id = proc.stdout.strip().split(";")[0].split()[0]
        submitted_ids.append(pretrain_id)

        # 3. preflight, afterok on pretrain
        preflight_cmd = [
            "sbatch", "--parsable",
            f"--dependency=afterok:{pretrain_id}", preflight_path,
        ]
        issued_cmds.append(" ".join(preflight_cmd))
        proc = job_tracking._run_slurm(preflight_cmd)
        preflight_id = proc.stdout.strip().split(";")[0].split()[0]
        submitted_ids.append(preflight_id)

        # 4. train array, afterok on BOTH pretrain and preflight (the colon-
        # list is valid SLURM afterok syntax: every listed job must succeed).
        train_cmd = [
            "sbatch", "--parsable",
            f"--dependency=afterok:{pretrain_id}:{preflight_id}", train_path,
        ]
        issued_cmds.append(" ".join(train_cmd))
        proc = job_tracking._run_slurm(train_cmd)
        train_id = proc.stdout.strip().split(";")[0].split()[0]
        submitted_ids.append(train_id)

        # 5. eval, three modes:
        #    (a) inline: the train array task ALREADY ran eval as its final
        #        step (see train_eval_inline_*.sbatch.tmpl); no further sbatch.
        #    (b) defer: a tiny launcher (afterany:train) submits the eval
        #        array only after train terminates.
        #    (c) default: queue the eval array now (aftercorr:train).
        eval_id = None
        launcher_id = None
        if inline:
            pass  # no separate eval submission
        elif defer:
            launcher_cmd = [
                "sbatch", "--parsable",
                f"--dependency=afterany:{train_id}", launcher_path,
            ]
            issued_cmds.append(" ".join(launcher_cmd))
            proc = job_tracking._run_slurm(launcher_cmd)
            launcher_id = proc.stdout.strip().split(";")[0].split()[0]
            submitted_ids.append(launcher_id)
        else:
            eval_cmd = [
                "sbatch", "--parsable",
                f"--dependency=aftercorr:{train_id}", eval_path,
            ]
            issued_cmds.append(" ".join(eval_cmd))
            proc = job_tracking._run_slurm(eval_cmd)
            eval_id = proc.stdout.strip().split(";")[0].split()[0]
            submitted_ids.append(eval_id)

        # 6. hold-out benchmark refs: single standalone job, eligible to start
        # once the train array has BEGUN ('after', not afterok), independent of
        # the training-refs preflight. Same rollback umbrella as the rest of
        # the graph.
        bench_id = None
        if bench_refs_dir:
            bench_cmd = [
                "sbatch", "--parsable",
                f"--dependency=after:{train_id}", bench_path,
            ]
            issued_cmds.append(" ".join(bench_cmd))
            proc = job_tracking._run_slurm(bench_cmd)
            bench_id = proc.stdout.strip().split(";")[0].split()[0]
            submitted_ids.append(bench_id)
    except Exception as exc:
        # Best-effort rollback: cancel everything submitted in THIS call.
        rollback_failed: list[str] = []
        for jid in submitted_ids:
            try:
                job_tracking._run_slurm(["scancel", str(jid)])
            except Exception:
                # Rollback is best-effort: a failed scancel must not mask the
                # original submission error, but a surviving orphan must be
                # logged prominently so the operator can cancel it manually.
                rollback_failed.append(str(jid))
        if rollback_failed:
            print(
                "submit_jobs: WARNING, scancel FAILED for job id(s) "
                f"{rollback_failed}; these arrays may be ORPHANED and must be "
                "cancelled manually (scancel <id>).",
                flush=True,
            )
        # Surface sbatch's actual stderr/stdout when present, CalledProcessError's
        # str() is only "Command '[...]' returned non-zero exit status N" and hides
        # the real SLURM rejection reason (e.g. wall-time exceeds partition limit),
        # which _run_slurm captured via capture_output=True. Without this the
        # operator cannot tell WHY the submission was rejected.
        detail = str(exc)
        slurm_err = (getattr(exc, "stderr", "") or "").strip()
        slurm_out = (getattr(exc, "stdout", "") or "").strip()
        if slurm_err:
            detail += f"\n  sbatch stderr: {slurm_err}"
        if slurm_out:
            detail += f"\n  sbatch stdout: {slurm_out}"
        raise RuntimeError(
            f"submit_jobs: SLURM rejected a job mid-graph ({detail}); rolled back "
            f"{len(submitted_ids)} already-submitted job(s) via scancel. No "
            "records were written to jobs.json."
        ) from exc

    # All accepted, now (and only now) write the append-only records. In
    # deferred mode the eval record is NOT written here; the launcher (or a
    # manual `submit-eval`) writes it once the eval array is actually submitted.
    # In inline-eval mode there is NO separate eval array, the eval runs in
    # the train SLURM task, so no eval record exists to write.
    indices = list(range(n_specs))
    arch_indices = list(range(n_archs))
    job_tracking.append_job_record(run_dir, "datagen", datagen_id, [0])
    job_tracking.append_job_record(run_dir, "pretrain", pretrain_id,
                                   arch_indices)
    job_tracking.append_job_record(run_dir, "preflight", preflight_id, [0])
    job_tracking.append_job_record(run_dir, "train", train_id, indices)
    if not (defer or inline):
        job_tracking.append_job_record(run_dir, "eval", eval_id, indices)
    if bench_refs_dir:
        job_tracking.append_job_record(run_dir, "benchmark_refs", bench_id,
                                       [0])

    _append_commands(run_dir, "submit", issued_cmds)

    result["dry_run"] = False
    result["commands"] = issued_cmds
    job_ids = {
        "datagen": datagen_id,
        "pretrain": pretrain_id,
        "preflight": preflight_id,
        "train": train_id,
    }
    if defer:
        job_ids["eval_launcher"] = launcher_id
        print(
            "submit_jobs: deferred-eval mode, the eval array will be submitted "
            f"by launcher job {launcher_id} after the train array terminates. "
            "If the launcher cannot submit from a compute node, run this from a "
            f"login node once train finishes:\n    {manual_eval_cmd}",
            flush=True,
        )
    elif inline:
        # No separate eval array, each train task ran eval as its final step
        # via train_eval_inline_*.sbatch.tmpl. Don't record eval=None.
        print(
            "submit_jobs: inline-eval mode, each train array task runs its "
            "own eval at the end of the SLURM task. No separate eval array "
            f"submitted (train array {train_id} carries both stages).",
            flush=True,
        )
    else:
        job_ids["eval"] = eval_id
    if bench_refs_dir:
        job_ids["benchmark_refs"] = bench_id
    result["job_ids"] = job_ids
    return result
