"""xcquinox.alec.cluster._pretrain -- per-architecture pretrain-array-task worker.

Pretraining is a harness STAGE: one ``run_pretrain`` job per distinct
architecture, submitted up front, feeding every downstream train task. The
pretrain-array sbatch template invokes this once per array task as::

    python -m xcquinox.alec.cluster._pretrain <RUN_DIR> <ARCH_IDX>

For one architecture index it:

  - Loads the resolved config (``<run_dir>/resolved_config.yaml``) via
    ``load_grid_config``.
  - Derives the list of distinct architectures from the ``arch`` sweep axis.
    The list MUST match the de-dup + sort that ``expand_grid`` applies to the
    arch axis (``_canon_axis`` -> ``sorted(set(...))``) so the i-th arch is
    byte-stable. ``<arch_idx>`` selects the i-th distinct arch; an out-of-range
    index fails fast.
  - Builds a :class:`~xcquinox.alec.config.PretrainSpec` for that architecture,
    threading every parameter from ``cfg.pretrain`` and writing the checkpoint
    into exactly ``<run_dir>/pretrain/<arch>/`` -- the directory each
    train spec's ``pretrain_checkpoint`` resolves to.
  - Calls :func:`run_pretrain` behind the :data:`_run_pretrain` seam, emitting a
    throttled ``[harness pretrain arch=...]`` heartbeat so a multi-hour job is
    never silent.
  - Verifies ``xnet.eqx`` + ``cnet.eqx`` landed under ``checkpoint_dir``. A
    worker that reports success but wrote no checkpoint exits non-zero (mirrors
    ``_train_task``'s silent-no-checkpoint guard) so SLURM ``afterok`` correctly
    blocks the train array.
  - Runs the per-architecture fidelity certificate
    (``cluster.fidelity.fidelity_certificate``, behind the
    :data:`_fidelity_certificate` seam) on the checkpoint it just wrote and
    exits non-zero unless the architecture reproduces its parent functional
    within ``cfg.fidelity``. The certificate is computed here, on this node,
    where the checkpoint is hot and the run identity is available, so the
    train array's ``afterok`` dependency blocks on an uncertified
    architecture without a further job kind. A second ``python -m`` line in
    ``templates/pretrain.sbatch.tmpl`` would pay the JAX and PySCF import a
    second time and would need failure semantics of its own to make that
    dependency block, and a separate SLURM job kind would add a dependency
    edge, a submission record and a log family for one function call; the
    template therefore carries exactly one invocation, which
    ``test_pretrain_template_invokes_only_the_certifying_worker`` pins. What
    the choice does cost is wall clock, which ``cluster.pretrain_time`` must
    cover: about forty systems (the pools' free atoms, the DFS pretraining
    molecules and H2O / N2 / CH4), each a reference SCF at the run's identity
    plus the network and parent XC evaluations -- 39 systems in 32 s at
    sto-3g / grid level 1, against a measured per-species precompute wall of
    0.6 s (Li) to 5.2 s (CH4) at 6-311++G(3df,2pd) / grid level 3 and several
    times that for the set's largest molecules. The verdict is then read back
    off the written certificate and put through
    ``fidelity.gate_certificate_from_read`` -- the release rule the train task
    and the preflight apply through ``gate_certificate`` -- so this job's exit
    code and their decisions come from one statement and one implementation of
    the rule. The summary line the job logs is formatted from that same read,
    so the numbers on record and the verdict acted on describe one document
    even if the file is rewritten while the job finishes. A run configured with
    ``fidelity.enforce: false`` AND a non-empty ``override_reason`` records
    the verdict and continues.

Exit code: 0 on success, non-zero on any failure.

JAX routing
-----------
Like :mod:`xcquinox.alec.cluster._train_task`'s ``--device auto`` default,
``JAX_ENABLE_X64=1`` is set before any ``import jax``; ``JAX_PLATFORMS`` is
left untouched so the sbatch-requested device (CPU or GPU) is honored.

This is a thin worker -- no resubmit / retry / outcome-classification
machinery. Pretrain is a handful of jobs; v1 recovery is re-running the graph.
"""
import argparse
import dataclasses
import os
import sys
import time

from xcquinox.alec.config import apply_model_block, get_architecture
from xcquinox.alec.cluster import fidelity
from xcquinox.alec.cluster.grid_config import (
    load_grid_config, _canon_axis, pretrain_checkpoint_dir,
)


# ---------------------------------------------------------------------------
# Progress throttle -- ported from _train_task._THROTTLE_*
# ---------------------------------------------------------------------------
# Emit a human-readable line at most this often; whichever limit trips first
# wins. Tuned so a fast pretrain does not spam the SLURM log and a slow one
# still shows a heartbeat roughly every 2 minutes.
_THROTTLE_STEPS = 100
_THROTTLE_SECONDS = 120.0


# ---------------------------------------------------------------------------
# JAX routing -- must run before ANY jax import
# ---------------------------------------------------------------------------

def _route_jax_env():
    """Pin JAX to fp64 via env var, before jax is imported.

    JAX defaults to float32 and equinox / optax may capture the default dtype
    before a post-import config update runs, so the env-var switch is the only
    reliable one. ``JAX_PLATFORMS`` is intentionally left untouched -- like
    ``_train_task``'s ``--device auto`` default, pretraining runs on whichever
    device the sbatch script requested.
    """
    os.environ["JAX_ENABLE_X64"] = "1"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(arch, message):
    """Emit one tagged harness log line to stdout -- the SLURM log."""
    sys.stdout.write(f"[harness pretrain arch={arch}] {message}\n")
    sys.stdout.flush()


def completed_pretraining(checkpoint_dir: str, cfg,
                          arch_name: str) -> tuple[bool, str]:
    """``(keep, reason)`` for pretraining already present in ``checkpoint_dir``.

    ``keep`` is True only when both networks are on disk, the certificate
    beside them is one an on-node gate releases, AND that certificate
    describes those networks at this run's identity.

    The release rule is :func:`fidelity.gate_certificate_from_read` applied to
    a single read -- PASS, or FAIL under a recorded waiver naming its reason
    -- which is exactly the rule this stage applies to its own verdict at the
    end of a fresh run and the rule the preflight sweep and the train task
    apply to the same file. A missing network, an absent or unreadable
    certificate, and an enforced FAIL are each a reason to pretrain again.

    The verdict alone is not enough, because it does not say WHICH networks
    were measured, at which basis and grid, against which parent, for which
    architecture, or by which code. Those five facts are recorded beside it
    and are re-checked by ``validate_run`` (the architecture the certificate
    names, its code version against the manifest's, parent, identity over the
    union of both key sets, and the two SHA-256 digests), so a certificate
    that disagrees on any of them costs the whole train and eval graph before
    the run is refused. Two routes reach that
    state through the recovery this check exists to serve: ``submit`` always
    creates a fresh run dir, so only ``resubmit-preflight`` re-runs pretrain
    into an existing one, and it reloads ``resolved_config.yaml`` precisely
    because the file can be edited between submissions -- none of its refusals
    covers an edited ``inputs.basis``, ``grid_level``, ``density_fit``,
    ``auxbasis`` or ``orientation_lock_strength``, none of which changes the
    cell count; and a redo interrupted between ``run_pretrain`` writing the
    two networks and the certificate being recomputed leaves new networks
    beside an older certificate with no edit at all. The comparison is
    :func:`fidelity.certificate_describes_run`, the one the record layer
    applies, and the reason names every fact that disagrees.

    A checkpoint with no certificate beside it is NOT kept even though the
    networks are usable: there is then no record of what was measured, and the
    gates downstream refuse it, so keeping it would leave the run blocked at
    the next stage with the expensive half already skipped. Requiring the
    certificate also settles the partial-write window -- ``run_pretrain``
    writes ``xnet.eqx`` and ``cnet.eqx`` before ``pretrain_metadata.json``,
    and the certificate is computed only after it returns, so a certificate on
    disk implies the metadata ``validate_run`` reads is there too.
    """
    missing = [name for name in ("xnet.eqx", "cnet.eqx")
               if not os.path.isfile(os.path.join(checkpoint_dir, name))]
    if missing:
        return False, (f"no complete checkpoint in {checkpoint_dir} "
                       f"(missing: {missing})")
    status, status_reason, on_disk = fidelity.read_certificate_status_in(
        checkpoint_dir)
    allowed, gate_reason = fidelity.gate_certificate_from_read(
        status, status_reason, on_disk)
    if not allowed:
        return False, gate_reason
    # Released, so the payload is the document the status was read from.
    mismatches = fidelity.certificate_describes_run(
        cfg, checkpoint_dir, arch_name, on_disk or {})
    if mismatches:
        return False, (
            f"{gate_reason}; but the certificate does not describe this "
            "run's pretraining: " + "; ".join(mismatches))
    return True, gate_reason


def _fmt_secs(seconds):
    """Compact h:mm:ss / m:ss formatting for elapsed/ETA."""
    if seconds is None or seconds != seconds:  # None or NaN
        return "?"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Distinct-arch list -- a pure function of the config
# ---------------------------------------------------------------------------

def _distinct_archs(cfg):
    """Return the de-duplicated, sorted list of arch names from the sweep.

    Reuses ``grid_config._canon_axis`` -- the EXACT de-dup + sort that
    ``expand_grid`` applies to the arch axis -- so the i-th distinct arch here
    is the same i-th arch the grid expansion sees. Keeping these in lock-step
    is what makes ``<arch_idx>`` a stable selector.
    """
    return _canon_axis(cfg.sweep.arch)


# ---------------------------------------------------------------------------
# Throttled progress callback
# ---------------------------------------------------------------------------

def _make_progress_callback(arch_name):
    """Build a throttled ``run_pretrain`` progress callback.

    ``run_pretrain`` invokes the callback with a dict carrying ``phase``
    (``"X"`` / ``"C"``), ``step``, ``total``, ``loss`` and ``timestamp``. The
    callback emits a ``[harness pretrain arch=...]`` heartbeat on the first
    step of each phase, then at most once per :data:`_THROTTLE_STEPS` steps or
    :data:`_THROTTLE_SECONDS`, and always on the last step -- so a multi-hour
    job is visibly alive without ballooning the SLURM log.
    """
    state = {"phase": None, "last_step": 0, "last_time": None, "t0": None}

    def _callback(info):
        phase = info.get("phase")
        step = int(info.get("step", 0))
        total = int(info.get("total", 0))
        loss = info.get("loss", float("nan"))
        now = time.monotonic()

        # A new phase resets the throttle bookkeeping.
        if phase != state["phase"]:
            state["phase"] = phase
            state["last_step"] = 0
            state["last_time"] = None
            state["t0"] = now

        due = (
            state["last_time"] is None
            or (step - state["last_step"]) >= _THROTTLE_STEPS
            or (now - state["last_time"]) >= _THROTTLE_SECONDS
            or (total and step >= total)
        )
        if not due:
            return

        elapsed = now - (state["t0"] or now)
        eta = None
        if step > 0 and total:
            eta = elapsed / step * max(0, total - step)
        try:
            loss_s = f"{float(loss):.4e}"
        except (TypeError, ValueError):
            loss_s = str(loss)
        _log(
            arch_name,
            f"phase={phase} step {step}/{total}, loss={loss_s}, "
            f"elapsed={_fmt_secs(elapsed)}, ETA={_fmt_secs(eta)}",
        )
        state["last_step"] = step
        state["last_time"] = now

    return _callback


# ---------------------------------------------------------------------------
# run_pretrain seam
# ---------------------------------------------------------------------------

def _run_pretrain(spec, progress_callback=None):
    """Run :func:`xcquinox.alec.pretrain.run_pretrain` -- the test seam.

    Isolated as a named module-level function so a unit test can monkeypatch it
    and avoid real pretraining / JAX compute. ``run_pretrain`` writes
    ``xnet.eqx`` + ``cnet.eqx`` + ``pretrain_metadata.json`` into
    ``spec.checkpoint_dir``.
    """
    from xcquinox.alec.pretrain import run_pretrain  # noqa: E402 -- after JAX routing
    return run_pretrain(spec, progress_callback=progress_callback)


# ---------------------------------------------------------------------------
# fidelity_certificate seam
# ---------------------------------------------------------------------------
# Bound at module level -- the same test-seam pattern as _run_pretrain above --
# so a unit test can monkeypatch it and avoid the real SCF sweep. The call
# writes <run_dir>/pretrain/<arch>/fidelity_certificate.json and returns its
# payload.
_fidelity_certificate = fidelity.fidelity_certificate


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """Pretrain-array-task entrypoint. Returns a process exit code (0 = ok)."""
    # Route JAX before any import that pulls it in. argparse / os / time are
    # jax-free; the xcquinox.alec.pretrain import in _run_pretrain transitively
    # imports jax, so this MUST run first.
    _route_jax_env()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="The materialized run directory.")
    parser.add_argument(
        "arch_idx", type=int,
        help="Index into the sorted list of distinct architectures.",
    )
    args = parser.parse_args(argv)

    run_dir = os.path.abspath(args.run_dir)
    arch_idx = args.arch_idx

    # --- load resolved config ----------------------------------------------
    cfg_path = os.path.join(run_dir, "resolved_config.yaml")
    if not os.path.isfile(cfg_path):
        # load_grid_config also accepts .json; fall back so a JSON-only run dir
        # still works (mirrors the test fixtures' yaml-or-json dispatch).
        json_path = os.path.join(run_dir, "resolved_config.json")
        if os.path.isfile(json_path):
            cfg_path = json_path
        else:
            sys.stdout.write(
                f"[harness pretrain] ERROR: resolved_config.yaml not found at "
                f"{cfg_path}\n"
            )
            sys.stdout.flush()
            return 1
    try:
        cfg = load_grid_config(cfg_path)
    except (ValueError, ImportError, OSError) as exc:
        sys.stdout.write(
            f"[harness pretrain] ERROR: failed to load resolved config: "
            f"{exc}\n"
        )
        sys.stdout.flush()
        return 1

    # --- select the architecture for this array index ----------------------
    archs = _distinct_archs(cfg)
    if not (0 <= arch_idx < len(archs)):
        sys.stdout.write(
            f"[harness pretrain] ERROR: arch_idx {arch_idx} is out of range; "
            f"the config has {len(archs)} distinct architecture(s) "
            f"(valid indices 0..{len(archs) - 1}): {archs}\n"
        )
        sys.stdout.flush()
        return 1
    arch_name = archs[arch_idx]
    _log(arch_name, f"pretrain task started (arch_idx={arch_idx}, run_dir={run_dir})")

    # --- build the PretrainSpec --------------------------------------------
    try:
        arch_config = get_architecture(arch_name)
    except KeyError:
        sys.stdout.write(
            f"[harness pretrain arch={arch_name}] ERROR: arch {arch_name!r} "
            f"is not a known architecture in xcquinox.alec.config.ARCHITECTURES\n"
        )
        sys.stdout.flush()
        return 1

    # Match the training-spec arch: when the run enables spin-polarized
    # correlation, pretrain the spin-polarization-aware cnet so the pretrained
    # checkpoint's shape matches what training/eval will load.
    if getattr(cfg, "use_polarized_correlation", False):
        arch_config = dataclasses.replace(
            arch_config, use_polarized_correlation=True)
    # The run's model block (the parent anchor, the descriptor coordinates),
    # applied as spec_builder applies it, so the pretrained networks are the
    # class the training specs load.
    model_block = getattr(cfg, "model", None)
    if model_block is not None:
        arch_config = apply_model_block(arch_config, model_block)

    pt = cfg.pretrain
    # Run-scoped (<run_dir>/pretrain/<arch>) so two runs pretraining the same
    # arch write to distinct dirs (run_dir is unique per submission) and every
    # artifact for the run stays in one folder.
    checkpoint_dir = pretrain_checkpoint_dir(run_dir, arch_name)

    # --- keep a pretraining that already completed -------------------------
    # A pretrain array task has no resubmit path of its own: ``cmd_resubmit``
    # reduces the train and eval kinds only, so a task killed AFTER writing its
    # networks and certificate -- a wall-kill, a node failure, or the teardown
    # abort measured on cluster job 2134455 -- leaves the graph stalled behind
    # a stage that had in fact finished, and the only recovery is
    # ``resubmit-preflight``, which re-submits the whole
    # pretrain -> preflight -> train -> eval graph. Repeating the pretraining
    # of an architecture whose result is already on disk is the expensive half
    # of that recovery, and it replaces the checkpoint the run's certificate
    # was measured on with a different one. Checked BEFORE any work, so the
    # recovery costs a directory listing, one certificate read and two
    # network digests per completed architecture (0.009 s for two 8 MB files).
    keep, keep_reason = completed_pretraining(checkpoint_dir, cfg, arch_name)
    if keep:
        _log(arch_name,
             f"pretrain KEPT: {checkpoint_dir} already holds xnet.eqx + "
             f"cnet.eqx and a certificate this node's gate releases, measured "
             f"on those networks at this run's identity ({keep_reason}); the "
             "completed pretraining stands and is not repeated")
        return 0
    _log(arch_name, f"pretraining from scratch: {keep_reason}")

    from xcquinox.alec.config import PretrainSpec
    spec = PretrainSpec(
        arch=arch_config,
        data_dir=pt.data_dir,
        checkpoint_dir=checkpoint_dir,
        n_steps=pt.n_steps,
        lr_start=pt.lr_start,
        lr_end=pt.lr_end,
        lr_decay_start=pt.lr_decay_start,
        grad_clip=pt.grad_clip,
        seed=pt.seed,
        loss_weighting=pt.loss_weighting,
        # --- pretraining protocol ------------------------------------------
        # Read through getattr with the pre-protocol default, so a
        # resolved_config.yaml written before these knobs existed still loads
        # here -- which is what the recovery and resubmit paths replay.
        parent_density=getattr(pt, "parent_density", "pbe"),
        energy_term_weight=getattr(pt, "energy_term_weight", 0.0),
        validation_fraction=getattr(pt, "validation_fraction", 0.0),
        validation_seed=getattr(pt, "validation_seed", 0),
        validate_every=getattr(pt, "validate_every", 50),
        patience=getattr(pt, "patience", 0),
        # Published-schedule tail and rho*w sampling (2026-09-03); the
        # defaults reproduce every earlier resolved snapshot exactly.
        lr_decay_end=getattr(pt, "lr_decay_end", 1.0),
        points_per_system=getattr(pt, "points_per_system", 800),
        sampling_seed=getattr(pt, "sampling_seed", 0),
    )
    # Every protocol knob the YAML can set is stated in the run record, so the
    # log says what the job trained with rather than what its defaults are.
    _log(
        arch_name,
        f"running run_pretrain: n_steps={pt.n_steps}, "
        f"loss_weighting={pt.loss_weighting!r}, "
        f"parent_density={getattr(pt, 'parent_density', 'pbe')!r}, "
        f"energy_term_weight={getattr(pt, 'energy_term_weight', 0.0)}, "
        f"validation_fraction={getattr(pt, 'validation_fraction', 0.0)}, "
        f"validation_seed={getattr(pt, 'validation_seed', 0)}, "
        f"validate_every={getattr(pt, 'validate_every', 50)}, "
        f"patience={getattr(pt, 'patience', 0)}, "
        f"dfs_set={getattr(pt, 'dfs_set', False)}, "
        f"pool_atoms={getattr(pt, 'pool_atoms', False)}, "
        f"exchange_footing={getattr(pt, 'exchange_footing', 'total')!r}, "
        f"mesh_fraction={getattr(pt, 'mesh_fraction', 0.3)}, "
        f"lr_decay_end={getattr(pt, 'lr_decay_end', 1.0)}, "
        f"points_per_system={getattr(pt, 'points_per_system', 800)}, "
        f"sampling_seed={getattr(pt, 'sampling_seed', 0)}, "
        f"checkpoint_dir={checkpoint_dir}",
    )

    # --- run pretraining behind the seam -----------------------------------
    t0 = time.time()
    try:
        _run_pretrain(spec, progress_callback=_make_progress_callback(arch_name))
    except Exception as exc:  # any failure must produce a non-zero exit
        elapsed = time.time() - t0
        _log(
            arch_name,
            f"run_pretrain FAILED after {_fmt_secs(elapsed)}: "
            f"{type(exc).__name__}: {exc}",
        )
        return 1
    elapsed = time.time() - t0

    # --- silent-no-checkpoint guard ----------------------------------------
    # run_pretrain writes xnet.eqx + cnet.eqx; if either is missing the worker
    # "succeeded" but produced no usable checkpoint -- a real failure.
    xnet_path = os.path.join(checkpoint_dir, "xnet.eqx")
    cnet_path = os.path.join(checkpoint_dir, "cnet.eqx")
    missing = [
        p for p in (xnet_path, cnet_path) if not os.path.isfile(p)
    ]
    if missing:
        _log(
            arch_name,
            "ERROR: pretrain reported success but wrote no checkpoint "
            f"(missing: {missing})",
        )
        return 1

    # --- pretraining-fidelity gate -----------------------------------------
    # The checkpoint is on disk; whether it is USABLE is a physics question.
    # The certificate answers it here, on this node, with the checkpoint hot
    # and the production identity available, and gates this job's exit code --
    # so the train array's afterok dependency blocks on an uncertified
    # architecture with no extra job kind and no extra scheduling round trip.
    t_cert = time.time()
    _log(arch_name,
         "running the per-architecture fidelity certificate against parent "
         f"{fidelity.resolve_parent(arch_name).upper()} ...")
    try:
        certificate = _fidelity_certificate(cfg, run_dir, arch_name)
    except Exception as exc:  # any failure must produce a non-zero exit
        _log(arch_name,
             f"fidelity certificate RAISED after "
             f"{_fmt_secs(time.time() - t_cert)}: {type(exc).__name__}: {exc}")
        return 1
    if not isinstance(certificate, dict):
        _log(arch_name,
             "fidelity certificate RETURNED "
             f"{type(certificate).__name__} rather than a certificate payload "
             f"after {_fmt_secs(time.time() - t_cert)}; there is no verdict to "
             "report")
        return 1
    # The verdict, the numbers reported and any waiver all come off the
    # certificate FILE, through the shared predicate, rather than from the
    # returned payload. That file is what the train task and the preflight
    # gate on, so this job's exit code is decided by exactly the statement
    # they will read, by one implementation of the release rule -- and the
    # summary line quotes the same document, so the log cannot state numbers
    # from one payload beside a verdict from another. The release rule is
    # applied to THAT read rather than to the path: a gate that opens the file
    # again lets a certificate rewritten between the two opens be reported as
    # one document and acted on as the next, which reads 'certificate FAILED
    # ... / gate: PASS / SUCCEEDED' for a run whose recorded verdict is FAIL.
    # Two consequences: a waiver needs here what the gate requires everywhere
    # -- a recorded enforced=false AND a non-empty prose override_reason,
    # which the config loader does not impose, since main never calls
    # validate_grid_semantics and a fidelity block may carry enforce=false
    # with no reason; and a payload that never reached disk stops the stage
    # here rather than leaving the later stages to refuse a run whose pretrain
    # job is recorded as successful.
    status, status_reason, on_disk = fidelity.read_certificate_status_in(
        pretrain_checkpoint_dir(run_dir, arch_name))
    summary = (on_disk or {}).get("summary")
    if not isinstance(summary, dict):
        summary = {}
    line = (f"max_atom={summary.get('max_atom_mHa')} mHa, "
            f"max_dAE={summary.get('max_dAE_kcalmol')} kcal/mol over "
            f"{summary.get('n_systems')} system(s) "
            f"({summary.get('n_atoms')} atom(s), "
            f"{summary.get('n_atomizations')} atomization(s)) in "
            f"{_fmt_secs(time.time() - t_cert)}")
    allowed, gate_message = fidelity.gate_certificate_from_read(
        status, status_reason, on_disk)
    label = {fidelity.VERDICT_PASS: "PASSED",
             fidelity.VERDICT_FAIL: "FAILED"}.get(status, status)
    _log(arch_name, f"fidelity certificate {label}: {line}")
    if status != fidelity.VERDICT_PASS:
        for reason in summary.get("failure_reasons") or ():
            _log(arch_name, f"  reason: {reason}")
    if not allowed:
        _log(arch_name, f"REFUSING to report success: {gate_message}")
        return 1
    if status != fidelity.VERDICT_PASS:
        _log(arch_name, f"fidelity gate: {gate_message}")

    _log(
        arch_name,
        f"pretrain SUCCEEDED ({_fmt_secs(elapsed)} elapsed) -- "
        f"xnet.eqx + cnet.eqx written to {checkpoint_dir}",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    # The stage's verdict is the status this process hands SLURM, and
    # JAX's atexit teardown can abort the interpreter AFTER main() has
    # returned it (cluster job 2134455: the pretrain worker logged
    # "pretrain SUCCEEDED" and then died in glibc's "corrupted size vs.
    # prev_size", rc -6, so the stage read as FAILED and the dependent
    # array never ran). run_and_exit flushes and leaves through os._exit,
    # so the status is the verdict. See xcquinox/alec/cluster/_exit.py.
    # Imported HERE rather than in the module body: several of these
    # modules pin what their import pulls in (``fidelity`` is held to a
    # whitelist of cheap readers so the on-node gates can read a
    # certificate without the training stack), and the helper is needed
    # only when the module is RUN.
    from xcquinox.alec.cluster._exit import run_and_exit
    run_and_exit(main)
