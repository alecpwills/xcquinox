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
    architecture without a further job kind. A run configured with
    ``fidelity.enforce: false`` records the verdict and continues.

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

from xcquinox.alec.config import get_architecture
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

    pt = cfg.pretrain
    # Run-scoped (<run_dir>/pretrain/<arch>) so two runs pretraining the same
    # arch write to distinct dirs (run_dir is unique per submission) and every
    # artifact for the run stays in one folder.
    checkpoint_dir = pretrain_checkpoint_dir(run_dir, arch_name)
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
    )
    _log(
        arch_name,
        f"running run_pretrain: n_steps={pt.n_steps}, "
        f"loss_weighting={pt.loss_weighting!r}, checkpoint_dir={checkpoint_dir}",
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
    summary = certificate.get("summary") or {}
    line = (f"max_atom={summary.get('max_atom_mHa')} mHa, "
            f"max_dAE={summary.get('max_dAE_kcalmol')} kcal/mol over "
            f"{summary.get('n_systems')} system(s) "
            f"({summary.get('n_atoms')} atom(s), "
            f"{summary.get('n_atomizations')} atomization(s)) in "
            f"{_fmt_secs(time.time() - t_cert)}")
    if certificate.get("verdict") != fidelity.VERDICT_PASS:
        _log(arch_name, f"fidelity certificate FAILED: {line}")
        for reason in summary.get("failure_reasons") or ():
            _log(arch_name, f"  reason: {reason}")
        # The enforcement flag is read back out of the certificate that was
        # just written, so the worker, the train task and the preflight all
        # decide from the same recorded statement rather than from three
        # readings of the config.
        if not certificate.get("enforced", True):
            _reason = (certificate.get("tolerances") or {}).get(
                "override_reason")
            _log(arch_name,
                 "fidelity enforcement is OFF for this run "
                 f"(fidelity.enforce=false, override_reason: {_reason!r}); "
                 "the verdict is on record and the stage continues. This run "
                 "cannot enter validate_run, merge_v4_arms or the figure "
                 "suite.")
        else:
            return 1
    else:
        _log(arch_name, f"fidelity certificate PASSED: {line}")

    _log(
        arch_name,
        f"pretrain SUCCEEDED ({_fmt_secs(elapsed)} elapsed) -- "
        f"xnet.eqx + cnet.eqx written to {checkpoint_dir}",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    sys.exit(main())
