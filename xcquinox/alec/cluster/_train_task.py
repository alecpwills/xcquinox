"""xcquinox.alec.cluster._train_task: per-SLURM-array-task training wrapper.

The train-array sbatch template invokes this once per array task as::

    python -m xcquinox.alec.cluster._train_task <RUN_DIR> <SLURM_ARRAY_TASK_ID>

It is the thin harness layer between SLURM and the existing per-spec worker
``xcquinox.alec._train_one_spec``. Responsibilities:

  - Locate this task's spec file (``<run_dir>/specs/spec_<idx>.spec``, pad
    ``width`` read from ``manifest.json``) and its checkpoint directory
    (``<run_dir>/checkpoints/spec_<idx>/``).
  - Refuse the spec when the architecture its grid cell names carries no
    PASS pretraining-fidelity certificate
    (``<run_dir>/pretrain/<arch>/fidelity_certificate.json``): exit 3 with a
    ``fidelity_certificate_missing`` / ``fidelity_certificate_failed``
    ``failure.json`` instead of spending a node training against networks that
    were never shown to reproduce their parent functional. The record layer's
    status for the file (PASS / FAIL / MISSING / UNREADABLE) is written beside
    the classification, which has only the two values, so a certificate that
    states no verdict is distinguishable from one that was never written.
  - Run the worker as a subprocess (``_run_worker``: the single test seam),
    consuming its JSON progress stream and emitting a throttled human
    readable progress line to our stdout (which IS the SLURM ``.out`` log).
  - Install a SIGTERM handler so a SLURM wall-clock pre-kill grace signal
    (``--signal=B:TERM@<grace>``) is recorded as a ``killed_by_signal``
    failure before the process exits.
  - Classify the worker outcome into one of four cases and, on failure,
    write an atomic ``failure.json`` that ``job_tracking.reduce_outcomes``
    can read.

DO NOT echo raw worker lines wholesale to our stdout, the SLURM log would
balloon on long runs. Only the throttled progress line and a bounded tail
(on failure) are emitted.
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time


_MANIFEST_FILENAME = "manifest.json"

# Bounded tail of raw worker output retained for failure post-mortems. Kept
# small enough that a `failure.json` log_excerpt stays readable, large enough
# to capture a full Python traceback.
_TAIL_MAX_LINES = 200
_TAIL_MAX_CHARS = 16_384

# Progress throttle: emit a human-readable line at most this often. Whichever
# limit trips first wins. Tuned so a fast worker (sub-second steps) does not
# spam the SLURM log, and a slow worker still shows a heartbeat every ~2 min.
_THROTTLE_STEPS = 500
_THROTTLE_SECONDS = 120.0


# GPU-failure signatures recognized as a GPU-side OOM / runtime fault. Ported
# verbatim from notebooks/_build_step7_notebook.py (_GPU_OOM_MARKERS). Kept
# loose so XLA, CUDA-driver, and cuSolver/cuBLAS/cuDNN messages are all caught.
_GPU_OOM_MARKERS = (
    "RESOURCE_EXHAUSTED",
    "Out of memory",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "cuMemAlloc",
    "cuSolver internal error",
    "cuSolver",
    "gpusolverDnCreate",
    "cuBLAS",
    "cuDNN",
    "CUDA_ERROR",
    "INTERNAL: jaxlib/gpu",
)


# Host (CPU) out-of-memory signatures. A large-basis XLA/LLVM *compile* -- building
# the fused SCF-step kernel for e.g. 6-311++G(3df,2pd)+grid3 -- can exhaust node
# RAM; the C++ runtime then aborts on ``std::bad_alloc``, the allocator returns
# ENOMEM (``Cannot allocate memory``), or a thread stack cannot be allocated
# (``pthread_create failed``). These strings appear in the failure tail of every
# such crash. Matching them lets the harness classify the crash as ``"oom"`` so
# ``resubmit`` retries it on a larger-memory partition instead of dropping it as a
# permanent ``"deterministic"`` failure. See HISTORY (2026-07-04).
_CPU_OOM_MARKERS = (
    "std::bad_alloc",
    "bad_alloc",
    "Cannot allocate memory",
    "pthread_create failed",
)


def _looks_like_gpu_oom(text, rc=None):
    """True iff ``text`` / ``rc`` look like an OOM failure -- GPU-side, CPU-host,
    or an OS OOM-kill. (Name kept for its many call sites / pins; scope is now any
    OOM, not just GPU.)

    Three signals:
      - a GPU/CUDA/XLA marker (``_GPU_OOM_MARKERS``), or
      - a host-allocator marker (``_CPU_OOM_MARKERS``) -- a large-basis XLA/LLVM
        CPU compile that exhausts node RAM (``std::bad_alloc`` /
        ``Cannot allocate memory`` / ``pthread_create failed``), or
      - an OOM-ish signal exit: SIGKILL from the OS OOM-killer (``-9`` / ``137``)
        or SIGABRT from a C++ ``std::bad_alloc`` -> ``abort()`` (``-6`` / ``134``).
        A signal exit can carry no textual marker, so the exit code is a necessary
        backstop.

    Deliberately broad: a false positive costs at most one resubmit onto a bigger
    node, whereas a missed OOM is dropped as a permanent ``"deterministic"``
    failure and never retried (exactly the 6-311++G(3df,2pd)+grid3 regression this
    widening fixes).
    """
    if any(m in text for m in _GPU_OOM_MARKERS):
        return True
    if any(m in text for m in _CPU_OOM_MARKERS):
        return True
    if rc is not None and rc in (-9, 137, -6, 134):
        return True
    return False


# Known-benign non-zero exit codes seen after a checkpoint is already on
# disk: C-extension teardown crashes (glibc / JAX / PySCF). 139 == 128+11
# (SIGSEGV), -11 is the POSIX subprocess form of the same.
_BENIGN_TEARDOWN_CODES = frozenset({139, -11})


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _read_width(run_dir):
    """Read the zero-pad ``width`` from ``<run_dir>/manifest.json``."""
    path = os.path.join(run_dir, _MANIFEST_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"_train_task: no {_MANIFEST_FILENAME} in {run_dir}; the run "
            "directory has not been materialized"
        )
    with open(path) as f:
        manifest = json.load(f)
    return int(manifest["width"])


def _read_cell_arch(run_dir, idx):
    """The architecture name grid cell ``idx`` carries, from ``manifest.json``.

    ``None`` when the manifest records no cell for the index (a truncated or
    pre-``specs``-entry manifest). The caller treats an unresolvable
    architecture as an unverifiable certificate: a spec whose pretraining
    provenance cannot be established does not train.
    """
    path = os.path.join(run_dir, _MANIFEST_FILENAME)
    try:
        with open(path) as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return None
    for entry in manifest.get("specs") or ():
        try:
            if int(entry.get("index", -1)) == int(idx):
                return (entry.get("cell") or {}).get("arch")
        except (TypeError, ValueError):
            continue
    return None


def _spec_path(run_dir, idx, width):
    """Path to this task's spec file ``<run_dir>/specs/spec_<idx>.spec``."""
    return os.path.join(run_dir, "specs", f"spec_{idx:0{width}d}.spec")


def _checkpoint_dir(run_dir, idx, width):
    """Per-spec checkpoint dir ``<run_dir>/checkpoints/spec_<idx>/``."""
    return os.path.join(run_dir, "checkpoints", f"spec_{idx:0{width}d}")


def _model_path(run_dir, idx, width):
    return os.path.join(_checkpoint_dir(run_dir, idx, width), "model.eqx")


# ---------------------------------------------------------------------------
# failure.json: atomic write
# ---------------------------------------------------------------------------

def _write_failure_json(checkpoint_dir, payload):
    """Atomically write ``failure.json`` (mkstemp + os.replace) into the dir.

    Mirrors the schema ``job_tracking.reduce_outcomes`` reads:
    ``{"classification", "rc", "log_excerpt", ...}``. Best-effort: a write
    failure is swallowed (the SIGTERM path in particular must never raise).
    """
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".mktmp_", dir=checkpoint_dir)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp_name, os.path.join(checkpoint_dir, "failure.json"))
            tmp_name = None
        finally:
            if tmp_name is not None and os.path.exists(tmp_name):
                os.unlink(tmp_name)
    except OSError:
        # Best-effort only, never let a failure-record write crash the task.
        pass


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _read_failure_classification(checkpoint_dir):
    """Return the ``classification`` of an existing ``failure.json`` in
    ``checkpoint_dir``, or ``None`` if absent/unreadable.

    Used to honor a preflight ``precompute_failed_species`` marker without
    re-running the worker or overwriting the marker.
    """
    path = os.path.join(checkpoint_dir, "failure.json")
    try:
        with open(path) as f:
            return json.load(f).get("classification")
    except (OSError, ValueError):
        return None


def _log(idx, message):
    """Emit one harness log line (tagged) to our stdout, the SLURM log."""
    sys.stdout.write(f"[harness idx={idx}] {message}\n")
    sys.stdout.flush()


def _dump_tail(idx, tail):
    """Emit the bounded worker tail, framed, to our stdout."""
    sys.stdout.write(f"[harness idx={idx}] --- worker tail begin ---\n")
    if tail:
        for line in tail.splitlines():
            sys.stdout.write(f"[harness idx={idx}] {line}\n")
    sys.stdout.write(f"[harness idx={idx}] --- worker tail end ---\n")
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
# Worker subprocess seam
# ---------------------------------------------------------------------------

def _run_worker(spec_path, device):
    """Run ``_train_one_spec`` for one spec, the single test monkeypatch seam.

    Returns ``(rc, tail_text)`` where ``tail_text`` is a bounded tail (last
    ~200 lines / ~16 KB) of the child's merged stdout+stderr.

    Spawned with ``env=None`` (full inheritance): the sbatch thread-cap
    env vars (OMP/MKL/OPENBLAS_NUM_THREADS) MUST reach the worker. The
    ``--no-progress`` flag is intentionally NOT passed, we want the JSON
    progress stream so the throttled SLURM heartbeat below can be emitted.
    """
    cmd = [
        sys.executable, "-m", "xcquinox.alec._train_one_spec",
        spec_path, "--device", device,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        env=None,  # full env inheritance, sbatch thread caps must reach worker
    )
    # Expose the child so the SIGTERM handler can terminate it.
    global _ACTIVE_CHILD
    _ACTIVE_CHILD = proc

    tail = []           # bounded ring of recent raw lines
    last_emit_step = 0
    last_emit_time = time.monotonic()
    start_time = time.monotonic()
    seen_any_step = False

    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            # Retain the bounded tail (drop oldest beyond the line cap).
            tail.append(line)
            if len(tail) > _TAIL_MAX_LINES:
                del tail[0]

            # A non-JSON line (banner, traceback, warning) is kept in the
            # tail but contributes nothing to progress, a worker that emits
            # ZERO JSON progress lines (import-time crash) must not break us.
            if not line or not line.startswith("{"):
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("kind") != "step":
                continue

            seen_any_step = True
            step = int(msg.get("step", 0))
            total = int(msg.get("total", 0))
            loss = msg.get("loss", float("nan"))
            now = time.monotonic()
            # Throttle: emit on the very first step, then at most once per
            # _THROTTLE_STEPS steps OR _THROTTLE_SECONDS, and always the last.
            due = (
                last_emit_step == 0
                or (step - last_emit_step) >= _THROTTLE_STEPS
                or (now - last_emit_time) >= _THROTTLE_SECONDS
                or (total and step >= total)
            )
            if due:
                elapsed = now - start_time
                eta = None
                if step > 0 and total:
                    eta = elapsed / step * max(0, total - step)
                try:
                    loss_s = f"{float(loss):.4e}"
                except (TypeError, ValueError):
                    loss_s = str(loss)
                rss = msg.get("rss_gb")
                rss_s = (f", rss={rss:.1f}G"
                         if isinstance(rss, (int, float))
                         and math.isfinite(rss) else "")
                _PROGRESS_SINK(
                    f"step {step}/{total}, loss={loss_s}, "
                    f"elapsed={_fmt_secs(elapsed)}, ETA={_fmt_secs(eta)}{rss_s}"
                )
                last_emit_step = step
                last_emit_time = now
    finally:
        rc = proc.wait()
        _ACTIVE_CHILD = None

    # A bounded tail by line count above; also clamp total characters.
    tail_text = "\n".join(tail)
    if len(tail_text) > _TAIL_MAX_CHARS:
        tail_text = tail_text[-_TAIL_MAX_CHARS:]
    if not seen_any_step:
        # Not an error in itself, recorded so the caller's log shows why no
        # progress heartbeat appeared (e.g. import-time crash).
        pass
    return rc, tail_text


# Module-level handle to the running worker subprocess, so the SIGTERM
# handler (which receives only signum/frame) can terminate it. None when no
# child is running.
_ACTIVE_CHILD = None

# WS5-SIG-4: bounded grace the parent SIGTERM handler waits for the terminated
# worker to finish its best-effort resume flush before the parent exits. Kept
# well inside a typical SLURM kill-grace window so it cannot wedge the parent.
_SIGTERM_CHILD_WAIT_S = 30.0

# Progress-line sink. Indirected through a module global so _run_worker (the
# test seam) stays decoupled from the idx-tagged logger; main() points this at
# the real logger. Default is a no-op so a direct _run_worker call in a test
# need not wire it up.
_PROGRESS_SINK = lambda message: None  # noqa: E731


# ---------------------------------------------------------------------------
# SIGTERM handling
# ---------------------------------------------------------------------------

def _write_signal_failure(run_dir, idx, rc):
    """Record a ``killed_by_signal`` failure for this task, directly testable.

    Called from the SIGTERM handler. SLURM sends SIGTERM ``<grace>`` seconds
    before the wall-clock SIGKILL (the train-array template requests
    ``--signal=B:TERM@<grace>``), so we get a short window to record why the
    task is about to die.

    NOTE: a cgroup memory-OOM kill is an immediate SIGKILL with NO grace
    period: this handler does NOT fire in that case. OOM recovery for an
    ungraceful kill is handled by ``resubmit``'s ``sacct`` fallback
    (``State == OUT_OF_MEMORY``).
    """
    width = None
    try:
        width = _read_width(run_dir)
    except (OSError, ValueError, KeyError):
        width = None
    if width is None:
        # Best-effort: without the manifest we cannot pad the dir name; skip.
        return
    checkpoint_dir = _checkpoint_dir(run_dir, idx, width)
    _write_failure_json(checkpoint_dir, {
        "classification": "killed_by_signal",
        "rc": rc,
        "log_excerpt": (
            "task received SIGTERM (SLURM wall-clock pre-kill grace signal); "
            "training did not finish"
        ),
    })


def _install_sigterm_handler(run_dir, idx):
    """Install the SIGTERM handler for this task and return it.

    The handler terminates the worker child (best-effort), records a
    ``killed_by_signal`` failure via :func:`_write_signal_failure`, and exits
    non-zero. Returning the handler lets ``main`` (and tests) confirm it is the
    installed handler via ``signal.getsignal(SIGTERM)``.
    """
    def _handler(signum, frame):  # noqa: ARG001, signal-handler signature
        child = _ACTIVE_CHILD
        if child is not None and child.poll() is None:
            try:
                child.terminate()
            except OSError:
                pass
            # WS5-SIG-4: the worker installs its OWN SIGTERM handler that flushes
            # the in-flight epoch's resume checkpoint before exiting. We MUST give
            # that best-effort flush time to finish before this parent exits,
            # otherwise the parent's exit cuts the flush off. Bounded so a hung
            # child can never wedge the parent past its own grace window.
            try:
                child.wait(timeout=_SIGTERM_CHILD_WAIT_S)
            except subprocess.TimeoutExpired:
                # Flush overran the grace window; proceed to record + exit anyway
                # (periodic checkpoints are the primary net, the flush is a bonus).
                pass
            except OSError:
                pass
        _log(idx, "received SIGTERM, recording killed_by_signal and exiting")
        _write_signal_failure(run_dir, idx, rc=-15)
        # rc -15 mirrors the POSIX subprocess form of a SIGTERM-induced exit.
        sys.exit(143)  # 128 + 15

    signal.signal(signal.SIGTERM, _handler)
    return _handler


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="The materialized run directory.")
    parser.add_argument("idx", type=int, help="SLURM array task index.")
    parser.add_argument(
        "--device", choices=("gpu", "cpu", "auto"), default="auto",
        help="Device routing passed through to the worker.",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    idx = args.idx

    # Wire the progress sink to the idx-tagged logger before any worker runs.
    global _PROGRESS_SINK
    _PROGRESS_SINK = lambda message: _log(idx, message)  # noqa: E731

    # Install the SIGTERM handler early, a kill can arrive any time.
    _install_sigterm_handler(run_dir, idx)

    width = _read_width(run_dir)
    spec_path = _spec_path(run_dir, idx, width)
    checkpoint_dir = _checkpoint_dir(run_dir, idx, width)
    model_path = _model_path(run_dir, idx, width)

    # honor a preflight ``precompute_failed_species`` marker. The
    # preflight writes this when a FIXED subset references a species whose CCSD
    # external reference failed to precompute, so the spec cannot train. Running
    # the worker would burn an exclusive node AND overwrite the precise preflight
    # diagnosis with a generic "deterministic" one. Exit fast, preserving the
    # marker for ``reduce_outcomes`` (which reads ``failure.json`` off disk).
    if _read_failure_classification(checkpoint_dir) == "precompute_failed_species":
        _log(idx, "preflight marked this spec 'precompute_failed_species' "
                  "(its fixed subset references a species whose CCSD reference "
                  "failed to precompute); skipping the worker and preserving the "
                  "marker.")
        return 0

    # --- pretraining-fidelity gate -----------------------------------------
    # A spec may not train against networks that were never shown to reproduce
    # their parent functional: the pre-certificate checkpoints were off by 2.3
    # to 56 kcal/mol in atomization energies (SPEC_pretrain_fidelity_program.md
    # Section 2), larger than every effect the training is meant to measure.
    # Neither classification is in ``__main__._RETRYABLE``, so ``resubmit``
    # treats both as deterministic -- a blind retry cannot make an absent or
    # failed certificate pass. ``gate_certificate_from_read`` (not the record
    # layer's ``certificate_status``) is the predicate here: a run configured
    # with ``fidelity.enforce: false`` records the FAIL and is allowed through,
    # because the workflow-verification matrix must reach the train stage with
    # a short pretrain that cannot meet the tolerance. Such a run is still
    # refused by ``validate_run``, ``merge_v4_arms`` and the figure suite.
    #
    # Imported inside main deliberately, matching this module's body, which
    # carries only the standard library. It buys no process weight: running
    # this module as ``python -m`` imports the xcquinox package first, and
    # ``xcquinox/__init__`` pulls jax, equinox and pyscf in (measured: all
    # three are in sys.modules after importing this module, 0.67 s). What it
    # keeps is the import-order discipline the file is written to -- the
    # parent process orchestrates a worker SUBPROCESS and reaches for library
    # code only where it uses it.
    from xcquinox.alec.cluster.fidelity import (
        CERTIFICATE_FILENAME, VERDICT_FAIL, gate_certificate_from_read,
        read_certificate_status_in)
    from xcquinox.alec.cluster.grid_config import pretrain_checkpoint_dir
    arch = _read_cell_arch(run_dir, idx)
    if arch is None:
        excerpt = (
            f"manifest.json in {run_dir} records no cell architecture for "
            f"index {idx}, so this spec's {CERTIFICATE_FILENAME} cannot be "
            "located")
        _log(idx, f"REFUSING to train: {excerpt}")
        _write_failure_json(checkpoint_dir, {
            "classification": "fidelity_certificate_missing",
            "rc": 3,
            "arch": None,
            "log_excerpt": excerpt,
        })
        return 3
    # ONE parse feeds the release, the classification, the status the record
    # states and the excerpt it quotes. Gating on one read and classifying on
    # a second let a certificate rewritten between the two opens assemble a
    # record out of both documents -- measured, a FAIL read by the gate beside
    # a PASS read by the classifier wrote ``certificate_status: "PASS"`` under
    # ``classification: fidelity_certificate_missing`` with an excerpt naming
    # the FAIL, which no single document produces.
    status, reason, payload = read_certificate_status_in(
        pretrain_checkpoint_dir(run_dir, arch))
    allowed, message = gate_certificate_from_read(status, reason, payload)
    if not allowed:
        # The classification vocabulary has two values, so everything that is
        # not a literal FAIL joins the absent case: MISSING and UNREADABLE
        # alike leave no verdict to act on. The record layer's own word for
        # the state is carried into the log line and the failure record, so a
        # certificate that states nothing is not reported in the language of a
        # deleted one.
        classification = ("fidelity_certificate_failed"
                          if status == VERDICT_FAIL
                          else "fidelity_certificate_missing")
        _log(idx, f"REFUSING to train arch {arch!r} (certificate {status}): "
                  f"{message}")
        _write_failure_json(checkpoint_dir, {
            "classification": classification,
            "rc": 3,
            "arch": arch,
            "certificate_status": status,
            "log_excerpt": message,
        })
        return 3
    _log(idx, f"fidelity gate for arch {arch!r}: {message}")

    if not os.path.exists(spec_path):
        _log(idx, f"spec file not found: {spec_path}")
        _write_failure_json(checkpoint_dir, {
            "classification": "deterministic",
            "rc": 2,
            "log_excerpt": f"spec file not found: {spec_path}",
        })
        return 2

    _log(idx, f"starting training for spec {spec_path} (device={args.device})")
    t0 = time.time()
    rc, tail = _run_worker(spec_path, args.device)
    elapsed = time.time() - t0
    model_exists = os.path.isfile(model_path)

    # --- four-way outcome classification -----------------------------------
    if rc == 0 and model_exists:
        _log(idx, f"success, model.eqx written ({_fmt_secs(elapsed)} elapsed)")
        return 0

    if rc == 0 and not model_exists:
        # A worker that exits 0 but wrote no checkpoint is a real failure:
        # something silently skipped the model write.
        _log(idx, "worker reported success (rc=0) but wrote no model.eqx: "
                  "classifying as deterministic failure")
        _dump_tail(idx, tail)
        _write_failure_json(checkpoint_dir, {
            "classification": "deterministic",
            "rc": rc,
            "log_excerpt": tail,
        })
        return 1

    if rc != 0 and model_exists:
        # model.eqx existing means _train_one_spec's run_training ran to
        # completion (it writes the model once, at the very end). A non-zero
        # exit here is a post-training C-extension teardown anomaly, the
        # training result is fully on disk, so this is NOT a failure.
        if rc in _BENIGN_TEARDOWN_CODES:
            _log(idx, f"worker exited {rc} AFTER writing model.eqx: benign "
                      "C-extension teardown crash; treating as success")
        else:
            _log(idx, f"worker exited {rc} AFTER writing model.eqx: "
                      "checkpoint is complete; treating as success")
        return 0

    # rc != 0 AND no model.eqx -> a genuine failure. Classify it.
    if _looks_like_gpu_oom(tail, rc):
        classification = "oom"
    else:
        classification = "deterministic"
    _log(idx, f"worker failed (rc={rc}, no model.eqx), "
              f"classification={classification}")
    _dump_tail(idx, tail)
    _write_failure_json(checkpoint_dir, {
        "classification": classification,
        "rc": rc,
        "log_excerpt": tail,
    })
    return 1


if __name__ == "__main__":
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
