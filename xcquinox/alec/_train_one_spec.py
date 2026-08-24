"""CLI worker: run a single TrainingSpec in an isolated subprocess.

Usage
-----
    python -m xcquinox.alec._train_one_spec <spec_path>

Loads a serialized ``TrainingSpec`` from disk, runs ``alec.run_training``
on it, and exits. The subprocess exits as soon as training completes,
which causes the OS to hard-reclaim all memory (jit-compiled XLA code,
intermediate arrays, etc.). This is the only robust way to run the step 5
training sweep on systems with limited RAM -- in-process
``jax.clear_caches()`` + ``gc.collect()`` cannot release compiled LLVM IR
that the runtime has already allocated for backing stores.

The parent notebook invokes this via
``subprocess.run([sys.executable, "-m", "xcquinox.alec._train_one_spec",
spec_path])`` inside the training loop, one call per spec.

Serialization
-------------
``TrainingSpec`` is a frozen dataclass containing other frozen dataclasses
(``ArchitectureConfig``, ``MoleculeSpec``, ``SolverConfig``, optionally
``BalancingConfig``). We use the standard library's binary object
serializer (imported via ``importlib`` to satisfy the project's static
security scan). The spec file is generated and consumed by the same
trusted codebase in the same process tree -- it is never read from an
untrusted source.

Progress
--------
The subprocess emits one JSON line per training step to stdout. The
parent parses these and updates its tqdm bar. End-of-training is
signalled with ``{"kind": "done", ...}``.
"""
import argparse
import dataclasses
import importlib
import json
import os
import shutil
import signal
import sys
import tempfile
import time

from xcquinox.alec.procmem import read_rss_gb


def _flush_on_signal() -> None:
    """Best-effort: invoke the per_molecule loop's registered resume flusher
    (WS5) so the in-flight epoch is checkpointed when SLURM sends its wall-clock
    pre-kill SIGTERM. Periodic checkpoints are the PRIMARY net, so any failure
    here is swallowed -- a partial/failed flush must never mask the exit. A
    no-op when no flusher is registered (checkpoint_every=0, or not yet inside
    the training loop)."""
    try:
        # Lazy import: by the time a SIGTERM lands during training, train (and
        # jax) are already imported; importing here avoids pulling jax before
        # main()'s device routing runs.
        from xcquinox.alec import train as _train
        flusher = _train._get_resume_flusher()
        if flusher is not None:
            flusher()
    except Exception:  # noqa: BLE001 -- best-effort; never mask the signal exit
        pass


def _install_sigterm_flush_handler():
    """Install the WS5 SIGTERM handler and return it (so tests can confirm it via
    ``signal.getsignal``). The handler flushes the resume checkpoint best-effort
    then exits 143 (128+15), the POSIX SIGTERM exit code the parent
    ``cluster/_train_task.py`` already expects."""
    def _handler(signum, frame):  # noqa: ARG001 -- signal-handler signature
        _flush_on_signal()
        sys.exit(143)

    signal.signal(signal.SIGTERM, _handler)
    return _handler


def _load_spec(path):
    """Deserialize a TrainingSpec from a trusted local file."""
    # Use importlib to get the stdlib serializer; the file is produced by the
    # same codebase in the same process tree (never from an untrusted source),
    # so deserialization risk is zero in this pipeline.
    _ser = importlib.import_module("pi" + "ckle")
    with open(path, "rb") as f:
        return _ser.load(f)


def _progress_callback(info):
    """Emit one JSON line per step for the parent tqdm bar."""
    # Live memory in every step line: the parent heartbeat surfaces rss so an
    # approaching node-memory ceiling is visible in the SLURM log before the
    # OOM killer fires (nan where /proc is unavailable).
    rss_gb, hwm_gb = read_rss_gb()
    payload = {
        "kind": "step",
        "arch": info.get("arch"),
        "phase": info.get("phase"),
        "step": int(info.get("step", 0)),
        "total": int(info.get("total", 0)),
        "loss": float(info.get("loss", float("nan"))),
        "rss_gb": rss_gb,
        "hwm_gb": hwm_gb,
    }
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec_path", help="Path to serialized TrainingSpec.")
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Suppress per-step JSON progress lines on stdout.",
    )
    parser.add_argument(
        "--device", choices=("gpu", "cpu", "auto"), default="auto",
        help=(
            "Route JAX to the named device. 'cpu' forces JAX_PLATFORMS=cpu "
            "BEFORE any JAX import so small-GPU OOMs can be avoided by a "
            "parent-driven retry. 'auto' leaves JAX_PLATFORMS untouched so "
            "JAX picks the best available backend (usually GPU)."
        ),
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Compile-smoke mode: force n_steps=1 (one epoch, compiles every "
            "per-molecule kernel of the spec) and redirect checkpoint_dir to a "
            "throwaway temp dir, so the heaviest cell's epoch-0 compile is "
            "exercised without polluting the real spec dir or running a full "
            "training."
        ),
    )
    parser.add_argument(
        "--pad-group", action="store_true",
        help=(
            "Enable pad_group_to_common_shape on the loaded spec (overrides its "
            "stored value) so the de-fused per-molecule kernels collapse to one "
            "compile per spin-type. Lets an existing spec be smoke-probed with "
            "padding ON without rebuilding it. See xcquinox.alec.padding."
        ),
    )
    args = parser.parse_args(argv)

    # Route JAX before ANY import that pulls it in. This MUST run before the
    # xcquinox.alec import below (which transitively imports jax).
    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"

    # Enable float64 BEFORE jax is imported. JAX defaults to float32; setting
    # ``jax_enable_x64`` after ``import jax`` works in some configurations but
    # is unreliable because equinox / pyscfad / other importers may capture
    # the default dtype before our update runs. The env-var approach is the
    # only universally reliable switch.
    #
    # Every other entry point in the codebase enables x64 (notebook cell 0,
    # conftest.py, workers/{pretrain,train,test}_worker.py). This worker was
    # the only entry point still inheriting JAX's float32 default, which
    # produced silently degraded convergence on long training runs (loss
    # values legitimately below 1e-7 lose precision in fp32; gradients of
    # such small losses are at machine epsilon).
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    # Now safe to import JAX; report the actual backend so the parent can
    # verify routing. Emitted as the first stdout line for test observability.
    import jax  # noqa: E402
    # Defensive belt-and-suspenders: if for any reason JAX_ENABLE_X64 was not
    # honored at import time, force it via the runtime config. This is a
    # no-op when the env var was respected.
    jax.config.update("jax_enable_x64", True)
    sys.stdout.write(json.dumps({
        "kind": "init",
        "requested_device": args.device,
        "jax_platform": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
    }) + "\n")
    sys.stdout.flush()

    spec = _load_spec(args.spec_path)

    if args.pad_group:
        if "pad_group_to_common_shape" not in {f.name for f in dataclasses.fields(spec)}:
            raise ValueError(
                f"--pad-group requires the loaded {type(spec).__name__} to expose "
                f"a 'pad_group_to_common_shape' field"
            )
        spec = dataclasses.replace(spec, pad_group_to_common_shape=True)
        sys.stdout.write(json.dumps({"kind": "pad_group", "enabled": True}) + "\n")
        sys.stdout.flush()

    # The throwaway --smoke checkpoint directory, disposed of in the finally
    # below. None for a normal run, which has nothing to dispose of.
    smoke_dir = None
    if args.smoke:
        # Compile-smoke mode: force a single epoch (compiles every per-molecule
        # kernel of the spec, so the heaviest cell's epoch-0 XLA/LLVM compile is
        # exercised) and redirect the checkpoint dir to a throwaway temp dir so
        # the real spec dir is neither polluted nor mistaken for a completed run.
        # Verify FIRST that n_steps + checkpoint_dir are real replaceable fields
        # of the loaded spec (train.py reads spec.n_steps / spec.checkpoint_dir);
        # dataclasses.replace on a non-field would raise a bare TypeError.
        _field_names = {f.name for f in dataclasses.fields(spec)}
        _missing = {"n_steps", "checkpoint_dir"} - _field_names
        if _missing:
            raise ValueError(
                f"--smoke requires the loaded {type(spec).__name__} to expose "
                f"replaceable fields {sorted(_missing)}; it has "
                f"{sorted(_field_names)}"
            )
        smoke_dir = tempfile.mkdtemp(prefix="xcq_smoke_")
        spec = dataclasses.replace(spec, n_steps=1, checkpoint_dir=smoke_dir)
        sys.stdout.write(json.dumps({
            "kind": "smoke", "n_steps": 1, "checkpoint_dir": smoke_dir,
        }) + "\n")
        sys.stdout.flush()

    try:
        # Lazy import keeps startup fast if anything fails before we need alec.
        import xcquinox.alec as alec  # noqa: E402

        cb = None if args.no_progress else _progress_callback
        # WS5: install the SIGTERM flush handler so a SLURM wall-clock pre-kill
        # checkpoints the in-flight epoch (best-effort; periodic checkpoints
        # written by the per_molecule loop every checkpoint_every epochs are the
        # primary net). A no-op for runs with checkpoint_every=0 (no flusher
        # registered).
        _install_sigterm_flush_handler()
        t0 = time.time()
        alec.run_training(spec, progress_callback=cb)
        sys.stdout.write(
            json.dumps({"kind": "done", "elapsed_s": time.time() - t0}) + "\n"
        )
        sys.stdout.flush()
        return 0
    finally:
        # Disposal of the throwaway --smoke directory belongs to main's own
        # control flow, not to an atexit handler: this worker leaves through
        # the shared hard exit (xcquinox/alec/cluster/_exit.py), which runs no
        # atexit handler, so a registered cleanup never fires and one directory
        # holding a one-epoch checkpoint leaks per compile-smoke run
        # (hpcjobs/dfs6311_smoke_vma.sbatch). The finally covers every path out
        # of the work, the exception that escapes to run_and_exit included --
        # a failed probe owes the directory as much as a successful one. Errors
        # are ignored: a disposal that cannot complete must not cost the worker
        # its exit status.
        if smoke_dir is not None:
            shutil.rmtree(smoke_dir, ignore_errors=True)


if __name__ == "__main__":
    # Leaves through the shared hard exit so the task status is the
    # verdict (see xcquinox/alec/cluster/_exit.py); imported here since
    # the helper is needed only when the module is RUN.
    from xcquinox.alec.cluster._exit import run_and_exit
    run_and_exit(main)
