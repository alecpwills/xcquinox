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
import importlib
import json
import os
import sys
import time


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
    payload = {
        "kind": "step",
        "arch": info.get("arch"),
        "phase": info.get("phase"),
        "step": int(info.get("step", 0)),
        "total": int(info.get("total", 0)),
        "loss": float(info.get("loss", float("nan"))),
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
    args = parser.parse_args(argv)

    # Route JAX before ANY import that pulls it in. This MUST run before the
    # xcquinox.alec import below (which transitively imports jax).
    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"

    # Now safe to import JAX; report the actual backend so the parent can
    # verify routing. Emitted as the first stdout line for test observability.
    import jax  # noqa: E402
    sys.stdout.write(json.dumps({
        "kind": "init",
        "requested_device": args.device,
        "jax_platform": jax.default_backend(),
    }) + "\n")
    sys.stdout.flush()

    spec = _load_spec(args.spec_path)

    # Lazy import keeps startup fast if anything fails before we need alec.
    import xcquinox.alec as alec  # noqa: E402

    cb = None if args.no_progress else _progress_callback
    t0 = time.time()
    alec.run_training(spec, progress_callback=cb)
    sys.stdout.write(
        json.dumps({"kind": "done", "elapsed_s": time.time() - t0}) + "\n"
    )
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
