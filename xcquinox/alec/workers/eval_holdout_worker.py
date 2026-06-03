"""xcquinox.alec.workers.eval_holdout_worker: held-out eval shard worker.

A thin subprocess entry that evaluates ONE molecule shard of a spec's held-out
set (a subset of BH76 + W4-11) and writes a small JSON payload the parent
orchestrator merges. Mirrors ``test_worker.py``: thread env is pinned from
``--threads`` BEFORE importing jax, the model is reloaded from its checkpoint
path (never pickled across the process boundary), and a status JSON is printed
to stdout for ``parallel.run_workers``.

The per-molecule compute itself lives in
``eval_holdout.compute_holdout_per_molecule``: shared verbatim with the serial
driver, so a shard's energies are byte-for-byte what the serial path produces.
"""
import argparse
import json
import os
import sys
import time
import traceback


def compute_shard(run_dir, spec_idx, names, basis, grid_level):
    """Evaluate ``names`` (a held-out molecule subset) for spec ``spec_idx`` of
    ``run_dir``. Returns ``{energies, pbe_energies, mol_records}``.

    Imports jax-touching modules lazily so the caller can pin thread env first.
    Reuses ``_eval_one_spec``'s path helpers and ``load_trained_model`` so the
    worker loads exactly what the serial eval task loads."""
    from pathlib import Path

    from xcquinox.alec.cluster._eval_one_spec import (
        _read_width, _checkpoint_dir, _spec_path, _load_spec,
    )
    from xcquinox.alec.eval_holdout import (
        load_trained_model, compute_holdout_per_molecule,
    )
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools

    width = _read_width(run_dir)
    checkpoint_dir = _checkpoint_dir(run_dir, spec_idx, width)
    spec_path = _spec_path(run_dir, spec_idx, width)
    model_path = os.path.join(checkpoint_dir, "model.eqx")

    training_spec = _load_spec(spec_path)
    model = load_trained_model(training_spec, Path(model_path))

    full_specs, _full_rxns = load_full_held_out_pools(
        basis=basis, grid_level=grid_level)
    subset = {n: full_specs[n] for n in names if n in full_specs}

    per = compute_holdout_per_molecule(training_spec, model, subset)
    return {
        "energies": per["energies"],
        "pbe_energies": per["pbe_energies"],
        "mol_records": per["mol_records"],
    }


def main(args=None):
    parser = argparse.ArgumentParser(description="Alec held-out eval shard worker")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--spec-idx", type=int, required=True)
    parser.add_argument("--names-file", required=True,
                        help="JSON file: list of held-out molecule names for this shard")
    parser.add_argument("--out-shard", required=True,
                        help="path to write the shard's {energies,pbe_energies,mol_records} JSON")
    parser.add_argument("--basis", required=True)
    parser.add_argument("--grid-level", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parsed = parser.parse_args(args)

    # Pin thread env BEFORE any JAX import (one BLAS thread per worker by
    # default so N workers saturate N cores without oversubscription).
    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen=true "
        f"intra_op_parallelism_threads={parsed.threads}"
    )
    os.environ["OMP_NUM_THREADS"] = str(parsed.threads)
    os.environ["MKL_NUM_THREADS"] = str(parsed.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(parsed.threads)
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    start = time.time()
    try:
        import jax
        jax.config.update("jax_enable_x64", True)

        with open(parsed.names_file) as f:
            names = json.load(f)

        shard = compute_shard(parsed.run_dir, parsed.spec_idx, names,
                              parsed.basis, parsed.grid_level)
        with open(parsed.out_shard, "w") as f:
            json.dump(shard, f)

        print(json.dumps({
            "status": "success",
            "n_done": len(shard["energies"]),
            "out_shard": parsed.out_shard,
            "duration": time.time() - start,
        }))
        return 0
    except Exception:
        print(json.dumps({
            "status": "failed",
            "error": traceback.format_exc().splitlines()[-1],
            "traceback": traceback.format_exc(),
            "duration": time.time() - start,
        }))
        return 1


if __name__ == "__main__":
    sys.exit(main())
