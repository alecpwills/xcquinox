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
import time
import traceback


def compute_shard(run_dir, spec_idx, names, basis, grid_level,
                  model_name="model.eqx",
                  coldstart=False):
    """Evaluate ``names`` (a held-out molecule subset) for spec ``spec_idx`` of
    ``run_dir``. Returns ``{energies, pbe_energies, mol_records}``.

    Imports jax-touching modules lazily so the caller can pin thread env first.
    Reuses ``_eval_one_spec``'s path helpers and ``load_trained_model`` so the
    worker loads exactly what the serial eval task loads. ``model_name`` selects
    which checkpoint in the spec dir to load (``model.eqx`` for the final-step
    eval, ``model_best.eqx`` for the best-loss eval); the orchestrator threads it
    so a best-checkpoint pass shards the BEST weights, not the final ones."""
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
    model_path = os.path.join(checkpoint_dir, model_name)

    training_spec = _load_spec(spec_path)
    if coldstart:
        # Shard subprocesses reload the spec themselves, so the orchestrator's
        # in-memory override cannot reach them; apply the SAME shared helper.
        import dataclasses

        from xcquinox.alec.eval_holdout import coldstart_solver_config
        training_spec = dataclasses.replace(
            training_spec,
            solver_config=coldstart_solver_config(training_spec.solver_config))
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
    parser.add_argument("--model-name", default="model.eqx",
                        help="checkpoint filename in the spec dir to evaluate "
                             "(model.eqx final / model_best.eqx best)")
    parser.add_argument("--coldstart", action="store_true",
                        help="apply the cold-start override (minao seed, 25 "
                             "cycles) to the reloaded spec's solver -- the "
                             "eval_holdout_coldstart channel")
    parsed = parser.parse_args(args)

    # Pin thread env BEFORE any JAX import (one BLAS thread per worker by
    # default so N workers saturate N cores without oversubscription). Respect
    # an XLA_FLAGS already set by the launcher (parallel._thread_env carries the
    # compile-memory trims); only fall back here when unset -- unconditional
    # assignment would clobber the launcher value, and the old
    # ``intra_op_parallelism_threads=<n>`` token was mis-prefixed (no ``--xla_``)
    # so XLA silently ignored it -- dropped. The eigen token is dropped:
    # measured inert on jaxlib 0.7.0 (thunk runtime), and the pool
    # bound is the CPU affinity applied below, not an XLA flag.
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_llvm_disable_expensive_passes=true "
        "--xla_backend_optimization_level=1",
    )
    os.environ["OMP_NUM_THREADS"] = str(parsed.threads)
    os.environ["MKL_NUM_THREADS"] = str(parsed.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(parsed.threads)

    # CPU bind BEFORE the JAX import: TSL sizes the XLA intra-op pool from
    # sched_getaffinity, so this -- not any XLA flag -- is what confines a
    # pool member to its share of the node (see parallel.apply_worker_cpu_bind
    # for the measurements; a worker launched by hand carries no bind request
    # and stays unbound).
    from xcquinox.alec.parallel import apply_worker_cpu_bind
    apply_worker_cpu_bind()
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    start = time.time()
    try:
        import jax
        jax.config.update("jax_enable_x64", True)

        with open(parsed.names_file) as f:
            names = json.load(f)

        shard = compute_shard(parsed.run_dir, parsed.spec_idx, names,
                              parsed.basis, parsed.grid_level,
                              model_name=parsed.model_name,
                              coldstart=parsed.coldstart)
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
    # The worker's verdict is the status this process hands its parent
    # (``parallel.run_workers`` reads the return code beside the JSON result
    # line), and JAX's atexit teardown can abort the interpreter AFTER main()
    # has returned it (cluster job 2134455: a harness stage logged its own
    # SUCCEEDED line and then died in glibc's "corrupted size vs. prev_size",
    # rc -6, so the completed stage read as FAILED). run_and_exit flushes and
    # leaves through os._exit, so the status is the verdict.
    #
    # The shared helper is loaded BY PATH rather than imported: these modules
    # are reached only via direct-file launch and set their thread caps inside
    # main() before the first JAX import, so a package-qualified import here
    # would pull xcquinox.alec.cluster -- and JAX with it -- before those caps
    # are in place. The helper itself is stdlib-only.
    import importlib.util as _importlib_util
    import pathlib as _pathlib

    _exit_path = (_pathlib.Path(__file__).resolve().parent.parent
                  / "cluster" / "_exit.py")
    _exit_spec = _importlib_util.spec_from_file_location(
        "_xcquinox_alec_hard_exit", _exit_path)
    _exit_mod = _importlib_util.module_from_spec(_exit_spec)
    _exit_spec.loader.exec_module(_exit_mod)
    _exit_mod.run_and_exit(main)
