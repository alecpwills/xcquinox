"""xcquinox.alec.workers.pretrain_worker: thin subprocess entry for pretraining."""
import argparse
import json
import os
import pickle
import time
import traceback


def main(args=None):
    parser = argparse.ArgumentParser(description="Alec pretrain worker")
    parser.add_argument("--arch", required=True)
    parser.add_argument("--spec-pickle", required=True)
    parser.add_argument("--checkpoint-base", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parsed = parser.parse_args(args)

    # Set thread env BEFORE any JAX import. Respect an XLA_FLAGS already set by
    # the launcher (e.g. the cluster sbatch, which carries the compile-memory
    # trims); only fall back to the trims here when it is unset -- unconditional
    # assignment would clobber the sbatch value. The old
    # `intra_op_parallelism_threads=<n>` token was mis-prefixed (no `--xla_`
    # prefix) so XLA silently ignored it -- dropped; intra-op width is bounded by
    # the OMP/MKL/OPENBLAS caps below.
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
    # pool member to its share of the node (see workers/_cpu_bind.py
    # for the measurements; a worker launched by hand carries no bind request
    # and stays unbound).
    import sys as _sys
    try:
        import _cpu_bind  # sibling file; loads WITHOUT the package __init__
    except ImportError:
        # Imported as a package module: the package (and its JAX backend)
        # are already up, so the pin below binds the calling thread only.
        # Said out loud -- a silently degraded bind ran at 10x the node
        # share once.
        print("[cpu-bind] WARNING: package-mediated import; the JAX pool "
              "predates the pin", file=_sys.stderr, flush=True)
        from xcquinox.alec.workers import _cpu_bind
    _bound = _cpu_bind.apply()
    if _bound is None and os.environ.get(_cpu_bind.WORKER_BIND_CPUS_ENV):
        print("[cpu-bind] WARNING: bind requested "
              f"({os.environ.get(_cpu_bind.WORKER_BIND_CPUS_ENV)} CPUs) but "
              "not applied (no slot, budget at or above the allowance, or no "
              "sched_setaffinity)", file=_sys.stderr, flush=True)

    start = time.time()
    try:
        import jax
        jax.config.update("jax_enable_x64", True)

        with open(parsed.spec_pickle, "rb") as f:
            spec = pickle.load(f)

        from xcquinox.alec.pretrain import run_pretrain
        result = run_pretrain(spec)
        duration = time.time() - start
        result["status"] = "success"
        result["duration"] = duration
        print(json.dumps(result))
        return 0
    except Exception:
        duration = time.time() - start
        payload = {
            "status": "failed",
            "arch": parsed.arch,
            "error": traceback.format_exc().splitlines()[-1],
            "traceback": traceback.format_exc(),
            "duration": duration,
        }
        print(json.dumps(payload))
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
