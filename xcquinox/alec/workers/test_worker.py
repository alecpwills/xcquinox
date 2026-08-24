"""xcquinox.alec.workers.test_worker: thin subprocess entry for evaluation."""
import argparse
import json
import os
import pickle
import time
import traceback


def main(args=None):
    parser = argparse.ArgumentParser(description="Alec test worker")
    parser.add_argument("--arch", required=True)
    parser.add_argument("--spec-pickle", required=True)
    parser.add_argument("--checkpoint-base", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parsed = parser.parse_args(args)

    # Set thread env BEFORE any JAX import. setdefault respects a launcher's
    # XLA_FLAGS; the old mis-prefixed ``intra_op_parallelism_threads`` token
    # (silently ignored by XLA) is dropped -- intra-op width is bounded by the
    # OMP/MKL/OPENBLAS caps below.
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=true "
        "--xla_llvm_disable_expensive_passes=true "
        "--xla_backend_optimization_level=1",
    )
    os.environ["OMP_NUM_THREADS"] = str(parsed.threads)
    os.environ["MKL_NUM_THREADS"] = str(parsed.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(parsed.threads)

    start = time.time()
    try:
        import jax
        jax.config.update("jax_enable_x64", True)

        with open(parsed.spec_pickle, "rb") as f:
            spec = pickle.load(f)

        from xcquinox.alec.evaluation import run_test
        result = run_test(spec)
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
