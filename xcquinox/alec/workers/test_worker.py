"""xcquinox.alec.workers.test_worker: thin subprocess entry for evaluation."""
import argparse
import json
import os
import pickle
import sys
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
    sys.exit(main())
