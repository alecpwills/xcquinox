"""xcquinox.alec.workers.pretrain_worker — thin subprocess entry for pretraining."""
import argparse
import json
import os
import pickle
import sys
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

    # Set thread env BEFORE any JAX import
    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen=true "
        f"intra_op_parallelism_threads={parsed.threads}"
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
    sys.exit(main())
