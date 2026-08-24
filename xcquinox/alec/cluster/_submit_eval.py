"""xcquinox.alec.cluster._submit_eval -- deferred-eval submitter.

In deferred-eval mode (``submit --defer-eval`` / ``defer_eval: true``) the
initial ``submit`` does NOT queue the eval array up front. Instead it submits a
tiny launcher job (``afterany:train``); when the train array terminates the
launcher runs this module::

    python -m xcquinox.alec.cluster._submit_eval <RUN_DIR>

which submits the eval array (``aftercorr:train`` -- identical per-index gating
to a non-deferred run) and records it in ``jobs.json``. This shrinks the per-run
queued-job footprint, which matters under SLURM per-user submit caps.

It is also the MANUAL fallback: if compute nodes are barred from ``sbatch``, run
the same thing from a login node once the train array finishes (the harness CLI
exposes it as ``python -m xcquinox.alec.cluster submit-eval <RUN_DIR>``).

The core ``submit_deferred_eval`` is idempotent: if a non-superseded ``eval``
record already exists it is a no-op, so the launcher and a manual run can never
double-submit -- whichever fires first wins.

Its own work is light -- it renders/sbatches a script, validates the resolved
config and reads/writes ``jobs.json`` -- but importing it is NOT light: the
``xcquinox`` package ``__init__`` chain pulls in jax, jaxlib, equinox, optax,
pyscf and ase, 0.75-0.94 s in a fresh interpreter. The launcher job must
therefore be sized for a full jax import rather than for a bare-stdlib one.
"""
import argparse
import os
import subprocess
import sys

from xcquinox.alec.cluster.grid_config import (
    load_grid_config,
    expand_grid,
    validate_grid_semantics,
)
from xcquinox.alec.cluster.domain import get_domain_profile
from xcquinox.alec.cluster import job_tracking


_RESOLVED_CONFIG = "resolved_config.yaml"


def _parse_sbatch_id(stdout: str) -> str:
    """Parse the array-job id from ``sbatch --parsable`` stdout."""
    return stdout.strip().split(";")[0].split()[0]


def _newest_live_train_record(records):
    """Return the newest non-superseded ``train`` record, or None.

    "Newest" = the highest ``generation`` among live train records (matches how
    the harness resolves the active train array elsewhere).
    """
    live = [
        r for r in records
        if r.get("kind") == "train" and not r.get("superseded", False)
    ]
    if not live:
        return None
    return max(live, key=lambda r: r.get("generation", 0))


def _newest_live_eval_id(records):
    """Return the array-job id of the newest non-superseded ``eval`` record."""
    live = [
        r for r in records
        if r.get("kind") == "eval" and not r.get("superseded", False)
    ]
    if not live:
        return None
    return max(live, key=lambda r: r.get("generation", 0))["array_job_id"]


def submit_deferred_eval(run_dir: str, *, force: bool = False) -> dict:
    """Submit the eval array for an already-submitted (deferred) run.

    Reads the run's ``resolved_config.yaml`` + ``jobs.json``, submits the eval
    array ``aftercorr:<train_id>`` (re-using ``scripts/eval_array.sbatch``, or
    re-rendering it if absent), and records the eval job.

    Idempotency: if a non-superseded ``eval`` record already exists this is a
    no-op (unless ``force``), so the launcher job and a manual invocation cannot
    double-submit.

    Returns a dict: ``{"submitted": bool, "eval_id": str|None, "train_id": str,
    "run_dir": str, "reason": str|None}``.

    Raises:
        RuntimeError: no live train record to gate eval on, or sbatch rejected
            the submission (the message carries sbatch's stderr + a hint to run
            this from a login node if compute-node submission is barred).
    """
    run_dir = os.path.abspath(run_dir)

    records = job_tracking.read_job_records(run_dir)

    # Idempotency guard -- whichever of {launcher, manual} fires first wins.
    existing_eval = _newest_live_eval_id(records)
    if existing_eval is not None and not force:
        print(f"submit-eval: eval already submitted: {existing_eval} "
              f"(no-op; pass --force to submit another)", flush=True)
        return {"submitted": False, "eval_id": existing_eval,
                "train_id": None, "run_dir": run_dir,
                "reason": "already_submitted"}

    train_rec = _newest_live_train_record(records)
    if train_rec is None:
        raise RuntimeError(
            f"submit-eval: no live train record in {run_dir}/jobs.json -- "
            "nothing to submit an eval array for. (A deferred submit writes "
            "pretrain/preflight/train records; run submit first.)"
        )
    train_id = train_rec["array_job_id"]
    indices = list(train_rec["indices"])

    cfg = load_grid_config(os.path.join(run_dir, _RESOLVED_CONFIG))
    # resolved_config.yaml is an ordinary file that outlives the `submit`
    # which validated it, and the branch below can RE-RENDER
    # eval_array.sbatch from it. The same semantic validation `submit` runs
    # is therefore re-run here, before the render and before sbatch, rather
    # than trusting whatever last wrote the file.
    validate_grid_semantics(cfg, get_domain_profile(cfg.domain_profile))
    n_specs = len(expand_grid(cfg))
    array_max = n_specs - 1

    eval_path = os.path.join(run_dir, "scripts", "eval_array.sbatch")
    if not os.path.isfile(eval_path):
        # Defensive: re-render from the resolved config if the script is gone.
        from xcquinox.alec.cluster.submit import render_sbatch
        os.makedirs(os.path.dirname(eval_path), exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as f:
            f.write(render_sbatch("eval", cfg, run_dir, array_max=array_max))

    eval_cmd = [
        "sbatch", "--parsable",
        f"--dependency=aftercorr:{train_id}", eval_path,
    ]
    try:
        proc = job_tracking._run_slurm(eval_cmd)
    except subprocess.CalledProcessError as exc:
        stderr = (getattr(exc, "stderr", "") or "").strip()
        raise RuntimeError(
            f"submit-eval: sbatch rejected the eval array ({exc})"
            + (f"\n  sbatch stderr: {stderr}" if stderr else "")
            + "\n  If this ran from a compute node that cannot submit jobs, "
            "run it from a login node once train finishes:\n    "
            f"python -m xcquinox.alec.cluster submit-eval {run_dir}"
        ) from exc

    eval_id = _parse_sbatch_id(proc.stdout)
    job_tracking.append_job_record(run_dir, "eval", eval_id, indices)
    print(f"submit-eval: submitted eval array {eval_id} "
          f"(aftercorr:{train_id}) for {len(indices)} specs", flush=True)
    return {"submitted": True, "eval_id": eval_id, "train_id": train_id,
            "run_dir": run_dir, "reason": None}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="The materialized run directory.")
    parser.add_argument(
        "--force", action="store_true",
        help="submit the eval array even if one is already recorded.")
    args = parser.parse_args(argv)

    try:
        submit_deferred_eval(args.run_dir, force=args.force)
    except RuntimeError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


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
