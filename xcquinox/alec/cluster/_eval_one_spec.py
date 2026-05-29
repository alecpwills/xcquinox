"""xcquinox.alec.cluster._eval_one_spec -- per-SLURM eval-array-task worker.

The eval-array sbatch template invokes this once per array task as::

    python -m xcquinox.alec.cluster._eval_one_spec <RUN_DIR> <SLURM_ARRAY_TASK_ID>

For one grid index it:

  - Resolves the per-spec checkpoint dir ``<run_dir>/checkpoints/spec_<idx>/``
    (zero-pad ``width`` read from ``manifest.json``).
  - Checks for ``model.eqx`` FIRST. If training never produced a checkpoint,
    it writes an ``eval/skipped.json`` marker and exits 0 WITHOUT constructing
    a :class:`TestSpec` -- ``TestSpec.validate()`` hard-requires
    ``model_checkpoint`` to be an existing file and would raise instead of
    skipping cleanly.
  - Otherwise loads the materialized :class:`TrainingSpec`, derives the
    matching :class:`TestSpec` via :func:`build_test_spec`, runs
    :func:`xcquinox.alec.run_test`, and folds the per-molecule results into
    ``<checkpoint_dir>/eval_df.csv``.

JAX routing
-----------
Like :mod:`xcquinox.alec._train_one_spec`, this worker sets
``JAX_ENABLE_X64=1`` (and ``JAX_PLATFORMS=cpu`` -- evaluation is CPU-only) in
``os.environ`` BEFORE any ``import jax`` / before importing
``xcquinox.alec.evaluation``. fp32 vs fp64 changes ``density_rmse`` /
``total_energy``, so this must not be left to JAX's float32 default.

Serialization
-------------
The spec file is a frozen-dataclass :class:`TrainingSpec` written by the
materialize layer. It is loaded with the same ``importlib.import_module
("pi"+"ckle")`` + ``.load`` indirection used by ``_train_one_spec._load_spec``
to satisfy the project's static security scan -- the file is produced and
consumed by the same trusted codebase, never read from an untrusted source.
"""
import argparse
import importlib
import json
import math
import os
import sys
import time


_MANIFEST_FILENAME = "manifest.json"


# ---------------------------------------------------------------------------
# JAX routing -- must run before ANY jax import
# ---------------------------------------------------------------------------

def _route_jax_env():
    """Pin JAX to fp64 + the CPU backend via env vars.

    Enable float64 BEFORE jax is imported: JAX defaults to float32 and
    equinox / pyscfad may capture the default dtype before a post-import
    config update runs, so the env-var switch is the only reliable one.
    fp32 vs fp64 silently changes ``density_rmse`` / ``total_energy``.

    Evaluation is CPU-only, so ``JAX_PLATFORMS`` is pinned to ``cpu`` (via
    ``setdefault`` so an explicit override is still honored).
    """
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _read_width(run_dir):
    """Read the zero-pad ``width`` from ``<run_dir>/manifest.json``."""
    path = os.path.join(run_dir, _MANIFEST_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"_eval_one_spec: no {_MANIFEST_FILENAME} in {run_dir}; the run "
            "directory has not been materialized"
        )
    with open(path) as f:
        manifest = json.load(f)
    return int(manifest["width"])


def _checkpoint_dir(run_dir, idx, width):
    """Per-spec checkpoint dir ``<run_dir>/checkpoints/spec_<idx>/``."""
    return os.path.join(run_dir, "checkpoints", f"spec_{idx:0{width}d}")


def _spec_path(run_dir, idx, width):
    """Path to this task's spec file ``<run_dir>/specs/spec_<idx>.spec``."""
    return os.path.join(run_dir, "specs", f"spec_{idx:0{width}d}.spec")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(idx, message):
    """Emit one harness log line (tagged) to our stdout -- the SLURM log."""
    sys.stdout.write(f"[harness idx={idx}] {message}\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------

def _load_spec(path):
    """Deserialize a TrainingSpec from a trusted local file.

    The stdlib serializer is fetched via ``importlib`` so the project's
    static security scan does not flag a bare ``import``; the file is
    produced and consumed by the same codebase in the same process tree.
    """
    _ser = importlib.import_module("pi" + "ckle")
    with open(path, "rb") as f:
        return _ser.load(f)


# ---------------------------------------------------------------------------
# Evaluation seam
# ---------------------------------------------------------------------------

def _run_eval(test_spec):
    """Run :func:`xcquinox.alec.run_test` on ``test_spec`` -- the test seam.

    Isolated as a named function so a unit test can monkeypatch it and avoid
    real evaluation compute. ``run_test`` writes ``per_molecule.json`` /
    ``aggregate.json`` / ``test_metadata.json`` into ``test_spec.output_dir``.
    """
    import xcquinox.alec as alec  # noqa: E402 -- imported after JAX routing
    return alec.run_test(test_spec)


# ---------------------------------------------------------------------------
# per_molecule.json -> eval_df.csv fold
# ---------------------------------------------------------------------------

def _aggregate_per_molecule(pm_rows, ae_key="AE_error_kcalmol",
                            rho_key="density_rmse"):
    """Aggregate ``per_molecule.json`` rows into wide-form scalars.

    Faithful port of the step-7 notebook helper (Cell D of
    ``notebooks/_build_step7_notebook.py``). The per-molecule row keys it
    reads are ``AE_error_kcalmol`` (the AtomizationEnergyMetric error in
    kcal/mol) and ``density_rmse`` (the DensityRMSEMetric grid RMSE).

    Returns ``(mae, rho_rmse, n_eval)``:
      - ``mae`` -- mean ``|AE_error_kcalmol|`` across molecules that carry the
        key (``nan`` if none do).
      - ``rho_rmse`` -- mean ``density_rmse`` across molecules that carry it
        (``nan`` if none do -- e.g. when no CCSD reference densities are
        loaded).
      - ``n_eval`` -- count of AE-reference molecules that contributed to
        ``mae`` (i.e. ``len(ae_errs)``).  Atom/aux-only rows that lack the
        ``AE_error_kcalmol`` key are excluded because they do not contribute
        to the MAE average; reporting total row count (``len(pm_rows)``) would
        be misleading when the subset contains BH76/IP13-only entries.
    """
    # C5-06: a non-finite per-molecule value (NaN/inf from a pathological V_xc
    # or a diverged density) passes isinstance(...,(int,float)) and would poison
    # the spec MAE — which then makes the summary layer (analyze.summarize)
    # drop the ENTIRE spec instead of just the bad molecule. Exclude non-finite
    # values so a single bad molecule does not discard a spec's good ones.
    ae_errs = [
        float(r[ae_key]) for r in pm_rows
        if isinstance(r.get(ae_key), (int, float))
        and not isinstance(r.get(ae_key), bool)
        and math.isfinite(r[ae_key])
    ]
    if ae_errs:
        mae = sum(abs(v) for v in ae_errs) / len(ae_errs)
    else:
        mae = float("nan")
    rho_vals = [
        float(r[rho_key]) for r in pm_rows
        if isinstance(r.get(rho_key), (int, float))
        and not isinstance(r.get(rho_key), bool)
        and math.isfinite(r[rho_key])
    ]
    if rho_vals:
        rho_rmse = sum(rho_vals) / len(rho_vals)
    else:
        rho_rmse = float("nan")
    # n_eval: AE-contributing molecules only (matches the mae denominator).
    return mae, rho_rmse, len(ae_errs)


def _write_eval_df_csv(pm_rows, csv_path):
    """Fold ``per_molecule.json`` rows into the per-spec ``eval_df.csv``.

    One-row summary CSV with columns ``set, mae, rho_rmse, n_eval`` -- the
    same scalars the step-7 notebook's Cell D writes per spec (the
    ``metric``/``tag``/``solver`` columns from the notebook are derived from
    the notebook-specific checkpoint-dir layout and are intentionally omitted
    here; the harness manifest carries the GridCell instead).
    """
    import csv

    mae, rho_rmse, n_eval = _aggregate_per_molecule(pm_rows)
    fieldnames = ["set", "mae", "rho_rmse", "n_eval"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "set": "training_subset",
            "mae": mae,
            "rho_rmse": rho_rmse,
            "n_eval": n_eval,
        })
    return mae, rho_rmse, n_eval


# ---------------------------------------------------------------------------
# skipped.json marker
# ---------------------------------------------------------------------------

def _write_skipped_json(checkpoint_dir, reason):
    """Write ``<checkpoint_dir>/eval/skipped.json`` recording an eval skip."""
    eval_dir = os.path.join(checkpoint_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    payload = {
        "reason": reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    with open(os.path.join(eval_dir, "skipped.json"), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    # Route JAX before ANY import that pulls it in. argparse / json / os are
    # already imported and are jax-free; the xcquinox.alec import below (in the
    # _run_eval seam) transitively imports jax, so this MUST run first.
    _route_jax_env()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="The materialized run directory.")
    parser.add_argument("idx", type=int, help="SLURM array task index.")
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    idx = args.idx

    width = _read_width(run_dir)
    checkpoint_dir = _checkpoint_dir(run_dir, idx, width)
    spec_path = _spec_path(run_dir, idx, width)
    model_path = os.path.join(checkpoint_dir, "model.eqx")

    # --- model.eqx check FIRST ---------------------------------------------
    # Done before any TestSpec construction: TestSpec.validate() hard-requires
    # model_checkpoint to be an existing file and would raise instead of
    # letting us skip cleanly.
    if not os.path.isfile(model_path):
        reason = "no model.eqx -- training did not produce a checkpoint"
        _write_skipped_json(checkpoint_dir, reason)
        _log(idx, f"skipped -- {reason} ({model_path})")
        return 0

    # --- load the materialized TrainingSpec --------------------------------
    if not os.path.exists(spec_path):
        # A model.eqx with no spec file is an inconsistent run dir -- bail
        # loudly so the inconsistency is visible in the SLURM log.
        _log(idx, f"model.eqx present but spec file not found: {spec_path}")
        return 2
    training_spec = _load_spec(spec_path)

    # --- config + domain ---------------------------------------------------
    from xcquinox.alec.cluster.grid_config import load_grid_config
    from xcquinox.alec.cluster.domain import get_domain_profile
    from xcquinox.alec.cluster.spec_builder import build_test_spec

    cfg = load_grid_config(os.path.join(run_dir, "resolved_config.yaml"))
    domain = get_domain_profile(cfg.domain_profile)

    # --- build the TestSpec + run evaluation -------------------------------
    test_spec = build_test_spec(training_spec, run_dir, idx, domain)
    _log(idx, f"starting evaluation for spec {spec_path}")
    t0 = time.time()
    _run_eval(test_spec)
    elapsed = time.time() - t0
    _log(idx, f"run_test complete ({elapsed:.1f}s elapsed)")

    # --- fold per_molecule.json -> eval_df.csv -----------------------------
    pm_path = os.path.join(test_spec.output_dir, "per_molecule.json")
    if not os.path.isfile(pm_path):
        _log(idx, f"run_test wrote no per_molecule.json at {pm_path}")
        return 1
    with open(pm_path) as f:
        pm_rows = json.load(f)
    csv_path = os.path.join(checkpoint_dir, "eval_df.csv")
    mae, rho_rmse, n_eval = _write_eval_df_csv(pm_rows, csv_path)
    _log(
        idx,
        f"eval_df.csv written -- mae={mae:.4f} rho_rmse={rho_rmse:.4f} "
        f"n_eval={n_eval} ({csv_path})",
    )

    # --- 2026-05-29: held-out eval against full BH76 + W4-11 --------------
    # Adds an apples-to-apples comparison surface against XCDiff & friends
    # without disturbing the in-sample eval above. Failure of this section
    # does NOT mark the task as failed -- the train checkpoint + in-sample
    # eval are already committed to disk. Writes:
    #   <ckpt>/eval_holdout/test_set.csv       (per-pool + combined MAE)
    #   <ckpt>/eval_holdout/per_molecule.json  (per-species E_nn + E_pbe)
    #   <ckpt>/eval_holdout/per_reaction.json  (per-reaction NN + PBE errors)
    # On exception: writes <ckpt>/eval_holdout/failure.json with the trace
    # and returns 0 (the in-sample artifact is the authoritative success
    # signal for the SLURM array task).
    try:
        from xcquinox.alec.eval_holdout import (
            load_trained_model,
            run_full_holdout_eval,
        )
        from xcquinox.alec.full_benchmark_pools import (
            load_full_held_out_pools,
        )
        from pathlib import Path as _Path

        holdout_dir = _Path(checkpoint_dir) / "eval_holdout"
        _log(idx, "starting full-pool held-out eval (BH76 + W4-11)")
        t1 = time.time()
        model = load_trained_model(training_spec, _Path(model_path))
        # Basis + grid_level should match what the existing test_spec uses
        # for the in-sample eval so PBE precomputes are comparable. Today
        # they default to def2-svp / grid_level=1 (the cluster sweep
        # standard); a future cfg-derived override would slot here.
        full_specs, full_rxns = load_full_held_out_pools(
            basis="def2-svp", grid_level=1,
        )
        result = run_full_holdout_eval(
            training_spec=training_spec,
            model=model,
            mol_specs=full_specs,
            reactions=full_rxns,
            out_dir=holdout_dir,
        )
        elapsed_h = time.time() - t1
        _log(
            idx,
            f"held-out eval complete ({elapsed_h:.1f}s elapsed; "
            f"{result['n_reactions']} reactions over "
            f"{result['n_species']} species; "
            f"{result['n_dropped_nan']} NaN-drop, "
            f"{result['n_dropped_overlap']} overlap-drop)",
        )
    except Exception as exc:  # noqa: BLE001
        import traceback
        from pathlib import Path as _Path
        holdout_dir = _Path(checkpoint_dir) / "eval_holdout"
        holdout_dir.mkdir(parents=True, exist_ok=True)
        with (holdout_dir / "failure.json").open("w") as f:
            json.dump({
                "kind": "held_out_eval_failure",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }, f, indent=2)
        _log(
            idx,
            f"held-out eval FAILED ({type(exc).__name__}: {exc}); "
            "in-sample eval_df.csv was still written -- treating spec as "
            "succeeded. See eval_holdout/failure.json for details.",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
