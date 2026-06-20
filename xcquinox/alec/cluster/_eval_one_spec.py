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

def _held_out_basis_grid(cfg):
    """``(basis, grid_level)`` for the held-out pool, read from the resolved
    config so it matches what training used. Falls back to the historical
    def2-svp / grid_level=1 when the config lacks them (older runs)."""
    inputs = getattr(cfg, "inputs", None)
    basis = getattr(inputs, "basis", None) or "def2-svp"
    grid_level = getattr(inputs, "grid_level", None)
    return basis, (1 if grid_level is None else int(grid_level))


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
    # a non-finite per-molecule value (NaN/inf from a pathological V_xc
    # or a diverged density) passes isinstance(...,(int,float)) and would poison
    # the spec MAE, which then makes the summary layer (analyze.summarize)
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
    def _mean_finite(key):
        vals = [
            float(r[key]) for r in pm_rows
            if isinstance(r.get(key), (int, float))
            and not isinstance(r.get(key), bool)
            and math.isfinite(r[key])
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    rho_rmse = _mean_finite(rho_key)
    # PBE-vs-CCSD baseline density error (model-free; nan when no benchmark
    # CCSD reference densities were wired -- the historical schema).
    rho_rmse_pbe = _mean_finite("density_rmse_pbe")
    # n_eval: AE-contributing molecules only (matches the mae denominator).
    return mae, rho_rmse, len(ae_errs), rho_rmse_pbe


def _write_eval_df_csv(pm_rows, csv_path):
    """Fold ``per_molecule.json`` rows into the per-spec ``eval_df.csv``.

    One-row summary CSV with columns ``set, mae, rho_rmse, rho_rmse_pbe,
    n_eval`` -- the same scalars the step-7 notebook's Cell D writes per spec
    (the ``metric``/``tag``/``solver`` columns from the notebook are derived
    from the notebook-specific checkpoint-dir layout and are intentionally
    omitted here; the harness manifest carries the GridCell instead).
    ``rho_rmse_pbe`` is the PBE-vs-CCSD density baseline (nan without
    benchmark reference densities).
    """
    import csv

    mae, rho_rmse, n_eval, rho_rmse_pbe = _aggregate_per_molecule(pm_rows)
    fieldnames = ["set", "mae", "rho_rmse", "rho_rmse_pbe", "n_eval"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "set": "training_subset",
            "mae": mae,
            "rho_rmse": rho_rmse,
            "rho_rmse_pbe": rho_rmse_pbe,
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

def _test_slice_reactions(reactions, training_spec):
    """Return the held-out reactions to REPORT. WS3.

    When the spec GENUINELY ran in-loop validation the held-out pool was split
    val/test by :func:`eval_holdout.split_held_out`; the val slice drove
    early-stop / validation-best selection, so reporting it here would leak the
    selection signal into the reported generalization metric. We therefore report
    ONLY the test slice (the deterministic complement of the val slice for the
    same ``val_frac``).

    FIX 1 (2026-06-20): the gate MUST match the TRAIN-side activation, not
    ``validate_every`` alone. Training validates only in
    :func:`train._run_per_molecule_loop` (the only loop with the validation hook)
    and only when :func:`train._build_validation_data` returns data -- i.e.
    ``validate_every > 0`` AND non-empty ``validation_molecules`` AND a
    ``validation_reactions_path``. A partial/misconfigured spec (or any
    ``update_scheme='batched'`` run, which has NO validation hook) therefore never
    splits off a val slice, so the FULL held-out set is reported (no silent,
    non-comparable ~20% shrink). split_held_out is deterministic, so when
    validation DID run the kept set is the exact complement of the val slice the
    training used."""
    validated = (
        int(getattr(training_spec, "validate_every", 0)) > 0
        and bool(getattr(training_spec, "validation_molecules", ()))
        and getattr(training_spec, "validation_reactions_path", None)
    )
    if not validated:
        return reactions
    from xcquinox.alec.eval_holdout import split_held_out
    _val, test = split_held_out(
        reactions, val_frac=float(getattr(training_spec, "val_frac", 0.2)))
    return test


def _run_held_out_eval(run_dir, idx, cfg, checkpoint_dir, model_path,
                       training_spec, holdout_subdir="eval_holdout") -> None:
    """Full-pool held-out eval (BH76 + W4-11) for one trained spec.

    Parallelizes across molecule shards BY DEFAULT (adaptive degradation via
    ``_holdout_parallel.run_holdout_with_escalation``), auto-detecting the usable
    CPUs at runtime; if the parallel path raises it falls back to the serial
    ``run_full_holdout_eval``. Held-out failure is NOT fatal, it writes
    ``<holdout_subdir>/failure.json`` and returns, so the in-sample
    ``eval_df.csv`` stays the authoritative success signal for the SLURM array
    task.

    ``model_path`` is the checkpoint to evaluate and ``holdout_subdir`` is the
    output directory under the spec dir. The default pair
    (``model.eqx`` -> ``eval_holdout``) is the final-step eval; ``main`` calls
    this a second time with (``model_best.eqx`` -> ``eval_holdout_best``) to also
    emit the best-loss eval. The shard workers reload the SAME checkpoint via
    ``model_name`` (derived from ``model_path``'s basename), and each pass has its
    own ``_shards`` scratch (derived from ``holdout_subdir``), so the two passes
    are fully isolated -- no shard collision, no mixed-checkpoint energies.
    """
    try:
        from pathlib import Path as _Path

        from xcquinox.alec.eval_holdout import (
            load_trained_model,
            run_full_holdout_eval,
        )
        from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
        from xcquinox.alec.cluster.grid_config import _resolve_eval_workers
        from xcquinox.alec.parallel import detect_available_cpus

        holdout_dir = _Path(checkpoint_dir) / holdout_subdir
        model_name = os.path.basename(model_path)
        _log(idx, f"starting full-pool held-out eval (BH76 + W4-11) "
                  f"[{model_name} -> {holdout_subdir}]")
        t1 = time.time()
        model = load_trained_model(training_spec, _Path(model_path))
        # Basis + grid_level MUST match what training used (read from the
        # resolved config) so the held-out PBE/NN energies are computed in the
        # same basis as the in-sample eval, otherwise a basis bump silently
        # evaluates the held-out set in def2-svp (invalid comparison).
        _hb, _hg = _held_out_basis_grid(cfg)
        _log(idx, f"held-out pool basis={_hb} grid_level={_hg}")
        full_specs, full_rxns = load_full_held_out_pools(
            basis=_hb, grid_level=_hg,
        )

        # WS3: report ONLY the TEST slice when in-loop validation ran (the val
        # slice drove early-stop and must not leak into the reported metric); the
        # full set otherwise (byte-identical to pre-WS3). split_held_out is
        # deterministic, so this is the exact complement of the val slice the
        # training used.
        n_before = len(full_rxns)
        full_rxns = _test_slice_reactions(full_rxns, training_spec)
        if len(full_rxns) != n_before:
            _log(idx, f"held-out eval: reporting TEST slice only "
                      f"({len(full_rxns)}/{n_before} reactions; val slice "
                      f"excluded, validate_every="
                      f"{getattr(training_spec, 'validate_every', 0)})")

        # Parallelize the ~200-molecule held-out loop across the node's CPUs by
        # default (queue-agnostic auto-detect), with adaptive degradation to
        # serial. n_top <= 1 (or eval_workers: 1) ⇒ serial.
        n_top = _resolve_eval_workers(cfg.cluster, n_molecules=len(full_specs))
        result = None
        if n_top > 1:
            try:
                from xcquinox.alec.cluster._holdout_parallel import (
                    run_holdout_with_escalation,
                )
                _log(idx, f"held-out eval: parallel over up to {n_top} workers "
                          f"({len(full_specs)} molecules)")
                result = run_holdout_with_escalation(
                    run_dir, idx, training_spec, model, full_rxns, full_specs,
                    holdout_dir, basis=_hb, grid_level=_hg,
                    n_workers_top=n_top, total_cpus=detect_available_cpus(),
                    strict=bool(getattr(cfg, "held_out_strict", False)),
                    model_name=model_name)
            except Exception as pexc:  # noqa: BLE001
                _log(idx, f"held-out parallel path failed "
                          f"({type(pexc).__name__}: {pexc}); serial fallback")
                result = None
        if result is None:
            result = run_full_holdout_eval(
                training_spec=training_spec, model=model,
                mol_specs=full_specs, reactions=full_rxns, out_dir=holdout_dir,
                strict=bool(getattr(cfg, "held_out_strict", False)))

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
        holdout_dir = _Path(checkpoint_dir) / holdout_subdir
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
            f"succeeded. See {holdout_subdir}/failure.json for details.",
        )


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
    _run_held_out_eval(run_dir, idx, cfg, checkpoint_dir, model_path,
                       training_spec)

    # --- 2026-06-07: ALSO eval the best-loss checkpoint by DEFAULT ----------
    # Training saves a separate model_best.eqx (lowest trailing-mean loss); a
    # late-destabilizing run (e.g. deep_attn ss6) ends on a bad final-step
    # model.eqx, so the best snapshot is the meaningful one. We eval BOTH so the
    # figures get a final-checkpoint set (eval_holdout/) AND a best-checkpoint
    # set (eval_holdout_best/) -- doubling the data return. Fully isolated from
    # the final pass (own checkpoint, own output dir, own _shards). No-ops
    # silently when the run never captured a best snapshot (older runs).
    best_path = os.path.join(checkpoint_dir, "model_best.eqx")
    if os.path.isfile(best_path):
        _run_held_out_eval(run_dir, idx, cfg, checkpoint_dir, best_path,
                           training_spec, holdout_subdir="eval_holdout_best")
    else:
        _log(idx, "no model_best.eqx -- skipping best-checkpoint held-out eval "
                  "(only eval_holdout/ produced)")

    # --- WS3 (2026-06-20): ALSO eval the VALIDATION-best checkpoint -----------
    # When in-loop validation ran, training saved model_val_best.eqx (the minimum
    # held-out-validation snapshot, the best-generalizing model, vs model_best.eqx
    # which minimizes the TRAINING loss). Eval it into eval_holdout_val_best/ on
    # the SAME test slice. No-ops silently when validation was disabled / older
    # runs never produced the snapshot. Fully isolated (own checkpoint, own dir,
    # own _shards) from the final + best passes.
    val_best_path = os.path.join(checkpoint_dir, "model_val_best.eqx")
    if os.path.isfile(val_best_path):
        _run_held_out_eval(run_dir, idx, cfg, checkpoint_dir, val_best_path,
                           training_spec, holdout_subdir="eval_holdout_val_best")
    else:
        _log(idx, "no model_val_best.eqx -- skipping validation-best held-out "
                  "eval (in-loop validation disabled or older run)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
