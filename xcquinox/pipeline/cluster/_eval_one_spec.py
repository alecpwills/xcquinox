"""xcquinox.pipeline.cluster._eval_one_spec -- per-SLURM eval-array-task worker.

The eval-array sbatch template invokes this once per array task as::

    python -m xcquinox.pipeline.cluster._eval_one_spec <RUN_DIR> <SLURM_ARRAY_TASK_ID>

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
    :func:`xcquinox.pipeline.run_test`, and folds the per-molecule results into
    ``<checkpoint_dir>/eval_df.csv``.

JAX routing
-----------
Like :mod:`xcquinox.pipeline._train_one_spec`, this worker sets
``JAX_ENABLE_X64=1`` (and ``JAX_PLATFORMS=cpu`` -- evaluation is CPU-only) in
``os.environ`` BEFORE any ``import jax`` / before importing
``xcquinox.pipeline.evaluation``. fp32 vs fp64 changes ``density_rmse`` /
``total_energy``, so this must not be left to JAX's float32 default.

Serialization
-------------
The spec file is a frozen-dataclass :class:`TrainingSpec` written by the
materialize layer. It is loaded through the checkpoint-class loader, as
``_train_one_spec._load_spec`` loads it, which maps the class paths of
specs pickled before the subpackage's rename -- the file is produced and
consumed by the same trusted codebase, never read from an untrusted source.
"""
import argparse
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

    The spec is read through the checkpoint-class loader, which maps the
    class paths of specs pickled before the subpackage's rename; the file is
    produced and consumed by this codebase in the same process tree.
    """
    from xcquinox.pipeline.checkpoint_class import load_pickle
    with open(path, "rb") as f:
        return load_pickle(f)


# ---------------------------------------------------------------------------
# Evaluation seam
# ---------------------------------------------------------------------------

def _run_eval(test_spec):
    """Run :func:`xcquinox.pipeline.run_test` on ``test_spec`` -- the test seam.

    Isolated as a named function so a unit test can monkeypatch it and avoid
    real evaluation compute. ``run_test`` writes ``per_molecule.json`` /
    ``aggregate.json`` / ``test_metadata.json`` into ``test_spec.output_dir``.
    """
    import xcquinox.pipeline as pipeline  # noqa: E402 -- imported after JAX routing
    return pipeline.run_test(test_spec)


# ---------------------------------------------------------------------------
# per_molecule.json -> eval_df.csv fold
# ---------------------------------------------------------------------------

def _aggregate_per_molecule(pm_rows, ae_key="AE_error_kcalmol",
                            rho_key="density_rmse"):
    """Aggregate ``per_molecule.json`` rows into wide-form scalars.

    Faithful port of the step-7 notebook helper (Cell D of
    ``notebooks/_build_gga_training_dfs_subsets_notebook.py``). The per-molecule row keys it
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
    # PBE-vs-CCSD baseline density error (model-free): nan when the records
    # carry no reference density, or a reference without its own PBE density
    # (the training-side OEP references carry none; the baseline is never the
    # locally recomputed twin).
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

def _apply_species_slice(idx, full_specs, full_rxns, holdout_dir):
    """Restrict the held-out pool to the environment's species slice.

    Returns ``(mol_specs, reactions, slice_names)``; ``slice_names`` is None
    when no slice is named, in which case the pool is returned untouched and
    the channel carries no mark -- the full 216-reaction BH76 + W4-11 pool
    (214 species, measured 2026-08-20) stays the default. ``holdout_dir`` is
    the channel directory as a :class:`pathlib.Path`, not a string: the mark
    is written through ``mkdir`` and the ``/`` operator, and the directory is
    created here if the evaluation has not yet made it.

    A sliced channel is marked TWICE. ``sliced_eval.json`` is written here,
    before any energy is computed, so an interrupted or failed sliced
    evaluation is still unmistakable; ``eval_metadata.json`` carries the same
    slice after the evaluation. The figure layer refuses a channel bearing
    either mark, because a slice covers a handful of species chosen for a
    workflow test and its MAE is not the pool MAE the architectures are
    compared on. The counts in ``sliced_eval.json`` are the species slice's
    own; ``n_reactions`` in ``eval_metadata.json`` is what was evaluated,
    i.e. after :func:`_test_slice_reactions` has also dropped the validation
    complement, so the two reaction counts differ for a spec that validated.
    ``n_species`` agrees between the two files: that filter drops reactions
    only, never species.

    The sliced containers are new objects (``slice_held_out_pools`` rebuilds
    the species dict and the reaction list); the reaction dicts and
    MoleculeSpec values inside them are the pool loaders' cached objects, so
    nothing here writes into them.
    """
    from xcquinox.pipeline.full_benchmark_pools import (
        HELDOUT_SPECIES_SLICE_ENV, resolve_species_slice,
        slice_held_out_pools)
    names = resolve_species_slice()
    if names is None:
        return full_specs, full_rxns, None
    sliced_specs, sliced_rxns = slice_held_out_pools(
        full_specs, full_rxns, names)
    holdout_dir.mkdir(parents=True, exist_ok=True)
    with open(holdout_dir / "sliced_eval.json", "w") as f:
        json.dump({
            "species_slice": list(names),
            "n_species": len(sliced_specs),
            "n_reactions": len(sliced_rxns),
            "env_var": HELDOUT_SPECIES_SLICE_ENV,
        }, f, indent=2, sort_keys=True)
        f.write("\n")
    _log(idx, f"held-out eval SLICED by {HELDOUT_SPECIES_SLICE_ENV} to "
              f"{len(sliced_specs)} species / {len(sliced_rxns)} reactions "
              f"({', '.join(names)}) -- this channel is NOT the full pool")
    return sliced_specs, sliced_rxns, names


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
    from xcquinox.pipeline.eval_holdout import (reaction_identity_key,
                                            split_held_out)
    # The RECORDED val slice governs when present: the staged
    # validation/val_reactions.json is what training's early-stop actually
    # consumed, so re-evals of existing runs keep their historical partition
    # even after the split hash changed keys. Exclusion is by PHYSICAL
    # identity, not name, so a pool duplicate of a val barrier under a
    # permuted-reactant name (four BH76 entries) is excluded with it --
    # validation-best selection saw that barrier regardless of its name.
    val_path = getattr(training_spec, "validation_reactions_path", None)
    if val_path and os.path.isfile(str(val_path)):
        try:
            with open(str(val_path)) as f:
                val_rxns = json.load(f)
            val_ids = {reaction_identity_key(r) for r in val_rxns}
            if val_ids:
                return [r for r in reactions
                        if reaction_identity_key(r) not in val_ids]
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    _val, test = split_held_out(
        reactions, val_frac=float(getattr(training_spec, "val_frac", 0.2)))
    return test


#: What one held-out pass writes into its channel directory: the three tables
#: and the stamp on success, the slice mark when a slice is named, the failure
#: record on an exception. A channel directory holds the output of the last
#: pass that ENDED. When a pass starts it removes the earlier
#: pass's failure record and slice mark (it rewrites the mark itself, before
#: any energy is computed, when it slices); when it fails it removes the
#: tables and the stamp before writing its own ``failure.json``, so a
#: re-evaluation that fails cannot leave the tables it was meant to replace
#: standing as an evaluated cell, while its own slice mark stays, as the mark
#: of a failed sliced evaluation must; a pass that succeeds overwrites the
#: tables and the stamp and leaves no earlier ``failure.json`` beside them.
#: The tables are never removed when a pass starts: a pass killed mid-way by
#: the scheduler leaves the earlier tables and stamp as they were (the stamp
#: carries the slice, so a sliced channel stays refused), since hours of a
#: valid evaluation must not be lost to a kill of its replacement. The
#: ``_shards`` scratch is not listed: the parallel driver consumes only the
#: shard files its own invocation writes.
_CLEARED_AT_START = ("failure.json", "sliced_eval.json")
_CLEARED_ON_FAILURE = ("test_set.csv", "per_molecule.json",
                       "per_reaction.json", "eval_metadata.json")
PASS_OUTPUTS = _CLEARED_ON_FAILURE + _CLEARED_AT_START


def _clear_pass_outputs(holdout_dir, names) -> None:
    """Remove the named outputs of an earlier pass from ``holdout_dir`` (a
    channel directory that does not exist yet is left alone)."""
    for name in names:
        path = os.path.join(str(holdout_dir), name)
        if os.path.isfile(path):
            os.unlink(path)


def _run_held_out_eval(run_dir, idx, cfg, checkpoint_dir, model_path,
                       training_spec, holdout_subdir="eval_holdout",
                       channel=None) -> None:
    """Full-pool held-out eval over the run's held-out pools for one trained
    spec.

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

    ``channel`` names the solver override a channel pass runs under
    (``"coldstart"`` / ``"converged"``, :data:`eval_holdout.CHANNEL_OVERRIDES`;
    None for the warm passes): the caller has already replaced the spec's
    solver, and the name travels to the shard workers, which reload the spec
    themselves, and into the channel's ``eval_metadata.json`` stamp.
    """
    try:
        from pathlib import Path as _Path

        from xcquinox.pipeline.eval_holdout import (
            load_trained_model,
            run_full_holdout_eval,
        )
        from xcquinox.pipeline.full_benchmark_pools import (
            load_held_out_pools_with_conflicts)
        from xcquinox.pipeline.cluster.grid_config import _resolve_eval_workers
        from xcquinox.pipeline.parallel import detect_available_cpus

        holdout_dir = _Path(checkpoint_dir) / holdout_subdir
        # an earlier pass's failure record and slice mark do not outlive this
        # pass's start; its tables do, until this pass ends (PASS_OUTPUTS)
        _clear_pass_outputs(holdout_dir, _CLEARED_AT_START)
        model_name = os.path.basename(model_path)
        # the pools the configuration names; the benchmark pair for a
        # configuration written before the selection existed
        pools = tuple(getattr(getattr(cfg, "inputs", None), "held_out_pools",
                              ("bh76", "w411")))
        _log(idx, f"starting full-pool held-out eval ({', '.join(pools)}) "
                  f"[{model_name} -> {holdout_subdir}]")
        t1 = time.time()
        model = load_trained_model(training_spec, _Path(model_path))
        # Basis + grid_level MUST match what training used (read from the
        # resolved config) so the held-out PBE/NN energies are computed in the
        # same basis as the in-sample eval, otherwise a basis bump silently
        # evaluates the held-out set in def2-svp (invalid comparison).
        _hb, _hg = _held_out_basis_grid(cfg)
        _log(idx, f"held-out pool basis={_hb} grid_level={_hg}")
        full_specs, full_rxns, _conflicts = load_held_out_pools_with_conflicts(
            pools, basis=_hb, grid_level=_hg,
        )
        if _conflicts:
            _log(idx, f"held-out pools: {len(_conflicts)} species names "
                      "carried by two pools with different geometries, the "
                      "first pool's kept: "
                      + ", ".join(f"{c['name']} ({c['kept']} over "
                                  f"{c['dropped']})" for c in _conflicts))
        n_pool = len(full_rxns)
        full_specs, full_rxns, _slice_names = _apply_species_slice(
            idx, full_specs, full_rxns, holdout_dir)

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

        # An empty reaction set has no MAE. The filter above runs AFTER the
        # species slice and can take everything the slice left (a slice
        # closing few reactions, all of them recorded in the val slice), so
        # the emptiness is checked here, before any energy is computed, and
        # the channel is failed rather than stamped over an average of
        # nothing. A pool that arrived empty is a different fault and is left
        # to the loader.
        if n_pool and not full_rxns:
            _slice_desc = (", ".join(_slice_names) if _slice_names
                           else "no slice")
            raise RuntimeError(
                f"held-out channel {holdout_subdir} for spec {idx} would "
                f"average NO reactions: the {n_pool}-reaction pool reduced "
                f"to {n_before} under the species slice ({_slice_desc}) and "
                f"to 0 under the validation-complement filter "
                f"(validate_every="
                f"{getattr(training_spec, 'validate_every', 0)}). An empty "
                "reaction set has no MAE.")

        # Parallelize the ~200-molecule held-out loop across the node's CPUs by
        # default (queue-agnostic auto-detect), with adaptive degradation to
        # serial. n_top <= 1 (or eval_workers: 1) => serial.
        n_top = _resolve_eval_workers(cfg.cluster, n_molecules=len(full_specs))
        result = None
        if n_top > 1:
            try:
                from xcquinox.pipeline.cluster._holdout_parallel import (
                    run_holdout_with_escalation,
                )
                _log(idx, f"held-out eval: parallel over up to {n_top} workers "
                          f"({len(full_specs)} molecules)")
                result = run_holdout_with_escalation(
                    run_dir, idx, training_spec, model, full_rxns, full_specs,
                    holdout_dir, basis=_hb, grid_level=_hg,
                    n_workers_top=n_top, total_cpus=detect_available_cpus(),
                    strict=bool(getattr(cfg, "held_out_strict", False)),
                    model_name=model_name, channel=channel, pools=pools)
            except Exception as pexc:  # noqa: BLE001
                _log(idx, f"held-out parallel path failed "
                          f"({type(pexc).__name__}: {pexc}); serial fallback")
                result = None
        if result is None:
            result = run_full_holdout_eval(
                training_spec=training_spec, model=model,
                mol_specs=full_specs, reactions=full_rxns, out_dir=holdout_dir,
                strict=bool(getattr(cfg, "held_out_strict", False)))

        # Channel provenance stamp: per-row columns alone cannot distinguish
        # a cold-start pass from a capped warm one (both report cycles_run at
        # the cap), so each channel records the solver it actually ran.
        try:
            _sc = getattr(training_spec, "solver_config", None)
            holdout_dir.mkdir(parents=True, exist_ok=True)
            with open(holdout_dir / "eval_metadata.json", "w") as f:
                json.dump({
                    "channel": holdout_subdir,
                    "model": model_name,
                    # the historical boolean, derived from the override name
                    "coldstart": channel == "coldstart",
                    "channel_override": channel,
                    "solver_config": (_sc.describe()
                                      if _sc is not None else None),
                    # None for the full pool. A list names the species the
                    # channel actually covers, so a sliced channel cannot be
                    # read as a full-pool one.
                    "species_slice": (list(_slice_names)
                                      if _slice_names else None),
                    "n_species": len(full_specs),
                    "n_reactions": len(full_rxns),
                }, f, indent=2, sort_keys=True)
        except Exception as _mexc:  # noqa: BLE001
            _log(idx, f"eval_metadata.json write failed ({_mexc}); non-fatal")

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
        # the failed pass's record stands with its slice mark alone: no table
        # of an earlier pass (or a partial one of this pass) beside it reads
        # as an evaluated cell
        _clear_pass_outputs(holdout_dir, _CLEARED_ON_FAILURE)
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
    # already imported and are jax-free; the xcquinox.pipeline import below (in the
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
        # WS6: distinguish a genuinely-never-trained spec from one whose
        # training is still IN PROGRESS. A mid-run per_molecule task leaves a
        # resume checkpoint (resume_state.pkl) and no model.eqx (and no
        # completion.json) -- mirrors train._has_resume_checkpoint. In that
        # case eval was scheduled before training finished/resumed, so we emit a
        # CLEAR "training incomplete (resume in progress)" message rather than a
        # confusing crash, and still skip cleanly (exit 0): a later resubmit
        # will continue training and re-run this eval.
        resume_in_progress = (
            os.path.isfile(os.path.join(checkpoint_dir, "resume_state.pkl"))
            and not os.path.isfile(
                os.path.join(checkpoint_dir, "completion.json"))
        )
        if resume_in_progress:
            reason = ("no model.eqx -- training incomplete (resume in "
                      "progress: resume_state.pkl present, no model.eqx). "
                      "Eval was scheduled before training finished; a "
                      "`resubmit` will continue it and re-run this eval.")
        else:
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
    from xcquinox.pipeline.cluster.grid_config import load_grid_config
    from xcquinox.pipeline.cluster.domain import get_domain_profile
    from xcquinox.pipeline.cluster.spec_builder import build_test_spec

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
    # signal for the SLURM array task). The channel directories and the
    # checkpoint files of every pass below are the held-out channel
    # vocabulary's (imported here, after the JAX routing above).
    from xcquinox.pipeline.holdout_channels import (
        CHANNEL_BEST, CHANNEL_COLDSTART, CHANNEL_COLDSTART_VAL_BEST,
        CHANNEL_CONVERGED, CHANNEL_CONVERGED_VAL_BEST, CHANNEL_VAL_BEST,
        MODEL_BEST, MODEL_VAL_BEST, OVERRIDE_COLDSTART, OVERRIDE_CONVERGED)
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
    best_path = os.path.join(checkpoint_dir, MODEL_BEST)
    if os.path.isfile(best_path):
        _run_held_out_eval(run_dir, idx, cfg, checkpoint_dir, best_path,
                           training_spec, holdout_subdir=CHANNEL_BEST)
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
    val_best_path = os.path.join(checkpoint_dir, MODEL_VAL_BEST)
    if os.path.isfile(val_best_path):
        _run_held_out_eval(run_dir, idx, cfg, checkpoint_dir, val_best_path,
                           training_spec, holdout_subdir=CHANNEL_VAL_BEST)
    else:
        _log(idx, "no model_val_best.eqx -- skipping validation-best held-out "
                  "eval (in-loop validation disabled or older run)")

    # --- OPTIONAL cold-start channel pair (eval_coldstart: true) -------------
    # The final checkpoint under the cold-start override and, when the
    # validation-best checkpoint exists, that checkpoint too: the latter is
    # the reporting channel, the channel a campaign's numbers are read from.
    # The spec's solver is REPLACED HERE, before dispatch, so the in-process
    # serial-leftover tier and the serial fallback inherit the override; the
    # shard workers apply the SAME shared helper via --coldstart (they reload
    # the spec pickle themselves). Only FULL-mode specs qualify (the override
    # is undefined for one-shot protocols).
    if bool(getattr(cfg, "eval_coldstart", False)):
        _sc = getattr(training_spec, "solver_config", None)
        if _sc is not None and getattr(getattr(_sc, "mode", None),
                                       "value", None) == "full":
            import dataclasses as _dc

            from xcquinox.pipeline.eval_holdout import coldstart_solver_config
            cold_spec = _dc.replace(
                training_spec, solver_config=coldstart_solver_config(_sc))
            _run_held_out_eval(run_dir, idx, cfg, checkpoint_dir, model_path,
                               cold_spec,
                               holdout_subdir=CHANNEL_COLDSTART,
                               channel=OVERRIDE_COLDSTART)
            if os.path.isfile(val_best_path):
                _run_held_out_eval(
                    run_dir, idx, cfg, checkpoint_dir, val_best_path,
                    cold_spec,
                    holdout_subdir=CHANNEL_COLDSTART_VAL_BEST,
                    channel=OVERRIDE_COLDSTART)
            else:
                _log(idx, "no model_val_best.eqx -- cold-start channel on the "
                          "final checkpoint only")
        else:
            _log(idx, "eval_coldstart requested but the spec has no FULL-mode "
                      "solver_config -- skipping the cold-start channel")

    # --- OPTIONAL converged-SCF channel pair (eval_converged: true) ----------
    # The FINAL checkpoint under a CONVERGED SCF (pyscfad backend, PBE seed,
    # DIIS, 100 cycles at 1e-8 Ha) and, when the val-best checkpoint exists,
    # that checkpoint too. The spec's solver is replaced HERE, before
    # dispatch, exactly as for the cold-start channel; the shard workers apply
    # the same shared override via --channel converged.
    if bool(getattr(cfg, "eval_converged", False)):
        _sc = getattr(training_spec, "solver_config", None)
        if _sc is not None and getattr(getattr(_sc, "mode", None),
                                       "value", None) == "full":
            import dataclasses as _dc

            from xcquinox.pipeline.eval_holdout import converged_solver_config
            conv_spec = _dc.replace(
                training_spec, solver_config=converged_solver_config(_sc))
            _run_held_out_eval(run_dir, idx, cfg, checkpoint_dir, model_path,
                               conv_spec,
                               holdout_subdir=CHANNEL_CONVERGED,
                               channel=OVERRIDE_CONVERGED)
            if os.path.isfile(val_best_path):
                _run_held_out_eval(
                    run_dir, idx, cfg, checkpoint_dir, val_best_path,
                    conv_spec,
                    holdout_subdir=CHANNEL_CONVERGED_VAL_BEST,
                    channel=OVERRIDE_CONVERGED)
            else:
                _log(idx, "no model_val_best.eqx -- converged channel on the "
                          "final checkpoint only")
        else:
            _log(idx, "eval_converged requested but the spec has no FULL-mode "
                      "solver_config -- skipping the converged channel")

    return 0


if __name__ == "__main__":
    # The stage's verdict is the status this process hands SLURM, and
    # JAX's atexit teardown can abort the interpreter AFTER main() has
    # returned it (cluster job 2134455: the pretrain worker logged
    # "pretrain SUCCEEDED" and then died in glibc's "corrupted size vs.
    # prev_size", rc -6, so the stage read as FAILED and the dependent
    # array never ran). run_and_exit flushes and leaves through os._exit,
    # so the status is the verdict. See xcquinox/pipeline/cluster/_exit.py.
    # Imported HERE rather than in the module body: several of these
    # modules pin what their import pulls in (``fidelity`` is held to a
    # whitelist of cheap readers so the on-node gates can read a
    # certificate without the training stack), and the helper is needed
    # only when the module is RUN.
    from xcquinox.pipeline.cluster._exit import run_and_exit
    run_and_exit(main)
