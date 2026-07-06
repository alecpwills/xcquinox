#!/usr/bin/env python
"""Local held-out test-set re-evaluation for cluster-trained networks.

**Why this exists.** The cluster eval array (`_eval_one_spec.py:282 →
spec_builder.py:545-556`) evaluates every trained network only on the
molecules it was trained on. The grid-config schema has no field to specify
a held-out pool. So the `eval_df.csv` rows on disk (labeled
``set=training_subset``) are an in-sample-fit metric, not a generalization
estimate. This script closes that gap locally: load a trained
``model.eqx``, build a held-out pool by SUBTRACTING the training molecules
from BH76 + W4-11 (default), evaluate the network on the held-out pool,
and write a real ``test_set`` MAE alongside the cluster's outputs so the
existing figure script can pick them up.

Outputs (one set per requested spec):

  - ``<run_dir>/checkpoints/spec_<NNNN>/local_test_set.csv``
    Rows: one per pool (``bh76``, ``w411``) plus a combined row
    (``held_out_combined``). Columns: ``set, mae_kcalmol, n_reactions,
    n_dropped_overlap, note``.
  - ``<run_dir>/checkpoints/spec_<NNNN>/eval/local_per_molecule.json``
    Schema-compatible with the cluster's
    ``eval/per_molecule.json`` so ``make_cluster_pulls_figure.py``'s
    :func:`collect_per_molecule_rows` can consume it without changes; adds
    a ``from_training_subset: bool`` flag so future plotters can split
    train vs test.

Usage::

    # 0. Pull the candidate checkpoint(s) first (see runbook §10.5):
    python -m xcquinox.alec.cluster pull latest \\
        --category alpha_on/runs --profile full --specs 0,1,21

    # 1. Run the local re-eval:
    python notebooks/analysis/local_reeval.py \\
        ~/Documents/Research/xcquinox-results/runs/alpha_on/runs/run_<UTC>Z \\
        --specs 0,1,21

By default this runs in **loose mode**: every BH76 and W4-11 reaction is
kept and any in-sample overlap is flagged in the output metadata.
Rationale:

  - H is in every training set as a Dick regularization anchor (not as a
    substantively learned target). Dropping every BH76 reaction because of
    H overlap would discard the entire pool -- see
    ``constraint_pretraining_gmtkn55_report.md``.
  - When a molecule like H2O is in the training set, evaluating its
    atomization energy AE(H2O) = E(H2O) − 2·E(H) − E(O) is a meaningful
    test of whether the model learned the right *atomization*, not just
    the total energy -- so we WANT to compute it.

Pass ``--strict`` to opt into the old behavior where any reaction
with a training-set species is dropped. The output ``note`` column
records the overlap either way so a downstream consumer can filter.
"""
from __future__ import annotations

# JAX fp64 + CPU env BEFORE any jax-backed import. Matches the convention
# used by notebooks/analysis/multimode_constraint_eval.py.
import os

os.environ["JAX_ENABLE_X64"] = "1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# 2026-05-29 refactor: the pure-function helpers + PBE-precompute / NN-eval
# primitives + JSON/CSV writers were lifted to ``xcquinox.alec.eval_holdout``
# so the cluster eval task can import them too. Re-exported here so existing
# call sites (tests, downstream notebooks, the rest of this module) keep
# working byte-identically.
from xcquinox.alec.eval_holdout import (  # noqa: E402
    KCAL_PER_HA,
    load_training_spec,
    held_out_pool_names,
    reaction_overlap,
    filter_reactions,
    per_reaction_errors,
    reaction_mae_kcalmol,
    make_per_molecule_record,
    make_per_reaction_records,
    precompute_holdout,
    evaluate_holdout,
    arch_polarized_flag,
    load_trained_model,
)
from xcquinox.alec.eval_holdout import (  # noqa: E402
    write_test_set_csv as _write_test_set_csv,
    write_per_molecule_json as _write_per_molecule_json,
    write_per_reaction_json as _write_per_reaction_json,
)


#: Available held-out pools. Keyed by the lowercase token the user passes to
#: ``--pools``.
_AVAILABLE_POOLS = ("bh76", "w411")


# ---------------------------------------------------------------------------
# Local-CLI-specific writer wrappers
# ---------------------------------------------------------------------------
# The cluster eval writes to ``<ckpt>/eval_holdout/{test_set,per_molecule,
# per_reaction}.{csv,json}``; the local CLI keeps the historical paths
# ``<ckpt>/local_test_set.csv``, ``<ckpt>/eval/local_per_molecule.json``,
# ``<ckpt>/eval/local_per_reaction.json`` so the existing figure script
# pipeline picks them up unchanged.


def write_local_test_set_csv(
    spec_dir: Path,
    per_pool_mae: Dict[str, Tuple[float, float, int, int, int]],
    combined_mae: Tuple[float, float, int, int, int],
    strict: bool,
) -> Path:
    """Write ``<spec_dir>/local_test_set.csv`` -- see
    :func:`xcquinox.alec.eval_holdout.write_test_set_csv` for the schema."""
    return _write_test_set_csv(
        spec_dir / "local_test_set.csv",
        per_pool_mae, combined_mae, strict,
    )


def write_local_per_molecule_json(
    spec_dir: Path,
    records: List[Dict[str, Any]],
) -> Path:
    """Write the per-spec ``eval/local_per_molecule.json``."""
    return _write_per_molecule_json(
        spec_dir / "eval" / "local_per_molecule.json", records,
    )


# ---------------------------------------------------------------------------
# Side-effectful (smoke-tested only, with stubs)
# ---------------------------------------------------------------------------

def _load_demo_builders() -> Tuple[Callable, Callable]:
    """Lazy import of the demo pool builders. Indirected so the unit tests
    can monkeypatch this without paying for the demo module's pyscf
    import."""
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    demo = importlib.import_module("constraint_pretrain_gmtkn55_demo")
    return demo.build_bh76_pool, demo.build_w411_ae_pool


def _load_category_discovery() -> Callable:
    """Lazy import of ``discover_pulled_categories`` from
    ``make_cluster_pulls_figure.py``. Indirected so the unit tests can
    monkeypatch."""
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    fig = importlib.import_module("make_cluster_pulls_figure")
    return fig.discover_pulled_categories


def discover_specs_in_run(run_dir: Path) -> List[int]:
    """``[spec_index, ...]`` for every ``checkpoints/spec_<NNNN>/model.eqx``
    actually present under ``run_dir``. Specs without a model.eqx (training
    failed or in-flight) are silently skipped.

    Pure (just a glob + sort). The width is inferred from the first
    matching spec_dir name so the function works on any future width
    bump."""
    ck = run_dir / "checkpoints"
    if not ck.is_dir():
        return []
    out: List[int] = []
    for sd in sorted(ck.glob("spec_*")):
        if not sd.is_dir():
            continue
        if not (sd / "model.eqx").is_file():
            continue
        token = sd.name[len("spec_"):]
        try:
            out.append(int(token))
        except ValueError:
            continue
    return sorted(set(out))


def _default_local_root() -> str:
    return os.environ.get(
        "XCQUINOX_CLUSTER_LOCAL_ROOT",
        str(Path.home() / "Documents/Research/xcquinox-results/runs"),
    )


def load_pools(pools: Sequence[str]) -> Tuple[
    Dict[str, Any], List[Dict[str, Any]]
]:
    """Build the combined ``(mol_specs, reactions)`` for the requested pool
    names. Unknown pool tokens raise ``ValueError``."""
    bh76_builder, w411_builder = _load_demo_builders()
    mol_specs: Dict[str, Any] = {}
    reactions: List[Dict[str, Any]] = []
    for tok in pools:
        if tok not in _AVAILABLE_POOLS:
            raise ValueError(
                f"unknown pool {tok!r}, expected one of {_AVAILABLE_POOLS}")
        if tok == "bh76":
            specs, rxns = bh76_builder()
        else:
            specs, rxns = w411_builder()
        for k, v in specs.items():
            mol_specs.setdefault(k, v)
        # Annotate each reaction with its source pool for the CSV split.
        for r in rxns:
            r2 = dict(r)
            r2.setdefault("source_pool", tok)
            reactions.append(r2)
    return mol_specs, reactions


def write_local_per_reaction_json(
    spec_dir: Path,
    records: List[Dict[str, Any]],
) -> Path:
    """Write the per-spec ``eval/local_per_reaction.json`` -- one entry per
    (BH76 + W4-11) reaction, paired NN vs PBE. Consumed by the per-pool
    figure (Fig 6), the grid heatmap (Fig 7), and the per-reaction
    comparison (Fig 8). Schema: see :func:`make_per_reaction_records`."""
    return _write_per_reaction_json(
        spec_dir / "eval" / "local_per_reaction.json", records,
    )



# ---------------------------------------------------------------------------
# Per-spec orchestration
# ---------------------------------------------------------------------------

def run_one_spec(
    run_dir: Path,
    spec_idx: int,
    pools: Sequence[str],
    *,
    strict: bool = False,
    width: int = 4,
) -> Dict[str, Any]:
    """Process one (run_dir, spec_idx). Returns a summary dict including
    the per-pool MAEs and the output paths. Raises on missing files."""
    spec_name = f"spec_{spec_idx:0{width}d}"
    spec_path = run_dir / "specs" / f"{spec_name}.spec"
    spec_dir = run_dir / "checkpoints" / spec_name
    model_path = spec_dir / "model.eqx"
    if not spec_path.is_file():
        raise FileNotFoundError(f"spec file missing: {spec_path}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"model checkpoint missing: {model_path} "
            "(pull with `--profile full --specs {idx}` first)"
        )

    print(f"[spec {spec_idx}] loading {spec_path.name} + "
          f"{model_path.name} ...", flush=True)
    training_spec = load_training_spec(spec_path)
    training_names = [m.name for m in training_spec.molecules]
    model = load_trained_model(training_spec, model_path)

    print(f"[spec {spec_idx}] building {','.join(pools)} pool(s) ...",
          flush=True)
    pool_specs, all_reactions = load_pools(pools)
    held_names = held_out_pool_names(training_names, pool_specs)
    if not held_names:
        print(f"[spec {spec_idx}] WARNING: every pool species is in the "
              f"training set; held-out pool is empty.", flush=True)
    held_pool_specs = {n: pool_specs[n] for n in held_names}

    # Partition reactions per source pool so we can report per-pool MAEs.
    per_pool_kept: Dict[str, List[Dict[str, Any]]] = {p: [] for p in pools}
    per_pool_dropped: Dict[str, List[Dict[str, Any]]] = {p: [] for p in pools}
    for rxn in all_reactions:
        pool = rxn.get("source_pool", "?")
        kept_one, dropped_one = filter_reactions([rxn], training_names,
                                                  strict=strict)
        per_pool_kept.setdefault(pool, []).extend(kept_one)
        per_pool_dropped.setdefault(pool, []).extend(dropped_one)

    # Materialize the model's descriptor list so precompute_fixed_density_data
    # computes the columns the model actually consumes (e.g. dm_statistics,
    # cusp for the deep_combined_attn arch). Empty descriptors here cause
    # `jnp.concatenate` to error inside the NN forward -- see the docstring
    # of precompute_holdout.
    try:
        descriptors = tuple(training_spec.arch.materialize_descriptors())
    except AttributeError:
        descriptors = ()

    # Precompute + evaluate on EVERY pool species (not just the held-out
    # subset). Loose mode keeps reactions that touch training-set species
    # like H or H2O, and reaction_mae_kcalmol needs every species in the
    # kept reaction to have a finite energy -- so we always evaluate on
    # the full pool. The held_names list is still used downstream for the
    # per-molecule records' in-vs-out-of-training flag.
    n_to_eval = len(pool_specs)
    # 2026-05-29: when training used a full SCF solver (FULL / FIXED_J), the
    # eval has to run the same SCF -- and run_scf needs the electron-repulsion
    # integrals (eri) precomputed to rebuild J each cycle. Without 'eri' in
    # required_keys the precompute leaves md["eri"] = None and run_scf raises
    # "Cannot determine the shape of None" on every species (silent 16/16 NaN
    # drop, surfaced by the new n_dropped_nan column). For ONESHOT specs the
    # 'eri' precompute is wasted overhead; only request it when the solver
    # mode is non-ONESHOT.
    spec_solver_config = getattr(training_spec, "solver_config", None)
    needs_scf = (
        spec_solver_config is not None
        and hasattr(spec_solver_config, "mode")
        and spec_solver_config.mode.value != "oneshot"
    )
    extra_required = ("eri",) if needs_scf else ()
    print(f"[spec {spec_idx}] precomputing {n_to_eval} pool species "
          f"(training set had {len(training_names)}; "
          f"held-out pool: {len(held_names)}; "
          f"descriptors: {[type(d).__name__ for d in descriptors] or 'none'}; "
          f"extra precompute keys: {list(extra_required) or 'none'})",
          flush=True)
    mol_data = precompute_holdout(pool_specs, descriptors=descriptors,
                                   required_keys=extra_required)

    # 2026-05-29: pass training_spec.solver_config so eval matches the
    # training V_xc / density supervision domain (full_3 → 3-iter SCF;
    # one-shot stays one-shot). When the spec carries no solver_config
    # (legacy data), evaluate_holdout falls back to fixed_density_total_energy.
    mode_str = (
        spec_solver_config.mode.name if spec_solver_config is not None
        and hasattr(spec_solver_config, "mode") else "fixed_density"
    )
    print(f"[spec {spec_idx}] evaluating model on {len(mol_data)} "
          f"species (solver: {mode_str}) ...", flush=True)
    energies = evaluate_holdout(model, mol_data,
                                 solver_config=spec_solver_config)

    # Per-pool MAEs.
    # PBE baseline energies -- by-product of precompute_fixed_density_data,
    # so a direct apples-to-apples comparison on the SAME held-out reactions
    # is essentially free. ``species_energies(model=None, ...)`` in the demo
    # uses exactly this pattern.
    pbe_energies = {n: float(md.get("E_pbe"))
                    for n, md in mol_data.items()
                    if md.get("E_pbe") is not None}

    # Per-pool MAEs -- both NN and PBE on the same kept reaction set.
    # Tuple shape: (mae_nn, mae_pbe, n_used, n_dropped_overlap, n_dropped_nan).
    # n_dropped_overlap is the strict-mode drop count; n_dropped_nan counts
    # reactions silently dropped because their species energies were missing
    # / non-finite (bug found 2026-05-29).
    per_pool_mae: Dict[str, Tuple[float, float, int, int, int]] = {}
    all_kept: List[Dict[str, Any]] = []
    n_dropped_total = 0
    n_nan_total = 0
    for pool, kept in per_pool_kept.items():
        n_dropped = len(per_pool_dropped.get(pool, []))
        mae_nn, n_used, n_nan_nn = reaction_mae_kcalmol(energies, kept)
        mae_pbe, _, n_nan_pbe = reaction_mae_kcalmol(pbe_energies, kept)
        # Take the max of the two NaN-drop counts so a reaction missing in
        # EITHER channel is surfaced.
        n_nan = max(n_nan_nn, n_nan_pbe)
        per_pool_mae[pool] = (mae_nn, mae_pbe, n_used, n_dropped, n_nan)
        all_kept.extend(kept)
        n_dropped_total += n_dropped
        n_nan_total += n_nan
    combined_mae_nn, combined_n_used, combined_n_nan_nn = reaction_mae_kcalmol(
        energies, all_kept)
    combined_mae_pbe, _, combined_n_nan_pbe = reaction_mae_kcalmol(
        pbe_energies, all_kept)
    combined = (combined_mae_nn, combined_mae_pbe, combined_n_used,
                n_dropped_total, max(combined_n_nan_nn, combined_n_nan_pbe))

    # Per-molecule records -- one per pool species. The
    # `from_training_subset` flag distinguishes the held-out species (False)
    # from the species that were in the training set (True). Downstream
    # plotters can split or filter accordingly.
    training_set = set(training_names)
    records: List[Dict[str, Any]] = []
    for name in sorted(mol_data):
        records.append(make_per_molecule_record(
            name, mol_data[name], energies.get(name, float("nan")),
            in_training_subset=(name in training_set),
        ))

    # Per-reaction records -- paired NN vs PBE for every reaction in the
    # combined pool (16 = 6 BH76 + 10 W4-11), with `in_sample_overlap`
    # listing any training-set species each reaction touches. Used by the
    # per-pool / heatmap / per-reaction figures.
    nn_per_rxn = per_reaction_errors(energies, all_reactions)
    pbe_per_rxn = per_reaction_errors(pbe_energies, all_reactions)
    per_reaction_records = make_per_reaction_records(
        all_reactions, nn_per_rxn, pbe_per_rxn, training_names,
    )

    csv_path = write_local_test_set_csv(spec_dir, per_pool_mae, combined,
                                        strict)
    json_path = write_local_per_molecule_json(spec_dir, records)
    reaction_json_path = write_local_per_reaction_json(
        spec_dir, per_reaction_records)

    print(f"[spec {spec_idx}] wrote {csv_path.name}, "
          f"{json_path.parent.name}/{json_path.name}", flush=True)
    for pool, (mae_nn, mae_pbe, n_used, n_dropped, n_nan) in per_pool_mae.items():
        nn_s = (f"{mae_nn:7.3f}" if math.isfinite(mae_nn) else "    NaN")
        pbe_s = (f"{mae_pbe:7.3f}" if math.isfinite(mae_pbe) else "    NaN")
        delta_s = (f"{mae_nn - mae_pbe:+7.3f}"
                   if math.isfinite(mae_nn) and math.isfinite(mae_pbe)
                   else "    NaN")
        nan_str = f", {n_nan} NaN-drop" if n_nan else ""
        print(f"    {pool:5s}  NN={nn_s}  PBE={pbe_s}  "
              f"NN-PBE={delta_s} kcal/mol  ({n_used} rxn, "
              f"{n_dropped} overlap-drop{nan_str})",
              flush=True)
    if math.isfinite(combined_mae_nn):
        print(f"    comb.  NN={combined_mae_nn:7.3f}  "
              f"PBE={combined_mae_pbe:7.3f}  "
              f"NN-PBE={combined_mae_nn - combined_mae_pbe:+7.3f} kcal/mol  "
              f"({combined_n_used} reactions)", flush=True)
    return {
        "idx": spec_idx, "spec_dir": spec_dir,
        "per_pool_mae": per_pool_mae, "combined_mae": combined,
        "csv": csv_path, "json": json_path,
        "n_training": len(training_names),
        "n_held_out": len(held_names),
    }


# ---------------------------------------------------------------------------
# Multi-category auto-discovery driver
# ---------------------------------------------------------------------------

def run_auto(
    local_root: Path,
    pools: Sequence[str],
    *,
    strict: bool = False,
    width: int = 4,
) -> Dict[str, Any]:
    """Discover every pulled category under ``local_root`` and run
    :func:`run_one_spec` on every spec whose ``model.eqx`` is present.

    Per-spec failures are logged and counted; they do NOT abort the batch.
    The PBE precompute cache in ``xcquinox.alec.data`` amortizes across all
    specs within a category (~24 species × 1-10 s = ~1-2 min once per
    category; the per-spec NN forward eval that follows is fast).

    Returns a summary dict: ``{category: {"n_specs": ..., "n_ok": ..., "n_failed":
    ..., "failed_specs": [(idx, reason), ...]}}``.
    """
    discover = _load_category_discovery()
    cats = discover(local_root)
    if not cats:
        print(f"no run_<UTC>Z dirs found under {local_root}", file=sys.stderr)
        return {}
    print(f"--auto: found {len(cats)} categories under {local_root}",
          flush=True)
    for cat, rd in cats.items():
        print(f"  {cat or '(root)'} -> {rd.name}", flush=True)

    summary: Dict[str, Any] = {}
    grand_total = 0
    grand_ok = 0
    t0_overall = time.time()
    for cat, run_dir in cats.items():
        cat_label = cat or "(root)"
        spec_indices = discover_specs_in_run(run_dir)
        if not spec_indices:
            print(f"\n=== {cat_label}: skipped (no model.eqx files; "
                  "manifest may not be materialized yet) ===", flush=True)
            summary[cat] = {"n_specs": 0, "n_ok": 0, "n_failed": 0,
                            "failed_specs": []}
            continue
        print(f"\n=== {cat_label}: {len(spec_indices)} spec(s) "
              "(model.eqx present) ===", flush=True)
        n_ok = 0
        failed: List[Tuple[int, str]] = []
        t0_cat = time.time()
        for idx in spec_indices:
            try:
                run_one_spec(run_dir, idx, pools, strict=strict, width=width)
                n_ok += 1
            except FileNotFoundError as exc:
                msg = str(exc)
                failed.append((idx, msg))
                print(f"[spec {idx}] FAILED: {msg}", file=sys.stderr,
                      flush=True)
            except Exception as exc:  # noqa: BLE001 -- batch resilience
                msg = f"{type(exc).__name__}: {exc}"
                failed.append((idx, msg))
                print(f"[spec {idx}] FAILED: {msg}", file=sys.stderr,
                      flush=True)
        elapsed_cat = time.time() - t0_cat
        print(f"--- {cat_label}: {n_ok}/{len(spec_indices)} succeeded "
              f"({len(failed)} failed) in {elapsed_cat:.0f}s", flush=True)
        summary[cat] = {"n_specs": len(spec_indices), "n_ok": n_ok,
                        "n_failed": len(failed), "failed_specs": failed}
        grand_total += len(spec_indices)
        grand_ok += n_ok

    elapsed = time.time() - t0_overall
    print(f"\n=== --auto: {grand_ok}/{grand_total} specs succeeded across "
          f"{len(cats)} categories in {elapsed:.0f}s ===", flush=True)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # run_dir is positional-optional so --auto can be used standalone.
    p.add_argument("run_dir", type=Path, nargs="?", default=None,
                   help="locally-staged run dir (the one a `pull` populated). "
                        "Omit when using --auto.")
    p.add_argument(
        "--specs", default=None,
        help="comma-separated spec indices to re-evaluate, e.g. '0,1,21'. "
             "Required when run_dir is given; rejected when --auto is set.")
    p.add_argument(
        "--auto", action="store_true",
        help="auto-discover every pulled category under --local-root and run "
             "every spec whose model.eqx is present. Skips categories with no "
             "model.eqx (e.g. those still being trained on the cluster). "
             "Per-spec failures do not abort the batch.")
    p.add_argument(
        "--local-root", default=_default_local_root(),
        help=f"root holding categories with run_<UTC>Z dirs (default: "
             "$XCQUINOX_CLUSTER_LOCAL_ROOT else "
             "~/Documents/Research/xcquinox-results/runs). Only used with "
             "--auto.")
    p.add_argument(
        "--pools", default="bh76,w411",
        help="comma-separated held-out pools (default: bh76,w411; available: "
             f"{','.join(_AVAILABLE_POOLS)})")
    p.add_argument(
        "--strict", action="store_true",
        help="Drop every reaction whose species set overlaps the training "
             "subset (e.g. H, H2O). Default is loose mode: keep all "
             "reactions and record the overlap in the output 'note' field. "
             "Rationale for the loose default: H is in every training set "
             "as a Dick regularization anchor (not a substantively learned "
             "target); the H2O atomization energy IS the verification we "
             "want when H2O is in the training set; dropping all reactions "
             "with H overlap would discard the entire BH76 pool.")
    p.add_argument(
        "--width", type=int, default=4,
        help="zero-pad width of spec_NNNN dir names (default 4; the harness "
             "uses 4 today)")
    args = p.parse_args(argv)

    pools = tuple(t.strip() for t in args.pools.split(",") if t.strip())
    for tok in pools:
        if tok not in _AVAILABLE_POOLS:
            print(f"unknown pool {tok!r}, expected one of "
                  f"{_AVAILABLE_POOLS}", file=sys.stderr)
            return 1

    # --auto and the explicit run_dir + --specs path are mutually exclusive.
    if args.auto:
        if args.run_dir is not None or args.specs is not None:
            print("--auto is incompatible with run_dir / --specs "
                  "(--auto discovers both automatically)", file=sys.stderr)
            return 1
        local_root = Path(args.local_root).expanduser().resolve()
        if not local_root.is_dir():
            print(f"--local-root does not exist: {local_root}",
                  file=sys.stderr)
            return 1
        summary = run_auto(local_root, pools,
                           strict=args.strict,
                           width=args.width)
        # Non-zero exit only if EVERY category was empty (nothing happened).
        if not summary or all(s["n_specs"] == 0 for s in summary.values()):
            return 1
        return 0

    # Single-run-dir path.
    if args.run_dir is None or args.specs is None:
        print("either pass --auto, or both <run_dir> and --specs",
              file=sys.stderr)
        return 1
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"run_dir does not exist: {run_dir}", file=sys.stderr)
        return 1
    try:
        spec_indices = [int(t.strip()) for t in args.specs.split(",")
                        if t.strip()]
    except ValueError as exc:
        print(f"--specs entries must be integers ({exc})", file=sys.stderr)
        return 1
    if not spec_indices:
        print("--specs is empty", file=sys.stderr)
        return 1

    t0 = time.time()
    for idx in spec_indices:
        try:
            run_one_spec(run_dir, idx, pools,
                         strict=args.strict,
                         width=args.width)
        except FileNotFoundError as exc:
            print(f"[spec {idx}] {exc}", file=sys.stderr)
            continue
    print(f"done in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
