#!/usr/bin/env python
"""Publication-quality figures from the locally-staged cluster pull results.

**IMPORTANT scientific caveat.** The cluster eval task evaluates every
trained network ONLY on the molecules it was trained on
(``_eval_one_spec.py`` calls ``build_test_spec`` without
``holdout_molecules``; ``spec_builder.py:545-556`` silently defaults to
in-distribution eval and emits a ``RuntimeWarning`` to eval-job stderr that
is easily missed). Every "MAE" plotted here is therefore the **in-sample
training-subset MAE** — a measure of how well training converged, NOT a
test-set generalization estimate. Large MAE in these figures indicates
**training failure**, not poor generalization. Held-out test-set MAE is
computed by a separate local script after pulling the relevant
``model.eqx`` checkpoints with
``python -m xcquinox.alec.cluster pull <run> --profile full --specs <…>``
(see ``hpcjobs/SEAWULF_RUNBOOK.md`` §10.5).

The script walks ``<local_root>/{category…}/run_<UTC>Z/`` directories that
the harness ``pull`` workflow stages, aggregates:

  - per-spec ``eval_df.csv`` (training-subset MAE / rho_rmse / n_eval),
  - per-molecule ``eval/per_molecule.json`` (AE error, density RMSE, SCF
    convergence) joined to the spec's grid cell (arch / loss / metric /
    subset_size / solver) from ``manifest.json``,
  - per-spec ``status`` and ``final_loss`` from
    :func:`xcquinox.alec.cluster.analyze.collect_results`.

…and renders four PNGs:

  1. ``fig1_training_diagnostics.png`` — training-subset MAE vs subset_size
     (log-y) + a training-success scatter (training-subset MAE vs final
     training loss, log-log; failure zone shaded).
  2. ``fig2_per_molecule_errors.png`` — top-20 hardest molecules across all
     specs + density-RMSE-vs-AE-error scatter + per-(metric, solver)
     per-molecule |AE_error| violin.
  3. ``fig3_coverage_dashboard.png`` — (metric, solver) × subset_size
     status heatmap per category.
  4. ``fig_composite_summary.png`` — 2x2 composite tying the three together.

The script is self-contained: only stdlib (``csv``, ``json``, ``os``,
``argparse``, ``pathlib``), ``numpy`` and ``matplotlib`` — no pandas.
Style matches the existing ``make_multimode_figure.py`` template (DPI 120,
font.size 9, soft grid).

Usage:
    python notebooks/analysis/make_cluster_pulls_figure.py \
        [--local-root ~/Documents/Research/xcquinox-results/runs] \
        [--out-dir notebooks/analysis] \
        [--prefix cluster_pulls]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ---------------------------------------------------------------------------
# House style — match notebooks/analysis/make_multimode_figure.py
# ---------------------------------------------------------------------------

_STYLE = {
    "figure.dpi": 120,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
}

# Color/marker/linestyle palette — `metric` color, `solver` marker.
_METRIC_COLOR: Dict[str, str] = {
    "jsd": "#4f81bd",   # blue family (matches existing multimode plot)
    "l2":  "#c0504d",   # red family (matches existing multimode plot)
}
_SOLVER_MARKER: Dict[str, str] = {
    "full_3":  "o",     # circle = full SCF
    "oneshot": "s",     # square = non-self-consistent
}
_STATUS_COLOR: Dict[str, str] = {
    # Dashboard heatmap colors — green = ready to plot, lighter shades = in
    # flight, red = trouble, gray = nothing here yet.
    "complete":         "#1e7d36",   # dark green
    "trained_no_eval":  "#a4d4ae",   # light green
    "train_failed":     "#b22222",   # firebrick
    "eval_skipped":     "#d4a017",   # gold
    "pending":          "#dddddd",   # light gray
    "missing":          "#ffffff",   # absent from manifest
}
_STATUS_ORDER = ("complete", "trained_no_eval", "eval_skipped", "train_failed",
                 "pending", "missing")

_CATEGORY_COLOR: Dict[str, str] = {
    # Per-category color palette used by every cross-category figure. New
    # categories that aren't listed here fall through to _CATEGORY_FALLBACKS
    # via _palette_for() — guarantees a stable color even before we hard-code
    # one (e.g. when polarized/alpha_on/runs lands).
    "alpha_on/runs":            "#c0504d",
    "alpha_off/runs":           "#4f81bd",
    "polarized/alpha_off/runs": "#4daf4a",
    "polarized/alpha_on/runs":  "#7a5fa3",
}
_CATEGORY_FALLBACKS = ("#e07b39", "#5b8a3a", "#8e4a8c", "#2e8b8b", "#888888")


def _palette_for(categories) -> Dict[str, str]:
    """Stable category → color mapping. Known categories use _CATEGORY_COLOR;
    unknown ones get the next color from _CATEGORY_FALLBACKS in iteration
    order (deterministic across a single script run)."""
    out: Dict[str, str] = {}
    fb_iter = iter(_CATEGORY_FALLBACKS)
    for cat in categories:
        if cat in _CATEGORY_COLOR:
            out[cat] = _CATEGORY_COLOR[cat]
        else:
            try:
                out[cat] = next(fb_iter)
            except StopIteration:
                out[cat] = "#888888"
    return out


# Stable axis orders for the dashboard.
_SUBSET_SIZES = (1, 2, 3, 4, 5, 6, 7, 12, 15, 18)
_METRIC_SOLVER_PAIRS = (
    ("jsd", "full_3"), ("jsd", "oneshot"), ("l2", "full_3"), ("l2", "oneshot"),
)

_RUN_ID_RE = re.compile(r"^run_\d{8}T\d{6}Z$")


# ---------------------------------------------------------------------------
# Data ingest — pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def discover_pulled_categories(local_root: Path) -> Dict[str, Path]:
    """Find every ``run_<UTC>Z`` under ``local_root``, grouped by category.

    Returns ``{category_relpath: latest_run_dir}``. ``category_relpath`` is
    POSIX-style relative to ``local_root`` and matches the ``--category``
    string that the harness ``pull`` subcommand uses (e.g. ``alpha_off/runs``,
    ``polarized/alpha_on/runs``). Picks the latest run by lexicographic stamp
    sort (zero-padded ISO timestamps sort chronologically).

    A category with no ``run_*Z`` subdir is omitted entirely. A category whose
    sole run-dir is missing ``manifest.json`` is still included — the
    dashboard will mark it ``missing`` so the operator can see it pulled but
    didn't have a materialized manifest yet.
    """
    if not local_root.is_dir():
        return {}
    out: Dict[str, Tuple[str, Path]] = {}
    for run_dir in local_root.rglob("run_*"):
        if not run_dir.is_dir():
            continue
        if not _RUN_ID_RE.match(run_dir.name):
            continue
        rel_parent = run_dir.parent.relative_to(local_root).as_posix()
        if rel_parent == ".":
            rel_parent = ""
        prev = out.get(rel_parent)
        if prev is None or run_dir.name > prev[0]:
            out[rel_parent] = (run_dir.name, run_dir)
    return {cat: rd for cat, (_, rd) in out.items()}


def _read_manifest_cells(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    """``manifest.json`` -> ``{spec_index: {arch, loss, metric, subset_size,
    solver}}``. Empty dict when the manifest is missing or malformed."""
    mpath = run_dir / "manifest.json"
    if not mpath.is_file():
        return {}
    try:
        with mpath.open() as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    cells: Dict[int, Dict[str, Any]] = {}
    for entry in manifest.get("specs", []):
        idx = entry.get("index")
        cell = entry.get("cell") or {}
        if isinstance(idx, int):
            cells[idx] = dict(cell)
    return cells


def _spec_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    """``[(spec_index, spec_dir)]`` sorted by index."""
    ck = run_dir / "checkpoints"
    if not ck.is_dir():
        return []
    out: List[Tuple[int, Path]] = []
    for p in ck.iterdir():
        if not p.is_dir() or not p.name.startswith("spec_"):
            continue
        try:
            idx = int(p.name.split("_", 1)[1])
        except ValueError:
            continue
        out.append((idx, p))
    out.sort(key=lambda t: t[0])
    return out


def collect_eval_df_rows(run_dir: Path) -> List[Dict[str, Any]]:
    """Read every ``checkpoints/spec_*/eval_df.csv`` and join with its
    manifest cell. One output row per CSV data row (one CSV row per "set" —
    on this harness only ``training_subset`` is written today).

    Output schema (each row): ``idx, arch, loss, metric, subset_size, solver,
    set, mae, rho_rmse, n_eval``. Specs without a CSV are silently skipped
    (the dashboard surfaces them via :func:`aggregate_status_grid`).
    """
    cells = _read_manifest_cells(run_dir)
    rows: List[Dict[str, Any]] = []
    for idx, spec_dir in _spec_dirs(run_dir):
        csv_path = spec_dir / "eval_df.csv"
        if not csv_path.is_file():
            continue
        cell = cells.get(idx, {})
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    mae = float(r["mae"]) if r.get("mae") not in (None, "") else None
                    rho_rmse = (float(r["rho_rmse"])
                                if r.get("rho_rmse") not in (None, "") else None)
                    n_eval = int(r["n_eval"]) if r.get("n_eval") not in (None, "") else None
                except (TypeError, ValueError):
                    continue
                rows.append({
                    "idx": idx,
                    "arch": cell.get("arch"),
                    "loss": cell.get("loss"),
                    "metric": cell.get("metric"),
                    "subset_size": cell.get("subset_size"),
                    "solver": cell.get("solver"),
                    "set": r.get("set"),
                    "mae": mae,
                    "rho_rmse": rho_rmse,
                    "n_eval": n_eval,
                })
    return rows


def collect_per_molecule_rows(run_dir: Path) -> List[Dict[str, Any]]:
    """Read every ``checkpoints/spec_*/eval/per_molecule.json``, join with
    the manifest cell, and return one row per (spec, molecule).

    Output schema: ``idx, arch, loss, metric, subset_size, solver, molecule,
    AE_error_kcalmol, density_rmse, density_l1, skipped, scf_converged``.
    Atoms (``skipped=true, skip_reason=atomic_system``) are kept — they
    contribute density-tracking diagnostics but typically lack AE_error.
    """
    cells = _read_manifest_cells(run_dir)
    rows: List[Dict[str, Any]] = []
    for idx, spec_dir in _spec_dirs(run_dir):
        pm_path = spec_dir / "eval" / "per_molecule.json"
        if not pm_path.is_file():
            continue
        try:
            with pm_path.open() as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        cell = cells.get(idx, {})
        for r in payload:
            rows.append({
                "idx": idx,
                "arch": cell.get("arch"),
                "loss": cell.get("loss"),
                "metric": cell.get("metric"),
                "subset_size": cell.get("subset_size"),
                "solver": cell.get("solver"),
                "molecule": r.get("molecule"),
                "AE_error_kcalmol": r.get("AE_error_kcalmol"),
                "density_rmse": r.get("density_rmse"),
                "density_l1": r.get("density_l1"),
                "skipped": bool(r.get("skipped", False)),
                "scf_converged": r.get("scf_converged"),
            })
    return rows


def collect_local_test_set_rows(run_dir: Path) -> List[Dict[str, Any]]:
    """Read every ``checkpoints/spec_*/local_test_set.csv`` (written by
    ``local_reeval.py``) and join with the manifest cell. Yields one dict
    per (spec, pool ∈ {bh76, w411, held_out_combined}).

    Output schema: ``idx, arch, loss, metric, subset_size, solver, pool,
    mae_kcalmol, n_reactions, n_dropped_overlap, note``. Specs without a
    local_test_set.csv are silently skipped (they haven't been re-evaluated
    yet); the rest of the dashboard / Fig 1 left panel keep showing
    cluster training-subset numbers as the baseline.
    """
    cells = _read_manifest_cells(run_dir)
    rows: List[Dict[str, Any]] = []
    for idx, spec_dir in _spec_dirs(run_dir):
        csv_path = spec_dir / "local_test_set.csv"
        if not csv_path.is_file():
            continue
        cell = cells.get(idx, {})
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                # The CSV's "set" column is one of "test_set_bh76",
                # "test_set_w411", "test_set_held_out_combined" — strip the
                # leading "test_set_" so the downstream consumer just sees
                # the pool token.
                set_label = (r.get("set") or "")
                pool = (set_label[len("test_set_"):]
                        if set_label.startswith("test_set_") else set_label)

                def _float_or_none(k):
                    v = r.get(k)
                    if v in (None, ""):
                        return None
                    try:
                        return float(v)
                    except ValueError:
                        return None

                # Back-compat: pre-2026-05-29 CSVs had a single "mae_kcalmol"
                # column (the NN MAE only). New CSVs split it into NN and
                # PBE columns plus the delta. Read either shape.
                mae_nn = _float_or_none("mae_nn_kcalmol")
                if mae_nn is None:
                    mae_nn = _float_or_none("mae_kcalmol")
                mae_pbe = _float_or_none("mae_pbe_kcalmol")
                delta = _float_or_none("delta_nn_minus_pbe")
                try:
                    n_reactions = int(r["n_reactions"])
                    n_dropped = int(r["n_dropped_overlap"])
                except (TypeError, ValueError):
                    continue
                rows.append({
                    "idx": idx,
                    "arch": cell.get("arch"),
                    "loss": cell.get("loss"),
                    "metric": cell.get("metric"),
                    "subset_size": cell.get("subset_size"),
                    "solver": cell.get("solver"),
                    "pool": pool,
                    # Keep the old key for tests that still read mae_kcalmol;
                    # surface both NN and PBE explicitly so the figure
                    # builder can plot both.
                    "mae_kcalmol": mae_nn,
                    "mae_nn_kcalmol": mae_nn,
                    "mae_pbe_kcalmol": mae_pbe,
                    "delta_nn_minus_pbe": delta,
                    "n_reactions": n_reactions,
                    "n_dropped_overlap": n_dropped,
                    "note": r.get("note", ""),
                })
    return rows


def collect_subset_descriptor_rows(run_dir: Path) -> List[Dict[str, Any]]:
    """Read every ``checkpoints/spec_*/eval/local_subset_descriptors.json``
    (written by ``extract_subset_descriptors.py``) and join with the
    manifest cell.

    Yields one dict per spec with the per-molecule feature matrix +
    per-subset stats as numpy arrays. Specs without the file (e.g. when
    the extractor hasn't been run yet for them) are silently skipped.
    """
    cells = _read_manifest_cells(run_dir)
    rows: List[Dict[str, Any]] = []
    for idx, spec_dir in _spec_dirs(run_dir):
        sd_path = spec_dir / "eval" / "local_subset_descriptors.json"
        if not sd_path.is_file():
            continue
        try:
            with sd_path.open() as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        cell = cells.get(idx, {})
        per_mol = np.asarray(payload.get("per_molecule_features", []),
                              dtype=float)
        rows.append({
            "idx": idx,
            "arch": cell.get("arch"),
            "loss": cell.get("loss"),
            "metric": cell.get("metric"),
            "subset_size": cell.get("subset_size"),
            "solver": cell.get("solver"),
            "training_molecule_names": payload.get(
                "training_molecule_names", []),
            "feature_names": payload.get("feature_names", []),
            "per_molecule_features": per_mol,
            "per_subset_stats": payload.get("per_subset_stats", {}),
        })
    return rows


def collect_per_reaction_rows(run_dir: Path) -> List[Dict[str, Any]]:
    """Read every ``checkpoints/spec_*/eval/local_per_reaction.json`` (written
    by ``local_reeval.py`` 2026-05-29+) and join with the manifest cell.

    Yields one dict per (spec, reaction) pair carrying both the NN and PBE
    absolute errors plus the grid cell. The per-reaction figures (Fig 7
    heatmap, Fig 8 per-reaction comparison) aggregate over this. Specs
    without the JSON are silently skipped — happens when local_reeval was
    last run before this writer landed.
    """
    cells = _read_manifest_cells(run_dir)
    rows: List[Dict[str, Any]] = []
    for idx, spec_dir in _spec_dirs(run_dir):
        rj_path = spec_dir / "eval" / "local_per_reaction.json"
        if not rj_path.is_file():
            continue
        try:
            with rj_path.open() as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        cell = cells.get(idx, {})
        for r in payload:
            rows.append({
                "idx": idx,
                "arch": cell.get("arch"),
                "loss": cell.get("loss"),
                "metric": cell.get("metric"),
                "subset_size": cell.get("subset_size"),
                "solver": cell.get("solver"),
                "name": r.get("name"),
                "pool": r.get("pool"),
                "ref_kcalmol": r.get("reaction_energy_ref_kcalmol"),
                "de_nn_kcalmol": r.get("de_nn_kcalmol"),
                "de_pbe_kcalmol": r.get("de_pbe_kcalmol"),
                "error_nn_kcalmol": r.get("error_nn_kcalmol"),
                "error_pbe_kcalmol": r.get("error_pbe_kcalmol"),
                "abs_error_nn_kcalmol": r.get("abs_error_nn_kcalmol"),
                "abs_error_pbe_kcalmol": r.get("abs_error_pbe_kcalmol"),
                "in_sample_overlap": r.get("in_sample_overlap", []),
            })
    return rows


def load_final_losses(run_dir: Path) -> Dict[int, Optional[float]]:
    """``{spec_idx: final_loss}`` joined from
    :func:`xcquinox.alec.cluster.analyze.collect_results`. Used to drive the
    training-success diagnostic in Fig 1 right. Empty dict if collect_results
    cannot read the run (missing manifest, etc.)."""
    try:
        from xcquinox.alec.cluster import analyze
    except ImportError:  # pragma: no cover - defensive
        return {}
    try:
        rows = analyze.collect_results(str(run_dir))
    except (FileNotFoundError, ValueError):
        return {}
    return {int(r["idx"]): r.get("final_loss") for r in rows
            if r.get("idx") is not None}


def aggregate_status_grid(
    run_dir: Path,
) -> Dict[Tuple[str, str], Dict[int, str]]:
    """``(metric, solver) -> {subset_size: status}`` from
    :func:`xcquinox.alec.cluster.analyze.collect_results`.

    Missing-manifest run dirs return an empty dict (the dashboard caller
    handles that by painting an all-``missing`` row).
    """
    try:
        from xcquinox.alec.cluster import analyze
    except ImportError:  # pragma: no cover - defensive
        return {}
    try:
        rows = analyze.collect_results(str(run_dir))
    except (FileNotFoundError, ValueError):
        return {}
    grid: Dict[Tuple[str, str], Dict[int, str]] = {}
    for r in rows:
        key = (r.get("metric") or "?", r.get("solver") or "?")
        grid.setdefault(key, {})[r.get("subset_size")] = r.get("status", "pending")
    return grid


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _series_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (row.get("metric") or "?", row.get("solver") or "?")


def _format_count(n: int, singular: str = "run") -> str:
    return f"{n} {singular}" + ("" if n == 1 else "s")


def _completion_summary(grids: Dict[str, Dict[Tuple[str, str], Dict[int, str]]],
                        ) -> Dict[str, Tuple[int, int]]:
    """``{category: (n_complete, n_total)}`` for header annotations."""
    out: Dict[str, Tuple[int, int]] = {}
    for cat, grid in grids.items():
        n_total = sum(len(v) for v in grid.values())
        n_done = sum(1 for v in grid.values() for s in v.values() if s == "complete")
        out[cat] = (n_done, n_total)
    return out


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

_FAILURE_MAE_THRESHOLD = 5.0  # kcal/mol — heuristic "this spec did not train"
_FAILURE_LOSS_THRESHOLD = 5e-3  # final-loss threshold matching the above


def _plot_mae_vs_subset(ax, rows: List[Dict[str, Any]],
                        local_rows: Optional[List[Dict[str, Any]]] = None,
                        local_pool: str = "held_out_combined",
                        compact_legend: bool = False) -> int:
    """Draw "MAE vs subset_size" on ``ax`` with the cluster's training-subset
    numbers as solid lines.

    When ``local_rows`` is non-empty, also overlay the held-out (test_set)
    MAE from ``local_test_set.csv`` as DASHED lines at the same color/marker
    per (metric, solver). ``local_pool`` selects which row of the local CSV
    feeds the overlay — by default the ``held_out_combined`` row
    (BH76 + W4-11). Returns the number of distinct in-sample specs plotted
    so the caller can put it in the title.
    """
    complete = [r for r in rows
                if r.get("set") == "training_subset"
                and r.get("mae") is not None
                and r.get("subset_size") is not None]
    series: Dict[Tuple[str, str], List[Tuple[int, float]]] = {}
    for r in complete:
        series.setdefault(_series_key(r), []).append(
            (int(r["subset_size"]), float(r["mae"])))
    # In-sample solid lines.
    for (metric, solver), pts in sorted(series.items()):
        pts.sort(key=lambda p: p[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker=_SOLVER_MARKER.get(solver, "x"),
                color=_METRIC_COLOR.get(metric, "k"), linewidth=1.5,
                markersize=6, label=f"{metric} · {solver} (in-sample)",
                zorder=3)
    # Held-out dashed overlay — only the selected pool, only finite MAE.
    if local_rows:
        held: Dict[Tuple[str, str], List[Tuple[int, float]]] = {}
        for r in local_rows:
            if (r.get("pool") != local_pool
                or r.get("mae_kcalmol") is None
                or r.get("subset_size") is None):
                continue
            try:
                ss = int(r["subset_size"]); mae = float(r["mae_kcalmol"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(mae) or mae <= 0:
                continue
            held.setdefault(_series_key(r), []).append((ss, mae))
        for (metric, solver), pts in sorted(held.items()):
            pts.sort(key=lambda p: p[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker=_SOLVER_MARKER.get(solver, "x"),
                    color=_METRIC_COLOR.get(metric, "k"),
                    linestyle="--", linewidth=1.5, markersize=6,
                    label=f"{metric} · {solver} (held-out)",
                    zorder=4, alpha=0.85)
    # PBE-on-the-same-held-out-set reference line. Local CSVs carry the
    # PBE MAE on the exact same kept reactions (computed from mol_data
    # E_pbe), so averaging it gives the right baseline. Drawn dotted so
    # the NN-vs-PBE comparison is at-a-glance.
    if local_rows:
        pbe_vals = [r["mae_pbe_kcalmol"] for r in local_rows
                    if r.get("pool") == local_pool
                    and r.get("mae_pbe_kcalmol") is not None
                    and math.isfinite(r["mae_pbe_kcalmol"])]
        if pbe_vals:
            pbe_ref = float(np.median(pbe_vals))
            ax.axhline(pbe_ref, color="#222", linestyle=":", linewidth=1.0,
                       alpha=0.7, zorder=2,
                       label=f"PBE on held-out ({pbe_ref:.2f} kcal/mol)")
    ax.set_yscale("log")
    ax.set_xlabel("subset size (# training molecules)")
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_xticks(list(_SUBSET_SIZES))
    ax.set_xticklabels([str(s) for s in _SUBSET_SIZES])
    if series:
        if compact_legend:
            # Composite-friendly: a small upper-right legend with just
            # color (metric) and marker (solver) keys, NOT one entry per
            # (metric × solver × set). Avoids the 8-line legend collision
            # with adjacent composite panels.
            from matplotlib.lines import Line2D
            handles = []
            for m, c in _METRIC_COLOR.items():
                handles.append(Line2D([0], [0], marker="o", linestyle="",
                                      markerfacecolor=c, markeredgecolor="none",
                                      markersize=6, label=f"metric={m}"))
            for s, mk in _SOLVER_MARKER.items():
                handles.append(Line2D([0], [0], marker=mk, linestyle="",
                                      markerfacecolor="white",
                                      markeredgecolor="black",
                                      markersize=6, label=f"solver={s}"))
            if local_rows:
                handles.append(Line2D([0], [0], color="#888",
                                      linestyle="-", label="in-sample"))
                handles.append(Line2D([0], [0], color="#888",
                                      linestyle="--", label="held-out"))
            ax.legend(handles=handles, loc="upper left", framealpha=0.9,
                      fontsize=6, ncol=2)
        else:
            # Standalone: move the (potentially 8-line) legend below the
            # panel where it has horizontal room.
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                      frameon=False, fontsize=8, ncol=4,
                      handletextpad=0.5, columnspacing=1.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return len({r["idx"] for r in complete})


def _plot_training_success(ax, rows: List[Dict[str, Any]],
                           final_losses: Dict[int, Optional[float]]) -> None:
    """Scatter training-subset MAE vs final training loss on log-log axes.

    Highlights a "likely training failure" region in the upper-right
    (large MAE + large final loss). Lower-left cluster = well-converged
    specs worth re-evaluating locally on a held-out set.
    """
    pts: List[Tuple[float, float, str, str, int]] = []
    for r in rows:
        if r.get("set") != "training_subset" or r.get("mae") is None:
            continue
        idx = r.get("idx")
        if idx is None:
            continue
        fl = final_losses.get(int(idx))
        if fl is None or not math.isfinite(fl) or fl <= 0:
            continue
        mae = float(r["mae"])
        if not math.isfinite(mae) or mae <= 0:
            continue
        pts.append((fl, mae, r.get("metric") or "?", r.get("solver") or "?",
                    int(idx)))
    if not pts:
        ax.text(0.5, 0.5, "no (MAE × final_loss) data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#888")
        ax.set_title("Training-success diagnostic")
        return

    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    colors = [_METRIC_COLOR.get(p[2], "#888") for p in pts]
    markers = [_SOLVER_MARKER.get(p[3], "x") for p in pts]
    # matplotlib's scatter doesn't take per-point markers — split by marker.
    for marker in set(markers):
        sel = [i for i, m in enumerate(markers) if m == marker]
        ax.scatter(xs[sel], ys[sel],
                   c=[colors[i] for i in sel],
                   marker=marker, s=42, alpha=0.78,
                   edgecolors="black", linewidths=0.4, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("final training loss (unitless, ↓ better convergence)")
    ax.set_ylabel("training-subset MAE (kcal/mol, ↓ better fit)")
    ax.set_title("Training-success diagnostic — lower-left = well-converged")

    # Shade the "likely training failure" zone.
    xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()
    fail_x = max(_FAILURE_LOSS_THRESHOLD, xlo)
    fail_y = max(_FAILURE_MAE_THRESHOLD, ylo)
    if fail_x < xhi and fail_y < yhi:
        ax.axhspan(fail_y, yhi, xmin=(np.log10(fail_x) - np.log10(xlo))
                                    / (np.log10(xhi) - np.log10(xlo)),
                   xmax=1.0, color="#c0504d", alpha=0.10, zorder=1)
        ax.axhline(fail_y, color="#c0504d", linestyle=":", linewidth=1.0, zorder=2)
        ax.axvline(fail_x, color="#c0504d", linestyle=":", linewidth=1.0, zorder=2)
        ax.text(xhi, yhi, "  likely training\n  failure region  ",
                ha="right", va="top", fontsize=7, color="#7a2c2a",
                style="italic")
    # Custom legend.
    from matplotlib.lines import Line2D
    handles: List[Line2D] = []
    for m, c in _METRIC_COLOR.items():
        handles.append(Line2D([0], [0], marker="o", linestyle="",
                              markerfacecolor=c, markeredgecolor="black",
                              markersize=7, label=f"metric={m}"))
    for s, mk in _SOLVER_MARKER.items():
        handles.append(Line2D([0], [0], marker=mk, linestyle="",
                              markerfacecolor="white", markeredgecolor="black",
                              markersize=7, label=f"solver={s}"))
    # Lower-right used to overlap the failure-zone annotation; upper-left
    # is empty space in this log-log scatter (points cluster around
    # mid-range).
    ax.legend(handles=handles, loc="upper left", framealpha=0.9, fontsize=7,
              ncol=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_generalization(
    eval_rows_by_cat: Dict[str, List[Dict[str, Any]]],
    per_mol_rows_by_cat: Dict[str, List[Dict[str, Any]]],
    final_losses_by_cat: Dict[str, Dict[int, Optional[float]]],
    lead_label: str,
    out_path: Path,
    local_rows_by_cat: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Path:
    """Fig 1 — cluster training diagnostics, faceted by category.

    One row per category. Left panel: MAE vs subset_size (solid in-sample,
    dashed held-out overlay when local rows present). Right panel:
    training-success scatter (training-subset MAE vs final loss, log-log
    with a "likely failure" shaded zone).

    ``per_mol_rows_by_cat`` is accepted for signature parity with the
    composite builder; unused here (per-molecule view lives in Fig 2).
    """
    del per_mol_rows_by_cat  # composite-only
    cats = list(eval_rows_by_cat.keys())
    local_rows_by_cat = local_rows_by_cat or {}
    have_local = any(local_rows_by_cat.get(c) for c in cats)
    with plt.rc_context(_STYLE):
        if not cats:
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.text(0.5, 0.5, "no eval data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#888")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path
        n_rows = len(cats)
        fig, axes = plt.subplots(n_rows, 2,
                                 figsize=(13, 4.6 * n_rows + 1.2))
        if n_rows == 1:
            axes = np.array([axes])
        spec_counts: List[Tuple[str, int]] = []
        for ri, cat in enumerate(cats):
            ax_l = axes[ri, 0]
            ax_r = axes[ri, 1]
            rows = eval_rows_by_cat[cat]
            local_rows = local_rows_by_cat.get(cat) or None
            final_losses = final_losses_by_cat.get(cat, {})
            # First row gets the full series-by-series legend (covers the
            # convention for all rows); subsequent rows get the compact
            # marker/color/linestyle key legend to avoid 4 huge legend
            # blocks crowding the figure.
            n_specs = _plot_mae_vs_subset(
                ax_l, rows, local_rows=local_rows,
                compact_legend=(ri != 0))
            title_l = (
                f"{cat}  —  MAE vs subset size  (solid=in-sample, "
                "dashed=held-out)") if local_rows else (
                f"{cat}  —  Training fit vs subset size  (in-sample only)")
            ax_l.set_title(title_l, fontsize=10)
            _plot_training_success(ax_r, rows, final_losses)
            spec_counts.append((cat, n_specs))
        head = "; ".join(f"{c}: {_format_count(n, 'spec')}"
                          for c, n in spec_counts)
        fig.suptitle(
            f"Cluster training diagnostics  ({head})", fontsize=12)
        subtitle = ("Solid = cluster training-subset MAE (in-sample, "
                    "training quality only); dashed = held-out MAE "
                    "computed locally on BH76+W4-11.") if have_local else (
                    "Cluster eval is in-sample only — large MAE here "
                    "indicates training failure, not poor generalization.")
        # Position subtitle just under the suptitle (works for any n_rows).
        fig.text(0.5, 0.952, subtitle,
                 ha="center", va="top", fontsize=8, style="italic",
                 color="#444", wrap=True)
        fig.tight_layout(rect=(0, 0.04, 1, 0.93))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def _plot_per_molecule_top_n(ax, per_mol_rows: List[Dict[str, Any]],
                             top_n: int = 20) -> None:
    by_mol: Dict[str, List[float]] = {}
    for r in per_mol_rows:
        ae = r.get("AE_error_kcalmol")
        if ae is None or not math.isfinite(ae):
            continue
        by_mol.setdefault(r.get("molecule") or "?", []).append(abs(float(ae)))
    summary: List[Tuple[str, float, float, int]] = [
        (mol, float(np.mean(v)), float(np.std(v)), int(len(v)))
        for mol, v in by_mol.items()
    ]
    summary.sort(key=lambda t: t[1], reverse=True)
    summary = summary[:top_n]
    if not summary:
        ax.text(0.5, 0.5, "no per-molecule data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#888")
        return
    mols = [t[0] for t in summary]
    means = np.asarray([t[1] for t in summary])
    stds = np.asarray([t[2] for t in summary])
    ns = [t[3] for t in summary]
    y = np.arange(len(summary))
    ax.barh(y, means, xerr=stds, color="#c0504d",
            alpha=0.85, edgecolor="black",
            error_kw={"ecolor": "#444", "linewidth": 0.8})
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m} (n={n})" for m, n in zip(mols, ns)])
    ax.invert_yaxis()
    ax.set_xlabel("mean |AE error| ± std across specs (kcal/mol)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _plot_per_molecule_violin(ax, per_mol_rows: List[Dict[str, Any]]) -> None:
    """Per-(metric, solver) violin of |AE_error_kcalmol| over all specs."""
    groups: Dict[Tuple[str, str], List[float]] = {}
    for r in per_mol_rows:
        ae = r.get("AE_error_kcalmol")
        if ae is None or not math.isfinite(ae):
            continue
        groups.setdefault(_series_key(r), []).append(abs(float(ae)))
    labels: List[str] = []
    data: List[List[float]] = []
    face_colors: List[str] = []
    for (metric, solver) in _METRIC_SOLVER_PAIRS:
        vals = groups.get((metric, solver), [])
        if vals:
            labels.append(f"{metric}\n{solver}")
            data.append(vals)
            face_colors.append(_METRIC_COLOR.get(metric, "#888"))
    if not data:
        ax.text(0.5, 0.5, "no per-molecule data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#888")
        return
    positions = list(range(1, len(data) + 1))
    parts = ax.violinplot(data, positions=positions, widths=0.75,
                          showmedians=True, showextrema=False)
    for body, fc in zip(parts["bodies"], face_colors):
        body.set_facecolor(fc); body.set_alpha(0.6)
        body.set_edgecolor("black")
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("per-molecule |AE error| (kcal/mol)")
    for i, (lbl, vals) in enumerate(zip(labels, data), start=1):
        med = float(np.median(vals))
        ax.text(i, max(vals) * 1.4 if max(vals) > 0 else 1.0,
                f"n={len(vals)}\nmed={med:.2g}",
                ha="center", va="bottom", fontsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _plot_density_vs_ae(ax, per_mol_rows: List[Dict[str, Any]]) -> None:
    xs: List[float] = []; ys: List[float] = []; cs: List[str] = []
    for r in per_mol_rows:
        ae = r.get("AE_error_kcalmol")
        rrho = r.get("density_rmse")
        if (ae is None or rrho is None
                or not math.isfinite(ae) or not math.isfinite(rrho)):
            continue
        xs.append(float(rrho))
        ys.append(abs(float(ae)))
        cs.append(_METRIC_COLOR.get(r.get("metric") or "", "#888"))
    if not xs:
        ax.text(0.5, 0.5, "no (AE × density) data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#888")
        return
    ax.scatter(xs, ys, c=cs, s=18, alpha=0.55, edgecolors="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("density RMSE (a.u.)")
    ax.set_ylabel("|AE error| (kcal/mol)")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", linestyle="",
                      markerfacecolor=c, markeredgecolor="none",
                      markersize=7, label=m)
               for m, c in _METRIC_COLOR.items()]
    ax.legend(handles=handles, loc="upper left", title="metric",
              framealpha=0.9, fontsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_per_molecule(
    per_mol_rows_by_cat: Dict[str, List[Dict[str, Any]]],
    lead_label: str,
    out_path: Path,
) -> Path:
    """Fig 2 — per-species error structure, faceted by category.

    One row per category, 3 panels per row: top-20 hardest molecules
    (in-sample mean |AE error|), per-(metric, solver) violin spread, and
    a density-RMSE vs |AE error| scatter. Empty categories show a
    placeholder.
    """
    cats = list(per_mol_rows_by_cat.keys())
    with plt.rc_context(_STYLE):
        if not cats:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.text(0.5, 0.5, "no per-molecule data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#888")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path
        n_rows = len(cats)
        fig, axes = plt.subplots(n_rows, 3,
                                 figsize=(18, 5.5 * n_rows + 0.6),
                                 gridspec_kw={
                                     "width_ratios": [1.1, 1.0, 1.2]})
        if n_rows == 1:
            axes = np.array([axes])
        for ri, cat in enumerate(cats):
            per_mol_rows = per_mol_rows_by_cat[cat]
            _plot_per_molecule_top_n(axes[ri, 0], per_mol_rows, top_n=20)
            axes[ri, 0].set_title(
                f"{cat}\nTop-20 hardest molecules (in-sample AE error)",
                fontsize=9)
            _plot_per_molecule_violin(axes[ri, 1], per_mol_rows)
            axes[ri, 1].set_title(
                f"{cat}\nPer-molecule spread by (metric, solver)",
                fontsize=9)
            _plot_density_vs_ae(axes[ri, 2], per_mol_rows)
            axes[ri, 2].set_title(
                f"{cat}\nDensity vs energy error (molecule × spec)",
                fontsize=9)
        fig.suptitle(
            "Per-species error structure  (in-sample, per category)",
            fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def _status_grid_matrix(
    grid: Dict[Tuple[str, str], Dict[int, str]],
) -> np.ndarray:
    """Render a (metric, solver) × subset_size grid as a status-index matrix
    suitable for ``imshow``. Returns an int array of shape
    ``(len(pairs), len(subset_sizes))`` whose values index into
    ``_STATUS_ORDER``.
    """
    nrows, ncols = len(_METRIC_SOLVER_PAIRS), len(_SUBSET_SIZES)
    out = np.full((nrows, ncols), _STATUS_ORDER.index("missing"), dtype=int)
    for ri, key in enumerate(_METRIC_SOLVER_PAIRS):
        row = grid.get(key, {})
        for ci, ss in enumerate(_SUBSET_SIZES):
            st = row.get(ss, "missing")
            if st not in _STATUS_ORDER:
                st = "pending"
            out[ri, ci] = _STATUS_ORDER.index(st)
    return out


def plot_dashboard(grids_by_category: Dict[str, Dict[Tuple[str, str], Dict[int, str]]],
                   summary: Dict[str, Tuple[int, int]],
                   out_path: Path) -> Path:
    """Fig 3: per-category (metric, solver) × subset_size status heatmap."""
    cats = list(grids_by_category.keys()) or ["(no categories pulled)"]
    cmap = ListedColormap([_STATUS_COLOR[s] for s in _STATUS_ORDER])

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(
            len(cats), 1,
            figsize=(10, 1.5 * len(cats) + 1.0),
            squeeze=False,
        )
        axes_list = [ax for col in axes for ax in col]
        for ax, cat in zip(axes_list, cats):
            grid = grids_by_category.get(cat, {})
            mat = _status_grid_matrix(grid)
            ax.imshow(mat, aspect="auto", cmap=cmap,
                      vmin=0, vmax=len(_STATUS_ORDER) - 1)
            ax.set_xticks(range(len(_SUBSET_SIZES)))
            ax.set_xticklabels([str(s) for s in _SUBSET_SIZES])
            ax.set_yticks(range(len(_METRIC_SOLVER_PAIRS)))
            ax.set_yticklabels([f"{m}·{s}" for m, s in _METRIC_SOLVER_PAIRS])
            done, total = summary.get(cat, (0, 0))
            if total:
                ax.set_title(f"{cat}   (complete: {done}/{total})",
                             fontsize=9, loc="left")
            else:
                ax.set_title(f"{cat}   (no manifest pulled)",
                             fontsize=9, loc="left", color="#888")
            ax.set_xlabel("subset_size")
            # Cell text: a single-letter status hint inside each cell.
            for ri in range(mat.shape[0]):
                for ci in range(mat.shape[1]):
                    st = _STATUS_ORDER[int(mat[ri, ci])]
                    letter = {
                        "complete": "✓", "trained_no_eval": "·",
                        "train_failed": "✗", "eval_skipped": "s",
                        "pending": "", "missing": "",
                    }.get(st, "")
                    if letter:
                        color = "white" if st == "complete" else "#222"
                        ax.text(ci, ri, letter, ha="center", va="center",
                                fontsize=7, color=color)
            ax.set_xticks(np.arange(-.5, len(_SUBSET_SIZES), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(_METRIC_SOLVER_PAIRS), 1),
                          minor=True)
            ax.grid(which="minor", color="white", linewidth=1.0)
            ax.tick_params(which="minor", bottom=False, left=False)

        # Single shared legend along the top.
        legend_handles = [
            Patch(facecolor=_STATUS_COLOR[s], edgecolor="#444", label=s)
            for s in ("complete", "trained_no_eval", "eval_skipped",
                      "train_failed", "pending", "missing")
        ]
        fig.legend(handles=legend_handles, loc="upper center",
                   ncol=len(legend_handles), frameon=False,
                   bbox_to_anchor=(0.5, 1.0), fontsize=8)
        fig.suptitle("Cluster sweep coverage by category", fontsize=12,
                     y=1.04)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# New dedicated test-set-eval figures (Figs 5–8)
# ---------------------------------------------------------------------------

def plot_nn_vs_pbe(local_rows_by_cat: Dict[str, List[Dict[str, Any]]],
                   lead_label: str,
                   out_path: Path) -> Path:
    """Fig 5 — NN vs PBE on the combined held-out set, all categories.

    Left panel: per-(metric, solver) strip plot, each cell split into one
    sub-column per category (color = category, marker = solver). Horizontal
    PBE = 9.56 reference + shaded "NN beats PBE" half-plane. Per-cell
    annotation stacks "n_wins / total" rows colored by category.

    Right panel: NN − PBE histogram stacked by category. Δ = 0 line; per
    category legend label includes win count + median Δ.

    Fig5 vs Fig16: same data, different decomposition. Fig5 emphasizes
    (metric, solver) cells with category as a sub-dimension; Fig16
    explodes every (category × metric × solver) onto its own column.
    """
    combined_by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for cat, rows in local_rows_by_cat.items():
        c = [r for r in rows
             if r.get("pool") == "held_out_combined"
             and r.get("mae_nn_kcalmol") is not None
             and r.get("mae_pbe_kcalmol") is not None
             and math.isfinite(r["mae_nn_kcalmol"])
             and math.isfinite(r["mae_pbe_kcalmol"])]
        if c:
            combined_by_cat[cat] = c

    with plt.rc_context(_STYLE):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.5))
        if not combined_by_cat:
            for ax in (ax_l, ax_r):
                ax.text(0.5, 0.5, "no local test_set data",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=11, color="#888")
            fig.suptitle("NN vs PBE on held-out  (no data)", fontsize=12)
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path

        cats = list(combined_by_cat.keys())
        palette = _palette_for(cats)
        all_combined = [r for v in combined_by_cat.values() for r in v]
        nn_all = np.asarray([r["mae_nn_kcalmol"] for r in all_combined])
        pbe_all = np.asarray([r["mae_pbe_kcalmol"] for r in all_combined])
        deltas_all = nn_all - pbe_all
        pbe_constant = float(np.median(pbe_all))

        # ---- Panel A: per-(metric, solver) strip, sub-columns per cat. -
        cell_order = list(_METRIC_SOLVER_PAIRS)
        cell_pos = {ms: i + 1 for i, ms in enumerate(cell_order)}
        # Sub-column offset within each cell — symmetric around center.
        n_cats = len(cats)
        if n_cats <= 1:
            sub_offsets = {cats[0]: 0.0} if cats else {}
        else:
            spread = 0.28
            sub_offsets = {c: -spread + (2 * spread * i / (n_cats - 1))
                           for i, c in enumerate(cats)}
        rng = np.random.default_rng(0)
        for cat in cats:
            color = palette.get(cat, "#888")
            for r in combined_by_cat[cat]:
                key = (r.get("metric") or "?", r.get("solver") or "?")
                x0 = cell_pos.get(key)
                if x0 is None:
                    continue
                jitter = float(rng.uniform(-0.06, 0.06))
                ax_l.scatter([x0 + sub_offsets[cat] + jitter],
                             [float(r["mae_nn_kcalmol"])],
                             c=color,
                             marker=_SOLVER_MARKER.get(key[1], "x"),
                             s=38, alpha=0.82, edgecolors="black",
                             linewidths=0.3, zorder=4)
        ax_l.axhline(pbe_constant, color="#222", linestyle="--",
                     linewidth=1.0, zorder=2,
                     label=f"PBE = {pbe_constant:.2f}")
        x_min, x_max = 0.4, len(cell_order) + 0.6
        y_min = min(float(nn_all.min()) * 0.92, pbe_constant * 0.85)
        y_max = float(nn_all.max()) * 1.05
        ax_l.axhspan(y_min, pbe_constant, color="#4daf4a", alpha=0.10,
                     zorder=1)
        ax_l.text(x_max - 0.05,
                  pbe_constant - (pbe_constant - y_min) * 0.06,
                  "NN beats PBE\n(below the line)",
                  ha="right", va="top", fontsize=8, style="italic",
                  color="#2a6a2f")
        ax_l.set_xlim(x_min, x_max)
        ax_l.set_ylim(y_min, y_max)
        ax_l.set_xticks(list(cell_pos.values()))
        ax_l.set_xticklabels([f"{m}\n{s}" for m, s in cell_order],
                              fontsize=8)
        ax_l.set_ylabel("NN held-out MAE (kcal/mol)")
        ax_l.set_xlabel("loss configuration")
        ax_l.set_title("Per-spec NN MAE by configuration  "
                       "(sub-columns: category)")
        # Per-cell win counts stacked by category.
        for ms, x0 in cell_pos.items():
            lines = []
            for cat in cats:
                cell_vals = [float(r["mae_nn_kcalmol"])
                             for r in combined_by_cat[cat]
                             if (r.get("metric"), r.get("solver")) == ms]
                if not cell_vals:
                    continue
                n_win = sum(1 for v in cell_vals if v < pbe_constant)
                lines.append((cat, f"{n_win}/{len(cell_vals)}"))
            if not lines:
                continue
            for ri, (cat, lbl) in enumerate(lines):
                ax_l.text(
                    x0, y_max - (y_max - y_min) * (0.03 + 0.045 * ri),
                    lbl, ha="center", va="top", fontsize=7,
                    color=palette.get(cat, "#888"), family="monospace")
        from matplotlib.lines import Line2D
        leg = [Line2D([0], [0], marker="o", linestyle="",
                      markerfacecolor=palette.get(c, "#888"),
                      markeredgecolor="k",
                      markersize=7, label=c)
               for c in cats]
        leg += [Line2D([0], [0], marker=mk, linestyle="",
                       markerfacecolor="white", markeredgecolor="k",
                       markersize=7, label=f"solver={s}")
                for s, mk in _SOLVER_MARKER.items()]
        leg.append(Line2D([0], [0], color="#222", linestyle="--",
                          linewidth=1.0,
                          label=f"PBE = {pbe_constant:.2f}"))
        ax_l.legend(handles=leg, loc="center right", framealpha=0.9,
                    fontsize=7, ncol=1, bbox_to_anchor=(0.99, 0.55))
        for sp in ("top", "right"):
            ax_l.spines[sp].set_visible(False)

        # ---- Panel B: NN−PBE histogram stacked by category. ------------
        bins = np.linspace(min(deltas_all.min(), -1.0) - 1,
                           max(deltas_all.max(), 1.0) + 1, 22)
        bottom = np.zeros(len(bins) - 1)
        headline_parts: List[str] = []
        for cat in cats:
            d = np.asarray([r["mae_nn_kcalmol"] - r["mae_pbe_kcalmol"]
                            for r in combined_by_cat[cat]])
            counts, _ = np.histogram(d, bins=bins)
            n_win = int(np.sum(d < 0))
            n_total = int(len(d))
            med = float(np.median(d))
            ax_r.bar(bins[:-1], counts, width=np.diff(bins),
                     bottom=bottom, color=palette.get(cat, "#888"),
                     edgecolor="black", linewidth=0.4, align="edge",
                     alpha=0.78,
                     label=f"{cat}: {n_win}/{n_total}  "
                           f"(median Δ = {med:+.2f})")
            bottom = bottom + counts
            headline_parts.append(f"{cat} {n_win}/{n_total}")
        ax_r.axvline(0, color="#222", linestyle="--", linewidth=1.0,
                     label="Δ = 0  (= PBE)")
        ax_r.set_xlabel("NN − PBE  (kcal/mol; <0 = NN beats PBE)")
        ax_r.set_ylabel("# specs")
        ax_r.set_title("Distribution of NN − PBE on combined held-out")
        # Annotation box moved to UPPER-LEFT (used to overlap the tallest
        # bin at upper-center on the May-29 figures).
        # Best / worst across ALL combined.
        best_delta = float(np.min(deltas_all))
        worst_delta = float(np.max(deltas_all))
        best_idx = int(all_combined[int(np.argmin(deltas_all))]["idx"])
        worst_idx = int(all_combined[int(np.argmax(deltas_all))]["idx"])
        ax_r.text(0.02, 0.98,
                  f"best  Δ = {best_delta:+.2f}  (spec_{best_idx:04d})\n"
                  f"worst Δ = {worst_delta:+.2f}  (spec_{worst_idx:04d})",
                  transform=ax_r.transAxes, ha="left", va="top",
                  fontsize=8, family="monospace",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                            edgecolor="#bbb", alpha=0.92),
                  zorder=5)
        ax_r.legend(loc="upper right", framealpha=0.9, fontsize=7)
        for sp in ("top", "right"):
            ax_r.spines[sp].set_visible(False)

        fig.suptitle(
            "Held-out NN vs PBE  ("
            + ", ".join(headline_parts) + " beat PBE)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path


_POOL_PBE_REF = {
    "bh76": 8.077,    # PBE MAE on the 6 BH76RC Probe-C subset reactions
    "w411": 10.450,   # PBE MAE on the 10 W4-11 closed-shell atomizations
}


def plot_per_pool(local_rows_by_cat: Dict[str, List[Dict[str, Any]]],
                  lead_label: str,
                  out_path: Path) -> Path:
    """Fig 6 — per-pool breakdown, faceted by category.

    Layout: 2 rows (BH76 / W4-11) × N category columns. Each subplot shows
    per-(metric, solver) line plots of NN MAE vs subset_size, with the PBE
    MAE for that pool as a horizontal dotted reference (BH76 = 8.08,
    W4-11 = 10.45 — same constant across all specs).
    """
    cats = list(local_rows_by_cat.keys())
    with plt.rc_context(_STYLE):
        if not cats:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "no local test_set data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#888")
            fig.suptitle("Per-benchmark held-out picture  (no data)")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path
        n_cols = max(1, len(cats))
        fig, axes = plt.subplots(2, n_cols,
                                 figsize=(5.5 * n_cols + 1.0, 8.0),
                                 sharey="row")
        if n_cols == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        for col_i, cat in enumerate(cats):
            local_rows = local_rows_by_cat[cat]
            for row_i, (pool_key, pool_label) in enumerate((
                ("bh76", "BH76 reaction energies (6 reactions)"),
                ("w411", "W4-11 atomizations (10 reactions)"),
            )):
                ax = axes[row_i, col_i]
                series: Dict[Tuple[str, str], List[Tuple[int, float]]] = {}
                for r in local_rows:
                    if (r.get("pool") != pool_key
                        or r.get("mae_nn_kcalmol") is None
                        or r.get("subset_size") is None):
                        continue
                    try:
                        ss = int(r["subset_size"])
                        mae = float(r["mae_nn_kcalmol"])
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(mae) or mae <= 0:
                        continue
                    series.setdefault(_series_key(r), []).append((ss, mae))
                for (metric, solver), pts in sorted(series.items()):
                    pts.sort(key=lambda p: p[0])
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    ax.plot(xs, ys,
                            marker=_SOLVER_MARKER.get(solver, "x"),
                            color=_METRIC_COLOR.get(metric, "k"),
                            linewidth=1.5, markersize=6,
                            label=f"{metric} · {solver}", zorder=3)
                pbe_ref = _POOL_PBE_REF[pool_key]
                ax.axhline(pbe_ref, color="#222", linestyle=":",
                           linewidth=1.2, alpha=0.85, zorder=2,
                           label=f"PBE = {pbe_ref:.2f}")
                ax.set_yscale("log")
                if row_i == 1:
                    ax.set_xlabel("subset size (# training molecules)")
                if col_i == 0:
                    ax.set_ylabel("NN held-out MAE (kcal/mol)")
                ax.set_xticks(list(_SUBSET_SIZES))
                ax.set_xticklabels([str(s) for s in _SUBSET_SIZES],
                                   fontsize=7)
                # Subplot title combines category + pool only on the top
                # row; bottom row gets pool-only since it's clearer.
                if row_i == 0:
                    ax.set_title(f"{cat}\n{pool_label}", fontsize=9)
                else:
                    ax.set_title(pool_label, fontsize=9)
                if row_i == 0 and col_i == 0:
                    ax.legend(loc="upper left", framealpha=0.9,
                              fontsize=7)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
        fig.suptitle("Per-benchmark held-out picture", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_grid_heatmap(
    local_rows_by_cat: Dict[str, List[Dict[str, Any]]],
    lead_label: str,
    out_path: Path,
) -> Path:
    """Fig 7 — NN−PBE delta heatmap, faceted by category.

    Each category becomes a column of 4 mini-heatmaps (one per (metric,
    solver) cell). Shared symmetric color scale across all categories so
    "alpha_on vs polarized" is visually directly comparable.
    """
    cats = list(local_rows_by_cat.keys())
    # cells_by_cat: {cat: {(metric, solver): {subset_size: delta}}}
    cells_by_cat: Dict[str, Dict[Tuple[str, str], Dict[int, float]]] = {}
    for cat in cats:
        cells_data: Dict[Tuple[str, str], Dict[int, float]] = {}
        for r in local_rows_by_cat[cat]:
            if (r.get("pool") != "held_out_combined"
                or r.get("delta_nn_minus_pbe") is None):
                continue
            d = r.get("delta_nn_minus_pbe")
            if d is None or not math.isfinite(d):
                continue
            try:
                ss = int(r["subset_size"])
            except (TypeError, ValueError):
                continue
            cells_data.setdefault(_series_key(r), {})[ss] = float(d)
        if cells_data:
            cells_by_cat[cat] = cells_data
    cats = [c for c in cats if c in cells_by_cat]
    with plt.rc_context(_STYLE):
        if not cats:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.text(0.5, 0.5, "no local NN−PBE delta data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#888")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path
        ms_pairs = [("jsd", "full_3"), ("jsd", "oneshot"),
                    ("l2", "full_3"), ("l2", "oneshot")]
        n_cols = len(cats)
        fig, axes = plt.subplots(4, n_cols,
                                 figsize=(4.0 * n_cols + 1.4, 8.5),
                                 sharex=True)
        if n_cols == 1:
            axes = axes.reshape(4, 1)
        all_vals = [v for cell_data in cells_by_cat.values()
                    for grid in cell_data.values()
                    for v in grid.values()]
        vmax = max(abs(min(all_vals)), abs(max(all_vals))) if all_vals else 1.0
        vmax = max(vmax, 1.0)
        cmap = plt.get_cmap("RdYlGn_r")
        im = None
        for ci, cat in enumerate(cats):
            cells_data = cells_by_cat[cat]
            for ri, ms in enumerate(ms_pairs):
                ax = axes[ri, ci]
                grid = cells_data.get(ms, {})
                row = np.full((1, len(_SUBSET_SIZES)), np.nan)
                for sli, ss in enumerate(_SUBSET_SIZES):
                    if ss in grid:
                        row[0, sli] = grid[ss]
                im = ax.imshow(row, aspect="auto", cmap=cmap,
                               vmin=-vmax, vmax=vmax)
                ax.set_xticks(range(len(_SUBSET_SIZES)))
                ax.set_xticklabels([str(s) for s in _SUBSET_SIZES],
                                   fontsize=7)
                ax.set_yticks([0])
                ax.set_yticklabels([f"{ms[0]}·{ms[1]}"], fontsize=7)
                title = f"metric={ms[0]}, solver={ms[1]}"
                if ri == 0:
                    title = f"{cat}\n{title}"
                ax.set_title(title, fontsize=8)
                for sli in range(row.shape[1]):
                    v = row[0, sli]
                    if not math.isfinite(v):
                        continue
                    fc = "white" if abs(v) > vmax * 0.55 else "#222"
                    ax.text(sli, 0, f"{v:+.1f}", ha="center", va="center",
                            fontsize=6, color=fc)
                if ri == 3:
                    ax.set_xlabel("subset_size", fontsize=8)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
        if im is not None:
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7,
                                pad=0.02, fraction=0.022)
            cbar.set_label("NN − PBE  (kcal/mol;  <0 = NN beats PBE)",
                           fontsize=9)
        fig.suptitle("NN − PBE delta  (green = NN beats PBE, "
                     "red = NN worse;  shared scale)", fontsize=12)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path


def plot_per_reaction(
    per_rxn_rows_by_cat: Dict[str, List[Dict[str, Any]]],
    lead_label: str,
    out_path: Path,
) -> Path:
    """Fig 8 — per-reaction NN vs PBE, faceted by category.

    One row per category × 2 columns (paired-bars left, ranked-delta
    right). Each row uses the same reaction ordering (BH76 first, then
    W4-11, alphabetical within). BH76/W4-11 are labeled ABOVE the axes
    instead of inside the bars (the prior layout put the labels in the
    bar group and clipped them). Rotation reduced to 30° with right-align
    to give the reaction names breathing room.
    """
    cats = list(per_rxn_rows_by_cat.keys())
    with plt.rc_context(_STYLE):
        if not cats:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.text(0.5, 0.5, "no per-reaction data\n"
                    "(re-run local_reeval.py --auto)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#888")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path
        n_rows = len(cats)
        fig, axes = plt.subplots(
            n_rows, 2,
            figsize=(15, 5.5 * n_rows + 0.6),
            gridspec_kw={"width_ratios": [1.2, 1.0]})
        if n_rows == 1:
            axes = np.array([axes])
        for ri, cat in enumerate(cats):
            ax_l = axes[ri, 0]
            ax_r = axes[ri, 1]
            per_rxn_rows = per_rxn_rows_by_cat[cat]
            by_rxn: Dict[str, Dict[str, List[float]]] = {}
            rxn_pool: Dict[str, str] = {}
            for r in per_rxn_rows:
                name = r.get("name")
                if not name:
                    continue
                nn = r.get("abs_error_nn_kcalmol")
                pbe = r.get("abs_error_pbe_kcalmol")
                if nn is None or pbe is None:
                    continue
                if not (math.isfinite(nn) and math.isfinite(pbe)):
                    continue
                d = by_rxn.setdefault(name, {"nn": [], "pbe": []})
                d["nn"].append(float(nn))
                d["pbe"].append(float(pbe))
                rxn_pool.setdefault(name, r.get("pool", "?"))
            if not by_rxn:
                for ax in (ax_l, ax_r):
                    ax.text(0.5, 0.5, f"{cat}: no per-reaction data",
                            ha="center", va="center",
                            transform=ax.transAxes,
                            fontsize=10, color="#888")
                continue
            order = sorted(by_rxn.keys(),
                           key=lambda n: (0 if rxn_pool.get(n) == "bh76"
                                          else 1, n))
            nn_med = np.asarray([np.median(by_rxn[n]["nn"]) for n in order])
            pbe_med = np.asarray([np.median(by_rxn[n]["pbe"]) for n in order])
            deltas = nn_med - pbe_med
            x = np.arange(len(order))
            w = 0.38
            ax_l.bar(x - w / 2, nn_med, w, color="#c0504d",
                     edgecolor="black", linewidth=0.4,
                     label="NN (median |error|)")
            ax_l.bar(x + w / 2, pbe_med, w, color="#777777",
                     edgecolor="black", linewidth=0.4,
                     label="PBE (median |error|)")
            ax_l.set_xticks(x)
            ax_l.set_xticklabels(order, rotation=30, ha="right",
                                 fontsize=7)
            ax_l.set_ylabel("median |reaction error|  (kcal/mol)")
            n_bh76 = sum(1 for n in order if rxn_pool.get(n) == "bh76")
            if 0 < n_bh76 < len(order):
                ax_l.axvline(n_bh76 - 0.5, color="#bbb",
                             linewidth=0.8, linestyle=":")
                # Place pool labels ABOVE the axes (not inside the bars)
                # to fix the May-29 overlap.
                ax_l.text(n_bh76 / 2 - 0.5, 1.02, "BH76",
                          transform=ax_l.get_xaxis_transform(),
                          ha="center", va="bottom", fontsize=8,
                          style="italic", color="#555")
                ax_l.text(n_bh76 + (len(order) - n_bh76) / 2 - 0.5, 1.02,
                          "W4-11",
                          transform=ax_l.get_xaxis_transform(),
                          ha="center", va="bottom", fontsize=8,
                          style="italic", color="#555")
            ax_l.set_title(
                f"{cat}  —  per-reaction median |error| (NN vs PBE)  "
                f"across {len({r['idx'] for r in per_rxn_rows})} specs",
                fontsize=9)
            ax_l.legend(loc="upper right", framealpha=0.9, fontsize=8)
            for sp in ("top", "right"):
                ax_l.spines[sp].set_visible(False)
            order_b = [order[i] for i in np.argsort(deltas)]
            deltas_b = deltas[np.argsort(deltas)]
            colors_b = ["#4daf4a" if d < 0 else "#c0504d"
                        for d in deltas_b]
            ax_r.barh(np.arange(len(order_b)), deltas_b, color=colors_b,
                      edgecolor="black", linewidth=0.4)
            ax_r.axvline(0, color="#222", linestyle="--", linewidth=0.8)
            ax_r.set_yticks(np.arange(len(order_b)))
            ax_r.set_yticklabels(order_b, fontsize=7)
            ax_r.invert_yaxis()
            ax_r.set_xlabel(
                "median (NN − PBE)  (kcal/mol;  <0 = NN beats PBE)")
            ax_r.set_title(
                f"{cat}  —  ranked delta (NN beats PBE on top)",
                fontsize=9)
            n_win = int(np.sum(deltas < 0))
            ax_r.text(0.98, 0.02,
                      f"NN beats PBE on {n_win} / {len(order)} reactions",
                      transform=ax_r.transAxes, ha="right", va="bottom",
                      fontsize=8, style="italic",
                      bbox=dict(boxstyle="round,pad=0.3",
                                facecolor="white",
                                edgecolor="#bbb", alpha=0.9))
            for sp in ("top", "right"):
                ax_r.spines[sp].set_visible(False)
        fig.suptitle("Per-reaction NN vs PBE on held-out", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Quick-win figures (9, 11, 12, 13, 14)
# ---------------------------------------------------------------------------

def plot_subset_size_correlation(
    local_rows_by_cat: Dict[str, List[Dict[str, Any]]],
    lead_label: str,
    out_path: Path,
) -> Path:
    """Fig 9 — held-out NN MAE vs subset_size, faceted by category.

    One row per category; each row carries the boxplot + jittered scatter
    + per-(metric, solver) Spearman annotation. Shared y-axis. Tells the
    PI whether "more training data helps on held-out" within EACH cluster
    sweep.
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        spearmanr = None  # type: ignore[assignment]

    cats: List[str] = []
    combined_by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for cat, rows in local_rows_by_cat.items():
        c = [r for r in rows
             if r.get("pool") == "held_out_combined"
             and r.get("mae_nn_kcalmol") is not None
             and r.get("subset_size") is not None
             and math.isfinite(r.get("mae_nn_kcalmol", float("nan")))]
        if c:
            cats.append(cat)
            combined_by_cat[cat] = c

    with plt.rc_context(_STYLE):
        if not cats:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.text(0.5, 0.5, "no local test_set data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#888")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path
        n_rows = len(cats)
        fig, axes = plt.subplots(n_rows, 1,
                                 figsize=(13, 3.0 + 2.6 * n_rows),
                                 sharey=True)
        if n_rows == 1:
            axes = [axes]
        # Shared PBE reference across categories (same 16 reactions).
        all_pbe = [r["mae_pbe_kcalmol"] for combined in combined_by_cat.values()
                   for r in combined
                   if r.get("mae_pbe_kcalmol") is not None
                   and math.isfinite(r["mae_pbe_kcalmol"])]
        pbe_ref = float(np.median(all_pbe)) if all_pbe else 9.56
        for ri, cat in enumerate(cats):
            ax = axes[ri]
            combined = combined_by_cat[cat]
            by_size: Dict[int, List[float]] = {}
            for r in combined:
                ss = int(r["subset_size"])
                mae = float(r["mae_nn_kcalmol"])
                by_size.setdefault(ss, []).append(mae)
            sizes_sorted = sorted(by_size.keys())
            data = [by_size[ss] for ss in sizes_sorted]
            ax.boxplot(data, positions=range(len(sizes_sorted)),
                       widths=0.55, patch_artist=True, showfliers=False,
                       boxprops={"facecolor": "#e8e8e8",
                                 "edgecolor": "#444", "linewidth": 0.8},
                       medianprops={"color": "#222", "linewidth": 1.2},
                       whiskerprops={"color": "#666", "linewidth": 0.8},
                       capprops={"color": "#666", "linewidth": 0.8})
            rng = np.random.default_rng(0)
            for ms in _METRIC_SOLVER_PAIRS:
                metric, solver = ms
                cell_pts = [(int(r["subset_size"]),
                             float(r["mae_nn_kcalmol"]))
                            for r in combined
                            if r.get("metric") == metric
                            and r.get("solver") == solver]
                for ss, mae in cell_pts:
                    if ss not in sizes_sorted:
                        continue
                    x0 = sizes_sorted.index(ss)
                    jx = float(rng.uniform(-0.18, 0.18))
                    ax.scatter([x0 + jx], [mae],
                               c=[_METRIC_COLOR.get(metric, "#888")],
                               marker=_SOLVER_MARKER.get(solver, "x"),
                               s=22, alpha=0.7, edgecolors="black",
                               linewidths=0.3, zorder=4)
            ax.axhline(pbe_ref, color="#222", linestyle="--",
                       linewidth=1.0, alpha=0.85, zorder=2,
                       label=f"PBE = {pbe_ref:.2f}")
            ax.set_xticks(range(len(sizes_sorted)))
            ax.set_xticklabels([str(s) for s in sizes_sorted])
            if ri == n_rows - 1:
                ax.set_xlabel("subset_size (# training molecules)")
            ax.set_ylabel("NN held-out MAE  (kcal/mol)")
            ax.set_title(cat, fontsize=10, loc="left")
            annotations: List[str] = []
            if spearmanr is not None:
                all_x = [int(r["subset_size"]) for r in combined]
                all_y = [float(r["mae_nn_kcalmol"]) for r in combined]
                try:
                    rho_all, p_all = spearmanr(all_x, all_y)
                    annotations.append(
                        f"all:   ρ = {rho_all:+.3f}  (p = {p_all:.2g})")
                except Exception:  # noqa: BLE001
                    pass
                for ms in _METRIC_SOLVER_PAIRS:
                    metric, solver = ms
                    xs = [int(r["subset_size"]) for r in combined
                          if r.get("metric") == metric
                          and r.get("solver") == solver]
                    ys = [float(r["mae_nn_kcalmol"]) for r in combined
                          if r.get("metric") == metric
                          and r.get("solver") == solver]
                    if len(set(xs)) < 3:
                        continue
                    try:
                        rho, p = spearmanr(xs, ys)
                        annotations.append(
                            f"{metric:>3}·{solver:<7}: ρ = {rho:+.3f}  "
                            f"(p = {p:.2g})")
                    except Exception:  # noqa: BLE001
                        pass
            if annotations:
                ax.text(0.02, 0.98, "Spearman ρ:\n" + "\n".join(annotations),
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=7, family="monospace",
                        bbox=dict(boxstyle="round,pad=0.4",
                                  facecolor="white",
                                  edgecolor="#bbb", alpha=0.9))
            if ri == 0:
                from matplotlib.lines import Line2D
                leg = [Line2D([0], [0], marker="o", linestyle="",
                              markerfacecolor=_METRIC_COLOR.get(m),
                              markeredgecolor="k", markersize=6,
                              label=f"metric={m}")
                       for m in _METRIC_COLOR]
                leg += [Line2D([0], [0], marker=mk, linestyle="",
                               markerfacecolor="white", markeredgecolor="k",
                               markersize=6, label=f"solver={s}")
                        for s, mk in _SOLVER_MARKER.items()]
                leg.append(Line2D([0], [0], color="#222", linestyle="--",
                                  linewidth=1.0,
                                  label=f"PBE = {pbe_ref:.2f}"))
                ax.legend(handles=leg, loc="upper right", framealpha=0.9,
                          fontsize=7, ncol=2)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.suptitle("Held-out NN MAE vs subset_size  (per category)",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_in_sample_vs_held_out(
    eval_rows_by_cat: Dict[str, List[Dict[str, Any]]],
    local_rows_by_cat: Dict[str, List[Dict[str, Any]]],
    lead_label: str,
    out_path: Path,
) -> Path:
    """Fig 11 — overfitting diagnostic, all categories overlaid.

    Per spec: x = cluster's training-subset MAE (from ``eval_df.csv``),
    y = local held-out MAE (from ``local_test_set.csv``). Each category gets
    its own color from ``_palette_for``; markers still encode solver. A
    log-log fit + y = x reference are drawn over the union. Above the
    diagonal = NN worse on held-out than on training (the expected overfit
    case); the gap is the overfit magnitude.

    Axis limits are clamped so the diagonal lives where data lives —
    subset_size=1 specs sometimes overfit their single training molecule to
    ~1e-4 MAE; without the clamp the y=x line stretches across an empty
    decade and looks misleading.
    """
    try:
        from scipy.stats import pearsonr
    except ImportError:
        pearsonr = None  # type: ignore[assignment]

    cats = [c for c in eval_rows_by_cat.keys() if c in local_rows_by_cat]
    palette = _palette_for(cats)
    # Per-category joined points: (in_mae, held_mae, metric, solver, idx).
    pts_by_cat: Dict[str, List[Tuple[float, float, str, str, int]]] = {}
    for cat in cats:
        in_sample_by_idx = {
            int(r["idx"]): float(r["mae"])
            for r in eval_rows_by_cat[cat]
            if r.get("set") == "training_subset"
            and r.get("mae") is not None
            and r.get("idx") is not None
            and math.isfinite(r["mae"])
        }
        held_by_idx = {
            int(r["idx"]): r for r in local_rows_by_cat[cat]
            if r.get("pool") == "held_out_combined"
            and r.get("mae_nn_kcalmol") is not None
            and math.isfinite(r["mae_nn_kcalmol"])
        }
        pts: List[Tuple[float, float, str, str, int]] = []
        for idx, in_mae in in_sample_by_idx.items():
            if idx not in held_by_idx:
                continue
            r = held_by_idx[idx]
            pts.append((in_mae, float(r["mae_nn_kcalmol"]),
                        r.get("metric", "?"), r.get("solver", "?"), idx))
        if pts:
            pts_by_cat[cat] = pts

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(8, 7))
        if not pts_by_cat:
            ax.text(0.5, 0.5, "no joinable in-sample/held-out specs",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#888")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path

        all_xs = np.concatenate(
            [np.array([p[0] for p in pts]) for pts in pts_by_cat.values()])
        all_ys = np.concatenate(
            [np.array([p[1] for p in pts]) for pts in pts_by_cat.values()])
        # Plot each category in its own color; solver still controls marker.
        for cat, pts in pts_by_cat.items():
            color = palette.get(cat, "#888")
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            for marker in set(_SOLVER_MARKER.values()):
                sel = [i for i, p in enumerate(pts)
                       if _SOLVER_MARKER.get(p[3], "x") == marker]
                if not sel:
                    continue
                ax.scatter(xs[sel], ys[sel],
                           c=color, marker=marker, s=42, alpha=0.78,
                           edgecolors="black", linewidths=0.3,
                           zorder=4, label=None)
        # Axis bounds: stretch to where positive-x data actually lives,
        # without clipping small-subset (subset_size=1) trivial-overfit
        # specs out of the figure. The user explicitly requested 2026-05-29
        # that these specs stay visible — they're descriptive even when
        # their in-sample MAE is near 0. We annotate the count separately
        # so the audience knows which markers are trivial fits.
        positive_xs = all_xs[all_xs > 0]
        if positive_xs.size > 0:
            lim_min = max(min(positive_xs.min(), all_ys.min()) * 0.5, 1e-5)
        else:
            lim_min = max(all_ys.min() * 0.5, 1e-5)
        lim_max = max(all_xs.max(), all_ys.max()) * 1.15
        diag = np.array([lim_min, lim_max])
        ax.plot(diag, diag, ls="--", color="#444", lw=1.0, zorder=2,
                label="y = x  (perfect generalization)")
        # Per-category "n trivial-fit specs included" annotation: count specs
        # with in-sample MAE < 0.4 kcal/mol (the prior clamp threshold).
        n_trivial_by_cat = {
            cat: int(np.sum(np.array([p[0] for p in pts]) < 0.4))
            for cat, pts in pts_by_cat.items()
        }
        if any(v > 0 for v in n_trivial_by_cat.values()):
            trivial_label = "trivial-fit specs (x < 0.4): " + ", ".join(
                f"{cat.split('/')[-1] if '/' in cat else cat}: {n}"
                for cat, n in n_trivial_by_cat.items() if n > 0)
            ax.text(0.99, 0.01, trivial_label, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, style="italic",
                    color="#555",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", edgecolor="#bbb",
                              alpha=0.85))
        # Per-category log-log fit lines.
        with np.errstate(invalid="ignore", divide="ignore"):
            for cat, pts in pts_by_cat.items():
                xs = np.array([p[0] for p in pts])
                ys = np.array([p[1] for p in pts])
                mask = (xs > 0) & (ys > 0)
                if mask.sum() < 2:
                    continue
                lx = np.log10(xs[mask])
                ly = np.log10(ys[mask])
                slope, intercept = np.polyfit(lx, ly, 1)
                fit_x = np.array([lim_min, lim_max])
                fit_y = 10 ** (slope * np.log10(fit_x) + intercept)
                ax.plot(fit_x, fit_y, color=palette.get(cat, "#888"),
                        linewidth=1.1, alpha=0.7, zorder=3,
                        label=f"{cat}  (slope={slope:+.2f}, "
                              f"n={len(pts)})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel("cluster training-subset MAE  (kcal/mol; in-sample)")
        ax.set_ylabel("local held-out MAE  (kcal/mol; BH76+W4-11)")
        ax.set_aspect("equal", adjustable="box")

        # Per-category overfit summary; placed bottom-left where no data sits
        # post-clamp (used to overlap the markers when at top-left).
        annotations: List[str] = []
        for cat, pts in pts_by_cat.items():
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            mask = (xs > 0) & (ys > 0)
            n_used = int(mask.sum())
            line = f"{cat}: n={n_used}"
            if pearsonr is not None and n_used >= 3:
                try:
                    lx = np.log10(xs[mask])
                    ly = np.log10(ys[mask])
                    r_val, p_val = pearsonr(lx, ly)
                    line += f"  r={r_val:+.2f} (p={p_val:.2g})"
                except Exception:  # noqa: BLE001
                    pass
            if n_used >= 1:
                overfit_log = float(np.median(
                    np.log10(ys[mask] / xs[mask])))
                line += (f"  med log10(held/in)={overfit_log:+.2f} "
                         f"({10 ** overfit_log:.1f}×)")
            annotations.append(line)
        ax.text(0.02, 0.02, "\n".join(annotations),
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#bbb", alpha=0.9))
        ax.legend(loc="lower right", framealpha=0.9, fontsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        title_cats = " · ".join(pts_by_cat.keys())
        fig.suptitle(
            f"In-sample vs held-out MAE — {title_cats}  "
            "(overfitting diagnostic)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_per_reaction_vs_subset(
    per_rxn_rows_by_cat: Dict[str, List[Dict[str, Any]]],
    lead_label: str,
    out_path: Path,
) -> Path:
    """Fig 12 — per-reaction NN abs-error vs subset_size, color by category.

    4×4 small-multiples (one per reaction). Within each subplot: one line
    per category, showing the median NN |error| across all (metric,
    solver) cells at each subset_size. PBE per-reaction abs-error as a
    dotted horizontal reference (shared across categories).

    The original per-(metric, solver) decomposition was unreadable at
    cross-category scale (12+ lines per tiny subplot) — collapsing the
    (metric, solver) dimension via median keeps the cross-category
    comparison legible.
    """
    cats = list(per_rxn_rows_by_cat.keys())
    palette = _palette_for(cats)
    # Aggregate per (category, name, subset_size).
    by_cat_name_ss: Dict[str, Dict[str, Dict[int, List[float]]]] = {}
    pbe_by_name: Dict[str, List[float]] = {}
    rxn_pool: Dict[str, str] = {}
    for cat in cats:
        by_cat_name_ss[cat] = {}
        for r in per_rxn_rows_by_cat[cat]:
            name = r.get("name")
            if not name:
                continue
            rxn_pool.setdefault(name, r.get("pool", "?"))
            ss = r.get("subset_size")
            nn = r.get("abs_error_nn_kcalmol")
            pbe = r.get("abs_error_pbe_kcalmol")
            if ss is not None and nn is not None and math.isfinite(nn):
                (by_cat_name_ss[cat]
                    .setdefault(name, {})
                    .setdefault(int(ss), [])
                    .append(float(nn)))
            if pbe is not None and math.isfinite(pbe):
                pbe_by_name.setdefault(name, []).append(float(pbe))
    names = sorted(rxn_pool.keys(),
                   key=lambda n: (0 if rxn_pool.get(n) == "bh76" else 1, n))
    n_rxn = len(names)
    if n_rxn == 0:
        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "no per-reaction data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#888")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
        return out_path
    ncols = 4
    nrows = (n_rxn + ncols - 1) // ncols
    with plt.rc_context(_STYLE):
        # Bigger overall figure so subplot titles can carry a larger font.
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(15, 3.0 * nrows + 1.5),
                                 sharex=True)
        axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
        for ax, name in zip(axes_flat, names):
            for cat in cats:
                by_ss = by_cat_name_ss.get(cat, {}).get(name, {})
                if not by_ss:
                    continue
                xs = sorted(by_ss.keys())
                ys = [float(np.median(by_ss[s])) for s in xs]
                ax.plot(xs, ys, marker="o", linestyle="-",
                        color=palette.get(cat, "#888"),
                        linewidth=1.4, markersize=4)
            pbe_vals = pbe_by_name.get(name, [])
            if pbe_vals:
                ax.axhline(float(np.median(pbe_vals)), color="#222",
                           linestyle=":", linewidth=0.9, alpha=0.85,
                           zorder=2)
            # Bigger subplot title — the May-29 figure had ~6pt titles
            # that were unreadable.
            ax.set_title(f"{name}  ({rxn_pool.get(name, '?')})",
                         fontsize=9)
            ax.set_yscale("log")
            ax.set_xticks(list(_SUBSET_SIZES))
            ax.tick_params(labelsize=7)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        for ax in list(axes_flat)[n_rxn:]:
            ax.set_visible(False)
        fig.supxlabel("subset_size", fontsize=10)
        fig.supylabel("median NN |reaction error|  (kcal/mol)",
                      fontsize=10)
        # Legend at TOP of the figure (the May-29 layout crammed it into
        # the bottom margin where matplotlib clipped it).
        from matplotlib.lines import Line2D
        leg = [Line2D([0], [0], marker="o", linestyle="-",
                      color=palette.get(c, "#888"), markersize=6, label=c)
               for c in cats]
        leg.append(Line2D([0], [0], color="#222", linestyle=":",
                          label="median PBE |error|"))
        fig.legend(handles=leg, loc="upper center",
                   bbox_to_anchor=(0.5, 0.965),
                   ncol=min(5, len(leg)),
                   frameon=False, fontsize=9)
        fig.suptitle(
            "Per-reaction NN |error| vs subset_size  (by category)",
            fontsize=12, y=0.995)
        # Add explicit spacing so the bigger titles and inter-subplot
        # gaps render correctly.
        fig.subplots_adjust(hspace=0.45, wspace=0.28,
                            top=0.91, bottom=0.07,
                            left=0.06, right=0.98)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_best_vs_worst_per_reaction(
    per_rxn_rows_by_cat: Dict[str, List[Dict[str, Any]]],
    local_rows_by_cat: Dict[str, List[Dict[str, Any]]],
    lead_label: str,
    out_path: Path,
) -> Path:
    """Fig 13 — best vs worst spec head-to-head, per reaction, per category.

    Each category gets its own subplot row showing that category's best
    and worst spec (by combined NN−PBE delta) vs PBE across the 16
    held-out reactions. Categories without enough data are silently
    skipped.
    """
    cats = [c for c in per_rxn_rows_by_cat
            if c in local_rows_by_cat]

    with plt.rc_context(_STYLE):
        if not cats:
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.text(0.5, 0.5, "no local data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="#888")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path
        n_rows = len(cats)
        fig, axes = plt.subplots(n_rows, 1,
                                 figsize=(14, 3.5 + 3.2 * n_rows),
                                 sharex=False)
        if n_rows == 1:
            axes = [axes]
        for ax, cat in zip(axes, cats):
            local_rows = local_rows_by_cat[cat]
            per_rxn_rows = per_rxn_rows_by_cat[cat]
            combined = [r for r in local_rows
                        if r.get("pool") == "held_out_combined"
                        and r.get("delta_nn_minus_pbe") is not None
                        and math.isfinite(r["delta_nn_minus_pbe"])]
            if not combined:
                ax.text(0.5, 0.5, f"{cat}: no local data",
                        ha="center", va="center",
                        transform=ax.transAxes,
                        fontsize=10, color="#888")
                continue
            best_r = min(combined, key=lambda r: r["delta_nn_minus_pbe"])
            worst_r = max(combined, key=lambda r: r["delta_nn_minus_pbe"])
            best_idx = int(best_r["idx"])
            worst_idx = int(worst_r["idx"])
            best_delta = float(best_r["delta_nn_minus_pbe"])
            worst_delta = float(worst_r["delta_nn_minus_pbe"])
            best_by_name: Dict[str, Dict[str, float]] = {}
            worst_by_name: Dict[str, Dict[str, float]] = {}
            pbe_by_name: Dict[str, float] = {}
            pool_by_name: Dict[str, str] = {}
            for r in per_rxn_rows:
                idx = int(r.get("idx", -1))
                name = r.get("name")
                if not name:
                    continue
                pool_by_name.setdefault(name, r.get("pool", "?"))
                if r.get("abs_error_pbe_kcalmol") is not None:
                    pbe_by_name[name] = float(r["abs_error_pbe_kcalmol"])
                if (idx == best_idx
                        and r.get("abs_error_nn_kcalmol") is not None):
                    best_by_name[name] = {
                        "nn": float(r["abs_error_nn_kcalmol"]),
                    }
                if (idx == worst_idx
                        and r.get("abs_error_nn_kcalmol") is not None):
                    worst_by_name[name] = {
                        "nn": float(r["abs_error_nn_kcalmol"]),
                    }
            names = sorted(pbe_by_name.keys(),
                           key=lambda n: (0 if pool_by_name.get(n) == "bh76"
                                          else 1, n))
            if not names:
                ax.text(0.5, 0.5, f"{cat}: no per-reaction data",
                        ha="center", va="center",
                        transform=ax.transAxes,
                        fontsize=10, color="#888")
                continue
            nn_best = [best_by_name.get(n, {}).get("nn", float("nan"))
                       for n in names]
            nn_worst = [worst_by_name.get(n, {}).get("nn", float("nan"))
                        for n in names]
            pbe_each = [pbe_by_name.get(n, float("nan")) for n in names]
            x = np.arange(len(names))
            w = 0.27
            ax.bar(x - w, nn_best, w, color="#4daf4a", edgecolor="black",
                   linewidth=0.4,
                   label=f"best NN  (spec_{best_idx:04d}, "
                         f"Δ={best_delta:+.2f})")
            ax.bar(x, pbe_each, w, color="#777777", edgecolor="black",
                   linewidth=0.4, label="PBE")
            ax.bar(x + w, nn_worst, w, color="#c0504d",
                   edgecolor="black", linewidth=0.4,
                   label=f"worst NN  (spec_{worst_idx:04d}, "
                         f"Δ={worst_delta:+.2f})")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel("|rxn error|  (kcal/mol)")
            ax.set_title(f"Best vs worst — {cat}", fontsize=10,
                         loc="left")
            n_bh76 = sum(1 for n in names
                         if pool_by_name.get(n) == "bh76")
            if 0 < n_bh76 < len(names):
                ax.axvline(n_bh76 - 0.5, color="#bbb", linewidth=0.8,
                           linestyle=":")
                ymax = max(max(nn_best + pbe_each + nn_worst),
                           1.0)
                ax.text(n_bh76 / 2.0 - 0.5, ymax * 0.95, "BH76",
                        ha="center", va="top", fontsize=7,
                        color="#888", style="italic")
                ax.text((n_bh76 + len(names)) / 2.0 - 0.5,
                        ymax * 0.95, "W4-11",
                        ha="center", va="top", fontsize=7,
                        color="#888", style="italic")
            ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.suptitle("Best vs worst spec per reaction (per category)",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_density_vs_energy_by_cell(
    per_mol_rows_by_cat: Dict[str, List[Dict[str, Any]]],
    lead_label: str,
    out_path: Path,
) -> Path:
    """Fig 14 — density-RMSE vs |AE error| split by (metric, solver), with
    one row per category.

    Each row is the existing 2×2 (metric × solver) grid for one category.
    Points are colored by category so a single panel still encodes both
    (metric, solver) AND category for direct cross-sweep comparison.
    """
    cats = list(per_mol_rows_by_cat.keys())
    palette = _palette_for(cats)
    with plt.rc_context(_STYLE):
        if not cats:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.text(0.5, 0.5, "no per-molecule data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#888")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path
        n_rows = len(cats)
        fig, axes = plt.subplots(n_rows * 2, 2,
                                 figsize=(11, 4.5 * n_rows + 1.0),
                                 sharex=True, sharey=True)
        if n_rows == 1:
            axes = axes.reshape(2, 2)
        cells_grid = [("jsd", "full_3"), ("jsd", "oneshot"),
                      ("l2", "full_3"), ("l2", "oneshot")]
        for ri, cat in enumerate(cats):
            color = palette.get(cat, "#888")
            by_cell: Dict[Tuple[str, str],
                          List[Tuple[float, float]]] = {}
            for r in per_mol_rows_by_cat[cat]:
                ae = r.get("AE_error_kcalmol")
                rho = r.get("density_rmse")
                if (ae is None or rho is None
                        or not math.isfinite(ae)
                        or not math.isfinite(rho)):
                    continue
                key = (r.get("metric") or "?", r.get("solver") or "?")
                by_cell.setdefault(key, []).append(
                    (float(rho), abs(float(ae))))
            for ci, key in enumerate(cells_grid):
                ax = axes[ri * 2 + ci // 2, ci % 2]
                pts = by_cell.get(key, [])
                if not pts:
                    ax.text(0.5, 0.5, f"{cat}: no data",
                            ha="center", va="center",
                            transform=ax.transAxes, color="#888",
                            fontsize=9)
                    ax.set_title(
                        f"{cat}  —  metric={key[0]}, solver={key[1]}",
                        fontsize=8)
                    continue
                xs = np.array([p[0] for p in pts])
                ys = np.array([p[1] for p in pts])
                ax.scatter(xs, ys, c=color,
                           marker=_SOLVER_MARKER.get(key[1], "x"),
                           s=14, alpha=0.55, edgecolors="none")
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_title(f"{cat}  —  metric={key[0]}, "
                             f"solver={key[1]}  (n={len(pts)})",
                             fontsize=8)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
        for ax in axes[-1, :]:
            ax.set_xlabel("density RMSE  (a.u.)")
        for ax in axes[:, 0]:
            ax.set_ylabel("|AE error|  (kcal/mol)")
        fig.suptitle("Density-quality vs energy-quality per loss "
                     "configuration  (per category)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Descriptor figures (Figs 10 + 15)
# ---------------------------------------------------------------------------


def plot_cross_category_nn_vs_pbe(
    local_rows_by_category: Dict[str, List[Dict[str, Any]]],
    out_path: Path,
) -> Path:
    """Fig 16 — cross-category NN vs PBE on the combined held-out set.

    ``local_rows_by_category`` maps category label (e.g. ``"alpha_on/runs"``)
    to its list of ``local_test_set.csv`` rows. Only categories with at
    least one ``held_out_combined`` row are plotted; categories whose
    pulls are still incomplete are silently skipped.

    Layout: 1×2.

    - Left panel: strip plot of per-spec NN held-out MAE, one column per
      (category × metric × solver). Horizontal PBE = 9.56 reference line;
      green-shaded "NN beats PBE" half-plane. Per-column win count
      (n_below_PBE / total) annotated at the top of each column.
    - Right panel: KDE-flavored stacked histograms of NN−PBE delta per
      category. Δ = 0 (PBE) vertical line; per-category subtitle with
      "k specs beat PBE; median Δ = …".
    """
    cats: List[str] = []
    combined_by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for cat, rows in local_rows_by_category.items():
        combined = [r for r in rows
                    if r.get("pool") == "held_out_combined"
                    and r.get("mae_nn_kcalmol") is not None
                    and math.isfinite(r["mae_nn_kcalmol"])
                    and r.get("mae_pbe_kcalmol") is not None
                    and math.isfinite(r["mae_pbe_kcalmol"])]
        if combined:
            cats.append(cat)
            combined_by_cat[cat] = combined

    with plt.rc_context(_STYLE):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
        if not cats:
            for ax in (ax_l, ax_r):
                ax.text(0.5, 0.5, "no categories with local test_set data",
                        ha="center", va="center",
                        transform=ax.transAxes,
                        fontsize=11, color="#888")
            fig.suptitle("Cross-category NN vs PBE  (no data)",
                         fontsize=12)
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path

        # Use the median PBE across all categories for the horizontal
        # reference. They should all match by construction since the same
        # 16 reactions are evaluated everywhere.
        pbe_all = [float(r["mae_pbe_kcalmol"])
                   for rows in combined_by_cat.values() for r in rows]
        pbe_ref = float(np.median(pbe_all))

        # ---- Panel A: strip plot per (category, metric, solver) -------
        cell_positions: Dict[Tuple[str, str, str], int] = {}
        x_labels: List[str] = []
        x_idx = 0
        for cat in cats:
            for ms in _METRIC_SOLVER_PAIRS:
                cell_positions[(cat, ms[0], ms[1])] = x_idx
                x_labels.append(f"{ms[0]}\n{ms[1]}")
                x_idx += 1
            # blank separator between categories
            if cat != cats[-1]:
                x_labels.append("")
                x_idx += 1
        rng = np.random.default_rng(0)
        y_max = max(float(r["mae_nn_kcalmol"]) for r in
                    [r for rows in combined_by_cat.values() for r in rows]
                    ) * 1.08
        y_min = min(min(float(r["mae_nn_kcalmol"]) for r in
                        combined_by_cat[cat]) for cat in cats) * 0.85
        y_min = min(y_min, pbe_ref * 0.85)
        for cat in cats:
            color = _CATEGORY_COLOR.get(cat, "#888")
            for r in combined_by_cat[cat]:
                key = (cat, r.get("metric") or "?", r.get("solver") or "?")
                x0 = cell_positions.get(key)
                if x0 is None:
                    continue
                jx = float(rng.uniform(-0.22, 0.22))
                ax_l.scatter([x0 + jx], [float(r["mae_nn_kcalmol"])],
                             c=[color],
                             marker=_SOLVER_MARKER.get(r.get("solver"), "x"),
                             s=34, alpha=0.85,
                             edgecolors="black", linewidths=0.3, zorder=4)
        # PBE reference + shaded "wins" half-plane.
        ax_l.axhline(pbe_ref, color="#222", linestyle="--", linewidth=1.0,
                     zorder=2, label=f"PBE = {pbe_ref:.2f}")
        ax_l.axhspan(y_min, pbe_ref, color="#4daf4a", alpha=0.10, zorder=1)

        # Category band labels at the top + per-cell win counts.
        for cat in cats:
            cell_keys = [(cat, ms[0], ms[1]) for ms in _METRIC_SOLVER_PAIRS]
            xs = [cell_positions[k] for k in cell_keys]
            band_center = (xs[0] + xs[-1]) / 2.0
            ax_l.text(band_center, y_max * 1.0, cat,
                      ha="center", va="bottom", fontsize=9, style="italic",
                      color=_CATEGORY_COLOR.get(cat, "#444"))
            # Per-cell n_below_PBE / total annotation.
            for ms, x0 in zip(_METRIC_SOLVER_PAIRS, xs):
                cell_specs = [float(r["mae_nn_kcalmol"]) for r in
                              combined_by_cat[cat]
                              if (r.get("metric"), r.get("solver")) == ms]
                if not cell_specs:
                    continue
                n_wins = sum(1 for v in cell_specs if v < pbe_ref)
                color_text = ("#2a6a2f" if n_wins
                              else "#888")
                ax_l.text(x0, y_max * 0.96,
                          f"{n_wins}/{len(cell_specs)}",
                          ha="center", va="top", fontsize=7,
                          color=color_text, family="monospace")
        ax_l.set_xlim(-0.6, x_idx - 0.4)
        ax_l.set_ylim(y_min, y_max * 1.05)
        ax_l.set_xticks(list(range(x_idx)))
        ax_l.set_xticklabels(x_labels, fontsize=7)
        ax_l.set_ylabel("NN held-out MAE  (kcal/mol)")
        ax_l.set_title(f"Per-spec NN MAE by category × loss configuration"
                       f"  (n_specs total = "
                       f"{sum(len(combined_by_cat[c]) for c in cats)})")
        # Legend.
        from matplotlib.lines import Line2D
        leg = []
        for cat in cats:
            leg.append(Line2D([0], [0], marker="o", linestyle="",
                              markerfacecolor=_CATEGORY_COLOR.get(cat, "#888"),
                              markeredgecolor="black",
                              markersize=7, label=cat))
        for s, mk in _SOLVER_MARKER.items():
            leg.append(Line2D([0], [0], marker=mk, linestyle="",
                              markerfacecolor="white",
                              markeredgecolor="black",
                              markersize=7, label=f"solver={s}"))
        leg.append(Line2D([0], [0], color="#222", linestyle="--",
                          label=f"PBE = {pbe_ref:.2f}"))
        ax_l.legend(handles=leg, loc="upper right", framealpha=0.9,
                    fontsize=7, ncol=1)
        for sp in ("top", "right"):
            ax_l.spines[sp].set_visible(False)

        # ---- Panel B: per-category NN−PBE delta histograms ------------
        all_deltas = []
        for cat in cats:
            d = [float(r["mae_nn_kcalmol"]) - float(r["mae_pbe_kcalmol"])
                 for r in combined_by_cat[cat]]
            all_deltas.extend(d)
        if all_deltas:
            bins = np.linspace(min(all_deltas) - 0.5,
                                max(all_deltas) + 0.5, 24)
        else:
            bins = np.linspace(-5, 20, 24)
        for cat in cats:
            d = np.asarray(
                [float(r["mae_nn_kcalmol"]) - float(r["mae_pbe_kcalmol"])
                 for r in combined_by_cat[cat]])
            n_wins = int(np.sum(d < 0))
            n_total = len(d)
            color = _CATEGORY_COLOR.get(cat, "#888")
            ax_r.hist(d, bins=bins, color=color, alpha=0.55,
                      edgecolor="black", linewidth=0.4,
                      label=(f"{cat}: {n_wins}/{n_total} beat PBE  "
                             f"(median Δ = {float(np.median(d)):+.2f})"))
        ax_r.axvline(0, color="#222", linestyle="--", linewidth=1.0,
                     label="Δ = 0  (= PBE)")
        ax_r.set_xlabel("NN − PBE  (kcal/mol;  <0 = NN beats PBE)")
        ax_r.set_ylabel("# specs")
        ax_r.set_title("Distribution of NN − PBE delta, per category")
        ax_r.legend(loc="upper right", framealpha=0.9, fontsize=7)
        for sp in ("top", "right"):
            ax_r.spines[sp].set_visible(False)

        # Overall title summarizes the categories' "beat PBE" counts.
        headline_parts = []
        for cat in cats:
            d = [float(r["mae_nn_kcalmol"]) - float(r["mae_pbe_kcalmol"])
                 for r in combined_by_cat[cat]]
            n_wins = sum(1 for x in d if x < 0)
            headline_parts.append(f"{cat} {n_wins}/{len(d)}")
        fig.suptitle(
            "Cross-category NN vs PBE on held-out  ("
            + ", ".join(headline_parts) + " beat PBE)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def _spec_diversity_score(per_subset_stats: Dict[str, Any]) -> Optional[float]:
    """Per-spec scalar diversity score from the ``per_subset_stats``.

    Uses the **L2 norm of the per-feature range vector**, normalized by
    the feature count so the score is comparable across architectures
    with different descriptor totals. Returns ``None`` if the range data
    is missing or all-zero.
    """
    rng = per_subset_stats.get("range")
    if not rng:
        return None
    arr = np.asarray(rng, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.sqrt(np.mean(arr ** 2)))


def plot_descriptor_range_vs_accuracy(
    descriptor_rows_by_cat: Dict[str, List[Dict[str, Any]]],
    local_rows_by_cat: Dict[str, List[Dict[str, Any]]],
    lead_label: str,
    out_path: Path,
) -> Path:
    """Fig 10 — per-spec descriptor diversity vs held-out NN MAE, cross-cat.

    x = diversity score (L2 norm of the per-feature range across training
    molecules), y = held-out combined NN MAE. Color = category. Marker =
    solver. Per-category Spearman ρ stacked in the annotation box.
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        spearmanr = None  # type: ignore[assignment]

    cats = [c for c in descriptor_rows_by_cat.keys()
            if c in local_rows_by_cat]
    palette = _palette_for(cats)
    pts_by_cat: Dict[str, List[Tuple[float, float, int, str, str]]] = {}
    for cat in cats:
        mae_by_idx = {int(r["idx"]): float(r["mae_nn_kcalmol"])
                      for r in local_rows_by_cat[cat]
                      if r.get("pool") == "held_out_combined"
                      and r.get("mae_nn_kcalmol") is not None
                      and math.isfinite(r["mae_nn_kcalmol"])}
        pts: List[Tuple[float, float, int, str, str]] = []
        for r in descriptor_rows_by_cat[cat]:
            idx = int(r["idx"])
            if idx not in mae_by_idx:
                continue
            score = _spec_diversity_score(r.get("per_subset_stats", {}))
            if score is None:
                continue
            pts.append((score, mae_by_idx[idx],
                        int(r.get("subset_size") or 0),
                        r.get("metric") or "?", r.get("solver") or "?"))
        if pts:
            pts_by_cat[cat] = pts

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(11, 6.5))
        if not pts_by_cat:
            ax.text(0.5, 0.5, "no descriptor data\n"
                    "(run extract_subset_descriptors.py --auto)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#888")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path
        for cat in pts_by_cat:
            color = palette.get(cat, "#888")
            pts = pts_by_cat[cat]
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            for marker in set(_SOLVER_MARKER.values()):
                sel = [i for i, p in enumerate(pts)
                       if _SOLVER_MARKER.get(p[4], "x") == marker]
                if not sel:
                    continue
                ax.scatter(xs[sel], ys[sel],
                           c=color, marker=marker, s=48, alpha=0.78,
                           edgecolors="black", linewidths=0.35, zorder=4)
        ax.set_xlabel("descriptor diversity score  "
                      "(L2 norm of per-feature range)")
        ax.set_ylabel("held-out NN MAE  (kcal/mol)")
        annotations: List[str] = []
        for cat, pts in pts_by_cat.items():
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            line = f"{cat}:  n={len(pts)}"
            if spearmanr is not None and len(pts) >= 4:
                try:
                    rho, p_val = spearmanr(xs, ys)
                    line += f"   ρ={rho:+.3f} (p={p_val:.2g})"
                except Exception:  # noqa: BLE001
                    pass
            annotations.append(line)
        ax.text(0.02, 0.98, "\n".join(annotations),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#bbb", alpha=0.9))
        from matplotlib.lines import Line2D
        leg = [Line2D([0], [0], marker="o", linestyle="",
                      markerfacecolor=palette.get(cat, "#888"),
                      markeredgecolor="black", markersize=7, label=cat)
               for cat in pts_by_cat]
        leg += [Line2D([0], [0], marker=mk, linestyle="",
                       markerfacecolor="white", markeredgecolor="black",
                       markersize=7, label=f"solver={s}")
                for s, mk in _SOLVER_MARKER.items()]
        ax.legend(handles=leg, loc="lower right", framealpha=0.9,
                  fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.suptitle("Training-set descriptor diversity vs held-out "
                     "accuracy  (all categories)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_descriptor_histograms_by_metric(
    descriptor_rows_by_cat: Dict[str, List[Dict[str, Any]]],
    lead_label: str,
    out_path: Path,
) -> Path:
    """Fig 15 — per-subset descriptor distributions per (category, metric).

    Layout: (n_categories × 2) rows, one per (cat, metric={jsd, l2});
    columns: one per descriptor feature. Each cell: per-subset_size
    histograms colored by subset_size, with vertical lines marking mean
    and ±1σ per subset. The largest subset's histogram is drawn as a
    thick reference outline. Tells the PI whether smaller subsets span
    the same descriptor distribution as the full subset, AND whether the
    distribution shape is consistent across α-sweep cluster categories.
    """
    cats = [c for c in descriptor_rows_by_cat
            if descriptor_rows_by_cat[c]]
    with plt.rc_context(_STYLE):
        if not cats:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.text(0.5, 0.5, "no descriptor data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#888")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path
        feature_names = (descriptor_rows_by_cat[cats[0]][0]
                         .get("feature_names", []) or [])
        if not feature_names:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.text(0.5, 0.5, "no feature_names in descriptor rows",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#888")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return out_path
        n_features = len(feature_names)
        metrics = ("jsd", "l2")
        n_rows = len(cats) * len(metrics)
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=min(_SUBSET_SIZES),
                             vmax=max(_SUBSET_SIZES))
        fig, axes = plt.subplots(
            n_rows, n_features,
            figsize=(3.0 * n_features + 0.8, 2.7 * n_rows + 0.4),
            sharex="col")
        if n_rows == 1 and n_features == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, n_features)
        elif n_features == 1:
            axes = axes.reshape(n_rows, 1)
        for cat_i, cat in enumerate(cats):
            descriptor_rows = descriptor_rows_by_cat[cat]
            for mi, metric in enumerate(metrics):
                ri = cat_i * len(metrics) + mi
                by_ss: Dict[int, List[np.ndarray]] = {}
                for r in descriptor_rows:
                    if r.get("metric") != metric:
                        continue
                    ss = r.get("subset_size")
                    pm = r.get("per_molecule_features")
                    if ss is None or pm is None or pm.size == 0:
                        continue
                    by_ss.setdefault(int(ss), []).append(pm)
                ref_ss = max(by_ss.keys()) if by_ss else None
                for ci in range(n_features):
                    ax = axes[ri, ci]
                    all_vals: List[float] = []
                    for vals_list in by_ss.values():
                        for pm in vals_list:
                            all_vals.extend(pm[:, ci].tolist())
                    if not all_vals:
                        ax.text(0.5, 0.5, "no data", ha="center",
                                va="center", transform=ax.transAxes,
                                color="#888", fontsize=9)
                        if ci == 0:
                            ax.set_ylabel(
                                f"{cat}\nmetric={metric}", fontsize=7)
                        continue
                    lo, hi = (float(np.min(all_vals)),
                              float(np.max(all_vals)))
                    if hi == lo:
                        hi = lo + 1.0
                    bins = np.linspace(lo, hi, 24)
                    for ss in sorted(by_ss.keys()):
                        vals = np.concatenate(
                            [pm[:, ci] for pm in by_ss[ss]])
                        if vals.size == 0:
                            continue
                        color = cmap(norm(ss))
                        if ss == ref_ss:
                            ax.hist(vals, bins=bins, histtype="step",
                                    color="black", linewidth=1.4,
                                    label=f"ref ss={ss}")
                        ax.hist(vals, bins=bins, histtype="stepfilled",
                                color=color, alpha=0.35,
                                linewidth=0.5, edgecolor=color)
                        mu = float(np.mean(vals))
                        sd = float(np.std(vals))
                        ax.axvline(mu, color=color, linestyle="-",
                                   linewidth=0.6, alpha=0.8)
                        ax.axvline(mu - sd, color=color, linestyle=":",
                                   linewidth=0.4, alpha=0.6)
                        ax.axvline(mu + sd, color=color, linestyle=":",
                                   linewidth=0.4, alpha=0.6)
                    if ri == n_rows - 1:
                        ax.set_xlabel(feature_names[ci], fontsize=8)
                    if ci == 0:
                        ax.set_ylabel(f"{cat}\nmetric={metric}\n#mol obs",
                                      fontsize=7)
                    for sp in ("top", "right"):
                        ax.spines[sp].set_visible(False)
                    ax.tick_params(labelsize=7)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6,
                            pad=0.02, fraction=0.02, location="right")
        cbar.set_label("subset_size  (line = mean, dotted = ±1σ; "
                       "black outline = largest)", fontsize=8)
        fig.suptitle(
            "Training-subset descriptor distributions, by (category × "
            "metric)", fontsize=12)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Composite Fig 4
# ---------------------------------------------------------------------------

def plot_composite(eval_rows_by_cat: Dict[str, List[Dict[str, Any]]],
                   per_mol_rows_by_cat: Dict[str, List[Dict[str, Any]]],
                   final_losses_by_cat: Dict[str, Dict[int, Optional[float]]],
                   grids_by_category: Dict[str, Dict[Tuple[str, str], Dict[int, str]]],
                   summary: Dict[str, Tuple[int, int]],
                   lead_label: str,
                   out_path: Path,
                   local_rows_by_cat: Optional[
                       Dict[str, List[Dict[str, Any]]]] = None,
                   ) -> Path:
    """Fig 4 — 2x2 cross-category composite suitable for a slide/poster.

    Layout:
      ┌────────────────────────┬─────────────────────────┐
      │ (A) lead MAE vs subset  │ (B) lead training-success │
      ├────────────────────────┼─────────────────────────┤
      │ (C) top-20 hardest mols │ (D) coverage by category │
      └────────────────────────┴─────────────────────────┘

    Panels A, B, C show the LEAD category's data (smaller panels can't
    cleanly host 3 overlapping series). Headline title lists all
    categories with completion counts so the audience sees the
    cross-category picture; the per-category drill-downs live in fig1,
    fig5, fig9, etc. Panel D's coverage grid is already cross-category.
    """
    local_rows_by_cat = local_rows_by_cat or {}
    lead_eval = eval_rows_by_cat.get(lead_label, [])
    lead_per_mol = per_mol_rows_by_cat.get(lead_label, [])
    lead_final_losses = final_losses_by_cat.get(lead_label, {})
    lead_local = local_rows_by_cat.get(lead_label) or None
    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])

        _plot_mae_vs_subset(ax_a, lead_eval, local_rows=lead_local,
                            compact_legend=True)
        title_a = (
            f"(A)  In-sample vs held-out MAE — {lead_label}"
            if lead_local
            else f"(A)  Training-subset MAE vs subset — {lead_label}"
        )
        ax_a.set_title(title_a)

        _plot_training_success(ax_b, lead_eval, lead_final_losses)
        ax_b.set_title("(B)  Training-success diagnostic  (low-low = best)")

        _plot_per_molecule_top_n(ax_c, lead_per_mol, top_n=20)
        ax_c.set_title("(C)  Top-20 hardest molecules  (in-sample AE error)")

        # --- Panel D: dashboard with all categories --------------------
        cats = list(grids_by_category.keys()) or ["(none)"]
        cmap = ListedColormap([_STATUS_COLOR[s] for s in _STATUS_ORDER])
        # Stack the per-category mini-grids vertically inside this one axes
        # by composing one big matrix with horizontal "separator" rows.
        SEP_ROWS = 1  # blank rows between categories (drawn as 'missing')
        blocks: List[np.ndarray] = []
        ytick_pos: List[int] = []
        ytick_lab: List[str] = []
        cur_row = 0
        for cat in cats:
            mat = _status_grid_matrix(grids_by_category.get(cat, {}))
            blocks.append(mat)
            ytick_pos.append(cur_row + mat.shape[0] // 2)
            done, total = summary.get(cat, (0, 0))
            label = f"{cat}\n({done}/{total})" if total else f"{cat}\n(no mfst)"
            ytick_lab.append(label)
            cur_row += mat.shape[0]
            if cat != cats[-1]:
                blocks.append(
                    np.full((SEP_ROWS, len(_SUBSET_SIZES)),
                            _STATUS_ORDER.index("missing"), dtype=int))
                cur_row += SEP_ROWS
        big = np.vstack(blocks) if blocks else np.zeros((1, len(_SUBSET_SIZES)),
                                                        dtype=int)
        ax_d.imshow(big, aspect="auto", cmap=cmap,
                    vmin=0, vmax=len(_STATUS_ORDER) - 1)
        ax_d.set_xticks(range(len(_SUBSET_SIZES)))
        ax_d.set_xticklabels([str(s) for s in _SUBSET_SIZES])
        ax_d.set_yticks(ytick_pos)
        ax_d.set_yticklabels(ytick_lab, fontsize=8)
        ax_d.set_xlabel("subset_size")
        ax_d.set_title("(D)  Coverage by category")
        for sp in ("top", "right"): ax_d.spines[sp].set_visible(False)

        # Single composite-wide legend at the very bottom.
        legend_handles = [
            Patch(facecolor=_STATUS_COLOR[s], edgecolor="#444", label=s)
            for s in ("complete", "trained_no_eval", "eval_skipped",
                      "train_failed", "pending", "missing")
        ]
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=len(legend_handles), frameon=False,
                   bbox_to_anchor=(0.5, 0.0), fontsize=8)
        # Headline: lead category called out + every other category's
        # completion count so the audience sees the cross-category picture
        # at a glance.
        other_cats = [c for c in summary.keys() if c != lead_label]
        head_parts = [f"lead: {lead_label} "
                      f"({summary.get(lead_label, (0, 0))[0]}/"
                      f"{summary.get(lead_label, (0, 0))[1]})"]
        for c in other_cats:
            done, total = summary.get(c, (0, 0))
            head_parts.append(f"{c} {done}/{total}" if total
                              else f"{c} (no manifest)")
        fig.suptitle(
            "Cluster sweep summary — " + ";  ".join(head_parts)
            + "  (in-sample fit only — runbook §10.5)",
            fontsize=12, y=0.995)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_local_root() -> str:
    return os.environ.get(
        "XCQUINOX_CLUSTER_LOCAL_ROOT",
        str(Path.home() / "Documents/Research/xcquinox-results/runs"),
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-root", default=_default_local_root(),
                   help="root holding categories with run_<UTC>Z dirs "
                        "(default: $XCQUINOX_CLUSTER_LOCAL_ROOT else "
                        "~/Documents/Research/xcquinox-results/runs)")
    p.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="where to write PNGs (default: this script's dir)")
    p.add_argument("--prefix", default="cluster_pulls",
                   help="filename prefix for the four PNGs")
    p.add_argument("--lead-category", default=None,
                   help="which category to feature in Figs 1, 2, 4 (default: "
                        "the most-complete category)")
    args = p.parse_args(argv)

    local_root = Path(args.local_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = discover_pulled_categories(local_root)
    if not categories:
        print(f"no run_<UTC>Z dirs found under {local_root}", file=sys.stderr)
        return 1
    print(f"found {len(categories)} categories under {local_root}:", flush=True)
    for cat, rd in categories.items():
        print(f"  {cat or '(root)'}  ->  {rd.name}", flush=True)

    grids = {cat: aggregate_status_grid(rd) for cat, rd in categories.items()}
    summary = _completion_summary(grids)

    if args.lead_category and args.lead_category in categories:
        lead = args.lead_category
    else:
        # Pick the category with the most `complete` specs.
        lead = max(summary, key=lambda c: summary[c][0]) if summary else None
    if lead is None:
        print("no category to plot — bailing", file=sys.stderr)
        return 1
    lead_label = lead or "(root)"
    print(f"lead category: {lead_label} "
          f"(complete: {summary[lead][0]}/{summary[lead][1]})", flush=True)

    # Collect data for EVERY pulled category. Each by_category dict is
    # category_label -> rows; categories with empty results are omitted so
    # downstream plot functions only iterate categories that have data.
    eval_rows_by_cat: Dict[str, List[Dict[str, Any]]] = {}
    per_mol_rows_by_cat: Dict[str, List[Dict[str, Any]]] = {}
    final_losses_by_cat: Dict[str, Dict[int, Optional[float]]] = {}
    local_rows_by_category: Dict[str, List[Dict[str, Any]]] = {}
    per_rxn_rows_by_cat: Dict[str, List[Dict[str, Any]]] = {}
    descriptor_rows_by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for cat, rd in categories.items():
        er = collect_eval_df_rows(rd)
        if er:
            eval_rows_by_cat[cat] = er
        pmr = collect_per_molecule_rows(rd)
        if pmr:
            per_mol_rows_by_cat[cat] = pmr
        fl = load_final_losses(rd)
        if fl:
            final_losses_by_cat[cat] = fl
        lr = collect_local_test_set_rows(rd)
        if lr:
            local_rows_by_category[cat] = lr
        prr = collect_per_reaction_rows(rd)
        if prr:
            per_rxn_rows_by_cat[cat] = prr
        dr = collect_subset_descriptor_rows(rd)
        if dr:
            descriptor_rows_by_cat[cat] = dr
    # Per-lead-category slices (used by plots that still need a single primary
    # category — currently only the dashboard summary). Plot functions that
    # have been migrated to cross-category accept the *_by_cat dicts above.
    eval_rows = eval_rows_by_cat.get(lead, [])
    per_mol_rows = per_mol_rows_by_cat.get(lead, [])
    final_losses = final_losses_by_cat.get(lead, {})
    local_rows = local_rows_by_category.get(lead, [])
    per_rxn_rows = per_rxn_rows_by_cat.get(lead, [])
    descriptor_rows = descriptor_rows_by_cat.get(lead, [])
    print(f"  eval rows: {len(eval_rows)}; per-molecule rows: "
          f"{len(per_mol_rows)}; final_losses: {len(final_losses)}; "
          f"local test_set rows: {len(local_rows)}; "
          f"per-reaction rows: {len(per_rxn_rows)}", flush=True)
    print(f"  per-category eval rows: "
          f"{ {c: len(v) for c, v in eval_rows_by_cat.items()} }",
          flush=True)

    out1 = out_dir / f"{args.prefix}_fig1_training_diagnostics.png"
    out2 = out_dir / f"{args.prefix}_fig2_per_molecule_errors.png"
    out3 = out_dir / f"{args.prefix}_fig3_coverage_dashboard.png"
    out4 = out_dir / f"{args.prefix}_fig_composite_summary.png"
    out5 = out_dir / f"{args.prefix}_fig5_nn_vs_pbe.png"
    out6 = out_dir / f"{args.prefix}_fig6_per_pool.png"
    out7 = out_dir / f"{args.prefix}_fig7_grid_heatmap.png"
    out8 = out_dir / f"{args.prefix}_fig8_per_reaction.png"
    out9 = out_dir / f"{args.prefix}_fig9_subset_size_correlation.png"
    out11 = out_dir / f"{args.prefix}_fig11_in_sample_vs_held_out.png"
    out12 = out_dir / f"{args.prefix}_fig12_per_reaction_vs_subset.png"
    out13 = out_dir / f"{args.prefix}_fig13_best_vs_worst_per_reaction.png"
    out14 = out_dir / f"{args.prefix}_fig14_density_vs_energy_by_cell.png"
    out10 = out_dir / f"{args.prefix}_fig10_descriptor_range_vs_accuracy.png"
    out15 = out_dir / f"{args.prefix}_fig15_descriptor_histograms.png"
    out16 = out_dir / f"{args.prefix}_fig16_cross_category_nn_vs_pbe.png"
    print(f"  descriptor rows: {len(descriptor_rows)} specs;  "
          f"cross-category local rows: "
          f"{ {c: len(v) for c, v in local_rows_by_category.items()} }",
          flush=True)

    plot_generalization(eval_rows_by_cat, per_mol_rows_by_cat,
                        final_losses_by_cat, lead_label, out1,
                        local_rows_by_cat=local_rows_by_category)
    plot_per_molecule(per_mol_rows_by_cat, lead_label, out2)
    plot_dashboard(grids, summary, out3)
    plot_composite(eval_rows_by_cat, per_mol_rows_by_cat,
                   final_losses_by_cat, grids, summary,
                   lead_label, out4,
                   local_rows_by_cat=local_rows_by_category)
    plot_nn_vs_pbe(local_rows_by_category, lead_label, out5)
    plot_per_pool(local_rows_by_category, lead_label, out6)
    plot_grid_heatmap(local_rows_by_category, lead_label, out7)
    plot_per_reaction(per_rxn_rows_by_cat, lead_label, out8)
    plot_subset_size_correlation(local_rows_by_category, lead_label, out9)
    plot_in_sample_vs_held_out(eval_rows_by_cat, local_rows_by_category,
                               lead_label, out11)
    plot_per_reaction_vs_subset(per_rxn_rows_by_cat, lead_label, out12)
    plot_best_vs_worst_per_reaction(per_rxn_rows_by_cat,
                                    local_rows_by_category,
                                    lead_label, out13)
    plot_density_vs_energy_by_cell(per_mol_rows_by_cat, lead_label, out14)
    plot_descriptor_range_vs_accuracy(descriptor_rows_by_cat,
                                       local_rows_by_category,
                                       lead_label, out10)
    plot_descriptor_histograms_by_metric(descriptor_rows_by_cat,
                                          lead_label, out15)
    plot_cross_category_nn_vs_pbe(local_rows_by_category, out16)

    print("figures:", flush=True)
    for p in (out1, out2, out3, out4, out5, out6, out7, out8,
              out9, out11, out12, out13, out14, out10, out15, out16):
        size_kb = p.stat().st_size / 1024 if p.exists() else 0
        print(f"  {p}  ({size_kb:.0f} KB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
