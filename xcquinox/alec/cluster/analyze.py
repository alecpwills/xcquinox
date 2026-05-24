"""xcquinox.alec.cluster.analyze — aggregate per-spec eval results.

The eval stage writes, per grid cell, ``checkpoints/spec_<idx>/eval_df.csv``
(``set,mae,rho_rmse,n_eval``). This module joins those scalars with each cell's
grid parameters (from ``manifest.json``) and reports them — re-runnable as
results trickle in, and **skipping incomplete spec dirs** (they appear in the
table with ``status`` but never contribute to the metric statistics).

Pure + importable (no SLURM, no dep on ``__main__``); the ``results`` CLI
subcommand is a thin wrapper. Per-spec status precedence:

  complete        -- a parseable ``eval_df.csv`` exists (carries the metrics)
  eval_skipped    -- ``eval/skipped.json`` (training produced no model.eqx)
  train_failed    -- ``failure.json`` (carries its ``classification``)
  trained_no_eval -- ``model.eqx`` present but no eval_df.csv yet
  pending         -- none of the above
"""
import csv
import json
import math
import os
import statistics

# Hartree -> kcal/mol (matches evaluation.py:25 HA_TO_KCAL); used only to show
# the predicted AE (stored in Hartree as AE_nn) in kcal/mol.
HA_TO_KCAL = 627.5094740631

# Public column order for rows + CSV export.
ROW_FIELDS = (
    "idx", "arch", "loss", "metric", "subset_size", "solver",
    "status", "mae", "rho_rmse", "n_eval", "final_loss", "min_loss", "detail",
)


# ---------------------------------------------------------------------------
# Run-dir readers
# ---------------------------------------------------------------------------

def _load_manifest(run_dir):
    """Load ``manifest.json``; raise FileNotFoundError if absent/corrupt."""
    path = os.path.join(run_dir, "manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no manifest.json in {run_dir} — the preflight has not "
            "materialized this run yet."
        )
    with open(path) as f:
        return json.load(f)


def _spec_dir(run_dir, idx, width):
    return os.path.join(run_dir, "checkpoints", f"spec_{idx:0{width}d}")


def _read_eval_df(csv_path):
    """Return ``(mae, rho_rmse, n_eval)`` from a per-spec eval_df.csv, or None
    if the file is missing/empty/unparseable (treated as not-complete)."""
    try:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return None
    if not rows:
        return None
    r = rows[0]
    try:
        mae = float(r["mae"])
        rho_rmse = float(r["rho_rmse"])
        n_eval = int(float(r["n_eval"]))
    except (KeyError, TypeError, ValueError):
        return None
    return mae, rho_rmse, n_eval


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _read_losses(losses_path):
    """Return ``(final_loss, min_loss)`` from a per-spec losses.npy, or
    ``(None, None)`` if absent/unreadable/empty. A high ``final_loss`` is the
    'training did not converge / diverged' signal that distinguishes an
    undertrained spec from a converged-but-high-MAE (data/metric) one."""
    if not os.path.isfile(losses_path):
        return None, None
    try:
        import numpy as np
        arr = np.load(losses_path)
    except Exception:
        return None, None
    if arr is None or getattr(arr, "size", 0) == 0:
        return None, None
    return float(arr[-1]), float(arr.min())


# ---------------------------------------------------------------------------
# collect_results
# ---------------------------------------------------------------------------

def collect_results(run_dir):
    """Return one result row per manifest spec, in index order.

    Each row is a dict with :data:`ROW_FIELDS`. ``mae``/``rho_rmse``/``n_eval``
    are floats/ints only for ``status == "complete"``; otherwise ``None`` (so
    incomplete specs are visible but excluded from metric statistics).
    """
    manifest = _load_manifest(run_dir)
    width = int(manifest["width"])
    rows = []
    for entry in sorted(manifest.get("specs", []), key=lambda e: e["index"]):
        idx = int(entry["index"])
        cell = entry.get("cell", {}) or {}
        d = _spec_dir(run_dir, idx, width)
        row = {
            "idx": idx,
            "arch": cell.get("arch"),
            "loss": cell.get("loss"),
            "metric": cell.get("metric"),
            "subset_size": cell.get("subset_size"),
            "solver": cell.get("solver"),
            "status": "pending",
            "mae": None,
            "rho_rmse": None,
            "n_eval": None,
            "final_loss": None,
            "min_loss": None,
            "detail": "",
        }

        final_loss, min_loss = _read_losses(os.path.join(d, "losses.npy"))
        row["final_loss"] = final_loss
        row["min_loss"] = min_loss

        eval_df = _read_eval_df(os.path.join(d, "eval_df.csv"))
        skipped = _read_json(os.path.join(d, "eval", "skipped.json"))
        failure = _read_json(os.path.join(d, "failure.json"))
        has_model = os.path.isfile(os.path.join(d, "model.eqx"))

        if eval_df is not None:
            mae, rho_rmse, n_eval = eval_df
            row.update(status="complete", mae=mae, rho_rmse=rho_rmse,
                       n_eval=n_eval)
        elif skipped is not None:
            row.update(status="eval_skipped",
                       detail=str(skipped.get("reason", "")))
        elif failure is not None:
            row.update(status="train_failed",
                       detail=str(failure.get("classification", "")))
        elif has_model:
            row["status"] = "trained_no_eval"
        else:
            row["status"] = "pending"
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def summarize(rows):
    """Aggregate statistics over the COMPLETE rows only.

    Returns a dict with per-status counts and (when >=1 complete) MAE
    min/max/mean/median, mean rho_rmse, and the best/worst-MAE spec indices.
    All MAE fields are ``None`` when no spec has completed (no divide-by-zero).
    """
    status_counts = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    complete = [r for r in rows if r["status"] == "complete"
                and isinstance(r["mae"], (int, float))]
    summary = {
        "n_specs": len(rows),
        "n_complete": len(complete),
        "status_counts": status_counts,
        "mae_min": None, "mae_max": None, "mae_mean": None,
        "mae_median": None, "rho_rmse_mean": None,
        "best_idx": None, "worst_idx": None,
    }
    if complete:
        # Drop NaN from MAE aggregation exactly as rho_rmse is filtered below.
        # isinstance(nan, (int, float)) is True, so the NaN-identity check
        # (x == x is False for NaN) is the reliable guard.
        maes_finite = [r["mae"] for r in complete
                       if r["mae"] == r["mae"]]  # drop NaN
        if maes_finite:
            summary["mae_min"] = min(maes_finite)
            summary["mae_max"] = max(maes_finite)
            summary["mae_mean"] = statistics.fmean(maes_finite)
            summary["mae_median"] = statistics.median(maes_finite)
        # else: all complete specs have NaN MAE; leave mae_* as None.
        rhos = [r["rho_rmse"] for r in complete
                if isinstance(r["rho_rmse"], (int, float))
                and r["rho_rmse"] == r["rho_rmse"]]  # drop NaN
        summary["rho_rmse_mean"] = statistics.fmean(rhos) if rhos else None
        # best/worst: restrict to specs with a finite MAE; skip NaN specs.
        finite_complete = [r for r in complete if r["mae"] == r["mae"]]
        if finite_complete:
            best = min(finite_complete, key=lambda r: r["mae"])
            worst = max(finite_complete, key=lambda r: r["mae"])
            summary["best_idx"] = best["idx"]
            summary["worst_idx"] = worst["idx"]
    return summary


# ---------------------------------------------------------------------------
# Output: table / CSV / plot
# ---------------------------------------------------------------------------

def _fmt(v, nd=4):
    if v is None:
        return "--"
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.{nd}f}"
    return str(v)


# ---------------------------------------------------------------------------
# Per-molecule drill-down ("which molecules performed poorly")
# ---------------------------------------------------------------------------

def load_per_molecule(run_dir, idx):
    """Return the ``eval/per_molecule.json`` rows for spec ``idx``, or None if
    that spec has no per-molecule file yet (eval not done)."""
    manifest = _load_manifest(run_dir)
    width = int(manifest["width"])
    path = os.path.join(_spec_dir(run_dir, idx, width), "eval",
                        "per_molecule.json")
    rows = _read_json(path)
    return rows if isinstance(rows, list) else None


def _abs_ae_err(r):
    """|AE_error_kcalmol| for sorting; rows without a finite key sink to the
    bottom. C5-07: a NaN/inf value passes isinstance(...,(int,float)) and would
    become a NaN sort key (NaN comparisons are all False), letting it rank
    arbitrarily — including above the real worst cases under reverse=True. Treat
    non-finite as 'no AE' so it sinks."""
    e = r.get("AE_error_kcalmol")
    return abs(e) if (isinstance(e, (int, float)) and not isinstance(e, bool)
                      and math.isfinite(e)) else -1.0


def format_per_molecule_table(pm_rows):
    """Per-molecule AE table, worst |error| first. Columns: predicted AE
    (kcal/mol, = AE_nn x HA_TO_KCAL), reference AE, error, density RMSE, and the
    density reference method/skip flag. Atom rows lacking an AE reference render
    with ``--`` rather than crashing."""
    ordered = sorted(pm_rows, key=_abs_ae_err, reverse=True)
    header = ("molecule", "AE_pred", "AE_ref", "AE_error", "dens_rmse", "ref")
    widths = [16, 11, 11, 11, 10, 12]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths))]
    lines.append("  ".join("-" * w for w in widths))
    for r in ordered:
        ae_nn = r.get("AE_nn")
        pred = (ae_nn * HA_TO_KCAL
                if isinstance(ae_nn, (int, float))
                and not isinstance(ae_nn, bool) else None)
        refm = r.get("ref_density_method") or (
            "skipped" if r.get("skipped") else "")
        cells = [
            str(r.get("molecule", "?")), _fmt(pred, 2),
            _fmt(r.get("AE_ref_kcalmol"), 2), _fmt(r.get("AE_error_kcalmol"), 2),
            _fmt(r.get("density_rmse")), refm or "--",
        ]
        lines.append("  ".join(str(c).ljust(w)
                               for c, w in zip(cells, widths)))
    return "\n".join(lines)


def worst_molecules(run_dir, n=20):
    """Across every spec that has a per_molecule.json, the ``n`` molecule
    instances with the largest |AE_error_kcalmol|, each tagged with its spec
    index + grid params. Surfaces whether one molecule is bad everywhere (data/
    reference bug) or only in specific specs."""
    manifest = _load_manifest(run_dir)
    width = int(manifest["width"])
    cell_by_idx = {int(e["index"]): (e.get("cell") or {})
                   for e in manifest.get("specs", [])}
    out = []
    for idx, cell in cell_by_idx.items():
        path = os.path.join(_spec_dir(run_dir, idx, width), "eval",
                            "per_molecule.json")
        rows = _read_json(path)
        if not isinstance(rows, list):
            continue
        for r in rows:
            e = r.get("AE_error_kcalmol")
            # C5-07: skip non-finite errors — a NaN/inf would corrupt the
            # reverse-sorted ranking and surface as a spurious "worst" molecule.
            if (not isinstance(e, (int, float)) or isinstance(e, bool)
                    or not math.isfinite(e)):
                continue
            out.append({
                "molecule": r.get("molecule", "?"),
                "ae_error_kcalmol": float(e),
                "idx": idx,
                "metric": cell.get("metric"),
                "subset_size": cell.get("subset_size"),
                "solver": cell.get("solver"),
            })
    out.sort(key=lambda d: abs(d["ae_error_kcalmol"]), reverse=True)
    return out[:n]


def format_worst_table(rows):
    """Table for :func:`worst_molecules` output."""
    header = ("molecule", "AE_error", "spec", "metric", "subset", "solver")
    widths = [16, 11, 5, 6, 6, 8]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths))]
    lines.append("  ".join("-" * w for w in widths))
    for r in rows:
        cells = [
            str(r["molecule"]), _fmt(r["ae_error_kcalmol"], 2), str(r["idx"]),
            r.get("metric") or "--", _fmt(r.get("subset_size")),
            r.get("solver") or "--",
        ]
        lines.append("  ".join(str(c).ljust(w)
                               for c, w in zip(cells, widths)))
    return "\n".join(lines)


def format_table(rows, summary):
    """Aligned per-spec table (sorted by subset_size, metric, solver) + a
    summary block, as a single string."""
    ordered = sorted(
        rows, key=lambda r: (r["subset_size"] if r["subset_size"] is not None
                             else -1, r["metric"] or "", r["solver"] or "")
    )
    header = ("idx", "subset", "metric", "solver", "MAE", "rho_rmse",
              "n_eval", "fin_loss", "status", "detail")
    widths = [3, 6, 6, 8, 9, 9, 6, 9, 15, 20]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths))]
    lines.append("  ".join("-" * w for w in widths))
    for r in ordered:
        cells = [
            str(r["idx"]), _fmt(r["subset_size"]), r["metric"] or "--",
            r["solver"] or "--", _fmt(r["mae"]), _fmt(r["rho_rmse"]),
            _fmt(r["n_eval"]), _fmt(r.get("final_loss")), r["status"],
            r["detail"] or "",
        ]
        lines.append("  ".join(str(c).ljust(w)
                               for c, w in zip(cells, widths)))

    sc = summary["status_counts"]
    lines.append("")
    lines.append(
        f"specs: {summary['n_specs']}  complete: {summary['n_complete']}  "
        + "  ".join(f"{k}={v}" for k, v in sorted(sc.items())
                    if k != "complete")
    )
    if summary["n_complete"]:
        lines.append(
            f"MAE  min={_fmt(summary['mae_min'])}  "
            f"mean={_fmt(summary['mae_mean'])}  "
            f"median={_fmt(summary['mae_median'])}  "
            f"max={_fmt(summary['mae_max'])}   "
            f"rho_rmse_mean={_fmt(summary['rho_rmse_mean'])}"
        )
        lines.append(
            f"best: spec_{summary['best_idx']} (MAE {_fmt(summary['mae_min'])})"
            f"   worst: spec_{summary['worst_idx']} "
            f"(MAE {_fmt(summary['mae_max'])})"
        )
    else:
        lines.append("MAE: (no completed evals yet)")
    return "\n".join(lines)


def write_csv(rows, path):
    """Write all result rows to ``path`` (one per spec, :data:`ROW_FIELDS`)."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ROW_FIELDS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ROW_FIELDS})
    return path


def plot_mae_vs_subset(rows, path):
    """Plot MAE vs subset_size — one line per (metric, solver) — to ``path``.

    Uses only the COMPLETE rows. matplotlib is imported lazily so the table /
    CSV paths work even without it; a clear error is raised if it is missing.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "plotting requires matplotlib — `pip install matplotlib` or omit "
            "--plot"
        ) from exc

    complete = [r for r in rows if r["status"] == "complete"
                and isinstance(r["mae"], (int, float))]
    series = {}
    for r in complete:
        series.setdefault((r["metric"], r["solver"]), []).append(
            (r["subset_size"], r["mae"])
        )
    fig, ax = plt.subplots(figsize=(7, 5))
    for (metric, solver), pts in sorted(series.items(),
                                        key=lambda kv: str(kv[0])):
        pts.sort(key=lambda p: (p[0] if p[0] is not None else -1))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=f"{metric}/{solver}")
    ax.set_xlabel("subset size")
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_title("Atomization-energy MAE vs subset size")
    if series:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
