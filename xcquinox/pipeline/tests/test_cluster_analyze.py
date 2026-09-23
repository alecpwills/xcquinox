"""Tests for xcquinox.pipeline.cluster.analyze: eval-results aggregation.

A fake run dir is built with one spec per status so the classifier, the
metric aggregation (which must EXCLUDE incomplete specs), and the CSV/plot
writers are all exercised without any real training/eval compute.
"""
import csv
import json
import os

import pytest

from xcquinox.pipeline.cluster import analyze


_WIDTH = 4

# idx -> (metric, subset_size, solver) for the 6-cell fake grid.
_CELLS = {
    0: ("l2", 2, "oneshot"),
    1: ("l2", 4, "oneshot"),
    2: ("jsd", 2, "oneshot"),
    3: ("l2", 2, "full_3"),
    4: ("jsd", 4, "oneshot"),
    5: ("jsd", 4, "full_3"),
}


def _spec_dir(run_dir, idx):
    d = os.path.join(run_dir, "checkpoints", f"spec_{idx:0{_WIDTH}d}")
    os.makedirs(d, exist_ok=True)
    return d


def _write_eval_df(run_dir, idx, mae, rho_rmse=0.01, n_eval=3):
    """Write a per-spec eval_df.csv exactly as _eval_one_spec does."""
    d = _spec_dir(run_dir, idx)
    with open(os.path.join(d, "eval_df.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "mae", "rho_rmse", "n_eval"])
        w.writeheader()
        w.writerow({"set": "training_subset", "mae": mae,
                    "rho_rmse": rho_rmse, "n_eval": n_eval})


def _make_run_dir(tmp_path):
    """A run dir whose 6 specs span every status the analyzer reports."""
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir)
    # manifest.json
    specs = []
    for idx, (metric, ss, solver) in _CELLS.items():
        specs.append({
            "index": idx,
            "cell": {"arch": "deep_combined_attn", "loss": "L5_step7",
                     "metric": metric, "subset_size": ss, "solver": solver},
            "spec_file": f"spec_{idx:0{_WIDTH}d}.spec",
        })
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump({"width": _WIDTH, "n_specs": len(_CELLS), "specs": specs}, f)

    # idx 0,1 -> complete (eval_df.csv)
    _write_eval_df(run_dir, 0, mae=2.0)
    _write_eval_df(run_dir, 1, mae=1.0)
    # idx 2 -> eval_skipped
    ed = os.path.join(_spec_dir(run_dir, 2), "eval")
    os.makedirs(ed, exist_ok=True)
    with open(os.path.join(ed, "skipped.json"), "w") as f:
        json.dump({"reason": "no model.eqx", "timestamp": "t"}, f)
    # idx 3 -> train_failed
    with open(os.path.join(_spec_dir(run_dir, 3), "failure.json"), "w") as f:
        json.dump({"classification": "timeout"}, f)
    # idx 4 -> trained_no_eval (model.eqx only)
    open(os.path.join(_spec_dir(run_dir, 4), "model.eqx"), "wb").close()
    # idx 5 -> pending (dir exists, nothing in it)
    _spec_dir(run_dir, 5)
    return run_dir


# ---------------------------------------------------------------------------
# collect_results
# ---------------------------------------------------------------------------

def test_collect_results_classifies_every_status(tmp_path):
    rd = _make_run_dir(tmp_path)
    rows = analyze.collect_results(rd)
    by_idx = {r["idx"]: r for r in rows}
    assert len(rows) == 6
    assert by_idx[0]["status"] == "complete"
    assert by_idx[1]["status"] == "complete"
    assert by_idx[2]["status"] == "eval_skipped"
    assert by_idx[3]["status"] == "train_failed"
    assert by_idx[4]["status"] == "trained_no_eval"
    assert by_idx[5]["status"] == "pending"


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_mae_over_complete_only(tmp_path):
    rd = _make_run_dir(tmp_path)
    summary = analyze.summarize(analyze.collect_results(rd))
    # Only idx 0 (2.0) and idx 1 (1.0) are complete.
    assert summary["n_complete"] == 2
    assert summary["mae_min"] == pytest.approx(1.0)
    assert summary["mae_max"] == pytest.approx(2.0)
    assert summary["mae_mean"] == pytest.approx(1.5)
    assert summary["mae_median"] == pytest.approx(1.5)
    assert summary["best_idx"] == 1   # lowest MAE
    assert summary["worst_idx"] == 0
    # status tally covers all six.
    assert summary["status_counts"]["complete"] == 2
    assert summary["status_counts"]["eval_skipped"] == 1
    assert summary["status_counts"]["train_failed"] == 1
    assert summary["status_counts"]["trained_no_eval"] == 1
    assert summary["status_counts"]["pending"] == 1


# ---------------------------------------------------------------------------
# format_table / write_csv / plot
# ---------------------------------------------------------------------------


def test_write_csv_one_row_per_spec(tmp_path):
    rd = _make_run_dir(tmp_path)
    rows = analyze.collect_results(rd)
    out = str(tmp_path / "results.csv")
    analyze.write_csv(rows, out)
    with open(out, newline="") as f:
        got = list(csv.DictReader(f))
    assert len(got) == 6
    # complete row carries MAE; incomplete row's mae cell is empty.
    by_idx = {int(r["idx"]): r for r in got}
    assert by_idx[0]["mae"] == "2.0"
    assert by_idx[5]["mae"] == ""          # pending -> None -> blank
    assert by_idx[3]["status"] == "train_failed"
    assert by_idx[3]["detail"] == "timeout"


# ---------------------------------------------------------------------------
# Per-molecule drill-down + loss convergence + worst-molecules
# ---------------------------------------------------------------------------


# non-finite AE errors must sink in the ranking, never rank as "worst"
