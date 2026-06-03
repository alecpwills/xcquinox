"""Tests for xcquinox.alec.cluster.analyze: eval-results aggregation.

A fake run dir is built with one spec per status so the classifier, the
metric aggregation (which must EXCLUDE incomplete specs), and the CSV/plot
writers are all exercised without any real training/eval compute.
"""
import csv
import json
import os

import pytest

from xcquinox.alec.cluster import analyze


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


def test_collect_results_joins_grid_cell(tmp_path):
    rd = _make_run_dir(tmp_path)
    by_idx = {r["idx"]: r for r in analyze.collect_results(rd)}
    assert by_idx[3]["metric"] == "l2"
    assert by_idx[3]["subset_size"] == 2
    assert by_idx[3]["solver"] == "full_3"


def test_collect_results_only_complete_have_metrics(tmp_path):
    rd = _make_run_dir(tmp_path)
    by_idx = {r["idx"]: r for r in analyze.collect_results(rd)}
    assert by_idx[0]["mae"] == 2.0
    assert by_idx[0]["n_eval"] == 3
    # Every non-complete spec has None metrics (NOT 0, excluded from stats).
    for i in (2, 3, 4, 5):
        assert by_idx[i]["mae"] is None
        assert by_idx[i]["rho_rmse"] is None
        assert by_idx[i]["n_eval"] is None


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


def test_summarize_no_complete_specs_is_safe(tmp_path):
    """A run with zero finished evals must not divide-by-zero."""
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump({"width": _WIDTH, "n_specs": 1, "specs": [
            {"index": 0, "cell": {"arch": "a", "loss": "l", "metric": "l2",
                                  "subset_size": 2, "solver": "oneshot"},
             "spec_file": "spec_0000.spec"}]}, f)
    _spec_dir(run_dir, 0)  # pending
    summary = analyze.summarize(analyze.collect_results(run_dir))
    assert summary["n_complete"] == 0
    assert summary["mae_mean"] is None


# ---------------------------------------------------------------------------
# format_table / write_csv / plot
# ---------------------------------------------------------------------------

def test_format_table_renders_and_excludes_incomplete_from_summary(tmp_path):
    rd = _make_run_dir(tmp_path)
    rows = analyze.collect_results(rd)
    text = analyze.format_table(rows, analyze.summarize(rows))
    assert "MAE" in text and "status" in text
    # complete count + a status tally line are present.
    assert "complete: 2" in text
    assert "best: spec_1" in text  # lowest MAE


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


def test_plot_mae_vs_subset_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    rd = _make_run_dir(tmp_path)
    rows = analyze.collect_results(rd)
    out = str(tmp_path / "mae.png")
    analyze.plot_mae_vs_subset(rows, out)
    assert os.path.isfile(out) and os.path.getsize(out) > 0


# ---------------------------------------------------------------------------
# Per-molecule drill-down + loss convergence + worst-molecules
# ---------------------------------------------------------------------------

def _write_per_molecule(run_dir, idx, rows):
    """Write checkpoints/spec_<idx>/eval/per_molecule.json."""
    ed = os.path.join(_spec_dir(run_dir, idx), "eval")
    os.makedirs(ed, exist_ok=True)
    with open(os.path.join(ed, "per_molecule.json"), "w") as f:
        json.dump(rows, f)


def _write_losses(run_dir, idx, losses):
    import numpy as np
    np.save(os.path.join(_spec_dir(run_dir, idx), "losses.npy"),
            np.array(losses, dtype=np.float64))


def test_load_per_molecule_reads_rows(tmp_path):
    rd = _make_run_dir(tmp_path)
    _write_per_molecule(rd, 0, [
        {"molecule": "H2O", "AE_nn": 0.37, "AE_ref_kcalmol": 232.2,
         "AE_error_kcalmol": -0.1, "density_rmse": 0.001,
         "ref_density_method": "ccsd"},
        {"molecule": "F2O", "AE_nn": -0.14, "AE_ref_kcalmol": 53.7,
         "AE_error_kcalmol": -141.6, "density_rmse": 0.002,
         "ref_density_method": "ccsd"},
    ])
    rows = analyze.load_per_molecule(rd, 0)
    assert rows is not None and len(rows) == 2
    # a spec with no per_molecule.json -> None
    assert analyze.load_per_molecule(rd, 5) is None


def test_format_per_molecule_sorts_by_abs_error_and_converts_pred(tmp_path):
    rd = _make_run_dir(tmp_path)
    _write_per_molecule(rd, 0, [
        {"molecule": "H2O", "AE_nn": 0.37, "AE_ref_kcalmol": 232.2,
         "AE_error_kcalmol": -0.1, "density_rmse": 0.001},
        {"molecule": "F2O", "AE_nn": -0.14, "AE_ref_kcalmol": 53.7,
         "AE_error_kcalmol": -141.6, "density_rmse": 0.002},
        {"molecule": "Li", "AE_nn": 0.0, "density_rmse": 0.0, "skipped": True},
    ])
    text = analyze.format_per_molecule_table(analyze.load_per_molecule(rd, 0))
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("-")]
    # worst (|err| largest) first after the header.
    body = lines[1:]
    assert body[0].split()[0] == "F2O"
    assert body[1].split()[0] == "H2O"
    # AE_pred shown in kcal/mol: H2O AE_nn 0.37 Ha * 627.5 ~ 232.2
    assert "232." in text  # H2O predicted ~232 kcal/mol
    # atom row with no AE_ref renders without crashing.
    assert "Li" in text


def test_collect_results_includes_loss_columns(tmp_path):
    rd = _make_run_dir(tmp_path)
    _write_losses(rd, 0, [5.0, 2.0, 1.0, 1.5])  # final 1.5, min 1.0
    by_idx = {r["idx"]: r for r in analyze.collect_results(rd)}
    assert by_idx[0]["final_loss"] == pytest.approx(1.5)
    assert by_idx[0]["min_loss"] == pytest.approx(1.0)
    # a spec without losses.npy -> None
    assert by_idx[1]["final_loss"] is None


def test_summarize_nan_mae_excluded_from_statistics(tmp_path):
    """CODE-01: a complete spec with mae=nan must NOT poison MAE statistics.

    Setup: two complete specs with finite MAE (1.0, 3.0) plus one complete
    spec whose mae is NaN (BH76/IP13-only subset -- no AE-reference compound
    contributes to the average, so _aggregate_per_molecule writes nan).

    Assertions:
      (a) mae_mean/min/max/median are computed over the two finite-MAE specs
          only (mean=2.0, min=1.0, max=3.0, median=2.0).
      (b) best_idx/worst_idx ignore the NaN spec and point at the finite ones.
      (c) n_complete still counts the NaN spec (it is complete; the field
          just cannot contribute to the metric aggregation).
    """
    run_dir = str(tmp_path / "run_nan")
    os.makedirs(run_dir)
    cells = [
        {"index": 0, "cell": {"arch": "a", "loss": "l", "metric": "l2",
                               "subset_size": 2, "solver": "oneshot"},
         "spec_file": "spec_0000.spec"},
        {"index": 1, "cell": {"arch": "a", "loss": "l", "metric": "l2",
                               "subset_size": 4, "solver": "oneshot"},
         "spec_file": "spec_0001.spec"},
        {"index": 2, "cell": {"arch": "a", "loss": "l", "metric": "bh76",
                               "subset_size": 4, "solver": "oneshot"},
         "spec_file": "spec_0002.spec"},
    ]
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump({"width": 4, "n_specs": 3, "specs": cells}, f)

    # idx 0 -> finite MAE 1.0
    _write_eval_df(run_dir, 0, mae=1.0)
    # idx 1 -> finite MAE 3.0
    _write_eval_df(run_dir, 1, mae=3.0)
    # idx 2 -> NaN MAE (BH76-only subset: no AE-ref compound)
    _write_eval_df(run_dir, 2, mae=float("nan"))

    rows = analyze.collect_results(run_dir)
    summary = analyze.summarize(rows)

    # All three are status=="complete".
    assert summary["n_complete"] == 3

    # (a) metric aggregation uses only the two finite-MAE specs.
    assert summary["mae_min"] == pytest.approx(1.0)
    assert summary["mae_max"] == pytest.approx(3.0)
    assert summary["mae_mean"] == pytest.approx(2.0)
    assert summary["mae_median"] == pytest.approx(2.0)

    # (b) best/worst must NOT be the NaN spec (idx 2).
    assert summary["best_idx"] == 0   # MAE 1.0
    assert summary["worst_idx"] == 1  # MAE 3.0


def test_summarize_all_nan_mae_leaves_fields_none(tmp_path):
    """CODE-01: when every complete spec has nan MAE, mae_* fields must be None."""
    run_dir = str(tmp_path / "run_all_nan")
    os.makedirs(run_dir)
    cells = [
        {"index": 0, "cell": {"arch": "a", "loss": "l", "metric": "bh76",
                               "subset_size": 2, "solver": "oneshot"},
         "spec_file": "spec_0000.spec"},
    ]
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump({"width": 4, "n_specs": 1, "specs": cells}, f)
    _write_eval_df(run_dir, 0, mae=float("nan"))

    rows = analyze.collect_results(run_dir)
    summary = analyze.summarize(rows)

    assert summary["n_complete"] == 1
    assert summary["mae_min"] is None
    assert summary["mae_max"] is None
    assert summary["mae_mean"] is None
    assert summary["mae_median"] is None
    assert summary["best_idx"] is None
    assert summary["worst_idx"] is None


def test_worst_molecules_ranks_across_complete_specs(tmp_path):
    rd = _make_run_dir(tmp_path)
    # idx 0 and 1 are complete (eval_df.csv from _make_run_dir).
    _write_per_molecule(rd, 0, [
        {"molecule": "F2O", "AE_nn": -0.1, "AE_ref_kcalmol": 53.7,
         "AE_error_kcalmol": -141.6},
        {"molecule": "H2O", "AE_nn": 0.37, "AE_ref_kcalmol": 232.2,
         "AE_error_kcalmol": -0.1},
    ])
    _write_per_molecule(rd, 1, [
        {"molecule": "N2", "AE_nn": 0.3, "AE_ref_kcalmol": 228.0,
         "AE_error_kcalmol": 40.0},
    ])
    worst = analyze.worst_molecules(rd, 2)
    assert len(worst) == 2
    assert worst[0]["molecule"] == "F2O"
    assert worst[0]["idx"] == 0
    assert abs(worst[0]["ae_error_kcalmol"]) > abs(worst[1]["ae_error_kcalmol"])


# non-finite AE errors must sink in the ranking, never rank as "worst"
def test_abs_ae_err_sinks_nonfinite():
    assert analyze._abs_ae_err({"AE_error_kcalmol": -5.0}) == 5.0
    assert analyze._abs_ae_err({"AE_error_kcalmol": float("nan")}) == -1.0
    assert analyze._abs_ae_err({"AE_error_kcalmol": float("inf")}) == -1.0
    assert analyze._abs_ae_err({}) == -1.0
    # a NaN row must NOT sort above a real worst case under reverse=True
    rows = [
        {"AE_error_kcalmol": float("nan"), "molecule": "bad"},
        {"AE_error_kcalmol": -10.0, "molecule": "worst"},
        {"AE_error_kcalmol": 2.0, "molecule": "ok"},
    ]
    ordered = sorted(rows, key=analyze._abs_ae_err, reverse=True)
    assert ordered[0]["molecule"] == "worst"
    assert ordered[-1]["molecule"] == "bad"
