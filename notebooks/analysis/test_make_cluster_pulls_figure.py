"""Tests for ``notebooks/analysis/make_cluster_pulls_figure.py``.

Two layers:

  - Pure data-ingest tests (no matplotlib needed) confirming the
    manifest-cell join, status grid, and category discovery.
  - Render canary tests that drive each plot builder against a small
    in-memory fixture; assert the PNG file exists and is non-trivially
    sized. Skipped iff matplotlib is unavailable.

Match the style of ``xcquinox/alec/tests/test_cluster_*.py``: pure-stdlib
fixtures, ``tmp_path``, no factory helpers.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

# Load the figure script as a module without requiring a package layout.
_FIG_PATH = (Path(__file__).resolve().parent / "make_cluster_pulls_figure.py")
_spec = importlib.util.spec_from_file_location("make_cluster_pulls_figure",
                                                _FIG_PATH)
fig_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules[_spec.name] = fig_mod  # type: ignore[union-attr]
_spec.loader.exec_module(fig_mod)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_GOOD_STAMP = "run_20260525T163822Z"


def _make_run_dir(root: Path, category: str = "alpha_on/runs",
                  stamp: str = _GOOD_STAMP, *,
                  n_specs: int = 4,
                  with_manifest: bool = True,
                  with_eval: bool = True) -> Path:
    """Build a tmp run-dir tree that mirrors the harness layout closely
    enough for the figure script's ingest helpers."""
    run_dir = root / category / stamp
    run_dir.mkdir(parents=True)

    cells = [
        {"arch": "deep_combined_attn", "loss": "L5_test",
         "metric": "jsd" if (i % 4) < 2 else "l2",
         "solver": "full_3" if (i % 2 == 0) else "oneshot",
         "subset_size": [1, 2, 3, 4][i % 4]}
        for i in range(n_specs)
    ]

    if with_manifest:
        manifest = {
            "n_specs": n_specs, "width": 4,
            "specs": [{"index": i, "spec_file": f"spec_{i:04d}.spec",
                       "sha256": "x" * 64, "cell": cells[i]}
                      for i in range(n_specs)],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

    (run_dir / "resolved_config.yaml").write_text("sweep: {}\n")

    for i in range(n_specs):
        sd = run_dir / "checkpoints" / f"spec_{i:04d}"
        sd.mkdir(parents=True)
        if not with_eval:
            continue
        # Skip eval for the LAST spec to mimic an in-flight harness.
        if i == n_specs - 1:
            continue
        (sd / "eval_df.csv").write_text(
            "set,mae,rho_rmse,n_eval\n"
            f"training_subset,{0.1 + 0.5 * i},{1e-4 * (i + 1)},{i + 1}\n"
        )
        (sd / "model.eqx").write_bytes(b"x" * 16)
        ev = sd / "eval"; ev.mkdir()
        ev.write_text  # noop — just to silence the linter
        (ev / "per_molecule.json").write_text(json.dumps([
            {"molecule": "H2O", "AE_error_kcalmol": 1.5 + i,
             "density_rmse": 5e-4 * (i + 1), "scf_converged": True},
            {"molecule": "CH4", "AE_error_kcalmol": 0.8 + 0.2 * i,
             "density_rmse": 7e-4 * (i + 1), "scf_converged": True},
            {"molecule": "H", "skipped": True, "skip_reason": "atomic_system",
             "density_rmse": None, "scf_converged": True},
        ]))
    return run_dir


# ---------------------------------------------------------------------------
# Pure-data tests
# ---------------------------------------------------------------------------

def test_discover_pulled_categories_finds_multiple_categories(tmp_path):
    _make_run_dir(tmp_path, "alpha_on/runs", "run_20260525T163822Z")
    _make_run_dir(tmp_path, "polarized/alpha_off/runs",
                  "run_20260526T180922Z", n_specs=2)
    cats = fig_mod.discover_pulled_categories(tmp_path)
    assert set(cats) == {"alpha_on/runs", "polarized/alpha_off/runs"}
    assert cats["alpha_on/runs"].name == "run_20260525T163822Z"


def test_discover_pulled_categories_picks_latest_per_category(tmp_path):
    _make_run_dir(tmp_path, "alpha_on/runs", "run_20260525T100000Z")
    _make_run_dir(tmp_path, "alpha_on/runs", "run_20260601T200000Z")
    cats = fig_mod.discover_pulled_categories(tmp_path)
    assert cats["alpha_on/runs"].name == "run_20260601T200000Z"


def test_collect_eval_df_rows_joins_manifest_cell(tmp_path):
    run = _make_run_dir(tmp_path, "alpha_on/runs", _GOOD_STAMP, n_specs=3)
    rows = fig_mod.collect_eval_df_rows(run)
    # 3 specs but the last one has no eval_df.csv (per fixture).
    assert len(rows) == 2
    # Every row carries the manifest cell + set/MAE.
    for r in rows:
        assert r["arch"] == "deep_combined_attn"
        assert r["loss"] == "L5_test"
        assert r["metric"] in ("jsd", "l2")
        assert r["solver"] in ("full_3", "oneshot")
        assert r["subset_size"] in (1, 2, 3, 4)
        assert r["set"] == "training_subset"
        assert r["mae"] is not None


def test_collect_eval_df_rows_skips_specs_without_csv(tmp_path):
    run = _make_run_dir(tmp_path, "alpha_on/runs", _GOOD_STAMP, n_specs=3)
    # Fixture leaves spec_0002 without eval_df.csv.
    rows = fig_mod.collect_eval_df_rows(run)
    spec_indices = {r["idx"] for r in rows}
    assert 2 not in spec_indices
    assert spec_indices == {0, 1}


def test_collect_per_molecule_rows_keeps_skipped_atoms(tmp_path):
    run = _make_run_dir(tmp_path, "alpha_on/runs", _GOOD_STAMP, n_specs=2)
    rows = fig_mod.collect_per_molecule_rows(run)
    h_rows = [r for r in rows if r["molecule"] == "H"]
    assert len(h_rows) == 1  # only spec_0000 has it, spec_0001 no eval
    h = h_rows[0]
    assert h["skipped"] is True
    assert h["density_rmse"] is None
    # And the cell fields still got joined.
    assert h["arch"] == "deep_combined_attn"


def test_aggregate_status_grid_pivots_on_metric_solver(tmp_path):
    run = _make_run_dir(tmp_path, "alpha_on/runs", _GOOD_STAMP, n_specs=4)
    grid = fig_mod.aggregate_status_grid(run)
    # Keys are (metric, solver) tuples; values are {subset_size: status}.
    assert all(isinstance(k, tuple) and len(k) == 2 for k in grid)
    # The cells the fixture generated must have a status from collect_results.
    for k, v in grid.items():
        for ss, st in v.items():
            assert isinstance(ss, int) or ss is None
            assert st in ("complete", "trained_no_eval", "train_failed",
                          "eval_skipped", "pending")


def test_load_final_losses_handles_missing_run(tmp_path):
    # Empty tmp dir -> no losses, no error.
    assert fig_mod.load_final_losses(tmp_path) == {}


def test_collect_local_test_set_rows_joins_manifest_cell(tmp_path):
    """The new local_test_set.csv ingest must join with manifest cell + strip
    the leading 'test_set_' from the pool token, mirroring the schema the
    figure-script overlay consumes. Reads the NEW schema with separate NN
    and PBE columns."""
    run = _make_run_dir(tmp_path, "alpha_on/runs", _GOOD_STAMP, n_specs=2)
    csv_path = (run / "checkpoints" / "spec_0000" / "local_test_set.csv")
    csv_path.write_text(
        "set,mae_nn_kcalmol,mae_pbe_kcalmol,delta_nn_minus_pbe,"
        "n_reactions,n_dropped_overlap,note\n"
        "test_set_bh76,2.5,8.1,-5.6,6,0,loose\n"
        "test_set_w411,3.7,10.4,-6.7,10,1,strict; 1 reactions dropped\n"
        "test_set_held_out_combined,3.2,9.3,-6.1,16,1,combined (loose)\n"
    )
    rows = fig_mod.collect_local_test_set_rows(run)
    assert len(rows) == 3
    by_pool = {r["pool"]: r for r in rows}
    assert set(by_pool) == {"bh76", "w411", "held_out_combined"}
    assert all(r["arch"] == "deep_combined_attn" for r in rows)
    # Both NN and PBE columns are surfaced; the legacy mae_kcalmol key
    # mirrors mae_nn_kcalmol so older readers still work.
    assert by_pool["bh76"]["mae_nn_kcalmol"] == pytest.approx(2.5)
    assert by_pool["bh76"]["mae_pbe_kcalmol"] == pytest.approx(8.1)
    assert by_pool["bh76"]["mae_kcalmol"] == pytest.approx(2.5)
    assert by_pool["bh76"]["delta_nn_minus_pbe"] == pytest.approx(-5.6)
    assert by_pool["w411"]["n_dropped_overlap"] == 1
    assert by_pool["held_out_combined"]["mae_nn_kcalmol"] == pytest.approx(3.2)
    assert by_pool["held_out_combined"]["mae_pbe_kcalmol"] == pytest.approx(9.3)


_PLT_OK = importlib.util.find_spec("matplotlib") is not None


def _make_fake_local_rows(n_specs: int = 8) -> list:
    """Synthetic local_test_set rows for the 4 new render canaries."""
    rows = []
    for idx in range(n_specs):
        for pool, (mae_nn, mae_pbe) in (("bh76", (5.0 + idx * 0.7, 8.077)),
                                         ("w411", (12.0 + idx * 1.1, 10.450)),
                                         ("held_out_combined",
                                          (9.5 + idx * 0.9, 9.560))):
            rows.append({
                "idx": idx,
                "arch": "deep_combined_attn", "loss": "L5_test",
                "metric": "jsd" if idx % 2 == 0 else "l2",
                "subset_size": 1 + (idx % 4),
                "solver": "full_3" if idx % 3 == 0 else "oneshot",
                "pool": pool,
                "mae_kcalmol": mae_nn,
                "mae_nn_kcalmol": mae_nn,
                "mae_pbe_kcalmol": mae_pbe,
                "delta_nn_minus_pbe": mae_nn - mae_pbe,
                "n_reactions": 6 if pool == "bh76" else 10 if pool == "w411" else 16,
                "n_dropped_overlap": 0,
                "note": "loose",
            })
    return rows


def _make_fake_per_reaction_rows(n_specs: int = 8) -> list:
    """Synthetic per_reaction rows covering 3 BH76 and 3 W4-11 reactions."""
    rxns = [("OH+H2_to_H2O+H", "bh76", 12.0),
            ("H+HCl_to_H2+Cl", "bh76", 3.0),
            ("CH3+H2_to_CH4+H", "bh76", 8.0),
            ("AE_h2o", "w411", 15.0),
            ("AE_ch4", "w411", 9.0),
            ("AE_co2", "w411", 22.0)]
    rows = []
    for idx in range(n_specs):
        for name, pool, ref in rxns:
            nn_err = ref + 2 * (idx % 3) - 4
            pbe_err = ref - 1
            rows.append({
                "idx": idx,
                "arch": "deep_combined_attn", "loss": "L5_test",
                "metric": "jsd" if idx % 2 == 0 else "l2",
                "subset_size": 1 + (idx % 4),
                "solver": "full_3" if idx % 3 == 0 else "oneshot",
                "name": name, "pool": pool,
                "ref_kcalmol": ref,
                "de_nn_kcalmol": ref + nn_err,
                "de_pbe_kcalmol": ref + pbe_err,
                "error_nn_kcalmol": nn_err,
                "error_pbe_kcalmol": pbe_err,
                "abs_error_nn_kcalmol": abs(nn_err),
                "abs_error_pbe_kcalmol": abs(pbe_err),
                "in_sample_overlap": [],
            })
    return rows


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_plot_nn_vs_pbe_renders(tmp_path):
    out = tmp_path / "fig5.png"
    fig_mod.plot_nn_vs_pbe(
        {"alpha_on/runs": _make_fake_local_rows()},
        "alpha_on/runs", out,
    )
    assert out.is_file()
    assert out.stat().st_size > 10_000


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_plot_per_pool_renders(tmp_path):
    out = tmp_path / "fig6.png"
    fig_mod.plot_per_pool(
        {"alpha_on/runs": _make_fake_local_rows()},
        "alpha_on/runs", out,
    )
    assert out.is_file()
    assert out.stat().st_size > 10_000


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_plot_grid_heatmap_renders(tmp_path):
    out = tmp_path / "fig7.png"
    fig_mod.plot_grid_heatmap(
        {"alpha_on/runs": _make_fake_local_rows()},
        "alpha_on/runs", out,
    )
    assert out.is_file()
    assert out.stat().st_size > 10_000


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_plot_per_reaction_renders(tmp_path):
    out = tmp_path / "fig8.png"
    fig_mod.plot_per_reaction(
        {"alpha_on/runs": _make_fake_per_reaction_rows()},
        "alpha_on/runs", out,
    )
    assert out.is_file()
    assert out.stat().st_size > 10_000


def _make_fake_eval_rows(n_specs: int = 8) -> list:
    """Synthetic cluster eval_df rows for the in-sample vs held-out test."""
    return [{"idx": idx, "set": "training_subset",
             "arch": "deep_combined_attn", "loss": "L5_test",
             "metric": "jsd" if idx % 2 == 0 else "l2",
             "subset_size": 1 + (idx % 4),
             "solver": "full_3" if idx % 3 == 0 else "oneshot",
             "mae": 0.01 + 0.5 * idx,
             "rho_rmse": 1e-4, "n_eval": 4}
            for idx in range(n_specs)]


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_plot_subset_size_correlation_renders_xc(tmp_path):
    out = tmp_path / "fig9.png"
    fig_mod.plot_subset_size_correlation(
        {"alpha_on/runs": _make_fake_local_rows(),
         "alpha_off/runs": _make_fake_local_rows()},
        "alpha_on/runs", out,
    )
    assert out.is_file()
    assert out.stat().st_size > 10_000


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_plot_in_sample_vs_held_out_renders(tmp_path):
    out = tmp_path / "fig11.png"
    fig_mod.plot_in_sample_vs_held_out(
        {"alpha_on/runs": _make_fake_eval_rows()},
        {"alpha_on/runs": _make_fake_local_rows()},
        "alpha_on/runs", out,
    )
    assert out.is_file()
    assert out.stat().st_size > 10_000


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_plot_in_sample_vs_held_out_cross_category(tmp_path):
    """Multi-category overlay: at least one curve per category in legend."""
    out = tmp_path / "fig11_xc.png"
    fig_mod.plot_in_sample_vs_held_out(
        {"alpha_on/runs": _make_fake_eval_rows(n_specs=6),
         "polarized/alpha_off/runs": _make_fake_eval_rows(n_specs=5)},
        {"alpha_on/runs": _make_fake_local_rows(n_specs=6),
         "polarized/alpha_off/runs": _make_fake_local_rows(n_specs=5)},
        "alpha_on/runs", out,
    )
    assert out.is_file()
    assert out.stat().st_size > 10_000


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_plot_in_sample_vs_held_out_keeps_small_subset_specs(tmp_path,
                                                              monkeypatch):
    """Regression (2026-05-29 update): the prior behavior CLAMPED the x-axis
    floor to 0.32 so subset_size=1 trivial-overfit specs were clipped out of
    fig11. The user explicitly requested those specs stay visible — they're
    descriptive (memorization signal), even if they sit far above the
    diagonal. This test confirms the lower axis limit now stretches DOWN to
    where the actual data lives (not clamped at 0.32).
    """
    # Inject one trivial-overfit spec into the otherwise-normal fake rows.
    eval_rows = _make_fake_eval_rows(n_specs=6)
    eval_rows.insert(0, {
        "idx": 999, "set": "training_subset",
        "mae": 1e-4, "rho_rmse": 1e-6, "n_eval": 1,
        "arch": "deep_combined_attn", "loss": "L5_test",
        "metric": "jsd", "solver": "full_3", "subset_size": 1,
    })
    local_rows = _make_fake_local_rows(n_specs=6)
    local_rows.insert(0, {
        "idx": 999, "pool": "held_out_combined",
        "mae_nn_kcalmol": 10.0, "mae_pbe_kcalmol": 9.5,
        "mae_kcalmol": 10.0, "delta_nn_minus_pbe": 0.5,
        "n_reactions": 16, "n_dropped_overlap": 0,
        "arch": "deep_combined_attn", "loss": "L5_test",
        "metric": "jsd", "solver": "full_3", "subset_size": 1,
        "note": "loose",
    })
    out = tmp_path / "fig11_visible.png"
    # Use the public API but inspect via matplotlib's saved Figure state by
    # monkeypatching savefig to capture xlim.
    captured: dict = {}
    real_savefig = fig_mod.plt.Figure.savefig

    def _spy(self, *a, **kw):
        if self.axes:
            captured["xlim"] = self.axes[0].get_xlim()
        return real_savefig(self, *a, **kw)

    monkeypatch.setattr(fig_mod.plt.Figure, "savefig", _spy)
    fig_mod.plot_in_sample_vs_held_out(
        {"alpha_on/runs": eval_rows},
        {"alpha_on/runs": local_rows},
        "alpha_on/runs", out,
    )
    assert out.is_file()
    assert "xlim" in captured, "savefig spy did not fire"
    lo, hi = captured["xlim"]
    # The 1e-4 outlier MUST pull the lower limit below the prior 0.32 floor
    # so the spec is visible (lim_min = max(min_positive_x * 0.5, 1e-5)).
    assert lo < 0.32, (
        f"x-axis lower limit {lo:g} is still >= 0.32 — the subset_size=1 "
        f"trivial-overfit spec is being clipped out of fig11. The user "
        f"explicitly requested 2026-05-29 that these specs stay visible."
    )
    # But the floor must not be ridiculous either; lower bound at 1e-5.
    assert lo >= 1e-5, (
        f"x-axis lower limit {lo:g} is below the 1e-5 absolute floor — the "
        f"axis would stretch across too many empty decades."
    )


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_plot_per_reaction_vs_subset_renders(tmp_path):
    out = tmp_path / "fig12.png"
    fig_mod.plot_per_reaction_vs_subset(
        {"alpha_on/runs": _make_fake_per_reaction_rows()},
        "alpha_on/runs", out,
    )
    assert out.is_file()
    assert out.stat().st_size > 10_000


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_plot_best_vs_worst_per_reaction_renders(tmp_path):
    out = tmp_path / "fig13.png"
    fig_mod.plot_best_vs_worst_per_reaction(
        {"alpha_on/runs": _make_fake_per_reaction_rows()},
        {"alpha_on/runs": _make_fake_local_rows()},
        "alpha_on/runs", out,
    )
    assert out.is_file()
    assert out.stat().st_size > 10_000


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_plot_density_vs_energy_by_cell_renders(tmp_path):
    out = tmp_path / "fig14.png"
    # Synthetic per_molecule rows with density_rmse + AE_error in each cell.
    per_mol = []
    for idx in range(8):
        for mol in ("H2O", "CH4", "CO2"):
            per_mol.append({
                "idx": idx, "metric": "jsd" if idx % 2 == 0 else "l2",
                "solver": "full_3" if idx % 3 == 0 else "oneshot",
                "subset_size": 1 + (idx % 4),
                "molecule": mol,
                "AE_error_kcalmol": 1.5 + idx * 0.2,
                "density_rmse": 1e-4 * (idx + 1),
            })
    fig_mod.plot_density_vs_energy_by_cell(
        {"alpha_on/runs": per_mol}, "alpha_on/runs", out,
    )
    assert out.is_file()
    assert out.stat().st_size > 10_000


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_plot_cross_category_nn_vs_pbe_renders(tmp_path):
    """Fig 16 canary — feed 3 fake categories' local rows and assert the
    figure renders. Empty-category cases handled by the builder, but the
    canary stress-tests the populated path."""
    rows_by_cat = {
        "alpha_on/runs":            _make_fake_local_rows(n_specs=6),
        "alpha_off/runs":           _make_fake_local_rows(n_specs=4),
        "polarized/alpha_off/runs": _make_fake_local_rows(n_specs=5),
    }
    out = tmp_path / "fig16.png"
    fig_mod.plot_cross_category_nn_vs_pbe(rows_by_cat, out)
    assert out.is_file()
    assert out.stat().st_size > 10_000


def test_plot_cross_category_nn_vs_pbe_handles_empty():
    """No categories with data → still renders a single-message figure."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "fig16_empty.png"
        fig_mod.plot_cross_category_nn_vs_pbe({}, out)
        assert out.is_file()


def _make_fake_descriptor_rows(n_specs: int = 8) -> list:
    """Synthetic descriptor rows mimicking local_subset_descriptors.json
    payloads. 3 training molecules per spec, 5 descriptor features.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    rows = []
    for idx in range(n_specs):
        n_mol = 3 + (idx % 4)
        pm = rng.uniform(-1.0, 1.0, size=(n_mol, 5))
        rows.append({
            "idx": idx,
            "arch": "deep_combined_attn", "loss": "L5_test",
            "metric": "jsd" if idx % 2 == 0 else "l2",
            "subset_size": 1 + (idx % 4),
            "solver": "full_3" if idx % 3 == 0 else "oneshot",
            "training_molecule_names": [f"mol_{i}" for i in range(n_mol)],
            "feature_names": ["DMStatisticsDescriptor_0",
                              "DMStatisticsDescriptor_1",
                              "DMStatisticsDescriptor_2",
                              "CuspDescriptor_0", "CuspDescriptor_1"],
            "per_molecule_features": pm,
            "per_subset_stats": {
                "mean":  pm.mean(axis=0).tolist(),
                "std":   pm.std(axis=0).tolist(),
                "min":   pm.min(axis=0).tolist(),
                "max":   pm.max(axis=0).tolist(),
                "range": (pm.max(axis=0) - pm.min(axis=0)).tolist(),
            },
        })
    return rows


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_plot_descriptor_range_vs_accuracy_renders(tmp_path):
    out = tmp_path / "fig10.png"
    fig_mod.plot_descriptor_range_vs_accuracy(
        {"alpha_on/runs": _make_fake_descriptor_rows(),
         "alpha_off/runs": _make_fake_descriptor_rows()},
        {"alpha_on/runs": _make_fake_local_rows(),
         "alpha_off/runs": _make_fake_local_rows()},
        "alpha_on/runs", out,
    )
    assert out.is_file()
    assert out.stat().st_size > 10_000


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_plot_descriptor_histograms_by_metric_renders(tmp_path):
    out = tmp_path / "fig15.png"
    fig_mod.plot_descriptor_histograms_by_metric(
        {"alpha_on/runs": _make_fake_descriptor_rows()},
        "alpha_on/runs", out,
    )
    assert out.is_file()
    assert out.stat().st_size > 10_000


def test_collect_subset_descriptor_rows_joins_manifest_cell(tmp_path):
    """Ingest of local_subset_descriptors.json must join with manifest cell
    and surface per_molecule_features as a numpy array."""
    import json as _json
    run = _make_run_dir(tmp_path, "alpha_on/runs", _GOOD_STAMP, n_specs=2)
    sd_path = (run / "checkpoints" / "spec_0000" / "eval"
               / "local_subset_descriptors.json")
    sd_path.parent.mkdir(exist_ok=True)
    sd_path.write_text(_json.dumps({
        "training_molecule_names": ["H2O", "CH4"],
        "feature_names": ["dm_0", "cusp_0"],
        "per_molecule_features": [[1.0, 10.0], [2.0, 20.0]],
        "per_subset_stats": {"mean": [1.5, 15.0], "std": [0.5, 5.0],
                              "min": [1.0, 10.0], "max": [2.0, 20.0],
                              "range": [1.0, 10.0]},
    }))
    rows = fig_mod.collect_subset_descriptor_rows(run)
    assert len(rows) == 1
    r = rows[0]
    assert r["arch"] == "deep_combined_attn"
    assert r["training_molecule_names"] == ["H2O", "CH4"]
    assert r["per_molecule_features"].shape == (2, 2)
    assert r["per_subset_stats"]["range"] == [1.0, 10.0]


def test_collect_per_reaction_rows_joins_manifest_cell(tmp_path):
    """Ingest of local_per_reaction.json must join with manifest cell."""
    run = _make_run_dir(tmp_path, "alpha_on/runs", _GOOD_STAMP, n_specs=2)
    rj_path = run / "checkpoints" / "spec_0000" / "eval" / "local_per_reaction.json"
    rj_path.parent.mkdir(exist_ok=True)
    rj_path.write_text(json.dumps([
        {"name": "AE_h2o", "pool": "w411",
         "reactants": ["h2o", "h", "o"], "products": [], "coeffs": [1, -1, -2],
         "reaction_energy_ref_kcalmol": 232.974,
         "de_nn_kcalmol": 220.0, "de_pbe_kcalmol": 223.0,
         "error_nn_kcalmol": -12.974, "error_pbe_kcalmol": -9.974,
         "abs_error_nn_kcalmol": 12.974, "abs_error_pbe_kcalmol": 9.974,
         "in_sample_overlap": ["h"]},
    ]))
    rows = fig_mod.collect_per_reaction_rows(run)
    assert len(rows) == 1
    r = rows[0]
    assert r["name"] == "AE_h2o"
    assert r["pool"] == "w411"
    assert r["arch"] == "deep_combined_attn"
    assert r["abs_error_nn_kcalmol"] == pytest.approx(12.974)
    assert r["abs_error_pbe_kcalmol"] == pytest.approx(9.974)
    assert r["in_sample_overlap"] == ["h"]


def test_collect_local_test_set_rows_back_compat_legacy_schema(tmp_path):
    """Pre-2026-05-29 CSVs with a single 'mae_kcalmol' column must still
    load (the columns get NN-MAE only; PBE columns surface as None)."""
    run = _make_run_dir(tmp_path, "alpha_on/runs", _GOOD_STAMP, n_specs=2)
    csv_path = (run / "checkpoints" / "spec_0000" / "local_test_set.csv")
    csv_path.write_text(
        "set,mae_kcalmol,n_reactions,n_dropped_overlap,note\n"
        "test_set_bh76,2.5,6,0,strict\n"
    )
    rows = fig_mod.collect_local_test_set_rows(run)
    assert len(rows) == 1
    r = rows[0]
    assert r["mae_kcalmol"] == pytest.approx(2.5)
    assert r["mae_nn_kcalmol"] == pytest.approx(2.5)
    assert r["mae_pbe_kcalmol"] is None
    assert r["delta_nn_minus_pbe"] is None


# ---------------------------------------------------------------------------
# Render canary tests (skip if matplotlib unavailable)
# ---------------------------------------------------------------------------

_PLT_OK = importlib.util.find_spec("matplotlib") is not None


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_plot_generalization_renders(tmp_path):
    run = _make_run_dir(tmp_path / "remote", "alpha_on/runs",
                        _GOOD_STAMP, n_specs=4)
    eval_rows = fig_mod.collect_eval_df_rows(run)
    pm_rows = fig_mod.collect_per_molecule_rows(run)
    final_losses = {0: 1e-3, 1: 2e-4, 2: 5e-3, 3: 1e-2}
    out = tmp_path / "fig1.png"
    fig_mod.plot_generalization(
        {"alpha_on/runs": eval_rows},
        {"alpha_on/runs": pm_rows},
        {"alpha_on/runs": final_losses},
        "alpha_on/runs", out,
    )
    assert out.is_file()
    assert out.stat().st_size > 10_000  # well above an empty canvas


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_plot_generalization_cross_category(tmp_path):
    """Multi-category facet: figure renders with 2 category rows."""
    run_a = _make_run_dir(tmp_path / "remote", "alpha_on/runs",
                          _GOOD_STAMP, n_specs=4)
    run_b = _make_run_dir(tmp_path / "remote", "polarized/alpha_off/runs",
                          "run_20260526T180922Z", n_specs=4)
    eval_rows_a = fig_mod.collect_eval_df_rows(run_a)
    eval_rows_b = fig_mod.collect_eval_df_rows(run_b)
    pm_rows_a = fig_mod.collect_per_molecule_rows(run_a)
    pm_rows_b = fig_mod.collect_per_molecule_rows(run_b)
    final_losses = {0: 1e-3, 1: 2e-4, 2: 5e-3, 3: 1e-2}
    out = tmp_path / "fig1_xc.png"
    fig_mod.plot_generalization(
        {"alpha_on/runs": eval_rows_a,
         "polarized/alpha_off/runs": eval_rows_b},
        {"alpha_on/runs": pm_rows_a,
         "polarized/alpha_off/runs": pm_rows_b},
        {"alpha_on/runs": final_losses,
         "polarized/alpha_off/runs": final_losses},
        "alpha_on/runs", out,
        local_rows_by_cat={
            "alpha_on/runs": _make_fake_local_rows(),
            "polarized/alpha_off/runs": _make_fake_local_rows(),
        },
    )
    assert out.is_file()
    assert out.stat().st_size > 10_000


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_plot_dashboard_with_partial_categories_renders(tmp_path):
    _make_run_dir(tmp_path / "remote", "alpha_on/runs", _GOOD_STAMP,
                  n_specs=4)
    _make_run_dir(tmp_path / "remote", "polarized/alpha_off/runs",
                  "run_20260526T180922Z", n_specs=4, with_eval=False)
    cats = fig_mod.discover_pulled_categories(tmp_path / "remote")
    grids = {c: fig_mod.aggregate_status_grid(rd) for c, rd in cats.items()}
    summary = fig_mod._completion_summary(grids)
    out = tmp_path / "fig3.png"
    fig_mod.plot_dashboard(grids, summary, out)
    assert out.is_file()
    assert out.stat().st_size > 10_000


@pytest.mark.skipif(not _PLT_OK, reason="matplotlib not installed")
def test_main_smoke_end_to_end(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    _make_run_dir(remote, "alpha_on/runs", _GOOD_STAMP, n_specs=4)
    out_dir = tmp_path / "out"

    rc = fig_mod.main([
        "--local-root", str(remote),
        "--out-dir", str(out_dir),
        "--prefix", "smoke",
    ])
    assert rc == 0
    expected = [
        "smoke_fig1_training_diagnostics.png",
        "smoke_fig2_per_molecule_errors.png",
        "smoke_fig3_coverage_dashboard.png",
        "smoke_fig_composite_summary.png",
    ]
    for name in expected:
        p = out_dir / name
        assert p.is_file(), f"missing output PNG: {name}"
        assert p.stat().st_size > 10_000, (
            f"{name} too small ({p.stat().st_size} B) — likely blank canvas"
        )
