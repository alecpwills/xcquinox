"""Tests for ``make_ablation_arch_figure.py`` and ``enhancement_factors.py``.

Three layers, mirroring ``test_make_cluster_pulls_figure.py``:
  * pure data-ingest tests on a synthetic run-dir fixture (no matplotlib),
  * render canaries that drive each plot builder and assert a non-trivial PNG,
  * pure physics-reference tests (PBE F_x / F_c) + a ``slow`` model-load test
    that deserialises one real checkpoint from the pulled run if present.
"""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

# Load both scripts as modules without a package layout.
_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


fig = _load("make_ablation_arch_figure")
ef = _load("enhancement_factors")

_STAMP = "run_20260529T165503Z"
_REAL_RUN = (Path.home() / "Documents/Research/xcquinox-results/runs"
             / "ablation_notransform/polarized/runs" / _STAMP)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_run_dir(root: Path) -> Path:
    """Two archs × two subset sizes, each with held-out reactions + in-sample
    per-molecule AE. spec_0004 is trained-but-uneval'd; spec_0005 is missing
    its model.eqx (untrained)."""
    run_dir = root / "ablation_notransform/polarized/runs" / _STAMP
    run_dir.mkdir(parents=True)

    specs = [
        {"arch": "deep", "subset_size": 1},
        {"arch": "deep", "subset_size": 3},
        {"arch": "deep_notransform", "subset_size": 1},
        {"arch": "deep_notransform", "subset_size": 3},
        {"arch": "deep_attn", "subset_size": 1},   # trained, no eval
        {"arch": "deep_attn", "subset_size": 3},   # untrained
    ]
    manifest = {
        "n_specs": len(specs), "width": 4,
        "specs": [{"index": i, "spec_file": f"spec_{i:04d}.spec",
                   "sha256": "x" * 64, "cell": c}
                  for i, c in enumerate(specs)],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "specs").mkdir()

    for i, cell in enumerate(specs):
        sd = run_dir / "checkpoints" / f"spec_{i:04d}"
        sd.mkdir(parents=True)
        # Training composition (shared across archs per subset_size); element
        # anchors h/o/c are filtered out by training_subsets_by_size.
        (sd / "train_metadata.json").write_text(json.dumps(
            {"molecules": (["HO", "h", "o"] if cell["subset_size"] == 1
                           else ["HO", "CH4", "h", "c", "o"])}))
        if i == 5:
            continue  # untrained: no model.eqx, no eval
        (sd / "model.eqx").write_bytes(b"x" * 16)
        # Training-loss trajectory (per-group-update). ss=3 specs get a late
        # bump (instability); ss=1 specs converge cleanly.
        import numpy as _np
        traj = _np.linspace(0.1, 1e-3, 60)
        if cell["subset_size"] == 3:
            traj[-15:] = 0.05  # late oscillation back up
        _np.save(sd / "losses.npy", traj)
        if i == 4:
            continue  # trained but no eval dirs
        # In-sample per-molecule AE (eval/per_molecule.json).
        ev = sd / "eval"; ev.mkdir()
        (ev / "per_molecule.json").write_text(json.dumps([
            {"molecule": "HO", "AE_error_kcalmol": 6.0 + i, "density_rmse": 3e-3,
             "skipped": False, "scf_converged": False},
            {"molecule": "CH4", "AE_error_kcalmol": -2.0 - i, "density_rmse": 1e-3,
             "skipped": False, "scf_converged": True},
            {"molecule": "H", "skipped": True, "skip_reason": "atomic_system",
             "AE_error_kcalmol": None, "density_rmse": None},
            {"molecule": "X", "AE_error_kcalmol": None, "skipped": False},
        ]))
        # Held-out reactions (eval_holdout/per_reaction.json).
        eh = sd / "eval_holdout"; eh.mkdir()
        (eh / "per_reaction.json").write_text(json.dumps([
            {"name": "bh76_a", "pool": "bh76",
             "reactants": ["HO", "h"], "products": ["HOh_ts"],
             "reaction_energy_ref_kcalmol": 17.7,
             "de_nn_kcalmol": -91.0 + i, "de_pbe_kcalmol": -91.2 + i,
             "abs_error_nn_kcalmol": 108.7 - i, "abs_error_pbe_kcalmol": 108.9 - i},
            {"name": "w411_b", "pool": "w411",
             "reactants": ["HO"], "products": ["h", "o"],
             "reaction_energy_ref_kcalmol": 120.0,
             "de_nn_kcalmol": 118.0 + i, "de_pbe_kcalmol": 119.0 + i,
             "abs_error_nn_kcalmol": 2.0 + i, "abs_error_pbe_kcalmol": 1.0 + i},
        ]))
    return run_dir


# ---------------------------------------------------------------------------
# Data-ingest tests
# ---------------------------------------------------------------------------

def test_collect_holdout_reaction_rows_joins_cell(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    # 4 evaluated specs × 2 reactions each = 8.
    assert len(rows) == 8
    archs = {r["arch"] for r in rows}
    assert archs == {"deep", "deep_notransform"}
    pools = {r["pool"] for r in rows}
    assert pools == {"bh76", "w411"}
    for r in rows:
        assert r["subset_size"] in (1, 3)
        assert isinstance(r["ref_kcalmol"], (int, float))


def test_collect_insample_ae_drops_skipped_and_null(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_insample_ae_rows(run)
    # 4 evaluated specs × 2 finite-AE molecules each = 8 (atom + null dropped).
    assert len(rows) == 8
    assert all(fig._is_num(r["AE_error_kcalmol"]) for r in rows)
    assert all(not r["skipped"] for r in rows)


def test_w411_rows_filters_pool(tmp_path):
    run = _make_run_dir(tmp_path)
    rxn = fig.collect_holdout_reaction_rows(run)
    w411 = fig._w411_rows(rxn)
    # Only the w411 reaction per evaluated spec survives (bh76 dropped).
    assert w411, "expected held-out W4-11 rows"
    assert all(r["pool"] == "w411" for r in w411)
    assert all(fig._is_num(r["de_nn_kcalmol"]) for r in w411)
    assert all(fig._is_num(r["de_pbe_kcalmol"]) for r in w411)


def test_w411_mae_by_subset_pools_archs(tmp_path):
    run = _make_run_dir(tmp_path)
    rxn = fig.collect_holdout_reaction_rows(run)
    mae = fig._w411_mae_by_subset(fig._w411_rows(rxn))
    # subset_size 1: specs 0 (deep) & 2 (deep_notransform); the w411 reaction
    # has abs_error_nn = 2.0+i -> {2.0 (i=0), 4.0 (i=2)} -> mean 3.0.
    assert mae[1] == pytest.approx((2.0 + 4.0) / 2, rel=1e-6)
    assert 3 in mae


def test_trained_spec_count(tmp_path):
    run = _make_run_dir(tmp_path)
    # specs 0-4 have model.eqx; spec 5 does not.
    assert fig.trained_spec_count(run) == 5


def test_best_subset_per_arch_picks_largest(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    best = fig._best_subset_per_arch(rows)
    assert best == {"deep": 3, "deep_notransform": 3}


def test_reaction_mae_by_arch_subset(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    mae = fig.reaction_mae_by_arch_subset(rows)
    # spec_0000 (deep, ss=1): mean(|108.7|, |2.0|) = 55.35
    assert mae[("deep", 1)] == pytest.approx((108.7 + 2.0) / 2, rel=1e-6)
    assert ("deep_notransform", 3) in mae


def test_ae_mae_by_arch_subset(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_insample_ae_rows(run)
    mae = fig.ae_mae_by_arch_subset(rows)
    # spec_0000 (deep, ss=1): mean(|6|, |-2|) = 4.0
    assert mae[("deep", 1)] == pytest.approx(4.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Render canaries
# ---------------------------------------------------------------------------

def _png_ok(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 2000


def test_plot_parity_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    out = fig.plot_parity(rows, tmp_path / "parity.png", _STAMP)
    assert _png_ok(out)


def test_plot_heatmap_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    rxn = fig.collect_holdout_reaction_rows(run)
    ae = fig.collect_insample_ae_rows(run)
    out = fig.plot_arch_subset_heatmap(
        rxn, ae, tmp_path / "heat.png", _STAMP,
        n_trained=5, n_total=6, n_holdout=4)
    assert _png_ok(out)


def test_plot_mae_by_arch_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    rxn = fig.collect_holdout_reaction_rows(run)
    ae = fig.collect_insample_ae_rows(run)
    out = fig.plot_mae_by_arch(rxn, ae, tmp_path / "bars.png", _STAMP)
    assert _png_ok(out)


def test_plot_mae_vs_subset_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    rxn = fig.collect_holdout_reaction_rows(run)
    ae = fig.collect_insample_ae_rows(run)
    out = fig.plot_mae_vs_subset(rxn, ae, tmp_path / "curves.png", _STAMP)
    assert _png_ok(out)


def test_plot_ae_parity_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    rxn = fig.collect_holdout_reaction_rows(run)
    out = fig.plot_ae_parity(rxn, tmp_path / "ae_parity.png", _STAMP)
    assert _png_ok(out)


def test_build_all_writes_seven_figures(tmp_path):
    run = _make_run_dir(tmp_path)
    written = fig.build_all(run, tmp_path / "out")
    assert len(written) == 7
    assert all(_png_ok(p) for p in written)
    assert (tmp_path / "out" / "ablation_ae_parity.png").is_file()
    assert (tmp_path / "out" / "ablation_parity_by_class.png").is_file()
    # the NN/PBE ratio heatmap rides along with the raw-MAE grid
    assert (tmp_path / "out" / "ablation_arch_subset_heatmap_vs_pbe.png").is_file()


def test_mae_vs_subset_panel_draws_reference_lines():
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    fig._mae_vs_subset_panel(ax, {("deep", 1): 5.0}, ["deep"], title="t",
                             pbe_line=11.5, scan_line=4.5, scan_suffix="")
    labels = [ln.get_label() for ln in ax.lines]
    assert any("PBE" in l for l in labels), labels
    assert any("SCAN" in l for l in labels), labels
    plt.close(f)


def test_mae_vs_subset_accepts_baselines_and_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    rxn = fig.collect_holdout_reaction_rows(run)
    ae = fig.collect_insample_ae_rows(run)
    pbe = {"bh76": 8.0, "w411": 13.7, "combined": 11.8}
    scan = {"bh76": 6.0, "w411": 3.8, "combined": 4.5,
            "coverage": {"combined": {"used": 10, "reference": 10}}}
    out = fig.plot_mae_vs_subset(rxn, ae, tmp_path / "curves.png", _STAMP,
                                 pbe_baseline=pbe, scan_baseline=scan)
    assert _png_ok(out)


def test_rung_linestyles_one_style_per_rung():
    ls = fig._rung_linestyles(["deep_3x16", "deep_cusp_3x16", "deep_mgga_3x16",
                               "deep_rung35_3x16", "deep_rung35_mgga_3x16"])
    assert ls["deep_3x16"] == ls["deep_cusp_3x16"]  # same rung -> same style
    styles = {ls["deep_3x16"], ls["deep_mgga_3x16"], ls["deep_rung35_3x16"],
              ls["deep_rung35_mgga_3x16"]}
    assert len(styles) == 4  # distinct style per rung


def test_build_all_passes_eval_subdir_to_scan_baseline(tmp_path, monkeypatch):
    run = _make_run_dir(tmp_path)
    seen = {}

    def _rec(run_dir, **kw):
        seen.update(kw)
        return {"bh76": float("nan"), "w411": float("nan"),
                "combined": float("nan")}

    monkeypatch.setattr(fig, "scan_pool_baseline", _rec)
    fig.build_all(run, tmp_path / "out")
    assert seen.get("eval_subdir") == "eval_holdout"


def test_heatmap_vs_pbe_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    rxn = fig.collect_holdout_reaction_rows(run)
    out = fig.plot_arch_subset_heatmap_vs_pbe(rxn, tmp_path / "heat_ratio.png",
                                              _STAMP)
    assert _png_ok(out)


def _write_pm(run, spec, entries, eval_subdir="eval_holdout"):
    import json as _json
    d = run / "checkpoints" / spec / eval_subdir
    d.mkdir(parents=True, exist_ok=True)
    (d / "per_molecule.json").write_text(_json.dumps(entries))


def test_pbe_energies_exclude_cross_spec_disagreement(tmp_path, capsys):
    # The c2 class of artifact: one spec's (arm's) PBE reference drifted. The
    # map must EXCLUDE the species (never silently inherit whichever spec
    # sorts first) and say so.
    run = tmp_path / "run_x"
    _write_pm(run, "spec_0000", [{"molecule": "h2", "E_pbe": -1.17},
                                 {"molecule": "c2", "E_pbe": -75.816711949}])
    _write_pm(run, "spec_0001", [{"molecule": "h2", "E_pbe": -1.17},
                                 {"molecule": "c2", "E_pbe": -75.757329256}])
    pbe = fig._first_pbe_energies(run)
    assert "h2" in pbe and "c2" not in pbe
    assert "c2" in capsys.readouterr().out


def test_pbe_energies_tolerate_scf_noise(tmp_path, capsys):
    run = tmp_path / "run_x"
    _write_pm(run, "spec_0000", [{"molecule": "h2", "E_pbe": -1.17}])
    _write_pm(run, "spec_0001", [{"molecule": "h2", "E_pbe": -1.17 + 2.5e-6}])
    pbe = fig._first_pbe_energies(run)
    assert pbe["h2"] == pytest.approx(-1.17, abs=1e-5)
    assert "WARNING" not in capsys.readouterr().out


def test_value_clusters_groups_within_tol():
    # SCF noise merges into one cluster; a genuine outlier opens its own;
    # ordering is largest-cluster-first (ties by value) so [0] is the
    # majority candidate when one exists.
    one = fig._value_clusters(
        [("spec_0000", -1.17), ("spec_0001", -1.17 + 2.5e-6)], 1e-4)
    assert len(one) == 1 and one[0][1] == ["spec_0000", "spec_0001"]
    # cluster value is a reported value (the smallest member), never a mean
    assert one[0][0] == pytest.approx(-1.17, abs=1e-12)
    maj = fig._value_clusters(
        [("spec_0000", 1.0), ("spec_0001", 1.00005), ("spec_0002", 2.0)], 1e-4)
    assert [c[1] for c in maj] == [["spec_0000", "spec_0001"], ["spec_0002"]]
    tie = fig._value_clusters([("b", 2.0), ("a", 1.0)], 1e-4)
    assert [c[1] for c in tie] == [["a"], ["b"]]   # tie -> by value


def test_outlier_clause_requires_strict_majority():
    # A plurality is not a majority: 3-2-2 must read as a split (nobody is
    # told to re-evaluate 4 of 7 evals against a 43% "reference"), and a
    # single cluster reports agreement, never a split.
    split_3_2_2 = fig._value_clusters(
        [("a", 1.0), ("b", 1.0), ("c", 1.0),
         ("d", 2.0), ("e", 2.0), ("f", 3.0), ("g", 3.0)], 1e-4)
    clause = fig._outlier_clause(split_3_2_2)
    assert "multi-spec split" in clause
    assert "re-evaluate" not in clause
    majority = fig._value_clusters(
        [("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 2.0)], 1e-4)
    assert "re-evaluate" in fig._outlier_clause(majority)
    lone = fig._outlier_clause([(1.0, ["a", "b"])])
    assert "agree" in lone and "split" not in lone


def test_pbe_energies_warning_names_outlier_spec(tmp_path, capsys):
    # The 42-vs-1 pattern (one degraded eval re-converged PBE elsewhere): the
    # warning must name the outlier spec so the re-evaluation target is
    # readable from the console, and must not claim a pending reference
    # repair -- the majority reference is intact.
    run = tmp_path / "run_x"
    _write_pm(run, "spec_0000", [{"molecule": "c2", "E_pbe": -75.757329256}])
    _write_pm(run, "spec_0001", [{"molecule": "c2", "E_pbe": -75.757329256}])
    _write_pm(run, "spec_0002", [{"molecule": "c2", "E_pbe": -75.781328401}])
    pbe = fig._first_pbe_energies(run)
    out = capsys.readouterr().out
    assert "c2" not in pbe
    assert "spec_0002" in out and "2 specs" in out
    assert "re-evaluate" in out
    assert "pending reference repair" not in out


def test_pbe_energies_warning_reports_multi_spec_split(tmp_path, capsys):
    # With no majority there is no re-evaluation target: every side is named.
    run = tmp_path / "run_x"
    _write_pm(run, "spec_0000", [{"molecule": "c2", "E_pbe": -75.816711949}])
    _write_pm(run, "spec_0001", [{"molecule": "c2", "E_pbe": -75.757329256}])
    fig._first_pbe_energies(run)
    out = capsys.readouterr().out
    assert "multi-spec split" in out
    assert "spec_0000" in out and "spec_0001" in out


_SCAN_FIXTURE_RECORDS = {
    "HO": {"density_rmse_scan": 2e-4, "density_eps_l1_scan": 6e-4},
    "CH4": {"density_rmse_scan": 1e-4, "density_eps_l1_scan": 2e-4}}


def test_headline_ed_receives_scan_legs(tmp_path, monkeypatch):
    """EVERY combined_ed_by_cell AND combined_ed_fixed_gamma call in the
    suite must carry the SCAN legs when the caches resolve -- the headline
    (wt + mae) calls omitted them first, then the DFS-units fixed-gamma
    calls did the same, so ablation_combined_energy_density* (and the
    _dfs_units twin CSV) drew/recorded no ed_scan."""
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    rows = fig.collect_holdout_reaction_rows(run)
    errs = {str(r["name"]): 1.0 for r in rows}
    monkeypatch.setattr(fig, "scan_reaction_errors", lambda *a, **k: errs)
    monkeypatch.setattr(fig, "_scan_density_records",
                        lambda *a, **k: dict(_SCAN_FIXTURE_RECORDS))
    seen_self, seen_fixed = [], []
    real_self = fig.combined_ed_by_cell
    real_fixed = fig.combined_ed_fixed_gamma

    def _rec_self(*a, **kw):
        seen_self.append(kw)
        return real_self(*a, **kw)

    def _rec_fixed(*a, **kw):
        seen_fixed.append(kw)
        return real_fixed(*a, **kw)

    monkeypatch.setattr(fig, "combined_ed_by_cell", _rec_self)
    monkeypatch.setattr(fig, "combined_ed_fixed_gamma", _rec_fixed)
    fig.build_density_energy_figures(run, tmp_path / "out")
    assert seen_self, "ED summaries did not run on the density fixture"
    assert all(fig._is_num(kw.get("e_scan")) and fig._is_num(kw.get("d_scan"))
               for kw in seen_self), seen_self
    assert seen_fixed, "fixed-gamma (DFS-units) ED summaries did not run"
    assert all(fig._is_num(kw.get("e_scan")) and fig._is_num(kw.get("d_scan"))
               for kw in seen_fixed), seen_fixed


def test_ed_csv_carries_scan_columns(tmp_path):
    summary = {"gamma": 1000.0, "e_pbe": 8.0, "d_pbe": 2e-4, "ed_pbe": 8.0,
               "e_scan": 4.0, "d_scan": 1.5e-4, "ed_scan": 3.5,
               "cells": {("deep", 1): {"E": 5.0, "D": 2e-4, "gammaD": 0.2,
                                       "ED": 4.0, "beats_pbe": True,
                                       "beats_scan": False}}}
    out = fig.write_combined_ed_csv({"wtmad2": summary}, tmp_path / "ed.csv",
                                    n_reactions={("deep", 1): 10},
                                    n_density={("deep", 1): 5})
    import csv as _csv
    row = next(_csv.DictReader(out.open()))
    assert row["E_scan_kcalmol"] == "4.0"
    assert row["ED_scan_kcalmol"] == "3.5"
    assert row["beats_scan"] == "False"


def test_scan_baseline_pool_restricted_to_pbe_computable(tmp_path):
    # A species excluded from the PBE map (cross-spec disagreement) must drop
    # its reactions from BOTH legs, keeping the SCAN and PBE lines averaging
    # the same reactions.
    rxns = [
        {"name": "r1", "reactants": ["a"], "products": ["b"],
         "coeffs": [-1.0, 1.0], "reaction_energy_ref": 0.01,
         "source_pool": "w411"},
        {"name": "r2", "reactants": ["c2"], "products": ["b"],
         "coeffs": [-1.0, 1.0], "reaction_energy_ref": 0.02,
         "source_pool": "w411"},
    ]
    scan = {"a": -1.0, "b": -0.99, "c2": -75.8}
    pbe = {"a": -1.0, "b": -0.985}
    out = fig.scan_pool_baseline(tmp_path, _loader=lambda: (None, rxns),
                                 _energies=scan, _pbe_energies=pbe)
    # reference = the UNRESTRICTED leg size, so a guard exclusion is visible
    # as used < reference on every footer instead of shrinking both counts.
    assert out["coverage"]["combined"] == {"used": 1, "reference": 2}


def test_pbe_pool_baseline_reports_reduced_coverage(tmp_path):
    # A species excluded by the cross-spec guard drops its reactions from the
    # pooled PBE legs; the coverage dict must record that against the
    # unrestricted leg size, or the reduction is invisible on every footer.
    rxns = [
        {"name": "r1", "source_pool": "w411", "reactants": ["a"],
         "products": ["b"], "coeffs": [-1.0, 1.0], "reaction_energy_ref": 0.01},
        {"name": "r2", "source_pool": "w411", "reactants": ["c2"],
         "products": ["b"], "coeffs": [-1.0, 1.0], "reaction_energy_ref": 0.02},
    ]
    run = tmp_path / "run_x"
    _write_pm(run, "spec_0000", [{"molecule": "a", "E_pbe": -1.0},
                                 {"molecule": "b", "E_pbe": -0.99},
                                 {"molecule": "c2", "E_pbe": -75.757329}])
    _write_pm(run, "spec_0001", [{"molecule": "a", "E_pbe": -1.0},
                                 {"molecule": "b", "E_pbe": -0.99},
                                 {"molecule": "c2", "E_pbe": -75.781328}])
    base = fig.pbe_pool_baseline(run, _loader=lambda: ({}, rxns))
    assert base["coverage"]["w411"] == {"used": 1, "reference": 2}
    assert base["coverage"]["combined"] == {"used": 1, "reference": 2}


def test_pool_line_suffix():
    reduced = {"combined": 4.79,
               "coverage": {"combined": {"used": 215, "reference": 216}}}
    assert fig.pool_line_suffix(reduced) == ", 215/216"
    full = {"combined": 4.89,
            "coverage": {"combined": {"used": 216, "reference": 216}}}
    assert fig.pool_line_suffix(full) == ""
    assert fig.pool_line_suffix({"combined": 4.89}) == ""      # legacy dict
    assert fig.pool_line_suffix(None) == ""


def test_provenance_footer_discloses_reduced_pbe_pool():
    full = {"bh76": 8.15, "w411": 14.01, "combined": 11.95,
            "coverage": {"combined": {"used": 216, "reference": 216}}}
    legacy = {"bh76": 8.15, "w411": 14.01, "combined": 11.95}
    # Full coverage renders byte-identically to a coverage-less baseline.
    assert fig.provenance_footer(full) == fig.provenance_footer(legacy)
    reduced = {"bh76": 8.15, "w411": 13.82, "combined": 11.82,
               "coverage": {"combined": {"used": 215, "reference": 216}}}
    s = fig.provenance_footer(reduced)
    assert "[215/216 reactions]" in s
    scan_reduced = {"bh76": 6.88, "w411": 3.65, "combined": 4.79,
                    "coverage": {"combined": {"used": 215, "reference": 216}}}
    s2 = fig.provenance_footer(legacy, scan_reduced)
    assert s2.count("[215/216 reactions]") == 1
    assert "SCAN (full pool):" in s2
    # the realistic guard-exclusion case: BOTH legs reduced, each section
    # carries its own bracket
    s3 = fig.provenance_footer(reduced, scan_reduced)
    assert s3.count("[215/216 reactions]") == 2


def test_wtmad2_scan_n_ref_counts_only_pbe_computable_rows():
    # The docstring contract is "the SAME reactions wtmad2_pbe_baseline
    # reduces" (pool + finite ref + finite PBE error). Rows PBE could not
    # score must neither inflate n_ref (a spurious coverage-floor withdrawal)
    # nor contribute SCAN errors to the average.
    rows = [
        {"name": "ok0", "pool": "bh76", "ref_kcalmol": 10.0,
         "abs_error_pbe_kcalmol": 1.0, "abs_error_nn_kcalmol": 1.0},
        {"name": "ok1", "pool": "bh76", "ref_kcalmol": 11.0,
         "abs_error_pbe_kcalmol": 2.0, "abs_error_nn_kcalmol": 1.0},
        {"name": "nopbe0", "pool": "w411", "ref_kcalmol": 20.0,
         "abs_error_pbe_kcalmol": None, "abs_error_nn_kcalmol": 1.0},
        {"name": "nopbe1", "pool": "w411", "ref_kcalmol": 21.0,
         "abs_error_pbe_kcalmol": None, "abs_error_nn_kcalmol": 1.0},
    ]
    scan = {"ok0": 2.0, "ok1": 4.0, "nopbe0": 100.0}
    val, used, ref = fig.wtmad2_scan_baseline(rows, scan)
    assert (used, ref) == (2, 2)
    # Excluding the no-PBE rows must equal restricting the input to them.
    val_restricted, _, _ = fig.wtmad2_scan_baseline(rows[:2], scan)
    assert val == pytest.approx(val_restricted)


def test_overview_provenance_scan_sentence_branches():
    # The overview footer must track panel F: it draws the SCAN ED comparator
    # exactly when the summary carries a finite ed_scan, and the footer must
    # never claim "no SCAN lines" over a drawn line.
    with_scan = fig._overview_provenance({"ed_scan": 6.3})
    assert "no SCAN lines" not in with_scan
    assert "SCAN" in with_scan
    for absent in (None, {}, {"ed_scan": None, "cells": {}}):
        prov = fig._overview_provenance(absent)
        assert "no SCAN lines" in prov, absent


def test_overview_footer_matches_scan_state(tmp_path, monkeypatch):
    # Integration form of the same contract: with both SCAN caches resolving,
    # the rendered overview's provenance must not deny the line panel F draws.
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    rows = fig.collect_holdout_reaction_rows(run)
    errs = {str(r["name"]): 1.0 for r in rows}
    monkeypatch.setattr(fig, "scan_reaction_errors", lambda *a, **k: errs)
    monkeypatch.setattr(fig, "_scan_density_records",
                        lambda *a, **k: dict(_SCAN_FIXTURE_RECORDS))
    got = {}
    real = fig.plot_density_energy_overview

    def _cap(*a, **kw):
        out_path = a[2] if len(a) > 2 else kw.get("out_path")
        if "_dfs_units" not in str(out_path):
            got["prov"] = kw.get("provenance")
        return real(*a, **kw)

    monkeypatch.setattr(fig, "plot_density_energy_overview", _cap)
    fig.build_density_energy_figures(run, tmp_path / "out")
    assert got.get("prov"), "base overview did not render"
    assert "no SCAN lines" not in got["prov"]
    assert "SCAN" in got["prov"]


def test_scan_ed_suffix_formats():
    # Complete coverage -> no suffix; partial legs named compactly, mirroring
    # scan_line_value's ", used/ref" convention.
    assert fig._scan_ed_suffix(6, 6, 4, 4) == ""
    assert fig._scan_ed_suffix(5, 6, 4, 4) == ", E 5/6"
    assert fig._scan_ed_suffix(6, 6, 3, 4) == ", D 3/4"
    assert fig._scan_ed_suffix(5, 6, 3, 4) == ", E 5/6 D 3/4"
    assert fig._scan_ed_suffix(0, 0, 0, 0) == ""


def test_ed_lines_panel_scan_label_carries_coverage_suffix():
    # A partially-covered SCAN comparator must not read as like-for-like:
    # the ED panel's legend label carries the summary's scan_suffix.
    import matplotlib.pyplot as plt
    cells = {("deep", 1): {"E": 5.0, "D": 2e-4, "gammaD": 0.2,
                           "ED": 0.39, "beats_pbe": True,
                           "beats_scan": True}}
    base = {"gamma": 1000.0, "gamma_mode": "fixed", "e_pbe": 8.0,
            "d_pbe": 2e-4, "ed_pbe": 7.0, "ed_scan": 5.0, "cells": cells}
    for sfx, want in ((", E 5/6 D 3/4", "SCAN, E 5/6 D 3/4"), (None, "SCAN")):
        summary = dict(base)
        if sfx is not None:
            summary["scan_suffix"] = sfx
        f, ax = plt.subplots()
        try:
            fig._ed_lines_panel(ax, summary, "t")
            labels = [ln.get_label() for ln in ax.lines]
            assert want in labels, (sfx, labels)
        finally:
            plt.close(f)


def test_channel_summaries_carry_scan_suffix(tmp_path, monkeypatch):
    # Every channel summary whose ed_scan resolves exposes scan_suffix for
    # the panels ("" at full coverage).
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    rows = fig.collect_holdout_reaction_rows(run)
    hd_rows = fig.collect_holdout_density_rows(run)
    pbe_table = fig.load_pbe_density_table(run)
    errs = {str(r["name"]): 1.0 for r in rows}
    out = fig.channel_ed_summaries(rows, hd_rows, pbe_table,
                                   scan_errors=errs,
                                   scan_density_records=dict(
                                       _SCAN_FIXTURE_RECORDS))
    checked = 0
    for ch, s in out.items():
        if s is not None and s.get("ed_scan") is not None:
            assert s.get("scan_suffix") == "", (ch, s.get("scan_suffix"))
            checked += 1
    assert checked, "no channel resolved a SCAN ED comparator"


def test_dfs_units_ed_scan_consistent_across_csvs(tmp_path, monkeypatch):
    # Decisive consistency check for the half-threaded fix: the same
    # combined-channel cell must carry the SAME ED_scan in the headline
    # DFS-units leg (ablation_combined_energy_density.csv) and the 3x3
    # DFS-units CSV -- previously blank in one and populated in the other.
    import csv as _csv
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    rows = fig.collect_holdout_reaction_rows(run)
    errs = {str(r["name"]): 1.0 for r in rows}
    monkeypatch.setattr(fig, "scan_reaction_errors", lambda *a, **k: errs)
    monkeypatch.setattr(fig, "_scan_density_records",
                        lambda *a, **k: dict(_SCAN_FIXTURE_RECORDS))
    outdir = tmp_path / "out"
    fig.build_density_energy_figures(run, outdir)
    with (outdir / "ablation_combined_energy_density.csv").open() as f:
        main = [r for r in _csv.DictReader(f)
                if r["leg"] == "wtmad2_eps_gamma_dfs"]
    assert main, "headline DFS-units leg missing from the CSV"
    assert all(r["ED_scan_kcalmol"] != "" for r in main), main
    with (outdir / "ablation_density_energy_3x3_dfs_units.csv").open() as f:
        chan = {(r["arch"], r["subset_size"]): r for r in _csv.DictReader(f)
                if r["leg"] == "combined_wtmad2_eps_gamma_dfs"}
    assert chan, "3x3 DFS-units combined leg missing from the CSV"
    for r in main:
        c = chan.get((r["arch"], r["subset_size"]))
        assert c is not None, (r["arch"], r["subset_size"])
        assert float(r["ED_scan_kcalmol"]) == pytest.approx(
            float(c["ED_scan_kcalmol"])), (r, c)


def test_suite_scan_console_reported_once(tmp_path, monkeypatch, capsys):
    # One annotated SCAN baseline line per suite pass (build_all +
    # build_density_energy_figures), not a bare duplicate followed by the
    # annotated form.
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    base = {"bh76": 6.9, "w411": 3.8, "combined": 4.9,
            "coverage": {"combined": {"used": 216, "reference": 216}}}
    monkeypatch.setattr(fig, "scan_pool_baseline", lambda *a, **k: dict(base))
    fig.build_all(run, tmp_path / "o1")
    fig.build_density_energy_figures(run, tmp_path / "o2")
    out = capsys.readouterr().out
    assert out.count("SCAN baseline (full pool)") == 1, out
    assert "[216/216 reactions]" in out


_STUB_POOL_SPECS = {
    "hcn": {"atom_composition": (("C", 1), ("H", 1), ("N", 1)),
            "charge": 0, "spin": 0,
            "atom": "C 0 0 0; N 0 0 1.15; H 0 0 -1.06"},
    "co2": {"atom_composition": (("C", 1), ("O", 2)), "charge": 0, "spin": 0,
            "atom": "C 0 0 0; O 0 0 1.16; O 0 0 -1.16"},
}


def _make_leak_run(root):
    """One spec trained on the Hill-named CHN whose pool twin hcn sits in the
    'held-out' rows (the cluster-side name filter cannot see the identity),
    plus a genuinely held-out co2 reaction."""
    run = root / "leak/polarized/runs" / _STAMP
    run.mkdir(parents=True)
    manifest = {"n_specs": 1, "width": 4, "specs": [
        {"index": 0, "spec_file": "spec_0000.spec", "sha256": "x" * 64,
         "cell": {"arch": "deep", "subset_size": 2}}]}
    (run / "manifest.json").write_text(json.dumps(manifest))
    sd = run / "checkpoints" / "spec_0000"
    (sd / "eval_holdout").mkdir(parents=True)
    (sd / "train_metadata.json").write_text(json.dumps(
        {"molecules": ["CHN", "h", "c", "n"]}))
    (sd / "eval_holdout" / "per_reaction.json").write_text(json.dumps([
        {"name": "w411_hcn_atomization", "pool": "w411",
         "reactants": ["hcn"], "products": ["h", "c", "n"],
         "reaction_energy_ref_kcalmol": 313.4,
         "abs_error_nn_kcalmol": 0.1, "abs_error_pbe_kcalmol": 13.7},
        {"name": "w411_co2_atomization", "pool": "w411",
         "reactants": ["co2"], "products": ["c", "o"],
         "reaction_energy_ref_kcalmol": 390.0,
         "abs_error_nn_kcalmol": 5.0, "abs_error_pbe_kcalmol": 9.0},
    ]))
    (sd / "eval_holdout" / "per_molecule.json").write_text(json.dumps([
        {"molecule": "hcn", "density_rmse": 1e-4, "density_rmse_pbe": 4e-4},
        {"molecule": "co2", "density_rmse": 2e-4, "density_rmse_pbe": 5e-4},
    ]))
    return run


_KCAL = 627.5094740631

_RECON_POOL_SPECS = {
    "hcn": {"atom_composition": (("C", 1), ("H", 1), ("N", 1)), "charge": 0,
            "spin": 0, "atom": "C 0 0 0; N 0 0 1.15; H 0 0 -1.06"},
    "hnc": {"atom_composition": (("C", 1), ("H", 1), ("N", 1)), "charge": 0,
            "spin": 0, "atom": "N 0 0 0; C 0 0 1.17; H 0 0 -1.00"},
    "co2": {"atom_composition": (("C", 1), ("O", 2)), "charge": 0, "spin": 0,
            "atom": "C 0 0 0; O 0 0 1.16; O 0 0 -1.16"},
    "o3": {"atom_composition": (("O", 3),), "charge": 0, "spin": 0,
           "atom": "O 0 0 0; O 0 0 1.27; O 1.1 0 -0.6"},
    "h": {"atom_composition": (("H", 1),), "charge": 0, "spin": 1,
          "atom": "H 0 0 0"},
    "c": {"atom_composition": (("C", 1),), "charge": 0, "spin": 2,
          "atom": "C 0 0 0"},
    "n": {"atom_composition": (("N", 1),), "charge": 0, "spin": 3,
          "atom": "N 0 0 0"},
    "o": {"atom_composition": (("O", 1),), "charge": 0, "spin": 2,
          "atom": "O 0 0 0"},
}

_RECON_POOL_RXNS = [
    {"name": "w411_hcn_atomization", "source_pool": "w411",
     "reactants": ["hcn"], "products": ["h", "c", "n"],
     "coeffs": [-1.0, 1.0, 1.0, 1.0], "reaction_energy_ref": 313.4},
    {"name": "w411_hnc_atomization", "source_pool": "w411",
     "reactants": ["hnc"], "products": ["h", "c", "n"],
     "coeffs": [-1.0, 1.0, 1.0, 1.0], "reaction_energy_ref": 298.7},
    {"name": "bh76_hcn_to_hcnts", "source_pool": "bh76",
     "reactants": ["hcn"], "products": ["hnc"],
     "coeffs": [-1.0, 1.0], "reaction_energy_ref": 15.0},
    {"name": "w411_co2_atomization", "source_pool": "w411",
     "reactants": ["co2"], "products": ["c", "o"],
     "coeffs": [-1.0, 1.0, 2.0], "reaction_energy_ref": 390.0},
    {"name": "w411_o3_atomization", "source_pool": "w411",
     "reactants": ["o3"], "products": ["o"],
     "coeffs": [-1.0, 3.0], "reaction_energy_ref": 147.0},
]

_RECON_E_NN = {"hcn": -93.30, "hnc": -93.27, "co2": -188.10,
               "h": -0.50, "c": -37.80, "n": -54.50, "o": -75.00}
_RECON_E_PBE = {k: v + 0.001 for k, v in _RECON_E_NN.items()}


def _make_recon_run(root):
    """Run whose per_molecule.json carries per-species energies: the
    verbatim-holdout reconstruction path. Trained: the CHN atomization
    (reaction form). Validation slice: the co2 atomization. o3 has no
    energies -> its reaction NaN-drops."""
    run = root / "recon/polarized/runs" / _STAMP
    run.mkdir(parents=True)
    manifest = {"n_specs": 1, "width": 4, "specs": [
        {"index": 0, "spec_file": "spec_0000.spec", "sha256": "x" * 64,
         "cell": {"arch": "deep", "subset_size": 2}}]}
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "validation").mkdir()
    (run / "validation" / "val_reactions.json").write_text(json.dumps([
        {"name": "w411_co2_atomization", "reactants": ["co2"],
         "products": ["c", "o"], "coeffs": [-1.0, 1.0, 2.0],
         "reaction_energy_ref": 390.0}]))
    sd = run / "checkpoints" / "spec_0000"
    (sd / "eval_holdout").mkdir(parents=True)
    (sd / "train_metadata.json").write_text(json.dumps({
        "molecules": ["CHN", "h", "c", "n"],
        "loss_kwargs": {"bh76_reactions": [
            {"name": "CHN", "reactants": ["CHN"],
             "products": ["C", "H", "N"],
             "coeffs": [-1.0, 1.0, 1.0, 1.0]}]}}))
    pm = [{"molecule": m, "E_total_nn": _RECON_E_NN.get(m),
           "E_pbe": _RECON_E_PBE.get(m)}
          for m in list(_RECON_E_NN) + ["o3"]]
    (sd / "eval_holdout" / "per_molecule.json").write_text(json.dumps(pm))
    return run


def test_reconstructed_rows_verbatim_holdout(tmp_path, monkeypatch, capsys):
    # The full test slice is rebuilt from per-species energies over the
    # canonical pool; ONLY the verbatim-trained reaction's pool twin and the
    # validation slice leave; a species-sharing barrier STAYS; reactions
    # with missing energies NaN-drop.
    run = _make_recon_run(tmp_path)
    monkeypatch.setattr(fig, "_canonical_pool",
                        lambda: (dict(_RECON_POOL_SPECS),
                                 list(_RECON_POOL_RXNS)))
    rows = fig.collect_holdout_reaction_rows(run)
    names = sorted(r["name"] for r in rows)
    assert names == ["bh76_hcn_to_hcnts", "w411_hnc_atomization"], names
    by = {r["name"]: r for r in rows}
    de = (_RECON_E_NN["hnc"] - _RECON_E_NN["hcn"]) * _KCAL
    assert by["bh76_hcn_to_hcnts"]["de_nn_kcalmol"] == pytest.approx(de)
    assert by["bh76_hcn_to_hcnts"]["abs_error_nn_kcalmol"] == pytest.approx(
        abs(de - 15.0))
    de_hnc = ((_RECON_E_NN["h"] + _RECON_E_NN["c"] + _RECON_E_NN["n"]
               - _RECON_E_NN["hnc"]) * _KCAL)
    assert by["w411_hnc_atomization"]["de_nn_kcalmol"] == pytest.approx(de_hnc)
    assert by["w411_hnc_atomization"]["pool"] == "w411"
    assert by["w411_hnc_atomization"]["reactants"] == ["hnc"]
    out = capsys.readouterr().out
    assert "verbatim" in out and "validation" in out


_NAN_RECON_E_NN = dict(_RECON_E_NN, o3=-225.10)
_NAN_RECON_E_PBE = {k: v + 0.001 for k, v in _NAN_RECON_E_NN.items()}


def _make_nan_recon_run(root, *, drop_pbe_too: bool = False):
    """Two-spec reconstruction run isolating NN-leg failures from the
    comparator: spec_0000 (deep) carries every species' energies; spec_0001
    (deep_attn) lacks o3's E_total_nn (and, with ``drop_pbe_too``, its E_pbe
    as well -- the comparator-degradation variant). Trained: the CHN
    atomization; validation slice: the co2 atomization; so each cell's test
    slice is {hnc atomization, hcn->hcnts barrier, o3 atomization}."""
    run = root / "recon/polarized/runs" / _STAMP
    run.mkdir(parents=True)
    manifest = {"n_specs": 2, "width": 4, "specs": [
        {"index": 0, "spec_file": "spec_0000.spec", "sha256": "x" * 64,
         "cell": {"arch": "deep", "subset_size": 2}},
        {"index": 1, "spec_file": "spec_0001.spec", "sha256": "y" * 64,
         "cell": {"arch": "deep_attn", "subset_size": 2}}]}
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "validation").mkdir()
    (run / "validation" / "val_reactions.json").write_text(json.dumps([
        {"name": "w411_co2_atomization", "reactants": ["co2"],
         "products": ["c", "o"], "coeffs": [-1.0, 1.0, 2.0],
         "reaction_energy_ref": 390.0}]))
    meta = {"molecules": ["CHN", "h", "c", "n"],
            "loss_kwargs": {"bh76_reactions": [
                {"name": "CHN", "reactants": ["CHN"],
                 "products": ["C", "H", "N"],
                 "coeffs": [-1.0, 1.0, 1.0, 1.0]}]}}
    for idx, nn_skip in ((0, set()), (1, {"o3"})):
        sd = run / "checkpoints" / f"spec_{idx:04d}"
        (sd / "eval_holdout").mkdir(parents=True)
        (sd / "train_metadata.json").write_text(json.dumps(meta))
        pm = []
        for m in _NAN_RECON_E_NN:
            e_nn = None if m in nn_skip else _NAN_RECON_E_NN[m]
            e_pbe = (None if (m in nn_skip and drop_pbe_too)
                     else _NAN_RECON_E_PBE[m])
            pm.append({"molecule": m, "E_total_nn": e_nn, "E_pbe": e_pbe})
        (sd / "eval_holdout" / "per_molecule.json").write_text(json.dumps(pm))
    return run


def _nan_recon_rows(tmp_path, monkeypatch, **kw):
    run = _make_nan_recon_run(tmp_path, **kw)
    monkeypatch.setattr(fig, "_canonical_pool",
                        lambda: (dict(_RECON_POOL_SPECS),
                                 list(_RECON_POOL_RXNS)))
    return fig.collect_holdout_reaction_rows(run)


def test_reconstruct_keeps_comparator_leg_on_nn_nan(tmp_path, monkeypatch,
                                                    capsys):
    # An NN SCF failure must not shrink the cell's comparator row set: the
    # o3 row survives with a finite PBE leg and NaN NN columns.
    rows = _nan_recon_rows(tmp_path, monkeypatch)
    attn = {r["name"]: r for r in rows if r["arch"] == "deep_attn"}
    assert sorted(attn) == ["bh76_hcn_to_hcnts", "w411_hnc_atomization",
                            "w411_o3_atomization"]
    o3 = attn["w411_o3_atomization"]
    assert fig._is_num(o3["abs_error_pbe_kcalmol"])
    assert math.isnan(o3["de_nn_kcalmol"])
    assert math.isnan(o3["abs_error_nn_kcalmol"])
    out = capsys.readouterr().out
    assert "NN-NaN" in out and "verbatim" in out and "validation" in out


def test_comparator_anchor_invariant_to_nn_coverage(tmp_path, monkeypatch):
    # Both cells share one test slice; the deep_attn NN failing o3 must not
    # move the cell's PBE anchor off the group's shared value.
    rows = _nan_recon_rows(tmp_path, monkeypatch)
    mae = fig.pbe_reaction_mae_by_cell(rows)
    assert mae[("deep", 2)] == mae[("deep_attn", 2)]
    wt = fig.wtmad2_pbe_by_arch_subset(rows)
    assert wt[("deep", 2)] == wt[("deep_attn", 2)]


def test_nn_metrics_exclude_nan_rows(tmp_path, monkeypatch):
    rows = _nan_recon_rows(tmp_path, monkeypatch)
    mae = fig.reaction_mae_by_arch_subset(rows)
    de_ts = (_NAN_RECON_E_NN["hnc"] - _NAN_RECON_E_NN["hcn"]) * _KCAL
    de_hnc = ((_NAN_RECON_E_NN["h"] + _NAN_RECON_E_NN["c"]
               + _NAN_RECON_E_NN["n"] - _NAN_RECON_E_NN["hnc"]) * _KCAL)
    expect = (abs(de_ts - 15.0) + abs(de_hnc - 298.7)) / 2.0
    assert mae[("deep_attn", 2)] == pytest.approx(expect)


def test_group_span_single_after_nn_nan(tmp_path, monkeypatch):
    rows = _nan_recon_rows(tmp_path, monkeypatch)
    anchors = fig.pbe_reaction_mae_by_cell(rows)
    xs, ys, hw = fig._group_span_points(anchors, ["deep", "deep_attn"],
                                        [2], 0.4)
    assert len(xs) == 1
    assert ys[0] == anchors[("deep", 2)]


def test_energy_cell_coverage_warning_names_cell(tmp_path, monkeypatch):
    rows = _nan_recon_rows(tmp_path, monkeypatch)
    w = fig._energy_cell_coverage_warning(rows)
    assert "incomplete hold-out eval" in w
    assert "deep_attn/ss2" in w and "2/3" in w
    assert "w411_o3_atomization" in w
    clean = [r for r in rows if r["arch"] == "deep"]
    assert fig._energy_cell_coverage_warning(clean) == ""


def test_span_fallback_fires_on_comparator_divergence(tmp_path, monkeypatch):
    # A missing COMPARATOR leg is a genuinely different slice: the group
    # splits into per-bar spans (the degraded-comparator detector), and the
    # NN-coverage warning stays silent -- each NN scored its own full slice.
    rows = _nan_recon_rows(tmp_path, monkeypatch, drop_pbe_too=True)
    anchors = fig.pbe_reaction_mae_by_cell(rows)
    assert anchors[("deep", 2)] != anchors[("deep_attn", 2)]
    xs, ys, hw = fig._group_span_points(anchors, ["deep", "deep_attn"],
                                        [2], 0.4)
    assert len(xs) == 2
    assert fig._energy_cell_coverage_warning(rows) == ""


def test_grouped_bars_star_marks_incomplete_cells(tmp_path):
    import matplotlib.pyplot as plt
    metric = {("a1", 2): 5.0}
    f, ax = plt.subplots()
    try:
        fig._grouped_arch_bars(ax, metric, ["a1"], [2],
                               pbe_line=8.0, title="t",
                               pbe_by_cell={("a1", 2): 6.5},
                               incomplete_cells={("a1", 2)})
        assert any(t.get_text() == "*" for t in ax.texts)
        labels = {ln.get_label() for ln in ax.lines}
        assert any(l.startswith("* incomplete hold-out eval") for l in labels)
        # the beats verdict is NOT withheld: bar 5.0 < slice anchor 6.5
        marks = {c.get_label() for c in ax.collections}
        assert "beats PBE" in marks
    finally:
        plt.close(f)
    f2, ax2 = plt.subplots()
    try:
        fig._grouped_arch_bars(ax2, metric, ["a1"], [2],
                               pbe_line=8.0, title="t",
                               pbe_by_cell={("a1", 2): 6.5})
        assert not any(t.get_text() == "*" for t in ax2.texts)
    finally:
        plt.close(f2)


def test_ed_csv_carries_slice_count_column(tmp_path):
    summary = {"gamma": 1000.0, "e_pbe": 8.0, "d_pbe": 2e-4, "ed_pbe": 8.0,
               "cells": {("deep", 1): {"E": 5.0, "D": 2e-4, "gammaD": 0.2,
                                       "ED": 4.0, "beats_pbe": False}}}
    out = fig.write_combined_ed_csv(
        {"wtmad2": summary}, tmp_path / "ed.csv",
        n_reactions={("deep", 1): 10}, n_density={("deep", 1): 5},
        n_reactions_slice={("deep", 1): 12})
    import csv as _csv
    row = next(_csv.DictReader(out.open()))
    assert row["n_reactions"] == "10"
    assert row["n_reactions_slice"] == "12"
    # 2-tuple counts_by_leg (older call shape) still accepted: blank column
    out2 = fig.write_combined_ed_csv(
        {"wtmad2": summary}, tmp_path / "ed2.csv",
        n_reactions={}, n_density={},
        counts_by_leg={"wtmad2": ({("deep", 1): 7}, {("deep", 1): 3})})
    row2 = next(_csv.DictReader(out2.open()))
    assert row2["n_reactions"] == "7"
    assert row2["n_reactions_slice"] == ""
    # 3-tuple counts_by_leg carries the per-leg slice count
    out3 = fig.write_combined_ed_csv(
        {"wtmad2": summary}, tmp_path / "ed3.csv",
        n_reactions={}, n_density={},
        counts_by_leg={"wtmad2": ({("deep", 1): 7}, {("deep", 1): 3},
                                  {("deep", 1): 9})})
    row3 = next(_csv.DictReader(out3.open()))
    assert row3["n_reactions_slice"] == "9"


def test_pbe_density_by_cell_keys_on_comparator_slice():
    # A species whose NN density leg failed still belongs to the cell's
    # comparator set (the PBE column is model-free).
    hd = [
        {"arch": "a", "subset_size": 1, "molecule": "m1",
         "density_rmse": 1e-4, "density_rmse_pbe": 2e-4},
        {"arch": "a", "subset_size": 1, "molecule": "m2",
         "density_rmse": None, "density_rmse_pbe": 4e-4},
    ]
    out = fig.pbe_density_by_cell(hd)
    assert out[("a", 1)] == pytest.approx(3e-4)


def test_parity_errbars_pair_ref_with_scored_rows(tmp_path, monkeypatch):
    # The subset-aggregate parity point averages ref and de_nn over the SAME
    # rows: an NN-NaN row (huge ref) must not enter the x mean.
    import matplotlib.axes
    recorded = []
    real = matplotlib.axes.Axes.errorbar

    def _rec(self, x, y, *a, **k):
        recorded.append((x, y))
        return real(self, x, y, *a, **k)

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", _rec)
    rows = [
        {"arch": "deep", "subset_size": 1, "pool": "bh76", "name": "r1",
         "ref_kcalmol": 10.0, "de_nn_kcalmol": 11.0,
         "abs_error_nn_kcalmol": 1.0, "abs_error_pbe_kcalmol": 2.0},
        {"arch": "deep", "subset_size": 1, "pool": "bh76", "name": "r2",
         "ref_kcalmol": 20.0, "de_nn_kcalmol": 21.0,
         "abs_error_nn_kcalmol": 1.0, "abs_error_pbe_kcalmol": 2.0},
        {"arch": "deep", "subset_size": 1, "pool": "bh76", "name": "r3",
         "ref_kcalmol": 1.0e6, "de_nn_kcalmol": float("nan"),
         "abs_error_nn_kcalmol": float("nan"),
         "abs_error_pbe_kcalmol": 2.0},
    ]
    fig.plot_parity_errbars_by_subset(rows, tmp_path / "p.png", _STAMP)
    xs = [x for x, _y in recorded if fig._is_num(x)]
    assert 15.0 in xs                    # mean over the two scored rows
    assert not any(x > 1e5 for x in xs)  # the NaN-NN row's ref stays out


def test_reaction_mae_dedup_prefers_finite_rows():
    # The pool lists four reactions twice under one name; a NaN first
    # instance must not consume the dedup name slot for a finite twin --
    # per-key finiteness precedes the seen-bookkeeping, matching
    # _cell_counts and _incomplete_energy_cells, so the reduction is
    # row-order independent when exactly one twin is finite.
    base = {"arch": "a", "subset_size": 1, "pool": "bh76",
            "ref_kcalmol": 10.0}
    nanrow = dict(base, name="r", abs_error_nn_kcalmol=float("nan"),
                  abs_error_pbe_kcalmol=2.0)
    finrow = dict(base, name="r", abs_error_nn_kcalmol=4.0,
                  abs_error_pbe_kcalmol=3.0)
    for rows in ([nanrow, finrow], [finrow, nanrow]):
        mae = fig.reaction_mae_by_arch_subset(rows)
        assert mae[("a", 1)] == pytest.approx(4.0), rows[0] is nanrow
        wt = fig.wtmad2_by_arch_subset(rows)
        assert wt[("a", 1)] == pytest.approx(
            fig._GMTKN55_SCALE * 4.0 / 10.0)


_V4_RUN = (Path.home() / "Documents/Research/xcquinox-results/runs/dfs_step7"
           / "dfs6311_grid3_v4gga/runs/run_20260810T202813Z")


@pytest.mark.slow
@pytest.mark.skipif(not _V4_RUN.is_dir(), reason="v4gga run not present")
def test_full_slice_anchor_matches_cluster_testset():
    # Independent oracle: the cluster-side test_set.csv reduces mae_pbe over
    # the finite-PBE set regardless of NN convergence. The local full-slice
    # anchors must (i) agree across every subset-size group and (ii) match
    # the cluster values on spec_0044 (deep_rung35_attn_3x16/ss1), whose NN
    # failed 9 species -- W4-11 directly; BH76 as the raw-row mean (the
    # local slice and the cluster file carry the identical row multiset),
    # with the figure's name-deduplicated anchor reconciling to the csv by
    # exactly the two duplicated BH76 pool entries (48 names over 50 rows).
    rows = fig.collect_holdout_reaction_rows(_V4_RUN)
    anchors = fig.pbe_reaction_mae_by_cell(rows)
    by_ss = {}
    for (a, s), v in anchors.items():
        by_ss.setdefault(s, []).append(v)
    # Each spec's eval re-converged its own PBE SCF, so per-spec E_pbe agree
    # only to SCF-tolerance ulps: measured cross-spec anchor spread reaches
    # 1.1e-11 relative (merged view, BH76 leg). A degraded slice moves an
    # anchor by >= 1e-3 relative (2.6e-2 measured on the pre-fix data), so
    # 1e-8 keeps three decades above the noise and five below the signal.
    for s, vals in sorted(by_ss.items()):
        lo, hi = min(vals), max(vals)
        assert (hi - lo) <= 1e-8 * max(abs(lo), abs(hi)), (s, lo, hi)
    assert anchors[("deep_3x16", 1)] == pytest.approx(12.034743071213333,
                                                      rel=1e-9)
    cell_rows = [r for r in rows
                 if r["arch"] == "deep_rung35_attn_3x16"
                 and r["subset_size"] == 1]
    w411 = fig._mae([r["abs_error_pbe_kcalmol"] for r in cell_rows
                     if r["pool"] == "w411"])
    import csv as _csv
    ts = {r["set"]: r for r in _csv.DictReader(
        (_V4_RUN / "checkpoints/spec_0044/eval_holdout/test_set.csv").open())}
    assert w411 == pytest.approx(
        float(ts["test_set_w411"]["mae_pbe_kcalmol"]), abs=5e-7)
    # BH76: same row multiset as the cluster file (the validation twins
    # never reach per_reaction.json), so the raw-row mean equals the csv;
    # the deduped figure anchor reconciles by the two duplicate rows.
    with (_V4_RUN / "checkpoints/spec_0044/eval_holdout"
          / "per_reaction.json").open() as fh:
        cluster = json.load(fh)
    bh_rows = [r for r in cell_rows if r["pool"] == "bh76"]
    assert (sorted(r["name"] for r in bh_rows)
            == sorted(str(r.get("name")) for r in cluster
                      if r.get("pool") == "bh76"))
    raw = [abs(r["abs_error_pbe_kcalmol"]) for r in bh_rows]
    csv_bh76 = float(ts["test_set_bh76"]["mae_pbe_kcalmol"])
    assert sum(raw) / len(raw) == pytest.approx(csv_bh76, abs=5e-7)
    dedup_mae = fig.reaction_mae_by_arch_subset(
        bh_rows, key="abs_error_pbe_kcalmol")[("deep_rung35_attn_3x16", 1)]
    seen: set = set()
    extras = []
    for r in bh_rows:
        if r["name"] in seen:
            extras.append(abs(r["abs_error_pbe_kcalmol"]))
        seen.add(r["name"])
    assert len(extras) == 2
    n_dedup = len(raw) - len(extras)
    assert ((n_dedup * dedup_mae + sum(extras)) / len(raw)
            == pytest.approx(csv_bh76, abs=5e-7))


def test_holdout_rows_exclude_trained_alias_reactions(tmp_path, monkeypatch,
                                                      capsys):
    # The strict filter ran name-level on the cluster; the figure layer must
    # close the naming blindness: reactions containing a pool twin of a
    # trained molecule (CHN -> hcn) are not held-out evidence.
    run = _make_leak_run(tmp_path)
    monkeypatch.setattr(fig, "_pool_specs_for_aliasing",
                        lambda: dict(_STUB_POOL_SPECS))
    rows = fig.collect_holdout_reaction_rows(run)
    names = sorted(r["name"] for r in rows)
    assert names == ["w411_co2_atomization"], names
    assert "hcn" in capsys.readouterr().out


def test_holdout_density_rows_exclude_trained_alias_species(tmp_path,
                                                            monkeypatch):
    run = _make_leak_run(tmp_path)
    monkeypatch.setattr(fig, "_pool_specs_for_aliasing",
                        lambda: dict(_STUB_POOL_SPECS))
    hd = fig.collect_holdout_density_rows(run)
    assert {r["molecule"] for r in hd} == {"co2"}


def test_holdout_rows_exclude_val_twin_reactions(tmp_path, monkeypatch):
    # The same physical barrier can sit in the validation slice under a
    # permuted-reactant name; val-best selection saw it, so its test-side
    # twin is not held-out evidence either.
    run = _make_run_dir(tmp_path)
    (run / "validation").mkdir()
    (run / "validation" / "val_reactions.json").write_text(json.dumps([
        {"name": "bh76_HO_h_to_HOh_ts", "reactants": ["h", "HO"],
         "products": ["HOh_ts"], "reaction_energy_ref": 17.7}]))
    rows = fig.collect_holdout_reaction_rows(run)
    # bh76_a (reactants HO,h -> HOh_ts) is the twin; w411_b survives.
    assert {r["name"] for r in rows} == {"w411_b"}


def test_holdout_density_rows_exclude_cross_spec_inconsistent_pbe(
        tmp_path, capsys):
    # The c2 class: two specs (arms) carrying incompatible PBE density
    # references for one species. The species must leave the density rows
    # entirely (anchor AND cell means), loudly.
    run = tmp_path / "run_x"
    for spec, val in (("spec_0000", 2.27e-4), ("spec_0001", 2.50e-3)):
        d = run / "checkpoints" / spec / "eval_holdout"
        d.mkdir(parents=True)
        (d / "per_reaction.json").write_text("[]")
        (d / "per_molecule.json").write_text(json.dumps([
            {"molecule": "c2", "density_rmse": 3e-4,
             "density_rmse_pbe": val},
            {"molecule": "h2o", "density_rmse": 1e-4,
             "density_rmse_pbe": 3e-4},
        ]))
    hd = fig.collect_holdout_density_rows(run)
    assert {r["molecule"] for r in hd} == {"h2o"}
    assert "c2" in capsys.readouterr().out


def test_density_guard_names_outlier_spec(tmp_path, capsys):
    # The density twin of the outlier attribution: the excluded species'
    # warning names the disagreeing spec and the channel that tripped.
    run = tmp_path / "run_x"
    for spec, val in (("spec_0000", 2.50e-3), ("spec_0001", 2.50e-3),
                      ("spec_0002", 2.00e-3)):
        d = run / "checkpoints" / spec / "eval_holdout"
        d.mkdir(parents=True)
        (d / "per_molecule.json").write_text(json.dumps([
            {"molecule": "c2", "density_rmse": 3e-4, "density_rmse_pbe": val},
        ]))
    fig.collect_holdout_density_rows(run)
    out = capsys.readouterr().out
    assert "outlier eval(s)" in out
    assert "spec_0002" in out and "2 specs" in out
    assert "[density_rmse_pbe]" in out


def test_cell_metrics_dedup_duplicate_reaction_names(tmp_path):
    # The pool carries four reactions twice under one name; the PBE baseline
    # dedups but the cell metrics did not, double-counting those rows.
    base = {"arch": "deep", "subset_size": 1, "pool": "bh76",
            "ref_kcalmol": 10.0}
    rows = [dict(base, name="dup", abs_error_nn_kcalmol=4.0,
                 abs_error_pbe_kcalmol=1.0),
            dict(base, name="dup", abs_error_nn_kcalmol=4.0,
                 abs_error_pbe_kcalmol=1.0),
            dict(base, name="other", abs_error_nn_kcalmol=1.0,
                 abs_error_pbe_kcalmol=1.0)]
    mae = fig.reaction_mae_by_arch_subset(rows)
    assert mae[("deep", 1)] == pytest.approx((4.0 + 1.0) / 2)
    wt = fig.wtmad2_by_arch_subset(rows)
    wt_dedup = fig.wtmad2_by_arch_subset(rows[1:])
    assert wt[("deep", 1)] == pytest.approx(wt_dedup[("deep", 1)])


def test_cell_counts_dedup_named_rows():
    # n_reactions must equal the deduped metric's effective N; unnamed
    # (density) rows keep raw counting.
    base = {"arch": "deep", "subset_size": 1, "abs_error_nn_kcalmol": 1.0}
    rows = [dict(base, name="dup"), dict(base, name="dup"),
            dict(base, name="other")]
    assert fig._cell_counts(rows, "abs_error_nn_kcalmol") == {("deep", 1): 2}
    dens = [{"arch": "deep", "subset_size": 1, "molecule": "m1",
             "density_rmse": 1e-4},
            {"arch": "deep", "subset_size": 1, "molecule": "m2",
             "density_rmse": 2e-4}]
    assert fig._cell_counts(dens, "density_rmse") == {("deep", 1): 2}


def test_wtmad2_pbe_by_arch_subset_cell_restricted(tmp_path):
    # Each cell's PBE anchor reduces exactly that cell's scored rows.
    rows = [
        {"arch": "deep", "subset_size": 1, "pool": "bh76", "name": "r1",
         "ref_kcalmol": 10.0, "abs_error_nn_kcalmol": 1.0,
         "abs_error_pbe_kcalmol": 2.0},
        {"arch": "deep", "subset_size": 2, "pool": "bh76", "name": "r1",
         "ref_kcalmol": 10.0, "abs_error_nn_kcalmol": 1.0,
         "abs_error_pbe_kcalmol": 2.0},
        {"arch": "deep", "subset_size": 2, "pool": "bh76", "name": "r2",
         "ref_kcalmol": 10.0, "abs_error_nn_kcalmol": 1.0,
         "abs_error_pbe_kcalmol": 8.0},
    ]
    by_cell = fig.wtmad2_pbe_by_arch_subset(rows)
    assert by_cell[("deep", 1)] == pytest.approx(
        fig.wtmad2_pbe_baseline(rows[:1]))
    assert by_cell[("deep", 2)] == pytest.approx(
        fig.wtmad2_pbe_baseline(rows[1:]))
    assert by_cell[("deep", 1)] != pytest.approx(by_cell[("deep", 2)])


def test_scan_by_cell_reductions_gate_per_cell():
    # Per-cell SCAN anchors reduce exactly the cell's rows, with the 90%
    # coverage floor applied PER CELL: a thinly-covered cell gets None.
    rows = [
        {"arch": "deep", "subset_size": 1, "pool": "bh76", "name": "r1",
         "ref_kcalmol": 10.0, "abs_error_pbe_kcalmol": 2.0,
         "abs_error_nn_kcalmol": 1.0},
        {"arch": "deep", "subset_size": 2, "pool": "bh76", "name": "r1",
         "ref_kcalmol": 10.0, "abs_error_pbe_kcalmol": 2.0,
         "abs_error_nn_kcalmol": 1.0},
        {"arch": "deep", "subset_size": 2, "pool": "bh76", "name": "r2",
         "ref_kcalmol": 10.0, "abs_error_pbe_kcalmol": 2.0,
         "abs_error_nn_kcalmol": 1.0},
    ]
    errs = {"r1": 3.0}      # r2 uncovered -> ss2 coverage 1/2 < 0.9
    wt = fig.wtmad2_scan_by_cell(rows, errs)
    assert wt[("deep", 1)] == pytest.approx(
        fig.wtmad2_scan_baseline(rows[:1], errs)[0])
    assert wt.get(("deep", 2)) is None or ("deep", 2) not in wt
    mae = fig.scan_reaction_mae_by_cell(rows, errs)
    assert mae[("deep", 1)] == pytest.approx(3.0)
    assert ("deep", 2) not in mae


def test_scan_density_by_cell_gates_per_cell():
    hd = [
        {"arch": "deep", "subset_size": 1, "molecule": "HO",
         "density_rmse": 1e-4, "density_rmse_pbe": 8e-4},
        {"arch": "deep", "subset_size": 2, "molecule": "HO",
         "density_rmse": 1e-4, "density_rmse_pbe": 8e-4},
        {"arch": "deep", "subset_size": 2, "molecule": "CH4",
         "density_rmse": 2e-4, "density_rmse_pbe": 3e-4},
    ]
    recs = {"HO": {"density_rmse_scan": 2e-4}}   # CH4 uncovered
    out = fig.scan_density_by_cell(hd, recs)
    assert out[("deep", 1)] == pytest.approx(2e-4)
    assert ("deep", 2) not in out


def test_beats_scan_uses_cell_matched_anchor():
    # Same misgrading class as the PBE anchors: a cell below the pooled SCAN
    # ED but above its own-rows SCAN ED must not read "beats SCAN".
    e_cells = {("deep", 26): 3.0}
    d_cells = {("deep", 26): 2.0e-4}
    s = fig.combined_ed_by_cell(
        e_cells, 8.0, d_cells, 2.4e-4,
        e_scan=6.0, d_scan=2.4e-4,
        e_scan_by_cell={("deep", 26): 2.0},
        d_scan_by_cell={("deep", 26): 1.5e-4})
    c = s["cells"][("deep", 26)]
    assert c["ED"] < s["ed_scan"]
    assert fig._is_num(c.get("ed_scan_cell")) and c["ed_scan_cell"] < c["ED"]
    assert c["beats_scan"] is False
    # pooled fallback keeps prior semantics when no cell legs are given
    s2 = fig.combined_ed_by_cell(e_cells, 8.0, d_cells, 2.4e-4,
                                 e_scan=6.0, d_scan=2.4e-4)
    assert s2["cells"][("deep", 26)]["beats_scan"] is True
    assert s2["cells"][("deep", 26)]["ed_scan_cell"] is None


def test_ed_csv_carries_scan_cell_anchor_column(tmp_path):
    summary = {"gamma": 1000.0, "e_pbe": 8.0, "d_pbe": 2e-4, "ed_pbe": 8.0,
               "e_scan": 4.0, "d_scan": 1.5e-4, "ed_scan": 3.5,
               "cells": {("deep", 1): {"E": 5.0, "D": 2e-4, "gammaD": 0.2,
                                       "ED": 4.0, "beats_pbe": False,
                                       "beats_scan": False,
                                       "ed_pbe_cell": 3.5,
                                       "ed_scan_cell": 3.2}}}
    out = fig.write_combined_ed_csv({"wtmad2": summary}, tmp_path / "ed.csv",
                                    n_reactions={("deep", 1): 10},
                                    n_density={("deep", 1): 5})
    import csv as _csv
    row = next(_csv.DictReader(out.open()))
    assert row["ED_scan_cell_kcalmol"] == "3.2"


def test_grouped_bars_scan_cell_ticks(tmp_path):
    import matplotlib.pyplot as plt
    metric = {("deep", 2): 5.0}
    f, ax = plt.subplots()
    try:
        fig._grouped_arch_bars(ax, metric, ["deep"], [2],
                               pbe_line=8.0, title="t",
                               scan_line=6.0,
                               scan_by_cell={("deep", 2): 4.5})
        labels = {c.get_label() for c in ax.containers}
        assert "SCAN (cell rows)" in labels, labels
        lines = {ln.get_label() for ln in ax.lines}
        assert "SCAN (pooled)" in lines, lines
    finally:
        plt.close(f)


def test_cell_rows_spans_are_capped_errorbars(tmp_path):
    # The cell-rows comparator mark is a capped horizontal span (error-bar
    # style: horizontal segment + vertical end caps demarking the group's
    # extent), black PBE / grey SCAN, and the pooled PBE line is dash-dot
    # (the thick dashed line read as another data element).
    import matplotlib.pyplot as plt
    metric = {("deep", 2): 5.0}
    f, ax = plt.subplots()
    try:
        fig._grouped_arch_bars(ax, metric, ["deep"], [2],
                               pbe_line=8.0, title="t", scan_line=6.0,
                               pbe_by_cell={("deep", 2): 6.5},
                               scan_by_cell={("deep", 2): 4.5})
        for lbl in ("PBE (cell rows)", "SCAN (cell rows)"):
            cont = next(c for c in ax.containers if c.get_label() == lbl)
            _data, caplines, barcols = cont
            assert len(caplines) == 2, lbl       # end caps on both sides
            segs = barcols[0].get_segments()
            assert len(segs) == 1, lbl
            (x0, y0), (x1, y1) = segs[0]
            assert y0 == y1                      # horizontal
            assert x1 > x0                       # finite span
        pooled = next(ln for ln in ax.lines
                      if ln.get_label() == "PBE (pooled)")
        assert pooled.get_linestyle() == "-."
    finally:
        plt.close(f)


def test_cell_anchor_note_explains_glyph():
    note = fig._cell_anchor_note({("deep", 2): 5.0})
    low = note.lower()
    assert "capped" in low and "span" in low
    assert "pbe" in low and "scan" in low
    assert "slice" in low
    # the ED line/scatter figures draw no glyphs: their note points at the
    # CSV columns instead of describing glyphs they do not carry
    flat = fig._cell_anchor_note({("deep", 2): 5.0}, glyphs=False)
    assert "capped" not in flat.lower()
    assert "CSV" in flat and "slice" in flat.lower()
    # the drawing docstrings describe the current marks, the incomplete-eval
    # star included
    assert "incomplete" in (fig._grouped_arch_bars.__doc__ or "").lower()
    assert "capped" in (fig._grouped_arch_bars.__doc__ or "").lower()
    assert "capped" in (fig.plot_energy_wtmad_mae.__doc__ or "").lower()


def test_mae_by_arch_note_explains_row_matched_lines(tmp_path, monkeypatch):
    # Two numbers for one functional on one figure (row-matched line vs the
    # full-pool footer) need the distinction stated where they are read.
    import matplotlib.figure as mfig
    seen = []
    real = mfig.Figure.savefig

    def _cap(self, *a, **k):
        seen.extend(t.get_text() for t in self.texts)
        return real(self, *a, **k)

    monkeypatch.setattr(mfig.Figure, "savefig", _cap)
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    ins = fig.collect_insample_ae_rows(run)
    errs = {str(r["name"]): 1.0 for r in rows}
    fig.plot_mae_by_arch(rows, ins, tmp_path / "m.png", "run",
                         scan_baseline={"bh76": 9.0, "w411": 9.0,
                                        "combined": 9.0}, scan_errors=errs)
    joined = " ".join(seen)
    assert "own deduped held-out rows" in joined
    assert "full-pool" in joined


def test_wtmad_glyph_note_survives_poolless_rows(tmp_path, monkeypatch):
    # Rows without pool labels empty the WTMAD-2 anchors while the MAE panel
    # still draws glyphs -- the key must reach the figure from either map.
    import matplotlib.figure as mfig
    seen = []
    real = mfig.Figure.savefig

    def _cap(self, *a, **k):
        seen.extend(t.get_text() for t in self.texts)
        return real(self, *a, **k)

    monkeypatch.setattr(mfig.Figure, "savefig", _cap)
    rows = [{"arch": "deep", "subset_size": 1, "name": f"r{i}", "pool": None,
             "ref_kcalmol": 10.0, "abs_error_nn_kcalmol": 1.0 + i,
             "abs_error_pbe_kcalmol": 2.0 + i} for i in range(3)]
    assert fig.wtmad2_pbe_by_arch_subset(rows) == {}
    assert fig.pbe_reaction_mae_by_cell(rows)
    fig.plot_energy_wtmad_mae(rows, tmp_path / "w.png", "run")
    joined = " ".join(seen).lower()
    assert "capped" in joined


def test_cell_rows_spans_one_per_group(tmp_path):
    # A subset-size group's cells score the same test slice, so their
    # anchors agree to fp noise: the mark is ONE capped span across the
    # whole group's bar cluster, not one per bar.
    import matplotlib.pyplot as plt
    metric = {("a1", 2): 5.0, ("a2", 2): 6.0}
    agree = {("a1", 2): 7.0, ("a2", 2): 7.0 + 1e-9}
    f, ax = plt.subplots()
    try:
        fig._grouped_arch_bars(ax, metric, ["a1", "a2"], [2],
                               pbe_line=8.0, title="t", pbe_by_cell=agree)
        cont = next(c for c in ax.containers
                    if c.get_label() == "PBE (cell rows)")
        segs = cont[2][0].get_segments()
        assert len(segs) == 1
        (x0, y0), (x1, y1) = segs[0]
        assert y0 == pytest.approx(7.0) and y1 == pytest.approx(7.0)
        # spans the full bar cluster (2 archs x bw 0.4 -> +-0.4 around the
        # group center at x=0)
        assert x0 == pytest.approx(-0.4) and x1 == pytest.approx(0.4)
    finally:
        plt.close(f)


def test_cell_rows_spans_split_on_disagreement(tmp_path):
    # A group whose cells genuinely disagree (a degraded eval scored a
    # different row set) keeps per-bar spans so the divergence is visible.
    import matplotlib.pyplot as plt
    metric = {("a1", 2): 5.0, ("a2", 2): 6.0}
    split = {("a1", 2): 7.0, ("a2", 2): 7.5}
    f, ax = plt.subplots()
    try:
        fig._grouped_arch_bars(ax, metric, ["a1", "a2"], [2],
                               pbe_line=8.0, title="t", pbe_by_cell=split)
        cont = next(c for c in ax.containers
                    if c.get_label() == "PBE (cell rows)")
        segs = sorted(cont[2][0].get_segments(),
                      key=lambda s: s[0][0])
        assert len(segs) == 2
        assert segs[0][0][1] == pytest.approx(7.0)     # a1's own value
        assert segs[1][0][1] == pytest.approx(7.5)     # a2's own value
        # bar-width spans (bw 0.4, shrunk 0.45*bw = 0.18) at bars -+0.2
        assert segs[0][0][0] == pytest.approx(-0.38)
        assert segs[0][1][0] == pytest.approx(-0.02)
        assert segs[1][0][0] == pytest.approx(0.02)
        assert segs[1][1][0] == pytest.approx(0.38)
    finally:
        plt.close(f)


def test_density_parity_by_channel_has_legend(tmp_path, monkeypatch):
    # The channel panels color every species point by architecture with no
    # per-point labels; the figure must carry an arch legend.
    import matplotlib.figure as mfig
    cap = {}
    real = mfig.Figure.savefig

    def _c(self, *a, **k):
        cap["labels"] = [t.get_text() for lg in self.legends
                         for t in lg.get_texts()]
        return real(self, *a, **k)

    monkeypatch.setattr(mfig.Figure, "savefig", _c)
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    rows = fig.collect_holdout_reaction_rows(run)
    hd = fig.collect_holdout_density_rows(run)
    fig.plot_density_parity_by_channel(rows, hd, tmp_path / "dp.png", "run")
    assert cap.get("labels"), "no figure legend on the density parity"
    assert "deep" in cap["labels"]


def test_plot_parity_by_class_grid(tmp_path, monkeypatch):
    # 2x3: AE | BH76 | total columns; by-architecture row over by-subset
    # row; one figure legend and one shared subset colorbar.
    import matplotlib.figure as mfig
    cap = {}
    real = mfig.Figure.savefig

    def _c(self, *a, **k):
        cap["titles"] = [ax.get_title() for ax in self.axes]
        cap["n_legends"] = len(self.legends)
        return real(self, *a, **k)

    monkeypatch.setattr(mfig.Figure, "savefig", _c)
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    out = fig.plot_parity_by_class(rows, tmp_path / "pc.png", "run")
    assert _png_ok(out)
    titles = cap["titles"]
    assert "W4-11" in titles[0] and "architecture" in titles[0]
    assert "BH76" in titles[1]
    assert "total" in titles[2]
    assert "subset" in titles[3]
    assert cap["n_legends"] >= 1


def test_dfs_units_figures_carry_glyph_note(tmp_path, monkeypatch):
    # Every figure drawing the cell-rows glyphs must explain them -- the
    # DFS-units overview/3x3 twins included (they carried no glyph key).
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    rows = fig.collect_holdout_reaction_rows(run)
    errs = {str(r["name"]): 1.0 for r in rows}
    monkeypatch.setattr(fig, "scan_reaction_errors", lambda *a, **k: errs)
    monkeypatch.setattr(fig, "_scan_density_records",
                        lambda *a, **k: dict(_SCAN_FIXTURE_RECORDS))
    provs = {}
    flat_provs = {}
    for name in ("plot_density_energy_overview", "plot_density_energy_3x3"):
        real = getattr(fig, name)

        def _cap(*a, _real=real, _name=name, **kw):
            out_path = a[2] if len(a) > 2 else kw.get("out_path")
            provs[(_name, str(out_path))] = kw.get("provenance")
            return _real(*a, **kw)

        monkeypatch.setattr(fig, name, _cap)
    # The ED line/scatter figures draw no glyphs: their provenance must NOT
    # claim asterisks and must point at the CSV columns instead.
    for name in ("plot_combined_energy_density", "plot_ed_decomposition"):
        real = getattr(fig, name)

        def _capf(*a, _real=real, _name=name, **kw):
            flat_provs[_name] = kw.get("provenance")
            return _real(*a, **kw)

        monkeypatch.setattr(fig, name, _capf)
    fig.build_density_energy_figures(run, tmp_path / "out")
    dfs_units = {k: v for k, v in provs.items() if "_dfs_units" in k[1]}
    assert dfs_units, "DFS-units figures did not render on the fixture"
    for key, prov in provs.items():
        assert prov and "capped" in prov.lower(), (key, prov)
    assert flat_provs, "ED line figures did not render on the fixture"
    for key, prov in flat_provs.items():
        assert prov and "capped" not in prov.lower(), (key, prov)
        assert "CSV" in prov, (key, prov)


def test_pbe_density_by_cell_iterates_sorted_species():
    # np.mean over a hash-ordered set permutes the fp summation between
    # processes (PYTHONHASHSEED), moving the per-cell anchors by ulps
    # between renders of identical data; the species must be visited
    # sorted so regenerated CSVs are byte-reproducible.
    class _Rec(dict):
        def __init__(self, d):
            super().__init__(d)
            self.order = []

        def __getitem__(self, k):
            self.order.append(k)
            return super().__getitem__(k)

    names = [f"m{i:02d}" for i in range(12)]
    rec = _Rec({n: 1.0 + i * 1e-3 for i, n in enumerate(names)})
    rows = [{"arch": "deep", "subset_size": 1, "molecule": n,
             "density_rmse": 1e-4} for n in reversed(names)]
    out = fig.pbe_density_by_cell(rows, {}, _pbe_mol=rec)
    assert ("deep", 1) in out
    assert rec.order == sorted(rec.order)


def test_scan_row_matched_ref_prefers_row_set():
    rows = [
        {"name": "r1", "pool": "bh76", "ref_kcalmol": 10.0,
         "abs_error_nn_kcalmol": 1.0, "abs_error_pbe_kcalmol": 3.0},
        {"name": "r2", "pool": "w411", "ref_kcalmol": 50.0,
         "abs_error_nn_kcalmol": 2.0, "abs_error_pbe_kcalmol": 5.0},
    ]
    pooled = {"bh76": 9.0, "w411": 9.0, "combined": 9.0,
              "coverage": {"combined": {"used": 2, "reference": 2}}}
    # Full row coverage: the row-matched reduction wins over the pooled value.
    val, label = fig.scan_row_matched_ref(rows, {"r1": 1.0, "r2": 3.0}, pooled)
    assert val == pytest.approx(2.0)
    assert "row-matched" in label
    # Absent cache: the pooled line exactly as before.
    val, label = fig.scan_row_matched_ref(rows, {}, pooled)
    assert val == pytest.approx(9.0)
    assert "full-pool" in label
    # Cache below the coverage floor: pooled fallback, never a half-covered
    # row reduction.
    val, label = fig.scan_row_matched_ref(rows, {"r1": 1.0}, pooled)
    assert val == pytest.approx(9.0)
    assert "full-pool" in label


def test_plot_mae_by_arch_renders_with_scan_errors(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    ins = fig.collect_insample_ae_rows(run)
    errs = {str(r["name"]): 1.0 for r in rows}
    out = fig.plot_mae_by_arch(rows, ins, tmp_path / "mae.png", "run",
                               scan_baseline={"bh76": 9.0, "w411": 9.0,
                                              "combined": 9.0},
                               scan_errors=errs)
    assert _png_ok(out)


def test_arch_reference_kinds_by_rung():
    # The green marker claims improvement over the arch's OWN-RUNG
    # nonempirical reference: PBE for pure-GGA architectures, SCAN for any
    # architecture carrying beyond-GGA information (meta-GGA, rung-3.5,
    # stacked -- the rung-3.5 assignment is the conservative convention).
    kinds = fig.arch_reference_kinds(
        ["deep_3x16", "deep_attn_3x16", "deep_mgga_3x16",
         "deep_rung35_3x16", "deep_rung35_mgga_3x16"])
    assert kinds["deep_3x16"] == "pbe"
    assert kinds["deep_attn_3x16"] == "pbe"
    assert kinds["deep_mgga_3x16"] == "scan"
    assert kinds["deep_rung35_3x16"] == "scan"
    assert kinds["deep_rung35_mgga_3x16"] == "scan"


def test_grouped_bars_rung_reference_marks(tmp_path):
    # A meta-GGA bar below the PBE tick but above the SCAN tick must NOT be
    # marked; a GGA bar in the same panel keeps its PBE grading; a
    # SCAN-referenced arch with NO scan anchor gets no mark at all.
    import matplotlib.pyplot as plt
    metric = {("deep_3x16", 2): 5.0, ("deep_mgga_3x16", 2): 5.0,
              ("deep_rung35_3x16", 2): 5.0}
    f, ax = plt.subplots()
    try:
        fig._grouped_arch_bars(
            ax, metric, ["deep_3x16", "deep_mgga_3x16", "deep_rung35_3x16"],
            [2], pbe_line=8.0, title="t", scan_line=6.0,
            pbe_by_cell={("deep_3x16", 2): 6.0, ("deep_mgga_3x16", 2): 6.0,
                         ("deep_rung35_3x16", 2): 6.0},
            scan_by_cell={("deep_mgga_3x16", 2): 4.5},
            reference_by_arch={"deep_3x16": "pbe", "deep_mgga_3x16": "scan",
                               "deep_rung35_3x16": "scan"})
        beat = [c for c in ax.collections if "beats" in str(c.get_label())]
        # only the GGA bar (5.0 < its PBE tick 6.0) marks: the mgga bar is
        # above its SCAN tick (4.5) and the rung35 bar has no SCAN anchor.
        assert beat and len(beat[0].get_offsets()) == 1, [
            (c.get_label(), len(c.get_offsets())) for c in ax.collections]
    finally:
        plt.close(f)


def test_ed_lines_panel_marks_by_rung_reference():
    # The ED line panel's green markers follow the per-arch reference:
    # a meta-GGA cell that beats PBE but not SCAN is unmarked.
    import matplotlib.pyplot as plt
    cells = {
        ("deep_3x16", 2): {"E": 5.0, "D": 2e-4, "gammaD": 5.0, "ED": 5.0,
                           "beats_pbe": True, "beats_scan": False,
                           "ed_pbe_cell": 6.0, "ed_scan_cell": 4.5},
        ("deep_mgga_3x16", 2): {"E": 5.0, "D": 2e-4, "gammaD": 5.0,
                                "ED": 5.0, "beats_pbe": True,
                                "beats_scan": False,
                                "ed_pbe_cell": 6.0, "ed_scan_cell": 4.5},
    }
    summary = {"gamma": 25000.0, "gamma_mode": "fixed", "e_pbe": 8.0,
               "d_pbe": 2.4e-4, "ed_pbe": 7.0, "ed_scan": 5.5,
               "cells": cells}
    f, ax = plt.subplots()
    try:
        fig._ed_lines_panel(ax, summary, "t",
                            reference_by_arch={"deep_3x16": "pbe",
                                               "deep_mgga_3x16": "scan"})
        beat = [c for c in ax.collections if "beats" in str(c.get_label())]
        assert beat and len(beat[0].get_offsets()) == 1, [
            (c.get_label(), len(c.get_offsets())) for c in ax.collections]
    finally:
        plt.close(f)


def test_beats_pbe_uses_cell_matched_anchor():
    # A cell below the pooled union anchor but above its own-rows anchor must
    # NOT read "beats PBE" (the deep_3x16 ss26 flip class). The verdict
    # anchor is the harmonic ED of the CELL-MATCHED PBE legs under the
    # summary's (global) gamma.
    e_cells = {("deep", 26): 6.8}
    d_cells = {("deep", 26): 2.0e-4}
    s = fig.combined_ed_by_cell(
        e_cells, 8.78, d_cells, 2.4e-4,
        e_pbe_by_cell={("deep", 26): 5.9},
        d_pbe_by_cell={("deep", 26): 1.8e-4})
    c = s["cells"][("deep", 26)]
    # cell ED ~7.05 sits below the pooled ed_pbe (8.78) but above its
    # cell-matched anchor (~6.22): the verdict must be False.
    assert c["ED"] < s["ed_pbe"]
    assert fig._is_num(c.get("ed_pbe_cell")) and c["ed_pbe_cell"] < c["ED"]
    assert c["beats_pbe"] is False
    # and a genuinely-beaten cell anchor still marks
    s2 = fig.combined_ed_by_cell(
        e_cells, 8.78, d_cells, 2.4e-4,
        e_pbe_by_cell={("deep", 26): 8.5},
        d_pbe_by_cell={("deep", 26): 2.4e-4})
    assert s2["cells"][("deep", 26)]["beats_pbe"] is True
    # without cell anchors the pooled fallback keeps the old semantics
    s3 = fig.combined_ed_by_cell(e_cells, 8.78, d_cells, 2.4e-4)
    assert s3["cells"][("deep", 26)]["beats_pbe"] is True
    assert s3["cells"][("deep", 26)]["ed_pbe_cell"] is None


def test_ed_csv_carries_cell_anchor_column(tmp_path):
    summary = {"gamma": 1000.0, "e_pbe": 8.0, "d_pbe": 2e-4, "ed_pbe": 8.0,
               "e_scan": None, "d_scan": None, "ed_scan": None,
               "cells": {("deep", 1): {"E": 5.0, "D": 2e-4, "gammaD": 0.2,
                                       "ED": 4.0, "beats_pbe": False,
                                       "beats_scan": None,
                                       "ed_pbe_cell": 3.5}}}
    out = fig.write_combined_ed_csv({"wtmad2": summary}, tmp_path / "ed.csv",
                                    n_reactions={("deep", 1): 10},
                                    n_density={("deep", 1): 5})
    import csv as _csv
    row = next(_csv.DictReader(out.open()))
    assert row["ED_pbe_cell_kcalmol"] == "3.5"


def test_grouped_bars_cell_anchor_marks(tmp_path):
    # With per-cell anchors, a bar below the pooled line but above its own
    # anchor gets no beats-PBE mark; the cell anchors are drawn as ticks.
    import matplotlib.pyplot as plt
    metric = {("deep", 26): 6.8, ("deep", 2): 5.0}
    f, ax = plt.subplots()
    try:
        fig._grouped_arch_bars(ax, metric, ["deep"], [2, 26],
                               pbe_line=8.78, title="t",
                               pbe_by_cell={("deep", 26): 5.9,
                                            ("deep", 2): 8.9})
        beat = [c for c in ax.collections
                if c.get_label() == "beats PBE"]
        assert beat and len(beat[0].get_offsets()) == 1  # only ss2 beats
    finally:
        plt.close(f)


def test_suite_scan_console_absent_note_once(tmp_path, monkeypatch, capsys):
    # Guard for the consolidation: the loud absent-cache note still appears,
    # exactly once, when no SCAN cache resolves.
    run = _make_run_dir(tmp_path)
    nan = float("nan")
    monkeypatch.setattr(fig, "scan_pool_baseline",
                        lambda *a, **k: {"bh76": nan, "w411": nan,
                                         "combined": nan})
    fig.build_all(run, tmp_path / "o1")
    fig.build_density_energy_figures(run, tmp_path / "o2")
    out = capsys.readouterr().out
    assert out.count("no SCAN cache next to the run") == 1, out


# ---------------------------------------------------------------------------
# Dynamic (non-hardcoded) footer baselines
# ---------------------------------------------------------------------------

_KCAL_PER_HA = 627.5094740631
_SVP_RUN = (Path.home() / "Documents/Research/xcquinox-results/runs"
            / "bh76w411_repr/svp_grid2/runs/run_20260603T163407Z")


def test_pbe_pool_baseline_computes_full_pool_mae(tmp_path):
    """Full-pool PBE MAE per pool + combined, from per_molecule PBE energies and
    an injected reaction pool (test seam). Hand-checked against the arithmetic."""
    fake_rxns = [
        {"name": "rb", "source_pool": "bh76", "reactants": ["a"], "products": ["b"],
         "coeffs": [-1.0, 1.0], "reaction_energy_ref": 10.0},   # de=12 -> |err|=2
        {"name": "rw", "source_pool": "w411", "reactants": ["a"], "products": ["c"],
         "coeffs": [-1.0, 1.0], "reaction_energy_ref": 100.0},  # de=90 -> |err|=10
    ]
    e_a = -1.0
    e_b = e_a + 12.0 / _KCAL_PER_HA
    e_c = e_a + 90.0 / _KCAL_PER_HA
    eh = tmp_path / "checkpoints" / "spec_0000" / "eval_holdout"
    eh.mkdir(parents=True)
    (eh / "per_molecule.json").write_text(json.dumps([
        {"molecule": "a", "E_pbe": e_a}, {"molecule": "b", "E_pbe": e_b},
        {"molecule": "c", "E_pbe": e_c}]))
    base = fig.pbe_pool_baseline(tmp_path, _loader=lambda: ({}, fake_rxns))
    assert base["bh76"] == pytest.approx(2.0, abs=1e-6)
    assert base["w411"] == pytest.approx(10.0, abs=1e-6)
    assert base["combined"] == pytest.approx((2.0 + 10.0) / 2, abs=1e-6)


def test_pbe_pool_baseline_missing_energies_is_nan(tmp_path):
    import math
    (tmp_path / "checkpoints").mkdir()  # no per_molecule.json anywhere
    base = fig.pbe_pool_baseline(tmp_path, _loader=lambda: ({}, [
        {"name": "rb", "source_pool": "bh76", "reactants": ["a"], "products": ["b"],
         "coeffs": [-1.0, 1.0], "reaction_energy_ref": 10.0}]))
    assert math.isnan(base["bh76"]) and math.isnan(base["combined"])


def test_provenance_footer_uses_live_baseline():
    s = fig.provenance_footer({"bh76": 11.825, "w411": 15.938, "combined": 14.490})
    assert "BH76 11.82" in s and "W4-11 15.94" in s and "combined 14.49" in s
    assert "GMTKN55-BH76" in s          # static methodology prefix preserved
    assert "11.83 / W4-11 15.93" not in s  # the OLD hardcoded string is gone


def test_provenance_footer_handles_missing_baseline():
    s = fig.provenance_footer({"bh76": float("nan"), "w411": None,
                               "combined": float("nan")})
    assert "n/a" in s


def test_provenance_footer_labels_full_pool():
    # the PBE/SCAN baselines are computed on the FULL canonical pool, not the
    # test slice the NN cells are evaluated on -- the label must say so
    s = fig.provenance_footer({"bh76": 11.8, "w411": 15.9, "combined": 14.5})
    assert "PBE (full pool):" in s
    s2 = fig.provenance_footer({"bh76": 11.8, "w411": 15.9, "combined": 14.5},
                               {"bh76": 8.0, "w411": 9.0, "combined": 8.5})
    assert "SCAN (full pool):" in s2


def test_energy_figures_accept_dataset_line(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    ds = fig._holdout_eval_note(rows, [])
    p1 = fig.plot_energy_wtmad_mae(rows, tmp_path / "ew.png", _STAMP,
                                   dataset=ds)
    assert _png_ok(p1)
    p2 = fig.plot_rung_summary(rows, tmp_path / "rs.png", _STAMP,
                               pbe_baseline={"bh76": 10.0, "w411": 3.0,
                                             "combined": 6.0},
                               dataset=ds)
    assert _png_ok(p2)
    p3 = fig.plot_parity_marginal(rows, tmp_path / "pm.png", _STAMP,
                                  dataset=ds)
    assert _png_ok(p3)


def test_nn_vs_pbe_caveat_picks_best_bh76_cell():
    rows = [
        {"arch": "deep", "subset_size": 5, "pool": "bh76", "abs_error_nn_kcalmol": 6.0},
        {"arch": "deep", "subset_size": 5, "pool": "bh76", "abs_error_nn_kcalmol": 8.0},
        {"arch": "deep_attn", "subset_size": 3, "pool": "bh76", "abs_error_nn_kcalmol": 20.0},
        {"arch": "deep", "subset_size": 5, "pool": "w411", "abs_error_nn_kcalmol": 99.0},
    ]
    s = fig.nn_vs_pbe_caveat(rows, {"bh76": 11.83})
    assert "deep/subset-5" in s and "7.00" in s   # best cell = mean(6,8)=7
    assert "1/2" in s                              # 1 of 2 bh76 cells beats 11.83
    assert "11.83" in s


def test_nn_vs_pbe_caveat_insufficient_data():
    assert "insufficient" in fig.nn_vs_pbe_caveat([], {"bh76": float("nan")})


@pytest.mark.slow
@pytest.mark.skipif(not _SVP_RUN.is_dir(), reason="svp run not present")
def test_pbe_pool_baseline_matches_validated_full_pool():
    base = fig.pbe_pool_baseline(_SVP_RUN)
    assert base["bh76"] == pytest.approx(11.82, abs=0.05)
    assert base["w411"] == pytest.approx(15.94, abs=0.05)
    assert base["combined"] == pytest.approx(14.49, abs=0.05)


# ---------------------------------------------------------------------------
# Parity layout variants (pools separated by scale; all arch x subset shown)
# ---------------------------------------------------------------------------

def test_pool_parity_limits_separates_scales():
    rows = [
        {"pool": "bh76", "arch": "deep", "subset_size": 1, "ref_kcalmol": 10.0,
         "de_nn_kcalmol": 12.0, "de_pbe_kcalmol": 11.0},
        {"pool": "bh76", "arch": "deep", "subset_size": 1, "ref_kcalmol": -5.0,
         "de_nn_kcalmol": -4.0, "de_pbe_kcalmol": -6.0},
        {"pool": "w411", "arch": "deep", "subset_size": 1, "ref_kcalmol": 900.0,
         "de_nn_kcalmol": 880.0, "de_pbe_kcalmol": 910.0},
    ]
    lo, hi = fig._pool_parity_limits(rows, "bh76")
    assert lo < -5 and 12 < hi < 100          # bh76-only window, not pulled to 900
    lo2, hi2 = fig._pool_parity_limits(rows, "w411")
    assert lo2 > 100 and lo2 < 900 < hi2      # w411 lives on its own scale


def test_plot_parity_marginal_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    out = fig.plot_parity_marginal(rows, tmp_path / "m.png", _STAMP)
    assert _png_ok(out)


def test_plot_parity_facet_subset_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    out = fig.plot_parity_facet_subset(rows, tmp_path / "f.png", _STAMP)
    assert _png_ok(out)


def test_plot_parity_arch_cols_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    out = fig.plot_parity_arch_cols(rows, tmp_path / "a.png", _STAMP)
    assert _png_ok(out)


def test_plot_parity_errbars_by_subset_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    out = fig.plot_parity_errbars_by_subset(rows, tmp_path / "e.png", _STAMP)
    assert _png_ok(out)


def test_plot_parity_grid_by_subset_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    out = fig.plot_parity_grid_by_subset(rows, tmp_path / "g.png", _STAMP)
    assert _png_ok(out)


def test_build_parity_variants_writes_five(tmp_path):
    run = _make_run_dir(tmp_path)
    written = fig.build_parity_variants(run, tmp_path / "out")
    assert len(written) == 5
    assert all(_png_ok(p) for p in written)
    names = {p.name for p in written}
    assert names == {"ablation_parity_arch_cols.png",
                     "ablation_parity_marginal_2x2.png",
                     "ablation_parity_facet_subset.png",
                     "ablation_parity_errbars_by_subset.png",
                     "ablation_parity_grid_by_subset.png"}


# ---------------------------------------------------------------------------
# 2-subset WTMAD-2 energy metric + in-sample density-vs-CCSD diagnostic
# ---------------------------------------------------------------------------

def test_wtmad2_by_arch_subset_reweights_pools():
    # one (deep, ss=1) cell; bh76 MAD=5 over |ref|mean=20; w411 MAD=30 over |ref|mean=300.
    rows = [
        {"arch": "deep", "subset_size": 1, "pool": "bh76",
         "abs_error_nn_kcalmol": 4.0, "reaction_energy_ref_kcalmol": 10.0},
        {"arch": "deep", "subset_size": 1, "pool": "bh76",
         "abs_error_nn_kcalmol": 6.0, "reaction_energy_ref_kcalmol": 30.0},
        {"arch": "deep", "subset_size": 1, "pool": "w411",
         "abs_error_nn_kcalmol": 20.0, "reaction_energy_ref_kcalmol": 200.0},
        {"arch": "deep", "subset_size": 1, "pool": "w411",
         "abs_error_nn_kcalmol": 40.0, "reaction_energy_ref_kcalmol": 400.0},
    ]
    w = fig.wtmad2_by_arch_subset(rows, scale=56.84)
    # (56.84/4)*(2*5/20 + 2*30/300) = 14.21*(0.5+0.2) = 9.947
    assert w[("deep", 1)] == pytest.approx(9.947, abs=1e-2)


def test_wtmad2_handles_empty_and_missing():
    assert fig.wtmad2_by_arch_subset([]) == {}
    # a pool with zero |ref| denominator must not blow up
    rows = [{"arch": "deep", "subset_size": 1, "pool": "bh76",
             "abs_error_nn_kcalmol": 5.0, "reaction_energy_ref_kcalmol": 0.0}]
    out = fig.wtmad2_by_arch_subset(rows)
    assert ("deep", 1) not in out or fig._is_num(out[("deep", 1)])


def test_collect_insample_density_rows_drops_atoms(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_insample_density_rows(run)
    # 4 evaluated specs x {HO, CH4} finite density_rmse = 8 (H atom None + X missing dropped).
    assert len(rows) == 8
    assert all(fig._is_num(r["density_rmse"]) for r in rows)
    assert {r["molecule"] for r in rows} == {"HO", "CH4"}
    assert all(r["subset_size"] in (1, 3) for r in rows)
    assert {r["arch"] for r in rows} == {"deep", "deep_notransform"}


def test_training_subsets_by_size(tmp_path):
    run = _make_run_dir(tmp_path)
    ts = fig.training_subsets_by_size(run)
    # element anchors (h, o, c) filtered; molecules sorted; one entry per size.
    assert ts == {1: ["HO"], 3: ["CH4", "HO"]}


def test_training_reactions_by_size_real_ledger(tmp_path):
    # uses the in-repo ledger + BH76 pool JSON (authoritative reaction defs)
    run = tmp_path / "run"
    run.mkdir()
    (run / "resolved_config.yaml").write_text(
        "subset_ledger_path: /gpfs/x/hpcjobs/ledgers/"
        "bh76w411_repr_alpha_on_r1-6.json\n")
    out = fig.training_reactions_by_size(run)
    # ss6 = exactly 5 W4-11 atomizations + 1 BH76 reaction (NOT 7 species)
    assert len(out[6]["ae"]) == 5 and len(out[6]["rxn"]) == 1
    assert out[6]["rxn"][0] == (["clch3clcomp"], ["clch3clts"])  # one SN2 complex->TS
    assert "b2h6" in out[6]["ae"] and "ocs" in out[6]["ae"]
    # ss2 reaction CH3 + ClF -> ch3fclts
    assert out[2]["rxn"] == [(["ch3", "clf"], ["ch3fclts"])]
    assert out[2]["ae"] == ["hocn"]
    # ss1 = a single atomization, no reactions
    assert out[1]["ae"] == ["hocn"] and out[1]["rxn"] == []


def test_plot_energy_wtmad_mae_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    out = fig.plot_energy_wtmad_mae(rows, tmp_path / "wt.png", _STAMP)
    assert _png_ok(out)


def test_plot_energy_wtmad_mae_with_subsets_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    ts = fig.training_subsets_by_size(run)
    out = fig.plot_energy_wtmad_mae(rows, tmp_path / "wt2.png", _STAMP,
                                    training_subsets=ts)
    assert _png_ok(out)


def test_plot_insample_density_ccsd_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    drows = fig.collect_insample_density_rows(run)
    out = fig.plot_insample_density_ccsd(drows, tmp_path / "dens.png", _STAMP)
    assert _png_ok(out)


def test_collect_training_losses(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_training_losses(run)
    # specs 0-4 have model.eqx + losses.npy (spec 5 untrained -> none).
    assert len(rows) == 5
    assert all(r["losses"].shape == (60,) for r in rows)
    assert {r["arch"] for r in rows} == {"deep", "deep_notransform", "deep_attn"}
    assert all(r["subset_size"] in (1, 3) for r in rows)


def test_plot_training_losses_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_training_losses(run)
    out = fig.plot_training_losses(rows, tmp_path / "tl.png", _STAMP)
    assert _png_ok(out)


def test_collect_training_losses_tags_basis(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_training_losses(run, basis_label="def2-svp")
    assert rows and all(r["basis"] == "def2-svp" for r in rows)
    # default: basis is None (backward compatible)
    assert all(r.get("basis") is None for r in fig.collect_training_losses(run))


def test_collect_training_losses_multi_merges_both_runs(tmp_path):
    r1 = _make_run_dir(tmp_path / "a")
    r2 = _make_run_dir(tmp_path / "b")
    merged = fig.collect_training_losses_multi([(r1, "def2-svp"),
                                                (r2, "def2-tzvpd+DF")])
    # every cell from BOTH runs is present, each tagged with its basis
    n1 = len(fig.collect_training_losses(r1))
    n2 = len(fig.collect_training_losses(r2))
    assert len(merged) == n1 + n2
    assert {r["basis"] for r in merged} == {"def2-svp", "def2-tzvpd+DF"}


def test_plot_training_losses_multi_basis_renders(tmp_path):
    r1 = _make_run_dir(tmp_path / "a")
    r2 = _make_run_dir(tmp_path / "b")
    merged = fig.collect_training_losses_multi([(r1, "def2-svp"),
                                                (r2, "def2-tzvpd+DF")])
    out = fig.plot_training_losses(merged, tmp_path / "tl_multi.png", _STAMP)
    assert _png_ok(out)


def test_classify_cell_logic():
    c = fig._classify_cell
    med = 0.0037
    # fails PBE AND final loss is an absolute outlier -> late instability
    assert c(77.0, 14.0, 0.071, med) == "late_instability"
    # fails PBE but final loss is healthy (near cohort median) -> overfitting
    assert c(44.0, 14.0, 0.0023, med) == "generalization_gap"
    # beats PBE -> pass (regardless of loss)
    assert c(9.0, 14.0, 0.0048, med) == "pass"
    assert c(9.0, 14.0, 0.071, med) == "pass"


def test_classify_failures_structure(tmp_path):
    rows = fig.classify_failures([(_make_run_dir(tmp_path), "def2-svp")])
    assert rows
    needed = {"arch", "subset_size", "basis", "heldout_mae", "pbe_mae",
              "final_loss", "classification"}
    for r in rows:
        assert needed <= set(r)
        assert r["classification"] in {"pass", "late_instability",
                                       "generalization_gap"}


def test_plot_failure_diagnostic_renders(tmp_path):
    # Two bases -> the right column carries one stacked capacity-ladder sub-panel
    # per basis (def2-svp + def2-tzvpd+DF), not just the primary one.
    r1 = _make_run_dir(tmp_path / "a")
    r2 = _make_run_dir(tmp_path / "b")
    out = fig.plot_failure_diagnostic(
        [(r1, "def2-svp"), (r2, "def2-tzvpd+DF")], tmp_path / "fail.png", _STAMP)
    assert _png_ok(out)


def test_heldout_pbe_ratio_matches_pass_boundary():
    # the Panel-A y value (held-out / own PBE) crosses 1.0 at exactly the same
    # place _classify_cell flips pass<->fail, so colour matches position.
    assert fig._heldout_pbe_ratio({"heldout_mae": 5.0, "pbe_mae": 10.0}) == 0.5
    assert fig._heldout_pbe_ratio({"heldout_mae": 12.0, "pbe_mae": 10.0}) == 1.2
    assert fig._heldout_pbe_ratio({"heldout_mae": 1.0, "pbe_mae": 0}) is None
    assert fig._heldout_pbe_ratio({"heldout_mae": None, "pbe_mae": 10.0}) is None
    # pass (green, below the line) <=> ratio <= 1 ; fail (above) <=> ratio > 1
    assert fig._classify_cell(9.0, 10.0, 1e-3, 1e-3) == "pass"
    assert fig._heldout_pbe_ratio({"heldout_mae": 9.0, "pbe_mae": 10.0}) <= 1.0
    assert fig._classify_cell(12.0, 10.0, 1e-3, 1e-3) == "generalization_gap"
    assert fig._heldout_pbe_ratio({"heldout_mae": 12.0, "pbe_mae": 10.0}) > 1.0


def test_ladder_bases_includes_both(tmp_path):
    r1 = _make_run_dir(tmp_path / "a")
    r2 = _make_run_dir(tmp_path / "b")
    cells = fig.classify_failures([(r1, "def2-svp"), (r2, "def2-tzvpd+DF")])
    # both bases rendered in the right column, svp first (run order preserved)
    assert fig._ladder_bases(cells) == ["def2-svp", "def2-tzvpd+DF"]


def test_failure_caption_drops_generalization_gap(tmp_path):
    r1 = _make_run_dir(tmp_path / "a")
    r2 = _make_run_dir(tmp_path / "b")
    cells = fig.classify_failures([(r1, "def2-svp"), (r2, "def2-tzvpd+DF")])
    cap = fig._failure_caption(cells, fig._ladder_bases(cells))
    assert "Late training instability" in cap
    assert "Beats PBE" in cap
    assert "Generalization gap" not in cap          # list removed from the caption


def test_build_per_run_diagnostics_writes_two(tmp_path):
    run = _make_run_dir(tmp_path)
    written = fig.build_per_run_diagnostics(run, tmp_path / "out", "def2-svp")
    assert {p.name for p in written} == {"diagnostic_size_consistency.png",
                                         "diagnostic_training_losses.png"}
    assert all(_png_ok(p) for p in written)


def test_heatmap_panel_diverging_renders(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    # center=1.0 -> diverging RdBu_r around PBE parity; values span <1 and >1
    fig._heatmap_panel(ax, {("deep", 1): 0.8, ("deep", 3): 2.0,
                            ("deep_combined", 1): 2.1},
                       ["deep", "deep_combined"], title="ratio",
                       cbar_label="MAE/PBE", center=1.0)
    out = tmp_path / "hp.png"
    f.savefig(out, dpi=80)
    plt.close(f)
    assert out.stat().st_size > 2000


def test_plot_capacity_trends_renders(tmp_path):
    r1 = _make_run_dir(tmp_path / "a")
    r2 = _make_run_dir(tmp_path / "b")
    out = fig.plot_capacity_trends([(r1, "def2-svp"), (r2, "def2-tzvpd+DF")],
                                   tmp_path / "trends.png", _STAMP)
    assert _png_ok(out)


def test_build_diagnostic_figures_renders_all(tmp_path):
    r1 = _make_run_dir(tmp_path / "a")
    r2 = _make_run_dir(tmp_path / "b")
    (r1 / "resolved_config.yaml").write_text("basis: def2-svp\n")
    (r2 / "resolved_config.yaml").write_text("basis: def2-tzvpd\ndensity_fit: true\n")
    out = fig.build_diagnostic_figures([r1, r2], tmp_path / "diag")
    assert {p.name for p in out} == {"diagnostic_training_losses.png",
                                     "diagnostic_failure_mechanisms.png",
                                     "diagnostic_capacity_trends.png"}
    assert all(_png_ok(p) for p in out)


def test_break_limits_detects_outlier():
    lims = fig._break_limits([10, 12, 9, 11, 8, 13, 10, 77])  # 77 dominates
    assert lims is not None
    (b_lo, b_hi), (u_lo, u_hi) = lims
    assert b_lo == 0.0 and b_hi < 30 and u_lo > 50 and u_hi > 77


def test_break_limits_none_without_outlier():
    assert fig._break_limits([10, 12, 9, 11, 8, 13, 10, 14]) is None
    assert fig._break_limits([1.0, 2.0]) is None  # too few


def test_broken_bar_panel_renders_with_break(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = plt.figure(figsize=(6, 4))
    gs = f.add_gridspec(1, 1)
    fig._broken_bar_panel(
        f, gs[0, 0], [("a", [10, 20, 77, 9]), ("b", [8, 15, float("nan"), 11])],
        ["c1", "c2", "c3", "c4"], [("a", 14.0)], "MAE", "kcal/mol",
        ["#4477aa", "#cc6677"], 0.4)
    out = tmp_path / "b.png"
    f.savefig(out, dpi=80)
    plt.close(f)
    assert out.stat().st_size > 2000


def test_methods_textblock_renders_mathtext(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = plt.figure(figsize=(10, 5))
    # malformed mathtext raises at draw -> a successful savefig proves validity.
    fig._methods_textblock(f, {1: ["hocn"], 6: ["alcl", "b2h6", "cf4"]}, y_top=0.4)
    out = tmp_path / "m.png"
    f.savefig(out, dpi=80)
    plt.close(f)
    assert out.stat().st_size > 2000


def test_chem_latex_formats_formulas():
    f = fig._chem_latex
    # plain formulas -> subscripted counts, proper element capitalization
    assert f("ch3") == "CH$_3$"
    assert f("h2s") == "H$_2$S"
    assert f("b2h6") == "B$_2$H$_6$"
    assert f("cf4") == "CF$_4$"
    assert f("clf") == "ClF"
    assert f("alcl") == "AlCl"
    assert f("alf") == "AlF"
    # no false 2-letter element match (Ho/Co/Os) inside H-O-C-N-S names
    assert f("hocn") == "HOCN"
    assert f("ocs") == "OCS"
    assert f("hnco") == "HNCO"
    assert f("NH3") == "NH$_3$"
    # transition-state ('ts') and complex ('comp') suffixes
    assert f("clch3clts") == r"ClCH$_3$Cl$^{\ddagger}$"
    assert f("clch3clcomp") == r"ClCH$_3$Cl$_{\mathrm{(c)}}$"
    assert f("ch3fclts") == r"CH$_3$FCl$^{\ddagger}$"
    # reaction-label species pass through unchanged
    assert f("RKT21") == "RKT21"


def test_chem_latex_renders_in_methods(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = plt.figure(figsize=(13, 7))
    # TS / complex / reaction species must render as valid mathtext
    fig._methods_textblock(
        f, {2: ["ch3fclts", "clf"], 6: ["clch3clcomp", "RKT21", "b2h6"]}, y_top=0.4)
    out = tmp_path / "cm.png"
    f.savefig(out, dpi=80)
    plt.close(f)
    assert out.stat().st_size > 2000


def test_arch_input_forms_match_config():
    forms = fig._arch_input_forms(fig.ARCH_ORDER)
    # base archs (polarized run): F_x(x_2), F_c(r_s, x_2, x_1)
    assert forms["deep"]["fx"] == ["x_2"]
    assert forms["deep"]["fc"] == ["r_s", "x_2", "x_1"]
    # cusp adds x_4,x_5 to BOTH nets; dm adds x_6,x_7 (the occupation-spread
    # entropy was removed 2026-08-06, so the DM block is 2 wide)
    assert forms["deep_cusp"]["fx"] == ["x_2", "x_4", "x_5"]
    assert forms["deep_cusp"]["fc"] == ["r_s", "x_2", "x_1", "x_4", "x_5"]
    assert forms["deep_dm"]["fx"] == ["x_2", "x_6", "x_7"]
    # combined packs the DM block (x_6,x_7) BEFORE cusp (x_4,x_5) -- the
    # networks.py concat order (descriptors=[dm_statistics, cusp])
    assert forms["deep_combined"]["fx"] == ["x_2", "x_6", "x_7", "x_4", "x_5"]
    assert forms["deep_combined"]["fc"] == [
        "r_s", "x_2", "x_1", "x_6", "x_7", "x_4", "x_5"]
    # _attn shares its base's inputs; notransform shares deep's inputs but raw
    assert forms["deep_combined_attn"]["fx"] == forms["deep_combined"]["fx"]
    assert forms["deep_attn"]["attention"] is True
    assert forms["deep_notransform"]["fx"] == forms["deep"]["fx"]
    assert forms["deep_notransform"]["log_transform"] is False


def test_descriptor_x_labels_match_registry_widths():
    """The figure's x-label map must track each descriptor's n_features.

    This map was once a hardcoded constant that silently kept a 3-wide DM
    block after the descriptor shrank to 2 -- the methods column then printed
    a definition for a feature that no longer existed. Deriving the check from
    the registry makes that drift impossible to reintroduce quietly.
    """
    from xcquinox.alec.descriptors import make_descriptor
    for name, labels in fig._DESCRIPTOR_X_LABELS.items():
        n = make_descriptor(name).n_features
        assert len(labels) == n, (
            f"{name}: figure declares {len(labels)} x-labels but the "
            f"descriptor has n_features={n}")
    # Every mapped label must also be DEFINED somewhere in the methods
    # columns, else an architecture joining ARCH_ORDER would print symbols
    # the figure never explains (x_11..x_16 had map entries but no
    # definition until 2026-08-09). Ranges like "x_{11}..x_{16}" count as
    # defining every label they span.
    joined = " ".join(sum(fig._methods_columns(subsets=(2, 6)), []))
    import re
    for lo, hi in re.findall(r"x_\{(\d+)\}\.\.x_\{(\d+)\}", joined):
        joined += " " + " ".join(
            f"x_{{{i}}}" for i in range(int(lo), int(hi) + 1))
    for labels in fig._DESCRIPTOR_X_LABELS.values():
        for lbl in labels:
            idx = lbl.split("_")[1]
            token = f"x_{idx}" if len(idx) == 1 else f"x_{{{idx}}}"
            assert token in joined, (
                f"label {lbl} is mapped but never defined in the methods "
                f"columns")


# ---------------------------------------------------------------------------
# V_xc-consistency provenance: runs that predate the 2026-08-06 correction
# hatch the architectures whose descriptors are DM-dependent (their training
# potential was not the exact functional derivative); post-fix runs draw no
# marks. Classification and rendering are both pinned here.
# ---------------------------------------------------------------------------

def test_vxc_predicate_keys_on_run_date():
    assert fig._run_predates_vxc_fix("run_20260728T140018Z") is True
    assert fig._run_predates_vxc_fix("run_20260806T000000Z") is False
    assert fig._run_predates_vxc_fix("run_20260810T120000Z") is False
    # ids without the stamp are conservatively unmarked
    assert fig._run_predates_vxc_fix("synthetic") is False
    assert fig._run_predates_vxc_fix("") is False


def test_vxc_classification_matches_descriptor_dependence():
    # DM-dependent families carry a class; grid-local families carry none.
    for a in ("deep_mgga_3x16", "deep_mgga_attn_3x16"):
        assert fig._vxc_hatch(a) == fig._VXC_HATCH_GATED, a
    for a in ("deep_rung35_3x16", "deep_rung35_attn_3x16",
              "deep_rung35_mgga_3x16"):
        assert fig._vxc_hatch(a) == fig._VXC_HATCH_READY, a
    for a in ("deep_3x16", "deep_attn_3x16", "deep_cusp_3x16"):
        assert fig._vxc_hatch(a) is None, a
    # The two hatches must be distinct from each other AND from the
    # cell-level channels already in use (mixed "//", missing "//////").
    assert fig._VXC_HATCH_GATED != fig._VXC_HATCH_READY
    assert fig._VXC_HATCH_GATED not in ("//", "//////")
    assert fig._VXC_HATCH_READY not in ("//", "//////")


def _tiny_metric():
    archs = ["deep_3x16", "deep_mgga_3x16", "deep_rung35_3x16"]
    subsets = [2, 6]
    metric = {(a, s): 1.0 + i for i, (a, s) in
              enumerate((a, s) for a in archs for s in subsets)}
    return archs, subsets, metric


def test_vxc_hatch_lands_on_bars_and_legend():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    archs, subsets, metric = _tiny_metric()
    f, ax = plt.subplots()
    fig._grouped_arch_bars(ax, metric, archs, subsets, pbe_line=None,
                           title="t", vxc_pre_fix=True)
    by_arch = {}
    for p in ax.patches:
        if p.get_width() > 0:               # real bars, not legend proxies
            by_arch.setdefault(round(p.get_width(), 6), []).append(p)
    hatches = {p.get_hatch() for p in ax.patches if p.get_width() > 0}
    assert fig._VXC_HATCH_GATED in hatches, "meta-GGA bars not hatched"
    assert fig._VXC_HATCH_READY in hatches, "rung-3.5 bars not hatched"
    assert None in hatches, "GGA bars must stay unhatched"
    _h, labels = ax.get_legend_handles_labels()
    assert any("gated on SCF stabilization" in l for l in labels)
    assert any("safe to re-run" in l for l in labels)
    plt.close(f)


def test_vxc_marks_absent_on_post_fix_runs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    archs, subsets, metric = _tiny_metric()
    f, ax = plt.subplots()
    fig._grouped_arch_bars(ax, metric, archs, subsets, pbe_line=None,
                           title="t", vxc_pre_fix=False)
    assert {p.get_hatch() for p in ax.patches} == {None}
    _h, labels = ax.get_legend_handles_labels()
    assert not any("pre-correction" in l for l in labels)
    plt.close(f)


def test_vxc_disclosure_stamped_by_footer_only_pre_fix():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for run_id, expect in (("run_20260728T140018Z", True),
                           ("run_20260810T000000Z", False)):
        f = plt.figure()
        fig._stamp_parity_footer(f, run_id=run_id, title="t", note="",
                                 provenance=None, caveat=None)
        texts = " ".join(t.get_text() for t in f.texts)
        assert ("V_xc PROVENANCE" in texts) is expect, run_id
        plt.close(f)


def test_arch_forms_lines_cover_each_arch():
    lines = fig._arch_forms_lines()
    joined = " ".join(lines)
    for a in fig.ARCH_ORDER:                 # every figure arch is named
        assert a in joined
    # explicit F_x / F_c forms appear verbatim
    assert "F_x(x_2, x_4, x_5)" in joined                                # cusp
    assert "F_c(r_s, x_2, x_1, x_6, x_7, x_4, x_5)" in joined             # combined
    assert "raw" in joined.lower()           # notransform note
    # renders as valid mathtext
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = plt.figure(figsize=(22, 4))
    f.text(0.02, 0.5, "\n".join(lines), fontsize=6.2, family="serif")
    import os
    out = "/tmp/_archforms_canary.png"
    f.savefig(out, dpi=80)
    plt.close(f)
    assert os.path.getsize(out) > 1500


def test_methods_textblock_can_omit_references(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    subsets = {1: ["hocn"], 2: ["ch3fclts", "clf"]}
    reactions = {1: {"ae": ["hocn"], "rxn": []},
                 2: {"ae": [], "rxn": [(["ch3", "clf"], ["ch3fclts"])]}}
    f1 = plt.figure(figsize=(13, 9))
    n_with = fig._methods_textblock(f1, subsets, y_top=0.95, fontsize=6.2,
                                    reactions=reactions, fig_h=9.0,
                                    include_references=True)
    plt.close(f1)
    f2 = plt.figure(figsize=(13, 9))
    n_without = fig._methods_textblock(f2, subsets, y_top=0.95, fontsize=6.2,
                                       reactions=reactions, fig_h=9.0,
                                       include_references=False)
    out = tmp_path / "norefs.png"
    f2.savefig(out, dpi=80)
    plt.close(f2)
    # omitting the references drops exactly their lines from the block height ...
    assert n_with - n_without == len(fig._methods_references())
    # ... but the figure still renders with the training-subset footer kept
    assert out.stat().st_size > 2000


def test_methods_textblock_accepts_column_offsets(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = plt.figure(figsize=(13, 7))
    # custom x positions + per-column vertical offsets must render + return the
    # max column line count (used by the caller to size the figure).
    n = fig._methods_textblock(
        f, {1: ["hocn"], 6: ["alcl", "b2h6"]}, y_top=0.45,
        xs=(0.05, 0.34, 0.715), y_deltas=(0.0, -0.06, 0.0))
    out = tmp_path / "mo.png"
    f.savefig(out, dpi=80)
    plt.close(f)
    assert out.stat().st_size > 2000
    assert isinstance(n, int) and n >= 6


def test_methods_columns_lists_spin_and_descriptor_purposes():
    cols = fig._methods_columns({1: ["hocn"]})
    col1, col2, col3 = (" ".join(cols[0]), " ".join(cols[1]), " ".join(cols[2]))
    alltext = col1 + col2 + col3
    refs = " ".join(fig._methods_references())
    # descriptors + spin clip
    assert "x_1" in col1 and "x_2" in col1 and r"\zeta" in col1
    assert "PW92" in col1 and "clip" in col1.lower()
    assert "1.804" in col1 and "1.174" in col1  # bounds with cites
    # loss: FORM ours ("this work"), weights/scheme cite dpyscf/DFS, MIXED metric
    assert r"\sum_k w_k" in col2 and "this work" in col2.lower()
    assert "dpyscf" in col2 and "DFS" in col2
    assert "absolute" in col2 and "relative" in col2          # mixed metric
    assert "DORMANT" in col2 and "per-molecule" in col2
    assert "3-cycle" in col2 and "one-shot" in col2           # rho SCF, vxc one-shot
    assert "W2-F12" in col2 and "CCSD(T)" not in col2         # GMTKN55-BH76 refs are W2-F12
    assert "[17]" in (col1 + col2 + col3)                     # W4-11 ref cited, not orphaned
    # extended descriptors: x4-x7, V_ext defined; the entropy feature (and its
    # INTENSIVE normalization) was removed 2026-08-06
    assert "x_4" in col3 and "x_7" in col3 and "INTENSIVE" not in col3
    assert "V_{ext}" in col3                                  # nuclear field defined
    # opaque shorthand + the corrected errors must be GONE
    assert "size-dependent" not in alltext
    assert "log = DFS" not in alltext and "log=DFS" not in alltext
    assert "Dick" not in alltext  # use the [n] cites / DFS, not "Dick"
    # de-editorialized: no narrative / condescending / value-judgment modifiers
    low = alltext.lower()
    assert "textbook" not in low                  # the called-out condescension
    assert "heuristic" not in low                 # value judgment on our own work
    assert "proxy" not in low                     # "delocalization proxy"
    assert "clean" not in low                     # "clean single-reference flag"
    assert "range-conditioning" not in low
    assert "core region" not in low
    assert "nans" not in low                      # informal jargon verb
    # physics fix: sum_A Z_A/r_A is a POTENTIAL (= -V_ext), not a field
    assert "potential" in col3 and "electrostatic field" not in col3
    assert "non-finite" in col1                   # the de-jargoned clip line
    # kept content survives the cleanup (honest labels + sourced terms)
    assert "this work" in col3.lower()
    assert "multireference" in col3.lower() and "[11]" in col3
    assert "Slater density envelope" in col3      # factual, kept
    # loss routing: W4-11 atomizations train through the reaction-energy channel
    # (kind="bh76"), NOT the relative-AE channel
    assert "reaction energy" in col2 and "W4-11" in col2 and "[17]" in col2
    assert "not populated by this pool" in col2   # AE-relative + IP13 inactive here
    # attention equation now cited [19]; DFS acronym glossed on [4]
    assert "[19]" in col3 and "Vaswani" in refs
    assert "DFS" in refs
    # references key: every contested citation is the CORRECT one
    assert "Steiner" in refs and "Kato" in refs               # -2Z density vs -Z wavefn
    assert "18A533" in refs                                   # Gedanken for 1.174
    assert "Xu" in refs and "721" in refs and "1218" not in refs
    assert "Oliver" in refs and "Loewdin" in refs and "Parr" in refs


def test_subset_reaction_lines_render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reactions = {2: {"ae": ["hocn"], "rxn": [(["ch3", "clf"], ["ch3fclts"])]},
                 6: {"ae": ["alcl", "b2h6", "cf4"],
                     "rxn": [(["clch3clcomp"], ["clch3clts"])]}}
    lines = fig._subset_reaction_lines(reactions)
    joined = "\n".join(lines)
    assert "AE:" in joined and "barriers:" in joined
    assert r"\to" in joined  # reactant -> product arrow
    # renders as valid mathtext
    f = plt.figure(figsize=(13, 4))
    f.text(0.02, 0.5, joined, fontsize=6.2, family="serif")
    out = "/tmp/_subset_rxn_canary.png"
    f.savefig(out, dpi=80)
    plt.close(f)
    import os
    assert os.path.getsize(out) > 1500


def test_reaction_footer_labels_transition_state_glyph():
    # the footer header must explain the rendered double-dagger / (c) glyphs
    reactions = {1: {"ae": ["hocn"],
                     "rxn": [(["clch3clcomp"], ["clch3clts"])]}}
    header = fig._subset_reaction_lines(reactions)[0]
    assert r"\ddagger" in header           # the TS glyph itself is shown
    assert "transition state" in header
    assert "(c)" in header                 # the reactant-complex glyph defined
    assert "subscript" in header           # (c) rendered as a subscript


def test_run_basis_label_reads_basis_and_df(tmp_path):
    (tmp_path / "resolved_config.yaml").write_text(
        "basis: def2-tzvpd\ndensity_fit: true\ngrid_level: 2\n")
    assert fig.run_basis_label(tmp_path) == "def2-tzvpd+DF"
    (tmp_path / "resolved_config.yaml").write_text(
        "basis: def2-svp\ndensity_fit: false\n")
    assert fig.run_basis_label(tmp_path) == "def2-svp"


def test_run_solver_label_reads_block_list(tmp_path):
    # block-style "solver:\n- full_3" (the form the v3 configs use)
    (tmp_path / "resolved_config.yaml").write_text(
        "basis: def2-svp\ndensity_fit: false\nsolver:\n- full_3\ngrid_level: 2\n")
    assert fig.run_solver_label(tmp_path) == "full_3"


def test_run_solver_label_reads_full25(tmp_path):
    (tmp_path / "resolved_config.yaml").write_text(
        "basis: def2-svp\nsolver:\n- full_25\n")
    assert fig.run_solver_label(tmp_path) == "full_25"


def test_run_solver_label_inline_form(tmp_path):
    (tmp_path / "resolved_config.yaml").write_text("solver: full_3\n")
    assert fig.run_solver_label(tmp_path) == "full_3"


def test_run_solver_label_empty_when_absent(tmp_path):
    (tmp_path / "resolved_config.yaml").write_text("basis: def2-svp\n")
    assert fig.run_solver_label(tmp_path) == ""


def test_disambiguated_labels_appends_solver_on_basis_collision(tmp_path):
    # two runs that share a basis (def2-svp) but differ in SCF cycles must get
    # DISTINCT display labels carrying the full_3 / full_25 tag.
    a = tmp_path / "a"; a.mkdir()
    (a / "resolved_config.yaml").write_text("basis: def2-svp\nsolver:\n- full_3\n")
    b = tmp_path / "b"; b.mkdir()
    (b / "resolved_config.yaml").write_text("basis: def2-svp\nsolver:\n- full_25\n")
    labels = fig._disambiguated_run_labels([a, b])
    assert len(set(labels)) == 2                       # distinct
    assert any("full_3" in lbl for lbl in labels)
    assert any("full_25" in lbl for lbl in labels)
    assert all("def2-svp" in lbl for lbl in labels)


def test_disambiguated_labels_unchanged_when_basis_differs(tmp_path):
    # distinct bases need no disambiguation -> bare labels preserved.
    a = tmp_path / "a"; a.mkdir()
    (a / "resolved_config.yaml").write_text("basis: def2-svp\ndensity_fit: false\n")
    b = tmp_path / "b"; b.mkdir()
    (b / "resolved_config.yaml").write_text("basis: def2-tzvpd\ndensity_fit: true\n")
    assert fig._disambiguated_run_labels([a, b]) == ["def2-svp", "def2-tzvpd+DF"]


def test_ckpt_label_maps_eval_subdir():
    assert fig._ckpt_label("eval_holdout") == "final-step"
    assert fig._ckpt_label("eval_holdout_val_best") == "val-best"
    # the legacy training-loss-best dir (no longer plotted) must not mislabel as
    # "final-step" -- it is the train-best checkpoint.
    assert fig._ckpt_label("eval_holdout_best") == "train-best"


def test_plot_basis_comparison_renders(tmp_path):
    ra = _make_run_dir(tmp_path / "a")
    rb = _make_run_dir(tmp_path / "b")
    out = fig.plot_basis_comparison(
        [(ra, "def2-svp"), (rb, "def2-tzvpd+DF")], tmp_path / "cmp.png", "cmp")
    assert _png_ok(out)


def test_plot_basis_comparison_union_keeps_unshared_cells(tmp_path):
    import shutil
    ra = _make_run_dir(tmp_path / "a")
    rb = _make_run_dir(tmp_path / "b")
    # rb loses one (arch, subset) cell; the UNION must still plot it (ra-only),
    # i.e. a completed cell is not dropped just because the other run lacks it.
    shutil.rmtree(rb / "checkpoints" / "spec_0001" / "eval_holdout")
    out = fig.plot_basis_comparison([(ra, "A"), (rb, "B")], tmp_path / "u.png", "u")
    assert _png_ok(out)


def test_build_basis_comparison_writes(tmp_path):
    ra = _make_run_dir(tmp_path / "a")
    (ra / "resolved_config.yaml").write_text("basis: def2-svp\ndensity_fit: false\n")
    rb = _make_run_dir(tmp_path / "b")
    (rb / "resolved_config.yaml").write_text("basis: def2-tzvpd\ndensity_fit: true\n")
    written = fig.build_basis_comparison_figures([ra, rb], tmp_path / "out")
    assert written and all(_png_ok(p) for p in written)
    names = {p.name for p in written}
    assert "basis_comparison.png" in names          # full figure (with refs)
    assert "basis_comparison_no_refs.png" in names  # variant w/o references key
    assert "basis_comparison_clean.png" in names    # bars-only (no bottom notes)


def test_plot_basis_comparison_omits_references(tmp_path):
    ra = _make_run_dir(tmp_path / "a")
    rb = _make_run_dir(tmp_path / "b")
    out = fig.plot_basis_comparison(
        [(ra, "def2-svp"), (rb, "def2-tzvpd+DF")], tmp_path / "nr.png", "cmp",
        include_references=False)
    assert _png_ok(out)


def test_plot_basis_comparison_bars_only_is_shorter(tmp_path):
    # bars-only drops every bottom annotation -> a much shorter figure than the
    # fully-annotated default (verified via the rendered pixel height).
    from PIL import Image
    ra = _make_run_dir(tmp_path / "a")
    rb = _make_run_dir(tmp_path / "b")
    runs = [(ra, "def2-svp"), (rb, "def2-tzvpd+DF")]
    full = fig.plot_basis_comparison(runs, tmp_path / "full.png", "cmp")
    clean = fig.plot_basis_comparison(runs, tmp_path / "clean.png", "cmp",
                                      bars_only=True)
    assert _png_ok(clean)
    assert Image.open(clean).size[1] < Image.open(full).size[1]


def test_comparison_cells_union_and_arch_filter():
    # union across runs; the arch filter keeps only the named archs (all their
    # subset sizes) and preserves the sorted cell order; empty input -> [].
    sets = [{("deep", 1), ("deep", 3), ("deep_attn", 1)},
            {("deep", 26), ("deep_cusp", 1)}]
    assert fig._comparison_cells(sets) == [
        ("deep", 1), ("deep", 3), ("deep", 26),
        ("deep_attn", 1), ("deep_cusp", 1)]
    assert fig._comparison_cells(sets, archs=("deep",)) == [
        ("deep", 1), ("deep", 3), ("deep", 26)]
    assert fig._comparison_cells(sets, archs=("deep", "deep_cusp")) == [
        ("deep", 1), ("deep", 3), ("deep", 26), ("deep_cusp", 1)]
    assert fig._comparison_cells([]) == []


def test_plot_basis_comparison_archs_filter_renders(tmp_path):
    ra = _make_run_dir(tmp_path / "a")
    rb = _make_run_dir(tmp_path / "b")
    out = fig.plot_basis_comparison(
        [(ra, "A"), (rb, "B")], tmp_path / "focus.png", "cmp",
        archs=("deep",))
    assert _png_ok(out)


def test_plot_basis_comparison_rejects_unknown_archs(tmp_path):
    # an arch filter matching zero cells must fail loud (a blank comparison
    # would otherwise render); a partially-matching filter renders with the
    # unknown names reported, not dropped silently into a blank figure.
    import pytest
    ra = _make_run_dir(tmp_path / "a")
    rb = _make_run_dir(tmp_path / "b")
    runs = [(ra, "A"), (rb, "B")]
    with pytest.raises(ValueError, match="match no"):
        fig.plot_basis_comparison(runs, tmp_path / "bogus.png", "cmp",
                                  archs=("no_such_arch",))
    out = fig.plot_basis_comparison(runs, tmp_path / "partial.png", "cmp",
                                    archs=("deep", "no_such_arch"))
    assert _png_ok(out)


def test_build_basis_comparison_rejects_empty_archs(tmp_path):
    # archs=() is a caller error: it is falsy (so the _focus suffix logic
    # would pick the FULL-UNION filenames) yet filters to zero cells -- the
    # blank output would overwrite the real comparison trio.
    import pytest
    ra = _make_run_dir(tmp_path / "a")
    (ra / "resolved_config.yaml").write_text("basis: def2-svp\ndensity_fit: false\n")
    rb = _make_run_dir(tmp_path / "b")
    (rb / "resolved_config.yaml").write_text("basis: def2-tzvpd\ndensity_fit: true\n")
    with pytest.raises(ValueError, match="non-empty"):
        fig.build_basis_comparison_figures([ra, rb], tmp_path / "out",
                                           archs=())


def test_build_basis_comparison_focus_names(tmp_path):
    # the focused render must not overwrite the full-union trio: it writes the
    # basis_comparison_focus* stems instead.
    ra = _make_run_dir(tmp_path / "a")
    (ra / "resolved_config.yaml").write_text("basis: def2-svp\ndensity_fit: false\n")
    rb = _make_run_dir(tmp_path / "b")
    (rb / "resolved_config.yaml").write_text("basis: def2-tzvpd\ndensity_fit: true\n")
    written = fig.build_basis_comparison_figures([ra, rb], tmp_path / "out",
                                                 archs=("deep",))
    names = {p.name for p in written}
    assert names == {"basis_comparison_focus.png",
                     "basis_comparison_focus_no_refs.png",
                     "basis_comparison_focus_clean.png"}
    assert all(_png_ok(p) for p in written)


def _make_bh76w411_results(tmp_path):
    """A results root with the real layout:
    <root>/bh76w411_repr/<basis>/runs/<stamp>, two bases, each with a newest run
    (full _make_run_dir content) + an older empty run_* (to test newest-pick)."""
    import shutil
    root = tmp_path / "results"
    runs = {}
    for basis, stamp in (("svp_grid2", "run_20260603T163407Z"),
                         ("tzvpd_grid2_df", "run_20260604T230749Z")):
        src = _make_run_dir(tmp_path / f"_src_{basis}")
        runs_dir = root / "bh76w411_repr" / basis / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, runs_dir / stamp)
        (runs_dir / "run_20260101T000000Z").mkdir()      # older, empty
        # a basis label so the cross-basis figure gets a real label
        (runs_dir / stamp / "resolved_config.yaml").write_text(
            f"basis: def2-{'svp' if basis=='svp_grid2' else 'tzvpd'}\n"
            f"density_fit: {'false' if basis=='svp_grid2' else 'true'}\n")
        runs[basis] = runs_dir / stamp
    return root, runs


def test_arch_coverage_evaled_without_weights_not_untrained(tmp_path):
    # eval-only pulls (no model.eqx synced): an arch WITH held-out eval must NOT
    # be reported 'untrained' (it was obviously trained) -> no false "NOT TRAINED"
    # footer on the figures.
    run = tmp_path / "r"
    run.mkdir()
    specs = [{"arch": "deep", "subset_size": 1},
             {"arch": "deep_cusp", "subset_size": 1}]
    (run / "manifest.json").write_text(json.dumps(
        {"n_specs": 2, "width": 4,
         "specs": [{"index": i, "spec_file": f"spec_{i:04d}.spec",
                    "sha256": "x" * 64, "cell": c} for i, c in enumerate(specs)]}))
    (run / "specs").mkdir()
    for i in range(2):
        eh = run / "checkpoints" / f"spec_{i:04d}" / "eval_holdout"
        eh.mkdir(parents=True)
        (eh / "per_reaction.json").write_text(json.dumps([
            {"name": "x", "pool": "bh76", "reaction_energy_ref_kcalmol": 1.0,
             "de_nn_kcalmol": 1.0, "de_pbe_kcalmol": 1.0,
             "abs_error_nn_kcalmol": 0.1, "abs_error_pbe_kcalmol": 0.1}]))
        # deliberately NO model.eqx (weights not pulled)
    cov = fig.arch_coverage(run)
    assert set(cov["holdout"]) == {"deep", "deep_cusp"}
    assert cov["untrained"] == []     # eval'd -> trained, despite missing weights
    # coverage count must not collapse to model.eqx count (0 here): both eval'd
    assert fig.trained_spec_count(run) == 2


def test_coverage_note_distinguishes_in_progress(tmp_path):
    # A spec mid-training (resume checkpoint on disk, no final weights) is IN
    # PROGRESS, not "NOT TRAINED" -- the roster note must say which, or a
    # running array reads as absent.
    run = _make_run_dir(tmp_path)
    (run / "checkpoints" / "spec_0004" / "model.eqx").unlink()
    (run / "checkpoints" / "spec_0005" / "resume_state.pkl").write_bytes(b"x")
    cov = fig.arch_coverage(run)
    assert "deep_attn" in cov["untrained"]     # no final weights anywhere
    assert "deep_attn" in cov["in_progress"]
    note = fig.coverage_note(run)
    assert "IN PROGRESS" in note and "deep_attn" in note
    assert "NOT TRAINED" not in note   # nothing purely not-started remains
    # A completion sentinel beats leftover resume files: completed is not
    # in-progress (matches the harness resume predicates).
    (run / "checkpoints" / "spec_0005" / "completion.json").write_text("{}")
    assert "deep_attn" not in fig.arch_coverage(run)["in_progress"]


def test_newest_run_per_basis_picks_latest(tmp_path):
    root, runs = _make_bh76w411_results(tmp_path)
    got = fig._newest_run_per_basis(root, ("svp_grid2", "tzvpd_grid2_df"))
    assert got["svp_grid2"].name == "run_20260603T163407Z"
    assert got["tzvpd_grid2_df"].name == "run_20260604T230749Z"


def test_figure_cell_coverage_reports_renderable_cells(tmp_path):
    root, runs = _make_bh76w411_results(tmp_path)
    cov = fig.figure_cell_coverage(runs["svp_grid2"])
    # deep×{1,3} + deep_notransform×{1,3} are eval'd (deep_attn trained-no-eval
    # / untrained -> not rendered)
    assert cov["n_cells"] == 4
    assert set(cov["archs"]) == {"deep", "deep_notransform"}
    assert cov["subsets"] == [1, 3]
    assert cov["archs_not_in_order"] == []     # all renderable -> no silent drop
    # ARCH_ORDER archs with no eval cell yet (judged by eval, not model.eqx):
    # deep_attn is trained-but-uneval'd in the fixture -> reported missing
    assert "deep_attn" in cov["archs_missing"]
    assert "deep" not in cov["archs_missing"] and \
           "deep_notransform" not in cov["archs_missing"]


def _add_val_best_eval(run_dir):
    """Duplicate each spec's eval_holdout/ -> eval_holdout_val_best/ so the suite's
    val-best figure set has data to render (mirrors the cluster's eval pass on
    model_val_best.eqx, the held-out-validation-best weights)."""
    import shutil
    for sd in (run_dir / "checkpoints").glob("spec_*"):
        eh = sd / "eval_holdout"
        if eh.is_dir():
            shutil.copytree(eh, sd / "eval_holdout_val_best", dirs_exist_ok=True)


def test_build_bh76w411_suite_writes_all_families(tmp_path):
    root, runs = _make_bh76w411_results(tmp_path)
    outroot = tmp_path / "figs"
    written = fig.build_bh76w411_suite(results_root=root, outroot=outroot)
    assert written and all(_png_ok(p) for p in written)
    parents = {p.parent.name for p in written}
    assert "figures_svp" in parents          # per-basis (svp_grid2 -> svp)
    assert "figures_tzvpd_df" in parents     # per-basis (tzvpd_grid2_df -> tzvpd_df)
    assert "figures_basis_comparison" in parents
    names = {p.name for p in written}
    assert "basis_comparison.png" in names and "basis_comparison_no_refs.png" in names
    assert "ablation_arch_subset_heatmap.png" in names   # per-basis ablation set
    # newly-wired per-basis families (previously generated by hand -> went stale)
    assert "ablation_parity_arch_cols.png" in names      # parity-layout variants
    assert "diagnostic_size_consistency.png" in names    # per-run diagnostics
    assert "diagnostic_training_losses.png" in names
    # no eval_holdout_val_best/ in this fixture -> NO val-best figure set
    assert not any(p.parent.name.endswith("_val_best") for p in written)


def test_collect_holdout_reads_named_eval_subdir(tmp_path):
    run = _make_run_dir(tmp_path)
    _add_val_best_eval(run)
    final = fig.collect_holdout_reaction_rows(run)
    vbest = fig.collect_holdout_reaction_rows(run, eval_subdir="eval_holdout_val_best")
    assert vbest and len(vbest) == len(final)      # val-best dir mirrors final here
    # absent subdir -> empty (no crash), so runs without val-best skip that set
    bare = _make_run_dir(tmp_path / "bare")
    assert fig.collect_holdout_reaction_rows(
        bare, eval_subdir="eval_holdout_val_best") == []


def test_build_bh76w411_suite_emits_val_best_set_when_present(tmp_path):
    # eval_holdout_val_best/ present -> a SECOND, parallel figure set into
    # figures_<alias>_val_best/ + figures_basis_comparison_val_best/ (doubled).
    root, runs = _make_bh76w411_results(tmp_path)
    for r in runs.values():
        _add_val_best_eval(r)
    outroot = tmp_path / "figs"
    written = fig.build_bh76w411_suite(results_root=root, outroot=outroot)
    assert written and all(_png_ok(p) for p in written)
    parents = {p.parent.name for p in written}
    # both the final set AND the val-best set are present
    assert {"figures_svp", "figures_svp_val_best",
            "figures_tzvpd_df", "figures_tzvpd_df_val_best",
            "figures_basis_comparison",
            "figures_basis_comparison_val_best"} <= parents


def test_build_bh76w411_suite_rejects_unknown_arch(tmp_path, monkeypatch):
    # an arch present in the data but absent from ARCH_ORDER must FAIL LOUD
    # (it would otherwise be silently dropped from the per-arch plots)
    root, runs = _make_bh76w411_results(tmp_path)
    monkeypatch.setattr(fig, "ARCH_ORDER", ("deep",))   # drop deep_notransform
    import pytest
    with pytest.raises(ValueError, match="not in ARCH_ORDER"):
        fig.build_bh76w411_suite(results_root=root, outroot=tmp_path / "f2")


def _make_dfs_results(tmp_path):
    """A results root with the dfs_step7 layout: ONE basis (svp_grid2), subset
    sizes up to the full 26-pt pool (ss=26), and one ARCH_ORDER arch
    (deep_cusp) with zero eval'd cells (run still in progress)."""
    import numpy as _np
    root = tmp_path / "results"
    stamp = "run_20260607T162842Z"
    run_dir = root / "dfs_step7" / "svp_grid2" / "runs" / stamp
    run_dir.mkdir(parents=True)
    specs = [
        {"arch": "deep", "subset_size": 1},
        {"arch": "deep", "subset_size": 26},
        {"arch": "deep_attn", "subset_size": 1},
        {"arch": "deep_attn", "subset_size": 26},
        {"arch": "deep_cusp", "subset_size": 1},   # not eval'd yet
    ]
    (run_dir / "manifest.json").write_text(json.dumps(
        {"n_specs": len(specs), "width": 4,
         "specs": [{"index": i, "spec_file": f"spec_{i:04d}.spec",
                    "sha256": "x" * 64, "cell": c}
                   for i, c in enumerate(specs)]}))
    (run_dir / "specs").mkdir()
    (run_dir / "resolved_config.yaml").write_text(
        "basis: def2-svp\ndensity_fit: false\n")
    for i, cell in enumerate(specs):
        sd = run_dir / "checkpoints" / f"spec_{i:04d}"
        sd.mkdir(parents=True)
        (sd / "train_metadata.json").write_text(json.dumps(
            {"molecules": ["HO", "CH4", "h", "c", "o"]}))
        if cell["arch"] == "deep_cusp":
            continue       # in-progress arch: no losses/eval yet
        (sd / "model.eqx").write_bytes(b"x" * 16)
        _np.save(sd / "losses.npy", _np.linspace(0.1, 1e-3, 60))
        ev = sd / "eval"; ev.mkdir()
        (ev / "per_molecule.json").write_text(json.dumps([
            {"molecule": "HO", "AE_error_kcalmol": 6.0 + i, "density_rmse": 3e-3,
             "skipped": False, "scf_converged": True},
            {"molecule": "CH4", "AE_error_kcalmol": -2.0 - i, "density_rmse": 1e-3,
             "skipped": False, "scf_converged": True},
        ]))
        eh = sd / "eval_holdout"; eh.mkdir()
        (eh / "per_reaction.json").write_text(json.dumps([
            {"name": "bh76_a", "pool": "bh76",
             "reactants": ["HO", "h"], "products": ["HOh_ts"],
             "reaction_energy_ref_kcalmol": 17.7,
             "de_nn_kcalmol": -91.0 + i, "de_pbe_kcalmol": -91.2 + i,
             "abs_error_nn_kcalmol": 108.7 - i, "abs_error_pbe_kcalmol": 108.9 - i},
            {"name": "w411_b", "pool": "w411",
             "reactants": ["HO"], "products": ["h", "o"],
             "reaction_energy_ref_kcalmol": 120.0,
             "de_nn_kcalmol": 118.0 + i, "de_pbe_kcalmol": 119.0 + i,
             "abs_error_nn_kcalmol": 2.0 + i, "abs_error_pbe_kcalmol": 1.0 + i},
        ]))
    return root, run_dir


def test_newest_run_per_basis_respects_domain(tmp_path):
    root, run_dir = _make_dfs_results(tmp_path)
    got = fig._newest_run_per_basis(root, ("svp_grid2",), domain="dfs_step7")
    assert got["svp_grid2"] == run_dir
    # the default domain has no runs in this root -> still fails loud
    with pytest.raises(FileNotFoundError):
        fig._newest_run_per_basis(root, ("svp_grid2",))


def test_build_suite_single_basis_dfs_domain(tmp_path):
    # one basis, subset sizes up to 26, one arch with zero eval'd cells: the
    # per-basis family renders into a DOMAIN-PREFIXED dir (no collision with
    # the bh76w411 figures_svp/) and the one-run basis comparison is skipped.
    root, _ = _make_dfs_results(tmp_path)
    outroot = tmp_path / "figs"
    written = fig.build_bh76w411_suite(results_root=root, outroot=outroot,
                                       bases=("svp_grid2",),
                                       domain="dfs_step7")
    assert written and all(_png_ok(p) for p in written)
    parents = {p.parent.name for p in written}
    assert parents == {"figures_dfs_step7_svp"}
    names = {p.name for p in written}
    assert "ablation_arch_subset_heatmap.png" in names   # ss=26 column renders
    assert "diagnostic_training_losses.png" in names
    assert "basis_comparison.png" not in names           # needs >= 2 bases


def test_heatmap_subset_axis_is_data_driven():
    # sizes outside the historical SUBSET_SIZES grid (e.g. the full 26-pt
    # dfs_step7 pool) must appear as heatmap columns, not be silently dropped
    rxn = [{"subset_size": 1}, {"subset_size": 26}]
    ae = [{"subset_size": 2}]
    assert fig._heatmap_subset_axis(rxn, ae) == [1, 2, 26]
    assert fig._heatmap_subset_axis([], []) == list(fig.SUBSET_SIZES)


def test_suite_cli_passes_domain_bases_outroot(tmp_path):
    root, _ = _make_dfs_results(tmp_path)
    outroot = tmp_path / "cli_figs"
    rc = fig.main(["--suite", "--domain", "dfs_step7", "--bases", "svp_grid2",
                   "--results-root", str(root), "--outroot", str(outroot)])
    assert rc == 0
    assert list(outroot.glob("figures_dfs_step7_svp/*.png"))
    # nothing written next to the script itself by this invocation
    assert not (outroot / "figures_svp").exists()


def _add_holdout_density(run_dir, *, with_nn=True):
    """Append density columns to each spec's eval_holdout/per_molecule.json
    (held-out per-species schema; the suite fixture only writes per_reaction)
    and a run-level pbe_density_errors.json."""
    for sd in (run_dir / "checkpoints").glob("spec_*"):
        eh = sd / "eval_holdout"
        if not (eh / "per_reaction.json").is_file():
            continue
        rows = [
            {"molecule": "HO", "density_rmse": 2e-4 if with_nn else None,
             "density_l1": 1e-5 if with_nn else None,
             "density_rmse_pbe": 8e-4, "density_l1_pbe": 5e-5,
             "density_eps_l1": 2.5e-4 if with_nn else None,
             "density_eps_l1_pbe": 7e-4,
             "n_electrons": 9.0, "grid_weight_sum": 100.0,
             "ref_density_method": "ccsd", "from_training_subset": False},
            {"molecule": "H", "density_rmse": None, "density_l1": None,
             "density_rmse_pbe": None, "density_l1_pbe": None,
             "density_eps_l1": None, "density_eps_l1_pbe": None,
             "n_electrons": None, "grid_weight_sum": None,
             "ref_density_method": None, "from_training_subset": False},
        ]
        (eh / "per_molecule.json").write_text(json.dumps(rows))
    (run_dir / "pbe_density_errors.json").write_text(json.dumps({
        "basis": "def2-svp", "grid_level": 2, "refs_dir": "/refs",
        "errors": {"HO": {"density_rmse_pbe": 8e-4, "density_l1_pbe": 5e-5},
                   "CH4": {"density_rmse_pbe": 3e-4, "density_l1_pbe": 2e-5}},
        "failures": {},
    }))


def test_collect_holdout_density_rows_keeps_either_channel(tmp_path):
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run, with_nn=False)   # PBE-only re-eval shape
    rows = fig.collect_holdout_density_rows(run)
    assert rows, "PBE-only rows must be kept (NN channel may lag the refs)"
    assert all(r["molecule"] == "HO" for r in rows)   # all-None H row dropped
    assert all(r["density_rmse"] is None for r in rows)
    assert all(r["density_rmse_pbe"] == pytest.approx(8e-4) for r in rows)
    assert {r["arch"] for r in rows} == {"deep", "deep_notransform"}


def test_load_pbe_density_table(tmp_path):
    run = _make_run_dir(tmp_path)
    assert fig.load_pbe_density_table(run) == {}        # absent -> empty
    _add_holdout_density(run)
    tab = fig.load_pbe_density_table(run)
    assert set(tab) == {"HO", "CH4"}
    assert tab["CH4"]["density_rmse_pbe"] == pytest.approx(3e-4)


def test_plot_holdout_density_ccsd_renders_parity_and_pbe_only(tmp_path):
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    rows = fig.collect_holdout_density_rows(run)
    tab = fig.load_pbe_density_table(run)
    p1 = fig.plot_holdout_density_ccsd(rows, tmp_path / "hd.png", "run_x",
                                       pbe_table=tab)
    assert _png_ok(p1)
    # PBE-only mode (no NN density anywhere) still renders the baseline strip
    pbe_rows = [dict(r, density_rmse=None) for r in rows]
    p2 = fig.plot_holdout_density_ccsd(pbe_rows, tmp_path / "hd2.png", "run_x",
                                       pbe_table=tab)
    assert _png_ok(p2)


def test_build_density_energy_figures_emits_holdout_density_when_present(tmp_path):
    run = _make_run_dir(tmp_path)
    out1 = tmp_path / "f1"
    names1 = {p.name for p in fig.build_density_energy_figures(run, out1)}
    # refs-free run: only the four unconditional figures
    assert names1 == {"ablation_rung_summary.png",
                      "ablation_energy_wtmad_mae.png",
                      "ablation_insample_density_ccsd.png",
                      "ablation_insample_overview.png"}
    assert "ablation_holdout_density_ccsd.png" not in names1
    # the combined-ED family is gated on the same holdout density columns,
    # and so is the held-out overview composite
    assert "ablation_combined_energy_density.png" not in names1
    assert "ablation_density_energy_overview.png" not in names1
    assert not (out1 / "ablation_combined_energy_density.csv").exists()
    _add_holdout_density(run)
    out2 = tmp_path / "f2"
    names2 = {p.name for p in fig.build_density_energy_figures(run, out2)}
    assert "ablation_holdout_density_ccsd.png" in names2
    assert "ablation_holdout_density_per_arch.png" in names2
    assert "ablation_combined_energy_density.png" in names2
    assert "ablation_density_energy_overview.png" in names2
    assert "ablation_density_energy_3x3.png" in names2
    assert "ablation_ed_decomposition.png" in names2
    # DFS-units twins ride along whenever the eps columns are present
    # (the fixture writes them)
    assert "ablation_combined_energy_density_dfs_units.png" in names2
    assert "ablation_ed_decomposition_dfs_units.png" in names2
    assert "ablation_density_energy_overview_dfs_units.png" in names2
    assert "ablation_density_energy_3x3_dfs_units.png" in names2
    # the 3x3s' former parity rows as standalone per-channel figures
    assert "ablation_density_parity_by_channel.png" in names2
    assert "ablation_density_parity_by_channel_dfs_units.png" in names2
    assert len(names2) == 16
    assert (out2 / "ablation_density_energy_3x3_dfs_units.csv").is_file()
    # the CSVs are written alongside but NEVER returned (return stays PNG-only)
    assert (out2 / "ablation_combined_energy_density.csv").is_file()
    assert (out2 / "ablation_density_energy_3x3.csv").is_file()


def test_insample_density_plot_with_pbe_baseline_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_insample_density_rows(run)
    assert rows
    # older runs: no density_rmse_pbe column -> collected as None, still renders
    assert all(r["density_rmse_pbe"] is None for r in rows)
    for r in rows:
        r["density_rmse_pbe"] = 9e-4
    p = fig.plot_insample_density_ccsd(rows, tmp_path / "ins.png", "run_x")
    assert _png_ok(p)


def test_w411_natoms_map_counts_atoms():
    nm = fig._w411_natoms_map()
    assert nm.get("w411_propane_atomization") == 11  # C3H8 = 11 atoms
    assert nm and all(v >= 2 for v in nm.values())


def test_plot_size_consistency_diagnostic_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    out = fig.plot_size_consistency_diagnostic(
        rows, tmp_path / "sc.png", _STAMP, cells=[("deep", 1), ("deep", 3)])
    assert _png_ok(out)


def test_build_density_energy_figures_writes_four(tmp_path):
    run = _make_run_dir(tmp_path)
    written = fig.build_density_energy_figures(run, tmp_path / "out")
    # headline rung summary + energy + in-sample density + in-sample overview
    # (no SCAN cache -> no SCAN line; refs-free run -> no holdout/ED family)
    assert len(written) == 4
    assert all(_png_ok(p) for p in written)
    assert {p.name for p in written} == {"ablation_rung_summary.png",
                                         "ablation_energy_wtmad_mae.png",
                                         "ablation_insample_density_ccsd.png",
                                         "ablation_insample_overview.png"}


# ---------------------------------------------------------------------------
# DFS Eq. 21 combined energy-density metric (ED)
# ---------------------------------------------------------------------------

def test_holdout_density_by_arch_subset_means_and_drops_nonfinite():
    rows = [
        {"arch": "deep", "subset_size": 1, "molecule": "HO",
         "density_rmse": 2e-4},
        {"arch": "deep", "subset_size": 1, "molecule": "CH4",
         "density_rmse": 4e-4},
        {"arch": "deep", "subset_size": 3, "molecule": "HO",
         "density_rmse": 6e-4},
        {"arch": "deep", "subset_size": 3, "molecule": "CH4",
         "density_rmse": None},                      # non-finite NN -> dropped
        {"arch": None, "subset_size": 1, "molecule": "HO",
         "density_rmse": 1e-4},                      # no cell -> dropped
    ]
    d = fig.holdout_density_by_arch_subset(rows)
    assert set(d) == {("deep", 1), ("deep", 3)}
    assert d[("deep", 1)] == pytest.approx(3e-4)
    assert d[("deep", 3)] == pytest.approx(6e-4)


def test_pbe_density_baseline_dedups_molecules():
    rows = [
        {"molecule": "HO", "density_rmse_pbe": 8e-4},
        {"molecule": "HO", "density_rmse_pbe": 8e-4},   # same molecule, 2nd spec
        {"molecule": "CH4", "density_rmse_pbe": 3e-4},
    ]
    # per-molecule mean first, then mean over molecules: (8e-4 + 3e-4)/2,
    # NOT the row-weighted (8+8+3)/3 e-4
    assert fig.pbe_density_baseline(rows) == pytest.approx(5.5e-4)
    # an explicit run-level table takes precedence over the rows
    tab = {"HO": {"density_rmse_pbe": 1e-3}}
    assert fig.pbe_density_baseline(rows, tab) == pytest.approx(1e-3)


def test_pbe_density_baseline_all_none_is_nan():
    rows = [{"molecule": "HO", "density_rmse_pbe": None},
            {"molecule": "CH4", "density_rmse_pbe": None}]
    assert math.isnan(fig.pbe_density_baseline(rows))
    assert math.isnan(fig.pbe_density_baseline([]))


def test_harmonic_mean_guards():
    assert fig._harmonic_mean(0.0, 5.0) == 0.0
    assert fig._harmonic_mean(5.0, -1.0) == 0.0
    assert fig._harmonic_mean(4.0, 4.0) == pytest.approx(4.0)
    # 2ab/(a+b): 2*3*6/9 = 4
    assert fig._harmonic_mean(3.0, 6.0) == pytest.approx(4.0)


def test_combined_ed_by_cell_gamma_self_calibration():
    energy = {("deep", 1): 8.0, ("deep", 3): 6.0, ("deep_attn", 1): 20.0}
    density = {("deep", 1): 0.004, ("deep", 3): 0.003, ("deep_attn", 1): 0.02}
    s = fig.combined_ed_by_cell(energy, 10.0, density, 0.005)
    assert s["gamma"] == pytest.approx(2000.0)          # 10 / 0.005
    assert s["ed_pbe"] == pytest.approx(10.0)           # ED_PBE == E_PBE identity
    c1 = s["cells"][("deep", 1)]
    assert c1["gammaD"] == pytest.approx(8.0)
    assert c1["ED"] == pytest.approx(8.0)               # equal legs -> the leg
    assert c1["beats_pbe"] is True
    c3 = s["cells"][("deep", 3)]
    assert c3["ED"] == pytest.approx(6.0)
    ca = s["cells"][("deep_attn", 1)]
    assert ca["gammaD"] == pytest.approx(40.0)
    assert ca["ED"] == pytest.approx(80.0 / 3.0)        # 2/(1/20 + 1/40)
    assert ca["beats_pbe"] is False


def test_combined_ed_by_cell_excludes_partial_cells():
    energy = {("deep", 1): 8.0, ("deep", 3): 6.0,
              ("deep_attn", 1): float("nan")}           # non-finite -> excluded
    density = {("deep", 1): 0.004, ("x", 1): 0.001}
    s = fig.combined_ed_by_cell(energy, 10.0, density, 0.005)
    # energy-only ("deep",3), density-only ("x",1) and the NaN cell all excluded
    assert set(s["cells"]) == {("deep", 1)}


def test_combined_ed_by_cell_raises_on_bad_anchors():
    energy = {("deep", 1): 8.0}
    density = {("deep", 1): 0.004}
    with pytest.raises(ValueError):
        fig.combined_ed_by_cell(energy, 10.0, density, 0.0)      # D_PBE <= 0
    with pytest.raises(ValueError):
        fig.combined_ed_by_cell(energy, float("nan"), density, 0.005)


def test_pbe_reaction_mae_baseline_dedups_by_name():
    rows = [
        {"name": "r1", "abs_error_pbe_kcalmol": 10.0},
        {"name": "r1", "abs_error_pbe_kcalmol": 10.0},  # dup name (2nd spec)
        {"name": "r2", "abs_error_pbe_kcalmol": 2.0},
    ]
    assert fig.pbe_reaction_mae_baseline(rows) == pytest.approx(6.0)
    assert math.isnan(fig.pbe_reaction_mae_baseline([]))


def test_spearman_rank_helper():
    assert fig._spearman([1, 2, 3], [1, 10, 100]) == pytest.approx(1.0)
    assert fig._spearman([1, 2, 3], [5, 4, 3]) == pytest.approx(-1.0)
    assert math.isnan(fig._spearman([1.0], [2.0]))              # n < 2
    assert math.isnan(fig._spearman([1, 1, 1], [1, 2, 3]))      # constant series


def test_ed_exclusion_and_coverage_notes():
    # exclusion note names one-leg-only cells; empty when the maps agree
    note = fig._ed_exclusion_note({("deep", 1): 1.0, ("deep", 3): 2.0},
                                  {("deep", 1): 1e-4, ("x", 1): 2e-4})
    assert "deep/ss3" in note and "x/ss1" in note
    assert fig._ed_exclusion_note({("deep", 1): 1.0}, {("deep", 1): 1e-4}) == ""
    # coverage warning fires when a cell's species set diverges from the union
    uniform = [
        {"arch": "deep", "subset_size": 1, "molecule": "HO",
         "density_rmse": 1e-4},
        {"arch": "deep", "subset_size": 3, "molecule": "HO",
         "density_rmse": 2e-4},
    ]
    assert fig._density_cell_coverage_warning(uniform) == ""
    divergent = uniform + [{"arch": "deep", "subset_size": 3,
                            "molecule": "CH4", "density_rmse": 2e-4}]
    warn = fig._density_cell_coverage_warning(divergent)
    assert "deep/ss1" in warn


def test_ed_exclusion_note_names_nonfinite_cells():
    # a cell keyed in BOTH maps but non-finite must not vanish silently
    note = fig._ed_exclusion_note(
        {("deep", 1): 1.0, ("deep", 5): float("nan")},
        {("deep", 1): 1e-4, ("deep", 5): float("nan")})
    assert "deep/ss5" in note
    # non-finite on one side only -> the finite side's *-only group
    note2 = fig._ed_exclusion_note({("deep", 2): float("nan")},
                                   {("deep", 2): 2e-4})
    assert "deep/ss2" in note2 and "density-only" in note2


def test_pbe_anchor_coverage_warning_flags_set_divergence():
    rows = [{"arch": "deep", "subset_size": 1, "molecule": "HO",
             "density_rmse": 2e-4, "density_rmse_pbe": 8e-4}]
    # run-level table carrying a species the NN legs never cover
    tab = {"HO": {"density_rmse_pbe": 8e-4},
           "CH4": {"density_rmse_pbe": 3e-4}}
    warn = fig._pbe_anchor_coverage_warning(rows, tab)
    assert "CH4" in warn
    # matched sets -> silent (table and inline variants)
    assert fig._pbe_anchor_coverage_warning(
        rows, {"HO": {"density_rmse_pbe": 8e-4}}) == ""
    assert fig._pbe_anchor_coverage_warning(rows, None) == ""
    # inline divergence: PBE column present where the NN channel failed
    rows2 = rows + [{"arch": "deep", "subset_size": 1, "molecule": "F2",
                     "density_rmse": None, "density_rmse_pbe": 5e-4}]
    assert "F2" in fig._pbe_anchor_coverage_warning(rows2, None)


def test_plot_combined_energy_density_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    rows = fig.collect_holdout_reaction_rows(run)
    hd = fig.collect_holdout_density_rows(run)
    d_cells = fig.holdout_density_by_arch_subset(hd)
    d_pbe = fig.pbe_density_baseline(hd, fig.load_pbe_density_table(run))
    wt = fig.combined_ed_by_cell(fig.wtmad2_by_arch_subset(rows),
                                 fig.wtmad2_pbe_baseline(rows), d_cells, d_pbe)
    mae = fig.combined_ed_by_cell(fig.reaction_mae_by_arch_subset(rows),
                                  fig.pbe_reaction_mae_baseline(rows),
                                  d_cells, d_pbe)
    p1 = fig.plot_combined_energy_density(wt, mae, tmp_path / "ed.png", "run_x")
    assert _png_ok(p1)
    # secondary leg unavailable -> placeholder panel, still a valid figure
    p2 = fig.plot_combined_energy_density(wt, None, tmp_path / "ed2.png",
                                          "run_x")
    assert _png_ok(p2)


def test_write_combined_ed_csv_columns_and_legs(tmp_path):
    energy = {("deep", 1): 8.0, ("deep_attn", 1): 20.0}
    density = {("deep", 1): 0.004, ("deep_attn", 1): 0.02}
    wt = fig.combined_ed_by_cell(energy, 10.0, density, 0.005)
    mae = fig.combined_ed_by_cell(energy, 12.0, density, 0.005)
    out = tmp_path / "ed.csv"
    fig.write_combined_ed_csv(
        {"wtmad2": wt, "mae": mae}, out,
        n_reactions={("deep", 1): 2, ("deep_attn", 1): 2},
        n_density={("deep", 1): 1, ("deep_attn", 1): 1})
    with out.open() as fh:
        rd = list(csv.DictReader(fh))
    assert rd
    assert set(rd[0]) == {
        "leg", "arch", "subset_size", "n_reactions", "n_density_species",
        "E_kcalmol", "D_rmse", "gamma", "gammaD_kcalmol", "ED_kcalmol",
        "E_pbe_kcalmol", "D_pbe_rmse", "ED_pbe_kcalmol", "beats_pbe",
        "E_scan_kcalmol", "D_scan_rmse", "ED_scan_kcalmol", "beats_scan",
        "ED_pbe_cell_kcalmol", "ED_scan_cell_kcalmol", "n_reactions_slice"}
    # absent SCAN legs write as EMPTY cells, never the string "None"
    assert all(r["ED_scan_kcalmol"] == "" and r["beats_scan"] == ""
               for r in rd)
    assert {r["leg"] for r in rd} == {"wtmad2", "mae"}
    for r in rd:
        # the self-calibration identity holds row-by-row
        assert (float(r["ED_pbe_kcalmol"])
                == pytest.approx(float(r["E_pbe_kcalmol"])))
    beat = {(r["leg"], r["arch"]): r["beats_pbe"] for r in rd}
    assert beat[("wtmad2", "deep")] == "True"
    assert beat[("wtmad2", "deep_attn")] == "False"
    # a None leg is skipped, not written as empty rows
    out2 = tmp_path / "ed2.csv"
    fig.write_combined_ed_csv({"wtmad2": wt, "mae": None}, out2,
                              n_reactions={}, n_density={})
    with out2.open() as fh:
        rd2 = list(csv.DictReader(fh))
    assert {r["leg"] for r in rd2} == {"wtmad2"}
    # counts_by_leg overrides the flat maps PER LEG -- the values must land
    # in the written rows (per-channel 3x3 CSV path)
    out3 = tmp_path / "ed3.csv"
    fig.write_combined_ed_csv(
        {"wtmad2": wt, "mae": mae}, out3, n_reactions={}, n_density={},
        counts_by_leg={"wtmad2": ({("deep", 1): 7, ("deep_attn", 1): 8},
                                  {("deep", 1): 5, ("deep_attn", 1): 6})})
    with out3.open() as fh:
        rd3 = list(csv.DictReader(fh))
    got = {(r["leg"], r["arch"]): (r["n_reactions"], r["n_density_species"])
           for r in rd3}
    assert got[("wtmad2", "deep")] == ("7", "5")
    assert got[("wtmad2", "deep_attn")] == ("8", "6")
    assert got[("mae", "deep")] == ("", "")     # no override -> flat maps


# ---------------------------------------------------------------------------
# Overview composites (per-pool WTMAD-2 + density + ED; in-sample companion)
# ---------------------------------------------------------------------------

def test_wtmad2_single_pool_reduces_to_scaled_mad():
    # One (deep, 1) cell. bh76: NN MAD=4 over mean|ref|=20 (PBE MAD=6);
    # w411: single reaction, NN err 5 over ref 100. Pool-filtered WTMAD-2 must
    # collapse to scale*MAD/mean|ref| (one-bucket reduction), while the full
    # 2-subset call is the genuine reweighting -- distinct from both.
    rows = [
        {"name": "b1", "arch": "deep", "subset_size": 1, "pool": "bh76",
         "abs_error_nn_kcalmol": 3.0, "abs_error_pbe_kcalmol": 5.0,
         "reaction_energy_ref_kcalmol": 10.0},
        {"name": "b2", "arch": "deep", "subset_size": 1, "pool": "bh76",
         "abs_error_nn_kcalmol": 5.0, "abs_error_pbe_kcalmol": 7.0,
         "reaction_energy_ref_kcalmol": 30.0},
        {"name": "w1", "arch": "deep", "subset_size": 1, "pool": "w411",
         "abs_error_nn_kcalmol": 5.0, "abs_error_pbe_kcalmol": 9.0,
         "reaction_energy_ref_kcalmol": 100.0},
    ]
    bh = [r for r in rows if r["pool"] == "bh76"]
    w4 = [r for r in rows if r["pool"] == "w411"]
    assert fig.wtmad2_by_arch_subset(bh)[("deep", 1)] == pytest.approx(
        56.84 * 4.0 / 20.0)
    assert fig.wtmad2_pbe_baseline(bh) == pytest.approx(56.84 * 6.0 / 20.0)
    assert fig.wtmad2_by_arch_subset(w4)[("deep", 1)] == pytest.approx(
        56.84 * 5.0 / 100.0)
    assert fig.wtmad2_by_arch_subset(rows)[("deep", 1)] == pytest.approx(
        56.84 / 3.0 * (2 * (4.0 / 20.0) + 1 * (5.0 / 100.0)))


def test_grouped_arch_bars_pbe_line_none_skips_baseline():
    f1, ax1 = fig.plt.subplots()
    fig._grouped_arch_bars(ax1, {("deep", 1): 5.0}, ["deep"], [1],
                           pbe_line=None, title="t")
    assert not ax1.lines                                  # no PBE axhline
    _, labels1 = ax1.get_legend_handles_labels()
    assert "PBE" not in labels1 and "beats PBE" not in labels1
    fig.plt.close(f1)
    f2, ax2 = fig.plt.subplots()
    fig._grouped_arch_bars(ax2, {("deep", 1): 5.0}, ["deep"], [1],
                           pbe_line=10.0, title="t")
    assert len(ax2.lines) == 1                            # the PBE axhline
    _, labels2 = ax2.get_legend_handles_labels()
    assert "PBE" in labels2 and "beats PBE" in labels2    # 5.0 beats 10.0
    fig.plt.close(f2)


def test_insample_ae_strip_panel_points():
    ae_rows = [
        {"arch": "deep", "subset_size": 1, "molecule": "HO",
         "AE_error_kcalmol": 6.0},
        {"arch": "deep", "subset_size": 1, "molecule": "CH4",
         "AE_error_kcalmol": -2.0},                      # plotted as |.| = 2.0
        {"arch": "deep", "subset_size": 1, "molecule": None,
         "AE_error_kcalmol": 1.0},                       # no molecule -> drop
        {"arch": "deep", "subset_size": 1, "molecule": "X",
         "AE_error_kcalmol": None},                      # no AE -> drop
    ]
    f1, ax = fig.plt.subplots()
    fig._insample_ae_strip_panel(ax, ae_rows)
    assert ax.get_yscale() == "log"
    assert len(ax.collections) == 2                      # HO + CH4 points only
    ticks = [t.get_text() for t in ax.get_xticklabels()]
    assert "HO" in ticks and "CH4" in ticks and "X" not in ticks
    fig.plt.close(f1)


def test_plot_insample_overview_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    ae = fig.collect_insample_ae_rows(run)
    dr = fig.collect_insample_density_rows(run)
    p1 = fig.plot_insample_overview(ae, dr, tmp_path / "io.png", "run_x")
    assert _png_ok(p1)
    # with the PBE density columns present, panel C gains the dashed line
    for r in dr:
        r["density_rmse_pbe"] = 9e-4
    p2 = fig.plot_insample_overview(ae, dr, tmp_path / "io2.png", "run_x")
    assert _png_ok(p2)


def test_holdout_eval_note_counts():
    rows = [
        {"name": "r1", "pool": "bh76", "abs_error_pbe_kcalmol": 1.0},
        {"name": "r1", "pool": "bh76", "abs_error_pbe_kcalmol": 1.0},  # dup name
        {"name": "r2", "pool": "bh76", "abs_error_pbe_kcalmol": 2.0},
        {"name": "w1", "pool": "w411", "abs_error_pbe_kcalmol": 3.0},
    ]
    hd = [
        {"molecule": "HO", "density_rmse": 1e-4, "density_rmse_pbe": 2e-4},
        {"molecule": "CH4", "density_rmse": 1e-4, "density_rmse_pbe": 2e-4},
        {"molecule": "F2", "density_rmse": None, "density_rmse_pbe": 5e-4},
    ]
    note = fig._holdout_eval_note(rows, hd)
    assert "BH76 2" in note and "W4-11 1" in note      # name-deduplicated
    assert "2 NN / 3 PBE" in note                      # unequal-channel branch
    hd_eq = [dict(r, density_rmse=1e-4) for r in hd]
    assert "3 species" in fig._holdout_eval_note(rows, hd_eq)
    assert fig._holdout_eval_note([], []) == ""
    # energy-figure variant: reactions clause only, no density clause
    note_e = fig._holdout_eval_note(rows, [])
    assert "BH76 2" in note_e and "density" not in note_e


def test_ed_decomposition_panel_draws_cells():
    s = fig.combined_ed_by_cell({("deep", 1): 8.0, ("deep", 3): 6.0}, 10.0,
                                {("deep", 1): 0.004, ("deep", 3): 0.003},
                                0.005)
    f1, ax = fig.plt.subplots()
    fig._ed_decomposition_panel(ax, s)
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    assert len(ax.collections) >= 1            # cell points + the PBE x
    assert len(ax.lines) == 4                  # y=x locus + 3 iso-ED contours
    assert "iso-" + fig._ED_SYM in ax.get_title()
    fig.plt.close(f1)


def test_plot_holdout_density_per_arch_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    hd = fig.collect_holdout_density_rows(run)
    tab = fig.load_pbe_density_table(run)
    p1 = fig.plot_holdout_density_per_arch(hd, tmp_path / "pa.png", "run_x",
                                           pbe_table=tab)
    assert _png_ok(p1)
    # PBE-only shape (no NN channel anywhere) still renders the baseline
    pbe_rows = [dict(r, density_rmse=None) for r in hd]
    p2 = fig.plot_holdout_density_per_arch(pbe_rows, tmp_path / "pa2.png",
                                           "run_x", pbe_table=tab)
    assert _png_ok(p2)


def test_collect_holdout_reaction_rows_carries_species(tmp_path):
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    r = next(x for x in rows if x["name"] == "bh76_a")
    assert r["reactants"] == ["HO", "h"] and r["products"] == ["HOh_ts"]


def test_species_pools_maps_overlap():
    rows = [
        {"pool": "bh76", "reactants": ["HO", "h"], "products": ["HOh_ts"]},
        {"pool": "w411", "reactants": ["HO"], "products": ["h", "o"]},
        {"pool": None, "reactants": ["ghost"], "products": []},
    ]
    m = fig._species_pools(rows)
    assert m["HO"] == {"bh76", "w411"}          # overlap species: both channels
    assert m["HOh_ts"] == {"bh76"}
    assert m["o"] == {"w411"}
    assert "ghost" not in m                     # pool-less rows ignored


def test_channel_ed_summaries_per_channel_gammas(tmp_path):
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    rows = fig.collect_holdout_reaction_rows(run)
    hd = fig.collect_holdout_density_rows(run)
    tab = fig.load_pbe_density_table(run)
    ch = fig.channel_ed_summaries(rows, hd, tab)
    assert set(ch) == {"bh76", "w411", "combined"}
    assert all(ch[c] is not None for c in ch)
    # each channel self-calibrates from its own PBE anchors
    assert ch["bh76"]["gamma"] != ch["w411"]["gamma"]
    assert ch["combined"]["ed_pbe"] == pytest.approx(ch["combined"]["e_pbe"])
    # a channel with no reactions degrades to None, others survive
    ch2 = fig.channel_ed_summaries(
        [r for r in rows if r["pool"] == "bh76"], hd, tab)
    assert ch2["w411"] is None and ch2["bh76"] is not None


def test_plot_density_energy_3x3_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    rows = fig.collect_holdout_reaction_rows(run)
    hd = fig.collect_holdout_density_rows(run)
    tab = fig.load_pbe_density_table(run)
    p1 = fig.plot_density_energy_3x3(rows, hd, tmp_path / "g.png", "run_x",
                                     pbe_table=tab)
    assert _png_ok(p1)
    # single-pool input: the w411 channel panels degrade to placeholders
    bh_only = [r for r in rows if r["pool"] == "bh76"]
    p2 = fig.plot_density_energy_3x3(bh_only, hd, tmp_path / "g2.png",
                                     "run_x", pbe_table=tab)
    assert _png_ok(p2)


def test_gamma_zero_intercept_hand_slope():
    # zero-intercept least squares: slope = sum(eps*W)/sum(eps^2)
    pairs = [(1.0, 2.0), (2.0, 4.0)]
    assert fig.gamma_zero_intercept(pairs) == pytest.approx(2.0)
    pairs2 = [(1.0, 3.0), (2.0, 2.0)]         # (3 + 4)/(1 + 4) = 7/5
    assert fig.gamma_zero_intercept(pairs2) == pytest.approx(1.4)
    assert math.isnan(fig.gamma_zero_intercept([]))
    assert math.isnan(fig.gamma_zero_intercept([(0.0, 5.0)]))


def test_combined_ed_fixed_gamma_hand_values():
    s = fig.combined_ed_fixed_gamma({("deep", 1): 8.0}, 10.0,
                                    {("deep", 1): 0.004}, 0.005, 2000.0)
    # fixed gamma: ed_pbe = harmonic(10, 2000*0.005=10) = 10 here, but with
    # gamma=1000 the PBE point moves OFF the diagonal:
    assert s["gamma"] == pytest.approx(2000.0)
    assert s["ed_pbe"] == pytest.approx(10.0)
    assert s["cells"][("deep", 1)]["ED"] == pytest.approx(8.0)
    s2 = fig.combined_ed_fixed_gamma({("deep", 1): 8.0}, 10.0,
                                     {("deep", 1): 0.004}, 0.005, 1000.0)
    # ed_pbe = harmonic(10, 5) = 2/(1/10+1/5) = 20/3 != e_pbe
    assert s2["ed_pbe"] == pytest.approx(20.0 / 3.0)
    assert s2["cells"][("deep", 1)]["gammaD"] == pytest.approx(4.0)
    assert s2["cells"][("deep", 1)]["ED"] == pytest.approx(
        2.0 / (1.0 / 8.0 + 1.0 / 4.0))


def test_nonempirical_gamma_from_cache(tmp_path):
    cache = {
        "m1": {"pbe": {"density_eps_l1": 0.010},
               "scan": {"density_eps_l1": 0.006}},
        "m2": {"pbe": {"density_eps_l1": 0.014},
               "scan": {"density_eps_l1": 0.008}},
    }
    (tmp_path / "nonempirical_pool_def2-svp.json").write_text(
        json.dumps(cache))
    # seam: WTMAD-2 per functional supplied directly (no pool loader)
    out = fig.nonempirical_gamma(tmp_path, basis="def2-svp",
                                 cache_dir=tmp_path,
                                 _wtmad={"pbe": 12.0, "scan": 7.0})
    # eps means: pbe 0.012, scan 0.007
    # slope = (0.012*12 + 0.007*7)/(0.012^2 + 0.007^2) = 0.193/0.000193
    assert out["gamma"] == pytest.approx(0.193 / 0.000193)
    assert out["n_functionals"] == 2
    assert set(out["pairs"]) == {"pbe", "scan"}
    # absent cache -> empty dict
    assert fig.nonempirical_gamma(tmp_path / "nope", basis="def2-svp") == {}


def test_holdout_density_by_arch_subset_key_param():
    rows = [{"arch": "deep", "subset_size": 1, "molecule": "HO",
             "density_rmse": 2e-4, "density_eps_l1": 3e-3}]
    assert fig.holdout_density_by_arch_subset(rows)[("deep", 1)] == \
        pytest.approx(2e-4)
    assert fig.holdout_density_by_arch_subset(
        rows, key="density_eps_l1")[("deep", 1)] == pytest.approx(3e-3)


def test_collectors_carry_eps_columns(tmp_path):
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    hd = fig.collect_holdout_density_rows(run)
    assert all("density_eps_l1" in r and "density_eps_l1_pbe" in r
               for r in hd)
    finite = [r for r in hd if fig._is_num(r.get("density_eps_l1"))]
    assert finite and all(r["density_eps_l1"] == pytest.approx(2.5e-4)
                          for r in finite)


def test_build_emits_dfs_units_ed_legs_when_eps_present(tmp_path):
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    out = tmp_path / "f"
    fig.build_density_energy_figures(run, out)
    with (out / "ablation_combined_energy_density.csv").open() as fh:
        legs = {r["leg"] for r in csv.DictReader(fh)}
    # the DFS-units leg (Letter's gamma transplanted to Eq. 20 units) rides
    # along whenever the eps columns are present in the pulled data
    assert "wtmad2" in legs and "mae" in legs
    assert "wtmad2_eps_gamma_dfs" in legs


def test_nonempirical_gamma_common_support_and_malformed(tmp_path):
    # unequal support (partial cache): m3 is pbe-only, so the fit restricts
    # to the {m1, m2} intersection -- slope identical to the equal-support
    # fixture -- and the coverage fields disclose the drop
    cache = {
        "m1": {"pbe": {"density_eps_l1": 0.010},
               "scan": {"density_eps_l1": 0.006}},
        "m2": {"pbe": {"density_eps_l1": 0.014},
               "scan": {"density_eps_l1": 0.008}},
        "m3": {"pbe": {"density_eps_l1": 0.100}},
    }
    p = tmp_path / "nonempirical_pool_def2-svp.json"
    p.write_text(json.dumps(cache))
    out = fig.nonempirical_gamma(tmp_path, basis="def2-svp",
                                 cache_dir=tmp_path,
                                 _wtmad={"pbe": 12.0, "scan": 7.0})
    assert out["gamma"] == pytest.approx(0.193 / 0.000193)
    assert out["n_species"] == 2 and out["n_species_dropped"] == 1
    # disjoint support -> empty intersection -> {}
    p.write_text(json.dumps({"m1": {"pbe": {"density_eps_l1": 1e-2}},
                             "m2": {"scan": {"density_eps_l1": 1e-2}}}))
    assert fig.nonempirical_gamma(tmp_path, basis="def2-svp",
                                  cache_dir=tmp_path,
                                  _wtmad={"pbe": 1.0, "scan": 1.0}) == {}
    # malformed-but-parseable caches degrade to {} / skip, never raise
    for payload in ([1, 2], {"m1": 3}, {"m1": {"pbe": 5}}):
        p.write_text(json.dumps(payload))
        assert fig.nonempirical_gamma(tmp_path, basis="def2-svp",
                                      cache_dir=tmp_path,
                                      _wtmad={"pbe": 1.0}) == {}


def test_gamma_mode_keys_and_fixed_stamp_truthfulness():
    self_s = fig.combined_ed_by_cell({("deep", 1): 8.0}, 10.0,
                                     {("deep", 1): 4e-4}, 5e-4)
    assert self_s["gamma_mode"] == "self_calibrated"
    fixed_s = fig.combined_ed_fixed_gamma({("deep", 1): 8.0}, 10.0,
                                          {("deep", 1): 0.004}, 0.005,
                                          1000.0)
    assert fixed_s["gamma_mode"] == "fixed"
    # under an EXTERNAL gamma the self-calibration claims are false; the
    # panels must not print them
    for panel in (lambda ax, s: fig._ed_lines_panel(ax, s, "t"),
                  fig._ed_decomposition_rich_panel,
                  fig._ed_decomposition_panel):
        f1, ax = fig.plt.subplots()
        panel(ax, fixed_s)
        texts = [t.get_text() for t in ax.texts]
        assert not any("self-calibrated" in t for t in texts)
        labels = ax.get_legend_handles_labels()[1]
        assert not any("by self-calibration" in lb or "by construction" in lb
                       for lb in labels)
        fig.plt.close(f1)
        # self-calibrated summaries keep the exact historical strings
        f2, ax2 = fig.plt.subplots()
        panel(ax2, self_s)
        lbls2 = ax2.get_legend_handles_labels()[1]
        txts2 = [t.get_text() for t in ax2.texts]
        assert (any("(self-calibrated)" in t for t in txts2)
                or any("by self-calibration" in lb or "by construction" in lb
                       for lb in lbls2))
        fig.plt.close(f2)
    # the two stamped panels print the fixed gamma explicitly
    for panel in (lambda ax, s: fig._ed_lines_panel(ax, s, "t"),
                  fig._ed_decomposition_rich_panel):
        f3, ax3 = fig.plt.subplots()
        panel(ax3, fixed_s)
        assert any("fixed, external" in t.get_text() for t in ax3.texts)
        fig.plt.close(f3)


def test_build_discloses_partial_eps_backfill(tmp_path, capsys):
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    # strip the NN eps column from ONE spec: the state a partial backfill
    # leaves (its cell keeps the RMSE channel but loses the eps channel)
    specs = sorted((run / "checkpoints").glob("spec_*"))
    pm = specs[0] / "eval_holdout" / "per_molecule.json"
    rows = json.loads(pm.read_text())
    for r in rows:
        r["density_eps_l1"] = None
    pm.write_text(json.dumps(rows))
    out = tmp_path / "f"
    fig.build_density_energy_figures(run, out)
    printed = capsys.readouterr().out
    assert "eps columns cover" in printed and "partial backfill" in printed
    with (out / "ablation_combined_energy_density.csv").open() as fh:
        rows_csv = list(csv.DictReader(fh))
    n_wt = sum(1 for r in rows_csv if r["leg"] == "wtmad2")
    n_eps = sum(1 for r in rows_csv if r["leg"] == "wtmad2_eps_gamma_dfs")
    assert 0 < n_eps < n_wt


def test_build_discloses_eps_cell_species_divergence(tmp_path, capsys):
    """Per-species strip WITHIN one spec (RMSE intact): the eps-channel
    cell-homogeneity guard must fire at the build site -- and neither
    sibling guard (whole-cell missing / anchor-vs-union) may fire, so the
    three disclosures partition the narrowing modes."""
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    for i, sd in enumerate(sorted((run / "checkpoints").glob("spec_*"))):
        pm = sd / "eval_holdout" / "per_molecule.json"
        if not pm.is_file():
            continue
        rows = json.loads(pm.read_text())
        rows.append({
            "molecule": "OH2", "density_rmse": 3e-4, "density_l1": 2e-5,
            "density_rmse_pbe": 9e-4, "density_l1_pbe": 6e-5,
            # first spec: eps stripped for this species only (RMSE intact)
            "density_eps_l1": None if i == 0 else 3.5e-4,
            "density_eps_l1_pbe": 8e-4,
            "n_electrons": 10.0, "grid_weight_sum": 110.0,
            "ref_density_method": "ccsd", "from_training_subset": False})
        pm.write_text(json.dumps(rows))
    fig.build_density_energy_figures(run, tmp_path / "f")
    printed = capsys.readouterr().out
    assert "DFS-units ED eps cells:" in printed
    assert "eps columns cover" not in printed        # whole-cell guard silent
    assert "DFS-units ED eps anchor:" not in printed  # anchor guard silent


def test_build_discloses_eps_anchor_only_species(tmp_path, capsys):
    """One species carrying a PBE eps but NO NN eps in ANY spec: the
    eps-channel anchor-vs-NN-union guard must fire at the build site (the
    cell-homogeneity guard stays silent -- every cell sees the same NN set)."""
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    for sd in sorted((run / "checkpoints").glob("spec_*")):
        pm = sd / "eval_holdout" / "per_molecule.json"
        if not pm.is_file():
            continue
        rows = json.loads(pm.read_text())
        rows.append({
            "molecule": "OF2", "density_rmse": 4e-4, "density_l1": 3e-5,
            "density_rmse_pbe": 9e-4, "density_l1_pbe": 6e-5,
            "density_eps_l1": None, "density_eps_l1_pbe": 8e-4,
            "n_electrons": 26.0, "grid_weight_sum": 120.0,
            "ref_density_method": "ccsd", "from_training_subset": False})
        pm.write_text(json.dumps(rows))
    fig.build_density_energy_figures(run, tmp_path / "f")
    printed = capsys.readouterr().out
    assert "DFS-units ED eps anchor:" in printed and "OF2" in printed
    assert "DFS-units ED eps cells:" not in printed


def test_build_dfs_units_png_notes_missing_cells(tmp_path, monkeypatch):
    """Partial eps coverage: the DFS-units parity figure renders from the
    FIXED-gamma summaries with the missing cells named in its note band (the
    on-figure twin of the stdout disclosure), no fit panel without a pool
    cache, and the fixed-gamma caveat instead of the self-calibration one."""
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    specs = sorted((run / "checkpoints").glob("spec_*"))
    pm = specs[0] / "eval_holdout" / "per_molecule.json"
    rows = json.loads(pm.read_text())
    for r in rows:
        r["density_eps_l1"] = None
    pm.write_text(json.dumps(rows))
    ed_calls, dec_calls = [], []

    def ed_spy(wt_summary, mae_summary, out_path, run_id, **kw):
        ed_calls.append((wt_summary, mae_summary, Path(out_path), kw))
        Path(out_path).write_bytes(b"x" * 4096)
        return Path(out_path)

    def dec_spy(summary, out_path, run_id, **kw):
        dec_calls.append((summary, Path(out_path), kw))
        Path(out_path).write_bytes(b"x" * 4096)
        return Path(out_path)

    monkeypatch.setattr(fig, "plot_combined_energy_density", ed_spy)
    monkeypatch.setattr(fig, "plot_ed_decomposition", dec_spy)
    fig.build_density_energy_figures(run, tmp_path / "f")
    dfs = [c for c in ed_calls if c[2].name
           == "ablation_combined_energy_density_dfs_units.png"]
    assert len(dfs) == 1
    wt_s, fit_s, _, kw = dfs[0]
    assert wt_s["gamma_mode"] == "fixed"
    assert wt_s["gamma"] == pytest.approx(1084.87)
    assert fit_s is None                    # no pool cache in the run dir
    assert "eps columns cover" in kw["note"] and "missing" in kw["note"]
    assert "deep" in kw["note"]             # the dropped cell is named
    assert "NOT self-calibrated" in kw["caveat"]
    assert "ED_PBE == E_PBE" not in kw["caveat"]
    assert "published" in kw["panel_titles"][0]
    dfs_dec = [c for c in dec_calls if c[1].name
               == "ablation_ed_decomposition_dfs_units.png"]
    assert len(dfs_dec) == 1
    assert dfs_dec[0][0]["gamma_mode"] == "fixed"
    # the twin's title identifies the DFS-units variant by the paper's
    # combined-metric symbol
    assert fig._ED_N_SYM in dfs_dec[0][2]["title"]
    assert "eps columns cover" in dfs_dec[0][2]["note"]


def test_build_dfs_units_png_absent_without_eps(tmp_path, capsys):
    """Old-schema pulls (no eps columns anywhere) must NOT gain the DFS-units
    figures -- the RMSE-channel ED family renders unchanged, and the skip is
    disclosed with the stale-file warning (the suite's convention for every
    gated figure)."""
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    for sd in (run / "checkpoints").glob("spec_*"):
        pm = sd / "eval_holdout" / "per_molecule.json"
        if not pm.is_file():
            continue
        rows = json.loads(pm.read_text())
        for r in rows:
            r["density_eps_l1"] = None
            r["density_eps_l1_pbe"] = None
        pm.write_text(json.dumps(rows))
    out = tmp_path / "f"
    names = {p.name for p in fig.build_density_energy_figures(run, out)}
    assert "ablation_combined_energy_density.png" in names
    assert "ablation_combined_energy_density_dfs_units.png" not in names
    assert not (out
                / "ablation_combined_energy_density_dfs_units.png").exists()
    assert not (out / "ablation_ed_decomposition_dfs_units.png").exists()
    assert not (out
                / "ablation_density_energy_overview_dfs_units.png").exists()
    assert not (out / "ablation_density_energy_3x3_dfs_units.png").exists()
    assert not (out / "ablation_density_energy_3x3_dfs_units.csv").exists()
    assert not (out
                / "ablation_density_parity_by_channel_dfs_units.png"
                ).exists()
    # the RMSE-channel standalone parity still renders (its gate is the
    # RMSE density data, present here)
    assert (out / "ablation_density_parity_by_channel.png").exists()
    printed = capsys.readouterr().out
    assert "skipping the DFS-units ED legs" in printed
    assert "a stale file from a prior render persists" in printed


def test_build_dfs_units_fit_panel_with_cache(tmp_path, monkeypatch):
    """A resolving nonempirical pool cache puts the own-axes-fit leg in panel
    C of the DFS-units figure, its provenance line in the note band, and --
    the fit being the calibration on THIS data's axes -- makes it the
    OPERATIVE gamma of every single-gamma DFS-units view (decomposition
    twin, overview twin, 3x3 twin); the twin CSV carries both leg
    families."""
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    monkeypatch.setattr(
        fig, "nonempirical_gamma",
        lambda run_dir, **kw: {"gamma": 900.0, "n_functionals": 6,
                               "n_species": 5, "n_species_dropped": 1})
    calls, dec_calls, ov_calls, x3_calls = [], [], [], []

    def ed_spy(wt_summary, mae_summary, out_path, run_id, **kw):
        calls.append((wt_summary, mae_summary, Path(out_path), kw))
        Path(out_path).write_bytes(b"x" * 4096)
        return Path(out_path)

    def dec_spy(summary, out_path, run_id, **kw):
        dec_calls.append((summary, Path(out_path), kw))
        Path(out_path).write_bytes(b"x" * 4096)
        return Path(out_path)

    def ov_spy(rows, hd_rows, out_path, run_id, **kw):
        ov_calls.append((Path(out_path), kw))
        Path(out_path).write_bytes(b"x" * 4096)
        return Path(out_path)

    def x3_spy(rows, hd_rows, out_path, run_id, **kw):
        x3_calls.append((Path(out_path), kw))
        Path(out_path).write_bytes(b"x" * 4096)
        return Path(out_path)

    monkeypatch.setattr(fig, "plot_combined_energy_density", ed_spy)
    monkeypatch.setattr(fig, "plot_ed_decomposition", dec_spy)
    monkeypatch.setattr(fig, "plot_density_energy_overview", ov_spy)
    monkeypatch.setattr(fig, "plot_density_energy_3x3", x3_spy)
    out = tmp_path / "f"
    fig.build_density_energy_figures(run, out)
    dfs = [c for c in calls if c[2].name
           == "ablation_combined_energy_density_dfs_units.png"]
    assert len(dfs) == 1
    dfs_s, fit_s, _, kw = dfs[0]
    assert fit_s is not None and fit_s["gamma_mode"] == "fixed"
    assert fit_s["gamma"] == pytest.approx(900.0)
    assert fit_s["gamma_source"] == "own-axes fit"
    assert dfs_s["gamma_source"] == "DFS published"   # panel A keeps both
    assert "own-axes gamma = 900" in kw["note"]
    assert "1 species dropped for unequal support" in kw["note"]
    # operative gamma on every single-gamma twin = the fit, not 1084.87
    dec_twin = [c for c in dec_calls if c[1].name
                == "ablation_ed_decomposition_dfs_units.png"]
    assert len(dec_twin) == 1
    assert dec_twin[0][0]["gamma"] == pytest.approx(900.0)
    assert dec_twin[0][0]["gamma_source"] == "own-axes fit"
    ov_twin = [kw2 for p, kw2 in ov_calls if p.name
               == "ablation_density_energy_overview_dfs_units.png"]
    assert len(ov_twin) == 1
    assert ov_twin[0]["ed_summary"]["gamma"] == pytest.approx(900.0)
    assert ov_twin[0]["ed_summary"]["gamma_source"] == "own-axes fit"
    x3_twin = [kw2 for p, kw2 in x3_calls if p.name
               == "ablation_density_energy_3x3_dfs_units.png"]
    assert len(x3_twin) == 1
    assert all(s["gamma"] == pytest.approx(900.0)
               and s["gamma_source"] == "own-axes fit"
               for s in x3_twin[0]["ch_summaries"].values())
    # the twin CSV carries BOTH leg families, each at its own gamma
    with (out / "ablation_density_energy_3x3_dfs_units.csv").open() as fh:
        rows_csv = list(csv.DictReader(fh))
    legs = {r["leg"] for r in rows_csv}
    assert legs == {f"{ch}_wtmad2_eps_gamma_{tag}"
                    for ch in ("bh76", "w411", "combined")
                    for tag in ("dfs", "fit")}
    for r in rows_csv:
        want = 1084.87 if r["leg"].endswith("_dfs") else 900.0
        assert float(r["gamma"]) == pytest.approx(want)


def test_plot_combined_energy_density_dfs_units_renders(tmp_path):
    """Real render of the DFS-units variant: fixed-gamma summaries with the
    panel/placeholder/title overrides, both with and without the fit leg,
    plus the decomposition twin's title override."""
    dfs_s = fig.combined_ed_fixed_gamma(
        {("deep", 1): 8.0, ("deep", 3): 5.0}, 10.0,
        {("deep", 1): 0.004, ("deep", 3): 0.003}, 0.005, 1084.87)
    fit_s = fig.combined_ed_fixed_gamma(
        {("deep", 1): 8.0, ("deep", 3): 5.0}, 10.0,
        {("deep", 1): 0.004, ("deep", 3): 0.003}, 0.005, 900.0)
    p1 = fig.plot_combined_energy_density(
        dfs_s, fit_s, tmp_path / "dfs_units.png", "run_x",
        panel_titles=("published-gamma panel", "own-axes panel"),
        second_leg_placeholder="no fit", title="DFS units")
    assert _png_ok(p1)
    p2 = fig.plot_combined_energy_density(
        dfs_s, None, tmp_path / "dfs_units_nofit.png", "run_x",
        panel_titles=("published-gamma panel", "own-axes panel"),
        second_leg_placeholder="no fit", title="DFS units")
    assert _png_ok(p2)
    p3 = fig.plot_ed_decomposition(
        dfs_s, tmp_path / "dfs_units_decomp.png", "run_x",
        title="DFS-units decomposition")
    assert _png_ok(p3)


def test_channel_ed_summaries_fixed_gamma_eps(tmp_path):
    """The fixed-gamma variant: one shared external gamma on the Eq. 20 eps
    channel across all three channels (gamma_mode="fixed"), D drawn from the
    eps columns; an RMSE-only pbe_table falls back to the inline eps
    columns. The no-kwargs call keeps the self-calibrated behavior."""
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    rows = fig.collect_holdout_reaction_rows(run)
    hd = fig.collect_holdout_density_rows(run)
    tab = fig.load_pbe_density_table(run)     # RMSE-only table: no eps keys
    ch = fig.channel_ed_summaries(rows, hd, tab, fixed_gamma=1084.87,
                                  density_key="density_eps_l1",
                                  pbe_density_key="density_eps_l1_pbe")
    assert set(ch) == {"bh76", "w411", "combined"}
    assert all(s is not None for s in ch.values())
    for s in ch.values():
        assert s["gamma_mode"] == "fixed"
        assert s["gamma"] == pytest.approx(1084.87)
        # D from the eps columns (fixture: NN 2.5e-4, PBE 7e-4 inline)
        assert s["d_pbe"] == pytest.approx(7e-4)
        for c in s["cells"].values():
            assert c["D"] == pytest.approx(2.5e-4)
    # shared gamma -> ED_PBE identical across channels' density anchors only
    # when E_PBE matches; the self-calibrated default is unchanged
    ch_default = fig.channel_ed_summaries(rows, hd, tab)
    assert ch_default["combined"]["gamma_mode"] == "self_calibrated"
    assert ch_default["bh76"]["gamma"] != ch_default["w411"]["gamma"]


def test_build_dfs_units_composite_twins(tmp_path, monkeypatch):
    """The build site renders DFS-units twins of the held-out overview and
    the per-channel 3x3: fixed-gamma summaries, eps parity keys, disclosure
    note, DFS-units caveats -- while the originals keep their defaults. The
    3x3 twin CSV carries the per-channel eps legs."""
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    ov_calls, x3_calls = [], []

    def ov_spy(rows, hd_rows, out_path, run_id, **kw):
        ov_calls.append((Path(out_path), kw))
        Path(out_path).write_bytes(b"x" * 4096)
        return Path(out_path)

    def x3_spy(rows, hd_rows, out_path, run_id, **kw):
        x3_calls.append((Path(out_path), kw))
        Path(out_path).write_bytes(b"x" * 4096)
        return Path(out_path)

    monkeypatch.setattr(fig, "plot_density_energy_overview", ov_spy)
    monkeypatch.setattr(fig, "plot_density_energy_3x3", x3_spy)
    out = tmp_path / "f"
    fig.build_density_energy_figures(run, out)
    ov_twin = [kw for p, kw in ov_calls if p.name
               == "ablation_density_energy_overview_dfs_units.png"]
    x3_twin = [kw for p, kw in x3_calls if p.name
               == "ablation_density_energy_3x3_dfs_units.png"]
    assert len(ov_twin) == 1 and len(x3_twin) == 1
    assert ov_twin[0]["ed_summary"]["gamma_mode"] == "fixed"
    assert ov_twin[0]["parity_nn_key"] == "density_eps_l1"
    # NO calibration cache in this fixture -> the operative gamma falls back
    # to the published slope, and the summaries say so
    assert ov_twin[0]["ed_summary"]["gamma"] == pytest.approx(1084.87)
    assert ov_twin[0]["ed_summary"]["gamma_source"] == "DFS published"
    # the D leg must be the EPS channel, not the RMSE one -- the published
    # gamma is dimensionally valid only on Eq. 20 units (fixture: NN eps
    # 2.5e-4 / PBE eps 7e-4, vs RMSE 2e-4 / 8e-4)
    assert ov_twin[0]["ed_summary"]["d_pbe"] == pytest.approx(7e-4)
    chs = x3_twin[0]["ch_summaries"]
    assert all(s is not None and s["gamma_mode"] == "fixed"
               and s["gamma"] == pytest.approx(1084.87)
               and s["gamma_source"] == "DFS published"
               for s in chs.values())
    for s in chs.values():
        assert s["d_pbe"] == pytest.approx(7e-4)
        for c in s["cells"].values():
            assert c["D"] == pytest.approx(2.5e-4)
    assert x3_twin[0]["density_nn_key"] == "density_eps_l1"
    assert x3_twin[0]["density_pbe_key"] == "density_eps_l1_pbe"
    # titles are clean -- the in-panel stamp carries value + source
    assert x3_twin[0]["ed_gamma_label"] == ""
    # the ORIGINAL calls keep their defaults (no parity/gamma overrides)
    ov_orig = [kw for p, kw in ov_calls if p.name
               == "ablation_density_energy_overview.png"]
    x3_orig = [kw for p, kw in x3_calls if p.name
               == "ablation_density_energy_3x3.png"]
    assert len(ov_orig) == 1 and "parity_nn_key" not in ov_orig[0]
    # the ORIGINAL 3x3 runs on its RMSE defaults -- no density-key override
    assert len(x3_orig) == 1 and "density_nn_key" not in x3_orig[0]
    assert x3_orig[0]["ch_summaries"]["combined"]["gamma_mode"] == \
        "self_calibrated"
    # the twin CSV carries the per-channel eps legs at the shared gamma
    with (out / "ablation_density_energy_3x3_dfs_units.csv").open() as fh:
        rows_csv = list(csv.DictReader(fh))
    legs = {r["leg"] for r in rows_csv}
    # no cache -> only the published-gamma legs
    assert legs == {"bh76_wtmad2_eps_gamma_dfs", "w411_wtmad2_eps_gamma_dfs",
                    "combined_wtmad2_eps_gamma_dfs"}
    assert all(float(r["gamma"]) == pytest.approx(1084.87) for r in rows_csv)
    assert all(float(r["D_pbe_rmse"]) == pytest.approx(7e-4)
               for r in rows_csv)
    assert all(float(r["D_rmse"]) == pytest.approx(2.5e-4)
               for r in rows_csv)


def test_plot_composite_dfs_units_twins_render(tmp_path):
    """Real renders of the two composite twins with the override kwargs."""
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    rows = fig.collect_holdout_reaction_rows(run)
    hd = fig.collect_holdout_density_rows(run)
    ch_eps = fig.channel_ed_summaries(rows, hd, None, fixed_gamma=1084.87,
                                      density_key="density_eps_l1",
                                      pbe_density_key="density_eps_l1_pbe")
    p1 = fig.plot_density_energy_3x3(
        rows, hd, tmp_path / "x3_dfs.png", "run_x",
        ch_summaries=ch_eps, density_nn_key="density_eps_l1",
        density_pbe_key="density_eps_l1_pbe",
        density_unit_label=fig._EPS_N_SYM,
        ed_gamma_label="", title="3x3, DFS units")
    assert _png_ok(p1)
    p2 = fig.plot_density_energy_overview(
        rows, hd, tmp_path / "ov_dfs.png", "run_x",
        ed_summary=ch_eps["combined"], parity_nn_key="density_eps_l1",
        parity_pbe_key="density_eps_l1_pbe",
        parity_unit_label=fig._EPS_N_SYM, title="Overview, DFS units")
    assert _png_ok(p2)
    p3 = fig.plot_density_parity_by_channel(
        rows, hd, tmp_path / "parity_dfs.png", "run_x",
        nn_key="density_eps_l1", pbe_key="density_eps_l1_pbe",
        unit_label=fig._EPS_N_SYM, title="Parity by channel, DFS units")
    assert _png_ok(p3)


def test_density_parity_panel_square_limits():
    """Asymmetric data (an NN outlier stretching one axis) must still give
    SQUARE shared limits -- cloud centered, y=x corner-to-corner -- instead
    of independently autoscaled axes."""
    rows = [
        {"molecule": "a", "arch": "deep", "density_rmse": 1e-3},
        {"molecule": "b", "arch": "deep", "density_rmse": 9e-2},  # outlier
        {"molecule": "c", "arch": "deep", "density_rmse": 2e-3},
    ]
    pbe = {"a": 2e-3, "b": 3e-3, "c": 2.5e-3}
    f1, ax = fig.plt.subplots()
    fig._density_parity_panel(ax, rows, pbe)
    assert ax.get_xlim() == ax.get_ylim()
    lo, hi = ax.get_xlim()
    # the exact padded envelope of the pooled pairs
    assert lo == pytest.approx(0.8 * 1e-3)
    assert hi == pytest.approx(1.25 * 9e-2)
    fig.plt.close(f1)
    # a zero-valued error (unrenderable on log axes) must not poison the
    # lower limit: limits stay square and strictly positive, from the
    # positive values alone
    rows0 = rows + [{"molecule": "z", "arch": "deep", "density_rmse": 0.0}]
    pbe0 = dict(pbe, z=2e-3)
    f2, ax2 = fig.plt.subplots()
    fig._density_parity_panel(ax2, rows0, pbe0)
    assert ax2.get_xlim() == ax2.get_ylim()
    assert ax2.get_xlim()[0] == pytest.approx(0.8 * 1e-3)
    fig.plt.close(f2)


def test_gamma_stamp_branches():
    """The shared in-panel gamma stamp: fixed summaries state the external
    value (plus its source when the summary carries one), self-calibrated
    ones the E_PBE/D_PBE construction; placed top-right."""
    fixed_s = fig.combined_ed_fixed_gamma({("deep", 1): 8.0}, 10.0,
                                          {("deep", 1): 0.004}, 0.005,
                                          1084.87)
    self_s = fig.combined_ed_by_cell({("deep", 1): 8.0}, 10.0,
                                     {("deep", 1): 4e-4}, 5e-4)
    f1, ax1 = fig.plt.subplots()
    fig._gamma_stamp(ax1, fixed_s)
    t1 = " ".join(t.get_text() for t in ax1.texts)
    assert "fixed, external" in t1 and "1084.87" in t1
    assert "self-calibrated" not in t1
    obj = ax1.texts[-1]
    assert obj.get_position() == (0.98, 0.98)
    assert obj.get_ha() == "right" and obj.get_va() == "top"
    fig.plt.close(f1)
    f2, ax2 = fig.plt.subplots()
    fig._gamma_stamp(ax2, self_s)
    t2 = " ".join(t.get_text() for t in ax2.texts)
    assert "(self-calibrated)" in t2 and "fixed, external" not in t2
    fig.plt.close(f2)
    # a sourced fixed summary names its gamma's origin -- on the shared
    # stamp AND on the rich decomposition panel's inline stamp (single
    # text source, no fork)
    src_s = fig.combined_ed_fixed_gamma({("deep", 1): 8.0}, 10.0,
                                        {("deep", 1): 0.004}, 0.005,
                                        1158.34, gamma_source="own-axes fit")
    f3, ax3 = fig.plt.subplots()
    fig._gamma_stamp(ax3, src_s)
    t3 = " ".join(t.get_text() for t in ax3.texts)
    assert "fixed: own-axes fit" in t3 and "1158.34" in t3
    assert "fixed, external" not in t3
    fig.plt.close(f3)
    f4, ax4 = fig.plt.subplots()
    fig._ed_decomposition_rich_panel(ax4, src_s)
    t4 = " ".join(t.get_text() for t in ax4.texts)
    assert "fixed: own-axes fit" in t4 and "fixed, external" not in t4
    fig.plt.close(f4)


def test_density_parity_panel_external_limits():
    """An externally supplied (lo, hi) is applied exactly and squarely --
    the 3x3 row-share mechanism."""
    rows = [{"molecule": "a", "arch": "deep", "density_rmse": 1e-3}]
    pbe = {"a": 2e-3}
    f1, ax = fig.plt.subplots()
    fig._density_parity_panel(ax, rows, pbe, limits=(1e-4, 1e-1))
    assert ax.get_xlim() == ax.get_ylim() == (1e-4, 1e-1)
    fig.plt.close(f1)


def test_parity_by_channel_shares_limits_and_3x3_has_no_parity(
        tmp_path, monkeypatch):
    """The standalone parity figure passes ONE row-wide envelope to all
    three channel panels (directly comparable frames), while the all-bars
    3x3 no longer draws any parity panel. The fixture carries a bh76-only
    outlier species so per-channel envelopes differ from the pooled one:
    identical per-channel frames cannot fake the share."""
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    # HOh_ts is bh76-only (see the _species_pools test); its large errors
    # stretch the pooled envelope beyond the w411 channel's own data
    for sd in (run / "checkpoints").glob("spec_*"):
        pm = sd / "eval_holdout" / "per_molecule.json"
        if not pm.is_file():
            continue
        rows_pm = json.loads(pm.read_text())
        rows_pm.append({
            "molecule": "HOh_ts", "density_rmse": 5e-3, "density_l1": 1e-4,
            "density_rmse_pbe": 6e-3, "density_l1_pbe": 2e-4,
            "density_eps_l1": 4e-3, "density_eps_l1_pbe": 5e-3,
            "n_electrons": 10.0, "grid_weight_sum": 100.0,
            "ref_density_method": "ccsd", "from_training_subset": False})
        pm.write_text(json.dumps(rows_pm))
    rows = fig.collect_holdout_reaction_rows(run)
    hd = fig.collect_holdout_density_rows(run)
    seen = []
    real = fig._density_parity_panel

    def spy(ax, density_rows, pbe_mol, **kw):
        seen.append(kw.get("limits"))
        return real(ax, density_rows, pbe_mol, **kw)

    monkeypatch.setattr(fig, "_density_parity_panel", spy)
    fig.plot_density_parity_by_channel(rows, hd, tmp_path / "par.png",
                                       "run_x")
    assert len(seen) == 3
    assert all(lim is not None and lim == seen[0] for lim in seen)
    # the shared envelope is the POOLED positive envelope: lo from the HO
    # NN RMSE (2e-4), hi from the bh76-only HOh_ts PBE value (6e-3) -- a
    # per-channel w411 frame would top out at 1.25*8e-4 instead
    assert seen[0][0] == pytest.approx(0.8 * 2e-4)
    assert seen[0][1] == pytest.approx(1.25 * 6e-3)
    # the all-bars 3x3 never calls the parity panel body
    seen.clear()
    fig.plot_density_energy_3x3(rows, hd, tmp_path / "x3.png", "run_x")
    assert seen == []


def test_3x3_density_bar_row_values(tmp_path, monkeypatch):
    """Row 2 of the all-bars 3x3: per-channel cell-mean density-error bars
    on the selected channel, PBE dashed at the channel's deduplicated
    anchor -- values pinned against the fixture. A bh76-only outlier
    species makes the three channels' values DIFFER, so a mutant feeding
    the pooled rows to every channel cannot pass."""
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    for sd in (run / "checkpoints").glob("spec_*"):
        pm = sd / "eval_holdout" / "per_molecule.json"
        if not pm.is_file():
            continue
        rows_pm = json.loads(pm.read_text())
        rows_pm.append({
            "molecule": "HOh_ts", "density_rmse": 5e-3, "density_l1": 1e-4,
            "density_rmse_pbe": 6e-3, "density_l1_pbe": 2e-4,
            "density_eps_l1": 4e-3, "density_eps_l1_pbe": 5e-3,
            "n_electrons": 10.0, "grid_weight_sum": 100.0,
            "ref_density_method": "ccsd", "from_training_subset": False})
        pm.write_text(json.dumps(rows_pm))
    rows = fig.collect_holdout_reaction_rows(run)
    hd = fig.collect_holdout_density_rows(run)
    calls = []
    real = fig._grouped_arch_bars

    def spy(ax, metric, archs, subsets, **kw):
        calls.append((dict(metric), kw))
        return real(ax, metric, archs, subsets, **kw)

    monkeypatch.setattr(fig, "_grouped_arch_bars", spy)
    fig.plot_density_energy_3x3(
        rows, hd, tmp_path / "x3.png", "run_x",
        density_nn_key="density_eps_l1",
        density_pbe_key="density_eps_l1_pbe",
        density_unit_label=fig._EPS_N_SYM)
    dens = {}
    for m, kw in calls:
        t = kw.get("title", "")
        if fig._EPS_N_SYM in t and "cell mean" in t:
            for ch in ("BH76", "W4-11", "combined"):
                if f"{ch} species" in t:
                    dens[ch] = (m, kw)
    assert set(dens) == {"BH76", "W4-11", "combined"}
    # channel-restricted values: HO (eps 2.5e-4) is in BOTH pools, the
    # HOh_ts outlier (4e-3) is bh76-only -> bh76/combined cell means are
    # mean(2.5e-4, 4e-3), w411 stays 2.5e-4; anchors mean(7e-4, 5e-3) vs
    # 7e-4. A pooled-rows mutant would show the combined values everywhere.
    for ch, want_m, want_pbe in (
            ("BH76", (2.5e-4 + 4e-3) / 2, (7e-4 + 5e-3) / 2),
            ("W4-11", 2.5e-4, 7e-4),
            ("combined", (2.5e-4 + 4e-3) / 2, (7e-4 + 5e-3) / 2)):
        m, kw = dens[ch]
        assert m, "density bar map must not be empty"
        assert all(v == pytest.approx(want_m) for v in m.values()), ch
        assert kw["pbe_line"] == pytest.approx(want_pbe), ch
    # nine bar panels in total: 3 energy + 3 density + 3 ED
    assert len(calls) == 9


def test_3x3_caveats_define_reduction_and_gamma():
    """Both 3x3 caveats spell out the one-bucket reduction formula on the
    figure; the DFS-units caveat states the published gamma value and its
    source. Two-line form keeps the canvas width bounded."""
    for cav in (fig._3X3_CAVEAT, fig._3X3_DFS_UNITS_CAVEAT):
        assert "56.84*MAD_pool/mean|dE_ref|_pool" in cav
        assert "scaled relative error" in cav
        assert "\n" in cav
    # the twin's caveat defers the plotted value to the in-panel stamp and
    # names both possible sources (fit operative, published fallback)
    assert "own-axes" in fig._3X3_DFS_UNITS_CAVEAT
    assert "1084.87" in fig._3X3_DFS_UNITS_CAVEAT
    assert "published" in fig._3X3_DFS_UNITS_CAVEAT
    assert "1084.87" not in fig._3X3_CAVEAT   # original stays self-calibrated
    # both caveats point at the standalone parity figure that replaced the
    # parity row
    assert "ablation_density_parity_by_channel.png" in fig._3X3_CAVEAT
    assert ("ablation_density_parity_by_channel_dfs_units.png"
            in fig._3X3_DFS_UNITS_CAVEAT)


def test_dfs_paper_notation_symbols():
    """Figure text carries the DFS paper's symbols: the combined metric is
    the CALLIGRAPHIC ED (ED_{|n|} on the eps-leg figures, per Eq. 21 /
    Table I), the density error is varepsilon_{|n|} (Eq. 20) -- verified
    against the Letter's PDF. Equation NUMBERS remain citations, not
    labels."""
    assert fig._ED_SYM == r"$\mathcal{ED}$"
    assert fig._ED_N_SYM == r"$\mathcal{ED}_{|n|}$"
    assert fig._EPS_N_SYM == r"$\varepsilon_{|n|}$"
    assert r"\int|n - n_{ref}|" in fig._EPS_N_EQ
    # the eps caveats define the density error by its equation and use the
    # paper's combined-metric symbol
    for cav in (fig._ED_DFS_UNITS_CAVEAT, fig._3X3_DFS_UNITS_CAVEAT,
                fig._HOLDOUT_OVERVIEW_DFS_UNITS_CAVEAT):
        assert fig._EPS_N_SYM in cav
        assert fig._ED_N_SYM in cav
        assert "Eq. 20 eps" not in cav      # the number is not the label
    assert fig._ED_SYM in fig._ED_CAVEAT
    assert fig._ED_SYM in fig._3X3_CAVEAT
    # the ED lines panel's ylabel carries the symbol
    s = fig.combined_ed_by_cell({("deep", 1): 8.0}, 10.0,
                                {("deep", 1): 4e-4}, 5e-4)
    f1, ax = fig.plt.subplots()
    fig._ed_lines_panel(ax, s, "t")
    assert fig._ED_SYM.strip("$") in ax.get_ylabel().replace("$", "")
    fig.plt.close(f1)


def _train_on_ch(run_dir, *spec_names):
    """Add CH to those specs' training molecules, so the relocked species is
    actually in their training set (the boundary only applies to such cells)."""
    for name in spec_names:
        meta = run_dir / "checkpoints" / name / "train_metadata.json"
        if not meta.is_file():
            continue
        md = json.loads(meta.read_text())
        md["molecules"] = sorted(set(md.get("molecules", [])) | {"CH"})
        meta.write_text(json.dumps(md))


def _write_lockfix_manifest(run_dir, pre=("spec_0000",), in_flight=(),
                            post=("spec_0001",)):
    """The manifest hpcjobs/dfs6311_lockfix_swap.py writes into the run dir."""
    (run_dir / "lockfix_swap_manifest.json").write_text(json.dumps({
        "what": "CH/NO training references relocked mid-sweep",
        "swap_time_local": "2026-08-03 15:53:39 EDT",
        "swap_time_epoch": 1785786819,
        "species": {"CH": {"live_lock_after": 3e-05},
                    "NO": {"live_lock_after": 3e-05}},
        "spec_partition_at_swap": {"complete": list(pre),
                                   "in_flight": list(in_flight),
                                   "not_started": list(post)},
    }))


def test_lockfix_boundary_absent_is_empty(tmp_path):
    """A run with no mid-run swap has no boundary and no disclosure -- those
    runs' figures must be unchanged."""
    run = _make_run_dir(tmp_path)
    assert fig.lockfix_boundary(run) == {}
    assert fig.lockfix_note(run) == ""


def test_lockfix_boundary_parses_partition(tmp_path):
    """Spec names -> indices, with in-flight counted on the OLD side (those
    tasks loaded their references before the swap)."""
    run = _make_run_dir(tmp_path)
    _write_lockfix_manifest(run, pre=("spec_0000", "spec_0002"),
                            in_flight=("spec_0003",),
                            post=("spec_0004", "spec_0005"))
    b = fig.lockfix_boundary(run)
    # in-flight specs are their OWN class: they trained on the old references
    # but their eval re-reads the new ones, so they are neither side
    assert b["pre"] == {0, 2}
    assert b["mixed"] == {3}
    assert b["post"] == {4, 5}
    assert b["species"] == ["CH", "NO"]
    assert "2026-08-03" in b["swap_time"]


def test_lockfix_note_reports_only_plotted_cells(tmp_path):
    """The disclosure counts the cells actually on the figure, not every spec
    in the manifest, and names the spec range on each side."""
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    # the fixture evaluates specs 0-4; put 0-2 pre-swap and 3+ post
    _train_on_ch(run, "spec_0000", "spec_0001", "spec_0002", "spec_0003",
                 "spec_0004")
    _write_lockfix_manifest(run, pre=("spec_0000", "spec_0001", "spec_0002"),
                            post=("spec_0003", "spec_0004", "spec_0087"))
    msg = fig.lockfix_note(run)
    assert "DENSITY-REFERENCE BOUNDARY" in msg
    assert "CH/NO" in msg
    assert "mix two reference sets" in msg
    # spec_0087 is in the manifest but has no eval -> must not be counted
    assert "0087" not in msg
    # single-sided runs say so instead of claiming a mix
    _write_lockfix_manifest(run, pre=(), post=("spec_0000", "spec_0001",
                                               "spec_0002", "spec_0003"))
    only_post = fig.lockfix_note(run)
    assert "post-swap" in only_post and "relocked references" in only_post
    assert "mix two reference sets" not in only_post


def test_lockfix_note_flags_mid_training_cells_as_uninterpretable(tmp_path):
    """A spec mid-training at the swap trained on the OLD references but its
    eval re-read the NEW ones, so its density numbers belong to neither side
    and the disclosure must say so rather than silently grouping it."""
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    _train_on_ch(run, "spec_0000", "spec_0001", "spec_0002", "spec_0003")
    _write_lockfix_manifest(run, pre=("spec_0000",),
                            in_flight=("spec_0001", "spec_0002"),
                            post=("spec_0003",))
    msg = fig.lockfix_note(run)
    assert "mid-training at the swap" in msg
    assert "NOT interpretable" in msg
    assert "spec 0001-0002" in msg
    # and they are not counted into either side's cell tally
    assert "1 affected cell(s) pre-swap" in msg and "1 post-swap" in msg


def test_density_figures_stamp_the_lockfix_boundary(tmp_path, capsys):
    """The builder stamps the boundary on the density figures' note band (and
    the console), so no density comparison can silently span it."""
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    _train_on_ch(run, "spec_0000", "spec_0001", "spec_0002", "spec_0003")
    _write_lockfix_manifest(run, pre=("spec_0000", "spec_0001"),
                            post=("spec_0002", "spec_0003"))
    seen = []

    def spy(rows, out_path, run_id, **kw):
        seen.append(kw.get("note", ""))
        Path(out_path).write_bytes(b"x" * 4096)
        return Path(out_path)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(fig, "plot_holdout_density_ccsd", spy)
    try:
        fig.build_density_energy_figures(run, tmp_path / "f")
    finally:
        monkey.undo()
    assert seen and "DENSITY-REFERENCE BOUNDARY" in seen[0]
    assert "DENSITY-REFERENCE BOUNDARY" in capsys.readouterr().out


def test_lockfix_cell_classes_only_marks_cells_training_on_swapped_species():
    """Glyphs mark only cells whose TRAINING SET holds a relocked species; a
    cell that never trains on CH/NO saw identical references either side and
    must carry no marker."""
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    run = _make_run_dir(tmp)
    _add_holdout_density(run)
    # spec_0000 trains on HO only (fixture), spec_0003 gets CH added
    meta3 = run / "checkpoints" / "spec_0003" / "train_metadata.json"
    md = json.loads(meta3.read_text())
    md["molecules"] = list(md.get("molecules", [])) + ["CH"]
    meta3.write_text(json.dumps(md))
    _write_lockfix_manifest(run, pre=("spec_0000",), in_flight=("spec_0001",),
                            post=("spec_0002", "spec_0003"))
    cls = fig.lockfix_cell_classes(run)
    manifest_cells = fig.ccp._read_manifest_cells(run)
    ch_cell = (manifest_cells[3]["arch"], manifest_cells[3]["subset_size"])
    # only the CH-bearing post-swap cell is marked relocked
    assert cls["relocked"] == {ch_cell}
    # spec_0002 is post-swap but trains on no swapped species -> unmarked
    no_ch = (manifest_cells[2]["arch"], manifest_cells[2]["subset_size"])
    assert no_ch not in cls["relocked"] and no_ch not in cls["mixed"]


def test_grouped_bars_draw_reference_provenance_glyphs():
    """The bar panel draws a star on relocked cells and a hatched X on cells
    whose references changed mid-training; unmarked runs are unchanged."""
    metric = {("deep", 1): 1.0, ("deep", 2): 2.0, ("deep", 3): 3.0}
    f0, ax0 = fig.plt.subplots()
    fig._grouped_arch_bars(ax0, metric, ["deep"], [1, 2, 3], title="t")
    base_collections = len(ax0.collections)
    base_labels = set(ax0.get_legend_handles_labels()[1])
    fig.plt.close(f0)
    f1, ax1 = fig.plt.subplots()
    fig._grouped_arch_bars(ax1, metric, ["deep"], [1, 2, 3], title="t",
                           relocked_cells={("deep", 3)},
                           mixed_cells={("deep", 2)})
    labels = set(ax1.get_legend_handles_labels()[1])
    assert "relocked refs" in labels
    assert any("not interpretable" in lb for lb in labels)
    assert len(ax1.collections) > base_collections
    assert not (base_labels - labels)      # existing legend entries kept
    hatched = [p for p in ax1.patches if p.get_hatch()]
    assert len(hatched) == 1               # only the mid-training cell
    fig.plt.close(f1)


def test_pbe_anchor_coverage_warning_key_params():
    rows = [
        {"molecule": "m1", "density_rmse": 1e-4, "density_rmse_pbe": 2e-4,
         "density_eps_l1": 1e-3, "density_eps_l1_pbe": 2e-3},
        {"molecule": "m2", "density_rmse": 1e-4, "density_rmse_pbe": 2e-4,
         "density_eps_l1": None, "density_eps_l1_pbe": 2e-3},
    ]
    assert fig._pbe_anchor_coverage_warning(rows) == ""      # RMSE aligned
    w = fig._pbe_anchor_coverage_warning(rows, nn_key="density_eps_l1",
                                         pbe_key="density_eps_l1_pbe")
    assert "m2" in w and "anchor-only" in w


def test_density_cell_coverage_warning_key_param():
    """Within-cell species homogeneity must be checkable on the eps channel
    independently of the RMSE channel (a per-species partial backfill leaves
    RMSE aligned while eps diverges)."""
    rows = [
        {"arch": "deep", "subset_size": 1, "molecule": "m1",
         "density_rmse": 1e-4, "density_eps_l1": 1e-3},
        {"arch": "deep", "subset_size": 1, "molecule": "m2",
         "density_rmse": 1e-4, "density_eps_l1": None},
        {"arch": "deep", "subset_size": 2, "molecule": "m1",
         "density_rmse": 1e-4, "density_eps_l1": 1e-3},
        {"arch": "deep", "subset_size": 2, "molecule": "m2",
         "density_rmse": 1e-4, "density_eps_l1": 1e-3},
    ]
    assert fig._density_cell_coverage_warning(rows) == ""    # RMSE uniform
    w = fig._density_cell_coverage_warning(rows, key="density_eps_l1")
    assert "deep/ss1" in w and "n=1" in w


def test_plot_ed_decomposition_renders(tmp_path):
    from matplotlib.collections import PolyCollection
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    rows = fig.collect_holdout_reaction_rows(run)
    hd = fig.collect_holdout_density_rows(run)
    tab = fig.load_pbe_density_table(run)
    wt = fig.combined_ed_by_cell(
        fig.wtmad2_by_arch_subset(rows), fig.wtmad2_pbe_baseline(rows),
        fig.holdout_density_by_arch_subset(hd),
        fig.pbe_density_baseline(hd, tab))
    p1 = fig.plot_ed_decomposition(wt, tmp_path / "iso.png", "run_x")
    assert _png_ok(p1)
    # structural check on a bare axes: contour family + shading + trajectories
    f1, ax = fig.plt.subplots()
    fig._ed_decomposition_rich_panel(ax, wt)
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    assert len(ax.lines) >= 5           # y=x + several iso-ED contour levels
    assert any(isinstance(c, PolyCollection) for c in ax.collections)
    # the gamma stamp must DEFINE gamma, not just print its value
    assert any("E$_{\\rm PBE}$/D$_{\\rm PBE}$" in t.get_text()
               for t in ax.texts)
    fig.plt.close(f1)
    f2, ax2 = fig.plt.subplots()
    fig._ed_lines_panel(ax2, wt, "t")
    assert any("E$_{\\rm PBE}$/D$_{\\rm PBE}$" in t.get_text()
               for t in ax2.texts)
    fig.plt.close(f2)


def test_plot_density_energy_overview_renders(tmp_path):
    run = _make_run_dir(tmp_path)
    _add_holdout_density(run)
    rows = fig.collect_holdout_reaction_rows(run)
    hd = fig.collect_holdout_density_rows(run)
    tab = fig.load_pbe_density_table(run)
    d_cells = fig.holdout_density_by_arch_subset(hd)
    d_pbe = fig.pbe_density_baseline(hd, tab)
    wt = fig.combined_ed_by_cell(fig.wtmad2_by_arch_subset(rows),
                                 fig.wtmad2_pbe_baseline(rows), d_cells, d_pbe)
    p1 = fig.plot_density_energy_overview(rows, hd, tmp_path / "ov.png",
                                          "run_x", pbe_table=tab,
                                          ed_summary=wt)
    assert _png_ok(p1)
    # ED anchors unavailable -> panel F placeholder, still a valid figure
    p2 = fig.plot_density_energy_overview(rows, hd, tmp_path / "ov2.png",
                                          "run_x", pbe_table=tab,
                                          ed_summary=None)
    assert _png_ok(p2)


# ---------------------------------------------------------------------------
# Jacob's-ladder rung summary + rung ordering + beats-PBE + SCAN baseline
# ---------------------------------------------------------------------------

def _make_multirung_rows():
    """Synthetic held-out reaction rows across all four Jacob's-ladder rungs
    (GGA / meta-GGA / rung-3.5 / combined), BH76 + W4-11, two subset sizes so
    best-subset selection is exercised."""
    archs = ["deep", "deep_mgga_3x16", "deep_rung35_3x16", "deep_rung35_mgga_3x16"]
    rows = []
    for i, a in enumerate(archs):
        for ss in (1, 3):
            rows.append({"arch": a, "subset_size": ss, "pool": "bh76",
                         "name": f"bh76_{a}_{ss}",
                         "reaction_energy_ref_kcalmol": 17.7,
                         "abs_error_nn_kcalmol": 20.0 - 3.0 * i - ss,
                         "abs_error_pbe_kcalmol": 14.0})
            rows.append({"arch": a, "subset_size": ss, "pool": "w411",
                         "name": f"w411_{a}_{ss}",
                         "reaction_energy_ref_kcalmol": 120.0,
                         "abs_error_nn_kcalmol": 30.0 - 2.0 * i - ss,
                         "abs_error_pbe_kcalmol": 16.0})
    return rows


def test_plot_rung_summary_renders_multirung(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    rows = _make_multirung_rows()
    # all four rungs present among the synthetic archs, in ladder order
    by_r = fig.arch_style.by_rung(fig._archs_present(rows))
    assert list(by_r) == list(fig.arch_style.RUNG_ORDER)
    out = fig.plot_rung_summary(
        rows, tmp_path / "rung.png", "run_x",
        pbe_baseline={"bh76": 14.0, "w411": 16.0, "combined": 15.0},
        scan_baseline={"bh76": float("nan"), "w411": float("nan"),
                       "combined": float("nan")})
    assert _png_ok(out)          # PBE line drawn, SCAN line omitted (NaN)
    # a finite SCAN baseline still renders (SCAN reference line added)
    out2 = fig.plot_rung_summary(
        rows, tmp_path / "rung2.png", "run_x",
        pbe_baseline={"combined": 15.0}, scan_baseline={"combined": 9.0})
    assert _png_ok(out2)
    # no baselines at all -> still renders (both reference lines omitted)
    assert _png_ok(fig.plot_rung_summary(rows, tmp_path / "rung3.png", "run_x"))


def test_energy_and_heatmap_arch_axes_are_rung_sorted():
    rows = _make_multirung_rows()
    # ARCH_ORDER order here is NOT rung order (r3.5 precedes mGGA in ARCH_ORDER),
    # so a passing rung-rank check proves these axes actually rung-sort.
    order = fig._energy_arch_axis(rows)
    assert order == fig.arch_style.sort_by_rung(order)          # idempotent
    assert [fig.arch_style.rung_rank(a) for a in order] == sorted(
        fig.arch_style.rung_rank(a) for a in order)             # GGA..combined
    assert order.index("deep") < order.index("deep_mgga_3x16")  # base before mGGA
    assert order.index("deep_mgga_3x16") < order.index("deep_rung35_3x16")
    hx = fig._heatmap_arch_axis(rows, [])
    assert [fig.arch_style.rung_rank(a) for a in hx] == sorted(
        fig.arch_style.rung_rank(a) for a in hx)


def test_beats_pbe_marks_flags_below_line_cells():
    m = fig._beats_pbe_marks
    assert m([0, 1, 2], [5.0, 20.0, 3.0], 10.0) == [(0.0, 5.0), (2.0, 3.0)]
    assert m([0, 1], [5.0, 8.0], float("nan")) == []           # no PBE -> no marks
    assert m([0, 1], [float("nan"), 4.0], 10.0) == [(1.0, 4.0)]  # NaN bar skipped
    assert m([], [], 10.0) == []
    assert m([0], [10.0], 10.0) == []                          # equal is not below


def test_plot_mae_by_arch_marks_below_pbe_cell(tmp_path):
    # a synthetic arch whose held-out reaction MAE (~5) sits well below the PBE
    # line (~14) must render with the beats-PBE marker layer (no crash).
    import matplotlib
    matplotlib.use("Agg")
    rows = _make_multirung_rows()
    out = fig.plot_mae_by_arch(rows, [], tmp_path / "mae.png", "run_x",
                               scan_baseline={"combined": 9.0})
    assert _png_ok(out)
    # at least one cell is below its own PBE-vs-benchmark line -> a mark exists
    pbe = fig._mae([r["abs_error_pbe_kcalmol"] for r in rows])
    mp = fig.reaction_mae_by_arch_subset(rows)
    assert any(v < pbe for v in mp.values())


def test_plot_energy_wtmad_mae_renders_with_scan(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    rows = _make_multirung_rows()
    out = fig.plot_energy_wtmad_mae(rows, tmp_path / "wt_scan.png", "run_x",
                                    scan_baseline={"combined": 9.0})
    assert _png_ok(out)


def test_scan_pool_baseline_via_energies_seam(tmp_path):
    """Full-pool SCAN MAE from an injected {name: E_scan} map + reaction pool
    (test seams). Same arithmetic as the PBE-baseline seam test."""
    fake_rxns = [
        {"name": "rb", "source_pool": "bh76", "reactants": ["a"], "products": ["b"],
         "coeffs": [-1.0, 1.0], "reaction_energy_ref": 10.0},   # de=12 -> |err|=2
        {"name": "rw", "source_pool": "w411", "reactants": ["a"], "products": ["c"],
         "coeffs": [-1.0, 1.0], "reaction_energy_ref": 100.0},  # de=90 -> |err|=10
    ]
    e_a = -1.0
    e_b = e_a + 12.0 / _KCAL_PER_HA
    e_c = e_a + 90.0 / _KCAL_PER_HA
    base = fig.scan_pool_baseline(tmp_path, _loader=lambda: ({}, fake_rxns),
                                  _energies={"a": e_a, "b": e_b, "c": e_c})
    assert base["bh76"] == pytest.approx(2.0, abs=1e-6)
    assert base["w411"] == pytest.approx(10.0, abs=1e-6)
    assert base["combined"] == pytest.approx((2.0 + 10.0) / 2, abs=1e-6)


def test_scan_pool_baseline_missing_cache_is_all_nan(tmp_path):
    import math
    # no scan_pool_energies_*.json anywhere -> all-NaN AND the pool loader is
    # never called (no xcquinox import for a cache-less run).
    called = []
    base = fig.scan_pool_baseline(
        tmp_path, _loader=lambda: (called.append(1), ({}, []))[1])
    assert math.isnan(base["bh76"]) and math.isnan(base["w411"])
    assert math.isnan(base["combined"])
    assert called == []          # short-circuits before the pool loader


def test_scan_energies_reads_cache_json(tmp_path):
    (tmp_path / "resolved_config.yaml").write_text("basis: def2-svp\n")
    (tmp_path / fig._scan_cache_name("def2-svp")).write_text(
        json.dumps({"a": -1.0, "b": -0.9, "bad": "x"}))
    # basis auto-resolved from resolved_config.yaml; non-numeric value dropped
    assert fig._scan_energies(tmp_path) == {"a": -1.0, "b": -0.9}
    # explicit basis + cache_dir also resolves the same filename
    assert fig._scan_energies(tmp_path, basis="def2-svp",
                              cache_dir=tmp_path) == {"a": -1.0, "b": -0.9}
    # +DF label maps to the same (undecorated) cache filename as precompute writes
    assert fig._scan_cache_name("def2-tzvpd+DF") == "scan_pool_energies_def2-tzvpd.json"


def test_scan_pool_baseline_reads_disk_cache(tmp_path):
    (tmp_path / "resolved_config.yaml").write_text("basis: def2-svp\n")
    e_a = -1.0
    e_b = e_a + 12.0 / _KCAL_PER_HA
    (tmp_path / fig._scan_cache_name("def2-svp")).write_text(
        json.dumps({"a": e_a, "b": e_b}))
    fake = [{"name": "rb", "source_pool": "bh76", "reactants": ["a"],
             "products": ["b"], "coeffs": [-1.0, 1.0], "reaction_energy_ref": 10.0}]
    base = fig.scan_pool_baseline(tmp_path, _loader=lambda: ({}, fake))
    assert base["bh76"] == pytest.approx(2.0, abs=1e-6)


def test_provenance_footer_appends_scan_when_present():
    s = fig.provenance_footer({"bh76": 11.8, "w411": 15.9, "combined": 14.5},
                              {"bh76": 8.0, "w411": 6.0, "combined": 7.0})
    assert "PBE (full pool): BH76 11.80" in s
    assert "SCAN (full pool): BH76 8.00 / W4-11 6.00 / combined 7.00." in s
    # absent/NaN SCAN -> byte-identical to the PBE-only footer (backward compat)
    pbe_only = fig.provenance_footer({"bh76": 11.8, "w411": 15.9, "combined": 14.5})
    assert "SCAN" not in pbe_only
    assert fig.provenance_footer(
        {"bh76": 11.8, "w411": 15.9, "combined": 14.5},
        {"bh76": float("nan"), "w411": None, "combined": float("nan")}) == pbe_only


# ---------------------------------------------------------------------------
# enhancement_factors -- grid geometry + physics references (fast)
# ---------------------------------------------------------------------------

def test_s_to_sigma_round_trips():
    import numpy as np
    rho = np.full(5, 0.3)
    s = np.array([0.1, 0.5, 1.0, 2.0, 3.0])
    sigma = ef.s_to_sigma(rho, s)
    k_F = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    s_back = np.sqrt(sigma) / (2.0 * k_F * rho)
    assert np.allclose(s_back, s, rtol=1e-10)


def test_rs_to_rho_matches_definition():
    import numpy as np
    rs = 2.0
    rho = ef.rs_to_rho(rs)
    # rs = (3/(4 pi rho))^(1/3)
    assert np.isclose((3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0), rs)


def test_pbe_fx_curve_monotone_and_bounded():
    import numpy as np
    s = np.linspace(0, 5, 50)
    fx = ef.pbe_fx_curve(s)
    assert np.isclose(fx[0], 1.0)             # F_x(0) = 1 (UEG limit)
    assert np.all(np.diff(fx) >= -1e-12)      # monotone increasing in s
    assert np.all(fx <= 1.804 + 1e-9)         # Lieb-Oxford ceiling


def test_pbe_fc_curve_shape_and_finite():
    import numpy as np
    s = np.linspace(1e-3, 3, 40)
    fc = ef.pbe_fc_curve(s, rs=2.0)
    if fc is None:                            # pyscf/libxc unavailable
        pytest.skip("libxc not available")
    assert fc.shape == s.shape
    assert np.all(np.isfinite(fc))
    # PBE correlation is suppressed by the gradient: F_c decreases with s.
    assert fc[0] >= fc[-1]


def test_representative_specs_picks_largest_trained(tmp_path):
    run = _make_run_dir(tmp_path)
    reps = ef.representative_specs(run)
    # deep + deep_notransform both have ss=1,3 trained -> idx of ss=3.
    # deep_attn ss=1 (idx 4) is trained-but-uneval'd but still has model.eqx.
    assert reps["deep"] == 1          # spec_0001 = deep, ss=3
    assert reps["deep_notransform"] == 3
    assert reps["deep_attn"] == 4     # only ss=1 trained for attn


# ---------------------------------------------------------------------------
# Slow: deserialise + forward a real checkpoint from the pulled run
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _REAL_RUN.is_dir(),
                    reason="pulled ablation run not present")
def test_real_checkpoint_fx_curve_finite():
    import numpy as np
    reps = ef.representative_specs(_REAL_RUN)
    assert reps, "no trained specs discovered in the real run"
    arch = "deep_notransform" if "deep_notransform" in reps else sorted(reps)[0]
    _spec, model = ef.load_trained_model(_REAL_RUN, reps[arch])
    s = np.linspace(1e-3, 3, 32)
    fx = ef.model_fx_curve(model, s)
    assert fx.shape == s.shape
    assert np.all(np.isfinite(fx))
    assert np.isclose(fx[0], 1.0, atol=0.2)   # near UEG limit at s->0


def test_arch_order_includes_3x16_twins_sharing_sibling_colors():
    # 2026-06-20 (WS7): the dfs_step7 v3 sweep uses the depth-3/width-16 twins;
    # the suite must RECOGNIZE them (it fails loud on unknown archs) and color
    # each twin like its 4x32 sibling (same architecture, reduced capacity), so
    # tab10's 10-color cap is never exceeded.
    base = ["deep", "deep_attn", "deep_cusp", "deep_dm", "deep_combined",
            "deep_combined_attn", "deep_notransform", "deep_notransform_attn"]
    for a in base:
        twin = f"{a}_3x16"
        assert twin in fig.ARCH_ORDER, f"{twin} missing from ARCH_ORDER"
        assert fig.ARCH_COLOR[twin] == fig.ARCH_COLOR[a]   # twin shares sibling color
    assert len({fig.ARCH_COLOR[a] for a in base}) == 8     # base-8 stay distinct


def test_arch_order_covers_v3_full25_sweep_archs():
    # 2026-06-29: the rung-3.5 sweep swap (deep_combined -> deep_rung35) MUST keep
    # every arch the v3/full25 YAMLs sweep inside ARCH_ORDER -- else
    # build_bh76w411_suite raises ValueError on figure regen once rung-3.5 eval
    # data is pulled. Pins the deep_rung35* additions + their colors + the
    # _arch_input_forms descriptor labels so the swap can't silently break figures.
    import yaml
    root = Path(__file__).resolve().parents[2]
    for fn in ("dfs_step7.svp_grid2_v3.yaml", "dfs_step7.svp_grid2_v3_full25.yaml",
               "dfs_step7.dfs6311_grid3_v3.yaml"):
        cfg = yaml.safe_load((root / "hpcjobs" / "configs" / fn).read_text())
        for a in cfg["sweep"]["arch"]:
            assert a in fig.ARCH_ORDER, f"{fn}: swept arch {a!r} not in ARCH_ORDER"
            assert fig.ARCH_COLOR.get(a) not in (None, "#333333"), \
                f"{fn}: {a!r} has no distinct color (fell back to gray)"
    for a in ("deep_rung35_3x16", "deep_rung35_attn_3x16", "deep_rung35only_3x16"):
        assert a in fig.ARCH_ORDER, f"{a} missing from ARCH_ORDER"
        assert fig.ARCH_COLOR.get(a) not in (None, "#333333"), f"{a} has no color"
    # 2026-07-02: the DFS-faithful meta-GGA archs (dfs6311 sweep) must likewise be
    # in ARCH_ORDER with distinct colors + resolvable descriptor labels.
    for a in ("deep_mgga_3x16", "deep_mgga_attn_3x16", "deep_rung35_mgga_3x16"):
        assert a in fig.ARCH_ORDER, f"{a} missing from ARCH_ORDER"
        assert fig.ARCH_COLOR.get(a) not in (None, "#333333"), f"{a} has no color"
    mgga_forms = fig._arch_input_forms(("deep_mgga_3x16", "deep_rung35_mgga_3x16"))
    assert "x_10" in mgga_forms["deep_mgga_3x16"]["fx"], "metagga label x_10 missing"
    assert all(lbl in mgga_forms["deep_rung35_mgga_3x16"]["fx"]
               for lbl in ("x_4", "x_8", "x_9", "x_10")), "combined mgga labels missing"
    # _arch_input_forms must resolve the rung-3.5 descriptor labels (no KeyError)
    forms = fig._arch_input_forms(("deep_rung35_3x16", "deep_rung35only_3x16"))
    assert all(lbl in forms["deep_rung35_3x16"]["fx"] for lbl in ("x_4", "x_8", "x_9")), \
        "deep_rung35 X-net inputs should carry cusp (x_4) + rung-3.5 (x_8,x_9) labels"


# ---------------------------------------------------------------------------
# Unequal training depth: a partial sweep fills the arch x subset_size grid
# column by column, so an arch that entered late carries only its smallest
# subsets. The figures that aggregate OVER subset_size must mark that, or the
# coverage gap reads as an architecture result.
# ---------------------------------------------------------------------------

def _make_uneven_rung_rows():
    """Held-out rows mirroring the real dfs6311 sweep at 32/88 cells: one GGA
    arch at the full depth, a second GGA arch stopped part-way, and the single
    meta-GGA arch present only at the smallest subset."""
    plan = {"deep_3x16": (1, 26),
            "deep_cusp_3x16": (1, 15),
            "deep_mgga_3x16": (1,)}
    rows = []
    for i, (arch, sizes) in enumerate(plan.items()):
        for ss in sizes:
            for pool, err in (("bh76", 20.0 - 3.0 * i), ("w411", 30.0 - 2.0 * i)):
                rows.append({"arch": arch, "subset_size": ss, "pool": pool,
                             "name": f"{pool}_{arch}_{ss}",
                             "reaction_energy_ref_kcalmol": 17.7,
                             "abs_error_nn_kcalmol": err,
                             "abs_error_pbe_kcalmol": 14.0})
    return rows


@contextlib.contextmanager
def _captured_figures():
    """Keep the figures the plot builders would close so their artists can be
    inspected, then release them on exit.

    A context manager rather than a monkeypatch helper for two reasons: the
    real ``plt.close`` must come back before anything tries to close a figure
    (calling the stand-in appends to the capture list instead of closing, which
    turns a ``for f in seen: plt.close(f)`` loop into an infinite one), and the
    figures must be released even when an assertion fails, or every test leaks
    one into the session.
    """
    seen = []
    real_close = fig.plt.close
    fig.plt.close = lambda f=None: seen.append(f) if f is not None else None
    try:
        yield seen
    finally:
        fig.plt.close = real_close
        real_close("all")


def _bar_rects(ax):
    """The bar Rectangles of ``ax``, ordered by x centre.

    Read off ``ax.containers`` (the BarContainers ``ax.bar`` registers), NOT
    ``ax.patches``: on matplotlib >= 3.8 ``axvspan`` also returns a Rectangle,
    so the rung background bands are indistinguishable from bars by type.
    """
    rects = [p for cont in ax.containers for p in cont.patches]
    return sorted(rects, key=lambda p: p.get_x())


def test_subset_coverage_and_shallow_archs():
    rows = _make_uneven_rung_rows()
    assert fig._subset_coverage(rows) == {"deep_3x16": (1, 26),
                                          "deep_cusp_3x16": (1, 15),
                                          "deep_mgga_3x16": (1, 1)}
    shallow, deepest = fig._shallow_archs(rows)
    assert deepest == 26
    assert shallow == {"deep_cusp_3x16", "deep_mgga_3x16"}
    # A level grid marks nothing -- this is what keeps complete runs unchanged.
    level, level_deep = fig._shallow_archs(_make_multirung_rows())
    assert (level, level_deep) == (set(), 3)
    assert fig._subset_coverage([]) == {}
    assert fig._shallow_archs([]) == (set(), 0)


def test_shallow_rungs_judged_on_the_rungs_deepest_arch():
    """GGA holds both a full-depth arch and a stopped one; the rung has still
    been probed at full depth and must NOT be marked. Judging a rung on its
    mean, or on all of its archs, would flag GGA too."""
    rows = _make_uneven_rung_rows()
    shallow, deepest = fig._shallow_rungs(rows)
    assert deepest == 26
    assert shallow == {fig.arch_style.RUNG_MGGA}
    assert fig.arch_style.RUNG_GGA not in shallow
    assert fig._rung_coverage(rows) == {fig.arch_style.RUNG_GGA: (15, 26),
                                        fig.arch_style.RUNG_MGGA: (1, 1)}
    assert fig._shallow_rungs(_make_multirung_rows())[0] == set()


def test_coverage_span_and_caveat_text():
    assert fig._coverage_span((1, 26)) == "1-26"
    assert fig._coverage_span((1, 1)) == "1"
    rows = _make_uneven_rung_rows()
    arch_cav = fig._coverage_caveat(rows)
    assert "deep_mgga_3x16 at subset_size 1" in arch_cav
    assert "deep_cusp_3x16 at subset_size 1-15" in arch_cav
    assert "against 26" in arch_cav and "architecture" in arch_cav
    rung_cav = fig._coverage_caveat(rows, by_rung=True)
    assert fig.arch_style.RUNG_MGGA in rung_cav and "rung" in rung_cav
    # RUNG_GGA ("GGA") is a SUBSTRING of RUNG_MGGA ("meta-GGA"), so a plain
    # `not in` on the text cannot say whether the GGA rung was named. Assert on
    # the entry count and on the classifier itself instead.
    assert rung_cav.count("at subset_size") == 1
    assert fig.arch_style.RUNG_GGA not in fig._shallow_rungs(rows)[0]
    # Level coverage -> no disclosure at all (footers stay as they were).
    assert fig._coverage_caveat(_make_multirung_rows()) == ""
    assert fig._coverage_caveat(_make_multirung_rows(), by_rung=True) == ""


def test_fade_keeps_hue_and_lightens():
    faded = fig._fade("#1f77b4", 0.55)
    raw = fig.matplotlib.colors.to_rgb("#1f77b4")
    assert all(f > r for f, r in zip(faded, raw))       # lighter on every channel
    assert all(f <= 1.0 for f in faded)
    # ordering of the channels survives, so the rung stays identifiable
    assert sorted(range(3), key=lambda i: raw[i]) == sorted(range(3),
                                                            key=lambda i: faded[i])
    assert fig._fade("#1f77b4", 0.0) == pytest.approx(raw)
    assert fig._fade("#1f77b4", 1.0) == pytest.approx((1.0, 1.0, 1.0))


def test_mae_by_arch_hatches_only_the_shallow_archs(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    rows = _make_uneven_rung_rows()
    with _captured_figures() as seen:
        out = fig.plot_mae_by_arch(rows, [], tmp_path / "mba.png", "run_x")
        assert _png_ok(out)
        ax = seen[-1].axes[0]
        archs = [t.get_text() for t in ax.get_xticklabels()]
        # tick labels state the depth each arch's bars aggregate
        assert any("deep_mgga_3x16" in a and "(ss 1)" in a for a in archs)
        assert any("deep_3x16" in a and "(ss 1-26)" in a for a in archs)
        order = fig.arch_style.sort_by_rung(["deep_3x16", "deep_cusp_3x16",
                                             "deep_mgga_3x16"])
        hatched, plain = set(), set()
        for p in _bar_rects(ax):
            idx = int(round(p.get_x() + p.get_width() / 2.0))
            if 0 <= idx < len(order):
                (hatched if p.get_hatch() else plain).add(order[idx])
        assert hatched == {"deep_cusp_3x16", "deep_mgga_3x16"}
        assert plain == {"deep_3x16"}
        assert any("shallower training depth" in t.get_text()
                   for t in ax.get_legend().get_texts())


def test_mae_by_arch_level_grid_has_no_mark(tmp_path):
    """Byte-stability guard: a complete grid must render with no hatch, no
    coverage legend key and no extra footer line."""
    import matplotlib
    matplotlib.use("Agg")
    with _captured_figures() as seen:
        assert _png_ok(fig.plot_mae_by_arch(_make_multirung_rows(), [],
                                            tmp_path / "level.png", "run_x"))
        ax = seen[-1].axes[0]
        assert not any(p.get_hatch() for p in _bar_rects(ax))
        assert not any("shallower training depth" in t.get_text()
                       for t in ax.get_legend().get_texts())
        assert not any("UNEQUAL TRAINING DEPTH" in t.get_text()
                       for t in seen[-1].texts)


def test_rung_summary_fades_the_shallow_rung(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    rows = _make_uneven_rung_rows()
    with _captured_figures() as seen:
        out = fig.plot_rung_summary(rows, tmp_path / "rs.png", "run_x",
                                    pbe_baseline={"combined": 15.0})
        assert _png_ok(out)
        f = seen[-1]
        ax = f.axes[0]
        rungs = [r for r in fig.arch_style.RUNG_ORDER
                 if r in fig.arch_style.by_rung(fig._archs_present(rows))]
        assert rungs == [fig.arch_style.RUNG_GGA, fig.arch_style.RUNG_MGGA]
        warn = fig.matplotlib.colors.to_rgba(fig._SHALLOW_EDGE)
        black = fig.matplotlib.colors.to_rgba("k")
        edges, faces = {}, {}
        for p in _bar_rects(ax):
            idx = int(round(p.get_x() + p.get_width() / 2.0))
            if 0 <= idx < len(rungs):
                edges.setdefault(rungs[idx], set()).add(tuple(p.get_edgecolor()))
                faces.setdefault(rungs[idx], set()).add(
                    tuple(p.get_facecolor()[:3]))
        assert edges[fig.arch_style.RUNG_MGGA] == {tuple(warn)}
        assert edges[fig.arch_style.RUNG_GGA] == {tuple(black)}
        # The faded face keeps the rung hue rather than switching color.
        # Compared element-wise -- a set of pytest.approx objects compares by
        # hash, not tolerance, so `{approx(x)} == {y}` would never match.
        for rung, want in (
                (fig.arch_style.RUNG_MGGA,
                 fig._fade(fig.arch_style.RUNG_ACCENT[fig.arch_style.RUNG_MGGA])),
                (fig.arch_style.RUNG_GGA,
                 fig.matplotlib.colors.to_rgb(
                     fig.arch_style.RUNG_ACCENT[fig.arch_style.RUNG_GGA]))):
            assert len(faces[rung]) == 1, faces[rung]
            assert next(iter(faces[rung])) == pytest.approx(want)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert any(t.startswith("meta-GGA") and "n=1" in t and "ss 1)" in t
                   for t in labels)
        assert any(t.startswith("GGA") and "n=2" in t and "ss 15-26)" in t
                   for t in labels)
        assert any("UNEQUAL TRAINING DEPTH" in t.get_text() for t in f.texts)


def test_rung_summary_level_grid_unmarked(tmp_path):
    """The level fixture keeps every bar at the plain black edge -- the guard
    that a complete sweep renders as it did before the mark existed."""
    import matplotlib
    matplotlib.use("Agg")
    with _captured_figures() as seen:
        assert _png_ok(fig.plot_rung_summary(_make_multirung_rows(),
                                             tmp_path / "rs_level.png", "run_x"))
        f = seen[-1]
        black = tuple(fig.matplotlib.colors.to_rgba("k"))
        assert {tuple(p.get_edgecolor())
                for p in _bar_rects(f.axes[0])} == {black}
        assert not any("UNEQUAL TRAINING DEPTH" in t.get_text() for t in f.texts)


def test_parity_legends_state_each_archs_subset_size(tmp_path):
    """Both parity figures pick each arch's DEEPEST spec, which need not be the
    same depth -- the legend has to say which, or the cloud mixes a 1-molecule
    net with a full-pool one invisibly."""
    import matplotlib
    matplotlib.use("Agg")
    run = _make_run_dir(tmp_path)
    rows = fig.collect_holdout_reaction_rows(run)
    best = fig._best_subset_per_arch(rows)
    assert best, "fixture should carry held-out rows"
    with _captured_figures() as seen:
        assert _png_ok(fig.plot_parity(rows, tmp_path / "par.png", _STAMP))
        assert _png_ok(fig.plot_ae_parity(rows, tmp_path / "aepar.png", _STAMP))
        assert len(seen) == 2
        for f in seen:
            labels = [t.get_text() for t in f.legends[0].get_texts()]
            named = [a for a in best if any(a in lbl for lbl in labels)]
            assert named, labels
            for arch in named:
                assert f"{arch} (ss {best[arch]})" in labels, labels


def test_coverage_marks_render_on_a_single_rung_and_single_cell(tmp_path):
    """Degenerate shapes the partial sweep actually produces: one rung only,
    and a rung whose single arch holds exactly one subset size."""
    import matplotlib
    matplotlib.use("Agg")
    with _captured_figures():
        one_rung = [r for r in _make_uneven_rung_rows()
                    if r["arch"] != "deep_mgga_3x16"]
        assert _png_ok(fig.plot_rung_summary(one_rung, tmp_path / "one.png",
                                             "run_x"))
        assert _png_ok(fig.plot_mae_by_arch(one_rung, [], tmp_path / "one_a.png",
                                            "run_x"))
        single = [r for r in _make_uneven_rung_rows() if r["subset_size"] == 1]
        assert _png_ok(fig.plot_rung_summary(single, tmp_path / "sgl.png",
                                             "run_x"))
        assert _png_ok(fig.plot_mae_by_arch(single, [], tmp_path / "sgl_a.png",
                                            "run_x"))


# ---------------------------------------------------------------------------
# SCAN meta-GGA reference. The _mgga archs pretrain to SCAN, so it is the
# comparator they are judged against -- but a SCAN line drawn beside a PBE line
# must reduce the SAME reactions/species, or the two are different benchmarks.
# ---------------------------------------------------------------------------

def _rxns(n: int, pool: str = "bh76"):
    return [{"source_pool": pool, "name": f"r{i}", "reactants": [f"a{i}"],
             "products": [f"b{i}"], "coeffs": [-1.0, 1.0],
             "reaction_energy_ref": 10.0} for i in range(n)]


def _energy_map(n: int, product: float):
    out = {f"a{i}": 0.0 for i in range(n)}
    out.update({f"b{i}": product for i in range(n)})
    return out


def test_scan_pool_baseline_without_a_cache_is_all_nan(tmp_path):
    """The state every run has been in until now: no cache -> all-NaN, empty
    coverage, and no line drawn. This is what keeps the change inert."""
    b = fig.scan_pool_baseline(tmp_path)
    assert all(math.isnan(b[k]) for k in ("bh76", "w411", "combined"))
    assert b["coverage"] == {}
    assert fig.scan_line_value(b) == (None, "")
    assert fig.scan_coverage(b) == (0, 0)


def test_scan_pool_baseline_counts_coverage_against_pbe():
    """A reaction SCAN cannot score is silently dropped by
    reaction_mae_kcalmol, so coverage is measured against what PBE scored on
    the same list -- not against what SCAN happened to have."""
    rx = _rxns(10)
    pbe = _energy_map(10, 0.02)
    full = fig.scan_pool_baseline(
        Path("."), _loader=lambda: ({}, rx), _energies=_energy_map(10, 0.03),
        _pbe_energies=pbe)
    assert fig.scan_coverage(full, "bh76") == (10, 10)
    val, suffix = fig.scan_line_value(full, "bh76")
    assert val is not None and suffix == ""      # full coverage -> unqualified

    partial = dict(_energy_map(10, 0.03))
    del partial["b9"]                            # 9/10 = 90%, at the floor
    p = fig.scan_pool_baseline(Path("."), _loader=lambda: ({}, rx),
                               _energies=partial, _pbe_energies=pbe)
    assert fig.scan_coverage(p, "bh76") == (9, 10)
    val_p, suffix_p = fig.scan_line_value(p, "bh76")
    assert val_p is not None and suffix_p == ", 9/10"   # drawn, but qualified


def test_scan_line_is_withdrawn_below_the_coverage_floor():
    """Below the floor SCAN is a different benchmark, not a reference."""
    rx = _rxns(10)
    pbe = _energy_map(10, 0.02)
    thin = dict(_energy_map(10, 0.03))
    for i in (7, 8, 9):
        del thin[f"b{i}"]                        # 7/10 = 70%
    b = fig.scan_pool_baseline(Path("."), _loader=lambda: ({}, rx),
                               _energies=thin, _pbe_energies=pbe)
    assert fig.scan_coverage(b, "bh76") == (7, 10)
    assert math.isfinite(b["bh76"])              # the number exists ...
    assert fig.scan_line_value(b, "bh76") == (None, "")   # ... but is not drawn


def test_wtmad2_scan_baseline_reduces_the_same_reactions_as_pbe():
    rows = [{"name": f"r{i}", "pool": "bh76", "ref_kcalmol": 20.0,
             "abs_error_pbe_kcalmol": 2.0} for i in range(5)]
    pbe = fig.wtmad2_pbe_baseline(rows)
    scan, used, ref = fig.wtmad2_scan_baseline(rows, {f"r{i}": 4.0
                                                      for i in range(5)})
    assert (used, ref) == (5, 5)
    assert scan == pytest.approx(2.0 * pbe)      # errors doubled -> WTMAD-2 doubles
    # a reaction SCAN could not score shrinks `used`, never the reference count
    scan_p, used_p, ref_p = fig.wtmad2_scan_baseline(
        rows, {f"r{i}": 4.0 for i in range(3)})
    assert (used_p, ref_p) == (3, 5)
    assert math.isnan(fig.wtmad2_scan_baseline(rows, {})[0])


def test_ed_scan_is_a_second_point_not_a_copy_of_its_energy_leg():
    """gamma is calibrated on PBE, so ed_pbe == e_pbe by construction but
    ed_scan does NOT equal e_scan -- SCAN's density leg moves it. A mutant
    reusing the PBE anchor, or setting ed_scan = e_scan, fails here."""
    s = fig.combined_ed_by_cell({("deep", 1): 8.0}, 10.0,
                                {("deep", 1): 0.004}, 0.005,
                                e_scan=9.0, d_scan=0.010)
    assert s["ed_pbe"] == pytest.approx(s["e_pbe"])          # the identity
    gamma = s["gamma"]
    assert gamma == pytest.approx(10.0 / 0.005)
    assert s["ed_scan"] == pytest.approx(
        2.0 / (1.0 / 9.0 + 1.0 / (gamma * 0.010)))
    assert s["ed_scan"] != pytest.approx(s["e_scan"])
    assert s["cells"][("deep", 1)]["beats_scan"] is True


def test_ed_scan_is_none_without_both_scan_legs():
    """Either leg missing -> no SCAN ED, so the panels omit the line rather
    than drawing a half-defined reference."""
    base = ({("deep", 1): 8.0}, 10.0, {("deep", 1): 0.004}, 0.005)
    assert fig.combined_ed_by_cell(*base)["ed_scan"] is None
    assert fig.combined_ed_by_cell(*base, e_scan=9.0)["ed_scan"] is None
    assert fig.combined_ed_by_cell(*base, d_scan=0.01)["ed_scan"] is None
    assert fig.combined_ed_by_cell(*base, e_scan=9.0,
                                   d_scan=0.01)["ed_scan"] is not None
    # the fixed-gamma twin behaves identically
    fixed = fig.combined_ed_fixed_gamma(*base, 2000.0)
    assert fixed["ed_scan"] is None
    assert fig.combined_ed_fixed_gamma(*base, 2000.0, e_scan=9.0,
                                       d_scan=0.01)["ed_scan"] is not None


def test_ed_summary_is_unchanged_when_no_scan_is_supplied():
    """Byte-stability: adding the SCAN legs must not move a single PBE number."""
    base = ({("deep", 1): 8.0, ("deep", 2): 6.0}, 10.0,
            {("deep", 1): 0.004, ("deep", 2): 0.003}, 0.005)
    a = fig.combined_ed_by_cell(*base)
    b = fig.combined_ed_by_cell(*base, e_scan=9.0, d_scan=0.01)
    for k in ("gamma", "e_pbe", "d_pbe", "ed_pbe"):
        assert a[k] == pytest.approx(b[k])
    for cell in a["cells"]:
        for k in ("E", "D", "gammaD", "ED", "beats_pbe"):
            assert a["cells"][cell][k] == b["cells"][cell][k]


def test_scan_density_mean_uses_the_pbe_anchors_species():
    """The PBE density anchor is a mean over the plotted species; SCAN's must
    be over THOSE species. A mutant averaging the whole cache picks up species
    the panel never plots."""
    recs = {"H2O": {"density_rmse_scan": 1.0e-4},
            "CH": {"density_rmse_scan": 3.0e-4},
            "NotPlotted": {"density_rmse_scan": 9.9e-1}}
    mean, used, ref = fig.scan_density_mean(recs, ["H2O", "CH"])
    assert mean == pytest.approx(2.0e-4)          # excludes NotPlotted
    assert (used, ref) == (2, 2)
    assert fig.scan_density_line(recs, ["H2O", "CH"]) == pytest.approx(2.0e-4)
    # a species the cache lacks counts against coverage
    mean2, used2, ref2 = fig.scan_density_mean(recs, ["H2O", "CH", "Missing"])
    assert (used2, ref2) == (2, 3)
    assert fig.scan_density_line(recs, ["H2O", "CH", "Missing"]) is None
    assert fig.scan_density_line({}, ["H2O"]) is None


def test_scan_density_key_map_covers_every_pbe_density_column():
    """Each PBE density column the figures select must have a SCAN twin, or the
    selector silently yields no line on that channel."""
    for pbe_key in ("density_rmse_pbe", "density_eps_l1_pbe"):
        assert pbe_key in fig._SCAN_DENSITY_KEY
        assert fig._SCAN_DENSITY_KEY[pbe_key].endswith("_scan")


def test_scan_density_baseline_degrades_and_matches_the_pbe_species(tmp_path):
    hd = [{"molecule": m, "density_rmse_pbe": 2.0e-4, "density_rmse": 3.0e-4}
          for m in ("H2O", "CH")]
    assert math.isnan(fig.scan_density_baseline(hd, tmp_path,
                                                _records={})["value"])
    b = fig.scan_density_baseline(
        hd, tmp_path, _records={"H2O": {"density_rmse_scan": 1.0e-4},
                                "CH": {"density_rmse_scan": 3.0e-4}})
    assert b["value"] == pytest.approx(2.0e-4)
    assert fig.scan_coverage(b, "value") == (2, 2)


def test_rung_summary_draws_the_scan_line_only_with_coverage(tmp_path):
    """End to end on the figure: full coverage draws an unqualified SCAN line,
    thin coverage draws none, and no cache leaves the figure as it was."""
    import matplotlib
    matplotlib.use("Agg")
    rows = _make_multirung_rows()

    def _labels(scan_baseline):
        with _captured_figures() as seen:
            fig.plot_rung_summary(rows, tmp_path / "rs.png", "run_x",
                                  pbe_baseline={"combined": 15.0},
                                  scan_baseline=scan_baseline)
            ax = seen[-1].axes[0]
            return [t.get_text() for t in ax.get_legend().get_texts()]

    full = {"combined": 9.0, "bh76": 9.0, "w411": 9.0,
            "coverage": {"combined": {"used": 100, "reference": 100}}}
    thin = {"combined": 9.0, "bh76": 9.0, "w411": 9.0,
            "coverage": {"combined": {"used": 50, "reference": 100}}}
    assert any("SCAN (combined 9.0)" in t for t in _labels(full))
    assert not any("SCAN" in t for t in _labels(thin))
    assert not any("SCAN" in t for t in _labels(fig._nan_baseline()))
    part = {"combined": 9.0, "bh76": 9.0, "w411": 9.0,
            "coverage": {"combined": {"used": 95, "reference": 100}}}
    assert any("SCAN (combined 9.0, 95/100)" in t for t in _labels(part))


# ---------------------------------------------------------------------------
# meta-GGA enhancement factors. The meta-GGA archs pretrain to SCAN, not PBE,
# and their zero-descriptor curve IS the alpha=0 (single-orbital) slice -- the
# least representative point of the domain. Both facts have to survive in the
# figure code or the meta-GGA family is read against the wrong reference at the
# wrong point.
# ---------------------------------------------------------------------------

def test_scan_fx_curve_matches_libxc_at_known_points():
    """SCAN's exchange enhancement, checked against values libxc produces --
    the INDEPENDENT oracle (not this repo's SCAN path, which also feeds the
    pretrain target, so a shared bug could not show up there)."""
    pytest.importorskip("pyscf")
    import numpy as np
    s = np.array([0.0, 1.0, 4.0])
    fx0 = ef.scan_fx_curve(s, alpha=0.0)
    assert fx0 is not None
    # alpha=0, s=0 is SCAN's single-orbital limit h_0x = 1.174.
    assert fx0[0] == pytest.approx(1.174, abs=2e-3)
    for a in (0.0, 1.0, 100.0):
        curve = ef.scan_fx_curve(s, alpha=a)
        assert np.all(curve <= 1.174 + 1e-6), (a, curve)
    # F_x decreases with alpha at fixed s (iso-orbital -> UEG -> overlap);
    # a mutant ignoring alpha collapses these onto one another.
    at_s1 = [float(ef.scan_fx_curve(np.array([1.0]), alpha=a)[0])
             for a in (0.0, 1.0, 5.0, 100.0)]
    assert at_s1 == sorted(at_s1, reverse=True), at_s1
    assert at_s1[0] - at_s1[-1] > 0.2      # a real spread, not numerical noise


def test_alpha_panels_span_the_physical_range():
    """The sweep must reach the uniform-gas point and the clip ceiling, or the
    panel cannot show where the net leaves its target."""
    assert 0.0 in ef._ALPHA_PANELS and 1.0 in ef._ALPHA_PANELS
    assert max(ef._ALPHA_PANELS) == 100.0     # metagga.py clips alpha there
    assert list(ef._ALPHA_PANELS) == sorted(ef._ALPHA_PANELS)


def test_provenance_stamps_dm_entropy_caveat_only_on_pre_fix_runs():
    """The caveat used to be stamped unconditionally, which printed a false
    provenance claim on every run newer than the fix."""
    old = ef._provenance(Path("run_20260529T165503Z"), False)
    new = ef._provenance(Path("run_20260728T140018Z"), False)
    assert "Pre-dm_entropy-fix" in old
    assert "Pre-dm_entropy-fix" not in new
    assert "extras=0" in new and "alpha" not in new
    with_alpha = ef._provenance(Path("run_20260728T140018Z"), True)
    assert "alpha=0 slice" in with_alpha


#: The dfs6311 production pull is the only local run carrying a meta-GGA
#: checkpoint (_REAL_RUN predates the meta-GGA archs entirely).
_MGGA_RUN = (Path.home() / "Documents/Research/xcquinox-results/runs/dfs_step7"
             / "dfs6311_grid3_v3/runs/run_20260728T140018Z")


def test_model_fx_curve_alpha_moves_the_curve():
    """``alpha=`` must reach the descriptor column the X-net reads. A mutant
    that drops the kwarg (or writes the wrong column) returns the alpha=0
    curve for every alpha, collapsing the panel to a single line."""
    import numpy as np
    reps = ef.representative_specs(_MGGA_RUN) if _MGGA_RUN.is_dir() else {}
    mgga = [a for a in reps if "mgga" in a]
    if not mgga:
        pytest.skip("no meta-GGA checkpoint in the reference run")
    _spec, model = ef.load_trained_model(_MGGA_RUN, reps[mgga[0]])
    assert ef.is_meta_gga(model)
    s = np.linspace(1e-3, 3.0, 40)
    curves = {a: ef.model_fx_curve(model, s, alpha=a) for a in (0.0, 1.0, 100.0)}
    for c in curves.values():
        assert np.all(np.isfinite(c))
    assert not np.allclose(curves[0.0], curves[1.0])
    assert not np.allclose(curves[1.0], curves[100.0])
    # alpha=None reproduces the historical zero-descriptor slice exactly
    assert np.allclose(ef.model_fx_curve(model, s), curves[0.0])


def test_gga_arch_is_not_meta_gga_and_ignores_alpha():
    """Byte-stability for the GGA family: no alpha column, so the new kwarg
    must not perturb their curves."""
    import numpy as np
    reps = ef.representative_specs(_MGGA_RUN) if _MGGA_RUN.is_dir() else {}
    gga = [a for a in reps if "mgga" not in a and "rung35" not in a]
    if not gga:
        pytest.skip("no GGA checkpoint in the reference run")
    _spec, model = ef.load_trained_model(_MGGA_RUN, reps[gga[0]])
    assert not ef.is_meta_gga(model)
    s = np.linspace(1e-3, 3.0, 40)
    base = ef.model_fx_curve(model, s)
    assert np.allclose(ef.model_fx_curve(model, s, alpha=None), base)
