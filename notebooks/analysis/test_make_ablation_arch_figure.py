"""Tests for ``make_ablation_arch_figure.py`` and ``enhancement_factors.py``.

Three layers, mirroring ``test_make_cluster_pulls_figure.py``:
  * pure data-ingest tests on a synthetic run-dir fixture (no matplotlib),
  * render canaries that drive each plot builder and assert a non-trivial PNG,
  * pure physics-reference tests (PBE F_x / F_c) + a ``slow`` model-load test
    that deserialises one real checkpoint from the pulled run if present.
"""
from __future__ import annotations

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


def test_build_all_writes_five_figures(tmp_path):
    run = _make_run_dir(tmp_path)
    written = fig.build_all(run, tmp_path / "out")
    assert len(written) == 5
    assert all(_png_ok(p) for p in written)
    assert (tmp_path / "out" / "ablation_ae_parity.png").is_file()


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
    # cusp adds x_4,x_5 to BOTH nets; dm adds x_6,x_7,x_8
    assert forms["deep_cusp"]["fx"] == ["x_2", "x_4", "x_5"]
    assert forms["deep_cusp"]["fc"] == ["r_s", "x_2", "x_1", "x_4", "x_5"]
    assert forms["deep_dm"]["fx"] == ["x_2", "x_6", "x_7", "x_8"]
    # combined packs the DM block (x_6,x_7,x_8) BEFORE cusp (x_4,x_5) -- the
    # networks.py concat order (descriptors=[dm_statistics, cusp])
    assert forms["deep_combined"]["fx"] == ["x_2", "x_6", "x_7", "x_8", "x_4", "x_5"]
    assert forms["deep_combined"]["fc"] == [
        "r_s", "x_2", "x_1", "x_6", "x_7", "x_8", "x_4", "x_5"]
    # _attn shares its base's inputs; notransform shares deep's inputs but raw
    assert forms["deep_combined_attn"]["fx"] == forms["deep_combined"]["fx"]
    assert forms["deep_attn"]["attention"] is True
    assert forms["deep_notransform"]["fx"] == forms["deep"]["fx"]
    assert forms["deep_notransform"]["log_transform"] is False


def test_arch_forms_lines_cover_each_arch():
    lines = fig._arch_forms_lines()
    joined = " ".join(lines)
    for a in fig.ARCH_ORDER:                 # every figure arch is named
        assert a in joined
    # explicit F_x / F_c forms appear verbatim
    assert "F_x(x_2, x_4, x_5)" in joined                                # cusp
    assert "F_c(r_s, x_2, x_1, x_6, x_7, x_8, x_4, x_5)" in joined        # combined
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
    # extended descriptors: x4-x8, V_ext defined, x7 intensive
    assert "x_4" in col3 and "x_7" in col3 and "INTENSIVE" in col3
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
    # x7 probabilities are normalized before the entropy (features.py)
    assert "normalized" in col3
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
    assert len(names2) == 14
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
        "E_pbe_kcalmol", "D_pbe_rmse", "ED_pbe_kcalmol", "beats_pbe"}
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
    assert "iso-ED" in ax.get_title()
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
    assert "DFS units" in dfs_dec[0][2]["title"]
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
    assert x3_twin[0]["parity_nn_key"] == "density_eps_l1"
    # titles are clean (the in-panel stamp carries value + source) and the
    # twin renders the ED row as grouped bars (the A/B/C form), not lines
    assert x3_twin[0]["ed_gamma_label"] == ""
    assert x3_twin[0]["ed_as_bars"] is True
    # the ORIGINAL calls keep their defaults (no parity/gamma overrides)
    ov_orig = [kw for p, kw in ov_calls if p.name
               == "ablation_density_energy_overview.png"]
    x3_orig = [kw for p, kw in x3_calls if p.name
               == "ablation_density_energy_3x3.png"]
    assert len(ov_orig) == 1 and "parity_nn_key" not in ov_orig[0]
    assert len(x3_orig) == 1 and "parity_nn_key" not in x3_orig[0]
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
        ch_summaries=ch_eps, parity_nn_key="density_eps_l1",
        parity_pbe_key="density_eps_l1_pbe", parity_unit_label="Eq. 20 eps",
        ed_gamma_label="", ed_as_bars=True, title="3x3, DFS units")
    assert _png_ok(p1)
    p2 = fig.plot_density_energy_overview(
        rows, hd, tmp_path / "ov_dfs.png", "run_x",
        ed_summary=ch_eps["combined"], parity_nn_key="density_eps_l1",
        parity_pbe_key="density_eps_l1_pbe", parity_unit_label="Eq. 20 eps",
        title="Overview, DFS units")
    assert _png_ok(p2)


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


def test_3x3_parity_row_shares_limits(tmp_path, monkeypatch):
    """The 3x3 passes ONE row-wide envelope to all three parity panels --
    the channels render in the same frame and are directly comparable. The
    fixture carries a bh76-only outlier species so per-channel envelopes
    differ from the pooled one: identical per-channel frames cannot fake
    the row share."""
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
    fig.plot_density_energy_3x3(rows, hd, tmp_path / "x3.png", "run_x")
    assert len(seen) == 3
    assert all(lim is not None and lim == seen[0] for lim in seen)
    # the shared envelope is the POOLED positive envelope: lo from the HO
    # NN RMSE (2e-4), hi from the bh76-only HOh_ts PBE value (6e-3) -- a
    # per-channel w411 frame would top out at 1.25*8e-4 instead
    assert seen[0][0] == pytest.approx(0.8 * 2e-4)
    assert seen[0][1] == pytest.approx(1.25 * 6e-3)


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
    assert "x_11" in mgga_forms["deep_mgga_3x16"]["fx"], "metagga label x_11 missing"
    assert all(lbl in mgga_forms["deep_rung35_mgga_3x16"]["fx"]
               for lbl in ("x_4", "x_9", "x_11")), "combined mgga labels missing"
    # _arch_input_forms must resolve the rung-3.5 descriptor labels (no KeyError)
    forms = fig._arch_input_forms(("deep_rung35_3x16", "deep_rung35only_3x16"))
    assert all(lbl in forms["deep_rung35_3x16"]["fx"] for lbl in ("x_4", "x_9")), \
        "deep_rung35 X-net inputs should carry cusp (x_4) + rung-3.5 (x_9) labels"
