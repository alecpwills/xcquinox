"""Tests for ``make_ablation_arch_figure.py`` and ``enhancement_factors.py``.

Three layers, mirroring ``test_make_cluster_pulls_figure.py``:
  * pure data-ingest tests on a synthetic run-dir fixture (no matplotlib),
  * render canaries that drive each plot builder and assert a non-trivial PNG,
  * pure physics-reference tests (PBE F_x / F_c) + a ``slow`` model-load test
    that deserialises one real checkpoint from the pulled run if present.
"""
from __future__ import annotations

import importlib.util
import json
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
             "reaction_energy_ref_kcalmol": 17.7,
             "de_nn_kcalmol": -91.0 + i, "de_pbe_kcalmol": -91.2 + i,
             "abs_error_nn_kcalmol": 108.7 - i, "abs_error_pbe_kcalmol": 108.9 - i},
            {"name": "w411_b", "pool": "w411",
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
    assert "Slater density envelope" in col3      # verified factual, retained
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


def _add_best_eval(run_dir):
    """Duplicate each spec's eval_holdout/ -> eval_holdout_best/ so the suite's
    best-checkpoint figure set has data to render (mirrors the cluster's default
    second eval pass on model_best.eqx)."""
    import shutil
    for sd in (run_dir / "checkpoints").glob("spec_*"):
        eh = sd / "eval_holdout"
        if eh.is_dir():
            shutil.copytree(eh, sd / "eval_holdout_best", dirs_exist_ok=True)


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
    # no eval_holdout_best/ in this fixture -> NO best figure set (backward compat)
    assert not any(p.parent.name.endswith("_best") for p in written)


def test_collect_holdout_reads_named_eval_subdir(tmp_path):
    run = _make_run_dir(tmp_path)
    _add_best_eval(run)
    final = fig.collect_holdout_reaction_rows(run)
    best = fig.collect_holdout_reaction_rows(run, eval_subdir="eval_holdout_best")
    assert best and len(best) == len(final)        # best dir mirrors final here
    # absent subdir -> empty (no crash), so older runs just skip the best set
    bare = _make_run_dir(tmp_path / "bare")
    assert fig.collect_holdout_reaction_rows(
        bare, eval_subdir="eval_holdout_best") == []


def test_build_bh76w411_suite_emits_best_set_when_present(tmp_path):
    # eval_holdout_best/ present -> a SECOND, parallel figure set into
    # figures_<alias>_best/ + figures_basis_comparison_best/ (doubled figures).
    root, runs = _make_bh76w411_results(tmp_path)
    for r in runs.values():
        _add_best_eval(r)
    outroot = tmp_path / "figs"
    written = fig.build_bh76w411_suite(results_root=root, outroot=outroot)
    assert written and all(_png_ok(p) for p in written)
    parents = {p.parent.name for p in written}
    # both the final set AND the best set are present
    assert {"figures_svp", "figures_svp_best",
            "figures_tzvpd_df", "figures_tzvpd_df_best",
            "figures_basis_comparison",
            "figures_basis_comparison_best"} <= parents


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
             "reaction_energy_ref_kcalmol": 17.7,
             "de_nn_kcalmol": -91.0 + i, "de_pbe_kcalmol": -91.2 + i,
             "abs_error_nn_kcalmol": 108.7 - i, "abs_error_pbe_kcalmol": 108.9 - i},
            {"name": "w411_b", "pool": "w411",
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
             "ref_density_method": "ccsd", "from_training_subset": False},
            {"molecule": "H", "density_rmse": None, "density_l1": None,
             "density_rmse_pbe": None, "density_l1_pbe": None,
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
    assert "ablation_holdout_density_ccsd.png" not in names1   # refs-free run
    _add_holdout_density(run)
    out2 = tmp_path / "f2"
    names2 = {p.name for p in fig.build_density_energy_figures(run, out2)}
    assert "ablation_holdout_density_ccsd.png" in names2


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


def test_build_density_energy_figures_writes_two(tmp_path):
    run = _make_run_dir(tmp_path)
    written = fig.build_density_energy_figures(run, tmp_path / "out")
    assert len(written) == 2
    assert all(_png_ok(p) for p in written)
    assert {p.name for p in written} == {"ablation_energy_wtmad_mae.png",
                                         "ablation_insample_density_ccsd.png"}


# ---------------------------------------------------------------------------
# enhancement_factors — grid geometry + physics references (fast)
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
    for fn in ("dfs_step7.svp_grid2_v3.yaml", "dfs_step7.svp_grid2_v3_full25.yaml"):
        cfg = yaml.safe_load((root / "hpcjobs" / "configs" / fn).read_text())
        for a in cfg["sweep"]["arch"]:
            assert a in fig.ARCH_ORDER, f"{fn}: swept arch {a!r} not in ARCH_ORDER"
            assert fig.ARCH_COLOR.get(a) not in (None, "#333333"), \
                f"{fn}: {a!r} has no distinct color (fell back to gray)"
    for a in ("deep_rung35_3x16", "deep_rung35_attn_3x16", "deep_rung35only_3x16"):
        assert a in fig.ARCH_ORDER, f"{a} missing from ARCH_ORDER"
        assert fig.ARCH_COLOR.get(a) not in (None, "#333333"), f"{a} has no color"
    # _arch_input_forms must resolve the rung-3.5 descriptor labels (no KeyError)
    forms = fig._arch_input_forms(("deep_rung35_3x16", "deep_rung35only_3x16"))
    assert all(lbl in forms["deep_rung35_3x16"]["fx"] for lbl in ("x_4", "x_9")), \
        "deep_rung35 X-net inputs should carry cusp (x_4) + rung-3.5 (x_9) labels"
