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
