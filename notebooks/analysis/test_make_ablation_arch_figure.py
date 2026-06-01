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
        if i == 5:
            continue  # untrained: no model.eqx, no eval
        (sd / "model.eqx").write_bytes(b"x" * 16)
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
