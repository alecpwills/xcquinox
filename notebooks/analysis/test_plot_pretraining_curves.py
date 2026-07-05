"""Smoke tests for plot_pretraining_curves.py (the pretraining-curve quick-look).

Fast + headless (matplotlib Agg): builds a synthetic run/pretrain tree, exercises
the loader + plotter, and checks a non-empty PNG is written. No cluster data.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_pretraining_curves as ppc  # noqa: E402


def _make_run(tmp_path, archs=("deep_3x16", "deep_cusp_3x16"), n=200):
    """Write a minimal <run>/pretrain/<arch>/ tree with decaying loss arrays."""
    pdir = tmp_path / "pretrain"
    for k, arch in enumerate(archs):
        adir = pdir / arch
        adir.mkdir(parents=True)
        # monotone-ish decaying curves so log-y plotting has finite positive data
        lx = np.exp(-np.linspace(0, 6, n)) * (0.02 + 0.001 * k) + 1e-5
        lc = np.exp(-np.linspace(0, 5, n)) * (0.30 + 0.01 * k) + 1e-5
        np.save(adir / "losses_x.npy", lx)
        np.save(adir / "losses_c.npy", lc)
        (adir / "pretrain_metadata.json").write_text(json.dumps({
            "arch_name": arch, "pretrain_steps": n,
            "final_loss_x": float(lx[-1]), "final_loss_c": float(lc[-1]),
        }))
    return tmp_path


def test_load_pretrain_curves_reads_all_archs(tmp_path):
    run = _make_run(tmp_path)
    curves = ppc.load_pretrain_curves(run)
    assert set(curves) == {"deep_3x16", "deep_cusp_3x16"}
    for d in curves.values():
        assert d["x"].shape == (200,)
        assert d["c"].shape == (200,)
        assert d["meta"]["pretrain_steps"] == 200


def test_load_pretrain_curves_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ppc.load_pretrain_curves(tmp_path)  # no pretrain/ subdir


def test_load_pretrain_curves_skips_arch_without_both_arrays(tmp_path):
    run = _make_run(tmp_path)
    # An arch dir with only losses_x.npy (no losses_c.npy) must be skipped.
    partial = run / "pretrain" / "deep_partial"
    partial.mkdir()
    np.save(partial / "losses_x.npy", np.ones(10))
    curves = ppc.load_pretrain_curves(run)
    assert "deep_partial" not in curves


def test_plot_pretraining_curves_writes_png(tmp_path):
    run = _make_run(tmp_path)
    curves = ppc.load_pretrain_curves(run)
    out = tmp_path / "out" / "pretraining_curves.png"
    written = ppc.plot_pretraining_curves(curves, out, run_label="run_test")
    assert written.is_file()
    assert written.stat().st_size > 1000  # a real rendered PNG, not an empty stub


def test_main_end_to_end(tmp_path):
    run = _make_run(tmp_path)
    out = tmp_path / "cli.png"
    rc = ppc.main([str(run), "-o", str(out)])
    assert rc == 0
    assert out.is_file()
