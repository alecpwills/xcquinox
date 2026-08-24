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


def _make_run(tmp_path, archs=("deep_3x16", "deep_cusp_3x16"), n=200,
              requested=None):
    """Write a minimal <run>/pretrain/<arch>/ tree with decaying loss arrays.

    ``requested`` is the schedule the metadata claims; ``n`` is the number of
    steps the curves record. They differ for a run that stopped early on its
    held-out-system validation.
    """
    requested = n if requested is None else int(requested)
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
            "arch_name": arch, "pretrain_steps": requested,
            "pretrain_steps_requested": requested, "pretrain_steps_run": n,
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


def test_run_length_is_the_curve_not_the_requested_schedule(tmp_path):
    """A run stopped early by its held-out-system validation writes fewer
    loss values than ``pretrain_steps`` asked for. The documented invariant is
    that the figure's step count comes from the CURVE: taking it from the
    metadata would put a step count on the figure that the run never reached.
    """
    run = _make_run(tmp_path, n=37, requested=200)
    curves = ppc.load_pretrain_curves(run)
    taken, asked = ppc.run_length(curves)
    assert (taken, asked) == (37, 200)
    for d in curves.values():
        assert d["x"].size == taken and d["c"].size == taken
        assert d["meta"]["pretrain_steps"] == 200


def test_suptitle_states_the_steps_taken_and_names_the_request(tmp_path):
    early = ppc.load_pretrain_curves(_make_run(tmp_path / "early", n=37,
                                               requested=200))
    full = ppc.load_pretrain_curves(_make_run(tmp_path / "full", n=200))
    title = ppc._suptitle(early, list(early), run_label="run_early")
    assert "37 steps" in title
    assert "200 requested" in title and "stopped early" in title
    assert "200 steps" not in title
    # A run that used its whole schedule says so plainly, with no aside.
    plain = ppc._suptitle(full, list(full), run_label="run_full")
    assert "200 steps" in plain and "requested" not in plain


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


def test_rung_ordering_and_render_mixed_rungs(tmp_path):
    # mixed rungs exercise the arch_style wiring (order + rung-keyed color/linestyle)
    archs = ("deep_rung35_mgga_3x16", "deep_3x16", "deep_mgga_3x16",
             "deep_rung35_3x16")
    run = _make_run(tmp_path, archs=archs, n=120)
    curves = ppc.load_pretrain_curves(run)
    if ppc.arch_style is not None:  # shared styling available -> rung-grouped order
        ordered = ppc._order_archs(list(curves))
        ranks = [ppc.arch_style.rung_rank(a) for a in ordered]
        assert ranks == sorted(ranks)
        assert ordered[0] == "deep_3x16"                # GGA first
        assert ordered[-1] == "deep_rung35_mgga_3x16"   # combined last
        # rung-keyed linestyles differ across rungs
        assert ppc._arch_linestyle("deep_3x16") != ppc._arch_linestyle("deep_mgga_3x16")
    out = tmp_path / "mixed.png"
    written = ppc.plot_pretraining_curves(curves, out, run_label="mixed")
    assert written.is_file() and written.stat().st_size > 1000
