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


# ---------------------------------------------------------------------------
# T6: architecture display names (2026-09-09)
#
# The pre-training directories keep the stored key (they are what the cluster
# wrote); the legend states what each network is.
# ---------------------------------------------------------------------------

def test_legend_labels_show_the_display_name(tmp_path, monkeypatch):
    """A ``pretrain/medium`` directory is labelled ``deep_3x16``.

    Kills the label half of m3/m6 here: the curve of ``medium`` and the curve
    of the registry's ``deep_3x16`` would otherwise sit in one legend under
    two names that say nothing about the difference between them (the
    zero-initialized last layer), while a reader takes ``medium`` for a size.
    """
    import matplotlib.figure as mfig
    captured = []
    real = mfig.Figure.savefig

    def _cap(self, *a, **k):
        captured.append(self)
        return real(self, *a, **k)

    monkeypatch.setattr(mfig.Figure, "savefig", _cap)
    run = _make_run(tmp_path, archs=("medium", "deep_3x16"), n=60)
    curves = ppc.load_pretrain_curves(run)
    # the loader still keys on the DIRECTORY, which is the stored key
    assert set(curves) == {"medium", "deep_3x16"}
    ppc.plot_pretraining_curves(curves, tmp_path / "out.png",
                                run_label="run_x")
    assert captured, "no figure was saved"
    f = captured[-1]
    labels = []
    for ax in f.axes:
        legend = ax.get_legend()
        if legend is not None:
            labels += [t.get_text() for t in legend.get_texts()]
    assert labels
    assert any(t.startswith("deep_3x16") for t in labels), labels
    assert any(t.startswith("deep0_3x16") for t in labels), labels
    assert not any(t.startswith("medium") for t in labels), labels


def test_ordering_keeps_directory_keys_while_color_resolves_the_display_name(
        tmp_path):
    """The curves are keyed by DIRECTORY (the stored key), so the ordering
    helper must hand those keys back -- the plot loop indexes the mapping with
    them -- while the palette, now keyed on the shown names, is reached
    through the display map.

    RED: ``arch_color("medium")`` and ``ARCH_COLOR["deep_3x16"]`` are two
    different colours today (medium's green and the old deep blue).
    """
    run = _make_run(tmp_path, archs=("medium", "deep_3x16"), n=60)
    curves = ppc.load_pretrain_curves(run)
    assert ppc._order_archs(list(curves)) == ["medium", "deep_3x16"]
    assert ppc._arch_color("medium", 0) == \
        ppc.arch_style.ARCH_COLOR["deep_3x16"]
    assert ppc._arch_color("deep_3x16", 1) == \
        ppc.arch_style.ARCH_COLOR["deep0_3x16"]


def test_ordering_keeps_two_directories_that_share_a_shown_name():
    """A directory named by an alias sits beside the registry directory of
    the same network; the ordering must hand both back (the plot loop indexes
    the curve mapping with them), never drop one."""
    got = ppc._order_archs(["shallow", "deep_2x8", "medium"])
    assert sorted(got) == ["deep_2x8", "medium", "shallow"]
    assert got.index("deep_2x8") < got.index("medium")
    assert got.index("shallow") < got.index("medium")
