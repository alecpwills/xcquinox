#!/usr/bin/env python
"""Tests for trained_fx_fc.py -- trained enhancement factors vs the parent.

Two load-bearing pins:

  * the anchored identity. A parent-anchored network with a zero-initialized
    final layer IS the parent (F = F_parent + T(g), g = 0 at init), so a
    checkpoint written straight from a fresh build must reproduce the parent
    baselines to round-off once it has travelled through the production
    writer, the class record, the discovery and the render -- PBE for the
    GGA rung, SCAN at the exact alpha slices for the meta-GGA rung. An
    unanchored build fails the same assertion by more than 1e-2, which is
    what makes the pin discriminating rather than vacuous.
  * the class record. The parent anchor and the descriptor coordinates change
    no parameter shape, so leaves of another class deserialize into this
    skeleton with nothing raising -- the test asserts exactly that (a bare
    ``tree_deserialise_leaves`` of the same file succeeds) and then that the
    module refuses it.

The checkpoints are written by the production writer
(``train.save_trained_checkpoint``), so the records under test are the records
the training stage writes.
"""
import csv
import dataclasses
import json
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trained_fx_fc as T  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _raw_config(archs, *, parent_anchor=True, coordinates="dfs"):
    """A complete-but-minimal raw config ``load_grid_config`` accepts, stating
    the v6 model class (the parent anchor + the DFS coordinates)."""
    return {
        "sweep": {"arch": list(archs), "loss": ["l2"], "metric": ["l2"],
                  "subset_size": [1], "solver": ["oneshot"]},
        "solvers": {"oneshot": {"mode": "oneshot", "max_cycles": 1}},
        "hyperparams": {"n_steps": 1, "lr_start": 1e-3, "lr_end": 1e-4,
                        "lr_decay_start": 0.5, "grad_clip": 1.0,
                        "gradnorm_alpha": 1.0, "vxc_weight": 1.0,
                        "density_weight": 1.0},
        "inputs": {"external_refs_dir": "/tmp/refs",
                   "subset_ledger_path": "/tmp/ledger.json",
                   "basis": "sto-3g", "grid_level": 1,
                   "output_root": "/tmp/out"},
        "pretrain": {"data_dir": "/tmp/pretrain_data"},
        "cluster": {"partition": "short", "time": "01:00:00", "mem": "8G",
                    "cpus_per_task": 1, "array_throttle": 1,
                    "eval_array_throttle": 1, "max_concurrent_tasks": 10},
        "domain_profile": "dfs_step7",
        "use_polarized_correlation": True,
        "model": {"parent_anchor": parent_anchor,
                  "descriptor_coordinates": coordinates},
    }


def _make_run(tmp_path, cells, *, arch="deep_3x16"):
    """A run dir with ``resolved_config.yaml`` and a manifest listing ``cells``
    (``{index: subset_size}``). No checkpoints are written."""
    run = tmp_path / "run_20260827T163330Z"
    (run / "checkpoints").mkdir(parents=True)
    with open(run / "resolved_config.yaml", "w") as fh:
        yaml.safe_dump(_raw_config([arch]), fh)
    with open(run / "manifest.json", "w") as fh:
        json.dump({"width": 4, "n_specs": len(cells),
                   "specs": [{"index": i,
                              "cell": {"arch": arch, "subset_size": ss}}
                             for i, ss in sorted(cells.items())]}, fh)
    for index in cells:
        (run / "checkpoints" / f"spec_{index:04d}").mkdir()
    return run


def _fresh_arch(arch_name, *, parent_anchor):
    from xcquinox.alec.config import anchored, get_architecture
    arch = dataclasses.replace(get_architecture(arch_name),
                               use_polarized_correlation=True)
    if parent_anchor:
        return anchored(dataclasses.replace(arch, descriptor_coordinates="dfs"))
    return dataclasses.replace(arch, parent_anchor=False,
                               descriptor_coordinates="legacy")


def _write_checkpoint(path, arch_name, *, parent_anchor=True):
    """Serialize a fresh model of ``arch_name`` through the PRODUCTION writer,
    which is what puts the class record beside the leaves."""
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.train import save_trained_checkpoint
    arch = _fresh_arch(arch_name, parent_anchor=parent_anchor)
    model = AlecGGAModel.from_arch(arch, seed=0)
    save_trained_checkpoint(str(path), model, arch)
    return arch, model


def _write_test_set(run, index, mae, *, subdir="eval_holdout_val_best"):
    d = run / "checkpoints" / f"spec_{index:04d}" / subdir
    d.mkdir(parents=True, exist_ok=True)
    (d / "test_set.csv").write_text(
        "set,mae_nn_kcalmol,mae_pbe_kcalmol,delta_nn_minus_pbe\n"
        f"test_set_bh76,{mae + 1.0:.6f},7.7,0.3\n"
        f"test_set_held_out_combined,{mae:.6f},11.6,-4.6\n")


def _csv_rows(outdir):
    with open(outdir / "trained_fx_fc_curves.csv") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# The seam: production checkpoint -> discovery -> load -> figures + CSV
# ---------------------------------------------------------------------------

def test_seam_renders_an_anchored_checkpoint_at_round_off(tmp_path):
    """End to end on a synthetic run: the figures and the CSV land, and the
    anchored checkpoint reproduces the parent's F_x to round-off."""
    run = _make_run(tmp_path, {0: 4})
    _write_checkpoint(run / "checkpoints" / "spec_0000" / "model_val_best.eqx",
                      "deep_3x16")
    _write_test_set(run, 0, 7.0)
    outdir = tmp_path / "figs"

    assert T.build_all(run, outdir, eval_channel="val_best") == 0
    assert (outdir / "trained_fx_fc_deep_3x16.png").stat().st_size > 0
    assert (outdir / "trained_fx_fc_delta_best.png").stat().st_size > 0

    rows = _csv_rows(outdir)
    assert set(rows[0]) == {"arch", "subset_size", "channel", "rs", "s",
                            "f_model", "f_parent", "eval_channel"}
    assert len(rows) == len(T.S_GRID) * (1 + len(T.RS_VALUES))
    assert {r["eval_channel"] for r in rows} == {"val_best"}
    assert {r["subset_size"] for r in rows} == {"4"}
    fx = [r for r in rows if r["channel"] == "fx"]
    worst = max(abs(float(r["f_model"]) - float(r["f_parent"])) for r in fx)
    assert worst < 1e-10, worst
    fc = [r for r in rows if r["channel"] == "fc" and r["rs"] == "2"]
    assert len(fc) == len(T.S_GRID)
    assert max(abs(float(r["f_model"]) - float(r["f_parent"]))
               for r in fc) < 1e-8


def test_an_unanchored_checkpoint_is_far_from_the_parent(tmp_path):
    """The identity pin above is discriminating: the same seam fed an
    UNANCHORED network of identical shape sits far from PBE, so a module that
    plotted the parent against itself could not pass both tests."""
    run = _make_run(tmp_path, {0: 4})
    with open(run / "resolved_config.yaml", "w") as fh:
        yaml.safe_dump(_raw_config(["deep_3x16"], parent_anchor=False,
                                   coordinates="legacy"), fh)
    _write_checkpoint(run / "checkpoints" / "spec_0000" / "model_val_best.eqx",
                      "deep_3x16", parent_anchor=False)
    outdir = tmp_path / "figs"

    assert T.build_all(run, outdir, eval_channel="val_best") == 0
    fx = [r for r in _csv_rows(outdir) if r["channel"] == "fx"]
    worst = max(abs(float(r["f_model"]) - float(r["f_parent"])) for r in fx)
    assert worst > 1e-2, worst


# ---------------------------------------------------------------------------
# The class record
# ---------------------------------------------------------------------------

def test_a_checkpoint_of_another_class_is_refused(tmp_path):
    """A run stating the anchored DFS class, holding a checkpoint written as
    the unanchored legacy class: the leaves fit the skeleton exactly (asserted
    here by a bare deserialise that SUCCEEDS), so only the record beside them
    can catch it -- and the module must surface that refusal, not render."""
    import equinox as eqx
    from xcquinox.alec.checkpoint_class import ModelClassMismatch
    from xcquinox.alec.models import AlecGGAModel

    run = _make_run(tmp_path, {0: 4})
    path = run / "checkpoints" / "spec_0000" / "model_val_best.eqx"
    _write_checkpoint(path, "deep_3x16", parent_anchor=False)
    record = json.loads((path.with_name(path.name + ".class.json")).read_text())
    assert record["parent_anchor"] is False
    assert record["descriptor_coordinates"] == "legacy"

    # The leaves themselves are indistinguishable: a bare deserialise into the
    # anchored skeleton the run's config builds raises nothing.
    cfg = T.load_run_config(run)
    anchored_arch = T.arch_from_config(cfg, "deep_3x16")
    assert anchored_arch.parent_anchor
    eqx.tree_deserialise_leaves(
        str(path), AlecGGAModel.from_arch(anchored_arch, seed=0))

    outdir = tmp_path / "figs"
    with pytest.raises(ModelClassMismatch, match="different model classes"):
        T.build_all(run, outdir, eval_channel="val_best")
    assert not (outdir / "trained_fx_fc_curves.csv").exists()


def test_a_stale_class_record_is_refused(tmp_path):
    """A record that does not describe the leaves beside it (a write killed
    between the two renames) is refused by digest before any class comparison."""
    from xcquinox.alec.checkpoint_class import ClassRecordStale

    run = _make_run(tmp_path, {0: 4})
    path = run / "checkpoints" / "spec_0000" / "model_val_best.eqx"
    _write_checkpoint(path, "deep_3x16")
    path.write_bytes(path.read_bytes() + b"\x00")  # leaves move, record does not

    with pytest.raises(ClassRecordStale, match="sha256"):
        T.build_all(run, tmp_path / "figs", eval_channel="val_best")


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

def test_the_val_best_fallback_labels_itself(tmp_path):
    """A cell with no model_val_best.eqx is drawn from model.eqx, and says so
    in the CSV column and in the figure footer."""
    run = _make_run(tmp_path, {0: 4})
    _write_checkpoint(run / "checkpoints" / "spec_0000" / "model.eqx",
                      "deep_3x16")
    cells, missing = T.discover_cells(run, "val_best")
    assert missing == []
    assert [(c.channel, c.fallback) for c in cells] == [("final", True)]

    note = T._fallback_note(cells, "val_best")
    assert "model_val_best.eqx" in note and "model.eqx" in note
    assert "deep_3x16/4" in note

    outdir = tmp_path / "figs"
    assert T.build_all(run, outdir, eval_channel="val_best") == 0
    assert {r["eval_channel"] for r in _csv_rows(outdir)} == {"final"}


def test_the_final_channel_has_no_fallback(tmp_path):
    """``--eval-channel final`` reads model.eqx and nothing else: a cell with
    only model_val_best.eqx is not on that channel."""
    run = _make_run(tmp_path, {0: 4})
    (run / "checkpoints" / "spec_0000" / "model_val_best.eqx").write_bytes(b"x")
    cells, missing = T.discover_cells(run, "final")
    assert cells == []
    assert missing == [(0, "deep_3x16", 4)]


# ---------------------------------------------------------------------------
# The run whose weights were not pulled
# ---------------------------------------------------------------------------

def test_a_run_without_checkpoints_names_the_files_and_the_pull(tmp_path,
                                                                capsys):
    """A run dir carrying its evaluation tables and no weights: the module
    names the missing files and the pull that fetches them, exits nonzero, and
    writes nothing."""
    run = _make_run(tmp_path, {0: 4, 1: 8})
    _write_test_set(run, 0, 7.0)
    _write_test_set(run, 1, 6.5)
    outdir = tmp_path / "figs"

    assert T.build_all(run, outdir, eval_channel="val_best") == 2
    out = capsys.readouterr().out
    assert "model_val_best.eqx" in out and "model.eqx" in out
    assert "checkpoints/spec_0000/model_val_best.eqx" in out
    assert "xcquinox.alec.cluster pull run_20260827T163330Z" in out
    assert "--profile summaries" in out
    assert not outdir.exists(), "a refused run must render nothing"


# ---------------------------------------------------------------------------
# Best-cell selection
# ---------------------------------------------------------------------------

def test_the_best_cell_is_the_smallest_combined_held_out_mae(tmp_path):
    """Selection reads the combined row of the channel's own held-out eval --
    not the bh76 row, not the largest subset size."""
    run = _make_run(tmp_path, {0: 4, 1: 8, 2: 12})
    for index, ss in ((0, 4), (1, 8), (2, 12)):
        (run / "checkpoints" / f"spec_{index:04d}"
         / "model_val_best.eqx").write_bytes(b"x")
    _write_test_set(run, 0, 9.0)
    _write_test_set(run, 1, 6.5)   # the winner, and NOT the largest subset
    _write_test_set(run, 2, 8.0)
    cells, _missing = T.discover_cells(run, "val_best")

    assert T.held_out_mae(run, 1, 4, "val_best") == pytest.approx(6.5)
    selected, unranked = T.best_cells(run, cells, 4)
    assert unranked == []
    assert [(c.index, c.subset_size, mae) for c, mae in selected] == [
        (1, 8, pytest.approx(6.5))]


def test_an_arch_without_a_held_out_eval_falls_back_to_the_largest_cell(tmp_path):
    """No evaluation on disk: the largest completed subset size is drawn and
    the arch is reported as unranked, so the figure can say the cell was not
    selected on a measurement."""
    run = _make_run(tmp_path, {0: 4, 1: 8})
    for index in (0, 1):
        (run / "checkpoints" / f"spec_{index:04d}"
         / "model_val_best.eqx").write_bytes(b"x")
    cells, _missing = T.discover_cells(run, "val_best")

    selected, unranked = T.best_cells(run, cells, 4)
    assert unranked == ["deep_3x16"]
    assert [(c.subset_size, mae) for c, mae in selected] == [(8, None)]


def test_the_selection_reads_the_channel_its_own_weights_came_from(tmp_path):
    """A fallback cell is ranked on eval_holdout/, an on-channel cell on
    eval_holdout_val_best/: scoring a cell on a sibling checkpoint's
    evaluation would rank the curve against weights it is not drawn from."""
    run = _make_run(tmp_path, {0: 4})
    (run / "checkpoints" / "spec_0000" / "model.eqx").write_bytes(b"x")
    _write_test_set(run, 0, 3.3, subdir="eval_holdout")
    _write_test_set(run, 0, 9.9, subdir="eval_holdout_val_best")
    cells, _missing = T.discover_cells(run, "val_best")

    assert cells[0].channel == "final"
    selected, _unranked = T.best_cells(run, cells, 4)
    assert selected[0][1] == pytest.approx(3.3)


# ---------------------------------------------------------------------------
# Meta-GGA routing, discovery, shading
# ---------------------------------------------------------------------------

def test_seam_renders_an_anchored_mgga_checkpoint_at_round_off(tmp_path):
    """End to end on a synthetic meta-GGA run: the arch routes to its SCAN
    parent, the alpha-sliced figure and the alpha-columned CSV land, and the
    anchored checkpoint reproduces ``parents.scan_*`` to round-off at BOTH
    alpha slices. ``deep_cusp_mgga_3x16`` holds the alpha column at index 2,
    so a curve helper that hardcodes index 0 fails here: the anchored parent
    would read the zeroed true column as alpha ~ 1."""
    run = _make_run(tmp_path, {0: 4}, arch="deep_cusp_mgga_3x16")
    _write_checkpoint(run / "checkpoints" / "spec_0000" / "model_val_best.eqx",
                      "deep_cusp_mgga_3x16")
    _write_test_set(run, 0, 7.0)
    outdir = tmp_path / "figs"

    assert T.build_all(run, outdir, eval_channel="val_best") == 0
    assert (outdir
            / "trained_fx_fc_deep_cusp_mgga_3x16.png").stat().st_size > 0
    assert (outdir / "trained_fx_fc_delta_best.png").stat().st_size > 0

    rows = _csv_rows(outdir)
    assert set(rows[0]) == {"arch", "subset_size", "channel", "rs", "alpha",
                            "s", "f_model", "f_parent", "eval_channel"}
    n_scan = len(T.S_GRID) * len(T.ALPHA_VALUES) * (1 + len(T.RS_VALUES))
    assert len(rows) == n_scan
    fx = [r for r in rows if r["channel"] == "fx"]
    assert sorted({r["alpha"] for r in fx}) == ["0", "1"]
    worst = max(abs(float(r["f_model"]) - float(r["f_parent"])) for r in fx)
    assert worst < 1e-10, worst
    fc0 = [r for r in rows if r["channel"] == "fc" and r["alpha"] == "0"
           and r["rs"] == "2"]
    assert len(fc0) == len(T.S_GRID)
    assert max(abs(float(r["f_model"]) - float(r["f_parent"]))
               for r in fc0) < 1e-10
    fc = [r for r in rows if r["channel"] == "fc"]
    assert max(abs(float(r["f_model"]) - float(r["f_parent"]))
               for r in fc) < 1e-8


def test_an_unanchored_mgga_checkpoint_is_far_from_scan(tmp_path):
    """The SCAN identity pin is discriminating: the unanchored twin sits at
    F = 1 (zero-initialized final layer), 0.174 under SCAN's alpha=0 ceiling
    at s=0, so a module drawing the parent against itself could not pass
    both tests."""
    run = _make_run(tmp_path, {0: 4}, arch="deep_mgga_3x16")
    with open(run / "resolved_config.yaml", "w") as fh:
        yaml.safe_dump(_raw_config(["deep_mgga_3x16"], parent_anchor=False,
                                   coordinates="legacy"), fh)
    _write_checkpoint(run / "checkpoints" / "spec_0000" / "model_val_best.eqx",
                      "deep_mgga_3x16", parent_anchor=False)
    outdir = tmp_path / "figs"

    assert T.build_all(run, outdir, eval_channel="val_best") == 0
    fx0 = [r for r in _csv_rows(outdir)
           if r["channel"] == "fx" and r["alpha"] == "0"]
    worst = max(abs(float(r["f_model"]) - float(r["f_parent"])) for r in fx0)
    assert worst > 1e-2, worst


def test_a_mixed_run_routes_each_arch_to_its_own_parent(tmp_path):
    """A run holding both rungs renders BOTH architectures in one invocation,
    the GGA one against PBE and the meta-GGA one against SCAN -- no refusal,
    no cross-parent draw: each family's parent column matches its own
    baseline to round-off while the two baselines differ by more than 1e-2."""
    run = _make_run(tmp_path, {0: 4})
    with open(run / "manifest.json", "w") as fh:
        json.dump({"width": 4, "n_specs": 2, "specs": [
            {"index": 0, "cell": {"arch": "deep_3x16", "subset_size": 4}},
            {"index": 1, "cell": {"arch": "deep_mgga_3x16",
                                  "subset_size": 4}}]}, fh)
    (run / "checkpoints" / "spec_0001").mkdir()
    _write_checkpoint(run / "checkpoints" / "spec_0000" / "model_val_best.eqx",
                      "deep_3x16")
    _write_checkpoint(run / "checkpoints" / "spec_0001" / "model_val_best.eqx",
                      "deep_mgga_3x16")
    outdir = tmp_path / "figs"

    assert T.build_all(run, outdir, eval_channel="val_best") == 0
    assert (outdir / "trained_fx_fc_deep_3x16.png").is_file()
    assert (outdir / "trained_fx_fc_deep_mgga_3x16.png").is_file()
    rows = _csv_rows(outdir)
    gga_fx = [r for r in rows if r["arch"] == "deep_3x16"
              and r["channel"] == "fx"]
    scan_fx0 = [r for r in rows if r["arch"] == "deep_mgga_3x16"
                and r["channel"] == "fx" and r["alpha"] == "0"]
    assert {r["alpha"] for r in gga_fx} == {""}
    assert len(gga_fx) == len(T.S_GRID)
    assert len(scan_fx0) == len(T.S_GRID)
    for family in (gga_fx, scan_fx0):
        worst = max(abs(float(r["f_model"]) - float(r["f_parent"]))
                    for r in family)
        assert worst < 1e-10, worst
    # No cross-parent draw: each family matched its own baseline to round-off
    # above, and the two baselines are far apart (both row lists are in
    # S_GRID order, so the zip compares equal-s points).
    gap = max(abs(float(g["f_parent"]) - float(m["f_parent"]))
              for g, m in zip(gga_fx, scan_fx0))
    assert gap > 1e-2, gap


def test_an_arch_restriction_matching_nothing_says_what_the_run_holds(
        tmp_path, capsys):
    run = _make_run(tmp_path, {0: 4})
    assert T.build_all(run, tmp_path / "figs", archs=("deep_cusp_3x16",)) == 2
    out = capsys.readouterr().out
    assert "matches no cell" in out and "deep_3x16" in out


def test_a_run_dir_without_a_manifest_is_reported(tmp_path, capsys):
    run = _make_run(tmp_path, {0: 4})
    (run / "manifest.json").unlink()
    assert T.build_all(run, tmp_path / "figs") == 2
    assert "manifest.json" in capsys.readouterr().out


def test_discovery_reports_cells_without_weights(tmp_path):
    run = _make_run(tmp_path, {0: 4, 1: 8})
    (run / "checkpoints" / "spec_0001" / "model_val_best.eqx").write_bytes(b"x")
    cells, missing = T.discover_cells(run, "val_best")
    assert [c.index for c in cells] == [1]
    assert missing == [(0, "deep_3x16", 4)]
    cells, missing = T.discover_cells(run, "val_best", archs=("other",))
    assert cells == [] and missing == []


def test_subset_shades_run_light_to_dark_in_one_hue():
    """Sequential shading: subset size is a magnitude, so the family must read
    as an ordered ramp of the arch's own hue."""
    import matplotlib.colors as mcolors
    base = T.ARCH_COLOR["deep_3x16"]
    shades = T.subset_shades(base, 5)
    assert len(shades) == 5
    assert shades[-1] == mcolors.to_hex(mcolors.to_rgb(base))
    lum = [0.299 * r + 0.587 * g + 0.114 * b
           for r, g, b in (mcolors.to_rgb(s) for s in shades)]
    assert all(a > b for a, b in zip(lum, lum[1:])), lum
    assert T.subset_shades(base, 1) == [mcolors.to_hex(mcolors.to_rgb(base))]


def test_the_parent_curves_are_the_anchors_own_parent():
    """The baseline is ``parents.pbe_fx`` / ``parents.pbe_fc`` (imported from
    pretrain_fx_fc), not the rounded-constant analytic helper: the two differ
    by 4.553e-6 in F_x on this grid, which under the anchor would read as a
    spurious learned correction."""
    from enhancement_factors import pbe_fx_curve
    parents = T.parent_curves()
    assert np.allclose(parents["fx"], T.parent_fx_curve(T.S_GRID))
    gap = float(np.max(np.abs(parents["fx"] - pbe_fx_curve(T.S_GRID))))
    assert 1e-7 < gap < 1e-4, gap
    assert set(parents["fc"]) == set(T.RS_VALUES)
