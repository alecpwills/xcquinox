"""The readers of a pulled run against the held-out channel vocabulary.

The figure suite and the command-line tools that read a pulled run each name a
held-out evaluation channel. What is asserted here: the suite renders one
figure set per channel the run actually carries and none for a channel it does
not; a reader handed no channel resolves the reporting one and says so when it
falls back; and the tools' channel tables are the vocabulary's values rather
than literals of their own.

The modules are loaded from their paths rather than imported as packages:
there is no ``__init__.py`` under ``tools``, and every consumer of these
scripts loads them the same way. No figure is drawn -- the suite's builders
are replaced by recorders, so the driver's channel selection is exercised with
no rendering behind it.

Oracles: the vocabulary's tables and suffixes, the directory names the driver
writes, the channel each builder is handed, and the tools' own module-level
tables.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

# A file-backed backend: the figure modules pull in ``matplotlib.pyplot`` at
# import, and a test run has no display.
matplotlib.use("Agg")

import pytest  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


FIG = _load("make_ablation_arch_figure",
            _HERE / "make_ablation_arch_figure.py")


def _vocab():
    """The channel vocabulary, imported inside each test so that a fixture
    fault surfaces as a fault of the code under test rather than an import."""
    from xcquinox.pipeline import holdout_channels
    return holdout_channels


# ---------------------------------------------------------------------------
# The fixture: a pulled-run tree of one evaluated cell, in the layout the
# suite driver walks (<root>/<domain>/<basis>/runs/run_<stamp>/).
# ---------------------------------------------------------------------------

#: after the pretraining-fidelity gate, so the run carries a certificate
_RUN_STAMP = "run_20260924T000000Z"
_BASIS = "svp_grid2"
_ALIAS = "svp"
_DOMAIN = "bh76w411_repr"
#: the manifest holds the stored registry key; the rows and the figures hold
#: the shown name, and the certificate sits under the stored one
_ARCH_STORED = "deep_3x16"
_ARCH_SHOWN = "deep0_3x16"
_SUBSET = 26

#: the channel directories of a run evaluated under the cold-start protocol
#: alone, and of a run of the v7 generation
_COLDSTART_ONLY = ("eval_holdout_coldstart", "eval_holdout_coldstart_val_best")
_V7_CHANNELS = ("eval_holdout", "eval_holdout_best", "eval_holdout_val_best",
                "eval_holdout_converged", "eval_holdout_converged_val_best")


def _reaction_rows():
    """Two held-out reaction rows in the cluster-written (legacy) schema: no
    per-species energies accompany them, so the collector reads them directly
    rather than reconstructing the slice."""
    return [
        {"name": "rxn_a", "pool": "bh76",
         "reaction_energy_ref_kcalmol": 10.0,
         "de_nn_kcalmol": 12.0, "de_pbe_kcalmol": 14.0,
         "abs_error_nn_kcalmol": 2.0, "abs_error_pbe_kcalmol": 4.0,
         "reactants": ["h2"], "products": ["h", "h"]},
        {"name": "rxn_b", "pool": "w4-11",
         "reaction_energy_ref_kcalmol": -5.0,
         "de_nn_kcalmol": -4.0, "de_pbe_kcalmol": -2.0,
         "abs_error_nn_kcalmol": 1.0, "abs_error_pbe_kcalmol": 3.0,
         "reactants": ["oh"], "products": ["o", "h"]},
    ]


_TEST_SET_CSV = ("set,mae_nn_kcalmol,mae_pbe_kcalmol,delta_nn_minus_pbe\n"
                 "test_set_held_out_combined,1.5,3.5,-2.000000\n")


def _make_run(tmp_path, channels, name="results"):
    """A results root holding one run with the named channel directories.

    Returns ``(results_root, run_dir)``. The validation record travels with the
    run: a run whose held-out evaluation used validation-best weights has a
    validation slice to remove from the test columns, and the collector refuses
    to render one whose record is absent.
    """
    root = tmp_path / name
    run = root / _DOMAIN / _BASIS / "runs" / _RUN_STAMP
    spec = run / "checkpoints" / "spec_0000"
    spec.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "width": 4, "n_specs": 1,
        "specs": [{"index": 0,
                   "cell": {"arch": _ARCH_STORED, "subset_size": _SUBSET,
                            "loss": "l2", "metric": "l2",
                            "solver": "full_25"}}]}))
    (run / "resolved_config.yaml").write_text(
        "basis: def2-svp\ndensity_fit: false\n")
    (run / "validation").mkdir()
    (run / "validation" / "val_reactions.json").write_text("[]\n")
    pretrain = run / "pretrain" / _ARCH_STORED
    pretrain.mkdir(parents=True)
    (pretrain / "fidelity_certificate.json").write_text(
        json.dumps({"verdict": "PASS",
                    "summary": {"max_atom_mHa": 0.1,
                                "max_dAE_kcalmol": 0.2}}))
    for channel in channels:
        d = spec / channel
        d.mkdir()
        (d / "per_reaction.json").write_text(json.dumps(_reaction_rows()))
        (d / "test_set.csv").write_text(_TEST_SET_CSV)
    return root, run


_BUILDERS = ("build_all", "build_density_energy_figures",
             "_build_outlier_free_variants", "build_parity_variants",
             "build_per_run_diagnostics", "build_basis_comparison_figures",
             "build_diagnostic_figures")


def _record_builders(monkeypatch, sink):
    """Replace every figure builder the driver calls with a recorder of
    ``(builder, run name, figure dir name, channel)`` that writes nothing."""
    def _make(builder):
        def _record(run, outdir, *args, **kwargs):
            runs = run if isinstance(run, (list, tuple)) else [run]
            sink.append((builder, tuple(Path(r).name for r in runs),
                         Path(outdir).name, kwargs.get("eval_subdir")))
            return []
        return _record

    for builder in _BUILDERS:
        monkeypatch.setattr(FIG, builder, _make(builder))


def test_the_suite_renders_a_set_per_channel_present(tmp_path, monkeypatch,
                                                     capsys):
    """One figure set per channel the run carries, named by that channel's
    suffix, and no set for a channel it does not carry.

    A driver holding a fixed list of channels renders the v7 sets from a
    cold-start run's absent directories (empty figures) and renders no set at
    all from its cold-start pair. Gating every set on cell coverage makes the
    rendered sets a statement about the run rather than about the list, and a
    run carrying no channel at all is a pull to repeat, not a silent empty
    directory -- hence the refusal.
    """
    cold_root, cold_run = _make_run(tmp_path, _COLDSTART_ONLY, name="cold")
    coverage = FIG.figure_cell_coverage(
        cold_run, eval_subdir="eval_holdout_coldstart_val_best")
    assert coverage["n_cells"] == 1, coverage
    assert coverage["archs"] == [_ARCH_SHOWN], coverage
    assert coverage["cells"] == [(_ARCH_SHOWN, _SUBSET)], coverage

    recorded = []
    _record_builders(monkeypatch, recorded)
    FIG.build_bh76w411_suite(results_root=cold_root,
                             outroot=tmp_path / "out_cold",
                             bases=(_BASIS,), domain=_DOMAIN)
    cold_out = capsys.readouterr().out

    hc = _vocab()
    expected = {c: f"figures_{_ALIAS}{hc.figure_suffix(c)}"
                for c in _COLDSTART_ONLY}
    assert {r[2] for r in recorded} == set(expected.values())
    # every builder was handed the channel of the directory it wrote into
    for _builder, _runs, fdir, channel in recorded:
        assert fdir == f"figures_{_ALIAS}{hc.figure_suffix(channel)}"
    # the per-run builders ran for each present channel
    per_run = set(_BUILDERS) - {"build_basis_comparison_figures",
                                "build_diagnostic_figures"}
    assert ({(r[0], r[3]) for r in recorded}
            == {(b, c) for b in per_run for c in _COLDSTART_ONLY})
    # every absent channel is named as skipped
    for channel in set(hc.FIGURE_CHANNELS) - set(_COLDSTART_ONLY):
        assert f"{channel}/" in cold_out, channel

    v7_root, v7_run = _make_run(tmp_path, _V7_CHANNELS, name="v7")
    v7_coverage = FIG.figure_cell_coverage(v7_run,
                                           eval_subdir="eval_holdout_val_best")
    assert v7_coverage["n_cells"] == 1, v7_coverage
    assert v7_coverage["archs"] == [_ARCH_SHOWN], v7_coverage

    recorded.clear()
    FIG.build_bh76w411_suite(results_root=v7_root,
                             outroot=tmp_path / "out_v7",
                             bases=(_BASIS,), domain=_DOMAIN)
    v7_rendered = {r[3] for r in recorded}
    assert v7_rendered == {c for c in hc.FIGURE_CHANNELS if c in _V7_CHANNELS}
    assert hc.REPORTING_CHANNEL not in v7_rendered
    assert hc.CHANNEL_COLDSTART not in v7_rendered
    assert ({r[2] for r in recorded}
            == {f"figures_{_ALIAS}{hc.figure_suffix(c)}" for c in v7_rendered})

    bare_root, _bare_run = _make_run(tmp_path, (), name="bare")
    recorded.clear()
    with pytest.raises(ValueError):
        FIG.build_bh76w411_suite(results_root=bare_root,
                                 outroot=tmp_path / "out_bare",
                                 bases=(_BASIS,), domain=_DOMAIN)
    assert recorded == []


def test_build_all_defaults_to_the_reporting_channel_and_names_a_fallback(
        tmp_path, monkeypatch, capsys):
    """The single-run entry point resolves the reporting channel from the run,
    reports the channel it resolved, and names the reporting channel when it
    falls back to an older one.

    A default of one fixed channel reads the same directory for every
    generation of run, so a cold-start run is scored from a directory it does
    not carry while a v7 run silently keeps the old headline. Printing the
    resolved channel is what makes the fallback visible: the figures of a v7
    run and of a cold-start run are otherwise indistinguishable in the console
    report. An explicit channel is still obeyed.
    """
    seen = []
    monkeypatch.setattr(
        FIG, "_build_all_inner",
        lambda run_dir, outdir, eval_subdir, archs: (
            seen.append(eval_subdir) or []))

    _cold_root, cold_run = _make_run(tmp_path, _COLDSTART_ONLY, name="cold")
    FIG.build_all(cold_run, tmp_path / "out_cold")
    cold_out = capsys.readouterr().out

    _v7_root, v7_run = _make_run(tmp_path, _V7_CHANNELS, name="v7")
    FIG.build_all(v7_run, tmp_path / "out_v7")
    v7_out = capsys.readouterr().out

    FIG.build_all(v7_run, tmp_path / "out_explicit",
                  eval_subdir="eval_holdout_converged")

    hc = _vocab()
    assert seen == [hc.REPORTING_CHANNEL, hc.CHANNEL_VAL_BEST,
                    hc.CHANNEL_CONVERGED]
    assert hc.REPORTING_CHANNEL in cold_out
    # the fallback is stated, naming both the channel read and the absent one
    assert hc.CHANNEL_VAL_BEST in v7_out
    assert hc.REPORTING_CHANNEL in v7_out


def test_the_tool_consumers_take_their_channels_from_the_vocabulary(tmp_path):
    """Every reader of a pulled run holds the vocabulary's channel values.

    The merge decides which specs count as evaluated, the label map tags each
    figure with the checkpoint it was scored from, the validation detector
    decides whether the validation slice must be removed, the NaN backfill and
    the local re-evaluation map checkpoints to channels, the c2 patch applies a
    channel's solver override, and the enhancement-factor tool scores the best
    cell from a channel. Each held its own literal list, so a channel added in
    one place was invisible in the others; a label map that answered
    ``final-step`` for an unknown name mislabelled rather than refused.
    """
    merge = _load("merge_family_runs", _HERE / "merge_family_runs.py")
    backfill = _load("backfill_holdout_nans", _HERE / "backfill_holdout_nans.py")
    reeval = _load("reeval_holdout_fixed", _HERE / "reeval_holdout_fixed.py")
    tfx = _load("trained_fx_fc", _HERE / "trained_fx_fc.py")
    c2 = _load("reeval_c2_patch", _REPO / "hpcjobs" / "reeval_c2_patch.py")

    _cold_root, cold_run = _make_run(tmp_path, _COLDSTART_ONLY, name="cold")
    _v7_root, v7_run = _make_run(tmp_path, _V7_CHANNELS, name="v7")

    # the detector answers on the run's directories, before the vocabulary is
    # consulted: a cold-start run validated, and its record must be honoured
    assert FIG._run_used_validation(cold_run) is True
    assert FIG._run_used_validation(v7_run) is True

    hc = _vocab()

    # the merge counts a spec evaluated on the channels the suite renders
    assert merge._EVAL_CHANNELS == hc.FIGURE_CHANNELS

    # the figure tag names the checkpoint and the protocol; an unknown channel
    # is refused rather than labelled as the final step
    for channel in hc.HOLDOUT_CHANNELS:
        assert FIG._ckpt_label(channel) == hc.CHANNEL_LABEL[channel]
    with pytest.raises(ValueError):
        FIG._ckpt_label("eval_holdout_nope")

    # the backfill re-runs species under the spec's own solver, so it covers
    # the trained-protocol channels only
    assert backfill.CHANNEL_MODELS == {
        c: hc.CHANNEL_MODEL[c] for c in hc.HOLDOUT_CHANNELS
        if hc.CHANNEL_OVERRIDE[c] is None}

    # the local re-evaluation maps a checkpoint name to its trained-protocol
    # channel
    assert reeval.eval_subdir_for("model") == hc.CHANNEL_FINAL
    assert reeval.eval_subdir_for("model_best") == hc.CHANNEL_BEST
    assert reeval.eval_subdir_for("model_val_best") == hc.CHANNEL_VAL_BEST

    # the c2 patch covers every channel and applies each one's override
    assert c2.CHANNELS == hc.HOLDOUT_CHANNELS
    assert dict(c2.CHANNEL_MODEL) == dict(hc.CHANNEL_MODEL)
    from xcquinox.pipeline.solver import (SolverBackend, SolverConfig,
                                          SolverMode)
    sc = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                      max_cycles=3)
    reporting_sc = c2.channel_solver_config(sc, hc.REPORTING_CHANNEL)
    assert reporting_sc.seed_source == "minao"
    assert reporting_sc.max_cycles == 25
    cold_sc = c2.channel_solver_config(sc, hc.CHANNEL_COLDSTART)
    assert cold_sc.seed_source == "minao"
    assert cold_sc.max_cycles == 25
    converged_sc = c2.channel_solver_config(sc, hc.CHANNEL_CONVERGED)
    assert converged_sc.seed_source == "pbe"
    assert converged_sc.max_cycles == 100
    assert c2.channel_solver_config(sc, hc.CHANNEL_FINAL) is sc
    assert c2.channel_solver_config(sc, hc.CHANNEL_VAL_BEST) is sc

    # the enhancement-factor tool scores the best cell from the channel the
    # run carries for each set of weights
    assert tfx.eval_dir_for(cold_run, "val_best") == hc.REPORTING_CHANNEL
    assert tfx.eval_dir_for(cold_run, "final") == hc.CHANNEL_COLDSTART
    assert tfx.eval_dir_for(v7_run, "val_best") == hc.CHANNEL_VAL_BEST
    assert tfx.eval_dir_for(v7_run, "final") == hc.CHANNEL_FINAL


def test_the_validation_detector_keys_on_the_channel_directory(tmp_path):
    """A run whose validation-best channel directory holds a failure record
    and no reaction file still trained with a validation slice.

    The directory is written by the evaluation of the validation-best weights
    whether or not that evaluation finished, and the slice those weights were
    selected against must be removed from the test columns either way; so the
    detector keys on the directory, not on the reaction file inside it. A run
    with no validation-best directory of any protocol did not validate.
    """
    hc = _vocab()
    _root, failed = _make_run(tmp_path, (), name="failed")
    spec = failed / "checkpoints" / "spec_0000"
    (spec / hc.REPORTING_CHANNEL).mkdir()
    (spec / hc.REPORTING_CHANNEL / "failure.json").write_text("{}\n")
    assert FIG._run_used_validation(failed) is True
    assert FIG._val_best_channel_present(failed) == hc.REPORTING_CHANNEL

    _root, bare = _make_run(tmp_path, ("eval_holdout",), name="bare")
    assert FIG._run_used_validation(bare) is False
    assert FIG._val_best_channel_present(bare) is None
