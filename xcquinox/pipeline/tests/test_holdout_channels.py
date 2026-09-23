"""The held-out channel vocabulary (``xcquinox.pipeline.holdout_channels``).

The seven held-out evaluation channels are named in one module, and every
consumer of a channel name takes it from there. What is asserted here: the
reporting channel is the cold-start protocol on the validation-best
checkpoint; the per-channel model, override and label tables cover exactly
the declared channels; the derivations (``channel_of``, ``val_best_twin``,
``channels_with_model``, ``override_passes``, ``figure_suffix``) agree with
those tables and refuse a name outside them; the resolver prefers the
reporting channel and falls back in the declared order over a run's own
directories; the pipeline's consumers hold the vocabulary's values rather
than their own literals; and the packaged pull filter carries a rule pair for
every channel, so a channel the eval stage writes cannot be one the default
pull drops.

Oracles: the vocabulary's own tables, the consumers' module-level tables, and
the ``+ /checkpoints/spec_*/<channel>/***`` rules of the packaged filter.
"""
from __future__ import annotations

import re

import pytest


def _vocab():
    """The channel vocabulary. Imported inside each test rather than at module
    scope, so that every test states its own requirement on the module."""
    from xcquinox.pipeline import holdout_channels
    return holdout_channels


def _write_channels(run_dir, channels, spec="spec_0000"):
    """Materialize ``checkpoints/<spec>/<channel>/per_reaction.json`` for each
    named channel under ``run_dir``; returns the spec directory."""
    spec_dir = run_dir / "checkpoints" / spec
    for channel in channels:
        (spec_dir / channel).mkdir(parents=True, exist_ok=True)
        (spec_dir / channel / "per_reaction.json").write_text("[]\n")
    spec_dir.mkdir(parents=True, exist_ok=True)
    return spec_dir


def test_the_reporting_channel_is_the_cold_start_on_the_validation_best_checkpoint():
    """The reporting channel evaluates ``model_val_best.eqx`` under the
    cold-start override, and the tables that describe the channels cover the
    declared set exactly.

    The two halves of the name are both load-bearing: the checkpoint is the
    validation-best one (not the final step, whose number the v7 figures no
    longer headline) and the protocol is the cold start (not the trained
    protocol, whose seed the evaluation must not inherit). A vocabulary that
    named either half differently would leave every consumer reporting a
    different quantity under the same word.

    The derivations are checked against the tables rather than restated: a
    table and a derivation that disagree are two vocabularies.
    """
    hc = _vocab()

    assert hc.REPORTING_CHANNEL == hc.CHANNEL_COLDSTART_VAL_BEST
    assert hc.REPORTING_CHANNEL in hc.HOLDOUT_CHANNELS
    assert hc.CHANNEL_MODEL[hc.REPORTING_CHANNEL] == hc.MODEL_VAL_BEST
    assert hc.CHANNEL_OVERRIDE[hc.REPORTING_CHANNEL] == hc.OVERRIDE_COLDSTART

    # no channel named twice, and each table covers exactly the declared set;
    # the label table also names the pretrained network's evaluation
    # directory, which a figure may be scored from though no eval-stage pass
    # writes it
    assert len(set(hc.HOLDOUT_CHANNELS)) == len(hc.HOLDOUT_CHANNELS)
    declared = set(hc.HOLDOUT_CHANNELS)
    for table in (hc.CHANNEL_MODEL, hc.CHANNEL_OVERRIDE):
        assert set(table) == declared
    assert set(hc.CHANNEL_LABEL) == declared | {hc.CHANNEL_PRETRAINED}
    assert hc.CHANNEL_PRETRAINED not in hc.FIGURE_CHANNELS

    # the model each channel evaluates, stated once
    assert hc.CHANNEL_MODEL[hc.CHANNEL_FINAL] == hc.MODEL_FINAL
    assert hc.CHANNEL_MODEL[hc.CHANNEL_BEST] == hc.MODEL_BEST
    assert hc.CHANNEL_MODEL[hc.CHANNEL_VAL_BEST] == hc.MODEL_VAL_BEST
    assert hc.CHANNEL_MODEL[hc.CHANNEL_COLDSTART] == hc.MODEL_FINAL
    assert hc.CHANNEL_MODEL[hc.CHANNEL_CONVERGED] == hc.MODEL_FINAL
    assert hc.CHANNEL_MODEL[hc.CHANNEL_CONVERGED_VAL_BEST] == hc.MODEL_VAL_BEST

    # the override each channel runs under; the trained protocol is None
    assert hc.CHANNEL_OVERRIDE[hc.CHANNEL_FINAL] is None
    assert hc.CHANNEL_OVERRIDE[hc.CHANNEL_BEST] is None
    assert hc.CHANNEL_OVERRIDE[hc.CHANNEL_VAL_BEST] is None
    assert hc.CHANNEL_OVERRIDE[hc.CHANNEL_COLDSTART] == hc.OVERRIDE_COLDSTART
    assert hc.CHANNEL_OVERRIDE[hc.CHANNEL_CONVERGED] == hc.OVERRIDE_CONVERGED
    assert (hc.CHANNEL_OVERRIDE[hc.CHANNEL_CONVERGED_VAL_BEST]
            == hc.OVERRIDE_CONVERGED)

    # channel_of inverts the two tables; the training-loss-best checkpoint is
    # written under the trained protocol alone, so no override evaluates it
    for channel in hc.HOLDOUT_CHANNELS:
        assert hc.channel_of(hc.CHANNEL_MODEL[channel],
                             hc.CHANNEL_OVERRIDE[channel]) == channel
    with pytest.raises(ValueError):
        hc.channel_of(hc.MODEL_BEST, hc.OVERRIDE_COLDSTART)
    with pytest.raises(ValueError):
        hc.channel_of(hc.MODEL_BEST, hc.OVERRIDE_CONVERGED)

    # the val-best channels, in the declared order
    assert hc.channels_with_model(hc.MODEL_VAL_BEST) == hc.VAL_BEST_CHANNELS
    assert hc.VAL_BEST_CHANNELS == (hc.CHANNEL_VAL_BEST,
                                    hc.REPORTING_CHANNEL,
                                    hc.CHANNEL_CONVERGED_VAL_BEST)
    assert hc.channels_with_model(hc.MODEL_BEST) == (hc.CHANNEL_BEST,)

    # each protocol's val-best twin, from its final-step channel
    assert hc.val_best_twin(hc.CHANNEL_COLDSTART) == hc.REPORTING_CHANNEL
    assert hc.val_best_twin(hc.CHANNEL_FINAL) == hc.CHANNEL_VAL_BEST
    assert (hc.val_best_twin(hc.CHANNEL_CONVERGED)
            == hc.CHANNEL_CONVERGED_VAL_BEST)
    for channel in (hc.REPORTING_CHANNEL, hc.CHANNEL_VAL_BEST, hc.CHANNEL_BEST):
        with pytest.raises(ValueError):
            hc.val_best_twin(channel)

    # the figure-set suffix: the warm final set keeps the unsuffixed name
    assert hc.figure_suffix(hc.CHANNEL_FINAL) == ""
    assert hc.figure_suffix(hc.CHANNEL_VAL_BEST) == "_val_best"
    assert hc.figure_suffix(hc.CHANNEL_COLDSTART) == "_coldstart"
    assert hc.figure_suffix(hc.REPORTING_CHANNEL) == "_coldstart_val_best"
    assert hc.figure_suffix(hc.CHANNEL_CONVERGED) == "_converged"
    assert (hc.figure_suffix(hc.CHANNEL_CONVERGED_VAL_BEST)
            == "_converged_val_best")
    with pytest.raises(ValueError):
        hc.figure_suffix("eval_holdout_nope")

    # the labels a figure tag is drawn from
    assert hc.CHANNEL_LABEL[hc.CHANNEL_FINAL] == "final-step"
    assert hc.CHANNEL_LABEL[hc.CHANNEL_BEST] == "train-best"
    assert hc.CHANNEL_LABEL[hc.CHANNEL_VAL_BEST] == "val-best"
    assert hc.CHANNEL_LABEL[hc.CHANNEL_COLDSTART] == "cold-start"
    assert hc.CHANNEL_LABEL[hc.REPORTING_CHANNEL] == "cold-start-val-best"
    assert hc.CHANNEL_LABEL[hc.CHANNEL_CONVERGED] == "converged"
    assert (hc.CHANNEL_LABEL[hc.CHANNEL_CONVERGED_VAL_BEST]
            == "converged-val-best")

    # the two passes an override runs, final checkpoint first
    assert hc.override_passes(hc.OVERRIDE_COLDSTART) == (
        (hc.MODEL_FINAL, hc.CHANNEL_COLDSTART),
        (hc.MODEL_VAL_BEST, hc.REPORTING_CHANNEL))
    assert hc.override_passes(hc.OVERRIDE_CONVERGED) == (
        (hc.MODEL_FINAL, hc.CHANNEL_CONVERGED),
        (hc.MODEL_VAL_BEST, hc.CHANNEL_CONVERGED_VAL_BEST))

    # the figure sets: the reporting channel first, the training-loss-best
    # channel drawn from no set of its own
    assert hc.FIGURE_CHANNELS[0] == hc.REPORTING_CHANNEL
    assert hc.CHANNEL_BEST not in hc.FIGURE_CHANNELS
    assert set(hc.FIGURE_CHANNELS) <= declared
    assert len(set(hc.FIGURE_CHANNELS)) == len(hc.FIGURE_CHANNELS)


def test_resolve_channel_prefers_the_reporting_channel_and_falls_back_in_order(
        tmp_path):
    """A reader handed no channel gets the reporting one where it exists, the
    trained protocol's channel for the same checkpoint where it does not, and
    the warm final-step channel for a run that predates validation.

    The fallback is what lets one default read every generation of pulled run:
    a v7 run carries no cold-start val-best directory, and a pre-validation run
    carries no val-best directory at all. Resolution is by the directories the
    run actually holds, so a run that carries the reporting channel is never
    read on an older one, and a merged family view -- whose spec directories
    are symlinks into the source runs -- resolves like the runs it links.
    """
    hc = _vocab()

    reporting = tmp_path / "reporting"
    _write_channels(reporting, ("eval_holdout", "eval_holdout_val_best",
                                "eval_holdout_coldstart",
                                "eval_holdout_coldstart_val_best"))
    assert hc.resolve_channel(reporting) == hc.REPORTING_CHANNEL
    assert hc.resolve_channel(reporting, hc.MODEL_FINAL) == hc.CHANNEL_COLDSTART

    v7 = tmp_path / "v7"
    _write_channels(v7, ("eval_holdout", "eval_holdout_best",
                         "eval_holdout_val_best", "eval_holdout_converged",
                         "eval_holdout_converged_val_best"))
    assert hc.resolve_channel(v7) == hc.CHANNEL_VAL_BEST
    assert hc.resolve_channel(v7, hc.MODEL_FINAL) == hc.CHANNEL_FINAL

    pre_validation = tmp_path / "pre_validation"
    _write_channels(pre_validation, ("eval_holdout",))
    assert hc.resolve_channel(pre_validation) == hc.CHANNEL_FINAL
    assert (hc.resolve_channel(pre_validation, hc.MODEL_FINAL)
            == hc.CHANNEL_FINAL)

    # nothing evaluated at all: the last candidate, so a reader on an empty
    # run reports the warm channel rather than raising
    empty = tmp_path / "empty"
    (empty / "checkpoints").mkdir(parents=True)
    assert hc.resolve_channel(empty) == hc.CHANNEL_FINAL
    assert hc.resolve_channel(tmp_path / "absent") == hc.CHANNEL_FINAL

    # a directory without the reaction file is not an evaluated channel
    hollow = tmp_path / "hollow"
    (hollow / "checkpoints" / "spec_0000"
     / "eval_holdout_coldstart_val_best").mkdir(parents=True)
    _write_channels(hollow, ("eval_holdout", "eval_holdout_val_best"))
    assert hc.resolve_channel(hollow) == hc.CHANNEL_VAL_BEST

    # a merged view: the spec directory is a symlink into a source run
    view = tmp_path / "view"
    (view / "checkpoints").mkdir(parents=True)
    (view / "checkpoints" / "spec_0000").symlink_to(
        reporting / "checkpoints" / "spec_0000", target_is_directory=True)
    assert hc.resolve_channel(view) == hc.REPORTING_CHANNEL


def test_the_pipeline_consumers_take_their_channels_from_the_vocabulary():
    """The retro driver, the verbatim re-finalizer and the override table hold
    the vocabulary's values, not literals of their own.

    Each of the three is a place a channel name was spelled out: the retro
    driver's per-override pass table, the re-finalizer's list of channels to
    rewrite, and the solver-override table whose keys are the override names.
    A literal that drifts from the vocabulary produces a channel written by one
    stage and ignored by the next, which is invisible until a figure set is
    missing.
    """
    hc = _vocab()
    from xcquinox.pipeline import eval_holdout, refinalize_verbatim
    from xcquinox.pipeline.cluster import channel_retro

    expected_dirs = {
        override: (hc.channel_of(hc.MODEL_FINAL, override),
                   hc.channel_of(hc.MODEL_VAL_BEST, override))
        for override in (hc.OVERRIDE_COLDSTART, hc.OVERRIDE_CONVERGED)}
    assert channel_retro.CHANNEL_DIRS == expected_dirs
    assert (channel_retro.CHANNEL_DIRS[hc.OVERRIDE_COLDSTART][1]
            == hc.REPORTING_CHANNEL)

    assert refinalize_verbatim.CHANNELS == hc.HOLDOUT_CHANNELS

    assert set(eval_holdout.CHANNEL_OVERRIDES) == {hc.OVERRIDE_COLDSTART,
                                                  hc.OVERRIDE_CONVERGED}


def test_the_summaries_filter_pulls_every_channel():
    """The packaged pull filter carries an include pair for every channel.

    The summaries profile is the default pull, so a channel with no rule is
    written on the cluster and never reaches a local figure. The rules are read
    as a set and compared with the declared channels in both directions: a
    missing rule drops a channel, and a rule for a name no longer declared
    pulls a directory nothing reads.
    """
    hc = _vocab()
    from pathlib import Path

    import xcquinox.pipeline.cluster as cluster_pkg

    filter_path = (Path(cluster_pkg.__file__).resolve().parent / "filters"
                   / "summaries.filter")
    rule = re.compile(r"^\+\s+/checkpoints/spec_\*/(eval_holdout[^/]*)/\*\*\*\s*$")
    named = {m.group(1) for m in
             (rule.match(line) for line in
              filter_path.read_text().splitlines()) if m}
    assert named == set(hc.HOLDOUT_CHANNELS)

    # each terminal rule is preceded by the directory rule rsync needs to
    # descend, so the pair travels together
    text = filter_path.read_text()
    for channel in hc.HOLDOUT_CHANNELS:
        assert f"+ /checkpoints/spec_*/{channel}/\n" in text, channel
