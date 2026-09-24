"""The held-out evaluation channels of a trained cell.

A trained cell is scored on the held-out sets through several channels, each
a directory under ``checkpoints/spec_NNNN/`` named here. A channel is one
checkpoint of the cell -- the final step (``model.eqx``), the lowest training
loss (``model_best.eqx``) or the validation-best step (``model_val_best.eqx``)
-- evaluated under one solver protocol: the trained protocol (the spec's own
solver, warm-started from the converged PBE density), the cold start
(:func:`xcquinox.pipeline.eval_holdout.coldstart_solver_config`: the
superposition of atomic densities, a fixed number of cycles) or the converged
SCF (:func:`xcquinox.pipeline.eval_holdout.converged_solver_config`).

The reporting channel is the cold start on the validation-best checkpoint:
the numbers a campaign reports are read from it, and the other channels are
diagnostics. A reader handed no channel takes :func:`resolve_channel`, which
prefers the reporting channel and falls back, for a run that predates it, to
the trained protocol's channel of the same checkpoint and then to the warm
final-step channel.

The module holds the names, the tables keyed by them and the derivations of
those tables, and imports nothing of the training stack, so the eval stage,
the retroactive channel driver, the pull filter's test and every reader of a
pulled run take the names from one place.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

#: the checkpoint files a completed cell may carry
MODEL_FINAL = "model.eqx"
MODEL_BEST = "model_best.eqx"
MODEL_VAL_BEST = "model_val_best.eqx"

#: the solver overrides, the keys of ``eval_holdout.CHANNEL_OVERRIDES``
OVERRIDE_COLDSTART = "coldstart"
OVERRIDE_CONVERGED = "converged"

CHANNEL_FINAL = "eval_holdout"
CHANNEL_BEST = "eval_holdout_best"
CHANNEL_VAL_BEST = "eval_holdout_val_best"
CHANNEL_COLDSTART = "eval_holdout_coldstart"
CHANNEL_COLDSTART_VAL_BEST = "eval_holdout_coldstart_val_best"
CHANNEL_CONVERGED = "eval_holdout_converged"
CHANNEL_CONVERGED_VAL_BEST = "eval_holdout_converged_val_best"

#: every channel the eval stage writes, in the order its passes run
HOLDOUT_CHANNELS: Tuple[str, ...] = (
    CHANNEL_FINAL, CHANNEL_BEST, CHANNEL_VAL_BEST,
    CHANNEL_COLDSTART, CHANNEL_COLDSTART_VAL_BEST,
    CHANNEL_CONVERGED, CHANNEL_CONVERGED_VAL_BEST,
)

#: the channel a campaign's numbers are read from
REPORTING_CHANNEL = CHANNEL_COLDSTART_VAL_BEST

#: the checkpoint each channel evaluates
CHANNEL_MODEL: Dict[str, str] = {
    CHANNEL_FINAL: MODEL_FINAL,
    CHANNEL_BEST: MODEL_BEST,
    CHANNEL_VAL_BEST: MODEL_VAL_BEST,
    CHANNEL_COLDSTART: MODEL_FINAL,
    CHANNEL_COLDSTART_VAL_BEST: MODEL_VAL_BEST,
    CHANNEL_CONVERGED: MODEL_FINAL,
    CHANNEL_CONVERGED_VAL_BEST: MODEL_VAL_BEST,
}

#: the solver override each channel evaluates under; None is the trained
#: protocol
CHANNEL_OVERRIDE: Dict[str, Optional[str]] = {
    CHANNEL_FINAL: None,
    CHANNEL_BEST: None,
    CHANNEL_VAL_BEST: None,
    CHANNEL_COLDSTART: OVERRIDE_COLDSTART,
    CHANNEL_COLDSTART_VAL_BEST: OVERRIDE_COLDSTART,
    CHANNEL_CONVERGED: OVERRIDE_CONVERGED,
    CHANNEL_CONVERGED_VAL_BEST: OVERRIDE_CONVERGED,
}

#: the held-out evaluation of the PRETRAINED network, written by
#: ``hpcjobs/dfs6311_pretrained_holdout.py`` beside the channels above. Not a
#: channel of the eval stage (no pull rule, no figure set of its own), but a
#: directory a figure may be scored from, so it carries a label.
CHANNEL_PRETRAINED = "eval_holdout_pretrained"

#: the tag a figure carries for the channel it was scored from
CHANNEL_LABEL: Dict[str, str] = {
    CHANNEL_FINAL: "final-step",
    CHANNEL_BEST: "train-best",
    CHANNEL_VAL_BEST: "val-best",
    CHANNEL_COLDSTART: "cold-start",
    CHANNEL_COLDSTART_VAL_BEST: "cold-start-val-best",
    CHANNEL_CONVERGED: "converged",
    CHANNEL_CONVERGED_VAL_BEST: "converged-val-best",
    CHANNEL_PRETRAINED: "pretrained",
}

#: the channels the figure suite renders as figure sets, the reporting
#: channel first. The lowest-training-loss checkpoint selects the most overfit
#: step and is drawn from no set of its own.
FIGURE_CHANNELS: Tuple[str, ...] = (
    REPORTING_CHANNEL, CHANNEL_COLDSTART,
    CHANNEL_FINAL, CHANNEL_VAL_BEST,
    CHANNEL_CONVERGED, CHANNEL_CONVERGED_VAL_BEST,
)


def _require_channel(channel: str) -> None:
    if channel not in CHANNEL_MODEL:
        raise ValueError(
            f"unknown held-out channel {channel!r}; the channels are "
            f"{HOLDOUT_CHANNELS}")


def figure_suffix(channel: str) -> str:
    """The suffix of the figure directory rendered from ``channel``: none for
    the warm final-step channel (``figures_<alias>/``), the channel's own
    tail otherwise (``figures_<alias>_coldstart_val_best/``)."""
    _require_channel(channel)
    return channel[len(CHANNEL_FINAL):]


def channels_with_model(model: str) -> Tuple[str, ...]:
    """The channels evaluating checkpoint ``model``, in declared order."""
    return tuple(c for c in HOLDOUT_CHANNELS if CHANNEL_MODEL[c] == model)


#: the channels evaluating the validation-best checkpoint
VAL_BEST_CHANNELS: Tuple[str, ...] = channels_with_model(MODEL_VAL_BEST)


def channel_of(model: str, override: Optional[str]) -> str:
    """The channel evaluating checkpoint ``model`` under ``override``.

    Raises ``ValueError`` when no channel does: the lowest-training-loss
    checkpoint is evaluated under the trained protocol only."""
    for channel in HOLDOUT_CHANNELS:
        if (CHANNEL_MODEL[channel] == model
                and CHANNEL_OVERRIDE[channel] == override):
            return channel
    raise ValueError(
        f"no held-out channel evaluates {model!r} under override "
        f"{override!r}")


def val_best_twin(channel: str) -> str:
    """The channel evaluating the validation-best checkpoint under the
    protocol of ``channel``, a final-step channel; ``ValueError`` for any
    other channel."""
    _require_channel(channel)
    if CHANNEL_MODEL[channel] != MODEL_FINAL:
        raise ValueError(
            f"{channel!r} is not a final-step channel; a protocol's "
            "validation-best channel is named from its final-step one")
    return channel_of(MODEL_VAL_BEST, CHANNEL_OVERRIDE[channel])


def override_passes(override: str) -> Tuple[Tuple[str, str], ...]:
    """``((checkpoint file, channel), ...)`` of the passes an override runs:
    the final checkpoint first, then the validation-best one."""
    final = channel_of(MODEL_FINAL, override)
    return ((MODEL_FINAL, final), (MODEL_VAL_BEST, val_best_twin(final)))


def _spec_dirs(run_dir: Union[str, Path]) -> List[str]:
    ck = os.path.join(str(run_dir), "checkpoints")
    if not os.path.isdir(ck):
        return []
    return sorted(os.path.join(ck, name) for name in os.listdir(ck)
                  if name.startswith("spec_")
                  and os.path.isdir(os.path.join(ck, name)))


def _evaluated_anywhere(run_dir: Union[str, Path], channel: str) -> bool:
    return any(os.path.isfile(os.path.join(sd, channel, "per_reaction.json"))
               for sd in _spec_dirs(run_dir))


def resolve_channel(run_dir: Union[str, Path],
                    model: str = MODEL_VAL_BEST) -> str:
    """The channel a reader of ``run_dir`` takes for checkpoint ``model`` when
    handed none.

    The reporting protocol's channel for that checkpoint when any spec of the
    run carries its ``per_reaction.json``; else the trained protocol's channel
    of the same checkpoint (the v7 headline for the validation-best weights);
    else the warm final-step channel. The last candidate is returned when the
    run carries none of them, so a reader of an empty run reports the warm
    channel with no cells rather than raising. ``model`` is
    :data:`MODEL_VAL_BEST` (the default, which gives the reporting channel
    itself) or :data:`MODEL_FINAL`; the lowest-training-loss checkpoint has
    one channel and is named directly. Spec directories are followed through
    symlinks, so a merged view resolves as the runs it links do."""
    candidates = [channel_of(model, OVERRIDE_COLDSTART),
                  channel_of(model, None)]
    if CHANNEL_FINAL not in candidates:
        candidates.append(CHANNEL_FINAL)
    for channel in candidates:
        if _evaluated_anywhere(run_dir, channel):
            return channel
    return candidates[-1]
