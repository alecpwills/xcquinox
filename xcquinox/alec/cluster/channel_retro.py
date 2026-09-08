"""Retroactive evaluation channel over a run's COMPLETED specs.

A channel pass (``eval_holdout_coldstart``, ``eval_holdout_converged``)
normally rides each spec's own eval task (``eval_coldstart: true``,
``eval_converged: true``). Runs evaluated before a channel existed get it
here: for every spec whose training finished ("completed" = ``model.eqx``
present) and whose channel is not fully written, the standard held-out eval
runs once more under the channel's shared solver override
(``eval_holdout.CHANNEL_OVERRIDES``). Resumable by construction (done specs
are skipped; a pass already written is not repeated), additive (never
touches existing channels, safe beside running arrays -- pending specs
simply are not completed yet), and it runs on the deployed eval code, so
new rows carry the current hold-out rule.

Passes per channel: the cold-start channel evaluates the FINAL checkpoint
only (its diagnostic is a trajectory); the converged channel evaluates the
final checkpoint into ``eval_holdout_converged`` and, when
``model_val_best.eqx`` exists, the validation-best one into
``eval_holdout_converged_val_best`` -- the figures' headline is the
val-best channel. "done" means every applicable pass carries its
``per_reaction.json``, so a killed val-best pass leaves the spec ``ready``.

Usage::

    python -m xcquinox.alec.cluster.channel_retro <run_dir> [<run_dir> ...]
        [--channel {coldstart,converged}] [--specs 0 3 7] [--dry-run]
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from typing import List, Optional, Sequence, Tuple

# channel name -> (final-checkpoint subdir, val-best subdir or None)
CHANNEL_DIRS = {
    "coldstart": ("eval_holdout_coldstart", None),
    "converged": ("eval_holdout_converged", "eval_holdout_converged_val_best"),
}


def _passes(checkpoint_dir: str, channel: str) -> List[Tuple[str, str]]:
    """``[(model file, channel subdir)]`` the channel evaluates for this spec:
    the final checkpoint always; the val-best checkpoint when the channel has
    a val-best twin and the checkpoint exists."""
    final_dir, vb_dir = CHANNEL_DIRS[channel]
    out = [("model.eqx", final_dir)]
    if vb_dir and os.path.isfile(os.path.join(checkpoint_dir,
                                              "model_val_best.eqx")):
        out.append(("model_val_best.eqx", vb_dir))
    return out


def _pass_done(checkpoint_dir: str, subdir: str) -> bool:
    return os.path.isfile(os.path.join(checkpoint_dir, subdir,
                                       "per_reaction.json"))


def spec_status(checkpoint_dir: str, channel: str = "coldstart") -> str:
    """``pending`` (no final checkpoint) | ``done`` (every applicable pass
    written) | ``ready`` (completed, at least one pass missing).

    A channel evaluated on a species slice is neither: reported ``done`` it
    would stand as this spec's channel over a handful of workflow-test
    species, and reported ``ready`` the retro pass would write pool rows
    beside a marker saying otherwise. It is refused before the channel is
    inspected (``eval_holdout.SlicedChannelError``). Imported here, as
    everything else in this module is, so ``python -m`` startup does not
    pull the training package in.
    """
    from xcquinox.alec.eval_holdout import assert_channel_not_sliced
    final_dir, vb_dir = CHANNEL_DIRS[channel]
    assert_channel_not_sliced(checkpoint_dir, final_dir)
    if vb_dir:
        assert_channel_not_sliced(checkpoint_dir, vb_dir)
    if not os.path.isfile(os.path.join(checkpoint_dir, "model.eqx")):
        return "pending"
    if all(_pass_done(checkpoint_dir, subdir)
           for _model, subdir in _passes(checkpoint_dir, channel)):
        return "done"
    return "ready"


def retro_one_spec(run_dir: str, idx: int, channel: str = "coldstart") -> str:
    """Run the channel's missing pass(es) for one spec; returns the status
    acted on (``pending`` / ``done`` / ``skipped-non-full`` / ``ran``)."""
    from xcquinox.alec.cluster._eval_one_spec import (_checkpoint_dir,
                                                      _load_spec,
                                                      _read_width,
                                                      _run_held_out_eval,
                                                      _spec_path)
    from xcquinox.alec.cluster.grid_config import load_grid_config
    from xcquinox.alec.eval_holdout import CHANNEL_OVERRIDES

    width = _read_width(run_dir)
    checkpoint_dir = _checkpoint_dir(run_dir, idx, width)
    status = spec_status(checkpoint_dir, channel)
    if status != "ready":
        print(f"[{channel}] spec {idx}: {status} -- skipped", flush=True)
        return status
    cfg = load_grid_config(os.path.join(run_dir, "resolved_config.yaml"))
    training_spec = _load_spec(_spec_path(run_dir, idx, width))
    sc = getattr(training_spec, "solver_config", None)
    if sc is None or getattr(getattr(sc, "mode", None), "value",
                             None) != "full":
        print(f"[{channel}] spec {idx}: no FULL-mode solver_config -- "
              "skipped", flush=True)
        return "skipped-non-full"
    new_spec = dataclasses.replace(
        training_spec, solver_config=CHANNEL_OVERRIDES[channel](sc))
    for model_name, subdir in _passes(checkpoint_dir, channel):
        if _pass_done(checkpoint_dir, subdir):
            print(f"[{channel}] spec {idx}: {subdir} already written -- "
                  "kept", flush=True)
            continue
        _run_held_out_eval(run_dir, idx, cfg, checkpoint_dir,
                           os.path.join(checkpoint_dir, model_name),
                           new_spec, holdout_subdir=subdir, channel=channel)
    return "ran"


def discover_spec_indices(run_dir: str) -> List[int]:
    ck = os.path.join(run_dir, "checkpoints")
    out: List[int] = []
    if not os.path.isdir(ck):
        return out
    for name in sorted(os.listdir(ck)):
        if name.startswith("spec_"):
            try:
                out.append(int(name[len("spec_"):]))
            except ValueError:
                continue
    return out


def main(argv: Optional[Sequence[str]] = None,
         default_channel: str = "coldstart") -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--channel", choices=tuple(CHANNEL_DIRS),
                   default=default_channel,
                   help=f"the channel to write (default {default_channel})")
    p.add_argument("--specs", nargs="*", type=int, default=None,
                   help="restrict to these spec indices (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="report statuses only; run nothing")
    args = p.parse_args(argv)
    channel = args.channel
    rc = 0
    for rd in args.run_dirs:
        if not os.path.isdir(os.path.join(rd, "checkpoints")):
            print(f"[{channel}] FATAL: {rd} has no checkpoints/ -- not a "
                  "run dir", flush=True)
            rc = 1
            continue
        indices = (args.specs if args.specs is not None
                   else discover_spec_indices(rd))
        counts: dict = {}
        for idx in indices:
            if args.dry_run:
                from xcquinox.alec.cluster._eval_one_spec import (
                    _checkpoint_dir, _read_width)
                st = spec_status(_checkpoint_dir(rd, idx, _read_width(rd)),
                                 channel)
                print(f"[{channel}] spec {idx}: {st}"
                      + (" -- would run" if st == "ready" else ""),
                      flush=True)
            else:
                st = retro_one_spec(rd, idx, channel)
            counts[st] = counts.get(st, 0) + 1
        print(f"[{channel}] {rd}: " + ", ".join(
            f"{v} {k}" for k, v in sorted(counts.items())), flush=True)
    return rc


if __name__ == "__main__":
    # The stage's verdict is the status this process hands SLURM, and JAX's
    # atexit teardown can abort the interpreter AFTER main() has returned it
    # (cluster job 2134455). run_and_exit flushes and leaves through
    # os._exit, so the status is the verdict. See cluster/_exit.py.
    from xcquinox.alec.cluster._exit import run_and_exit
    run_and_exit(main)
