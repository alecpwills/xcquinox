"""Retroactive cold-start channel over a run's COMPLETED specs.

The ``eval_holdout_coldstart`` channel normally rides each spec's own eval
task (``eval_coldstart: true``). Runs evaluated before the channel existed
get it retroactively here: for every spec whose training finished
("completed" = ``model.eqx`` present) and whose channel is not already
written ("done" = ``eval_holdout_coldstart/per_reaction.json`` present),
the standard held-out eval runs once more on the FINAL checkpoint under
the shared cold-start override. Resumable by construction (done specs are
skipped), additive (never touches existing channels, safe beside running
arrays -- pending specs simply are not completed yet), and it runs on the
deployed eval code, so new rows carry the current hold-out rule.

Usage::

    python -m xcquinox.alec.cluster.coldstart_retro <run_dir> [<run_dir> ...]
        [--specs 0 3 7] [--dry-run]
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from typing import List, Optional, Sequence


def spec_status(checkpoint_dir: str) -> str:
    """``pending`` (no final checkpoint) | ``done`` (channel written) |
    ``ready`` (completed, channel missing).

    A cold-start channel evaluated on a species slice is neither: reported
    ``done`` it would stand as this spec's trajectory over a handful of
    workflow-test species, and reported ``ready`` the retro pass would write
    pool rows beside a marker saying otherwise. It is refused before the
    channel is inspected (``eval_holdout.SlicedChannelError``). Imported
    here, as everything else in this module is, so ``python -m`` startup
    does not pull the training package in.
    """
    from xcquinox.alec.eval_holdout import assert_channel_not_sliced
    assert_channel_not_sliced(checkpoint_dir, "eval_holdout_coldstart")
    if not os.path.isfile(os.path.join(checkpoint_dir, "model.eqx")):
        return "pending"
    if os.path.isfile(os.path.join(
            checkpoint_dir, "eval_holdout_coldstart", "per_reaction.json")):
        return "done"
    return "ready"


def coldstart_one_spec(run_dir: str, idx: int) -> str:
    """Run the cold-start pass for one spec; returns the status acted on."""
    from xcquinox.alec.cluster._eval_one_spec import (_checkpoint_dir,
                                                      _load_spec,
                                                      _read_width,
                                                      _run_held_out_eval,
                                                      _spec_path)
    from xcquinox.alec.cluster.grid_config import load_grid_config
    from xcquinox.alec.eval_holdout import coldstart_solver_config

    width = _read_width(run_dir)
    checkpoint_dir = _checkpoint_dir(run_dir, idx, width)
    status = spec_status(checkpoint_dir)
    if status != "ready":
        print(f"[coldstart] spec {idx}: {status} -- skipped", flush=True)
        return status
    cfg = load_grid_config(os.path.join(run_dir, "resolved_config.yaml"))
    training_spec = _load_spec(_spec_path(run_dir, idx, width))
    sc = getattr(training_spec, "solver_config", None)
    if sc is None or getattr(getattr(sc, "mode", None), "value",
                             None) != "full":
        print(f"[coldstart] spec {idx}: no FULL-mode solver_config -- "
              "skipped", flush=True)
        return "skipped-non-full"
    cold_spec = dataclasses.replace(
        training_spec, solver_config=coldstart_solver_config(sc))
    model_path = os.path.join(checkpoint_dir, "model.eqx")
    _run_held_out_eval(run_dir, idx, cfg, checkpoint_dir, model_path,
                       cold_spec, holdout_subdir="eval_holdout_coldstart",
                       coldstart=True)
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--specs", nargs="*", type=int, default=None,
                   help="restrict to these spec indices (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="report statuses only; run nothing")
    args = p.parse_args(argv)
    rc = 0
    for rd in args.run_dirs:
        if not os.path.isdir(os.path.join(rd, "checkpoints")):
            print(f"[coldstart] FATAL: {rd} has no checkpoints/ -- not a "
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
                st = spec_status(_checkpoint_dir(rd, idx, _read_width(rd)))
                print(f"[coldstart] spec {idx}: {st}"
                      + (" -- would run" if st == "ready" else ""),
                      flush=True)
            else:
                st = coldstart_one_spec(rd, idx)
            counts[st] = counts.get(st, 0) + 1
        print(f"[coldstart] {rd}: " + ", ".join(
            f"{v} {k}" for k, v in sorted(counts.items())), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
