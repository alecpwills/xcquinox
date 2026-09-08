"""Retroactive cold-start channel over a run's COMPLETED specs.

The cold-start entry point of the generalized channel tool
(:mod:`xcquinox.alec.cluster.channel_retro`), kept so the deployed job
script (``hpcjobs/coldstart_holdout.sbatch``) and its tests keep working:
``spec_status`` and ``coldstart_one_spec`` are the generalized functions with
the channel fixed to ``coldstart``; ``main`` is the generalized entry point
with ``coldstart`` as its default channel (an explicit ``--channel`` on the
command line is honoured). See ``channel_retro`` for the semantics
(resumable, additive, the deployed eval code).

Usage::

    python -m xcquinox.alec.cluster.coldstart_retro <run_dir> [<run_dir> ...]
        [--specs 0 3 7] [--dry-run]
"""
from __future__ import annotations

from typing import Optional, Sequence

from xcquinox.alec.cluster.channel_retro import (  # noqa: F401 (re-export)
    discover_spec_indices,
)
from xcquinox.alec.cluster import channel_retro as _cr


def spec_status(checkpoint_dir: str) -> str:
    """``pending`` | ``done`` | ``ready`` for the cold-start channel."""
    return _cr.spec_status(checkpoint_dir, "coldstart")


def coldstart_one_spec(run_dir: str, idx: int) -> str:
    """Run the cold-start pass for one spec; returns the status acted on."""
    return _cr.retro_one_spec(run_dir, idx, "coldstart")


def main(argv: Optional[Sequence[str]] = None) -> int:
    return _cr.main(argv, default_channel="coldstart")


if __name__ == "__main__":
    from xcquinox.alec.cluster._exit import run_and_exit
    run_and_exit(main)
