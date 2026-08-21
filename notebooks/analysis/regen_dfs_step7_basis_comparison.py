#!/usr/bin/env python
"""Guarded launcher for the dfs_step7 def2-svp vs def2-tzvpd+DF basis comparison.

The cross-basis comparison needs >=2 bases with held-out eval coverage; the suite
silently skips it otherwise (make_ablation_arch_figure.py, ``len(ordered_runs) < 2``).
As of this writing only ``dfs_step7/svp_grid2`` is pulled locally -- the
``dfs_step7/tzvpd_grid2_df`` run was resubmitted and not yet synced. This launcher
refuses with a clear message (exit 1) until BOTH bases have eval data, and otherwise
invokes the existing (tested) suite, which writes
``figures_dfs_step7_basis_comparison/`` (+ ``_best``) under ``--outroot``.

No new figure logic lives here -- only the presence guard + a thin call-through, so the
figure rendering stays covered by the existing make_ablation_arch_figure tests.

Run it after pulling the tzvpd run:
    python notebooks/analysis/regen_dfs_step7_basis_comparison.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DOMAIN = "dfs_step7"
BASES = ("svp_grid2", "tzvpd_grid2_df")
_DEFAULT_RESULTS_ROOT = "~/Documents/Research/xcquinox-results/runs"
_SUITE_SCRIPT = Path(__file__).resolve().parent / "make_ablation_arch_figure.py"


def basis_has_eval(results_root: Path, basis: str) -> bool:
    """True if the newest-or-any run for ``<results_root>/dfs_step7/<basis>`` has at
    least one ``checkpoints/spec_*/eval_holdout/per_reaction.json`` (held-out coverage)."""
    # Imported at the guard, not at module scope: this script has no
    # other use for the training package and importing it here would
    # pull jax / pyscf / equinox into every invocation.
    from xcquinox.alec.eval_holdout import assert_channel_not_sliced
    runs = Path(results_root) / DOMAIN / basis / "runs"
    if not runs.is_dir():
        return False
    for run_dir in sorted(runs.glob("run_*"), reverse=True):
        if not run_dir.is_dir():
            continue
        # Every spec is checked before the verdict, not just up to the first
        # hit: a six-species workflow slice is not held-out coverage of the
        # pool, and this predicate is what releases the figure suite.
        found = False
        for sd in sorted((run_dir / "checkpoints").glob("spec_*")):
            assert_channel_not_sliced(sd, "eval_holdout")
            if (sd / "eval_holdout" / "per_reaction.json").is_file():
                found = True
        if found:
            return True
    return False


def missing_bases(results_root: Path) -> list[str]:
    """Bases (in canonical order) that lack held-out eval data under ``results_root``."""
    return [b for b in BASES if not basis_has_eval(results_root, b)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate the dfs_step7 svp-vs-tzvpd basis comparison (guarded).")
    ap.add_argument("--results-root", default=_DEFAULT_RESULTS_ROOT,
                    help=f"results runs root (default: {_DEFAULT_RESULTS_ROOT})")
    ap.add_argument("--outroot", default=str(_SUITE_SCRIPT.parent),
                    help="dir the figures_* dirs are written under (default: next to this script)")
    args = ap.parse_args(argv)

    results_root = Path(args.results_root).expanduser()
    missing = missing_bases(results_root)
    if missing:
        print(f"[regen] dfs_step7 basis comparison NOT generated -- missing held-out eval "
              f"data for: {', '.join(missing)}", file=sys.stderr)
        print(f"[regen]   looked under {results_root}/{DOMAIN}/<basis>/runs/", file=sys.stderr)
        print("[regen]   pull the resubmitted tzvpd_grid2_df run, then re-run this script.",
              file=sys.stderr)
        return 1

    cmd = [sys.executable, str(_SUITE_SCRIPT), "--suite",
           "--domain", DOMAIN, "--bases", ",".join(BASES),
           "--results-root", str(results_root), "--outroot", args.outroot]
    print(f"[regen] both bases present; running: {' '.join(cmd)}")
    print(f"[regen] comparison -> {args.outroot}/figures_{DOMAIN}_basis_comparison/ (+ _best)")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
