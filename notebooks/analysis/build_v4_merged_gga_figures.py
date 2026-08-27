#!/usr/bin/env python
"""Merged-v4 figure suite restricted to the PBE-parented architectures.

The merged cross-arm view (``merged_v4_arms``) carries eleven architectures.
Five are rendered here -- ``deep_3x16``, ``deep_attn_3x16``,
``deep_cusp_3x16``, ``deep_rung35_3x16`` and ``deep_rung35_attn_3x16``: the
GGA and rung-3.5 families, whose networks were pretrained to PBE and whose
held-out numbers are therefore comparable with the PBE baselines these figures
draw.

Withheld, and why:

* the five meta-GGA architectures (``deep_mgga_3x16``,
  ``deep_mgga_attn_3x16``, ``deep_cusp_mgga_3x16``, ``deep_rung35_mgga_3x16``,
  ``deep_rung35ms_mgga_3x16``). In this run the open-shell exchange spin
  scaling evaluated the iso-orbital indicator on the total density instead of
  per spin channel, which puts every open-shell meta-GGA number kcal/mol per
  atom off its own parent functional; those cells are not readable against the
  parent baselines and are not citable.
* ``deep_rung35ms_3x16``, for legibility (it also carries the NaN-inclusive
  cells).

With no meta-GGA architecture rendered, no architecture on these figures is
parented by SCAN, so the SCAN comparator is withdrawn rather than drawn beside
bars it does not describe (``scan_comparator_applies``).

Usage (from the repository root)::

    python notebooks/analysis/build_v4_merged_gga_figures.py

Writes ``figures_dfs6311_v4_merged_val_best_gga/`` next to this file; the
unrestricted directories are not touched. Figures are regenerated outputs and
are not version-controlled.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_ablation_arch_figure as fig  # noqa: E402

#: The PBE-parented architectures of the merged v4 view, in ladder order.
PBE_PARENTED_ARCHS = ("deep_3x16", "deep_attn_3x16", "deep_cusp_3x16",
                      "deep_rung35_3x16", "deep_rung35_attn_3x16")

_DEFAULT_RUN = (Path.home() / "Documents/Research/xcquinox-results/runs"
                / "dfs_step7" / "merged_v4_arms")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=str(_DEFAULT_RUN),
                   help="merged view directory (default: %(default)s)")
    p.add_argument("--eval-subdir", default="eval_holdout_val_best",
                   help="checkpoint variant to score (default: %(default)s)")
    p.add_argument("--outdir", default=None,
                   help="output directory (default: "
                        "figures_dfs6311_v4_merged_val_best_gga/ next to this "
                        "script, or _gga for the final-step channel)")
    p.add_argument("--archs", default=",".join(PBE_PARENTED_ARCHS),
                   help="comma-separated architectures to render "
                        "(default: %(default)s)")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    archs = tuple(a.strip() for a in args.archs.split(",") if a.strip())
    here = Path(__file__).resolve().parent
    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        suffix = ("_val_best_gga" if args.eval_subdir.endswith("val_best")
                  else "_gga")
        outdir = here / f"figures_dfs6311_v4_merged{suffix}"

    cov = fig.figure_cell_coverage(run_dir, eval_subdir=args.eval_subdir,
                                   archs=archs)
    print(f"run_dir: {run_dir}")
    print(f"eval:    {args.eval_subdir}")
    print(f"cells:   {cov['n_cells']}  archs={cov['archs']}  "
          f"subsets={cov['subsets']}")
    if cov["archs_missing"]:
        print(f"   (no held-out cell yet: {cov['archs_missing']})")

    written = fig.build_all(run_dir, outdir, eval_subdir=args.eval_subdir,
                            archs=archs)
    written += fig.build_density_energy_figures(
        run_dir, outdir, eval_subdir=args.eval_subdir, archs=archs)
    for pth in written:
        print(f"  wrote {pth}")
    print(f"{len(written)} figures -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
