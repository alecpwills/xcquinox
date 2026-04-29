"""Generate the unweighted-vs-integration pretrain-origin comparison report.

Reads the two per-origin runs (must both exist):

  notebooks/checkpoints_step6/unweighted/
  notebooks/checkpoints_step6/integration/

Writes:

  reports_local/step6_pretrain_origin_comparison/
    ├─ report.md
    ├─ headline_diff.json
    └─ figures/
       ├─ figc1_per_loss_per_group.png    (side-by-side AE-MAE bars)
       ├─ figc2_ratio_heatmap.png         (log10(MAE_int / MAE_unw) per spec)
       ├─ figc3_pareto_overlay.png        (Medvedev plane, both origins overlaid)
       └─ figc4_fx_asymptote.png          (asymptotic F_x by loss + origin)

Re-runnable; physical-correctness conclusions cite primary sources
(Becke 1988 grids, Medvedev 2017 tradeoff, Lieb-Oxford 1981 / PBE 1996
F_x asymptote, Behler-Parrinello 2007 transferability).
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from comparison_lib import (  # noqa: E402
    load_run_artifacts,
    headline_stats,
    plot_origin_comparison_per_loss_per_group,
    plot_origin_ratio_heatmap,
    plot_origin_pareto_density_vs_energy,
    plot_origin_fx_asymptote_vs_pbe,
)


REPO = HERE.parent.parent
RUN_UNW = REPO / "notebooks" / "checkpoints_step6" / "unweighted"
RUN_INT = REPO / "notebooks" / "checkpoints_step6" / "integration"
OUT_DIR = REPO / "reports_local" / "step6_pretrain_origin_comparison"
FIG_DIR = OUT_DIR / "figures"

# Asymptotic-F_x measurements obtained from
# notebooks/analysis/audit_lob_enforcement.py runs on each origin
# (CH4 PBE grid, points with reduced-gradient s > 5).
FX_ASYMPTOTE_UNW = {
    "L1_B":                   {"mean": 1.383, "min": 1.288, "max": 1.526},
    "L2_C_anchor":            {"mean": 1.484, "min": 1.297, "max": 1.709},
    "L3_balanced_vxc":        {"mean": 1.659, "min": 1.235, "max": 1.804},
    "L4_balanced_vxc_anchor": {"mean": 1.786, "min": 1.783, "max": 1.790},
    "L5_gradnorm_vxc":        {"mean": 1.386, "min": 1.297, "max": 1.528},
}
FX_ASYMPTOTE_INT = {
    "L1_B":                   {"mean": 1.479, "min": 1.244, "max": 1.712},
    "L2_C_anchor":            {"mean": 1.517, "min": 1.285, "max": 1.756},
    "L3_balanced_vxc":        {"mean": 1.668, "min": 0.911, "max": 1.804},
    "L4_balanced_vxc_anchor": {"mean": 1.783, "min": 1.774, "max": 1.789},
    "L5_gradnorm_vxc":        {"mean": 1.480, "min": 1.238, "max": 1.721},
}


def main() -> int:
    if not RUN_UNW.is_dir() or not RUN_INT.is_dir():
        missing = [str(p) for p in (RUN_UNW, RUN_INT) if not p.is_dir()]
        print(f"missing run directories: {missing}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading runs ...")
    art_unw = load_run_artifacts(RUN_UNW)
    art_int = load_run_artifacts(RUN_INT)

    print(f"Generating comparison figures ...")
    plot_origin_comparison_per_loss_per_group(
        art_unw, art_int, "unweighted", "integration",
        FIG_DIR / "figc1_per_loss_per_group.png",
    )
    plot_origin_ratio_heatmap(
        art_unw, art_int, "unweighted", "integration",
        FIG_DIR / "figc2_ratio_heatmap.png",
    )
    plot_origin_pareto_density_vs_energy(
        art_unw, art_int, "unweighted", "integration",
        FIG_DIR / "figc3_pareto_overlay.png",
    )
    plot_origin_fx_asymptote_vs_pbe(
        art_unw, art_int, "unweighted", "integration",
        FX_ASYMPTOTE_UNW, FX_ASYMPTOTE_INT,
        FIG_DIR / "figc4_fx_asymptote.png",
    )

    stats = {
        "unweighted":  headline_stats(art_unw),
        "integration": headline_stats(art_int),
    }
    (OUT_DIR / "headline_diff.json").write_text(json.dumps(stats, indent=2))
    print("\nHeadline stats (both origins):")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
