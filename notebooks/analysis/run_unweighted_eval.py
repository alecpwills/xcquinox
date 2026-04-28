"""Generate the step-6 unweighted-pretrain analysis report + new comparison plots.

Reads:
    notebooks/checkpoints_step6/unweighted/{eval_df,baseline_df,
    transfer_primary_df,transfer_secondary_df}.{parquet,csv}

Writes:
    reports_local/step6_unweighted_analysis/
      ├─ report.md                    (markdown analysis with figure refs)
      ├─ headline_stats.json          (machine-readable summary)
      └─ figures/
         ├─ fig1_baseline_reduction.png
         ├─ fig2_density_vs_energy_tradeoff.png
         ├─ fig3_in_dist_vs_transfer.png
         ├─ fig4_loss_strategy_heatmap.png
         └─ fig5_arch_comparison.png

Designed to be re-runnable for the integration-pretrain origin once
that sweep finishes -- swap RUN_DIR and RUN_LABEL to ``integration``
and the same five figures + report land under
``reports_local/step6_integration_analysis/``.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

# Make the helper lib importable regardless of CWD.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from comparison_lib import (  # noqa: E402
    load_run_artifacts,
    headline_stats,
    plot_baseline_reduction,
    plot_baseline_reduction_transfer,
    plot_density_vs_energy_tradeoff,
    plot_in_dist_vs_transfer,
    plot_loss_strategy_heatmap,
    plot_arch_comparison,
)


REPO = HERE.parent.parent  # /home/awills/Documents/Research/xcquinox
RUN_DIR = REPO / "notebooks" / "checkpoints_step6" / "unweighted"
RUN_LABEL = "unweighted"
OUT_DIR = REPO / "reports_local" / "step6_unweighted_analysis"
FIG_DIR = OUT_DIR / "figures"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading artifacts from {RUN_DIR} ...")
    art = load_run_artifacts(RUN_DIR)
    for k, v in art.items():
        print(f"  {k}: {None if v is None else v.shape}")

    print(f"\nGenerating figures into {FIG_DIR} ...")
    plot_baseline_reduction(art,           FIG_DIR / "fig1_baseline_reduction.png",          RUN_LABEL)
    plot_baseline_reduction_transfer(art,  FIG_DIR / "fig1b_baseline_reduction_transfer.png", RUN_LABEL)
    plot_density_vs_energy_tradeoff(art,   FIG_DIR / "fig2_density_vs_energy_tradeoff.png",  RUN_LABEL)
    plot_in_dist_vs_transfer(art,          FIG_DIR / "fig3_in_dist_vs_transfer.png",        RUN_LABEL)
    plot_loss_strategy_heatmap(art,        FIG_DIR / "fig4_loss_strategy_heatmap.png",      RUN_LABEL)
    plot_arch_comparison(art,              FIG_DIR / "fig5_arch_comparison.png",            RUN_LABEL)

    stats = headline_stats(art)
    (OUT_DIR / "headline_stats.json").write_text(json.dumps(stats, indent=2))
    print("\nHeadline stats:")
    print(json.dumps(stats, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
