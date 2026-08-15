#!/usr/bin/env bash
# =============================================================================
# Pull the campaign arms (v4gga + the SCAN-seeded v5 arms) and render every figure --
# per-arm suites plus the merged cross-arm view. Safe to run at ANY level of
# completion: partial grids render with the standard hatched-missing-cell
# marks, arms not yet pulled are skipped, and re-running refreshes in place.
#
#     bash notebooks/analysis/pull_and_plot_v4.sh          # pull + plot
#     bash notebooks/analysis/pull_and_plot_v4.sh --plot-only
#
# Requires $swpath in the environment (set in ~/.bashrc) for the pull step.
# =============================================================================
set -uo pipefail

RESULTS_ROOT="$HOME/Documents/Research/xcquinox-results/runs/dfs_step7"
CLUSTER_ROOT="/gpfs/scratch/awills/xcquinox_runs/dfs_step7"
# v5 era (2026-08-14): the retired v4 mgga arms are neither pulled nor
# merged; the roster = the GGA/rung-3.5 v4 arm + the two SCAN-seeded v5
# arms (merge_v4_arms validates per-arch seed provenance before merging).
ARMS="dfs6311_grid3_v4gga dfs6311_grid3_v5 dfs6311_grid3_v5mgga2"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

if [ "${1:-}" != "--plot-only" ]; then
  if [ -z "${swpath:-}" ]; then
    echo "[pull-v4] FATAL: \$swpath is not set (needed for the pull); use --plot-only to skip"
    exit 1
  fi
  for arm in $ARMS; do
    mkdir -p "$RESULTS_ROOT/$arm"
    # --ignore-missing-args: an arm not yet submitted simply is not there.
    rsync -a --info=stats1 --ignore-missing-args \
        "$swpath":"$CLUSTER_ROOT/$arm/runs" "$RESULTS_ROOT/$arm/" \
      && echo "[pull-v4] pulled $arm" \
      || echo "[pull-v4] NOTE: $arm not pulled (not on cluster yet, or transfer error)"
  done
fi

cd "$REPO"

# --- seed each arm's newest run dir with the local SCAN caches --------------
# The caches are model-independent constants for the production identity
# (basis/grid/DF); the figure loaders search the run-dir root, and the merged
# view propagates them from the arm runs. Canonical local copy mirrors the
# cluster dir /gpfs/scratch/awills/scan_pool_6311ppg3df2pd_g3.
SCAN_CACHE_DIR="$HOME/Documents/Research/xcquinox-results/scan_pool_6311ppg3df2pd_g3"
if compgen -G "$SCAN_CACHE_DIR/scan_pool_*.json" > /dev/null; then
  for arm in $ARMS; do
    newest=$(ls -d "$RESULTS_ROOT/$arm/runs"/run_*/ 2>/dev/null | sort | tail -1)
    [ -n "$newest" ] && cp -f "$SCAN_CACHE_DIR"/scan_pool_*.json "$newest/"
  done
else
  echo "[pull-v4] NOTE: no SCAN caches under $SCAN_CACHE_DIR (SCAN lines will be omitted)"
fi

# --- per-arm figure suites (final-step + val-best variants each) ------------
for arm in $ARMS; do
  if [ -d "$RESULTS_ROOT/$arm/runs" ]; then
    JAX_PLATFORMS=cpu python notebooks/analysis/make_ablation_arch_figure.py \
        --suite --domain dfs_step7 --bases "$arm" \
        --outroot notebooks/analysis \
      || echo "[pull-v4] WARNING: per-arm suite failed for $arm (see above)"
  else
    echo "[pull-v4] skip figures: $arm has no pulled runs"
  fi
done

# --- merged cross-arm view: one directory of renumbered symlinks, then the
#     FULL figure families on it (incl. the SCAN-line set), final-step AND
#     val-best variants -- the one-plot-all-arms primary output ---------------
JAX_PLATFORMS=cpu python notebooks/analysis/merge_v4_arms.py \
    --results-root "$RESULTS_ROOT"
RC=$?
if [ "$RC" -eq 0 ]; then
  JAX_PLATFORMS=cpu python - <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, "notebooks/analysis")
import make_ablation_arch_figure as fig

view = Path.home() / "Documents/Research/xcquinox-results/runs/dfs_step7/merged_v4_arms"
out = Path("notebooks/analysis/figures_dfs6311_v4_merged")
written = fig.build_all(view, out)
written += fig.build_density_energy_figures(view, out)
if fig.figure_cell_coverage(view, eval_subdir="eval_holdout_val_best")["n_cells"]:
    outv = Path("notebooks/analysis/figures_dfs6311_v4_merged_val_best")
    written += fig.build_all(view, outv, eval_subdir="eval_holdout_val_best")
    written += fig.build_density_energy_figures(
        view, outv, eval_subdir="eval_holdout_val_best")
else:
    print("[pull-v4] merged val-best skipped (no val-best eval coverage yet)")
print(f"[pull-v4] merged view: {len(written)} figures -> {out}")
EOF
else
  echo "[pull-v4] merged view skipped (no arm has pulled specs yet)"
fi
echo "[pull-v4] done"
