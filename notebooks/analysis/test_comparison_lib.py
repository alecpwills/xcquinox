"""Unit tests for ``notebooks.analysis.comparison_lib``.

Locks in the contracts the report depends on:
  - data-loader handles parquet OR csv per artifact
  - mae_of returns mean(|value|) per group
  - trained_molecules_only drops atom rows
  - all five plot functions run without raising on real data
  - headline_stats produces the exact keys the markdown report references
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from comparison_lib import (  # noqa: E402
    KCAL_PER_HA,
    CHEMICAL_ACCURACY_KCALMOL,
    LOSS_DISPLAY_ORDER,
    LOSS_FAMILY,
    load_run_artifacts,
    trained_molecules_only,
    mae_of,
    headline_stats,
    plot_baseline_reduction,
    plot_baseline_reduction_transfer,
    plot_density_vs_energy_tradeoff,
    plot_in_dist_vs_transfer,
    plot_loss_strategy_heatmap,
    plot_arch_comparison,
)


REPO = HERE.parent.parent
UNWEIGHTED_RUN_DIR = REPO / "notebooks" / "checkpoints_step6" / "unweighted"


def test_constants_match_codebase():
    """KCAL_PER_HA must match alec.config / W4-11 conventions to 12 digits."""
    assert abs(KCAL_PER_HA - 627.5094740631) < 1e-12
    assert CHEMICAL_ACCURACY_KCALMOL == 1.0
    assert LOSS_DISPLAY_ORDER == (
        "L1_B", "L2_C_anchor", "L3_balanced_vxc",
        "L4_balanced_vxc_anchor", "L5_gradnorm_vxc",
    )


def test_loss_family_is_total():
    """Every entry in LOSS_DISPLAY_ORDER must have a family assignment."""
    for loss in LOSS_DISPLAY_ORDER:
        assert loss in LOSS_FAMILY
    assert set(LOSS_FAMILY.values()) == {"no-Vxc", "Vxc-LossNorm", "Vxc-GradNorm"}


def test_mae_of_returns_mean_absolute():
    """``mae_of`` must compute mean(|value|) per group_keys, not mean(value).

    DFT atomization-energy errors are signed; a model that overshoots and
    undershoots in equal proportion has a tiny mean(value) but a large
    mean(|value|) — only the latter is the reported MAE.
    """
    df = pd.DataFrame({
        "loss":       ["L1", "L1", "L1"],
        "value_name": ["AE_error_kcalmol"] * 3,
        "value":      [+5.0, -5.0, +1.0],
    })
    out = mae_of(df, "AE_error_kcalmol", group_keys=["loss"])
    # mean(|+5|, |-5|, |+1|) = 11/3
    assert abs(out["mae"].iloc[0] - 11.0 / 3) < 1e-12


def test_trained_molecules_only_drops_atoms():
    df = pd.DataFrame({
        "molecule":   ["H2O", "C2H2", "H", "O", "C"],
        "value_name": ["AE_error_kcalmol"] * 5,
        "value":      [0.1, 0.2, 1e6, 1e6, 1e6],
    })
    out = trained_molecules_only(df)
    assert set(out["molecule"]) == {"H2O", "C2H2"}
    # The astronomical atom rows must NOT skew the mean.
    assert out["value"].max() < 1.0


@pytest.mark.skipif(
    not UNWEIGHTED_RUN_DIR.is_dir(),
    reason="unweighted-pretrain run not present; skip live-data tests",
)
class TestLiveData:
    """Run the helpers against the actual unweighted-pretrain artifacts."""

    @pytest.fixture(scope="class")
    def art(self):
        return load_run_artifacts(UNWEIGHTED_RUN_DIR)

    def test_artifacts_load(self, art):
        for k in ("eval_df", "baseline_df",
                  "transfer_primary_df", "transfer_secondary_df"):
            assert art[k] is not None, f"{k} failed to load"
            assert isinstance(art[k], pd.DataFrame)
            assert len(art[k]) > 0

    def test_eval_df_shape(self, art):
        df = art["eval_df"]
        # 90 specs × 5 molecules × ~12 metrics; allow some sparsity.
        assert len(df) >= 5000
        for col in ("group", "arch", "loss", "solver",
                    "molecule", "value_name", "value"):
            assert col in df.columns

    def test_headline_stats_has_all_keys(self, art):
        stats = headline_stats(art)
        for k in (
            "n_specs", "best_ae_mae", "best_ae_spec",
            "best_rmse", "best_rmse_spec",
            "random_ae", "pretrained_ae", "pbe_ae",
            "random_to_best_x", "pretrained_to_best_x", "pbe_to_best_x",
            "g3_mae_by_loss",
        ):
            assert k in stats, f"headline_stats missing {k}"
        # Sanity: best AE is below 1 kcal/mol on this run, well below PBE.
        assert stats["best_ae_mae"] < 1.0
        assert stats["pbe_ae"] > stats["best_ae_mae"]
        assert stats["random_ae"] > stats["pretrained_ae"] > stats["pbe_ae"]

    def test_g3_mae_ordering_matches_report(self, art):
        """The report claims L2/L1/L5 < L3 < L4 on group 3 AE-MAE.
        Lock that ordering so we know if a future run differs."""
        stats = headline_stats(art)
        m = stats["g3_mae_by_loss"]
        # Energy-only / GradNorm cluster is well below LossNorm cluster.
        assert max(m["L1_B"], m["L2_C_anchor"], m["L5_gradnorm_vxc"]) < \
               min(m["L3_balanced_vxc"], m["L4_balanced_vxc_anchor"])

    def test_all_plots_run(self, art, tmp_path):
        """Every plot function must produce a non-empty PNG without raising."""
        for fn, name in [
            (plot_baseline_reduction,           "f1.png"),
            (plot_baseline_reduction_transfer,  "f1b.png"),
            (plot_density_vs_energy_tradeoff,   "f2.png"),
            (plot_in_dist_vs_transfer,          "f3.png"),
            (plot_loss_strategy_heatmap,        "f4.png"),
            (plot_arch_comparison,              "f5.png"),
        ]:
            out = tmp_path / name
            fn(art, out, run_label="test")
            assert out.is_file(), f"{name} not written"
            assert out.stat().st_size > 1024, f"{name} too small to be a real plot"
