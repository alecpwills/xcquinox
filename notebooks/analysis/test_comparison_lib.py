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

import jax.numpy as jnp
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
    plot_origin_comparison_per_loss_per_group,
    plot_origin_ratio_heatmap,
    plot_origin_pareto_density_vs_energy,
    plot_origin_fx_asymptote_vs_pbe,
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
    mean(|value|) -- only the latter is the reported MAE.
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

    def test_medvedev_tradeoff_signs_per_family(self, art):
        """Pin the per-loss-family medians the report claims:

          - V_xc-LossNorm has LOWER density-RMSE than no-V_xc (Medvedev)
            AND HIGHER AE-MAE (the price)
          - V_xc-GradNorm under vxc_weight=0.01 ends up at the no-V_xc
            density-RMSE cluster, NOT the V_xc-LossNorm cluster -- the
            dynamic balancer effectively suppresses the V_xc term.

        These signs are the report's headline physical claims (§2 fig2,
        §4 conclusion 3); regression here means the qualitative report
        narrative no longer matches the data.
        """
        from comparison_lib import LOSS_FAMILY, mae_of, trained_molecules_only
        eval_df = art["eval_df"]
        ae = mae_of(trained_molecules_only(eval_df), "AE_error_kcalmol",
                    group_keys=["group","arch","loss","solver"])
        rho = mae_of(trained_molecules_only(eval_df), "density_rmse",
                     group_keys=["group","arch","loss","solver"]).rename(columns={"mae":"rmse"})
        m = ae.merge(rho, on=["group","arch","loss","solver"])
        m["family"] = m["loss"].map(LOSS_FAMILY)
        med = m.groupby("family")[["mae","rmse"]].median()
        # Medvedev tradeoff: V_xc-LossNorm has lower density error.
        assert med.loc["Vxc-LossNorm","rmse"] < med.loc["no-Vxc","rmse"], (
            "V_xc-LossNorm should give LOWER density-RMSE than no-V_xc "
            "(Medvedev tradeoff); a regression here breaks fig2's narrative."
        )
        # Medvedev tradeoff: V_xc-LossNorm pays in AE.
        assert med.loc["Vxc-LossNorm","mae"]  > med.loc["no-Vxc","mae"], (
            "V_xc-LossNorm should give HIGHER AE-MAE than no-V_xc "
            "(the price of better density)."
        )
        # GradNorm at vxc_weight=0.01 sits at no-V_xc density level.
        # We do not require an exact tie -- just that the Vxc-GradNorm
        # density advantage over no-V_xc is smaller than 25% in absolute
        # value (i.e. NOT a real density advantage like LossNorm's 56%).
        no_med   = med.loc["no-Vxc","rmse"]
        gn_med   = med.loc["Vxc-GradNorm","rmse"]
        rel_diff = abs(no_med - gn_med) / no_med
        assert rel_diff < 0.25, (
            f"V_xc-GradNorm density-RMSE differs from no-V_xc by "
            f"{rel_diff*100:.1f}% (expected < 25%, i.e. effectively "
            f"tied). A large advantage here would mean GradNorm is "
            f"actually fitting V_xc, contradicting the report claim "
            f"that vxc_weight=0.01 suppresses V_xc under dynamic balancing."
        )

    def test_arch_per_loss_winners_match_report(self, art):
        """The report claims (§2 fig5):
          On TRAINED mols, attn wins on L1, L2, L4; loses on L3, L5.

        Pin those signs so a future regeneration can't silently flip
        the architecture story.
        """
        from comparison_lib import mae_of, trained_molecules_only
        eval_df = art["eval_df"]
        g = mae_of(trained_molecules_only(eval_df), "AE_error_kcalmol",
                   group_keys=["loss","arch"])
        pv = g.pivot(index="loss", columns="arch", values="mae")
        # winners: True if attn beats deep_combined
        attn_wins = pv["deep_combined_attn"] < pv["deep_combined"]
        assert attn_wins.loc["L1_B"]                   == True
        assert attn_wins.loc["L2_C_anchor"]            == True
        assert attn_wins.loc["L3_balanced_vxc"]        == False, (
            "deep_combined should beat attn on L3_balanced_vxc on trained mols")
        assert attn_wins.loc["L4_balanced_vxc_anchor"] == True
        assert attn_wins.loc["L5_gradnorm_vxc"]        == False, (
            "deep_combined should beat attn on L5_gradnorm_vxc on trained mols")

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


_INT_RUN_DIR = REPO / "notebooks" / "checkpoints_step6" / "integration"


@pytest.mark.skipif(
    not (UNWEIGHTED_RUN_DIR.is_dir() and _INT_RUN_DIR.is_dir()),
    reason="needs both unweighted and integration runs present",
)
class TestOriginComparison:
    """Comparison helpers (unweighted vs integration pretrain origin)."""

    @pytest.fixture(scope="class")
    def arts(self):
        return (load_run_artifacts(UNWEIGHTED_RUN_DIR),
                load_run_artifacts(_INT_RUN_DIR))

    def test_comparison_plots_run(self, arts, tmp_path):
        art_a, art_b = arts
        # Tiny stub asymptote dicts (covers the only loss the function
        # iterates over); the real comparison runner uses measured numbers.
        fxa = {l: {"mean": 1.5, "min": 1.4, "max": 1.6} for l in LOSS_DISPLAY_ORDER}
        fxb = {l: {"mean": 1.55, "min": 1.45, "max": 1.65} for l in LOSS_DISPLAY_ORDER}
        cases = [
            (plot_origin_comparison_per_loss_per_group, "c1.png", (art_a, art_b, "A", "B")),
            (plot_origin_ratio_heatmap,                 "c2.png", (art_a, art_b, "A", "B")),
            (plot_origin_pareto_density_vs_energy,      "c3.png", (art_a, art_b, "A", "B")),
        ]
        for fn, name, args in cases:
            out = tmp_path / name
            fn(*args, out)
            assert out.is_file()
            assert out.stat().st_size > 1024
        # f4 has a different signature (extra fx_audit dicts).
        out = tmp_path / "c4.png"
        plot_origin_fx_asymptote_vs_pbe(
            art_a, art_b, "A", "B", fxa, fxb, out,
        )
        assert out.is_file()
        assert out.stat().st_size > 1024

    def test_int_better_ae_than_unw(self, arts):
        """The headline finding from §1 of the comparison report:
        integration pretrain produces a ~2.86x tighter best AE-MAE."""
        unw, integ = arts
        s_unw = headline_stats(unw)
        s_int = headline_stats(integ)
        ratio = s_unw["best_ae_mae"] / s_int["best_ae_mae"]
        assert ratio > 1.5, (
            f"integration's best AE-MAE ({s_int['best_ae_mae']:.5f}) should be "
            f"meaningfully tighter than unweighted's ({s_unw['best_ae_mae']:.5f}); "
            f"observed ratio = {ratio:.2f}x. A regression here means the "
            f"reported 'integration is 2.86x tighter' headline no longer holds."
        )

    def test_lob_identically_enforced_per_origin(self, arts):
        """Both origins must respect F_x <= 1.804 (Lieb & Oxford 1981);
        the architectural _AlecLOB clamp is pretrain-origin-independent.

        Smoke test only -- the heavy 90-spec sweep lives in the
        audit_lob_enforcement.py driver; this test just verifies
        the contract by computing F_x on a fresh model with extreme
        pre-clamp activations.
        """
        from xcquinox.alec.networks import _AlecLOB
        lob = _AlecLOB(limit=1.804)
        for x in [-1e9, -10.0, 0.0, 10.0, 1e9]:
            fx = 1.0 + float(lob(jnp.array(float(x))))
            assert 0.0 <= fx <= 1.804 + 1e-9, (
                f"_AlecLOB returned F_x = {fx} for input {x}; "
                f"Lieb-Oxford bound says F_x in [0, 1.804] always"
            )
