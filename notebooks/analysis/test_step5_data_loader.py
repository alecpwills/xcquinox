"""Tests for ``notebooks.analysis.step5_data_loader``.

Locks in the contracts the step-5 comparison report depends on:

  - load_step5_eval_df returns the trained-mol AE rows for all 72 specs
  - load_step5_transfer_df covers the 3 transfer mols × 8 archs × 3 losses × 3 solvers
  - load_step5_baseline_df covers random + pretrained × 8 archs
  - load_step5_pretrain_metadata returns 8 rows per origin with x/c losses
  - integration's pretrain F_x is meaningfully tighter than unweighted's
    (the headline finding that drives the comparison report's conclusion)
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from step5_data_loader import (  # noqa: E402
    STEP5_ARCHS, STEP5_LOSSES, STEP5_SOLVERS,
    load_step5_eval_df,
    load_step5_transfer_df,
    load_step5_baseline_df,
    load_step5_pretrain_metadata,
    load_step5_run,
)

REPO = HERE.parent.parent
RUN_UNW = REPO / "notebooks" / "checkpoints_step5" / "unweighted"
RUN_INT = REPO / "notebooks" / "checkpoints_step5" / "integration"


def test_step5_constants():
    """The step-5 sweep matrix is 8 archs × 3 losses × 3 solvers = 72 specs."""
    assert len(STEP5_ARCHS) == 8
    assert len(STEP5_LOSSES) == 3
    assert len(STEP5_SOLVERS) == 3


def test_per_molecule_melt_drops_bookkeeping_columns():
    """n_electrons / grid_weight_sum are quadrature bookkeeping -- the long
    form must not carry them as ~1e5-scale pseudo-metric rows, while genuine
    error columns (the Eq. 20 eps) still melt."""
    import pandas as pd
    from step5_data_loader import _per_molecule_rows_to_long
    df = pd.DataFrame([{"molecule": "h2o", "density_eps_l1": 1e-3,
                        "n_electrons": 10.0, "grid_weight_sum": 2.1e5}])
    long = _per_molecule_rows_to_long(df)
    names = set(long["value_name"])
    assert "density_eps_l1" in names
    assert "n_electrons" not in names
    assert "grid_weight_sum" not in names


@pytest.mark.skipif(
    not (RUN_UNW.is_dir() and RUN_INT.is_dir()),
    reason="needs both step-5 unweighted and integration runs present",
)
class TestLiveStep5Data:

    @pytest.fixture(scope="class")
    def runs(self):
        return {
            "unweighted":  load_step5_run(RUN_UNW),
            "integration": load_step5_run(RUN_INT),
        }

    def test_eval_df_covers_72_specs(self, runs):
        """Each origin must have eval rows for all 72 specs (per_molecule.csv
        per spec). H2O is the only non-atomic training mol."""
        for origin, run in runs.items():
            df = run["eval_df"]
            assert not df.empty, f"step5 {origin} eval_df empty"
            assert set(df["arch"].unique()) == set(STEP5_ARCHS), (
                f"step5 {origin} eval_df arch mismatch"
            )
            assert set(df["loss"].unique()) == set(STEP5_LOSSES)
            assert set(df["solver"].unique()) == set(STEP5_SOLVERS)
            assert "H2O" in set(df["molecule"].unique())

    def test_transfer_df_covers_three_mols(self, runs):
        for origin, run in runs.items():
            df = run["transfer_df"]
            assert not df.empty, f"step5 {origin} transfer_df empty"
            assert set(df["molecule"].unique()) == {"CH4", "H2", "OH"}, (
                f"step5 {origin} transfer mols: "
                f"{sorted(df['molecule'].unique())}; expected CH4/H2/OH"
            )

    def test_baseline_df_has_random_and_pretrained(self, runs):
        for origin, run in runs.items():
            df = run["baseline_df"]
            assert not df.empty, f"step5 {origin} baseline_df empty"
            assert set(df["baseline"].unique()) == {"random", "pretrained"}
            assert set(df["arch"].unique()) == set(STEP5_ARCHS), (
                f"step5 {origin} baseline arch mismatch"
            )

    def test_pretrain_metadata_eight_archs(self, runs):
        for origin, run in runs.items():
            pm = run["pretrain_meta"]
            assert len(pm) == 8, f"step5 {origin}: {len(pm)} pretrain metadata rows, expected 8"
            assert set(pm["loss_weighting"].unique()) == {origin}, (
                f"step5 {origin} loss_weighting mismatch in pretrain_metadata.json: "
                f"{sorted(pm['loss_weighting'].unique())}"
            )

    def test_integration_pretrain_fx_tighter_than_unweighted(self, runs):
        """Headline physical finding: integration weighting concentrates
        pretrain F_x fit on bonding-region high-density grid points (per
        PBE 1996 eq. 10 integrand structure), so F_x final loss is
        meaningfully tighter for every architecture under integration.

        A regression of this contract -- e.g. if a future change
        accidentally swapped the integration weight back to a per-grid-
        point MSE -- would break the report's headline narrative.
        """
        unw_pm = runs["unweighted"]["pretrain_meta"].set_index("arch")
        int_pm = runs["integration"]["pretrain_meta"].set_index("arch")
        for arch in STEP5_ARCHS:
            assert int_pm.loc[arch, "final_loss_x"] < unw_pm.loc[arch, "final_loss_x"], (
                f"arch {arch}: expected integration F_x ({int_pm.loc[arch, 'final_loss_x']:.3e})"
                f" < unweighted F_x ({unw_pm.loc[arch, 'final_loss_x']:.3e})"
            )

    def test_integration_best_h2o_ae_tighter_than_unweighted(self, runs):
        """Headline downstream finding: integration's tighter pretrain
        basin propagates to a meaningfully tighter best AE-MAE on H2O.
        The step-5 comparison report claims ~12× tighter; pin a 2× floor
        here to allow for run-to-run variability."""
        bests = {}
        for origin, run in runs.items():
            ae = run["eval_df"][
                (run["eval_df"].value_name == "AE_error_kcalmol") &
                (run["eval_df"].molecule == "H2O")
            ]
            bests[origin] = float(ae["value"].abs().min())
        ratio = bests["unweighted"] / bests["integration"]
        assert ratio > 2.0, (
            f"integration's best H2O AE ({bests['integration']:.4e}) should be "
            f"meaningfully tighter than unweighted's ({bests['unweighted']:.4e}); "
            f"observed ratio = {ratio:.2f}× (expected > 2×)"
        )
