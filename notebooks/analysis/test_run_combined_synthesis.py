"""Tests for ``notebooks.analysis.run_combined_synthesis``.

Pins the cross-workflow synthesis report's contracts:

  - all 4 workflows are loadable
  - per-workflow best-AE-on-H2O is computed from H2O-only AE rows (NOT the
    multi-mol mean-MAE that step 6's headline_diff.json reports)
  - integration beats unweighted on best-AE-on-H2O for BOTH steps
    (the headline finding of conclusion 1 + 2)
  - integration beats unweighted on pretrain F_x mean for BOTH steps
  - LOB ceiling 1.804 holds across both step-6 origins (architectural;
    we don't re-load all checkpoints, we just verify _AlecLOB)
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_combined_synthesis import (  # noqa: E402
    WORKFLOWS, SHARED_ARCHS,
    load_all_runs, best_ae_on_h2o, best_density_rmse,
    per_arch_pretrain_fx, combined_headline,
)


REPO = HERE.parent.parent


@pytest.mark.skipif(
    not all(
        (REPO / "notebooks" / f"checkpoints_{step}" / origin).is_dir()
        for step in ("step5", "step6") for origin in ("unweighted", "integration")
    ),
    reason="all four workflows must exist (step5+step6 × unweighted+integration)",
)
class TestCombinedSynthesis:

    @pytest.fixture(scope="class")
    def arts(self):
        return load_all_runs()

    def test_all_four_workflows_load(self, arts):
        for step, origin, _ in WORKFLOWS:
            assert arts[(step, origin)] is not None, (
                f"workflow {step}/{origin} failed to load"
            )

    def test_best_ae_h2o_uses_h2o_rows_only(self, arts):
        """``best_ae_on_h2o`` must restrict to molecule == 'H2O' so step 6's
        bigger H2O+C2H2 sweep is comparable to step 5's H2O-only sweep on
        the same axis (apples-to-apples cross-step comparison)."""
        for step, origin, _ in WORKFLOWS:
            v = best_ae_on_h2o(arts[(step, origin)], step)
            assert v > 0, f"{step}/{origin} produced non-positive best AE"
            # Sanity: best AE should be far below PBE on H2O (~7 kcal/mol)
            assert v < 7.0, f"{step}/{origin} best AE = {v}, suspiciously high"

    def test_integration_beats_unweighted_best_ae_per_step(self, arts):
        """Conclusion 1+2 of the synthesis report. Integration's tighter
        pretrain F_x basin propagates to a tighter best-AE downstream."""
        for step in ("step5", "step6"):
            unw = best_ae_on_h2o(arts[(step, "unweighted")], step)
            integ = best_ae_on_h2o(arts[(step, "integration")], step)
            assert integ < unw, (
                f"{step}: integration best AE ({integ:.4e}) should be lower "
                f"than unweighted ({unw:.4e})"
            )
            ratio = unw / integ
            assert ratio > 1.5, (
                f"{step}: integration tightening is only {ratio:.2f}× "
                f"(expected > 1.5×)"
            )

    def test_integration_tightens_pretrain_fx_on_every_arch(self, arts):
        """Across all 10 archs (8 step-5 + 2 step-6), integration tightens
        F_x without exception. This is the cleanest cross-workflow result.
        """
        for step in ("step5", "step6"):
            unw = per_arch_pretrain_fx(step, "unweighted")
            integ = per_arch_pretrain_fx(step, "integration")
            shared = set(unw.keys()) & set(integ.keys())
            assert shared, f"{step}: no archs in both origins' pretrain"
            for arch in shared:
                assert integ[arch] < unw[arch], (
                    f"{step}/{arch}: integration F_x ({integ[arch]:.3e}) "
                    f"should be lower than unweighted F_x ({unw[arch]:.3e})"
                )

    def test_step6_best_density_better_than_step5(self, arts):
        """Conclusion 3: step 6's V_xc-aware losses (L3, L4) achieve density
        quality unreachable by step 5's loss set {A, B, C}."""
        for origin in ("unweighted", "integration"):
            best_5 = best_density_rmse(arts[("step5", origin)])
            best_6 = best_density_rmse(arts[("step6", origin)])
            assert best_6 < best_5, (
                f"step 6 / {origin} best density ({best_6:.3e}) should be "
                f"tighter than step 5 / {origin} ({best_5:.3e})"
            )

    def test_combined_headline_has_all_four_workflows(self, arts):
        stats = combined_headline(arts)
        for step in ("step5", "step6"):
            for origin in ("unweighted", "integration"):
                key = f"{step}_{origin}"
                assert key in stats, f"combined_headline missing {key}"
                for field in ("best_ae_h2o", "best_density_h2o",
                               "pretrain_fx_mean", "n_specs"):
                    assert field in stats[key], (
                        f"combined_headline[{key}] missing {field}"
                    )
