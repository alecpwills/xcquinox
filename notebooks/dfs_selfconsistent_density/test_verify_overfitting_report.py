"""Tests for verify_overfitting_report.py -- the data-driven overfitting report.

Covers the pure Hill parser, the data-derived verdict logic (with synthetic
records so it needs no eval data), and -- when the committed eval data is present
-- an integration check that the report reproduces the known headline numbers
(meta-GGA held-out 51.79, in-sample 0.46; GGA held-out 10.28)."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_overfitting_report as V  # noqa: E402


def test_hill_comp():
    assert V.hill_comp("N2") == {"N": 2}
    assert V.hill_comp("HLi") == {"H": 1, "Li": 1}
    assert V.hill_comp("NO2") == {"N": 1, "O": 2}
    assert V.hill_comp("H2O") == {"H": 2, "O": 1}
    assert V.hill_comp("HO") == {"H": 1, "O": 1}


def _synthetic(scf=True, nan=0, dens=True):
    """4 archs (GGA best, pure-mgga worst held-out) with all discriminators
    passing by default; knobs flip a single discriminator for the bug branch."""
    def cell(ins, hel):
        return dict(insample_mae=ins, heldout_mae=hel, heldout_rows=[], atoms={},
                    scf_all_converged=scf, n_nonfinite=nan,
                    heldout_density_beats_pbe=dens)
    return {
        ("deep_3x16", "full_25"): cell(1.72, 10.3),
        ("deep_rung35_3x16", "full_25"): cell(1.94, 35.7),
        ("deep_mgga_3x16", "full_25"): cell(0.46, 51.8),
        ("deep_rung35_mgga_3x16", "full_25"): cell(1.07, 23.4),
    }


def test_verdict_overfitting_when_all_discriminators_pass():
    _, verdict = V.build_report(_synthetic(), {}, {})
    assert verdict is True


@pytest.mark.parametrize("kw", [dict(scf=False), dict(nan=1), dict(dens=False)])
def test_verdict_flags_bug_when_a_discriminator_fails(kw):
    # a failed convergence / NaN / density check must NOT return the overfitting verdict
    _, verdict = V.build_report(_synthetic(**kw), {}, {})
    assert verdict is False


_HAS_DATA = os.path.exists(
    os.path.join(HERE, "runs", "heldout", "deep_mgga_3x16__full_25", "eval",
                 "per_molecule.json"))


@pytest.mark.skipif(not _HAS_DATA, reason="committed eval data not present")
def test_reproduces_known_headline_numbers():
    data, _comp, _refs = V.collect()
    dm = data[("deep_mgga_3x16", "full_25")]
    assert abs(dm["heldout_mae"] - 51.79) < 0.5   # README/figure value
    assert abs(dm["insample_mae"] - 0.46) < 0.10
    gga = data[("deep_3x16", "full_25")]
    assert abs(gga["heldout_mae"] - 10.28) < 0.5
    # the discriminators the report leans on
    assert dm["scf_all_converged"] and dm["n_nonfinite"] == 0
    assert dm["heldout_density_beats_pbe"]
