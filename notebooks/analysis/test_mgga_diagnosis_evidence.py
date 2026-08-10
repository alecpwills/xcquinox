"""Tests for the meta-GGA learning-curve decision rule.

The rest of ``mgga_diagnosis_evidence.py`` is reporting over pulled artifacts,
but ``saturation`` and ``mgga_verdict`` encode a PRE-COMMITTED prediction about
specs 0035-0037. A rule that can be nudged after the data lands is not a
prediction, so its thresholds and its refusal-to-decide band are pinned here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


ev = _load("mgga_diagnosis_evidence")


# ---------------------------------------------------------------------------
# saturation
# ---------------------------------------------------------------------------

def test_saturation_refuses_fewer_than_three_points():
    """Two points are trivially 0% and 100% of the span between them -- a
    shape read off them is an artifact of having two points, which is exactly
    the state the meta-GGA was in when the rule was written."""
    assert ev.saturation({}) == {}
    assert ev.saturation({1: 30.0}) == {}
    assert ev.saturation({1: 30.0, 2: 20.0}) == {}
    assert ev.saturation({1: 30.0, 2: 20.0, 3: 10.0}) != {}


def test_saturation_measures_against_the_arch_s_own_span():
    """Fractions are relative to each arch's own first and best values, so a
    family sitting at higher absolute error is still comparable on SHAPE."""
    sat = ev.saturation({1: 20.0, 2: 15.0, 3: 12.0, 4: 10.0})
    assert sat[1] == pytest.approx(0.0)
    assert sat[2] == pytest.approx(0.5)      # 5 of the 10 total gain
    assert sat[3] == pytest.approx(0.8)
    assert sat[4] == pytest.approx(1.0)
    # An arch ten times worse in absolute terms but the same shape scores the
    # same -- the property that lets the meta-GGA be compared to the GGA curves.
    scaled = ev.saturation({1: 200.0, 2: 150.0, 3: 120.0, 4: 100.0})
    assert scaled == pytest.approx(sat)


def test_saturation_is_monotone_through_a_bump():
    """Uses best-so-far, so a cell that regresses (the GGA curves do this --
    deep_attn goes 7.9 -> 11.3 between ss=2 and ss=3) never drives the realized
    fraction backwards."""
    sat = ev.saturation({1: 20.0, 2: 12.0, 3: 16.0, 4: 10.0})
    vals = [sat[k] for k in sorted(sat)]
    assert vals == sorted(vals), vals
    assert sat[3] == sat[2]                  # the regression does not undo gain


def test_saturation_handles_a_flat_curve():
    sat = ev.saturation({1: 10.0, 2: 10.0, 3: 10.0})
    assert set(sat.values()) == {0.0}        # no span, no division by zero


# ---------------------------------------------------------------------------
# the pre-committed verdict
# ---------------------------------------------------------------------------

def test_thresholds_are_ordered_and_leave_a_refusal_band():
    """A later edit that inverts or collapses the band would let any outcome
    be declared a confirmation."""
    assert ev.MGGA_RECOVER_BELOW < ev.MGGA_PERSIST_ABOVE
    assert ev.MGGA_VERDICT_SS == 5


def test_verdict_is_undecided_until_the_deciding_cell_exists():
    assert "UNDECIDED" in ev.mgga_verdict({})
    assert "UNDECIDED" in ev.mgga_verdict({1: 33.32, 2: 26.62})
    # and it names what it is waiting for rather than just declining
    assert "ss=5" in ev.mgga_verdict({1: 33.32, 2: 26.62})


def test_verdict_branches_on_the_deciding_cell():
    below = ev.MGGA_RECOVER_BELOW - 1.0
    above = ev.MGGA_PERSIST_ABOVE + 1.0
    mid = 0.5 * (ev.MGGA_RECOVER_BELOW + ev.MGGA_PERSIST_ABOVE)
    assert "RECOVERS" in ev.mgga_verdict({1: 33.0, 2: 26.0, 5: below})
    assert "PERSISTS" in ev.mgga_verdict({1: 33.0, 2: 26.0, 5: above})
    assert "AMBIGUOUS" in ev.mgga_verdict({1: 33.0, 2: 26.0, 5: mid})


def test_verdict_ignores_cells_other_than_the_deciding_one():
    """A very good ss=26 cell must not be able to rescue a bad ss=5 -- the rule
    was fixed on ss=5 precisely because the GGA curves plateau by then."""
    v = ev.mgga_verdict({1: 33.0, 2: 26.0, 5: ev.MGGA_PERSIST_ABOVE + 1.0,
                         26: 6.0})
    assert "PERSISTS" in v
