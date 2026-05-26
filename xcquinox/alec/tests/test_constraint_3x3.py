"""Unit test for the 3x3 assembler's demo-log parser (notebooks/analysis/
make_constraint_3x3.py). Loaded by file path via importlib (standalone script)."""
import importlib.util
import os

import pytest

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "notebooks", "analysis",
    "make_constraint_3x3.py",
)


@pytest.fixture(scope="module")
def mod():
    p = os.path.abspath(_PATH)
    if not os.path.isfile(p):
        pytest.skip(f"assembler not found at {p}")
    spec = importlib.util.spec_from_file_location("make_constraint_3x3", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# A minimal log with the three table blocks the demo prints (two levels each).
_LOG = """\
[4/4] Random ...
      unconstrained    bh76[r~21.4] ...

==============================================================================
BH76 reaction-energy MAE vs GMTKN55-BH76RC (kcal/mol)  (lower is better)
==============================================================================
level               random mean   random max  random std   pretrained
------------------------------------------------------------------------------
PBE baseline               8.08                                 (n/a)
unconstrained             21.37        30.79        5.84        (n/a)
+LO(x)                    20.94        25.18        2.63        10.55
==============================================================================

==============================================================================
per-species |E_nn - E_pbe| MAE (kcal/mol; deviation from PBE)  (lower is better)
==============================================================================
level               random mean   random max  random std   pretrained
------------------------------------------------------------------------------
unconstrained            434.85       910.10      287.83        (n/a)
+LO(x)                   389.11       612.92      143.91         2.55
==============================================================================

==============================================================================
atomization-energy MAE vs GMTKN55 W4-11 (kcal/mol)  (lower is better)
==============================================================================
level               random mean   random max  random std   pretrained
------------------------------------------------------------------------------
PBE baseline              10.45                                 (n/a)
unconstrained             20.71        29.98        4.54        (n/a)
+LO(x)                    17.83        20.56        1.35        19.59
==============================================================================
"""


def test_parse_demo_log_extracts_tables(mod, tmp_path):
    p = tmp_path / "demo.log"
    p.write_text(_LOG)
    res = mod.parse_demo_log(str(p))

    # levels in order, deduped across the three tables
    assert res["levels"] == ["unconstrained", "+LO(x)"]

    # PBE baselines (bh76 + w411 have one; pbe_dev does not)
    assert res["pbe"]["bh76"] == pytest.approx(8.08)
    assert res["pbe"]["w411"] == pytest.approx(10.45)
    assert "pbe_dev" not in res["pbe"]

    # random stats
    assert res["rand"]["unconstrained"]["bh76"] == {
        "mean": 21.37, "max": 30.79, "std": 5.84}
    assert res["rand"]["+LO(x)"]["pbe_dev"]["max"] == pytest.approx(612.92)
    assert res["rand"]["unconstrained"]["w411"]["mean"] == pytest.approx(20.71)

    # pretrained: None for the unconstrained '(n/a)' level, float otherwise
    assert res["pre"]["unconstrained"]["bh76"] is None
    assert res["pre"]["+LO(x)"]["bh76"] == pytest.approx(10.55)
    assert res["pre"]["+LO(x)"]["pbe_dev"] == pytest.approx(2.55)
    assert res["pre"]["+LO(x)"]["w411"] == pytest.approx(19.59)


def test_parse_demo_log_rejects_incomplete(mod, tmp_path):
    # A level present in only one table -> missing-metric ValueError.
    bad = _LOG.replace("+LO(x)                    17.83        20.56        1.35        19.59\n", "")
    p = tmp_path / "bad.log"
    p.write_text(bad)
    with pytest.raises(ValueError):
        mod.parse_demo_log(str(p))
