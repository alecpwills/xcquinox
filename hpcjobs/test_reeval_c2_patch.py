"""The c2 patch's recomputation of ``test_set.csv`` from stored per-reaction rows."""
import csv
import importlib.util
import io
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _patch_module():
    spec = importlib.util.spec_from_file_location(
        "reeval_c2_patch", _HERE / "reeval_c2_patch.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(name, pool, abs_nn, abs_pbe, weight=None):
    row = {"name": name, "pool": pool, "reactants": [f"{name}_r"],
           "products": [f"{name}_p"], "coeffs": [-1.0, 1.0],
           "abs_error_nn_kcalmol": abs_nn, "abs_error_pbe_kcalmol": abs_pbe}
    if weight is not None:
        row["weight"] = weight
    return row


def test_the_recomputation_reproduces_the_weighted_row_of_a_pool():
    """A pool's weighted row is the mean over its identities of the weight
    times the absolute error, its plain row the plain mean and the combined
    row the mean over every row; a weighted row whose stored rows carry no
    weight is refused rather than rewritten.

    Oracle: three stored rows with errors 1, 3 and 5 kcal/mol, the diet rows
    (the ones carrying the patched species) under weights 2 and 4; the BH76
    row, untouched by the species, is recorded as the recomputation writes
    it, the patch's own check that the recomputation reproduces the writer.
    """
    c2 = _patch_module()
    header = ",".join(c2.CSV_FIELDNAMES)
    old = "\n".join([header,
                     "test_set_bh76,1.000000,2.000000,-1.000000,1,0,0,",
                     "test_set_diet150,9.000000,9.000000,+0.000000,2,0,0,",
                     "test_set_diet150_wtmad2,9.000000,9.000000,+0.000000,2,0,0,",
                     "test_set_held_out_combined,9.000000,9.000000,+0.000000,3,0,0,"
                     ]) + "\n"
    rows = [_row("a", "bh76", 1.0, 2.0),
            _row("b", "diet150", 3.0, 1.0, weight=2.0),
            _row("c", "diet150", 5.0, 1.0, weight=4.0)]
    for row in rows[1:]:
        row["products"] = [c2.SPECIES]
    text = c2.recompute_test_set_csv(old, rows)
    new = {r["set"]: r for r in csv.DictReader(io.StringIO(text))}
    assert float(new["test_set_bh76"]["mae_nn_kcalmol"]) == pytest.approx(1.0)
    assert float(new["test_set_diet150"]["mae_nn_kcalmol"]) == pytest.approx(4.0)
    assert float(new["test_set_diet150_wtmad2"]["mae_nn_kcalmol"]) == \
        pytest.approx((2.0 * 3.0 + 4.0 * 5.0) / 2.0)
    assert float(new["test_set_diet150_wtmad2"]["mae_pbe_kcalmol"]) == \
        pytest.approx((2.0 * 1.0 + 4.0 * 1.0) / 2.0)
    assert float(new["test_set_held_out_combined"]["mae_nn_kcalmol"]) == \
        pytest.approx(3.0)

    unweighted = [rows[0], _row("b", "diet150", 3.0, 1.0),
                  _row("c", "diet150", 5.0, 1.0)]
    with pytest.raises(c2.PatchRefused):
        c2.recompute_test_set_csv(old, unweighted)
