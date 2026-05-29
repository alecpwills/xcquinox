"""Tests for ``xcquinox.alec.eval_holdout`` — the shared eval module.

Pure-function helpers get hand-crafted inputs with KNOWN expected outputs.
The side-effectful PBE precompute + NN forward functions are exercised
via the existing local_reeval test suite (notebooks/analysis/test_local_reeval.py)
that monkeypatches their internals; here we add targeted assertions on
the new module's own surface.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from xcquinox.alec.eval_holdout import (
    KCAL_PER_HA,
    filter_reactions,
    held_out_pool_names,
    make_per_molecule_record,
    make_per_reaction_records,
    per_reaction_errors,
    reaction_mae_kcalmol,
    reaction_overlap,
    write_per_molecule_json,
    write_per_reaction_json,
    write_test_set_csv,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_held_out_pool_names_subtracts_training_set():
    pool = {"H": object(), "O": object(), "H2O": object(), "C": object()}
    out = held_out_pool_names(("H", "O"), pool)
    assert out == ["C", "H2O"]  # sorted, training-set removed


def test_reaction_overlap_finds_in_sample_species():
    rxn = {"reactants": ["H", "O"], "products": ["HO"]}
    has, overlap = reaction_overlap(rxn, {"H"})
    assert has is True
    assert overlap == ["H"]
    has2, overlap2 = reaction_overlap(rxn, {"X"})
    assert has2 is False
    assert overlap2 == []


def test_filter_reactions_loose_keeps_all_with_overlap_field():
    rxns = [
        {"name": "r1", "reactants": ["H"], "products": ["A"]},
        {"name": "r2", "reactants": ["B"], "products": ["C"]},
    ]
    kept, dropped = filter_reactions(rxns, training_names=["H"], strict=False)
    assert len(kept) == 2
    assert len(dropped) == 0
    # The new in_sample_overlap key is added.
    r1 = next(r for r in kept if r["name"] == "r1")
    assert r1["in_sample_overlap"] == ["H"]


def test_filter_reactions_strict_drops_overlapping():
    rxns = [
        {"name": "r1", "reactants": ["H"], "products": ["A"]},
        {"name": "r2", "reactants": ["B"], "products": ["C"]},
    ]
    kept, dropped = filter_reactions(rxns, training_names=["H"], strict=True)
    names_kept = {r["name"] for r in kept}
    names_dropped = {r["name"] for r in dropped}
    assert names_kept == {"r2"}
    assert names_dropped == {"r1"}


# ---------------------------------------------------------------------------
# Reaction MAE math
# ---------------------------------------------------------------------------

def test_reaction_mae_kcalmol_with_known_values():
    # Reaction B - 2A = ref. Pick energies so ΔE = 1 Ha = 627.5094... kcal/mol;
    # ref = 600 kcal/mol; MAE = ~27.51 kcal/mol.
    energies = {"A": 1.0, "B": 3.0}
    reactions = [{
        "name": "r",
        "reactants": ["B"], "products": ["A"],
        "coeffs": [1, -2],
        "reaction_energy_ref": 600.0,
    }]
    mae, n, n_nan = reaction_mae_kcalmol(energies, reactions)
    assert n == 1
    assert n_nan == 0
    assert mae == pytest.approx(KCAL_PER_HA - 600.0, abs=1e-9)


def test_reaction_mae_skips_nonfinite_and_reports_dropped_count():
    energies = {"A": float("nan"), "B": 3.0, "C": 1.0}
    reactions = [
        # uses NaN A -> silently dropped
        {"name": "bad", "reactants": ["A"], "products": ["B"],
         "coeffs": [1, -1], "reaction_energy_ref": 100.0},
        # finite -> kept
        {"name": "ok",  "reactants": ["B"], "products": ["C"],
         "coeffs": [1, -1], "reaction_energy_ref": 1255.0},
    ]
    mae, n, n_nan = reaction_mae_kcalmol(energies, reactions)
    assert n == 1
    assert n_nan == 1  # the NaN reaction got surfaced via the audit-gap fix
    assert mae == pytest.approx(abs(2.0 * KCAL_PER_HA - 1255.0), abs=1e-9)


def test_reaction_mae_no_finite_returns_nan_with_count():
    energies = {"A": float("nan")}
    reactions = [{"name": "x", "reactants": ["A"], "products": [],
                  "coeffs": [1], "reaction_energy_ref": 50.0}]
    mae, n, n_nan = reaction_mae_kcalmol(energies, reactions)
    assert n == 0
    assert n_nan == 1
    assert math.isnan(mae)


def test_per_reaction_errors_records_signed_and_abs_error():
    energies = {"A": 1.0, "B": 2.0}
    rxns = [{"name": "r", "reactants": ["B"], "products": ["A"],
             "coeffs": [-1, 1], "reaction_energy_ref": -600.0}]
    out = per_reaction_errors(energies, rxns)
    assert len(out) == 1
    row = out[0]
    de = (1 * 1.0 + -1 * 2.0) * KCAL_PER_HA  # = -627.509...
    assert row["de_kcalmol"] == pytest.approx(de, abs=1e-9)
    assert row["error_kcalmol"] == pytest.approx(de - (-600.0), abs=1e-9)
    assert row["abs_error_kcalmol"] == pytest.approx(abs(de + 600.0),
                                                      abs=1e-9)


# ---------------------------------------------------------------------------
# Per-record builders
# ---------------------------------------------------------------------------

def test_make_per_molecule_record_carries_flags_and_E_pbe():
    mol_data = {"E_pbe": -76.27}
    rec = make_per_molecule_record(
        "H2O", mol_data, e_nn_ha=-76.43,
        in_training_subset=True,
    )
    assert rec["molecule"] == "H2O"
    assert rec["E_pbe"] == pytest.approx(-76.27)
    assert rec["E_total_nn"] == pytest.approx(-76.43)
    assert rec["AE_nn"] == pytest.approx(-76.43 - (-76.27))
    assert rec["from_training_subset"] is True


def test_make_per_reaction_records_pairs_nn_pbe_and_marks_overlap():
    rxns = [{
        "name": "r1", "source_pool": "test",
        "reactants": ["H", "X"], "products": ["Y"],
        "coeffs": [-1, -1, 1], "reaction_energy_ref": 100.0,
    }]
    nn_err = [{"de_kcalmol": 110.0, "ref_kcalmol": 100.0,
                "error_kcalmol": 10.0, "abs_error_kcalmol": 10.0}]
    pbe_err = [{"de_kcalmol": 105.0, "ref_kcalmol": 100.0,
                 "error_kcalmol": 5.0, "abs_error_kcalmol": 5.0}]
    recs = make_per_reaction_records(rxns, nn_err, pbe_err,
                                       training_names=["H"])
    r = recs[0]
    assert r["name"] == "r1"
    assert r["pool"] == "test"
    assert r["abs_error_nn_kcalmol"] == 10.0
    assert r["abs_error_pbe_kcalmol"] == 5.0
    assert r["in_sample_overlap"] == ["H"]


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def test_write_test_set_csv_includes_n_dropped_nan_column(tmp_path):
    out = tmp_path / "test_set.csv"
    per_pool_mae = {
        "bh76": (12.3456, 8.0774, 6, 0, 0),
        "w411": (15.0000, 10.450, 10, 0, 1),
    }
    combined = (13.5, 9.2, 16, 0, 1)
    p = write_test_set_csv(out, per_pool_mae, combined, strict=False)
    assert p == out
    rows = out.read_text().splitlines()
    # Header gained n_dropped_nan
    assert rows[0] == (
        "set,mae_nn_kcalmol,mae_pbe_kcalmol,delta_nn_minus_pbe,"
        "n_reactions,n_dropped_overlap,n_dropped_nan,note"
    )
    # The w411 row carries the new "1 reactions silently dropped" note text.
    w411 = next(r for r in rows[1:] if "test_set_w411" in r)
    assert ",1," in w411 or w411.split(",")[6] == "1"
    assert "silently dropped" in w411


def test_write_per_molecule_and_per_reaction_json(tmp_path):
    mol_path = tmp_path / "eval_holdout" / "per_molecule.json"
    rxn_path = tmp_path / "eval_holdout" / "per_reaction.json"
    write_per_molecule_json(mol_path, [{"molecule": "H"}])
    write_per_reaction_json(rxn_path, [{"name": "r"}])
    assert mol_path.is_file()
    assert rxn_path.is_file()
    assert json.loads(mol_path.read_text()) == [{"molecule": "H"}]
    assert json.loads(rxn_path.read_text()) == [{"name": "r"}]
