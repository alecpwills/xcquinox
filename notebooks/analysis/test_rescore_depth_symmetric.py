"""Tests for rescore_depth_symmetric.py: the identity-deduped reduction,
the two validation-record layouts, and the two slice recipes."""
from __future__ import annotations

import importlib.util
import json
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


rds = _load("rescore_depth_symmetric")


def test_identity_dedup_mae_counts_twins_once():
    rows = [
        {"reactants": ["a", "b"], "products": ["ts"],
         "coeffs": [-1, -1, 1], "abs_error_nn_kcalmol": 2.0},
        {"reactants": ["b", "a"], "products": ["ts"],
         "coeffs": [-1, -1, 1], "abs_error_nn_kcalmol": 2.0},
        {"reactants": ["a"], "products": ["b"],
         "coeffs": [-1, 1], "abs_error_nn_kcalmol": 8.0},
    ]
    mae, n = rds._identity_dedup_mae(rows, "abs_error_nn_kcalmol", {})
    assert n == 2
    assert mae == pytest.approx(5.0)


def test_val_record_entries_reads_both_layouts(tmp_path):
    """v4-era root val_reactions.json and v5+ validation/val_reactions.json
    both resolve; the validation/ layout wins when both exist."""
    entries = [{"name": "r", "reactants": ["a"], "products": ["b"]}]
    r_v4 = tmp_path / "v4run"
    r_v4.mkdir()
    (r_v4 / "val_reactions.json").write_text(json.dumps(entries))
    assert rds._val_record_entries(r_v4) == entries

    r_v6 = tmp_path / "v6run"
    (r_v6 / "validation").mkdir(parents=True)
    (r_v6 / "validation" / "val_reactions.json").write_text(
        json.dumps({"reactions": entries}))
    (r_v6 / "val_reactions.json").write_text(json.dumps([]))
    assert rds._val_record_entries(r_v6) == entries

    assert rds._val_record_entries(tmp_path / "absent") == []


def _fake_run(tmp_path, name, val_rxn, trained_rxn):
    run = tmp_path / name
    sd = run / "checkpoints" / "spec_0000"
    sd.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps(
        {"n_specs": 1, "specs": [{"index": 0, "cell":
                                  {"arch": "a", "subset_size": 1}}]}))
    (sd / "train_metadata.json").write_text(json.dumps({
        "molecules": [], "loss_kwargs": {"bh76_reactions": [trained_rxn]}}))
    vdir = run / "validation"
    vdir.mkdir()
    (vdir / "val_reactions.json").write_text(json.dumps([val_rxn]))
    return run


def test_slice_recipes_differ_only_by_supervised_exclusions(tmp_path):
    """The strict recipe removes trained-reaction identities; the
    validation-only recipe keeps them (the previously reported
    '134-reaction' construction)."""
    _specs, pool_rxns, key_map = rds._pool_and_key_map()
    # Two real pool reactions to stand in as val / trained.
    val_rxn = dict(pool_rxns[0])
    trained_rxn = dict(pool_rxns[1])
    run = _fake_run(tmp_path, "runA", val_rxn, trained_rxn)

    strict = rds.common_slice_identities([run], include_supervised=True)
    loose = rds.common_slice_identities([run], include_supervised=False)

    vid = rds._row_identities(val_rxn, key_map)
    tid = rds._row_identities(trained_rxn, key_map)
    assert vid not in strict["kept"] and vid not in loose["kept"]
    assert tid not in strict["kept"]
    assert tid in loose["kept"], (
        "validation-only recipe must keep the supervised reaction")
    assert (loose["provenance"]["n_common_slice"]
            - strict["provenance"]["n_common_slice"]) >= 1
