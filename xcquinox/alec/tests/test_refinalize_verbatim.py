"""Refinalization of completed held-out evals under the verbatim rule
(refinalize_verbatim): rewrite-with-backup, idempotence, dry-run, skips."""
import json

import pytest

from xcquinox.alec import refinalize_verbatim as rv

_POOL_SPECS = {
    "hcn": {"atom_composition": (("C", 1), ("H", 1), ("N", 1)), "charge": 0,
            "spin": 0, "atom": "C 0 0 0; N 0 0 1.15; H 0 0 -1.06"},
    "hnc": {"atom_composition": (("C", 1), ("H", 1), ("N", 1)), "charge": 0,
            "spin": 0, "atom": "N 0 0 0; C 0 0 1.17; H 0 0 -1.00"},
    "co2": {"atom_composition": (("C", 1), ("O", 2)), "charge": 0, "spin": 0,
            "atom": "C 0 0 0; O 0 0 1.16; O 0 0 -1.16"},
    "h": {"atom_composition": (("H", 1),), "charge": 0, "spin": 1,
          "atom": "H 0 0 0"},
    "c": {"atom_composition": (("C", 1),), "charge": 0, "spin": 2,
          "atom": "C 0 0 0"},
    "n": {"atom_composition": (("N", 1),), "charge": 0, "spin": 3,
          "atom": "N 0 0 0"},
    "o": {"atom_composition": (("O", 1),), "charge": 0, "spin": 2,
          "atom": "O 0 0 0"},
}

_POOL_RXNS = [
    {"name": "w411_hcn_atomization", "source_pool": "w411",
     "reactants": ["hcn"], "products": ["h", "c", "n"],
     "coeffs": [-1.0, 1.0, 1.0, 1.0], "reaction_energy_ref": 313.4},
    {"name": "w411_hnc_atomization", "source_pool": "w411",
     "reactants": ["hnc"], "products": ["h", "c", "n"],
     "coeffs": [-1.0, 1.0, 1.0, 1.0], "reaction_energy_ref": 298.7},
    {"name": "bh76_hcn_to_hcnts", "source_pool": "bh76",
     "reactants": ["hcn"], "products": ["hnc"],
     "coeffs": [-1.0, 1.0], "reaction_energy_ref": 15.0},
    {"name": "w411_co2_atomization", "source_pool": "w411",
     "reactants": ["co2"], "products": ["c", "o"],
     "coeffs": [-1.0, 1.0, 2.0], "reaction_energy_ref": 390.0},
]

_E = {"hcn": -93.30, "hnc": -93.27, "co2": -188.10,
      "h": -0.50, "c": -37.80, "n": -54.50, "o": -75.00}


def _mk_run(root):
    """One completed spec evaluated under the OLD species-strict rule:
    per_reaction.json lacks the species-sharing barrier and the hnc twin
    that the verbatim rule keeps. Trained: the CHN atomization. Validation
    slice: the co2 atomization."""
    run = root / "run_x"
    (run / "validation").mkdir(parents=True)
    (run / "validation" / "val_reactions.json").write_text(json.dumps([
        {"name": "w411_co2_atomization", "reactants": ["co2"],
         "products": ["c", "o"], "coeffs": [-1.0, 1.0, 2.0],
         "reaction_energy_ref": 390.0}]))
    sd = run / "checkpoints" / "spec_0000"
    (sd / "eval_holdout").mkdir(parents=True)
    (sd / "train_metadata.json").write_text(json.dumps({
        "molecules": ["CHN", "h", "c", "n"],
        "loss_kwargs": {"bh76_reactions": [
            {"name": "CHN", "reactants": ["CHN"],
             "products": ["C", "H", "N"],
             "coeffs": [-1.0, 1.0, 1.0, 1.0]}]}}))
    (sd / "eval_holdout" / "per_molecule.json").write_text(json.dumps([
        {"molecule": m, "E_total_nn": _E[m], "E_pbe": _E[m] + 0.001}
        for m in _E]))
    # old-rule artifacts: only the hnc atomization survived species strict
    (sd / "eval_holdout" / "per_reaction.json").write_text(json.dumps([
        {"name": "w411_hnc_atomization", "pool": "w411",
         "abs_error_nn_kcalmol": 1.0}]))
    (sd / "eval_holdout" / "test_set.csv").write_text("old\n")
    return run


def _names(run):
    p = run / "checkpoints" / "spec_0000" / "eval_holdout" \
        / "per_reaction.json"
    return sorted(r["name"] for r in json.loads(p.read_text()))


def test_refinalize_rewrites_to_verbatim_rule_with_backups(tmp_path, capsys):
    run = _mk_run(tmp_path)
    reports = rv.refinalize_run(run, channels=("eval_holdout",),
                                _pool=(_POOL_SPECS, _POOL_RXNS))
    assert [r["status"] for r in reports] == ["rewritten"]
    # verbatim rule: hcn twin (trained) and co2 (validation) leave; the
    # species-sharing barrier and the hnc atomization stay.
    assert _names(run) == ["bh76_hcn_to_hcnts", "w411_hnc_atomization"]
    sd = run / "checkpoints" / "spec_0000" / "eval_holdout"
    assert (sd / "per_reaction.pre_verbatim.json").is_file()
    assert json.loads((sd / "per_reaction.pre_verbatim.json").read_text())[
        0]["name"] == "w411_hnc_atomization"
    assert (sd / "test_set.pre_verbatim.csv").read_text() == "old\n"
    assert "rewritten" in capsys.readouterr().out


def test_refinalize_is_idempotent_and_preserves_backups(tmp_path):
    run = _mk_run(tmp_path)
    rv.refinalize_run(run, channels=("eval_holdout",),
                      _pool=(_POOL_SPECS, _POOL_RXNS))
    sd = run / "checkpoints" / "spec_0000" / "eval_holdout"
    bak = (sd / "per_reaction.pre_verbatim.json").read_text()
    reports = rv.refinalize_run(run, channels=("eval_holdout",),
                                _pool=(_POOL_SPECS, _POOL_RXNS))
    assert [r["status"] for r in reports] == ["unchanged"]
    assert (sd / "per_reaction.pre_verbatim.json").read_text() == bak


def test_refinalize_dry_run_writes_nothing(tmp_path):
    run = _mk_run(tmp_path)
    sd = run / "checkpoints" / "spec_0000" / "eval_holdout"
    before = (sd / "per_reaction.json").read_text()
    reports = rv.refinalize_run(run, channels=("eval_holdout",),
                                dry_run=True,
                                _pool=(_POOL_SPECS, _POOL_RXNS))
    assert [r["status"] for r in reports] == ["would-rewrite"]
    assert (sd / "per_reaction.json").read_text() == before
    assert not (sd / "per_reaction.pre_verbatim.json").exists()


def test_refinalize_skips_channels_without_energy_columns(tmp_path):
    run = _mk_run(tmp_path)
    sd = run / "checkpoints" / "spec_0000"
    (sd / "eval_holdout_val_best").mkdir()
    (sd / "eval_holdout_val_best" / "per_molecule.json").write_text(
        json.dumps([{"molecule": "hcn", "density_rmse": 1e-4}]))
    reports = rv.refinalize_run(
        run, channels=("eval_holdout", "eval_holdout_val_best"),
        _pool=(_POOL_SPECS, _POOL_RXNS))
    by = {r["channel"]: r["status"] for r in reports}
    assert by["eval_holdout"] == "rewritten"
    assert by["eval_holdout_val_best"] == "skipped-no-energy-columns"


def test_rewritten_rows_match_cluster_schema(tmp_path):
    run = _mk_run(tmp_path)
    rv.refinalize_run(run, channels=("eval_holdout",),
                      _pool=(_POOL_SPECS, _POOL_RXNS))
    p = run / "checkpoints" / "spec_0000" / "eval_holdout" \
        / "per_reaction.json"
    rows = json.loads(p.read_text())
    for r in rows:
        for k in ("name", "pool", "reaction_energy_ref_kcalmol",
                  "de_nn_kcalmol", "de_pbe_kcalmol",
                  "abs_error_nn_kcalmol", "abs_error_pbe_kcalmol",
                  "reactants", "products"):
            assert k in r, (k, sorted(r))
    kcal = 627.5094740631
    de = ( _E["hnc"] - _E["hcn"]) * kcal
    row = [r for r in rows if r["name"] == "bh76_hcn_to_hcnts"][0]
    assert row["de_nn_kcalmol"] == pytest.approx(de)


def test_per_molecule_is_never_rewritten(tmp_path):
    """The energy record is the one artifact refinalization must not touch:
    its bytes survive a rewrite verbatim."""
    run = _mk_run(tmp_path)
    pm = run / "checkpoints" / "spec_0000" / "eval_holdout" \
        / "per_molecule.json"
    before = pm.read_bytes()
    rv.refinalize_run(run, channels=("eval_holdout",),
                      _pool=(_POOL_SPECS, _POOL_RXNS))
    assert pm.read_bytes() == before


def test_interrupted_csv_write_heals_on_rerun(tmp_path):
    """A crash between the per_reaction and test_set writes must be healed:
    the unchanged check covers BOTH artifacts."""
    run = _mk_run(tmp_path)
    rv.refinalize_run(run, channels=("eval_holdout",),
                      _pool=(_POOL_SPECS, _POOL_RXNS))
    sd = run / "checkpoints" / "spec_0000" / "eval_holdout"
    (sd / "test_set.csv").write_text("stale-after-crash\n")
    reports = rv.refinalize_run(run, channels=("eval_holdout",),
                                _pool=(_POOL_SPECS, _POOL_RXNS))
    assert [r["status"] for r in reports] == ["rewritten"]
    assert "stale-after-crash" not in (sd / "test_set.csv").read_text()
    # the original backup from the FIRST rewrite is preserved
    assert (sd / "test_set.pre_verbatim.csv").read_text() == "old\n"


def test_refuses_run_without_validation_record(tmp_path, capsys):
    run = _mk_run(tmp_path)
    (run / "validation" / "val_reactions.json").unlink()
    reports = rv.refinalize_run(run, channels=("eval_holdout",),
                                _pool=(_POOL_SPECS, _POOL_RXNS))
    assert [r["status"] for r in reports] == [
        "refused-no-validation-record"]
    assert "REFUSING" in capsys.readouterr().out
    # explicit override proceeds (validation reactions then stay in)
    reports = rv.refinalize_run(run, channels=("eval_holdout",),
                                allow_missing_validation=True,
                                _pool=(_POOL_SPECS, _POOL_RXNS))
    assert [r["status"] for r in reports] == ["rewritten"]
    names = _names(run)
    assert "w411_co2_atomization" in names


def test_skipped_channels_are_printed(tmp_path, capsys):
    run = _mk_run(tmp_path)
    rv.refinalize_run(run, channels=("eval_holdout",
                                     "eval_holdout_val_best"),
                      _pool=(_POOL_SPECS, _POOL_RXNS))
    out = capsys.readouterr().out
    assert "skipped-no-channel" in out


def test_main_flags_non_run_dir(tmp_path, capsys):
    assert rv.main([str(tmp_path / "nope")]) == 1
    assert "not a run dir" in capsys.readouterr().out


def test_no_metadata_warning_for_channel_less_specs(tmp_path, capsys):
    """Pending/untrained specs (no metadata AND no eval channels) skip
    silently; the metadata warning is reserved for refinalizable specs."""
    run = _mk_run(tmp_path)
    sd = run / "checkpoints" / "spec_0001"
    sd.mkdir()
    rv.refinalize_run(run, channels=("eval_holdout",),
                      _pool=(_POOL_SPECS, _POOL_RXNS))
    out = capsys.readouterr().out
    assert "spec_0001 has no readable" not in out
    # a trained-but-metadata-less spec WITH a channel still warns
    sd2 = run / "checkpoints" / "spec_0000"
    (sd2 / "train_metadata.json").unlink()
    rv.refinalize_run(run, channels=("eval_holdout",),
                      _pool=(_POOL_SPECS, _POOL_RXNS))
    assert "spec_0000 has no readable" in capsys.readouterr().out


def test_channels_include_coldstart():
    """The verbatim-rule refinalizer must reach the cold-start channel too,
    or its rows silently keep a stale hold-out rule forever."""
    assert "eval_holdout_coldstart" in rv.CHANNELS
