"""Refinalization of completed held-out evals under the verbatim rule
(refinalize_holdout): rewrite-with-backup, idempotence, dry-run, skips."""
import json

import pytest

from xcquinox.pipeline import refinalize_holdout as rv

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


def test_refinalize_rewrites_every_reaction_with_backups(tmp_path, capsys):
    """The rewrite reports EVERY reaction of the pool, the trained twin and
    the validation reaction included: nothing is excluded from a held-out set,
    and the previous artifacts are backed up once.

    Oracle: the pool the stub supplies, against the rewritten table.
    """
    run = _mk_run(tmp_path)
    reports = rv.refinalize_run(run, channels=("eval_holdout",),
                                _pool=(_POOL_SPECS, _POOL_RXNS))
    assert [r["status"] for r in reports] == ["rewritten"]
    assert _names(run) == sorted(r["name"] for r in _POOL_RXNS)
    sd = run / "checkpoints" / "spec_0000" / "eval_holdout"
    assert (sd / "per_reaction.pre_refinalize.json").is_file()
    assert json.loads((sd / "per_reaction.pre_refinalize.json").read_text())[
        0]["name"] == "w411_hnc_atomization"
    assert (sd / "test_set.pre_refinalize.csv").read_text() == "old\n"
    assert "rewritten" in capsys.readouterr().out


def test_refinalize_is_idempotent_and_preserves_backups(tmp_path):
    run = _mk_run(tmp_path)
    rv.refinalize_run(run, channels=("eval_holdout",),
                      _pool=(_POOL_SPECS, _POOL_RXNS))
    sd = run / "checkpoints" / "spec_0000" / "eval_holdout"
    bak = (sd / "per_reaction.pre_refinalize.json").read_text()
    reports = rv.refinalize_run(run, channels=("eval_holdout",),
                                _pool=(_POOL_SPECS, _POOL_RXNS))
    assert [r["status"] for r in reports] == ["unchanged"]
    assert (sd / "per_reaction.pre_refinalize.json").read_text() == bak


def test_refinalize_refuses_a_sliced_channel(tmp_path):
    """The refinalize stage re-selects a channel's test slice from the FULL
    pool. On a sliced channel it would write full-pool-shaped artifacts over
    a handful of species' energies -- and the marker beside them would still
    say the channel is a slice."""
    from xcquinox.pipeline.eval_holdout import SlicedChannelError
    run = _mk_run(tmp_path)
    sd = run / "checkpoints" / "spec_0000" / "eval_holdout"
    (sd / "sliced_eval.json").write_text(json.dumps(
        {"species_slice": ["h", "h2", "o", "oh", "n2o", "n2ohts"],
         "n_species": 6, "n_reactions": 1,
         "env_var": "XCQUINOX_HELDOUT_SPECIES_SLICE"}))
    before = (sd / "per_reaction.json").read_text()
    with pytest.raises(SlicedChannelError) as exc:
        rv.refinalize_run(run, channels=("eval_holdout",),
                          _pool=(_POOL_SPECS, _POOL_RXNS))
    msg = str(exc.value)
    assert "run_x" in msg
    assert "spec_0000" in msg
    assert "eval_holdout" in msg
    assert "'n2ohts'" in msg
    # nothing rewritten, no backup taken
    assert (sd / "per_reaction.json").read_text() == before
    assert not (sd / "per_reaction.pre_refinalize.json").exists()
    assert (sd / "test_set.csv").read_text() == "old\n"


def test_refinalize_reads_the_pools_from_the_run_s_resolved_config(tmp_path,
                                                                   monkeypatch):
    """Refinalization re-selects each channel's test slice from the pool the run
    evaluated. Reading the pair regardless would rebuild a wider run's tables from a
    narrower pool and drop every reaction the run actually reported.

    Oracle: the pool names the loader received, against the run's resolved config.
    """
    import yaml

    run = _mk_run(tmp_path)
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(
        {"inputs": {"held_out_pools": ["bh76", "w411", "diet150"]}}))

    seen = {}

    def _fake_load(names, basis=None, grid_level=None, refs_dir=None):
        seen["pools"] = tuple(names)
        return _POOL_SPECS, _POOL_RXNS

    monkeypatch.setattr(rv, "_load_held_out_pools", _fake_load)
    rv.refinalize_run(run, channels=("eval_holdout",))
    assert seen["pools"] == ("bh76", "w411", "diet150")

    bare = _mk_run(tmp_path / "bare")
    rv.refinalize_run(bare, channels=("eval_holdout",))
    assert seen["pools"] == ("bh76", "w411")




