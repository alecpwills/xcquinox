"""Tests for refilter_holdout_from_energies: the no-SCF held-out regeneration
must reconstruct the molecule-level overlap correctly (charge + case aware) and
produce a leak-free held-out per_reaction file."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refilter_holdout_from_energies as R  # noqa: E402
from xcquinox.alec import eval_holdout as eh  # noqa: E402
from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools  # noqa: E402


def test_neutral_atom_names_excludes_anions_includes_atoms_and_li():
    specs, _ = load_full_held_out_pools()
    neutral = R.neutral_atom_names(specs)
    for a in ("h", "c", "n", "o", "f", "cl", "li"):
        assert a in neutral, a
    assert "f-" not in neutral and "cl-" not in neutral  # anions are molecules
    assert "ch4" not in neutral and "nh3" not in neutral  # molecules are not atoms


def test_reconstruction_matches_canonical_training_molecule_names():
    specs, _ = load_full_held_out_pools()
    neutral = R.neutral_atom_names(specs)
    meta_names = ["hocn", "h", "c", "n", "o", "f-", "nh3"]

    def spec_for(name):
        if name in specs:
            return specs[name]
        return SimpleNamespace(  # hocn stub (multi-atom, neutral)
            name=name, atom_composition=(("H", 1), ("O", 1), ("C", 1), ("N", 1)),
            charge=0)

    canonical = set(eh.training_molecule_names(
        SimpleNamespace(molecules=[spec_for(n) for n in meta_names])))
    recon = R.training_molecule_names_from_meta(meta_names, neutral)
    assert recon == canonical
    assert recon == {"hocn", "f-", "nh3"}  # anion + case-twin + molecule kept; atoms dropped


def _make_spec_dir(tmp_path, train_molecules):
    specs, _ = load_full_held_out_pools()
    sd = tmp_path / "checkpoints" / "spec_0000"
    (sd / "eval_holdout").mkdir(parents=True)
    pm = [{"molecule": n, "E_total_nn": -1.0 - i * 1e-3, "E_pbe": -1.0 - i * 1e-3}
          for i, n in enumerate(sorted(specs))]  # dummy finite energies
    (sd / "eval_holdout" / "per_molecule.json").write_text(json.dumps(pm))
    (sd / "train_metadata.json").write_text(json.dumps({"molecules": train_molecules}))
    (sd / "eval_holdout" / "per_reaction.json").write_text("[]")  # original, to test backup
    return sd


def test_refilter_spec_holdout_is_leakfree_and_backs_up(tmp_path):
    specs, full_rxns = load_full_held_out_pools()
    neutral = R.neutral_atom_names(specs)
    sd = _make_spec_dir(tmp_path, ["f-", "cl-", "nh3", "h", "c"])
    res = R.refilter_spec(sd, full_rxns, neutral)

    assert (sd / "eval_holdout" / "test_set.csv").is_file()
    assert (sd / "eval_holdout" / "per_reaction.cluster_buggy.json").is_file()
    pr = json.loads((sd / "eval_holdout" / "per_reaction.json").read_text())

    trained_cf = {"f-", "cl-", "nh3"}
    for r in pr:  # NO held-out reaction references a trained molecule (case-insensitive)
        names = {x.casefold() for x in (set(r["reactants"]) | set(r["products"]))}
        assert not (names & trained_cf), f"LEAK: {r['name']}"
    # case-twin closed: neither nh3 (W4-11) nor NH3 (BH76) survives
    assert not any("nh3" in {x.casefold() for x in (set(r["reactants"]) | set(r["products"]))}
                   for r in pr)
    assert res["n_kept"] > 0.8 * len(full_rxns)  # only trained-molecule reactions dropped


def test_main_refuses_without_legacy_flag(capsys):
    """The species-strict rewriter is retired: without the explicit legacy
    flag it must refuse rather than clobber verbatim-rule artifacts."""
    import refilter_holdout_from_energies as rf
    assert rf.main([]) == 2
    assert "REFUSING" in capsys.readouterr().out


def test_refilter_spec_refuses_a_sliced_channel(tmp_path):
    """A sliced channel must be refused BEFORE the rewrite: regenerating its
    per_reaction.json / test_set.csv from the full pool would produce
    full-pool-shaped artifacts backed by a handful of species' energies, and
    the marker would still be there claiming otherwise."""
    import pytest

    specs, full_rxns = load_full_held_out_pools()
    neutral = R.neutral_atom_names(specs)
    sd = _make_spec_dir(tmp_path, ["f-", "cl-", "nh3", "h", "c"])
    (sd / "eval_holdout" / "sliced_eval.json").write_text(json.dumps(
        {"species_slice": ["h", "h2", "o", "oh", "n2o", "n2ohts"],
         "n_species": 6, "n_reactions": 1,
         "env_var": "XCQUINOX_HELDOUT_SPECIES_SLICE"}))
    before = (sd / "eval_holdout" / "per_reaction.json").read_text()
    with pytest.raises(eh.SlicedChannelError) as exc:
        R.refilter_spec(sd, full_rxns, neutral)
    msg = str(exc.value)
    assert "spec_0000" in msg
    assert "eval_holdout" in msg
    assert "'n2ohts'" in msg
    # nothing was rewritten and no backup was taken
    assert (sd / "eval_holdout" / "per_reaction.json").read_text() == before
    assert not (sd / "eval_holdout" / "per_reaction.cluster_buggy.json").exists()
    assert not (sd / "eval_holdout" / "test_set.csv").exists()
