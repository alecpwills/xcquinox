"""Tests for ``xcquinox.alec.eval_holdout``: the shared eval module.

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

import xcquinox.alec.eval_holdout as eh
from xcquinox.alec.eval_holdout import (
    KCAL_PER_HA,
    filter_reactions,
    held_out_pool_names,
    make_per_molecule_record,
    make_per_reaction_records,
    per_reaction_errors,
    reaction_mae_kcalmol,
    reaction_overlap,
    run_full_holdout_eval,
    write_per_molecule_json,
    write_per_reaction_json,
    write_test_set_csv,
)


class _FakeSolverConfig:
    """Minimal stand-in with a non-oneshot ``.mode`` (drives the SCF path)."""
    class _Mode:
        value = "full"
        name = "FULL"
    mode = _Mode()


class _FakeArch:
    use_polarized_correlation = False

    def materialize_descriptors(self):
        return ()


class _FakeMol:
    def __init__(self, name):
        self.name = name


class _FakeSpec:
    arch = _FakeArch()
    solver_config = _FakeSolverConfig()
    molecules = ()


def test_run_full_holdout_eval_orchestration_no_compute(tmp_path, monkeypatch):
    """Exercise the full run_full_holdout_eval control flow with the heavy
    compute stubbed, guards against orchestration regressions (e.g. an
    undefined ``spec_solver_config``) that the pure-function tests miss. Also
    asserts the per-SCF-step trace is threaded into per_molecule.json."""
    mol_specs = {"h2": _FakeMol("h2"), "h": _FakeMol("h")}
    reactions = [{
        "name": "w411_h2_atomization", "source_pool": "w411",
        "reactants": ["h2"], "products": ["h"], "coeffs": [-1.0, 2.0],
        "reaction_energy_ref": 109.493,
    }]
    mol_data = {"h2": {"E_pbe": -1.16}, "h": {"E_pbe": -0.50}}

    # Stub the precompute + NN eval (which would otherwise run pyscf/SCF).
    monkeypatch.setattr(eh, "precompute_holdout",
                        lambda specs, **kw: dict(mol_data))

    def _fake_eval(model, md, *, solver_config=None,
                   verbose_first_failure=True, scf_info_out=None):
        energies = {"h2": -1.17, "h": -0.50}
        if scf_info_out is not None:
            for n in md:
                scf_info_out[n] = {
                    "cycles_run": 3, "converged": True,
                    "total_energy": energies[n],
                    "energy_trace": [energies[n] - 0.01, energies[n] - 0.002,
                                     energies[n]],
                }
        return energies

    monkeypatch.setattr(eh, "evaluate_holdout", _fake_eval)

    out_dir = tmp_path / "eval_holdout"
    summary = run_full_holdout_eval(
        _FakeSpec(), object(), mol_specs, reactions, out_dir)

    assert summary["n_reactions"] == 1
    pm = json.loads((out_dir / "per_molecule.json").read_text())
    rec = {r["molecule"]: r for r in pm}["h2"]
    assert rec["cycles_run"] == 3
    assert rec["scf_energy_step_0"] == pytest.approx(-1.18)
    assert rec["scf_energy_residual_2"] == pytest.approx(0.0, abs=1e-9)


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
    # No SCF info -> one-shot sentinels, no per-step columns.
    assert rec["cycles_run"] == 0
    assert "scf_energy_step_0" not in rec


def test_make_per_molecule_record_emits_per_scf_step_trace():
    """With SCF info, the record gains per-cycle energy + residual columns,
    the per-molecule, per-SCF-step convergence the user asked to see."""
    mol_data = {"E_pbe": -76.27}
    scf = {
        "cycles_run": 3,
        "converged": False,
        "total_energy": -76.40,
        "energy_trace": [-76.30, -76.38, -76.40],
    }
    rec = make_per_molecule_record(
        "H2O", mol_data, e_nn_ha=-76.40, in_training_subset=False, scf=scf)
    assert rec["cycles_run"] == 3
    assert rec["scf_converged"] is False
    assert rec["scf_total_energy"] == pytest.approx(-76.40)
    # Per-step total energies preserved verbatim.
    assert rec["scf_energy_step_0"] == pytest.approx(-76.30)
    assert rec["scf_energy_step_1"] == pytest.approx(-76.38)
    assert rec["scf_energy_step_2"] == pytest.approx(-76.40)
    # Residuals = |E_i - E_final|, monotonically shrinking here.
    assert rec["scf_energy_residual_0"] == pytest.approx(0.10)
    assert rec["scf_energy_residual_1"] == pytest.approx(0.02)
    assert rec["scf_energy_residual_2"] == pytest.approx(0.0, abs=1e-12)


def test_make_per_molecule_record_skips_nonfinite_trace_steps():
    """A NaN in the trace (diverged cycle) is skipped, not crashed on."""
    scf = {"cycles_run": 2, "converged": False, "total_energy": -10.0,
           "energy_trace": [-9.5, float("nan")]}
    rec = make_per_molecule_record(
        "X", {"E_pbe": -10.0}, e_nn_ha=-10.0, in_training_subset=False, scf=scf)
    assert rec["scf_energy_step_0"] == pytest.approx(-9.5)
    assert "scf_energy_step_1" not in rec


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


def test_n_nan_union_counts_either_metric_dropped():
    """Reactions dropped (non-finite abs error) in EITHER the NN or the PBE
    metric must be unioned. The two metrics can drop DIFFERENT reactions, so the
    old max(n_nan_nn, n_nan_pbe) undercounts the true dropped set."""
    reactions = [
        {"name": "r1", "reactants": ["x"], "products": [],
         "coeffs": [1.0], "reaction_energy_ref": 0.0},
        {"name": "r2", "reactants": ["y"], "products": [],
         "coeffs": [1.0], "reaction_energy_ref": 0.0},
    ]
    nn = {"x": float("nan"), "y": 1.0}    # NN drops r1 only  -> n_nan_nn = 1
    pbe = {"x": 1.0, "y": float("nan")}   # PBE drops r2 only -> n_nan_pbe = 1
    # union = {r1, r2} = 2; the buggy max(1, 1) would report 1.
    assert eh._n_nan_union(nn, pbe, reactions) == 2

    # Both metrics drop the SAME single reaction -> union is 1 (not 2).
    same = {"x": float("nan"), "y": 1.0}
    assert eh._n_nan_union(same, dict(same), reactions) == 1

    # Nothing dropped -> 0.
    finite = {"x": 1.0, "y": 1.0}
    assert eh._n_nan_union(finite, dict(finite), reactions) == 0


def _mol(name, comp):
    from types import SimpleNamespace
    return SimpleNamespace(name=name, atom_composition=comp)


def test_training_molecule_names_excludes_atoms():
    """Held-out overlap must be molecule-level: training_molecule_names returns
    only multi-atom species, dropping single atoms (atom_composition sums to 1)."""
    from types import SimpleNamespace
    spec = SimpleNamespace(molecules=[
        _mol("hocn", (("H", 1), ("O", 1), ("C", 1), ("N", 1))),  # molecule
        _mol("b2h6", (("B", 2), ("H", 6))),                      # molecule
        _mol("h", (("H", 1),)), _mol("c", (("C", 1),)),          # atoms
        _mol("o", (("O", 1),)), _mol("n", (("N", 1),)),          # atoms
    ])
    assert set(eh.training_molecule_names(spec)) == {"hocn", "b2h6"}


def _verbatim_pool():
    pool = {
        "hcn": {"atom_composition": (("C", 1), ("H", 1), ("N", 1)),
                "charge": 0, "spin": 0,
                "atom": "C 0 0 0; N 0 0 1.15; H 0 0 -1.06"},
        "hnc": {"atom_composition": (("C", 1), ("H", 1), ("N", 1)),
                "charge": 0, "spin": 0,
                "atom": "N 0 0 0; C 0 0 1.17; H 0 0 -1.00"},
        "co2": {"atom_composition": (("C", 1), ("O", 2)), "charge": 0,
                "spin": 0, "atom": "C 0 0 0; O 0 0 1.16; O 0 0 -1.16"},
        "h": {"atom_composition": (("H", 1),), "charge": 0, "spin": 1,
              "atom": "H 0 0 0"},
        "c": {"atom_composition": (("C", 1),), "charge": 0, "spin": 2,
              "atom": "C 0 0 0"},
        "n": {"atom_composition": (("N", 1),), "charge": 0, "spin": 3,
              "atom": "N 0 0 0"},
        "o": {"atom_composition": (("O", 1),), "charge": 0, "spin": 2,
              "atom": "O 0 0 0"},
    }
    return pool


class _VerbatimSpec:
    """Training-spec stub: trained the CHN atomization (reaction form)."""
    def loss_kwargs_dict(self):
        return {"bh76_reactions": [
            {"name": "CHN", "reactants": ["CHN"],
             "products": ["C", "H", "N"],
             "coeffs": [-1.0, 1.0, 1.0, 1.0]}]}
    molecules = ()


def test_trained_reaction_exclusion_identities():
    excl, key_map = eh.trained_reaction_exclusion(_VerbatimSpec(),
                                                  _verbatim_pool())
    assert excl and key_map
    from xcquinox.alec.species_matching import reaction_identity_keys
    hcn_atom = {"name": "w411_hcn_atomization", "reactants": ["hcn"],
                "products": ["h", "c", "n"],
                "coeffs": [-1.0, 1.0, 1.0, 1.0]}
    hnc_atom = {"name": "w411_hnc_atomization", "reactants": ["hnc"],
                "products": ["h", "c", "n"],
                "coeffs": [-1.0, 1.0, 1.0, 1.0]}
    barrier = {"name": "bh76_hcn_to_hcnts", "reactants": ["hcn"],
               "products": ["hnc"], "coeffs": [-1.0, 1.0]}
    assert set(reaction_identity_keys(hcn_atom, key_map)) & excl
    assert not set(reaction_identity_keys(hnc_atom, key_map)) & excl
    assert not set(reaction_identity_keys(barrier, key_map)) & excl


def test_trained_reaction_exclusion_reads_property_and_tuple_specs():
    """The real TrainingSpec exposes loss_kwargs_dict as a PROPERTY and
    loss_kwargs as a tuple of pairs; both shapes must resolve."""
    rxn = {"name": "CHN", "reactants": ["CHN"], "products": ["C", "H", "N"],
           "coeffs": [-1.0, 1.0, 1.0, 1.0]}

    class _PropSpec:
        @property
        def loss_kwargs_dict(self):
            return {"bh76_reactions": [rxn]}

    class _TupleSpec:
        loss_kwargs = (("bh76_reactions", [rxn]),)

    for spec in (_PropSpec(), _TupleSpec()):
        excl, km = eh.trained_reaction_exclusion(spec, _verbatim_pool())
        assert excl and km, type(spec).__name__


def test_finalize_warns_on_empty_exclusion_with_trained_molecules(
        tmp_path, capsys):
    """Strict mode with trained molecules but an empty verbatim-exclusion
    set (a training record predating reaction-form points) must be loud."""
    e = {"a": -1.0, "b": -0.5}
    eh._finalize_holdout_outputs(
        [{"name": "r1", "source_pool": "w411", "reactants": ["a"],
          "products": ["b"], "coeffs": [-1.0, 2.0],
          "reaction_energy_ref": 0.1}],
        e, dict(e), mol_records=[], training_names=("X",), n_species=2,
        out_dir=tmp_path, strict=True, excluded_identities=set(),
        species_key_map={})
    assert "EMPTY verbatim-exclusion" in capsys.readouterr().out


def test_finalize_drops_verbatim_supervised_only(tmp_path):
    """The strict eval drops the trained reaction's pool twin and NOTHING
    else -- reactions merely containing the trained molecule stay."""
    pool = _verbatim_pool()
    spec = _VerbatimSpec()
    excl, key_map = eh.trained_reaction_exclusion(spec, pool)
    reactions = [
        {"name": "w411_hcn_atomization", "source_pool": "w411",
         "reactants": ["hcn"], "products": ["h", "c", "n"],
         "coeffs": [-1.0, 1.0, 1.0, 1.0], "reaction_energy_ref": 0.5},
        {"name": "w411_hnc_atomization", "source_pool": "w411",
         "reactants": ["hnc"], "products": ["h", "c", "n"],
         "coeffs": [-1.0, 1.0, 1.0, 1.0], "reaction_energy_ref": 0.52},
        {"name": "bh76_hcn_to_hcnts", "source_pool": "bh76",
         "reactants": ["hcn"], "products": ["hnc"],
         "coeffs": [-1.0, 1.0], "reaction_energy_ref": 0.02},
        {"name": "w411_co2_atomization", "source_pool": "w411",
         "reactants": ["co2"], "products": ["c", "o"],
         "coeffs": [-1.0, 1.0, 2.0], "reaction_energy_ref": 0.6},
    ]
    e = {"hcn": -93.0, "hnc": -92.98, "co2": -188.0, "h": -0.5,
         "c": -37.8, "n": -54.5, "o": -75.0}
    out = eh._finalize_holdout_outputs(
        reactions, e, dict(e), mol_records=[], training_names=("CHN",),
        n_species=len(e), out_dir=tmp_path, strict=True,
        excluded_identities=excl, species_key_map=key_map)
    import json as _json
    rows = _json.loads((tmp_path / "per_reaction.json").read_text())
    names = sorted(r["name"] for r in rows)
    assert names == ["bh76_hcn_to_hcnts", "w411_co2_atomization",
                     "w411_hnc_atomization"], names
    assert out["n_dropped_overlap"] == 1


def test_split_held_out_keeps_permuted_name_twins_together():
    """The pool's duplicate barriers (same physics, permuted-reactant names)
    must land on the SAME side of the val/test split -- the name-keyed hash
    put one copy per slice, so validation-best selection saw four reported
    test barriers."""
    twins = [
        {"name": "bh76_h_hf_to_hfhts", "reactants": ["h", "hf"],
         "products": ["hfhts"], "reaction_energy_ref": 17.7},
        {"name": "bh76_hf_h_to_hfhts", "reactants": ["hf", "h"],
         "products": ["hfhts"], "reaction_energy_ref": 17.7},
    ]
    filler = [{"name": f"r{i}", "reactants": [f"a{i}"], "products": [f"b{i}"],
               "reaction_energy_ref": 1.0} for i in range(20)]
    val, test = eh.split_held_out(twins + filler, val_frac=0.5)
    val_names = {r["name"] for r in val}
    twin_names = {t["name"] for t in twins}
    assert twin_names <= val_names or twin_names.isdisjoint(val_names)


def test_held_out_filter_names_include_composition_aliases():
    """A trained Hill-named molecule (CHN) must expand to its pool twin (hcn)
    so the strict filter can drop the twin's reactions; name matching alone
    left the trained reaction "held-out". Composition twins that are NOT
    trained (co2) stay out of the filter set."""
    from types import SimpleNamespace
    spec = SimpleNamespace(molecules=[
        _mol("CHN", (("C", 1), ("H", 1), ("N", 1))),
        _mol("h", (("H", 1),)),
    ])
    pool = {
        "hcn": {"atom_composition": (("C", 1), ("H", 1), ("N", 1)),
                "charge": 0, "spin": 0,
                "atom": "C 0 0 0; N 0 0 1.15; H 0 0 -1.06"},
        "co2": {"atom_composition": (("C", 1), ("O", 2)),
                "charge": 0, "spin": 0,
                "atom": "C 0 0 0; O 0 0 1.16; O 0 0 -1.16"},
    }
    names = eh.held_out_filter_names_with_aliases(spec, pool)
    assert "CHN" in names and "hcn" in names
    assert "co2" not in names and "h" not in names
    kept, dropped = eh.filter_reactions(
        [{"name": "w411_hcn_atomization", "reactants": ["hcn"],
          "products": ["h", "c", "n"], "coeffs": [1.0, -1.0, -1.0, -1.0],
          "reaction_energy_ref": 0.5},
         {"name": "w411_co2_atomization", "reactants": ["co2"],
          "products": ["c", "o"], "coeffs": [1.0, -1.0, -2.0],
          "reaction_energy_ref": 0.6}],
        names, strict=True)
    assert [r["name"] for r in kept] == ["w411_co2_atomization"]
    assert [r["name"] for r in dropped] == ["w411_hcn_atomization"]


def test_filter_reactions_molecule_level_keeps_atom_sharing_holdout():
    """An atomization whose ATOMS (not its MOLECULE) are trained must stay
    HELD-OUT. The old full-species list (atoms included) wrongly dropped every
    atomization on shared atoms -- 6 training reactions discarded ~all of W4-11."""
    from types import SimpleNamespace
    spec = SimpleNamespace(molecules=[
        _mol("hocn", (("H", 1), ("O", 1), ("C", 1), ("N", 1))),
        _mol("h", (("H", 1),)), _mol("o", (("O", 1),)),
        _mol("c", (("C", 1),)), _mol("n", (("N", 1),)),
    ])
    reactions = [
        # co2 shares atoms c,o with the trained ATOMS but co2 is NOT trained.
        {"name": "co2_atomization", "reactants": ["co2"],
         "products": ["c", "o"], "coeffs": [1.0, -1.0, -2.0],
         "reaction_energy_ref": 0.0},
        # hocn IS a trained molecule -> in-sample.
        {"name": "hocn_atomization", "reactants": ["hocn"],
         "products": ["h", "o", "c", "n"], "coeffs": [1.0, -1, -1, -1, -1],
         "reaction_energy_ref": 0.0},
    ]
    mol_names = eh.training_molecule_names(spec)
    kept, dropped = eh.filter_reactions(reactions, mol_names, strict=True)
    assert {r["name"] for r in kept} == {"co2_atomization"}
    assert {r["name"] for r in dropped} == {"hocn_atomization"}
    # Regression guard: the OLD full list (atoms included) drops BOTH.
    full_names = [m.name for m in spec.molecules]
    assert eh.filter_reactions(reactions, full_names, strict=True)[0] == []


def test_holdout_overlap_molecule_level_on_real_pools():
    """On the REAL BH76 + W4-11 pools: a small training set (a few molecules +
    ALL atoms) holds out almost the whole pool under the molecule-level overlap,
    where the old atom-level overlap kept almost nothing. Verifies the FIX
    brings the correct species into the held-out set on real data."""
    from types import SimpleNamespace
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    specs, reactions = load_full_held_out_pools()
    atoms = [s for s in specs.values() if eh._spec_is_atom(s)]
    mols = [s for s in specs.values() if not eh._spec_is_atom(s)]
    assert len(atoms) >= 10 and len(mols) >= 150, (len(atoms), len(mols))

    # Realistic spec_0000-style training set: 3 molecules + every atom anchor.
    trained = mols[:3]
    training_spec = SimpleNamespace(molecules=trained + atoms)
    mol_names = eh.training_molecule_names(training_spec)
    full_names = [getattr(m, "name") for m in training_spec.molecules]
    assert set(mol_names) == {m.name for m in trained}  # atoms excluded

    kept_new, dropped_new = eh.filter_reactions(reactions, mol_names, strict=True)
    kept_old, _ = eh.filter_reactions(reactions, full_names, strict=True)

    n = len(reactions)
    assert len(kept_new) > 0.7 * n, (len(kept_new), n)   # molecule-level: pool survives
    assert len(kept_old) < 0.25 * n, (len(kept_old), n)  # atom-level bug: pool nuked
    assert len(kept_new) > 3 * len(kept_old)
    # Every newly-dropped reaction genuinely references a TRAINED molecule
    # (no atom-level leak).
    trained_names = {m.name for m in trained}
    for r in dropped_new:
        rn = set(r.get("reactants", [])) | set(r.get("products", []))
        assert rn & trained_names, f"atom-leak: {r.get('name')} dropped w/o a trained molecule"


def test_holdout_pools_reaction_integrity_both_pools():
    """Phase-0 integrity check: every reaction in BOTH real pools is
    mass-balanced and has a finite reference. W4-11 atomization refs are strictly
    positive; BH76 barrier refs may be negative ONLY for the four gas-phase
    ion-molecule SN2 reactions (submerged barriers), each with an anion reactant."""
    from xcquinox.alec.full_benchmark_pools import load_full_bh76, load_full_w411
    for pool, (specs, rxns), atomization in (
        ("bh76", load_full_bh76(), False),
        ("w411", load_full_w411(), True),
    ):
        for r in rxns:
            names = list(r["reactants"]) + list(r["products"])
            assert len(names) == len(r["coeffs"]), (pool, r["name"])
            bal = {}
            for nm, c in zip(names, r["coeffs"]):
                for el, cnt in dict(specs[nm].atom_composition).items():
                    bal[el] = bal.get(el, 0.0) + c * cnt
            assert all(abs(v) < 1e-9 for v in bal.values()), \
                f"{pool}:{r['name']} not mass-balanced: {bal}"
            ref = r["reaction_energy_ref"]
            assert isinstance(ref, (int, float)) and math.isfinite(ref), (pool, r["name"])
            if atomization:
                assert ref > 0, f"W4-11 atomization ref must be > 0: {r['name']}={ref}"
        if pool == "bh76":
            neg = [r for r in rxns if r["reaction_energy_ref"] <= 0]
            assert len(neg) == 4, [r["name"] for r in neg]
            for r in neg:  # legitimate submerged SN2 barrier -> anion reactant
                assert any(s.endswith("-") for s in r["reactants"]), (r["name"], r["reactants"])


def test_holdout_overlap_charge_and_case_aware_no_leak():
    """Phase-0 integrity check (NON-circular). The earlier oracle used the SAME
    comp==1 atom rule as the code and never trained anions/case-twins, so it was
    circular (the earlier oracle could not reach the two real leaks). This
    oracle uses an INDEPENDENT rule -- a universal anchor is a NEUTRAL monatomic
    -- and case-folds names, and it actually TRAINS the monatomic anions (f-,
    cl-) and cross-pool case-twins (NH3/nh3). Asserts molecule-level overlap ==
    oracle with ZERO case-insensitive leakage on the COMBINED pool.

    Guards both leaks: (A) anion-as-atom, (B) case-variant."""
    from types import SimpleNamespace
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    specs, rxns = load_full_held_out_pools()

    def cf(s):
        return str(s).casefold()

    def rcf(r):
        return {cf(x) for x in (set(r["reactants"]) | set(r["products"]))}

    def neutral_monatomic(s):          # INDEPENDENT oracle rule (charge + comp)
        comp = dict(getattr(s, "atom_composition", ()) or ())
        return sum(comp.values()) == 1 and int(getattr(s, "charge", 0) or 0) == 0

    by = specs
    mols = [s for s in specs.values() if not neutral_monatomic(s)]
    subsets = [
        ("f-", [by["f-"]]),                                   # Vector A
        ("cl-", [by["cl-"]]),
        ("nh3", [by["nh3"]]),                                 # Vector B (lower)
        ("NH3", [by["NH3"]]),                                 # Vector B (upper)
        ("f-,cl-,nh3", [by[n] for n in ("f-", "cl-", "nh3")]),
        ("25mol+anions", mols[:25] + [by["f-"], by["cl-"]]),
    ]
    for label, trained in subsets:
        ts = SimpleNamespace(molecules=trained)
        mol_names = set(eh.training_molecule_names(ts))
        kept, dropped = eh.filter_reactions(rxns, mol_names, strict=True)
        mol_cf = {cf(s.name) for s in trained if not neutral_monatomic(s)}
        oracle_kept = {r["name"] for r in rxns if not (rcf(r) & mol_cf)}
        assert {r["name"] for r in kept} == oracle_kept, f"{label}: kept != oracle"
        assert all(not (rcf(r) & mol_cf) for r in kept), f"{label}: case-insensitive LEAK"
        assert len(kept) + len(dropped) == len(rxns), f"{label}: not conserved"

    # (A) monatomic anions are MOLECULES; neutral monatomics are excluded.
    assert set(eh.training_molecule_names(
        SimpleNamespace(molecules=[by["f-"], by["cl-"]]))) == {"f-", "cl-"}
    assert eh.training_molecule_names(
        SimpleNamespace(molecules=[by[n] for n in ("h", "f", "cl", "o")])) == ()
    # (B) training a lower-case twin drops the upper-case reaction (no leak).
    nm = set(eh.training_molecule_names(SimpleNamespace(molecules=[by["nh3"]])))
    kept_nh3, _ = eh.filter_reactions(rxns, nm, strict=True)
    assert all("nh3" not in {s.casefold() for s in
               (set(r["reactants"]) | set(r["products"]))} for r in kept_nh3), \
        "NH3/nh3 case-twin leaked into held-out"


# ---------------------------------------------------------------------------
# held-out density errors (NN-vs-CCSD + model-free PBE-vs-CCSD)
# ---------------------------------------------------------------------------

def test_density_errors_for_record_all_none_for_atom_and_no_ref():
    # atoms: density matching skipped (mirrors DensityRMSEMetric)
    out = eh.density_errors_for_record(None, {"atom_composition": (("H", 1),)})
    assert set(out) == set(eh._DENSITY_RECORD_KEYS)
    assert all(v is None for v in out.values())
    # no CCSD reference loaded (external_data_path unresolved) -> all None,
    # so runs without benchmark refs keep the historical schema
    out2 = eh.density_errors_for_record(
        None, {"atom_composition": (("H", 2),), "rho_ref_grid": None})
    assert all(v is None for v in out2.values())


def test_density_errors_for_record_pbe_closed_form(monkeypatch):
    import numpy as np
    import xcquinox.alec.evaluation as ev_mod

    class FakeMetric:
        def compute(self, model, md, solver_config=None):
            return {"density_rmse": 0.123, "density_l1": 0.045,
                    "density_eps_l1": 0.011,
                    "ref_density_method": "ccsd"}

    monkeypatch.setattr(ev_mod, "DensityRMSEMetric", FakeMetric)
    md = {
        "atom_composition": (("H", 2),),
        "rho_ref_grid": np.array([2.0, 1.0]),
        "rho_grid": np.array([2.5, 0.5]),       # PBE density on the same grid
        "grid_weights": np.array([3.0, 1.0]),
        "ref_density_method": "ccsd",
    }
    out = eh.density_errors_for_record(object(), md, solver_config=None)
    # hand-computed weighted errors: diff = [0.5, -0.5], wsum = 4
    # RMSE = sqrt((3*0.25 + 1*0.25)/4) = 0.5 ; L1 = (3*0.5 + 1*0.5)/4 = 0.5
    assert out["density_rmse_pbe"] == pytest.approx(0.5)
    assert out["density_l1_pbe"] == pytest.approx(0.5)
    # DFS Eq. 20 per-electron L1: sum(w|diff|)/N_e = 2/(3*2 + 1*1) = 2/7,
    # deliberately distinct from the volume-averaged L1 (0.5)
    assert out["density_eps_l1_pbe"] == pytest.approx(2.0 / 7.0)
    assert out["n_electrons"] == pytest.approx(7.0)
    assert out["grid_weight_sum"] == pytest.approx(4.0)
    # NN channel comes from DensityRMSEMetric (stubbed; model-dependent)
    assert out["density_rmse"] == pytest.approx(0.123)
    assert out["density_l1"] == pytest.approx(0.045)
    assert out["density_eps_l1"] == pytest.approx(0.011)
    assert out["ref_density_method"] == "ccsd"


def test_density_errors_for_record_pbe_shape_mismatch_raises(monkeypatch):
    import numpy as np
    import xcquinox.alec.evaluation as ev_mod

    class FakeMetric:
        def compute(self, model, md, solver_config=None):
            return {"density_rmse": 0.0, "density_l1": 0.0,
                    "ref_density_method": "ccsd"}

    monkeypatch.setattr(ev_mod, "DensityRMSEMetric", FakeMetric)
    md = {
        "atom_composition": (("H", 2),),
        "rho_ref_grid": np.ones(3),
        "rho_grid": np.ones(5),
        "grid_weights": np.ones(3),
    }
    with pytest.raises(ValueError, match="density shape mismatch"):
        eh.density_errors_for_record(object(), md)


def test_make_per_molecule_record_density_kwarg():
    md = {"E_pbe": -1.0}
    rec = eh.make_per_molecule_record("x", md, -1.1, in_training_subset=False)
    for k in eh._DENSITY_RECORD_KEYS:
        assert k in rec and rec[k] is None     # omitted -> historical all-None
    dens = {"density_rmse": 1e-4, "density_l1": 2e-5,
            "density_rmse_pbe": 3e-4, "density_l1_pbe": 4e-5,
            "density_eps_l1": 5e-5, "density_eps_l1_pbe": 6e-5,
            "n_electrons": 10.0, "grid_weight_sum": 123.4,
            "ref_density_method": "ccsd"}
    rec2 = eh.make_per_molecule_record("x", md, -1.1,
                                       in_training_subset=False, density=dens)
    for k, v in dens.items():
        assert rec2[k] == v


# 2026-06-20 (WS3): deterministic val/test split of the held-out pools. The val
# slice drives in-training early-stop/selection; the test slice is what eval
# REPORTS. Must be stable (same partition every run/process/order) and a clean
# partition, so val never leaks into the reported test metric.
def _mk_rxns(n):
    # Distinct species per reaction: the split hashes the PHYSICAL identity
    # (sorted species tuples), so a shared-species fixture would collapse to
    # one identity and land every reaction on one side.
    return [{"name": f"rxn_{i:03d}", "source_pool": "w411",
             "reactants": [f"a{i:03d}"], "products": [f"b{i:03d}"],
             "coeffs": [1.0, -1.0],
             "reaction_energy_ref": float(i)} for i in range(n)]


def test_split_held_out_is_deterministic_and_partitions():
    rxns = _mk_rxns(200)
    val, test = eh.split_held_out(rxns, val_frac=0.2)
    val_names = {r["name"] for r in val}
    test_names = {r["name"] for r in test}
    assert val_names.isdisjoint(test_names)
    assert val_names | test_names == {r["name"] for r in rxns}
    assert 0.12 < len(val) / len(rxns) < 0.28      # ~20% in val
    import random
    shuffled = list(rxns)
    random.Random(0).shuffle(shuffled)
    val2, _ = eh.split_held_out(shuffled, val_frac=0.2)
    assert {r["name"] for r in val2} == val_names   # order-independent + stable


def test_split_held_out_rejects_bad_val_frac():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            eh.split_held_out(_mk_rxns(3), val_frac=bad)


# 2026-06-24: DFS tail loss -> held-out eval must REPORT the convergence-aware
# tail-weighted mean (denoised), not the arbitrary final SCF step, while still
# recording the raw final energy + full trace for forensics.
def test_evaluate_holdout_reports_tail_weighted_energy(monkeypatch):
    import jax.numpy as jnp
    import xcquinox.alec.solver as solver_mod
    from xcquinox.alec.eval_holdout import evaluate_holdout
    from xcquinox.alec.solver import SolverConfig, SolverMode, SolverBackend
    from xcquinox.alec.oneshot import tail_weighted_mean_energy

    # non-converged period-2-ish tail; final step (-76.3) is an arbitrary phase.
    trace = jnp.array([-76.0, -76.2, -76.5, -76.3])

    class FakeResult:
        total_energy = jnp.array(-76.3)
        cycles_run = jnp.int32(4)
        converged = jnp.array(False)
        energy_trace = trace

    monkeypatch.setattr(solver_mod, "run_scf",
                        lambda cfg, model, md, forward_only=False: FakeResult())

    full = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL, max_cycles=4,
        scf_loss_use_tail=True, scf_loss_tail=2, scf_loss_weight_power=2.0,
    )
    info = {}
    out = evaluate_holdout(None, {"X": {}}, solver_config=full, scf_info_out=info)
    expected = float(tail_weighted_mean_energy(trace, 2, 2.0))
    assert out["X"] == pytest.approx(expected, abs=1e-12)
    assert out["X"] != pytest.approx(-76.3, abs=1e-6)  # NOT the final step
    # raw final + reported both recorded for forensics
    assert info["X"]["total_energy"] == pytest.approx(-76.3, abs=1e-12)
    assert info["X"]["reported_energy"] == pytest.approx(expected, abs=1e-12)
    assert info["X"]["energy_trace"] == pytest.approx([-76.0, -76.2, -76.5, -76.3])


def test_evaluate_holdout_tail_off_reports_final_step(monkeypatch):
    import jax.numpy as jnp
    import xcquinox.alec.solver as solver_mod
    from xcquinox.alec.eval_holdout import evaluate_holdout
    from xcquinox.alec.solver import SolverConfig, SolverMode, SolverBackend

    trace = jnp.array([-76.0, -76.2, -76.5, -76.3])

    class FakeResult:
        total_energy = jnp.array(-76.3)
        cycles_run = jnp.int32(4)
        converged = jnp.array(True)
        energy_trace = trace

    monkeypatch.setattr(solver_mod, "run_scf",
                        lambda cfg, model, md, forward_only=False: FakeResult())
    full_off = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL, max_cycles=4,
    )  # scf_loss_use_tail defaults False
    out = evaluate_holdout(None, {"X": {}}, solver_config=full_off)
    assert out["X"] == pytest.approx(-76.3, abs=1e-12)  # final step, unchanged


def test_precompute_holdout_forwards_seed_fields(monkeypatch):
    """The holdout precompute hands the spec's seed identity to the supply
    layer for every pool species."""
    import xcquinox.alec as alec
    from xcquinox.alec import eval_holdout as eh

    captured = {}

    def _fake_precompute(spec, descriptors=(), required_keys=(),
                         auxbasis=None, orientation_lock_strength=0.0, **kw):
        captured.update(kw)
        return {"name": spec.name}

    monkeypatch.setattr(alec, "precompute_fixed_density_data",
                        _fake_precompute)
    from types import SimpleNamespace
    specs = {"h2o": SimpleNamespace(name="h2o")}
    eh.precompute_holdout(specs, seed_source="scan",
                          seed_cache_dir="/seeds", seed_density_fit=True)
    assert captured == {"seed_source": "scan", "seed_cache_dir": "/seeds",
                        "seed_density_fit": True}


def test_precompute_holdout_for_spec_derives_seed_from_solver_config(
        monkeypatch):
    from types import SimpleNamespace
    from xcquinox.alec import eval_holdout as eh
    from xcquinox.alec.solver import SolverConfig, SolverMode

    captured = {}

    def _fake_holdout(mol_specs, descriptors=(), *, required_keys=(),
                      auxbasis=None, orientation_lock_strength=0.0, **kw):
        captured.update(kw)
        return {}

    monkeypatch.setattr(eh, "precompute_holdout", _fake_holdout)
    sc = SolverConfig(mode=SolverMode.FULL, max_cycles=3,
                      seed_source="minao", density_fit=True)
    spec = SimpleNamespace(
        solver_config=sc,
        arch=SimpleNamespace(materialize_descriptors=lambda: ()))
    eh.precompute_holdout_for_spec(spec, {})
    assert captured["seed_source"] == "minao"
    assert captured["seed_cache_dir"] is None
    assert captured["seed_density_fit"] is True


def test_coldstart_solver_config_override():
    """The shared cold-start override: minao seed, the Letter's 25 cycles,
    conv_tol far below any per-cycle step (the latched freeze never masks
    the trajectory); mode stays FULL; everything else preserved."""
    from xcquinox.alec.eval_holdout import coldstart_solver_config
    from xcquinox.alec.solver import (SolverBackend, SolverConfig,
                                      SolverMode)
    sc = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                      max_cycles=3, scf_loss_use_tail=True,
                      orientation_lock_strength=3e-05,
                      seed_cache_dir="/seeds")
    cold = coldstart_solver_config(sc)
    assert cold.seed_source == "minao"
    assert cold.max_cycles == 25
    assert cold.conv_tol == 1e-12
    assert cold.mode == SolverMode.FULL
    assert cold.scf_loss_use_tail is True
    assert cold.orientation_lock_strength == 3e-05
    # a minao seed needs no cache dir; the override clears it
    assert cold.seed_cache_dir is None


def test_coldstart_solver_config_rejects_non_full():
    """A ONESHOT-trained solver cannot be cold-started (the override would
    be a different protocol entirely); the constructor validation fires."""
    from xcquinox.alec.eval_holdout import coldstart_solver_config
    from xcquinox.alec.solver import SolverConfig, SolverMode
    sc = SolverConfig(mode=SolverMode.ONESHOT, max_cycles=0)
    with pytest.raises(ValueError):
        coldstart_solver_config(sc)
