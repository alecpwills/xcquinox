"""Tests for the energy-term-weight measurement job.

Four things are pinned here. The COMMAND SURFACE: what the sweep runs when
nothing is said, and what ``--smoke`` substitutes, because a sweep launched at
the wrong basis, grid or correlation objective measures a different question
and costs a node-day to find out. The MEASURED QUANTITIES: the gate is
``max |dE_x + dE_c|`` over the atoms and ``max |dAE|`` over the molecules, both
reduced from per-system residuals, on synthetic residuals where the two
channels disagree about which system is worst and by how much -- reading one
channel alone, or every system as an atom, gives a different number on the same
fit. The RECOMMENDATION RULE: exercised on synthetic tables, including every
edge the real table can present -- no weight clearing the gates, a weight
clearing one and missing the other, a weight clearing them only on part of the
architecture set, a weight buying the energy by destroying the point-wise fit,
a gate quantity that was never measured, and a diverged cell, and -- because
a sweep batched over architectures accumulates into one table -- that a verdict
NAMES the architectures it read and says which sweep defaults it did not. The
JOB SCRIPT: the house shell idiom, the standing mail directives, the wall
against the arithmetic that asks for it, and the exact invocation, since a
script that activates the wrong environment or drops a flag produces a table
that looks valid and is not. The EXIT: the entry point is read as source, so
the hard exit that keeps a corrupted interpreter teardown from replacing a
completed sweep's code is pinned whether or not the teardown happens to abort
on the machine running the tests, and an exception that escapes the sweep is
executed end to end for its own exit code and its partial table.

The ``--smoke`` end-to-end leg runs the real sweep at a two-system STO-3G
identity in a SUBPROCESS, at BOTH polarizations -- ~20 s each, measured -- so
it is neither slow-marked nor able to take the test session down with it if JAX
aborts at interpreter exit.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE / "probe_pretrain_energy_weight.py"
_SBATCH = _HERE / "probe_pretrain_energy_weight.sbatch"

#: Sentinel for "this synthetic row does not distinguish the two maxima".
_SAME = object()


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


pw = _load("probe_pretrain_energy_weight")


# --------------------------------------------------------------------------- #
# Argument parsing and defaults
# --------------------------------------------------------------------------- #

def _args(*extra):
    return pw.parse_args(["--data-dir", "d", "--out", "o", *extra])


def test_production_defaults_are_the_swept_identity():
    a = _args()
    assert a.archs == ("deep_3x16", "deep_cusp_3x16", "deep_rung35_3x16",
                       "deep_rung35_attn_3x16", "deep_rung35ms_3x16",
                       "deep_mgga_3x16")
    assert a.weights == (0.0, 0.1, 1.0, 10.0, 100.0)
    assert a.n_steps == 1000
    assert a.basis == "def2-svp"
    # Level 3 is a floor, not a taste: below it the generator refuses the
    # degenerate open-shell atoms this set is full of.
    assert a.grid_level == 3
    assert a.seed == 42
    assert a.loss_weighting == "integration"
    assert a.smoke is False
    assert a.tol_atom_mha == 1.0        # the certificate's tol_atom
    assert a.tol_ae_kcal == 1.0         # the certificate's tol_AE
    assert a.margin_fraction == 0.5
    assert a.pointwise_factor == 3.0
    assert a.resume is False
    # The production value of the run-level flag: every configuration of the
    # dfs6311 v3-v5 lineage and the v6 template set
    # use_polarized_correlation: true, so the weight is measured on the
    # objective it will be applied to rather than on the registry default.
    assert a.polarized is True


#: Worst-case atomization-energy offset from the parent, in kcal/mol, of every
#: architecture SPEC_pretrain_fidelity_program.md Section 2 measured: the
#: largest magnitude of its H2O / N2 / CH4 triple. These are the offsets the
#: certificate exists to close, so they are what decides which architectures a
#: weight has to be measured on.
_SECTION_2_WORST_OFFSET_KCAL = {
    "deep_3x16": 4.2,                  # -2.5 / -4.2 / -2.4
    "deep_attn_3x16": 4.1,             # -2.3 / -4.1 / -3.1
    "deep_cusp_3x16": 25.7,            # -13.2 / -4.2 / -25.7
    "deep_rung35_3x16": 29.1,          # -13.5 / -3.5 / -29.1
    "deep_rung35_attn_3x16": 56.1,     # -29.5 / -20.4 / -56.1
    "deep_rung35ms_3x16": 42.8,        # -22.0 / -30.9 / -42.8
    "deep_mgga_3x16": 55.9,            # -30.5 / -55.9 / -20.8
}


def test_the_default_set_holds_the_largest_recorded_parent_offsets():
    """The default set is the six worst offenders on record, not a sample.

    The certificate gate is "every architecture clears both halves", and the
    train array depends ``afterok`` on the pretrain ARRAY, so a single
    architecture whose certificate fails blocks all 341 cells of campaign v6
    rather than its own eleven. A weight chosen on architectures whose
    pre-certificate offsets were small is then extrapolated onto the ones that
    decide whether anything runs at all.

    Ranked by worst-case offset, Section 2's table puts
    ``deep_rung35_attn_3x16`` (56.1 kcal/mol) and ``deep_rung35ms_3x16``
    (42.8) first and third; both were outside the four-architecture default.
    The set is now the six largest, which is every architecture the table
    measured except ``deep_attn_3x16`` -- the smallest offset of the seven
    (4.1), and a descriptor-free network whose family ``deep_3x16`` already
    represents. The six also span one member of every descriptor family the
    campaign carries: none, cusp, rung-3.5, rung-3.5 with attention,
    rung-3.5 multishell, and meta-GGA.
    """
    defaults = set(pw.DEFAULT_ARCHS)
    assert len(pw.DEFAULT_ARCHS) == len(defaults) == 6
    ranked = sorted(_SECTION_2_WORST_OFFSET_KCAL,
                    key=lambda n: -_SECTION_2_WORST_OFFSET_KCAL[n])
    assert defaults == set(ranked[:6]), sorted(defaults ^ set(ranked[:6]))
    # The two the four-architecture set omitted, named so a future narrowing
    # of the default has to delete this line rather than drift past it.
    assert {"deep_rung35_attn_3x16", "deep_rung35ms_3x16"} <= defaults
    # No architecture inside the set is smaller than the one left out.
    excluded = set(_SECTION_2_WORST_OFFSET_KCAL) - defaults
    assert excluded == {"deep_attn_3x16"}
    assert min(_SECTION_2_WORST_OFFSET_KCAL[n] for n in defaults) > \
        max(_SECTION_2_WORST_OFFSET_KCAL[n] for n in excluded)


def test_every_default_architecture_is_in_the_registry():
    """A default the registry does not carry fails at the first cell, hours
    into a reservation, with the data files already built."""
    from xcquinox.alec.config import ARCHITECTURES
    for name in pw.DEFAULT_ARCHS:
        assert name in ARCHITECTURES, name


def test_the_default_set_needs_both_parent_densities_and_only_those():
    """Six architectures, still two data files.

    The sweep generates one pretraining data file per distinct
    (polarization, parent) pair and reuses it across cells, so the datagen
    hour in the wall derivation is a property of the PARENT set, not of the
    architecture count. Both added architectures are rung-3.5 and PBE-parent,
    so the set still resolves to exactly {pbe, scan} and the derivation's
    one-hour data allowance is unchanged.
    """
    from xcquinox.alec.config import get_architecture
    from xcquinox.alec.pretrain_data_gen import resolve_parent_density
    parents = {resolve_parent_density(get_architecture(n), "auto")
               for n in pw.DEFAULT_ARCHS}
    assert parents == {"pbe", "scan"}
    assert resolve_parent_density(
        get_architecture("deep_rung35_attn_3x16"), "auto") == "pbe"
    assert resolve_parent_density(
        get_architecture("deep_rung35ms_3x16"), "auto") == "pbe"


def test_polarization_is_a_flag_and_defaults_to_the_production_value():
    assert _args("--polarized").polarized is True
    assert _args("--no-polarized").polarized is False
    with pytest.raises(SystemExit):                 # mutually exclusive
        _args("--polarized", "--no-polarized")


def test_the_sweep_patches_the_run_level_polarization_onto_every_arch(
        tmp_path, monkeypatch):
    """The flag reaches the architectures, not just the identity block.

    No registry architecture carries use_polarized_correlation, so a sweep that
    read the flag off the architecture would measure the unpolarized objective
    whatever the command line said.
    """
    seen = {}

    def _cell(arch, arch_name, data_path, work_dir, *, weight, **kwargs):
        seen[(arch_name, weight)] = bool(arch.use_polarized_correlation)
        return _row(arch_name, weight, 0.2)

    monkeypatch.setattr(pw, "ensure_data",
                        lambda *a, **k: "/nonexistent/pretrain_data.npz")
    monkeypatch.setattr(pw, "run_cell", _cell)
    for flag, want in (("--polarized", True), ("--no-polarized", False)):
        seen.clear()
        pw.main(["--data-dir", str(tmp_path), "--out",
                 str(tmp_path / f"t{want}.json"), "--archs", "deep_3x16",
                 "--weights", "0,1", flag])
        assert set(seen.values()) == {want}, seen
        identity = json.loads(
            (tmp_path / f"t{want}.json").read_text())["identity"]
        assert identity["polarized"] is want


def test_the_data_file_follows_the_polarization_flag(tmp_path, monkeypatch):
    """The generated file is the one run_pretrain will open for that arch."""
    calls = []
    monkeypatch.setattr(pw, "ensure_data",
                        lambda *a, **k: (calls.append(k), "/none.npz")[1])
    monkeypatch.setattr(
        pw, "run_cell",
        lambda arch, name, *a, **k: _row(name, k["weight"], 0.2))
    pw.main(["--data-dir", str(tmp_path), "--out", str(tmp_path / "t.json"),
             "--archs", "deep_3x16,deep_mgga_3x16", "--weights", "1",
             "--polarized"])
    assert [(c["polarized"], c["reference_xc"]) for c in calls] == [
        (True, "pbe"), (True, "scan")]


def test_smoke_substitutes_a_seconds_long_identity():
    a = _args("--smoke")
    assert a.smoke is True
    assert a.basis == "sto-3g"
    assert a.grid_level == 1
    assert a.n_steps == 5
    assert a.archs == ("deep_3x16", "deep_mgga_3x16")
    assert a.weights == (0.0, 1.0)
    # Both rungs, so the smoke exercises both parent densities.
    assert len(set(a.archs)) == 2


def test_explicit_flags_win_over_smoke():
    a = _args("--smoke", "--basis", "def2-svp", "--grid-level", "3",
              "--n-steps", "11", "--archs", "deep_3x16", "--weights", "2")
    assert (a.basis, a.grid_level, a.n_steps) == ("def2-svp", 3, 11)
    assert a.archs == ("deep_3x16",)
    assert a.weights == (2.0,)


def test_weights_are_deduplicated_and_ascending():
    # The rule takes the SMALLEST clearing weight, so the sweep order and the
    # read order have to be the same one.
    assert _args("--weights", "10, 1,1 0").weights == (0.0, 1.0, 10.0)


def test_archs_accept_commas_and_spaces():
    assert _args("--archs", "a, b c").archs == ("a", "b", "c")


@pytest.mark.parametrize("bad", [
    ["--weights", "-1"],            # a negative weight is not an objective
    ["--weights", "nan"],
    ["--weights", "inf"],
    ["--weights", "banana"],
    ["--weights", ""],
    ["--archs", ""],
    ["--n-steps", "0"],
    ["--grid-level", "-1"],
    ["--recon-rtol", "0"],
    ["--tol-atom-mha", "0"],
    ["--tol-ae-kcal", "0"],
    ["--tol-ae-kcal", "-1"],
    ["--margin-fraction", "-0.5"],
    ["--pointwise-factor", "nan"],
    ["--loss-weighting", "sortof"],
])
def test_bad_arguments_are_refused(bad):
    with pytest.raises(SystemExit):
        _args(*bad)


@pytest.mark.parametrize("missing", [
    ["--out", "o"],                  # no --data-dir
    ["--data-dir", "d"],             # no --out
])
def test_data_dir_and_out_are_required(missing):
    with pytest.raises(SystemExit):
        pw.parse_args(missing)


# --------------------------------------------------------------------------- #
# The recommendation rule
# --------------------------------------------------------------------------- #

def _row(arch, weight, max_mha, loss_x=1.0e-3, loss_c=1.0e-3, *,
         atom_mha=_SAME, n_atoms=2, ae_kcal=None, n_ae=0):
    """One measured cell. ``atom_mha`` defaults to the all-system maximum, so
    a table written for the point-wise or cap edges does not have to state a
    split it is not testing; the atom / atomization gates are exercised by the
    tests that pass them explicitly."""
    return {"arch": arch, "weight": float(weight), "final_loss_x": loss_x,
            "final_loss_c": loss_c, "max_dE_xc_mHa": max_mha,
            "rms_dE_xc_mHa": (None if max_mha is None else 0.5 * max_mha),
            "max_atom_dE_xc_mHa": (max_mha if atom_mha is _SAME
                                   else atom_mha),
            "n_atom_systems": n_atoms,
            "max_dAE_kcal": ae_kcal, "n_ae_molecules": n_ae}


def test_smallest_clearing_weight_wins():
    rows = [_row("a", 0.0, 9.0), _row("b", 0.0, 8.0),
            _row("a", 1.0, 0.4), _row("b", 1.0, 0.3),
            _row("a", 10.0, 0.1), _row("b", 10.0, 0.1)]
    out = pw.recommend(rows)
    assert out["cleared"] is True
    # Both 1 and 10 clear; the rule says the smallest, not the best.
    assert out["weight"] == 1.0


def test_the_gate_is_the_max_over_every_architecture():
    # Weight 1 clears on 'a' and misses on 'b'; the rule is "every
    # architecture", so the choice moves up to 10.
    rows = [_row("a", 0.0, 9.0), _row("b", 0.0, 9.0),
            _row("a", 1.0, 0.1), _row("b", 1.0, 0.9),
            _row("a", 10.0, 0.1), _row("b", 10.0, 0.2)]
    out = pw.recommend(rows)
    assert (out["cleared"], out["weight"]) == (True, 10.0)


def test_margin_is_half_the_tolerance_and_the_boundary_clears():
    # A weight sitting exactly ON the margin clears it. Asserting the weight
    # alone cannot see a flipped comparison: the fallback returns the same
    # weight (it minimizes the worst gate quantity), so the verdict itself has
    # to be asserted.
    out = pw.recommend([_row("a", 0.0, 9.0), _row("a", 1.0, 0.5)])
    assert (out["cleared"], out["weight"]) == (True, 1.0)
    entry = next(e for e in out["per_weight"] if e["weight"] == 1.0)
    assert entry["gate_ok"] is True
    out = pw.recommend([_row("a", 0.0, 9.0), _row("a", 1.0, 0.500001)])
    assert out["cleared"] is False


def test_the_atom_gate_reads_the_atoms_not_every_system():
    # The certificate bounds ATOMS at tol_atom; a molecule's absolute XC error
    # is bounded through the atomization energy instead. A cell whose worst
    # system is a molecule at 4 mHa but whose worst ATOM is inside the margin
    # clears.
    rows = [_row("a", 0.0, 9.0, atom_mha=9.0),
            _row("a", 1.0, 4.0, atom_mha=0.4, ae_kcal=0.2, n_ae=3)]
    out = pw.recommend(rows)
    assert (out["cleared"], out["weight"]) == (True, 1.0)


def test_the_atomization_gate_refuses_a_weight_the_atom_gate_admits():
    # Every atom is well inside tol_atom while the atomization error is 0.9
    # kcal/mol, above the 0.5 kcal/mol margin: the errors do not cancel between
    # a molecule and its atoms, which is what the deployment sees.
    rows = [_row("a", 0.0, 9.0, atom_mha=9.0, ae_kcal=9.0, n_ae=3),
            _row("a", 1.0, 1.0, atom_mha=0.1, ae_kcal=0.9, n_ae=3),
            _row("a", 10.0, 1.0, atom_mha=0.2, ae_kcal=0.4, n_ae=3)]
    out = pw.recommend(rows)
    assert (out["cleared"], out["weight"]) == (True, 10.0)
    entry = next(e for e in out["per_weight"] if e["weight"] == 1.0)
    assert entry["gate_ok"] is False
    assert entry["worst_dAE_kcal"] == 0.9


def test_the_atomization_gate_is_vacuous_when_the_set_has_no_molecule():
    # The smoke identity is two free atoms: no atomization energy exists, so
    # the gate claims nothing rather than failing.
    rows = [_row("a", 0.0, 9.0), _row("a", 1.0, 0.4)]
    out = pw.recommend(rows)
    assert (out["cleared"], out["weight"]) == (True, 1.0)
    entry = next(e for e in out["per_weight"] if e["weight"] == 1.0)
    assert entry["worst_dAE_kcal"] is None


def test_a_gate_quantity_that_was_not_measured_is_not_a_pass():
    # A row that carries atoms but no atom maximum cannot certify them.
    rows = [_row("a", 0.0, 9.0), _row("a", 1.0, 0.1, atom_mha=None,
                                     n_atoms=14),
            _row("a", 10.0, 0.1)]
    out = pw.recommend(rows)
    assert (out["cleared"], out["weight"]) == (True, 10.0)
    entry = next(e for e in out["per_weight"] if e["weight"] == 1.0)
    assert entry["gate_ok"] is False
    assert entry["unmeasured_gates"] == ["a:max_atom_dE_xc_mHa"]


def test_the_fallback_compares_the_two_gates_in_units_of_their_margins():
    # Weight 1 is 4x its atom margin; weight 10 is 1.8x its atomization
    # margin. Neither clears, and the one that is closest to ITS OWN margin is
    # the one reported.
    rows = [_row("a", 0.0, 9.0, atom_mha=9.0, ae_kcal=9.0, n_ae=3),
            _row("a", 1.0, 2.0, atom_mha=2.0, ae_kcal=0.1, n_ae=3),
            _row("a", 10.0, 1.0, atom_mha=0.1, ae_kcal=0.9, n_ae=3)]
    out = pw.recommend(rows)
    assert out["cleared"] is False
    assert out["weight"] == 10.0
    entry = next(e for e in out["per_weight"] if e["weight"] == 10.0)
    assert entry["worst_gate_ratio"] == pytest.approx(1.8)
    assert out["margin_AE_kcal"] == 0.5
    assert out["tol_AE_kcal"] == 1.0


def test_a_weight_that_destroys_the_pointwise_fit_is_refused():
    # Weight 10 clears the gate but its exchange loss is 10x the weight-0
    # value; weight 100 clears it inside the cap.
    rows = [_row("a", 0.0, 9.0, loss_x=1.0e-3),
            _row("a", 10.0, 0.1, loss_x=1.0e-2),
            _row("a", 100.0, 0.2, loss_x=2.0e-3)]
    out = pw.recommend(rows)
    assert (out["cleared"], out["weight"]) == (True, 100.0)


def test_correlation_loss_counts_toward_the_cap_too():
    rows = [_row("a", 0.0, 9.0, loss_c=1.0e-3),
            _row("a", 1.0, 0.1, loss_c=9.0e-3)]
    out = pw.recommend(rows)
    assert out["cleared"] is False


def test_nothing_clears_reports_the_tradeoff_and_the_best_weight():
    rows = [_row("a", 0.0, 9.0), _row("a", 1.0, 4.0), _row("a", 10.0, 2.0),
            _row("b", 0.0, 9.0), _row("b", 1.0, 5.0), _row("b", 10.0, 3.0)]
    out = pw.recommend(rows)
    assert out["cleared"] is False
    assert out["weight"] == 10.0                    # minimizes the worst max
    assert "NO swept weight clears" in out["reason"]
    assert out["margin_mHa"] == 0.5


def test_a_tie_on_the_worst_error_goes_to_the_smaller_weight():
    rows = [_row("a", 0.0, 9.0), _row("a", 1.0, 3.0), _row("a", 10.0, 3.0)]
    out = pw.recommend(rows)
    assert (out["cleared"], out["weight"]) == (False, 1.0)


def test_a_weight_measured_on_only_part_of_the_set_is_not_eligible():
    rows = [_row("a", 0.0, 9.0), _row("b", 0.0, 9.0),
            _row("a", 1.0, 0.1),                     # 'b' at w=1 is missing
            _row("a", 10.0, 0.1), _row("b", 10.0, 0.1)]
    out = pw.recommend(rows)
    assert (out["cleared"], out["weight"]) == (True, 10.0)
    entry = next(e for e in out["per_weight"] if e["weight"] == 1.0)
    assert entry["missing_archs"] == ["b"]
    assert entry["gate_ok"] is False


def test_a_diverged_cell_is_a_failure_not_a_gap():
    rows = [_row("a", 0.0, 9.0), _row("a", 1.0, float("nan")),
            _row("a", 10.0, 0.1)]
    out = pw.recommend(rows)
    assert (out["cleared"], out["weight"]) == (True, 10.0)
    entry = next(e for e in out["per_weight"] if e["weight"] == 1.0)
    assert entry["gate_ok"] is False


def test_without_a_weight_zero_baseline_nothing_can_be_certified():
    # The cap is a ratio against weight 0; with no such cell the rise cannot
    # be measured and the sweep must say so rather than certify on the gate
    # alone.
    out = pw.recommend([_row("a", 1.0, 0.1), _row("a", 10.0, 0.1)])
    assert out["cleared"] is False
    assert out["per_weight"][0]["archs_without_baseline"] == ["a"]


def test_an_empty_table_chooses_nothing():
    out = pw.recommend([])
    assert out["cleared"] is False
    assert out["weight"] is None
    assert "nothing to choose between" in out["reason"]


def test_the_verdict_names_the_architectures_it_measured():
    """A batch is not the set. ``EWSWEEP_ARCHS`` plus ``--resume`` is the
    documented way to split the sweep across walls, and the architectures are
    deliberately not part of the resume identity, so a first batch writes a
    complete-looking table for its own architectures alone. Its verdict has to
    say which those were, or the job mails a clearing result for a set it
    never measured."""
    rows = [_row("deep_3x16", 0.0, 9.0), _row("deep_3x16", 1.0, 0.2)]
    out = pw.recommend(rows)
    assert out["cleared"] is True
    assert out["archs_measured"] == ["deep_3x16"]
    assert out["covers_default_archs"] is False
    assert set(out["archs_unmeasured_default"]) == (
        set(pw.DEFAULT_ARCHS) - {"deep_3x16"})
    assert "deep_3x16" in out["reason"]
    assert "SUBSET" in out["reason"]
    for absent in out["archs_unmeasured_default"]:
        assert absent in out["reason"], absent


def test_a_verdict_over_the_whole_default_set_claims_the_whole_set():
    rows = []
    for name in pw.DEFAULT_ARCHS:
        rows += [_row(name, 0.0, 9.0), _row(name, 1.0, 0.2)]
    out = pw.recommend(rows)
    assert (out["cleared"], out["weight"]) == (True, 1.0)
    assert out["covers_default_archs"] is True
    assert out["archs_unmeasured_default"] == []
    assert "SUBSET" not in out["reason"]
    for name in pw.DEFAULT_ARCHS:
        assert name in out["reason"], name


def test_the_fallback_verdict_names_its_coverage_too():
    # Nothing clears: the reported weight is a finding, and a finding read off
    # part of the set is a finding about part of the set.
    out = pw.recommend([_row("deep_3x16", 0.0, 9.0),
                        _row("deep_3x16", 1.0, 4.0)])
    assert out["cleared"] is False
    assert "deep_3x16" in out["reason"] and "SUBSET" in out["reason"]


def test_a_table_outside_the_default_set_is_not_called_a_subset_of_it():
    """``--archs`` is free, so a table need not be a subset of the defaults.

    ``deep_attn_3x16`` is in the registry and is the one architecture Section
    2 measured that the default six leave out, so a sweep of it alone is
    DISJOINT from the default set. The coverage warning still stands -- none
    of the six was measured -- but calling that table "a SUBSET of the sweep's
    default set" states a set relation that is false, in the same sentence
    that a reader consults to decide whether a weight may be adopted.
    """
    assert "deep_attn_3x16" not in pw.DEFAULT_ARCHS
    out = pw.recommend([_row("deep_attn_3x16", 0.0, 9.0),
                        _row("deep_attn_3x16", 1.0, 0.2)])
    assert out["cleared"] is True
    assert out["covers_default_archs"] is False
    assert sorted(out["archs_unmeasured_default"]) == sorted(pw.DEFAULT_ARCHS)
    reason = out["reason"]
    assert "SUBSET" not in reason, reason
    assert "does NOT COVER the sweep's default set" in reason, reason
    # The architecture that is outside the default set is named as such,
    # rather than left to be inferred from a list of what is missing.
    assert "deep_attn_3x16 not in that set" in reason, reason
    for name in pw.DEFAULT_ARCHS:
        assert name in reason, name
    # A table that IS a subset keeps the stronger, true word.
    subset = pw.recommend([_row("deep_3x16", 0.0, 9.0),
                           _row("deep_3x16", 1.0, 0.2)])
    assert "SUBSET" in subset["reason"]
    assert "does NOT COVER" not in subset["reason"]
    # A mixed table -- one default plus one outsider -- is not a subset either.
    mixed = pw.recommend([_row("deep_3x16", 0.0, 9.0),
                          _row("deep_3x16", 1.0, 0.2),
                          _row("deep_attn_3x16", 0.0, 9.0),
                          _row("deep_attn_3x16", 1.0, 0.2)])
    assert "SUBSET" not in mixed["reason"], mixed["reason"]
    assert "does NOT COVER" in mixed["reason"], mixed["reason"]
    # Only the outsider is named as outside; the default that WAS measured is
    # not swept into that clause.
    assert "and deep_attn_3x16 not in that set" in mixed["reason"]
    assert "deep_3x16 not in that set" not in mixed["reason"].replace(
        "deep_attn_3x16 not in that set", "")
    assert "deep_3x16" not in mixed["archs_unmeasured_default"]


def test_the_tolerance_and_margin_are_configurable():
    rows = [_row("a", 0.0, 9.0), _row("a", 1.0, 1.5)]
    assert pw.recommend(rows)["cleared"] is False
    out = pw.recommend(rows, tol_atom_mha=2.0, margin_fraction=1.0)
    assert (out["cleared"], out["weight"]) == (True, 1.0)


# --------------------------------------------------------------------------- #
# The measured quantities, on synthetic per-system errors
# --------------------------------------------------------------------------- #

#: Four systems: two free atoms, a molecule built from them, and a molecule
#: whose element the set does not carry as a free atom (the Section 7 set is
#: exactly this shape -- Na2 with no sodium atom).
_SYSTEMS = [["H", "H 0 0 0", 0, 1],
            ["O", "O 0 0 0", 0, 2],
            ["H2O", "O 0 0 0; H 0 0 1; H 0 1 0", 0, 0],
            ["Na2", "Na 0 0 0; Na 0 0 2", 0, 0]]
_NAMES = ["H", "O", "H2O", "Na2"]
#: Chosen so the two channels disagree about which system is worst and about
#: how large the error is: |dE_x| peaks at 5 mHa on H2O while |dE_x + dE_c|
#: peaks at 4 mHa there, and the worst ATOM is 1 mHa.
_DX = [1.0e-3, -2.0e-3, 5.0e-3, 0.0]
_DC = [-0.5e-3, 3.0e-3, -1.0e-3, 0.0]


def test_geometry_elements_reads_the_composition_from_the_geometry():
    assert pw._geometry_elements("O 0 0 0; H 0 0 1; H 0 1 0") == \
        ["O", "H", "H"]
    assert pw._geometry_elements("CL 0 0 0\nc 0 0 1") == ["Cl", "C"]
    assert pw._geometry_elements("H1 0 0 0; H2 0 0 1") == ["H", "H"]


def test_classify_systems_splits_atoms_ions_and_molecules():
    systems = _SYSTEMS + [["F-", "F 0 0 0", -1, 0]]
    atoms, neutral, molecules = pw.classify_systems(systems)
    # Every single-center system is held to the atom tolerance, the pool's
    # anions included.
    assert atoms == [0, 1, 4]
    # Only the NEUTRAL ones are atomization references.
    assert neutral == {"H": 0, "O": 1}
    assert [(name, counts) for _i, name, counts in molecules] == [
        ("H2O", {"O": 1, "H": 2}), ("Na2", {"Na": 2})]


def test_atomization_error_is_the_molecule_minus_its_atoms():
    delta_xc = [x + c for x, c in zip(_DX, _DC)]
    errors, skipped = pw.atomization_errors(delta_xc, _SYSTEMS)
    # dAE(H2O) = dE(H2O) - [2 dE(H) + dE(O)]
    #          = 4.0e-3 - [2 (0.5e-3) + 1.0e-3] = 2.0e-3 Ha
    assert dict(errors)["H2O"] == pytest.approx(2.0e-3 * 627.5094740631,
                                                rel=1e-12)
    # Na2 has no sodium atom in the set: reported as skipped, never as zero.
    assert skipped == ["Na2"]
    assert [name for name, _v in errors] == ["H2O"]


def test_the_gate_quantity_is_the_sum_of_the_two_channels():
    """max |dE_xc| is max |dE_x + dE_c|, not max |dE_x|.

    Exchange and correlation are two halves of one functional and the
    certificate bounds their sum; reading the exchange channel alone reports a
    different number (5 mHa against 4 here) on the same fit.
    """
    out = pw.summarize_energy_errors(_DX, _DC, _SYSTEMS, _NAMES)
    assert out["max_dE_xc_mHa"] == pytest.approx(4.0, rel=1e-12)
    assert out["worst_system"] == "H2O"
    assert out["max_dE_x_mHa"] == pytest.approx(5.0, rel=1e-12)
    assert out["max_dE_c_mHa"] == pytest.approx(3.0, rel=1e-12)
    # The atom gate reads the same sum, over the atoms alone.
    assert out["max_atom_dE_xc_mHa"] == pytest.approx(1.0, rel=1e-12)
    assert out["worst_atom"] == "O"
    assert out["n_atom_systems"] == 2
    # And the atomization gate reads it through the atomization energy.
    assert out["max_dAE_kcal"] == pytest.approx(2.0e-3 * 627.5094740631,
                                                rel=1e-12)
    assert out["worst_ae_system"] == "H2O"
    assert (out["n_ae_molecules"], out["ae_skipped"]) == (1, ["Na2"])


def test_the_row_carries_the_per_system_errors_it_was_reduced_from():
    out = pw.summarize_energy_errors(_DX, _DC, _SYSTEMS, _NAMES)
    per = out["per_system"]
    assert per["names"] == _NAMES
    assert per["delta_x_mHa"] == pytest.approx([1000.0 * v for v in _DX])
    assert per["delta_c_mHa"] == pytest.approx([1000.0 * v for v in _DC])
    # The published maxima are the ones the stored residuals give.
    worst = max(abs(x + c) for x, c in zip(per["delta_x_mHa"],
                                           per["delta_c_mHa"]))
    assert out["max_dE_xc_mHa"] == pytest.approx(worst, rel=1e-12)


def test_a_file_without_a_manifest_measures_no_gate_quantity():
    # No system list means no atom / molecule split; the row says so rather
    # than reporting the all-system maximum as if it were the atom one.
    out = pw.summarize_energy_errors(_DX, _DC, None, _NAMES)
    assert out["max_dE_xc_mHa"] == pytest.approx(4.0, rel=1e-12)
    assert out["max_atom_dE_xc_mHa"] is None
    # UNKNOWN, not zero: a count of zero would read as "the set holds no atom"
    # and let the gate pass vacuously on a row that measured nothing.
    assert (out["n_atom_systems"], out["n_ae_molecules"]) == (None, None)
    assert pw.recommend([_row("a", 0.0, 9.0),
                         dict(_row("a", 1.0, 0.1), **out)])["cleared"] is False


_RECORDED = {"energy_term_x_final": 1.0e-4, "energy_term_c_final": 4.0e-5,
             "energy_term_max_abs_dE_mHa": 12.0,
             "energy_term_rms_dE_mHa": 8.0}
_RECONSTRUCTED = {"energy_term_x_recon": 1.0e-4, "energy_term_c_recon": 4.0e-5,
                  "max_dE_xc_mHa": 12.0, "rms_dE_xc_mHa": 8.0}


def test_check_reconstruction_refuses_a_drifted_cell():
    """The --recon-rtol guard is what stands between a silently divergent
    reconstruction and a table that looks valid."""
    assert pw.check_reconstruction(_RECONSTRUCTED, _RECORDED, 1e-6, "cell",
                                   weight=1.0) == pytest.approx(0.0, abs=1e-15)
    drifted = dict(_RECONSTRUCTED,
                   energy_term_c_recon=4.0e-5 * (1 + 1e-9))
    assert pw.check_reconstruction(drifted, _RECORDED, 1e-6, "cell",
                                   weight=1.0) == pytest.approx(1e-9, rel=1e-3)
    with pytest.raises(RuntimeError) as excinfo:
        pw.check_reconstruction(dict(_RECONSTRUCTED,
                                     energy_term_c_recon=5.0e-5),
                                _RECORDED, 1.0e-6, "cell", weight=1.0)
    message = str(excinfo.value)
    assert "--recon-rtol=1e-06" in message
    assert "correlation energy term" in message
    assert message.startswith("cell: ")
    for key in ("energy_term_x_recon", "max_dE_xc_mHa", "rms_dE_xc_mHa"):
        with pytest.raises(RuntimeError):
            pw.check_reconstruction(dict(_RECONSTRUCTED,
                                         **{key: 0.5 * _RECONSTRUCTED[key]}),
                                    _RECORDED, 1.0e-6, "cell", weight=1.0)


def test_the_gate_quantity_is_pinned_against_the_runs_own_measurement():
    """run_pretrain records 1000 max|dE_x + dE_c| for the network it saved; the
    reconstruction here reads the checkpoint back and must land on it."""
    with pytest.raises(RuntimeError) as excinfo:
        # What a substitution of max|dE_x| for max|dE_x + dE_c| looks like:
        # the same fit, a maximum about twice as large.
        pw.check_reconstruction(dict(_RECONSTRUCTED, max_dE_xc_mHa=25.0),
                                _RECORDED, 1.0e-6, "cell", weight=1.0)
    assert "maximum |dE_xc| in mHa" in str(excinfo.value)
    # The check runs at weight 0 too: the record is measured on the saved
    # network whatever the weight was.
    with pytest.raises(RuntimeError):
        pw.check_reconstruction(dict(_RECONSTRUCTED, max_dE_xc_mHa=25.0),
                                _RECORDED, 1.0e-6, "cell", weight=0.0)
    # A record predating that convention writes 0.0 at weight 0, which is not
    # the same quantity; there is nothing to compare and nothing is claimed.
    legacy = {"energy_term_x_final": 0.0, "energy_term_c_final": 0.0}
    assert pw.check_reconstruction(_RECONSTRUCTED, legacy, 1e-6, "cell",
                                   weight=0.0) is None
    # A quantity the record does not carry is skipped, not invented.
    assert pw.check_reconstruction(
        _RECONSTRUCTED, {"energy_term_max_abs_dE_mHa": 12.0}, 1e-6, "cell",
        weight=1.0) == pytest.approx(0.0, abs=1e-15)


# --------------------------------------------------------------------------- #
# The set the weight is measured on
# --------------------------------------------------------------------------- #

def _v6_config_path():
    """The campaign configuration the measured weight is written into."""
    path = _HERE / "configs" / "dfs_step7.dfs6311_grid3_v6.yaml"
    return path if path.is_file() else None


def test_the_campaign_atom_list_is_the_one_the_campaign_states():
    """The probe's explicit atom list IS campaign v6's, read from the file.

    ``pretrain.atoms`` is nearly redundant under ``dfs_set`` + ``pool_atoms``
    and contributes exactly one system neither inventory supplies -- free Na,
    on which Na2's atomization energy rests. A constant here that drifts from
    the YAML would measure the weight on a set the campaign does not train
    on, silently, so the two are compared rather than both transcribed.
    """
    pytest.importorskip("yaml")
    path = _v6_config_path()
    if path is None:
        pytest.skip("no hpcjobs/configs in this checkout")
    from xcquinox.alec.cluster.grid_config import load_grid_config
    cfg = load_grid_config(str(path))
    assert tuple(tuple(a) for a in cfg.pretrain.atoms) == pw.CAMPAIGN_ATOMS
    assert cfg.pretrain.dfs_set is True and cfg.pretrain.pool_atoms is True
    assert ("Na", 1) in pw.CAMPAIGN_ATOMS
    assert not any(sym == "He" for sym, _spin in pw.CAMPAIGN_ATOMS)


def test_the_sweep_measures_the_campaigns_own_pretraining_set():
    """Same builder, same arguments, same systems: 38 on the PBE parent and
    36 on SCAN.

    The weight balances a squared system energy against an integration-
    weighted point-wise residual, so WHICH systems are in the set changes the
    number it lands on. The sweep is run at a reduced identity deliberately
    (see the module header), but the SET is not part of that reduction: an
    absent free Na would leave the one row Na2's atomization energy rests on
    unmeasured on exactly the gate the certificate reads.
    """
    from xcquinox.alec.pretrain_data_gen import resolve_pretrain_systems
    for reference_xc, expected in (("pbe", 38), ("scan", 36)):
        systems = resolve_pretrain_systems(
            atoms=pw.CAMPAIGN_ATOMS, dfs_set=True, pool_atoms=True,
            reference_xc=reference_xc)
        assert len(systems) == expected, reference_xc
        assert "Na" in {s.name for s in systems}, reference_xc
        assert "He" not in {s.name for s in systems}, reference_xc


def test_the_generator_is_asked_for_the_campaign_set(tmp_path, monkeypatch):
    """The arguments the probe hands the generator ARE the campaign's three
    set switches, so the file it measures on holds the campaign's rows."""
    from xcquinox.alec import pretrain_data_gen

    seen = []
    monkeypatch.setattr(pretrain_data_gen, "ensure_pretrain_data",
                        lambda directory, **kw: (seen.append(kw),
                                                 "/none.npz")[1])
    pw.ensure_data(str(tmp_path), polarized=True, reference_xc="pbe",
                   basis="def2-svp", grid_level=3, lock_strength=3.0e-5)
    assert seen[-1]["dfs_set"] is True
    assert seen[-1]["pool_atoms"] is True
    assert tuple(tuple(a) for a in seen[-1]["atoms"]) == pw.CAMPAIGN_ATOMS
    # The smoke leg still substitutes its own two systems and neither
    # inventory, so it stays a seconds-long plumbing check.
    pw.ensure_data(str(tmp_path), polarized=True, reference_xc="pbe",
                   basis="sto-3g", grid_level=1, lock_strength=3.0e-5,
                   smoke_atoms=pw.SMOKE_ATOMS)
    assert seen[-1]["dfs_set"] is False and seen[-1]["pool_atoms"] is False
    assert tuple(tuple(a) for a in seen[-1]["atoms"]) == pw.SMOKE_ATOMS


def test_the_atom_list_is_part_of_the_resume_identity(tmp_path):
    """A table measured on another set is REFUSED rather than merged.

    ``atoms`` is a resume-identity key, so a table written before the set was
    widened does not silently contribute rows measured without free Na.
    """
    identity = pw.build_identity(_args(), 3.0e-5, None)
    assert identity["atoms"] == [list(a) for a in pw.CAMPAIGN_ATOMS]
    assert "atoms" in pw._RESUME_IDENTITY_KEYS
    stale = dict(identity, atoms=None)          # the pre-widening identity
    path = _stored(tmp_path, stale, [_row("deep_3x16", 0.0, 1.0)])
    with pytest.raises(SystemExit) as excinfo:
        pw.load_resumable_rows(str(path), identity)
    assert "atoms" in str(excinfo.value)


def test_the_header_states_the_reduced_identity_as_a_decision():
    """The sweep runs at def2-svp / 1000 steps / no validation hold-out while
    the campaign runs at 6-311++G(3df,2pd) / 2500 steps / a 20 percent
    hold-out. That gap is deliberate and cheap, but it has to be WRITTEN
    somewhere the reader of the recommendation will see it, or a certificate
    FAIL on an unmeasured architecture is read as a code fault instead of a
    transfer question."""
    text = _SCRIPT.read_text()
    head = text.split('"""')[1]
    assert "def2-svp" in head
    assert "reduced" in head.lower()
    assert "6-311++G(3df,2pd)" in head
    assert "certificate" in head


def test_the_generator_is_never_asked_to_waive_the_degenerate_refusal(
        tmp_path, monkeypatch):
    """Neither identity needs the waiver, so neither asks for it.

    At the production identity grid level 3 is a floor precisely so the
    refusal never fires, and the smoke's He and Li are not spatially
    degenerate at any level. The manifest reads false either way -- it records
    whether the permission was EXERCISED, not whether it was offered -- so
    what is pinned here is the argument the probe passes.
    """
    from xcquinox.alec import pretrain_data_gen

    seen = []
    monkeypatch.setattr(pretrain_data_gen, "ensure_pretrain_data",
                        lambda directory, **kw: (seen.append(kw),
                                                 "/none.npz")[1])
    for smoke_atoms in (None, pw.SMOKE_ATOMS):
        pw.ensure_data(str(tmp_path), polarized=True, reference_xc="pbe",
                       basis="sto-3g", grid_level=1, lock_strength=3.0e-5,
                       smoke_atoms=smoke_atoms)
    assert [kw["allow_irreproducible_degenerate"] for kw in seen] == \
        [False, False]
    # Executed rather than asserted from the comment: the smoke pair carries
    # no spatially degenerate system, at its own coarse grid.
    systems = pretrain_data_gen.resolve_pretrain_systems(
        atoms=tuple(pw.SMOKE_ATOMS))
    assert pretrain_data_gen._degenerate_systems(systems, "sto-3g", 1) == ()


# --------------------------------------------------------------------------- #
# The two readings of "meta-GGA rung"
# --------------------------------------------------------------------------- #

class _FakeDescriptor:
    def __init__(self, name):
        self.name = name


class _FakeArch:
    def __init__(self, meta_gga, descriptor_names):
        self.meta_gga = meta_gga
        self.descriptors = tuple(_FakeDescriptor(n) for n in descriptor_names)


def test_an_architecture_whose_rung_readings_disagree_is_refused():
    """The backstop fires on an architecture-LIKE object.

    ``ArchitectureConfig`` refuses a flag / descriptor disagreement at
    construction, so this state is unreachable through the registry; the
    predicate the readers share is duck-typed, though, so an object assembled
    outside that class carries no such guarantee and reaches the same code
    paths. ``_FakeArch`` is exactly such an object, which is what makes this
    check reachable at all."""
    pw._check_rung_consistency("gga", _FakeArch(False, ()))
    pw._check_rung_consistency("mgga", _FakeArch(True, ("metagga",)))
    for arch in (_FakeArch(False, ("metagga",)), _FakeArch(True, ("cusp",))):
        with pytest.raises(SystemExit) as excinfo:
            pw._check_rung_consistency("mixed", arch)
        assert "resolve_parent_density" in str(excinfo.value)
        assert "run_pretrain" in str(excinfo.value)


def test_the_library_refuses_the_disagreement_before_the_probe_can_see_it():
    """The reason the check above is a backstop rather than the guard.

    A registry architecture cannot be in the refused state: the class refuses
    the pairing in ``__post_init__``, in both directions, so the probe's own
    check is unreachable for anything built by name or by ``from_spec``.
    """
    from xcquinox.alec.config import ArchitectureConfig, FeatureSpec
    with pytest.raises(ValueError, match="disagrees with its descriptor list"):
        ArchitectureConfig(name="probe_rung_mismatch", depth=3, nodes=16,
                           descriptors=(FeatureSpec(name="metagga"),))
    with pytest.raises(ValueError, match="disagrees with its descriptor list"):
        ArchitectureConfig(name="probe_rung_mismatch", depth=3, nodes=16,
                           meta_gga=True)


def test_every_registered_architecture_passes_the_rung_check():
    from xcquinox.alec.config import get_architecture, list_architectures
    for name in list_architectures():
        pw._check_rung_consistency(name, get_architecture(name))


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

def test_write_table_creates_its_directory_and_round_trips(tmp_path):
    target = tmp_path / "deep" / "nested" / "table.json"
    payload = {"rows": [_row("a", 1.0, 0.25)], "recommendation": {"x": 1}}
    assert pw.write_table(str(target), payload) == str(target)
    assert json.loads(target.read_text()) == payload


def test_format_table_renders_every_row_and_the_verdict():
    rows = [_row("a", 0.0, 9.0), _row("a", 1.0, 0.25)]
    text = pw.format_table(rows, pw.recommend(rows))
    lines = text.splitlines()
    assert lines[0].split()[:3] == ["arch", "parent", "w_E"]
    # The table body: header, rule, then one line per measured cell.
    body = lines[2:lines.index("")]
    assert len(body) == 2
    assert len([line for line in body if " 9.0000" in line]) == 1
    assert len([line for line in body if " 0.2500" in line]) == 1
    assert "rule:" in text
    assert "recommendation: energy_term_weight = 1  [CLEARS]" in text


def test_format_table_names_the_architectures_the_verdict_speaks_for():
    rows = [_row("deep_3x16", 0.0, 9.0), _row("deep_3x16", 1.0, 0.2)]
    text = pw.format_table(rows, pw.recommend(rows))
    assert "architectures measured: deep_3x16" in text
    assert "sweep defaults NOT in this table:" in text
    for absent in set(pw.DEFAULT_ARCHS) - {"deep_3x16"}:
        assert absent in text, absent


def test_format_table_renders_a_verdict_written_before_the_coverage_keys():
    # A table on disk from an earlier run carries no coverage keys; it still
    # renders, rather than raising in the middle of a completed sweep's log.
    verdict = pw.recommend([_row("a", 0.0, 9.0), _row("a", 1.0, 0.2)])
    verdict.pop("archs_measured")
    text = pw.format_table([_row("a", 1.0, 0.2)], verdict)
    assert "architectures measured" not in text
    assert "rule:" in text


def test_format_table_renders_an_absent_cell_as_a_dash():
    row = _row("a", 1.0, 0.25)
    row["reference_xc"] = None
    row["wall_seconds"] = None
    text = pw.format_table([row])
    assert " - " in text or text.rstrip().endswith("-")


# --------------------------------------------------------------------------- #
# Exit-code contract (the sweep loop with the measurement stubbed out)
# --------------------------------------------------------------------------- #

def _stub_sweep(monkeypatch, maxima):
    """Run main() with the data generation and the pretraining replaced by a
    table the caller dictates. ``maxima`` maps weight -> max |dE_xc| in mHa."""
    monkeypatch.setattr(pw, "ensure_data",
                        lambda *a, **k: "/nonexistent/pretrain_data.npz")

    def _cell(arch, arch_name, data_path, work_dir, *, weight, **kwargs):
        return _row(arch_name, weight, maxima[weight])

    monkeypatch.setattr(pw, "run_cell", _cell)


def test_exit_zero_when_a_weight_clears(tmp_path, monkeypatch):
    _stub_sweep(monkeypatch, {0.0: 9.0, 1.0: 0.2})
    rc = pw.main(["--data-dir", str(tmp_path), "--out",
                  str(tmp_path / "t.json"), "--archs", "deep_3x16",
                  "--weights", "0,1"])
    assert rc == 0
    assert json.loads((tmp_path / "t.json").read_text())[
        "recommendation"]["weight"] == 1.0


def test_exit_nonzero_when_nothing_clears_but_the_table_is_still_written(
        tmp_path, monkeypatch):
    _stub_sweep(monkeypatch, {0.0: 9.0, 1.0: 4.0})
    rc = pw.main(["--data-dir", str(tmp_path), "--out",
                  str(tmp_path / "t.json"), "--archs", "deep_3x16",
                  "--weights", "0,1"])
    assert rc == 2
    payload = json.loads((tmp_path / "t.json").read_text())
    assert payload["recommendation"]["cleared"] is False
    assert len(payload["rows"]) == 2


def test_an_unknown_architecture_is_refused_by_name(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        pw.main(["--data-dir", str(tmp_path), "--out",
                 str(tmp_path / "t.json"), "--archs", "deep_3x16,not_an_arch"])
    assert "not_an_arch" in str(excinfo.value)
    assert not (tmp_path / "t.json").exists()


def test_a_failed_cell_is_recorded_and_exits_one(tmp_path, monkeypatch):
    monkeypatch.setattr(pw, "ensure_data", lambda *a, **k: "/nonexistent.npz")

    def _boom(*a, **k):
        raise RuntimeError("segment table disagrees")

    monkeypatch.setattr(pw, "run_cell", _boom)
    rc = pw.main(["--data-dir", str(tmp_path), "--out",
                  str(tmp_path / "t.json"), "--archs", "deep_3x16",
                  "--weights", "1"])
    assert rc == 1
    payload = json.loads((tmp_path / "t.json").read_text())
    assert payload["failures"][0]["arch"] == "deep_3x16"
    assert "segment table disagrees" in payload["failures"][0]["error"]


# --------------------------------------------------------------------------- #
# Surviving a wall-clock kill: the incremental table and --resume
# --------------------------------------------------------------------------- #

def test_the_table_is_written_after_every_cell(tmp_path, monkeypatch):
    """A twelve-hour reservation killed at the wall must keep what it paid for.

    The rows are written as they are measured, so the table on disk after cell
    N holds N rows and says it is not complete.
    """
    out = tmp_path / "t.json"
    monkeypatch.setattr(pw, "ensure_data", lambda *a, **k: "/none.npz")
    measured = []

    def _cell(arch, name, *a, **k):
        if len(measured) == 1:
            raise KeyboardInterrupt("wall clock")   # not caught as a failure
        measured.append(k["weight"])
        return _row(name, k["weight"], 0.2)

    monkeypatch.setattr(pw, "run_cell", _cell)
    with pytest.raises(KeyboardInterrupt):
        pw.main(["--data-dir", str(tmp_path), "--out", str(out),
                 "--archs", "deep_3x16", "--weights", "0,1,10"])
    payload = json.loads(out.read_text())
    assert payload["complete"] is False
    assert [r["weight"] for r in payload["rows"]] == [0.0]


def test_resume_measures_only_the_cells_the_table_is_missing(
        tmp_path, monkeypatch):
    out = tmp_path / "t.json"
    monkeypatch.setattr(pw, "ensure_data", lambda *a, **k: "/none.npz")
    first = []

    def _cell_a(arch, name, *a, **k):
        if len(first) == 1:
            raise KeyboardInterrupt("wall clock")
        first.append(k["weight"])
        return _row(name, k["weight"], 9.0)

    monkeypatch.setattr(pw, "run_cell", _cell_a)
    argv = ["--data-dir", str(tmp_path), "--out", str(out),
            "--archs", "deep_3x16", "--weights", "0,1,10"]
    with pytest.raises(KeyboardInterrupt):
        pw.main(argv)

    second = []

    def _cell_b(arch, name, *a, **k):
        second.append(k["weight"])
        return _row(name, k["weight"], 0.2)

    monkeypatch.setattr(pw, "run_cell", _cell_b)
    rc = pw.main(argv + ["--resume"])
    assert second == [1.0, 10.0]                 # weight 0 was not re-measured
    payload = json.loads(out.read_text())
    assert payload["complete"] is True
    assert sorted(r["weight"] for r in payload["rows"]) == [0.0, 1.0, 10.0]
    # The carried-over row is the one the first pass measured, not a fresh one.
    assert next(r for r in payload["rows"]
                if r["weight"] == 0.0)["max_dE_xc_mHa"] == 9.0
    assert rc == 0


def test_resume_without_a_table_measures_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(pw, "ensure_data", lambda *a, **k: "/none.npz")
    seen = []
    monkeypatch.setattr(pw, "run_cell", lambda arch, name, *a, **k: (
        seen.append(k["weight"]), _row(name, k["weight"], 0.2))[1])
    pw.main(["--data-dir", str(tmp_path), "--out", str(tmp_path / "t.json"),
             "--archs", "deep_3x16", "--weights", "0,1", "--resume"])
    assert seen == [0.0, 1.0]


def _stored(tmp_path, identity, rows):
    path = tmp_path / "stored.json"
    path.write_text(json.dumps({"identity": identity, "rows": rows}))
    return path


def test_resume_refuses_a_table_measured_at_another_identity(tmp_path):
    identity = pw.build_identity(_args("--smoke"), 3.0e-5, pw.SMOKE_ATOMS)
    stored = dict(identity, basis="def2-svp", n_steps=1000)
    path = _stored(tmp_path, stored, [_row("deep_3x16", 0.0, 1.0)])
    with pytest.raises(SystemExit) as excinfo:
        pw.load_resumable_rows(str(path), identity)
    message = str(excinfo.value)
    assert "basis" in message and "n_steps" in message
    assert "def2-svp" in message and "sto-3g" in message


def test_resume_refuses_a_table_measured_at_the_other_polarization(tmp_path):
    identity = pw.build_identity(_args(), 3.0e-5, None)
    path = _stored(tmp_path, dict(identity, polarized=False), [])
    with pytest.raises(SystemExit) as excinfo:
        pw.load_resumable_rows(str(path), identity)
    assert "polarized" in str(excinfo.value)


def test_resume_carries_over_every_row_of_the_same_identity(tmp_path):
    identity = pw.build_identity(_args(), 3.0e-5, None)
    rows = [_row("deep_3x16", 0.0, 1.0), _row("deep_3x16", 1.0, 1.0),
            _row("deep_cusp_3x16", 0.0, 1.0),
            _row("deep_3x16", 0.0, 7.0)]        # a duplicate cell
    path = _stored(tmp_path, identity, rows)
    kept = pw.load_resumable_rows(str(path), identity)
    # Rows measured for ANOTHER architecture are carried, not dropped: that is
    # how a sweep batched over architectures accumulates into one table the
    # rule can read across all of them. A repeated cell keeps its first value.
    assert [(r["arch"], r["weight"]) for r in kept] == [
        ("deep_3x16", 0.0), ("deep_3x16", 1.0), ("deep_cusp_3x16", 0.0)]
    assert kept[0]["max_dE_xc_mHa"] == 1.0


def test_a_batched_sweep_accumulates_into_one_table(tmp_path, monkeypatch):
    """Two submissions, one architecture each, one table and one verdict."""
    out = tmp_path / "t.json"
    monkeypatch.setattr(pw, "ensure_data", lambda *a, **k: "/none.npz")
    monkeypatch.setattr(
        pw, "run_cell",
        lambda arch, name, *a, **k: _row(
            name, k["weight"], 9.0 if k["weight"] == 0.0 else 0.2))
    base = ["--data-dir", str(tmp_path), "--out", str(out), "--weights",
            "0,1", "--resume"]
    pw.main(base + ["--archs", "deep_3x16"])
    pw.main(base + ["--archs", "deep_cusp_3x16"])
    payload = json.loads(out.read_text())
    assert sorted({r["arch"] for r in payload["rows"]}) == [
        "deep_3x16", "deep_cusp_3x16"]
    # The rule reads the accumulated table: both architectures at weight 1.
    assert payload["recommendation"]["weight"] == 1.0
    entry = next(e for e in payload["recommendation"]["per_weight"]
                 if e["weight"] == 1.0)
    assert entry["missing_archs"] == []
    # The batch is recorded BESIDE the identity, not inside it -- inside, the
    # second submission would be refused as a different measurement.
    assert payload["archs_requested"] == ["deep_cusp_3x16"]
    assert "archs" not in payload["identity"]
    assert "archs_requested" not in payload["identity"]
    # And the verdict says what it covers: four of the six defaults are still
    # unmeasured, so the exit-0 line is not a statement about the whole set.
    recommendation = payload["recommendation"]
    assert recommendation["archs_measured"] == ["deep_3x16",
                                                "deep_cusp_3x16"]
    assert recommendation["covers_default_archs"] is False
    assert sorted(recommendation["archs_unmeasured_default"]) == [
        "deep_mgga_3x16", "deep_rung35_3x16", "deep_rung35_attn_3x16",
        "deep_rung35ms_3x16"]
    assert "SUBSET" in recommendation["reason"]


# --------------------------------------------------------------------------- #
# End to end at the smoke identity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("flag,polarized", [("--polarized", True),
                                            ("--no-polarized", False)])
def test_smoke_sweep_end_to_end(tmp_path, flag, polarized):
    """The real sweep at two systems / STO-3G / grid 1 / five steps.

    Run at BOTH polarizations, because the production objective is the
    polarized one and the two read different correlation rows and a differently
    shaped cnet input. Measured ~30 s wall per leg including both
    parent-density generations, so neither is slow-marked. Run in a subprocess:
    JAX can abort at interpreter exit on this backend, and a measurement probe
    must not be able to take the session with it.
    """
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MKL_NUM_THREADS="4", JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_cpu_multi_thread_eigen=false")
    out = tmp_path / "table.json"
    data_dir = tmp_path / "data"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--smoke", flag,
         "--data-dir", str(data_dir), "--out", str(out)],
        env=env, capture_output=True, text=True, timeout=900)
    # 0 = a weight cleared, 2 = none did. Five steps from a random
    # initialization will not clear; both are completions, and 1 is not.
    # A negative code is a signal death: JAX can corrupt the heap during
    # interpreter shutdown after the table is written, which would hand SLURM
    # a FAILED job for a completed sweep, so the script leaves through a hard
    # exit and this assertion is what says so.
    assert proc.returncode in (0, 2), (
        f"returncode={proc.returncode}\n"
        + proc.stdout[-4000:] + proc.stderr[-4000:])

    payload = json.loads(out.read_text())
    assert payload["failures"] == []
    assert payload["complete"] is True
    assert payload["identity"]["basis"] == "sto-3g"
    assert payload["identity"]["grid_level"] == 1
    assert payload["identity"]["exchange_footing"] == "spin_channel"
    assert payload["identity"]["polarized"] is polarized

    # Each parent density is written under the generator's OWN name, with no
    # alias: the worker resolves the parent before it names the file, so the
    # PBE name never stands for the SCAN-density rows.
    stem = "pretrain_data_polarized" if polarized else "pretrain_data"
    assert sorted(os.listdir(data_dir / "parent_scan")) == [
        f"{stem}_scan.npz", f"{stem}_scan.npz.manifest.json"]
    assert sorted(os.listdir(data_dir / "parent_pbe")) == [
        f"{stem}.npz", f"{stem}.npz.manifest.json"]
    for parent, name in (("pbe", f"{stem}.npz"),
                         ("scan", f"{stem}_scan.npz")):
        manifest = json.loads(
            (data_dir / f"parent_{parent}" / f"{name}.manifest.json"
             ).read_text())
        # Neither He nor Li is spatially degenerate, so the refusal the
        # generator applies to the production set never fires here and nothing
        # is waived.
        assert manifest["allow_irreproducible_degenerate"] is False
        assert manifest["reference_xc"] == parent

    rows = payload["rows"]
    assert len(rows) == 4
    # Both rungs ran, each against its own parent density.
    assert {r["reference_xc"] for r in rows} == {"pbe", "scan"}
    assert {r["exchange_footing"] for r in rows} == {"spin_channel"}
    assert {r["polarized"] for r in rows} == {polarized}
    for row in rows:
        assert row["n_systems"] == 2
        assert row["max_dE_xc_mHa"] > 0.0
        assert row["rms_dE_xc_mHa"] > 0.0
        # A max over systems can never exceed the sum of the two channels'
        # RMS times sqrt(N); what it must never do is come back as the mean.
        assert row["max_dE_x_mHa"] > 0.0 and row["max_dE_c_mHa"] > 0.0
        assert row["worst_system"] in ("He", "Li")
        # The gate quantity is the SUM of the two channels, per system,
        # recomputed here from the residuals the row carries.
        per = row["per_system"]
        assert per["names"] == ["He", "Li"]
        recomputed = max(abs(x + c) for x, c in zip(per["delta_x_mHa"],
                                                    per["delta_c_mHa"]))
        assert row["max_dE_xc_mHa"] == pytest.approx(recomputed, rel=1e-12)
        assert row["max_dE_x_mHa"] == pytest.approx(
            max(abs(x) for x in per["delta_x_mHa"]), rel=1e-12)
        # Two free atoms: the atom gate reads both of them and the
        # atomization gate has nothing to read.
        assert row["n_atom_systems"] == 2
        assert row["max_atom_dE_xc_mHa"] == pytest.approx(
            row["max_dE_xc_mHa"], rel=1e-12)
        assert row["worst_atom"] == row["worst_system"]
        assert (row["n_ae_molecules"], row["max_dAE_kcal"]) == (0, None)

    by_weight = {(r["arch"], r["weight"]): r for r in rows}
    for (arch, weight), row in by_weight.items():
        if weight == 0.0:
            # THE reason the table is reconstructed rather than read off the
            # objective: at weight 0 the loss short-circuits before the energy
            # term, so the FITTED value is 0 while the error is not. This is
            # the baseline row every ratio in the table is taken against.
            assert row["energy_term_x_recon"] > 0.0
            assert row["energy_term_c_recon"] > 0.0
        # At every weight the run records the same quantities, measured on the
        # network it saved, and the reconstruction lands on them.
        assert row["recon_max_rel_dev"] is not None
        assert row["recon_max_rel_dev"] <= 1.0e-6
        assert row["energy_term_x_recon"] == pytest.approx(
            row["energy_term_x_final"], rel=1e-12)
        assert row["max_dE_xc_mHa"] == pytest.approx(
            row["energy_term_max_abs_dE_mHa"], rel=1e-12)
    # The meta-GGA cell is the one that carries the synthetic mesh.
    assert by_weight[("deep_mgga_3x16", 0.0)]["pretrain_mesh"] is True
    assert by_weight[("deep_3x16", 0.0)]["pretrain_mesh"] is False


# --------------------------------------------------------------------------- #
# The job script
# --------------------------------------------------------------------------- #

def _sbatch_text() -> str:
    return _SBATCH.read_text()


def test_mail_directives_present():
    t = _sbatch_text()
    assert "#SBATCH --mail-user=alec.wills@stonybrook.edu" in t
    assert "#SBATCH --mail-type=BEGIN,END,FAIL" in t


def test_the_header_names_the_architectures_the_default_run_measures():
    """The WHAT block is the only statement of the swept set a reader of the
    job's mail ever sees, so it has to hold every name ``DEFAULT_ARCHS`` does
    -- an architecture added to the default set and not to the header would be
    measured silently, and one removed would be claimed without being run."""
    t = _sbatch_text()
    for name in pw.DEFAULT_ARCHS:
        assert name in t, name


def test_house_shell_idiom():
    t = _sbatch_text()
    assert "set -uo pipefail" in t
    for line in t.splitlines():
        assert not line.strip().startswith("set -e"), line
        assert "errexit" not in line, line


def test_single_node_one_task_with_a_thread_cap_from_slurm():
    t = _sbatch_text()
    assert "#SBATCH --nodes=1" in t
    assert "#SBATCH --ntasks=1" in t
    assert "#SBATCH --cpus-per-task=40" in t
    # The PySCF-serving pools are capped at parallel.PYSCF_POOL_THREADS_MAX
    # from the allocation, as the stage templates cap theirs; the shell form
    # is evaluated below against the module rule.
    from xcquinox.alec.parallel import PYSCF_POOL_THREADS_MAX, pyscf_pool_threads
    assert f'THREADS="${{SLURM_CPUS_PER_TASK:-{PYSCF_POOL_THREADS_MAX}}}"' in t
    assert f'[ "$THREADS" -le {PYSCF_POOL_THREADS_MAX} ] || THREADS={PYSCF_POOL_THREADS_MAX}' in t
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert f'export {var}="$THREADS"' in t
    lines = t.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('THREADS="${SLURM_CPUS_PER_TASK'))
    end = next(i for i, l in enumerate(lines) if l.startswith("export OPENBLAS_NUM_THREADS="))
    snippet = "\n".join(lines[start:end + 1]) + '\necho "$OMP_NUM_THREADS $MKL_NUM_THREADS $OPENBLAS_NUM_THREADS"'
    for n in (1, 4, 8, 9, 28, 40, 96):
        out = subprocess.run(["bash", "-euo", "pipefail", "-c", snippet],
                             env={"PATH": os.environ.get("PATH", ""),
                                  "SLURM_CPUS_PER_TASK": str(n)},
                             capture_output=True, text=True, check=True)
        assert out.stdout.split() == [str(pyscf_pool_threads(n))] * 3, (n, out.stdout)
    out = subprocess.run(["bash", "-euo", "pipefail", "-c", snippet],
                         env={"PATH": os.environ.get("PATH", "")},
                         capture_output=True, text=True, check=True)
    assert out.stdout.split() == [str(PYSCF_POOL_THREADS_MAX)] * 3, out.stdout


def test_the_wall_matches_its_derivation():
    """The request and the arithmetic in the header have to be the same job.

    The per-step law measured on the path the job runs (least squares
    -1.36 ms + 0.4356 us/row over four row counts; 0.5218 us/row in a second
    session, which is the slope the header carries) over the production
    sweep's 2.46e6 rows puts THIRTY cells -- the six default architectures at
    five weights -- at 10.7 h and the estimate at 11.7 h with the data
    generation; twice that is 23.4 h, inside the 24 h request. The cell count
    is read from ``DEFAULT_ARCHS`` and ``DEFAULT_WEIGHTS`` rather than
    transcribed, so widening either axis again turns this red instead of
    leaving a stale wall in the header.

    The row count itself is the part that has to be checked and not only
    quoted: the exchange block spans two channels per OPEN shell and one per
    closed one, so it is 2 x 186000 + 981512 pruned points and not twice the
    whole set's. The short-* queues cap at 4 h, so the request and the
    partition also have to agree on a long queue.

    The census is the 38-system set -- the campaign's, free Na included --
    which is why the open-shell mass is 186000 and not the 168344 the
    37-system reading gave.

    The superseded pre-rework path (a fixed 648 ms recompile per optimizer
    step, retired when the training loop moved in-module) is quoted for the
    same thirty cells at 17.1 h, which fits the wall but at 1.4x rather than
    2x -- so the header states the split as the remedy if that regression ever
    returns, and the number is pinned here so the statement cannot go stale.
    """
    t = _sbatch_text()
    assert "#SBATCH --time=24:00:00" in t
    assert "#SBATCH --partition=long-" in t
    for quoted in (
            # the per-step law, and the slope actually carried
            "0.4356 us/row", "0.5218 us/row",
            # the grids: Becke-Lebedev, then what the mean field integrates
            "1305304", "1167512", "186000", "981512",
            # the exchange BLOCK count and the measured floor survivals
            "1353512", "0.9775", "0.9743", "0.9663",
            # the rows and the wall they buy
            "1.141e6", "1.319e6", "2.46e6", "1284 s", "10.7 h", "11.7 h",
            "23.4 h",
            # and the pre-rework path, which no longer fits at 2x
            "648 ms", "17.1 h"):
        assert quoted in t, quoted
    # The arithmetic, not the strings alone. An exchange block per SPIN
    # CHANNEL of every system -- the reading this replaces -- would give
    # 2 x 1167512, and the quoted total would not reproduce.
    assert 186000 + 981512 == 1167512
    assert 2 * 186000 + 981512 == 1353512
    rows = 1.141e6 + 1.319e6
    assert abs(rows - 2.46e6) <= 0.01e6
    cell_seconds = 1000 * rows * 0.5218e-6
    assert abs(cell_seconds - 1284.0) < 15.0
    # The cell count IS the swept grid, not a number written down beside it.
    n_cells = len(pw.DEFAULT_ARCHS) * len(pw.DEFAULT_WEIGHTS)
    assert n_cells == 30
    assert f"{n_cells} x " not in t          # the header spells it in words
    assert "thirty cells" in t
    assert f"({len(pw.DEFAULT_ARCHS)} architectures x " \
           f"{len(pw.DEFAULT_WEIGHTS)} weights)" in t
    assert abs(n_cells * cell_seconds / 3600.0 - 10.7) < 0.1
    estimate_h = n_cells * cell_seconds / 3600.0 + 1.0   # + the data hour
    assert abs(estimate_h - 11.7) < 0.1
    assert 2.0 * estimate_h <= 24.0                      # the wall, at 2x
    assert abs(2.0 * estimate_h - 23.4) < 0.2
    # The pre-rework path: inside the wall, but no longer at twice the
    # estimate, which is why the header now names the split instead.
    pre_rework_h = n_cells * (cell_seconds + 0.6484 * 1000) / 3600.0 + 1.0
    assert abs(pre_rework_h - 17.1) < 0.1
    assert pre_rework_h < 24.0 < 2.0 * pre_rework_h


def test_the_census_the_wall_rests_on_is_the_campaigns_own_set():
    """The row census is a property of the SET, so the set it was measured on
    is stated rather than assumed: 38 systems on the PBE parent, 25 of them
    closed shell and 13 open. A change to the set moves every figure in the
    derivation, which is why the composition is asserted here beside it."""
    from xcquinox.alec.pretrain_data_gen import resolve_pretrain_systems
    systems = resolve_pretrain_systems(
        atoms=pw.CAMPAIGN_ATOMS, dfs_set=True, pool_atoms=True,
        reference_xc="pbe")
    n_open = sum(1 for s in systems if int(s.spin) != 0)
    assert len(systems) == 38
    assert (n_open, len(systems) - n_open) == (13, 25)
    t = _sbatch_text()
    assert "38-system set" in t
    assert "Thirteen of the 38 systems are open" in t
    assert "the other 25 are closed" in t
    assert "38 systems on the PBE\n# parent and 36 on SCAN" in t


def test_x64_is_on_and_the_platform_is_cpu():
    t = _sbatch_text()
    assert "export JAX_ENABLE_X64=1" in t
    assert "export JAX_PLATFORMS=cpu" in t


def test_activation_by_effect_and_an_import_probe():
    t = _sbatch_text()
    assert 'conda activate "$ENV_PREFIX" || true' in t
    assert '"$ENV_PREFIX"/*) : ;;' in t
    assert 'python -c "import xcquinox.alec.pretrain"' in t
    assert "FATAL: repo import failed" in t


def test_the_invocation_carries_the_swept_identity_and_a_log():
    t = _sbatch_text()
    assert "python -u hpcjobs/probe_pretrain_energy_weight.py" in t
    for flag in ('--data-dir "$DATA_DIR"', '--out      "$OUT"',
                 "--basis def2-svp", "--grid-level 3", "--n-steps 1000",
                 # The production objective, stated rather than defaulted, and
                 # the resume that makes a wall-clock kill recoverable.
                 "--polarized", "--resume"):
        assert flag in t, flag
    # The log is a file, and the exit code read is python's, not tee's.
    # Appended, not truncated: a resumed or batched submission adds to the log
    # of the run it continues rather than replacing it.
    assert 'tee -a "$LOG"' in t
    assert 'RC="${PIPESTATUS[0]}"' in t
    assert "#SBATCH --output=" in t


def test_the_script_the_job_runs_exists():
    assert _SCRIPT.is_file()


# --------------------------------------------------------------------------- #
# The exit
# --------------------------------------------------------------------------- #

def _main_block():
    """The script's ``if __name__ == "__main__":`` block, as an AST."""
    for node in ast.parse(_SCRIPT.read_text()).body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"):
            return node
    raise AssertionError("the script has no __main__ block")


def _called_name(call):
    """The dotted name a call node names -- ``sys.stdout.flush``, ``os._exit``,
    ``main`` -- or '' for a call on anything but a plain attribute chain."""
    parts, node = [], call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def test_the_entry_point_leaves_through_the_hard_exit():
    """The hard exit is read out of the source, not inferred from a run.

    JAX's atexit cleanup aborts the interpreter only sometimes on this
    backend, so a returncode assertion detects a lost hard exit only on the
    runs where the teardown happens to corrupt the heap -- it is a symptom
    check, not a pin. (It does happen: an unrelated driver run against this
    same path exited with "corrupted size vs. prev_size" while these numbers
    were being measured.) The idiom is therefore asserted directly, and it is
    the SHARED one: the last statement of the entry point is
    ``cluster._exit.run_and_exit``, which flushes both streams and then leaves
    through ``os._exit``, and no ``sys.exit`` -- which runs the teardown the
    idiom exists to skip -- is left anywhere in the block. A local copy of the
    idiom would be a second implementation of an exit rule every other job
    stage takes from one module; ``run_and_exit``'s own last-statement pin and
    the behavioural measurements in real subprocesses are in
    ``xcquinox/alec/tests/test_worker_hard_exit.py``, which enumerates this
    script among the entry points it covers.

    The callable handed to the helper is checked too: it is the wrapper that
    owns the partial table on an escape, not ``main`` itself, so the cells a
    broken run paid for are still written.
    """
    block = _main_block()
    last = block.body[-1]
    assert isinstance(last, ast.Expr) and isinstance(last.value, ast.Call), (
        ast.dump(last))
    assert _called_name(last.value).split(".")[-1] == "run_and_exit"
    assert [a.id for a in last.value.args
            if isinstance(a, ast.Name)] == ["_sweep_and_own_the_table"]
    names = [_called_name(node) for node in ast.walk(block)
             if isinstance(node, ast.Call)]
    assert "sys.exit" not in names
    assert not any(isinstance(node, ast.Raise)
                   and isinstance(node.exc, ast.Call)
                   and _called_name(node.exc) == "SystemExit"
                   for node in ast.walk(block))


def test_the_wrapper_hands_the_sweeps_own_status_back(monkeypatch):
    """0, 1 and 2 are the sweep's verdicts and pass through untouched."""
    for rc in (0, 1, 2):
        monkeypatch.setattr(pw, "main", lambda _rc=rc: _rc)
        assert pw._sweep_and_own_the_table() == rc


def test_the_wrapper_lets_a_usage_exit_keep_its_code(monkeypatch):
    """argparse refuses inside ``main``; its status is the interpreter's own,
    which the shared exit reproduces from the ``SystemExit`` it re-raises."""
    def _refuse():
        raise SystemExit(2)

    monkeypatch.setattr(pw, "main", _refuse)
    with pytest.raises(SystemExit) as excinfo:
        pw._sweep_and_own_the_table()
    assert excinfo.value.code == 2


def test_the_wrapper_hands_an_interrupt_on_after_writing_the_table(
        monkeypatch, capsys):
    """An interrupt owes the same partial table and is then handed on, so the
    shared exit reports it as an interrupt (130) rather than as an escape (3):
    the one status class the wrapper does not own."""
    written = []

    def _interrupt():
        raise KeyboardInterrupt("wall clock")

    monkeypatch.setattr(pw, "main", _interrupt)
    pw._install_partial_writer(lambda complete: written.append(complete))
    try:
        with pytest.raises(KeyboardInterrupt):
            pw._sweep_and_own_the_table()
    finally:
        pw._install_partial_writer(None)
    assert written == [False]
    assert "the partial table was written" in capsys.readouterr().err


def test_the_wrapper_writes_the_partial_table_on_an_escape(monkeypatch,
                                                           capsys):
    """An escaped exception takes the distinct code and still owes the table.

    The subprocess measurement above exercises the same path end to end; this
    one pins the two effects on the wrapper itself, where a lost partial-table
    call would otherwise only show up as a table that never appeared.
    """
    written = []

    def _boom():
        raise RuntimeError("the sweep broke")

    monkeypatch.setattr(pw, "main", _boom)
    pw._install_partial_writer(lambda complete: written.append(complete))
    try:
        assert pw._sweep_and_own_the_table() == pw.EXIT_UNHANDLED
    finally:
        pw._install_partial_writer(None)
    assert written == [False]
    captured = capsys.readouterr()
    assert "the sweep broke" in captured.err
    assert "the partial table was written" in captured.err


def test_an_unhandled_exception_still_leaves_through_that_exit(tmp_path):
    """An exception the sweep does not catch is not a licence to tear down.

    The interpreter shutdown is no safer because the sweep broke, and a run
    that dies with a traceback still owes the cells it finished. Executed on
    a real escape -- an unknown basis, which raises out of the data
    generation -- for the distinct exit code, the traceback and the partial
    table.
    """
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MKL_NUM_THREADS="4", JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_cpu_multi_thread_eigen=false")
    out = tmp_path / "table.json"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--smoke", "--basis", "no_such_basis",
         "--data-dir", str(tmp_path / "data"), "--out", str(out)],
        env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == pw.EXIT_UNHANDLED, (
        f"returncode={proc.returncode}\n"
        + proc.stdout[-2000:] + proc.stderr[-2000:])
    # Distinct from the sweep's own outcomes, which is the point of the code.
    assert pw.EXIT_UNHANDLED not in (0, 1, 2)
    assert "Traceback" in proc.stderr
    assert "no_such_basis" in proc.stderr
    payload = json.loads(out.read_text())
    assert payload["complete"] is False
    assert payload["rows"] == []
    assert payload["identity"]["basis"] == "no_such_basis"


def test_the_partial_table_writer_is_a_no_op_when_none_is_installed():
    pw._install_partial_writer(None)
    assert pw.write_partial_table() is False


def test_the_partial_table_writer_never_raises_out_of_a_failure(capsys):
    # It runs on the way out of an exception; raising here would replace that
    # exception's traceback with one about the table.
    def _boom(complete):
        raise OSError("read-only file system")

    pw._install_partial_writer(_boom)
    try:
        assert pw.write_partial_table() is False
    finally:
        pw._install_partial_writer(None)
    assert "read-only file system" in capsys.readouterr().err


def test_main_installs_a_writer_that_holds_the_cells_it_measured(
        tmp_path, monkeypatch):
    """The writer the entry point reaches is main's own, over main's rows."""
    out = tmp_path / "t.json"
    _stub_sweep(monkeypatch, {0.0: 9.0, 1.0: 0.2})
    pw._install_partial_writer(None)
    try:
        assert pw.main(["--data-dir", str(tmp_path), "--out", str(out),
                        "--archs", "deep_3x16", "--weights", "0,1"]) == 0
        out.unlink()
        assert pw.write_partial_table() is True
        payload = json.loads(out.read_text())
        assert payload["complete"] is False
        assert [r["weight"] for r in payload["rows"]] == [0.0, 1.0]
    finally:
        pw._install_partial_writer(None)


def test_the_sweep_can_be_batched_over_architectures():
    """A wall that cannot hold the whole sweep is split into submissions, each
    carrying its own architectures into the same table."""
    t = _sbatch_text()
    assert 'ARCHS="${EWSWEEP_ARCHS-}"' in t
    assert "ARCH_ARGS=(--archs \"$ARCHS\")" in t
    # The empty-array expansion, without which `set -u` kills the job when no
    # batch is named.
    assert '${ARCH_ARGS[@]+"${ARCH_ARGS[@]}"}' in t


def test_the_verdict_line_names_the_architectures_the_batch_measured():
    """Exit 0 mails "cleared"; a batched submission must not let that read as
    a statement about the six defaults it did not measure."""
    t = _sbatch_text()
    assert "${ARCHS:-the six default architectures}" in t
    assert "a swept weight cleared both gates on every architecture." not in t
    assert "every architecture MEASURED" in t


def test_an_escaped_exception_has_its_own_documented_code():
    t = _sbatch_text()
    assert "3 = an exception escaped the sweep" in t
    assert "3) echo" in t


def test_exit_code_two_is_documented_as_a_finding():
    # SLURM mails FAIL on any non-zero code; the log has to say that a 2 is a
    # completed sweep, or the mail reads as a crash.
    t = _sbatch_text()
    assert "the gates NOT cleared" in t
    assert "Not a crash" in t
    # And a wall-clock kill is recoverable rather than a loss: the log says so.
    assert "continues from the cells in" in t


# --------------------------------------------------------------------------- #
# Exit code 2 is two outcomes
# --------------------------------------------------------------------------- #

def _epilogue_case_block() -> str:
    """The shipped ``case "$RC" in ... esac`` block, verbatim.

    Extracted rather than paraphrased so what the test runs IS what the job
    runs; a rewritten copy would pass while the script kept the old text.
    """
    t = _sbatch_text()
    start = t.index('case "$RC" in')
    end = t.index("\nesac", start) + len("\nesac")
    return t[start:end]


def _run_epilogue(rc, run_out_text, tmp_path, archs=""):
    """Execute the shipped epilogue for one exit code and one captured run."""
    run_out = tmp_path / "probe.last"
    run_out.write_text(run_out_text)
    script = (f'RC={rc}\nARCHS="{archs}"\nRUN_OUT="{run_out}"\n'
              f'{_epilogue_case_block()}\n')
    done = subprocess.run(["bash", "-c", script], capture_output=True,
                          text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_argparse_refuses_a_dash_leading_architecture_list_with_code_two():
    """The reachability the epilogue has to cover, measured on the real
    parser rather than assumed.

    ``EWSWEEP_ARCHS`` is the only token the job script takes from its caller,
    and it is passed through as the value of ``--archs``. A value beginning
    with ``-`` is read by argparse as an option, not as a value, and argparse
    exits with its USAGE code -- 2, which is also the code the sweep returns
    when it completes and no weight clears the gates. The marker the epilogue
    keys on is argparse's own usage line, so it is pinned here against the
    parser that emits it.
    """
    done = subprocess.run(
        [sys.executable, str(_SCRIPT), "--data-dir", "d", "--out", "o",
         "--archs", "-deep_3x16"],
        capture_output=True, text=True)
    assert done.returncode == 2, (done.returncode, done.stderr)
    text = done.stdout + done.stderr
    assert "usage: probe_pretrain_energy_weight" in text, text
    # The sweep's OWN refusals keep code 1, which the header documents
    # separately -- the two must not be collapsed.
    other = subprocess.run(
        [sys.executable, str(_SCRIPT), "--data-dir", "d", "--out", "o",
         "--archs", "no_such_arch"],
        capture_output=True, text=True)
    assert other.returncode == 1, (other.returncode, other.stderr)


def test_the_epilogue_tells_a_usage_refusal_from_the_sweeps_finding(tmp_path):
    """Exit 2 is reported as what it was, on the shipped ``case`` block.

    Without this, a mistyped ``EWSWEEP_ARCHS`` mails "COMPLETED, the gates NOT
    cleared by any swept weight" for a job in which no cell ran -- a finding
    about the pretraining objective, invented out of a shell typo.
    """
    usage = ("usage: probe_pretrain_energy_weight [-h] --data-dir DATA_DIR\n"
             "probe_pretrain_energy_weight: error: unrecognized arguments\n")
    refused = _run_epilogue(2, usage, tmp_path, archs="-deep_3x16")
    assert "REFUSED" in refused, refused
    assert "argparse rejected the command line" in refused, refused
    assert "NOT a finding" in refused, refused
    assert "EWSWEEP_ARCHS=-deep_3x16" in refused, refused
    assert "the gates NOT cleared" not in refused, refused

    verdict = ("[probe] cells: 30 requested, 0 already measured\n"
               "recommendation: energy_term_weight = 0  [DOES NOT CLEAR]\n")
    finding = _run_epilogue(2, verdict, tmp_path)
    assert "COMPLETED, the gates NOT cleared" in finding, finding
    assert "Not a crash" in finding, finding
    assert "REFUSED" not in finding, finding


def test_the_epilogue_reads_this_submissions_output_not_the_appended_log():
    """``$LOG`` is opened with ``tee -a`` and accumulates across
    resubmissions, so a usage line from an earlier submission would still be
    in it. The classification therefore reads ``$RUN_OUT``, written fresh by
    this submission, and the invocation has to feed it."""
    t = _sbatch_text()
    assert 'RUN_OUT="${RUN_ROOT}/probe_${SLURM_JOB_ID:-manual}.last"' in t
    assert '2>&1 | tee -a "$LOG" | tee "$RUN_OUT"' in t
    assert 'RC="${PIPESTATUS[0]}"' in t
    assert 'grep -q "^usage: probe_pretrain_energy_weight" "$RUN_OUT"' in t
    # And the header says 2 carries two meanings, rather than one.
    assert "2 IS TWO OUTCOMES" in t
