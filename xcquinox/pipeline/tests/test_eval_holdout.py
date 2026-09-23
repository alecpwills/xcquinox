"""Tests for ``xcquinox.pipeline.eval_holdout``: the shared eval module.

Pure-function helpers get hand-crafted inputs with KNOWN expected outputs.
The side-effectful PBE precompute and network forward functions are exercised by the
evaluation and held-out modules' own tests; here the assertions are on this module's
own surface.
"""
from __future__ import annotations

import json

import pytest

import xcquinox.pipeline.eval_holdout as eh
from xcquinox.pipeline.eval_holdout import (
    KCAL_PER_HA,
    filter_reactions,
    held_out_pool_names,
    make_per_molecule_record,
    per_reaction_errors,
    reaction_mae_kcalmol,
    run_full_holdout_eval,
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
                   verbose_failures=True, scf_info_out=None):
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
    assert n_nan == 1  # the NaN reaction got surfaced via the gap fix
    assert mae == pytest.approx(abs(2.0 * KCAL_PER_HA - 1255.0), abs=1e-9)


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


def test_make_per_molecule_record_failure_row_reports_no_scf():
    """A species whose evaluation raised must NOT claim a converged zero-cycle
    SCF; the honest row is null/null and names the error."""
    rec = make_per_molecule_record(
        "X", {"E_pbe": -10.0}, e_nn_ha=float("nan"), in_training_subset=False,
        scf={"eval_error": "RuntimeError: alloc failed"})
    assert rec["E_total_nn"] is None
    assert rec["cycles_run"] is None
    assert rec["scf_converged"] is None
    assert rec["eval_error"] == "RuntimeError: alloc failed"
    assert "scf_total_energy" not in rec
    assert "scf_energy_step_0" not in rec


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


def _mol(name, comp):
    from types import SimpleNamespace
    return SimpleNamespace(name=name, atom_composition=comp)


def test_reaction_mae_dedups_identity_twins():
    """Permuted-name and duplicate-name twins are ONE physical reaction and
    contribute ONE term to the MAE (production BH76: 61 rows -> 54
    identities in 7 twin groups, each group carrying a single reference).
    Here: 3 rows, 2 identities -- the deduped MAE equals the MAE over one
    row per identity."""
    e = {"a": -1.0, "b": -2.0, "ts": -2.9}
    twin_a = {"name": "fwd", "reactants": ["a", "b"], "products": ["ts"],
              "coeffs": [-1.0, -1.0, 1.0], "reaction_energy_ref": 10.0}
    twin_b = {"name": "fwd_permuted", "reactants": ["b", "a"],
              "products": ["ts"], "coeffs": [-1.0, -1.0, 1.0],
              "reaction_energy_ref": 10.0}
    other = {"name": "other", "reactants": ["a"], "products": ["b"],
             "coeffs": [-1.0, 1.0], "reaction_energy_ref": 5.0}
    mae, n_used, n_nan = eh.reaction_mae_kcalmol(e, [twin_a, twin_b, other])
    mae_unique, n_unique, _ = eh.reaction_mae_kcalmol(e, [twin_a, other])
    assert n_used == 2, f"3 rows over 2 identities must count 2, got {n_used}"
    assert mae == pytest.approx(mae_unique, rel=1e-12)
    assert n_nan == 0


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
    from xcquinox.pipeline.full_benchmark_pools import load_full_held_out_pools
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


def test_density_errors_for_record_pbe_closed_form(monkeypatch):
    import numpy as np
    import xcquinox.pipeline.evaluation as ev_mod

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


# 2026-06-24: DFS tail loss -> held-out eval must REPORT the convergence-aware
# tail-weighted mean (denoised), not the arbitrary final SCF step, while still
# recording the raw final energy + full trace for forensics.
def test_evaluate_holdout_reports_tail_weighted_energy(monkeypatch):
    import jax.numpy as jnp
    import xcquinox.pipeline.solver as solver_mod
    from xcquinox.pipeline.eval_holdout import evaluate_holdout
    from xcquinox.pipeline.solver import SolverConfig, SolverMode, SolverBackend
    from xcquinox.pipeline.oneshot import tail_weighted_mean_energy

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


# ---------------------------------------------------------------------------
# assert_channel_not_sliced -- a sliced held-out channel is not a pool channel
# ---------------------------------------------------------------------------

def _slice_marked_channel(tmp_path, spec="spec_0000", chan="eval_holdout"):
    """``(run_dir, spec_dir, channel_dir)`` in the canonical pull layout."""
    run = tmp_path / "run_20260821T000000Z"
    spec_dir = run / "checkpoints" / spec
    channel = spec_dir / chan
    channel.mkdir(parents=True)
    return run, spec_dir, channel


def test_assert_channel_not_sliced_refuses_a_sliced_stamp(tmp_path):
    run, spec_dir, chan = _slice_marked_channel(tmp_path)
    (chan / "eval_metadata.json").write_text(json.dumps(
        {"channel": "eval_holdout", "species_slice": ["h", "h2"],
         "n_species": 2, "n_reactions": 1}))
    with pytest.raises(eh.SlicedChannelError) as exc:
        eh.assert_channel_not_sliced(spec_dir, "eval_holdout")
    msg = str(exc.value)
    assert str(run) in msg
    assert "spec_0000" in msg
    assert "eval_holdout" in msg
    assert "'h', 'h2'" in msg
    assert "eval_metadata.json" in msg
    assert "XCQUINOX_HELDOUT_SPECIES_SLICE" in msg
    assert "BH76 + W4-11" in msg


def test_assert_channel_not_sliced_refuses_before_any_energy_is_written(
        tmp_path):
    """The position-independent contract: the refusal keys on the marks alone,
    so a channel whose energies never landed -- the very state the pre-eval
    marker exists for -- refuses rather than reading as an empty channel."""
    _run, spec_dir, chan = _slice_marked_channel(tmp_path)
    (chan / "sliced_eval.json").write_text(json.dumps(
        {"species_slice": ["h", "h2"], "n_species": 2, "n_reactions": 1}))
    assert not (chan / "per_molecule.json").exists()
    assert not (chan / "per_reaction.json").exists()
    with pytest.raises(eh.SlicedChannelError):
        eh.assert_channel_not_sliced(spec_dir, "eval_holdout")


#: ``species_slice`` values that are truthy but not a species list. None can
#: be produced by ``cluster/_eval_one_spec`` (it writes ``list(names)`` or
#: None), so each stands for a hand-edited or corrupted mark. The refusal must
#: still be a SlicedChannelError -- a mark is a mark -- and must not iterate
#: the value: a string would render per character and a mapping its keys,
#: either of which reads as a species list that was never there.
_NON_LIST_SLICE_VALUES = ["3", "1.5", "true", '"h,h2"', '{"h": 1}']


# ---------------------------------------------------------------------------
# A reference-set schema error and an empty precompute are refused
# ---------------------------------------------------------------------------


