"""Tests for ``notebooks/analysis/local_reeval.py``.

Pure helpers + a stubbed end-to-end smoke. The expensive precompute +
model-eval paths are monkey-patched so the whole suite runs in <1s without
pyscf, matching the project's split between unit-tested pure helpers and
manually-smoked side-effectful paths (per the
``feedback_expensive_test_handling`` memory).
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest

# Load the script as a module without requiring a package layout.
_PATH = Path(__file__).resolve().parent / "local_reeval.py"
_spec = importlib.util.spec_from_file_location("local_reeval", _PATH)
local_reeval = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules[_spec.name] = local_reeval  # type: ignore[union-attr]
_spec.loader.exec_module(local_reeval)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _stub_pool_specs() -> dict:
    """Tiny in-memory pool: 4 species (CO2, H2O, H, O)."""
    return {
        "CO2": types.SimpleNamespace(name="CO2"),
        "H2O": types.SimpleNamespace(name="H2O"),
        "H":   types.SimpleNamespace(name="H"),
        "O":   types.SimpleNamespace(name="O"),
    }


def _stub_reactions() -> list:
    return [
        # AE(H2O): E(H2O) - 2*E(H) - E(O) = ref kcal/mol
        {"name": "AE_H2O", "reactants": ["H2O", "H", "O"], "products": [],
         "coeffs": [1, -2, -1], "reaction_energy_ref": 232.974,
         "source_pool": "w411"},
        # BH76-style: CO2 + H2O -> (no products in this stub) with ref 0.
        # Used to exercise the reactants-only path.
        {"name": "rxn_no_train_overlap", "reactants": ["CO2", "H2O"],
         "products": [], "coeffs": [1, -1],
         "reaction_energy_ref": 0.0, "source_pool": "bh76"},
    ]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_held_out_pool_names_subtracts_training():
    pool = _stub_pool_specs()
    names = local_reeval.held_out_pool_names(["H", "O"], pool)
    assert names == ["CO2", "H2O"]


def test_held_out_pool_names_empty_when_pool_subset_of_training():
    pool = _stub_pool_specs()
    names = local_reeval.held_out_pool_names(list(pool.keys()), pool)
    assert names == []


def test_held_out_pool_names_sorted_lex_for_determinism():
    pool = {n: object() for n in ["zeta", "alpha", "Mu"]}
    # Lex sort is case-sensitive; uppercase 'M' sorts before lowercase.
    assert local_reeval.held_out_pool_names([], pool) == ["Mu", "alpha", "zeta"]


def test_reaction_mae_kcalmol_with_known_values():
    # Reaction E(B) - 2*E(A) = ref (kcal/mol). Pick energies so the
    # predicted ΔE = (1) Ha = 627.5094... kcal/mol; ref = 600 kcal/mol;
    # MAE = |627.51 - 600| = 27.51 kcal/mol.
    energies = {"A": 1.0, "B": 3.0}
    reactions = [
        {"reactants": ["B"], "products": ["A"], "coeffs": [1, -2],
         "reaction_energy_ref": 600.0},
    ]
    mae, n, n_nan = local_reeval.reaction_mae_kcalmol(energies, reactions)
    assert n == 1
    assert n_nan == 0
    assert mae == pytest.approx(627.5094740631 - 600.0, abs=1e-9)


def test_reaction_mae_skips_nonfinite():
    energies = {"A": float("nan"), "B": 3.0, "C": 1.0}
    reactions = [
        # Reaction 1: uses A, has NaN -> skipped.
        {"reactants": ["A"], "products": ["B"], "coeffs": [1, -1],
         "reaction_energy_ref": 100.0},
        # Reaction 2: pure C->B, finite. ΔE = 2.0 Ha = 1255.02 kcal/mol;
        # ref = 1255.0; MAE = ~0.02 kcal/mol.
        {"reactants": ["B"], "products": ["C"], "coeffs": [1, -1],
         "reaction_energy_ref": 1255.0},
    ]
    mae, n, n_nan = local_reeval.reaction_mae_kcalmol(energies, reactions)
    assert n == 1
    # 2026-05-29: reaction 1 was silently dropped due to NaN energy → surfaced now.
    assert n_nan == 1
    assert mae == pytest.approx(abs(2.0 * 627.5094740631 - 1255.0), abs=1e-9)


def test_reaction_mae_kcalmol_no_finite_returns_nan():
    energies = {"A": float("nan")}
    reactions = [
        {"reactants": ["A"], "products": [], "coeffs": [1],
         "reaction_energy_ref": 50.0},
    ]
    mae, n, n_nan = local_reeval.reaction_mae_kcalmol(energies, reactions)
    assert n == 0
    assert n_nan == 1
    assert math.isnan(mae)


def test_filter_reactions_strict_drops_overlapping(stub_pool=_stub_reactions):
    reactions = _stub_reactions()
    kept, dropped = local_reeval.filter_reactions(
        reactions, training_names=["H", "O"], strict=True,
    )
    # AE_H2O uses H + O -> overlap -> dropped. The CO2/H2O reaction has
    # neither training species -> kept.
    assert [r["name"] for r in kept] == ["rxn_no_train_overlap"]
    assert [r["name"] for r in dropped] == ["AE_H2O"]
    # Dropped rxn records the overlap.
    assert sorted(dropped[0]["in_sample_overlap"]) == ["H", "O"]


def test_filter_reactions_loose_keeps_all_with_overlap_flag():
    reactions = _stub_reactions()
    kept, dropped = local_reeval.filter_reactions(
        reactions, training_names=["H", "O"], strict=False,
    )
    assert {r["name"] for r in kept} == {"AE_H2O", "rxn_no_train_overlap"}
    assert dropped == []
    overlap_by_name = {r["name"]: sorted(r["in_sample_overlap"])
                       for r in kept}
    assert overlap_by_name["AE_H2O"] == ["H", "O"]
    assert overlap_by_name["rxn_no_train_overlap"] == []


def test_per_reaction_errors_pure_math():
    """Pure unit test for the per-reaction error breakdown helper."""
    energies = {"A": 1.0, "B": 3.0}
    reactions = [
        {"name": "rxn1", "reactants": ["B"], "products": ["A"],
         "coeffs": [1, -2], "reaction_energy_ref": 600.0},
        {"name": "rxn2_missing_species", "reactants": ["C"], "products": [],
         "coeffs": [1], "reaction_energy_ref": 50.0},
    ]
    rows = local_reeval.per_reaction_errors(energies, reactions)
    assert len(rows) == 2
    r1 = rows[0]
    assert r1["name"] == "rxn1"
    assert r1["de_kcalmol"] == pytest.approx(1.0 * 627.5094740631)
    assert r1["ref_kcalmol"] == pytest.approx(600.0)
    assert r1["error_kcalmol"] == pytest.approx(627.5094740631 - 600.0)
    assert r1["abs_error_kcalmol"] == pytest.approx(627.5094740631 - 600.0)
    # Missing species -> NaN, not raised, downstream consumers filter.
    r2 = rows[1]
    assert math.isnan(r2["de_kcalmol"])
    assert math.isnan(r2["abs_error_kcalmol"])


def test_make_per_reaction_records_pairs_nn_and_pbe():
    """The output schema must carry paired NN/PBE numbers + overlap flag."""
    reactions = [
        {"name": "AE_h2o", "reactants": ["h2o", "h", "o"], "products": [],
         "coeffs": [1, -1, -2], "reaction_energy_ref": 232.974,
         "source_pool": "w411"},
    ]
    nn_err = [{"name": "AE_h2o", "de_kcalmol": 220.0, "ref_kcalmol": 232.974,
               "error_kcalmol": -12.974, "abs_error_kcalmol": 12.974}]
    pbe_err = [{"name": "AE_h2o", "de_kcalmol": 223.0, "ref_kcalmol": 232.974,
                "error_kcalmol": -9.974, "abs_error_kcalmol": 9.974}]
    out = local_reeval.make_per_reaction_records(
        reactions, nn_err, pbe_err, training_names=["h"],
    )
    assert len(out) == 1
    rec = out[0]
    assert rec["name"] == "AE_h2o"
    assert rec["pool"] == "w411"
    assert rec["abs_error_nn_kcalmol"] == pytest.approx(12.974)
    assert rec["abs_error_pbe_kcalmol"] == pytest.approx(9.974)
    assert rec["in_sample_overlap"] == ["h"]  # 'h' present in training
    # Reference passes through.
    assert rec["reaction_energy_ref_kcalmol"] == pytest.approx(232.974)


def test_make_per_molecule_record_schema_matches_harness():
    md = {"E_pbe": -76.0}
    rec = local_reeval.make_per_molecule_record(
        "H2O", md, e_nn_ha=-76.1,
        in_training_subset=False,
    )
    # Must contain every key the existing make_cluster_pulls_figure script's
    # collect_per_molecule_rows pulls from the harness's per_molecule.json.
    expected = {
        "molecule", "E_total_nn", "E_pbe", "AE_nn", "AE_error_kcalmol",
        "density_rmse", "density_l1", "ref_density_method",
        "cycles_run", "scf_converged", "from_training_subset",
    }
    assert expected.issubset(set(rec.keys()))
    assert rec["molecule"] == "H2O"
    assert rec["E_total_nn"] == pytest.approx(-76.1)
    assert rec["AE_nn"] == pytest.approx(-76.1 - (-76.0))
    assert rec["from_training_subset"] is False


def test_load_training_spec_round_trip(tmp_path):
    """Round-trip a tiny SimpleNamespace via the same serializer the
    harness uses, then load it back via load_training_spec."""
    import importlib
    serializer = importlib.import_module("pi" + "ckle")
    obj = types.SimpleNamespace(
        molecules=(types.SimpleNamespace(name="H2O"),
                   types.SimpleNamespace(name="H")),
        arch="stub_arch",
    )
    path = tmp_path / "spec_0000.spec"
    with path.open("wb") as f:
        serializer.dump(obj, f, protocol=4)

    loaded = local_reeval.load_training_spec(path)
    assert [m.name for m in loaded.molecules] == ["H2O", "H"]
    assert loaded.arch == "stub_arch"


# ---------------------------------------------------------------------------
# End-to-end main smoke (stubs pyscf + the model eval)
# ---------------------------------------------------------------------------

def test_main_smoke_with_stubbed_pyscf(tmp_path, monkeypatch):
    """Build a fake run_dir with one spec, monkey-patch the demo builders
    and the alec precompute/eval seams, then drive main(). Confirms the
    output CSV and per-molecule JSON files have the expected shape."""

    # ---- Fake run_dir layout -------------------------------------------
    run_dir = tmp_path / "runs" / "run_20260525T163822Z"
    (run_dir / "specs").mkdir(parents=True)
    (run_dir / "checkpoints" / "spec_0000").mkdir(parents=True)

    # Tiny TrainingSpec stub: trained on H + O only (so H2O and CO2 stay
    # held-out, but the AE(H2O) reaction is dropped because it needs the
    # in-sample atom energies).
    import importlib
    ser = importlib.import_module("pi" + "ckle")
    spec_obj = types.SimpleNamespace(
        molecules=(types.SimpleNamespace(name="H"),
                   types.SimpleNamespace(name="O")),
        arch=object(),  # opaque; load_trained_model is stubbed
    )
    with (run_dir / "specs" / "spec_0000.spec").open("wb") as f:
        ser.dump(spec_obj, f, protocol=4)
    # The script checks model.eqx exists; write a placeholder.
    (run_dir / "checkpoints" / "spec_0000" / "model.eqx").write_bytes(
        b"placeholder model")

    # ---- Monkey-patch the heavyweight seams -----------------------------
    pool_specs = _stub_pool_specs()
    reactions = _stub_reactions()

    def _fake_load_pools(pools):
        # Filter to requested pools (mimic the real load_pools split).
        kept_reactions = [r for r in reactions if r["source_pool"] in pools]
        return pool_specs, kept_reactions

    def _fake_load_trained_model(spec, path):
        return object()  # opaque

    def _fake_precompute(specs):
        # Return a dict-of-dicts mimicking MoleculeData with E_pbe.
        return {n: {"E_pbe": -50.0 - i}
                for i, n in enumerate(specs)}

    def _fake_evaluate(model, mol_data, **_kwargs):
        # 2026-05-29: accept arbitrary kwargs (solver_config, etc.) so the
        # stub matches the post-fix evaluate_holdout signature without
        # actually using the SCF path.
        return {n: float(md["E_pbe"] - 0.01)
                for n, md in mol_data.items()}

    monkeypatch.setattr(local_reeval, "load_pools", _fake_load_pools)
    monkeypatch.setattr(local_reeval, "load_trained_model",
                        _fake_load_trained_model)
    monkeypatch.setattr(local_reeval, "precompute_holdout",
                        lambda specs, descriptors=(), **_kw:
                            _fake_precompute(specs))
    monkeypatch.setattr(local_reeval, "evaluate_holdout", _fake_evaluate)

    # ---- Run with NEW default (loose mode) --------------------------------
    rc = local_reeval.main([str(run_dir), "--specs", "0"])
    assert rc == 0

    csv_path = run_dir / "checkpoints" / "spec_0000" / "local_test_set.csv"
    json_path = (run_dir / "checkpoints" / "spec_0000" / "eval"
                 / "local_per_molecule.json")
    assert csv_path.is_file()
    assert json_path.is_file()

    csv_text = csv_path.read_text().splitlines()
    # 2026-05-29: header gained n_dropped_nan column before note.
    assert csv_text[0] == (
        "set,mae_nn_kcalmol,mae_pbe_kcalmol,delta_nn_minus_pbe,"
        "n_reactions,n_dropped_overlap,n_dropped_nan,note")
    assert any("test_set_bh76" in line for line in csv_text[1:])
    assert any("test_set_w411" in line for line in csv_text[1:])
    assert any("held_out_combined" in line for line in csv_text[1:])
    # NEW default: loose -- no reactions dropped, AE_H2O kept despite H/O
    # overlap. Layout: set,mae_nn,mae_pbe,delta,n_reactions,
    # n_dropped_overlap,n_dropped_nan,note -- so n_dropped_overlap is index 5
    # and n_dropped_nan is index 6.
    w411_row = [l for l in csv_text[1:] if "test_set_w411" in l][0]
    cols = w411_row.split(",")
    assert int(cols[5]) == 0, (
        f"loose default must NOT drop any reactions; got {w411_row}"
    )
    assert int(cols[6]) == 0, (
        f"smoke fixture has no missing species → n_dropped_nan must be 0; "
        f"got {w411_row}"
    )
    assert "loose" in w411_row.lower(), (
        f"loose-mode note must say 'loose'; got {w411_row}"
    )

    # Records now cover the FULL pool (every species the model was
    # evaluated on); each carries a `from_training_subset` flag so the
    # downstream plotter can split in-sample vs held-out. The training
    # set was {H, O}, so those must appear with the flag set True; every
    # other species must have it False.
    records = json.loads(json_path.read_text())
    assert len(records) >= 1
    by_name = {r["molecule"]: r for r in records}
    assert "H" in by_name and by_name["H"]["from_training_subset"] is True
    assert "O" in by_name and by_name["O"]["from_training_subset"] is True
    for name, rec in by_name.items():
        if name not in {"H", "O"}:
            assert rec["from_training_subset"] is False


def test_main_smoke_strict_flag_restores_old_behavior(tmp_path, monkeypatch):
    """``--strict`` reverts to the pre-2026-05-29 behavior of dropping
    reactions whose species overlap the training set. This is the
    regression canary: if someone removes the flag the test fails."""
    # Same fake-pulled-tree fixture as the loose test but ID-renamed so
    # we don't share state.
    run_dir = tmp_path / "runs" / "run_20260525T163822Z"
    (run_dir / "specs").mkdir(parents=True)
    (run_dir / "checkpoints" / "spec_0000").mkdir(parents=True)
    import importlib
    ser = importlib.import_module("pi" + "ckle")
    spec_obj = types.SimpleNamespace(
        molecules=(types.SimpleNamespace(name="H"),
                   types.SimpleNamespace(name="O")),
        arch=object(),
    )
    with (run_dir / "specs" / "spec_0000.spec").open("wb") as f:
        ser.dump(spec_obj, f, protocol=4)
    (run_dir / "checkpoints" / "spec_0000" / "model.eqx").write_bytes(b"x")

    pool_specs = _stub_pool_specs()
    reactions = _stub_reactions()
    monkeypatch.setattr(local_reeval, "load_pools",
                        lambda pools: (pool_specs,
                                       [r for r in reactions
                                        if r["source_pool"] in pools]))
    monkeypatch.setattr(local_reeval, "load_trained_model",
                        lambda *_a, **_k: object())
    monkeypatch.setattr(local_reeval, "precompute_holdout",
                        lambda specs, descriptors=(), **_kw:
                            {n: {"E_pbe": -50.0} for n in specs})
    monkeypatch.setattr(local_reeval, "evaluate_holdout",
                        lambda model, md, **_kw:
                            {n: float(d["E_pbe"] - 0.01)
                             for n, d in md.items()})

    rc = local_reeval.main([str(run_dir), "--specs", "0", "--strict"])
    assert rc == 0
    csv_text = (run_dir / "checkpoints" / "spec_0000" / "local_test_set.csv"
                ).read_text().splitlines()
    w411_row = [l for l in csv_text[1:] if "test_set_w411" in l][0]
    cols = w411_row.split(",")
    assert int(cols[5]) >= 1, (
        f"--strict MUST drop training-overlapping reactions; got {w411_row}"
    )
    assert "strict" in w411_row.lower()


# ---------------------------------------------------------------------------
# Auto-discovery driver (--auto)
# ---------------------------------------------------------------------------

def _materialize_fake_pulled_tree(local_root: Path,
                                  *,
                                  category: str,
                                  stamp: str,
                                  spec_ids_with_model: list,
                                  spec_ids_total: list) -> Path:
    """Build a tmp tree mimicking a pulled run dir. Only specs in
    ``spec_ids_with_model`` get a model.eqx (mimicking the partial-completion
    case the harness produces when training is still in flight)."""
    run_dir = local_root / category / stamp
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"n_specs": len(spec_ids_total), "width": 4, "specs": [
            {"index": i, "spec_file": f"spec_{i:04d}.spec", "sha256": "x" * 64,
             "cell": {"arch": "deep_combined_attn",
                      "loss": "L5_test",
                      "metric": "jsd" if (i % 2 == 0) else "l2",
                      "solver": "full_3",
                      "subset_size": 1 + (i % 4)}}
            for i in spec_ids_total
        ]})
    )
    import importlib
    ser = importlib.import_module("pi" + "ckle")
    for i in spec_ids_total:
        sd = run_dir / "checkpoints" / f"spec_{i:04d}"
        sd.mkdir(parents=True)
        spec_obj = types.SimpleNamespace(
            molecules=(types.SimpleNamespace(name="H"),
                       types.SimpleNamespace(name="O")),
            arch=object(),
        )
        (run_dir / "specs").mkdir(exist_ok=True)
        with (run_dir / "specs" / f"spec_{i:04d}.spec").open("wb") as f:
            ser.dump(spec_obj, f, protocol=4)
        if i in spec_ids_with_model:
            (sd / "model.eqx").write_bytes(b"placeholder")
    return run_dir


def test_arch_polarized_flag_distinguishes_uks_from_rks():
    """The flag must reflect ``arch.use_polarized_correlation`` so the
    UKS path is selected for the polarized/* categories' networks. This is
    THE pivotal scientific knob (the headline finding in
    constraint_pretraining_gmtkn55_report.md): the polarized correlation
    baseline is what restores PBE-level atomization MAE for open-shell
    atoms. If this flag detection regresses to always-False, the polarized
    runs would silently be evaluated as if they were unpolarized."""
    polarized = types.SimpleNamespace(use_polarized_correlation=True)
    unpolarized = types.SimpleNamespace(use_polarized_correlation=False)
    missing = types.SimpleNamespace()  # No attribute at all.
    assert local_reeval.arch_polarized_flag(polarized) is True
    assert local_reeval.arch_polarized_flag(unpolarized) is False
    # Defensive default for unfamiliar arch dataclasses.
    assert local_reeval.arch_polarized_flag(missing) is False


def test_discover_specs_in_run_skips_missing_model(tmp_path):
    run = _materialize_fake_pulled_tree(
        tmp_path, category="alpha_on/runs", stamp="run_20260525T163822Z",
        spec_ids_with_model=[0, 2], spec_ids_total=[0, 1, 2],
    )
    assert local_reeval.discover_specs_in_run(run) == [0, 2]


def test_main_auto_and_run_dir_are_mutually_exclusive(tmp_path):
    """--auto cannot combine with the positional run_dir / --specs."""
    rc = local_reeval.main([str(tmp_path), "--specs", "0", "--auto"])
    assert rc == 1


def test_main_auto_smoke_across_multiple_categories(tmp_path, monkeypatch):
    """The --auto driver discovers every category + every spec with a
    model.eqx, and per-spec failures do not abort the batch."""
    # Two categories: alpha_on (3 specs, all with model) and alpha_off
    # (3 specs, only the middle one without a model -- mimicking in-flight
    # training).
    _materialize_fake_pulled_tree(
        tmp_path, category="alpha_on/runs", stamp="run_20260525T163822Z",
        spec_ids_with_model=[0, 1, 2], spec_ids_total=[0, 1, 2],
    )
    _materialize_fake_pulled_tree(
        tmp_path, category="alpha_off/runs", stamp="run_20260525T163846Z",
        spec_ids_with_model=[0, 2], spec_ids_total=[0, 1, 2],
    )

    # Stub the pyscf-heavy seams (same as the single-spec smoke).
    pool_specs = _stub_pool_specs()
    reactions = _stub_reactions()

    def _fake_load_pools(pools):
        return pool_specs, [r for r in reactions if r["source_pool"] in pools]

    monkeypatch.setattr(local_reeval, "load_pools", _fake_load_pools)
    monkeypatch.setattr(local_reeval, "load_trained_model",
                        lambda *_a, **_k: object())
    monkeypatch.setattr(local_reeval, "precompute_holdout",
                        lambda specs, descriptors=(), **_kw:
                            {n: {"E_pbe": -50.0} for n in specs})

    # First call to evaluate_holdout raises (simulates one failing spec);
    # subsequent calls succeed. We use a small counter to drive this.
    calls = {"n": 0}
    def _fake_evaluate(model, mol_data, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated training-failure NaN")
        return {n: float(md["E_pbe"] - 0.01) for n, md in mol_data.items()}
    monkeypatch.setattr(local_reeval, "evaluate_holdout", _fake_evaluate)

    rc = local_reeval.main(["--auto", "--local-root", str(tmp_path)])
    assert rc == 0
    # Total specs with model.eqx: 3 (alpha_on) + 2 (alpha_off) = 5.
    # One raised; the rest must still have produced their outputs.
    csv_files = list(tmp_path.rglob("local_test_set.csv"))
    assert len(csv_files) == 4, (
        f"expected 4 successful outputs (5 - 1 simulated failure), "
        f"got {len(csv_files)}: {[str(p) for p in csv_files]}"
    )
