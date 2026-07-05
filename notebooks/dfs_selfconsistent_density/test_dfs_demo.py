"""Unit tests for dfs_demo.py -- assert the demo spec is DFS-exact.

These exercise only the pure spec-assembly / pool-selection / aggregation logic
(no PySCF SCF, no CCSD generation, no training), so they run in seconds. The
end-to-end density training is exercised separately by the SMOKE notebook run.
"""
import glob
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import dfs_demo  # noqa: E402
from xcquinox.alec.solver import SolverMode  # noqa: E402
from xcquinox.alec.cluster.domain import KCAL_PER_HA  # noqa: E402


# ---------------------------------------------------------------------------
# Pool selection
# ---------------------------------------------------------------------------

def test_select_dfs_points_returns_four_ae_reactions():
    chosen = dfs_demo.select_dfs_points()
    assert len(chosen) == 4
    assert {tp.name for tp in chosen} == set(dfs_demo.DEFAULT_MOLECULE_HILLS)
    # ae_as_reactions=True => every AE point is a bh76-kind predicted-atom reaction.
    for tp in chosen:
        assert tp.kind == "bh76"
        assert tp.metadata.get("ae_form") == "predicted_atom_reaction"
        assert tp.metadata.get("e_rxn_ref") is not None


def test_smoke_hills_are_subset_of_default():
    assert set(dfs_demo.SMOKE_MOLECULE_HILLS) <= set(dfs_demo.DEFAULT_MOLECULE_HILLS)


def test_select_dfs_points_rejects_unknown_molecule():
    with pytest.raises(ValueError, match="not in the DFS pool"):
        dfs_demo.select_dfs_points(["NoSuchMolecule"])


# ---------------------------------------------------------------------------
# Molecule specs (species union) + spins
# ---------------------------------------------------------------------------

def test_build_mol_specs_species_union_and_spins():
    chosen = dfs_demo.select_dfs_points()
    with tempfile.TemporaryDirectory() as refs:
        specs = dfs_demo.build_mol_specs(
            chosen, basis="sto-3g", grid_level=1, refs_dir=refs)
    by_name = {ms.name: ms for ms in specs}
    # 4 molecules + the H/O/Li/N atom anchors carried by the AE reactions.
    assert set(by_name) == {"H2O", "HLi", "HO", "HN", "H", "O", "Li", "N"}
    # Spin-diverse: closed shells + a doublet + a triplet + open-shell atoms.
    assert by_name["H2O"].spin == 0
    assert by_name["HLi"].spin == 0
    assert by_name["HO"].spin == 1      # OH doublet
    assert by_name["HN"].spin == 2      # NH triplet
    assert by_name["H"].spin == 1
    assert by_name["O"].spin == 2
    assert by_name["N"].spin == 3
    # No CCSD refs generated -> external paths stay None.
    assert all(ms.external_data_path is None for ms in specs)


def test_build_mol_specs_injects_missing_li_anchor():
    # H2O + OH has no Li-bearing molecule, so the Dick ("H","Li") regularizer
    # anchor for Li must be injected (matching spec_builder), else L5 loss
    # construction would reject the spec.
    chosen = dfs_demo.select_dfs_points(dfs_demo.SMOKE_MOLECULE_HILLS)
    with tempfile.TemporaryDirectory() as refs:
        specs = dfs_demo.build_mol_specs(
            chosen, basis="sto-3g", grid_level=1, refs_dir=refs)
    by_name = {ms.name: ms for ms in specs}
    assert {"H2O", "HO", "H", "O"} <= set(by_name)
    assert "Li" in by_name                         # injected anchor
    assert by_name["Li"].spin == 1                 # Li doublet ground state
    assert sum(dict(by_name["Li"].atom_composition).values()) == 1


def test_molecule_specs_excludes_atoms():
    chosen = dfs_demo.select_dfs_points()
    with tempfile.TemporaryDirectory() as refs:
        specs = dfs_demo.build_mol_specs(
            chosen, basis="sto-3g", grid_level=1, refs_dir=refs)
    mols = dfs_demo.molecule_specs(specs)
    assert {ms.name for ms in mols} == {"H2O", "HLi", "HO", "HN"}


# ---------------------------------------------------------------------------
# Solvers + architecture
# ---------------------------------------------------------------------------

def test_solver_configs_full3_full25():
    cfgs = dfs_demo.solver_configs()
    f3, f25 = cfgs["full_3"], cfgs["full_25"]
    assert f3.mode == SolverMode.FULL and f3.max_cycles == 3
    assert f25.mode == SolverMode.FULL and f25.max_cycles == 25
    assert f25.scf_grad_checkpoint is True
    for c in (f3, f25):
        assert c.mixer_name == "decaying_linear"
        assert dict(c.mixer_kwargs) == {"base": 0.3, "floor": 0.3}
        assert c.scf_loss_use_tail is True
        assert c.scf_loss_tail == 10
        assert c.scf_loss_weight_power == 2.0


def test_dfs_arch_polarized_correlation_on():
    arch = dfs_demo.dfs_arch("deep_3x16")
    assert arch.use_polarized_correlation is True
    assert dfs_demo.dfs_arch("deep_3x16", polarized=False).use_polarized_correlation is False


# ---------------------------------------------------------------------------
# THE DFS-exactness assertions on the assembled TrainingSpec
# ---------------------------------------------------------------------------

def test_build_dfs_training_spec_is_dfs_exact():
    chosen = dfs_demo.select_dfs_points()
    with tempfile.TemporaryDirectory() as tmp:
        specs = dfs_demo.build_mol_specs(
            chosen, basis="sto-3g", grid_level=1, refs_dir=tmp)
        spec = dfs_demo.build_dfs_training_spec(
            arch=dfs_demo.dfs_arch("deep_3x16"),
            solver_cfg=dfs_demo.solver_configs()["full_3"],
            chosen_points=chosen,
            mol_specs=specs,
            checkpoint_dir=os.path.join(tmp, "ckpt"),
            n_steps=5,
        )
        # The critical DFS knobs.
        assert spec.update_scheme == "per_molecule"
        assert spec.loss_name == "L5_gradnorm_vxc_step7"
        assert spec.channel_weights == ()          # -> _DEFAULT_CHANNEL_WEIGHTS (rho 20x)
        assert spec.require_atom_anchors is False
        assert spec.arch.use_polarized_correlation is True

        lk = spec.loss_kwargs_dict
        assert lk["density_per_electron"] is True
        assert lk["regularize_atom_syms"] == ("H", "Li")
        assert len(lk["bh76_reactions"]) == 4      # the 4 AE-as-reactions
        assert list(lk["ip13_pairs"]) == []
        # AE compounds are aux-only in the fixed-anchor channel (trained via rxn).
        assert set(dfs_demo.DEFAULT_MOLECULE_HILLS) <= set(lk["aux_only_names"])

        # Optimizer / schedule = dfs_step7 recipe.
        assert spec.lr_start == 1e-3
        assert spec.lr_end == 1e-5
        assert spec.lr_decay_start == 0.5
        assert spec.grad_clip == 1.0
        assert spec.weight_decay == 1e-4
        assert spec.seed == 42

        # Solver threaded through.
        assert spec.solver_config.mode == SolverMode.FULL
        assert spec.solver_config.max_cycles == 3

        # AE reference conversion kcal/mol -> Ha (H2O = 232.974 kcal/mol).
        h2o_rxn = next(r for r in lk["bh76_reactions"] if r["name"] == "H2O")
        assert abs(h2o_rxn["e_rxn_ref"] - 232.974 / KCAL_PER_HA) < 1e-9
        assert h2o_rxn["coeffs"] == (-1.0, 2.0, 1.0)   # -H2O + 2H + O

        # And it is a valid spec.
        spec.validate()


def test_targets_atoms_use_chakravorty():
    chosen = dfs_demo.select_dfs_points()
    with tempfile.TemporaryDirectory() as tmp:
        specs = dfs_demo.build_mol_specs(
            chosen, basis="sto-3g", grid_level=1, refs_dir=tmp)
        spec = dfs_demo.build_dfs_training_spec(
            arch=dfs_demo.dfs_arch("deep_3x16"),
            solver_cfg=dfs_demo.solver_configs()["full_3"],
            chosen_points=chosen, mol_specs=specs,
            checkpoint_dir=os.path.join(tmp, "ckpt"), n_steps=5,
        )
    t = spec.targets_dict
    assert t["O"] == -75.0673      # Chakravorty neutral O
    assert t["N"] == -54.5892      # Chakravorty neutral N
    assert t["H"] == -0.5
    ae = spec.atom_energies_dict
    assert ae["Li"] == -7.4781


# ---------------------------------------------------------------------------
# Density-diagnostic aggregation (pure logic)
# ---------------------------------------------------------------------------

def test_aggregate_density_diagnostics_beats_pbe_and_skips_atoms():
    records = [
        {"name": "H2O", "density_rmse": 0.010, "density_rmse_pbe": 0.020},
        {"name": "HO", "density_rmse": 0.030, "density_rmse_pbe": 0.020},
        {"name": "H", "density_rmse": None, "density_rmse_pbe": None, "skipped": True},
    ]
    rows = dfs_demo.aggregate_density_diagnostics(records)
    assert len(rows) == 2      # atom skipped
    by = {r["name"]: r for r in rows}
    assert by["H2O"]["beats_pbe"] is True
    assert by["HO"]["beats_pbe"] is False
    assert abs(by["H2O"]["improvement"] - 0.010) < 1e-12


# ---------------------------------------------------------------------------
# Pretrain-atom derivation (must never emit He, which is absent at the paper basis)
# ---------------------------------------------------------------------------

def test_pretrain_atoms_for_default_systems():
    chosen = dfs_demo.select_dfs_points()
    with tempfile.TemporaryDirectory() as refs:
        specs = dfs_demo.build_mol_specs(
            chosen, basis="sto-3g", grid_level=1, refs_dir=refs)
    atoms = dict(dfs_demo.pretrain_atoms_for(specs))
    assert atoms == {"H": 1, "Li": 1, "N": 3, "O": 2}   # system elements, ground-state 2S
    assert "He" not in atoms                              # the bug that crashed 6-311++G(3df,2pd)


def test_pretrain_atoms_for_smoke_subset():
    chosen = dfs_demo.select_dfs_points(dfs_demo.SMOKE_MOLECULE_HILLS)  # H2O + OH (+ injected Li)
    with tempfile.TemporaryDirectory() as refs:
        specs = dfs_demo.build_mol_specs(
            chosen, basis="sto-3g", grid_level=1, refs_dir=refs)
    assert dict(dfs_demo.pretrain_atoms_for(specs)) == {"H": 1, "Li": 1, "O": 2}


# ---------------------------------------------------------------------------
# PBE pretraining wiring (runs a small PBE SCF + pretrain regression)
# ---------------------------------------------------------------------------

def test_pretrain_to_pbe_writes_checkpoint(tmp_path):
    fired = []
    ckpt = dfs_demo.pretrain_to_pbe(
        dfs_demo.dfs_arch("deep_3x16"),
        data_dir=str(tmp_path / "pdata"),
        checkpoint_dir=str(tmp_path / "pre"),
        basis="sto-3g", grid_level=1, n_steps=2, atoms=(("H", 1),),
        progress_callback=lambda p: fired.append(p["phase"]))
    assert os.path.isfile(os.path.join(ckpt, "xnet.eqx"))
    assert os.path.isfile(os.path.join(ckpt, "cnet.eqx"))
    # Progress callback fired for both the exchange (X) and correlation (C) nets.
    assert {"X", "C"} <= set(fired)
    # Rerun reuses the checkpoint (skips pretraining -> no callback fires).
    fired2 = []
    dfs_demo.pretrain_to_pbe(
        dfs_demo.dfs_arch("deep_3x16"),
        data_dir=str(tmp_path / "pdata"), checkpoint_dir=str(tmp_path / "pre"),
        basis="sto-3g", grid_level=1, n_steps=2, atoms=(("H", 1),),
        progress_callback=lambda p: fired2.append(p["phase"]))
    assert fired2 == []
    # The training spec then loads it (pretrain_checkpoint dir must validate).
    chosen = dfs_demo.select_dfs_points(dfs_demo.SMOKE_MOLECULE_HILLS)
    specs = dfs_demo.build_mol_specs(
        chosen, basis="sto-3g", grid_level=1, refs_dir=str(tmp_path / "refs"))
    spec = dfs_demo.build_dfs_training_spec(
        arch=dfs_demo.dfs_arch("deep_3x16"),
        solver_cfg=dfs_demo.solver_configs()["full_3"],
        chosen_points=chosen, mol_specs=specs,
        checkpoint_dir=str(tmp_path / "run"), n_steps=2,
        pretrain_checkpoint=ckpt)
    spec.validate()
    assert spec.pretrain_checkpoint == ckpt


# ---------------------------------------------------------------------------
# Self-consistent atomization energy + combined energy-density metric
# ---------------------------------------------------------------------------

_KCAL = dfs_demo._HARTREE_TO_KCAL


def test_self_consistent_ae_uses_own_atoms():
    # AE = (2*E(H) + E(O)) - E(H2O), from the functional's OWN atom energies.
    records = [
        {"molecule": "H", "E_total_nn": -0.5, "E_pbe": -0.5, "skip_reason": "atomic_system"},
        {"molecule": "O", "E_total_nn": -75.0, "E_pbe": -75.0, "skip_reason": "atomic_system"},
        {"molecule": "H2O", "E_total_nn": -76.4, "E_pbe": -76.4},
    ]
    rows = dfs_demo.self_consistent_ae(records, {"H2O": {"H": 2, "O": 1}}, {"H2O": 230.0})
    assert len(rows) == 1 and rows[0]["name"] == "H2O"
    # AE = 2*(-0.5) + (-75.0) - (-76.4) = 0.4 Ha -- a physical AE (~251 kcal/mol),
    # NOT the anchored absolute-energy offset.
    assert abs(rows[0]["ae_nn_kcal"] - 0.4 * _KCAL) < 1e-6
    assert abs(rows[0]["err_nn"] - (0.4 * _KCAL - 230.0)) < 1e-6
    assert rows[0]["ae_nn_kcal"] > 200.0


def test_self_consistent_ae_beats_pbe_when_atoms_closer():
    # NN reproduces the OH atomization energy better than PBE.
    records = [
        {"molecule": "H", "E_total_nn": -0.5, "E_pbe": -0.5, "skip_reason": "atomic_system"},
        {"molecule": "O", "E_total_nn": -75.0, "E_pbe": -75.02, "skip_reason": "atomic_system"},
        {"molecule": "HO", "E_total_nn": -75.67, "E_pbe": -75.70},
    ]
    rows = dfs_demo.self_consistent_ae(records, {"HO": {"H": 1, "O": 1}}, {"HO": 106.4})
    assert rows[0]["beats_pbe"] and abs(rows[0]["err_nn"]) < abs(rows[0]["err_pbe"])


def test_self_consistent_ae_skips_molecule_missing_an_atom():
    records = [
        {"molecule": "H", "E_total_nn": -0.5, "E_pbe": -0.5, "skip_reason": "atomic_system"},
        {"molecule": "HO", "E_total_nn": -75.67, "E_pbe": -75.70},  # O never evaluated
    ]
    assert dfs_demo.self_consistent_ae(records, {"HO": {"H": 1, "O": 1}}, {"HO": 106.4}) == []


def test_combined_energy_density_self_calibrated_harmonic_mean():
    ae_rows = [{"err_nn": 1.0, "err_pbe": 4.0}, {"err_nn": -2.0, "err_pbe": -4.0}]
    density_rows = [{"density_rmse": 1e-4, "density_rmse_pbe": 2e-4},
                    {"density_rmse": 3e-4, "density_rmse_pbe": 4e-4}]
    m = dfs_demo.combined_energy_density(ae_rows, density_rows)
    assert abs(m["E_MAE_nn"] - 1.5) < 1e-12 and abs(m["E_MAE_pbe"] - 4.0) < 1e-12
    assert abs(m["gamma"] - (4.0 / 3e-4)) < 1e-6
    # gamma self-calibrated from PBE => ED_pbe == E_MAE_pbe.
    assert abs(m["ED_pbe"] - 4.0) < 1e-9
    gd_nn = (4.0 / 3e-4) * 2e-4
    assert abs(m["ED_nn"] - 2.0 / (1 / 1.5 + 1 / gd_nn)) < 1e-9
    assert m["beats_pbe"] and m["ED_nn"] < m["ED_pbe"]


def test_combined_energy_density_requires_nonempty():
    with pytest.raises(ValueError):
        dfs_demo.combined_energy_density([], [{"density_rmse": 1e-4, "density_rmse_pbe": 2e-4}])


# Literature electronic atomization energies (De, kcal/mol) for the demo set; the
# corrected self-consistent AE must beat PBE against these on every trained model.
_LIT_AE_KCAL = {"HLi": 57.8, "HO": 106.4, "HN": 82.8, "H2O": 232.2}
_DEMO_COMP = {"HLi": {"H": 1, "Li": 1}, "HO": {"H": 1, "O": 1},
              "HN": {"H": 1, "N": 1}, "H2O": {"H": 2, "O": 1}}
_RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


@pytest.mark.skipif(not os.path.isdir(_RUNS_DIR), reason="no trained runs/ present")
def test_corrected_ae_and_combined_beat_pbe_on_real_runs():
    pmjs = sorted(glob.glob(os.path.join(_RUNS_DIR, "*__full_*", "eval", "per_molecule.json")))
    assert pmjs, "expected trained per_molecule.json outputs"
    for pmj in pmjs:
        with open(pmj) as fh:
            records = json.load(fh)
        rows = dfs_demo.self_consistent_ae(records, _DEMO_COMP, _LIT_AE_KCAL)
        assert rows, f"{pmj}: no AE rows"
        e_mae_nn = sum(abs(r["err_nn"]) for r in rows) / len(rows)
        e_mae_pbe = sum(abs(r["err_pbe"]) for r in rows) / len(rows)
        assert e_mae_nn < e_mae_pbe, f"{pmj}: AE-MAE NN {e_mae_nn:.2f} !< PBE {e_mae_pbe:.2f}"
        m = dfs_demo.combined_energy_density(rows, dfs_demo.aggregate_density_diagnostics(records))
        assert m["beats_pbe"], f"{pmj}: ED NN {m['ED_nn']:.2f} !< PBE {m['ED_pbe']:.2f}"


# ---------------------------------------------------------------------------
# Orientation lock threaded through the demo
# ---------------------------------------------------------------------------

def test_orientation_lock_strength_on_both_solvers():
    """The demo turns the lock ON, so BOTH FULL solvers carry the same nonzero
    strength -> training AND eval precompute bias h_core identically."""
    cfgs = dfs_demo.solver_configs()
    assert dfs_demo.ORIENTATION_LOCK_STRENGTH > 0.0
    for name in ("full_3", "full_25"):
        assert cfgs[name].orientation_lock_strength == dfs_demo.ORIENTATION_LOCK_STRENGTH


# ---------------------------------------------------------------------------
# Held-out generalization set (N2 + NO + NO2)
# ---------------------------------------------------------------------------

def test_heldout_points_are_pool_entries_with_correct_spins():
    pts = dfs_demo.heldout_points()
    assert {tp.name for tp in pts} == set(dfs_demo.HELDOUT_MOLECULE_HILLS)
    specs = dfs_demo.build_mol_specs(pts, basis="def2-svp", grid_level=2,
                                     refs_dir=tempfile.mkdtemp())
    by = {ms.name: ms for ms in specs}
    # real pool spins (NO/NO2 are open-shell doublets; N2 closed-shell)
    assert by["N2"].spin == 0
    assert by["NO"].spin == 1
    assert by["NO2"].spin == 1
    # the N and O atom anchors must be in the union for own-atom AE
    assert "N" in by and "O" in by


def test_heldout_comp_and_ae_from_pool_no_fabrication():
    pts = dfs_demo.heldout_points()
    specs = dfs_demo.build_mol_specs(pts, basis="def2-svp", grid_level=2,
                                     refs_dir=tempfile.mkdtemp())
    comp, ae = dfs_demo.heldout_comp_and_ae(pts, specs)
    assert comp["N2"] == {"N": 2}
    assert comp["NO"] == {"N": 1, "O": 1}
    assert comp["NO2"] == {"N": 1, "O": 2}
    # AE references come from the pool (positive kcal/mol), not hand-typed here
    for name in ("N2", "NO", "NO2"):
        assert ae[name] > 0.0


def test_heldout_summary_tallies_beats_pbe():
    combined = {
        "deep_3x16__full_3": {  # beats on all three
            "E_MAE_nn": 1.0, "E_MAE_pbe": 2.0, "D_nn": 1e-4, "D_pbe": 2e-4,
            "ED_nn": 1.0, "ED_pbe": 2.0, "beats_pbe": True},
        "deep_3x16__full_25": {  # loses on all three
            "E_MAE_nn": 3.0, "E_MAE_pbe": 2.0, "D_nn": 3e-4, "D_pbe": 2e-4,
            "ED_nn": 3.0, "ED_pbe": 2.0, "beats_pbe": False},
        "deep_rung35_3x16__full_3": {  # beats density only
            "E_MAE_nn": 3.0, "E_MAE_pbe": 2.0, "D_nn": 1e-4, "D_pbe": 2e-4,
            "ED_nn": 3.0, "ED_pbe": 2.0, "beats_pbe": False},
    }
    s = dfs_demo.heldout_summary(combined)
    assert s["n_models"] == 3
    assert s["n_beat_ae"] == 1
    assert s["n_beat_density"] == 2
    assert s["n_beat_ed"] == 1
    assert len(s["rows"]) == 3


_HELDOUT_RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs", "heldout")


@pytest.mark.skipif(not os.path.isdir(_HELDOUT_RUNS_DIR),
                    reason="no held-out eval outputs present (run the notebook §9)")
def test_heldout_generalization_beats_pbe_on_real_runs():
    """When the notebook's §9 held-out eval has run, every trained model's
    held-out combined energy-density error must beat PBE."""
    pts = dfs_demo.heldout_points()
    specs = dfs_demo.build_mol_specs(pts, basis=dfs_demo.DFS_BASIS,
                                     grid_level=dfs_demo.DFS_GRID_LEVEL,
                                     refs_dir=tempfile.mkdtemp())
    comp, ae = dfs_demo.heldout_comp_and_ae(pts, specs)
    pmjs = sorted(glob.glob(os.path.join(_HELDOUT_RUNS_DIR, "*", "eval",
                                         "per_molecule.json")))
    assert pmjs, "expected held-out per_molecule.json outputs"
    for pmj in pmjs:
        with open(pmj) as fh:
            records = json.load(fh)
        ae_rows = dfs_demo.self_consistent_ae(records, comp, ae)
        d_rows = dfs_demo.aggregate_density_diagnostics(records)
        assert ae_rows and d_rows, f"{pmj}: empty held-out rows"
        m = dfs_demo.combined_energy_density(ae_rows, d_rows)
        assert m["beats_pbe"], f"{pmj}: held-out ED NN {m['ED_nn']:.2f} !< PBE {m['ED_pbe']:.2f}"


# ---------------------------------------------------------------------------
# SCAN self-consistent baseline (a meta-GGA comparator alongside PBE). The demo
# asks whether a trained meta-GGA net improves on SCAN itself at the CCSD density
# + atomization energy. The aggregation fns gain a *_scan series parallel to PBE.
# ---------------------------------------------------------------------------

def test_weighted_density_rmse_matches_hand_formula():
    import numpy as np
    rho = np.array([1.0, 2.0, 3.0])
    rho_ref = np.array([1.5, 1.0, 3.0])
    w = np.array([2.0, 1.0, 1.0])
    got = dfs_demo._weighted_density_rmse(rho, rho_ref, w)
    # sqrt( (2*0.25 + 1*1.0 + 1*0.0) / 4 ) = sqrt(1.5/4)
    assert abs(got - (1.5 / 4.0) ** 0.5) < 1e-12


def test_attach_scan_baseline_merges_by_name_without_mutating():
    records = [
        {"molecule": "H", "E_total_nn": -0.5, "E_pbe": -0.5,
         "skip_reason": "atomic_system"},
        {"name": "H2O", "E_total_nn": -76.4, "E_pbe": -76.4,
         "density_rmse": 0.01, "density_rmse_pbe": 0.02},
    ]
    scan = {"H": {"E_scan": -0.49, "density_rmse_scan": None},
            "H2O": {"E_scan": -76.3, "density_rmse_scan": 0.015}}
    out = dfs_demo.attach_scan_baseline(records, scan)
    by = {r.get("name") or r.get("molecule"): r for r in out}
    assert by["H"]["E_scan"] == -0.49
    assert "density_rmse_scan" not in by["H"]        # None is not attached (atom)
    assert by["H2O"]["E_scan"] == -76.3
    assert by["H2O"]["density_rmse_scan"] == 0.015
    assert "E_scan" not in records[0]                # inputs untouched


def test_aggregate_density_diagnostics_emits_scan_when_present():
    records = [
        {"name": "H2O", "density_rmse": 0.010, "density_rmse_pbe": 0.020,
         "density_rmse_scan": 0.015},
        {"name": "HO", "density_rmse": 0.030, "density_rmse_pbe": 0.020,
         "density_rmse_scan": 0.025},
    ]
    rows = dfs_demo.aggregate_density_diagnostics(records)
    by = {r["name"]: r for r in rows}
    assert by["H2O"]["density_rmse_scan"] == 0.015
    assert by["H2O"]["beats_scan"] is True      # 0.010 < 0.015
    assert by["HO"]["beats_scan"] is False      # 0.030 !< 0.025


def test_self_consistent_ae_emits_scan_series():
    records = [
        {"molecule": "H", "E_total_nn": -0.5, "E_pbe": -0.5, "E_scan": -0.51,
         "skip_reason": "atomic_system"},
        {"molecule": "O", "E_total_nn": -75.0, "E_pbe": -75.0, "E_scan": -75.05,
         "skip_reason": "atomic_system"},
        {"molecule": "H2O", "E_total_nn": -76.4, "E_pbe": -76.4, "E_scan": -76.45},
    ]
    rows = dfs_demo.self_consistent_ae(records, {"H2O": {"H": 2, "O": 1}},
                                       {"H2O": 230.0})
    r = rows[0]
    # AE_scan = 2*(-0.51) + (-75.05) - (-76.45) = 0.38 Ha
    assert abs(r["ae_scan_kcal"] - 0.38 * _KCAL) < 1e-6
    assert abs(r["err_scan"] - (0.38 * _KCAL - 230.0)) < 1e-6
    assert "beats_scan" in r


def test_self_consistent_ae_omits_scan_when_atom_missing_e_scan():
    # A molecule whose constituent atom has no SCAN energy must not emit a scan
    # series (mirrors the NN own-atom rule), but the PBE/NN series still emit.
    records = [
        {"molecule": "H", "E_total_nn": -0.5, "E_pbe": -0.5, "E_scan": -0.51,
         "skip_reason": "atomic_system"},
        {"molecule": "O", "E_total_nn": -75.0, "E_pbe": -75.0,
         "skip_reason": "atomic_system"},   # no E_scan
        {"molecule": "H2O", "E_total_nn": -76.4, "E_pbe": -76.4, "E_scan": -76.45},
    ]
    rows = dfs_demo.self_consistent_ae(records, {"H2O": {"H": 2, "O": 1}},
                                       {"H2O": 230.0})
    assert rows and "err_nn" in rows[0]
    assert "err_scan" not in rows[0]


def test_combined_energy_density_emits_scan_series():
    ae_rows = [{"err_nn": 1.0, "err_pbe": 4.0, "err_scan": 2.0},
               {"err_nn": -2.0, "err_pbe": -4.0, "err_scan": -3.0}]
    density_rows = [
        {"density_rmse": 1e-4, "density_rmse_pbe": 2e-4, "density_rmse_scan": 1.5e-4},
        {"density_rmse": 3e-4, "density_rmse_pbe": 4e-4, "density_rmse_scan": 3.5e-4}]
    m = dfs_demo.combined_energy_density(ae_rows, density_rows)
    assert abs(m["E_MAE_scan"] - 2.5) < 1e-12         # mean(|2|, |-3|)
    assert abs(m["D_scan"] - 2.5e-4) < 1e-16          # mean(1.5e-4, 3.5e-4)
    # gamma is STILL self-calibrated from PBE (common scale), not from SCAN.
    assert abs(m["gamma"] - (4.0 / 3e-4)) < 1e-6
    gd_scan = (4.0 / 3e-4) * 2.5e-4
    assert abs(m["ED_scan"] - 2.0 / (1 / 2.5 + 1 / gd_scan)) < 1e-9
    assert m["beats_scan"] == (m["ED_nn"] < m["ED_scan"])


def test_combined_energy_density_no_scan_keys_when_absent():
    # Backward compatible: without a scan series the scan keys are absent.
    m = dfs_demo.combined_energy_density(
        [{"err_nn": 1.0, "err_pbe": 4.0}],
        [{"density_rmse": 1e-4, "density_rmse_pbe": 2e-4}])
    assert "ED_scan" not in m and "beats_scan" not in m


def test_heldout_summary_tallies_scan_when_present():
    combined = {
        "m1": {"E_MAE_nn": 1.0, "E_MAE_pbe": 2.0, "D_nn": 1e-4, "D_pbe": 2e-4,
               "ED_nn": 1.0, "ED_pbe": 2.0, "beats_pbe": True,
               "E_MAE_scan": 1.5, "D_scan": 1.5e-4, "ED_scan": 1.5,
               "beats_scan": True},
    }
    s = dfs_demo.heldout_summary(combined)
    assert s["n_beat_ae_scan"] == 1        # 1.0 < 1.5
    assert s["n_beat_density_scan"] == 1   # 1e-4 < 1.5e-4
    assert s["n_beat_ed_scan"] == 1
    assert s["rows"][0]["beats_ae_scan"] is True


def test_scan_baseline_runs_scf_energy_and_density(tmp_path):
    """scan_baseline runs a real SCAN KS-SCF per species: finite SCAN energies for
    a molecule AND an atom, and a finite density RMSE vs the reference grid for
    the molecule (atoms are skipped -> None)."""
    import numpy as np
    from pyscf import dft, gto
    from xcquinox.alec.config import MoleculeSpec

    h2 = MoleculeSpec(name="H2", atom="H 0.0 0.0 0.0; H 0.0 0.0 0.741",
                      basis="def2-svp", spin=0, charge=0,
                      atom_composition=(("H", 2),), grid_level=1)
    h = MoleculeSpec(name="H", atom="H 0.0 0.0 0.0", basis="def2-svp", spin=1,
                     charge=0, atom_composition=(("H", 1),), grid_level=1)

    # A reference grid + AO values for H2 (stands in for the eval mol_data): the
    # SCAN dm is grid-independent, so scoring it on this grid is well-defined.
    mol = gto.M(atom="H 0.0 0.0 0.0; H 0.0 0.0 0.741", basis="def2-svp",
                spin=0, charge=0, unit="angstrom", verbose=0)
    g = dft.gen_grid.Grids(mol); g.level = 1; g.build()
    ao = dft.numint.eval_ao(mol, g.coords, deriv=0)
    mol_data_by_name = {"H2": {"ao_grid": ao, "grid_weights": g.weights,
                               "rho_ref_grid": np.zeros(g.weights.shape)}}

    out = dfs_demo.scan_baseline(
        [h2, h], mol_data_by_name, refs_dir=str(tmp_path), basis="def2-svp",
        grid_level=1, orientation_lock_strength=0.0, progress=False)

    assert np.isfinite(out["H2"]["E_scan"]) and out["H2"]["E_scan"] < 0.0
    assert np.isfinite(out["H"]["E_scan"]) and out["H"]["E_scan"] < 0.0
    d = out["H2"]["density_rmse_scan"]
    assert isinstance(d, float) and np.isfinite(d) and d > 0.0
    assert out["H"]["density_rmse_scan"] is None       # atom: density skipped


def test_scan_baseline_skips_nonconverging_species(tmp_path, monkeypatch):
    """A species whose SCAN SCF raises RuntimeError (genuine non-convergence) is
    skipped with E_scan=None rather than crashing the whole baseline (the 'Li'
    failure mode). Downstream, a missing E_scan just drops that species' SCAN."""
    from xcquinox.alec.config import MoleculeSpec

    h = MoleculeSpec(name="H", atom="H 0.0 0.0 0.0", basis="def2-svp", spin=1,
                     charge=0, atom_composition=(("H", 1),), grid_level=1)

    def _boom(*a, **k):
        raise RuntimeError("SCAN SCF for 'H' did not converge after tiered escalation")

    monkeypatch.setattr(dfs_demo, "run_scf_with_cache", _boom)
    out = dfs_demo.scan_baseline(
        [h], {}, refs_dir=str(tmp_path), basis="def2-svp",
        grid_level=1, orientation_lock_strength=0.0, progress=False)
    assert out["H"] == {"E_scan": None, "density_rmse_scan": None}
