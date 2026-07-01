"""Unit tests for dfs_demo.py -- assert the demo spec is DFS-exact.

These exercise only the pure spec-assembly / pool-selection / aggregation logic
(no PySCF SCF, no CCSD generation, no training), so they run in seconds. The
end-to-end density training is exercised separately by the SMOKE notebook run.
"""
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
