"""Unit tests for dfs_demo.py -- assert the demo spec is DFS-exact.

These exercise only the pure spec-assembly / pool-selection / aggregation logic
(no PySCF SCF, no CCSD generation, no training), so they run in seconds. The
end-to-end density training is exercised separately by the SMOKE notebook run.
"""
import os
import sys
import tempfile


sys.path.insert(0, os.path.dirname(__file__))

import dfs_demo  # noqa: E402
from xcquinox.pipeline.solver import SolverMode  # noqa: E402
from xcquinox.pipeline.cluster.domain import KCAL_PER_HA  # noqa: E402


# ---------------------------------------------------------------------------
# Pool selection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Molecule specs (species union) + spins
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Solvers + architecture
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Pretrain-atom derivation (must never emit He, which is absent at the paper basis)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PBE pretraining wiring (runs a small PBE SCF + pretrain regression)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Self-consistent atomization energy + combined energy-density metric
# ---------------------------------------------------------------------------

_KCAL = dfs_demo._HARTREE_TO_KCAL


# Literature electronic atomization energies (De, kcal/mol) for the demo set; the
# corrected self-consistent AE must beat PBE against these on every trained model.
_LIT_AE_KCAL = {"HLi": 57.8, "HO": 106.4, "HN": 82.8, "H2O": 232.2}
_DEMO_COMP = {"HLi": {"H": 1, "Li": 1}, "HO": {"H": 1, "O": 1},
              "HN": {"H": 1, "N": 1}, "H2O": {"H": 2, "O": 1}}
_RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


# ---------------------------------------------------------------------------
# Orientation lock threaded through the demo
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Held-out generalization set (N2 + NO + NO2)
# ---------------------------------------------------------------------------


_HELDOUT_RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs", "heldout")


# ---------------------------------------------------------------------------
# SCAN self-consistent baseline (a meta-GGA comparator alongside PBE). The demo
# asks whether a trained meta-GGA net improves on SCAN itself at the CCSD density
# + atomization energy. The aggregation fns gain a *_scan series parallel to PBE.
# ---------------------------------------------------------------------------


