"""Unit tests for the PURE helpers in notebooks/analysis/multimode_constraint_eval.py
(the self-consistency-ladder evaluation driver). Loaded by file path. These touch
no SCF / pyscf compute — only the mode->SolverConfig mapping, the species grouping,
the divergence-robust metric reductions, and the seed aggregator."""
import importlib.util
import math
import os

import pytest

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "notebooks", "analysis",
    "multimode_constraint_eval.py",
)


@pytest.fixture(scope="module")
def mod():
    p = os.path.abspath(_PATH)
    if not os.path.isfile(p):
        pytest.skip(f"driver not found at {p}")
    spec = importlib.util.spec_from_file_location("multimode_constraint_eval", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --- solver_config_for_mode --------------------------------------------------

def test_mode_to_config_mapping(mod):
    from xcquinox.alec.solver import SolverMode, FeaturePolicy
    assert mod.solver_config_for_mode("fixed_rho") is None
    one = mod.solver_config_for_mode("one_shot")
    assert one.mode == SolverMode.FIXED_J and one.max_cycles == 1
    assert one.feature_policy == FeaturePolicy.FROZEN
    three = mod.solver_config_for_mode("3step")
    assert three.mode == SolverMode.FULL and three.max_cycles == 3
    assert three.feature_policy == FeaturePolicy.REASSEMBLE


def test_mode_unknown_raises(mod):
    with pytest.raises(ValueError):
        mod.solver_config_for_mode("nope")


# --- group_species -----------------------------------------------------------

def test_group_species_atoms_vs_molecules(mod):
    dev = {"H": 1.0, "C": 2.0, "Cl": 3.0, "h2o": 4.0, "co2": 5.0}
    # atom_element_by_name: only single-atom species appear; molecules omitted.
    atom_elem = {"H": "H", "C": "C", "Cl": "Cl"}
    g = mod.group_species(dev, atom_elem)
    assert set(g["pretrain_atoms"]) == {"H"}          # H is a pretrain atom
    assert set(g["other_atoms"]) == {"C", "Cl"}       # C, Cl are not pretrained
    assert set(g["molecules"]) == {"h2o", "co2"}      # absent from atom map
    # values preserved
    assert g["pretrain_atoms"]["H"] == 1.0 and g["molecules"]["co2"] == 5.0


# --- reaction_mae_robust -----------------------------------------------------

def test_reaction_mae_robust_skips_diverged(mod):
    # one reaction is exact; another touches a NaN species and must be skipped.
    energies = {"A": 1.0, "B": 2.0, "C": float("nan")}
    rxns = [
        {"reactants": ["A"], "products": ["B"], "coeffs": [-1, 1],
         "reaction_energy_ref": (2.0 - 1.0) * mod.KCAL_PER_HA},  # error 0
        {"reactants": ["A"], "products": ["C"], "coeffs": [-1, 1],
         "reaction_energy_ref": 0.0},                            # touches NaN -> skip
    ]
    assert mod.reaction_mae_robust(energies, rxns) == pytest.approx(0.0)


def test_reaction_mae_robust_all_diverged_is_nan(mod):
    energies = {"A": float("nan"), "B": 2.0}
    rxns = [{"reactants": ["A"], "products": ["B"], "coeffs": [-1, 1],
             "reaction_energy_ref": 0.0}]
    assert math.isnan(mod.reaction_mae_robust(energies, rxns))


# --- pbe_dev_mae_robust ------------------------------------------------------

def test_pbe_dev_mae_robust_finite_only(mod):
    energies = {"A": 1.0, "B": float("nan")}
    e_pbe = {"A": 1.0 + 1.0 / mod.KCAL_PER_HA, "B": 0.0}  # A off by exactly 1 kcal/mol
    # B is NaN -> excluded; mean over {A} = 1.0 kcal/mol
    assert mod.pbe_dev_mae_robust(energies, e_pbe) == pytest.approx(1.0, abs=1e-9)
    assert math.isnan(mod.pbe_dev_mae_robust({"B": float("nan")}, {"B": 0.0}))


# --- aggregate_seed_metrics --------------------------------------------------

def test_aggregate_seed_metrics_divergence_and_stats(mod):
    per_seed = [
        {"m": 10.0}, {"m": 20.0}, {"m": float("nan")}, {"m": 30.0},
    ]
    agg = mod.aggregate_seed_metrics(per_seed, ("m",))["m"]
    assert agg["n_total"] == 4 and agg["n_used"] == 3
    assert agg["divergence_rate"] == pytest.approx(0.25)
    assert agg["mean"] == pytest.approx(20.0)
    assert agg["worst"] == pytest.approx(30.0)
    assert agg["std"] == pytest.approx(__import__("numpy").std([10.0, 20.0, 30.0]))


def test_aggregate_all_diverged_is_nan(mod):
    agg = mod.aggregate_seed_metrics([{"m": float("nan")}], ("m",))["m"]
    assert agg["n_used"] == 0 and agg["divergence_rate"] == pytest.approx(1.0)
    assert math.isnan(agg["mean"]) and math.isnan(agg["worst"])


# --- steps_to_converge -------------------------------------------------------

def test_steps_to_converge_basic(mod):
    # monotonically decreasing; min=1.0, frac=1.05 -> threshold 1.05; first loss
    # <= 1.05 is the 10.0? no: [10,5,2,1.04,1.0] -> 1.04 (index 3, 1-based 4).
    traj = [10.0, 5.0, 2.0, 1.04, 1.0]
    assert mod.steps_to_converge(traj, frac=1.05) == 4
    # frac=1.0 -> only the exact min qualifies -> index 5 (1-based)
    assert mod.steps_to_converge(traj, frac=1.0) == 5


def test_steps_to_converge_nan_filtered_and_empty(mod):
    assert mod.steps_to_converge([float("nan"), 3.0, 2.0], frac=1.0) == 2
    assert math.isnan(mod.steps_to_converge([]))
    assert math.isnan(mod.steps_to_converge([float("nan")]))


# --- config-aware arch builder (polarized flag) ------------------------------

def test_build_arch_polarized_flag(mod):
    import types
    from xcquinox.alec.config import ArchitectureConfig
    demo_stub = types.SimpleNamespace(
        ArchitectureConfig=ArchitectureConfig, DEPTH=2, NODES=8)
    a_pol = mod._build_arch(demo_stub, "unconstrained", (), (), True)
    a_unp = mod._build_arch(demo_stub, "unconstrained", (), (), False)
    assert a_pol.use_polarized_correlation is True
    assert a_unp.use_polarized_correlation is False


def test_should_reuse_checkpoint_guards_steps_and_weighting(mod):
    # match → reuse
    assert mod.should_reuse_checkpoint(
        {"pretrain_steps": 1000, "loss_weighting": "unweighted"},
        1000, "unweighted") is True
    # mismatched step count (e.g. a 20-step smoke ckpt) → do NOT reuse
    assert mod.should_reuse_checkpoint(
        {"pretrain_steps": 20, "loss_weighting": "unweighted"},
        1000, "unweighted") is False
    # mismatched weighting → do NOT reuse
    assert mod.should_reuse_checkpoint(
        {"pretrain_steps": 1000, "loss_weighting": "integration"},
        1000, "unweighted") is False
    # missing keys → do NOT reuse
    assert mod.should_reuse_checkpoint({}, 1000, "unweighted") is False
