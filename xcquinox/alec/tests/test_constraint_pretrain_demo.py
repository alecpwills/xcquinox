"""Fast unit tests for the GMTKN55 constraint+pretraining demo script.

The demo lives under ``notebooks/analysis/`` (a standalone script, not a package),
so it is loaded by file path via importlib. These tests exercise the PURE logic
(metric aggregation, constraint-level construction, the W4-11 parser) with no SCF
and no pretraining — the full end-to-end run is exercised by running the script.
"""
import importlib.util
import os

import pytest

_DEMO_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "notebooks", "analysis",
    "constraint_pretrain_gmtkn55_demo.py",
)


@pytest.fixture(scope="module")
def demo():
    path = os.path.abspath(_DEMO_PATH)
    if not os.path.isfile(path):
        pytest.skip(f"demo script not found at {path}")
    spec = importlib.util.spec_from_file_location("constraint_pretrain_demo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# reaction_energy_mae — pure aggregation (Ha -> kcal/mol, coeff ordering).
# Used for BOTH BH76 reaction energies and W4-11 atomization energies.
# ---------------------------------------------------------------------------

def test_reaction_energy_mae_single_reaction(demo):
    rxn = {
        "reactants": ["HO", "H2"], "products": ["H2O", "H"],
        "coeffs": [-1.0, -1.0, 1.0, 1.0], "reaction_energy_ref": -16.39,
    }
    dE_ref_ha = -16.39 / demo.KCAL_PER_HA
    energies = {"HO": -75.0, "H2": -1.1, "H": -0.5}
    e_others = energies["H"] - energies["HO"] - energies["H2"]
    energies["H2O"] = dE_ref_ha - e_others  # make the reaction energy hit the ref
    assert demo.reaction_energy_mae(energies, [rxn]) == pytest.approx(0.0, abs=1e-9)


def test_reaction_energy_mae_known_offset(demo):
    rxn = {"reactants": ["A"], "products": ["B"], "coeffs": [-1.0, 1.0],
           "reaction_energy_ref": 0.0}
    energies = {"A": -10.0, "B": -9.0}  # dE = +1 Ha
    assert demo.reaction_energy_mae(energies, [rxn]) == pytest.approx(
        demo.KCAL_PER_HA, rel=1e-9)


def test_reaction_energy_mae_averages_over_reactions(demo):
    rxns = [
        {"reactants": ["A"], "products": ["B"], "coeffs": [-1.0, 1.0],
         "reaction_energy_ref": 0.0},
        {"reactants": ["A"], "products": ["C"], "coeffs": [-1.0, 1.0],
         "reaction_energy_ref": 0.0},
    ]
    energies = {"A": -10.0, "B": -10.0, "C": -9.0}
    assert demo.reaction_energy_mae(energies, rxns) == pytest.approx(
        demo.KCAL_PER_HA / 2.0, rel=1e-9)


def test_w411_atomization_uses_reaction_scorer(demo):
    # W4-11 H2 atomization: 2*E(H) - E(H2) = 109.493 kcal/mol.
    ref = 109.493
    rxn = {"reactants": ["h2", "h"], "products": [], "coeffs": [-1, 2],
           "reaction_energy_ref": ref}
    e_h = -0.5
    e_h2 = 2 * e_h - ref / demo.KCAL_PER_HA  # so 2*E(H) - E(H2) == ref exactly
    energies = {"h2": e_h2, "h": e_h}
    assert demo.reaction_energy_mae(energies, [rxn]) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# pbe_total_energy_dev_mae — per-species deviation from PBE (kcal/mol).
# ---------------------------------------------------------------------------

def test_pbe_total_energy_dev_mae(demo):
    # mol_data carries E_pbe; energies are the NN totals. Mean |E_nn - E_pbe|*KCAL.
    mol_data = {"A": {"E_pbe": -10.0}, "B": {"E_pbe": -20.0}}
    energies = {"A": -10.0 + 1.0, "B": -20.0}  # devs: 1 Ha, 0 Ha
    got = demo.pbe_total_energy_dev_mae(energies, mol_data)
    assert got == pytest.approx(demo.KCAL_PER_HA / 2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# constraint progression + model construction
# ---------------------------------------------------------------------------

def test_constraint_levels_start_unconstrained_and_increase(demo):
    levels = demo.make_constraint_levels()
    assert len(levels) == 4
    # First level is the truly-unconstrained baseline (spec is None).
    assert levels[0][1] is None
    counts = [0] + [len(xc) + len(cc) for _, spec in levels[1:] for (xc, cc) in [spec]]
    assert counts == sorted(counts)
    assert counts[-1] > counts[1]  # full stack has more constraints than +LO


def test_build_random_model_unconstrained_is_truly_raw(demo):
    # spec is None -> built-in Lieb-Oxford squash disabled, no constraint objects.
    raw = demo.build_random_model(None, seed=0)
    assert raw.xnet.lob_lim is None
    assert raw.cnet.lob_lim is None
    assert tuple(raw.x_constraints) == () and tuple(raw.c_constraints) == ()
    # A constrained level carries the constraint and (for LO) disables built-in lob.
    _, spec = demo.make_constraint_levels()[1]  # +LO(x)
    con = demo.build_random_model(spec, seed=0)
    assert [c.registry_name for c in con.x_constraints] == ["lieb_oxford"]
    assert con.xnet.lob_lim is None


def test_build_arch_activates_named_constraints(demo):
    _, spec3 = demo.make_constraint_levels()[3]  # +LO+UEG+NNc
    xc3, cc3 = spec3
    arch3 = demo.build_arch("lvl", xc3, cc3)
    assert {"lieb_oxford", "ueg_limit"} <= {s.name for s in arch3.x_constraints}
    assert "non_negative_correlation" in {s.name for s in arch3.c_constraints}
    assert " " not in arch3.name and ":" not in arch3.name


# ---------------------------------------------------------------------------
# W4-11 parser (skipped when the GMTKN55 clone is absent).
# ---------------------------------------------------------------------------

def test_build_w411_ae_pool_parses_real_refs(demo):
    if not os.path.isfile(os.path.join(demo.W411_DIR, ".res")):
        pytest.skip("GMTKN55 W4-11 clone not present")
    mol_specs, reactions = demo.build_w411_ae_pool()
    assert reactions and mol_specs
    by_name = {r["name"]: r for r in reactions}
    # H2 atomization is the canonical W4-11 entry: 2*E(H) - E(H2) = 109.493 kcal/mol.
    assert "AE_h2" in by_name
    h2 = by_name["AE_h2"]
    assert h2["reactants"] == ["h2", "h"]
    assert h2["coeffs"] == [-1, 2]
    assert h2["reaction_energy_ref"] == pytest.approx(109.493, abs=1e-3)
    # Every species referenced by a reaction has a MoleculeSpec built.
    needed = set()
    for r in reactions:
        needed.update(r["reactants"]); needed.update(r["products"])
    assert needed <= set(mol_specs)
