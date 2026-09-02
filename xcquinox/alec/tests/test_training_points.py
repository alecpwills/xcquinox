"""Tests for xcquinox.alec.training_points: mixed-pool TrainingPoint
abstraction for Dick 2021-style subset selection across AE + BH76 + IP13."""
import pytest


def test_build_dfs_pool_points_count_and_kinds():
    """The pool has exactly 26 points: 21 AE + 3 BH76 + 2 IP13."""
    from collections import Counter
    from xcquinox.alec.training_points import build_dfs_pool_points
    points = build_dfs_pool_points()
    assert len(points) == 26
    by_kind = Counter(p.kind for p in points)
    assert by_kind == {"ae": 21, "bh76": 3, "ip13": 2}


def test_dick_atom_regularizer_set_is_h_and_li():
    """DICK_ATOM_REGULARIZER_SYMS exactly mirrors DFS_ATOM_REFS."""
    from xcquinox.alec.training_points import DICK_ATOM_REGULARIZER_SYMS
    from xcquinox.alec.dfs_pool import DFS_ATOM_REFS
    assert DICK_ATOM_REGULARIZER_SYMS == ("H", "Li")
    assert {r["sym"] for r in DFS_ATOM_REFS} == set(DICK_ATOM_REGULARIZER_SYMS)


def test_ae_point_only_carries_dick_atom_anchors():
    """AE point species = (compound,) + atom anchors ONLY for Dick elements
    (H, Li) appearing in the compound. C, N, O, F, ... never get their own
    MoleculeSpec."""
    from xcquinox.alec.training_points import build_dfs_pool_points
    points = build_dfs_pool_points()
    by_name = {p.name: p for p in points}
    # CHN contains C, H, N; Dick anchor is H only:
    chn = by_name["CHN"]
    sp_names = [s.info["name"] for s in chn.species]
    assert sp_names == ["CHN", "H"]
    # CO contains C, O; no Dick anchors -> only the compound:
    co = by_name["CO"]
    assert [s.info["name"] for s in co.species] == ["CO"]
    # HLi contains H + Li, both Dick anchors:
    hli = by_name["HLi"]
    assert sorted(s.info["name"] for s in hli.species) == ["H", "HLi", "Li"]


def test_bh76_point_carries_full_reaction_plus_dick_anchors_only():
    """BH76 point species = (all reactants + products, deduped) +
    atom anchors ONLY for Dick elements not already in the reaction."""
    from xcquinox.alec.training_points import build_dfs_pool_points
    points = build_dfs_pool_points()
    by_name = {p.name: p for p in points}
    # OH+N2 -> H+N2O: H is a product -> no extra H anchor; no N or O anchors.
    p = by_name["OH+N2_to_H+N2O"]
    assert sorted(s.info["name"] for s in p.species) == [
        "H", "HO", "N2", "N2O",
    ]
    # OH+CH3 -> O+CH4: H is NOT a reactant or product -> H anchor added.
    p2 = by_name["OH+CH3_to_O+CH4"]
    assert sorted(s.info["name"] for s in p2.species) == [
        "CH3", "CH4", "H", "HO", "O",
    ]


def test_ip13_point_carries_neutral_and_cation_only():
    """IP13 point species = (neutral, cation), no extra atom anchors."""
    from xcquinox.alec.training_points import build_dfs_pool_points
    points = build_dfs_pool_points()
    by_name = {p.name: p for p in points}
    li = by_name["Li_IP"]
    assert [s.info["name"] for s in li.species] == ["Li", "Li+"]
    assert [int(s.info["charge"]) for s in li.species] == [0, 1]
    c = by_name["C_IP"]
    assert [s.info["name"] for s in c.species] == ["C", "C+"]


def test_species_union_dedupes_by_name_charge_spin():
    """When two chosen points share an atom (e.g. AE compound's H anchor
    + BH76 reaction's H reactant), spec.molecules has it once."""
    from xcquinox.alec.training_points import (
        build_dfs_pool_points, species_union_from_points,
    )
    points = build_dfs_pool_points()
    by_name = {p.name: p for p in points}
    chosen = [by_name["CHN"], by_name["OH+N2_to_H+N2O"]]
    sp = species_union_from_points(chosen)
    names = [s.info["name"] for s in sp]
    # H appears in CHN (anchor) and as a product of OH+N2 -> only once:
    assert names.count("H") == 1
    # CHN compound + 4 BH76 species + 1 H anchor (deduped) = 5 total.
    assert sorted(names) == ["CHN", "H", "HO", "N2", "N2O"]


def test_r2_mixed_subset_matches_user_example():
    """User example: 'r=2 = a CH4 AE (if in pool) + a C -> C+ IP'.
    CH4 isn't in DFS AE pool (only BH76), so use the closest AE compound
    (CH3) instead. Verifies the API supports mixed-kind subset selection."""
    from xcquinox.alec.training_points import (
        build_dfs_pool_points, species_union_from_points,
    )
    points = build_dfs_pool_points()
    by_name = {p.name: p for p in points}
    chosen = [by_name["CH3"], by_name["C_IP"]]
    sp = species_union_from_points(chosen)
    names = sorted(s.info["name"] for s in sp)
    # CH3 (compound) + H (Dick anchor for CH3) + C (neutral IP) + C+ (cation):
    assert names == ["C", "C+", "CH3", "H"]


def test_ip13_neutral_serves_dual_role_when_dick_regularized():
    """For Li_IP, the neutral Li IS already a single-atom MoleculeSpec
    (charge=0). When this point is chosen alongside e.g. an FLi AE
    point (which adds its own Li anchor), the spec dedupes to one Li."""
    from xcquinox.alec.training_points import (
        build_dfs_pool_points, species_union_from_points,
    )
    points = build_dfs_pool_points()
    by_name = {p.name: p for p in points}
    chosen = [by_name["FLi"], by_name["Li_IP"]]
    sp = species_union_from_points(chosen)
    names = sorted(s.info["name"] for s in sp)
    assert names == ["FLi", "Li", "Li+"]
    li_entries = [s for s in sp if s.info["name"] == "Li"]
    assert len(li_entries) == 1
    assert int(li_entries[0].info["charge"]) == 0
    assert int(li_entries[0].info["spin"]) == 1


def test_training_point_validates_kind_and_species():
    """TrainingPoint.__post_init__ rejects malformed inputs."""
    from xcquinox.alec.training_points import TrainingPoint
    from ase import Atoms
    with pytest.raises(ValueError, match="kind must be"):
        TrainingPoint(kind="bogus", name="x", species=(Atoms("H", info={"name": "H"}),))
    with pytest.raises(ValueError, match="non-empty"):
        TrainingPoint(kind="ae", name="x", species=())
    with pytest.raises(ValueError, match="info\\['name'\\]"):
        TrainingPoint(kind="ae", name="x", species=(Atoms("H"),))


def test_training_point_metadata_preserved():
    """AE / BH76 / IP13 metadata round-trips correctly."""
    from xcquinox.alec.training_points import build_dfs_pool_points
    points = build_dfs_pool_points()
    by_name = {p.name: p for p in points}
    # AE
    chn = by_name["CHN"]
    assert chn.metadata.get("ae_kcalmol") is not None
    assert chn.metadata["ae_kcalmol"] != 0.0
    # BH76, default bh76_mode is 'reaction_energy', so e_rxn_ref is
    # the true reaction energy ΔE (GMTKN55-BH76RC), not the barrier.
    bh = by_name["OH+N2_to_H+N2O"]
    assert bh.metadata["e_rxn_ref"] == 64.91
    assert bh.metadata["bh76_mode"] == "reaction_energy"
    # GMTKN55 BH76/.res 'oh n2 n2ohts' forward barrier.
    assert bh.metadata["barrier_ref"] == 82.6
    assert bh.metadata["reaction_energy_ref"] == 64.91
    assert bh.metadata["reactants"] == ("HO", "N2")
    assert bh.metadata["products"] == ("H", "N2O")
    # IP13 (ip_ref may be None if not set; just check shape)
    li = by_name["Li_IP"]
    assert li.metadata["neutral"] == "Li"
    assert li.metadata["cation"] == "Li+"


# --- bh76_mode toggle (reaction_energy default vs barrier_height) -----------

# Expected true reaction energies ΔE for the reactant -> product direction of
# each DFS_BH76_REACTIONS entry, taken DIRECTLY from GMTKN55-BH76RC (W2-F12;
# grimme-lab/GMTKN55 BH76/.resRC). Realigned 2026-05-24 from the prior
# Minnesota Vr−Vf values (65.14/−5.57/103.53), which differ by ~0.2 kcal/mol.
_EXPECTED_REACTION_ENERGIES = {
    "OH+N2_to_H+N2O":  64.91,
    "OH+CH3_to_O+CH4": -5.44,
    "HF+F_to_H+F2":    103.28,
}
# Forward barrier heights for the barrier_height mode: the GMTKN55-BH76
# values (BH76/.res: 'oh n2 n2ohts' = 82.6, 'oh ch3 RKT11' = 8.9,
# 'hf f hf2ts' = 104.8 kcal/mol), the same reference layer the held-out
# BH76 evaluation scores against. The Minnesota REF1 barriers (82.27,
# 7.90, 105.80) differ at the sub-kcal/mol level and are retained in the
# dfs_pool provenance comments only.
_EXPECTED_BARRIERS = {
    "OH+N2_to_H+N2O":  82.6,
    "OH+CH3_to_O+CH4": 8.9,
    "HF+F_to_H+F2":    104.8,
}
# The transition state each reaction climbs through, and the tracked
# bh76_full_pool.json reaction row carrying the same forward barrier.
_EXPECTED_TS = {
    "OH+N2_to_H+N2O":  ("n2ohts", "bh76_oh_n2_to_n2ohts"),
    "OH+CH3_to_O+CH4": ("RKT11", "bh76_oh_ch3_to_RKT11"),
    "HF+F_to_H+F2":    ("hf2ts", "bh76_hf_f_to_hf2ts"),
}


def test_bh76_default_mode_is_reaction_energy():
    """The default bh76_mode yields BH76 points whose e_rxn_ref is the
    true GMTKN55-BH76RC reaction energy ΔE (+64.91 / −5.44 / +103.28 kcal/mol),
    not the barrier height. _rxn_residual_term computes
    Σ coeffs·E = E(products) − E(reactants), a reaction energy."""
    from xcquinox.alec.training_points import build_dfs_pool_points
    # Explicit default and implicit default must agree.
    for points in (build_dfs_pool_points(),
                    build_dfs_pool_points(bh76_mode="reaction_energy")):
        by_name = {p.name: p for p in points if p.kind == "bh76"}
        assert set(by_name) == set(_EXPECTED_REACTION_ENERGIES)
        for name, expected in _EXPECTED_REACTION_ENERGIES.items():
            tp = by_name[name]
            assert tp.metadata["e_rxn_ref"] == pytest.approx(expected, abs=1e-9)
            assert tp.metadata["bh76_mode"] == "reaction_energy"
            # The reactant/product structure is preserved unchanged.
            assert len(tp.metadata["reactants"]) == 2
            assert len(tp.metadata["products"]) == 2
            assert tp.metadata["coeffs"] == (-1.0, -1.0, +1.0, +1.0)


def test_bh76_reaction_energy_species_are_reactants_and_products():
    """reaction_energy-mode BH76 points carry reactant/product species
    (no transition-state species)."""
    from xcquinox.alec.training_points import build_dfs_pool_points
    points = build_dfs_pool_points(bh76_mode="reaction_energy")
    by_name = {p.name: p for p in points}
    p = by_name["OH+N2_to_H+N2O"]
    assert sorted(s.info["name"] for s in p.species) == ["H", "HO", "N2", "N2O"]


def test_bh76_reaction_energy_consistent_with_vr_minus_vf():
    """Internal consistency of the GMTKN55 reference layer: the BH76RC
    reaction energy equals forward minus reverse barrier of the same TS, both
    read from the tracked bh76_full_pool.json (BH76/.res values). One W2-F12
    layer, so agreement is ~0.1 kcal/mol (82.6-17.7=64.9 vs 64.91;
    8.9-14.4=-5.5 vs -5.44; 104.8-1.5=103.3 vs 103.28)."""
    import json
    from pathlib import Path
    json_path = (Path(__file__).parents[1] / "data" / "bh76_full_pool.json")
    with open(json_path) as f:
        rows = {r["name"]: r for r in json.load(f)["reactions"]}
    reverse_row = {"OH+N2_to_H+N2O": "bh76_h_n2o_to_n2ohts",
                   "OH+CH3_to_O+CH4": "bh76_O_CH4_to_RKT11",
                   "HF+F_to_H+F2": "bh76_h_f2_to_hf2ts"}
    for name, re_gmtkn55 in _EXPECTED_REACTION_ENERGIES.items():
        fwd = _EXPECTED_BARRIERS[name]
        rev = rows[reverse_row[name]]["reaction_energy_ref"]
        assert abs((fwd - rev) - re_gmtkn55) < 0.1, (
            f"{name}: GMTKN55 dE {re_gmtkn55} vs fwd-rev {fwd - rev} "
            f"disagree beyond rounding within one reference layer")


def test_bh76_reactions_carry_both_reference_values():
    """Each DFS_BH76_REACTIONS entry carries BOTH the barrier and the
    reaction energy, plus a (default-None) ts_species slot."""
    from xcquinox.alec.dfs_pool import DFS_BH76_REACTIONS
    for rxn in DFS_BH76_REACTIONS:
        name = rxn["name"]
        assert rxn["barrier_ref"] == pytest.approx(
            _EXPECTED_BARRIERS[name], abs=0.01)
        assert rxn["reaction_energy_ref"] == pytest.approx(
            _EXPECTED_REACTION_ENERGIES[name], abs=0.01)
        # The legacy e_rxn_ref alias has been removed from the source
        # dict: it held the barrier height under a name the metadata
        # uses for the mode-selected value, which is misleading.
        assert "e_rxn_ref" not in rxn
        # TS slot names the staged transition state (bh76_full_pool.json).
        assert rxn["ts_species"] == _EXPECTED_TS[name][0]


def test_bh76_barrier_refs_match_tracked_benchmark_json():
    """Each barrier_ref equals the forward-barrier reference of the SAME
    reaction in the tracked bh76_full_pool.json (GMTKN55 BH76/.res values),
    so training and held-out evaluation share one reference layer."""
    import json
    from pathlib import Path
    from xcquinox.alec.dfs_pool import DFS_BH76_REACTIONS
    json_path = (Path(__file__).parents[1] / "data" / "bh76_full_pool.json")
    with open(json_path) as f:
        rows = {r["name"]: r for r in json.load(f)["reactions"]}
    for rxn in DFS_BH76_REACTIONS:
        ts_name, row_name = _EXPECTED_TS[rxn["name"]]
        row = rows[row_name]
        assert row["products"] == [ts_name]
        assert rxn["barrier_ref"] == pytest.approx(
            row["reaction_energy_ref"], abs=1e-9)


def test_bh76_unknown_mode_raises():
    """An unrecognized bh76_mode raises ValueError."""
    from xcquinox.alec.training_points import build_dfs_pool_points
    with pytest.raises(ValueError, match="Unknown bh76_mode"):
        build_dfs_pool_points(bh76_mode="not_a_mode")


def test_bh76_barrier_height_mode_builds_ts_points():
    """bh76_mode='barrier_height' builds reactants -> TS points: species are
    the reactants plus the staged transition state, coeffs (-1, ..., +1), and
    e_rxn_ref is the forward barrier, so sum(coeffs*E) = E(TS) - E(reactants)
    is a true forward barrier height."""
    from xcquinox.alec.training_points import build_dfs_pool_points

    points = build_dfs_pool_points(bh76_mode="barrier_height")
    bh76 = [p for p in points if p.kind == "bh76"
            and p.name in _EXPECTED_BARRIERS]
    assert len(bh76) == 3
    ts_spin = {"n2ohts": 1, "RKT11": 2, "hf2ts": 1}
    for p in bh76:
        ts_name = _EXPECTED_TS[p.name][0]
        md = p.metadata
        assert md["bh76_mode"] == "barrier_height"
        assert md["e_rxn_ref"] == pytest.approx(
            _EXPECTED_BARRIERS[p.name], abs=1e-9)
        # Stoichiometry: every reactant at -1, the TS at +1, no products.
        assert md["products"] == (ts_name,)
        n_react = len(md["reactants"])
        assert md["coeffs"] == (-1.0,) * n_react + (1.0,)
        # The TS species itself is staged with the benchmark identity.
        ts = next(a for a in p.species if a.info.get("name") == ts_name)
        assert ts.info["spin"] == ts_spin[ts_name]
        assert ts.info["charge"] == 0
        assert len(ts) >= 3  # a real polyatomic geometry, not an atom stub


def test_bh76_barrier_mode_missing_ts_raises():
    """A reaction dict without a staged TS still refuses barrier mode with
    an error naming the reaction (no silent reaction-energy fallback)."""
    from xcquinox.alec.training_points import _bh76_point_from_dict
    rxn = {
        "name": "synthetic_no_ts",
        "reactants": ["H2"], "products": ["H"],
        "coeffs": [-1.0, 2.0],
        "barrier_ref": 1.0, "reaction_energy_ref": 2.0,
        "ts_species": None,
        "species_spins": {"H2": 0, "H": 1},
        "species_charges": {"H2": 0, "H": 0},
    }
    with pytest.raises(ValueError, match="synthetic_no_ts"):
        _bh76_point_from_dict(rxn, atoms_by_name={},
                              bh76_mode="barrier_height")


def test_ae_reaction_point_predicted_atom_form():
    """Reaction-form AE point: bh76-kind, SAME name as the fixed-anchor form
    (ledger compatibility), species = compound + one neutral atom per element,
    coeffs = (-1, n_Z...) so sum(coeffs*E) = AE."""
    from ase import Atoms
    from xcquinox.alec.training_points import (
        _ae_point_from_atoms, _ae_reaction_point_from_atoms,
    )
    h2o = Atoms("OH2", positions=[(0, 0, 0), (0, 0.76, 0.59),
                                  (0, -0.76, 0.59)])
    h2o.info.update(dfs_hill="H2O", ae_kcalmol=232.2, ae_source="test")
    p = _ae_reaction_point_from_atoms(h2o)
    assert p.kind == "bh76"
    assert p.name == _ae_point_from_atoms(h2o).name == "H2O"
    assert {s.info["name"] for s in p.species} == {"H2O", "H", "O"}
    md = p.metadata
    assert md["reactants"] == ("H2O",)
    assert md["products"] == ("H", "O")
    assert md["coeffs"] == (-1.0, 2.0, 1.0)
    assert md["e_rxn_ref"] == pytest.approx(232.2)     # kcal/mol (loss converts)
    assert md["ae_form"] == "predicted_atom_reaction"
    # atom species are neutral with NIST ground-state spins
    atoms = {s.info["name"]: s for s in p.species if s.info["name"] != "H2O"}
    assert all(a.info.get("charge", 0) == 0 for a in atoms.values())
    assert atoms["O"].info["spin"] == 2 and atoms["H"].info["spin"] == 1
    # homonuclear: one atom species, multiplicity in the coeff
    na2 = Atoms("Na2", positions=[(0, 0, 0), (0, 0, 3.08)])
    na2.info.update(dfs_hill="Na2", ae_kcalmol=17.0)
    p2 = _ae_reaction_point_from_atoms(na2)
    assert {s.info["name"] for s in p2.species} == {"Na2", "Na"}
    assert p2.metadata["coeffs"] == (-1.0, 2.0)


def test_build_dfs_pool_points_ae_as_reactions_names_unchanged():
    """ae_as_reactions=True keeps EVERY point name identical (the JSD subset
    ledgers resolve by name) while converting the 21 AE points to bh76-kind
    atomization reactions; BH76/IP13 points are untouched."""
    from xcquinox.alec.training_points import build_dfs_pool_points
    base = build_dfs_pool_points()
    rxn = build_dfs_pool_points(ae_as_reactions=True)
    assert [p.name for p in base] == [p.name for p in rxn]
    assert len(rxn) == 26
    kinds = [p.kind for p in rxn]
    assert kinds.count("bh76") == 24 and kinds.count("ip13") == 2
    assert "ae" not in kinds
    # references carry over exactly (kcal/mol, same value as the ae form)
    by_name_base = {p.name: p for p in base}
    for p in rxn:
        if p.metadata.get("ae_form") == "predicted_atom_reaction":
            assert (p.metadata["e_rxn_ref"]
                    == by_name_base[p.name].metadata["ae_kcalmol"])
