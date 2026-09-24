"""Tests for xcquinox.pipeline.training_points: mixed-pool TrainingPoint
abstraction for Dick 2021-style subset selection across AE + BH76 + IP13."""
import pytest


def test_build_dfs_pool_points_count_and_kinds():
    """The pool has exactly 26 points: 21 AE + 3 BH76 + 2 IP13."""
    from collections import Counter
    from xcquinox.pipeline.training_points import build_dfs_pool_points
    points = build_dfs_pool_points()
    assert len(points) == 26
    by_kind = Counter(p.kind for p in points)
    assert by_kind == {"ae": 21, "bh76": 3, "ip13": 2}


def test_ae_point_only_carries_dick_atom_anchors():
    """AE point species = (compound,) + atom anchors ONLY for Dick elements
    (H, Li) appearing in the compound. C, N, O, F, ... never get their own
    MoleculeSpec."""
    from xcquinox.pipeline.training_points import build_dfs_pool_points
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
    from xcquinox.pipeline.training_points import build_dfs_pool_points
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
    from xcquinox.pipeline.training_points import build_dfs_pool_points
    points = build_dfs_pool_points()
    by_name = {p.name: p for p in points}
    li = by_name["Li_IP"]
    assert [s.info["name"] for s in li.species] == ["Li", "Li+"]
    assert [int(s.info["charge"]) for s in li.species] == [0, 1]
    c = by_name["C_IP"]
    assert [s.info["name"] for s in c.species] == ["C", "C+"]


# ASE Hill formulas of the nine AE molecules the published reference
# trajectory marks non-self-consistent (NO2, NH, O3, N2O, CH3, CH2, H2O, NH3
# and O2, the last of which this pool does not hold), written here
# independently of the constant under test so the constant is compared
# against the trajectory's list and not against itself.
_PUBLISHED_NONSC_AE_HILLS = ("NO2", "HN", "O3", "N2O", "CH3", "CH2",
                             "H2O", "H3N", "O2")


def test_the_pool_records_the_published_self_consistency_split():
    """The per-entry self-consistency flag of the published protocol travels as
    data on the pool Atoms, on the reaction and IP dicts, and on the points.

    Oracle: the published reference trajectory, whose nine AE entries NO2, NH,
    O3, N2O, CH3, CH2, H2O, NH3 and O2 and whose every reaction species carry
    ``sc=False``, while the remaining AE molecules, the atom references and the
    IP entries carry no flag and are therefore self-consistent. O2 is not one
    of this pool's entries, so the stamping is compared over the constant's
    members the pool holds. The flag is a Python bool, so identity comparison
    is used throughout.

    Asserted: ``DFS_NONSC_AE_HILLS`` equals that list of Hill formulas;
    ``build_dfs_pool`` stamps ``info['sc']`` on every AE Atoms (False on
    exactly the eight the pool holds), ``'sc': False`` on every reaction dict,
    ``'sc': True`` on every IP dict and ``info['sc'] True`` on every atom
    reference; ``build_dfs_pool_points`` records ``metadata['sc']`` on every
    point (the compound's flag for an AE point, False for a reaction, True for
    an IP pair) and stamps ``info['sc']`` on every species Atoms, the reactants
    and products of a reaction carrying False and an AE point's atom anchors
    carrying True, in the reaction form of the AE points as well.
    """
    from xcquinox.pipeline.dfs_pool import DFS_NONSC_AE_HILLS, build_dfs_pool
    from xcquinox.pipeline.training_points import build_dfs_pool_points

    assert DFS_NONSC_AE_HILLS == _PUBLISHED_NONSC_AE_HILLS

    pool = build_dfs_pool()
    ae_by_hill = {a.info["dfs_hill"]: a for a in pool["ae_molecules"]}
    # the published entries outside this pool: O2 and nothing else
    assert set(DFS_NONSC_AE_HILLS) - set(ae_by_hill) == {"O2"}
    nonsc = set(DFS_NONSC_AE_HILLS) & set(ae_by_hill)
    assert len(nonsc) == 8
    assert {h for h, a in ae_by_hill.items() if a.info["sc"] is False} == nonsc
    assert all(a.info["sc"] is True
               for h, a in ae_by_hill.items() if h not in nonsc)
    for rxn in pool["bh76_reactions"]:
        assert rxn["sc"] is False, rxn["name"]
    for pair in pool["ip13_pairs"]:
        assert pair["sc"] is True, pair["name"]
    for atom in pool["atom_refs"]:
        assert atom.info["sc"] is True, atom.info["name"]

    points = build_dfs_pool_points()
    by_name_sc = {p.name: p for p in points}
    assert nonsc <= set(by_name_sc), sorted(nonsc - set(by_name_sc))
    for p in points:
        assert "sc" in p.metadata, p.name
        # Every species carries the key, whatever its value: an unstamped
        # species would read as self-consistent by default downstream.
        for s in p.species:
            assert "sc" in s.info, (p.name, s.info["name"])
    for p in (q for q in points if q.kind == "ae"):
        expected = p.name not in nonsc
        assert p.metadata["sc"] is expected, p.name
        compound = next(s for s in p.species if s.info["name"] == p.name)
        assert compound.info["sc"] is expected, p.name
        # An AE point's H / Li anchors are atom references, which the
        # trajectory leaves self-consistent.
        for anchor in (s for s in p.species if s.info["name"] != p.name):
            assert anchor.info["sc"] is True, (p.name, anchor.info["name"])
    for p in (q for q in points if q.kind == "bh76"):
        assert p.metadata["sc"] is False, p.name
        reacting = set(p.metadata["reactants"]) | set(p.metadata["products"])
        for s in (s for s in p.species if s.info["name"] in reacting):
            assert s.info["sc"] is False, (p.name, s.info["name"])
    for p in (q for q in points if q.kind == "ip13"):
        assert p.metadata["sc"] is True, p.name
        for s in p.species:
            assert s.info["sc"] is True, (p.name, s.info["name"])

    # The reaction-form AE points (ae_as_reactions) carry the compound's flag
    # as the fixed-anchor form does: a run in that mode keeps the split. The
    # compound Atoms carries it too; the atoms it atomizes into stay
    # self-consistent, as the trajectory's atom entries are.
    rxn_form = {p.name: p for p in build_dfs_pool_points(ae_as_reactions=True)}
    for hill in ae_by_hill:
        p = rxn_form[hill]
        expected = hill not in nonsc
        assert p.metadata["sc"] is expected, hill
        compound = next(s for s in p.species if s.info["name"] == hill)
        assert compound.info["sc"] is expected, hill
        for atom in (s for s in p.species if s.info["name"] != hill):
            assert atom.info["sc"] is True, (hill, atom.info["name"])


def test_species_union_dedupes_by_name_charge_spin():
    """When two chosen points share an atom (e.g. AE compound's H anchor
    + BH76 reaction's H reactant), spec.molecules has it once."""
    from xcquinox.pipeline.training_points import (
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


def test_training_point_validates_kind_and_species():
    """TrainingPoint.__post_init__ rejects malformed inputs."""
    from xcquinox.pipeline.training_points import TrainingPoint
    from ase import Atoms
    with pytest.raises(ValueError, match="kind must be"):
        TrainingPoint(kind="bogus", name="x", species=(Atoms("H", info={"name": "H"}),))
    with pytest.raises(ValueError, match="non-empty"):
        TrainingPoint(kind="ae", name="x", species=())
    with pytest.raises(ValueError, match="info\\['name'\\]"):
        TrainingPoint(kind="ae", name="x", species=(Atoms("H"),))


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


def test_bh76_barrier_refs_match_tracked_benchmark_json():
    """Each barrier_ref equals the forward-barrier reference of the SAME
    reaction in the tracked bh76_full_pool.json (GMTKN55 BH76/.res values),
    so training and held-out evaluation share one reference layer."""
    import json
    from pathlib import Path
    from xcquinox.pipeline.dfs_pool import DFS_BH76_REACTIONS
    json_path = (Path(__file__).parents[1] / "data" / "bh76_full_pool.json")
    with open(json_path) as f:
        rows = {r["name"]: r for r in json.load(f)["reactions"]}
    for rxn in DFS_BH76_REACTIONS:
        ts_name, row_name = _EXPECTED_TS[rxn["name"]]
        row = rows[row_name]
        assert row["products"] == [ts_name]
        assert rxn["barrier_ref"] == pytest.approx(
            row["reaction_energy_ref"], abs=1e-9)


def test_bh76_barrier_height_mode_builds_ts_points():
    """bh76_mode='barrier_height' builds reactants -> TS points: species are
    the reactants plus the staged transition state, coeffs (-1, ..., +1), and
    e_rxn_ref is the forward barrier, so sum(coeffs*E) = E(TS) - E(reactants)
    is a true forward barrier height."""
    from xcquinox.pipeline.training_points import build_dfs_pool_points

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


def test_ae_reaction_point_predicted_atom_form():
    """Reaction-form AE point: bh76-kind, SAME name as the fixed-anchor form
    (ledger compatibility), species = compound + one neutral atom per element,
    coeffs = (-1, n_Z...) so sum(coeffs*E) = AE."""
    from ase import Atoms
    from xcquinox.pipeline.training_points import (
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


