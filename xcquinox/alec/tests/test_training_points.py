"""Tests for xcquinox.alec.training_points — mixed-pool TrainingPoint
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
    # CO contains C, O; no Dick anchors → only the compound:
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
    # OH+N2 → H+N2O: H is a product → no extra H anchor; no N or O anchors.
    p = by_name["OH+N2_to_H+N2O"]
    assert sorted(s.info["name"] for s in p.species) == [
        "H", "HO", "N2", "N2O",
    ]
    # OH+CH3 → O+CH4: H is NOT a reactant or product → H anchor added.
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
    # H appears in CHN (anchor) and as a product of OH+N2 → only once:
    assert names.count("H") == 1
    # CHN compound + 4 BH76 species + 1 H anchor (deduped) = 5 total.
    assert sorted(names) == ["CHN", "H", "HO", "N2", "N2O"]


def test_r2_mixed_subset_matches_user_example():
    """User example: 'r=2 = a CH4 AE (if in pool) + a C → C+ IP'.
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
    # BH76 — default bh76_mode is 'reaction_energy', so e_rxn_ref is
    # the true reaction energy ΔE (GMTKN55-BH76RC), not the barrier.
    bh = by_name["OH+N2_to_H+N2O"]
    assert bh.metadata["e_rxn_ref"] == 65.14
    assert bh.metadata["bh76_mode"] == "reaction_energy"
    assert bh.metadata["barrier_ref"] == 82.27
    assert bh.metadata["reaction_energy_ref"] == 65.14
    assert bh.metadata["reactants"] == ("HO", "N2")
    assert bh.metadata["products"] == ("H", "N2O")
    # IP13 (ip_ref may be None if not set; just check shape)
    li = by_name["Li_IP"]
    assert li.metadata["neutral"] == "Li"
    assert li.metadata["cation"] == "Li+"


# --- bh76_mode toggle (reaction_energy default vs barrier_height) -----------

# Expected true reaction energies ΔE = Vr − Vf (GMTKN55-BH76RC) for the
# reactant→product direction of each DFS_BH76_REACTIONS entry.
_EXPECTED_REACTION_ENERGIES = {
    "OH+N2_to_H+N2O":  65.14,
    "OH+CH3_to_O+CH4": -5.57,
    "HF+F_to_H+F2":    103.53,
}
# Forward barrier heights kept for the opt-in barrier_height mode.
_EXPECTED_BARRIERS = {
    "OH+N2_to_H+N2O":  82.27,
    "OH+CH3_to_O+CH4": 7.90,
    "HF+F_to_H+F2":    105.80,
}


def test_bh76_default_mode_is_reaction_energy():
    """The default bh76_mode yields BH76 points whose e_rxn_ref is the
    true reaction energy ΔE (+65.14 / −5.57 / +103.53 kcal/mol), not the
    barrier height. This is the bug fix: _rxn_residual_term computes
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


def test_bh76_reaction_energy_equals_vr_minus_vf():
    """Cross-check: ΔE = Vr − Vf exactly (e.g. 82.27 − 17.13 == 65.14).
    Vf values from the in-code dfs_pool.py provenance comments."""
    vf = {"OH+N2_to_H+N2O": 17.13,
          "OH+CH3_to_O+CH4": 13.47,
          "HF+F_to_H+F2": 2.27}
    for name, re_expected in _EXPECTED_REACTION_ENERGIES.items():
        vr = _EXPECTED_BARRIERS[name]
        assert vr - vf[name] == pytest.approx(re_expected, abs=1e-9)


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
        # TS geometry slot exists and is not yet staged.
        assert "ts_species" in rxn
        assert rxn["ts_species"] is None


def test_bh76_unknown_mode_raises():
    """An unrecognized bh76_mode raises ValueError."""
    from xcquinox.alec.training_points import build_dfs_pool_points
    with pytest.raises(ValueError, match="Unknown bh76_mode"):
        build_dfs_pool_points(bh76_mode="not_a_mode")


def test_bh76_barrier_height_mode_raises_gated_error():
    """bh76_mode='barrier_height' is wired but gated on transition-state
    geometries that are not yet staged — it must raise a clear,
    actionable NotImplementedError rather than silently mislabelling."""
    from xcquinox.alec.training_points import build_dfs_pool_points
    with pytest.raises(NotImplementedError, match="transition-state"):
        build_dfs_pool_points(bh76_mode="barrier_height")
