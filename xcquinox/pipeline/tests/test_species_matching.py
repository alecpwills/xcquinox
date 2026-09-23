"""Composition-level training/pool species identity (species_matching).

The held-out filter is name-based; these tests pin the alias layer that
connects the DFS Hill-formula training vocabulary to the GMTKN55-style pool
vocabulary. The real benchmark pool is used directly: the mapping it must
produce is small, known, and physically checkable.
"""
import pytest

from xcquinox.pipeline import species_matching as sm


def test_parse_formula_name_basic():
    assert sm.parse_formula_name("H3N") == ((("H", 3), ("N", 1)), 0)
    assert sm.parse_formula_name("CHN") == ((("C", 1), ("H", 1), ("N", 1)), 0)
    assert sm.parse_formula_name("HO") == ((("H", 1), ("O", 1)), 0)
    assert sm.parse_formula_name("FLi") == ((("F", 1), ("Li", 1)), 0)
    assert sm.parse_formula_name("Na2") == ((("Na", 2),), 0)
    assert sm.parse_formula_name("C+") == ((("C", 1),), 1)
    assert sm.parse_formula_name("Li+") == ((("Li", 1),), 1)


def test_trained_key_spin_from_dfs_tables():
    # CH2 was built as the triplet (dfs_pool ground-state table): the key must
    # carry spin=2 so it can distinguish the pool's ch2-trip from ch2-sing.
    comp, charge, spin = sm.trained_species_key("CH2")
    assert comp == (("C", 1), ("H", 2)) and charge == 0 and spin == 2
    comp, charge, spin = sm.trained_species_key("HO")
    assert spin == 1  # hydroxyl radical doublet


@pytest.fixture(scope="module")
def pool_specs():
    from xcquinox.pipeline.full_benchmark_pools import load_full_held_out_pools
    specs, _ = load_full_held_out_pools()
    return specs


def _iso_pool():
    return {
        "hcn": {"atom_composition": (("C", 1), ("H", 1), ("N", 1)),
                "charge": 0, "spin": 0,
                "atom": "C 0 0 0; N 0 0 1.15; H 0 0 -1.06"},
        "hnc": {"atom_composition": (("C", 1), ("H", 1), ("N", 1)),
                "charge": 0, "spin": 0,
                "atom": "N 0 0 0; C 0 0 1.17; H 0 0 -1.00"},
    }


def test_isomer_ambiguity_resolved_by_geometry():
    provider = (lambda name: (("C", "N", "H"),
                              ((0.0, 0.0, 0.0), (0.0, 0.0, 1.16),
                               (0.0, 0.0, -1.07)))
                if name == "CHN" else None)
    aliases = sm.trained_pool_aliases(["CHN"], _iso_pool(), verbose=False,
                                      _geometry_provider=provider)
    assert aliases == {"hcn"}


def test_real_pool_acetylene_does_not_exclude_vinylidene(pool_specs):
    # Same trap against the real pool + real trained geometry: trained
    # acetylene must not alias vinylidene (ch2c).
    aliases = sm.trained_pool_aliases(["C2H2"], pool_specs, verbose=False)
    assert "ch2c" not in aliases


def test_charge_separates_atom_from_anion(pool_specs):
    # A trained neutral F atom must not alias the pool fluoride anion.
    aliases = sm.trained_pool_aliases(["F"], pool_specs, verbose=False)
    assert "f-" not in aliases
    # (the neutral pool 'f' is a case-twin of the name itself: covered by the
    # name-based filter, deliberately not repeated in the alias set)
    assert "f" not in aliases


def test_canonical_keys_separate_geometry_classes():
    # hcn and hnc share (composition, charge, spin); their pool keys must
    # differ by geometry class, and a trained CHN with a geometry resolves to
    # hcn's key only.
    pool = _iso_pool()
    provider = (lambda name: (("C", "N", "H"),
                              ((0.0, 0.0, 0.0), (0.0, 0.0, 1.16),
                               (0.0, 0.0, -1.07)))
                if name == "CHN" else None)
    keys = sm.canonical_species_keys(pool, ["CHN"],
                                     _geometry_provider=provider)
    assert keys["hcn"] != keys["hnc"]
    assert keys["CHN"] == keys["hcn"]
    assert len(keys["CHN"]) == 1


def test_full_training_vocabulary_maps_cleanly(pool_specs):
    # Every Hill name in the DFS AE table parses, and its alias set contains
    # no pool species whose composition differs -- a parser regression on any
    # current training name fails here.
    from xcquinox.pipeline.dfs_pool import DFS_AE_HILL
    for name in DFS_AE_HILL:
        parsed = sm.parse_formula_name(name)
        assert parsed is not None, name
    aliases = sm.trained_pool_aliases(DFS_AE_HILL, pool_specs, verbose=False)
    keys = {}
    for n in DFS_AE_HILL:
        k = sm.trained_species_key(n)
        keys[n] = (k[0], k[1])
    for pool_name in aliases:
        pk = sm.pool_species_key(pool_specs[pool_name])
        assert (pk[0], pk[1]) in set(keys.values()), pool_name
