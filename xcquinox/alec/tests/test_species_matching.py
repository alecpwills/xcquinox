"""Composition-level training/pool species identity (species_matching).

The held-out filter is name-based; these tests pin the alias layer that
connects the DFS Hill-formula training vocabulary to the GMTKN55-style pool
vocabulary. The real benchmark pool is used directly: the mapping it must
produce is small, known, and physically checkable.
"""
import pytest

from xcquinox.alec import species_matching as sm


def test_parse_formula_name_basic():
    assert sm.parse_formula_name("H3N") == ((("H", 3), ("N", 1)), 0)
    assert sm.parse_formula_name("CHN") == ((("C", 1), ("H", 1), ("N", 1)), 0)
    assert sm.parse_formula_name("HO") == ((("H", 1), ("O", 1)), 0)
    assert sm.parse_formula_name("FLi") == ((("F", 1), ("Li", 1)), 0)
    assert sm.parse_formula_name("Na2") == ((("Na", 2),), 0)
    assert sm.parse_formula_name("C+") == ((("C", 1),), 1)
    assert sm.parse_formula_name("Li+") == ((("Li", 1),), 1)


def test_parse_formula_name_rejects_non_formulas():
    # Pool common names must fail the parse -- their identity comes from the
    # pool specs, never from name parsing.
    for bad in ("methanol", "ch2-trip", "RKT01", "c-hcoh", "oxirane",
                "", "+", "Xx3", "acetaldehyde"):
        assert sm.parse_formula_name(bad) is None, bad


def test_is_atomic():
    assert sm.is_atomic(((("H"), 1),)) or sm.is_atomic((("H", 1),))
    assert sm.is_atomic((("C", 1),))
    assert not sm.is_atomic((("C", 1), ("H", 4)))
    assert not sm.is_atomic((("H", 2),))


def test_trained_key_spin_from_dfs_tables():
    # CH2 was built as the triplet (dfs_pool ground-state table): the key must
    # carry spin=2 so it can distinguish the pool's ch2-trip from ch2-sing.
    comp, charge, spin = sm.trained_species_key("CH2")
    assert comp == (("C", 1), ("H", 2)) and charge == 0 and spin == 2
    comp, charge, spin = sm.trained_species_key("HO")
    assert spin == 1  # hydroxyl radical doublet


@pytest.fixture(scope="module")
def pool_specs():
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    specs, _ = load_full_held_out_pools()
    return specs


def test_known_hill_pool_twins_alias(pool_specs):
    # The confirmed leak set: trained Hill names whose pool twins carry
    # GMTKN55 common names invisible to name matching.
    aliases = sm.trained_pool_aliases(
        ["HO", "HN", "H3N", "CHN", "CH2"], pool_specs, verbose=False)
    for expected in ("oh", "hcn", "ch2-trip"):
        assert expected in aliases, (expected, sorted(aliases))
    assert {"nh", "NH"} & aliases
    assert {"nh3", "NH3"} & aliases
    # The singlet carbene is a DIFFERENT physical species from the trained
    # triplet: it must NOT be excluded.
    assert "ch2-sing" not in aliases
    # Hydrogen isocyanide shares CHN's composition/charge/spin but not its
    # geometry: the trained hydrogen cyanide must resolve to hcn only.
    assert "hnc" not in aliases


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


def test_isomer_ambiguity_without_geometry_keeps_all():
    # Conservative: with no geometry to discriminate, both candidates are
    # excluded from the held-out set (over-matching can only shrink it).
    aliases = sm.trained_pool_aliases(["CHN"], _iso_pool(), verbose=False,
                                      _geometry_provider=lambda name: None)
    assert aliases == {"hcn", "hnc"}


def test_name_visible_isomer_does_not_leave_twin_as_lone_match():
    # The acetylene/vinylidene trap: pool c2h2 is a casefold name-twin of
    # trained C2H2 and ch2c its composition-degenerate isomer. Removing the
    # name-visible candidate BEFORE ambiguity detection leaves ch2c as a
    # lone "unambiguous" match that skips the geometry check -- wrongly
    # excluding a legitimately held-out species. Ambiguity must be assessed
    # over ALL composition matches; name-visible ones drop only from the
    # returned set.
    pool = {
        "c2h2": {"atom_composition": (("C", 2), ("H", 2)), "charge": 0,
                 "spin": 0,
                 "atom": ("C 0 0 0.6; C 0 0 -0.6; "
                          "H 0 0 1.66; H 0 0 -1.66")},
        "ch2c": {"atom_composition": (("C", 2), ("H", 2)), "charge": 0,
                 "spin": 0,
                 "atom": ("C 0 0 0; C 0 0 1.3; "
                          "H 0 0.94 -0.55; H 0 -0.94 -0.55")},
    }
    provider = (lambda name: (("C", "C", "H", "H"),
                              ((0.0, 0.0, 0.6), (0.0, 0.0, -0.6),
                               (0.0, 0.0, 1.66), (0.0, 0.0, -1.66)))
                if name == "C2H2" else None)
    aliases = sm.trained_pool_aliases(["C2H2"], pool, verbose=False,
                                      _geometry_provider=provider)
    assert aliases == set()


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


def test_exact_and_case_matches_not_duplicated(pool_specs):
    # ch4/CH4 are name-visible (casefold) matches; the alias layer only adds
    # the differently-named twins.
    aliases = sm.trained_pool_aliases(["CH4"], pool_specs, verbose=False)
    assert "ch4" not in aliases and "CH4" not in aliases


def test_unparseable_training_names_are_ignored(pool_specs):
    assert sm.trained_pool_aliases(["not-a-formula"], pool_specs,
                                   verbose=False) == set()


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


def test_canonical_keys_case_twins_share_class_isomers_split():
    # The pool lists the same species twice under case-twin names (O/o,
    # NH3/nh3): they share one geometry class. A true isomer in the same
    # (composition, charge, spin) family gets its own class.
    pool = dict(_iso_pool())          # hcn + hnc (isomers)
    pool["HCN"] = dict(pool["hcn"])   # case twin of hcn
    pool["o"] = {"atom_composition": (("O", 1),), "charge": 0, "spin": 2,
                 "atom": "O 0 0 0"}
    pool["O"] = {"atom_composition": (("O", 1),), "charge": 0, "spin": 2,
                 "atom": "O 0 0 0"}
    keys = sm.canonical_species_keys(pool, [],
                                     _geometry_provider=lambda n: None)
    assert keys["o"] == keys["O"]
    assert keys["HCN"] == keys["hcn"]
    assert keys["hcn"] != keys["hnc"]


def test_canonical_keys_casefold_and_unresolved():
    pool = {
        "ch4": {"atom_composition": (("C", 1), ("H", 4)), "charge": 0,
                "spin": 0, "atom": "C 0 0 0; H 1 0 0; H 0 1 0; H 0 0 1; "
                                    "H -1 -1 -1"},
    }
    keys = sm.canonical_species_keys(pool, ["CH4", "methanol"],
                                     _geometry_provider=lambda n: None)
    assert keys["CH4"] == keys["ch4"]          # casefold twin shares the key
    assert keys["methanol"] == ("name:methanol",)   # never matches a pool key


def test_canonical_keys_unresolved_degenerate_keeps_all_candidates():
    pool = _iso_pool()
    keys = sm.canonical_species_keys(pool, ["CHN"],
                                     _geometry_provider=lambda n: None)
    assert set(keys["CHN"]) == set(keys["hcn"]) | set(keys["hnc"])


def test_reaction_identity_keys_cross_vocabulary():
    # The trained CHN atomization must share an identity with the pool's
    # w411_hcn_atomization and NOT with w411_hnc_atomization; permuted
    # reactant order must not matter.
    pool = dict(_iso_pool())
    for a in ("h", "c", "n"):
        pool[a] = {"atom_composition": ((a.upper(), 1),), "charge": 0,
                   "spin": 1 if a != "c" else 2,
                   "atom": f"{a.upper()} 0 0 0"}
    provider = (lambda name: (("C", "N", "H"),
                              ((0.0, 0.0, 0.0), (0.0, 0.0, 1.16),
                               (0.0, 0.0, -1.07)))
                if name == "CHN" else None)
    keys = sm.canonical_species_keys(pool, ["CHN", "H", "C", "N"],
                                     _geometry_provider=provider)
    trained = {"name": "CHN", "reactants": ["CHN"],
               "products": ["C", "H", "N"],
               "coeffs": [-1.0, 1.0, 1.0, 1.0]}
    hcn_pool = {"name": "w411_hcn_atomization", "reactants": ["hcn"],
                "products": ["h", "c", "n"],
                "coeffs": [-1.0, 1.0, 1.0, 1.0]}
    hnc_pool = {"name": "w411_hnc_atomization", "reactants": ["hnc"],
                "products": ["h", "c", "n"],
                "coeffs": [-1.0, 1.0, 1.0, 1.0]}
    t = set(sm.reaction_identity_keys(trained, keys))
    assert t & set(sm.reaction_identity_keys(hcn_pool, keys))
    assert not t & set(sm.reaction_identity_keys(hnc_pool, keys))
    # permuted product order coincides
    perm = dict(hcn_pool, products=["n", "h", "c"])
    assert set(sm.reaction_identity_keys(perm, keys)) & t


def test_full_training_vocabulary_maps_cleanly(pool_specs):
    # Every Hill name in the DFS AE table parses, and its alias set contains
    # no pool species whose composition differs -- a parser regression on any
    # current training name fails here.
    from xcquinox.alec.dfs_pool import DFS_AE_HILL
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
