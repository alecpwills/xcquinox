"""BH76+W4-11 trainable pool + the generalizable reaction-pool builder it wraps."""
from ase import Atoms

from xcquinox.alec.training_points import (
    build_reaction_pool_points,
    build_bh76w411_pool_points,
    species_union_from_points,
)


def _atom(name, charge=0, spin=0):
    # A single-atom stand-in (geometry/identity irrelevant to the builder logic,
    # which only maps names -> Atoms and assembles reaction metadata).
    a = Atoms("He", positions=[(0.0, 0.0, 0.0)])
    a.info.update(name=name, charge=charge, spin=spin)
    return a


# ---------------------------------------------------------------------------
# Generic builder (arbitrary training set)
# ---------------------------------------------------------------------------

def test_build_reaction_pool_points_generic():
    atoms_by_name = {
        "h2": _atom("h2"), "h": _atom("h", spin=1),
        "o2": _atom("o2", spin=2), "o": _atom("o", spin=2),
    }
    reactions = [
        {"name": "h2_atomization", "reactants": ["h2"], "products": ["h"],
         "coeffs": [-1.0, 2.0], "reaction_energy_ref": 109.5,
         "source_pool": "custom"},
        {"name": "o2_atomization", "reactants": ["o2"], "products": ["o"],
         "coeffs": [-1.0, 2.0], "reaction_energy_ref": 120.0,
         "source_pool": "custom"},
    ]
    pts = build_reaction_pool_points(reactions, atoms_by_name)
    assert len(pts) == 2
    assert all(p.kind == "bh76" for p in pts)
    assert [p.name for p in pts] == ["h2_atomization", "o2_atomization"]
    p0 = pts[0]
    assert p0.metadata["e_rxn_ref"] == 109.5
    assert p0.metadata["coeffs"] == (-1.0, 2.0)
    assert {s.info["name"] for s in p0.species} == {"h2", "h"}


def test_build_reaction_pool_points_dedups_by_name():
    atoms_by_name = {"a": _atom("a"), "b": _atom("b")}
    rxn = {"name": "r", "reactants": ["a"], "products": ["b"],
           "coeffs": [-1.0, 1.0], "reaction_energy_ref": 1.0}
    # same-name reaction twice (identical) -> collapses to one point
    pts = build_reaction_pool_points([rxn, dict(rxn)], atoms_by_name)
    assert len(pts) == 1


def test_build_reaction_pool_points_missing_species_raises():
    import pytest
    reactions = [{"name": "r", "reactants": ["x"], "products": ["y"],
                  "coeffs": [-1.0, 1.0], "reaction_energy_ref": 1.0}]
    with pytest.raises(KeyError):
        build_reaction_pool_points(reactions, {"x": _atom("x")})


# ---------------------------------------------------------------------------
# BH76+W4-11 concrete pool
# ---------------------------------------------------------------------------

def test_build_bh76w411_pool_points():
    pts = build_bh76w411_pool_points()
    # 216 reactions with 4 identical-name duplicates -> 212 unique points
    assert len(pts) == 212
    assert all(p.kind == "bh76" for p in pts)
    names = [p.name for p in pts]
    assert len(set(names)) == len(names)            # unique (name-based resolver)
    # both benchmarks represented
    pools = {p.metadata.get("source_pool") for p in pts}
    assert "bh76" in pools and "w411" in pools
    for p in pts:
        assert p.metadata["e_rxn_ref"] is not None
        assert len(p.species) >= 1
        assert all("name" in s.info for s in p.species)


def test_bh76w411_species_union_is_small_subset():
    """The CCSD species-union of a few chosen reactions is far smaller than the
    full 214 species, the property that keeps the preflight feasible."""
    pts = build_bh76w411_pool_points()
    chosen = pts[:6]
    union = species_union_from_points(chosen)
    assert 0 < len(union) < 60
