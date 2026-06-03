"""Tests for ``xcquinox.alec.full_benchmark_pools``.

The pre-built JSON caches under ``xcquinox/alec/data/{bh76,w411}_full_pool.json``
are committed to the repo so these tests do NOT touch the GMTKN55 source
on disk. The regen script
``scripts/rebuild_full_benchmark_pools.py`` is what (re)generates the
JSON; if the source data changes, the JSON gets rebuilt and the test
assertions on counts adjust accordingly.
"""
from __future__ import annotations

import math

import pytest

from xcquinox.alec.full_benchmark_pools import (
    load_full_bh76,
    load_full_w411,
    load_full_held_out_pools,
)
from xcquinox.alec.config import MoleculeSpec


# ---------------------------------------------------------------------------
# Geometry helpers (for the units-regression tests below)
# ---------------------------------------------------------------------------

def _parse_atom_str(atom_str: str):
    """``'H x y z; O x y z'`` -> [(elem, x, y, z), ...] as floats (Angstrom)."""
    out = []
    for tok in atom_str.split(";"):
        p = tok.split()
        if len(p) < 4:
            continue
        out.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return out


def _bond_length(atom_str: str, i: int, j: int) -> float:
    c = _parse_atom_str(atom_str)
    return math.dist(c[i][1:], c[j][1:])


# ---------------------------------------------------------------------------
# Sanity counts
# ---------------------------------------------------------------------------

def test_load_full_bh76_returns_76_reactions():
    """GMTKN55-BH76 has exactly 76 forward-barrier reactions."""
    mol_specs, reactions = load_full_bh76()
    assert len(reactions) == 76, len(reactions)
    # Each reaction has at least 2 species touching it (lowest-bound; some
    # have 3+ for the transition state path).
    for r in reactions:
        assert len(r["reactants"]) + len(r["products"]) >= 2, r["name"]


def test_load_full_w411_returns_140_atomizations():
    """GMTKN55-W4-11 has exactly 140 atomization reactions."""
    mol_specs, reactions = load_full_w411()
    assert len(reactions) == 140, len(reactions)
    # Each W4-11 reaction has exactly one molecule reactant + ≥1 atom product.
    for r in reactions:
        assert len(r["reactants"]) == 1, r["name"]
        assert len(r["products"]) >= 1, r["name"]


@pytest.mark.parametrize("loader_name,cache_name", [
    ("load_full_bh76", "_BH76_CACHE"),
    ("load_full_w411", "_W411_CACHE"),
])
def test_pool_load_hits_cache_on_second_call(loader_name, cache_name, monkeypatch):
    """REGRESSION (D10_pools-01/02): the (basis, grid_level) cache must HIT on a
    repeat call. The cache compared keys with ``is`` (identity) against a freshly
    built tuple, so it never hit and re-parsed the JSON on every call."""
    import xcquinox.alec.full_benchmark_pools as fbp
    monkeypatch.setattr(fbp, cache_name, None)
    calls = {"n": 0}
    orig = fbp._load_pool_from_json

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(fbp, "_load_pool_from_json", _counting)
    loader = getattr(fbp, loader_name)
    first_specs, first_rxns = loader("def2-svp", 1)
    second_specs, second_rxns = loader("def2-svp", 1)
    # The second identical call must hit the cache, i.e. NOT re-parse the JSON.
    assert calls["n"] == 1, (
        f"cache never hit: _load_pool_from_json called {calls['n']}x for two "
        f"identical {loader_name}(...) calls")
    # And it must return the cached payload (same species/reaction objects).
    assert second_specs is getattr(fbp, cache_name)[1][0]
    assert len(second_rxns) == len(first_rxns)


def test_full_bh76_species_count_at_least_50():
    """BH76 covers at least 50 distinct species (including transition states)."""
    mol_specs, _ = load_full_bh76()
    assert len(mol_specs) >= 50, len(mol_specs)


def test_full_w411_species_count_at_least_100():
    """W4-11 covers ~99 molecules + ~10 atoms; at least 100 species."""
    mol_specs, _ = load_full_w411()
    assert len(mol_specs) >= 100, len(mol_specs)


# ---------------------------------------------------------------------------
# Schema parity with PROBE_C_BH76_OUT_OF_TRAINING
# ---------------------------------------------------------------------------

# The reaction-dict keys that reaction_mae_kcalmol + per_reaction_errors
# read from. Any reaction in the full-pool must carry all of these.
_REQUIRED_RXN_KEYS = (
    "name", "source_pool", "reactants", "products", "coeffs",
    "reaction_energy_ref", "species_spins", "species_charges", "source",
)


@pytest.mark.parametrize("loader,expected_pool",
                          [(load_full_bh76, "bh76"),
                           (load_full_w411, "w411")])
def test_reaction_schema_matches_probe_c_keys(loader, expected_pool):
    """Every reaction dict has every key reaction_mae_kcalmol consumes."""
    _, reactions = loader()
    for r in reactions:
        for k in _REQUIRED_RXN_KEYS:
            assert k in r, f"{r.get('name')} missing {k}"
        assert r["source_pool"] == expected_pool, r["name"]
        # coeffs aligned with reactants + products
        assert len(r["coeffs"]) == len(r["reactants"]) + len(r["products"]), (
            r["name"], r["coeffs"], r["reactants"], r["products"]
        )
        # signed convention: reactant coeffs negative, product coeffs positive
        n_r = len(r["reactants"])
        for c in r["coeffs"][:n_r]:
            assert c < 0, (r["name"], r["coeffs"])
        for c in r["coeffs"][n_r:]:
            assert c > 0, (r["name"], r["coeffs"])


# ---------------------------------------------------------------------------
# MoleculeSpec construction
# ---------------------------------------------------------------------------

def test_species_dicts_yield_valid_mol_specs():
    """Each species in BH76+W4-11 builds a hashable MoleculeSpec."""
    mol_specs, _ = load_full_bh76()
    for name, ms in mol_specs.items():
        assert isinstance(ms, MoleculeSpec), name
        assert ms.name == name, (ms.name, name)
        # atom_composition is a tuple of pairs (element, count) — hashable
        for elem, count in ms.atom_composition:
            assert isinstance(elem, str), elem
            assert isinstance(count, int) and count >= 1, count
        # spin = 2S (non-negative); charge can be negative
        assert ms.spin >= 0, (name, ms.spin)


def test_h_atom_appears_in_both_pools():
    """The H atom is in both BH76 and W4-11. Confirms the merge logic in
    ``load_full_held_out_pools`` deduplicates correctly."""
    bh76_mols, _ = load_full_bh76()
    w411_mols, _ = load_full_w411()
    assert "h" in bh76_mols, list(bh76_mols)[:10]
    assert "h" in w411_mols, list(w411_mols)[:10]
    # H is a doublet (1 unpaired electron, 2S = 1)
    assert bh76_mols["h"].spin == 1
    # Same atom regardless of pool
    assert bh76_mols["h"].atom_composition == w411_mols["h"].atom_composition


def test_load_full_held_out_pools_merges_species():
    """Combined pool: BH76 + W4-11 species merged (no duplicate by name)."""
    bh76_mols, bh76_rxns = load_full_bh76()
    w411_mols, w411_rxns = load_full_w411()
    all_mols, all_rxns = load_full_held_out_pools()
    # Reactions concatenate; species merge.
    assert len(all_rxns) == len(bh76_rxns) + len(w411_rxns)
    assert len(all_mols) <= len(bh76_mols) + len(w411_mols)
    # Every species in BH76 or W4-11 must appear in the merged dict.
    for name in bh76_mols:
        assert name in all_mols, name
    for name in w411_mols:
        assert name in all_mols, name


# ---------------------------------------------------------------------------
# Round-trip a single reaction (sanity that the parser preserves the math)
# ---------------------------------------------------------------------------

def test_bh76_first_reaction_h_n2o_to_n2ohts_has_ref_17_7():
    """The first BH76 line is ``$tmer h n2o n2ohts x -1 -1 1 $w 17.7``. The
    parsed reaction must carry that exact reference energy."""
    _, reactions = load_full_bh76()
    target = None
    for r in reactions:
        if r["reactants"] == ["h", "n2o"] and r["products"] == ["n2ohts"]:
            target = r
            break
    assert target is not None, "h + n2o -> n2ohts not found in BH76"
    assert target["reaction_energy_ref"] == pytest.approx(17.7, abs=1e-6)
    assert target["coeffs"] == [-1.0, -1.0, 1.0]
    # H is open-shell (doublet); n2o and the transition state should be
    # closed-shell singlet by GMTKN55 convention.
    assert target["species_spins"]["h"] == 1
    assert target["species_charges"]["h"] == 0


def test_w411_h2_atomization_ref_109_493():
    """The first W4-11 line is ``$tmer {h2,h} x -1 2 $w 109.493``. The parsed
    atomization must carry the reference 2*E(h) - E(h2) = 109.493."""
    _, reactions = load_full_w411()
    target = None
    for r in reactions:
        if r["reactants"] == ["h2"] and r["products"] == ["h"]:
            target = r
            break
    assert target is not None
    assert target["reaction_energy_ref"] == pytest.approx(109.493, abs=1e-6)
    assert target["coeffs"] == [-1.0, 2.0]


# ---------------------------------------------------------------------------
# Geometry-units regression (the held-out struc.xyz-as-Bohr bug, 2026-05-31)
# ---------------------------------------------------------------------------
#
# GMTKN55 ``struc.xyz`` files are in ANGSTROM (standard .xyz convention). A bug
# in ``_atoms_to_pyscf_str`` divided them by BOHR_PER_ANGSTROM, shrinking every
# molecule ~1.89x and producing catastrophically wrong held-out reaction
# energies (W4-11 atomizations came out negative; BH76 barriers ~20x too big).
# The pre-2026-05-31 suite never checked a bond length or an energy, so it
# missed this entirely. These tests pin the physical geometry + energy sign.

def test_w411_h2_bond_length_is_physical_angstrom():
    """H2 equilibrium bond length is 0.741 Angstrom. The buggy (shrunk)
    geometry gives ~0.393 A (0.741 / 1.8897)."""
    mol_specs, _ = load_full_w411()
    d = _bond_length(mol_specs["h2"].atom, 0, 1)
    assert d == pytest.approx(0.741, abs=0.03), (
        f"H2 bond length {d:.4f} A is not physical (expect ~0.741 A). "
        f"~0.393 A indicates the struc.xyz-as-Bohr units bug.")


def test_w411_n2o_bond_lengths_are_physical_angstrom():
    """N2O is linear with N-N ~ 1.128 A and N-O ~ 1.184 A. The buggy geometry
    shrinks these to ~0.60 / ~0.63 A."""
    mol_specs, _ = load_full_w411()
    coords = _parse_atom_str(mol_specs["n2o"].atom)
    # Distances between consecutive atoms along the molecular axis; the two
    # nearest-neighbour bonds should be ~1.13 and ~1.18 A (order-independent).
    import itertools
    dists = sorted(math.dist(a[1:], b[1:])
                   for a, b in itertools.combinations(coords, 2))
    assert dists[0] == pytest.approx(1.13, abs=0.05), dists
    assert dists[1] == pytest.approx(1.18, abs=0.05), dists


@pytest.mark.slow
def test_w411_h2_atomization_pbe_sign_and_magnitude():
    """End-to-end: PBE/def2-svp atomization 2*E(H) - E(H2) must be ~+105
    kcal/mol (positive). The shrunk geometry gives ~-46 kcal/mol (wrong sign)."""
    from pyscf import gto, dft
    KCAL = 627.5094740631
    mol_specs, _ = load_full_w411()

    def _e(name, spin):
        ms = mol_specs[name]
        mol = gto.M(atom=ms.atom, basis="def2-svp", charge=ms.charge,
                    spin=spin, unit="angstrom", verbose=0)
        mf = (dft.UKS(mol) if spin else dft.RKS(mol))
        mf.xc = "pbe"
        return float(mf.kernel())

    e_h2 = _e("h2", 0)
    e_h = _e("h", 1)
    atomization = (2.0 * e_h - e_h2) * KCAL
    assert atomization == pytest.approx(105.0, abs=15.0), (
        f"PBE H2 atomization {atomization:.1f} kcal/mol (expect ~+105; "
        f"~-46 indicates the geometry units bug).")
