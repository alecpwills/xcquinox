"""Tests for ``xcquinox.alec.full_benchmark_pools``.

The pre-built JSON caches under ``xcquinox/alec/data/{bh76,w411}_full_pool.json``
are committed to the repo so these tests do NOT touch the GMTKN55 source
on disk. The regen script
``scripts/rebuild_full_benchmark_pools.py`` is what (re)generates the
JSON; if the source data changes, the JSON gets rebuilt and the test
assertions on counts adjust accordingly.
"""
from __future__ import annotations

import pytest

from xcquinox.alec.full_benchmark_pools import (
    load_full_bh76,
    load_full_w411,
    load_full_held_out_pools,
)
from xcquinox.alec.config import MoleculeSpec


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
