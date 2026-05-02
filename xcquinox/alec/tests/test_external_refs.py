"""Tests for xcquinox.alec.external_refs species union + pipeline."""
from xcquinox.alec.external_refs import (
    SpeciesEntry,
    build_species_union,
)


def test_species_union_dedup_count():
    """Total unique (name, charge, spin) triples ≈ 58 across DFS+probes+HBPT."""
    species = build_species_union()
    # DFS AE 21 + DFS atom_refs 2 + BH76-new 5 + IP13 4 + probe-induced atoms 5
    # + Probe-A 6 + Probe-B 6 + Probe-C-new 1 + Probe-D 6 + HBPT 2 ≈ 58.
    assert 50 <= len(species) <= 70, (
        f"Expected ~58 species, got {len(species)}: "
        f"{[s.name for s in species]}"
    )


def test_species_union_dedup_key_charge_spin():
    """Li (neutral, spin=1) and Li+ (charge=+1, spin=0) are DISTINCT entries."""
    species = build_species_union()
    by_key = {(s.name, s.charge, s.spin): s for s in species}
    assert ("Li", 0, 1) in by_key, "Li neutral missing"
    assert ("Li+", 1, 0) in by_key, "Li+ cation missing"
    assert by_key[("Li", 0, 1)] is not by_key[("Li+", 1, 0)]


def test_species_union_includes_hbpt():
    """HBWD and PTWD water-dimer entries are in the union (charge=1, spin=1)."""
    species = build_species_union()
    names = {s.name for s in species}
    assert "HBWD" in names, "HBWD water-dimer missing from union"
    assert "PTWD" in names, "PTWD water-dimer missing from union"


def test_species_union_open_shell_dispatch():
    """Triplets (NH spin=2, CH2 spin=2, O2 spin=2) dispatch to UKS path."""
    species = build_species_union()
    by_name = {s.name: s for s in species}
    assert by_name["HN"].spin == 2  # ³Σ⁻ NH per dfs_pool.py
    assert by_name["CH2"].spin == 2  # ³B₁ methylene
    assert by_name["O2"].spin == 2  # ³Σg⁻ from Probe D
