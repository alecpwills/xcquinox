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


def test_resolve_geometry_dfs_ae():
    """DFS AE molecules resolved from g2_97.traj by Hill formula."""
    from xcquinox.alec.external_refs import (
        SpeciesEntry,
        resolve_geometry,
    )
    spec = SpeciesEntry(name="H2O", charge=0, spin=0, source="dfs_ae")
    atoms = resolve_geometry(spec)
    assert len(atoms) == 3, "H2O should have 3 atoms"
    assert atoms.info.get("spin") == 0
    assert atoms.info.get("charge") == 0


def test_resolve_geometry_atom():
    """Atomic species resolved as bare atom at origin with NIST spin."""
    from xcquinox.alec.external_refs import (
        SpeciesEntry,
        resolve_geometry,
    )
    spec = SpeciesEntry(name="N", charge=0, spin=3, source="bh76")
    atoms = resolve_geometry(spec)
    assert len(atoms) == 1
    assert atoms.get_chemical_symbols() == ["N"]
    assert atoms.info["spin"] == 3
    assert atoms.info["charge"] == 0


def test_resolve_geometry_cation():
    """IP13 cation: bare atom with charge=+1 and cation_spin."""
    from xcquinox.alec.external_refs import (
        SpeciesEntry,
        resolve_geometry,
    )
    spec = SpeciesEntry(name="C+", charge=1, spin=1, source="ip13")
    atoms = resolve_geometry(spec)
    assert len(atoms) == 1
    assert atoms.get_chemical_symbols() == ["C"]
    assert atoms.info["charge"] == 1
    assert atoms.info["spin"] == 1


def test_resolve_geometry_hbpt():
    """HBPT pairs return the 6-atom water-dimer geometries."""
    from xcquinox.alec.external_refs import (
        SpeciesEntry,
        resolve_geometry,
    )
    hb = resolve_geometry(SpeciesEntry("HBWD", 1, 1, "hbpt"))
    pt = resolve_geometry(SpeciesEntry("PTWD", 1, 1, "hbpt"))
    assert len(hb) == 6 and len(pt) == 6
    # Distinct positions
    import numpy as np
    assert not np.allclose(hb.get_positions(), pt.get_positions())


def test_run_scf_rks_for_closed_shell(tmp_path):
    """Closed-shell H2O dispatches to RKS; produces (n_ao, n_ao) DM."""
    from xcquinox.alec.external_refs import (
        SpeciesEntry, resolve_geometry, run_scf_with_cache,
    )
    spec = SpeciesEntry("H2O", 0, 0, "dfs_ae")
    atoms = resolve_geometry(spec)
    payload = run_scf_with_cache(spec, atoms, cache_dir=tmp_path,
                                 basis="def2-svp", grid_level=1)
    assert payload["spin_unrestricted"] is False
    assert payload["dm"].ndim == 2
    assert payload["dm"].shape[0] == payload["dm"].shape[1]


def test_run_scf_uks_for_doublet(tmp_path):
    """Doublet H atom dispatches to UKS; produces (2, n_ao, n_ao) DM."""
    from xcquinox.alec.external_refs import (
        SpeciesEntry, resolve_geometry, run_scf_with_cache,
    )
    spec = SpeciesEntry("H", 0, 1, "dfs_atom")
    atoms = resolve_geometry(spec)
    payload = run_scf_with_cache(spec, atoms, cache_dir=tmp_path,
                                 basis="def2-svp", grid_level=1)
    assert payload["spin_unrestricted"] is True
    assert payload["dm"].shape[0] == 2  # (2, n_ao, n_ao)


def test_run_scf_cache_hit(tmp_path):
    """Second call with same cache_dir reads from cache, no second SCF."""
    from xcquinox.alec.external_refs import (
        SpeciesEntry, resolve_geometry, run_scf_with_cache,
    )
    spec = SpeciesEntry("H2", 0, 0, "dfs_ae")
    atoms = resolve_geometry(spec)
    p1 = run_scf_with_cache(spec, atoms, cache_dir=tmp_path,
                            basis="def2-svp", grid_level=1)
    cache_path = tmp_path / "_intermediates" / "H2_scf.npz"
    assert cache_path.is_file(), "SCF cache not written"
    mtime = cache_path.stat().st_mtime
    p2 = run_scf_with_cache(spec, atoms, cache_dir=tmp_path,
                            basis="def2-svp", grid_level=1)
    assert cache_path.stat().st_mtime == mtime, "cache rewritten on hit"
