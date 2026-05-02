"""Tests for xcquinox.alec.external_refs species union + pipeline."""
import pytest
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


def test_run_ccsd_rccsd_h2(tmp_path):
    """RCCSD on H2 produces dm_ao shape (n_ao, n_ao) and rho_ref_grid (N_grid,)."""
    from xcquinox.alec.external_refs import (
        SpeciesEntry, resolve_geometry,
        run_scf_with_cache, run_ccsd_with_cache,
    )
    spec = SpeciesEntry("H2", 0, 0, "dfs_ae")
    atoms = resolve_geometry(spec)
    scf = run_scf_with_cache(spec, atoms, cache_dir=tmp_path,
                             basis="def2-svp", grid_level=1)
    cc = run_ccsd_with_cache(spec, atoms, scf_payload=scf,
                             cache_dir=tmp_path,
                             basis="def2-svp", grid_level=1)
    assert cc["dm_ao"].ndim == 2
    assert cc["rho_ref_grid"].ndim == 1, (
        "rho_ref_grid must be spin-summed 1D shape (N_grid,)")
    assert cc["rho_ref_grid"].size == scf["n_grid"]


def test_run_ccsd_uccsd_h_atom_spin_summed_rho(tmp_path):
    """UCCSD on H atom: dm_ao spin-resolved (2, n_ao, n_ao); rho is SUMMED 1D."""
    from xcquinox.alec.external_refs import (
        SpeciesEntry, resolve_geometry,
        run_scf_with_cache, run_ccsd_with_cache,
    )
    import numpy as np
    spec = SpeciesEntry("H", 0, 1, "dfs_atom")
    atoms = resolve_geometry(spec)
    scf = run_scf_with_cache(spec, atoms, cache_dir=tmp_path,
                             basis="def2-svp", grid_level=1)
    cc = run_ccsd_with_cache(spec, atoms, scf_payload=scf,
                             cache_dir=tmp_path,
                             basis="def2-svp", grid_level=1)
    assert cc["dm_ao"].ndim == 3 and cc["dm_ao"].shape[0] == 2
    assert cc["rho_ref_grid"].ndim == 1, (
        "rho_ref_grid must be spin-summed 1D not (2, N_grid) — see "
        "data.py:296-299 for the canonical spin-summing pattern")
    # H atom has 1 electron total — integrated rho must equal 1.0
    integ = float(np.sum(cc["grid_weights"] * cc["rho_ref_grid"]))
    assert abs(integ - 1.0) < 0.05, f"integrated rho={integ} != 1 for H atom"


def test_oep_cascade_writes_npz_with_required_keys(tmp_path):
    """Stage 3 OEP for H2 produces npz with vxc_ref, dm_target, rho_ref_grid."""
    from xcquinox.alec.external_refs import (
        SpeciesEntry, resolve_geometry,
        run_scf_with_cache, run_ccsd_with_cache, run_oep_cascade,
    )
    import numpy as np
    spec = SpeciesEntry("H2", 0, 0, "dfs_ae")
    atoms = resolve_geometry(spec)
    scf = run_scf_with_cache(spec, atoms, cache_dir=tmp_path,
                             basis="def2-svp", grid_level=1)
    cc = run_ccsd_with_cache(spec, atoms, scf_payload=scf,
                             cache_dir=tmp_path,
                             basis="def2-svp", grid_level=1)
    npz_path = run_oep_cascade(spec, atoms, ccsd_payload=cc,
                               cache_dir=tmp_path,
                               basis="def2-svp", grid_level=1)
    assert npz_path.is_file()
    with np.load(npz_path, allow_pickle=False) as z:
        for key in ("vxc_ref", "dm_target", "rho_ref_grid",
                    "ref_density_method", "oep_baseline_xc",
                    "oep_aux_basis"):
            assert key in z.files, f"missing {key} in {npz_path}"


def test_oep_cascade_skip_if_cached(tmp_path):
    """Re-invocation with full cache: skip-if-cached returns existing path."""
    from xcquinox.alec.external_refs import (
        SpeciesEntry, resolve_geometry,
        run_scf_with_cache, run_ccsd_with_cache, run_oep_cascade,
    )
    spec = SpeciesEntry("H2", 0, 0, "dfs_ae")
    atoms = resolve_geometry(spec)
    scf = run_scf_with_cache(spec, atoms, cache_dir=tmp_path,
                             basis="def2-svp", grid_level=1)
    cc = run_ccsd_with_cache(spec, atoms, scf_payload=scf,
                             cache_dir=tmp_path,
                             basis="def2-svp", grid_level=1)
    p1 = run_oep_cascade(spec, atoms, ccsd_payload=cc,
                         cache_dir=tmp_path,
                         basis="def2-svp", grid_level=1)
    mtime = p1.stat().st_mtime
    p2 = run_oep_cascade(spec, atoms, ccsd_payload=cc,
                         cache_dir=tmp_path,
                         basis="def2-svp", grid_level=1)
    assert p2 == p1
    assert p2.stat().st_mtime == mtime, "OEP npz rewritten on cache hit"


def test_preflight_uks_oep_signature_and_imports():
    """Fast structural test: preflight_uks_oep is importable, kw-only, and
    its smoke_specs use SpeciesEntry(name, charge, spin, source) order
    with the documented HO doublet + HN triplet pair.

    This test does NOT execute the function -- execution is covered by
    scripts/smoke_preflight_uks_oep.py (run manually) and by the
    @pytest.mark.slow integration test below (deselected by default per
    setup.cfg addopts='-m "not slow"').

    Note: assertions are whitespace- and quote-style-sensitive against the literal source text; if a code formatter is applied to external_refs.py, update these strings to match.
    """
    import inspect
    from xcquinox.alec.external_refs import preflight_uks_oep, SpeciesEntry

    sig = inspect.signature(preflight_uks_oep)
    params = sig.parameters

    assert list(params) == ["cache_dir", "basis", "grid_level"], (
        f"signature drift: got {list(params)}"
    )
    for name in ("cache_dir", "basis", "grid_level"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, (
            f"{name} must be keyword-only (matches T3/T4/T5 contract)"
        )
    assert params["basis"].default == "def2-svp"
    assert params["grid_level"].default == 1

    src = inspect.getsource(preflight_uks_oep)
    assert 'SpeciesEntry("HO", 0, 1, "dfs_ae")' in src, (
        "HO smoke spec must be (name='HO', charge=0, spin=1, source='dfs_ae')"
    )
    assert 'SpeciesEntry("HN", 0, 2, "dfs_ae")' in src, (
        "HN smoke spec must be (name='HN', charge=0, spin=2, source='dfs_ae')"
    )


@pytest.mark.slow
def test_preflight_uks_runs_ho_and_hn(tmp_path):
    """SLOW: runs full HO+HN SCF->CCSD->2-tier OEP cascade (~10-30 min).

    Deselected by default via setup.cfg `addopts = -m "not slow"`. Run
    explicitly with `pytest -m slow` if you want pytest to drive it; for
    a more diagnostic run with progress heartbeat use:
        python scripts/smoke_preflight_uks_oep.py --cache-dir /tmp/smoke
    """
    from xcquinox.alec.external_refs import preflight_uks_oep
    preflight_uks_oep(cache_dir=tmp_path, basis="def2-svp", grid_level=1)
    assert (tmp_path / "HO.npz").is_file()
    assert (tmp_path / "HN.npz").is_file()
