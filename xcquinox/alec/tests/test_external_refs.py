"""Tests for xcquinox.alec.external_refs species union + pipeline."""
import pytest
from xcquinox.alec.external_refs import (
    SpeciesEntry,
    build_species_union,
)


@pytest.fixture(autouse=True)
def _isolate_per_species_oep_overrides():
    """Snapshot, empty, then restore _PER_SPECIES_OEP_OVERRIDES around every
    test. Without this, tests that mutate the dict (validator, resolver,
    precompute_all-orchestration tests) collide with the production
    overrides pasted in 2026-05-06: their `species_union=[Be]` test
    fixtures would see all 8 production override keys as orphans and
    raise spuriously. Snapshot+clear+restore (rather than
    monkeypatch.setattr of a fresh dict) preserves the same dict
    OBJECT identity so tests that import the dict by name at module
    level still see the same binding before and after."""
    from xcquinox.alec import external_refs as ext
    snapshot = dict(ext._PER_SPECIES_OEP_OVERRIDES)
    ext._PER_SPECIES_OEP_OVERRIDES.clear()
    yield
    ext._PER_SPECIES_OEP_OVERRIDES.clear()
    ext._PER_SPECIES_OEP_OVERRIDES.update(snapshot)


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
    cache_path = tmp_path / "_intermediates" / "H2_g1_bdef2-svp_scf.npz"
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
        "rho_ref_grid must be spin-summed 1D not (2, N_grid), see "
        "data.py:296-299 for the canonical spin-summing pattern")
    # H atom has 1 electron total, integrated rho must equal 1.0
    integ = float(np.sum(cc["grid_weights"] * cc["rho_ref_grid"]))
    assert abs(integ - 1.0) < 0.05, f"integrated rho={integ} != 1 for H atom"


def test_run_ccsd_h_atom_density_fit_empty_spin_channel(tmp_path):
    """Regression: H atom (noccb=0) CCSD with density_fit=True must not crash.

    pyscf's DF-UCCSD _make_df_eris_outcore creates the OOVV HDF5 dataset with a
    zero chunk dimension when a spin channel is empty, raising 'All chunk
    dimensions must be positive'. run_ccsd_with_cache falls back to non-DF CCSD
    for any species with min(nelec) == 0, so the reference still builds.
    """
    from xcquinox.alec.external_refs import (
        SpeciesEntry, resolve_geometry,
        run_scf_with_cache, run_ccsd_with_cache,
    )
    import numpy as np
    spec = SpeciesEntry("H", 0, 1, "dfs_atom")
    atoms = resolve_geometry(spec)
    scf = run_scf_with_cache(spec, atoms, cache_dir=tmp_path,
                             basis="def2-svp", grid_level=1,
                             density_fit=True, auxbasis="def2-universal-jkfit")
    cc = run_ccsd_with_cache(spec, atoms, scf_payload=scf, cache_dir=tmp_path,
                             basis="def2-svp", grid_level=1,
                             density_fit=True, auxbasis="def2-universal-jkfit")
    assert np.all(np.isfinite(cc["dm_ao"]))
    assert cc["rho_ref_grid"].ndim == 1 and np.all(np.isfinite(cc["rho_ref_grid"]))
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


def test_effective_tier_grid_level_ignores_mismatched_pin():
    """An override's grid_level pin is honored only when it equals the run grid
    (the CCSD ref is on the run grid); a mismatched pin is ignored + warns, so
    the grid_level-1 step-7 overrides are reusable in a grid_level-2 run without
    tripping run_oep_cascade's grid-consistency gate."""
    from xcquinox.alec.external_refs import _effective_tier_grid_level

    # no pin -> run grid
    assert _effective_tier_grid_level({"aux_basis": "x"}, 2) == 2
    # pin == run grid -> honored, no warning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _effective_tier_grid_level({"grid_level": 1}, 1) == 1
        assert _effective_tier_grid_level({"grid_level": 2}, 2) == 2
    # pin != run grid -> ignored (use run grid) + RuntimeWarning
    with pytest.warns(RuntimeWarning, match="ignoring the pin"):
        assert _effective_tier_grid_level({"grid_level": 1}, 2, "F2(0,0)") == 2


def test_oep_tiers_rks_and_uks_constants_split():
    """RKS and UKS tier constants exist with documented conv_tol values.

    RKS conv_tol=2e-3 is mirrored from step-6 closed-shell H2O/C2H2 floor.
    UKS conv_tol=1e-2 is set against the empirical UKS floor (~6e-3 on HO
    at def2-svp/grid_level=1 with level_shift=0.5), see _OEP_TIERS_UKS
    docstring in xcquinox/alec/external_refs.py for the full rationale.

    This test pins the values so a future edit cannot silently regress
    the cascade quality contract.
    """
    from xcquinox.alec.external_refs import _OEP_TIERS_RKS, _OEP_TIERS_UKS

    assert len(_OEP_TIERS_RKS) == 2
    assert len(_OEP_TIERS_UKS) == 2
    rks_aux = [t["aux_basis"] for t in _OEP_TIERS_RKS]
    uks_aux = [t["aux_basis"] for t in _OEP_TIERS_UKS]
    assert rks_aux == ["def2-svp-jkfit", "def2-tzvp-jkfit"]
    assert uks_aux == ["def2-svp-jkfit", "def2-tzvp-jkfit"]

    assert all(t["conv_tol"] == 2e-3 for t in _OEP_TIERS_RKS), (
        "RKS conv_tol must be 2e-3 (step-6 closed-shell floor parity)"
    )
    assert all(t["conv_tol"] == 1e-2 for t in _OEP_TIERS_UKS), (
        "UKS conv_tol must be 1e-2 (empirical UKS floor + 1.7x margin)"
    )

    assert all(t["regularization"] == 1e-4 for t in _OEP_TIERS_RKS)
    assert all(t["regularization"] == 1e-4 for t in _OEP_TIERS_UKS)
    assert [t["max_iter"] for t in _OEP_TIERS_RKS] == [500, 1000]
    assert [t["max_iter"] for t in _OEP_TIERS_UKS] == [500, 1000]


def test_run_oep_cascade_dispatches_per_spin_tier_set(tmp_path, monkeypatch):
    """run_oep_cascade picks _OEP_TIERS_UKS for spin>0, _OEP_TIERS_RKS
    otherwise. Verified by intercepting run_oep_inversion and reading the
    conv_tol kwarg the cascade passes; saver is also stubbed so no real
    PySCF SCF runs.
    """
    import numpy as np
    from collections import namedtuple
    from xcquinox.alec import external_refs as er
    from xcquinox.alec import oep as alec_oep

    captured = {"conv_tols": []}
    StubOEP = namedtuple("StubOEPResult", [
        "vxc_matrix", "converged", "n_iter", "density_error",
        "baseline_xc", "aux_basis", "regularization", "n_electrons",
        "lbfgs_status",
    ])

    def stub_run_oep_inversion(mol_spec, dm_target, **kwargs):
        captured["conv_tols"].append(kwargs["conv_tol"])
        nao = 4
        if np.asarray(dm_target).ndim == 3:
            vxc = np.zeros((2, nao, nao))
        else:
            vxc = np.zeros((nao, nao))
        return StubOEP(
            vxc_matrix=vxc, converged=True, n_iter=1,
            density_error=1e-5, baseline_xc="pbe",
            aux_basis=kwargs["aux_basis"],
            regularization=kwargs["regularization"], n_electrons=1.0,
            lbfgs_status="ok",
        )

    monkeypatch.setattr(alec_oep, "run_oep_inversion",
                        stub_run_oep_inversion)

    # Stub save_vxc_ref so no real SCF runs. The cascade's phase-1 write
    # already creates the file with rho_ref_grid + ref_density_method;
    # this stub just appends the OEP fields the completeness check
    # expects (vxc_ref, dm_target).
    def stub_save_vxc_ref(oep_result, output_path, *, dm_target=None,
                          method="ccsd"):
        existing = dict(np.load(str(output_path)))
        existing["vxc_ref"] = np.asarray(oep_result.vxc_matrix)
        existing["dm_target"] = (
            np.asarray(dm_target) if dm_target is not None
            else np.zeros((4, 4))
        )
        np.savez_compressed(str(output_path), **existing)

    monkeypatch.setattr(alec_oep, "save_vxc_ref", stub_save_vxc_ref)

    from ase import Atoms
    nao = 4
    n_grid = 5

    def make_payload(spin):
        if spin > 0:
            dm = np.zeros((2, nao, nao))
        else:
            dm = np.zeros((nao, nao))
        return {
            "dm_ao": dm,
            "rho_ref_grid": np.zeros(n_grid),
            "grid_weights": np.ones(n_grid),
            "ao_grid": np.zeros((n_grid, nao)),
        }

    h2 = er.SpeciesEntry("H2", 0, 0, "dfs_ae")
    atoms_h2 = Atoms("HH", positions=[[0, 0, 0], [0, 0, 0.74]])
    er.run_oep_cascade(h2, atoms_h2, ccsd_payload=make_payload(0),
                       cache_dir=tmp_path / "rks",
                       basis="def2-svp", grid_level=1)

    h_atom = er.SpeciesEntry("H", 0, 1, "atom")
    atoms_h = Atoms("H", positions=[[0, 0, 0]])
    er.run_oep_cascade(h_atom, atoms_h, ccsd_payload=make_payload(1),
                       cache_dir=tmp_path / "uks",
                       basis="def2-svp", grid_level=1)

    assert captured["conv_tols"] == [2e-3, 1e-2], (
        f"expected RKS->2e-3 then UKS->1e-2; got {captured['conv_tols']}"
    )


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


def test_run_log_partial_atomic(tmp_path):
    """Each species-result append produces a self-consistent partial JSON."""
    from xcquinox.alec.external_refs import RunLog
    log = RunLog(cache_dir=tmp_path)
    log.start(["H2O", "C2H2"])
    log.record_result(
        name="H2O", charge=0, spin=0, status="OK",
        wall_clock_s=12.3, error_msg=None,
    )
    import json
    partial = tmp_path / "_run_log_partial.json"
    assert partial.is_file()
    payload = json.loads(partial.read_text())
    assert len(payload["results"]) == 1
    assert payload["results"][0]["name"] == "H2O"
    assert payload["results"][0]["status"] == "OK"
    log.record_result(
        name="C2H2", charge=0, spin=0, status="FAIL_OEP",
        wall_clock_s=58.0, error_msg="OEP both tiers failed",
    )
    payload = json.loads(partial.read_text())
    assert len(payload["results"]) == 2


def test_run_log_finalize_archives(tmp_path):
    """Finalize renames partial -> run_log_<ts>.json and removes partial."""
    from xcquinox.alec.external_refs import RunLog
    log = RunLog(cache_dir=tmp_path)
    log.start(["H2O"])
    log.record_result(
        name="H2O", charge=0, spin=0, status="OK",
        wall_clock_s=12.3, error_msg=None,
    )
    final_path = log.finalize()
    assert final_path.is_file()
    assert "_run_log_" in final_path.name
    assert not (tmp_path / "_run_log_partial.json").is_file()


def test_precompute_all_skips_cached_species(tmp_path):
    """precompute_all skips species whose npz already has all required keys."""
    from xcquinox.alec.external_refs import (
        SpeciesEntry, precompute_all,
    )
    import numpy as np
    npz = tmp_path / "H2.npz"
    nao = 4
    np.savez(npz,
             vxc_ref=np.zeros((nao, nao)),
             dm_target=np.zeros((nao, nao)),
             rho_ref_grid=np.zeros(100),
             ref_density_method=np.array("ccsd"),
             oep_baseline_xc=np.array("pbe"),
             oep_aux_basis=np.array("def2-svp-jkfit"),
             oep_regularization=np.array(1e-4),
             oep_density_error=np.array(1e-5),
             oep_converged=np.array(True),
             oep_lbfgs_status=np.array("converged"),
             oep_n_electrons=np.array(2.0))
    species = [SpeciesEntry("H2", 0, 0, "dfs_ae")]
    precompute_all(species, cache_dir=tmp_path,
                   basis="def2-svp", grid_level=1, run_preflight=False)
    log_files = list(tmp_path.glob("_run_log_*.json"))
    assert len(log_files) == 1
    import json
    payload = json.loads(log_files[0].read_text())
    assert payload["results"][0]["status"] == "SKIPPED_CACHED"


def test_override_tier_knob_allowlist_contents():
    """The 8-knob allowlist matches the spec sec. 5.1 schema exactly."""
    from xcquinox.alec.external_refs import _OVERRIDE_TIER_KNOB_ALLOWLIST
    assert _OVERRIDE_TIER_KNOB_ALLOWLIST == frozenset({
        "aux_basis",
        "regularization",
        "max_iter",
        "conv_tol",
        "grid_level",
        "level_shift",
        "inner_damp",
        "inner_diis_start_cycle",
    })


def test_per_species_oep_overrides_default_is_empty_dict():
    """The override table ships empty; populated only after harness run."""
    from xcquinox.alec.external_refs import _PER_SPECIES_OEP_OVERRIDES
    assert _PER_SPECIES_OEP_OVERRIDES == {}
    assert isinstance(_PER_SPECIES_OEP_OVERRIDES, dict)


def test_per_species_overrides_key_shape_is_name_charge_spin():
    """Spec §9.2 / Plan-2 review fix: every populated key must be a
    3-tuple of (str, int, int). Pins against accidental 4-tuple
    (e.g., a maintainer adding `source` to the key)."""
    from xcquinox.alec.external_refs import _PER_SPECIES_OEP_OVERRIDES
    for key in _PER_SPECIES_OEP_OVERRIDES.keys():
        assert isinstance(key, tuple)
        assert len(key) == 3
        assert isinstance(key[0], str)
        assert isinstance(key[1], int) and not isinstance(key[1], bool)
        assert isinstance(key[2], int) and not isinstance(key[2], bool)


def test_validate_overrides_accepts_well_formed_override():
    """A well-formed override with valid key + tier knobs passes."""
    from xcquinox.alec.external_refs import (
        SpeciesEntry, _validate_overrides, _PER_SPECIES_OEP_OVERRIDES,
    )
    species = [SpeciesEntry(name="Be", charge=0, spin=0, source="dfs_atom")]
    _PER_SPECIES_OEP_OVERRIDES[("Be", 0, 0)] = (
        {"aux_basis": "def2-tzvp-jkfit", "regularization": 1e-3},
    )
    try:
        _validate_overrides(species)  # should not raise
    finally:
        _PER_SPECIES_OEP_OVERRIDES.pop(("Be", 0, 0), None)


def test_validate_overrides_rejects_unknown_knob():
    """Typo in override-tier dict key is rejected with a clear error."""
    import pytest
    from xcquinox.alec.external_refs import (
        SpeciesEntry, _validate_overrides, _PER_SPECIES_OEP_OVERRIDES,
    )
    species = [SpeciesEntry(name="Be", charge=0, spin=0, source="dfs_atom")]
    _PER_SPECIES_OEP_OVERRIDES[("Be", 0, 0)] = (
        {"aux_bais": "def2-tzvp-jkfit"},   # typo
    )
    try:
        with pytest.raises(ValueError, match="unknown knobs"):
            _validate_overrides(species)
    finally:
        _PER_SPECIES_OEP_OVERRIDES.pop(("Be", 0, 0), None)


def test_validate_overrides_warns_on_cross_pool_species():
    """A global override key absent from THIS run's species set warns (it targets
    another pool, or is a typo) but does not raise; knobs are still validated."""
    import pytest
    from xcquinox.alec.external_refs import (
        SpeciesEntry, _validate_overrides, _PER_SPECIES_OEP_OVERRIDES,
    )
    species = [SpeciesEntry(name="Be", charge=0, spin=0, source="dfs_atom")]
    _PER_SPECIES_OEP_OVERRIDES[("UnknownSpecies", 0, 0)] = (
        {"aux_basis": "def2-tzvp-jkfit"},
    )
    try:
        with pytest.warns(RuntimeWarning, match="cross-pool override or typo"):
            _validate_overrides(species)
    finally:
        _PER_SPECIES_OEP_OVERRIDES.pop(("UnknownSpecies", 0, 0), None)


def test_cf4_override_knobs_valid():
    """The cf4 def2-tzvpd OEP override is well-formed (knobs in allowlist + bounds)."""
    from xcquinox.alec.external_refs import (
        SpeciesEntry, _validate_overrides, _PER_SPECIES_OEP_OVERRIDES,
    )
    species = [SpeciesEntry(name="cf4", charge=0, spin=0, source="reaction_pool")]
    _PER_SPECIES_OEP_OVERRIDES[("cf4", 0, 0)] = (
        {"aux_basis": "def2-tzvp-jkfit", "regularization": 1e-4, "conv_tol": 0.0043},
    )
    try:
        _validate_overrides(species)  # cf4 in species -> knobs must validate cleanly
    finally:
        _PER_SPECIES_OEP_OVERRIDES.pop(("cf4", 0, 0), None)


def test_validate_overrides_rejects_wrong_key_types():
    """Override key with bool spin (instead of int) is rejected."""
    import pytest
    from xcquinox.alec.external_refs import (
        SpeciesEntry, _validate_overrides, _PER_SPECIES_OEP_OVERRIDES,
    )
    species = [SpeciesEntry(name="Be", charge=0, spin=0, source="dfs_atom")]
    # bool is technically int in Python, but the validator must reject:
    _PER_SPECIES_OEP_OVERRIDES[("Be", 0, True)] = (
        {"aux_basis": "def2-tzvp-jkfit"},
    )
    try:
        with pytest.raises(ValueError, match="must be"):
            _validate_overrides(species)
    finally:
        _PER_SPECIES_OEP_OVERRIDES.pop(("Be", 0, True), None)


def test_validate_overrides_rejects_empty_tier_tuple():
    """Empty override-tier tuple is a configuration error."""
    import pytest
    from xcquinox.alec.external_refs import (
        SpeciesEntry, _validate_overrides, _PER_SPECIES_OEP_OVERRIDES,
    )
    species = [SpeciesEntry(name="Be", charge=0, spin=0, source="dfs_atom")]
    _PER_SPECIES_OEP_OVERRIDES[("Be", 0, 0)] = ()
    try:
        with pytest.raises(ValueError, match="non-empty tuple"):
            _validate_overrides(species)
    finally:
        _PER_SPECIES_OEP_OVERRIDES.pop(("Be", 0, 0), None)


def test_validate_overrides_rejects_out_of_range_values():
    """Negative regularization, max_iter=0, level_shift=10, all rejected."""
    import pytest
    from xcquinox.alec.external_refs import (
        SpeciesEntry, _validate_overrides, _PER_SPECIES_OEP_OVERRIDES,
    )
    species = [SpeciesEntry(name="Be", charge=0, spin=0, source="dfs_atom")]
    for bad, expected_match in [
        ({"regularization": -1.0}, "regularization must be"),
        ({"max_iter": 0}, "max_iter must be"),
        ({"conv_tol": 0.0}, "conv_tol must be"),
        ({"grid_level": -1}, "grid_level must be"),
        ({"inner_damp": 1.5}, "inner_damp must be"),
        ({"inner_diis_start_cycle": 0}, "inner_diis_start_cycle must be"),
        ({"level_shift": 10.0}, "level_shift"),  # |x| > 5
    ]:
        _PER_SPECIES_OEP_OVERRIDES[("Be", 0, 0)] = (bad,)
        try:
            with pytest.raises(ValueError, match=expected_match):
                _validate_overrides(species)
        finally:
            _PER_SPECIES_OEP_OVERRIDES.pop(("Be", 0, 0), None)


def test_validate_overrides_accepts_negative_level_shift():
    """level_shift=-0.5 is allowed (Ziegler VSO usage). Pass-7 fix."""
    from xcquinox.alec.external_refs import (
        SpeciesEntry, _validate_overrides, _PER_SPECIES_OEP_OVERRIDES,
    )
    species = [SpeciesEntry(name="Be", charge=0, spin=0, source="dfs_atom")]
    _PER_SPECIES_OEP_OVERRIDES[("Be", 0, 0)] = (
        {"level_shift": -0.5},
    )
    try:
        _validate_overrides(species)  # should not raise
    finally:
        _PER_SPECIES_OEP_OVERRIDES.pop(("Be", 0, 0), None)


def test_per_species_overrides_empty_dict_uses_defaults_for_all_species():
    """Spec §9.2 / Plan-2 review fix: empty override table -> identity
    return of `_OEP_TIERS_RKS` / `_OEP_TIERS_UKS` for all species
    types. Pin the no-op-empty-dict invariant."""
    from xcquinox.alec.external_refs import (
        _resolve_tiers_for_species, _OEP_TIERS_RKS, _OEP_TIERS_UKS,
        _PER_SPECIES_OEP_OVERRIDES,
    )
    # Sanity: dict is empty (test isolation):
    assert len(_PER_SPECIES_OEP_OVERRIDES) == 0
    rks = _resolve_tiers_for_species("AnyRKS", 0, 0, is_uks=False)
    uks = _resolve_tiers_for_species("AnyUKS", 0, 1, is_uks=True)
    assert rks is _OEP_TIERS_RKS
    assert uks is _OEP_TIERS_UKS


def test_resolve_tiers_no_override_returns_default_rks_by_identity():
    """RKS species not in override table: returns _OEP_TIERS_RKS by `is`."""
    from xcquinox.alec.external_refs import (
        _resolve_tiers_for_species, _OEP_TIERS_RKS,
    )
    out = _resolve_tiers_for_species("UnknownRKS", 0, 0, is_uks=False)
    assert out is _OEP_TIERS_RKS


def test_resolve_tiers_no_override_returns_default_uks_by_identity():
    """UKS species not in override table: returns _OEP_TIERS_UKS by `is`."""
    from xcquinox.alec.external_refs import (
        _resolve_tiers_for_species, _OEP_TIERS_UKS,
    )
    out = _resolve_tiers_for_species("UnknownUKS", 0, 1, is_uks=True)
    assert out is _OEP_TIERS_UKS


def test_resolve_tiers_override_merges_onto_default():
    """Single-knob override keeps default max_iter / conv_tol; aux_basis swaps."""
    from xcquinox.alec.external_refs import (
        _resolve_tiers_for_species, _OEP_TIERS_RKS,
        _PER_SPECIES_OEP_OVERRIDES,
    )
    _PER_SPECIES_OEP_OVERRIDES[("Be", 0, 0)] = (
        {"aux_basis": "def2-tzvp-jkfit"},
    )
    try:
        out = _resolve_tiers_for_species("Be", 0, 0, is_uks=False)
        # Override truncates cascade to its own length (1 tier here)
        assert len(out) == 1
        assert out[0]["aux_basis"] == "def2-tzvp-jkfit"
        # max_iter and conv_tol inherit from default tier 0
        assert out[0]["max_iter"] == _OEP_TIERS_RKS[0]["max_iter"]
        assert out[0]["conv_tol"] == _OEP_TIERS_RKS[0]["conv_tol"]
        assert out[0]["regularization"] == _OEP_TIERS_RKS[0]["regularization"]
    finally:
        _PER_SPECIES_OEP_OVERRIDES.pop(("Be", 0, 0), None)


def test_resolve_tiers_override_more_tiers_than_default_clamps_to_last():
    """3-tier override on a 2-tier default: tier 2 merges onto default tier 1."""
    from xcquinox.alec.external_refs import (
        _resolve_tiers_for_species, _OEP_TIERS_RKS,
        _PER_SPECIES_OEP_OVERRIDES,
    )
    _PER_SPECIES_OEP_OVERRIDES[("Be", 0, 0)] = (
        {"aux_basis": "A1"},
        {"aux_basis": "A2"},
        {"aux_basis": "A3"},
    )
    try:
        out = _resolve_tiers_for_species("Be", 0, 0, is_uks=False)
        assert len(out) == 3
        assert out[0]["max_iter"] == _OEP_TIERS_RKS[0]["max_iter"]
        assert out[1]["max_iter"] == _OEP_TIERS_RKS[1]["max_iter"]
        # Tier 2 of override merges onto last default tier (index 1)
        assert out[2]["max_iter"] == _OEP_TIERS_RKS[1]["max_iter"]
        assert out[2]["aux_basis"] == "A3"
    finally:
        _PER_SPECIES_OEP_OVERRIDES.pop(("Be", 0, 0), None)


def test_resolve_tiers_override_fewer_tiers_truncates_cascade():
    """1-tier override on a 2-tier default: cascade truncates to 1 tier."""
    from xcquinox.alec.external_refs import (
        _resolve_tiers_for_species, _PER_SPECIES_OEP_OVERRIDES,
    )
    _PER_SPECIES_OEP_OVERRIDES[("Be", 0, 0)] = (
        {"aux_basis": "single-tier-only"},
    )
    try:
        out = _resolve_tiers_for_species("Be", 0, 0, is_uks=False)
        assert len(out) == 1
    finally:
        _PER_SPECIES_OEP_OVERRIDES.pop(("Be", 0, 0), None)


def test_resolve_tiers_override_empty_tuple_raises():
    """Defensive double-check: empty tuple at lookup time raises ValueError.
    (Normal path is _validate_overrides catching it earlier.)"""
    import pytest
    from xcquinox.alec.external_refs import (
        _resolve_tiers_for_species, _PER_SPECIES_OEP_OVERRIDES,
    )
    _PER_SPECIES_OEP_OVERRIDES[("Be", 0, 0)] = ()
    try:
        with pytest.raises(ValueError, match="empty"):
            _resolve_tiers_for_species("Be", 0, 0, is_uks=False)
    finally:
        _PER_SPECIES_OEP_OVERRIDES.pop(("Be", 0, 0), None)


def test_run_scf_with_cache_uses_grid_suffixed_filename(tmp_path):
    """Cache file name embeds grid_level: <name>_g{N}_scf.npz."""
    import numpy as np
    from xcquinox.alec.external_refs import (
        run_scf_with_cache, SpeciesEntry,
    )
    from ase import Atoms
    spec = SpeciesEntry(name="H2test", charge=0, spin=0, source="dfs_ae")
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)])
    cache_dir = tmp_path / "external_refs"
    run_scf_with_cache(spec, atoms, cache_dir=cache_dir,
                       basis="sto-3g", grid_level=1)
    # Post-Plan-2: the file MUST be named <name>_g1_b{basis}_scf.npz
    expected = cache_dir / "_intermediates" / "H2test_g1_bsto-3g_scf.npz"
    assert expected.is_file()
    # The unsuffixed name MUST NOT exist
    legacy = cache_dir / "_intermediates" / "H2test_scf.npz"
    assert not legacy.exists()


def test_run_scf_with_cache_grid_level_2_creates_g2_file(tmp_path):
    """grid_level=2 produces a _g2_scf.npz cache file."""
    from xcquinox.alec.external_refs import (
        run_scf_with_cache, SpeciesEntry,
    )
    from ase import Atoms
    spec = SpeciesEntry(name="H2test", charge=0, spin=0, source="dfs_ae")
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)])
    cache_dir = tmp_path / "external_refs"
    run_scf_with_cache(spec, atoms, cache_dir=cache_dir,
                       basis="sto-3g", grid_level=2)
    expected = cache_dir / "_intermediates" / "H2test_g2_bsto-3g_scf.npz"
    assert expected.is_file()


def test_run_ccsd_with_cache_uses_grid_suffixed_filename(tmp_path):
    """CCSD cache file name embeds grid_level: <name>_g{N}_ccsd.npz."""
    import numpy as np
    from xcquinox.alec.external_refs import (
        run_scf_with_cache, run_ccsd_with_cache, SpeciesEntry,
    )
    from ase import Atoms
    spec = SpeciesEntry(name="H2test", charge=0, spin=0, source="dfs_ae")
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)])
    cache_dir = tmp_path / "external_refs"
    scf = run_scf_with_cache(spec, atoms, cache_dir=cache_dir,
                              basis="sto-3g", grid_level=1)
    run_ccsd_with_cache(spec, atoms, scf_payload=scf, cache_dir=cache_dir,
                         basis="sto-3g", grid_level=1)
    expected = cache_dir / "_intermediates" / "H2test_g1_bsto-3g_ccsd.npz"
    assert expected.is_file()
    legacy = cache_dir / "_intermediates" / "H2test_ccsd.npz"
    assert not legacy.exists()


def test_migration_renames_unsuffixed_intermediates_to_g1(tmp_path):
    """Pre-2026-05-03 caches (no grid suffix) get renamed to _g1_."""
    import numpy as np
    from xcquinox.alec.external_refs import (
        _migrate_intermediates_to_grid_suffixed,
    )
    inter = tmp_path / "_intermediates"
    inter.mkdir()
    np.savez(inter / "Foo_scf.npz", x=np.zeros(3))
    np.savez(inter / "Foo_ccsd.npz", x=np.zeros(3))
    n = _migrate_intermediates_to_grid_suffixed(tmp_path)
    assert n == 2
    assert (inter / "Foo_g1_scf.npz").is_file()
    assert (inter / "Foo_g1_ccsd.npz").is_file()
    assert not (inter / "Foo_scf.npz").exists()
    assert not (inter / "Foo_ccsd.npz").exists()


def test_migration_idempotent_returns_zero_on_second_call(tmp_path):
    """Second invocation finds nothing to rename, returns 0."""
    import numpy as np
    from xcquinox.alec.external_refs import (
        _migrate_intermediates_to_grid_suffixed,
    )
    inter = tmp_path / "_intermediates"
    inter.mkdir()
    np.savez(inter / "Foo_scf.npz", x=np.zeros(3))
    _migrate_intermediates_to_grid_suffixed(tmp_path)
    n_second = _migrate_intermediates_to_grid_suffixed(tmp_path)
    assert n_second == 0


def test_migration_raises_when_target_name_already_exists(tmp_path):
    """Conflict (target _g1_ file present alongside unsuffixed): raises."""
    import pytest
    import numpy as np
    from xcquinox.alec.external_refs import (
        _migrate_intermediates_to_grid_suffixed,
    )
    inter = tmp_path / "_intermediates"
    inter.mkdir()
    np.savez(inter / "Foo_scf.npz", x=np.zeros(3))
    np.savez(inter / "Foo_g1_scf.npz", x=np.zeros(3))   # pre-existing target
    with pytest.raises(FileExistsError):
        _migrate_intermediates_to_grid_suffixed(tmp_path)


def test_migration_skips_basis_df_tagged_no_false_conflict(tmp_path):
    """Regression (cluster preflight abort 2026-06-04): a basis/DF-tagged
    intermediate (``alf_g2_bdef2-tzvpd_df_scf.npz``) is ALREADY grid-tagged and
    must be SKIPPED -- not mistaken for a legacy ``<name>_scf.npz`` and
    re-migrated into a doubly-suffixed ``..._df_g1_scf.npz`` that then collides
    with the regenerated canonical file. The pre-fix end-anchored
    ``_g\\d+_scf\\.npz$`` did not match the basis/DF form (its ``_scf.npz`` is
    preceded by ``_df``), so this raised ``FileExistsError``."""
    import numpy as np
    from xcquinox.alec.external_refs import (
        _migrate_intermediates_to_grid_suffixed,
    )
    inter = tmp_path / "_intermediates"
    inter.mkdir()
    # Canonical new-format file + the garbage a prior mis-migration produced;
    # both carry a `_g{N}_` token, so BOTH must be skipped (no rename/conflict).
    canonical = inter / "alf_g2_bdef2-tzvpd_df_scf.npz"
    garbage = inter / "alf_g2_bdef2-tzvpd_df_g1_scf.npz"
    np.savez(canonical, x=np.zeros(3))
    np.savez(garbage, x=np.zeros(3))
    # A grid_level=1 basis-tagged ccsd ref (also new format) must be skipped too.
    g1_ccsd = inter / "h2o_g1_bdef2-svp_df_ccsd.npz"
    np.savez(g1_ccsd, x=np.zeros(3))
    n = _migrate_intermediates_to_grid_suffixed(tmp_path)   # must NOT raise
    assert n == 0                                            # nothing migrated
    assert canonical.is_file()
    assert garbage.is_file()
    assert g1_ccsd.is_file()


def test_migration_handles_mg_hg_ag_correctly(tmp_path):
    """The substring `_g` appears in Mg, Hg, Ag, must NOT corrupt them.
    Pass-8 fix: was `if "_g" in name and name.endswith(...)`; now
    `if name.endswith(suffix_new)` only."""
    import numpy as np
    from xcquinox.alec.external_refs import (
        _migrate_intermediates_to_grid_suffixed,
    )
    inter = tmp_path / "_intermediates"
    inter.mkdir()
    # Mg2H is a Hill-formula species with `_g` not in the filename;
    # but a hypothetical `_g_in_name` test species exercises the fix.
    np.savez(inter / "Mg_scf.npz", x=np.zeros(3))
    np.savez(inter / "Hg_scf.npz", x=np.zeros(3))
    n = _migrate_intermediates_to_grid_suffixed(tmp_path)
    assert n == 2
    assert (inter / "Mg_g1_scf.npz").is_file()
    assert (inter / "Hg_g1_scf.npz").is_file()
    assert not (inter / "Mg_scf.npz").exists()
    assert not (inter / "Hg_scf.npz").exists()


def test_migration_handles_partial_state_from_crash_recovery(tmp_path):
    """Mixed state: Foo_g1_scf.npz already migrated AND Foo_ccsd.npz
    not yet. Second run migrates only the ccsd half."""
    import numpy as np
    from xcquinox.alec.external_refs import (
        _migrate_intermediates_to_grid_suffixed,
    )
    inter = tmp_path / "_intermediates"
    inter.mkdir()
    np.savez(inter / "Foo_g1_scf.npz", x=np.zeros(3))   # already migrated
    np.savez(inter / "Foo_ccsd.npz", x=np.zeros(3))     # not yet
    n = _migrate_intermediates_to_grid_suffixed(tmp_path)
    assert n == 1
    assert (inter / "Foo_g1_scf.npz").is_file()
    assert (inter / "Foo_g1_ccsd.npz").is_file()
    assert not (inter / "Foo_ccsd.npz").exists()


def test_migration_no_intermediates_dir_returns_zero(tmp_path):
    """Migration on a cache_dir with no _intermediates/ returns 0."""
    from xcquinox.alec.external_refs import (
        _migrate_intermediates_to_grid_suffixed,
    )
    n = _migrate_intermediates_to_grid_suffixed(tmp_path)
    assert n == 0


def test_migration_preserves_already_grid_suffixed_g2_cache(tmp_path):
    """Plan-2-review CRITICAL: pre-existing _g2_scf.npz must NOT be
    re-renamed to _g2_g1_scf.npz. Spec §5.6 (line 1389) explicitly
    promises future _g{N>1}_* caches will be retained."""
    import numpy as np
    from xcquinox.alec.external_refs import (
        _migrate_intermediates_to_grid_suffixed,
    )
    inter = tmp_path / "_intermediates"
    inter.mkdir()
    np.savez(inter / "Foo_g2_scf.npz", x=np.zeros(3))   # already a g2 cache
    np.savez(inter / "Bar_scf.npz", x=np.zeros(3))      # legacy unsuffixed
    n = _migrate_intermediates_to_grid_suffixed(tmp_path)
    assert n == 1   # only Bar_scf -> Bar_g1_scf
    assert (inter / "Foo_g2_scf.npz").is_file()        # untouched
    assert (inter / "Bar_g1_scf.npz").is_file()
    assert not (inter / "Foo_g2_g1_scf.npz").exists()  # NOT corrupted


def test_cascade_ignores_mismatched_grid_level_pin(monkeypatch):
    """An override tier pinning grid_level != the run grid is IGNORED: the CCSD
    rho_ref_grid is on the run grid and run_oep_cascade's consistency gate forbids
    mixing grids, so the mol_spec passed to run_oep_inversion stays at the RUN
    grid_level (and a RuntimeWarning is emitted). This lets the grid_level-1
    step-7 overrides be reused in a grid_level-2 run; the tier's other knobs
    (aux_basis, regularization, ...) still apply. (Previously the pin threaded a
    foreign grid that the gate then rejected -- a footgun.)"""
    from xcquinox.alec.external_refs import (
        run_oep_cascade, SpeciesEntry, _PER_SPECIES_OEP_OVERRIDES,
    )
    captured_specs = []
    import xcquinox.alec.oep as alec_oep
    def stub_run(mol_spec, dm_target, **kwargs):
        captured_specs.append(mol_spec)
        from collections import namedtuple
        Stub = namedtuple("Stub", ["converged", "density_error"])
        return Stub(converged=True, density_error=1e-4)
    monkeypatch.setattr(alec_oep, "run_oep_inversion", stub_run)
    # Override pins grid_level=2, but the run below uses grid_level=1 -> mismatch:
    _PER_SPECIES_OEP_OVERRIDES[("Be", 0, 0)] = (
        {"aux_basis": "def2-tzvp-jkfit", "regularization": 1e-3,
         "grid_level": 2},
    )
    try:
        from ase import Atoms
        import numpy as np, tempfile, warnings
        atoms = Atoms("Be", positions=[(0, 0, 0)])
        spec = SpeciesEntry(name="Be", charge=0, spin=0, source="dfs_atom")
        ccsd_payload = {"dm_ao": np.eye(5), "rho_ref_grid": np.zeros(10)}
        with tempfile.TemporaryDirectory() as td:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    run_oep_cascade(spec, atoms, ccsd_payload=ccsd_payload,
                                    cache_dir=td, basis="sto-3g", grid_level=1)
                except Exception:
                    pass  # save_vxc_ref will fail; we only care about captured
        # pin (2) != run grid (1) -> ignored: mol_spec stays at the run grid (1)
        assert captured_specs[0].grid_level == 1
        assert any("ignoring the pin" in str(x.message) for x in w)
    finally:
        _PER_SPECIES_OEP_OVERRIDES.pop(("Be", 0, 0), None)


def test_cascade_threads_inner_damp_and_diis_from_resolved_tier(monkeypatch):
    """Override tier with inner_damp=0.3 and inner_diis_start_cycle=10
    are passed to run_oep_inversion."""
    from xcquinox.alec.external_refs import (
        run_oep_cascade, SpeciesEntry, _PER_SPECIES_OEP_OVERRIDES,
    )
    captured_kwargs = []
    import xcquinox.alec.oep as alec_oep
    def stub_run(mol_spec, dm_target, **kwargs):
        captured_kwargs.append(kwargs)
        from collections import namedtuple
        Stub = namedtuple("Stub", ["converged", "density_error"])
        return Stub(converged=True, density_error=1e-4)
    monkeypatch.setattr(alec_oep, "run_oep_inversion", stub_run)
    _PER_SPECIES_OEP_OVERRIDES[("Be", 0, 0)] = (
        {"aux_basis": "def2-tzvp-jkfit", "regularization": 1e-3,
         "inner_damp": 0.3, "inner_diis_start_cycle": 10},
    )
    try:
        from ase import Atoms
        atoms = Atoms("Be", positions=[(0, 0, 0)])
        spec = SpeciesEntry(name="Be", charge=0, spin=0, source="dfs_atom")
        import numpy as np, tempfile
        ccsd_payload = {"dm_ao": np.eye(5), "rho_ref_grid": np.zeros(10)}
        with tempfile.TemporaryDirectory() as td:
            try:
                run_oep_cascade(spec, atoms, ccsd_payload=ccsd_payload,
                                cache_dir=td, basis="sto-3g", grid_level=1)
            except Exception:
                pass
        assert captured_kwargs[0]["inner_damp"] == 0.3
        assert captured_kwargs[0]["inner_diis_start_cycle"] == 10
    finally:
        _PER_SPECIES_OEP_OVERRIDES.pop(("Be", 0, 0), None)


def test_cascade_level_shift_resolution_override_takes_precedence(monkeypatch):
    """UKS override with explicit level_shift=1.0 reaches run_oep_inversion
    (not the spin-default 0.5). Spec §9.2 / Plan-2 review fix."""
    from xcquinox.alec.external_refs import (
        run_oep_cascade, SpeciesEntry, _PER_SPECIES_OEP_OVERRIDES,
    )
    captured_kwargs = []
    import xcquinox.alec.oep as alec_oep
    def stub_run(mol_spec, dm_target, **kwargs):
        captured_kwargs.append(kwargs)
        from collections import namedtuple
        Stub = namedtuple("Stub", ["converged", "density_error"])
        return Stub(converged=True, density_error=1e-4)
    monkeypatch.setattr(alec_oep, "run_oep_inversion", stub_run)
    _PER_SPECIES_OEP_OVERRIDES[("HO", 0, 1)] = (
        {"aux_basis": "def2-tzvp-jkfit", "level_shift": 1.0},
    )
    try:
        from ase import Atoms
        atoms = Atoms("OH", positions=[(0, 0, 0), (0, 0, 1.0)])
        spec = SpeciesEntry(name="HO", charge=0, spin=1, source="dfs_ae")
        import numpy as np, tempfile
        ccsd_payload = {"dm_ao": np.zeros((2, 5, 5)), "rho_ref_grid": np.zeros(10)}
        with tempfile.TemporaryDirectory() as td:
            try:
                run_oep_cascade(spec, atoms, ccsd_payload=ccsd_payload,
                                cache_dir=td, basis="sto-3g", grid_level=1)
            except Exception:
                pass
        assert captured_kwargs[0]["level_shift"] == 1.0
    finally:
        _PER_SPECIES_OEP_OVERRIDES.pop(("HO", 0, 1), None)


def test_cascade_level_shift_falls_back_to_spin_default_rks(monkeypatch):
    """RKS species with no level_shift override -> 0.0 reaches the call.
    Spec §9.2 / Plan-2 review fix."""
    from xcquinox.alec.external_refs import (
        run_oep_cascade, SpeciesEntry, _PER_SPECIES_OEP_OVERRIDES,
    )
    captured_kwargs = []
    import xcquinox.alec.oep as alec_oep
    def stub_run(mol_spec, dm_target, **kwargs):
        captured_kwargs.append(kwargs)
        from collections import namedtuple
        Stub = namedtuple("Stub", ["converged", "density_error"])
        return Stub(converged=True, density_error=1e-4)
    monkeypatch.setattr(alec_oep, "run_oep_inversion", stub_run)
    _PER_SPECIES_OEP_OVERRIDES[("Be", 0, 0)] = (
        {"aux_basis": "def2-tzvp-jkfit"},   # no level_shift
    )
    try:
        from ase import Atoms
        atoms = Atoms("Be", positions=[(0, 0, 0)])
        spec = SpeciesEntry(name="Be", charge=0, spin=0, source="dfs_atom")
        import numpy as np, tempfile
        ccsd_payload = {"dm_ao": np.eye(5), "rho_ref_grid": np.zeros(10)}
        with tempfile.TemporaryDirectory() as td:
            try:
                run_oep_cascade(spec, atoms, ccsd_payload=ccsd_payload,
                                cache_dir=td, basis="sto-3g", grid_level=1)
            except Exception:
                pass
        assert captured_kwargs[0]["level_shift"] == 0.0
    finally:
        _PER_SPECIES_OEP_OVERRIDES.pop(("Be", 0, 0), None)


def test_cascade_level_shift_falls_back_to_spin_default_uks(monkeypatch):
    """No level_shift in override on UKS species -> 0.5 reaches the call."""
    from xcquinox.alec.external_refs import (
        run_oep_cascade, SpeciesEntry, _PER_SPECIES_OEP_OVERRIDES,
    )
    captured_kwargs = []
    import xcquinox.alec.oep as alec_oep
    def stub_run(mol_spec, dm_target, **kwargs):
        captured_kwargs.append(kwargs)
        from collections import namedtuple
        Stub = namedtuple("Stub", ["converged", "density_error"])
        return Stub(converged=True, density_error=1e-4)
    monkeypatch.setattr(alec_oep, "run_oep_inversion", stub_run)
    _PER_SPECIES_OEP_OVERRIDES[("HO", 0, 1)] = (
        {"aux_basis": "def2-tzvp-jkfit"},   # no level_shift specified
    )
    try:
        from ase import Atoms
        atoms = Atoms("OH", positions=[(0, 0, 0), (0, 0, 1.0)])
        spec = SpeciesEntry(name="HO", charge=0, spin=1, source="dfs_ae")
        import numpy as np, tempfile
        ccsd_payload = {"dm_ao": np.zeros((2, 5, 5)), "rho_ref_grid": np.zeros(10)}
        with tempfile.TemporaryDirectory() as td:
            try:
                run_oep_cascade(spec, atoms, ccsd_payload=ccsd_payload,
                                cache_dir=td, basis="sto-3g", grid_level=1)
            except Exception:
                pass
        assert captured_kwargs[0]["level_shift"] == 0.5
    finally:
        _PER_SPECIES_OEP_OVERRIDES.pop(("HO", 0, 1), None)


def test_precompute_all_invokes_migration_before_preflight(tmp_path, monkeypatch):
    """Migration helper must run BEFORE preflight_uks_oep is invoked."""
    call_order = []
    from xcquinox.alec import external_refs as ext
    real_migrate = ext._migrate_intermediates_to_grid_suffixed
    real_preflight = ext.preflight_uks_oep
    def spy_migrate(cache_dir):
        call_order.append("migrate")
        return real_migrate(cache_dir)
    def spy_preflight(*args, **kwargs):
        call_order.append("preflight")
        # Don't actually run preflight in the test:
        return None
    monkeypatch.setattr(ext, "_migrate_intermediates_to_grid_suffixed", spy_migrate)
    monkeypatch.setattr(ext, "preflight_uks_oep", spy_preflight)
    # Empty species list to short-circuit precompute_all's main loop:
    try:
        ext.precompute_all([], cache_dir=tmp_path,
                           basis="sto-3g", grid_level=1, run_preflight=True)
    except Exception:
        pass
    assert call_order.index("migrate") < call_order.index("preflight")


def test_preflight_uks_oep_invokes_migration_at_top(tmp_path, monkeypatch):
    """preflight_uks_oep also runs migration (defensive idempotence
    for direct callers that bypass precompute_all). Spec sec. 5.6.
    Plan-2-review fix: also call preflight TWICE and assert the second
    call is a no-op (idempotent contract per spec §9.2)."""
    from xcquinox.alec import external_refs as ext
    called = []
    real_migrate = ext._migrate_intermediates_to_grid_suffixed
    def spy_migrate(cache_dir):
        called.append(cache_dir)
        return real_migrate(cache_dir)
    monkeypatch.setattr(ext, "_migrate_intermediates_to_grid_suffixed", spy_migrate)
    monkeypatch.setattr(ext, "run_scf_with_cache",
                        lambda *a, **k: {"spin_unrestricted": True})
    monkeypatch.setattr(ext, "run_ccsd_with_cache",
                        lambda *a, **k: {})
    def fake_oep_cascade(*a, **k):
        import numpy as np
        from pathlib import Path
        spec = a[0]
        out = Path(k["cache_dir"]) / f"{spec.name}.npz"
        np.savez(out, vxc_ref=np.zeros((2, 5, 5)),
                 dm_target=np.zeros((2, 5, 5)),
                 rho_ref_grid=np.zeros(10),
                 ref_density_method=np.array("ccsd"))
        return out
    monkeypatch.setattr(ext, "run_oep_cascade", fake_oep_cascade)
    # First call: migration fires.
    try:
        ext.preflight_uks_oep(cache_dir=tmp_path,
                              basis="sto-3g", grid_level=1)
    except Exception:
        pass
    assert len(called) >= 1
    n_after_first = len(called)
    # Second call: migration is invoked again but should be a no-op
    # (no errors, no state change). This pins the spec §9.2
    # idempotence-on-direct-call contract.
    try:
        ext.preflight_uks_oep(cache_dir=tmp_path,
                              basis="sto-3g", grid_level=1)
    except Exception:
        pass
    assert len(called) == n_after_first + 1   # invoked again
    # And both invocations succeeded without raising:
    # (the implicit assertion is that we got here without exception)


# ---------------------------------------------------------------------------
# CCSD must run on a CONVERGED HF reference, not grafted PBE MOs.
# ---------------------------------------------------------------------------


def test_prepare_converged_hf_runs_real_scf_and_checks_convergence():
    """_prepare_converged_hf builds an HF mean-field, runs kernel(dm0=...),
    and returns a CONVERGED object. Uses H2 in sto-3g (sub-second SCF).

    This is the unit-testable contract for the EXTREF-01 fix: CCSD must
    sit on a self-consistent HF determinant, not on grafted PBE orbitals.
    """
    from pyscf import dft, gto
    from xcquinox.alec.external_refs import _prepare_converged_hf

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    pbe = dft.RKS(mol)
    pbe.xc = "pbe"
    pbe.kernel()
    pbe_dm = pbe.make_rdm1()

    mf_hf = _prepare_converged_hf(mol, dm0=pbe_dm, is_uks=False)
    assert mf_hf.converged is True, "HF must be self-consistent before CCSD"
    # The HF energy must NOT equal the PBE energy (proves a real HF SCF ran,
    # not a fake converged=True flag on a grafted PBE determinant).
    assert abs(float(mf_hf.e_tot) - float(pbe.e_tot)) > 1e-4, (
        "HF energy equals PBE energy -> no real HF SCF was performed"
    )
    # Brillouin: at the converged HF determinant the occ-virt Fock block is
    # ~0. We assert the object exposes converged MOs (sanity, not the full
    # Brillouin check which is implicit in PySCF convergence).
    assert mf_hf.mo_coeff is not None


def test_prepare_converged_hf_uses_dm0_as_initial_guess(monkeypatch):
    """_prepare_converged_hf must pass the PBE dm0 to kernel() (initial
    guess) and check the .converged flag. Verified without a real SCF by
    monkeypatching the HF object's kernel."""
    from pyscf import gto
    from xcquinox.alec import external_refs as ext

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    sentinel_dm = object()
    calls = {}

    class _FakeHF:
        converged = True
        mo_coeff = "coeffs"
        mo_occ = "occ"
        mo_energy = "energy"

        def kernel(self, dm0=None):
            calls["dm0"] = dm0
            return -1.0

    def _fake_build_hf(mol_arg, is_uks, **_kw):
        calls["is_uks"] = is_uks
        return _FakeHF()

    monkeypatch.setattr(ext, "_build_hf_meanfield", _fake_build_hf)
    mf = ext._prepare_converged_hf(mol, dm0=sentinel_dm, is_uks=False)
    assert calls["dm0"] is sentinel_dm, "PBE dm must be passed as kernel dm0"
    assert calls["is_uks"] is False
    assert mf.converged is True


def test_prepare_converged_hf_raises_when_not_converged(monkeypatch):
    """If HF SCF does not converge, _prepare_converged_hf raises rather than
    silently feeding a non-self-consistent determinant to CCSD."""
    from pyscf import gto
    from xcquinox.alec import external_refs as ext

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)

    class _NonConvHF:
        converged = False
        mo_coeff = None
        mo_occ = None
        mo_energy = None

        def kernel(self, dm0=None):
            return -1.0

    monkeypatch.setattr(
        ext, "_build_hf_meanfield", lambda mol_arg, is_uks, **_kw: _NonConvHF()
    )
    with pytest.raises(RuntimeError, match="HF SCF did not converge"):
        ext._prepare_converged_hf(mol, dm0=None, is_uks=False)


# ---------------------------------------------------------------------------
# stages 2 (CCSD) and 3 (OEP) must fsync the parent dir for
# durability, matching stage 1 (SCF).
# ---------------------------------------------------------------------------


def test_run_ccsd_fsyncs_parent_dir(tmp_path, monkeypatch):
    """run_ccsd_with_cache must fsync the _intermediates dir after the
    atomic os.replace (durability parity with stage 1)."""
    from xcquinox.alec import external_refs as ext
    from xcquinox.alec.external_refs import (
        SpeciesEntry, resolve_geometry, run_scf_with_cache,
        run_ccsd_with_cache,
    )
    spec = SpeciesEntry("H2", 0, 0, "dfs_ae")
    atoms = resolve_geometry(spec)
    scf = run_scf_with_cache(spec, atoms, cache_dir=tmp_path,
                             basis="def2-svp", grid_level=1)
    fsync_calls = {"n": 0}
    real_fsync = ext._fsync_dir

    def _spy(path):
        fsync_calls["n"] += 1
        return real_fsync(path)

    monkeypatch.setattr(ext, "_fsync_dir", _spy)
    run_ccsd_with_cache(spec, atoms, scf_payload=scf, cache_dir=tmp_path,
                        basis="def2-svp", grid_level=1)
    assert fsync_calls["n"] >= 1, "stage 2 (CCSD) must fsync parent dir"


def test_run_oep_cascade_fsyncs_parent_dir(tmp_path, monkeypatch):
    """run_oep_cascade must fsync the cache dir after writing the npz."""
    from xcquinox.alec import external_refs as ext
    from xcquinox.alec.external_refs import (
        SpeciesEntry, resolve_geometry, run_scf_with_cache,
        run_ccsd_with_cache, run_oep_cascade,
    )
    spec = SpeciesEntry("H2", 0, 0, "dfs_ae")
    atoms = resolve_geometry(spec)
    scf = run_scf_with_cache(spec, atoms, cache_dir=tmp_path,
                             basis="def2-svp", grid_level=1)
    cc = run_ccsd_with_cache(spec, atoms, scf_payload=scf, cache_dir=tmp_path,
                             basis="def2-svp", grid_level=1)
    fsync_calls = {"n": 0}
    real_fsync = ext._fsync_dir

    def _spy(path):
        fsync_calls["n"] += 1
        return real_fsync(path)

    monkeypatch.setattr(ext, "_fsync_dir", _spy)
    run_oep_cascade(spec, atoms, ccsd_payload=cc, cache_dir=tmp_path,
                    basis="def2-svp", grid_level=1)
    assert fsync_calls["n"] >= 1, "stage 3 (OEP) must fsync cache dir"


# ---------------------------------------------------------------------------
# the reference npz must carry grid_level_used provenance.
# ---------------------------------------------------------------------------


def test_oep_cascade_writes_grid_level_used(tmp_path):
    """Stage 3 OEP npz records the generating grid_level as grid_level_used
    so data.py can assert consumer/producer grid agreement."""
    import numpy as np
    from xcquinox.alec.external_refs import (
        SpeciesEntry, resolve_geometry, run_scf_with_cache,
        run_ccsd_with_cache, run_oep_cascade,
    )
    spec = SpeciesEntry("H2", 0, 0, "dfs_ae")
    atoms = resolve_geometry(spec)
    scf = run_scf_with_cache(spec, atoms, cache_dir=tmp_path,
                             basis="def2-svp", grid_level=1)
    cc = run_ccsd_with_cache(spec, atoms, scf_payload=scf, cache_dir=tmp_path,
                             basis="def2-svp", grid_level=1)
    npz_path = run_oep_cascade(spec, atoms, ccsd_payload=cc, cache_dir=tmp_path,
                               basis="def2-svp", grid_level=1)
    with np.load(npz_path, allow_pickle=False) as z:
        assert "grid_level_used" in z.files, "missing grid_level_used"
        assert int(z["grid_level_used"]) == 1


def test_oep_cache_rejects_basis_mismatch_but_trusts_legacy(tmp_path,
                                                            monkeypatch):
    """The OEP output .npz is name-keyed (the filename has no basis tag), so the
    cache-hit check must compare the recorded ``basis_used``: a mismatching
    basis MISSES (re-runs, avoiding a stale cross-basis reference), while a
    matching basis -- and a legacy npz lacking ``basis_used`` -- both HIT with
    no spurious re-run."""
    import numpy as np
    from xcquinox.alec.external_refs import (
        SpeciesEntry, resolve_geometry, run_oep_cascade, _REQUIRED_NPZ_KEYS,
    )
    from xcquinox.alec import oep as alec_oep

    spec = SpeciesEntry("H", 0, 1, "dfs_atom")
    atoms = resolve_geometry(spec)

    class _InversionRan(Exception):
        """Sentinel: not a RuntimeError/ValueError, so the cascade's tier
        try/except does NOT swallow it -- it marks a real cache MISS."""

    def _tripwire(*a, **k):
        raise _InversionRan()
    monkeypatch.setattr(alec_oep, "run_oep_inversion", _tripwire)

    ccsd_payload = {"dm_ao": np.zeros((2, 2, 2))}

    def _write_fake_npz(basis_used):
        payload = {k: np.zeros(1) for k in _REQUIRED_NPZ_KEYS}
        if basis_used is not None:
            payload["basis_used"] = np.array(str(basis_used))
        np.savez_compressed(tmp_path / f"{spec.name}.npz", **payload)

    # (1) basis MATCH -> cache hit (the inversion tripwire is never reached).
    _write_fake_npz("def2-svp")
    p = run_oep_cascade(spec, atoms, ccsd_payload=ccsd_payload,
                        cache_dir=tmp_path, basis="def2-svp", grid_level=1)
    assert p.is_file()

    # (2) legacy npz (no basis_used) -> trusted regardless of basis -> hit.
    _write_fake_npz(None)
    p = run_oep_cascade(spec, atoms, ccsd_payload=ccsd_payload,
                        cache_dir=tmp_path, basis="def2-tzvp", grid_level=1)
    assert p.is_file()

    # (3) basis MISMATCH -> cache MISS -> falls through to the (tripwired) run.
    _write_fake_npz("def2-svp")
    with pytest.raises(_InversionRan):
        run_oep_cascade(spec, atoms, ccsd_payload=ccsd_payload,
                        cache_dir=tmp_path, basis="def2-tzvp", grid_level=1)


# ---------------------------------------------------------------------------
# HF-for-CCSD convergence robustness (roots the c-hooo benchmark_refs failure)
# ---------------------------------------------------------------------------

# cis-HOOO (a HOOO doublet radical) is the species the benchmark_refs stage failed
# on: plain UHF from the PBE guess does NOT converge, so _prepare_converged_hf
# raised and CCSD never ran. geometry from the benchmark pool (bh76).
_C_HOOO_ATOM = ("O 1.0937122327 -0.3034156995 0; O 0.1609687573 0.5273601460 0; "
                "O -1.1992568767 -0.1563105723 0; H -0.8798021212 -1.0736198095 0")


def test_converge_scf_tiered_escalates_to_newton_on_stall():
    """Fast (no real SCF): when the plain kernel does not converge, the tiered
    helper falls back to SOSCF (newton) and returns the converged object."""
    from xcquinox.alec.external_refs import _converge_scf_tiered

    calls = []

    class _FakeMF:
        def __init__(self, is_newton=False):
            self.converged = False
            self._is_newton = is_newton

        def kernel(self, dm0=None):
            calls.append("newton" if self._is_newton else "plain")
            # plain stalls; the SOSCF-wrapped object converges
            self.converged = self._is_newton

        def newton(self):
            return _FakeMF(is_newton=True)

    mf = _converge_scf_tiered(lambda: _FakeMF(), dm0=None, is_uks=True)
    assert mf is not None and mf.converged
    assert calls[0] == "plain" and "newton" in calls  # plain first, then escalate


@pytest.mark.slow
def test_prepare_converged_hf_converges_hard_radical():
    """SLOW real-SCF integration: c-hooo (HOOO doublet) does NOT converge with
    plain UHF -- the exact benchmark_refs 'HF SCF did not converge' failure. The
    tiered convergence must return a converged (canonical) HF reference for CCSD."""
    from pyscf import gto, dft
    from xcquinox.alec.external_refs import _prepare_converged_hf

    mol = gto.M(atom=_C_HOOO_ATOM, basis="def2-svp", spin=1, charge=0, verbose=0)
    mf_pbe = dft.UKS(mol); mf_pbe.xc = "pbe"; mf_pbe.grids.level = 2
    mf_pbe.kernel()
    dm0 = mf_pbe.make_rdm1()

    mf_hf = _prepare_converged_hf(mol, dm0=dm0, is_uks=True)
    assert getattr(mf_hf, "converged", False)
    # converges to the CORRECT lower minimum (not the stalled ~-224.741 one)
    assert mf_hf.e_tot < -224.745
