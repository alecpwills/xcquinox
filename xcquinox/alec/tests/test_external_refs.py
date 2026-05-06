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


def test_oep_tiers_rks_and_uks_constants_split():
    """RKS and UKS tier constants exist with documented conv_tol values.

    RKS conv_tol=2e-3 is mirrored from step-6 closed-shell H2O/C2H2 floor.
    UKS conv_tol=1e-2 is set against the empirical UKS floor (~6e-3 on HO
    at def2-svp/grid_level=1 with level_shift=0.5) — see _OEP_TIERS_UKS
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
