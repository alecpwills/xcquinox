"""Tests for xcquinox.pipeline.external_refs species union + pipeline."""
import pytest
from xcquinox.pipeline.external_refs import (
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
    from xcquinox.pipeline import external_refs as ext
    snapshot = dict(ext._PER_SPECIES_OEP_OVERRIDES)
    ext._PER_SPECIES_OEP_OVERRIDES.clear()
    yield
    ext._PER_SPECIES_OEP_OVERRIDES.clear()
    ext._PER_SPECIES_OEP_OVERRIDES.update(snapshot)


def test_species_union_dedup_key_charge_spin():
    """Li (neutral, spin=1) and Li+ (charge=+1, spin=0) are DISTINCT entries."""
    species = build_species_union()
    by_key = {(s.name, s.charge, s.spin): s for s in species}
    assert ("Li", 0, 1) in by_key, "Li neutral missing"
    assert ("Li+", 1, 0) in by_key, "Li+ cation missing"
    assert by_key[("Li", 0, 1)] is not by_key[("Li+", 1, 0)]


def test_resolve_geometry_dfs_ae():
    """DFS AE molecules resolved from g2_97.traj by Hill formula."""
    from xcquinox.pipeline.external_refs import (
        SpeciesEntry,
        resolve_geometry,
    )
    spec = SpeciesEntry(name="H2O", charge=0, spin=0, source="dfs_ae")
    atoms = resolve_geometry(spec)
    assert len(atoms) == 3, "H2O should have 3 atoms"
    assert atoms.info.get("spin") == 0
    assert atoms.info.get("charge") == 0


def test_resolve_geometry_cation():
    """IP13 cation: bare atom with charge=+1 and cation_spin."""
    from xcquinox.pipeline.external_refs import (
        SpeciesEntry,
        resolve_geometry,
    )
    spec = SpeciesEntry(name="C+", charge=1, spin=1, source="ip13")
    atoms = resolve_geometry(spec)
    assert len(atoms) == 1
    assert atoms.get_chemical_symbols() == ["C"]
    assert atoms.info["charge"] == 1
    assert atoms.info["spin"] == 1


def test_run_scf_rks_for_closed_shell(tmp_path):
    """Closed-shell H2O dispatches to RKS; produces (n_ao, n_ao) DM."""
    from xcquinox.pipeline.external_refs import (
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
    from xcquinox.pipeline.external_refs import (
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
    from xcquinox.pipeline.external_refs import (
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
    from xcquinox.pipeline.external_refs import (
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
    from xcquinox.pipeline.external_refs import (
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


def test_oep_cascade_writes_npz_with_required_keys(tmp_path):
    """Stage 3 OEP for H2 produces npz with vxc_ref, dm_target, rho_ref_grid."""
    from xcquinox.pipeline.external_refs import (
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
    from xcquinox.pipeline.external_refs import (
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
    at def2-svp/grid_level=1 with level_shift=0.5), see _OEP_TIERS_UKS
    docstring in xcquinox/pipeline/external_refs.py for the full rationale.

    This test pins the values so a future edit cannot silently regress
    the cascade quality contract.
    """
    from xcquinox.pipeline.external_refs import _OEP_TIERS_RKS, _OEP_TIERS_UKS

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
    from xcquinox.pipeline import external_refs as er
    from xcquinox.pipeline import oep as pipeline_oep

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

    monkeypatch.setattr(pipeline_oep, "run_oep_inversion",
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

    monkeypatch.setattr(pipeline_oep, "save_vxc_ref", stub_save_vxc_ref)

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


def test_run_log_finalize_archives(tmp_path):
    """Finalize renames partial -> run_log_<ts>.json and removes partial."""
    from xcquinox.pipeline.external_refs import RunLog
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


def test_npz_is_complete_checks_lock_and_basis_currency(tmp_path):
    """The final reference npz is name-keyed (no basis or lock tag in the
    filename), so completeness alone would let precompute_all serve a file
    generated at ANY lock or basis. With a stated identity the check must
    MISS on a lock mismatch (in both directions) and on a basis mismatch
    when the file records one; a legacy file missing the lock key reads as
    0.0 (a hit for an unlocked run only), and one missing basis_used is
    trusted at any basis -- the run_oep_cascade cache rule."""
    import numpy as np
    from xcquinox.pipeline.external_refs import (
        _REQUIRED_NPZ_KEYS, _npz_is_complete)
    path = tmp_path / "X.npz"

    def write(**extra):
        payload = {k: np.zeros(1) for k in _REQUIRED_NPZ_KEYS}
        payload.update(extra)
        np.savez_compressed(path, **payload)

    write(basis_used=np.array("def2-svp"),
          orientation_lock_strength=np.array(3e-5))
    assert _npz_is_complete(path, basis="def2-svp",
                            orientation_lock_strength=3e-5)
    # A locked reference must not serve an unlocked run either.
    assert not _npz_is_complete(path, basis="def2-svp",
                                orientation_lock_strength=0.0)
    assert not _npz_is_complete(path, basis="def2-tzvp",
                                orientation_lock_strength=3e-5)
    # Legacy: no lock key reads as 0.0.
    write(basis_used=np.array("def2-svp"))
    assert _npz_is_complete(path, basis="def2-svp",
                            orientation_lock_strength=0.0)
    assert not _npz_is_complete(path, basis="def2-svp",
                                orientation_lock_strength=3e-5)
    # Legacy: a file recording no basis is trusted at any requested basis.
    write()
    assert _npz_is_complete(path, basis="def2-tzvp",
                            orientation_lock_strength=0.0)
    # Stating no identity preserves the key-presence contract.
    assert _npz_is_complete(path)
    # Completeness still gates however well the identity matches.
    np.savez_compressed(path, rho_ref_grid=np.zeros(1),
                        basis_used=np.array("def2-svp"),
                        orientation_lock_strength=np.array(3e-5))
    assert not _npz_is_complete(path, basis="def2-svp",
                                orientation_lock_strength=3e-5)
    assert not _npz_is_complete(tmp_path / "absent.npz", basis="def2-svp",
                                orientation_lock_strength=0.0)


def test_precompute_all_does_not_serve_an_unlocked_reference_to_a_locked_run(
        tmp_path, monkeypatch):
    """precompute_all's skip decision must miss on a lock mismatch: a
    complete file generated without the lock is regenerated, not skipped,
    when the run requests one -- while the same file stays a skip for an
    unlocked run (the legacy-cache contract)."""
    import json
    import numpy as np
    from xcquinox.pipeline import external_refs as ext
    payload = {k: np.zeros(1) for k in ext._REQUIRED_NPZ_KEYS}
    np.savez_compressed(tmp_path / "H2.npz", **payload)
    species = [ext.SpeciesEntry("H2", 0, 0, "dfs_ae")]

    class _RegenerationReached(Exception):
        """Raised where the generation path starts (resolve_geometry, the
        first call after the skip decision): reaching it IS the miss."""

    def _tripwire(spec):
        raise _RegenerationReached()

    monkeypatch.setattr(ext, "resolve_geometry", _tripwire)

    def _last_statuses():
        logs = sorted(tmp_path.glob("_run_log_*.json"))
        return [r["status"]
                for r in json.loads(logs[-1].read_text())["results"]]

    with pytest.raises(RuntimeError, match="failed for 1 species"):
        ext.precompute_all(species, cache_dir=tmp_path, basis="def2-svp",
                           grid_level=1, run_preflight=False,
                           orientation_lock_strength=3e-5)
    assert _last_statuses() == ["FAIL"]
    ext.precompute_all(species, cache_dir=tmp_path, basis="def2-svp",
                       grid_level=1, run_preflight=False)
    assert _last_statuses() == ["SKIPPED_CACHED"]


def test_validate_overrides_rejects_unknown_knob():
    """Typo in override-tier dict key is rejected with a clear error."""
    import pytest
    from xcquinox.pipeline.external_refs import (
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


def test_resolve_tiers_override_merges_onto_default():
    """Single-knob override keeps default max_iter / conv_tol; aux_basis swaps."""
    from xcquinox.pipeline.external_refs import (
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


def test_migration_renames_unsuffixed_intermediates_to_g1(tmp_path):
    """Pre-2026-05-03 caches (no grid suffix) get renamed to _g1_."""
    import numpy as np
    from xcquinox.pipeline.external_refs import (
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


    # And both invocations succeeded without raising:
    # (the implicit assertion is that we got here without exception)


# ---------------------------------------------------------------------------
# CCSD must run on a CONVERGED HF reference, not grafted PBE MOs.
# ---------------------------------------------------------------------------


def test_prepare_converged_hf_raises_when_not_converged(monkeypatch):
    """If HF SCF does not converge, _prepare_converged_hf raises rather than
    silently feeding a non-self-consistent determinant to CCSD."""
    from pyscf import gto
    from xcquinox.pipeline import external_refs as ext

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


# ---------------------------------------------------------------------------
# the reference npz must carry grid_level_used provenance.
# ---------------------------------------------------------------------------


def test_run_oep_cascade_cache_keys_on_orientation_lock(tmp_path, monkeypatch):
    """The final per-species reference cache must key on orientation_lock_strength
    too, so a LOCKED run cannot silently reuse an UNLOCKED reference from a prior
    run in the same cache_dir (the degenerate OH/CH/NO radical density fix). An
    inversion tripwire marks a cache MISS (regeneration)."""
    import numpy as np
    from xcquinox.pipeline.external_refs import (
        SpeciesEntry, resolve_geometry, run_oep_cascade, _REQUIRED_NPZ_KEYS,
    )
    from xcquinox.pipeline import oep as pipeline_oep

    spec = SpeciesEntry("H", 0, 1, "dfs_atom")
    atoms = resolve_geometry(spec)

    class _InversionRan(Exception):
        """Sentinel (not Runtime/ValueError, so the cascade tier try/except does
        not swallow it) -> a real cache MISS."""

    monkeypatch.setattr(pipeline_oep, "run_oep_inversion",
                        lambda *a, **k: (_ for _ in ()).throw(_InversionRan()))
    ccsd_payload = {"dm_ao": np.zeros((2, 2, 2))}

    def _write_fake_npz(ol):
        payload = {k: np.zeros(1) for k in _REQUIRED_NPZ_KEYS}
        payload["basis_used"] = np.array("def2-svp")
        if ol is not None:
            payload["orientation_lock_strength"] = np.array(float(ol))
        np.savez_compressed(tmp_path / f"{spec.name}.npz", **payload)

    def _run(ol):
        return run_oep_cascade(spec, atoms, ccsd_payload=ccsd_payload,
                               cache_dir=tmp_path, basis="def2-svp", grid_level=1,
                               orientation_lock_strength=ol)

    # (1) locked run + UNLOCKED cache (no ol key -> treated 0.0) -> MISS.
    _write_fake_npz(None)
    with pytest.raises(_InversionRan):
        _run(3e-5)
    # (2) locked run + matching LOCKED cache -> HIT (tripwire never reached).
    _write_fake_npz(3e-5)
    assert _run(3e-5).is_file()
    # (3) unlocked run + legacy UNLOCKED cache -> HIT (byte-identical to pre-fix).
    _write_fake_npz(None)
    assert _run(0.0).is_file()
    # (4) unlocked run + LOCKED cache -> MISS (must not reuse a locked ref either).
    _write_fake_npz(3e-5)
    with pytest.raises(_InversionRan):
        _run(0.0)


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
    from xcquinox.pipeline.external_refs import _converge_scf_tiered

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


# ---------------------------------------------------------------------------
# xc keyword on run_scf_with_cache (SCAN baseline in the DFS density demo).
# The default xc="pbe" MUST reproduce the pre-existing cache name + numerics.
# ---------------------------------------------------------------------------


def test_require_ccsd_converged_refuses_unconverged():
    """An unconverged CCSD must refuse before any density write: make_rdm1
    on unconverged amplitudes has no stated accuracy and nothing downstream
    can detect it (the npz carries no residual)."""
    from xcquinox.pipeline.external_refs import _require_ccsd_converged

    class _CC:
        def __init__(self, ok):
            self.converged = ok

    with pytest.raises(RuntimeError, match="c2"):
        _require_ccsd_converged(_CC(False), "c2")
    _require_ccsd_converged(_CC(True), "c2")  # no raise

    class _NoAttr:
        pass

    with pytest.raises(RuntimeError, match="did not converge"):
        _require_ccsd_converged(_NoAttr(), "h2o")


def test_ccsd_cache_hit_returns_the_written_key_set(tmp_path):
    """A served cache carries what the computation carried, the convergence
    stamp included: a caller cannot tell from the payload whether the density
    was computed in this process or read from disk, so a provenance check that
    reads the stamp behaves the same either way.

    Oracle: the two returns of the same call on H2, the second served (the
    cache file is not rewritten, so the branch under test is the hit branch).
    """
    from xcquinox.pipeline.external_refs import (
        SpeciesEntry, resolve_geometry,
        run_scf_with_cache, run_ccsd_with_cache,
    )
    spec = SpeciesEntry("H2", 0, 0, "dfs_ae")
    atoms = resolve_geometry(spec)
    scf = run_scf_with_cache(spec, atoms, cache_dir=tmp_path,
                             basis="def2-svp", grid_level=1)
    cold = run_ccsd_with_cache(spec, atoms, scf_payload=scf,
                               cache_dir=tmp_path,
                               basis="def2-svp", grid_level=1)
    caches = sorted((tmp_path / "_intermediates").glob("*_ccsd.npz"))
    assert len(caches) == 1, caches
    mtime = caches[0].stat().st_mtime

    served = run_ccsd_with_cache(spec, atoms, scf_payload=scf,
                                 cache_dir=tmp_path,
                                 basis="def2-svp", grid_level=1)
    assert caches[0].stat().st_mtime == mtime, (
        "the cache was rewritten, so the second call did not take the hit "
        "branch this test is about")

    assert set(served.keys()) == set(cold.keys()), (
        f"served {sorted(served)} against computed {sorted(cold)}")
    assert cold["ccsd_converged"] is True
    assert served["ccsd_converged"] is True
