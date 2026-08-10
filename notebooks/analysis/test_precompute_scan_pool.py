"""Tests for ``precompute_scan_pool.py`` -- the SCAN density leg + the lock.

The SCF itself is not exercised here (a real SCAN KS-SCF at the production basis
is a cluster job). What IS pinned: the density arithmetic against an independent
oracle, the guards that stop a wrong grid or an unverifiable reference from
producing a plausible-looking number, and that the orientation lock reaches the
SCF call.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


psp = _load("precompute_scan_pool")


def _synthetic(n_grid: int = 40, n_ao: int = 3, seed: int = 0):
    """``(dm, ao, weights, rho)`` with ``rho`` the density ``dm`` puts on the
    grid -- so a reference equal to ``rho`` must score exactly zero error."""
    rng = np.random.default_rng(seed)
    ao = rng.normal(size=(n_grid, n_ao))
    dm = np.eye(n_ao) * 0.5
    w = np.abs(rng.normal(size=n_grid)) + 0.1
    rho = np.einsum("ij,gj,gi->g", dm, ao, ao)
    return dm, ao, w, rho


# ---------------------------------------------------------------------------
# cache naming
# ---------------------------------------------------------------------------

def test_cache_names_share_the_basis_slug():
    """Both caches must resolve from ONE basis label, and drop the +DF suffix
    exactly as the figure-side ``_scan_cache_name`` does -- else the figure
    looks for a filename the generator never wrote."""
    assert (psp._scan_cache_name("6-311++G(3df,2pd)+DF")
            == "scan_pool_energies_6-311++G_3df_2pd_.json")
    assert (psp._scan_density_cache_name("6-311++G(3df,2pd)+DF")
            == "scan_pool_density_6-311++G_3df_2pd_.json")
    # identical slug rule on both, for any basis
    for b in ("def2-svp", "def2-tzvpd+DF", "6-311++G(3df,2pd)"):
        assert (psp._scan_cache_name(b).replace("energies", "density")
                == psp._scan_density_cache_name(b))

    fig = _load("make_ablation_arch_figure")
    assert fig._scan_cache_name("def2-svp") == psp._scan_cache_name("def2-svp")
    assert (fig._scan_density_cache_name("def2-svp")
            == psp._scan_density_cache_name("def2-svp"))


# ---------------------------------------------------------------------------
# density arithmetic
# ---------------------------------------------------------------------------

def test_scan_density_record_matches_an_independent_oracle():
    dm, ao, w, rho = _synthetic()
    rho_ref = rho * 1.10
    rec = psp.scan_density_record(dm, ao, w, rho_ref)
    # oracle recomputed here from the definitions, not from the module
    want_rmse = float(np.sqrt(np.sum(w * (rho - rho_ref) ** 2) / np.sum(w)))
    want_eps = float(np.sum(w * np.abs(rho - rho_ref)) / np.sum(w * rho_ref))
    assert rec["density_rmse_scan"] == pytest.approx(want_rmse)
    assert rec["density_eps_l1_scan"] == pytest.approx(want_eps)
    assert rec["n_electrons"] == pytest.approx(float(np.sum(w * rho_ref)))


def test_scan_density_record_is_zero_against_its_own_density():
    dm, ao, w, rho = _synthetic()
    rec = psp.scan_density_record(dm, ao, w, rho)
    assert rec["density_rmse_scan"] == pytest.approx(0.0, abs=1e-12)
    assert rec["density_eps_l1_scan"] == pytest.approx(0.0, abs=1e-12)


def test_eps_normalizes_by_the_reference_electron_count():
    """eps = sum(w|rho-rho_ref|)/N_e with N_e from the REFERENCE, so swapping
    the two densities changes it. A mutant normalizing by SCAN's own count
    (or by sum(w)) fails here."""
    dm, ao, w, rho = _synthetic()
    rho_ref = rho * 2.0
    rec = psp.scan_density_record(dm, ao, w, rho_ref)
    n_e_ref = float(np.sum(w * rho_ref))
    n_e_scan = float(np.sum(w * rho))
    assert n_e_ref != pytest.approx(n_e_scan)
    assert rec["density_eps_l1_scan"] == pytest.approx(
        float(np.sum(w * np.abs(rho - rho_ref))) / n_e_ref)
    assert rec["density_eps_l1_scan"] != pytest.approx(
        float(np.sum(w * np.abs(rho - rho_ref))) / n_e_scan)


def test_spin_resolved_dm_is_summed():
    """A UKS dm arrives as (2, nao, nao); the density is the SPIN SUM. A mutant
    taking only the alpha block halves every density error."""
    dm, ao, w, rho = _synthetic()
    dm_uks = np.stack([dm * 0.5, dm * 0.5])
    rec_r = psp.scan_density_record(dm, ao, w, rho)
    rec_u = psp.scan_density_record(dm_uks, ao, w, rho)
    assert rec_u["density_rmse_scan"] == pytest.approx(rec_r["density_rmse_scan"])
    alpha_only = psp.scan_density_record(dm * 0.5, ao, w, rho)
    assert alpha_only["density_rmse_scan"] != pytest.approx(
        rec_u["density_rmse_scan"])


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def test_electron_count_guard_catches_a_mismatched_grid():
    """sum(w*rho) must reproduce the true electron count; it cannot if the AO
    values came from a different grid than the weights. This is the only check
    that the recomputed grid IS the reference's grid."""
    dm, ao, w, rho = _synthetic()
    n_e = float(np.sum(w * rho))
    psp.scan_density_record(dm, ao, w, rho, n_electrons_expected=n_e)  # ok
    with pytest.raises(ValueError, match="electrons"):
        psp.scan_density_record(dm, ao, w, rho, n_electrons_expected=n_e + 1.0)


def test_grid_length_mismatch_raises():
    dm, ao, w, rho = _synthetic()
    with pytest.raises(ValueError, match="grid length mismatch"):
        psp.scan_density_record(dm, ao, w[:-1], rho)
    with pytest.raises(ValueError, match="grid length mismatch"):
        psp.scan_density_record(dm, ao, w, rho[:-1])


# ---------------------------------------------------------------------------
# reference npz reading + the lock stamp
# ---------------------------------------------------------------------------

def _write_ref(tmp_path: Path, name: str, *, rho_ref, weights,
               lock=None) -> Path:
    arrays = {"rho_ref_grid": np.asarray(rho_ref),
              "grid_weights": np.asarray(weights)}
    if lock is not None:
        arrays["orientation_lock_strength"] = np.array(float(lock))
    p = tmp_path / f"{name}.npz"
    np.savez_compressed(p, **arrays)
    return p


def test_reference_density_reads_and_degrades(tmp_path):
    _, _, w, rho = _synthetic()
    _write_ref(tmp_path, "H2O", rho_ref=rho, weights=w, lock=3e-05)
    got_rho, got_w = psp._reference_density(tmp_path, "H2O")
    assert np.allclose(got_rho, rho) and np.allclose(got_w, w)
    # absent species -> (None, None), not an exception
    assert psp._reference_density(tmp_path, "NoSuchSpecies") == (None, None)
    # a reference lacking grid_weights cannot be scored -> (None, None)
    np.savez_compressed(tmp_path / "Old.npz", rho_ref_grid=np.asarray(rho))
    assert psp._reference_density(tmp_path, "Old") == (None, None)


def test_missing_lock_stamp_is_not_read_as_unlocked(tmp_path):
    """A reference with no orientation_lock_strength key has an UNKNOWN lock.
    Reading that as 0.0 is exactly the blindness that let the CH/NO references
    drift out of agreement with the training SCF, so it must return None."""
    _, _, w, rho = _synthetic()
    _write_ref(tmp_path, "Unstamped", rho_ref=rho, weights=w, lock=None)
    _write_ref(tmp_path, "Stamped", rho_ref=rho, weights=w, lock=3e-05)
    _write_ref(tmp_path, "Unlocked", rho_ref=rho, weights=w, lock=0.0)
    assert psp._reference_lock(tmp_path, "Unstamped") is None
    assert psp._reference_lock(tmp_path, "Stamped") == pytest.approx(3e-05)
    assert psp._reference_lock(tmp_path, "Unlocked") == pytest.approx(0.0)
    # a MISSING stamp and an explicit 0.0 must not compare equal
    assert psp._reference_lock(tmp_path, "Unstamped") != \
        psp._reference_lock(tmp_path, "Unlocked")


class _MolSpec:
    """Minimal MoleculeSpec stand-in for the density leg."""
    def __init__(self, name):
        self.name = name
        self.atom = "H 0 0 0"
        self.basis = "sto-3g"
        self.charge = 0
        self.spin = 1
        self.grid_level = 1
        self.atom_composition = (("H", 1),)


def test_density_leg_refuses_an_unstamped_or_mismatched_reference(tmp_path,
                                                                  monkeypatch):
    _, _, w, rho = _synthetic()
    _write_ref(tmp_path, "A", rho_ref=rho, weights=w, lock=None)
    _write_ref(tmp_path, "B", rho_ref=rho, weights=w, lock=1e-05)
    dj = tmp_path / "density.json"
    with pytest.raises(ValueError, match="no orientation_lock_strength"):
        psp._density_leg(_MolSpec("A"), {"dm": np.eye(1)}, name="A",
                         refs_dir=tmp_path, basis="sto-3g", grid_level=1,
                         density_fit=False, orientation_lock_strength=3e-05,
                         densities={}, density_json=dj)
    with pytest.raises(ValueError, match="orientation lock mismatch"):
        psp._density_leg(_MolSpec("B"), {"dm": np.eye(1)}, name="B",
                         refs_dir=tmp_path, basis="sto-3g", grid_level=1,
                         density_fit=False, orientation_lock_strength=3e-05,
                         densities={}, density_json=dj)


def test_density_leg_records_none_when_the_species_has_no_reference(tmp_path):
    """Atoms and anything outside the reference set are RECORDED as None, not
    dropped -- so the consumer can tell 'no reference' from 'not yet run'."""
    dj = tmp_path / "density.json"
    densities = {}
    msg = psp._density_leg(_MolSpec("Atom"), {"dm": np.eye(1)}, name="Atom",
                           refs_dir=tmp_path, basis="sto-3g", grid_level=1,
                           density_fit=False, orientation_lock_strength=3e-05,
                           densities=densities, density_json=dj)
    assert "no reference" in msg
    assert densities["Atom"] == {"density_rmse_scan": None,
                                 "density_eps_l1_scan": None}
    assert json.loads(dj.read_text())["Atom"]["density_rmse_scan"] is None


def test_density_leg_requires_a_refs_dir(tmp_path):
    with pytest.raises(ValueError, match="--refs-dir"):
        psp._density_leg(_MolSpec("X"), {"dm": np.eye(1)}, name="X",
                         refs_dir=None, basis="sto-3g", grid_level=1,
                         density_fit=False, orientation_lock_strength=0.0,
                         densities={}, density_json=tmp_path / "d.json")


# ---------------------------------------------------------------------------
# the lock reaches the SCF
# ---------------------------------------------------------------------------

def test_orientation_lock_is_passed_to_the_scf(monkeypatch, tmp_path):
    """The whole point of the flag: a SCAN SCF run unlocked against locked
    references scores a component mismatch, not a functional error. A mutant
    dropping the kwarg leaves it at run_scf_with_cache's 0.0 default."""
    seen = {}

    def _fake_scf(spec, atoms, **kw):
        seen.update(kw)
        return {"e_tot": -1.0, "dm": np.eye(1)}

    import xcquinox.alec.external_refs as ext
    import xcquinox.alec.benchmark_refs as bench
    monkeypatch.setattr(ext, "run_scf_with_cache", _fake_scf)
    monkeypatch.setattr(bench, "_mol_spec_to_atoms", lambda ms: object())
    monkeypatch.setattr(psp, "_load_pool",
                        lambda pool, **kw: {"X": _MolSpec("X")})

    n_fail = psp.run("all", basis="def2-svp", grid_level=1, out_dir=tmp_path,
                     orientation_lock_strength=3e-05)
    assert n_fail == 0
    assert seen["orientation_lock_strength"] == pytest.approx(3e-05)
    assert seen["xc"] == "scan"
    # and the energy cache landed under the shared slug
    cached = json.loads((tmp_path / psp._scan_cache_name("def2-svp")).read_text())
    assert cached == {"X": -1.0}


# ---------------------------------------------------------------------------
# The reference grid. Job 2114184 failed because the grid was REBUILT from the
# molecule + level instead of read from the reference: PySCF prunes during
# kernel(), so a fresh Grids object carries more points than the reference was
# written on (10128 vs 9264 for H2O/def2-svp/grid 1) and every species tripped
# the length guard.
# ---------------------------------------------------------------------------

def test_reference_ao_reads_the_intermediate_by_its_canonical_name(tmp_path):
    """``_reference_ao`` must look under exactly the filename
    ``external_refs`` wrote, including the basis / DF / lock tags -- a mutant
    that rebuilds the grid, or spells the cache name itself, silently gets the
    wrong points or nothing."""
    from xcquinox.alec.external_refs import _intermediate_cache_name
    inter = tmp_path / "_intermediates"
    inter.mkdir()
    ao = np.arange(12, dtype=float).reshape(4, 3)
    fname = _intermediate_cache_name(
        "H2O", grid_level=3, basis="6-311++G(3df,2pd)", density_fit=True,
        kind="ccsd", orientation_lock_strength=3e-05)
    np.savez_compressed(inter / fname, ao_grid=ao)
    got = psp._reference_ao(tmp_path, "H2O", basis="6-311++G(3df,2pd)",
                            grid_level=3, density_fit=True,
                            orientation_lock_strength=3e-05)
    assert got is not None and np.allclose(got, ao)
    # every identity tag participates: a different lock/basis/DF/grid must MISS
    for kw in ({"orientation_lock_strength": 0.0}, {"density_fit": False},
               {"grid_level": 2}, {"basis": "def2-svp"}):
        args = dict(basis="6-311++G(3df,2pd)", grid_level=3, density_fit=True,
                    orientation_lock_strength=3e-05)
        args.update(kw)
        assert psp._reference_ao(tmp_path, "H2O", **args) is None, kw


def test_reference_ao_degrades_when_the_intermediate_is_absent(tmp_path):
    """A deleted intermediate is a cache-state fact, not an error: it must
    return None so the density leg records the species as unscoreable instead
    of sinking the (already persisted) energy leg."""
    assert psp._reference_ao(tmp_path, "H2O", basis="def2-svp", grid_level=1,
                             density_fit=False,
                             orientation_lock_strength=0.0) is None
    inter = tmp_path / "_intermediates"
    inter.mkdir()
    from xcquinox.alec.external_refs import _intermediate_cache_name
    fname = _intermediate_cache_name("H2O", grid_level=1, basis="def2-svp",
                                     density_fit=False, kind="ccsd")
    np.savez_compressed(inter / fname, something_else=np.zeros(3))
    assert psp._reference_ao(tmp_path, "H2O", basis="def2-svp", grid_level=1,
                             density_fit=False,
                             orientation_lock_strength=0.0) is None


def test_density_leg_records_none_when_the_intermediate_is_missing(tmp_path):
    _, _, w, rho = _synthetic()
    _write_ref(tmp_path, "H2O", rho_ref=rho, weights=w, lock=0.0)
    dj = tmp_path / "density.json"
    densities = {}
    msg = psp._density_leg(_MolSpec("H2O"), {"dm": np.eye(1)}, name="H2O",
                           refs_dir=tmp_path, basis="def2-svp", grid_level=1,
                           density_fit=False, orientation_lock_strength=0.0,
                           densities=densities, density_json=dj)
    assert "intermediate absent" in msg
    assert densities["H2O"]["density_rmse_scan"] is None


def test_projection_reproduces_the_generator_s_own_pbe_density_error(tmp_path):
    """END-TO-END oracle: push the reference's OWN PBE density matrix through
    this module's projection and scoring, and it must reproduce the
    PBE-vs-CCSD RMSE that ``benchmark_refs`` computed and stored -- proving the
    grid, the contraction and the metric all match the reference pipeline.

    Deliberately NOT marked slow despite running a real SCF+CCSD (~15 s at
    def2-svp/grid 1): it is the only test that checks this module against the
    reference pipeline rather than against fixtures this file generates itself,
    and it is the one that would have caught the rebuilt-grid bug (job 2114184,
    201/214 species lost). A regression test deselected by default is not
    protection.
    """
    pytest.importorskip("pyscf")
    from xcquinox.alec import benchmark_refs
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.external_refs import SpeciesEntry, run_scf_with_cache

    ms = MoleculeSpec.from_dict(
        name="H2O", atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24",
        basis="def2-svp", charge=0, spin=0,
        atom_composition={"O": 1, "H": 2}, grid_level=1)
    assert benchmark_refs.generate_one(
        ms, out_dir=str(tmp_path), basis="def2-svp", grid_level=1,
        density_fit=False, auxbasis=None,
        orientation_lock_strength=0.0) == "OK"

    rho_ref, w = psp._reference_density(tmp_path, "H2O")
    ao = psp._reference_ao(tmp_path, "H2O", basis="def2-svp", grid_level=1,
                           density_fit=False, orientation_lock_strength=0.0)
    assert ao is not None
    assert ao.shape[0] == w.shape[0] == rho_ref.shape[0]

    scf = run_scf_with_cache(
        SpeciesEntry(name="H2O", charge=0, spin=0, source="benchmark"),
        benchmark_refs._mol_spec_to_atoms(ms), cache_dir=str(tmp_path),
        basis="def2-svp", grid_level=1, density_fit=False, auxbasis=None)
    rec = psp.scan_density_record(scf["dm"], ao, w, rho_ref,
                                  n_electrons_expected=psp._n_electrons(ms))
    with np.load(tmp_path / "H2O.npz", allow_pickle=False) as z:
        stored = float(np.sqrt(
            np.sum(z["grid_weights"] * (z["rho_pbe_grid"] - z["rho_ref_grid"]) ** 2)
            / np.sum(z["grid_weights"])))
    assert rec["density_rmse_scan"] == pytest.approx(stored, rel=1e-10)
