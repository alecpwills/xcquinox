"""Tests for ``precompute_nonempirical_pool.py`` (seam-driven; no SCF)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


pnp = _load("precompute_nonempirical_pool")


def test_pool_cache_name_slug():
    # same slug rule as the SCAN cache: +DF dropped, unsafe chars -> _
    assert pnp._pool_cache_name("def2-svp") == "nonempirical_pool_def2-svp.json"
    assert (pnp._pool_cache_name("6-311++G(3df,2pd)")
            == "nonempirical_pool_6-311++G_3df_2pd_.json")
    assert (pnp._pool_cache_name("def2-tzvpd+DF")
            == "nonempirical_pool_def2-tzvpd.json")


def _fake_env(tmp_path, *, scan_rho=None, e_by_xc=None):
    """Seam bundle: 2-species pool, hand-sized ref arrays, recording fakes."""
    ms1 = SimpleNamespace(name="h2o", charge=0, spin=0)
    ms2 = SimpleNamespace(name="oh", charge=0, spin=1)
    pool = {"h2o": ms1, "oh": ms2}
    rho_ref = np.array([2.0, 1.0])
    weights = np.array([3.0, 1.0])
    rho_pbe = np.array([2.5, 0.5])         # eps_pbe = 2/7 vs this ref
    scan_rho = scan_rho if scan_rho is not None else np.array([2.0, 0.5])
    e_by_xc = e_by_xc or {"pbe": -76.0, "scan": -76.1}
    calls = {"density": [], "scf": []}

    def fake_pool(pool_name, *, basis, grid_level):
        return dict(sorted(pool.items()))

    def fake_refs(ms, refs_dir):
        return rho_ref, weights, rho_pbe

    def fake_scf(spec, atoms, **kw):
        calls["scf"].append((spec.name if hasattr(spec, "name") else spec,
                             kw.get("xc")))
        return {"e_tot": e_by_xc[kw["xc"]], "dm": None, "grid_coords": None}

    def fake_density(scf, ms, *, basis):
        calls["density"].append(ms.name)
        return scan_rho

    def fake_spec_atoms(ms):
        return SimpleNamespace(name=ms.name), ()

    return dict(pool=pool, rho_ref=rho_ref, weights=weights, rho_pbe=rho_pbe,
                scan_rho=scan_rho, calls=calls, seams=dict(
                    _pool_loader=fake_pool, _refs_loader=fake_refs,
                    _scf=fake_scf, _density=fake_density,
                    _spec_atoms=fake_spec_atoms))


def test_run_writes_cache_with_hand_computed_eps(tmp_path):
    env = _fake_env(tmp_path)
    n_fail = pnp.run("all", basis="def2-svp", grid_level=2, out_dir=tmp_path,
                     xcs=("pbe", "scan"), **env["seams"])
    assert n_fail == 0
    cache = json.loads((tmp_path / "nonempirical_pool_def2-svp.json").read_text())
    assert set(cache) == {"h2o", "oh"}
    for name in ("h2o", "oh"):
        # PBE leg: density from the npz fast path (rho_pbe), NOT the SCF dm
        # eps = sum(w|rho_pbe - rho_ref|)/N_e = (3*0.5 + 1*0.5)/7 = 2/7
        pbe = cache[name]["pbe"]
        assert pbe["e_tot"] == pytest.approx(-76.0)
        assert pbe["density_eps_l1"] == pytest.approx(2.0 / 7.0)
        assert pbe["n_electrons"] == pytest.approx(7.0)
        assert pbe["grid_weight_sum"] == pytest.approx(4.0)
        # SCAN leg: density from the (fake) SCF dm recipe
        # eps = (3*|2-2| + 1*|0.5-1|)/7 = 0.5/7
        scan = cache[name]["scan"]
        assert scan["e_tot"] == pytest.approx(-76.1)
        assert scan["density_eps_l1"] == pytest.approx(0.5 / 7.0)
    # the density seam ran ONLY for scan (pbe took the rho_pbe fast path)
    assert env["calls"]["density"] == ["h2o", "oh"]


def test_run_resume_skips_cached_and_force_recomputes(tmp_path):
    env = _fake_env(tmp_path)
    pnp.run("all", basis="def2-svp", grid_level=2, out_dir=tmp_path,
            xcs=("pbe",), **env["seams"])
    n_scf_first = len(env["calls"]["scf"])
    pnp.run("all", basis="def2-svp", grid_level=2, out_dir=tmp_path,
            xcs=("pbe",), **env["seams"])
    assert len(env["calls"]["scf"]) == n_scf_first      # resume: no new SCF
    pnp.run("all", basis="def2-svp", grid_level=2, out_dir=tmp_path,
            xcs=("pbe",), force=True, **env["seams"])
    assert len(env["calls"]["scf"]) == 2 * n_scf_first  # force: recomputed


def test_run_counts_failures_without_raising(tmp_path):
    env = _fake_env(tmp_path, scan_rho=np.array([1.0, 2.0, 3.0]))  # wrong shape
    n_fail = pnp.run("all", basis="def2-svp", grid_level=2, out_dir=tmp_path,
                     xcs=("scan",), **env["seams"])
    assert n_fail == 2                        # both species fail the shape gate
    # nothing succeeded -> no cache file is ever written (atomic-write only
    # fires on success); if a partial file existed it must not carry scan
    p = tmp_path / "nonempirical_pool_def2-svp.json"
    if p.is_file():
        cache = json.loads(p.read_text())
        assert all("scan" not in v for v in cache.values())


def test_run_refs_failure_skips_all_xcs(tmp_path):
    env = _fake_env(tmp_path)

    def bad_refs(ms, refs_dir):
        raise FileNotFoundError("no refs")

    seams = dict(env["seams"], _refs_loader=bad_refs)
    n_fail = pnp.run("all", basis="def2-svp", grid_level=2, out_dir=tmp_path,
                     xcs=("pbe", "scan"), **seams)
    assert n_fail == 4                        # 2 species x 2 xcs
    assert env["calls"]["scf"] == []          # never reached the SCF seam


def test_run_threads_lock_into_scf_seam(tmp_path):
    env = _fake_env(tmp_path)
    seen: list = []

    def rec_scf(spec, atoms, **kw):
        seen.append(kw.get("orientation_lock_strength"))
        return {"e_tot": -1.0, "dm": None, "grid_coords": None}

    seams = dict(env["seams"], _scf=rec_scf)
    pnp.run("all", basis="def2-svp", grid_level=2, out_dir=tmp_path,
            xcs=("pbe",), orientation_lock_strength=3e-05, **seams)
    assert seen and all(v == 3e-05 for v in seen)


# ---------------------------------------------------------------------------
# REAL seams (no stubs): the pyscf density recipe and the refs identity gate.
# These exist because the seam-driven tests above cannot catch a defect
# inside the default implementations themselves.
# ---------------------------------------------------------------------------

def test_real_seams_h2_scan_density(tmp_path):
    """End-to-end on one real H2 SCF at a non-PBE functional: the real
    _default_spec_atoms -> run_scf_with_cache -> _density_on_grid chain must
    produce a density on the stored grid that integrates to N_e = 2."""
    pytest.importorskip("pyscf")
    ms = SimpleNamespace(name="h2", atom="H 0 0 0; H 0 0 0.74",
                         charge=0, spin=0)
    spec, atoms = pnp._default_spec_atoms(ms)
    scf = pnp._default_scf(spec, atoms, cache_dir=tmp_path, basis="def2-svp",
                           grid_level=1, density_fit=False, auxbasis=None,
                           xc="scan")
    assert scf["e_tot"] is not None
    rho = pnp._density_on_grid(scf, ms, basis="def2-svp")
    w = np.asarray(scf["grid_weights"])
    assert rho.shape == w.shape
    assert float(np.sum(w * rho)) == pytest.approx(2.0, abs=1e-3)


def test_load_ref_arrays_identity_gate(tmp_path):
    """The real refs loader must verify the reference identity stamps
    (basis/grid/DF/lock) against the run parameters -- the shape gate alone
    cannot catch these mismatches."""
    np.savez_compressed(
        tmp_path / "x.npz",
        rho_ref_grid=np.array([1.0, 2.0]),
        grid_weights=np.array([1.0, 1.0]),
        ref_density_method=np.array("ccsd"),
        basis_used=np.array("def2-svp"),
        grid_level_used=np.array(1),
        density_fit_used=np.array(False),
        orientation_lock_strength=np.array(0.0))
    ms = SimpleNamespace(name="x", external_data_path=None)
    rho_ref, w, rho_pbe = pnp._load_ref_arrays(
        ms, tmp_path, basis="def2-svp", grid_level=1,
        density_fit=False, orientation_lock_strength=0.0)
    assert rho_ref.tolist() == [1.0, 2.0] and rho_pbe is None
    for bad in (dict(orientation_lock_strength=3e-05),
                dict(basis="6-311++G(3df,2pd)"),
                dict(density_fit=True),
                dict(grid_level=3)):
        kw = dict(basis="def2-svp", grid_level=1, density_fit=False,
                  orientation_lock_strength=0.0)
        kw.update(bad)
        with pytest.raises(ValueError, match="identity mismatch"):
            pnp._load_ref_arrays(ms, tmp_path, **kw)
    # identity params omitted (pure seam use) -> loads without the gate
    rho_ref2, _, _ = pnp._load_ref_arrays(ms, tmp_path)
    assert rho_ref2.tolist() == [1.0, 2.0]
