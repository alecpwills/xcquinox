"""SCAN seed-cache generation driver (cluster/seed_cache): programmatic
species enumeration from the full training-point pool, run identity
threading (incl. the orientation lock), skip-if-cached reporting."""
import json

import pytest

from xcquinox.alec.cluster import seed_cache as sc_mod


class _Atoms:
    """ase.Atoms stand-in: .info plus the geometry accessors the
    qualification tag reads."""
    def __init__(self, name, charge=0, spin=0):
        self.info = {"name": name, "charge": charge, "spin": spin}

    def get_chemical_symbols(self):
        return ["H"]

    def get_positions(self):
        return [(0.0, 0.0, 0.0)]


class _Cfg:
    class inputs:
        basis = "6-311++G(3df,2pd)"
        grid_level = 3
        density_fit = True
        auxbasis = None
        orientation_lock_strength = 3e-05
        seed_cache_dir = "/gpfs/scratch/x/seed_cache"
    domain_profile = "dfs_step7"


def test_enumerates_species_from_full_pool_and_threads_identity(monkeypatch):
    species = [_Atoms("h2o"), _Atoms("o", spin=2), _Atoms("li+", charge=1,
                                                          spin=0)]
    monkeypatch.setattr(sc_mod, "_load_cfg", lambda path: _Cfg())
    monkeypatch.setattr(sc_mod, "_pool_species", lambda cfg: species)
    calls = []

    def _fake_run(entry, atoms, *, cache_dir, basis, grid_level, density_fit,
                  auxbasis, orientation_lock_strength, xc):
        calls.append((entry.name, entry.charge, entry.spin, cache_dir, basis,
                      grid_level, density_fit, orientation_lock_strength, xc))
        return {"e_tot": -1.0}

    monkeypatch.setattr(sc_mod, "_run_scf_with_cache", _fake_run)
    rc = sc_mod.main(["/cfg.yaml"])
    assert rc == 0
    assert len(calls) == 3
    names = [c[0] for c in calls]
    # geometry-qualified cache names: <name>_gh<8hex> so training-vs-pool
    # same-name twins at different geometries resolve to distinct files
    import re
    assert all(re.fullmatch(r"(h2o|o|li\+)_gh[0-9a-f]{8}", n) for n in names), names
    assert [n.split("_gh")[0] for n in names] == ["h2o", "o", "li+"]
    for c in calls:
        # the FULL seed identity: run inputs incl. the orientation lock and
        # xc='scan' -- an unlocked or PBE-tagged cache would be a different
        # (wrong) identity
        assert c[3] == "/gpfs/scratch/x/seed_cache"
        assert c[4] == "6-311++G(3df,2pd)"
        assert c[5] == 3
        assert c[6] is True
        assert c[7] == 3e-05
        assert c[8] == "scan"
    # charge/spin threaded per species
    assert calls[1][2] == 2 and calls[2][1] == 1


def test_requires_seed_cache_dir(monkeypatch, capsys):
    class _NoDir(_Cfg):
        class inputs(_Cfg.inputs):
            seed_cache_dir = None
    monkeypatch.setattr(sc_mod, "_load_cfg", lambda path: _NoDir())
    monkeypatch.setattr(sc_mod, "_pool_species", lambda cfg: [_Atoms("h2o")])
    assert sc_mod.main(["/cfg.yaml"]) == 1
    assert "seed_cache_dir" in capsys.readouterr().out


def test_failures_collected_and_exit_nonzero(monkeypatch, capsys):
    species = [_Atoms("good"), _Atoms("bad")]
    monkeypatch.setattr(sc_mod, "_load_cfg", lambda path: _Cfg())
    monkeypatch.setattr(sc_mod, "_pool_species", lambda cfg: species)

    def _fake_run(entry, atoms, **kw):
        if entry.name.split("_gh")[0] == "bad":
            raise RuntimeError("SCF did not converge")
        return {"e_tot": -1.0}

    monkeypatch.setattr(sc_mod, "_run_scf_with_cache", _fake_run)
    assert sc_mod.main(["/cfg.yaml"]) == 1
    out = capsys.readouterr().out
    assert "bad" in out and "1 failed" in out and "1 cached" in out


def test_link_pool_creates_geometry_qualified_links(tmp_path, monkeypatch):
    """--link-pool re-keys the scan-pool intermediates under
    geometry-qualified names derived from the POOL geometries, so eval
    lookups (which qualify by the pool spec) resolve them."""
    import numpy as np

    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.data import seed_cache_file
    from xcquinox.alec.external_refs import _intermediate_cache_name

    class _Cfg2(_Cfg):
        class inputs(_Cfg.inputs):
            seed_cache_dir = None  # set below

    seed_dir = tmp_path / "seed"
    pool_dir = tmp_path / "scanpool"
    (pool_dir / "_intermediates").mkdir(parents=True)
    _Cfg2.inputs.seed_cache_dir = str(seed_dir)

    ps = MoleculeSpec(name="h2o", atom="O 0 0 0; H 0 0 0.96; H 0.96 0 0",
                      basis=_Cfg.inputs.basis, charge=0, spin=0,
                      atom_composition=(("H", 2), ("O", 1)), grid_level=3)
    # the pool cache's UNQUALIFIED file, as dfs6311_scan_pool wrote it
    src = pool_dir / "_intermediates" / _intermediate_cache_name(
        "h2o", grid_level=3, basis=_Cfg.inputs.basis, density_fit=True,
        kind="scf", orientation_lock_strength=3e-05, xc="scan")
    np.savez_compressed(src, dm=np.eye(2))

    monkeypatch.setattr(sc_mod, "_load_cfg", lambda path: _Cfg2())
    monkeypatch.setattr(sc_mod, "_pool_species", lambda cfg: [])
    monkeypatch.setattr(
        sc_mod, "_held_out_pool_specs", lambda cfg: {"h2o": ps})
    rc = sc_mod.main(["/cfg.yaml", "--link-pool", str(pool_dir)])
    assert rc == 0
    dst = seed_cache_file(ps, seed_cache_dir=str(seed_dir),
                          density_fit=True,
                          orientation_lock_strength=3e-05)
    import os
    assert os.path.islink(dst)
    assert os.path.realpath(dst) == str(src)
    # idempotent rerun
    assert sc_mod.main(["/cfg.yaml", "--link-pool", str(pool_dir)]) == 0
