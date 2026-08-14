"""SCAN seed-cache generation driver (cluster/seed_cache): programmatic
species enumeration from the full training-point pool, run identity
threading (incl. the orientation lock), skip-if-cached reporting."""
import json

import pytest

from xcquinox.alec.cluster import seed_cache as sc_mod


class _Atoms:
    """ase.Atoms stand-in: only .info is read by the driver."""
    def __init__(self, name, charge=0, spin=0):
        self.info = {"name": name, "charge": charge, "spin": spin}


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
    assert names == ["h2o", "o", "li+"]
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
        if entry.name == "bad":
            raise RuntimeError("SCF did not converge")
        return {"e_tot": -1.0}

    monkeypatch.setattr(sc_mod, "_run_scf_with_cache", _fake_run)
    assert sc_mod.main(["/cfg.yaml"]) == 1
    out = capsys.readouterr().out
    assert "bad" in out and "1 failed" in out and "1 cached" in out
