"""Tests for the density-only benchmark CCSD reference generator."""
import json

import numpy as np
import pytest

import xcquinox.alec.benchmark_refs as br
from xcquinox.alec.config import MoleculeSpec


def _ms(name="H2O", atom="O 0.000000 0.000000 0.117000; "
                         "H 0.000000 0.757000 -0.468000; "
                         "H 0.000000 -0.757000 -0.468000",
        charge=0, spin=0, comp=(("H", 2), ("O", 1))) -> MoleculeSpec:
    return MoleculeSpec(name=name, atom=atom, basis="def2-svp", charge=charge,
                        spin=spin, atom_composition=comp,
                        external_data_path=None, grid_level=2)


def _fake_stages(monkeypatch, calls):
    """Stub SCF/CCSD seams (pattern: test_external_refs_df.py) so no PySCF
    runs; the CCSD payload carries a recognizable rho_ref_grid."""
    def fake_scf(spec, atoms, *, cache_dir, basis, grid_level,
                 density_fit=False, auxbasis=None):
        calls.append(("scf", spec.name, density_fit, auxbasis))
        return {"dm": np.eye(2), "spin_unrestricted": False,
                "grid_coords": np.zeros((4, 3)), "grid_weights": np.ones(4)}

    def fake_ccsd(spec, atoms, *, scf_payload, cache_dir, basis, grid_level,
                  density_fit=False, auxbasis=None):
        calls.append(("ccsd", spec.name, density_fit, auxbasis))
        return {"rho_ref_grid": np.array([1.0, 2.0, 3.0, 4.0]),
                "grid_weights": np.ones(4), "ao_grid": np.zeros((4, 2)),
                "dm_ao": np.eye(2)}

    monkeypatch.setattr(br, "run_scf_with_cache", fake_scf)
    monkeypatch.setattr(br, "run_ccsd_with_cache", fake_ccsd)


def test_mol_spec_to_atoms_roundtrip():
    ms = _ms(charge=-1, spin=1)
    at = br._mol_spec_to_atoms(ms)
    assert at.get_chemical_symbols() == ["O", "H", "H"]
    np.testing.assert_allclose(
        at.get_positions(),
        [[0.0, 0.0, 0.117], [0.0, 0.757, -0.468], [0.0, -0.757, -0.468]],
        atol=1e-12)
    assert at.info["name"] == "H2O"
    assert at.info["charge"] == -1 and at.info["spin"] == 1


def test_resolve_slice_shards_are_disjoint_and_cover():
    n = 214
    covered = []
    for i in range(1, 17):
        sl = br.resolve_slice(n, shard=f"{i}/16")
        covered.extend(range(n)[sl])
    assert covered == list(range(n))      # disjoint, ordered, complete
    # explicit slice + error cases
    assert br.resolve_slice(10, species_slice="2:5") == slice(2, 5)
    assert br.resolve_slice(10) == slice(0, 10)
    with pytest.raises(ValueError, match="mutually exclusive"):
        br.resolve_slice(10, shard="1/2", species_slice="0:1")
    with pytest.raises(ValueError, match="1 <= i <= N"):
        br.resolve_slice(10, shard="0/16")
    with pytest.raises(ValueError, match="1 <= i <= N"):
        br.resolve_slice(10, shard="17/16")


def test_generate_one_writes_density_only_npz(tmp_path, monkeypatch):
    calls = []
    _fake_stages(monkeypatch, calls)
    ms = _ms()
    status = br.generate_one(ms, out_dir=tmp_path, basis="def2-svp",
                             grid_level=2)
    assert status == "OK"
    with np.load(tmp_path / "H2O.npz", allow_pickle=False) as z:
        # exactly the density-only contract: NO vxc_ref/dm_target (the OEP
        # stage is skipped for benchmark refs)
        assert set(z.files) == set(br._DENSITY_NPZ_KEYS)
        assert z["rho_ref_grid"] == pytest.approx([1.0, 2.0, 3.0, 4.0])
        assert str(z["ref_density_method"]) == "ccsd"
        assert int(z["grid_level_used"]) == 2
        assert str(z["basis_used"]) == "def2-svp"
    # complete -> second call SKIPs without touching the stages again
    n_calls = len(calls)
    assert br.generate_one(ms, out_dir=tmp_path, basis="def2-svp",
                           grid_level=2) == "SKIP"
    assert len(calls) == n_calls


def test_generate_one_regenerates_on_basis_or_grid_mismatch(tmp_path,
                                                            monkeypatch):
    calls = []
    _fake_stages(monkeypatch, calls)
    ms = _ms()
    np.savez_compressed(tmp_path / "H2O.npz",
                        rho_ref_grid=np.ones(4),
                        ref_density_method=np.array("ccsd"),
                        grid_level_used=np.array(2),
                        basis_used=np.array("def2-tzvp"))
    # stale basis -> not complete -> regenerated for def2-svp
    assert br.generate_one(ms, out_dir=tmp_path, basis="def2-svp",
                           grid_level=2) == "OK"
    assert calls, "stale-basis npz must be regenerated, not skipped"
    with np.load(tmp_path / "H2O.npz", allow_pickle=False) as z:
        assert str(z["basis_used"]) == "def2-svp"
    # grid mismatch likewise
    assert br._benchmark_npz_is_complete(tmp_path / "H2O.npz",
                                         basis="def2-svp", grid_level=1) is False


def test_benchmark_npz_is_complete_rejects_corrupt_and_partial(tmp_path):
    p = tmp_path / "x.npz"
    p.write_bytes(b"not an npz")
    assert br._benchmark_npz_is_complete(p, basis="def2-svp",
                                         grid_level=2) is False
    np.savez_compressed(tmp_path / "y.npz", rho_ref_grid=np.ones(3))
    assert br._benchmark_npz_is_complete(tmp_path / "y.npz", basis="def2-svp",
                                         grid_level=2) is False  # missing keys
    assert br._benchmark_npz_is_complete(tmp_path / "missing.npz",
                                         basis="def2-svp", grid_level=2) is False


def test_run_shard_fail_continues_and_ledger_records(tmp_path, monkeypatch):
    calls = []
    _fake_stages(monkeypatch, calls)
    good = _ms(name="GOOD")
    bad = _ms(name="BAD", atom="H 0 0 0", comp=(("H", 1),))

    real_generate = br.generate_one

    def flaky(ms, **kw):
        if ms.name == "BAD":
            raise RuntimeError("ccsd exploded")
        return real_generate(ms, **kw)

    monkeypatch.setattr(br, "generate_one", flaky)
    n_fail = br.run_shard(["BAD", "GOOD"], {"GOOD": good, "BAD": bad},
                          out_dir=tmp_path, basis="def2-svp", grid_level=2,
                          shard_label="2/4", progress=False)
    assert n_fail == 1
    assert (tmp_path / "GOOD.npz").is_file()      # FAIL did not sink the shard
    assert not (tmp_path / "BAD.npz").exists()
    ledgers = list((tmp_path / "_runlogs" / "shard_2_of_4").glob("_run_log_*.json"))
    assert ledgers, "finalized RunLog ledger expected"
    results = json.loads(ledgers[0].read_text())["results"]
    by_name = {r["name"]: r for r in results}
    assert by_name["BAD"]["status"] == "FAIL"
    assert "ccsd exploded" in by_name["BAD"]["error_msg"]
    assert by_name["GOOD"]["status"] == "OK"


def test_load_benchmark_species_sorted_and_counted():
    bh76 = br.load_benchmark_species("bh76", basis="def2-svp", grid_level=2)
    both = br.load_benchmark_species("all", basis="def2-svp", grid_level=2)
    assert list(bh76) == sorted(bh76)
    assert list(both) == sorted(both)
    assert len(bh76) == 79
    assert len(both) == 214                       # 79 + 152 - 17 overlap
    with pytest.raises(ValueError, match="pool must be one of"):
        br.load_benchmark_species("nope")


def test_main_empty_slice_and_auxbasis_guard(tmp_path, capsys):
    rc = br.main(["--out-dir", str(tmp_path), "--pool", "bh76",
                  "--species-slice", "0:0"])
    assert rc == 0
    assert "empty slice" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        br.main(["--out-dir", str(tmp_path), "--auxbasis", "x"])  # no --density-fit


@pytest.mark.slow
def test_generate_one_h_atom_end_to_end(tmp_path):
    """Real SCF+CCSD on the H atom (cheap; exercises the empty-spin-channel
    non-DF CCSD path). The reference density must integrate to 1 electron."""
    ms = MoleculeSpec(name="h_test", atom="H 0.0 0.0 0.0", basis="def2-svp",
                      charge=0, spin=1, atom_composition=(("H", 1),),
                      external_data_path=None, grid_level=1)
    assert br.generate_one(ms, out_dir=tmp_path, basis="def2-svp",
                           grid_level=1, density_fit=True) == "OK"
    from xcquinox.alec.external_refs import _intermediate_cache_name
    scf_npz = (tmp_path / "_intermediates"
               / _intermediate_cache_name("h_test", grid_level=1,
                                          basis="def2-svp", density_fit=True,
                                          kind="scf"))
    with np.load(tmp_path / "h_test.npz") as z, np.load(scf_npz) as s:
        n_elec = float(np.sum(z["rho_ref_grid"] * s["grid_weights"]))
    assert n_elec == pytest.approx(1.0, abs=1e-3)
