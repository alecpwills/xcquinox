"""Tests for the density-only benchmark CCSD reference generator."""
import json

import numpy as np
import pytest

import xcquinox.pipeline.benchmark_refs as br
from xcquinox.pipeline.config import MoleculeSpec


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
                 density_fit=False, auxbasis=None,
                 orientation_lock_strength=0.0):
        calls.append(("scf", spec.name, density_fit, auxbasis))
        return {"dm": np.eye(2), "spin_unrestricted": False,
                "grid_coords": np.zeros((4, 3)), "grid_weights": np.ones(4)}

    def fake_ccsd(spec, atoms, *, scf_payload, cache_dir, basis, grid_level,
                  density_fit=False, auxbasis=None,
                  orientation_lock_strength=0.0):
        calls.append(("ccsd", spec.name, density_fit, auxbasis))
        return {"rho_ref_grid": np.array([1.0, 2.0, 3.0, 4.0]),
                "grid_weights": np.ones(4), "ao_grid": np.zeros((4, 2)),
                "dm_ao": np.eye(2)}

    monkeypatch.setattr(br, "run_scf_with_cache", fake_scf)
    monkeypatch.setattr(br, "run_ccsd_with_cache", fake_ccsd)


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
        # stage is skipped for benchmark refs); the optional provenance keys
        # (the convergence stamp, the T1 diagnostic when the CCSD stage
        # supplies it) are not part of the contract and may be present
        assert (set(z.files) - {"ccsd_converged", "t1_diagnostic"}
                == set(br._DENSITY_NPZ_KEYS))
        assert z["rho_ref_grid"] == pytest.approx([1.0, 2.0, 3.0, 4.0])
        # generator-side PBE density + weights stored for the SCF-free
        # PBE-vs-CCSD baseline (fake dm=I, ao=0 -> rho_pbe = 0)
        assert z["rho_pbe_grid"].shape == (4,)
        assert z["grid_weights"] == pytest.approx([1.0, 1.0, 1.0, 1.0])
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


@pytest.mark.slow
def test_generate_one_h_atom_end_to_end(tmp_path):
    """Real SCF+CCSD on the H atom (cheap; exercises the empty-spin-channel
    non-DF CCSD path). The reference density must integrate to 1 electron."""
    ms = MoleculeSpec(name="h_test", atom="H 0.0 0.0 0.0", basis="def2-svp",
                      charge=0, spin=1, atom_composition=(("H", 1),),
                      external_data_path=None, grid_level=1)
    assert br.generate_one(ms, out_dir=tmp_path, basis="def2-svp",
                           grid_level=1, density_fit=True) == "OK"
    with np.load(tmp_path / "h_test.npz") as z:
        n_elec = float(np.sum(z["rho_ref_grid"] * z["grid_weights"]))
        n_elec_pbe = float(np.sum(z["rho_pbe_grid"] * z["grid_weights"]))
    assert n_elec == pytest.approx(1.0, abs=1e-3)
    assert n_elec_pbe == pytest.approx(1.0, abs=1e-3)


def test_run_shard_threads_orientation_lock_strength(tmp_path, monkeypatch):
    """run_shard forwards orientation_lock_strength to generate_one so the
    held-out refs lock the same degenerate component as the training refs."""
    seen = {}

    def fake_generate_one(ms, **kw):
        seen["ol"] = kw.get("orientation_lock_strength")
        return "OK"

    monkeypatch.setattr(br, "generate_one", fake_generate_one)
    br.run_shard(["X"], {"X": _ms(name="X")}, out_dir=tmp_path, basis="def2-svp",
                 grid_level=2, orientation_lock_strength=3e-5, progress=False)
    assert seen["ol"] == 3e-5


def test_the_final_npz_writer_refuses_a_key_outside_its_declared_set(tmp_path):
    """Every key a final reference npz may carry is declared in ``NPZ_KEYS``, the
    set the loader's whitelist is checked against; a key written without being
    declared would be a reference no evaluation reads, so the one
    writer of final files, the atomic save itself, refuses it."""
    ok = tmp_path / "ok.npz"
    br._atomic_savez(ok, rho_ref_grid=np.zeros(3),
                     t1_diagnostic=np.array(0.01), ccsd_converged=np.array(True))
    with np.load(ok, allow_pickle=False) as z:
        assert set(z.files) == {"rho_ref_grid", "t1_diagnostic",
                                "ccsd_converged"}
    with pytest.raises(RuntimeError, match="bogus"):
        br._atomic_savez(tmp_path / "bad.npz", rho_ref_grid=np.zeros(3),
                         bogus=np.zeros(1))
    assert not (tmp_path / "bad.npz").exists()
    assert not list(tmp_path.glob("tmp*.npz"))


def test_the_pool_argument_takes_a_comma_list_and_keeps_the_historical_names():
    """The job template passes ``--pool all``, which stays the BH76 + W4-11 union; a
    comma list covers the species of several pools in one reference build, and an
    unknown name is refused rather than generating a smaller set than was asked for.

    Oracle: the species names the loader returns per pool selection.
    """
    assert "all" in br.POOL_CHOICES
    everything = br.load_benchmark_species("all")
    pair = br.load_benchmark_species("bh76,w411")
    assert sorted(pair) == sorted(everything)
    only_bh76 = br.load_benchmark_species("bh76")
    assert set(only_bh76) < set(everything)
    assert list(everything) == sorted(everything)
    with pytest.raises(ValueError):
        br.load_benchmark_species("bh76,bh77")


def test_the_size_cap_drops_the_largest_species_and_names_them(capsys):
    """A cap bounds what the reference build will spend on one species. The species it
    drops are named, so a reference set that is smaller than the pool is readable as a
    capped set rather than as a failed one.

    Oracle: the species counts of the uncapped pool.
    """
    uncapped = br.load_benchmark_species("bh76")
    n_atoms = {name: sum(int(c) for _e, c in ms.atom_composition)
               for name, ms in uncapped.items()}
    cap = 3
    capped = br.load_benchmark_species("bh76", max_atoms=cap)
    assert set(capped) == {n for n, k in n_atoms.items() if k <= cap}
    assert 0 < len(capped) < len(uncapped)
    dropped = sorted(n for n, k in n_atoms.items() if k > cap)
    out = capsys.readouterr().out
    for name in dropped[:3]:
        assert name in out, name


def test_listing_the_species_runs_no_reference_generation(tmp_path, monkeypatch,
                                                          capsys):
    """The sizing question is asked before a job is submitted, so the listing exits
    without generating anything.

    Oracle: a ``generate_one`` that fails the test if it is reached.
    """
    monkeypatch.setattr(br, "generate_one", lambda *a, **k: pytest.fail(
        "generate_one ran under --list-species"))
    rc = br.main(["--out-dir", str(tmp_path), "--pool", "bh76",
                  "--max-atoms", "3", "--list-species"])
    assert rc == 0
    out = capsys.readouterr().out
    assert not list(tmp_path.glob("*.npz"))
    names = br.load_benchmark_species("bh76", max_atoms=3)
    assert str(len(names)) in out
    for name in list(names)[:3]:
        assert name in out, name




