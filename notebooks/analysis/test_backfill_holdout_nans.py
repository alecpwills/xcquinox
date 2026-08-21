"""Tests for ``backfill_holdout_nans.py`` -- the local repair driver that
recomputes NN-NaN holdout species under the production eval identity and
patches them into the channel's ``per_molecule.json``.

Pure-logic layer only: the SCF itself is exercised through the worker CLI at
run time (and gated on control-species reproduction there); here the compute
seam is injected, mirroring ``reeval_holdout_fixed``'s injectable-callable
style.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


bf = _load("backfill_holdout_nans")


# ---------------------------------------------------------------------------
# Fixtures: synthetic per_molecule records + a worker payload
# ---------------------------------------------------------------------------

def _rec(name, e_nn, e_pbe, *, training=False, dens=1e-4, cycles=3,
         steps=None):
    r = {
        "molecule": name,
        "E_total_nn": e_nn,
        "E_pbe": e_pbe,
        "AE_nn": (e_nn - e_pbe) if e_nn is not None else None,
        "AE_error_kcalmol": None,
        "density_rmse": dens,
        "density_l1": dens / 10,
        "density_rmse_pbe": 2 * dens,
        "density_l1_pbe": dens / 5,
        "density_eps_l1": 5 * dens,
        "density_eps_l1_pbe": 6 * dens,
        "n_electrons": 10.0,
        "grid_weight_sum": 1000.0,
        "ref_density_method": "ccsd",
        "cycles_run": cycles if e_nn is not None else 0,
        "scf_converged": e_nn is None,
        "from_training_subset": training,
    }
    if steps:
        r["scf_total_energy"] = steps[-1]
        for i, e in enumerate(steps):
            r[f"scf_energy_step_{i}"] = e
            r[f"scf_energy_residual_{i}"] = abs(e - steps[-1])
    return r


def _payload_rec(name, e_nn, e_pbe, steps=(-1.01, -1.02, -1.0)):
    r = {
        "molecule": name,
        "E_total_nn": e_nn,
        "E_pbe": e_pbe,
        "AE_nn": (e_nn - e_pbe) if e_nn is not None else None,
        "AE_error_kcalmol": None,
        "density_rmse": None,          # local box has no bench refs
        "density_l1": None,
        "density_rmse_pbe": None,
        "density_l1_pbe": None,
        "density_eps_l1": None,
        "density_eps_l1_pbe": None,
        "n_electrons": None,
        "grid_weight_sum": None,
        "ref_density_method": None,
        "cycles_run": 3 if e_nn is not None else 0,
        "scf_converged": False,
        "from_training_subset": False,
    }
    if e_nn is not None:
        r["scf_total_energy"] = steps[-1]
        for i, e in enumerate(steps):
            r[f"scf_energy_step_{i}"] = e
            r[f"scf_energy_residual_{i}"] = abs(e - steps[-1])
    return r


def _base_records():
    # two finite held-out species (controls), one finite training species,
    # two NaN species (one healthy to backfill, one whose PBE will mismatch)
    return [
        _rec("aa_ctl", -10.0, -9.9),
        _rec("bb_ctl", -20.0, -19.9),
        _rec("h2o", -76.5, -76.4, training=True,
             steps=[-76.51, -76.49, -76.5]),
        _rec("mm_nan", None, -30.0),
        _rec("pp_nan", None, -40.0),
    ]


def _good_payload():
    return {
        "energies": {"aa_ctl": -10.0, "bb_ctl": -20.0,
                     "mm_nan": -30.1, "pp_nan": -40.1},
        "pbe_energies": {"aa_ctl": -9.9, "bb_ctl": -19.9,
                         "mm_nan": -30.0, "pp_nan": -40.0 + 5e-4},
        "mol_records": [
            _payload_rec("aa_ctl", -10.0, -9.9),
            _payload_rec("bb_ctl", -20.0, -19.9),
            _payload_rec("mm_nan", -30.1, -30.0,
                         steps=(-30.09, -30.11, -30.1)),
            _payload_rec("pp_nan", -40.1, -40.0 + 5e-4),
        ],
    }


# ---------------------------------------------------------------------------
# Target discovery / controls
# ---------------------------------------------------------------------------

def test_nonfinite_species_finds_null_and_nan():
    recs = _base_records()
    recs.append({"molecule": "qq", "E_total_nn": float("nan"),
                 "E_pbe": -1.0})
    recs.append({"molecule": "rr", "E_pbe": -1.0})       # key missing
    assert bf.nonfinite_species(recs) == ["mm_nan", "pp_nan", "qq", "rr"]


def test_pick_controls_deterministic_holdout_only():
    recs = _base_records()
    ctl = bf.pick_controls(recs, 2)
    assert ctl == sorted(ctl)
    assert set(ctl) <= {"aa_ctl", "bb_ctl"}      # training species excluded
    assert bf.pick_controls(recs, 2) == ctl      # deterministic
    assert bf.pick_controls(recs, 99) == ["aa_ctl", "bb_ctl"]


# ---------------------------------------------------------------------------
# Patch logic
# ---------------------------------------------------------------------------

def test_patch_records_backfills_energy_and_preserves_density():
    recs = _base_records()
    new, rep = bf.patch_records(recs, _good_payload(),
                                controls=["aa_ctl", "bb_ctl"],
                                gate_nn=1e-6, gate_pbe=1e-6)
    assert not rep["aborted"]
    by = {r["molecule"]: r for r in new}
    mm = by["mm_nan"]
    assert mm["E_total_nn"] == -30.1
    # AE_nn recomputed against the RECORDED E_pbe, which is preserved
    assert mm["E_pbe"] == -30.0
    assert mm["AE_nn"] == pytest.approx(-0.1)
    # density columns kept verbatim (locally unreproducible)
    assert mm["density_rmse"] == 1e-4
    assert mm["ref_density_method"] == "ccsd"
    # SCF trace carried over from the recomputation
    assert mm["cycles_run"] == 3
    assert mm["scf_total_energy"] == -30.1
    assert mm["scf_energy_step_1"] == -30.11
    assert "mm_nan" in rep["patched"]


def test_patch_records_drops_stale_eval_error():
    """A NaN row written by the cluster eval names the exception that produced
    it (``eval_error``). Once the species is recomputed successfully that text
    describes an evaluation the row no longer holds, so it must go with the
    other recomputation-owned fields."""
    recs = _base_records()
    for r in recs:
        if r["molecule"] == "mm_nan":
            r["eval_error"] = "XlaRuntimeError: RESOURCE_EXHAUSTED"
            r["cycles_run"] = None
            r["scf_converged"] = None
    new, rep = bf.patch_records(recs, _good_payload(),
                                controls=["aa_ctl", "bb_ctl"],
                                gate_nn=1e-6, gate_pbe=1e-6)
    assert "mm_nan" in rep["patched"]
    by = {r["molecule"]: r for r in new}
    assert "eval_error" not in by["mm_nan"]
    assert by["mm_nan"]["cycles_run"] == 3
    assert by["mm_nan"]["E_total_nn"] == -30.1


def test_patch_records_keeps_eval_error_on_unpatched_row():
    """The failure text stays on a row the driver refuses to patch."""
    recs = _base_records()
    for r in recs:
        if r["molecule"] == "pp_nan":            # PBE mismatch -> skipped
            r["eval_error"] = "XlaRuntimeError: RESOURCE_EXHAUSTED"
    new, rep = bf.patch_records(recs, _good_payload(),
                                controls=["aa_ctl", "bb_ctl"],
                                gate_nn=1e-6, gate_pbe=1e-6)
    assert "pp_nan" in rep["skipped"]
    by = {r["molecule"]: r for r in new}
    assert by["pp_nan"]["eval_error"].startswith("XlaRuntimeError:")


def test_patch_records_gates_species_on_pbe_mismatch():
    # pp_nan's local PBE landed 5e-4 Ha away (a c2-class multi-solution):
    # the species is skipped, left NaN, and named with its delta.
    recs = _base_records()
    new, rep = bf.patch_records(recs, _good_payload(),
                                controls=["aa_ctl", "bb_ctl"],
                                gate_nn=1e-6, gate_pbe=1e-6)
    by = {r["molecule"]: r for r in new}
    assert by["pp_nan"]["E_total_nn"] is None
    assert "pp_nan" in rep["skipped"]
    assert rep["skipped"]["pp_nan"]["reason"] == "pbe-mismatch"
    assert rep["skipped"]["pp_nan"]["dE_pbe"] == pytest.approx(5e-4)


def test_patch_records_aborts_on_control_mismatch():
    payload = _good_payload()
    payload["energies"]["aa_ctl"] = -10.0 + 1e-3     # NN not reproduced
    recs = _base_records()
    new, rep = bf.patch_records(recs, payload,
                                controls=["aa_ctl", "bb_ctl"],
                                gate_nn=1e-6, gate_pbe=1e-6)
    assert rep["aborted"] and "aa_ctl" in rep["abort_reason"]
    assert new is None
    assert rep["controls"]["aa_ctl"]["dE_nn"] == pytest.approx(1e-3)


def test_patch_records_aborts_on_nonfinite_control():
    payload = _good_payload()
    payload["energies"]["bb_ctl"] = float("nan")
    _new, rep = bf.patch_records(_base_records(), payload,
                                 controls=["aa_ctl", "bb_ctl"],
                                 gate_nn=1e-6, gate_pbe=1e-6)
    assert rep["aborted"]


def test_patch_records_keeps_locally_diverged_species_nan():
    payload = _good_payload()
    payload["energies"]["mm_nan"] = float("nan")
    payload["mol_records"][2] = _payload_rec("mm_nan", None, -30.0)
    new, rep = bf.patch_records(_base_records(), payload,
                                controls=["aa_ctl", "bb_ctl"],
                                gate_nn=1e-6, gate_pbe=1e-6)
    by = {r["molecule"]: r for r in new}
    assert by["mm_nan"]["E_total_nn"] is None
    assert "mm_nan" in rep["diverged"]


def test_patch_records_replaces_stale_scf_keys():
    recs = _base_records()
    # pretend an earlier (longer) trace exists on the NaN record
    recs[3]["scf_energy_step_0"] = -1.0
    recs[3]["scf_energy_step_7"] = -2.0
    recs[3]["scf_energy_residual_7"] = 0.5
    new, _rep = bf.patch_records(recs, _good_payload(),
                                 controls=["aa_ctl", "bb_ctl"],
                                 gate_nn=1e-6, gate_pbe=1e-6)
    mm = {r["molecule"]: r for r in new}["mm_nan"]
    assert "scf_energy_step_7" not in mm
    assert "scf_energy_residual_7" not in mm
    assert mm["scf_energy_step_2"] == -30.1


def test_patch_records_reports_targets_absent_from_payload():
    # A worker can drop a requested species entirely (a per-species
    # precompute failure inside the shard process): the target must be
    # named as unresolved, never silently left NaN with no report entry.
    payload = _good_payload()
    payload["energies"].pop("mm_nan")
    payload["pbe_energies"].pop("mm_nan")
    payload["mol_records"] = [r for r in payload["mol_records"]
                              if r["molecule"] != "mm_nan"]
    new, rep = bf.patch_records(_base_records(), payload,
                                controls=["aa_ctl", "bb_ctl"],
                                gate_nn=1e-6, gate_pbe=1e-6)
    assert rep["unresolved"] == ["mm_nan"]
    assert "mm_nan" not in rep["patched"]
    by = {r["molecule"]: r for r in new}
    assert by["mm_nan"]["E_total_nn"] is None
    # every target lands in exactly one bucket
    n = (len(rep["patched"]) + len(rep["skipped"])
         + len(rep["diverged"]) + len(rep["unresolved"]))
    assert n == 2


def test_worker_tag_unique_per_chunk():
    # Equal-length chunks of one (spec, model) must not share work-file
    # names -- the tag carries the caller's chunk ordinal.
    a = bf._worker_tag(45, "model.eqx", ["x", "y"], seq=0)
    b = bf._worker_tag(45, "model.eqx", ["p", "q"], seq=1)
    assert a != b
    assert a.startswith("s0045_model_")


def test_patch_records_aborts_without_controls():
    # An empty control list means the identity was never confirmed: the
    # channel must abort, never write. Covers both --controls 0 and the
    # silent route (every finite species in the channel is a
    # training-subset row, so pick_controls returns []).
    new, rep = bf.patch_records(_base_records(), _good_payload(),
                                controls=[], gate_nn=1e-6, gate_pbe=1e-6)
    assert rep["aborted"] and "control" in rep["abort_reason"]
    assert new is None


def test_cli_rejects_low_control_count(tmp_path):
    with pytest.raises(SystemExit):
        bf.main([str(tmp_path), "--controls", "1"])


def test_ledger_gate_rejected_target_falls_through_to_compute(tmp_path):
    # A ledger that covers a target only with a gate-rejected value must
    # not deadlock the channel as nothing-to-do: the channel recomputes.
    ch = _channel_dir(tmp_path, _base_records())
    bad = _good_payload()
    bad["pbe_energies"]["mm_nan"] = -30.0 + 5e-4     # will gate-skip
    bad["pbe_energies"]["pp_nan"] = -40.0 + 5e-4     # will gate-skip
    bf.save_ledger(ch, bad)
    calls = []

    def _fake_compute(names):
        calls.append(sorted(names))
        return _good_payload()          # fresh run: mm_nan passes the gate

    rep = bf.process_channel_records(ch, controls=["aa_ctl", "bb_ctl"],
                                     gate_nn=1e-6, gate_pbe=1e-6,
                                     compute_fn=_fake_compute)
    assert calls, "compute must be reached when the ledger cannot patch"
    assert rep["source"] == "compute"
    assert "mm_nan" in rep["patched"]
    assert "pp_nan" in rep["skipped"]           # still gated on fresh data
    assert (ch / "backfill_meta.json").is_file()


def test_stamp_merges_across_passes(tmp_path):
    # A retry pass must extend the stamp, not erase the earlier pass's
    # per-species outcomes; a later patch clears the same species from
    # the skipped/unresolved buckets.
    ch = _channel_dir(tmp_path, _base_records())

    def _first(names):
        p = _good_payload()
        # mm_nan patches; pp_nan gate-skips (payload pbe 5e-4 off)
        return p

    bf.process_channel_records(ch, controls=["aa_ctl", "bb_ctl"],
                               gate_nn=1e-6, gate_pbe=1e-6,
                               compute_fn=_first)
    s1 = json.loads((ch / "backfill_meta.json").read_text())
    assert "mm_nan" in s1["patched"] and "pp_nan" in s1["skipped"]

    def _second(names):
        p = _good_payload()
        p["pbe_energies"]["pp_nan"] = -40.0     # now reproduces exactly
        return p

    # drop the stale ledger so the retry recomputes pp_nan
    (ch / "backfill_ledger.json").unlink()
    bf.process_channel_records(ch, controls=["aa_ctl", "bb_ctl"],
                               gate_nn=1e-6, gate_pbe=1e-6,
                               compute_fn=_second)
    s2 = json.loads((ch / "backfill_meta.json").read_text())
    assert "mm_nan" in s2["patched"]            # first pass survives
    assert "pp_nan" in s2["patched"]            # second pass added
    assert "pp_nan" not in s2["skipped"]        # resolved names leave


def test_patch_records_unresolved_when_record_missing():
    # energies carries the name but mol_records does not: refusing beats
    # writing a schema-incomplete row (no cycles_run/scf_converged).
    payload = _good_payload()
    payload["mol_records"] = [r for r in payload["mol_records"]
                              if r["molecule"] != "mm_nan"]
    new, rep = bf.patch_records(_base_records(), payload,
                                controls=["aa_ctl", "bb_ctl"],
                                gate_nn=1e-6, gate_pbe=1e-6)
    assert "mm_nan" in rep["unresolved"]
    by = {r["molecule"]: r for r in new}
    assert by["mm_nan"]["E_total_nn"] is None
    assert by["mm_nan"]["cycles_run"] == 0      # row untouched


def _make_mini_run(tmp_path):
    run = tmp_path / "mini/runs/run_x"
    (run / "checkpoints/spec_0001/eval_holdout").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps(
        {"n_specs": 2, "width": 4, "specs": []}))
    (run / "resolved_config.yaml").write_text(
        "inputs:\n  basis: def2-svp\n  grid_level: 1\n")
    with (run / "checkpoints/spec_0001/eval_holdout"
          / "per_molecule.json").open("w") as f:
        json.dump(_base_records(), f, indent=2)
    return run


def test_main_exit_code_flags_unresolved(tmp_path, monkeypatch):
    run = _make_mini_run(tmp_path)
    payload = _good_payload()
    # worker returns nothing at all for pp_nan
    for k in ("energies", "pbe_energies"):
        payload[k].pop("pp_nan")
    payload["mol_records"] = [r for r in payload["mol_records"]
                              if r["molecule"] != "pp_nan"]
    monkeypatch.setattr(bf, "run_worker",
                        lambda *a, **k: payload)
    rc = bf.main([str(run), "--specs", "1", "--channels", "eval_holdout",
                  "--no-refinalize"])
    assert rc == 1                              # unresolved -> non-zero
    recs = json.loads((run / "checkpoints/spec_0001/eval_holdout"
                       / "per_molecule.json").read_text())
    assert {r["molecule"]: r for r in recs}["mm_nan"]["E_total_nn"] == -30.1


def test_patch_records_idempotent_when_no_targets():
    recs = [_rec("aa_ctl", -10.0, -9.9)]
    new, rep = bf.patch_records(recs, {"energies": {}, "pbe_energies": {},
                                       "mol_records": []},
                                controls=[], gate_nn=1e-6, gate_pbe=1e-6)
    assert new == recs and not rep["patched"] and not rep["aborted"]


# ---------------------------------------------------------------------------
# File plumbing: backup, atomic write, ledger, stamp
# ---------------------------------------------------------------------------

def _channel_dir(tmp_path, records):
    ch = tmp_path / "checkpoints/spec_0007/eval_holdout"
    ch.mkdir(parents=True)
    with (ch / "per_molecule.json").open("w") as f:
        json.dump(records, f, indent=2)
    return ch


def test_write_patched_backs_up_once_and_replaces_atomically(tmp_path):
    ch = _channel_dir(tmp_path, _base_records())
    first = [_rec("aa_ctl", -1.0, -0.9)]
    bf.write_patched(ch, first)
    backup = ch / "per_molecule.pre_backfill.json"
    assert backup.is_file()
    original = json.loads(backup.read_text())
    assert {r["molecule"] for r in original} == {
        "aa_ctl", "bb_ctl", "h2o", "mm_nan", "pp_nan"}
    # second write must NOT overwrite the original backup
    bf.write_patched(ch, [_rec("zz", -2.0, -1.9)])
    assert json.loads(backup.read_text()) == original
    assert json.loads((ch / "per_molecule.json").read_text())[0][
        "molecule"] == "zz"


def test_ledger_roundtrip_and_merge(tmp_path):
    ch = _channel_dir(tmp_path, _base_records())
    bf.save_ledger(ch, _good_payload())
    second = {"energies": {"qq": -5.0}, "pbe_energies": {"qq": -4.9},
              "mol_records": [_payload_rec("qq", -5.0, -4.9)]}
    bf.save_ledger(ch, second)
    led = bf.load_ledger(ch)
    assert led["energies"]["mm_nan"] == -30.1     # first payload kept
    assert led["energies"]["qq"] == -5.0          # second merged in
    assert {r["molecule"] for r in led["mol_records"]} >= {"mm_nan", "qq"}


def test_process_channel_ledger_reapply_needs_no_compute(tmp_path):
    ch = _channel_dir(tmp_path, _base_records())
    bf.save_ledger(ch, _good_payload())

    def _boom(names):  # compute seam must not be reached
        raise AssertionError(f"compute called for {names}")

    rep = bf.process_channel_records(ch, controls=["aa_ctl", "bb_ctl"],
                                     gate_nn=1e-6, gate_pbe=1e-6,
                                     compute_fn=_boom)
    assert rep["patched"] and "mm_nan" in rep["patched"]
    assert rep["source"] == "ledger"
    recs = json.loads((ch / "per_molecule.json").read_text())
    assert {r["molecule"]: r for r in recs}["mm_nan"]["E_total_nn"] == -30.1
    stamp = json.loads((ch / "backfill_meta.json").read_text())
    assert stamp["patched"] and stamp["gates"]["gate_pbe"] == 1e-6
    # second call: the ledger covers the remaining target only with a
    # gate-rejected value, so the channel RECOMPUTES rather than
    # reporting nothing-to-do (the deadlock the ledger short-circuit
    # would otherwise create); the seam raising proves compute is
    # reached.
    with pytest.raises(AssertionError, match="compute called"):
        bf.process_channel_records(ch, controls=["aa_ctl", "bb_ctl"],
                                   gate_nn=1e-6, gate_pbe=1e-6,
                                   compute_fn=_boom)


def test_process_channel_computes_when_ledger_missing(tmp_path):
    ch = _channel_dir(tmp_path, _base_records())
    calls = []

    def _fake_compute(names):
        calls.append(sorted(names))
        return _good_payload()

    rep = bf.process_channel_records(ch, controls=["aa_ctl", "bb_ctl"],
                                     gate_nn=1e-6, gate_pbe=1e-6,
                                     compute_fn=_fake_compute)
    assert rep["source"] == "compute"
    # one call covering the NaN targets plus the controls
    assert calls == [["aa_ctl", "bb_ctl", "mm_nan", "pp_nan"]]
    assert "mm_nan" in rep["patched"]
    # the payload is banked as the ledger for clobber-recovery
    assert (ch / "backfill_ledger.json").is_file()


def test_process_channel_dry_run_touches_nothing(tmp_path):
    ch = _channel_dir(tmp_path, _base_records())
    before = (ch / "per_molecule.json").read_text()

    def _fake_compute(names):
        return _good_payload()

    rep = bf.process_channel_records(ch, controls=["aa_ctl", "bb_ctl"],
                                     gate_nn=1e-6, gate_pbe=1e-6,
                                     compute_fn=_fake_compute, dry_run=True)
    assert rep["status"] == "would-patch"
    assert (ch / "per_molecule.json").read_text() == before
    assert not (ch / "backfill_meta.json").exists()


# ---------------------------------------------------------------------------
# Config parsing / CLI plumbing
# ---------------------------------------------------------------------------

def test_read_basis_grid_parses_resolved_config(tmp_path):
    (tmp_path / "resolved_config.yaml").write_text(
        "cluster:\n  eval_time: 02:00:00\n"
        "inputs:\n  basis: 6-311++G(3df,2pd)\n  grid_level: 3\n"
        "  density_fit: true\n")
    assert bf.read_basis_grid(tmp_path) == ("6-311++G(3df,2pd)", 3)


def test_channel_models_cover_the_three_channels():
    assert bf.CHANNEL_MODELS == {
        "eval_holdout": "model.eqx",
        "eval_holdout_best": "model_best.eqx",
        "eval_holdout_val_best": "model_val_best.eqx",
    }


# ---------------------------------------------------------------------------
# A sliced channel is refused before any read or rewrite
# ---------------------------------------------------------------------------

_SLICE = ["h", "h2", "o", "oh", "n2o", "n2ohts"]


def _sliced_marker(ch):
    (ch / "sliced_eval.json").write_text(json.dumps(
        {"species_slice": list(_SLICE), "n_species": len(_SLICE),
         "n_reactions": 1, "env_var": "XCQUINOX_HELDOUT_SPECIES_SLICE"}))


def _sliced_stamp(ch):
    (ch / "eval_metadata.json").write_text(json.dumps(
        {"channel": "eval_holdout", "species_slice": list(_SLICE),
         "n_species": len(_SLICE), "n_reactions": 1}))


@pytest.mark.parametrize("mark", [_sliced_marker, _sliced_stamp])
def test_process_channel_records_refuses_a_sliced_channel(tmp_path, mark):
    """The backfill recomputes species under the PRODUCTION eval identity and
    patches them into the channel. On a channel evaluated over a
    workflow-verification species slice that identity does not hold -- the
    energies beside the patched ones came from a different reaction set -- so
    the channel is refused before it is read, let alone rewritten."""
    from xcquinox.alec.eval_holdout import SlicedChannelError
    ch = _channel_dir(tmp_path, _base_records())
    mark(ch)
    before = (ch / "per_molecule.json").read_bytes()
    calls = []

    def _fake_compute(names):
        calls.append(sorted(names))
        return _good_payload()

    with pytest.raises(SlicedChannelError) as exc:
        bf.process_channel_records(ch, controls=["aa_ctl", "bb_ctl"],
                                   gate_nn=1e-6, gate_pbe=1e-6,
                                   compute_fn=_fake_compute)
    msg = str(exc.value)
    assert "spec_0007" in msg
    assert "eval_holdout" in msg
    assert "'n2ohts'" in msg
    # nothing computed, nothing read into a rewrite, nothing written
    assert calls == []
    assert (ch / "per_molecule.json").read_bytes() == before
    assert not (ch / "per_molecule.pre_backfill.json").exists()
    assert not (ch / "backfill_ledger.json").exists()
    assert not (ch / "backfill_meta.json").exists()


def test_process_channel_records_refuses_a_sliced_channel_dry_run(tmp_path):
    """--dry-run reads the channel too; the refusal precedes that read."""
    from xcquinox.alec.eval_holdout import SlicedChannelError
    ch = _channel_dir(tmp_path, _base_records())
    _sliced_marker(ch)
    with pytest.raises(SlicedChannelError):
        bf.process_channel_records(ch, controls=["aa_ctl", "bb_ctl"],
                                   gate_nn=1e-6, gate_pbe=1e-6,
                                   compute_fn=lambda names: _good_payload(),
                                   dry_run=True)


def test_process_channel_records_passes_a_full_pool_stamp(tmp_path):
    """The guard is a no-op on the mark a full-pool evaluation writes."""
    ch = _channel_dir(tmp_path, _base_records())
    (ch / "eval_metadata.json").write_text(json.dumps(
        {"channel": "eval_holdout", "species_slice": None}))
    rep = bf.process_channel_records(ch, controls=["aa_ctl", "bb_ctl"],
                                     gate_nn=1e-6, gate_pbe=1e-6,
                                     compute_fn=lambda names: _good_payload())
    assert rep["status"] == "patched"
