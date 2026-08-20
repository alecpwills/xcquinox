"""Parallel held-out eval: refactor seams (merge + finalize), the degradation
ladder, escalation/retry, graceful fallback, runtime CPU detection, and the
eval_workers config knob.

The per-molecule eval is parallelized across molecule shards (subprocess
workers via xcquinox.alec.parallel.run_workers), with an adaptive ladder that
retries failed molecules at lower parallelism and ends in in-process serial.
These tests use synthetic data + monkeypatched run_workers (no real SCF /
subprocess) except the explicitly-slow end-to-end smoke.
"""
import json
import math
import os

import pytest

import xcquinox.alec.eval_holdout as eh
from xcquinox.alec import parallel as par
from xcquinox.alec.workers import eval_holdout_worker as ehw


# ---------------------------------------------------------------------------
# Queue-agnostic CPU detection
# ---------------------------------------------------------------------------

def test_detect_available_cpus_prefers_affinity(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3, 4},
                        raising=False)
    assert par.detect_available_cpus() == 5


def test_detect_available_cpus_falls_back_to_slurm_env(monkeypatch):
    def _boom(pid):
        raise OSError("no affinity")
    monkeypatch.setattr(os, "sched_getaffinity", _boom, raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "12")
    assert par.detect_available_cpus() == 12


def test_detect_available_cpus_final_fallback_cpu_count(monkeypatch):
    def _boom(pid):
        raise OSError("no affinity")
    monkeypatch.setattr(os, "sched_getaffinity", _boom, raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 7)
    assert par.detect_available_cpus() == 7


# ---------------------------------------------------------------------------
# Degradation ladder
# ---------------------------------------------------------------------------

def test_eval_worker_ladder_24():
    assert par.eval_worker_ladder(24) == [(24, 1), (12, 2), (6, 4)]


def test_eval_worker_ladder_8():
    assert par.eval_worker_ladder(8) == [(8, 1), (4, 2), (2, 4)]


def test_eval_worker_ladder_single_core_is_empty():
    assert par.eval_worker_ladder(1) == []          # serial only


def test_eval_worker_ladder_top_override_caps_start():
    # top=1 forces serial-only; top below cpus starts the ladder lower
    assert par.eval_worker_ladder(24, top=1) == []
    assert par.eval_worker_ladder(24, top=8) == [(8, 3), (4, 6), (2, 12)]


# ---------------------------------------------------------------------------
# Config knob + resolution
# ---------------------------------------------------------------------------

def test_cluster_resources_eval_workers_defaults_none():
    from xcquinox.alec.cluster.grid_config import ClusterResources
    cl = ClusterResources(
        partition="p", time="01:00:00", mem="8G", cpus_per_task=4,
        array_throttle=2, eval_array_throttle=2, max_concurrent_tasks=4)
    assert cl.eval_workers is None


def test_build_cluster_parses_eval_workers():
    from xcquinox.alec.cluster.grid_config import _build_cluster
    cl = _build_cluster({
        "partition": "p", "time": "01:00:00", "mem": "8G",
        "cpus_per_task": 4, "array_throttle": 2, "eval_array_throttle": 2,
        "max_concurrent_tasks": 4, "eval_workers": 6})
    assert cl.eval_workers == 6


def test_resolve_eval_workers_auto_uses_detected_cpus(monkeypatch):
    from xcquinox.alec.cluster.grid_config import (
        ClusterResources, _resolve_eval_workers)
    monkeypatch.setattr(par, "detect_available_cpus", lambda: 16)
    cl = ClusterResources(
        partition="p", time="01:00:00", mem="8G", cpus_per_task=4,
        array_throttle=2, eval_array_throttle=2, max_concurrent_tasks=4,
        eval_workers=None)
    # auto = detected cpus, capped at the molecule count
    assert _resolve_eval_workers(cl, n_molecules=200) == 16
    assert _resolve_eval_workers(cl, n_molecules=5) == 5


def test_resolve_eval_workers_explicit_override(monkeypatch):
    from xcquinox.alec.cluster.grid_config import (
        ClusterResources, _resolve_eval_workers)
    monkeypatch.setattr(par, "detect_available_cpus", lambda: 96)
    cl = ClusterResources(
        partition="p", time="01:00:00", mem="8G", cpus_per_task=4,
        array_throttle=2, eval_array_throttle=2, max_concurrent_tasks=4,
        eval_workers=8)
    assert _resolve_eval_workers(cl, n_molecules=200) == 8


# ---------------------------------------------------------------------------
# Shard worker
# ---------------------------------------------------------------------------

def test_compute_shard_evaluates_only_named_subset(monkeypatch):
    import xcquinox.alec.cluster._eval_one_spec as eos
    import xcquinox.alec.full_benchmark_pools as fbp
    monkeypatch.setattr(eos, "_read_width", lambda rd: 3)
    monkeypatch.setattr(eos, "_checkpoint_dir", lambda rd, i, w: "/ckpt")
    monkeypatch.setattr(eos, "_spec_path", lambda rd, i, w: "/spec.pkl")
    monkeypatch.setattr(eos, "_load_spec", lambda p: "TSPEC")
    monkeypatch.setattr(eh, "load_trained_model", lambda ts, mp: "MODEL")
    full = {"h2": "s_h2", "h": "s_h", "o": "s_o"}
    monkeypatch.setattr(fbp, "load_full_held_out_pools",
                        lambda *, basis, grid_level: (full, []))

    captured = {}

    def fake_compute(ts, model, subset):
        captured["subset"] = dict(subset)
        return {"energies": {n: -1.0 for n in subset},
                "pbe_energies": {n: -0.9 for n in subset},
                "mol_records": [{"molecule": n} for n in subset]}
    monkeypatch.setattr(eh, "compute_holdout_per_molecule", fake_compute)

    shard = ehw.compute_shard("/run", 2, ["h2", "h"], "def2-svp", 1)
    assert set(captured["subset"]) == {"h2", "h"}      # only requested names
    assert set(shard["energies"]) == {"h2", "h"}
    assert "o" not in shard["energies"]                # other shards own 'o'


def test_worker_main_writes_shard_and_prints_success(tmp_path, monkeypatch, capsys):
    names_file = tmp_path / "names.json"
    names_file.write_text(json.dumps(["h2", "h"]))
    out_shard = tmp_path / "shard.json"
    monkeypatch.setattr(
        ehw, "compute_shard",
        lambda rd, idx, names, basis, gl, model_name="model.eqx",
        coldstart=False: {
            "energies": {"h2": -1.17}, "pbe_energies": {"h2": -1.16},
            "mol_records": [{"molecule": "h2"}]})

    rc = ehw.main(["--run-dir", "/run", "--spec-idx", "2",
                   "--names-file", str(names_file), "--out-shard", str(out_shard),
                   "--basis", "def2-svp", "--grid-level", "1", "--threads", "1"])
    assert rc == 0
    assert json.loads(out_shard.read_text())["energies"] == {"h2": -1.17}
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(last)["status"] == "success"


def test_worker_main_forwards_model_name(tmp_path, monkeypatch):
    # --model-name defaults to model.eqx (final) and is forwarded to compute_shard
    # verbatim, so a best pass shards model_best.eqx instead.
    names_file = tmp_path / "names.json"
    names_file.write_text(json.dumps(["h2"]))
    seen = []
    monkeypatch.setattr(
        ehw, "compute_shard",
        lambda rd, idx, names, basis, gl, model_name="model.eqx",
        coldstart=False: (
            seen.append(model_name)
            or {"energies": {}, "pbe_energies": {}, "mol_records": []}))

    base = ["--run-dir", "/run", "--spec-idx", "0", "--names-file", str(names_file),
            "--basis", "def2-svp", "--grid-level", "1"]
    ehw.main(base + ["--out-shard", str(tmp_path / "a.json")])
    ehw.main(base + ["--out-shard", str(tmp_path / "b.json"),
                     "--model-name", "model_best.eqx"])
    assert seen == ["model.eqx", "model_best.eqx"]


def test_compute_shard_loads_named_checkpoint(tmp_path, monkeypatch):
    # compute_shard builds the checkpoint path from model_name, so the best pass
    # genuinely loads model_best.eqx from the spec dir.
    import xcquinox.alec.cluster._eval_one_spec as ev
    import xcquinox.alec.eval_holdout as eh2
    import xcquinox.alec.full_benchmark_pools as fbp
    monkeypatch.setattr(ev, "_read_width", lambda rd: 4)
    monkeypatch.setattr(ev, "_checkpoint_dir", lambda rd, idx, w: "/ck/spec_0000")
    monkeypatch.setattr(ev, "_spec_path", lambda rd, idx, w: "/ck/spec_0000.spec")
    monkeypatch.setattr(ev, "_load_spec", lambda p: object())
    seen = {}
    monkeypatch.setattr(eh2, "load_trained_model",
                        lambda spec, path: seen.setdefault("path", str(path)))
    monkeypatch.setattr(fbp, "load_full_held_out_pools",
                        lambda basis, grid_level: ({}, []))
    monkeypatch.setattr(eh2, "compute_holdout_per_molecule",
                        lambda spec, model, subset: {
                            "energies": {}, "pbe_energies": {}, "mol_records": []})
    ehw.compute_shard("/run", 0, [], "def2-svp", 1, model_name="model_best.eqx")
    assert seen["path"].endswith("/spec_0000/model_best.eqx")


def test_worker_main_reports_failure_json(tmp_path, monkeypatch, capsys):
    names_file = tmp_path / "names.json"
    names_file.write_text(json.dumps(["h2"]))

    def _boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(ehw, "compute_shard", _boom)

    rc = ehw.main(["--run-dir", "/run", "--spec-idx", "0",
                   "--names-file", str(names_file),
                   "--out-shard", str(tmp_path / "s.json"),
                   "--basis", "def2-svp", "--grid-level", "1"])
    assert rc == 1
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["status"] == "failed" and "kaboom" in out["error"]


# ---------------------------------------------------------------------------
# Escalation orchestrator
# ---------------------------------------------------------------------------

class _FakeSpec:
    molecules = ()


def _cmd_arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def _make_fake_run_workers(should_succeed):
    """Return a run_workers stub that simulates the real worker: for each job it
    reads the shard's --names-file and, when ``should_succeed(call_no, names)``,
    writes the --out-shard JSON and reports success; else reports failure with no
    file. ``call_no`` is the 1-based tier invocation count."""
    state = {"calls": 0}

    def _fake(jobs, max_parallel=4, **kw):
        state["calls"] += 1
        results = []
        for job in jobs:
            names = json.loads(open(_cmd_arg(job.cmd, "--names-file")).read())
            out_shard = _cmd_arg(job.cmd, "--out-shard")
            if should_succeed(state["calls"], names):
                with open(out_shard, "w") as f:
                    json.dump({
                        "energies": {n: -1.0 for n in names},
                        "pbe_energies": {n: -0.9 for n in names},
                        "mol_records": [{"molecule": n} for n in names],
                    }, f)
                status = "success"
            else:
                status = "failed"
            results.append(par.WorkerResult(
                job=job, status=status, returncode=0 if status == "success" else 1,
                payload={}, stderr="", duration=0.01))
        return results
    return _fake


def _molecules_in_per_molecule_json(out_dir):
    pm = json.loads((out_dir / eh.DEFAULT_PER_MOLECULE_NAME).read_text())
    return {r["molecule"] for r in pm}


def test_escalation_retries_only_failed_names_at_lower_tier(tmp_path, monkeypatch):
    from xcquinox.alec.cluster import _holdout_parallel as hp
    full_specs = {n: object() for n in ("a", "b", "c", "d")}

    # Tier 1 (call 1): any shard containing "d" fails; others succeed.
    # Tier 2 (call 2+): everything succeeds -> "d" finishes on retry.
    def should_succeed(call_no, names):
        return call_no >= 2 or "d" not in names
    monkeypatch.setattr(par, "run_workers", _make_fake_run_workers(should_succeed))
    # Serial fallback must NOT be reached, make it explode if it is.
    monkeypatch.setattr(eh, "compute_holdout_per_molecule",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("serial fallback should not run")))

    out_dir = tmp_path / "eval_holdout"
    summary = hp.run_holdout_with_escalation(
        "/run", 0, _FakeSpec(), object(), [], full_specs, out_dir,
        basis="def2-svp", grid_level=1, n_workers_top=4, total_cpus=4)

    assert summary["n_species"] == 4
    assert _molecules_in_per_molecule_json(out_dir) == {"a", "b", "c", "d"}


def test_escalation_threads_model_name_to_worker_cmd(tmp_path, monkeypatch):
    # The orchestrator must pass --model-name to the shard workers; the workers
    # re-derive the checkpoint independently, so without this they would silently
    # shard model.eqx into eval_holdout_best/ -> mixed-checkpoint data corruption.
    from xcquinox.alec.cluster import _holdout_parallel as hp
    full_specs = {n: object() for n in ("a", "b")}
    seen = []

    def _capture(jobs, max_parallel=4, **kw):
        results = []
        for job in jobs:
            seen.append(_cmd_arg(job.cmd, "--model-name"))
            names = json.loads(open(_cmd_arg(job.cmd, "--names-file")).read())
            with open(_cmd_arg(job.cmd, "--out-shard"), "w") as f:
                json.dump({"energies": {n: -1.0 for n in names},
                           "pbe_energies": {n: -0.9 for n in names},
                           "mol_records": [{"molecule": n} for n in names]}, f)
            results.append(par.WorkerResult(
                job=job, status="success", returncode=0, payload={}, stderr="",
                duration=0.01))
        return results
    monkeypatch.setattr(par, "run_workers", _capture)

    # default -> model.eqx (final pass, unchanged contract)
    hp.run_holdout_with_escalation(
        "/run", 0, _FakeSpec(), object(), [], full_specs,
        tmp_path / "eval_holdout", basis="def2-svp", grid_level=1,
        n_workers_top=2, total_cpus=2)
    assert seen and all(m == "model.eqx" for m in seen)

    # explicit -> model_best.eqx (best pass)
    seen.clear()
    hp.run_holdout_with_escalation(
        "/run", 0, _FakeSpec(), object(), [], full_specs,
        tmp_path / "eval_holdout_best", basis="def2-svp", grid_level=1,
        n_workers_top=2, total_cpus=2, model_name="model_best.eqx")
    assert seen and all(m == "model_best.eqx" for m in seen)


def test_graceful_total_fallback_to_serial(tmp_path, monkeypatch):
    from xcquinox.alec.cluster import _holdout_parallel as hp
    full_specs = {n: object() for n in ("a", "b", "c", "d")}

    # Every worker tier fails -> all molecules fall through to the serial tier.
    monkeypatch.setattr(par, "run_workers",
                        _make_fake_run_workers(lambda call_no, names: False))
    serial_calls = {"n": 0}

    def _fake_serial(training_spec, model, subset):
        serial_calls["n"] += 1
        serial_calls["subset"] = set(subset)
        return {"energies": {n: -1.0 for n in subset},
                "pbe_energies": {n: -0.9 for n in subset},
                "mol_records": [{"molecule": n} for n in subset]}
    monkeypatch.setattr(eh, "compute_holdout_per_molecule", _fake_serial)

    out_dir = tmp_path / "eval_holdout"
    summary = hp.run_holdout_with_escalation(
        "/run", 0, _FakeSpec(), object(), [], full_specs, out_dir,
        basis="def2-svp", grid_level=1, n_workers_top=4, total_cpus=4)

    assert serial_calls["n"] == 1                      # one serial sweep
    assert serial_calls["subset"] == {"a", "b", "c", "d"}
    assert _molecules_in_per_molecule_json(out_dir) == {"a", "b", "c", "d"}
    assert summary["n_species"] == 4


# ---------------------------------------------------------------------------
# Non-finite species are re-queued (a shard that WROTE a NaN energy is not done)
# ---------------------------------------------------------------------------

def _make_recording_run_workers(energy_for):
    """run_workers stub whose shards always write a JSON payload, with the
    per-species energy taken from ``energy_for(call_no, name)`` (None / NaN
    model the silently-failed species). Records the sorted name list handed to
    each tier so a retry can be observed."""
    state = {"calls": 0, "names": []}

    def _fake(jobs, max_parallel=4, **kw):
        state["calls"] += 1
        tier_names = []
        results = []
        for job in jobs:
            names = json.loads(open(_cmd_arg(job.cmd, "--names-file")).read())
            tier_names.extend(names)
            energies = {n: energy_for(state["calls"], n) for n in names}
            with open(_cmd_arg(job.cmd, "--out-shard"), "w") as f:
                json.dump({
                    "energies": energies,
                    "pbe_energies": {n: -0.9 for n in names},
                    "mol_records": [
                        {"molecule": n,
                         "E_total_nn": (energies[n]
                                        if isinstance(energies[n], float)
                                        and math.isfinite(energies[n])
                                        else None)}
                        for n in names],
                }, f)
            results.append(par.WorkerResult(
                job=job, status="success", returncode=0, payload={}, stderr="",
                duration=0.01))
        state["names"].append(sorted(tier_names))
        return results
    return _fake, state


def test_escalation_requeues_nonfinite_species_to_next_tier(tmp_path, monkeypatch):
    """A tier-1 shard that completes but writes a null energy for one species
    leaves that species UNFINISHED: it must reach the lower-parallelism tier,
    where the finite retry wins."""
    from xcquinox.alec.cluster import _holdout_parallel as hp
    full_specs = {n: object() for n in ("a", "b", "c", "d")}

    def energy_for(call_no, name):
        return None if (name == "d" and call_no == 1) else -1.0
    fake, state = _make_recording_run_workers(energy_for)
    monkeypatch.setattr(par, "run_workers", fake)
    monkeypatch.setattr(eh, "compute_holdout_per_molecule",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("serial fallback should not run")))

    out_dir = tmp_path / "eval_holdout"
    hp.run_holdout_with_escalation(
        "/run", 0, _FakeSpec(), object(), [], full_specs, out_dir,
        basis="def2-svp", grid_level=1, n_workers_top=4, total_cpus=4)

    assert state["names"][0] == ["a", "b", "c", "d"]
    assert len(state["names"]) == 2
    assert state["names"][1] == ["d"]         # only the null species retried
    pm = json.loads((out_dir / eh.DEFAULT_PER_MOLECULE_NAME).read_text())
    by = {r["molecule"]: r for r in pm}
    assert len(pm) == 4                       # one row per species, not two
    assert by["d"]["E_total_nn"] == pytest.approx(-1.0)


def test_escalation_accepts_and_names_species_failing_every_tier(
        tmp_path, monkeypatch, capsys):
    """A species that is non-finite in every tier AND in the serial sweep is
    accepted (its last payload lands in per_molecule.json) and named once."""
    from xcquinox.alec.cluster import _holdout_parallel as hp
    full_specs = {n: object() for n in ("a", "b", "d")}

    fake, state = _make_recording_run_workers(
        lambda call_no, name: float("nan") if name == "d" else -1.0)
    monkeypatch.setattr(par, "run_workers", fake)

    def _serial(training_spec, model, subset):
        return {"energies": {n: float("nan") for n in subset},
                "pbe_energies": {n: -0.9 for n in subset},
                "mol_records": [{"molecule": n, "E_total_nn": None}
                                for n in subset]}
    monkeypatch.setattr(eh, "compute_holdout_per_molecule", _serial)

    out_dir = tmp_path / "eval_holdout"
    summary = hp.run_holdout_with_escalation(
        "/run", 0, _FakeSpec(), object(), [], full_specs, out_dir,
        basis="def2-svp", grid_level=1, n_workers_top=4, total_cpus=4)

    assert len(state["names"]) == 2 and state["names"][1] == ["d"]
    named = [ln for ln in capsys.readouterr().out.splitlines()
             if "failed in every tier" in ln]
    assert len(named) == 1
    assert "1 species failed in every tier: d" in named[0]
    assert summary["n_species"] == 3
    assert _molecules_in_per_molecule_json(out_dir) == {"a", "b", "d"}


# ---------------------------------------------------------------------------
# _eval_one_spec wiring (parallel-by-default + graceful fallback)
# ---------------------------------------------------------------------------

def _fake_cfg():
    from types import SimpleNamespace
    return SimpleNamespace(
        inputs=SimpleNamespace(basis="def2-svp", grid_level=1),
        cluster=object())


_OK_SUMMARY = {"n_reactions": 0, "n_species": 2,
               "n_dropped_nan": 0, "n_dropped_overlap": 0}


def _wire_common(monkeypatch, n_top):
    import xcquinox.alec.full_benchmark_pools as fbp
    import xcquinox.alec.cluster.grid_config as gc
    monkeypatch.setattr(eh, "load_trained_model", lambda ts, mp: "MODEL")
    monkeypatch.setattr(fbp, "load_full_held_out_pools",
                        lambda *, basis, grid_level: ({"a": 1, "b": 2}, []))
    monkeypatch.setattr(gc, "_resolve_eval_workers",
                        lambda cl, *, n_molecules: n_top)


def test_eval_one_spec_uses_parallel_when_workers_gt_1(tmp_path, monkeypatch):
    import xcquinox.alec.cluster._eval_one_spec as eos
    import xcquinox.alec.cluster._holdout_parallel as hp
    _wire_common(monkeypatch, n_top=2)
    called = {}

    def _orch(*a, **k):
        called["p"] = True
        return _OK_SUMMARY
    monkeypatch.setattr(hp, "run_holdout_with_escalation", _orch)
    monkeypatch.setattr(eh, "run_full_holdout_eval",
                        lambda **k: pytest.fail("serial must not run"))

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    eos._run_held_out_eval("/run", 0, _fake_cfg(), str(ckpt),
                           str(ckpt / "model.eqx"), _FakeSpec())
    assert called.get("p") is True
    assert not (ckpt / "eval_holdout" / "failure.json").exists()


def test_eval_one_spec_falls_back_to_serial_on_parallel_raise(tmp_path, monkeypatch):
    import xcquinox.alec.cluster._eval_one_spec as eos
    import xcquinox.alec.cluster._holdout_parallel as hp
    _wire_common(monkeypatch, n_top=2)

    def _raise(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(hp, "run_holdout_with_escalation", _raise)
    serial = {}

    def _serial(**k):
        serial["ran"] = True
        return _OK_SUMMARY
    monkeypatch.setattr(eh, "run_full_holdout_eval", _serial)

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    eos._run_held_out_eval("/run", 0, _fake_cfg(), str(ckpt),
                           str(ckpt / "model.eqx"), _FakeSpec())
    assert serial.get("ran") is True                   # fell back, no crash
    assert not (ckpt / "eval_holdout" / "failure.json").exists()


def test_eval_one_spec_serial_when_workers_le_1(tmp_path, monkeypatch):
    import xcquinox.alec.cluster._eval_one_spec as eos
    import xcquinox.alec.cluster._holdout_parallel as hp
    _wire_common(monkeypatch, n_top=1)
    monkeypatch.setattr(hp, "run_holdout_with_escalation",
                        lambda *a, **k: pytest.fail("parallel must not run"))
    serial = {}

    def _serial(**k):
        serial["ran"] = True
        return _OK_SUMMARY
    monkeypatch.setattr(eh, "run_full_holdout_eval", _serial)

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    eos._run_held_out_eval("/run", 0, _fake_cfg(), str(ckpt),
                           str(ckpt / "model.eqx"), _FakeSpec())
    assert serial.get("ran") is True


# ---------------------------------------------------------------------------
# merge_holdout_shards
# ---------------------------------------------------------------------------

def test_merge_holdout_shards_unions_maps_and_concats_records():
    shards = [
        {"energies": {"h2": -1.17}, "pbe_energies": {"h2": -1.16},
         "mol_records": [{"molecule": "h2", "E_nn": -1.17}]},
        {"energies": {"o": -75.0}, "pbe_energies": {"o": -74.9},
         "mol_records": [{"molecule": "o", "E_nn": -75.0}]},
        {"energies": {"h": -0.5}, "pbe_energies": {"h": -0.5},
         "mol_records": [{"molecule": "h", "E_nn": -0.5}]},
    ]
    energies, pbe, records = eh.merge_holdout_shards(shards)
    assert energies == {"h2": -1.17, "o": -75.0, "h": -0.5}
    assert pbe == {"h2": -1.16, "o": -74.9, "h": -0.5}
    # records re-sorted by molecule name (matches the serial ordering)
    assert [r["molecule"] for r in records] == ["h", "h2", "o"]


def test_merge_holdout_shards_tolerates_missing_keys():
    energies, pbe, records = eh.merge_holdout_shards([{}, {"energies": {"a": 1.0}}])
    assert energies == {"a": 1.0}
    assert pbe == {} and records == []


# ---------------------------------------------------------------------------
# merge_holdout_shards precedence (a re-queued species appears in >1 payload)
# ---------------------------------------------------------------------------

def _one_name_shard(name, e_nn, e_pbe, tag):
    """One shard payload for a single species, shaped like the worker's."""
    finite = isinstance(e_nn, float) and math.isfinite(e_nn)
    return {
        "energies": {name: e_nn},
        "pbe_energies": {name: e_pbe},
        "mol_records": [{"molecule": name,
                         "E_total_nn": e_nn if finite else None,
                         "tag": tag}],
    }


def test_merge_precedence_nan_then_finite_takes_finite():
    energies, pbe, recs = eh.merge_holdout_shards([
        _one_name_shard("d", float("nan"), None, "t1"),
        _one_name_shard("d", -1.0, -0.95, "t2"),
    ])
    assert energies["d"] == pytest.approx(-1.0)
    assert pbe["d"] == pytest.approx(-0.95)
    assert [r["tag"] for r in recs] == ["t2"]


def test_merge_precedence_finite_then_nan_keeps_finite():
    energies, pbe, recs = eh.merge_holdout_shards([
        _one_name_shard("d", -1.0, -0.95, "t1"),
        _one_name_shard("d", float("nan"), None, "t2"),
    ])
    assert energies["d"] == pytest.approx(-1.0)
    assert pbe["d"] == pytest.approx(-0.95)
    assert [r["tag"] for r in recs] == ["t1"]


def test_merge_precedence_both_finite_later_tier_wins():
    energies, pbe, recs = eh.merge_holdout_shards([
        _one_name_shard("d", -1.0, -0.90, "t1"),
        _one_name_shard("d", -1.5, -0.95, "t2"),
    ])
    assert energies["d"] == pytest.approx(-1.5)
    assert pbe["d"] == pytest.approx(-0.95)
    assert [r["tag"] for r in recs] == ["t2"]


def test_merge_precedence_both_nonfinite_keeps_last_payload():
    energies, pbe, recs = eh.merge_holdout_shards([
        _one_name_shard("d", None, None, "t1"),
        _one_name_shard("d", float("nan"), None, "t2"),
    ])
    assert not (isinstance(energies["d"], float)
                and math.isfinite(energies["d"]))
    assert [r["tag"] for r in recs] == ["t2"]        # last payload accepted


def test_merge_dedupes_records_by_molecule_name():
    """A re-queued species has one record per payload; per_molecule.json must
    still carry exactly one row for it."""
    _e, _p, recs = eh.merge_holdout_shards([
        _one_name_shard("d", float("nan"), None, "t1"),
        _one_name_shard("d", -1.0, -0.95, "t2"),
        _one_name_shard("a", -2.0, -1.95, "t1"),
    ])
    assert [r["molecule"] for r in recs] == ["a", "d"]


# ---------------------------------------------------------------------------
# _finalize_holdout_outputs
# ---------------------------------------------------------------------------

def test_finalize_writes_artifacts_and_summary(tmp_path):
    # One reaction h2 -> 2 H; ref atomization 109.493 kcal/mol.
    reactions = [{
        "name": "w411_h2", "source_pool": "w411",
        "reactants": ["h2"], "products": ["h"], "coeffs": [-1.0, 2.0],
        "reaction_energy_ref": 109.493,
    }]
    energies = {"h2": -1.17, "h": -0.50}
    pbe_energies = {"h2": -1.16, "h": -0.50}
    mol_records = [{"molecule": "h", "E_nn": -0.50},
                   {"molecule": "h2", "E_nn": -1.17}]
    out_dir = tmp_path / "eval_holdout"

    summary = eh._finalize_holdout_outputs(
        reactions, energies, pbe_energies, mol_records,
        training_names=(), n_species=2, out_dir=out_dir, strict=False)

    assert summary["n_reactions"] == 1
    assert summary["n_species"] == 2
    assert (out_dir / eh.DEFAULT_CSV_NAME).is_file()
    pm = json.loads((out_dir / eh.DEFAULT_PER_MOLECULE_NAME).read_text())
    assert {r["molecule"] for r in pm} == {"h", "h2"}
    assert (out_dir / eh.DEFAULT_PER_REACTION_NAME).is_file()


def test_finalize_via_merge_equals_single_shard(tmp_path):
    """Finalizing the merge of two shards equals finalizing the same data as one
    block: the property the parallel path relies on."""
    reactions = [{
        "name": "w411_h2", "source_pool": "w411",
        "reactants": ["h2"], "products": ["h"], "coeffs": [-1.0, 2.0],
        "reaction_energy_ref": 109.493,
    }]
    shard_a = {"energies": {"h2": -1.17}, "pbe_energies": {"h2": -1.16},
               "mol_records": [{"molecule": "h2", "E_nn": -1.17}]}
    shard_b = {"energies": {"h": -0.50}, "pbe_energies": {"h": -0.50},
               "mol_records": [{"molecule": "h", "E_nn": -0.50}]}

    e, p, recs = eh.merge_holdout_shards([shard_a, shard_b])
    merged = eh._finalize_holdout_outputs(
        reactions, e, p, recs, training_names=(), n_species=2,
        out_dir=tmp_path / "merged", strict=False)

    e1 = {"h2": -1.17, "h": -0.50}
    p1 = {"h2": -1.16, "h": -0.50}
    recs1 = [{"molecule": "h", "E_nn": -0.50}, {"molecule": "h2", "E_nn": -1.17}]
    single = eh._finalize_holdout_outputs(
        reactions, e1, p1, recs1, training_names=(), n_species=2,
        out_dir=tmp_path / "single", strict=False)

    assert merged["combined"] == single["combined"]
    assert merged["per_pool_mae"] == single["per_pool_mae"]
    assert merged["n_reactions"] == single["n_reactions"]


# ---------------------------------------------------------------------------
# Real-subprocess integration (run_workers + argv + file merge), SCF-free
# ---------------------------------------------------------------------------

_FAKE_WORKER = '''\
import argparse, json
p = argparse.ArgumentParser()
for f in ("--run-dir", "--spec-idx", "--names-file", "--out-shard",
          "--basis", "--grid-level", "--threads", "--model-name"):
    p.add_argument(f)
a = p.parse_args()
names = json.load(open(a.names_file))
json.dump({"energies": {n: -1.0 for n in names},
           "pbe_energies": {n: -0.9 for n in names},
           "mol_records": [{"molecule": n} for n in names]}, open(a.out_shard, "w"))
print(json.dumps({"status": "success", "n_done": len(names)}))
'''


def test_orchestrator_drives_real_worker_subprocesses(tmp_path, monkeypatch):
    """End-to-end through the REAL parallel.run_workers subprocess machinery
    (argv contract, thread_env, stdout status, shard-file round-trip, merge),
    using a trivial SCF-free worker script. The per-molecule physics
    (compute_holdout_per_molecule) is covered separately; this guards the
    plumbing that unit mocks cannot."""
    from xcquinox.alec.cluster import _holdout_parallel as hp

    worker = tmp_path / "fake_worker.py"
    worker.write_text(_FAKE_WORKER)
    monkeypatch.setattr(par, "worker_script_path", lambda name: str(worker))

    full_specs = {n: object() for n in ("a", "b", "c", "d")}
    out_dir = tmp_path / "eval_holdout"
    summary = hp.run_holdout_with_escalation(
        "/run", 0, _FakeSpec(), object(), [], full_specs, out_dir,
        basis="def2-svp", grid_level=1, n_workers_top=4, total_cpus=4)

    assert summary["n_species"] == 4
    assert _molecules_in_per_molecule_json(out_dir) == {"a", "b", "c", "d"}


_NOISY_FAKE_WORKER = '''\
import argparse, json, sys
p = argparse.ArgumentParser()
for f in ("--run-dir", "--spec-idx", "--names-file", "--out-shard",
          "--basis", "--grid-level", "--threads", "--model-name"):
    p.add_argument(f)
a = p.parse_args()
names = json.load(open(a.names_file))
print("[worker] precompute done")
print("  eval[%s] FAILED: RuntimeError: stdout side" % names[0])
print("  eval[%s] FAILED: RuntimeError: stderr side" % names[0],
      file=sys.stderr)
json.dump({"energies": {n: -1.0 for n in names},
           "pbe_energies": {n: -0.9 for n in names},
           "mol_records": [{"molecule": n, "E_total_nn": -1.0} for n in names]},
          open(a.out_shard, "w"))
print(json.dumps({"status": "success", "n_done": len(names)}))
'''


def test_orchestrator_persists_worker_logs_and_forwards_failed_lines(
        tmp_path, monkeypatch, capsys):
    """Worker diagnostics survive the subprocess boundary: both streams land in
    a per-shard log next to the shard JSON, and the per-species FAILED lines
    are echoed into the task log so a silent shard cannot go unnoticed."""
    from xcquinox.alec.cluster import _holdout_parallel as hp

    worker = tmp_path / "noisy_worker.py"
    worker.write_text(_NOISY_FAKE_WORKER)
    monkeypatch.setattr(par, "worker_script_path", lambda name: str(worker))

    out_dir = tmp_path / "eval_holdout"
    summary = hp.run_holdout_with_escalation(
        "/run", 0, _FakeSpec(), object(), [], {"a": object()}, out_dir,
        basis="def2-svp", grid_level=1, n_workers_top=2, total_cpus=2)

    assert summary["n_species"] == 1          # the JSON result is still parsed
    log = out_dir / "_shards" / "worker_t1_s0.log"
    assert log.is_file()
    text = log.read_text()
    assert "[worker] precompute done" in text                 # stdout
    assert "FAILED: RuntimeError: stderr side" in text        # stderr
    forwarded = [ln for ln in capsys.readouterr().err.splitlines()
                 if "[holdout-parallel] worker t1/s0:" in ln]
    assert len(forwarded) == 2                # both streams are scanned
    assert any("stdout side" in ln for ln in forwarded)
    assert any("stderr side" in ln for ln in forwarded)


def test_worker_main_forwards_coldstart_flag(tmp_path, monkeypatch):
    """--coldstart reaches compute_shard so the worker applies the shared
    override to its OWN spec reload (the orchestrator's in-memory replace
    cannot reach shard subprocesses)."""
    names_file = tmp_path / "names.json"
    names_file.write_text(json.dumps(["h2"]))
    seen = []
    monkeypatch.setattr(
        ehw, "compute_shard",
        lambda rd, idx, names, basis, gl, model_name="model.eqx",
        coldstart=False: (
            seen.append(coldstart)
            or {"energies": {}, "pbe_energies": {}, "mol_records": []}))
    base = ["--run-dir", "/run", "--spec-idx", "0",
            "--names-file", str(names_file),
            "--out-shard", str(tmp_path / "s.json"),
            "--basis", "def2-svp", "--grid-level", "1", "--threads", "1"]
    assert ehw.main(base) == 0
    assert ehw.main(base + ["--coldstart"]) == 0
    assert seen == [False, True]


def test_compute_shard_coldstart_applies_shared_override(monkeypatch):
    """Under --coldstart the shard evaluates with the SAME override the
    orchestrator applies (single source of truth): minao seed, 25 cycles,
    conv_tol 1e-12, FULL mode."""
    import dataclasses as _dc
    from types import SimpleNamespace

    from xcquinox.alec.solver import (SolverBackend, SolverConfig,
                                      SolverMode)

    @_dc.dataclass(frozen=True)
    class _Spec:
        solver_config: object

    spec = _Spec(solver_config=SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL, max_cycles=3))
    # compute_shard resolves its collaborators via in-function imports, so
    # the seams are the SOURCE modules, not worker module attributes.
    import xcquinox.alec.cluster._eval_one_spec as ev_mod
    import xcquinox.alec.eval_holdout as eh
    import xcquinox.alec.full_benchmark_pools as fbp
    monkeypatch.setattr(ev_mod, "_load_spec", lambda path: spec)
    monkeypatch.setattr(ev_mod, "_read_width", lambda rd: 4)
    monkeypatch.setattr(eh, "load_trained_model",
                        lambda spec, path: "MODEL")
    captured = {}

    def _fake_compute(training_spec, model, subset, **kw):
        captured["sc"] = training_spec.solver_config
        return {"energies": {}, "pbe_energies": {}, "mol_records": []}

    monkeypatch.setattr(eh, "compute_holdout_per_molecule", _fake_compute)
    monkeypatch.setattr(
        fbp, "load_full_held_out_pools",
        lambda basis=None, grid_level=None:
            ({"h2": SimpleNamespace(name="h2")}, []))
    ehw.compute_shard("/run", 0, ["h2"], "def2-svp", 1, coldstart=True)
    sc = captured["sc"]
    assert sc.seed_source == "minao"
    assert sc.max_cycles == 25
    assert sc.conv_tol == 1e-12
