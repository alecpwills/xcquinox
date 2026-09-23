"""Parallel held-out eval: refactor seams (merge + finalize), the degradation
ladder, escalation/retry, graceful fallback, runtime CPU detection, and the
eval_workers config knob.

The per-molecule eval is parallelized across molecule shards (subprocess
workers via xcquinox.pipeline.parallel.run_workers), with an adaptive ladder that
retries failed molecules at lower parallelism and ends in in-process serial.
These tests use synthetic data + monkeypatched run_workers (no real SCF /
subprocess) except the explicitly-slow end-to-end smoke.
"""
import json
import math

import pytest

import xcquinox.pipeline.eval_holdout as eh
from xcquinox.pipeline import parallel as par
from xcquinox.pipeline.workers import eval_holdout_worker as ehw


# ---------------------------------------------------------------------------
# Queue-agnostic CPU detection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Degradation ladder
# ---------------------------------------------------------------------------

def test_eval_worker_ladder_24():
    assert par.eval_worker_ladder(24) == [(24, 1), (12, 2), (6, 4)]


# ---------------------------------------------------------------------------
# Config knob + resolution
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shard worker
# ---------------------------------------------------------------------------

def test_compute_shard_evaluates_only_named_subset(monkeypatch):
    import xcquinox.pipeline.cluster._eval_one_spec as eos
    import xcquinox.pipeline.full_benchmark_pools as fbp
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
        channel=None, pools=("bh76", "w411"): {
            "energies": {"h2": -1.17}, "pbe_energies": {"h2": -1.16},
            "mol_records": [{"molecule": "h2"}]})

    rc = ehw.main(["--run-dir", "/run", "--spec-idx", "2",
                   "--names-file", str(names_file), "--out-shard", str(out_shard),
                   "--basis", "def2-svp", "--grid-level", "1", "--threads", "1"])
    assert rc == 0
    assert json.loads(out_shard.read_text())["energies"] == {"h2": -1.17}
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(last)["status"] == "success"


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
    from xcquinox.pipeline.cluster import _holdout_parallel as hp
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


# ---------------------------------------------------------------------------
# Non-finite species are re-queued (a shard that WROTE a NaN energy is not done)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _eval_one_spec wiring (parallel-by-default + graceful fallback)
# ---------------------------------------------------------------------------


_OK_SUMMARY = {"n_reactions": 0, "n_species": 2,
               "n_dropped_nan": 0, "n_dropped_overlap": 0}


# ---------------------------------------------------------------------------
# merge_holdout_shards
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# A pool with no precomputed species is refused at the whole-pool boundary
# ---------------------------------------------------------------------------


