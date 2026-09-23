"""Tests for xcquinox.pipeline.cluster.job_tracking.

These tests NEVER shell out to a real SLURM controller: ``_run_slurm`` (and,
for the retry-policy tests, ``subprocess.run``) is monkeypatched with canned
behavior. A synthetic ``run_dir`` (manifest.json + checkpoints/spec_* dirs +
jobs.json) is built per-test in a tmp directory.
"""
import json
import os
import subprocess

import pytest

from xcquinox.pipeline.cluster import job_tracking as jt
from xcquinox.pipeline.cluster.job_tracking import (
    SlurmTransientError,
    _run_slurm,
    append_job_record,
    mark_superseded,
    read_job_records,
    reduce_outcomes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_manifest(run_dir, n_specs, width=4):
    """Write a minimal manifest.json the way materialize.write_manifest would."""
    payload = {
        "xcquinox_version": "test",
        "python_version": "3.x",
        "width": width,
        "n_specs": n_specs,
        "specs": [{"index": i, "spec_file": f"spec_{i:0{width}d}.spec"}
                  for i in range(n_specs)],
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(payload, f)


def _spec_dir(run_dir, idx, width=4):
    d = os.path.join(run_dir, "checkpoints", f"spec_{idx:0{width}d}")
    os.makedirs(d, exist_ok=True)
    return d


def _write_model(run_dir, idx, width=4):
    open(os.path.join(_spec_dir(run_dir, idx, width), "model.eqx"), "wb").close()


def _write_resume_state(run_dir, idx, width=4):
    """Write a WS5 mid-run ``resume_state.pkl`` marker (contents irrelevant; the
    harness keys only on its PRESENCE)."""
    open(os.path.join(_spec_dir(run_dir, idx, width), "resume_state.pkl"),
         "wb").close()


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    return str(d)


# ---------------------------------------------------------------------------
# append_job_record / read_job_records
# ---------------------------------------------------------------------------

def test_append_job_record_per_kind_monotonic_generation(run_dir):
    r0 = append_job_record(run_dir, "train", "1001", [0, 1, 2])
    r1 = append_job_record(run_dir, "train", "1002", [0, 1, 2])
    # eval generation counter is independent of train.
    e0 = append_job_record(run_dir, "eval", "2001", [0, 1])
    r2 = append_job_record(run_dir, "train", "1003", [0])

    assert r0["generation"] == 0
    assert r1["generation"] == 1
    assert r2["generation"] == 2
    assert e0["generation"] == 0
    assert r0["superseded"] is False
    assert "submitted_utc" in r0


def test_jobs_json_is_append_only(run_dir):
    append_job_record(run_dir, "train", "1001", [0])
    append_job_record(run_dir, "eval", "2001", [0])
    append_job_record(run_dir, "train", "1002", [1])

    records = read_job_records(run_dir)
    # All three records are present and in append order.
    assert [r["array_job_id"] for r in records] == ["1001", "2001", "1002"]


# ---------------------------------------------------------------------------
# mark_superseded
# ---------------------------------------------------------------------------

def test_mark_superseded_flips_flag_and_rewrites_atomically(run_dir):
    append_job_record(run_dir, "train", "1001", [0])
    append_job_record(run_dir, "train", "1002", [0])
    append_job_record(run_dir, "eval", "2001", [0])

    mark_superseded(run_dir, "train", 0)

    records = read_job_records(run_dir)
    by_id = {r["array_job_id"]: r for r in records}
    assert by_id["1001"]["superseded"] is True
    # Only the targeted (kind, generation) is touched.
    assert by_id["1002"]["superseded"] is False
    assert by_id["2001"]["superseded"] is False
    # No orphan temp file left behind by the atomic rewrite.
    assert not any(n.startswith(".mktmp_") for n in os.listdir(run_dir))


# ---------------------------------------------------------------------------
# reduce_outcomes: disk-first
# ---------------------------------------------------------------------------

def test_reduce_outcomes_disk_first_success_skips_sacct(run_dir, monkeypatch):
    _write_manifest(run_dir, n_specs=3)
    _write_model(run_dir, 0)
    _write_model(run_dir, 1)
    _write_model(run_dir, 2)

    calls = []
    monkeypatch.setattr(jt, "_run_slurm",
                        lambda *a, **k: calls.append(a) or _fake_proc(""))

    out = reduce_outcomes(run_dir, "train")
    assert out == {0: "success", 1: "success", 2: "success"}
    # All indices resolved by disk evidence -> sacct never consulted.
    assert calls == []


# ---------------------------------------------------------------------------
# _disk_outcome: WS6 incomplete_resumable detection (resume_state.pkl)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# reduce_outcomes: manifest-driven (never glob)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# reduce_outcomes: sacct fallback
# ---------------------------------------------------------------------------

def _fake_proc(stdout, returncode=0):
    return subprocess.CompletedProcess(args=["sacct"], returncode=returncode,
                                       stdout=stdout, stderr="")


def test_reduce_outcomes_sacct_state_mapping(run_dir, monkeypatch):
    _write_manifest(run_dir, n_specs=4)
    append_job_record(run_dir, "train", "5000", [0, 1, 2, 3])

    sacct_out = "\n".join([
        "5000|PENDING|0:0",          # array container row, skipped
        "5000_0|OUT_OF_MEMORY|0:125",
        "5000_1|TIMEOUT|0:0",
        "5000_2|CANCELLED|0:137",    # OOM-ish exit signal
        "5000_3.batch|FAILED|1:0",   # step row: skipped
        "5000_3|FAILED|1:0",
    ])
    monkeypatch.setattr(jt, "_run_slurm", lambda *a, **k: _fake_proc(sacct_out))

    out = reduce_outcomes(run_dir, "train")
    assert out == {0: "oom", 1: "timeout", 2: "oom",
                   3: "dependency_never_satisfied"}


# ---------------------------------------------------------------------------
# reduce_outcomes: superseded generations ignored / newest wins
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# reduce_outcomes: SlurmTransientError short-circuits
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _run_slurm: retry policy
# ---------------------------------------------------------------------------

def test_run_slurm_query_verb_retries_then_raises(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=1,
                                           stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(jt.time, "sleep", lambda s: None)  # no real backoff

    with pytest.raises(SlurmTransientError):
        _run_slurm(["sacct", "--jobs=1"], retries=3)
    # A query verb retried the full 3 attempts before giving up.
    assert len(calls) == 3


