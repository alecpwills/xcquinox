"""Tests for xcquinox.alec.cluster.job_tracking.

These tests NEVER shell out to a real SLURM controller: ``_run_slurm`` (and,
for the retry-policy tests, ``subprocess.run``) is monkeypatched with canned
behavior. A synthetic ``run_dir`` (manifest.json + checkpoints/spec_* dirs +
jobs.json) is built per-test in a tmp directory.
"""
import json
import os
import subprocess

import pytest

from xcquinox.alec.cluster import job_tracking as jt
from xcquinox.alec.cluster.job_tracking import (
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


def _write_failure(run_dir, idx, classification, width=4):
    path = os.path.join(_spec_dir(run_dir, idx, width), "failure.json")
    with open(path, "w") as f:
        json.dump({"classification": classification, "rc": 1}, f)


def _write_resume_state(run_dir, idx, width=4):
    """Write a WS5 mid-run ``resume_state.pkl`` marker (contents irrelevant; the
    harness keys only on its PRESENCE)."""
    open(os.path.join(_spec_dir(run_dir, idx, width), "resume_state.pkl"),
         "wb").close()


def _write_completion(run_dir, idx, width=4, *, early_stopped=False,
                      epochs_run=1):
    """Write the WS5 ``completion.json`` sentinel a clean/early-stopped finish
    leaves alongside ``model.eqx``."""
    path = os.path.join(_spec_dir(run_dir, idx, width), "completion.json")
    with open(path, "w") as f:
        json.dump({"completed": True, "early_stopped": early_stopped,
                   "epochs_run": epochs_run}, f)


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


def test_append_job_record_accepts_pretrain_kind(run_dir):
    """The pretrain stage is a first-class record kind."""
    rec = append_job_record(run_dir, "pretrain", "9100", [0, 1])
    assert rec["kind"] == "pretrain"
    assert rec["generation"] == 0
    assert rec["array_job_id"] == "9100"
    assert rec["indices"] == [0, 1]
    # It round-trips through read_job_records.
    records = read_job_records(run_dir)
    assert records[0]["kind"] == "pretrain"
    # pretrain generation counter is independent of train/eval.
    rec2 = append_job_record(run_dir, "pretrain", "9101", [0, 1])
    assert rec2["generation"] == 1


def test_append_job_record_rejects_unknown_kind(run_dir):
    with pytest.raises(ValueError, match="kind"):
        append_job_record(run_dir, "not-a-stage", "1", [0])


def test_jobs_json_is_append_only(run_dir):
    append_job_record(run_dir, "train", "1001", [0])
    append_job_record(run_dir, "eval", "2001", [0])
    append_job_record(run_dir, "train", "1002", [1])

    records = read_job_records(run_dir)
    # All three records are present and in append order.
    assert [r["array_job_id"] for r in records] == ["1001", "2001", "1002"]


def test_append_job_record_rejects_empty_array_job_id(run_dir):
    with pytest.raises(ValueError, match="array_job_id"):
        append_job_record(run_dir, "train", "", [0])
    with pytest.raises(ValueError, match="array_job_id"):
        append_job_record(run_dir, "train", None, [0])
    # Nothing was written.
    assert read_job_records(run_dir) == []


def test_read_job_records_empty_when_absent(run_dir):
    assert read_job_records(run_dir) == []


def test_read_job_records_rejects_invalid_array_job_id(run_dir):
    with open(os.path.join(run_dir, "jobs.json"), "w") as f:
        json.dump([{"kind": "train", "generation": 0, "array_job_id": "",
                    "indices": [0], "submitted_utc": "x", "superseded": False}], f)
    with pytest.raises(ValueError, match="array_job_id"):
        read_job_records(run_dir)


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


def test_reduce_outcomes_disk_first_failure_classification(run_dir, monkeypatch):
    _write_manifest(run_dir, n_specs=2)
    _write_failure(run_dir, 0, "oom")
    _write_failure(run_dir, 1, "deterministic")

    monkeypatch.setattr(jt, "_run_slurm",
                        lambda *a, **k: pytest.fail("sacct must not run"))

    out = reduce_outcomes(run_dir, "train")
    assert out == {0: "oom", 1: "deterministic"}


def test_reduce_outcomes_model_eqx_beats_stale_failure_json(run_dir, monkeypatch):
    _write_manifest(run_dir, n_specs=1)
    # A stale failure.json from an earlier attempt + a model from a later one.
    _write_failure(run_dir, 0, "oom")
    _write_model(run_dir, 0)

    monkeypatch.setattr(jt, "_run_slurm",
                        lambda *a, **k: pytest.fail("sacct must not run"))

    out = reduce_outcomes(run_dir, "train")
    # model.eqx is checked BEFORE failure.json: success wins.
    assert out == {0: "success"}


# ---------------------------------------------------------------------------
# _disk_outcome: WS6 incomplete_resumable detection (resume_state.pkl)
# ---------------------------------------------------------------------------

def test_disk_outcome_resume_state_only_is_incomplete_resumable(run_dir):
    """A killed mid-run dir (resume_state.pkl present, NO model.eqx, NO
    completion.json) is classified incomplete_resumable so resubmit RESUMES it
    rather than fresh-retrying from scratch (WS6)."""
    sd = _spec_dir(run_dir, 0)
    _write_resume_state(run_dir, 0)
    assert jt._disk_outcome(sd) == "incomplete_resumable"


def test_disk_outcome_resume_state_plus_killed_failure_resume_wins(run_dir):
    """A grace-SIGTERM kill writes BOTH a killed_by_signal failure.json AND the
    resume_* set. The resumable checkpoint MUST win (return incomplete_resumable,
    NOT the failure classification) so resubmit continues from the checkpoint
    instead of archiving + retrying fresh."""
    sd = _spec_dir(run_dir, 0)
    _write_resume_state(run_dir, 0)
    _write_failure(run_dir, 0, "killed_by_signal")
    assert jt._disk_outcome(sd) == "incomplete_resumable"


def test_disk_outcome_model_eqx_beats_resume_state(run_dir):
    """model.eqx wins first: an orphan resume_* set left next to a produced
    model (a completion that did not finish its resume_* cleanup) is still a
    SUCCESS, never incomplete_resumable."""
    sd = _spec_dir(run_dir, 0)
    _write_model(run_dir, 0)
    _write_resume_state(run_dir, 0)
    assert jt._disk_outcome(sd) == "success"


def test_disk_outcome_completion_with_resume_state_is_success(run_dir):
    """An early-stop finish writes model.eqx + completion.json; if its resume_*
    cleanup was interrupted leaving resume_state.pkl, model.eqx still wins ->
    success (completion.json present means NOT resumable regardless)."""
    sd = _spec_dir(run_dir, 0)
    _write_model(run_dir, 0)
    _write_completion(run_dir, 0, early_stopped=True)
    _write_resume_state(run_dir, 0)
    assert jt._disk_outcome(sd) == "success"


def test_disk_outcome_completion_no_model_no_resume_is_failure_unclassified(
        run_dir):
    """completion.json + resume_state.pkl but NO model.eqx is NOT resumable
    (completion.json present blocks resume per train._has_resume_checkpoint);
    with no failure.json it falls through to None (no evidence)."""
    sd = _spec_dir(run_dir, 0)
    _write_completion(run_dir, 0)
    _write_resume_state(run_dir, 0)
    # No model.eqx, completion.json present -> resume blocked; no failure.json
    # -> no disk evidence at all.
    assert jt._disk_outcome(sd) is None


def test_disk_outcome_failure_without_resume_state_is_its_class(run_dir):
    """A plain failure.json with NO resume_state.pkl keeps its own
    classification (the resume branch must not steal ordinary failures)."""
    sd = _spec_dir(run_dir, 0)
    _write_failure(run_dir, 0, "oom")
    assert jt._disk_outcome(sd) == "oom"


def test_reduce_outcomes_propagates_incomplete_resumable(run_dir, monkeypatch):
    """reduce_outcomes is disk-first, so it surfaces incomplete_resumable for a
    killed mid-run index WITHOUT consulting sacct."""
    _write_manifest(run_dir, n_specs=2)
    _write_model(run_dir, 0)
    _write_resume_state(run_dir, 1)

    monkeypatch.setattr(jt, "_run_slurm",
                        lambda *a, **k: pytest.fail("sacct must not run"))

    out = reduce_outcomes(run_dir, "train")
    assert out == {0: "success", 1: "incomplete_resumable"}


# ---------------------------------------------------------------------------
# reduce_outcomes: manifest-driven (never glob)
# ---------------------------------------------------------------------------

def test_reduce_outcomes_is_manifest_driven_ignores_stale_dir(run_dir, monkeypatch):
    _write_manifest(run_dir, n_specs=3)
    _write_model(run_dir, 0)
    _write_model(run_dir, 1)
    _write_model(run_dir, 2)
    # A leftover spec dir from a larger prior grid (index N+5).
    _write_model(run_dir, 8)

    monkeypatch.setattr(jt, "_run_slurm",
                        lambda *a, **k: pytest.fail("sacct must not run"))

    out = reduce_outcomes(run_dir, "train")
    # Only indices 0..2 (from the manifest) are counted; spec_0008 ignored.
    assert set(out.keys()) == {0, 1, 2}
    assert 8 not in out


# ---------------------------------------------------------------------------
# reduce_outcomes: sacct fallback
# ---------------------------------------------------------------------------

def _fake_proc(stdout, returncode=0):
    return subprocess.CompletedProcess(args=["sacct"], returncode=returncode,
                                       stdout=stdout, stderr="")


def test_reduce_outcomes_sacct_purged_when_empty(run_dir, monkeypatch):
    _write_manifest(run_dir, n_specs=2)
    # No disk evidence for either index.
    append_job_record(run_dir, "train", "5000", [0, 1])

    monkeypatch.setattr(jt, "_run_slurm", lambda *a, **k: _fake_proc(""))

    out = reduce_outcomes(run_dir, "train")
    assert out == {0: "unknown_sacct_purged", 1: "unknown_sacct_purged"}


def test_reduce_outcomes_sacct_none_stdout_purged(run_dir, monkeypatch):
    _write_manifest(run_dir, n_specs=1)
    append_job_record(run_dir, "train", "5000", [0])

    monkeypatch.setattr(jt, "_run_slurm", lambda *a, **k: _fake_proc(None))

    out = reduce_outcomes(run_dir, "train")
    assert out == {0: "unknown_sacct_purged"}


def test_reduce_outcomes_sacct_cancelled_is_dependency_never_satisfied(
        run_dir, monkeypatch):
    _write_manifest(run_dir, n_specs=2)
    append_job_record(run_dir, "train", "5000", [0, 1])

    sacct_out = "\n".join([
        "5000_0|CANCELLED by 0|0:0",
        "5000_1|CANCELLED|0:0",
    ])
    monkeypatch.setattr(jt, "_run_slurm", lambda *a, **k: _fake_proc(sacct_out))

    out = reduce_outcomes(run_dir, "train")
    assert out == {0: "dependency_never_satisfied",
                   1: "dependency_never_satisfied"}


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

def test_reduce_outcomes_ignores_superseded_generation(run_dir, monkeypatch):
    _write_manifest(run_dir, n_specs=1)
    append_job_record(run_dir, "train", "5000", [0])  # generation 0
    append_job_record(run_dir, "train", "5001", [0])  # generation 1
    mark_superseded(run_dir, "train", 0)

    queried = []

    def fake(cmd, **kw):
        queried.append(cmd)
        # Only the newest, non-superseded job id (5001) should be queried.
        assert "--jobs=5001" in cmd
        return _fake_proc("5001_0|TIMEOUT|0:0")

    monkeypatch.setattr(jt, "_run_slurm", fake)

    out = reduce_outcomes(run_dir, "train")
    assert out == {0: "timeout"}
    # Exactly one sacct call, the superseded generation was never queried.
    assert len(queried) == 1


def test_reduce_outcomes_newest_generation_wins(run_dir, monkeypatch):
    _write_manifest(run_dir, n_specs=1)
    append_job_record(run_dir, "train", "5000", [0])  # gen 0
    append_job_record(run_dir, "train", "5001", [0])  # gen 1, newest

    def fake(cmd, **kw):
        if "--jobs=5001" in cmd:
            return _fake_proc("5001_0|COMPLETED|0:0")
        return _fake_proc("5000_0|FAILED|1:0")

    monkeypatch.setattr(jt, "_run_slurm", fake)

    out = reduce_outcomes(run_dir, "train")
    # Newest generation (5001) resolves index 0 first.
    assert out == {0: "success"}


def test_reduce_outcomes_falls_to_older_generation_when_newest_purged(
        run_dir, monkeypatch):
    _write_manifest(run_dir, n_specs=1)
    append_job_record(run_dir, "train", "5000", [0])  # gen 0
    append_job_record(run_dir, "train", "5001", [0])  # gen 1, newest

    def fake(cmd, **kw):
        if "--jobs=5001" in cmd:
            return _fake_proc("")          # newest purged
        return _fake_proc("5000_0|TIMEOUT|0:0")

    monkeypatch.setattr(jt, "_run_slurm", fake)

    out = reduce_outcomes(run_dir, "train")
    assert out == {0: "timeout"}


# ---------------------------------------------------------------------------
# reduce_outcomes: SlurmTransientError short-circuits
# ---------------------------------------------------------------------------

def test_reduce_outcomes_short_circuits_on_transient_error(run_dir, monkeypatch):
    _write_manifest(run_dir, n_specs=1)
    append_job_record(run_dir, "train", "5000", [0])  # gen 0
    append_job_record(run_dir, "train", "5001", [0])  # gen 1, newest

    calls = []

    def fake(cmd, **kw):
        calls.append(cmd)
        raise SlurmTransientError("controller unreachable")

    monkeypatch.setattr(jt, "_run_slurm", fake)

    with pytest.raises(SlurmTransientError):
        reduce_outcomes(run_dir, "train")
    # The FIRST transient error stops the reduction, the older generation
    # is NOT queried.
    assert len(calls) == 1


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


def test_run_slurm_query_verb_succeeds_after_retry(monkeypatch):
    seq = [1, 1, 0]  # fail, fail, succeed
    calls = []

    def fake_run(cmd, **kw):
        rc = seq[len(calls)]
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=rc,
                                           stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(jt.time, "sleep", lambda s: None)

    proc = _run_slurm(["squeue"], retries=3)
    assert proc.returncode == 0
    assert proc.stdout == "ok"
    assert len(calls) == 3


def test_run_slurm_mutating_verb_does_not_retry(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        # check=True means a non-zero exit raises CalledProcessError.
        raise subprocess.CalledProcessError(1, cmd, output="", stderr="bad")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        _run_slurm(["sbatch", "job.sh"])
    # sbatch is a mutating verb, run exactly ONCE, never retried.
    assert len(calls) == 1


def test_run_slurm_scancel_does_not_retry(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        raise subprocess.CalledProcessError(1, cmd, output="", stderr="bad")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        _run_slurm(["scancel", "5000"])
    assert len(calls) == 1


def test_run_slurm_applies_timeout(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["timeout"] = kw.get("timeout")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="",
                                           stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    _run_slurm(["sacct", "--jobs=1"])
    assert seen["timeout"] == 30.0


def test_run_slurm_rejects_unknown_verb():
    with pytest.raises(ValueError, match="unrecognized SLURM verb"):
        _run_slurm(["ls", "-l"])
