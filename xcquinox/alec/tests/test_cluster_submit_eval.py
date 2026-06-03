"""Tests for the deferred-eval submitter (xcquinox.alec.cluster._submit_eval).

Covers the launcher/manual core: submitting the eval array (aftercorr on the
train array) + recording it, idempotency, the no-train-record error, and that a
compute-node sbatch failure surfaces a clear message pointing at the login-node
fallback. SLURM is always mocked, no real controller is contacted.
"""
import os
import shutil
import subprocess

import pytest

from xcquinox.alec.cluster import _submit_eval as SE
from xcquinox.alec.cluster import job_tracking as jt

# Bound without a trailing "(" on the function name on purpose; invoked as
# _sde(...) below.
_sde = SE.submit_deferred_eval

_EXAMPLE = "xcquinox/alec/cluster/examples/grid_step7.yaml"


class _Proc:
    def __init__(self, stdout):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


def _make_run_dir(tmp_path, *, with_train=True):
    rd = str(tmp_path / "run")
    os.makedirs(os.path.join(rd, "scripts"), exist_ok=True)
    shutil.copy(_EXAMPLE, os.path.join(rd, "resolved_config.yaml"))
    # An eval_array.sbatch to be re-used (content irrelevant to these tests).
    with open(os.path.join(rd, "scripts", "eval_array.sbatch"), "w") as f:
        f.write("#!/usr/bin/env bash\n#SBATCH --array=0-2\n")
    if with_train:
        jt.append_job_record(rd, "pretrain", "100", [0])
        jt.append_job_record(rd, "preflight", "101", [0])
        jt.append_job_record(rd, "train", "102", [0, 1, 2])
    return rd


def _fake_slurm(ids):
    calls = []
    box = {"i": 0}

    def fake(cmd, *, retries=3):
        calls.append(list(cmd))
        if os.path.basename(cmd[0]) == "sbatch":
            box["i"] += 1
            return _Proc(f"{ids[box['i'] - 1]}\n")
        return _Proc("")

    fake.calls = calls
    return fake


def test_submit_deferred_eval_submits_aftercorr_and_records(tmp_path, monkeypatch):
    rd = _make_run_dir(tmp_path)
    fake = _fake_slurm(["999"])
    monkeypatch.setattr(jt, "_run_slurm", fake)

    result = _sde(rd)

    assert result["submitted"] is True
    assert result["eval_id"] == "999"
    assert result["train_id"] == "102"
    sb = [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"][0]
    joined = " ".join(sb)
    # aftercorr gating on the train array, re-using the existing eval script.
    assert "--dependency=aftercorr:102" in joined
    assert joined.endswith("eval_array.sbatch")
    # The eval record now exists, carrying the train record's indices.
    recs = jt.read_job_records(rd)
    evrecs = [r for r in recs if r["kind"] == "eval"]
    assert len(evrecs) == 1
    assert evrecs[0]["array_job_id"] == "999"
    assert evrecs[0]["indices"] == [0, 1, 2]


def test_submit_deferred_eval_is_idempotent(tmp_path, monkeypatch):
    rd = _make_run_dir(tmp_path)
    monkeypatch.setattr(jt, "_run_slurm", _fake_slurm(["999"]))
    _sde(rd)

    # Second call: no-op, no new sbatch, still exactly one eval record.
    fake2 = _fake_slurm(["zzz"])
    monkeypatch.setattr(jt, "_run_slurm", fake2)
    r2 = _sde(rd)
    assert r2["submitted"] is False
    assert r2["reason"] == "already_submitted"
    assert r2["eval_id"] == "999"
    assert [c for c in fake2.calls if os.path.basename(c[0]) == "sbatch"] == []
    assert sum(1 for r in jt.read_job_records(rd) if r["kind"] == "eval") == 1


def test_submit_deferred_eval_force_resubmits(tmp_path, monkeypatch):
    rd = _make_run_dir(tmp_path)
    monkeypatch.setattr(jt, "_run_slurm", _fake_slurm(["999"]))
    _sde(rd)

    fake2 = _fake_slurm(["1000"])
    monkeypatch.setattr(jt, "_run_slurm", fake2)
    r2 = _sde(rd, force=True)
    assert r2["submitted"] is True
    assert r2["eval_id"] == "1000"
    assert sum(1 for r in jt.read_job_records(rd) if r["kind"] == "eval") == 2


def test_submit_deferred_eval_no_train_record_raises(tmp_path, monkeypatch):
    rd = _make_run_dir(tmp_path, with_train=False)
    monkeypatch.setattr(jt, "_run_slurm", _fake_slurm(["999"]))
    with pytest.raises(RuntimeError, match="no live train record"):
        _sde(rd)


def test_submit_deferred_eval_surfaces_sbatch_error(tmp_path, monkeypatch):
    rd = _make_run_dir(tmp_path)

    def boom(cmd, *, retries=3):
        raise subprocess.CalledProcessError(
            1, cmd, stderr="Batch job submission failed")

    monkeypatch.setattr(jt, "_run_slurm", boom)
    with pytest.raises(RuntimeError, match="login node"):
        _sde(rd)
    # No eval record written when sbatch is rejected.
    assert [r for r in jt.read_job_records(rd) if r["kind"] == "eval"] == []
