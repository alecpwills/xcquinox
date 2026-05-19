"""Tests for xcquinox.alec.cluster.__main__ — the harness CLI.

These tests NEVER shell out to a real SLURM controller: ``job_tracking._run_slurm``
is monkeypatched with canned ``sbatch`` / ``sacct`` / ``scancel`` behavior. Temp
run dirs are built with ``manifest.json`` / ``jobs.json`` / ``checkpoints/`` /
``resolved_config.yaml`` as each test needs. Grid configs are written as JSON
(no PyYAML dependency for the config-load path), but ``resolved_config.yaml`` is
exercised as a real YAML round-trip where the subcommand writes it.
"""
import json
import os
import subprocess

import pytest

from xcquinox.alec.cluster import job_tracking as jt
from xcquinox.alec.cluster import __main__ as cli
from xcquinox.alec.cluster.__main__ import main


# ---------------------------------------------------------------------------
# Config + run-dir fixtures
# ---------------------------------------------------------------------------

def _base_config_dict():
    """A complete, valid raw config dict. arch(1) x loss(1) x metric(2) x
    subset_size(3) x solver(1) = 6 grid cells -> array indices 0..5."""
    return {
        "sweep": {
            "arch": ["medium"],
            "loss": ["delta_ae"],
            "metric": ["l2", "jsd"],
            "subset_size": [4, 8, 12],
            "solver": ["fast"],
        },
        "solvers": {
            "fast": {"mode": "fixed_density", "max_cycles": 1},
        },
        "hyperparams": {
            "n_steps": 200,
            "lr_start": 1e-3,
            "lr_end": 1e-5,
            "lr_decay_start": 0.2,
            "grad_clip": 1.0,
            "gradnorm_alpha": 1.5,
            "vxc_weight": 1.0,
            "density_weight": 0.5,
        },
        "inputs": {
            "external_refs_dir": "/shared/refs",
            "descriptor_cache": "/shared/desc_cache",
            "refhist_cache": "/shared/refhist_cache",
            "subset_ledger_path": "/shared/ledger.json",
            "basis": "def2-tzvp",
            "grid_level": 3,
            "output_root": "/shared/runs",
        },
        "cluster": {
            "partition": "long-40core",
            "time": "12:00:00",
            "mem": "32G",
            "cpus_per_task": 4,
            "array_throttle": 4,
            "eval_array_throttle": 8,
            "max_concurrent_tasks": 40,
            "conda_profile": "/opt/conda/etc/profile.d/conda.sh",
            "conda_env": "xcq",
        },
        "domain_profile": "dfs_step7",
    }


# arch(1) x loss(1) x metric(2) x subset_size(3) x solver(1) = 6 cells.
_N = 6
_WIDTH = 4


def _write_grid(tmp_path, mutate=None):
    """Write a JSON grid config; return its path. ``mutate`` may edit the dict."""
    d = _base_config_dict()
    if mutate is not None:
        mutate(d)
    p = tmp_path / "grid.json"
    p.write_text(json.dumps(d))
    return str(p)


def _spec_dir(run_dir, idx, width=_WIDTH):
    d = os.path.join(run_dir, "checkpoints", f"spec_{idx:0{width}d}")
    os.makedirs(d, exist_ok=True)
    return d


def _write_manifest(run_dir, n=_N, width=_WIDTH, *, spec_hashes=None):
    """Write a manifest.json the way materialize.write_manifest would."""
    specs = []
    for i in range(n):
        entry = {"index": i, "cell": {}, "spec_file": f"spec_{i:0{width}d}.spec"}
        if spec_hashes is not None and i in spec_hashes:
            entry["sha256"] = spec_hashes[i]
        specs.append(entry)
    payload = {
        "xcquinox_version": "test",
        "python_version": "3.x",
        "width": width,
        "n_specs": n,
        "specs": specs,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(payload, f)


def _write_resolved_config(run_dir):
    """Write a real resolved_config.yaml from the base grid (via the CLI helper)."""
    from xcquinox.alec.cluster.grid_config import load_grid_config

    # Build a GridConfig from a temp JSON file, then serialize it to YAML.
    p = os.path.join(run_dir, "_tmp_grid.json")
    with open(p, "w") as f:
        json.dump(_base_config_dict(), f)
    cfg = load_grid_config(p)
    os.unlink(p)
    cli._write_resolved_config(cfg, run_dir)


def _make_run_dir(tmp_path, name="run", *, manifest=True, resolved=True,
                  n=_N, spec_hashes=None):
    """Create a run dir with the requested artifacts."""
    run_dir = tmp_path / name
    run_dir.mkdir()
    rd = str(run_dir)
    if resolved:
        _write_resolved_config(rd)
    if manifest:
        _write_manifest(rd, n=n, spec_hashes=spec_hashes)
    return rd


# ---------------------------------------------------------------------------
# Canned SLURM seam
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, stdout=""):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0


def _fake_slurm(ids=None, sacct_rows=None, fail_sbatch_index=None,
                fail_scancel=False, transient=False):
    """Build a fake ``_run_slurm``.

    ``ids``        — sequence of array-job ids returned for ``sbatch`` calls.
    ``sacct_rows`` — dict {array_job_id: "<JobID|State|ExitCode>\\n..."} for
                     ``sacct --jobs=<id>`` lookups.
    ``fail_sbatch_index`` — Nth (0-based) ``sbatch`` raises CalledProcessError.
    ``fail_scancel``      — every ``scancel`` raises CalledProcessError.
    ``transient``         — every ``sacct`` raises SlurmTransientError.
    Every cmd seen is recorded on ``.calls``.
    """
    ids = list(ids or ["1001", "1002", "1003", "1004", "1005", "1006"])
    sacct_rows = sacct_rows or {}
    state = {"sbatch_n": 0}
    calls = []

    def _fake(cmd, *, retries=3):
        calls.append(list(cmd))
        verb = os.path.basename(cmd[0])
        if verb == "sbatch":
            i = state["sbatch_n"]
            state["sbatch_n"] += 1
            if fail_sbatch_index is not None and i == fail_sbatch_index:
                raise subprocess.CalledProcessError(1, cmd, stderr="rejected")
            return _FakeProc(stdout=ids[i] + "\n")
        if verb == "scancel":
            if fail_scancel:
                raise subprocess.CalledProcessError(1, cmd, stderr="no perm")
            return _FakeProc(stdout="")
        if verb == "sacct":
            if transient:
                raise jt.SlurmTransientError("controller unreachable")
            job_id = None
            for tok in cmd:
                if tok.startswith("--jobs="):
                    job_id = tok.split("=", 1)[1]
            return _FakeProc(stdout=sacct_rows.get(job_id, ""))
        raise AssertionError(f"unexpected SLURM verb in test: {verb}")

    _fake.calls = calls
    return _fake


@pytest.fixture(autouse=True)
def _patch_slurm(monkeypatch):
    """Default: a no-op SLURM seam so a stray call is loud, not real."""
    monkeypatch.setattr(jt, "_run_slurm", _fake_slurm())


# ===========================================================================
# argparse dispatch
# ===========================================================================

def test_dispatch_unknown_subcommand_errors():
    with pytest.raises(SystemExit):
        main(["not-a-subcommand"])


def test_dispatch_all_six_subcommands_are_registered():
    parser = cli._build_parser()
    sub = [a for a in parser._subparsers._group_actions]
    choices = set()
    for action in sub:
        choices |= set(action.choices)
    assert choices == {
        "prepare", "submit", "status", "resubmit",
        "resubmit-preflight", "repair-manifest",
    }


# ===========================================================================
# prepare
# ===========================================================================

def test_prepare_regenerate_refused_on_login_node(tmp_path, monkeypatch):
    grid = _write_grid(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)  # simulate login node
    rc = main(["prepare", grid, "--regenerate"])
    assert rc == 2


def test_prepare_reuse_mode_runs_without_slurm_alloc(tmp_path, monkeypatch):
    grid = _write_grid(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    called = {}

    def _fake_prepare(cfg, regenerate):
        called["regenerate"] = regenerate

        class _S:
            points = [1, 2, 3]
            subset_ledger = {"entries": {"l2:4": {}}}
        return _S()

    monkeypatch.setattr(cli, "prepare_inputs", _fake_prepare)
    rc = main(["prepare", grid])  # reuse mode — no --regenerate
    assert rc == 0
    assert called["regenerate"] is False


def test_prepare_regenerate_allowed_inside_allocation(tmp_path, monkeypatch):
    grid = _write_grid(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "987654")  # simulate compute node
    monkeypatch.setattr(cli, "prepare_inputs", lambda cfg, regen: type(
        "S", (), {"points": [1], "subset_ledger": {"entries": {}}})())
    rc = main(["prepare", grid, "--regenerate"])
    assert rc == 0


# ===========================================================================
# submit
# ===========================================================================

def test_submit_creates_run_dir_and_resolved_config_dry_run(tmp_path,
                                                            monkeypatch):
    grid = _write_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main(["submit", grid, "--run-root", str(run_root)])
    assert rc == 0

    runs = os.listdir(run_root / "runs")
    assert len(runs) == 1 and runs[0].startswith("run_")
    run_dir = run_root / "runs" / runs[0]
    # resolved_config.yaml exists and round-trips through load_grid_config.
    from xcquinox.alec.cluster.grid_config import load_grid_config
    cfg = load_grid_config(str(run_dir / "resolved_config.yaml"))
    assert cfg.domain_profile == "dfs_step7"
    assert sorted(cfg.sweep.metric) == ["jsd", "l2"]
    # scripts/ + logs/ created; dry-run made NO sbatch call.
    assert os.path.isdir(run_dir / "scripts")
    assert os.path.isdir(run_dir / "logs")
    assert [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"] == []
    # no jobs.json in a dry run.
    assert not os.path.exists(run_dir / "jobs.json")


def test_submit_with_flag_calls_sbatch(tmp_path, monkeypatch):
    grid = _write_grid(tmp_path)
    fake = _fake_slurm(ids=["5000", "5001", "5002"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main(["submit", grid, "--run-root", str(run_root), "--submit"])
    assert rc == 0
    sbatch = [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"]
    assert len(sbatch) == 3


def test_submit_run_dir_collision_gets_counter_suffix(tmp_path, monkeypatch):
    """Two run dirs created in the same second must not collide."""
    monkeypatch.setattr(cli, "_utc_stamp", lambda: "20260519T120000Z")
    root = str(tmp_path / "out")
    d1 = cli._make_run_dir(root)
    d2 = cli._make_run_dir(root)
    assert d1 != d2
    assert os.path.basename(d2).endswith("_1")


# ===========================================================================
# status
# ===========================================================================

def test_status_aggregates_across_generations(tmp_path, monkeypatch):
    """Two train generations; gen-1 sacct resolves what gen-0 left pending."""
    run_dir = _make_run_dir(tmp_path)
    # disk evidence: index 0 succeeded, index 1 failed deterministically.
    open(os.path.join(_spec_dir(run_dir, 0), "model.eqx"), "wb").close()
    with open(os.path.join(_spec_dir(run_dir, 1), "failure.json"), "w") as f:
        json.dump({"classification": "assertion_error"}, f)

    # jobs.json: train gen0 (superseded) + gen1 (live); eval gen0 (live).
    jt.append_job_record(run_dir, "train", "1000", list(range(_N)))
    jt.mark_superseded(run_dir, "train", 0)
    jt.append_job_record(run_dir, "train", "2000", list(range(_N)))
    jt.append_job_record(run_dir, "eval", "3000", list(range(_N)))

    # gen-1 train sacct: indices 2,3 oom, 4,5 dependency-never-satisfied.
    # eval sacct: nothing scheduled (dependency never cleared).
    train_rows = "\n".join([
        "2000_2|OUT_OF_MEMORY|0:125",
        "2000_3|OUT_OF_MEMORY|0:125",
        "2000_4|CANCELLED by 0|0:0",
        "2000_5|CANCELLED by 0|0:0",
    ])
    fake = _fake_slurm(sacct_rows={"2000": train_rows, "3000": ""})
    monkeypatch.setattr(jt, "_run_slurm", fake)

    rc = main(["status", run_dir])
    assert rc == 0
    # status is read-only — it must NOT take the lock.
    assert not os.path.exists(os.path.join(run_dir, ".harness.lock"))


def test_status_handles_slurm_transient_error(tmp_path, monkeypatch):
    run_dir = _make_run_dir(tmp_path)
    jt.append_job_record(run_dir, "train", "1000", list(range(_N)))
    monkeypatch.setattr(jt, "_run_slurm", _fake_slurm(transient=True))
    # Must not crash — reports + returns non-zero.
    rc = main(["status", run_dir])
    assert rc == 1


def test_status_missing_manifest_directs_to_repair(tmp_path):
    run_dir = _make_run_dir(tmp_path, manifest=False)
    rc = main(["status", run_dir])
    assert rc == 1


# ===========================================================================
# resubmit
# ===========================================================================

def _make_resubmit_run(tmp_path, monkeypatch, spec_bytes=b"SPEC"):
    """Build a run dir whose specs/ + manifest hashes are consistent."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rd = str(run_dir)
    _write_resolved_config(rd)

    # Materialize real spec files + record their hashes in the manifest.
    specs_dir = os.path.join(rd, "specs")
    os.makedirs(specs_dir)
    import hashlib
    hashes = {}
    for i in range(_N):
        path = os.path.join(specs_dir, f"spec_{i:0{_WIDTH}d}.spec")
        with open(path, "wb") as f:
            f.write(spec_bytes + str(i).encode())
        hashes[i] = hashlib.sha256(spec_bytes + str(i).encode()).hexdigest()
    _write_manifest(rd, spec_hashes=hashes)

    # train gen0 covers all indices.
    jt.append_job_record(rd, "train", "1000", list(range(_N)))
    jt.append_job_record(rd, "eval", "2000", list(range(_N)))
    return rd


def test_resubmit_classifies_oom_via_sacct_and_submits_sparse(tmp_path,
                                                              monkeypatch):
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    # index 0 succeeded; indices 1,2 have NO failure.json -> sacct fallback.
    open(os.path.join(_spec_dir(rd, 0), "model.eqx"), "wb").close()
    # indices 3,4,5 failed deterministically (failure.json says so).
    for i in (3, 4, 5):
        with open(os.path.join(_spec_dir(rd, i), "failure.json"), "w") as f:
            json.dump({"classification": "value_error"}, f)

    # sacct: index 1 OOM, index 2 OOM.
    train_rows = "\n".join([
        "1000_1|OUT_OF_MEMORY|0:125",
        "1000_2|OUT_OF_MEMORY|0:125",
    ])
    fake = _fake_slurm(ids=["7001", "7002"], sacct_rows={"1000": train_rows})
    monkeypatch.setattr(jt, "_run_slurm", fake)

    rc = main(["resubmit", rd, "--submit"])
    assert rc == 0
    sbatch = [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"]
    assert len(sbatch) == 2  # one sparse train + one sparse eval array.
    # Both arrays span the SAME indices {1,2} (byte-identical, throttle aside).
    def _arr(cmd):
        for tok in cmd:
            if tok.startswith("--array="):
                return tok.split("=", 1)[1].split("%", 1)[0]
        raise AssertionError("no --array")
    assert _arr(sbatch[0]) == _arr(sbatch[1]) == "1,2"
    # eval array has aftercorr on the new train id.
    assert any("--dependency=aftercorr:7001" in t for t in sbatch[1])
    # stale failure-evidence-free indices archived (1,2 had no artifacts here);
    # attempts.json bumped for the two retried indices.
    attempts = json.load(open(os.path.join(rd, "attempts.json")))
    assert attempts == {"1": 1, "2": 1}


def test_resubmit_dry_run_makes_no_sbatch_call(tmp_path, monkeypatch):
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    open(os.path.join(_spec_dir(rd, 0), "model.eqx"), "wb").close()
    for i in range(1, _N):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    # All succeeded -> nothing to resubmit.
    rc = main(["resubmit", rd])
    assert rc == 0


def test_resubmit_respects_attempt_cap(tmp_path, monkeypatch):
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    open(os.path.join(_spec_dir(rd, 0), "model.eqx"), "wb").close()
    for i in (2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    # index 1 failed with OOM, but has already hit the attempt cap.
    with open(os.path.join(_spec_dir(rd, 1), "failure.json"), "w") as f:
        json.dump({"classification": "oom"}, f)
    cli._write_attempts(rd, {"1": 3})

    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd, "--submit", "--attempt-cap", "3"])
    assert rc == 0
    # Capped out -> no sbatch.
    assert [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"] == []


def test_resubmit_archives_stale_artifacts(tmp_path, monkeypatch):
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    open(os.path.join(_spec_dir(rd, 0), "model.eqx"), "wb").close()
    for i in (2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    # index 1: OOM with a stale failure.json — must be archived to *.gen0.
    with open(os.path.join(_spec_dir(rd, 1), "failure.json"), "w") as f:
        json.dump({"classification": "oom"}, f)

    fake = _fake_slurm(ids=["8001", "8002"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd, "--submit"])
    assert rc == 0
    spec1 = _spec_dir(rd, 1)
    # The stale failure.json was renamed; no plain failure.json remains.
    assert not os.path.exists(os.path.join(spec1, "failure.json"))
    assert os.path.exists(os.path.join(spec1, "failure.json.gen0"))


def test_resubmit_oom_retry_knob_none_fallback(tmp_path, monkeypatch):
    """With no oom_retry_partition/mem set, the retried index uses defaults
    and the run still submits."""
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    open(os.path.join(_spec_dir(rd, 0), "model.eqx"), "wb").close()
    for i in (2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    with open(os.path.join(_spec_dir(rd, 1), "failure.json"), "w") as f:
        json.dump({"classification": "oom"}, f)

    fake = _fake_slurm(ids=["9001", "9002"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd, "--submit"])
    assert rc == 0  # default-knob fallback still submits.
    assert len([c for c in fake.calls
                if os.path.basename(c[0]) == "sbatch"]) == 2


def test_resubmit_eval_failure_rolls_back_train(tmp_path, monkeypatch):
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    open(os.path.join(_spec_dir(rd, 0), "model.eqx"), "wb").close()
    for i in (2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    with open(os.path.join(_spec_dir(rd, 1), "failure.json"), "w") as f:
        json.dump({"classification": "oom"}, f)

    # Train sbatch (call 0) succeeds; eval sbatch (call 1) fails.
    fake = _fake_slurm(ids=["9100", "9101"], fail_sbatch_index=1)
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd, "--submit"])
    assert rc == 1
    # Train array was rolled back via scancel.
    scancels = [c for c in fake.calls if os.path.basename(c[0]) == "scancel"]
    assert scancels == [["scancel", "9100"]]
    # No new train/eval records appended (only the gen-0 ones from setup).
    records = jt.read_job_records(rd)
    assert len([r for r in records if r["kind"] == "train"]) == 1


def test_resubmit_respects_lock(tmp_path, monkeypatch):
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    # Pre-place a live lock (this very process's PID, same host).
    cli.acquire_lock(rd)
    rc = main(["resubmit", rd])
    assert rc == 1  # lock held by a live process -> refused.


# ===========================================================================
# resubmit-preflight
# ===========================================================================

def test_resubmit_preflight_refuses_when_train_evidence_exists(tmp_path):
    run_dir = _make_run_dir(tmp_path, manifest=False)
    # A train task left a failure.json -> training started; refuse.
    with open(os.path.join(_spec_dir(run_dir, 0), "failure.json"), "w") as f:
        json.dump({"classification": "oom"}, f)
    rc = main(["resubmit-preflight", run_dir])
    assert rc == 1


def test_resubmit_preflight_refuses_when_grid_changed(tmp_path):
    """Manifest records a different n_specs than the current grid expands to."""
    run_dir = _make_run_dir(tmp_path, manifest=True, n=_N - 1)
    rc = main(["resubmit-preflight", run_dir])
    assert rc == 1


def test_resubmit_preflight_refuses_when_manifest_complete(tmp_path):
    """A complete manifest means the preflight succeeded — use resubmit."""
    run_dir = _make_run_dir(tmp_path, manifest=True, n=_N)
    rc = main(["resubmit-preflight", run_dir])
    assert rc == 1


def test_resubmit_preflight_resubmits_and_supersedes(tmp_path, monkeypatch):
    """Genuinely preflight-stuck run: no manifest, no train evidence.
    Re-submits then scancels + marks the old arrays superseded."""
    run_dir = _make_run_dir(tmp_path, manifest=False)
    # Old train/eval arrays recorded from the failed first submit.
    jt.append_job_record(run_dir, "preflight", "100", [0])
    jt.append_job_record(run_dir, "train", "200", list(range(_N)))
    jt.append_job_record(run_dir, "eval", "300", list(range(_N)))

    fake = _fake_slurm(ids=["400", "401", "402"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit-preflight", run_dir, "--submit"])
    assert rc == 0
    # Three new sbatch calls + two scancels (old train + old eval).
    sbatch = [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"]
    scancel = [c for c in fake.calls if os.path.basename(c[0]) == "scancel"]
    assert len(sbatch) == 3
    assert sorted(c[1] for c in scancel) == ["200", "300"]
    # Old train/eval gen-0 records are now superseded.
    records = jt.read_job_records(run_dir)
    old = [r for r in records
           if r["array_job_id"] in ("200", "300")]
    assert all(r["superseded"] for r in old)


def test_resubmit_preflight_scancel_failure_skips_supersede(tmp_path,
                                                            monkeypatch):
    run_dir = _make_run_dir(tmp_path, manifest=False)
    jt.append_job_record(run_dir, "train", "200", list(range(_N)))
    jt.append_job_record(run_dir, "eval", "300", list(range(_N)))

    # All three new sbatch succeed; scancel fails.
    fake = _fake_slurm(ids=["400", "401", "402"], fail_scancel=True)
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit-preflight", run_dir, "--submit"])
    assert rc == 1
    # mark_superseded was SKIPPED — old records remain un-superseded.
    records = jt.read_job_records(run_dir)
    old = [r for r in records if r["array_job_id"] in ("200", "300")]
    assert all(not r["superseded"] for r in old)


def test_resubmit_preflight_dry_run_makes_no_call(tmp_path, monkeypatch):
    run_dir = _make_run_dir(tmp_path, manifest=False)
    jt.append_job_record(run_dir, "train", "200", list(range(_N)))
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit-preflight", run_dir])
    assert rc == 0
    assert fake.calls == []


# ===========================================================================
# repair-manifest
# ===========================================================================

def _make_specs_dir(run_dir, n=_N, width=_WIDTH):
    """Write n real spec files into <run_dir>/specs/ ; return their hashes."""
    import hashlib
    specs_dir = os.path.join(run_dir, "specs")
    os.makedirs(specs_dir, exist_ok=True)
    hashes = {}
    for i in range(n):
        path = os.path.join(specs_dir, f"spec_{i:0{width}d}.spec")
        data = b"SPECDATA" + str(i).encode()
        with open(path, "wb") as f:
            f.write(data)
        hashes[i] = hashlib.sha256(data).hexdigest()
    return hashes


def test_repair_manifest_rebuilds_corrupt_manifest(tmp_path):
    run_dir = _make_run_dir(tmp_path, manifest=False)
    hashes = _make_specs_dir(run_dir)
    # Write a corrupt manifest.json.
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        f.write("{ this is not valid json")

    rc = main(["repair-manifest", run_dir])
    assert rc == 0
    manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
    assert manifest["n_specs"] == _N
    by_idx = {e["index"]: e for e in manifest["specs"]}
    for i in range(_N):
        assert by_idx[i]["sha256"] == hashes[i]


def test_repair_manifest_rebuilds_missing_manifest(tmp_path):
    run_dir = _make_run_dir(tmp_path, manifest=False)
    _make_specs_dir(run_dir)
    assert not os.path.exists(os.path.join(run_dir, "manifest.json"))
    rc = main(["repair-manifest", run_dir])
    assert rc == 0
    assert os.path.exists(os.path.join(run_dir, "manifest.json"))


def test_repair_manifest_does_not_touch_model_eqx(tmp_path):
    run_dir = _make_run_dir(tmp_path, manifest=False)
    _make_specs_dir(run_dir)
    # A model.eqx + jobs.json + attempts.json must survive untouched.
    model = os.path.join(_spec_dir(run_dir, 0), "model.eqx")
    with open(model, "wb") as f:
        f.write(b"MODEL")
    jt.append_job_record(run_dir, "train", "500", list(range(_N)))
    cli._write_attempts(run_dir, {"3": 2})

    rc = main(["repair-manifest", run_dir])
    assert rc == 0
    assert open(model, "rb").read() == b"MODEL"
    assert jt.read_job_records(run_dir)[0]["array_job_id"] == "500"
    assert json.load(open(os.path.join(run_dir, "attempts.json"))) == {"3": 2}


def test_repair_manifest_errors_when_spec_count_mismatch(tmp_path):
    run_dir = _make_run_dir(tmp_path, manifest=False)
    _make_specs_dir(run_dir, n=_N - 2)  # too few spec files.
    rc = main(["repair-manifest", run_dir])
    assert rc == 1
    assert not os.path.exists(os.path.join(run_dir, "manifest.json"))


def test_repair_manifest_directs_to_fresh_dir_when_config_missing(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _make_specs_dir(str(run_dir))
    # No resolved_config.yaml at all.
    rc = main(["repair-manifest", str(run_dir)])
    assert rc == 1


def test_repair_manifest_directs_to_fresh_dir_when_config_corrupt(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _make_specs_dir(str(run_dir))
    with open(os.path.join(str(run_dir), "resolved_config.yaml"), "w") as f:
        f.write(": : not valid yaml mapping : :\n\t- broken")
    rc = main(["repair-manifest", str(run_dir)])
    assert rc == 1


# ===========================================================================
# .harness.lock — stale-lock reclaim
# ===========================================================================

def test_lock_reclaims_stale_lock_dead_pid(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rd = str(run_dir)
    # Plant a lock owned by a dead PID on THIS host.
    lock_path = os.path.join(rd, ".harness.lock")
    with open(lock_path, "w") as f:
        json.dump({"pid": 999999999, "hostname": __import__("socket").gethostname(),
                   "started_utc": "2020-01-01T00:00:00+00:00"}, f)
    # acquire_lock should reclaim it (dead PID, same host) and succeed.
    got = cli.acquire_lock(rd)
    assert got == lock_path
    info = json.load(open(lock_path))
    assert info["pid"] == os.getpid()


def test_lock_refuses_when_held_by_live_process(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rd = str(run_dir)
    cli.acquire_lock(rd)  # this process holds it now (live PID, same host).
    with pytest.raises(cli.HarnessLockError):
        cli.acquire_lock(rd)
    # --force reclaims it.
    cli.acquire_lock(rd, force=True)


def test_lock_release_removes_file(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rd = str(run_dir)
    lock_path = cli.acquire_lock(rd)
    assert os.path.exists(lock_path)
    cli.release_lock(lock_path)
    assert not os.path.exists(lock_path)
