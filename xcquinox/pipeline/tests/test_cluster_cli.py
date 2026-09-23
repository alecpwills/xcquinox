"""Tests for xcquinox.pipeline.cluster.__main__: the harness CLI.

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

from xcquinox.pipeline.cluster import job_tracking as jt
from xcquinox.pipeline.cluster import __main__ as cli
from xcquinox.pipeline.cluster.__main__ import main


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
            "subset_ledger_path": "/shared/ledger.json",
            "basis": "def2-tzvp",
            "grid_level": 3,
            "output_root": "/shared/runs",
        },
        "pretrain": {
            "data_dir": "/shared/pretrain_data",
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
        # prepare/submit refuse a DFS-domain FILE that leaves the BH76
        # objective silent (require_explicit_bh76_mode); the fixture states
        # the substitution the historical campaigns trained.
        "bh76_mode": "reaction_energy",
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
    from xcquinox.pipeline.cluster.grid_config import load_grid_config

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

    ``ids``: sequence of array-job ids returned for ``sbatch`` calls.
    ``sacct_rows``: dict {array_job_id: "<JobID|State|ExitCode>\\n..."} for
                     ``sacct --jobs=<id>`` lookups.
    ``fail_sbatch_index``: Nth (0-based) ``sbatch`` raises CalledProcessError.
    ``fail_scancel``: every ``scancel`` raises CalledProcessError.
    ``transient``: every ``sacct`` raises SlurmTransientError.
    Every cmd seen is recorded on ``.calls``.
    """
    ids = list(ids or ["1001", "1002", "1003", "1004", "1005", "1006",
                       "1007", "1008"])
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


def test_dispatch_all_subcommands_are_registered():
    parser = cli._build_parser()
    sub = [a for a in parser._subparsers._group_actions]
    choices = set()
    for action in sub:
        choices |= set(action.choices)
    assert choices == {
        "prepare", "submit", "submit-eval", "status", "results", "pull",
        "list-runs", "resubmit", "resubmit-preflight", "regate-certificates",
        "repair-manifest",
    }


# ===========================================================================
# prepare
# ===========================================================================

def test_prepare_refused_on_login_node(tmp_path, monkeypatch):
    """`prepare` runs the heavy CCSD precompute by default, refused on a
    login node (no $SLURM_JOB_ID)."""
    grid = _write_grid(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)  # simulate login node
    rc = main(["prepare", grid])
    assert rc == 2


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

    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "long-40core"])
    assert rc == 0

    runs = os.listdir(run_root / "runs")
    assert len(runs) == 1 and runs[0].startswith("run_")
    run_dir = run_root / "runs" / runs[0]
    # resolved_config.yaml exists and round-trips through load_grid_config.
    from xcquinox.pipeline.cluster.grid_config import load_grid_config
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
    fake = _fake_slurm(ids=["5000", "5001", "5002", "5003", "5004"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main(["submit", grid, "--run-root", str(run_root), "--submit",
               "--partition", "long-40core"])
    assert rc == 0
    sbatch = [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"]
    # 5-stage graph: datagen + pretrain + preflight + train + eval.
    assert len(sbatch) == 5
    # jobs.json records all five stages.
    runs = os.listdir(run_root / "runs")
    run_dir = str(run_root / "runs" / runs[0])
    kinds = sorted(r["kind"] for r in jt.read_job_records(run_dir))
    assert kinds == ["datagen", "eval", "preflight", "pretrain", "train"]


def test_submit_requires_partition(tmp_path):
    """submit without --partition is rejected by argparse (required; no default)."""
    grid = _write_grid(tmp_path)
    run_root = tmp_path / "out"
    run_root.mkdir()
    with pytest.raises(SystemExit):
        main(["submit", grid, "--run-root", str(run_root)])


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
    # status is read-only, it must NOT take the lock.
    assert not os.path.exists(os.path.join(run_dir, ".harness.lock"))


def test_status_missing_manifest_directs_to_repair(tmp_path):
    run_dir = _make_run_dir(tmp_path, manifest=False)
    rc = main(["status", run_dir])
    assert rc == 1


# ===========================================================================
# resubmit
# ===========================================================================

def _make_resubmit_run(tmp_path, monkeypatch, spec_bytes=b"SPEC"):
    """Build a run dir whose specs/ + manifest hashes are consistent.

    The captured ``scripts/train_array.sbatch`` + ``eval_array.sbatch`` are
    written too: ``resubmit`` refuses a run dir carrying no train script rather
    than handing ``sbatch`` a path that does not exist, so a run dir without
    them is not a run dir a resubmit can act on.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rd = str(run_dir)
    _write_resolved_config(rd)
    scripts_dir = os.path.join(rd, "scripts")
    os.makedirs(scripts_dir)
    for name in ("train_array.sbatch", "eval_array.sbatch"):
        with open(os.path.join(scripts_dir, name), "w") as f:
            f.write("#!/bin/bash\n#SBATCH --time=12:00:00\n")

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


def test_resubmit_respects_lock(tmp_path, monkeypatch):
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    # Pre-place a live lock (this very process's PID, same host).
    cli.acquire_lock(rd)
    rc = main(["resubmit", rd])
    assert rc == 1  # lock held by a live process -> refused.


# ===========================================================================
# resubmit-preflight
# ===========================================================================


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


# ===========================================================================
# .harness.lock: stale-lock reclaim
# ===========================================================================


def test_lock_refuses_when_held_by_live_process(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rd = str(run_dir)
    cli.acquire_lock(rd)  # this process holds it now (live PID, same host).
    with pytest.raises(cli.HarnessLockError):
        cli.acquire_lock(rd)
    # --force reclaims it.
    cli.acquire_lock(rd, force=True)


# ---------------------------------------------------------------------------
# --time overrides: the same walltime rule as the config fields they replace
# ---------------------------------------------------------------------------

#: Every stage script a ``--time`` base override reaches.
_TIME_SCRIPTS = ("pretrain.sbatch", "preflight.sbatch",
                 "train_array.sbatch", "eval_array.sbatch")


def test_results_subcommand_prints_table_and_writes_csv(tmp_path):
    """`results <run_dir>` returns 0, and --csv writes a file."""
    run_dir = _make_run_dir(tmp_path)
    # one completed eval so the table has a metric row.
    import csv as _csv
    d = _spec_dir(run_dir, 0)
    with open(os.path.join(d, "eval_df.csv"), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["set", "mae", "rho_rmse", "n_eval"])
        w.writeheader()
        w.writerow({"set": "training_subset", "mae": 1.5,
                    "rho_rmse": 0.02, "n_eval": 4})
    csv_out = str(tmp_path / "results.csv")
    rc = main(["results", run_dir, "--csv", csv_out])
    assert rc == 0
    assert os.path.isfile(csv_out)


# ===========================================================================
# WS6: incomplete_resumable -> RESUME path (resubmit) + status tally
# ===========================================================================

def _write_resume_state(run_dir, idx, width=_WIDTH):
    """Write a WS5 mid-run ``resume_state.pkl`` marker (presence is the signal)."""
    open(os.path.join(_spec_dir(run_dir, idx, width), "resume_state.pkl"),
         "wb").close()


def test_resolved_config_round_trip_preserves_every_field(tmp_path):
    """EVERY GridConfig field must survive serialize -> resolved_config.yaml
    -> load_grid_config. The preflight re-reads the resolved file before
    building specs, so a field the serializer drops silently reverts to its
    default for the whole run: ae_as_reactions was lost exactly this way,
    and every production sweep trained the AE channel in the fixed-anchor
    form its source YAML had turned off. Iterating dataclasses.fields keeps
    this test binding on fields added later.

    A field is guarded here ONLY while the config under test carries a
    NON-DEFAULT value for it: a field the serializer drops reloads at its
    default, which equals the value under test whenever the fixture YAML
    leaves that field alone, and the comparison then passes against a
    serializer that never wrote it. The fixture predates the fidelity block,
    so that block is injected below with all four of its fields off their
    defaults; a field added later needs the same treatment here."""
    import dataclasses

    import yaml

    from xcquinox.pipeline.cluster.grid_config import FidelityConfig

    cfg = cli.load_grid_config(
        "hpcjobs/configs/dfs_step7.dfs6311_grid3_v7g1_size.yaml")
    assert cfg.ae_as_reactions is True  # the field that was being dropped
    cfg = dataclasses.replace(cfg, fidelity=FidelityConfig(
        tol_AE=0.5, tol_atom=0.25,
        tol_AE_aggregate="mae", tol_AE_max_backstop=1.5,
        override_reason="round-trip fixture", enforce=False))
    _fid_default = FidelityConfig()
    for fld in dataclasses.fields(FidelityConfig):
        assert getattr(cfg.fidelity, fld.name) != getattr(
            _fid_default, fld.name), (
            f"FidelityConfig.{fld.name} is at its default in this fixture, so "
            "the round trip is NOT guarded for it")
    p = tmp_path / "resolved_config.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(cli._config_to_raw_dict(cfg), f)
    cfg2 = cli.load_grid_config(str(p))
    for fld in dataclasses.fields(type(cfg)):
        a, b = getattr(cfg, fld.name), getattr(cfg2, fld.name)
        assert a == b, (
            f"GridConfig.{fld.name} does not survive the resolved-config "
            f"round-trip: {a!r} -> {b!r}")


# ===========================================================================
# Certificate-config validation on every command that loads a config
# ===========================================================================


# ===========================================================================
# Inline-eval recovery: resubmit into the SAME run dir, and the wall semantics
#
# A run submitted with ``inline_eval: true`` renders ONE
# ``scripts/train_eval_inline.sbatch`` (train and eval in the same task) and no
# ``train_array.sbatch``/``eval_array.sbatch`` pair. ``resubmit`` used to refuse
# such a run outright, which left a wall-killed train cell with no recovery at
# all: a fresh ``submit`` opens a NEW timestamped run directory and never sees
# the checkpoints under the old one. These pins drive ``cmd_resubmit`` on
# synthetic run dirs, one per recovery path.
# ===========================================================================

def _write_scripts(run_dir, names):
    """Create ``scripts/<name>`` for each name; drop any other sbatch script."""
    scripts = os.path.join(run_dir, "scripts")
    os.makedirs(scripts, exist_ok=True)
    for existing in os.listdir(scripts):
        if existing.endswith(".sbatch"):
            os.unlink(os.path.join(scripts, existing))
    for name in names:
        with open(os.path.join(scripts, name), "w") as f:
            f.write("#!/bin/bash\n#SBATCH --time=48:00:00\n")
    return scripts


def _rewrite_resolved(tmp_path, run_dir, mutate, tag):
    """Rewrite ``run_dir``'s resolved_config.yaml from a mutated base dict."""
    from xcquinox.pipeline.cluster.grid_config import load_grid_config
    d = _base_config_dict()
    mutate(d)
    p = tmp_path / f"_cfg_{tag}.json"
    p.write_text(json.dumps(d))
    cli._write_resolved_config(load_grid_config(str(p)), run_dir)


# ===========================================================================
# status: the remedy for a dead PRETRAIN stage
# ===========================================================================
# A pretrain array task has no resubmit path -- `cmd_resubmit` reduces the
# train and eval kinds only -- so a pretrain stage that died takes the same
# on-disk signature as a dead preflight (nothing downstream ran, because the
# afterok dependency never fired) and the same recovery, `resubmit-preflight`.
# The remedy has to say which stage is incomplete, or an operator reads
# "preflight" for a run whose preflight never started.


# ===========================================================================
# bh76_mode explicitness: prepare/submit refuse a DFS-domain grid FILE that
# does not state its BH76 objective (the silent default trained the
# reaction-energy substitution through every campaign to v6).
# ===========================================================================


# ===========================================================================
# regate-certificates: in-place re-verdict under a changed gate
# ===========================================================================

_REGATE_FIDELITY = {"tol_AE": 1.0, "tol_atom": 1.0,
                    "tol_AE_aggregate": "mae", "tol_AE_max_backstop": 2.0,
                    "override_reason": None, "enforce": True}


def _regate_cert_payload(mol_dae=1.42, verdict="FAIL"):
    """A certificate for the base sweep's one arch, shaped like the writer's."""
    return {
        "verdict": verdict,
        "arch": "medium",
        "per_system": [
            {"name": "atom_H", "dE_xc_mHa": 0.5, "is_atom": True,
             "parent_grid_diff_Ha": 0.0, "parent_record_diff_Ha": 0.0,
             "reference_scf_converged": True},
            {"name": "H2", "dE_xc_mHa": 1.5, "is_atom": False,
             "parent_grid_diff_Ha": 0.0, "parent_record_diff_Ha": 0.0,
             "reference_scf_converged": True},
        ],
        "per_atomization": [{"name": "H2", "dAE_kcalmol": mol_dae},
                            {"name": "H2O", "dAE_kcalmol": 0.2}],
        "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                       "override_reason": None},
        "summary": {"max_atom_mHa": 0.5, "max_dAE_kcalmol": mol_dae,
                    "failure_reasons": (
                        [] if verdict == "PASS" else ["max |dAE| ..."])},
    }


def _regate_fixture(tmp_path, *, mol_dae=1.42, verdict="FAIL",
                    with_cert=True, tracked_overrides=None):
    """(run_dir, tracked_config_path, cert_path) for regate tests."""
    rd = _make_run_dir(tmp_path, manifest=False)
    cert_path = os.path.join(cli.pretrain_checkpoint_dir(rd, "medium"),
                             fid_CERTIFICATE_FILENAME)
    if with_cert:
        os.makedirs(os.path.dirname(cert_path), exist_ok=True)
        with open(cert_path, "w") as f:
            json.dump(_regate_cert_payload(mol_dae, verdict), f)
    raw = _base_config_dict()
    raw["fidelity"] = dict(_REGATE_FIDELITY)
    for key, value in (tracked_overrides or {}).items():
        section, _, name = key.partition(".")
        raw[section][name] = value
    tracked = str(tmp_path / "tracked_config.json")
    with open(tracked, "w") as f:
        json.dump(raw, f)
    return rd, tracked, cert_path


# The certificate filename constant, through the module the CLI imports it
# from, so a rename breaks here and not silently in the fixture.
from xcquinox.pipeline.cluster.fidelity import (  # noqa: E402
    CERTIFICATE_FILENAME as fid_CERTIFICATE_FILENAME)


def test_regate_apply_rewrites_the_certificate_and_the_resolved_config(
        tmp_path):
    rd, tracked, cert_path = _regate_fixture(tmp_path)
    rc = main(["regate-certificates", rd, "--config", tracked, "--apply"])
    assert rc == 0
    with open(cert_path) as f:
        cert = json.load(f)
    assert cert["verdict"] == "PASS"
    assert cert["regate"]["original_verdict"] == "FAIL"
    assert cert["regate"]["config_source"] == tracked
    assert cert["tolerances"]["tol_AE_aggregate"] == "mae"
    assert cert["summary"]["species_over_1_kcalmol"] == ["H2"]
    cfg2 = cli.load_grid_config(os.path.join(rd,
                                             cli._RESOLVED_CONFIG_FILENAME))
    assert cfg2.fidelity.tol_AE_aggregate == "mae"
    assert cfg2.fidelity.tol_AE_max_backstop == 2.0


