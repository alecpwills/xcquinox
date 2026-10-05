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


# ---------------------------------------------------------------------------
# Submit from any directory (the submit-anywhere rule)
# ---------------------------------------------------------------------------

def _fake_checkout(tmp_path):
    """A fake checkout root carrying the layout marker (hpcjobs/ + pyproject)."""
    root = tmp_path / "fake_checkout"
    (root / "hpcjobs").mkdir(parents=True)
    (root / "pyproject.toml").write_text("# fake checkout\n", encoding="utf-8")
    return root


def test_submit_anywhere_relative_run_root_lands_under_the_checkout(
        tmp_path, monkeypatch):
    """A relative --run-root resolves against the checkout, not the caller's
    CWD: the runbook's ``--run-root hpcjobs`` spelling works from anywhere."""
    checkout = _fake_checkout(tmp_path)
    monkeypatch.setattr(cli, "_repo_root", lambda: str(checkout))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    grid = _write_grid(tmp_path)
    monkeypatch.setattr(jt, "_run_slurm", _fake_slurm())

    rc = main(["submit", grid, "--run-root", "hpcjobs/zz_out",
               "--partition", "long-40core"])
    assert rc == 0
    runs = list((checkout / "hpcjobs" / "zz_out" / "runs").iterdir())
    assert len(runs) == 1, "the run dir must sit under the checkout"
    assert not (elsewhere / "hpcjobs").exists(), \
        "a relative run root must not resolve against the caller's CWD"


def test_submit_anywhere_checkout_relative_grid_path_resolves(
        tmp_path, monkeypatch):
    """A grid path spelled relative to the checkout (the runbook's
    ``hpcjobs/configs/...``) resolves there when it does not exist as given."""
    checkout = _fake_checkout(tmp_path)
    grid_rel = _write_grid(checkout / "hpcjobs")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(cli, "_repo_root", lambda: str(checkout))
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(jt, "_run_slurm", _fake_slurm())

    rc = main(["submit", os.path.relpath(grid_rel, checkout),
               "--run-root", str(out), "--partition", "long-40core"])
    assert rc == 0, "the checkout-relative spelling must resolve and dry-run"
    assert len(os.listdir(out / "runs")) == 1


# ---------------------------------------------------------------------------
# Donor warm-starts: the status tally and regate read the donors
# ---------------------------------------------------------------------------

def test_pretrain_counts_reads_donors(tmp_path):
    """The status tally counts a donor arch on its donor (the networks and
    the certificate live there); a tally that counted run-local dirs would
    call a released donor arch uncertified and mis-state the stalled stage."""
    from xcquinox.pipeline.cluster.grid_config import load_grid_config

    donor = tmp_path / "donor_run" / "pretrain" / "medium"
    donor.mkdir(parents=True)
    (donor / "xnet.eqx").write_bytes(b"x")
    (donor / "cnet.eqx").write_bytes(b"c")
    with open(donor / "fidelity_certificate.json", "w") as f:
        json.dump({"verdict": "PASS", "arch": "medium",
                   "summary": {"max_atom_mHa": 0.1,
                               "max_dAE_kcalmol": 0.2}}, f)

    d = _base_config_dict()
    d["pretrain"]["donor_checkpoints"] = {"medium": str(donor)}
    p = tmp_path / "g.json"
    p.write_text(json.dumps(d))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cli._write_resolved_config(load_grid_config(str(p)), str(run_dir))

    assert cli._pretrain_counts(str(run_dir)) == (1, 1, 1, 1, [])


def test_regate_skips_donor_archs_and_reports_nothing_to_regate(
        tmp_path, capsys):
    """A run whose every arch warm-starts from a donor has no run-local
    certificate to re-verdict: the command says so and returns 1 (a success
    exit having done nothing would mask the state)."""
    rd, tracked, _cert = _regate_fixture(tmp_path)
    _rewrite_resolved(
        tmp_path, rd,
        lambda d: d["pretrain"].__setitem__(
            "donor_checkpoints", {"medium": "/gpfs/donors/medium"}),
        "donor")

    rc = main(["regate-certificates", rd, "--config", tracked, "--apply"])

    assert rc == 1
    out = capsys.readouterr().out
    assert "donor" in out
    assert "nothing to regate" in out


def test_prepare_anywhere_checkout_relative_grid_path_resolves(
        tmp_path, monkeypatch, capsys):
    """``prepare`` resolves the grid argument the same way ``submit`` does:
    the checkout-relative spelling works from any directory, and the refs
    precompute is skipped (``prepare_inputs`` stubbed; an unresolved grid
    would fail at load with rc 1 and no resolution log line)."""
    from types import SimpleNamespace

    checkout = _fake_checkout(tmp_path)
    grid_rel = _write_grid(checkout / "hpcjobs")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(cli, "_repo_root", lambda: str(checkout))
    calls = []
    monkeypatch.setattr(
        cli, "prepare_inputs",
        lambda cfg, recompute_refs: calls.append(recompute_refs)
        or SimpleNamespace(points=[1, 2], subset_ledger=[{}, {}]))

    rc = main(["prepare", os.path.relpath(grid_rel, checkout),
               "--no-recompute-refs"])
    assert rc == 0, "the checkout-relative spelling must resolve and prepare"
    assert calls == [False]
    assert "resolved against the checkout" in capsys.readouterr().out
    assert not (elsewhere / "hpcjobs").exists(), \
        "a checkout-relative grid must not resolve against the caller's CWD"




# ===========================================================================
# Stage-selective submission: --stages pretrain|optimize and --donor-run
# ===========================================================================

def test_submit_stages_pretrain_cli_dry_run(tmp_path, monkeypatch):
    """``submit --stages pretrain`` dry-runs the datagen -> pretrain prefix
    only: scripts/ carries exactly the two selected stage scripts, no sbatch
    is made, no jobs.json is written, and the submit-commands record names no
    other stage's script (a line for an unsubmitted stage would read as a
    submission that failed)."""
    grid = _write_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "long-40core", "--stages", "pretrain"])
    assert rc == 0

    runs = os.listdir(run_root / "runs")
    assert len(runs) == 1
    run_dir = run_root / "runs" / runs[0]
    assert sorted(os.listdir(run_dir / "scripts")) == [
        "datagen.sbatch", "pretrain.sbatch"]
    assert [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"] == []
    assert not os.path.exists(run_dir / "jobs.json")
    cmds = open(os.path.join(str(run_dir), "submit_commands.txt")).read()
    for absent in ("preflight.sbatch", "train_array.sbatch",
                   "eval_array.sbatch"):
        assert absent not in cmds


def _two_arch_grid(tmp_path):
    """The base grid with two registry keys, medium and shallow, swept."""
    return _write_grid(tmp_path, mutate=lambda d: d["sweep"].__setitem__(
        "arch", ["medium", "shallow"]))


def test_submit_archs_and_pretrain_seed_state_a_one_arch_refit(tmp_path):
    """``submit --stages pretrain --archs shallow --pretrain-seed 7`` on a
    grid sweeping medium and shallow stages a pretraining run of shallow
    alone at initialization seed 7. The run's resolved_config.yaml, which
    every node stage re-reads, sweeps shallow only and states pretrain.seed
    7; hyperparams.seed, the optimization seed, keeps the value the grid file
    loads to; and the pretrain array is the single task 0-0. Shallow is the
    later name in canonical order, so index 0 of that array is shallow only
    if the axis itself was restricted."""
    from xcquinox.pipeline.cluster.grid_config import load_grid_config

    grid = _two_arch_grid(tmp_path)
    base = load_grid_config(grid)
    assert sorted(set(base.sweep.arch)) == ["medium", "shallow"]
    assert 7 not in (base.pretrain.seed, base.hyperparams.seed)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "long-40core", "--stages", "pretrain",
               "--archs", "shallow", "--pretrain-seed", "7"])
    assert rc == 0

    runs = os.listdir(run_root / "runs")
    assert len(runs) == 1
    run_dir = run_root / "runs" / runs[0]
    cfg = load_grid_config(str(run_dir / "resolved_config.yaml"))
    assert sorted(set(cfg.sweep.arch)) == ["shallow"]
    assert cfg.pretrain.seed == 7
    assert cfg.hyperparams.seed == base.hyperparams.seed
    # The template renders "#SBATCH --array=0-<ARRAY_MAX>%<THROTTLE>"; the
    # index range is the part before the throttle.
    script = (run_dir / "scripts" / "pretrain.sbatch").read_text()
    ranges = [line.split("=", 1)[1].split("%", 1)[0]
              for line in script.splitlines()
              if line.startswith("#SBATCH --array=")]
    assert ranges == ["0-0"]


def test_submit_archs_outside_the_sweep_refused_before_run_dir(
        tmp_path, monkeypatch):
    """``--archs`` naming a registry key the grid does not sweep is refused
    before the run dir exists: rc 1, nothing under the run root, no sbatch.
    The override restricts the configuration's sweep and never extends it.
    The list pairs a swept name with the unswept one, so only the membership
    check refuses it; a restriction that dropped the unswept name without a
    word would stage and submit a run of the swept one."""
    grid = _two_arch_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "long-40core", "--stages", "pretrain",
               "--archs", "shallow,deep", "--submit"])

    assert rc == 1
    assert os.listdir(str(run_root)) == []
    assert [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"] == []

    # A list that names nothing is refused the same way, under the whole
    # graph too, where an emptied axis would otherwise reach the grid
    # expansion.
    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "long-40core", "--archs", ","])
    assert rc == 1
    assert os.listdir(str(run_root)) == []


def test_submit_pretrain_seed_is_held_to_the_loaders_seed_range(tmp_path):
    """``--pretrain-seed`` takes exactly the range the configuration loader
    holds ``pretrain.seed`` to. A value the loader refuses, written into
    resolved_config.yaml unchecked, would be refused only when the first node
    stage reloads that file, after the graph was queued; it is refused at
    submit instead, before the run dir exists. The two ends of the range are
    seeds like any other and are written into the run."""
    from xcquinox.pipeline.cluster.grid_config import (_MAX_SEED,
                                                       load_grid_config)

    grid = _write_grid(tmp_path)
    run_root = tmp_path / "out"
    run_root.mkdir()

    for seed in (-1, _MAX_SEED + 1):
        rc = main(["submit", grid, "--run-root", str(run_root),
                   "--partition", "long-40core", "--stages", "pretrain",
                   f"--pretrain-seed={seed}"])
        assert rc == 1
        assert os.listdir(str(run_root)) == []

    for seed in (0, _MAX_SEED):
        root = tmp_path / f"accepted_{seed}"
        root.mkdir()
        rc = main(["submit", grid, "--run-root", str(root),
                   "--partition", "long-40core", "--stages", "pretrain",
                   f"--pretrain-seed={seed}"])
        assert rc == 0
        (run,) = os.listdir(root / "runs")
        cfg = load_grid_config(
            str(root / "runs" / run / "resolved_config.yaml"))
        assert cfg.pretrain.seed == seed


def test_submit_pretrain_seed_refused_on_a_run_that_pretrains_nothing(
        tmp_path):
    """``--pretrain-seed`` names the seed of a pretraining. A run seeded
    entirely from donors pretrains nothing, and a seed written into its
    resolved_config.yaml would state an initialization none of its networks
    had: the flag is refused there, before the run dir exists, whether the
    stage group excludes the pretraining or the donor map empties it."""
    arch = "medium"  # _write_grid's single sweep arch
    donor_run = tmp_path / "pretrain_run"
    donor_dir = donor_run / "pretrain" / arch
    donor_dir.mkdir(parents=True)
    with open(donor_dir / "fidelity_certificate.json", "w") as f:
        json.dump({"verdict": "PASS", "arch": arch}, f)

    grid = _write_grid(tmp_path)
    for stages in ("optimize", "all"):
        run_root = tmp_path / f"out_{stages}"
        run_root.mkdir()
        rc = main(["submit", grid, "--run-root", str(run_root),
                   "--partition", "long-40core", "--stages", stages,
                   "--donor-run", str(donor_run), "--pretrain-seed", "7"])
        assert rc == 1
        assert os.listdir(str(run_root)) == []


def test_submit_donor_run_is_asked_for_the_architectures_archs_keeps(
        tmp_path):
    """A ``--donor-run`` holding the one architecture ``--archs`` keeps seeds
    the restricted run: the donor map is built over the axis the run sweeps,
    so the architectures left out are not asked of the donor run. This is the
    optimization of a refit, a pretraining suite of one network."""
    from xcquinox.pipeline.cluster.grid_config import load_grid_config

    donor_run = tmp_path / "refit_run"
    donor_dir = donor_run / "pretrain" / "shallow"
    donor_dir.mkdir(parents=True)
    with open(donor_dir / "fidelity_certificate.json", "w") as f:
        json.dump({"verdict": "PASS", "arch": "shallow"}, f)

    grid = _two_arch_grid(tmp_path)
    run_root = tmp_path / "out"
    run_root.mkdir()
    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "long-40core", "--stages", "optimize",
               "--archs", "shallow", "--donor-run", str(donor_run)])
    assert rc == 0
    (run,) = os.listdir(run_root / "runs")
    cfg = load_grid_config(
        str(run_root / "runs" / run / "resolved_config.yaml"))
    assert cfg.pretrain.donor_checkpoints == {"shallow": str(donor_dir)}


def test_submit_archs_narrows_the_donor_map_to_the_swept_architectures(
        tmp_path):
    """A configuration-stated donor for an architecture ``--archs`` leaves
    out goes with it: the run's resolved_config.yaml names the donors of the
    architectures it sweeps and no other. A donor key the configuration's own
    sweep never carried is a different matter, a misspelt key, and stays
    refused whatever ``--archs`` says."""
    from xcquinox.pipeline.cluster.grid_config import load_grid_config

    def _donors(extra):
        def _mutate(d):
            d["sweep"]["arch"] = ["medium", "shallow"]
            d["pretrain"]["donor_checkpoints"] = {
                "medium": "/shared/donors/medium",
                "shallow": "/shared/donors/shallow", **extra}
        return _mutate

    grid = _write_grid(tmp_path, mutate=_donors({}))
    run_root = tmp_path / "out"
    run_root.mkdir()
    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "long-40core", "--stages", "optimize",
               "--archs", "shallow"])
    assert rc == 0
    runs = os.listdir(run_root / "runs")
    assert len(runs) == 1
    cfg = load_grid_config(
        str(run_root / "runs" / runs[0] / "resolved_config.yaml"))
    assert cfg.pretrain.donor_checkpoints == {
        "shallow": "/shared/donors/shallow"}

    misspelt = _write_grid(tmp_path, mutate=_donors(
        {"medim": "/shared/donors/medim"}))
    with pytest.raises(ValueError):
        main(["submit", misspelt, "--run-root", str(tmp_path / "other"),
              "--partition", "long-40core", "--stages", "optimize",
              "--archs", "shallow"])


def test_submit_stages_optimize_with_donor_run(tmp_path, monkeypatch):
    """``submit --stages optimize --donor-run DIR`` builds the donor map from
    a completed pretraining run (one entry per canonical sweep arch), and
    that map round-trips through resolved_config.yaml -- the recovery
    commands and the node stages re-read it there, so a map the serializer
    dropped would silently revert every arch to run-local pretrain dirs.
    The submitted graph is preflight (no dependency) -> train -> eval."""
    from xcquinox.pipeline.cluster.grid_config import load_grid_config

    arch = "medium"  # _write_grid's single sweep arch
    donor_run = tmp_path / "pretrain_run"
    donor_dir = donor_run / "pretrain" / arch
    donor_dir.mkdir(parents=True)
    with open(donor_dir / "fidelity_certificate.json", "w") as f:
        json.dump({"verdict": "PASS", "arch": arch}, f)

    grid = _write_grid(tmp_path)
    fake = _fake_slurm(ids=["5000", "5001", "5002"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "long-40core", "--stages", "optimize",
               "--donor-run", str(donor_run), "--submit"])
    assert rc == 0

    runs = os.listdir(run_root / "runs")
    assert len(runs) == 1
    run_dir = str(run_root / "runs" / runs[0])
    cfg = load_grid_config(os.path.join(run_dir, "resolved_config.yaml"))
    assert cfg.pretrain.donor_checkpoints == {arch: str(donor_dir)}
    kinds = [r["kind"] for r in jt.read_job_records(run_dir)]
    assert kinds == ["preflight", "train", "eval"]
    sbatch = [" ".join(c) for c in fake.calls
              if os.path.basename(c[0]) == "sbatch"]
    assert len(sbatch) == 3
    assert "--dependency" not in sbatch[0]  # preflight, no datagen before it


def test_submit_donor_run_missing_certificate_refused_before_run_dir(
        tmp_path, monkeypatch, capsys):
    """G5: a --donor-run whose ``<DIR>/pretrain/<arch>`` exists but carries
    no fidelity_certificate.json cannot be gated, so the submission is
    refused BEFORE the run dir is created (rc 1, no runs/ directory, zero
    sbatch) and the message names the path. A refusal that left a run dir
    behind would hand list-runs and status a half-built run."""
    arch = "medium"
    donor_run = tmp_path / "pretrain_run"
    (donor_run / "pretrain" / arch).mkdir(parents=True)  # dir, no certificate

    grid = _write_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "long-40core", "--stages", "optimize",
               "--donor-run", str(donor_run), "--submit"])

    assert rc == 1
    # Refusal lands before _make_run_dir: nothing under the run root, and
    # the message names the offending donor path.
    assert os.listdir(str(run_root)) == []
    assert not os.path.exists(os.path.join(str(run_root), "runs"))
    assert [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"] == []
    assert str(donor_run / "pretrain" / arch) in capsys.readouterr().out


def test_submit_stages_optimize_refused_before_run_dir(tmp_path, monkeypatch,
                                                       capsys):
    """O1: a refused stage selection must not leave a half-built run behind.
    ``--stages optimize`` on a donorless grid is refused in cmd_submit BEFORE
    the run dir, resolved_config.yaml and scripts/ exist (rc 1, empty run
    root, zero sbatch) -- the same pre-run-dir shape the --donor-run
    certificate refusal has. A refusal that fired inside submit_jobs would
    leave a stray timestamped run dir for list-runs and status to trip over."""
    grid = _write_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "long-40core", "--stages", "optimize",
               "--submit"])

    assert rc == 1
    assert os.listdir(str(run_root)) == []
    assert not os.path.exists(os.path.join(str(run_root), "runs"))
    assert [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"] == []
    assert "no donor" in capsys.readouterr().out


def test_submit_relative_donor_run_resolves_against_checkout(
        tmp_path, monkeypatch, capsys):
    """O3: a relative ``--donor-run`` resolves against the checkout, the same
    rule the grid path and ``--run-root`` follow (submit-anywhere). Resolving
    it against the caller's CWD would build a donor map of nonexistent paths
    that the login-node certificate check then refuses with a confusing
    message."""
    from xcquinox.pipeline.cluster.grid_config import load_grid_config

    arch = "medium"  # _write_grid's single sweep arch
    checkout = tmp_path / "fake_checkout"
    donor_dir = checkout / "hpcjobs" / "donor_run" / "pretrain" / arch
    donor_dir.mkdir(parents=True)
    with open(donor_dir / "fidelity_certificate.json", "w") as f:
        json.dump({"verdict": "PASS", "arch": arch}, f)

    grid = _write_grid(tmp_path)  # absolute, outside the fake checkout
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(cli, "_repo_root", lambda: str(checkout))
    fake = _fake_slurm(ids=["5000", "5001", "5002"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "long-40core", "--stages", "optimize",
               "--donor-run", "hpcjobs/donor_run", "--submit"])
    assert rc == 0

    runs = os.listdir(run_root / "runs")
    assert len(runs) == 1
    run_dir = str(run_root / "runs" / runs[0])
    cfg = load_grid_config(os.path.join(run_dir, "resolved_config.yaml"))
    assert cfg.pretrain.donor_checkpoints == {arch: str(donor_dir)}
    out = capsys.readouterr().out
    assert "resolved against the checkout" in out
    assert not (elsewhere / "hpcjobs").exists(), \
        "a checkout-relative donor run must not resolve against the CWD"


def test_resubmit_preflight_recovers_the_pretrain_group_only(
        tmp_path, monkeypatch, capsys):
    """O2: ``resubmit-preflight`` on a pretrain-only run dir derives its stage
    group from the recorded kinds, so recovery re-submits datagen + pretrain
    and NEVER grows a preflight/train/eval graph the run never had. A dead
    pretrain task is precisely the recovery case, and a recovery that queued
    the full graph would train with no manifest from a preflight the operator
    never asked this run to run."""
    grid = _write_grid(tmp_path)
    fake = _fake_slurm(ids=["3000", "3001", "3002", "3003"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "long-40core", "--stages", "pretrain",
               "--submit"])
    assert rc == 0
    run_dir = str(run_root / "runs" / os.listdir(run_root / "runs")[0])

    rc = main(["resubmit-preflight", run_dir, "--submit"])
    assert rc == 0

    # The recovered graph is still the pretrain group: only datagen/pretrain
    # records exist (old pair now superseded, new pair live), and exactly two
    # new sbatch calls were made (plus the original two).
    records = jt.read_job_records(run_dir)
    kinds = {r["kind"] for r in records}
    assert kinds == {"datagen", "pretrain"}
    sbatch = [" ".join(c) for c in fake.calls
              if os.path.basename(c[0]) == "sbatch"]
    assert len(sbatch) == 4
    scancels = [c for c in fake.calls if os.path.basename(c[0]) == "scancel"]
    assert scancels == [["scancel", "3000"], ["scancel", "3001"]]
    out = capsys.readouterr().out
    assert "pretrain stage group" in out


def test_resubmit_preflight_recovers_the_optimize_group(tmp_path,
                                                        monkeypatch,
                                                        capsys):
    """The other branch of the recovery-stage derivation: an optimize-only
    run dir (no datagen/pretrain records -- every arch donor-backed) recovers
    as optimize, re-submitting preflight -> train -> eval with preflight
    dependency-free, never growing a datagen or pretrain stage the run never
    had."""
    arch = "medium"  # _write_grid's single sweep arch
    donor_run = tmp_path / "pretrain_run"
    donor_dir = donor_run / "pretrain" / arch
    donor_dir.mkdir(parents=True)
    with open(donor_dir / "fidelity_certificate.json", "w") as f:
        json.dump({"verdict": "PASS", "arch": arch}, f)

    grid = _write_grid(tmp_path)
    fake = _fake_slurm(ids=["4000", "4001", "4002", "4003", "4004", "4005"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "long-40core", "--stages", "optimize",
               "--donor-run", str(donor_run), "--submit"])
    assert rc == 0
    run_dir = str(run_root / "runs" / os.listdir(run_root / "runs")[0])

    rc = main(["resubmit-preflight", run_dir, "--submit"])
    assert rc == 0

    records = jt.read_job_records(run_dir)
    assert {r["kind"] for r in records} == {"preflight", "train", "eval"}
    sbatch = [" ".join(c) for c in fake.calls
              if os.path.basename(c[0]) == "sbatch"]
    assert len(sbatch) == 6                       # 3 first submission + 3 recovery
    # The recovery re-submits preflight dependency-free (no datagen before it).
    assert "--dependency" not in sbatch[3]
    assert sbatch[3].endswith("preflight.sbatch")
    # The old train and eval arrays are cancelled (preflight is a single job
    # the recovery has never tracked for superseding).
    scancels = [c for c in fake.calls if os.path.basename(c[0]) == "scancel"]
    assert scancels == [["scancel", "4001"], ["scancel", "4002"]]
    out = capsys.readouterr().out
    assert "optimize stage group" in out
