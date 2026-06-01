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

def test_dispatch_unknown_subcommand_errors():
    with pytest.raises(SystemExit):
        main(["not-a-subcommand"])


def test_dispatch_all_subcommands_are_registered():
    parser = cli._build_parser()
    sub = [a for a in parser._subparsers._group_actions]
    choices = set()
    for action in sub:
        choices |= set(action.choices)
    assert choices == {
        "prepare", "submit", "submit-eval", "status", "results", "pull",
        "list-runs", "resubmit", "resubmit-preflight", "repair-manifest",
    }


# ===========================================================================
# prepare
# ===========================================================================

def test_prepare_refused_on_login_node(tmp_path, monkeypatch):
    """`prepare` runs the heavy CCSD precompute by default — refused on a
    login node (no $SLURM_JOB_ID)."""
    grid = _write_grid(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)  # simulate login node
    rc = main(["prepare", grid])
    assert rc == 2


def test_prepare_no_recompute_refs_runs_on_login_node(tmp_path, monkeypatch):
    """`--no-recompute-refs` skips the precompute, so `prepare` is allowed on
    a login node and calls prepare_inputs with recompute_refs=False."""
    grid = _write_grid(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    called = {}

    def _fake_prepare(cfg, *, recompute_refs=True):
        called["recompute_refs"] = recompute_refs

        class _S:
            points = [1, 2, 3]
            subset_ledger = {"l2/4": {"chosen_indices": [0], "metric_value": 0.0, "point_kinds": ["ae"], "point_names": ["H2"], "tag": "bin04"}}
        return _S()

    monkeypatch.setattr(cli, "prepare_inputs", _fake_prepare)
    rc = main(["prepare", grid, "--no-recompute-refs"])
    assert rc == 0
    assert called["recompute_refs"] is False


def test_prepare_default_recompute_refs_inside_allocation(tmp_path, monkeypatch):
    """Inside a SLURM allocation `prepare` runs the precompute (recompute_refs
    defaults True)."""
    grid = _write_grid(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "987654")  # simulate compute node
    called = {}

    def _fake_prepare(cfg, *, recompute_refs=True):
        called["recompute_refs"] = recompute_refs
        return type("S", (), {"points": [1],
                              "subset_ledger": {}})()

    monkeypatch.setattr(cli, "prepare_inputs", _fake_prepare)
    rc = main(["prepare", grid])
    assert rc == 0
    assert called["recompute_refs"] is True


def test_prepare_has_no_regenerate_flag(tmp_path):
    """The removed `--regenerate` flag must no longer be accepted."""
    grid = _write_grid(tmp_path)
    with pytest.raises(SystemExit):
        main(["prepare", grid, "--regenerate"])


# ===========================================================================
# submit
# ===========================================================================

def test_resolved_config_persists_held_out_strict(tmp_path):
    """Regression: ``held_out_strict`` must survive the resolved_config.yaml
    round trip — otherwise the cluster reloads it as False and the held-out eval
    silently stops being the strict (no-leakage) complement."""
    import dataclasses
    from xcquinox.alec.cluster.grid_config import load_grid_config

    cfg = load_grid_config(_write_grid(tmp_path))
    cfg = dataclasses.replace(cfg, held_out_strict=True)
    rd = tmp_path / "rd"
    rd.mkdir()
    cli._write_resolved_config(cfg, str(rd))
    back = load_grid_config(str(rd / "resolved_config.yaml"))
    assert back.held_out_strict is True


def test_resolved_config_persists_update_scheme_and_channel_weights(tmp_path):
    """Regression: ``hyperparams.update_scheme`` and ``channel_weights`` must
    survive the resolved_config.yaml round trip. channel_weights serializes as a
    list of [name, weight] pairs (dataclasses.asdict on the tuple) and must
    reparse back to the same sorted tuple — not silently drop to ()."""
    import dataclasses
    from xcquinox.alec.cluster.grid_config import load_grid_config

    cfg = load_grid_config(_write_grid(tmp_path))
    cfg = dataclasses.replace(
        cfg,
        hyperparams=dataclasses.replace(
            cfg.hyperparams, update_scheme="per_molecule",
            channel_weights=(("loss_AE", 1.0), ("loss_rho", 20.0)),
        ),
    )
    rd = tmp_path / "rd"
    rd.mkdir()
    cli._write_resolved_config(cfg, str(rd))
    back = load_grid_config(str(rd / "resolved_config.yaml"))
    assert back.hyperparams.update_scheme == "per_molecule"
    assert back.hyperparams.channel_weights == (
        ("loss_AE", 1.0), ("loss_rho", 20.0))


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
    fake = _fake_slurm(ids=["5000", "5001", "5002", "5003"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main(["submit", grid, "--run-root", str(run_root), "--submit",
               "--partition", "long-40core"])
    assert rc == 0
    sbatch = [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"]
    # 4-stage graph: pretrain + preflight + train + eval.
    assert len(sbatch) == 4
    # jobs.json records the pretrain stage.
    runs = os.listdir(run_root / "runs")
    run_dir = str(run_root / "runs" / runs[0])
    kinds = sorted(r["kind"] for r in jt.read_job_records(run_dir))
    assert kinds == ["eval", "preflight", "pretrain", "train"]


def _script_partition(run_dir, name):
    """Read the ``#SBATCH --partition=`` value from a rendered sbatch script."""
    path = os.path.join(run_dir, "scripts", name)
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#SBATCH --partition="):
                return stripped.split("=", 1)[1]
    raise AssertionError(f"{name} has no '#SBATCH --partition=' line")


def test_submit_requires_partition(tmp_path):
    """submit without --partition is rejected by argparse (required; no default)."""
    grid = _write_grid(tmp_path)
    run_root = tmp_path / "out"
    run_root.mkdir()
    with pytest.raises(SystemExit):
        main(["submit", grid, "--run-root", str(run_root)])


def test_submit_partition_applies_to_all_stages(tmp_path, monkeypatch):
    """A single --partition routes all four stage scripts to that queue."""
    grid = _write_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "short-28core"])
    assert rc == 0
    runs = os.listdir(run_root / "runs")
    run_dir = str(run_root / "runs" / runs[0])
    for name in ("pretrain.sbatch", "preflight.sbatch",
                 "train_array.sbatch", "eval_array.sbatch"):
        assert _script_partition(run_dir, name) == "short-28core"


def test_submit_per_stage_partition_overrides(tmp_path, monkeypatch):
    """Per-stage --*-partition flags override the base; unset ones fall back."""
    grid = _write_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main([
        "submit", grid, "--run-root", str(run_root),
        "--partition", "short-28core",
        "--train-partition", "long-28core",
        "--preflight-partition", "extended-28core",
    ])
    assert rc == 0
    runs = os.listdir(run_root / "runs")
    run_dir = str(run_root / "runs" / runs[0])
    # train + preflight overridden; eval + pretrain fall back to the base.
    assert _script_partition(run_dir, "train_array.sbatch") == "long-28core"
    assert _script_partition(run_dir, "preflight.sbatch") == "extended-28core"
    assert _script_partition(run_dir, "eval_array.sbatch") == "short-28core"
    assert _script_partition(run_dir, "pretrain.sbatch") == "short-28core"


def test_submit_per_stage_partition_persists_to_resolved_config(tmp_path,
                                                                monkeypatch):
    """Resolved partitions are baked into resolved_config.yaml so recovery
    commands (which re-render from it) inherit them with no extra flag."""
    grid = _write_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main([
        "submit", grid, "--run-root", str(run_root),
        "--partition", "short-28core",
        "--eval-partition", "long-28core",
    ])
    assert rc == 0
    runs = os.listdir(run_root / "runs")
    run_dir = run_root / "runs" / runs[0]
    from xcquinox.alec.cluster.grid_config import load_grid_config
    cfg = load_grid_config(str(run_dir / "resolved_config.yaml"))
    assert cfg.cluster.partition == "short-28core"
    assert cfg.cluster.eval_partition == "long-28core"


def _script_array(run_dir, name):
    """Read the ``#SBATCH --array=`` value from a rendered sbatch script."""
    path = os.path.join(run_dir, "scripts", name)
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#SBATCH --array="):
                return stripped.split("=", 1)[1]
    raise AssertionError(f"{name} has no '#SBATCH --array=' line")


def test_submit_max_nodes_sets_all_array_throttles(tmp_path, monkeypatch):
    """--max-nodes N sets the simultaneous-node count (array throttle) on every
    array stage — with 1 whole node per task, throttle == nodes-at-once."""
    grid = _write_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "short-28core", "--max-nodes", "3"])
    assert rc == 0
    runs = os.listdir(run_root / "runs")
    run_dir = str(run_root / "runs" / runs[0])
    assert _script_array(run_dir, "train_array.sbatch").endswith("%3")
    assert _script_array(run_dir, "eval_array.sbatch").endswith("%3")
    assert _script_array(run_dir, "pretrain.sbatch").endswith("%3")


def test_submit_per_stage_max_nodes_overrides(tmp_path, monkeypatch):
    """Per-stage --{train,eval,pretrain}-max-nodes override the base."""
    grid = _write_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "short-28core", "--max-nodes", "3",
               "--eval-max-nodes", "2", "--pretrain-max-nodes", "5"])
    assert rc == 0
    runs = os.listdir(run_root / "runs")
    run_dir = str(run_root / "runs" / runs[0])
    assert _script_array(run_dir, "train_array.sbatch").endswith("%3")
    assert _script_array(run_dir, "eval_array.sbatch").endswith("%2")
    assert _script_array(run_dir, "pretrain.sbatch").endswith("%5")


def test_submit_without_max_nodes_keeps_config_throttle(tmp_path, monkeypatch):
    """Omitting --max-nodes leaves the config's array throttle untouched."""
    grid = _write_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()

    # _base_config_dict sets array_throttle=4 (the cli-test default).
    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "short-28core"])
    assert rc == 0
    runs = os.listdir(run_root / "runs")
    run_dir = str(run_root / "runs" / runs[0])
    assert _script_array(run_dir, "train_array.sbatch").endswith("%4")


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


def test_retry_resource_flags_oom_force_cpu():
    """oom_retry_force_cpu adds the GPU-release + JAX-cpu flags so the retry
    actually runs on the CPU instead of re-OOMing on the GPU (CW2-M1)."""
    from types import SimpleNamespace

    def _cl(**kw):
        base = dict(oom_retry_partition=None, oom_retry_mem=None,
                    oom_retry_force_cpu=False, timeout_retry_partition=None,
                    timeout_retry_time=None)
        base.update(kw)
        return SimpleNamespace(**base)

    # Default (force_cpu False): no CPU-route flags.
    flags = cli._retry_resource_flags(
        "oom", _cl(oom_retry_partition="hbm-96core", oom_retry_mem="512G"))
    assert "--partition=hbm-96core" in flags and "--mem=512G" in flags
    assert not any("gpu:0" in f for f in flags)
    assert not any("JAX_PLATFORMS" in f for f in flags)

    # force_cpu True: GPU released + JAX pinned to CPU.
    flags_cpu = cli._retry_resource_flags(
        "oom", _cl(oom_retry_mem="512G", oom_retry_force_cpu=True))
    assert "--gres=gpu:0" in flags_cpu
    assert "--export=ALL,JAX_PLATFORMS=cpu" in flags_cpu

    # timeout class is unaffected by the cpu knob.
    assert cli._retry_resource_flags(
        "timeout", _cl(timeout_retry_time="24:00:00",
                       oom_retry_force_cpu=True)) == ["--time=24:00:00"]


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
    """Genuinely pretrain/preflight-stuck run: no manifest, no train evidence.
    Re-submits then scancels + marks the old arrays superseded."""
    run_dir = _make_run_dir(tmp_path, manifest=False)
    # Old pretrain/train/eval arrays recorded from the failed first submit.
    jt.append_job_record(run_dir, "pretrain", "50", [0])
    jt.append_job_record(run_dir, "preflight", "100", [0])
    jt.append_job_record(run_dir, "train", "200", list(range(_N)))
    jt.append_job_record(run_dir, "eval", "300", list(range(_N)))

    fake = _fake_slurm(ids=["400", "401", "402", "403"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit-preflight", run_dir, "--submit"])
    assert rc == 0
    # Four new sbatch calls + three scancels (old pretrain + train + eval).
    sbatch = [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"]
    scancel = [c for c in fake.calls if os.path.basename(c[0]) == "scancel"]
    assert len(sbatch) == 4
    assert sorted(c[1] for c in scancel) == ["200", "300", "50"]
    # Old pretrain/train/eval gen-0 records are now superseded.
    records = jt.read_job_records(run_dir)
    old = [r for r in records
           if r["array_job_id"] in ("50", "200", "300")]
    assert all(r["superseded"] for r in old)


def test_resubmit_preflight_scancel_failure_skips_supersede(tmp_path,
                                                            monkeypatch):
    run_dir = _make_run_dir(tmp_path, manifest=False)
    jt.append_job_record(run_dir, "train", "200", list(range(_N)))
    jt.append_job_record(run_dir, "eval", "300", list(range(_N)))

    # All four new sbatch succeed; scancel fails.
    fake = _fake_slurm(ids=["400", "401", "402", "403"], fail_scancel=True)
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


def test_status_pretrain_checkpoint_uses_run_scoped_path(tmp_path):
    """status' pretrain-presence check looks at the RUN-SCOPED
    <run_dir>/pretrain/<arch> (where the pretrain worker now writes)."""
    from xcquinox.alec.cluster.grid_config import (
        load_grid_config, pretrain_checkpoint_dir,
    )
    run_dir = tmp_path / "run_TESTID"
    run_dir.mkdir()

    d = _base_config_dict()
    gp = tmp_path / "_g.json"
    gp.write_text(json.dumps(d))
    cfg = load_grid_config(str(gp))
    cli._write_resolved_config(cfg, str(run_dir))

    arch = sorted(set(cfg.sweep.arch))[0]
    ck = pretrain_checkpoint_dir(str(run_dir), arch)
    os.makedirs(ck, exist_ok=True)
    open(os.path.join(ck, "xnet.eqx"), "wb").close()
    open(os.path.join(ck, "cnet.eqx"), "wb").close()

    # The checkpoint lives under the run dir, co-located with its other artifacts.
    assert ck == os.path.join(os.path.abspath(str(run_dir)), "pretrain", arch)

    line = cli._pretrain_status(str(run_dir))
    assert line == "1/1 architecture checkpoint pair(s) present"


def test_classify_failure_treats_killed_by_signal_as_timeout(tmp_path):
    """A wall-clock grace SIGTERM (recorded by _train_task as
    'killed_by_signal') must classify as 'timeout' — i.e. retryable and routed
    to the timeout_retry resources — not 'deterministic' (which resubmit skips)."""
    run_dir = _make_run_dir(tmp_path)
    sd = _spec_dir(run_dir, 3)
    with open(os.path.join(sd, "failure.json"), "w") as f:
        json.dump({"classification": "killed_by_signal", "rc": 143}, f)
    assert cli._classify_failure(run_dir, 3, _WIDTH, {}) == "timeout"


def test_classify_failure_deterministic_still_skipped(tmp_path):
    """A genuine code error stays 'deterministic' (not retried)."""
    run_dir = _make_run_dir(tmp_path)
    sd = _spec_dir(run_dir, 3)
    with open(os.path.join(sd, "failure.json"), "w") as f:
        json.dump({"classification": "assertion_error"}, f)
    assert cli._classify_failure(run_dir, 3, _WIDTH, {}) == "deterministic"


def test_resubmit_timeout_applies_timeout_retry_resources(tmp_path, monkeypatch):
    """A timeout failure must be resubmitted with the timeout_retry partition +
    time applied as sbatch overrides (previously the knobs were dead, so a
    timeout retried on the same wall and timed out forever). The eval array
    must NOT inherit the train-only retry overrides."""
    from xcquinox.alec.cluster.grid_config import load_grid_config
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    d = _base_config_dict()
    d["cluster"]["timeout_retry_partition"] = "long-28core"
    d["cluster"]["timeout_retry_time"] = "12:00:00"
    cp = tmp_path / "_cfg_to.json"
    cp.write_text(json.dumps(d))
    cli._write_resolved_config(load_grid_config(str(cp)), rd)

    open(os.path.join(_spec_dir(rd, 0), "model.eqx"), "wb").close()
    for i in (2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    # index 1: wall-grace SIGTERM (recorded as killed_by_signal -> timeout).
    with open(os.path.join(_spec_dir(rd, 1), "failure.json"), "w") as f:
        json.dump({"classification": "killed_by_signal"}, f)

    fake = _fake_slurm(ids=["7700", "7701"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd, "--submit"])
    assert rc == 0
    sbatch = [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"]
    assert len(sbatch) == 2
    train_cmd, eval_cmd = sbatch[0], sbatch[1]
    # train carries the timeout_retry overrides:
    assert "--partition=long-28core" in train_cmd
    assert "--time=12:00:00" in train_cmd
    # eval does NOT get the train-only wall override:
    assert not any(t.startswith("--time=") for t in eval_cmd)


def test_resubmit_oom_applies_oom_retry_resources(tmp_path, monkeypatch):
    """An oom failure is resubmitted with oom_retry partition + mem overrides."""
    from xcquinox.alec.cluster.grid_config import load_grid_config
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    d = _base_config_dict()
    d["cluster"]["oom_retry_partition"] = "hbm-long-96core"
    d["cluster"]["oom_retry_mem"] = "512G"
    cp = tmp_path / "_cfg_oom.json"
    cp.write_text(json.dumps(d))
    cli._write_resolved_config(load_grid_config(str(cp)), rd)

    open(os.path.join(_spec_dir(rd, 0), "model.eqx"), "wb").close()
    for i in (2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    with open(os.path.join(_spec_dir(rd, 1), "failure.json"), "w") as f:
        json.dump({"classification": "oom"}, f)

    fake = _fake_slurm(ids=["8800", "8801"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd, "--submit"])
    assert rc == 0
    sbatch = [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"]
    train_cmd = sbatch[0]
    assert "--partition=hbm-long-96core" in train_cmd
    assert "--mem=512G" in train_cmd


def _script_time(run_dir, name):
    """Read the ``#SBATCH --time=`` value from a rendered sbatch script."""
    path = os.path.join(run_dir, "scripts", name)
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("#SBATCH --time="):
                return s.split("=", 1)[1]
    raise AssertionError(f"{name} has no '#SBATCH --time=' line")


def test_submit_time_applies_to_all_stages(tmp_path, monkeypatch):
    """--time sets the wall for every stage (base, like --partition)."""
    grid = _write_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()
    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "short-28core", "--time", "06:00:00"])
    assert rc == 0
    run_dir = str(run_root / "runs" / os.listdir(run_root / "runs")[0])
    for name in ("pretrain.sbatch", "preflight.sbatch",
                 "train_array.sbatch", "eval_array.sbatch"):
        assert _script_time(run_dir, name) == "06:00:00"


def test_submit_per_stage_time_overrides(tmp_path, monkeypatch):
    """Per-stage --{train,preflight,...}-time override the base --time."""
    grid = _write_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()
    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "short-28core", "--time", "02:00:00",
               "--preflight-time", "08:00:00", "--pretrain-time", "08:00:00"])
    assert rc == 0
    run_dir = str(run_root / "runs" / os.listdir(run_root / "runs")[0])
    assert _script_time(run_dir, "train_array.sbatch") == "02:00:00"
    assert _script_time(run_dir, "eval_array.sbatch") == "02:00:00"
    assert _script_time(run_dir, "preflight.sbatch") == "08:00:00"
    assert _script_time(run_dir, "pretrain.sbatch") == "08:00:00"


def test_submit_without_time_keeps_config(tmp_path, monkeypatch):
    """Omitting --time leaves the config's per-stage walls untouched."""
    grid = _write_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()
    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "short-28core"])
    assert rc == 0
    run_dir = str(run_root / "runs" / os.listdir(run_root / "runs")[0])
    # _base_config_dict sets train time "12:00:00".
    assert _script_time(run_dir, "train_array.sbatch") == "12:00:00"


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


def test_results_subcommand_missing_manifest_is_graceful(tmp_path):
    """`results` on a run dir with no manifest returns non-zero (not a crash)."""
    run_dir = tmp_path / "bare"
    run_dir.mkdir()
    rc = main(["results", str(run_dir)])
    assert rc == 1


def test_results_spec_prints_per_molecule(tmp_path):
    """`results <run_dir> --spec <idx>` prints the per-molecule AE table."""
    run_dir = _make_run_dir(tmp_path)
    d = _spec_dir(run_dir, 0)
    open(os.path.join(d, "model.eqx"), "wb").close()
    ed = os.path.join(d, "eval")
    os.makedirs(ed, exist_ok=True)
    with open(os.path.join(ed, "per_molecule.json"), "w") as f:
        json.dump([
            {"molecule": "F2O", "AE_nn": -0.1, "AE_ref_kcalmol": 53.7,
             "AE_error_kcalmol": -141.6, "density_rmse": 0.002},
            {"molecule": "H2O", "AE_nn": 0.37, "AE_ref_kcalmol": 232.2,
             "AE_error_kcalmol": -0.1, "density_rmse": 0.001},
        ], f)
    rc = main(["results", run_dir, "--spec", "0"])
    assert rc == 0


def test_results_spec_without_per_molecule_is_graceful(tmp_path):
    """--spec on a spec with no per_molecule.json returns non-zero (no crash)."""
    run_dir = _make_run_dir(tmp_path)
    rc = main(["results", run_dir, "--spec", "5"])
    assert rc == 1


def test_results_worst_prints_ranked_table(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    ed = os.path.join(_spec_dir(run_dir, 0), "eval")
    os.makedirs(ed, exist_ok=True)
    with open(os.path.join(ed, "per_molecule.json"), "w") as f:
        json.dump([{"molecule": "F2O", "AE_nn": -0.1,
                    "AE_error_kcalmol": -141.6}], f)
    rc = main(["results", run_dir, "--worst", "5"])
    assert rc == 0


def test_submit_n_steps_override_rides_into_resolved_config(tmp_path, monkeypatch):
    """--n-steps overrides the training-optimization step count; it is baked
    into resolved_config.yaml so the preflight materializes specs with it."""
    grid = _write_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()
    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "long-28core", "--n-steps", "250"])
    assert rc == 0
    runs = os.listdir(run_root / "runs")
    run_dir = run_root / "runs" / runs[0]
    from xcquinox.alec.cluster.grid_config import load_grid_config
    cfg = load_grid_config(str(run_dir / "resolved_config.yaml"))
    assert cfg.hyperparams.n_steps == 250


def test_submit_without_n_steps_keeps_config(tmp_path, monkeypatch):
    grid = _write_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()
    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "long-28core"])
    assert rc == 0
    runs = os.listdir(run_root / "runs")
    run_dir = run_root / "runs" / runs[0]
    from xcquinox.alec.cluster.grid_config import load_grid_config
    cfg = load_grid_config(str(run_dir / "resolved_config.yaml"))
    assert cfg.hyperparams.n_steps == 200   # _base_config_dict default


def test_submit_pretrain_n_steps_override(tmp_path, monkeypatch):
    """--pretrain-n-steps overrides pretrain.n_steps into resolved_config."""
    grid = _write_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()
    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "long-28core",
               "--n-steps", "250", "--pretrain-n-steps", "2000"])
    assert rc == 0
    runs = os.listdir(run_root / "runs")
    run_dir = run_root / "runs" / runs[0]
    from xcquinox.alec.cluster.grid_config import load_grid_config
    cfg = load_grid_config(str(run_dir / "resolved_config.yaml"))
    assert cfg.hyperparams.n_steps == 250
    assert cfg.pretrain.n_steps == 2000


def test_submit_polarized_flag_and_override():
    """`submit --polarized` parses to args.polarized and the override helper
    flips cfg.use_polarized_correlation (default off)."""
    import types
    parser = cli._build_parser()
    assert parser.parse_args(
        ["submit", "g.yaml", "--partition", "short", "--polarized"]).polarized is True
    assert parser.parse_args(
        ["submit", "g.yaml", "--partition", "short"]).polarized is False
    cfg = cli.load_grid_config("xcquinox/alec/cluster/examples/grid_step7.yaml")
    assert cfg.use_polarized_correlation is False
    on = cli._apply_polarized_override(cfg, types.SimpleNamespace(polarized=True))
    assert on.use_polarized_correlation is True
    off = cli._apply_polarized_override(cfg, types.SimpleNamespace(polarized=False))
    assert off.use_polarized_correlation is False


def test_submit_defer_eval_flag_override_and_roundtrip():
    """`submit --defer-eval` parses to args.defer_eval, the override helper flips
    cfg.defer_eval (default off), and the value round-trips through
    _config_to_raw_dict -> load_grid_config."""
    import types
    parser = cli._build_parser()
    assert parser.parse_args(
        ["submit", "g.yaml", "--partition", "short", "--defer-eval"]
    ).defer_eval is True
    assert parser.parse_args(
        ["submit", "g.yaml", "--partition", "short"]).defer_eval is False
    cfg = cli.load_grid_config("xcquinox/alec/cluster/examples/grid_step7.yaml")
    assert cfg.defer_eval is False
    on = cli._apply_defer_eval_override(cfg, types.SimpleNamespace(defer_eval=True))
    assert on.defer_eval is True
    off = cli._apply_defer_eval_override(cfg, types.SimpleNamespace(defer_eval=False))
    assert off.defer_eval is False
    # Round-trip: serialized "defer_eval" survives a re-parse.
    assert cli._config_to_raw_dict(on)["defer_eval"] is True


def test_submit_eval_subcommand_parses():
    """The `submit-eval` subcommand parses run_dir + --force and binds the
    cmd_submit_eval handler."""
    parser = cli._build_parser()
    ns = parser.parse_args(["submit-eval", "/some/run_dir"])
    assert ns.run_dir == "/some/run_dir"
    assert ns.force is False
    assert ns.func is cli.cmd_submit_eval
    assert parser.parse_args(
        ["submit-eval", "/some/run_dir", "--force"]).force is True
