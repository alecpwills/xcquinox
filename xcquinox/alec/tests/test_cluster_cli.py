"""Tests for xcquinox.alec.cluster.__main__: the harness CLI.

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
    round trip, otherwise the cluster reloads it as False and the held-out eval
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


def test_resolved_config_persists_inline_eval(tmp_path):
    """Regression: ``inline_eval`` must survive the resolved_config.yaml round
    trip: otherwise a recovery/resubmit reloads it as False and the run
    silently reverts from inline eval to a separate eval array."""
    import dataclasses
    from xcquinox.alec.cluster.grid_config import load_grid_config

    cfg = load_grid_config(_write_grid(tmp_path))
    cfg = dataclasses.replace(cfg, inline_eval=True)
    rd = tmp_path / "rd"
    rd.mkdir()
    cli._write_resolved_config(cfg, str(rd))
    back = load_grid_config(str(rd / "resolved_config.yaml"))
    assert back.inline_eval is True


def test_resolved_config_persists_datagen_resources(tmp_path):
    """The datagen-stage cluster knobs survive the resolved_config.yaml round
    trip (so a re-submit reproduces the same datagen resources)."""
    import dataclasses
    from xcquinox.alec.cluster.grid_config import load_grid_config

    cfg = load_grid_config(_write_grid(tmp_path))
    cfg = dataclasses.replace(
        cfg, cluster=dataclasses.replace(
            cfg.cluster, datagen_time="02:00:00",
            datagen_cpus_per_task=8, datagen_allocation="shared"),
    )
    rd = tmp_path / "rd"
    rd.mkdir()
    cli._write_resolved_config(cfg, str(rd))
    back = load_grid_config(str(rd / "resolved_config.yaml"))
    assert back.cluster.datagen_time == "02:00:00"
    assert back.cluster.datagen_cpus_per_task == 8
    assert back.cluster.datagen_allocation == "shared"


def test_resolved_config_persists_update_scheme_and_channel_weights(tmp_path):
    """Regression: ``hyperparams.update_scheme`` and ``channel_weights`` must
    survive the resolved_config.yaml round trip. channel_weights serializes as a
    list of [name, weight] pairs (dataclasses.asdict on the tuple) and must
    reparse back to the same sorted tuple, not silently drop to ()."""
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
    array stage, with 1 whole node per task, throttle == nodes-at-once."""
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
    # status is read-only, it must NOT take the lock.
    assert not os.path.exists(os.path.join(run_dir, ".harness.lock"))


def test_status_handles_slurm_transient_error(tmp_path, monkeypatch):
    run_dir = _make_run_dir(tmp_path)
    jt.append_job_record(run_dir, "train", "1000", list(range(_N)))
    monkeypatch.setattr(jt, "_run_slurm", _fake_slurm(transient=True))
    # Must not crash, reports + returns non-zero.
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


def test_resubmit_skips_live_indices_and_retries_the_rest(tmp_path,
                                                          monkeypatch):
    """A live index must never be resubmitted (double-writes its checkpoint
    dir beside the scheduler's copy) -- but a dead index beside a draining
    array stays recoverable: live indices are dropped from the plan per
    index and the retryable remainder is submitted. Liveness is judged on
    the RAW sacct states, so a RUNNING index with disk evidence (the
    resume-class case: resume_state.pkl written by the live task itself) is
    skipped too, not resumed on top of itself."""
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    open(os.path.join(_spec_dir(rd, 0), "model.eqx"), "wb").close()
    # Index 4: RUNNING in the queue AND carrying a resume checkpoint --
    # disk reduces it to incomplete_resumable, the raw queue says live.
    open(os.path.join(_spec_dir(rd, 4), "resume_state.pkl"), "wb").close()
    # Index 5: deterministic failure (never retried).
    with open(os.path.join(_spec_dir(rd, 5), "failure.json"), "w") as f:
        json.dump({"classification": "value_error"}, f)
    train_rows = "\n".join([
        "1000_1|OUT_OF_MEMORY|0:125",   # dead: retryable now
        "1000_2|CONFIGURING|0:0",       # live transient (nodes booting)
        "1000_[3-3%2]|PENDING|0:0",     # live behind a throttle
        "1000_4|RUNNING|0:0",           # live WITH disk evidence
    ])
    fake = _fake_slurm(ids=["7001", "7002", "7003", "7004"],
                       sacct_rows={"1000": train_rows})
    monkeypatch.setattr(jt, "_run_slurm", fake)

    rc = main(["resubmit", rd, "--submit"])
    assert rc == 0
    sbatch = [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"]
    assert len(sbatch) == 2  # sparse train + sparse eval for index 1 only

    def _arr(cmd):
        for tok in cmd:
            if tok.startswith("--array="):
                return tok.split("=", 1)[1].split("%", 1)[0]
        raise AssertionError("no --array")
    assert _arr(sbatch[0]) == _arr(sbatch[1]) == "1", (
        "only the dead index may be retried; live 2/3/4 must be skipped")
    attempts = json.load(open(os.path.join(rd, "attempts.json")))
    assert attempts == {"1": 1}


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
    # index 1: OOM with a stale failure.json: must be archived to *.gen0.
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
    """A complete manifest means the preflight succeeded, use resubmit."""
    run_dir = _make_run_dir(tmp_path, manifest=True, n=_N)
    rc = main(["resubmit-preflight", run_dir])
    assert rc == 1


def test_resubmit_preflight_resubmits_and_supersedes(tmp_path, monkeypatch):
    """Genuinely pretrain/preflight-stuck run: no manifest, no train evidence.
    Re-submits then scancels + marks the old arrays superseded."""
    run_dir = _make_run_dir(tmp_path, manifest=False)
    # Old datagen/pretrain/train/eval arrays recorded from the failed first submit.
    jt.append_job_record(run_dir, "datagen", "40", [0])
    jt.append_job_record(run_dir, "pretrain", "50", [0])
    jt.append_job_record(run_dir, "preflight", "100", [0])
    jt.append_job_record(run_dir, "train", "200", list(range(_N)))
    jt.append_job_record(run_dir, "eval", "300", list(range(_N)))

    fake = _fake_slurm(ids=["400", "401", "402", "403", "404"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit-preflight", run_dir, "--submit"])
    assert rc == 0
    # Five new sbatch calls (datagen+pretrain+preflight+train+eval) + four
    # scancels (old datagen + pretrain + train + eval).
    sbatch = [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"]
    scancel = [c for c in fake.calls if os.path.basename(c[0]) == "scancel"]
    assert len(sbatch) == 5
    assert sorted(c[1] for c in scancel) == ["200", "300", "40", "50"]
    # Old datagen/pretrain/train/eval gen-0 records are now superseded.
    records = jt.read_job_records(run_dir)
    old = [r for r in records
           if r["array_job_id"] in ("40", "50", "200", "300")]
    assert all(r["superseded"] for r in old)


def test_resubmit_preflight_scancel_failure_skips_supersede(tmp_path,
                                                            monkeypatch):
    run_dir = _make_run_dir(tmp_path, manifest=False)
    jt.append_job_record(run_dir, "train", "200", list(range(_N)))
    jt.append_job_record(run_dir, "eval", "300", list(range(_N)))

    # All five new sbatch succeed; scancel fails.
    fake = _fake_slurm(ids=["400", "401", "402", "403", "404"], fail_scancel=True)
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit-preflight", run_dir, "--submit"])
    assert rc == 1
    # mark_superseded was SKIPPED, old records remain un-superseded.
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
# .harness.lock: stale-lock reclaim
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
    assert line == ("1/1 architecture checkpoint pair(s) present, "
                    "0/1 architecture certificate(s) PASS")


def test_pretrain_status_counts_passing_certificates(tmp_path):
    """A checkpoint pair on disk is not the same as a certified architecture:
    `status` must show both counts so an operator can see the pretrain array
    finished but the physics gate did not."""
    import json
    from xcquinox.alec.cluster.grid_config import (
        load_grid_config, pretrain_checkpoint_dir,
    )
    run_dir = tmp_path / "run_TESTID"
    run_dir.mkdir()
    d = _base_config_dict()
    d["sweep"]["arch"] = ["medium", "shallow"]
    gp = tmp_path / "_g.json"
    gp.write_text(json.dumps(d))
    cfg = load_grid_config(str(gp))
    cli._write_resolved_config(cfg, str(run_dir))

    for arch in sorted(set(cfg.sweep.arch)):
        ck = pretrain_checkpoint_dir(str(run_dir), arch)
        os.makedirs(ck, exist_ok=True)
        open(os.path.join(ck, "xnet.eqx"), "wb").close()
        open(os.path.join(ck, "cnet.eqx"), "wb").close()
    # Only one of the two certified.
    ck = pretrain_checkpoint_dir(str(run_dir), "medium")
    with open(os.path.join(ck, "fidelity_certificate.json"), "w") as f:
        json.dump({"verdict": "PASS", "arch": "medium"}, f)

    assert cli._pretrain_status(str(run_dir)) == (
        "2/2 architecture checkpoint pair(s) present, "
        "1/2 architecture certificate(s) PASS")


def test_pretrain_status_counts_only_pass_certificates(tmp_path):
    """The certificate count is the RECORD predicate, not the on-node gate.

    Five architectures cover the five states a certificate can be found in:
    PASS, a FAIL waived by ``enforced: false`` with a reason, an enforced
    FAIL, no certificate at all, and one that is not readable JSON. Only the
    first is certified. A run configured ``fidelity.enforce: false`` reaches
    training on its waived FAIL, but that architecture still cannot enter
    ``validate_run``, ``merge_v4_arms`` or the figure suite, so counting it
    as PASS here would hide the very state this line exists to show -- the
    count must therefore come from ``certificate_status_in`` and not from
    ``gate_certificate``.

    The checkpoint-pair count is asserted independently: it is a separate
    on-disk fact, and the pretrain array can leave a pair with no certificate
    (or a certificate with no pair).
    """
    from xcquinox.alec.cluster.grid_config import (
        load_grid_config, pretrain_checkpoint_dir,
    )
    run_dir = tmp_path / "run_TESTID"
    run_dir.mkdir()
    d = _base_config_dict()
    d["sweep"]["arch"] = ["deep", "deep_attn", "deep_cusp", "medium",
                          "shallow"]
    gp = tmp_path / "_g.json"
    gp.write_text(json.dumps(d))
    cfg = load_grid_config(str(gp))
    cli._write_resolved_config(cfg, str(run_dir))

    def _place(arch, *, pair=True, certificate=None, raw=None):
        ck = pretrain_checkpoint_dir(str(run_dir), arch)
        os.makedirs(ck, exist_ok=True)
        if pair:
            open(os.path.join(ck, "xnet.eqx"), "wb").close()
            open(os.path.join(ck, "cnet.eqx"), "wb").close()
        path = os.path.join(ck, "fidelity_certificate.json")
        if certificate is not None:
            with open(path, "w") as f:
                json.dump(certificate, f)
        elif raw is not None:
            with open(path, "w") as f:
                f.write(raw)

    _place("deep", certificate={"verdict": "PASS", "arch": "deep"})
    # Waived FAIL: the workflow-verification matrix's own record. It is
    # released by the on-node gate and is NOT a certified architecture.
    _place("deep_attn", certificate={
        "verdict": "FAIL", "arch": "deep_attn", "enforced": False,
        "tolerances": {"override_reason": "workflow-verification matrix"},
        "summary": {"max_atom_mHa": 13.7, "max_dAE_kcalmol": 25.7}})
    _place("deep_cusp", certificate={
        "verdict": "FAIL", "arch": "deep_cusp",
        "summary": {"max_atom_mHa": 13.7, "max_dAE_kcalmol": 25.7}})
    _place("medium")
    # Truncated file, and its pretrain job left no checkpoint pair either.
    _place("shallow", pair=False, raw='{"verdict": "PA')

    assert cli._pretrain_status(str(run_dir)) == (
        "4/5 architecture checkpoint pair(s) present, "
        "1/5 architecture certificate(s) PASS")


def test_classify_failure_treats_killed_by_signal_as_timeout(tmp_path):
    """A wall-clock grace SIGTERM (recorded by _train_task as
    'killed_by_signal') must classify as 'timeout', i.e. retryable and routed
    to the timeout_retry resources, not 'deterministic' (which resubmit skips)."""
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


def test_classify_failure_no_disk_evidence_is_recoverable(tmp_path):
    """A train index with NO disk trace (no model.eqx / resume_state.pkl /
    failure.json) whose sacct state is a no-evidence catch-all
    (dependency_never_satisfied / unknown_sacct_purged) classifies as
    'no_evidence' -- a bounded FRESH relaunch, NOT stranded as 'deterministic'.
    Regression for the NODE_FAIL / cancel of a materialized train task whose
    preflight succeeded (previously unrecoverable by resubmit + resubmit-preflight)."""
    run_dir = _make_run_dir(tmp_path)
    _spec_dir(run_dir, 3)  # spec dir exists; the task left no artifacts in it
    for outcome in ("dependency_never_satisfied", "unknown_sacct_purged"):
        cls = cli._classify_failure(run_dir, 3, _WIDTH, {3: outcome})
        assert cls == "no_evidence", (outcome, cls)
    # 'no_evidence' must be retryable (resubmit does not skip it as deterministic)
    # and fresh (nothing to archive -- the task never ran).
    assert "no_evidence" in cli._RETRYABLE
    assert "no_evidence" in cli._FRESH_RETRY


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


# ---------------------------------------------------------------------------
# --time overrides: the same walltime rule as the config fields they replace
# ---------------------------------------------------------------------------

#: Every stage script a ``--time`` base override reaches.
_TIME_SCRIPTS = ("pretrain.sbatch", "preflight.sbatch",
                 "train_array.sbatch", "eval_array.sbatch")


@pytest.mark.parametrize("flag,value", [
    ("--time", "30"),            # SLURM minutes; the flag documents HH:MM:SS
    ("--time", "30:00"),         # minutes:seconds
    ("--time", "1-12"),          # days-hours
    ("--time", "1-12:00"),       # days-hours:minutes
    ("--time", "8:60:00"),       # 60 minutes is out of range
    ("--time", "8h"),
    ("--time", "later"),
    ("--train-time", "30"),
    ("--eval-time", "45:00"),
    ("--preflight-time", "0"),
    ("--pretrain-time", "8h"),
])
def test_submit_refuses_a_bad_time_override(tmp_path, monkeypatch, flag, value):
    """A CLI wall is checked exactly as the config field it overrides.

    ``_apply_time_overrides`` writes the override onto ``cfg.cluster`` by
    ``dataclasses.replace``, which does not pass through the loader, so an
    unusable wall would otherwise reach ``#SBATCH --time=`` and be caught only
    when a later stage re-reads ``resolved_config.yaml``. The refusal names the
    flag and lands before the run directory exists, so a rejected submission
    leaves nothing behind.
    """
    grid = _write_grid(tmp_path)
    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    run_root = tmp_path / "out"
    run_root.mkdir()
    with pytest.raises(ValueError, match=flag):
        main(["submit", grid, "--run-root", str(run_root),
              "--partition", "short-28core", flag, value])
    assert not (run_root / "runs").exists(), "a refused submit left a run dir"
    assert not [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"]


@pytest.mark.parametrize("value", ["8:00:00", "1-12:00:00", "00:30:00"])
def test_submit_accepts_the_two_walltime_shapes(tmp_path, monkeypatch, value):
    """``H:MM:SS`` and ``D-HH:MM:SS`` pass through to every stage script."""
    grid = _write_grid(tmp_path)
    monkeypatch.setattr(jt, "_run_slurm", _fake_slurm())
    run_root = tmp_path / "out"
    run_root.mkdir()
    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "short-28core", "--time", value])
    assert rc == 0
    run_dir = str(run_root / "runs" / os.listdir(run_root / "runs")[0])
    for name in _TIME_SCRIPTS:
        assert _script_time(run_dir, name) == value


def test_submit_time_override_round_trips_through_resolved_config(
        tmp_path, monkeypatch):
    """The resolved wall survives ``resolved_config.yaml``, which the recovery
    subcommands re-render from. ``8:00:00`` is the case that matters: written
    unquoted it would re-read as the integer 28800."""
    from xcquinox.alec.cluster.grid_config import load_grid_config
    grid = _write_grid(tmp_path)
    monkeypatch.setattr(jt, "_run_slurm", _fake_slurm())
    run_root = tmp_path / "out"
    run_root.mkdir()
    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "short-28core", "--time", "8:00:00"])
    assert rc == 0
    run_dir = str(run_root / "runs" / os.listdir(run_root / "runs")[0])
    reloaded = load_grid_config(os.path.join(run_dir, "resolved_config.yaml"))
    assert reloaded.cluster.time == "8:00:00"
    assert reloaded.cluster.eval_time == "8:00:00"


def test_submit_time_override_is_stripped(tmp_path, monkeypatch):
    """Surrounding whitespace would render ``#SBATCH --time= 8:00:00``."""
    grid = _write_grid(tmp_path)
    monkeypatch.setattr(jt, "_run_slurm", _fake_slurm())
    run_root = tmp_path / "out"
    run_root.mkdir()
    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "short-28core", "--time", "  8:00:00  "])
    assert rc == 0
    run_dir = str(run_root / "runs" / os.listdir(run_root / "runs")[0])
    assert _script_time(run_dir, "train_array.sbatch") == "8:00:00"


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


def test_the_model_block_round_trips_through_the_resolved_config():
    """The v6 model class survives ``submit``'s serialization.

    Every stage rebuilds its architectures from ``resolved_config.yaml``, so a
    block missing from ``_config_to_raw_dict`` resolves the run to the field
    defaults whatever the source YAML states. The first v6 group ran that way
    on 2026-08-27: a source file stating ``parent_anchor: true`` produced four
    pretrained architectures whose metadata and certificate both recorded
    ``parent_anchor: false``, refused at 0.8 to 3.0 mHa against the parent.
    """
    import yaml
    cfg = cli.load_grid_config(
        "hpcjobs/configs/dfs_step7.dfs6311_grid3_v6g1_size.yaml")
    assert cfg.model.parent_anchor is True
    assert cfg.model.descriptor_coordinates == "dfs"
    raw = cli._config_to_raw_dict(cfg)
    assert raw["model"] == {"parent_anchor": True,
                            "descriptor_coordinates": "dfs"}, raw.get("model")
    back = cli.load_grid_config_from_raw(yaml.safe_load(yaml.safe_dump(raw))) \
        if hasattr(cli, "load_grid_config_from_raw") else None
    if back is not None:
        assert back.model == cfg.model


def test_every_grid_config_block_round_trips(tmp_path):
    """``_config_to_raw_dict`` carries EVERY field of the loaded configuration.

    The serializer is written key by key, so a block added to ``GridConfig``
    after it is silently dropped -- the failure mode that reverted
    ``ae_as_reactions`` in 2026-08 and the model class in 2026-08-27. Rather
    than pin one key per incident, the whole dataclass is compared after a
    write-and-reload of a configuration whose blocks are all non-default.
    """
    import dataclasses
    import yaml
    cfg = cli.load_grid_config(
        "hpcjobs/configs/dfs_step7.dfs6311_grid3_v6g1_size.yaml")
    raw = cli._config_to_raw_dict(cfg)
    path = tmp_path / "resolved_config.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(raw, fh)
    back = cli.load_grid_config(str(path))
    for field in dataclasses.fields(cfg):
        name = field.name
        assert getattr(back, name) == getattr(cfg, name), (
            f"{name} did not survive the resolved-config round trip: "
            f"{getattr(cfg, name)!r} -> {getattr(back, name)!r}")


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


def test_resubmit_oom_force_cpu_not_flagged_default_resources(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    """An oom retry configured with cluster.oom_retry_force_cpu (and no
    partition/mem override) must NOT print the 'use DEFAULT partition/resources'
    notice: force_cpu IS a dedicated retry knob, so the retry is not running on
    default resources."""
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    # Re-render the resolved config with cluster.oom_retry_force_cpu = True.
    from xcquinox.alec.cluster.grid_config import load_grid_config
    data = _base_config_dict()
    data["cluster"]["oom_retry_force_cpu"] = True
    p = os.path.join(rd, "_force_cpu.json")
    with open(p, "w") as f:
        json.dump(data, f)
    cfg = load_grid_config(p)
    os.unlink(p)
    cli._write_resolved_config(cfg, rd)

    # All indices succeed EXCEPT index 1, which OOMs (via sacct) -> retried.
    for i in (0, 2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    fake = _fake_slurm(ids=["7001", "7002"],
                       sacct_rows={"1000": "1000_1|OUT_OF_MEMORY|0:125"})
    monkeypatch.setattr(jt, "_run_slurm", fake)

    rc = main(["resubmit", rd, "--submit"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "retry=[1]" in out                                 # oom IS retried
    assert "use DEFAULT partition/resources" not in out, (    # but not "default"
        "force_cpu retry wrongly flagged as default-resources")


# ===========================================================================
# WS6: incomplete_resumable -> RESUME path (resubmit) + status tally
# ===========================================================================

def _write_resume_state(run_dir, idx, width=_WIDTH):
    """Write a WS5 mid-run ``resume_state.pkl`` marker (presence is the signal)."""
    open(os.path.join(_spec_dir(run_dir, idx, width), "resume_state.pkl"),
         "wb").close()


def test_classify_failure_incomplete_resumable_is_resume(tmp_path):
    """WS6: a killed mid-run index (resume_state.pkl, no model.eqx/completion)
    classifies as 'resume' -- retryable, but routed to the RESUME path (continue
    from the checkpoint), distinct from oom/timeout fresh-retry."""
    run_dir = _make_run_dir(tmp_path)
    _write_resume_state(run_dir, 3)
    out = jt.reduce_outcomes(run_dir, "train")
    assert cli._classify_failure(run_dir, 3, _WIDTH, out) == "resume"


def test_classify_failure_resume_wins_over_killed_by_signal(tmp_path):
    """The grace-SIGTERM path writes BOTH killed_by_signal failure.json AND the
    resume_* set; classify must return 'resume' (continue), NOT 'timeout'
    (fresh-retry), so the resume_* survive."""
    run_dir = _make_run_dir(tmp_path)
    _write_resume_state(run_dir, 3)
    sd = _spec_dir(run_dir, 3)
    with open(os.path.join(sd, "failure.json"), "w") as f:
        json.dump({"classification": "killed_by_signal"}, f)
    out = jt.reduce_outcomes(run_dir, "train")
    assert cli._classify_failure(run_dir, 3, _WIDTH, out) == "resume"


def test_resubmit_resume_path_does_not_archive_resume_files(tmp_path,
                                                            monkeypatch):
    """A 'resume' index is relaunched WITHOUT archiving its resume_* set -- the
    relaunched train task auto-continues (train.py resumes when resume_state.pkl
    is present and model.eqx/completion.json are absent). The sparse train+eval
    pair is still submitted and attempts bumped."""
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    # All other indices succeeded.
    for i in (0, 2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    # index 1 was killed mid-run: resume_state.pkl + killed_by_signal failure.json.
    _write_resume_state(rd, 1)
    with open(os.path.join(_spec_dir(rd, 1), "failure.json"), "w") as f:
        json.dump({"classification": "killed_by_signal"}, f)

    fake = _fake_slurm(ids=["7100", "7101"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd, "--submit"])
    assert rc == 0

    spec1 = _spec_dir(rd, 1)
    # CRITICAL: the resume_* set MUST survive (NOT archived) so the relaunched
    # train task continues from it.
    assert os.path.exists(os.path.join(spec1, "resume_state.pkl"))
    assert not os.path.exists(os.path.join(spec1, "resume_state.pkl.gen0"))
    # The resume path archives NOTHING for that index -- even the
    # killed_by_signal failure.json is left in place (train.py resumes when
    # resume_state.pkl is present; a fresh-retry archive would wrongly route it
    # like a timeout). It must NOT have been moved to *.gen0.
    assert os.path.exists(os.path.join(spec1, "failure.json"))
    assert not os.path.exists(os.path.join(spec1, "failure.json.gen0"))
    # A sparse train + eval pair was submitted for the resumed index.
    sbatch = [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"]
    assert len(sbatch) == 2

    def _arr(cmd):
        for tok in cmd:
            if tok.startswith("--array="):
                return tok.split("=", 1)[1].split("%", 1)[0]
        raise AssertionError("no --array")
    assert _arr(sbatch[0]) == _arr(sbatch[1]) == "1"
    # The resume relaunch carries NO oom/timeout resource overrides (it is a
    # plain continuation, not a resource-rerouted retry).
    train_cmd = sbatch[0]
    assert not any(t.startswith("--partition=") for t in train_cmd)
    assert not any(t.startswith("--mem=") for t in train_cmd)
    assert not any(t.startswith("--time=") for t in train_cmd)
    # attempts bumped so a genuinely-stuck resume cannot loop forever.
    attempts = json.load(open(os.path.join(rd, "attempts.json")))
    assert attempts.get("1") == 1


def test_resubmit_oom_still_archives_and_fresh_retries(tmp_path, monkeypatch):
    """An oom index (NO resume checkpoint) keeps the old behavior: archive its
    stale artifacts to *.gen0 and fresh-retry."""
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    for i in (0, 2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    with open(os.path.join(_spec_dir(rd, 1), "failure.json"), "w") as f:
        json.dump({"classification": "oom"}, f)

    fake = _fake_slurm(ids=["7200", "7201"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd, "--submit"])
    assert rc == 0
    spec1 = _spec_dir(rd, 1)
    # oom path archives the stale failure.json (fresh retry).
    assert not os.path.exists(os.path.join(spec1, "failure.json"))
    assert os.path.exists(os.path.join(spec1, "failure.json.gen0"))


def test_resubmit_resume_respects_attempt_cap(tmp_path, monkeypatch):
    """A resume index that has already hit the attempt cap is not relaunched
    (a genuinely-stuck resume cannot loop forever)."""
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    for i in (0, 2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    _write_resume_state(rd, 1)
    cli._write_attempts(rd, {"1": 3})

    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd, "--submit", "--attempt-cap", "3"])
    assert rc == 0
    # Capped out -> no sbatch; resume_* untouched.
    assert [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"] == []
    assert os.path.exists(os.path.join(_spec_dir(rd, 1), "resume_state.pkl"))


def test_resubmit_resume_dry_run_lists_and_does_not_archive(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    """Dry-run lists the resume index for RESUME and makes NO sbatch call NOR
    archives the resume_* set."""
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    for i in (0, 2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    _write_resume_state(rd, 1)

    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd])      # dry-run (no --submit)
    assert rc == 0
    assert [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"] == []
    # resume_* untouched on dry-run.
    assert os.path.exists(os.path.join(_spec_dir(rd, 1), "resume_state.pkl"))
    out = capsys.readouterr().out
    assert "retry=[1]" in out


def test_resubmit_deterministic_skipped_with_resume_index(tmp_path,
                                                          monkeypatch):
    """A deterministic-failure index is still skipped even when a separate
    resume index is present (the two routes coexist)."""
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    for i in (0, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    # index 1 is resumable; index 2 is a genuine code error -> skipped.
    _write_resume_state(rd, 1)
    with open(os.path.join(_spec_dir(rd, 2), "failure.json"), "w") as f:
        json.dump({"classification": "assertion_error"}, f)

    fake = _fake_slurm(ids=["7300", "7301"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd, "--submit"])
    assert rc == 0
    sbatch = [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"]

    def _arr(cmd):
        for tok in cmd:
            if tok.startswith("--array="):
                return tok.split("=", 1)[1].split("%", 1)[0]
        raise AssertionError("no --array")
    # Only index 1 (resume) submitted; index 2 (deterministic) skipped.
    arrays = {_arr(c) for c in sbatch}
    assert arrays == {"1"}


def test_status_tallies_incomplete_resumable_and_remedy(tmp_path, monkeypatch,
                                                        capsys):
    """status counts incomplete_resumable in the train tally and prints a remedy
    line telling the operator a resume checkpoint will be continued."""
    run_dir = _make_run_dir(tmp_path)
    open(os.path.join(_spec_dir(run_dir, 0), "model.eqx"), "wb").close()
    _write_resume_state(run_dir, 1)
    jt.append_job_record(run_dir, "train", "1000", list(range(_N)))
    jt.append_job_record(run_dir, "eval", "2000", list(range(_N)))

    # everything else never scheduled (no sacct rows).
    fake = _fake_slurm(sacct_rows={"1000": "", "2000": ""})
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["status", run_dir])
    assert rc == 0
    out = capsys.readouterr().out
    assert "incomplete_resumable=1" in out
    # remedy mentions the resume checkpoint / continue.
    assert "resume checkpoint" in out
    assert "continue" in out


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

    from xcquinox.alec.cluster.grid_config import FidelityConfig

    cfg = cli.load_grid_config(
        "hpcjobs/configs/dfs_step7.dfs6311_grid3_v4.yaml")
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

def _handedit_resolved_fidelity(run_dir, **fidelity):
    """Rewrite ``<run_dir>/resolved_config.yaml`` with a given fidelity block.

    Models the exposure these tests pin: ``resolved_config.yaml`` is a plain
    file that outlives the ``submit`` that validated it, so its certificate
    settings can be edited afterwards. Every command that reloads it must
    re-run the same semantic validation.
    """
    import yaml

    p = os.path.join(run_dir, cli._RESOLVED_CONFIG_FILENAME)
    with open(p) as f:
        raw = yaml.safe_load(f)
    raw["fidelity"] = dict(fidelity)
    with open(p, "w") as f:
        yaml.safe_dump(raw, f)
    return p


def test_prepare_refuses_a_resolved_config_with_unreasoned_enforce_false(
        tmp_path, monkeypatch):
    """`prepare` takes any grid config, a run's resolved_config.yaml included.
    An un-reasoned enforce=false must be refused before any input is staged."""
    rd = _make_run_dir(tmp_path, manifest=False)
    cfg_path = _handedit_resolved_fidelity(
        rd, tol_AE=1.0, tol_atom=1.0, enforce=False)
    monkeypatch.setenv("SLURM_JOB_ID", "987654")
    called = {}

    def _fake_prepare(cfg, *, recompute_refs=True):
        called["ran"] = True
        raise AssertionError("prepare_inputs must not run on a refused config")

    monkeypatch.setattr(cli, "prepare_inputs", _fake_prepare)
    with pytest.raises(ValueError, match="override_reason"):
        main(["prepare", cfg_path, "--no-recompute-refs"])
    assert "ran" not in called


def test_resubmit_refuses_a_resolved_config_with_unreasoned_enforce_false(
        tmp_path, monkeypatch):
    """`resubmit` re-renders and re-submits train+eval arrays from the run's
    resolved_config.yaml, so it must re-validate that config: refusal happens
    before any sbatch, and the harness lock is released on the way out."""
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    open(os.path.join(_spec_dir(rd, 0), "model.eqx"), "wb").close()
    for i in (3, 4, 5):
        with open(os.path.join(_spec_dir(rd, i), "failure.json"), "w") as f:
            json.dump({"classification": "value_error"}, f)
    train_rows = "\n".join([
        "1000_1|OUT_OF_MEMORY|0:125",
        "1000_2|OUT_OF_MEMORY|0:125",
    ])
    fake = _fake_slurm(ids=["7001", "7002"], sacct_rows={"1000": train_rows})
    monkeypatch.setattr(jt, "_run_slurm", fake)
    _handedit_resolved_fidelity(rd, tol_AE=1.0, tol_atom=1.0, enforce=False)

    with pytest.raises(ValueError, match="override_reason"):
        main(["resubmit", rd, "--submit"])
    assert [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"] == []
    assert not os.path.exists(os.path.join(rd, cli._LOCK_FILENAME))


def test_resubmit_preflight_refuses_a_resolved_config_with_unreasoned_enforce_false(
        tmp_path, monkeypatch):
    """`resubmit-preflight` re-submits the whole pretrain->eval graph from the
    resolved config, so the same refusal applies before any SLURM call."""
    run_dir = _make_run_dir(tmp_path, manifest=False)
    jt.append_job_record(run_dir, "train", "200", list(range(_N)))
    _handedit_resolved_fidelity(
        run_dir, tol_AE=1.0, tol_atom=1.0, enforce=False)
    fake = _fake_slurm(ids=["400", "401", "402", "403", "404"])
    monkeypatch.setattr(jt, "_run_slurm", fake)

    with pytest.raises(ValueError, match="override_reason"):
        main(["resubmit-preflight", run_dir, "--submit"])
    assert fake.calls == []


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
    from xcquinox.alec.cluster.grid_config import load_grid_config
    d = _base_config_dict()
    mutate(d)
    p = tmp_path / f"_cfg_{tag}.json"
    p.write_text(json.dumps(d))
    cli._write_resolved_config(load_grid_config(str(p)), run_dir)


def _make_inline_resubmit_run(tmp_path, monkeypatch, *, retry_knobs=True):
    """A resubmit-ready run dir rendered the way an inline-eval submit renders:
    ``inline_eval: true`` in the resolved config and ONLY
    ``scripts/train_eval_inline.sbatch`` on disk."""
    rd = _make_resubmit_run(tmp_path, monkeypatch)

    def _mutate(d):
        d["inline_eval"] = True
        if retry_knobs:
            d["cluster"]["timeout_retry_partition"] = "long-96core"
            d["cluster"]["timeout_retry_time"] = "96:00:00"
            d["cluster"]["oom_retry_partition"] = "long-96core"
            d["cluster"]["oom_retry_mem"] = "512G"

    _rewrite_resolved(tmp_path, rd, _mutate, "inline")
    _write_scripts(rd, ["train_eval_inline.sbatch"])
    return rd


def _sbatch_calls(fake):
    return [c for c in fake.calls if os.path.basename(c[0]) == "sbatch"]


def test_resubmit_inline_eval_checkpointed_timeout_resumes_same_resources(
        tmp_path, monkeypatch):
    """A wall-killed inline-eval train cell WITH a resume checkpoint is
    resubmitted into the SAME run directory, at the SAME partition and wall.

    This is the recovery the 48 h GGA groups rely on: the cell continues from
    ``resume_state.pkl`` inside its existing spec dir, so a ~50 h cell finishes
    inside a second window of the same wall. Escalating instead would be wrong
    twice over -- it restarts nothing (the work is on disk) and it moves the
    cell to a queue it does not need.
    """
    rd = _make_inline_resubmit_run(tmp_path, monkeypatch)
    for i in (0, 2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    # index 1: wall-grace SIGTERM AND a mid-run checkpoint.
    _write_resume_state(rd, 1)
    with open(os.path.join(_spec_dir(rd, 1), "failure.json"), "w") as f:
        json.dump({"classification": "killed_by_signal"}, f)

    fake = _fake_slurm(ids=["6100"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd, "--submit"])
    assert rc == 0

    sbatch = _sbatch_calls(fake)
    # ONE array: the inline script runs train and eval in the same task, so
    # there is no separate eval array and no aftercorr dependency.
    assert len(sbatch) == 1, sbatch
    cmd = sbatch[0]
    assert cmd[-1] == os.path.join(rd, "scripts", "train_eval_inline.sbatch")
    assert not any(t.startswith("--dependency=") for t in cmd), cmd
    # SAME resources: the checkpointed timeout resumes where it was.
    assert not any(t.startswith("--partition=") for t in cmd), cmd
    assert not any(t.startswith("--time=") for t in cmd), cmd
    assert not any(t.startswith("--mem=") for t in cmd), cmd
    # SAME run dir: the checkpoint survives, unarchived, where the relaunched
    # task will read it.
    spec1 = _spec_dir(rd, 1)
    assert os.path.exists(os.path.join(spec1, "resume_state.pkl"))
    assert not os.path.exists(os.path.join(spec1, "resume_state.pkl.gen0"))
    # Only a train record is appended (gen0 train + gen0 eval came from setup).
    records = jt.read_job_records(rd)
    assert len([r for r in records if r["kind"] == "train"]) == 2
    assert len([r for r in records if r["kind"] == "eval"]) == 1
    attempts = json.load(open(os.path.join(rd, "attempts.json")))
    assert attempts.get("1") == 1


def test_resubmit_inline_eval_uncheckpointed_timeout_escalates(tmp_path,
                                                               monkeypatch):
    """A wall-killed inline-eval cell with NO checkpoint restarts from scratch,
    so it takes the escalated ``timeout_retry_partition``/``timeout_retry_time``
    -- the whole run has to fit in the retry window."""
    rd = _make_inline_resubmit_run(tmp_path, monkeypatch)
    for i in (0, 2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    with open(os.path.join(_spec_dir(rd, 1), "failure.json"), "w") as f:
        json.dump({"classification": "killed_by_signal"}, f)

    fake = _fake_slurm(ids=["6200"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd, "--submit"])
    assert rc == 0

    sbatch = _sbatch_calls(fake)
    assert len(sbatch) == 1, sbatch
    cmd = sbatch[0]
    assert cmd[-1] == os.path.join(rd, "scripts", "train_eval_inline.sbatch")
    assert "--partition=long-96core" in cmd, cmd
    assert "--time=96:00:00" in cmd, cmd


def test_resubmit_no_longer_refuses_an_inline_eval_run(tmp_path, monkeypatch,
                                                       capsys):
    """The blanket inline-eval refusal is gone.

    It returned 1 for every inline run, which is every v6 group submission, so
    a wall-killed cell had no recovery path at all.
    """
    rd = _make_inline_resubmit_run(tmp_path, monkeypatch)
    for i in (0, 2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    _write_resume_state(rd, 1)

    fake = _fake_slurm(ids=["6300"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd, "--submit"])
    out = capsys.readouterr().out
    assert "does not support inline" not in out, out
    assert rc == 0
    assert len(_sbatch_calls(fake)) == 1


def test_resubmit_inline_eval_oom_keeps_its_escalation(tmp_path, monkeypatch):
    """An OOM-killed inline cell with no checkpoint keeps the existing OOM
    escalation (partition + memory), on the inline script."""
    rd = _make_inline_resubmit_run(tmp_path, monkeypatch)
    for i in (0, 2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    with open(os.path.join(_spec_dir(rd, 1), "failure.json"), "w") as f:
        json.dump({"classification": "oom"}, f)

    fake = _fake_slurm(ids=["6400"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd, "--submit"])
    assert rc == 0
    cmd = _sbatch_calls(fake)[0]
    assert cmd[-1] == os.path.join(rd, "scripts", "train_eval_inline.sbatch")
    assert "--partition=long-96core" in cmd
    assert "--mem=512G" in cmd


def test_resubmit_inline_eval_dry_run_names_the_inline_script(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    """The dry run prints the single inline sbatch line and makes no call."""
    rd = _make_inline_resubmit_run(tmp_path, monkeypatch)
    for i in (0, 2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    _write_resume_state(rd, 1)

    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd])
    assert rc == 0
    assert _sbatch_calls(fake) == []
    out = capsys.readouterr().out
    assert "train_eval_inline.sbatch" in out
    assert "aftercorr" not in out, out


def test_resubmit_checkpointed_timeout_keeps_partition_and_wall(tmp_path,
                                                                monkeypatch):
    """The same wall semantics on a NON-inline run, with the retry knobs set so
    the assertion is not vacuous: a checkpointed timeout draws neither the
    escalated partition nor the escalated wall."""
    rd = _make_resubmit_run(tmp_path, monkeypatch)

    def _mutate(d):
        d["cluster"]["timeout_retry_partition"] = "long-96core"
        d["cluster"]["timeout_retry_time"] = "96:00:00"

    _rewrite_resolved(tmp_path, rd, _mutate, "arr_to")
    _write_scripts(rd, ["train_array.sbatch", "eval_array.sbatch"])
    for i in (0, 2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    _write_resume_state(rd, 1)
    with open(os.path.join(_spec_dir(rd, 1), "failure.json"), "w") as f:
        json.dump({"classification": "killed_by_signal"}, f)

    fake = _fake_slurm(ids=["6500", "6501"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd, "--submit"])
    assert rc == 0
    sbatch = _sbatch_calls(fake)
    assert len(sbatch) == 2  # train + eval arrays, the non-inline shape.
    train_cmd = sbatch[0]
    assert not any(t.startswith("--partition=") for t in train_cmd), train_cmd
    assert not any(t.startswith("--time=") for t in train_cmd), train_cmd


def test_resubmit_refuses_when_no_train_script_was_captured(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    """A run dir carrying neither script is refused before any SLURM call.

    This is what the former inline refusal actually protected against: handing
    ``sbatch`` a path that does not exist. It is now checked directly, so it
    covers a pruned run dir as well as an inline one.
    """
    rd = _make_resubmit_run(tmp_path, monkeypatch)
    _write_scripts(rd, [])
    for i in (0, 2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    with open(os.path.join(_spec_dir(rd, 1), "failure.json"), "w") as f:
        json.dump({"classification": "oom"}, f)

    fake = _fake_slurm()
    monkeypatch.setattr(jt, "_run_slurm", fake)
    rc = main(["resubmit", rd, "--submit"])
    assert rc == 1
    assert _sbatch_calls(fake) == []
    assert "train_array.sbatch" in capsys.readouterr().out


def test_status_remedy_for_an_inline_run_is_a_resubmit_that_works(
        tmp_path, monkeypatch, capsys):
    """``status`` recommends ``resubmit`` for a resumable inline-eval run, and
    that command succeeds on the same run dir.

    The two used to disagree: status printed the remedy and resubmit returned 1
    on every inline run.
    """
    rd = _make_inline_resubmit_run(tmp_path, monkeypatch)
    for i in (0, 2, 3, 4, 5):
        open(os.path.join(_spec_dir(rd, i), "model.eqx"), "wb").close()
    _write_resume_state(rd, 1)

    fake = _fake_slurm(ids=["6600"])
    monkeypatch.setattr(jt, "_run_slurm", fake)
    assert main(["status", rd]) == 0
    out = capsys.readouterr().out
    assert "resubmit --submit <run_dir>" in out, out
    assert "resume checkpoint" in out, out
    assert main(["resubmit", rd, "--submit"]) == 0
    assert len(_sbatch_calls(fake)) == 1


# ===========================================================================
# status: the remedy for a dead PRETRAIN stage
# ===========================================================================
# A pretrain array task has no resubmit path -- `cmd_resubmit` reduces the
# train and eval kinds only -- so a pretrain stage that died takes the same
# on-disk signature as a dead preflight (nothing downstream ran, because the
# afterok dependency never fired) and the same recovery, `resubmit-preflight`.
# The remedy has to say which stage is incomplete, or an operator reads
# "preflight" for a run whose preflight never started.

def _certify(run_dir, arch, verdict="PASS", **extra):
    """Write the artifacts a completed, certified pretrain leaves behind.

    ``extra`` sets any further certificate field, which is how the waived FAIL
    -- ``enforced=False`` beside a ``tolerances.override_reason`` -- is built.
    """
    from xcquinox.alec.cluster.grid_config import pretrain_checkpoint_dir

    ck = pretrain_checkpoint_dir(run_dir, arch)
    os.makedirs(ck, exist_ok=True)
    open(os.path.join(ck, "xnet.eqx"), "wb").close()
    open(os.path.join(ck, "cnet.eqx"), "wb").close()
    payload = {"verdict": verdict, "arch": arch}
    payload.update(extra)
    with open(os.path.join(ck, "fidelity_certificate.json"), "w") as f:
        json.dump(payload, f)
    return ck


def _status_with_nothing_downstream(tmp_path, monkeypatch):
    """A run dir whose train array never ran, and its `status` output."""
    run_dir = _make_run_dir(tmp_path)
    jt.append_job_record(run_dir, "train", "1000", list(range(_N)))
    jt.append_job_record(run_dir, "eval", "2000", list(range(_N)))
    monkeypatch.setattr(
        jt, "_run_slurm",
        _fake_slurm(sacct_rows={"1000": "\n".join(
            f"1000_{i}|CANCELLED by 0|0:0" for i in range(_N)), "2000": ""}))
    return run_dir


def test_status_names_the_pretrain_stage_when_it_is_the_incomplete_one(
        tmp_path, monkeypatch, capsys):
    """No train task ran and the pretrain checkpoints are absent.

    The remedy must name `resubmit-preflight` -- the only command that can
    recover a pretrain stage -- and it must say that the recovery re-runs only
    what is incomplete, which is what makes it cheap now that a completed
    architecture's pretraining is kept (`cluster/_pretrain.py`).
    """
    run_dir = _status_with_nothing_downstream(tmp_path, monkeypatch)
    assert main(["status", run_dir]) == 0
    out = capsys.readouterr().out
    assert "0/1 architecture checkpoint pair(s) present" in out, out
    assert "resubmit-preflight <run_dir>" in out, out
    assert "pretrain" in out.split("remedy:")[-1], out
    assert "incomplete" in out.split("remedy:")[-1], out


def test_status_still_names_the_preflight_when_the_pretrain_is_certified(
        tmp_path, monkeypatch, capsys):
    """Same on-disk signature with the pretrain stage finished and certified.

    The recovery command is unchanged; only the stage named moves, so the
    remedy is not simply relabelled for every dead graph.
    """
    run_dir = _status_with_nothing_downstream(tmp_path, monkeypatch)
    _certify(run_dir, "medium")
    assert main(["status", run_dir]) == 0
    out = capsys.readouterr().out
    assert "1/1 architecture certificate(s) PASS" in out, out
    remedy = out.split("remedy:")[-1]
    assert "resubmit-preflight <run_dir>" in remedy, out
    assert "preflight" in remedy, out


def test_status_names_a_waived_certificate_as_released_not_gated_out(
        tmp_path, monkeypatch, capsys):
    """A FAIL under a recorded waiver releases the on-node gates.

    ``fidelity.gate_certificate_from_read`` releases a FAIL that records
    ``enforced: false`` and a non-empty ``tolerances.override_reason``, and the
    pretrain task keeps such an architecture, so the train array was NOT gated
    out by that certificate -- the workflow-verification matrix, and cluster
    job 2134455, run in exactly this state. A remedy deciding on a PASS-only
    count states a cause that did not happen and disagrees with the pretrain
    stage about the same file.
    """
    run_dir = _status_with_nothing_downstream(tmp_path, monkeypatch)
    _certify(run_dir, "medium", verdict="FAIL", enforced=False,
             tolerances={"override_reason": "workflow matrix: wiring check"})
    assert main(["status", run_dir]) == 0
    out = capsys.readouterr().out
    remedy = out.split("remedy:")[-1]
    assert "gated out" not in remedy, out
    assert "waived" in remedy, out
    assert "medium" in remedy, out


def test_status_says_gated_out_for_an_enforced_failing_certificate(
        tmp_path, monkeypatch, capsys):
    """An enforced FAIL really does block the train array's dependency.

    The pretrain task exits non-zero on it, so ``afterok`` never fires. The
    branch above must not swallow this one: the two differ only in the waiver.
    """
    run_dir = _status_with_nothing_downstream(tmp_path, monkeypatch)
    _certify(run_dir, "medium", verdict="FAIL")
    assert main(["status", run_dir]) == 0
    out = capsys.readouterr().out
    remedy = out.split("remedy:")[-1]
    assert "gated out" in remedy, out
    assert "0/1" in remedy, out


# ===========================================================================
# bh76_mode explicitness: prepare/submit refuse a DFS-domain grid FILE that
# does not state its BH76 objective (the silent default trained the
# reaction-energy substitution through every campaign to v6).
# ===========================================================================

def test_prepare_refuses_a_dfs_grid_without_explicit_bh76_mode(tmp_path, capsys):
    grid = _write_grid(tmp_path, mutate=lambda d: d.pop("bh76_mode", None))
    rc = main(["prepare", grid, "--no-recompute-refs"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "bh76_mode" in out
    assert "barrier_height" in out


def test_submit_refuses_a_dfs_grid_without_explicit_bh76_mode(tmp_path, capsys):
    """The refusal lands BEFORE the run directory is created: a refused
    submission must leave no half-staged run tree behind. (The fixture grid
    WITH its stated mode passing submit is covered by the pre-existing
    dry-run submit test.)"""
    grid = _write_grid(tmp_path, mutate=lambda d: d.pop("bh76_mode", None))
    run_root = tmp_path / "out"
    run_root.mkdir()
    rc = main(["submit", grid, "--run-root", str(run_root),
               "--partition", "long-40core"])
    assert rc == 1
    assert list(run_root.iterdir()) == []
    out = capsys.readouterr().out
    assert "bh76_mode" in out


@pytest.mark.parametrize("command", ["resubmit", "resubmit-preflight",
                                     "repair-manifest"])
def test_resubmission_commands_refuse_a_resolved_config_stripped_of_bh76_mode(
        tmp_path, monkeypatch, capsys, command):
    """The resubmission family re-renders and re-submits work from
    resolved_config.yaml precisely because that file is untrusted after the
    submit that validated it; a hand-edit that deletes the bh76_mode key must
    refuse like prepare/submit do, not silently load the dataclass default
    and train a different objective than the run was created with."""
    rd = _make_run_dir(tmp_path)
    resolved = os.path.join(rd, "resolved_config.yaml")
    lines = [ln for ln in open(resolved).read().splitlines(keepends=True)
             if not ln.startswith("bh76_mode:")]
    with open(resolved, "w") as f:
        f.writelines(lines)
    rc = main([command, rd])
    assert rc == 1
    out = capsys.readouterr().out
    assert "bh76_mode" in out


@pytest.mark.parametrize("command", ["resubmit", "resubmit-preflight"])
def test_resubmission_commands_handle_a_corrupt_resolved_config(
        tmp_path, capsys, command):
    """A syntactically invalid resolved_config.yaml must produce each
    command's own unrecoverable-config refusal (rc 1, named path, direction
    to a fresh run dir), not an uncaught parser traceback -- repair-manifest
    already had this handling; resubmit and resubmit-preflight did not."""
    rd = _make_run_dir(tmp_path)
    with open(os.path.join(rd, "resolved_config.yaml"), "w") as f:
        f.write("a: [unclosed\nb: {broken\n")
    rc = main([command, rd])
    assert rc == 1
    out = capsys.readouterr().out
    assert "resolved_config.yaml" in out
    assert "fresh run dir" in out


def test_submit_refuses_a_null_bh76_mode_at_the_cli(tmp_path):
    """Pin of the CLI-level refusal for a stated-but-null bh76_mode: the
    presence guard passes (the key exists), validation raises, nothing is
    staged. Validation failures surface as the raised ValueError, the
    pre-existing behavior for every semantic refusal in submit."""
    grid = _write_grid(tmp_path, mutate=lambda d: d.update(bh76_mode=None))
    run_root = tmp_path / "out"
    run_root.mkdir()
    with pytest.raises(ValueError, match="bh76_mode"):
        main(["submit", grid, "--run-root", str(run_root),
              "--partition", "long-40core"])
    assert list(run_root.iterdir()) == []


@pytest.mark.parametrize("ext,body", [("yaml", "a: [unclosed\nb: {broken\n"),
                                      ("json", "{,broken")])
def test_prepare_and_submit_refuse_a_corrupt_grid_cleanly(
        tmp_path, capsys, ext, body):
    """A grid file that does not parse must refuse with rc 1 and a message
    naming the file, at both submission entry points and for both formats --
    not surface as a raw parser traceback (the resubmission family gained
    this in the same round; prepare/submit were the last config-reading
    commands without it)."""
    p = tmp_path / f"grid.{ext}"
    p.write_text(body)
    run_root = tmp_path / "out"
    run_root.mkdir()
    for argv in (["prepare", str(p), "--no-recompute-refs"],
                 ["submit", str(p), "--run-root", str(run_root),
                  "--partition", "long-40core"]):
        rc = main(argv)
        assert rc == 1, argv
        out = capsys.readouterr().out
        assert "cannot parse" in out, (argv, out)
        assert str(p) in out, (argv, out)
    assert list(run_root.iterdir()) == []


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
from xcquinox.alec.cluster.fidelity import (  # noqa: E402
    CERTIFICATE_FILENAME as fid_CERTIFICATE_FILENAME)


def test_regate_dry_run_flips_nothing_and_exits_zero_when_all_would_pass(
        tmp_path):
    rd, tracked, cert_path = _regate_fixture(tmp_path)
    resolved = os.path.join(rd, cli._RESOLVED_CONFIG_FILENAME)
    before_cert = open(cert_path, "rb").read()
    before_resolved = open(resolved, "rb").read()
    rc = main(["regate-certificates", rd, "--config", tracked])
    assert rc == 0
    assert open(cert_path, "rb").read() == before_cert
    assert open(resolved, "rb").read() == before_resolved


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


def test_regate_apply_is_idempotent(tmp_path):
    rd, tracked, cert_path = _regate_fixture(tmp_path)
    assert main(["regate-certificates", rd, "--config", tracked,
                 "--apply"]) == 0
    after_first = open(cert_path, "rb").read()
    assert main(["regate-certificates", rd, "--config", tracked,
                 "--apply"]) == 0
    assert open(cert_path, "rb").read() == after_first


def test_regate_missing_certificate_is_reported_not_raised(tmp_path):
    rd, tracked, _ = _regate_fixture(tmp_path, with_cert=False)
    rc = main(["regate-certificates", rd, "--config", tracked, "--apply"])
    assert rc == 1


def test_regate_backstop_fail_exits_nonzero_but_records_the_new_gate(
        tmp_path):
    rd, tracked, cert_path = _regate_fixture(tmp_path, mol_dae=4.6)
    rc = main(["regate-certificates", rd, "--config", tracked, "--apply"])
    assert rc == 1
    with open(cert_path) as f:
        cert = json.load(f)
    assert cert["verdict"] == "FAIL"
    assert any("tol_AE_max_backstop" in r
               for r in cert["summary"]["failure_reasons"])
    assert cert["tolerances"]["tol_AE_aggregate"] == "mae"


def test_regate_refuses_an_identity_mismatch_and_writes_nothing(tmp_path):
    rd, tracked, cert_path = _regate_fixture(
        tmp_path, tracked_overrides={"inputs.basis": "def2-svp"})
    before = open(cert_path, "rb").read()
    rc = main(["regate-certificates", rd, "--config", tracked, "--apply"])
    assert rc == 1
    assert open(cert_path, "rb").read() == before


def test_regate_refuses_a_run_dir_without_a_resolved_config(tmp_path):
    rd, tracked, _ = _regate_fixture(tmp_path)
    os.unlink(os.path.join(rd, cli._RESOLVED_CONFIG_FILENAME))
    assert main(["regate-certificates", rd, "--config", tracked]) == 1


def test_regate_reports_a_corrupt_certificate_without_crashing(tmp_path):
    """... and, on --apply, does NOT rewrite resolved_config.yaml: a run
    with an unreadable certificate is a broken run, and changing its gate
    policy on disk after a failed command leaves state the operator never
    successfully applied (a later resubmit-preflight would certify fresh
    pretrains under it)."""
    rd, tracked, cert_path = _regate_fixture(tmp_path)
    resolved = os.path.join(rd, cli._RESOLVED_CONFIG_FILENAME)
    before_resolved = open(resolved, "rb").read()
    with open(cert_path, "w") as f:
        f.write("{not json")
    rc = main(["regate-certificates", rd, "--config", tracked, "--apply"])
    assert rc == 1
    assert open(resolved, "rb").read() == before_resolved
    cfg = cli.load_grid_config(resolved)
    assert cfg.fidelity.tol_AE_aggregate == "max"


def test_regate_apply_names_the_fidelity_values_it_writes(tmp_path, capsys):
    """The resolved rewrite replaces the WHOLE fidelity block from the
    tracked config; the log must state old -> new so a tracked-config edit
    to tol_atom or enforce cannot ride along unremarked."""
    rd, tracked, _ = _regate_fixture(tmp_path)
    rc = main(["regate-certificates", rd, "--config", tracked, "--apply"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "resolved fidelity" in out
    assert "tol_AE_aggregate" in out and "mae" in out


def test_regate_skips_a_certificate_that_changed_since_it_was_read(
        tmp_path, monkeypatch):
    """Compute-node workers hold no .harness.lock: a certificate rewritten
    between the command's read and its write (a fresh pretrain finishing)
    must not be clobbered with a re-verdict of the STALE content."""
    rd, tracked, cert_path = _regate_fixture(tmp_path)
    sneaky = json.dumps(_regate_cert_payload(0.3, "PASS"))
    real = cli.regate_certificate_payload

    def _swap_then_regate(payload, fid_cfg, *, config_source):
        with open(cert_path, "w") as f:
            f.write(sneaky)
        return real(payload, fid_cfg, config_source=config_source)

    monkeypatch.setattr(cli, "regate_certificate_payload", _swap_then_regate)
    rc = main(["regate-certificates", rd, "--config", tracked, "--apply"])
    assert rc == 1
    with open(cert_path) as f:
        on_disk = json.load(f)
    assert on_disk == json.loads(sneaky), \
        "the concurrent write was clobbered with a stale re-verdict"
