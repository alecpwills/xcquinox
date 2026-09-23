"""Tests for xcquinox.pipeline.cluster.sync: the ``pull`` subcommand helpers.

Two layers:

  - Pure unit tests for :func:`build_rsync_command` and
    :func:`resolve_run_id`, covering the argv shape, dry-run toggle,
    profile -> filter-file mapping, and "latest"-resolution via an injected
    ssh_runner fake.
  - One end-to-end filter canary that drives the real ``rsync``
    executable against a tmp-path fixture mimicking a harness run dir, with
    ``host=""`` (local-to-local). This is the test that catches future drift
    between ``filters/summaries.filter`` and the artifact layout the harness
    writes. If rsync is unavailable in the test environment the test is
    skipped: the pure tests above still cover the argv shape.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from xcquinox.pipeline.cluster import sync


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GOOD_STAMP = "run_20260528T143052Z"


def _expected_filter_arg(profile: str) -> str:
    """The exact ``--filter=. <abs path>`` arg :func:`build_rsync_command` emits."""
    return f"--filter=. {sync.filter_file_path(profile)}"


# ---------------------------------------------------------------------------
# filter_file_path
# ---------------------------------------------------------------------------

def test_filter_file_path_summaries_exists():
    p = sync.filter_file_path("summaries")
    assert p.is_file()
    body = p.read_text()
    # spot-check critical rules
    assert "+ /manifest.json" in body
    assert "+ /checkpoints/spec_*/eval_df.csv" in body
    # The weights tier the enhancement-factor figures read, and the records
    # without which their loaders refuse an anchored checkpoint.
    for rule in ("+ /checkpoints/spec_*/model.eqx",
                 "+ /checkpoints/spec_*/model.eqx.class.json",
                 "+ /checkpoints/spec_*/model_val_best.eqx",
                 "+ /checkpoints/spec_*/model_val_best.eqx.class.json",
                 "+ /pretrain/*/xnet.eqx",
                 "+ /pretrain/*/cnet.eqx",
                 "+ /pretrain/*/xnet/xnet_val_best.eqx",
                 "+ /pretrain/*/cnet/cnet_val_best.eqx"):
        assert rule in body, rule
    assert "+ /checkpoints/spec_*/model_best.eqx" not in body, (
        "model_best.eqx (minimum TRAINING loss) is deliberately excluded: no "
        "figure reads it, and the default pull stays lean without it")
    assert body.rstrip().endswith("- *"), "the final catch-all exclude must be last"


# ---------------------------------------------------------------------------
# build_rsync_command: pure argv shape
# ---------------------------------------------------------------------------

def test_build_rsync_command_summaries_default():
    argv = sync.build_rsync_command(
        host="seawulf",
        remote_root="/gpfs/scratch/awills/xcquinox_runs/runs",
        local_root="/data/results",
        run_id=GOOD_STAMP,
    )
    assert argv[0] == "rsync"
    # base flags appear in order, exactly once each
    for flag in ("-a", "-v", "-z", "--partial", "--info=progress2"):
        assert argv.count(flag) == 1, f"{flag!r} missing or duplicated: {argv}"
    # filter arg points at the packaged summaries.filter
    assert _expected_filter_arg("summaries") in argv
    # default is NOT a dry-run
    assert "--dry-run" not in argv
    # src/dst are the last two args, in the right order, with trailing slashes
    assert argv[-2] == f"seawulf:/gpfs/scratch/awills/xcquinox_runs/runs/{GOOD_STAMP}/"
    assert argv[-1] == f"/data/results/{GOOD_STAMP}/"


# ---------------------------------------------------------------------------
# build_rsync_command: category (multi-series layout)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# build_rsync_command: spec_indices (surgical checkpoint extraction)
# ---------------------------------------------------------------------------

def test_build_rsync_command_spec_indices_emit_zero_padded_includes():
    argv = sync.build_rsync_command(
        host="h", remote_root="/r", local_root="/l",
        run_id=GOOD_STAMP, profile="full",
        spec_indices=[0, 36],
    )
    # Required rsync include/exclude rules, in order: the checkpoints dir,
    # each spec_<NNNN>/, each spec_<NNNN>/***, then a catch-all exclude.
    expected = [
        "--include=/checkpoints/",
        "--include=/checkpoints/spec_0000/",
        "--include=/checkpoints/spec_0000/***",
        "--include=/checkpoints/spec_0036/",
        "--include=/checkpoints/spec_0036/***",
        "--exclude=/checkpoints/spec_*",
    ]
    for rule in expected:
        assert rule in argv, f"missing {rule!r}: {argv}"
    # The first --include must come BEFORE the --filter= so it wins the
    # rsync first-match race against the catch-all exclude that follows.
    first_inc = next(i for i, a in enumerate(argv) if a.startswith("--include="))
    filter_idx = next(i for i, a in enumerate(argv) if a.startswith("--filter="))
    assert first_inc < filter_idx


# ---------------------------------------------------------------------------
# resolve_run_id
# ---------------------------------------------------------------------------


def test_resolve_run_id_latest_picks_newest_by_ls_order():
    calls = []

    def _ssh(argv):
        calls.append(list(argv))
        # `ls -1tr` returns oldest-first, so the newest is LAST.
        return [
            "run_20260528T100000Z",
            "run_20260528T120000Z",
            "run_20260528T143052Z",
        ]

    out = sync.resolve_run_id(
        "latest", ssh_runner=_ssh, remote_root="/gpfs/scratch/awills/xcquinox_runs/runs",
    )
    assert out == "run_20260528T143052Z"
    # the resolver issued exactly one SSH ls, with the right args
    assert calls == [["ls", "-1tr", "/gpfs/scratch/awills/xcquinox_runs/runs"]]


# ---------------------------------------------------------------------------
# discover_runs
# ---------------------------------------------------------------------------

def test_discover_runs_groups_by_relative_category():
    """A find listing mixing several categories must group correctly."""
    def _ssh(argv):
        assert argv[0] == "find"
        # the find target (argv[1]) is what we're asserting groups relative to
        return [
            "/gpfs/scratch/awills/xcquinox_runs/alpha_off/runs/run_20260601T120000Z",
            "/gpfs/scratch/awills/xcquinox_runs/alpha_off/runs/run_20260528T140000Z",
            "/gpfs/scratch/awills/xcquinox_runs/alpha_on/runs/run_20260530T100000Z",
            "/gpfs/scratch/awills/xcquinox_runs/polarized/alpha_on/run_20260527T090000Z",
            "/gpfs/scratch/awills/xcquinox_runs/run_20260101T000000Z",  # at root
        ]

    groups = sync.discover_runs(
        ssh_runner=_ssh,
        remote_root="/gpfs/scratch/awills/xcquinox_runs",
    )
    assert set(groups.keys()) == {
        "alpha_off/runs", "alpha_on/runs", "polarized/alpha_on", "",
    }
    # Sorted oldest-first => [-1] is latest. Stamp lex sort == time sort.
    assert groups["alpha_off/runs"] == [
        "run_20260528T140000Z", "run_20260601T120000Z",
    ]
    assert groups["alpha_off/runs"][-1] == "run_20260601T120000Z"
    assert groups["polarized/alpha_on"] == ["run_20260527T090000Z"]
    assert groups[""] == ["run_20260101T000000Z"]  # root-level run


# ---------------------------------------------------------------------------
# format_ssh_stderr_tail (SBU banner stripping)
# ---------------------------------------------------------------------------

# Real banner observed from a SeaWulf failing trace (compressed; the actual
# banner is longer but this captures the "long bureaucratic preamble" shape).
_SBU_BANNER_FIXTURE = """\
This SBU computing resource may NOT be used to train or assist in the training of AI models for or on behalf of entities headquartered in the following countries:

Afghanistan, Belarus, Burma (Myanmar), Cambodia, Central African Republic, China (PRC), Congo (Democratic Republic of), Cuba, Eritrea, Haiti, Iran, Iraq, Macau, North Korea, Lebanon, Libya, Nicaragua, Russia, Somalia, South Sudan, Sudan, Syria, Venezuela, Zimbabwe.

If you are unable to comply with this restriction, please contact: OVPR_researchsecurity_admin@stonybrook.edu for further guidance before using this resource.

ls: cannot access /gpfs/scratch/awills/xcquinox_runs/runs: No such file or directory
"""


# ---------------------------------------------------------------------------
# End-to-end filter canary (drives real rsync against a tmp fixture)
# ---------------------------------------------------------------------------

def _materialize_fake_run(root: Path) -> Path:
    """Build a tmp tree mirroring the on-disk artifacts the harness writes.

    Mirrors xcquinox/pipeline/cluster/__main__.py + _train_task.py + _eval_one_spec.py
    + _pretrain.py outputs. Returns the run dir's path.
    """
    run = root / GOOD_STAMP
    run.mkdir(parents=True)
    # Top-level metadata
    (run / "manifest.json").write_text('{"n_specs": 1}\n')
    (run / "resolved_config.yaml").write_text("sweep: {}\n")
    (run / "jobs.json").write_text("[]\n")
    (run / "attempts.json").write_text("{}\n")
    # Per-spec checkpoints
    spec = run / "checkpoints" / "spec_0000"
    (spec / "eval").mkdir(parents=True)
    (spec / "eval_df.csv").write_text("set,mae\nbh76,4.2\n")
    (spec / "failure.json").write_text('{"classification": "ok"}\n')
    (spec / "losses.npy").write_bytes(b"\x93NUMPY")  # bytes header is enough
    # Trained weights + their model-class records: the enhancement-factor
    # figures forward-evaluate these, and the loaders refuse an anchored
    # checkpoint whose record did not travel with it.
    (spec / "model.eqx").write_bytes(b"FAKE_MODEL_CHECKPOINT_BLOB" * 100)
    (spec / "model.eqx.class.json").write_text('{"parent_anchor": true}\n')
    (spec / "model_val_best.eqx").write_bytes(b"FAKE_VAL_BEST_BLOB" * 100)
    (spec / "model_val_best.eqx.class.json").write_text(
        '{"parent_anchor": true}\n')
    (spec / "model_best.eqx").write_bytes(b"FAKE_BEST_CHECKPOINT_BLOB" * 100)
    (spec / "model_best.eqx.class.json").write_text('{"parent_anchor": true}\n')
    # The in-flight resume set (mid-run state, no analysis use).
    (spec / "resume_model.eqx").write_bytes(b"FAKE_RESUME_BLOB" * 100)
    (spec / "resume_val_best.eqx").write_bytes(b"FAKE_RESUME_VB_BLOB" * 100)
    (spec / "resume_opt_state.eqx").write_bytes(b"FAKE_RESUME_OPT_BLOB" * 100)
    (spec / "eval" / "per_molecule.json").write_text("[]\n")
    # Held-out (BH76 + W4-11) reaction eval -- the "beats PBE?" headline dir.
    (spec / "eval_holdout").mkdir()
    (spec / "eval_holdout" / "test_set.csv").write_text(
        "set,mae_nn_kcalmol,mae_pbe_kcalmol,delta_nn_minus_pbe\n"
        "test_set_held_out_combined,9.1,11.8,-2.700000\n")
    (spec / "eval_holdout" / "per_reaction.json").write_text("[]\n")
    (spec / "eval_holdout" / "per_molecule.json").write_text("[]\n")
    # Best-loss-checkpoint held-out eval (model_best.eqx) -- the sibling dir.
    (spec / "eval_holdout_best").mkdir()
    (spec / "eval_holdout_best" / "test_set.csv").write_text(
        "set,mae_nn_kcalmol,mae_pbe_kcalmol,delta_nn_minus_pbe\n"
        "test_set_held_out_combined,8.4,11.8,-3.400000\n")
    (spec / "eval_holdout_best" / "per_reaction.json").write_text("[]\n")
    (spec / "eval_holdout_best" / "per_molecule.json").write_text("[]\n")
    # Validation-best-checkpoint held-out eval (model_val_best.eqx) -- the figures'
    # "best" selector; same small CSV/JSON, must be pulled by summaries.
    (spec / "eval_holdout_val_best").mkdir()
    (spec / "eval_holdout_val_best" / "test_set.csv").write_text(
        "set,mae_nn_kcalmol,mae_pbe_kcalmol,delta_nn_minus_pbe\n"
        "test_set_held_out_combined,8.0,11.8,-3.800000\n")
    (spec / "eval_holdout_val_best" / "per_reaction.json").write_text("[]\n")
    (spec / "eval_holdout_val_best" / "per_molecule.json").write_text("[]\n")
    # Cold-start trajectory-diagnostic channel (the 4th pass) + its
    # provenance stamp -- must be pulled by summaries like its siblings.
    (spec / "eval_holdout_coldstart").mkdir()
    (spec / "eval_holdout_coldstart" / "test_set.csv").write_text(
        "set,mae_nn_kcalmol,mae_pbe_kcalmol,delta_nn_minus_pbe\n"
        "test_set_held_out_combined,15.0,11.8,3.200000\n")
    (spec / "eval_holdout_coldstart" / "per_reaction.json").write_text("[]\n")
    (spec / "eval_holdout_coldstart" / "per_molecule.json").write_text("[]\n")
    (spec / "eval_holdout_coldstart" / "eval_metadata.json").write_text(
        '{"channel": "eval_holdout_coldstart", "coldstart": true}\n')
    # Converged-SCF channel (the 5th and 6th passes): the final and the
    # validation-best checkpoints re-evaluated under a converged pyscfad SCF.
    # Same small CSV/JSON artifacts plus the provenance stamp naming the
    # override, no *.eqx.
    for _conv in ("eval_holdout_converged", "eval_holdout_converged_val_best"):
        (spec / _conv).mkdir()
        (spec / _conv / "test_set.csv").write_text(
            "set,mae_nn_kcalmol,mae_pbe_kcalmol,delta_nn_minus_pbe\n"
            "test_set_held_out_combined,7.9,11.8,-3.900000\n")
        (spec / _conv / "per_reaction.json").write_text("[]\n")
        (spec / _conv / "per_molecule.json").write_text("[]\n")
        (spec / _conv / "eval_metadata.json").write_text(
            '{"channel": "' + _conv + '", "channel_override": "converged"}\n')
    (spec / "eval_holdout_converged" / "_shards").mkdir()
    (spec / "eval_holdout_converged" / "_shards" / "shard_t1_s0.json").write_text(
        '{"energies": {}}\n' * 200)
    # The parallel eval's shard scratch: worker names/payload JSON, the bulk
    # of an eval_holdout*/ tree by bytes (~60 percent of a pull), of no
    # analysis use once merged into per_molecule/per_reaction. Must NOT be
    # pulled by summaries.
    (spec / "eval_holdout" / "_shards").mkdir()
    (spec / "eval_holdout" / "_shards" / "shard_t1_s0.json").write_text(
        '{"energies": {}}\n' * 200)
    (spec / "eval_holdout_val_best" / "_shards").mkdir()
    (spec / "eval_holdout_val_best" / "_shards" / "names_t1_s0.json").write_text(
        '["h2"]\n' * 200)
    # The staged validation slice (inputs._stage_validation_slice): the
    # identity record the figures' validation-column reader requires; a pull
    # without it silently rendered a DIFFERENT slice.
    (run / "validation").mkdir()
    (run / "validation" / "val_reactions.json").write_text(
        '{"reactions": []}\n')
    # The run-level representative-subset ledger.
    (run / "subset_ledger.json").write_text('{"jsd": {}}\n')
    # The run-level CCSD T1 diagnostic table (benchmark_refs --write-t1-json):
    # the model-free multireference list the outlier-free figure variant reads.
    (run / "t1_diagnostics.json").write_text(
        '{"t1": {"NO": 0.05}, "threshold": 0.02, "source": "/refs"}\n')
    # Pretrain
    pre = run / "pretrain" / "deep_combined_attn"
    pre.mkdir(parents=True)
    (pre / "pretrain_metadata.json").write_text('{"steps": 1000}\n')
    (pre / "losses_x.npy").write_bytes(b"\x93NUMPY")
    (pre / "losses_c.npy").write_bytes(b"\x93NUMPY")
    (pre / "xnet.eqx").write_bytes(b"FAKE_XNET_BLOB" * 100)
    (pre / "cnet.eqx").write_bytes(b"FAKE_CNET_BLOB" * 100)
    (pre / "xnet.eqx.class.json").write_text('{"parent_anchor": true}\n')
    (pre / "cnet.eqx.class.json").write_text('{"parent_anchor": true}\n')
    # The per-network subdirs pretrain.py writes: the validated best pair, and
    # the periodic xc.eqx.<step> trajectory snapshots that must stay remote.
    (pre / "xnet").mkdir()
    (pre / "cnet").mkdir()
    (pre / "xnet" / "xnet_val_best.eqx").write_bytes(b"FAKE_XVB_BLOB" * 100)
    (pre / "cnet" / "cnet_val_best.eqx").write_bytes(b"FAKE_CVB_BLOB" * 100)
    (pre / "xnet" / "xc.eqx.500").write_bytes(b"FAKE_SNAPSHOT_BLOB" * 100)
    (pre / "cnet" / "xc.eqx.500").write_bytes(b"FAKE_SNAPSHOT_BLOB" * 100)
    # Junk that must NOT be pulled by summaries
    (run / "stray.eqx").write_bytes(b"STRAY_BLOB" * 100)
    (run / "logs").mkdir()
    (run / "logs" / "train_42_0.out").write_text("chatty slurm log\n" * 50)
    (run / "scripts").mkdir()
    (run / "scripts" / "train_array.sbatch").write_text("#!/bin/bash\n")
    (run / "specs").mkdir()
    (run / "specs" / "spec_0000.spec").write_bytes(b"opaque-spec-blob")
    # An archived .gen<N> artifact from a previous resubmit
    (spec / "model.eqx.gen1").write_bytes(b"PREV_GEN")
    return run


@pytest.fixture
def fake_remote_root(tmp_path):
    root = tmp_path / "remote"
    root.mkdir()
    _materialize_fake_run(root)
    return root


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
def test_summaries_filter_canary_against_real_rsync(tmp_path, fake_remote_root):
    """Drive real rsync against the fixture; assert exactly the right files land.

    This is the CANARY: when a future commit adds a new harness artifact and
    forgets to update ``filters/summaries.filter``, this test fails loudly
    and identifies the new artifact in its assertion message.
    """
    local_root = tmp_path / "local"
    local_root.mkdir()
    (local_root / GOOD_STAMP).mkdir()  # cmd_pull does this; mirror it here

    argv = sync.build_rsync_command(
        host="",  # local-to-local
        remote_root=str(fake_remote_root),
        local_root=str(local_root),
        run_id=GOOD_STAMP,
        profile="summaries",
    )
    completed = subprocess.run(
        argv, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, (
        f"rsync failed (rc={completed.returncode}); "
        f"stderr=\n{completed.stderr}"
    )

    dest = local_root / GOOD_STAMP
    # --- must be present (summaries-tier artifacts) ----------------------
    must_have = [
        "manifest.json",
        "resolved_config.yaml",
        "jobs.json",
        "attempts.json",
        "checkpoints/spec_0000/eval_df.csv",
        "checkpoints/spec_0000/failure.json",
        "checkpoints/spec_0000/losses.npy",
        "checkpoints/spec_0000/eval/per_molecule.json",
        "checkpoints/spec_0000/eval_holdout/test_set.csv",
        "checkpoints/spec_0000/eval_holdout/per_reaction.json",
        "checkpoints/spec_0000/eval_holdout/per_molecule.json",
        "checkpoints/spec_0000/eval_holdout_best/test_set.csv",
        "checkpoints/spec_0000/eval_holdout_best/per_reaction.json",
        "checkpoints/spec_0000/eval_holdout_best/per_molecule.json",
        "checkpoints/spec_0000/eval_holdout_val_best/test_set.csv",
        "checkpoints/spec_0000/eval_holdout_val_best/per_reaction.json",
        "checkpoints/spec_0000/eval_holdout_val_best/per_molecule.json",
        "checkpoints/spec_0000/eval_holdout_coldstart/test_set.csv",
        "checkpoints/spec_0000/eval_holdout_coldstart/per_reaction.json",
        "checkpoints/spec_0000/eval_holdout_coldstart/per_molecule.json",
        "checkpoints/spec_0000/eval_holdout_coldstart/eval_metadata.json",
        "checkpoints/spec_0000/eval_holdout_converged/test_set.csv",
        "checkpoints/spec_0000/eval_holdout_converged/per_reaction.json",
        "checkpoints/spec_0000/eval_holdout_converged/per_molecule.json",
        "checkpoints/spec_0000/eval_holdout_converged/eval_metadata.json",
        "checkpoints/spec_0000/eval_holdout_converged_val_best/test_set.csv",
        "checkpoints/spec_0000/eval_holdout_converged_val_best/per_reaction.json",
        "checkpoints/spec_0000/eval_holdout_converged_val_best/per_molecule.json",
        "checkpoints/spec_0000/eval_holdout_converged_val_best/eval_metadata.json",
        "pretrain/deep_combined_attn/pretrain_metadata.json",
        "pretrain/deep_combined_attn/losses_x.npy",
        "pretrain/deep_combined_attn/losses_c.npy",
        # The network weights the enhancement-factor figures read, each with
        # the model-class record the loaders require beside it.
        "checkpoints/spec_0000/model.eqx",
        "checkpoints/spec_0000/model.eqx.class.json",
        "checkpoints/spec_0000/model_val_best.eqx",
        "checkpoints/spec_0000/model_val_best.eqx.class.json",
        "pretrain/deep_combined_attn/xnet.eqx",
        "pretrain/deep_combined_attn/cnet.eqx",
        "pretrain/deep_combined_attn/xnet.eqx.class.json",
        "pretrain/deep_combined_attn/cnet.eqx.class.json",
        "pretrain/deep_combined_attn/xnet/xnet_val_best.eqx",
        "pretrain/deep_combined_attn/cnet/cnet_val_best.eqx",
        # The staged validation slice + the run-level subset ledger: the
        # figures' validation-column reader hard-requires the former, and
        # the ledger names every cell's selected points.
        "validation/val_reactions.json",
        "subset_ledger.json",
        "t1_diagnostics.json",
    ]
    for rel in must_have:
        assert (dest / rel).is_file(), (
            f"summaries.filter dropped an artifact it should keep: {rel} "
            "(if this artifact was renamed, update filters/summaries.filter)"
        )

    # --- must NOT be present (excluded tier) -----------------------------
    # The *.eqx tier is now split rather than excluded wholesale: what the
    # figures read comes, what only a re-run would read stays remote.
    must_not_have = [
        "checkpoints/spec_0000/model_best.eqx",
        "checkpoints/spec_0000/model_best.eqx.class.json",
        "checkpoints/spec_0000/resume_model.eqx",
        "checkpoints/spec_0000/resume_val_best.eqx",
        "checkpoints/spec_0000/resume_opt_state.eqx",
        "checkpoints/spec_0000/model.eqx.gen1",
        "pretrain/deep_combined_attn/xnet/xc.eqx.500",
        "pretrain/deep_combined_attn/cnet/xc.eqx.500",
        "stray.eqx",
        "logs",
        "logs/train_42_0.out",
        "scripts",
        "scripts/train_array.sbatch",
        "specs",
        "specs/spec_0000.spec",
        # Shard scratch: merged into the summary JSONs already; ~60 percent
        # of an eval_holdout tree's bytes.
        "checkpoints/spec_0000/eval_holdout/_shards",
        "checkpoints/spec_0000/eval_holdout/_shards/shard_t1_s0.json",
        "checkpoints/spec_0000/eval_holdout_val_best/_shards",
        "checkpoints/spec_0000/eval_holdout_converged/_shards",
        "checkpoints/spec_0000/eval_holdout_converged/_shards/shard_t1_s0.json",
    ]
    for rel in must_not_have:
        assert not (dest / rel).exists(), (
            f"summaries.filter leaked an artifact it should exclude: {rel} "
            "(this indicates the filter is over-permissive; tighten the "
            "include rules or move this to filters/full.filter)"
        )


# ---------------------------------------------------------------------------
# pull auto: quoted ssh transport, activity discovery, multi-run filter/argv
# ---------------------------------------------------------------------------

SECOND_STAMP = "run_20260831T011905Z"


def test_ssh_remote_command_quotes_globs_and_scripts(tmp_path):
    from xcquinox.pipeline.cluster.__main__ import _ssh_remote_command
    root = tmp_path / "root" / "cat" / "runs" / GOOD_STAMP
    root.mkdir(parents=True)
    argv = ["find", str(tmp_path / "root"), "-mindepth", "1", "-maxdepth", "5",
            "-type", "d", "-name", "run_*Z", "-prune", "-print"]
    cmd = _ssh_remote_command(argv)
    assert "'run_*Z'" in cmd
    # Remote-shell round trip with a decoy glob match in the CWD: the pattern
    # must reach find as a literal, not be expanded by the shell first.
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "run_DECOYZ").write_text("")
    completed = subprocess.run(["sh", "-c", cmd], cwd=cwd,
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    lines = [ln for ln in completed.stdout.splitlines() if ln.strip()]
    assert lines == [str(root)]


# ---------------------------------------------------------------------------
# cmd_pull auto orchestration (subprocess.run recorded, nothing executed)
# ---------------------------------------------------------------------------

class _FakeCompleted:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


def _auto_args(tmp_path, **overrides):
    import argparse
    ns = argparse.Namespace(
        run_id="auto", profile="summaries", category="", host="hpc",
        remote_root="/scr/root", local_root=str(tmp_path / "local"),
        specs=None, dry_run=False, days=30.0, depth=5, yes=False,
        no_control_master=False, ssh_persist=3600)
    for key, val in overrides.items():
        setattr(ns, key, val)
    return ns


def _install_fake_run(monkeypatch, calls, *, find_lines, rsync_rc=0):
    from xcquinox.pipeline.cluster import __main__ as cm_mod

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if argv[0] == "ssh" and "-O" in argv:
            return _FakeCompleted(0)
        if argv[0] == "ssh":
            return _FakeCompleted(0, stdout="\n".join(find_lines) + "\n")
        if argv[0] == "rsync":
            return _FakeCompleted(rsync_rc)
        raise AssertionError(f"unexpected subprocess argv: {argv}")

    monkeypatch.setattr(cm_mod.subprocess, "run", fake_run)
    return cm_mod


def test_cmd_pull_auto_one_ssh_one_rsync(monkeypatch, tmp_path):
    calls = []
    find_lines = [f"A /scr/root/catA/runs/{GOOD_STAMP}",
                  "I /scr/root/catA/runs/run_20260101T000000Z"]
    cm_mod = _install_fake_run(monkeypatch, calls, find_lines=find_lines)
    rc = cm_mod.cmd_pull(_auto_args(tmp_path))
    assert rc == 0
    discovery = [a for a in calls if a[0] == "ssh" and "-O" not in a]
    rsyncs = [a for a in calls if a[0] == "rsync"]
    assert len(discovery) == 1, "discovery must be ONE ssh shot"
    assert len(rsyncs) == 1, "the pull must be ONE rsync invocation"
    (ssh_argv,) = discovery
    # Options before the host; the remote command as one quoted string.
    assert ssh_argv[0] == "ssh" and ssh_argv[1] == "-o"
    assert ssh_argv[-2] == "hpc"
    assert "'run_*Z'" in ssh_argv[-1]
    (argv,) = rsyncs
    assert "-R" in argv
    i_e = argv.index("-e")
    assert "ControlMaster=auto" in argv[i_e + 1]
    assert f"hpc:/scr/root/./catA/runs/{GOOD_STAMP}" in argv
    assert not any("run_20260101T000000Z" in a for a in argv)
    assert argv[-1] == str(tmp_path / "local") + "/"


