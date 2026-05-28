"""Tests for xcquinox.alec.cluster.sync — the ``pull`` subcommand helpers.

Two layers:

  - Pure unit tests for :func:`build_rsync_command` and
    :func:`resolve_run_id`, covering the argv shape, dry-run toggle,
    profile→filter-file mapping, and "latest"-resolution via an injected
    ssh_runner fake.
  - One **end-to-end filter canary** that drives the real ``rsync``
    executable against a tmp-path fixture mimicking a harness run dir, with
    ``host=""`` (local-to-local). This is the test that catches future drift
    between ``filters/summaries.filter`` and the artifact layout the harness
    writes. If rsync is unavailable in the test environment the test is
    skipped — the pure tests above still cover the argv shape.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from xcquinox.alec.cluster import sync


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
    assert body.rstrip().endswith("- *"), "the final catch-all exclude must be last"


def test_filter_file_path_full_exists():
    p = sync.filter_file_path("full")
    assert p.is_file()
    body = p.read_text()
    assert "- /logs/" in body
    assert "+ /***" in body


def test_filter_file_path_unknown_profile_raises():
    with pytest.raises(ValueError, match="unknown pull profile"):
        sync.filter_file_path("bogus")


# ---------------------------------------------------------------------------
# build_rsync_command — pure argv shape
# ---------------------------------------------------------------------------

def test_build_rsync_command_summaries_default():
    argv = sync.build_rsync_command(
        host="seawulf",
        remote_root="/gpfs/scratch/awills/xcquinox_runs/runs",
        local_root="/home/awills/results",
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
    assert argv[-1] == f"/home/awills/results/{GOOD_STAMP}/"


def test_build_rsync_command_full_profile_uses_full_filter():
    argv = sync.build_rsync_command(
        host="seawulf",
        remote_root="/gpfs/scratch/awills/xcquinox_runs/runs",
        local_root="/home/awills/results",
        run_id=GOOD_STAMP,
        profile="full",
    )
    assert _expected_filter_arg("full") in argv
    assert _expected_filter_arg("summaries") not in argv


def test_build_rsync_command_dry_run_appends_flag():
    argv = sync.build_rsync_command(
        host="seawulf", remote_root="/r", local_root="/l",
        run_id=GOOD_STAMP, dry_run=True,
    )
    assert "--dry-run" in argv
    # --dry-run must appear before the src/dst args
    assert argv.index("--dry-run") < len(argv) - 2


def test_build_rsync_command_empty_host_is_local_to_local():
    argv = sync.build_rsync_command(
        host="",
        remote_root="/tmp/fake_remote",
        local_root="/tmp/fake_local",
        run_id=GOOD_STAMP,
    )
    # no `host:` prefix on the source path
    assert argv[-2] == f"/tmp/fake_remote/{GOOD_STAMP}/"
    assert argv[-1] == f"/tmp/fake_local/{GOOD_STAMP}/"


def test_build_rsync_command_strips_trailing_slashes():
    """``--remote-root /r/`` and ``/r`` must yield the same source path."""
    argv_no_slash = sync.build_rsync_command(
        host="h", remote_root="/r", local_root="/l", run_id=GOOD_STAMP,
    )
    argv_slash = sync.build_rsync_command(
        host="h", remote_root="/r/", local_root="/l/", run_id=GOOD_STAMP,
    )
    assert argv_no_slash[-2] == argv_slash[-2]
    assert argv_no_slash[-1] == argv_slash[-1]


# ---------------------------------------------------------------------------
# build_rsync_command — category (multi-series layout)
# ---------------------------------------------------------------------------

def test_build_rsync_command_with_category_joins_remote_and_local_paths():
    argv = sync.build_rsync_command(
        host="seawulf",
        remote_root="/gpfs/scratch/awills/xcquinox_runs",
        local_root="/home/awills/results",
        run_id=GOOD_STAMP,
        category="alpha_off/runs",
    )
    assert argv[-2] == (
        f"seawulf:/gpfs/scratch/awills/xcquinox_runs/alpha_off/runs/{GOOD_STAMP}/"
    )
    # Local dest mirrors the remote category layout — collision protection.
    assert argv[-1] == f"/home/awills/results/alpha_off/runs/{GOOD_STAMP}/"


def test_build_rsync_command_multi_segment_category():
    argv = sync.build_rsync_command(
        host="seawulf", remote_root="/r", local_root="/l",
        run_id=GOOD_STAMP, category="polarized/alpha_on",
    )
    assert argv[-2] == f"seawulf:/r/polarized/alpha_on/{GOOD_STAMP}/"
    assert argv[-1] == f"/l/polarized/alpha_on/{GOOD_STAMP}/"


def test_build_rsync_command_empty_category_is_noop():
    """Back-compat: empty category must reproduce the pre-category behavior."""
    argv_no_cat = sync.build_rsync_command(
        host="h", remote_root="/r", local_root="/l", run_id=GOOD_STAMP,
    )
    argv_empty_cat = sync.build_rsync_command(
        host="h", remote_root="/r", local_root="/l", run_id=GOOD_STAMP,
        category="",
    )
    assert argv_no_cat == argv_empty_cat


def test_build_rsync_command_category_with_surrounding_slashes_is_trimmed():
    argv = sync.build_rsync_command(
        host="h", remote_root="/r", local_root="/l",
        run_id=GOOD_STAMP, category="/alpha_off/runs/",
    )
    # No double slashes in the assembled paths.
    assert "//" not in argv[-2][len("h:"):]
    assert "//" not in argv[-1]
    assert argv[-2] == f"h:/r/alpha_off/runs/{GOOD_STAMP}/"
    assert argv[-1] == f"/l/alpha_off/runs/{GOOD_STAMP}/"


def test_build_rsync_command_extra_flags_inserted_before_paths():
    argv = sync.build_rsync_command(
        host="h", remote_root="/r", local_root="/l", run_id=GOOD_STAMP,
        extra_flags=("--bwlimit=2048",),
    )
    i = argv.index("--bwlimit=2048")
    assert i == len(argv) - 3, "extra_flags must sit just before src/dst"


def test_build_rsync_command_bad_run_id_raises():
    with pytest.raises(ValueError, match="run_id must match"):
        sync.build_rsync_command(
            host="h", remote_root="/r", local_root="/l", run_id="latest",
        )


def test_build_rsync_command_unknown_profile_raises():
    with pytest.raises(ValueError, match="unknown pull profile"):
        sync.build_rsync_command(
            host="h", remote_root="/r", local_root="/l", run_id=GOOD_STAMP,
            profile="bogus",
        )


# ---------------------------------------------------------------------------
# resolve_run_id
# ---------------------------------------------------------------------------

def test_resolve_run_id_passthrough_good_stamp():
    # ssh_runner must NOT be called when run_id is already a valid stamp.
    def _no_call(_argv):
        raise AssertionError("ssh_runner was called for a well-formed stamp")

    assert sync.resolve_run_id(
        GOOD_STAMP, ssh_runner=_no_call, remote_root="/r",
    ) == GOOD_STAMP


def test_resolve_run_id_bad_stamp_raises():
    with pytest.raises(ValueError, match="run_id must be 'latest' or match"):
        sync.resolve_run_id(
            "yesterdays-run", ssh_runner=lambda a: [], remote_root="/r",
        )


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


def test_resolve_run_id_latest_skips_non_run_dir_entries():
    """Stray files (a tarball snapshot, a README) must not be picked."""
    def _ssh(_argv):
        return [
            "README",
            "run_20260528T100000Z",
            "runs.tar.gz",
            "run_20260528T143052Z",
            "foo",
        ]

    out = sync.resolve_run_id(
        "latest", ssh_runner=_ssh, remote_root="/r",
    )
    assert out == "run_20260528T143052Z"


def test_resolve_run_id_latest_no_match_raises():
    with pytest.raises(ValueError, match="no run_<UTC>Z entries"):
        sync.resolve_run_id(
            "latest", ssh_runner=lambda a: ["README", "foo"], remote_root="/r",
        )


def test_resolve_run_id_latest_with_category_lists_subdir():
    """`latest` + category must `ls` <remote_root>/<category>, not <remote_root>."""
    calls = []

    def _ssh(argv):
        calls.append(list(argv))
        return [GOOD_STAMP]

    out = sync.resolve_run_id(
        "latest", ssh_runner=_ssh,
        remote_root="/gpfs/scratch/awills/xcquinox_runs",
        category="alpha_off/runs",
    )
    assert out == GOOD_STAMP
    assert calls == [["ls", "-1tr",
                      "/gpfs/scratch/awills/xcquinox_runs/alpha_off/runs"]]


def test_resolve_run_id_latest_empty_category_back_compat():
    """`category=""` must reproduce the pre-category ls path."""
    calls = []

    def _ssh(argv):
        calls.append(list(argv))
        return [GOOD_STAMP]

    sync.resolve_run_id(
        "latest", ssh_runner=_ssh, remote_root="/r", category="",
    )
    assert calls == [["ls", "-1tr", "/r"]]


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


def test_discover_runs_ignores_non_run_basenames():
    """`find` matches `run_*Z` glob, but a stray dir like `run_old_format`
    that happens to match -name 'run_*Z' must still be rejected by the
    stricter run-id regex."""
    def _ssh(_argv):
        return [
            "/r/alpha_off/runs/run_20260601T120000Z",
            "/r/alpha_off/runs/run_old_or_typoZ",   # matches -name but not regex
            "/r/alpha_off/runs/run_20260528T140000Z",
            "",  # blank line — must be skipped
        ]

    groups = sync.discover_runs(ssh_runner=_ssh, remote_root="/r")
    assert groups == {"alpha_off/runs": [
        "run_20260528T140000Z", "run_20260601T120000Z",
    ]}


def test_discover_runs_skips_paths_outside_remote_root():
    """Defensive — symlinks / bind mounts can surface paths outside the root."""
    def _ssh(_argv):
        return [
            "/r/alpha_off/runs/run_20260601T120000Z",
            "/somewhere/else/run_20260530T100000Z",
        ]

    groups = sync.discover_runs(ssh_runner=_ssh, remote_root="/r")
    assert groups == {"alpha_off/runs": ["run_20260601T120000Z"]}


def test_discover_runs_empty_result_is_empty_dict():
    assert sync.discover_runs(
        ssh_runner=lambda _a: [], remote_root="/r",
    ) == {}


def test_discover_runs_rejects_zero_or_negative_depth():
    with pytest.raises(ValueError, match="max_depth must be >= 1"):
        sync.discover_runs(
            ssh_runner=lambda _a: [], remote_root="/r", max_depth=0,
        )


def test_discover_runs_passes_depth_to_find():
    captured = {}

    def _ssh(argv):
        captured["argv"] = list(argv)
        return []

    sync.discover_runs(ssh_runner=_ssh, remote_root="/r", max_depth=5)
    assert "-maxdepth" in captured["argv"]
    i = captured["argv"].index("-maxdepth")
    assert captured["argv"][i + 1] == "5"
    # also -prune to stop descent into matched dirs
    assert "-prune" in captured["argv"]
    assert "-print" in captured["argv"]


def test_discover_runs_default_depth_catches_polarized_layout():
    """Regression: the production SeaWulf layout has ``polarized/<axis>/runs/
    run_<UTC>Z`` at depth 4 below the scratch root. The default ``max_depth``
    MUST catch that — an earlier default of 3 silently dropped the polarized
    branch (user-reported: `"the alpha_on/runs is found, but not the
    polarized/alpha_on/runs"`).

    This test pins the default at >= 4 forever. If you lower it, this test
    fails and tells you exactly why.
    """
    captured: dict[str, list[str]] = {}

    def _ssh(argv):
        captured["argv"] = list(argv)
        # Simulate a `find -maxdepth N` listing that includes a depth-4 path.
        # Whether the depth-4 entry is returned mimics the cluster's `find`:
        # it would only show up if -maxdepth >= 4.
        i = argv.index("-maxdepth")
        depth_cap = int(argv[i + 1])
        out = [
            "/gpfs/scratch/awills/xcquinox_runs/alpha_off/runs/run_20260601T120000Z",
            "/gpfs/scratch/awills/xcquinox_runs/alpha_on/runs/run_20260530T100000Z",
        ]
        if depth_cap >= 4:
            out.append(
                "/gpfs/scratch/awills/xcquinox_runs/polarized/alpha_on/runs/"
                "run_20260527T143052Z"
            )
        return out

    groups = sync.discover_runs(  # NO max_depth override — exercises the default
        ssh_runner=_ssh,
        remote_root="/gpfs/scratch/awills/xcquinox_runs",
    )
    # The default `find -maxdepth` must be at least 4 so the polarized branch
    # is in the listing.
    i = captured["argv"].index("-maxdepth")
    actual_default = int(captured["argv"][i + 1])
    assert actual_default >= 4, (
        f"discover_runs default max_depth={actual_default} is too shallow — "
        "polarized/<axis>/runs/run_<UTC>Z is at depth 4 and would be missed"
    )
    # And confirm the polarized run is actually present in the grouping.
    assert "polarized/alpha_on/runs" in groups
    assert groups["polarized/alpha_on/runs"] == ["run_20260527T143052Z"]
    # Sanity: the shallower entries still group correctly.
    assert groups["alpha_off/runs"] == ["run_20260601T120000Z"]
    assert groups["alpha_on/runs"] == ["run_20260530T100000Z"]


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


def test_format_ssh_stderr_tail_strips_sbu_banner():
    """The user's actual failure mode: real `ls:` error buried under the
    SBU banner. The tail formatter must surface just the error."""
    out = sync.format_ssh_stderr_tail(_SBU_BANNER_FIXTURE, n=1)
    assert out == (
        "ls: cannot access /gpfs/scratch/awills/xcquinox_runs/runs: "
        "No such file or directory"
    )


def test_format_ssh_stderr_tail_n_lines():
    out = sync.format_ssh_stderr_tail(
        "first\nsecond\nthird\nfourth\n", n=2,
    )
    assert out == "third\nfourth"


def test_format_ssh_stderr_tail_skips_blank_lines():
    out = sync.format_ssh_stderr_tail("a\n\n\nb\n\nc\n", n=2)
    assert out == "b\nc"


def test_format_ssh_stderr_tail_empty_input_returns_empty():
    assert sync.format_ssh_stderr_tail("") == ""
    assert sync.format_ssh_stderr_tail("\n\n\n") == ""


# ---------------------------------------------------------------------------
# End-to-end filter canary (drives real rsync against a tmp fixture)
# ---------------------------------------------------------------------------

def _materialize_fake_run(root: Path) -> Path:
    """Build a tmp tree mirroring the on-disk artifacts the harness writes.

    Mirrors xcquinox/alec/cluster/__main__.py + _train_task.py + _eval_one_spec.py
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
    (spec / "model.eqx").write_bytes(b"FAKE_MODEL_CHECKPOINT_BLOB" * 100)
    (spec / "eval" / "per_molecule.json").write_text("[]\n")
    # Pretrain
    pre = run / "pretrain" / "deep_combined_attn"
    pre.mkdir(parents=True)
    (pre / "pretrain_metadata.json").write_text('{"steps": 1000}\n')
    (pre / "losses_x.npy").write_bytes(b"\x93NUMPY")
    (pre / "losses_c.npy").write_bytes(b"\x93NUMPY")
    (pre / "xnet.eqx").write_bytes(b"FAKE_XNET_BLOB" * 100)
    (pre / "cnet.eqx").write_bytes(b"FAKE_CNET_BLOB" * 100)
    # Junk that must NOT be pulled by summaries
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
        "pretrain/deep_combined_attn/pretrain_metadata.json",
        "pretrain/deep_combined_attn/losses_x.npy",
        "pretrain/deep_combined_attn/losses_c.npy",
    ]
    for rel in must_have:
        assert (dest / rel).is_file(), (
            f"summaries.filter dropped an artifact it should keep: {rel} "
            "(if this artifact was renamed, update filters/summaries.filter)"
        )

    # --- must NOT be present (excluded tier) -----------------------------
    must_not_have = [
        "checkpoints/spec_0000/model.eqx",
        "checkpoints/spec_0000/model.eqx.gen1",
        "pretrain/deep_combined_attn/xnet.eqx",
        "pretrain/deep_combined_attn/cnet.eqx",
        "logs",
        "logs/train_42_0.out",
        "scripts",
        "scripts/train_array.sbatch",
        "specs",
        "specs/spec_0000.spec",
    ]
    for rel in must_not_have:
        assert not (dest / rel).exists(), (
            f"summaries.filter leaked an artifact it should exclude: {rel} "
            "(this indicates the filter is over-permissive; tighten the "
            "include rules or move this to filters/full.filter)"
        )


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
def test_full_filter_canary_excludes_only_logs(tmp_path, fake_remote_root):
    """The 'full' profile mirrors the run dir minus /logs/."""
    local_root = tmp_path / "local_full"
    local_root.mkdir()
    (local_root / GOOD_STAMP).mkdir()

    argv = sync.build_rsync_command(
        host="",
        remote_root=str(fake_remote_root),
        local_root=str(local_root),
        run_id=GOOD_STAMP,
        profile="full",
    )
    completed = subprocess.run(
        argv, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr

    dest = local_root / GOOD_STAMP
    # Big artifacts are present
    assert (dest / "checkpoints/spec_0000/model.eqx").is_file()
    assert (dest / "pretrain/deep_combined_attn/xnet.eqx").is_file()
    assert (dest / "pretrain/deep_combined_attn/cnet.eqx").is_file()
    assert (dest / "scripts/train_array.sbatch").is_file()
    assert (dest / "specs/spec_0000.spec").is_file()
    # Only /logs/ is excluded
    assert not (dest / "logs").exists()


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
def test_summaries_filter_dry_run_transfers_nothing(tmp_path, fake_remote_root):
    """``--dry-run`` must report success without writing any files."""
    local_root = tmp_path / "local_dry"
    local_root.mkdir()
    (local_root / GOOD_STAMP).mkdir()

    argv = sync.build_rsync_command(
        host="",
        remote_root=str(fake_remote_root),
        local_root=str(local_root),
        run_id=GOOD_STAMP,
        profile="summaries",
        dry_run=True,
    )
    completed = subprocess.run(
        argv, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    # Dest dir was pre-created (mirrors cmd_pull), but should be otherwise empty.
    dest = local_root / GOOD_STAMP
    assert list(dest.iterdir()) == [], (
        "dry-run leaked files into the destination: "
        f"{[p.name for p in dest.iterdir()]}"
    )


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
def test_summaries_filter_canary_with_category(tmp_path):
    """End-to-end with a multi-segment category mirrored to local.

    Builds <remote>/polarized/alpha_on/run_<stamp>/{manifest.json, model.eqx,...}
    and confirms the summaries filter still works *and* the local dest
    mirrors the category layout (collision protection).
    """
    remote_root = tmp_path / "remote"
    category_dir = remote_root / "polarized" / "alpha_on"
    category_dir.mkdir(parents=True)
    _materialize_fake_run(category_dir)  # creates category_dir/<GOOD_STAMP>/...

    local_root = tmp_path / "local"
    local_root.mkdir()
    # cmd_pull would create the mirrored category path; do the same here.
    local_dest_parent = local_root / "polarized" / "alpha_on" / GOOD_STAMP
    local_dest_parent.mkdir(parents=True)

    argv = sync.build_rsync_command(
        host="",
        remote_root=str(remote_root),
        local_root=str(local_root),
        run_id=GOOD_STAMP,
        category="polarized/alpha_on",
        profile="summaries",
    )
    completed = subprocess.run(
        argv, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr

    dest = local_root / "polarized" / "alpha_on" / GOOD_STAMP
    # Category-mirrored path is populated...
    assert (dest / "manifest.json").is_file()
    assert (dest / "checkpoints/spec_0000/eval_df.csv").is_file()
    # ...and the summaries filter still excludes the big stuff.
    assert not (dest / "checkpoints/spec_0000/model.eqx").exists()
    assert not (dest / "logs").exists()
    # The un-categorized local root must NOT have been touched.
    assert not (local_root / GOOD_STAMP).exists(), (
        "category-mirrored pull leaked a top-level (non-categorized) dest"
    )
