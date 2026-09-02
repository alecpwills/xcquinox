"""Tests for xcquinox.alec.cluster.sync: the ``pull`` subcommand helpers.

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


def test_filter_file_path_full_exists():
    p = sync.filter_file_path("full")
    assert p.is_file()
    body = p.read_text()
    # 'full' mirrors the entire run dir -- no exclusions, logs included.
    assert "+ /***" in body
    assert "- /logs/" not in body, (
        "the 'full' profile must NOT exclude /logs/ -- the SLURM logs are "
        "needed to diagnose failed runs off-cluster"
    )


def test_filter_file_path_unknown_profile_raises():
    with pytest.raises(ValueError, match="unknown pull profile"):
        sync.filter_file_path("bogus")


# ---------------------------------------------------------------------------
# build_rsync_command: pure argv shape
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
# build_rsync_command: category (multi-series layout)
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
    # Local dest mirrors the remote category layout, collision protection.
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


def test_build_rsync_command_empty_spec_indices_is_noop():
    """Default and explicit-empty must match the original (pre-flag) shape."""
    argv_default = sync.build_rsync_command(
        host="h", remote_root="/r", local_root="/l",
        run_id=GOOD_STAMP, profile="full",
    )
    argv_empty = sync.build_rsync_command(
        host="h", remote_root="/r", local_root="/l",
        run_id=GOOD_STAMP, profile="full", spec_indices=(),
    )
    assert argv_default == argv_empty
    # And neither carries any --include= rules.
    assert not any(a.startswith("--include=") for a in argv_default)


def test_build_rsync_command_negative_spec_index_raises():
    with pytest.raises(ValueError, match="non-negative ints"):
        sync.build_rsync_command(
            host="h", remote_root="/r", local_root="/l",
            run_id=GOOD_STAMP, spec_indices=[0, -1, 5],
        )


def test_build_rsync_command_non_int_spec_index_raises():
    with pytest.raises(ValueError, match="non-negative ints"):
        sync.build_rsync_command(
            host="h", remote_root="/r", local_root="/l",
            run_id=GOOD_STAMP, spec_indices=["0", "1"],  # type: ignore[list-item]
        )


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
def test_spec_indices_canary_against_real_rsync(tmp_path):
    """End-to-end: a fixture with three spec dirs (0000, 0001, 0036), pull
    with spec_indices=[0, 36], confirm exactly 0000 + 0036 land locally."""
    remote_root = tmp_path / "remote"
    src_run = remote_root / GOOD_STAMP
    src_run.mkdir(parents=True)
    (src_run / "manifest.json").write_text('{"n_specs": 3}\n')
    for idx in (0, 1, 36):
        spec = src_run / "checkpoints" / f"spec_{idx:04d}"
        spec.mkdir(parents=True)
        # A model.eqx, exercising the spec filter on the weights tier (both
        # profiles carry this name; --specs is what must restrict it here).
        (spec / "model.eqx").write_bytes(b"FAKE_MODEL_BLOB" * 1000)
        (spec / "eval_df.csv").write_text("set,mae\ntraining_subset,1.0\n")

    local_root = tmp_path / "local"
    local_root.mkdir()
    (local_root / GOOD_STAMP).mkdir()

    argv = sync.build_rsync_command(
        host="", remote_root=str(remote_root), local_root=str(local_root),
        run_id=GOOD_STAMP, profile="full", spec_indices=[0, 36],
    )
    completed = subprocess.run(
        argv, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr

    dest = local_root / GOOD_STAMP / "checkpoints"
    assert (dest / "spec_0000" / "model.eqx").is_file()
    assert (dest / "spec_0036" / "model.eqx").is_file()
    # The requested 2; the rejected 1 must be entirely absent.
    assert not (dest / "spec_0001").exists(), (
        "spec_0001 leaked despite --specs=0,36, the rsync include/exclude "
        "order is wrong (the catch-all -exclude must come AFTER all -include "
        "rules so spec_0000 and spec_0036 win the first-match race)"
    )
    # Top-level manifest still arrives (full profile, not gated by --specs).
    assert (local_root / GOOD_STAMP / "manifest.json").is_file()


def test_summaries_canary_pretrain_certificate_transfers(tmp_path):
    """End-to-end: the summaries profile carries the per-arch fidelity
    certificate (the figure suite refuses to render an arch whose pulled run
    holds no readable certificate) beside pretrain_metadata.json, the loss
    curves and the pretrained networks the enhancement-factor figures read."""
    remote_root = tmp_path / "remote"
    arch_dir = remote_root / GOOD_STAMP / "pretrain" / "deep_3x16"
    arch_dir.mkdir(parents=True)
    (arch_dir / "fidelity_certificate.json").write_text('{"verdict": "PASS"}\n')
    (arch_dir / "pretrain_metadata.json").write_text('{"arch": "deep_3x16"}\n')
    (arch_dir / "losses_x.npy").write_bytes(b"\x93NUMPY_FAKE")
    (arch_dir / "xnet.eqx").write_bytes(b"FAKE_NET_BLOB" * 100)

    local_root = tmp_path / "local"
    local_root.mkdir()
    (local_root / GOOD_STAMP).mkdir()

    argv = sync.build_rsync_command(
        host="", remote_root=str(remote_root), local_root=str(local_root),
        run_id=GOOD_STAMP, profile="summaries",
    )
    completed = subprocess.run(
        argv, check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr

    dest = local_root / GOOD_STAMP / "pretrain" / "deep_3x16"
    assert (dest / "fidelity_certificate.json").is_file(), (
        "the summaries filter dropped the fidelity certificate; the figure "
        "suite then refuses every arch of a default pull as uncertified"
    )
    assert (dest / "pretrain_metadata.json").is_file()
    assert (dest / "losses_x.npy").is_file()
    assert (dest / "xnet.eqx").is_file(), (
        "the summaries filter dropped the pretrained exchange network; "
        "pretrain_fx_fc.py forward-evaluates it, and no summary table stands "
        "in for the network itself"
    )


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
            "",  # blank line, must be skipped
        ]

    groups = sync.discover_runs(ssh_runner=_ssh, remote_root="/r")
    assert groups == {"alpha_off/runs": [
        "run_20260528T140000Z", "run_20260601T120000Z",
    ]}


def test_discover_runs_skips_paths_outside_remote_root():
    """Defensive: symlinks / bind mounts can surface paths outside the root."""
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
    MUST catch that, an earlier default of 3 silently dropped the polarized
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

    groups = sync.discover_runs(  # NO max_depth override, exercises the default
        ssh_runner=_ssh,
        remote_root="/gpfs/scratch/awills/xcquinox_runs",
    )
    # The default `find -maxdepth` must be at least 4 so the polarized branch
    # is in the listing.
    i = captured["argv"].index("-maxdepth")
    actual_default = int(captured["argv"][i + 1])
    assert actual_default >= 4, (
        f"discover_runs default max_depth={actual_default} is too shallow, "
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
    ]
    for rel in must_not_have:
        assert not (dest / rel).exists(), (
            f"summaries.filter leaked an artifact it should exclude: {rel} "
            "(this indicates the filter is over-permissive; tighten the "
            "include rules or move this to filters/full.filter)"
        )


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
def test_full_filter_canary_mirrors_run_dir_including_logs(tmp_path, fake_remote_root):
    """The 'full' profile mirrors the ENTIRE run dir, logs included."""
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
    # The SLURM log tree now comes too (the whole point of 'full': diagnosing
    # failed runs off-cluster requires the .out logs).
    assert (dest / "logs" / "train_42_0.out").is_file()


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
    and confirms the summaries filter still works and the local dest
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
    assert (dest / "checkpoints/spec_0000/model_val_best.eqx").is_file()
    # ...and the summaries filter still excludes the big stuff.
    assert not (dest / "checkpoints/spec_0000/model_best.eqx").exists()
    assert not (dest / "logs").exists()
    # The un-categorized local root must NOT have been touched.
    assert not (local_root / GOOD_STAMP).exists(), (
        "category-mirrored pull leaked a top-level (non-categorized) dest"
    )


def test_build_rsync_command_spec_indices_emit_all_pad_widths():
    """The spec dir pad width = max(4, len(str(n_specs-1))) is unknown at pull
    time (the manifest is not loaded), so an include is emitted for EVERY
    plausible width (4..8). A single fixed :04d silently matched nothing once a
    grid had >9999 specs (dirs padded to width >=5)."""
    argv = sync.build_rsync_command(
        host="h", remote_root="/r", local_root="/l",
        run_id=GOOD_STAMP, profile="full", spec_indices=[48],
    )
    for width in range(4, 9):
        name = f"spec_{48:0{width}d}"
        assert f"--include=/checkpoints/{name}/" in argv, \
            f"missing width-{width} include {name}: {argv}"
        assert f"--include=/checkpoints/{name}/***" in argv
    # width-4 form still present (back-compat with <=9999-spec grids)
    assert "--include=/checkpoints/spec_0048/" in argv


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
def test_spec_indices_canary_width5_grid(tmp_path):
    """End-to-end #10 guard: a grid with >9999 specs pads dir names to width 5
    (spec_00048). Pull with spec_indices=[48] must still land it; the old
    :04d-only include (spec_0048) would have matched NOTHING."""
    remote_root = tmp_path / "remote"
    src_run = remote_root / GOOD_STAMP
    src_run.mkdir(parents=True)
    (src_run / "manifest.json").write_text('{"n_specs": 12000}\n')
    for idx in (48, 49):
        spec = src_run / "checkpoints" / f"spec_{idx:05d}"  # width-5 padding
        spec.mkdir(parents=True)
        (spec / "model.eqx").write_bytes(b"BLOB" * 100)

    local_root = tmp_path / "local"
    local_root.mkdir()
    (local_root / GOOD_STAMP).mkdir()

    argv = sync.build_rsync_command(
        host="", remote_root=str(remote_root), local_root=str(local_root),
        run_id=GOOD_STAMP, profile="full", spec_indices=[48],
    )
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

    dest = local_root / GOOD_STAMP / "checkpoints"
    assert (dest / "spec_00048" / "model.eqx").is_file(), \
        "width-5 spec dir did not land; the :04d-only include regressed"
    # The unrequested neighbor must stay out.
    assert not (dest / "spec_00049").exists()


# ---------------------------------------------------------------------------
# pull auto: quoted ssh transport, activity discovery, multi-run filter/argv
# ---------------------------------------------------------------------------

SECOND_STAMP = "run_20260831T011905Z"


def test_ssh_remote_command_quotes_globs_and_scripts(tmp_path):
    from xcquinox.alec.cluster.__main__ import _ssh_remote_command
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


def test_ssh_control_opts_shape_and_persist_validation():
    opts = sync.ssh_control_opts("/home/u/.ssh", 3600)
    assert opts == ("-o", "ControlMaster=auto",
                    "-o", "ControlPath=/home/u/.ssh/xcq-cm-%C",
                    "-o", "ControlPersist=3600")
    # ssh defines ControlPersist=0 as "persist forever", the opposite of
    # disable, so 0 (and negatives) must be refused here.
    with pytest.raises(ValueError):
        sync.ssh_control_opts("/home/u/.ssh", 0)
    with pytest.raises(ValueError):
        sync.ssh_control_opts("/home/u/.ssh", -5)


def test_ssh_control_opts_tokens_are_accepted_by_the_real_ssh(tmp_path):
    # Executed oracle: ssh must EXPAND the ControlPath, not die on an
    # unsupported percent token (ControlPath accepts %C but NOT %d --
    # a %d form shipped once and killed every connection with
    # "percent_expand: unknown key" before any network activity).
    if shutil.which("ssh") is None:
        pytest.skip("ssh executable not available")
    opts = sync.ssh_control_opts(str(tmp_path), 60)
    completed = subprocess.run(
        ["ssh", *opts, "-O", "check",
         "xcq-nonexistent-host.invalid"],
        capture_output=True, text=True)
    assert "percent_expand" not in completed.stderr, completed.stderr
    assert completed.returncode != 0  # no master exists; the probe still ran


def test_ssh_transport_arg_survives_spaced_socket_dirs():
    from xcquinox.alec.cluster.__main__ import _ssh_transport_arg
    import shlex as _shlex
    opts = sync.ssh_control_opts("/home/a user/.ssh", 3600)
    value = _ssh_transport_arg(opts)
    # rsync splits the -e value shell-style: the round trip must restore
    # every option word intact, spaces included.
    assert _shlex.split(value) == ["ssh", *opts]


def test_run_stamp_datetime_display_only():
    from datetime import timezone
    dt = sync.run_stamp_datetime(SECOND_STAMP)
    assert dt is not None and dt.tzinfo == timezone.utc
    assert (dt.year, dt.month, dt.day, dt.hour) == (2026, 8, 31, 1)
    assert sync.run_stamp_datetime("not_a_run") is None
    # Regex-valid but calendar-invalid stamps must yield None, not raise.
    assert sync.run_stamp_datetime("run_20260230T120000Z") is None


class _RecordingRunner:
    def __init__(self, lines):
        self.lines = list(lines)
        self.calls = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        return list(self.lines)


def test_discover_runs_with_activity_argv_and_parsing():
    lines = [
        "A /r/catA/runs/run_20260830T000000Z",
        "I /r/catA/runs/run_20260101T000000Z",
        "A /r/run_20260829T000000Z",
        "X /r/catA/runs/run_20260828T000000Z",   # unknown tag: skipped
        "A /r/catA/runs/runs.tar.gz",            # non-stamp basename: skipped
        "",
    ]
    runner = _RecordingRunner(lines)
    groups = sync.discover_runs_with_activity(
        ssh_runner=runner, remote_root="/r", active_within_epoch=1785000000)
    (argv,) = runner.calls
    assert argv[:10] == ["find", "/r", "-mindepth", "1", "-maxdepth", "5",
                         "-type", "d", "-name", "run_*Z"]
    assert argv[10] == "-prune"
    assert argv[11:14] == ["-exec", "sh", "-c"]
    script = argv[14]
    assert "-newermt @1785000000" in script
    assert '"A $1"' in script and '"I $1"' in script
    assert "-print -quit" in script and "| grep -q ." in script
    assert argv[15:] == ["_", "{}", ";"]
    assert groups == {
        "catA/runs": [("run_20260101T000000Z", False),
                      ("run_20260830T000000Z", True)],
        "": [("run_20260829T000000Z", True)],
    }


def test_discover_runs_with_activity_no_epoch_is_plain_listing():
    lines = ["/r/catA/runs/run_20260830T000000Z"]
    runner = _RecordingRunner(lines)
    groups = sync.discover_runs_with_activity(
        ssh_runner=runner, remote_root="/r", active_within_epoch=None)
    runner_plain = _RecordingRunner(lines)
    sync.discover_runs(ssh_runner=runner_plain, remote_root="/r")
    # Byte-identical find argv to the plain discovery when no horizon is set.
    assert runner.calls[0] == runner_plain.calls[0]
    assert groups == {"catA/runs": [("run_20260830T000000Z", True)]}


def test_build_multi_filter_summaries_prefix_expansion():
    text = sync.filter_file_path("summaries").read_text()
    p1 = f"catA/runs/{GOOD_STAMP}"
    p2 = f"catB/deep/runs/{SECOND_STAMP}"
    out = sync.build_multi_filter(text, [p1, p2])
    lines = out.splitlines()
    for anc in ("+ /catA/", "+ /catA/runs/", f"+ /{p1}/",
                "+ /catB/", "+ /catB/deep/", "+ /catB/deep/runs/", f"+ /{p2}/"):
        assert anc in lines, anc
    assert f"+ /{p1}/manifest.json" in lines
    assert f"+ /{p2}/checkpoints/spec_*/eval_holdout_val_best/***" in lines
    assert lines[-1] == "- *" and lines.count("- *") == 1
    # Packaged rule order is preserved within each run's expansion.
    assert lines.index(f"+ /{p1}/manifest.json") \
        < lines.index(f"+ /{p1}/checkpoints/")


def test_build_multi_filter_full_profile_and_rule_refusals():
    full_text = sync.filter_file_path("full").read_text()
    p1 = f"catA/runs/{GOOD_STAMP}"
    out = sync.build_multi_filter(full_text, [p1])
    lines = out.splitlines()
    assert f"+ /{p1}/***" in lines
    assert lines[-1] == "- *"
    with pytest.raises(ValueError, match="unsupported filter rule"):
        sync.build_multi_filter("+ checkpoints/\n- *\n", [p1])
    with pytest.raises(ValueError, match="terminal"):
        sync.build_multi_filter("- *\n+ /manifest.json\n", [p1])
    # Anchored excludes are part of the grammar (the shard-scratch drop):
    # re-emitted per run, packaged order preserved ahead of later includes.
    out_exc = sync.build_multi_filter(
        "- /logs/\n+ /checkpoints/\n- *\n", [p1]).splitlines()
    assert f"- /{p1}/logs/" in out_exc
    assert out_exc.index(f"- /{p1}/logs/") < out_exc.index(
        f"+ /{p1}/checkpoints/")
    # An UNANCHORED exclude stays refused.
    with pytest.raises(ValueError, match="unsupported filter rule"):
        sync.build_multi_filter("- logs/\n- *\n", [p1])
    with pytest.raises(ValueError):
        sync.build_multi_filter("+ /x\n- *\n", ["bad path/with space/run"])
    with pytest.raises(ValueError):
        sync.build_multi_filter("+ /x\n- *\n", [])


def test_build_multi_rsync_command_argv_shape():
    p1 = f"catA/runs/{GOOD_STAMP}"
    p2 = f"catB/runs/{SECOND_STAMP}"
    argv = sync.build_multi_rsync_command(
        host="hpc", remote_root="/scratch/root/", local_root="/tmp/dest",
        run_paths=[p1, p2], filter_path="/tmp/f.rules",
        dry_run=True, extra_flags=("-e", "ssh -o X=1"))
    single = sync.build_rsync_command(
        host="hpc", remote_root="/scratch/root", local_root="/tmp/dest",
        run_id=GOOD_STAMP)
    # Both builders share the same base-flag prefix (drift pin).
    n_base = 1 + len(sync._BASE_FLAGS)
    assert argv[:n_base] == single[:n_base] == ["rsync", *sync._BASE_FLAGS]
    assert argv[n_base] == "-R"
    assert "--filter=. /tmp/f.rules" in argv
    assert "--dry-run" in argv
    i_e = argv.index("-e")
    assert argv[i_e + 1] == "ssh -o X=1"
    assert argv[-3:] == [f"hpc:/scratch/root/./{p1}",
                         f"hpc:/scratch/root/./{p2}", "/tmp/dest/"]
    with pytest.raises(ValueError):
        sync.build_multi_rsync_command(
            host="h", remote_root="/r", local_root="/l",
            run_paths=["cat/runs/not_a_stamp"], filter_path="/f")
    argv_local = sync.build_multi_rsync_command(
        host="", remote_root="/r", local_root="/l",
        run_paths=[p1], filter_path="/f")
    assert argv_local[-2] == f"/r/./{p1}"


def test_multi_run_canary_against_real_rsync(tmp_path):
    if shutil.which("rsync") is None:
        pytest.skip("rsync executable not available")
    remote = tmp_path / "remote"
    p1 = f"catA/runs/{GOOD_STAMP}"
    p2 = f"catB/deep/runs/{SECOND_STAMP}"
    run_a = remote / p1
    spec = run_a / "checkpoints" / "spec_0000"
    (spec / "eval_holdout_val_best").mkdir(parents=True)
    (spec / "eval_holdout_val_best" / "test_set.csv").write_text("x")
    (spec / "model_val_best.eqx").write_bytes(b"W")
    (spec / "model_val_best.eqx.class.json").write_text("{}")
    (spec / "model_best.eqx").write_bytes(b"EXCLUDED")
    pre = run_a / "pretrain" / "arch"
    (pre / "xnet").mkdir(parents=True)
    (pre / "xnet.eqx").write_bytes(b"X")
    (pre / "fidelity_certificate.json").write_text("{}")
    (pre / "xnet" / "xc.eqx.100").write_bytes(b"SNAP")
    (run_a / "manifest.json").write_text("{}")
    (run_a / "logs").mkdir()
    (run_a / "logs" / "big.out").write_text("L")
    run_b = remote / p2
    run_b.mkdir(parents=True)
    (run_b / "manifest.json").write_text("{}")
    sibling = remote / "catA" / "runs" / "run_20260101T000000Z"
    sibling.mkdir(parents=True)
    (sibling / "manifest.json").write_text("{}")

    generated = sync.build_multi_filter(
        sync.filter_file_path("summaries").read_text(), [p1, p2])
    fpath = tmp_path / "gen.rules"
    fpath.write_text(generated)
    local = tmp_path / "local"
    local.mkdir()
    argv = sync.build_multi_rsync_command(
        host="", remote_root=str(remote), local_root=str(local),
        run_paths=[p1, p2], filter_path=str(fpath))
    completed = subprocess.run(argv, capture_output=True, text=True,
                               check=False)
    assert completed.returncode == 0, completed.stderr
    assert (local / p1 / "manifest.json").is_file()
    assert (local / p1 / "checkpoints/spec_0000/model_val_best.eqx").is_file()
    assert (local / p1 /
            "checkpoints/spec_0000/eval_holdout_val_best/test_set.csv").is_file()
    assert (local / p1 / "pretrain/arch/xnet.eqx").is_file()
    assert (local / p1 / "pretrain/arch/fidelity_certificate.json").is_file()
    assert (local / p2 / "manifest.json").is_file()
    assert not (local / p1 / "checkpoints/spec_0000/model_best.eqx").exists()
    assert not (local / p1 / "pretrain/arch/xnet/xc.eqx.100").exists()
    assert not (local / p1 / "logs").exists()
    assert not (local / "catA/runs/run_20260101T000000Z").exists()


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
    from xcquinox.alec.cluster import __main__ as cm_mod

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


def test_cmd_pull_auto_scope_composition(monkeypatch, tmp_path):
    calls = []
    find_lines = [f"A /scr/root/dfs_step7/catA/runs/{GOOD_STAMP}"]
    cm_mod = _install_fake_run(monkeypatch, calls, find_lines=find_lines)
    rc = cm_mod.cmd_pull(_auto_args(tmp_path, category="dfs_step7"))
    assert rc == 0
    (ssh_argv,) = [a for a in calls if a[0] == "ssh" and "-O" not in a]
    assert "/scr/root/dfs_step7" in ssh_argv[-1]
    (argv,) = [a for a in calls if a[0] == "rsync"]
    src = f"hpc:/scr/root/./dfs_step7/catA/runs/{GOOD_STAMP}"
    assert src in argv
    # The -R mirror (dest + path after /./) equals the single-mode dest.
    single = sync.build_rsync_command(
        host="hpc", remote_root="/scr/root",
        local_root=str(tmp_path / "local"), run_id=GOOD_STAMP,
        category="dfs_step7/catA/runs")
    assert single[-1] == (str(tmp_path / "local")
                          + f"/dfs_step7/catA/runs/{GOOD_STAMP}/")
    assert argv[-1] == str(tmp_path / "local") + "/"


def test_cmd_pull_auto_refusals(monkeypatch, tmp_path, capsys):
    calls = []
    cm_mod = _install_fake_run(monkeypatch, calls, find_lines=[])
    assert cm_mod.cmd_pull(_auto_args(tmp_path, specs="0,1")) == 1
    assert "--specs" in capsys.readouterr().out
    assert cm_mod.cmd_pull(_auto_args(tmp_path, days=-1.0)) == 1
    assert "--days" in capsys.readouterr().out
    assert cm_mod.cmd_pull(_auto_args(tmp_path, ssh_persist=0)) == 1
    assert "--ssh-persist" in capsys.readouterr().out
    assert calls == [], "refusals must fire before any subprocess"


def test_cmd_pull_auto_sanity_gate(monkeypatch, tmp_path):
    find_lines = [f"A /scr/root/c{i}/runs/{GOOD_STAMP}" for i in range(16)]
    calls = []
    cm_mod = _install_fake_run(monkeypatch, calls, find_lines=find_lines)
    assert cm_mod.cmd_pull(_auto_args(tmp_path)) == 1
    assert [a for a in calls if a[0] == "rsync"] == []
    calls.clear()
    assert cm_mod.cmd_pull(_auto_args(tmp_path, yes=True)) == 0
    assert len([a for a in calls if a[0] == "rsync"]) == 1
    calls.clear()
    assert cm_mod.cmd_pull(_auto_args(tmp_path, dry_run=True)) == 0
    assert len([a for a in calls if a[0] == "rsync"]) == 1


def test_cmd_pull_auto_rc24_maps_to_success(monkeypatch, tmp_path):
    find_lines = [f"A /scr/root/catA/runs/{GOOD_STAMP}"]
    calls = []
    cm_mod = _install_fake_run(monkeypatch, calls, find_lines=find_lines,
                               rsync_rc=24)
    assert cm_mod.cmd_pull(_auto_args(tmp_path)) == 0
    calls.clear()
    cm_mod = _install_fake_run(monkeypatch, calls, find_lines=find_lines,
                               rsync_rc=23)
    assert cm_mod.cmd_pull(_auto_args(tmp_path)) == 23


def test_cmd_pull_single_mode_control_master(monkeypatch, tmp_path):
    calls = []
    cm_mod = _install_fake_run(monkeypatch, calls, find_lines=[])
    args = _auto_args(tmp_path, run_id=GOOD_STAMP, category="catA/runs")
    (tmp_path / "local").mkdir(exist_ok=True)
    assert cm_mod.cmd_pull(args) == 0
    # Explicit stamp: no discovery ssh; the rsync carries the CM transport.
    assert [a for a in calls if a[0] == "ssh" and "-O" not in a] == []
    (argv,) = [a for a in calls if a[0] == "rsync"]
    i_e = argv.index("-e")
    assert "ControlMaster=auto" in argv[i_e + 1]
    calls.clear()
    args = _auto_args(tmp_path, run_id=GOOD_STAMP, category="catA/runs",
                      no_control_master=True)
    assert cm_mod.cmd_pull(args) == 0
    (argv,) = [a for a in calls if a[0] == "rsync"]
    assert "-e" not in argv
    assert [a for a in calls if a[0] == "ssh"] == []


def test_cmd_pull_auto_pulls_every_active_run_in_a_category(monkeypatch,
                                                             tmp_path):
    # Two ACTIVE runs in ONE category: both must be pulled -- never
    # latest-only, a dead-or-stale newer launch must not mask other work.
    find_lines = [f"A /scr/root/catA/runs/{GOOD_STAMP}",
                  f"A /scr/root/catA/runs/{SECOND_STAMP}"]
    calls = []
    cm_mod = _install_fake_run(monkeypatch, calls, find_lines=find_lines)
    assert cm_mod.cmd_pull(_auto_args(tmp_path)) == 0
    (argv,) = [a for a in calls if a[0] == "rsync"]
    assert f"hpc:/scr/root/./catA/runs/{GOOD_STAMP}" in argv
    assert f"hpc:/scr/root/./catA/runs/{SECOND_STAMP}" in argv


def test_cmd_pull_auto_sanity_gate_boundary(monkeypatch, tmp_path):
    # Exactly the gate value passes without --yes; one more requires it.
    calls = []
    at_gate = [f"A /scr/root/c{i}/runs/{GOOD_STAMP}" for i in range(15)]
    cm_mod = _install_fake_run(monkeypatch, calls, find_lines=at_gate)
    assert cm_mod.cmd_pull(_auto_args(tmp_path)) == 0
    assert len([a for a in calls if a[0] == "rsync"]) == 1


def test_cmd_pull_single_mode_rc24_stays_a_failure(monkeypatch, tmp_path):
    # The vanished-files mapping is an AUTO-mode semantic only; single-run
    # pulls keep rsync's own exit code.
    calls = []
    cm_mod = _install_fake_run(monkeypatch, calls, find_lines=[],
                               rsync_rc=24)
    args = _auto_args(tmp_path, run_id=GOOD_STAMP, category="catA/runs")
    assert cm_mod.cmd_pull(args) == 24


def test_cmd_pull_auto_depth_and_filter_refusals(monkeypatch, tmp_path,
                                                 capsys):
    calls = []
    cm_mod = _install_fake_run(monkeypatch, calls, find_lines=[])
    assert cm_mod.cmd_pull(_auto_args(tmp_path, depth=0)) == 1
    assert "--depth" in capsys.readouterr().out
    assert calls == []
    # A packaged filter gaining a non-conforming rule surfaces as the CLI
    # error line, not a traceback.
    find_lines = [f"A /scr/root/catA/runs/{GOOD_STAMP}"]
    cm_mod = _install_fake_run(monkeypatch, calls, find_lines=find_lines)
    monkeypatch.setattr(
        cm_mod._sync, "build_multi_filter",
        lambda *a, **k: (_ for _ in ()).throw(
            ValueError("unsupported filter rule at line 1")))
    assert cm_mod.cmd_pull(_auto_args(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "unsupported filter rule" in out
    assert [a for a in calls if a[0] == "rsync"] == []


def test_pull_parser_accepts_auto_flags():
    from xcquinox.alec.cluster.__main__ import _build_parser
    parser = _build_parser()
    args = parser.parse_args(
        ["pull", "auto", "--days", "10", "--ssh-persist", "600",
         "--depth", "6", "--yes", "--no-control-master"])
    assert args.run_id == "auto"
    assert args.days == 10.0
    assert args.ssh_persist == 600
    assert args.depth == 6
    assert args.yes is True
    assert args.no_control_master is True


def test_pull_inventory_counts(tmp_path):
    from xcquinox.alec.cluster.__main__ import _pull_inventory
    run = tmp_path / "run"
    for i in range(2):
        spec = run / "checkpoints" / f"spec_{i:04d}"
        spec.mkdir(parents=True)
        (spec / "model_val_best.eqx").write_bytes(b"W")
    (run / "checkpoints/spec_0000/eval_holdout_val_best").mkdir()
    (run / "checkpoints/spec_0000/eval_holdout").mkdir()
    pre = run / "pretrain" / "arch"
    pre.mkdir(parents=True)
    (pre / "xnet.eqx").write_bytes(b"X")
    (pre / "fidelity_certificate.json").write_text("{}")
    line = _pull_inventory(run)
    assert line == ("val-best weights 2 | val-best evals 1 | holdout evals 1 "
                    "| pretrain xnets 1 | certificates 1")
