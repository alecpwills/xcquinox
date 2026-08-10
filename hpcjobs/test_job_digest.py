"""Tests for the SLURM email-digest machinery (job_digest.sh).

The digest exists so a cluster failure can be diagnosed from the email alone,
with no shell access: the job's rc, the error-context lines, the log tail and
the report tail are composed into one plain-text digest, mailed, and always
written beside the log. These tests execute the helper through real bash --
arming, the EXIT trap, the TERM (wall-limit) trap, rc preservation and the
duplicate-send guard -- and pin the wiring inside each production sbatch.
"""
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "job_digest.sh")
SBATCHES = [
    os.path.join(HERE, "dfs6311_nan_verify.sbatch"),
    os.path.join(HERE, "dfs6311_pretrained_holdout.sbatch"),
    os.path.join(HERE, "dfs6311_scan_pool.sbatch"),
]

_FAKE_LOG = """[job] START
Traceback (most recent call last):
  File "hpcjobs/x.py", line 1, in leg1
FloatingPointError: invalid value in divide
[job] FATAL: leg 1 non-finite
"""


def _bash(script, cwd):
    return subprocess.run(["bash", "-c", script], cwd=cwd,
                          capture_output=True, text=True)


def test_shell_syntax_all_files():
    for f in [HELPER] + SBATCHES:
        r = subprocess.run(["bash", "-n", f], capture_output=True, text=True)
        assert r.returncode == 0, f"{f}: {r.stderr}"


def test_digest_written_on_failure_exit_and_rc_preserved(tmp_path):
    log = tmp_path / "fake.log"
    log.write_text(_FAKE_LOG)
    rep = tmp_path / "fake_report.json"
    rep.write_text('{"leg1": [{"grad_finite": false}]}\n')
    r = _bash(
        f"set -uo pipefail\nsource '{HELPER}'\n"
        f"job_digest_arm testtag '' '{log}' '{tmp_path}/fake_*.json'\n"
        "exit 7\n", tmp_path)
    assert r.returncode == 7, "digest machinery must not change the job rc"
    digest = log.with_name(log.name + ".digest.txt")
    assert digest.is_file(), "digest file not written on nonzero exit"
    text = digest.read_text()
    assert "rc=7" in text
    assert "Traceback" in text and "FloatingPointError" in text
    assert "leg 1 non-finite" in text
    assert "grad_finite" in text, "report tail missing from digest"


def test_digest_on_term_is_single_and_rc_143(tmp_path):
    log = tmp_path / "fake.log"
    log.write_text(_FAKE_LOG)
    r = _bash(
        f"set -uo pipefail\nsource '{HELPER}'\n"
        f"job_digest_arm t2 '' '{log}' ''\n"
        "kill -TERM $$\necho UNREACHABLE\n", tmp_path)
    assert r.returncode == 143
    assert "UNREACHABLE" not in r.stdout
    digest = log.with_name(log.name + ".digest.txt")
    assert digest.read_text().count("=== t2 job") == 1, \
        "TERM followed by EXIT must not produce a duplicate digest"


def test_exit_trap_is_load_bearing(tmp_path):
    """A helper with the arming traps removed must produce NO digest -- so
    the trap lines, not some side effect, are what delivers the content."""
    stripped = tmp_path / "stripped.sh"
    lines = [l for l in open(HELPER)
             if not re.match(r"\s*trap ", l)]
    stripped.write_text("".join(lines))
    log = tmp_path / "fake.log"
    log.write_text(_FAKE_LOG)
    r = _bash(
        f"set -uo pipefail\nsource '{stripped}'\n"
        f"job_digest_arm t3 '' '{log}' ''\n"
        "exit 5\n", tmp_path)
    assert r.returncode == 5
    assert not log.with_name(log.name + ".digest.txt").exists()


def test_empty_recipient_never_invokes_a_mailer(tmp_path):
    """With no recipient the digest is still written but no mail transport
    runs (the local dry-run path; also the safety if the address is unset)."""
    log = tmp_path / "fake.log"
    log.write_text(_FAKE_LOG)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for m in ("mail", "mailx", "sendmail"):
        stub = bin_dir / m
        stub.write_text(f"#!/bin/sh\ntouch '{tmp_path}/{m}_CALLED'\n")
        stub.chmod(0o755)
    r = _bash(
        f"set -uo pipefail\nexport PATH='{bin_dir}':$PATH\n"
        f"source '{HELPER}'\n"
        f"job_digest_arm t4 '' '{log}' ''\nexit 0\n", tmp_path)
    assert r.returncode == 0
    assert log.with_name(log.name + ".digest.txt").is_file()
    called = [m for m in ("mail", "mailx", "sendmail")
              if (tmp_path / f"{m}_CALLED").exists()]
    assert called == [], f"mailer invoked with empty recipient: {called}"


def test_recipient_gets_digest_via_first_available_mailer(tmp_path):
    log = tmp_path / "fake.log"
    log.write_text(_FAKE_LOG)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "mail"
    stub.write_text(
        f"#!/bin/sh\necho \"$@\" > '{tmp_path}/mail_args'\n"
        f"cat > '{tmp_path}/mail_body'\n")
    stub.chmod(0o755)
    r = _bash(
        f"set -uo pipefail\nexport PATH='{bin_dir}':$PATH\n"
        f"source '{HELPER}'\n"
        f"job_digest_arm t5 someone@example.edu '{log}' ''\nexit 3\n",
        tmp_path)
    assert r.returncode == 3
    args = (tmp_path / "mail_args").read_text()
    assert "someone@example.edu" in args
    assert "rc=3" in args, f"subject must carry the rc: {args}"
    body = (tmp_path / "mail_body").read_text()
    assert "FloatingPointError" in body


def test_every_sbatch_arms_the_digest_with_its_own_log():
    for f in SBATCHES:
        text = open(f).read()
        assert "job_digest.sh" in text, f"{f}: helper not sourced"
        arm = re.search(r"job_digest_arm\s+(\S+)\s+(\S+)\s+\"\$LOG\"", text)
        assert arm, f"{f}: job_digest_arm not wired to $LOG"
        assert arm.group(2) == "alec.wills@stonybrook.edu", \
            f"{f}: digest recipient must be the SBU address"
        out = re.search(r"#SBATCH --output=(\S+)", text).group(1)
        log = re.search(r'\nLOG="([^"]+)"', text).group(1)
        # The LOG the digest reads must be the SAME file SLURM writes:
        # identical stems with %j <-> ${SLURM_JOB_ID:-manual} interchanged.
        assert out.replace("%j", "JOBID") == log.replace(
            "${SLURM_JOB_ID:-manual}", "JOBID"), \
            f"{f}: digest LOG {log!r} does not match --output {out!r}"
        assert re.search(r"--mail-type=\S*TIME_LIMIT", text), \
            f"{f}: TIME_LIMIT missing from --mail-type"
