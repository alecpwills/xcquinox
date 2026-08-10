"""Tests for the SLURM email-digest machinery (job_digest.sh).

The digest exists so a cluster failure can be diagnosed from the email alone,
with no shell access: the job's rc, the error-pattern lines, the log tail and
the report tail are composed in memory, mailed, and best-effort written
beside the log. These tests execute the helper through real bash -- arming,
the EXIT trap, the TERM (wall-limit) trap, rc preservation, the
duplicate-send guard, the mailer timeout, and the unwritable-digest and
stale-digest paths -- and pin the wiring inside each production sbatch.

The fixture log is deliberately longer than the digest's tail window, with
the error block early and a marker inside the tail window, so the
error-pattern section and the tail section each carry content the other
cannot supply and a mutation dropping either one fails.
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

# Error lines early, then >100 filler lines, then a tail-only marker: the
# traceback is only reachable through the error-pattern block, the marker
# only through the tail-100 block.
_FAKE_LOG = (
    "[job] START\n"
    "Traceback (most recent call last):\n"
    '  File "hpcjobs/x.py", line 1, in leg1\n'
    "FloatingPointError: invalid value in divide\n"
    "[job] FATAL: leg 1 non-finite\n"
    + "".join(f"[job] progress line {i}\n" for i in range(120))
    + "TAIL_MARKER_LINE\n[job] END\n"
)


def _bash(script, cwd, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(["bash", "-c", script], cwd=cwd,
                          capture_output=True, text=True, env=env)


def _mail_stub_dir(tmp_path, behavior=""):
    """A PATH dir whose `mail` records every invocation and its stdin."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "mail"
    stub.write_text(
        "#!/bin/sh\n"
        f"echo \"$@\" >> '{tmp_path}/mail_args'\n"
        f"cat >> '{tmp_path}/mail_body'\n"
        + behavior)
    stub.chmod(0o755)
    return bin_dir


def test_shell_syntax_all_files():
    for f in [HELPER] + SBATCHES:
        r = subprocess.run(["bash", "-n", f], capture_output=True, text=True)
        assert r.returncode == 0, f"{f}: {r.stderr}"


def test_digest_carries_both_error_block_and_tail_and_rc_preserved(tmp_path):
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
    # Only the error-pattern block can carry these (they are >100 lines
    # from the end of the log):
    assert "Traceback" in text and "FloatingPointError" in text
    assert "leg 1 non-finite" in text
    # Only the tail block can carry this:
    assert "TAIL_MARKER_LINE" in text
    # A populated error block must not be followed by the empty-block line
    # (the pipefail-on-head contradiction).
    assert "(no error-pattern matches)" not in text
    assert "grad_finite" in text, "report tail missing from digest"


def test_error_patterns_cover_signal_deaths_and_slurmstepd(tmp_path):
    log = tmp_path / "fake.log"
    log.write_text(
        "Segmentation fault (core dumped)\n"
        "slurmstepd: error: Exceeded step memory limit\n"
        "Error: something failed at line start\n"
        "Fatal Python error: Aborted\n"
        + "".join(f"filler {i}\n" for i in range(120)))
    r = _bash(
        f"set -uo pipefail\nsource '{HELPER}'\n"
        f"job_digest_arm t '' '{log}' ''\nexit 1\n", tmp_path)
    assert r.returncode == 1
    text = log.with_name(log.name + ".digest.txt").read_text()
    head = text.split("--- last 100")[0]           # the error block only
    assert "Segmentation fault" in head
    assert "Exceeded step memory limit" in head
    assert "Error: something failed" in head
    assert "Fatal Python error" in head


def test_digest_on_term_is_single_email_and_rc_143(tmp_path):
    log = tmp_path / "fake.log"
    log.write_text(_FAKE_LOG)
    bin_dir = _mail_stub_dir(tmp_path)
    r = _bash(
        f"set -uo pipefail\nexport PATH='{bin_dir}':$PATH\n"
        f"source '{HELPER}'\n"
        f"job_digest_arm t2 x@example.edu '{log}' ''\n"
        "kill -TERM $$\necho UNREACHABLE\n", tmp_path)
    assert r.returncode == 143
    assert "UNREACHABLE" not in r.stdout
    digest = log.with_name(log.name + ".digest.txt")
    assert digest.read_text().count("=== t2 job") == 1
    # The mailer itself must have been invoked exactly once: TERM fires the
    # digest, and the EXIT trap that follows must be suppressed by the guard.
    args = (tmp_path / "mail_args").read_text()
    assert args.count("x@example.edu") == 1, \
        f"duplicate digest email on the TERM-then-EXIT path: {args!r}"


def test_second_run_truncates_rather_than_appends(tmp_path):
    log = tmp_path / "fake.log"
    log.write_text(_FAKE_LOG)
    script = (
        f"set -uo pipefail\nsource '{HELPER}'\n"
        f"job_digest_arm t6 '' '{log}' ''\nexit 0\n")
    assert _bash(script, tmp_path).returncode == 0
    assert _bash(script, tmp_path).returncode == 0
    digest = log.with_name(log.name + ".digest.txt")
    assert digest.read_text().count("=== t6 job") == 1, \
        "digest file accumulated across runs (append instead of truncate)"


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
    log = tmp_path / "fake.log"
    log.write_text(_FAKE_LOG)
    bin_dir = _mail_stub_dir(tmp_path)
    r = _bash(
        f"set -uo pipefail\nexport PATH='{bin_dir}':$PATH\n"
        f"source '{HELPER}'\n"
        f"job_digest_arm t4 '' '{log}' ''\nexit 0\n", tmp_path)
    assert r.returncode == 0
    assert log.with_name(log.name + ".digest.txt").is_file()
    assert not (tmp_path / "mail_args").exists(), \
        "mailer invoked with empty recipient"


def test_recipient_gets_fresh_body_even_when_digest_file_is_unwritable(tmp_path):
    """The email is composed in memory: a pre-existing unwritable digest
    file (full disk / quota / permissions) must neither suppress the email
    nor let yesterday's content ship under today's subject."""
    log = tmp_path / "fake.log"
    log.write_text(_FAKE_LOG)
    stale = log.with_name(log.name + ".digest.txt")
    stale.write_text("STALE-CONTENT-FROM-A-PREVIOUS-RUN\n")
    stale.chmod(0o444)
    bin_dir = _mail_stub_dir(tmp_path)
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
    assert "STALE-CONTENT" not in body, \
        "stale on-disk digest was mailed under a fresh subject"
    stale.chmod(0o644)


def test_hung_mailer_cannot_hold_the_job(tmp_path):
    """A mailer that never returns must be cut off by the timeout, not hold
    a finished job (and its exclusive node) until the wall limit. The
    timeout value is env-overridable precisely so this test runs in ~1 s."""
    log = tmp_path / "fake.log"
    log.write_text(_FAKE_LOG)
    bin_dir = _mail_stub_dir(tmp_path, behavior="sleep 300\n")
    r = _bash(
        f"set -uo pipefail\nexport PATH='{bin_dir}':$PATH\n"
        f"source '{HELPER}'\n"
        f"job_digest_arm t7 x@example.edu '{log}' ''\nexit 0\n",
        tmp_path, env_extra={"JOB_DIGEST_MAIL_TIMEOUT": "1"})
    assert r.returncode == 0, \
        "job rc altered (or job hung) by a hanging mailer"
    assert (tmp_path / "mail_args").exists(), "mailer was never invoked"


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
        # Signal death discards a buffered final stdio block, so the driver
        # each job runs must be unbuffered or the digest tail lies about
        # how far the job got.
        for m in re.finditer(r"^python (\S+)", text, re.M):
            assert m.group(1) == "-u", \
                f"{f}: driver runs buffered python ({m.group(0)!r})"
