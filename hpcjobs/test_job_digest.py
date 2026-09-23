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
