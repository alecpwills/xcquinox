"""Tests for the per-architecture workflow-matrix job script.

Four things are pinned here. The SLURM SURFACE: the standing mail directives,
the house shell idiom and a single-node one-task allocation whose thread cap
comes from SLURM rather than from a literal. The WALL DERIVATION: the request
in the header is recomputed from the script's own defaults and the matrix's own
stage count, so a knob that moves without the request moving is a failure here
rather than a job killed at its wall with no report. The COMMAND SURFACE: every
flag the script passes is checked against ``workflow_matrix.main``'s argparse,
read out of the real parser, because a renamed flag turns a 24 h allocation
into a usage error. The SHELL BEHAVIOUR: the script is EXECUTED against a stub
interpreter -- environment defaulting, the batch split, the refusals of a bad
knob or a missing cached input, and the propagation of the matrix's exit code
to SLURM -- since none of that can be established by reading the text.

The stub interpreter answers the three invocations the script makes (the
contract probe, ``--list``, and the matrix run), records the argument vector it
was given and exits with a code the test chooses. It answers the contract probe
with whatever contract the test wants the checkout to have, which is how a
partially synced checkout is exercised without one. Nothing here runs the
matrix, and nothing here needs SLURM.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_SBATCH = _HERE / "workflow_matrix.sbatch"

#: Timed subprocesses the matrix runs per architecture: the stages of
#: ``STAGE_ORDER`` plus the architecture's oracle selection. The wall
#: derivation in the script's header is this number times the per-stage cap.
_ORACLE_STAGES = 1


def _sbatch_text() -> str:
    return _SBATCH.read_text()


def _default_of(name: str) -> str:
    """The default the script gives an environment knob, from its own text.

    Reading the value out of the script rather than restating it is what makes
    the wall-derivation test a check and not a duplicate.
    """
    text = _sbatch_text()
    # The default may itself contain an expansion (the work root carries
    # ${STAMP}), so the alternation admits one nested ${...} and the match is
    # closed on the quote that ends the assignment.
    match = re.search(rf'\$\{{{name}:-((?:\$\{{[^}}]*\}}|[^}}"])*)\}}"', text)
    assert match is not None, f"no `${{{name}:-...}}` default in the script"
    return match.group(1)


def _expected_contract() -> str:
    """The contract line the script requires of the checkout, from its text.

    Empty when the script states none, so the stub can stand in for a checkout
    that answers whatever the script asks for.
    """
    match = re.search(r'^MATRIX_CONTRACT_EXPECTED="([^"]*)"', _sbatch_text(),
                      re.MULTILINE)
    return match.group(1) if match else ""


def _contract_probe() -> str:
    """The python the script runs to read a checkout's contract back."""
    match = re.search(r"^CONTRACT_PROBE='\n(.*?)^'\n", _sbatch_text(),
                      re.DOTALL | re.MULTILINE)
    assert match is not None, "no CONTRACT_PROBE in the script"
    return match.group(1)


def _long_request(hours) -> dict:
    """The environment SLURM gives a job whose wall is ``hours``.

    ``SLURM_JOB_END_TIME`` is what the script reads on a compute node, and it
    is the only thing that carries a ``--time`` given at submission: the
    header directive still says 24 h.
    """
    return {"SLURM_JOB_END_TIME": str(int(time.time()) + int(hours * 3600))}


# --------------------------------------------------------------------------- #
# The matrix's own interfaces, read once from the installed module
# --------------------------------------------------------------------------- #

_PROBE = r"""
import contextlib, io, json
from xcquinox.alec.cluster import workflow_matrix as wm
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        wm.main(["--help"])
except SystemExit:
    pass
print(json.dumps({
    "help": buf.getvalue(),
    "stage_order": list(wm.STAGE_ORDER),
    "archs": sorted(wm.ARCHITECTURES),
    "max_shards": wm.MAX_SHARDS,
    "default_timeout_s": wm.DEFAULT_STAGE_TIMEOUT_S,
    "stage_marker": wm.STAGE_MARKER,
}))
"""


@pytest.fixture(scope="module")
def matrix():
    """The parser's own help text, the stage order and the registry.

    One interpreter start for the whole module: the alternative is restating
    the flag names here, which is what the test exists to catch.
    """
    env = dict(os.environ)
    env.update({"JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
    proc = subprocess.run([sys.executable, "-c", _PROBE], cwd=str(_REPO),
                          env=env, capture_output=True, text=True,
                          timeout=900)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


# --------------------------------------------------------------------------- #
# SLURM surface
# --------------------------------------------------------------------------- #

def test_mail_directives_present():
    t = _sbatch_text()
    assert "#SBATCH --mail-user=alec.wills@stonybrook.edu" in t
    assert "#SBATCH --mail-type=BEGIN,END,FAIL" in t


def test_house_shell_idiom():
    t = _sbatch_text()
    assert "set -uo pipefail" in t
    for line in t.splitlines():
        assert not line.strip().startswith("set -e"), line
        assert "errexit" not in line, line


def test_single_node_one_task_with_a_thread_cap_from_slurm():
    t = _sbatch_text()
    assert "#SBATCH --nodes=1" in t
    assert "#SBATCH --ntasks=1" in t
    assert "#SBATCH --cpus-per-task=40" in t
    assert "#SBATCH --exclusive" in t
    assert 'THREADS="${SLURM_CPUS_PER_TASK:-40}"' in t
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert f'export {var}="$THREADS"' in t


def test_x64_is_on_and_the_platform_is_cpu():
    # fp32 would change every energy the stages compute, silently.
    t = _sbatch_text()
    assert "export JAX_ENABLE_X64=1" in t
    assert "export JAX_PLATFORMS=cpu" in t


def test_activation_by_effect_and_an_import_probe():
    t = _sbatch_text()
    assert 'conda activate "$ENV_PREFIX" || true' in t
    assert '"$ENV_PREFIX"/*) : ;;' in t
    # The contract probe is also the import probe: it imports the module, so a
    # dead environment fails it before any contract is compared.
    assert 'python -c "$CONTRACT_PROBE"' in t
    assert "from xcquinox.alec.cluster import workflow_matrix" in _contract_probe()
    assert "repo import failed" in t


def test_the_output_file_and_the_run_log_are_both_named():
    t = _sbatch_text()
    assert "#SBATCH --output=/gpfs/scratch/awills/workflow_matrix_%j.out" in t
    # The exit code read is python's, not tee's: tee succeeds when python
    # aborts, and pipefail alone would report the first non-zero of the pipe.
    assert 'tee "$LOG"' in t
    assert 'RC="${PIPESTATUS[0]}"' in t
    assert 'exit "$RC"' in t


def test_the_work_root_is_a_stamped_directory_outside_the_repository():
    # `main` refuses a work root inside the repository: the stages write run
    # directories, a copy of the reference cache and pretrain data.
    default = _default_of("MATRIX_WORK_ROOT")
    assert default.startswith("/gpfs/scratch/awills/workflow_matrix/")
    assert "${STAMP}" in default
    t = _sbatch_text()
    assert 'STAMP="$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-manual}"' in t
    assert 'REPORT="${WORK_ROOT}/workflow_matrix.md"' in t


def test_the_script_does_not_queue_anything_of_its_own():
    # The matrix drives every stage in-process on this node. A second sbatch
    # or an srun here would mean the job graph ran twice.
    t = _sbatch_text()
    body = [ln for ln in t.splitlines() if not ln.strip().startswith("#")]
    for token in ("sbatch", "srun", "salloc"):
        offenders = [ln for ln in body if re.search(rf"\b{token}\b", ln)]
        assert not offenders, offenders


# --------------------------------------------------------------------------- #
# Wall derivation
# --------------------------------------------------------------------------- #

def _walltime_seconds(text: str) -> int:
    match = re.search(r"^#SBATCH --time=(\d+):(\d\d):(\d\d)$", text,
                      re.MULTILINE)
    assert match is not None, "no #SBATCH --time=H:MM:SS directive"
    h, m, s = (int(g) for g in match.groups())
    return 3600 * h + 60 * m + s


def test_the_request_bounds_the_matrix_at_its_own_defaults(matrix):
    """The request is an upper bound, recomputed here from the script's own
    knob defaults and the matrix's own stage count.

    Per architecture the matrix runs one subprocess per stage plus the oracle
    selection, each killed at ``--timeout-s``; architectures are dealt into
    shards that run concurrently. If a default moves without the request
    moving, the job is killed at its wall and writes no report -- the whole
    pass is then lost, since the report is written after the last architecture.
    """
    text = _sbatch_text()
    timeout_s = int(_default_of("MATRIX_TIMEOUT_S"))
    shards = int(_default_of("MATRIX_SHARDS"))
    batches = int(_default_of("MATRIX_BATCHES"))
    n_archs = len(matrix["archs"])
    stages = len(matrix["stage_order"]) + _ORACLE_STAGES

    per_batch = math.ceil(n_archs / batches)
    per_shard = math.ceil(per_batch / shards)
    bound_s = per_shard * stages * timeout_s
    assert _walltime_seconds(text) >= bound_s, (
        f"{stages} stages x {timeout_s} s x {per_shard} architectures per "
        f"shard = {bound_s} s exceeds the request")
    # An upper bound with no margin is a job that dies before its report.
    assert _walltime_seconds(text) >= bound_s + 3600


def test_the_stated_arithmetic_matches_the_defaults(matrix):
    """The header states the derivation in prose; the numbers in it are the
    ones the script and the registry actually carry."""
    text = _sbatch_text()
    n_archs = len(matrix["archs"])
    stages = len(matrix["stage_order"]) + _ORACLE_STAGES
    per_batch = math.ceil(n_archs / int(_default_of("MATRIX_BATCHES")))
    assert f"{stages} stages x {_default_of('MATRIX_TIMEOUT_S')} s" in text
    assert f"ceil({per_batch} archs / {_default_of('MATRIX_SHARDS')} shards)" \
        in text
    assert f"registry holds {n_archs}" in text


def test_the_per_stage_cap_is_not_looser_than_the_matrix_default(matrix):
    # The module's own cap (3600 s) is a hang detector for a serial local run;
    # the request here is derived from the script's, so the script's must be
    # the tighter of the two or the derivation understates the bound.
    assert int(_default_of("MATRIX_TIMEOUT_S")) <= matrix["default_timeout_s"]


def test_the_shard_default_is_within_the_matrix_ceiling(matrix):
    assert 1 <= int(_default_of("MATRIX_SHARDS")) <= matrix["max_shards"]


def test_the_walltime_exceeds_the_short_and_medium_queue_caps():
    # SeaWulf: short-* caps at 4 h, medium-* at 12 h. The request and the
    # partition have to agree on a long queue.
    t = _sbatch_text()
    assert _walltime_seconds(t) > 12 * 3600
    assert "#SBATCH --partition=long-" in t


# --------------------------------------------------------------------------- #
# The stub interpreter: the script executed without the matrix
# --------------------------------------------------------------------------- #

_STUB = """#!/usr/bin/env bash
# Stands in for the environment's python. Answers the three invocations the
# job script makes and records the last argument vector it was given.
#
# `-c` is the contract probe: STUB_CONTRACT is the contract this stand-in
# checkout reports, and STUB_CONTRACT_RC is what a checkout whose import
# fails outright returns.
if [ "${1:-}" = "-c" ]; then
    printf '%s\\n' "${STUB_CONTRACT-}"
    exit "${STUB_CONTRACT_RC:-0}"
fi
for arg in "$@"; do
    if [ "$arg" = "--list" ]; then
        if [ "${STUB_NO_EOL:-0}" = "1" ]; then
            printf '%s' "$(printf '%s\\n' ${STUB_ARCHS:-})"
        else
            printf '%s\\n' ${STUB_ARCHS:-}
        fi
        exit 0
    fi
done
printf '%s\\n' "$@" > "$STUB_ARGV"
echo "[stub] the matrix ran"
exit "${STUB_RC:-0}"
"""


def _make_tree(tmp_path, *, ledger=True, refs="npz"):
    """A repository and an environment the script accepts, plus a work root.

    ``refs`` is ``"npz"`` (a cache carrying references), ``"empty"`` (a
    directory carrying none) or ``None`` (no cache directory at all).
    """
    env_prefix = tmp_path / "env"
    (env_prefix / "bin").mkdir(parents=True, exist_ok=True)
    stub = env_prefix / "bin" / "python"
    stub.write_text(_STUB)
    stub.chmod(0o755)
    repo = tmp_path / "repo"
    cache = repo / "notebooks" / "checkpoints_step7"
    (cache / "alpha_on").mkdir(parents=True, exist_ok=True)
    if ledger:
        (cache / "alpha_on" / "subset_index_log.json").write_text("{}\n")
    if refs is not None:
        (cache / "external_refs").mkdir(parents=True, exist_ok=True)
        if refs == "npz":
            (cache / "external_refs" / "H2O.npz").write_bytes(b"")
    work = tmp_path / "work"
    return env_prefix, repo, work


#: The registry the stub reports when a test does not name one. Eight names
#: rather than the real 31: the whole-registry tests pass their own list, and a
#: test that only needs the script to reach its command line should not have to
#: carry a request long enough for 31 architectures.
_STUB_REGISTRY = [f"a{i}" for i in range(8)]

#: The real registry's size. The batch split and the wall bound are checked at
#: this number, since that is what the cluster job selects from.
_FULL_REGISTRY = [f"a{i}" for i in range(31)]


def _run_script(tmp_path, *, extra_env=None, rc=0, archs=None, ledger=True,
                refs="npz", env_prefix_override=None):
    """Execute the job script against the stub interpreter."""
    env_prefix, repo, work = _make_tree(tmp_path, ledger=ledger, refs=refs)
    argv_file = tmp_path / "argv.txt"
    env = {
        "PATH": f"{env_prefix / 'bin'}:{os.environ.get('PATH', '')}",
        "HOME": str(tmp_path),
        "XCQ_ENV_PREFIX": str(env_prefix_override or env_prefix),
        "XCQ_CONDA_PROFILE": str(tmp_path / "no_such_conda.sh"),
        "XCQ_REPO": str(repo),
        "MATRIX_WORK_ROOT": str(work),
        "STUB_ARGV": str(argv_file),
        "STUB_RC": str(rc),
        "STUB_ARCHS": " ".join(_STUB_REGISTRY if archs is None else archs),
        # The stand-in checkout is the one the script asks for unless a test
        # says otherwise; a checkout that is NOT is finding 1's case.
        "STUB_CONTRACT": _expected_contract(),
    }
    env.update(extra_env or {})
    proc = subprocess.run(["bash", str(_SBATCH)], env=env, capture_output=True,
                          text=True, timeout=300)
    recorded = (argv_file.read_text().splitlines()
                if argv_file.is_file() else None)
    return proc, recorded, work


def _flag_value(argv, flag):
    return argv[argv.index(flag) + 1]


def _staged_refs(path, *, marker=False):
    """A directory carrying what a staged reference copy carries.

    The two criteria ``workflow_matrix.staged_refs_dir`` accepts, each on its
    own: a COMPLETE staging manifest -- one recording at least one file, every
    recorded file present -- or the per-species ``.npz`` files. The manifest
    case records a non-``.npz`` file so that it exercises the manifest and not
    the glob; a manifest whose files are absent is the shape an interrupted
    copy leaves, and is refused (see
    ``test_the_reference_precheck_is_the_matrix_predicate``).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if marker:
        (path / "H2O.dat").write_text("x")
        (path / "_stage_complete").write_text("source: test\nH2O.dat\n")
    else:
        (path / "H2O.npz").write_bytes(b"")
    return path


def test_the_default_invocation_carries_the_whole_registry(tmp_path):
    proc, argv, work = _run_script(tmp_path, extra_env={"MATRIX_BATCHES": "1"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert argv is not None, proc.stdout
    assert argv[:3] == ["-u", "-m", "xcquinox.alec.cluster.workflow_matrix"]
    assert _flag_value(argv, "--archs") == "all"
    assert _flag_value(argv, "--work-root") == str(work)
    assert _flag_value(argv, "--report") == str(work / "workflow_matrix.md")
    assert _flag_value(argv, "--shards") == _default_of("MATRIX_SHARDS")
    assert _flag_value(argv, "--timeout-s") == _default_of("MATRIX_TIMEOUT_S")
    assert "--external-refs-dir" not in argv
    assert "--no-oracles" not in argv


def test_every_flag_the_script_passes_is_a_flag_the_matrix_accepts(
        tmp_path, matrix):
    """Checked against the parser's own help text: a renamed flag would
    otherwise turn the allocation into a usage error at second zero."""
    proc, argv, _ = _run_script(
        tmp_path, extra_env={
            "MATRIX_BATCHES": "1",
            "MATRIX_EXTERNAL_REFS": str(_staged_refs(tmp_path / "refs")),
            "MATRIX_NO_ORACLES": "1"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    passed = [tok for tok in argv if tok.startswith("--")]
    assert set(passed) == {"--archs", "--work-root", "--report", "--shards",
                           "--timeout-s", "--external-refs-dir",
                           "--no-oracles"}
    for flag in passed:
        assert flag in matrix["help"], flag


def test_named_architectures_are_taken_verbatim_and_skip_the_batch_split(
        tmp_path):
    proc, argv, _ = _run_script(
        tmp_path, extra_env={"MATRIX_ARCHS": "a3,a4"},
        archs=_FULL_REGISTRY)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _flag_value(argv, "--archs") == "a3,a4"
    assert "batching not applied" in proc.stdout


@pytest.mark.parametrize("batch,expected", [
    (0, _FULL_REGISTRY[:16]),
    (1, _FULL_REGISTRY[16:]),
])
def test_the_batch_split_covers_the_registry_exactly_once(tmp_path, batch,
                                                          expected):
    """Two batches of a 31-name registry: 16 then 15, no name in both and none
    left out. The split is what keeps a whole-registry pass inside one wall."""
    proc, argv, _ = _run_script(
        tmp_path, extra_env={"MATRIX_BATCHES": "2", "MATRIX_BATCH": str(batch)},
        archs=_FULL_REGISTRY)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _flag_value(argv, "--archs").split(",") == expected


def test_the_last_listed_architecture_survives_a_missing_final_newline(
        tmp_path):
    """A dropped final line is a silently missing row: the architecture is not
    run, not reported, and nothing says so."""
    proc, argv, _ = _run_script(
        tmp_path, extra_env={"MATRIX_BATCHES": "2", "MATRIX_BATCH": "1",
                             "STUB_NO_EOL": "1"},
        archs=_FULL_REGISTRY)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _flag_value(argv, "--archs").split(",")[-1] == "a30"
    assert len(_flag_value(argv, "--archs").split(",")) == 15


def test_a_single_batch_leaves_the_selection_as_the_whole_registry(tmp_path):
    """One batch covers everything, so the selection is handed over as 'all'
    rather than expanded into a list the registry would have to match.

    31 architectures in one job bound at 44 h, so this is also the case the
    header documents: it needs a request that covers the bound, and the only
    thing carrying a submission-time ``--time`` into the job is
    ``SLURM_JOB_END_TIME``.
    """
    proc, argv, _ = _run_script(
        tmp_path, extra_env={"MATRIX_BATCHES": "1", **_long_request(46)},
        archs=_FULL_REGISTRY)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _flag_value(argv, "--archs") == "all"
    assert "SLURM_JOB_END_TIME" in proc.stdout


def test_the_knobs_reach_the_command_line(tmp_path):
    refs = _staged_refs(tmp_path / "staged_refs")
    proc, argv, _ = _run_script(
        tmp_path, extra_env={"MATRIX_BATCHES": "1", "MATRIX_SHARDS": "2",
                             "MATRIX_TIMEOUT_S": "600",
                             "MATRIX_NO_ORACLES": "1",
                             "MATRIX_EXTERNAL_REFS": str(refs)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _flag_value(argv, "--shards") == "2"
    assert _flag_value(argv, "--timeout-s") == "600"
    assert _flag_value(argv, "--external-refs-dir") == str(refs)
    assert "--no-oracles" in argv


@pytest.mark.parametrize("marker", [False, True])
def test_a_supplied_reference_copy_replaces_the_repository_cache(tmp_path,
                                                                 marker):
    """With MATRIX_EXTERNAL_REFS the repository's own (untracked) cache need
    not be present: the supplied copy is the input. Both criteria the matrix
    accepts -- the staging manifest and the species files -- are accepted."""
    refs = _staged_refs(tmp_path / "staged_refs", marker=marker)
    proc, argv, _ = _run_script(
        tmp_path, refs=None,
        extra_env={"MATRIX_BATCHES": "1", "MATRIX_EXTERNAL_REFS": str(refs)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _flag_value(argv, "--external-refs-dir") == str(refs)


@pytest.mark.parametrize("rc", [0, 1, 7])
def test_the_matrix_exit_code_reaches_slurm(tmp_path, rc):
    """`main` returns 0 for a clean pass and 1 for a matrix that found a
    defect; SLURM has to see that number, not tee's."""
    proc, _argv, _ = _run_script(tmp_path, rc=rc,
                                 extra_env={"MATRIX_BATCHES": "1"})
    assert proc.returncode == rc, proc.stdout + proc.stderr


def test_a_non_clean_matrix_is_reported_as_a_finding_not_a_crash(tmp_path):
    # SLURM mails FAIL on any non-zero code; the log has to say that a 1 is a
    # completed matrix, or the mail reads as a broken job.
    proc, _argv, _ = _run_script(tmp_path, rc=1,
                                 extra_env={"MATRIX_BATCHES": "1"})
    assert proc.returncode == 1
    assert "Findings block" in proc.stdout
    assert "Not a crash" in proc.stdout


def test_the_run_log_is_a_tailable_file_under_the_work_root(tmp_path):
    proc, _argv, work = _run_script(tmp_path,
                                    extra_env={"MATRIX_BATCHES": "1"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    log = work / "workflow_matrix.log"
    assert log.is_file()
    assert "[stub] the matrix ran" in log.read_text()


# --------------------------------------------------------------------------- #
# Refusals: nothing measured is reported as exit 3, never as a matrix finding
# --------------------------------------------------------------------------- #

def test_a_missing_subset_ledger_names_the_file_and_the_rsync(tmp_path):
    proc, argv, _ = _run_script(tmp_path, ledger=False,
                                extra_env={"MATRIX_BATCHES": "1"})
    assert proc.returncode == 3, proc.stdout
    assert "subset_index_log.json" in proc.stdout
    assert "rsync -av" in proc.stdout
    assert argv is None, "the matrix must not be launched without its inputs"


def test_a_missing_reference_cache_names_the_directory_and_the_alternative(
        tmp_path):
    proc, argv, _ = _run_script(tmp_path, refs=None,
                                extra_env={"MATRIX_BATCHES": "1"})
    assert proc.returncode == 3, proc.stdout
    assert "external_refs" in proc.stdout
    assert "MATRIX_EXTERNAL_REFS" in proc.stdout
    assert argv is None


def test_a_supplied_reference_path_that_is_not_a_directory_is_refused(
        tmp_path):
    absent = tmp_path / "mistyped_refs"
    proc, argv, _ = _run_script(
        tmp_path, extra_env={"MATRIX_BATCHES": "1",
                             "MATRIX_EXTERNAL_REFS": str(absent)})
    assert proc.returncode == 3, proc.stdout
    assert str(absent) in proc.stdout
    assert argv is None


def test_a_reference_directory_holding_no_references_is_refused(tmp_path):
    """An empty directory satisfies every path check and fails at the first
    reference read, a stage deep into the sequence. The matrix applies this
    same criterion in ``staged_refs_dir``; applied here it is one line in the
    log instead."""
    empty = tmp_path / "empty_refs"
    empty.mkdir()
    proc, argv, _ = _run_script(
        tmp_path, extra_env={"MATRIX_BATCHES": "1",
                             "MATRIX_EXTERNAL_REFS": str(empty)})
    assert proc.returncode == 3, proc.stdout
    assert "carries no references" in proc.stdout
    assert argv is None


def test_a_repository_cache_holding_no_references_is_refused(tmp_path):
    """The directory exists and is empty -- what a partial rsync of the
    untracked cache leaves behind."""
    proc, argv, _ = _run_script(tmp_path, refs="empty",
                                extra_env={"MATRIX_BATCHES": "1"})
    assert proc.returncode == 3, proc.stdout
    assert "external_refs" in proc.stdout
    assert "rsync -av" in proc.stdout
    assert argv is None


def test_a_work_root_inside_the_repository_is_refused(tmp_path):
    """`main` refuses it too, but as a traceback and exit 1 -- the code a
    non-clean matrix returns."""
    inside = tmp_path / "repo" / "scratch_root"
    proc, argv, _ = _run_script(
        tmp_path, extra_env={"MATRIX_BATCHES": "1",
                             "MATRIX_WORK_ROOT": str(inside)})
    assert proc.returncode == 3, proc.stdout
    assert "inside the repository" in proc.stdout
    assert argv is None


@pytest.mark.parametrize("knobs", [
    {"MATRIX_SHARDS": "5"},        # above workflow_matrix.MAX_SHARDS
    {"MATRIX_SHARDS": "0"},
    {"MATRIX_SHARDS": "four"},
    {"MATRIX_TIMEOUT_S": "0"},
    {"MATRIX_TIMEOUT_S": "-1"},
    {"MATRIX_BATCHES": "0"},
    {"MATRIX_BATCHES": "2", "MATRIX_BATCH": "2"},   # 0-based
    {"MATRIX_BATCHES": "2", "MATRIX_BATCH": "-1"},
    {"MATRIX_ARCHS": ""},
])
def test_a_bad_knob_is_refused_before_the_allocation_is_spent(tmp_path, knobs):
    """A knob the matrix would reject surfaces there as a traceback and exit 1
    -- the code a non-clean matrix returns. Refusing it here keeps exit 1
    meaning what the report says it means."""
    proc, argv, _ = _run_script(tmp_path, extra_env=knobs,
                                archs=_FULL_REGISTRY)
    assert proc.returncode == 3, proc.stdout
    assert "FATAL" in proc.stdout
    assert argv is None


def test_an_inactive_environment_is_refused(tmp_path):
    """Activation is verified by effect: the resolved interpreter has to sit
    under the environment prefix, whatever conda's return code said."""
    proc, argv, _ = _run_script(
        tmp_path, env_prefix_override=Path("/gpfs/nowhere/env"),
        extra_env={"MATRIX_BATCHES": "1"})
    assert proc.returncode == 3, proc.stdout
    assert "env python not active" in proc.stdout
    assert argv is None


# --------------------------------------------------------------------------- #
# The checkout's contract: a partial sync is refused before anything runs
# --------------------------------------------------------------------------- #

def test_the_contract_the_script_requires_is_what_the_module_reports(matrix):
    """The expectation in the script is the installed module's own answer.

    Executed: the script's probe is run against the real checkout and its line
    compared with the script's expectation, so neither can drift from the
    other. The three fields are the three interfaces the job depends on -- the
    stage list the wall bound counts, the shard ceiling the knob validation
    admits, and the parser refusing a non-positive ``--timeout-s`` with its
    usage code rather than reporting every stage as killed.
    """
    expected = _expected_contract()
    assert expected, "the script states no MATRIX_CONTRACT_EXPECTED"
    env = dict(os.environ)
    env.update({"JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
    proc = subprocess.run([sys.executable, "-c", _contract_probe()],
                          cwd=str(_REPO), env=env, capture_output=True,
                          text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert proc.stdout.strip() == expected
    # ... and the expectation is the module's interfaces, not a string that
    # happens to match today.
    assert f"stages={','.join(matrix['stage_order'])}" in expected
    assert f"shards={matrix['max_shards']}" in expected
    assert "timeout_bound=1" in expected


def test_a_checkout_whose_contract_moved_is_refused_with_the_package_push(
        tmp_path):
    """Finding: a sync carrying the two hpcjobs files and not the package.

    The matrix drives the stage modules of the same tree, so an older
    ``workflow_matrix.py`` accepts the same flags and then drives contracts
    that have moved -- the damaging case, since the log looks like a running
    matrix. The refusal names both contracts and the push that fixes it.
    """
    moved = re.sub(r"stages=[^ ]*", "stages=submit,datagen",
                   _expected_contract())
    proc, argv, _ = _run_script(tmp_path,
                                extra_env={"STUB_CONTRACT": moved})
    assert proc.returncode == 3, proc.stdout
    assert argv is None, "the matrix must not be launched against moved stages"
    assert moved in proc.stdout
    assert _expected_contract() in proc.stdout
    assert "rsync -av xcquinox hpcjobs" in proc.stdout
    assert "partial sync" in proc.stdout


def test_a_checkout_that_reports_no_contract_is_refused(tmp_path):
    """An installed module too old to answer the probe at all: the probe
    prints nothing and the comparison has nothing to compare."""
    proc, argv, _ = _run_script(tmp_path, extra_env={"STUB_CONTRACT": ""})
    assert proc.returncode == 3, proc.stdout
    assert argv is None
    assert "rsync -av xcquinox hpcjobs" in proc.stdout


def test_a_checkout_that_cannot_be_imported_is_refused_as_an_environment(
        tmp_path):
    """The contract probe is also the import probe: a checkout that raises on
    import is an environment fault, and says so rather than reporting a
    contract mismatch."""
    proc, argv, _ = _run_script(tmp_path,
                                extra_env={"STUB_CONTRACT_RC": "1"})
    assert proc.returncode == 3, proc.stdout
    assert argv is None
    assert "repo import failed" in proc.stdout


def test_the_header_states_the_whole_package_push(matrix):
    """The push line in the header is the one the handover carries: the
    package and hpcjobs, in the repository-relative form."""
    t = _sbatch_text()
    assert "rsync -av xcquinox hpcjobs" in t
    assert '"$swpath":' in t


# --------------------------------------------------------------------------- #
# Architecture names: a typo is a job that could not start
# --------------------------------------------------------------------------- #

def test_an_unregistered_architecture_name_is_refused_naming_the_registry(
        tmp_path):
    """Finding: a typed name reached the matrix, which refused it as a usage
    error after the queue wait had been paid. It is checked here against the
    installed registry, which the script already knows how to obtain."""
    proc, argv, _ = _run_script(
        tmp_path, extra_env={"MATRIX_ARCHS": "a3,deep_3x17"})
    assert proc.returncode == 3, proc.stdout
    assert argv is None
    assert "deep_3x17" in proc.stdout
    for name in _STUB_REGISTRY:
        assert name in proc.stdout, "the refusal names the registry"


def test_a_repeated_architecture_name_is_refused(tmp_path):
    """Two rows for one architecture share one working directory: the matrix
    refuses it as a usage error, so the job refuses it before the wait."""
    proc, argv, _ = _run_script(tmp_path, extra_env={"MATRIX_ARCHS": "a3,a3"})
    assert proc.returncode == 3, proc.stdout
    assert argv is None
    assert "more than once" in proc.stdout


def test_an_architecture_list_of_separators_only_is_refused(tmp_path):
    proc, argv, _ = _run_script(tmp_path, extra_env={"MATRIX_ARCHS": ",,"})
    assert proc.returncode == 3, proc.stdout
    assert argv is None
    assert "names no architecture" in proc.stdout


def test_a_registered_name_list_reaches_the_matrix(tmp_path):
    proc, argv, _ = _run_script(tmp_path,
                                extra_env={"MATRIX_ARCHS": "a1,a6"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _flag_value(argv, "--archs") == "a1,a6"


def test_a_usage_error_from_the_matrix_is_not_reported_as_a_finding(tmp_path):
    """The matrix's exit 2 is a refused command line -- a job that never ran.

    Reported as the matrix's own 1 it would read as "at least one architecture
    did not", which is the sentence reserved for a completed matrix; it is
    carried into this script's class 3 instead, with the matrix's own code
    kept in the log.
    """
    proc, _argv, _ = _run_script(tmp_path, rc=2)
    assert proc.returncode == 3, proc.stdout
    assert "COMPLETED; at least one architecture did not" not in proc.stdout
    assert "refused its command line" in proc.stdout
    assert "matrix rc=2" in proc.stdout


# --------------------------------------------------------------------------- #
# The derived wall bound against the wall actually requested
# --------------------------------------------------------------------------- #

def test_the_defaults_run_and_state_the_bound_beside_the_request(tmp_path):
    """The default submission -- batch 0 of 2 over a 31-name registry -- and
    the two numbers it is judged on, printed before the matrix starts."""
    proc, argv, _ = _run_script(tmp_path, archs=_FULL_REGISTRY)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert len(_flag_value(argv, "--archs").split(",")) == 16
    # ceil(16 / 4) x 11 timed stages x 1800 s against the header's 24 h.
    assert "wall bound=79200 s (22.0 h)" in proc.stdout
    assert "wall request=86400 s (24.0 h)" in proc.stdout
    assert "#SBATCH --time=24:00:00" in proc.stdout


@pytest.mark.parametrize("knobs,bound", [
    ({"MATRIX_BATCHES": "1"}, "158400 s (44.0 h)"),
    ({"MATRIX_SHARDS": "1"}, "316800 s (88.0 h)"),
])
def test_a_knob_that_outruns_the_request_is_refused_with_both_numbers(
        tmp_path, knobs, bound):
    """Finding: neither knob was compared with the wall.

    ``MATRIX_BATCHES=1`` puts all 31 architectures in one job and
    ``MATRIX_SHARDS=1`` runs a batch of 16 serially; both outrun the 24 h
    request, and a job killed at its wall writes NO report -- the table is
    written only after the last architecture returns, so the whole allocation
    is lost.
    """
    proc, argv, _ = _run_script(tmp_path, extra_env=knobs,
                                archs=_FULL_REGISTRY)
    assert proc.returncode == 3, proc.stdout
    assert argv is None
    assert f"wall bound={bound}" in proc.stdout
    assert "wall request=86400 s (24.0 h)" in proc.stdout
    assert "exceeds the wall request" in proc.stdout


def test_the_request_is_read_from_slurms_own_end_time_when_it_is_set(tmp_path):
    """A ``--time`` given at submission never reaches the header directive, so
    on a compute node the allocation's own end time is what bounds the job."""
    proc, argv, _ = _run_script(tmp_path, extra_env=_long_request(6),
                                archs=_FULL_REGISTRY)
    assert proc.returncode == 3, proc.stdout
    assert argv is None
    assert "SLURM_JOB_END_TIME" in proc.stdout
    assert "wall bound=79200 s (22.0 h)" in proc.stdout
    assert "exceeds the wall request" in proc.stdout


def test_dropping_the_oracles_drops_a_stage_from_the_bound(tmp_path):
    """The oracle selection is one of the timed subprocesses; without it the
    bound is the ten stages of STAGE_ORDER."""
    proc, argv, _ = _run_script(
        tmp_path, extra_env={"MATRIX_NO_ORACLES": "1"}, archs=_FULL_REGISTRY)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "--no-oracles" in argv
    assert "wall bound=72000 s (20.0 h)" in proc.stdout
    assert "x 10 timed stages x 1800 s" in proc.stdout


# --------------------------------------------------------------------------- #
# Minors: the reference predicate, the smoke's shards, the supplied copy
# --------------------------------------------------------------------------- #

_REFS_PROBE = r"""
import json, sys
from xcquinox.alec.cluster.workflow_matrix import (
    CachedInputsMissing, staged_refs_dir)
out = {}
for name, path in json.loads(sys.argv[1]).items():
    try:
        staged_refs_dir(path)
        out[name] = True
    except CachedInputsMissing:
        out[name] = False
print(json.dumps(out))
"""

#: Directory shapes a reference copy can be in. ``manifest_missing_file`` is
#: what an interrupted rsync of a warmed copy leaves, and is the shape the
#: script's shortcut used to accept.
_REFS_SHAPES = ("absent", "empty", "npz_only", "complete_manifest",
                "manifest_missing_file", "manifest_lists_nothing",
                "empty_manifest", "manifest_missing_file_with_npz",
                "nested_manifest", "directory_named_npz")


def _build_refs_shape(root, shape, marker):
    root = Path(root)
    if shape == "absent":
        return root
    root.mkdir(parents=True, exist_ok=True)
    if shape == "empty":
        pass
    elif shape == "npz_only":
        (root / "H2O.npz").write_bytes(b"")
    elif shape == "complete_manifest":
        (root / "N2.dat").write_text("x")
        (root / marker).write_text("source: test\nN2.dat\n")
    elif shape == "manifest_missing_file":
        (root / marker).write_text("source: test\nH2O.npz\n")
    elif shape == "manifest_lists_nothing":
        (root / marker).write_text("source: test\n")
    elif shape == "empty_manifest":
        (root / marker).write_text("")
    elif shape == "manifest_missing_file_with_npz":
        (root / marker).write_text("source: test\nN2.dat\n")
        (root / "H2O.npz").write_bytes(b"")
    elif shape == "nested_manifest":
        (root / "_intermediates").mkdir()
        (root / "_intermediates" / "H2O_scf.dat").write_text("x")
        (root / marker).write_text("source: test\n_intermediates/H2O_scf.dat\n")
    elif shape == "directory_named_npz":
        (root / "H2O.npz").mkdir()
    else:  # pragma: no cover - the parametrisation is closed
        raise AssertionError(shape)
    return root


def test_the_reference_precheck_is_the_matrix_predicate(tmp_path, matrix):
    """The script's precheck and ``staged_refs_dir`` accept the same shapes.

    The precheck exists to turn a reference directory the matrix would refuse
    -- deep inside a stage, once per architecture -- into one line in this
    log. It is worth that only while the two agree, so the two are run against
    the same ten shapes and compared.
    """
    marker = matrix["stage_marker"]
    assert f"/{marker}" in _sbatch_text(), "the script names another marker"
    paths = {shape: str(_build_refs_shape(tmp_path / "shapes" / shape, shape,
                                          marker))
             for shape in _REFS_SHAPES}
    env = dict(os.environ)
    env.update({"JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
    proc = subprocess.run(
        [sys.executable, "-c", _REFS_PROBE, json.dumps(paths)],
        cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-4000:]
    module = json.loads(proc.stdout.strip().splitlines()[-1])
    script = {}
    for shape in _REFS_SHAPES:
        run, _argv, _work = _run_script(
            tmp_path / f"run_{shape}",
            extra_env={"MATRIX_EXTERNAL_REFS": paths[shape]})
        assert run.returncode in (0, 3), run.stdout + run.stderr
        script[shape] = run.returncode == 0
    assert script == module
    # The comparison is worthless if everything is refused (or accepted).
    assert set(module.values()) == {True, False}


def test_a_supplied_reference_directory_inside_the_repository_is_refused(
        tmp_path):
    """Every preflight writes a ``_run_log_<UTC>.json`` into the directory it
    reads, so a reference copy in the tree writes the run's own output into
    the tree the run measures. The matrix refuses it as a usage error; the job
    refuses it before the wait."""
    inside = tmp_path / "repo" / "notebooks" / "checkpoints_step7" / "warmed"
    _staged_refs(inside)
    proc, argv, _ = _run_script(
        tmp_path, extra_env={"MATRIX_EXTERNAL_REFS": str(inside)})
    assert proc.returncode == 3, proc.stdout
    assert argv is None
    assert "inside the repository" in proc.stdout


def test_a_supplied_reference_directory_is_marked_as_this_jobs_alone(
        tmp_path):
    """Two jobs pointed at one copy write run logs into it concurrently.
    Unset, each job stages its own; set, the log says the directory is this
    job's."""
    refs = _staged_refs(tmp_path / "warmed_refs")
    proc, argv, _ = _run_script(
        tmp_path, extra_env={"MATRIX_EXTERNAL_REFS": str(refs)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _flag_value(argv, "--external-refs-dir") == str(refs)
    assert "read AND written by this job" in proc.stdout
    assert "job's copy alone" in proc.stdout


def test_a_single_architecture_takes_the_whole_allocation(tmp_path):
    """The smoke is one architecture on an exclusive node, and it gates the
    matrix: at the default four shards the matrix would give it a quarter of
    the allocation."""
    proc, argv, _ = _run_script(tmp_path, extra_env={"MATRIX_ARCHS": "a3"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _flag_value(argv, "--shards") == "1"
    assert "the whole allocation" in proc.stdout


def test_two_named_architectures_keep_the_default_shard_count(tmp_path):
    proc, argv, _ = _run_script(tmp_path, extra_env={"MATRIX_ARCHS": "a3,a4"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _flag_value(argv, "--shards") == _default_of("MATRIX_SHARDS")


def test_an_explicit_shard_count_survives_the_single_architecture_rule(
        tmp_path):
    proc, argv, _ = _run_script(
        tmp_path, extra_env={"MATRIX_ARCHS": "a3", "MATRIX_SHARDS": "2"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _flag_value(argv, "--shards") == "2"


def test_the_stage_cap_is_justified_against_the_modules_own_constant(matrix):
    """The script halves the module's hang threshold, so the header carries
    the module's own constant and the measurement the choice rests on: the
    build walls of the eight species the two cells need, and the one measured
    tail that the two caps disagree about."""
    t = _sbatch_text()
    assert f"DEFAULT_STAGE_TIMEOUT_S = {matrix['default_timeout_s']}" in t
    assert int(_default_of("MATRIX_TIMEOUT_S")) * 2 == \
        matrix["default_timeout_s"]
    assert "160.8 s" in t          # the eight species, measured
    assert "2558 s" in t           # the tail 1800 s kills and 3600 s does not
    assert "_run_log_" in t        # where both numbers were read


def test_the_partition_cap_is_stated_or_marked_for_verification():
    """The runbook records no cap for long-40core, so the header says so and
    names the command that settles it."""
    t = _sbatch_text()
    assert "long-40core" in t
    assert "sinfo" in t
    assert "SEAWULF_RUNBOOK.md" in t


# --------------------------------------------------------------------------- #
# Static checks
# --------------------------------------------------------------------------- #

def test_the_script_parses():
    proc = subprocess.run(["bash", "-n", str(_SBATCH)], capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(shutil.which("shellcheck") is None,
                    reason="shellcheck is not installed")
def test_shellcheck_is_clean():
    proc = subprocess.run(
        ["shellcheck", "--shell=bash", "--severity=warning", str(_SBATCH)],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
