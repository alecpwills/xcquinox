"""Tests for the per-architecture workflow-matrix job script.

Four things are pinned here. The SLURM SURFACE: the standing mail directives,
the house shell idiom and a single-node one-task allocation whose thread cap
comes from SLURM rather than from a literal. The WALL DERIVATION: the request
in the header is recomputed from the script's own defaults and the matrix's own
stage count, so a knob that moves without the request moving is a failure here
rather than a job killed at its wall with no report. The COMMAND SURFACE: every
flag the script passes is checked against ``workflow_matrix.main``'s argparse,
read out of the real parser, because a renamed flag turns a 30 h allocation
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


def _contract_field(name: str, contract: str | None = None) -> str:
    """One field of a `MATRIX_CONTRACT` line (the script's own by default)."""
    match = re.search(rf"\b{name}=(\S+)", contract or _expected_contract())
    assert match is not None, f"no {name}= field in the contract"
    return match.group(1)


# --------------------------------------------------------------------------- #
# The matrix's own interfaces, read once from the installed module
# --------------------------------------------------------------------------- #

_PROBE = r"""
import contextlib, io, json
from xcquinox.pipeline.cluster import workflow_matrix as wm
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
    """One task on one node, holding the whole node, with the thread cap read
    from the allocation rather than restated. The count is the milan node's 96
    cores, what 40 was on the 40-core node the script was written for, and the
    default the script falls back to outside SLURM follows it, so a run started
    by hand caps the pools where the allocation would.

    Oracle: the directives of the script itself.
    """
    t = _sbatch_text()
    assert "#SBATCH --nodes=1" in t
    assert "#SBATCH --ntasks=1" in t
    assert "#SBATCH --cpus-per-task=96" in t
    assert "#SBATCH --exclusive" in t
    assert 'THREADS="${SLURM_CPUS_PER_TASK:-96}"' in t
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert f'export {var}="$THREADS"' in t


def test_activation_by_effect_and_an_import_probe():
    t = _sbatch_text()
    assert 'conda activate "$ENV_PREFIX" || true' in t
    assert '"$ENV_PREFIX"/*) : ;;' in t
    # The contract probe is also the import probe: it imports the module, so a
    # dead environment fails it before any contract is compared.
    assert 'python -c "$CONTRACT_PROBE"' in t
    assert "from xcquinox.pipeline.cluster import workflow_matrix" in _contract_probe()
    assert "repo import failed" in t


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
    copy leaves, and is refused.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if marker:
        (path / "H2O.dat").write_text("x")
        (path / "_stage_complete").write_text("source: test\nH2O.dat\n")
    else:
        (path / "H2O.npz").write_bytes(b"")
    return path


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


# --------------------------------------------------------------------------- #
# Refusals: nothing measured is reported as exit 3, never as a matrix finding
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# The checkout's contract: a partial sync is refused before anything runs
# --------------------------------------------------------------------------- #


#: A stand-in checkout for the contract probe. The probe is executed
#: VERBATIM, as the script runs it; only the module it imports is the test's,
#: which is how a checkout that has LOST one of the contract's refusals is
#: exercised without one. ``run_matrix`` records that it was entered and then
#: raises what the real one raises against an unwritable work root, so both
#: halves of the failure are visible: the matrix started, and the probe died.
_PROBE_HARNESS = r"""
import sys
import types

(probe_path, behaviour, sentinel, oracle_target, stages, max_shards,
 writable, drop) = sys.argv[1:9]

for name in ("xcquinox", "xcquinox.pipeline", "xcquinox.pipeline.cluster"):
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package

wm = types.ModuleType("xcquinox.pipeline.cluster.workflow_matrix")
wm.STAGE_ORDER = tuple(stages.split(","))
wm.MAX_SHARDS = int(max_shards)
wm.ORACLE_TEST_TARGET = oracle_target


def _run_matrix(*args, **kwargs):
    with open(sentinel, "w") as fh:
        fh.write("the matrix was started\n")
    if writable != "1":
        raise PermissionError(13, "Permission denied", "/probe-work-root")
    return []


def _main(argv=None, **kwargs):
    argv = list(argv or [])

    def _flag(flag, default):
        return int(argv[argv.index(flag) + 1]) if flag in argv else default

    if behaviour == "raises":
        raise RuntimeError("this checkout raises before it refuses anything")
    if behaviour != "no_timeout_refusal" and _flag("--timeout-s", 1800) <= 0:
        raise SystemExit(2)
    if behaviour != "no_shards_refusal" and not 1 <= _flag("--shards", 1) <= 4:
        raise SystemExit(2)
    # The progress line the real main prints on its way into run_matrix.
    print("[workflow-matrix] 31 architectures, 4 shard(s), work root /x")
    return wm.run_matrix(argv)


wm.run_matrix = _run_matrix
wm.main = _main
if drop != "none":
    delattr(wm, drop)
sys.modules["xcquinox.pipeline.cluster.workflow_matrix"] = wm
sys.modules["xcquinox.pipeline.cluster"].workflow_matrix = wm

with open(probe_path) as fh:
    source = fh.read()
exec(compile(source, "<contract probe>", "exec"), {"__name__": "__main__"})
"""


# --------------------------------------------------------------------------- #
# Architecture names: a typo is a job that could not start
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# The derived wall bound against the wall actually requested
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Minors: the reference predicate, the smoke's shards, the supplied copy
# --------------------------------------------------------------------------- #

_REFS_PROBE = r"""
import json, sys
from xcquinox.pipeline.cluster.workflow_matrix import (
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
                "nested_manifest", "directory_named_npz",
                "hidden_npz_only", "crlf_manifest", "broken_symlink_npz")


#: What the oracle stage returns and how it is classified, measured against a
#: stand-in checkout rather than restated. The collection target is one FILE,
#: so a checkout that does not carry the oracle module makes pytest exit with
#: its usage code (a target it cannot find), not with the no-tests-collected
#: code an empty target gives.
_ORACLE_PROBE = r"""
import json, subprocess, sys
from pathlib import Path
from xcquinox.pipeline.cluster import workflow_matrix as wm

out = {"module": wm.ORACLE_MODULE, "target": wm.ORACLE_TEST_TARGET,
       "no_tests_rc": wm.ORACLE_NO_TESTS_RC}
for name, install in (("absent", False), ("present_empty", True)):
    checkout = Path(sys.argv[1]) / name
    target = checkout / wm.ORACLE_TEST_TARGET
    target.parent.mkdir(parents=True, exist_ok=True)
    if install:
        target.write_text("# a module carrying no oracle\n")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(target), "-k", "deep", "-q",
         "-p", "no:randomly", "-p", "no:cacheprovider"],
        cwd=str(checkout), capture_output=True, text=True)
    note = wm._oracle_failure_note(proc.returncode, target, "deep")
    out[name] = {"rc": proc.returncode,
                 "note": list(note) if note is not None else None}
print(json.dumps(out))
"""


# --------------------------------------------------------------------------- #
# Static checks
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(shutil.which("shellcheck") is None,
                    reason="shellcheck is not installed")
def test_shellcheck_is_clean():
    proc = subprocess.run(
        ["shellcheck", "--shell=bash", "--severity=warning", str(_SBATCH)],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
