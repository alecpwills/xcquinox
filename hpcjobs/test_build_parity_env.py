"""The parity environment's build job: its queue, its prefix, its source of pins and the order
of its steps.

The environment every cluster job activates is built by ``build_parity_env.sbatch``: conda
supplies the interpreter alone, pip installs ``requirements.txt`` (the exact releases the
workstation runs, every one a PyPI wheel on the nodes' glibc 2.34, measured 2026-09-22) and
the package from the checkout. The job states no version of its own: ``requirements.txt`` is
the one list, held to the declaration by ``tools/test_packaging.py``.

ORDER: the environment is created, activated and the activation checked before anything is
installed into it; an install placed before the check lands in whatever interpreter the
submitting shell had, after the allocation has already been spent.

REFUSAL: the activation is read by its effect rather than by its return code. Conda returns
non-zero on a successful activation of a prefix path (the record in
``nonempirical_pool.sbatch``), and a job whose activation silently
failed would otherwise install the stack into the base interpreter.

QUEUE: the job runs on a short queue of the milan (96-core) partitions, the nodes the
campaigns use; the build takes minutes.

PREFIX: the environment path equals ``cluster.conda_env`` of the v7 size configuration --
the value every standalone script's prefix is held to -- and its last component names the
jax release ``requirements.txt`` pins, so a stack change that leaves the prefix alone is
caught here instead of by a job that runs silently in the previous environment.

A syntax check over the script guards the insertions themselves.
"""
from __future__ import annotations

import re
from pathlib import Path

HPCJOBS = Path(__file__).resolve().parent
ROOT = HPCJOBS.parent
JOB = HPCJOBS / "build_parity_env.sbatch"
CONFIG = HPCJOBS / "configs" / "dfs_step7.dfs6311_grid3_v7g1_size.yaml"
REQUIREMENTS = ROOT / "requirements.txt"
PYPROJECT = ROOT / "pyproject.toml"

ENV_ASSIGNMENT = re.compile(r"^ENV=(?P<path>\S+)\s*$", re.M)
CONDA_ENV_VALUE = re.compile(r"^\s*conda_env:\s*(?P<path>\S+)\s*$", re.M)
PARTITION_DIRECTIVE = re.compile(r"^#SBATCH\s+--partition=(?P<name>\S+)", re.M)
PYTHON_SPEC = re.compile(r"\bpython=(?P<version>[0-9][0-9.]*)")
REQUIREMENT_PIN = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>\S+)\s*$")
#: a version written into the job itself: ``name==1.2`` or ``NAME_VERSION=1.2`` on an executed
#: line. Every pin lives in ``requirements.txt``; a second statement of one version drifts.
VERSION_IN_JOB = re.compile(r"[A-Za-z0-9_.-]+==[0-9]|^[A-Z_]*VERSION=[0-9]")

#: the environment's last component: the campaign prefix, then ``j`` and the jax release
#: without its dots (jax 0.10.2 gives ``xcquinox_j0102``)
PREFIX_STEM = "xcquinox_j"

#: the milan nodes' partitions carry this token; a 28- or 40-core queue does not
MILAN_TOKEN = "96core"
SHORT_QUEUE = "short-"

# the steps, in the order the job must run them
CREATE = 'conda create -y -p "$ENV"'
ACTIVATE = 'conda activate "$ENV"'
PROBE = "command -v python"
GUARD = 'case "$PYBIN" in'
REQUIREMENTS_INSTALL = 'pip install -r "$REPO/requirements.txt"'
EDITABLE = 'pip install -e "$REPO" --no-deps'
VALIDATION = "pyscfad.__version__"
READY = "== ENV READY =="

#: what the validation must run, on executed lines rather than in comments: the resolver's
#: own consistency check, both halves' versions, the differentiable layer's import, an actual
#: calculation through it (the one check that reaches the compiled half), the package, and
#: the test runner the regression job needs
VALIDATION_SUBJECTS = ("pip check", "pyscfad.__version__", "pyscfadlib.__version__",
                       "from pyscfad import gto, dft", "mf.kernel()",
                       "import xcquinox", "pytest.__version__")

#: what the wheel route has no use for; any of these on an executed line is a source build
SOURCE_BUILD_RESIDUE = ("git+", "--no-build-isolation", "LDFLAGS", "cmake", "BUILD_TMP",
                        "conda env create", "environment-cluster-parity")

#: submission, completion and failure each reach the institutional inbox
MAIL_DIRECTIVES = ("#SBATCH --mail-user=alec.wills@stonybrook.edu",
                   "#SBATCH --mail-type=BEGIN,END,FAIL")


def _lines() -> list[str]:
    return JOB.read_text().splitlines()


def _executed(lines: list[str]) -> list[str]:
    """The lines the shell runs: everything that is not a comment (the ``#SBATCH``
    directives are comments to the shell and are read by their own tests)."""
    return [line for line in lines if not line.lstrip().startswith("#")]


def _index_of(lines: list[str], needle: str) -> int | None:
    """The first executed line carrying ``needle``. A comment line is not a step: the
    header names the ready marker in its watch instruction, before the job prints it."""
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        if needle in line:
            return idx
    return None


def _configured_prefix() -> str:
    """``cluster.conda_env`` of the v7 size configuration, the environment the campaigns
    run in and the value every standalone script's prefix is held to."""
    match = CONDA_ENV_VALUE.search(CONFIG.read_text())
    assert match, f"{CONFIG.name}: conda_env not found"
    return match.group("path")


def _requirement_pins() -> dict[str, str]:
    """``{distribution: version}`` of the exact pins of ``requirements.txt``."""
    out = {}
    for line in REQUIREMENTS.read_text().splitlines():
        spec = line.split("#")[0].strip()
        match = REQUIREMENT_PIN.match(spec) if spec else None
        if match:
            out[match.group("name").lower()] = match.group("version")
    return out


def test_the_prefix_is_the_campaign_environment_named_for_the_jax_release():
    match = ENV_ASSIGNMENT.search(JOB.read_text())
    assert match, "the job assigns no ENV"
    prefix = match.group("path")
    assert prefix == _configured_prefix(), \
        f"the job builds {prefix}, the configuration names {_configured_prefix()}"
    jax = _requirement_pins().get("jax")
    assert jax, "requirements.txt pins no jax"
    expected_tail = PREFIX_STEM + jax.replace(".", "")
    tail = prefix.rsplit("/", 1)[-1]
    assert tail == expected_tail, \
        f"the prefix ends in {tail}, the pinned jax {jax} names {expected_tail}"


def test_the_job_installs_the_requirements_file_and_pins_nothing_itself():
    """The one list of versions is ``requirements.txt``: the job installs it and states no
    version of its own, and none of the source-build recipe remains on an executed line."""
    executed = _executed(_lines())
    assert any(REQUIREMENTS_INSTALL in line for line in executed), \
        f"the job does not run {REQUIREMENTS_INSTALL!r}"
    pinned = [line.strip() for line in executed if VERSION_IN_JOB.search(line)]
    assert pinned == [], f"versions stated inside the job: {pinned}"
    residue = sorted({token for token in SOURCE_BUILD_RESIDUE
                      for line in executed if token in line})
    assert residue == [], f"source-build residue on executed lines: {residue}"
    assert "environment-cluster-parity" not in JOB.read_text(), \
        "the job still names the deleted environment file"


def test_the_guard_refuses_an_interpreter_outside_the_environment():
    """The activation is verified by its effect: the ``case`` block reads the probed python,
    accepts a path under the prefix, and exits non-zero for anything else."""
    lines = _lines()
    start = _index_of(lines, GUARD)
    assert start is not None, "no case block over the probed python"
    end = next((idx for idx in range(start, len(lines)) if lines[idx].strip() == "esac"),
               None)
    assert end is not None, "the case block never closes"
    block = lines[start:end + 1]
    assert any('"$ENV"/*)' in line for line in block), \
        "the guard has no arm accepting a python under the prefix"
    assert any(re.search(r"\bexit\s+[1-9]", line) for line in block), \
        "the guard carries no non-zero exit"


def test_the_job_mails_its_submission_completion_and_failure():
    text = JOB.read_text()
    missing = [directive for directive in MAIL_DIRECTIVES if directive not in text]
    assert missing == [], f"mail directives missing: {missing}"


