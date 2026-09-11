"""Parity-environment activation in the standalone batch scripts.

The scripts submitted by hand (as opposed to the task scripts rendered by the
submission harness, which carry their own activation preamble) inherit whatever
interpreter the submitting shell exported. On a compute node that interpreter is
the base one, so the first repository import fails and the job exits within
seconds having consumed an allocation and produced nothing; the queue records a
completed job and the failure is visible only in the output file (job 2159273,
2026-09-10, ``No module named 'jax'`` after 3 s). Four properties are pinned for
every ``hpcjobs/*.sbatch`` except the script that builds the environment.

ORDER: the environment prefix, the activation call and the by-effect guard all
appear before the first python INVOCATION, an invocation being the interpreter
as a command word (at the head of a line, or after whitespace, ``;``, ``|``,
``&``, ``(`` or a backtick), with the block's own ``command -v python`` probe and
commented mentions excluded. A block placed after the first invocation
activates nothing for it.

REFUSAL: the guard accepts only an interpreter resolving inside the prefix and
leaves a fatal message and a non-zero exit for anything else, so a job whose
activation silently failed stops instead of running under the base interpreter.
Conda may return non-zero on a successful activation of a prefix path (the
record in ``nonempirical_pool.sbatch`` and HISTORY 2026-08-13), which is why the
effect rather than the return code is read.

IDENTITY: the block is the one in service in ``nonempirical_pool.sbatch``,
differing only in the log tag; every script's tag is read from its own echo
line and substituted before the comparison, so divergence in the prefix path,
the profile script or the guard is caught. ``workflow_matrix.sbatch`` refuses
through its own ``fatal`` helper and is exempt from the identity comparison only.

ORACLE: the prefix and the profile path equal the values the harness renders
from the v7 configuration (``cluster.conda_env`` and ``cluster.conda_profile`` of
``dfs_step7.dfs6311_grid3_v7g1_size.yaml``), so the standalone scripts cannot
drift from the environment the campaign runs in.

A syntax check over each script guards the insertions themselves.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

HPCJOBS = Path(__file__).resolve().parent
CONFIG = HPCJOBS / "configs" / "dfs_step7.dfs6311_grid3_v7g1_size.yaml"

REFERENCE = "nonempirical_pool.sbatch"
# builds the environment the others activate; it has no interpreter call to guard
EXEMPT = ("build_parity_env.sbatch",)
# refuses through its own ``fatal`` helper: order and refusal are checked, the
# byte identity of the block is not
IDENTITY_EXEMPT = ("workflow_matrix.sbatch",)

ALL_SCRIPTS = tuple(sorted(p.name for p in HPCJOBS.glob("*.sbatch") if p.name not in EXEMPT))
IDENTITY_SCRIPTS = tuple(s for s in ALL_SCRIPTS if s != REFERENCE and s not in IDENTITY_EXEMPT)

# The block proper: from its comment line to ``esac``, eleven lines. The
# ``ENV_PREFIX=`` assignment sits above it, adjacent in the newer scripts and
# among the other path variables in the older ones, and is checked on its own.
BLOCK_COMMENT = "# --- conda activation (verified by effect, not by return code)"
BLOCK_LINES = 11

PREFIX_ASSIGNMENT = "ENV_PREFIX="
# ``ENV_PREFIX=<path>`` or ``ENV_PREFIX="${XCQ_ENV_PREFIX:-<path>}"``; likewise
# ``CONDA_PROFILE=`` in the script that sources the profile through a variable.
ASSIGNED_PATH = re.compile(r'^(?P<name>ENV_PREFIX|CONDA_PROFILE)="?(?:\$\{[A-Z_]+:-)?(?P<path>/[^}"\s]+)\}?"?\s*$')
PROFILE_SOURCE = re.compile(r'^\s*(?:source|\.) (?:"?\$CONDA_PROFILE"?|(?P<path>\S+/etc/profile\.d/conda\.sh))\s*$')
FUNCTION_OPEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\(\) \{\s*$")
ACTIVATE_LINE = 'conda activate "$ENV_PREFIX"'
GUARD_OPENING = 'case "$PYBIN" in'
GUARD_ACCEPT_ARM = '"$ENV_PREFIX"/*) : ;;'
TAG_LINE = re.compile(r'^echo "(\[[^"\]]+\]) host=\$\(hostname\) python=\$PYBIN"$')

# An invocation, not a mention: the interpreter as a command word, at the head of
# a line or after whitespace, ";", "|", "&", "(" or a backtick, so that "$(python",
# "<(python", "exec python", "time python" and "... && python" all count, followed
# by whitespace or the end of the line. "python=$PYBIN" (the block's own echo) and
# PYTHONPATH are not matched; the block's "command -v python" probe is excluded
# by name. A versioned name such as python3.11 is not an invocation here.
PYTHON_INVOCATION = re.compile(r"(?:^|[ \t;|&(`])python3?(?=[ \t]|$)")
PROBE = "command -v python"


def _lines(script: str) -> list[str]:
    return (HPCJOBS / script).read_text().splitlines()


def _index_of(lines: list[str], needle: str) -> int | None:
    for idx, line in enumerate(lines):
        if needle in line:
            return idx
    return None


def _first_python_invocation(lines: list[str]) -> int | None:
    """The first interpreter invocation executed at the top level of the script.

    A function body (``name() {`` to its closing ``}`` at column 0) is skipped:
    it runs when the function is called, which the scripts do after their
    activation. Comment lines and the block's own probe are skipped as well.
    """
    in_function = False
    for idx, line in enumerate(lines):
        if in_function:
            if line.startswith("}"):
                in_function = False
            continue
        if FUNCTION_OPEN.match(line):
            in_function = True
            continue
        if line.lstrip().startswith("#"):
            continue
        if PROBE in line:
            continue
        if PYTHON_INVOCATION.search(line):
            return idx
    return None


def _activation_block(script: str) -> list[str] | None:
    """The eleven block lines of ``script``, located by the block's comment line."""
    lines = _lines(script)
    start = next((i for i, line in enumerate(lines) if line.startswith(BLOCK_COMMENT)), None)
    if start is None:
        return None
    return lines[start:start + BLOCK_LINES]


def _assigned_paths(lines: list[str]) -> dict[str, str]:
    """``ENV_PREFIX`` and ``CONDA_PROFILE`` values, the default of a ``${VAR:-path}`` form."""
    found = {}
    for line in lines:
        m = ASSIGNED_PATH.match(line)
        if m and m.group("name") not in found:
            found[m.group("name")] = m.group("path")
    return found


def _tag(block: list[str]) -> str | None:
    for line in block:
        m = TAG_LINE.match(line)
        if m:
            return m.group(1)
    return None


def _config_values() -> tuple[str, str]:
    """``cluster.conda_env`` and ``cluster.conda_profile`` of the v7 size configuration."""
    text = CONFIG.read_text()
    env = re.search(r"^\s*conda_env:\s*(\S+)\s*$", text, re.M)
    profile = re.search(r"^\s*conda_profile:\s*(\S+)\s*$", text, re.M)
    assert env and profile, f"{CONFIG.name}: conda_env / conda_profile not found"
    return env.group(1), profile.group(1)


def test_the_script_set_is_the_directory():
    """The universal quantifier of the tests below is the directory listing, not a hand list."""
    assert len(ALL_SCRIPTS) >= 19, ALL_SCRIPTS
    assert REFERENCE in ALL_SCRIPTS
    for name in EXEMPT + IDENTITY_EXEMPT:
        assert (HPCJOBS / name).is_file(), name


@pytest.mark.parametrize("script", ALL_SCRIPTS)

def test_every_standalone_script_activates_before_its_first_python_call(script):
    lines = _lines(script)

    prefix_idx = next((i for i, line in enumerate(lines)
                       if line.strip().startswith(PREFIX_ASSIGNMENT)), None)
    activate_idx = _index_of(lines, ACTIVATE_LINE)
    guard_idx = _index_of(lines, GUARD_OPENING)
    assert prefix_idx is not None, f"{script}: no {PREFIX_ASSIGNMENT} line"
    assert activate_idx is not None, f"{script}: no {ACTIVATE_LINE!r} line"
    assert guard_idx is not None, f"{script}: no {GUARD_OPENING!r} guard"

    first_call = _first_python_invocation(lines)
    assert first_call is not None, (
        f"{script}: no python invocation found -- the ordering assertion below "
        "would hold vacuously"
    )

    for label, idx in (
        (PREFIX_ASSIGNMENT, prefix_idx),
        (ACTIVATE_LINE, activate_idx),
        (GUARD_OPENING, guard_idx),
    ):
        assert idx < first_call, (
            f"{script}: {label!r} at line {idx + 1} does not precede the first "
            f"python invocation at line {first_call + 1}: {lines[first_call]!r}"
        )


@pytest.mark.parametrize("script", ALL_SCRIPTS)

def test_the_guard_refuses_a_python_outside_the_prefix(script):
    lines = _lines(script)
    text = "\n".join(lines)
    assert GUARD_ACCEPT_ARM in text, f"{script}: no {GUARD_ACCEPT_ARM!r} arm"

    # The refusing arm: from the guard's "*)" line to its "esac". The refusal must
    # sit inside it (an ``exit 1`` or a call to the script's ``fatal`` helper), so
    # an exit elsewhere in the script cannot satisfy the check and the arm cannot
    # fall through to the next command.
    guard_idx = _index_of(lines, GUARD_OPENING)
    assert guard_idx is not None, f"{script}: no {GUARD_OPENING!r} guard"
    arm_idx = next((i for i in range(guard_idx, len(lines))
                    if lines[i].strip().startswith("*)")), None)
    assert arm_idx is not None, f"{script}: no '*)' arm after the guard"
    esac_idx = next((i for i in range(arm_idx, len(lines)) if lines[i].strip() == "esac"), None)
    assert esac_idx is not None, f"{script}: the guard has no esac"
    arm = "\n".join(lines[arm_idx:esac_idx])
    assert "not active" in arm, f"{script}: the '*)' arm carries no refusal message: {arm!r}"
    assert re.search(r"\bexit 1\b", arm) or re.search(r"\bfatal ", arm), (
        f"{script}: no exit 1 or fatal call inside the '*)' arm; saw {arm!r}"
    )


@pytest.mark.parametrize("script", IDENTITY_SCRIPTS)

def test_the_block_is_the_reference_block_up_to_its_tag(script):
    reference = _activation_block(REFERENCE)
    assert reference is not None, f"{REFERENCE}: no block comment line"
    assert len(reference) == BLOCK_LINES, f"{REFERENCE}: block runs off the end of the file"
    assert reference[-1].strip() == "esac", (
        f"{REFERENCE}: block located at its comment line does not end at its guard; "
        f"last line {reference[-1]!r}"
    )
    reference_tag = _tag(reference)
    assert reference_tag is not None, f"{REFERENCE}: no tagged echo line in the block"

    actual = _activation_block(script)
    assert actual is not None, f"{script}: no block comment line"
    tag = _tag(actual)
    assert tag is not None, f"{script}: no tagged echo line in its block: {actual!r}"
    expected = [line.replace(reference_tag, tag) for line in reference]
    assert actual == expected, (
        f"{script}: activation block differs from {REFERENCE} up to its tag "
        f"({len(actual)} lines against {len(expected)})\n"
        + "\n".join(
            f"  line {i + 1}: {got!r} != {want!r}"
            for i, (got, want) in enumerate(zip(actual, expected))
            if got != want
        )
    )


@pytest.mark.parametrize("script", ALL_SCRIPTS)

def test_the_prefix_and_profile_are_the_campaign_configuration_values(script):
    conda_env, conda_profile = _config_values()
    lines = _lines(script)
    assigned = _assigned_paths(lines)
    assert assigned.get("ENV_PREFIX") == conda_env, (
        f"{script}: ENV_PREFIX {assigned.get('ENV_PREFIX')!r} is not the configuration's "
        f"conda_env {conda_env}")
    sources = []
    for line in lines:
        m = PROFILE_SOURCE.match(line)
        if m:
            sources.append(m.group("path") or assigned.get("CONDA_PROFILE"))
    assert sources == [conda_profile], (
        f"{script}: profile sources {sources} are not the configuration's {conda_profile}")


@pytest.mark.parametrize("script", ALL_SCRIPTS + EXEMPT)

def test_scripts_parse(script):
    proc = subprocess.run(
        ["bash", "-n", str(HPCJOBS / script)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"{script}: bash -n rc={proc.returncode}\n{proc.stderr}"
