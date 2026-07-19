"""Exit-code contract of the session hard-exit hook in this directory's
conftest.py.

The hook exists because JAX's atexit backend cleanup can corrupt the
interpreter heap AFTER a fully green pytest summary ("corrupted size vs.
prev_size", SIGABRT, exit code 134; observed on cluster regression job
2091615 batch 2), flipping exit-code consumers to a false red. Each test
copies the real conftest into an isolated directory and runs a real pytest
session in a subprocess, pinning: status propagation (0 pass / 1 fail), the
summary reaching stdout through the explicit flush, interpreter atexit being
skipped (the discriminating behavior stock teardown does not have), pytest-cov
XML surviving the hard exit (the CI consumer), and the XCQ_NO_HARD_EXIT
escape hatch restoring stock teardown.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CONFTEST = Path(__file__).resolve().parent / "conftest.py"

# Registers an interpreter-atexit hook that force-exits with 77. Under stock
# teardown the handler runs after pytest returns and the process exits 77;
# with the hard-exit hook the interpreter never reaches atexit and the
# session's own status is preserved.
ATEXIT_PROBE = (
    "import atexit, os\n"
    "atexit.register(os._exit, 77)\n"
    "def test_ok():\n"
    "    assert True\n"
)


def _run_session(tmp_path, test_body, *, extra_args=(), env_extra=None):
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "test_case.py").write_text(test_body)
    env = os.environ.copy()
    env.pop("XCQ_NO_HARD_EXIT", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path), "-q",
         "-p", "no:randomly", "-p", "no:cacheprovider", *extra_args],
        capture_output=True, text=True, timeout=300, env=env,
        cwd=str(tmp_path),
    )


def _describe(proc):
    return f"rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_passing_session_exits_zero_with_summary(tmp_path):
    proc = _run_session(tmp_path, "def test_ok():\n    assert True\n")
    assert proc.returncode == 0, _describe(proc)
    assert "1 passed" in proc.stdout, _describe(proc)


def test_failing_session_exits_one(tmp_path):
    proc = _run_session(tmp_path, "def test_bad():\n    assert False\n")
    assert proc.returncode == 1, _describe(proc)
    assert "1 failed" in proc.stdout, _describe(proc)


def test_interpreter_atexit_is_skipped(tmp_path):
    proc = _run_session(tmp_path, ATEXIT_PROBE)
    assert proc.returncode == 0, _describe(proc)
    assert "1 passed" in proc.stdout, _describe(proc)


def test_escape_hatch_restores_stock_teardown(tmp_path):
    proc = _run_session(tmp_path, ATEXIT_PROBE,
                        env_extra={"XCQ_NO_HARD_EXIT": "1"})
    assert proc.returncode == 77, _describe(proc)


def test_cov_xml_survives_hard_exit(tmp_path):
    pytest.importorskip("pytest_cov")
    (tmp_path / "modx.py").write_text("def f():\n    return 1\n")
    body = "import modx\n\ndef test_ok():\n    assert modx.f() == 1\n"
    cov_xml = tmp_path / "cov.xml"
    proc = _run_session(
        tmp_path, body,
        extra_args=("--cov=modx", f"--cov-report=xml:{cov_xml}"))
    assert proc.returncode == 0, _describe(proc)
    assert "1 passed" in proc.stdout, _describe(proc)
    assert cov_xml.exists() and cov_xml.stat().st_size > 0, _describe(proc)
