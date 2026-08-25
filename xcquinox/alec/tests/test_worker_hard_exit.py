"""Exit-status contract of the harness entry points.

A harness stage is judged by the exit status its process hands SLURM, and that
status is not the stage's verdict when the interpreter dies during teardown:
JAX's atexit backend cleanup can corrupt the glibc heap AFTER the work is done
and its outputs are on disk (``corrupted size vs. prev_size while
consolidating``, SIGABRT, return code -6). Measured on the workflow-matrix
smoke, cluster job 2134455 on node dn024: ``_pretrain`` ran both phases, wrote
``fidelity_certificate.json``, ``xnet.eqx`` and ``cnet.eqx``, logged ``pretrain
SUCCEEDED``, and then aborted -- so the stage was recorded as failed with
rc -6. In a campaign that status marks the pretrain array task FAILED and the
train array's ``afterok`` dependency never fires.

Two things are pinned here. The BEHAVIOUR, in real subprocesses with an
``atexit`` handler that calls ``os.abort()`` -- the discriminator, since the
real teardown fault is intermittent: the status is the return code the entry
point produced, not -6, and nothing buffered on stdout, stderr or an open log
file is lost by the exit that preserves it. And the SOURCE of every entry
point under ``cluster/`` and ``workers/``, so a worker added later cannot
regress silently: the enumeration is over the directories, not over a list.
"""
import ast
import os
import resource
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CLUSTER_DIR = REPO_ROOT / "xcquinox" / "alec" / "cluster"
WORKERS_DIR = REPO_ROOT / "xcquinox" / "alec" / "workers"
EXIT_MODULE = CLUSTER_DIR / "_exit.py"

#: The entry points spec 3.4's smoke and the campaign job graph actually
#: launch. The source pin enumerates the directories rather than this list --
#: the list is here so that an entry point DISAPPEARING from the enumeration
#: (a renamed module, a deleted ``__main__`` block) is caught too.
NAMED_ENTRY_POINTS = {
    "cluster/_pretrain.py", "cluster/_datagen.py", "cluster/_preflight.py",
    "cluster/_train_task.py", "cluster/_eval_one_spec.py",
    "cluster/_submit_eval.py", "cluster/validate_run.py",
    "cluster/workflow_matrix.py", "cluster/fidelity.py",
    "cluster/seed_cache.py", "cluster/coldstart_retro.py",
    "cluster/__main__.py",
    "workers/pretrain_worker.py", "workers/train_worker.py",
    "workers/test_worker.py", "workers/eval_holdout_worker.py",
    # rendered job stages that live one level up from the two directories
    "alec/benchmark_refs.py",
    "alec/_train_one_spec.py",
    # standalone job stages: launched by a checked-in sbatch script, so SLURM
    # reads the status each one hands back
    "alec/refinalize_verbatim.py",
    "hpcjobs/dfs6311_nan_isolate.py",
    "hpcjobs/dfs6311_nan_verify.py",
    "hpcjobs/dfs6311_pretrained_holdout.py",
    "hpcjobs/probe_pretrain_energy_weight.py",
    "analysis/precompute_scan_pool.py",
    "analysis/precompute_nonempirical_pool.py",
}

#: Entry points outside the two enumerated directories: job stages whose exit
#: status is likewise the scheduler's verdict, either rendered by ``submit``
#: or launched by a checked-in ``hpcjobs/*.sbatch``. The criterion for
#: inclusion is the exposure the helper closes: the stage runs under SLURM,
#: it loads JAX (so the atexit backend cleanup that aborted job 2134455 is
#: registered), and it writes outputs before returning, so an abort at
#: teardown reports FAILED for work that completed.
#:
#: Two SLURM-launched readers are deliberately absent.
#: ``hpcjobs/dfs6311_c2_ref_probe.py`` and
#: ``hpcjobs/dfs6311_lock_stamp_probe.py`` import neither ``xcquinox`` nor
#: JAX -- they read cached ``.npz`` files with numpy and write nothing -- so
#: no teardown handler is registered and an abort costs a re-read. Reaching
#: the helper through ``xcquinox.alec.cluster._exit`` would import JAX and
#: PySCF into those processes (measured: 1686 modules against a numpy-only
#: base), i.e. it would create the exposure it is meant to close.
EXTRA_ENTRY_FILES = (
    CLUSTER_DIR.parent / "benchmark_refs.py",
    CLUSTER_DIR.parent / "_train_one_spec.py",
    CLUSTER_DIR.parent / "refinalize_verbatim.py",
    REPO_ROOT / "hpcjobs" / "dfs6311_nan_isolate.py",
    REPO_ROOT / "hpcjobs" / "dfs6311_nan_verify.py",
    REPO_ROOT / "hpcjobs" / "dfs6311_pretrained_holdout.py",
    REPO_ROOT / "hpcjobs" / "probe_pretrain_energy_weight.py",
    REPO_ROOT / "notebooks" / "analysis" / "precompute_scan_pool.py",
    REPO_ROOT / "notebooks" / "analysis" / "precompute_nonempirical_pool.py",
)

#: SIGABRT as ``subprocess`` reports it. This is what the smoke recorded and
#: what every RED control below must produce.
ABORT_RC = -6


@pytest.fixture(scope="module", autouse=True)
def _no_core_dumps():
    """No cores for the deliberate aborts in this file.

    The limit is lowered on THIS process and inherited by every subprocess it
    launches, rather than being set child-side: the argument that would do
    that forces ``subprocess`` down the ``fork()`` path, and forking a process
    with JAX loaded is what the interpreter warns can deadlock the child.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
    resource.setrlimit(resource.RLIMIT_CORE, (0, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_CORE, (soft, hard))


def _run(script, *, env_extra=None, argv=(), cwd=None):
    """Run one script in a subprocess with stdout/stderr on PIPES.

    Pipes, not a terminal: that is what a SLURM log and a parent reading a
    worker's JSON result line are, and it is what makes stdout BLOCK-buffered,
    so a lost flush is observable. ``-u`` would defeat the point and is not
    passed.
    """
    env = os.environ.copy()
    env.pop("XCQ_NO_HARD_EXIT", None)
    env.pop("XCQ_NO_FLUSH_SWEEP", None)
    env.setdefault("JAX_PLATFORMS", "cpu")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(script), *argv], capture_output=True, text=True,
        timeout=600, env=env,
        cwd=str(cwd) if cwd else None,
    )


def _describe(proc):
    return (f"rc={proc.returncode}\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}")


# The abort is registered BEFORE the work, exactly as JAX's backend cleanup is
# registered when jax is imported: an atexit handler that runs only if the
# interpreter reaches teardown.
_PREAMBLE = """\
import atexit, importlib.util, os, sys
atexit.register(os.abort)
_spec = importlib.util.spec_from_file_location("_hard_exit", {exit_module!r})
_exit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exit)
"""

_FIXED = _PREAMBLE + """\
def main():
{body}
_exit.run_and_exit(main)
"""

#: The pre-change entry point, verbatim: ``sys.exit(main())``. Every
#: behavioural assertion below is paired against this so the test would FAIL
#: on the code as it stood when job 2134455 ran.
_STOCK = """\
import atexit, os, sys
atexit.register(os.abort)
def main():
{body}
sys.exit(main())
"""


def _script(tmp_path, template, body, name="entry.py"):
    path = tmp_path / name
    path.write_text(template.format(
        exit_module=str(EXIT_MODULE),
        body="\n".join("    " + line for line in body.strip("\n").split("\n")),
    ))
    return path


# --------------------------------------------------------------------------- #
# Behaviour: the status is the verdict, whatever teardown does
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rc", [0, 7])
def test_atexit_abort_does_not_hijack_the_exit_status(tmp_path, rc):
    """The measured defect: a completed worker reported as a signal death."""
    proc = _run(_script(tmp_path, _FIXED, f"return {rc}"))
    assert proc.returncode == rc, _describe(proc)


@pytest.mark.parametrize("rc", [0, 7])
def test_red_stock_exit_is_hijacked_by_the_atexit_abort(tmp_path, rc):
    """The same worker through ``sys.exit(main())``: the verdict is lost.

    This is the state the harness was in for cluster job 2134455 -- the pin
    that the test above measures a change rather than a property the code
    already had.
    """
    proc = _run(_script(tmp_path, _STOCK, f"return {rc}"))
    assert proc.returncode == ABORT_RC, _describe(proc)


def test_a_nonzero_verdict_is_never_swallowed(tmp_path):
    """The helper must not turn a failure into a success."""
    for rc in (1, 2, 3, 42):
        proc = _run(_script(tmp_path, _FIXED, f"return {rc}"))
        assert proc.returncode == rc, _describe(proc)


def test_none_is_success_and_systemexit_carries_its_code(tmp_path):
    """The interpreter's own status rules, so no consumer's contract moves.

    ``SystemExit(2)`` is argparse's usage exit -- every harness entry point
    parses its arguments inside ``main`` -- and a string payload is the
    interpreter's "message on stderr, status 1".
    """
    assert _run(_script(tmp_path, _FIXED, "return None")).returncode == 0
    assert _run(_script(tmp_path, _FIXED,
                        "raise SystemExit(2)")).returncode == 2
    assert _run(_script(tmp_path, _FIXED,
                        "raise SystemExit(0)")).returncode == 0
    proc = _run(_script(tmp_path, _FIXED, "raise SystemExit('refused: no')"))
    assert proc.returncode == 1, _describe(proc)
    assert "refused: no" in proc.stderr, _describe(proc)


def test_an_escaping_exception_keeps_status_one_and_prints_its_traceback(
        tmp_path):
    """An exception that escapes ``main`` is no reason to run the teardown.

    The status stays the interpreter's own 1 -- ``_train_task``'s four-way
    outcome rule and the sbatch epilogues already classify these codes -- and
    the traceback still reaches stderr, which is the SLURM log.
    """
    proc = _run(_script(tmp_path, _FIXED, "raise RuntimeError('boom')"))
    assert proc.returncode == 1, _describe(proc)
    assert "RuntimeError: boom" in proc.stderr, _describe(proc)
    assert "Traceback" in proc.stderr, _describe(proc)


def test_an_interrupt_exits_128_plus_sigint(tmp_path):
    proc = _run(_script(tmp_path, _FIXED, "raise KeyboardInterrupt"))
    assert proc.returncode == 130, _describe(proc)


def test_buffered_stdout_stderr_and_log_file_all_survive_the_exit(tmp_path):
    """``os._exit`` runs no finalizers, so the flush must be explicit.

    Everything below is written WITHOUT an explicit flush and with stdout on a
    pipe (block-buffered): the worker's JSON result line that
    ``parallel.run_workers`` parses, the tail of the SLURM log, and a log file
    the stage still holds open. The abort handler is registered as well, so
    this measures the flush and the status together.
    """
    log = tmp_path / "stage.log"
    body = (
        "sys.stdout.write('RESULT-LINE-ON-STDOUT\\n')\n"
        "sys.stderr.write('TAIL-ON-STDERR\\n')\n"
        f"fh = open({str(log)!r}, 'w')\n"
        "fh.write('LINE-IN-THE-LOG-FILE\\n')\n"
        "return 0"
    )
    proc = _run(_script(tmp_path, _FIXED, body))
    assert proc.returncode == 0, _describe(proc)
    assert "RESULT-LINE-ON-STDOUT" in proc.stdout, _describe(proc)
    assert "TAIL-ON-STDERR" in proc.stderr, _describe(proc)
    assert log.read_text() == "LINE-IN-THE-LOG-FILE\n"


def test_red_a_bare_os_exit_loses_the_buffered_writes(tmp_path):
    """The control for the flush: ``os._exit`` alone drops every buffer.

    Without this the flush requirement would be untested -- a test that passes
    against an unflushed exit proves nothing about the flush. stderr is
    excluded from the discriminator on purpose: the interpreter keeps it
    line-buffered even on a pipe, so it survives an unflushed exit and cannot
    distinguish anything. stdout on a pipe and an open log file are
    block-buffered, and both are lost here.
    """
    log = tmp_path / "stage.log"
    script = tmp_path / "bare.py"
    script.write_text(
        "import os, sys\n"
        "sys.stdout.write('RESULT-LINE-ON-STDOUT\\n')\n"
        "sys.stderr.write('TAIL-ON-STDERR\\n')\n"
        f"fh = open({str(log)!r}, 'w')\n"
        "fh.write('LINE-IN-THE-LOG-FILE\\n')\n"
        "os._exit(0)\n"
    )
    proc = _run(script)
    assert proc.returncode == 0, _describe(proc)
    assert proc.stdout == "", _describe(proc)
    assert log.read_text() == ""
    # stderr is line-buffered by the interpreter, so it arrives either way.
    assert "TAIL-ON-STDERR" in proc.stderr, _describe(proc)


def test_the_flush_reaches_a_log_file_opened_before_a_nonzero_verdict(
        tmp_path):
    """A failing stage owes its log too -- the flush is not tied to success."""
    log = tmp_path / "failed.log"
    body = (
        f"fh = open({str(log)!r}, 'w')\n"
        "fh.write('WHY-IT-FAILED\\n')\n"
        "return 5"
    )
    proc = _run(_script(tmp_path, _FIXED, body))
    assert proc.returncode == 5, _describe(proc)
    assert log.read_text() == "WHY-IT-FAILED\n"


def test_the_escape_hatch_restores_stock_teardown(tmp_path):
    """``XCQ_NO_HARD_EXIT=1`` is the documented way back to ``sys.exit``.

    It has to be exercised: it is what a coverage run or a teardown-fault
    investigation uses, and it is the RED control for the real entry points
    below.
    """
    proc = _run(_script(tmp_path, _FIXED, "return 0"),
                env_extra={"XCQ_NO_HARD_EXIT": "1"})
    assert proc.returncode == ABORT_RC, _describe(proc)


# --------------------------------------------------------------------------- #
# Behaviour: the REAL entry points, launched the way the job graph launches them
# --------------------------------------------------------------------------- #

@pytest.fixture
def abort_injection(tmp_path):
    """A ``sitecustomize`` that registers the abort handler in ANY subprocess.

    This is how a real entry point is put in the failing state without editing
    it: ``site`` imports ``sitecustomize`` at startup, so the handler is
    registered before the module runs, exactly as JAX registers its backend
    cleanup on import.
    """
    inject = tmp_path / "inject"
    inject.mkdir()
    (inject / "sitecustomize.py").write_text(
        "import atexit, os\natexit.register(os.abort)\n")
    return {"PYTHONPATH": str(inject)}


@pytest.mark.parametrize("entry,argv,expected_rc", [
    # Direct-file launch, the way parallel.run_workers starts a worker: the
    # argument parser refuses an empty argv with the usage exit.
    (WORKERS_DIR / "train_worker.py", (), 2),
    (WORKERS_DIR / "pretrain_worker.py", (), 2),
    (WORKERS_DIR / "test_worker.py", (), 2),
    (WORKERS_DIR / "eval_holdout_worker.py", (), 2),
])
def test_real_worker_entry_survives_the_abort(entry, argv, expected_rc,
                                              abort_injection):
    proc = _run(entry, argv=argv, env_extra=abort_injection)
    assert proc.returncode == expected_rc, _describe(proc)


@pytest.mark.parametrize("entry", [
    WORKERS_DIR / "train_worker.py",
    WORKERS_DIR / "eval_holdout_worker.py",
])
def test_red_real_worker_under_stock_teardown_aborts(entry, abort_injection):
    """The same launch with the hatch set is the pre-change behaviour."""
    env = dict(abort_injection)
    env["XCQ_NO_HARD_EXIT"] = "1"
    proc = _run(entry, env_extra=env)
    assert proc.returncode == ABORT_RC, _describe(proc)


@pytest.mark.parametrize("module,expected_rc", [
    ("xcquinox.alec.cluster.coldstart_retro", 2),
    ("xcquinox.alec.cluster.validate_run", 2),
])
def test_real_module_entry_survives_the_abort(module, expected_rc,
                                              abort_injection):
    """``python -m`` launch, the form every rendered sbatch script uses.

    This one imports the whole cluster package -- JAX included -- so it is the
    end-to-end case: the abort handler and JAX's own are both registered, and
    the usage exit still reaches the caller.
    """
    env = os.environ.copy()
    env.update(abort_injection)
    env.pop("XCQ_NO_HARD_EXIT", None)
    env.setdefault("JAX_PLATFORMS", "cpu")
    proc = subprocess.run(
        [sys.executable, "-m", module], capture_output=True, text=True,
        timeout=600, env=env, cwd=str(REPO_ROOT))
    assert proc.returncode == expected_rc, _describe(proc)


def test_red_real_module_entry_under_stock_teardown_aborts(abort_injection):
    env = os.environ.copy()
    env.update(abort_injection)
    env["XCQ_NO_HARD_EXIT"] = "1"
    env.setdefault("JAX_PLATFORMS", "cpu")
    proc = subprocess.run(
        [sys.executable, "-m", "xcquinox.alec.cluster.coldstart_retro"],
        capture_output=True, text=True, timeout=600, env=env,
        cwd=str(REPO_ROOT))
    assert proc.returncode == ABORT_RC, _describe(proc)


# --------------------------------------------------------------------------- #
# Source: every entry point ends in the helper
# --------------------------------------------------------------------------- #

def _main_block(path):
    """The module's ``if __name__ == "__main__":`` block, or None."""
    for node in ast.parse(path.read_text()).body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"):
            return node
    return None


def _called_name(call):
    """The dotted name a call node names, or '' for anything else."""
    parts, node = [], call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _entry_points():
    """Every module under cluster/ and workers/ with a ``__main__`` block.

    Enumerated from the DIRECTORIES: a worker added later is covered without
    anyone remembering to list it here, which is the only way this pin holds
    against a silent regression.
    """
    found = {}
    for directory in (CLUSTER_DIR, WORKERS_DIR):
        for path in sorted(directory.glob("*.py")):
            block = _main_block(path)
            if block is not None:
                found[f"{directory.name}/{path.name}"] = (path, block)
    for path in EXTRA_ENTRY_FILES:
        block = _main_block(path)
        if block is not None:
            found[f"{path.parent.name}/{path.name}"] = (path, block)
    return found


def test_the_enumeration_finds_every_named_entry_point():
    found = set(_entry_points())
    assert NAMED_ENTRY_POINTS <= found, NAMED_ENTRY_POINTS - found


@pytest.mark.parametrize("relpath", sorted(_entry_points()))
def test_every_entry_point_ends_in_the_shared_hard_exit(relpath):
    """The idiom is read out of the source, not inferred from a run.

    The teardown fault is intermittent -- a return-code assertion catches a
    lost hard exit only on the runs where the heap happens to be corrupted --
    so the pin is on the shape: the LAST statement of the block is the shared
    helper, and no ``sys.exit`` or ``raise SystemExit`` survives in it, since
    either one runs the teardown the helper exists to skip.
    """
    _path, block = _entry_points()[relpath]
    last = block.body[-1]
    assert isinstance(last, ast.Expr) and isinstance(last.value, ast.Call), (
        f"{relpath}: last statement is {ast.dump(last)}")
    assert _called_name(last.value).split(".")[-1] == "run_and_exit", relpath
    names = [_called_name(node) for node in ast.walk(block)
             if isinstance(node, ast.Call)]
    assert "sys.exit" not in names, relpath
    assert not any(isinstance(node, ast.Raise)
                   and isinstance(node.exc, ast.Call)
                   and _called_name(node.exc) == "SystemExit"
                   for node in ast.walk(block)), relpath


def test_the_helper_itself_ends_in_os_exit():
    """``hard_exit`` is where the process actually leaves.

    Its last statement must be ``os._exit`` -- anything after it is teardown
    -- and the flush must come first, because ``os._exit`` does not flush.
    """
    tree = ast.parse(EXIT_MODULE.read_text())
    functions = {node.name: node for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
    hard_exit = functions["hard_exit"]
    last = hard_exit.body[-1]
    assert isinstance(last, ast.Expr) and isinstance(last.value, ast.Call), (
        ast.dump(last))
    assert _called_name(last.value) == "os._exit"
    names = [_called_name(node) for node in ast.walk(hard_exit)
             if isinstance(node, ast.Call)]
    assert "flush_all" in names
    # run_and_exit must not exit by any other route.
    run_and_exit = functions["run_and_exit"]
    assert isinstance(run_and_exit.body[-1], ast.Expr)
    assert _called_name(run_and_exit.body[-1].value) == "hard_exit"


def test_the_workers_load_the_helper_without_importing_the_package():
    """The four direct-file workers must not import ``xcquinox`` to exit.

    They set their thread caps inside ``main`` BEFORE the first JAX import, so
    a package-qualified import in the entry block would pull
    ``xcquinox.alec.cluster`` -- and JAX with it -- before those caps are in
    place, and the module they load by path is the same one the cluster
    modules import.
    """
    for name in ("pretrain_worker", "train_worker", "test_worker",
                 "eval_holdout_worker"):
        path = WORKERS_DIR / f"{name}.py"
        tree = ast.parse(path.read_text())
        block = _main_block(path)
        assert block is not None, name
        # No import of the package anywhere the entry block or the module
        # scope reaches -- checked on the AST, so a comment naming the package
        # (which is where the reason is written down) cannot satisfy it.
        for node in list(ast.walk(block)) + list(tree.body):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("xcquinox"), name
            if isinstance(node, ast.Import):
                assert not any(alias.name.startswith("xcquinox")
                               for alias in node.names), name
        # The helper is reached by PATH, and it is the one in cluster/.
        constants = [node.value for node in ast.walk(block)
                     if isinstance(node, ast.Constant)
                     and isinstance(node.value, str)]
        assert "_exit.py" in constants, name
        assert "cluster" in constants, name
        calls = [_called_name(node) for node in ast.walk(block)
                 if isinstance(node, ast.Call)]
        assert any(c.endswith("spec_from_file_location") for c in calls), name


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
