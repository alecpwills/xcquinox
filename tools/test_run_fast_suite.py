"""The fast suite's chunk protocol: what runs where, and that nothing falls between.

A single pytest process over this suite aborts at the kernel's mapping ceiling
(``vm.max_map_count``, 65530 on the workstation) on jaxlib 0.10.2: each compiled executable
leaves mappings behind, and one quarter of the pipeline tests reached 65235 before the
process died. The suite is therefore run as a sequence of interpreters, each measured below
the ceiling, and ``tools/run_fast_suite.py`` is that sequence.

Splitting a suite introduces one failure a green run cannot show: a module that is in no
chunk simply stops running, and nothing says so. The rules here are therefore about
coverage rather than about speed --

* every tracked test module of the five test paths lies in exactly one chunk, a chunk named
  by a directory holding every tracked module under it, and no chunk names a path the index
  does not carry;
* the three parts the training-gradient module is divided into by ``-k`` select each of its
  top-level test functions exactly once, so that dividing a module by name can neither drop
  a test nor run one twice;
* each chunk carries the mapping peak measured for it, below the ceiling;
* each chunk's command runs pytest under the marker over the chunk's paths into one
  temporary directory, with the keyword expression where the chunk carries one;
* the sequence can be printed without being run.

The runner is imported inside each test rather than at the top of the module, so that its
absence is a failure of the rules it would satisfy rather than a collection error that
takes the whole file with it.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = _ROOT / "tools"

#: the directories pytest collects when it is given no path, as the packaging file states
TEST_PATHS = ("xcquinox/tests", "xcquinox/pipeline/tests", "tools", "notebooks", "hpcjobs")

#: the module the keyword chunks divide between them: its tests are the slowest and the
#: most mapping-hungry of the suite, and it does not fit one interpreter whole
SPLIT_MODULE = "xcquinox/pipeline/tests/test_training_gradient_consistency.py"

#: the mapping ceiling of the hosts the suite runs on, which the split exists to stay under
MEASURED_CEILING = 65530


def _runner():
    """``tools/run_fast_suite.py``, under the name pytest gives a module of this directory
    (the directory of a collected file is on the path; the repository is not a package)."""
    if str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))
    return importlib.import_module("run_fast_suite")


def _tracked_test_modules() -> list[str]:
    """Every tracked ``test_*.py`` under the five test paths, at any depth: what a run of
    the suite with no path collects, which is what the chunks together must be."""
    out = subprocess.run(["git", "ls-files", "-z", *TEST_PATHS], cwd=_ROOT,
                         capture_output=True, check=True).stdout.decode("utf-8")
    return sorted(path for path in out.split("\0")
                  if path.endswith(".py") and Path(path).name.startswith("test_"))


def _chunk_modules(chunk, tracked: list[str]) -> set[str]:
    """The tracked test modules a chunk runs: a path naming a module is that module, a path
    naming a directory is every tracked module under it, at any depth."""
    found: set[str] = set()
    for path in chunk.paths:
        if path.endswith(".py"):
            found.add(path)
        else:
            prefix = path.rstrip("/") + "/"
            found |= {module for module in tracked if module.startswith(prefix)}
    return found


def _owner(chunk) -> tuple:
    """What counts as one chunk for the partition. The parts of a module divided by keyword
    are several commands over one module and own that module together; every other chunk
    owns what it names alone."""
    return tuple(chunk.paths) if chunk.keyword else (chunk.name,)


def test_the_chunks_partition_the_tracked_test_modules():
    """Every tracked test module of the five test paths is run by exactly one chunk, and no
    chunk names a path the index does not carry. A module in no chunk stops running when
    the suite is run this way and reports nothing; a module in two runs twice and pays for
    it in the wall the split exists to buy."""
    runner = _runner()
    tracked = _tracked_test_modules()
    assert tracked, "no tracked test module under the test paths"

    named = [path for chunk in runner.CHUNKS for path in chunk.paths]
    assert named, "the chunks name no path"
    strays = sorted(path for path in named if path.endswith(".py") and path not in tracked)
    assert strays == [], f"chunks name modules the index does not carry: {strays}"
    missing_dirs = sorted(path for path in named
                          if not path.endswith(".py") and not (_ROOT / path).is_dir())
    assert missing_dirs == [], f"chunks name directories that do not exist: {missing_dirs}"

    owners: dict[str, set] = {}
    for chunk in runner.CHUNKS:
        for module in _chunk_modules(chunk, tracked):
            owners.setdefault(module, set()).add(_owner(chunk))
    uncovered = sorted(module for module in tracked if module not in owners)
    doubled = sorted(module for module, who in owners.items() if len(who) > 1)
    assert uncovered == [], f"tracked test modules no chunk runs: {uncovered}"
    assert doubled == [], f"tracked test modules more than one chunk runs: {doubled}"


def test_every_chunk_carries_its_measured_peak_below_the_ceiling():
    """The chunks are the measured division, not a guess: each carries the mapping peak
    measured for it and every peak is below the ceiling the undivided quarter died at, so
    that a chunk added without a measurement, or one measured at the ceiling, is refused
    here rather than at the abort."""
    runner = _runner()
    assert runner.MARKER == "not slow and not oracle", runner.MARKER
    assert dataclasses.is_dataclass(runner.Chunk), runner.Chunk
    assert runner.Chunk.__dataclass_params__.frozen, "Chunk is not frozen"
    names = [chunk.name for chunk in runner.CHUNKS]
    assert len(set(names)) == len(names), f"chunk names repeat: {names}"
    unmeasured = [chunk.name for chunk in runner.CHUNKS
                  if not isinstance(chunk.peak, int) or chunk.peak <= 0]
    assert unmeasured == [], f"chunks with no measured mapping peak: {unmeasured}"
    over = {chunk.name: chunk.peak for chunk in runner.CHUNKS
            if chunk.peak >= MEASURED_CEILING}
    assert over == {}, f"chunks measured at or above the ceiling {MEASURED_CEILING}: {over}"


def test_the_keyword_parts_select_each_test_of_their_module_once():
    """The module the parts divide is divided by test function, so the division is checked
    against the functions themselves: every top-level ``test_*`` function the module defines
    is selected by exactly one part, and every name a part's expression carries is a
    function the module defines. An ``-k`` expression that is an ``or`` of plain names
    selects a function whose name is one of them, which is how the parts are written."""
    runner = _runner()
    parts = [chunk for chunk in runner.CHUNKS if chunk.keyword]
    assert parts, "no chunk divides a module by keyword"
    for chunk in parts:
        assert tuple(chunk.paths) == (SPLIT_MODULE,), (chunk.name, chunk.paths)

    tree = ast.parse((_ROOT / SPLIT_MODULE).read_text(encoding="utf-8"))
    functions = [node.name for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name.startswith("test_")]
    assert functions, f"{SPLIT_MODULE} defines no top-level test function"

    selected: dict[str, list[str]] = {name: [] for name in functions}
    for chunk in parts:
        names = [token for token in chunk.keyword.split() if token != "or"]
        unknown = sorted(set(names) - set(functions))
        assert unknown == [], f"{chunk.name}: -k names no such test function: {unknown}"
        for name in names:
            selected[name].append(chunk.name)
    unselected = sorted(name for name, who in selected.items() if not who)
    repeated = sorted(name for name, who in selected.items() if len(who) > 1)
    assert unselected == [], f"tests of {SPLIT_MODULE} no part selects: {unselected}"
    assert repeated == [], f"tests of {SPLIT_MODULE} several parts select: {repeated}"


def test_the_commands_carry_the_marker_the_paths_and_the_temporary_directory(tmp_path):
    """One command per chunk: the running interpreter, pytest under the marker, the chunk's
    paths and the one temporary directory the suite is given, with the keyword expression
    where the chunk carries one and no ``-k`` where it does not. The shared temporary
    directory is what keeps a chunked run from leaving a directory per interpreter."""
    runner = _runner()
    basetemp = tmp_path / "suite"
    argvs = runner.commands(_ROOT, basetemp, tmp_path / "logs")
    assert len(argvs) == len(runner.CHUNKS), argvs
    for chunk, argv in zip(runner.CHUNKS, argvs):
        assert argv[0] == sys.executable, (chunk.name, argv)
        assert argv[1:3] == ["-m", "pytest"], (chunk.name, argv)
        assert argv[3] == "-m" and argv[4] == runner.MARKER, (chunk.name, argv)
        for path in chunk.paths:
            assert path in argv, (chunk.name, path, argv)
        assert f"--basetemp={basetemp}" in argv, (chunk.name, argv)
        if chunk.keyword:
            assert "-k" in argv[5:], (chunk.name, argv)
            assert argv[argv.index("-k", 5) + 1] == chunk.keyword, (chunk.name, argv)
        else:
            assert "-k" not in argv[5:], (chunk.name, argv)
    named = runner.commands(_ROOT, basetemp, tmp_path / "logs", python="/usr/bin/python3")
    assert [argv[0] for argv in named] == ["/usr/bin/python3"] * len(runner.CHUNKS)


def test_the_dry_run_prints_one_command_per_chunk():
    """The sequence can be read before it is run: the runner, called as a program from the
    repository root, prints one pytest command per chunk and exits zero without running
    anything."""
    runner = _runner()
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", JAX_PLATFORMS="cpu")
    proc = subprocess.run([sys.executable, "-m", "tools.run_fast_suite", "--dry-run"],
                          cwd=_ROOT, capture_output=True, text=True, env=environment)
    assert proc.returncode == 0, f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    printed = [line for line in proc.stdout.splitlines() if line.strip()]
    assert len(printed) == len(runner.CHUNKS), printed
    for line in printed:
        assert "pytest" in line and runner.MARKER in line, line
