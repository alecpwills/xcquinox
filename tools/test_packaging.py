"""The distribution declares what it needs and what it offers.

The rules, read from the files a user and a runner see:

* every third-party library the package imports is a declared runtime dependency, and
  everything the repository's own tests import is installable from the declaration, so
  that ``pip install xcquinox[test]`` gives a working import and a suite that runs;
* the floor of every declared library the parity file (``requirements.txt``) pins is that
  pin, the file pins every declared runtime and test-extra library exactly, and it pins each
  binary half (jaxlib, pyscfadlib) at its library's version, so that an install cannot
  resolve a stack the results were not produced on and the file describes one stack;
* the workflow raises the kernel's mapping ceiling before it runs the tests, since a
  single pytest process over this suite exhausts the default ceiling;
* no runtime dependency carries an upper bound of this file's own: the floors are one
  coherent set a resolver reaches, and the bounds a library states about its own
  dependencies are that library's to state and the resolver's to enforce;
* the conda test environment offers the floors and the caps the packaging file declares,
  and the packaging file is the only dependency list the index carries beside that
  environment and the parity file, so that no two files describe different stacks;
* the declared interpreter floor is the lowest version the workflow's matrix runs, the
  version the conda environment floors at, at or below the parity environment's pin, and
  at or above what every installed dependency requires of the interpreter, so that the
  floor is a version the tests are known to pass on and the libraries admit;
* every tracked non-module file under the package is carried by a package-data pattern,
  so that a wheel and a source distribution hold the data and the fixtures;
* the command-line harness is a console script whose target exists;
* the test configuration lives with the packaging metadata and nowhere else, so that one
  file states the test paths and the markers;
* the workflow runs on the branch the work happens on, installs the package with its
  declared dependencies rather than over a hand-written environment, and runs the tests.

The imports are read with the parser rather than by importing the package: a rule that
needed the quantum-chemistry stack on the path to say what the package needs would be
unusable in the one place it matters, a fresh environment.
"""
from __future__ import annotations

import ast
import configparser
import importlib.metadata
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _ROOT / "pyproject.toml"
_SETUP_CFG = _ROOT / "setup.cfg"
_WORKFLOW = _ROOT / ".github" / "workflows" / "CI.yaml"

#: the import name of a library whose distribution is named differently, or that another
#: distribution provides (jaxlib is jax's own binary half and comes with it)
DISTRIBUTION_OF = {"yaml": "pyyaml", "PIL": "pillow", "jaxlib": "jax"}

#: the branches the workflow must run on: the default branch and the branch the campaigns'
#: work happens on
WORKFLOW_BRANCHES = ("main", "alec_dev")

#: the directories pytest collects when it is given no path
TEST_PATHS = ("xcquinox/tests", "xcquinox/pipeline/tests", "tools", "notebooks", "hpcjobs")

#: the cluster parity file: the exact releases the stack resolves to on the workstation, the
#: one list the cluster's build job installs; the floor of every library it pins must be that
#: version, so that an install cannot silently resolve a stack the results were never
#: produced on
PARITY_FILE = "requirements.txt"

#: the conda spelling of a library whose distribution is named differently
PARITY_ALIAS = {"matplotlib-base": "matplotlib"}

#: the binary half each pinned library ships beside, which the parity environment must pin
#: at the library's own version; neither half is a declared dependency, so the floors rule
#: cannot see the two drift apart
PARITY_PAIRS = (("jax", "jaxlib"), ("pyscfad", "pyscfadlib"))

#: the conda environment the repository offers for the test suite; it stands in for the
#: declaration for a conda user, so every bound it carries must be the declared one
CONDA_TEST_ENVIRONMENT = "devtools/conda-envs/test_env.yaml"

#: the package's tracked non-module files that are NOT shipped; none at present.
UNSHIPPED = ()


def _config() -> dict:
    with open(_PYPROJECT, "rb") as fh:
        return tomllib.load(fh)


def _package_files() -> list[Path]:
    """Every tracked module of the package itself, its tests excluded: the tests may import
    what only a developer has."""
    out = subprocess.run(["git", "ls-files", "-z", "xcquinox"], cwd=_ROOT,
                         capture_output=True, check=True).stdout.decode("utf-8")
    return [_ROOT / p for p in out.split("\0")
            if p.endswith(".py") and "/tests/" not in p]


def _repository_modules() -> set[str]:
    """The module names this repository itself provides: a script that a test reaches by
    putting its directory on the path is not a distribution. Read from the index once."""
    global _REPOSITORY_MODULES
    if _REPOSITORY_MODULES is None:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=_ROOT, capture_output=True,
                             check=True).stdout.decode("utf-8")
        _REPOSITORY_MODULES = {Path(p).stem for p in out.split("\0") if p.endswith(".py")}
    return _REPOSITORY_MODULES


_REPOSITORY_MODULES = None


def imported_libraries(paths) -> set[str]:
    """The distribution names of the third-party libraries imported anywhere in ``paths``.

    The parser reads every ``import x`` and ``from x import y``, the ones inside functions
    included, since a deferred import is still a dependency at run time. Dropped: the
    package's own name, the standard library, relative imports, and a name that resolves to
    a module beside the importing file (the workers run as scripts and import their
    siblings that way, which no distribution provides)."""
    found: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    names.add(node.module.split(".")[0])
        found.update(name for name in names
                     if not (path.parent / f"{name}.py").is_file()
                     and name not in _repository_modules())
    found = {name for name in found
             if name != "xcquinox" and name not in sys.stdlib_module_names}
    return {DISTRIBUTION_OF.get(name, name) for name in found}


def declared_dependencies(config: dict) -> set[str]:
    """The distribution names of the declared runtime dependencies, without their bounds."""
    out = set()
    for spec in config.get("project", {}).get("dependencies", []):
        name = spec.split(";")[0]
        for sep in ("[", ">", "<", "=", "!", "~", " "):
            name = name.split(sep)[0]
        out.add(name.strip().lower())
    return out


def entry_points(config: dict) -> dict[str, str]:
    """The console scripts, as ``{name: "module:attribute"}``."""
    return dict(config.get("project", {}).get("scripts", {}))


def target_defines(target: str) -> bool:
    """True if ``module:attribute`` names a function defined at the top level of a module of
    this repository, read with the parser rather than imported."""
    module, _, attribute = target.partition(":")
    path = _ROOT / Path(*module.split("."))
    source = next((c for c in (path.with_suffix(".py"), path / "__init__.py")
                   if c.is_file()), None)
    if source is None:
        return False
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == attribute for node in tree.body)


def workflow_offences(workflow: dict) -> list[str]:
    """What the workflow does not do: the branches it misses on either trigger, an install
    step that refuses the declared dependencies, a missing install or test step, and a test
    step that runs before the kernel's mapping ceiling is raised.

    The ceiling rule reads the steps in order rather than as one text: a step that raises
    ``vm.max_map_count`` after the tests have run raises it for nothing."""
    out = []
    triggers = workflow.get(True, workflow.get("on", {}))
    for trigger in ("push", "pull_request"):
        branches = (triggers.get(trigger) or {}).get("branches", [])
        for branch in WORKFLOW_BRANCHES:
            if branch not in branches:
                out.append(f"{trigger} does not run on {branch}")
    runs = [step.get("run", "")
            for job in workflow.get("jobs", {}).values()
            for step in job.get("steps", [])]
    commands = "\n".join(runs)
    if "--no-deps" in commands:
        out.append("the install step refuses the declared dependencies (--no-deps)")
    if "pip install ." not in commands:
        out.append("no step installs the package from the tree")
    if "pytest" not in commands:
        out.append("no step runs the tests")
    else:
        tests = next(i for i, run in enumerate(runs) if "pytest" in run)
        ceiling = next((i for i, run in enumerate(runs) if "vm.max_map_count" in run), None)
        if ceiling is None or ceiling > tests:
            out.append("the tests run before the mapping ceiling is raised")
    return out


def test_the_rules_fire_on_fixtures():
    """Each rule fires on a configuration that breaks it and passes one that does not."""
    assert declared_dependencies({"project": {"dependencies": [
        "jax>=0.4.35", "pyyaml>=6.0", 'tqdm>=4.66; python_version>"3.9"', "ase[extra]>=3"]}}
    ) == {"jax", "pyyaml", "tqdm", "ase"}
    assert declared_dependencies({}) == set()
    assert entry_points({"project": {"scripts": {"x": "m:f"}}}) == {"x": "m:f"}
    assert not target_defines("xcquinox.pipeline.cluster.__main__:no_such_name")
    assert not target_defines("no.such.module:main")
    here = _ROOT / "xcquinox" / "pipeline" / "workers"
    assert "_cpu_bind" not in imported_libraries([here / "train_worker.py"])

    good = {"on": {"push": {"branches": ["main", "alec_dev"]},
                   "pull_request": {"branches": ["main", "alec_dev"]}},
            "jobs": {"test": {"steps": [{"run": "pip install .[test]"},
                                        {"run": "sudo sysctl -w vm.max_map_count=262144"},
                                        {"run": "pytest -m 'not slow'"}]}}}
    assert workflow_offences(good) == []
    narrow = {"on": {"push": {"branches": ["main"]}, "pull_request": {"branches": ["main"]}},
              "jobs": {"test": {"steps": [{"run": "pip install . --no-deps"}]}}}
    offences = workflow_offences(narrow)
    assert len(offences) == 4 and any("alec_dev" in o for o in offences), offences

    unraised = {"on": {"push": {"branches": ["main", "alec_dev"]},
                       "pull_request": {"branches": ["main", "alec_dev"]}},
                "jobs": {"test": {"steps": [{"run": "pip install .[test]"},
                                            {"run": "pytest -m 'not slow'"}]}}}
    assert workflow_offences(unraised) == [
        "the tests run before the mapping ceiling is raised"]
    late = {"on": {"push": {"branches": ["main", "alec_dev"]},
                   "pull_request": {"branches": ["main", "alec_dev"]}},
            "jobs": {"test": {"steps": [{"run": "pip install .[test]"},
                                        {"run": "pytest -m 'not slow'"},
                                        {"run": "sudo sysctl -w vm.max_map_count=262144"}]}}}
    assert workflow_offences(late) == [
        "the tests run before the mapping ceiling is raised"]

    assert parity_pins("jax==0.10.2\n# a comment\nPyYAML==6.0.3  # note\nnumpy>=2.5\n\n") == {
        "jax": "0.10.2", "pyyaml": "6.0.3"}
    assert loose_requirements("jax==0.10.2\nnumpy>=2.5\npyscf\n# c\n") == [
        "numpy>=2.5", "pyscf"]
    assert extra_names({"project": {"optional-dependencies": {"test": [
        "pytest>=8", "nbclient", "pillow>=10; python_version>'3.9'"]}}}, "test") == {
        "pytest", "nbclient", "pillow"}
    assert declared_floors({"project": {"dependencies": [
        "jax>=0.7.0", "pyscfad>=0.1.11,<0.2", "tqdm"]}}) == {
        "jax": "0.7.0", "pyscfad": "0.1.11"}
    config = {"tool": {"setuptools": {"package-data": {
        "xcquinox": ["data/*.md"], "xcquinox.pipeline.tests": ["fixtures/*.npz"]}}}}
    assert unshipped_data(config, ["xcquinox/data/README.md", "xcquinox/x.py",
                                   "xcquinox/pipeline/tests/fixtures/a.npz"]) == []
    assert unshipped_data(config, ["xcquinox/pipeline/data/pool.json"]) == [
        "xcquinox/pipeline/data/pool.json"]
    assert _version_tuple("3.9") < _version_tuple("3.10")
    assert min(["3.9", "3.10"], key=_version_tuple) == "3.9"     # "3.10" sorts first as text

    environment = ("name: test\n"
                   "channels:\n"
                   "  - conda-forge\n"
                   "dependencies:\n"
                   "    # Base depends\n"
                   "  - python >=3.11\n"
                   "  - pip\n"
                   "  - jax >=0.4.35\n"
                   "  - matplotlib-base >=3.8\n"
                   "  - pytest-cov\n"
                   "  - pandas <=3\n"
                   "  - pip:\n"
                   "    - pyscfad>=0.1.11,<0.2\n"
                   "    - brokenaxes>=0.6  # a comment\n")
    assert conda_specs(environment) == {
        "jax": ("0.4.35", None), "matplotlib": ("3.8", None), "pytest-cov": (None, None),
        "pandas": (None, None), "pyscfad": ("0.1.11", "0.2"),
        "brokenaxes": ("0.6", None)}
    assert conda_python_floor(environment) == "3.11"
    assert conda_python_floor("dependencies:\n  - pip\n") == ""
    assert parity_drift({"project": {"dependencies": ["jax>=0.10.2", "numpy>=2.5", "tqdm"]}},
                        {"jax": "0.10.2", "numpy": "2.5", "python": "3.12"}) == {}
    assert parity_drift({"project": {"dependencies": ["jax>=0.10.2"]}},
                        {"jax": "0.1"}) == {"jax": ("0.10.2", "0.1")}
    assert parity_drift({"project": {"dependencies": ["jax"]}},
                        {"jax": "0.10.2"}) == {"jax": (None, "0.10.2")}
    assert dependency_lists(["pyproject.toml", "requirements.txt", "docs/requirements.txt",
                             "environment-cluster-parity.yml", "devtools/conda-envs/x.yaml",
                             "hpcjobs/environment_notes.md"]) == [
        "devtools/conda-envs/x.yaml", "docs/requirements.txt",
        "environment-cluster-parity.yml", "requirements.txt"]
    assert conda_specs("dependencies:\n"
                       "  - jax >=0.10.2\n"
                       "# a comment at column zero does not end the block\n"
                       "  - pip:\n"
                       "    - pyscfad>=0.3.4\n") == {
        "jax": ("0.10.2", None), "pyscfad": ("0.3.4", None)}
    assert unpaired_parity_pins({"jax": "0.10.2", "jaxlib": "0.10.2", "pyscfad": "0.3.4",
                                 "pyscfadlib": "0.3.4", "numpy": "2.5"}) == {}
    assert unpaired_parity_pins({"jax": "0.10.2", "jaxlib": "0.7.0"}) == {
        "jaxlib": ("0.7.0", "0.10.2")}
    assert unpaired_parity_pins({"pyscfad": "0.3.4"}) == {"pyscfadlib": (None, "0.3.4")}
    assert unpaired_parity_pins({"numpy": "2.5"}) == {}
    assert required_python_floor(">=3.12") == "3.12"
    assert required_python_floor(">=3.9, <4") == "3.9"
    assert required_python_floor("") == ""
    assert installed_python_requirements(["no-such-distribution-xyz"]) == {}
    assert _version_tuple("3.12") > _version_tuple("3.11")
    declaration = {"project": {
        "dependencies": ["jax>=0.7.0", "pyscf>=2.11.0,<2.12", "scipy>=1.16"],
        "optional-dependencies": {"test": ["pandas>=2.0"], "docs": ["sphinx>=7.0"]}}}
    assert packaging_bounds(declaration) == {
        "jax": ("0.7.0", None), "pyscf": ("2.11.0", "2.12"), "scipy": ("1.16", None),
        "pandas": ("2.0", None)}
    assert conda_drift(declaration,
                       "dependencies:\n"
                       "  - jax >=0.4.35\n"
                       "  - scipy >=1.16\n"
                       "  - pip:\n"
                       "    - pyscf>=2.11.0\n") == {
        "jax": (("0.7.0", None), ("0.4.35", None)),
        "pyscf": (("2.11.0", "2.12"), ("2.11.0", None)),
        "pandas": (("2.0", None), (None, None))}
    assert conda_drift(declaration,
                       "dependencies:\n"
                       "  - jax >=0.7.0\n"
                       "  - scipy >=1.16\n"
                       "  - h5py\n"
                       "  - pip:\n"
                       "    - pyscf>=2.11.0,<2.12\n"
                       "    - pandas>=2.0\n") == {}
    assert declared_caps({"project": {"dependencies": [
        "pyscf>=2.11.0,<2.12", "numpy>=2.3", "jax<=0.8", 'tqdm<5; python_version>"3.9"']}}
    ) == {"pyscf": "2.12", "tqdm": "5"}
    assert declared_caps({"project": {"dependencies": ["numpy>=2.3", "tqdm"]}}) == {}

    assert python_floor({"project": {"requires-python": ">=3.11"}}) == "3.11"
    assert python_floor({"project": {}}) == ""
    low = {"jobs": {"test": {"strategy": {"matrix": {
        "python-version": ["3.10", "3.11", "3.12"]}}}}}
    assert workflow_python_versions(low) == ["3.10", "3.11", "3.12"]
    assert workflow_python_versions({"jobs": {"test": {"steps": []}}}) == []
    assert min(workflow_python_versions(low), key=_version_tuple) == "3.10"
    assert _version_tuple(python_floor({"project": {"requires-python": ">=3.11"}})) \
        != _version_tuple(min(workflow_python_versions(low), key=_version_tuple))
    lifted = {"jobs": {"test": {"strategy": {"matrix": {
        "python-version": ["3.11", "3.12"]}}}}}
    assert _version_tuple(python_floor({"project": {"requires-python": ">=3.11"}})) \
        == _version_tuple(min(workflow_python_versions(lifted), key=_version_tuple))


#: an exact pin of a requirements file
_EXACT_PIN = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[0-9][0-9A-Za-z.+!-]*)$")


def parity_pins(text: str) -> dict[str, str]:
    """``{distribution: version}`` for every exact pin (``name==version``) of a requirements
    file. Blank lines, comment lines and inline comments are skipped; a line that is not an
    exact pin is not read (:func:`loose_requirements` lists those)."""
    out = {}
    for line in text.splitlines():
        spec = line.split("#")[0].strip()
        match = _EXACT_PIN.match(spec) if spec else None
        if match:
            name = match.group("name").lower()
            out[PARITY_ALIAS.get(name, name)] = match.group("version")
    return out


def loose_requirements(text: str) -> list[str]:
    """The lines of a requirements file that are not exact pins, in file order: a range, a
    bare name or a URL describes a set of stacks, and the parity file describes one."""
    out = []
    for line in text.splitlines():
        spec = line.split("#")[0].strip()
        if spec and not _EXACT_PIN.match(spec):
            out.append(spec)
    return out


def extra_names(config: dict, extra: str) -> set[str]:
    """The distribution names of one optional-dependency group, without their bounds."""
    out = set()
    for spec in config.get("project", {}).get("optional-dependencies", {}).get(extra, []):
        name = spec.split(";")[0]
        for sep in ("[", ">", "<", "=", "!", "~", " "):
            name = name.split(sep)[0]
        out.add(name.strip().lower())
    return out


def declared_floors(config: dict) -> dict[str, str]:
    """``{distribution: floor}`` for every declared runtime dependency that carries one."""
    out = {}
    for spec in config.get("project", {}).get("dependencies", []):
        head = spec.split(";")[0]
        for clause in head.split(","):
            if ">=" in clause:
                name, _, version = clause.partition(">=")
                name = name.split("[")[0].strip().lower()
                if name:
                    out[name] = version.strip()
    return out


def conda_specs(text: str) -> dict[str, tuple[str | None, str | None]]:
    """``{distribution: (floor, cap)}`` for every library a conda environment file lists,
    across its conda list (``name >=version``) and its pip sub-list (``name>=version,<v``).

    The floor is the version after ``>=`` and the cap the version after a strict ``<``,
    each ``None`` where the entry carries no such clause, so that a bare name reads as
    ``(None, None)``. Only the ``dependencies:`` block is read (the ``channels:`` list is
    not a list of libraries), the interpreter and ``pip`` itself are not libraries of the
    declaration and are left out, comment text is dropped, and a conda spelling is
    translated the way the parity rule translates it."""
    return {name: bounds for name, bounds in _conda_entries(text)
            if name not in ("python", "pip")}


def conda_python_floor(text: str) -> str:
    """The interpreter floor the conda environment states (``- python >=3.11``), or ``""``
    when it states none."""
    for name, (floor, _cap) in _conda_entries(text):
        if name == "python":
            return floor or ""
    return ""


def _conda_entries(text: str):
    """``(distribution, (floor, cap))`` for every entry of the ``dependencies:`` block of
    a conda environment file, the pip sub-list included, in file order."""
    inside = False
    for line in text.splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if not line[0].isspace():
            inside = entry.split("#")[0].strip() == "dependencies:"
            continue
        if not inside or not entry.startswith("- "):
            continue
        spec = entry[2:].split("#")[0].strip().rstrip(":")
        head = spec
        for sep in (">", "<", "=", "!", "~", " "):
            head = head.split(sep)[0]
        name = head.strip().lower()
        if not name:
            continue
        floor = cap = None
        for clause in spec[len(head):].split(","):
            clause = clause.strip()
            if clause.startswith(">="):
                floor = clause[len(">="):].strip()
            elif clause.startswith("<") and not clause.startswith("<="):
                cap = clause[len("<"):].strip()
        yield PARITY_ALIAS.get(name, name), (floor, cap)


def packaging_bounds(config: dict) -> dict[str, tuple[str | None, str | None]]:
    """``{distribution: (floor, cap)}`` for every library the packaging file declares: the
    runtime dependencies and the test extra together, which is what an environment built
    from the declaration installs and so what a conda environment must offer. The
    environment marker beyond a ``;`` is not a bound and is dropped."""
    project = config.get("project", {})
    specs = list(project.get("dependencies", []))
    specs += list(project.get("optional-dependencies", {}).get("test", []))
    out: dict[str, tuple[str | None, str | None]] = {}
    for spec in specs:
        head = spec.split(";")[0]
        name = head
        for sep in ("[", ">", "<", "=", "!", "~", " "):
            name = name.split(sep)[0]
        name = name.strip().lower()
        if not name:
            continue
        floor = cap = None
        for clause in head.split(","):
            clause = clause.strip()
            if ">=" in clause:
                floor = clause.partition(">=")[2].strip()
            if "<" in clause and "<=" not in clause:
                cap = clause.partition("<")[2].strip()
        out[name] = (floor, cap)
    return out


def conda_drift(config: dict, text: str) -> dict[str, tuple]:
    """``{distribution: (declared bounds, conda bounds)}`` for every declared library the
    conda environment describes differently. A library the conda file omits altogether
    drifts with ``(None, None)`` on its side: a user who builds that environment is then
    without it. What the conda file lists beyond the declaration is its own affair and is
    not drift."""
    listed = conda_specs(text)
    out = {}
    for name, bounds in packaging_bounds(config).items():
        found = listed.get(name, (None, None))
        if found != bounds:
            out[name] = (bounds, found)
    return out


def parity_drift(config: dict, pins: dict[str, str]) -> dict[str, tuple[str | None, str]]:
    """``{distribution: (declared floor, parity pin)}`` for every declared runtime
    dependency the parity environment pins whose floor is not that pin, compared as
    versions rather than as text (a pin ``0.1`` is not the floor ``0.10.2``), a missing
    floor counting as drift: a declared library the parity environment pins is floored at
    exactly that version or the rule says so. A pin of a library the declaration does not
    name at all (a transitive one, or the interpreter) is not the declaration's to floor."""
    floors = declared_floors(config)
    out = {}
    for name in sorted(declared_dependencies(config)):
        if name not in pins:
            continue
        floor = floors.get(name)
        if floor is None or _version_tuple(floor) != _version_tuple(pins[name]):
            out[name] = (floor, pins[name])
    return out


def unpaired_parity_pins(pins: dict[str, str]) -> dict[str, tuple[str | None, str | None]]:
    """``{binary half: (its pin, the library's pin)}`` for every pair of :data:`PARITY_PAIRS`
    whose two pins are not one version, a missing half counting as ``None``. A parity file
    that pins jax at one release and jaxlib at another builds an environment ``pip check``
    refuses, and no declared floor sees it, since the half is not a declared dependency."""
    out = {}
    for library, half in PARITY_PAIRS:
        if library not in pins and half not in pins:
            continue
        if pins.get(library) != pins.get(half):
            out[half] = (pins.get(half), pins.get(library))
    return out


def required_python_floor(requires: str) -> str:
    """The interpreter floor a ``Requires-Python`` value states (its ``>=`` clause), or
    ``""`` when it states none."""
    for clause in requires.split(","):
        clause = clause.strip()
        if clause.startswith(">="):
            return clause[len(">="):].strip()
    return ""


def installed_python_requirements(names) -> dict[str, str]:
    """``{distribution: interpreter floor}`` for every named distribution that is installed
    in the running environment and states a ``Requires-Python`` floor in its own metadata;
    one that is not installed is left out, since the rule reads what the environment holds
    rather than the index."""
    out = {}
    for name in names:
        try:
            requires = importlib.metadata.metadata(name).get("Requires-Python") or ""
        except importlib.metadata.PackageNotFoundError:
            continue
        floor = required_python_floor(requires)
        if floor:
            out[name] = floor
    return out


def dependency_lists(tracked) -> list[str]:
    """The tracked files that describe a dependency set: every ``requirements*.txt``, every
    ``environment*.yml`` or ``.yaml`` at the root, and every file under the conda
    environments directory. The declaration is the packaging file; the two files the rules
    above hold to it are the only other lists the index may carry."""
    out = []
    for path in tracked:
        name = path.rsplit("/", 1)[-1]
        at_root = "/" not in path
        if name.startswith("requirements") and name.endswith(".txt"):
            out.append(path)
        elif at_root and name.startswith("environment") and name.endswith((".yml", ".yaml")):
            out.append(path)
        elif path.startswith("devtools/conda-envs/"):
            out.append(path)
    return sorted(out)


def declared_caps(config: dict) -> dict[str, str]:
    """``{distribution: cap}`` for every runtime dependency the packaging file holds below
    a strict upper bound. A ``<=`` clause is a pin rather than a cap and is not counted."""
    out = {}
    for spec in config.get("project", {}).get("dependencies", []):
        head = spec.split(";")[0]
        name = head
        for sep in ("[", ">", "<", "=", "!", "~", " "):
            name = name.split(sep)[0]
        name = name.strip().lower()
        if not name:
            continue
        for clause in head.split(","):
            clause = clause.strip()
            if "<" in clause and "<=" not in clause:
                out[name] = clause.partition("<")[2].strip()
    return out


def python_floor(config: dict) -> str:
    """The interpreter version the packaging file floors at, from ``requires-python``."""
    requires = config.get("project", {}).get("requires-python", "")
    return requires.partition(">=")[2].split(",")[0].strip()


def workflow_python_versions(workflow: dict) -> list[str]:
    """The interpreter versions the workflow's matrix runs, as the strings it names, read
    from the parsed workflow the branch rule is given."""
    out = []
    for job in workflow.get("jobs", {}).values():
        matrix = job.get("strategy", {}).get("matrix", {})
        out.extend(str(version) for version in matrix.get("python-version", []))
    return out


def _version_tuple(version: str) -> tuple[int, ...]:
    """A dotted version as integers, so that 3.10 orders above 3.9 rather than below it."""
    return tuple(int(part) for part in version.split(".") if part.isdigit())


def _test_path_files() -> list[Path]:
    """Every tracked module under the test paths."""
    out = subprocess.run(["git", "ls-files", "-z", *TEST_PATHS], cwd=_ROOT,
                         capture_output=True, check=True).stdout.decode("utf-8")
    return [_ROOT / p for p in out.split("\0") if p.endswith(".py")]


def unshipped_data(config: dict, paths) -> list[str]:
    """The package's tracked non-module files that no package-data pattern carries."""
    patterns = config.get("tool", {}).get("setuptools", {}).get("package-data", {})
    out = []
    for path in paths:
        if path.endswith(".py") or path in UNSHIPPED:
            continue
        inside = path[len("xcquinox/"):]
        shipped = False
        for package, globs in patterns.items():
            prefix = package.replace(".", "/")[len("xcquinox"):].lstrip("/")
            base = inside[len(prefix):].lstrip("/") if prefix else inside
            if prefix and not inside.startswith(prefix + "/"):
                continue
            if any(Path(base).match(glob) for glob in globs):
                shipped = True
                break
        if not shipped:
            out.append(path)
    return out


def test_every_imported_library_is_declared():
    """Every third-party library the package imports is a declared runtime dependency, and
    every declared dependency is imported: a list that drifts either way is a claim the
    code does not support."""
    imported = {name.lower() for name in imported_libraries(_package_files())}
    declared = declared_dependencies(_config())
    assert sorted(imported - declared) == [], \
        f"imported but not declared: {sorted(imported - declared)}"
    assert sorted(declared - imported) == [], \
        f"declared but not imported: {sorted(declared - imported)}"


def test_every_library_the_tests_import_is_installable_from_the_declaration():
    """Everything the repository's own tests import is either a runtime dependency or in the
    test extra, so that the workflow's one install gives a suite that runs."""
    imported = {name.lower() for name in imported_libraries(_test_path_files())}
    config = _config()
    declared = declared_dependencies(config)
    extra = set()
    for specs in config.get("project", {}).get("optional-dependencies", {}).values():
        extra |= declared_dependencies({"project": {"dependencies": specs}})
    available = declared | extra | {"pytest", "_pytest"}
    missing = sorted(name for name in imported if name not in available)
    assert missing == [], f"the tests import what no declaration installs: {missing}"


def _parity_text() -> str:
    return (_ROOT / PARITY_FILE).read_text(encoding="utf-8")


def test_the_floors_are_the_stack_the_campaigns_ran():
    """Every declared floor of a library the parity file pins is that pin: the stack the
    results are produced on, and the one list the cluster's build job installs."""
    pins = parity_pins(_parity_text())
    drifted = parity_drift(_config(), pins)
    assert drifted == {}, f"floors that are not the parity version: {drifted}"
    assert {"jax", "pyscfad", "pyscf", "equinox", "optax", "numpy", "scipy"} <= set(pins), \
        sorted(pins)


def test_the_parity_pins_pair_their_binary_halves():
    """The parity file pins jaxlib at jax's version and pyscfadlib at pyscfad's: each half is
    one release with its library, which the floors rule cannot check because neither half is
    a declared dependency."""
    unpaired = unpaired_parity_pins(parity_pins(_parity_text()))
    assert unpaired == {}, f"binary halves pinned apart from their library: {unpaired}"


def test_the_parity_file_pins_the_whole_declared_set_exactly():
    """Every line of the parity file is an exact pin, and every runtime and test-extra
    library of the declaration is among them: a range or a missing library describes a set
    of stacks, and the parity file describes the one the results are produced on."""
    text = _parity_text()
    loose = loose_requirements(text)
    assert loose == [], f"{PARITY_FILE} lines that are not exact pins: {loose}"
    config = _config()
    declared = declared_dependencies(config) | extra_names(config, "test")
    missing = sorted(declared - set(parity_pins(text)))
    assert missing == [], f"declared libraries {PARITY_FILE} does not pin: {missing}"


def test_the_python_floor_admits_the_installed_stack():
    """The declared interpreter floor is at or above what every installed dependency
    requires of the interpreter, read from the distributions' own metadata: numpy 2.5 and
    scipy 1.18 admit nothing below 3.12, and a floor the four files agree on but no library
    admits is a declaration nothing can install."""
    floor = python_floor(_config())
    required = installed_python_requirements(sorted(declared_dependencies(_config())))
    assert required, "no declared dependency is installed with a Requires-Python floor"
    below = {name: need for name, need in required.items()
             if _version_tuple(need) > _version_tuple(floor)}
    assert below == {}, \
        f"the declared floor is {floor} but the installed stack requires: {below}"


def test_the_python_floor_is_the_workflow_matrix_minimum():
    """The declared interpreter floor is the lowest version the workflow's matrix runs (a
    floor below the matrix is a claim no run supports, and a floor above it spends a matrix
    entry on a version the declaration refuses to install on) and the version the conda test
    environment floors at. The parity file cannot pin the interpreter; the cluster's build
    job creates it at this floor, which ``hpcjobs/test_build_parity_env.py`` holds."""
    yaml = pytest.importorskip("yaml")
    with open(_WORKFLOW, encoding="utf-8") as fh:
        workflow = yaml.safe_load(fh)
    versions = workflow_python_versions(workflow)
    assert versions, "the workflow's matrix names no interpreter version"
    lowest = min(versions, key=_version_tuple)
    floor = python_floor(_config())
    assert _version_tuple(floor) == _version_tuple(lowest), \
        f"the declared floor is {floor} and the matrix runs {versions}"
    conda = conda_python_floor((_ROOT / CONDA_TEST_ENVIRONMENT).read_text(encoding="utf-8"))
    assert _version_tuple(conda) == _version_tuple(floor), \
        f"the declared floor is {floor} and the conda environment floors at {conda!r}"


def test_the_declaration_is_the_only_dependency_list():
    """Beside the packaging file, the index carries exactly two dependency lists, the conda
    test environment and the parity file, both held to the declaration by the rules above;
    any other requirements file or environment file is a stack description nothing checks."""
    tracked = subprocess.run(["git", "ls-files"], cwd=_ROOT, capture_output=True,
                             check=True).stdout.decode("utf-8").split()
    assert dependency_lists(tracked) == sorted([PARITY_FILE, CONDA_TEST_ENVIRONMENT]), \
        f"dependency lists in the index: {dependency_lists(tracked)}"


def test_the_conda_environment_carries_the_declared_bounds():
    """The conda test environment offers every declared library, runtime and test extra
    alike, at the floor and the cap the packaging file declares: a conda user who builds it
    gets the stack the declaration describes, not a second one."""
    text = (_ROOT / CONDA_TEST_ENVIRONMENT).read_text(encoding="utf-8")
    drift = conda_drift(_config(), text)
    assert drift == {}, \
        f"declared bounds the conda environment does not carry: {drift}"


def test_no_runtime_dependency_carries_a_cap_of_ours():
    """No runtime dependency is held below an upper bound written here. The declared floors
    are the coherent set a resolver reaches around pyscfad 0.3.4 (2026-09-21), and the
    bounds a library states about its own dependencies -- pyscfad's ``jax<0.11`` -- are the
    resolver's to enforce; a cap written here fences a defect in instead of fixing it."""
    caps = declared_caps(_config())
    assert caps == {}, f"upper bounds this file imposes on a dependency: {caps}"


def test_the_packages_data_files_are_declared():
    """Every tracked non-module file under the package is carried by a package-data pattern,
    so that a wheel and a source distribution hold the data and the fixtures. The exemption
    must name tracked files only: an entry for a file that is no longer there exempts
    nothing today and would exempt it silently if it came back."""
    tracked = subprocess.run(["git", "ls-files", "xcquinox"], cwd=_ROOT,
                             capture_output=True, check=True).stdout.decode("utf-8").split()
    stale = [path for path in UNSHIPPED if path not in tracked]
    assert stale == [], f"the exemption names what is not tracked: {stale}"
    missing = unshipped_data(_config(), tracked)
    assert missing == [], f"tracked under the package but shipped by nothing: {missing}"


def test_the_console_script_points_at_a_callable():
    """The harness is a console script, and its target is defined where it says."""
    scripts = entry_points(_config())
    assert "xcquinox-cluster" in scripts, scripts
    assert target_defines(scripts["xcquinox-cluster"]), scripts["xcquinox-cluster"]


def test_the_pytest_configuration_lives_in_the_packaging_file():
    """One file states the test paths and the markers: the packaging file, not setup.cfg."""
    options = _config().get("tool", {}).get("pytest", {}).get("ini_options", {})
    assert sorted(options.get("testpaths", [])) == sorted(TEST_PATHS), options
    assert any(marker.startswith("slow:") for marker in options.get("markers", [])), options
    parser = configparser.ConfigParser()
    parser.read(_SETUP_CFG)
    assert not parser.has_section("tool:pytest"), "setup.cfg still carries a pytest section"


def test_the_workflow_runs_on_the_working_branch_and_installs_what_is_declared():
    """The workflow runs on both branches, installs the package with its dependencies and
    runs the tests."""
    yaml = pytest.importorskip("yaml")
    with open(_WORKFLOW, encoding="utf-8") as fh:
        workflow = yaml.safe_load(fh)
    assert workflow_offences(workflow) == []


def test_the_test_extra_carries_the_notebook_kernel():
    """The end-to-end notebook tests execute the tracked notebooks through nbclient, which
    needs a kernel for the interpreter running the tests; ``ipykernel`` provides it and no
    test imports it, so the import rule above cannot see it and this rule holds it (an
    environment without it resolves the kernel named python3 to whatever registers the
    name elsewhere, which would run a notebook on another stack)."""
    assert "ipykernel" in extra_names(_config(), "test"), \
        "the test extra does not carry ipykernel"
