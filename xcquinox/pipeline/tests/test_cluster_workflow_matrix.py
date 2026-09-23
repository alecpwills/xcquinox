"""Tests for the per-architecture workflow matrix
(``xcquinox.pipeline.cluster.workflow_matrix``).

The matrix drives the harness stage sequence at a tiny def2-svp /
grid-level-1 identity for every registered architecture.
What the module carries at this point
is that identity: the checked-in template, the renderer that writes one
architecture's grid config, and the staging of the cached inputs the identity
consumes. These tests cover those three; they start no stage and run no SCF,
so the whole file runs in seconds.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from xcquinox.pipeline.cluster import workflow_matrix as wm
from xcquinox.pipeline.config import ARCHITECTURES


def cache_missing_reason(repo_root):
    """The refusal the cache-bound requirements below skip with, or None.

    The step-7 cache is untracked (.gitignore excludes
    ``notebooks/checkpoints_step7/``), so a fresh clone, a worktree and the
    cluster repository carry neither half of it; the requirements that consume
    it are skipped there rather than failed, with the module's own refusal as
    the reason so the wording cannot drift from what the code raises. Presence
    alone is not the criterion, as it is not in ``staged_refs_dir``: a
    references directory left empty by an interrupted copy would otherwise let
    those requirements run and one of them compare two empty listings as
    equal. The directory must carry references by the same disjunction
    ``staged_refs_dir`` applies to a supplied one, a complete staging manifest
    or species ``.npz`` files; the repository's own cache carries the files and
    no manifest, and a staged copy carries both.
    """
    try:
        refs = wm.cached_refs_dir(repo_root)
        if not (wm._stage_is_complete(refs) or any(refs.glob("*.npz"))):
            staging = wm._missing_cache_message(refs, "x").split(". ", 1)[1]
            raise wm.CachedInputsMissing(
                f"cached CCSD references at {refs} carry neither a complete "
                f"staging manifest nor any species .npz. {staging}")
        wm.cached_ledger_path(repo_root)
    except wm.CachedInputsMissing as exc:
        return str(exc)
    return None


#: the reason every cache-bound requirement skips with where the cache is absent
_CACHE_MISSING = cache_missing_reason(wm.repo_root_path())
_needs_cache = pytest.mark.skipif(_CACHE_MISSING is not None,
                                  reason=_CACHE_MISSING or "")


# ---------------------------------------------------------------------------
# Template + renderer
# ---------------------------------------------------------------------------

def test_template_exists_and_is_package_data():
    path = wm.template_path()
    assert path.is_file(), path
    assert not (path.parent / "__init__.py").exists(), (
        "cluster/examples/ ships as package DATA, not as a subpackage")


#: A mail address in any shape. The template must carry none, anywhere.
_ADDRESS_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def test_template_carries_no_address_and_no_account():
    """The template is rendered in dry-run only and never submitted, so it
    carries no mail address and no allocation; a shipped example with a real
    address would mail a person from anybody's copy.

    Blank ``mail_user``/``mail_type`` is a deliberate deviation from the rule
    that every cluster job script carries the Stony Brook address on
    BEGIN/END/FAIL: nothing rendered from this file is queued, and a blank mail
    block also keeps a matrix config from being mistaken for a production one.

    The WHOLE text is scanned rather than the tail after ``cluster:``: an
    address in the header comment mails a person just as effectively as one in
    the mail field.
    """
    text = wm.template_path().read_text()
    found = _ADDRESS_RE.findall(text)
    assert not found, f"mail address(es) in the shipped template: {found}"
    assert '\n  mail_user: ""\n' in text
    assert '\n  account: ""\n' in text


@_needs_cache
def test_stage_cached_inputs_copies_the_refs_out_of_the_repository(tmp_path):
    staged = wm.stage_cached_inputs(tmp_path, repo_root=wm.repo_root_path())
    refs = Path(staged["external_refs_dir"])
    assert refs.is_dir()
    assert not refs.is_symlink()
    assert (refs / "H2O.npz").is_file()
    assert (refs / "_intermediates" / "HO_g1_scf.npz").is_file()
    assert str(refs).startswith(str(tmp_path))
    # The run log precompute_all writes on every call must land here, never in
    # the tracked tree.
    assert not any(p.name.startswith("_run_log_") for p in refs.iterdir())
    assert Path(staged["subset_ledger_path"]).is_file()
    # The completion manifest is what a later call reads to decide whether the
    # copy is whole; the temporary copy directory is renamed away, not left.
    assert (refs / wm.STAGE_MARKER).is_file()
    assert not list(refs.parent.glob("external_refs.partial-*"))


def _staged_relpaths(root: Path) -> set:
    """Every staged file below ``root``, relative, excluding the manifest."""
    return {str(p.relative_to(root)) for p in root.rglob("*")
            if p.is_file() and p.name != wm.STAGE_MARKER}


def _cached_relpaths() -> set:
    """The reference cache as it stands in the tree, run logs excluded."""
    src = wm.repo_root_path() / wm.CACHED_REFS_RELPATH
    return {str(p.relative_to(src)) for p in src.rglob("*")
            if p.is_file() and not p.name.startswith("_run_log_")}


@_needs_cache
def test_stage_cached_inputs_never_publishes_an_interrupted_copy(tmp_path,
                                                                  monkeypatch):
    """An interrupted copy must stay under its ``.partial-<pid>`` name.

    The copy is built beside the destination and moved in with ``os.replace``
    only after the manifest is written, so a copy that dies part-way (a full
    filesystem, a killed job) leaves nothing at the destination path. Copying
    into the destination directly and writing the manifest at the end would
    leave a half-populated directory that the next call has to repair; here
    there is nothing to repair.
    """
    real_copy2 = shutil.copy2
    state = {"n": 0, "fail_after": 20}

    def counting_copy2(src, dst, *args, **kwargs):
        state["n"] += 1
        if state["n"] > state["fail_after"]:
            raise OSError(28, "No space left on device")
        return real_copy2(src, dst, *args, **kwargs)
    monkeypatch.setattr(shutil, "copy2", counting_copy2)

    with pytest.raises(shutil.Error):
        wm.stage_cached_inputs(tmp_path, repo_root=wm.repo_root_path())
    refs = tmp_path / "_inputs" / "external_refs"
    assert not refs.exists(), (
        "an interrupted copy was published at the destination path")
    partial = refs.parent / f"external_refs.partial-{os.getpid()}"
    assert partial.is_dir(), "the interrupted copy is kept under its own name"
    assert not (partial / wm.STAGE_MARKER).exists(), (
        "the manifest is written before the rename, not before the copy")

    state["fail_after"] = float("inf")
    staged = wm.stage_cached_inputs(tmp_path, repo_root=wm.repo_root_path())
    assert Path(staged["external_refs_dir"]) == refs
    # 165 files at the measured cache size.
    assert _staged_relpaths(refs) == _cached_relpaths()
    assert not partial.exists(), (
        "this process's partial copy must be cleared, not accumulated")


def _fake_staged_refs(path) -> Path:
    """A directory carrying what a staged reference copy carries.

    A supplied ``external_refs_dir`` is checked for the staging manifest or a
    species ``.npz``, so a test that needs only a supplied PATH (rather than
    the 74 MB of references behind it) builds one here instead of staging.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "H2O.npz").write_bytes(b"")
    return path


@_needs_cache
def test_write_matrix_yaml_renders_one_arch_and_two_cells(tmp_path):
    from xcquinox.pipeline.cluster.grid_config import expand_grid, load_grid_config
    out = tmp_path / "deep_3x16"
    path = wm.write_matrix_yaml("deep_3x16", out, repo_root=wm.repo_root_path())
    assert path == out.resolve() / "grid.yaml"
    cfg = load_grid_config(str(path))
    assert list(cfg.sweep.arch) == ["deep_3x16"]
    cells = expand_grid(cfg)
    assert len(cells) == 2
    assert sorted(c.subset_size for c in cells) == [1, 2]
    assert cfg.hyperparams.n_steps == 3
    assert cfg.pretrain.n_steps == 50
    assert cfg.pretrain.atoms == (("H", 1), ("O", 2))
    assert cfg.cluster.eval_workers == 1
    assert cfg.inputs.benchmark_refs_dir is None
    assert cfg.inputs.val_refs_dir is None


@_needs_cache
def test_write_matrix_yaml_paths_are_absolute_and_outside_the_repository(
        tmp_path):
    from xcquinox.pipeline.cluster.grid_config import load_grid_config
    out = tmp_path / "deep"
    cfg = load_grid_config(str(wm.write_matrix_yaml(
        "deep", out, repo_root=wm.repo_root_path())))
    # Real paths on both sides: the rendered paths are resolved, so a repository
    # reached through a symlink (or a cache directory that is one, as it is on
    # cluster scratch) would otherwise make a string prefix answer a question
    # about identity.
    repo = os.path.realpath(wm.repo_root_path())
    for value in (cfg.inputs.external_refs_dir, cfg.inputs.output_root,
                  cfg.pretrain.data_dir):
        assert os.path.isabs(value), value
        assert not os.path.realpath(value).startswith(repo + os.sep), value
    # The ledger is READ-ONLY (only the JSON is read; no subset.traj is
    # opened), so it is consumed in place: the rendered path is the cached
    # ledger itself, not a copy of it under the work root.
    assert os.path.realpath(cfg.inputs.subset_ledger_path) == os.path.realpath(
        wm.repo_root_path() / wm.CACHED_LEDGER_RELPATH)
    assert os.path.isfile(cfg.inputs.subset_ledger_path)


def test_write_matrix_yaml_refuses_an_unregistered_architecture(tmp_path):
    with pytest.raises(ValueError, match="not a registered architecture"):
        wm.write_matrix_yaml("no_such_arch", tmp_path / "x",
                             repo_root=wm.repo_root_path())


# ---------------------------------------------------------------------------
# Oracle selector
# ---------------------------------------------------------------------------

def _compile_k(expression_text: str):
    """Compile a ``-k`` expression with pytest's own parser.

    The parser is the oracle for whether ``-k`` accepts an expression, and its
    evaluator is the oracle for what the expression then selects; matching it
    by hand would share the selector's own assumptions. It is private API, so
    the import is local: a rename in a future pytest breaks the tests that
    call it rather than the collection of the whole file.
    """
    from _pytest.mark.expression import Expression
    return Expression.compile(expression_text)


def _item_names(node_id: str, module: str | None = None) -> set:
    """The names pytest matches a ``-k`` term against for one collected test.

    ``KeywordMatcher.from_item`` gathers the name of the item and of each of
    its parents except the session and the root directory: for a test in
    ``<repo>/xcquinox/pipeline/tests/<module>`` that is the three directory
    components below the root, the module file name, and the test name
    carrying its parametrisation id.
    """
    module = f"{wm.ORACLE_MODULE}.py" if module is None else module
    return {"xcquinox", "pipeline", "tests", module, node_id}


def _k_matcher(names):
    """pytest's ``-k`` matching rule: a term matches when it is a
    case-insensitive substring of any of the item's names
    (``KeywordMatcher.__call__``)."""
    def matcher(ident, /, **kwargs):
        assert not kwargs, f"unexpected call parameters on {ident!r}"
        return any(ident.lower() in name.lower() for name in names)
    return matcher


def test_oracle_selector_names_the_module_and_the_architecture():
    got = wm.oracle_selector("deep_rung35_mgga_3x16")
    assert got.startswith("test_spin_scaling_oracles and ")
    assert " and deep_rung35_mgga_3x16" in got


def test_oracle_selector_excludes_a_name_it_is_embedded_in():
    """Containment is not the same rule as prefixing: -k matches anywhere in
    the id, so a registry name sitting in the MIDDLE of a longer one has to be
    excluded as well. Every containment in the registry as it stands is also a
    prefix, so the general rule is pinned on an injected registry instead --
    a later entry named ``mgga`` would otherwise carry every ``*_mgga_*``
    architecture's oracles into its own selection.
    """
    got = wm.oracle_selector("mgga", archs=["mgga", "deep_mgga_3x16"])
    assert got == "test_spin_scaling_oracles and mgga and not deep_mgga_3x16"
    expr = _compile_k(got)
    assert expr.evaluate(_k_matcher(
        _item_names("test_o1_uniform_scaling[mgga]")))
    assert not expr.evaluate(_k_matcher(
        _item_names("test_o1_uniform_scaling[deep_mgga_3x16]")))


def test_oracle_selector_refuses_an_unregistered_architecture():
    with pytest.raises(ValueError, match="not a registered architecture"):
        wm.oracle_selector("no_such_arch")


def _collect_oracles(log_path, selector) -> tuple:
    """Collect the oracle module under ``selector``; return (rc, node ids).

    The collection runs into a log file rather than a pipe, so a run that
    hangs or dies part-way leaves its output on disk.
    """
    with Path(log_path).open("w") as fh:
        rc = subprocess.run(
            [sys.executable, "-m", "pytest", wm.ORACLE_TEST_TARGET,
             "--collect-only", "-q", "-p", "no:randomly", "-k", selector],
            cwd=str(wm.repo_root_path()), stdout=fh,
            stderr=subprocess.STDOUT, check=False).returncode
    text = Path(log_path).read_text()
    return rc, [ln.strip() for ln in text.splitlines()
                if f"{wm.ORACLE_MODULE}.py::" in ln], text


@pytest.fixture(scope="module")
def collected_oracle_ids(tmp_path_factory):
    """Every test the spec-3.1 oracle module collects, as pytest reports it.

    One collection serves every architecture's check below: those evaluate
    their selectors against these REAL node ids under pytest's own matching
    rule, which is what a synthetic id cannot settle -- an oracle function
    named after an architecture, a parametrisation id in the other order, or
    an architecture the module does not cover shows up here and nowhere else.
    Skipped until the module is installed, so this file is executable on its
    own.
    """
    module = (wm.repo_root_path() / "xcquinox" / "pipeline" / "tests"
              / f"{wm.ORACLE_MODULE}.py")
    if not module.is_file():
        pytest.skip(f"{module} not installed yet")
    log = tmp_path_factory.mktemp("oracle_collect") / "collect.log"
    rc, node_ids, text = _collect_oracles(log, wm.ORACLE_MODULE)
    assert rc == 0, text
    assert node_ids, text
    return node_ids


def _names_of(node_id: str) -> set:
    """The names pytest matches a ``-k`` term against, for a collected id."""
    path, _, name = node_id.partition("::")
    return _item_names(name, module=Path(path).name)


def _arch_params(node_id: str) -> list:
    """The parametrisation ids of a collected node, or [] when it carries
    none. Stacked ``parametrize`` puts the architecture either first or last
    in the bracket depending on the decorator order, so the architecture is
    read as a PARAMETER rather than as a suffix."""
    name = node_id.split("::")[-1]
    if "[" not in name or not name.endswith("]"):
        return []
    return name[name.index("[") + 1:name.rindex("]")].split("-")


# ---------------------------------------------------------------------------
# run_arch, driven by a fake runner (no subprocess is started)
# ---------------------------------------------------------------------------

def _template_fidelity() -> dict:
    """The shipped template's ``fidelity`` block, read from the file.

    The waiver's reason string is checked against the template rather than
    restated: the certificate copies it verbatim, and ``gate_certificate``
    re-checks it there, so a test carrying its own copy would keep passing
    after the template's reason changed.
    """
    import yaml
    return yaml.safe_load(wm.template_path().read_text())["fidelity"]


def _certificate_payload(verdict) -> dict:
    """The fields of a certificate that the on-node gates read.

    ``cluster/fidelity`` releases a FAIL only for a certificate recording the
    JSON literal ``false`` in ``enforced`` AND a non-empty string in
    ``tolerances.override_reason``, so a fake certificate carrying only a
    verdict would be refused where the real one is released.
    """
    return {"verdict": verdict, "enforced": False,
            "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                           "override_reason":
                               _template_fidelity()["override_reason"]}}


class FakeRunner:
    """Stand-in for ``subprocess.run``.

    Records ``(argv, env)`` per call, echoes the submit stage's run-dir line
    into the stage log, materializes whatever artefacts the caller asked for,
    and returns the return code scheduled for that stage name.
    """

    def __init__(self, run_dir, *, rc_by_stage=None, artefacts=(),
                 verdict="PASS", oracle_summary="12 passed in 3.4s"):
        self.run_dir = Path(run_dir)
        self.rc_by_stage = dict(rc_by_stage or {})
        self.artefacts = tuple(artefacts)
        self.verdict = verdict
        self.oracle_summary = oracle_summary
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs.get("env") or {})))
        stream = kwargs.get("stdout")
        stage = self._stage_of(argv)
        # The tag goes FIRST: run_arch reads the LAST non-empty line of the
        # oracle log as its summary line.
        stream.write(f"[fake] {stage}\n")
        if stage == "submit":
            self.run_dir.mkdir(parents=True, exist_ok=True)
            stream.write(f"submit: created run dir {self.run_dir}\n")
            stream.write(f"submit: run dir = {self.run_dir}\n")
        elif stage == "certificate" and self.verdict is not None:
            cert = self.run_dir / "pretrain" / "deep" / \
                "fidelity_certificate.json"
            cert.parent.mkdir(parents=True, exist_ok=True)
            cert.write_text(json.dumps(_certificate_payload(self.verdict)))
        elif stage == "oracles":
            stream.write("......\n")
            stream.write(f"{self.oracle_summary}\n")
        if stage == "validate_run":
            for rel in self.artefacts:
                target = self.run_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("x")

        class _Completed:
            returncode = self.rc_by_stage.get(stage, 0)

        return _Completed()

    @staticmethod
    def _stage_of(argv):
        joined = " ".join(argv)
        if " -m pytest " in f" {joined} ":
            return "oracles"
        for token, name in (
                ("xcquinox.pipeline.cluster._datagen", "datagen"),
                ("xcquinox.pipeline.cluster._pretrain", "pretrain"),
                ("xcquinox.pipeline.cluster.fidelity", "certificate"),
                ("xcquinox.pipeline.cluster._preflight", "preflight"),
                ("xcquinox.pipeline.cluster._train_task", "train"),
                ("xcquinox.pipeline.cluster._eval_one_spec", "eval"),
                ("xcquinox.pipeline.cluster.validate_run", "validate_run"),
        ):
            if token in joined:
                return name
        return "submit"


def _run_arch(tmp_path, arch="deep", **kw):
    fake = kw.pop("fake", None)
    run_dir = tmp_path / arch / "runs" / "run_20260821T000000Z"
    if fake is None:
        fake = FakeRunner(run_dir)
    # A supplied external_refs_dir is an INPUT and is refused when it carries
    # no references (staged_refs_dir), so the stand-in is built rather than
    # named: these tests exercise the stage sequence, not the 74 MB copy.
    result = wm.run_arch(arch, tmp_path, runner=fake,
                         repo_root=wm.repo_root_path(),
                         external_refs_dir=_fake_staged_refs(
                             tmp_path / "refs"), **kw)
    return result, fake


@_needs_cache
def test_run_arch_runs_the_ten_stages_in_order(tmp_path):
    result, fake = _run_arch(tmp_path)
    assert [s["name"] for s in result["stages"]] == list(wm.STAGE_ORDER)
    assert [s["rc"] for s in result["stages"]] == [0] * len(wm.STAGE_ORDER)
    assert result["arch"] == "deep"
    assert result["run_dir"].endswith("run_20260821T000000Z")
    assert result["seconds"] >= 0.0


@_needs_cache
def test_run_arch_issues_the_exact_stage_command_lines(tmp_path):
    result, fake = _run_arch(tmp_path)
    run_dir = result["run_dir"]
    argvs = [argv for argv, _env in fake.calls]
    assert argvs[0][1:5] == [
        "-m", "xcquinox.pipeline.cluster", "submit",
        str(Path(tmp_path).resolve() / "deep" / "grid.yaml")]
    assert argvs[0][5:] == ["--partition", "local"]
    assert argvs[1][1:] == ["-m", "xcquinox.pipeline.cluster._datagen", run_dir]
    assert argvs[2][1:] == ["-m", "xcquinox.pipeline.cluster._pretrain", run_dir,
                            "0"]
    assert argvs[3][1:] == ["-m", "xcquinox.pipeline.cluster.fidelity", run_dir,
                            "0"]
    assert argvs[4][1:] == ["-m", "xcquinox.pipeline.cluster._preflight", run_dir]
    assert argvs[5][1:] == ["-m", "xcquinox.pipeline.cluster._train_task", run_dir,
                            "0", "--device", "cpu"]
    assert argvs[6][1:] == ["-m", "xcquinox.pipeline.cluster._train_task", run_dir,
                            "1", "--device", "cpu"]
    assert argvs[7][1:] == ["-m", "xcquinox.pipeline.cluster._eval_one_spec",
                            run_dir, "0"]
    assert argvs[8][1:] == ["-m", "xcquinox.pipeline.cluster._eval_one_spec",
                            run_dir, "1"]
    assert argvs[9][1:] == ["-m", "xcquinox.pipeline.cluster.validate_run",
                            run_dir]


@_needs_cache
def test_run_arch_stops_at_the_first_non_zero_stage(tmp_path):
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    fake = FakeRunner(run_dir, rc_by_stage={"preflight": 1})
    result, fake = _run_arch(tmp_path, fake=fake)
    assert [s["name"] for s in result["stages"]] == [
        "submit", "datagen", "pretrain", "certificate", "preflight"]
    assert result["stages"][-1]["rc"] == 1
    # The oracles still run: they are a property of the installed code, not of
    # this run directory.
    assert result["oracle_tests"]["rc"] == 0


@_needs_cache
def test_run_arch_does_not_stop_on_a_failing_certificate(tmp_path):
    """The certificate's VERDICT is recorded, not required: a 50-step pretrain
    on two atoms cannot meet tol_AE = 1.0 kcal/mol, and the matrix asks the
    matrix to record the verdict while every stage exits zero."""
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    fake = FakeRunner(run_dir, rc_by_stage={"certificate": 1}, verdict="FAIL")
    result, _fake = _run_arch(tmp_path, fake=fake)
    assert [s["name"] for s in result["stages"]] == list(wm.STAGE_ORDER)
    assert result["certificate_verdict"] == "FAIL"
    cert_stage = [s for s in result["stages"] if s["name"] == "certificate"][0]
    assert cert_stage["rc"] == 1


# ---------------------------------------------------------------------------
# One document per certificate record
# ---------------------------------------------------------------------------

def _serve_documents(monkeypatch, path, documents):
    """Serve ``documents`` to successive READ opens of ``path``.

    The list returned collects one entry per read served, so a caller can
    state how many parses a record rested on. Writes and every other path are
    passed through; once the list is exhausted its last entry repeats, so a
    caller that reads more often than the sequence is long is handed a
    complete document rather than an empty file.
    """
    import builtins
    import io
    real_open = builtins.open
    served: list = []

    def fake_open(file, *args, **kwargs):
        mode = kwargs.get("mode", args[0] if args else "r")
        if str(file) == str(path) and "r" in mode:
            doc = documents[min(len(served), len(documents) - 1)]
            served.append(doc)
            return io.StringIO(doc if isinstance(doc, str)
                               else json.dumps(doc))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    return served


# Three FAIL documents, each refused by the gate on its own and each producing
# a DIFFERENT record: D1 records no waiver, D2 records one that states no
# reason, D3 states a reason beside enforcement that is ON. A record that asked
# the file each question separately took the verdict from the first, the gate's
# decision from the second and the waiver from the third.
_MD1 = {"verdict": "FAIL",
        "summary": {"max_atom_mHa": 13.7, "max_dAE_kcalmol": 25.7}}
_MD2 = {"verdict": "FAIL", "enforced": False}
_MD3 = {"verdict": "FAIL", "enforced": True,
        "tolerances": {"override_reason": "workflow matrix"}}


# ---------------------------------------------------------------------------
# Held-out slice: the channel the eval stages leave behind must be MARKED
# ---------------------------------------------------------------------------

#: The six species of :data:`workflow_matrix.HELDOUT_SPECIES_SLICE`, as
#: ``_eval_one_spec._apply_species_slice`` writes them into the marks.
_SLICE_NAMES = ["h", "h2", "o", "oh", "n2o", "n2ohts"]


class SliceMarkingRunner(FakeRunner):
    """A fake runner whose eval stages leave the marks a sliced channel carries.

    ``cluster/_eval_one_spec`` marks a sliced held-out channel twice:
    ``sliced_eval.json`` before any energy is computed, and a ``species_slice``
    entry in ``eval_metadata.json`` after the evaluation. Both are reproduced
    here so the runner's own check of the channel is exercised on the files the
    stage really writes.
    """

    def __init__(self, run_dir, *, n_reactions=3, marker=True, stamp=True,
                 **kwargs):
        super().__init__(run_dir, **kwargs)
        self.n_reactions = n_reactions
        self.marker = marker
        self.stamp = stamp

    def __call__(self, argv, **kwargs):
        completed = super().__call__(argv, **kwargs)
        argv = [str(a) for a in argv]
        if "xcquinox.pipeline.cluster._eval_one_spec" in " ".join(argv):
            channel = (self.run_dir / "checkpoints" / f"spec_{int(argv[-1]):04d}"
                       / "eval_holdout")
            channel.mkdir(parents=True, exist_ok=True)
            if self.marker:
                (channel / "sliced_eval.json").write_text(json.dumps({
                    "species_slice": _SLICE_NAMES,
                    "n_species": len(_SLICE_NAMES),
                    "n_reactions": self.n_reactions,
                    "env_var": "XCQUINOX_HELDOUT_SPECIES_SLICE"}))
            (channel / "eval_metadata.json").write_text(json.dumps({
                "channel": "eval_holdout",
                "species_slice": _SLICE_NAMES if self.stamp else None,
                "n_species": len(_SLICE_NAMES),
                "n_reactions": self.n_reactions}))
        return completed


@_needs_cache
def test_run_arch_refuses_an_eval_channel_that_carries_no_slice_marker(
        tmp_path):
    """A channel without ``sliced_eval.json`` was evaluated on the full
    216-reaction pool, or the slice never reached the stage: either way the
    matrix's held-out assertion did not run, and the figure layer would read
    the channel as a full-pool one."""
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    result, _fake = _run_arch(tmp_path,
                              fake=SliceMarkingRunner(run_dir, marker=False))
    check = result["slice_check"]
    assert check["ok"] is False
    assert "sliced_eval.json" in check["detail"]


# ---------------------------------------------------------------------------
# run_matrix + report
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_refs(tmp_path_factory):
    """One real copy of the 74 MB reference cache for the whole module.

    ``stage_cached_inputs`` copies rather than symlinks (the tracked cache
    receives a run log from every preflight), and every ``run_matrix`` test
    would otherwise pay that copy again.
    """
    if _CACHE_MISSING:
        pytest.skip(_CACHE_MISSING)
    root = tmp_path_factory.mktemp("shared_inputs")
    return wm.stage_cached_inputs(root, repo_root=wm.repo_root_path())[
        "external_refs_dir"]


class MatrixFakeRunner:
    """Fake runner for several architectures: derives each run directory from
    the grid path the submit stage is handed, so one instance serves the whole
    matrix. Thread-safe enough for the shard test: appends only."""

    def __init__(self, *, rc_by_arch=None, verdict="PASS"):
        self.rc_by_arch = dict(rc_by_arch or {})
        self.verdict = verdict
        self.calls = []

    def __call__(self, argv, **kwargs):
        argv = [str(a) for a in argv]
        self.calls.append((argv, dict(kwargs.get("env") or {})))
        stream = kwargs.get("stdout")
        stage = FakeRunner._stage_of(argv)
        arch = self._arch_of(argv)
        # The tag goes FIRST: run_arch reads the LAST non-empty line of the
        # oracle log as its summary line.
        stream.write(f"[fake] {stage} {arch}\n")
        if stage == "submit":
            run_dir = Path(argv[4]).parent / "runs" / f"run_{arch}"
            run_dir.mkdir(parents=True, exist_ok=True)
            stream.write(f"submit: run dir = {run_dir}\n")
        elif stage == "certificate":
            cert = Path(argv[3]) / "pretrain" / arch / \
                "fidelity_certificate.json"
            cert.parent.mkdir(parents=True, exist_ok=True)
            cert.write_text(json.dumps(_certificate_payload(self.verdict)))
        elif stage == "eval":
            # ``cluster/_eval_one_spec`` marks a sliced held-out channel twice
            # -- ``sliced_eval.json`` before any energy is computed, a
            # ``species_slice`` entry in ``eval_metadata.json`` after them --
            # and ``_slice_check`` reads both, so the fake writes both for the
            # slice the stage was actually handed. The reaction closure is
            # known here for the default slice only; any other slice is
            # recorded without one, which the channel check reports as the
            # mismatch it is.
            from xcquinox.pipeline.full_benchmark_pools import (
                HELDOUT_SPECIES_SLICE_ENV,
            )
            default = [part.strip() for part
                       in wm.HELDOUT_SPECIES_SLICE.split(",") if part.strip()]
            names = [part.strip() for part in
                     (kwargs.get("env") or {}).get(
                         HELDOUT_SPECIES_SLICE_ENV, "").split(",")
                     if part.strip()] or default
            channel = (Path(argv[3]) / "checkpoints"
                       / f"spec_{int(argv[4]):04d}" / "eval_holdout")
            channel.mkdir(parents=True, exist_ok=True)
            payload = {
                "species_slice": names, "n_species": len(names),
                "n_reactions": (wm.SLICE_CLOSED_REACTIONS
                                if names == default else None),
                "env_var": HELDOUT_SPECIES_SLICE_ENV}
            (channel / "sliced_eval.json").write_text(json.dumps(payload))
            (channel / "eval_metadata.json").write_text(
                json.dumps(dict(payload, channel="eval_holdout")))
        elif stage == "oracles":
            stream.write("7 passed in 2.0s\n")

        class _Completed:
            returncode = (self.rc_by_arch.get(arch, 0)
                          if stage == "preflight" else 0)

        return _Completed()

    @staticmethod
    def _arch_of(argv):
        for token in argv:
            if token.endswith("grid.yaml"):
                return Path(token).parent.name
            if "/runs/run_" in token:
                return Path(token).name[len("run_"):]
            if token.startswith("test_spin_scaling_oracles and "):
                return token.split(" and ")[1]
        return "?"


def test_write_matrix_report_writes_markdown_and_json(tmp_path):
    results = [
        {"arch": "deep", "seconds": 702.0, "certificate_verdict": "PASS",
         "run_dir": "/w/deep/runs/run_x",
         "stages": [{"name": n, "rc": 0} for n in wm.STAGE_ORDER],
         "artefacts": {"manifest": {"path": "/w/m.json", "exists": True}},
         "oracle_tests": {"rc": 0, "summary_line": "12 passed in 3.4s"}},
        {"arch": "deep_dm", "seconds": 61.0, "certificate_verdict": "FAIL",
         "run_dir": "/w/deep_dm/runs/run_y",
         "stages": [{"name": "submit", "rc": 0},
                    {"name": "datagen", "rc": 2}],
         "artefacts": {"manifest": {"path": "/w/n.json", "exists": False}},
         "oracle_tests": {"rc": 0, "summary_line": "12 passed in 3.1s"}},
    ]
    path = wm.write_matrix_report(results, tmp_path / "matrix.md")
    text = path.read_text()
    assert "| arch | stages rc | certificate | oracles | wall |" in text
    assert "| deep | 0.0.0.0.0.0.0.0.0.0 | PASS |" in text
    assert "| deep_dm | 0.2.-.-.-.-.-.-.-.- | FAIL |" in text
    assert ", ".join(wm.STAGE_ORDER) in text
    assert "1 of 2" in text
    sidecar = json.loads((tmp_path / "matrix.json").read_text())
    assert [r["arch"] for r in sidecar["results"]] == ["deep", "deep_dm"]
    assert sidecar["species_slice"] == wm.HELDOUT_SPECIES_SLICE
    assert sidecar["stage_order"] == list(wm.STAGE_ORDER)


def _clean_result(arch="deep", **overrides):
    """A result record in which every acceptance item of the matrix is met."""
    result = {
        "arch": arch, "seconds": 12.0, "run_dir": f"/w/{arch}/runs/run_x",
        "certificate_verdict": "PASS",
        "stages": [{"name": n, "rc": 0, "seconds": 1.0,
                    "log": f"/w/{arch}/logs/{n}.log"} for n in wm.STAGE_ORDER],
        "artefacts": {"manifest": {"path": "/w/m.json", "exists": True}},
        "slice_check": {"checked": True, "ok": True, "n_reactions": 3,
                        "channels": ["spec_0000/eval_holdout"], "detail": "ok"},
        "oracle_tests": {"rc": 0, "summary_line": "12 passed in 3.4s"},
    }
    result.update(overrides)
    return result


def test_matrix_exit_code_is_zero_only_for_a_matrix_that_met_every_item():
    """Spec 3.4's acceptance list, as one number: every stage exits zero, the
    held-out channel is written AND marked sliced, and the oracles pass. The
    certificate's verdict is exempt -- it is recorded, not required -- so a
    FAIL verdict alone leaves the matrix clean, while a stage that never ran,
    an unmarked channel or a failing oracle does not.
    """
    assert wm.matrix_exit_code([_clean_result()]) == 0
    assert wm.matrix_exit_code(
        [_clean_result(certificate_verdict="FAIL")]) == 0
    assert wm.matrix_exit_code([_clean_result(stages=[
        {"name": "submit", "rc": 0}, {"name": "datagen", "rc": 1}])]) == 1
    assert wm.matrix_exit_code([_clean_result(
        oracle_tests={"rc": 1, "summary_line": "1 failed"})]) == 1
    assert wm.matrix_exit_code([_clean_result(slice_check={
        "checked": True, "ok": False, "n_reactions": None, "channels": [],
        "detail": "no sliced_eval.json"})]) == 1
    assert wm.matrix_exit_code([_clean_result(), _clean_result(
        "shallow", error="RuntimeError: stage launch failed")]) == 1
    # A record from a run that skipped the oracles deliberately is not a
    # failure of the oracles.
    assert wm.matrix_exit_code([_clean_result(
        oracle_tests={"rc": None, "summary_line": ""})]) == 0
    # The certificate: a FAIL under the run's own waiver is the expected
    # outcome of this identity, a certificate that was never written is not.
    assert wm.matrix_exit_code([_clean_result(certificate={
        "present": True, "verdict": "FAIL", "enforced": False,
        "override_reason": "workflow matrix", "gate_released": True})]) == 0
    assert wm.matrix_exit_code([_clean_result(certificate={
        "present": False, "verdict": None, "enforced": None,
        "override_reason": None, "gate_released": False,
        "path": "/w/deep/pretrain/deep/fidelity_certificate.json",
        "gate_message": "no certificate"})]) == 1


# ---------------------------------------------------------------------------
# Oracle selection: case, and oracle function names
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The certificate: a waived FAIL is the expected outcome, an absent one is not
# ---------------------------------------------------------------------------


# --------------------------------------------------------------------------- #
# A stage that finished its work and then aborted at interpreter teardown
# --------------------------------------------------------------------------- #

class AbortAfterWritingRunner(FakeRunner):
    """``_pretrain`` as cluster job 2134455 ran it on node dn024.

    The stage did all of its work -- both pretraining phases, the fidelity
    certificate computed and written, ``xnet.eqx`` and ``cnet.eqx`` on disk,
    ``pretrain SUCCEEDED`` in the log -- and the interpreter then died in
    glibc's ``corrupted size vs. prev_size while consolidating`` during
    teardown, so the process was reported as killed by SIGABRT (rc -6). The
    certificate is written HERE, at the pretrain stage, because that is the
    stage that computes it; the sequence never reaches the certificate stage.
    """

    def __call__(self, argv, **kwargs):
        stage = self._stage_of(argv)
        completed = super().__call__(argv, **kwargs)
        if stage == "pretrain":
            cert = (self.run_dir / "pretrain" / "deep"
                    / "fidelity_certificate.json")
            cert.parent.mkdir(parents=True, exist_ok=True)
            cert.write_text(json.dumps(_certificate_payload(self.verdict)))
            kwargs["stdout"].write(
                "[harness pretrain arch=deep] pretrain SUCCEEDED\n")
        return completed


# ---------------------------------------------------------------------------
# validate_run: a record layer that MUST refuse this identity's run
# ---------------------------------------------------------------------------


class ValidateRunRunner(SliceMarkingRunner):
    """A fake runner whose validate_run stage writes what that module writes.

    ``cluster/validate_run.main`` prints a checked-count line, one
    ``[validate_run] FAIL: <text>`` per failure and a count line, then exits 1;
    with no failure it prints its clean line and exits 0. The matrix reads
    those lines, so the fake produces them rather than a bare exit code.
    """

    def __init__(self, run_dir, *, failures=(), validate_rc=1, **kwargs):
        super().__init__(run_dir, **kwargs)
        self.failures = tuple(failures)
        self.validate_rc = validate_rc

    def __call__(self, argv, **kwargs):
        argv_s = [str(a) for a in argv]
        if FakeRunner._stage_of(argv_s) != "validate_run":
            return super().__call__(argv, **kwargs)
        self.calls.append((argv_s, dict(kwargs.get("env") or {})))
        stream = kwargs.get("stdout")
        stream.write("[fake] validate_run\n")
        stream.write(f"[validate_run] checked 2 spec(s) under {self.run_dir}\n")
        for failure in self.failures:
            stream.write(f"[validate_run] FAIL: {failure}\n")
        if self.failures:
            stream.write(f"[validate_run] {len(self.failures)} failure(s), "
                         "0 warning(s)\n")
        else:
            stream.write("[validate_run] clean (0 warning(s))\n")

        class _Completed:
            returncode = self.validate_rc

        return _Completed()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_main_runs_the_named_architectures_and_writes_the_report(
        tmp_path, shared_refs, capsys):
    rc = wm.main(["--archs", "deep,shallow", "--work-root", str(tmp_path),
                  "--external-refs-dir", str(shared_refs),
                  "--report", str(tmp_path / "matrix.md")],
                 runner=MatrixFakeRunner())
    assert rc == 0
    text = (tmp_path / "matrix.md").read_text()
    assert "| deep |" in text and "| shallow |" in text
    assert (tmp_path / "matrix.json").is_file()
    out = capsys.readouterr().out
    assert "deep" in out and "shallow" in out


# ---------------------------------------------------------------------------
# The expected pretrain-data artefact follows the run's parent density
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_oracle_selector_selects_this_architecture_only(arch,
                                                        collected_oracle_ids):
    """Contract with the spec-3.1 oracle module, for EVERY registered
    architecture: the selector resolves to a non-empty set of collected tests,
    it selects all of this architecture's cases, and it selects nothing else.

    Checking one architecture would leave the other 30 selectors unmeasured,
    and the failures this catches are per-name: a registry entry that is a
    substring of another, an architecture the module does not parametrise
    over, an oracle function named after an architecture.
    """
    expr = _compile_k(wm.oracle_selector(arch))
    mine, selected = [], []
    for node_id in collected_oracle_ids:
        if arch in _arch_params(node_id):
            mine.append(node_id)
        if expr.evaluate(_k_matcher(_names_of(node_id))):
            selected.append(node_id)
    assert mine, f"the oracle module collects no case for {arch!r}"
    assert selected == mine


def test_the_oracle_module_names_no_test_function_after_an_architecture(
        collected_oracle_ids):
    """``-k`` matches the function name as well as the parametrisation id, so
    an oracle function carrying an architecture name would be reported as that
    architecture's oracle whatever its parameters say. No selector can exclude
    it; the collected names are what has to stay clean."""
    conflicts = wm.oracle_function_name_conflicts(collected_oracle_ids)
    assert conflicts == [], "\n".join(conflicts)
