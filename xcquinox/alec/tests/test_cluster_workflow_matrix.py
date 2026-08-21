"""Tests for the per-architecture workflow matrix
(``xcquinox.alec.cluster.workflow_matrix``).

The matrix drives the harness stage sequence at a tiny def2-svp /
grid-level-1 identity for every registered architecture
(SPEC_pretrain_fidelity_program.md 3.4). What the module carries at this point
is that identity: the checked-in template, the renderer that writes one
architecture's grid config, and the staging of the cached inputs the identity
consumes. These tests cover those three; they start no stage and run no SCF,
so the whole file runs in seconds.
"""
from __future__ import annotations

import json
import keyword
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from xcquinox.alec.cluster import workflow_matrix as wm
from xcquinox.alec.config import ARCHITECTURES


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


def test_template_names_only_environment_variables_that_exist():
    """A comment naming an ``XCQUINOX_*`` knob the package never reads sends a
    reader looking for a bypass that does not exist (the certificate's
    enforcement bypass is the YAML ``fidelity`` block, not an environment
    variable)."""
    text = wm.template_path().read_text()
    names = sorted(set(re.findall(r"XCQUINOX_[A-Z0-9_]+", text)))
    sources = [p.read_text(errors="ignore")
               for p in (wm.repo_root_path() / "xcquinox").rglob("*.py")]
    for name in names:
        assert any(name in src for src in sources), (
            f"the template names {name}, which appears nowhere in the package")


def test_template_is_the_tiny_identity_the_spec_fixes():
    import yaml
    raw = yaml.safe_load(wm.template_path().read_text())
    assert raw["inputs"]["basis"] == "def2-svp"
    assert raw["inputs"]["grid_level"] == 1
    assert raw["sweep"]["solver"] == ["oneshot"]
    assert raw["sweep"]["subset_size"] == [1, 2]
    assert raw["hyperparams"]["n_steps"] == 3
    assert raw["hyperparams"]["validate_every"] == 0
    assert raw["hyperparams"]["checkpoint_every"] == 0
    assert raw["pretrain"]["n_steps"] == 50
    assert raw["pretrain"]["atoms"] == {"H": 1, "O": 2}
    assert raw["cluster"]["eval_workers"] == 1
    assert raw["cluster"]["device"] == "cpu"
    assert "benchmark_refs_dir" not in raw["inputs"]
    assert "val_refs_dir" not in raw["inputs"]
    # The certificate runs at the production identity and its verdict is the
    # real one; what the template waives is the ENFORCEMENT, without which the
    # on-node gates (cluster/_preflight.py, cluster/_train_task.py, both through
    # fidelity.gate_certificate) would block every architecture at a FAIL that
    # 50 pretraining steps on two atoms cannot avoid. The tolerances are NOT
    # written here: they stay at the program's binding defaults.
    assert raw["fidelity"]["enforce"] is False
    assert raw["fidelity"]["override_reason"] == (
        "workflow matrix: wiring check at a 50-step pretrain, never a campaign")
    assert "tol_AE" not in raw["fidelity"]
    assert "tol_atom" not in raw["fidelity"]


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


def test_stage_cached_inputs_is_idempotent(tmp_path):
    first = wm.stage_cached_inputs(tmp_path, repo_root=wm.repo_root_path())
    probe = Path(first["external_refs_dir"]) / "_matrix_probe.txt"
    probe.write_text("kept")
    second = wm.stage_cached_inputs(tmp_path, repo_root=wm.repo_root_path())
    assert second == first
    assert probe.read_text() == "kept"


def _staged_relpaths(root: Path) -> set:
    """Every staged file below ``root``, relative, excluding the manifest."""
    return {str(p.relative_to(root)) for p in root.rglob("*")
            if p.is_file() and p.name != wm.STAGE_MARKER}


def _cached_relpaths() -> set:
    """The reference cache as it stands in the tree, run logs excluded."""
    src = wm.repo_root_path() / wm.CACHED_REFS_RELPATH
    return {str(p.relative_to(src)) for p in src.rglob("*")
            if p.is_file() and not p.name.startswith("_run_log_")}


def test_stage_cached_inputs_repairs_a_partial_stage(tmp_path):
    """A destination that is present but incomplete -- an interrupted copy, or
    files removed under it -- must be re-staged, not reported as done. The
    manifest lists every file the copy carries, so a missing entry forces the
    copy again (165 files at the measured cache size)."""
    expected = _cached_relpaths()
    first = wm.stage_cached_inputs(tmp_path, repo_root=wm.repo_root_path())
    refs = Path(first["external_refs_dir"])
    assert _staged_relpaths(refs) == expected

    victims = sorted(refs / rel for rel in expected)[:len(expected) // 2]
    for victim in victims:
        victim.unlink()
    assert _staged_relpaths(refs) != expected, "the deletion must bite"

    second = wm.stage_cached_inputs(tmp_path, repo_root=wm.repo_root_path())
    assert second == first
    assert _staged_relpaths(refs) == expected, (
        "a half-populated destination was accepted as a complete stage")
    assert (refs / "H2O.npz").is_file()
    assert (refs / wm.STAGE_MARKER).is_file()
    assert not list(refs.parent.glob("external_refs.partial-*"))


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


def test_stage_cached_inputs_names_the_missing_cache(tmp_path):
    """``notebooks/checkpoints_step7/`` is untracked (.gitignore), so a fresh
    clone, a worktree or the cluster repository has neither the references nor
    the ledger. The failure must name the directory to stage, not surface as a
    copytree traceback."""
    with pytest.raises(wm.CachedInputsMissing,
                       match="notebooks/checkpoints_step7"):
        wm.stage_cached_inputs(tmp_path / "work",
                               repo_root=tmp_path / "empty_repo")


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


def test_write_matrix_yaml_checks_the_ledger_on_the_shared_refs_path(tmp_path):
    """With ``external_refs_dir`` supplied the staging branch is skipped, so
    the ledger check has to stand on its own: otherwise a wrong ``repo_root``
    renders a config whose ``subset_ledger_path`` does not exist, and
    ``validate_grid_semantics`` (which does not stat that path) accepts it.
    The references are supplied intact so the ledger is the only defect."""
    with pytest.raises(wm.CachedInputsMissing, match="subset_index_log.json"):
        wm.write_matrix_yaml("deep", tmp_path / "deep",
                             repo_root=tmp_path / "empty_repo",
                             external_refs_dir=_fake_staged_refs(
                                 tmp_path / "refs"))


def test_write_matrix_yaml_refuses_a_supplied_refs_dir_that_is_absent(tmp_path):
    """A supplied ``external_refs_dir`` is an INPUT -- the shared copy an
    earlier ``stage_cached_inputs`` wrote -- so a mistyped path has to be
    refused rather than created. ``validate_grid_semantics`` never stats
    ``inputs.external_refs_dir`` (only ``pretrain.data_dir`` and the parent of
    ``inputs.output_root`` are checked, both advisory), so an empty directory
    created here passes the login-node gate and moves the failure to a compute
    node, at the first reference read."""
    absent = tmp_path / "mistyped_refs"
    with pytest.raises(wm.CachedInputsMissing,
                       match=re.escape(os.path.realpath(absent))):
        wm.write_matrix_yaml("deep", tmp_path / "deep",
                             repo_root=wm.repo_root_path(),
                             external_refs_dir=absent)
    assert not absent.exists(), (
        "a supplied external_refs_dir is an input; the renderer must not "
        "create it")


def test_write_matrix_yaml_refuses_an_empty_supplied_refs_dir(tmp_path):
    """Existence is not the criterion. A directory holding neither the staging
    manifest nor a species ``.npz`` carries no references, so accepting it
    would defer the same failure to a compute node."""
    empty = tmp_path / "empty_refs"
    empty.mkdir()
    with pytest.raises(wm.CachedInputsMissing,
                       match=re.escape(os.path.realpath(empty))):
        wm.write_matrix_yaml("deep", tmp_path / "deep",
                             repo_root=wm.repo_root_path(),
                             external_refs_dir=empty)


def test_write_matrix_yaml_accepts_a_staged_supplied_refs_dir(tmp_path):
    """The acceptance criterion is what the staging actually writes: the
    output of ``stage_cached_inputs`` is accepted unchanged, which is the
    shared-directory case the matrix runs (one copy per shard)."""
    from xcquinox.alec.cluster.grid_config import load_grid_config
    staged = Path(wm.stage_cached_inputs(
        tmp_path / "shard",
        repo_root=wm.repo_root_path())["external_refs_dir"])
    cfg = load_grid_config(str(wm.write_matrix_yaml(
        "deep", tmp_path / "deep", repo_root=wm.repo_root_path(),
        external_refs_dir=staged)))
    assert cfg.inputs.external_refs_dir == str(staged.resolve())


def test_rendered_walltimes_are_quoted_and_reload_as_strings(tmp_path,
                                                             monkeypatch):
    """A walltime must never reach ``#SBATCH --time`` as an integer.

    YAML 1.1 resolves an unquoted clock token as a sexagesimal number:
    ``8:00:00`` loads as 28800, ``submit.render_sbatch`` substitutes it into
    ``#SBATCH --time=${TIME}``, and SLURM reads a bare integer as MINUTES
    (28800 minutes = 20 days). The rendered config therefore has to carry the
    clock string quoted, whatever the template's own quoting was.
    """
    import yaml
    from xcquinox.alec.cluster.grid_config import load_grid_config
    mutant = tmp_path / "unquoted_template.yaml"
    mutant.write_text(wm.template_path().read_text().replace(
        '"00:30:00"', "8:00:00"))
    assert yaml.safe_load(mutant.read_text())["cluster"]["time"] == 28800, (
        "the unquoted token must be the sexagesimal case under test")
    monkeypatch.setattr(wm, "template_path", lambda: mutant)

    path = wm.write_matrix_yaml("deep", tmp_path / "deep",
                                repo_root=wm.repo_root_path(),
                                external_refs_dir=_fake_staged_refs(
                                    tmp_path / "refs"))
    assert "time: '8:00:00'" in path.read_text()
    cfg = load_grid_config(str(path))
    for field in ("time", "preflight_time", "eval_time", "pretrain_time"):
        value = getattr(cfg.cluster, field)
        assert isinstance(value, str), (field, value, type(value).__name__)
        assert value == "8:00:00", (field, value)


def _template_with_walltime(tmp_path, monkeypatch, token) -> Path:
    """The shipped template with every ``"00:30:00"`` walltime replaced."""
    mutant = tmp_path / "walltime_template.yaml"
    mutant.write_text(
        wm.template_path().read_text().replace('"00:30:00"', token))
    monkeypatch.setattr(wm, "template_path", lambda: mutant)
    return mutant


@pytest.mark.parametrize("token,loads_as", [
    ('"30"', "30"),
    ("30", 30),
    ('"30:00"', "30:00"),
    ("30:00", 1800),
    ('"8:00"', "8:00"),
])
def test_walltime_that_is_not_a_clock_is_refused(tmp_path, monkeypatch,
                                                 token, loads_as):
    """Quoting is not the criterion, the SHAPE is.

    SLURM reads a bare integer as MINUTES and ``MM:SS`` as minutes and seconds,
    while every walltime field of this harness is documented as HH:MM:SS. A
    quoted ``"30"`` loads as a string and would reach ``#SBATCH --time=30`` --
    half an hour where thirty hours were meant -- exactly as the unquoted 30
    does, so the shape is checked on the loaded string as well as on the source
    token.
    """
    import yaml
    mutant = _template_with_walltime(tmp_path, monkeypatch, token)
    assert yaml.safe_load(mutant.read_text())["cluster"]["time"] == loads_as, (
        "the mutant must be the case under test")
    with pytest.raises(ValueError, match="HH:MM:SS"):
        wm.write_matrix_yaml("deep", tmp_path / "deep",
                             repo_root=wm.repo_root_path(),
                             external_refs_dir=_fake_staged_refs(
                                 tmp_path / "refs"))


@pytest.mark.parametrize("token,value", [
    ('"8:00:00"', "8:00:00"),
    ("8:00:00", "8:00:00"),
    ('"48:00:00"', "48:00:00"),
    ('"1-00:00:00"', "1-00:00:00"),
])
def test_walltime_accepts_the_slurm_clock_forms(tmp_path, monkeypatch,
                                                token, value):
    """``H:MM:SS`` and ``D-HH:MM:SS`` are the two forms SLURM reads as a wall
    clock rather than as minutes; both survive the round trip."""
    from xcquinox.alec.cluster.grid_config import load_grid_config
    _template_with_walltime(tmp_path, monkeypatch, token)
    cfg = load_grid_config(str(wm.write_matrix_yaml(
        "deep", tmp_path / "deep", repo_root=wm.repo_root_path(),
        external_refs_dir=_fake_staged_refs(tmp_path / "refs"))))
    for field in ("time", "preflight_time", "eval_time", "pretrain_time"):
        assert getattr(cfg.cluster, field) == value, field


def test_write_matrix_yaml_renders_one_arch_and_two_cells(tmp_path):
    from xcquinox.alec.cluster.grid_config import expand_grid, load_grid_config
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


def test_write_matrix_yaml_paths_are_absolute_and_outside_the_repository(
        tmp_path):
    from xcquinox.alec.cluster.grid_config import load_grid_config
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


def test_write_matrix_yaml_honours_shared_directories(tmp_path):
    from xcquinox.alec.cluster.grid_config import load_grid_config
    shared_refs = _fake_staged_refs(tmp_path / "shared_refs")
    shared_data = tmp_path / "shared_pretrain_data"
    cfg = load_grid_config(str(wm.write_matrix_yaml(
        "deep", tmp_path / "deep", repo_root=wm.repo_root_path(),
        external_refs_dir=shared_refs, pretrain_data_dir=shared_data)))
    assert cfg.inputs.external_refs_dir == str(shared_refs)
    assert cfg.pretrain.data_dir == str(shared_data)
    # pretrain.data_dir is an OUTPUT -- datagen writes the pretraining set into
    # it, and validate_grid_semantics warns when it is missing -- so it is
    # created here. external_refs_dir is an INPUT and is only checked.
    assert shared_data.is_dir(), "pretrain.data_dir must exist before datagen"


def test_write_matrix_yaml_refuses_an_unregistered_architecture(tmp_path):
    with pytest.raises(ValueError, match="not a registered architecture"):
        wm.write_matrix_yaml("no_such_arch", tmp_path / "x",
                             repo_root=wm.repo_root_path())


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_every_registered_architecture_renders_a_valid_grid(arch, tmp_path):
    """All 30-odd registry entries, not the 25 the figure layer renders."""
    from xcquinox.alec.cluster.domain import get_domain_profile
    from xcquinox.alec.cluster.grid_config import (load_grid_config,
                                                   validate_grid_semantics)
    cfg = load_grid_config(str(wm.write_matrix_yaml(
        arch, tmp_path / arch, repo_root=wm.repo_root_path(),
        external_refs_dir=_fake_staged_refs(tmp_path / "refs"))))
    validate_grid_semantics(cfg, get_domain_profile(cfg.domain_profile))
    # The rendered config carries the waiver through to every architecture:
    # validate_grid_semantics refuses enforce: false without a non-empty
    # override_reason, and the tolerances stay at the binding defaults.
    assert cfg.fidelity.enforce is False
    assert cfg.fidelity.override_reason.strip()
    assert (cfg.fidelity.tol_AE, cfg.fidelity.tol_atom) == (1.0, 1.0)


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
    ``<repo>/xcquinox/alec/tests/<module>`` that is the three directory
    components below the root, the module file name, and the test name
    carrying its parametrisation id.
    """
    module = f"{wm.ORACLE_MODULE}.py" if module is None else module
    return {"xcquinox", "alec", "tests", module, node_id}


def _k_matcher(names):
    """pytest's ``-k`` matching rule: a term matches when it is a
    case-insensitive substring of any of the item's names
    (``KeywordMatcher.__call__``)."""
    def matcher(ident, /, **kwargs):
        assert not kwargs, f"unexpected call parameters on {ident!r}"
        return any(ident.lower() in name.lower() for name in names)
    return matcher


def _oracle_node_ids(arch: str) -> tuple:
    """Node ids one architecture contributes to the spec-3.1 oracle module.

    Three shapes: the architecture-only parametrisation, and a species x
    architecture one in both id orders. Which order a stacked ``parametrize``
    produces is decided by the order the decorators are applied -- the one
    closest to the function supplies the leading id component -- so the
    selector has to hold for ``[Li-<arch>]`` and ``[<arch>-Li]`` alike.
    """
    return (f"test_o1_uniform_scaling[{arch}]",
            f"test_o3_spin_scaling_open_shell[Li-{arch}]",
            f"test_o4_spin_scaling_relation[{arch}-Li]")


def test_oracle_selector_names_the_module_and_the_architecture():
    got = wm.oracle_selector("deep_rung35_mgga_3x16")
    assert got.startswith("test_spin_scaling_oracles and ")
    assert " and deep_rung35_mgga_3x16" in got


def test_oracle_selector_excludes_names_that_contain_this_one():
    """pytest -k matches SUBSTRINGS of the node id, so a bare 'deep_cusp'
    selects deep_cusp_3x16 and deep_cusp_mgga_3x16 as well. Every longer
    registry name containing this one is excluded explicitly."""
    got = wm.oracle_selector("deep_cusp",
                             archs=["deep_cusp", "deep_cusp_3x16",
                                    "deep_cusp_mgga_3x16", "deep_dm"])
    assert got == ("test_spin_scaling_oracles and deep_cusp "
                   "and not deep_cusp_3x16 and not deep_cusp_mgga_3x16")


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


def test_oracle_selector_adds_no_exclusion_when_the_name_is_unique():
    got = wm.oracle_selector("deep_dm_3x16",
                             archs=["deep_dm", "deep_dm_3x16"])
    assert got == "test_spin_scaling_oracles and deep_dm_3x16"


def test_oracle_selector_refuses_an_unregistered_architecture():
    with pytest.raises(ValueError, match="not a registered architecture"):
        wm.oracle_selector("no_such_arch")


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_oracle_selector_is_a_valid_k_expression(arch):
    """Every term must be a single pytest expression identifier. ``-k`` is
    lexed on whitespace with ``and``/``or``/``not`` reserved as operators, so a
    term carrying a space, a comma, an ``=``, a quote or a parenthesis is read
    as several tokens and the expression fails to parse; a bare Python
    identifier that is not a keyword carries none of those characters, which
    makes it the conservative form of the condition. Compiling with pytest's
    parser is what settles it.
    """
    selector = wm.oracle_selector(arch)
    for token in selector.split():
        if token in ("and", "not"):
            continue
        assert token.isidentifier() and not keyword.iskeyword(token), token
    _compile_k(selector)


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_oracle_selector_matches_this_architecture_and_no_other(arch):
    """The registry holds 31 names and several are substrings of others
    (``deep`` of ``deep_attn``, ``deep_cusp`` of ``deep_cusp_mgga_3x16``,
    ``medium`` of ``medium_attn``, ``shallow`` of ``shallow_attn``), so a
    bare name would report a sibling architecture's oracles as this one's.
    Each architecture's expression is evaluated against every
    architecture's node ids under pytest's own matching rule: it must accept
    its own three and reject the other 90.
    """
    expr = _compile_k(wm.oracle_selector(arch))
    for other in sorted(ARCHITECTURES):
        for node_id in _oracle_node_ids(other):
            got = expr.evaluate(_k_matcher(_item_names(node_id)))
            assert got == (other == arch), (
                f"selector for {arch!r} {'accepted' if got else 'rejected'} "
                f"{node_id!r}")


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_oracle_selector_rejects_the_same_architecture_elsewhere(arch):
    """ORACLE_TEST_TARGET is the tests DIRECTORY, so collection offers every
    module in it. This file parametrises over the registry as well, so without
    the module term its cases would be selected as oracles.
    """
    expr = _compile_k(wm.oracle_selector(arch))
    node_id = f"test_every_registered_architecture_renders_a_valid_grid[{arch}]"
    names = _item_names(node_id, module="test_cluster_workflow_matrix.py")
    assert not expr.evaluate(_k_matcher(names)), node_id


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
    module = (wm.repo_root_path() / "xcquinox" / "alec" / "tests"
              / f"{wm.ORACLE_MODULE}.py")
    if not module.is_file():
        pytest.skip(f"{module} not installed yet (spec 3.1)")
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


def test_the_longest_selector_survives_the_pytest_command_line(
        tmp_path, collected_oracle_ids):
    """The selector has to survive the CLI, not only pytest's expression
    parser. The architecture whose expression carries the most exclusions is
    the longest one the matrix ever passes to ``-k``, so that is the one run
    end to end."""
    arch = max(sorted(ARCHITECTURES), key=lambda a: len(wm.oracle_selector(a)))
    rc, node_ids, text = _collect_oracles(
        tmp_path / "collect.log", wm.oracle_selector(arch))
    assert rc == 0, text
    assert node_ids, text
    for node_id in node_ids:
        assert arch in _arch_params(node_id), node_id


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
                ("xcquinox.alec.cluster._datagen", "datagen"),
                ("xcquinox.alec.cluster._pretrain", "pretrain"),
                ("xcquinox.alec.cluster.fidelity", "certificate"),
                ("xcquinox.alec.cluster._preflight", "preflight"),
                ("xcquinox.alec.cluster._train_task", "train"),
                ("xcquinox.alec.cluster._eval_one_spec", "eval"),
                ("xcquinox.alec.cluster.validate_run", "validate_run"),
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


def test_run_arch_runs_the_ten_stages_in_order(tmp_path):
    result, fake = _run_arch(tmp_path)
    assert [s["name"] for s in result["stages"]] == list(wm.STAGE_ORDER)
    assert [s["rc"] for s in result["stages"]] == [0] * len(wm.STAGE_ORDER)
    assert result["arch"] == "deep"
    assert result["run_dir"].endswith("run_20260821T000000Z")
    assert result["seconds"] >= 0.0


def test_run_arch_issues_the_exact_stage_command_lines(tmp_path):
    result, fake = _run_arch(tmp_path)
    run_dir = result["run_dir"]
    argvs = [argv for argv, _env in fake.calls]
    assert argvs[0][1:5] == [
        "-m", "xcquinox.alec.cluster", "submit",
        str(Path(tmp_path).resolve() / "deep" / "grid.yaml")]
    assert argvs[0][5:] == ["--partition", "local"]
    assert argvs[1][1:] == ["-m", "xcquinox.alec.cluster._datagen", run_dir]
    assert argvs[2][1:] == ["-m", "xcquinox.alec.cluster._pretrain", run_dir,
                            "0"]
    assert argvs[3][1:] == ["-m", "xcquinox.alec.cluster.fidelity", run_dir,
                            "0"]
    assert argvs[4][1:] == ["-m", "xcquinox.alec.cluster._preflight", run_dir]
    assert argvs[5][1:] == ["-m", "xcquinox.alec.cluster._train_task", run_dir,
                            "0", "--device", "cpu"]
    assert argvs[6][1:] == ["-m", "xcquinox.alec.cluster._train_task", run_dir,
                            "1", "--device", "cpu"]
    assert argvs[7][1:] == ["-m", "xcquinox.alec.cluster._eval_one_spec",
                            run_dir, "0"]
    assert argvs[8][1:] == ["-m", "xcquinox.alec.cluster._eval_one_spec",
                            run_dir, "1"]
    assert argvs[9][1:] == ["-m", "xcquinox.alec.cluster.validate_run",
                            run_dir]


def test_run_arch_passes_the_species_slice_to_the_eval_stages_only(tmp_path):
    from xcquinox.alec.full_benchmark_pools import HELDOUT_SPECIES_SLICE_ENV
    _result, fake = _run_arch(tmp_path)
    by_stage = {FakeRunner._stage_of(argv): env for argv, env in fake.calls}
    assert by_stage["eval"][HELDOUT_SPECIES_SLICE_ENV] == \
        "h,h2,o,oh,n2o,n2ohts"
    for stage in ("datagen", "pretrain", "preflight", "train",
                  "validate_run"):
        assert HELDOUT_SPECIES_SLICE_ENV not in by_stage[stage]


def test_run_arch_pins_cpu_and_float64_for_every_stage(tmp_path):
    _result, fake = _run_arch(tmp_path)
    for _argv, env in fake.calls:
        assert env["JAX_PLATFORMS"] == "cpu"
        assert env["JAX_ENABLE_X64"] == "1"
        assert env["OMP_NUM_THREADS"] == "4"


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


def test_run_arch_does_not_stop_on_a_failing_certificate(tmp_path):
    """The certificate's VERDICT is recorded, not required: a 50-step pretrain
    on two atoms cannot meet tol_AE = 1.0 kcal/mol, and spec 3.4 asks the
    matrix to record the verdict while every stage exits zero."""
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    fake = FakeRunner(run_dir, rc_by_stage={"certificate": 1}, verdict="FAIL")
    result, _fake = _run_arch(tmp_path, fake=fake)
    assert [s["name"] for s in result["stages"]] == list(wm.STAGE_ORDER)
    assert result["certificate_verdict"] == "FAIL"
    cert_stage = [s for s in result["stages"] if s["name"] == "certificate"][0]
    assert cert_stage["rc"] == 1


def test_run_arch_records_no_verdict_when_the_certificate_is_absent(tmp_path):
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    fake = FakeRunner(run_dir, verdict=None)
    result, _fake = _run_arch(tmp_path, fake=fake)
    assert result["certificate_verdict"] is None


def test_run_arch_writes_one_log_per_stage(tmp_path):
    result, _fake = _run_arch(tmp_path)
    for stage in result["stages"]:
        path = Path(stage["log"])
        assert path.is_file(), path
        assert path.parent == Path(tmp_path).resolve() / "deep" / "logs"
        assert "[fake]" in path.read_text()
    assert Path(result["oracle_tests"]["log"]).is_file()


def test_run_arch_reports_missing_artefacts(tmp_path):
    result, _fake = _run_arch(tmp_path)
    art = result["artefacts"]
    assert art["manifest"]["exists"] is False
    assert art["eval_df[0]"]["path"].endswith(
        "checkpoints/spec_0000/eval_df.csv")
    assert art["holdout_sliced[1]"]["path"].endswith(
        "checkpoints/spec_0001/eval_holdout/sliced_eval.json")


def test_run_arch_marks_the_artefacts_the_stages_produced(tmp_path):
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    produced = (
        "resolved_config.yaml",
        "manifest.json",
        "scripts/datagen.sbatch", "scripts/pretrain.sbatch",
        "scripts/preflight.sbatch", "scripts/train_array.sbatch",
        "scripts/eval_array.sbatch",
        "pretrain/deep/xnet.eqx", "pretrain/deep/cnet.eqx",
        "pretrain/deep/pretrain_metadata.json",
        "specs/spec_0000.spec", "specs/spec_0001.spec",
        "checkpoints/spec_0000/model.eqx", "checkpoints/spec_0001/model.eqx",
        "checkpoints/spec_0000/eval_df.csv",
        "checkpoints/spec_0001/eval_df.csv",
        "checkpoints/spec_0000/eval_holdout/test_set.csv",
        "checkpoints/spec_0001/eval_holdout/test_set.csv",
        "checkpoints/spec_0000/eval_holdout/eval_metadata.json",
        "checkpoints/spec_0001/eval_holdout/eval_metadata.json",
        "checkpoints/spec_0000/eval_holdout/sliced_eval.json",
        "checkpoints/spec_0001/eval_holdout/sliced_eval.json",
    )
    fake = FakeRunner(run_dir, artefacts=produced)
    result, _fake = _run_arch(tmp_path, fake=fake)
    missing = [k for k, v in result["artefacts"].items() if not v["exists"]]
    assert missing == ["pretrain_data"], missing


def test_run_arch_records_the_oracle_summary_line(tmp_path):
    result, _fake = _run_arch(tmp_path)
    oracles = result["oracle_tests"]
    assert oracles["rc"] == 0
    assert oracles["summary_line"] == "12 passed in 3.4s"
    assert oracles["selector"] == wm.oracle_selector("deep")


def test_run_arch_can_skip_the_oracles(tmp_path):
    result, fake = _run_arch(tmp_path, run_oracles=False)
    assert result["oracle_tests"]["rc"] is None
    assert not any(" -m pytest " in " " + " ".join(argv) + " "
                   for argv, _env in fake.calls)


def test_run_arch_records_a_timeout_as_rc_124(tmp_path):
    import subprocess as sp
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"

    class _Timeout(FakeRunner):
        def __call__(self, argv, **kwargs):
            if "_preflight" in " ".join(argv):
                self.calls.append((list(argv), dict(kwargs.get("env") or {})))
                raise sp.TimeoutExpired(cmd=argv, timeout=1)
            return super().__call__(argv, **kwargs)

    result, _fake = _run_arch(tmp_path, fake=_Timeout(run_dir))
    assert result["stages"][-1]["name"] == "preflight"
    assert result["stages"][-1]["rc"] == wm.TIMEOUT_RC
    assert "exceeded" in Path(result["stages"][-1]["log"]).read_text()


def test_run_arch_reports_a_submit_that_printed_no_run_dir(tmp_path):
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"

    class _Silent(FakeRunner):
        def __call__(self, argv, **kwargs):
            if FakeRunner._stage_of(argv) == "submit":
                self.calls.append((list(argv), dict(kwargs.get("env") or {})))
                kwargs["stdout"].write("submit: DRY-RUN\n")

                class _C:
                    returncode = 0
                return _C()
            return super().__call__(argv, **kwargs)

    result, _fake = _run_arch(tmp_path, fake=_Silent(run_dir))
    assert result["run_dir"] is None
    assert [s["name"] for s in result["stages"]] == ["submit"]
    assert result["stages"][0]["rc"] != 0


def test_run_arch_asks_submit_for_a_dry_run_and_never_for_a_submission(
        tmp_path):
    """Nothing the matrix runs may reach SLURM.

    ``submit`` is dry-run by DEFAULT and ``--submit`` is the opt-in that calls
    ``sbatch`` (``cluster/__main__.cmd_submit``: ``submit_jobs(..., submit=
    args.submit)``), so the property to hold is that the argv the runner
    receives carries no submitting flag. The CLI's own parser settles what that
    argv means -- reading the flag list by eye would restate the parser's
    defaults instead of measuring them.
    """
    from xcquinox.alec.cluster.__main__ import _build_parser
    result, fake = _run_arch(tmp_path)
    argvs = [argv for argv, _env in fake.calls]
    # `python -m xcquinox.alec.cluster <argv...>`: the CLI sees argv[3:].
    parsed = _build_parser().parse_args(argvs[0][3:])
    assert parsed.submit is False, "the matrix must never submit to SLURM"
    assert parsed.grid == str(Path(tmp_path).resolve() / "deep" / "grid.yaml")
    for argv in argvs:
        assert "--submit" not in argv, argv
        assert not any("sbatch" in token for token in argv), argv


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
        if "xcquinox.alec.cluster._eval_one_spec" in " ".join(argv):
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


def test_the_slice_constant_closes_the_reactions_it_claims():
    """``HELDOUT_SPECIES_SLICE`` and ``SLICE_CLOSED_REACTIONS`` are measured
    against the real pool, not restated: the six species close exactly three
    reactions -- one BH76 barrier and two W4-11 atomizations -- and a species
    dropped from the slice or a reaction renamed in the pool changes that
    number, which is what the runner's channel check compares against."""
    from xcquinox.alec.full_benchmark_pools import (
        load_full_held_out_pools, slice_held_out_pools)
    mols, rxns = load_full_held_out_pools(basis="def2-svp", grid_level=1)
    names = wm.HELDOUT_SPECIES_SLICE.split(",")
    assert len(names) == 6
    kept_mols, kept_rxns = slice_held_out_pools(mols, rxns, names)
    assert len(kept_mols) == 6
    assert sorted(r["name"] for r in kept_rxns) == [
        "bh76_h_n2o_to_n2ohts", "w411_h2_atomization", "w411_oh_atomization"]
    assert len(kept_rxns) == wm.SLICE_CLOSED_REACTIONS


def test_run_arch_accepts_a_channel_marked_with_the_full_slice(tmp_path):
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    result, _fake = _run_arch(tmp_path, fake=SliceMarkingRunner(run_dir))
    check = result["slice_check"]
    assert check["ok"] is True, check
    assert check["checked"] is True
    assert sorted(check["channels"]) == [
        "spec_0000/eval_holdout", "spec_0001/eval_holdout"]


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


def test_run_arch_refuses_a_channel_whose_stamp_carries_no_slice(tmp_path):
    """``species_slice: null`` is what a FULL-pool evaluation stamps, so a
    stamp without the slice contradicts the marker written before it."""
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    result, _fake = _run_arch(tmp_path,
                              fake=SliceMarkingRunner(run_dir, stamp=False))
    check = result["slice_check"]
    assert check["ok"] is False
    assert "species_slice" in check["detail"]


def test_run_arch_refuses_a_slice_that_closed_the_wrong_reaction_count(
        tmp_path):
    """The six species close three reactions. A channel reporting another
    count evaluated a different pool than the one the matrix asked for."""
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    result, _fake = _run_arch(
        tmp_path, fake=SliceMarkingRunner(run_dir, n_reactions=1))
    check = result["slice_check"]
    assert check["ok"] is False
    assert str(wm.SLICE_CLOSED_REACTIONS) in check["detail"]


def test_run_arch_does_not_check_the_channel_when_no_slice_was_asked_for(
        tmp_path):
    """``species_slice=None`` is a full-pool evaluation by request: the stage
    carries no slice variable and the channel carries no mark, so there is
    nothing to check and nothing to refuse."""
    from xcquinox.alec.full_benchmark_pools import HELDOUT_SPECIES_SLICE_ENV
    result, fake = _run_arch(tmp_path, species_slice=None)
    assert result["slice_check"]["checked"] is False
    assert result["slice_check"]["ok"] is True
    by_stage = {FakeRunner._stage_of(argv): env for argv, env in fake.calls}
    assert HELDOUT_SPECIES_SLICE_ENV not in by_stage["eval"]


def test_run_arch_reports_no_channel_when_the_eval_stages_never_ran(tmp_path):
    """A sequence that stopped before the eval stages has no channel to judge:
    the stage's own non-zero exit is the finding, and a slice verdict invented
    on top of it would report a second, spurious failure."""
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    fake = FakeRunner(run_dir, rc_by_stage={"preflight": 1})
    result, _fake = _run_arch(tmp_path, fake=fake)
    assert result["slice_check"]["ok"] is None
    assert result["slice_check"]["checked"] is False


def test_run_arch_names_the_absent_oracle_module(tmp_path):
    """The oracle module is spec 3.1's, and this runner does not own it. When
    it is not installed the selector matches nothing and pytest exits 5
    (``ExitCode.NO_TESTS_COLLECTED``); the stage must report that by name
    rather than as an anonymous non-zero exit, and never as a pass."""
    assert pytest.ExitCode.NO_TESTS_COLLECTED == wm.ORACLE_NO_TESTS_RC
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    fake = FakeRunner(run_dir, rc_by_stage={"oracles": wm.ORACLE_NO_TESTS_RC},
                      oracle_summary="no tests ran in 0.31s")
    result, _fake = _run_arch(tmp_path, fake=fake)
    oracles = result["oracle_tests"]
    assert oracles["rc"] == wm.ORACLE_NO_TESTS_RC
    assert wm.ORACLE_MODULE in oracles["summary_line"]
    log = Path(oracles["log"]).read_text()
    assert wm.ORACLE_MODULE in log
    assert "no tests ran in 0.31s" in log, "pytest's own output is kept"


def test_the_stage_table_covers_every_cell_the_template_expands_to(tmp_path):
    """``STAGE_ORDER`` carries one train and one eval stage per grid cell.

    The template fixes two cells (subset sizes 1 and 2 of the cached ledger);
    a template that grew a third would leave that cell untrained and
    unevaluated while the matrix still reported every stage green, so the
    stage table is checked against what the grid actually expands to.
    """
    from xcquinox.alec.cluster.grid_config import expand_grid, load_grid_config
    cfg = load_grid_config(str(wm.write_matrix_yaml(
        "deep", tmp_path / "deep", repo_root=wm.repo_root_path(),
        external_refs_dir=_fake_staged_refs(tmp_path / "refs"))))
    n_cells = len(expand_grid(cfg))
    assert n_cells >= 1
    assert [s for s in wm.STAGE_ORDER if s.startswith("train[")] == \
        [f"train[{i}]" for i in range(n_cells)]
    assert [s for s in wm.STAGE_ORDER if s.startswith("eval[")] == \
        [f"eval[{i}]" for i in range(n_cells)]


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


def test_run_matrix_returns_one_result_per_arch_in_input_order(tmp_path,
                                                               shared_refs):
    archs = ["shallow", "deep", "medium"]
    results = wm.run_matrix(archs, tmp_path, runner=MatrixFakeRunner(),
                            repo_root=wm.repo_root_path(),
                            external_refs_dir=shared_refs)
    assert [r["arch"] for r in results] == archs
    for r in results:
        assert [s["rc"] for s in r["stages"]] == [0] * len(wm.STAGE_ORDER)
        assert r["certificate_verdict"] == "PASS"


def test_run_matrix_stages_the_reference_copy_once(tmp_path):
    """Default path: one copy under the work root, shared by every arch."""
    import yaml
    wm.run_matrix(["shallow", "deep"], tmp_path, runner=MatrixFakeRunner(),
                  repo_root=wm.repo_root_path())
    shared = tmp_path / "_inputs" / "external_refs"
    assert shared.is_dir()
    for arch in ("shallow", "deep"):
        raw = yaml.safe_load((tmp_path / arch / "grid.yaml").read_text())
        assert raw["inputs"]["external_refs_dir"] == str(shared)


def test_run_matrix_gives_each_shard_its_own_pretrain_data_dir(tmp_path,
                                                               shared_refs):
    """Two shards generating pretrain_data_polarized.npz into one directory
    would race on a fixed filename; within a shard the architectures run
    serially, so the second one's datagen is a skip-if-current no-op."""
    import yaml
    archs = ["shallow", "deep", "medium", "shallow_attn"]
    results = wm.run_matrix(archs, tmp_path, shards=2,
                            runner=MatrixFakeRunner(),
                            repo_root=wm.repo_root_path(),
                            external_refs_dir=shared_refs)
    assert [r["arch"] for r in results] == archs
    dirs = {}
    for arch in archs:
        raw = yaml.safe_load((tmp_path / arch / "grid.yaml").read_text())
        dirs[arch] = raw["pretrain"]["data_dir"]
    assert dirs["shallow"] == dirs["medium"]        # shard 0: archs[0::2]
    assert dirs["deep"] == dirs["shallow_attn"]     # shard 1: archs[1::2]
    assert dirs["shallow"] != dirs["deep"]
    assert dirs["shallow"].endswith("pretrain_data_shard0")
    assert dirs["deep"].endswith("pretrain_data_shard1")


def test_run_matrix_refuses_an_out_of_range_shard_count(tmp_path, shared_refs):
    for bad in (0, -1, wm.MAX_SHARDS + 1):
        with pytest.raises(ValueError, match="shards"):
            wm.run_matrix(["deep"], tmp_path, shards=bad,
                          runner=MatrixFakeRunner(),
                          repo_root=wm.repo_root_path(),
                          external_refs_dir=shared_refs)


def test_run_matrix_refuses_an_unregistered_architecture(tmp_path,
                                                         shared_refs):
    with pytest.raises(ValueError, match="no_such_arch"):
        wm.run_matrix(["deep", "no_such_arch"], tmp_path,
                      runner=MatrixFakeRunner(),
                      repo_root=wm.repo_root_path(),
                      external_refs_dir=shared_refs)


def test_run_matrix_calls_the_progress_hook_once_per_arch(tmp_path,
                                                          shared_refs):
    seen = []
    wm.run_matrix(["deep", "shallow"], tmp_path, runner=MatrixFakeRunner(),
                  repo_root=wm.repo_root_path(),
                  external_refs_dir=shared_refs, progress=seen.append)
    assert sorted(r["arch"] for r in seen) == ["deep", "shallow"]


def test_arch_row_renders_a_complete_run():
    result = {
        "arch": "deep", "seconds": 702.0, "certificate_verdict": "PASS",
        "stages": [{"name": n, "rc": 0} for n in wm.STAGE_ORDER],
        "oracle_tests": {"rc": 0, "summary_line": "12 passed in 3.4s"},
    }
    row = wm.arch_row(result)
    assert row["arch"] == "deep"
    assert row["stages_rc"] == ".".join(["0"] * len(wm.STAGE_ORDER))
    assert row["certificate"] == "PASS"
    assert row["oracles"] == "0 (12 passed in 3.4s)"
    assert row["wall"] == "11m42s"


def test_arch_row_marks_the_stages_a_failure_never_reached():
    result = {
        "arch": "deep_dm", "seconds": 61.0, "certificate_verdict": "FAIL",
        "stages": [{"name": "submit", "rc": 0}, {"name": "datagen", "rc": 0},
                   {"name": "pretrain", "rc": 1}],
        "oracle_tests": {"rc": 1, "summary_line": "1 failed, 11 passed"},
    }
    row = wm.arch_row(result)
    assert row["stages_rc"] == "0.0.1.-.-.-.-.-.-.-"
    assert row["certificate"] == "FAIL"
    assert row["oracles"] == "1 (1 failed, 11 passed)"
    assert row["wall"] == "1m01s"


def test_arch_row_renders_skipped_oracles():
    result = {"arch": "deep", "seconds": 0.0, "certificate_verdict": None,
              "stages": [], "oracle_tests": {"rc": None, "summary_line": ""}}
    row = wm.arch_row(result)
    assert row["oracles"] == "skipped"
    assert row["certificate"] == "-"


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


class RaisingRunner(MatrixFakeRunner):
    """A runner that raises for one architecture instead of returning a code.

    Stands for the launch itself failing -- an interpreter that is not there, a
    permission error on the log directory -- rather than for a stage that ran
    and exited non-zero.
    """

    def __init__(self, *, raise_for, **kwargs):
        super().__init__(**kwargs)
        self.raise_for = raise_for

    def __call__(self, argv, **kwargs):
        argv = [str(a) for a in argv]
        if (FakeRunner._stage_of(argv) == "preflight"
                and self._arch_of(argv) == self.raise_for):
            raise RuntimeError("stage launch failed: no such file")
        return super().__call__(argv, **kwargs)


def test_run_matrix_records_an_architecture_whose_sequence_raised(
        tmp_path, shared_refs):
    """One architecture that dies in the runner must not take the other 30
    with it: the matrix exists to report on every architecture in one pass, so
    the exception is recorded against that architecture and the sequence moves
    on."""
    results = wm.run_matrix(["deep", "shallow"], tmp_path,
                            runner=RaisingRunner(raise_for="deep"),
                            repo_root=wm.repo_root_path(),
                            external_refs_dir=shared_refs)
    assert [r["arch"] for r in results] == ["deep", "shallow"]
    failed, clean = results
    assert "RuntimeError" in failed["error"]
    assert "no such file" in failed["error"]
    assert wm.arch_row(failed)["stages_rc"] == \
        ".".join(["-"] * len(wm.STAGE_ORDER))
    assert [s["rc"] for s in clean["stages"]] == [0] * len(wm.STAGE_ORDER)


def _clean_result(arch="deep", **overrides):
    """A result record in which every acceptance item of spec 3.4 is met."""
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


def test_write_matrix_report_counts_an_unmarked_channel_as_not_clean(tmp_path):
    """The held-out assertion is part of the acceptance list, so a run whose
    stages all exited zero but whose channel carries no slice mark is not a
    clean row."""
    path = wm.write_matrix_report([_clean_result(slice_check={
        "checked": True, "ok": False, "n_reactions": None, "channels": [],
        "detail": "spec_0000/eval_holdout carries no readable "
                  "sliced_eval.json"})], tmp_path / "matrix.md")
    text = path.read_text()
    assert "0 of 1" in text
    assert "sliced_eval.json" in text


def test_write_matrix_report_names_an_architecture_that_raised(tmp_path):
    results = [_clean_result(),
               _clean_result("shallow", stages=[], run_dir=None,
                             certificate_verdict=None,
                             error="RuntimeError: stage launch failed: "
                                   "no such file")]
    text = wm.write_matrix_report(results, tmp_path / "matrix.md").read_text()
    assert "1 of 2" in text
    assert "shallow" in text
    assert "no such file" in text


def test_write_matrix_report_keeps_every_stage_record_in_the_sidecar(tmp_path):
    """The table cannot hold the per-stage wall clock, the log paths or the
    artefact records, and a later comparison needs them; the JSON sidecar
    carries the records verbatim."""
    wm.write_matrix_report([_clean_result()], tmp_path / "matrix.md")
    sidecar = json.loads((tmp_path / "matrix.json").read_text())
    stages = sidecar["results"][0]["stages"]
    assert [s["name"] for s in stages] == list(wm.STAGE_ORDER)
    assert all(s["log"].endswith(".log") and s["seconds"] >= 0.0
               for s in stages)
    assert sidecar["results"][0]["certificate_verdict"] == "PASS"
    assert sidecar["results"][0]["slice_check"]["ok"] is True
    assert sidecar["n_clean"] == 1


# ---------------------------------------------------------------------------
# Oracle selection: case, and oracle function names
# ---------------------------------------------------------------------------

def test_oracle_selector_excludes_a_longer_name_whatever_its_case():
    """pytest's ``-k`` matching is case-INSENSITIVE
    (``KeywordMatcher.__call__`` lowercases both sides), so the containment
    test that builds the exclusions has to be case-insensitive too. A registry
    entry differing only in case from a longer one would otherwise be left
    unexcluded, and the shorter name's selector would silently carry the
    longer architecture's cases.
    """
    got = wm.oracle_selector("mgga", archs=["mgga", "deep_MGGA_3x16"])
    assert got == "test_spin_scaling_oracles and mgga and not deep_MGGA_3x16"
    expr = _compile_k(got)
    assert expr.evaluate(_k_matcher(
        _item_names("test_o1_uniform_scaling[mgga]")))
    assert not expr.evaluate(_k_matcher(
        _item_names("test_o1_uniform_scaling[deep_MGGA_3x16]")))


def _oracle_node_id(name: str) -> str:
    """A collected node id in the oracle module, as pytest prints it."""
    return f"xcquinox/alec/tests/{wm.ORACLE_MODULE}.py::{name}"


def test_oracle_function_names_carrying_a_registry_name_are_reported():
    """``-k`` matches the FUNCTION name as well as the parametrisation id, so
    an oracle called ``test_deep_channel_gradient`` answers to every selector
    naming ``deep`` and would be collected -- and reported -- as that
    architecture's oracle whatever its own parameters say. The selector cannot
    defend against it, so the collected names are checked instead.
    """
    clean = [_oracle_node_id("test_o1_uniform_scaling[deep]"),
             _oracle_node_id("test_o3_spin_scaling_open_shell[Li-shallow]"),
             _oracle_node_id("test_o4_spin_scaling_relation[medium_attn-Li]")]
    assert wm.oracle_function_name_conflicts(clean) == []

    dirty = clean + [_oracle_node_id("test_deep_channel_gradient[shallow]")]
    conflicts = wm.oracle_function_name_conflicts(dirty)
    assert len(conflicts) == 1, conflicts
    assert "test_deep_channel_gradient" in conflicts[0]
    assert "'deep'" in conflicts[0]
    # The comparison follows -k: case-insensitive on both sides.
    assert wm.oracle_function_name_conflicts(
        [_oracle_node_id("test_DEEP_probe[shallow]")], archs=["deep"])
    assert wm.oracle_function_name_conflicts(
        [_oracle_node_id("test_o1_uniform_scaling[deep]")],
        archs=["deep"]) == []


def test_write_matrix_yaml_refuses_a_refs_dir_whose_manifest_lists_nothing(
        tmp_path):
    """The staging manifest is a criterion only when it is CHECKED against the
    files it records. A manifest recording no file, or one whose files are
    gone, describes a directory carrying no references, and accepting it on
    the manifest's mere presence defers the failure to a compute node exactly
    as an empty directory does -- which is the case this refusal exists for.
    """
    for name, manifest in (("no_files", "source: /nowhere\n"),
                           ("gone", "source: /nowhere\nH2O.npz\n")):
        refs = tmp_path / f"refs_{name}"
        refs.mkdir()
        (refs / wm.STAGE_MARKER).write_text(manifest)
        with pytest.raises(wm.CachedInputsMissing,
                           match=re.escape(os.path.realpath(refs))):
            wm.write_matrix_yaml("deep", tmp_path / f"out_{name}",
                                 repo_root=wm.repo_root_path(),
                                 external_refs_dir=refs)


# ---------------------------------------------------------------------------
# The certificate: a waived FAIL is the expected outcome, an absent one is not
# ---------------------------------------------------------------------------

def test_run_arch_carries_the_certificate_verdict_and_its_waiver(tmp_path):
    """A FAIL verdict is the EXPECTED outcome of this identity -- 50
    pretraining steps on two atoms cannot reproduce the parent functional to
    tol_AE = 1.0 kcal/mol -- so it is recorded, together with the waiver that
    let the sequence continue past the on-node gates, and it does not make the
    architecture a failed one.
    """
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    result, _fake = _run_arch(tmp_path,
                              fake=SliceMarkingRunner(run_dir, verdict="FAIL"))
    assert [s["name"] for s in result["stages"]] == list(wm.STAGE_ORDER)
    certificate = result["certificate"]
    assert certificate["present"] is True
    assert certificate["verdict"] == "FAIL"
    assert result["certificate_verdict"] == "FAIL"
    assert certificate["enforced"] is False
    assert certificate["override_reason"] == \
        _template_fidelity()["override_reason"]
    # What the preflight sweep and the train task will do with it.
    assert certificate["gate_released"] is True
    assert certificate["path"].endswith(
        "pretrain/deep/fidelity_certificate.json")
    assert wm.arch_row(result)["certificate"] == "FAIL (waived)"
    assert wm.matrix_exit_code([result]) == 0


def test_run_arch_fails_the_certificate_stage_when_none_was_written(tmp_path):
    """The waiver covers a FAIL VERDICT, not a missing certificate.
    ``fidelity.gate_certificate`` refuses an absent one outright, so the
    preflight sweep and the train task would refuse the run, and the matrix
    would have no verdict to report; the stage is therefore recorded as failed
    whatever its own exit code said, and the sequence stops.
    """
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    result, _fake = _run_arch(tmp_path, fake=FakeRunner(run_dir, verdict=None))
    assert [s["name"] for s in result["stages"]] == [
        "submit", "datagen", "pretrain", "certificate"]
    assert result["stages"][-1]["rc"] == wm.CERTIFICATE_MISSING_RC
    assert result["certificate"]["present"] is False
    assert result["certificate"]["gate_released"] is False
    assert result["certificate_verdict"] is None
    assert wm.arch_row(result)["certificate"] == "missing"
    assert wm.matrix_exit_code([result]) == 1
    log = Path(result["stages"][-1]["log"]).read_text()
    assert "fidelity_certificate.json" in log


def test_run_arch_reports_an_unreadable_certificate_as_missing(tmp_path):
    """A truncated or half-written certificate is not a verdict: the gate
    refuses it exactly as it refuses an absent one, so the matrix must not
    report the stage as done."""
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"

    class _Truncated(FakeRunner):
        def __call__(self, argv, **kwargs):
            completed = super().__call__(argv, **kwargs)
            if FakeRunner._stage_of(argv) == "certificate":
                (self.run_dir / "pretrain" / "deep"
                 / "fidelity_certificate.json").write_text('{"verdict": "PA')
            return completed

    result, _fake = _run_arch(tmp_path, fake=_Truncated(run_dir))
    assert result["stages"][-1]["name"] == "certificate"
    assert result["stages"][-1]["rc"] == wm.CERTIFICATE_MISSING_RC
    assert result["certificate"]["present"] is False


def test_write_matrix_report_names_a_missing_certificate(tmp_path):
    text = wm.write_matrix_report([_clean_result(certificate={
        "present": False, "verdict": None, "enforced": None,
        "override_reason": None, "gate_released": False,
        "path": "/w/deep/pretrain/deep/fidelity_certificate.json",
        "gate_message": "no fidelity certificate at /w/deep/pretrain/deep"})],
        tmp_path / "matrix.md").read_text()
    assert "0 of 1" in text
    assert "wrote no certificate" in text
    assert "/w/deep/pretrain/deep/fidelity_certificate.json" in text
    # The waiver is stated where a reader meets the certificate column.
    assert "fidelity.enforce: false" in text
    assert "EXPECTED" in text
