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

import itertools
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
    # The certificate is computed at the identity this template renders --
    # def2-svp / grid level 1, since fidelity.run_identity and
    # build_oracle_set both read cfg.inputs -- and its verdict is the real one
    # there; what the template waives is the ENFORCEMENT, without which the
    # on-node gates (cluster/_preflight.py, cluster/_train_task.py, both through
    # fidelity.gate_certificate) would block every architecture at a FAIL that
    # 50 pretraining steps on two atoms cannot avoid. The tolerances are NOT
    # written here: they stay at the program's binding defaults.
    assert "production identity" not in wm.template_path().read_text(), (
        "the certificate runs at the RENDERED identity (def2-svp, grid level "
        "1), which test_the_certificate_runs_at_the_rendered_identity measures")
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


def _certificate_run_dir(tmp_path, arch="deep"):
    """A run directory holding one real certificate for ``arch``."""
    run_dir = tmp_path / "runs" / "run_20260821T000000Z"
    pretrain = Path(wm.pretrain_checkpoint_dir(str(run_dir), arch))
    pretrain.mkdir(parents=True)
    path = pretrain / "fidelity_certificate.json"
    path.write_text(json.dumps(_MD1))
    return run_dir, path


def test_the_certificate_record_describes_one_document(tmp_path, monkeypatch):
    """The report's record corresponds to a document, not to a sequence.

    The verdict, the waiver and the gate's decision were each read from the
    file separately, so a certificate rewritten between the opens assembled a
    record out of three documents. Three documents that are each refused on
    their own, and each produce a different record, must reproduce the record
    of the FIRST -- on one read -- in every order they are served in.
    """
    run_dir, path = _certificate_run_dir(tmp_path)
    documents = (_MD1, _MD2, _MD3)
    alone = []
    for doc in documents:
        served = _serve_documents(monkeypatch, path, [doc])
        record = wm._certificate_record(run_dir, "deep")
        monkeypatch.undo()
        assert len(served) == 1, (doc, served)
        alone.append(record)
    assert len({json.dumps(r, sort_keys=True) for r in alone}) == 3, alone
    for order in itertools.permutations(range(len(documents))):
        served = _serve_documents(monkeypatch, path,
                                  [documents[i] for i in order])
        record = wm._certificate_record(run_dir, "deep")
        monkeypatch.undo()
        assert len(served) == 1, (order, served)
        assert record == alone[order[0]], (order, record)


def test_no_document_sequence_records_a_waiver_beside_a_refusal(
        tmp_path, monkeypatch):
    """A COMPLETE waiver beside a gate that refused describes no certificate.

    ``enforced: false`` together with a recorded ``override_reason`` is what
    releases the gate, so the pair can never sit beside ``gate_released``
    False. Measured on the sequence D3 -> D1 -> D2, it did: the reason came
    from the document read for the verdict, the refusal from the document the
    gate read, and the waiver from a third.
    """
    run_dir, path = _certificate_run_dir(tmp_path)
    for order in itertools.permutations((_MD1, _MD2, _MD3)):
        served = _serve_documents(monkeypatch, path, list(order))
        record = wm._certificate_record(run_dir, "deep")
        monkeypatch.undo()
        assert len(served) == 1, (order, served)
        complete_waiver = (record["enforced"] is False
                           and bool(record["override_reason"]))
        assert not (complete_waiver and not record["gate_released"]), (
            order, record)


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


@pytest.mark.parametrize("recorded", [0, 2, 4, None, "3"])
def test_run_arch_refuses_a_slice_that_closed_the_wrong_reaction_count(
        tmp_path, recorded):
    """The six species close three reactions. A channel reporting another
    count evaluated a different pool than the one the matrix asked for.

    The count is checked for EQUALITY with :data:`SLICE_CLOSED_REACTIONS`, so
    every near miss is refused and not only an obviously empty one: a count
    below it (0) or above it (4), the off-by-one that a single dropped species
    produces (2), an absent or null field (``None``, which a channel written by
    an older eval stage carries), and the string ``"3"``, which compares
    unequal to the integer and would otherwise let a JSON schema change pass
    unnoticed.
    """
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    result, _fake = _run_arch(
        tmp_path, fake=SliceMarkingRunner(run_dir, n_reactions=recorded))
    check = result["slice_check"]
    assert check["ok"] is False
    assert f"n_reactions={recorded!r}" in check["detail"]
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
    """The oracle module is spec 3.1's, and this runner does not own it. An
    EMPTY target -- the module installed but no node id matching the selector
    -- exits 5 (``ExitCode.NO_TESTS_COLLECTED``); an ABSENT module cannot be
    collected at all and exits pytest's USAGE code 4. The stage must report
    either by name rather than as an anonymous non-zero exit, and never as a
    pass."""
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


def test_the_oracle_stage_targets_the_module_and_leaves_no_cache(tmp_path):
    """The oracle stage names the oracle MODULE, not the directory holding it.

    ``-k`` narrows what is RUN; it does not narrow what is COLLECTED, so with
    the directory as the target one unrelated module that fails to import
    stops the run before a single oracle executes -- identically for every
    architecture, and with an exit code that reads as a failed oracle.
    ``-p no:cacheprovider`` keeps pytest's cache directory out of the
    checkout, which is the one tree the job otherwise never writes into.
    """
    _result, fake = _run_arch(tmp_path)
    argvs = [argv for argv, _env in fake.calls
             if FakeRunner._stage_of(argv) == "oracles"]
    assert len(argvs) == 1
    argv = argvs[0]
    assert wm.ORACLE_TEST_TARGET.endswith(f"{wm.ORACLE_MODULE}.py")
    assert wm.ORACLE_TEST_TARGET in argv
    assert "no:cacheprovider" in argv


def test_the_oracle_stage_survives_a_broken_module_elsewhere_in_the_tree(
        tmp_path):
    """A REAL pytest run, in a tree carrying one module that raises on import.

    The oracle stage runs the installed test tree, which the matrix does not
    own: any module in it that fails to import is a collection error, and
    pytest stops the whole session on one. Collected as a directory, the
    broken module below takes the oracles of all 31 architectures with it;
    collected as the module, the oracles run.
    """
    tree = tmp_path / "checkout"
    tests_dir = tree / "xcquinox" / "alec" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / f"{wm.ORACLE_MODULE}.py").write_text(
        "import pytest\n\n"
        f"@pytest.mark.parametrize('arch', {sorted(ARCHITECTURES)!r})\n"
        "def test_o1(arch):\n"
        "    assert arch\n")
    (tests_dir / "test_unrelated_module.py").write_text(
        "raise RuntimeError('this module does not import')\n")
    record = wm._run_oracles("deep", tmp_path / "oracles.log",
                             runner=subprocess.run, env=dict(os.environ),
                             timeout_s=600, cwd=tree)
    log = Path(record["log"]).read_text()
    assert record["rc"] == 0, log
    assert "1 passed" in record["summary_line"], log
    assert not (tree / ".pytest_cache").exists()


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
        elif stage == "eval":
            # ``cluster/_eval_one_spec`` marks a sliced held-out channel twice
            # -- ``sliced_eval.json`` before any energy is computed, a
            # ``species_slice`` entry in ``eval_metadata.json`` after them --
            # and ``_slice_check`` reads both, so the fake writes both for the
            # slice the stage was actually handed. The reaction closure is
            # known here for the default slice only; any other slice is
            # recorded without one, which the channel check reports as the
            # mismatch it is.
            from xcquinox.alec.full_benchmark_pools import (
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


def test_run_matrix_sizes_the_stage_threads_from_the_allocation(
        tmp_path, shared_refs, monkeypatch):
    """The per-stage BLAS budget is the ALLOCATION divided by the shards.

    ``os.cpu_count()`` reports the MACHINE: on an SMT node it is twice the
    cores, and on a shared queue it is the whole box rather than the slice
    this job holds, so shards sized from it oversubscribe every stage and the
    stages then contend for the same cores. SLURM states what the job may use
    in ``SLURM_CPUS_PER_TASK``.
    """
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "40")
    fake = MatrixFakeRunner()
    wm.run_matrix(["deep", "shallow", "medium", "shallow_attn"], tmp_path,
                  shards=4, runner=fake, repo_root=wm.repo_root_path(),
                  external_refs_dir=shared_refs)
    assert fake.calls
    for argv, env in fake.calls:
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS"):
            assert env[key] == "10", (key, env[key], argv)


@pytest.mark.parametrize("value", ["", "0", "not-a-number"])
def test_run_matrix_falls_back_to_the_machine_without_an_allocation(
        tmp_path, shared_refs, monkeypatch, value):
    """Outside SLURM -- a workstation run, or a site that leaves the variable
    empty -- the machine's own count is the only budget there is."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", value)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    fake = MatrixFakeRunner()
    wm.run_matrix(["deep", "shallow"], tmp_path, shards=2, runner=fake,
                  repo_root=wm.repo_root_path(),
                  external_refs_dir=shared_refs)
    assert fake.calls
    for _argv, env in fake.calls:
        assert env["OMP_NUM_THREADS"] == "4"


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


def test_run_matrix_refuses_a_repeated_architecture(tmp_path, shared_refs):
    """One architecture named twice is two rows over ONE working directory.

    ``run_arch`` derives everything from ``<work_root>/<arch>``: both rows
    write the same ``grid.yaml`` and, dealt into different shards, write it
    concurrently while the other row's submit stage reads it, so the second run
    directory replaces the first and the two rows report on one run. The report
    would show two independent-looking lines for it.
    """
    with pytest.raises(ValueError, match="more than once"):
        wm.run_matrix(["deep", "shallow", "deep"], tmp_path,
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


def test_matrix_exit_code_refuses_an_empty_matrix():
    """``all(())`` is True, so an empty record list would report a clean pass
    of a matrix that ran nothing -- the one answer that must never be reached
    by accident, since a pass that died before its first architecture produces
    exactly that list. ``run_matrix`` refuses an empty architecture list for
    the same reason, and the predicate refuses the empty result of one.
    """
    with pytest.raises(ValueError, match="no architecture records"):
        wm.matrix_exit_code([])


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


def test_write_matrix_report_states_the_identity_the_certificate_ran_at(
        tmp_path):
    """The verdict is only interpretable against the identity it was measured
    at, and that identity is the RENDERED one: ``fidelity.run_identity`` and
    ``build_oracle_set`` both read ``cfg.inputs``, which this template sets to
    def2-svp / grid level 1 (measured by
    ``test_the_certificate_runs_at_the_rendered_identity``), not the campaign's
    6-311++G(3df,2pd) / grid 3.
    """
    text = wm.write_matrix_report([_clean_result()],
                                  tmp_path / "matrix.md").read_text()
    assert "production identity" not in text
    assert "def2-svp" in text
    assert "grid level 1" in text


def _findings_of(text):
    """The findings block of a rendered report, or None when it has none."""
    parts = text.split("## Findings", 1)
    return parts[1] if len(parts) == 2 else None


def test_write_matrix_report_names_the_stage_that_failed(tmp_path):
    """A datagen failure stops the sequence four stages before the certificate,
    so the record carries none -- but the certificate is not what failed, and a
    findings block naming only the absent certificate points the reader at the
    wrong stage. The first non-zero exit this identity does not expect is named
    with its return code and its log, and the certificate finding is withheld
    while that stage never ran.
    """
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    result, _fake = _run_arch(
        tmp_path, fake=FakeRunner(run_dir, rc_by_stage={"datagen": 1}))
    assert [s["name"] for s in result["stages"]] == ["submit", "datagen"]
    assert result["certificate"]["present"] is False
    text = wm.write_matrix_report([result], tmp_path / "matrix.md").read_text()
    findings = _findings_of(text)
    assert findings is not None, text
    assert "datagen" in findings
    assert "exited 1" in findings
    assert result["stages"][-1]["log"] in findings
    # The header explains that an absent certificate IS a stage failure; what
    # must not appear is that finding against this architecture, whose
    # certificate stage never ran.
    assert "wrote no certificate" not in findings


def test_write_matrix_report_names_a_failure_after_the_certificate(tmp_path):
    """An architecture stopped at the preflight wrote its certificate and
    marked no held-out channel, so no existing finding applies to it: the
    return-code column was the only record that anything had gone wrong, and a
    report with no findings block reads as a clean pass.
    """
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    result, _fake = _run_arch(
        tmp_path, fake=FakeRunner(run_dir, rc_by_stage={"preflight": 1}))
    assert result["certificate"]["present"] is True
    assert result["slice_check"]["ok"] is None
    assert wm.matrix_exit_code([result]) == 1
    text = wm.write_matrix_report([result], tmp_path / "matrix.md").read_text()
    findings = _findings_of(text)
    assert findings is not None, text
    assert "preflight" in findings
    assert "exited 1" in findings
    assert result["stages"][-1]["log"] in findings


def test_write_matrix_report_names_an_enforced_failing_certificate(tmp_path):
    """With no waiver in the rendered config a FAIL verdict is what the on-node
    gates REFUSE (``gate_released`` False), and that is a different outcome
    from the waived FAIL this template expects. The column renders both as
    ``FAIL``, so the un-waived case has to name itself below the table.
    """
    record = _clean_result(certificate={
        "present": True, "verdict": "FAIL", "enforced": True,
        "override_reason": None, "gate_released": False,
        "path": "/w/deep/pretrain/deep/fidelity_certificate.json",
        "gate_message": "pretrain/deep: fidelity certificate verdict 'FAIL', "
                        "expected 'PASS'"})
    text = wm.write_matrix_report([record], tmp_path / "matrix.md").read_text()
    findings = _findings_of(text)
    assert findings is not None, text
    assert "ENFORCED" in findings
    assert "'FAIL'" in findings
    # A certificate that exists but is refused is not a certificate that was
    # never written, and the two findings must not be confused.
    assert "wrote no certificate" not in findings
    # The finding and the count above the table are one judgement. A row the
    # on-node gates refuse must not be counted among the clean ones while the
    # block below the table says the gates refuse it.
    assert wm.matrix_exit_code([record]) == 1
    assert "0 of 1" in text


def test_write_matrix_report_names_a_stage_list_that_ended_short(tmp_path):
    """``_is_clean`` requires one stage record per :data:`STAGE_ORDER` entry,
    whatever the return codes are: the acceptance item is that the whole
    sequence RAN. Through ``run_arch`` a short list always carries the non-zero
    exit that truncated it and that stage is named above, so this record --
    short, every return code zero -- is the one way of being scored non-clean
    that no other finding reaches. The invariant is asserted outright rather
    than left to hold by the caller's construction.
    """
    short = [{"name": name, "rc": 0, "seconds": 1.0,
              "log": f"/w/deep/logs/{name}.log"}
             for name in wm.STAGE_ORDER[:4]]
    record = _clean_result(stages=short)
    assert wm.matrix_exit_code([record]) == 1
    findings = _findings_of(wm.write_matrix_report(
        [record], tmp_path / "matrix.md").read_text())
    assert findings is not None, "a short stage list scored in silence"
    assert record["arch"] in findings
    for name in wm.STAGE_ORDER[4:]:
        assert name in findings, name


def _shuffled_stage_order():
    """The stage order with two entries transposed: the same names, the same
    number of them, a sequence that is not the one spec 3.4 fixes."""
    names = list(wm.STAGE_ORDER)
    names[3], names[4] = names[4], names[3]
    return names


def test_is_clean_requires_the_stage_records_to_be_the_stage_order(tmp_path):
    """The acceptance item is that the whole SEQUENCE ran, in its order.

    Counting the records instead admits a record carrying the right number of
    stages under the wrong ones -- one stage recorded twice while another
    never ran, or a renamed stage -- and scores it clean on a sequence the
    matrix never drove. The transposed record below has ten records, all zero,
    and is refused by name.
    """
    record = _clean_result(stages=[
        {"name": name, "rc": 0, "seconds": 1.0,
         "log": f"/w/deep/logs/{name}.log"}
        for name in _shuffled_stage_order()])
    assert len(record["stages"]) == len(wm.STAGE_ORDER)
    assert wm.matrix_exit_code([record]) == 1
    findings = _findings_of(wm.write_matrix_report(
        [record], tmp_path / "matrix.md").read_text())
    assert findings is not None, "an out-of-order sequence scored in silence"
    assert record["arch"] in findings
    assert "stage order" in findings


def test_write_matrix_report_names_a_certificate_stating_no_gate_decision(
        tmp_path):
    """``_is_clean`` requires the certificate record to STATE that the on-node
    gates released the run: a record that states nothing is no evidence that
    they would. Scored on that field above the table, it has to be named below
    it, or the count and the findings disagree about a record.
    """
    record = _clean_result(certificate={
        "present": True, "verdict": "FAIL", "enforced": False,
        "override_reason": "workflow matrix", "gate_released": None,
        "path": "/w/deep/pretrain/deep/fidelity_certificate.json",
        "gate_message": ""})
    assert wm.matrix_exit_code([record]) == 1
    findings = _findings_of(wm.write_matrix_report(
        [record], tmp_path / "matrix.md").read_text())
    assert findings is not None, "a record with no gate decision scored in silence"
    assert record["arch"] in findings
    assert "gate decision" in findings
    # The enforced-FAIL wording is a different finding and must not be reused:
    # this record's config DOES carry the waiver.
    assert "is ENFORCED" not in findings


def test_write_matrix_report_names_every_architecture_it_scores_non_clean(
        tmp_path):
    """The findings block is the whole of what a reader has below the table,
    so no architecture may be scored non-clean in silence.

    One record is built for each way ``_is_clean`` can refuse one -- the
    sequence raising, a stage exiting non-zero, no certificate, an unmarked
    held-out channel, a ``validate_run`` that accepted the waived run, and a
    failing oracle (which is not one of ``STAGE_ORDER`` and so is reachable by
    no stage finding) -- and each must appear by name. The clean record is the
    control: it must produce no findings block at all.
    """
    cases = {
        "sequence raised": _clean_result(
            stages=[], error="RuntimeError: stage launch failed"),
        "stage failed": _clean_result(stages=[
            {"name": "submit", "rc": 0, "log": "/w/deep/logs/submit.log"},
            {"name": "datagen", "rc": 1, "log": "/w/deep/logs/datagen.log"}]),
        "no certificate": _clean_result(certificate={
            "present": False, "verdict": None, "enforced": None,
            "override_reason": None, "gate_released": False,
            "path": "/w/deep/pretrain/deep/fidelity_certificate.json",
            "gate_message": "no fidelity certificate"}),
        "unmarked channel": _clean_result(slice_check={
            "checked": True, "ok": False, "n_reactions": None,
            "channels": [], "detail": "no readable sliced_eval.json"}),
        "validate_run accepted": _clean_result(validate_run={
            "expected": False, "rc": 0, "failures": [],
            "detail": "validate_run exited 0 on a waived FAIL certificate"}),
        "oracles failed": _clean_result(oracle_tests={
            "rc": 1, "summary_line": "1 failed, 11 passed in 4.2s",
            "log": "/w/deep/logs/oracles.log"}),
        "certificate enforced": _clean_result(certificate={
            "present": True, "verdict": "FAIL", "enforced": True,
            "override_reason": None, "gate_released": False,
            "path": "/w/deep/pretrain/deep/fidelity_certificate.json",
            "gate_message": "verdict FAIL and nothing waives it"}),
        "stage list short": _clean_result(stages=[
            {"name": name, "rc": 0, "log": f"/w/deep/logs/{name}.log"}
            for name in wm.STAGE_ORDER[:2]]),
        "certificate states no gate decision": _clean_result(certificate={
            "present": True, "verdict": "FAIL", "enforced": False,
            "override_reason": "workflow matrix", "gate_released": None,
            "path": "/w/deep/pretrain/deep/fidelity_certificate.json",
            "gate_message": ""}),
        "stage sequence out of order": _clean_result(stages=[
            {"name": name, "rc": 0, "log": f"/w/deep/logs/{name}.log"}
            for name in _shuffled_stage_order()]),
    }
    for label, record in cases.items():
        assert wm.matrix_exit_code([record]) == 1, label
        findings = _findings_of(wm.write_matrix_report(
            [record], tmp_path / "matrix.md").read_text())
        assert findings is not None, f"{label}: no findings block"
        assert record["arch"] in findings, label
    assert _findings_of(wm.write_matrix_report(
        [_clean_result()], tmp_path / "clean.md").read_text()) is None


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
    # validate_run must REFUSE such a run -- it requires a PASS verdict and
    # ignores the waiver -- so the fake refuses it for that one reason; a
    # validate_run that exited 0 here would itself be the matrix's failure.
    result, _fake = _run_arch(tmp_path, fake=ValidateRunRunner(
        run_dir, verdict="FAIL",
        failures=(_certificate_refusal(tmp_path),)))
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
    # The findings block carries the stage's return code and log path, so the
    # reader reaches the evidence without opening the JSON sidecar.
    findings = _findings_of(wm.write_matrix_report(
        [result], tmp_path / "matrix.md").read_text())
    assert findings is not None
    assert f"exited {wm.CERTIFICATE_MISSING_RC}" in findings
    assert result["stages"][-1]["log"] in findings


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


# ---------------------------------------------------------------------------
# validate_run: a record layer that MUST refuse this identity's run
# ---------------------------------------------------------------------------

def _certificate_refusal(tmp_path, arch="deep", verdict="FAIL") -> str:
    """``cluster/validate_run``'s OWN certificate-verdict refusal of ``arch``.

    The line is not spelled a second time here. A run directory carrying the
    waived certificate this identity produces is handed to
    ``validate_run.validate_run``, and the failure it returns is what the
    matrix's parser is then given. A hand-written copy drifts, and had: the
    validator appends a waiver clause for a certificate recording
    ``enforced: false`` -- the matrix's own case, every run of it -- and the
    copy carried only the verdict clause, so the parser was exercised on text
    no run produces.

    The synthetic run directory holds the certificate and a manifest and
    nothing else, so the validator also reports the absent specs and the
    certificate's absent identity block; the verdict refusal is selected out of
    them and there is exactly one of it.
    """
    from xcquinox.alec.cluster import validate_run as validate_run_module

    root = Path(tmp_path) / "_validate_run_text" / f"{arch}_{verdict}"
    grid = wm.write_matrix_yaml(
        arch, root / "render", repo_root=wm.repo_root_path(),
        external_refs_dir=_fake_staged_refs(root / "refs"))
    run_dir = root / "run"
    pretrain_dir = run_dir / "pretrain" / arch
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    (pretrain_dir / "fidelity_certificate.json").write_text(
        json.dumps(_certificate_payload(verdict)))
    (run_dir / "manifest.json").write_text(json.dumps({"width": 4}))
    failures, _warnings, _n_specs = validate_run_module.validate_run(
        str(run_dir), str(grid))
    verdicts = [f for f in failures if "fidelity certificate verdict" in f]
    assert len(verdicts) == 1, failures
    return verdicts[0]


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


def _validate_run_arch(tmp_path, **kwargs):
    """One architecture through the sequence with a waived FAIL certificate."""
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    fake = ValidateRunRunner(run_dir, verdict="FAIL", **kwargs)
    return _run_arch(tmp_path, fake=fake)[0]


def test_run_arch_accepts_the_expected_validate_run_refusal(tmp_path):
    """``validate_run`` is a record layer and stays strict: it requires a PASS
    certificate and ignores the waiver, so it MUST refuse a run rendered from
    this template. That refusal, alone, is the expected outcome of the stage
    and does not make the architecture a failed one.
    """
    result = _validate_run_arch(
        tmp_path, failures=(_certificate_refusal(tmp_path),))
    assert [s["name"] for s in result["stages"]] == list(wm.STAGE_ORDER)
    assert result["validate_run"]["expected"] is True
    assert result["validate_run"]["detail"] == wm.VALIDATE_RUN_EXPECTED_DETAIL
    assert result["validate_run"]["rc"] == 1
    # The fingerprint says "expected non-zero", not "broken".
    assert wm.arch_row(result)["stages_rc"].endswith(".1w")
    assert wm.matrix_exit_code([result]) == 0
    text = wm.write_matrix_report([result], tmp_path / "matrix.md").read_text()
    assert "validate_run: refused the waived certificate as expected" in text
    assert "1 of 1" in text


def test_run_arch_refuses_a_validate_run_that_accepted_the_waived_run(
        tmp_path):
    """A zero exit would mean the record layer had stopped refusing a run whose
    certificate records a FAIL -- the guarantee that keeps a workflow-matrix
    run out of the results."""
    result = _validate_run_arch(tmp_path, failures=(), validate_rc=0)
    assert result["validate_run"]["expected"] is False
    assert "exited 0" in result["validate_run"]["detail"]
    assert wm.matrix_exit_code([result]) == 1
    text = wm.write_matrix_report([result], tmp_path / "matrix.md").read_text()
    assert "0 of 1" in text
    assert "validate_run --" in text


def test_run_arch_refuses_a_second_validate_run_failure(tmp_path):
    """A second failure produces the same exit code as the expected one, so
    without reading the report it would hide behind it."""
    refusal = _certificate_refusal(tmp_path)
    other = "specs/spec_0000.spec: arch 'deep' has n_extra_features 3, expected 4"
    result = _validate_run_arch(tmp_path, failures=(refusal, other))
    assert result["validate_run"]["expected"] is False
    assert result["validate_run"]["failures"] == [refusal, other]
    assert "n_extra_features" in result["validate_run"]["detail"]
    assert wm.matrix_exit_code([result]) == 1
    assert wm.arch_row(result)["stages_rc"].endswith(".1")


def test_run_arch_refuses_a_refusal_naming_another_architecture(tmp_path):
    """The refusal has to be the one for the architecture under test: a
    certificate refusal of a DIFFERENT architecture means the run directory
    carries a sweep this matrix did not render."""
    result = _validate_run_arch(
        tmp_path,
        failures=(_certificate_refusal(tmp_path, arch="shallow"),))
    assert result["validate_run"]["expected"] is False
    assert wm.matrix_exit_code([result]) == 1


def test_the_expected_refusal_is_read_from_validate_runs_own_text(tmp_path):
    """The matrix's parser is exercised on the text ``validate_run`` writes for
    the case the matrix actually produces: a FAIL certificate recording
    ``enforced: false``. For that certificate the validator appends a waiver
    clause after the verdict clause, quoting the run's own override reason, so
    the refusal does not end where a reading of the verdict check alone would
    put its end. One such line is still exactly one failure, and the expected
    one.
    """
    from xcquinox.alec.cluster.fidelity import VERDICT_PASS

    line = _certificate_refusal(tmp_path)
    head = f"expected {VERDICT_PASS!r}"
    assert head in line
    assert not line.endswith(head), (
        "validate_run appends a waiver clause for a certificate recording "
        "enforced: false; a line ending at the verdict clause is not the text "
        "this identity produces")
    assert _template_fidelity()["override_reason"] in line

    result = _validate_run_arch(tmp_path, failures=(line,))
    assert result["validate_run"]["failures"] == [line]
    assert result["validate_run"]["expected"] is True
    assert result["validate_run"]["detail"] == wm.VALIDATE_RUN_EXPECTED_DETAIL
    assert wm.matrix_exit_code([result]) == 0


def test_run_arch_expects_a_clean_validate_run_without_the_waiver(tmp_path):
    """With a PASS certificate no waiver is in play and the ordinary contract
    applies: exit zero is the expected outcome and carries no marker."""
    result, _fake = _run_arch(tmp_path)      # default fake: PASS certificate
    assert result["certificate"]["verdict"] == "PASS"
    assert result["validate_run"]["expected"] is True
    assert result["validate_run"]["rc"] == 0
    assert wm.arch_row(result)["stages_rc"].endswith(".0")


def test_the_certificate_runs_at_the_rendered_identity(tmp_path):
    """The certificate is computed at the RUN's identity, not at a separate
    production one: ``fidelity.run_identity`` reads the config's own basis,
    grid level and Coulomb backend, and ``build_oracle_set`` builds every
    oracle system at those values. ``validate_run`` in turn refuses a
    certificate whose identity block differs from the config's, so the two
    cannot drift. At the matrix identity that is def2-svp / grid level 1 --
    the cost of the certificate stage is a def2-svp calculation, not a
    6-311++G(3df,2pd) one.
    """
    from xcquinox.alec.cluster import fidelity
    from xcquinox.alec.cluster.grid_config import load_grid_config
    cfg = load_grid_config(str(wm.write_matrix_yaml(
        "deep", tmp_path / "deep", repo_root=wm.repo_root_path(),
        external_refs_dir=_fake_staged_refs(tmp_path / "refs"))))
    identity = fidelity.run_identity(cfg)
    assert identity["basis"] == cfg.inputs.basis == "def2-svp"
    assert identity["grid_level"] == cfg.inputs.grid_level == 1
    assert identity["density_fit"] is False
    systems = fidelity.build_oracle_set(cfg, "deep")
    assert systems
    assert {s.basis for s in systems} == {"def2-svp"}
    assert {s.grid_level for s in systems} == {1}


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


def test_main_defaults_the_report_into_the_work_root(tmp_path, shared_refs):
    assert wm.main(["--archs", "deep", "--work-root", str(tmp_path),
                    "--external-refs-dir", str(shared_refs)],
                   runner=MatrixFakeRunner()) == 0
    assert (tmp_path / "workflow_matrix.md").is_file()
    assert (tmp_path / "workflow_matrix.json").is_file()


def test_main_all_selects_every_registered_architecture(tmp_path,
                                                        shared_refs):
    wm.main(["--archs", "all", "--work-root", str(tmp_path),
             "--external-refs-dir", str(shared_refs), "--no-oracles"],
            runner=MatrixFakeRunner())
    sidecar = json.loads((tmp_path / "workflow_matrix.json").read_text())
    assert [r["arch"] for r in sidecar["results"]] == sorted(ARCHITECTURES)


def test_main_returns_non_zero_when_a_stage_failed(tmp_path, shared_refs):
    rc = wm.main(["--archs", "deep,shallow", "--work-root", str(tmp_path),
                  "--external-refs-dir", str(shared_refs)],
                 runner=MatrixFakeRunner(rc_by_arch={"shallow": 1}))
    assert rc == 1
    text = (tmp_path / "workflow_matrix.md").read_text()
    assert "| shallow | 0.0.0.0.1.-.-.-.-.- |" in text


def test_main_refuses_a_work_root_inside_the_repository(capsys):
    """Exit 2, not a traceback. Exit 1 out of this CLI means the matrix found a
    defect in the code it drove -- it is :func:`matrix_exit_code`'s answer --
    and a path typed wrongly is not that; argparse's own status for a bad
    command line is 2, and the refusal still names the flag, the path and the
    repository it is inside.
    """
    inside = wm.repo_root_path() / "notebooks" / "matrix_scratch"
    with pytest.raises(SystemExit) as caught:
        wm.main(["--archs", "deep", "--work-root", str(inside)],
                runner=MatrixFakeRunner())
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert "--work-root" in err
    assert "inside the repository" in err
    assert not inside.exists()


def test_main_refuses_a_report_inside_the_repository(tmp_path, capsys):
    """The work-root rule is that every byte the matrix writes stays out of the
    tracked tree, and the report is one of those bytes: ``--report`` names a
    markdown table and a JSON sidecar beside it. A report path under the
    repository writes the run's own output into the tree it is measuring.
    """
    inside = wm.repo_root_path() / "notebooks" / "_matrix_report_refusal" / \
        "matrix.md"
    with pytest.raises(SystemExit) as caught:
        wm.main(["--archs", "deep", "--work-root", str(tmp_path),
                 "--external-refs-dir", str(_fake_staged_refs(
                     tmp_path / "refs")),
                 "--no-oracles", "--report", str(inside)],
                runner=MatrixFakeRunner())
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert "--report" in err
    assert "inside the repository" in err
    assert not inside.parent.exists(), (
        "the refusal must precede the report writer, which creates the "
        "directory it is given")


def test_main_refuses_an_external_refs_dir_inside_the_repository(tmp_path,
                                                                 capsys):
    """The reference directory is a WRITE target as much as the work root:
    every preflight stage drops a ``_run_log_<UTC>.json`` beside the references
    it reads, which is why ``stage_cached_inputs`` copies the cache out of the
    tree instead of pointing at it. The repository's own cached copy is the
    path most likely to be typed here, and it is the one that must be refused.
    """
    inside = wm.repo_root_path() / wm.CACHED_REFS_RELPATH
    with pytest.raises(SystemExit) as caught:
        wm.main(["--archs", "deep", "--work-root", str(tmp_path),
                 "--no-oracles", "--external-refs-dir", str(inside)],
                runner=MatrixFakeRunner())
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert "--external-refs-dir" in err
    assert "inside the repository" in err


@pytest.mark.parametrize("value", ["0", "-1"])
def test_main_refuses_a_non_positive_stage_timeout(tmp_path, capsys, value):
    """The cap is handed to every stage as ``subprocess.run``'s own
    ``timeout``, which treats a non-positive value as ALREADY expired --
    ``/bin/true`` under ``timeout=0`` raises ``TimeoutExpired`` in about two
    milliseconds. Accepted, it would kill every stage of every architecture at
    launch and fill the report with 124s, which reads as a stuck machine
    rather than as a mistyped flag.
    """
    fake = MatrixFakeRunner()
    with pytest.raises(SystemExit) as caught:
        wm.main(["--archs", "deep", "--work-root", str(tmp_path),
                 "--external-refs-dir", str(_fake_staged_refs(
                     tmp_path / "refs")),
                 "--no-oracles", "--timeout-s", value], runner=fake)
    assert caught.value.code == 2
    assert "--timeout-s" in capsys.readouterr().err
    assert fake.calls == [], "no stage may be launched under a refused cap"


def _usage_error(capsys, phrase, argv):
    """``main`` refuses a bad flag the way argparse does: exit status 2 with
    the reason on stderr, so exit 1 keeps its meaning of a defect found.

    The refusal has to come BEFORE anything is launched. A flag refused after
    the first stage has already started has cost the allocation it was meant
    to save, and the exit status alone cannot tell the two apart, so the
    runner is kept and asserted untouched.
    """
    fake = MatrixFakeRunner()
    with pytest.raises(SystemExit) as excinfo:
        wm.main(argv, runner=fake)
    assert excinfo.value.code == 2
    assert phrase in capsys.readouterr().err
    assert fake.calls == [], "no stage may be launched under a refused flag"


@pytest.mark.parametrize("value", ["0", "-1", str(wm.MAX_SHARDS + 1)])
def test_main_refuses_a_shard_count_outside_the_ceiling(tmp_path, capsys,
                                                        value):
    """``run_matrix`` refuses the same range, but reached through the CLI it
    raises: the traceback exits 1, the code reserved for a defect the matrix
    FOUND, and a job script reading that number reports a mistyped flag as a
    failing architecture. It is a refused flag, so it is exit 2."""
    _usage_error(capsys, "--shards",
                 ["--archs", "deep", "--work-root", str(tmp_path),
                  "--shards", value])


def test_main_refuses_an_unknown_architecture(tmp_path, capsys):
    _usage_error(capsys, "no_such_arch",
                 ["--archs", "no_such_arch", "--work-root", str(tmp_path)])


def test_main_passes_the_slice_and_the_oracle_switch_through(tmp_path,
                                                             shared_refs):
    from xcquinox.alec.full_benchmark_pools import HELDOUT_SPECIES_SLICE_ENV
    fake = MatrixFakeRunner()
    wm.main(["--archs", "deep", "--work-root", str(tmp_path),
             "--external-refs-dir", str(shared_refs),
             "--species-slice", "h,h2", "--no-oracles"], runner=fake)
    eval_envs = [env for argv, env in fake.calls
                 if "_eval_one_spec" in " ".join(argv)]
    assert eval_envs and all(env[HELDOUT_SPECIES_SLICE_ENV] == "h,h2"
                             for env in eval_envs)
    assert not any("-m pytest" in " ".join(argv) for argv, _e in fake.calls)


def test_main_refuses_an_archs_list_that_names_nothing(tmp_path, capsys):
    """An empty selection is a typo, not a request to run the whole registry:
    ``all`` is how the registry is asked for."""
    _usage_error(capsys, "no architecture",
                 ["--archs", " , ", "--work-root", str(tmp_path)])


def test_main_refuses_an_archs_list_that_repeats_a_name(tmp_path, capsys):
    """A repeated name is a typo in a comma-separated list, and it costs more
    than a duplicated row: both rows drive ``<work-root>/<arch>``, so they
    render one ``grid.yaml`` and one run directory between them. It is refused
    at the flag, where the name the user typed can be quoted back.
    """
    _usage_error(capsys, "more than once",
                 ["--archs", "deep,shallow,deep", "--work-root", str(tmp_path)])


def test_the_archs_flag_refuses_a_repeated_name_before_the_matrix_runs(
        tmp_path, monkeypatch, capsys):
    """Both layers refuse a repeated name with the same words -- the flag
    resolver and ``run_matrix`` -- so a test that only calls ``main`` cannot
    tell which one answered, and the flag-level rule can be deleted with the
    suite still green. It is pinned here directly: the resolver is called on
    its own, and ``main`` is driven with ``run_matrix`` replaced by a hook that
    must never be reached. The flag layer is where the name the user typed can
    still be quoted back, before a work root is resolved or a reference copy
    staged.
    """
    with pytest.raises(ValueError, match="more than once"):
        wm._resolve_archs("deep,shallow,deep")

    def _never_runs(*args, **kwargs):
        raise AssertionError(
            "run_matrix was reached: the repeated name was not refused at "
            "the flag")

    monkeypatch.setattr(wm, "run_matrix", _never_runs)
    _usage_error(capsys, "more than once",
                 ["--archs", "deep,deep", "--work-root", str(tmp_path)])


def test_main_lists_the_registry_without_running_anything(capsys):
    """``--list`` answers what ``--archs`` accepts, and needs neither a work
    root nor a staged cache to do it."""
    fake = MatrixFakeRunner()
    assert wm.main(["--list"], runner=fake) == 0
    assert capsys.readouterr().out.split() == sorted(ARCHITECTURES)
    assert fake.calls == []


def test_main_requires_a_work_root_before_it_runs(capsys):
    """``--work-root`` is dispensable only for ``--list``: a run has to put its
    run directories, logs and staged inputs somewhere."""
    with pytest.raises(SystemExit) as caught:
        wm.main(["--archs", "deep"], runner=MatrixFakeRunner())
    assert caught.value.code == 2
    assert "--work-root" in capsys.readouterr().err


def test_main_returns_the_exit_predicate_the_report_records(tmp_path,
                                                            shared_refs):
    """The process exit status and the sidecar's ``exit_code`` are one
    predicate (:func:`matrix_exit_code`) read twice. A CLI deriving its own
    would be free to disagree with the report it had just written, and the
    report is what a reader has afterwards.
    """
    for name, rc_by_arch in (("clean", {}), ("failed", {"shallow": 1})):
        root = tmp_path / name
        rc = wm.main(["--archs", "deep,shallow", "--work-root", str(root),
                      "--external-refs-dir", str(shared_refs), "--no-oracles"],
                     runner=MatrixFakeRunner(rc_by_arch=rc_by_arch))
        sidecar = json.loads((root / "workflow_matrix.json").read_text())
        assert rc == sidecar["exit_code"] == (1 if rc_by_arch else 0)


class _TimeoutRecordingRunner(MatrixFakeRunner):
    """Records the wall-clock cap every stage was launched under."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timeouts = []

    def __call__(self, argv, **kwargs):
        self.timeouts.append(kwargs.get("timeout"))
        return super().__call__(argv, **kwargs)


def test_main_caps_every_stage_at_the_requested_wall(tmp_path, shared_refs):
    """The cap is a hang detector, so it applies to every stage alike; left
    unset it is the runner's own default rather than no cap at all."""
    default = _TimeoutRecordingRunner()
    wm.main(["--archs", "deep", "--work-root", str(tmp_path / "default"),
             "--external-refs-dir", str(shared_refs), "--no-oracles"],
            runner=default)
    assert default.timeouts
    assert set(default.timeouts) == {wm.DEFAULT_STAGE_TIMEOUT_S}
    asked = _TimeoutRecordingRunner()
    wm.main(["--archs", "deep", "--work-root", str(tmp_path / "asked"),
             "--external-refs-dir", str(shared_refs), "--no-oracles",
             "--timeout-s", "7"], runner=asked)
    assert set(asked.timeouts) == {7}


def test_main_deals_the_architectures_into_the_requested_shards(tmp_path,
                                                                shared_refs):
    """Each shard gets its own pretrain-data directory: the generator writes a
    fixed filename and two concurrent datagen stages would race on it."""
    import yaml
    wm.main(["--archs", "deep,shallow", "--work-root", str(tmp_path),
             "--external-refs-dir", str(shared_refs), "--shards", "2",
             "--no-oracles"], runner=MatrixFakeRunner())
    dirs = {arch: yaml.safe_load(
        (tmp_path / arch / "grid.yaml").read_text())["pretrain"]["data_dir"]
        for arch in ("deep", "shallow")}
    assert dirs["deep"].endswith("pretrain_data_shard0")
    assert dirs["shallow"].endswith("pretrain_data_shard1")


def test_main_never_asks_for_a_submission(tmp_path, shared_refs):
    """Nothing the matrix runs reaches SLURM: the submit stage runs in its
    dry-run, which creates the run directory and renders the scripts without
    queueing them, and the matrix invokes the stage modules itself."""
    fake = MatrixFakeRunner()
    wm.main(["--archs", "deep", "--work-root", str(tmp_path),
             "--external-refs-dir", str(shared_refs), "--no-oracles"],
            runner=fake)
    submits = [argv for argv, _env in fake.calls
               if "xcquinox.alec.cluster" in " ".join(argv)
               and "submit" in argv]
    assert len(submits) == 1
    assert "--submit" not in submits[0]
    assert not any("--submit" in argv for argv, _env in fake.calls)


def test_main_reads_the_public_exit_predicate(tmp_path, shared_refs,
                                              monkeypatch):
    """The status comes from :func:`matrix_exit_code` rather than from a second
    copy of its rule inside the CLI. Two copies are free to drift, and the
    record a reader has afterwards is the report, whose sidecar carries the
    same predicate.
    """
    seen = []

    def _exit_code(results):
        seen.append([r["arch"] for r in results])
        return 7

    monkeypatch.setattr(wm, "matrix_exit_code", _exit_code)
    rc = wm.main(["--archs", "deep", "--work-root", str(tmp_path),
                  "--external-refs-dir", str(shared_refs), "--no-oracles"],
                 runner=MatrixFakeRunner())
    assert rc == 7
    assert seen and seen[-1] == ["deep"]


# ---------------------------------------------------------------------------
# The expected pretrain-data artefact follows the run's parent density
# ---------------------------------------------------------------------------

def _grid_yaml(tmp_path, **raw):
    path = tmp_path / "grid.yaml"
    import yaml
    path.write_text(yaml.safe_dump(raw))
    return path


def test_required_data_files_follows_the_parent_density(tmp_path):
    """Under ``parent_density: auto`` a meta-GGA-rung architecture pretrains on
    the SCAN-density file and a GGA-rung one on the PBE-density file; the
    artefact record has to expect the file datagen will actually write."""
    grid = _grid_yaml(tmp_path, use_polarized_correlation=True,
                      pretrain={"parent_density": "auto"})
    assert wm._required_data_files(str(grid), "deep_mgga_3x16") == [
        (True, "scan")]
    assert wm._required_data_files(str(grid), "deep_3x16") == [(True, "pbe")]


def test_required_data_files_defaults_to_the_pbe_parent(tmp_path):
    grid = _grid_yaml(tmp_path, use_polarized_correlation=False)
    assert wm._required_data_files(str(grid), "deep_mgga_3x16") == [
        (False, "pbe")]


def test_required_data_files_survives_an_unreadable_config(tmp_path):
    """An unreadable config is not fatal here -- the record falls back to the
    architecture's own polarization at the default parent and the stage logs
    carry the real failure."""
    grid = tmp_path / "grid.yaml"
    grid.write_text("{[not: yaml\n")
    assert wm._required_data_files(str(grid), "deep") == [(False, "pbe")]
    assert wm._required_data_files(str(tmp_path / "absent.yaml"),
                                   "deep") == [(False, "pbe")]


def test_artefact_record_names_the_scan_density_file(tmp_path):
    grid = _grid_yaml(tmp_path, use_polarized_correlation=True,
                      pretrain={"parent_density": "auto"})
    data_dir = tmp_path / "pretrain_data"
    art = wm._artefact_paths(
        str(tmp_path / "run"), "deep_mgga_3x16", str(data_dir),
        wm._required_data_files(str(grid), "deep_mgga_3x16"))
    assert art["pretrain_data"]["path"] == str(
        data_dir / "pretrain_data_polarized_scan.npz")


def test_artefact_record_keeps_the_historical_name_at_the_pbe_parent(tmp_path):
    data_dir = tmp_path / "pretrain_data"
    art = wm._artefact_paths(str(tmp_path / "run"), "deep", str(data_dir),
                             [(True, "pbe")])
    assert art["pretrain_data"]["path"] == str(
        data_dir / "pretrain_data_polarized.npz")
    assert art["pretrain_data"]["exists"] is False
