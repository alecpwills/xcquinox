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

import keyword
import os
import re
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


def test_stage_cached_inputs_names_the_missing_cache(tmp_path):
    """``notebooks/checkpoints_step7/`` is untracked (.gitignore), so a fresh
    clone, a worktree or the cluster repository has neither the references nor
    the ledger. The failure must name the directory to stage, not surface as a
    copytree traceback."""
    with pytest.raises(wm.CachedInputsMissing,
                       match="notebooks/checkpoints_step7"):
        wm.stage_cached_inputs(tmp_path / "work",
                               repo_root=tmp_path / "empty_repo")


def test_write_matrix_yaml_checks_the_ledger_on_the_shared_refs_path(tmp_path):
    """With ``external_refs_dir`` supplied the staging branch is skipped, so
    the ledger check has to stand on its own: otherwise a wrong ``repo_root``
    renders a config whose ``subset_ledger_path`` does not exist, and
    ``validate_grid_semantics`` (which does not stat that path) accepts it."""
    with pytest.raises(wm.CachedInputsMissing, match="subset_index_log.json"):
        wm.write_matrix_yaml("deep", tmp_path / "deep",
                             repo_root=tmp_path / "empty_repo",
                             external_refs_dir=tmp_path / "refs")


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
                                external_refs_dir=tmp_path / "refs")
    assert "time: '8:00:00'" in path.read_text()
    cfg = load_grid_config(str(path))
    for field in ("time", "preflight_time", "eval_time", "pretrain_time"):
        value = getattr(cfg.cluster, field)
        assert isinstance(value, str), (field, value, type(value).__name__)
        assert value == "8:00:00", (field, value)


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
    repo = str(wm.repo_root_path())
    for value in (cfg.inputs.external_refs_dir, cfg.inputs.output_root,
                  cfg.pretrain.data_dir):
        assert os.path.isabs(value), value
        assert not value.startswith(repo), value
    # The ledger is READ-ONLY (only the JSON is read; no subset.traj is
    # opened), so it is consumed in place from the repository.
    assert cfg.inputs.subset_ledger_path.startswith(repo)
    assert os.path.isfile(cfg.inputs.subset_ledger_path)


def test_write_matrix_yaml_honours_shared_directories(tmp_path):
    from xcquinox.alec.cluster.grid_config import load_grid_config
    shared_refs = tmp_path / "shared_refs"
    shared_data = tmp_path / "shared_pretrain_data"
    cfg = load_grid_config(str(wm.write_matrix_yaml(
        "deep", tmp_path / "deep", repo_root=wm.repo_root_path(),
        external_refs_dir=shared_refs, pretrain_data_dir=shared_data)))
    assert cfg.inputs.external_refs_dir == str(shared_refs)
    assert cfg.pretrain.data_dir == str(shared_data)
    # Both staged directories are created, not just the pretrain one: datagen
    # writes into data_dir and the reference precompute writes into
    # external_refs_dir, and validate_grid_semantics reports a missing
    # directory the same way for either.
    assert shared_data.is_dir(), "pretrain.data_dir must exist before datagen"
    assert shared_refs.is_dir(), (
        "inputs.external_refs_dir must exist before the reference precompute")


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
        external_refs_dir=tmp_path / "refs")))
    validate_grid_semantics(cfg, get_domain_profile(cfg.domain_profile))


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


def test_oracle_selector_selects_this_architecture_only(tmp_path):
    """Contract with the spec-3.1 oracle module: the selector must resolve to a
    non-empty set of collected tests, all of them this architecture's. Skipped
    until that module is installed, so this plan is executable on its own."""
    module = (wm.repo_root_path() / "xcquinox" / "alec" / "tests"
              / f"{wm.ORACLE_MODULE}.py")
    if not module.is_file():
        pytest.skip(f"{module} not installed yet (spec 3.1)")
    arch = "deep_cusp" if "deep_cusp" in ARCHITECTURES else sorted(ARCHITECTURES)[0]
    log = tmp_path / "collect.log"
    with log.open("w") as fh:
        rc = subprocess.run(
            [sys.executable, "-m", "pytest", wm.ORACLE_TEST_TARGET,
             "--collect-only", "-q", "-p", "no:randomly",
             "-k", wm.oracle_selector(arch)],
            cwd=str(wm.repo_root_path()), stdout=fh,
            stderr=subprocess.STDOUT, check=False).returncode
    text = log.read_text()
    assert rc == 0, text
    node_ids = [ln for ln in text.splitlines() if "::" in ln]
    assert node_ids, text
    for node in node_ids:
        # The architecture is checked as a PARAMETER of the id, not as a
        # suffix of it: stacked parametrisation puts the architecture either
        # first or last in the bracket depending on the decorator order, and a
        # collected oracle carrying no architecture parameter at all would mean
        # the expression had matched something other than a parametrisation.
        name = node.split("::")[-1]
        assert "[" in name and name.endswith("]"), node
        params = name[name.index("[") + 1:name.rindex("]")].split("-")
        assert arch in params, node
