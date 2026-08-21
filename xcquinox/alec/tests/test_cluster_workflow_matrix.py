"""Tests for the per-architecture workflow matrix
(``xcquinox.alec.cluster.workflow_matrix``).

The matrix drives the harness stage sequence -- submit (dry-run), datagen,
pretrain, certificate, preflight, train, eval, validate_run -- as plain Python
at a tiny def2-svp / grid-level-1 identity for every registered architecture
(SPEC_pretrain_fidelity_program.md 3.4). These tests exercise the runner with a
FAKE runner in place of ``subprocess.run``: no stage subprocess is started, so
the whole file runs in seconds. The single real end-to-end pass is the
``slow``-marked smoke at the bottom.
"""
from __future__ import annotations

import json
import os
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


def test_template_carries_no_address_and_no_account():
    """The template is rendered in dry-run only and never submitted, so it
    carries no mail address and no allocation; a shipped example with a real
    address would mail a person from anybody's copy."""
    text = wm.template_path().read_text()
    assert "@" not in text.split("cluster:", 1)[1]
    assert '\n  mail_user: ""\n' in text
    assert '\n  account: ""\n' in text


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


def test_stage_cached_inputs_is_idempotent(tmp_path):
    first = wm.stage_cached_inputs(tmp_path, repo_root=wm.repo_root_path())
    probe = Path(first["external_refs_dir"]) / "_matrix_probe.txt"
    probe.write_text("kept")
    second = wm.stage_cached_inputs(tmp_path, repo_root=wm.repo_root_path())
    assert second == first
    assert probe.read_text() == "kept"


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
        external_refs_dir=tmp_path / "refs")))
    validate_grid_semantics(cfg, get_domain_profile(cfg.domain_profile))
