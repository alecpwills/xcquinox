"""Argument tests for ``scripts/generate_polarized_pretrain_data.py``.

The script is the one-off staging entry point for a run's polarized
pretrain-data file. Its defaults ARE the identity of the file it writes, so
they are pinned here; the generator itself is faked, since what is under test
is which arguments reach it, not the rows it would compute.
"""
import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

g = importlib.import_module("generate_polarized_pretrain_data")


@pytest.fixture
def calls(monkeypatch):
    """Capture the generator call the script makes."""
    seen = []

    def _fake(out_dir, **kw):
        seen.append((out_dir, kw))
        return f"{out_dir}/pretrain_data_polarized.npz"

    monkeypatch.setattr(g, "generate_pretrain_data_npz", _fake)
    return seen


def test_default_grid_level_is_the_production_level(calls, tmp_path):
    """The default atom set contains O, a spatially degenerate free atom whose
    rows below grid level 3 are one arbitrary member of the P-term manifold
    rather than a reproducible quantity. The script's default is therefore the
    production level, not the library's historical 1, so running it with no
    grid argument writes a file whose manifest identity it actually has."""
    g.main(["--out-dir", str(tmp_path)])
    assert len(calls) == 1
    out_dir, kw = calls[0]
    assert out_dir == str(tmp_path)
    assert kw["grid_level"] == 3
    assert kw["atoms"] == g.DEFAULT_PRETRAIN_ATOMS
    assert kw["polarized"] is True
    assert kw["descriptors"] is True
    assert kw["density_fit"] is False
    assert kw["allow_irreproducible_degenerate"] is False


def test_grid_level_help_states_the_production_level():
    parser = g._build_parser()
    action = {a.dest: a for a in parser._actions}["grid_level"]
    assert action.default == 3
    assert "production" in (action.help or "")


def test_the_waiver_flag_is_passed_through(calls, tmp_path):
    """An unreproducible build is possible but never accidental: the flag the
    generator's refusal names is offered here so the caller can state it."""
    g.main(["--out-dir", str(tmp_path), "--grid-level", "1",
            "--allow-irreproducible-degenerate"])
    assert calls[0][1]["grid_level"] == 1
    assert calls[0][1]["allow_irreproducible_degenerate"] is True


def test_explicit_arguments_reach_the_generator(calls, tmp_path):
    g.main(["--out-dir", str(tmp_path), "--basis", "def2-tzvp",
            "--grid-level", "2", "--atoms", "He:0,H:1", "--no-descriptors",
            "--density-fit"])
    _, kw = calls[0]
    assert kw["basis"] == "def2-tzvp"
    assert kw["grid_level"] == 2
    assert kw["atoms"] == (("He", 0), ("H", 1))
    assert kw["descriptors"] is False
    assert kw["density_fit"] is True
