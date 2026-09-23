"""Hygiene of the notebooks directory and of what the notebooks write.

The training notebooks write checkpoints (trained networks, logs, extracted outputs) into
directories named ``checkpoints*``. The first test pins that no
tree of that name is tracked anywhere, so a notebook's outputs cannot enter the tree.

A notebook is kept under ``notebooks/`` only as the product of a
tracked builder, written as the builder writes it (no outputs, no execution counts), and named
in ``notebooks/README.md``; the other tests pin that rule in both directions. Every test
reads the index with ``git ls-files``, so it sees a staged removal or addition, not only the
last commit.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

#: every notebook the directory keeps and the builder that writes it; a notebook without a
#: builder is not tracked, and a builder without a tracked product is a defect
BUILDER_PRODUCTS = {
    "notebooks/_build_gga_training_pipeline_notebook.py": "notebooks/gga_training_pipeline.ipynb",
    "notebooks/_build_gga_training_scf_solvers_notebook.py": "notebooks/gga_training_scf_solvers.ipynb",
    "notebooks/_build_gga_training_anchor_transfer_notebook.py": "notebooks/gga_training_anchor_transfer.ipynb",
    "notebooks/_build_gga_training_dfs_subsets_notebook.py": "notebooks/gga_training_dfs_subsets.ipynb",
    "notebooks/_build_subset_generation_notebook.py": "notebooks/gga_subset_generation.ipynb",
    "notebooks/dfs_selfconsistent_density/_build_notebook.py":
        "notebooks/dfs_selfconsistent_density/train_dfs_density.ipynb",
}


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=_ROOT, capture_output=True,
                         check=True).stdout.decode("utf-8", errors="replace")
    return [p for p in out.split("\0") if p]


def _tracked_notebooks() -> list[str]:
    """The tracked notebooks under ``notebooks/``."""
    return sorted(p for p in _tracked()
                  if p.startswith("notebooks/") and p.endswith(".ipynb"))


def stored_outputs(notebook: dict) -> int:
    """The number of code cells that carry outputs or an execution count."""
    return sum(1 for cell in notebook.get("cells", [])
               if cell.get("cell_type") == "code"
               and (cell.get("outputs") or cell.get("execution_count") is not None))


def test_output_rule_fires_on_a_cell_with_outputs_or_a_count():
    """A code cell with an output, or with an execution count alone, fires; a clean code cell
    and a markdown cell do not."""
    clean = {"cells": [{"cell_type": "code", "outputs": [], "execution_count": None,
                        "source": "x = 1"},
                       {"cell_type": "markdown", "source": "text"}]}
    assert stored_outputs(clean) == 0
    run = {"cells": [{"cell_type": "code", "execution_count": 3, "source": "print(1)",
                      "outputs": [{"output_type": "stream", "name": "stdout", "text": "1"}]}]}
    assert stored_outputs(run) == 1
    counted = {"cells": [{"cell_type": "code", "outputs": [], "execution_count": 7,
                          "source": "x = 1"}]}
    assert stored_outputs(counted) == 1


def test_no_tracked_notebook_stores_outputs():
    """Every tracked notebook is stored as its builder writes it: no outputs, no counts."""
    offenders = []
    for path in _tracked_notebooks():
        with open(_ROOT / path, encoding="utf-8") as fh:
            n = stored_outputs(json.load(fh))
        if n:
            offenders.append((path, n))
    assert offenders == [], (
        f"{len(offenders)} tracked notebooks store outputs (path, cells): {offenders[:3]}")


def test_every_tracked_notebook_is_a_builder_product():
    """Every tracked notebook is the product of a builder in the table, and every builder and
    every product in the table is tracked."""
    tracked = set(_tracked())
    products = set(BUILDER_PRODUCTS.values())
    orphans = [p for p in _tracked_notebooks() if p not in products]
    assert orphans == [], (
        f"{len(orphans)} tracked notebooks have no builder, for example {orphans[:3]}")
    missing = [p for p in list(BUILDER_PRODUCTS) + sorted(products) if p not in tracked]
    assert missing == [], f"builders or products in the table but not tracked: {missing}"


def _load_builder(builder: str):
    """The builder module loaded from its file (``notebooks/`` is not a package; the subset
    builder imports the step-7 builder by module name, so the directory is put on the path
    for the load)."""
    path = _ROOT / builder
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


@pytest.mark.parametrize("builder,product", sorted(BUILDER_PRODUCTS.items()))
def test_every_tracked_notebook_is_its_builders_output(builder, product, tmp_path):
    """The tracked notebook is byte for byte what its builder writes: the builder is loaded
    from its file and writes into a temporary path, and the bytes are compared. A builder
    whose cell ids or metadata varied from run to run could not pass this, nor could a
    notebook edited or executed by hand after its generation."""
    out = tmp_path / os.path.basename(product)
    _load_builder(builder).main(output_path=str(out))
    assert out.read_bytes() == (_ROOT / product).read_bytes(), (
        f"{product} is not what {builder} writes; regenerate it with the builder")


def test_every_tracked_notebook_is_named_in_the_readme():
    """``notebooks/README.md`` names every tracked notebook by its file name."""
    readme = _ROOT / "notebooks" / "README.md"
    assert readme.is_file(), "notebooks/README.md is missing"
    text = readme.read_text(encoding="utf-8")
    unnamed = [p for p in _tracked_notebooks() if os.path.basename(p) not in text]
    assert unnamed == [], f"tracked notebooks the README does not name: {unnamed}"


def test_no_checkpoint_tree_is_tracked():
    """No tracked path has a directory component that starts with ``checkpoints``."""
    offenders = sorted(p for p in _tracked()
                       if any(part.startswith("checkpoints") for part in p.split("/")[:-1]))
    assert offenders == [], (
        f"{len(offenders)} tracked files sit under a checkpoints directory, for example "
        f"{offenders[:3]}: the notebooks' outputs are ignored, never committed")


def test_no_tracked_notebook_anywhere_stores_outputs():
    """The output rule of the notebooks directory holds for every tracked notebook wherever
    it sits (the G2/97 parsing notebook is package data under xcquinox/data): none stores
    outputs or execution counts."""
    offenders = []
    for path in sorted(p for p in _tracked() if p.endswith(".ipynb")):
        with open(_ROOT / path, encoding="utf-8") as fh:
            n = stored_outputs(json.load(fh))
        if n:
            offenders.append((path, n))
    assert offenders == [], (
        f"{len(offenders)} tracked notebooks store outputs (path, cells): {offenders[:3]}")
