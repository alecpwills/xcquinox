"""Unit tests + smoke test for the step 4 notebook generator.

The generator lives at ``notebooks/_build_step4_notebook.py`` and is not part
of an importable package (``notebooks/`` intentionally has no ``__init__.py``).
Tests load the generator via ``importlib.util.spec_from_file_location`` so
test discovery does not depend on ``sys.path`` tricks.

Per ``docs/superpowers/plans/2026-04-12-step4-notebook-implementation.md``, this
module starts with a single scaffolding test in Task 1 and grows one builder
test group per downstream task (Tasks 2 through 13).
"""
import importlib.util
import pathlib

import nbformat
import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
GENERATOR_PATH = REPO_ROOT / "notebooks" / "_build_step4_notebook.py"


def load_generator():
    """Import ``_build_step4_notebook`` as ``step4_generator`` via spec loader.

    ``notebooks/`` is not a package, and ``sys.path`` does not normally expose
    it, so direct ``import`` fails. ``spec_from_file_location`` sidesteps the
    question without requiring a spurious ``__init__.py``.
    """
    if not GENERATOR_PATH.is_file():
        pytest.fail(
            f"Step 4 notebook generator not found at {GENERATOR_PATH}. "
            "Did Task 1 fail to land?"
        )
    spec = importlib.util.spec_from_file_location(
        "step4_generator", str(GENERATOR_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_produces_valid_notebook(tmp_path):
    """``main()`` must emit a notebook that passes ``nbformat.validate``."""
    gen = load_generator()
    out_path = tmp_path / "step4_scaffold.ipynb"
    returned = gen.main(str(out_path))

    # main() returns the notebook object directly
    assert returned is not None
    assert len(returned.cells) >= 1

    # The written file must round-trip through nbformat.read without error
    assert out_path.is_file()
    nb = nbformat.read(str(out_path), as_version=4)
    nbformat.validate(nb)
    assert len(nb.cells) >= 1
