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


# ---------------------------------------------------------------------------
# Task 2 — Cells 1-5 builder tests
# ---------------------------------------------------------------------------


def test_cell_02_imports_includes_jax_x64_before_jnp():
    """The x64 config update must precede ``import jax.numpy as jnp``.

    Flipping ``jax_enable_x64`` after ``jnp`` has triggered tracing poisons
    cached JIT lowerings with the wrong dtype (spec Round C10-2 regression
    guard). The order is load-bearing, not cosmetic.
    """
    gen = load_generator()
    source = gen.build_cell_02_imports().source
    x64_idx = source.find('jax.config.update("jax_enable_x64", True)')
    jnp_idx = source.find("import jax.numpy as jnp")
    assert x64_idx != -1, "missing jax.config.update x64 call"
    assert jnp_idx != -1, "missing 'import jax.numpy as jnp'"
    assert x64_idx < jnp_idx, (
        "jax_enable_x64 update must appear before 'import jax.numpy as jnp' "
        f"(x64 at {x64_idx}, jnp at {jnp_idx})"
    )


def test_cell_02_imports_includes_jax_default_device_cpu():
    """Cell 2 must pin the JAX default device to CPU for reproducibility."""
    gen = load_generator()
    source = gen.build_cell_02_imports().source
    assert (
        'jax.config.update("jax_default_device", jax.devices("cpu")[0])'
        in source
    )


def test_cell_03_constants_match_spec():
    """Cell 3 must bind the exact literal forms frozen by the spec."""
    gen = load_generator()
    source = gen.build_cell_03_constants().source
    assert "BASIS = 'def2-svp'" in source
    assert "GRID_LEVEL = 1" in source
    assert (
        'H2O_COORDS = "O 0.0000 0.0000 0.1173; '
        'H 0.0000 0.7572 -0.4692; '
        'H 0.0000 -0.7572 -0.4692"'
    ) in source
    assert 'PRETRAIN_ATOMS = (("H", 1), ("He", 0), ("O", 2), ("N", 3))' in source


def test_cell_03_constants_checkpoint_base_honors_override():
    """The ``checkpoint_base`` override must flow into the cell source via repr."""
    gen = load_generator()
    source = gen.build_cell_03_constants("smoke_ckpt").source
    assert "CHECKPOINT_BASE = 'smoke_ckpt'" in source


def test_cell_05_binds_arch_colors_before_cell_9():
    """Cell 5 must bind ``arch_colors`` so Cell 9 can reference it.

    Deferring the binding to Cell 25 (the visualization section) produces
    ``NameError`` at Cell 9's pretrain loss plot on any fresh top-to-bottom
    run (spec Round B11-1 regression guard).
    """
    gen = load_generator()
    source = gen.build_cell_05_arch_names().source
    assert "arch_colors = {" in source
    assert 'plt.get_cmap("tab20")' in source


def test_main_cells_1_to_5_validate(tmp_path):
    """``main()`` must produce at least 5 cells with the expected types."""
    gen = load_generator()
    out_path = tmp_path / "step4_cells_1_5.ipynb"
    nb = gen.main(str(out_path))
    assert len(nb.cells) >= 5
    expected_types = ["markdown", "code", "code", "code", "code"]
    actual_types = [c.cell_type for c in nb.cells[:5]]
    assert actual_types == expected_types, (
        f"first 5 cell types {actual_types} != expected {expected_types}"
    )
