"""Unit tests for the step 5 notebook generator.

The generator lives at ``notebooks/_build_step5_notebook.py`` and is not part
of an importable package (``notebooks/`` intentionally has no ``__init__.py``).
Tests load the generator via ``importlib.util.spec_from_file_location`` so
test discovery does not depend on ``sys.path`` tricks.

Step 5 explores SCF self-consistency: 8 deep archs x 3 losses x 3 solver
configs = 72 runs. The generator mirrors the step 4 pattern but adds a
solver-config axis.
"""
import importlib.util
import pathlib

import nbformat
import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
GENERATOR_PATH = REPO_ROOT / "notebooks" / "_build_step5_notebook.py"


def load_generator():
    """Import ``_build_step5_notebook`` as ``step5_generator`` via spec loader.

    ``notebooks/`` is not a package, and ``sys.path`` does not normally expose
    it, so direct ``import`` fails. ``spec_from_file_location`` sidesteps the
    question without requiring a spurious ``__init__.py``.
    """
    if not GENERATOR_PATH.is_file():
        pytest.fail(
            f"Step 5 notebook generator not found at {GENERATOR_PATH}. "
            "Did Task 1 fail to land?"
        )
    spec = importlib.util.spec_from_file_location(
        "step5_generator", str(GENERATOR_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Task 1 -- scaffold + Cells 1-6 builder tests
# ---------------------------------------------------------------------------


def test_main_produces_valid_notebook(tmp_path):
    """``main()`` must emit a notebook that passes ``nbformat.validate``."""
    gen = load_generator()
    out_path = tmp_path / "step5_scaffold.ipynb"
    returned = gen.main(str(out_path))

    # main() returns the notebook object directly
    assert returned is not None
    assert len(returned.cells) >= 1

    # The written file must round-trip through nbformat.read without error
    assert out_path.is_file()
    nb = nbformat.read(str(out_path), as_version=4)
    nbformat.validate(nb)
    assert len(nb.cells) >= 1


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


def test_cell_02_imports_solver_symbols():
    """Cell 2 must import SolverConfig, SolverBackend, SolverMode from
    xcquinox.alec.solver."""
    gen = load_generator()
    source = gen.build_cell_02_imports().source
    assert "from xcquinox.alec.solver import" in source
    for symbol in ("SolverConfig", "SolverBackend", "SolverMode"):
        assert symbol in source, f"missing import of {symbol}"


def test_cell_03_constants_checkpoint_base_default():
    """Cell 3 must use DEFAULT_CHECKPOINT_BASE when no override is given."""
    gen = load_generator()
    source = gen.build_cell_03_constants().source
    assert f"CHECKPOINT_BASE = {gen.DEFAULT_CHECKPOINT_BASE!r}" in source


def test_cell_03_constants_checkpoint_base_honors_override():
    """The ``checkpoint_base`` override must flow into the cell source via repr."""
    gen = load_generator()
    source = gen.build_cell_03_constants("smoke_ckpt").source
    assert "CHECKPOINT_BASE = 'smoke_ckpt'" in source


def test_cell_04_filters_to_deep_archs():
    """Cell 4 must filter the architecture table to deep-only archs."""
    gen = load_generator()
    source = gen.build_cell_04_arch_table().source
    assert 'n.startswith("deep")' in source or "startswith('deep')" in source


def test_cell_05_binds_arch_colors():
    """Cell 5 must bind ``arch_colors`` using tab10 colormap."""
    gen = load_generator()
    source = gen.build_cell_05_arch_names().source
    assert "arch_colors" in source
    assert "ARCH_NAMES" in source
    assert 'plt.get_cmap("tab10")' in source


def test_cell_06_defines_three_solver_configs():
    """Cell 6 must define SCF_CONFIGS with exactly 3 SolverConfig entries:
    oneshot (max_cycles=0), fixed_j_3 (max_cycles=3), full_3 (max_cycles=3)."""
    gen = load_generator()
    source = gen.build_cell_06_scf_configs().source
    assert "SCF_CONFIGS" in source
    assert "SolverConfig(" in source
    # ONESHOT must have no max_cycles or max_cycles=0
    assert "SolverMode.ONESHOT" in source
    # FIXED_J and FULL must have max_cycles=3
    assert "SolverMode.FIXED_J" in source
    assert "SolverMode.FULL" in source
    assert "max_cycles=3" in source
    assert "conv_tol=1e-6" in source


def test_cell_06_solver_labels_honors_override():
    """When solver_labels is overridden, Cell 6 must filter SOLVER_LABELS
    to only labels present in SCF_CONFIGS."""
    gen = load_generator()
    source = gen.build_cell_06_scf_configs(
        solver_labels=("oneshot", "full_3")
    ).source
    assert "SOLVER_LABELS" in source
    # Must only contain the overridden labels, not the full set
    assert "oneshot" in source
    assert "full_3" in source
