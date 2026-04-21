"""Structural tests for the step 6 notebook generator."""
import ast
import hashlib
import importlib.util
from pathlib import Path

import nbformat
import pytest

_REPO = Path(__file__).resolve().parents[3]
_GENERATOR_PATH = _REPO / "notebooks" / "_build_step6_notebook.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("_build_step6_notebook", _GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_imports_cleanly():
    gen = load_generator()
    assert gen is not None


def test_generator_exposes_main():
    gen = load_generator()
    assert callable(getattr(gen, "main", None))


def test_generator_has_default_constants():
    gen = load_generator()
    assert gen.DEFAULT_ARCH_NAMES == ("deep_combined", "deep_combined_attn")
    assert gen.DEFAULT_LOSS_NAMES == (
        "L1_B", "L2_C_anchor", "L3_balanced_vxc", "L4_balanced_vxc_anchor",
    )
    assert gen.DEFAULT_SOLVER_LABELS == ("oneshot", "fixed_j_3", "full_3")
    assert gen.DEFAULT_CHECKPOINT_BASE == "checkpoints_step6"


def test_cell_01_is_markdown_title():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6_smoke.ipynb")
    assert nb.cells[0].cell_type == "markdown"
    src = "".join(nb.cells[0].source) if isinstance(nb.cells[0].source, list) else nb.cells[0].source
    assert "Step 6" in src
    assert "C2H2" in src or "C₂H₂" in src


def test_cell_03_contains_step6_constants():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6_smoke.ipynb")
    src = "".join(nb.cells[2].source) if isinstance(nb.cells[2].source, list) else nb.cells[2].source
    for tok in ("CHECKPOINT_BASE", "PRETRAIN_SKIP_IF_EXISTS",
                "PRETRAIN_N_STEPS", "TRAIN_N_STEPS_SHORT", "TRAIN_N_STEPS_LONG",
                "PBE_ANCHOR_WEIGHT", "PBE_ANCHOR_N_POINTS", "PBE_ANCHOR_SEED"):
        assert tok in src, f"expected {tok!r} in constants cell"


def test_main_output_is_deterministic_byte_identical():
    import hashlib
    gen = load_generator()
    gen.main(output_path="/tmp/_step6_det_1.ipynb")
    gen.main(output_path="/tmp/_step6_det_2.ipynb")
    h1 = hashlib.sha256(open("/tmp/_step6_det_1.ipynb", "rb").read()).hexdigest()
    h2 = hashlib.sha256(open("/tmp/_step6_det_2.ipynb", "rb").read()).hexdigest()
    assert h1 == h2


# ---------------------------------------------------------------------------
# Phase 5.1 tests: pretrain cells 6-9
# ---------------------------------------------------------------------------


def test_main_produces_9_cells_after_phase5():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    assert len(nb.cells) == 9


def test_pretrain_loop_cell_uses_skip_flag():
    """Cell 8 (pretrain loop) respects PRETRAIN_SKIP_IF_EXISTS + checks both xnet+cnet ckpts."""
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[7].source) if isinstance(nb.cells[7].source, list) else nb.cells[7].source
    assert "PRETRAIN_SKIP_IF_EXISTS" in src
    assert "xnet.eqx" in src and "cnet.eqx" in src
    assert "PretrainSpec" in src
    assert "run_pretrain" in src
    assert "PRETRAIN_N_STEPS" in src


def test_pretrain_data_gen_cell_uses_pretrain_atoms():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[6].source) if isinstance(nb.cells[6].source, list) else nb.cells[6].source
    assert "PRETRAIN_ATOMS" in src
    assert "pretrain_data.npz" in src


def test_constants_cell_has_pretrain_atoms_and_weighting():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[2].source) if isinstance(nb.cells[2].source, list) else nb.cells[2].source
    assert "PRETRAIN_ATOMS" in src
    assert 'PRETRAIN_LOSS_WEIGHTING' in src
    assert "unweighted" in src  # default value


def test_imports_cell_has_tqdm_and_features():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[1].source) if isinstance(nb.cells[1].source, list) else nb.cells[1].source
    assert "from tqdm.auto import tqdm" in src
    assert "xcquinox.features" in src
    assert "import equinox as eqx" in src
    # JAX config
    assert "jax_enable_x64" in src
