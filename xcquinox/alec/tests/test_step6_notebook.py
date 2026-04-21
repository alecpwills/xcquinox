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
