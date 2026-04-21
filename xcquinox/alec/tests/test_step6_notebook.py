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


def test_main_produces_20_cells_after_phase7_1():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    assert len(nb.cells) == 20


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


# ---------------------------------------------------------------------------
# Phase 6.1 tests: data md + Chakravorty dict (cells 10-11)
# ---------------------------------------------------------------------------


def test_chakravorty_cell_has_all_five_atoms():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[10].source) if isinstance(nb.cells[10].source, list) else nb.cells[10].source
    assert "ATOMIC_ENERGIES_CHAKRAVORTY" in src
    for atom, value in (("H", "-0.5"), ("C", "-37.845"),
                        ("N", "-54.5892"), ("O", "-75.0673"), ("F", "-99.7339")):
        assert f'"{atom}"' in src, f"missing atom key {atom}"
        assert value in src, f"missing value {value}"


# ---------------------------------------------------------------------------
# Phase 6.2 tests: H2O data cell (cell 12)
# ---------------------------------------------------------------------------


def test_h2o_cell_uses_w411_geometry_and_hardened_oep():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[11].source) if isinstance(nb.cells[11].source, list) else nb.cells[11].source
    # W4-11 canonical H2O geometry
    assert "0.117790" in src and "0.755453" in src and "-0.471161" in src
    # W4-11 AE reference
    assert "232.974" in src
    # Hardened OEP primary + fallback
    assert "def2-tzvp-jkfit" in src
    assert "max_iter=500" in src and "max_iter=1000" in src
    assert "regularization=1e-3" in src and "regularization=1e-2" in src
    # Real save_vxc_ref signature (OEPResult first, not vxc_matrix)
    assert "save_vxc_ref(_oep" in src
    assert "method=\"ccsd\"" in src or "method='ccsd'" in src


# ---------------------------------------------------------------------------
# Phase 6.3 tests: C2H2 data cell (cell 13)
# ---------------------------------------------------------------------------


def test_c2h2_cell_uses_w411_geometry():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[12].source) if isinstance(nb.cells[12].source, list) else nb.cells[12].source
    assert "C2H2" in src
    assert "1.666650" in src and "0.603250" in src
    assert "405.525" in src
    assert "def2-tzvp-jkfit" in src
    assert "save_vxc_ref(_oep" in src


# ---------------------------------------------------------------------------
# Phase 6.4 tests: atoms cell (cell 14)
# ---------------------------------------------------------------------------


def test_atoms_cell_includes_carbon_uks():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[13].source) if isinstance(nb.cells[13].source, list) else nb.cells[13].source
    # H, O, C present
    assert "(\"H\", \"H 0 0 0\", 1)" in src or "('H', 'H 0 0 0', 1)" in src
    assert "(\"O\", \"O 0 0 0\", 2)" in src or "('O', 'O 0 0 0', 2)" in src
    assert "(\"C\", \"C 0 0 0\", 2)" in src or "('C', 'C 0 0 0', 2)" in src
    # UKS / UHF / UCCSD branches
    assert "UKS" in src and "UHF" in src and "UCCSD" in src
    # spin-resolved stacked DM
    assert "np.stack" in src
    # Chakravorty consumption
    assert "ATOMIC_ENERGIES_CHAKRAVORTY" in src


# ---------------------------------------------------------------------------
# Phase 6.5 + 6.6 tests: PBE-anchor sample (cell 15) + MoleculeSpec/precompute (cell 16)
# ---------------------------------------------------------------------------


def test_pbe_anchor_build_cell():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[14].source) if isinstance(nb.cells[14].source, list) else nb.cells[14].source
    assert "build_pbe_anchor_sample" in src
    assert "PBE_ANCHOR_N_POINTS" in src
    assert "PBE_ANCHOR_SEED" in src


def test_mol_specs_and_precompute_cell():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[15].source) if isinstance(nb.cells[15].source, list) else nb.cells[15].source
    # Five entities
    for _ent in ("H2O_spec", "C2H2_spec", "H_spec", "O_spec", "C_spec"):
        assert _ent in src, f"missing {_ent}"
    # precompute iterates per-spec (not list)
    assert "precompute_fixed_density_data" in src
    assert "materialize_descriptors" in src
    # mol_data_by_name dict
    assert "mol_data_by_name" in src


# ---------------------------------------------------------------------------
# Phase 7.1 tests: training section header + three training spec groups
# (cells 17-20)
# ---------------------------------------------------------------------------


def test_three_training_groups_present():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src_all = "\n".join(
        ("".join(c.source) if isinstance(c.source, list) else c.source)
        for c in nb.cells
    )
    for tok in ("_specs_group1", "_specs_group2", "_specs_group3",
                "L1_B", "L2_C_anchor", "L3_balanced_vxc", "L4_balanced_vxc_anchor",
                "LossNormConfig", "pbe_anchor_weight", "pbe_anchor_sample",
                "ATOMIC_ENERGIES_CHAKRAVORTY", "SOLVER_CONFIGS",
                "from_dicts"):
        assert tok in src_all, f"missing {tok}"


def test_group3_uses_long_steps():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[19].source) if isinstance(nb.cells[19].source, list) else nb.cells[19].source
    assert "TRAIN_N_STEPS_LONG" in src
    assert "group3_dir" in src
