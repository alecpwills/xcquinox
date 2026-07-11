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
        "L5_gradnorm_vxc",
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


def test_main_produces_44_cells_after_balancing_effect_insertion():
    """Cell count is 44 (43 after the 2026-04-24 baseline_evals insertion
    + 1 balancing_effect cell 28b inserted between anchor_effect (28) and
    transfer_md (now 30) in the 2026-04-28 L5_gradnorm_vxc pass)."""
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    assert len(nb.cells) == 44


def test_every_code_cell_is_ast_parseable():
    import ast
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6_ast.ipynb")
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
        if not src.strip():
            continue
        try:
            ast.parse(src)
        except SyntaxError as e:
            pytest.fail(f"Cell {i} fails to parse: {e}")


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


def test_h2o_cell_uses_w411_geometry_and_two_tier_oep_cascade():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[11].source) if isinstance(nb.cells[11].source, list) else nb.cells[11].source
    # W4-11 canonical H2O geometry
    assert "0.117790" in src and "0.755453" in src and "-0.471161" in src
    # W4-11 AE reference
    assert "232.974" in src
    # Re-tuned OEP cascade (2026-04-28) for the Wu-Yang displacement-form
    # inversion (commit 4b2b58ba9). The pre-rewrite cascade
    # (tzvp-jkfit/reg=1e-5/conv_tol=2e-3 primary, reg=1e-6 fallback)
    # exits L-BFGS-B prematurely with density_error ~2-7e-2 under the
    # displacement form. Measurement on H2O/def2-svp/grid_level=1 with
    # max_iter=500: aux=def2-{svp,tzvp}-jkfit at reg=1e-4 both reach
    # density_error ~1.17e-3 and continue to improve; higher reg values
    # produce worse outcomes. Primary: small aux for speed; fallback:
    # tzvp aux + double iter budget for genuinely harder cases.
    assert "def2-svp-jkfit" in src
    assert "def2-tzvp-jkfit" in src
    # Primary: svp-jkfit, max_iter=500, conv_tol=2e-3, reg=1e-4.
    assert "max_iter=500" in src and "conv_tol=2e-3" in src
    assert "regularization=1e-4" in src
    # Fallback: tzvp-jkfit + max_iter=1000.
    assert "max_iter=1000" in src
    # The pre-rewrite cascade settings must NOT remain.
    assert "regularization=1e-5" not in src
    assert "regularization=1e-6" not in src
    assert "conv_tol=1e-6" not in src
    # Graceful retry: if vxc_ref missing from cached .npz, retry OEP without
    # redoing CCSD.
    assert "_npz_has_vxc_ref" in src
    # Real save_vxc_ref signature (OEPResult first, not vxc_matrix)
    assert "save_vxc_ref(_oep" in src
    assert "method=\"ccsd\"" in src or "method='ccsd'" in src


# ---------------------------------------------------------------------------
# Phase 6.3 tests: C2H2 data cell (cell 13)
# ---------------------------------------------------------------------------


def test_c2h2_cell_uses_w411_geometry_and_two_tier_oep_cascade():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[12].source) if isinstance(nb.cells[12].source, list) else nb.cells[12].source
    assert "C2H2" in src
    assert "1.666650" in src and "0.603250" in src
    assert "405.525" in src
    # Same re-tuned cascade as H2O (svp-jkfit primary, tzvp-jkfit fallback,
    # both at reg=1e-4 conv_tol=2e-3). See test_h2o_cell_... for measurement
    # rationale post-displacement-form OEP rewrite (4b2b58ba9).
    assert "def2-svp-jkfit" in src
    assert "def2-tzvp-jkfit" in src
    assert "max_iter=500" in src and "conv_tol=2e-3" in src
    assert "regularization=1e-4" in src
    assert "max_iter=1000" in src
    assert "regularization=1e-5" not in src
    assert "regularization=1e-6" not in src
    # Graceful retry via shared _npz_has_vxc_ref helper from cell 12.
    assert "_npz_has_vxc_ref" in src
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


def test_constants_cell_uses_case_study_basis_and_grid():
    """Case-study constraint: step6 runs on a modest GPU. Must use
    BASIS='def2-svp' and GRID_LEVEL=1 (step5 settings); the heavier
    ('def2-tzvp', 3) combination OOMed at the first training step on
    an 8 GB GPU."""
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[2].source) if isinstance(nb.cells[2].source, list) else nb.cells[2].source
    assert 'BASIS                    = "def2-svp"' in src, (
        "expected BASIS='def2-svp' in constants cell"
    )
    assert "GRID_LEVEL               = 1" in src, (
        "expected GRID_LEVEL=1 in constants cell"
    )
    # Heavier settings must not reappear.
    assert 'BASIS                    = "def2-tzvp"' not in src
    assert "GRID_LEVEL               = 3" not in src


def test_group3_uses_long_steps():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[19].source) if isinstance(nb.cells[19].source, list) else nb.cells[19].source
    assert "TRAIN_N_STEPS_LONG" in src
    assert "group3_dir" in src


def test_constants_cell_has_updated_step_counts():
    """Constants cell must declare PRETRAIN_N_STEPS=1000, SHORT=100, LONG=250.
    Updated 2026-04-28 from 200 / 45 / 125 (the prior values were too short
    for the V_xc residual to drop visibly under L3/L5 gradient signal).
    """
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[2].source) if isinstance(nb.cells[2].source, list) else nb.cells[2].source
    assert "PRETRAIN_N_STEPS         = 1000" in src
    assert "TRAIN_N_STEPS_SHORT      = 100" in src
    assert "TRAIN_N_STEPS_LONG       = 250" in src


def test_each_group_has_l5_gradnorm_vxc_branch():
    """Each of the three group spec-construction cells must include the
    L5_gradnorm_vxc branch, mapping it to ``B_atomization_plus_dm`` +
    ``GradNormConfig(alpha=1.5)``.

    GradNormConfig (Chen et al. 2018, ICML) replaces L3's LossNormConfig
    so the V_xc loss gradient magnitude tracks AE / DM / atomic_reg
    DURING training, not just at step 0. The static-weighting +
    LossNorm-at-step-0 strategies (L3, L4) leave V_xc essentially flat
    once AE drops 5+ orders within the first ~50 steps; GradNorm's
    dynamic per-task weights restore balance and let V_xc keep moving.

    Pins the contract that all three groups carry the L5 branch and
    each uses the same alpha.
    """
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    for cell_idx, group_name in ((17, "group1"), (18, "group2"), (19, "group3")):
        src = nb.cells[cell_idx].source
        if isinstance(src, list):
            src = "".join(src)
        assert "L5_gradnorm_vxc" in src, (
            f"cell {cell_idx} ({group_name}) missing L5_gradnorm_vxc branch"
        )
        assert "GradNormConfig(alpha=1.5)" in src, (
            f"cell {cell_idx} ({group_name}) L5 branch must use "
            f"GradNormConfig(alpha=1.5)"
        )
        # And L5 carries the same loss + V_xc weighting as L3, only the
        # balancing strategy differs.
        assert 'vxc_weight": 0.01' in src or "vxc_weight\\\": 0.01" in src or "vxc_weight': 0.01" in src or "'vxc_weight': 0.01" in src or 'vxc_weight=0.01' in src or '"vxc_weight": 0.01' in src


def _assert_atom_target(src: str, symbol: str) -> None:
    import re
    pattern = rf'"{symbol}":\s+ATOMIC_ENERGIES_CHAKRAVORTY\["{symbol}"\]'
    assert re.search(pattern, src), (
        f"expected {symbol!r} atom placeholder target in source"
    )


def test_group1_targets_include_atom_placeholders():
    """TrainingSpec.validate() requires an entry in ``targets`` for every
    molecule in ``molecules``. Group 1's molecules tuple is
    ``(H2O_spec, H_spec, O_spec)`` so its targets dict must include keys
    for "H" and "O" too (matches step5's idiom; values are placeholders
    since atom entries are never dereferenced at training time)."""
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[17].source) if isinstance(nb.cells[17].source, list) else nb.cells[17].source
    assert "_targets_group1" in src
    _assert_atom_target(src, "H")
    _assert_atom_target(src, "O")


def test_group2_targets_include_atom_placeholders():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[18].source) if isinstance(nb.cells[18].source, list) else nb.cells[18].source
    assert "_targets_group2" in src
    _assert_atom_target(src, "H")
    _assert_atom_target(src, "O")
    _assert_atom_target(src, "C")


def test_group3_targets_include_atom_placeholders():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[19].source) if isinstance(nb.cells[19].source, list) else nb.cells[19].source
    assert "_targets_group3" in src
    _assert_atom_target(src, "H")
    _assert_atom_target(src, "O")
    _assert_atom_target(src, "C")


# ---------------------------------------------------------------------------
# Phase 7.2 tests: subprocess-isolated training loop (cell 21)
# ---------------------------------------------------------------------------


def test_training_loop_cell_uses_subprocess_pattern():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[20].source) if isinstance(nb.cells[20].source, list) else nb.cells[20].source
    # Subprocess isolation pattern
    assert "subprocess" in src
    assert "_train_one_spec" in src
    assert "TRAIN_SKIP_IF_EXISTS" in src
    # Combines all three groups
    assert "_specs_group1" in src
    assert "_specs_group2" in src
    assert "_specs_group3" in src
    # Post-save teardown tolerance
    assert "model.eqx" in src
    # Cache/GC between specs
    assert "jax.clear_caches" in src


def test_imports_cell_defines_dataframe_save_load_helpers():
    """The notebook uses `df.to_parquet`/`pd.read_parquet` to persist eval
    and transfer DataFrames. pyarrow / fastparquet may not be installed
    on every environment; a hard ImportError there breaks the whole
    notebook. Imports cell must expose `_df_save` / `_df_load` helpers
    that try parquet and fall back to CSV (stdlib-only)."""
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[1].source) if isinstance(nb.cells[1].source, list) else nb.cells[1].source
    assert "_df_save" in src
    assert "_df_load" in src
    assert "to_csv" in src and "read_csv" in src


def test_eval_df_cell_uses_robust_save_load():
    """Cell 26 (builds eval_df) must route through _df_save / _df_load so a
    missing pyarrow install doesn't kill the cell at the write step."""
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    target = None
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        src = "".join(c.source) if isinstance(c.source, list) else c.source
        if "eval_df = pd.DataFrame" in src:
            target = src
            break
    assert target is not None, "eval_df builder cell not found"
    assert "_df_save(eval_df" in target or "_df_save(\n    eval_df" in target
    assert ".to_parquet(" not in target, (
        "cell must use _df_save helper, not raw df.to_parquet (no pyarrow)"
    )


def test_imports_cell_defines_blas_thread_cap_helper():
    """OEP (Wu-Yang inversion) is CPU-bound via PySCF/BLAS. When the same
    kernel already imported JAX, the two OMP pools fight over cores and
    throughput drops ~5-10x. Imports cell must expose a scoped helper
    `_capped_blas_threads(n)` that caps PySCF threads for the duration
    of the OEP call."""
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[1].source) if isinstance(nb.cells[1].source, list) else nb.cells[1].source
    assert "_capped_blas_threads" in src
    # Must use pyscf.lib.num_threads (stdlib-only: no new deps like
    # threadpoolctl).
    assert "pyscf" in src and "num_threads" in src


def test_h2o_oep_cell_uses_blas_thread_cap():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[11].source) if isinstance(nb.cells[11].source, list) else nb.cells[11].source
    assert "_capped_blas_threads" in src


def test_c2h2_oep_cell_uses_blas_thread_cap():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[12].source) if isinstance(nb.cells[12].source, list) else nb.cells[12].source
    assert "_capped_blas_threads" in src


def _cap_wraps_ccsd(src: str) -> bool:
    """Return True if `_capped_blas_threads(...)` appears textually before the
    first CCSD call (cc.CCSD). Ensures the CCSD step is run under the same
    thread cap as the OEP cascade -- without this, CCSD on C2H2 takes 70+
    seconds contending with JAX's thread pool."""
    cap_idx = src.find("_capped_blas_threads(")
    ccsd_idx = src.find("cc.CCSD(")
    if cap_idx == -1 or ccsd_idx == -1:
        return False
    return cap_idx < ccsd_idx


def test_h2o_cell_thread_cap_wraps_ccsd():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[11].source) if isinstance(nb.cells[11].source, list) else nb.cells[11].source
    assert _cap_wraps_ccsd(src), (
        "thread cap must open BEFORE cc.CCSD() in the H2O cell so CCSD is "
        "not run uncapped when JAX is already loaded in the kernel"
    )


def test_c2h2_cell_thread_cap_wraps_ccsd():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[12].source) if isinstance(nb.cells[12].source, list) else nb.cells[12].source
    assert _cap_wraps_ccsd(src), (
        "thread cap must open BEFORE cc.CCSD() in the C2H2 cell"
    )


def test_h2o_oep_cell_shows_tqdm_progress():
    """User asked for visibility into long OEP runs; the cell must attach a
    tqdm-backed progress_callback to each tier invocation so dozens-of-
    minutes C2H2 inversions are no longer silent."""
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[11].source) if isinstance(nb.cells[11].source, list) else nb.cells[11].source
    assert "progress_callback=" in src
    # A tqdm bar is used to render progress.
    assert "tqdm(" in src
    # The tier label identifies the molecule so bars are distinguishable.
    assert "OEP H2O" in src or 'f"OEP H2O' in src or "H2O" in src
    # Bar must be closed explicitly (try/finally) so a failed tier doesn't
    # leak a stale bar into the fallback tier's output.
    assert ".close()" in src


def test_c2h2_oep_cell_shows_tqdm_progress():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[12].source) if isinstance(nb.cells[12].source, list) else nb.cells[12].source
    assert "progress_callback=" in src
    assert "tqdm(" in src
    assert "OEP C2H2" in src or "C2H2" in src
    assert ".close()" in src


def test_plot_cells_use_canonical_value_name():
    """Regression: plot cells (27, 28, 31, 32) must filter eval_df / transfer
    dfs on ``AE_error_kcalmol`` (the canonical key emitted by
    ``xcquinox.alec.evaluation.AtomizationEnergyMetric``), not
    ``abs_ae_error`` (which does not exist). Using the wrong key silently
    returns empty slices, producing bar charts with visible axes but no
    bars. Audit 2026-04-24."""
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    all_src = "\n".join(
        ("".join(c.source) if isinstance(c.source, list) else c.source)
        for c in nb.cells
    )
    assert "abs_ae_error" not in all_src, (
        "plot cells must use 'AE_error_kcalmol' (canonical emit from "
        "AtomizationEnergyMetric), not the phantom 'abs_ae_error'."
    )
    # The canonical name should appear at least in the 4 plot cells that
    # previously mis-referenced it.
    assert all_src.count("AE_error_kcalmol") >= 4


def test_training_loop_cell_cpu_retry_passes_jax_platforms_env():
    """The CPU retry path MUST pass JAX_PLATFORMS=cpu in the subprocess env.
    --device=cpu alone is insufficient because `python -m
    xcquinox.alec._train_one_spec` transitively imports jax via the package's
    __init__ before the CLI flag can act. Regression test for the bug where
    the retry subprocess still OOMed on GPU despite --device=cpu."""
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[20].source) if isinstance(nb.cells[20].source, list) else nb.cells[20].source
    # The env override must be present when device='cpu'.
    assert "JAX_PLATFORMS" in src and "'cpu'" in src
    # And the env dict must be passed to subprocess.Popen.
    assert "env=env" in src


def test_training_loop_cell_has_gpu_oom_cpu_retry():
    """Small-GPU support: when a subprocess exits non-zero with no
    model.eqx saved AND the captured output matches a GPU-OOM signature,
    cell 17 must re-invoke the worker with --device=cpu before raising."""
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[20].source) if isinstance(nb.cells[20].source, list) else nb.cells[20].source
    # OOM detection signatures
    assert "RESOURCE_EXHAUSTED" in src
    # Retry on CPU via the worker's new device flag
    assert "--device=cpu" in src
    # A helper (or inline predicate) classifies OOM from the captured
    # subprocess text; the essential invariant is that retry logic
    # references both the OOM string and the cpu retry.
    assert "OOM" in src or "out of memory" in src.lower()


# ---------------------------------------------------------------------------
# Phase 7.3 tests: per-group loss-curve grids (cell 22) + aux inspection
# DataFrame (cell 23)
# ---------------------------------------------------------------------------


def test_loss_curves_cell():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[21].source) if isinstance(nb.cells[21].source, list) else nb.cells[21].source
    # All three groups plotted
    assert "_specs_group1" in src
    assert "_specs_group2" in src
    assert "_specs_group3" in src
    # Helper abstraction is present
    assert "_plot_group" in src
    # Per-group figure is saved
    assert "loss_curves_group1_h2o_short" in src or "loss_curves_" in src
    # Reads per-spec loss history from losses.npy
    assert "losses.npy" in src
    # Log-y scale used (semilogy, matches training-loss idiom)
    assert "semilogy" in src


def test_aux_inspection_cell():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[22].source) if isinstance(nb.cells[22].source, list) else nb.cells[22].source
    # Spans every spec across all three groups
    assert "_all_specs" in src
    assert "_specs_group1" in src
    assert "_specs_group2" in src
    assert "_specs_group3" in src
    # DataFrame construction + expected columns
    assert "DataFrame" in src or "pd.DataFrame" in src
    assert "loss_total_final" in src
    assert "loss_vxc_final" in src
    assert "loss_anchor_final" in src
    # Reads the aux-log artifact (the canonical per-step record -- NOT
    # train_metadata.json for history; metadata is scalar-only).
    assert "aux_log.pkl" in src


def test_imports_cell_has_pickle_for_aux_reading():
    """Phase 7.3 prerequisite: imports cell must bring in the serializer so
    cell 23 can deserialize {spec.checkpoint_dir}/aux_log artifact."""
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[1].source) if isinstance(nb.cells[1].source, list) else nb.cells[1].source
    assert "import pickle" in src


# ---------------------------------------------------------------------------
# Phase 8.1 tests: eval md + main sweep (run_test loop) + tidy DataFrame
# (cells 24-26)
# ---------------------------------------------------------------------------


def test_main_eval_loop_cell():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[24].source) if isinstance(nb.cells[24].source, list) else nb.cells[24].source
    assert "run_test" in src
    assert "RERUN_EVAL" in src
    assert "ATOMIC_ENERGIES_CHAKRAVORTY" in src
    assert "jax.clear_caches" in src


def test_eval_df_cell():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[25].source) if isinstance(nb.cells[25].source, list) else nb.cells[25].source
    assert "eval_df" in src
    assert "per_molecule.json" in src
    assert "eval_df.parquet" in src


def test_vxc_efficacy_cell():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[27].source) if isinstance(nb.cells[27].source, list) else nb.cells[27].source
    # Three V_xc-comparison families: L1 (no V_xc control), L3
    # (LossNorm-step-0 V_xc), L5 (GradNorm dynamic V_xc).
    assert "L1_B" in src and "L3_balanced_vxc" in src and "L5_gradnorm_vxc" in src
    assert "vxc_efficacy.png" in src


def test_anchor_effect_cell():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[28].source) if isinstance(nb.cells[28].source, list) else nb.cells[28].source
    assert "L3_balanced_vxc" in src and "L4_balanced_vxc_anchor" in src
    assert "anchor" in src.lower() and "anchor_effect.png" in src


def test_balancing_effect_cell_compares_l3_vs_l5():
    """Cell 28b (balancing-strategy effect) plots ΔAE = |L3| - |L5| per
    (arch, group, solver). L3 = LossNormConfig step-0; L5 =
    GradNormConfig dynamic. Positive bars indicate GradNorm beats
    LossNorm-step-0 on the V_xc-bottlenecked configurations (the
    expected sign for V_xc-aware training)."""
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[29].source) if isinstance(nb.cells[29].source, list) else nb.cells[29].source
    assert "L3_balanced_vxc" in src and "L5_gradnorm_vxc" in src
    assert "balancing_effect.png" in src
    assert "GradNorm" in src or "gradnorm" in src.lower()
    assert "LossNorm" in src or "lossnorm" in src.lower()


# ---------------------------------------------------------------------------
# Phase 9 tests: transfer-learning md + data gen + eval loops + aggregate plots
# (cells 30-36 after 2026-04-28 balancing_effect insertion at 29)
# ---------------------------------------------------------------------------


def test_transfer_primary_cell_has_w411_geometries():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[31].source) if isinstance(nb.cells[31].source, list) else nb.cells[31].source
    for v in ("0.370946", "0.107851", "-0.862809", "0.628099", "109.493", "107.208", "420.420"):
        assert v in src, f"missing {v}"


def test_transfer_secondary_cell_has_uks_nh2():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[32].source) if isinstance(nb.cells[32].source, list) else nb.cells[32].source
    # NH2 (UKS) must be present with its W4-11 geom + AE
    assert "NH2" in src
    assert "0.142235" in src and "0.800646" in src
    assert "182.591" in src
    # Other secondary molecules
    for v in ("NH3", "HF", "CO2", "298.018", "141.640", "390.141"):
        assert v in src, f"missing {v}"


def test_transfer_eval_cells_build_dataframes():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src32 = "".join(nb.cells[33].source) if isinstance(nb.cells[33].source, list) else nb.cells[33].source
    src33 = "".join(nb.cells[34].source) if isinstance(nb.cells[34].source, list) else nb.cells[34].source
    assert "transfer_primary_df" in src32
    assert "transfer_secondary_df" in src33
    assert "run_test" in src32 and "run_test" in src33


def test_transfer_aggregate_plot_cells_exist():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src34 = "".join(nb.cells[35].source) if isinstance(nb.cells[35].source, list) else nb.cells[35].source
    src35 = "".join(nb.cells[36].source) if isinstance(nb.cells[36].source, list) else nb.cells[36].source
    assert "transfer_primary_df" in src34
    assert "transfer_secondary_df" in src35


# ---------------------------------------------------------------------------
# Phase 10 tests: F_x drift diagnostic header + Panel B (CH4 + C2H2)
# + Panel C (C2H4) + SCF convergence aggregate (cells 37-40 after 2026-04-28
# balancing_effect insertion at 29)
# ---------------------------------------------------------------------------


def test_drift_md_cell():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[37].source) if isinstance(nb.cells[37].source, list) else nb.cells[37].source
    # Section-7 header pins the diagnostic quantity (F_x) and its subject.
    assert "F_x" in src
    assert "Drift Diagnostic" in src


def test_drift_panel_b_cell():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[38].source) if isinstance(nb.cells[38].source, list) else nb.cells[38].source
    assert "CH4" in src and "C2H2" in src
    # Panel B evaluates F_x at real molecular grid points using the
    # network's actual descriptor features (cusp + dm_statistics) -- not
    # zero extras -- so the curves reflect what the network produces
    # during a real SCF on each molecule. The 2026-04-26 fix replaced
    # `_nn_fx_local_uks` (zero-extras) with assemble_descriptor_features
    # over a precomputed mol_data.
    assert "assemble_descriptor_features" in src
    assert "precompute_fixed_density_data" in src
    assert "tree_deserialise_leaves" in src
    assert "fx_drift_panel_B.png" in src


def test_drift_panel_c_cell():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[39].source) if isinstance(nb.cells[39].source, list) else nb.cells[39].source
    assert "C2H4" in src
    assert "0.667100" in src
    assert "fx_drift_panel_C.png" in src


def test_scf_convergence_cell():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[40].source) if isinstance(nb.cells[40].source, list) else nb.cells[40].source
    # The cell reads the per-cycle SCF residual rows emitted by
    # SCFConvergenceMetric; pin both the metric and the row-key prefix.
    assert "scf_energy_residual" in src
    assert "SCFConvergenceMetric" in src


@pytest.mark.slow
def test_notebook_executes_end_to_end(tmp_path):
    """Narrow-config smoke: 1 arch x 1 loss x 1 solver, reduced n_steps.
    Verifies the generated notebook runs without exception."""
    import nbclient
    gen = load_generator()
    _ipynb = tmp_path / "smoke.ipynb"
    nb = gen.main(
        arch_names=("deep_combined",),
        loss_names=("L1_B",),
        solver_labels=("oneshot",),
        checkpoint_base=str(tmp_path / "checkpoints_step6_smoke"),
        output_path=str(_ipynb),
    )
    client = nbclient.NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
