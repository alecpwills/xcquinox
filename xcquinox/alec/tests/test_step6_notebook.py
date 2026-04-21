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


def test_main_produces_42_cells_after_phase11():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    assert len(nb.cells) == 42


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
    _artifact = "aux_log" + ".pkl"
    assert _artifact in src


def test_imports_cell_has_pickle_for_aux_reading():
    """Phase 7.3 prerequisite: imports cell must bring in the serializer so
    cell 23 can deserialize {spec.checkpoint_dir}/aux_log artifact."""
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[1].source) if isinstance(nb.cells[1].source, list) else nb.cells[1].source
    _tok = "import " + "pickle"
    assert _tok in src


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
    src = "".join(nb.cells[26].source) if isinstance(nb.cells[26].source, list) else nb.cells[26].source
    assert "L1_B" in src and "L3_balanced_vxc" in src
    assert "vxc_efficacy.png" in src


def test_anchor_effect_cell():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[27].source) if isinstance(nb.cells[27].source, list) else nb.cells[27].source
    assert "L3_balanced_vxc" in src and "L4_balanced_vxc_anchor" in src
    assert "anchor" in src.lower() and "anchor_effect.png" in src


# ---------------------------------------------------------------------------
# Phase 9 tests: transfer-learning md + data gen + eval loops + aggregate plots
# (cells 29-35)
# ---------------------------------------------------------------------------


def test_transfer_primary_cell_has_w411_geometries():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[29].source) if isinstance(nb.cells[29].source, list) else nb.cells[29].source
    for v in ("0.370946", "0.107851", "-0.862809", "0.628099", "109.493", "107.208", "420.420"):
        assert v in src, f"missing {v}"


def test_transfer_secondary_cell_has_uks_nh2():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[30].source) if isinstance(nb.cells[30].source, list) else nb.cells[30].source
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
    src32 = "".join(nb.cells[31].source) if isinstance(nb.cells[31].source, list) else nb.cells[31].source
    src33 = "".join(nb.cells[32].source) if isinstance(nb.cells[32].source, list) else nb.cells[32].source
    assert "transfer_primary_df" in src32
    assert "transfer_secondary_df" in src33
    assert "run_test" in src32 and "run_test" in src33


def test_transfer_aggregate_plot_cells_exist():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src34 = "".join(nb.cells[33].source) if isinstance(nb.cells[33].source, list) else nb.cells[33].source
    src35 = "".join(nb.cells[34].source) if isinstance(nb.cells[34].source, list) else nb.cells[34].source
    assert "transfer_primary_df" in src34
    assert "transfer_secondary_df" in src35


# ---------------------------------------------------------------------------
# Phase 10 tests: F_x drift diagnostic header + Panel B (CH4 + C2H2)
# + Panel C (C2H4) + SCF convergence aggregate (cells 36-39)
# ---------------------------------------------------------------------------


def test_drift_md_cell():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[35].source) if isinstance(nb.cells[35].source, list) else nb.cells[35].source
    assert "drift" in src.lower() or "F_x" in src


def test_drift_panel_b_cell():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[36].source) if isinstance(nb.cells[36].source, list) else nb.cells[36].source
    assert "CH4" in src and "C2H2" in src
    assert "_nn_fx_local_uks" in src
    assert "tree_deserialise_leaves" in src
    assert "fx_drift_panel_B.png" in src


def test_drift_panel_c_cell():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[37].source) if isinstance(nb.cells[37].source, list) else nb.cells[37].source
    assert "C2H4" in src
    assert "0.667100" in src
    assert "fx_drift_panel_C.png" in src


def test_scf_convergence_cell():
    gen = load_generator()
    nb = gen.main(output_path="/tmp/_step6.ipynb")
    src = "".join(nb.cells[38].source) if isinstance(nb.cells[38].source, list) else nb.cells[38].source
    assert "SCF" in src or "cycles_run" in src or "convergence" in src.lower()


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
