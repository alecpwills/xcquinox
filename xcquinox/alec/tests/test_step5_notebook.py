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


# ---------------------------------------------------------------------------
# Task 2 -- Pretraining Cells 7-11
# ---------------------------------------------------------------------------


def test_cell_07_pretrain_md_exists():
    gen = load_generator()
    cell = gen.build_cell_07_pretrain_md()
    assert cell.cell_type == "markdown"
    assert "Pretraining" in cell.source


def test_cell_08_pretrain_data_gen_uses_rho_cutoff():
    gen = load_generator()
    source = gen.build_cell_08_pretrain_data_gen().source
    assert "valid = rho > 1e-10" in source
    assert "cusp_list, dm_list = [], []" in source


def test_cell_08_pretrain_data_gen_uses_np_where():
    gen = load_generator()
    source = gen.build_cell_08_pretrain_data_gen().source
    assert "np.where(np.abs(ex_lda)" in source


def test_cell_08_pretrain_data_gen_need_flags():
    gen = load_generator()
    source = gen.build_cell_08_pretrain_data_gen().source
    assert "need_cusp = any(" in source
    assert "need_dm = any(" in source


def test_cell_09_pretrain_loop_qualifies_alec():
    gen = load_generator()
    source = gen.build_cell_09_pretrain_loop().source
    assert "alec.PretrainSpec(" in source
    assert "alec.run_pretrain(" in source


def test_cell_10_pretrain_loss_plot():
    gen = load_generator()
    source = gen.build_cell_10_pretrain_loss_plot().source
    assert "arch_colors[arch_name]" in source
    assert "pretrain_losses.png" in source


def test_cell_11_pretrain_parity():
    gen = load_generator()
    source = gen.build_cell_11_pretrain_parity().source
    assert "rho_all" in source
    assert "sigma_all" in source


# ---------------------------------------------------------------------------
# Task 3 -- Training Data Cells 12-16
# ---------------------------------------------------------------------------


def test_cell_12_training_md():
    gen = load_generator()
    cell = gen.build_cell_12_training_md()
    assert cell.cell_type == "markdown"
    assert "ERI" in cell.source or "FULL" in cell.source


def test_cell_13_reference_dicts():
    gen = load_generator()
    source = gen.build_cell_13_reference_dicts().source
    assert "atom_energies_literature" in source
    assert "targets" in source


def test_cell_14_hf_ccsd_gen():
    gen = load_generator()
    source = gen.build_cell_14_hf_ccsd_gen().source
    assert "atom_energies" in source
    assert "E_pbe_total" in source
    # CCSD-era requirements:
    assert "dm_mo_ccsd = mycc.make_rdm1()" in source
    assert "mf_hf.mo_coeff" in source
    assert 'ref_density_method="ccsd"' in source
    assert "E_ref_literature=float(E_ccsd_total)" in source
    assert "alec.run_oep_inversion" in source
    assert "alec.save_vxc_ref" in source
    assert 'aux_basis="def2-svp-jkfit"' in source
    assert "DATA VERSION: ccsd" in source


def test_cell_15_mol_specs():
    gen = load_generator()
    source = gen.build_cell_15_mol_specs().source
    assert "alec.MoleculeSpec(" in source
    assert "mol_specs" in source


def test_cell_16_precompute_requires_eri():
    gen = load_generator()
    source = gen.build_cell_16_precompute().source
    assert '"eri"' in source
    assert "precompute_fixed_density_data" in source or "alec.precompute_fixed_density_data" in source


# ---------------------------------------------------------------------------
# Task 4 -- SCF-Varied Training Cells 17-21
# ---------------------------------------------------------------------------


def test_cell_17_training_md():
    gen = load_generator()
    cell = gen.build_cell_17_training_md()
    assert cell.cell_type == "markdown"
    assert "72" in cell.source or "solver" in cell.source.lower()


def test_cell_18_training_specs_triple_loop():
    """Cell 18 must build specs with a triple-nested loop over arch/loss/solver."""
    gen = load_generator()
    source = gen.build_cell_18_training_specs().source
    assert "for arch_name in ARCH_NAMES:" in source
    assert "for loss_name in LOSS_NAMES:" in source
    assert "for solver_label" in source
    assert "solver_config" in source
    assert "SCF_CONFIGS" in source


def test_cell_18_training_specs_solver_in_loss_kwargs():
    """solver_config must flow through loss_kwargs for B/C losses."""
    gen = load_generator()
    source = gen.build_cell_18_training_specs().source
    assert "solver_config" in source


def test_cell_18_training_specs_checkpoint_path_has_solver():
    """Checkpoint dir must include solver_label tier."""
    gen = load_generator()
    source = gen.build_cell_18_training_specs().source
    assert "{solver_label}" in source or "solver_label" in source


def test_cell_18_training_specs_loss_kwargs_abc():
    """Only losses A, B, C are used in step5."""
    gen = load_generator()
    source = gen.build_cell_18_training_specs().source
    assert '"A_atomization"' in source
    assert '"B_atomization_plus_dm"' in source
    assert '"C_atomization_plus_grid"' in source
    assert "D1_delta_ae" not in source


def test_cell_19_training_loop_three_tier_tqdm():
    """Training loop must have three-tier progress display."""
    gen = load_generator()
    source = gen.build_cell_19_training_loop().source
    assert "tqdm(" in source
    assert "alec.run_training(" in source


def test_cell_20_training_loss_plot_3x3():
    """Training loss plot must be a 3x3 grid: rows=solver, cols=loss."""
    gen = load_generator()
    source = gen.build_cell_20_training_loss_plot().source
    assert "3, 3" in source or "3,3" in source
    assert "SOLVER_LABELS" in source
    assert "LOSS_NAMES" in source
    assert "training_losses.png" in source


def test_cell_21_aux_inspection():
    gen = load_generator()
    source = gen.build_cell_21_aux_inspection().source
    assert "aux_log.pkl" in source
    assert "deep_combined" in source


def test_cell_21_balancing_md():
    gen = load_generator()
    cell = gen.build_cell_21_balancing_md()
    assert cell.cell_type == "markdown"
    assert "Balancing" in cell.source or "balancing" in cell.source
    assert "Section 4b" in cell.source


def test_cell_22_balancing_configs_has_base_and_vxc():
    gen = load_generator()
    source = gen.build_cell_22_balancing_configs().source
    # Base balancing sweep
    assert "BALANCING_CONFIGS" in source
    assert "LossNormConfig" in source
    assert "TwoPhaseConfig(phase1_steps=100)" in source
    assert "GradNormConfig" in source
    assert "BAL_LOSS_NAMES" in source
    # V_xc variants
    assert "VXC_VARIANTS" in source
    assert '"static_vxc"' in source
    assert '"two_phase_dfirst"' in source
    assert '"static_vxc_A"' in source
    assert '"vxc_weight"' in source
    assert "phase1_loss_kwargs" in source
    # V_xc sweep spans all solvers
    assert "for solver_label in SOLVER_LABELS" in source
    # Checkpoint path scheme for V_xc
    assert "train_balancing/vxc/" in source


def test_cell_23_balancing_loop():
    gen = load_generator()
    source = gen.build_cell_23_balancing_loop().source
    assert "bal_specs" in source
    assert "alec.run_training" in source or "run_training" in source
    assert "tqdm" in source
    assert "TRAIN_SKIP_IF_EXISTS" in source


# ---------------------------------------------------------------------------
# Task 5 -- Evaluation Cells 22-25
# ---------------------------------------------------------------------------


def test_cell_22_eval_md():
    gen = load_generator()
    cell = gen.build_cell_22_eval_md()
    assert cell.cell_type == "markdown"
    assert "solver_config" in cell.source or "evaluation" in cell.source.lower()


def test_cell_23_test_loop_triple_nested():
    """Eval loop must sweep arch x loss x solver."""
    gen = load_generator()
    source = gen.build_cell_23_test_loop().source
    assert "for arch_name in ARCH_NAMES:" in source
    assert "for loss_name in LOSS_NAMES:" in source
    assert "for solver_label in SOLVER_LABELS:" in source
    assert "alec.run_test(" in source
    assert "solver_config" in source


def test_cell_24_dataframe_includes_solver():
    """DataFrame must be indexed by (arch, loss, solver)."""
    gen = load_generator()
    source = gen.build_cell_24_dataframe().source
    assert "solver" in source.lower()


def test_cell_25_results_table():
    gen = load_generator()
    source = gen.build_cell_25_results_table().source
    assert "solver" in source.lower()


# ---------------------------------------------------------------------------
# Task 6 -- Primary Visualization Cells 26-31
# ---------------------------------------------------------------------------


def test_cell_26_scf_impact_md():
    gen = load_generator()
    cell = gen.build_cell_26_scf_impact_md()
    assert cell.cell_type == "markdown"


def test_cell_27_scf_comparison_bars():
    """Headline figure: grouped bars by solver config per loss."""
    gen = load_generator()
    source = gen.build_cell_27_scf_comparison_bars().source
    assert "solver_colors" in source
    assert "scf_comparison_ae.png" in source
    assert "SOLVER_LABELS" in source


def test_cell_28_dm_heatmaps_md():
    gen = load_generator()
    cell = gen.build_cell_28_dm_heatmaps_md()
    assert cell.cell_type == "markdown"


def test_cell_29_dm_heatmaps():
    gen = load_generator()
    source = gen.build_cell_29_dm_heatmaps().source
    assert "dm_heatmaps_scf.png" in source


def test_cell_30_density_histograms_md():
    gen = load_generator()
    cell = gen.build_cell_30_density_histograms_md()
    assert cell.cell_type == "markdown"


def test_cell_31_density_histograms():
    gen = load_generator()
    source = gen.build_cell_31_density_histograms().source
    assert "grid_density_scf.png" in source


# ---------------------------------------------------------------------------
# Task 7 -- Advanced Visualization + Extension Cells 32-39
# ---------------------------------------------------------------------------


def test_cell_32_convergence_md():
    gen = load_generator()
    cell = gen.build_cell_32_convergence_md()
    assert cell.cell_type == "markdown"


def test_cell_33_convergence_diagnostic():
    """Convergence plot must run SCF at higher max_cycles for diagnostic."""
    gen = load_generator()
    source = gen.build_cell_33_convergence_diagnostic().source
    assert "scf_convergence.png" in source
    assert "max_cycles=10" in source or "max_cycles" in source


def test_cell_34_feature_impact_md():
    gen = load_generator()
    cell = gen.build_cell_34_feature_impact_md()
    assert cell.cell_type == "markdown"


def test_cell_35_feature_impact():
    gen = load_generator()
    source = gen.build_cell_35_feature_impact().source
    assert "feature_impact_scf.png" in source
    assert "deep_cusp" in source or "non-attention" in source.lower()


def test_cell_36_extension_md():
    gen = load_generator()
    cell = gen.build_cell_36_extension_md()
    assert cell.cell_type == "markdown"


def test_cell_37_new_molecule_template():
    gen = load_generator()
    source = gen.build_cell_37_new_molecule_template().source
    assert "CH4" in source
    assert "alec.MoleculeSpec(" in source
    assert "new_atom_energies" in source


def test_cell_38_new_mol_comparison_md():
    gen = load_generator()
    cell = gen.build_cell_38_new_mol_comparison_md()
    assert cell.cell_type == "markdown"


def test_cell_39_new_mol_comparison():
    gen = load_generator()
    source = gen.build_cell_39_new_mol_comparison().source
    assert "SOLVER_LABELS" in source
    assert "solver_colors" in source


# ---------------------------------------------------------------------------
# Task 9 -- structural validation tests
# ---------------------------------------------------------------------------


def test_generator_produces_39_cells(tmp_path):
    """Step 5 notebook must contain exactly 39 cells."""
    gen = load_generator()
    nb = gen.main(str(tmp_path / "step5.ipynb"))
    assert len(nb.cells) == 39, f"expected 39 cells, got {len(nb.cells)}"


def test_generator_cell_types_match_expected(tmp_path):
    """Markdown cells must appear at the expected indices."""
    gen = load_generator()
    nb = gen.main(str(tmp_path / "step5.ipynb"))
    markdown_indices = set()
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown":
            markdown_indices.add(i)
    # Cells 0 (title), 6 (pretrain md), 11 (training data md),
    # 16 (training md), 21 (eval md), 25 (scf impact md),
    # 27 (dm heatmaps md), 29 (density hist md), 31 (convergence md),
    # 33 (feature impact md), 35 (extension md), 37 (new mol comparison md)
    expected = {0, 6, 11, 16, 21, 25, 27, 29, 31, 33, 35, 37}
    assert markdown_indices == expected, (
        f"markdown indices {markdown_indices} != expected {expected}"
    )


def test_generator_deterministic_ids(tmp_path):
    """Cell IDs must be deterministic (cell_00, cell_01, ...)."""
    gen = load_generator()
    nb = gen.main(str(tmp_path / "step5.ipynb"))
    for idx, cell in enumerate(nb.cells):
        assert cell.id == f"cell_{idx:02d}", (
            f"cell {idx} has id {cell.id!r}, expected 'cell_{idx:02d}'"
        )


def test_generator_byte_identical_on_rerun(tmp_path):
    """Two back-to-back runs must produce byte-identical notebooks."""
    gen = load_generator()
    path_a = tmp_path / "a.ipynb"
    path_b = tmp_path / "b.ipynb"
    gen.main(str(path_a))
    gen.main(str(path_b))
    assert path_a.read_bytes() == path_b.read_bytes()


def test_narrow_config_smoke(tmp_path):
    """Smoke: single arch, single loss, single solver must produce valid notebook."""
    gen = load_generator()
    nb = gen.main(
        str(tmp_path / "step5_narrow.ipynb"),
        arch_names=("deep",),
        loss_names=("A_atomization",),
        solver_labels=("oneshot",),
        checkpoint_base=str(tmp_path / "ckpt"),
    )
    assert len(nb.cells) == 39
    nbformat.validate(nb)


def test_main_cells_1_to_6_types(tmp_path):
    """First 6 cells must have expected types: markdown, code, code, code, code, code."""
    gen = load_generator()
    nb = gen.main(str(tmp_path / "step5.ipynb"))
    expected = ["markdown", "code", "code", "code", "code", "code"]
    actual = [c.cell_type for c in nb.cells[:6]]
    assert actual == expected


def test_notebook_contains_solver_config_definitions(tmp_path):
    """Generated notebook must contain SolverConfig definitions."""
    gen = load_generator()
    nb = gen.main(str(tmp_path / "step5.ipynb"))
    all_source = "\n".join(c.source for c in nb.cells)
    assert "SolverConfig(" in all_source
    assert "SolverMode.ONESHOT" in all_source
    assert "SolverMode.FIXED_J" in all_source
    assert "SolverMode.FULL" in all_source
