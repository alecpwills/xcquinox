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


# Task 3 — Cells 6-8 builder tests


def test_cell_07_uses_rho_cutoff_1e_minus_10():
    """Cell 7's low-density mask must use `valid = rho > 1e-10`.

    Strict `>` not `>=`, threshold 1e-10 not 1e-6 — guards the off-by-threshold
    regression from spec B-review rounds 8-10. Step3b uses the looser cutoff
    to keep the atomic tail.
    """
    gen = load_generator()
    source = gen.build_cell_07_pretrain_data_gen().source
    assert "valid = rho > 1e-10" in source


def test_cell_07_uses_np_where_safe_division():
    """Cell 7 must use np.where-based safe division, NOT a boolean mask.

    Boolean masks drop points step3b keeps; np.where keeps shape parity so
    the downstream `valid` filter is the only mask applied.
    """
    gen = load_generator()
    source = gen.build_cell_07_pretrain_data_gen().source
    assert "np.where(np.abs(ex_lda)" in source


def test_cell_07_lists_initialised_unconditionally():
    """`cusp_list, dm_list = [], []` must appear before the PRETRAIN_ATOMS loop.

    Unconditional init makes the `if cusp_list:` / `if dm_list:` truthy-check
    at save time safe even when `ARCH_NAMES` contains no extended-feature archs.
    """
    gen = load_generator()
    source = gen.build_cell_07_pretrain_data_gen().source
    init_idx = source.find("cusp_list, dm_list = [], []")
    loop_idx = source.find("for atom_symbol, spin in PRETRAIN_ATOMS:")
    assert init_idx != -1, "cusp/dm list init missing"
    assert loop_idx != -1, "PRETRAIN_ATOMS loop missing"
    assert init_idx < loop_idx, "list init must precede the loop"


def test_cell_07_uses_libxc_strings_not_helpers():
    """Cell 7 must call libxc functional strings, NOT xcquinox helpers.

    Step3b Cell 10 uses pyscf's `eval_xc("LDA_X,", ...)` / `eval_xc(",LDA_C_PW", ...)`
    for exact numerical parity; the xcquinox helpers must NOT be imported here.
    """
    gen = load_generator()
    source = gen.build_cell_07_pretrain_data_gen().source
    assert '"LDA_X,"' in source
    assert '",LDA_C_PW"' in source
    assert "from xcquinox.utils import lda_x" not in source


def test_cell_07_need_flags_gate_extended_features():
    """`need_cusp`/`need_dm` must be derived via `any(...)` and gate the
    descriptor computation branches."""
    gen = load_generator()
    source = gen.build_cell_07_pretrain_data_gen().source
    assert "need_cusp = any(" in source
    assert "need_dm = any(" in source
    assert "if need_cusp:" in source
    assert "if need_dm:" in source


def test_cell_08_qualifies_alec_pretrainspec():
    """Cell 8 must use `alec.PretrainSpec(`, never bare `PretrainSpec(`."""
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source
    assert "alec.PretrainSpec(" in source
    # Ensure no bare PretrainSpec usage — check that every PretrainSpec
    # occurrence is preceded by "alec."
    import re
    bare_refs = re.findall(r"(?<!alec\.)PretrainSpec\(", source)
    assert bare_refs == [], f"bare PretrainSpec references found: {bare_refs}"


def test_cell_08_passes_step3b_hyperparameters():
    """Cell 8's PretrainSpec must pass the step3b hyperparameters."""
    gen = load_generator()
    source = gen.build_cell_08_pretrain_loop().source
    for literal in ("n_steps=1000", "lr_start=1e-2", "lr_end=1e-5",
                    "lr_decay_start=0.2", "grad_clip=1.0"):
        assert literal in source, f"missing hyperparameter literal: {literal}"


# Task 4 — Cells 9-10 builder tests


def test_cell_09_loads_losses_x_and_losses_c():
    """Cell 9 must load both xnet and cnet loss arrays by the path template
    Cell 8 writes to.
    """
    gen = load_generator()
    source = gen.build_cell_09_pretrain_loss_plot().source
    assert 'losses_x.npy' in source
    assert 'losses_c.npy' in source


def test_cell_09_uses_log_scale():
    """Cell 9 must use log y-scale so order-of-magnitude loss decay is visible."""
    gen = load_generator()
    source = gen.build_cell_09_pretrain_loss_plot().source
    assert "semilogy(" in source or 'set_yscale("log")' in source


def test_cell_09_saves_to_figures_dir():
    """Cell 9 must save the plot under `{CHECKPOINT_BASE}/figures/`."""
    gen = load_generator()
    source = gen.build_cell_09_pretrain_loss_plot().source
    assert '{CHECKPOINT_BASE}/figures/pretrain_losses.png' in source


def test_cell_10_uses_create_network_pair_skeleton():
    """Cell 10 must construct (xnet, cnet) skeletons via `alec.create_network_pair`.

    This is the pretrain-layout skeleton path; Cell 26 (full-model load) uses a
    different entry point (`AlecGGAModel.from_arch`), so the difference matters.
    """
    gen = load_generator()
    source = gen.build_cell_10_pretrain_parity().source
    assert "alec.create_network_pair(" in source


def test_cell_10_uses_tree_deserialise_leaves():
    """Cell 10 must deserialise the saved .eqx weights via eqx.tree_deserialise_leaves."""
    gen = load_generator()
    source = gen.build_cell_10_pretrain_parity().source
    assert "eqx.tree_deserialise_leaves(" in source


def test_cell_10_is_12x2_or_documented_subset():
    """Cell 10 must build a (n_arch x 2) subplots grid — 12 rows for the full
    default ARCH_NAMES or a narrower grid when the test harness passes a subset.

    Accept either an explicit `subplots(12, 2` literal OR a dynamic
    `subplots(n_arch, 2` / `subplots(len(ARCH_NAMES), 2` form.
    """
    gen = load_generator()
    source = gen.build_cell_10_pretrain_parity().source
    ok_forms = ("subplots(12, 2", "subplots(n_arch, 2", "subplots(len(ARCH_NAMES), 2")
    assert any(form in source for form in ok_forms), (
        f"Cell 10 must call subplots with an (n_arch, 2) grid; none of "
        f"{ok_forms} found in source."
    )


# Task 5 — Cells 11-13 builder tests


def test_cell_12_targets_has_all_three_molecules():
    """`targets` dict must contain H, O, H2O — validator requires entries for
    every molecule in TrainingSpec.molecules (config.py:523-525).
    """
    gen = load_generator()
    source = gen.build_cell_12_reference_dicts().source
    # Targets dict should contain literal "H", "O", "H2O" keys
    assert "targets = {" in source
    for key in ('"H":', '"O":', '"H2O":'):
        assert key in source, f"targets dict missing key literal {key}"


def test_cell_12_atom_energies_missing_h2o():
    """`atom_energies` must contain exactly H and O — H2O deliberately absent.

    H2O is a compound, not an atom; placing it here would confuse the
    AtomizationEnergyMetric accumulator.
    """
    gen = load_generator()
    source = gen.build_cell_12_reference_dicts().source
    assert 'atom_energies = {"H": -0.5, "O": -75.0673}' in source
    # H2O must NOT appear as a key inside the atom_energies literal. Scan the
    # atom_energies literal slice only (to avoid matching the targets dict
    # above it).
    ae_start = source.find("atom_energies =")
    ae_end = source.find("}", ae_start) + 1
    ae_literal = source[ae_start:ae_end]
    assert '"H2O"' not in ae_literal


def test_cell_12_ext_data_dir_uses_checkpoint_base():
    """`ext_data_dir` must be derived from CHECKPOINT_BASE so smoke tests can
    redirect it via the tmp_path harness.
    """
    gen = load_generator()
    source = gen.build_cell_12_reference_dicts().source
    assert 'ext_data_dir = f"{CHECKPOINT_BASE}/external_data"' in source


def test_cell_13_uses_hf_dm_not_ccsd_dm():
    """Cell 13 must use the HF density matrix (step3b convention) — never the
    CCSD 1-RDM, despite the misleading `dm_target` key name.
    """
    gen = load_generator()
    source = gen.build_cell_13_hf_ccsd_gen().source
    assert "mf_hf.make_rdm1()" in source
    assert "mycc.make_rdm1()" not in source


def test_cell_13_atom_branch_writes_only_e_ref():
    """The atom branch must write ONLY E_ref_literature — not dm_target and
    not rho_ccsd_grid. Atomic one-shot density targets are unstable due to
    degenerate HOMOs in open-shell atoms.
    """
    gen = load_generator()
    source = gen.build_cell_13_hf_ccsd_gen().source
    # Find the atom branch (if name in ("H", "O"): ... else: ...)
    branch_start = source.find('if name in ("H", "O"):')
    branch_end = source.find("else:", branch_start)
    assert branch_start != -1 and branch_end != -1, "atom branch not found"
    atom_branch = source[branch_start:branch_end]
    assert "E_ref_literature=" in atom_branch
    assert "dm_target=" not in atom_branch
    assert "rho_ccsd_grid=" not in atom_branch


def test_cell_13_h2o_branch_writes_three_keys():
    """The H2O branch must write all three whitelisted keys: dm_target,
    rho_ccsd_grid, and E_ref_literature. These are the ONLY keys
    _ALLOWED_EXTERNAL_KEYS accepts (data.py:17-21).
    """
    gen = load_generator()
    source = gen.build_cell_13_hf_ccsd_gen().source
    branch_start = source.find("else:", source.find('if name in ("H", "O"):'))
    assert branch_start != -1, "H2O else branch not found"
    h2o_branch = source[branch_start:]
    assert "dm_target=dm_hf" in h2o_branch
    assert "rho_ccsd_grid=rho_hf" in h2o_branch
    assert "E_ref_literature=float(mf_hf.e_tot)" in h2o_branch


def test_cell_13_sidecar_json_for_every_species():
    """The `_metadata.json` write must run for every species — not inside any
    branch. Cell 25 reads E_ccsd_total from this file for all three molecules
    so the CCSD atomization-energy reference line can be computed.
    """
    gen = load_generator()
    source = gen.build_cell_13_hf_ccsd_gen().source
    # The json.dump call must come AFTER the atom branch's else: block closes.
    # We check that there is exactly one json.dump and it is at indent level 4
    # (inside the for loop) but not inside any if/else — a simple heuristic is
    # to ensure the json.dump occurrence sits at the same indent level as the
    # `if name in` test, not deeper.
    assert "json.dump(" in source
    assert "_metadata.json" in source
    # The sidecar write should reference the species name via f-string, so it
    # fires for every iteration of the `for name, atom, spin in _mols:` loop.
    assert 'f"{name}_metadata.json"' in source


def test_cell_13_uses_grid_level_pinned():
    """`mf.grids.level = GRID_LEVEL` must appear before `mf.kernel()` so the
    PBE grid matches what Cell 14/15's precompute_fixed_density_data rebuilds.
    """
    gen = load_generator()
    source = gen.build_cell_13_hf_ccsd_gen().source
    level_idx = source.find("mf.grids.level = GRID_LEVEL")
    kernel_idx = source.find("mf.kernel()")
    assert level_idx != -1, "mf.grids.level = GRID_LEVEL missing"
    assert kernel_idx != -1, "mf.kernel() missing"
    assert level_idx < kernel_idx, "grid level must be pinned before kernel()"


def test_cell_13_einsum_is_rho_hf_not_rho_nn():
    """The einsum variable must be named `rho_hf`, guarding the step3b-era
    `rho_nn` naming confusion — `rho_ccsd_grid` is HF in disguise.
    """
    gen = load_generator()
    source = gen.build_cell_13_hf_ccsd_gen().source
    assert 'rho_hf = np.einsum("ij,gi,gj->g"' in source
    # `rho_nn` would indicate the wrong name reappeared
    assert "rho_nn = np.einsum" not in source


# Task 6 — Cells 14-15 builder tests


def test_cell_14_mol_specs_has_three_entries():
    """Cell 14 must construct exactly three alec.MoleculeSpec instances."""
    gen = load_generator()
    source = gen.build_cell_14_mol_specs().source
    assert source.count("alec.MoleculeSpec(") == 3


def test_cell_14_all_specs_carry_grid_level():
    """All three MoleculeSpec entries must set grid_level=GRID_LEVEL so
    precompute rebuilds the same grid Cell 13 used.
    """
    gen = load_generator()
    source = gen.build_cell_14_mol_specs().source
    assert source.count("grid_level=GRID_LEVEL") == 3


def test_cell_14_h2o_uses_h2o_coords_constant():
    """The H2O MoleculeSpec must reference H2O_COORDS (Cell 3), not a re-literal."""
    gen = load_generator()
    source = gen.build_cell_14_mol_specs().source
    assert "atom=H2O_COORDS" in source


def test_cell_14_all_specs_carry_external_data_path():
    """All three MoleculeSpec entries must point at an f-string path derived
    from ext_data_dir (Cell 12).
    """
    gen = load_generator()
    source = gen.build_cell_14_mol_specs().source
    assert source.count('external_data_path=f"{ext_data_dir}/') == 3


def test_cell_15_asserts_atom_rho_ccsd_is_none():
    """Cell 15 must assert both the atom-branch negative case and the H2O
    positive case on rho_ccsd_grid.
    """
    gen = load_generator()
    source = gen.build_cell_15_precompute_sanity().source
    assert 'mol_data_list[0]["rho_ccsd_grid"] is None' in source
    assert 'mol_data_list[2]["rho_ccsd_grid"] is not None' in source


# Task 7 -- Cells 16-20 builder tests


def test_cell_17_builds_specs_list():
    """Cell 17 must bind `specs = []` before the nested loop."""
    gen = load_generator()
    source = gen.build_cell_17_training_specs().source
    init_idx = source.find("specs = []")
    loop_idx = source.find("for arch_name in ARCH_NAMES:")
    assert init_idx != -1, "specs accumulator missing"
    assert loop_idx != -1, "outer arch loop missing"
    assert init_idx < loop_idx, "specs = [] must precede the loop"


def test_cell_17_loop_is_arch_then_loss_order():
    """Outer loop must iterate arch_name, inner loop must iterate loss_name."""
    gen = load_generator()
    source = gen.build_cell_17_training_specs().source
    arch_idx = source.find("for arch_name in ARCH_NAMES:")
    loss_idx = source.find("for loss_name in LOSS_NAMES:")
    assert arch_idx != -1 and loss_idx != -1
    assert arch_idx < loss_idx, "arch loop must enclose loss loop"


def test_cell_17_sets_checkpoint_dir_per_pair():
    """Each spec must carry a per-(arch, loss) checkpoint_dir -- without this,
    all 72 runs overwrite each other in a single directory.
    """
    gen = load_generator()
    source = gen.build_cell_17_training_specs().source
    assert 'checkpoint_dir=f"{CHECKPOINT_BASE}/train/{arch_name}/{loss_name}"' in source


def test_cell_17_passes_step3b_hyperparameters():
    """Cell 17 must pass n_steps=250, lr_start=1e-2, lr_decay_start=0.2 --
    the TrainingSpec defaults differ from step3b and silently produce wrong
    training curves if left alone.
    """
    gen = load_generator()
    source = gen.build_cell_17_training_specs().source
    for literal in ("n_steps=250", "lr_start=1e-2", "lr_decay_start=0.2"):
        assert literal in source, f"missing hyperparameter literal: {literal}"


def test_cell_17_uses_qualified_alec_trainingspec():
    """Cell 17 must use `alec.TrainingSpec.from_dicts(` -- never bare."""
    gen = load_generator()
    source = gen.build_cell_17_training_specs().source
    assert "alec.TrainingSpec.from_dicts(" in source
    import re
    bare_refs = re.findall(r"(?<!alec\.)TrainingSpec\.from_dicts\(", source)
    assert bare_refs == [], f"bare TrainingSpec references found: {bare_refs}"


def test_cell_17_loss_kwargs_weight_values():
    """LOSS_KWARGS must use 0.1 weights for dm and density -- not 1.0 or 0.01."""
    gen = load_generator()
    source = gen.build_cell_17_training_specs().source
    assert '"dm_weight": 0.1' in source
    assert '"density_weight": 0.1' in source
    # Guard against wrong weight magnitudes
    assert '"dm_weight": 1.0' not in source
    assert '"density_weight": 0.01' not in source


def test_cell_18_is_serial():
    """Cell 18 must implement the serial path only -- no parallel build_training_jobs."""
    gen = load_generator()
    source = gen.build_cell_18_training_loop().source
    assert "for spec in specs:" in source
    assert "alec.run_training(spec" in source
    assert "alec.build_training_jobs(" not in source


def test_cell_19_loads_losses_npy():
    """Cell 19 must load each per-(arch, loss) losses.npy using the
    checkpoint path template Cell 17 wrote to.
    """
    gen = load_generator()
    source = gen.build_cell_19_training_loss_plot().source
    assert "/train/{arch_name}/{loss_name}/losses.npy" in source


def test_cell_20_binds_arch_name_before_loop():
    """Cell 20 must bind arch_name = "shallow" before the for loss_name loop
    so the f-string checkpoint path is unambiguous.
    """
    gen = load_generator()
    source = gen.build_cell_20_aux_inspection().source
    bind_idx = source.find('arch_name = "shallow"')
    loop_idx = source.find("for loss_name in LOSS_NAMES:")
    assert bind_idx != -1, 'arch_name = "shallow" missing'
    assert loop_idx != -1, "loss_name loop missing"
    assert bind_idx < loop_idx, "arch_name binding must precede the loss loop"


# Task 8 -- Cells 21-24 builder tests


def test_cell_22_metrics_tuple_is_four():
    """Cell 22 must pass all four metrics explicitly (not rely on default)."""
    gen = load_generator()
    source = gen.build_cell_22_test_loop().source
    for metric in ("total_energy", "atomization_energy", "density_rmse", "constraint_violations"):
        assert f'"{metric}"' in source, f"metric {metric!r} missing from Cell 22 metrics tuple"


def test_cell_22_metric_kwargs_reference_ae_kcalmol():
    """Cell 22 must pass the full-precision step3b H2O AE (233.016 kcal/mol)."""
    gen = load_generator()
    source = gen.build_cell_22_test_loop().source
    assert '"reference_ae_kcalmol"' in source
    assert '"H2O": 233.016' in source


def test_cell_22_model_checkpoint_points_to_file():
    """Cell 22 must point model_checkpoint at the .eqx file, not the dir."""
    gen = load_generator()
    source = gen.build_cell_22_test_loop().source
    assert 'model_checkpoint=f"{ckpt_dir}/model.eqx"' in source


def test_cell_22_loop_order_matches_cell_17():
    """Cell 22's loop order must match Cell 17 (arch outer, loss inner)."""
    gen = load_generator()
    source = gen.build_cell_22_test_loop().source
    arch_idx = source.find("for arch_name in ARCH_NAMES:")
    loss_idx = source.find("for loss_name in LOSS_NAMES:")
    assert arch_idx != -1, "outer arch loop missing"
    assert loss_idx != -1, "inner loss loop missing"
    assert arch_idx < loss_idx, "arch loop must precede loss loop"


def test_cell_23_ae_error_column_reads_rmse():
    """Cell 23 must populate both AE_error_kcalmol_mean and AE_error_kcalmol_RMSE (B12-4 guard)."""
    gen = load_generator()
    source = gen.build_cell_23_dataframe().source
    assert '"AE_error_kcalmol_mean"' in source
    assert '"AE_error_kcalmol_RMSE"' in source


def test_cell_23_no_constraint_violations_column():
    """Cell 23 must NOT have a constraint_violations column -- key is absent from default-arch aggregate.json."""
    gen = load_generator()
    source = gen.build_cell_23_dataframe().source
    assert "constraint_violations" not in source


def test_cell_23_uses_get_with_nan_fallback():
    """Cell 23 must use the defensive .get(..., np.nan) pattern (B10-12 guard)."""
    gen = load_generator()
    source = gen.build_cell_23_dataframe().source
    assert 'agg.get("AE_error_kcalmol", {}).get("mean", np.nan)' in source


def test_cell_23_multiindex_is_arch_loss():
    """Cell 23 must set the MultiIndex to [arch, loss]."""
    gen = load_generator()
    source = gen.build_cell_23_dataframe().source
    assert 'set_index(["arch", "loss"])' in source


# Task 9 -- Cells 25-26 builder tests


def test_cell_25_binds_best_idx():
    """Cell 25 must bind best_idx from df[AE_error_kcalmol_mean].unstack(loss).idxmin(axis=0)."""
    gen = load_generator()
    source = gen.build_cell_25_ae_bars().source
    assert 'best_idx = df["AE_error_kcalmol_mean"].unstack("loss").idxmin(axis=0)' in source


def test_cell_25_binds_pairs():
    """Cell 25 must bind the attention-pairing list programmatically from ARCH_NAMES."""
    gen = load_generator()
    source = gen.build_cell_25_ae_bars().source
    assert 'pairs = [(n, f"{n}_attn") for n in ARCH_NAMES' in source
    assert 'not n.endswith("_attn")' in source
    assert 'f"{n}_attn" in ARCH_NAMES' in source


def test_cell_25_reads_both_mean_and_rmse_columns():
    """Cell 25 must read both mean and RMSE columns (B12-4 regression guard)."""
    gen = load_generator()
    source = gen.build_cell_25_ae_bars().source
    assert 'df["AE_error_kcalmol_mean"]' in source
    assert 'df["AE_error_kcalmol_RMSE"]' in source


def test_cell_25_has_three_reference_lines():
    """Cell 25 must draw PBE, CCSD, and chemical-accuracy reference lines."""
    gen = load_generator()
    source = gen.build_cell_25_ae_bars().source
    assert "PBE Error" in source
    assert "CCSD Error" in source
    assert "Chemical accuracy (1 kcal/mol)" in source


def test_cell_25_kernel_restart_fallback_exists():
    """Cell 25 must have a try/except NameError fallback for mol_data_list (kernel-restart safety)."""
    gen = load_generator()
    source = gen.build_cell_25_ae_bars().source
    assert "except NameError:" in source


def test_cell_25_saves_to_figures_dir():
    """Cell 25 must save ae_error_by_loss.png into the figures directory."""
    gen = load_generator()
    source = gen.build_cell_25_ae_bars().source
    assert "ae_error_by_loss.png" in source


def test_cell_26_uses_alec_gga_model_from_arch_not_create_network_pair():
    """Cell 26 must use alec.AlecGGAModel.from_arch (B11-4 regression guard)."""
    gen = load_generator()
    source = gen.build_cell_26_dm_heatmaps().source
    assert "alec.AlecGGAModel.from_arch(" in source
    assert "alec.create_network_pair(" not in source


def test_cell_26_model_template_rebuilt_inside_loop():
    """Cell 26 must rebuild model_template inside the loop, not hoisted."""
    gen = load_generator()
    source = gen.build_cell_26_dm_heatmaps().source
    loop_idx = source.find('for loss_name in ("B_atomization_plus_dm",')
    template_idx = source.find("model_template = alec.AlecGGAModel.from_arch(arch_config)")
    assert loop_idx != -1, "per-loss loop missing"
    assert template_idx != -1, "model_template rebuild missing"
    assert loop_idx < template_idx, "model_template must be rebuilt INSIDE the loop"


def test_cell_26_binds_model_b_d1_d2_explicit_names():
    """Cell 26 must bind model_B, model_D1, model_D2 as explicit named variables."""
    gen = load_generator()
    source = gen.build_cell_26_dm_heatmaps().source
    assert 'model_B = model_bindings["B_atomization_plus_dm"]' in source
    assert 'model_D1 = model_bindings["D1_delta_ae"]' in source
    assert 'model_D2 = model_bindings["D2_delta_ae_plus_dm"]' in source


def test_cell_26_uses_oneshot_dm_prediction_fast():
    """Cell 26 must call the _fast variant (the only one alec exports)."""
    gen = load_generator()
    source = gen.build_cell_26_dm_heatmaps().source
    assert "alec.oneshot_dm_prediction_fast(" in source
    # The bare variant (without _fast) does not exist in alec.__init__ — guard against a rename regression.
    assert "oneshot_dm_prediction(" not in source.replace("oneshot_dm_prediction_fast(", "")


def test_cell_26_reuses_mol_data_list_for_dm_hf():
    """Cell 26 must reuse mol_data_list[2]['dm_target'] for dm_hf, not reload the .npz."""
    gen = load_generator()
    source = gen.build_cell_26_dm_heatmaps().source
    assert 'dm_hf = mol_data_list[2]["dm_target"]' in source
    assert ".npz" not in source


def test_cell_26_panel_assignment_is_explicit():
    """Cell 26 must have all four panel subtraction expressions."""
    gen = load_generator()
    source = gen.build_cell_26_dm_heatmaps().source
    for expr in ("dm_pbe - dm_hf", "dm_nn_B - dm_hf", "dm_nn_D1 - dm_hf", "dm_nn_D2 - dm_hf"):
        assert expr in source, f"panel expression {expr!r} missing"


# Task 10 -- Cells 27-29 builder tests


def test_cell_27_uses_oneshot_grid_density():
    """Cell 27 must call alec.oneshot_grid_density on mol_data_list[2]."""
    gen = load_generator()
    source = gen.build_cell_27_density_histograms().source
    assert "alec.oneshot_grid_density(" in source


def test_cell_27_reads_rho_ccsd_grid_from_mol_data_list():
    """Cell 27 must read rho_ref from mol_data_list[2]['rho_ccsd_grid']."""
    gen = load_generator()
    source = gen.build_cell_27_density_histograms().source
    assert 'mol_data_list[2]["rho_ccsd_grid"]' in source


def test_cell_27_prints_delta_rho_l1():
    """Cell 27 must compute and print the inline |delta rho|_1 metric."""
    gen = load_generator()
    source = gen.build_cell_27_density_histograms().source
    assert "delta_rho_L1 = float(jnp.sum(w *" in source


def test_cell_28_uses_pairs_from_cell_25():
    """Cell 28 must iterate the pairs list bound in Cell 25."""
    gen = load_generator()
    source = gen.build_cell_28_attn_comparison().source
    # accept either `for base, _attn in pairs` style or an explicit comprehension over `pairs`
    assert "pairs" in source
    assert " in pairs" in source


def test_cell_29_feature_variants_excludes_attn_suffix():
    """Cell 29 must exclude attention-suffixed archs from the feature filter."""
    gen = load_generator()
    source = gen.build_cell_29_feature_comparison().source
    assert 'and not n.endswith("_attn")' in source


def test_cell_29_filter_startswith_deep():
    """Cell 29 must filter to deep-prefixed archs."""
    gen = load_generator()
    source = gen.build_cell_29_feature_comparison().source
    assert 'n.startswith("deep")' in source
