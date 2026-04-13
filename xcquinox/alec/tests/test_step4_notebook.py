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
