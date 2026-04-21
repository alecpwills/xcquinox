"""Generator for notebooks/gga_training_example-step6.ipynb.

Step 6 tests whether (a) adding C2H2 training data and (b) a PBE-anchor
regularization term close the F_x(s) drift at s > 0.7 that step 5 found.
All geometries + AE refs come from the W4-11 tarball at
/home/awills/Documents/Research/xcdiff/testing/small/W4-11/<name>/struc.xyz.

Spec:  docs/superpowers/specs/2026-04-21-step6-notebook-design.md
Plan:  docs/superpowers/plans/2026-04-21-step6-notebook-implementation.md
"""
import os

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


DEFAULT_ARCH_NAMES = ("deep_combined", "deep_combined_attn")

DEFAULT_LOSS_NAMES = (
    "L1_B",
    "L2_C_anchor",
    "L3_balanced_vxc",
    "L4_balanced_vxc_anchor",
)

DEFAULT_SOLVER_LABELS = ("oneshot", "fixed_j_3", "full_3")

DEFAULT_CHECKPOINT_BASE = "checkpoints_step6"


def build_cell_01_title():
    source = r"""# GGA Network Training -- Step 6: Data Expansion + PBE-Anchor + Overfitting

Tests two hypothesized fixes for the F_x(s) drift at s > 0.7 (step-5 finding
on CH4) plus an overfitting diagnostic.

## Training Matrix: 2 archs x 4 losses x 3 solvers x 3 groups = 72 runs

| Loss | Kind | V_xc? | PBE-anchor? |
|---|---|---|---|
| L1 | B_atomization_plus_dm | -- | -- |
| L2 | C_atomization_plus_grid | -- | yes |
| L3 | balanced + V_xc | yes | -- |
| L4 | balanced + V_xc + anchor | yes | yes |

| Group | Data | Phase length |
|---|---|---|
| 1 | H2O only | 45 steps (short) |
| 2 | H2O + C2H2 | 45 steps (short) |
| 3 | H2O + C2H2 | 125 steps (long) |

Geometries + AE refs: W4-11 (Karton et al. 2011). Atomic refs: Chakravorty 1993.

Spec: docs/superpowers/specs/2026-04-21-step6-notebook-design.md
"""
    return new_markdown_cell(source)


def build_cell_02_imports():
    source = r"""import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import jax
import jax.numpy as jnp

from pyscf import gto, scf, cc, dft

import xcquinox.alec as alec
from xcquinox.alec import (
    ARCHITECTURES,
    MoleculeSpec,
    PretrainSpec, TrainingSpec, TestSpec,
    TwoPhaseConfig, LossNormConfig,
    build_pbe_anchor_sample, PBEAnchorSample,
    run_pretrain, run_training, run_test, run_oep_inversion, save_vxc_ref,
    precompute_fixed_density_data,
)
# SolverConfig + enums are not re-exported on xcquinox.alec; import from submodule.
from xcquinox.alec.solver import SolverConfig, SolverMode, FeaturePolicy
"""
    return new_code_cell(source)


def build_cell_03_constants(checkpoint_base: str = DEFAULT_CHECKPOINT_BASE):
    source = f"""# Step-6 knobs. All training / eval cells read from these.
CHECKPOINT_BASE          = {checkpoint_base!r}
PRETRAIN_N_STEPS         = 200
PRETRAIN_SKIP_IF_EXISTS  = True
TRAIN_N_STEPS_SHORT      = 45
TRAIN_N_STEPS_LONG       = 125
TRAIN_SKIP_IF_EXISTS     = True
RERUN_EVAL               = False
PBE_ANCHOR_WEIGHT        = 1e-3
PBE_ANCHOR_N_POINTS      = 200
PBE_ANCHOR_SEED          = 20260421
BASIS                    = "def2-tzvp"
GRID_LEVEL               = 3

ext_data_dir       = os.path.join(CHECKPOINT_BASE, "external_data")
pretrain_dir       = os.path.join(CHECKPOINT_BASE, "pretrain")
group1_dir         = os.path.join(CHECKPOINT_BASE, "group1_h2o_short")
group2_dir         = os.path.join(CHECKPOINT_BASE, "group2_h2o_c2h2_short")
group3_dir         = os.path.join(CHECKPOINT_BASE, "group3_h2o_c2h2_long")
figures_dir        = os.path.join(CHECKPOINT_BASE, "figures")
transfer_primary   = os.path.join(CHECKPOINT_BASE, "transfer_data", "primary")
transfer_secondary = os.path.join(CHECKPOINT_BASE, "transfer_data", "secondary")
for _d in (ext_data_dir, pretrain_dir, group1_dir, group2_dir, group3_dir,
           figures_dir, transfer_primary, transfer_secondary):
    os.makedirs(_d, exist_ok=True)

print("DATA VERSION: step6-v1")
print("  Training:   {{H2O, C2H2}} + atoms {{H, O, C}}")
print("  Transfer P: {{H2, OH, CH4}} (W4-11)")
print("  Transfer S: {{NH3, HF, CO2, NH2}} (W4-11)")
print(f"  Wipe {{CHECKPOINT_BASE}}/ to regenerate")
"""
    return new_code_cell(source)


def build_cell_04_arch_table(arch_names: tuple[str, ...] | None = None):
    arch_names = arch_names or DEFAULT_ARCH_NAMES
    source = f"""ARCH_NAMES = {tuple(arch_names)!r}
print(f"Architectures ({{len(ARCH_NAMES)}}):")
for _n in ARCH_NAMES:
    _cfg = ARCHITECTURES[_n]
    print(f"  {{_n:30s}} depth={{_cfg.depth}} nodes={{_cfg.nodes}} "
          f"attention={{_cfg.attention}} descriptors={{len(_cfg.descriptors)}}")
"""
    return new_code_cell(source)


def build_cell_05_solver_table(solver_labels: tuple[str, ...] | None = None):
    solver_labels = solver_labels or DEFAULT_SOLVER_LABELS
    source = f"""SOLVER_LABELS = {tuple(solver_labels)!r}
# SolverConfig: mode uses SolverMode enum; feature_policy is FeaturePolicy enum
# or None. ONESHOT requires max_cycles=0; non-oneshot requires max_cycles>0.
SOLVER_CONFIGS = {{
    "oneshot":   SolverConfig(mode=SolverMode.ONESHOT, max_cycles=0),
    "fixed_j_3": SolverConfig(mode=SolverMode.FIXED_J, max_cycles=3),
    "full_3":    SolverConfig(mode=SolverMode.FULL, max_cycles=3,
                              feature_policy=FeaturePolicy.REASSEMBLE),
}}
print("Solver configs:")
for _lbl in SOLVER_LABELS:
    _sc = SOLVER_CONFIGS[_lbl]
    print(f"  {{_lbl:12s}} mode={{_sc.mode.value:8s}} max_cycles={{_sc.max_cycles}} "
          f"feature_policy={{_sc.feature_policy}}")
"""
    return new_code_cell(source)


def main(
    arch_names: tuple[str, ...] | None = None,
    loss_names: tuple[str, ...] | None = None,
    solver_labels: tuple[str, ...] | None = None,
    checkpoint_base: str | None = None,
    output_path: str = "notebooks/gga_training_example-step6.ipynb",
) -> nbformat.NotebookNode:
    """Assemble the step-6 notebook."""
    arch_names = arch_names or DEFAULT_ARCH_NAMES
    loss_names = loss_names or DEFAULT_LOSS_NAMES
    solver_labels = solver_labels or DEFAULT_SOLVER_LABELS
    checkpoint_base = checkpoint_base or DEFAULT_CHECKPOINT_BASE

    nb = new_notebook()
    cells = [
        build_cell_01_title(),
        build_cell_02_imports(),
        build_cell_03_constants(checkpoint_base=checkpoint_base),
        build_cell_04_arch_table(arch_names=arch_names),
        build_cell_05_solver_table(solver_labels=solver_labels),
    ]
    for idx, cell in enumerate(cells):
        cell.id = f"cell_{idx:02d}"
    nb.cells = cells

    nbformat.validate(nb)
    with open(output_path, "w") as fh:
        nbformat.write(nb, fh)
    return nb


if __name__ == "__main__":
    main()
