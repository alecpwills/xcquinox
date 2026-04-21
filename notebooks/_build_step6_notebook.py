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
    cells = []
    for idx, cell in enumerate(cells):
        cell.id = f"cell_{idx:02d}"
    nb.cells = cells

    nbformat.validate(nb)
    with open(output_path, "w") as fh:
        nbformat.write(nb, fh)
    return nb


if __name__ == "__main__":
    main()
