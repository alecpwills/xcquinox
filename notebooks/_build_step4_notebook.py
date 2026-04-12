"""Generator for notebooks/gga_training_example-step4.ipynb.

The step 4 notebook is **not** hand-edited as ``.ipynb`` JSON. Every cell is
produced by a ``build_cell_NN_<topic>()`` function in this module, and
``main()`` assembles the builders into an ``nbformat`` notebook, validates it,
writes it to disk, and returns the notebook object for in-process inspection.

Regeneration is deterministic: same generator source -> byte-identical
notebook. Users must never edit the ``.ipynb`` directly; all edits go through
this module. See ``docs/superpowers/plans/2026-04-12-step4-notebook-implementation.md``
for the full contract.

Naming convention: each builder returns an ``nbformat.notebooknode.NotebookNode``
(a code cell or a markdown cell). Cell-index order in ``main()`` is the order
the notebook presents to the user.
"""
import os

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


# Module-level defaults. Tests and the smoke harness override these through
# ``main()`` kwargs to produce narrow-config notebooks. Keep the names in
# ``DEFAULT_ARCH_NAMES`` synchronized with ``xcquinox.alec.ARCHITECTURES``.
DEFAULT_ARCH_NAMES = (
    "shallow",
    "shallow_attn",
    "medium",
    "medium_attn",
    "deep",
    "deep_attn",
    "deep_cusp",
    "deep_cusp_attn",
    "deep_dm",
    "deep_dm_attn",
    "deep_combined",
    "deep_combined_attn",
)

DEFAULT_LOSS_NAMES = (
    "A_atomization",
    "B_atomization_plus_dm",
    "C_atomization_plus_grid",
    "D1_delta_ae",
    "D2_delta_ae_plus_dm",
    "D3_delta_ae_plus_grid",
)

DEFAULT_CHECKPOINT_BASE = "checkpoints_step4"


def build_cell_00_smoke_marker():
    """Placeholder cell used by Task 1's scaffolding test.

    This builder exists so the ``test_main_produces_valid_notebook`` test can
    round-trip ``main()`` end-to-end before any real cell builders are added.
    It is deleted entirely in Task 12.
    """
    return new_markdown_cell("# Step 4 Notebook (generated, do not edit)")


def main(
    output_path: str,
    *,
    arch_names: tuple[str, ...] | None = None,
    loss_names: tuple[str, ...] | None = None,
    checkpoint_base: str | None = None,
):
    """Assemble the step 4 notebook, validate it, write it to ``output_path``.

    Parameters
    ----------
    output_path
        Filesystem path where the generated ``.ipynb`` is written.
    arch_names
        Optional override for ``DEFAULT_ARCH_NAMES``. Used by the smoke test
        to produce a single-architecture notebook.
    loss_names
        Optional override for ``DEFAULT_LOSS_NAMES``. Used by the smoke test
        to produce a single-loss notebook.
    checkpoint_base
        Optional override for ``DEFAULT_CHECKPOINT_BASE``. Used by the smoke
        test to redirect artifacts into a ``tmp_path``-backed directory.

    Returns
    -------
    nbformat.notebooknode.NotebookNode
        The assembled notebook, already written to disk.
    """
    if arch_names is None:
        arch_names = DEFAULT_ARCH_NAMES
    if loss_names is None:
        loss_names = DEFAULT_LOSS_NAMES
    if checkpoint_base is None:
        checkpoint_base = DEFAULT_CHECKPOINT_BASE

    nb = new_notebook()
    nb.cells = [
        build_cell_00_smoke_marker(),
    ]

    nbformat.validate(nb)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    nbformat.write(nb, output_path)
    return nb


if __name__ == "__main__":
    main("notebooks/gga_training_example-step4.ipynb")
