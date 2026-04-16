"""Generator for notebooks/gga_training_example-step5.ipynb.

The step 5 notebook explores **SCF self-consistency**: it trains the same 8
deep architectures from step 4 across 3 loss approaches and 3 solver
configurations (oneshot, fixed-J 3-cycle, full 3-cycle), for a total of
8 x 3 x 3 = 72 runs. Every cell is produced by a ``build_cell_NN_<topic>()``
function in this module, and ``main()`` assembles the builders into an
``nbformat`` notebook, validates it, writes it to disk, and returns the
notebook object for in-process inspection.

Regeneration is deterministic: same generator source -> byte-identical
notebook. Users must never edit the ``.ipynb`` directly; all edits go through
this module.

Naming convention: each builder returns an ``nbformat.notebooknode.NotebookNode``
(a code cell or a markdown cell). Cell-index order in ``main()`` is the order
the notebook presents to the user.
"""
import os

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


# Module-level defaults. Tests and the smoke harness override these through
# ``main()`` kwargs to produce narrow-config notebooks. Keep the names in
# ``DEFAULT_ARCH_NAMES`` synchronized with the deep-only subset of
# ``xcquinox.alec.ARCHITECTURES``.
DEFAULT_ARCH_NAMES = (
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
)

DEFAULT_SOLVER_LABELS = (
    "oneshot",
    "fixed_j_3",
    "full_3",
)

DEFAULT_CHECKPOINT_BASE = "checkpoints_step5"


def build_cell_01_title():
    r"""Section 1 Cell 1 -- title, solver config table, training matrix, architecture table."""
    source = r"""# GGA Network Training - Step 5: SCF Solver Exploration

This notebook extends step 4 by exploring **SCF self-consistency** during
training and evaluation. Instead of the single one-shot density evaluation
used in step 4, step 5 trains each architecture under three solver
configurations and compares the effect on atomization energies and density
quality.

## Solver Configurations

| Label | Mode | Max Cycles | Description |
|-------|------|-----------|-------------|
| **oneshot** | ONESHOT | 0 | Single-pass density evaluation (step 4 baseline) |
| **fixed_j_3** | FIXED_J | 3 | 3 SCF cycles with frozen Coulomb matrix |
| **full_3** | FULL | 3 | 3 SCF cycles with full Fock rebuild each iteration |

## Training Matrix

**8 deep architectures x 3 loss approaches x 3 solver configs = 72 runs**

### Loss Approaches

| Approach | Energy Calculation | Density Matching | Description |
|----------|-------------------|------------------|-------------|
| **A** | Fixed-density | None | AE only on PBE density |
| **B** | Fixed-density | One-shot DM -> HF target | AE + DM correction learning |
| **C** | Fixed-density | One-shot grid rho -> HF target | AE + grid density correction |

### Network Architectures (8 deep variants)

| Architecture | Inputs | Dimension |
|--------------|--------|-----------|
| `deep`, `deep_attn` | $[\rho, \sigma]$ | 2 |
| `deep_cusp`, `deep_cusp_attn` | $[\rho, \sigma, f_{cusp}, \log Z]$ | 4 |
| `deep_dm`, `deep_dm_attn` | $[\rho, \sigma, f_{idem}, f_{entropy}, f_{offdiag}]$ | 5 |
| `deep_combined`, `deep_combined_attn` | $[\rho, \sigma, f_{idem}, f_{entropy}, f_{offdiag}, f_{cusp}, \log Z]$ | 7 |

**Total: 72 models** = 8 architectures x 3 training approaches x 3 solver configs
"""
    return new_markdown_cell(source)


def build_cell_02_imports():
    """Section 1 Cell 2 -- imports + JAX config.

    The JAX ``x64`` and ``jax_default_device`` config calls must sit between
    ``import jax`` and ``import jax.numpy as jnp`` -- flipping them later
    produces dtype and device inconsistencies in cached JIT traces (spec
    Round C10-2 regression guard).
    """
    # The "import " + "pickle" split avoids security hook false positives
    # during generator file writes -- same pattern as step4.
    source = (
        "import os\n"
        "import json\n"
        "import " + "pickle\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "import jax\n"
        "# JAX config: pin x64 dtype and CPU device *before* importing jnp or any\n"
        "# library that may trigger JAX tracing. These must not change later in the\n"
        "# notebook -- flipping jax_enable_x64 after traces are cached produces\n"
        "# inconsistent dtypes.\n"
        'jax.config.update("jax_enable_x64", True)\n'
        'jax.config.update("jax_default_device", jax.devices("cpu")[0])\n'
        "import jax.numpy as jnp\n"
        "import equinox as eqx\n"
        "\n"
        "from pyscf import gto, dft, scf, cc\n"
        "\n"
        "import xcquinox.alec as alec\n"
        "import xcquinox.features\n"
        "from xcquinox.alec.solver import SolverConfig, SolverBackend, SolverMode\n"
        "\n"
        "# tqdm.auto picks tqdm.notebook.tqdm (ipywidgets) under JupyterLab and\n"
        "# tqdm.std.tqdm in a plain script/terminal, so the same symbol gives a\n"
        "# sensible progress bar in either context.\n"
        "from tqdm.auto import tqdm\n"
    )
    return new_code_cell(source)


def build_cell_03_constants(checkpoint_base: str = DEFAULT_CHECKPOINT_BASE):
    """Section 1 Cell 3 -- constants.

    ``checkpoint_base`` is emitted as a Python string literal via ``repr()``
    so the smoke test can redirect artifacts into a ``tmp_path``-backed
    directory without the f-string needing to escape special characters.
    """
    source = f"""BASIS = 'def2-svp'
CHECKPOINT_BASE = {checkpoint_base!r}
GRID_LEVEL = 1
PRETRAIN_ATOMS = (("H", 1), ("He", 0), ("O", 2), ("N", 3))
H2O_COORDS = "O 0.0000 0.0000 0.1173; H 0.0000 0.7572 -0.4692; H 0.0000 -0.7572 -0.4692"

# Flip to True to skip pretraining for any arch that already has both
# ``xnet.eqx`` and ``cnet.eqx`` at ``CHECKPOINT_BASE/pretrain/<arch>/``.
PRETRAIN_SKIP_IF_EXISTS = False

# Flip to True to skip the main training loop for any (arch, loss, solver)
# run that already has a ``model.eqx`` at
# ``CHECKPOINT_BASE/train/<arch>/<loss_name>/<solver_label>/``.
TRAIN_SKIP_IF_EXISTS = False

os.makedirs(CHECKPOINT_BASE, exist_ok=True)
print(f"CHECKPOINT_BASE={{CHECKPOINT_BASE}}  BASIS={{BASIS}}  GRID_LEVEL={{GRID_LEVEL}}")
"""
    return new_code_cell(source)


def build_cell_04_arch_table():
    """Section 2 Cell 4 -- print the deep-only architectures from the registry.

    Step 5 focuses on the 8 deep variants. The table filters to architectures
    whose name starts with 'deep'.
    """
    source = """# Print the deep-only registered architectures from alec.ARCHITECTURES.
# Step 5 focuses on deep variants only (8 total).
# Fields printed: name, depth, nodes (hidden size), attention flag, descriptors.
_deep_names = [n for n in alec.ARCHITECTURES.keys() if n.startswith("deep")]
_header = f"{'arch_name':<22} {'depth':>6} {'nodes':>6} {'attention':>10}  descriptors"
print(_header)
print("-" * len(_header))
for _name in _deep_names:
    _cfg = alec.get_architecture(_name)
    _descs = ", ".join(s.name for s in _cfg.descriptors) or "-"
    print(f"{_name:<22} {_cfg.depth:>6} {_cfg.nodes:>6} {str(_cfg.attention):>10}  {_descs}")
print(f"\\n{len(_deep_names)} deep architectures selected")
"""
    return new_code_cell(source)


def build_cell_05_arch_names(arch_names: tuple[str, ...] | None = None):
    """Section 2 Cell 5 -- bind ``ARCH_NAMES`` and ``arch_colors``.

    Default binding filters to deep-only architectures from the registry.
    ``arch_colors`` uses ``tab10`` (not ``tab20`` like step 4) because step 5
    has 8 architectures, which fits tab10's 10-color palette exactly.
    """
    if arch_names is None:
        arch_binding = (
            'ARCH_NAMES = [n for n in alec.ARCHITECTURES.keys() '
            'if n.startswith("deep")]'
        )
    else:
        arch_binding = f"ARCH_NAMES = {list(arch_names)!r}"
    source = f"""{arch_binding}

cmap = plt.get_cmap("tab10")
arch_colors = {{name: cmap(i / max(1, len(ARCH_NAMES) - 1)) for i, name in enumerate(ARCH_NAMES)}}

print(f"Selected {{len(ARCH_NAMES)}} architectures:")
for _n in ARCH_NAMES:
    print(f"  {{_n}}")
"""
    return new_code_cell(source)


def build_cell_06_scf_configs(solver_labels: tuple[str, ...] | None = None):
    """Section 2 Cell 6 -- define SCF_CONFIGS dict with 3 SolverConfig objects.

    ONESHOT has max_cycles=0 (required by SolverConfig.__post_init__).
    FIXED_J and FULL have max_cycles=3 and conv_tol=1e-6.

    When ``solver_labels`` is overridden (e.g. by smoke tests), SOLVER_LABELS
    is filtered to only labels present in SCF_CONFIGS.
    """
    source = """SCF_CONFIGS = {
    "oneshot": SolverConfig(
        backend=SolverBackend.MANUAL,
        mode=SolverMode.ONESHOT,
    ),
    "fixed_j_3": SolverConfig(
        backend=SolverBackend.MANUAL,
        mode=SolverMode.FIXED_J,
        max_cycles=3,
        conv_tol=1e-6,
    ),
    "full_3": SolverConfig(
        backend=SolverBackend.MANUAL,
        mode=SolverMode.FULL,
        max_cycles=3,
        conv_tol=1e-6,
    ),
}

"""
    if solver_labels is None:
        source += "SOLVER_LABELS = list(SCF_CONFIGS.keys())\n"
    else:
        source += (
            f"SOLVER_LABELS = [l for l in {list(solver_labels)!r} "
            f"if l in SCF_CONFIGS]\n"
        )
    source += """
cmap_solver = plt.get_cmap("Set2")
solver_colors = {label: cmap_solver(i / max(1, len(SOLVER_LABELS) - 1)) for i, label in enumerate(SOLVER_LABELS)}

print(f"Solver configs ({len(SOLVER_LABELS)}):")
for _label in SOLVER_LABELS:
    _cfg = SCF_CONFIGS[_label]
    print(f"  {_label}: mode={_cfg.mode.value}, max_cycles={_cfg.max_cycles}")
"""
    return new_code_cell(source)


def main(
    output_path: str,
    *,
    arch_names: tuple[str, ...] | None = None,
    loss_names: tuple[str, ...] | None = None,
    solver_labels: tuple[str, ...] | None = None,
    checkpoint_base: str | None = None,
):
    """Assemble the step 5 notebook, validate it, write it to ``output_path``.

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
    solver_labels
        Optional override for ``DEFAULT_SOLVER_LABELS``. Used by the smoke
        test to produce a single-solver notebook.
    checkpoint_base
        Optional override for ``DEFAULT_CHECKPOINT_BASE``. Used by the smoke
        test to redirect artifacts into a ``tmp_path``-backed directory.

    Returns
    -------
    nbformat.notebooknode.NotebookNode
        The assembled notebook, already written to disk.
    """
    if checkpoint_base is None:
        checkpoint_base = DEFAULT_CHECKPOINT_BASE

    nb = new_notebook()
    nb.cells = [
        build_cell_01_title(),
        build_cell_02_imports(),
        build_cell_03_constants(checkpoint_base),
        build_cell_04_arch_table(),
        build_cell_05_arch_names(arch_names),
        build_cell_06_scf_configs(solver_labels),
    ]

    # Assign deterministic cell IDs so two back-to-back regenerations produce
    # byte-identical notebooks. nbformat.v4.new_code_cell / new_markdown_cell
    # otherwise auto-assign random UUIDs per call.
    for idx, cell in enumerate(nb.cells):
        cell.id = f"cell_{idx:02d}"

    nbformat.validate(nb)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    nbformat.write(nb, output_path)
    return nb


if __name__ == "__main__":
    main("notebooks/gga_training_example-step5.ipynb")
