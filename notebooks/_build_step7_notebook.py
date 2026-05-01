"""Generate gga_training_example-step7.ipynb.

Pattern matches notebooks/_build_step{4,5,6}_notebook.py: assemble cells
into nbformat structure, write .ipynb, optionally execute end-to-end.

Step-7 plan:
  - Build Dick 2021 SI II training pool (28 entries) via
    xcquinox.alec.dick_pool.build_dick_pool
  - Extract (rho^{1/3}, s, alpha) per species, cache to disk
  - Build reference 3-histogram via build_reference_histograms
  - Sweep r in {1,2,3,4,5,6,7,12,15,18,21} x {l2,jsd} x {oneshot,full_3}
    x {with_hbpt, no_hbpt} = 88 training runs
  - Post-process: 6+1 figures + headline.json

Citations:
  - PBE 1996  (PRL 77, 3865) -- descriptor s
  - SCAN 2015 (PRL 115, 036402) -- descriptor alpha
  - Lin 1991  (IEEE TIT 37, 145) -- JSD
  - Chen 2018 (arXiv:1711.02257) -- GradNorm
  - Dick 2021 (PRB 104, L161109) -- candidate pool
"""
from __future__ import annotations

import nbformat as nbf
from pathlib import Path

NOTEBOOK_OUT = Path(__file__).resolve().parent / "gga_training_example-step7.ipynb"


def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def _code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


# Constants -------------------------------------------------------------
SUBSET_SIZES = (1, 2, 3, 4, 5, 6, 7, 12, 15, 18, 21)
METRICS = ("l2", "jsd")
SOLVERS = ("oneshot", "full_3")
AUGMENTATIONS = (False, True)  # with_hbpt False/True
ARCH_NAME = "deep_combined_attn"
LOSS_NAME = "L5_gradnorm_vxc_step7"
PRETRAIN_ORIGIN = "integration"
TRAIN_N_STEPS = 250
LR_START, LR_END = 1e-2, 1e-5
LR_DECAY_START = 0.2
GRAD_CLIP = 1.0


def build_cells() -> list:
    cells: list = []
    cells.append(_md(
        "# Step-7: Histogram-Matched Subset Selection from Dick 2021 Training Pool\n\n"
        "Generate optimally-representative training subsets (1..21 size sweep) by\n"
        "minimizing distance between candidate-subset and full-pool histograms over\n"
        "$(\\rho^{1/3}, s, \\alpha)$.\n\n"
        "**Critical:** $\\alpha$ enters the subset-selection objective only -- the\n"
        "trained GGA network does NOT consume it. Future MGGA extension is step-8+.\n\n"
        "Reference: Dick & Fernandez-Serra, *Phys. Rev. B* **104**, L161109 (2021), SI II.\n"
    ))
    cells.append(_code(
        "from xcquinox.alec import subset_selection as ss\n"
        "from xcquinox.alec import dick_pool\n"
        "from xcquinox.alec import losses\n"
        "import numpy as np\n"
        "from pathlib import Path\n\n"
        "REPO = Path('/home/awills/Documents/Research/xcquinox')\n"
        "STEP7_ROOT = REPO / 'notebooks' / 'checkpoints_step7'\n"
        "DESCRIPTOR_CACHE = STEP7_ROOT / 'subset_descriptors'\n"
        "REF_HIST_CACHE = STEP7_ROOT / 'dick_pool_full_hist'\n"
        "DESCRIPTOR_CACHE.mkdir(parents=True, exist_ok=True)\n"
        "REF_HIST_CACHE.mkdir(parents=True, exist_ok=True)\n\n"
        "pool = dick_pool.build_dick_pool()\n"
        "print(f'Dick 2021 SI II training pool: {pool[\"n_total\"]} entries')\n"
        "print(f'  AE molecules: {len(pool[\"ae_molecules\"])}')\n"
        "print(f'  BH76 reactions: {len(pool[\"bh76_reactions\"])}')\n"
        "print(f'  IP13 pairs: {len(pool[\"ip13_pairs\"])}')\n"
        "print(f'  Atom refs: {len(pool[\"atom_refs\"])}')\n"
    ))
    return cells


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = build_cells()
    NOTEBOOK_OUT.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"wrote {NOTEBOOK_OUT}")


if __name__ == "__main__":
    main()
