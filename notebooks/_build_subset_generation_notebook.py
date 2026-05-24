"""Generate gga_subset_generation.ipynb.

Standalone subset-SELECTION pre-process, split out of the step-7 training
notebook (`_build_step7_notebook.py`). Reuses that module's shared cell
builders via ``build_cells(mode="subset")`` so the config/import cell and the
descriptor-extraction + reference-histogram + selection-sweep cells stay in
ONE place and cannot drift between the two notebooks.

This notebook PRODUCES the `subset_index_log.json` ledger + per-spec
`subset.traj` files (under the C4-03 alpha-mode `STEP7_ROOT` subdir); the
step-7 training notebook and the SLURM harness CONSUME them read-only.

Citations (inherited from the shared cells):
  - PBE 1996  (PRL 77, 3865)      -- descriptor s
  - SCAN 2015 (PRL 115, 036402)   -- descriptor alpha
  - Lin 1991  (IEEE TIT 37, 145)  -- JSD
  - Dick 2021 (PRB 104, L161109)  -- candidate pool
"""
from __future__ import annotations

import nbformat as nbf
from pathlib import Path

from _build_step7_notebook import build_cells

NOTEBOOK_OUT = Path(__file__).resolve().parent / "gga_subset_generation.ipynb"


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = build_cells(mode="subset")
    NOTEBOOK_OUT.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"wrote {NOTEBOOK_OUT}")


if __name__ == "__main__":
    main()
