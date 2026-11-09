"""Builder for notebooks/registry_pt_prong_B.ipynb: the registry
proton-transfer case study at the production identity, prong B (gradient
accumulation: the twelve group gradients averaged into one optimizer step per
epoch). The cells live in ``_registry_pt_prongs_cells.py``, shared with
prong A.

Regenerating:  python notebooks/_build_registry_pt_prong_B_notebook.py
The tracked .ipynb is the byte-exact product of this builder, enforced by
notebooks/test_notebook_hygiene.py.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _cells_module():
    spec = importlib.util.spec_from_file_location(
        "_registry_pt_prongs_cells", str(_HERE / "_registry_pt_prongs_cells.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(output_path=None):
    if output_path is None:
        output_path = str(_HERE / "registry_pt_prong_B.ipynb")
    return _cells_module().write_notebook(output_path, "B")


if __name__ == "__main__":
    main()
