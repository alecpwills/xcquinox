"""Compute random + pretrained NN baselines on the W4-11 transfer sets.

The notebook's cell 26b ("baseline_evals") only evaluates the
random-init and pretrained-only models on the TRAINED molecules
(H2O + C2H2 + their atoms). This script extends those evaluations to
the same two W4-11 transfer sets used in the analysis report:

  primary   = {CH4, H2, OH}
  secondary = {CO2, HF, NH2, NH3}

The output dataframes (``baseline_transfer_primary_df.csv`` and
``baseline_transfer_secondary_df.csv``) are written into the run dir,
mirroring the schema of the existing trained-spec transfer dataframes
(``transfer_primary_df.csv`` / ``transfer_secondary_df.csv``):

  columns: arch, baseline, molecule, value_name, value

The two baseline ``_baseline_model.eqx`` files written by cell 26b are
re-used directly (path:
``eval_baseline_{kind}/{arch}/_baseline_model.eqx``); no model is
re-initialized here, so this script's output is bit-identical to what
cell 26b produces if you were to evaluate the same models on a
transfer molecule set.

Run-agnostic by design: pass ``--run-dir`` to evaluate the
integration-pretrain origin sweep instead.

Why this script is separate from the notebook
----------------------------------------------

Cell 26b runs INSIDE the notebook and writes its outputs eagerly. We
need the same baselines on transfer mols, but adding them to the
notebook would mean re-running cell 26b's siblings as well. This
script does the minimal additional work without disturbing the
notebook's cached state.

Physical correctness
--------------------

The TestSpec we build mirrors cell 26b precisely:
  * solver_config = None  -> oneshot evaluation (no SCF)
    (a random NN cannot converge an SCF; oneshot is the only fair
    baseline)
  * metrics = (total_energy, atomization_energy, density_rmse,
               pbe_reference)
  * pbe_reference uses the same ATOMIC_ENERGIES_CHAKRAVORTY
    (Chakravorty et al. PRA 47, 3649, 1993) atomic-totals dictionary
    that the notebook generator pins at the top of cell 16.

Atomization-energy reference values for the transfer molecules are
the W4-11 literature values (Karton, Daon, Martin, *CPL* **510**, 165,
2011), pulled from the same molecule-spec definitions cell 30/31 use.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd

# Match the notebook's pre-import env config.
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # baseline evals are tiny; CPU is enough

import equinox as eqx  # noqa: E402
import jax  # noqa: E402

import xcquinox.alec as alec  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent

# These constants match the canonical step-6 notebook generator
# (notebooks/_build_step6_notebook.py); update if the generator changes.
BASIS = "def2-svp"
GRID_LEVEL = 1
ARCH_NAMES = ("deep_combined", "deep_combined_attn")

# Atomic totals (Chakravorty et al. PRA 47, 3649, 1993) as used by the
# notebook's pretrain anchor.
ATOMIC_ENERGIES_CHAKRAVORTY = {
    "H": -0.5,
    "C": -37.845,
    "N": -54.5892,
    "O": -75.0673,
    "F": -99.7339,
}

# W4-11 transfer molecules + AE references (Karton et al. CPL 510, 165, 2011).
PRIMARY_TRANSFER = (
    {"name": "CH4", "atom":
        "C 0.000000 0.000000 0.000000; "
        "H 0.628099 0.628099 0.628099; "
        "H -0.628099 -0.628099 0.628099; "
        "H 0.628099 -0.628099 -0.628099; "
        "H -0.628099 0.628099 -0.628099",
     "spin": 0, "charge": 0, "ae_ref_kcalmol": 420.420,
     "atom_composition": (("C", 1), ("H", 4))},
    {"name": "H2", "atom": "H 0 0 -0.370946; H 0 0 0.370946",
     "spin": 0, "charge": 0, "ae_ref_kcalmol": 109.493,
     "atom_composition": (("H", 2),)},
    {"name": "OH", "atom": "O 0 0 0.107851; H 0 0 -0.862809",
     "spin": 1, "charge": 0, "ae_ref_kcalmol": 107.208,
     "atom_composition": (("O", 1), ("H", 1))},
)
SECONDARY_TRANSFER = (
    {"name": "NH3", "atom":
        "N 0.000000 0.000000 0.111858; "
        "H 0.000000 0.939460 -0.260788; "
        "H 0.813632 -0.469730 -0.260788; "
        "H -0.813632 -0.469730 -0.260788",
     "spin": 0, "charge": 0, "ae_ref_kcalmol": 298.018,
     "atom_composition": (("N", 1), ("H", 3))},
    {"name": "HF", "atom": "F 0 0 0; H 0 0 0.916826",
     "spin": 0, "charge": 0, "ae_ref_kcalmol": 141.640,
     "atom_composition": (("F", 1), ("H", 1))},
    {"name": "CO2", "atom": "C 0 0 0; O 0 0 1.162; O 0 0 -1.162",
     "spin": 0, "charge": 0, "ae_ref_kcalmol": 390.141,
     "atom_composition": (("C", 1), ("O", 2))},
    {"name": "NH2", "atom":
        "N 0.000000 0.000000 0.142235; "
        "H 0.000000 0.800646 -0.494841; "
        "H 0.000000 -0.800646 -0.494841",
     "spin": 1, "charge": 0, "ae_ref_kcalmol": 182.591,
     "atom_composition": (("N", 1), ("H", 2))},
)


def _build_atom_specs(transfer_set):
    """Atoms required by every TrainingSpec/TestSpec (per-element targets).

    Returns the deduplicated tuple of {"H","O","C","N","F"} atoms that
    appear in any molecule in ``transfer_set``.
    """
    needed = set()
    for m in transfer_set:
        for elem, _n in m["atom_composition"]:
            needed.add(elem)
    atom_defs = {
        "H": {"name": "H", "atom": "H 0 0 0", "spin": 1},
        "C": {"name": "C", "atom": "C 0 0 0", "spin": 2},
        "N": {"name": "N", "atom": "N 0 0 0", "spin": 3},
        "O": {"name": "O", "atom": "O 0 0 0", "spin": 2},
        "F": {"name": "F", "atom": "F 0 0 0", "spin": 1},
    }
    return tuple(atom_defs[e] for e in sorted(needed))


def _build_mol_specs(transfer_set):
    """Convert dict molecule defs to alec.MoleculeSpec list."""
    return tuple(
        alec.MoleculeSpec(
            name=m["name"], atom=m["atom"], basis=BASIS,
            charge=m.get("charge", 0), spin=m.get("spin", 0),
            grid_level=GRID_LEVEL,
            atom_composition=m["atom_composition"],
        )
        for m in transfer_set
    )


def _build_atom_mol_specs(atom_set):
    return tuple(
        alec.MoleculeSpec(
            name=a["name"], atom=a["atom"], basis=BASIS,
            charge=0, spin=a["spin"], grid_level=GRID_LEVEL,
            atom_composition=((a["name"], 1),),
        )
        for a in atom_set
    )


def evaluate_baseline_on_transfer(
    run_dir: Path,
    arch_name: str,
    kind: str,                # "random" | "pretrained"
    transfer_set,
    set_label: str,
) -> list[dict]:
    """Run the baseline ``_baseline_model.eqx`` against ``transfer_set``.

    Reuses the exact same model checkpoint that cell 26b serialised, so
    the random-init seed and the pretrained xnet/cnet are bit-identical
    to the in-notebook baseline.
    """
    ckpt = run_dir / f"eval_baseline_{kind}" / arch_name / "_baseline_model.eqx"
    if not ckpt.is_file():
        print(f"  SKIP {arch_name}/{kind}: {ckpt} missing (run cell 26b first).")
        return []

    out_dir = run_dir / f"eval_baseline_{kind}_transfer_{set_label}" / arch_name
    out_dir.mkdir(parents=True, exist_ok=True)

    atoms = _build_atom_specs(transfer_set)
    mol_specs = _build_mol_specs(transfer_set)
    atom_specs = _build_atom_mol_specs(atoms)
    ae_ref_dict = {m["name"]: m["ae_ref_kcalmol"] for m in transfer_set}

    spec = alec.TestSpec.from_dicts(
        arch=alec.get_architecture(arch_name),
        model_checkpoint=str(ckpt),
        molecules=mol_specs + atom_specs,
        # Drop density_rmse: it requires a precomputed rho_ref_grid in
        # mol_data, which the trained-spec transfer pipeline pre-builds
        # via cells 30/31 but our standalone script does not. AE is the
        # only metric we use in the report's baseline-vs-trained
        # comparison, so this is a fair pragmatic choice -- not a
        # physical compromise.
        metrics=("total_energy", "atomization_energy", "pbe_reference"),
        metric_kwargs={
            "atomization_energy": {"reference_ae_kcalmol": ae_ref_dict},
            "pbe_reference": {
                "atom_energies": ATOMIC_ENERGIES_CHAKRAVORTY,
                "reference_ae_kcalmol": ae_ref_dict,
            },
        },
        atom_energies=ATOMIC_ENERGIES_CHAKRAVORTY,
        output_dir=str(out_dir),
        solver_config=None,
        pbe_anchor_weight=0.0, pbe_anchor_sample=None,
    )
    alec.run_test(spec)
    jax.clear_caches(); gc.collect()

    # Read per_molecule.json into the report-friendly shape.
    per_mol = out_dir / "per_molecule.json"
    if not per_mol.is_file():
        return []
    with per_mol.open() as f:
        rows = json.load(f)
    out = []
    for r in rows:
        mol = r.get("name") or r.get("molecule")
        if mol not in {m["name"] for m in transfer_set}:
            # Filter out the atom rows; only molecules go into the
            # baseline_transfer_*_df we use for plotting.
            continue
        for k, v in r.items():
            if k in ("name", "molecule") or isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                out.append({
                    "arch": arch_name, "baseline": kind,
                    "molecule": mol, "value_name": k, "value": float(v),
                })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path,
        default=REPO / "notebooks" / "checkpoints_step6" / "unweighted",
        help="Step-6 run directory (must contain eval_baseline_{random,pretrained}/...).",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    if not run_dir.is_dir():
        print(f"run_dir does not exist: {run_dir}", file=sys.stderr)
        return 1

    print(f"Computing baseline transfer evals for {run_dir}")

    for set_label, transfer_set in (("primary", PRIMARY_TRANSFER),
                                    ("secondary", SECONDARY_TRANSFER)):
        out_csv = run_dir / f"baseline_transfer_{set_label}_df.csv"
        rows: list[dict] = []
        for arch in ARCH_NAMES:
            for kind in ("random", "pretrained"):
                print(f"  {arch} / {kind} / {set_label} ...")
                rows.extend(evaluate_baseline_on_transfer(
                    run_dir, arch, kind, transfer_set, set_label,
                ))
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(f"  wrote {out_csv}  ({len(df)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
