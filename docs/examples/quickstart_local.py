#!/usr/bin/env python
"""Local quick-start for the xcquinox.alec ML exchange-correlation trainer.

Trains a tiny neural XC functional on three molecules (H, O, H2O) and then
evaluates it -- entirely on your laptop, CPU, in ~1 minute. No cluster, no
pretrained checkpoint, no reference-data files. It exists so you can watch the
whole train -> evaluate loop run with your own eyes before scaling up.

Run it:
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python docs/examples/quickstart_local.py

The narrated, step-by-step walkthrough of this script is in docs/user_guide.md.
"""
import os
# Quantum chemistry needs double precision; pin CPU so the example is
# deterministic and dependency-light. (Set before JAX initialises.)
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import tempfile
import numpy as np

from xcquinox.alec.config import (
    TrainingSpec, TestSpec, MoleculeSpec, get_architecture,
)
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.train import run_training
from xcquinox.alec.evaluation import run_test


def main() -> str:
    # ---- 1. Define the molecules -------------------------------------------
    # A MoleculeSpec is a PySCF atom string + spin/charge + an element count.
    # spin is 2*S (number of unpaired electrons): H and O are open-shell atoms.
    H = MoleculeSpec(name="H", atom="H 0 0 0", basis="sto-3g",
                     charge=0, spin=1, atom_composition=(("H", 1),))
    O = MoleculeSpec(name="O", atom="O 0 0 0", basis="sto-3g",
                     charge=0, spin=2, atom_composition=(("O", 1),))
    H2O = MoleculeSpec(name="H2O", atom="O 0 0 0; H 0 0 0.96; H 0.96 0 0",
                       basis="sto-3g", charge=0, spin=0,
                       atom_composition=(("H", 2), ("O", 1)))
    mols = (H, O, H2O)

    # ---- 2. Precompute the fixed-density data ------------------------------
    # One PBE SCF per molecule, caching the density, integration grid, one- and
    # two-electron integrals, etc. The trainer reuses this so it never re-runs
    # a SCF just to read the density.
    md = {m.name: precompute_fixed_density_data(m) for m in mols}

    # ---- 3. Reference targets + atomic anchors -----------------------------
    # In a real run these are CCSD / experimental references. For a runnable
    # demo we use each molecule's own PBE energy as a stand-in "reference", and
    # the PBE atomization energy of water (E_atoms - E_molecule).
    atom_energies = {"H": float(md["H"]["E_pbe"]), "O": float(md["O"]["E_pbe"])}
    ae_h2o = 2 * md["H"]["E_pbe"] + md["O"]["E_pbe"] - md["H2O"]["E_pbe"]
    targets = {
        "H": float(md["H"]["E_pbe"]),
        "O": float(md["O"]["E_pbe"]),
        "H2O": float(max(ae_h2o, 1e-3)),
    }

    # ---- 4. Build a TrainingSpec -------------------------------------------
    # Pick a named architecture, a loss, and how long to train. `A_atomization`
    # fits atomization energies -- the simplest of the registered losses.
    workdir = tempfile.mkdtemp(prefix="xcq_quickstart_")
    ckpt = os.path.join(workdir, "ckpt")
    spec = TrainingSpec.from_dicts(
        arch=get_architecture("deep"),       # depth-4, 32-node x/c MLPs
        molecules=mols, targets=targets, atom_energies=atom_energies,
        loss_name="A_atomization",
        n_steps=100, lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.0,
        grad_clip=1.0, checkpoint_dir=ckpt, seed=42,
    )
    spec.validate()                           # fail fast on an inconsistent spec

    # ---- 5. Train ----------------------------------------------------------
    # Writes model.eqx / losses.npy / aux_log.pkl / train_metadata.json to ckpt/.
    print("Training (100 steps on H, O, H2O) ...")
    run_training(spec)
    losses = np.load(os.path.join(ckpt, "losses.npy"))
    print(f"  loss: {losses[0]:.4e}  ->  {losses[-1]:.4e}   ({len(losses)} steps)")
    assert np.all(np.isfinite(losses)), "training produced a non-finite loss"
    assert losses[-1] < losses[0], "loss did not decrease"

    # ---- 6. Evaluate the trained model -------------------------------------
    # Re-load model.eqx and score it on the same molecules (in-sample). A real
    # run would also score a held-out benchmark; see docs/user_guide.md.
    test = TestSpec.from_dicts(
        model_checkpoint=os.path.join(ckpt, "model.eqx"),
        arch=get_architecture("deep"),
        molecules=mols,
        metrics=("total_energy", "atomization_energy"),
        atom_energies=atom_energies,
        output_dir=os.path.join(workdir, "eval"),
    )
    test.validate()
    results = run_test(test)
    print(f"  evaluated {len(results['per_molecule'])} molecules; "
          f"aggregate keys: {sorted(results['aggregate'])}")

    print(f"\nDone. Artifacts under {workdir}")
    return workdir


if __name__ == "__main__":
    main()
