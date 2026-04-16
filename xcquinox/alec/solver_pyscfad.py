"""xcquinox.alec.solver_pyscfad — pyscfad-based SCF backend.

Wraps pyscfad's dft.RKS with an alec-specific eval_xc callback built
from AlecGGAModel.eval_exc_scalar. pyscfad is imported lazily inside
run_pyscfad_scf so that users of the manual backend don't pay the
import cost.
"""
import warnings

import jax
import jax.numpy as jnp

from xcquinox.alec.solver import (
    SolverConfig,
    SolverMode,
    FeaturePolicy,
    SCFResult,
    _oneshot_result,
    _contract_dm_to_grid,
    _reassemble_features,
)


def _rebuild_mol_from_mol_data(mol_data: dict):
    """Rebuild a pyscf gto.Mole from the metadata stashed by precompute."""
    from pyscf import gto
    md = mol_data["mol_metadata"]
    return gto.M(
        atom=md["atom"],
        basis=md["basis"],
        charge=md["charge"],
        spin=md["spin"],
        verbose=0,
    )


def run_pyscfad_scf(config: SolverConfig, model, mol_data: dict) -> SCFResult:
    if config.mode == SolverMode.ONESHOT:
        return _oneshot_result(model, mol_data)
    raise NotImplementedError(
        "pyscfad non-ONESHOT modes are added in Tasks 6.2-6.3"
    )
