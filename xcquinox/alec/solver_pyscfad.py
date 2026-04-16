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


def _make_alec_eval_xc(model, descriptors, mol_data, policy):
    """Return a libxc-compatible eval_xc callback that uses alec's XC NN.

    Only FROZEN policy is supported in this task; REASSEMBLE support with
    a _current_dm_holder closure is added in Task 6.3. For FROZEN,
    features are captured at construction time and reused every cycle.
    """
    from xcquinox.alec.descriptors import assemble_descriptor_features

    if policy != FeaturePolicy.FROZEN:
        raise NotImplementedError(
            "REASSEMBLE policy in pyscfad backend is added in Task 6.3"
        )

    features_frozen = assemble_descriptor_features(descriptors, mol_data)

    def eval_xc_alec_gga(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
        rho0 = jnp.asarray(rho[0])
        dx, dy, dz = jnp.asarray(rho[1]), jnp.asarray(rho[2]), jnp.asarray(rho[3])
        sigma = dx * dx + dy * dy + dz * dz

        def eval_single(r, s, f):
            return model.eval_exc_scalar(r, s, f)

        exc_density = jax.vmap(eval_single)(rho0, sigma, features_frozen)
        exc = exc_density / (rho0 + 1e-18)

        drho_fn = lambda r, s, f: jax.grad(eval_single, argnums=0)(r, s, f)
        dsigma_fn = lambda r, s, f: jax.grad(eval_single, argnums=1)(r, s, f)
        vrho = jax.vmap(drho_fn)(rho0, sigma, features_frozen)
        vsigma = jax.vmap(dsigma_fn)(rho0, sigma, features_frozen)
        vxc = (vrho, vsigma, None, None)
        return exc, vxc, None, None

    return eval_xc_alec_gga


def run_pyscfad_scf(config: SolverConfig, model, mol_data: dict) -> SCFResult:
    if config.mode == SolverMode.ONESHOT:
        return _oneshot_result(model, mol_data)
    raise NotImplementedError(
        "pyscfad non-ONESHOT modes are added in Tasks 6.2-6.3"
    )
