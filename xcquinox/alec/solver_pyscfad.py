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
    """Rebuild a pyscfad gto.Mole from the metadata stashed by precompute.

    Must use pyscfad.gto.Mole (not pyscf.gto.Mole) because
    pyscfad.dft.RKS requires a pyscfad-wrapped molecule object.
    """
    import pyscfad.gto
    md = mol_data["mol_metadata"]
    mol = pyscfad.gto.Mole()
    mol.atom = md["atom"]
    mol.basis = md["basis"]
    mol.charge = md["charge"]
    mol.spin = md["spin"]
    mol.verbose = 0
    mol.build()
    return mol


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
    from xcquinox.alec.descriptors import assemble_descriptor_features

    if config.mode == SolverMode.ONESHOT:
        return _oneshot_result(model, mol_data)

    import pyscfad.dft  # noqa: F401 — lazy import

    policy = config.effective_feature_policy
    descriptors = model.descriptors

    if policy == FeaturePolicy.REASSEMBLE:
        warnings.warn(
            "REASSEMBLE policy on pyscfad backend is not yet implemented; "
            "falling back to FROZEN features.",
            RuntimeWarning,
            stacklevel=2,
        )
        policy = FeaturePolicy.FROZEN

    eval_xc_callback = _make_alec_eval_xc(
        model=model,
        descriptors=descriptors,
        mol_data=mol_data,
        policy=policy,
    )

    mol = _rebuild_mol_from_mol_data(mol_data)
    mf = pyscfad.dft.RKS(mol)
    mf.define_xc_(eval_xc_callback, "GGA")
    mf.max_cycle = int(config.max_cycles)
    mf.conv_tol = float(config.conv_tol)

    if config.mode == SolverMode.FIXED_J:
        J_pinned = mol_data["j_matrix"]

        def fixed_get_j(mol_=None, dm=None, hermi=1, **kwargs):
            return J_pinned

        mf.get_j = fixed_get_j

    mf.kernel(dm0=mol_data["dm_pbe"])

    D_final = jnp.asarray(mf.make_rdm1())
    E_final = jnp.asarray(mf.e_tot)
    cycles_run = jnp.int32(getattr(mf, "cycles", config.max_cycles))
    converged = jnp.bool_(bool(mf.converged))
    features_used = assemble_descriptor_features(descriptors, mol_data)

    return SCFResult(
        density_matrix=D_final,
        total_energy=E_final,
        cycles_run=cycles_run,
        converged=converged,
        features_used=features_used,
    )
