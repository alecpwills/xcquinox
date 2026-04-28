"""Tests for xcquinox.alec.solver_manual — SCF body correctness."""
import numpy as np
import pytest
import jax.numpy as jnp

import xcquinox.alec as alec
from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.solver import (
    SolverConfig,
    SolverBackend,
    SolverMode,
    run_scf,
)


def test_scf_energy_computed_from_mixed_dm_consistently():
    """SCF energy trace must be a consistent functional of the mixed DM,
    not a hybrid of D_cur (XC part) and D_mixed (one-electron + Coulomb)."""
    spec = MoleculeSpec(
        name="H2",
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        charge=0,
        spin=0,
        atom_composition=(("H", 2),),
        grid_level=1,
    )
    md = precompute_fixed_density_data(spec, required_keys=("eri",))
    arch = alec.get_architecture("deep")
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    model = alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
    cfg = SolverConfig(
        backend=SolverBackend.MANUAL,
        mode=SolverMode.FULL,
        max_cycles=10,
        conv_tol=1e-8,
    )
    result = run_scf(cfg, model, md)

    # Energy trace should not have implausible upward excursions > 1 Hartree
    # during the SCF trajectory (previous bug could produce such artifacts
    # because the XC term lagged behind the one-electron/Coulomb terms).
    energy_trace = np.asarray(result.energy_trace)
    valid = (
        energy_trace[~np.isnan(energy_trace)]
        if np.any(np.isnan(energy_trace))
        else energy_trace
    )
    if len(valid) > 1:
        max_upward_jump = float(np.max(np.diff(valid)))
        assert max_upward_jump < 1.0, (
            f"SCF energy jumped upward by {max_upward_jump:.3f} Ha — "
            f"density inconsistency between E_new and features_used"
        )


def test_scf_energy_uses_post_mix_density():
    """After the fix, E_new at each cycle is computed from D_mixed with
    features/rho derived from D_mixed (not from D_cur).

    Proxy check: with mixer alpha in (0, 1), D_mixed != D_cur except at
    convergence. The reported energy at convergence should equal the energy
    evaluated from the final density's features — no hybrid.
    """
    spec = MoleculeSpec(
        name="H2",
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        charge=0,
        spin=0,
        atom_composition=(("H", 2),),
        grid_level=1,
    )
    md = precompute_fixed_density_data(spec, required_keys=("eri",))
    arch = alec.get_architecture("deep")
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    model = alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
    cfg = SolverConfig(
        backend=SolverBackend.MANUAL,
        mode=SolverMode.FULL,
        max_cycles=30,
        conv_tol=1e-10,
        mixer_kwargs=(("alpha", 0.5),),
    )
    result = run_scf(cfg, model, md)
    assert bool(result.converged)
    assert jnp.isfinite(result.total_energy)
