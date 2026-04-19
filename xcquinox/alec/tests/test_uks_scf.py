"""UKS SCF tests — MANUAL backend FIXED_J and FULL modes.

Regression tests for Task 11 of the alec physics-fixes plan: run_manual_scf
must handle UKS molecules via spin-resolved Fock matrices (F_a, F_b) using
the Task 10 `_uks_spin_resolved_vxc` helper.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import xcquinox.alec as alec
from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.solver import SolverConfig, SolverBackend, SolverMode, run_scf


@pytest.fixture
def o_atom_uks():
    spec = MoleculeSpec(
        name="O", atom="O 0 0 0", basis="sto-3g",
        charge=0, spin=2, atom_composition=(("O", 1),), grid_level=1,
    )
    md = precompute_fixed_density_data(spec, required_keys=("eri",))
    arch = alec.get_architecture("deep")
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    model = alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
    return spec, md, model


def test_uks_manual_oneshot_shape(o_atom_uks):
    """UKS oneshot via MANUAL backend must return (2, nao, nao) DM."""
    spec, md, model = o_atom_uks
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.ONESHOT)
    result = run_scf(cfg, model, md)
    dm = np.asarray(result.density_matrix)
    assert dm.ndim == 3 and dm.shape[0] == 2, f"expected UKS shape, got {dm.shape}"


def test_uks_manual_fixed_j_runs(o_atom_uks):
    """FIXED_J UKS SCF on O atom must return a (2, nao, nao) DM."""
    spec, md, model = o_atom_uks
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FIXED_J,
                       max_cycles=3, conv_tol=1e-5)
    result = run_scf(cfg, model, md)
    dm = np.asarray(result.density_matrix)
    assert dm.ndim == 3 and dm.shape[0] == 2


def test_uks_manual_full_electrons_correct(o_atom_uks):
    """FULL UKS SCF must preserve n_alpha=5, n_beta=3 for O."""
    spec, md, model = o_atom_uks
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                       max_cycles=5, conv_tol=1e-5)
    result = run_scf(cfg, model, md)
    dm = np.asarray(result.density_matrix)
    s = np.asarray(md["s_matrix"])
    n_a = float(np.trace(s @ dm[0]))
    n_b = float(np.trace(s @ dm[1]))
    assert abs(n_a - 5.0) < 0.1, f"n_alpha = {n_a}"
    assert abs(n_b - 3.0) < 0.1, f"n_beta = {n_b}"


def test_uks_manual_full_dms_distinct(o_atom_uks):
    """Alpha and beta DMs must differ substantially (open-shell physics)."""
    spec, md, model = o_atom_uks
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                       max_cycles=5, conv_tol=1e-5)
    result = run_scf(cfg, model, md)
    dm = np.asarray(result.density_matrix)
    assert float(np.max(np.abs(dm[0] - dm[1]))) > 0.1
