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


@pytest.fixture
def o_atom_uks_polarized():
    """Same open-shell O atom but with a spin-polarization-aware cnet (P2-03)."""
    spec = MoleculeSpec(
        name="O", atom="O 0 0 0", basis="sto-3g",
        charge=0, spin=2, atom_composition=(("O", 1),), grid_level=1,
    )
    md = precompute_fixed_density_data(spec, required_keys=("eri",))
    arch = alec.ArchitectureConfig.from_spec(
        "polc_uks", 4, 32, use_polarized_correlation=True)
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    assert cnet.use_spin_polarization is True
    model = alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
    return spec, md, model


def test_uks_manual_polarized_full_runs_and_conserves_electrons(
        o_atom_uks_polarized):
    """The per-spin correlation path (P2-03) must run end-to-end in the manual
    UKS SCF and conserve n_alpha=5, n_beta=3 with a finite energy."""
    spec, md, model = o_atom_uks_polarized
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                       max_cycles=5, conv_tol=1e-5)
    result = run_scf(cfg, model, md)
    dm = np.asarray(result.density_matrix)
    assert dm.ndim == 3 and dm.shape[0] == 2
    assert np.isfinite(float(result.total_energy))
    s = np.asarray(md["s_matrix"])
    assert abs(float(np.trace(s @ dm[0])) - 5.0) < 0.1
    assert abs(float(np.trace(s @ dm[1])) - 3.0) < 0.1
    # Open-shell: per-spin correlation must keep the channels distinct.
    assert float(np.max(np.abs(dm[0] - dm[1]))) > 0.1
