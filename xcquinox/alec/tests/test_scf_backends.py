"""Integration tests for SCF backends.

Golden system: H2/STO-3G (2 AOs, 1 occ). Runs in milliseconds.
"""
import pytest
import jax.numpy as jnp

from xcquinox.alec.config import ArchitectureConfig
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.solver import (
    SolverConfig, SolverBackend, SolverMode, FeaturePolicy, run_scf,
)
from xcquinox.alec.oneshot import fixed_density_total_energy
from xcquinox.alec.tests.fixtures.molecules import h2_molecule


def _make_h2():
    arch = ArchitectureConfig(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    model = AlecGGAModel.from_arch(arch, seed=0)
    data = precompute_fixed_density_data(h2_molecule())
    return model, data


def test_manual_oneshot_matches_legacy():
    """manual backend, oneshot mode, zero cycles — byte-identical to legacy path."""
    model, data = _make_h2()
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.ONESHOT)
    result = run_scf(cfg, model, data)
    e_legacy = float(fixed_density_total_energy(model, data))
    assert float(result.total_energy) == pytest.approx(e_legacy, abs=1e-12)
    assert int(result.cycles_run) == 0
    assert bool(result.converged) is True
