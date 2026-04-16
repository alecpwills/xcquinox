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


def test_manual_fixed_j_converges_on_h2():
    """H2/STO-3G fixed_j should converge in <=10 cycles at default tol."""
    model, data = _make_h2()
    cfg = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FIXED_J,
        max_cycles=10, conv_tol=1e-8,
    )
    result = run_scf(cfg, model, data)
    assert bool(result.converged) is True
    assert int(result.cycles_run) <= 10
    assert int(result.cycles_run) >= 1
    assert jnp.isfinite(result.total_energy)


def test_manual_full_converges_on_h2_with_eri():
    """FULL mode requires the eri tensor in mol_data; test converges in <=15 cycles."""
    from xcquinox.alec.config import ArchitectureConfig, FeatureSpec
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.data import precompute_fixed_density_data

    arch = ArchitectureConfig(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(FeatureSpec.of("cusp"), FeatureSpec.of("dm_statistics")),
        x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    model = AlecGGAModel.from_arch(arch, seed=0)
    data = precompute_fixed_density_data(
        h2_molecule(),
        descriptors=arch.materialize_descriptors(),
        required_keys=("eri",),
    )
    cfg = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
        max_cycles=15, conv_tol=1e-6,
    )
    result = run_scf(cfg, model, data)
    assert bool(result.converged) is True
    assert jnp.isfinite(result.total_energy)
