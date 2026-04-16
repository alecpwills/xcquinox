"""Phase 0 discovery probes for SCF solver implementation.

Each test here is a fail-fast probe verifying a spec assumption before
dependent phases are unblocked. Failures here mean the spec must be
revised, not that the test should be weakened.

See: docs/superpowers/plans/2026-04-14-alec-scf-solver-and-ref-density-rename.md
"""
import pytest
import jax
import jax.numpy as jnp

from xcquinox.alec.config import ArchitectureConfig
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.tests.fixtures.molecules import h2_molecule


def _make_h2_model_and_data(seed: int = 0):
    arch = ArchitectureConfig(
        name="probe", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    model = AlecGGAModel.from_arch(arch, seed=seed)
    data = precompute_fixed_density_data(h2_molecule())
    return model, data
