"""Differentiability tests for the manual SCF backend.

Covers caveat 5 from the design spec: the full D -> rho -> vxc -> F -> D chain
must flow jax.grad through jax.lax.scan.
"""
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from xcquinox.alec.config import ArchitectureConfig, FeatureSpec
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.solver import (
    SolverConfig, SolverBackend, SolverMode, FeaturePolicy, run_scf,
)
from xcquinox.alec.tests.fixtures.molecules import h2_molecule


def _make_h2_full():
    arch = ArchitectureConfig(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    model = AlecGGAModel.from_arch(arch, seed=0)
    data = precompute_fixed_density_data(h2_molecule(), required_keys=("eri",))
    return model, data


def test_grad_through_2_cycle_fixed_j_is_finite():
    """jax.grad through a 2-cycle FIXED_J SCF should produce finite gradients."""
    model, data = _make_h2_full()
    cfg = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FIXED_J,
        max_cycles=2, conv_tol=1e-12,
    )

    def total_energy_fn(m):
        return run_scf(cfg, m, data).total_energy

    grads = eqx.filter_grad(total_energy_fn)(model)
    # xnet + cnet each have parameter trees; check at least one finite gradient
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert len(leaves) > 0
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf))


def test_grad_through_3_cycle_full_is_finite():
    """jax.grad through a 3-cycle FULL SCF with REASSEMBLE features."""
    model, data = _make_h2_full()
    cfg = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
        max_cycles=3, conv_tol=1e-12,
    )

    def total_energy_fn(m):
        return run_scf(cfg, m, data).total_energy

    grads = eqx.filter_grad(total_energy_fn)(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert len(leaves) > 0
    for leaf in leaves:
        assert jnp.all(jnp.isfinite(leaf))
