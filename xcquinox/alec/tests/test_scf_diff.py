"""Differentiability tests for the manual SCF backend.

Covers caveat 5 from the design spec: the full D -> rho -> vxc -> F -> D chain
must flow jax.grad through jax.lax.scan.
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from xcquinox.alec.config import ArchitectureConfig, FeatureSpec
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.solver import (
    SolverConfig, SolverBackend, SolverMode, FeaturePolicy, run_scf,
)
from xcquinox.alec.tests.fixtures.molecules import h2_molecule, h_atom


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


# ---------------------------------------------------------------------------
# forward_only (eval) de-fuse: the Python-loop SCF must give the SAME result as
# the jax.lax.scan SCF. This is what lets eval skip the giant fused per-molecule
# XLA compile without changing any number the demo/cluster report.
# ---------------------------------------------------------------------------

def _make_h_atom_full():
    arch = ArchitectureConfig(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    model = AlecGGAModel.from_arch(arch, seed=0)
    data = precompute_fixed_density_data(h_atom(), required_keys=("eri",))
    return model, data


def _assert_scf_result_equal(a, b):
    # forward_only runs the SAME body as the scan, so results match to XLA float
    # reassociation (fused scan vs eager ops), not machine-epsilon-exactly.
    np.testing.assert_allclose(
        np.asarray(a.density_matrix), np.asarray(b.density_matrix),
        rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(
        float(a.total_energy), float(b.total_energy), rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(a.energy_trace), np.asarray(b.energy_trace), rtol=1e-8, atol=1e-9)
    assert bool(a.converged) == bool(b.converged)
    assert int(a.cycles_run) == int(b.cycles_run)


@pytest.mark.parametrize("max_cycles", [3, 12])
def test_forward_only_matches_scan_rks(max_cycles):
    """RKS: forward_only Python-loop SCF == jax.lax.scan SCF (H2, closed-shell)."""
    model, data = _make_h2_full()
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                       max_cycles=max_cycles, conv_tol=1e-12)
    scan_res = run_scf(cfg, model, data, forward_only=False)
    fwd_res = run_scf(cfg, model, data, forward_only=True)
    _assert_scf_result_equal(scan_res, fwd_res)


@pytest.mark.parametrize("max_cycles", [3, 12])
def test_forward_only_matches_scan_uks(max_cycles):
    """UKS: forward_only Python-loop SCF == jax.lax.scan SCF (H atom, open-shell)."""
    model, data = _make_h_atom_full()
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                       max_cycles=max_cycles, conv_tol=1e-12)
    scan_res = run_scf(cfg, model, data, forward_only=False)
    fwd_res = run_scf(cfg, model, data, forward_only=True)
    _assert_scf_result_equal(scan_res, fwd_res)


def test_forward_only_grad_checkpoint_irrelevant_for_forward():
    """scf_grad_checkpoint changes only the scan's reverse tape; the forward path
    ignores it (raw body), so both checkpoint settings give the same forward SCF."""
    model, data = _make_h2_full()
    common = dict(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                  max_cycles=5, conv_tol=1e-12)
    r_ckpt = run_scf(SolverConfig(scf_grad_checkpoint=True, **common), model, data,
                     forward_only=True)
    r_plain = run_scf(SolverConfig(scf_grad_checkpoint=False, **common), model, data,
                      forward_only=True)
    _assert_scf_result_equal(r_ckpt, r_plain)
