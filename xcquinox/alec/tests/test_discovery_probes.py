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


def test_p01_compute_vxc_nn_flows_grad_through_dynamic_rho():
    """P0.1: compute_vxc_nn must accept dynamic rho/sigma and let jax.grad
    flow through. Otherwise the manual SCF backend's D → rho → F → D' loop
    is not differentiable."""
    from xcquinox.alec.oneshot import compute_vxc_nn
    from xcquinox.alec.descriptors import assemble_descriptor_features

    model, data = _make_h2_model_and_data()
    features = assemble_descriptor_features(model.descriptors, data)
    ao_grid = data["ao_grid"]
    grid_weights = data["grid_weights"]

    def scalar_from_vxc(rho_dyn, sigma_dyn):
        vxc = compute_vxc_nn(
            model, rho_dyn, sigma_dyn, features, ao_grid, grid_weights,
        )
        return jnp.sum(vxc ** 2)

    rho0 = data["rho_grid"]
    sigma0 = data["sigma_grid"]

    grad_rho = jax.grad(scalar_from_vxc, argnums=0)(rho0, sigma0)
    grad_sigma = jax.grad(scalar_from_vxc, argnums=1)(rho0, sigma0)

    assert jnp.all(jnp.isfinite(grad_rho))
    assert jnp.all(jnp.isfinite(grad_sigma))
    assert grad_rho.shape == rho0.shape
    assert grad_sigma.shape == sigma0.shape


def test_p02_mol_data_has_metadata_for_pyscfad_rebuild():
    """P0.2: mol_data must contain enough metadata to rebuild a pyscf.gto.Mole
    object. Required: atom spec, basis, charge, spin."""
    _, data = _make_h2_model_and_data()
    assert "mol_metadata" in data, (
        "mol_data lacks 'mol_metadata' — extend precompute_fixed_density_data "
        "to stash atom/basis/charge/spin for pyscfad backend rebuild."
    )
    md = data["mol_metadata"]
    for k in ("atom", "basis", "charge", "spin"):
        assert k in md, f"mol_metadata missing required key {k!r}"
    assert isinstance(md["atom"], str)
    assert isinstance(md["basis"], str)
    assert isinstance(md["charge"], int)
    assert isinstance(md["spin"], int)
