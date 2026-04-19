"""Regression tests guarding against NaN gradients on UKS atoms in FIXED_J / FULL SCF.

Before the combined fix in commit acd1a6e0a, three distinct issues produced
NaN gradients for open-shell atoms during training via run_scf:

1. ``compute_vxc_nn`` JVP for v_sigma passes through ``sqrt(sigma)`` at
   sigma=0 (the beta channel of H spin=1 has rho_b=sigma_bb=0 identically),
   whose derivative is infinite. Fixed by sanitizing JVP inputs to safe
   defaults at low-rho points and masking output to zero.

2. ``eval_exc`` / ``eval_exc_scalar`` in models.py propagate the same
   sqrt(sigma)-derivative NaN through the network forward pass when called
   on beta-channel grid points. Fixed by sanitizing NN inputs to safe
   defaults at tail and masking Fx/Fc to the LDA/PW92 limit (=1).

3. ``_diagonalize_roothaan_unrestricted`` with ``nocc == 0`` produced a
   zero DM via ``C[:, :0] @ C[:, :0].T``, but the eigh JVP on a Fock
   with degenerate p-orbital eigenvalues (from atomic symmetry)
   propagated NaN through multi-cycle ``lax.scan``. Fixed by (a) static
   bypass for nocc=0 and (b) replacing the uniform eye regularization on
   ``F_orth`` with a non-uniform diagonal perturbation that breaks
   degeneracy instead of just shifting eigenvalues uniformly.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pytest

import xcquinox.alec as alec
from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.solver import (
    SolverBackend, SolverConfig, SolverMode, run_scf,
)


_CASES = [
    pytest.param("H", "H 0 0 0", 1, (("H", 1),), id="H_spin1"),
    pytest.param("O", "O 0 0 0", 2, (("O", 1),), id="O_spin2"),
]
_MODES = [
    pytest.param(SolverMode.FIXED_J, 3, id="fixed_j_3"),
    pytest.param(SolverMode.FULL, 3, id="full_3"),
]


def _build_atom_case(mol_name, atom_str, spin, comp):
    spec = MoleculeSpec(
        name=mol_name, atom=atom_str, basis="def2-svp", charge=0, spin=spin,
        atom_composition=comp, grid_level=1,
    )
    md = precompute_fixed_density_data(spec, required_keys=("eri",))
    arch = alec.get_architecture("deep_attn")
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    model = alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
    return spec, md, model


@pytest.mark.parametrize("mol_name,atom_str,spin,comp", _CASES)
@pytest.mark.parametrize("mode,max_cycles", _MODES)
def test_uks_atom_scf_gradient_is_finite(mol_name, atom_str, spin, comp, mode, max_cycles):
    """eqx.filter_grad through run_scf on a UKS atom must produce finite
    gradients across all NN parameters. Guards against the H spin=1 beta
    channel NaN (nocc_b=0 + sqrt(sigma=0) + degenerate eigh)."""
    spec, md, model = _build_atom_case(mol_name, atom_str, spin, comp)
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=mode, max_cycles=max_cycles)

    def loss_fn(m):
        return run_scf(cfg, m, md).total_energy

    E = float(loss_fn(model))
    assert np.isfinite(E), f"total_energy is not finite: {E}"

    grad_m = eqx.filter_grad(loss_fn)(model)
    leaves = [l for l in jax.tree_util.tree_leaves(grad_m)
              if hasattr(l, "shape") and l.size > 0]
    assert leaves, "no gradient leaves found"
    for leaf in leaves:
        n_nan = int(jnp.sum(jnp.isnan(leaf)))
        n_inf = int(jnp.sum(jnp.isinf(leaf)))
        assert n_nan == 0, (
            f"{mol_name} {mode.name} mc={max_cycles}: "
            f"gradient has {n_nan}/{leaf.size} NaN entries"
        )
        assert n_inf == 0, (
            f"{mol_name} {mode.name} mc={max_cycles}: "
            f"gradient has {n_inf}/{leaf.size} Inf entries"
        )


def test_h_atom_full_scf_multi_cycle_gradient_is_finite():
    """Dedicated guard on the exact failure mode reported in the notebook:
    H atom (spin=1) under ``A_atomization + full_3`` training produced NaN
    gradients at step 38/250. This test exercises the same code path with
    a minimal synthetic A-like loss to detect the underlying bug without
    requiring a full training run."""
    spec, md, model = _build_atom_case("H", "H 0 0 0", 1, (("H", 1),))
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                      max_cycles=3, conv_tol=1e-6)
    target_E = -0.5

    def a_loss(m):
        E = run_scf(cfg, m, md).total_energy
        return (E - target_E) ** 2

    g = eqx.filter_grad(a_loss)(model)
    leaves = [l for l in jax.tree_util.tree_leaves(g)
              if hasattr(l, "shape") and l.size > 0]
    any_nan = any(bool(jnp.any(jnp.isnan(l))) for l in leaves)
    assert not any_nan, "A-family loss gradient has NaN on H atom FULL mode"
