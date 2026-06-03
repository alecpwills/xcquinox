"""Slow end-to-end smokes for the SCF solver.

Not run in normal CI, marked with @pytest.mark.slow. Run manually when
touching SCF code: `pytest xcquinox/alec/tests/slow/ -v -m slow`.
"""
import pytest
import jax.numpy as jnp
import equinox as eqx
import optax

from xcquinox.alec.config import ArchitectureConfig, MoleculeSpec
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.solver import (
    SolverConfig, SolverBackend, SolverMode, run_scf,
)
from xcquinox.alec.tests.fixtures.molecules import h2_molecule


@pytest.mark.slow
def test_train_h2_fixed_j_1cycle_decreases_loss():
    """Mini training loop: 20 steps, loss must decrease from the initial value."""
    arch = ArchitectureConfig(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    model = AlecGGAModel.from_arch(arch, seed=0)
    data = precompute_fixed_density_data(h2_molecule())
    cfg = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FIXED_J,
        max_cycles=1, conv_tol=1e-4,
    )

    def loss_fn(m):
        result = run_scf(cfg, m, data)
        return result.total_energy ** 2

    opt = optax.adam(1e-3)
    params = eqx.filter(model, eqx.is_inexact_array)
    state = opt.init(params)

    loss_init = float(loss_fn(model))
    for _ in range(20):
        grads = eqx.filter_grad(loss_fn)(model)
        updates, state = opt.update(grads, state, params)
        model = eqx.apply_updates(model, updates)
        params = eqx.filter(model, eqx.is_inexact_array)
    loss_final = float(loss_fn(model))

    assert loss_final < loss_init * 0.99, (
        f"20 optimizer steps did not decrease loss below 99% of initial: "
        f"init={loss_init} final={loss_final}"
    )


@pytest.mark.slow
def test_eval_he_full_scf_is_finite_and_converged():
    """Slow smoke: He/cc-pVDZ (non-degenerate, 5 basis functions) full SCF."""
    he = MoleculeSpec(
        name="He", atom="He 0 0 0", basis="cc-pvdz",
        charge=0, spin=0, atom_composition=(("He", 1),),
    )
    arch = ArchitectureConfig(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    model = AlecGGAModel.from_arch(arch, seed=0)
    data = precompute_fixed_density_data(he, required_keys=("eri",))
    cfg = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
        max_cycles=30, conv_tol=1e-6,
    )
    result = run_scf(cfg, model, data)
    assert bool(result.converged) is True
    assert jnp.isfinite(result.total_energy)
