"""Loss-agnostic de-fused value-and-gradient for per-molecule training groups.

Motivation
----------
The fused training step wraps the whole group loss in one
``eqx.filter_value_and_grad`` (train.py ``_run_per_molecule_loop._step``). Its
loss traces every molecule's self-consistent SCF energy inside a single graph
(via :func:`losses._compute_energies` / :func:`losses._compute_energy_trajectories`,
each a ``jnp.stack`` over the group's molecules), so an ``N``-molecule group
compiles ``N`` SCF forward+backward passes into one XLA kernel whose LLVM CPU
codegen scales with ``N``. For the DFS-parity basis/grid (6-311++G(3df,2pd),
grid 3) that kernel exhausts host RAM at COMPILE time on the largest groups.

Method
------
This utility computes the identical value and gradient without that fusion:

1. **Per-molecule energies (de-fused).** Each molecule's SCF energy (or tail
   trajectory) is evaluated by a per-molecule ``eqx.filter_jit`` -- one small
   compile per molecule *shape*, reused across the group and across epochs --
   and stacked into ``E``.
2. **Assembly + energy cotangents.** ``E`` is injected (see
   :mod:`xcquinox.alec.energy_override`) so the loss's channel assembly consumes
   it as an ordinary array; differentiating the total loss with respect to
   ``(E, model)`` yields the cotangent on each molecule's energy and the model
   gradient of the remaining (non-energy) channels such as V_xc / density.
   Those channels are differentiated directly here (holding ``E`` constant);
   under a FULL solver the density channel itself runs a per-molecule SCF
   (``_grid_term`` -> ``oneshot_grid_density`` -> ``run_scf``), so it is NOT
   one-shot -- but, like the energy, it compiles at per-molecule size (eager,
   no group-wide jit), so it does not re-fuse across the group.
3. **Per-molecule backward.** For each molecule a per-molecule vjp of the SCF
   energy, seeded by that molecule's energy cotangent, gives its contribution to
   the model gradient; summed over the group these reconstruct the exact energy
   gradient (``dL/dtheta = sum_i (dL/dE_i) (dE_i/dtheta)`` -- the reaction/AE
   coupling enters only through the scalar energies).

The result equals the fused ``value_and_grad`` to float64 round-off while the
expensive SCF compiles at per-molecule size. It uses only the shared energy
helpers and the loss's ``compute_components`` / ``solver_config``, never any
channel-specific structure. The injected energy FORM (final-step ``"scalar"``
vs convergence-tail ``"trajectory"``) is chosen from the solver's tail setting,
matching how ``L5GradnormVxcStep7`` selects it. A scalar-only loss (the A/B/C/D
families always consume ``"scalar"``) run under a FULL + ``scf_loss_use_tail``
solver would request a form that was not injected and raise loudly
(:func:`energy_override.get_energy_override`) rather than return a wrong
gradient -- that pairing is not a production config (production uses L5); every
other loss/solver pairing is handled.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from xcquinox.alec.oneshot import (
    total_energy_for_solver,
    energy_trajectory_for_solver,
    scf_loss_tail_weights,
)
from xcquinox.alec.energy_override import (
    set_energy_override,
    reset_energy_override,
)


# Per-molecule energy evaluators, jitted once at module scope so JAX caches the
# compiled kernel per (molecule shape, solver_config) and reuses it across every
# group and epoch. solver_config is a static (non-array) argument.
@eqx.filter_jit
def _energy_trajectory_jit(model, mol_data, solver_config):
    return energy_trajectory_for_solver(model, mol_data, solver_config)


@eqx.filter_jit
def _energy_scalar_jit(model, mol_data, solver_config):
    return total_energy_for_solver(model, mol_data, solver_config)


@eqx.filter_jit
def _energy_trajectory_vjp_jit(model, mol_data, solver_config, cotangent):
    _val, vjp = eqx.filter_vjp(
        lambda m: energy_trajectory_for_solver(m, mol_data, solver_config), model)
    return vjp(cotangent)[0]


@eqx.filter_jit
def _energy_scalar_vjp_jit(model, mol_data, solver_config, cotangent):
    _val, vjp = eqx.filter_vjp(
        lambda m: total_energy_for_solver(m, mol_data, solver_config), model)
    return vjp(cotangent)[0]


def _tree_add(a, b):
    return jax.tree_util.tree_map(jnp.add, a, b)


def defused_value_and_grad(loss, model, batch, channel_weights, relative=False):
    """Loss-agnostic drop-in for the fused per-molecule
    ``eqx.filter_value_and_grad``.

    Parameters mirror the fused ``_step``: ``loss`` is the group loss (any
    :class:`AlecLoss`), ``batch`` its sub-batch (``batch["mol_data"]`` is the
    tuple of the group's per-molecule data), ``channel_weights`` the fixed
    per-channel weights, ``relative`` the loss metric flag.

    Returns ``((total, components), grads)`` where ``grads`` is the model
    gradient as an inexact-array pytree (matching
    ``eqx.filter(model, eqx.is_inexact_array)``), identical to the fused
    gradient to float64 round-off.
    """
    mol_data = batch["mol_data"]
    n_mol = len(mol_data)
    solver_config = loss.solver_config

    # Which energy form does the loss consume? Mirror the loss's own decision
    # (tail-weighted trajectory vs final-step scalar) so the injected stack is
    # exactly the array the shared helper would have returned.
    step_w = scf_loss_tail_weights(solver_config)
    if step_w is not None:
        kind = "trajectory"
        energy_jit = _energy_trajectory_jit
        vjp_jit = _energy_trajectory_vjp_jit
    else:
        kind = "scalar"
        energy_jit = _energy_scalar_jit
        vjp_jit = _energy_scalar_vjp_jit

    # Pass 1: per-molecule energies, one small compile per molecule shape.
    E_stack = jnp.stack([energy_jit(model, mol_data[i], solver_config)
                         for i in range(n_mol)])

    # Assembly: differentiate the total loss wrt (injected energy stack, model).
    # With the stack injected, the energy path is a constant array, so the model
    # gradient here carries ONLY the non-energy (V_xc / density) channels; the
    # energy gradient is reconstructed in pass 2 from the energy cotangents.
    arrays, static = eqx.partition(model, eqx.is_inexact_array)

    def assemble(energy_stack, arrays_):
        m = eqx.combine(arrays_, static)
        token = set_energy_override({kind: energy_stack})
        try:
            comps = loss.compute_components(m, batch, relative=relative)
            total = jnp.zeros(())
            for key, value in comps.items():
                total = total + channel_weights.get(key, 1.0) * value
        finally:
            reset_energy_override(token)
        return total, comps

    (total, comps), (ct_energy, grad_nonenergy) = jax.value_and_grad(
        assemble, argnums=(0, 1), has_aux=True)(E_stack, arrays)

    # Pass 2: per-molecule seeded backward through the SCF energies, summed.
    grad_energy = None
    for i in range(n_mol):
        gi = vjp_jit(model, mol_data[i], solver_config, ct_energy[i])
        grad_energy = gi if grad_energy is None else _tree_add(grad_energy, gi)

    grad_energy_arrays = eqx.filter(grad_energy, eqx.is_inexact_array)
    grads = _tree_add(grad_energy_arrays, grad_nonenergy)
    return (total, comps), grads
