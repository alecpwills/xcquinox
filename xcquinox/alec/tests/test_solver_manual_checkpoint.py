"""Tests for the opt-in SCF gradient-checkpointing toggle
(``SolverConfig.scf_grad_checkpoint``).

Checkpointing (jax.remat) of the unrolled SCF ``lax.scan`` body changes the
reverse-mode memory profile, NOT the numerical result: forward energy and the
converged density matrix must be byte/near-bit identical, and the parameter
gradient must agree to fp tolerance. Both scan sites are exercised — the RKS
path (closed-shell H2O) and the UKS path (open-shell O triplet).

Default-off byte-identity is guaranteed structurally in solver_manual.py
(``scan_body = jax.checkpoint(body) if config.scf_grad_checkpoint else body`` —
when off, ``scan_body is body``), so these tests focus on the ON path matching
the OFF path.
"""
import equinox as eqx
import jax
import jax.numpy as jnp

from xcquinox.alec.config import MoleculeSpec, get_architecture
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.solver import SolverConfig, SolverBackend, SolverMode, run_scf


def _model():
    return AlecGGAModel.from_arch(get_architecture("deep"), seed=0)


def _rks_spec():
    return MoleculeSpec.from_dict(
        name="H2O", atom="O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468",
        basis="def2-svp", charge=0, spin=0,
        atom_composition={"O": 1, "H": 2}, grid_level=1, external_data_path=None)


def _uks_spec():
    return MoleculeSpec(
        name="O", atom="O 0 0 0", basis="sto-3g",
        charge=0, spin=2, atom_composition=(("O", 1),), grid_level=1,
        external_data_path=None)


def _cfgs(max_cycles):
    """(checkpoint-off, checkpoint-on) configs that differ ONLY in the toggle."""
    off = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                       max_cycles=max_cycles, conv_tol=1e-5)
    on = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                      max_cycles=max_cycles, conv_tol=1e-5,
                      scf_grad_checkpoint=True)
    return off, on


def test_checkpoint_rks_energy_and_dm_match():
    """RKS scan site: checkpoint-on energy + density matrix match off."""
    model = _model()
    md = precompute_fixed_density_data(_rks_spec(), required_keys=("eri",))
    off, on = _cfgs(max_cycles=3)
    r_off = run_scf(off, model, md)
    r_on = run_scf(on, model, md)
    assert abs(float(r_on.total_energy) - float(r_off.total_energy)) < 1e-8
    assert float(jnp.max(jnp.abs(r_on.density_matrix - r_off.density_matrix))) < 1e-8


def test_checkpoint_uks_energy_and_dm_match():
    """UKS scan site: checkpoint-on energy + density matrix match off."""
    model = _model()
    md = precompute_fixed_density_data(_uks_spec(), required_keys=("eri",))
    off, on = _cfgs(max_cycles=3)
    r_off = run_scf(off, model, md)
    r_on = run_scf(on, model, md)
    assert abs(float(r_on.total_energy) - float(r_off.total_energy)) < 1e-8
    assert float(jnp.max(jnp.abs(r_on.density_matrix - r_off.density_matrix))) < 1e-8


def test_checkpoint_gradient_matches_uncheckpointed():
    """The parameter gradient through the checkpointed scan equals the
    non-checkpointed gradient to fp tolerance (remat preserves exact grads)."""
    md = precompute_fixed_density_data(_rks_spec(), required_keys=("eri",))
    off, on = _cfgs(max_cycles=2)

    def loss_for(cfg):
        def loss(m):
            return run_scf(cfg, m, md).total_energy
        return loss

    g_off = eqx.filter_grad(loss_for(off))(_model())
    g_on = eqx.filter_grad(loss_for(on))(_model())
    l_off = [g for g in jax.tree_util.tree_leaves(g_off) if g is not None]
    l_on = [g for g in jax.tree_util.tree_leaves(g_on) if g is not None]
    assert l_on and len(l_on) == len(l_off)
    max_abs_diff = max(
        float(jnp.max(jnp.abs(a - b))) for a, b in zip(l_on, l_off)
    )
    assert max_abs_diff < 1e-6, max_abs_diff


def test_checkpoint_off_is_default():
    """The toggle defaults off so existing specs are byte-identical."""
    assert SolverConfig().scf_grad_checkpoint is False
    assert SolverConfig(mode=SolverMode.FULL, max_cycles=3).scf_grad_checkpoint is False
