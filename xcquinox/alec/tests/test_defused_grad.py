"""De-fused per-molecule gradient: parity with the fused step, loss-agnostic,
and the validation forward-only energy neutrality.

These lock the fixes for the dfs6311 compile-OOM (see HISTORY): the training step
no longer fuses an N-molecule group's SCF into one kernel, and the in-loop
validation no longer compiles the differentiable fused-scan kernel. Correctness
is pinned on REAL PySCF-backed molecules and a real (small) differentiable SCF.
"""
import tempfile

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pytest

from xcquinox.alec.config import ArchitectureConfig, TrainingSpec
from xcquinox.alec.tests.fixtures.molecules import h_atom, h2_molecule
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.solver import SolverConfig, SolverBackend, SolverMode
from xcquinox.alec.oneshot import total_energy_for_solver
from xcquinox.alec.train import (
    _build_model, _training_groups, _build_group_loss_and_batch,
    _effective_channel_weights, run_training,
)
from xcquinox.alec.losses import make_loss
from xcquinox.alec.defused_grad import defused_value_and_grad


_ARCH = ArchitectureConfig(name="t", depth=2, nodes=8, attention=False,
                           descriptors=(), x_constraints=(), c_constraints=(),
                           double_lob_clamp_allowed=False)
_SC_FULL = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                        max_cycles=3, scf_loss_use_tail=True, scf_loss_tail=3)


def _batch(spec, extra_keys=()):
    required = set(extra_keys)
    for d in spec.arch.materialize_descriptors():
        required |= set(d.required_mol_keys)
    md = [precompute_fixed_density_data(
              m, required_keys=tuple(required),
              descriptors=spec.arch.materialize_descriptors(),
              orientation_lock_strength=0.0)
          for m in spec.molecules]
    return {"mol_data": tuple(md), "targets": spec.targets_dict,
            "atom_energies": spec.atom_energies_dict}


def _inject_synthetic_refs(batch):
    """Force loss_vxc/loss_rho non-zero without external ref data: set each
    molecule's vxc_ref := vxc_pbe and rho_ref_grid := rho_grid, so the channel
    measures (NN - PBE)^2 != 0 and the non-energy gradient branch is exercised."""
    md = []
    for d in batch["mol_data"]:
        d2 = dict(d)
        if d2.get("vxc_ref") is None and d2.get("vxc_pbe") is not None:
            d2["vxc_ref"] = d2["vxc_pbe"]
        if d2.get("rho_ref_grid") is None and d2.get("rho_grid") is not None:
            d2["rho_ref_grid"] = d2["rho_grid"]
        md.append(d2)
    return {**batch, "mol_data": tuple(md)}


def _l5_group(solver_config=None, extra_keys=(), synth_refs=False):
    """Real H2 <-> 2H BH76 group over PySCF mol_data: (model, gloss, gbatch, cw, relative)."""
    lk = {"bh76_reactions": [{"name": "r1", "reactants": ["H2"], "products": ["H"],
                              "coeffs": [-1.0, 2.0], "e_rxn_ref": 0.17}]}
    if solver_config is not None:
        lk["solver_config"] = solver_config
    spec = TrainingSpec.from_dicts(
        arch=_ARCH, molecules=(h_atom(), h2_molecule()),
        targets={"H": -0.5, "H2": 0.17}, atom_energies={"H": -0.5},
        loss_name="L5_gradnorm_vxc_step7", loss_kwargs=lk,
        update_scheme="per_molecule", require_atom_anchors=False,
        n_steps=1, lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.0,
        grad_clip=1.0, checkpoint_dir=None, seed=42)
    model = _build_model(spec)
    batch = _batch(spec, extra_keys)
    g = next(gr for gr in _training_groups(spec) if gr["label"] == "bh76:r1")
    gloss, gbatch = _build_group_loss_and_batch(spec, g, batch)
    if synth_refs:
        gbatch = _inject_synthetic_refs(gbatch)
    cw = _effective_channel_weights(spec.channel_weights_dict)
    return model, gloss, gbatch, cw, (spec.loss_metric == "relative")


def _fused_value_and_grad(gloss, model, gbatch, cw, relative):
    """Oracle: the exact fused computation the pre-fix _step used (train.py)."""
    def scalar(m):
        comps = gloss.compute_components(m, gbatch, relative=relative)
        total = jnp.array(0.0)
        for k, v in comps.items():
            total = total + cw.get(k, 1.0) * v
        return total, comps
    return eqx.filter_value_and_grad(scalar, has_aux=True)(model)


def _assert_parity(model, gloss, gbatch, cw, relative, expect_channels=None):
    (loss_f, comps_f), grad_f = _fused_value_and_grad(gloss, model, gbatch, cw, relative)
    (loss_d, comps_d), grad_d = defused_value_and_grad(gloss, model, gbatch, cw, relative)
    # loss
    assert jnp.allclose(loss_f, loss_d, rtol=1e-9, atol=1e-12), (loss_f, loss_d)
    # per-channel components
    assert set(comps_f) == set(comps_d)
    for k in comps_f:
        assert jnp.allclose(comps_f[k], comps_d[k], rtol=1e-9, atol=1e-12), k
    # gradients, leaf by leaf
    leaves_f = jax.tree_util.tree_leaves(eqx.filter(grad_f, eqx.is_inexact_array))
    leaves_d = jax.tree_util.tree_leaves(grad_d)
    assert len(leaves_f) == len(leaves_d) > 0
    for a, b in zip(leaves_f, leaves_d):
        assert jnp.allclose(a, b, rtol=1e-6, atol=1e-9), float(jnp.max(jnp.abs(a - b)))
    if expect_channels is not None:
        nz = {k for k in comps_d if abs(float(comps_d[k])) > 0}
        assert expect_channels <= nz, (expect_channels, nz)


def test_defused_matches_fused_oneshot():
    """Scalar (ONESHOT) energy form: AE + BH76 channels non-zero."""
    _assert_parity(*_l5_group(solver_config=None),
                   expect_channels={"loss_AE", "loss_BH76"})


def test_defused_matches_fused_full_trajectory():
    """The real training path: differentiable manual SCF + tail-trajectory form."""
    _assert_parity(*_l5_group(solver_config=_SC_FULL, extra_keys=("eri",)),
                   expect_channels={"loss_AE", "loss_BH76"})


def test_defused_matches_fused_with_active_vxc_rho():
    """Exercise the NON-energy gradient branch: all four L5 channels non-zero."""
    _assert_parity(*_l5_group(solver_config=None, synth_refs=True),
                   expect_channels={"loss_AE", "loss_BH76", "loss_vxc", "loss_rho"})


def test_defused_is_loss_agnostic():
    """The utility works for a DIFFERENT loss class (A_atomization), using only
    the shared energy hook + compute_components -- no L5-specific structure."""
    spec = TrainingSpec.from_dicts(
        arch=_ARCH, molecules=(h_atom(), h2_molecule()),
        targets={"H2": 0.17}, atom_energies={"H": -0.5},
        loss_name="A_atomization", loss_kwargs={"vxc_weight": 0.0},
        update_scheme="per_molecule", require_atom_anchors=False,
        n_steps=1, lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.0,
        grad_clip=1.0, checkpoint_dir=None, seed=42)
    model = _build_model(spec)
    loss = make_loss(spec.loss_name, molecules=spec.molecules, **spec.loss_kwargs_dict)
    batch = _batch(spec)
    _assert_parity(model, loss, batch, {}, spec.loss_metric == "relative")


def test_defused_matches_fused_full_trajectory_with_vxc_rho():
    """Trajectory (tail) energy form AND non-zero one-shot V_xc/density channels
    together -- the full production L5 channel shape on the real differentiable
    SCF (the prior 4-channel test was scalar-mode only)."""
    _assert_parity(*_l5_group(solver_config=_SC_FULL, extra_keys=("eri",),
                              synth_refs=True),
                   expect_channels={"loss_AE", "loss_BH76", "loss_vxc", "loss_rho"})


def test_defused_matches_fused_single_molecule():
    """n_mol=1: the single-element jnp.stack + single-pass vjp loop must still
    reproduce the fused gradient (every other parity test uses a 2-molecule
    group, so the one-molecule path is otherwise unexercised)."""
    spec = TrainingSpec.from_dicts(
        arch=_ARCH, molecules=(h2_molecule(),),
        targets={"H2": 0.17}, atom_energies={"H": -0.5},
        loss_name="A_atomization", loss_kwargs={"vxc_weight": 0.0},
        update_scheme="per_molecule", require_atom_anchors=False,
        n_steps=1, lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.0,
        grad_clip=1.0, checkpoint_dir=None, seed=42)
    model = _build_model(spec)
    loss = make_loss(spec.loss_name, molecules=spec.molecules, **spec.loss_kwargs_dict)
    batch = _batch(spec)
    assert len(batch["mol_data"]) == 1
    _assert_parity(model, loss, batch, {}, spec.loss_metric == "relative")


def test_scalar_loss_under_full_tail_solver_raises_loudly():
    """#6 constraint: a scalar-only loss (A_atomization always consumes the
    'scalar' energy form) under a FULL + scf_loss_use_tail solver has
    'trajectory' injected; the form mismatch raises LOUDLY (never a silent wrong
    gradient). Not a production config -- production uses L5 -- but the loud
    failure is the documented contract."""
    spec = TrainingSpec.from_dicts(
        arch=_ARCH, molecules=(h_atom(), h2_molecule()),
        targets={"H2": 0.17}, atom_energies={"H": -0.5},
        loss_name="A_atomization",
        loss_kwargs={"vxc_weight": 0.0, "solver_config": _SC_FULL},
        update_scheme="per_molecule", require_atom_anchors=False,
        n_steps=1, lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.0,
        grad_clip=1.0, checkpoint_dir=None, seed=42)
    model = _build_model(spec)
    loss = make_loss(spec.loss_name, molecules=spec.molecules, **spec.loss_kwargs_dict)
    batch = _batch(spec, extra_keys=("eri",))
    with pytest.raises(RuntimeError, match="override active but form"):
        defused_value_and_grad(loss, model, batch, {}, False)


def test_validation_energy_forward_only_is_neutral():
    """The validation fix: forward_only=True gives the SAME converged energy as
    the differentiable scan (results-neutral) on a real FULL SCF, so it can skip
    the fused-scan compile without changing the metric."""
    model = _build_model(TrainingSpec.from_dicts(
        arch=_ARCH, molecules=(h2_molecule(),), targets={"H2": 0.17},
        atom_energies={"H": -0.5}, loss_name="L5_gradnorm_vxc_step7",
        loss_kwargs={"solver_config": _SC_FULL}, update_scheme="per_molecule",
        require_atom_anchors=False, n_steps=1, lr_start=1e-3, lr_end=1e-5,
        lr_decay_start=0.0, grad_clip=1.0, checkpoint_dir=None, seed=42))
    md = precompute_fixed_density_data(h2_molecule(), required_keys=("eri",))
    e_scan = float(total_energy_for_solver(model, md, _SC_FULL, forward_only=False))
    e_fwd = float(total_energy_for_solver(model, md, _SC_FULL, forward_only=True))
    assert abs(e_scan - e_fwd) < 1e-8, (e_scan, e_fwd)


@pytest.mark.slow
def test_defused_run_training_completes():
    """End-to-end: a real per_molecule run over a BH76 group with the de-fused
    step drives the optimizer to finite, decreasing loss and writes model.eqx."""
    rxn = {"name": "r1", "reactants": ["H2"], "products": ["H"],
           "coeffs": [-1.0, 2.0], "e_rxn_ref": 0.17}
    with tempfile.TemporaryDirectory() as ckpt:
        spec = TrainingSpec.from_dicts(
            arch=_ARCH, molecules=(h_atom(), h2_molecule()),
            targets={"H": -0.5, "H2": 0.17}, atom_energies={"H": -0.5},
            loss_name="L5_gradnorm_vxc_step7",
            loss_kwargs={"bh76_reactions": [rxn], "solver_config": _SC_FULL},
            update_scheme="per_molecule", require_atom_anchors=False,
            n_steps=3, lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.0,
            grad_clip=1.0, checkpoint_dir=ckpt, seed=42, checkpoint_every=0)
        run_training(spec)
        import os
        assert os.path.exists(os.path.join(ckpt, "model.eqx"))
        losses = np.load(os.path.join(ckpt, "losses.npy"))
    assert len(losses) > 0 and np.isfinite(losses).all()
