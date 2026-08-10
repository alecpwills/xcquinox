"""Opt-in shape-padding pass (:mod:`xcquinox.alec.padding`): results-neutral, and it
collapses the de-fused per-molecule JIT to one kernel per spin-type.

The per-molecule kernel is retained one-per-distinct-JIT-signature in a never-evicted
process-global cache; at deep_attn x ~26 molecules that exhausts the process mmap ceiling
at compile time (see HISTORY). The pass pads arrays to one common shape, strips the
molecule-identifying leaves the energy never reads, and traces the scalar energies and
occupation counts, so molecules of one spin-type share ONE kernel (RKS + UKS => two).

Contract pinned here: padding does NOT change energy, per-channel loss, or the model
gradient (weight-0 padded grid; decoupled padded AO block; occupation via a traced mask),
verified against the unpadded de-fuse (which the sibling suite pins to the fused oracle).
"""
import logging
import os
import tempfile

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from xcquinox.alec.config import ArchitectureConfig, TrainingSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.train import _build_model, run_training
from xcquinox.alec.oneshot import scf_loss_tail_weights, total_energy_for_solver
from xcquinox.alec.tests.fixtures.molecules import h_atom, o_atom, h2_molecule
from xcquinox.alec.defused_grad import defused_value_and_grad, _energy_trajectory_jit
from xcquinox.alec.padding import (common_pad_target, canonicalize_mol_data,
                                   PadTarget)
from xcquinox.alec.tests.test_defused_grad import _SC_FULL, _l5_group, _ARCH


def _run(group, pad_target):
    """Drive the de-fuse for a ``(model, loss, batch, cw, relative)`` group."""
    model, loss, batch, cw, relative = group
    return defused_value_and_grad(loss, model, batch, cw, relative,
                                  pad_target=pad_target)


def _assert_same(a, b):
    """Two ``((total, comps), grads)`` results equal to float64 round-off."""
    (la, ca), ga = a
    (lb, cb), gb = b
    assert jnp.allclose(la, lb, rtol=1e-9, atol=1e-12), (la, lb)
    assert set(ca) == set(cb)
    for k in ca:
        assert jnp.allclose(ca[k], cb[k], rtol=1e-9, atol=1e-12), (k, ca[k], cb[k])
    leaves_a = jax.tree_util.tree_leaves(ga)
    leaves_b = jax.tree_util.tree_leaves(gb)
    assert len(leaves_a) == len(leaves_b) > 0
    for x, y in zip(leaves_a, leaves_b):
        assert jnp.allclose(x, y, rtol=1e-6, atol=1e-9), float(jnp.max(jnp.abs(x - y)))


def _count_compiles(thunk, needle=None):
    """Number of XLA compiles ``thunk`` triggers, from JAX's own log records (same
    mechanism as test_defused_grad::test_defused_no_rejit_over_repeated_shapes). Starts
    from a cold cache. ``needle`` restricts the count to compile messages naming a
    specific kernel, filtering out the many shape-independent scalar-op compiles."""
    jax.clear_caches()
    msgs = []

    class _H(logging.Handler):
        def emit(self, record):
            try:
                m = record.getMessage()
            except Exception:
                return
            if "Compiling" in m and (needle is None or needle in m):
                msgs.append(m)

    lg = logging.getLogger("jax")
    prev_level, prev_prop = lg.level, lg.propagate
    handler = _H(level=logging.NOTSET)
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    try:
        with jax.log_compiles():
            thunk()
    finally:
        lg.removeHandler(handler)
        lg.setLevel(prev_level)
        lg.propagate = prev_prop
    return len(msgs)


def _uks_pair():
    """A model + two DIFFERENT open-shell (UKS) molecules (H doublet, O triplet) --
    distinct electron counts AND distinct shapes -- plus the common pad target."""
    arch = ArchitectureConfig(name="t", depth=2, nodes=8, attention=False,
                              descriptors=(), x_constraints=(), c_constraints=(),
                              double_lob_clamp_allowed=False)
    spec = TrainingSpec.from_dicts(
        arch=arch, molecules=(h_atom(),), targets={"H": 0.0},
        atom_energies={"H": -0.5}, loss_name="A_atomization",
        loss_kwargs={"vxc_weight": 0.0}, update_scheme="per_molecule",
        require_atom_anchors=False, n_steps=1, lr_start=1e-3, lr_end=1e-5,
        lr_decay_start=0.0, grad_clip=1.0, checkpoint_dir=None, seed=42)
    model = _build_model(spec)
    md_h = precompute_fixed_density_data(h_atom(), required_keys=("eri",),
                                         orientation_lock_strength=0.0)
    md_o = precompute_fixed_density_data(o_atom(), required_keys=("eri",),
                                         orientation_lock_strength=0.0)
    return model, md_h, md_o, common_pad_target([md_h, md_o])


def test_padding_neutral_scalar_oneshot():
    """Scalar (ONESHOT) energy form: canonicalizing the group's molecules (pad + strip
    identifiers + trace scalars/occupation) leaves loss, channels, and gradient unchanged."""
    group = _l5_group(solver_config=None)
    target = common_pad_target(group[2]["mol_data"])
    _assert_same(_run(group, None), _run(group, target))


def test_padding_neutral_full_trajectory_all_channels():
    """The real training path: differentiable manual SCF, tail-trajectory energy, and all
    four L5 channels (synthetic refs) active -- canonicalize stays results-neutral on energy
    AND on the V_xc / density gradient branch (RKS + UKS molecules in the group)."""
    group = _l5_group(solver_config=_SC_FULL, extra_keys=("eri",), synth_refs=True)
    target = common_pad_target(group[2]["mol_data"])
    _assert_same(_run(group, None), _run(group, target))


def test_padding_collapses_same_spin_to_one_kernel():
    """The OOM fix property: two DIFFERENT UKS molecules (distinct shape AND electron
    count) each compile the energy kernel once (2 total) when raw, but ONCE when
    canonicalized to a common shape with traced occupation -- one kernel per spin-type."""
    model, md_h, md_o, target = _uks_pair()
    assert md_h["is_unrestricted"] and md_o["is_unrestricted"]
    assert (md_h["nocc_a"], md_h["nocc_b"]) != (md_o["nocc_a"], md_o["nocc_b"])
    assert scf_loss_tail_weights(_SC_FULL) is not None  # -> trajectory kernel
    ch, co = canonicalize_mol_data(md_h, target), canonicalize_mol_data(md_o, target)

    n_raw = _count_compiles(
        lambda: [_energy_trajectory_jit(model, m, _SC_FULL) for m in (md_h, md_o)],
        needle="_energy_")
    n_canon = _count_compiles(
        lambda: [_energy_trajectory_jit(model, m, _SC_FULL) for m in (ch, co)],
        needle="_energy_")
    assert n_raw == 2, n_raw
    assert n_canon == 1, n_canon


def test_padding_nan_safe_traced_empty_channel():
    """Padding the fully-polarized H atom (``nocc_b=0``) up to a larger shape makes
    ``nocc_b`` a TRACED 0-d array, so the solver's static empty-channel fast-path is
    bypassed and the all-zero occupation mask must still yield a FINITE energy AND gradient
    (no ``0*NaN`` from the eigh on the empty beta Fock)."""
    model, md_h, md_o, target = _uks_pair()
    assert md_h["nocc_b"] == 0  # the empty beta channel
    ch = canonicalize_mol_data(md_h, target)  # target > h -> real padding, traced nocc_b
    val, grad = eqx.filter_value_and_grad(
        lambda m: _energy_trajectory_jit(m, ch, _SC_FULL)[-1])(model)
    assert jnp.isfinite(val), val
    for leaf in jax.tree_util.tree_leaves(grad):
        assert jnp.all(jnp.isfinite(leaf))


def test_padding_none_is_default_noop():
    """``pad_target=None`` is byte-identical to not passing the argument at all --
    padding is strictly opt-in and off by default."""
    model, loss, batch, cw, relative = _l5_group(solver_config=None)
    r_kw = defused_value_and_grad(loss, model, batch, cw, relative, pad_target=None)
    r_default = defused_value_and_grad(loss, model, batch, cw, relative)
    _assert_same(r_kw, r_default)


def test_padded_training_generalizes_to_unpadded_forward_eval():
    """ACCEPTANCE CRITERION: a model trained WITH padding is identical (to round-off) to
    one trained WITHOUT it, and gives the same energy under a STANDARD UNPADDED
    forward-only SCF -- i.e. the accuracy learned from padded inputs transfers to plain,
    non-padded, forward-pass inference. Trains a real BH76 per-molecule loop three steps
    with ``pad_group_to_common_shape`` ON vs OFF (same seed/data) and compares the loss
    trajectories, the learned model params, and a forward-only energy on the UNPADDED
    molecule."""
    rxn = {"name": "r1", "reactants": ["H2"], "products": ["H"],
           "coeffs": [-1.0, 2.0], "e_rxn_ref": 0.17}

    def _train(pad):
        with tempfile.TemporaryDirectory() as ckpt:
            spec = TrainingSpec.from_dicts(
                arch=_ARCH, molecules=(h_atom(), h2_molecule()),
                targets={"H": -0.5, "H2": 0.17}, atom_energies={"H": -0.5},
                loss_name="L5_gradnorm_vxc_step7",
                loss_kwargs={"bh76_reactions": [rxn], "solver_config": _SC_FULL},
                update_scheme="per_molecule", require_atom_anchors=False,
                pad_group_to_common_shape=pad,
                n_steps=3, lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.0,
                grad_clip=1.0, checkpoint_dir=ckpt, seed=42, checkpoint_every=0)
            run_training(spec)
            model = eqx.tree_deserialise_leaves(
                os.path.join(ckpt, "model.eqx"), _build_model(spec))
            losses = np.load(os.path.join(ckpt, "losses.npy"))
        return model, losses

    m_pad, l_pad = _train(True)
    m_off, l_off = _train(False)

    # (1) padded training tracks unpadded training step-for-step (identical learning).
    assert np.allclose(l_pad, l_off, rtol=1e-6, atol=1e-8), (l_pad, l_off)
    lp = jax.tree_util.tree_leaves(eqx.filter(m_pad, eqx.is_inexact_array))
    lo = jax.tree_util.tree_leaves(eqx.filter(m_off, eqx.is_inexact_array))
    assert len(lp) == len(lo) > 0
    for a, b in zip(lp, lo):
        assert jnp.allclose(a, b, rtol=1e-5, atol=1e-7), float(jnp.max(jnp.abs(a - b)))

    # (2) the padded-trained model under a STANDARD UNPADDED forward-only SCF matches the
    # unpadded-trained model -- padded-input learning transfers to non-padded inference.
    md = precompute_fixed_density_data(h2_molecule(), required_keys=("eri",),
                                       orientation_lock_strength=0.0)
    e_pad = float(total_energy_for_solver(m_pad, md, _SC_FULL, forward_only=True))
    e_off = float(total_energy_for_solver(m_off, md, _SC_FULL, forward_only=True))
    assert abs(e_pad - e_off) < 1e-6, (e_pad, e_off)


def test_padding_rung35_multishell_keys_axes_and_bit_identity():
    """The multi-width rung-3.5 keys pad on the RIGHT axes and leave the real
    block bit-identical.

    ``rung35ms_proj_ao`` is a 3-D ``(n_alpha, N, nao)`` stack and must NOT take
    the ``_PAD_AO_ON_GRID`` route (``grid_axis=0, ao_axis=1``), which would pad
    the width axis as if it were the grid -- measured to inflate the array ~90x
    and feed a shape mismatch into the occupancy einsum. The correct case pads
    ``grid_axis=1, ao_axis=2``, mirroring ``ao_grid_deriv``.
    ``rung35ms_features`` ``(N, 2*n_alpha)`` edge-pads its grid axis.
    """
    import numpy as _np
    import jax.numpy as _jnp
    md = precompute_fixed_density_data(h_atom(), required_keys=("eri",),
                                       orientation_lock_strength=0.0)
    n_grid = int(_np.asarray(md["grid_weights"]).shape[0])
    nao = int(_np.asarray(md["s_matrix"]).shape[0])
    n_alpha = 3
    rng = _np.random.default_rng(0)
    proj = _jnp.asarray(rng.standard_normal((n_alpha, n_grid, nao)))
    feats = _jnp.asarray(rng.uniform(0.0, 1.0, size=(n_grid, 2 * n_alpha)))
    md2 = dict(md)
    md2["rung35ms_proj_ao"] = proj
    md2["rung35ms_features"] = feats

    target = PadTarget(n_ao=nao + 3, n_grid=n_grid + 7, naux=None)
    padded = canonicalize_mol_data(md2, target)

    a = _np.asarray(padded["rung35ms_proj_ao"])
    assert a.shape == (n_alpha, n_grid + 7, nao + 3), (
        f"projector stack padded to {a.shape}; the width axis must be "
        f"PRESERVED at {n_alpha} -- shape[0] != n_alpha means the stack took "
        f"the 2-D _PAD_AO_ON_GRID route with the wrong axes")
    _np.testing.assert_array_equal(a[:, :n_grid, :nao], _np.asarray(proj))
    assert _np.all(_np.isfinite(a))

    f = _np.asarray(padded["rung35ms_features"])
    assert f.shape == (n_grid + 7, 2 * n_alpha), f.shape
    _np.testing.assert_array_equal(f[:n_grid], _np.asarray(feats))
    # edge mode replicates the boundary row into the padded region; those rows
    # carry zero grid weight, so replication is results-neutral and finite.
    for row in f[n_grid:]:
        _np.testing.assert_array_equal(row, f[n_grid - 1])
