"""The DFS meta-GGA indicator coordinate on and off its physical domain.

``ln((alpha + 1)/2)`` of the raw iso-orbital indicator is the meta-GGA MLP
coordinate under the DFS coordinates (PRB 104, L161109 (2021), Eq. 10). For
any N-representable density ``tau_W <= tau`` pointwise (Cauchy-Schwarz on the
orbitals), so ``alpha = (tau - tau_W)/tau_unif >= 0`` and the logarithm is
defined. An intermediate density matrix outside the positive-semidefinite
cone -- the ``decaying_linear`` mixer's step-0 coefficient 1.3 produces one --
carries ``tau < tau_W`` in the far tail of a doubled spin channel, the raw
indicator recovered from the stored column drops below -1, and the
coordinate was undefined: every meta-GGA architecture returned NaN at SCF
cycle 1 on the open-shell species (measured 2026-09-04 on Li, LiH and OH in
a small basis). The coordinate is floored at alpha = -1/2, halfway to the
singularity: the identity, in value and gradient, on every physical row --
including the floating-point residues below zero that one-orbital regions
produce (measured -3.5e-9 on H and -1.6e-6 on the Li beta channel) -- and
finite, with finite gradients, below the floor.
"""
import dataclasses
import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from xcquinox.alec import networks
from xcquinox.alec.metagga import (
    _ALPHA_SMOOTHING_WIDTH, smooth_positive_part)


def _reference_coordinate(alpha_raw):
    """The unguarded expression, ``ln((alpha + 1)/2)``, as written."""
    return jnp.log((jnp.asarray(alpha_raw) + 1.0) / 2.0)


def test_the_coordinate_is_the_reference_expression_on_the_physical_domain():
    alpha = jnp.concatenate([
        # the floor itself and the residue band physical rows occupy
        jnp.array([-0.5, -0.49, -1e-3, -1e-4, -1.62e-6, -3.45e-9, -1e-300]),
        jnp.array([0.0, 1e-12, 1e-6]),
        jnp.geomspace(1e-3, 1e6, 400),
        jnp.linspace(0.0, 50.0, 201),
    ])
    got = networks._dfs_indicator_coordinate(alpha)
    want = _reference_coordinate(alpha)
    assert np.array_equal(np.asarray(got), np.asarray(want))
    g_got = jax.vmap(jax.grad(networks._dfs_indicator_coordinate))(alpha)
    g_want = jax.vmap(jax.grad(_reference_coordinate))(alpha)
    assert np.array_equal(np.asarray(g_got), np.asarray(g_want))


@pytest.mark.parametrize("alpha_raw", [-0.51, -1.0, -1.22, -50.0, -1e12])
def test_the_coordinate_is_finite_with_a_finite_gradient_off_the_domain(alpha_raw):
    value = networks._dfs_indicator_coordinate(jnp.asarray(alpha_raw))
    grad = jax.grad(networks._dfs_indicator_coordinate)(jnp.asarray(alpha_raw))
    assert np.isfinite(float(value)), float(value)
    assert float(grad) == 0.0, float(grad)
    # Below the floor the coordinate sits at the floor value ln(1/4).
    assert float(value) == pytest.approx(float(np.log(0.25)), abs=0.0)


@pytest.mark.parametrize("column", [1e-3, 1e-10, 1e-30, 1e-100, 1e-200, 1e-300])
def test_the_coordinate_through_the_stored_column_is_finite(column):
    """Through ``_raw_indicator`` (the exact inverse of the smooth positive
    part) a tiny stored column encodes a large negative raw indicator; value
    and gradient with respect to the column must stay finite."""
    def coord(p):
        return networks._dfs_indicator_coordinate(networks._domain_indicator(p))
    p = jnp.asarray(column)
    assert np.isfinite(float(coord(p)))
    assert np.isfinite(float(jax.grad(coord)(p)))


def _mgga_model(seed=7, bias=0.0):
    import equinox as eqx
    from xcquinox.alec.config import ARCHITECTURES, apply_model_block
    from xcquinox.alec.models import AlecGGAModel
    block = SimpleNamespace(descriptor_coordinates="dfs", parent_anchor=False)
    arch = dataclasses.replace(
        apply_model_block(ARCHITECTURES["deep_mgga_3x16"], block),
        use_polarized_correlation=True, zero_init_final_layer=True)
    model = AlecGGAModel.from_arch(arch, seed=seed)
    if bias:
        def _set(net):
            lay = net.net.layers[-1]
            return eqx.tree_at(lambda n: n.net.layers[-1].bias, net,
                               jnp.full_like(lay.bias, bias))
        model = eqx.tree_at(lambda m: (m.xnet, m.cnet), model,
                            (_set(model.xnet), _set(model.cnet)))
    return model


def test_a_meta_gga_forward_is_finite_on_a_column_encoding_a_negative_indicator():
    """A stored column encoding alpha_raw = -1.22 (the smallest value
    measured on the Li beta channel after one over-relaxed cycle) through
    the deep_mgga_3x16 exchange and correlation networks under the DFS
    coordinates: finite outputs, finite parameter gradients."""
    import equinox as eqx
    model = _mgga_model(bias=0.02)
    p_bad = float(smooth_positive_part(-1.22, _ALPHA_SMOOTHING_WIDTH))
    p_ok = float(smooth_positive_part(0.7, _ALPHA_SMOOTHING_WIDTH))
    rho = jnp.asarray(1e-9)
    sigma = jnp.asarray(1e-19)
    zeta = jnp.asarray(0.0)

    def fx(m, p):
        return jnp.sum(m.xnet(jnp.array([rho, sigma, p])))

    def fc(m, p):
        return jnp.sum(m.cnet(jnp.array([rho, sigma, zeta, p])))

    for fn in (fx, fc):
        assert np.isfinite(float(fn(model, p_ok)))
        assert np.isfinite(float(fn(model, p_bad))), fn.__name__
        grads = eqx.filter_grad(fn)(model, p_bad)
        leaves = [g for g in jax.tree_util.tree_leaves(grads) if g is not None]
        assert leaves
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves), fn.__name__


_SYSTEMS = {
    # name: (geometry, spin, composition) -- LiH is the closed-shell case that
    # failed at clone-scale mismatch; Li and OH are the open-shell cases where
    # the doubled beta channel carries the negative indicator.
    "LiH": ("Li 0 0 0; H 0 0 1.5957", 0, (("H", 1), ("Li", 1))),
    "Li": ("Li 0 0 0", 1, (("Li", 1),)),
    "OH": ("O 0 0 0; H 0 0 0.9697", 1, (("H", 1), ("O", 1))),
}


def _solver(floor):
    from xcquinox.alec.solver import (
        FeaturePolicy, SolverBackend, SolverConfig, SolverMode)
    return SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL, max_cycles=3,
        conv_tol=1e-6, feature_policy=FeaturePolicy.REASSEMBLE,
        mixer_name="decaying_linear",
        mixer_kwargs=(("base", 0.3), ("floor", floor)),
        scf_loss_use_tail=True, scf_loss_tail=10,
        scf_loss_weight_power=2.0, orientation_lock_strength=0.0,
        seed_source="pbe")


def _precompute(name, model):
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.data import (
        clear_precompute_cache, precompute_fixed_density_data)
    atom, spin, comp = _SYSTEMS[name]
    spec = MoleculeSpec(name=name, atom=atom, basis="6-31g", charge=0,
                        spin=spin, atom_composition=comp, grid_level=1)
    clear_precompute_cache()
    req = set()
    for d in model.descriptors:
        req.update(d.required_mol_keys)
    req.add("eri")
    return precompute_fixed_density_data(
        spec, required_keys=tuple(sorted(req)), descriptors=model.descriptors,
        orientation_lock_strength=0.0, reference_xc="pbe")


@pytest.mark.parametrize("system", sorted(_SYSTEMS))
@pytest.mark.parametrize("bias", [0.02, 0.2])
def test_the_three_cycle_scf_of_a_meta_gga_clone_is_finite(system, bias):
    """The measured failure, as measured: LiH (closed shell, clone-scale
    mismatch), Li and OH (open shells, the doubled beta channel) in 6-31G at
    grid level 1, deep_mgga_3x16 with a final-layer bias of clone scale, the
    ``decaying_linear`` mixer at base 0.3 / floor 0.3 (coefficients 1.3, 0.6,
    0.39) from the PBE seed, three cycles. The energy trace was NaN from
    cycle 1; it must be finite, and within 0.1 Ha of the PSD-preserving
    schedule's trace (floor 0.0: coefficients 1.0, 0.3, 0.09)."""
    from xcquinox.alec.solver import run_scf
    model = _mgga_model(bias=bias)
    md = _precompute(system, model)
    trace = np.asarray(run_scf(_solver(0.3), model, md,
                               forward_only=True).energy_trace, dtype=float)
    trace_psd = np.asarray(run_scf(_solver(0.0), model, md,
                                   forward_only=True).energy_trace, dtype=float)
    assert np.all(np.isfinite(trace_psd)), trace_psd
    assert np.all(np.isfinite(trace)), trace
    assert abs(trace[-1] - trace_psd[-1]) < 0.1, (trace, trace_psd)


@pytest.mark.parametrize("system", ["Li", "LiH"])
def test_the_training_gradient_through_the_three_cycle_scf_is_finite(system):
    """What the train stage differentiates: the tail-weighted energy of the
    over-relaxed three-cycle SCF with respect to every network parameter,
    through the differentiable SCF itself (not forward-only). A NaN here is
    exactly what `_abort_if_nonfinite` would kill a training cell on."""
    import equinox as eqx
    from xcquinox.alec.oneshot import scf_tail_window
    from xcquinox.alec.solver import run_scf
    model = _mgga_model(bias=0.02)
    md = _precompute(system, model)
    cfg = _solver(0.3)
    _skip, w = scf_tail_window(3, 10, 2)
    w = jnp.asarray(w)

    def objective(m):
        trace = run_scf(cfg, m, md).energy_trace
        return jnp.sum(w * jnp.asarray(trace)) / jnp.sum(w)

    value = float(objective(model))
    assert np.isfinite(value), value
    grads = eqx.filter_grad(objective)(model)
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if g is not None]
    assert leaves
    bad = [g for g in leaves if not bool(jnp.all(jnp.isfinite(g)))]
    assert not bad, f"{len(bad)} of {len(leaves)} gradient leaves non-finite"
    assert any(bool(jnp.any(g != 0)) for g in leaves)

def test_the_domain_indicator_is_the_exact_inverse_above_the_floor_and_the_floor_below_it():
    width = _ALPHA_SMOOTHING_WIDTH
    alpha = jnp.concatenate([
        jnp.array([-0.5, -0.49, -1e-2, -1e-4, -1.62e-6, -3.45e-9, 0.0]),
        jnp.geomspace(1e-9, 50.0, 300)])
    column = smooth_positive_part(alpha, width)
    back = networks._domain_indicator(column)
    exact = networks._raw_indicator(column)
    assert np.array_equal(np.asarray(back), np.asarray(exact))
    # Round-trip precision: exact to 1e-9 on the physical side and the
    # residue band; the encoding itself loses digits near the floor (the
    # smooth positive part of -1/2 is 5e-11, formed by cancellation), which
    # is the column's property, not the guard's, and lives on unphysical rows.
    physical = np.asarray(alpha) >= -1e-2
    assert np.allclose(np.asarray(back)[physical], np.asarray(alpha)[physical],
                       rtol=1e-9, atol=1e-9)
    assert np.allclose(np.asarray(back)[~physical], np.asarray(alpha)[~physical],
                       rtol=1e-6, atol=0.0)
    # The gradient through the floored path equals the exact inverse's there.
    g_back = jax.vmap(jax.grad(networks._domain_indicator))(column)
    g_exact = jax.vmap(jax.grad(networks._raw_indicator))(column)
    assert np.array_equal(np.asarray(g_back), np.asarray(g_exact))
    below = smooth_positive_part(jnp.array([-0.51, -1.0, -1.22, -50.0, -1e12]), width)
    assert np.all(np.asarray(networks._domain_indicator(below)) == -0.5)
    g = jax.vmap(jax.grad(networks._domain_indicator))(
        jnp.concatenate([below, jnp.array([1e-300, 0.0])]))
    assert np.all(np.asarray(g) == 0.0)
