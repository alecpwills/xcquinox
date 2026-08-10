"""End-to-end derivative-wiring checks per architecture family (GGA, meta-GGA,
rung-3.5, multi-width rung-3.5), on small networks and a small molecule.

Three properties are pinned, chosen so that together they establish the chain
``parameters -> descriptor features -> V_xc (incl. the feature-response term)
-> Fock -> SCF density -> loss`` is differentiated correctly and is usable for
training:

1. ``V_xc == sym(dE_xc/dP)`` ELEMENTWISE, per family, with the feature-response
   term shown to be load-bearing for every DM-dependent family: assembling V_xc
   without it reproduces the pre-fix inconsistency (1e-4 to 1e-3 elementwise),
   while the descriptor-free GGA control is unaffected either way. This is the
   discriminating form of the fix -- an assembly that dropped the term again
   fails here immediately.

2. The TRAINING gradient ``dL/dtheta`` through the production solver
   (MANUAL / FULL / REASSEMBLE, unrolled cycles) matches a central finite
   difference along a random parameter direction, for BOTH an energy loss and a
   density-matrix loss. Autodiff of an unrolled computation is always
   self-consistent, so what this catches is any ``stop_gradient`` / detached
   constant on the chain -- the class of defect that twice produced a potential
   whose parameter gradient was not the gradient of the potential (the
   feature-response weighting and the metagga tail freeze). The density-matrix
   loss is included because that channel showed the larger error (5.4e-02
   relative) when the weighting was frozen; an energy-only check would have
   missed it.

3. TRAINING DESCENDS: a short optimization run on (a) an energy target and
   (b) a grid-density target against the PBE reference density reduces the loss
   by a required factor, per family. This is the usability statement -- the
   corrected chain does not merely have a truthful gradient, it trains the
   energy and DENSITY channels the production loss is built from.

Spin coverage: the families run RKS here for cost; the polarized-UKS chain
(spin-scaled exchange + per-spin correlation + accumulated feature derivative)
is pinned by the parametrized suite in ``test_solv01_split_xc.py``. One UKS
training-gradient case is included for the meta-GGA family, since its
descriptor recomputes from the live spin-summed DM inside the UKS loop.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pytest

import xcquinox.alec as alec
from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.solver import (SolverConfig, SolverBackend, SolverMode,
                                  FeaturePolicy, run_scf,
                                  _reassemble_features,
                                  _contract_dm_to_grid_with_nabla)

# family label -> registry architecture. Small 3x16 networks throughout.
_FAMILIES = {
    "gga": "deep_3x16",
    "mgga": "deep_mgga_3x16",
    "rung35": "deep_rung35_3x16",
    "rung35_multishell": "deep_rung35ms_3x16",
}
_DM_DEPENDENT = ("mgga", "rung35", "rung35_multishell")


def _build(family, seed=0):
    """Production configuration: polarized correlation, non-degenerate init.

    ``zero_init_final_layer=False`` is load-bearing: a zero-init final layer
    makes the enhancement factors constant, the features then have no effect on
    the energy, and every check below passes vacuously.
    """
    arch = dataclasses.replace(alec.get_architecture(_FAMILIES[family]),
                               use_polarized_correlation=True,
                               zero_init_final_layer=False)
    xnet, cnet = alec.create_network_pair(arch, seed=seed)
    return alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def _mol_data(model, name="H2", atom="H 0 0 0; H 0 0 0.74", spin=0,
              composition=(("H", 2),)):
    keys = tuple(sorted({k for d in model.descriptors
                         for k in d.required_mol_keys} | {"eri"}))
    spec = MoleculeSpec(name=name, atom=atom, basis="def2-svp", charge=0,
                        spin=spin, atom_composition=composition, grid_level=1)
    return precompute_fixed_density_data(spec, descriptors=model.descriptors,
                                         required_keys=keys)


def _features_closure(model, md):
    """The exact ``P -> features`` map the manual solver uses, as a closure."""
    ao_deriv = jnp.asarray(md["ao_grid_deriv"])
    s_matrix = jnp.asarray(md["s_matrix"])
    n_grid = int(np.asarray(md["grid_weights"]).shape[0])
    cusp = md.get("cusp_features")
    proj = md.get("rung35_proj_ao")
    proj_ms = md.get("rung35ms_proj_ao")
    has_mgga = any(type(d).__name__ == "MetaGGAAlphaDescriptor"
                   for d in model.descriptors)

    def features_of(P):
        if not model.descriptors:
            return jnp.zeros((n_grid, 0))
        kw = {}
        if has_mgga:
            rho_t, _nab, sigma_t = _contract_dm_to_grid_with_nabla(P, ao_deriv)
            kw = dict(ao_grad=ao_deriv[1:4], rho=rho_t, sigma=sigma_t)
        return _reassemble_features(
            descriptors=model.descriptors, dm=P, s_matrix=s_matrix,
            cusp_features=cusp, n_grid=n_grid, rung35_proj_ao=proj,
            rung35ms_proj_ao=proj_ms, **kw)
    return features_of


def _solver_cfg(cycles=3):
    return SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                        max_cycles=cycles, conv_tol=1e-12,
                        feature_policy=FeaturePolicy.REASSEMBLE)


# ---------------------------------------------------------------------------
# 1. V_xc == sym(dE_xc/dP) elementwise, and the feature term is load-bearing.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_vxc_equals_whole_energy_gradient_elementwise(family):
    from xcquinox.alec.oneshot import (compute_vxc_nn,
                                       feature_energy_derivative,
                                       feature_response_vxc,
                                       has_dm_dependent_descriptor)
    model = _build(family)
    md = _mol_data(model)
    ao_grid = jnp.asarray(md["ao_grid"])
    ao_deriv = jnp.asarray(md["ao_grid_deriv"])
    weights = jnp.asarray(md["grid_weights"])
    features_of = _features_closure(model, md)
    dm = np.asarray(md["dm_pbe"])
    P0 = jnp.asarray(dm.sum(axis=0) if dm.ndim == 3 else dm)

    def energy(P):
        rho, _nab, sigma = _contract_dm_to_grid_with_nabla(P, ao_deriv)
        return jnp.sum(weights * model.eval_exc(rho, sigma, features_of(P)))

    G = jax.grad(energy)(P0)
    G = 0.5 * (G + G.T)

    rho0, nab0, sig0 = _contract_dm_to_grid_with_nabla(P0, ao_deriv)
    f0 = features_of(P0)
    V_analytic = compute_vxc_nn(model, rho0, sig0, f0, ao_grid, weights,
                                nabla_rho=nab0, ao_grad=ao_deriv)
    scale = float(jnp.max(jnp.abs(G)))
    if has_dm_dependent_descriptor(model):
        V_term = feature_response_vxc(
            feature_energy_derivative(model, rho0, sig0, f0),
            weights, features_of, P0)
        V_full = V_analytic + V_term
        # The term is load-bearing: without it the assembly reproduces the
        # pre-fix inconsistency; with it the potential IS the derivative.
        gap_without = float(jnp.max(jnp.abs(V_analytic - G))) / scale
        gap_with = float(jnp.max(jnp.abs(V_full - G))) / scale
        assert gap_with < 1e-8, (
            f"{family}: assembled V_xc is not dE_xc/dP elementwise "
            f"(rel max {gap_with:.3e})")
        assert gap_without > 100.0 * max(gap_with, 1e-12), (
            f"{family}: dropping the feature-response term changed nothing "
            f"(without {gap_without:.3e}, with {gap_with:.3e}) -- the term is "
            f"not load-bearing, so either the descriptor stopped responding to "
            f"the density matrix or the test lost its power")
    else:
        gap = float(jnp.max(jnp.abs(V_analytic - G))) / scale
        assert gap < 1e-8, (
            f"{family}: analytic V_xc drifted from dE_xc/dP ({gap:.3e})")


# ---------------------------------------------------------------------------
# 2. Training gradient through the production solver matches a central finite
#    difference, for an energy loss AND a density-matrix loss.
# ---------------------------------------------------------------------------
def _param_direction(model, seed=5):
    params, static = eqx.partition(model, eqx.is_inexact_array)
    rng = np.random.default_rng(seed)
    direction = jax.tree_util.tree_map(
        lambda a: jnp.asarray(rng.normal(size=a.shape)), params)
    return params, static, direction


def _fd_vs_ad(loss_of_model, model, eps=1e-5, seed=5):
    params, static, direction = _param_direction(model, seed)

    def along(t):
        shifted = jax.tree_util.tree_map(lambda a, d: a + t * d,
                                         params, direction)
        return loss_of_model(eqx.combine(shifted, static))

    ad = float(jax.grad(along)(0.0))
    fd = float((along(eps) - along(-eps)) / (2.0 * eps))
    rel = abs(ad - fd) / max(abs(ad), abs(fd), 1e-30)
    return ad, fd, rel


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_training_gradient_matches_fd_energy_and_dm_loss(family):
    model = _build(family)
    md = _mol_data(model)
    cfg = _solver_cfg()
    dm_ref = jnp.asarray(np.asarray(md["dm_pbe"]))

    def loss_energy(m):
        return run_scf(cfg, m, md).total_energy

    def loss_dm(m):
        D = run_scf(cfg, m, md).density_matrix
        return jnp.mean((D - dm_ref) ** 2)

    ad_e, fd_e, rel_e = _fd_vs_ad(loss_energy, model)
    assert np.isfinite(ad_e) and np.isfinite(fd_e), f"{family}: non-finite"
    assert rel_e < 1e-4, (
        f"{family}: energy-loss training gradient disagrees with the finite "
        f"difference (AD={ad_e:.6e} FD={fd_e:.6e} rel={rel_e:.3e}) -- a "
        f"detached constant is on the parameter chain")

    ad_d, fd_d, rel_d = _fd_vs_ad(loss_dm, model)
    assert np.isfinite(ad_d) and np.isfinite(fd_d), f"{family}: non-finite"
    assert rel_d < 1e-4, (
        f"{family}: density-matrix-loss training gradient disagrees with the "
        f"finite difference (AD={ad_d:.6e} FD={fd_d:.6e} rel={rel_d:.3e}); "
        f"this is the channel that reads the SCF-predicted density, where a "
        f"frozen weighting previously cost 5.4e-02 relative")


def test_training_gradient_matches_fd_uks_mgga():
    """One open-shell case: the meta-GGA descriptor recomputes from the live
    spin-summed DM inside the UKS loop, a distinct code path from RKS."""
    model = _build("mgga")
    md = _mol_data(model, name="Li", atom="Li 0 0 0", spin=1,
                   composition=(("Li", 1),))
    cfg = _solver_cfg()

    def loss_energy(m):
        return run_scf(cfg, m, md).total_energy

    ad, fd, rel = _fd_vs_ad(loss_energy, model)
    assert np.isfinite(ad) and np.isfinite(fd)
    assert rel < 1e-4, (
        f"UKS mgga: training gradient disagrees with the finite difference "
        f"(AD={ad:.6e} FD={fd:.6e} rel={rel:.3e})")


# ---------------------------------------------------------------------------
# 3. Training descends, per family, on the energy and density channels.
# ---------------------------------------------------------------------------
def _descend(loss_of_model, model, steps=10, lr=3e-3):
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @eqx.filter_jit
    def step(p, s):
        l, g = jax.value_and_grad(
            lambda q: loss_of_model(eqx.combine(q, static)))(p)
        updates, s = opt.update(g, s, p)
        return optax.apply_updates(p, updates), s, l

    losses = []
    for _ in range(steps):
        params, opt_state, l = step(params, opt_state)
        losses.append(float(l))
    final = float(loss_of_model(eqx.combine(params, static)))
    return losses[0], final


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_energy_training_descends(family):
    model = _build(family)
    md = _mol_data(model)
    cfg = _solver_cfg()
    e0 = float(run_scf(cfg, model, md).total_energy)
    e_target = e0 - 0.02  # a reachable shift, Ha

    def loss(m):
        return (run_scf(cfg, m, md).total_energy - e_target) ** 2

    first, final = _descend(loss, model)
    assert np.isfinite(final)
    assert final < 0.5 * first, (
        f"{family}: energy training did not descend "
        f"(first {first:.6e} -> final {final:.6e})")


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_density_training_descends(family):
    """The density channel: grid-weighted squared error of the SCF density
    against the PBE reference density -- the production rho loss's shape. For
    the DM-dependent families this gradient flows through the feature-response
    term; a chain that dropped it would still descend sometimes, but the
    combination with the elementwise and finite-difference checks above pins
    that the descent is along the TRUE gradient."""
    model = _build(family)
    md = _mol_data(model)
    cfg = _solver_cfg()
    ao_deriv = jnp.asarray(md["ao_grid_deriv"])
    weights = jnp.asarray(md["grid_weights"])
    rho_ref = jnp.asarray(md["rho_grid"])
    wsum = float(jnp.sum(weights))

    def loss(m):
        D = run_scf(cfg, m, md).density_matrix
        rho, _nab, _sig = _contract_dm_to_grid_with_nabla(D, ao_deriv)
        return jnp.sum(weights * (rho - rho_ref) ** 2) / wsum

    # Calibrated, not chosen: measured final/first ratios at lr=1e-2, 15 steps
    # are 0.278 (gga), 0.554 (mgga), 0.277 (rung35), 0.275 (multishell) -- the
    # meta-GGA family descends more slowly because its descriptor recomputes
    # alpha from the live density matrix, partially compensating the density
    # response. At the weaker lr=3e-3 / 10 steps the ratios sit at 0.81-0.89,
    # too close to flat to assert against. The 0.75 bound clears the worst
    # measured family by 1.35x and the others by ~2.7x.
    first, final = _descend(loss, model, steps=15, lr=1e-2)
    assert np.isfinite(final)
    assert final < 0.75 * first, (
        f"{family}: density training did not descend "
        f"(first {first:.6e} -> final {final:.6e}, "
        f"ratio {final / first:.3f} >= 0.75)")
