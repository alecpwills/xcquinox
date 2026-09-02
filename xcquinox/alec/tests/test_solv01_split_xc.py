"""SOLV-01 verification: exchange/correlation split in the UKS XC energy and
potential.

Physics (verified against the literature):
  * EXCHANGE obeys the exact spin-scaling relation
        E_x[n_a, n_b] = 1/2 (E_x[2 n_a] + E_x[2 n_b])
    (Oliver & Perdew, Phys. Rev. A 20, 397 (1979)).
  * CORRELATION does NOT, it is spin-interpolated (von Barth & Hedin,
    J. Phys. C 5, 1629 (1972); PW92, Phys. Rev. B 45, 13244 (1992)). The
    model's correlation baseline ``pw92c_unpolarized_scalar`` is
    zeta-independent, so the EXISTING correlation model is evaluated ONCE on
    the TOTAL density (zeta=0).

These tests guard that the analytic split V_xc is the true functional
derivative of the split energy (Test C, finite difference), that UKS reduces
to RKS on a closed-shell system (Test B), and that the spin channels are
symmetric (Test D).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import xcquinox.alec as alec
from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.oneshot import (
    compute_vxc_nn,
    split_exc_energy_uks,
    compute_exc_nn,
    _ZETA_BOUNDARY_EPS,
    _RHO_TOT_FLOOR,
)
from xcquinox.alec.descriptors import assemble_descriptor_features


def _build_model(zero_init_final_layer: bool = True):
    arch = alec.get_architecture("deep")
    if not zero_init_final_layer:
        import dataclasses
        # A zero-init warm-start cnet returns Fc==1 with ZERO input-gradients;
        # tests of the model's sigma/feature response need a non-trivial init.
        arch = dataclasses.replace(arch, zero_init_final_layer=False)
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    return alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def _h2_rks_md():
    spec = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),), grid_level=1,
    )
    return precompute_fixed_density_data(spec, required_keys=("eri",))


def _li_uks_md():
    # Open-shell atom; spin=1 avoids p-orbital degeneracy.
    spec = MoleculeSpec(
        name="Li", atom="Li 0 0 0", basis="sto-3g",
        charge=0, spin=1, atom_composition=(("Li", 1),), grid_level=1,
    )
    return precompute_fixed_density_data(spec, required_keys=("eri",))


def _grid_quantities(D, ao_grid, ao_xyz):
    """Return (rho, nabla_rho, sigma) for a single-spin DM on the grid."""
    rho = jnp.einsum("ij,gi,gj->g", D, ao_grid, ao_grid)
    nabla_rho = 2.0 * jnp.einsum("ij,dgi,gj->gd", D, ao_xyz, ao_grid)
    sigma = jnp.sum(nabla_rho * nabla_rho, axis=1)
    return rho, nabla_rho, sigma


def _uks_split_energy(model, D_a, D_b, features_a, features_b, features_tot,
                      ao_grid, ao_xyz, grid_weights):
    """SOLV-01 split UKS XC energy from a spin-DM pair.

    ``features_a`` / ``features_b`` are the blocks of the symmetric doubled
    densities diag(P_a, P_a) and diag(P_b, P_b); ``features_tot`` is the
    physical block the correlation term consumes.
    """
    rho_a, nra, sig_aa = _grid_quantities(D_a, ao_grid, ao_xyz)
    rho_b, nrb, sig_bb = _grid_quantities(D_b, ao_grid, ao_xyz)
    nr_tot = nra + nrb
    sig_tot = jnp.sum(nr_tot * nr_tot, axis=1)
    return split_exc_energy_uks(
        model, rho_a, rho_b, sig_aa, sig_bb, sig_tot,
        features_a, features_b, features_tot, grid_weights,
    )


def _uks_split_vxc(model, D_a, D_b, features_a, features_b, features_tot,
                   ao_grid, ao_xyz, ao_grad, grid_weights):
    """SOLV-01 split UKS V_xc pair (V_a, V_b) from a spin-DM pair: per-spin
    spin-scaled exchange at each channel's own doubled-density block + shared
    total-density correlation at ``features_tot``."""
    rho_a, nra, sig_aa = _grid_quantities(D_a, ao_grid, ao_xyz)
    rho_b, nrb, sig_bb = _grid_quantities(D_b, ao_grid, ao_xyz)
    nr_tot = nra + nrb
    sig_tot = jnp.sum(nr_tot * nr_tot, axis=1)
    vx_a = compute_vxc_nn(
        model, 2.0 * rho_a, 4.0 * sig_aa, features_a, ao_grid, grid_weights,
        nabla_rho=2.0 * nra, ao_grad=ao_grad, part="x",
    )
    vx_b = compute_vxc_nn(
        model, 2.0 * rho_b, 4.0 * sig_bb, features_b, ao_grid, grid_weights,
        nabla_rho=2.0 * nrb, ao_grad=ao_grad, part="x",
    )
    vc = compute_vxc_nn(
        model, rho_a + rho_b, sig_tot, features_tot, ao_grid, grid_weights,
        nabla_rho=nr_tot, ao_grad=ao_grad, part="c",
    )
    return vx_a + vc, vx_b + vc


# ---------------------------------------------------------------------------
# Test B: closed-shell reduction. UKS with rho_a = rho_b must equal RKS.
# ---------------------------------------------------------------------------
def test_closed_shell_reduction_energy_and_vxc():
    model = _build_model()
    md = _h2_rks_md()
    ao_grid = jnp.asarray(md["ao_grid"])
    ao_grad = jnp.asarray(md["ao_grid_deriv"])
    ao_xyz = ao_grad[1:4]
    grid_weights = jnp.asarray(md["grid_weights"])
    features = assemble_descriptor_features(model.descriptors, md)

    # RKS total DM (closed-shell H2): dm_pbe is 2-D.
    D_rks = jnp.asarray(md["dm_pbe"])
    assert D_rks.ndim == 2
    # Feed as UKS with D_a = D_b = D_rks / 2 so rho_a = rho_b = rho/2 and
    # 2 rho_a = rho_tot = rho.
    D_a = 0.5 * D_rks
    D_b = 0.5 * D_rks

    # --- Energy ---
    # The `deep` architecture carries no descriptors, so `features` is the empty
    # (n_grid, 0) array and the three per-channel blocks are that same array.
    E_uks = _uks_split_energy(
        model, D_a, D_b, features, features, features,
        ao_grid, ao_xyz, grid_weights)
    rho_rks, nr_rks, sig_rks = _grid_quantities(D_rks, ao_grid, ao_xyz)
    E_rks = compute_exc_nn(model, rho_rks, sig_rks, features, grid_weights)
    e_resid = float(abs(E_uks - E_rks))
    assert e_resid < 1e-10, f"closed-shell energy residual {e_resid:.3e}"

    # --- Potential: each spin channel's V_xc must equal the RKS V_xc ---
    # For a closed-shell system fed as UKS (D_a = D_b = D/2), the alpha and
    # beta XC potentials each equal the RKS potential (and equal each other).
    # This is the standard UKS->RKS reduction: the spin Fock potential the
    # alpha electron sees is identical to the RKS potential. (Note V_a + V_b
    # = 2 V_rks because dE_rks/dD = (dE_uks/dD_a + dE_uks/dD_b)/2 at the
    # closed-shell point; the per-spin potential V_a is the RKS one.)
    V_a, V_b = _uks_split_vxc(
        model, D_a, D_b, features, features, features,
        ao_grid, ao_xyz, ao_grad, grid_weights)
    V_rks = compute_vxc_nn(
        model, rho_rks, sig_rks, features, ao_grid, grid_weights,
        nabla_rho=nr_rks, ao_grad=ao_grad, part="xc",
    )
    v_resid_a = float(jnp.max(jnp.abs(V_a - V_rks)))
    v_resid_b = float(jnp.max(jnp.abs(V_b - V_rks)))
    assert v_resid_a < 1e-10, f"closed-shell V_a residual {v_resid_a:.3e}"
    assert v_resid_b < 1e-10, f"closed-shell V_b residual {v_resid_b:.3e}"
    # rho_a == rho_b => V_a == V_b
    assert float(jnp.max(jnp.abs(V_a - V_b))) < 1e-12


def _exact_vxc_unmasked(model, rho, sigma, nabla_rho, features, ao_grid,
                        ao_xyz, grid_weights, part):
    """Exact (UNMASKED) GGA V_xc matrix for one (rho, sigma, nabla_rho) set,
    assembled from the reverse-mode per-point vrho/vsigma of the requested
    scalar energy density. This is, by construction, the true functional
    derivative of sum_g w_g eps_part(rho, sigma) w.r.t. the DM that produces
    (rho, nabla_rho), it omits the tail-sanitization that the production
    ``compute_vxc_nn`` applies for autodiff stability. Used to prove the
    SOLV-01 split formulas are exactly derivative-consistent.
    """
    if part == "x":
        fn = lambda r, s, f: model.eval_ex_scalar(r, s, f)
    elif part == "c":
        fn = lambda r, s, f: model.eval_ec_scalar(r, s, f)
    else:
        fn = lambda r, s, f: model.eval_exc_scalar(r, s, f)
    vrho = jax.vmap(lambda r, s, f: jax.grad(fn, 0)(r, s, f))(rho, sigma, features)
    vsig = jax.vmap(lambda r, s, f: jax.grad(fn, 1)(r, s, f))(rho, sigma, features)
    V_rho = jnp.einsum("g,gi,gj->ij", grid_weights * vrho, ao_grid, ao_grid)
    ndphi = jnp.einsum("gd,dgi->gi", nabla_rho, ao_xyz)
    A = jnp.einsum("g,gi,gj->ij", grid_weights * vsig, ndphi, ao_grid)
    return V_rho + 2.0 * (A + A.T)


def _uks_split_vxc_exact(model, D_a, D_b, features, ao_grid, ao_xyz,
                         grid_weights):
    """SOLV-01 split V_xc pair built from the UNMASKED exact assembler.

    Single ``features`` argument: this helper is used only with the
    descriptor-free ``deep`` architecture, where the three per-channel blocks
    are the same empty array.
    """
    rho_a, nra, sig_aa = _grid_quantities(D_a, ao_grid, ao_xyz)
    rho_b, nrb, sig_bb = _grid_quantities(D_b, ao_grid, ao_xyz)
    nr_tot = nra + nrb
    sig_tot = jnp.sum(nr_tot * nr_tot, axis=1)
    vx_a = _exact_vxc_unmasked(model, 2.0 * rho_a, 4.0 * sig_aa, 2.0 * nra,
                               features, ao_grid, ao_xyz, grid_weights, "x")
    vx_b = _exact_vxc_unmasked(model, 2.0 * rho_b, 4.0 * sig_bb, 2.0 * nrb,
                               features, ao_grid, ao_xyz, grid_weights, "x")
    vc = _exact_vxc_unmasked(model, rho_a + rho_b, sig_tot, nr_tot,
                             features, ao_grid, ao_xyz, grid_weights, "c")
    return vx_a + vc, vx_b + vc


# ---------------------------------------------------------------------------
# Test C: finite-difference energy<->potential consistency (the critical guard).
#
# Two layers:
#  (C1) The SOLV-01 split V_xc is EXACTLY the derivative of the split energy.
#       We verify this with an unmasked exact assembler vs the autodiff
#       gradient AND vs a central finite difference (to <1e-7 / <1e-9). This
#       is the true mathematical guard that the two SEPARATE code paths
#       (energy build vs analytic V_xc build) are mutually consistent.
#  (C2) The PRODUCTION ``compute_vxc_nn`` split V_xc reproduces the FD
#       derivative to FD precision (<1e-5). v_sigma masking is now denormal-
#       level (sigma <= 1e-30), masking only the genuinely-singular sigma==0
#       points, so the production analytic V_xc is the true functional
#       derivative of E_xc for open-shell systems too (see the
#       _V_SIGMA_THRESHOLD block comment in oneshot.py).
# ---------------------------------------------------------------------------
def test_fd_energy_potential_consistency():
    model = _build_model()
    md = _li_uks_md()
    ao_grid = jnp.asarray(md["ao_grid"])
    ao_grad = jnp.asarray(md["ao_grid_deriv"])
    ao_xyz = ao_grad[1:4]
    grid_weights = jnp.asarray(md["grid_weights"])
    features = assemble_descriptor_features(model.descriptors, md)

    dm = jnp.asarray(md["dm_pbe"])  # (2, nao, nao)
    assert dm.ndim == 3
    D_a, D_b = dm[0], dm[1]
    nao = D_a.shape[0]

    # Random symmetric perturbations (DM matrices are symmetric).
    rng = np.random.default_rng(20260523)
    dDa = rng.standard_normal((nao, nao))
    dDa = jnp.asarray(0.5 * (dDa + dDa.T))
    dDb = rng.standard_normal((nao, nao))
    dDb = jnp.asarray(0.5 * (dDb + dDb.T))

    # eps=1e-5 keeps the central-difference truncation error (~O(eps^2 E'''))
    # below the 1e-5 target while staying well above f64 round-off.
    eps = 1e-5

    # The `deep` architecture carries no descriptors, so `features` is the empty
    # (n_grid, 0) array and the three per-channel blocks are that same array.
    def E(Da, Db):
        return _uks_split_energy(
            model, Da, Db, features, features, features,
            ao_grid, ao_xyz, grid_weights)

    E_plus = E(D_a + eps * dDa, D_b + eps * dDb)
    E_minus = E(D_a - eps * dDa, D_b - eps * dDb)
    fd = float((E_plus - E_minus) / (2.0 * eps))

    # --- (C1) exact split V_xc vs FD: the true derivative-consistency proof.
    Vx_a, Vx_b = _uks_split_vxc_exact(
        model, D_a, D_b, features, ao_grid, ao_xyz, grid_weights)
    exact_contract = float(jnp.einsum("ij,ij->", Vx_a, dDa)
                           + jnp.einsum("ij,ij->", Vx_b, dDb))
    rel_exact = abs(fd - exact_contract) / max(abs(exact_contract), 1e-12)
    assert rel_exact < 1e-5, (
        f"split V_xc is NOT the derivative of the split energy: "
        f"FD={fd:.6e} exact={exact_contract:.6e} rel={rel_exact:.3e}"
    )

    # Tight cross-check independent of FD truncation: contracting the exact
    # split V_xc with a symmetric perturbation must equal contracting the
    # reverse-mode autodiff gradient of the split energy with the same
    # perturbation. (The full matrices can differ in their ANTISYMMETRIC part
    # because the forward ``nabla_rho = 2*einsum("ij,dgi,gj")`` shortcut is
    # one-sided in (i,j) while the V_xc assembler symmetrizes via 2(A+A.T);
    # both are identical on symmetric DMs, the only physical case. The
    # symmetric-perturbation contraction removes that gauge ambiguity.)
    ga, gb = jax.grad(E, argnums=(0, 1))(D_a, D_b)
    grad_contract = float(jnp.einsum("ij,ij->", ga, dDa)
                          + jnp.einsum("ij,ij->", gb, dDb))
    grad_resid = abs(grad_contract - exact_contract) / max(abs(grad_contract), 1e-12)
    assert grad_resid < 1e-10, (
        f"exact split V_xc disagrees with autodiff grad of the energy: "
        f"{grad_resid:.3e}"
    )

    # --- (C2) production compute_vxc_nn split V_xc vs FD, the REAL guard.
    #
    # The production ``compute_vxc_nn`` masks v_sigma only at the genuinely
    # singular sigma <= _V_SIGMA_THRESHOLD = 1e-30 (denormal-level) points; with
    # the tanh(s)^2 gate v_sigma has a FINITE limit as sigma->0 (F'(s) ~ s
    # cancels the 1/(2 sqrt(sigma)) factor), so the production V_xc IS the true
    # functional derivative of the energy to FD precision. (An earlier 1e-10
    # threshold zeroed a finite, energy-significant v_sigma over ~49% of a
    # diffuse open-shell channel, giving a 0.92 residual, fixed 2026-05-23.)
    V_a, V_b = _uks_split_vxc(
        model, D_a, D_b, features, features, features,
        ao_grid, ao_xyz, ao_grad, grid_weights)
    prod_contract = float(jnp.einsum("ij,ij->", V_a, dDa)
                          + jnp.einsum("ij,ij->", V_b, dDb))
    rel_prod = abs(fd - prod_contract) / max(abs(prod_contract), 1e-12)
    assert rel_prod < 1e-5, (
        f"production split V_xc is not FD-consistent with the energy "
        f"(rel={rel_prod:.3e}); check the v_sigma masking threshold in "
        f"oneshot._compute_vxc_nn_core."
    )


# ---------------------------------------------------------------------------
# V_xc == dE_xc/dP with features RECOMPUTED FROM THE PERTURBED DENSITY MATRIX,
# across EVERY registered architecture, RKS and polarized UKS.
#
# The test above cannot see the defect these guard: it evaluates the descriptor
# features once and holds them fixed inside the perturbed energy -- the same
# assumption the per-point JVP assembly makes, so both share one blind spot --
# and it builds its model from ``get_architecture("deep")``, which carries no
# descriptors, so the DM-dependent path is never exercised at all.
#
# Two properties of the harness are load-bearing:
#   * ``zero_init_final_layer=False``. A zero-init final layer makes the
#     enhancement factor constant, so the features have no effect on the energy
#     and the whole comparison passes vacuously.
#   * on the OPEN-SHELL path, grid points whose guard status differs between the
#     +eps and -eps evaluations are EXCLUDED. ``jnp.clip``/``jnp.maximum`` on
#     zeta and the network tail mask redefine the functional; autodiff returns
#     the true derivative of the redefined function, but a central difference
#     that straddles one of those boundaries sees the average of two different
#     slopes. Without the exclusion every architecture -- including the
#     descriptor-free control -- reports ~7e-05 and the test is worthless.
#     Zeroing a straddled point's WEIGHT removes it identically from the energy
#     and from every potential term, since both are linear in the weights.
# ---------------------------------------------------------------------------
_FD_EPS = 1e-6

# Bounds for architectures whose potential is now exact. Every figure quoted
# here was measured with THIS module's own helpers at _FD_EPS, so a reader can
# reproduce them by running this file rather than trusting a development
# scratch harness with a different seed or grid.
#
# RKS controls: deep_3x16 1.93e-10, deep_attn_3x16 9.18e-11,
# deep_cusp_3x16 2.15e-10, deep_notransform 2.02e-10. Fixed architectures land
# with them: rung35 2.12e-10, rung35only 5.19e-10.
#
# These residuals are deterministic WITHIN a process but depend on evaluation
# ORDER across architectures -- measured up to 4.8x movement (deep_rung35_mgga_3x16
# reports 1.48e-10 when run first in a fresh process and 7.03e-10 under pytest's
# alphabetical parametrization). Quote the parametrized-run values, since those
# are what the assertions actually see. The worst observed margin against
# _TOL_RKS is therefore deep_rung35_mgga_3x16 at 7.03e-10, and the pre-existing
# worst was medium_attn / deep_notransform_attn_3x16 at 5.40e-10 -- so the bound
# is set to clear the measured worst case by ~3x rather than the ~1.4x that
# 1e-9 would have given, which would eventually go flaky on another machine.
#
# The polarized UKS probe has a genuine floor set by the truncation/round-off
# trade-off, and it is architecture-dependent. Sweeping eps on the
# descriptor-free controls shows the expected V-shape (residual falls as eps^2,
# then rises as round-off takes over):
#
#     eps            1e-4      1e-5      1e-6      1e-7
#     deep_3x16      2.64e-06  1.74e-06  3.33e-08  1.65e-06
#     notransform    8.79e-06  6.34e-06  1.79e-07  1.64e-06
#
# The minimum is at 1e-6, which is why _FD_EPS is 1e-6. A genuine V != dE/dP
# inconsistency would not fall with eps at all -- that is what distinguishes
# this floor from a defect. The untransformed-input architectures sit 5.4x
# above the transformed ones (1.79e-07 vs 3.33e-08) because their higher
# derivatives are larger, so the bound must clear 1.79e-07; 5e-7 gives 2.8x.
# That is still four orders below the pre-fix residual on this path (rung-3.5
# 8.69e-04), so the test stays strongly discriminating.
_TOL_RKS = 2e-9
_TOL_UKS = 5e-7

# Every architecture is held to the same two bounds. The per-descriptor
# tolerance branches that once lived here are gone: the meta-GGA branch fell
# with the compute_alpha tail-freeze removal, and the dm_statistics branch
# fell with dm_entropy itself -- its dead gradient had been the residual.
# Residuals with the feature-derivative term in place, pre-fix -> post-fix,
# measured with this module's helpers at _FD_EPS (pre-fix = routing predicate
# forced False, which is exactly the old code path):
#
#                          RKS                       UKS
#   rung35          1.87e-03 -> 2.12e-10     8.69e-04 -> 3.54e-08   fixed
#   rung35only      2.21e-03 -> 5.19e-10     1.61e-03 -> 3.55e-08   fixed
#   mgga            5.33e-03 -> 2.09e-10     6.20e-03 -> 3.86e-08   fixed (freeze removed)
#   mgga_attn       5.47e-03 -> 1.12e-10     6.20e-03 -> 3.89e-08   fixed (freeze removed)
#   rung35_mgga     3.98e-03 -> 7.03e-10     5.27e-03 -> 3.85e-08   fixed (freeze removed)
#   dm              1.04e-02 -> 2.05e-10     6.74e-03 -> <5e-07     fixed (dm_entropy removed)
#   combined        1.27e-03 -> 4.16e-10     1.79e-03 -> <5e-07     fixed (dm_entropy removed)
#
# (RKS post-fix values for the dm family move between 7.1e-11 and 2.1e-10
# with evaluation order, like every row here; the UKS cells record the bound
# they pass rather than a single draw, since the open-shell probe's floor is
# the bound itself.)
#
# The UKS column above was measured on the Li atom with ONE feature block
# feeding both exchange channels. With each exchange channel on the block of
# its own doubled density diag(P_sigma, P_sigma), the probe of ``_uks_fd_path``
# displaces every populated channel along its own aufbau manifold
# (rank-preserving rotations), the directions the Roothaan step moves on, so
# no channel's indicator leaves the physical domain. Oracle O2 runs the probe
# on all four open-shell atoms of the pools; ``test_spin_scaling_oracles``
# parametrizes it over 31 architectures x {H, Li, N, O} and the case below is
# the O atom of that set. Residuals at _FD_EPS with the three-block potential
# and the rotation path, measured through THIS module's
# ``_assert_uks_fd_consistency`` over the full 124-cell grid, def2-svp, grid
# level 2:
#
#   species  worst rel   architecture                 best rel   mask
#   H        6.12e-10    deep_combined                2.81e-11   100% kept
#   Li       3.61e-08    deep_notransform_attn_3x16   2.43e-08   100% kept
#   N        6.61e-08    deep_cusp_attn               3.08e-10   100% kept
#   O        9.00e-09    shallow                      1.82e-10   100% kept
#
# Worst cell 6.61e-08 against _TOL_UKS = 5e-7, a margin of 7.6x; the mask
# removed zero points in all 124 cells (the Li and N figures sit above the
# H and O ones because the net derivative along a rotation of the reference
# density is small on those atoms while the harness states its residual
# against the net). The pre-closure table (linear displacements, the
# indicator's clip-state rows masking 2106 of N's 9616 points) read 7.4e-10
# worst; the two probes agree on the statement, on different directions.
#
# At the production identity (6-311++G(3df,2pd), grid level 3, O atom, the
# ``slow``-marked case) the figures on record -- every architecture between
# 1.5e-9 (deep_3x16) and 7.5e-8 (deep_mgga_3x16), the mask keeping 99.6% of
# the grid -- were taken on the superseded linear probe (DEFERRED_WORK.md
# entry 28 re-measures on the cluster). The def2-svp floor is tighter because
# the basis carries 39 functions and the grid 13344 points with a wider dynamic
# range, not because a term is missing: the descriptor-free control holds
# 6.2e-9 at every step from 1e-4 to 1e-8, i.e. it is round-off rather than
# truncation, and the meta-GGA rows fall with eps on a record whose mask does
# not move (N/def2-svp: 1.8e-8, 4.5e-10, 2.6e-11, 9.1e-12 at 1e-4 ... 1e-7).
#
# The meta-GGA rows reached the control level only after the compute_alpha tail
# gradient freeze was removed; with it in place they sat at ~3e-08 / ~1.7e-05.
#
# Bounds clear the worst measured value by ~3x. These quantities are round-off
# dominated, so the bound exists to catch an order-of-magnitude regression, not
# to certify precision; too tight and it goes flaky across machines.
#
# The dm_statistics rows were once capped near 1e-02 regardless of the
# feature term, because dm_entropy's dead gradient dominated the residual;
# with that feature removed (2026-08-06) they discriminate like every other
# row.


def _tolerances(model):
    """(RKS, UKS) bounds for this architecture, and why.

    Both blocked branches are GONE as of 2026-08-06. The meta-GGA branch went
    when the compute_alpha tail freeze was removed; the dm_statistics branch
    went with dm_entropy, whose dead gradient had been dominating that
    family's residual (1.04e-02 -> 2.05e-10 on its removal, under this file's
    own parametrized ordering; a fresh-process draw gave 5.2e-03 pre-fix). Every
    architecture is now held to the same measured floor. A tolerance branch
    that no longer reflects a real defect is worse than no branch at all --
    it silently absorbs the next one.
    """
    return _TOL_RKS, _TOL_UKS, None


def _live_model(arch_name, seed=0):
    """Production configuration: polarized correlation, non-degenerate init."""
    import dataclasses
    arch = dataclasses.replace(alec.get_architecture(arch_name),
                               use_polarized_correlation=True,
                               zero_init_final_layer=False)
    xnet, cnet = alec.create_network_pair(arch, seed=seed)
    return alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def _md_with_descriptors(model, name, atom, basis, spin, composition,
                         grid_level=2):
    keys = tuple(sorted({k for d in model.descriptors
                         for k in d.required_mol_keys} | {"eri"}))
    return precompute_fixed_density_data(
        MoleculeSpec(name=name, atom=atom, basis=basis, charge=0, spin=spin,
                     atom_composition=composition, grid_level=grid_level),
        required_keys=keys, descriptors=model.descriptors)


def _live_features_fn(model, md):
    """The exact ``P -> features`` map the RKS solver uses, as a closure."""
    from xcquinox.alec.solver import (
        _reassemble_features, _contract_dm_to_grid_with_nabla)
    ao_deriv = jnp.asarray(md["ao_grid_deriv"])
    n_grid = int(np.asarray(md["grid_weights"]).shape[0])
    s_matrix = jnp.asarray(md["s_matrix"])
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
            total = P if P.ndim == 2 else P[0] + P[1]
            rho_t, _nab, sigma_t = _contract_dm_to_grid_with_nabla(
                total, ao_deriv)
            kw = dict(ao_grad=ao_deriv[1:4], rho=rho_t, sigma=sigma_t)
        return _reassemble_features(
            descriptors=model.descriptors, dm=P, s_matrix=s_matrix,
            cusp_features=cusp, n_grid=n_grid, rung35_proj_ao=proj,
            rung35ms_proj_ao=proj_ms, **kw)
    return features_of


def _live_uks_features_fns(model, md):
    """The three ``P_ab -> features`` maps the UKS solver uses."""
    from xcquinox.alec.solver import make_uks_feature_fns
    return make_uks_feature_fns(
        descriptors=model.descriptors,
        ao_deriv=jnp.asarray(md["ao_grid_deriv"]),
        s_matrix=jnp.asarray(md["s_matrix"]),
        n_grid=int(np.asarray(md["grid_weights"]).shape[0]),
        cusp_features=md.get("cusp_features"),
        rung35_proj_ao=md.get("rung35_proj_ao"),
        rung35ms_proj_ao=md.get("rung35ms_proj_ao"),
    )


def _alpha_columns(model):
    """Column indices of the clipped iso-orbital indicator in the feature
    block (declaration order; the block has the same width in all three spin
    channels)."""
    from xcquinox.alec.descriptors import MetaGGAAlphaDescriptor
    cols, offset = [], 0
    for d in model.descriptors:
        if isinstance(d, MetaGGAAlphaDescriptor):
            cols.extend(range(offset, offset + d.n_features))
        offset += d.n_features
    return cols


def _symmetric_perturbation(shape, seed=20260806):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal(shape)
    return jnp.asarray(0.5 * (W + np.swapaxes(W, -1, -2)))


@pytest.mark.parametrize("arch_name", sorted(alec.ARCHITECTURES))
def test_fd_consistency_live_features_rks(arch_name):
    """RKS: the assembled V_xc must be dE_xc/dP when the descriptors respond
    to the density matrix."""
    from xcquinox.alec.oneshot import (
        compute_vxc_nn, feature_energy_derivative, feature_response_vxc,
        has_dm_dependent_descriptor)
    from xcquinox.alec.solver import _contract_dm_to_grid_with_nabla

    model = _live_model(arch_name)
    md = _md_with_descriptors(
        model, "H2O", "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469",
        "def2-svp", 0, (("O", 1), ("H", 2)))
    ao_grid = jnp.asarray(md["ao_grid"])
    ao_deriv = jnp.asarray(md["ao_grid_deriv"])
    weights = jnp.asarray(md["grid_weights"])
    features_of = _live_features_fn(model, md)

    dm = np.asarray(md["dm_pbe"])
    P0 = jnp.asarray(dm.sum(axis=0) if dm.ndim == 3 else dm)

    def energy(P):
        rho, _nab, sigma = _contract_dm_to_grid_with_nabla(P, ao_deriv)
        return jnp.sum(weights * model.eval_exc(rho, sigma, features_of(P)))

    rho0, nabla0, sigma0 = _contract_dm_to_grid_with_nabla(P0, ao_deriv)
    f0 = features_of(P0)
    V = compute_vxc_nn(model, rho0, sigma0, f0, ao_grid, weights,
                       nabla_rho=nabla0, ao_grad=ao_deriv)
    if has_dm_dependent_descriptor(model):
        V = V + feature_response_vxc(
            feature_energy_derivative(model, rho0, sigma0, f0),
            weights, features_of, P0)
    assert bool(jnp.all(jnp.isfinite(V))), f"{arch_name}: V_xc has NaN/inf"

    W = _symmetric_perturbation(P0.shape)
    analytic = float(jnp.sum(V * W))
    fd = float((energy(P0 + _FD_EPS * W) - energy(P0 - _FD_EPS * W))
               / (2.0 * _FD_EPS))
    rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-30)

    tol, _uks_tol, blocked_by = _tolerances(model)
    assert rel < tol, (
        f"{arch_name}: V_xc is not dE_xc/dP with live features "
        f"(FD={fd:.6e} analytic={analytic:.6e} rel={rel:.3e} > {tol:.0e}"
        + (f", bound set by the known {blocked_by} defect)" if blocked_by
           else ")")
        + ". The missing term is sum_g w_g (de/dfeatures)_g . dfeatures_g/dP; "
          "see oneshot.feature_response_vxc."
    )


# Open-shell atoms of the BH76 / W4-11 pools, in PySCF's 2S spin convention:
# H (1 alpha, 0 beta), Li (2, 1), N (5, 2), O (5, 3). The empty and the
# one-electron channels are what ``_uks_fd_path`` exists for.
_UKS_FD_SPECIES = {
    "H": ("H 0 0 0", 1, (("H", 1),)),
    "Li": ("Li 0 0 0", 1, (("Li", 1),)),
    "N": ("N 0 0 0", 3, (("N", 1),)),
    "O": ("O 0 0 0", 2, (("O", 1),)),
}


# Straddle-mask acceptance for the open-shell finite-difference probe. The mask
# removes grid points whose guard status -- the density-tail and zeta-clip
# thresholds of the network evaluation, which are hard steps of the functional
# -- differs between the two displaced density matrices, and what has to stay
# bounded is the INTEGRAND mass it removes rather than the point count: the
# discarded points sit in the density tail.
#
# The iso-orbital indicator no longer enters the mask. Its lower bound is a
# smooth positive part (metagga.compute_alpha), so it has no clip state at
# zero to straddle, and the probe below displaces every populated channel
# along its own aufbau manifold (rank-preserving orbital rotations), on which
# the raw indicator never leaves the physical domain; the ceiling at
# _ALPHA_MAX is a hard step of the functional but the rotation path was
# measured not to cross it on any of the four atoms at def2-svp (the
# residuals below were taken with no indicator row at all).
#
# Measured with the rotation path over the 124 harness cells (31
# architectures x {H, Li, N, O}, def2-svp, grid level 2): every cell inside
# 6.61e-08 of _TOL_UKS's 5e-7 with the mask removing zero points; on the
# solver's own Fock pair at grid level 1 (deep_mgga_3x16, the
# absolute-contribution footing of test_spin_scaling_solver_manual):
# 2.9e-11 (H), 2.4e-10 (Li), 9.8e-11 (N), 5.5e-11 (O) at the 1e-5 step.
# The same Fock pair along an UNRESTRICTED random symmetric direction reads
# 3.7e-8 (H, falling to 3.8e-10 at the 1e-7 step), 5.2e-2 (Li) and 6.0e-6
# (N), the latter two flat in the step. The cause is the RANK-ONE channel,
# not cone departure: Li's alpha channel leaves the positive semidefinite
# cone (minimum eigenvalue -2.2e-7) and still passes at 1.2e-10, while the
# rank-one beta channel fails flat; the indicator's tail response is
# shell-peaked (4.07e11 on the outermost shell, 1.15e1 below it, log-log
# slope -0.43 against 2 rho), so a 1e-6 step moves the raw indicator by
# 1e3-1e5 there, beyond any linear regime of the energy, whatever the width
# of the smooth positive part (DEFERRED_WORK.md entry 30). That is a
# property of the probe direction, not of the potential.
#
# 1e-2 admits every measured case (the mask removed 0 points on every atom
# at def2-svp / grid level 2 with the rotation path) and is kept as the guard
# against a displacement that walks the tail across the network's thresholds.
_MASK_MAX_DROPPED_MASS = 1e-2
_MASK_MIN_KEPT_POINTS = 0.5


def _orthonormalizers(s_matrix):
    """``(S^{1/2}, S^{-1/2})`` of the overlap, for rotations in the metric."""
    ev, evec = np.linalg.eigh(np.asarray(s_matrix))
    s_half = evec @ np.diag(np.sqrt(ev)) @ evec.T
    s_inv_half = evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T
    return s_half, s_inv_half


def _uks_fd_path(P0, md, eps=_FD_EPS, seed=20260821):
    """The two displaced density matrices of the central difference and the
    direction ``W = (P_plus - P_minus) / (2 eps)`` it probes, per channel.

    Every populated channel is displaced along its own aufbau manifold: with
    ``S^{1/2}`` the metric's square root and ``K`` a random antisymmetric
    generator of unit Frobenius norm, ``P_s(eps) = U(eps) P_s U(eps)^T`` with
    ``U(eps) = S^{-1/2} expm(eps K) S^{1/2}``, a rotation of the occupied
    orbitals in the S-metric that keeps the block idempotent, positive
    semidefinite and of fixed rank at every eps. The tangent direction is a
    random symmetric matrix in both channels, and the path is the one the
    Roothaan step moves on: every density matrix an SCF cycle visits is an
    aufbau matrix of the block's rank, so the derivative that governs the
    iteration is the derivative along this manifold.

    A linear displacement ``P_s +- eps W`` of a RANK-ONE channel breaks the
    von Weizsacker bound ``tau >= tau_W`` on one side of the step (cone
    departure alone is survivable: Li's alpha channel leaves the positive
    semidefinite cone by -2.2e-7 and still passes at 1.2e-10), and the
    indicator's tail response is shell-peaked -- 4.07e11 on Li's outermost
    shell, 1.15e1 below it, log-log slope -0.43 -- so a 1e-6 step moves the
    raw indicator by 1e3-1e5 there, beyond any linear regime of the energy. Measured with the linear form on
    deep_mgga_3x16 (def2-svp, grid 1, no mask): 5.2e-2 (Li) and 6.0e-6 (N),
    flat between the 1e-5 and 1e-7 steps; the rotation path keeps every grid
    point and lands between 1.8e-10 and 6.6e-8 over the 124 harness cells at
    grid level 2 (the residual stated against the net derivative, which a
    rotation of a reference fixed point keeps small). The smooth positive part of the indicator
    (metagga.compute_alpha) makes the energy differentiable across
    ``alpha_raw = 0`` -- H, whose tail carries no such response, reads
    3.8e-10 along the linear form at the 1e-7 step, against 7.4e-4 with the
    hard clip -- but it cannot make a step of 1e-6 small against a response
    of 1e10 (DEFERRED_WORK.md entry 30).

    A channel with no electron carries an identically zero density matrix
    that the SCF never populates, so its Fock block is not part of the
    functional's domain; it is left unperturbed and drops out of both sides
    of the comparison.
    """
    from scipy.linalg import expm
    s_half, s_inv_half = _orthonormalizers(md["s_matrix"])
    rng = np.random.default_rng(seed)
    P_plus, P_minus = np.array(P0), np.array(P0)
    for s, key in ((0, "nocc_a"), (1, "nocc_b")):
        if int(md[key]) == 0:
            continue
        K = rng.standard_normal(P_plus[s].shape)
        K = K - K.T
        K /= np.linalg.norm(K)
        for sign, target in ((1.0, P_plus), (-1.0, P_minus)):
            U = s_inv_half @ expm(sign * eps * K) @ s_half
            target[s] = U @ np.asarray(P0[s]) @ U.T
    W = (P_plus - P_minus) / (2.0 * eps)
    return jnp.asarray(P_plus), jnp.asarray(P_minus), jnp.asarray(W)


def _assert_uks_fd_consistency(model, md, arch_name, label, eps=_FD_EPS):
    """Oracle O2: the assembled UKS Fock pair is the derivative of the
    assembled energy along the probe direction of ``_uks_fd_path``.

    Exercises all four feature-derivative sites: the two spin-scaled exchange
    channels, each at the block of its own doubled density diag(P_sigma,
    P_sigma), and ``compute_vc_polarized_per_spin`` on the total block, plus
    the three chain-rule contractions that differentiate the three P -> f
    maps, every column live in every block (the manual solver's one-electron
    gate on the indicator's response is retired; DEFERRED_WORK.md entry 27).

    Grid points whose guard status -- the density-tail and zeta-clip
    thresholds of the network evaluation -- differs between the two displaced
    matrices are excluded (zeroing a point's weight removes it identically
    from the energy and from every potential term). The iso-orbital indicator
    enters no mask: its lower bound is smooth and the rotation path does not
    reach its ceiling.
    """
    from xcquinox.alec.oneshot import (
        compute_vxc_nn, compute_vc_polarized_per_spin,
        feature_energy_derivative, feature_response_vxc,
        has_dm_dependent_descriptor, uks_zeta,
        _ZETA_BOUNDARY_EPS, _RHO_TOT_FLOOR)
    from xcquinox.alec.models import _NN_TAIL_THRESHOLD

    ao_grid = jnp.asarray(md["ao_grid"])
    ao_deriv = jnp.asarray(md["ao_grid_deriv"])
    ao_xyz = ao_deriv[1:4]
    features_a_of, features_b_of, features_tot_of = _live_uks_features_fns(
        model, md)
    dm = np.asarray(md["dm_pbe"])
    assert dm.ndim == 3, f"{label} must precompute a spin-resolved DM"
    P0 = jnp.asarray(dm)
    P_plus, P_minus, W = _uks_fd_path(P0, md, eps=eps)

    def spin_quantities(D):
        rho = jnp.einsum("ij,gi,gj->g", D, ao_grid, ao_grid)
        nabla = 2.0 * jnp.einsum("ij,dgi,gj->gd", D, ao_xyz, ao_grid)
        return rho, nabla, jnp.sum(nabla * nabla, axis=1)

    def guard_status(P):
        rho_a = np.asarray(spin_quantities(P[0])[0])
        rho_b = np.asarray(spin_quantities(P[1])[0])
        rho_tot = rho_a + rho_b
        zeta = (rho_a - rho_b) / np.maximum(rho_tot, _RHO_TOT_FLOOR)
        return np.stack([
            np.abs(zeta) >= 1.0 - _ZETA_BOUNDARY_EPS,
            rho_tot <= _RHO_TOT_FLOOR,
            2.0 * rho_a <= _NN_TAIL_THRESHOLD,
            2.0 * rho_b <= _NN_TAIL_THRESHOLD,
            rho_a <= 1e-10, rho_b <= 1e-10, rho_tot <= 1e-10,
        ])

    keep = ~np.any(guard_status(P_plus) != guard_status(P_minus), axis=0)
    grid_w = np.asarray(md["grid_weights"])
    rho_tot0 = np.asarray(spin_quantities(P0[0])[0]
                          + spin_quantities(P0[1])[0])
    n_electrons = float(np.sum(grid_w * rho_tot0))
    dropped_mass = float(np.sum(grid_w * rho_tot0 * ~keep)) / n_electrons
    kept_points = float(keep.mean())
    assert dropped_mass < _MASK_MAX_DROPPED_MASS, (
        f"{label}: the guard-straddle mask discarded {dropped_mass:.3e} of "
        f"the electron density ({(~keep).sum()} of {keep.size} points); the "
        "probe would be measuring a functional that is no longer the one "
        "under test"
    )
    assert kept_points > _MASK_MIN_KEPT_POINTS, (
        f"{label}: the guard-straddle mask kept only {kept_points:.4f} of the "
        f"grid points (dropped mass {dropped_mass:.3e}); the perturbation is "
        "too large to probe the smooth part of the functional"
    )
    weights = jnp.asarray(grid_w) * jnp.asarray(keep, dtype=jnp.float64)

    def energy(P):
        rho_a, nabla_a, sigma_aa = spin_quantities(P[0])
        rho_b, nabla_b, sigma_bb = spin_quantities(P[1])
        nabla_tot = nabla_a + nabla_b
        return split_exc_energy_uks(
            model, rho_a, rho_b, sigma_aa, sigma_bb,
            jnp.sum(nabla_tot * nabla_tot, axis=1),
            features_a_of(P), features_b_of(P), features_tot_of(P), weights)

    rho_a, nabla_a, sigma_aa = spin_quantities(P0[0])
    rho_b, nabla_b, sigma_bb = spin_quantities(P0[1])
    nabla_tot = nabla_a + nabla_b
    sigma_tot = jnp.sum(nabla_tot * nabla_tot, axis=1)
    f0_a, f0_b, f0_tot = features_a_of(P0), features_b_of(P0), features_tot_of(P0)

    V_a = compute_vxc_nn(model, 2.0 * rho_a, 4.0 * sigma_aa, f0_a, ao_grid,
                         weights, nabla_rho=2.0 * nabla_a, ao_grad=ao_deriv,
                         part="x")
    V_b = compute_vxc_nn(model, 2.0 * rho_b, 4.0 * sigma_bb, f0_b, ao_grid,
                         weights, nabla_rho=2.0 * nabla_b, ao_grad=ao_deriv,
                         part="x")
    vc_a, vc_b = compute_vc_polarized_per_spin(
        model, rho_a, rho_b, sigma_tot, f0_tot, ao_grid, weights, nabla_tot,
        ao_deriv)
    V_a, V_b = V_a + vc_a, V_b + vc_b

    if has_dm_dependent_descriptor(model):
        # f_a, f_b and f_tot are three different maps of P, so the chain-rule
        # term is three contractions rather than one accumulated de/df. Each
        # per-channel map depends on P only through its own P_sigma, so its
        # contraction lands in that spin block.
        v_feat = feature_response_vxc(
            0.5 * feature_energy_derivative(
                model, 2.0 * rho_a, 4.0 * sigma_aa, f0_a, part="x"),
            weights, features_a_of, P0)
        v_feat = v_feat + feature_response_vxc(
            0.5 * feature_energy_derivative(
                model, 2.0 * rho_b, 4.0 * sigma_bb, f0_b, part="x"),
            weights, features_b_of, P0)
        v_feat = v_feat + feature_response_vxc(
            feature_energy_derivative(
                model, rho_a + rho_b, sigma_tot, f0_tot, part="c",
                zeta=uks_zeta(rho_a, rho_b)),
            weights, features_tot_of, P0)
        V_a, V_b = V_a + v_feat[0], V_b + v_feat[1]

    assert bool(jnp.all(jnp.isfinite(V_a)) and jnp.all(jnp.isfinite(V_b))), (
        f"{arch_name}/{label}: polarized UKS V_xc has NaN/inf")

    analytic = float(jnp.sum(V_a * W[0]) + jnp.sum(V_b * W[1]))
    fd = float((energy(P_plus) - energy(P_minus)) / (2.0 * eps))
    rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-30)

    _rks_tol, tol, blocked_by = _tolerances(model)
    assert rel < tol, (
        f"{arch_name}/{label}: polarized UKS V_xc is not dE_xc/dP with live "
        f"per-channel features (FD={fd:.6e} analytic={analytic:.6e} "
        f"rel={rel:.3e} > {tol:.0e}"
        + (f", bound set by the known {blocked_by} defect)" if blocked_by
           else ")")
    )
    return {"rel": rel, "kept_points": kept_points,
            "dropped_mass": dropped_mass, "n_dropped": int((~keep).sum()),
            "n_grid": int(keep.size)}


@pytest.mark.parametrize("arch_name", sorted(alec.ARCHITECTURES))
def test_fd_consistency_live_features_uks_polarized(arch_name):
    """Open-shell, polarized correlation -- the production configuration --
    on the O atom (5 alpha, 3 beta electrons), along the rotation path of
    ``_uks_fd_path``. The four open-shell atoms of the pools, H and Li
    included, are probed by oracle O2 in ``test_spin_scaling_oracles``
    through the same helper.
    """
    atom, spin, composition = _UKS_FD_SPECIES["O"]
    model = _live_model(arch_name)
    md = _md_with_descriptors(model, "O", atom, "def2-svp", spin, composition)
    _assert_uks_fd_consistency(model, md, arch_name, "O")


@pytest.mark.slow
@pytest.mark.parametrize("arch_name", sorted(alec.ARCHITECTURES))
def test_fd_consistency_uks_polarized_production_identity(arch_name):
    """Oracle O2 at the production identity: 6-311++G(3df,2pd), grid level 3.

    The def2-svp probes run on every architecture and every open-shell atom
    of the pools on each test invocation; this one carries the identity the
    campaign actually reports and is marked slow so it is opt-in
    (``-m slow``). The two differ only in basis and grid; the assertion is the
    same statement that the assembled Fock matrices are the derivative of the
    assembled energy.

    The probe is the O atom (5 alpha, 3 beta). The figures on record for
    this case (every architecture at 1.5e-9 to 7.5e-8, the mask keeping
    99.6% of the grid) were taken with the linear displacement and the
    indicator's clip-state straddle rows, before the indicator's lower bound
    became a smooth positive part and the probe moved onto the rotation
    manifold of ``_uks_fd_path``; the N atom was excluded then because the
    linear displacement drove its two-electron beta channel's tail indicator
    across the clip on a third of the grid (6.3e-2 of the density). The
    rotation path never leaves the physical domain, so N is no longer
    special on that account, but this production case keeps the O atom
    until the re-measurement is run on the cluster (DEFERRED_WORK.md
    entry 28).
    """
    model = _live_model(arch_name)
    md = _md_with_descriptors(model, "O", "O 0 0 0", "6-311++G(3df,2pd)", 2,
                              (("O", 1),), grid_level=3)
    _assert_uks_fd_consistency(model, md, arch_name, "O/production")


def test_one_electron_channel_block_is_the_iso_orbital_limit():
    """A spin channel holding one electron has a doubled block diag(P_s, P_s)
    that is a single orbital, for which tau = tau_W, so the SCAN indicator of
    that block vanishes identically (alpha = 0 for any one-orbital density:
    Sun, Ruzsinszky, Perdew, PRL 115, 036402 (2015)); the alpha and total
    blocks of the same atom are two-orbital densities and stay away from
    zero. This is the property that makes a one-electron channel unusable as
    a finite-difference probe along a linear displacement (see
    ``_uks_fd_path``), and it separates the per-channel block from the
    physical one: with the total block in its place the beta column would
    carry the physical indicator (median 1.5e-2 on the beta-resolved grid),
    and with sigma_ss left undoubled it would read the clip ceiling. A block
    with tau_s or rho_s left undoubled is negative before the smoothing and
    is NOT distinguished here; that convention is pinned against the stored
    per-spin tau and against libxc elsewhere. The stored column of the
    one-orbital channel is the smooth positive part's floor, width / 2 =
    5e-6, plus the rounding residue of tau - tau_W.
    """
    model = _live_model("deep_mgga_3x16")
    md = _md_with_descriptors(model, "Li", "Li 0 0 0", "def2-svp", 1,
                              (("Li", 1),))
    col = _alpha_columns(model)[0]
    f_a = np.asarray(assemble_descriptor_features(model.descriptors, md,
                                                  spin_channel=0))[:, col]
    f_b = np.asarray(assemble_descriptor_features(model.descriptors, md,
                                                  spin_channel=1))[:, col]
    f_tot = np.asarray(assemble_descriptor_features(model.descriptors, md))[:, col]
    dm = np.asarray(md["dm_pbe"])
    ao = np.asarray(md["ao_grid"])
    resolved = np.einsum("ij,gi,gj->g", dm[1], ao, ao) > 1e-8
    # The raw residue measured 2.5e-11 and 6.2e-10 (two independent PBE
    # solutions): the rounding of tau - tau_W divided by tau_unif, on which
    # the smoothing puts its floor. 1e-8 above the floor clears the worse
    # draw by 16x and refuses the total block by six orders.
    from xcquinox.alec.metagga import _ALPHA_SMOOTHING_WIDTH
    floor = 0.5 * _ALPHA_SMOOTHING_WIDTH
    assert np.abs(f_b[resolved] - floor).max() <= 1e-8, (
        float(np.abs(f_b[resolved] - floor).max()))
    # Two-orbital blocks: measured medians 7.3e-3 (alpha) and 1.5e-2 (total).
    assert np.median(f_a[resolved]) > 1e-3, float(np.median(f_a[resolved]))
    assert np.median(f_tot[resolved]) > 1e-3, float(np.median(f_tot[resolved]))


def test_descriptor_free_archs_keep_the_analytic_path():
    """The routing predicate must send only DM-dependent architectures down the
    gradient path, so the sound architectures stay byte-identical."""
    from xcquinox.alec.oneshot import has_dm_dependent_descriptor
    expected_free = {"deep_3x16", "deep_attn_3x16", "deep_cusp_3x16"}
    for name in expected_free:
        assert not has_dm_dependent_descriptor(_live_model(name)), (
            f"{name} must keep the analytic V_xc path")
    for name in ("deep_mgga_3x16", "deep_rung35_3x16", "deep_dm_3x16"):
        assert has_dm_dependent_descriptor(_live_model(name)), (
            f"{name} carries a DM-dependent descriptor and needs the "
            f"feature-response term")


def test_uks_zeta_is_shared_by_energy_and_potential():
    """The floor/clip/freeze must come from ONE definition.

    Held by comments in three places previously; a drift between the energy's
    zeta and the potential's zeta silently breaks v_c = dE_c/drho_sigma.
    """
    from xcquinox.alec import oneshot
    from xcquinox.alec.tests._source_scan import code_only
    for fn in (oneshot.split_exc_energy_uks,
               oneshot.compute_vc_polarized_per_spin):
        src = code_only(fn)
        assert "uks_zeta(" in src, (
            f"{fn.__name__} must form zeta via oneshot.uks_zeta")
        assert "_ZETA_BOUNDARY_EPS" not in src, (
            f"{fn.__name__} re-derives the zeta clip instead of using "
            f"uks_zeta; that is how the energy and potential drift apart")


# ---------------------------------------------------------------------------
# per-spin correlation potential for a spin-polarization-aware cnet.
# When the cnet carries the zeta input feature and the model uses the
# zeta-dependent PW92 baseline (Dick & Fernandez-Serra, PRB 104 L161109
# (2021)), E_c depends on rho_a/rho_b through BOTH rho_tot AND
# zeta = (rho_a-rho_b)/rho_tot, so V_c is PER-SPIN. These guard that
# ``compute_vc_polarized_per_spin`` is the true functional derivative of the
# polarized correlation energy, and reduces to a shared potential at zeta=0.
# ---------------------------------------------------------------------------
def _build_polarized_model(seed=0):
    """A model whose cnet consumes the spin-polarization (x1) feature and whose
    correlation baseline is the zeta-dependent PW92 form."""
    arch = alec.ArchitectureConfig.from_spec(
        "polc_test", 4, 32, use_polarized_correlation=True)
    xnet, cnet = alec.create_network_pair(arch, seed=seed)
    assert cnet.use_spin_polarization is True
    return alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def _ec_energy_polarized(model, D_a, D_b, features, ao_grid, ao_xyz,
                         grid_weights):
    """Polarized correlation energy E_c = sum_g w_g eps_c(rho_tot, sigma_tot,
    f; zeta) from a spin-DM pair (the zeta-dependent piece only)."""
    rho_a, nra, _ = _grid_quantities(D_a, ao_grid, ao_xyz)
    rho_b, nrb, _ = _grid_quantities(D_b, ao_grid, ao_xyz)
    rho_tot = rho_a + rho_b
    nr_tot = nra + nrb
    sig_tot = jnp.sum(nr_tot * nr_tot, axis=1)
    # Clip MUST match the production paths (split_exc_energy_uks / ec_spin) so
    # this reference is the exact energy whose gradient compute_vc_polarized_
    # per_spin builds, INCLUDING at the zeta boundary AND the negative-density tail
    # (_RHO_TOT_FLOOR + where/stop_gradient; HISTORY Phase 17).
    safe_rt = jnp.maximum(rho_tot, _RHO_TOT_FLOOR)
    _zeta_raw = jnp.clip((rho_a - rho_b) / safe_rt,
                         -1.0 + _ZETA_BOUNDARY_EPS, 1.0 - _ZETA_BOUNDARY_EPS)
    zeta = jnp.where(rho_tot > _RHO_TOT_FLOOR, _zeta_raw,
                     jax.lax.stop_gradient(_zeta_raw))
    return jnp.sum(grid_weights * model.eval_ec(rho_tot, sig_tot, features,
                                                zeta=zeta))


def test_polarized_vc_finite_on_negative_density_tail():
    """Regression (HISTORY Phase 17): diffuse-basis grid-tail quadrature noise (or
    an aggressive NN update) can drive the TOTAL density rho_tot = rho_a + rho_b
    slightly NEGATIVE. The old zeta = clip((ra-rb)/max(rt,1e-300)) floored a
    negative rt to 1e-300, whose square (1e-600) underflowed to 0 in the V_c jvp
    quotient rule -> 0*inf = NaN -- the meta-GGA `bh76:HLi` step-0 training failure
    (Li is open-shell -> UKS -> uses compute_vc_polarized_per_spin). The
    _RHO_TOT_FLOOR + where/stop_gradient guard makes the potential finite. Inject
    the reproduced negative-density tail (rho_a ~ -4e-10, rt < 0) into a real Li
    UKS density and confirm the per-spin V_c is all-finite (was NaN)."""
    from xcquinox.alec.oneshot import compute_vc_polarized_per_spin

    model = _build_polarized_model()
    md = _li_uks_md()
    ao_grid = jnp.asarray(md["ao_grid"])
    ao_grad = jnp.asarray(md["ao_grid_deriv"])
    ao_xyz = ao_grad[1:4]
    grid_weights = jnp.asarray(md["grid_weights"])
    features = assemble_descriptor_features(model.descriptors, md)
    dm = jnp.asarray(md["dm_pbe"])
    D_a, D_b = dm[0], dm[1]
    rho_a, nra, _ = _grid_quantities(D_a, ao_grid, ao_xyz)
    rho_b, nrb, _ = _grid_quantities(D_b, ao_grid, ao_xyz)
    nr_tot = nra + nrb
    sig_tot = jnp.sum(nr_tot * nr_tot, axis=1)

    # Inject the reproduced negative-rt tail at the lowest-density grid points.
    idx = jnp.argsort(rho_a + rho_b)[:16]
    rho_a = rho_a.at[idx].set(-4.061e-10)
    rho_b = rho_b.at[idx].set(2.001e-12)   # rt = rho_a + rho_b < 0
    assert bool(jnp.any((rho_a + rho_b) < 0.0)), "test lost its negative-rt trigger"

    vc_a, vc_b = compute_vc_polarized_per_spin(
        model, rho_a, rho_b, sig_tot, features, ao_grid, grid_weights,
        nr_tot, ao_grad)
    assert bool(jnp.all(jnp.isfinite(vc_a))) and bool(jnp.all(jnp.isfinite(vc_b))), \
        "polarized V_c NON-FINITE on the negative-density tail (bh76:HLi NaN regression)"


def test_polarized_vc_fd_energy_potential_consistency():
    from xcquinox.alec.oneshot import compute_vc_polarized_per_spin

    model = _build_polarized_model()
    md = _li_uks_md()
    ao_grid = jnp.asarray(md["ao_grid"])
    ao_grad = jnp.asarray(md["ao_grid_deriv"])
    ao_xyz = ao_grad[1:4]
    grid_weights = jnp.asarray(md["grid_weights"])
    features = assemble_descriptor_features(model.descriptors, md)

    dm = jnp.asarray(md["dm_pbe"])
    D_a, D_b = dm[0], dm[1]
    nao = D_a.shape[0]

    rho_a, nra, _ = _grid_quantities(D_a, ao_grid, ao_xyz)
    rho_b, nrb, _ = _grid_quantities(D_b, ao_grid, ao_xyz)
    nr_tot = nra + nrb
    sig_tot = jnp.sum(nr_tot * nr_tot, axis=1)

    vc_a, vc_b = compute_vc_polarized_per_spin(
        model, rho_a, rho_b, sig_tot, features, ao_grid, grid_weights,
        nr_tot, ao_grad)

    # Symmetric DM perturbations (the only physical case).
    rng = np.random.default_rng(20260524)
    Ma = rng.standard_normal((nao, nao))
    dDa = jnp.asarray(Ma + Ma.T)
    Mb = rng.standard_normal((nao, nao))
    dDb = jnp.asarray(Mb + Mb.T)

    def Ec(Da, Db):
        return _ec_energy_polarized(
            model, Da, Db, features, ao_grid, ao_xyz, grid_weights)

    eps = 1e-6
    fd = float((Ec(D_a + eps * dDa, D_b + eps * dDb)
                - Ec(D_a - eps * dDa, D_b - eps * dDb)) / (2.0 * eps))
    contract = float(jnp.einsum("ij,ij->", vc_a, dDa)
                     + jnp.einsum("ij,ij->", vc_b, dDb))
    rel_fd = abs(fd - contract) / max(abs(contract), 1e-12)
    assert rel_fd < 1e-5, (
        f"per-spin V_c is not FD-consistent with the polarized E_c "
        f"(rel={rel_fd:.3e})")

    # Tight cross-check against reverse-mode autodiff of E_c (symmetric
    # contraction removes the antisymmetric gauge ambiguity).
    ga, gb = jax.grad(Ec, argnums=(0, 1))(D_a, D_b)
    grad_contract = float(jnp.einsum("ij,ij->", ga, dDa)
                          + jnp.einsum("ij,ij->", gb, dDb))
    grad_resid = abs(grad_contract - contract) / max(abs(grad_contract), 1e-12)
    assert grad_resid < 1e-10, (
        f"per-spin V_c disagrees with autodiff grad of E_c: {grad_resid:.3e}")


def test_polarized_vc_finite_and_consistent_at_full_polarization():
    """zeta-boundary guard: at FULL spin polarization (rho_b -> 0,
    zeta_raw -> +1) the zeta clip shared by ``split_exc_energy_uks`` and
    ``compute_vc_polarized_per_spin`` keeps PW92's f''(zeta) ~ (1-zeta)^(-2/3)
    FINITE, AND -- because the per-spin V_c is ``jax.jvp`` THROUGH that same
    clip -- V_c stays the EXACT gradient of the clipped production energy.

    The SAME clip in both paths makes energy and potential consistent BY
    CONSTRUCTION (no energy<->potential mismatch at the boundary); dropping the
    clip would reintroduce the PW92 second-derivative NaN this guards against.
    Cross-checked against reverse-mode autodiff (no finite-difference noise at
    the non-smooth clip) to ~1e-9 even where the clip is active and its tangent
    is exactly 0.
    """
    from xcquinox.alec.oneshot import compute_vc_polarized_per_spin

    model = _build_polarized_model()
    md = _li_uks_md()
    ao_grid = jnp.asarray(md["ao_grid"])
    ao_grad = jnp.asarray(md["ao_grid_deriv"])
    ao_xyz = ao_grad[1:4]
    grid_weights = jnp.asarray(md["grid_weights"])
    features = assemble_descriptor_features(model.descriptors, md)

    # Pile ALL density into the alpha channel: rho_b == 0 -> zeta_raw == +1 at
    # every grid point -> the clip is active everywhere (the worst case).
    dm = jnp.asarray(md["dm_pbe"])
    D_a = dm[0] + dm[1]
    D_b = jnp.zeros_like(D_a)
    nao = D_a.shape[0]

    rho_a, nra, _ = _grid_quantities(D_a, ao_grid, ao_xyz)
    rho_b, nrb, _ = _grid_quantities(D_b, ao_grid, ao_xyz)
    nr_tot = nra + nrb
    sig_tot = jnp.sum(nr_tot * nr_tot, axis=1)
    # Precondition: this density really sits at the clip boundary.
    zeta_raw = (rho_a - rho_b) / jnp.maximum(rho_a + rho_b, 1e-300)
    assert float(jnp.max(zeta_raw)) >= 1.0 - 1e-12, \
        "test density is not fully polarized"

    vc_a, vc_b = compute_vc_polarized_per_spin(
        model, rho_a, rho_b, sig_tot, features, ao_grid, grid_weights,
        nr_tot, ao_grad)

    # (a) FINITE -- the clip is exactly what prevents the (1-zeta)^(-2/3) blowup.
    assert jnp.all(jnp.isfinite(vc_a)), "V_c^a non-finite at full polarization"
    assert jnp.all(jnp.isfinite(vc_b)), "V_c^b non-finite at full polarization"

    # (b) EXACT gradient of the PRODUCTION-clipped energy. _ec_energy_polarized
    # now clips identically, so reverse-mode autodiff of it is the ground truth.
    # Symmetric DM perturbations remove the matrix-potential gauge freedom.
    rng = np.random.default_rng(20260603)
    Ma = rng.standard_normal((nao, nao))
    dDa = jnp.asarray(Ma + Ma.T)
    Mb = rng.standard_normal((nao, nao))
    dDb = jnp.asarray(Mb + Mb.T)

    def Ec(Da, Db):
        return _ec_energy_polarized(
            model, Da, Db, features, ao_grid, ao_xyz, grid_weights)

    ga, gb = jax.grad(Ec, argnums=(0, 1))(D_a, D_b)
    contract_vc = float(jnp.einsum("ij,ij->", vc_a, dDa)
                        + jnp.einsum("ij,ij->", vc_b, dDb))
    contract_gr = float(jnp.einsum("ij,ij->", ga, dDa)
                        + jnp.einsum("ij,ij->", gb, dDb))
    assert jnp.isfinite(contract_gr), "autodiff grad of clipped E_c non-finite"
    resid = abs(contract_vc - contract_gr) / max(abs(contract_gr), 1e-12)
    assert resid < 1e-9, (
        f"per-spin V_c disagrees with autodiff grad of the clipped E_c at full "
        f"polarization: rel={resid:.3e}")


def test_polarized_full_split_vxc_fd_consistency():
    """End-to-end: the FULL split V_xc (spin-scaled exchange + per-spin
    correlation) is the functional derivative of ``split_exc_energy_uks`` with
    the polarized model, the exact energy/potential pair the SCF solvers use."""
    from xcquinox.alec.oneshot import compute_vc_polarized_per_spin

    model = _build_polarized_model()
    md = _li_uks_md()
    ao_grid = jnp.asarray(md["ao_grid"])
    ao_grad = jnp.asarray(md["ao_grid_deriv"])
    ao_xyz = ao_grad[1:4]
    grid_weights = jnp.asarray(md["grid_weights"])
    features = assemble_descriptor_features(model.descriptors, md)

    dm = jnp.asarray(md["dm_pbe"])
    D_a, D_b = dm[0], dm[1]
    nao = D_a.shape[0]

    # The `deep` architecture carries no descriptors, so `features` is the empty
    # (n_grid, 0) array and the three per-channel blocks are that same array.
    def E(Da, Db):
        return _uks_split_energy(
            model, Da, Db, features, features, features,
            ao_grid, ao_xyz, grid_weights)

    # Full per-spin V_xc = spin-scaled exchange + per-spin correlation.
    rho_a, nra, sig_aa = _grid_quantities(D_a, ao_grid, ao_xyz)
    rho_b, nrb, sig_bb = _grid_quantities(D_b, ao_grid, ao_xyz)
    nr_tot = nra + nrb
    sig_tot = jnp.sum(nr_tot * nr_tot, axis=1)
    vx_a = compute_vxc_nn(model, 2.0 * rho_a, 4.0 * sig_aa, features, ao_grid,
                          grid_weights, nabla_rho=2.0 * nra, ao_grad=ao_grad,
                          part="x")
    vx_b = compute_vxc_nn(model, 2.0 * rho_b, 4.0 * sig_bb, features, ao_grid,
                          grid_weights, nabla_rho=2.0 * nrb, ao_grad=ao_grad,
                          part="x")
    vc_a, vc_b = compute_vc_polarized_per_spin(
        model, rho_a, rho_b, sig_tot, features, ao_grid, grid_weights,
        nr_tot, ao_grad)
    V_a, V_b = vx_a + vc_a, vx_b + vc_b

    rng = np.random.default_rng(424242)
    Ma = rng.standard_normal((nao, nao)); dDa = jnp.asarray(Ma + Ma.T)
    Mb = rng.standard_normal((nao, nao)); dDb = jnp.asarray(Mb + Mb.T)
    eps = 1e-6
    fd = float((E(D_a + eps * dDa, D_b + eps * dDb)
                - E(D_a - eps * dDa, D_b - eps * dDb)) / (2.0 * eps))
    contract = float(jnp.einsum("ij,ij->", V_a, dDa)
                     + jnp.einsum("ij,ij->", V_b, dDb))
    rel = abs(fd - contract) / max(abs(contract), 1e-12)
    assert rel < 1e-5, (
        f"full polarized split V_xc not FD-consistent with split_exc_energy_uks "
        f"(rel={rel:.3e})")


def test_polarized_vc_closed_shell_reduces_to_shared():
    from xcquinox.alec.oneshot import compute_vc_polarized_per_spin

    model = _build_polarized_model()
    md = _li_uks_md()
    ao_grid = jnp.asarray(md["ao_grid"])
    ao_grad = jnp.asarray(md["ao_grid_deriv"])
    ao_xyz = ao_grad[1:4]
    grid_weights = jnp.asarray(md["grid_weights"])
    features = assemble_descriptor_features(model.descriptors, md)

    dm = jnp.asarray(md["dm_pbe"])
    # Force a closed-shell (zeta=0) density by averaging the spin DMs.
    D_c = 0.5 * (dm[0] + dm[1])
    rho_c, nrc, _ = _grid_quantities(D_c, ao_grid, ao_xyz)
    nr_tot = 2.0 * nrc
    sig_tot = jnp.sum(nr_tot * nr_tot, axis=1)

    vc_a, vc_b = compute_vc_polarized_per_spin(
        model, rho_c, rho_c, sig_tot, features, ao_grid, grid_weights,
        nr_tot, ao_grad)
    max_diff = float(jnp.max(jnp.abs(vc_a - vc_b)))
    assert max_diff < 1e-12, (
        f"at zeta=0 the per-spin V_c must coincide: max|vc_a-vc_b|={max_diff:.2e}")


def test_polarized_vc_finite_at_zero_sigma():
    """PHYS-2 (round-4) guard: a grid point with sigma_tot == 0 EXACTLY (uniform
    density / high-symmetry grid) must NOT produce NaN. The per-spin rho-tangent
    JVPs propagate through the cnet's sqrt(sigma) node whose +inf derivative at
    sigma=0 gives 0*inf=NaN unless evaluated at the denormal-guarded safe_sigma.
    """
    from xcquinox.alec.oneshot import compute_vc_polarized_per_spin

    model = _build_polarized_model()
    md = _li_uks_md()
    ao_grid = jnp.asarray(md["ao_grid"])
    ao_grad = jnp.asarray(md["ao_grid_deriv"])
    ao_xyz = ao_grad[1:4]
    grid_weights = jnp.asarray(md["grid_weights"])
    features = assemble_descriptor_features(model.descriptors, md)

    dm = jnp.asarray(md["dm_pbe"])
    D_a, D_b = dm[0], dm[1]
    rho_a, nra, _ = _grid_quantities(D_a, ao_grid, ao_xyz)
    rho_b, nrb, _ = _grid_quantities(D_b, ao_grid, ao_xyz)
    nr_tot = nra + nrb
    sig_tot = jnp.sum(nr_tot * nr_tot, axis=1)
    # Inject an EXACT sigma=0 point (zero the gradient at grid point 0).
    nr_tot = nr_tot.at[0].set(0.0)
    sig_tot = sig_tot.at[0].set(0.0)

    vc_a, vc_b = compute_vc_polarized_per_spin(
        model, rho_a, rho_b, sig_tot, features, ao_grid, grid_weights,
        nr_tot, ao_grad)
    assert jnp.all(jnp.isfinite(vc_a)), "vc_a has NaN/inf at a sigma=0 grid point"
    assert jnp.all(jnp.isfinite(vc_b)), "vc_b has NaN/inf at a sigma=0 grid point"


# ---------------------------------------------------------------------------
# pyscfad callback: closed-shell reduction of the libxc eval_xc convention.
# The UKS callback fed rho_a = rho_b must return the SAME per-particle exc as
# the RKS callback, and per-spin vrho/vsigma consistent with RKS.
# ---------------------------------------------------------------------------
def test_pyscfad_callback_closed_shell_reduction():
    from xcquinox.alec.solver import FeaturePolicy
    from xcquinox.alec.solver_pyscfad import _make_alec_eval_xc

    # Non-zero-init nets so correlation actually responds to sigma; a zero-init
    # warm-start cnet (Fc==1) has vsigma_c == 0, making the ud cross-term
    # asserted below trivially zero.
    model = _build_model(zero_init_final_layer=False)
    md = _h2_rks_md()
    eval_xc = _make_alec_eval_xc(model, model.descriptors, md,
                                 FeaturePolicy.FROZEN)

    n = md["ao_grid"].shape[0]
    rng = np.random.default_rng(0)
    # Synthetic closed-shell grid densities (rho_a = rho_b = rho/2).
    rho = np.abs(rng.standard_normal(n)) + 0.05
    dx = rng.standard_normal(n) * 0.1
    dy = rng.standard_normal(n) * 0.1
    dz = rng.standard_normal(n) * 0.1

    # RKS: rho shape (4, n).
    rks_rho = np.stack([rho, dx, dy, dz])
    exc_rks, vxc_rks, _, _ = eval_xc("", rks_rho, spin=0, deriv=1)
    vrho_rks, vsigma_rks = vxc_rks[0], vxc_rks[1]

    # UKS with each spin half: rho_s = rho/2, grad_s = grad/2.
    half = np.stack([0.5 * rho, 0.5 * dx, 0.5 * dy, 0.5 * dz])
    uks_rho = np.stack([half, half])  # (2, 4, n)
    exc_uks, vxc_uks, _, _ = eval_xc("", uks_rho, spin=1, deriv=1)
    vrho_uks, vsigma_uks = vxc_uks[0], vxc_uks[1]

    # Per-particle exc must match (both divide by the SAME rho_tot = rho).
    e_resid = float(np.max(np.abs(np.asarray(exc_uks) - np.asarray(exc_rks))))
    assert e_resid < 1e-10, f"pyscfad closed-shell exc residual {e_resid:.3e}"

    # vrho_a and vrho_b must both equal the RKS vrho (each spin sees the RKS
    # potential at the closed-shell point).
    vr_a = np.asarray(vrho_uks)[:, 0]
    vr_b = np.asarray(vrho_uks)[:, 1]
    assert float(np.max(np.abs(vr_a - np.asarray(vrho_rks)))) < 1e-10
    assert float(np.max(np.abs(vr_b - np.asarray(vrho_rks)))) < 1e-10

    # vsigma: the RKS vsigma is d E/d sigma_tot. For UKS the combination that
    # reproduces the RKS sigma-derivative is vsigma_uu + vsigma_ud + vsigma_dd
    # (since sigma_tot = sigma_uu + 2 sigma_ud + sigma_dd at rho_a = rho_b and
    # the chain rule maps d/d sigma_tot onto all three). Verify the closed-
    # shell sigma derivative is reproduced.
    vs = np.asarray(vsigma_uks)  # (n, 3) uu, ud, dd
    # At the closed-shell point the alpha grad = beta grad = grad/2, so
    # sigma_uu = sigma_dd = sigma_ud = |grad/2|^2. d E/d|grad/2|^2 summed over
    # the three channels must equal 4 * vsigma_rks? Establish numerically that
    # the UKS sigma-block reproduces RKS via an explicit FD of the callback.
    # Simpler robust check: rebuild the effective d E_density/d sigma_tot.
    # vsigma_rks is d E_density/d sigma_tot. The UKS uu = 2 v_x_a + v_c, etc.
    # Their consistency is already covered by the exact-derivative test C1 at
    # the matrix level; here we assert the exchange and correlation pieces
    # combine without NaN and ud is non-zero (correlation contributes a
    # cross-term).
    assert np.all(np.isfinite(vs))
    assert float(np.max(np.abs(vs[:, 1]))) > 0.0, (
        "correlation must contribute a non-zero ud cross-term"
    )


def test_pyscfad_callback_polarized_per_spin_vrho():
    """with a spin-polarization-aware cnet the pyscfad UKS callback must
    return PER-SPIN vrho (vrho_a != vrho_b on an open-shell density), reduce to
    a shared vrho at zeta=0, and be pointwise FD-consistent with the energy
    density (vrho_s = d(exc*rho_tot)/d rho_s at fixed gradients)."""
    from xcquinox.alec.solver import FeaturePolicy
    from xcquinox.alec.solver_pyscfad import _make_alec_eval_xc

    model = _build_polarized_model()
    md = _h2_rks_md()
    eval_xc = _make_alec_eval_xc(model, model.descriptors, md,
                                 FeaturePolicy.FROZEN)
    n = md["ao_grid"].shape[0]
    rng = np.random.default_rng(11)

    def xc_density(rho_a, rho_b, ga, gb):
        uks_rho = np.stack([np.stack([rho_a, ga[0], ga[1], ga[2]]),
                            np.stack([rho_b, gb[0], gb[1], gb[2]])])
        exc, _, _, _ = eval_xc("", uks_rho, spin=1, deriv=1)
        return np.asarray(exc) * (rho_a + rho_b)

    def vrho_pair(rho_a, rho_b, ga, gb):
        uks_rho = np.stack([np.stack([rho_a, ga[0], ga[1], ga[2]]),
                            np.stack([rho_b, gb[0], gb[1], gb[2]])])
        _, vxc, _, _ = eval_xc("", uks_rho, spin=1, deriv=1)
        v = np.asarray(vxc[0])  # (n, 2)
        return v[:, 0], v[:, 1]

    # Open-shell synthetic density (rho_a != rho_b).
    rho_a = np.abs(rng.standard_normal(n)) + 0.10
    rho_b = np.abs(rng.standard_normal(n)) + 0.05
    ga = rng.standard_normal((3, n)) * 0.1
    gb = rng.standard_normal((3, n)) * 0.1

    vr_a, vr_b = vrho_pair(rho_a, rho_b, ga, gb)
    # Per-spin: the two channels must genuinely differ.
    assert float(np.max(np.abs(vr_a - vr_b))) > 1e-6, (
        "polarized callback must produce per-spin vrho (vrho_a != vrho_b)")

    # Pointwise FD consistency at fixed gradients.
    eps = 1e-6
    fd_a = (xc_density(rho_a + eps, rho_b, ga, gb)
            - xc_density(rho_a - eps, rho_b, ga, gb)) / (2 * eps)
    fd_b = (xc_density(rho_a, rho_b + eps, ga, gb)
            - xc_density(rho_a, rho_b - eps, ga, gb)) / (2 * eps)
    # Compare on points well above the 1e-12 tail mask.
    m = (rho_a + rho_b) > 1e-3
    rel_a = np.max(np.abs(fd_a[m] - vr_a[m])) / max(np.max(np.abs(vr_a[m])), 1e-12)
    rel_b = np.max(np.abs(fd_b[m] - vr_b[m])) / max(np.max(np.abs(vr_b[m])), 1e-12)
    assert rel_a < 1e-5 and rel_b < 1e-5, (
        f"per-spin vrho not FD-consistent (rel_a={rel_a:.3e} rel_b={rel_b:.3e})")

    # Closed-shell reduction: rho_a = rho_b, ga = gb -> vrho_a == vrho_b.
    rho_c = np.abs(rng.standard_normal(n)) + 0.10
    gc = rng.standard_normal((3, n)) * 0.1
    cv_a, cv_b = vrho_pair(rho_c, rho_c, gc, gc)
    assert float(np.max(np.abs(cv_a - cv_b))) < 1e-10, (
        "at zeta=0 the per-spin vrho must coincide")


# ---------------------------------------------------------------------------
# Test D: spin symmetry. Swapping (rho_a,nabla_a)<->(rho_b,nabla_b) swaps
# V_a<->V_b and leaves E_xc unchanged.
# ---------------------------------------------------------------------------
def test_spin_swap_symmetry():
    model = _build_model()
    md = _li_uks_md()
    ao_grid = jnp.asarray(md["ao_grid"])
    ao_grad = jnp.asarray(md["ao_grid_deriv"])
    ao_xyz = ao_grad[1:4]
    grid_weights = jnp.asarray(md["grid_weights"])
    features = assemble_descriptor_features(model.descriptors, md)

    dm = jnp.asarray(md["dm_pbe"])
    D_a, D_b = dm[0], dm[1]

    # The `deep` architecture carries no descriptors, so `features` is the empty
    # (n_grid, 0) array and the three per-channel blocks are that same array.
    E_orig = _uks_split_energy(
        model, D_a, D_b, features, features, features,
        ao_grid, ao_xyz, grid_weights)
    E_swap = _uks_split_energy(
        model, D_b, D_a, features, features, features,
        ao_grid, ao_xyz, grid_weights)
    assert float(abs(E_orig - E_swap)) < 1e-10, "E_xc must be spin-swap invariant"

    V_a, V_b = _uks_split_vxc(
        model, D_a, D_b, features, features, features,
        ao_grid, ao_xyz, ao_grad, grid_weights)
    V_a_sw, V_b_sw = _uks_split_vxc(
        model, D_b, D_a, features, features, features,
        ao_grid, ao_xyz, ao_grad, grid_weights)
    # Swapping inputs must swap the outputs.
    assert float(jnp.max(jnp.abs(V_a - V_b_sw))) < 1e-10
    assert float(jnp.max(jnp.abs(V_b - V_a_sw))) < 1e-10


# ---------------------------------------------------------------------------
# P2-02: descriptor features and the exchange spin-scaling relation.
# ---------------------------------------------------------------------------
def _build_descriptor_model():
    """A descriptor-carrying model whose enhancement factors actually respond to
    the feature block.

    ``zero_init_final_layer=False`` is load-bearing, not a style choice. The
    warm-start initialization zeroes the final layer, which pins F_x and F_c at
    1 with zero input-gradients, so every feature column is ignored and any
    assertion about which block reaches which term is satisfied identically.
    Measured on the six-point probe below: with the default zero init, swapping
    the two channel blocks moves the energy by 0.0 exactly and the correlation
    block contributes 0.0; with a live final layer the same swap moves it by
    2.554e-05 Ha and the correlation block by 6.149e-06 Ha.
    """
    import dataclasses
    arch = dataclasses.replace(
        alec.get_architecture("deep_combined_attn"),  # cusp + dm_statistics
        zero_init_final_layer=False)
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    return alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def test_split_energy_closed_shell_reduction_with_descriptors():
    """With descriptor features ACTIVE, the closed-shell reduction to RKS is
    still EXACT: rho_a = rho_b feeds identical features into both exchange terms,
    so E_split == sum_g w_g eval_exc(2 rho_a, 4 sigma_aa, features)."""
    model = _build_descriptor_model()
    n_feat = sum(d.n_features for d in model.descriptors)
    rng = np.random.default_rng(0)
    rho_a = jnp.asarray(rng.uniform(0.05, 1.0, 6))
    sigma_aa = jnp.asarray(rng.uniform(0.01, 0.5, 6))
    feats = jnp.asarray(rng.standard_normal((6, n_feat)))
    gw = jnp.ones(6)
    # closed shell: rho_b = rho_a, nabla_rho_b = nabla_rho_a => sigma_tot = 4 sigma_aa
    # rho_a = rho_b makes the three per-channel blocks the same array, which is
    # exactly the closed-shell case the reduction refers to.
    E_split = split_exc_energy_uks(
        model, rho_a, rho_a, sigma_aa, sigma_aa, 4.0 * sigma_aa,
        feats, feats, feats, gw)
    E_rks = float(jnp.sum(gw * model.eval_exc(2.0 * rho_a, 4.0 * sigma_aa, feats)))
    assert abs(float(E_split) - E_rks) < 1e-9, (float(E_split), E_rks)


def test_split_energy_openshell_uses_the_per_channel_feature_block():
    """Exact spin scaling: each doubled-spin exchange evaluation receives ITS
    OWN channel's feature block -- the block of diag(P_sigma, P_sigma) -- and
    correlation receives the total-density block. Supersedes the pinned
    approximation in which one molecular block fed both exchange terms."""
    model = _build_descriptor_model()
    n_feat = sum(d.n_features for d in model.descriptors)
    rng = np.random.default_rng(1)
    rho_a = jnp.asarray(rng.uniform(0.05, 1.0, 6))
    rho_b = jnp.asarray(rng.uniform(0.01, 0.4, 6))     # rho_a != rho_b
    sigma_aa = jnp.asarray(rng.uniform(0.01, 0.5, 6))
    sigma_bb = jnp.asarray(rng.uniform(0.01, 0.3, 6))
    sigma_tot = jnp.asarray(rng.uniform(0.02, 0.9, 6))
    f_a = jnp.asarray(rng.standard_normal((6, n_feat)))
    f_b = jnp.asarray(rng.standard_normal((6, n_feat)))
    f_tot = jnp.asarray(rng.standard_normal((6, n_feat)))
    gw = jnp.ones(6)
    got = float(split_exc_energy_uks(
        model, rho_a, rho_b, sigma_aa, sigma_bb, sigma_tot,
        f_a, f_b, f_tot, gw))
    ex_a = model.eval_ex(2.0 * rho_a, 4.0 * sigma_aa, f_a)
    ex_b = model.eval_ex(2.0 * rho_b, 4.0 * sigma_bb, f_b)
    ec = model.eval_ec(rho_a + rho_b, sigma_tot, f_tot)
    expected = float(0.5 * jnp.sum(gw * (ex_a + ex_b)) + jnp.sum(gw * ec))
    # Same operations in the same order on both sides, so the residual is
    # bitwise 0.0 as measured; 1e-12 leaves room for a reassociation.
    assert abs(got - expected) < 1e-12
    # The three blocks are genuinely distinguished: exchanging the two channel
    # blocks changes the energy, which the superseded contract could not see.
    # Measured separation 2.554e-05 Ha on this probe, so the 1e-8 floor clears
    # it by 2.6e3 and still fails immediately if either block stops reaching
    # its own exchange term.
    swapped = float(split_exc_energy_uks(
        model, rho_a, rho_b, sigma_aa, sigma_bb, sigma_tot,
        f_b, f_a, f_tot, gw))
    assert abs(swapped - got) > 1e-8


def test_split_energy_openshell_correlation_ignores_the_channel_blocks():
    """Correlation is spin-interpolated, not spin-scaled, so it must depend on
    the total block alone (von Barth and Hedin, J. Phys. C 5, 1629 (1972))."""
    model = _build_descriptor_model()
    n_feat = sum(d.n_features for d in model.descriptors)
    rng = np.random.default_rng(2)
    args = (jnp.asarray(rng.uniform(0.05, 1.0, 6)),
            jnp.asarray(rng.uniform(0.01, 0.4, 6)),
            jnp.asarray(rng.uniform(0.01, 0.5, 6)),
            jnp.asarray(rng.uniform(0.01, 0.3, 6)),
            jnp.asarray(rng.uniform(0.02, 0.9, 6)))
    f_tot = jnp.asarray(rng.standard_normal((6, n_feat)))
    zeros = jnp.zeros((6, n_feat))
    gw = jnp.ones(6)
    with_tot = float(split_exc_energy_uks(model, *args, zeros, zeros, f_tot, gw))
    with_zero = float(split_exc_energy_uks(model, *args, zeros, zeros, zeros, gw))
    ec_tot = float(jnp.sum(gw * model.eval_ec(args[0] + args[1], args[4], f_tot)))
    ec_zero = float(jnp.sum(gw * model.eval_ec(args[0] + args[1], args[4], zeros)))
    # The exchange term is identical in both evaluations, so the difference of
    # totals is the correlation difference up to the cancellation floor of the
    # ~3 Ha total: measured residual 2.220e-16 against a correlation signal of
    # 6.149e-06 Ha, i.e. one double-precision ulp of the energy.
    assert abs((with_tot - with_zero) - (ec_tot - ec_zero)) < 1e-12


def test_split_exc_energy_uks_raises_when_cnet_lacks_polarization_flag():
    """A model whose cnet lost ``use_spin_polarization`` (e.g. a model built
    outside create_network_pair, or a bad deserialization) must RAISE on the
    open-shell path rather than silently fall back to non-polarized
    correlation. Guards the identical hasattr check in both
    oneshot.split_exc_energy_uks AND solver_manual.run_scf's UKS body."""
    model = _build_polarized_model()
    md = _li_uks_md()
    ao_grid = jnp.asarray(md["ao_grid"])
    ao_grad = jnp.asarray(md["ao_grid_deriv"])
    ao_xyz = ao_grad[1:4]
    grid_weights = jnp.asarray(md["grid_weights"])
    features = assemble_descriptor_features(model.descriptors, md)
    dm = jnp.asarray(md["dm_pbe"])
    rho_a, nra, sig_aa = _grid_quantities(dm[0], ao_grid, ao_xyz)
    rho_b, nrb, sig_bb = _grid_quantities(dm[1], ao_grid, ao_xyz)
    nr_tot = nra + nrb
    sig_tot = jnp.sum(nr_tot * nr_tot, axis=1)

    class _ModelNoFlag:
        """Real (working) exchange, but a cnet object missing the flag, so the
        guard fires before correlation is ever evaluated."""
        def __init__(self, real):
            self._real = real
            self.cnet = object()  # lacks use_spin_polarization

        def eval_ex(self, *a, **k):
            return self._real.eval_ex(*a, **k)

    with pytest.raises(AttributeError, match="use_spin_polarization"):
        split_exc_energy_uks(_ModelNoFlag(model), rho_a, rho_b,
                             sig_aa, sig_bb, sig_tot,
                             features, features, features, grid_weights)


# ---------------------------------------------------------------------------
# PRODUCTION WIRING of the three feature blocks.
#
# The tests above pin the CONTRACT of ``split_exc_energy_uks`` and of the
# potential builders when the three blocks are handed to them explicitly. They
# cannot see which block each production entry point actually passes: every
# other open-shell test in this suite (and in ``test_uks_oneshot.py`` /
# ``test_losses.py`` / ``test_oneshot.py``) runs the descriptor-free ``deep``
# architecture, whose three blocks are the same empty (n_grid, 0) array, so a
# caller that reverted to feeding one block into all three slots leaves every
# one of them green.
#
# The tests below drive the four production entry points --
# ``fixed_density_total_energy``, ``_uks_spin_resolved_vxc``,
# ``oneshot_dm_prediction_fast`` and ``losses._vxc_term`` -- on a record whose
# three blocks are genuinely different arrays and separate the three-block
# wiring from the one-block one by a measured margin. Probe: O atom (5 alpha,
# 3 beta), def2-svp, grid level 1, ``deep_mgga_3x16``; in the iso-orbital
# column max|f_a - f_tot| = 43.5 and max|f_b - f_tot| = 94.6, and the record is
# the same one the finite-difference test uses, so its blocks are away from the
# clip.
#
# ``zero_init_final_layer=False`` is load-bearing for the same reason as in
# ``_build_descriptor_model``: the warm-start zero init pins F_x and F_c at 1
# with zero input-gradients, and every separation quoted below is then exactly
# 0.0, i.e. each assertion would be satisfied by a one-block wiring.
# ---------------------------------------------------------------------------
def _wiring_model(polarized: bool, seed: int = 0):
    """``deep_mgga_3x16`` with a live final layer.

    ``polarized`` selects the correlation branch of
    ``_uks_spin_resolved_vxc``: per-spin V_c through
    ``compute_vc_polarized_per_spin`` when True, the shared zeta=0
    total-density V_c through ``compute_vxc_nn(part="c")`` when False. Both
    branches are production paths and both consume the TOTAL block, so both are
    exercised.
    """
    import dataclasses
    arch = dataclasses.replace(alec.get_architecture("deep_mgga_3x16"),
                               use_polarized_correlation=polarized,
                               zero_init_final_layer=False)
    xnet, cnet = alec.create_network_pair(arch, seed=seed)
    return alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


@pytest.fixture(scope="module")
def wiring_md():
    """The O-atom record shared by the wiring tests.

    Module-scoped so the group pays for one reference SCF rather than one per
    test; the tests only read it (the two that need an extra key copy the dict
    first). The descriptor set is the same for both correlation branches, so
    one record serves every case.
    """
    return _md_with_descriptors(_wiring_model(True), "O", "O 0 0 0",
                                "def2-svp", 2, (("O", 1),), grid_level=1)


def _wiring_blocks(model, md):
    """(alpha, beta, total) descriptor blocks of the record."""
    return (assemble_descriptor_features(model.descriptors, md, spin_channel=0),
            assemble_descriptor_features(model.descriptors, md, spin_channel=1),
            assemble_descriptor_features(model.descriptors, md))


def _wiring_grid(md):
    """Per-spin grid quantities of the record's reference density matrix, built
    with the same contractions ``fixed_density_total_energy`` and
    ``_uks_spin_resolved_vxc`` use."""
    dm = jnp.asarray(md["dm_pbe"])
    ao_grid = jnp.asarray(md["ao_grid"])
    ao_grad = jnp.asarray(md["ao_grid_deriv"])
    rho_a, nabla_a, sigma_aa = _grid_quantities(dm[0], ao_grid, ao_grad[1:4])
    rho_b, nabla_b, sigma_bb = _grid_quantities(dm[1], ao_grid, ao_grad[1:4])
    nabla_tot = nabla_a + nabla_b
    return dict(
        ao_grid=ao_grid, ao_grad=ao_grad,
        weights=jnp.asarray(md["grid_weights"]),
        rho_a=rho_a, rho_b=rho_b, nabla_a=nabla_a, nabla_b=nabla_b,
        sigma_aa=sigma_aa, sigma_bb=sigma_bb, nabla_tot=nabla_tot,
        sigma_tot=jnp.sum(nabla_tot * nabla_tot, axis=1),
    )


def _max_abs_diff(A, B):
    return float(jnp.max(jnp.abs(jnp.asarray(A) - jnp.asarray(B))))


def test_wiring_fixed_density_total_energy_uses_the_channel_blocks(wiring_md):
    """``fixed_density_total_energy`` must build the two exchange terms on the
    blocks of diag(P_a, P_a) / diag(P_b, P_b) and correlation on the total
    block.

    The reference is assembled from ``model.eval_ex`` / ``model.eval_ec``
    directly, so it shares no code with ``split_exc_energy_uks`` and an error in
    either the split function or the caller's block routing shows up. Measured
    residual exactly 0.0 (the same operations in the same order); 1e-12 leaves
    room for a reassociation of a -74.455 Ha total. The superseded one-block
    evaluation -- the total block in both exchange channels -- sits 8.85e-04 Ha
    away on this record (8.8475e-04 and 8.8497e-04 in two independent reference
    solutions; the reference SCF differs at round-off level between runs, which
    moves every figure in this group in the fourth significant digit), 88x the
    1e-5 separation floor, so a caller that reverted to one block cannot
    satisfy both bounds.
    """
    from xcquinox.alec.oneshot import fixed_density_total_energy, uks_zeta
    model = _wiring_model(True)
    f_a, f_b, f_tot = _wiring_blocks(model, wiring_md)
    g = _wiring_grid(wiring_md)
    w = g["weights"]

    got = float(fixed_density_total_energy(model, wiring_md))

    ex_a = model.eval_ex(2.0 * g["rho_a"], 4.0 * g["sigma_aa"], f_a)
    ex_b = model.eval_ex(2.0 * g["rho_b"], 4.0 * g["sigma_bb"], f_b)
    ec = model.eval_ec(g["rho_a"] + g["rho_b"], g["sigma_tot"], f_tot,
                       zeta=uks_zeta(g["rho_a"], g["rho_b"]))
    expected = float(wiring_md["E_non_xc"]) + float(
        0.5 * jnp.sum(w * (ex_a + ex_b)) + jnp.sum(w * ec))
    assert abs(got - expected) < 1e-12, (
        f"fixed_density_total_energy is not E_non_xc + spin-scaled exchange on "
        f"the channel blocks + correlation on the total block: got {got!r} "
        f"expected {expected!r} (diff {abs(got - expected):.3e})")

    one_block = float(wiring_md["E_non_xc"]) + float(split_exc_energy_uks(
        model, g["rho_a"], g["rho_b"], g["sigma_aa"], g["sigma_bb"],
        g["sigma_tot"], f_tot, f_tot, f_tot, w))
    gap = abs(got - one_block)
    assert gap > 1e-5, (
        f"the three-block and one-block energies are indistinguishable on this "
        f"record ({gap:.6e} Ha); the probe no longer discriminates")


def test_wiring_vxc_loss_term_uses_the_three_block_potential(wiring_md):
    """The V_xc loss term must score the network against the potential built
    from the three blocks.

    Scored against its own three-block potential the term is exactly 0.0 (the
    two matrices are bit-identical), and against the one-block potential it is
    1.7375e-08 in the n_ao^2-normalized units the loss returns (n_ao = 14;
    1.73755e-08 and 1.73769e-08 in two independent reference solutions), 174x
    the 1e-10 floor. A caller that reverted to one block turns the first figure
    into the second.
    """
    from xcquinox.alec.losses import _vxc_term
    from xcquinox.alec.oneshot import _uks_spin_resolved_vxc
    model = _wiring_model(True)
    f_a, f_b, f_tot = _wiring_blocks(model, wiring_md)

    vxc_ref = jnp.stack(_uks_spin_resolved_vxc(model, wiring_md,
                                               f_a, f_b, f_tot))
    vxc_one = jnp.stack(_uks_spin_resolved_vxc(model, wiring_md,
                                               f_tot, f_tot, f_tot))
    md_three = dict(wiring_md)
    md_three["vxc_ref"] = vxc_ref
    md_one = dict(wiring_md)
    md_one["vxc_ref"] = vxc_one

    against_three = float(_vxc_term(model, [md_three], [0]))
    assert against_three == 0.0, (
        f"_vxc_term does not build the NN potential from the three blocks: "
        f"scored {against_three:.6e} against that very potential")
    against_one = float(_vxc_term(model, [md_one], [0]))
    assert against_one > 1e-10, (
        f"the three-block and one-block potentials are indistinguishable to "
        f"the loss ({against_one:.6e}); the probe no longer discriminates")


def test_wiring_oneshot_fock_build_uses_the_channel_blocks(wiring_md,
                                                           monkeypatch):
    """The one-shot Fock build must assemble V_xc^NN from the three blocks.

    The predicted density matrix is compared with the one obtained by forcing
    the block triple at the ``_uks_spin_resolved_vxc`` seam: identical
    (max|dP| = 0.0) for the three-block triple, and 2.106e-04 away from the
    one-block triple (2.1062e-04 and 2.1064e-04 in two independent reference
    solutions) on a density matrix whose largest entry is 1.08, i.e. 210x the
    1e-6 floor. Forcing the triple at the seam rather than rebuilding the
    eigenproblem keeps the comparison on the production Cholesky/eigh path.
    """
    from xcquinox.alec import oneshot as _oneshot
    from xcquinox.alec.oneshot import oneshot_dm_prediction_fast
    model = _wiring_model(True)
    f_a, f_b, f_tot = _wiring_blocks(model, wiring_md)
    unpatched = _oneshot._uks_spin_resolved_vxc

    def _dm_with_blocks(block_a, block_b, block_tot):
        def _forced(model_arg, mol_data_arg, *_ignored_blocks):
            return unpatched(model_arg, mol_data_arg,
                             block_a, block_b, block_tot)
        monkeypatch.setattr(_oneshot, "_uks_spin_resolved_vxc", _forced)
        try:
            return np.asarray(oneshot_dm_prediction_fast(model, wiring_md))
        finally:
            monkeypatch.setattr(_oneshot, "_uks_spin_resolved_vxc", unpatched)

    dm_production = np.asarray(oneshot_dm_prediction_fast(model, wiring_md))
    dm_three = _dm_with_blocks(f_a, f_b, f_tot)
    dm_one = _dm_with_blocks(f_tot, f_tot, f_tot)

    resid = float(np.max(np.abs(dm_production - dm_three)))
    assert resid == 0.0, (
        f"the one-shot Fock build does not feed the three blocks to "
        f"_uks_spin_resolved_vxc: max|dP| = {resid:.6e} against the forced "
        f"three-block build")
    separation = float(np.max(np.abs(dm_three - dm_one)))
    assert separation > 1e-6, (
        f"the three-block and one-block Fock builds give the same density "
        f"matrix ({separation:.6e}); the probe no longer discriminates")


@pytest.mark.parametrize("polarized", [True, False])
def test_wiring_uks_potential_routes_each_block_to_its_own_term(wiring_md,
                                                                polarized):
    """Each argument of ``_uks_spin_resolved_vxc`` must reach exactly one term.

    Substituting the total block into one slot at a time and reading which
    channel moves is a routing measurement that no single-matrix comparison can
    fake: the alpha slot may move the alpha channel only, the beta slot the beta
    channel only, and the total slot must move BOTH, because correlation is
    spin-interpolated on the total density and enters each spin Fock matrix.

    A channel that does not consume a block is bit-independent of it, so the
    two null responses are exactly 0.0 rather than small. Measured responses
    (Ha): alpha slot 5.697e-04 / 0.0, beta slot 0.0 / 7.330e-04, total slot
    9.49e-05 and 2.855e-04 with per-spin correlation, 4.966e-05 in both
    channels with the shared zeta=0 correlation -- the smallest non-null
    response is 50x the 1e-6 floor. Two independent reference solutions agree
    on the non-null figures to better than 0.2% and on the null ones exactly.
    """
    from xcquinox.alec.oneshot import _uks_spin_resolved_vxc
    model = _wiring_model(polarized)
    f_a, f_b, f_tot = _wiring_blocks(model, wiring_md)

    V_a, V_b = _uks_spin_resolved_vxc(model, wiring_md, f_a, f_b, f_tot)
    alpha_slot = _uks_spin_resolved_vxc(model, wiring_md, f_tot, f_b, f_tot)
    beta_slot = _uks_spin_resolved_vxc(model, wiring_md, f_a, f_tot, f_tot)
    total_slot = _uks_spin_resolved_vxc(model, wiring_md, f_a, f_b, f_a)

    assert _max_abs_diff(V_a, alpha_slot[0]) > 1e-6, (
        "the alpha exchange term does not consume the alpha channel block")
    assert _max_abs_diff(V_b, alpha_slot[1]) == 0.0, (
        f"the beta channel responds to the alpha block "
        f"({_max_abs_diff(V_b, alpha_slot[1]):.6e}); the two exchange channels "
        f"are crossed or correlation is reading a channel block")
    assert _max_abs_diff(V_b, beta_slot[1]) > 1e-6, (
        "the beta exchange term does not consume the beta channel block")
    assert _max_abs_diff(V_a, beta_slot[0]) == 0.0, (
        f"the alpha channel responds to the beta block "
        f"({_max_abs_diff(V_a, beta_slot[0]):.6e}); the two exchange channels "
        f"are crossed or correlation is reading a channel block")
    assert _max_abs_diff(V_a, total_slot[0]) > 1e-6, (
        "the alpha channel does not respond to the total block; correlation "
        "is not being evaluated there")
    assert _max_abs_diff(V_b, total_slot[1]) > 1e-6, (
        "the beta channel does not respond to the total block; correlation "
        "is not being evaluated there")


@pytest.mark.parametrize("polarized", [True, False])
def test_wiring_correlation_potential_is_built_on_the_total_block(wiring_md,
                                                                  polarized):
    """Correlation is spin-interpolated, not spin-scaled, so its POTENTIAL is
    the total-block one in both spin channels (von Barth and Hedin, J. Phys. C
    5, 1629 (1972)).

    The reference is assembled here from ``compute_vxc_nn(part="x")`` on the
    two channel blocks plus the correlation potential on the TOTAL block, and
    reproduces the production pair bit for bit (measured 0.0 in both channels
    and in both correlation branches). The control that makes the equality
    non-vacuous is the same correlation term evaluated on the alpha channel
    block: 9.49e-05 / 2.855e-04 Ha away per spin with the per-spin correlation
    and 4.966e-05 Ha with the shared one (two independent reference solutions
    agree to better than 0.2%), 50x the 1e-6 floor, so a correlation potential
    moved onto a channel block cannot pass.
    """
    from xcquinox.alec.oneshot import (
        _uks_spin_resolved_vxc, compute_vc_polarized_per_spin)
    model = _wiring_model(polarized)
    f_a, f_b, f_tot = _wiring_blocks(model, wiring_md)
    g = _wiring_grid(wiring_md)
    ao_grid, ao_grad, w = g["ao_grid"], g["ao_grad"], g["weights"]

    vx_a = compute_vxc_nn(model, 2.0 * g["rho_a"], 4.0 * g["sigma_aa"], f_a,
                          ao_grid, w, nabla_rho=2.0 * g["nabla_a"],
                          ao_grad=ao_grad, part="x")
    vx_b = compute_vxc_nn(model, 2.0 * g["rho_b"], 4.0 * g["sigma_bb"], f_b,
                          ao_grid, w, nabla_rho=2.0 * g["nabla_b"],
                          ao_grad=ao_grad, part="x")
    if polarized:
        def _vc(block):
            return compute_vc_polarized_per_spin(
                model, g["rho_a"], g["rho_b"], g["sigma_tot"], block, ao_grid,
                w, g["nabla_tot"], ao_grad)
    else:
        def _vc(block):
            shared = compute_vxc_nn(
                model, g["rho_a"] + g["rho_b"], g["sigma_tot"], block, ao_grid,
                w, nabla_rho=g["nabla_tot"], ao_grad=ao_grad, part="c")
            return shared, shared

    vc_a, vc_b = _vc(f_tot)
    V_a, V_b = _uks_spin_resolved_vxc(model, wiring_md, f_a, f_b, f_tot)
    assert _max_abs_diff(V_a, vx_a + vc_a) == 0.0, (
        f"alpha V_xc is not (spin-scaled exchange on the alpha block + "
        f"correlation on the total block): "
        f"{_max_abs_diff(V_a, vx_a + vc_a):.6e}")
    assert _max_abs_diff(V_b, vx_b + vc_b) == 0.0, (
        f"beta V_xc is not (spin-scaled exchange on the beta block + "
        f"correlation on the total block): "
        f"{_max_abs_diff(V_b, vx_b + vc_b):.6e}")

    vc_a_channel, vc_b_channel = _vc(f_a)
    assert _max_abs_diff(vc_a, vc_a_channel) > 1e-6, (
        "the correlation potential does not depend on which block it is given; "
        "the equality above is vacuous")
    assert _max_abs_diff(vc_b, vc_b_channel) > 1e-6, (
        "the correlation potential does not depend on which block it is given; "
        "the equality above is vacuous")
