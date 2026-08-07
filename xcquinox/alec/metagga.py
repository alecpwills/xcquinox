"""Meta-GGA kinetic-energy density (tau) and SCAN iso-orbital indicator (alpha).

This is the meta-GGA (Jacob's-ladder rung 3) ingredient of the DFS functional: the
iso-orbital indicator introduced by SCAN (Sun, Ruzsinszky, Perdew, PRL 115, 036402
(2015), Eq. 2), reused by DFS (Dick & Fernandez-Serra, PRB 104, L161109 (2021),
Eq. 6):

    alpha(r) = (tau - tau_W) / tau_unif,   tau_W = |grad n|^2 / (8 n),
               tau_unif = (3/10)(3 pi^2)^{2/3} n^{5/3}

where ``tau`` is the (total) positive kinetic-energy density

    tau(r) = 1/2 sum_i |grad psi_i(r)|^2
           = 1/2 sum_{mu nu} P_{mu nu} grad chi_mu(r) . grad chi_nu(r).

Like the rung-3.5 occupancy (:mod:`xcquinox.alec.rung35`), ``tau`` is a **linear
contraction of the live one-particle DM** against a quantity that is a **constant
precompute** -- here the AO gradients on the grid, ``grad chi_mu(r)``, which are
already computed (``deriv=1``) for the GGA reduced gradient ``s``. So ``tau`` is
self-consistent (a functional of the live DM, recomputed each SCF cycle),
differentiable through the SCF, and needs **no new integrals, no laplacian, no
deriv=2**. ``alpha`` is then a pointwise function of ``(tau, rho, sigma)``.

alpha = 1 for the uniform electron gas (tau_W = 0, tau = tau_unif); alpha = 0 for a
single orbital (tau = tau_W). It is clamped to >= 0 for low-density grid noise,
matching the repo's existing numpy formula
(:func:`xcquinox.alec.subset_selection.compute_descriptor_triple`).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# Floor for the density / uniform-gas denominators (matches
# subset_selection.compute_descriptor_triple so alpha is bit-comparable).
_RHO_FLOOR: float = 1e-30

# alpha = (tau - tau_W)/tau_unif divides by n and n^{5/3}; on a low-density grid
# tail its VALUE reaches ~1e4-1e7 (alpha is not O(1) even at normal bases such as
# H2O/def2-svp) and its DERIVATIVE (d alpha/d n ~ n^{-8/3})
# blows up (~1e28 in the deep tail), and the unrolled FULL-SCF backprop compounds
# that XC kernel into a NaN training gradient -- the same failure class as the
# polarized-correlation zeta clip in oneshot.py, which meta-GGA alpha re-introduced
# by dividing by n. Fix: clip alpha to [0, _ALPHA_MAX]. This is energy-faithful NOT
# because alpha is small (it isn't) but because (a) the DFS/SCAN enhancement gate
# has SATURATED by alpha~100 (tanh^2(log((100+1)/2)) = 0.998), and (b) the SAME
# [0, _ALPHA_MAX] clip is applied to the pretrain-data alpha (subset_selection), so
# live and precomputed alpha agree -- clipped points carry ~0 exchange-integrand
# mass, so the XC-energy change is <=1e-9 relative. See HISTORY Phase 17.
#
# The companion tail-gradient FREEZE was removed 2026-08-06 (see compute_alpha):
# it misreported the derivative of an energy ingredient and broke
# V_xc = dE_xc/dP for every meta-GGA architecture. _RHO_GRAD_CUTOFF is retained
# only as the documented threshold of the regime that motivated the freeze, and
# as the density below which alpha is grid-tail noise carrying no integrand mass.
# It no longer gates any gradient.
_ALPHA_MAX: float = 100.0
_RHO_GRAD_CUTOFF: float = 1e-6


def compute_tau_from_dm(ao_grad, dm) -> jnp.ndarray:
    """Total positive kinetic-energy density ``tau(r)`` from the live DM.

    Parameters
    ----------
    ao_grad : array, shape (3, N, nao)
        AO gradients on the grid ``[d/dx, d/dy, d/dz] chi_mu(r_g)`` -- i.e. the
        ``[1:4]`` slice of PySCF ``eval_ao(deriv=1)``. A constant (DM-independent)
        precompute, never differentiated.
    dm : array, shape (nao, nao) or (2, nao, nao)
        The live density matrix. A 2-D ``dm`` is the total (RKS) DM; a 3-D ``dm``
        is the spin-resolved UKS ``[P_alpha, P_beta]`` and is summed to the total
        ``P = P_alpha + P_beta`` (the iso-orbital tau is a total-density quantity).

    Returns
    -------
    array, shape (N,)
        ``tau(r) = 1/2 sum_d sum_{mu nu} (d_d chi_mu) P_{mu nu} (d_d chi_nu)``.
        Linear in ``dm`` (``ao_grad`` is constant) => differentiable through the SCF.
    """
    ag = jnp.asarray(ao_grad)          # (3, N, nao)
    dm = jnp.asarray(dm)
    p_total = dm if dm.ndim == 2 else dm[0] + dm[1]
    # tau(g) = 1/2 sum_d sum_ij ag[d,g,i] P_ij ag[d,g,j]
    return 0.5 * jnp.einsum("dgi,ij,dgj->g", ag, p_total, ag)


def compute_alpha(rho, sigma, tau) -> jnp.ndarray:
    """Iso-orbital indicator ``alpha = (tau - tau_W)/tau_unif`` (SCAN, Sun, Ruzsinszky,
    Perdew, PRL 115, 036402 (2015), Eq. 2; reused by DFS, PRB 104, L161109 (2021),
    Eq. 6).

    Parameters
    ----------
    rho, sigma, tau : arrays, shape (N,)
        Total density, ``|grad n|^2``, and the kinetic-energy density
        (:func:`compute_tau_from_dm`). All on the same grid.

    Returns
    -------
    array, shape (N,)
        ``alpha`` clipped to ``[0, _ALPHA_MAX]``, matching
        :func:`subset_selection.compute_descriptor_triple`. Differentiable in
        ``tau`` (hence in the DM); the clip keeps both value and reverse-mode
        gradient finite through the full differentiable SCF.

        The companion tail-gradient freeze below ``_RHO_GRAD_CUTOFF`` was
        REMOVED 2026-08-06 -- it misreported the derivative of an energy
        ingredient. See the note in the body.
    """
    rho_safe = jnp.maximum(rho, _RHO_FLOOR)
    tau_w = sigma / (8.0 * rho_safe)
    tau_unif = (3.0 / 10.0) * (3.0 * jnp.pi**2) ** (2.0 / 3.0) * rho_safe ** (5.0 / 3.0)
    # Lower clamp (>= 0) is load-bearing: it guards the network gate
    # x3 = log((alpha+1)/2) from log(neg)=NaN. Upper clip bounds the astronomical
    # tail value (no forward inf) and zeroes the gradient of the worst tail points.
    # 2026-08-06: the tail gradient freeze was REMOVED. It read
    #     jnp.where(rho < _RHO_GRAD_CUTOFF, jax.lax.stop_gradient(alpha), alpha)
    # and it made autodiff return something that is not the derivative of this
    # function wherever it was live. alpha is a descriptor feature, so that
    # inconsistency propagates straight into V_xc = dE_xc/dP: with the freeze in
    # place the meta-GGA architectures sat at 2.5e-08 on the energy/potential
    # finite-difference test against ~1e-10 for every sound architecture, and
    # removing it closes that gap. A stop_gradient inside an energy functional
    # guarantees energy/potential inconsistency by construction; the remedy for a
    # divergent tail derivative is to make the ENERGY well-behaved, never to
    # misreport its derivative.
    #
    # What the clip does and does not do, measured rather than assumed: the
    # [0, _ALPHA_MAX] clip bounds the VALUE and zeroes the derivative of points
    # pinned at the ceiling, but it does NOT bound the derivative of points below
    # it. Removing the freeze takes max|d alpha/d sigma| from 1.15e14 to 2.20e31
    # on Li at 6-311++G(3df,2pd), so the freeze was load-bearing for the
    # magnitude even though it was wrong for the derivative. The full-SCF
    # training gradient nonetheless stays finite -- guarded by
    # test_meta_gga_full_scf_gradient_finite_on_diffuse_tail, the regression test
    # for the HISTORY Phase 17 bh76:HLi failure, which reaches rho_min ~ 6e-10 at
    # the production basis. The 25-cycle pretrained-checkpoint case exceeds this
    # workstation's memory and is checked by hpcjobs/dfs6311_nan_verify.
    # If that returns non-finite, the fix is a smooth damping applied to the
    # ENERGY -- not a reinstated stop_gradient.
    return jnp.clip((tau - tau_w) / jnp.maximum(tau_unif, _RHO_FLOOR),
                    0.0, _ALPHA_MAX)
