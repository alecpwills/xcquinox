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
single orbital (tau = tau_W). The lower bound alpha >= 0 (the von Weizsacker
inequality tau >= tau_W, which holds pointwise for every positive semidefinite
density matrix) is imposed by a smooth positive part rather than a hard clip, so
that the descriptor and its derivative are continuous where a channel's density is
one orbital (see :func:`compute_alpha`); the numpy twin
:func:`xcquinox.alec.subset_selection.compute_descriptor_triple` keeps the hard
clip and agrees with this function away from alpha = 0.
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
# by dividing by n. Fix: bound alpha above by _ALPHA_MAX. This is energy-faithful
# NOT because alpha is small (it isn't) but because (a) the DFS/SCAN enhancement
# gate has SATURATED by alpha~100 (tanh^2(log((100+1)/2)) = 0.998), and (b) the
# SAME ceiling is applied to the pretrain-data alpha (subset_selection), so live
# and precomputed alpha agree -- clipped points carry ~0 exchange-integrand mass,
# so the XC-energy change is <=1e-9 relative. See HISTORY Phase 17.
#
# The companion tail-gradient FREEZE was removed 2026-08-06 (see compute_alpha):
# it misreported the derivative of an energy ingredient and broke
# V_xc = dE_xc/dP for every meta-GGA architecture. _RHO_GRAD_CUTOFF is retained
# only as the documented threshold of the regime that motivated the freeze, and
# as the density below which alpha is grid-tail noise carrying no integrand mass.
# It no longer gates any gradient.
_ALPHA_MAX: float = 100.0
_RHO_GRAD_CUTOFF: float = 1e-6

# Width of the smooth positive part that imposes the lower bound alpha >= 0, in
# units of the raw indicator (equivalently, in kinetic-energy-density units the
# width is _ALPHA_SMOOTHING_WIDTH * tau_unif(n), so the construction is invariant
# under the uniform density scaling n(r) -> lambda^3 n(lambda r) that alpha
# itself is invariant under). Anchors, all measured (see compute_alpha):
#   (a) on a one-orbital spin channel (H alpha, Li beta) tau = tau_W identically
#       and the raw indicator is the rounding residue of tau - tau_W divided by
#       tau_unif; on every grid point with 2 rho_sigma > 1e-8 (all but ~1e-6 of
#       the channel's electron) that residue is at most 3.1e-12 (H) and 9.7e-11
#       (Li) at def2-svp / grid 1, 6.2e-12 and 1.7e-10 at def2-tzvp / grid 2,
#       7.5e-12 and 6.6e-10 at 6-311++G(3df,2pd) / grid 3, and the response of
#       the raw indicator to a 1e-14 relative change of the density matrix is
#       below 1e-8 there. 1e-5 exceeds both by >= 1e3, so the derivative of the
#       smoothed indicator is a deterministic function of the density on that
#       region. Between the network's tail mask (2 rho_sigma = 1e-10) and 1e-8
#       the residue reaches 1.6e-7 (def2-svp), 1.8e-6 (def2-tzvp) and 3.7e-8
#       (production), and below the mask 5.5e-2 (Li, production, rho ~ 1e-12);
#       no width could dominate the deep tail, and none needs to: those points
#       carry no integrand mass and the energy does not read them.
#   (b) the SCAN exchange energy of the H atom -- the system whose density is one
#       orbital everywhere, so the smoothed indicator sits at its floor
#       _ALPHA_SMOOTHING_WIDTH / 2 on every point -- evaluated through the
#       library's own path (parent adapter, libxc MGGA_X_SCAN at the kinetic-
#       energy density the column encodes) moves by +1.17e-7 Ha against libxc at
#       the true tau, identically at def2-svp, def2-tzvp and the production
#       identity, and Li's beta channel by +3.1e-7 Ha; the shift is linear in the
#       width (0.0117 Ha per unit width on H), so the 1e-12 Ha level would need a
#       width of 1e-10, below the rounding residue it has to dominate.
#   (c) the pretraining-fidelity certificate's atomic tolerance is 1.0 mHa
#       (cluster/fidelity.py); the shift in (b) is 8.5e3 below it.
_ALPHA_SMOOTHING_WIDTH: float = 1e-5

#: Identity of the indicator's definition, recorded in every pretraining-data
#: manifest and compared by ``pretrain_data_gen.pretrain_data_is_current``: a
#: file whose alpha rows were written under another definition (the hard
#: clip, or another width) is stale for a run at this one, exactly as a file
#: built at another basis or orientation lock is. Built from the width so the
#: two cannot drift apart.
ALPHA_DEFINITION: str = f"smooth_positive_part:width={_ALPHA_SMOOTHING_WIDTH:.0e}"


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


def smooth_positive_part(x, width):
    """``(x + sqrt(x^2 + width^2)) / 2``: a C-infinity positive part.

    Equals ``max(x, 0)`` up to ``width^2 / (4 |x|)`` for ``|x| >> width``, is
    ``width / 2`` at ``x = 0`` with slope ``1/2`` there, is strictly positive
    everywhere, and obeys ``p(x) - p(-x) = x`` exactly, so a central difference
    across ``x = 0`` reproduces its derivative. Homogeneous of degree one in
    ``(x, width)`` together, which is what makes the indicator's smoothing
    scale-invariant when the width is a multiple of ``tau_unif``.
    """
    return 0.5 * (x + jnp.sqrt(x * x + width * width))


def invert_smooth_positive_part(p, width):
    """The ``x`` with ``smooth_positive_part(x, width) == p`` (``p > 0``):
    ``x = p - width^2 / (4 p)``. Used where a stored indicator column has to be
    read back as the raw ``(tau - tau_W) / tau_unif`` it encodes."""
    p = jnp.asarray(p)
    return p - (width * width) / (4.0 * p)


def compute_alpha(rho, sigma, tau) -> jnp.ndarray:
    """Iso-orbital indicator ``alpha = (tau - tau_W)/tau_unif`` (SCAN, Sun, Ruzsinszky,
    Perdew, PRL 115, 036402 (2015), Eq. 2; reused by DFS, PRB 104, L161109 (2021),
    Eq. 6), with its lower bound imposed by a smooth positive part.

    Parameters
    ----------
    rho, sigma, tau : arrays, shape (N,)
        Total density, ``|grad n|^2``, and the kinetic-energy density
        (:func:`compute_tau_from_dm`). All on the same grid.

    Returns
    -------
    array, shape (N,)
        ``min(p(alpha_raw), _ALPHA_MAX)`` with ``p`` the smooth positive part of
        width ``_ALPHA_SMOOTHING_WIDTH`` (:func:`smooth_positive_part`), i.e. in
        kinetic-energy-density units ``p_delta(tau - tau_W) / tau_unif`` with
        ``delta = _ALPHA_SMOOTHING_WIDTH * tau_unif``. Differentiable in ``tau``
        (hence in the density matrix) with a derivative that is continuous
        everywhere below the ceiling; the ceiling bounds the astronomical tail
        value and zeroes the derivative of the points pinned at it.

        The bound alpha >= 0 is the von Weizsacker inequality, exact on every
        positive semidefinite density matrix; a raw value below zero is
        rounding (or a probe that has left the physical domain). A hard clip
        ``max(alpha_raw, 0)`` made the derivative one-sided there, and on a
        one-orbital spin channel (H alpha, Li beta; the H atom's total density)
        -- where tau = tau_W identically and the raw indicator is the rounding
        residue of that cancellation -- autodiff returned whichever side the
        rounding selected: the beta-channel feature-response term of Li's Fock
        matrix moved by 0.93 Ha under a 1e-14 relative change of the density
        matrix, and H's by 4.2e-3 Ha (deep_mgga_3x16, def2-svp, grid 1). With
        the smooth positive part the same probe moves H's by 3.6e-12 Ha and the
        H atom's Fock pair reproduces a central difference of the energy along
        a random symmetric direction to 6.2e-10 relative (h = 1e-7); the
        manual UKS loop's occupancy-keyed gate on the indicator response, which
        the clip had made necessary, is retired. What the smoothing does NOT
        change is the indicator's response amplification in the density tail,
        d alpha / dP ~ n^{-5/3}: on Li's beta channel at def2-svp the outermost
        radial shell (rho_sigma = 1.0e-9, 898 points) carries a response of
        2.9e-3 Ha per point in one Fock element with d alpha_raw / dP ~ 4e10
        there, so a 1e-14 change of the density matrix still moves that
        channel's virtual-virtual Fock block by 0.4 Ha through the smoothed
        derivative (a continuous function with a Lipschitz constant of that
        size), and a finite-difference probe along a direction that reaches
        those points with a usable step is not a derivative estimate. The
        response annihilates the occupied orbital of a one-orbital channel
        exactly (alpha_raw is stationary along every rank-preserving
        rotation), so the SCF fixed point is unaffected; see
        DEFERRED_WORK.md entry 27 (closure) and entry 30.

        Anchor of the width, all measured: (a) the rounding residue of the raw
        indicator on a one-orbital channel is at most 6.6e-10 on every grid
        point with 2 rho_sigma > 1e-8 (H and Li; def2-svp / grid 1, def2-tzvp
        / grid 2 and 6-311++G(3df,2pd) / grid 3), and its response to a 1e-14
        relative change of the density matrix is below 1e-8 there, so 1e-5
        exceeds both by three orders; (b) the largest change of E_x^SCAN the
        smoothing induces through the library's own path against libxc at the
        true kinetic-energy density is +1.17e-7 Ha (the H atom, one orbital
        everywhere; identical at the three identities) and +3.1e-7 Ha on Li's
        beta channel, linear in the width; (c) the certificate tolerance is
        1.0 mHa per atom, 8.5e3 above (b). See the comment at
        ``_ALPHA_SMOOTHING_WIDTH``.

        The companion tail-gradient freeze below ``_RHO_GRAD_CUTOFF`` was
        REMOVED 2026-08-06 -- it misreported the derivative of an energy
        ingredient. See the note in the body.
    """
    rho_safe = jnp.maximum(rho, _RHO_FLOOR)
    tau_w = sigma / (8.0 * rho_safe)
    tau_unif = (3.0 / 10.0) * (3.0 * jnp.pi**2) ** (2.0 / 3.0) * rho_safe ** (5.0 / 3.0)
    # The lower bound is a smooth positive part (see the docstring); it also
    # guards the network gate x3 = log((alpha+1)/2) from log(neg)=NaN, being
    # strictly positive. The ceiling bounds the astronomical tail value (no
    # forward inf) and zeroes the gradient of the worst tail points.
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
    # What the ceiling does and does not do, measured rather than assumed: it
    # bounds the VALUE and zeroes the derivative of points pinned at it, but it
    # does NOT bound the derivative of points below it. Removing the freeze
    # took max|d alpha/d sigma| from 1.15e14 to 2.20e31 on Li at
    # 6-311++G(3df,2pd), so the freeze was load-bearing for the magnitude even
    # though it was wrong for the derivative. The full-SCF training gradient
    # nonetheless stays finite -- guarded by
    # test_meta_gga_full_scf_gradient_finite_on_diffuse_tail, the regression
    # test for the HISTORY Phase 17 bh76:HLi failure, which reaches
    # rho_min ~ 6e-10 at the production basis. The 25-cycle
    # pretrained-checkpoint case exceeds this workstation's memory and is
    # checked by hpcjobs/dfs6311_nan_verify. If that returns non-finite, the
    # fix is a smooth damping applied to the ENERGY -- not a reinstated
    # stop_gradient.
    alpha_raw = (tau - tau_w) / jnp.maximum(tau_unif, _RHO_FLOOR)
    return jnp.minimum(smooth_positive_part(alpha_raw, _ALPHA_SMOOTHING_WIDTH),
                       _ALPHA_MAX)
