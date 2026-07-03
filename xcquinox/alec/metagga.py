"""Meta-GGA kinetic-energy density (tau) and SCAN iso-orbital indicator (alpha).

This is the meta-GGA (Jacob's-ladder rung 3) ingredient of the DFS functional
(Dick & Fernandez-Serra, PRB 104 L161109 (2021), Eq. 6): the iso-orbital indicator

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

import jax.numpy as jnp

# Floor for the density / uniform-gas denominators (matches
# subset_selection.compute_descriptor_triple so alpha is bit-comparable).
_RHO_FLOOR: float = 1e-30


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
    """SCAN iso-orbital indicator ``alpha = (tau - tau_W)/tau_unif`` (DFS Eq. 6).

    Parameters
    ----------
    rho, sigma, tau : arrays, shape (N,)
        Total density, ``|grad n|^2``, and the kinetic-energy density
        (:func:`compute_tau_from_dm`). All on the same grid.

    Returns
    -------
    array, shape (N,)
        ``alpha`` clamped to ``>= 0`` (grid-tail noise), matching
        :func:`subset_selection.compute_descriptor_triple`. Differentiable in
        ``tau`` (hence in the DM).
    """
    rho_safe = jnp.maximum(rho, _RHO_FLOOR)
    tau_w = sigma / (8.0 * rho_safe)
    tau_unif = (3.0 / 10.0) * (3.0 * jnp.pi**2) ** (2.0 / 3.0) * rho_safe ** (5.0 / 3.0)
    return jnp.maximum((tau - tau_w) / jnp.maximum(tau_unif, _RHO_FLOOR), 0.0)
