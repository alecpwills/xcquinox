"""Rung-3.5 localized density-matrix descriptor machinery.

Implements the bounded local *occupancy* of Janesko's unified Rung-3.5 / DFT+U
formalism (arXiv:2206.07118, Eq. 12-13; M11plus, Verma et al. *J. Chem. Theory
Comput.* 15, 4804 (2019)):

    n_sigma(r_m) = sum_i |<psi_i,sigma | phi^G_{r_m}>|^2
                 = A(r_m)^T P^sigma A(r_m)          in [0, 1]

where ``phi^G_{r_m}(r) = (2 alpha / pi)^{3/4} exp(-alpha |r - r_m|^2)`` is an
L2-normalized Gaussian projector centered at the grid point ``r_m`` and

    A_mu(r_m) = <chi_mu | phi^G_{r_m}>   (the "projected-AO" overlap vector).

This contracts the *non-local* Kohn-Sham one-particle density matrix
``gamma_sigma(r, r') = sum_mn chi_mu(r) P^sigma_mn chi_nu(r')`` once (linearly)
against the model projector -- a genuine Rung-3.5 ingredient (its own ladder
rung between meta-GGA and hybrid), NOT reducible to tau, NOT a static reference
DM, and per grid point so it is leak-free (size-intensive).

Two properties drive the implementation:

* ``A_mu(r_m)`` depends only on the basis, the grid, and the *fixed* width
  ``alpha`` -- NOT on the density matrix or the density. So it is a precomputed
  CONSTANT (a plain PySCF overlap integral) and is never differentiated. The
  occupancy ``n_sigma = A^T P A`` is then a trivial einsum, linear and
  differentiable in the *live* DM ``P^sigma`` (the self-consistent SCF property).
* ``n_sigma`` is bounded ``[0, 1]`` by Bessel's inequality (``P^sigma`` is PSD
  => ``>= 0``; the occupied KS orbitals are L2-orthonormal and ``phi^G`` is
  normalized => ``<= 1``), so it is NaN-safe by construction.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# Default projector width (a0^-2). Grounded at the M11plus rung-3.5 kernel scale
# d^2 = 5 a0^2 (Verma et al. 2019); a configurable hyperparameter of the descriptor.
DEFAULT_RUNG35_ALPHA: float = 0.2


def compute_projected_ao(mol, coords, alpha: float = DEFAULT_RUNG35_ALPHA) -> np.ndarray:
    """Projected-AO overlap ``A_mu(r_m) = <chi_mu | phi^G_{r_m}>``.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        The molecule whose AO basis ``{chi_mu}`` is projected.
    coords : array_like, shape (N, 3)
        Grid points ``r_m`` (atomic units / Bohr), e.g. ``mf.grids.coords``.
    alpha : float
        Width of the L2-normalized Gaussian projector
        ``phi^G = (2 alpha/pi)^{3/4} exp(-alpha |r - r_m|^2)``.

    Returns
    -------
    np.ndarray, shape (N, nao)
        ``A`` -- a constant (DM/density-independent) precompute.

    Notes
    -----
    Implemented with PySCF's ``fakemol_for_charges`` (s-Gaussians at the grid
    points) and ``intor_cross('int1e_ovlp', ...)``. Each projector is rescaled by
    its own L2 norm ``||phi_fake||`` (identical for every center since they share
    the exponent ``alpha``), so the result is the overlap against the
    L2-NORMALIZED projector regardless of the helper's internal normalization.
    """
    from pyscf import gto
    try:
        from pyscf.gto.mole import fakemol_for_charges
    except ImportError:  # older/newer layout
        from pyscf.df.incore import fakemol_for_charges

    coords = np.ascontiguousarray(np.asarray(coords, dtype=float).reshape(-1, 3))
    n_pts = coords.shape[0]
    nao = mol.nao_nr()
    a = float(alpha)
    if not (a > 0.0):
        raise ValueError(f"rung-3.5 projector width alpha must be > 0; got {alpha!r}")

    # ||phi_fake|| for a single projector (translation-invariant: same exponent
    # => same norm for every grid center), so one 1x1 self-overlap suffices.
    fm0 = fakemol_for_charges(coords[:1], expnt=a)
    norm_fake = float(np.sqrt(np.asarray(fm0.intor("int1e_ovlp"))[0, 0]))

    out = np.empty((n_pts, nao), dtype=float)
    batch = 4000
    for i in range(0, n_pts, batch):
        pts = coords[i:i + batch]
        fm = fakemol_for_charges(pts, expnt=a)
        cross = np.asarray(gto.intor_cross("int1e_ovlp", mol, fm))  # (nao, nbatch)
        out[i:i + pts.shape[0]] = (cross / norm_fake).T
    return out


def compute_rung35_occupancy(proj_ao: jnp.ndarray, dm: jnp.ndarray) -> jnp.ndarray:
    """Per-spin local occupancy ``n_sigma(r) = A(r)^T P^sigma A(r)``.

    Parameters
    ----------
    proj_ao : array, shape (N, nao)
        The constant projected-AO matrix ``A`` from :func:`compute_projected_ao`.
    dm : array, shape (nao, nao) or (2, nao, nao)
        The (live) density matrix. A 2-D ``dm`` is treated as the RKS *total*
        DM and split evenly into the two spin channels (``P^sigma = dm / 2``);
        a 3-D ``dm`` is the spin-resolved UKS ``[P^alpha, P^beta]``.

    Returns
    -------
    array, shape (N, 2)
        ``[n_alpha(r), n_beta(r)]`` per grid point, each in ``[0, 1]``. Linear in
        ``dm`` (``A`` is constant) and therefore differentiable through the SCF.
    """
    A = jnp.asarray(proj_ao)
    dm = jnp.asarray(dm)
    if dm.ndim == 2:
        half = 0.5 * dm
        dm_spin = jnp.stack([half, half], axis=0)        # (2, nao, nao)
    else:
        dm_spin = dm                                     # (2, nao, nao)
    # n_sigma(r) = sum_{mn} A_{r,m} P^sigma_{mn} A_{r,n}
    return jnp.einsum("gm,smn,gn->gs", A, dm_spin, A)    # (N, 2)
