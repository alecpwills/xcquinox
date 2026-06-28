"""Phase-1 tests for the rung-3.5 projected-AO occupancy machinery.

The descriptor is the bounded local occupancy
    n_sigma(r) = A(r)^T P^sigma A(r)  in [0, 1]
(Janesko, arXiv:2206.07118 Eq. 12-13; M11plus, Verma et al. JCTC 15, 4804 (2019)),
where
    A_mu(r) = <chi_mu | phi^G_r>,   phi^G = (2 alpha/pi)^{3/4} exp(-alpha |r - r_m|^2)
is the overlap of basis function chi_mu with an L2-normalized Gaussian projector at
the grid point r_m.

Key property exploited throughout: A_mu(r) depends only on the basis, the grid, and
alpha -- NOT on the density matrix or the density -- so it is a precomputed CONSTANT
(a plain PySCF overlap), never differentiated. The occupancy n_sigma = A^T P A is then
a trivial einsum, linear and differentiable in the live DM, and bounded [0, 1] by
Bessel's inequality (P^sigma is PSD => >= 0; {psi_i} L2-orthonormal + ||phi^G||=1 => <= 1).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


ALPHA = 0.2  # projector width (a0^-2), grounded at the M11plus kernel scale d^2=5 a0^2


def _h2():
    """Small real closed-shell molecule: H2 / def2-svp PBE."""
    from pyscf import dft, gto
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="def2-svp", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.kernel()
    return (mol, np.asarray(mf.make_rdm1()),
            np.asarray(mf.grids.coords), np.asarray(mf.grids.weights))


def test_projected_ao_analytic_s_s_overlap():
    """Exact closed-form check pinning the normalization, independent of the
    intor_cross implementation. For a single s-primitive chi (exponent beta) and
    the normalized s-Gaussian projector phi^G (exponent alpha) at r_m:
        A = N_beta N_alpha (pi/(alpha+beta))^{3/2} exp(-alpha beta/(alpha+beta) |R-r_m|^2),
        N_x = (2x/pi)^{3/4}.
    """
    from pyscf import gto
    from xcquinox.alec.rung35 import compute_projected_ao
    beta = 0.8
    mol = gto.M(atom="H 0 0 0", basis={"H": [[0, [beta, 1.0]]]}, spin=1, verbose=0)
    rm = np.array([[0.5, 0.1, -0.2]])
    A = float(np.asarray(compute_projected_ao(mol, rm, ALPHA))[0, 0])
    R2 = float(np.sum(rm[0] ** 2))
    Nb = (2 * beta / np.pi) ** 0.75
    Na = (2 * ALPHA / np.pi) ** 0.75
    ref = Nb * Na * (np.pi / (ALPHA + beta)) ** 1.5 * \
        np.exp(-ALPHA * beta / (ALPHA + beta) * R2)
    np.testing.assert_allclose(A, ref, rtol=1e-9, atol=1e-12)


def test_projected_ao_matches_numerical_quadrature():
    """A_mu(r_m) = integral chi_mu(r) phi^G(r-r_m) dr matches a direct grid
    quadrature sum_g w_g chi_mu(r_g) phi^G(r_g - r_m) -- an oracle independent of
    the intor_cross implementation, valid for general angular momentum."""
    from xcquinox.alec.rung35 import compute_projected_ao
    mol, _dm, coords, weights = _h2()
    test_pts = coords[:: max(1, len(coords) // 6)][:5]
    A = np.asarray(compute_projected_ao(mol, test_pts, ALPHA))
    ao = mol.eval_gto("GTOval", coords)               # (Ngrid, nao)
    norm = (2 * ALPHA / np.pi) ** 0.75
    for p, rm in enumerate(test_pts):
        g = norm * np.exp(-ALPHA * np.sum((coords - rm) ** 2, axis=1))
        ref = np.einsum("g,gm->m", weights * g, ao)
        np.testing.assert_allclose(A[p], ref, rtol=2e-2, atol=2e-3,
                                   err_msg=f"projected-AO row {p} vs quadrature")


def test_occupancy_bounded_0_1_real_dm():
    """n_sigma(r) = A(r)^T P^sigma A(r) in [0, 1] for the real PBE
    single-determinant DM (PSD P^sigma => >= 0; Bessel + normalized projector => <= 1)."""
    from xcquinox.alec.rung35 import (compute_projected_ao,
                                       compute_rung35_occupancy)
    mol, dm, coords, _w = _h2()
    A = compute_projected_ao(mol, coords, ALPHA)
    n = np.asarray(compute_rung35_occupancy(jnp.asarray(A), jnp.asarray(dm)))
    assert n.shape == (len(coords), 2), n.shape
    assert np.all(np.isfinite(n)), "occupancy has non-finite entries"
    assert n.min() >= -1e-9, f"occupancy < 0: min={n.min()}"
    assert n.max() <= 1.0 + 1e-6, f"occupancy > 1: max={n.max()}"


def test_occupancy_linear_and_differentiable_in_dm():
    """n_sigma is linear in P^sigma (A constant) => finite gradient wrt the live
    DM -- the property the self-consistent SCF relies on."""
    from xcquinox.alec.rung35 import compute_rung35_occupancy
    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.standard_normal((20, 6)))
    f = lambda d: jnp.sum(compute_rung35_occupancy(A, d + d.T))
    dm = jnp.asarray(rng.standard_normal((6, 6)))
    grad = jax.grad(f)(dm)
    assert np.all(np.isfinite(np.asarray(grad))), "non-finite gradient wrt DM"
