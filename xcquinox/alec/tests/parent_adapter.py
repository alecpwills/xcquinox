"""The parent functional wearing the model's evaluation surface.

The library's UKS energy path (``oneshot.split_exc_energy_uks``) touches a model
through exactly three names: ``eval_ex(rho, sigma, features)``,
``eval_ec(rho, sigma, features, zeta=...)`` and
``cnet.use_spin_polarization``. Substituting the parent functional for the
network turns that path into a pure quadrature of the parent, so the library's
own assembly can be compared against libxc's spin-polarized evaluation on the
same grid with no fit in the way. A discrepancy is then a defect in the
assembly.

This module lives in the test package and is deliberately NOT named ``test_*``,
so pytest imports it on demand rather than collecting it.
"""
import jax.numpy as jnp
import numpy as np
from pyscf import dft as _pyscf_dft

_LIBXC = _pyscf_dft.libxc


def gga_rho_row(rho, nabla_rho) -> np.ndarray:
    """libxc GGA input row ``(4, N)``: ``[rho, d/dx rho, d/dy rho, d/dz rho]``.

    ``nabla_rho`` is ``(N, 3)``, the layout the library stores.
    """
    r = np.asarray(rho, dtype=np.float64)
    g = np.asarray(nabla_rho, dtype=np.float64).reshape(r.shape[0], 3)
    return np.vstack([r[None, :], g.T])


def mgga_rho_row(rho, nabla_rho, tau) -> np.ndarray:
    """libxc meta-GGA input row ``(6, N)``: the GGA row, a zero Laplacian
    slot (no functional used here reads it) and the kinetic-energy density.
    """
    return np.vstack([gga_rho_row(rho, nabla_rho),
                      np.zeros((1, np.asarray(rho).shape[0])),
                      np.asarray(tau, dtype=np.float64)[None, :]])


def _row_from_sigma(rho, sigma, n_components) -> np.ndarray:
    """libxc input row encoding a KNOWN ``sigma`` rather than a real gradient.

    The gradient magnitude is placed in the x component and the other two are
    left at zero, so ``sigma_libxc = dx^2 + dy^2 + dz^2`` is the requested
    value. Only the invariant enters a GGA or meta-GGA, so this encoding is
    exact.
    """
    r = np.asarray(rho, dtype=np.float64)
    row = np.zeros((n_components, r.shape[0]), dtype=np.float64)
    row[0] = r
    row[1] = np.sqrt(np.clip(np.asarray(sigma, dtype=np.float64), 0.0, None))
    return row


def tau_from_alpha(rho, sigma, alpha) -> np.ndarray:
    """Invert the stored iso-orbital indicator to the kinetic-energy density it
    encodes: ``tau = alpha_raw tau_unif + tau_W`` with ``alpha_raw`` the exact
    inverse of the smooth positive part ``metagga.compute_alpha`` applies
    (``alpha_raw = alpha - width^2 / (4 alpha)``).

    ``alpha_raw = (tau - tau_W) / tau_unif`` with ``tau_W = sigma / (8 rho)``
    and ``tau_unif = (3/10) (3 pi^2)^{2/3} rho^{5/3}`` (Sun, Ruzsinszky and
    Perdew, Phys. Rev. Lett. 115, 036402 (2015), Eq. 2). Inverting rather than
    recontracting the density matrix keeps the descriptor's value ceiling out
    of a comparison: whatever alpha the library assembled below the ceiling,
    this is the kinetic-energy density that alpha stands for, the smoothing
    included, so a parent evaluated here reads the same tau the raw indicator
    was built from (on a one-orbital point, tau_W itself).
    """
    from xcquinox.alec.metagga import (
        _ALPHA_SMOOTHING_WIDTH, invert_smooth_positive_part)
    r = np.maximum(np.asarray(rho, dtype=np.float64), 1e-300)
    s = np.asarray(sigma, dtype=np.float64)
    a = np.asarray(invert_smooth_positive_part(
        np.asarray(alpha, dtype=np.float64), _ALPHA_SMOOTHING_WIDTH))
    tau_unif = (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0) * r ** (5.0 / 3.0)
    return a * tau_unif + s / (8.0 * r)


def _is_meta_gga(functional: str) -> bool:
    return _LIBXC.xc_type(functional) == "MGGA"


class _PolarizationFlag:
    """Stand-in for the model's cnet, carrying only the flag the energy reads."""

    def __init__(self, use_spin_polarization: bool):
        self.use_spin_polarization = bool(use_spin_polarization)


class LibxcParentModel:
    """The parent functional with the model's evaluation surface.

    ``x_functional`` / ``c_functional`` are libxc names ("GGA_X_PBE",
    "GGA_C_PBE", "MGGA_X_SCAN", "MGGA_C_SCAN", ...); ``None`` makes that
    channel return exactly zero, so exchange and correlation can be oracled
    independently.

    ``alpha_column`` (meta-GGA parents only) is the index of the iso-orbital
    indicator inside the feature block. The adapter inverts it to the
    kinetic-energy density the parent needs, so a meta-GGA parent reads
    precisely the alpha the library assembled. That is what makes the
    per-channel ingredients testable: under the exact spin scaling the alpha
    column of ``features_a`` is ``alpha(2 rho_a, 4 sigma_aa, 2 tau_a)`` and the
    reconstructed tau is ``2 tau_a``, which is the alpha channel's ingredient in
    libxc's own spin-polarized meta-GGA. The correlation term receives the
    total block, whose column is ``alpha(rho, sigma, tau)`` of the physical
    density, and inverts it to the total tau.

    Exchange is evaluated SPIN-UNPOLARIZED at the arguments it is handed. That
    is the correct surface: the caller has already applied the Oliver-Perdew
    doubling (Phys. Rev. A 20, 397 (1979)), so the adapter must not double
    again. Correlation is evaluated through libxc's spin-polarized entry point
    at ``(rho_a, rho_b) = rho (1 +- zeta) / 2`` with the two spin gradients
    taken parallel and proportional to the spin densities, so that
    ``sigma_aa + 2 sigma_ab + sigma_bb`` reproduces the requested total
    invariant exactly, and with the total kinetic-energy density split in the
    same proportion. PBE correlation is a functional of the total-density
    gradient alone, and SCAN correlation of the total gradient, the total tau
    and zeta (the per-spin quantities enter libxc's SCAN correlation only
    through the Fermi-hole bound ``sigma_ss <= 8 rho_s tau_s``, which the
    proportional split satisfies exactly when ``sigma <= 8 rho tau``, i.e.
    alpha >= 0), so the choice is exact rather than approximate.
    """

    def __init__(self, x_functional: str | None = None,
                 c_functional: str | None = None,
                 alpha_column: int | None = None,
                 use_spin_polarization: bool = True,
                 descriptors: tuple = ()):
        self.x_functional = x_functional
        self.c_functional = c_functional
        self.alpha_column = alpha_column
        self.cnet = _PolarizationFlag(use_spin_polarization)
        self.descriptors = tuple(descriptors)
        for functional in (x_functional, c_functional):
            if functional is not None and _is_meta_gga(functional) \
                    and alpha_column is None:
                raise ValueError(
                    f"{functional} is a meta-GGA and needs alpha_column, the "
                    "index of the iso-orbital indicator in the feature block")

    def _tau(self, rho, sigma, features):
        alpha = np.asarray(features, dtype=np.float64)[:, self.alpha_column]
        return tau_from_alpha(rho, sigma, alpha)

    def eval_ex(self, rho, sigma, features):
        """``rho * eps_x^parent`` at the arguments handed in, spin-unpolarized."""
        r = np.asarray(rho, dtype=np.float64)
        if self.x_functional is None:
            return jnp.zeros(r.shape[0])
        positive = r > 0.0
        r_safe = np.where(positive, r, 1.0)
        s = np.asarray(sigma, dtype=np.float64)
        if _is_meta_gga(self.x_functional):
            row = _row_from_sigma(r_safe, s, 6)
            row[5] = self._tau(r_safe, s, features)
        else:
            row = _row_from_sigma(r_safe, s, 4)
        eps = np.asarray(
            _LIBXC.eval_xc(self.x_functional, row, spin=0, deriv=0)[0])
        # A nonpositive density carries no exchange; masking here matches libxc's
        # own clamp on an empty spin channel and keeps a quadrature-noise
        # negative from entering the integrand.
        return jnp.asarray(np.where(positive, r * eps, 0.0))

    def eval_ec(self, rho, sigma, features, zeta=0.0):
        """``rho * eps_c^parent(rho, sigma[, tau], zeta)``, spin-polarized in
        zeta."""
        r = np.asarray(rho, dtype=np.float64)
        if self.c_functional is None:
            return jnp.zeros(r.shape[0])
        positive = r > 0.0
        z = np.asarray(zeta, dtype=np.float64) * np.ones_like(r)
        r_safe = np.where(positive, r, 1.0)
        s = np.asarray(sigma, dtype=np.float64)
        share_a = 0.5 * (1.0 + z)
        share_b = 0.5 * (1.0 - z)
        n_components = 6 if _is_meta_gga(self.c_functional) else 4
        row_a = np.zeros((n_components, r.shape[0]), dtype=np.float64)
        row_b = np.zeros((n_components, r.shape[0]), dtype=np.float64)
        row_a[0] = r_safe * share_a
        row_b[0] = r_safe * share_b
        g = np.sqrt(np.clip(s, 0.0, None))
        row_a[1] = g * share_a
        row_b[1] = g * share_b
        if n_components == 6:
            tau = self._tau(r_safe, s, features)
            row_a[5] = tau * share_a
            row_b[5] = tau * share_b
        eps = np.asarray(_LIBXC.eval_xc(
            self.c_functional, np.stack([row_a, row_b]), spin=1, deriv=0)[0])
        return jnp.asarray(np.where(positive, r * eps, 0.0))
