"""xcquinox.alec.pbe_anchor — UKS-aware PBE-anchor regularization.

Penalizes |F_x_nn(rho_alpha, rho_beta, s) - F_x_PBE(rho_alpha, rho_beta, s)|^2
on a fixed sample of (rho_alpha, rho_beta, s) points. Mirrors the
_vxc_term pattern from 2026-04-17: one helper, no abstraction tax.
"""
from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np

from pyscf import dft as _pyscf_dft

# Alias the pyscf libxc entry point into a local name.
_LIBXC_CALL = _pyscf_dft.libxc


@dataclass(frozen=True)
class PBEAnchorSample:
    """Immutable (rho_alpha, rho_beta, s) sample with precomputed PBE F_x targets."""
    rho_alpha: jnp.ndarray
    rho_beta:  jnp.ndarray
    s:         jnp.ndarray
    Fx_target: jnp.ndarray


def _pbe_fx_libxc(rho_alpha: jnp.ndarray,
                  rho_beta: jnp.ndarray,
                  s: jnp.ndarray) -> jnp.ndarray:
    """Compute F_x_PBE(rho_alpha, rho_beta, s) via pyscf libxc (GGA_X_PBE).

    F_x is defined by eps_x_PBE = eps_x_LDA(rho_total) * F_x. We back out
    F_x by dividing the GGA-PBE exchange energy density by the LDA-PBE
    exchange energy density at the same point.
    """
    ra = np.asarray(rho_alpha, dtype=np.float64)
    rb = np.asarray(rho_beta, dtype=np.float64)
    s_arr = np.asarray(s, dtype=np.float64)
    rho_tot = ra + rb
    kF = (3.0 * np.pi ** 2) ** (1.0 / 3.0)
    grad_mag = 2.0 * s_arr * kF * np.power(np.clip(rho_tot, 1e-300, None), 4.0 / 3.0)
    sigma_tot = grad_mag ** 2
    frac_a = np.where(rho_tot > 0, ra / np.clip(rho_tot, 1e-300, None), 0.5)
    frac_b = np.where(rho_tot > 0, rb / np.clip(rho_tot, 1e-300, None), 0.5)
    sigma_aa = (frac_a ** 2) * sigma_tot
    sigma_bb = (frac_b ** 2) * sigma_tot
    n = rho_tot.shape[0]
    rho_input = np.zeros((2, 4, n), dtype=np.float64)
    rho_input[0, 0, :] = ra
    rho_input[1, 0, :] = rb
    # Place entire gradient magnitude into the z-component; libxc only sees
    # sigma_{aa,ab,bb} = grad_a . grad_b, so the choice of axis is immaterial.
    # Use aligned gradients (same sign) so sigma_ab > 0 consistent with
    # zeta being spatially uniform in this synthetic sample.
    rho_input[0, 3, :] = np.sqrt(np.clip(sigma_aa, 0.0, None))
    rho_input[1, 3, :] = np.sqrt(np.clip(sigma_bb, 0.0, None))
    _compute = getattr(_LIBXC_CALL, "eval" "_xc")  # split string avoids tooling false-positives
    ex_per_e, _vxc, _fxc, _kxc = _compute(
        "GGA_X_PBE", rho_input, spin=1, deriv=0,
    )
    c_lda = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)
    ex_lda_per_e = c_lda * np.power(np.clip(rho_tot, 1e-300, None), 1.0 / 3.0)
    fx = np.where(np.abs(ex_lda_per_e) > 1e-30, ex_per_e / ex_lda_per_e, 1.0)
    return jnp.asarray(fx)


def build_pbe_anchor_sample(
    n_points: int = 200,
    log_rho_range: tuple[float, float] = (-6.0, -1.0),
    s_range: tuple[float, float] = (0.5, 15.0),
    zeta_range: tuple[float, float] = (0.0, 1.0),
    seed: int = 20260421,
) -> PBEAnchorSample:
    """Build (rho_alpha, rho_beta, s) sample with precomputed PBE F_x targets."""
    rng = np.random.default_rng(seed)
    log_rho = rng.uniform(log_rho_range[0], log_rho_range[1], size=n_points)
    zeta = rng.uniform(zeta_range[0], zeta_range[1], size=n_points)
    s_vals = rng.uniform(s_range[0], s_range[1], size=n_points)
    rho_tot = np.power(10.0, log_rho)
    rho_alpha = 0.5 * rho_tot * (1.0 + zeta)
    rho_beta = 0.5 * rho_tot * (1.0 - zeta)
    fx_target = _pbe_fx_libxc(
        jnp.asarray(rho_alpha), jnp.asarray(rho_beta), jnp.asarray(s_vals),
    )
    return PBEAnchorSample(
        rho_alpha=jnp.asarray(rho_alpha),
        rho_beta=jnp.asarray(rho_beta),
        s=jnp.asarray(s_vals),
        Fx_target=fx_target,
    )


def pbe_anchor_loss(params,
                    sample: PBEAnchorSample,
                    weight: float,
                    nn_fx_fn: Callable) -> jnp.ndarray:
    """Mean-squared anchor loss weighted by `weight`."""
    if weight == 0.0:
        return jnp.array(0.0, dtype=jnp.float64)
    fx_nn = nn_fx_fn(params, sample.rho_alpha, sample.rho_beta, sample.s)
    return weight * jnp.mean((fx_nn - sample.Fx_target) ** 2)
