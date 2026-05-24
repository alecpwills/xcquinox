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


def _fx_pbe_analytic(s: np.ndarray) -> np.ndarray:
    """PBE F_x(s) closed form — Perdew-Burke-Ernzerhof 1996 §3 eq. (14):

        F_x(s) = 1 + kappa - kappa / (1 + mu * s^2 / kappa)

    with kappa = 0.804 (Lieb-Oxford) and mu = 0.21951 (gradient
    expansion). Used as the rho->0 / spin-polarized-boundary limit
    where the libxc rho/sigma division is 0/0; the analytic formula is
    rho-independent at fixed s, making it the correct limit (in
    contrast to the pre-fix fallback F_x=1, which biased the anchor
    target toward UEG).
    """
    kappa = 0.804
    mu = 0.21951
    return 1.0 + kappa - kappa / (1.0 + mu * s ** 2 / kappa)


def _pbe_fx_libxc(rho_alpha: jnp.ndarray,
                  rho_beta: jnp.ndarray,
                  s: jnp.ndarray) -> jnp.ndarray:
    """Compute F_x_PBE(rho_alpha, rho_beta, s) via the spin-scaling approximation.

    F_x_SS(ra, rb, s) = 0.5 * (F_x_RKS(2*ra, sigma_aa_eff) + F_x_RKS(2*rb, sigma_bb_eff))

    where sigma_sigma_eff = (1 +/- zeta)**2 * sigma_tot, zeta = (ra-rb)/(ra+rb),
    and sigma_tot = (2*kF(rho_tot)*s*rho_tot**(4/3))**2. This matches the
    NN's exchange functional form at UKS SCF time (see
    ``xcquinox.alec.oneshot._uks_spin_resolved_vxc``) and the anchor
    helper ``_nn_fx_local_uks``.

    The implementation note that ``zeta`` is treated as spatially
    constant (per-sample): each (rho_alpha, rho_beta, s) row is scalar,
    so this assumption is automatic for the anchor sample. It would
    NOT be correct for a spatially-varying grid where the rows live on
    different spatial regions.

    F_x_RKS is computed by calling libxc GGA_X_PBE in spin=0 (RKS) mode
    on each spin channel's doubled density, dividing the GGA exchange
    energy per electron by the LDA exchange energy per electron at the
    same rho_tot=2*rho_sigma. At rho_sigma -> 0 (spin-polarized
    boundary or rho_tot -> 0), the libxc / LDA ratio is 0/0; we fall
    back to the analytic PBE F_x(s) formula since F_x is rho-
    independent at fixed s.
    """
    ra = np.asarray(rho_alpha, dtype=np.float64)
    rb = np.asarray(rho_beta, dtype=np.float64)
    s_arr = np.asarray(s, dtype=np.float64)
    rho_tot = ra + rb
    kF_tot = (3.0 * np.pi ** 2) ** (1.0 / 3.0)
    sigma_tot = (
        2.0 * s_arr * kF_tot
        * np.power(np.clip(rho_tot, 1e-300, None), 4.0 / 3.0)
    ) ** 2
    zeta = np.where(
        rho_tot > 0,
        (ra - rb) / np.clip(rho_tot, 1e-300, None),
        0.0,
    )
    sigma_aa_eff = (1.0 + zeta) ** 2 * sigma_tot
    sigma_bb_eff = (1.0 - zeta) ** 2 * sigma_tot

    _compute = _LIBXC_CALL.eval_xc
    c_lda = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)
    fx_pbe_at_s = _fx_pbe_analytic(s_arr)

    def _fx_rks(rho_spin_doubled: np.ndarray,
                sigma_spin_eff: np.ndarray,
                fallback_fx: np.ndarray) -> np.ndarray:
        # libxc GGA spin=0 input is shape (n_components, n_grid). Components
        # are (rho, drho_x, drho_y, drho_z, [lapl, tau]); we only need the
        # first 4 for GGA. To pass a known sigma value, we set
        # drho_x = sqrt(sigma) and drho_y = drho_z = 0, so that
        # sigma_libxc = drho_x^2 + drho_y^2 + drho_z^2 = sigma. Slot 1 is
        # less ambiguous than slot 3 (gradient_z) for this magnitude-only
        # encoding.
        n = rho_spin_doubled.shape[0]
        rho_input = np.zeros((4, n), dtype=np.float64)
        rho_input[0, :] = rho_spin_doubled
        rho_input[1, :] = np.sqrt(np.clip(sigma_spin_eff, 0.0, None))
        ex_per_e, *_ = _compute("GGA_X_PBE", rho_input, spin=0, deriv=0)
        ex_lda_per_e = c_lda * np.power(
            np.clip(rho_spin_doubled, 1e-300, None), 1.0 / 3.0,
        )
        # rho -> 0: 0/0 limit becomes the rho-independent F_x_PBE(s) (NOT
        # the pre-fix fallback F_x = 1, which biased the target toward UEG
        # at every spin-polarized boundary).
        return np.where(
            np.abs(ex_lda_per_e) > 1e-30,
            ex_per_e / ex_lda_per_e,
            fallback_fx,
        )

    fx_a = _fx_rks(2.0 * ra, sigma_aa_eff, fx_pbe_at_s)
    fx_b = _fx_rks(2.0 * rb, sigma_bb_eff, fx_pbe_at_s)
    return jnp.asarray(0.5 * (fx_a + fx_b))


def build_pbe_anchor_sample(
    n_points: int = 200,
    log_rho_range: tuple[float, float] = (-6.0, -1.0),
    s_range: tuple[float, float] = (0.5, 15.0),
    zeta_range: tuple[float, float] = (0.0, 1.0),
    seed: int = 20260421,
) -> PBEAnchorSample:
    """Build (rho_alpha, rho_beta, s) sample with precomputed PBE F_x targets."""
    if not (-1.0 <= zeta_range[0] <= zeta_range[1] <= 1.0):
        raise ValueError(
            f"zeta_range must lie in [-1, 1] with low <= high; got {zeta_range!r}"
        )
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
    """Mean-squared anchor loss weighted by `weight`.

    Short-circuits to 0.0 when `weight == 0.0`. Otherwise returns:
        weight * mean((F_x_nn - sample.Fx_target) ** 2)

    The caller's `nn_fx_fn(params, rho_alpha, rho_beta, s) -> (N,) array`
    must return NN-predicted F_x at the N sample points, using the same
    UKS forward-pass convention the loss class uses (spin-scaling:
    F_x_UKS(rho/2, rho/2, s) = F_x_RKS(rho, s)).
    """
    if weight == 0.0:
        # R2-A N6 audit fix: match the dtype of the sample arrays
        # rather than hardcoding float64 (caller may have x32-only JAX
        # configured, in which case the hardcoded f64 zero gets cast
        # back to f32 and the cast itself shows up as a small
        # discontinuity at the weight=0 boundary).
        return jnp.zeros((), dtype=sample.Fx_target.dtype)
    fx_nn = nn_fx_fn(params, sample.rho_alpha, sample.rho_beta, sample.s)
    return weight * jnp.mean((fx_nn - sample.Fx_target) ** 2)
