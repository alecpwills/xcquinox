"""xcquinox.alec.parents: parent enhancement factors in JAX.

The functions here evaluate the parent functional -- PBE for the GGA rungs,
SCAN for the meta-GGA rungs -- as the enhancement factors the networks in
``networks.py`` return, on the networks' own conventions, so that an anchored
network (``AlecGGA_XNet.parent`` / ``AlecGGA_CNet.parent``) reproduces its
parent at initialization pointwise on every row it integrates
(``SPEC_parent_anchor.md`` Section 3.1). Pure JAX, float64, per-point scalar
functions of the row's physical inputs; every constant is the value libxc
carries, since libxc through ``pyscf.dft.libxc.eval_xc`` is the oracle the
certificate's parent energies come from.

Conventions
-----------
Exchange is posed per spin channel on the DOUBLED density (Oliver and Perdew,
Phys. Rev. A 20, 397 (1979)): the exchange row carries ``(2 rho_sigma,
4 sigma_sigma_sigma)`` and ``pbe_fx(rho, sigma)`` is the spin-unpolarized PBE
enhancement factor at those inputs, so that ``rho eps_x^LDA(rho) pbe_fx`` on
the doubled channel is libxc's spin-resolved exchange energy density,
``E_x[n_a, n_b] = (E_x[2 n_a] + E_x[2 n_b]) / 2``. Correlation is posed on the
total density with the row's spin polarization ``zeta`` and is defined
RELATIVE TO THE MODEL'S OWN BASELINE: the model's correlation energy density
is ``rho eps_c^base F_c`` with ``eps_c^base = pw92c_polarized_scalar`` (the
zeta-dependent PW92 of ``models.AlecGGAModel._ec_baseline``), so
``pbe_fc = eps_c^PBE / eps_c^base`` and ``rho eps_c^base pbe_fc`` is libxc's
PBE correlation energy density exactly. This is the convention the
pretraining data stores its ``Fc`` targets in (``pretrain_data_gen``: the
open-shell ratio is formed against ``LDA_C_PW`` at ``spin=1``).

Two PW92 parameter sets appear, deliberately. libxc's ``GGA_C_PBE`` is built
on ``LDA_C_PW_MOD`` -- the PW92 form with the parameters given to more
figures (A = 0.0310907, 0.01554535, 0.0168869) and the exact
``f''(0) = 8 / (9 (2^(4/3) - 2))`` -- and agrees with it to 2.2e-16 at
``sigma = 0``; the repository's ``pw92c_polarized_scalar`` carries the
original Table I values of Perdew and Wang, Phys. Rev. B 45, 13244 (1992)
(A = 0.031091, 0.015545, 0.016887), agrees with libxc's ``LDA_C_PW`` to 6.5e-9
and differs from ``LDA_C_PW_MOD`` by 4.4e-6 relative. The numerator of
``pbe_fc`` therefore evaluates PBE on the modified set (``_pw92_mod_eps``),
which is what libxc returns, and the denominator is the model's baseline;
the two are not interchangeable, and ``pbe_fc(s -> 0)`` is
``1 + O(4e-6)`` -- the ratio of the two parameter sets -- rather than 1.

At full polarization libxc's regularization is reproduced, since libxc is
the oracle: libxc floors each spin density at its density threshold
(``LIBXC_DENS_THRESHOLD = 1e-12`` for ``GGA_C_PBE``; measured, the empty
channel's ``eps_c`` is constant below it and equals the analytic value at the
floored channel to 1.5e-15), so ``pbe_fc`` evaluates the parent at the
floored spin densities and reports the derivative libxc reports, the one
taken at the floored point. Without it the analytic parent at ``zeta = 1``
sits 1.2e-5 relative off libxc in ``F_c`` on the H atom's rows and 1.3e-9 Ha
in its ``E_c``. ``1 -+ zeta`` is floored at ``ZETA_FLOOR``, libxc's
``zeta_threshold`` (DBL_EPSILON), in ``phi`` and in the PW92 spin
interpolation. The production UKS path never reaches full polarization:
``oneshot.uks_zeta`` holds zeta inside ``+-(1 - 1e-6)``.

The SCAN parent (``scan_fx``, ``scan_fc``; Sun, Ruzsinszky and Perdew,
Phys. Rev. Lett. 115, 036402 (2015) and its supplemental material) is posed
on the same two conventions and reads the row's iso-orbital indicator on the
repository's definition (``metagga.compute_alpha``, the RAW indicator
recovered from the stored column by ``networks._raw_indicator``):

* The exchange row of an open shell carries the indicator of the DOUBLED
  channel, ``(2 tau_s - 4 sigma_ss / (8 2 rho_s)) / tau_unif(2 rho_s)``
  (``data.precompute_fixed_density_data``, the per-channel blocks), which is
  libxc's per-spin indicator ``(tau_s - tau_W,s) / (2^(2/3) tau_unif(rho_s))``
  identically, since ``tau_unif`` is homogeneous of degree 5/3 in the
  density; a closed shell's total density IS its doubled channel. ``scan_fx``
  therefore reads the row's indicator as it is.
* The correlation row carries the indicator of the TOTAL density,
  ``(tau - tau_W) / tau_unif(rho)``, WITHOUT the spin factor
  ``d_s(zeta) = [(1 + zeta)^(5/3) + (1 - zeta)^(5/3)] / 2`` that SCAN's
  ``tau_unif`` carries for a polarized density (libxc's ``scan_alpha`` divides
  by ``t_total(zeta, 1, 1)``, which is ``2^(-2/3) d_s``). ``scan_fc`` divides
  the row's indicator by ``d_s(zeta)``; at zeta = 0 the two agree.

Every constant is the value libxc 7.0.0 carries (``maple/mgga_exc/
mgga_x_scan.mpl``, ``mgga_c_scan.mpl``, ``maple/gga_exc/gga_c_scan_e0.mpl``
and ``gga_c_regtpss.mpl``), and four of them differ from the paper's printed
values by more than the 1e-12 the parent is held to: ``chi_infinity`` is
``0.12802585262625815`` (the paper's 0.128026 puts ``F_c`` 3.3e-7 off at
s = 6), the coefficient of ``G_c(zeta)`` is 2.363 (the paper's 2.3631 is
3.8e-5 off at zeta = 0.9), ``beta(rs)`` is built on libxc's PBE beta
(the paper's 0.066725 is 3.3e-6 off) and ``gamma`` is PBE's
``(1 - ln 2) / pi^2`` (the supplement's rounded 0.031091 is 1.2e-5 off); all
four measured on the (rs, s, alpha) grid of the tests. The PW92 inside
``eps_c^1`` is ``LDA_C_PW_MOD``, as for PBE.

libxc's input sanitation (``work_mgga_inc.c``) is reproduced, since libxc is
the oracle: each spin density is floored at ``LIBXC_SCAN_DENS_THRESHOLD``
(1e-15, the SCAN info structs), sigma at ``LIBXC_SCAN_SIGMA_FLOOR`` (the
square of ``sigma_threshold = dens_threshold^(4/3)``), tau at
``LIBXC_SCAN_TAU_FLOOR`` (1e-20), and then sigma is capped at ``8 rho tau``
per spin channel -- the von Weizsacker bound, i.e. ``alpha >= 0`` -- with
``sigma_ab`` held inside the mean of the capped diagonal invariants. The row
carries ``(rho, sigma, alpha)``, so tau is reconstructed as
``alpha tau_unif + sigma / (8 rho)`` and, for the correlation, the per-channel
sanitation is applied under the proportional split of the total quantities
(exact for a closed shell and the split the oracle uses; on an open-shell row
it can differ from libxc's per-channel arithmetic only where a channel
violates its own bound, which is a rounding residue of the stored indicator
on the model's domain). The value is taken at the sanitized point and the
derivative at that point unscaled (``_floor_as_libxc``). Where nothing is
clipped the reconstruction is exact: the indicator handed to the functional
is the row's own, with no cancellation. Two consequences are visible at the
1e-12 level and are matched: the empty channel of a fully polarized row
enters at 1e-15, which shifts ``tau_W = sigma / (8 rho)`` by ``2e-15 / rho``
relative and, through the ``tau - tau_W`` difference, the indicator by far
more (the F_c of the H atom's rows at alpha = 0 is that shift times
``G_c``); and the tau floor puts the indicator at ``1e-20 / tau_unif`` on
a row with ``tau = 0``. The switching function is libxc's: the branch
``exp(-c1 alpha / (1 - alpha))`` below alpha = 1 and ``-d exp(c2 / (1 -
alpha))`` above, each set to zero where its value falls under DBL_EPSILON
(alpha in [0.98183, 1.02206] for exchange, [0.98255, 1.04203] for
correlation), so both branches and their derivatives are finite at
alpha = 1, where all derivatives vanish. The conditioning of the indicator
bounds the agreement on the density tail: where ``tau_W`` exceeds
``tau - tau_W`` by a factor kappa, libxc's own indicator carries a rounding
residue of ``kappa`` ulps (``scan_fx`` docstring, measured), and no function
of the row's ``(rho, sigma, alpha)`` can follow it closer.
"""
import math

import jax
import jax.numpy as jnp

from xcquinox.alec.config import ArchitectureConfig
from xcquinox.utils import pw92c_polarized_scalar

#: Parent functionals an architecture can be anchored to, by rung.
PARENTS = ("pbe", "scan")

# --- PBE constants, as libxc defines them ----------------------------------
# Perdew, Burke, Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996), eqs. 3-8 and 14.
# beta is libxc's value (0.06672455060314922); the paper's 0.066725 is 6.7e-6
# relative off it, and the paper's mu = 0.21951 is 2.3e-5 off beta pi^2 / 3,
# which puts F_x 2.6e-6 relative off libxc at s = 1 (measured).
PBE_KAPPA = 0.804
PBE_BETA = 0.06672455060314922
PBE_MU = PBE_BETA * math.pi ** 2 / 3.0          # 0.2195149727645171
PBE_GAMMA = (1.0 - math.log(2.0)) / math.pi ** 2  # 0.031090690869654901

#: libxc's zeta_threshold: ``1 -+ zeta`` is floored here before a fractional
#: power is taken of it (``opz_pow_n`` in libxc's maple sources).
ZETA_FLOOR = 2.220446049250313e-16

#: libxc's density threshold for GGA_C_PBE: each spin density is floored here
#: before the functional is evaluated, so an empty channel enters at 1e-12
#: rather than 0 (measured on libxc 7.0.0, see the module docstring).
LIBXC_DENS_THRESHOLD = 1e-12


def _clip_zeta(zeta):
    """``zeta`` held in ``[-1, 1]`` with a unit derivative AT the bounds:
    ``jnp.clip`` is a min/max pair whose JAX derivative at an exact tie is
    the balanced 1/2, which would halve the chain through zeta on a row at
    exactly full polarization; the ``where`` form is 1 on the closed
    interval and 0 outside it."""
    return jnp.where(zeta > 1.0, 1.0, jnp.where(zeta < -1.0, -1.0, zeta))


def _floor_as_libxc(x, threshold):
    """``max(x, threshold)`` in value, with the derivative taken AT the floored
    point and passed through unscaled, which is how libxc reports ``vrho`` for
    a floored channel (its expression is differentiated at the clamped
    density). Below the threshold the function is therefore evaluated at the
    threshold and its derivative there is returned, finite, rather than the
    zero a plain clamp would give."""
    return x + jax.lax.stop_gradient(jnp.maximum(x, threshold) - x)

# --- PW92 on libxc's LDA_C_PW_MOD parameters --------------------------------
# The parameter set GGA_C_PBE is built on (libxc gga_c_pbe.c initializes its
# LDA part as XC_LDA_C_PW_MOD). Order: [eps_c(rs, 0), eps_c(rs, 1), -alpha_c].
_PW_MOD_A = (0.0310907, 0.01554535, 0.0168869)
_PW_MOD_ALPHA1 = (0.21370, 0.20548, 0.11125)
_PW_MOD_BETA1 = (7.5957, 14.1189, 10.357)
_PW_MOD_BETA2 = (3.5876, 6.1977, 3.6231)
_PW_MOD_BETA3 = (1.6382, 3.3662, 0.88026)
_PW_MOD_BETA4 = (0.49294, 0.62517, 0.49671)
#: f''(0) = 8 / (9 (2^(4/3) - 2)), exact, as LDA_C_PW_MOD carries it.
_PW_MOD_FZ20 = 8.0 / (9.0 * (2.0 ** (4.0 / 3.0) - 2.0))

_K_F_COEF = (3.0 * math.pi ** 2) ** (1.0 / 3.0)   # k_F = _K_F_COEF rho^(1/3)


def _pw92_mod_g(k, rs):
    """One PW92 ``G(rs)`` parametrization (Perdew and Wang eq. 10) on the
    modified parameter set."""
    a = _PW_MOD_A[k]
    b = (_PW_MOD_BETA1[k] * jnp.sqrt(rs) + _PW_MOD_BETA2[k] * rs
         + _PW_MOD_BETA3[k] * rs ** 1.5 + _PW_MOD_BETA4[k] * rs ** 2)
    return -2.0 * a * (1.0 + _PW_MOD_ALPHA1[k] * rs) * jnp.log1p(1.0 / (2.0 * a * b))


def _pw92_mod_eps(rs, zeta, opz, omz):
    """``eps_c^PW92(rs, zeta)`` on libxc's ``LDA_C_PW_MOD`` parameters
    (Perdew and Wang eqs. 8-10, the spin interpolation with the exact
    ``f''(0)``). ``opz`` and ``omz`` are ``1 + zeta`` and ``1 - zeta``,
    supplied by the caller formed from the spin densities directly
    (``2 rho_a / rho``, ``2 rho_b / rho``) so that near full polarization
    they carry no cancellation, and floored at :data:`ZETA_FLOOR`."""
    g0 = _pw92_mod_g(0, rs)
    g1 = _pw92_mod_g(1, rs)
    g2 = _pw92_mod_g(2, rs)
    opz = jnp.maximum(opz, ZETA_FLOOR)
    omz = jnp.maximum(omz, ZETA_FLOOR)
    fz = (opz ** (4.0 / 3.0) + omz ** (4.0 / 3.0) - 2.0) / (2.0 ** (4.0 / 3.0) - 2.0)
    z4 = zeta ** 4
    return g0 - g2 * fz / _PW_MOD_FZ20 * (1.0 - z4) + (g1 - g0) * fz * z4


# --- PBE ---------------------------------------------------------------------

def pbe_fx(rho, sigma):
    """PBE exchange enhancement factor ``F_x(s)`` (PBE eq. 14) at libxc's
    constants, for ONE spin channel posed on its doubled density: ``rho`` and
    ``sigma`` are the row's ``rho_x = 2 rho_sigma`` and ``sigma_x =
    4 sigma_sigma_sigma``. ``s^2 = sigma / (4 k_F^2 rho^2)`` is formed without
    a square root, so the derivative with respect to ``sigma`` is finite at
    ``sigma = 0`` as libxc's ``vsigma`` is."""
    rho = jnp.asarray(rho)
    sigma = jnp.asarray(sigma)
    k_f2 = _K_F_COEF ** 2 * rho ** (2.0 / 3.0)
    s2 = sigma / (4.0 * k_f2 * rho ** 2)
    return 1.0 + PBE_KAPPA - PBE_KAPPA / (1.0 + PBE_MU * s2 / PBE_KAPPA)


def pbe_fc(rho, sigma, zeta):
    """PBE correlation enhancement factor relative to the model's baseline:
    ``eps_c^PBE(rs, zeta, t) / eps_c^PW92(rs, zeta)`` with the numerator on
    libxc's parameters (PBE eqs. 3-8 on ``LDA_C_PW_MOD``) and the denominator
    the repository's ``pw92c_polarized_scalar``, so that ``rho eps_c^base
    pbe_fc`` is libxc's PBE correlation energy density (see the module
    docstring). ``rho`` and ``sigma`` are the TOTAL density and its gradient
    invariant; ``zeta`` the row's spin polarization, clipped to ``[-1, 1]``.
    ``t^2`` is formed from ``sigma`` without a square root.

    The numerator is evaluated at the spin densities floored at
    :data:`LIBXC_DENS_THRESHOLD`, as libxc evaluates it; the denominator is
    the model's baseline at the row's own ``(rho, zeta)``, as
    ``models.AlecGGAModel._ec_baseline`` forms it, so the model's
    ``rho eps_c^base F_c`` is the density times libxc's per-particle value,
    which is what pyscf integrates."""
    rho = jnp.asarray(rho)
    sigma = jnp.asarray(sigma)
    zeta_c = _clip_zeta(jnp.asarray(zeta))
    half = 0.5 * (1.0 + zeta_c)
    rho_a = _floor_as_libxc(rho * half, LIBXC_DENS_THRESHOLD)
    rho_b = _floor_as_libxc(rho * (1.0 - half), LIBXC_DENS_THRESHOLD)
    rho_f = rho_a + rho_b
    zeta_f = _clip_zeta((rho_a - rho_b) / rho_f)
    rs = (3.0 / (4.0 * math.pi * rho_f)) ** (1.0 / 3.0)
    # 1 +- zeta from the spin densities themselves: 1 - zeta near full
    # polarization is 2 rho_b / rho, of order 1e-11 at the floored channel,
    # and the difference 1 - (rho_a - rho_b) / rho would carry a relative
    # rounding error of order eps_mach rho / (2 rho_b) into phi's derivative.
    opz = jnp.maximum(2.0 * rho_a / rho_f, ZETA_FLOOR)
    omz = jnp.maximum(2.0 * rho_b / rho_f, ZETA_FLOOR)
    phi = 0.5 * (opz ** (2.0 / 3.0) + omz ** (2.0 / 3.0))
    eps_lda = _pw92_mod_eps(rs, zeta_f, opz, omz)
    # t^2 = sigma / (2 phi k_s rho)^2 with k_s^2 = 4 k_F / pi (PBE eq. 4).
    k_f = _K_F_COEF * rho_f ** (1.0 / 3.0)
    ks2 = 4.0 * k_f / math.pi
    t2 = sigma / (4.0 * phi ** 2 * ks2 * rho_f ** 2)
    gamma_phi3 = PBE_GAMMA * phi ** 3
    # A = (beta/gamma) / (exp(-eps_c^LDA / (gamma phi^3)) - 1)  (PBE eq. 8);
    # eps_c^LDA < 0 so the exponent is positive and expm1 is positive.
    a_coef = (PBE_BETA / PBE_GAMMA) / jnp.expm1(-eps_lda / gamma_phi3)
    at2 = a_coef * t2
    h_arg = (PBE_BETA / PBE_GAMMA) * t2 * (1.0 + at2) / (1.0 + at2 + at2 * at2)
    h_term = gamma_phi3 * jnp.log1p(h_arg)                  # PBE eq. 7
    eps_pbe = eps_lda + h_term                               # PBE eq. 3
    eps_base = pw92c_polarized_scalar(rho * half, rho * (1.0 - half))
    return eps_pbe / eps_base


# --- SCAN constants, as libxc defines them -----------------------------------
# Sun, Ruzsinszky and Perdew, Phys. Rev. Lett. 115, 036402 (2015), eqs. 2-4
# and the supplemental material, at the values of libxc 7.0.0 (maple sources
# named in the module docstring). Exchange: eqs. S6-S13 of the supplement.
MU_GE = 10.0 / 81.0                    # second-order gradient expansion
SCAN_K1 = 0.065
SCAN_H0X = 1.174                       # h_x^0, the exchange ceiling
SCAN_A1 = 4.9479
SCAN_C1X = 0.667
SCAN_C2X = 0.8
SCAN_DX = 1.24
SCAN_B2 = math.sqrt(5913.0 / 405000.0)
SCAN_B1 = (511.0 / 13500.0) / (2.0 * SCAN_B2)
SCAN_B3 = 0.5
SCAN_B4 = MU_GE ** 2 / SCAN_K1 - 1606.0 / 18225.0 - SCAN_B1 ** 2
# Correlation: eqs. S14-S29 of the supplement.
SCAN_C1C = 0.64
SCAN_C2C = 1.5
SCAN_DC = 0.7
SCAN_B1C = 0.0285764
SCAN_B2C = 0.0889
SCAN_B3C = 0.125541
#: libxc's ``scan_chi_infty``; the paper prints 0.128026 (1.2e-6 relative
#: off, 3.3e-7 in F_c at s = 6, measured).
SCAN_CHI_INF = 0.12802585262625815
#: The coefficient of ``G_c(zeta) = [1 - c (d_x(zeta) - 1)] (1 - zeta^12)``;
#: libxc carries 2.363 (``scan_G_cnst``), the paper prints 2.3631.
SCAN_G_CNST = 2.363
#: ``beta(rs) = beta_a (1 + beta_b rs) / (1 + beta_c rs)`` with libxc's PBE
#: beta (``gga_c_regtpss.mpl``), not the paper's rounded 0.066725.
SCAN_BETA_A = PBE_BETA
SCAN_BETA_B = 0.1
SCAN_BETA_C = 0.1778
#: gamma of ``H_1``: PBE's ``(1 - ln 2) / pi^2`` (``gga_c_pbe_params``), not
#: the supplement's rounded 0.031091.
SCAN_GAMMA = PBE_GAMMA

#: libxc's thresholds for ``MGGA_X_SCAN`` and ``MGGA_C_SCAN`` (libxc 7.0.0:
#: ``dens_threshold`` 1e-15 in both info structs; ``functionals.c`` sets
#: ``sigma_threshold = dens_threshold^(4/3)`` and ``tau_threshold = 1e-20``;
#: ``work_mgga_inc.c`` floors sigma at ``sigma_threshold^2`` and tau at
#: ``tau_threshold``, then caps sigma at ``8 rho tau``). Measured: the
#: exchange is zero at rho = 1e-15 and finite at 1e-14; ``F_x - 1`` at
#: sigma = 0 is ``mu p`` with ``p = 1e-40 / (4 k_F^2 rho^2)`` (3.2e-11 at
#: rho = 1e-12); ``F_x`` at tau = 0 sits at the indicator ``1e-20 / tau_unif``
#: (1.6e-4 at rho = 1e-10).
LIBXC_SCAN_DENS_THRESHOLD = 1e-15
LIBXC_SCAN_SIGMA_FLOOR = 1e-40
LIBXC_SCAN_TAU_FLOOR = 1e-20

#: ``tau_unif = _TAU_UNIF_COEF rho^(5/3)`` (``3/10 (3 pi^2)^(2/3)``).
_TAU_UNIF_COEF = 0.3 * _K_F_COEF ** 2


def _clip_as_libxc(x, lo=None, hi=None):
    """``x`` held to ``[lo, hi]`` in value with the derivative taken at the
    clipped point and passed through unscaled (:func:`_floor_as_libxc` with
    an upper bound as well), which is how libxc reports its potentials for a
    sanitized input."""
    y = x
    if lo is not None:
        y = jnp.maximum(y, lo)
    if hi is not None:
        y = jnp.minimum(y, hi)
    return x + jax.lax.stop_gradient(y - x)


def _row_tau(rho, sigma, alpha):
    """The kinetic-energy density a row's raw indicator stands for,
    ``tau = alpha tau_unif(rho) + sigma / (8 rho)`` (the inverse of
    ``metagga.compute_alpha`` on the same conventions)."""
    return alpha * _TAU_UNIF_COEF * rho ** (5.0 / 3.0) + sigma / (8.0 * rho)


def _sanitized_indicator(alpha, rho, sigma, rho_f, sigma_f, tau, tau_f,
                         tau_unif_f):
    """The indicator at libxc's sanitized inputs ``(rho_f, sigma_f, tau_f)``,
    written as the row's indicator plus the increments the sanitation made,
    so that it is the row's ``alpha`` exactly (no ``tau - tau_W``
    cancellation) whenever nothing was clipped, and libxc's value, with
    libxc's derivative semantics, where something was."""
    tau_unif = _TAU_UNIF_COEF * rho ** (5.0 / 3.0)
    dtau = tau_f - tau
    dsig = sigma_f - sigma
    drho_term = sigma * (rho_f - rho) / (8.0 * rho * rho_f)
    return (alpha * tau_unif + dtau + drho_term
            - dsig / (8.0 * rho_f)) / tau_unif_f


def _scan_switch(alpha, c1, c2, d):
    """SCAN's interpolation ``f(alpha)`` as libxc evaluates it
    (``scan_f_alpha`` of ``mgga_x_scan.mpl``, shared with the correlation at
    its own constants): ``exp(-c1 alpha / (1 - alpha))`` for ``alpha <= 1``,
    set to zero above ``ln(1/eps) / (ln(1/eps) + c1)`` where it has fallen
    under ``eps = DBL_EPSILON``; ``-d exp(c2 / (1 - alpha))`` for
    ``alpha > 1``, zero below ``(ln(d/eps) + c2) / ln(d/eps)``. Each branch
    is evaluated on an argument held inside its own domain, so the value and
    every derivative are finite at alpha = 1 (where all of them vanish) and
    for every alpha, negative values included (the left branch is bounded by
    ``exp(c1)``)."""
    ln_eps = -math.log(ZETA_FLOOR)
    left_cut = ln_eps / (ln_eps + c1)
    ln_d = -math.log(ZETA_FLOOR / abs(d))
    right_cut = (ln_d + c2) / ln_d
    a_l = jnp.minimum(alpha, left_cut)
    om_l = 1.0 - a_l
    left = jnp.where(alpha > left_cut, 0.0,
                     jnp.exp(-c1 * a_l / jnp.where(om_l > 0.0, om_l, 1.0)))
    a_r = jnp.maximum(alpha, right_cut)
    om_r = 1.0 - a_r
    right = jnp.where(alpha < right_cut, 0.0,
                      -d * jnp.exp(c2 / jnp.where(om_r < 0.0, om_r, -1.0)))
    return jnp.where(alpha <= 1.0, left, right)


def _scan_fx_core(p, alpha):
    """``F_x^SCAN(p = s^2, alpha)``: ``[h_x^1(y) (1 - f) + h_x^0 f] g_x(s)``
    with ``y`` of eq. S9 (the ``b4 p^2 exp(-b4 p / mu)`` term and the
    squared ``b1 p + b2 (1 - alpha) exp(-b3 (1 - alpha)^2)``), ``h_x^1(y) =
    1 + k1 - k1^2 / (k1 + y)`` and ``g_x = 1 - exp(-a1 / s^(1/2))``. ``p`` is
    positive after libxc's sigma floor; the guard keeps the derivative of
    ``g_x`` finite (it is zero to double precision) on any smaller value."""
    oma = 1.0 - alpha
    y = (MU_GE * p * (1.0 + (SCAN_B4 * p / MU_GE) * jnp.exp(-SCAN_B4 * p / MU_GE))
         + (SCAN_B1 * p + SCAN_B2 * oma * jnp.exp(-SCAN_B3 * oma * oma)) ** 2)
    h1x = 1.0 + SCAN_K1 - SCAN_K1 * SCAN_K1 / (SCAN_K1 + y)
    fx = _scan_switch(alpha, SCAN_C1X, SCAN_C2X, SCAN_DX)
    p_g = jnp.where(p > 1e-100, p, 1e-100)
    gx = 1.0 - jnp.exp(-SCAN_A1 / p_g ** 0.25)
    return (h1x * (1.0 - fx) + SCAN_H0X * fx) * gx


def scan_fx(rho, sigma, alpha):
    """SCAN exchange enhancement factor ``F_x(s, alpha)`` at libxc's
    constants, for ONE spin channel posed on its doubled density: ``rho``
    and ``sigma`` are the row's ``2 rho_sigma`` and ``4 sigma_sigma_sigma``
    and ``alpha`` the RAW iso-orbital indicator of that doubled channel
    (``metagga.compute_alpha`` before its smoothing and ceiling, which
    ``networks._raw_indicator`` recovers from the stored column), so that
    ``rho eps_x^LDA(rho) scan_fx`` is libxc's ``MGGA_X_SCAN`` energy density
    at ``(rho, sigma, tau = alpha tau_unif + sigma / (8 rho))``. The row's
    indicator on the doubled channel is libxc's per-spin indicator
    identically (module docstring).

    libxc's sanitation is applied first (module docstring): rho floored at
    1e-15, sigma at 1e-40, tau at 1e-20, sigma capped at ``8 rho tau`` so
    that ``alpha >= 0``; a row whose raw indicator is negative (a rounding
    residue of ``tau - tau_W`` on a one-orbital channel, at most 1e-11 on
    the model's domain, ``metagga.py``) is therefore evaluated at
    ``alpha = 0`` and ``s^2 - 0.6 |alpha|``, which is libxc's value at the
    true tau, and its derivatives are libxc's, taken at that point. Where
    nothing is clipped the function is the analytic SCAN at the row's
    ``(rho, sigma, alpha)`` exactly.

    Measured against ``pyscf.dft.libxc.eval_xc("MGGA_X_SCAN", spin=0)`` (libxc
    7.0.0): 2.6e-15 relative over the (rs, s, alpha) grid of the tests with
    alpha in {0, 0.5, 0.99, 1, 1.01, 2, 10, 100} (936 points), and on the
    stored rows of H2O (sto-3g) and OH (def2-svp) at grid level 1 with
    ``2 rho_sigma > 1e-10`` below the indicator ceiling (8413, 6283 and 6324
    rows): 3.4e-15 where ``kappa = tau_W / (tau - tau_W) < 1e2``, 3.2e-13
    where ``kappa < 1e4``, and up to 7.9e-11 on the 130 to 286 tail rows
    beyond (rho below 3e-6, s above ~100, kappa up to 5.6e12), which is
    1.6 kappa ulps at most: there the indicator libxc recomputes from tau is
    itself determined only to kappa ulps (the tail figure moves between
    7.2e-11 and 7.9e-11 across two SCF solutions of the same system, the
    reference density's own rounding entering the difference). First derivatives
    with respect to rho, sigma and the indicator (through ``dtau/dalpha =
    tau_unif``) against ``deriv=1``: 2.1e-13 on the grid, 1.5e-9 on the
    stored rows. Accepts any broadcastable leading shape (elementwise), as
    ``pbe_fx`` does.
    """
    rho = jnp.asarray(rho)
    sigma = jnp.asarray(sigma)
    alpha = jnp.asarray(alpha)
    tau = _row_tau(rho, sigma, alpha)
    rho_f = _floor_as_libxc(rho, LIBXC_SCAN_DENS_THRESHOLD)
    tau_f = _clip_as_libxc(tau, lo=LIBXC_SCAN_TAU_FLOOR)
    sigma_f = _clip_as_libxc(sigma, lo=LIBXC_SCAN_SIGMA_FLOOR,
                             hi=8.0 * rho_f * tau_f)
    k_f2 = _K_F_COEF ** 2 * rho_f ** (2.0 / 3.0)
    p = sigma_f / (4.0 * k_f2 * rho_f ** 2)
    alpha_f = _sanitized_indicator(alpha, rho, sigma, rho_f, sigma_f, tau,
                                   tau_f, 0.3 * k_f2 * rho_f)
    return _scan_fx_core(p, alpha_f)


def scan_fc(rho, sigma, zeta, alpha):
    """SCAN correlation enhancement factor relative to the model's baseline:
    ``eps_c^SCAN(rs, zeta, t, alpha) / eps_c^PW92(rs, zeta)`` with the
    numerator libxc's ``MGGA_C_SCAN`` (``eps_c^1 + f_c(alpha) (eps_c^0 -
    eps_c^1)``: ``eps_c^1`` the PBE form on ``LDA_C_PW_MOD`` with SCAN's
    ``beta(rs)``, ``gamma`` and ``g(A t^2) = (1 + 4 A t^2)^(-1/4)``;
    ``eps_c^0 = (eps_c^LDA0 + H_0) G_c(zeta)`` with ``H_0 = b1c ln[1 + w0 (1 -
    g_inf(s))]``) and the denominator the repository's
    ``pw92c_polarized_scalar`` at the row's ``(rho, zeta)``, exactly as
    :func:`pbe_fc` is built, so that the model's ``rho eps_c^base scan_fc``
    is libxc's SCAN correlation energy density. ``rho`` and ``sigma`` are the
    TOTAL density and its gradient invariant, ``zeta`` the row's spin
    polarization (clipped to ``[-1, 1]``) and ``alpha`` the RAW indicator of
    the total density on the repository's definition, WITHOUT the spin
    factor; it is divided by ``d_s(zeta)`` here (module docstring).

    libxc's per-channel sanitation is applied under the proportional split
    of the row's total quantities (module docstring); the reconstruction of
    the indicator at the floored spin densities is what makes the fully
    polarized rows agree: at zeta = +-1 and alpha = 0 the parent is
    ``G_c(zeta_f) (eps_c^LDA0 + H_0) / eps_c^PW92`` with ``1 - zeta_f =
    2e-15 / rho``, of order 1e-12 or smaller -- the correlation-free
    one-orbital limit as libxc reaches it -- and the pre-image clamp of the
    anchored map takes it as a parent at its bound.

    Measured against ``eval_xc("MGGA_C_SCAN", spin=1)`` on proportionally
    split rows (libxc 7.0.0), the (rs, s, alpha) grid at zeta in {0, +-0.5,
    +-0.9, +-(1 - 1e-6), +-1} (8424 points): 1.5e-13 relative wherever
    ``|F_c| > 1e-5`` and ``|zeta| <= 1 - 1e-6`` (the production clip of
    ``oneshot.uks_zeta``), 3.8e-12 at zeta = +-1 exactly, where libxc forms
    ``1 - zeta`` from the rounded zeta of the floored empty channel (a
    quantity quantized in units of 1.1e-16 around the true 1.6e-16 at rs =
    0.27; the same effect the PBE parent's test bounds at 3e-5); on the 468
    rows at alpha = 0 with |zeta| -> 1, where ``F_c`` falls to 1e-6 and below
    because ``G_c`` vanishes and the two evaluations sit on libxc's
    Fermi-hole cap (``sigma_ss = 8 rho_s tau_s`` to rounding), the agreement
    is 4.3e-15 absolute. On the stored rows of H2O and OH (total density,
    ``uks_zeta``, ``rho > 1e-10``, below the indicator ceiling; 8413 and
    6324 rows): 3.1e-14 where ``kappa = tau_W / (tau - tau_W) < 1e2``,
    1.3e-12 where ``kappa < 1e4``, up to 1.6e-9 on the tail rows beyond it
    (the indicator's own conditioning, as for ``scan_fx``; 1.2 kappa ulps at
    most), 1.1e-14 Ha/bohr^3 absolute in the energy density; derivatives
    against ``deriv=1`` to 5.6e-9 on the grid (zeta, sigma and rho at the
    small-``F_c`` rows; 2.8e-13 where ``|F_c| > 1e-5`` except zeta at
    5.6e-9) and 6.2e-9 on the stored rows. Accepts any broadcastable
    leading shape, as ``pbe_fc``.
    """
    rho = jnp.asarray(rho)
    sigma = jnp.asarray(sigma)
    alpha = jnp.asarray(alpha)
    zeta_c = _clip_zeta(jnp.asarray(zeta))
    half = 0.5 * (1.0 + zeta_c)
    omhalf = 1.0 - half
    tau = _row_tau(rho, sigma, alpha)
    # libxc's per-channel sanitation under the proportional split of the
    # row's total quantities: each spin density floored, each channel's tau
    # floored, each diagonal gradient invariant floored and then capped at
    # the channel's Fermi-hole bound, and sigma_ab held inside the mean of
    # the two capped diagonals (work_mgga_inc.c, the polarized branch).
    rho_a = _floor_as_libxc(rho * half, LIBXC_SCAN_DENS_THRESHOLD)
    rho_b = _floor_as_libxc(rho * omhalf, LIBXC_SCAN_DENS_THRESHOLD)
    tau_a = _clip_as_libxc(tau * half, lo=LIBXC_SCAN_TAU_FLOOR)
    tau_b = _clip_as_libxc(tau * omhalf, lo=LIBXC_SCAN_TAU_FLOOR)
    sig_aa = _clip_as_libxc(sigma * half * half, lo=LIBXC_SCAN_SIGMA_FLOOR,
                            hi=8.0 * rho_a * tau_a)
    sig_bb = _clip_as_libxc(sigma * omhalf * omhalf,
                            lo=LIBXC_SCAN_SIGMA_FLOOR, hi=8.0 * rho_b * tau_b)
    s_ave = 0.5 * (sig_aa + sig_bb)
    sig_ab = _clip_as_libxc(sigma * half * omhalf, lo=-s_ave, hi=s_ave)
    rho_f = rho_a + rho_b
    tau_f = tau_a + tau_b
    sigma_f = sig_aa + 2.0 * sig_ab + sig_bb
    zeta_f = _clip_zeta((rho_a - rho_b) / rho_f)
    rs = (3.0 / (4.0 * math.pi * rho_f)) ** (1.0 / 3.0)
    # 1 +- zeta from the spin densities (see pbe_fc), floored at libxc's
    # zeta threshold before the fractional powers.
    opz = jnp.maximum(2.0 * rho_a / rho_f, ZETA_FLOOR)
    omz = jnp.maximum(2.0 * rho_b / rho_f, ZETA_FLOOR)
    phi = 0.5 * (opz ** (2.0 / 3.0) + omz ** (2.0 / 3.0))
    d_x = 0.5 * (opz ** (4.0 / 3.0) + omz ** (4.0 / 3.0))
    d_s = 0.5 * (opz ** (5.0 / 3.0) + omz ** (5.0 / 3.0))
    eps_lsda1 = _pw92_mod_eps(rs, zeta_f, opz, omz)
    k_f2 = _K_F_COEF ** 2 * rho_f ** (2.0 / 3.0)
    k_f = jnp.sqrt(k_f2)
    p = sigma_f / (4.0 * k_f2 * rho_f ** 2)                 # s^2
    # The indicator on SCAN's polarized tau_unif: the row's, re-derived at
    # the sanitized inputs, divided by d_s(zeta).
    alpha_f = _sanitized_indicator(alpha, rho, sigma, rho_f, sigma_f, tau,
                                   tau_f, 0.3 * k_f2 * rho_f) / d_s
    # t^2 = sigma / (2 phi k_s rho)^2 with k_s^2 = 4 k_F / pi (PBE eq. 4).
    ks2 = 4.0 * k_f / math.pi
    t2 = p * k_f2 / (phi ** 2 * ks2)
    # eps_c^1: eqs. S17-S20 of the supplement at libxc's beta(rs) and gamma.
    beta = SCAN_BETA_A * (1.0 + SCAN_BETA_B * rs) / (1.0 + SCAN_BETA_C * rs)
    gphi3 = SCAN_GAMMA * phi ** 3
    w1 = jnp.expm1(-eps_lsda1 / gphi3)
    a_coef = beta / (SCAN_GAMMA * w1)
    g1 = (1.0 + 4.0 * a_coef * t2) ** (-0.25)
    h1 = gphi3 * jnp.log1p(w1 * (1.0 - g1))
    eps1 = eps_lsda1 + h1
    # eps_c^0: eqs. S21-S29.
    eps_lda0 = -SCAN_B1C / (1.0 + SCAN_B2C * jnp.sqrt(rs) + SCAN_B3C * rs)
    w0 = jnp.expm1(-eps_lda0 / SCAN_B1C)
    g_inf = (1.0 + 4.0 * SCAN_CHI_INF * p) ** (-0.25)
    h0 = SCAN_B1C * jnp.log1p(w0 * (1.0 - g_inf))
    g_c = (1.0 - SCAN_G_CNST * (d_x - 1.0)) * (1.0 - zeta_f ** 12)
    eps0 = (eps_lda0 + h0) * g_c
    fc = _scan_switch(alpha_f, SCAN_C1C, SCAN_C2C, SCAN_DC)
    eps_scan = eps1 + fc * (eps0 - eps1)
    eps_base = pw92c_polarized_scalar(rho * half, rho * omhalf)
    return eps_scan / eps_base


# --- dispatch ----------------------------------------------------------------

def _check_parent(parent: str) -> None:
    if parent not in PARENTS:
        raise ValueError(
            f"unknown parent functional {parent!r}; the anchor knows "
            f"{PARENTS}")


def _require_indicator(where: str, alpha) -> None:
    if alpha is None:
        raise ValueError(
            f"{where}: the SCAN parent reads the raw iso-orbital indicator "
            "and none was given (alpha=None); a meta-GGA row carries it at "
            "metagga_alpha_index, and networks._raw_indicator recovers the "
            "raw value from the stored column")


def parent_fx(parent: str, rho, sigma, alpha=None):
    """``F_x`` of the named parent (``"pbe"`` | ``"scan"``) on a doubled spin
    channel; ``alpha`` is the raw iso-orbital indicator the SCAN parent
    requires (``ValueError`` when it is missing) and PBE ignores."""
    _check_parent(parent)
    if parent == "pbe":
        return pbe_fx(rho, sigma)
    _require_indicator("parent_fx('scan')", alpha)
    return scan_fx(rho, sigma, alpha)


def parent_fc(parent: str, rho, sigma, zeta, alpha=None):
    """``F_c`` of the named parent on the total density relative to the
    model's polarized PW92 baseline; ``alpha`` as in :func:`parent_fx`."""
    _check_parent(parent)
    if parent == "pbe":
        return pbe_fc(rho, sigma, zeta)
    _require_indicator("parent_fc('scan')", alpha)
    return scan_fc(rho, sigma, zeta, alpha)


def parent_for_arch(arch) -> str:
    """The parent an architecture is anchored to, by rung: ``"scan"`` on the
    meta-GGA rung (``ArchitectureConfig.is_meta_gga``, the one predicate),
    ``"pbe"`` otherwise. Agrees with ``cluster.fidelity.resolve_parent`` by
    construction, since that reads the same predicate through ``rungs``."""
    return "scan" if ArchitectureConfig.is_meta_gga(arch) else "pbe"


# --- the pre-image of the bounded map ----------------------------------------

#: Smallest positive normal float64: the floor under the pre-image's logarithm
#: arguments, so a parent at or beyond a bound yields a finite value that the
#: clamp then takes to ``+-z_max`` rather than a NaN.
_TINY = 2.2250738585072014e-308


def lob_preimage(F_parent, limit, z_max=40.0):
    """The point ``z`` at which the networks' bounded map returns the parent:
    ``1 + L(z) = F_parent`` with ``L(x) = limit sigmoid(x - ln(limit - 1)) - 1``
    (``networks._AlecLOB``), i.e. ``z = ln[(limit - 1) F_parent /
    (limit - F_parent)]``, clamped to ``[-z_max, z_max]``.

    The clamp binds only where the parent sits within ``limit e^(-z_max)`` of
    a bound of ``(0, limit)`` (``z_max = 40``: 8e-18 for the correlation
    squash at 2.0); there the map returns the parent to that accuracy and the
    network cannot move it, which is the parent's own limit rather than a
    degeneracy of the transform (``SPEC_parent_anchor.md`` Section 3.2). Both
    logarithm arguments are floored at the smallest normal float so a parent
    evaluated at or past a bound by round-off (``F_c`` of order 1e-17 in a
    density tail can round to zero or below) clamps instead of returning NaN.
    """
    f = jnp.asarray(F_parent)
    num = jnp.maximum((limit - 1.0) * f, _TINY)
    den = jnp.maximum(limit - f, _TINY)
    return jnp.clip(jnp.log(num) - jnp.log(den), -z_max, z_max)
