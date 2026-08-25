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

The SCAN parent (``scan_fx``, ``scan_fc``) is the next commit's; both raise
``NotImplementedError`` here and ``networks.create_network_pair`` refuses a
meta-GGA architecture with ``parent_anchor=True`` until it lands.
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


# --- SCAN (the next commit) --------------------------------------------------

_SCAN_MESSAGE = (
    "the SCAN parent (Sun, Ruzsinszky and Perdew, Phys. Rev. Lett. 115, "
    "036402 (2015)) lands in the SCAN commit that follows the PBE anchor; "
    "SPEC_parent_anchor.md Section 3.7 sequences it second")


def scan_fx(rho, sigma, alpha):
    """SCAN exchange enhancement factor ``F_x(s, alpha)`` on the raw
    indicator; not yet implemented (see the module docstring)."""
    raise NotImplementedError(f"scan_fx: {_SCAN_MESSAGE}")


def scan_fc(rho, sigma, zeta, alpha):
    """SCAN correlation enhancement factor relative to the model's baseline;
    not yet implemented (see the module docstring)."""
    raise NotImplementedError(f"scan_fc: {_SCAN_MESSAGE}")


# --- dispatch ----------------------------------------------------------------

def _check_parent(parent: str) -> None:
    if parent not in PARENTS:
        raise ValueError(
            f"unknown parent functional {parent!r}; the anchor knows "
            f"{PARENTS}")


def parent_fx(parent: str, rho, sigma, alpha=None):
    """``F_x`` of the named parent (``"pbe"`` | ``"scan"``) on a doubled spin
    channel; ``alpha`` is the raw iso-orbital indicator the SCAN parent reads
    and is ignored by PBE."""
    _check_parent(parent)
    if parent == "pbe":
        return pbe_fx(rho, sigma)
    return scan_fx(rho, sigma, alpha)


def parent_fc(parent: str, rho, sigma, zeta, alpha=None):
    """``F_c`` of the named parent on the total density relative to the
    model's polarized PW92 baseline; ``alpha`` as in :func:`parent_fx`."""
    _check_parent(parent)
    if parent == "pbe":
        return pbe_fc(rho, sigma, zeta)
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
