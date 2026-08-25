"""The SCAN parent's oracles beyond the three ``test_parents`` carries: the
conventions the row's iso-orbital indicator reaches ``parents.scan_fx`` /
``scan_fc`` on, libxc's input sanitation, the switching function's two
branches, the stored molecular grids, and the constants that separate libxc
from the printed paper.

``test_parents`` states the SCAN value and the SCAN exchange potential on the
``(rs, s, alpha)`` mesh. What is stated here is everything that mesh does not
reach:

* the CORRELATION potential (``rho``, ``sigma``, ``zeta`` and the indicator)
  against libxc's ``deriv=1``;
* the spin factor ``d_s(zeta) = [(1 + zeta)^(5/3) + (1 - zeta)^(5/3)] / 2``.
  The repository's indicator (``metagga.compute_alpha``) is posed on
  ``tau_unif`` of the TOTAL density, while SCAN's correlation carries the
  polarized ``tau_unif``; ``scan_fc`` divides the row's indicator by ``d_s``
  and the case below fails if it does not;
* libxc's Fermi-hole cap ``sigma_ss <= 8 rho_s tau_s``, which is what makes a
  row whose raw indicator has gone negative on the rounding of ``tau - tau_W``
  evaluate at libxc's own value;
* the two DBL_EPSILON cutoffs of the switching function, on both sides of
  ``alpha = 1``, where the naive form is ``0/0``;
* full polarization, ``zeta = +-1`` exactly, where libxc forms ``1 - zeta``
  from the rounded zeta of a floored empty channel;
* the STORED rows of a closed and an open shell, binned by the conditioning
  of the indicator itself (``kappa = tau_W / (tau - tau_W)``: libxc's own
  indicator, recomputed from ``tau``, is determined only to ``kappa`` ulps, so
  no function of the row's ``(rho, sigma, alpha)`` can follow it closer), with
  the rows AT the descriptor's ceiling (``metagga._ALPHA_MAX = 100``) stated
  separately as the saturation floor they are rather than as agreement;
* the four constants at which libxc departs from the printed paper, so that a
  later reading of the paper does not "correct" them back.

Conventions are ``test_parents``'s (exchange on the doubled spin channel,
correlation on the total density relative to the model's polarized PW92
baseline) and its oracle helpers are imported rather than restated.

Environment the numbers quoted in the docstrings were measured on: pyscf
2.11.0, libxc 7.0.0, ``jax_enable_x64``, CPU, 4 threads.
"""
import math

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from pyscf import dft

from xcquinox.alec import parents
from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.descriptors import (assemble_descriptor_features,
                                       make_descriptor)
from xcquinox.alec.metagga import _ALPHA_MAX
from xcquinox.alec.networks import _raw_indicator
from xcquinox.alec.oneshot import uks_zeta
from xcquinox.alec.tests.test_parents import (_RS_GRID, _S_GRID, _libxc_fc,
                                              _libxc_fx, _mgga_row, _rel,
                                              _rho_of_rs, _scan_tau,
                                              _sigma_of_s, _spin_split)
from xcquinox.utils import lda_x, pw92c_polarized_scalar


#: The indicator values every case here is stated at. ``test_parents``'s grid
#: with the two sides of the switching function's branch point added: 0.99 and
#: 1.01 sit INSIDE the DBL_EPSILON dead bands of both branches, which the
#: coarser grid steps over.
_ALPHA_GRID = (0.0, 0.5, 0.99, 1.0, 1.01, 2.0, 10.0, 100.0)

#: The polarizations. ``+-(1 - 1e-6)`` is the production clip
#: (``oneshot.uks_zeta``); exactly ``+-1`` is its own case below.
_ZETA_GRID = (0.0, 0.5, -0.5, 0.9, -0.9)
_ZETA_CLIP = 1.0 - 1e-6

#: ``tau_unif = _TAU_UNIF (rho)^(5/3)``, ``3/10 (3 pi^2)^(2/3)``.
_TAU_UNIF = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)

#: The model's tail threshold (``models._NN_TAIL_THRESHOLD``): below it the
#: model masks ``F`` to 1 and the parent is not compared pointwise.
_RHO_FLOOR = 1e-10

#: ``DBL_EPSILON``, the cutoff libxc puts under each branch of the switching
#: function and the floor it puts under ``1 -+ zeta`` (``parents.ZETA_FLOOR``).
_DBL_EPSILON = float(np.finfo(np.float64).eps)


def _d_s(zeta):
    """The spin factor SCAN's ``tau_unif`` carries, ``[(1 + zeta)^(5/3) +
    (1 - zeta)^(5/3)] / 2`` (Sun, Ruzsinszky and Perdew, PRL 115, 036402
    (2015), eq. 3 and its supplement; libxc's ``t_total(zeta, 1, 1)`` is
    ``2^(-2/3) d_s``). It is 1 at zeta = 0 and 2^(2/3) at full
    polarization."""
    return 0.5 * ((1.0 + zeta) ** (5.0 / 3.0) + (1.0 - zeta) ** (5.0 / 3.0))


def _dead_band(c1, c2, d):
    """The interval of ``alpha`` on which ``parents._scan_switch`` returns
    zero: the left branch ``exp(-c1 alpha / (1 - alpha))`` has fallen under
    DBL_EPSILON above ``ln(1/eps) / (ln(1/eps) + c1)`` and the right branch
    ``-d exp(c2 / (1 - alpha))`` below ``(ln(d/eps) + c2) / ln(d/eps)``, which
    is how libxc cuts them off."""
    ln_eps = -math.log(_DBL_EPSILON)
    ln_d = -math.log(_DBL_EPSILON / abs(d))
    return ln_eps / (ln_eps + c1), (ln_d + c2) / ln_d


# ---------------------------------------------------------------------------
# Stored molecular rows
# ---------------------------------------------------------------------------

#: Records are read and never written, so one per (system, basis, grid) is
#: shared by the cases below.
_RECORDS = {}

#: The bins the stored-row agreement is stated in. ``kappa = tau_W /
#: (tau - tau_W)`` is the conditioning of the indicator: libxc recomputes it
#: from ``tau`` and loses ``kappa`` ulps to the cancellation, so the agreement
#: a function of ``(rho, sigma, alpha)`` can reach degrades with it and a
#: single bound over the whole grid would be a statement about the tail alone.
_KAPPA_BINS = (1e2, 1e3, 1e4, 1e5, np.inf)


def _record(name, atom, basis, spin, composition, grid_level=1):
    """A reference record carrying the meta-GGA indicator block."""
    key = (name, basis, spin, grid_level)
    if key not in _RECORDS:
        descriptor = make_descriptor("metagga")
        _RECORDS[key] = precompute_fixed_density_data(
            MoleculeSpec(name=name, atom=atom, basis=basis, charge=0,
                         spin=spin, atom_composition=composition,
                         grid_level=grid_level),
            required_keys=tuple(sorted(descriptor.required_mol_keys)),
            descriptors=(descriptor,))
    return _RECORDS[key]


def _h2o_record():
    """A closed shell: H2O at sto-3g, grid level 1."""
    return _record("H2O", "O 0.0 0.0 0.0; H 0.0 0.757 0.587; "
                          "H 0.0 -0.757 0.587", "sto-3g", 0,
                   (("H", 2), ("O", 1)))


def _oh_record():
    """An open shell: the OH radical at def2-svp, grid level 1."""
    return _record("OH", "O 0.0 0.0 0.0; H 0.0 0.0 0.97", "def2-svp", 1,
                   (("H", 1), ("O", 1)))


def _spin_pieces(md):
    """``(rho_sigma, sigma_sigma_sigma, tau_sigma)`` per spin channel,
    contracted from the record's stored AO derivative table and density matrix
    exactly as ``data.precompute_fixed_density_data`` builds its blocks. The
    kinetic-energy density is ``1/2 sum_k grad_k chi . P . grad_k chi``
    (``metagga.compute_tau_from_dm``), which is what the stored indicator
    column was formed from."""
    ao = np.asarray(md["ao_grid_deriv"])
    dm = np.asarray(md["dm_pbe"])
    out = []
    for s in (0, 1):
        d = dm[s] if dm.ndim == 3 else 0.5 * dm
        rho = np.einsum("gi,ij,gj->g", ao[0], d, ao[0])
        grad = [2.0 * np.einsum("gi,ij,gj->g", ao[k], d, ao[0])
                for k in (1, 2, 3)]
        tau = 0.5 * sum(np.einsum("gi,ij,gj->g", ao[k], d, ao[k])
                        for k in (1, 2, 3))
        out.append((rho, grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2, tau))
    return out


def _indicator_column(md, spin_channel):
    """The stored ``metagga`` column of one block: the doubled spin channel
    (``spin_channel`` 0 or 1) or the total density (``None``). A closed-shell
    record carries no per-channel block -- ``rho_a = rho_b`` makes the doubled
    channel the total density -- so its block IS the total one."""
    descriptors = (make_descriptor("metagga"),)
    channel = spin_channel if bool(md["is_unrestricted"]) else None
    return np.asarray(assemble_descriptor_features(
        descriptors, md, spin_channel=channel)).ravel()


def _exchange_rows(md, spin_channel):
    """``(rho, sigma, tau, alpha_raw, at_ceiling)`` of one doubled spin
    channel above the model's tail threshold. ``alpha_raw`` is the RAW
    indicator ``networks._raw_indicator`` recovers from the stored column --
    the value the anchored network hands the parent -- and ``at_ceiling``
    marks the rows whose column sits at ``metagga._ALPHA_MAX``, where the
    recovery returns the ceiling instead."""
    rho_s, sigma_ss, tau_s = _spin_pieces(md)[spin_channel]
    column = _indicator_column(md, spin_channel)
    keep = 2.0 * rho_s > _RHO_FLOOR
    return (2.0 * rho_s[keep], 4.0 * sigma_ss[keep], 2.0 * tau_s[keep],
            np.asarray(_raw_indicator(jnp.asarray(column[keep]))),
            column[keep] >= _ALPHA_MAX)


def _correlation_rows(md):
    """``(rho, sigma, tau, zeta, alpha_raw, at_ceiling)`` of the total density
    above the tail threshold, at the production spin polarization
    (``oneshot.uks_zeta``, the guards the energy path applies)."""
    pieces = _spin_pieces(md)
    rho = np.asarray(md["rho_grid"])
    sigma = np.asarray(md["sigma_grid"])
    tau = pieces[0][2] + pieces[1][2]
    zeta = np.asarray(uks_zeta(jnp.asarray(pieces[0][0]),
                               jnp.asarray(pieces[1][0])))
    column = _indicator_column(md, None)
    keep = rho > _RHO_FLOOR
    return (rho[keep], sigma[keep], tau[keep], zeta[keep],
            np.asarray(_raw_indicator(jnp.asarray(column[keep]))),
            column[keep] >= _ALPHA_MAX)


def _kappa(rho, sigma, alpha):
    """``tau_W / (tau - tau_W)`` of a row, written from the quantities the row
    carries: ``tau_W = sigma / (8 rho)`` and ``tau - tau_W = alpha
    tau_unif``."""
    return (sigma / (8.0 * rho)) / np.maximum(alpha * _TAU_UNIF
                                              * rho ** (5.0 / 3.0), 1e-300)


def _assert_binned(got, want, kappa, bounds, label):
    """The relative agreement, bin by bin in ``kappa``, against ``bounds``.

    Each bin is asserted separately and reported with the count it held, so a
    regression in the well-conditioned bulk is not hidden behind the tail's
    looser bound.
    """
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    rel = np.abs(got - want) / np.maximum(np.abs(want), 1e-300)
    low = 0.0
    for high, bound in zip(_KAPPA_BINS, bounds):
        mask = (kappa >= low) & (kappa < high)
        assert int(mask.sum()) > 0, (label, low, high, "empty bin")
        worst = float(rel[mask].max())
        assert worst <= bound, (label, low, high, int(mask.sum()), worst, bound)
        low = high


def _assert_ulp_bound(got, want, kappa, label, bound=50.0):
    """The disagreement, in ulps of the indicator's own conditioning.

    ``kappa`` ulps is the floor no function of ``(rho, sigma, alpha)`` can
    beat, since libxc recomputes the indicator from ``tau`` and loses that
    much to the cancellation. Stating the ratio as well as the bins makes the
    bound a property of the arithmetic rather than of the particular record,
    which is what lets the bins be read as measurements of one system.
    """
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    mask = kappa >= 1.0
    assert int(mask.sum()) > 1000, (label, int(mask.sum()))
    rel = np.abs(got[mask] - want[mask]) / np.maximum(np.abs(want[mask]),
                                                      1e-300)
    worst = float((rel / kappa[mask] / _DBL_EPSILON).max())
    assert worst <= bound, (label, "kappa ulps", worst)


def _libxc_correlation(rho, sigma, tau, zeta):
    """``(eps_c^SCAN, eps_c^base)`` of a set of total-density rows: libxc's
    polarized ``MGGA_C_SCAN`` on the proportional spin split and the model's
    own polarized PW92 baseline at the same split."""
    share_a, share_b = 0.5 * (1.0 + zeta), 0.5 * (1.0 - zeta)
    grad = np.sqrt(np.clip(sigma, 0.0, None))
    rows = np.stack([
        _mgga_row(rho * share_a, (grad * share_a) ** 2, tau * share_a),
        _mgga_row(rho * share_b, (grad * share_b) ** 2, tau * share_b)])
    eps = np.asarray(dft.libxc.eval_xc("MGGA_C_SCAN", rows, spin=1,
                                       deriv=0)[0])
    base = np.asarray(pw92c_polarized_scalar(jnp.asarray(rho * share_a),
                                             jnp.asarray(rho * share_b)))
    return eps, base


# ---------------------------------------------------------------------------
# V2: the correlation potential
# ---------------------------------------------------------------------------

def _ec_density(rho, sigma, zeta, alpha):
    """``rho eps_c`` of the anchored model at ``gated = 0``, parametrized by
    the four quantities the correlation ROW carries: the total density, its
    gradient invariant, the spin polarization and the raw indicator."""
    rho_a = rho * 0.5 * (1.0 + zeta)
    rho_b = rho * 0.5 * (1.0 - zeta)
    return (rho * pw92c_polarized_scalar(rho_a, rho_b)
            * parents.scan_fc(rho, sigma, zeta, alpha))


def test_scan_correlation_potential_terms_match_libxc_deriv_one():
    """V2, SCAN correlation: ``jax.grad`` of ``rho eps_c`` with respect to
    ``rho``, ``sigma``, ``zeta`` and the indicator, against libxc's polarized
    ``deriv=1``.

    libxc's derivatives are with respect to ``(rho_a, rho_b, sigma_aa,
    sigma_ab, sigma_bb, tau_a, tau_b)``, so they are chained onto the row's
    four variables through the proportional split the value comparison uses
    (``test_parents._spin_split``): the shares are functions of ``zeta``
    alone, and ``tau = alpha tau_unif(rho) + sigma / (8 rho)`` carries the
    indicator into both ``vtau`` terms. Nothing of ``scan_fc`` enters the
    reference, so the case is an oracle and not a restatement.

    Measured, on the mesh below (6048 points, ``zeta`` extended to the
    production clip ``+-(1 - 1e-6)``), worst relative deviation where libxc's
    own derivative is above 1e-10:

    * where ``|F_c| > 1e-5`` (5832 of the points): ``rho`` 1.29e-13,
      ``sigma`` 3.45e-13, ``zeta`` 4.47e-9, ``alpha`` 1.42e-12;
    * over the whole mesh: ``rho`` 2.71e-9 and ``sigma`` 7.56e-9, both reached
      on the ``alpha = 0`` rows at ``|zeta| -> 1`` where ``F_c`` itself has
      fallen to 2e-6 and the two evaluations sit on the Fermi-hole cap.

    ``zeta`` is the loose one at every ``|F_c|`` because the derivative of the
    correlation with respect to the polarization is the one quantity the
    proportional split does not hold fixed: it moves the two spin densities,
    both diagonal gradient invariants, the off-diagonal one and both kinetic
    energy densities at once, and libxc's seven partial derivatives are summed
    against those seven analytic responses.

    Bounds asserted: 1e-12 / 3e-12 / 3e-8 / 1e-11 on the restricted set and
    5e-8 over the whole mesh -- between 5 and 9 times the measurement, and
    four orders under the design's stated 1e-8 for the derivative comparison
    on the well-conditioned part.

    RED against: evaluating each branch of ``parents._scan_switch`` on the
    bare ``1 - alpha`` instead of on an argument held inside its own domain
    (dropping both the ``jnp.minimum`` / ``jnp.maximum`` clamp and the inner
    ``where`` guard). Every value is unchanged and the derivative becomes NaN
    at ``alpha = 1``, which propagates through the unselected branch of the
    outer ``where`` to the whole mesh.
    """
    zeta_grid = _ZETA_GRID + (_ZETA_CLIP, -_ZETA_CLIP)
    s_grid = np.linspace(0.05, 6.0, 12)
    mesh = np.meshgrid(_RS_GRID, s_grid, np.asarray(_ALPHA_GRID),
                       np.asarray(zeta_grid), indexing="ij")
    rho = _rho_of_rs(mesh[0].ravel())
    sigma = _sigma_of_s(rho, mesh[1].ravel())
    alpha = mesh[2].ravel()
    zeta = mesh[3].ravel()
    tau_unif = _TAU_UNIF * rho ** (5.0 / 3.0)
    tau = _scan_tau(rho, sigma, alpha)
    share_a, share_b = 0.5 * (1.0 + zeta), 0.5 * (1.0 - zeta)

    out = dft.libxc.eval_xc("MGGA_C_SCAN",
                            _spin_split(rho, sigma, zeta, tau), spin=1,
                            deriv=1)
    vrho = np.asarray(out[1][0])
    vsigma = np.asarray(out[1][1])
    vtau = np.asarray(out[1][3])
    v_tau_total = vtau[:, 0] * share_a + vtau[:, 1] * share_b
    want = {
        "rho": (vrho[:, 0] * share_a + vrho[:, 1] * share_b
                + v_tau_total * ((5.0 / 3.0) * alpha * tau_unif / rho
                                 - sigma / (8.0 * rho ** 2))),
        "sigma": (vsigma[:, 0] * share_a ** 2
                  + vsigma[:, 1] * share_a * share_b
                  + vsigma[:, 2] * share_b ** 2
                  + v_tau_total / (8.0 * rho)),
        "zeta": (vrho[:, 0] * (rho / 2.0) - vrho[:, 1] * (rho / 2.0)
                 + vsigma[:, 0] * sigma * share_a
                 + vsigma[:, 1] * sigma * (share_b - share_a) / 2.0
                 - vsigma[:, 2] * sigma * share_b
                 + vtau[:, 0] * (tau / 2.0) - vtau[:, 1] * (tau / 2.0)),
        "alpha": v_tau_total * tau_unif,
    }
    f_c = np.asarray(_libxc_fc("MGGA_C_SCAN", rho, sigma, zeta, tau))
    args = (jnp.asarray(rho), jnp.asarray(sigma), jnp.asarray(zeta),
            jnp.asarray(alpha))
    bounds = {"rho": 1e-12, "sigma": 3e-12, "zeta": 3e-8, "alpha": 1e-11}
    for index, name in enumerate(("rho", "sigma", "zeta", "alpha")):
        got = np.asarray(jax.jit(jax.vmap(
            jax.grad(_ec_density, argnums=index)))(*args))
        reference = want[name]
        alive = np.abs(reference) > 1e-10
        assert int(alive.sum()) > 1000, (name, int(alive.sum()))
        rel = _rel(got[alive], reference[alive])
        assert float(rel.max()) <= 5e-8, (name, "whole mesh", float(rel.max()))
        big = alive & (np.abs(f_c) > 1e-5)
        worst = float(_rel(got[big], reference[big]).max())
        assert worst <= bounds[name], (name, worst, bounds[name])


# ---------------------------------------------------------------------------
# The conventions the indicator reaches the parent on
# ---------------------------------------------------------------------------

def test_scan_correlation_divides_the_row_indicator_by_the_spin_factor():
    """The correlation row's indicator is ``compute_alpha(rho, sigma, tau)``
    of the TOTAL density, which carries no ``d_s(zeta)``; SCAN's does.

    ``metagga.compute_alpha`` divides by ``tau_unif(rho)`` of the total
    density, while SCAN's ``alpha`` divides by the POLARIZED uniform-gas
    kinetic energy density, ``d_s(zeta) tau_unif(rho)`` (libxc's
    ``scan_alpha`` divides by ``t_total(zeta, 1, 1) = 2^(-2/3) d_s``). The two
    agree at ``zeta = 0`` and nowhere else, so the conversion is invisible on
    every closed shell and wrong on every open one -- which is why it is
    stated here at ``zeta = 0.9``, where ``d_s = 1.468107``.

    Measured on the ``(rs, s, alpha)`` mesh at ``zeta = 0.9``, over the rows
    with ``|F_c| > 1e-5``: with the division, worst relative deviation from
    libxc 2.65e-14; feeding the same function an indicator already multiplied
    by ``d_s`` -- so that its own division cancels and the row's value is used
    raw -- 0.494, a factor 1.9e13 worse. The bounds are 3e-13 (11 times the
    measurement) and 0.1 (a fifth of it), a gap of twelve orders.

    RED against: removing the ``/ d_s`` from ``parents.scan_fc``.
    """
    zeta = 0.9
    factor = _d_s(zeta)
    assert abs(factor - 1.468107) < 1e-6, factor
    worst_converted = worst_raw = 0.0
    for rs in _RS_GRID:
        rho = _rho_of_rs(rs)
        for s in _S_GRID:
            sigma = _sigma_of_s(rho, s)
            for alpha in _ALPHA_GRID:
                tau = _scan_tau(rho, sigma, alpha)
                want = float(_libxc_fc("MGGA_C_SCAN", rho, sigma, zeta, tau)[0])
                if abs(want) <= 1e-5:
                    continue
                converted = float(parents.scan_fc(
                    jnp.asarray(rho), jnp.asarray(sigma), jnp.asarray(zeta),
                    jnp.asarray(alpha)))
                raw = float(parents.scan_fc(
                    jnp.asarray(rho), jnp.asarray(sigma), jnp.asarray(zeta),
                    jnp.asarray(alpha * factor)))
                worst_converted = max(worst_converted,
                                      float(_rel(converted, want)))
                worst_raw = max(worst_raw, float(_rel(raw, want)))
    assert worst_converted <= 3e-13, worst_converted
    assert worst_raw > 0.1, worst_raw


def test_the_fermi_hole_cap_evaluates_a_negative_indicator_at_libxcs_value():
    """libxc caps ``sigma`` at ``8 rho tau`` before evaluating, which is the
    von Weizsacker bound ``alpha >= 0``; ``scan_fx`` reproduces the cap, so a
    row whose raw indicator has gone negative returns libxc's number.

    Such rows are real: on a one-orbital spin channel ``tau = tau_W``
    identically and the raw indicator is the rounding residue of that
    cancellation, which can land either side of zero (``metagga.py``, the
    smoothing width's anchors). The parent must not extrapolate SCAN below
    ``alpha = 0`` there -- libxc does not.

    Measured over the ``(rs, s)`` mesh at ``alpha`` in ``{-0.5, -0.1,
    -1e-3}``: with the cap, worst relative deviation from libxc 1.09e-15; the
    same expression WITHOUT it (``parents._scan_fx_core`` at the row's own
    ``(p, alpha)``, i.e. SCAN continued analytically to negative alpha)
    2.36e-2, seven orders of magnitude worse and far above anything the
    parent is held to. Bounds 1e-14 (9 times the measurement) and 1e-3.

    RED against: dropping ``hi=8.0 * rho_f * tau_f`` from the ``sigma`` clip
    in ``parents.scan_fx``.
    """
    k_f_squared = (3.0 * np.pi ** 2) ** (2.0 / 3.0)
    worst_capped = worst_uncapped = 0.0
    n = 0
    for rs in _RS_GRID:
        rho = _rho_of_rs(rs)
        for s in (0.5, 1.0, 2.0, 6.0):
            sigma = _sigma_of_s(rho, s)
            for alpha in (-0.5, -0.1, -1e-3):
                want = float(_libxc_fx("MGGA_X_SCAN", rho, sigma,
                                       _scan_tau(rho, sigma, alpha))[0])
                capped = float(parents.scan_fx(
                    jnp.asarray(rho), jnp.asarray(sigma), jnp.asarray(alpha)))
                p = sigma / (4.0 * k_f_squared * rho ** (2.0 / 3.0) * rho ** 2)
                uncapped = float(parents._scan_fx_core(jnp.asarray(p),
                                                       jnp.asarray(alpha)))
                worst_capped = max(worst_capped, float(_rel(capped, want)))
                worst_uncapped = max(worst_uncapped,
                                     float(_rel(uncapped, want)))
                n += 1
    assert n == len(_RS_GRID) * 4 * 3, n
    assert worst_capped <= 1e-14, worst_capped
    assert worst_uncapped > 1e-3, worst_uncapped


# ---------------------------------------------------------------------------
# The switching function at its branch point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("term", ["x", "c"])
def test_the_switching_function_carries_libxcs_dbl_epsilon_cutoffs(term):
    """``f(alpha)`` is zero on an interval AROUND ``alpha = 1``, not at the
    single point, and the interval is libxc's.

    Both branches are exponentials of ``1 / (1 - alpha)``: the left one falls
    under DBL_EPSILON at ``alpha = ln(1/eps) / (ln(1/eps) + c1)`` and the
    right one at ``(ln(d/eps) + c2) / ln(d/eps)``, and libxc returns zero
    beyond each. Written that way both branches evaluate on an argument held
    inside their own domain, so the value and every derivative are finite at
    ``alpha = 1`` where the bare form is ``0/0``.

    Measured: the bands are ``[0.9818308872650647, 1.0220636308242315]``
    (exchange, ``c1 = 0.667``, ``c2 = 0.8``, ``d = 1.24``) and
    ``[0.9825535370424727, 1.0420321379212383]`` (correlation, ``c1 = 0.64``,
    ``c2 = 1.5``, ``d = 0.7``); ``f`` is exactly 0.0 strictly inside and
    ``+-2.2204460492e-16`` -- DBL_EPSILON itself -- at the two edges, where
    the branch is still evaluated. ``scan_fx`` and ``scan_fc`` at nine
    indicator values across each band agree with libxc to 4.27e-16 and
    8.21e-16 relative (bound 1e-14, 12 times the measurement); ``f`` and
    ``df/dalpha`` are 0.0 and -0.0 at ``alpha = 1`` and one ulp either side,
    and ``d(rho eps_x)/d alpha / tau_unif`` at ``alpha = 1`` reproduces
    libxc's ``vtau`` there (9.8605735e-4 at rs = 1, s = 1) rather than a NaN.

    RED against: evaluating each branch of ``parents._scan_switch`` on the
    bare ``1 - alpha`` -- the values survive, and ``df/dalpha`` at
    ``alpha = 1`` becomes NaN in both terms.
    """
    if term == "x":
        c1, c2, d = parents.SCAN_C1X, parents.SCAN_C2X, parents.SCAN_DX
        expected = (0.9818308872650647, 1.0220636308242315)
    else:
        c1, c2, d = parents.SCAN_C1C, parents.SCAN_C2C, parents.SCAN_DC
        expected = (0.9825535370424727, 1.0420321379212383)
    left, right = _dead_band(c1, c2, d)
    assert left == pytest.approx(expected[0], abs=1e-15), left
    assert right == pytest.approx(expected[1], abs=1e-15), right

    def switch(alpha):
        return parents._scan_switch(alpha, c1, c2, d)

    # The edges are the last points at which the branch is evaluated; just
    # inside them libxc's cutoff has taken hold and the value is exactly zero.
    for edge in (left, right):
        assert abs(float(switch(jnp.asarray(edge)))) == pytest.approx(
            _DBL_EPSILON, rel=1e-9), edge
    for alpha in np.linspace(left + 1e-9, right - 1e-9, 9):
        assert float(switch(jnp.asarray(alpha))) == 0.0, alpha

    # ... and both the switch and the enhancement factor are differentiable
    # through the branch point, where the naive form is 0/0.
    derivative = jax.grad(lambda a: switch(a))
    for alpha in (1.0 - 1e-12, 1.0, 1.0 + 1e-12):
        assert float(switch(jnp.asarray(alpha))) == 0.0, alpha
        assert float(derivative(jnp.asarray(alpha))) == 0.0, alpha

    # The parent itself, against libxc, across the whole band.
    rho = _rho_of_rs(1.0)
    worst = 0.0
    for s in (0.5, 1.0, 3.0):
        sigma = _sigma_of_s(rho, s)
        for alpha in np.linspace(left, right, 9):
            tau = _scan_tau(rho, sigma, alpha)
            if term == "x":
                got = float(parents.scan_fx(
                    jnp.asarray(rho), jnp.asarray(sigma), jnp.asarray(alpha)))
                want = float(_libxc_fx("MGGA_X_SCAN", rho, sigma, tau)[0])
            else:
                got = float(parents.scan_fc(
                    jnp.asarray(rho), jnp.asarray(sigma), jnp.asarray(0.0),
                    jnp.asarray(alpha)))
                want = float(_libxc_fc("MGGA_C_SCAN", rho, sigma, 0.0, tau)[0])
            worst = max(worst, float(_rel(got, want)))
    assert worst <= 1e-14, (term, worst)

    if term != "x":
        return
    # The indicator derivative at the branch point is libxc's vtau, finite.
    sigma = _sigma_of_s(rho, 1.0)
    tau_unif = _TAU_UNIF * rho ** (5.0 / 3.0)

    def ex_density(alpha):
        return (rho * lda_x(jnp.asarray(rho))
                * parents.scan_fx(jnp.asarray(rho), jnp.asarray(sigma), alpha))

    for alpha in (1.0 - 1e-12, 1.0, 1.0 + 1e-12):
        got = float(jax.grad(ex_density)(jnp.asarray(alpha))) / tau_unif
        reference = float(np.asarray(dft.libxc.eval_xc(
            "MGGA_X_SCAN", _mgga_row(rho, sigma, _scan_tau(rho, sigma, alpha)),
            spin=0, deriv=1)[1][3])[0])
        assert np.isfinite(got), alpha
        assert float(_rel(got, reference)) <= 1e-12, (alpha, got, reference)


# ---------------------------------------------------------------------------
# Full polarization
# ---------------------------------------------------------------------------

def test_scan_fc_at_exactly_full_polarization():
    """``zeta = +-1`` exactly: the empty spin channel enters at libxc's
    density threshold rather than at zero, and ``scan_fc`` reproduces what
    that does.

    libxc floors each spin density at 1e-15 and then forms ``1 - zeta`` from
    the floored pair, so at ``rho_b = 0`` the quantity it actually evaluates
    is ``2e-15 / rho`` -- a number quantized in units of an ulp of 1 around a
    true value of order 1e-16 -- and the empty channel's ``tau_W = sigma /
    (8 rho_b)`` moves with it. The parent is held to what libxc does, not to
    the analytic ``zeta = 1`` limit; the production path never reaches here
    anyway (``oneshot.uks_zeta`` clips at ``+-(1 - 1e-6)``).

    Measured on the ``(rs, s, alpha)`` mesh at ``zeta = +-1`` (1638 of the
    2016 rows carry ``|F_c| > 1e-5``): worst relative deviation 3.78e-12,
    reached at ``rs = 0.267``, ``s = 0.5``, ``alpha = 10``. On the remaining
    rows -- ``alpha = 0``, where ``G_c(+-1) = 0`` drives ``F_c`` itself to
    1e-16 and the relative measure of a quantity at round-off is unbounded --
    the agreement is 4.27e-15 ABSOLUTE. Bounds 3e-11 and 4e-14, both 8 to 9
    times the measurement.

    RED against: removing the ``_floor_as_libxc`` from the empty channel in
    ``parents.scan_fc`` (``rho_b = rho * omhalf`` unfloored), which evaluates
    the analytic ``zeta = 1`` limit instead of libxc's floored one and reads
    1.48e-8 against the 3e-11 bound.
    """
    worst_relative = worst_absolute = 0.0
    n_relative = n_absolute = 0
    for rs in _RS_GRID:
        rho = _rho_of_rs(rs)
        for s in _S_GRID:
            sigma = _sigma_of_s(rho, s)
            for alpha in _ALPHA_GRID:
                tau = _scan_tau(rho, sigma, alpha)
                for zeta in (1.0, -1.0):
                    got = float(parents.scan_fc(
                        jnp.asarray(rho), jnp.asarray(sigma),
                        jnp.asarray(zeta), jnp.asarray(alpha)))
                    want = float(_libxc_fc("MGGA_C_SCAN", rho, sigma, zeta,
                                           tau)[0])
                    assert np.isfinite(got), (rs, s, alpha, zeta)
                    if abs(want) > 1e-5:
                        n_relative += 1
                        worst_relative = max(worst_relative,
                                             float(_rel(got, want)))
                    else:
                        n_absolute += 1
                        worst_absolute = max(worst_absolute, abs(got - want))
    assert n_relative > 1000 and n_absolute > 100, (n_relative, n_absolute)
    assert worst_relative <= 3e-11, worst_relative
    assert worst_absolute <= 4e-14, worst_absolute


# ---------------------------------------------------------------------------
# V1 on the stored molecular grids
# ---------------------------------------------------------------------------

def test_scan_on_the_stored_rows_of_a_closed_shell():
    """H2O at sto-3g, grid level 1: the two doubled channels (identical, the
    shell being closed) and the total density, binned by the conditioning of
    the indicator.

    The rows the model integrates carry a stored indicator column, and the
    parent is evaluated at the raw value ``networks._raw_indicator`` recovers
    from it. Where ``tau_W`` exceeds ``tau - tau_W`` by ``kappa``, libxc's own
    indicator -- which it recomputes from ``tau`` -- carries a rounding
    residue of ``kappa`` ulps, so the bins are the statement and a single
    bound would be the tail's.

    Measured over three independent evaluations of the record (the reference
    SCF is threaded, so its density matrix moves in its last ulps and the
    ill-conditioned rows move with it), worst relative deviation per bin,
    8413 rows below the ceiling:

    * ``F_x``: 5.6e-15 (kappa < 1e2, 7405 rows), 2.8e-14 (< 1e3, 486),
      3.4e-13 (< 1e4, 236), 1.4e-12 (< 1e5, 130), 2.8e-11 (beyond, 156);
    * ``F_c``: 4.0e-14, 1.2e-13, 1.9e-12, 7.8e-12, 2.3e-10 on the same bins;
    * the energy densities ``rho eps_x`` and ``rho eps_c`` against libxc's own:
      8.0e-13 and 5.3e-15 Ha/bohr^3 ABSOLUTE.

    Across the rows with ``kappa >= 1`` the disagreement is at most 6.2
    ``kappa`` ulps, which is the conditioning bound rather than a property of
    this system; it is asserted at 50.

    Bounds are 5 to 10 times each measurement.

    RED against: handing the parent the stored column itself rather than the
    raw indicator ``networks._raw_indicator`` recovers from it -- the column
    is the SMOOTH positive part of width 1e-5, which shifts the indicator by
    ``width^2 / (4 alpha)``, and the well-conditioned exchange bin then reads
    2.6e-9 against its 5e-14 bound.
    """
    md = _h2o_record()
    x_bounds = (5e-14, 2e-13, 3e-12, 1e-11, 2e-10)
    c_bounds = (3e-13, 1e-12, 1e-11, 5e-11, 2e-9)
    for channel in (0, 1):
        rho, sigma, tau, alpha, at_ceiling = _exchange_rows(md, channel)
        free = ~at_ceiling
        assert int(free.sum()) > 8000, int(free.sum())
        eps = np.asarray(dft.libxc.eval_xc(
            "MGGA_X_SCAN", _mgga_row(rho[free], sigma[free], tau[free]),
            spin=0, deriv=0)[0])
        want = eps / np.asarray(lda_x(jnp.asarray(rho[free])))
        got = np.asarray(jax.vmap(parents.scan_fx)(
            jnp.asarray(rho[free]), jnp.asarray(sigma[free]),
            jnp.asarray(alpha[free])))
        kappa = _kappa(rho[free], sigma[free], alpha[free])
        _assert_binned(got, want, kappa, x_bounds, ("H2O", "x", channel))
        _assert_ulp_bound(got, want, kappa, ("H2O", "x", channel))
        energy = rho[free] * np.asarray(lda_x(jnp.asarray(rho[free]))) * got
        assert float(np.abs(energy - rho[free] * eps).max()) <= 5e-12

    rho, sigma, tau, zeta, alpha, at_ceiling = _correlation_rows(md)
    free = ~at_ceiling
    assert float(np.max(np.abs(zeta))) < 1e-10, "closed shell"
    eps, base = _libxc_correlation(rho[free], sigma[free], tau[free],
                                   zeta[free])
    want = eps / base
    got = np.asarray(jax.vmap(parents.scan_fc)(
        jnp.asarray(rho[free]), jnp.asarray(sigma[free]),
        jnp.asarray(zeta[free]), jnp.asarray(alpha[free])))
    kappa = _kappa(rho[free], sigma[free], alpha[free])
    _assert_binned(got, want, kappa, c_bounds, ("H2O", "c"))
    _assert_ulp_bound(got, want, kappa, ("H2O", "c"))
    assert float(np.abs(rho[free] * base * got - rho[free] * eps).max()) <= 5e-14


def test_scan_on_the_stored_rows_of_an_open_shell():
    """The OH radical at def2-svp, grid level 1: an open shell, so the two
    doubled channels differ and the correlation rows carry a real zeta.

    Same construction as the closed-shell case. Measured over three
    independent evaluations of the record, worst relative deviation per bin,
    on 6283 to 6328 rows below the ceiling per exchange channel and 6318
    correlation rows:

    * ``F_x``, worst over both channels: 1.1e-14 (kappa < 1e2), 4.9e-14
      (< 1e3), 3.7e-13 (< 1e4), 3.3e-12 (< 1e5), 1.3e-10 (beyond, where kappa
      itself reaches 5.6e12);
    * ``F_c``: 3.7e-14, 4.2e-13, 2.0e-12, 2.1e-11, 1.5e-9;
    * ``rho eps_x`` 1.14e-12 and ``rho eps_c`` 1.16e-14 Ha/bohr^3 ABSOLUTE,
      and ``F_c`` itself to 2.9e-11 absolute.

    The disagreement is at most 6.4 ``kappa`` ulps where ``kappa >= 1``,
    asserted at 50. Bounds are 5 to 10 times each measurement.

    RED against: the same substitution of the stored column for the raw
    indicator that the closed-shell case names; on this record the
    well-conditioned exchange bin reads 1.3e-9 against its 1e-13 bound.
    """
    md = _oh_record()
    x_bounds = (1e-13, 4e-13, 3e-12, 3e-11, 1e-9)
    c_bounds = (3e-13, 4e-12, 1e-11, 2e-10, 1e-8)
    for channel in (0, 1):
        rho, sigma, tau, alpha, at_ceiling = _exchange_rows(md, channel)
        free = ~at_ceiling
        assert int(free.sum()) > 6000, int(free.sum())
        eps = np.asarray(dft.libxc.eval_xc(
            "MGGA_X_SCAN", _mgga_row(rho[free], sigma[free], tau[free]),
            spin=0, deriv=0)[0])
        want = eps / np.asarray(lda_x(jnp.asarray(rho[free])))
        got = np.asarray(jax.vmap(parents.scan_fx)(
            jnp.asarray(rho[free]), jnp.asarray(sigma[free]),
            jnp.asarray(alpha[free])))
        kappa = _kappa(rho[free], sigma[free], alpha[free])
        _assert_binned(got, want, kappa, x_bounds, ("OH", "x", channel))
        _assert_ulp_bound(got, want, kappa, ("OH", "x", channel))
        energy = rho[free] * np.asarray(lda_x(jnp.asarray(rho[free]))) * got
        assert float(np.abs(energy - rho[free] * eps).max()) <= 1e-11

    rho, sigma, tau, zeta, alpha, at_ceiling = _correlation_rows(md)
    free = ~at_ceiling
    # The radical's rows carry a real polarization (measured worst |zeta|
    # 0.8486 above the tail threshold), which is what separates this record
    # from the closed shell's identically zero one.
    assert float(np.max(np.abs(zeta))) > 0.5, float(np.max(np.abs(zeta)))
    eps, base = _libxc_correlation(rho[free], sigma[free], tau[free],
                                   zeta[free])
    want = eps / base
    got = np.asarray(jax.vmap(parents.scan_fc)(
        jnp.asarray(rho[free]), jnp.asarray(sigma[free]),
        jnp.asarray(zeta[free]), jnp.asarray(alpha[free])))
    kappa = _kappa(rho[free], sigma[free], alpha[free])
    _assert_binned(got, want, kappa, c_bounds, ("OH", "c"))
    _assert_ulp_bound(got, want, kappa, ("OH", "c"))
    assert float(np.abs(got - want).max()) <= 2e-10
    assert float(np.abs(rho[free] * base * got - rho[free] * eps).max()) <= 1e-13


def test_the_rows_at_the_indicator_ceiling_sit_at_the_saturation_floor():
    """The rows whose stored column is pinned at ``metagga._ALPHA_MAX = 100``
    are evaluated at 100, and the difference from libxc at their TRUE tau is
    the ceiling's saturation, not a disagreement of the functional.

    ``_raw_indicator`` returns the ceiling unchanged there rather than
    inverting the smoothing, because above it the recovery is meaningless: the
    column carries no information about how far past 100 the row went. SCAN's
    switching function has saturated by then -- ``f(100)`` is
    ``-1.24 exp(-0.00808)`` for exchange -- so what remains is the residual
    slope, and the spec (Section 3.1) records it as a stated floor of the
    oracle rather than as agreement.

    Measured: 526 rows on H2O and 514 to 546 per channel on OH, of the 8939
    and 6797 to 6872 above the tail threshold. Worst relative departure from
    libxc at the true tau, 1.1698e-3 in ``F_x`` on every channel of both
    systems (the asymptote of ``F_x(100) - F_x(inf)``), 3.4287e-3 in ``F_c``
    on H2O and 5.074e-3 on OH; the smallest departure on the same rows is
    6.2e-6, so the whole set sits between 1e-6 and 1e-2 -- seven orders above
    the below-ceiling floor the two cases above assert, which is what makes
    the separate statement necessary. Bounds 5e-3 (``F_x``) and 2e-2
    (``F_c``), four times each measurement, with a lower bound of 1e-4 so that
    a ceiling that stopped saturating (or rows that stopped reaching it) is
    caught too.

    RED against: making ``networks._raw_indicator`` invert the smoothing above
    the ceiling as well -- the departure then follows the column's own pinned
    value and this bound is exceeded on the exchange rows.
    """
    for name, md in (("H2O", _h2o_record()), ("OH", _oh_record())):
        for channel in (0, 1):
            rho, sigma, tau, alpha, at_ceiling = _exchange_rows(md, channel)
            assert int(at_ceiling.sum()) > 400, (name, channel,
                                                 int(at_ceiling.sum()))
            assert np.all(alpha[at_ceiling] == _ALPHA_MAX), (name, channel)
            eps = np.asarray(dft.libxc.eval_xc(
                "MGGA_X_SCAN",
                _mgga_row(rho[at_ceiling], sigma[at_ceiling],
                          tau[at_ceiling]), spin=0, deriv=0)[0])
            want = eps / np.asarray(lda_x(jnp.asarray(rho[at_ceiling])))
            got = np.asarray(jax.vmap(parents.scan_fx)(
                jnp.asarray(rho[at_ceiling]), jnp.asarray(sigma[at_ceiling]),
                jnp.asarray(alpha[at_ceiling])))
            worst = float(_rel(got, want).max())
            assert 1e-4 < worst <= 5e-3, (name, "x", channel, worst)

        rho, sigma, tau, zeta, alpha, at_ceiling = _correlation_rows(md)
        assert int(at_ceiling.sum()) > 400, (name, "c", int(at_ceiling.sum()))
        eps, base = _libxc_correlation(rho[at_ceiling], sigma[at_ceiling],
                                       tau[at_ceiling], zeta[at_ceiling])
        got = np.asarray(jax.vmap(parents.scan_fc)(
            jnp.asarray(rho[at_ceiling]), jnp.asarray(sigma[at_ceiling]),
            jnp.asarray(zeta[at_ceiling]), jnp.asarray(alpha[at_ceiling])))
        worst = float(_rel(got, eps / base).max())
        assert 1e-4 < worst <= 2e-2, (name, "c", worst)


# ---------------------------------------------------------------------------
# The constants
# ---------------------------------------------------------------------------

def test_the_scan_constants_are_the_libxc_values_not_the_rounded_paper_ones():
    """Four of SCAN's constants differ between libxc 7.0.0 and the printed
    paper by more than the 1e-12 the parent is held to, and the parent carries
    libxc's, since libxc is the oracle.

    Each is substituted with the paper's rounded value in turn and the change
    in ``F_c`` measured over the ``(rs, s, alpha, zeta)`` mesh (4680 points),
    so the case states what "correcting" the constant back to the paper would
    cost:

    * ``chi_infinity`` 0.12802585262625815 against the paper's 0.128026:
      3.28e-7, reached at s = 6, where ``g_infinity`` weighs most;
    * the coefficient of ``G_c(zeta)``, 2.363 against 2.3631 (libxc's
      ``scan_G_cnst``): 3.79e-5, at zeta = 0.9, ``G_c`` being flat at zeta = 0;
    * ``beta(rs)``'s prefactor, PBE's 0.06672455060314922 against the
      supplement's 0.066725: 3.40e-6;
    * ``gamma``, PBE's ``(1 - ln 2) / pi^2 = 0.0310906908696549`` against the
      supplement's 0.031091: 9.11e-6.

    Each is asserted to move ``F_c`` by more than a third of the measurement
    and less than three times it, so both a constant silently replaced by the
    paper's value and a mesh that stopped reaching the region where the
    constant matters turn the case red.

    RED against: setting any of the four to the paper's rounded value in
    ``parents`` -- which is exactly what the case drives, in reverse.
    """
    mesh = np.meshgrid(_RS_GRID, _S_GRID, np.asarray(_ALPHA_GRID),
                       np.asarray(_ZETA_GRID), indexing="ij")
    rho = _rho_of_rs(mesh[0].ravel())
    args = (jnp.asarray(rho), jnp.asarray(_sigma_of_s(rho, mesh[1].ravel())),
            jnp.asarray(mesh[3].ravel()), jnp.asarray(mesh[2].ravel()))
    reference = np.asarray(parents.scan_fc(*args))
    assert reference.shape[0] == 4680, reference.shape

    for name, paper, measured in (
            ("SCAN_CHI_INF", 0.128026, 3.276e-7),
            ("SCAN_G_CNST", 2.3631, 3.787e-5),
            ("SCAN_BETA_A", 0.066725, 3.399e-6),
            ("SCAN_GAMMA", 0.031091, 9.107e-6)):
        live = getattr(parents, name)
        assert live != paper, name
        setattr(parents, name, paper)
        try:
            moved = np.asarray(parents.scan_fc(*args))
        finally:
            setattr(parents, name, live)
        worst = float(np.max(_rel(moved, reference)))
        assert measured / 3.0 <= worst <= 3.0 * measured, (name, worst,
                                                           measured)


# ---------------------------------------------------------------------------
# The dispatch surface
# ---------------------------------------------------------------------------

def test_parent_fx_and_parent_fc_refuse_scan_without_the_indicator():
    """``parent_fx("scan", rho, sigma)`` with no indicator is a ValueError
    naming the column it needs, not a silent evaluation at some default.

    The dispatch takes ``alpha=None`` because PBE ignores it, so a caller that
    reaches the SCAN branch through a GGA-shaped call site would otherwise
    anchor a meta-GGA network to whatever ``None`` became. The message names
    ``metagga_alpha_index`` and ``networks._raw_indicator``, the two places the
    value comes from.

    RED against: removing ``parents._require_indicator`` from either
    dispatcher -- ``jnp.asarray(None)`` inside ``scan_fx`` then raises jax's
    own conversion error, whose message names neither the column the value
    comes from nor the caller that failed to supply it.
    """
    for call in (lambda: parents.parent_fx("scan", 1.0, 1.0),
                 lambda: parents.parent_fc("scan", 1.0, 1.0, 0.0)):
        with pytest.raises(ValueError, match="metagga_alpha_index"):
            call()
    # ... while PBE ignores it, which is why the argument is optional at all.
    assert float(parents.parent_fx("pbe", jnp.asarray(1.0), jnp.asarray(1.0))) \
        == float(parents.pbe_fx(jnp.asarray(1.0), jnp.asarray(1.0)))
    # ... and the SCAN branch is reached once the indicator is supplied.
    assert float(parents.parent_fx("scan", jnp.asarray(1.0), jnp.asarray(1.0),
                                   jnp.asarray(0.5))) == \
        float(parents.scan_fx(jnp.asarray(1.0), jnp.asarray(1.0),
                              jnp.asarray(0.5)))


def test_scan_is_elementwise_in_every_shape_the_model_hands_it():
    """``scan_fx`` and ``scan_fc`` are elementwise: a scalar, a ``(N,)`` batch
    and a ``(N, 1)`` column give the same numbers, BITWISE, and so does
    ``jax.vmap`` of the scalar form.

    The three shapes are all live. The model's ``eval_ex`` / ``eval_ec`` pass
    ``(N,)`` grid columns; the networks' ``_core`` passes scalars under
    ``jax.vmap``; the packed-row helpers carry ``(N, 1)``. A parent that
    reduced over the batch -- or that broadcast a scalar differently from a
    length-one axis -- would be wrong in one of them and right in the others.

    Measured on three rows spanning the domain: all four evaluations
    bit-identical (``np.array_equal``), for both functions.

    RED against: replacing the ``jnp.where`` branch selection in
    ``parents._scan_switch`` with a Python ``if alpha <= 1.0``, which raises
    on any array argument and so passes only the scalar form.
    """
    rho = np.asarray([0.1, 1.0, 3.0])
    sigma = np.asarray([0.05, 2.0, 40.0])
    alpha = np.asarray([0.2, 1.0, 7.0])
    zeta = np.asarray([0.0, 0.4, -0.7])

    for fn, columns in ((parents.scan_fx, (rho, sigma, alpha)),
                        (parents.scan_fc, (rho, sigma, zeta, alpha))):
        flat = np.asarray(fn(*[jnp.asarray(c) for c in columns]))
        column = np.asarray(fn(*[jnp.asarray(c[:, None]) for c in columns]))
        scalars = np.asarray([float(fn(*[jnp.asarray(c[i]) for c in columns]))
                              for i in range(rho.shape[0])])
        mapped = np.asarray(jax.vmap(fn)(*[jnp.asarray(c) for c in columns]))
        assert flat.shape == (3,) and column.shape == (3, 1)
        np.testing.assert_array_equal(flat, column.ravel())
        np.testing.assert_array_equal(flat, scalars)
        np.testing.assert_array_equal(flat, mapped)
