"""V1 and V2 of the parent-anchor verification: the JAX
parent enhancement factors of :mod:`xcquinox.pipeline.parents`
against libxc, and their first derivatives against libxc's ``deriv=1``.

Conventions the oracles are built on, stated once because every comparison
below depends on them:

* EXCHANGE is posed on the DOUBLED spin channel (Oliver and Perdew, Phys. Rev.
  A 20, 397 (1979)). A row carrying ``(rho_sigma, sigma_sigma_sigma)`` reaches
  the network as ``(2 rho_sigma, 4 sigma_sigma_sigma)``, and the model's
  ``F_x`` at that row is the SPIN-UNPOLARIZED enhancement factor evaluated
  there. ``parents.pbe_fx(rho, sigma)`` therefore takes the already-doubled
  pair, and the libxc reference is ``eval_xc("GGA_X_PBE", ..., spin=0)``
  divided by ``lda_x`` at the same density -- the construction
  ``pretrain_data_gen.spin_channel_exchange_rows`` uses for its targets. A
  closed shell has ``rho_a = rho_b``, so its doubled channel IS the total
  density and the two rows coincide.
* CORRELATION is posed on the TOTAL density with the row's spin polarization,
  and the enhancement factor is stated relative to the MODEL's own baseline,
  ``utils.pw92c_polarized_scalar`` (``models.AlecGGAModel._ec_baseline`` under
  ``use_polarized_correlation``), so that ``rho eps_c^base F_c`` is the
  parent's correlation energy density. The libxc reference is therefore
  ``eval_xc("GGA_C_PBE", ..., spin=1) / pw92c_polarized_scalar``.

Environment the numbers quoted in the docstrings were measured on: pyscf
2.11.0, libxc 7.0.0, ``jax_enable_x64``, CPU.

The SCAN cases are written against ``MGGA_X_SCAN`` / ``MGGA_C_SCAN`` on the
same two conventions, at the RAW iso-orbital indicator the row's stored column
encodes. What they do not reach -- the correlation potential, the
``d_s(zeta)`` conversion of the row's indicator, libxc's Fermi-hole cap, the
switching function's two DBL_EPSILON cutoffs, full polarization, the stored
molecular grids and the constants at which libxc departs from the printed
paper -- is in ``test_parents_scan``.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from pyscf import dft

from xcquinox.pipeline import parents
from xcquinox.pipeline.config import MoleculeSpec
from xcquinox.pipeline.data import precompute_fixed_density_data
from xcquinox.utils import lda_x, pw92c_polarized_scalar


# ---------------------------------------------------------------------------
# libxc reference rows
# ---------------------------------------------------------------------------

#: The (rs, s) mesh of Section 3.1: rs from 0.02 (a core density) to 20 (a
#: valence tail), s from the uniform gas to 6, the largest reduced gradient the
#: stored molecular rows reach at grid level 1.
_RS_GRID = np.geomspace(0.02, 20.0, 9)
_S_GRID = np.linspace(0.0, 6.0, 13)

#: The polarizations V1 is stated at. ``1 - 1e-6`` is the production clip
#: (``oneshot._ZETA_BOUNDARY_EPS``), which is as close to full polarization as
#: any row the model integrates ever comes; exactly +-1 is a separate case
#: because libxc floors the empty spin channel at its density threshold rather
#: than evaluating at it, which ``pbe_fc`` reproduces.
_ZETA_GRID = (0.0, 0.3, -0.3, 0.9, -0.9)
_ZETA_CLIP = 1.0 - 1e-6


def _rho_of_rs(rs):
    return 3.0 / (4.0 * np.pi * rs ** 3)


def _sigma_of_s(rho, s):
    """The gradient invariant giving reduced gradient ``s`` at density ``rho``."""
    k_F = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    return (s * 2.0 * k_F * rho) ** 2


def _gga_row(rho, sigma):
    """libxc GGA input row ``(4, N)`` encoding a KNOWN ``sigma``.

    The gradient magnitude goes in the x component and the other two stay
    zero, so ``sigma = dx^2 + dy^2 + dz^2`` is the requested value; only the
    invariant enters a GGA, so the encoding is exact.
    """
    r = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    out = np.zeros((4, r.shape[0]), dtype=np.float64)
    out[0] = r
    out[1] = np.sqrt(np.clip(np.atleast_1d(np.asarray(sigma, dtype=np.float64)),
                             0.0, None))
    return out


def _mgga_row(rho, sigma, tau):
    """libxc meta-GGA row ``(6, N)``: the GGA row, an unread Laplacian slot,
    and the positive kinetic-energy density."""
    r = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    out = np.zeros((6, r.shape[0]), dtype=np.float64)
    out[:4] = _gga_row(r, sigma)
    out[5] = np.atleast_1d(np.asarray(tau, dtype=np.float64))
    return out


def _libxc_fx(functional, rho, sigma, tau=None):
    """libxc's spin-unpolarized ``F_x`` on the doubled channel: the energy
    density per electron divided by the uniform-gas value at the same
    density, which is the factor the model multiplies ``lda_x`` by."""
    rho = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    row = (_gga_row(rho, sigma) if tau is None
           else _mgga_row(rho, sigma, tau))
    eps = np.asarray(dft.libxc.eval_xc(functional, row, spin=0, deriv=0)[0])
    return eps / np.asarray(lda_x(jnp.asarray(rho)))


def _spin_split(rho, sigma, zeta, tau=None):
    """The two libxc spin rows of a total density at polarization ``zeta``.

    The spin gradients are taken parallel and proportional to the spin
    densities, so ``sigma_aa + 2 sigma_ab + sigma_bb`` reproduces the requested
    total invariant exactly; PBE correlation is a functional of the total
    gradient alone, so the split is exact rather than a choice. The same
    proportional split carries the kinetic-energy density for a meta-GGA.
    """
    rho = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    sigma = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
    zeta = np.atleast_1d(np.asarray(zeta, dtype=np.float64)) * np.ones_like(rho)
    share_a, share_b = 0.5 * (1.0 + zeta), 0.5 * (1.0 - zeta)
    grad = np.sqrt(np.clip(sigma, 0.0, None))
    if tau is None:
        row_a = _gga_row(rho * share_a, (grad * share_a) ** 2)
        row_b = _gga_row(rho * share_b, (grad * share_b) ** 2)
    else:
        tau = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        row_a = _mgga_row(rho * share_a, (grad * share_a) ** 2, tau * share_a)
        row_b = _mgga_row(rho * share_b, (grad * share_b) ** 2, tau * share_b)
    return np.stack([row_a, row_b])


def _libxc_fc(functional, rho, sigma, zeta, tau=None):
    """libxc's spin-polarized correlation of the parent, divided by the
    model's own PW92 baseline: the factor ``F_c`` the anchored network has to
    return so that ``rho eps_c^base F_c`` is the parent's energy density."""
    rho = np.atleast_1d(np.asarray(rho, dtype=np.float64))
    zeta_arr = np.atleast_1d(np.asarray(zeta, dtype=np.float64)) * np.ones_like(rho)
    rows = _spin_split(rho, sigma, zeta_arr, tau)
    eps = np.asarray(dft.libxc.eval_xc(functional, rows, spin=1, deriv=0)[0])
    base = np.asarray(pw92c_polarized_scalar(
        jnp.asarray(rho * 0.5 * (1.0 + zeta_arr)),
        jnp.asarray(rho * 0.5 * (1.0 - zeta_arr))))
    return eps / base


def _rel(got, want):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    return np.abs(got - want) / np.maximum(np.abs(want), 1e-300)


# ---------------------------------------------------------------------------
# Stored molecular rows
# ---------------------------------------------------------------------------

#: Records are read and never written, so one per (system, basis, grid) is
#: shared by the cases below; the package conftest wipes the library's own memo
#: cache between tests, which this dict is deliberately not subject to.
_RECORDS = {}

#: The model's tail threshold (``models._NN_TAIL_THRESHOLD``). Below it the
#: model masks ``F`` to 1 and the parent is not compared pointwise at all; the
#: energy those rows carry is 1.2e-12 Ha on the N atom.
_RHO_FLOOR = 1e-10


def _record(name, atom, basis, spin, composition, grid_level=1):
    key = (name, basis, spin, grid_level)
    if key not in _RECORDS:
        _RECORDS[key] = precompute_fixed_density_data(
            MoleculeSpec(name=name, atom=atom, basis=basis, charge=0,
                         spin=spin, atom_composition=composition,
                         grid_level=grid_level))
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


def _spin_densities(md):
    """``(rho_a, sigma_aa, rho_b, sigma_bb)`` of a record, contracted from its
    stored AO derivative table and density matrix exactly as
    ``data.precompute_fixed_density_data`` builds its per-channel blocks."""
    ao = np.asarray(md["ao_grid_deriv"])
    dm = np.asarray(md["dm_pbe"])
    out = []
    for s in (0, 1):
        d = dm[s] if dm.ndim == 3 else 0.5 * dm
        r = np.einsum("gi,ij,gj->g", ao[0], d, ao[0])
        grad = [2.0 * np.einsum("gi,ij,gj->g", ao[k], d, ao[0])
                for k in (1, 2, 3)]
        out.append((r, grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2))
    return out[0][0], out[0][1], out[1][0], out[1][1]


# ---------------------------------------------------------------------------
# V1: the parents against libxc, pointwise
# ---------------------------------------------------------------------------

def test_pbe_constants_are_the_libxc_values_not_the_rounded_paper_ones():
    """The constants of PBE eq. 14 and eqs. 3-8, as libxc carries them.

    ``mu = beta pi^2 / 3`` with ``beta = 0.06672455060314922``; the rounded
    ``mu = 0.21951`` that ``pbe_anchor._fx_pbe_analytic`` carries puts ``F_x``
    2.6e-6 relative off libxc at ``s = 1``, which is 4.6 orders above the
    tolerance V1 is stated at (measured in the design review).
    ``gamma = (1 - ln 2) / pi^2``.
    """
    assert parents.PBE_KAPPA == 0.804
    assert float(parents.PBE_BETA) == 0.06672455060314922
    assert float(parents.PBE_MU) == pytest.approx(
        0.06672455060314922 * np.pi ** 2 / 3.0, rel=1e-15, abs=0.0)
    assert float(parents.PBE_GAMMA) == pytest.approx(
        (1.0 - np.log(2.0)) / np.pi ** 2, rel=1e-15, abs=0.0)


def test_pbe_fx_matches_libxc_on_the_reduced_gradient_grid():
    """V1, exchange: ``pbe_fx`` is libxc's ``GGA_X_PBE`` enhancement factor on
    the doubled-channel convention over the model's whole domain.

    Grid: rs from 0.02 to 20 (9 points, geometric), s from 0 to 6 (13 points).
    Worst relative deviation measured 4.69e-16 -- the rounding of one
    evaluation against the other -- against the 1e-12 the design states for
    the PBE parent, a margin of three orders.
    """
    worst = 0.0
    for rs in _RS_GRID:
        rho = _rho_of_rs(rs)
        for s in _S_GRID:
            sigma = _sigma_of_s(rho, s)
            got = float(parents.pbe_fx(jnp.asarray(rho), jnp.asarray(sigma)))
            want = float(_libxc_fx("GGA_X_PBE", rho, sigma)[0])
            worst = max(worst, float(_rel(got, want)))
    assert worst <= 1e-12, worst


def test_pbe_fx_is_bounded_by_one_and_the_lieb_oxford_ceiling():
    """The range the anchored transform's pre-image assumes: ``F_x^PBE`` runs
    from the uniform-gas value 1 at ``s = 0`` up to ``1 + kappa = 1.804``,
    never reaching it. Measured maximum 1.7739 at ``s = 6`` over the grid
    above, so the pre-image ``ln[(limit - 1) F / (limit - F)]`` is finite on
    every row a PBE-anchored network sees.
    """
    values = []
    for rs in _RS_GRID:
        rho = _rho_of_rs(rs)
        for s in _S_GRID:
            value = float(parents.pbe_fx(
                jnp.asarray(rho), jnp.asarray(_sigma_of_s(rho, s))))
            if s == 0.0:
                assert value == 1.0, (rs, value)
            values.append(value)
    values = np.asarray(values)
    assert float(values.min()) >= 1.0
    assert float(values.max()) < 1.804


def test_pbe_fc_matches_libxc_on_the_rs_s_zeta_grid():
    """V1, correlation: ``pbe_fc`` against libxc's ``GGA_C_PBE`` divided by
    the model's own polarized PW92 baseline.

    Grid: the (rs, s) mesh above at zeta in {0, +-0.3, +-0.9}. Worst relative
    deviation measured 8.71e-13, at (rs = 0.02, s = 6, zeta = 0.9) where
    ``eps_c`` has fallen to 1.6e-5 Ha and the ``H`` term is a cancellation
    against it; 1e-12 is the design's stated bound for the PBE parent.

    The baseline is load-bearing and is the reason the bound is attainable at
    all: libxc's ``GGA_C_PBE`` reduces at ``sigma = 0`` to ``LDA_C_PW_MOD``,
    not to ``LDA_C_PW`` (measured 0.0 against 6.3e-6 relative at rs = 0.02),
    while the repository's ``pw92c_polarized_scalar`` is the ``LDA_C_PW``
    parameter set. The parent's NUMERATOR must therefore be libxc's PBE, and
    only the DENOMINATOR is the repository's PW92; building the numerator on
    ``pw92c_polarized_scalar`` instead puts ``eps_c^PBE`` up to 1.65e-4
    relative off libxc, which is what this case refuses.
    """
    worst = 0.0
    for rs in _RS_GRID:
        rho = _rho_of_rs(rs)
        for s in _S_GRID:
            sigma = _sigma_of_s(rho, s)
            for zeta in _ZETA_GRID:
                got = float(parents.pbe_fc(jnp.asarray(rho), jnp.asarray(sigma),
                                           jnp.asarray(zeta)))
                want = float(_libxc_fc("GGA_C_PBE", rho, sigma, zeta)[0])
                worst = max(worst, float(_rel(got, want)))
    assert worst <= 1e-12, worst


def test_pbe_fc_at_the_production_zeta_clip():
    """V1, correlation, at the most polarized row the model ever integrates.

    ``oneshot.uks_zeta`` clips the polarization to ``1 - 1e-6``
    (``_ZETA_BOUNDARY_EPS``), so this is the boundary case of the energy path.
    Worst relative deviation measured 1.13e-12 over the (rs, s) mesh, at
    (rs = 0.02, s = 6); 1e-11 clears it by an order. The interior of the
    polarization range is held to 1e-12 by the case above.
    """
    worst = 0.0
    for rs in _RS_GRID:
        rho = _rho_of_rs(rs)
        for s in _S_GRID:
            sigma = _sigma_of_s(rho, s)
            for zeta in (_ZETA_CLIP, -_ZETA_CLIP):
                got = float(parents.pbe_fc(jnp.asarray(rho), jnp.asarray(sigma),
                                           jnp.asarray(zeta)))
                want = float(_libxc_fc("GGA_C_PBE", rho, sigma, zeta)[0])
                worst = max(worst, float(_rel(got, want)))
    assert worst <= 1e-11, worst


#: Ceiling on the relative deviation of ``pbe_fc`` from libxc at ``zeta = +-1``
#: on the (rs, s) mesh. Measured worst with the shipped spin floor: 1.121e-10;
#: with the floor removed, 2.725e-5. The ceiling leaves one order over the
#: former and stands more than four orders under the latter, so it is the floor
#: that has to be in place for it to be met -- which
#: :func:`test_the_libxc_spin_floor_is_load_bearing` asserts from the other
#: side.
_FULL_POLARIZATION_CEILING = 1e-9


def _worst_fc_at_full_polarization():
    """Worst relative deviation of ``pbe_fc`` from libxc over the (rs, s) mesh
    at ``zeta = +-1``, with ``F_c`` checked finite, positive and no greater
    than 1 on every row.

    Shared by the two cases below so that the floored and the unfloored
    figures are taken on identical rows. The ``1e-4`` of slack on the upper
    bound is there because the two PW92 parameter sets do not cancel in the
    ratio: ``H >= 0`` makes ``eps_c^PBE`` no more negative than its own LDA
    limit, so the exact ratio is at most 1, but the numerator's limit is
    ``LDA_C_PW_MOD`` and the denominator is the repository's ``LDA_C_PW``,
    which stand 1.49e-5 apart at rs = 0.02 and zeta = 1 (measured maximum of
    ``F_c`` here 1.0000149, floored and unfloored alike).
    """
    worst = 0.0
    for rs in _RS_GRID:
        rho = _rho_of_rs(rs)
        for s in _S_GRID:
            sigma = _sigma_of_s(rho, s)
            for zeta in (1.0, -1.0):
                got = float(parents.pbe_fc(jnp.asarray(rho), jnp.asarray(sigma),
                                           jnp.asarray(zeta)))
                assert np.isfinite(got), (rs, s, zeta, got)
                assert 0.0 < got <= 1.0 + 1e-4, (rs, s, zeta, got)
                want = float(_libxc_fc("GGA_C_PBE", rho, sigma, zeta)[0])
                worst = max(worst, float(_rel(got, want)))
    return worst


def test_the_libxc_spin_floor_is_load_bearing(monkeypatch):
    """V1, correlation: removing libxc's 1e-12 spin floor breaks the agreement
    at zeta = +-1, which is what makes the case above a test OF the floor.

    ``parents.LIBXC_DENS_THRESHOLD`` is the one regularization ``pbe_fc``
    carries, and it is not a smoothing of the parent's own form: it puts the
    evaluation at the point the oracle evaluates. With it set to zero the
    empty channel stays empty while libxc's is still floored, so the two are
    at different densities, and the worst relative deviation on the same mesh
    rises from 1.121e-10 to 2.725e-5 -- at (rs = 20, s = 6), monotone in both
    rs and s, and more than four orders above
    ``_FULL_POLARIZATION_CEILING``. The floor's consequence for the energy is
    recorded in the module docstring of :mod:`xcquinox.pipeline.parents`: 1.3e-9
    Ha in the H atom's integrated ``E_c``.

    ``F_c`` stays finite, positive and bounded by 1 without the floor (the
    shared helper asserts it on every row), so the failure the floor prevents
    is one of AGREEMENT with the oracle, not one of definedness.
    """
    monkeypatch.setattr(parents, "LIBXC_DENS_THRESHOLD", 0.0)
    worst = _worst_fc_at_full_polarization()
    assert worst > _FULL_POLARIZATION_CEILING, worst
    assert worst > 1e-5, worst


def test_parent_fx_and_parent_fc_dispatch_on_the_parent_name():
    """The dispatchers return the named parent's factor and refuse an unknown
    name, so a typo cannot silently select PBE."""
    rho, sigma, zeta = 0.1, 0.05, 0.25
    assert float(parents.parent_fx("pbe", jnp.asarray(rho), jnp.asarray(sigma))) \
        == float(parents.pbe_fx(jnp.asarray(rho), jnp.asarray(sigma)))
    assert float(parents.parent_fc("pbe", jnp.asarray(rho), jnp.asarray(sigma),
                                   jnp.asarray(zeta))) \
        == float(parents.pbe_fc(jnp.asarray(rho), jnp.asarray(sigma),
                                jnp.asarray(zeta)))
    with pytest.raises((ValueError, KeyError)):
        parents.parent_fx("b3lyp", jnp.asarray(rho), jnp.asarray(sigma))
    with pytest.raises((ValueError, KeyError)):
        parents.parent_fc("b3lyp", jnp.asarray(rho), jnp.asarray(sigma),
                          jnp.asarray(zeta))


def test_parent_for_arch_resolves_by_rung():
    """The parent is the rung's: PBE for a GGA architecture, SCAN for a
    meta-GGA one, the same resolution ``cluster.fidelity.resolve_parent``
    makes, so the anchor and the certificate cannot disagree about which
    functional a network is measured against."""
    from xcquinox.pipeline.cluster import fidelity as fid
    from xcquinox.pipeline.config import get_architecture

    gga = get_architecture("deep_3x16")
    mgga = get_architecture("deep_mgga_3x16")
    assert parents.parent_for_arch(gga) == "pbe"
    assert parents.parent_for_arch(mgga) == "scan"
    assert parents.parent_for_arch(gga) == fid.resolve_parent(gga.name)
    assert parents.parent_for_arch(mgga) == fid.resolve_parent(mgga.name)


# ---------------------------------------------------------------------------
# V2: first derivatives against libxc deriv=1
# ---------------------------------------------------------------------------

def _ex_density(rho, sigma):
    """``rho eps_x`` of the anchored model at ``gated = 0``: the uniform-gas
    exchange energy density times the parent's enhancement factor."""
    return rho * lda_x(rho) * parents.pbe_fx(rho, sigma)


def _ec_density(rho_a, rho_b, sigma):
    """``rho eps_c`` of the anchored model at ``gated = 0``, parametrized by
    the two spin densities and the TOTAL gradient invariant so that each
    argument is independent, as libxc's polarized ``vrho`` requires."""
    rho = rho_a + rho_b
    zeta = (rho_a - rho_b) / rho
    return rho * pw92c_polarized_scalar(rho_a, rho_b) * parents.pbe_fc(
        rho, sigma, zeta)


def test_pbe_exchange_potential_terms_match_libxc_deriv_one():
    """V2, exchange: ``jax.grad`` of ``rho eps_x`` against libxc's ``vrho``
    and ``vsigma``.

    The SCF potential is the autodiff of the model's energy density, so the
    parent's derivatives are inside ``V_xc`` by construction and are oracled
    here rather than only through the energy. Grid as in V1 with ``s`` from
    0.05 (``vsigma`` vanishes identically at ``s = 0``). Worst relative
    deviation measured 7.23e-16 (``vrho``) and 1.08e-15 (``vsigma``) against
    the design's 1e-8, which is where the derivative comparison is stated
    because a derivative loses digits the value does not.
    """
    d_rho = jax.grad(_ex_density, argnums=0)
    d_sigma = jax.grad(_ex_density, argnums=1)
    worst_rho = worst_sigma = 0.0
    for rs in _RS_GRID:
        rho = _rho_of_rs(rs)
        for s in np.linspace(0.05, 6.0, 12):
            sigma = _sigma_of_s(rho, s)
            out = dft.libxc.eval_xc("GGA_X_PBE", _gga_row(rho, sigma),
                                    spin=0, deriv=1)
            vrho = float(np.asarray(out[1][0])[0])
            vsigma = float(np.asarray(out[1][1])[0])
            got_rho = float(d_rho(jnp.asarray(rho), jnp.asarray(sigma)))
            got_sigma = float(d_sigma(jnp.asarray(rho), jnp.asarray(sigma)))
            if abs(vrho) > 1e-10:
                worst_rho = max(worst_rho, float(_rel(got_rho, vrho)))
            if abs(vsigma) > 1e-10:
                worst_sigma = max(worst_sigma, float(_rel(got_sigma, vsigma)))
    assert worst_rho <= 1e-8, worst_rho
    assert worst_sigma <= 1e-8, worst_sigma


def test_pbe_correlation_potential_terms_match_libxc_deriv_one():
    """V2, correlation: ``jax.grad`` of ``rho eps_c`` with respect to each
    spin density and to the total gradient invariant, against libxc's
    polarized ``vrho`` and ``vsigma``.

    libxc is called with ``sigma_aa = sigma_ab = sigma_bb = sigma / 4``, whose
    total invariant ``sigma_aa + 2 sigma_ab + sigma_bb`` is the requested
    ``sigma`` and which leaves the two spin densities free, so ``vrho`` is the
    derivative at fixed total gradient that the JAX form takes. PBE
    correlation depends on the gradient only through that total, which the
    test also states directly: libxc's ``vsigma_ab`` is exactly twice its
    ``vsigma_aa``. Worst relative deviation measured 7.46e-13 (``vrho``) and
    6.69e-13 (``vsigma``) against the design's 1e-8.
    """
    d_a = jax.grad(_ec_density, argnums=0)
    d_b = jax.grad(_ec_density, argnums=1)
    d_sigma = jax.grad(_ec_density, argnums=2)
    worst_rho = worst_sigma = 0.0
    for rs in _RS_GRID:
        rho = _rho_of_rs(rs)
        for s in np.linspace(0.05, 6.0, 12):
            sigma = _sigma_of_s(rho, s)
            for zeta in _ZETA_GRID:
                rho_a, rho_b = rho * (1 + zeta) / 2, rho * (1 - zeta) / 2
                rows = np.stack([_gga_row(rho_a, sigma / 4.0),
                                 _gga_row(rho_b, sigma / 4.0)])
                out = dft.libxc.eval_xc("GGA_C_PBE", rows, spin=1, deriv=1)
                vrho = np.asarray(out[1][0])[0]
                vsigma = np.asarray(out[1][1])[0]
                assert abs(vsigma[1] - 2.0 * vsigma[0]) <= \
                    1e-10 * abs(vsigma[1]) + 1e-30, vsigma
                args = (jnp.asarray(rho_a), jnp.asarray(rho_b),
                        jnp.asarray(sigma))
                for got, want in ((float(d_a(*args)), float(vrho[0])),
                                  (float(d_b(*args)), float(vrho[1]))):
                    if abs(want) > 1e-10:
                        worst_rho = max(worst_rho, float(_rel(got, want)))
                if abs(vsigma[0]) > 1e-10:
                    worst_sigma = max(
                        worst_sigma,
                        float(_rel(float(d_sigma(*args)), float(vsigma[0]))))
    assert worst_rho <= 1e-8, worst_rho
    assert worst_sigma <= 1e-8, worst_sigma


# ---------------------------------------------------------------------------
# The pre-image of the live bounded map
# ---------------------------------------------------------------------------

#: The three limits registered architectures build ``_AlecLOB`` at: 1.804 for
#: the GGA exchange nets (1 + kappa_PBE), 1.174 for the meta-GGA exchange nets
#: (Dick and Fernandez-Serra, PRB 104 L161109, the SCAN ceiling h0x) and 2.0
#: for the correlation nets (a non-negativity squash, not a bound).
_LIMITS = (1.804, 1.174, 2.0)


@pytest.mark.parametrize("limit", _LIMITS)
def test_lob_preimage_round_trips_through_the_live_transform(limit):
    """``1 + L(lob_preimage(F, limit)) == F``, the identity the anchor rests
    on: at ``gated = 0`` the anchored forward returns the parent exactly.

    ``L`` is the live ``networks._AlecLOB(limit)``, so this is the map the
    network applies and not a restatement of it.

    The statement is ABSOLUTE, because the map is written as
    ``limit sigmoid(x - ln(limit - 1)) - 1`` and the forward adds the 1 back:
    for a parent well under 1 the intermediate is a number of order 1, so the
    round trip carries an absolute error of order the ulp of 1 whatever the
    parent's own size. Measured worst absolute deviation 2.2e-16 over 200
    values of ``F`` spanning (0.01, limit - 0.01) at each of the three
    registered limits -- one ulp of 1 -- and worst RELATIVE deviation
    2.03e-15, reached at the smallest ``F``, which is the same 2.2e-16 divided
    by 0.01. 1e-15 absolute is a factor 4.5 above the floor; the relative form
    is stated separately for parents above 0.1, where the cancellation is mild
    (measured 8.9e-16).

    The design's "to round-off" is this: the transform is not bitwise either,
    ``_AlecLOB(1.174)`` returning -1.11e-16 rather than 0.0 at argument 0.
    """
    from xcquinox.pipeline.networks import _AlecLOB
    lob = _AlecLOB(limit=limit)
    worst_abs = worst_rel = 0.0
    for target in np.linspace(0.01, limit - 0.01, 200):
        z = float(parents.lob_preimage(jnp.asarray(target), limit))
        assert np.isfinite(z)
        got = float(1.0 + lob(jnp.asarray(z)))
        worst_abs = max(worst_abs, abs(got - target))
        if target >= 0.1:
            worst_rel = max(worst_rel, float(_rel(got, target)))
    assert worst_abs <= 1e-15, worst_abs
    assert worst_rel <= 1e-15, worst_rel


# ---------------------------------------------------------------------------
# SCAN: the same oracles
# ---------------------------------------------------------------------------

def _scan_tau(rho, sigma, alpha):
    """The kinetic-energy density an iso-orbital indicator stands for:
    ``tau = alpha tau_unif + tau_W`` with ``tau_W = sigma / (8 rho)`` and
    ``tau_unif = (3/10) (3 pi^2)^(2/3) rho^(5/3)`` (Sun, Ruzsinszky and
    Perdew, Phys. Rev. Lett. 115, 036402 (2015), eq. 2)."""
    tau_unif = (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
    return alpha * tau_unif + sigma / (8.0 * rho)


#: Indicator values the SCAN cases are stated at: the single-orbital limit,
#: the uniform gas, the two sides of the switching function's branch point and
#: the descriptor's ceiling (``metagga._ALPHA_MAX``).
_ALPHA_GRID = (0.0, 0.5, 1.0, 2.0, 10.0, 100.0)


def test_scan_fx_matches_libxc_on_the_reduced_gradient_indicator_grid():
    """V1, SCAN exchange: ``scan_fx`` against libxc's ``MGGA_X_SCAN`` on the
    doubled-channel convention, at the RAW iso-orbital indicator.

    The row carries the smoothed, ceiling-capped indicator
    (``metagga.compute_alpha``); the parent is evaluated at the raw value
    recovered from it, which is what the pretraining targets are posed at
    (``pretrain_data_gen.spin_channel_exchange_rows`` calls libxc at the row's
    true tau). The design's stated departures -- 1.8e-3 relative in ``F_x`` on
    rows above ``_ALPHA_MAX = 100``, and 4.2e-7 from the smoothing floor at
    ``alpha = 0`` -- lie outside this grid, whose largest indicator IS the
    ceiling.

    Measured 2.627e-15 relative over the 702 points of the grid, reached at
    rs = 20 and s = 6 (the valence tail at the largest reduced gradient the
    stored rows reach). Bound 2e-14, 7.6 times the measurement, rather than
    the 1e-12 the design states for the value comparison: the implementation
    is at round-off here and the looser figure would not see a departure two
    orders above it.
    """
    worst = 0.0
    for rs in _RS_GRID:
        rho = _rho_of_rs(rs)
        for s in _S_GRID:
            sigma = _sigma_of_s(rho, s)
            for alpha in _ALPHA_GRID:
                tau = _scan_tau(rho, sigma, alpha)
                got = float(parents.scan_fx(jnp.asarray(rho), jnp.asarray(sigma),
                                            jnp.asarray(alpha)))
                want = float(_libxc_fx("MGGA_X_SCAN", rho, sigma, tau)[0])
                worst = max(worst, float(_rel(got, want)))
    assert worst <= 2e-14, worst


def test_scan_fc_matches_libxc_on_the_rs_s_zeta_indicator_grid():
    """V1, SCAN correlation: ``scan_fc`` against libxc's ``MGGA_C_SCAN``
    divided by the model's polarized PW92 baseline, at zeta in
    {0, +-0.3, +-0.9} and the indicator grid above. The kinetic-energy density
    is split between the spin rows in proportion to the spin densities, the
    split libxc's SCAN correlation is exact under (its per-spin quantities
    enter only through the Fermi-hole bound ``sigma_ss <= 8 rho_s tau_s``,
    which a proportional split satisfies whenever ``alpha >= 0``).

    ``scan_fc`` reads the row's indicator on the repository's convention --
    ``tau_unif`` of the TOTAL density, no ``d_s(zeta)`` -- and divides by
    ``d_s`` itself; without that division the same grid is 0.494 off at
    zeta = 0.9 (``test_parents_scan``, which also carries zeta = +-1 and the
    correlation potential).

    Measured 2.650e-14 relative over the 3510 points of the grid, reached at
    rs = 1.50, s = 6, alpha = 0.5, zeta = 0.9; no point on it has ``|F_c|``
    below 1e-5, so the relative measure is meaningful everywhere. Bound
    2e-13, 7.5 times the measurement, rather than the design's 1e-12.
    """
    worst = 0.0
    for rs in _RS_GRID:
        rho = _rho_of_rs(rs)
        for s in _S_GRID:
            sigma = _sigma_of_s(rho, s)
            for alpha in _ALPHA_GRID:
                tau = _scan_tau(rho, sigma, alpha)
                for zeta in _ZETA_GRID:
                    got = float(parents.scan_fc(
                        jnp.asarray(rho), jnp.asarray(sigma), jnp.asarray(zeta),
                        jnp.asarray(alpha)))
                    want = float(_libxc_fc("MGGA_C_SCAN", rho, sigma, zeta,
                                           tau)[0])
                    worst = max(worst, float(_rel(got, want)))
    assert worst <= 2e-13, worst


def test_scan_exchange_potential_terms_match_libxc_deriv_one():
    """V2, SCAN exchange: ``jax.grad`` of ``rho eps_x`` with respect to the
    density, the gradient invariant and the indicator, against libxc's
    ``deriv=1`` (``vrho``, ``vsigma``, ``vtau`` carried through
    ``dtau/dalpha = tau_unif``).

    The indicator derivative is taken through the SMOOTHED quantity the
    network differentiates, so the potential inherits the model's own
    regularization rather than the raw indicator's response (whose
    ``d alpha / d sigma`` reaches 2.2e31 in the tail, the divergence
    ``metagga._ALPHA_MAX`` bounds).

    Measured 2.090e-13 relative, worst over the three terms at the 648 points
    of the grid, wherever libxc's own derivative is above 1e-10. The s axis
    starts at 0.05 because at s = 0 with alpha <= 0 the comparison reads
    1.6e-2 relative: the chain-corrected combination vsigma + vtau/(8 rho)
    cancels to an indicator-scale residual (got -2.107e-10, want -2.142e-10,
    an absolute gap of 3.5e-12 against an energy density of 6.8e-7) through
    the 1 - f_x cancellation at an indicator of 1e-13, which libxc's own
    arithmetic carries identically; libxc's vsigma itself does not vanish at
    s = 0 (-1.3e-4 at rs = 0.27 to -3.9e3 at rs = 20), and at s = 0 with
    alpha >= 1e-3 the agreement is 8.4e-13.
    Bound 2e-12, 9.6 times the measurement, rather than the design's 1e-8:
    the grid includes ``alpha = 1``, where both branches of the switching
    function are cut off at DBL_EPSILON and a form that is finite in value but
    not under differentiation would return NaN rather than a large number.
    """
    def ex_density(rho, sigma, alpha):
        return rho * lda_x(rho) * parents.scan_fx(rho, sigma, alpha)

    d_rho = jax.grad(ex_density, argnums=0)
    d_sigma = jax.grad(ex_density, argnums=1)
    d_alpha = jax.grad(ex_density, argnums=2)
    worst = 0.0
    for rs in _RS_GRID:
        rho = _rho_of_rs(rs)
        tau_unif = (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
        for s in np.linspace(0.05, 6.0, 12):
            sigma = _sigma_of_s(rho, s)
            for alpha in _ALPHA_GRID:
                tau = _scan_tau(rho, sigma, alpha)
                out = dft.libxc.eval_xc("MGGA_X_SCAN", _mgga_row(rho, sigma, tau),
                                        spin=0, deriv=1)
                vrho = float(np.asarray(out[1][0])[0])
                vsigma = float(np.asarray(out[1][1])[0])
                vtau = float(np.asarray(out[1][3])[0])
                args = (jnp.asarray(rho), jnp.asarray(sigma), jnp.asarray(alpha))
                # d/d rho and d/d sigma at FIXED alpha differ from libxc's at
                # fixed tau by the chain rule through tau(rho, sigma, alpha).
                got_rho = float(d_rho(*args)) - vtau * float(
                    (5.0 / 3.0) * alpha * tau_unif / rho - sigma / (8.0 * rho ** 2))
                got_sigma = float(d_sigma(*args)) - vtau / (8.0 * rho)
                got_alpha = float(d_alpha(*args)) / tau_unif
                for got, want in ((got_rho, vrho), (got_sigma, vsigma),
                                  (got_alpha, vtau)):
                    if abs(want) > 1e-10:
                        worst = max(worst, float(_rel(got, want)))
    assert worst <= 2e-12, worst
