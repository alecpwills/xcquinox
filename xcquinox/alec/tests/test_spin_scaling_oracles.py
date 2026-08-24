"""Oracles O1-O4 of the pretraining-fidelity program, Section 3.1.

The module is consumed by ``cluster.workflow_matrix.oracle_selector``: every
architecture-carrying case is parametrized over ``sorted(alec.ARCHITECTURES)``
with a node id of the form ``test_x[<arch>]`` or ``test_x[<arch>-<species>]``,
and no test FUNCTION name carries a registry name (``-k`` matches the function
name as well as the id, so such a function would answer to every selector
naming that architecture).

O1 replaces the network with the parent functional's own enhancement factors,
taken from libxc, and asks whether the library's UKS code path reproduces
libxc's spin-polarized evaluation on open-shell atoms. Any discrepancy is a
defect in the assembly rather than in a fit, because there is no fit left. The
parent is the one each architecture pretrains to (PBE for the GGA rungs, SCAN
for the meta-GGA rung), evaluated with that architecture's own descriptor
blocks in place. The exchange comparison is posed at the kinetic-energy
density the channel block's indicator encodes, so the 1e-10 Ha it holds is a
statement about the assembly: against libxc at PySCF's own per-spin tau the
meta-GGA rung differs by up to 6.3e-10 Ha (O atom), all of it on the
descriptor's ``_ALPHA_MAX`` ceiling in the density tail, which carries 1.7e-4
of the electron density there.

O2 is the central-difference check of the assembled UKS Fock pair against the
assembled energy on H, Li, N and O with every descriptor active; the probe
itself lives in ``test_solv01_split_xc`` beside the finite-difference harness.

O3 is the closed-shell byte identity against the tree at ae204537e: rho_a =
rho_b makes the three per-channel feature blocks identical, so the exact spin
scaling has no closed-shell content at all. The record, the archived fixture
and the comparison live in ``test_closed_shell_byte_identity``; the case below
is the per-architecture entry point the matrix selects.

O4 is the H atom: one electron in one orbital, so the symmetric doubled density
diag(P_a, P_a) is a two-electron single-orbital system with tau = tau_W and
alpha identically zero, the rung-3.5 block is the doubled orbital's occupancy in
both spin slots, and the exchange energy is exactly half the model's
spin-unpolarized evaluation on that system.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import dft, gto

jax.config.update("jax_enable_x64", True)

import xcquinox.alec as alec
from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.descriptors import (
    DMRung35Descriptor, MetaGGAAlphaDescriptor, assemble_descriptor_features,
    doubled_spin_dm)
from xcquinox.alec.oneshot import split_exc_energy_uks, uks_zeta
from xcquinox.alec.pretrain_data_gen import resolve_parent_density
from xcquinox.alec.solver import make_uks_feature_fns
from xcquinox.alec.tests.parent_adapter import (
    LibxcParentModel, gga_rho_row, mgga_rho_row, tau_from_alpha)
from xcquinox.alec.tests.test_closed_shell_byte_identity import (
    assert_closed_shell_record_matches)
from xcquinox.alec.tests.test_solv01_split_xc import (
    _UKS_FD_SPECIES, _alpha_columns, _assert_uks_fd_consistency,
    _live_model, _md_with_descriptors)

# Open-shell atoms of the pools, in PySCF's 2S spin convention.
_ATOMS = [("H", 1), ("Li", 1), ("N", 3), ("O", 2)]
_ATOM_IDS = [symbol for symbol, _spin in _ATOMS]
_ARCHS = sorted(alec.ARCHITECTURES)

# libxc names of the two parents, keyed by the rung baseline
# ``pretrain_data_gen.resolve_parent_density`` resolves (PBE for the GGA
# rungs, SCAN for the meta-GGA rung).
_PARENT_LIBXC = {"pbe": ("GGA_X_PBE", "GGA_C_PBE"),
                 "scan": ("MGGA_X_SCAN", "MGGA_C_SCAN")}


def _parent_of(arch_name):
    """``(x_functional, c_functional)`` of the parent this architecture
    pretrains to, read the way the pretraining resolves it."""
    return _PARENT_LIBXC[resolve_parent_density(
        alec.get_architecture(arch_name), "auto")]


# ---------------------------------------------------------------------------
# Records. One reference SCF per (species, basis, grid, descriptor set),
# shared across the parametrized cases of this module: the record is read and
# never written, and the descriptor set is what determines its blocks. The
# library's own memo cache is wiped before every test by the package conftest
# so that tests patching PySCF internals stay isolated; nothing in this module
# patches anything.
# ---------------------------------------------------------------------------
_RECORDS = {}


def _descriptor_identity(descriptors):
    return tuple(
        (type(d).__name__,)
        + tuple((f.name, getattr(d, f.name)) for f in dataclasses.fields(d))
        for d in descriptors)


def _precompute(symbol, spin, descriptors, basis="def2-svp", grid_level=1):
    key = (symbol, spin, basis, grid_level, _descriptor_identity(descriptors))
    if key not in _RECORDS:
        keys = tuple(sorted({k for d in descriptors for k in d.required_mol_keys}))
        _RECORDS[key] = precompute_fixed_density_data(
            MoleculeSpec(name=symbol, atom=f"{symbol} 0 0 0", basis=basis,
                         charge=0, spin=spin, atom_composition=((symbol, 1),),
                         grid_level=grid_level),
            required_keys=keys, descriptors=tuple(descriptors))
    return _RECORDS[key]


def _spin_quantities(md, s):
    """(rho_sigma, nabla_rho_sigma (N, 3), sigma_sigma_sigma) for one channel."""
    ao = np.asarray(md["ao_grid_deriv"])
    d = np.asarray(md["dm_pbe"])[s]
    rho = np.einsum("pi,ij,pj->p", ao[0], d, ao[0])
    grad = np.stack([2 * np.einsum("pi,ij,pj->p", ao[k], d, ao[0])
                     for k in (1, 2, 3)], axis=-1)
    return rho, grad, np.sum(grad * grad, axis=1)


def _pyscf_tau(md, s):
    """Kinetic-energy density of one spin channel from PySCF's own
    ``eval_rho`` on the record's AO derivatives: an implementation of tau
    independent of ``metagga.compute_tau_from_dm``.

    The Mole is built from the record's own ``mol_metadata``. PySCF does not
    consult it in the MGGA branch of ``eval_rho`` -- Moles of nao 1, 2, 5, 6,
    9, 14 and 31 (sto-3g, 6-31G and def2-tzvp on each of the four atoms)
    return tau bit-identical to the record's own def2-svp Mole on both
    channels, max|dtau| = 0.0 with no exception raised -- so a fixed basis
    here is inert rather than wrong at the identity this module runs. Reading
    it off the record is what keeps it inert for a record at another basis.
    """
    meta = md["mol_metadata"]
    mol = gto.M(atom=meta["atom"], basis=meta["basis"],
                charge=meta["charge"], spin=meta["spin"], verbose=0)
    ao = np.asarray(md["ao_grid_deriv"])
    dm = np.asarray(md["dm_pbe"])[s]
    return dft.numint.eval_rho(mol, ao, dm, xctype="MGGA", with_lapl=False)[4]


def _positive_mass_weights(md, rho_a, rho_b):
    """Grid weights with quadrature-noise-negative spin densities removed.

    libxc and the adapter each clamp a nonpositive density to zero exchange but
    need not clamp it identically; such points carry no integrand mass, so they
    are dropped from BOTH sides of the comparison rather than absorbed into the
    tolerance. Measured on the four def2-svp/grid-1 records: no negative point
    (the smallest spin density is 2.6e-14 on Li's beta channel; H's beta
    channel is exactly zero everywhere and is kept).
    """
    keep = (rho_a >= 0.0) & (rho_b >= 0.0)
    return np.asarray(md["grid_weights"]) * keep, int((~keep).sum())


class _Ingredients:
    """Per-channel and total ingredients of one record, the way the library's
    UKS energy consumes them, plus the descriptor blocks of one model."""

    def __init__(self, md, descriptors):
        self.md = md
        self.rho_a, self.nabla_a, self.sigma_aa = _spin_quantities(md, 0)
        self.rho_b, self.nabla_b, self.sigma_bb = _spin_quantities(md, 1)
        self.nabla_tot = self.nabla_a + self.nabla_b
        self.sigma_tot = np.sum(self.nabla_tot * self.nabla_tot, axis=1)
        self.w, self.n_dropped = _positive_mass_weights(md, self.rho_a,
                                                        self.rho_b)
        self.f_a = assemble_descriptor_features(descriptors, md, spin_channel=0)
        self.f_b = assemble_descriptor_features(descriptors, md, spin_channel=1)
        self.f_tot = assemble_descriptor_features(descriptors, md)

    def split_energy(self, parent, sigma_a=None, sigma_b=None):
        sa = self.sigma_aa if sigma_a is None else sigma_a
        sb = self.sigma_bb if sigma_b is None else sigma_b
        return float(split_exc_energy_uks(
            parent, jnp.asarray(self.rho_a), jnp.asarray(self.rho_b),
            jnp.asarray(sa), jnp.asarray(sb), jnp.asarray(self.sigma_tot),
            self.f_a, self.f_b, self.f_tot, jnp.asarray(self.w)))

    def integrate(self, eps):
        return float(np.sum(self.w * (self.rho_a + self.rho_b) * eps))

    def channel_tau_from_block(self, s):
        """tau_sigma encoded by the channel block's iso-orbital column:
        the block is alpha(2 rho_s, 4 sigma_ss, 2 tau_s), so inverting at the
        doubled arguments and halving returns tau_s. Inverting rather than
        recontracting the density matrix keeps the descriptor's
        ``[0, _ALPHA_MAX]`` value clip out of the comparison: the oracle asks
        whether the assembly is the parent's own, not whether the clip is
        active in the deep tail."""
        rho, sigma, block = ((self.rho_a, self.sigma_aa, self.f_a) if s == 0
                             else (self.rho_b, self.sigma_bb, self.f_b))
        return 0.5 * tau_from_alpha(2.0 * rho, 4.0 * sigma,
                                    np.asarray(block)[:, self.alpha_column])

    def total_tau_from_block(self):
        """tau of the total density encoded by the total block's column."""
        return tau_from_alpha(self.rho_a + self.rho_b, self.sigma_tot,
                              np.asarray(self.f_tot)[:, self.alpha_column])

    def true_tau(self, s):
        """tau_sigma from PySCF's own ``eval_rho`` on the record's density
        matrix: the ingredient itself, independent of every block."""
        return _pyscf_tau(self.md, s)


def _ingredients(arch_name, symbol, spin):
    model = _live_model(arch_name)
    md = _precompute(symbol, spin, model.descriptors)
    ing = _Ingredients(md, model.descriptors)
    ing.symbol, ing.spin = symbol, spin
    cols = _alpha_columns(model)
    ing.alpha_column = cols[0] if cols else None
    ing.descriptors = model.descriptors
    return ing


def _parent_model(ing, x_functional=None, c_functional=None,
                  use_spin_polarization=True):
    return LibxcParentModel(x_functional=x_functional,
                            c_functional=c_functional,
                            alpha_column=ing.alpha_column,
                            use_spin_polarization=use_spin_polarization,
                            descriptors=ing.descriptors)


def _libxc_x_reference(ing, x_functional):
    """libxc's spin-polarized exchange of the parent on the record, at the
    per-spin ingredients (rho_s, nabla rho_s) and, for a meta-GGA parent, at
    the tau_s the channel blocks encode."""
    if ing.alpha_column is None:
        rows = np.stack([gga_rho_row(ing.rho_a, ing.nabla_a),
                         gga_rho_row(ing.rho_b, ing.nabla_b)])
    else:
        rows = (mgga_rho_row(ing.rho_a, ing.nabla_a,
                             ing.channel_tau_from_block(0)),
                mgga_rho_row(ing.rho_b, ing.nabla_b,
                             ing.channel_tau_from_block(1)))
    eps = np.asarray(dft.libxc.eval_xc(x_functional, rows, spin=1,
                                       deriv=0)[0])
    return ing.integrate(eps)


def _libxc_c_reference(ing, c_functional, tau_a=None, tau_b=None, spin=1):
    """libxc's correlation of the parent on the record: spin-polarized at the
    true per-spin ingredients (``spin=1``), or once on the total density
    (``spin=0``). A meta-GGA parent takes the per-spin kinetic-energy
    densities ``tau_a`` / ``tau_b``."""
    if spin == 0:
        rho_tot = ing.rho_a + ing.rho_b
        if ing.alpha_column is None:
            rows = gga_rho_row(rho_tot, ing.nabla_tot)
        else:
            rows = mgga_rho_row(rho_tot, ing.nabla_tot,
                                ing.total_tau_from_block())
    elif ing.alpha_column is None:
        rows = np.stack([gga_rho_row(ing.rho_a, ing.nabla_a),
                         gga_rho_row(ing.rho_b, ing.nabla_b)])
    else:
        rows = (mgga_rho_row(ing.rho_a, ing.nabla_a, tau_a),
                mgga_rho_row(ing.rho_b, ing.nabla_b, tau_b))
    eps = np.asarray(dft.libxc.eval_xc(c_functional, rows, spin=spin,
                                       deriv=0)[0])
    return ing.integrate(eps)


def _libxc_c_reference_at_the_library_zeta(ing, c_functional):
    """libxc's spin-polarized correlation of the parent at the spin densities
    ``rho (1 +- zeta_lib) / 2`` implied by the library's own clipped zeta,
    with the gradient and (for a meta-GGA parent) the total-block tau split
    in proportion to the spin densities. Differs from the adapter only in
    where the rows are built, so it isolates the zeta clip from the wiring.
    """
    rho = ing.rho_a + ing.rho_b
    z = np.asarray(uks_zeta(jnp.asarray(ing.rho_a), jnp.asarray(ing.rho_b)))
    share_a, share_b = 0.5 * (1.0 + z), 0.5 * (1.0 - z)
    rho_a, rho_b = rho * share_a, rho * share_b
    nabla_a = ing.nabla_tot * share_a[:, None]
    nabla_b = ing.nabla_tot * share_b[:, None]
    if ing.alpha_column is None:
        rows = np.stack([gga_rho_row(rho_a, nabla_a),
                         gga_rho_row(rho_b, nabla_b)])
    else:
        tau = ing.total_tau_from_block()
        rows = (mgga_rho_row(rho_a, nabla_a, tau * share_a),
                mgga_rho_row(rho_b, nabla_b, tau * share_b))
    eps = np.asarray(dft.libxc.eval_xc(c_functional, rows, spin=1,
                                       deriv=0)[0])
    return ing.integrate(eps)


def _assert_block_tau_is_the_channel_tau(ing):
    """The tau the channel block's indicator encodes IS that channel's own
    kinetic-energy density.

    ``_libxc_x_reference`` inverts the block's own column to build the
    reference tau, so the exchange comparison below holds whatever that column
    carries; this is the statement that closes the circle, and it is taken
    against PySCF's ``eval_rho`` on the record's density matrix rather than
    against the library's own contraction. Points where the column sits ON its
    ``[0, _ALPHA_MAX]`` clip are excluded: there the column no longer encodes a
    tau at all (H's beta channel has no interior point and its alpha channel is
    at alpha = 0 exactly -- that is oracle O4), as are unresolved tail points
    (rho_sigma <= 1e-8).

    Bound: the reference SCF differs at round-off between runs, so this is
    quoted over draws rather than from one. Measured 1.06e-15 to 1.48e-15
    relative (worst 2.7e-12 absolute, on a tau of order 4.05e3, O atom) over
    the five meta-GGA architectures x {H, Li, N, O} x twelve draws, 420
    channel comparisons; 1e-13 clears the worst draw by 67x. An undoubled
    rho, sigma or tau in the block fails by
    O(1): reading alpha(rho_s, sigma_ss, tau_s) and inverting at the doubled
    arguments returns ``tau_W + 2^{2/3} (tau_s - tau_W)``, i.e. 0.59 of the
    channel's own (tau - tau_W) too much.

    The inversion (``parent_adapter.tau_from_alpha``) undoes the smooth
    positive part of ``metagga.compute_alpha`` exactly, so a one-orbital
    channel, whose stored column is the smoothing's floor ``width / 2``, is
    read back as tau_W plus the rounding residue and is included here (the
    H atom's alpha channel among them); only the ceiling excludes a point.
    """
    from xcquinox.alec.metagga import _ALPHA_MAX
    for s in (0, 1):
        rho_s = ing.rho_a if s == 0 else ing.rho_b
        column = np.asarray(ing.f_a if s == 0 else ing.f_b)[:, ing.alpha_column]
        interior = (column < _ALPHA_MAX) & (rho_s > 1e-8)
        if not interior.any():
            continue
        got = ing.channel_tau_from_block(s)[interior]
        expect = ing.true_tau(s)[interior]
        gap = np.max(np.abs(got - expect)
                     / np.maximum(np.abs(expect), 1e-30))
        assert gap < 1e-13, (ing.symbol, s, gap, int(interior.sum()))


# ---------------------------------------------------------------------------
# O1: the parent functional wearing the model's evaluation surface.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("symbol,spin", _ATOMS, ids=_ATOM_IDS)
@pytest.mark.parametrize("arch_name", _ARCHS)
def test_o1_exchange_path_equals_libxc_spin_polarized_parent(
        arch_name, symbol, spin):
    """O1: with the parent's own F_x in place of the network and this
    architecture's descriptor blocks in place, the library's UKS exchange
    assembly is libxc's spin-polarized exchange of the parent.

    For the meta-GGA rung this is the discriminating case: the parent reads
    the iso-orbital indicator out of the per-channel block, so the assembly
    equals libxc's spin-polarized SCAN exchange only if that block carries
    alpha(2 rho_sigma, 4 sigma_sigma_sigma, 2 tau_sigma). Feeding the
    total-density block into both channels moves this by tens of mHa on every
    open-shell atom with more than one electron (the last test of this
    section measures it).

    The reference inverts the channel block's own indicator to recover the
    tau the parent needs, which keeps the descriptor's value clip out of the
    energy comparison but would hold whatever that column carried. The
    statement that the column is the CHANNEL's indicator is made first, by
    :func:`_assert_block_tau_is_the_channel_tau`, against PySCF's own
    ``eval_rho``.

    Bound: measured residuals of 0.0 to 1.8e-15 Ha on Li, N and O and 7.0e-14
    Ha (PBE) / 4.4e-14 Ha (SCAN) on H over the 11 descriptor sets, the
    rounding of one libxc row against the other; 1e-10 Ha is the program's
    stated tolerance for O1 (Section 3.1) and sits three orders above the
    floor.

    What that bound covers is the ASSEMBLY, and it is not the number a reader
    gets by evaluating libxc independently: at PySCF's own per-spin tau, the
    meta-GGA rung differs from this path by 6.86e-11 Ha (N) and 5.30e-10 to
    6.25e-10 Ha (O), which is ABOVE the 1e-10 Ha bound above. All of it sits
    on the points where the block's indicator is on ``_ALPHA_MAX`` -- the
    clipped indicator is what the network reads, so the clipped functional is
    the model, and the descriptor's ceiling is deliberately held out of the
    assembly comparison. That whole-grid statement and its attribution are
    measured by
    :func:`test_o1_exchange_at_the_true_tau_is_the_indicator_ceiling`.
    """
    ing = _ingredients(arch_name, symbol, spin)
    x_functional, _c = _parent_of(arch_name)
    if ing.alpha_column is not None:
        _assert_block_tau_is_the_channel_tau(ing)
    got = ing.split_energy(_parent_model(ing, x_functional=x_functional))
    ref = _libxc_x_reference(ing, x_functional)
    assert abs(got - ref) < 1e-10, (arch_name, symbol, got, ref, ing.n_dropped)


@pytest.mark.parametrize("symbol,spin", _ATOMS, ids=_ATOM_IDS)
def test_o1_exchange_at_the_true_tau_is_the_indicator_ceiling(symbol, spin):
    """The whole-grid form of the exchange statement: against libxc at
    PySCF's own per-spin tau, the residual of the UKS exchange path is the
    ``_ALPHA_MAX`` ceiling of the iso-orbital indicator and nothing else.

    The oracle above inverts the channel block's own indicator column to build
    its reference, which holds the descriptor's ceiling out of the comparison
    deliberately -- the network reads the clipped indicator, so the clipped
    functional is the model, and what that oracle is about is the assembly.
    The statement a reader can check against libxc without adopting the
    block's tau is this one, and it is a different number. Both are recorded
    so neither is mistaken for the other.

    Measured on the SCAN parent at def2-svp / grid level 1 over twenty-two
    draws of the reference SCF: 4.43e-14 to 4.44e-14 Ha (H) and 0.0 to
    6.7e-16 Ha (Li), neither carrying one point on the ceiling -- H's two
    columns sit at the smoothing floor of 5e-6 (one orbital in alpha, an
    empty beta channel) and Li's alpha column tops out at 6.24; 6.862e-11 Ha
    (N, whose alpha column is on the ceiling at 510 of 4608 points carrying
    1.52e-5 of the electron density, while its beta column tops out at 5.10)
    and 5.30e-10 to 6.25e-10 Ha (O, both channels on the ceiling, 594 to 602
    of 4504 points, 1.53e-4 to 1.70e-4 of the density). The five meta-GGA
    descriptor sets of the registry, each on its own record, span 5.62e-10 to
    5.96e-10 Ha on O, inside the same range; the worst of all is 6.25e-10 Ha,
    against which 1e-8 Ha leaves 16x. The mass bound of 1e-3 leaves 5.9x on
    the worst draw: it is there to hold the clip to the TAIL, not to be
    tight.

    Attribution: with the ceiling points removed from BOTH sides the residual
    falls to the rounding of one libxc row against the other -- 0.0 to
    1.8e-15 Ha on N and O, and on H, which carries no ceiling point at all,
    the whole 4.44e-14 Ha, which is that floor itself; 1e-12 leaves 22x. The
    ceiling's occupancy is pinned per species, so raising or lowering
    ``_ALPHA_MAX`` fails here and names the constant rather than silently
    moving a documented number.
    """
    from xcquinox.alec.metagga import _ALPHA_MAX
    descriptors = (MetaGGAAlphaDescriptor(),)
    md = _precompute(symbol, spin, descriptors)
    ing = _Ingredients(md, descriptors)
    ing.alpha_column, ing.descriptors = 0, descriptors
    parent = _parent_model(ing, x_functional="MGGA_X_SCAN")
    rows = (mgga_rho_row(ing.rho_a, ing.nabla_a, ing.true_tau(0)),
            mgga_rho_row(ing.rho_b, ing.nabla_b, ing.true_tau(1)))
    eps_true = np.asarray(dft.libxc.eval_xc("MGGA_X_SCAN", rows, spin=1,
                                            deriv=0)[0])
    residual = abs(ing.split_energy(parent) - ing.integrate(eps_true))
    assert residual < 1e-8, (symbol, residual)

    ceiling = ((np.asarray(ing.f_a)[:, 0] >= _ALPHA_MAX)
               | (np.asarray(ing.f_b)[:, 0] >= _ALPHA_MAX))
    # N and O reach the ceiling in their density tails; H and Li do not come
    # near it at this identity (5e-6 and 6.24 as the largest column values).
    populated = symbol in ("N", "O")
    assert bool(ceiling.any()) is populated, (
        symbol, int(ceiling.sum()), _ALPHA_MAX,
        "the ceiling's occupancy moved; the residual below is measured at "
        "_ALPHA_MAX = 100 and has to be re-measured at any other value")
    rho_tot = ing.rho_a + ing.rho_b
    mass = float(np.sum(ing.w * rho_tot * ceiling)
                 / np.sum(ing.w * rho_tot))
    # 1.70e-4 (O) is the worst measured; 1e-3 leaves 5.9x, and the point of
    # the bound is that the ceiling stays a tail effect.
    assert mass < 1e-3, (symbol, mass, int(ceiling.sum()))

    ing.w = ing.w * ~ceiling
    off_ceiling = abs(ing.split_energy(parent) - ing.integrate(eps_true))
    assert off_ceiling < 1e-12, (symbol, residual, off_ceiling,
                                 int(ceiling.sum()))
    if populated:
        assert residual > 1e-12, (symbol, residual, int(ceiling.sum()))


@pytest.mark.parametrize("symbol,spin", _ATOMS, ids=_ATOM_IDS)
@pytest.mark.parametrize("arch_name", _ARCHS)
def test_o1_correlation_path_equals_libxc_parent_on_the_total_density(
        arch_name, symbol, spin):
    """O1: unpolarized cnet flag -- correlation is the parent's own
    correlation evaluated once on the total density with the total block,
    exactly. Bound as in the exchange test (measured 0.0 to 5.6e-17 Ha over
    the 11 descriptor sets, libxc's polarized entry point at zeta = 0
    against its unpolarized one).
    """
    ing = _ingredients(arch_name, symbol, spin)
    _x, c_functional = _parent_of(arch_name)
    parent = _parent_model(ing, c_functional=c_functional,
                           use_spin_polarization=False)
    zeros = np.zeros_like(ing.rho_a)
    got = ing.split_energy(parent, sigma_a=zeros, sigma_b=zeros)
    ref = _libxc_c_reference(ing, c_functional, spin=0)
    assert abs(got - ref) < 1e-10, (arch_name, symbol, got, ref)


@pytest.mark.parametrize("symbol,spin", _ATOMS, ids=_ATOM_IDS)
@pytest.mark.parametrize("arch_name", _ARCHS)
def test_o1_polarized_correlation_tracks_libxc_within_the_zeta_clip(
        arch_name, symbol, spin):
    """O1: polarized cnet flag -- correlation is libxc's spin-polarized
    correlation of the parent at the library's own zeta.

    Two statements. At the spin densities the library's clipped zeta implies
    the path IS libxc's polarized correlation: measured 0.0 to 5.6e-17 Ha
    over the 11 descriptor sets against a bound of 1e-10 (a zeta of the
    wrong sign, of the doubled densities, or dropped to 0 all fail it).
    Against libxc at the TRUE spin densities the residual is the documented
    boundary clip ``oneshot._ZETA_BOUNDARY_EPS = 1e-6``, which holds |zeta|
    strictly inside 1 so the PW92 spin interpolation stays twice
    differentiable: measured 7.1e-7 Ha for PBE and 7.3e-8 Ha for SCAN on the
    H atom (fully polarized everywhere; PBE's spin-scaling factor
    phi(zeta) carries (1 - zeta)^{2/3}, so a clip of 1e-6 moves E_c by
    1.2e-4 of itself, 7.10e-7 Ha on E_c = -6.006e-3 Ha -- the
    (1 - zeta)^{4/3} estimate in the oneshot comment understates it by 70x),
    6.92e-11 Ha for SCAN on N and 6.23e-11 on O (the _ALPHA_MAX clip of the
    total block in the density tail, absent from the per-spin PySCF tau of
    the reference), and 0.0 to 5.6e-17 elsewhere.

    The 7.1e-7 Ha figure is IDENTITY-SPECIFIC: it is measured on the H atom
    at def2-svp / grid level 1 with the PBE parent, where it is 1.182e-4 of
    E_c = -6.0066614638e-03 Ha. The shift is a fixed FRACTION of |E_c| at
    fixed polarization, so another fully polarized species, or the same one
    at a larger basis, raises it in proportion; 1e-6 clears this identity by
    1.4x AND NO MORE, and has to be re-measured rather than carried over if
    O1 is run at another basis or grid. The SCAN figure on H is not a
    fraction of its own E_c -- SCAN correlation vanishes on a one-electron
    density, E_c = -1.03e-11 Ha there -- so 7.3e-8 Ha is the clip's whole
    contribution rather than a perturbation of a finite value.

    The bound of 1e-6 is the program's; it refuses the zeta = 0 evaluation
    on the same records by 6.21e-3 to 2.99e-2 Ha. For a meta-GGA parent the
    adapter splits the total tau in proportion to the spin densities, which
    is exact because the SCAN correlation reads only the total tau, the total
    gradient invariant and zeta.
    """
    ing = _ingredients(arch_name, symbol, spin)
    _x, c_functional = _parent_of(arch_name)
    parent = _parent_model(ing, c_functional=c_functional,
                           use_spin_polarization=True)
    zeros = np.zeros_like(ing.rho_a)
    got = ing.split_energy(parent, sigma_a=zeros, sigma_b=zeros)
    at_library_zeta = _libxc_c_reference_at_the_library_zeta(ing, c_functional)
    assert abs(got - at_library_zeta) < 1e-10, (arch_name, symbol, got,
                                                 at_library_zeta)
    tau_a = tau_b = None
    if ing.alpha_column is not None:
        tau_a, tau_b = ing.true_tau(0), ing.true_tau(1)
    ref = _libxc_c_reference(ing, c_functional, tau_a=tau_a, tau_b=tau_b,
                             spin=1)
    assert abs(got - ref) < 1e-6, (arch_name, symbol, got, ref)


@pytest.mark.parametrize("symbol,spin", _ATOMS, ids=_ATOM_IDS)
def test_o1_per_channel_ingredients_are_the_libxc_spin_polarized_ingredients(
        symbol, spin):
    """O1, ingredient form: (2 rho_sigma, 4 sigma_sigma_sigma, 2 tau_sigma) is
    what libxc's spin-polarized meta-GGA reads for the channel, and the stored
    alpha column is exactly that alpha, with tau_sigma taken from PySCF's own
    ``eval_rho`` rather than from the library's contraction.

    The indicator amplifies the rounding of tau - tau_W by tau / tau_unif,
    which grows without bound into the density tail (up to 9.04e7 on Li's
    beta channel at grid level 1), so the comparison is posed pointwise
    relative to that amplification. Both gaps are draw-dependent -- the
    reference SCF differs at round-off between runs -- and are quoted over
    thirty draws rather than from one. Scaled by the amplification, the worst
    gap of a draw runs 7.8e-16 to 1.18e-15 (O, either channel) against a
    bound of 1e-11, which clears the worst draw by 8.5e3x. On the resolved
    region rho_sigma > 1e-8 the raw gap runs 4.6e-11 to 1.12e-10, on O's beta
    channel in every draw, against a bound of 1e-9 that the worst draw clears
    by 8.9x -- LESS THAN 10x, the thinnest margin in O1 after the zeta bound
    and the one to watch for cross-machine flakiness. The same quantity read
    1.68e-10 (6.0x) in an independent solution before the indicator's lower
    bound was smoothed, and the maximum sits where the smoothing is inert
    (alpha_raw = 9.88 at that point against a width of 1e-5, so the smooth
    positive part's slope there is 1 to twelve digits), so 6x is the margin
    to plan against. The two tau implementations agree to 2.7e-12 absolute
    (O, on a tau of 4.05e3 at the nucleus).
    """
    from xcquinox.alec.metagga import compute_alpha, _RHO_FLOOR
    descriptors = (MetaGGAAlphaDescriptor(),)
    md = _precompute(symbol, spin, descriptors)
    for s, suffix in ((0, "_a"), (1, "_b")):
        rho_s, _grad, sigma_ss = _spin_quantities(md, s)
        tau_s = _pyscf_tau(md, s)
        expect = np.asarray(compute_alpha(jnp.asarray(2.0 * rho_s),
                                          jnp.asarray(4.0 * sigma_ss),
                                          jnp.asarray(2.0 * tau_s)))
        got = np.asarray(md["metagga_features" + suffix])[:, 0]
        rho_d = np.maximum(2.0 * rho_s, _RHO_FLOOR)
        tau_unif = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0) * rho_d ** (5.0 / 3.0)
        amplification = np.maximum(2.0 * tau_s / np.maximum(tau_unif, _RHO_FLOOR),
                                   1.0)
        gap = np.abs(got - expect)
        assert float(np.max(gap / amplification)) < 1e-11, (
            symbol, suffix, float(np.max(gap / amplification)))
        resolved = rho_s > 1e-8
        if resolved.any():
            assert float(np.max(gap[resolved])) < 1e-9, (
                symbol, suffix, float(np.max(gap[resolved])))


def test_o1_total_block_in_both_channels_breaks_the_parent_reproduction():
    """The superseded contract, exercised on purpose: feeding the total-density
    block into both exchange channels does NOT reproduce libxc, which is the
    measurement that makes the passing oracle meaningful rather than vacuous.
    Measured on the N atom: 4.05e-2 Ha (25.4 kcal/mol) against a bound of
    1e-3 Ha.
    """
    descriptors = (MetaGGAAlphaDescriptor(),)
    md = _precompute("N", 3, descriptors)
    ing = _Ingredients(md, descriptors)
    ing.alpha_column, ing.descriptors = 0, descriptors
    parent = _parent_model(ing, x_functional="MGGA_X_SCAN")
    exact = ing.split_energy(parent)
    approx = float(split_exc_energy_uks(
        parent, jnp.asarray(ing.rho_a), jnp.asarray(ing.rho_b),
        jnp.asarray(ing.sigma_aa), jnp.asarray(ing.sigma_bb),
        jnp.asarray(ing.sigma_tot), ing.f_tot, ing.f_tot, ing.f_tot,
        jnp.asarray(ing.w)))
    assert abs(exact - approx) > 1e-3, (exact, approx)


# ---------------------------------------------------------------------------
# O2: the assembled Fock pair is the derivative of the assembled energy.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("species", sorted(_UKS_FD_SPECIES))
@pytest.mark.parametrize("arch_name", _ARCHS)
def test_o2_fock_pair_is_the_derivative_of_the_energy_on_the_open_shell_atoms(
        arch_name, species):
    """Oracle O2: central-difference check of the assembled UKS Fock matrices
    against the assembled energy on H, Li, N and O with every descriptor
    active, in the production configuration (polarized correlation, live
    per-channel feature blocks, the solver's one-electron gate). The probe
    and its bound live in ``test_solv01_split_xc``
    (``_assert_uks_fd_consistency``): a one-electron channel is displaced
    along its own rank-one manifold, the others linearly; def2-svp, grid
    level 2.

    Measured through this very helper over the 124 cases with the rotation
    path, worst relative residual per species against ``_TOL_UKS = 5e-7``:
    H 6.12e-10 (deep_combined), Li 3.61e-8 (deep_notransform_attn_3x16),
    N 6.61e-8 (deep_cusp_attn), O 9.00e-9 (shallow) -- a margin of 7.6x on
    the worst cell, the mask removing zero points in every cell (the
    residual is stated against the net derivative, which a rotation of the
    reference fixed point keeps small on Li and N). The indicator enters no
    mask: its lower bound is the smooth positive part of
    ``metagga.compute_alpha`` and the rotation path stays on the physical
    manifold (DEFERRED_WORK.md entries 27 and 30).
    """
    atom, spin, composition = _UKS_FD_SPECIES[species]
    model = _live_model(arch_name)
    md = _md_with_descriptors(model, species, atom, "def2-svp", spin,
                              composition)
    _assert_uks_fd_consistency(model, md, arch_name, species)


# ---------------------------------------------------------------------------
# O3: the closed shell carries no per-channel content.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch_name", _ARCHS)
def test_o3_closed_shell_record_is_byte_identical_to_the_archived_tree(
        arch_name):
    """Oracle O3: on a closed-shell molecule the RKS and closed-shell UKS
    energies and potentials of this architecture reproduce the tree at
    ae204537e digit for digit.

    rho_a = rho_b makes the three per-channel blocks the same array, so the
    exact spin scaling has no closed-shell content and the comparison is
    equality rather than a tolerance. The record, the archived fixture and
    the comparison itself are in ``test_closed_shell_byte_identity``; this
    case exists so the workflow matrix's ``oracle_selector`` reaches O3 for
    one architecture along with O1, O2 and O4. The record of an architecture
    is computed once per process and shared by the two entry points.
    """
    assert_closed_shell_record_matches(arch_name)


# ---------------------------------------------------------------------------
# O4: the H atom. One electron, one orbital, fully polarized.
# ---------------------------------------------------------------------------

def test_o4_h_atom_alpha_is_zero_at_every_grid_point():
    """diag(P_a, P_a) is a two-electron single-orbital system, so tau = tau_W
    exactly and the iso-orbital indicator vanishes identically (Sun, Ruzsinszky
    and Perdew, Phys. Rev. Lett. 115, 036402 (2015): alpha = 0 marks a single
    orbital). The raw indicator the stored column encodes is the rounding
    residue of tau - tau_W divided by tau_unif, so its size is
    draw-dependent: the reference solution differs at round-off between runs
    and that moves the residue. Measured maximum 8.3e-11, 1.7e-10 and 1.1e-10
    over the 2336-point grid in three independent solutions (1589 and 1732 of
    the points exactly 0.0 in the first two); 1e-8 clears the worst draw by
    60x and refuses a two-orbital block (the Li alpha block's median is
    1.81e-2) by six orders.

    The stored column itself is the smooth positive part of that residue
    (``metagga.compute_alpha``): ``width / 2 = 5e-6`` on every point, to the
    residue's own half. Both statements are made: the column sits at the
    floor, and the raw indicator read back through the exact inverse is
    zero to 1e-8.
    """
    from xcquinox.alec.metagga import (
        _ALPHA_SMOOTHING_WIDTH, invert_smooth_positive_part)
    md = _precompute("H", 1, (MetaGGAAlphaDescriptor(),))
    alpha_a = np.asarray(md["metagga_features_a"])[:, 0]
    floor = 0.5 * _ALPHA_SMOOTHING_WIDTH
    assert float(np.max(np.abs(alpha_a - floor))) < 1e-8, (
        float(np.max(np.abs(alpha_a - floor))))
    raw = np.asarray(invert_smooth_positive_part(alpha_a, _ALPHA_SMOOTHING_WIDTH))
    assert float(np.max(np.abs(raw))) < 1e-8, float(np.max(np.abs(raw)))


def test_o4_h_atom_rung35_block_is_the_doubled_single_orbital():
    """The alpha channel's block is [n_a, n_a] -- the occupancy of the doubled
    orbital in BOTH spin slots -- while the physical block is [n_a, 0]. The two
    are not the same feature vector, which is the whole content of the fix on a
    one-electron system.

    Bounds, all measured on this record: the two slots of the channel block
    agree to 1.11e-16 against atol 1e-14, and its first slot reproduces the
    total block's first slot exactly (the doubled matrix is a binary scaling
    of the physical one, so the two contractions differ only in the order
    the projector sums); the physical block's beta slot is exactly 0.0; the
    channel block's beta slot reaches 0.938 against the 1e-3 floor that
    separates it from the empty physical channel; the largest occupancy is
    0.938 against the Bessel bound of 1, and the smallest is 1.62e-05,
    positive.
    """
    md = _precompute("H", 1, (DMRung35Descriptor(),))
    block = np.asarray(md["rung35_features_a"])
    total = np.asarray(md["rung35_features"])
    np.testing.assert_allclose(block[:, 0], block[:, 1], rtol=0, atol=1e-14)
    np.testing.assert_allclose(block[:, 0], total[:, 0], rtol=0, atol=1e-14)
    assert float(np.max(np.abs(total[:, 1]))) < 1e-14, "H has no beta electron"
    assert float(np.max(block[:, 1])) > 1e-3, (
        "the doubled system's second slot must carry the SAME occupancy as the "
        "first, not the empty physical beta channel")
    assert float(np.max(block)) < 1.0 + 1e-12, "Bessel bound"
    assert float(np.min(block)) > -1e-12, "positive semidefinite P"


@pytest.mark.parametrize("arch_name", _ARCHS)
def test_o4_h_atom_exchange_equals_the_spin_scaled_unpolarized_evaluation(
        arch_name):
    """The alpha channel's block IS the block an RKS run on diag(P_a, P_a) would
    assemble, so the H-atom exchange energy is exactly half the model's
    spin-unpolarized evaluation on that system. The beta channel is empty and
    contributes only the model's rho_cutoff floor. Evaluated at the block the
    library assembles, where the indicator sits at the smoothing's floor
    ``width / 2`` (its raw value is zero; DEFERRED_WORK #27).

    Bounds: the block identity is measured bitwise on all 31 architectures
    (doubling a density matrix is a binary scaling, so every contraction of
    it is the doubled contraction); the energy identity is measured at
    exactly 0.0 on all 31 against 1e-12, the empty beta channel contributing
    1.1e-21 to 2.2e-21 Ha over two reference solutions (the rho_cutoff
    floor).
    """
    model = _live_model(arch_name)
    md = _precompute("H", 1, model.descriptors)
    rho_a, _grad_a, sigma_aa = _spin_quantities(md, 0)
    rho_b, _grad_b, sigma_bb = _spin_quantities(md, 1)
    w = jnp.asarray(md["grid_weights"])
    features_a_of, features_b_of, features_tot_of = make_uks_feature_fns(
        descriptors=model.descriptors,
        ao_deriv=jnp.asarray(md["ao_grid_deriv"]),
        s_matrix=jnp.asarray(md["s_matrix"]),
        n_grid=int(np.asarray(md["grid_weights"]).shape[0]),
        cusp_features=md.get("cusp_features"),
        rung35_proj_ao=md.get("rung35_proj_ao"),
        rung35ms_proj_ao=md.get("rung35ms_proj_ao"))
    P0 = jnp.asarray(md["dm_pbe"])
    doubled = doubled_spin_dm(P0, 0)
    # The channel block is the doubled system's OWN total block. Under the
    # superseded contract the left side was the physical molecular block and
    # this equality did not hold.
    np.testing.assert_allclose(np.asarray(features_a_of(P0)),
                               np.asarray(features_tot_of(doubled)),
                               rtol=0, atol=1e-14)
    ex_uks = 0.5 * float(jnp.sum(w * (
        model.eval_ex(jnp.asarray(2.0 * rho_a), jnp.asarray(4.0 * sigma_aa),
                      features_a_of(P0))
        + model.eval_ex(jnp.asarray(2.0 * rho_b), jnp.asarray(4.0 * sigma_bb),
                        features_b_of(P0)))))
    ex_rks_doubled = float(jnp.sum(w * model.eval_ex(
        jnp.asarray(2.0 * rho_a), jnp.asarray(4.0 * sigma_aa),
        features_tot_of(doubled))))
    assert abs(ex_uks - 0.5 * ex_rks_doubled) < 1e-12, (
        arch_name, ex_uks, 0.5 * ex_rks_doubled)


def test_o4_h_atom_beta_channel_carries_no_density():
    """The precondition the previous test rests on, stated separately so a
    failure names the cause."""
    md = _precompute("H", 1, ())
    rho_b, _grad, sigma_bb = _spin_quantities(md, 1)
    assert float(np.max(np.abs(rho_b))) < 1e-14
    assert float(np.max(np.abs(sigma_bb))) < 1e-14
