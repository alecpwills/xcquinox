"""V3 to V5 and the configuration layer of the parent anchor
(``SPEC_parent_anchor.md`` Section 4).

V3 -- an anchored network at initialization returns its parent's enhancement
factor pointwise, on the rows a real molecule integrates and on a row where the
parent sits within 1e-6 of the Lieb-Oxford ceiling; an UNANCHORED network is
today's forward, bitwise, reconstructed here from the documented expressions
rather than restated from the code.

V4 -- the pretraining-fidelity certificate PASSes at initialization from an
untrained anchored checkpoint written the way the pretrain stage writes one.

V5 -- the spin-scaling oracles O1 to O4 hold on an anchored architecture. The
cases are their own here because ``test_spin_scaling_oracles`` parametrizes
over the registry, which carries no anchored entry: anchoring is a run-level
switch, not a registry name.

The configuration layer -- ``model.parent_anchor`` parsed, refused where it
cannot be honoured, and carried into the training specs, the manifest and the
checkpoint loader's identity check.

Environment the numbers quoted below were measured on: pyscf 2.11.0, libxc
7.0.0, ``jax_enable_x64``, CPU.
"""
import dataclasses
import json
import os
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import xcquinox.alec as alec
from xcquinox.alec import parents
from xcquinox.alec.config import ArchitectureConfig, MoleculeSpec, anchored
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.descriptors import assemble_descriptor_features
from xcquinox.alec.models import _pack_row, _pack_row_polarized
from xcquinox.alec.networks import create_network_pair
from xcquinox.alec.oneshot import uks_zeta


#: The model's tail threshold (``models._NN_TAIL_THRESHOLD``): below it the
#: model masks ``F`` to 1 and the parent is not compared pointwise at all. The
#: energy those rows carry is 1.2e-12 Ha on the N atom and 1.9e-12 Ha on Ne
#: (measured in the design review of ``SPEC_parent_anchor.md``, 2026-08-25),
#: nine orders under the certificate's 1.0 mHa.
_RHO_FLOOR = 1e-10

#: Every registered architecture on the GGA rung. The meta-GGA entries are
#: refused by ``create_network_pair`` in this commit and are covered by the
#: refusal case below.
_GGA_ARCHS = tuple(name for name in sorted(alec.ARCHITECTURES)
                   if not ArchitectureConfig.is_meta_gga(alec.ARCHITECTURES[name]))

#: The descriptors any registered GGA architecture can ask for. One record
#: carrying all of them serves every architecture's feature block, so V3 costs
#: one reference SCF per system rather than one per descriptor set.
_ALL_GGA_DESCRIPTORS = ("cusp", "dm_statistics", "rung35", "rung35_multishell")

_RECORDS = {}


def _anchored_arch(name, coordinates=None):
    """The registry entry as an anchored run builds it: polarized correlation
    (the anchored correlation parent divides by the polarized PW92 baseline)
    and ``parent_anchor`` on."""
    arch = dataclasses.replace(alec.get_architecture(name),
                               use_polarized_correlation=True)
    arch = anchored(arch)
    if coordinates is not None:
        arch = dataclasses.replace(arch, descriptor_coordinates=coordinates)
    return arch


def _record(name, atom, basis, spin, composition, grid_level=1):
    """A reference record carrying every GGA descriptor block."""
    key = (name, basis, spin, grid_level)
    if key not in _RECORDS:
        from xcquinox.alec.descriptors import make_descriptor
        descriptors = tuple(make_descriptor(d) for d in _ALL_GGA_DESCRIPTORS)
        keys = tuple(sorted({k for d in descriptors
                             for k in d.required_mol_keys}))
        _RECORDS[key] = precompute_fixed_density_data(
            MoleculeSpec(name=name, atom=atom, basis=basis, charge=0,
                         spin=spin, atom_composition=composition,
                         grid_level=grid_level),
            required_keys=keys, descriptors=descriptors)
    return _RECORDS[key]


def _oh_record():
    """The OH radical at def2-svp, grid level 1: an open shell, so the two
    exchange channels differ and the correlation rows carry a real zeta."""
    return _record("OH", "O 0.0 0.0 0.0; H 0.0 0.0 0.97", "def2-svp", 1,
                   (("H", 1), ("O", 1)))


def _h2o_record():
    """H2O at sto-3g, grid level 1: a closed shell, where the doubled channel
    IS the total density and zeta is identically zero."""
    return _record("H2O", "O 0.0 0.0 0.0; H 0.0 0.757 0.587; "
                          "H 0.0 -0.757 0.587", "sto-3g", 0,
                   (("H", 2), ("O", 1)))


def _spin_densities(md):
    """``(rho_a, sigma_aa, rho_b, sigma_bb)``, contracted from the record's
    stored AO derivative table and density matrix as ``data.py`` does."""
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


def _exchange_rows(md, descriptors, spin_channel, stride=17):
    """Packed XNet rows of one spin channel: ``[2 rho_sigma,
    4 sigma_sigma_sigma, *features]`` with the features of the doubled density
    ``diag(P_sigma, P_sigma)``, which is what ``models._batched_network_apply``
    hands the network. Every ``stride``-th point above the tail threshold, so
    the case spans the whole grid at a bounded cost."""
    rho_a, sigma_aa, rho_b, sigma_bb = _spin_densities(md)
    rho_s, sigma_ss = ((rho_a, sigma_aa) if spin_channel == 0
                       else (rho_b, sigma_bb))
    rho = 2.0 * rho_s
    sigma = 4.0 * sigma_ss
    # A closed-shell record carries no per-channel blocks: rho_a = rho_b makes
    # the doubled channel the total density, so its block IS the total one and
    # the precompute stores it once (``assemble_descriptor_features`` reaches
    # it with spin_channel=None).
    channel = spin_channel if bool(md["is_unrestricted"]) else None
    features = np.asarray(assemble_descriptor_features(
        descriptors, md, spin_channel=channel))
    keep = np.where(rho > _RHO_FLOOR)[0][::stride]
    return (jnp.asarray(rho[keep]), jnp.asarray(sigma[keep]),
            jnp.asarray(features[keep]))


def _correlation_rows(md, descriptors, stride=17):
    """Packed CNet rows: ``[rho, sigma, zeta, *features]`` on the total
    density with the production spin polarization (``oneshot.uks_zeta``)."""
    rho_a, _sa, rho_b, _sb = _spin_densities(md)
    rho = np.asarray(md["rho_grid"])
    sigma = np.asarray(md["sigma_grid"])
    zeta = np.asarray(uks_zeta(jnp.asarray(rho_a), jnp.asarray(rho_b)))
    features = np.asarray(assemble_descriptor_features(descriptors, md))
    keep = np.where(rho > _RHO_FLOOR)[0][::stride]
    return (jnp.asarray(rho[keep]), jnp.asarray(sigma[keep]),
            jnp.asarray(zeta[keep]), jnp.asarray(features[keep]))


def _rel(got, want):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    return np.abs(got - want) / np.maximum(np.abs(want), 1e-300)


def _assert_is_the_parent(got, want, label, atol=1e-15):
    """The anchored forward returns its parent, to ``atol`` ABSOLUTE.

    The statement is absolute because the map is: ``networks._AlecLOB`` is
    written as ``limit sigmoid(x - ln(limit - 1)) - 1`` and the forward adds
    the 1 back, so the round trip carries an absolute error of order the ulp
    of 1 (measured 2.2e-16) whatever the parent's own size. A relative bound
    would therefore be a statement about how small the parent is rather than
    about the anchor: ``F_c^PBE`` falls to 6.6e-12 on a fifth of the N atom's
    rows and to 7.5e-13 on OH's, where an ulp of 1 is a relative 1e-4.

    Absolute is also the form the energy needs. ``F`` multiplies
    ``rho eps^base``, so an absolute error of 1e-15 in ``F`` is 1e-15 of the
    uniform-gas energy density of the same row -- for exchange, where
    ``F_x >= 1``, the two forms coincide anyway.
    """
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    dabs = np.abs(got - want)
    assert float(np.max(dabs)) <= atol, (label, "abs", float(np.max(dabs)))


# ---------------------------------------------------------------------------
# V3: the anchored forward at initialization IS the parent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch_name", _GGA_ARCHS)
def test_anchored_networks_return_the_parent_at_initialization(arch_name):
    """V3: at ``gated = 0`` the anchored forward returns ``F^parent``
    pointwise, for every registered GGA architecture, on stored rows.

    The rows are the ones the model integrates: the two doubled spin channels
    of the OH radical (def2-svp, grid level 1) for the exchange net, and the
    total density with ``oneshot.uks_zeta``'s polarization for the correlation
    net, above the model's 1e-10 tail threshold. The features are the
    architecture's own descriptor blocks, so an architecture whose descriptors
    reached the parent would fail here.

    Bound: 1e-15 ABSOLUTE (see :func:`_assert_is_the_parent` for why the
    absolute form is the statement). It is not bitwise -- ``_AlecLOB(1.174)``
    returns -1.11e-16 rather than 0.0 at argument 0, and the pre-image round
    trip carries the same residue of one ulp of 1 (measured in the design
    review, 2026-08-25).
    """
    arch = _anchored_arch(arch_name)
    xnet, cnet = create_network_pair(arch, seed=11)
    descriptors = arch.materialize_descriptors()
    md = _oh_record()

    for channel in (0, 1):
        rho, sigma, features = _exchange_rows(md, descriptors, channel)
        got = np.asarray(jax.vmap(
            lambda r, s, f: xnet(_pack_row(r, s, f)).squeeze()
        )(rho, sigma, features))
        want = np.asarray(jax.vmap(parents.pbe_fx)(rho, sigma))
        _assert_is_the_parent(got, want, (arch_name, "x", channel))

    rho, sigma, zeta, features = _correlation_rows(md, descriptors)
    got = np.asarray(jax.vmap(
        lambda r, s, z, f: cnet(_pack_row_polarized(r, s, z, f)).squeeze()
    )(rho, sigma, zeta, features))
    want = np.asarray(jax.vmap(parents.pbe_fc)(rho, sigma, zeta))
    _assert_is_the_parent(got, want, (arch_name, "c"))


def test_anchored_networks_return_the_parent_on_a_closed_shell():
    """V3 on a closed shell, where ``rho_a = rho_b`` makes the doubled channel
    the total density and zeta is identically zero: the anchored exchange net
    is ``F_x^PBE`` of the molecular density itself, and the correlation net is
    ``F_c^PBE`` at zeta = 0. H2O at sto-3g, grid level 1; bound 1e-15."""
    arch = _anchored_arch("deep_3x16")
    xnet, cnet = create_network_pair(arch, seed=3)
    descriptors = arch.materialize_descriptors()
    md = _h2o_record()

    rho, sigma, features = _exchange_rows(md, descriptors, 0)
    got = np.asarray(jax.vmap(
        lambda r, s, f: xnet(_pack_row(r, s, f)).squeeze())(rho, sigma, features))
    want = np.asarray(jax.vmap(parents.pbe_fx)(rho, sigma))
    _assert_is_the_parent(got, want, "H2O x")

    rho, sigma, zeta, features = _correlation_rows(md, descriptors)
    assert float(np.max(np.abs(np.asarray(zeta)))) < 1e-10, "closed shell"
    got = np.asarray(jax.vmap(
        lambda r, s, z, f: cnet(_pack_row_polarized(r, s, z, f)).squeeze()
    )(rho, sigma, zeta, features))
    want = np.asarray(jax.vmap(parents.pbe_fc)(rho, sigma, zeta))
    _assert_is_the_parent(got, want, "H2O c")


def test_anchored_exchange_holds_where_the_parent_is_at_the_bound():
    """V3 at the row the pre-image clamp exists for: a reduced gradient large
    enough that ``F_x^PBE`` sits within 1e-6 of the Lieb-Oxford ceiling 1.804.

    ``F_x = 1 + kappa - kappa / (1 + mu s^2 / kappa)`` sits ``gap`` under the
    ceiling at ``s^2 = (kappa / gap - 1) kappa / mu``: ``s = 1716.0`` for
    ``gap = 1e-6`` and ``s = 1.7e4`` for ``gap = 1e-8``, both built below. The
    anchored network must return the parent there to 1e-12 relative -- the
    pre-image ``ln[(limit - 1) F / (limit - F)]`` reaches 20.5 and 25.1, well
    inside ``Z_MAX = 40``, so no clamping is active and the identity is the
    ordinary one; what the case refuses is a transform that loses the parent
    as it approaches its own bound.
    """
    arch = _anchored_arch("deep_3x16")
    xnet, _cnet = create_network_pair(arch, seed=5)
    rho = 0.1
    k_F = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    kappa = float(parents.PBE_KAPPA)
    mu = float(parents.PBE_MU)
    for gap in (1e-6, 1e-8):
        s = float(np.sqrt((kappa / gap - 1.0) * kappa / mu))
        sigma = (s * 2.0 * k_F * rho) ** 2
        want = float(parents.pbe_fx(jnp.asarray(rho), jnp.asarray(sigma)))
        assert 0.0 < 1.804 - want < 2.0 * gap, (s, want)
        row = _pack_row(jnp.asarray(rho), jnp.asarray(sigma), jnp.zeros(0))
        got = float(xnet(row).squeeze())
        z = float(parents.lob_preimage(jnp.asarray(want), 1.804))
        assert abs(z) < 40.0, (gap, z)
        assert float(_rel(got, want)) <= 1e-12, (s, got, want)
        assert got < 1.804


@pytest.mark.parametrize("arch_name", _GGA_ARCHS)
def test_an_unanchored_network_is_todays_forward_bitwise(arch_name):
    """The unanchored class is untouched: with ``parent_anchor`` off the
    forward is ``1 + LOB(tanh(s)^2 MLP(inputs))``, term for term.

    Stated twice. First, the architecture with the flag explicitly off returns
    arrays IDENTICAL to the registry entry's at the same seed. Second, the
    forward is rebuilt here from the documented expressions -- the reduced
    gradient ``s = |grad rho| / (2 k_F rho)``, the optional Dick XCDiff
    log-compression of the MLP's inputs, the ``tanh(s)^2`` uniform-gas gate and
    ``networks._AlecLOB`` -- and compared bitwise, so a change to the
    unanchored path is caught here rather than in a recorded fixture.
    """
    plain = alec.get_architecture(arch_name)
    explicit = dataclasses.replace(plain, parent_anchor=False)
    xnet_a, cnet_a = create_network_pair(plain, seed=7)
    xnet_b, cnet_b = create_network_pair(explicit, seed=7)

    md = _h2o_record()
    descriptors = plain.materialize_descriptors()
    rho, sigma, features = _exchange_rows(md, descriptors, 0, stride=53)
    fx_a = np.asarray(jax.vmap(
        lambda r, s, f: xnet_a(_pack_row(r, s, f)).squeeze())(rho, sigma, features))
    fx_b = np.asarray(jax.vmap(
        lambda r, s, f: xnet_b(_pack_row(r, s, f)).squeeze())(rho, sigma, features))
    np.testing.assert_array_equal(fx_a, fx_b)

    fc_a = np.asarray(jax.vmap(
        lambda r, s, f: cnet_a(_pack_row(r, s, f)).squeeze())(rho, sigma, features))
    fc_b = np.asarray(jax.vmap(
        lambda r, s, f: cnet_b(_pack_row(r, s, f)).squeeze())(rho, sigma, features))
    np.testing.assert_array_equal(fc_a, fc_b)

    rebuilt = np.asarray(_rebuild_unanchored_xnet_forward(
        xnet_a, plain, rho, sigma, features))
    np.testing.assert_array_equal(fx_a, rebuilt)


def _mlp_apply(net, attention, netinp):
    """The MLP path the networks run: the plain ``eqx.nn.MLP`` call, or the
    same layers with the attention block after the first hidden layer."""
    if attention is None:
        return net(netinp)
    x = netinp
    layers = net.layers
    for i, layer in enumerate(layers[:-1]):
        x = layer(x)
        x = jax.nn.gelu(x)
        if i == 0:
            x = attention(x)
    return layers[-1](x)


def _rebuild_unanchored_xnet_forward(xnet, arch, rho, sigma, features):
    """Today's exchange forward, rebuilt from the documented expressions."""
    def one(r, s_inv, f):
        k_F = (3.0 * jnp.pi ** 2 * r) ** (1.0 / 3.0)
        s = jnp.sqrt(s_inv) / (2.0 * k_F * r)
        s = jnp.atleast_1d(s)
        s_mlp = ((1.0 - jnp.exp(-s * s)) * jnp.log(s + 1.0)
                 if arch.descriptor_log_transform else s)
        netinp = (jnp.concatenate([s_mlp, jnp.atleast_1d(f).flatten()])
                  if f.shape[0] > 0 else s_mlp)
        netterm = _mlp_apply(xnet.net, xnet.attention, netinp)
        gated = jnp.tanh(s) ** 2 * netterm
        return 1.0 + xnet.lobf(gated).squeeze()
    return jax.vmap(one)(rho, sigma, features)


@pytest.mark.parametrize("arch_name", ["deep_attn_3x16", "deep_combined_attn_3x16",
                                       "deep_rung35_attn_3x16"])
def test_zero_initialization_is_exact_on_the_attention_path(arch_name):
    """The anchor rests on ``gated = 0`` at initialization, and the attention
    block does not disturb it: the block sits BEFORE the final layer, whose
    weight and bias ``parent_anchor`` forces to zero, so the MLP output is
    exactly 0.0 on every row whatever the attention produces.

    Measured 0.0 (not merely small) on 64 random rows per architecture, which
    is why V3's residual is the transform's round-off alone.
    """
    arch = _anchored_arch(arch_name)
    xnet, cnet = create_network_pair(arch, seed=13)
    key = jax.random.PRNGKey(20260825)
    for net in (xnet, cnet):
        np.testing.assert_array_equal(
            np.asarray(net.net.layers[-1].weight),
            np.zeros_like(np.asarray(net.net.layers[-1].weight)))
        np.testing.assert_array_equal(
            np.asarray(net.net.layers[-1].bias),
            np.zeros_like(np.asarray(net.net.layers[-1].bias)))
        in_size = net.net.layers[0].weight.shape[1]
        rows = jax.random.normal(key, (64, in_size))
        out = np.asarray(jax.vmap(
            lambda r: _mlp_apply(net.net, net.attention, r))(rows))
        np.testing.assert_array_equal(out, np.zeros_like(out))


# ---------------------------------------------------------------------------
# Refusals at construction
# ---------------------------------------------------------------------------

def test_an_anchored_zeta_blind_correlation_network_is_refused():
    """An anchored architecture whose correlation net cannot see zeta is
    refused by name.

    The parent factor divides by the model's own baseline, and a zeta-blind
    cnet's baseline is the UNPOLARIZED PW92 while the pretraining data divides
    its open-shell targets by the polarized one: the two conventions differ by
    1.844e-3 in pointwise mean square and by +14.90 mHa in the per-system
    correlation energy of the N atom (measured in the design review,
    2026-08-25), fifteen times the certificate's 1.0 mHa atom tolerance. No v6
    configuration is built this way, so the case is refused rather than
    supported.
    """
    arch = anchored(dataclasses.replace(alec.get_architecture("deep_3x16"),
                                        use_polarized_correlation=False))
    with pytest.raises(ValueError, match="deep_3x16"):
        create_network_pair(arch, seed=0)


@pytest.mark.parametrize("arch_name", ["deep_mgga_3x16", "deep_rung35_mgga_3x16"])
def test_an_anchored_meta_gga_architecture_names_the_scan_commit(arch_name):
    """The meta-GGA rung's parent is SCAN, which is not implemented in this
    commit: the refusal names it rather than silently anchoring a meta-GGA
    network to PBE, which would pretrain the rung to the wrong functional
    (measured 24.0 mHa per system in the fidelity program's Section 2)."""
    arch = _anchored_arch(arch_name)
    with pytest.raises(NotImplementedError, match="(?i)scan"):
        create_network_pair(arch, seed=0)


def test_the_anchored_helper_forces_the_zero_initialized_final_layer():
    """``config.anchored`` turns on ``zero_init_final_layer`` whatever the
    registry entry says: ``shallow``, ``shallow_attn``, ``medium`` and
    ``medium_attn`` carry False, and without the override their ``gated`` at
    initialization is not zero (measured +1.826e-3 in ``F_x`` and -8.063e-3 in
    ``F_c`` on a live packed row for ``shallow``), so they would start off
    their parent."""
    for name in ("shallow", "shallow_attn", "medium", "medium_attn"):
        base = alec.get_architecture(name)
        assert base.zero_init_final_layer is False, name
        arch = anchored(base)
        assert arch.parent_anchor is True
        assert arch.zero_init_final_layer is True


# ---------------------------------------------------------------------------
# V4: the certificate at initialization
# ---------------------------------------------------------------------------

def _tiny_oracle_set(basis="def2-svp", grid_level=1):
    """The cheap system set the certificate is exercised on in tests: the H
    atom and H2, an open and a closed shell with one atomization energy
    between them."""
    return (
        MoleculeSpec(name="atom_H", atom="H 0.0 0.0 0.0", basis=basis, spin=1,
                     atom_composition=(("H", 1),), grid_level=grid_level),
        MoleculeSpec(name="H2", atom="H 0 0 0.371395; H 0 0 -0.371395",
                     basis=basis, spin=0, atom_composition=(("H", 2),),
                     grid_level=grid_level),
    )


def _write_untrained_pretrain_checkpoint(run_dir, arch, arch_name, *, seed):
    """An untrained checkpoint in the layout the certificate reads.

    ``fidelity._build_model`` deserialises ``xnet.eqx`` and ``cnet.eqx`` from
    ``<run_dir>/pretrain/<arch>``, so "at initialization" needs a checkpoint on
    disk; ``pretrain.run_pretrain`` writes exactly these two files with
    ``eqx.tree_serialise_leaves`` and a ``pretrain_metadata.json`` beside them.
    The metadata written here carries the two shape-and-identity keys the
    loader compares -- ``use_polarized_correlation`` and ``parent_anchor`` --
    at zero steps.
    """
    from xcquinox.alec.cluster.grid_config import pretrain_checkpoint_dir
    xnet, cnet = create_network_pair(arch, seed=seed)
    d = pretrain_checkpoint_dir(run_dir, arch_name)
    os.makedirs(d, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(d, "xnet.eqx"), xnet)
    eqx.tree_serialise_leaves(os.path.join(d, "cnet.eqx"), cnet)
    with open(os.path.join(d, "pretrain_metadata.json"), "w") as f:
        json.dump({
            "arch_name": arch_name,
            "depth": arch.depth,
            "nodes": arch.nodes,
            "pretrain_steps": 0,
            "use_polarized_correlation": bool(arch.use_polarized_correlation),
            "parent_anchor": bool(arch.parent_anchor),
            "meta_gga": ArchitectureConfig.is_meta_gga(arch),
            "n_extra_features": int(arch.n_extra_features),
        }, f)
    return d


def _anchored_cfg(arch=("deep_3x16",), basis="def2-svp", grid_level=1,
                  tol_AE=1.0, tol_atom=1.0, pretrain_seed=0,
                  parent_anchor=True):
    """The attribute surface ``cluster.fidelity`` reads off a ``GridConfig``,
    with the run-level ``model`` block the anchor adds."""
    return SimpleNamespace(
        sweep=SimpleNamespace(arch=tuple(arch)),
        inputs=SimpleNamespace(basis=basis, grid_level=grid_level,
                               density_fit=False, auxbasis=None,
                               orientation_lock_strength=0.0),
        pretrain=SimpleNamespace(seed=pretrain_seed),
        fidelity=SimpleNamespace(tol_AE=tol_AE, tol_atom=tol_atom,
                                 override_reason=None, enforce=True),
        use_polarized_correlation=True,
        model=SimpleNamespace(parent_anchor=parent_anchor,
                              descriptor_coordinates="legacy"),
    )


@pytest.mark.slow
def test_certificate_passes_at_initialization_for_an_anchored_architecture(
        tmp_path):
    """V4: the certificate PASSes at initialization, at the oracle floor.

    An untrained anchored ``deep_3x16`` is written with the pretrain stage's
    own serialization and certified at def2-svp / grid level 1 on the H atom
    and H2. At ``gated = 0`` the model IS its parent, so the certificate reads
    the parent against itself, on the production energy path.

    Measured, per system:

    * H2, a closed shell (zeta = 0): ``dE_xc = -2.22e-13 mHa``, which is
      2.2e-16 Ha on an ``E_xc`` of -0.6895 Ha -- one ulp. On a closed shell
      the anchored model reproduces libxc's PBE exchange-correlation energy to
      round-off, and that is the sharp form of the statement.
    * The H atom, fully spin-polarized: ``dE_xc = 7.10e-4 mHa``, and its
      atomization fold gives -8.91e-4 kcal/mol. The whole residual is here,
      and it is the FULLY-POLARIZED LIMIT rather than the anchor: the model
      evaluates correlation at ``oneshot.uks_zeta``'s clipped polarization
      ``1 - 1e-6`` while libxc is called at an empty beta channel, and libxc's
      own zeta = +-1 branch departs from the closed form of PBE eqs. 3-8 by up
      to 2.7e-5 relative (measured in ``test_parents``). On the H atom's
      ``E_c`` that is of order 1e-7 Ha, which is what is seen.

    The bound asserted is 1e-2 mHa and 1e-2 kcal/mol -- fourteen times the
    measurement and two orders under the binding 1.0 / 1.0 gate.

    The same architecture UNANCHORED is the control: ``zero_init_final_layer``
    gives it ``F_x = F_c = 1``, the LDA/PW92 limit, which is nowhere near PBE
    and FAILs the same gate. Without that half the PASS would be consistent
    with a certificate that cannot fail.
    """
    from xcquinox.alec.cluster import fidelity as fid

    run_dir = str(tmp_path / "run")
    arch = _anchored_arch("deep_3x16")
    _write_untrained_pretrain_checkpoint(run_dir, arch, "deep_3x16", seed=0)
    payload = fid.fidelity_certificate(
        _anchored_cfg(), run_dir, "deep_3x16",
        oracle_set=_tiny_oracle_set())

    assert payload["verdict"] == "PASS", payload["summary"]
    assert payload["parent"] == "pbe"
    assert payload["summary"]["failure_reasons"] == []
    assert payload["summary"]["max_atom_mHa"] < 1e-2, payload["summary"]
    assert payload["summary"]["max_dAE_kcalmol"] < 1e-2, payload["summary"]
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "PASS"

    by_name = {r["name"]: r for r in payload["per_system"]}
    assert abs(by_name["H2"]["dE_xc_mHa"]) < 1e-11, by_name["H2"]
    assert abs(by_name["atom_H"]["dE_xc_mHa"]) < 1e-2, by_name["atom_H"]

    control_dir = str(tmp_path / "control")
    plain = dataclasses.replace(alec.get_architecture("deep_3x16"),
                                use_polarized_correlation=True)
    _write_untrained_pretrain_checkpoint(control_dir, plain, "deep_3x16", seed=0)
    control = fid.fidelity_certificate(
        _anchored_cfg(parent_anchor=False), control_dir, "deep_3x16",
        oracle_set=_tiny_oracle_set())
    assert control["verdict"] == "FAIL"
    assert control["summary"]["max_atom_mHa"] > 1.0


@pytest.mark.slow
def test_certificate_records_the_anchor_state(tmp_path):
    """The certificate payload states the anchor beside the architecture, so
    the identity comparison the run validator makes can see it: a checkpoint
    certified anchored and a run configured unanchored are different model
    classes and must not be read as one."""
    from xcquinox.alec.cluster import fidelity as fid

    run_dir = str(tmp_path / "run")
    arch = _anchored_arch("deep_3x16")
    _write_untrained_pretrain_checkpoint(run_dir, arch, "deep_3x16", seed=0)
    payload = fid.fidelity_certificate(
        _anchored_cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set())
    assert payload.get("parent_anchor") is True, sorted(payload)


# ---------------------------------------------------------------------------
# V5: the spin-scaling oracles on an anchored architecture
# ---------------------------------------------------------------------------

def _anchored_live_model(arch_name, seed=0, coordinates=None):
    """An anchored model in the production configuration, the anchored twin of
    ``test_solv01_split_xc._live_model``. ``zero_init_final_layer`` is NOT
    turned off here: an anchored network is defined at ``gated = 0``, so the
    oracles are stated on the model the campaign starts from."""
    from xcquinox.alec.models import AlecGGAModel
    arch = _anchored_arch(arch_name, coordinates=coordinates)
    xnet, cnet = create_network_pair(arch, seed=seed)
    return arch, AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


@pytest.mark.slow
@pytest.mark.parametrize("symbol,spin", [("H", 1), ("Li", 1), ("N", 3),
                                         ("O", 2)],
                         ids=["H", "Li", "N", "O"])
def test_o1_anchored_exchange_path_equals_libxc_spin_polarized_pbe(symbol, spin):
    """Oracle O1 on an anchored ``deep_3x16``, stated on the ANCHORED network
    rather than on the parent adapter.

    O1 as ``test_spin_scaling_oracles`` runs it substitutes libxc for the
    network, so on an anchored model at ``gated = 0`` it would compare the
    parent with itself and assert nothing new. The statement that carries
    content here is the one the anchor makes: the library's UKS exchange
    assembly, driven by the ANCHORED network, reproduces libxc's spin-polarized
    PBE exchange of the same record. Bound 1e-10 Ha, the program's stated O1
    tolerance; measured worst 1.4e-13 Ha over the four atoms at def2-svp /
    grid level 1.
    """
    from pyscf import dft as pyscf_dft
    from xcquinox.alec.oneshot import split_exc_energy_uks
    from xcquinox.alec.tests.parent_adapter import gga_rho_row

    arch, model = _anchored_live_model("deep_3x16")
    md = _record(symbol, f"{symbol} 0 0 0", "def2-svp", spin,
                 ((symbol, 1),))
    ao = np.asarray(md["ao_grid_deriv"])
    dm = np.asarray(md["dm_pbe"])
    w = np.asarray(md["grid_weights"])
    rho_s, nabla_s = [], []
    for s in (0, 1):
        d = dm[s]
        rho_s.append(np.einsum("gi,ij,gj->g", ao[0], d, ao[0]))
        nabla_s.append(np.stack(
            [2.0 * np.einsum("gi,ij,gj->g", ao[k], d, ao[0])
             for k in (1, 2, 3)], axis=1))
    sigma_s = [np.sum(g * g, axis=1) for g in nabla_s]
    keep = (rho_s[0] >= 0.0) & (rho_s[1] >= 0.0)
    weights = w * keep

    descriptors = arch.materialize_descriptors()
    f_a = assemble_descriptor_features(descriptors, md, spin_channel=0)
    f_b = assemble_descriptor_features(descriptors, md, spin_channel=1)
    f_tot = assemble_descriptor_features(descriptors, md)
    nabla_tot = nabla_s[0] + nabla_s[1]
    sigma_tot = np.sum(nabla_tot * nabla_tot, axis=1)

    # Correlation is switched off by handing the assembly a model whose
    # correlation piece is zero, so the exchange statement stands alone.
    class _ExchangeOnly:
        cnet = model.cnet

        def eval_ex(self, rho, sigma, features):
            return model.eval_ex(rho, sigma, features)

        def eval_ec(self, rho, sigma, features, zeta=0.0):
            return jnp.zeros(np.asarray(rho).shape[0])

    got = float(split_exc_energy_uks(
        _ExchangeOnly(), jnp.asarray(rho_s[0]), jnp.asarray(rho_s[1]),
        jnp.asarray(sigma_s[0]), jnp.asarray(sigma_s[1]),
        jnp.asarray(sigma_tot), f_a, f_b, f_tot, jnp.asarray(weights)))
    rows = np.stack([gga_rho_row(rho_s[0], nabla_s[0]),
                     gga_rho_row(rho_s[1], nabla_s[1])])
    eps = np.asarray(pyscf_dft.libxc.eval_xc("GGA_X_PBE", rows, spin=1,
                                             deriv=0)[0])
    ref = float(np.sum(weights * (rho_s[0] + rho_s[1]) * eps))
    assert abs(got - ref) < 1e-10, (symbol, got, ref)


@pytest.mark.slow
@pytest.mark.parametrize("species", ["H", "Li", "N", "O"])
def test_o2_anchored_fock_pair_is_the_derivative_of_the_energy(species):
    """Oracle O2 on an anchored ``deep_3x16``: the assembled UKS Fock pair is
    the central difference of the assembled energy, with the parent's own
    derivatives inside ``V_xc``.

    The probe and its bound are ``test_solv01_split_xc``'s
    (``_assert_uks_fd_consistency``, ``_TOL_UKS = 5e-7`` relative, def2-svp /
    grid level 2, every channel rotated along its own aufbau manifold). This
    is the case that would catch a parent whose derivative is not the
    derivative of its value -- a ``jnp.where`` guard that is not the
    double-``where`` form, or a clamp differentiated through.
    """
    from xcquinox.alec.tests.test_solv01_split_xc import (
        _UKS_FD_SPECIES, _assert_uks_fd_consistency, _md_with_descriptors)

    atom, spin, composition = _UKS_FD_SPECIES[species]
    _arch, model = _anchored_live_model("deep_3x16")
    md = _md_with_descriptors(model, species, atom, "def2-svp", spin,
                              composition)
    _assert_uks_fd_consistency(model, md, "deep_3x16 (anchored)", species)


@pytest.mark.slow
def test_o3_anchored_closed_shell_carries_no_per_channel_content():
    """Oracle O3 for the anchored class.

    O3 as the oracle module runs it is a BYTE identity against fixtures
    recorded for the UNANCHORED class; those fixtures stay the unanchored
    class's (``SPEC_parent_anchor.md`` Section 3.6), so the anchored statement
    is the physical one they encode: on a closed shell ``rho_a = rho_b``, the
    three per-channel feature blocks are the same array, and the closed-shell
    UKS exchange assembly equals the RKS one exactly. Measured 0.0 on H2O at
    sto-3g, grid level 1, against 1e-14 Ha.
    """
    from xcquinox.alec.oneshot import split_exc_energy_uks

    arch, model = _anchored_live_model("deep_3x16")
    md = _h2o_record()
    descriptors = arch.materialize_descriptors()
    rho_a, sigma_aa, rho_b, sigma_bb = _spin_densities(md)
    np.testing.assert_array_equal(rho_a, rho_b)
    f_a = assemble_descriptor_features(descriptors, md, spin_channel=0)
    f_b = assemble_descriptor_features(descriptors, md, spin_channel=1)
    f_tot = assemble_descriptor_features(descriptors, md)
    np.testing.assert_array_equal(np.asarray(f_a), np.asarray(f_b))
    np.testing.assert_array_equal(np.asarray(f_a), np.asarray(f_tot))

    w = jnp.asarray(md["grid_weights"])
    rho = jnp.asarray(np.asarray(md["rho_grid"]))
    sigma = jnp.asarray(np.asarray(md["sigma_grid"]))
    uks = float(split_exc_energy_uks(
        model, jnp.asarray(rho_a), jnp.asarray(rho_b),
        jnp.asarray(sigma_aa), jnp.asarray(sigma_bb), sigma,
        f_a, f_b, f_tot, w))
    rks = float(jnp.sum(w * model.eval_exc(rho, sigma, f_tot, zeta=0.0)))
    assert abs(uks - rks) < 1e-14, (uks, rks)


@pytest.mark.slow
def test_o4_anchored_h_atom_exchange_is_the_spin_scaled_evaluation():
    """Oracle O4 on an anchored ``deep_3x16``: the H atom's exchange energy is
    exactly half the model's spin-unpolarized evaluation on the doubled alpha
    channel, the beta channel being empty.

    Measured 0.0 against 1e-12 Ha, the same bound the registry-parametrized
    case carries. Under the anchor the identity is the parent's own spin
    scaling as well as the network's, so a parent evaluated at the physical
    rather than the doubled channel would fail here by O(10 mHa).
    """
    from xcquinox.alec.descriptors import doubled_spin_dm
    from xcquinox.alec.solver import make_uks_feature_fns

    arch, model = _anchored_live_model("deep_3x16")
    md = _record("H", "H 0 0 0", "def2-svp", 1, (("H", 1),))
    rho_a, sigma_aa, rho_b, sigma_bb = _spin_densities(md)
    assert float(np.max(np.abs(rho_b))) < 1e-14, "H has no beta electron"
    w = jnp.asarray(md["grid_weights"])
    features_a_of, features_b_of, features_tot_of = make_uks_feature_fns(
        descriptors=arch.materialize_descriptors(),
        ao_deriv=jnp.asarray(md["ao_grid_deriv"]),
        s_matrix=jnp.asarray(md["s_matrix"]),
        n_grid=int(np.asarray(md["grid_weights"]).shape[0]),
        cusp_features=md.get("cusp_features"),
        rung35_proj_ao=md.get("rung35_proj_ao"),
        rung35ms_proj_ao=md.get("rung35ms_proj_ao"))
    P0 = jnp.asarray(md["dm_pbe"])
    doubled = doubled_spin_dm(P0, 0)
    ex_uks = 0.5 * float(jnp.sum(w * (
        model.eval_ex(jnp.asarray(2.0 * rho_a), jnp.asarray(4.0 * sigma_aa),
                      features_a_of(P0))
        + model.eval_ex(jnp.asarray(2.0 * rho_b), jnp.asarray(4.0 * sigma_bb),
                        features_b_of(P0)))))
    ex_rks_doubled = float(jnp.sum(w * model.eval_ex(
        jnp.asarray(2.0 * rho_a), jnp.asarray(4.0 * sigma_aa),
        features_tot_of(doubled))))
    assert abs(ex_uks - 0.5 * ex_rks_doubled) < 1e-12, (ex_uks, ex_rks_doubled)


# ---------------------------------------------------------------------------
# Configuration: model.parent_anchor
# ---------------------------------------------------------------------------

def _config_dict(**model_block):
    """A complete raw grid config carrying a ``model`` block."""
    from xcquinox.alec.tests.test_cluster_grid_config import _base_config_dict
    raw = _base_config_dict()
    raw["sweep"]["arch"] = ["deep_3x16"]
    raw["use_polarized_correlation"] = True
    if model_block:
        raw["model"] = dict(model_block)
    return raw


def _write_config(tmp_path, raw, name="grid.yaml"):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw))
    return str(path)


def test_grid_config_parses_the_model_parent_anchor_block(tmp_path):
    """``model: {parent_anchor: true}`` is read, and its absence leaves the
    unanchored default, so every configuration written before the anchor
    existed loads as the model class it was run under."""
    from xcquinox.alec.cluster.grid_config import load_grid_config

    cfg = load_grid_config(_write_config(tmp_path, _config_dict(parent_anchor=True)))
    assert cfg.model.parent_anchor is True

    plain = load_grid_config(_write_config(tmp_path, _config_dict(),
                                           name="plain.yaml"))
    assert plain.model.parent_anchor is False


def test_the_model_block_is_a_closed_schema(tmp_path):
    """A key the loader does not read is refused by name, as in every other
    block: an unread ``parent_ancor`` would run the sweep unanchored while the
    file appears to state the anchor."""
    from xcquinox.alec.cluster.grid_config import load_grid_config

    raw = _config_dict()
    raw["model"] = {"parent_ancor": True}
    with pytest.raises(ValueError, match="parent_ancor"):
        load_grid_config(_write_config(tmp_path, raw, name="typo.yaml"))


def test_an_anchored_meta_gga_sweep_is_refused_before_submission(tmp_path):
    """``model.parent_anchor: true`` with a meta-GGA architecture on the sweep
    axis is refused on the login node in this commit: the SCAN parent lands in
    the second one, and the refusal names it rather than letting the array
    fail per task."""
    from xcquinox.alec.cluster.grid_config import (
        load_grid_config, validate_grid_semantics)

    raw = _config_dict(parent_anchor=True)
    raw["sweep"]["arch"] = ["deep_mgga_3x16"]
    cfg = load_grid_config(_write_config(tmp_path, raw, name="mgga.yaml"))
    with pytest.raises((ValueError, NotImplementedError), match="(?i)scan"):
        validate_grid_semantics(cfg, SimpleNamespace(pool_size=64))


def test_an_anchored_zeta_blind_run_is_refused_before_submission(tmp_path):
    """``model.parent_anchor: true`` without ``use_polarized_correlation`` is
    refused for the reason the construction refusal states: the anchored
    correlation parent divides by the polarized PW92 baseline the model
    multiplies by, and a zeta-blind run's baseline is the unpolarized one
    (+14.90 mHa on the N atom)."""
    from xcquinox.alec.cluster.grid_config import (
        load_grid_config, validate_grid_semantics)

    raw = _config_dict(parent_anchor=True)
    raw["use_polarized_correlation"] = False
    cfg = load_grid_config(_write_config(tmp_path, raw, name="blind.yaml"))
    with pytest.raises(ValueError, match="(?i)polarized"):
        validate_grid_semantics(cfg, SimpleNamespace(pool_size=64))


def test_the_weight_zero_placeholder_refusal_is_lifted_under_the_anchor(tmp_path):
    """The energy-term weight is no longer the value that decides whether the
    certificate can be met, so the placeholder refusal -- which sends the
    author to the weight sweep -- applies to UNANCHORED configurations only.
    An anchored configuration states ``energy_term_weight: 0.0`` and is
    accepted."""
    from xcquinox.alec.cluster.grid_config import (
        load_grid_config, validate_grid_semantics)

    raw = _config_dict(parent_anchor=True)
    raw["pretrain"]["dfs_set"] = True
    raw["pretrain"]["energy_term_weight"] = 0.0
    cfg = load_grid_config(_write_config(tmp_path, raw, name="anchored_w0.yaml"))
    validate_grid_semantics(cfg, SimpleNamespace(pool_size=64))

    raw_plain = _config_dict(parent_anchor=False)
    raw_plain["pretrain"]["dfs_set"] = True
    raw_plain["pretrain"]["energy_term_weight"] = 0.0
    cfg_plain = load_grid_config(
        _write_config(tmp_path, raw_plain, name="plain_w0.yaml"))
    with pytest.raises(ValueError, match="energy_term_weight"):
        validate_grid_semantics(cfg_plain, SimpleNamespace(pool_size=64))


def _anchored_spec_builder_cfg(tmp_path, parent_anchor=True):
    """The spec-builder's own fixture config, with the run-level model block
    the anchor adds and the polarized correlation it requires."""
    from xcquinox.alec.cluster.grid_config import ModelConfig
    from xcquinox.alec.tests.test_cluster_spec_builder import _make_cfg
    cfg = _make_cfg(tmp_path)
    cfg = dataclasses.replace(
        cfg,
        sweep=dataclasses.replace(cfg.sweep, arch=("deep_3x16",)),
        use_polarized_correlation=True,
        model=ModelConfig(parent_anchor=parent_anchor))
    return cfg


def test_the_anchor_reaches_the_training_specs(tmp_path):
    """The anchor is part of the architecture identity everywhere the
    architecture is identified: every spec the harness materializes carries
    ``spec.arch.parent_anchor``, so the model class a task builds is read from
    the spec rather than re-derived from the YAML at each stage."""
    from xcquinox.alec.cluster.spec_builder import build_training_specs
    from xcquinox.alec.cluster.domain import get_domain_profile
    from xcquinox.alec.tests.test_cluster_spec_builder import (
        _make_ledger, _make_pool)

    domain = get_domain_profile("dfs_step7")
    cfg = _anchored_spec_builder_cfg(tmp_path)
    built = build_training_specs(_make_pool(), _make_ledger(), cfg, domain,
                                 str(tmp_path / "run"))
    assert built, "no specs built"
    for _cell, spec in built:
        assert spec.arch.parent_anchor is True
        assert spec.arch.use_polarized_correlation is True
        assert spec.arch.zero_init_final_layer is True

    plain = build_training_specs(
        _make_pool(), _make_ledger(),
        _anchored_spec_builder_cfg(tmp_path, parent_anchor=False), domain,
        str(tmp_path / "run_plain"))
    for _cell, spec in plain:
        assert spec.arch.parent_anchor is False


def test_the_manifest_records_the_anchor_state(tmp_path):
    """The preflight's ``manifest.json`` states the run's model class.

    The manifest is what a later reader identifies a run's artifacts by, and
    an anchored run and an unanchored one at the same architecture name are
    different functionals. ``write_manifest`` takes the configuration when it
    needs one, so the call below adapts to the signature rather than pinning
    an argument list.
    """
    import inspect
    from xcquinox.alec.cluster.materialize import (
        materialize_specs, write_manifest)
    from xcquinox.alec.cluster.grid_config import expand_grid
    from xcquinox.alec.cluster.domain import get_domain_profile
    from xcquinox.alec.cluster.spec_builder import build_training_specs
    from xcquinox.alec.tests.test_cluster_spec_builder import (
        _make_ledger, _make_pool)

    domain = get_domain_profile("dfs_step7")
    cfg = _anchored_spec_builder_cfg(tmp_path)
    out_dir = str(tmp_path / "specs")
    os.makedirs(out_dir, exist_ok=True)
    built = build_training_specs(_make_pool(), _make_ledger(), cfg, domain,
                                 str(tmp_path / "run"))
    paths = materialize_specs(built, out_dir)
    cells = expand_grid(cfg)
    kwargs = {}
    if "cfg" in inspect.signature(write_manifest).parameters:
        kwargs["cfg"] = cfg
    manifest_path = write_manifest(cells, paths, out_dir, **kwargs)
    with open(manifest_path) as f:
        payload = json.load(f)
    recorded = payload.get("parent_anchor")
    if recorded is None:
        recorded = (payload.get("model") or {}).get("parent_anchor")
    assert recorded is True, sorted(payload)


def test_the_architecture_describes_its_anchor_state():
    """``describe()`` is what the resolved configuration, the manifest and the
    certificate identity are written from, so the anchor has to appear in it:
    two architectures that differ only in the anchor are different model
    classes and must not serialize to the same description."""
    plain = dataclasses.replace(alec.get_architecture("deep_3x16"),
                                use_polarized_correlation=True)
    anchor = anchored(plain)
    assert plain.describe() != anchor.describe()
    assert anchor.describe().get("parent_anchor") is True
    assert plain.describe().get("parent_anchor") is False


# ---------------------------------------------------------------------------
# V7: the loader refuses an anchor-state mismatch
# ---------------------------------------------------------------------------

def _training_spec(arch, pretrain_checkpoint, tmp_path):
    from xcquinox.alec.config import TrainingSpec
    return TrainingSpec(
        arch=arch,
        molecules=(MoleculeSpec(name="H", atom="H 0 0 0", basis="sto-3g",
                                charge=0, spin=1,
                                atom_composition=(("H", 1),), grid_level=1),),
        targets=(("H", -0.5),),
        atom_energies=(("H", -0.5),),
        loss_name="A_atomization", n_steps=1, lr_start=1e-3, lr_end=1e-5,
        lr_decay_start=0.0, grad_clip=1.0,
        checkpoint_dir=str(tmp_path / "ckpt"), seed=0,
        pretrain_checkpoint=pretrain_checkpoint)


@pytest.mark.parametrize("recorded,requested",
                         [(False, True), (True, False)])
def test_the_loader_refuses_an_anchor_state_mismatch(tmp_path, monkeypatch,
                                                     recorded, requested):
    """V7: networks recorded under one anchor state are not loadable into a
    model of the other, and the refusal names both states.

    The flag is a STATIC field, so it lives in the treedef and not in the eqx
    leaf stream: ``tree_deserialise_leaves`` would load an unanchored
    checkpoint into an anchored skeleton silently, the leaf shapes being
    identical. The refusal therefore reads the state recorded beside the
    checkpoint (``pretrain_metadata.json``, which already records
    ``use_polarized_correlation`` for the same reason) and compares it with the
    model being built. The two are different functionals -- an unanchored
    checkpoint's networks are corrections to F = 1, an anchored one's to the
    parent -- so loading one as the other is a silently wrong model.
    """
    from xcquinox.alec import train as train_mod

    monkeypatch.delenv(train_mod._ALLOW_UNCERTIFIED_ENV, raising=False)
    base = dataclasses.replace(alec.get_architecture("deep_3x16"),
                               use_polarized_correlation=True)
    recorded_arch = anchored(base) if recorded else base
    requested_arch = anchored(base) if requested else base

    run_dir = str(tmp_path / f"run_{recorded}_{requested}")
    d = _write_untrained_pretrain_checkpoint(run_dir, recorded_arch,
                                             "deep_3x16", seed=0)
    with open(os.path.join(d, "fidelity_certificate.json"), "w") as f:
        json.dump({"verdict": "PASS", "arch": "deep_3x16",
                   "parent_anchor": recorded}, f)

    spec = _training_spec(requested_arch, d, tmp_path)
    with pytest.raises(ValueError, match="(?i)anchor"):
        train_mod._build_model(spec)


def test_the_loader_accepts_a_matching_anchor_state(tmp_path, monkeypatch):
    """The control for the refusal above: an anchored checkpoint loads into an
    anchored model, so the refusal is a statement about the mismatch and not
    about the anchor."""
    from xcquinox.alec import train as train_mod

    monkeypatch.delenv(train_mod._ALLOW_UNCERTIFIED_ENV, raising=False)
    arch = anchored(dataclasses.replace(alec.get_architecture("deep_3x16"),
                                        use_polarized_correlation=True))
    run_dir = str(tmp_path / "run_match")
    d = _write_untrained_pretrain_checkpoint(run_dir, arch, "deep_3x16", seed=0)
    with open(os.path.join(d, "fidelity_certificate.json"), "w") as f:
        json.dump({"verdict": "PASS", "arch": "deep_3x16",
                   "parent_anchor": True}, f)
    model = train_mod._build_model(_training_spec(arch, d, tmp_path))
    assert model is not None
    assert getattr(model.xnet, "parent", None) == "pbe"
    assert getattr(model.cnet, "parent", None) == "pbe"
