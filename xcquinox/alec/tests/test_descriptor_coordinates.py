"""The network input coordinates: ``ArchitectureConfig.descriptor_coordinates``.

``"legacy"`` is the committed forward, byte for byte. ``"dfs"`` is the
coordinate set of the vendored reference implementation this repository
replicates (Dick and Fernandez-Serra, Phys. Rev. B 104, L161109 (2021);
``dpyscfl/net.py``, ``get_descriptors`` / ``get_scf``), which the anchored
campaign runs on:

* EXCHANGE, per doubled spin channel and with ``spin_scaling`` on, consumes
  ONLY the transformed reduced gradient at the GGA level
  (``X_L(n_input=1, use=[1])``) and ``[x_s, x_alpha]`` at the meta-GGA level
  (``X_L(n_input=2, use=[1, 2])``): the density itself is never an input, which
  is what makes the exchange enhancement factor invariant under uniform density
  scaling. ``x_s = (1 - exp(-s^2)) ln(s + 1)`` with
  ``s = |grad rho| / (2 (3 pi^2)^(1/3) rho^(4/3))``;
  ``x_alpha = ln((alpha + 1) / 2)``.
* CORRELATION, on the total density, consumes
  ``x_0 = ln(rho^(1/3) + 1e-5)``,
  ``x_1 = ln(0.5 [(1 + zeta)^(4/3) + (1 - zeta)^(4/3)])`` and ``x_s`` (and
  ``x_alpha`` on the meta-GGA rung). The reduced gradient is NOT rescaled by a
  power of ``(1 +- zeta)``: that line is an XCDiff addition carrying an obvious
  typo in the vendored source and is not part of the coordinates taken here.
* The descriptor extras and both uniform-gas gates are unchanged; on the
  meta-GGA rung the indicator column the MLP receives becomes ``x_alpha``,
  which removes a deviation from the reference implementation that the
  networks' own comments record.

Every case below rebuilds the forward from those expressions and the network's
own MLP, gate and output squash, so what is pinned is the coordinate map rather
than a restatement of the code that computes it.
"""
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import xcquinox.alec as alec
from xcquinox.alec import parents
from xcquinox.alec.config import ArchitectureConfig, anchored
from xcquinox.alec.models import _pack_row, _pack_row_polarized
from xcquinox.alec.networks import create_network_pair


#: The offset the reference implementation adds inside the density logarithm
#: (``dpyscfl/net.py``: ``self.loge = 1e-5``).
_LOGE = 1e-5

_GGA_ARCHS = tuple(name for name in sorted(alec.ARCHITECTURES)
                   if not ArchitectureConfig.is_meta_gga(alec.ARCHITECTURES[name]))
_MGGA_ARCHS = tuple(name for name in sorted(alec.ARCHITECTURES)
                    if ArchitectureConfig.is_meta_gga(alec.ARCHITECTURES[name]))


# ---------------------------------------------------------------------------
# The coordinates, written out independently
# ---------------------------------------------------------------------------

def _reduced_gradient(rho, sigma):
    k_F = (3.0 * jnp.pi ** 2 * rho) ** (1.0 / 3.0)
    return jnp.sqrt(sigma) / (2.0 * k_F * rho)


def _x_s(s):
    return (1.0 - jnp.exp(-s * s)) * jnp.log(s + 1.0)


def _x_0(rho):
    return jnp.log(rho ** (1.0 / 3.0) + _LOGE)


def _x_1(zeta):
    return jnp.log(0.5 * ((1.0 + zeta) ** (4.0 / 3.0)
                          + (1.0 - zeta) ** (4.0 / 3.0)))


def _raw_indicator(column):
    """The RAW iso-orbital indicator a stored ``metagga`` column encodes.

    The column is ``min(p(alpha_raw), _ALPHA_MAX)`` with ``p`` the smooth
    positive part of width ``w = metagga._ALPHA_SMOOTHING_WIDTH``, whose exact
    inverse is ``alpha_raw = a - w^2 / (4 a)``; at and above the ceiling the
    column no longer encodes an indicator and the ceiling is returned. Written
    out here rather than imported so the case states the inverse rather than
    re-using whatever the library computes.
    """
    from xcquinox.alec.metagga import _ALPHA_MAX, _ALPHA_SMOOTHING_WIDTH
    a = jnp.where(column > 0.0, column, 1.0)
    raw = a - _ALPHA_SMOOTHING_WIDTH ** 2 / (4.0 * a)
    return jnp.where(column < _ALPHA_MAX, raw, column)


def _x_alpha(alpha_raw):
    return jnp.log((alpha_raw + 1.0) / 2.0)


def _rows(n, n_features, seed=20260825):
    """Random physical rows: densities over six decades, reduced gradients up
    to 6, polarizations inside the production clip, and feature columns in the
    unit interval (every descriptor of the registry is bounded there)."""
    rng = np.random.default_rng(seed)
    rho = 10.0 ** rng.uniform(-4.0, 2.0, size=n)
    s = rng.uniform(0.0, 6.0, size=n)
    k_F = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    sigma = (s * 2.0 * k_F * rho) ** 2
    zeta = rng.uniform(-1.0 + 1e-6, 1.0 - 1e-6, size=n)
    features = rng.uniform(0.0, 1.0, size=(n, n_features))
    return (jnp.asarray(rho), jnp.asarray(sigma), jnp.asarray(zeta),
            jnp.asarray(features))


def _mlp_apply(net, attention, netinp):
    """The MLP path the networks run: the plain call, or the same layers with
    the attention block after the first hidden layer."""
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


def _extras(features, alpha_index, coordinates):
    """The descriptor extras as the MLP receives them: unchanged, except that
    under ``"dfs"`` the iso-orbital indicator column becomes ``x_alpha`` of the
    RAW indicator the column encodes.

    Which indicator the coordinate is taken at is a choice, and it is pinned
    here: the raw one, recovered from the stored column, the same value the
    SCAN parent is evaluated at (``SPEC_parent_anchor.md`` Section 3.1). Taking
    it at the stored column instead moves ``F_x`` by up to 2.97e-11 relative on
    the rows below -- small, but a different functional, and this is where a
    silent change of the reading would be caught.
    """
    extras = jnp.atleast_1d(features).flatten()
    if coordinates != "dfs" or alpha_index is None or alpha_index < 0:
        return extras
    return extras.at[alpha_index].set(
        _x_alpha(_raw_indicator(extras[alpha_index])))


def _xnet_inputs(arch, rho, sigma, features, alpha_index):
    """The XNet's MLP input vector, from the coordinate definitions."""
    s = jnp.atleast_1d(_reduced_gradient(rho, sigma))
    if arch.descriptor_coordinates == "dfs":
        base = _x_s(s)
    else:
        base = _x_s(s) if arch.descriptor_log_transform else s
    extras = _extras(features, alpha_index, arch.descriptor_coordinates)
    return jnp.concatenate([base, extras]) if extras.shape[0] else base


def _cnet_inputs(arch, rho, sigma, zeta, features, alpha_index):
    """The CNet's MLP input vector, from the coordinate definitions."""
    s = jnp.atleast_1d(_reduced_gradient(rho, sigma))
    rs = jnp.atleast_1d((3.0 / (4.0 * jnp.pi * rho)) ** (1.0 / 3.0))
    zeta_c = jnp.clip(zeta, -1.0, 1.0)
    extras = _extras(features, alpha_index, arch.descriptor_coordinates)
    if arch.descriptor_coordinates == "dfs":
        head = [jnp.atleast_1d(_x_0(rho)), jnp.atleast_1d(_x_1(zeta_c)),
                _x_s(s)]
    elif arch.descriptor_log_transform:
        head = [_x_s(rs), _x_s(s),
                jnp.atleast_1d(0.5 * ((1.0 + zeta_c) ** (4.0 / 3.0)
                                      + (1.0 - zeta_c) ** (4.0 / 3.0)))]
    else:
        head = [rs, s,
                jnp.atleast_1d(0.5 * ((1.0 + zeta_c) ** (4.0 / 3.0)
                                      + (1.0 - zeta_c) ** (4.0 / 3.0)))]
    return jnp.concatenate(head + ([extras] if extras.shape[0] else []))


def _gate(arch, rho, sigma, features, alpha_index):
    """The uniform-gas gate, unchanged by the coordinate choice: ``tanh(s)^2``
    on the GGA rung, ``x_s + tanh^2(ln((alpha + 1) / 2))`` on the meta-GGA
    rung (the reference implementation's eq. 12 prefactor)."""
    s = jnp.atleast_1d(_reduced_gradient(rho, sigma))
    if not ArchitectureConfig.is_meta_gga(arch):
        return jnp.tanh(s) ** 2
    alpha = jnp.atleast_1d(features).flatten()[alpha_index]
    return _x_s(s) + jnp.tanh(_x_alpha(alpha)) ** 2


def _rebuild(net, arch, rho, sigma, features, alpha_index, *, zeta=None,
             parent=None):
    """The whole forward, rebuilt: coordinates, MLP, gate, output squash and
    -- when the network is anchored -- the parent's pre-image."""
    if zeta is None:
        netinp = _xnet_inputs(arch, rho, sigma, features, alpha_index)
    else:
        netinp = _cnet_inputs(arch, rho, sigma, zeta, features, alpha_index)
    netterm = _mlp_apply(net.net, net.attention, netinp)
    gated = _gate(arch, rho, sigma, features, alpha_index) * netterm
    if parent is None:
        return 1.0 + net.lobf(gated).squeeze()
    f_parent = (parents.pbe_fx(rho, sigma) if zeta is None
                else parents.pbe_fc(rho, sigma, zeta))
    z = parents.lob_preimage(f_parent, net.lobf.limit)
    return 1.0 + net.lobf(z + gated).squeeze()


def _alpha_index(net):
    idx = int(getattr(net, "metagga_alpha_index", -1))
    return idx if idx >= 0 else None


def _arch(name, coordinates, anchor=False, zero_init=False):
    """The registry entry with the coordinates set.

    ``zero_init_final_layer`` is turned OFF by default here, and that is the
    whole reason these cases have content: a zero-initialized final layer makes
    the MLP output exactly 0.0, so the forward is ``F = 1`` (or the parent,
    when anchored) whatever the MLP was fed and the coordinates could not be
    observed from outside at all. Every registered ``deep_*`` entry carries the
    flag on, so a case built at the registry's own value would pass against any
    coordinate map whatsoever.
    """
    arch = dataclasses.replace(alec.get_architecture(name),
                               use_polarized_correlation=True,
                               zero_init_final_layer=zero_init,
                               descriptor_coordinates=coordinates)
    return anchored(arch) if anchor else arch


def _perturb_final_layer(net, seed=20260826, scale=0.35):
    """A network whose final layer is NOT zero, so ``gated != 0`` and the
    coordinates reach the output. Used where the architecture must stay
    anchored (``config.anchored`` forces the zero initialization, which is
    what makes the anchor exact at step 0)."""
    key = jax.random.PRNGKey(seed)
    k_w, k_b = jax.random.split(key)
    weight = scale * jax.random.normal(k_w, net.net.layers[-1].weight.shape)
    bias = scale * jax.random.normal(k_b, net.net.layers[-1].bias.shape)
    return eqx.tree_at(
        lambda m: (m.net.layers[-1].weight, m.net.layers[-1].bias),
        net, replace=(weight, bias))


def _forward(net, rows, polarized):
    rho, sigma, zeta, features = rows
    if polarized:
        return np.asarray(jax.vmap(
            lambda r, s, z, f: net(_pack_row_polarized(r, s, z, f)).squeeze()
        )(rho, sigma, zeta, features))
    return np.asarray(jax.vmap(
        lambda r, s, f: net(_pack_row(r, s, f)).squeeze())(rho, sigma, features))


def _rebuilt(net, arch, rows, polarized, parent=None):
    rho, sigma, zeta, features = rows
    idx = _alpha_index(net)
    if polarized:
        return np.asarray(jax.vmap(
            lambda r, s, z, f: _rebuild(net, arch, r, s, f, idx, zeta=z,
                                        parent=parent))(rho, sigma, zeta, features))
    return np.asarray(jax.vmap(
        lambda r, s, f: _rebuild(net, arch, r, s, f, idx, parent=parent)
    )(rho, sigma, features))


# ---------------------------------------------------------------------------
# The default, and the legacy path unchanged
# ---------------------------------------------------------------------------

def test_the_default_coordinates_are_legacy():
    """Every registry entry defaults to the committed coordinates, so a
    configuration written before this field existed builds the model class it
    was run under."""
    assert ArchitectureConfig("t", 2, 8).descriptor_coordinates == "legacy"
    for name in sorted(alec.ARCHITECTURES):
        assert alec.ARCHITECTURES[name].descriptor_coordinates == "legacy", name


def test_an_unknown_coordinate_set_is_refused():
    """A misspelled coordinate set is refused at construction: silently
    falling back to ``legacy`` would run the campaign on the coordinates it
    was written to replace."""
    with pytest.raises(ValueError, match="descriptor_coordinates"):
        ArchitectureConfig("t", 2, 8, descriptor_coordinates="DFS")


@pytest.mark.parametrize("arch_name", _GGA_ARCHS)
def test_legacy_coordinates_are_the_committed_forward(arch_name):
    """``"legacy"`` stated explicitly returns arrays IDENTICAL to the registry
    entry's at the same seed, and the forward is the documented one:
    ``1 + LOB(tanh(s)^2 MLP([s_mlp, *extras]))`` for exchange and
    ``[rs_mlp, s_mlp, x1, *extras]`` for the polarized correlation net, with
    ``s_mlp`` the Dick XCDiff compression when the architecture carries
    ``descriptor_log_transform`` and the raw reduced gradient otherwise.
    Compared bitwise on 96 random rows, with the final layer left at its
    Glorot values so the MLP's output -- and therefore the coordinates it was
    fed -- actually reaches the forward."""
    default = dataclasses.replace(alec.get_architecture(arch_name),
                                  use_polarized_correlation=True,
                                  zero_init_final_layer=False)
    explicit = dataclasses.replace(default, descriptor_coordinates="legacy")
    xnet_a, cnet_a = create_network_pair(default, seed=17)
    xnet_b, cnet_b = create_network_pair(explicit, seed=17)
    rows = _rows(96, default.n_extra_features)

    for net_a, net_b, polarized in ((xnet_a, xnet_b, False),
                                    (cnet_a, cnet_b, True)):
        got = _forward(net_a, rows, polarized)
        assert float(np.max(np.abs(got - 1.0))) > 1e-3, "the MLP must move F"
        np.testing.assert_array_equal(got, _forward(net_b, rows, polarized))
        np.testing.assert_array_equal(
            got, _rebuilt(net_a, explicit, rows, polarized))


# ---------------------------------------------------------------------------
# The dfs coordinates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch_name", _GGA_ARCHS)
def test_dfs_coordinates_are_the_reference_expressions(arch_name):
    """``"dfs"`` feeds the MLPs exactly the reference implementation's
    coordinates, rebuilt here from the expressions rather than read from the
    code: ``[x_s, *extras]`` for exchange and ``[x_0, x_1, x_s, *extras]`` for
    correlation. 96 random rows spanning six decades of density and reduced
    gradients to 6; measured agreement 0.0 (bitwise) against the 1e-15 the
    case asserts.

    The correlation half also states that the LEGACY coordinates do NOT
    reproduce the same forward, so the case is a statement about the dfs map
    rather than about the rebuild machinery. The exchange half carries no such
    statement on an architecture whose ``descriptor_log_transform`` is on:
    there the two maps agree by construction, the legacy exchange input
    already being the compressed reduced gradient and the exchange net reading
    no density under either. Where the flag is off they differ, which the next
    case measures.
    """
    arch = _arch(arch_name, "dfs")
    xnet, cnet = create_network_pair(arch, seed=23)
    legacy = _arch(arch_name, "legacy")
    rows = _rows(96, arch.n_extra_features)
    for net, polarized in ((xnet, False), (cnet, True)):
        got = _forward(net, rows, polarized)
        want = _rebuilt(net, arch, rows, polarized)
        assert np.all(np.isfinite(got))
        assert float(np.max(np.abs(got - 1.0))) > 1e-3, "the MLP must move F"
        np.testing.assert_allclose(got, want, rtol=1e-15, atol=0.0)
        other = _rebuilt(net, legacy, rows, polarized)
        differs = float(np.max(np.abs(other - got)))
        if polarized:
            assert differs > 1e-6, ("the correlation coordinates differ",
                                    differs)
        elif arch.descriptor_log_transform:
            np.testing.assert_array_equal(other, got)
        else:
            assert differs > 1e-6, ("the exchange coordinates differ", differs)


@pytest.mark.parametrize("arch_name", _MGGA_ARCHS)
def test_dfs_coordinates_log_transform_the_indicator_column(arch_name):
    """On the meta-GGA rung the MLP receives ``x_alpha = ln((alpha + 1) / 2)``
    in the indicator's column, not the raw clamped indicator.

    That is the reference implementation's eq. 10/12 input, and the networks'
    own comments record the raw clamped column as a deviation from it;
    ``"dfs"`` removes the deviation. The indicator is taken at the RAW value
    the stored column encodes, the same value the SCAN parent reads. Both
    uniform-gas gates are unchanged, which the rebuild states by keeping the
    STORED column in the gate while the MLP gets the transformed raw one.

    The alpha column spans the whole range the descriptor can produce,
    the smoothing floor ``width / 2 = 5e-6`` and the ceiling
    ``_ALPHA_MAX = 100`` included, since the raw reconstruction changes branch
    at the ceiling.
    """
    from xcquinox.alec.metagga import _ALPHA_MAX, _ALPHA_SMOOTHING_WIDTH

    arch = _arch(arch_name, "dfs")
    xnet, cnet = create_network_pair(arch, seed=29)
    idx = _alpha_index(xnet)
    assert idx is not None
    rho, sigma, zeta, features = _rows(64, arch.n_extra_features)
    column = np.geomspace(0.5 * _ALPHA_SMOOTHING_WIDTH, _ALPHA_MAX, 64)
    column[-1] = _ALPHA_MAX
    features = np.array(features, dtype=np.float64, copy=True)
    features[:, idx] = column
    rows = (rho, sigma, zeta, jnp.asarray(features))
    for net, polarized in ((xnet, False), (cnet, True)):
        got = _forward(net, rows, polarized)
        want = _rebuilt(net, arch, rows, polarized)
        assert float(np.max(np.abs(got - 1.0))) > 1e-3, "the MLP must move F"
        np.testing.assert_allclose(got, want, rtol=1e-15, atol=0.0)

    # And the transform is not a no-op: feeding the raw clamped column instead
    # moves the forward far above the tolerance above.
    legacy = _arch(arch_name, "legacy")
    raw = _rebuilt(xnet, legacy, rows, False)
    assert float(np.max(np.abs(raw - _forward(xnet, rows, False)))) > 1e-6


def test_dfs_coordinates_override_the_legacy_log_transform_flag():
    """``"dfs"`` states the coordinates outright: the reduced gradient reaches
    the MLP compressed even on an architecture whose ``descriptor_log_transform``
    is False, so the coordinate set is not the old flag under a new name.
    ``deep_notransform_3x16`` carries the flag off; under ``"dfs"`` its forward
    matches the compressed rebuild and differs from the raw one."""
    arch = _arch("deep_notransform_3x16", "dfs")
    assert arch.descriptor_log_transform is False
    xnet, _cnet = create_network_pair(arch, seed=31)
    rows = _rows(64, arch.n_extra_features)
    got = _forward(xnet, rows, False)
    assert float(np.max(np.abs(got - 1.0))) > 1e-3, "the MLP must move F"
    np.testing.assert_allclose(got, _rebuilt(xnet, arch, rows, False),
                               rtol=1e-15, atol=0.0)
    raw = _rebuilt(xnet, _arch("deep_notransform_3x16", "legacy"), rows, False)
    assert float(np.max(np.abs(raw - got))) > 1e-6


@pytest.mark.parametrize("arch_name", _GGA_ARCHS + _MGGA_ARCHS)
def test_the_mlp_input_widths_per_rung(arch_name):
    """The input widths the coordinate sets define.

    Exchange: ``1 + n_extra`` on both rungs and under both coordinate sets --
    the reference exchange network reads no density, and on the meta-GGA rung
    the indicator arrives in the descriptor extras rather than as a second
    head column, so ``"dfs"`` changes what the columns CARRY and not how many
    there are. Correlation with a polarized net: ``3 + n_extra`` under both,
    the head being ``[rs, s, x1]`` in ``legacy`` and ``[x_0, x_1, x_s]`` in
    ``"dfs"``. Equal widths are what keeps a checkpoint's leaf shapes the same
    across the two, which is exactly why the coordinate set has to be recorded
    beside the checkpoint rather than inferred from it.
    """
    for coordinates in ("legacy", "dfs"):
        arch = _arch(arch_name, coordinates)
        xnet, cnet = create_network_pair(arch, seed=37)
        n_extra = arch.n_extra_features
        assert xnet.net.layers[0].weight.shape[1] == 1 + n_extra, \
            (arch_name, coordinates, "xnet")
        assert cnet.net.layers[0].weight.shape[1] == 3 + n_extra, \
            (arch_name, coordinates, "cnet")


@pytest.mark.parametrize("lam", [2.0, 0.5, 3.0])
def test_the_dfs_exchange_net_is_invariant_under_uniform_density_scaling(lam):
    """The constraint the density-free exchange input enforces.

    Under ``rho -> lambda^3 rho`` and ``sigma -> lambda^8 sigma`` the reduced
    gradient ``s = |grad rho| / (2 (3 pi^2)^(1/3) rho^(4/3))`` is unchanged, so
    an exchange network whose only density-derived input is ``x_s`` returns the
    SAME enhancement factor -- uniform coordinate scaling of the exchange
    functional (Levy and Perdew, Phys. Rev. A 32, 2010 (1985)). The anchored
    parent obeys it too, ``F_x^PBE`` being a function of ``s`` alone, so the
    whole anchored forward is invariant.

    The network's final layer is perturbed away from its zero initialization
    first, so the MLP -- and therefore its input coordinates -- reaches the
    output; at ``gated = 0`` the case would only be measuring that
    ``F_x^PBE`` is a function of ``s``.

    Bound 1e-15 relative. The identity is not bitwise even at ``lambda = 2``,
    where the rescalings themselves are exact: ``(8 rho)^(1/3)`` and
    ``2 rho^(1/3)`` differ by an ulp in the library's power, which moved 2 of
    64 rows by 6.66e-16 absolute (4.62e-16 relative) as measured. A network
    that read the density would move by O(1) instead, which is what the bound
    separates.
    """
    arch = _arch("deep_3x16", "dfs", anchor=True)
    xnet, _cnet = create_network_pair(arch, seed=41)
    xnet = _perturb_final_layer(xnet)
    rows = _rows(64, arch.n_extra_features)
    rho, sigma, _zeta, features = rows
    base = _forward(xnet, rows, False)
    assert float(np.max(np.abs(base - 1.0))) > 1e-3, "the MLP must move F_x"
    scaled = _forward(xnet, (rho * lam ** 3, sigma * lam ** 8, _zeta, features),
                      False)
    np.testing.assert_allclose(scaled, base, rtol=1e-15, atol=0.0)


def test_anchored_dfs_networks_still_return_the_parent_at_initialization():
    """The coordinates do not touch the anchor: at ``gated = 0`` an anchored
    ``"dfs"`` network returns ``F^parent`` to 1e-15 absolute, the same bound
    the legacy coordinates carry, because the coordinate map changes only what
    the MLP reads and the MLP's output is exactly zero at initialization."""
    from xcquinox.alec.tests.test_parent_anchor import _assert_is_the_parent

    arch = _arch("deep_3x16", "dfs", anchor=True)
    xnet, cnet = create_network_pair(arch, seed=43)
    rho, sigma, zeta, features = _rows(96, arch.n_extra_features)
    got_x = np.asarray(jax.vmap(
        lambda r, s, f: xnet(_pack_row(r, s, f)).squeeze())(rho, sigma, features))
    want_x = np.asarray(jax.vmap(parents.pbe_fx)(rho, sigma))
    _assert_is_the_parent(got_x, want_x, "dfs x")
    got_c = np.asarray(jax.vmap(
        lambda r, s, z, f: cnet(_pack_row_polarized(r, s, z, f)).squeeze()
    )(rho, sigma, zeta, features))
    want_c = np.asarray(jax.vmap(parents.pbe_fc)(rho, sigma, zeta))
    _assert_is_the_parent(got_c, want_c, "dfs c")


def test_the_anchored_dfs_forward_away_from_initialization():
    """The anchored ``"dfs"`` forward with a NON-zero final layer.

    ``config.anchored`` forces the zero initialization, so the case above
    measures the anchor and not the coordinates; here the final layer is
    perturbed and the whole forward -- dfs coordinates, uniform-gas gate,
    parent pre-image, output squash -- is rebuilt from the expressions and
    compared bitwise. This is the forward the campaign trains, one optimizer
    step in.
    """
    arch = _arch("deep_3x16", "dfs", anchor=True)
    xnet, cnet = create_network_pair(arch, seed=47)
    xnet = _perturb_final_layer(xnet, seed=1)
    cnet = _perturb_final_layer(cnet, seed=2)
    rows = _rows(96, arch.n_extra_features)
    for net, polarized in ((xnet, False), (cnet, True)):
        got = _forward(net, rows, polarized)
        want = _rebuilt(net, arch, rows, polarized, parent="pbe")
        np.testing.assert_allclose(got, want, rtol=1e-15, atol=1e-16)
        parent_only = _rebuilt(net, arch, rows, polarized)
        assert float(np.max(np.abs(parent_only - got))) > 1e-3, \
            "the anchor must move the forward off the unanchored map"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _config_dict(**model_block):
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


def test_grid_config_parses_the_coordinate_set(tmp_path):
    """``model: {descriptor_coordinates: dfs}`` is read, and its absence
    leaves ``legacy``."""
    from xcquinox.alec.cluster.grid_config import load_grid_config

    cfg = load_grid_config(_write_config(
        tmp_path, _config_dict(descriptor_coordinates="dfs")))
    assert cfg.model.descriptor_coordinates == "dfs"
    plain = load_grid_config(_write_config(tmp_path, _config_dict(),
                                           name="plain.yaml"))
    assert plain.model.descriptor_coordinates == "legacy"


def test_grid_config_refuses_an_unknown_coordinate_set(tmp_path):
    """An unrecognised value is refused on the login node rather than per
    array task."""
    from xcquinox.alec.cluster.grid_config import load_grid_config

    with pytest.raises(ValueError, match="descriptor_coordinates"):
        load_grid_config(_write_config(
            tmp_path, _config_dict(descriptor_coordinates="xcdiff"),
            name="bad.yaml"))


def test_the_coordinate_set_reaches_the_training_specs(tmp_path):
    """The coordinate set is part of the architecture identity: the specs
    carry it, so a task builds the model class the run was configured for."""
    from xcquinox.alec.cluster.grid_config import ModelConfig
    from xcquinox.alec.cluster.domain import get_domain_profile
    from xcquinox.alec.cluster.spec_builder import build_training_specs
    from xcquinox.alec.tests.test_cluster_spec_builder import (
        _make_cfg, _make_ledger, _make_pool)

    cfg = _make_cfg(tmp_path)
    cfg = dataclasses.replace(
        cfg, sweep=dataclasses.replace(cfg.sweep, arch=("deep_3x16",)),
        use_polarized_correlation=True,
        model=ModelConfig(parent_anchor=True, descriptor_coordinates="dfs"))
    built = build_training_specs(_make_pool(), _make_ledger(), cfg,
                                 get_domain_profile("dfs_step7"),
                                 str(tmp_path / "run"))
    assert built, "no specs built"
    for _cell, spec in built:
        assert spec.arch.descriptor_coordinates == "dfs"
        assert spec.arch.parent_anchor is True


def test_the_architecture_describes_its_coordinate_set():
    """Two architectures that differ only in the coordinates are different
    model classes and must not serialize to the same description."""
    legacy = _arch("deep_3x16", "legacy")
    dfs = _arch("deep_3x16", "dfs")
    assert legacy.describe() != dfs.describe()
    assert dfs.describe().get("descriptor_coordinates") == "dfs"
    assert legacy.describe().get("descriptor_coordinates") == "legacy"
