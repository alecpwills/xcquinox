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

import xcquinox.pipeline as pipeline
from xcquinox.pipeline import parents
from xcquinox.pipeline.config import ArchitectureConfig, anchored
from xcquinox.pipeline.models import _pack_row, _pack_row_polarized
from xcquinox.pipeline.networks import create_network_pair


#: The offset the reference implementation adds inside the density logarithm
#: (``dpyscfl/net.py``: ``self.loge = 1e-5``).
_LOGE = 1e-5

_GGA_ARCHS = tuple(name for name in sorted(pipeline.ARCHITECTURES)
                   if not ArchitectureConfig.is_meta_gga(pipeline.ARCHITECTURES[name]))
_MGGA_ARCHS = tuple(name for name in sorted(pipeline.ARCHITECTURES)
                    if ArchitectureConfig.is_meta_gga(pipeline.ARCHITECTURES[name]))


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
    from xcquinox.pipeline.metagga import _ALPHA_MAX, _ALPHA_SMOOTHING_WIDTH
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
    SCAN parent is evaluated at. Taking
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
    arch = dataclasses.replace(pipeline.get_architecture(name),
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
    for name in sorted(pipeline.ARCHITECTURES):
        assert pipeline.ARCHITECTURES[name].descriptor_coordinates == "legacy", name


def test_an_unknown_coordinate_set_is_refused():
    """A misspelled coordinate set is refused at construction: silently
    falling back to ``legacy`` would run the campaign on the coordinates it
    was written to replace."""
    with pytest.raises(ValueError, match="descriptor_coordinates"):
        ArchitectureConfig("t", 2, 8, descriptor_coordinates="DFS")


# ---------------------------------------------------------------------------
# The dfs coordinates
# ---------------------------------------------------------------------------


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
    from xcquinox.pipeline.tests.test_parent_anchor import _assert_is_the_parent

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _config_dict(**model_block):
    from xcquinox.pipeline.tests.test_cluster_grid_config import _base_config_dict
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
    from xcquinox.pipeline.cluster.grid_config import load_grid_config

    cfg = load_grid_config(_write_config(
        tmp_path, _config_dict(descriptor_coordinates="dfs")))
    assert cfg.model.descriptor_coordinates == "dfs"
    plain = load_grid_config(_write_config(tmp_path, _config_dict(),
                                           name="plain.yaml"))
    assert plain.model.descriptor_coordinates == "legacy"


def test_the_coordinate_set_reaches_the_training_specs(tmp_path):
    """The coordinate set is part of the architecture identity: the specs
    carry it, so a task builds the model class the run was configured for."""
    from xcquinox.pipeline.cluster.grid_config import ModelConfig
    from xcquinox.pipeline.cluster.domain import get_domain_profile
    from xcquinox.pipeline.cluster.spec_builder import build_training_specs
    from xcquinox.pipeline.tests.test_cluster_spec_builder import (
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


# ---------------------------------------------------------------------------
# The published clone's coordinates and uniform-gas gate
# ---------------------------------------------------------------------------

class _SelectColumn(eqx.Module):
    """A stand-in for a network's MLP returning one column of the vector it is
    handed, so the coordinate at that index is read straight off the forward."""
    index: int = eqx.field(static=True)

    def __call__(self, netinp):
        return netinp[self.index:self.index + 1]


class _Unit(eqx.Module):
    """A stand-in for a network's MLP returning one, so the forward carries
    the uniform-gas gate alone."""

    def __call__(self, netinp):
        return jnp.ones((1,))


def _with_probe(net, probe):
    """``net`` with ``probe`` in place of its MLP."""
    return eqx.tree_at(lambda m: m.net, net, probe)


def _gated_value(net, row, *, polarized):
    """The gated MLP output the forward carries, recovered by inverting the
    network's own bounded map (``parents.lob_preimage``)."""
    rho, sigma, zeta = row
    packed = (_pack_row_polarized(rho, sigma, zeta, jnp.zeros((0,))) if polarized
              else _pack_row(rho, sigma, jnp.zeros((0,))))
    return float(parents.lob_preimage(net(packed).squeeze(), net.lobf.limit))


def _separated_rows(n=24, seed=20260923):
    """Rows whose reduced gradient stays away from zero, so the uniform-gas
    gate is of order one and the gated value divides by it without loss, and
    whose polarizations span the interval."""
    rng = np.random.default_rng(seed)
    rho = 10.0 ** rng.uniform(-3.0, 1.0, size=n)
    s = rng.uniform(0.5, 4.0, size=n)
    k_F = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
    sigma = (s * 2.0 * k_F * rho) ** 2
    zeta = rng.uniform(-1.0, 1.0, size=n)
    zeta[0] = 0.0
    return [(float(rho[i]), float(sigma[i]), float(zeta[i])) for i in range(n)]


def _spinscale(zeta):
    return 0.5 * ((1.0 + zeta) ** (4.0 / 3.0) + (1.0 - zeta) ** (4.0 / 3.0))


def test_paper_coordinates_add_the_epsilon_to_x1():
    """Under ``"paper"`` the correlation network's spin coordinate is
    ``ln(spinscale + 1e-5)``, the offset the published preprocessing puts
    inside BOTH logarithms; under ``"dfs"`` it is ``ln(spinscale)``.

    Oracle: the published ``preprocessing.transform_inputs_for_pyscf_grad_rho``
    (``eps_log = 1e-5``; ``x1 = log(zeta_prime + eps_log)``), against the
    coordinate read off the forward with the MLP replaced by the column
    selector at index 1 and the bounded map inverted.
    """
    rows = _separated_rows()
    for coordinates, expected in (("paper", lambda z: np.log(_spinscale(z) + _LOGE)),
                                  ("dfs", lambda z: np.log(_spinscale(z)))):
        arch = _arch("deep_3x16", coordinates)
        _xnet, cnet = create_network_pair(arch, seed=31)
        probed = _with_probe(cnet, _SelectColumn(index=1))
        for rho, sigma, zeta in rows:
            gate = float(np.tanh(np.sqrt(sigma)
                                 / (2.0 * (3.0 * np.pi ** 2 * rho) ** (1 / 3) * rho)) ** 2)
            got = _gated_value(probed, (rho, sigma, zeta), polarized=True) / gate
            assert abs(got - float(expected(zeta))) < 1e-12, (
                coordinates, rho, sigma, zeta)

    # The offset is what separates the two coordinate sets: at zeta = 0 the
    # DFS coordinate is exactly zero and the published one is not.
    arch = _arch("deep_3x16", "paper")
    _xnet, cnet = create_network_pair(arch, seed=31)
    probed = _with_probe(cnet, _SelectColumn(index=1))
    assert abs(_gated_value(probed, rows[0], polarized=True)) > 0.0


def test_paper_coordinates_for_exchange_are_the_dfs_coordinates():
    """The published exchange network reads the transformed reduced gradient
    alone, which is what ``"dfs"`` already gives it: the two exchange networks
    agree bit for bit on the same leaves.

    Oracle: the two forwards themselves, compared exactly. The correlation
    networks are compared too, and must NOT agree, so the case cannot pass by
    the two coordinate sets being the same everywhere.
    """
    rows = _separated_rows()
    xnet_paper, cnet_paper = create_network_pair(_arch("deep_3x16", "paper"),
                                                 seed=31)
    xnet_dfs, cnet_dfs = create_network_pair(_arch("deep_3x16", "dfs"), seed=31)
    for rho, sigma, _zeta in rows:
        packed = _pack_row(rho, sigma, jnp.zeros((0,)))
        np.testing.assert_array_equal(np.asarray(xnet_paper(packed)),
                                      np.asarray(xnet_dfs(packed)))
    differs = [abs(_gated_value(_with_probe(cnet_paper, _SelectColumn(index=1)),
                                row, polarized=True)
                   - _gated_value(_with_probe(cnet_dfs, _SelectColumn(index=1)),
                                  row, polarized=True))
               for row in rows]
    assert max(differs) > 0.0


def test_the_x2_gate_multiplies_the_network_by_the_transformed_reduced_gradient():
    """``ueg_gate="x2"`` puts the published prefactor
    ``(1 - exp(-s^2)) ln(1 + s)`` in front of the MLP in both networks, in
    place of ``tanh(s)^2``.

    Oracle: the published ``models.py`` forward (``lobterm =
    self.lobf(x2 * netterm)``) with ``x2`` the third transformed input of
    ``preprocessing.transform_inputs_for_pyscf_grad_rho``, against the gate
    read off the forward with the MLP replaced by the constant one.
    """
    rows = _separated_rows()
    for gate_name, expected in (("x2", _x_s), ("tanh2", lambda s: jnp.tanh(s) ** 2)):
        arch = dataclasses.replace(_arch("deep_3x16", "paper"),
                                   ueg_gate=gate_name)
        xnet, cnet = create_network_pair(arch, seed=31)
        for rho, sigma, zeta in rows:
            s = _reduced_gradient(rho, sigma)
            want = float(jnp.atleast_1d(expected(s)).flatten()[0])
            got_x = _gated_value(_with_probe(xnet, _Unit()), (rho, sigma, zeta),
                                 polarized=False)
            got_c = _gated_value(_with_probe(cnet, _Unit()), (rho, sigma, zeta),
                                 polarized=True)
            assert abs(got_x - want) < 1e-12, (gate_name, rho, sigma)
            assert abs(got_c - want) < 1e-12, (gate_name, rho, sigma)


def test_the_x2_gate_is_refused_on_the_meta_gga_rung():
    """The meta-GGA gate is the reference implementation's
    ``x2 + tanh^2(x3)``, which already carries ``x2``; asking for the GGA
    ``x2`` gate there would drop the iso-orbital term in silence.

    Oracle: the constructors, seen to raise.
    """
    from xcquinox.pipeline.networks import AlecGGA_CNet, AlecGGA_XNet

    with pytest.raises(ValueError, match="ueg_gate"):
        AlecGGA_XNet(n_extra_features=1, depth=3, nodes=16, meta_gga=True,
                     metagga_alpha_index=0, ueg_gate="x2")
    with pytest.raises(ValueError, match="ueg_gate"):
        AlecGGA_CNet(n_extra_features=1, depth=3, nodes=16, meta_gga=True,
                     metagga_alpha_index=0, use_spin_polarization=True,
                     ueg_gate="x2")
    with pytest.raises(ValueError, match="ueg_gate"):
        AlecGGA_XNet(n_extra_features=0, depth=3, nodes=16, ueg_gate="X2")


def test_the_gate_and_the_coordinates_reach_the_networks_from_the_model_block():
    """A run's ``model:`` block carries both fields to the built networks
    through the one helper every resolver of an architecture uses.

    Oracle: the static fields of the networks the block's architecture
    builds, and the architecture's own description.
    """
    from xcquinox.pipeline.cluster.grid_config import ModelConfig
    from xcquinox.pipeline.config import apply_model_block

    block = ModelConfig(ueg_gate="x2", descriptor_coordinates="paper")
    arch = apply_model_block(_arch("deep_3x16", "legacy"), block)
    assert arch.descriptor_coordinates == "paper"
    assert arch.ueg_gate == "x2"
    assert arch.describe().get("ueg_gate") == "x2"

    xnet, cnet = create_network_pair(arch, seed=31)
    assert xnet.descriptor_coordinates == "paper"
    assert cnet.descriptor_coordinates == "paper"
    assert xnet.ueg_gate == "x2"
    assert cnet.ueg_gate == "x2"

    default = create_network_pair(_arch("deep_3x16", "dfs"), seed=31)[0]
    assert default.ueg_gate == "tanh2"


@pytest.mark.parametrize("arch_name", ("deep_cusp_3x16", "deep_geom_3x16",
                                       "deep_geom_attn_3x16"))
def test_the_geometric_architectures_pretrain_on_the_published_inputs_plus_the_cusp_pair(
        arch_name):
    """A descriptor-carrying architecture under the published coordinates
    keeps the published MLP inputs and appends its descriptor columns: the
    exchange MLP reads ``[x2, cusp0, cusp1]`` and the correlation MLP
    ``[x0, x1, x2, cusp0, cusp1]``.

    The statement is made on each of the three cusp-carrying 3x16 entries.
    The attention block sits after the first layer, so it changes neither
    MLP's input width, and the assembled block is read off the architecture's
    descriptor list, which is the same two-column cusp block on all three.

    Oracle: the first linear layer's input width of each network, and the
    column count ``pretrain._assemble_pretrain_descriptors`` assembles for the
    exchange network from a file carrying a two-column cusp block.
    """
    from xcquinox.pipeline.pretrain import _assemble_pretrain_descriptors

    arch = _arch(arch_name, "paper")
    xnet, cnet = create_network_pair(arch, seed=31)
    assert xnet.net.layers[0].in_features == 3
    assert cnet.net.layers[0].in_features == 5

    pretrain_data = {
        "rho_all": jnp.asarray([0.5, 0.1, 2.0]),
        "sigma_all": jnp.asarray([0.2, 0.5, 5.0]),
        "zeta_all": jnp.asarray([0.0, -0.8, 0.3]),
        "cusp_all": jnp.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
    }
    assembled = _assemble_pretrain_descriptors(arch, pretrain_data)
    assert assembled.shape == (3, 4)
    np.testing.assert_array_equal(np.asarray(assembled[:, 2:]),
                                  np.asarray(pretrain_data["cusp_all"]))
