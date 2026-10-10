"""The two MLP front ends: the fixed Fourier-feature map and the sine network
with SIREN's initialization, the class fields that state them, and the plain
network they leave unchanged (jax 0.10.2, equinox 0.13.8, x64)."""
import dataclasses
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import pytest  # noqa: E402

jax.config.update("jax_enable_x64", True)

from xcquinox.pipeline import config as C  # noqa: E402
from xcquinox.pipeline import networks as N  # noqa: E402
from xcquinox.pipeline.parents import pbe_fx  # noqa: E402


from xcquinox.pipeline.tests import test_checkpoint_class as _ckpt  # noqa: E402
from xcquinox.pipeline.tests import test_descriptor_coordinates as _coord  # noqa: E402
from xcquinox.pipeline.tests import test_parent_anchor as _pa  # noqa: E402
from xcquinox.pipeline.tests import test_validate_run as _vrt  # noqa: E402
from xcquinox.pipeline.tests.test_pretrain import tiny_pretrain_data_dir  # noqa: E402,F401


def _arch_style():
    """``tools/analysis/arch_style.py``, the figure tools' order and colour
    table, loaded by path (the directory is not a package)."""
    path = Path(__file__).resolve().parents[3] / "tools" / "analysis" / "arch_style.py"
    spec = importlib.util.spec_from_file_location("arch_style", path)
    sys.modules["arch_style"] = module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


AS = _arch_style()

_NEW = ("deep_ff_3x16", "deep_sine_3x16")
_FRONTS = {"deep_ff_3x16": dict(fourier_features=16, fourier_scale=1.0),
           "deep_sine_3x16": dict(activation="sine", omega_0=1.0)}
#: What each registry entry states beyond its front end: nothing; both keep
#: the legacy coordinates of every entry (a run's model block sets them).
_ENTRY_EXTRA = {"deep_ff_3x16": {}, "deep_sine_3x16": {}}
_DEEP = dict(dm_entropy_intensive=True, descriptor_log_transform=True)
_CAMPAIGN = dict(use_polarized_correlation=True,
                 descriptor_coordinates="paper", ueg_gate="x2")
_MLP0 = {"activation": "gelu", "omega_0": 1.0, "fourier_features": 0,
         "fourier_scale": 1.0, "fourier_seed": 0, "fourier_digest": None,
         "network": "mlp", "kan_grid": 0, "kan_order": 0, "kan_digest": None}
#: The ranges each base coordinate is divided by before the map (the page;
#: the spans over the campaign atoms' rho*w pretraining sample): x_s 4.76,
#: x_0 8.78, the legacy transformed r_s 5.72; x_1 = ln(spinscale) up to
#: ln 2 / 3; the raw spin scale up to 2^(1/3) - 1. The paper rows: the
#: exchange [x_s, *extras], the polarized correlation [x_0, x_1, x_s,
#: *extras]; a descriptor column by the range its descriptor declares.
_X_S, _X_0, _X_1 = 4.76, 8.78, math.log(2.0) / 3.0
_RS_T = 5.72
_SPIN = 2.0 ** (1.0 / 3.0) - 1.0
_SCALES = {"x": (_X_S,), "c": (_X_0, _X_1, _X_S)}
_WORDS = (r"activation|omega_0|fourier_features|fourier_scale|fourier_seed"
          r"|fourier_digest")


def _differing(a, b):
    """The front-end fields on which two readings differ, in their order."""
    return [f for f in _MLP0 if a[f] != b[f]]
#: deep_3x16's F_x, F_c at seed 0 per gate, as the library computes them
_ROWS = ((0.1, 0.02), (1.0, 0.5), (5.0, 12.0))
_PINS = {"tanh2": ([1.0065394293598022, 1.0004112048891511, 1.0001358150218094],
                   [0.9959153854463009, 0.9997472790030433, 0.9999159118142301]),
         "x2": ([1.0027110938328223, 1.0000445950237633, 1.0000086207481058],
                [0.9983074225900436, 0.9999725937239865, 0.9999946626535423])}


def _campaign(name):
    return dataclasses.replace(_coord._arch(name, "paper"), ueg_gate="x2")


def _leaves(tree):
    filtered = eqx.filter(tree, eqx.is_array)
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(filtered)]


def _draw(kind, seed, m, extras=(), sigma=1.0, polarized=True, coords="paper"):
    """The (m, d) frequency matrix the page states for network ``kind``:
    sigma N(0, 1) from numpy's RandomState(seed + role), role 0 for the
    exchange and 1 for the correlation network, each column divided by its
    coordinate's range (a descriptor column by the range its descriptor
    declares, ``extras``). The paper rows [x_s] and [x_0, x_1, x_s]; the
    legacy rows under the log transform [x_s] and [r_s transformed, x_s,
    (spin scale when polarized)]."""
    if coords == "paper":
        base = _SCALES[kind]
    elif kind == "x":
        base = (_X_S,)
    else:
        base = (_RS_T, _X_S) + ((_SPIN,) if polarized else ())
    scales = np.asarray(base + tuple(extras))
    rows = np.random.RandomState(seed + "xc".index(kind)).standard_normal(
        (m, len(scales)))
    return sigma * rows / scales[None, :]


def _digest(*matrices):
    """The first sixteen hex digits of the SHA-256 of the matrices' float64
    bytes in order, the class record's statement of the draw."""
    import hashlib
    sha = hashlib.sha256()
    for matrix in matrices:
        sha.update(np.asarray(matrix, dtype=np.float64).tobytes())
    return sha.hexdigest()[:16]


def _rows(n, s_max=6.0, seed=20261006):
    rng = np.random.default_rng(seed)
    rho, s = 10.0 ** rng.uniform(-4.0, 2.0, n), rng.uniform(0.0, s_max, n)
    zeta, k_f = rng.uniform(-0.9, 0.9, n), (3.0 * np.pi ** 2 * rho) ** (1 / 3)
    return list(zip(rho, (2.0 * k_f * rho * s) ** 2, zeta))


def _gelu(z):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * z * (1.0 + np.tanh(c * (z + 0.044715 * z ** 3)))


def _hand(net, kind, rows, front, act, omega=1.0):
    """F against numpy to 1e-13: paper coordinates, ``front``, act(omega (W0 v
    + b0)), the net's attention if any, act(W h + b), linear, x2 gate, map."""
    w = [np.asarray(layer.weight) for layer in net.net.layers]
    b = [np.asarray(layer.bias) for layer in net.net.layers]
    a = 1.804 if kind == "x" else 2.0
    for rho, sigma, zeta in rows:
        s = np.sqrt(sigma) / (2.0 * (3.0 * np.pi ** 2 * rho) ** (1 / 3) * rho)
        xs = (1.0 - np.exp(-s * s)) * np.log(s + 1.0)
        spin = 0.5 * ((1.0 + zeta) ** (4 / 3) + (1.0 - zeta) ** (4 / 3))
        v = np.array([xs] if kind == "x" else [
            np.log(rho ** (1 / 3) + 1e-5), np.log(spin + 1e-5), xs])
        h = act(omega * (w[0] @ front(v) + b[0]))
        if net.attention is not None:
            h = np.asarray(net.attention(jnp.asarray(h)))
        for wi, bi in zip(w[1:-1], b[1:-1]):
            # SIREN applies omega_0 in every sine layer; omega is 1 for gelu
            h = act(omega * (wi @ h + bi))
        z = xs * float((w[-1] @ h + b[-1])[0]) - math.log(a - 1.0)
        want = 1.0 + (a / (1.0 + math.exp(-z)) - 1.0)
        got = float(net(jnp.array([rho, sigma, zeta][: 2 + (kind == "c")])))
        assert abs(got - want) < 1e-13, (kind, rho, sigma, got, want)


def _verdict(accepted, call, exc=ValueError, words=_WORDS):
    """Returns when ``accepted``, else raises ``exc`` naming the field."""
    if accepted:
        return call()
    with pytest.raises(exc) as excinfo:
        call()
    text = str(excinfo.value)
    assert re.search(words, text) and "unexpected keyword" not in text, text


def test_the_fourier_map_is_the_stated_draw(tmp_path):
    """B = sigma N(0, 1) from numpy's RandomState(fourier_seed + role), (m, d),
    d the MLP width, each column divided by its coordinate's range (x_s 4.76;
    x_0 8.78, x_1 ln 2 / 3; a descriptor column by its declared range), alike at network seeds 0, 1,
    7, static (8 leaves), layer 0 (16, 2m); F is numpy via [sin(2 pi B v),
    cos(2 pi B v)] and the GELU MLP to 1e-13 (2.2e-16 on the present pair);
    m 8, sigma 2, seed 3 too; a seed-7 skeleton loads a pair."""
    camp = _campaign("deep_ff_3x16")
    pairs = [N.create_network_pair(camp, seed=seed) for seed in (0, 1, 7)]
    for net, kind in ((pair[i], "xc"[i]) for pair in pairs for i in (0, 1)):
        assert isinstance(net.fourier_b, tuple), type(net.fourier_b)
        assert np.array_equal(np.asarray(net.fourier_b), _draw(kind, 0, 16))
        assert net.net.layers[0].weight.shape == (16, 32)
        assert len(_leaves(net)) == 8
        assert np.abs(np.asarray(net.net.layers[1].weight)).max() <= 0.25
    other = C.ArchitectureConfig.from_spec(
        "t", 3, 16, fourier_features=8, fourier_scale=2.0, fourier_seed=3,
        **_CAMPAIGN)
    xo, co = N.create_network_pair(other, seed=0)
    assert np.array_equal(np.asarray(xo.fourier_b), _draw("x", 3, 8, sigma=2.0))
    assert np.array_equal(np.asarray(co.fourier_b), _draw("c", 3, 8, sigma=2.0))
    assert xo.net.layers[0].weight.shape == (16, 16)
    # the attention twin: the map precedes the block, which acts after the
    # first activation (the hand forward applies the net's own block)
    attn = dataclasses.replace(other, name="ta", attention=True, num_heads=4)
    for k, net in enumerate(N.create_network_pair(attn, seed=0)):
        b = _draw("xc"[k], 3, 8, sigma=2.0)
        assert net.attention is not None
        _hand(net, "xc"[k], _rows(12), lambda v, b=b: np.concatenate(
            [np.sin(2 * np.pi * (b @ v)), np.cos(2 * np.pi * (b @ v))]), _gelu)
    # the cusp pair joins each row at the ranges its descriptor declares,
    # 1 and 2 (the columns in [0, 1] and (-1, 1))
    cusp = C.ArchitectureConfig.from_spec(
        "tc", 3, 16, descriptors=["cusp"], fourier_features=8, fourier_seed=3,
        **_DEEP, **_CAMPAIGN)
    assert cusp.extra_feature_ranges == (1.0, 2.0)
    xcu, ccu = N.create_network_pair(cusp, seed=0)
    assert np.array_equal(np.asarray(xcu.fourier_b),
                          _draw("x", 3, 8, extras=(1.0, 2.0)))
    assert np.array_equal(np.asarray(ccu.fourier_b),
                          _draw("c", 3, 8, extras=(1.0, 2.0)))
    assert np.asarray(ccu.fourier_b).shape == (8, 5)
    # the registry entry itself, legacy coordinates under the log transform:
    # the rows [x_s] and [r_s transformed, x_s], the spin scale joining the
    # correlation row when polarized, each column scaled by its own range
    entry = C.get_architecture("deep_ff_3x16")
    for polarized in (False, True):
        xl, cl = N.create_network_pair(
            dataclasses.replace(entry, use_polarized_correlation=polarized), seed=0)
        assert np.array_equal(np.asarray(xl.fourier_b),
                              _draw("x", 0, 16, coords="legacy"))
        assert np.array_equal(np.asarray(cl.fourier_b),
                              _draw("c", 0, 16, coords="legacy", polarized=polarized))
        assert np.asarray(cl.fourier_b).shape == (16, 2 + polarized)
    for k, net in enumerate(N.create_network_pair(camp, seed=0)):
        b = _draw("xc"[k], 0, 16)
        _hand(net, "xc"[k], _rows(12), lambda v, b=b: np.concatenate(
            [np.sin(2 * np.pi * (b @ v)), np.cos(2 * np.pi * (b @ v))]), _gelu)
        path, rows = str(tmp_path / f"n{k}.eqx"), [r[: 2 + k] for r in _rows(12)]
        eqx.tree_serialise_leaves(path, net)
        skeleton = N.create_network_pair(camp, seed=7)[k]
        loaded = eqx.tree_deserialise_leaves(path, skeleton)
        out = [float(net(jnp.array(r))) for r in rows]
        assert [float(loaded(jnp.array(r))) for r in rows] == out
        assert [float(skeleton(jnp.array(r))) for r in rows] != out


def test_the_sine_network_is_the_hand_forward():
    """deep_sine_3x16 (campaign), omega_0 2.5 and an attention twin: F is numpy
    sin(omega_0 (W0 v + b0)) -> sin(W h + b) -> linear to 1e-13, per seed;
    ranges: layer 0 1/n_in (c: 1/3 vs 0.577, P = 3.5e-12),
    later sqrt(6/n_in)/omega_0, biases 1/sqrt(n_in); at omega_0 = 1 every later
    layer, the final linear included, reaches past the library's 0.25."""
    sine = _campaign("deep_sine_3x16")
    w25 = C.ArchitectureConfig.from_spec("deep_sinew_3x16", 3, 16, activation="sine",
                                         omega_0=2.5, **_DEEP, **_CAMPAIGN)
    attn = dataclasses.replace(w25, name="t", attention=True, num_heads=4)
    for arch, omega in ((sine, 1.0), (w25, 2.5), (attn, 2.5)):
        hidden = []
        for seed in (0, 1):
            pair = N.create_network_pair(arch, seed=seed)
            for net, kind in zip(pair, "xc"):
                _hand(net, kind, _rows(12), lambda v: v, np.sin, omega)
                w = [np.abs(np.asarray(layer.weight)) for layer in net.net.layers]
                b = [np.abs(np.asarray(layer.bias)) for layer in net.net.layers]
                n = [x.shape[1] for x in w]
                assert w[0].max() <= 1.0 / n[0], kind
                assert all(wi.max() <= math.sqrt(6.0 / ni) / omega
                           for wi, ni in zip(w[1:], n[1:])), kind
                assert all(bi.max() <= 1 / math.sqrt(ni) for bi, ni in zip(b, n))
                assert omega != 1.0 or min(wi.max() for wi in w[1:]) > 0.25
            hidden.append(np.asarray(pair[0].net.layers[1].weight))
        assert not np.array_equal(*hidden), arch.name


def test_the_constraints_hold():
    """Registered (tanh2) and campaign (x2): F = 1 at s = 0 to 1e-15 (exactly
    1 on the present pair); F in (0, a) on 200 rows to s = 50, the term live;
    zero-init outlives the SIREN draw; anchored, PBE (tracked V3, 8.9 s)."""
    rows = _rows(200, 50.0, seed=7)
    for name in _NEW:
        registered = C.get_architecture(name)
        for arch in (registered, _campaign(name)):
            k = 2 + arch.use_polarized_correlation
            xnet, cnet = N.create_network_pair(arch, seed=0)
            for rho in (1e-3, 0.1, 1.0, 50.0):
                assert abs(float(xnet(jnp.array([rho, 0.0]))) - 1.0) <= 1e-15
                assert abs(float(cnet(jnp.array([rho, 0.0, 0.3][:k]))) - 1) < 1e-15
            xr, cr = (jnp.asarray([r[:j] for r in rows]) for j in (2, k))
            fx, fc = np.asarray(jax.vmap(xnet)(xr)), np.asarray(jax.vmap(cnet)(cr))
            assert np.all((fx > 0) & (fx < 1.804)) and np.all((fc > 0) & (fc < 2))
            assert np.abs(fx - 1.0).max() > 1e-3, (name, arch.ueg_gate)
            zeroed = N.create_network_pair(
                dataclasses.replace(arch, zero_init_final_layer=True), seed=0)
            for net, batch in zip(zeroed, (xr, cr)):
                assert np.abs(np.asarray(jax.vmap(net)(batch)) - 1.0).max() <= 1e-15
        _pa.test_anchored_networks_return_the_parent_at_initialization(name)


def test_the_plain_network_is_unchanged():
    """deep_3x16 per gate is the library's own value, 8
    leaves per network (a layer loop is self.net bit for bit
    there, so a GELU path rewritten as that loop is invisible to any pin)."""
    base = C.get_architecture("deep_3x16")
    for gate, (fx, fc) in _PINS.items():
        arch = dataclasses.replace(base, ueg_gate=gate)
        for net, want in zip(N.create_network_pair(arch, seed=0), (fx, fc)):
            assert [float(net(jnp.array(r))) for r in _ROWS] == want, gate
            assert len(_leaves(net)) == 8


def test_the_registry_entries_are_deep_3x16_with_one_front_end():
    """deep_3x16 plus one front end (dataclass equality), 41 entries, own
    shown name, GGA rung, expanded key naming it, describe() carrying it, both
    figure orders after deep_gea_3x16 in #9e9ac8, #9ecae1."""
    from xcquinox.pipeline import arch_names, rungs
    base = C.ARCHITECTURES["deep_3x16"]
    accents = {v.lower() for v in AS.RUNG_ACCENT.values()}
    for (name, change), word, colour in zip(
            _FRONTS.items(), ("fourier", "sine"), ("#9e9ac8", "#9ecae1")):
        entry = C.ARCHITECTURES[name]
        assert entry == dataclasses.replace(base, name=name, **change,
                                            **_ENTRY_EXTRA[name]), name
        assert arch_names.DISPLAY_NAME[name] == name
        assert rungs.rung_of(name) == rungs.RUNG_GGA == AS.rung_of(name)
        assert word in arch_names.expanded_key(name).lower(), name
        assert all(entry.describe()[k] == v for k, v in change.items())
        assert AS.arch_color(name).lower() == colour
        others = {v.lower() for k, v in AS.ARCH_COLOR.items()
                  if not k.startswith(name[:-5])}
        assert colour not in others | accents, name
    # the digests are the record's statements of the draw and of the spline
    # grids, not fields
    assert all(base.describe()[k] == v for k, v in _MLP0.items()
               if not k.endswith("_digest"))
    for order in (AS._STORED_ORDER, AS._DISPLAY_ORDER, AS.ARCH_ORDER):
        g = order.index("deep_gea_3x16")
        assert order[g - 1:g + 3] == ("deep_attn_3x16", "deep_gea_3x16", *_NEW)


_NAN, _INF, _M = float("nan"), float("inf"), dict(fourier_features=16)
_BAD = (*(("activation", dict(activation=v)) for v in ("relu", "Sine", "")),
        *(("omega_0", dict(activation="sine", omega_0=v))
          for v in (0.0, -1.0, _NAN, _INF, True)),
        *(("fourier_features", dict(fourier_features=v)) for v in (-1, 1.5, True)),
        *(("fourier_scale", dict(_M, fourier_scale=v))
          for v in (0.0, -1.0, _NAN, _INF)),
        *(("fourier_seed", dict(_M, fourier_seed=v)) for v in (1.5, "0", True)),
        ("omega_0", dict(omega_0=2.0)), ("fourier_seed", dict(fourier_seed=3)),
        ("fourier_scale", dict(fourier_scale=2.0)),
        ("fourier_features|activation", dict(_M, activation="sine")),
        ("fourier_features", dict(_M, descriptor_log_transform=False)),
        # the correlation network draws from fourier_seed + 1, which must
        # stay a seed numpy accepts
        ("fourier_seed", dict(_M, fourier_seed=2 ** 32 - 1)))


def test_the_fields_are_validated():
    """from_spec, replace of an entry and both constructors refuse each case
    of _BAD naming the field (a bool is no number, as for gea_mu); valid
    values build; anchored() and apply_model_block keep them."""
    for kw in (dict(activation="sine", omega_0=2.5),
               dict(fourier_features=4, fourier_scale=0.5, fourier_seed=9,
                    descriptor_coordinates="paper"),
               dict(fourier_features=4),
               dict(fourier_features=4, fourier_seed=2 ** 32 - 2)):
        arch = C.ArchitectureConfig.from_spec("deep_t_3x16", 3, 16, **kw, **_DEEP)
        assert all(getattr(arch, f) == v for f, v in kw.items()), kw
    # the largest seed builds both networks (the correlation draw at 2^32 - 1)
    N.create_network_pair(dataclasses.replace(arch, use_polarized_correlation=True),
                          seed=0)
    plain, either = C.ARCHITECTURES["deep_3x16"], (ValueError, TypeError)
    for words, kw in _BAD:
        builds = [lambda: C.ArchitectureConfig.from_spec("t", 3, 16, **kw),
                  lambda: dataclasses.replace(plain, **kw)] + [
            lambda cls=cls: cls(n_extra_features=0, depth=3, nodes=16, **kw)
            for cls in (N.AlecGGA_XNet, N.AlecGGA_CNet)]
        for build in builds:
            _verdict(False, build, either, words)
    # a descriptor column not bounded by construction (the meta-GGA
    # indicator, the density-matrix statistics) refuses the map; the cusp
    # pair and the rung-3.5 occupancies carry it at their declared ranges
    for base in ("deep_mgga_3x16", "deep_dm_3x16"):
        assert None in C.ARCHITECTURES[base].extra_feature_ranges, base
        _verdict(False, lambda base=base: dataclasses.replace(
            C.ARCHITECTURES[base], fourier_features=16), ValueError,
            "fourier_features")
    for base in ("deep_cusp_3x16", "deep_rung35_3x16"):
        arch = dataclasses.replace(C.ARCHITECTURES[base], fourier_features=16)
        assert None not in arch.extra_feature_ranges, base
        N.create_network_pair(arch, seed=0)
    # the networks need one range per descriptor column with the map
    raw = dict(n_extra_features=2, depth=3, nodes=16, fourier_features=16,
               descriptor_log_transform=True)
    for cls in (N.AlecGGA_XNet, N.AlecGGA_CNet):
        _verdict(False, lambda cls=cls: cls(**raw), ValueError,
                 "extra_feature_ranges")
        _verdict(False, lambda cls=cls: cls(**raw, extra_feature_ranges=(1.0,)),
                 ValueError, "extra_feature_ranges")
        _verdict(False, lambda cls=cls: cls(**raw, extra_feature_ranges=(1.0, None)),
                 ValueError, "fourier_features")
        _verdict(False, lambda cls=cls: cls(**raw, extra_feature_ranges=(1.0, 0.0)),
                 ValueError, "extra_feature_ranges")
        cls(**raw, extra_feature_ranges=(1.0, 2.0))
    # the scales follow the resolved coordinate set and the row's composition;
    # a row that is not bounded (legacy without the transform) or cannot
    # carry the polarized row the paper coordinates need is refused
    from xcquinox.pipeline import fourier_features as FF
    _verdict(False, lambda: FF.scales_for("c", "legacy", False, True, ()),
             ValueError, "fourier_features")
    _verdict(False, lambda: FF.scales_for("c", "paper", True, False, ()),
             ValueError, "fourier_features")
    assert FF.scales_for("x", "legacy", True, False, (1.0,)) == (_X_S, 1.0)
    assert FF.scales_for("c", "legacy", True, False, (1.0, 2.0)) == (
        _RS_T, _X_S, 1.0, 2.0)
    assert FF.scales_for("c", "legacy", True, True, ()) == (_RS_T, _X_S, _SPIN)
    assert FF.scales_for("c", "dfs", True, True, (1.0,)) == (_X_0, _X_1, _X_S, 1.0)
    from xcquinox.pipeline.cluster.grid_config import ModelConfig
    block = ModelConfig(parent_anchor=True, descriptor_coordinates="paper",
                        ueg_gate="x2")
    for arch in map(C.get_architecture, _NEW):
        for out in (C.anchored(arch), C.apply_model_block(arch, block)):
            assert all(getattr(out, f) == getattr(arch, f)
                       for f in _MLP0 if not f.endswith("_digest"))


def test_the_class_record_and_the_loaders_state_the_front_end(
        tmp_path, request, monkeypatch):
    """The six front-end fields in both readings and the record, each class
    refused by the other two, a silent record or no metadata the default; the
    run_pretrain stamp; both loaders, the certificate keep check, validate_run."""
    from xcquinox.pipeline import checkpoint_class as K
    from xcquinox.pipeline.cluster import fidelity as fid
    from xcquinox.pipeline.pretrain import _metadata_preflight, run_pretrain
    from xcquinox.pipeline.train import _require_matching_model_class as load
    plain_class = K.model_class_of_arch(C.get_architecture("deep_3x16"))
    assert {f: plain_class[f] for f in _MLP0} == _MLP0
    assert tuple(K.MLP_FIELDS) == tuple(_MLP0)
    archs = {k: C.get_architecture(n) for k, n in zip("pfs", ("deep_3x16",) + _NEW)}
    # the Fourier entry on the paper coordinates with the polarized row they
    # need (the gate as registered): the scales and the digest follow the
    # resolved coordinate set
    archs["f"] = _coord._arch("deep_ff_3x16", "paper")
    mlp = {"p": _MLP0, **{k: dict(_MLP0, **_FRONTS[n]) for k, n in zip("fs", _NEW)}}
    mlp["f"]["fourier_digest"] = _digest(_draw("x", 0, 16), _draw("c", 0, 16))
    # the registry entry (legacy coordinates under the log transform,
    # unpolarized) states the digest of the legacy rows' matrices
    legacy = K.mlp_class_of(C.get_architecture("deep_ff_3x16"))
    assert legacy["fourier_digest"] == _digest(
        _draw("x", 0, 16, coords="legacy"),
        _draw("c", 0, 16, coords="legacy", polarized=False))
    assert legacy["fourier_digest"] != mlp["f"]["fourier_digest"]
    want = {k: K.model_class_of_arch(a) for k, a in archs.items()}
    ckpt = {k: str(tmp_path / f"c{k}.eqx") for k in archs}
    for k, arch in archs.items():
        assert {f: want[k][f] for f in _MLP0} == K.mlp_class_of(arch) == mlp[k], k
        assert K.model_class_of_model(_ckpt._model(arch)) == want[k], k
        assert K.is_legacy_class(want[k]) is (k == "p"), k
        _ckpt._write_checkpoint(ckpt[k], arch)
        rec = K.read_class_record(ckpt[k])
        assert {f: rec[f] for f in _MLP0} == mlp[k], k
    assert len({K.describe_class(w) for w in want.values()}) == 3
    record = Path(K.class_record_path(ckpt["p"]))
    for silent in (False, True):
        if silent:
            kept = json.loads(record.read_text())
            for f in _MLP0:
                kept.pop(f)
            record.write_text(json.dumps(kept))
        for k, j in ((k, j) for k in ("p" if silent else "pfs") for j in archs):
            _verdict(k == j, lambda: K.require_matching_class(ckpt[k], want[j]),
                     K.ModelClassMismatch)
    # the polarized correlation row the map requires reads polarized rows
    from xcquinox.pipeline.pretrain_data_gen import generate_pretrain_data_npz
    data = str(tmp_path / "data")
    os.makedirs(data)
    generate_pretrain_data_npz(data, atoms=(("He", 0),), basis="sto-3g",
                               grid_level=0, polarized=True, descriptors=True,
                               density_fit=False)
    md = run_pretrain(C.PretrainSpec(arch=archs["f"], data_dir=data, n_steps=2,
                                     checkpoint_dir=str(tmp_path / "pre"), seed=0))
    on_disk = json.loads((tmp_path / "pre" / "pretrain_metadata.json").read_text())
    assert {f: md[f] for f in _MLP0} == {f: on_disk[f] for f in _MLP0} == mlp["f"]
    load(str(tmp_path / "pre"), archs["f"])
    # the metadata describes spec.arch: a supplied pair of another class is
    # refused before anything is written
    _verdict(False, lambda: run_pretrain(
        C.PretrainSpec(arch=archs["f"], data_dir=data, n_steps=2,
                       checkpoint_dir=str(tmp_path / "guard"), seed=0),
        networks=N.create_network_pair(archs["s"], seed=0)), ValueError,
        "activation|fourier")
    # the directory is made before the pair is examined; nothing is written
    assert not any((tmp_path / "guard").glob("*")), list((tmp_path / "guard").glob("*"))
    # The keep check and the loaders read the architecture under the run's
    # model block: the Fourier entry's coordinates are paper (its block and
    # its metadata state them), the other two are read on the legacy ones.
    coords = {"p": "legacy", "f": "paper", "s": "legacy"}
    md_path = tmp_path / "pretrain_metadata.json"
    for stated in (None, "p", "f", "s"):
        extra = {} if stated is None else dict(mlp[stated])
        for k, arch in archs.items():
            base = {"depth": 3, "nodes": 16, "parent_anchor": False,
                    "descriptor_coordinates": coords[k], "ueg_gate": "tanh2"}
            md_path.write_text(json.dumps(dict(base, **extra)))
            ok = k == (stated or "p")
            _verdict(ok, lambda: load(str(tmp_path), arch))
            _verdict(ok, lambda: _metadata_preflight(metadata_path=str(md_path),
                                                     arch=arch))
            block = SimpleNamespace(parent_anchor=False,
                                    descriptor_coordinates=coords[k],
                                    ueg_gate="tanh2")
            cert = dict(extra, arch=arch.name, descriptor_coordinates=coords[k])
            got = fid.model_class_mismatches(
                SimpleNamespace(model=block, use_polarized_correlation=True),
                cert, arch.name)
            assert [m[0] for m in got] == (
                [] if ok else _differing(mlp[stated or "p"], mlp[k])), (
                    k, stated, got)
    md_path.unlink()
    for k, arch in archs.items():
        _verdict(k == "p", lambda: load(str(tmp_path), arch))
    for i, k in enumerate("fs"):
        name, vcfg = archs[k].name, _vrt._cfg()
        vcfg.sweep.arch = (name,)
        vcfg.model = SimpleNamespace(parent_anchor=False,
                                     descriptor_coordinates=coords[k],
                                     ueg_gate="tanh2")
        monkeypatch.setattr(_vrt.vr, "load_grid_config", lambda p, c=vcfg: c)
        run = _vrt._write_run(tmp_path / f"v{i}", [_vrt._spec_for(name)],
                              certificates=False)
        cert_path = Path(_vrt._write_certificate(run, name))
        cert0 = json.loads(cert_path.read_text())
        for extra in ({}, dict(mlp[k])):
            cert_path.write_text(json.dumps(
                dict(cert0, descriptor_coordinates=coords[k], **extra)))
            cert_path.with_name("pretrain_metadata.json").write_text(json.dumps(
                dict(use_polarized_correlation=True, parent_anchor=False,
                     descriptor_coordinates=coords[k], **extra)))
            named = [f for f in _vrt.vr.validate_run(run)[0]
                     if re.search(_WORDS, f)]
            certified = sum("certificate records" in f for f in named)
            assert (certified > 0, len(named) > certified) == (not extra,) * 2


def test_the_pretraining_step_runs():
    """One Adam step (1e-3) of _PretrainLoss per new exchange network: finite
    loss and gradients, one nonzero, 8 leaves; B still the stated draw (a leaf
    B would move); the sine network's weights all move."""
    from xcquinox.pipeline.pretrain import _PretrainLoss
    rho, s = jnp.asarray(np.geomspace(0.05, 5.0, 64)), jnp.linspace(0.0, 4.0, 64)
    sigma = (2.0 * (3.0 * jnp.pi ** 2 * rho) ** (1 / 3) * rho * s) ** 2
    rows, ref = jnp.stack([rho, sigma], axis=1), pbe_fx(rho, sigma) - 1.0
    loss_fn, optimizer = _PretrainLoss(), optax.adam(1e-3)
    for name in _NEW:
        xnet = N.create_network_pair(_campaign(name), seed=0)[0]
        loss, grads = eqx.filter_value_and_grad(loss_fn)(xnet, rows, ref)
        flat = _leaves(grads)
        assert np.isfinite(float(loss)) and all(np.isfinite(g).all() for g in flat)
        assert max(float(np.abs(g).max()) for g in flat) > 0.0, name
        params = eqx.filter(xnet, eqx.is_array)
        updates, _ = optimizer.update(eqx.filter(grads, eqx.is_array),
                                      optimizer.init(params), params)
        stepped = eqx.apply_updates(xnet, updates)
        assert len(_leaves(stepped)) == len(_leaves(xnet)) == 8, name
        assert np.isfinite(float(loss_fn(stepped, rows, ref))), name
        if name == "deep_ff_3x16":
            assert np.array_equal(np.asarray(stepped.fourier_b), _draw("x", 0, 16))
        else:
            assert all(np.abs(np.asarray(a.weight) - np.asarray(b.weight)).max()
                       > 0.0 for b, a in zip(xnet.net.layers, stepped.net.layers))


@pytest.mark.slow
def test_the_run_readers_accept_the_runs_they_build(tmp_path, monkeypatch):
    """Each new architecture through the production writers and readers: a
    two-step pretraining and its certificate stamp the front end (the six
    fields, the digest among them), the keep check and ``certificate_describes_run`` report no
    disagreement, ``completed_pretraining`` refuses on the gate alone (two
    steps do not pass it), a certificate without the field is reported by
    the keep check, and ``validate_run`` reports nothing on the run and a
    failure naming the field once the metadata is tampered."""
    import pickle
    from xcquinox.pipeline import checkpoint_class as K
    from xcquinox.pipeline.cluster import fidelity as fid
    from xcquinox.pipeline.cluster._pretrain import (completed_pretraining,
                                                     resolve_run_architecture)
    from xcquinox.pipeline.cluster.grid_config import pretrain_checkpoint_dir
    from xcquinox.pipeline.pretrain import run_pretrain
    from xcquinox.pipeline.pretrain_data_gen import generate_pretrain_data_npz
    data = tmp_path / "data"
    data.mkdir()
    generate_pretrain_data_npz(str(data), atoms=(("He", 0),), basis="sto-3g",
                               grid_level=0, polarized=True, descriptors=True,
                               density_fit=False)
    for name in _NEW:
        cfg = _pa._anchored_cfg(arch=(name,))
        cfg.model.parent_anchor = False
        cfg.model.descriptor_coordinates = "paper"
        cfg.model.ueg_gate = "x2"
        arch = resolve_run_architecture(cfg, C.get_architecture(name))
        assert arch.use_polarized_correlation and not arch.parent_anchor
        want = K.mlp_class_of(arch)
        run_dir = str(tmp_path / name)
        pre = pretrain_checkpoint_dir(run_dir, name)
        md = run_pretrain(C.PretrainSpec(arch=arch, data_dir=str(data),
                                         checkpoint_dir=pre, n_steps=2, seed=0))
        payload = fid.fidelity_certificate(cfg, run_dir, name,
                                           oracle_set=_pa._tiny_oracle_set())
        assert {f: md[f] for f in _MLP0} == {f: payload[f] for f in _MLP0} == want
        if name == "deep_ff_3x16":
            assert want["fourier_digest"] == _digest(_draw("x", 0, 16),
                                                     _draw("c", 0, 16))
        assert fid.model_class_mismatches(cfg, payload, name) == []
        assert not any(re.search(_WORDS, line) for line in
                       fid.certificate_describes_run(cfg, pre, name, payload))
        keep, reason = completed_pretraining(pre, cfg, name)
        assert keep is False and not re.search(_WORDS, reason), (name, reason)
        tampered = {k: v for k, v in payload.items() if k not in _MLP0}
        assert [m[0] for m in fid.model_class_mismatches(cfg, tampered, name)
                ] == _differing(_MLP0, want), name
        vcfg = _vrt._cfg()
        vcfg.sweep.arch = (name,)
        vcfg.inputs.basis = "def2-svp"
        vcfg.inputs.orientation_lock_strength = 0.0
        vcfg.model = cfg.model
        vcfg.use_polarized_correlation = True
        vcfg.pretrain = SimpleNamespace(n_steps=2)
        monkeypatch.setattr(_vrt.vr, "load_grid_config", lambda p, _c=vcfg: _c)
        spec = _vrt._spec_for(name, basis="def2-svp", arch_override=arch)
        os.makedirs(os.path.join(run_dir, "specs"), exist_ok=True)
        with open(os.path.join(run_dir, "specs", "spec_0000.spec"), "wb") as f:
            pickle.dump(spec, f)
        Path(run_dir, "resolved_config.yaml").write_text("placeholder: true\\n")
        with open(os.path.join(run_dir, "manifest.json"), "w") as f:
            json.dump({"width": 4,
                       "xcquinox_version": payload.get("xcquinox_version")}, f)
        failures, warnings, _n = _vrt.vr.validate_run(run_dir)
        assert not any(re.search(_WORDS, line) for line in failures + warnings), (
            name, failures, warnings)
        md_path = Path(pre, "pretrain_metadata.json")
        kept = json.loads(md_path.read_text())
        for f in _MLP0:
            kept.pop(f)
        md_path.write_text(json.dumps(kept))
        failures, _warnings, _n = _vrt.vr.validate_run(run_dir)
        assert any(f"metadata {f}" in line for f in _differing(_MLP0, want)
                   for line in failures), (name, failures)
