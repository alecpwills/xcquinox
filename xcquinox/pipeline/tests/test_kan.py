"""The Kolmogorov-Arnold networks: the spline basis against the Cox-de Boor
recursion, the forward against its numpy form, the bounded inputs and the
rows beyond them, the constraints and the anchor, the class record, the
registry entries and the training step (jax 0.10.2, equinox 0.13.8, x64)."""
import dataclasses
import importlib
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


def _arch_style():
    """``tools/analysis/arch_style.py``, the figure tools' order and colour
    table, loaded by path (the directory is not a package)."""
    path = Path(__file__).resolve().parents[3] / "tools" / "analysis" / "arch_style.py"
    spec = importlib.util.spec_from_file_location("arch_style", path)
    sys.modules["arch_style"] = module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


AS = _arch_style()


def _kan():
    """The module the item adds."""
    return importlib.import_module("xcquinox.pipeline.kan")


_NEW = ("deep_kan_2x6", "deep_kan_geom_2x6")
#: Each entry is the plain or the geometric 3x16 entry at depth 2 and width
#: 6 with the KAN in place of the MLP, grid 5 and order 3.
_BASE = {"deep_kan_2x6": "deep_3x16", "deep_kan_geom_2x6": "deep_geom_3x16"}
_G, _K = 5, 3
_KAN = dict(network="kan", kan_grid=_G, kan_order=_K)
_DEEP = dict(dm_entropy_intensive=True, descriptor_log_transform=True)
_CAMPAIGN = dict(use_polarized_correlation=True,
                 descriptor_coordinates="paper", ueg_gate="x2")
#: The class record's front end: the six fields of the MLP front ends and
#: the four of the spline network. A
#: record that states none of them states the MLP, network "mlp" at grid 0
#: and order 0 with no digest; a KAN record states its grid, its order and
#: the digest of its first layers' bounds (the knots a skeleton would
#: otherwise take from the module silently).
_MLP0 = {"activation": "gelu", "omega_0": 1.0, "fourier_features": 0,
         "fourier_scale": 1.0, "fourier_seed": 0, "fourier_digest": None,
         "network": "mlp", "kan_grid": 0, "kan_order": 0, "kan_digest": None}
_KAN0 = dict(_MLP0, network="kan", kan_grid=_G, kan_order=_K)
_WORDS = r"network|kan_grid|kan_order|kan_digest"
#: The first layer's bounds: x_s, x_0 and the legacy transformed r_s span
#: the campaign atoms' rho*w pretraining sample (0.011 to 4.770, -6.185 to
#: 2.600, 9.6e-5 to 5.716, the sample the Fourier map's ranges were measured
#: on), the lower ends of x_s and r_s at their analytic zero and the upper
#: ends rounded outward to the hundredth; x_1 = ln(spinscale + 1e-5) and
#: the raw spin scale over their analytic ranges (the paper row's offset
#: puts full polarization 7.9e-6 past ln 2 / 3). The hidden layers read
#: tanh, on [-1, 1]; the cusp and rung-3.5 columns the bounds their
#: descriptors declare.
_BOUNDS = {"x_s": (0.0, 4.77), "x_0": (-6.19, 2.60),
           "x_1": (0.0, math.log(2.0 ** (1.0 / 3.0) + 1e-5)),
           "rs_t": (0.0, 5.72), "spinscale": (1.0, 2.0 ** (1.0 / 3.0))}
_HIDDEN = (-1.0, 1.0)
_CUSP = ((0.0, 1.0), (-1.0, 1.0))
_RUNG35 = ((0.0, 1.0), (0.0, 1.0))
_EXPANDED = ("Kolmogorov-Arnold network, cubic B-splines on 5 intervals per "
             "edge, SiLU base")
#: The colours chosen as the front ends' were, by the largest worst-case
#: CIEDE2000 separation from every colour a same-figure architecture, rung
#: accent or band can carry: tab20c's #e6550d (12.49, against tab10's
#: orange) and, with it carried, tab20c's #bdbdbd (12.40, against the GGA
#: rung band).
_COLOUR = {"deep_kan_2x6": "#e6550d", "deep_kan_geom_2x6": "#bdbdbd"}
#: Parameters per network: (6 d_in + 36 + 6) edges times G + k + 1 = 9,
#: (exchange, correlation) per entry, coordinate set and polarization. The
#: exchange row has d_in 1, 3 with the cusp pair; the correlation row 2
#: (legacy), 3 (legacy polarized, paper), two more with the cusp pair.
_PARAMS = {("deep_kan_2x6", "legacy", False): (432, 486),
           ("deep_kan_2x6", "legacy", True): (432, 540),
           ("deep_kan_2x6", "paper", True): (432, 540),
           ("deep_kan_geom_2x6", "legacy", False): (540, 594),
           ("deep_kan_geom_2x6", "legacy", True): (540, 648),
           ("deep_kan_geom_2x6", "paper", True): (540, 648)}


def _differing(a, b):
    """The front-end fields on which two readings differ, in their order."""
    return [f for f in _MLP0 if a[f] != b[f]]


def _digest(*bounds):
    """The first sixteen hex digits of the SHA-256 of the bounds' float64
    bytes in order, the record's statement of the grids."""
    import hashlib
    sha = hashlib.sha256()
    for b in bounds:
        sha.update(np.asarray(b, dtype=np.float64).tobytes())
    return sha.hexdigest()[:16]


def _kan_class(coords, polarized, extras=(), grid=_G, order=_K):
    """The class record's front end of a KAN architecture on the resolved
    row: the digest covers every layer's bounds, the exchange network's
    three layers (the row's bounds, then twice six inputs on [-1, 1]) then
    the correlation network's."""
    layers = []
    for kind in "xc":
        layers.append(_row_bounds(kind, coords, polarized, extras))
        layers.extend([(_HIDDEN,) * 6] * 2)
    return dict(_KAN0, kan_grid=grid, kan_order=order,
                kan_digest=_digest(*layers))


def _leaves(tree):
    filtered = eqx.filter(tree, eqx.is_array)
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(filtered)]


def _count(kan):
    """``KAN.parameter_count``, an attribute or a method."""
    value = kan.parameter_count
    return int(value() if callable(value) else value)


def _zeroed(KN, kan):
    """``zeroed_last_layer``, a function of the module or a method."""
    fn = getattr(KN, "zeroed_last_layer", None)
    return fn(kan) if fn is not None else kan.zeroed_last_layer()


def _campaign(name):
    return dataclasses.replace(_coord._arch(name, "paper"), ueg_gate="x2")


def _verdict(accepted, call, exc=ValueError, words=_WORDS):
    """Returns when ``accepted``, else raises ``exc`` naming the field."""
    if accepted:
        return call()
    with pytest.raises(exc) as excinfo:
        call()
    text = str(excinfo.value)
    assert re.search(words, text) and "unexpected keyword" not in text, text


def _sigma(rho, s):
    """sigma = (2 k_F rho s)^2: the row of density ``rho`` at reduced
    gradient ``s``."""
    k_f = (3.0 * math.pi ** 2 * rho) ** (1.0 / 3.0)
    return (2.0 * k_f * rho * s) ** 2


def _np_xs(s):
    return (1.0 - math.exp(-s * s)) * math.log(s + 1.0)


def _s_of_xs(target):
    """The reduced gradient at which x_s = (1 - e^{-s^2}) ln(1 + s), an
    increasing function, takes ``target`` (bisection)."""
    a, b = 0.0, 1e6
    for _ in range(200):
        m = 0.5 * (a + b)
        a, b = (m, b) if _np_xs(m) < target else (a, m)
    return 0.5 * (a + b)


def _rows(n, s_max=6.0, seed=20261007):
    rng = np.random.default_rng(seed)
    rho, s = 10.0 ** rng.uniform(-4.0, 2.0, n), rng.uniform(0.0, s_max, n)
    zeta = rng.uniform(-0.9, 0.9, n)
    return [(float(r), _sigma(r, si), float(z)) for r, si, z in zip(rho, s, zeta)]


def _edge_rows():
    """Rows at and beyond the first layer's bounds: x_0 below its bound
    (rho 1e-9) and in its upper extension (1e5) and past the extended grid
    (2e10, x_0 = 7.91 against 2.60 + 3h = 7.87); x_s in its upper extension
    (s 200, x_s 5.30) and past it (s 1e4, 9.21 against 7.63); full
    polarization (the paper row's x_1 = ln(2^(1/3) + 1e-5) lies 7.9e-6 past
    ln 2 / 3); the density floor; s = 0."""
    return [(1e-9, _sigma(1e-9, 0.5), 1.0), (1e5, _sigma(1e5, 0.2), -1.0),
            (2e10, _sigma(2e10, 0.1), 0.0), (0.5, _sigma(0.5, 200.0), 0.0),
            (0.5, _sigma(0.5, 1e4), 0.3), (1e-14, 1e-30, 0.5), (0.2, 0.0, 0.0)]


def _cusp_features(n, seed=11):
    """Cusp columns over their declared bounds, [0, 1] and (-1, 1), the
    bounds themselves among them."""
    rng = np.random.default_rng(seed)
    f = np.stack([rng.uniform(0.0, 1.0, n), rng.uniform(-1.0, 1.0, n)], axis=1)
    f[:3] = ((0.0, -0.999999), (1.0, 0.999999), (0.5, 0.0))[: min(3, n)]
    return f


# ---------------------------------------------------------------------------
# The basis and the forward, written out here
# ---------------------------------------------------------------------------

def _knots(lo, hi, g, k):
    """The extended uniform knot vector t_j = lo + (j - k) h, j = 0..G + 2k."""
    return lo + (np.arange(g + 2 * k + 1) - k) * ((hi - lo) / g)


def _np_orders(x, t, k):
    """[B_0, ..., B_k] at the points ``x`` (Cox-de Boor), B_p of shape
    (n, len(t) - 1 - p); B_k is the order-k basis, G + k functions."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    B = ((x[:, None] >= t[None, :-1]) & (x[:, None] < t[None, 1:])).astype(float)
    out = [B]
    for p in range(1, k + 1):
        B = ((x[:, None] - t[None, :-(p + 1)])
             / (t[None, p:-1] - t[None, :-(p + 1)]) * B[:, :-1]
             + (t[None, p + 1:] - x[:, None])
             / (t[None, p + 1:] - t[None, 1:-p]) * B[:, 1:])
        out.append(B)
    return out


def _np_deriv(x, lo, hi, g, k, r):
    """The r-th derivative of the order-k basis between the knots, from the
    B-spline derivative recursion applied r times upward from order k - r."""
    t = _knots(lo, hi, g, k)
    D = _np_orders(x, t, k)[k - r]
    for p in range(k - r + 1, k + 1):
        m = D.shape[1] - 1
        D = p * (D[:, :-1] / (t[p:p + m] - t[:m])
                 - D[:, 1:] / (t[p + 1:p + 1 + m] - t[1:1 + m]))
    return D


def _np_basis(x, lo, hi, g=_G, k=_K):
    return _np_orders([x], _knots(lo, hi, g, k), k)[k][0]


def _np_kan(params, bounds, v):
    """The KAN forward in numpy: per layer y_j = sum_i (w_b,ji silu(x_i) +
    sum_m c_jim B_m(x_i)) on the layer's bounds, tanh between layers, the
    last layer's sums out."""
    x = np.asarray(v, dtype=float)
    for i, ((coef, base), layer_bounds) in enumerate(zip(params, bounds)):
        B = np.stack([_np_basis(xj, lo, hi) for xj, (lo, hi) in
                      zip(x, layer_bounds)])
        y = np.einsum("oim,im->o", coef, B) + base @ (x / (1.0 + np.exp(-x)))
        x = np.tanh(y) if i < len(params) - 1 else y
    return float(x[0])


def _row_bounds(kind, coords, polarized, extras=()):
    """The first layer's bounds for the resolved row: the exchange [x_s,
    *extras]; the paper and dfs correlation [x_0, x_1, x_s, *extras]; the
    legacy correlation under the log transform [r_s transformed, x_s,
    (spin scale), *extras]."""
    b = _BOUNDS
    if kind == "x":
        return (b["x_s"],) + tuple(extras)
    if coords in ("paper", "dfs"):
        return (b["x_0"], b["x_1"], b["x_s"]) + tuple(extras)
    return ((b["rs_t"], b["x_s"]) + ((b["spinscale"],) if polarized else ())
            + tuple(extras))


def _np_coords(kind, coords, polarized, rho, sigma, zeta):
    """The base coordinates of a row as the networks compute them, and s."""
    rho, sigma = max(rho, 1e-12), max(sigma, 0.0)
    s = math.sqrt(sigma) / (2.0 * (3.0 * math.pi ** 2 * rho) ** (1.0 / 3.0) * rho)
    xs = _np_xs(s)
    if kind == "x":
        return [xs], s
    z = min(max(zeta, -1.0), 1.0)
    spin = 0.5 * ((1.0 + z) ** (4.0 / 3.0) + (1.0 - z) ** (4.0 / 3.0))
    if coords in ("paper", "dfs"):
        x1 = math.log(spin + 1e-5) if coords == "paper" else math.log(spin)
        return [math.log(rho ** (1.0 / 3.0) + 1e-5), x1, xs], s
    r_s = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    return [_np_xs(r_s), xs] + ([spin] if polarized else []), s


def _pack(kind, polarized, row, feats=None):
    rho, sigma, zeta = row
    return ([rho, sigma] + ([zeta] if kind == "c" and polarized else [])
            + ([] if feats is None else [float(c) for c in feats]))


def _layer_bounds(layer):
    return tuple(tuple(float(v) for v in pair) for pair in layer.bounds)


def _hand(net, kind, coords, polarized, gate, rows, feats=None):
    """F of ``net`` against numpy to 1e-13: the resolved row's coordinates,
    the edge functions on the stated grids (the first layer on the row's
    bounds, the hidden layers on [-1, 1] after tanh), the last layer's edge
    sum, the gate (tanh(s)^2 or x_s) and the bounded map (1.804 or 2)."""
    first = _row_bounds(kind, coords, polarized, _CUSP if feats is not None else ())
    layers = net.net.layers
    bounds = [first] + [(_HIDDEN,) * 6] * (len(layers) - 1)
    assert len(layers) == 3, len(layers)
    for layer, want in zip(layers, bounds):
        assert np.allclose(_layer_bounds(layer), want, rtol=1e-15, atol=0.0), (
            _layer_bounds(layer), want)
        assert (int(layer.grid), int(layer.order)) == (_G, _K)
    params = [(np.asarray(layer.coef), np.asarray(layer.base)) for layer in layers]
    a = 1.804 if kind == "x" else 2.0
    packed = jnp.asarray([_pack(kind, polarized, r, None if feats is None else feats[i])
                          for i, r in enumerate(rows)])
    got = np.asarray(jax.vmap(net)(packed))
    for i, row in enumerate(rows):
        v, s = _np_coords(kind, coords, polarized, *row)
        f = [] if feats is None else [float(c) for c in feats[i]]
        out = _np_kan(params, bounds, v + f)
        g = math.tanh(s) ** 2 if gate == "tanh2" else _np_xs(s)
        z = g * out - math.log(a - 1.0)
        want = 1.0 + (a / (1.0 + math.exp(-z)) - 1.0)
        assert abs(float(got[i]) - want) < 1e-13, (kind, coords, row, got[i], want)


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------

def test_the_spline_basis_is_the_stated_one():
    """``kan.bspline_basis`` against the Cox-de Boor written here (itself
    scipy's BSpline.basis_element to 3.3e-16) on the extended uniform knots
    t_j = lo + (j - k) h, j = 0..G + 2k: G + k functions, equal to 1e-14
    from lo - (k + 1) h to hi + (k + 1) h; their sum one on [lo, hi], the
    bounds included; zero at and beyond lo - k h and hi + k h; non-negative.
    C^(k-1) at every knot: under jax autodiff the r-th derivative's
    one-sided gap shrinks a hundredfold from e = 1e-4 h to 1e-6 h for r < k,
    and the k-th jumps by at least 1/h^k (6/h^3 at the cubic's interior
    knots); between the knots the derivatives are the B-spline derivative
    recursion's."""
    KN = _kan()
    for lo, hi in (*_BOUNDS.values(), _HIDDEN, _CUSP[0]):
        for g, k in ((_G, _K), (1, 1), (3, 2), (8, 3), (2, 1)):
            h = (hi - lo) / g
            t = _knots(lo, hi, g, k)

            def basis(x, lo=lo, hi=hi, g=g, k=k):
                return KN.bspline_basis(x, lo, hi, g, k)

            f = jax.vmap(basis)
            xs = np.concatenate([np.linspace(lo - (k + 1) * h,
                                             hi + (k + 1) * h, 801), t])
            got = np.asarray(f(jnp.asarray(xs)))
            assert got.shape == (xs.size, g + k), (lo, g, k, got.shape)
            assert np.abs(got - _np_orders(xs, t, k)[k]).max() <= 1e-14, (lo, g, k)
            assert got.min() >= -1e-15
            inside = np.concatenate([np.linspace(lo, hi, 401), [lo, hi]])
            unity = np.asarray(f(jnp.asarray(inside))).sum(axis=1)
            assert np.abs(unity - 1.0).max() <= 1e-14, (lo, g, k)
            steps = np.geomspace(1e-9, 10.0, 7) * h
            beyond = np.concatenate([[t[0], t[-1]], t[0] - steps, t[-1] + steps])
            assert np.abs(np.asarray(f(jnp.asarray(beyond)))).max() <= 1e-30
            if (lo, hi) not in (_BOUNDS["x_s"], _HIDDEN, _BOUNDS["x_1"]):
                continue
            mids = 0.5 * (t[:-1] + t[1:])
            for r in range(k + 1):
                d = basis
                for _ in range(r):
                    d = jax.jacfwd(d)
                d = jax.vmap(d)
                if r:
                    assert np.allclose(np.asarray(d(jnp.asarray(mids))),
                                       _np_deriv(mids, lo, hi, g, k, r),
                                       rtol=1e-10, atol=1e-10 / h ** r), (g, k, r)
                gaps = []
                for e in (1e-4, 1e-6):
                    plus = np.asarray(d(jnp.asarray(t + e * h)))
                    minus = np.asarray(d(jnp.asarray(t - e * h)))
                    gaps.append(np.abs(plus - minus).max(axis=1))
                if r < k:
                    assert np.all(gaps[1] <= 0.03 * gaps[0] + 1e-12 / h ** r), (
                        lo, g, k, r, gaps)
                else:
                    assert np.all(gaps[1] >= 0.99 / h ** k), (lo, g, k, gaps)


def test_the_network_is_the_hand_forward():
    """F of both entries equals the numpy forward to 1e-13 on random rows and
    on rows at and beyond the bounds: as registered (the legacy rows under
    the log transform, the zeta-blind correlation row, tanh^2), the legacy
    polarized row (the spin scale on [1, 2^(1/3)]), the paper rows with the
    x2 gate and the dfs rows (x_1 without the offset), seeds 0 and 1; the
    networks state network "kan", grid 5, order 3; ``bounds_for`` and
    ``COORDINATE_BOUNDS`` are the stated bounds. The initialization: the
    coefficients N(0, 0.1^2) (816 draws of the registered pair: std within
    0.088..0.112, |mean| < 0.02, the bounds of 4.8 and 5.7 standard errors),
    the base weights within +-sqrt(6/(n_in + n_out)) and uniform there (102
    draws: std of w/bound within 0.45..0.70 about 1/sqrt(3)), the draws the
    networks' own (a seed reproduces them; seeds and the two networks
    differ). A KAN built directly is the same forward, its output of shape
    (1,), and ``zeroed_last_layer`` zeroes the last layer alone, so the
    output is 0.0."""
    KN = _kan()
    stated = sorted(tuple(map(float, v)) for v in KN.COORDINATE_BOUNDS.values())
    assert len(stated) == len(_BOUNDS)
    assert np.allclose(stated, sorted(_BOUNDS.values()), rtol=1e-15, atol=0)
    for kind, coords, log, pol in (("x", "legacy", True, False),
                                   ("x", "paper", True, True),
                                   ("c", "legacy", True, False),
                                   ("c", "legacy", True, True),
                                   ("c", "paper", False, True),
                                   ("c", "dfs", True, True)):
        for extras in ((), _CUSP):
            got = KN.bounds_for(kind, coords, log, pol, extras)
            assert np.allclose(np.asarray(got, dtype=float),
                               _row_bounds(kind, coords, pol, extras),
                               rtol=1e-15, atol=0.0), (kind, coords, pol, got)
    rows = _rows(12) + _edge_rows()
    feats = _cusp_features(len(rows))
    for name in _NEW:
        entry = C.get_architecture(name)
        cases = (("legacy", False, "tanh2", entry),
                 ("legacy", True, "tanh2",
                  dataclasses.replace(entry, use_polarized_correlation=True)),
                 ("paper", True, "x2", _campaign(name)),
                 ("dfs", True, "tanh2",
                  dataclasses.replace(entry, use_polarized_correlation=True,
                                      descriptor_coordinates="dfs")))
        for coords, polarized, gate, arch in cases:
            for seed in (0, 1):
                for net, kind in zip(N.create_network_pair(arch, seed=seed), "xc"):
                    assert isinstance(net.net, KN.KAN), type(net.net)
                    assert (net.network, net.kan_grid, net.kan_order) == (
                        "kan", _G, _K)
                    _hand(net, kind, coords, polarized, gate, rows,
                          feats if "geom" in name else None)
    x0, c0 = N.create_network_pair(C.get_architecture("deep_kan_2x6"), seed=0)
    coefs, units = [], []
    for net, d_in in ((x0, 1), (c0, 2)):
        sizes = (d_in, 6, 6, 1)
        for i, layer in enumerate(net.net.layers):
            n_in, n_out = sizes[i], sizes[i + 1]
            coef, base = np.asarray(layer.coef), np.asarray(layer.base)
            assert coef.shape == (n_out, n_in, _G + _K), coef.shape
            assert base.shape == (n_out, n_in), base.shape
            bound = math.sqrt(6.0 / (n_in + n_out))
            assert np.abs(base).max() <= bound
            coefs.append(coef.ravel())
            units.append(base.ravel() / bound)
    c, u = np.concatenate(coefs), np.concatenate(units)
    assert c.size == 816 and 0.088 < c.std() < 0.112 and abs(c.mean()) < 0.02, (
        c.size, c.std(), c.mean())
    assert u.size == 102 and 0.45 < u.std() < 0.70, (u.size, u.std())
    # Each layer draws from its own key and the base weights from a key of
    # their own: no two layers' coefficient draws share a prefix (one key for
    # every layer repeats the exchange network's 48 first-layer values in its
    # last layer), and the base weights are uncorrelated with the leading
    # coefficients (drawn from the coefficients' key they correlate at 0.995
    # through the common uniform bits; the 102 independent pairs read |r|
    # of order 0.1).
    for net in (x0, c0):
        flat = [np.asarray(layer.coef).ravel() for layer in net.net.layers]
        for i in range(3):
            for j in range(i + 1, 3):
                n = min(flat[i].size, flat[j].size)
                assert not np.array_equal(flat[i][:n], flat[j][:n]), (i, j)
    bases = np.concatenate([np.asarray(layer.base).ravel()
                            for net in (x0, c0) for layer in net.net.layers])
    leading = np.concatenate([np.asarray(layer.coef).ravel()[: layer.base.size]
                              for net in (x0, c0) for layer in net.net.layers])
    assert bases.size == 102 and abs(np.corrcoef(bases, leading)[0, 1]) < 0.5
    again = N.create_network_pair(C.get_architecture("deep_kan_2x6"), seed=0)
    assert all(np.array_equal(a, b) for a, b in
               zip(_leaves(x0) + _leaves(c0), _leaves(again[0]) + _leaves(again[1])))
    other = N.create_network_pair(C.get_architecture("deep_kan_2x6"), seed=1)
    assert not np.array_equal(np.asarray(x0.net.layers[1].coef),
                              np.asarray(other[0].net.layers[1].coef))
    assert not np.array_equal(np.asarray(x0.net.layers[1].coef),
                              np.asarray(c0.net.layers[1].coef))
    bounds = (_BOUNDS["x_0"], _BOUNDS["x_1"], _BOUNDS["x_s"])
    kan = KN.KAN(in_size=3, width=6, depth=2, bounds=bounds, grid=_G,
                 order=_K, key=jax.random.PRNGKey(5))
    params = [(np.asarray(layer.coef), np.asarray(layer.base))
              for layer in kan.layers]
    layer_bounds = [bounds] + [(_HIDDEN,) * 6] * 2
    vs = np.random.default_rng(13).uniform(-8.0, 8.0, (16, 3))
    for v in vs:
        out = kan(jnp.asarray(v))
        assert out.shape == (1,)
        assert abs(float(out[0]) - _np_kan(params, layer_bounds, v)) < 1e-13
    zeroed = _zeroed(KN, kan)
    assert not np.asarray(zeroed.layers[-1].coef).any()
    assert not np.asarray(zeroed.layers[-1].base).any()
    for a, b in zip(kan.layers[:-1], zeroed.layers[:-1]):
        assert np.array_equal(np.asarray(a.coef), np.asarray(b.coef))
        assert np.array_equal(np.asarray(a.base), np.asarray(b.base))
    assert all(float(zeroed(jnp.asarray(v))[0]) == 0.0 for v in vs)


def test_a_row_beyond_the_grid_is_finite():
    """The basis is zero at and beyond the extended grid's ends, lo - k h and
    hi + k h (1e-30), and sums to one at the bounds themselves, where the
    nonzero functions are the three nearest at 1/6, 2/3, 1/6: an edge's
    spline part vanishes only beyond the extension (at the bounds it is a
    weighted sum of three coefficients). Beyond it the base term carries the edge: under
    the tanh^2 gate, 1.0 to the last bit past s = 20, the registered
    exchange network's F differs between s = 1e4 and 1e5 (x_s 9.21 and
    11.51, past hi + 3h = 7.63) and its s-derivative there is finite and
    nonzero. On rows at and beyond every bound, both networks of both
    entries, registered and campaign, return a finite F in (0, a) with a
    finite gradient."""
    KN = _kan()
    for lo, hi in (*_BOUNDS.values(), _HIDDEN, _CUSP[0]):
        h = (hi - lo) / _G

        def basis(x, lo=lo, hi=hi):
            return KN.bspline_basis(x, lo, hi, _G, _K)

        f = jax.vmap(basis)
        out = np.asarray(f(jnp.asarray(
            [lo - _K * h, lo - (_K + 1) * h, lo - 100.0 * h,
             hi + _K * h, hi + (_K + 1) * h, hi + 100.0 * h])))
        assert np.abs(out).max() <= 1e-30, (lo, hi, out)
        at = np.asarray(f(jnp.asarray([lo, hi])))
        assert np.abs(at.sum(axis=1) - 1.0).max() <= 1e-14
        assert np.allclose(at[0, :3], (1 / 6, 2 / 3, 1 / 6), atol=1e-14, rtol=0)
        assert np.allclose(at[1, -3:], (1 / 6, 2 / 3, 1 / 6), atol=1e-14, rtol=0)
        assert not at[0, 3:].any() and not at[1, :-3].any()
    xnet = N.create_network_pair(C.get_architecture("deep_kan_2x6"), seed=0)[0]

    def fx(s):
        return xnet(jnp.stack([jnp.asarray(0.3), (2.0 * (3.0 * jnp.pi ** 2 * 0.3)
                                                  ** (1.0 / 3.0) * 0.3 * s) ** 2]))

    far, farther = float(fx(1e4)), float(fx(1e5))
    assert np.isfinite(far) and np.isfinite(farther) and far != farther
    slope = float(jax.grad(fx)(1e4))
    assert np.isfinite(slope) and slope != 0.0
    rows = _edge_rows()
    rows += [(0.3, _sigma(0.3, _s_of_xs(b)), z) for b in _BOUNDS["x_s"]
             for z in (0.0, 1.0)]
    feats = _cusp_features(len(rows))
    smooth = [i for i, r in enumerate(rows) if r[1] > 1e-20]
    for name in _NEW:
        geom = "geom" in name
        for arch in (C.get_architecture(name), _campaign(name)):
            pol = arch.use_polarized_correlation
            for net, kind, a in zip(N.create_network_pair(arch, seed=0), "xc",
                                    (1.804, 2.0)):
                packed = jnp.asarray([_pack(kind, pol, r, feats[i] if geom else None)
                                      for i, r in enumerate(rows)])
                F = np.asarray(jax.vmap(net)(packed))
                assert np.all(np.isfinite(F) & (F > 0.0) & (F < a)), (name, kind, F)
                grads = np.asarray(jax.vmap(jax.grad(net))(packed[jnp.asarray(smooth)]))
                assert np.isfinite(grads).all(), (name, kind)


def test_the_parameter_count_is_the_closed_form():
    """``parameter_count`` of each network's KAN is (6 d_in + 42)(G + k + 1):
    432 and 540 (the exchange networks, the paper and legacy polarized
    correlation rows of deep_kan_2x6), 486, 594 and 648 for the other rows
    the entries build; it equals the leaf count of the KAN and of the
    network around it. A KAN of in 2, width 4, depth 3, grid 7, order 2
    carries (8 + 16 + 16 + 4) x 10 = 440."""
    KN = _kan()
    for (name, coords, pol), want in _PARAMS.items():
        arch = dataclasses.replace(C.get_architecture(name),
                                   use_polarized_correlation=pol,
                                   descriptor_coordinates=coords)
        for net, n in zip(N.create_network_pair(arch, seed=0), want):
            d_in = net.net.layers[0].coef.shape[1]
            assert n == (6 * d_in + 42) * (_G + _K + 1)
            assert _count(net.net) == n, (name, coords, pol, _count(net.net), n)
            assert sum(x.size for x in _leaves(net.net)) == n
            assert sum(x.size for x in _leaves(net)) == n
    direct = KN.KAN(in_size=1, width=6, depth=2, bounds=(_BOUNDS["x_s"],),
                    grid=_G, order=_K, key=jax.random.PRNGKey(0))
    assert _count(direct) == 432 == sum(x.size for x in _leaves(direct))
    other = KN.KAN(in_size=2, width=4, depth=3, bounds=_CUSP, grid=7, order=2,
                   key=jax.random.PRNGKey(1))
    assert _count(other) == 440 == sum(x.size for x in _leaves(other))


def test_the_constraints_hold():
    """Registered (tanh2) and campaign (x2): F = 1 at s = 0 to 1e-15 on both
    networks; F in (0, a) on 200 rows to s = 50 with the term live (|F - 1|
    above 1e-3 somewhere); the zeroed last layer has every coefficient and
    base weight of the last layer at zero, the KAN's output 0.0 and F = 1 to
    1e-15; anchored, the parent (tracked V3 on OH)."""
    rows = _rows(200, 50.0, seed=7)
    feats = _cusp_features(len(rows))
    for name in _NEW:
        registered = C.get_architecture(name)
        geom = "geom" in name
        for arch in (registered, _campaign(name)):
            pol = arch.use_polarized_correlation
            xnet, cnet = N.create_network_pair(arch, seed=0)
            for net, kind, a in ((xnet, "x", 1.804), (cnet, "c", 2.0)):
                zero_s = jnp.asarray([_pack(kind, pol, (rho, 0.0, 0.3),
                                            feats[i] if geom else None)
                                      for i, rho in enumerate((1e-3, 0.1, 1.0, 50.0))])
                assert np.abs(np.asarray(jax.vmap(net)(zero_s)) - 1.0).max() <= 1e-15
                packed = jnp.asarray([_pack(kind, pol, r, feats[i] if geom else None)
                                      for i, r in enumerate(rows)])
                F = np.asarray(jax.vmap(net)(packed))
                assert np.all((F > 0.0) & (F < a)), (name, kind)
                assert np.abs(F - 1.0).max() > 1e-3, (name, kind, arch.ueg_gate)
            zeroed = N.create_network_pair(
                dataclasses.replace(arch, zero_init_final_layer=True), seed=0)
            for net, kind in zip(zeroed, "xc"):
                last = net.net.layers[-1]
                assert not np.asarray(last.coef).any()
                assert not np.asarray(last.base).any()
                packed = jnp.asarray([_pack(kind, pol, r, feats[i] if geom else None)
                                      for i, r in enumerate(rows)])
                assert np.abs(np.asarray(jax.vmap(net)(packed)) - 1.0).max() <= 1e-15
                assert float(net.net(jnp.asarray(
                    [0.3] * net.net.layers[0].coef.shape[1]))[0]) == 0.0
        _pa.test_anchored_networks_return_the_parent_at_initialization(name)


def _fx_orders(rho, n):
    """F_x of a network at density ``rho`` as a function of s, and its first
    ``n`` s-derivatives under autodiff, compiled per order."""
    k_f = (3.0 * math.pi ** 2 * rho) ** (1.0 / 3.0)

    def f(net, s):
        return net(jnp.stack([jnp.asarray(rho), (2.0 * k_f * rho * s) ** 2]))

    out = []
    for _ in range(n + 1):
        out.append(eqx.filter_jit(f))
        f = jax.grad(f, argnums=1)
    return out


def test_the_derivative_is_continuous_across_a_knot():
    """At the interior knots lo + h and lo + 2h of the first layer's x_s grid
    (s = 1.75 and 5.78) the registered exchange network's F, dF/ds and
    d2F/ds2 (autodiff) are continuous: their one-sided gaps shrink a
    hundredfold from e = 1e-6 s to 1e-8 s; d3F/ds3 jumps: its gap holds
    (ratio 0.999 to 1.003 over 20 seeds). At the midpoint lo + 1.5h, no
    knot, the third derivative's gap shrinks with the rest, so the jump
    marks the knot. The central differences of F on the two sides agree to
    O(e), as they do for any C1 spline (a quadratic spline passes this form
    and fails the second derivative's). The potential matrix of
    ``oneshot.compute_vxc_nn`` on a three-point record whose first point's
    sigma crosses the knot is finite, and its derivative in that sigma,
    which carries d2F/ds2, is continuous: the gap shrinks a hundredfold
    from e = 1e-4 sigma to 1e-6 sigma (9.2e-7 to 9.2e-9 on the cubic,
    constant 6.5e-3 on the quadratic)."""
    KN = _kan()
    from xcquinox.pipeline.models import AlecGGAModel
    from xcquinox.pipeline.oneshot import compute_vxc_nn
    arch = C.get_architecture("deep_kan_2x6")
    xnet = N.create_network_pair(arch, seed=0)[0]
    assert isinstance(xnet.net, KN.KAN)
    lo, hi = _BOUNDS["x_s"]
    h = (hi - lo) / _G
    orders = _fx_orders(0.3, 3)

    def gaps(s0, rel):
        e = rel * s0
        return [abs(float(d(xnet, jnp.asarray(s0 + e)))
                    - float(d(xnet, jnp.asarray(s0 - e)))) for d in orders]

    for j in (1.0, 2.0, 1.5):
        s_k = _s_of_xs(lo + j * h)
        wide, narrow = gaps(s_k, 1e-6), gaps(s_k, 1e-8)
        for r in range(3):
            assert narrow[r] <= 0.05 * wide[r] + 1e-14, (j, r, wide, narrow)
        if j == 1.5:
            assert narrow[3] <= 0.05 * wide[3] + 1e-13, (j, wide, narrow)
        else:
            assert narrow[3] >= 0.5 * wide[3] and narrow[3] > 1e-9, (j, wide, narrow)
            cd = []
            for rel in (1e-4, 1e-6):
                e, dd = rel * s_k, 0.5 * rel * s_k

                def central(s, dd=dd):
                    return (float(orders[0](xnet, jnp.asarray(s + dd)))
                            - float(orders[0](xnet, jnp.asarray(s - dd)))) / (2.0 * dd)

                cd.append(abs(central(s_k + e) - central(s_k - e)))
            assert cd[1] <= 0.05 * cd[0] + 1e-9, (j, cd)
    model = AlecGGAModel.from_arch(arch, seed=0)
    rng = np.random.default_rng(3)
    rho = jnp.asarray([0.3, 0.05, 1.2])
    nabla = jnp.asarray(rng.normal(0.0, 0.3, (3, 3)))
    sigma0 = jnp.sum(nabla ** 2, axis=1)
    ao = jnp.asarray(rng.normal(0.0, 1.0, (3, 2)))
    ao_grad = jnp.asarray(rng.normal(0.0, 1.0, (3, 3, 2)))
    weights = jnp.asarray([0.7, 0.2, 0.4])
    features = jnp.zeros((3, 0))
    sig_k = _sigma(0.3, _s_of_xs(lo + h))

    def V(sig_first):
        return compute_vxc_nn(model, rho, sigma0.at[0].set(sig_first), features,
                              ao, weights, nabla, ao_grad, part="x")

    dV = jax.jacfwd(V)
    for x in (sig_k, sig_k * (1 + 1e-6), sig_k * (1 - 1e-6)):
        assert np.isfinite(np.asarray(V(x))).all()
    jumps = []
    for rel in (1e-4, 1e-6):
        e = rel * sig_k
        jumps.append(float(np.abs(np.asarray(dV(sig_k + e))
                                  - np.asarray(dV(sig_k - e))).max()))
    assert np.abs(np.asarray(dV(sig_k))).max() > 1e-6
    assert jumps[1] <= 0.05 * jumps[0] + 1e-13, jumps


def test_the_registry_entries_are_the_stated_ones():
    """Each entry is its 3x16 base at depth 2, width 6 with network "kan",
    grid 5, order 3 (dataclass equality: deep_3x16 and deep_geom_3x16, so
    dm_entropy_intensive, the log transform and the cusp pair as there); 43
    entries once the Laplacian pair joins; its own shown name; the GGA
    rung; the expanded key naming the
    network; describe() carrying the three fields (the MLP entries the
    configuration's defaults, "mlp", 0, 0); both figure orders with the two
    after deep_sine_3x16; the stated colours, each carried by no other
    architecture, rung accent or band; every ordered name coloured."""
    from xcquinox.pipeline import arch_names, rungs
    for name in _NEW:
        entry = C.ARCHITECTURES[name]
        assert entry == dataclasses.replace(C.ARCHITECTURES[_BASE[name]], name=name,
                                            depth=2, nodes=6, **_KAN), name
        assert arch_names.DISPLAY_NAME[name] == name
        assert rungs.rung_of(name) == rungs.RUNG_GGA == AS.rung_of(name)
        assert _EXPANDED in arch_names.expanded_key(name), arch_names.expanded_key(name)
        assert all(entry.describe()[k] == v for k, v in _KAN.items())
    assert len(C.ARCHITECTURES) == 43
    assert all(C.ARCHITECTURES["deep_3x16"].describe()[k] == v
               for k, v in dict(network="mlp", kan_grid=0, kan_order=0).items())
    accents = {v.lower() for v in AS.RUNG_ACCENT.values()}
    bands = {v.lower() for v in AS.RUNG_BAND.values()} | {"#ffffff"}
    for name, colour in _COLOUR.items():
        assert AS.arch_color(name).lower() == colour, (name, AS.arch_color(name))
        others = {v.lower() for k, v in AS.ARCH_COLOR.items()
                  if k != name and not name.startswith(k + "_")}
        assert colour not in others | accents | bands, name
    for order in (AS._STORED_ORDER, AS._DISPLAY_ORDER, AS.ARCH_ORDER):
        i = order.index("deep_sine_3x16")
        assert tuple(order[i:i + 3]) == ("deep_sine_3x16", *_NEW), order
    assert [n for n in AS.ARCH_ORDER if n not in AS.ARCH_COLOR] == []
    # a sized name outside the figure orders keeps its rung's accent (the
    # meta-GGA capacity probes); only the ordered sized names inherit a base
    # colour, so the probes stay apart from deep_mgga_3x16
    for probe in ("deep_mgga_3x32", "deep_mgga_4x16", "deep_mgga_4x32"):
        assert AS.arch_color(probe) == AS.RUNG_ACCENT[AS.rung_of(probe)], probe
        assert AS.arch_color(probe) != AS.arch_color("deep_mgga_3x16"), probe


_NAN, _INF = float("nan"), float("inf")
_KL = dict(network="kan", kan_grid=_G, kan_order=_K, descriptor_log_transform=True)
_BAD = (*(("network", dict(network=v)) for v in ("KAN", "", "mlp ", None, 1)),
        *(("kan_grid", dict(_KL, kan_grid=v)) for v in (0, -1, 1.5, True, "5")),
        *(("kan_order", dict(_KL, kan_order=v)) for v in (0, -2, 2.5, True)),
        # a silent setting is not a configuration: the MLP keeps grid 0 and
        # order 0, the KAN states both
        ("kan_grid", dict(kan_grid=6)), ("kan_order", dict(kan_order=2)),
        ("network|fourier_features", dict(_KL, fourier_features=16)),
        ("network|activation", dict(_KL, activation="sine")),
        # the legacy coordinates without the transform are not bounded
        ("network|descriptor_log_transform",
         dict(_KL, descriptor_log_transform=False)))


def test_the_fields_are_validated():
    """from_spec, replace of an entry and both constructors refuse each case
    of _BAD naming the field, and the attention block; an unbounded
    descriptor column (dm_statistics, metagga) refuses the network while the
    cusp, rung-3.5 and multishell columns carry it at their declared
    bounds; the networks need one bound per column, finite, lo < hi;
    ``bounds_for`` refuses the rows that are not bounded; valid values build;
    anchored() and apply_model_block keep the three fields; the descriptors
    declare their bounds and ``column_ranges`` is hi - lo of them, the values
    A3 states."""
    from xcquinox.pipeline.descriptors import make_descriptor
    for kw in (dict(network="kan", kan_grid=_G, kan_order=_K, **_DEEP),
               dict(network="kan", kan_grid=1, kan_order=1, **_DEEP),
               dict(network="kan", kan_grid=8, kan_order=2,
                    descriptor_coordinates="paper", use_polarized_correlation=True),
               dict(network="kan", kan_grid=_G, kan_order=_K,
                    descriptors=["cusp"], **_DEEP)):
        arch = C.ArchitectureConfig.from_spec("deep_t_2x6", 2, 6, **kw)
        want = {k: kw[k] for k in ("network", "kan_grid", "kan_order")}
        assert all(getattr(arch, f) == v for f, v in want.items()), (kw, want)
        for net in N.create_network_pair(arch, seed=0):
            assert all((int(layer.grid), int(layer.order))
                       == (want["kan_grid"], want["kan_order"])
                       for layer in net.net.layers)
            assert net.net.layers[0].coef.shape[2] == want["kan_grid"] + want["kan_order"]
    plain_default = C.ArchitectureConfig("t", 2, 8)
    assert (plain_default.network, plain_default.kan_grid,
            plain_default.kan_order) == ("mlp", 0, 0)
    plain, either = C.ARCHITECTURES["deep_3x16"], (ValueError, TypeError)
    for words, kw in _BAD:
        builds = [lambda kw=kw: C.ArchitectureConfig.from_spec("t", 3, 16, **kw),
                  lambda kw=kw: dataclasses.replace(plain, **kw)] + [
            lambda cls=cls, kw=kw: cls(n_extra_features=0, depth=2, nodes=6, **kw)
            for cls in (N.AlecGGA_XNet, N.AlecGGA_CNet)]
        for build in builds:
            _verdict(False, build, either, words)
    for build in (lambda: C.ArchitectureConfig.from_spec(
                      "t", 2, 6, attention=True, num_heads=2, **_KL),
                  lambda: dataclasses.replace(plain, attention=True, num_heads=4,
                                              **_KAN),
                  lambda: N.AlecGGA_XNet(n_extra_features=0, depth=2, nodes=6,
                                         use_self_attention=True, **_KL),
                  lambda: N.AlecGGA_CNet(n_extra_features=0, depth=2, nodes=6,
                                         use_self_attention=True, **_KL)):
        _verdict(False, build, either, r"network|attention")
    for base in ("deep_dm_3x16", "deep_mgga_3x16"):
        assert (None, None) in C.ARCHITECTURES[base].extra_feature_bounds, base
        _verdict(False, lambda base=base: dataclasses.replace(
            C.ARCHITECTURES[base], **_KAN), ValueError, r"network")
    for base, bounds in (("deep_cusp_3x16", _CUSP),
                         ("deep_rung35_3x16", _CUSP + _RUNG35),
                         ("deep_rung35ms_3x16", _CUSP + ((0.0, 1.0),) * 6)):
        assert C.ARCHITECTURES[base].extra_feature_bounds == bounds, base
        arch = dataclasses.replace(C.ARCHITECTURES[base], **_KAN)
        xnet, cnet = N.create_network_pair(arch, seed=0)
        assert _layer_bounds(xnet.net.layers[0])[1:] == bounds, base
        assert _layer_bounds(cnet.net.layers[0])[-len(bounds):] == bounds, base
    for name, bounds, ranges in (("cusp", _CUSP, (1.0, 2.0)),
                                 ("rung35", _RUNG35, (1.0, 1.0)),
                                 ("rung35_multishell", ((0.0, 1.0),) * 6, (1.0,) * 6),
                                 ("dm_statistics", ((None, None),) * 2, (None, None)),
                                 ("metagga", ((None, None),), (None,))):
        d = make_descriptor(name)
        assert tuple(map(tuple, d.column_bounds)) == bounds, name
        assert tuple(d.column_ranges) == ranges, name
    raw = dict(n_extra_features=2, depth=2, nodes=6, **_KL)
    for cls in (N.AlecGGA_XNet, N.AlecGGA_CNet):
        for bad, words in ((None, "extra_feature_bounds"),
                           (((0.0, 1.0),), "extra_feature_bounds"),
                           (((0.0, 1.0), (None, None)), "network|extra_feature_bounds"),
                           (((0.0, 1.0), (1.0, 1.0)), "extra_feature_bounds"),
                           (((0.0, 1.0), (2.0, 1.0)), "extra_feature_bounds"),
                           (((0.0, 1.0), (0.0, _INF)), "extra_feature_bounds"),
                           (((0.0, 1.0), (_NAN, 1.0)), "extra_feature_bounds")):
            _verdict(False, lambda cls=cls, bad=bad: cls(**raw, extra_feature_bounds=bad),
                     ValueError, words)
        cls(**raw, extra_feature_bounds=_CUSP)
    KN = _kan()
    for args in (("c", "legacy", False, True, ()), ("x", "legacy", False, False, ()),
                 ("c", "paper", True, False, ())):
        _verdict(False, lambda args=args: KN.bounds_for(*args), ValueError,
                 r"network|kan|bound")
    from xcquinox.pipeline.cluster.grid_config import ModelConfig
    block = ModelConfig(parent_anchor=True, descriptor_coordinates="paper",
                        ueg_gate="x2")
    for arch in map(C.get_architecture, _NEW):
        for out in (C.anchored(arch), C.apply_model_block(arch, block)):
            assert all(getattr(out, f) == getattr(arch, f) for f in _KAN)


def test_the_class_record_and_the_loaders_state_the_network(tmp_path, monkeypatch):
    """The ten front-end fields in both readings and the record for the MLP
    (deep_3x16), the KAN at grid 5, order 3 (deep_kan_2x6 on the paper
    coordinates) and at grid 6, order 2, whose leaves have the same shapes
    (G + k = 8) and load into the first KAN's skeleton without complaint:
    the record is what tells them apart, as its digest tells the paper
    rows' bounds from the legacy rows'. Each class is refused by the other
    two; a silent record is the MLP, refused by both KAN classes; the
    run_pretrain stamp and its guard on a supplied pair of another class;
    both loaders, the certificate keep check (which reads the registry
    entry by name) and validate_run."""
    from xcquinox.pipeline import checkpoint_class as K
    from xcquinox.pipeline.cluster import fidelity as fid
    from xcquinox.pipeline.pretrain import _metadata_preflight, run_pretrain
    from xcquinox.pipeline.train import _require_matching_model_class as load
    entry = C.get_architecture("deep_kan_2x6")
    archs = {"p": C.get_architecture("deep_3x16"),
             "k": _coord._arch("deep_kan_2x6", "paper")}
    archs["q"] = dataclasses.replace(archs["k"], kan_grid=6, kan_order=2)
    mlp = {"p": _MLP0, "k": _kan_class("paper", True),
           "q": _kan_class("paper", True, grid=6, order=2)}
    assert tuple(K.MLP_FIELDS) == tuple(_MLP0)
    assert K.DEFAULT_MLP == _MLP0
    assert K.normalize_mlp({}) == K.normalize_mlp(None) == _MLP0
    # the registered entry reads the legacy rows: its digest is of their
    # bounds, not the paper rows'
    assert K.mlp_class_of(entry) == _kan_class("legacy", False)
    assert K.mlp_class_of(entry)["kan_digest"] != mlp["k"]["kan_digest"]
    # the digest covers every layer's knots: the hidden layers' bounds are a
    # module constant a skeleton would otherwise take silently, on both the
    # configuration's side and the built model's
    KN = _kan()
    before = K.mlp_class_of(archs["k"])["kan_digest"]
    assert K.model_class_of_model(_ckpt._model(archs["k"]))["kan_digest"] == before
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(KN, "HIDDEN_BOUNDS", (-2.0, 2.0))
        assert K.mlp_class_of(archs["k"])["kan_digest"] != before
        moved = K.model_class_of_model(_ckpt._model(archs["k"]))["kan_digest"]
        assert moved != before and moved == K.mlp_class_of(archs["k"])["kan_digest"]
    assert K.mlp_class_of(archs["k"])["kan_digest"] == before
    # a row the spline network cannot carry (the paper coordinates with the
    # zeta-blind correlation network) is reported under the spline digest's
    # name, not the map's
    unpolarized = SimpleNamespace(
        model=SimpleNamespace(parent_anchor=False, descriptor_coordinates="paper",
                              ueg_gate="tanh2"),
        use_polarized_correlation=False)
    got = fid.model_class_mismatches(
        unpolarized, dict(mlp["k"], arch="deep_kan_2x6",
                          descriptor_coordinates="paper"), "deep_kan_2x6")
    assert [m[0] for m in got if m[0].endswith("_digest")] == ["kan_digest"], got
    assert any(m[0] == "kan_digest" and "unresolvable" in str(m[2]) for m in got), got
    assert ([x.shape for x in _leaves(N.create_network_pair(archs["k"], seed=0))]
            == [x.shape for x in _leaves(N.create_network_pair(archs["q"], seed=0))])
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
    eqx.tree_deserialise_leaves(ckpt["q"], _ckpt._model(archs["k"]))
    record = Path(K.class_record_path(ckpt["p"]))
    for silent in (False, True):
        if silent:
            kept = json.loads(record.read_text())
            for f in _MLP0:
                kept.pop(f)
            record.write_text(json.dumps(kept))
        for k, j in ((k, j) for k in ("p" if silent else "pkq") for j in archs):
            _verdict(k == j, lambda: K.require_matching_class(ckpt[k], want[j]),
                     K.ModelClassMismatch)
    from xcquinox.pipeline.pretrain_data_gen import generate_pretrain_data_npz
    data = str(tmp_path / "data")
    os.makedirs(data)
    generate_pretrain_data_npz(data, atoms=(("He", 0),), basis="sto-3g",
                               grid_level=0, polarized=True, descriptors=True,
                               density_fit=False)
    md = run_pretrain(C.PretrainSpec(arch=archs["k"], data_dir=data, n_steps=2,
                                     checkpoint_dir=str(tmp_path / "pre"), seed=0))
    on_disk = json.loads((tmp_path / "pre" / "pretrain_metadata.json").read_text())
    assert {f: md[f] for f in _MLP0} == {f: on_disk[f] for f in _MLP0} == mlp["k"]
    load(str(tmp_path / "pre"), archs["k"])
    _verdict(False, lambda: load(str(tmp_path / "pre"), archs["q"]))
    _verdict(False, lambda: run_pretrain(
        C.PretrainSpec(arch=archs["k"], data_dir=data, n_steps=2,
                       checkpoint_dir=str(tmp_path / "guard"), seed=0),
        networks=N.create_network_pair(_coord._arch("deep_3x16", "paper"), seed=0)))
    assert not any((tmp_path / "guard").glob("*")), list((tmp_path / "guard").glob("*"))
    coords = {"p": "legacy", "k": "paper", "q": "paper"}
    registry_class = {"p": mlp["p"], "k": mlp["k"], "q": mlp["k"]}
    md_path = tmp_path / "pretrain_metadata.json"
    for stated in (None, "p", "k", "q"):
        extra = {} if stated is None else dict(mlp[stated])
        said = _MLP0 if stated is None else mlp[stated]
        for k, arch in archs.items():
            base = {"depth": arch.depth, "nodes": arch.nodes, "parent_anchor": False,
                    "descriptor_coordinates": coords[k], "ueg_gate": "tanh2"}
            md_path.write_text(json.dumps(dict(base, **extra)))
            ok = said == mlp[k]
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
            assert [m[0] for m in got] == _differing(said, registry_class[k]), (
                k, stated, got)
    md_path.unlink()
    for k, arch in archs.items():
        _verdict(k == "p", lambda: load(str(tmp_path), arch))
    for i, (name, coords_i) in enumerate(zip(_NEW, ("paper", "legacy"))):
        vcfg = _vrt._cfg()
        vcfg.sweep.arch = (name,)
        vcfg.model = SimpleNamespace(parent_anchor=False,
                                     descriptor_coordinates=coords_i,
                                     ueg_gate="tanh2")
        monkeypatch.setattr(_vrt.vr, "load_grid_config", lambda p, c=vcfg: c)
        run = _vrt._write_run(tmp_path / f"v{i}", [_vrt._spec_for(name)],
                              certificates=False)
        cert_path = Path(_vrt._write_certificate(run, name))
        cert0 = json.loads(cert_path.read_text())
        stated_class = _kan_class(coords_i, True, _CUSP if "geom" in name else ())
        for extra in ({}, dict(stated_class)):
            cert_path.write_text(json.dumps(
                dict(cert0, descriptor_coordinates=coords_i, **extra)))
            cert_path.with_name("pretrain_metadata.json").write_text(json.dumps(
                dict(use_polarized_correlation=True, parent_anchor=False,
                     descriptor_coordinates=coords_i, **extra)))
            named = [f for f in _vrt.vr.validate_run(run)[0]
                     if re.search(_WORDS, f)]
            certified = sum("certificate records" in f for f in named)
            assert (certified > 0, len(named) > certified) == (not extra,) * 2, (
                name, extra, named)


def test_the_pretraining_step_runs():
    """One Adam step (1e-3) of _PretrainLoss per exchange network of the two
    campaign entries: finite loss and gradients, one nonzero; six leaves (a
    coefficient and a base-weight array per layer) before and after; every
    layer's coefficients and base weights move; the bounds, grid and order
    are the same objects' values after the step (static, no leaf)."""
    KN = _kan()
    from xcquinox.pipeline.pretrain import _PretrainLoss
    rho, s = jnp.asarray(np.geomspace(0.05, 5.0, 64)), jnp.linspace(0.0, 4.0, 64)
    sigma = (2.0 * (3.0 * jnp.pi ** 2 * rho) ** (1 / 3) * rho * s) ** 2
    ref = pbe_fx(rho, sigma) - 1.0
    feats = jnp.asarray(_cusp_features(64))
    loss_fn, optimizer = _PretrainLoss(), optax.adam(1e-3)
    for name in _NEW:
        xnet = N.create_network_pair(_campaign(name), seed=0)[0]
        assert isinstance(xnet.net, KN.KAN)
        rows = jnp.stack([rho, sigma], axis=1)
        if "geom" in name:
            rows = jnp.concatenate([rows, feats], axis=1)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(xnet, rows, ref)
        flat = _leaves(grads)
        assert np.isfinite(float(loss)) and all(np.isfinite(g).all() for g in flat)
        assert max(float(np.abs(g).max()) for g in flat) > 0.0, name
        params = eqx.filter(xnet, eqx.is_array)
        updates, _ = optimizer.update(eqx.filter(grads, eqx.is_array),
                                      optimizer.init(params), params)
        stepped = eqx.apply_updates(xnet, updates)
        assert len(_leaves(stepped)) == len(_leaves(xnet)) == 6, name
        assert np.isfinite(float(loss_fn(stepped, rows, ref))), name
        for before, after in zip(xnet.net.layers, stepped.net.layers):
            for field in ("coef", "base"):
                moved = np.abs(np.asarray(getattr(after, field))
                               - np.asarray(getattr(before, field))).max()
                assert moved > 0.0, (name, field)
            assert _layer_bounds(after) == _layer_bounds(before)
            assert (after.grid, after.order) == (before.grid, before.order)


@pytest.mark.slow
def test_the_run_readers_accept_the_runs_they_build(tmp_path, monkeypatch):
    """Each entry through the production writers and readers: a two-step
    pretraining and its certificate stamp the ten fields, the keep check
    and ``certificate_describes_run`` report no disagreement,
    ``completed_pretraining`` refuses on the gate alone (two steps do not
    pass it), a certificate without the fields is reported by the keep check
    on network, kan_grid and kan_order, and ``validate_run`` reports nothing
    on the run and a failure naming each field once the metadata is
    tampered."""
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
    for name in _NEW:
        cfg = _pa._anchored_cfg(arch=(name,))
        cfg.model.parent_anchor = False
        cfg.model.descriptor_coordinates = "paper"
        cfg.model.ueg_gate = "x2"
        arch = resolve_run_architecture(cfg, C.get_architecture(name))
        if not any(data.iterdir()):
            generate_pretrain_data_npz(str(data), atoms=(("He", 0),),
                                       basis="sto-3g", grid_level=0,
                                       polarized=True, descriptors=True,
                                       density_fit=False)
        assert arch.use_polarized_correlation and not arch.parent_anchor
        want = K.mlp_class_of(arch)
        assert want == _kan_class("paper", True, _CUSP if "geom" in name else ())
        run_dir = str(tmp_path / name)
        pre = pretrain_checkpoint_dir(run_dir, name)
        md = run_pretrain(C.PretrainSpec(arch=arch, data_dir=str(data),
                                         checkpoint_dir=pre, n_steps=2, seed=0))
        payload = fid.fidelity_certificate(cfg, run_dir, name,
                                           oracle_set=_pa._tiny_oracle_set())
        assert {f: md[f] for f in _MLP0} == {f: payload[f] for f in _MLP0} == want
        assert fid.model_class_mismatches(cfg, payload, name) == []
        assert not any(re.search(_WORDS, line) for line in
                       fid.certificate_describes_run(cfg, pre, name, payload))
        keep, reason = completed_pretraining(pre, cfg, name)
        assert keep is False and not re.search(_WORDS, reason), (name, reason)
        tampered = {k: v for k, v in payload.items() if k not in _MLP0}
        assert [m[0] for m in fid.model_class_mismatches(cfg, tampered, name)
                ] == _differing(_MLP0, want) == ["network", "kan_grid",
                                                 "kan_order", "kan_digest"], name
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
        Path(run_dir, "resolved_config.yaml").write_text("placeholder: true\n")
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
        assert all(any(f"metadata {f}" in line for line in failures)
                   for f in _differing(_MLP0, want)), (name, failures)
