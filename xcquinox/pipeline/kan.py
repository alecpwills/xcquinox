"""Kolmogorov-Arnold networks on fixed spline grids (Liu et al.,
arXiv:2404.19756 (2024), section 2.2): the learnable functions sit on the
edges, each a B-spline of order ``order`` on a uniform grid of ``grid``
intervals over its input's bounds plus a SiLU base term with its own weight,
and a layer sums the edge functions into each output. Two departures from the
published form are this version's and are named here: the grids are FIXED
(Liu et al. update each grid from its input activations during training) and
every hidden layer's output passes through tanh before the next layer, so the
hidden grids are [-1, 1] and every spline reads a bounded input by
construction. The first layer's inputs are the enhancement networks'
coordinates, bounded over the pretraining domain (:data:`COORDINATE_BOUNDS`,
measured on the sample the Fourier map's scales were measured on), and the
descriptor columns' declared bounds. From the second layer on an edge
function is therefore a spline in tanh of the previous layer's output. The
coefficients start at N(0, 0.1^2) so every spline starts near zero and the
base weights at Xavier's uniform bound, from the network's own key, as the
published implementation initializes them (the spline scale w_s of the
paper's eq. 2.10 is absorbed into the coefficients).

A spline sums to one on its grid, tapers over the ``order`` extension
intervals on each side and vanishes beyond them (:func:`bspline_basis`); on
a row past the extension the base term alone carries the edge, so the value
and the gradient stay finite. The anchor protocol zeroes the last layer's
coefficients and base weights (:func:`zeroed_last_layer`), so an anchored
pair returns the parent exactly. The class record states the network kind,
the grid, the order and the digest of every layer's bounds (:func:`digest`;
the first layer's from the coordinates and the descriptor columns, the
hidden layers' from :data:`HIDDEN_BOUNDS`), the knots a skeleton would
otherwise take from this module silently.
"""
from __future__ import annotations

import hashlib
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

#: The bounds of the base coordinates, the first layer's grids. The three
#: coordinates without an analytic bound span the density-weighted
#: pretraining sample of the campaign identity (the campaign's atoms and the
#: pool atoms at 6-311++G(3df,2pd), grid level 3, density fitting; 800 rho*w
#: points per system and channel at sampling seed 42, the sample the Fourier
#: map's ``COORDINATE_RANGES`` were measured on): x_s = (1 - e^{-s^2}) ln(1 +
#: s) from 0.011 to 4.770, x_0 = ln(rho^{1/3} + 1e-5) from -6.185 to 2.600
#: and the legacy transformed r_s from 9.6e-5 to 5.716; the lower ends of
#: x_s and r_s sit at their analytic zero and the upper ends are rounded
#: outward to the hundredth. The two bounded coordinates keep their analytic
#: ranges: x_1 = ln(spinscale + 1e-5) from 0 to its value at full
#: polarization (the paper row's offset puts it 7.9e-6 past ln 2 / 3, which
#: the dfs row reaches), and the legacy set's raw spin scale from 1 to
#: 2^(1/3).
COORDINATE_BOUNDS = {
    "x_s": (0.0, 4.77),
    "x_0": (-6.19, 2.60),
    "x_1": (0.0, math.log(2.0 ** (1.0 / 3.0) + 1e-5)),
    "rs_t": (0.0, 5.72),
    "spinscale": (1.0, 2.0 ** (1.0 / 3.0)),
}

#: The hidden layers read tanh of the previous layer's output.
HIDDEN_BOUNDS = (-1.0, 1.0)


def bounds_for(network: str, descriptor_coordinates: str,
               descriptor_log_transform: bool, polarized: bool,
               extra_bounds: tuple) -> tuple:
    """The bounds of ``network``'s input row ("x" or "c") under the RESOLVED
    coordinate set, in the row's order (``fourier_features.scales_for``
    states the compositions): the paper and dfs rows [x_s, *extras] and
    [x_0, x_1, x_s, *extras], the legacy rows under the log transform [x_s,
    *extras] and [r_s transformed, x_s, (spin scale,) *extras]; each
    descriptor column has the bounds its descriptor declares. The network is
    refused where the inputs are not bounded: the legacy set without the log
    transform, and a descriptor column whose bounds are ``None``."""
    extras = tuple(tuple(pair) for pair in extra_bounds)
    for i, (lo, hi) in enumerate(extras):
        if lo is None or hi is None:
            raise ValueError(
                f"network='kan': descriptor column {i} is not bounded by "
                "construction (Descriptor.column_bounds); every spline grid "
                "needs bounds")
    b = COORDINATE_BOUNDS
    if descriptor_coordinates in ("dfs", "paper"):
        if network == "x":
            return (b["x_s"],) + extras
        if not polarized:
            raise ValueError(
                "network='kan': the paper and dfs coordinates feed the "
                "correlation network the polarized row [x_0, x_1, x_s]; "
                "use_polarized_correlation=False cannot carry it")
        return (b["x_0"], b["x_1"], b["x_s"]) + extras
    if not descriptor_log_transform:
        raise ValueError(
            "network='kan': the legacy coordinates without the descriptor "
            "log transform feed the raw reduced gradient and r_s, which are "
            "not bounded; the grids need the transformed inputs or the paper "
            "or dfs coordinates")
    if network == "x":
        return (b["x_s"],) + extras
    return ((b["rs_t"], b["x_s"]) + ((b["spinscale"],) if polarized else ())
            + extras)


def layer_bounds_for(first_bounds, width: int, depth: int) -> tuple:
    """The bounds of every layer of a :class:`KAN` of ``depth`` hidden layers
    of ``width`` outputs, in order: the first layer's (``first_bounds``, one
    pair per input), then ``depth`` layers of ``width`` inputs on
    :data:`HIDDEN_BOUNDS` (the tanh outputs of the layer before)."""
    first = tuple(tuple(pair) for pair in first_bounds)
    return (first,) + ((HIDDEN_BOUNDS,) * int(width),) * int(depth)


def digest(*layer_bounds) -> str | None:
    """The first sixteen hex digits of the SHA-256 of the float64 bytes of
    every layer's bounds in order, the class record's statement of all the
    knots; ``None`` when every argument is ``None`` (no network of this
    kind)."""
    if all(b is None for b in layer_bounds):
        return None
    sha = hashlib.sha256()
    for b in layer_bounds:
        sha.update(np.asarray(b, dtype=np.float64).tobytes())
    return sha.hexdigest()[:16]


def digest_for(network: str, descriptor_coordinates: str,
               descriptor_log_transform: bool, polarized: bool,
               extra_bounds: tuple, depth: int, width: int) -> str | None:
    """The digest an architecture's networks carry, from its resolved fields
    alone: every layer of the exchange network then every layer of the
    correlation network (``None`` for the MLP)."""
    if network != "kan":
        return None
    layers = ()
    for kind in ("x", "c"):
        first = bounds_for(kind, descriptor_coordinates,
                           descriptor_log_transform, polarized, extra_bounds)
        layers += layer_bounds_for(first, width, depth)
    return digest(*layers)


def bspline_basis(x, lo: float, hi: float, grid: int, order: int):
    """The ``grid + order`` B-splines of degree ``order`` on the uniform knot
    vector of ``grid`` intervals over [lo, hi] extended by ``order`` knots at
    the same spacing on each side (Cox-de Boor), evaluated at the scalar
    ``x``: they sum to one on [lo, hi], the bounds included, taper over the
    extension intervals and vanish at and beyond lo - order h and hi +
    order h. The interior knots are ``linspace(lo, hi, grid + 1)``, so both
    bounds are knots exactly (lo + grid h lands an ulp past hi on the x_0
    bounds), and the extension knots are lo - j h and hi + j h."""
    dtype = jnp.result_type(x, float)
    h = (hi - lo) / grid
    knots = jnp.concatenate([
        lo - h * jnp.arange(order, 0, -1, dtype=dtype),
        jnp.linspace(lo, hi, grid + 1, dtype=dtype),
        hi + h * jnp.arange(1, order + 1, dtype=dtype)])
    basis = ((x >= knots[:-1]) & (x < knots[1:])).astype(knots.dtype)
    for degree in range(1, order + 1):
        left = ((x - knots[:-(degree + 1)])
                / (knots[degree:-1] - knots[:-(degree + 1)]))
        right = ((knots[degree + 1:] - x)
                 / (knots[degree + 1:] - knots[1:-degree]))
        basis = left * basis[:-1] + right * basis[1:]
    return basis


class KANLayer(eqx.Module):
    """One layer: the edge functions phi_ji(x_i) = base[j, i] silu(x_i) +
    sum_m coef[j, i, m] B_m(x_i) summed over the inputs i into each output
    j; no bias (a constant is a spline). ``bounds`` is one (lo, hi) per
    input, the grid of its splines."""
    coef: jax.Array
    base: jax.Array
    bounds: tuple = eqx.field(static=True)
    grid: int = eqx.field(static=True)
    order: int = eqx.field(static=True)

    def __init__(self, in_size: int, out_size: int, bounds, grid: int,
                 order: int, key):
        bounds = tuple(tuple(pair) for pair in bounds)
        if len(bounds) != in_size:
            raise ValueError(
                f"KANLayer: {len(bounds)} bounds for {in_size} inputs")
        for lo, hi in bounds:
            if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                raise ValueError(
                    f"KANLayer: bounds {bounds!r} must be finite with lo < hi")
        if int(grid) < 1 or int(order) < 1:
            raise ValueError(
                f"KANLayer: grid {grid!r} and order {order!r} must be positive")
        key_coef, key_base = jax.random.split(key)
        self.coef = 0.1 * jax.random.normal(
            key_coef, (out_size, in_size, int(grid) + int(order)))
        limit = math.sqrt(6.0 / (in_size + out_size))
        self.base = jax.random.uniform(
            key_base, (out_size, in_size), minval=-limit, maxval=limit)
        self.bounds = tuple((float(lo), float(hi)) for lo, hi in bounds)
        self.grid = int(grid)
        self.order = int(order)

    def __call__(self, x):
        basis = jnp.stack([
            bspline_basis(x[i], lo, hi, self.grid, self.order)
            for i, (lo, hi) in enumerate(self.bounds)])
        return (jnp.einsum("oim,im->o", self.coef, basis)
                + self.base @ jax.nn.silu(x))


class KAN(eqx.Module):
    """``depth`` hidden layers of ``width`` outputs and a final layer of one
    output, tanh between the layers; ``bounds`` are the first layer's, one
    (lo, hi) per input; the hidden layers' are :data:`HIDDEN_BOUNDS`."""
    layers: tuple

    def __init__(self, in_size: int, width: int, depth: int, bounds,
                 grid: int, order: int, key):
        sizes = [int(in_size)] + [int(width)] * int(depth) + [1]
        keys = jax.random.split(key, len(sizes) - 1)
        per_layer = layer_bounds_for(bounds, width, depth)
        self.layers = tuple(
            KANLayer(n_in, n_out, layer_bounds, grid, order, keys[i])
            for i, (n_in, n_out, layer_bounds)
            in enumerate(zip(sizes[:-1], sizes[1:], per_layer)))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)

    @property
    def parameter_count(self) -> int:
        """The coefficients and base weights of every layer."""
        return int(sum(layer.coef.size + layer.base.size
                       for layer in self.layers))


def zeroed_last_layer(net: KAN) -> KAN:
    """``net`` with the last layer's coefficients and base weights at zero, so
    its output is zero everywhere (the anchor protocol's zeroed final layer)."""
    last = net.layers[-1]
    return eqx.tree_at(lambda m: (m.layers[-1].coef, m.layers[-1].base), net,
                       replace=(jnp.zeros_like(last.coef),
                                jnp.zeros_like(last.base)))
