"""The fixed Fourier-feature map in front of an enhancement network's MLP
(Tancik et al., NeurIPS 33 (2020)): the frequency matrix, drawn once from
numpy's frozen stream and scaled per coordinate, and its digest.

A network with ``fourier_features = m`` feeds its MLP gamma(v) = [sin(2 pi B
v), cos(2 pi B v)] in place of its input row v, B an (m, d) matrix of
``fourier_scale`` N(0, 1) draws, each column divided by its coordinate's
range so that the scale counts frequencies per range. The matrix is part of
the architecture, not of the fit: it is a static field of the networks, no
optimizer touches it, a skeleton built from another network seed carries the
same matrix, and the class record beside a checkpoint states its digest.
Numpy's ``RandomState`` is the stream because it is frozen across versions
and independent of JAX's PRNG implementation and precision flags. This
module imports numpy alone, so the configuration and the class record can
compute the digest without the networks.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

#: The ranges of the base coordinates, the scale each is divided by so that
#: ``fourier_scale`` counts frequencies per range. The three coordinates
#: without an analytic bound are scaled by their spans over the
#: density-weighted pretraining sample of the campaign identity (the
#: campaign's atoms and the pool atoms at 6-311++G(3df,2pd), grid level 3,
#: density fitting; 800 rho*w points per system and channel at sampling seed
#: 42, the pretraining's own draw): x_s = (1 - e^{-s^2}) ln(1 + s) spans
#: 4.76, x_0 = ln(rho^{1/3} + 1e-5) spans 8.78 (from -6.19 in the tails to
#: 2.60 at the chlorine nucleus), and the legacy set's transformed r_s,
#: (1 - e^{-r_s^2}) ln(1 + r_s), spans 5.72. The two bounded coordinates
#: keep their analytic spans, which the sample reproduces: x_1 =
#: ln(spinscale), from 0 at zeta = 0 to ln 2 / 3 at full polarization, and
#: the legacy set's raw spin scale, from 1 to 2^(1/3). The transformed
#: reduced gradient is x_s on every set. A descriptor column is divided by
#: the range its descriptor declares (``Descriptor.column_ranges``).
COORDINATE_RANGES = {
    "x_s": 4.76,
    "x_0": 8.78,
    "x_1": math.log(2.0) / 3.0,
    "rs_t": 5.72,
    "spinscale": 2.0 ** (1.0 / 3.0) - 1.0,
}

#: The correlation network's matrix is drawn from ``fourier_seed`` plus this
#: offset, the exchange network's from ``fourier_seed`` itself (as the
#: networks' own seeds are ``seed`` and ``seed + 1``), so the two are
#: independent draws.
CORRELATION_SEED_OFFSET = 1


def scales_for(network: str, descriptor_coordinates: str,
               descriptor_log_transform: bool, polarized: bool,
               extra_ranges: tuple) -> tuple:
    """The scales of ``network``'s MLP row ("x" or "c") under the RESOLVED
    coordinate set: the paper and dfs rows [x_s, *extras] and [x_0, x_1,
    x_s, *extras] (their correlation row is the polarized one), the legacy
    rows under the log transform [x_s, *extras] and [r_s transformed, x_s,
    (spin scale,) *extras]; each descriptor column is divided by the range
    its descriptor declares (``extra_ranges``, ``Descriptor.column_ranges``).
    The map is refused where the inputs are not bounded: the legacy set
    without the log transform feeds the raw reduced gradient and r_s."""
    extras = tuple(float(r) for r in extra_ranges)
    r = COORDINATE_RANGES
    if descriptor_coordinates in ("dfs", "paper"):
        if network == "x":
            return (r["x_s"],) + extras
        if not polarized:
            raise ValueError(
                "fourier_features: the paper and dfs coordinates feed the "
                "correlation network the polarized row [x_0, x_1, x_s]; "
                "use_polarized_correlation=False cannot carry the map")
        return (r["x_0"], r["x_1"], r["x_s"]) + extras
    if not descriptor_log_transform:
        raise ValueError(
            "fourier_features: the legacy coordinates without the descriptor "
            "log transform feed the raw reduced gradient and r_s, which are "
            "not bounded; the map needs the transformed inputs or the paper "
            "or dfs coordinates")
    if network == "x":
        return (r["x_s"],) + extras
    return ((r["rs_t"], r["x_s"]) + ((r["spinscale"],) if polarized else ())
            + extras)


def frequency_matrix(seed: int, m: int, scales, sigma: float) -> tuple:
    """The (m, d) frequency matrix as nested tuples of Python floats: sigma
    N(0, 1) draws from ``RandomState(seed)``, column j divided by
    ``scales[j]``."""
    draws = np.random.RandomState(int(seed)).standard_normal(
        (int(m), len(scales)))
    matrix = float(sigma) * draws / np.asarray(scales, dtype=float)[None, :]
    return tuple(tuple(float(v) for v in row) for row in matrix)


def digest(*matrices) -> str | None:
    """The first sixteen hex digits of the SHA-256 of the matrices' float64
    bytes in order, the value the class record states; ``None`` when every
    matrix is ``None`` (no map)."""
    if all(matrix is None for matrix in matrices):
        return None
    sha = hashlib.sha256()
    for matrix in matrices:
        if matrix is not None:
            sha.update(np.asarray(matrix, dtype=np.float64).tobytes())
    return sha.hexdigest()[:16]


def digest_for(fourier_features: int, fourier_scale: float,
               fourier_seed: int, extra_ranges: tuple,
               descriptor_coordinates: str, descriptor_log_transform: bool,
               polarized: bool) -> str | None:
    """The digest an architecture's networks carry, from its resolved fields
    alone: the exchange matrix then the correlation matrix (``None`` without
    a map)."""
    if not fourier_features:
        return None
    matrices = []
    for network, seed in (("x", fourier_seed),
                          ("c", fourier_seed + CORRELATION_SEED_OFFSET)):
        matrices.append(frequency_matrix(
            seed, fourier_features,
            scales_for(network, descriptor_coordinates,
                       descriptor_log_transform, polarized, extra_ranges),
            fourier_scale))
    return digest(*matrices)
