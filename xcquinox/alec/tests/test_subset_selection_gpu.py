"""GPU JSD selector: on-device combinadic unranking correctness, equality with
the serial reference, the inf-on-zero-mass guard, and batch invariance. Runs on
JAX-CPU (the kernel is device-agnostic), so no GPU is needed in CI."""
import itertools
import math

import numpy as np
import pytest

from xcquinox.alec import subset_selection as ss
from xcquinox.alec import subset_selection_gpu as ssg


def _synth_pool(n, seed, ngrid=60):
    rng = np.random.default_rng(seed)
    pool = []
    for i in range(n):
        pool.append({
            "rho_third": np.abs(rng.standard_normal(ngrid)) + 0.05 * (i + 1),
            "s": np.abs(rng.standard_normal(ngrid)) + 0.02 * (i + 1),
            "alpha": np.abs(rng.standard_normal(ngrid)),
            "weights": np.abs(rng.standard_normal(ngrid)) + 0.01,
        })
    return pool


# ---------------------------------------------------------------------------
# Combinadic unranking is a bijection onto all C(n, r) combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,r", [(6, 3), (8, 4), (9, 2), (7, 5), (10, 1)])
def test_unrank_is_lex_bijection(n, r):
    C = ssg._binomial_table(n, r)
    total = math.comb(n, r)
    combos = ssg._unrank_batch_np(np.arange(total), n, r, C)
    got = [tuple(int(x) for x in row) for row in combos]
    expected = list(itertools.combinations(range(n), r))
    # each row is an ascending tuple; the map is a bijection onto ALL
    # combinations (combinadic enumeration order, not lex, order is irrelevant
    # since selection takes the argmin over every combination).
    assert all(list(g) == sorted(g) for g in got)
    assert len(set(got)) == total
    assert sorted(got) == expected


# ---------------------------------------------------------------------------
# GPU selector == serial select_subset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r", [2, 3, 4])
def test_gpu_matches_serial_jsd(r):
    pool = _synth_pool(8, seed=r)
    h_ref, edges = ss.build_reference_histograms(pool)
    s_combo, s_val = ss.select_subset(
        pool, edges, h_ref, r=r, metric="jsd", progress=False)
    g_combo, g_val = ssg.select_subset_gpu(
        pool, edges, h_ref, r=r, metric="jsd", batch=10_000)
    assert tuple(g_combo) == tuple(s_combo)
    assert g_val == pytest.approx(s_val, rel=1e-9, abs=1e-12)


def test_gpu_batch_invariance():
    pool = _synth_pool(8, seed=5)
    h_ref, edges = ss.build_reference_histograms(pool)
    small = ssg.select_subset_gpu(pool, edges, h_ref, r=3, metric="jsd", batch=7)
    big = ssg.select_subset_gpu(pool, edges, h_ref, r=3, metric="jsd",
                                batch=100_000)
    assert tuple(small[0]) == tuple(big[0])
    assert small[1] == pytest.approx(big[1])


def test_gpu_rejects_non_jsd_metric():
    pool = _synth_pool(5, seed=1)
    h_ref, edges = ss.build_reference_histograms(pool)
    with pytest.raises(ValueError, match="jsd"):
        ssg.select_subset_gpu(pool, edges, h_ref, r=2, metric="l2")


# ---------------------------------------------------------------------------
# inf-on-zero-in-range-mass guard (kernel level)
# ---------------------------------------------------------------------------

def test_kernel_marks_zero_mass_subset_infinite():
    import jax.numpy as jnp
    n, r = 3, 2
    C = ssg._binomial_table(n, r)
    # points 0 and 1 carry NO mass; only point 2 does. The size-2 subset {0,1}
    # therefore has zero in-range mass -> JSD must be +inf (never selected).
    counts = np.array([[[0., 0.], [0., 0.], [1., 1.]]], dtype=np.float32)
    pref = np.array([[0.5, 0.5]], dtype=np.float32)
    pref_logpref = (pref * np.log(pref)).astype(np.float32)
    w = np.array([1.0], dtype=np.float32)
    kernel = ssg._make_kernel(r, 1)

    vals = []
    for t in range(math.comb(n, r)):
        v, _idx = kernel(jnp.array([t], dtype=jnp.int64), jnp.asarray(C),
                         jnp.asarray(counts), jnp.asarray(pref),
                         jnp.asarray(pref_logpref), jnp.asarray(w))
        vals.append(float(v))
    assert sum(math.isinf(v) for v in vals) == 1     # exactly {0,1}
    assert all(not math.isinf(v) for v, combo in zip(
        vals, itertools.combinations(range(n), r)) if 2 in combo)
