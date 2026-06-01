"""GPU-accelerated exhaustive subset selection (JSD) for large pools.

The CPU selector (:mod:`subset_selection_parallel`) caps at ~0.3M combos/s — the
per-combo JSD is logs over the descriptor bins — so exhaustive selection over the
216-reaction BH76+W4-11 pool is infeasible past r≈5 (r=7 has 3.9e12 combos). This
module runs the whole sweep on a JAX device: combinations are generated
**on-device** by vectorized combinadic unranking (so the host never enumerates
3.9e12 tuples), and unrank → JSD → batch-argmin happen in one jit'd kernel.

Correctness contract: ``select_subset_gpu(..., metric="jsd")`` returns the SAME
``(chosen, value)`` as :func:`subset_selection.select_subset` — the batch ranking
runs in float32 for speed, then the single global-winner combo's JSD is recomputed
in float64 on the host via the exact ``metric_jsd`` path. Verified in tests.
"""
import math

import numpy as np

from xcquinox.alec import subset_selection as ss
from xcquinox.alec.subset_selection import (
    NBINS,
    KL_PROB_CLIP,
    _DESCRIPTOR_KEYS,
    _prebin_pool,
    _to_pmf,
    _resolve_descriptor_weights,
    _bin_with_edges,
    metric_jsd,
)


def _binomial_table(n, r):
    """``C[i, k] = comb(i, k)`` for ``0<=i<=n``, ``0<=k<=r`` (int64; exact)."""
    C = np.zeros((n + 1, r + 1), dtype=np.int64)
    for i in range(n + 1):
        kmax = min(i, r)
        for k in range(kmax + 1):
            C[i, k] = math.comb(i, k)
    return C


def _unrank_batch_np(ranks, n, r, C):
    """Host (numpy) combinadic unranking — used by tests to validate the kernel.

    ``ranks`` (B,) in ``[0, C(n,r))`` → ``(B, r)`` combos (ascending). For
    ``i=r..1``: ``c = searchsorted(C[:,i], t, 'right') - 1``; ``t -= C[c, i]``."""
    t = np.asarray(ranks, dtype=np.int64).copy()
    cols = []
    for i in range(r, 0, -1):
        col = C[:, i]
        c = np.searchsorted(col, t, side="right") - 1
        cols.append(c)
        t = t - C[c, i]
    combos = np.stack(cols[::-1], axis=1)          # ascending (c_1 < ... < c_r)
    return combos


def _make_kernel(r, ndesc):
    """Build the jit'd (unrank → JSD → batch-argmin) kernel for fixed r/ndesc."""
    import jax
    import jax.numpy as jnp

    def kernel(ranks, C, counts, pref, pref_logpref, ref_weights):
        # ranks (B,) int64; C (n+1, r+1) int64; counts (ndesc, n, NBINS) f32;
        # pref/pref_logpref (ndesc, NBINS) f32 = max(ref_pmf,CLIP) and pc*log(pc).
        t = ranks
        cols = []
        for i in range(r, 0, -1):
            c = jnp.searchsorted(C[:, i], t, side="right") - 1
            cols.append(c)
            t = t - C[c, i]
        combos = jnp.stack(cols, axis=1)           # (B, r), order irrelevant (a set)

        B = ranks.shape[0]
        total = jnp.zeros(B, dtype=jnp.float32)
        bad = jnp.zeros(B, dtype=bool)
        for d in range(ndesc):
            cd = counts[d]                          # (n, NBINS)
            sub = cd[combos[:, 0]]
            for tt in range(1, r):
                sub = sub + cd[combos[:, tt]]       # incremental: no (B,r,NBINS)
            ssum = sub.sum(axis=1)
            bad = bad | (ssum <= 0.0)
            q = sub / jnp.maximum(ssum[:, None], jnp.float32(1e-30))
            p = pref[d][None, :]
            m = 0.5 * (p + q)
            qc = jnp.maximum(q, jnp.float32(KL_PROB_CLIP))
            mc = jnp.maximum(m, jnp.float32(KL_PROB_CLIP))
            logmc = jnp.log(mc)
            kl = jnp.sum(pref_logpref[d][None, :] - p * logmc
                         + qc * jnp.log(qc) - qc * logmc, axis=1)
            total = total + ref_weights[d] * 0.5 * kl
        total = jnp.where(bad, jnp.inf, total)
        j = jnp.argmin(total)
        return total[j], j

    return jax.jit(kernel)


def select_subset_gpu(pool, edges, h_ref, *, r, metric="jsd", batch=None,
                      descriptor_weights=None):
    """Exhaustive size-``r`` subset selection minimizing JSD, on a JAX device.

    Returns ``(best_combo, best_value)`` matching
    :func:`subset_selection.select_subset` (lex-first tie-break on real data).
    Only ``metric="jsd"`` is supported. Raises ``RuntimeError`` if no JAX device
    is available so the caller can fall back to the CPU selector."""
    if metric != "jsd":
        raise ValueError(
            f"select_subset_gpu supports metric='jsd' only, got {metric!r}")
    npool = len(pool)
    if not (1 <= r <= npool):
        raise ValueError(f"r={r} out of range for npool={npool}")

    try:
        import jax
        # x64 is REQUIRED: ranks/binomials reach C(216,7)=3.9e12 >> int32 max,
        # so without 64-bit ints the unranking arithmetic would silently
        # overflow. (Matches the eval pipeline, which also enables x64.)
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        _ = jax.devices()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"JAX device unavailable: {exc!r}") from exc

    weights = _resolve_descriptor_weights(descriptor_weights)
    per_key_counts, _w = _prebin_pool(pool, edges)
    ndesc = len(_DESCRIPTOR_KEYS)
    counts = np.stack([per_key_counts[k] for k in _DESCRIPTOR_KEYS]
                      ).astype(np.float32)                       # (ndesc, n, NBINS)
    ref_pmf = np.stack([_to_pmf(h_ref[k]) for k in _DESCRIPTOR_KEYS])
    pref = np.maximum(ref_pmf, KL_PROB_CLIP).astype(np.float32)
    pref_logpref = (pref * np.log(pref)).astype(np.float32)
    ref_weights = np.array([weights[k] for k in _DESCRIPTOR_KEYS],
                           dtype=np.float32)

    C = _binomial_table(npool, r)
    total = int(C[npool, r])
    if batch is None:
        batch = 1 << 20

    C_j = jnp.asarray(C)
    counts_j = jnp.asarray(counts)
    pref_j = jnp.asarray(pref)
    pref_logpref_j = jnp.asarray(pref_logpref)
    ref_weights_j = jnp.asarray(ref_weights)
    kernel = _make_kernel(r, ndesc)

    best_val = math.inf
    best_rank = -1
    start = 0
    while start < total:
        end = min(start + batch, total)
        ranks = jnp.arange(start, end, dtype=jnp.int64)
        val, idx = kernel(ranks, C_j, counts_j, pref_j, pref_logpref_j,
                          ref_weights_j)
        val = float(val)
        rank = start + int(idx)
        if val < best_val or (val == best_val and 0 <= rank < best_rank):
            best_val = val
            best_rank = rank
        start = end

    best_combo = tuple(int(x) for x in
                       _unrank_batch_np(np.array([best_rank]), npool, r, C)[0])
    # Exact float64 JSD of the winner via the serial path (matches select_subset).
    cat = {k: np.concatenate([pool[i][k] for i in best_combo])
           for k in _DESCRIPTOR_KEYS}
    cat["weights"] = np.concatenate([pool[i].get(
        "weights", np.ones_like(pool[i]["rho_third"])) for i in best_combo])
    exact_val = float(metric_jsd(h_ref, _bin_with_edges(cat, edges),
                                 weights=weights))
    return best_combo, exact_val
