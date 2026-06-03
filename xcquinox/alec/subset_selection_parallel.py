"""Parallel exhaustive subset selection for large candidate pools.

The serial :func:`xcquinox.alec.subset_selection.select_subset` enumerates all
``C(npool, r)`` combinations in one process, fine for the 26-point training pool
but intractable as a single stream for the ~216-reaction BH76+W4-11 pool. The
enumeration is embarrassingly parallel, so this module partitions the combination
space across worker processes, evaluates the JSD metric on each chunk with the
per-point descriptor histograms precomputed once, and keeps only the running
argmin (nothing per-combo is written to disk).

Correctness contract: ``select_subset_parallel(..., metric="jsd")`` returns the
SAME ``(chosen, value)`` as the serial ``select_subset`` (same lex-first
tie-break): verified in tests on small pools.

Partition scheme (adapts granularity to ``r`` for load balance):
  * ``r == 1`` -> 1 task (216 singletons, trivial),
  * ``r == 2`` -> one task per leading index (``npool`` tasks),
  * ``r >= 3`` -> one task per leading PAIR of indices (``C(npool, 2)`` tasks),
each task fixes that prefix and enumerates the remaining ``r - p`` indices over
the suffix via ``itertools.combinations``, batched into numpy for a vectorized
JSD. ``multiprocessing.Pool`` load-balances the (uneven) tasks; a shared
``initializer`` hands every worker the precomputed count matrix once.
"""
import itertools
import multiprocessing as _mp
import os

import numpy as np

from xcquinox.alec.subset_selection import (
    NBINS,
    KL_PROB_CLIP,
    _DESCRIPTOR_KEYS,
    _prebin_pool,
    _to_pmf,
    _resolve_descriptor_weights,
)

# Worker-global scratch populated by ``_init_worker`` (one copy per process, set
# once via the Pool initializer so the count matrix is not re-pickled per task).
_G: dict = {}


def _init_worker(counts, ref_pmf, weights, batch):
    _G["counts"] = counts          # {k: (npool, NBINS) float64 raw counts}
    _G["ref_pmf"] = ref_pmf        # {k: (NBINS,) reference PMF}
    _G["weights"] = weights        # {k: float}
    _G["batch"] = int(batch)


def _jsd_batch(combo_idx):
    """Vectorized JSD of a batch of subsets against the reference.

    ``combo_idx`` is an ``(B, r)`` int array of pool indices. Subset histogram =
    index-sum of the precomputed per-point counts (B×r×NBINS, no ``npool``
    factor). Matches ``subset_selection.metric_jsd`` exactly, including the
    inf-on-zero-in-range-mass rule. Returns ``(B,)`` JSD values."""
    counts = _G["counts"]
    ref_pmf = _G["ref_pmf"]
    weights = _G["weights"]
    total = np.zeros(combo_idx.shape[0], dtype=np.float64)
    bad = np.zeros(combo_idx.shape[0], dtype=bool)
    for k in _DESCRIPTOR_KEYS:
        sub = counts[k][combo_idx].sum(axis=1)          # (B, NBINS)
        ssum = sub.sum(axis=1)                          # (B,)
        bad |= ssum <= 0.0
        q = sub / np.maximum(ssum[:, None], 1e-300)     # candidate PMF
        p = ref_pmf[k][None, :]                         # (1, NBINS)
        m = 0.5 * (p + q)
        pc = np.maximum(p, KL_PROB_CLIP)
        qc = np.maximum(q, KL_PROB_CLIP)
        mc = np.maximum(m, KL_PROB_CLIP)
        kl_pm = np.sum(pc * (np.log(pc) - np.log(mc)), axis=1)
        kl_qm = np.sum(qc * (np.log(qc) - np.log(mc)), axis=1)
        total += weights[k] * 0.5 * (kl_pm + kl_qm)
    total[bad] = np.inf
    return total


def _scan_prefix(prefix):
    """Worker task: scan every combination whose fixed leading indices are
    ``prefix`` (a tuple of length p), enumerating the remaining indices over the
    suffix. Returns ``(best_val, best_combo)`` for this task (lex-first on ties)."""
    npool = _G["counts"][_DESCRIPTOR_KEYS[0]].shape[0]
    r_total = _G["r"]
    p = len(prefix)
    rem = r_total - p
    suffix_start = (prefix[-1] + 1) if prefix else 0
    batch = _G["batch"]
    prefix_arr = np.asarray(prefix, dtype=np.int64)

    best_val = np.inf
    best_combo = None
    it = itertools.combinations(range(suffix_start, npool), rem)
    while True:
        chunk = list(itertools.islice(it, batch))
        if not chunk:
            break
        b = len(chunk)
        arr = np.empty((b, r_total), dtype=np.int64)
        if p:
            arr[:, :p] = prefix_arr
        arr[:, p:] = np.asarray(chunk, dtype=np.int64)
        vals = _jsd_batch(arr)
        j = int(np.argmin(vals))
        cand_val = float(vals[j])
        cand_combo = tuple(int(x) for x in arr[j])
        if (cand_val < best_val) or (
            cand_val == best_val and (best_combo is None or cand_combo < best_combo)
        ):
            best_val = cand_val
            best_combo = cand_combo
    return best_val, best_combo


def _init_worker_with_r(counts, ref_pmf, weights, batch, r):
    _init_worker(counts, ref_pmf, weights, batch)
    _G["r"] = int(r)


def _partition_prefixes(npool, r):
    """Yield the fixed-prefix tasks partitioning C(npool, r). Prefix length p =
    min(2, r-1): r=1 -> () (single task), r=2 -> leading index, r>=3 -> leading pair."""
    p = min(2, max(0, r - 1))
    if p == 0:
        yield ()
        return
    yield from itertools.combinations(range(npool), p)


def select_subset_parallel(pool, edges, h_ref, *, r, metric="jsd",
                           n_jobs=None, descriptor_weights=None,
                           batch=None):
    """Parallel exhaustive size-``r`` subset selection minimizing the metric.

    Returns ``(best_combo, best_value)`` identical to
    :func:`subset_selection.select_subset` (lex-first tie-break). Only
    ``metric="jsd"`` is supported (the held-out representative-subset use case).

    Parameters mirror ``select_subset`` plus ``n_jobs`` (worker count; default =
    ``parallel.detect_available_cpus()``) and ``batch`` (combos per numpy batch;
    default ``STEP7_SUBSET_BATCH`` env or 16384)."""
    if metric != "jsd":
        raise ValueError(
            f"select_subset_parallel supports metric='jsd' only, got {metric!r}")
    npool = len(pool)
    if not (1 <= r <= npool):
        raise ValueError(f"r={r} out of range for npool={npool}")

    weights = _resolve_descriptor_weights(descriptor_weights)
    per_key_counts, _w_in_range = _prebin_pool(pool, edges)
    ref_pmf = {k: _to_pmf(h_ref[k]) for k in _DESCRIPTOR_KEYS}
    if batch is None:
        batch = int(os.environ.get("STEP7_SUBSET_BATCH", "16384"))
    if n_jobs is None:
        from xcquinox.alec.parallel import detect_available_cpus
        n_jobs = detect_available_cpus()
    n_jobs = max(1, int(n_jobs))

    prefixes = list(_partition_prefixes(npool, r))
    initargs = (per_key_counts, ref_pmf, weights, batch, r)

    if n_jobs == 1 or len(prefixes) == 1:
        _init_worker_with_r(*initargs)
        results = [_scan_prefix(pfx) for pfx in prefixes]
    else:
        # 'spawn' (not fork): the parent typically has pyscf/OpenMP threads
        # loaded by the descriptor extraction, and forking after threads can
        # deadlock. Spawn workers re-import only this module + subset_selection
        # (numpy/ase, no jax/pyscf), so startup is light; the count matrix is
        # handed over once via the initializer.
        ctx = _mp.get_context("spawn")
        with ctx.Pool(processes=min(n_jobs, len(prefixes)),
                      initializer=_init_worker_with_r,
                      initargs=initargs) as pool_:
            # chunksize>1 amortizes dispatch over the many small leading-pair tasks
            results = pool_.map(_scan_prefix, prefixes,
                                chunksize=max(1, len(prefixes) // (n_jobs * 8)))

    best_val = np.inf
    best_combo = None
    for val, combo in results:
        if combo is None:
            continue
        if (val < best_val) or (
            val == best_val and (best_combo is None or combo < best_combo)
        ):
            best_val = val
            best_combo = combo
    return best_combo, best_val
