"""Step-7 histogram-matched subset selection from Dick 2021 training pool.

This module ports the legacy data_binning2.ipynb cell-17 algorithm into
the alec subpackage and extends it with a Jensen-Shannon divergence
metric in addition to the original Euclidean L2-on-bins metric.

Binning: all three descriptors are binned on a LINEAR scale, with
per-feature ``[0.1, 99.9]``-percentile edges. The legacy method used
``log10(x + 1e-10)`` to give cross-scale feature values equal treatment, but
the descriptors are ALREADY scale-compressed (ρ^{1/3}, not ρ) and each feature
is independently percentile-ranged, so that range is handled by the descriptor
choice + percentile edges, log on top was largely redundant and dumped the
physical α=0 (single-orbital) spike onto a floor bin. Cross-FEATURE equal
treatment is instead enforced by normalizing BOTH metrics to PMFs (sum=1) per
marginal (``_to_pmf``), so a feature's contribution no longer depends on its
raw binwidth. Distributions are still PLOTTED on log-scale axes for readability.

Three-descriptor objective: histograms over (ρ^{1/3}, s, α) where
- s is the PBE-1996 reduced gradient (Perdew, Burke, Ernzerhof, PRL 77, 3865, 1996)
- α is the SCAN-2015 iso-orbital indicator (Sun, Ruzsinszky, Perdew,
  PRL 115, 036402, 2015, eq. 4); used for subset selection only,
  NOT consumed by the trained GGA network.

Candidate pool is Dick & Fernandez-Serra 2021 SI §II training data:
21 G2/97 atomization-energy entries + 3 BH76 reactions + 2 IP13 IPs
+ 2 atomic-density references = 28 distinct training points. Selection
varies the 21 AE entries; auxiliaries are fixed per Dick's protocol.

Public API:
- extract_descriptors(atoms_obj, *, basis="def2-svp", grid_level=1, cache_dir)
- build_reference_histograms(descriptor_arrays, *, nbins=200)
- metric_l2(h_ref, h_cand) -> float       # 3-histogram sum
- metric_jsd(h_ref, h_cand) -> float      # 3-histogram sum, nats
- select_subset(pool, r, metric, fixed_indices=())
- compute_atom_set(ae_subset_atoms_list)
- augment_with_hbpt(ae_subset_atoms_list, atom_refs, *, with_hbpt: bool)
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from itertools import combinations
from typing import Callable, Iterable

import numpy as np
import ase
from ase import Atoms
from ase.io import read, write

# Constants ---------------------------------------------------------------
NBINS = 200
KL_PROB_CLIP = 1e-12

# HB and PT water-dimer geometries verbatim from
# /home/awills/Documents/Research/Python/jup/data_binning2.ipynb cell 20.
# Original at.info: basis='6-311++G(3df,2pd)', grid_level=4. Step-7
# overrides these to def2-svp / grid_level=1 to keep histograms commensurate
# with the rest of the candidate pool.
_HB_POSITIONS = (
    (1.317021, -0.128356, 0.006258),
    (1.527437, 0.387478, -0.795622),
    (1.505382, 0.474880, 0.750724),
    (-1.017021, 0.128356, 0.006258),
    (-1.227437, -0.387478, -0.795622),
    (-1.205382, -0.474880, 0.750724),
)
_PT_POSITIONS = (
    (1.310944, -0.092374, 0.053983),
    (1.955110, 0.571413, -0.263648),
    (-0.101366, 0.045774, -0.012031),
    (-1.149037, 0.029559, -0.084434),
    (-1.608104, 0.722348, 0.414070),
    (-1.540923, -0.836961, 0.105186),
)
_HB_SYMBOLS = "OHHOHH"
_PT_SYMBOLS = "OHHOHH"


def compute_descriptor_triple(
    rho: np.ndarray,
    sigma: np.ndarray,
    tau: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute (ρ^{1/3}, s, α) from SCF outputs on the integration grid.

    Parameters
    ----------
    rho : (N,) electron density on grid points
    sigma : (N,) |∇ρ|² (gradient squared)
    tau : (N,) kinetic-energy density

    Returns
    -------
    dict with keys "rho_third", "s", "alpha", each (N,) ndarray. α is
    clipped at 0 to handle grid noise in low-density tails.

    Formulas:
    - s = |∇ρ| / [2 (3π²)^{1/3} ρ^{4/3}]   (PBE 1996, before eq. 12)
    - τ_W = |∇ρ|²/(8ρ),  τ_unif = (3/10)(3π²)^{2/3} ρ^{5/3}
    - α = (τ - τ_W) / τ_unif               (SCAN 2015, eq. 4)
    """
    rho_safe = np.maximum(rho, 1e-30)
    grad_rho = np.sqrt(np.maximum(sigma, 0.0))
    rho_third = rho_safe ** (1.0 / 3.0)
    kf_factor = 2.0 * (3.0 * np.pi**2) ** (1.0 / 3.0)
    s = grad_rho / (kf_factor * rho_safe ** (4.0 / 3.0))
    tau_w = sigma / (8.0 * rho_safe)
    tau_unif = (
        (3.0 / 10.0) * (3.0 * np.pi**2) ** (2.0 / 3.0) * rho_safe ** (5.0 / 3.0)
    )
    alpha = np.maximum((tau - tau_w) / np.maximum(tau_unif, 1e-30), 0.0)
    return {"rho_third": rho_third, "s": s, "alpha": alpha}


_DESCRIPTOR_KEYS = ("rho_third", "s", "alpha")

# Per-descriptor weights for the selection objective. DEFAULT is equal
# (1.0 each) so existing subset selections reproduce EXACTLY. Rationale for
# making it configurable: the trained GGA model is structurally blind to tau, so
# the meta-GGA descriptor ``alpha = (tau - tau_W)/tau_unif`` is a coordinate the
# functional cannot represent. A caller that wants the selection to optimize
# only descriptors the GGA can see may pass ``descriptor_weights={"alpha": 0.0}``
# to ``select_subset`` (this CHANGES which subset is selected, so it is opt-in,
# never the default).
_DEFAULT_DESCRIPTOR_WEIGHTS = {k: 1.0 for k in _DESCRIPTOR_KEYS}


def _resolve_descriptor_weights(descriptor_weights):
    """Return a complete {descriptor: weight} dict (missing keys default to 1.0,
    unknown keys rejected)."""
    if descriptor_weights is None:
        return dict(_DEFAULT_DESCRIPTOR_WEIGHTS)
    unknown = set(descriptor_weights) - set(_DESCRIPTOR_KEYS)
    if unknown:
        raise ValueError(
            f"descriptor_weights has unknown descriptor(s) {sorted(unknown)}; "
            f"valid keys are {_DESCRIPTOR_KEYS}")
    return {k: float(descriptor_weights.get(k, 1.0)) for k in _DESCRIPTOR_KEYS}


def metric_l2(h_ref: dict, h_cand: dict, weights=None) -> float:
    """Per-bin Euclidean distance summed across the 3 marginals, on PMFs.

    err = sum_b sqrt( (p^ref_rho - p^cand_rho)^2_b
                    + (p^ref_s   - p^cand_s)^2_b
                    + (p^ref_a   - p^cand_a)^2_b )

    where each ``p`` is the marginal normalized to a probability MASS function
    (sum=1) via ``_to_pmf``: like ``metric_jsd``. PMF-normalization makes the
    three marginals dimensionless and binwidth-independent, so each descriptor
    is treated equally regardless of its raw range (the prior ``density=True``
    form let a narrow-range descriptor dominate). ``weights`` scales each
    descriptor's squared contribution; None = equal. A candidate marginal with
    zero in-range mass is maximally divergent (cannot represent the reference)
 -> +inf, never selected (mirrors ``metric_jsd``).
    """
    w = _resolve_descriptor_weights(weights)
    diffs_sq = np.zeros(NBINS)
    for k in _DESCRIPTOR_KEYS:
        if float(np.sum(h_cand[k])) <= 0.0:
            return float("inf")
        p = _to_pmf(h_ref[k])
        q = _to_pmf(h_cand[k])
        diffs_sq += w[k] * (p - q) ** 2
    return float(np.sum(np.sqrt(diffs_sq)))


def _to_pmf(h: np.ndarray) -> np.ndarray:
    """Normalize a non-negative histogram to a probability MASS function.

    Lin's JSD (Lin 1991, IEEE Trans. Inf. Theory 37, 145) is defined on
    PMFs that SUM to 1, not probability densities (which integrate to 1
    via the bin width, so their raw sum is bin-width-dependent and their
    individual values can exceed 1). We divide by the total mass so the
    result sums to 1; this also makes the divergence invariant to a
    uniform rescale (e.g. differing per-descriptor bin widths). A
    descriptor whose grid points all fall outside the histogram range has
    zero total mass; we return an all-zero vector for it. Callers
    (:func:`metric_jsd`, :func:`_metric_jsd_batch`) detect a zero-mass
    CANDIDATE marginal directly and treat it as maximally divergent
    (``+inf``) so it is never selected, ``_kl`` itself does not flag it.
    """
    total = float(np.sum(h))
    if total <= 0.0:
        return np.zeros_like(h, dtype=np.float64)
    return np.asarray(h, dtype=np.float64) / total


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback-Leibler divergence in nats between two PMFs.

    Probabilities are lower-floored at KL_PROB_CLIP to avoid log(0); there
    is NO upper clip, a PMF entry never exceeds 1 once normalized, so
    upper-clipping would distort legitimate (density) peaks. Inputs are
    normalized to PMFs (sum=1) before the divergence so the result is the
    genuine KL divergence bounded such that the resulting JSD lies in
    [0, ln 2].
    """
    p_pmf = _to_pmf(p)
    q_pmf = _to_pmf(q)
    p_c = np.maximum(p_pmf, KL_PROB_CLIP)
    q_c = np.maximum(q_pmf, KL_PROB_CLIP)
    return float(np.sum(p_c * (np.log(p_c) - np.log(q_c))))


def metric_jsd(h_ref: dict, h_cand: dict, weights=None) -> float:
    """Jensen-Shannon divergence summed across the 3 marginals (nats).

    JSD(P||Q) = 0.5 [ KL(P||M) + KL(Q||M) ],   M = (P+Q)/2.

    Reference: Lin, IEEE Trans. Inf. Theory 37 (1991) eq. (4.1).
    ``weights`` scales each marginal's JSD; None = equal (default).

    Each marginal histogram is normalized to a probability MASS function
    (sum=1) before the divergence (see :func:`_to_pmf`); the per-marginal
    JSD is therefore bounded in [0, ln 2] (Lin 1991), and the 3-marginal
    total in [0, 3 ln 2].

    NOTE: do NOT use scipy.spatial.distance.jensenshannon: that returns
    the JS distance (sqrt of the divergence), not the divergence itself.
    """
    w = _resolve_descriptor_weights(weights)
    total = 0.0
    for k in _DESCRIPTOR_KEYS:
        # a candidate marginal with zero in-range mass has no
        # grid points in this descriptor's range, it cannot represent the
        # reference distribution at all, so it is MAXIMALLY divergent and must
        # never be selected. Without this guard _to_pmf -> all-zeros makes
        # M = 0.5*p, KL(p||M)=0, and KL(0||M) stays small, so the candidate
        # would score ~0 (a spurious "perfect match"). The batch path
        # (_metric_jsd_batch) already disqualifies such rows; mirror it here.
        if float(np.sum(h_cand[k])) <= 0.0:
            return float("inf")
        p = _to_pmf(h_ref[k])
        q = _to_pmf(h_cand[k])
        m = 0.5 * (p + q)
        total += w[k] * 0.5 * (_kl(p, m) + _kl(q, m))
    return float(total)


def _bin_with_edges(arrs: dict, edges: dict) -> dict:
    """Bin descriptors using fixed pre-computed LINEAR edges; returns 3 histograms.

    Bins the raw descriptor values (no log10). ``density=True`` is retained, but
    the metrics PMF-normalize, so the binwidth factor cancels regardless.
    """
    out = {}
    w = arrs.get("weights")
    for k in _DESCRIPTOR_KEYS:
        h, _ = np.histogram(arrs[k], bins=edges[k], weights=w, density=True)
        out[k] = h
    return out


def _prebin_pool(pool, edges: dict) -> tuple[dict, dict]:
    """Pre-compute per-pool-entry weight-counts per bin and per-descriptor
    in-range total weight.

    Density-normalized histograms are additive over disjoint samples:
    for a union of pool entries (i1, ..., ik), descriptor key k,
    ``np.histogram(..., density=True)`` returns
    ``sum(weight_counts_i_k) / (sum(W_in_range_i_k) * bin_width_k)``
    where ``W_in_range_i_k = per_key_counts[k][i, :].sum()`` is the
    weight of grid points from pool entry i whose (linear) descriptor
    fell INSIDE the histogram range (samples outside the range are
    dropped by ``np.histogram`` and excluded from its normalization).
    The in-range weight differs across descriptors because the (rho^1/3,
    s, alpha) ranges are picked independently from the [0.1, 99.9]
    percentiles in ``build_reference_histograms``, so we must store
    a separate W_in_range[k] vector per descriptor.

    Returns (per_key_counts, W_in_range) where:
      - per_key_counts[k] has shape (npool, NBINS): unnormalized
        weight-counts per pool entry, descriptor k.
      - W_in_range[k] has shape (npool,): in-range total weight per
        pool entry, descriptor k.
    """
    npool = len(pool)
    per_key_counts = {k: np.zeros((npool, NBINS), dtype=np.float64)
                      for k in _DESCRIPTOR_KEYS}
    W_in_range = {k: np.zeros(npool, dtype=np.float64) for k in _DESCRIPTOR_KEYS}
    for i, arrs in enumerate(pool):
        w = arrs.get("weights")
        if w is None:
            w = np.ones_like(arrs["rho_third"])
        for k in _DESCRIPTOR_KEYS:
            h, _ = np.histogram(arrs[k], bins=edges[k], weights=w, density=False)
            per_key_counts[k][i, :] = h
            W_in_range[k][i] = float(h.sum())
    return per_key_counts, W_in_range


def _metric_l2_batch(h_ref: dict, h_cand_batch: dict, weights=None) -> np.ndarray:
    """Vectorized L2 metric across a batch of candidate histograms.

    Equivalent to ``[metric_l2(h_ref, {k: h_cand_batch[k][b] ...}) for b in batch]``
    but computed in a single numpy expression.  Returns shape ``(batch,)``.
    ``weights`` scales each descriptor's squared contribution; None = equal
    weights (default, behavior-preserving).
    """
    w = _resolve_descriptor_weights(weights)
    batch_size = next(iter(h_cand_batch.values())).shape[0]
    diffs_sq = np.zeros((batch_size, NBINS), dtype=np.float64)
    empty_row = np.zeros(batch_size, dtype=bool)
    for k in _DESCRIPTOR_KEYS:
        p_raw = h_ref[k][None, :]                          # (1, NBINS)
        q_raw = h_cand_batch[k]                            # (batch, NBINS)
        # PMF-normalize each row (sum=1); flag zero-mass candidates (SUBSET-05).
        q_mass = q_raw.sum(axis=1, keepdims=True)          # (batch, 1)
        empty_row |= (q_mass[:, 0] <= 0.0)
        q_mass_safe = np.where(q_mass > 0.0, q_mass, 1.0)
        p = p_raw / max(float(p_raw.sum()), KL_PROB_CLIP)
        q = q_raw / q_mass_safe
        diffs_sq += w[k] * (p - q) ** 2
    out = np.sqrt(diffs_sq).sum(axis=1)
    out[empty_row] = np.inf
    return out


def _metric_jsd_batch(h_ref: dict, h_cand_batch: dict, weights=None) -> np.ndarray:
    """Vectorized JSD metric across a batch of candidate histograms.

    Equivalent to ``[metric_jsd(h_ref, {k: h_cand_batch[k][b] ...}) for b in batch]``
    but computed in a single numpy expression. Returns shape ``(batch,)``.

    Each reference and candidate marginal is normalized to a PMF (sum=1)
    before the divergence (SUBSET-01): rows are divided by their total
    mass and entries are lower-floored at KL_PROB_CLIP to avoid log(0).
    There is NO upper clip. A candidate row whose mass is zero for ANY
    descriptor (all grid points fell outside the histogram range) is
    disqualified by returning +inf for that row (SUBSET-05), so it is
    never selected as the minimizer.
    """
    w = _resolve_descriptor_weights(weights)
    batch_size = next(iter(h_cand_batch.values())).shape[0]
    total = np.zeros(batch_size, dtype=np.float64)
    empty_row = np.zeros(batch_size, dtype=bool)
    for k in _DESCRIPTOR_KEYS:
        p_raw = h_ref[k][None, :]              # (1, NBINS)
        q_raw = h_cand_batch[k]                # (batch, NBINS)
        # Normalize each row to a PMF (sum=1).  Candidate rows with zero
        # total mass are degenerate (empty-in-range) and flagged below.
        q_mass = q_raw.sum(axis=1, keepdims=True)        # (batch, 1)
        empty_row |= (q_mass[:, 0] <= 0.0)
        q_mass_safe = np.where(q_mass > 0.0, q_mass, 1.0)
        p = p_raw / max(float(p_raw.sum()), KL_PROB_CLIP)
        q = q_raw / q_mass_safe
        m = 0.5 * (p + q)
        p_c = np.maximum(p, KL_PROB_CLIP)
        q_c = np.maximum(q, KL_PROB_CLIP)
        m_c = np.maximum(m, KL_PROB_CLIP)
        kl_pm = np.sum(p_c * (np.log(p_c) - np.log(m_c)), axis=1)
        kl_qm = np.sum(q_c * (np.log(q_c) - np.log(m_c)), axis=1)
        total += w[k] * 0.5 * (kl_pm + kl_qm)
    total[empty_row] = np.inf
    return total


def bin_descriptors(arrs: dict) -> dict:
    """Bin a single descriptor-array set with auto-computed LINEAR edges."""
    edges = {}
    for k in _DESCRIPTOR_KEYS:
        lo, hi = np.percentile(arrs[k], [0.1, 99.9])
        edges[k] = np.linspace(lo, hi, NBINS + 1)
    return _bin_with_edges(arrs, edges)


def extract_descriptors_for_species(
    species,
    *,
    basis: str = "def2-svp",
    grid_level: int = 1,
    cache_dir,
) -> dict[tuple, dict]:
    """Extract per-species descriptors and cache by ``(name, charge, spin)``.

    Returns a dict keyed by ``(name, charge, spin)`` whose values are the
    same descriptor dicts produced by :func:`extract_descriptors`. Used as
    the building block for multi-species TrainingPoints, each point's
    candidate descriptors are then assembled by concatenating its
    species' entries from this dict.
    """
    out: dict[tuple, dict] = {}
    seen: set[tuple] = set()
    for at in species:
        name = at.info.get("name") or at.info.get("dfs_hill") or at.get_chemical_formula()
        charge = int(at.info.get("charge", 0))
        spin = int(at.info.get("spin", 0))
        key = (name, charge, spin)
        if key in seen:
            continue
        seen.add(key)
        # Use a name+charge+spin-stable cache filename: dedupes across
        # points that share an atom anchor.
        idx = f"{name.replace('+','plus').replace('/', '_')}_c{charge}_s{spin}"
        out[key] = extract_descriptors(
            at, idx=idx, basis=basis, grid_level=grid_level, cache_dir=cache_dir
        )
    return out


def concatenate_point_descriptors(points, species_descriptors: dict[tuple, dict]) -> list[dict]:
    """For each TrainingPoint, concatenate descriptors across its species
    (design choice "a": full union of grid points across participating
    species; reactions weigh more proportional to grid-point count).

    Returns a list parallel to ``points`` whose i-th entry is the same
    shape as one ``extract_descriptors`` output (rho_third / s / alpha /
    weights), ready to feed into :func:`build_reference_histograms` /
    :func:`select_subset`.
    """
    out: list[dict] = []
    for tp in points:
        cat = {k: [] for k in _DESCRIPTOR_KEYS}
        cat_w: list = []
        for sp in tp.species:
            name = sp.info.get("name") or sp.info.get("dfs_hill") or sp.get_chemical_formula()
            charge = int(sp.info.get("charge", 0))
            spin = int(sp.info.get("spin", 0))
            key = (name, charge, spin)
            if key not in species_descriptors:
                raise KeyError(
                    f"concatenate_point_descriptors: TrainingPoint {tp.name!r} "
                    f"references species {key} not in species_descriptors. Run "
                    f"extract_descriptors_for_species() over the union of "
                    f"every point's species first."
                )
            d = species_descriptors[key]
            for k in _DESCRIPTOR_KEYS:
                cat[k].append(d[k])
            cat_w.append(d.get("weights", np.ones_like(d["rho_third"])))
        out.append({
            "rho_third": np.concatenate(cat["rho_third"]),
            "s":         np.concatenate(cat["s"]),
            "alpha":     np.concatenate(cat["alpha"]),
            "weights":   np.concatenate(cat_w),
        })
    return out


def build_reference_histograms(pool):
    """Concatenate descriptor arrays across the full candidate pool, build
    the 3 reference 200-bin LINEAR-edge histograms (per-feature [0.1, 99.9]
    percentile range; metrics PMF-normalize), and return the edges used so
    that candidate-subset histograms align."""
    cat: dict = {k: [] for k in _DESCRIPTOR_KEYS}
    cat_w: list = []
    for arrs in pool:
        for k in _DESCRIPTOR_KEYS:
            cat[k].append(arrs[k])
        cat_w.append(arrs.get("weights", np.ones_like(arrs["rho_third"])))
    full = {k: np.concatenate(cat[k]) for k in _DESCRIPTOR_KEYS}
    full["weights"] = np.concatenate(cat_w)
    edges = {}
    for k in _DESCRIPTOR_KEYS:
        # Linear-scale edges on the RAW descriptor (no log10): [0.1, 99.9]
        # percentile range per feature clips outliers and gives cross-feature
        # comparability; uniform-width bins within the range.
        lo, hi = np.percentile(full[k], [0.1, 99.9])
        edges[k] = np.linspace(lo, hi, NBINS + 1)
    h_ref = _bin_with_edges(full, edges)
    return h_ref, edges


def select_subset(
    pool,
    edges: dict,
    h_ref: dict,
    *,
    r: int,
    metric: str,
    fixed_indices: tuple = (),
    progress: bool = True,
    progress_desc: str | None = None,
    return_all: bool = False,
    distribution_path: str | None = None,
    descriptor_weights: dict | None = None,
):
    """Exhaustively enumerate all C(npool, r) subsets and return the
    indices of the size-r combination that minimizes the chosen metric.

    Parameters
    ----------
    pool : list of per-species descriptor-array dicts
    edges : pre-built bin edges from build_reference_histograms
    h_ref : reference 3-histogram tuple from build_reference_histograms
    r : target subset size
    metric : "l2" or "jsd"
    fixed_indices : pool indices that must be present in every candidate
        subset. The chosen subset has exactly r entries TOTAL including
        the fixed ones.
    progress : when True (default), wraps the combinatorial enumeration in
        a tqdm.auto progress bar so long-running r (e.g. r=14 over 28
        gives ~40M combinations) gives the operator live ETA / it/s.
        Set False for unit tests or callers that handle their own bar.
    progress_desc : optional override for the tqdm description label.
        Defaults to ``"select_subset r={r} {metric}"``.
    """
    if metric == "l2":
        m_batch = _metric_l2_batch
    elif metric == "jsd":
        m_batch = _metric_jsd_batch
    else:
        raise ValueError(f"unknown metric: {metric!r}")
    # Bind per-descriptor weights (default equal => behavior-preserving).
    # Validate up front so a bad key fails loudly before the long enumeration.
    _weights = _resolve_descriptor_weights(descriptor_weights)
    import functools as _functools
    m_batch = _functools.partial(m_batch, weights=_weights)

    npool = len(pool)
    if r > npool:
        raise ValueError(f"r={r} exceeds pool size {npool}")
    if r < len(fixed_indices):
        raise ValueError(f"r={r} smaller than fixed_indices count {len(fixed_indices)}")

    fixed_set = set(fixed_indices)
    free_indices = [i for i in range(npool) if i not in fixed_set]
    free_r = r - len(fixed_indices)

    from math import comb as _comb
    n_combos = _comb(len(free_indices), free_r)

    # Cache-read-back: when return_all=True and distribution_path is set
    # AND the file already exists, trust it and return without re-running
    # the C(npool, r) enumeration (which can be ~40M combinations for
    # r=14 over npool=28). The caller invalidates by deleting the file
    # (same pattern as the SCF/CCSD caches in external_refs.py).
    # We sanity-check the cache's shape against the requested (r, n_combos)
    # so a mismatch (e.g., pool size or r changed) raises loudly rather
    # than returning stale results silently.
    if return_all and distribution_path is not None:
        from pathlib import Path as _Path
        _dp = _Path(distribution_path)
        if _dp.is_file():
            with np.load(_dp) as z:
                vals = np.asarray(z["vals"])
                idx_array = np.asarray(z["indices"])
                best_combo_arr = np.asarray(z["best_combo"])
                best_val_arr = np.asarray(z["best_val"])
            if vals.shape != (n_combos,) or idx_array.shape != (n_combos, r):
                raise ValueError(
                    f"select_subset: cached distribution at {_dp} has shape "
                    f"vals={vals.shape}, indices={idx_array.shape} but the "
                    f"requested r={r} with this pool gives n_combos={n_combos} "
                    f"and per-row width={r}. Pool size, r, or fixed_indices "
                    f"likely changed, delete the cache file to force a re-run."
                )
            best_combo = tuple(int(i) for i in best_combo_arr)
            best_val = float(best_val_arr)
            return best_combo, best_val, vals, idx_array

    if return_all:
        vals = np.empty(n_combos, dtype=np.float64)
        idx_array = np.empty((n_combos, r), dtype=np.int64)

    # Pre-bin pool entries ONCE so the inner combo loop only has to sum
    # weight-counts across r entries (cheap) rather than re-binning the
    # concatenation of r descriptor arrays per combo (expensive).  At
    # r=12 over npool=26 this drops the per-combo cost from ~µs of bin
    # work + Python overhead to a vectorized fancy-index + sum, taking
    # the C(26,12)=9.66M-combo enumeration from ~hours to ~seconds.
    per_key_counts, W_in_range = _prebin_pool(pool, edges)
    bin_widths = {k: np.diff(edges[k]) for k in _DESCRIPTOR_KEYS}

    fixed_arr = np.asarray(sorted(fixed_set), dtype=np.int64)
    has_fixed = fixed_arr.size > 0

    # Batch size: trades Python overhead vs peak memory.  Each batch
    # allocates ~(BATCH * NBINS * 8 * 3 keys) bytes of float64 = ~150 MB
    # at BATCH=32768 with NBINS=200.  Tunable via STEP7_SUBSET_BATCH env.
    _batch_size = int(os.environ.get("STEP7_SUBSET_BATCH", "32768"))
    _batch_size = max(1, min(_batch_size, max(1, n_combos)))

    if progress:
        from tqdm.auto import tqdm
        pbar = tqdm(
            total=n_combos,
            desc=progress_desc or f"select_subset r={r} {metric}",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
            unit="combo",
        )
    else:
        pbar = None

    best_val = float("inf")
    best_combo: tuple = ()
    combo_iter = combinations(free_indices, free_r)
    base_idx = 0
    from itertools import islice as _islice
    while True:
        batch = list(_islice(combo_iter, _batch_size))
        if not batch:
            break
        # Build full-combo array of shape (b, r): prepend fixed indices,
        # then keep them sorted (downstream consumers expect sorted tuples).
        free_batch = np.asarray(batch, dtype=np.int64)            # (b, free_r)
        b = free_batch.shape[0]
        if has_fixed:
            full_batch = np.empty((b, r), dtype=np.int64)
            full_batch[:, : fixed_arr.size] = fixed_arr[None, :]
            full_batch[:, fixed_arr.size:] = free_batch
            full_batch.sort(axis=1)
        else:
            full_batch = free_batch                               # already sorted ascending
        # Build candidate density histograms via BLAS matmul.  A naive
        # fancy index ``per_key_counts[k][full_batch, :]`` materializes
        # a (b, r, NBINS) intermediate (≈630 MB at b=32k, r=12, NBINS=200)
        # whose sum-over-r is bandwidth-bound.  Replace it with a dense
        # one-hot indicator ``M[i, j] = 1 if j in combo_i`` (shape
        # (b, npool)) and a matrix multiply ``M @ per_key_counts[k]``
        # (shape (b, NBINS)).  np.matmul dispatches to BLAS gemm, hitting
        # 50–100× the throughput of the fancy-index path.
        M = np.zeros((b, npool), dtype=np.float64)
        np.put_along_axis(M, full_batch, 1.0, axis=1)
        h_cand_batch = {}
        empty_in_range = np.zeros(b, dtype=bool)
        for key in _DESCRIPTOR_KEYS:
            counts_combo = M @ per_key_counts[key]                # (b, NBINS)
            W_combo_k = M @ W_in_range[key]                       # (b,)
            # a combo whose grid points ALL fell outside the
            # histogram range for some descriptor has zero in-range weight
            # and is degenerate (empty candidate). Flag it for
            # disqualification rather than dividing by a fudged W=1.0,
            # which previously made an empty candidate score a
            # misleadingly-moderate ~0.5*ln2 JSD. We still compute a
            # finite (all-zero) histogram here to keep the metric
            # vectorized, then overwrite the score with +inf below.
            empty_in_range |= (W_combo_k <= 0.0)
            W_safe = np.where(W_combo_k > 0.0, W_combo_k, 1.0)
            h_cand_batch[key] = counts_combo / (W_safe[:, None] * bin_widths[key][None, :])
        vals_batch = m_batch(h_ref, h_cand_batch)                 # (b,)
        # Disqualify empty-in-range candidates: maximally divergent so the
        # argmin never picks them (SUBSET-05). Applies to both metrics.
        if empty_in_range.any():
            vals_batch = np.where(empty_in_range, np.inf, vals_batch)
        if return_all:
            vals[base_idx: base_idx + b] = vals_batch
            idx_array[base_idx: base_idx + b, :] = full_batch
        # Best update: pick the argmin within the batch and compare.
        local_best = int(np.argmin(vals_batch))
        if vals_batch[local_best] < best_val:
            best_val = float(vals_batch[local_best])
            best_combo = tuple(int(x) for x in full_batch[local_best])
        base_idx += b
        if pbar is not None:
            pbar.update(b)
    if pbar is not None:
        pbar.close()

    if return_all and distribution_path is not None:
        # Atomic write: tempfile + os.replace so an interrupted enumeration
        # cannot leave a half-written cache. Long enumerations (~40M
        # combos for r=14, npool=28) are precisely the case where Ctrl-C
        # is likely; a partial write would be silently mis-loaded by the
        # cache-read-back path on a subsequent invocation.
        import os as _os
        import tempfile as _tf
        from pathlib import Path as _Path
        _dp = _Path(distribution_path)
        out_dir = str(_dp.parent if _dp.parent != _Path("") else _Path("."))
        fd, tmp_name = _tf.mkstemp(dir=out_dir, suffix=".npz")
        try:
            _os.close(fd)
            np.savez_compressed(
                tmp_name,
                vals=vals, indices=idx_array,
                best_combo=np.array(best_combo),
                best_val=np.array(best_val),
            )
            _os.replace(tmp_name, str(_dp))
        finally:
            if _os.path.exists(tmp_name):
                _os.unlink(tmp_name)

    if return_all:
        return best_combo, best_val, vals, idx_array

    return best_combo, best_val


def compute_atom_set(ae_subset) -> set:
    """Return the union of chemical-element symbols across the given AE
    molecules. Used to determine which atomic-energy references must
    appear in `subset.traj` for total-energy regularization (§5c)."""
    out: set = set()
    for a in ae_subset:
        out.update(a.get_chemical_symbols())
    return out


def _make_hb_atoms() -> Atoms:
    """H-bonded water-dimer reference. Geometry from data_binning2.ipynb cell 20.
    Basis/grid-level overrides applied here: def2-svp / grid_level=1, NOT the
    legacy 6-311++G(3df,2pd) / level=4."""
    a = Atoms(_HB_SYMBOLS, positions=list(_HB_POSITIONS))
    a.info.update({
        # dfs_hill must be unique per HBPT pair so the step-7 notebook
        # builder's _atoms_to_mol_spec produces distinct MoleculeSpec.name
        # values; both pairs share Hill formula H4O2 so falling through to
        # ASE's get_chemical_formula() would collide.
        "charge": 1, "spin": 1, "dfs_hill": "HBWD", "name": "HBWD",
        "openshell": True,
        "sc": False, "sym": False, "reaction": "reactant",
        "grid_level": 1, "basis": "def2-svp", "pol": True,
    })
    return a


def _make_pt_atoms() -> Atoms:
    """Proton-transfer water-dimer reference. Geometry from data_binning2.ipynb cell 20.
    Basis/grid override: def2-svp / grid_level=1."""
    a = Atoms(_PT_SYMBOLS, positions=list(_PT_POSITIONS))
    a.info.update({
        # See _make_hb_atoms re: dfs_hill disambiguation.
        "charge": 1, "spin": 1, "dfs_hill": "PTWD", "name": "PTWD",
        "openshell": True,
        "sc": False, "sym": False, "reaction": 1,
        "grid_level": 1, "basis": "def2-svp", "pol": True,
    })
    return a


def augment_with_hbpt(
    ae_subset,
    atom_refs,
    *,
    with_hbpt: bool,
):
    """Build the final list of Atoms objects to write to subset.traj.

    Composition: AE-subset entries + atomic-reference entries + (optionally)
    the HB and PT water-dimer pair. Atom-refs are added unconditionally and
    are determined by `compute_atom_set` upstream. HBPT geometries are
    overridden to def2-svp / grid_level=1 so descriptor histograms remain
    commensurate with the rest of the candidate pool."""
    out = list(ae_subset) + list(atom_refs)
    if with_hbpt:
        out.append(_make_hb_atoms())
        out.append(_make_pt_atoms())
    return out


def _ase_atoms_to_pyscf_mol(at: Atoms, *, basis: str, charge: int = 0, spin: int = 0):
    """Build a PySCF gto.M from an ASE Atoms object. Mirrors the inline
    builder in xcquinox/alec/data.py:precompute_fixed_density_data line 237."""
    from pyscf import gto

    coords = at.get_positions()
    atom_lines = [
        (sym, tuple(coords[i])) for i, sym in enumerate(at.get_chemical_symbols())
    ]
    return gto.M(
        atom=atom_lines,
        basis=basis,
        charge=int(at.info.get("charge", charge)),
        spin=int(at.info.get("spin", spin)),
        unit="angstrom",
        verbose=0,
    )


def _descriptor_triple_from_mol(mol, *, grid_level: int) -> dict:
    """Run a PBE SCF on a built pyscf ``mol`` and return the
    ``(ρ^{1/3}, s, α, weights)`` descriptor dict.

    The SCF -> MGGA ``eval_rho`` -> :func:`compute_descriptor_triple` core shared
    by the ASE-Atoms (:func:`extract_descriptors`) and MoleculeSpec
    (:func:`extract_descriptors_for_molspecs`) extractors."""
    from pyscf import dft

    is_uhf = (mol.spin or 0) != 0
    mf = dft.UKS(mol, xc="PBE,PBE") if is_uhf else dft.RKS(mol, xc="PBE,PBE")
    mf.grids.level = grid_level
    mf.grids.build()
    mf.kernel()
    dm = mf.make_rdm1()
    ao = mf._numint.eval_ao(mol, mf.grids.coords, deriv=2)
    dm_total = (dm[0] + dm[1]) if is_uhf else dm
    rho_full = mf._numint.eval_rho(mol, ao, dm_total, xctype="MGGA")
    # rho_full shape (6, ngrid): [rho, rho_x, rho_y, rho_z, lapl, tau]
    rho = rho_full[0]
    sigma = rho_full[1] ** 2 + rho_full[2] ** 2 + rho_full[3] ** 2
    tau = rho_full[5]
    descriptors = compute_descriptor_triple(rho, sigma, tau)
    return {
        "rho_third": descriptors["rho_third"],
        "s": descriptors["s"],
        "alpha": descriptors["alpha"],
        "weights": mf.grids.weights,
    }


def _write_descriptor_cache(cache_path, out: dict) -> None:
    """Atomic ``.npz`` write (tempfile + ``os.replace``) so an interrupted
    extraction cannot leave a half-written cache that future runs mis-load."""
    import os as _os
    import tempfile as _tf

    cache_path = Path(cache_path)
    fd, tmp_name = _tf.mkstemp(dir=str(cache_path.parent), suffix=".npz")
    try:
        _os.close(fd)
        np.savez(tmp_name, **out)
        _os.replace(tmp_name, cache_path)
    finally:
        if _os.path.exists(tmp_name):
            _os.unlink(tmp_name)


def _pyscf_thread_init():
    """Pin one BLAS/OpenMP thread per worker (set before pyscf does work) so N
    extraction workers saturate N cores without thread oversubscription."""
    import os as _os

    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"] = "1"
    _os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        from pyscf import lib as _lib
        _lib.num_threads(1)
    except Exception:
        pass


def _extract_one_molspec_worker(args):
    """Build one MoleculeSpec's pyscf mol from plain args, extract the descriptor
    triple, write its atomic cache, and return the dict. Module-level + plain-arg
    (no MoleculeSpec object) so it is picklable for a ``spawn`` pool."""
    atom, charge, spin, basis, grid_level, cache_path = args
    from pyscf import gto

    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
    d = _descriptor_triple_from_mol(mol, grid_level=grid_level)
    _write_descriptor_cache(cache_path, d)
    return d


def extract_descriptors(
    at: Atoms,
    *,
    idx: int,
    cache_dir,
    basis: str = "def2-svp",
    grid_level: int = 1,
) -> dict:
    """Run a single PBE SCF and extract (ρ^{1/3}, s, α) on the molecular grid.

    Returns dict with keys rho_third, s, alpha, weights. Caches as
    <cache_dir>/<idx>_<species>.npz; second call hits the cache.

    Conventions match step-5/step-6 (def2-svp, grid_level=1) per
    _build_step6_notebook.py:528-532.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    species = at.info.get("species", at.get_chemical_formula())
    safe = species.replace("/", "_").replace(" ", "_")
    cache_path = cache_dir / f"{idx}_{safe}.npz"
    if cache_path.exists():
        z = np.load(cache_path)
        return {k: z[k] for k in ("rho_third", "s", "alpha", "weights")}

    mol = _ase_atoms_to_pyscf_mol(at, basis=basis)
    out = _descriptor_triple_from_mol(mol, grid_level=grid_level)
    _write_descriptor_cache(cache_path, out)
    return out


def extract_descriptors_for_molspecs(
    specs,
    *,
    basis: str = "def2-svp",
    grid_level: int = 1,
    cache_dir,
    n_jobs: int = 1,
) -> dict[tuple, dict]:
    """Per-:class:`MoleculeSpec` ``(ρ^{1/3}, s, α, weights)`` descriptors, cached
    by ``(name, charge, spin)``.

    The :func:`extract_descriptors_for_species` analogue for
    ``full_benchmark_pools`` held-out species: each ``MoleculeSpec`` carries a
    pyscf-format ``.atom`` string plus ``.charge``/``.spin``, so the pyscf mol is
    built directly (no ASE round-trip). ``basis``/``grid_level`` are taken from
    the arguments (def2-svp / 1 to match the pool-descriptor convention), not the
    spec's own fields, so every reaction's descriptors share one footing.

    ``n_jobs`` (>1) runs the per-species PBE SCFs in a ``spawn`` multiprocessing
    pool (1 BLAS thread per worker, so N workers saturate N cores). The SCFs are
    independent and each writes its own atomic ``.npz`` cache, so cached species
    are skipped and a partial run resumes. ``n_jobs=1`` (default) is the
    sequential, byte-identical path."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[tuple, dict] = {}
    seen: set[tuple] = set()
    todo: list[tuple] = []          # (key, work-args)
    for spec in specs:
        name = spec.name
        charge = int(getattr(spec, "charge", 0) or 0)
        spin = int(getattr(spec, "spin", 0) or 0)
        key = (name, charge, spin)
        if key in seen:
            continue
        seen.add(key)
        safe = name.replace("+", "plus").replace("/", "_").replace(" ", "_")
        cache_path = cache_dir / f"{safe}_c{charge}_s{spin}.npz"
        if cache_path.exists():
            z = np.load(cache_path)
            out[key] = {k: z[k] for k in ("rho_third", "s", "alpha", "weights")}
            continue
        todo.append((key, (spec.atom, charge, spin, basis, grid_level,
                           str(cache_path))))
    if not todo:
        return out

    work = [w for _key, w in todo]
    if int(n_jobs) <= 1 or len(todo) == 1:
        results = [_extract_one_molspec_worker(w) for w in work]
    else:
        import multiprocessing as _mp
        ctx = _mp.get_context("spawn")
        with ctx.Pool(processes=min(int(n_jobs), len(todo)),
                      initializer=_pyscf_thread_init) as pool:
            results = pool.map(_extract_one_molspec_worker, work)
    for (key, _w), d in zip(todo, results):
        out[key] = d
    return out


def concatenate_reaction_descriptors(reactions, species_descriptors, specs_by_name):
    """For each reaction, concatenate descriptors across its participating
    species (``reactants + products``), the reaction-pool analogue of
    :func:`concatenate_point_descriptors`.

    ``reactions`` are ``full_benchmark_pools`` reaction dicts (species-name
    lists); ``specs_by_name`` maps name -> ``MoleculeSpec`` (to resolve each
    species' ``(name,charge,spin)`` key into ``species_descriptors``). Returns a
    list parallel to ``reactions``, each a descriptor dict ready for
    :func:`build_reference_histograms` / ``select_subset_parallel``."""
    out: list[dict] = []
    for rxn in reactions:
        names = list(rxn.get("reactants", [])) + list(rxn.get("products", []))
        cat = {k: [] for k in _DESCRIPTOR_KEYS}
        cat_w: list = []
        for nm in names:
            spec = specs_by_name[nm]
            key = (spec.name, int(getattr(spec, "charge", 0) or 0),
                   int(getattr(spec, "spin", 0) or 0))
            d = species_descriptors[key]
            for k in _DESCRIPTOR_KEYS:
                cat[k].append(d[k])
            cat_w.append(d.get("weights", np.ones_like(d["rho_third"])))
        out.append({
            "rho_third": np.concatenate(cat["rho_third"]),
            "s":         np.concatenate(cat["s"]),
            "alpha":     np.concatenate(cat["alpha"]),
            "weights":   np.concatenate(cat_w),
        })
    return out
