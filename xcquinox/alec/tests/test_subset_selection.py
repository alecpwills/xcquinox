"""Unit tests for xcquinox.alec.subset_selection."""
from __future__ import annotations

import numpy as np
import pytest

from xcquinox.alec import subset_selection as ss


def test_compute_descriptor_triple_uniform_gas_returns_alpha_one():
    """For a uniform electron gas: ∇ρ = 0 → τ_W = 0; τ = τ_unif by construction → α = 1.

    Reference: Sun, Ruzsinszky, Perdew, PRL 115, 036402 (2015), eq. (5).
    """
    n_grid = 10**3
    rho = np.full(n_grid, 0.1)
    sigma = np.zeros(n_grid)  # |∇ρ|² = 0 (uniform)
    tau_unif = (3.0 / 10.0) * (3.0 * np.pi**2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
    tau = tau_unif.copy()
    desc = ss.compute_descriptor_triple(rho, sigma, tau)
    assert desc["rho_third"].shape == (n_grid,)
    assert desc["s"].shape == (n_grid,)
    assert desc["alpha"].shape == (n_grid,)
    assert np.allclose(desc["alpha"], 1.0, atol=1e-6), \
        f"α for uniform gas should be 1.0; got {desc['alpha'][:5]}"
    assert np.allclose(desc["s"], 0.0, atol=1e-12)


def test_compute_descriptor_triple_iso_orbital_returns_alpha_zero():
    """For a single-orbital iso-orbital region, τ = τ_W → α = 0."""
    n_grid = 100
    rho = np.linspace(0.05, 0.5, n_grid)
    sigma = np.linspace(0.001, 0.01, n_grid)  # |∇ρ|²
    tau_W = sigma / (8.0 * rho)
    tau = tau_W.copy()  # τ = τ_W → iso-orbital
    desc = ss.compute_descriptor_triple(rho, sigma, tau)
    assert np.allclose(desc["alpha"], 0.0, atol=1e-12), \
        f"α should be 0 in iso-orbital region; got max |α|={np.abs(desc['alpha']).max()}"


def test_compute_descriptor_triple_s_formula_matches_pbe1996():
    """s = |∇ρ| / [2 (3π²)^{1/3} ρ^{4/3}], PBE 1996 eq. block before eq. (12)."""
    rho = np.array([0.5, 1.0, 2.0])
    sigma = np.array([1.0, 4.0, 9.0])  # |∇ρ|² → |∇ρ| = sqrt(σ) = [1, 2, 3]
    tau = np.zeros_like(rho)  # don't care about α here
    desc = ss.compute_descriptor_triple(rho, sigma, tau)
    grad_rho = np.sqrt(sigma)
    expected_s = grad_rho / (2.0 * (3.0 * np.pi**2) ** (1.0 / 3.0) * rho ** (4.0 / 3.0))
    np.testing.assert_allclose(desc["s"], expected_s, rtol=1e-12)


def test_compute_descriptor_triple_no_negative_alpha_under_clip():
    """α can in principle be negative due to grid noise; values are clipped at 0
    (matches data_binning2.ipynb cell 17 implicit behavior of histogramming
    only positive values via log10)."""
    rho = np.full(10, 0.1)
    sigma = np.full(10, 1.0)  # |∇ρ|² = 1
    tau_W = sigma / (8.0 * rho)
    # τ < τ_W → α < 0; we expect the clip
    tau = 0.5 * tau_W
    desc = ss.compute_descriptor_triple(rho, sigma, tau)
    assert (desc["alpha"] >= 0.0).all(), "α must be clipped to non-negative"


def _mock_three_histograms(seed=0):
    rng = np.random.default_rng(seed)
    h1 = rng.uniform(size=ss.NBINS)
    h2 = rng.uniform(size=ss.NBINS)
    h3 = rng.uniform(size=ss.NBINS)
    return {
        "rho_third": h1 / h1.sum(),
        "s": h2 / h2.sum(),
        "alpha": h3 / h3.sum(),
    }


def test_metric_l2_self_zero():
    h = _mock_three_histograms(seed=42)
    assert ss.metric_l2(h, h) == pytest.approx(0.0, abs=1e-15)


def test_metric_jsd_self_zero():
    h = _mock_three_histograms(seed=42)
    assert ss.metric_jsd(h, h) == pytest.approx(0.0, abs=1e-12)


def test_metric_jsd_symmetric():
    p = _mock_three_histograms(seed=1)
    q = _mock_three_histograms(seed=2)
    assert ss.metric_jsd(p, q) == pytest.approx(ss.metric_jsd(q, p), abs=1e-15)


def test_metric_l2_three_histogram_sum():
    """The L2 error sums sqrt over all 3 histograms per bin (matching
    data_binning2.ipynb cell 17). Constructed input where only h1 differs
    in a single bin should yield exactly the displacement magnitude."""
    p = _mock_three_histograms(seed=3)
    q = {k: v.copy() for k, v in p.items()}
    q["rho_third"] = q["rho_third"].copy()
    q["rho_third"][0] += 0.5
    err = ss.metric_l2(p, q)
    # Only one bin differs; sqrt(0.25 + 0 + 0) = 0.5; sum over 200 bins of
    # sqrt of the per-bin sum-of-squared-diffs => 0.5 + sum(sqrt(0)) = 0.5
    assert err == pytest.approx(0.5, abs=1e-12)


def test_metric_jsd_three_histogram_sum():
    """JSD totals over the 3 marginals."""
    p = _mock_three_histograms(seed=4)
    q = _mock_three_histograms(seed=5)
    err = ss.metric_jsd(p, q)
    assert err > 0.0
    assert err <= 3.0 * np.log(2.0) + 1e-12


def test_metric_jsd_uses_natural_log():
    """JSD with natural log: max value per marginal is ln(2)."""
    p = {"rho_third": np.zeros(ss.NBINS), "s": np.zeros(ss.NBINS), "alpha": np.zeros(ss.NBINS)}
    q = {"rho_third": np.zeros(ss.NBINS), "s": np.zeros(ss.NBINS), "alpha": np.zeros(ss.NBINS)}
    p["rho_third"][0] = 1.0
    q["rho_third"][1] = 1.0
    p["s"][0] = 1.0
    q["s"][1] = 1.0
    p["alpha"][0] = 1.0
    q["alpha"][1] = 1.0
    err = ss.metric_jsd(p, q)
    assert err == pytest.approx(3.0 * np.log(2.0), rel=1e-6)


def _toy_descriptor_arrays(seed):
    rng = np.random.default_rng(seed)
    n = 5000
    return {
        "rho_third": np.abs(rng.normal(loc=0.5, scale=0.2, size=n)) + 1e-6,
        "s": np.abs(rng.normal(loc=1.0, scale=0.5, size=n)) + 1e-6,
        "alpha": np.abs(rng.normal(loc=1.0, scale=0.3, size=n)) + 1e-6,
        "weights": np.ones(n) / n,
    }


def test_bin_descriptors_returns_three_normalized_marginals():
    arrs = _toy_descriptor_arrays(seed=0)
    hist = ss.bin_descriptors(arrs)
    for k in ("rho_third", "s", "alpha"):
        assert hist[k].shape == (ss.NBINS,)
        assert hist[k].min() >= 0.0
        assert hist[k].sum() > 0.0


def test_build_reference_histograms_concats_pool():
    pool = [_toy_descriptor_arrays(seed=i) for i in range(3)]
    h_ref, edges = ss.build_reference_histograms(pool)
    assert set(h_ref.keys()) == {"rho_third", "s", "alpha"}
    assert set(edges.keys()) == {"rho_third", "s", "alpha"}
    for k in ("rho_third", "s", "alpha"):
        assert h_ref[k].shape == (ss.NBINS,)
        assert edges[k].shape == (ss.NBINS + 1,)


def _build_toy_pool(npool=8, seed=0):
    """Build a synthetic 8-entry pool with consistent log10 edges."""
    pool = [_toy_descriptor_arrays(seed=seed + i) for i in range(npool)]
    h_ref, edges = ss.build_reference_histograms(pool)
    return pool, h_ref, edges


def test_select_subset_recovers_pool_when_r_eq_n_l2():
    pool, h_ref, edges = _build_toy_pool(npool=5)
    chosen, val = ss.select_subset(pool, edges, h_ref, r=5, metric="l2")
    assert sorted(chosen) == [0, 1, 2, 3, 4]
    assert val == pytest.approx(0.0, abs=1e-12)


def test_select_subset_recovers_pool_when_r_eq_n_jsd():
    pool, h_ref, edges = _build_toy_pool(npool=5)
    chosen, val = ss.select_subset(pool, edges, h_ref, r=5, metric="jsd")
    assert sorted(chosen) == [0, 1, 2, 3, 4]
    assert val == pytest.approx(0.0, abs=1e-12)


def test_select_subset_exhaustive_for_small_r():
    pool, h_ref, edges = _build_toy_pool(npool=6)
    chosen, val = ss.select_subset(pool, edges, h_ref, r=2, metric="l2")
    assert len(chosen) == 2
    from itertools import combinations as _C
    best_val, best_pair = float("inf"), None
    for pair in _C(range(6), 2):
        cat = {k: np.concatenate([pool[i][k] for i in pair]) for k in ss._DESCRIPTOR_KEYS}
        cat["weights"] = np.concatenate([pool[i].get("weights", np.ones_like(pool[i]["rho_third"])) for i in pair])
        h_cand = ss._bin_with_edges(cat, edges)
        v = ss.metric_l2(h_ref, h_cand)
        if v < best_val:
            best_val, best_pair = v, pair
    assert sorted(chosen) == sorted(best_pair)
    assert val == pytest.approx(best_val, rel=1e-12)
