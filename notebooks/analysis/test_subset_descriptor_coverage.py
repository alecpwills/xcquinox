"""Tests for ``subset_descriptor_coverage`` -- pure descriptor/coverage math +
render canaries over the full descriptor set. No pyscf / density precompute."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "subset_descriptor_coverage", _HERE / "subset_descriptor_coverage.py")
sdc = importlib.util.module_from_spec(_spec)
sys.modules["subset_descriptor_coverage"] = sdc
_spec.loader.exec_module(sdc)


# ---------------------------------------------------------------------------
# point_s_rs -- closed forms
# ---------------------------------------------------------------------------

def test_point_s_rs_closed_forms():
    rho = np.array([1.0, 2.0])
    s, rs = sdc.point_s_rs(rho, np.array([0.0, 0.0]))
    assert s[0] == pytest.approx(0.0)
    assert rs[0] == pytest.approx((3.0 / (4 * math.pi)) ** (1 / 3), rel=1e-9)
    assert rs[1] == pytest.approx((3.0 / (4 * math.pi * 2.0)) ** (1 / 3), rel=1e-9)


def test_point_s_rs_known_gradient():
    kf = (3 * math.pi ** 2) ** (1 / 3)
    sigma = (2 * kf) ** 2  # makes s = 1 at rho = 1
    s, _ = sdc.point_s_rs(np.array([1.0]), np.array([sigma]))
    assert s[0] == pytest.approx(1.0, rel=1e-9)


def test_point_s_rs_floors_low_density_to_nan():
    s, rs = sdc.point_s_rs(np.array([0.0]), np.array([0.0]))
    assert not np.isfinite(s[0]) and not np.isfinite(rs[0])


# ---------------------------------------------------------------------------
# weighted_hist
# ---------------------------------------------------------------------------

def test_weighted_hist_normalizes_to_unit_area():
    edges = np.linspace(0, 3, 4)
    h = sdc.weighted_hist(np.array([0.5, 1.5, 1.5, 2.5]), np.ones(4), edges)
    assert h.sum() == pytest.approx(1.0)
    assert h[1] == pytest.approx(0.5)


def test_weighted_hist_respects_weights_and_skips_nan():
    edges = np.array([0.0, 1.0, 2.0])
    h = sdc.weighted_hist(np.array([0.5, 1.5, np.nan]),
                          np.array([3.0, 1.0, 99.0]), edges)
    assert h[0] == pytest.approx(0.75)
    assert h[1] == pytest.approx(0.25)


def test_weighted_hist_empty_returns_zeros():
    edges = np.array([0.0, 1.0, 2.0])
    h = sdc.weighted_hist(np.array([np.nan]), np.array([1.0]), edges)
    assert h.shape == (2,) and h.sum() == 0.0


# ---------------------------------------------------------------------------
# adaptive_edges
# ---------------------------------------------------------------------------

def test_adaptive_edges_spans_robust_range():
    # rare (0.1%) extreme outlier must be clipped by the 99.5 percentile.
    vals = np.concatenate([np.linspace(0.2, 0.8, 1000), [1e6]])
    e = sdc.adaptive_edges(vals, n_bins=10)
    assert len(e) == 11
    assert e[0] >= 0.19 and e[-1] < 100


def test_adaptive_edges_degenerate_fallback():
    e = sdc.adaptive_edges(np.array([5.0, 5.0, 5.0]), n_bins=4)
    assert len(e) == 5 and e[-1] > e[0]


# ---------------------------------------------------------------------------
# histogram_intersection + completeness
# ---------------------------------------------------------------------------

def test_histogram_intersection_bounds():
    a = np.array([0.5, 0.5, 0.0])
    b = np.array([0.0, 0.5, 0.5])
    assert sdc.histogram_intersection(a, a) == pytest.approx(1.0)
    assert sdc.histogram_intersection(a, b) == pytest.approx(0.5)
    assert sdc.histogram_intersection(np.array([1.0, 0]),
                                      np.array([0, 1.0])) == 0.0


def test_completeness_averages_over_given_dims():
    train = {"s": np.array([1.0, 0.0]), "rs": np.array([0.0, 1.0]),
             "CuspDescriptor_0": np.array([1.0, 0.0])}
    held = {"s": np.array([1.0, 0.0]), "rs": np.array([0.5, 0.5]),
            "CuspDescriptor_0": np.array([0.0, 1.0])}
    # restrict to {s, rs}: (1.0 + 0.5)/2 = 0.75
    assert sdc.completeness(train, held, dims=["s", "rs"]) == pytest.approx(0.75)
    # include cusp (intersection 0.0): (1 + 0.5 + 0)/3
    assert sdc.completeness(
        train, held, dims=["s", "rs", "CuspDescriptor_0"]) == pytest.approx(0.5)


def test_hist_pool_from_values_pools_and_normalizes():
    edges = np.array([0.0, 1.0, 2.0])
    pooled = sdc.hist_pool_from_values(
        [np.array([0.5]), np.array([1.5])],
        [np.array([1.0]), np.array([1.0])], edges)
    assert pooled.sum() == pytest.approx(1.0)
    assert pooled[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _spearman
# ---------------------------------------------------------------------------

def test_spearman_monotone_and_anti():
    assert sdc._spearman(np.array([1, 2, 3, 4.]),
                         np.array([2, 4, 6, 8.])) == pytest.approx(1.0)
    assert sdc._spearman(np.array([1, 2, 3, 4.]),
                         np.array([4, 3, 2, 1.])) == pytest.approx(-1.0)


def test_spearman_handles_ties():
    assert sdc._spearman(np.array([1, 2, 3.]), np.array([5, 5, 5.])) == 0.0


# ---------------------------------------------------------------------------
# Render canaries
# ---------------------------------------------------------------------------

def _png_ok(p: Path) -> bool:
    return p.is_file() and p.stat().st_size > 2000


def test_plot_completeness_vs_mae_renders(tmp_path):
    rows = [
        {"idx": 0, "arch": "deep", "completeness": 0.4, "mae": 24.8, "pbe_mae": 14.5},
        {"idx": 2, "arch": "deep", "completeness": 0.7, "mae": 21.0, "pbe_mae": 14.5},
        {"idx": 40, "arch": "deep_combined", "completeness": 0.6, "mae": 18.0,
         "pbe_mae": 14.5},
    ]
    out = sdc.plot_completeness_vs_mae(rows, tmp_path / "cov.png", "run_x")
    assert _png_ok(out)


def test_plot_descriptor_histograms_renders_variable_dims(tmp_path):
    dims = ["s", "rs", "CuspDescriptor_0", "DMStatisticsDescriptor_1"]
    edges = {d: np.linspace(0, 1, 21) for d in dims}
    held = {d: np.ones(20) / 20 for d in dims}
    subset_hists = {
        1: {d: np.ones(20) / 20 for d in ("s", "rs")},  # deep: only s, rs
        40: {d: np.ones(20) / 20 for d in dims},        # combined: all dims
    }
    out = sdc.plot_descriptor_histograms(subset_hists, held, edges,
                                         tmp_path / "hist.png", "run_x")
    assert _png_ok(out)
