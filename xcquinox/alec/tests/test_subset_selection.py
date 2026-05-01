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
