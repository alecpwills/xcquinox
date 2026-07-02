"""Tests for the DFS SCF-trajectory tail reducers in xcquinox.alec.oneshot.

DFS (Dick & Fernandez-Serra 2021) forces N SCF cycles but scores only the
convergence TAIL with a quadratic weight rising toward convergence
(``weights = linspace(0,1,N)**2``, ``dE = dE[skip:]`` with
``skip = max(5, N-10)``). These helpers generalize that to any N (full_3,
full_25, ...) -- the fix to DFS's ``max(5, N-10)`` which underflows to an
empty slice at small N.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from xcquinox.alec.oneshot import (
    scf_tail_window,
    tail_weighted_mean_energy,
    scf_loss_tail_weights,
)
from xcquinox.alec.solver import SolverConfig, SolverMode


# --- scf_tail_window: (skip, kept-tail weights) -----------------------------

def test_scf_tail_window_dfs_exact_at_25():
    # DFS default: 25 cycles, tail 10 -> skip 15, last 10 steps, w[-1]==1.
    skip, w = scf_tail_window(25, 10, 2.0)
    assert skip == 15
    assert len(w) == 10
    expected = (np.linspace(0.0, 1.0, 25) ** 2)[15:]
    assert np.allclose(w, expected)
    assert np.isclose(w[-1], 1.0)


def test_scf_tail_window_small_n_keeps_all():
    # full_3: 3 cycles < tail 10 -> skip 0, keep all 3 (NOT an empty slice,
    # which DFS's max(5, N-10)=5 -> [5:] would give).
    skip, w = scf_tail_window(3, 10, 2.0)
    assert skip == 0
    assert np.allclose(w, [0.0, 0.25, 1.0])


def test_scf_tail_window_n1_guard():
    # linspace(0,1,1)**2 == [0.0] would zero out the only step; guard -> [1.0].
    skip, w = scf_tail_window(1, 10, 2.0)
    assert skip == 0
    assert np.allclose(w, [1.0])


def test_scf_tail_window_n_equals_tail():
    skip, w = scf_tail_window(10, 10, 2.0)
    assert skip == 0
    assert len(w) == 10


def test_scf_tail_window_partial_tail():
    skip, w = scf_tail_window(12, 10, 2.0)
    assert skip == 2
    assert len(w) == 10


# --- tail_weighted_mean_energy: convergence-aware reported scalar ------------

def test_tail_weighted_mean_flat_trace_equals_value():
    # A converged SCF freezes -> flat tail -> weighted mean == that value.
    trace = jnp.full((25,), -76.5)
    assert jnp.isclose(tail_weighted_mean_energy(trace, 10, 2.0), -76.5)


def test_tail_weighted_mean_denoises_period2_oscillation():
    # A non-converged period-2 oscillation: the weighted mean lands BETWEEN the
    # two phases (denoised), unlike the arbitrary final step which is one phase.
    trace = jnp.asarray([-223.3 if i % 2 == 0 else -223.7 for i in range(25)])
    val = float(tail_weighted_mean_energy(trace, 10, 2.0))
    assert -223.7 < val < -223.3


# --- scf_loss_tail_weights: gated weight vector for the per-step loss --------

def test_scf_loss_tail_weights_none_when_disabled():
    cfg = SolverConfig(mode=SolverMode.FULL, max_cycles=25)  # use_tail default False
    assert scf_loss_tail_weights(cfg) is None


def test_scf_loss_tail_weights_none_for_oneshot():
    assert scf_loss_tail_weights(SolverConfig()) is None  # ONESHOT default


def test_scf_loss_tail_weights_present_when_enabled():
    cfg = SolverConfig(
        mode=SolverMode.FULL, max_cycles=25,
        scf_loss_use_tail=True, scf_loss_tail=10, scf_loss_weight_power=2.0,
    )
    w = scf_loss_tail_weights(cfg)
    assert w is not None
    assert len(w) == 10
    assert float(w[-1]) == 1.0
