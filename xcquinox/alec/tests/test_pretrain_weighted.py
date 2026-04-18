"""Tests for integration-weighted pretraining loss."""
import jax.numpy as jnp
import numpy as np
import pytest

from xcquinox.alec.pretrain import _compute_integration_weights


def test_integration_weights_shape_matches_rho():
    rho = jnp.array([0.1, 0.5, 1.0, 2.0])
    w_x, w_c = _compute_integration_weights(rho)
    assert w_x.shape == rho.shape
    assert w_c.shape == rho.shape


def test_integration_weights_nonnegative():
    rho = jnp.array([1e-8, 0.01, 0.5, 1.0])
    w_x, w_c = _compute_integration_weights(rho)
    assert jnp.all(w_x >= 0)
    assert jnp.all(w_c >= 0)


def test_integration_weights_high_rho_dominates():
    w_x, _ = _compute_integration_weights(jnp.array([0.01, 10.0]))
    assert float(w_x[1]) > 100 * float(w_x[0])


def test_integration_weights_zero_rho_gives_near_zero_weight():
    w_x, w_c = _compute_integration_weights(jnp.array([0.0]))
    assert float(w_x[0]) < 1e-6
    assert float(w_c[0]) < 1e-6
