"""Step-7 loss-extension unit tests."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from xcquinox.alec import losses


def test_rxn_residual_basic_zero_residual():
    """E_NN_products − E_NN_reactants = E_ref → residual = 0."""
    e_nn = jnp.array([2.0, 1.0])  # [reactant, product]
    coeffs = jnp.array([-1.0, +1.0])  # reactant subtracted, product added
    e_rxn_ref = jnp.array(-1.0)  # 1.0 - 2.0 = -1.0
    res = losses._rxn_residual_term(e_nn, coeffs, e_rxn_ref)
    assert float(res) == pytest.approx(0.0, abs=1e-12)


def test_rxn_residual_off_by_one():
    e_nn = jnp.array([2.0, 1.0])
    coeffs = jnp.array([-1.0, +1.0])
    e_rxn_ref = jnp.array(0.0)  # but actual rxn energy = -1.0 → residual = 1.0
    res = losses._rxn_residual_term(e_nn, coeffs, e_rxn_ref)
    assert float(res) == pytest.approx(1.0, abs=1e-12)


def test_ip_residual_basic_zero_residual():
    """IP = E_cation - E_neutral; residual = (IP_NN - IP_ref)^2 → 0 when match."""
    e_cation = jnp.array(5.0)
    e_neutral = jnp.array(2.0)
    ip_ref = jnp.array(3.0)
    res = losses._ip_residual_term(e_cation, e_neutral, ip_ref)
    assert float(res) == pytest.approx(0.0, abs=1e-12)


def test_ip_residual_squared_displacement():
    e_cation = jnp.array(5.0)
    e_neutral = jnp.array(2.0)
    ip_ref = jnp.array(2.0)  # actual = 3.0; residual^2 = 1.0
    res = losses._ip_residual_term(e_cation, e_neutral, ip_ref)
    assert float(res) == pytest.approx(1.0, abs=1e-12)
