"""Tests for xcquinox.alec.pbe_anchor — PBE-anchor regularization."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xcquinox.alec.pbe_anchor import (
    PBEAnchorSample,
    build_pbe_anchor_sample,
    pbe_anchor_loss,
)


def test_build_pbe_anchor_sample_is_deterministic():
    s1 = build_pbe_anchor_sample(n_points=50, seed=123)
    s2 = build_pbe_anchor_sample(n_points=50, seed=123)
    assert jnp.array_equal(s1.rho_alpha, s2.rho_alpha)
    assert jnp.array_equal(s1.rho_beta, s2.rho_beta)
    assert jnp.array_equal(s1.s, s2.s)
    assert jnp.array_equal(s1.Fx_target, s2.Fx_target)


def test_build_pbe_anchor_sample_shapes_and_ranges():
    N = 200
    sample = build_pbe_anchor_sample(
        n_points=N,
        log_rho_range=(-6.0, -1.0),
        s_range=(0.5, 15.0),
        zeta_range=(0.0, 1.0),
        seed=42,
    )
    assert sample.rho_alpha.shape == (N,)
    assert sample.rho_beta.shape == (N,)
    assert sample.s.shape == (N,)
    assert sample.Fx_target.shape == (N,)
    assert jnp.all(sample.rho_alpha >= 0.0)
    assert jnp.all(sample.rho_beta >= 0.0)
    assert jnp.all(sample.s >= 0.5) and jnp.all(sample.s <= 15.0)
    log_rho = jnp.log10(sample.rho_alpha + sample.rho_beta)
    assert jnp.all(log_rho >= -6.01) and jnp.all(log_rho <= -0.99)


def test_pbe_anchor_lda_limit_at_zero_gradient():
    sample = build_pbe_anchor_sample(
        n_points=4,
        log_rho_range=(0.0, 0.0),
        s_range=(0.0, 0.0),
        zeta_range=(0.0, 0.0),
        seed=0,
    )
    assert jnp.allclose(sample.Fx_target, 1.0, atol=1e-6)


def test_pbe_anchor_spin_flip_symmetry():
    from xcquinox.alec.pbe_anchor import _pbe_fx_libxc
    rho_a = jnp.array([0.1, 0.2, 0.5])
    rho_b_zero = jnp.zeros_like(rho_a)
    s = jnp.array([1.0, 1.0, 1.0])
    fx_up = _pbe_fx_libxc(rho_a, rho_b_zero, s)
    fx_dn = _pbe_fx_libxc(rho_b_zero, rho_a, s)
    assert jnp.allclose(fx_up, fx_dn, atol=1e-10)


def test_pbe_anchor_loss_zero_at_target_match():
    sample = build_pbe_anchor_sample(n_points=50, seed=7)
    def trivial_nn(params, ra, rb, s):
        return sample.Fx_target
    loss = pbe_anchor_loss({}, sample, weight=1e-3, nn_fx_fn=trivial_nn)
    assert jnp.allclose(loss, 0.0, atol=1e-12)


def test_pbe_anchor_loss_zero_when_weight_is_zero():
    sample = build_pbe_anchor_sample(n_points=50, seed=7)
    def wild_nn(params, ra, rb, s):
        return jnp.full_like(sample.Fx_target, 999.0)
    loss = pbe_anchor_loss({}, sample, weight=0.0, nn_fx_fn=wild_nn)
    assert jnp.allclose(loss, 0.0, atol=1e-12)


def test_pbe_anchor_loss_gradient_finite():
    sample = build_pbe_anchor_sample(n_points=30, seed=13)
    params = {"scale": jnp.array(1.2)}
    def linear_nn(p, ra, rb, s):
        return p["scale"] * sample.Fx_target
    grad_fn = jax.grad(pbe_anchor_loss, argnums=0)
    g = grad_fn(params, sample, 1e-3, linear_nn)
    assert jnp.all(jnp.isfinite(g["scale"]))
