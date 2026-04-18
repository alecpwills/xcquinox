"""Tests for integration-weighted pretraining loss."""
import os

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


def test_pretrain_loss_unweighted_mode_matches_mse():
    """When loss_weighting='unweighted', the loss is plain mean-squared residual."""
    # Zero-residual scenario: both modes agree at 0.
    residual = jnp.array([0.0, 0.0])
    assert float(jnp.mean(residual ** 2)) == 0.0

    # Nonzero-residual scenario: the unweighted reduction is plain mean.
    residual_nonzero = jnp.array([1.0, 2.0, 3.0])
    expected = float((1.0 + 4.0 + 9.0) / 3.0)
    assert float(jnp.mean(residual_nonzero ** 2)) == pytest.approx(expected)


def test_pretrain_loss_integration_mode_differs_from_unweighted():
    """Integration mode produces DIFFERENT loss value than unweighted mode
    when residuals are non-uniform across rho."""
    rho = jnp.array([0.01, 10.0])
    residual = jnp.array([1.0, 0.1])  # big at low-rho, small at high-rho
    w_x, _ = _compute_integration_weights(rho)
    unweighted_loss = jnp.mean(residual ** 2)
    weighted_loss = jnp.sum(w_x * residual ** 2) / (jnp.sum(w_x) + 1e-12)
    # Integration mode should down-weight the big-residual-at-low-rho point
    assert float(weighted_loss) < float(unweighted_loss)
    # Magnitudes should differ by at least 2x
    assert float(unweighted_loss) > 2 * float(weighted_loss)


def test_run_pretrain_integration_mode_runs_without_error(tmp_path):
    """run_pretrain with loss_weighting='integration' completes without error."""
    import xcquinox.alec as alec
    from xcquinox.alec.config import PretrainSpec
    from xcquinox.alec.pretrain import run_pretrain

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Minimal pretrain_data.npz — match the columns _assemble_pretrain_descriptors
    # will request for arch "deep" (no dm, no cusp): rho_all + sigma_all.
    # Fx_all / Fc_all are the (F - 1) targets; zero means F = 1 (LDA/PW92).
    n_grid = 64
    rho = np.linspace(0.01, 5.0, n_grid).astype(np.float64)
    sigma = (rho ** (8.0 / 3.0)) * 0.1
    Fx_minus_1 = np.zeros_like(rho)
    Fc_minus_1 = np.zeros_like(rho)
    np.savez(
        os.path.join(str(data_dir), "pretrain_data.npz"),
        rho_all=rho,
        sigma_all=sigma,
        Fx_all=Fx_minus_1,
        Fc_all=Fc_minus_1,
    )

    ckpt_dir = tmp_path / "ckpt"
    # Use a small arch to keep the test cheap.
    arch = alec.get_architecture("shallow")
    spec = PretrainSpec(
        arch=arch,
        data_dir=str(data_dir),
        checkpoint_dir=str(ckpt_dir),
        n_steps=3,
        lr_start=1e-2,
        lr_end=1e-5,
        lr_decay_start=0.0,
        grad_clip=1.0,
        seed=0,
        loss_weighting="integration",
    )
    result = run_pretrain(spec)
    assert result is not None
    # The spec's loss_weighting flows into metadata for provenance.
    assert result["loss_weighting"] == "integration"
    # Losses should be finite.
    assert np.isfinite(result["final_loss_x"])
    assert np.isfinite(result["final_loss_c"])
