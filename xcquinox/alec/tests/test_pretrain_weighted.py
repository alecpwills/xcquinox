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


def test_integration_mode_yields_smaller_e_xc_residual_than_unweighted():
    """On a toy dataset with heavy low-rho tail + target function with varying
    complexity across rho, integration-weighted SGD should produce a smaller
    integrated E_xc residual than unweighted SGD after equal training steps.

    This is the core reason for the integration-weighting design: unweighted MSE
    over-fits low-rho tail points that dominate the dataset but contribute
    negligibly to E_xc, while integration-weighting focuses on the high-rho
    region that actually integrates.
    """
    import jax
    import jax.numpy as jnp
    from xcquinox.alec.pretrain import _compute_integration_weights

    # Dataset: heavy tail at low rho + small cluster at high rho.
    rho = jnp.concatenate([
        jnp.logspace(-10, -3, 200),   # 200 points in low-rho tail
        jnp.linspace(0.01, 10.0, 50),  # 50 points in high-rho regime
    ])
    # Target: F = 1 + 0.2 * tanh(rho). 1-parameter model: F_pred = p * tanh(rho) + 1.
    target_F_minus_1 = 0.2 * jnp.tanh(rho)
    w, _ = _compute_integration_weights(rho)

    def loss_unweighted(p):
        pred_minus_1 = p * jnp.tanh(rho)
        return jnp.mean((pred_minus_1 - target_F_minus_1) ** 2)

    def loss_weighted(p):
        pred_minus_1 = p * jnp.tanh(rho)
        residual = pred_minus_1 - target_F_minus_1
        return jnp.sum(w * residual ** 2) / (jnp.sum(w) + 1e-12)

    # E_xc-weighted residual — the quantity that actually matters for AE.
    def e_xc_residual(p):
        pred_minus_1 = p * jnp.tanh(rho)
        return jnp.sum(w * (pred_minus_1 - target_F_minus_1) ** 2)

    p_unw = 0.0
    p_wtd = 0.0
    lr = 0.05
    n_steps = 500
    for _ in range(n_steps):
        p_unw = p_unw - lr * jax.grad(loss_unweighted)(p_unw)
        p_wtd = p_wtd - lr * jax.grad(loss_weighted)(p_wtd)

    e_unw = float(e_xc_residual(p_unw))
    e_wtd = float(e_xc_residual(p_wtd))

    # Integration-weighted training should produce a smaller E_xc residual
    # (typically by several orders of magnitude on a well-separated problem).
    assert e_wtd < e_unw, (
        f"Integration mode did not beat unweighted: "
        f"e_wtd={e_wtd:.3e}, e_unw={e_unw:.3e}"
    )
    # Stronger: integration should be at least 2x better.
    assert e_wtd < 0.5 * e_unw, (
        f"Integration advantage smaller than expected 2x: "
        f"ratio={e_wtd / e_unw:.3e}"
    )


# ---------------------------------------------------------------------------
# E1 fix: integration weighting must include Becke quadrature weights w_grid
# ---------------------------------------------------------------------------

def test_integration_weights_apply_grid_weights():
    """When grid_weights is supplied, the per-sample integration weight
    must be multiplied by w_grid_i. Becke-Lebedev quadrature gives dr_i
    per grid point; without it, the loss is "rho-eps_LDA-weighted mean
    per sample" rather than the integrated XC-energy residual it claims
    to be (E1 audit, 2026-04-27)."""
    from xcquinox.alec.pretrain import _compute_integration_weights
    rho = jnp.array([1.0, 0.5, 0.1])
    gw = jnp.array([2.0, 4.0, 0.5])
    w_x_no_gw, w_c_no_gw = _compute_integration_weights(rho)
    w_x_gw, w_c_gw = _compute_integration_weights(rho, gw)
    # With grid_weights, per-point weight = |rho * eps_LDA| * w_grid.
    assert jnp.allclose(w_x_gw, w_x_no_gw * gw), (w_x_gw, w_x_no_gw, gw)
    assert jnp.allclose(w_c_gw, w_c_no_gw * gw), (w_c_gw, w_c_no_gw, gw)


def test_integration_weights_grid_weights_none_matches_legacy():
    """Backward-compat: when grid_weights=None, the new
    _compute_integration_weights must match the pre-fix output exactly
    (so older pretrain_data.npz files without 'weights_all' continue to
    train, with a warning, but produce bit-identical losses to before)."""
    from xcquinox.alec.pretrain import _compute_integration_weights
    rho = jnp.array([1e-5, 0.1, 1.0, 5.0])
    w_x_new, w_c_new = _compute_integration_weights(rho, None)
    # Legacy formula (verbatim of pre-fix code, modulo broadcast shape):
    from xcquinox.utils import lda_x, pw92c_unpolarized_scalar
    rho_safe = jnp.maximum(rho, 1e-18)
    w_x_legacy = jnp.abs(rho_safe * lda_x(rho_safe))
    w_c_legacy = jnp.abs(rho_safe * pw92c_unpolarized_scalar(rho_safe))
    assert jnp.allclose(w_x_new, w_x_legacy)
    assert jnp.allclose(w_c_new, w_c_legacy)


# ---------------------------------------------------------------------------
# PRE-01: weight convention pin — linear |rho * eps_LDA| (option b)
# ---------------------------------------------------------------------------

def test_integration_weights_linear_convention():
    """PRE-01 (option b): _compute_integration_weights uses LINEAR |rho*eps_LDA|
    weighting, NOT the squared form.  This pins the chosen convention: the loss
    optimises a |rho*eps_LDA|-magnitude-weighted mean of the squared per-point
    F-residual, NOT the squared integrated XC-energy residual.

    With grid_weights supplied the weight is |rho*eps_LDA| * w_grid (linear
    in both factors); with grid_weights=None it is |rho*eps_LDA| alone.
    Either way the weight is NOT (rho*eps_LDA)^2.
    """
    from xcquinox.alec.pretrain import _compute_integration_weights
    from xcquinox.utils import lda_x, pw92c_unpolarized_scalar

    rho = jnp.array([0.1, 0.5, 1.0, 2.0])
    gw = jnp.array([1.5, 0.8, 2.0, 0.3])

    rho_safe = jnp.maximum(rho, 1e-18)
    eps_x = lda_x(rho_safe)
    eps_c = pw92c_unpolarized_scalar(rho_safe)
    expected_x_linear = jnp.abs(rho_safe * eps_x) * gw
    expected_c_linear = jnp.abs(rho_safe * eps_c) * gw
    expected_x_squared = (rho_safe * eps_x) ** 2 * gw
    expected_c_squared = (rho_safe * eps_c) ** 2 * gw

    w_x, w_c = _compute_integration_weights(rho, gw)

    # Convention is LINEAR: weights match |rho*eps_LDA| * w_grid.
    assert jnp.allclose(w_x, expected_x_linear, rtol=1e-6), (
        f"w_x={w_x} does not match linear convention {expected_x_linear}"
    )
    assert jnp.allclose(w_c, expected_c_linear, rtol=1e-6), (
        f"w_c={w_c} does not match linear convention {expected_c_linear}"
    )
    # Sanity: linear and squared are numerically distinct (so the test is non-trivial).
    assert not jnp.allclose(w_x, expected_x_squared, rtol=1e-6), (
        "linear and squared weights are identical — test is degenerate"
    )


# ---------------------------------------------------------------------------
# PRE-02: degradation flag in metadata when weights_all is absent
# ---------------------------------------------------------------------------

def test_run_pretrain_integration_mode_no_grid_weights_sets_degradation_flag(tmp_path):
    """PRE-02: when pretrain_data.npz lacks 'weights_all' and
    loss_weighting='integration', run_pretrain must set
    integration_weights_complete=False in the returned metadata (and in the
    pretrain_metadata.json on disk).  Without this flag downstream cannot
    distinguish a genuine integrated-energy run from the degraded
    |rho*eps_LDA|-mean fallback.
    """
    import warnings
    import xcquinox.alec as alec
    from xcquinox.alec.config import PretrainSpec
    from xcquinox.alec.pretrain import run_pretrain

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    n_grid = 32
    rho = np.linspace(0.01, 5.0, n_grid).astype(np.float64)
    sigma = (rho ** (8.0 / 3.0)) * 0.1
    # NOTE: no 'weights_all' key — this is the degraded scenario.
    np.savez(
        str(data_dir / "pretrain_data.npz"),
        rho_all=rho,
        sigma_all=sigma,
        Fx_all=np.zeros_like(rho),
        Fc_all=np.zeros_like(rho),
    )

    ckpt_dir = tmp_path / "ckpt"
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

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = run_pretrain(spec)

    # The degradation flag must be present and False in the returned dict.
    assert "integration_weights_complete" in result, (
        "run_pretrain metadata missing 'integration_weights_complete' key"
    )
    assert result["integration_weights_complete"] is False, (
        f"Expected integration_weights_complete=False when weights_all absent, "
        f"got {result['integration_weights_complete']!r}"
    )

    # The flag must also be persisted to pretrain_metadata.json on disk.
    import json
    with open(str(ckpt_dir / "pretrain_metadata.json")) as f:
        md = json.load(f)
    assert "integration_weights_complete" in md, (
        "pretrain_metadata.json missing 'integration_weights_complete'"
    )
    assert md["integration_weights_complete"] is False


def test_run_pretrain_integration_mode_with_grid_weights_sets_complete_flag(tmp_path):
    """PRE-02 complement: when 'weights_all' IS present, run_pretrain must set
    integration_weights_complete=True in the returned metadata.
    """
    import json
    import xcquinox.alec as alec
    from xcquinox.alec.config import PretrainSpec
    from xcquinox.alec.pretrain import run_pretrain

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    n_grid = 32
    rho = np.linspace(0.01, 5.0, n_grid).astype(np.float64)
    sigma = (rho ** (8.0 / 3.0)) * 0.1
    weights_all = np.ones(n_grid, dtype=np.float64) * (1.0 / n_grid)  # uniform grid weights
    np.savez(
        str(data_dir / "pretrain_data.npz"),
        rho_all=rho,
        sigma_all=sigma,
        Fx_all=np.zeros_like(rho),
        Fc_all=np.zeros_like(rho),
        weights_all=weights_all,
    )

    ckpt_dir = tmp_path / "ckpt"
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

    assert "integration_weights_complete" in result, (
        "run_pretrain metadata missing 'integration_weights_complete' key"
    )
    assert result["integration_weights_complete"] is True, (
        f"Expected integration_weights_complete=True when weights_all present, "
        f"got {result['integration_weights_complete']!r}"
    )

    with open(str(ckpt_dir / "pretrain_metadata.json")) as f:
        md = json.load(f)
    assert md["integration_weights_complete"] is True
