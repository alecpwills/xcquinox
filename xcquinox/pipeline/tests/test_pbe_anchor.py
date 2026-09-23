"""Tests for xcquinox.pipeline.pbe_anchor: PBE-anchor regularization."""
import jax
import jax.numpy as jnp
import pytest

from xcquinox.pipeline.pbe_anchor import (
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
    from xcquinox.pipeline.pbe_anchor import _pbe_fx_libxc
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


def test_pbe_anchor_loss_gradient_finite():
    sample = build_pbe_anchor_sample(n_points=30, seed=13)
    params = {"scale": jnp.array(1.2)}
    def linear_nn(p, ra, rb, s):
        return p["scale"] * sample.Fx_target
    grad_fn = jax.grad(pbe_anchor_loss, argnums=0)
    g = grad_fn(params, sample, 1e-3, linear_nn)
    assert jnp.all(jnp.isfinite(g["scale"]))


def test_pbe_anchor_rks_reduction_via_spin_scaling():
    """F_x_UKS(rho/2, rho/2, s) must equal F_x_RKS(rho, s) by spin-scaling."""
    from xcquinox.pipeline.pbe_anchor import _pbe_fx_libxc
    from pyscf import dft as _pyscf_dft
    import numpy as np
    import jax.numpy as jnp

    rho_total = np.array([0.01, 0.1, 0.5, 1.0])
    s_vals    = np.array([0.5, 1.0, 5.0, 10.0])

    # UKS path via _pbe_fx_libxc at rho/2, rho/2.
    fx_uks = _pbe_fx_libxc(
        jnp.asarray(rho_total / 2.0),
        jnp.asarray(rho_total / 2.0),
        jnp.asarray(s_vals),
    )

    # Reference RKS path: call libxc in spin-unpolarized mode.
    kF = (3.0 * np.pi ** 2) ** (1.0 / 3.0)
    grad_mag = 2.0 * s_vals * kF * rho_total ** (4.0 / 3.0)
    sigma = grad_mag ** 2
    rho_input_rks = np.zeros((4, rho_total.shape[0]), dtype=np.float64)
    rho_input_rks[0, :] = rho_total
    rho_input_rks[3, :] = np.sqrt(sigma)
    _compute = _pyscf_dft.libxc.eval_xc
    ex_per_e_rks, *_ = _compute("GGA_X_PBE", rho_input_rks, spin=0, deriv=0)
    c_lda = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)
    ex_lda_per_e = c_lda * rho_total ** (1.0 / 3.0)
    fx_rks = ex_per_e_rks / ex_lda_per_e

    assert jnp.allclose(fx_uks, jnp.asarray(fx_rks), atol=1e-10), \
        f"spin-scaling identity broken: UKS={fx_uks}, RKS={fx_rks}"


# ---------------------------------------------------------------------------
# E5/E6 fix: rho -> 0 falls back to analytic PBE F_x(s), NOT 1.0
# ---------------------------------------------------------------------------


def test_fx_pbe_analytic_matches_canonical_values():
    """Pin a few canonical values of the PBE F_x(s) closed form against
    Perdew-Burke-Ernzerhof 1996 §3 eq. (14):
        F_x(s) = 1 + kappa - kappa / (1 + mu * s^2 / kappa)
    with kappa = 0.804, mu = 0.21951.
    """
    import numpy as np
    from xcquinox.pipeline.pbe_anchor import _fx_pbe_analytic
    s = np.array([0.0, 1.0, 2.0, 5.0, 100.0])
    # By hand:
    kappa, mu = 0.804, 0.21951
    expected = 1.0 + kappa - kappa / (1.0 + mu * s ** 2 / kappa)
    out = _fx_pbe_analytic(s)
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-12)
    # F_x(0) = 1 (uniform-electron-gas limit)
    assert abs(_fx_pbe_analytic(np.array([0.0]))[0] - 1.0) < 1e-12
    # F_x(s -> infty) = 1 + kappa = 1.804 (Lieb-Oxford bound)
    assert abs(_fx_pbe_analytic(np.array([1e6]))[0] - 1.804) < 1e-6


# ---------------------------------------------------------------------------
# Physics pin: PBE F_x(s) at canonical s values (against PBE 1996 §3 eq. (14))
# ---------------------------------------------------------------------------

def test_pbe_anchor_libxc_matches_analytic_at_high_density_low_polarization():
    """Sanity-pin _pbe_fx_libxc against the analytic PBE formula at a
    nearly-closed-shell, high-density point. At zeta=0 + sigma_aa = sigma_bb,
    the spin-scaling F_x_SS(rho/2, rho/2, s) reduces to F_x_RKS(rho, s),
    which equals the analytic PBE formula at fixed s. PBE 1996 §III
    spin-scaling: F_x_UKS(zeta=0) = F_x_RKS.
    """
    import jax.numpy as jnp
    import numpy as np
    from xcquinox.pipeline.pbe_anchor import _pbe_fx_libxc, _fx_pbe_analytic
    s_vals = jnp.array([0.5, 1.0, 1.5, 2.0])
    rho_tot = jnp.array([0.3, 0.3, 0.3, 0.3])  # dense (no rho-floor effect)
    rho_a = 0.5 * rho_tot
    rho_b = 0.5 * rho_tot
    fx_libxc = _pbe_fx_libxc(rho_a, rho_b, s_vals)
    fx_analytic = _fx_pbe_analytic(np.asarray(s_vals))
    np.testing.assert_allclose(
        np.asarray(fx_libxc), fx_analytic, atol=1e-5,
        err_msg="At zeta=0 high-density, libxc UKS PBE F_x should match "
                "the analytic PBE formula F_x(s) = 1 + κ - κ/(1 + μs²/κ)",
    )


# ---------------------------------------------------------------------------
# The anchor and the per-channel feature blocks.
# ---------------------------------------------------------------------------

def _anchor_model(arch_name, seed=0):
    import dataclasses
    import xcquinox.pipeline as pipeline
    arch = dataclasses.replace(pipeline.get_architecture(arch_name),
                               zero_init_final_layer=False)
    xnet, cnet = pipeline.create_network_pair(arch, seed=seed)
    return pipeline.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def test_anchor_term_refuses_a_descriptor_architecture_at_non_zero_weight():
    """A synthetic (rho_alpha, rho_beta, s) point has no density matrix, so the
    per-channel block of diag(P_sigma, P_sigma) is undefined there and the
    zero-extras row is a fixed slice of the feature space, not the block the
    network is evaluated on for any system."""
    from xcquinox.pipeline.losses import _anchor_term
    from xcquinox.pipeline.pbe_anchor import build_pbe_anchor_sample
    sample = build_pbe_anchor_sample(n_points=8, seed=3)
    model = _anchor_model("deep_rung35_mgga_3x16")
    with pytest.raises(ValueError, match="pbe_anchor_weight"):
        _anchor_term(model, sample, 1e-3)


# ---------------------------------------------------------------------------
# The anchor's footing: PBE exchange through the anchor path is round-off.
# ---------------------------------------------------------------------------


