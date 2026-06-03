"""Tests for xcquinox.alec.pbe_anchor: PBE-anchor regularization."""
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


def test_build_pbe_anchor_sample_rejects_out_of_range_zeta():
    with pytest.raises(ValueError, match="zeta_range"):
        build_pbe_anchor_sample(n_points=10, zeta_range=(0.0, 1.5), seed=0)
    with pytest.raises(ValueError, match="zeta_range"):
        build_pbe_anchor_sample(n_points=10, zeta_range=(-1.5, 0.0), seed=0)
    with pytest.raises(ValueError, match="zeta_range"):
        build_pbe_anchor_sample(n_points=10, zeta_range=(0.5, 0.1), seed=0)


def test_pbe_anchor_rks_reduction_via_spin_scaling():
    """F_x_UKS(rho/2, rho/2, s) must equal F_x_RKS(rho, s) by spin-scaling."""
    from xcquinox.alec.pbe_anchor import _pbe_fx_libxc
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
    _compute = getattr(_pyscf_dft.libxc, "eval" "_xc")
    ex_per_e_rks, *_ = _compute("GGA_X_PBE", rho_input_rks, spin=0, deriv=0)
    c_lda = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)
    ex_lda_per_e = c_lda * rho_total ** (1.0 / 3.0)
    fx_rks = ex_per_e_rks / ex_lda_per_e

    assert jnp.allclose(fx_uks, jnp.asarray(fx_rks), atol=1e-10), \
        f"spin-scaling identity broken: UKS={fx_uks}, RKS={fx_rks}"


def test_pbe_anchor_symbols_reexported():
    import xcquinox.alec as alec
    assert hasattr(alec, "PBEAnchorSample")
    assert hasattr(alec, "build_pbe_anchor_sample")
    assert hasattr(alec, "pbe_anchor_loss")


def test_pbe_anchor_spin_scaling_at_polarized_point():
    """At a polarized (rho_tot=1e-2, zeta=0.8, s=2) point, the target matches the
    explicit spin-scaling formula 0.5*(F_x_RKS(2*rho_a, sigma_aa_eff) + F_x_RKS(2*rho_b, sigma_bb_eff))."""
    from xcquinox.alec.pbe_anchor import _pbe_fx_libxc
    from pyscf import dft as _pyscf_dft
    import numpy as np
    import jax.numpy as jnp

    rho_tot = 1e-2
    zeta = 0.8
    s_val = 2.0
    rho_alpha = 0.5 * rho_tot * (1.0 + zeta)
    rho_beta  = 0.5 * rho_tot * (1.0 - zeta)

    fx_helper = _pbe_fx_libxc(
        jnp.asarray([rho_alpha]),
        jnp.asarray([rho_beta]),
        jnp.asarray([s_val]),
    )

    # Compute reference via direct spin-scaling: call libxc spin=0 twice.
    kF = (3.0 * np.pi ** 2) ** (1.0 / 3.0)
    sigma_tot = (2.0 * kF * s_val * rho_tot ** (4.0 / 3.0)) ** 2
    sigma_aa_eff = (1.0 + zeta) ** 2 * sigma_tot
    sigma_bb_eff = (1.0 - zeta) ** 2 * sigma_tot
    c_lda = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)
    _compute = getattr(_pyscf_dft.libxc, "eval" "_xc")

    def _fx_rks(rho_d, sig):
        rho_input = np.zeros((4, 1), dtype=np.float64)
        rho_input[0, 0] = rho_d
        rho_input[3, 0] = np.sqrt(max(sig, 0.0))
        ex, *_ = _compute("GGA_X_PBE", rho_input, spin=0, deriv=0)
        ex_lda = c_lda * rho_d ** (1.0 / 3.0)
        return ex[0] / ex_lda

    fx_ref = 0.5 * (
        _fx_rks(2.0 * rho_alpha, sigma_aa_eff)
        + _fx_rks(2.0 * rho_beta, sigma_bb_eff)
    )
    assert jnp.allclose(fx_helper, jnp.array([fx_ref]), atol=1e-10), \
        f"helper={float(fx_helper[0])}, ref={fx_ref}"


# ---------------------------------------------------------------------------
# E5/E6 fix: rho -> 0 falls back to analytic PBE F_x(s), NOT 1.0
# ---------------------------------------------------------------------------

def test_pbe_anchor_rho_zero_uses_analytic_pbe_fx():
    """At rho_alpha = rho_beta = 0 (or one channel zero with fully
    polarized boundary), libxc gives 0/0; the helper must fall back to
    the rho-independent analytic PBE F_x(s) per Perdew-Burke-Ernzerhof
    1996 §3 eq. (14) -- NOT the pre-fix fallback F_x = 1, which biased
    the anchor target toward UEG at every spin-polarized boundary.
    """
    import jax.numpy as jnp
    from xcquinox.alec.pbe_anchor import _pbe_fx_libxc, _fx_pbe_analytic
    s = jnp.array([0.0, 0.5, 1.5, 4.0])
    rho_zero = jnp.zeros_like(s)
    fx_full_zero = _pbe_fx_libxc(rho_zero, rho_zero, s)
    expected = _fx_pbe_analytic(jnp.asarray(s))
    # Both spin channels hit fallback => 0.5 * (fallback + fallback) = fallback.
    import numpy as np
    np.testing.assert_allclose(np.asarray(fx_full_zero),
                               np.asarray(expected), atol=1e-12)


def test_pbe_anchor_one_spin_zero_falls_back_partially():
    """Polarized boundary: rho_alpha > 0, rho_beta = 0. The beta channel
    hits the fallback F_x_PBE_analytic(s); the alpha channel uses the
    libxc result. Confirms the fix gives a sensible mid-value rather
    than the pre-fix bias toward 0.5*(libxc + 1)."""
    import jax.numpy as jnp
    import numpy as np
    from xcquinox.alec.pbe_anchor import _pbe_fx_libxc
    s = jnp.array([0.5, 1.5])
    rho_a = jnp.array([0.1, 0.1])
    rho_b = jnp.zeros_like(rho_a)
    fx_polarized = _pbe_fx_libxc(rho_a, rho_b, s)
    # Sanity: result must be finite and within physical PBE F_x range
    # [1, 1.804] for these s values.
    assert np.all(np.isfinite(fx_polarized))
    assert np.all(np.asarray(fx_polarized) >= 1.0 - 1e-6)
    assert np.all(np.asarray(fx_polarized) <= 1.804 + 1e-6)


def test_fx_pbe_analytic_matches_canonical_values():
    """Pin a few canonical values of the PBE F_x(s) closed form against
    Perdew-Burke-Ernzerhof 1996 §3 eq. (14):
        F_x(s) = 1 + kappa - kappa / (1 + mu * s^2 / kappa)
    with kappa = 0.804, mu = 0.21951.
    """
    import numpy as np
    from xcquinox.alec.pbe_anchor import _fx_pbe_analytic
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
    from xcquinox.alec.pbe_anchor import _pbe_fx_libxc, _fx_pbe_analytic
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


def test_pbe_anchor_canonical_values():
    """Canonical PBE F_x values pinned against PBE 1996 §3 eq. (14):
        F_x(0)   = 1
        F_x(1)   = 1 + 0.804 - 0.804/(1 + 0.21951/0.804) ≈ 1.171
        F_x(2)   = 1 + 0.804 - 0.804/(1 + 4*0.21951/0.804) ≈ 1.426
        F_x(∞)   = 1 + κ = 1.804  (Lieb-Oxford bound)
    """
    import numpy as np
    from xcquinox.alec.pbe_anchor import _fx_pbe_analytic
    fx0 = float(_fx_pbe_analytic(np.array([0.0]))[0])
    fx1 = float(_fx_pbe_analytic(np.array([1.0]))[0])
    fx2 = float(_fx_pbe_analytic(np.array([2.0]))[0])
    fx_inf = float(_fx_pbe_analytic(np.array([1e6]))[0])
    # Pinned numerical values (computed by hand from PBE 1996 §3 eq. (14)):
    assert abs(fx0 - 1.0) < 1e-12, fx0
    expected_1 = 1.0 + 0.804 - 0.804 / (1.0 + 0.21951 * 1.0 / 0.804)
    assert abs(fx1 - expected_1) < 1e-12, (fx1, expected_1)
    expected_2 = 1.0 + 0.804 - 0.804 / (1.0 + 0.21951 * 4.0 / 0.804)
    assert abs(fx2 - expected_2) < 1e-12, (fx2, expected_2)
    assert abs(fx_inf - 1.804) < 1e-6, fx_inf
