"""Regression guard: cusp descriptor columns must be bounded for MLP input.

Prior to commit (this fix), ``compute_cusp_descriptor`` returned
``[cusp_factor, log_weighted_Z]`` where ``log_weighted_Z`` had a dynamic
range of ~14 units on physical atomic grids (tail values ~-2,
near-nucleus values ~12). This unnormalized heavy-tail input caused the
xnet MLP to saturate at F_x ~ 1.4 for architectures using the cusp
descriptor (deep_cusp, deep_cusp_attn, deep_combined, deep_combined_attn),
producing flat parity plots that did not reach the Lieb-Oxford bound
(F_x -> 1.804).

The fix bounds the second column via ``tanh(log_weighted_Z / 5)``, keeping
the feature in (-1, 1) while preserving monotonicity.
"""
import jax.numpy as jnp
import numpy as np

from xcquinox.features import compute_cusp_descriptor


def test_cusp_descriptor_columns_are_bounded():
    """Both columns of compute_cusp_descriptor must lie in [0, 1] (col 0)
    and [-1, 1] (col 1) on a physical atomic grid spanning a wide range
    of distances from the nucleus."""
    # O atom at origin, radial grid 0.001 to 20 Å, covers core through tail.
    r = np.logspace(-3, 1.3, 500)  # 0.001 to ~20
    grid_coords = np.stack(
        [r, np.zeros_like(r), np.zeros_like(r)], axis=1,
    )
    # Heavy nucleus (Z=8, oxygen) amplifies extreme values, worst case.
    nuc_coords = np.array([[0.0, 0.0, 0.0]])
    nuc_charges = np.array([8])
    # log_transform=True is the bounded form this test validates (and the form
    # every cusp-using arch, all descriptor_log_transform=True, receives). The
    # default raw form tanh(weighted_Z/5) intentionally saturates to 1.0 near
    # the nucleus and is exercised by test_cusp_descriptor_columns_raw_saturates.
    d = np.asarray(compute_cusp_descriptor(
        jnp.asarray(grid_coords),
        jnp.asarray(nuc_coords),
        jnp.asarray(nuc_charges),
        log_transform=True,
    ))

    # Column 0: cusp_factor = exp(-2 Z r), physically in [0, 1].
    assert d[:, 0].min() >= 0.0
    assert d[:, 0].max() <= 1.0 + 1e-12

    # Column 1: tanh(log_weighted_Z / 5), physically in (-1, 1).
    # The bounded form should never leave this interval on ANY grid.
    assert d[:, 1].min() > -1.0, (
        f"cusp col 1 unbounded below: min = {d[:, 1].min():.3f}"
    )
    assert d[:, 1].max() < 1.0, (
        f"cusp col 1 unbounded above: max = {d[:, 1].max():.3f}"
    )

    # Stronger guard: the old unbounded form hit +11.8 on this grid. The
    # bounded form must stay well below 2.0, which catches any accidental
    # revert to the raw log transform.
    assert d[:, 1].max() < 2.0, (
        "cusp col 1 appears unbounded (regression to unbounded log form?): "
        f"max = {d[:, 1].max():.3f}"
    )


def test_cusp_descriptor_columns_raw_saturates():
    """Default (log_transform=False) col 1 = tanh(weighted_Z/5) saturates to
    exactly 1.0 at near-nucleus points, documented legacy behavior. The
    notransform archs that take this default carry NO cusp descriptor, so the
    saturation is never fed to a network; cusp-using archs all pass
    log_transform=True (see test_cusp_descriptor_columns_are_bounded)."""
    r = np.logspace(-3, 1.3, 500)
    grid_coords = np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=1)
    d = np.asarray(compute_cusp_descriptor(
        jnp.asarray(grid_coords),
        jnp.asarray(np.array([[0.0, 0.0, 0.0]])),
        jnp.asarray(np.array([8])),
    ))
    assert d[:, 1].max() >= 1.0 - 1e-12          # saturates near the nucleus
    assert d[:, 1].max() <= 1.0 + 1e-12          # still bounded above by tanh


def test_cusp_descriptor_monotone_in_distance():
    """log_weighted_Z_bounded should monotonically decrease with distance
    from the single nucleus, preserved by tanh's monotonicity."""
    r = np.linspace(0.1, 5.0, 50)
    grid_coords = np.stack(
        [r, np.zeros_like(r), np.zeros_like(r)], axis=1,
    )
    nuc_coords = np.array([[0.0, 0.0, 0.0]])
    nuc_charges = np.array([6])
    d = np.asarray(compute_cusp_descriptor(
        jnp.asarray(grid_coords),
        jnp.asarray(nuc_coords),
        jnp.asarray(nuc_charges),
    ))
    # As r increases, weighted_Z_sum = Z/r decreases, so col 1 should decrease.
    diffs = np.diff(d[:, 1])
    assert np.all(diffs <= 1e-10), (
        "col 1 not monotonically non-increasing with distance"
    )


def test_cusp_descriptor_shape_preserved():
    """The descriptor still returns shape (N, 2), no schema change."""
    coords = np.random.default_rng(0).standard_normal((17, 3))
    nuc_coords = np.array([[0.0, 0.0, 0.0]])
    nuc_charges = np.array([3])
    d = compute_cusp_descriptor(
        jnp.asarray(coords),
        jnp.asarray(nuc_coords),
        jnp.asarray(nuc_charges),
    )
    assert d.shape == (17, 2)
