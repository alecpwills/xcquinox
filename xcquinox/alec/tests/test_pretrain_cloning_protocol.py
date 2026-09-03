"""The published cloning protocol's two completing features.

``pretrain._lr_schedule`` (the constant-``lr_end`` tail of arXiv:2605.10331
Sect. II.2) and ``pretrain._rho_w_sampling_mask`` (the rho*w point sampling
of Sect. II.3) are pinned here: the published shape at its boundary steps,
bit-compatibility of the default with the pre-change schedule, exact
per-system draw counts, seed/channel determinism, the draw's bias toward the
w*rho measure, every refusal (each seen to fire), the masked loss reducing to
the plain sample mean on a hand case, the energy term's independence from the
point mask, and the config-layer bounds and inert-knob refusals.
"""
import os

import jax.numpy as jnp
import numpy as np
import pytest

from xcquinox.alec.pretrain import (_PretrainLoss, _build_optimizer,
                                    _lr_schedule, _rho_w_sampling_mask)

N = 20000


def _sched(**kw):
    args = dict(lr_start=1e-3, lr_end=1e-5, n_steps=N, lr_decay_start=0.5)
    args.update(kw)
    return _lr_schedule(**args)


# ---------------------------------------------------------------------------
# The LR schedule
# ---------------------------------------------------------------------------

def test_published_shape_holds_the_floor_from_ninety_percent():
    pub = _sched(lr_decay_end=0.9)
    assert float(pub(0)) == pytest.approx(1e-3)
    assert float(pub(10000)) == pytest.approx(1e-3)
    assert float(pub(14000)) == pytest.approx((1e-3 + 1e-5) / 2, rel=1e-3)
    assert float(pub(18000)) == pytest.approx(1e-5, abs=5e-8)
    assert float(pub(19000)) == pytest.approx(1e-5, abs=5e-8)
    assert float(pub(N - 1)) == pytest.approx(1e-5, abs=5e-8)


def test_decay_end_one_reproduces_the_prior_shape_exactly():
    """The pre-change schedule: constant to 0.5 n, then linear to the last
    step -- so the LR at 90 percent of steps is 2.08e-4, the measured 20.8x
    deviation the v7 headers stated before the tail landed."""
    old = _sched(lr_decay_end=1.0)

    def analytic(s):
        if s < 10000:
            return 1e-3
        return 1e-3 + (1e-5 - 1e-3) * min((s - 10000) / 10000.0, 1.0)

    for s in range(0, N, 7):
        assert abs(float(old(s)) - analytic(s)) < 1e-12, s
    assert float(old(18000)) == pytest.approx(2.08e-4, rel=1e-6)


def test_zero_warmup_with_a_tail_decays_from_step_zero():
    z = _sched(lr_decay_start=0.0, lr_decay_end=0.9)
    assert float(z(0)) == pytest.approx(1e-3)
    assert float(z(18000)) == pytest.approx(1e-5, abs=5e-8)
    assert float(z(N - 1)) == pytest.approx(1e-5, abs=5e-8)


def test_build_optimizer_accepts_the_decay_end():
    opt = _build_optimizer(lr_start=1e-3, lr_end=1e-5, n_steps=100,
                           lr_decay_start=0.5, grad_clip=1.0,
                           lr_decay_end=0.9)
    assert opt is not None


# ---------------------------------------------------------------------------
# The rho*w sampling mask
# ---------------------------------------------------------------------------

def _two_system_columns(n0=100, n1=60, seed=7):
    rng = np.random.default_rng(seed)
    n = n0 + n1
    rho = np.abs(rng.normal(1.0, 0.5, n)) + 1e-3
    w = np.abs(rng.normal(0.1, 0.02, n)) + 1e-4
    seg = np.array([0] * n0 + [1] * n1)
    return rho, w, seg


def test_mask_counts_values_and_determinism():
    rho, w, seg = _two_system_columns()
    m1 = np.asarray(_rho_w_sampling_mask(rho, w, seg, 30, 42, channel="x"))
    m2 = np.asarray(_rho_w_sampling_mask(rho, w, seg, 30, 42, channel="x"))
    assert np.array_equal(m1, m2)
    assert set(np.unique(m1)) <= {0.0, 1.0}
    assert m1[:100].sum() == 30 and m1[100:].sum() == 30


def test_mask_differs_across_channels_and_seeds():
    rho, w, seg = _two_system_columns()
    mx = np.asarray(_rho_w_sampling_mask(rho, w, seg, 30, 42, channel="x"))
    mc = np.asarray(_rho_w_sampling_mask(rho, w, seg, 30, 42, channel="c"))
    m43 = np.asarray(_rho_w_sampling_mask(rho, w, seg, 30, 43, channel="x"))
    assert not np.array_equal(mx, mc)
    assert not np.array_equal(mx, m43)


def test_a_small_system_contributes_all_of_its_rows():
    rho, w, _ = _two_system_columns(n0=5, n1=100)
    seg = np.array([0] * 5 + [1] * 100)
    m = np.asarray(_rho_w_sampling_mask(rho, w, seg, 30, 1, channel="x"))
    assert m[:5].sum() == 5 and m[5:].sum() == 30


def test_the_draw_is_without_replacement():
    # Exact per-system ones-count == draw size is only possible without
    # replacement; additionally pin it at draw size == system size, where a
    # with-replacement draw would almost surely leave holes.
    rho, w, _ = _two_system_columns(n0=40, n1=40)
    seg = np.array([0] * 40 + [1] * 40)
    m = np.asarray(_rho_w_sampling_mask(rho, w, seg, 40, 3, channel="c"))
    assert m.sum() == 80 and set(np.unique(m)) == {1.0}


def test_the_draw_is_biased_toward_the_w_rho_measure():
    rho = np.ones(100)
    w = np.array([9.0] * 50 + [1.0] * 50)
    seg = np.zeros(100, dtype=int)
    hi = lo = 0.0
    for s in range(40):
        m = np.asarray(_rho_w_sampling_mask(rho, w, seg, 20, s, channel="x"))
        hi += m[:50].sum()
        lo += m[50:].sum()
    assert hi > 2.5 * lo, (hi, lo)


@pytest.mark.parametrize("kwargs, fragment", [
    (dict(grid_weights=None), "quadrature"),
    (dict(channel="y"), "channel"),
])
def test_mask_refusals_fire_by_name(kwargs, fragment):
    rho, w, seg = _two_system_columns()
    args = dict(rho=rho, grid_weights=w, segment=seg, points_per_system=10,
                seed=0, channel="x")
    args.update(kwargs)
    with pytest.raises(ValueError, match=fragment):
        _rho_w_sampling_mask(**args)


def test_mask_refuses_a_zero_measure_system():
    with pytest.raises(ValueError, match="positive finite"):
        _rho_w_sampling_mask(np.zeros(10), np.ones(10),
                             np.zeros(10, dtype=int), 3, 0, channel="c")


def test_mask_refuses_mismatched_column_lengths():
    with pytest.raises(ValueError, match="lengths disagree"):
        _rho_w_sampling_mask(np.ones(10), np.ones(9),
                             np.zeros(10, dtype=int), 3, 0, channel="x")


# ---------------------------------------------------------------------------
# The masked loss and the energy term
# ---------------------------------------------------------------------------

def test_masked_loss_is_the_plain_mse_over_the_sampled_rows():
    desc = jnp.array([[0.1], [0.2], [0.3]])
    ref_f = jnp.array([0.05, 0.10, 0.15])

    def model(d):
        return d[0] * 2.0  # pred - 1 aligns with ref_F

    mask = jnp.array([1.0, 0.0, 1.0])
    got = float(_PretrainLoss(weights=mask)(model, desc, ref_f))
    pred = np.array([0.2, 0.4, 0.6]) - 1.0
    want = float(np.mean(((pred - np.array([0.05, 0.10, 0.15])) ** 2)[[0, 2]]))
    assert got == pytest.approx(want, abs=1e-12)


def test_energy_term_is_independent_of_the_point_mask():
    desc = jnp.array([[0.1], [0.2], [0.3]])
    ref_f = jnp.array([0.05, 0.10, 0.15])

    def model(d):
        return d[0] * 2.0

    energy = dict(energy_row_weight=jnp.array([0.3, 0.2, 0.5]),
                  energy_segment=jnp.array([0, 0, 0], dtype=jnp.int32),
                  energy_target=jnp.array([0.4]), n_systems=1,
                  energy_weight=0.1)
    masked = _PretrainLoss(weights=jnp.array([1.0, 0.0, 1.0]), **energy)
    full = _PretrainLoss(weights=jnp.array([1.0, 1.0, 1.0]), **energy)
    ea = float(masked.parts(model, desc, ref_f)[1])
    eb = float(full.parts(model, desc, ref_f)[1])
    assert ea == eb


# ---------------------------------------------------------------------------
# Spec and config-layer bounds
# ---------------------------------------------------------------------------

def test_pretrainspec_accepts_and_bounds_the_new_fields(tmp_path):
    import xcquinox.alec as alec
    from xcquinox.alec.config import PretrainSpec

    arch = alec.get_architecture("shallow")
    base = dict(arch=arch, data_dir=str(tmp_path),
                checkpoint_dir=str(tmp_path / "ckpt"))
    ok = PretrainSpec(**base, loss_weighting="rho_w_sampled",
                      lr_decay_start=0.5, lr_decay_end=0.9,
                      points_per_system=800, sampling_seed=7)
    ok.validate()
    for kw, fragment in (
            (dict(lr_decay_start=0.5, lr_decay_end=0.4), "lr_decay_end"),
            (dict(lr_decay_end=1.5), "lr_decay_end"),
            (dict(points_per_system=0), "points_per_system"),
            (dict(sampling_seed=-1), "sampling_seed"),
    ):
        with pytest.raises(ValueError, match=fragment):
            PretrainSpec(**base, **kw).validate()
    with pytest.raises(ValueError, match="loss_weighting"):
        PretrainSpec(**base, loss_weighting="rho_w")


def test_grid_config_refuses_inert_sampling_keys_and_a_backward_window():
    from xcquinox.alec.cluster.grid_config import _build_pretrain

    with pytest.raises(ValueError, match="rho_w_sampled"):
        _build_pretrain({"data_dir": "/d", "loss_weighting": "integration",
                         "points_per_system": 800})
    with pytest.raises(ValueError, match="rho_w_sampled"):
        _build_pretrain({"data_dir": "/d", "loss_weighting": "unweighted",
                         "sampling_seed": 3})
    with pytest.raises(ValueError, match="lr_decay_end"):
        _build_pretrain({"data_dir": "/d", "lr_decay_start": 0.5,
                         "lr_decay_end": 0.3})


def test_pretrain_raw_dict_round_trips_both_modes():
    from xcquinox.alec.cluster.grid_config import (_build_pretrain,
                                                   pretrain_to_raw_dict)

    plain = _build_pretrain({"data_dir": "/d",
                             "loss_weighting": "integration"})
    raw = pretrain_to_raw_dict(plain)
    assert "points_per_system" not in raw and "sampling_seed" not in raw
    assert _build_pretrain(raw) == plain

    sampled = _build_pretrain({"data_dir": "/d",
                               "loss_weighting": "rho_w_sampled",
                               "points_per_system": 640,
                               "sampling_seed": 5, "lr_decay_end": 0.9})
    raw = pretrain_to_raw_dict(sampled)
    assert raw["points_per_system"] == 640 and raw["sampling_seed"] == 5
    assert _build_pretrain(raw) == sampled


# ---------------------------------------------------------------------------
# Review round: mesh coexistence, degenerate windows, zero-weight rows,
# and spec-bound hardening
# ---------------------------------------------------------------------------

def test_mesh_rows_ride_at_zero_weight_and_are_never_sampled():
    """Meta-GGA pretraining data appends a synthetic (r_s, s, alpha) mesh
    block with NO quadrature measure; the published objective carries no
    mesh regularizer, so under rho_w_sampled the mesh is never drawn and its
    rows enter the loss at weight zero -- a refusal here would make the
    protocol unrunnable on every meta-GGA architecture."""
    rho, w, seg = _two_system_columns()
    m = np.asarray(_rho_w_sampling_mask(rho, w, seg, 30, 42, channel="x",
                                        n_mesh_rows=7))
    assert m.shape[0] == rho.shape[0] + 7
    assert np.all(m[-7:] == 0.0)
    assert m[:100].sum() == 30 and m[100:160].sum() == 30


def test_degenerate_decay_window_is_a_step_function_not_a_full_decay():
    """lr_decay_start == lr_decay_end < 1 passed both config layers and fell
    through to a FULL-LENGTH linear decay from step 0 -- neither the
    documented constant/constant shape nor the legacy one. The empty window
    now degenerates to a step: constant lr_start to the boundary, constant
    lr_end after. The legacy quirk (end == 1.0) is preserved bit-for-bit."""
    z = _sched(lr_decay_start=0.9, lr_decay_end=0.9)
    assert float(z(0)) == pytest.approx(1e-3)
    assert float(z(17999)) == pytest.approx(1e-3)
    assert float(z(18000)) == pytest.approx(1e-5, abs=5e-8)
    assert float(z(N - 1)) == pytest.approx(1e-5, abs=5e-8)
    # end == 1.0 keeps the legacy fallthrough (full-length linear decay).
    legacy = _sched(lr_decay_start=1.0, lr_decay_end=1.0)
    assert float(legacy(N // 2)) == pytest.approx((1e-3 + 1e-5) / 2, rel=1e-3)


def test_zero_measure_rows_shrink_the_draw_instead_of_crashing():
    """A production grid can carry exactly-zero quadrature weights (the repo
    reference file does: 570/620 positive rows); numpy's choice() raises an
    opaque error when size exceeds the positive-probability support. The
    draw clamps to the positive-measure row count instead."""
    rho = np.ones(10)
    w = np.array([1.0] * 6 + [0.0] * 4)
    seg = np.zeros(10, dtype=int)
    m = np.asarray(_rho_w_sampling_mask(rho, w, seg, 8, 0, channel="c"))
    assert m.sum() == 6
    assert np.all(m[6:] == 0.0)


def test_spec_bounds_reject_aliasing_and_non_integral_values(tmp_path):
    import xcquinox.alec as alec
    from xcquinox.alec.config import PretrainSpec

    arch = alec.get_architecture("shallow")
    base = dict(arch=arch, data_dir=str(tmp_path),
                checkpoint_dir=str(tmp_path / "ckpt"))
    # A truncating float seed aliases another mask while the record shows the
    # written value; whole numbers only, booleans included.
    for kw, fragment in (
            (dict(sampling_seed=0.7), "sampling_seed"),
            (dict(points_per_system=float("inf")), "points_per_system"),
            (dict(points_per_system=True), "points_per_system"),
            (dict(sampling_seed=True), "sampling_seed"),
            (dict(lr_decay_end=True), "lr_decay_end"),
    ):
        with pytest.raises(ValueError, match=fragment):
            PretrainSpec(**base, **kw).validate()


def test_missing_system_column_is_named_once(tmp_path):
    """Under the total footing the exchange and correlation blocks share
    'system_all'; the refusal must not name the same key twice."""
    import os
    import xcquinox.alec as alec
    from xcquinox.alec.config import PretrainSpec
    from xcquinox.alec.pretrain import run_pretrain

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    n = 40
    rho = np.linspace(0.01, 5.0, n)
    np.savez(os.path.join(str(data_dir), "pretrain_data.npz"),
             rho_all=rho, sigma_all=rho ** 2,
             Fx_all=np.zeros(n), Fc_all=np.zeros(n),
             weights_all=np.full(n, 0.05))
    spec = PretrainSpec(arch=alec.get_architecture("shallow"),
                        data_dir=str(data_dir),
                        checkpoint_dir=str(tmp_path / "ckpt"),
                        n_steps=2, lr_start=1e-2, lr_end=1e-5,
                        lr_decay_start=0.0, grad_clip=1.0, seed=0,
                        loss_weighting="rho_w_sampled",
                        points_per_system=5, sampling_seed=1)
    with pytest.raises(ValueError) as err:
        run_pretrain(spec)
    assert str(err.value).count("'system_all'") == 1


def test_metagga_run_with_a_mesh_completes_under_rho_w_sampled(tmp_path):
    """The v7 meta-GGA group's exact combination: a data file carrying the
    synthetic mesh block, a pure meta-GGA architecture, and the sampled
    objective. Before the mesh-at-zero-weight rule this raised at the node
    and would have killed two of the five mgga pretrain tasks on the next
    fresh submission; now the fit runs, the mesh contributes zero loss
    share, and the metadata records the physical sample."""
    from xcquinox.alec.config import ArchitectureConfig, PretrainSpec
    from xcquinox.alec.pretrain import run_pretrain

    n_atomic, n_mesh = 60, 12
    np.savez(tmp_path / "pretrain_data.npz",
             rho_all=np.linspace(0.1, 2.0, n_atomic),
             sigma_all=np.linspace(0.0, 1.0, n_atomic),
             metagga_all=np.linspace(0.0, 2.0, n_atomic).reshape(-1, 1),
             Fx_all=np.zeros(n_atomic), Fc_all=np.full(n_atomic, -0.1),
             Fx_scan_all=np.full(n_atomic, 0.3),
             Fc_scan_all=np.full(n_atomic, -0.4),
             weights_all=np.full(n_atomic, 0.05),
             system_all=np.array([0] * 40 + [1] * 20, dtype=np.int64),
             rho_mesh=np.linspace(0.2, 1.0, n_mesh),
             sigma_mesh=np.linspace(0.1, 0.6, n_mesh),
             metagga_mesh=np.linspace(0.0, 3.0, n_mesh).reshape(-1, 1),
             Fx_scan_mesh=np.full(n_mesh, 0.7),
             Fc_scan_mesh=np.full(n_mesh, -0.7),
             weights_mesh=np.full(n_mesh, 0.25))
    arch = ArchitectureConfig.from_spec(
        "t_mgga_sampled", 2, 8, descriptors=["metagga"], meta_gga=True)
    spec = PretrainSpec(arch=arch, data_dir=str(tmp_path),
                        checkpoint_dir=str(tmp_path / "ck"),
                        n_steps=3, lr_start=1e-2, lr_end=1e-5,
                        lr_decay_start=0.0, grad_clip=1.0, seed=0,
                        loss_weighting="rho_w_sampled",
                        points_per_system=15, sampling_seed=4)
    result = run_pretrain(spec)
    assert result["loss_weighting"] == "rho_w_sampled"
    assert result["sampled_rows_x"] == 30 and result["sampled_rows_c"] == 30
    assert np.isfinite(result["final_loss_x"])
    assert np.isfinite(result["final_loss_c"])
    assert result["mesh_loss_share_x"] == pytest.approx(0.0)
    assert result["mesh_loss_share_c"] == pytest.approx(0.0)
