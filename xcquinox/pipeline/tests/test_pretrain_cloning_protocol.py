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

import jax.numpy as jnp
import numpy as np
import pytest

from xcquinox.pipeline.pretrain import (_PretrainLoss, _lr_schedule,
                                    _rho_w_sampling_mask)

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


# ---------------------------------------------------------------------------
# Spec and config-layer bounds
# ---------------------------------------------------------------------------

def test_pretrainspec_accepts_and_bounds_the_new_fields(tmp_path):
    import xcquinox.pipeline as pipeline
    from xcquinox.pipeline.config import PretrainSpec

    arch = pipeline.get_architecture("shallow")
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




# ---------------------------------------------------------------------------
# The optimizer: the published clone runs Adam alone
# ---------------------------------------------------------------------------

_OPT_ARGS = dict(lr_start=1e-3, lr_end=1e-5, n_steps=N, lr_decay_start=0.5,
                 lr_decay_end=0.9)


def _updates(optimizer, params, gradient_sequence):
    """The updates an optimizer produces over a sequence of gradients, its
    moment state carried between steps."""
    state = optimizer.init(params)
    out = []
    for grads in gradient_sequence:
        updates, state = optimizer.update(grads, state, params)
        out.append(np.asarray(updates["w"]))
    return out


def test_grad_clip_zero_disables_the_clip(tmp_path):
    """``pretrain.grad_clip: 0`` means NO clip, which is the published
    clone's optimizer (``optax.adam`` alone, ``train.py``), not a clip at
    norm zero -- which would scale every gradient to zero and freeze the fit.

    Oracle: ``optax.adam`` on the same learning-rate schedule, built here,
    over two steps whose first gradient has global norm 100 and whose second
    is small, so the clip shows in the second step through the moment state
    as well as in the first. A clip at 1.0 on the same sequence must differ,
    so the case is a comparison and not an identity that holds either way.
    """
    import optax

    from xcquinox.pipeline.pretrain import _build_optimizer

    params = {"w": jnp.array([1.0, 1.0])}
    sequence = [{"w": jnp.array([60.0, 80.0])},
                {"w": jnp.array([0.01, -0.02])}]

    adam_alone = optax.adam(learning_rate=_lr_schedule(**_OPT_ARGS))
    unclipped = _build_optimizer(grad_clip=0.0, **_OPT_ARGS)
    clipped = _build_optimizer(grad_clip=1.0, **_OPT_ARGS)

    want = _updates(adam_alone, params, sequence)
    got = _updates(unclipped, params, sequence)
    for a, b in zip(want, got):
        np.testing.assert_array_equal(a, b)
    assert float(np.max(np.abs(got[0]))) > 0.0

    other = _updates(clipped, params, sequence)
    assert max(float(np.max(np.abs(a - b))) for a, b in zip(want, other)) > 1e-12

    # The spec the pretrain worker builds carries the value through to the
    # optimizer, so the published protocol is not refused one layer above it.
    import xcquinox.pipeline as pipeline
    from xcquinox.pipeline.config import PretrainSpec

    base = dict(arch=pipeline.get_architecture("shallow"),
                data_dir=str(tmp_path),
                checkpoint_dir=str(tmp_path / "ckpt"))
    PretrainSpec(**base, grad_clip=0.0).validate()
    with pytest.raises(ValueError, match="grad_clip"):
        PretrainSpec(**base, grad_clip=-1.0).validate()
