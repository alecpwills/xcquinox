"""Compile stability of the de-fused training step across successive steps.

Molecule shapes are fixed for the whole of a training run, so once the kernels
are warm a step must compile nothing. A steady-state compile rate is a resource
defect rather than a numerical one: each compilation permanently retains its LLVM
code mapping (``jax.clear_caches()`` cannot release backing stores the runtime has
already allocated), so a per-step compile drives the process mapping count toward
``vm.max_map_count`` = 65530 and the job dies with SIGSEGV once it is crossed --
observed on the cluster as ``LLVM compilation error: Cannot allocate memory`` at a
resident set far below node RAM.

The instrument is checked against its own oracle first (a repeated call must add
no compile, a new shape must add one), so a silently broken counter cannot pass
the ratchet test by reporting zeros.
"""
import jax
import equinox as eqx
import jax.numpy as jnp
import pytest

from xcquinox.alec.defused_grad import defused_value_and_grad
from xcquinox.alec.oneshot import dm_prediction_for_loss, grid_density_for_loss
from xcquinox.alec.padding import common_pad_target
from xcquinox.alec.solver import SolverBackend, SolverConfig, SolverMode
from xcquinox.alec.train import build_optimizer, _trainable_params
from xcquinox.alec.tests.fixtures.compile_counter import CompileCounter
from xcquinox.alec.tests.fixtures.molecules import o_atom
from xcquinox.alec.tests.test_defused_grad import (
    _atom_anchor_l5_group, _l5_group, _SC_FULL,
)


def _mapping_count():
    """Process memory-mapping count -- the quantity bounded by vm.max_map_count."""
    with open("/proc/self/maps") as fh:
        return sum(1 for _ in fh)


def test_compile_counter_detects_repeat_and_new_shape():
    """Oracle for the instrument used by the ratchet test."""
    with CompileCounter() as cc:
        fn = jax.jit(lambda x: x + x)
        jax.block_until_ready(fn(jnp.ones(3)))
        first = cc.count
        jax.block_until_ready(fn(jnp.ones(3)))
        repeat = cc.count
        jax.block_until_ready(fn(jnp.ones(5)))
        new_shape = cc.count
    assert first >= 1, "a first call must compile"
    assert repeat == first, "an identical repeat call must not recompile"
    assert new_shape > repeat, "a new input shape must compile"


def _run_defused_steps(*, padded, n_steps=5):
    """Train a real 2-molecule FULL-SCF group, returning per-step counters.

    The group is H2 <-> 2H, whose two molecules have DIFFERENT AO and grid
    extents, so the padded variant genuinely exercises the shape-padding pass
    rather than a degenerate single-shape case.
    """
    model, gloss, gbatch, cw, relative = _l5_group(
        solver_config=_SC_FULL, extra_keys=("eri",), synth_refs=True)
    pad_target = common_pad_target(gbatch["mol_data"]) if padded else None
    opt = build_optimizer(lr_start=1e-3, lr_end=1e-5, n_steps=16,
                          lr_decay_start=0.0, grad_clip=1.0, weight_decay=0.0)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    compiles, mappings = [], []
    with CompileCounter() as cc:
        for _ in range(n_steps):
            before = cc.count
            (_total, _comps), grads = defused_value_and_grad(
                gloss, model, gbatch, cw, relative, pad_target=pad_target)
            jax.block_until_ready(grads)
            updates, opt_state = opt.update(grads, opt_state, _trainable_params(model))
            model = eqx.apply_updates(model, updates)
            compiles.append(cc.count - before)
            mappings.append(_mapping_count())
    return compiles, mappings


# Allocator arenas and thread stacks move the mapping count by a few tens in
# either direction between steps, so exact equality is not a usable oracle. The
# defect being guarded against grew the count by ~510 per step, monotonically;
# a per-step budget of 100 sits well above the observed jitter and well below
# the leak, so the two cannot be confused.
_MAPPING_STEP_TOLERANCE = 100


def _assert_steady_state_is_quiet(compiles, mappings, *, label):
    """From the third step on, shapes are unchanged: nothing may compile, and
    the process mapping count must not trend upward."""
    steady = compiles[2:]
    assert steady == [0] * len(steady), (
        f"de-fused step ({label}) recompiles on every training step at fixed "
        f"shapes; compiles per step = {compiles}, "
        f"mapping count per step = {mappings}"
    )
    growth = mappings[-1] - mappings[2]
    budget = _MAPPING_STEP_TOLERANCE * max(1, len(mappings) - 3)
    assert growth < budget, (
        f"process mapping count still climbs ({label}) after warmup: "
        f"{mappings} (growth {growth} over the steady-state steps, budget "
        f"{budget}); this is what crosses vm.max_map_count and kills the job"
    )


def test_defused_step_stops_compiling_after_warmup():
    """Successive de-fused FULL-SCF steps on fixed shapes must not recompile.

    Two warmup steps are allowed: the first compiles the forward and backward
    kernels, the second any kernel first reached once an update has been
    applied. From the third step on the count must be exactly zero, and the
    process mapping count must stop growing.
    """
    compiles, mappings = _run_defused_steps(padded=False)
    _assert_steady_state_is_quiet(compiles, mappings, label="unpadded")


@pytest.mark.parametrize("channel_fn",
                         [grid_density_for_loss, dm_prediction_for_loss],
                         ids=["grid_density", "dm_prediction"])
def test_scf_channel_seam_stops_compiling_under_eager_grad(channel_fn):
    """Both self-consistent loss channels must compile once, not once per call.

    ``_grid_term`` and ``_dm_term`` are structural twins: each reaches
    ``run_scf`` -> ``lax.scan`` with ``forward_only=False``. Under the de-fused
    step's eager reverse-mode AD an unjitted scan of either shape leaks two XLA
    executables per call. Fixing one channel and not the other leaves the
    ratchet intact for any loss that uses the other (``B_atomization_plus_dm``,
    ``D2_delta_ae_plus_dm``), so the invariant is asserted per channel.
    """
    model, _gloss, gbatch, _cw, _relative = _l5_group(
        solver_config=_SC_FULL, extra_keys=("eri",), synth_refs=True)
    mol_data = gbatch["mol_data"][0]

    def scalar(m):
        return jnp.sum(jnp.asarray(channel_fn(m, mol_data, _SC_FULL)) ** 2)

    compiles = []
    with CompileCounter() as cc:
        for _ in range(4):
            before = cc.count
            jax.block_until_ready(eqx.filter_grad(scalar)(model))
            compiles.append(cc.count - before)

    steady = compiles[1:]
    assert steady == [0] * len(steady), (
        f"{channel_fn.__name__} recompiles on every eager-grad call: "
        f"compiles per call = {compiles}"
    )


def test_density_channel_does_not_trace_the_pyscfad_backend():
    """The pyscfad backend must reach the density channel untraced.

    ``run_pyscfad_scf`` requires concrete arrays for libcint integral
    construction and raises ``RuntimeError`` the moment it sees a tracer. That
    guard sits AHEAD of the ONESHOT short-circuit, so ONESHOT is caught too:
    jitting the density channel unconditionally breaks every pyscfad mode, not
    just the self-consistent ones. Only the MANUAL backend has the ``lax.scan``
    the jit exists to stabilise.
    """
    model, _gloss, gbatch, _cw, _relative = _l5_group(
        solver_config=_SC_FULL, extra_keys=("eri",), synth_refs=True)
    pyscfad_oneshot = SolverConfig(backend=SolverBackend.PYSCFAD,
                                   mode=SolverMode.ONESHOT, max_cycles=0)
    rho = grid_density_for_loss(model, gbatch["mol_data"][0], pyscfad_oneshot)
    assert bool(jnp.all(jnp.isfinite(rho))), "pyscfad density must be finite"


def test_defused_step_stops_compiling_after_warmup_padded():
    """Same invariant with the shape-padding pass active.

    Padding rewrites every molecule to one common shape and traces the
    occupation counts, so it changes what the SCF kernel is keyed on. The
    steady-state compile count must still be zero: padding must not reintroduce
    a per-step recompile, and the two passes must not fight over cache keys.
    """
    compiles, mappings = _run_defused_steps(padded=True)
    _assert_steady_state_is_quiet(compiles, mappings, label="padded")


def test_multi_group_padded_loop_stops_compiling_after_first_epoch():
    """A NEW group each step under ONE global pad target, as in production.

    The per_molecule loop computes a single ``common_pad_target`` over the whole
    batch (train.py) and then presents a DIFFERENT group every optimizer step.
    The single-group tests above repeat one group, so a kernel keyed on group
    identity -- a different molecule-field-presence pattern, spin type, or
    molecule count -- would pass them and still compile on every step of a real
    run, which is the per-step mapping-count growth observed on the cluster.

    Epoch 1 may compile freely (each variant's first encounter). After one full
    epoch every variant has been seen, so epochs 2-3 must compile NOTHING and
    the mapping count must stop trending upward. Group A is a two-molecule
    reaction group (RKS H2 + UKS H, synthetic vxc/rho refs present); group B is
    a single-atom anchor group (UKS O, no refs) -- different species, different
    natural shapes, different field-presence pattern, different molecule count.
    """
    model, gloss_a, gbatch_a, cw, relative = _l5_group(
        solver_config=_SC_FULL, extra_keys=("eri",), synth_refs=True)
    _mb, loss_b, batch_b, _cwb, _relb = _atom_anchor_l5_group(
        o_atom(), "O", -74.8, solver_config=_SC_FULL, extra_keys=("eri",))
    groups = [(gloss_a, gbatch_a), (loss_b, batch_b)]
    pad_target = common_pad_target(
        tuple(gbatch_a["mol_data"]) + tuple(batch_b["mol_data"]))

    opt = build_optimizer(lr_start=1e-3, lr_end=1e-5, n_steps=16,
                          lr_decay_start=0.0, grad_clip=1.0, weight_decay=0.0)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    epoch_compiles, mappings = [], []
    with CompileCounter() as cc:
        for _epoch in range(3):
            before = cc.count
            for gloss, gbatch in groups:
                (_total, _comps), grads = defused_value_and_grad(
                    gloss, model, gbatch, cw, relative, pad_target=pad_target)
                jax.block_until_ready(grads)
                updates, opt_state = opt.update(
                    grads, opt_state, _trainable_params(model))
                model = eqx.apply_updates(model, updates)
                mappings.append(_mapping_count())
            epoch_compiles.append(cc.count - before)

    assert epoch_compiles[0] >= 1, (
        "first encounters must compile -- a zero here means the counter is "
        "broken, not that the loop is quiet")
    assert epoch_compiles[1:] == [0, 0], (
        f"kernels keyed on group identity keep compiling after every group "
        f"has been seen once; compiles per epoch = {epoch_compiles}, "
        f"mapping count per step = {mappings}"
    )
    n_groups = len(groups)
    growth = mappings[-1] - mappings[n_groups - 1]
    budget = _MAPPING_STEP_TOLERANCE * (len(mappings) - n_groups)
    assert growth < budget, (
        f"process mapping count still climbs across epochs 2-3: {mappings} "
        f"(growth {growth} over the steady-state steps, budget {budget})"
    )
