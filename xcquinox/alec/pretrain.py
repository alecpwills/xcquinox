"""xcquinox.alec.pretrain: Pretraining and legacy checkpoint loading.

Provides run_pretrain plus the legacy-checkpoint loaders (from_legacy_step3b,
_load_one_network, _count_disk_records, _metadata_preflight,
_disk_record_preflight) and the legacy lob_lim constants.

Pretrain data is stored as .npz (numpy compressed archive) rather than .pkl.
The loader falls back to .pkl for legacy files.
"""
import json
import os
import pickle  # noqa: S403 -- only used for the trusted legacy local .pkl fallback below
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from equinox._filters import is_array_like

import xcquinox.net
import xcquinox.train

from xcquinox.alec.config import ArchitectureConfig, PretrainSpec
from xcquinox.alec.networks import AlecGGA_XNet, AlecGGA_CNet, create_network_pair
from xcquinox.utils import lda_x, pw92c_unpolarized_scalar


_RHO_FLOOR_INTEGRATION = 1e-18


def _compute_integration_weights(rho, grid_weights=None):
    """Return ``(w_x, w_c)`` integration weights for pretraining.

    Weight convention: the per-point weight is the
    FIRST power (linear) of ``|ρ_i · ε_LDA_i|``, optionally multiplied by
    the Becke-Lebedev quadrature weight ``w_grid_i``.  The resulting loss
    is a ``|ρ · ε_LDA|``-magnitude-weighted mean of the squared per-point
    enhancement-factor residual::

        L = Σ_i w_i · (F_nn_i - F_ref_i)²  /  Σ_i w_i
            where w_i = |ρ_i · ε_LDA_i|   (or  |ρ_i · ε_LDA_i| · w_grid_i
                                             when grid_weights is supplied).

    This is NOT the squared integrated XC-energy residual.  A true
    integrated-energy L2 loss would require the square of the energy-density
    weight, i.e.  ``w_i = (ρ_i · ε_LDA_i · w_grid_i)²``.  The linear form
    was retained because it is the established convention in the codebase
    (all prior pretrain runs used it) and existing tests pin this behavior.
    The ``|ρ · ε_LDA|`` factor still steers gradient attention toward
    energetically important regions without the large dynamic range that the
    squared form would introduce.

    When ``grid_weights`` is supplied the weights incorporate the quadrature
    measure ``dr_i``, improving the energy-density calibration relative to the
    unweighted form.  When ``grid_weights`` is ``None`` the loss reverts to
    the legacy ``|ρ ε_LDA|``-weighted mean per sample; ``run_pretrain`` emits
    a ``RuntimeWarning`` in that case and records
    ``integration_weights_complete=False`` in the pretrain metadata.

    Parameters
    ----------
    rho : jnp.ndarray
        Electron density at grid points, shape ``(N,)``.
    grid_weights : jnp.ndarray | None
        Becke-Lebedev quadrature weights ``dr_i`` per grid point, shape
        ``(N,)``.  When supplied the per-point weight becomes
        ``|ρ ε_LDA| · w_grid``.  When ``None`` (legacy / pre-fix behavior)
        the ``dr_i`` factor is omitted; callers should expect the
        ``RuntimeWarning`` from ``run_pretrain``.

    Returns
    -------
    (w_x, w_c) : tuple of jnp.ndarray, each shape ``(N,)``
        Non-negative integration weights for the exchange and correlation
        networks respectively.
    """
    rho_safe = jnp.maximum(rho, _RHO_FLOOR_INTEGRATION)
    eps_x_lda = lda_x(rho_safe)
    # The correlation weight intentionally uses the unpolarized PW92 baseline on
    # the total density even when training a spin-polarization-aware (polarized)
    # cnet: the integration weights are a grid-importance measure on the total
    # density, and spin polarization enters the model via the zeta descriptor
    # column, not the loss weighting. (The Fc targets in pretrain_data_gen use
    # spin-resolved libxc for open-shell atoms; that asymmetry is intentional,
    # not a baseline bug.)
    eps_c_lda = pw92c_unpolarized_scalar(rho_safe)
    w_x = jnp.abs(rho_safe * eps_x_lda)
    w_c = jnp.abs(rho_safe * eps_c_lda)
    if grid_weights is not None:
        gw = jnp.asarray(grid_weights)
        w_x = w_x * gw
        w_c = w_c * gw
    return jnp.broadcast_to(w_x, rho_safe.shape), \
           jnp.broadcast_to(w_c, rho_safe.shape)


class _PretrainLoss(eqx.Module):
    """Pretraining objective: point-wise enhancement-factor residual plus an
    optional per-system energy term.

    Networks return the enhancement factor F; targets are stored as ``F - 1``,
    so ``pred - 1`` aligns with ``ref_F``. ``weights=None`` gives the plain
    mean of squared residuals; a 1-D ``weights`` aligned with the rows gives
    the integration-weighted reduction
    ``sum(w r^2) / (sum(w) + 1e-12)``.

    The energy term is

        w_E * (1 / N_sys) sum_s ( sum_{i in s} w_i e_LDA_i F^NN_i - E_s )^2

    in Hartree^2, with ``E_s`` the parent's own value of the same quadrature
    (``pretrain_data_gen._system_energy_targets``). It exists because the
    point-wise residual alone does not bound a system's energy: measured across
    seven architectures, the one with the LOWEST exchange residual carried the
    LARGEST atomization-energy offset from its parent. The mean over systems
    rather than the sum keeps the term's magnitude independent of how many
    systems the set holds, so ``w_E`` means the same thing for a four-atom file
    and a thirty-seven-system one. Rows belonging to no system -- the synthetic
    (r_s, s, alpha) mesh -- carry zero weight and the sink segment index
    ``n_systems``, which is asked of ``segment_sum`` and then dropped.

    At ``energy_weight == 0`` the term is not evaluated at all, so the returned
    value is the pre-existing loss bit for bit; that short circuit is the
    reason the reduction is written twice.
    """
    weights: jnp.ndarray | None = None
    energy_row_weight: jnp.ndarray | None = None
    energy_segment: jnp.ndarray | None = None
    energy_target: jnp.ndarray | None = None
    energy_weight: float = eqx.field(static=True, default=0.0)
    n_systems: int = eqx.field(static=True, default=0)

    def parts(self, model, descriptors, ref_F):
        """``(pointwise, energy)`` -- the two terms, the second unweighted."""
        pred = jax.vmap(model)(descriptors).squeeze()
        shifted = pred - 1.0
        residual_sq = (shifted - ref_F) ** 2
        if self.weights is None:
            pointwise = jnp.mean(residual_sq)
        else:
            w = self.weights
            pointwise = jnp.sum(w * residual_sq) / (jnp.sum(w) + 1e-12)
        if self.energy_target is None:
            return pointwise, jnp.zeros_like(pointwise)
        e_nn = jax.ops.segment_sum(
            self.energy_row_weight * pred, self.energy_segment,
            num_segments=self.n_systems + 1)[:self.n_systems]
        delta = e_nn - self.energy_target
        return pointwise, jnp.sum(delta * delta) / self.n_systems

    def __call__(self, model, descriptors, ref_F):
        if self.energy_weight == 0.0 or self.energy_target is None:
            pred = jax.vmap(model)(descriptors).squeeze()
            pred = pred - 1.0
            residual_sq = (pred - ref_F) ** 2
            if self.weights is None:
                return jnp.mean(residual_sq)
            w = self.weights
            return jnp.sum(w * residual_sq) / (jnp.sum(w) + 1e-12)
        pointwise, energy = self.parts(model, descriptors, ref_F)
        return pointwise + self.energy_weight * energy


def _energy_term_inputs(pretrain_data, *, weight_key, lda_key, segment_key,
                        target_key, n_mesh):
    """``(row_weight, segment, target, n_systems)`` for one network's energy term.

    ``row_weight_i = w_i e_LDA_i`` is Hartree per unit enhancement factor, so
    ``sum_{i in s} row_weight_i F^NN_i`` is the network's XC energy of system
    ``s`` on the rows the file stores. ``n_mesh`` synthetic rows are appended
    with zero weight and the sink segment index, so the row set matches the
    descriptor tensor the mesh was concatenated onto.
    """
    target = jnp.asarray(pretrain_data[target_key])
    n_systems = int(target.shape[0])
    row_weight = (jnp.asarray(pretrain_data[weight_key])
                  * jnp.asarray(pretrain_data[lda_key]))
    segment = jnp.asarray(pretrain_data[segment_key], dtype=jnp.int32)
    if n_mesh:
        row_weight = jnp.concatenate([row_weight, jnp.zeros(n_mesh)])
        segment = jnp.concatenate(
            [segment, jnp.full(n_mesh, n_systems, dtype=jnp.int32)])
    return row_weight, segment, target, n_systems


# ---------------------------------------------------------------------------
# Held-out-system validation and the stop criterion
# ---------------------------------------------------------------------------

def _validation_systems(system_natoms, fraction, seed):
    """Indices of the systems held out of the fit, as a sorted tuple.

    The split draws from the MOLECULES only. Every single-atom system is an
    anchor: the Section 3.3 certificate bounds each pool atom's E_xc at
    tol_atom, and every atomization energy is a molecule minus its atoms, so
    an atom the fit never saw would fail the acceptance test by construction.
    What validation is for here is the molecular extrapolation of the
    density-matrix features -- the failure the campaign measured -- and that
    is what the molecules measure.

    ``fraction`` is a fraction of the ELIGIBLE (multi-nucleus) systems, rounded
    to the nearest integer with a tie rounding up, then floored at one and
    capped at all but one. Both bounds are consistency requirements rather
    than tuned values: a non-zero fraction that held out nothing would leave
    the stop criterion with no score, and a fit left with atoms alone is the
    coverage failure the set change exists to remove. Fewer than two eligible
    systems cannot satisfy both bounds at once, so the split is then empty.
    The permutation is seeded so every architecture in a sweep holds out the
    same systems and their validation numbers are comparable; the held-out
    names are written to the run's metadata, so the split is checkable
    independently of the generator's stream.
    """
    natoms = np.asarray(system_natoms).reshape(-1)
    eligible = np.flatnonzero(natoms > 1)
    if float(fraction) <= 0.0 or eligible.shape[0] < 2:
        return ()
    k = int(np.floor(float(fraction) * eligible.shape[0] + 0.5))
    k = max(1, min(k, int(eligible.shape[0]) - 1))
    order = np.random.default_rng(int(seed)).permutation(eligible)
    return tuple(sorted(int(i) for i in order[:k]))


def _system_split_arrays(segment, n_systems, held_out):
    """Row masks and segment renumberings for a held-out-system split.

    ``segment`` is the per-row system index, with ``n_systems`` marking a row
    that belongs to no system -- the synthetic (r_s, s, alpha) mesh. Mesh rows
    always train: the mesh is a regularizer of the functional form, not a
    system whose energy is predicted, so holding it out would measure nothing.

    Returns ``(train_mask, val_mask, train_remap, val_remap, train_ids,
    val_ids)``. The remaps carry the kept systems onto ``0..n_kept-1`` and
    everything else, including the sink, onto ``n_kept``, so a restricted
    segment array is still a valid ``segment_sum`` index with its own sink.
    """
    seg = np.asarray(segment).reshape(-1)
    n_systems = int(n_systems)
    if seg.shape[0] and int(seg.max()) > n_systems:
        raise ValueError(
            f"_system_split_arrays: row index {int(seg.max())} exceeds the "
            f"sink index {n_systems}; the segment array and the system table "
            "disagree.")
    held = np.zeros(n_systems + 1, dtype=bool)
    for i in held_out:
        if not 0 <= int(i) < n_systems:
            raise ValueError(
                f"_system_split_arrays: held-out index {int(i)} is not a "
                f"system of a {n_systems}-system table.")
        held[int(i)] = True
    val_mask = held[seg]
    train_mask = ~val_mask
    train_ids = np.flatnonzero(~held[:n_systems]).astype(np.int64)
    val_ids = np.flatnonzero(held[:n_systems]).astype(np.int64)
    train_remap = np.full(n_systems + 1, train_ids.shape[0], dtype=np.int32)
    train_remap[train_ids] = np.arange(train_ids.shape[0], dtype=np.int32)
    val_remap = np.full(n_systems + 1, val_ids.shape[0], dtype=np.int32)
    val_remap[val_ids] = np.arange(val_ids.shape[0], dtype=np.int32)
    return (train_mask, val_mask, train_remap, val_remap, train_ids, val_ids)


def _restrict_loss(loss, descriptors, ref_F, mask, remap, kept_ids):
    """Restrict a loss and its rows to one side of a held-out-system split.

    The point-wise weights are sliced, the energy term's row weights are
    sliced, its segment indices are renumbered onto the kept systems, and its
    target vector is sliced to the same systems, so the restricted term is the
    same objective over fewer systems rather than a differently normalized one.
    """
    idx = jnp.asarray(np.flatnonzero(np.asarray(mask)))
    desc = jnp.asarray(descriptors)[idx]
    ref = jnp.asarray(ref_F).reshape(-1)[idx]
    kwargs = {"weights": (None if loss.weights is None
                          else jnp.asarray(loss.weights)[idx])}
    if loss.energy_target is not None:
        kept = np.asarray(kept_ids, dtype=np.int64).reshape(-1)
        if kept.shape[0] == 0:
            raise ValueError(
                "_restrict_loss: the energy term cannot be restricted to no "
                "system (its mean over systems would be undefined).")
        kwargs.update(
            energy_row_weight=jnp.asarray(loss.energy_row_weight)[idx],
            energy_segment=jnp.asarray(remap, dtype=jnp.int32)[
                jnp.asarray(loss.energy_segment)[idx]],
            energy_target=jnp.asarray(loss.energy_target)[jnp.asarray(kept)],
            n_systems=int(kept.shape[0]),
            energy_weight=loss.energy_weight,
        )
    return _PretrainLoss(**kwargs), desc, ref


def _padded_segment(segment, n_mesh, n_systems):
    """Per-row system index with ``n_mesh`` sink rows appended: the row
    layout of a descriptor tensor after the mesh was concatenated onto it."""
    seg = np.asarray(segment, dtype=np.int32).reshape(-1)
    if n_mesh:
        seg = np.concatenate(
            [seg, np.full(int(n_mesh), int(n_systems), dtype=np.int32)])
    return seg


def _renormalize_mesh_share(weights, n_mesh, mesh_share):
    """Reset the flat mesh block at the end of ``weights`` so that it carries
    ``mesh_share`` of the total, by the expression the integration branch of
    :func:`run_pretrain` builds it with.

    The block is normalized against the atomic rows it trains beside. A
    restriction that drops atomic rows lowers that total, and an untouched
    block would then pull harder on a validated fit than on an unvalidated
    one; recomputing it on the rows kept keeps the regularizer's share the
    one the data was built at.
    """
    w = jnp.asarray(weights)
    n_mesh = int(n_mesh)
    atomic = w[:w.shape[0] - n_mesh]
    scale = float(mesh_share) / (1.0 - float(mesh_share))
    return jnp.concatenate(
        [atomic, jnp.full(n_mesh, float(jnp.sum(atomic)) * scale / n_mesh)])


def _mesh_loss_share(weights, n_mesh, n_rows):
    """The fraction of one channel's total loss weight the mesh block carries.

    Read back from the weight vector the loss was built with, so what is
    recorded is the share the fit felt rather than the share the data asked
    for. ``weights`` of ``None`` is the unweighted reduction, where every row
    counts once and the share is the row count's.
    """
    n_mesh, n_rows = int(n_mesh), int(n_rows)
    if not n_mesh:
        return 0.0
    if weights is None:
        return float(n_mesh) / float(n_rows)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(w.sum())
    return float(w[-n_mesh:].sum() / total) if total else float("nan")


def _validation_split(loss, descriptors, ref_F, segment, n_systems, held_out,
                      *, n_mesh=0, mesh_share=None):
    """The training and validation restrictions of one network's fit.

    Returns ``((loss, descriptors, ref_F), (loss, descriptors, ref_F))`` for
    the training side and the held-out side. ``segment`` is the padded
    per-row index (:func:`_padded_segment`); ``n_mesh`` mesh rows sit at the
    end of the row set and train always. When the loss is integration
    weighted and the mesh is present, the training side's mesh block is
    renormalized to ``mesh_share`` (:func:`_renormalize_mesh_share`); the
    validation side holds physical rows only. Under the unweighted reduction
    there is no weight to rescale -- every row counts once by definition, so
    the mesh's share there is a row count and moves with the split.
    """
    tm, vm, trm, vrm, tid, vid = _system_split_arrays(segment, n_systems,
                                                      held_out)
    train = _restrict_loss(loss, descriptors, ref_F, tm, trm, tid)
    val = _restrict_loss(loss, descriptors, ref_F, vm, vrm, vid)
    if n_mesh and mesh_share is not None and train[0].weights is not None:
        loss_tr = eqx.tree_at(
            lambda l: l.weights, train[0],
            _renormalize_mesh_share(train[0].weights, n_mesh, mesh_share))
        train = (loss_tr, train[1], train[2])
    return train, val


_PRETRAIN_MONITORS = ("pointwise", "loss")


def _train_pretrain_network(model, optimizer, loss_train, desc_train,
                            ref_train, loss_val, desc_val, ref_val, *,
                            n_steps, validate_every, patience, monitor,
                            progress_callback=None, checkpoint_path=None):
    """Full-batch pretraining with held-out-system validation and early stop.

    Returns ``(best_model, losses, record)``. The loop is written here rather
    than driven through ``xcquinox.train.xcTrainer`` because a stop criterion
    needs the optimizer STATE and the learning-rate schedule to survive across
    validations: ``xcTrainer`` initializes its optimizer state in its
    constructor and returns no state, so chunking a run through it would reset
    Adam's moments and restart the schedule at every validation. The
    unvalidated path still goes through ``xcTrainer`` unchanged, which is what
    keeps a run without validation byte-identical. On identical rows the two
    are the same arithmetic (one full-batch step per iteration on the same
    loss and optimizer chain) and reproduce each other's trajectory bit for
    bit.

    ``monitor`` names the quantity scored on the held-out rows every
    ``validate_every`` steps and at the last step: ``"loss"`` is the objective
    itself, the point-wise term plus the energy term at the run's weight, so
    the checkpoint kept is the one that generalizes on what was optimized;
    ``"pointwise"`` is the point-wise term alone, which is the same quantity
    when the energy term is off. A strict improvement resets the patience
    count; ``patience`` validations without one stop the run (``patience`` of
    0 disables the stop, and the loop then runs the full schedule). The model
    returned is the best one SEEN, never the last one, and the record carries
    the step it was seen at; when ``checkpoint_path`` is given the best model
    is written there at every improvement, so a job that dies mid-run leaves
    its best weights on disk. A non-finite score never counts as an
    improvement; a run whose every score is non-finite returns the initial
    model with ``best_step`` 0.
    """
    if monitor not in _PRETRAIN_MONITORS:
        raise ValueError(
            f"monitor must be one of {_PRETRAIN_MONITORS}, got {monitor!r}")
    n_steps, every, patience = int(n_steps), int(validate_every), int(patience)
    # The same bounds PretrainSpec.validate enforces, restated for a direct
    # caller: an empty schedule, a zero interval or a negative patience have
    # no meaning here.
    if n_steps <= 0 or every <= 0 or patience < 0:
        raise ValueError(
            f"n_steps must be > 0, validate_every > 0 and patience >= 0; got "
            f"{n_steps}, {every}, {patience}")
    w_e = float(loss_val.energy_weight) if monitor == "loss" else 0.0
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def _step(m, s, loss, desc, ref):
        value, grads = eqx.filter_value_and_grad(loss)(m, desc, ref)
        updates, s = optimizer.update(grads, s, m)
        return eqx.apply_updates(m, updates), s, value

    @eqx.filter_jit
    def _evaluate(m, loss, desc, ref):
        return loss.parts(m, desc, ref)

    losses, history = [], []
    best_value, best_step, best_model = float("inf"), 0, model
    stale, stopped_early = 0, False
    for step in range(1, n_steps + 1):
        model, opt_state, value = _step(model, opt_state, loss_train,
                                        desc_train, ref_train)
        losses.append(float(value))
        if progress_callback is not None:
            try:
                progress_callback(step, n_steps, losses[-1])
            except Exception:  # noqa: BLE001 - a logging callback never stops a fit
                pass
        if step % every and step != n_steps:
            continue
        pointwise, energy = _evaluate(model, loss_val, desc_val, ref_val)
        pointwise, energy = float(pointwise), float(energy)
        monitored = pointwise if w_e == 0.0 else pointwise + w_e * energy
        history.append((step, pointwise, energy, monitored))
        if monitored < best_value:
            best_value, best_step, best_model, stale = (monitored, step,
                                                        model, 0)
            if checkpoint_path is not None:
                eqx.tree_serialise_leaves(checkpoint_path, model)
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                stopped_early = True
        print(f"[pretrain] validation at step {step}/{n_steps}: train "
              f"{losses[-1]:.6e}, held-out pointwise {pointwise:.6e}, energy "
              f"{energy:.6e}, {monitor} {monitored:.6e}; best {best_value:.6e} "
              f"at step {best_step}"
              + (f"; no improvement for {stale} validation(s), stopping"
                 if stopped_early else ""), flush=True)
        if stopped_early:
            break
    record = {"monitor": monitor, "best_step": best_step,
              "best_value": best_value, "stopped_early": stopped_early,
              "steps_run": len(losses),
              "n_rows_train": int(jnp.asarray(desc_train).shape[0]),
              "n_rows_val": int(jnp.asarray(desc_val).shape[0]),
              "history": history}
    return best_model, losses, record


# Legacy lob_lim constants.
#
# The library-shaped skeletons constructed below are used ONLY to
# deserialize the on-disk byte stream into a pytree whose leaves we
# then graft onto the alec skeleton. Those skeletons therefore MUST
# receive the exact `lob_lim` values that the legacy checkpoints were
# written with, which are the hardcoded library defaults `1.804` (xnet,
# per `xcquinox/net.py:2049` `GGA_FxNet_extended.__init__` default) and
# `2.0` (cnet, per `xcquinox/net.py:2228` `GGA_FcNet_extended.__init__`
# default). Passing `arch.resolved_xnet_lob_lim` /
# `arch.resolved_cnet_lob_lim` here is wrong: when a
# `LiebOxfordBound` constraint is registered on `arch.x_constraints`
# AND `arch.double_lob_clamp_allowed=False`, the `resolved_xnet_lob_lim`
# property returns `None` so that
# `create_network_pair` can build an alec XNet that skips its built-in
# LOB wrap (the constraint becomes the sole clamp). But `None` is
# invalid for the LIBRARY class, `GGA_FxNet_extended(lob_lim=None, ...)`
# flows into `self.lobf = LOB(limit=None)` which crashes with TypeError
# the instant `LOB.__call__` dereferences `self.limit`, and the
# post-load `abs(loaded_lim - expected_lob_lim)` check would also crash
# with `TypeError: unsupported operand type(s) for -: 'float' and
# 'NoneType'`. The legacy checkpoints were all trained with the
# library defaults (the step3b notebook never configured
# `double_lob_clamp_allowed` because that field does not exist on the
# notebook's architecture), so hardcoding the legacy values is
# semantically correct, we are loading bytes that were written with
# these values, not configuring a fresh network. The alec-side
# `create_network_pair` still honors `arch.resolved_xnet_lob_lim` (which
# may be `None`); the library -> alec
# leaf graft is oblivious to `lob_lim` because `_AlecLOB.limit` is
# `eqx.field(static=True)` and therefore not in the pytree
# leaf stream.
_legacy_xnet_lob_lim: float = 1.804  # `xcquinox/net.py:2049` default
_legacy_cnet_lob_lim: float = 2.0    # `xcquinox/net.py:2228` default


# ---------------------------------------------------------------------------
# Pretrain data assembly helper
# ---------------------------------------------------------------------------

def _assemble_pretrain_descriptors(arch: ArchitectureConfig, pretrain_data: dict,
                                   *, for_cnet: bool = False,
                                   suffix: str = "_all") -> jnp.ndarray:
    """Assemble the (N, F) input array for pretraining from pretrain_data.

    Column order: [rho_all, sigma_all, *(per-descriptor columns)] where
    the per-descriptor columns follow ``arch.descriptors`` declaration
    order (matching `descriptors.assemble_descriptor_features`'s
    runtime contract). Each descriptor's columns are pulled from a
    pretrain_data key derived by stripping ``_statistics`` and
    appending ``_all`` (e.g. ``dm_statistics`` -> ``dm_all``,
    ``cusp`` -> ``cusp_all``).

    When ``for_cnet`` and ``arch.use_polarized_correlation``, a spin
    polarization column ``zeta_all`` is inserted at index 2 (right after
    sigma, BEFORE the descriptor extras) to match the polarized cnet's
    expected ``[rho, sigma, zeta, *extras]`` input layout (see
    ``AlecGGA_CNet.__call__``). The xnet input (``for_cnet=False``) NEVER
    carries zeta, exchange is zeta-independent (Oliver & Perdew, PRA 20,
    397 (1979)). If ``zeta_all`` is absent, zeta defaults to zeros: a valid
    unpolarized warm-start (zeta=0 -> x1=1, recovering the unpolarized cnet
    input and Fc target), to be refined by zeta-resolved training data.

    Raises KeyError if any declared descriptor's pretrain key is
    absent from pretrain_data, there is NO zero-array fallback.

    ``suffix`` selects the row block. ``"_all"`` (default) is the
    total-density block, which carries the correlation rows always and the
    exchange rows under the historical footing. ``"_x"`` is the per-channel
    exchange block a file built on the exact-spin-scaling footing carries; the
    correlation network never reads it, because correlation is
    spin-interpolated rather than spin-scaled (von Barth and Hedin, J. Phys. C
    5, 1629 (1972); Perdew and Wang, Phys. Rev. B 45, 13244 (1992)) and stays
    on the total density with zeta.
    """
    from xcquinox.alec.descriptors import make_descriptor
    if for_cnet and suffix != "_all":
        raise ValueError(
            "the correlation network is posed on the total density, so its "
            f"rows are the '_all' block; got suffix={suffix!r}."
        )
    cols = [pretrain_data["rho" + suffix], pretrain_data["sigma" + suffix]]
    if for_cnet and arch.use_polarized_correlation:
        zeta_all = pretrain_data.get("zeta_all")
        if zeta_all is None:
            zeta_all = jnp.zeros_like(pretrain_data["rho_all"])
        cols.append(zeta_all)
    # Map descriptor.name -> the pretrain_data column STEM; the block suffix
    # is appended, so one map serves both row blocks.
    _key_map = {"dm_statistics": "dm", "cusp": "cusp", "rung35": "rung35",
                "rung35_multishell": "rung35ms", "metagga": "metagga"}
    for spec in arch.descriptors:
        stem = _key_map.get(spec.name)
        if stem is None:
            raise KeyError(
                f"_assemble_pretrain_descriptors: no pretrain_data key "
                f"mapping registered for descriptor {spec.name!r}; update "
                f"_key_map in pretrain.py"
            )
        key = stem + suffix
        arr = pretrain_data[key]
        # Width gate. A stale .npz written before a descriptor's feature count
        # changed silently widens the network input instead of failing: a
        # 3-column dm_all against the 2-feature dm_statistics (width dropped
        # 2026-08-06 with the removal of dm_entropy) produced a 6-wide input
        # where n_input_features was 5, which trains without complaint and is
        # wrong. Fail loudly and name the regeneration.
        n_cols = 1 if arr.ndim == 1 else arr.shape[1]
        expected = make_descriptor(spec.name, **spec.as_kwargs()).n_features
        if n_cols != expected:
            raise ValueError(
                f"_assemble_pretrain_descriptors: pretrain column {key!r} has "
                f"{n_cols} column(s) but descriptor {spec.name!r} declares "
                f"n_features={expected}. The pretrain .npz predates a change to "
                f"this descriptor's width; regenerate it with "
                f"pretrain_data_gen rather than training against a mismatched "
                f"input layout."
            )
        if arr.ndim == 1:
            cols.append(arr)
        else:
            for i in range(arr.shape[1]):
                cols.append(arr[:, i])
    return jnp.stack(cols, axis=1)


# ---------------------------------------------------------------------------
# Optimizer builder
# ---------------------------------------------------------------------------

def _append_pretrain_mesh(arch, pretrain_data, descriptors, descriptors_c,
                          fx_target, fc_target):
    """Append the ``(s, alpha)`` mesh rows to the pretrain inputs and targets.

    Returns ``(descriptors, descriptors_c, Fx, Fc)`` with the mesh
    concatenated onto each. The mesh rows are laid out in the SAME column
    order :func:`_assemble_pretrain_descriptors` produces
    (``[rho, sigma, (zeta,) *extras]``) so the two blocks are one array as
    far as the network is concerned. The mesh rows' LOSS weights are NOT
    handled here: under ``integration`` weighting the caller appends a flat
    per-channel block normalized to ``MESH_WEIGHT_FRACTION`` of each
    channel's total AFTER the ``|rho*eps_LDA|`` factor -- pushing the mesh's
    synthesized densities through that factor was measured to hand the mesh
    ~0.99997 of the loss.

    Only called for an arch whose descriptor set is exactly ``(metagga,)``; see
    the gate in :func:`run_pretrain` for why.
    """
    n = pretrain_data["rho_mesh"].shape[0]
    cols = [jnp.asarray(pretrain_data["rho_mesh"]),
            jnp.asarray(pretrain_data["sigma_mesh"])]
    cols_c = list(cols)
    if arch.use_polarized_correlation:
        zeta_mesh = pretrain_data.get("zeta_mesh")
        cols_c.append(jnp.zeros(n) if zeta_mesh is None
                      else jnp.asarray(zeta_mesh))
    alpha = jnp.asarray(pretrain_data["metagga_mesh"]).reshape(-1)
    cols.append(alpha)
    cols_c.append(alpha)
    mesh_x = jnp.stack(cols, axis=1)
    mesh_c = jnp.stack(cols_c, axis=1)
    if mesh_x.shape[1] != descriptors.shape[1]:
        raise ValueError(
            f"mesh X-net column count {mesh_x.shape[1]} != atomic "
            f"{descriptors.shape[1]}; the mesh layout has drifted from "
            "_assemble_pretrain_descriptors")
    if mesh_c.shape[1] != descriptors_c.shape[1]:
        raise ValueError(
            f"mesh C-net column count {mesh_c.shape[1]} != atomic "
            f"{descriptors_c.shape[1]}; the mesh layout has drifted from "
            "_assemble_pretrain_descriptors")
    return (
        jnp.concatenate([descriptors, mesh_x], axis=0),
        jnp.concatenate([descriptors_c, mesh_c], axis=0),
        jnp.concatenate([jnp.asarray(fx_target).reshape(-1),
                         jnp.asarray(pretrain_data["Fx_scan_mesh"]).reshape(-1)]),
        jnp.concatenate([jnp.asarray(fc_target).reshape(-1),
                         jnp.asarray(pretrain_data["Fc_scan_mesh"]).reshape(-1)]),
    )


def _build_optimizer(
    *,
    lr_start: float,
    lr_end: float,
    n_steps: int,
    lr_decay_start: float,
    grad_clip: float,
) -> optax.GradientTransformation:
    """Build canonical optimizer chain for pretraining.

    Chain order: clip_by_global_norm -> adam(lr_schedule).
    LR schedule: optional constant warmup then linear decay.
    """
    decay_start_step = int(lr_decay_start * n_steps)
    decay_steps = n_steps - decay_start_step

    if lr_decay_start > 0 and decay_steps > 0:
        lr_schedule = optax.join_schedules(
            schedules=[
                optax.constant_schedule(lr_start),
                optax.linear_schedule(
                    init_value=lr_start,
                    end_value=lr_end,
                    transition_steps=decay_steps,
                ),
            ],
            boundaries=[decay_start_step],
        )
    else:
        lr_schedule = optax.linear_schedule(
            init_value=lr_start,
            end_value=lr_end,
            transition_steps=n_steps,
        )

    return optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(learning_rate=lr_schedule),
    )


# ---------------------------------------------------------------------------
# run_pretrain
# ---------------------------------------------------------------------------

def _pretrain_data_filename(arch, parent_density="pbe") -> str:
    """Pretrain-data filename for an architecture and its parent density.

    A spin-polarization-aware arch uses the zeta-aware
    ``pretrain_data_polarized.npz`` (carrying a per-grid-point ``zeta_all``
    column); the default unpolarized arch uses ``pretrain_data.npz``; a parent
    other than PBE carries the parent's suffix (``_scan``), because that file is
    built on a different self-consistent density. The name itself comes from
    ``pretrain_data_gen.pretrain_data_filename``, the single naming function
    the generator and the datagen stage write through, so the two ends of the
    hand-off cannot drift, and the parent qualifier is the one
    ``resolve_parent_density`` gives the architecture (``"auto"`` is the rung
    baseline), so a meta-GGA run opens the SCAN-density file the datagen stage
    wrote for it rather than the PBE name. Pure so it can be unit-tested
    without touching disk."""
    from xcquinox.alec.pretrain_data_gen import (
        pretrain_data_filename, resolve_parent_density)
    return pretrain_data_filename(
        bool(getattr(arch, "use_polarized_correlation", False)),
        reference_xc=resolve_parent_density(arch, parent_density))


def run_pretrain(spec: PretrainSpec, progress_callback=None, *, networks=None) -> dict:
    """Pretrain xnet and cnet on synthetic grid data.

    Steps:
    1. spec.validate()
    2. Load pretrain_data.npz (or .pkl fallback) from spec.data_dir
    3. Assemble descriptor tensor
    4. Create network pair
    5. Train xnet on Fx_target, then cnet on Fc_target (both MSE)
    6. Build optimizer (optax chain)
    7. Save artifacts under spec.checkpoint_dir
    8. Return metadata dict

    Constraint awareness: the networks built by ``create_network_pair`` enforce
    ``spec.arch``'s physical constraints INTRINSICALLY in their forward pass, so
    the MSE here fits the CONSTRAINED enhancement to the PBE/LDA targets, the
    same constrained functional that training and evaluation use. (Constraints
    are static, so the saved ``xnet.eqx``/``cnet.eqx`` leaf streams are unchanged
    and remain compatible with existing checkpoints.)

    Args:
        networks: optional ``(xnet, cnet)`` pair to pretrain INSTEAD of building
            the pair from ``spec.arch`` via ``create_network_pair``. Use this to
            pretrain a network the arch cannot express, e.g. a truly-unconstrained
            net (``lob_lim=None``, no constraints), which ``create_network_pair``
            cannot produce (a None lob_lim there requires the LO constraint to be
            active). The provided networks MUST carry whatever constraints they are
            meant to enforce; ``spec.arch`` is still used for the pretrain-data file
            selection, descriptor assembly, and metadata. Default ``None`` =>
            byte-identical to the prior behavior (build from ``spec.arch``).

    Returns:
        dict with pretrain_metadata.json fields.
    """
    spec.validate()

    # --- Load pretrain data (npz; .pkl legacy fallback for the unpolarized file) ---
    # A spin-polarized run uses the zeta-aware ``pretrain_data_polarized.npz``;
    # an unpolarized run uses ``pretrain_data.npz``. Selected purely from the arch
    # flag, so the cluster picks the right file automatically.
    polarized = bool(getattr(spec.arch, "use_polarized_correlation", False))
    # The parent density is resolved BEFORE the file is named: a meta-GGA
    # architecture under the rung baseline pretrains on the SCAN-density file,
    # which carries the parent's suffix, and naming the PBE file first would
    # open the wrong density (or nothing) for every non-PBE parent.
    from xcquinox.alec.pretrain_data_gen import resolve_parent_density
    parent_density = getattr(spec, "parent_density", "pbe")
    want_reference = resolve_parent_density(spec.arch, parent_density)
    npz_path = os.path.join(
        spec.data_dir, _pretrain_data_filename(spec.arch, parent_density))
    pkl_path = os.path.join(spec.data_dir, "pretrain_data.pkl")
    if want_reference != "pbe" and not os.path.isfile(npz_path):
        pbe_path = os.path.join(spec.data_dir, _pretrain_data_filename(spec.arch))
        raise ValueError(
            f"run_pretrain: architecture {spec.arch.name!r} resolves to the "
            f"{want_reference!r} parent density, but {npz_path!r} does not exist"
            + (f" (only the PBE-density file {pbe_path!r} does)"
               if os.path.isfile(pbe_path) else "")
            + f". Generate the {want_reference} file with "
            "pretrain_data_gen.ensure_pretrain_data(..., reference_xc="
            f"{want_reference!r}) or set pretrain.parent_density explicitly.")

    if os.path.isfile(npz_path):
        raw = np.load(npz_path)
        pretrain_data_np = {k: np.array(raw[k]) for k in raw.files}
    elif not polarized and os.path.isfile(pkl_path):
        # pkl fallback for legacy (unpolarized) files, safe because only array data
        with open(pkl_path, "rb") as _f:
            pretrain_data_np = pickle.loads(_f.read())  # noqa: S301 -- trusted local data
    elif polarized:
        # Fail fast: a spin-polarized run MUST use the zeta-aware file (never a
        # silent zeta=0 fallback that would defeat the purpose).
        raise FileNotFoundError(
            f"run_pretrain: spin-polarized run expects "
            f"{_pretrain_data_filename(spec.arch, parent_density)!r} in "
            f"data_dir={spec.data_dir!r} "
            f"(it carries the zeta_all column). Generate it with `python "
            f"scripts/generate_polarized_pretrain_data.py --out-dir {spec.data_dir!r}`."
        )
    else:
        raise FileNotFoundError(
            f"run_pretrain: neither pretrain_data.npz nor pretrain_data.pkl "
            f"found in data_dir={spec.data_dir!r}"
        )

    # Lift every array into JAX
    pretrain_data = {k: jnp.array(v) for k, v in pretrain_data_np.items()}

    # Which row block the exchange network reads. A file built on the
    # exact-spin-scaling footing carries the open-shell exchange rows
    # separately, because the per-channel rows of an open shell are not its
    # total-density rows; a file built on the historical footing has one block
    # and the xnet reads it, byte-identically.
    x_suffix = "_x" if "rho_x" in pretrain_data else "_all"
    # The parent whose SELF-CONSISTENT density this architecture must pretrain
    # on. A meta-GGA network fit on a PBE density is fit to a density its SCF
    # never sees.
    from xcquinox.alec.pretrain_data_gen import read_pretrain_manifest
    _manifest = read_pretrain_manifest(npz_path)
    file_reference = str((_manifest or {}).get("reference_xc", "pbe"))
    if _manifest is not None and file_reference != want_reference:
        raise ValueError(
            f"run_pretrain: architecture {spec.arch.name!r} resolves to the "
            f"{want_reference!r} parent density, but {npz_path!r} was built on "
            f"the {file_reference!r} density. Point data_dir at the "
            f"{want_reference} file or set pretrain.parent_density explicitly."
        )
    # The footing the file DECLARES must be one its blocks can serve. A
    # manifest claiming the per-channel footing beside a file with no exchange
    # block would otherwise be pretrained at the historical footing in
    # silence, undoing the Section 3.2 correction without a trace.
    manifest_footing = str((_manifest or {}).get("exchange_footing", "total"))
    if manifest_footing == "spin_channel" and x_suffix != "_x":
        raise ValueError(
            f"run_pretrain: {npz_path!r} declares the 'spin_channel' exchange "
            "footing but carries no per-channel exchange block (no 'rho_x'). "
            "Regenerate it with pretrain_data_gen.ensure_pretrain_data."
        )
    energy_weight = float(getattr(spec, "energy_term_weight", 0.0))
    if energy_weight > 0.0:
        # Named one by one rather than probed on 'system_all' alone: a file
        # carrying the row index but not the per-system table, the LDA column
        # or the block's own quadrature weights would otherwise fail with a
        # bare KeyError deep in the loss assembly.
        _scan_rung = bool(getattr(spec.arch, "meta_gga", False))
        _needed = ("system_all", "weights_all", "e_lda_c_all",
                   "system" + x_suffix, "weights" + x_suffix,
                   "e_lda_x" + x_suffix,
                   "e_x_parent_scan_sys" if _scan_rung else "e_x_parent_sys",
                   "e_c_parent_scan_sys" if _scan_rung else "e_c_parent_sys")
        _missing = [k for k in _needed if k not in pretrain_data]
        if _missing:
            raise ValueError(
                "run_pretrain: pretrain.energy_term_weight > 0 needs the "
                "per-row system index 'system_all' and the per-system energy "
                f"table; {npz_path!r} is missing {_missing}. Regenerate it "
                "with pretrain_data_gen.ensure_pretrain_data."
            )

    # --- Assemble descriptor tensors ---
    # The xnet input is zeta-blind; the cnet input carries the zeta column
    # when the architecture uses polarized correlation. They are
    # identical for the unpolarized (default) architecture.
    # At the historical footing the call is the historical one, so a wrapper
    # installed on the pre-protocol seam ``(arch, data, for_cnet)`` keeps
    # serving the total-density block unchanged.
    if x_suffix == "_all":
        descriptors = _assemble_pretrain_descriptors(spec.arch, pretrain_data)
    else:
        descriptors = _assemble_pretrain_descriptors(spec.arch, pretrain_data,
                                                     suffix=x_suffix)
    descriptors_c = _assemble_pretrain_descriptors(
        spec.arch, pretrain_data, for_cnet=True)

    # Targets are stored as (F - 1), not F; PretrainLoss subtracts 1 from
    # network output (networks return 1 + enhancement).
    # DFS-faithful meta_gga archs pretrain to SCAN (a GGA structurally cannot fit
    # SCAN's alpha-dependence); GGA archs keep the PBE targets. The SCAN columns are
    # always present (pretrain_data_gen writes them + the staleness guard regens).
    if bool(getattr(spec.arch, "meta_gga", False)):
        Fx_target = pretrain_data["Fx_scan" + x_suffix]
        Fc_target = pretrain_data["Fc_scan_all"]
        e_x_key, e_c_key = "e_x_parent_scan_sys", "e_c_parent_scan_sys"
    else:
        Fx_target = pretrain_data["Fx" + x_suffix]
        Fc_target = pretrain_data["Fc_all"]
        e_x_key, e_c_key = "e_x_parent_sys", "e_c_parent_sys"

    # (s, alpha) parameter mesh -- appended ONLY for a meta-GGA arch, and only
    # when the mesh can define every descriptor that arch consumes.
    #
    # WHY: SCAN's F_c is 3-D in (r_s, s, alpha) and the atomic grids leave the
    # alpha axis underdetermined -- the meta-GGA C-net was measured at up to
    # 0.457 from SCAN away from alpha=1, where the GGA C-net (same atoms, same
    # weighting, same optimizer) sits within 0.013 of PBE. The mesh determines
    # that axis directly.
    #
    # WHY THE DESCRIPTOR GATE: a mesh node is a synthetic (rho, sigma, tau)
    # triple with NO geometry, so it cannot define cusp or rung-3.5 occupancy.
    # Appending it for an arch that consumes those would teach the net that
    # fabricated descriptor values pair with real SCAN targets. Archs whose
    # descriptor set is exactly (metagga,) -- deep_mgga_3x16,
    # deep_mgga_attn_3x16 -- take the mesh; deep_rung35_mgga_3x16 does NOT and
    # keeps the atoms-only seed until the mesh can carry its extras.
    mesh_used = False
    # The share the DATA was built at, read once for the banner, the
    # integration weighting and the validation split. A file written before
    # the share became configurable carries no such key and falls back to the
    # constant it was built with.
    mesh_share = None
    if bool(getattr(spec.arch, "meta_gga", False)):
        desc_names = tuple(getattr(d, "name", None) for d in spec.arch.descriptors)
        has_mesh = "Fx_scan_mesh" in pretrain_data
        if desc_names == ("metagga",) and has_mesh:
            (descriptors, descriptors_c, Fx_target,
             Fc_target) = _append_pretrain_mesh(
                spec.arch, pretrain_data, descriptors, descriptors_c,
                Fx_target, Fc_target)
            mesh_used = True
            from xcquinox.alec.pretrain_data_gen import MESH_WEIGHT_FRACTION
            mesh_share = float(pretrain_data_np.get("mesh_weight_fraction",
                                                    MESH_WEIGHT_FRACTION))
            print(f"[pretrain] (s, alpha) mesh appended: "
                  f"{pretrain_data['rho_mesh'].shape[0]} nodes "
                  f"({100.0 * mesh_share:.0f}% effective "
                  "loss-weight share per channel, by construction)",
                  flush=True)
        elif not has_mesh:
            print("[pretrain] WARNING: pretrain data carries no (s, alpha) mesh; "
                  "the meta-GGA alpha axis will be underdetermined (regenerate "
                  "the pretrain data to add it).", flush=True)
        else:
            print(f"[pretrain] NOTE: {getattr(spec.arch, 'name', '?')} consumes "
                  f"descriptors {desc_names}, which a geometry-free mesh node "
                  "cannot define; pretraining on the atomic grids alone.",
                  flush=True)
    # --- Create network pair (or use the caller-supplied override) ---
    if networks is not None:
        xnet, cnet = networks
    else:
        xnet, cnet = create_network_pair(spec.arch, seed=spec.seed)

    # --- PretrainLoss (scalar MSE, compatible with xcTrainer) ---
    #
    # Loss weighting is controlled by `spec.loss_weighting`:
    #   "unweighted" (default): plain mean of squared residuals.
    #   "integration":          integration-weighted sum-of-squared residuals,
    #                           with per-component weights |rho * eps^LDA|
    #                           computed once from pretrain_data["rho_all"]
    #                           (see _compute_integration_weights).
    #
    # The xnet and cnet are trained in separate trainer calls using different
    # weight arrays (w_x vs w_c). Each is baked into its own closure-like
    # eqx.Module instance so that xcTrainer can call `loss(model, descriptors,
    # ref_F)` with the familiar 3-arg API. Weights are carried as a static
    # array field on the module (eqx handles the pytree correctly).
    n_mesh_rows = int(pretrain_data["rho_mesh"].shape[0]) if mesh_used else 0
    energy_kwargs_x = {}
    energy_kwargs_c = {}
    if energy_weight > 0.0:
        _rw, _seg, _tgt, _ns = _energy_term_inputs(
            pretrain_data, weight_key="weights" + x_suffix,
            lda_key="e_lda_x" + x_suffix, segment_key="system" + x_suffix,
            target_key=e_x_key, n_mesh=n_mesh_rows)
        energy_kwargs_x = dict(energy_row_weight=_rw, energy_segment=_seg,
                               energy_target=_tgt, n_systems=_ns,
                               energy_weight=energy_weight)
        _rw, _seg, _tgt, _ns = _energy_term_inputs(
            pretrain_data, weight_key="weights_all", lda_key="e_lda_c_all",
            segment_key="system_all", target_key=e_c_key, n_mesh=n_mesh_rows)
        energy_kwargs_c = dict(energy_row_weight=_rw, energy_segment=_seg,
                               energy_target=_tgt, n_systems=_ns,
                               energy_weight=energy_weight)
    n_systems = int(pretrain_data[e_x_key].shape[0]) \
        if e_x_key in pretrain_data else 0

    integration_weights_complete: bool | None = None  # set below for "integration" mode
    if spec.loss_weighting == "integration":
        # The atomic rows carry the physical importance measure; the mesh
        # rows (when appended) get a FLAT per-channel weight added AFTER the
        # |rho*eps_LDA| factor, below.
        rho_all = pretrain_data["rho_all"]
        # Becke-Lebedev quadrature weights ``dr_i`` improve energy-density
        # calibration; older pretrain_data files don't carry them, fall back
        # with a warning and record the degradation in metadata.
        grid_weights = pretrain_data.get("weights_all")
        # The flag covers BOTH weight vectors the run builds. The exchange
        # block has its own quadrature column under the per-channel footing,
        # so a file carrying 'weights_all' alone still leaves the exchange
        # loss without the dr_i measure.
        grid_weights_x = pretrain_data.get("weights" + x_suffix)
        if grid_weights is None or grid_weights_x is None:
            integration_weights_complete = False
            import warnings as _warn
            _absent = [k for k, v in (("weights_all", grid_weights),
                                      ("weights" + x_suffix, grid_weights_x))
                       if v is None]
            _msg = (
                f"pretrain_data.npz lacks {' and '.join(repr(k) for k in _absent)}"
                "; integration-mode loss is missing Becke quadrature weights "
                "on that block and approximates a |rho*eps_LDA|-weighted mean "
                "rather than the integrated XC-energy residual. Regenerate "
                "pretrain_data.npz from a post-2026-04-27 notebook generator "
                "to get correct weights."
            )
            _warn.warn(_msg, RuntimeWarning, stacklevel=2)
            # RuntimeWarnings are easy to miss in a SLURM .out log; also emit a
            # flushed banner so the degradation is unmissable there.
            print(f"\n{'!' * 72}\n[PRETRAIN WARNING] {_msg}\n{'!' * 72}\n",
                  flush=True)
        else:
            integration_weights_complete = True
        w_x, _unused = _compute_integration_weights(
            pretrain_data["rho" + x_suffix], grid_weights_x)
        _unused, w_c = _compute_integration_weights(rho_all, grid_weights)
        if mesh_used:
            # The |rho*eps_LDA| factor is a grid-importance measure for
            # PHYSICAL densities. A mesh node carries no quadrature measure,
            # and pushing its synthesized rho (up to ~2.4e2 a.u. at
            # r_s = 0.1) through that factor hands the mesh ~0.99997 of the
            # loss weight (measured on H+O data), burying the atomic rows at
            # ~3e-5 -- the training then fits the synthetic mesh and forgets
            # the physical densities. Integration weights are therefore
            # computed on the ATOMIC block alone, and each channel's mesh
            # block gets a FLAT weight normalized so the mesh's share of
            # that channel's total loss weight is exactly the share the data
            # was built at (``mesh_share``), by construction.
            n_mesh = int(pretrain_data["rho_mesh"].shape[0])
            scale = mesh_share / (1.0 - mesh_share)
            w_x = jnp.concatenate(
                [w_x, jnp.full(n_mesh, float(jnp.sum(w_x)) * scale / n_mesh)])
            w_c = jnp.concatenate(
                [w_c, jnp.full(n_mesh, float(jnp.sum(w_c)) * scale / n_mesh)])
        loss_fn_x = _PretrainLoss(weights=w_x, **energy_kwargs_x)
        loss_fn_c = _PretrainLoss(weights=w_c, **energy_kwargs_c)
    else:  # "unweighted": validated at construction
        loss_fn_x = _PretrainLoss(**energy_kwargs_x)
        loss_fn_c = _PretrainLoss(**energy_kwargs_c)

    checkpoint_dir = spec.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Helper: build phase progress callbacks for xcTrainer 3-arg form
    def _x_callback(step, total, loss):
        if progress_callback is not None:
            progress_callback({
                "arch": spec.arch.name,
                "phase": "X",
                "step": step,
                "total": total,
                "loss": float(loss),
                "timestamp": time.time(),
            })

    def _c_callback(step, total, loss):
        if progress_callback is not None:
            progress_callback({
                "arch": spec.arch.name,
                "phase": "C",
                "step": step,
                "total": total,
                "loss": float(loss),
                "timestamp": time.time(),
            })

    # Per-network checkpoint subdirs. xcTrainer serialises its periodic
    # best-loss snapshots as ``<checkpoint_dir>/xc.eqx.<step>``: if both the
    # xnet and cnet trainers share one checkpoint_dir they clobber each other's
    # snapshots. Give each its own subdir; the FINAL xnet.eqx/cnet.eqx still
    # land at the top level (what downstream consumes).
    xnet_ckpt_dir = os.path.join(checkpoint_dir, "xnet")
    cnet_ckpt_dir = os.path.join(checkpoint_dir, "cnet")
    os.makedirs(xnet_ckpt_dir, exist_ok=True)
    os.makedirs(cnet_ckpt_dir, exist_ok=True)
    xnet_path = os.path.join(checkpoint_dir, "xnet.eqx")
    cnet_path = os.path.join(checkpoint_dir, "cnet.eqx")

    # --- Held-out-system validation split ---------------------------------
    # A fraction of the MOLECULES is withheld from the fit and scored between
    # optimizer steps; training stops when the monitored quantity has not
    # improved for `patience` validations, and the weights kept are the best
    # ones seen. What is monitored is the objective itself on the held-out
    # rows -- the point-wise term plus the energy term at the run's weight --
    # so the checkpoint kept is the one that generalizes on what was
    # optimized. A fraction of 0 (the default) reproduces the unvalidated
    # schedule exactly, through the same xcTrainer call as before. The
    # fallbacks are the PretrainSpec defaults themselves, read off the class,
    # for a spec object built before the protocol fields existed.
    val_fraction = float(getattr(spec, "validation_fraction",
                                 PretrainSpec.validation_fraction))
    val_seed = int(getattr(spec, "validation_seed",
                           PretrainSpec.validation_seed))
    validate_every = int(getattr(spec, "validate_every",
                                 PretrainSpec.validate_every))
    patience = int(getattr(spec, "patience", PretrainSpec.patience))
    held_out = ()
    n_split = 0
    if val_fraction > 0.0:
        # A file written before the system table exists cannot say which rows
        # belong to which molecule; the request is refused by name rather than
        # silently trained without a split.
        if ("system_natoms" not in pretrain_data
                or "system" + x_suffix not in pretrain_data
                or "system_all" not in pretrain_data):
            raise ValueError(
                "run_pretrain: pretrain.validation_fraction > 0 needs the "
                "system table 'system_natoms' and the per-row system index "
                f"'system{x_suffix}' / 'system_all', which {npz_path!r} "
                "predates. Regenerate it with "
                "pretrain_data_gen.ensure_pretrain_data."
            )
        natoms = np.asarray(pretrain_data_np["system_natoms"]).reshape(-1)
        n_split = int(natoms.shape[0])
        if n_systems and n_split != n_systems:
            # The split renumbers the energy term's segment array through a
            # table of its own length. JAX CLAMPS an out-of-range index rather
            # than raising, so a disagreement here would silently fold one
            # system's rows onto another's energy.
            raise ValueError(
                f"run_pretrain: {npz_path!r} lists {n_split} systems in "
                f"'system_natoms' but {n_systems} per-system energies; the "
                "held-out split and the energy term would index different "
                "tables.")
        held_out = _validation_systems(natoms, val_fraction, val_seed)
    monitor = "loss" if energy_weight > 0.0 else "pointwise"
    system_names = [str(row[0]) for row in
                    ((_manifest or {}).get("systems") or [])]
    validation_record = {
        "fraction": val_fraction, "seed": val_seed,
        "validate_every": validate_every, "patience": patience,
        "monitor": monitor, "active": bool(held_out),
        "systems": [system_names[i] if i < len(system_names) else f"sys{i}"
                    for i in held_out],
    }
    if held_out:
        print(f"[pretrain] validation: holding out {len(held_out)} of "
              f"{int(np.count_nonzero(natoms > 1))} molecules (fraction "
              f"{val_fraction}, seed {val_seed}): "
              f"{', '.join(validation_record['systems'])}; scored every "
              f"{validate_every} step(s) on the {monitor}, patience "
              f"{patience}", flush=True)
    elif val_fraction > 0.0:
        print(f"[pretrain] NOTE: validation_fraction {val_fraction} requested "
              "but the set carries fewer than two molecules; nothing is held "
              "out and the full schedule runs unvalidated.", flush=True)

    # --- Train xnet ---
    t0 = time.time()
    optimizer_x = _build_optimizer(
        lr_start=spec.lr_start,
        lr_end=spec.lr_end,
        n_steps=spec.n_steps,
        lr_decay_start=spec.lr_decay_start,
        grad_clip=spec.grad_clip,
    )
    if held_out:
        seg_x = _padded_segment(pretrain_data["system" + x_suffix],
                                n_mesh_rows, n_split)
        (lx_tr, dx_tr, fx_tr), (lx_va, dx_va, fx_va) = _validation_split(
            loss_fn_x, descriptors, Fx_target, seg_x, n_split, held_out,
            n_mesh=n_mesh_rows, mesh_share=mesh_share)
        xnet_trained, losses_x, record_x = _train_pretrain_network(
            xnet, optimizer_x, lx_tr, dx_tr, fx_tr, lx_va, dx_va, fx_va,
            n_steps=spec.n_steps, validate_every=validate_every,
            patience=patience, monitor=monitor,
            progress_callback=_x_callback,
            checkpoint_path=os.path.join(xnet_ckpt_dir, "xc.eqx.best"))
        validation_record["x"] = record_x
    else:
        trainer_x = xcquinox.train.xcTrainer(
            model=xnet,
            optim=optimizer_x,
            loss=loss_fn_x,
            steps=spec.n_steps,
            do_jit=True,
            serialize_every=max(50, spec.n_steps // 10),
            checkpoint_dir=xnet_ckpt_dir,
            progress_callback=_x_callback,
        )
        xnet_trained, losses_x = trainer_x(1, [descriptors], [Fx_target])
    # Persist the final xnet immediately, BEFORE cnet training starts, so a
    # job that dies or times out during the (separate) cnet phase does not lose
    # the already-completed xnet result.
    eqx.tree_serialise_leaves(xnet_path, xnet_trained)

    # --- Train cnet (fresh optimizer) ---
    optimizer_c = _build_optimizer(
        lr_start=spec.lr_start,
        lr_end=spec.lr_end,
        n_steps=spec.n_steps,
        lr_decay_start=spec.lr_decay_start,
        grad_clip=spec.grad_clip,
    )
    if held_out:
        seg_c = _padded_segment(pretrain_data["system_all"], n_mesh_rows,
                                n_split)
        (lc_tr, dc_tr, fc_tr), (lc_va, dc_va, fc_va) = _validation_split(
            loss_fn_c, descriptors_c, Fc_target, seg_c, n_split, held_out,
            n_mesh=n_mesh_rows, mesh_share=mesh_share)
        cnet_trained, losses_c, record_c = _train_pretrain_network(
            cnet, optimizer_c, lc_tr, dc_tr, fc_tr, lc_va, dc_va, fc_va,
            n_steps=spec.n_steps, validate_every=validate_every,
            patience=patience, monitor=monitor,
            progress_callback=_c_callback,
            checkpoint_path=os.path.join(cnet_ckpt_dir, "xc.eqx.best"))
        validation_record["c"] = record_c
    else:
        trainer_c = xcquinox.train.xcTrainer(
            model=cnet,
            optim=optimizer_c,
            loss=loss_fn_c,
            steps=spec.n_steps,
            do_jit=True,
            serialize_every=max(50, spec.n_steps // 10),
            checkpoint_dir=cnet_ckpt_dir,
            progress_callback=_c_callback,
        )
        cnet_trained, losses_c = trainer_c(1, [descriptors_c], [Fc_target])
    duration = time.time() - t0

    # --- Save artifacts ---
    eqx.tree_serialise_leaves(cnet_path, cnet_trained)

    losses_x_np = np.array(losses_x, dtype=np.float64)
    losses_c_np = np.array(losses_c, dtype=np.float64)
    np.save(os.path.join(checkpoint_dir, "losses_x.npy"), losses_x_np)
    np.save(os.path.join(checkpoint_dir, "losses_c.npy"), losses_c_np)

    use_cusp = any(s.name == "cusp" for s in spec.arch.descriptors)
    use_dm = any(s.name == "dm_statistics" for s in spec.arch.descriptors)

    metadata = {
        "arch_name": spec.arch.name,
        "depth": spec.arch.depth,
        "nodes": spec.arch.nodes,
        "pretrain_steps": spec.n_steps,
        "lr_start": spec.lr_start,
        "lr_end": spec.lr_end,
        "lr_decay_start": spec.lr_decay_start,
        "grad_clip": spec.grad_clip,
        "loss_weighting": spec.loss_weighting,
        "final_loss_x": float(losses_x_np[-1]) if len(losses_x_np) > 0 else float("nan"),
        "final_loss_c": float(losses_c_np[-1]) if len(losses_c_np) > 0 else float("nan"),
        "min_loss_x": float(np.min(losses_x_np)) if len(losses_x_np) > 0 else float("nan"),
        "min_loss_c": float(np.min(losses_c_np)) if len(losses_c_np) > 0 else float("nan"),
        "use_cusp": use_cusp,
        "use_dm": use_dm,
        # Shape-changing flag: polarized cnet input width +1.
        "use_polarized_correlation": bool(spec.arch.use_polarized_correlation),
        # Architecture-shape keys the run validator cross-checks
        # (validate_run.py); the step count is already recorded as
        # "pretrain_steps" above. Absent from files written before
        # 2026-08-06, which is why the validator treats their absence as a
        # legacy warning rather than a failure.
        "meta_gga": bool(spec.arch.meta_gga),
        "n_extra_features": int(spec.arch.n_extra_features),
        # Whether the (s, alpha) parameter mesh was appended to this
        # pretrain's inputs -- checkpoint provenance the run validator
        # cross-checks (a meta-GGA checkpoint trained WITHOUT the mesh has
        # the underdetermined-alpha clone this key exists to expose).
        "pretrain_mesh": bool(mesh_used),
        # Pretraining-set provenance the Section 3.3 certificate and HISTORY
        # read: which systems the fit saw, on which parent density, at which
        # exchange footing, and how hard the per-system energy term pulled.
        "reference_xc": want_reference,
        # Derived from the BLOCK THE RUN READ, not copied from the manifest:
        # `x_suffix` is the selector itself and `descriptors` is the tensor
        # the exchange loss was built on, so a run that fell back to the
        # total-density rows cannot record the per-channel footing. Both row
        # counts include the mesh rows when the mesh was appended, which is
        # what `pretrain_mesh` above distinguishes.
        "exchange_footing": "spin_channel" if x_suffix == "_x" else "total",
        "energy_term_weight": energy_weight,
        "n_systems": n_systems,
        "n_rows_x": int(descriptors.shape[0]),
        "n_rows_c": int(descriptors_c.shape[0]),
        "energy_term_x_final": float(
            loss_fn_x.parts(xnet_trained, descriptors, Fx_target)[1]),
        "energy_term_c_final": float(
            loss_fn_c.parts(cnet_trained, descriptors_c, Fc_target)[1]),
        # The held-out split and the stop criterion: fraction, seed, interval,
        # patience, the monitored quantity, the held-out systems BY NAME, and
        # per network the step of the best value, the value, whether the run
        # stopped early, and every validation score.
        "validation": validation_record,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "duration_seconds": round(duration, 1),
    }
    # Record whether Becke quadrature weights were available for
    # integration mode.  None means the run did not use integration weighting.
    if integration_weights_complete is not None:
        metadata["integration_weights_complete"] = integration_weights_complete
    if mesh_used:
        # The mesh's pull on each channel, READ BACK from the weight vector
        # the loss was built with rather than restated from the constant: a
        # run at a `pretrain.mesh_fraction` other than the generator's default
        # is otherwise indistinguishable in the record from one at the
        # default, and a loss that ignored the file's share would leave no
        # trace at all.
        metadata["mesh_weight_fraction"] = (
            None if mesh_share is None else float(mesh_share))
        metadata["mesh_loss_share_x"] = _mesh_loss_share(
            loss_fn_x.weights, n_mesh_rows, int(descriptors.shape[0]))
        metadata["mesh_loss_share_c"] = _mesh_loss_share(
            loss_fn_c.weights, n_mesh_rows, int(descriptors_c.shape[0]))
    md_path = os.path.join(checkpoint_dir, "pretrain_metadata.json")
    with open(md_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


# ---------------------------------------------------------------------------
# Legacy checkpoint helpers
# ---------------------------------------------------------------------------

def _count_disk_records(path: str) -> int:
    """Count numpy magic-marker occurrences in a .eqx file.

    Byte-level heuristic, not a true numpy header parser. Safe for the
    dtypes this loader sees; the failure mode is an over-count, at which
    point a real header parser would be needed.

    Streams the file in chunks with a small overlap to avoid slurping a
    multi-hundred-MB checkpoint into memory.
    """
    marker = b'\x93NUMPY'
    count = 0
    chunk_size = 65536
    overlap = len(marker) - 1
    with open(path, 'rb') as f:
        tail = b''
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            count += (tail + chunk).count(marker)
            tail = chunk[-overlap:]
    return count


def _metadata_preflight(
    *,
    metadata_path: str,
    arch: ArchitectureConfig,
) -> dict:
    """Validate checkpoint metadata against arch."""
    try:
        with open(metadata_path, "r") as f:
            md = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(
            f"legacy checkpoint metadata unreadable: {metadata_path}: {e}"
        ) from e
    for key in ("depth", "nodes"):
        if key not in md:
            raise ValueError(
                f"legacy checkpoint metadata missing required key "
                f"{key!r}: {metadata_path}"
            )
    if md["depth"] != arch.depth or md["nodes"] != arch.nodes:
        raise ValueError(
            f"legacy checkpoint metadata mismatch: file has "
            f"depth={md['depth']}, nodes={md['nodes']}; "
            f"arch expects depth={arch.depth}, nodes={arch.nodes}"
        )
    return md


def _disk_record_preflight(
    *,
    path: str,
    library_skeleton: eqx.Module,
    arch: ArchitectureConfig,
) -> None:
    """Byte-level leaf-count sanity on the library skeleton."""
    n_disk_records = _count_disk_records(path)
    n_expected = sum(
        1 for leaf in jax.tree_util.tree_leaves(library_skeleton)
        if is_array_like(leaf)
    )
    if n_disk_records != n_expected:
        raise ValueError(
            f"legacy checkpoint {path} has {n_disk_records} on-disk "
            f"records, expected {n_expected} for library skeleton "
            f"(depth={arch.depth}, nodes={arch.nodes}, "
            f"use_self_attention={arch.attention})"
        )


def _load_one_network(
    *,
    path: str,
    library_skeleton: eqx.Module,
    alec_skeleton: eqx.Module,
    expected_lob_lim: float,
) -> eqx.Module:
    """Load a single network from a legacy-layout checkpoint and graft its
    trainable array leaves onto a fresh alec skeleton.

    Steps:
    1. Deserialize into library_skeleton (capture the return value).
    2. Check lobf.limit against expected_lob_lim.
    3. Graft eqx.is_array leaves from library onto alec via tree_flatten/unflatten
       (NOT eqx.tree_at).
    """
    # Step 2: deserialize, MUST capture the return value
    library_skeleton = eqx.tree_deserialise_leaves(path, library_skeleton)

    # Parameterised lob_lim check
    loaded_lim = library_skeleton.lobf.limit
    if abs(loaded_lim - expected_lob_lim) >= 1e-12:
        raise ValueError(
            f"legacy checkpoint {path}: lobf.limit={loaded_lim} does not "
            f"match expected {expected_lob_lim} for this architecture"
        )

    # Step 3: structural tree walk, NOT eqx.tree_at
    src_leaves_all, _src_treedef = jax.tree_util.tree_flatten(library_skeleton)
    src_array_leaves = [l for l in src_leaves_all if eqx.is_array(l)]

    dst_leaves_all, dst_treedef = jax.tree_util.tree_flatten(alec_skeleton)

    dst_array_count = sum(1 for l in dst_leaves_all if eqx.is_array(l))
    if len(src_array_leaves) != dst_array_count:
        schema_hint = ""
        if "_attn" in path or "/attention" in path:
            schema_hint = (
                "\n\nThis path includes an attention checkpoint. The "
                "self-attention block was rewritten 2026-04-27 to real "
                "multi-head scaled-dot-product attention; old `_attn` "
                "checkpoints are NOT loadable under the new schema. "
                "Delete the old checkpoint and retrain."
            )
        raise ValueError(
            f"legacy->alec graft leaf mismatch at {path}: library has "
            f"{len(src_array_leaves)} eqx.is_array leaves, alec skeleton "
            f"expects {dst_array_count}. Likely causes: arch.depth or "
            f"arch.nodes differs from the pretrained checkpoint, or the "
            f"checkpoint predates the 2026-04-27 attention rewrite."
            f"{schema_hint}"
        )

    # Per-leaf shape/dtype sanity before the graft
    dst_array_positions = [i for i, l in enumerate(dst_leaves_all) if eqx.is_array(l)]
    for pair_idx, dst_pos in enumerate(dst_array_positions):
        s = src_array_leaves[pair_idx]
        d = dst_leaves_all[dst_pos]
        if s.shape != d.shape or s.dtype != d.dtype:
            raise ValueError(
                f"legacy -> alec graft leaf #{pair_idx} shape/dtype mismatch "
                f"at {path}: library {s.shape}/{s.dtype} vs alec "
                f"{d.shape}/{d.dtype}"
            )

    # Final graft: build new destination leaf list and unflatten
    src_iter = iter(src_array_leaves)
    new_leaves = [
        next(src_iter) if eqx.is_array(l) else l
        for l in dst_leaves_all
    ]
    return jax.tree_util.tree_unflatten(dst_treedef, new_leaves)


def from_legacy_step3b(
    legacy_dir: str,
    arch: ArchitectureConfig,
) -> tuple:
    """Load a step3b-era checkpoint into fresh AlecGGA_XNet / AlecGGA_CNet.

    Handles two on-disk layouts:
      Pretrain: legacy_dir/xnet.eqx + cnet.eqx + pretrain_metadata.json
      Training: legacy_dir/xcmodel.eqx + train_metadata.json

    Raises:
        ValueError: ambiguous layout (both present), or metadata mismatch.
        FileNotFoundError: no recognised layout found.
        OSError: file exists but cannot be read.
    """
    pretrain_x_path = os.path.join(legacy_dir, "xnet.eqx")
    pretrain_c_path = os.path.join(legacy_dir, "cnet.eqx")
    pretrain_md_path = os.path.join(legacy_dir, "pretrain_metadata.json")
    training_model_path = os.path.join(legacy_dir, "xcmodel.eqx")
    training_md_path = os.path.join(legacy_dir, "train_metadata.json")

    has_pretrain = (
        os.path.isfile(pretrain_x_path)
        and os.path.isfile(pretrain_c_path)
        and os.path.isfile(pretrain_md_path)
    )
    has_training = (
        os.path.isfile(training_model_path)
        and os.path.isfile(training_md_path)
    )

    if has_pretrain and has_training:
        raise ValueError(
            f"legacy checkpoint directory {legacy_dir} contains BOTH a "
            f"pretrain layout (xnet.eqx + cnet.eqx + pretrain_metadata.json) "
            f"AND a training layout (xcmodel.eqx + train_metadata.json); "
            f"cannot disambiguate. Delete one set before retrying."
        )
    if not has_pretrain and not has_training:
        raise FileNotFoundError(
            f"legacy checkpoint directory {legacy_dir} contains neither a "
            f"pretrain layout nor a training layout. Expected one of: "
            f"(xnet.eqx, cnet.eqx, pretrain_metadata.json) OR "
            f"(xcmodel.eqx, train_metadata.json)."
        )

    if has_pretrain:
        # --- Pretrain-layout branch ---
        _metadata_preflight(metadata_path=pretrain_md_path, arch=arch)

        # Library skeletons use hardcoded legacy values
        lib_xnet_skel = xcquinox.net.GGA_FxNet_extended(
            depth=arch.depth, nodes=arch.nodes, seed=0,
            lob_lim=_legacy_xnet_lob_lim,
            lower_rho_cutoff=1e-12,
            use_self_attention=arch.attention,
            use_laplacian=False,
            use_dm_features=any(d.name == "dm_statistics" for d in arch.descriptors),
            use_cusp=any(d.name == "cusp" for d in arch.descriptors),
            n_dm_features=3,
        )
        lib_cnet_skel = xcquinox.net.GGA_FcNet_extended(
            depth=arch.depth, nodes=arch.nodes, seed=0,
            lob_lim=_legacy_cnet_lob_lim,
            lower_rho_cutoff=1e-12,
            use_self_attention=arch.attention,
            use_laplacian=False,
            use_dm_features=any(d.name == "dm_statistics" for d in arch.descriptors),
            use_cusp=any(d.name == "cusp" for d in arch.descriptors),
            n_dm_features=3,
        )
        alec_xnet_skel, alec_cnet_skel = create_network_pair(arch, seed=0)

        _disk_record_preflight(
            path=pretrain_x_path, library_skeleton=lib_xnet_skel, arch=arch,
        )
        _disk_record_preflight(
            path=pretrain_c_path, library_skeleton=lib_cnet_skel, arch=arch,
        )

        xnet_loaded = _load_one_network(
            path=pretrain_x_path,
            library_skeleton=lib_xnet_skel,
            alec_skeleton=alec_xnet_skel,
            expected_lob_lim=_legacy_xnet_lob_lim,
        )
        cnet_loaded = _load_one_network(
            path=pretrain_c_path,
            library_skeleton=lib_cnet_skel,
            alec_skeleton=alec_cnet_skel,
            expected_lob_lim=_legacy_cnet_lob_lim,
        )
        return xnet_loaded, cnet_loaded

    else:  # has_training
        # --- Training-layout branch ---
        _metadata_preflight(metadata_path=training_md_path, arch=arch)

        class _RXCModelWrapper(eqx.Module):
            """Minimal inline replica of the notebook's RXCModel_GGA_extended.

            Two fields, no methods, just a pytree container so
            `tree_deserialise_leaves` can consume the on-disk records in
            the same order the notebook wrote them.
            """
            xnet: eqx.Module
            cnet: eqx.Module

        # Same hardcoded legacy values here
        lib_xnet_skel = xcquinox.net.GGA_FxNet_extended(
            depth=arch.depth, nodes=arch.nodes, seed=0,
            lob_lim=_legacy_xnet_lob_lim,
            lower_rho_cutoff=1e-12,
            use_self_attention=arch.attention,
            use_laplacian=False,
            use_dm_features=any(d.name == "dm_statistics" for d in arch.descriptors),
            use_cusp=any(d.name == "cusp" for d in arch.descriptors),
            n_dm_features=3,
        )
        lib_cnet_skel = xcquinox.net.GGA_FcNet_extended(
            depth=arch.depth, nodes=arch.nodes, seed=0,
            lob_lim=_legacy_cnet_lob_lim,
            lower_rho_cutoff=1e-12,
            use_self_attention=arch.attention,
            use_laplacian=False,
            use_dm_features=any(d.name == "dm_statistics" for d in arch.descriptors),
            use_cusp=any(d.name == "cusp" for d in arch.descriptors),
            n_dm_features=3,
        )
        wrapper_skel = _RXCModelWrapper(xnet=lib_xnet_skel, cnet=lib_cnet_skel)

        # MUST capture the return value
        wrapper_loaded = eqx.tree_deserialise_leaves(training_model_path, wrapper_skel)
        lib_xnet_loaded = wrapper_loaded.xnet
        lib_cnet_loaded = wrapper_loaded.cnet

        # Parameterised lob_lim check
        for loaded, expected, label in [
            (lib_xnet_loaded, _legacy_xnet_lob_lim, "xnet"),
            (lib_cnet_loaded, _legacy_cnet_lob_lim, "cnet"),
        ]:
            if abs(loaded.lobf.limit - expected) >= 1e-12:
                raise ValueError(
                    f"legacy training checkpoint {training_model_path}: "
                    f"{label}.lobf.limit={loaded.lobf.limit} does not match "
                    f"expected {expected} for this architecture"
                )

        alec_xnet_skel, alec_cnet_skel = create_network_pair(arch, seed=0)

        def _graft_arrays(src: eqx.Module, dst: eqx.Module) -> eqx.Module:
            src_leaves_all, _ = jax.tree_util.tree_flatten(src)
            src_array_leaves = [l for l in src_leaves_all if eqx.is_array(l)]
            dst_leaves_all, dst_treedef = jax.tree_util.tree_flatten(dst)
            dst_array_count = sum(1 for l in dst_leaves_all if eqx.is_array(l))
            if len(src_array_leaves) != dst_array_count:
                raise ValueError(
                    f"legacy -> alec training-layout graft leaf mismatch: "
                    f"library has {len(src_array_leaves)} eqx.is_array "
                    f"leaves, alec expects {dst_array_count}."
                )
            src_iter = iter(src_array_leaves)
            new_leaves = [
                next(src_iter) if eqx.is_array(l) else l
                for l in dst_leaves_all
            ]
            return jax.tree_util.tree_unflatten(dst_treedef, new_leaves)

        xnet_loaded = _graft_arrays(lib_xnet_loaded, alec_xnet_skel)
        cnet_loaded = _graft_arrays(lib_cnet_loaded, alec_cnet_skel)
        return xnet_loaded, cnet_loaded
