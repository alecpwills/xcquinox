"""xcquinox.alec.train -- custom training loop for AlecGGAModel.

Implements THE SPEC Task 5.2: run_training using eqx.filter_value_and_grad
with has_aux=True (xcTrainer does not support aux dicts).

Public API:
  build_optimizer  -- canonical optimizer chain (shared with pretrain)
  run_training     -- dispatcher that selects strategy-specific loop

Internal:
  _adapt_progress_callback -- wraps user callback for 3-arg xcTrainer form
  _train_step              -- JIT-compiled single training step
  _build_model             -- model construction helper
  _build_batch             -- batch precomputation helper
  _save_artifacts          -- checkpoint/metadata saving helper
  _run_static_loop         -- static-weight training loop (default)
  _run_lossnorm_loop       -- loss normalization balancing loop
  _run_twophase_loop       -- two-phase balancing loop
  _run_gradnorm_loop       -- GradNorm balancing loop (Chen et al. 2018)
"""
import json
import math
import os
import pickle  # noqa: S403, saving trusted aux_log data only
import struct
import time
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from xcquinox.alec.config import TrainingSpec, ArchitectureConfig
from xcquinox.alec.solver import SolverConfig, SolverMode
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.losses import make_loss
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.networks import create_network_pair


# ---------------------------------------------------------------------------
# Optimizer builder (public -- canonical chain shared with pretrain)
# ---------------------------------------------------------------------------

def build_optimizer(
    *,
    lr_start: float,
    lr_end: float,
    n_steps: int,
    lr_decay_start: float,
    grad_clip: float,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """Build canonical optimizer chain for training.

    Chain order: clip_by_global_norm -> adamw(lr_schedule, weight_decay).
    ``weight_decay`` is DECOUPLED L2 (adamw); the default 0.0 makes adamw
    byte-identical to the former adam, so existing (decay-free) runs are
    unchanged. A positive value regularizes the (over-capacity) nets -- the
    2026-06-20 review traced the DFS-pool generalization gap partly to training
    with no weight decay while DFS uses it (og_dpyscf/scripts/train.py:47,289).
    LR schedule: a constant-LR warmup for the first ``lr_decay_start`` fraction
    of ``n_steps`` THEN linear decay to ``lr_end``, but ONLY when
    ``lr_decay_start > 0``. With the common ``lr_decay_start = 0`` there is no
    warmup: it is pure linear decay from ``lr_start`` to ``lr_end`` over all
    ``n_steps``.

    Parameters
    ----------
    lr_start : float
        Initial learning rate.
    lr_end : float
        Final learning rate after decay.
    n_steps : int
        Total number of training steps.
    lr_decay_start : float
        Fraction of n_steps before decay begins (0 = immediate decay).
    grad_clip : float
        Global norm clipping threshold.
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
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay),
    )


def _abort_if_nonfinite(loss_value, components, *, loop, step, group=None):
    """Fail-loud finite guard: raise ``FloatingPointError`` the instant a
    training step produces a non-finite loss or loss component, naming the
    offending loop/step/group/channel.

    A non-finite loss propagates as ``0 * NaN = NaN`` into every weight, so the
    run must stop at the first one rather than keep training on garbage. Called
    after every optimizer step in every update loop; the per-step loss is
    already on the host there, so the check is effectively free.
    """
    bad = []
    for k, v in (components or {}).items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            bad.append(k)
    loss_f = float(loss_value)
    if math.isfinite(loss_f) and not bad:
        return
    where = f"loop={loop!r}, step={step}"
    if group is not None:
        where += f", group={group!r}"
    raise FloatingPointError(
        f"non-finite training value ({where}): loss={loss_f!r}, non-finite "
        f"channel(s)={sorted(bad) or '[loss itself]'}. Training aborts, a "
        f"NaN/Inf corrupts every subsequent weight. This is almost always a "
        f"functional/solver gradient singularity on this group's species "
        f"(e.g. polarized correlation differentiated through the SCF at full "
        f"spin polarization)."
    )


# ---------------------------------------------------------------------------
# Progress callback adapter
# ---------------------------------------------------------------------------

def _adapt_progress_callback(user_callback, *, arch, phase):
    """Wrap user's dict-based callback into the 3-arg form expected by
    the xcTrainer progress hook interface (step, total, loss)."""
    if user_callback is None:
        return None

    def xctrainer_hook(step, total, loss):
        user_callback({
            "arch": arch,
            "phase": phase,
            "step": int(step),
            "total": int(total),
            "loss": float(loss),
            "timestamp": time.time(),
        })

    return xctrainer_hook


# ---------------------------------------------------------------------------
# JIT-compiled training step
# ---------------------------------------------------------------------------

@eqx.filter_jit
def _train_step(model, opt_state, batch, loss_fn, optimizer):
    """Single training step: forward + backward + optimizer update.

    Uses eqx.filter_value_and_grad(has_aux=True) because AlecLoss.__call__
    returns (scalar, aux_dict) and xcTrainer does not support aux.
    """
    (loss_value, aux), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(model, batch)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value, aux


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(spec: TrainingSpec) -> AlecGGAModel:
    """Build model from scratch or pretrain checkpoint."""
    if spec.pretrain_checkpoint is None:
        return AlecGGAModel.from_arch(spec.arch, seed=spec.seed)
    xnet_skeleton, cnet_skeleton = create_network_pair(spec.arch, seed=spec.seed)
    xnet_path = os.path.join(spec.pretrain_checkpoint, "xnet.eqx")
    cnet_path = os.path.join(spec.pretrain_checkpoint, "cnet.eqx")
    try:
        loaded_xnet = eqx.tree_deserialise_leaves(xnet_path, xnet_skeleton)
        loaded_cnet = eqx.tree_deserialise_leaves(cnet_path, cnet_skeleton)
    except (ValueError, EOFError, struct.error) as e:
        _path_for_hint = (
            xnet_path
            if ("_attn" in xnet_path or "/attention" in xnet_path)
            else cnet_path
        )
        if "_attn" in _path_for_hint or "/attention" in _path_for_hint:
            raise ValueError(
                f"Failed to deserialise {_path_for_hint}: {e}\n\n"
                "This path includes an attention checkpoint. The "
                "self-attention block was rewritten 2026-04-27 to real "
                "multi-head scaled-dot-product attention; old `_attn` "
                "checkpoints are NOT loadable under the new schema. "
                "Delete the old checkpoint and retrain."
            ) from e
        raise
    return AlecGGAModel.from_arch(spec.arch, xnet=loaded_xnet, cnet=loaded_cnet)


def _build_batch(spec: TrainingSpec, loss) -> dict:
    """Precompute mol_data and build batch dict."""
    required = set()
    required |= set(loss.required_mol_keys)
    for d in spec.arch.materialize_descriptors():
        required |= set(d.required_mol_keys)

    sc = spec.loss_kwargs_dict.get("solver_config") or spec.solver_config
    density_fit = False
    if isinstance(sc, SolverConfig) and sc.mode == SolverMode.FULL:
        density_fit = bool(getattr(sc, "density_fit", False))
        required.add("cderi" if density_fit else "eri")
    # Only forward auxbasis when DF is active; otherwise stays None so the
    # full-ERI path (and its precompute cache key) is byte-identical to before.
    auxbasis = getattr(sc, "auxbasis", None) if density_fit else None

    required_keys = tuple(required)
    mol_data_list = [
        precompute_fixed_density_data(
            m, required_keys=required_keys,
            descriptors=spec.arch.materialize_descriptors(),
            auxbasis=auxbasis,
        )
        for m in spec.molecules
    ]
    return {
        "mol_data": tuple(mol_data_list),
        "targets": spec.targets_dict,
        "atom_energies": spec.atom_energies_dict,
    }


class _BestModelTracker:
    """Tracks the model snapshot at the minimum trailing-mean loss so a best-loss
    checkpoint can be saved ALONGSIDE the final one. The window smooths per-step
    noise (the per-molecule scheme's per-group losses are noisy; a one-epoch
    window gives a stable estimate). This protects against a run that converges
    then destabilizes late and ends on a high-loss snapshot -- observed for
    deep_attn at large subset sizes (final loss ~1e4x its own best-ever)."""

    def __init__(self, window: int = 1):
        self.window = max(1, int(window))
        self._recent: list = []
        self.best_loss = float("inf")
        self.best_model = None

    def update(self, loss_py: float, model) -> None:
        # JAX arrays are immutable, so keeping the reference snapshots the
        # current params (the next apply_updates allocates fresh arrays).
        if not np.isfinite(loss_py):
            return
        self._recent.append(loss_py)
        if len(self._recent) > self.window:
            self._recent.pop(0)
        if len(self._recent) >= self.window:
            avg = sum(self._recent) / len(self._recent)
            if avg < self.best_loss:
                self.best_loss = avg
                self.best_model = model


class _BestValidationTracker:
    """Tracks the model snapshot at the minimum HELD-OUT VALIDATION metric
    (lower = better, e.g. reaction-energy MAE in kcal/mol) and drives in-loop
    early-stop. WS3 (2026-06-20): the over-capacity DFS nets overfit the tiny
    training subset, so we periodically score a disjoint validation slice and
    keep the validation-best snapshot, stopping once it stops improving.

    Distinct from :class:`_BestModelTracker` (which minimizes the TRAINING loss,
    a quantity that keeps dropping as the net overfits): this minimizes a
    GENERALIZATION metric, so its best snapshot is the one that generalizes best,
    not merely the lowest train loss.

    ``update(metric, model)`` records one validation check (the metric is the
    val MAE; ``model`` is the live snapshot, JAX arrays are immutable so the
    reference snapshots the current params). Non-finite metrics are IGNORED:
    they update neither the best snapshot nor the no-improvement streak (a NaN
    val score is a transient, not evidence of non-improvement).

    ``should_stop(patience, min_delta)`` returns True once the last ``patience``
    consecutive FINITE checks each failed to improve the running best by more
    than ``min_delta`` (the streak is computed from the recorded finite-metric
    history, so ``min_delta`` lives entirely in ``should_stop`` and a single
    tracker can be probed at different thresholds). ``patience <= 0`` => never
    stops (the documented no-op).
    """

    def __init__(self):
        self.best_mae = float("inf")
        self.best_model = None
        # Finite val metrics in arrival order; the no-improvement streak that
        # drives early-stop is derived from this (using should_stop's min_delta),
        # NOT stored, so min_delta is a should_stop-only concern.
        self._finite_metrics: list = []

    def update(self, metric: float, model) -> None:
        m = float(metric)
        if not np.isfinite(m):
            return                       # transient NaN/inf: ignore entirely
        self._finite_metrics.append(m)
        # The best SNAPSHOT is the numerically-lowest val metric seen so far
        # (the best-generalizing model), even when the last drop was tiny.
        if m < self.best_mae:
            self.best_mae = m
            self.best_model = model

    def should_stop(self, patience: int, min_delta: float = 0.0) -> bool:
        if int(patience) <= 0:
            return False                 # no-op: early-stop disabled
        patience = int(patience)
        md = float(min_delta)
        # No-improvement streak: count trailing finite checks that each failed to
        # beat the best-so-far (over the prefix BEFORE that check) by > min_delta.
        streak = 0
        for i in range(len(self._finite_metrics) - 1, 0, -1):
            best_prefix = min(self._finite_metrics[:i])
            if self._finite_metrics[i] < best_prefix - md:
                break                    # this check improved -> streak ends
            streak += 1
            if streak >= patience:
                return True
        return False


def _build_validation_data(spec):
    """Build ``(val_mol_data, val_reactions)`` for the in-loop held-out
    validation, or ``(None, None)`` when validation is DISABLED. WS3.

    Disabled (returns ``(None, None)``, the byte-identical no-op) when
    ``spec.validate_every <= 0`` OR no ``validation_molecules`` OR no
    ``validation_reactions_path``. Otherwise precompute density-only MoleculeData
    for each ``validation_molecules`` entry via the SAME
    :func:`precompute_fixed_density_data` path the training batch uses (matching
    descriptor signature + DF/auxbasis from the spec's solver_config), and load
    the val reaction dicts from the JSON at ``validation_reactions_path``.

    Module-level so :func:`_run_per_molecule_loop` can call it through a
    monkeypatchable seam (tests stub it to avoid PySCF).
    """
    if int(getattr(spec, "validate_every", 0)) <= 0:
        return (None, None)
    val_mols = tuple(getattr(spec, "validation_molecules", ()) or ())
    rxn_path = getattr(spec, "validation_reactions_path", None)
    if not val_mols or not rxn_path:
        return (None, None)

    # Descriptor + DF/auxbasis signature mirrors _build_batch so the val SCF
    # inputs match what the model consumes.
    required = set()
    for d in spec.arch.materialize_descriptors():
        required |= set(d.required_mol_keys)
    sc = spec.loss_kwargs_dict.get("solver_config") or spec.solver_config
    density_fit = False
    if isinstance(sc, SolverConfig) and sc.mode == SolverMode.FULL:
        density_fit = bool(getattr(sc, "density_fit", False))
        required.add("cderi" if density_fit else "eri")
    auxbasis = getattr(sc, "auxbasis", None) if density_fit else None
    required_keys = tuple(required)

    val_mol_data = {
        m.name: precompute_fixed_density_data(
            m, required_keys=required_keys,
            descriptors=spec.arch.materialize_descriptors(),
            auxbasis=auxbasis,
        )
        for m in val_mols
    }
    with open(rxn_path) as f:
        val_reactions = json.load(f)
    return (val_mol_data, val_reactions)


def _validation_reaction_mae(model, val_mol_data, val_reactions,
                             solver_config=None, *, energy_fn=None) -> float:
    """In-loop held-out VALIDATION reaction-energy MAE (kcal/mol). WS3.

    For every species in ``val_mol_data`` compute a total energy via the SAME
    energy path the training loss uses (``oneshot.total_energy_for_solver``,
    dispatched on the solver MODE so FULL re-runs the SCF and ONESHOT/FIXED_J
    use the fixed-density functional), then aggregate per-reaction errors against
    ``reaction_energy_ref`` with :func:`eval_holdout.reaction_mae_kcalmol`. A
    species whose energy is non-finite drops its reaction; with no finite
    reactions the MAE is NaN (the tracker ignores it).

    ``energy_fn(model, mol_data)`` is an injectable seam: the default is
    ``total_energy_for_solver`` bound to ``solver_config``; tests pass a stub so
    the pure MAE assembly runs with NO PySCF. Returns a Python float.
    """
    from xcquinox.alec.eval_holdout import reaction_mae_kcalmol
    if energy_fn is None:
        from xcquinox.alec.oneshot import total_energy_for_solver

        def energy_fn(m, md):
            return float(total_energy_for_solver(m, md,
                                                 solver_config=solver_config))
    energies_ha: dict = {}
    for name, md in val_mol_data.items():
        try:
            e = float(energy_fn(model, md))
        except Exception:  # noqa: BLE001 -- a diverged species drops its reaction
            e = float("nan")
        energies_ha[name] = e if math.isfinite(e) else float("nan")
    mae, _n_used, _n_nan = reaction_mae_kcalmol(energies_ha, val_reactions)
    return float(mae)


def _save_artifacts(spec, model, losses, aux_log, duration, best_model=None,
                    val_best_model=None, extra_metadata=None) -> dict:
    """Save model.eqx (final), losses.npy, aux_log, train_metadata.json. If a
    best-loss snapshot is given, ALSO write model_best.eqx side-by-side (the
    final model.eqx is still written) so eval can opt into the pre-instability
    checkpoint.

    WS3: ``val_best_model`` (the minimum held-out-validation snapshot) is written
    to ``model_val_best.eqx`` when present, and ``extra_metadata`` (e.g.
    ``early_stopped`` / ``epochs_run`` / ``val_best_mae``) is merged into
    train_metadata.json. Both default to no-op so non-validating runs are
    byte-identical."""
    os.makedirs(spec.checkpoint_dir, exist_ok=True)

    model_path = os.path.join(spec.checkpoint_dir, "model.eqx")
    eqx.tree_serialise_leaves(model_path, model)
    if best_model is not None:
        eqx.tree_serialise_leaves(
            os.path.join(spec.checkpoint_dir, "model_best.eqx"), best_model)
    if val_best_model is not None:
        eqx.tree_serialise_leaves(
            os.path.join(spec.checkpoint_dir, "model_val_best.eqx"),
            val_best_model)

    losses_np = np.array(losses, dtype=np.float64)
    np.save(os.path.join(spec.checkpoint_dir, "losses.npy"), losses_np)

    aux_path = os.path.join(spec.checkpoint_dir, "aux_log.pkl")
    with open(aux_path, "wb") as f:
        pickle.dump(aux_log, f, protocol=4)

    loss_kwargs_ser = {
        k: v.describe() if isinstance(v, SolverConfig) else v
        for k, v in spec.loss_kwargs_dict.items()
    }
    metadata = {
        "arch_name": spec.arch.name,
        # Shape-changing flag: a polarized cnet has input width
        # +1, so two checkpoints with the same arch_name but different
        # polarization are NOT interchangeable, record it so loaders can tell.
        "use_polarized_correlation": bool(spec.arch.use_polarized_correlation),
        "loss_name": spec.loss_name,
        "loss_kwargs": loss_kwargs_ser,
        "solver_config": (
            spec.solver_config.describe()
            if spec.solver_config is not None
            else None
        ),
        "n_steps": spec.n_steps,
        "lr_start": spec.lr_start,
        "lr_end": spec.lr_end,
        "lr_decay_start": spec.lr_decay_start,
        "grad_clip": spec.grad_clip,
        "pretrain_checkpoint": spec.pretrain_checkpoint,
        "molecules": [m.name for m in spec.molecules],
        "targets": spec.targets_dict,
        "atom_energies": spec.atom_energies_dict,
        "loss_metric": spec.loss_metric,
        "balancing": spec.balancing.describe() if spec.balancing is not None else None,
        "final_loss": float(losses_np[-1]) if len(losses_np) > 0 else float("nan"),
        "min_loss": float(np.min(losses_np)) if len(losses_np) > 0 else float("nan"),
        "has_best_checkpoint": best_model is not None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "duration_seconds": round(duration, 1),
    }
    # FIX 3 (WS3-ESV-1): add the validation metadata keys ONLY when validation
    # actually ran (a val-best snapshot was taken, or the loop passed validation
    # extras). Otherwise the key set is BYTE-IDENTICAL to pre-WS3 -- a non-
    # validating run (any loop, or per_molecule with validate_every=0) must not
    # gain has_val_best_checkpoint / early_stopped / val_* keys.
    val_ran = val_best_model is not None or bool(extra_metadata)
    if val_ran:
        metadata["has_val_best_checkpoint"] = val_best_model is not None
    if extra_metadata:
        metadata.update(extra_metadata)

    md_path = os.path.join(spec.checkpoint_dir, "train_metadata.json")
    with open(md_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


# ---------------------------------------------------------------------------
# WS5 (2026-06-20): RESUMABLE per_molecule training.
#
# A ``per_molecule`` run killed by walltime/maintenance must RESUME from its
# last PERIODIC checkpoint and finish; an early-stopped run is recognized as
# complete. The mechanism is gated entirely behind ``spec.checkpoint_every``,
# which DEFAULTS to 0 (a no-op): with 0, none of the functions below run and the
# loop is byte-identical to before.
#
# WS6 CONTRACT (do NOT violate):
#   * Periodic checkpoints use ``resume_*`` filenames and MUST NOT write
#     ``model.eqx`` (the harness success signal). A mid-run dir therefore has
#     ``resume_state.pkl`` present AND ``model.eqx`` ABSENT.
#   * Completion (clean OR early-stop) writes ``model.eqx`` (via _save_artifacts)
#     + a NEW ``completion.json`` sentinel, then DELETES the ``resume_*`` files.
# ---------------------------------------------------------------------------

# resume_* artifact filenames (the WS6-contract resume set) + completion sentinel.
_RESUME_MODEL = "resume_model.eqx"
_RESUME_OPT_STATE = "resume_opt_state.eqx"
_RESUME_BEST = "resume_best.eqx"
_RESUME_VAL_BEST = "resume_val_best.eqx"
_RESUME_STATE = "resume_state.pkl"
_COMPLETION_SENTINEL = "completion.json"
# Every resume_* file the periodic checkpoint may write (for cleanup on
# completion). model.eqx / completion.json are deliberately NOT in this list.
_RESUME_FILES = (
    _RESUME_MODEL, _RESUME_OPT_STATE, _RESUME_BEST, _RESUME_VAL_BEST,
    _RESUME_STATE,
)


def _atomic_serialise(path, pytree) -> None:
    """``eqx.tree_serialise_leaves`` to ``path`` ATOMICALLY (write to a sibling
    temp file then ``os.replace``), so a SIGKILL mid-write can never leave a
    half-written resume artifact that would crash the resuming run."""
    tmp = path + ".tmp"
    eqx.tree_serialise_leaves(tmp, pytree)
    os.replace(tmp, path)


def _write_resume_checkpoint(checkpoint_dir, *, model, opt_state, rng_state,
                             order, train_best_loss, train_recent, train_window,
                             train_best_model, val_present, val_best_mae,
                             val_finite_metrics, val_best_model, epoch, update,
                             losses, aux_log, early_stopped) -> None:
    """Write one resume checkpoint ATOMICALLY from PRE-CAPTURED state (WS5).

    Persists everything needed to continue the per_molecule loop exactly where
    it left off WITHOUT re-walking trained groups: the ``model`` / ``opt_state``
    (the latter carries the adamw step count, so the LR schedule resumes), the
    ``rng_state`` AND the per-epoch group ``order`` permutation (so the next
    epoch shuffles the SAME sequence as a continuing run), both trackers' scalars
    + best_model snapshots, and the accumulated ``losses`` / ``aux_log`` / loop
    counters.

    CRITICAL (WS5-RESUME-02 / WS5-SIG-1): every argument is a PRE-CAPTURED
    epoch-boundary value (the caller snapshots ``rng.get_state()``,
    ``list(order)``, the tracker scalars/``_recent``/models, ``list(losses)`` at
    the LAST COMPLETED epoch). This function NEVER reaches into a live ``rng`` or
    ``tracker`` whose fields advance mid-epoch, so a mid-epoch SIGTERM flush
    writes a SELF-CONSISTENT (stale-but-exact) snapshot of the last completed
    epoch -- never a torn one (advanced rng/losses paired with a stale epoch).

    Each file is written to a ``.tmp`` sibling then ``os.replace``-d into place,
    so the set is crash-consistent per-file. Does NOT write ``model.eqx`` (the
    harness success signal): a mid-run dir is ``resume_state.pkl`` present +
    ``model.eqx`` absent.

    ``train_best_model`` / ``val_best_model`` are the captured best_model pytrees
    (or ``None``); ``resume_best.eqx`` / ``resume_val_best.eqx`` are written ONLY
    when the respective snapshot is present. ``val_present`` is True iff
    validation ran (its scalars are then meaningful).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    _atomic_serialise(os.path.join(checkpoint_dir, _RESUME_MODEL), model)
    _atomic_serialise(os.path.join(checkpoint_dir, _RESUME_OPT_STATE), opt_state)

    has_train_best = train_best_model is not None
    if has_train_best:
        _atomic_serialise(os.path.join(checkpoint_dir, _RESUME_BEST),
                          train_best_model)
    has_val_best = bool(val_present) and val_best_model is not None
    if has_val_best:
        _atomic_serialise(os.path.join(checkpoint_dir, _RESUME_VAL_BEST),
                          val_best_model)

    state = {
        "epoch": int(epoch),
        "update": int(update),
        # The per-epoch group permutation captured at epoch boundary. Restoring
        # it (then continuing rng.shuffle) reproduces the killed run's order.
        "order": [int(x) for x in order],
        # np.random.RandomState uses get_state()/set_state() (NOT the stdlib
        # random getstate/setstate); the loop's rng is a RandomState.
        "rng_state": rng_state,
        # _BestModelTracker (train-loss best) scalars.
        "best_loss": float(train_best_loss),
        "_recent": list(train_recent),
        "window": int(train_window),
        "has_train_best": bool(has_train_best),
        # _BestValidationTracker scalars (only meaningful when val ran).
        "val_present": bool(val_present),
        "best_mae": (float(val_best_mae) if val_present else None),
        "_finite_metrics": (list(val_finite_metrics)
                            if val_present else None),
        "has_val_best": bool(has_val_best),
        "losses": list(losses),
        "aux_log": list(aux_log),
        "early_stopped": bool(early_stopped),
    }
    state_path = os.path.join(checkpoint_dir, _RESUME_STATE)
    tmp = state_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=4)
    os.replace(tmp, state_path)


def _load_resume_checkpoint(checkpoint_dir, *, model_skeleton,
                            opt_state_skeleton) -> dict:
    """Inverse of :func:`_write_resume_checkpoint` (WS5).

    Deserialises ``resume_model.eqx`` / ``resume_opt_state.eqx`` against the
    supplied skeletons (a freshly-init model + ``optimizer.init`` opt_state,
    exactly the :func:`_build_model` skeleton pattern), rehydrates a
    :class:`_BestModelTracker` and (when validation ran) a
    :class:`_BestValidationTracker` INCLUDING their best_model snapshots from
    ``resume_best.eqx`` / ``resume_val_best.eqx``, and returns a dict with keys:
    ``model, opt_state, rng_state, order, train_tracker, val_tracker, epoch,
    update, losses, aux_log, early_stopped``. ``val_tracker`` is ``None`` when
    none was saved; ``order`` is the restored per-epoch group permutation.
    """
    with open(os.path.join(checkpoint_dir, _RESUME_STATE), "rb") as f:
        state = pickle.load(f)  # noqa: S301 -- trusted, written by this codebase

    model = eqx.tree_deserialise_leaves(
        os.path.join(checkpoint_dir, _RESUME_MODEL), model_skeleton)
    opt_state = eqx.tree_deserialise_leaves(
        os.path.join(checkpoint_dir, _RESUME_OPT_STATE), opt_state_skeleton)

    train_tracker = _BestModelTracker(window=int(state["window"]))
    train_tracker.best_loss = float(state["best_loss"])
    train_tracker._recent = list(state["_recent"])
    if state.get("has_train_best"):
        train_tracker.best_model = eqx.tree_deserialise_leaves(
            os.path.join(checkpoint_dir, _RESUME_BEST), model_skeleton)

    val_tracker = None
    if state.get("val_present"):
        val_tracker = _BestValidationTracker()
        val_tracker.best_mae = (float(state["best_mae"])
                                if state["best_mae"] is not None
                                else float("inf"))
        val_tracker._finite_metrics = list(state["_finite_metrics"] or [])
        if state.get("has_val_best"):
            val_tracker.best_model = eqx.tree_deserialise_leaves(
                os.path.join(checkpoint_dir, _RESUME_VAL_BEST), model_skeleton)

    return {
        "model": model,
        "opt_state": opt_state,
        "rng_state": state["rng_state"],
        "order": [int(x) for x in state["order"]],
        "train_tracker": train_tracker,
        "val_tracker": val_tracker,
        "epoch": int(state["epoch"]),
        "update": int(state["update"]),
        "losses": list(state["losses"]),
        "aux_log": list(state["aux_log"]),
        "early_stopped": bool(state["early_stopped"]),
    }


def _has_resume_checkpoint(checkpoint_dir) -> bool:
    """A resumable checkpoint is present iff ``resume_state.pkl`` exists AND the
    run is NOT already complete (no ``completion.json`` / ``model.eqx``)."""
    if os.path.isfile(os.path.join(checkpoint_dir, _COMPLETION_SENTINEL)):
        return False
    if os.path.isfile(os.path.join(checkpoint_dir, "model.eqx")):
        return False
    return os.path.isfile(os.path.join(checkpoint_dir, _RESUME_STATE))


def _finalize_completion(checkpoint_dir, *, early_stopped, epochs_run) -> None:
    """Mark a run COMPLETE (WS5/WS6): write the ``completion.json`` sentinel and
    DELETE the ``resume_*`` set. Call AFTER ``_save_artifacts`` has written
    ``model.eqx``. Deleting an absent resume file is tolerated (idempotent;
    a checkpoint_every=0 run never wrote them)."""
    sentinel = {
        "completed": True,
        "early_stopped": bool(early_stopped),
        "epochs_run": int(epochs_run),
    }
    with open(os.path.join(checkpoint_dir, _COMPLETION_SENTINEL), "w") as f:
        json.dump(sentinel, f, indent=2)
    for fn in _RESUME_FILES:
        try:
            os.remove(os.path.join(checkpoint_dir, fn))
        except FileNotFoundError:
            pass


# Module-level resume-flusher holder. The per_molecule loop registers a
# zero-arg flush fn (writes the live resume checkpoint); the worker's SIGTERM
# handler calls it best-effort before exiting so an in-flight epoch is not lost
# between periodic checkpoints. A single-slot holder is sufficient: one training
# loop runs per worker process.
_RESUME_FLUSHER = None


def _register_resume_flusher(fn) -> None:
    """Register the loop's zero-arg resume-flush fn (WS5)."""
    global _RESUME_FLUSHER
    _RESUME_FLUSHER = fn


def _clear_resume_flusher() -> None:
    """Clear the registered resume-flush fn (WS5)."""
    global _RESUME_FLUSHER
    _RESUME_FLUSHER = None


def _get_resume_flusher():
    """Return the registered resume-flush fn, or ``None`` (WS5)."""
    return _RESUME_FLUSHER


# ---------------------------------------------------------------------------
# Training loop strategies
# ---------------------------------------------------------------------------

def _run_static_loop(spec, model, batch, loss, progress_callback):
    """Static weighting loop -- unchanged behavior from original run_training."""
    t0 = time.time()
    optimizer = build_optimizer(
        lr_start=spec.lr_start, lr_end=spec.lr_end,
        n_steps=spec.n_steps, lr_decay_start=spec.lr_decay_start,
        grad_clip=spec.grad_clip,
        weight_decay=spec.weight_decay,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    losses = []
    aux_log = []
    tracker = _BestModelTracker()
    progress_hook = _adapt_progress_callback(
        progress_callback, arch=spec.arch.name, phase="train"
    )

    for step in range(spec.n_steps):
        model, opt_state, loss_value, aux = _train_step(
            model, opt_state, batch, loss, optimizer
        )
        loss_py = float(loss_value)
        _abort_if_nonfinite(loss_value, aux, loop="batched/static", step=step)
        losses.append(loss_py)
        tracker.update(loss_py, model)
        aux_log.append({"step": step, "loss": loss_py, "aux": aux})
        if progress_hook is not None:
            progress_hook(step + 1, spec.n_steps, loss_py)

    duration = time.time() - t0
    return _save_artifacts(spec, model, losses, aux_log, duration,
                           best_model=tracker.best_model)


def _run_lossnorm_loop(spec, model, batch, loss, progress_callback):
    """Loss normalization: divide each component by its step-0 magnitude."""
    t0 = time.time()
    optimizer = build_optimizer(
        lr_start=spec.lr_start, lr_end=spec.lr_end,
        n_steps=spec.n_steps, lr_decay_start=spec.lr_decay_start,
        grad_clip=spec.grad_clip,
        weight_decay=spec.weight_decay,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    relative = spec.loss_metric == "relative"
    progress_hook = _adapt_progress_callback(
        progress_callback, arch=spec.arch.name, phase="train"
    )

    components_0 = loss.compute_components(model, batch, relative=relative)
    norms = {k: jnp.maximum(jnp.abs(v), 1e-12) for k, v in components_0.items()}
    component_keys = tuple(sorted(norms.keys()))

    @eqx.filter_jit
    def _normed_step(model, opt_state, batch):
        def normed_loss_fn(model, batch):
            components = loss.compute_components(model, batch, relative=relative)
            total = sum(components[k] / norms[k] for k in component_keys)
            return total, components
        (loss_val, components), grads = eqx.filter_value_and_grad(
            normed_loss_fn, has_aux=True)(model, batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss_val, components

    losses = []
    aux_log = []
    tracker = _BestModelTracker()
    for step in range(spec.n_steps):
        model, opt_state, loss_value, components = _normed_step(
            model, opt_state, batch)
        loss_py = float(loss_value)
        _abort_if_nonfinite(loss_value, components, loop="batched/lossnorm",
                            step=step)
        losses.append(loss_py)
        tracker.update(loss_py, model)
        eff_weights = {k: float(1.0 / norms[k]) for k in component_keys}
        aux_log.append({
            "step": step, "loss": loss_py, "aux": components,
            "balancing_info": {"strategy": "loss_norm", "effective_weights": eff_weights},
        })
        if progress_hook is not None:
            progress_hook(step + 1, spec.n_steps, loss_py)

    duration = time.time() - t0
    return _save_artifacts(spec, model, losses, aux_log, duration,
                           best_model=tracker.best_model)


def _filter_loss_kwargs(loss_kwargs_dict, target_loss_name):
    """Return only kwargs accepted by target_loss_name's __init__."""
    import inspect
    from xcquinox.alec.losses import LOSS_REGISTRY
    loss_cls = LOSS_REGISTRY[target_loss_name]
    sig = inspect.signature(loss_cls.__init__)
    allowed = set(sig.parameters.keys()) - {"self", "molecules"}
    return {k: v for k, v in loss_kwargs_dict.items() if k in allowed}


def _run_twophase_loop(spec, model, batch, loss, progress_callback):
    """Two-phase training: energy-only then compound loss with fresh optimizer."""
    t0 = time.time()
    balancing = spec.balancing
    phase2_steps = spec.n_steps - balancing.phase1_steps
    progress_hook = _adapt_progress_callback(
        progress_callback, arch=spec.arch.name, phase="train"
    )

    losses = []
    aux_log = []
    tracker = _BestModelTracker()

    # Phase 1: loss with optional kwarg overrides from TwoPhaseConfig
    phase1_kwargs = _filter_loss_kwargs(spec.loss_kwargs_dict, balancing.phase1_loss)
    if balancing.phase1_loss_kwargs:
        phase1_kwargs.update(dict(balancing.phase1_loss_kwargs))
    # Inject step-6 anchor fields unless explicitly overridden above.
    phase1_kwargs.setdefault("pbe_anchor_weight", spec.pbe_anchor_weight)
    phase1_kwargs.setdefault("pbe_anchor_sample", spec.pbe_anchor_sample)
    phase1_loss = make_loss(
        balancing.phase1_loss, molecules=spec.molecules, **phase1_kwargs)
    phase1_optimizer = build_optimizer(
        lr_start=spec.lr_start, lr_end=spec.lr_end,
        n_steps=balancing.phase1_steps,
        lr_decay_start=spec.lr_decay_start, grad_clip=spec.grad_clip,
        weight_decay=spec.weight_decay,
    )
    opt_state = phase1_optimizer.init(eqx.filter(model, eqx.is_array))

    for step in range(balancing.phase1_steps):
        model, opt_state, loss_value, aux = _train_step(
            model, opt_state, batch, phase1_loss, phase1_optimizer)
        loss_py = float(loss_value)
        _abort_if_nonfinite(loss_value, aux, loop="batched/twophase", step=step)
        losses.append(loss_py)
        tracker.update(loss_py, model)
        aux_log.append({
            "step": step, "loss": loss_py, "aux": aux,
            "balancing_info": {"strategy": "two_phase", "phase": 1},
        })
        if progress_hook is not None:
            progress_hook(step + 1, spec.n_steps, loss_py)

    # Phase 2: compound loss with FRESH optimizer
    phase2_optimizer = build_optimizer(
        lr_start=spec.lr_start, lr_end=spec.lr_end,
        n_steps=phase2_steps,
        lr_decay_start=spec.lr_decay_start, grad_clip=spec.grad_clip,
        weight_decay=spec.weight_decay,
    )
    opt_state = phase2_optimizer.init(eqx.filter(model, eqx.is_array))

    for step in range(phase2_steps):
        global_step = balancing.phase1_steps + step
        model, opt_state, loss_value, aux = _train_step(
            model, opt_state, batch, loss, phase2_optimizer)
        loss_py = float(loss_value)
        _abort_if_nonfinite(loss_value, aux, loop="batched/twophase",
                            step=global_step)
        losses.append(loss_py)
        tracker.update(loss_py, model)
        aux_log.append({
            "step": global_step, "loss": loss_py, "aux": aux,
            "balancing_info": {"strategy": "two_phase", "phase": 2},
        })
        if progress_hook is not None:
            progress_hook(global_step + 1, spec.n_steps, loss_py)

    duration = time.time() - t0
    return _save_artifacts(spec, model, losses, aux_log, duration,
                           best_model=tracker.best_model)


# Channels whose step-0 loss L_i(0) is at or below this floor have no
# meaningful inverse-training-rate r_i = L_i(t)/L_i(0) (GradNorm, Chen et al.
# 2018 arXiv:1711.02257, assumes L_i(0) > 0). Such channels arise in the
# small-subset sweep (e.g. a BH76-only/IP13-only subset has a zero AE channel,
# an all-None vxc/rho channel is constant 0, or a well-pretrained delta-AE
# channel starts at exactly 0). Without guarding, a 0 -> nonzero excursion makes
# r_i ~ comp/floor ~ 1e10 and the softmax target corrupts ALL task weights.
_GRADNORM_L0_FLOOR = 1e-8


def _gradnorm_relative_rates(comp_values, L0_values, floor=_GRADNORM_L0_FLOOR):
    """GradNorm relative inverse-training-rates r_i / mean(r), robust to L0~=0.

    For valid channels (L0 > floor) returns r_i / mean_valid(r) exactly as
    Chen et al. 2018. Channels with L0 <= floor are NEUTRALIZED: their relative
    rate is set to 1 (so the GradNorm target reduces to the mean gradient norm,
    i.e. no rebalancing pressure) and they are excluded from the mean, so they
    can neither spike nor distort the valid channels' rates.
    """
    valid = L0_values > floor
    safe_L0 = jnp.where(valid, L0_values, 1.0)
    r = jnp.where(valid, comp_values / safe_L0, 1.0)
    n_valid = jnp.maximum(jnp.sum(valid), 1.0)
    r_mean = jnp.sum(jnp.where(valid, r, 0.0)) / n_valid
    return jnp.where(valid, r / (r_mean + 1e-12), 1.0)


def _run_gradnorm_loop(spec, model, batch, loss, progress_callback):
    """GradNorm (Chen et al. 2018): learned per-task weights via gradient norm equalization."""
    t0 = time.time()
    balancing = spec.balancing
    relative = spec.loss_metric == "relative"
    optimizer = build_optimizer(
        lr_start=spec.lr_start, lr_end=spec.lr_end,
        n_steps=spec.n_steps, lr_decay_start=spec.lr_decay_start,
        grad_clip=spec.grad_clip,
        weight_decay=spec.weight_decay,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    progress_hook = _adapt_progress_callback(
        progress_callback, arch=spec.arch.name, phase="train"
    )

    component_keys = tuple(sorted(
        loss.compute_components(model, batch, relative=relative).keys()
    ))
    n_tasks = len(component_keys)

    log_weights = jnp.zeros(n_tasks)
    weight_optimizer = optax.adam(balancing.weight_lr)
    weight_opt_state = weight_optimizer.init(log_weights)

    L0 = loss.compute_components(model, batch, relative=relative)
    L0_values = jnp.stack([L0[k] for k in component_keys])
    # Warn (once, at setup) about channels with ~0 step-0 loss; these are
    # neutralized in the GradNorm rebalancing (see _gradnorm_relative_rates).
    # Predicate identical to _gradnorm_relative_rates' `L0 > floor` (loss
    # components are non-negative by construction, so no abs() needed).
    _zero_L0 = [k for k in component_keys if float(L0[k]) <= _GRADNORM_L0_FLOOR]
    if _zero_L0:
        warnings.warn(
            f"GradNorm: task channel(s) {_zero_L0} have ~0 step-0 loss "
            f"(<= {_GRADNORM_L0_FLOOR:g}); their inverse-training-rate is "
            f"neutralized (relative rate 1) so they cannot distort the learned "
            f"task weights.",
            RuntimeWarning, stacklevel=2,
        )

    # GradNorm step is deliberately split into 1 + n_tasks + 1 small JITs
    # rather than one monolithic graph.  L5_gradnorm_vxc_step7 has 5 task
    # channels and the full_3 solver differentiates through 3 SCF cycles
    # per molecule; a single jit that contains the weighted-loss grad +
    # 5 per-task grads forces XLA to keep activations from 6 forward
    # passes alive simultaneously (~31 GiB peak observed at 7 species,
    # OOMs both GPU and a 32 GiB CPU box).  Splitting drops peak memory
    # to one forward+backward at a time (~5 GiB) and replaces the single
    # 11+ minute compile with several ~30 s compiles.
    @eqx.filter_jit
    def _model_grad_step(model, batch, weights):
        """Forward pass + weighted-loss gradient.

        Returns (components_dict, comp_values, model_grads).  This is
        the SHARED forward pass; per-task gnorms below re-execute the
        forward (they need their own backward graph with a different
        upstream grad) but each runs in its own JIT and frees its
        intermediates before the next one starts.
        """
        components = loss.compute_components(model, batch, relative=relative)
        comp_values = jnp.stack([components[k] for k in component_keys])

        def weighted_loss(m):
            c = loss.compute_components(m, batch, relative=relative)
            cv = jnp.stack([c[k] for k in component_keys])
            return jnp.sum(weights * cv)
        _, model_grads = eqx.filter_value_and_grad(weighted_loss)(model)
        return components, comp_values, model_grads

    def _make_task_gnorm_jit(task_key):
        """Compile one ``filter_jit`` per task channel.

        ``task_key`` is captured Python-side so each kernel is a small,
        single-output forward+backward graph.  XLA caches the compile by
        function identity; we get exactly ``n_tasks`` cached kernels,
        amortized across all training steps.
        """
        @eqx.filter_jit
        def _task_gnorm(model, batch, weight_i):
            def task_loss(m):
                c = loss.compute_components(m, batch, relative=relative)
                return weight_i * c[task_key]
            g = eqx.filter_grad(task_loss)(model)
            return optax.global_norm(eqx.filter(g, eqx.is_array))
        return _task_gnorm
    _task_gnorm_jits = {k: _make_task_gnorm_jit(k) for k in component_keys}

    @eqx.filter_jit
    def _apply_updates(model, opt_state, log_weights, weight_opt_state,
                       comp_values, model_grads, G, L0_values, weights):
        """Optimizer update (weights + model).  Pure jnp/optax math,
        compiles in <1 s and uses negligible memory.
        """
        # Robust relative inverse-training-rates (neutralizes L0~=0
        # channels instead of letting r ~ comp/1e-12 spike to ~1e10).
        r_relative = _gradnorm_relative_rates(comp_values, L0_values)
        G_mean = jnp.mean(G)
        targets = G_mean * (r_relative ** balancing.alpha)
        # The GradNorm loss is minimized w.r.t. log_weights via the
        # reparameterization w = softmax(lw)*T (which keeps sum(w)=T exactly),
        # rather than Chen et al.'s direct grad w.r.t. w. Both share the same
        # stationary point (targets are stop_gradient'd); this form is stable
        # and avoids a separate renormalization step. The L2 surrogate of their
        # L1 |G_i - target_i| residual likewise shares the fixed point.
        weight_grads = jax.grad(
            lambda lw: jnp.sum(
                (G * (jax.nn.softmax(lw) * n_tasks / (weights + 1e-12))
                 - jax.lax.stop_gradient(targets)) ** 2
            )
        )(log_weights)
        w_updates, new_weight_opt_state = weight_optimizer.update(
            weight_grads, weight_opt_state)
        new_log_weights = log_weights + w_updates
        updates, new_opt_state = optimizer.update(model_grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)
        total = jnp.sum(weights * comp_values)
        return (new_model, new_opt_state, new_log_weights,
                new_weight_opt_state, total)

    losses_list = []
    aux_log = []
    tracker = _BestModelTracker()
    for step in range(spec.n_steps):
        weights = jax.nn.softmax(log_weights) * n_tasks
        components, comp_values, model_grads = _model_grad_step(
            model, batch, weights)
        # Per-task gnorms: each call is its own JIT.  XLA frees
        # intermediates between calls so peak memory ≈ ONE forward+
        # backward at a time, not n_tasks of them simultaneously.
        per_task_gnorms = []
        for i, k in enumerate(component_keys):
            per_task_gnorms.append(
                _task_gnorm_jits[k](model, batch, weights[i])
            )
        G = jnp.stack(per_task_gnorms)
        (model, opt_state, log_weights,
         weight_opt_state, loss_value) = _apply_updates(
            model, opt_state, log_weights, weight_opt_state,
            comp_values, model_grads, G, L0_values, weights)
        loss_py = float(loss_value)
        _abort_if_nonfinite(loss_value, components, loop="batched/gradnorm",
                            step=step)
        losses_list.append(loss_py)
        tracker.update(loss_py, model)
        eff = {k: float(weights[i]) for i, k in enumerate(component_keys)}
        gn = {k: float(G[i]) for i, k in enumerate(component_keys)}
        aux_log.append({
            "step": step, "loss": loss_py, "aux": components,
            "balancing_info": {
                "strategy": "gradnorm",
                "effective_weights": eff,
                "gradient_norms": gn,
            },
        })
        if progress_hook is not None:
            progress_hook(step + 1, spec.n_steps, loss_py)

    duration = time.time() - t0
    return _save_artifacts(spec, model, losses_list, aux_log, duration,
                           best_model=tracker.best_model)


# ---------------------------------------------------------------------------
# Per-molecule (DFS/dpyscf-style) stochastic update loop
# ---------------------------------------------------------------------------

# Density-dominant fixed channel weights (dpyscf: density L_n weight ~20,
# atomization/reaction L_RE ~1, total-energy L_E ~0.01). Energy channels at 1.0,
# density at 20.0, vxc at 1.0. Used when update_scheme="per_molecule" and the
# spec sets no explicit channel_weights.
_DEFAULT_CHANNEL_WEIGHTS = {
    "loss_AE": 1.0,
    "loss_BH76": 1.0,
    "loss_IP13": 1.0,
    "loss_vxc": 1.0,
    "loss_rho": 20.0,
}


def _effective_channel_weights(channel_weights_dict: dict) -> dict:
    """Merge a (possibly partial) user channel_weights over the density-dominant
    defaults: a PARTIAL dict overrides ONLY the channels it names; omitted
    channels inherit :data:`_DEFAULT_CHANNEL_WEIGHTS` (NOT 1.0, which would
    silently de-emphasize e.g. loss_rho from its 20.0 default). An empty dict
    yields the defaults unchanged."""
    return {**_DEFAULT_CHANNEL_WEIGHTS, **dict(channel_weights_dict)}


def _training_groups(spec: TrainingSpec) -> list:
    """Decompose a spec into per-target groups for per-molecule updates.

    One group per BH76 reaction (its reactant/product species), per IP13 pair
    (neutral+cation), per polyatomic AE compound carrying a target, and per
    regularized single-atom anchor. Each group is a dict ``{label, species,
    bh76, ip13}`` where ``species`` is a tuple of MoleculeSpec. Mirrors
    dpyscf's per-molecule loop while reusing the multi-channel loss via scoped
    sub-losses (see :func:`_build_group_loss_and_batch`).
    """
    by_name = {m.name: m for m in spec.molecules}
    targets = spec.targets_dict
    lk = spec.loss_kwargs_dict
    reg_set = set(lk.get("regularize_atom_syms") or ())

    def _n_atoms(m):
        return sum(dict(m.atom_composition).values())

    groups: list = []

    for r in (lk.get("bh76_reactions") or ()):
        names: list = []
        for s in (*r["reactants"], *r["products"]):
            if s not in names:
                names.append(s)
        species = tuple(by_name[n] for n in names if n in by_name)
        groups.append({"label": f"bh76:{r['name']}", "species": species,
                       "bh76": (r,), "ip13": ()})

    for p in (lk.get("ip13_pairs") or ()):
        species = tuple(by_name[n] for n in (p["neutral"], p["cation"])
                        if n in by_name)
        groups.append({"label": f"ip13:{p['name']}", "species": species,
                       "bh76": (), "ip13": (p,)})

    for m in spec.molecules:
        if _n_atoms(m) > 1 and m.name in targets:
            groups.append({"label": f"ae:{m.name}", "species": (m,),
                           "bh76": (), "ip13": ()})

    for m in spec.molecules:
        comp = dict(m.atom_composition)
        # NEUTRAL single atoms only: the Chakravorty anchor table holds
        # neutral ground-state totals, and a charged species (e.g. the Li+
        # of an IP13 pair, element symbol still 'Li') in its own group makes
        # build_indices map atom_map['Li'] -> Li+ -- the scoped regularizer
        # would then pull E_NN(Li+) toward the NEUTRAL value, opposing the
        # IP channel. Cations train through their IP13 group only.
        if (sum(comp.values()) == 1 and next(iter(comp)) in reg_set
                and int(getattr(m, "charge", 0)) == 0):
            groups.append({"label": f"anchor:{m.name}", "species": (m,),
                           "bh76": (), "ip13": ()})

    if not groups:
        raise ValueError(
            "update_scheme='per_molecule': no training groups derived from the "
            "spec (no BH76 reactions, IP13 pairs, AE compounds, or regularized "
            "atom anchors). Check the molecule/target/reaction configuration."
        )
    return groups


def _build_group_loss_and_batch(spec: TrainingSpec, group: dict, batch: dict):
    """Scoped loss + sub-batch for one group. Channels are RAW (vxc/density
    pre-weights forced to 1.0), the outer fixed ``channel_weights`` are the
    sole weighting control in per-molecule mode."""
    name_to_idx = {m.name: i for i, m in enumerate(spec.molecules)}
    species = group["species"]
    sub_mol_data = tuple(batch["mol_data"][name_to_idx[s.name]] for s in species)

    lk = dict(spec.loss_kwargs_dict)
    lk["vxc_weight"] = 1.0
    lk["density_weight"] = 1.0
    lk["bh76_reactions"] = list(group["bh76"])
    lk["ip13_pairs"] = list(group["ip13"])
    lk["solver_config"] = (spec.loss_kwargs_dict.get("solver_config")
                           or spec.solver_config)
    # Scope the atom-anchor allowlist to atoms actually present in this group.
    group_atom_syms = {
        next(iter(dict(s.atom_composition)))
        for s in species if sum(dict(s.atom_composition).values()) == 1
    }
    scoped_reg = tuple(s for s in (spec.loss_kwargs_dict.get(
        "regularize_atom_syms") or ()) if s in group_atom_syms)
    lk["regularize_atom_syms"] = scoped_reg or None
    present = {s.name for s in species}
    if lk.get("aux_only_names"):
        lk["aux_only_names"] = tuple(a for a in lk["aux_only_names"]
                                     if a in present)
    lk.pop("pbe_anchor_weight", None)
    lk.pop("pbe_anchor_sample", None)

    sub_loss = make_loss(spec.loss_name, molecules=species, **lk)
    sub_targets = {s.name: batch["targets"][s.name]
                   for s in species if s.name in batch["targets"]}
    sub_batch = {
        "mol_data": sub_mol_data,
        "targets": sub_targets,
        "atom_energies": batch["atom_energies"],
    }
    return sub_loss, sub_batch


def _run_per_molecule_loop(spec, model, batch, loss, progress_callback):
    """DFS/dpyscf-style stochastic loop: each epoch shuffles the per-target
    groups and takes ONE optimizer step per group with fixed channel weights.
    ``spec.n_steps`` is the number of EPOCHS; total updates = n_steps*n_groups.
    """
    t0 = time.time()
    relative = spec.loss_metric == "relative"
    # Fill omitted channels from the density-dominant defaults (see
    # _effective_channel_weights): a PARTIAL channel_weights dict overrides ONLY
    # the channels it names; previously a partial dict bypassed the defaults and
    # omitted channels fell back to 1.0 (silently de-emphasizing loss_rho).
    cw = _effective_channel_weights(spec.channel_weights_dict)
    groups = _training_groups(spec)
    n_groups = len(groups)
    n_epochs = spec.n_steps
    total_updates = max(1, n_epochs * n_groups)

    optimizer = build_optimizer(
        lr_start=spec.lr_start, lr_end=spec.lr_end,
        n_steps=total_updates, lr_decay_start=spec.lr_decay_start,
        grad_clip=spec.grad_clip,
        weight_decay=spec.weight_decay,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    progress_hook = _adapt_progress_callback(
        progress_callback, arch=spec.arch.name, phase="train")

    # WS5: periodic-resume checkpointing. checkpoint_every<=0 (default) => the
    # whole mechanism is OFF and the loop below is byte-identical to pre-WS5.
    checkpoint_every = int(getattr(spec, "checkpoint_every", 0))
    resume_enabled = checkpoint_every > 0
    checkpoint_dir = spec.checkpoint_dir

    @eqx.filter_jit
    def _step(model, opt_state, gbatch, gloss):
        def scalar_loss(m):
            comps = gloss.compute_components(m, gbatch, relative=relative)
            total = jnp.array(0.0)
            for k, v in comps.items():
                total = total + cw.get(k, 1.0) * v
            return total, comps
        (loss_val, comps), grads = eqx.filter_value_and_grad(
            scalar_loss, has_aux=True)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val, comps

    prepared = [
        (g["label"], *_build_group_loss_and_batch(spec, g, batch))
        for g in groups
    ]

    rng = np.random.RandomState(spec.seed)
    order = np.arange(n_groups)
    losses_list: list = []
    aux_log: list = []
    tracker = _BestModelTracker(window=n_groups)  # epoch-scale trailing mean
    update = 0
    loss_py = float("nan")

    # WS3: held-out validation -> early-stop + validation-best snapshot. A no-op
    # when validate_every<=0 / no val data (_build_validation_data -> None,None):
    # the loop is then byte-identical to before. solver_config matches training
    # so the val energy path == the loss energy path.
    val_every = int(getattr(spec, "validate_every", 0))
    val_mol_data, val_reactions = _build_validation_data(spec)
    val_enabled = val_every > 0 and val_mol_data is not None
    val_tracker = _BestValidationTracker() if val_enabled else None
    val_solver_config = (spec.loss_kwargs_dict.get("solver_config")
                         or spec.solver_config)
    early_stopped = False
    epochs_run = 0

    # WS5: RESUME from the last checkpoint if one is present and the run is not
    # already complete. The opt_state skeleton is the fresh-init opt_state above;
    # the model skeleton is the current (fresh/pretrained) model. Restoring
    # opt_state carries the adamw step count so the LR schedule continues;
    # restoring rng_state AND the per-epoch `order` permutation makes the next
    # epoch shuffle identically to a never-killed run (a multi-group epoch's
    # shuffle result depends on BOTH the rng draws AND the array's starting
    # arrangement, so `order` MUST be restored to the killed run's last
    # permutation -- see WS5-RESUME-01). start_epoch = saved epoch (1-based
    # count) -> range resumes at the NEXT epoch index without re-walking groups.
    #
    # WS5-SIG-3: the load is GUARDED. A corrupt/truncated resume_state.pkl or a
    # missing resume_*.eqx must not crash the task -- on ANY load error we log a
    # warning and START FRESH (treat as no resume). The whole restore is staged
    # through `restored` (one call) and applied only on success, so a failed load
    # never leaves half-restored state.
    start_epoch = 0
    if resume_enabled and _has_resume_checkpoint(checkpoint_dir):
        try:
            restored = _load_resume_checkpoint(
                checkpoint_dir, model_skeleton=model, opt_state_skeleton=opt_state)
        except Exception as exc:  # noqa: BLE001 -- corrupt ckpt -> start fresh
            warnings.warn(
                f"WS5: could not load resume checkpoint in {checkpoint_dir} "
                f"({type(exc).__name__}: {exc}); starting training FRESH.",
                RuntimeWarning, stacklevel=2,
            )
        else:
            model = restored["model"]
            opt_state = restored["opt_state"]
            rng.set_state(restored["rng_state"])
            order[:] = restored["order"]     # continue the killed run's perm
            tracker = restored["train_tracker"]
            losses_list = restored["losses"]
            aux_log = restored["aux_log"]
            early_stopped = restored["early_stopped"]
            update = restored["update"]
            start_epoch = restored["epoch"]      # 1-based completed-epoch count
            epochs_run = restored["epoch"]
            if val_enabled and restored["val_tracker"] is not None:
                val_tracker = restored["val_tracker"]

    # WS5: an epoch-boundary SNAPSHOT of the loop state for the SIGTERM flush. The
    # worker's signal handler calls _flush_live() (registered below) to write a
    # resume checkpoint. CRITICAL (WS5-RESUME-02/WS5-SIG-1): _live holds PLAIN
    # COPIES captured ONLY at epoch boundaries -- NEVER advanced mid-epoch. So a
    # mid-epoch flush writes the LAST COMPLETED epoch's self-consistent snapshot
    # (stale but exact-on-resume), never a TORN one (advanced rng/losses paired
    # with a stale epoch/model). _capture_live() is the single boundary-copy
    # point; it is called to seed _live before the loop and again at each epoch
    # end. Seeded with the resume/initial state so a flush before the first epoch
    # completes still writes a consistent (no-progress) checkpoint.
    _live: dict = {}

    def _capture_live():
        """Snapshot the CURRENT (epoch-boundary) loop state into _live as plain
        copies. Must only be called when the state is self-consistent (before the
        loop, or at an epoch end), never mid-epoch."""
        _live["model"] = model
        _live["opt_state"] = opt_state
        _live["rng_state"] = rng.get_state()
        _live["order"] = list(order)
        _live["epoch"] = epochs_run
        _live["update"] = update
        _live["losses"] = list(losses_list)
        _live["aux_log"] = list(aux_log)
        _live["early_stopped"] = early_stopped
        _live["train_best_loss"] = tracker.best_loss
        _live["train_recent"] = list(tracker._recent)
        _live["train_window"] = tracker.window
        _live["train_best_model"] = tracker.best_model
        _live["val_present"] = val_tracker is not None
        _live["val_best_mae"] = (val_tracker.best_mae
                                 if val_tracker is not None else None)
        _live["val_finite_metrics"] = (list(val_tracker._finite_metrics)
                                       if val_tracker is not None else None)
        _live["val_best_model"] = (val_tracker.best_model
                                   if val_tracker is not None else None)

    def _flush_live():
        if not resume_enabled or not _live:
            return
        _write_resume_checkpoint(
            checkpoint_dir, model=_live["model"], opt_state=_live["opt_state"],
            rng_state=_live["rng_state"], order=_live["order"],
            train_best_loss=_live["train_best_loss"],
            train_recent=_live["train_recent"],
            train_window=_live["train_window"],
            train_best_model=_live["train_best_model"],
            val_present=_live["val_present"], val_best_mae=_live["val_best_mae"],
            val_finite_metrics=_live["val_finite_metrics"],
            val_best_model=_live["val_best_model"],
            epoch=_live["epoch"], update=_live["update"],
            losses=_live["losses"], aux_log=_live["aux_log"],
            early_stopped=_live["early_stopped"])

    if resume_enabled:
        _capture_live()                  # seed with the resume/initial boundary
        _register_resume_flusher(_flush_live)

    # WS5-SIG-2/SIG-5: the epoch loop AND the completion sequence run under a
    # try/finally that ALWAYS clears the registered flusher -- a raised exception
    # (e.g. _abort_if_nonfinite) or an interrupt during _save_artifacts must not
    # leave a stale flusher pointing at a finished/aborted dir for the next run in
    # the same process.
    try:
        for epoch in range(start_epoch, n_epochs):
            rng.shuffle(order)
            for gi in order:
                label, gloss, gbatch = prepared[gi]
                model, opt_state, loss_val, comps = _step(
                    model, opt_state, gbatch, gloss)
                loss_py = float(loss_val)
                _abort_if_nonfinite(loss_val, comps, loop="per_molecule",
                                    step=update, group=label)
                losses_list.append(loss_py)
                tracker.update(loss_py, model)
                aux_log.append({
                    "step": update, "epoch": epoch, "group": label,
                    "loss": loss_py,
                    "aux": {k: float(v) for k, v in comps.items()},
                    "update_scheme": "per_molecule",
                })
                update += 1
            epochs_run = epoch + 1
            if progress_hook is not None:
                progress_hook(epoch + 1, n_epochs, loss_py)
            # Validation check every `validate_every` epochs.
            if val_enabled and (epoch + 1) % val_every == 0:
                val_mae = _validation_reaction_mae(
                    model, val_mol_data, val_reactions,
                    solver_config=val_solver_config)
                val_tracker.update(val_mae, model)
                aux_log.append({
                    "step": update, "epoch": epoch, "group": "__validation__",
                    "val_mae_kcalmol": (float(val_mae)
                                        if np.isfinite(val_mae) else None),
                    "update_scheme": "per_molecule",
                })
                if val_tracker.should_stop(spec.patience,
                                           spec.early_stop_min_delta):
                    early_stopped = True
                    # Capture the early-stop boundary before leaving the loop so
                    # a flush in flight reflects it.
                    if resume_enabled:
                        _capture_live()
                    break
            # WS5: snapshot the JUST-COMPLETED epoch into _live (for the SIGTERM
            # flush) and write a PERIODIC resume checkpoint every checkpoint_every
            # epochs. epoch is 0-based; epochs_run = epoch+1 is the completed
            # count persisted as `epoch` so resume continues at
            # range(epochs_run, n_epochs). `order` now holds THIS epoch's
            # permutation -- the arrangement the resumed run must shuffle FROM.
            if resume_enabled:
                _capture_live()
                if epochs_run % checkpoint_every == 0:
                    _flush_live()

        duration = time.time() - t0
        # FIX 3 (WS3-ESV-1): only emit the validation extras when validation ran;
        # a per_molecule run with validate_every=0 then writes the PRE-WS3
        # metadata key set byte-identically (no early_stopped / val_* keys).
        extra_metadata = None
        val_best_model = None
        if val_enabled:
            extra_metadata = {
                "epochs_run": epochs_run,
                "n_epochs_configured": n_epochs,
                "early_stopped": early_stopped,
                "validate_every": val_every,
                "patience": int(getattr(spec, "patience", 0)),
                "val_best_mae": (float(val_tracker.best_mae)
                                 if val_tracker is not None
                                 and np.isfinite(val_tracker.best_mae)
                                 else None),
            }
            val_best_model = val_tracker.best_model
        metadata = _save_artifacts(
            spec, model, losses_list, aux_log, duration,
            best_model=tracker.best_model,
            val_best_model=val_best_model,
            extra_metadata=extra_metadata)
        # WS5/WS6: the run is COMPLETE (clean or early-stopped). _save_artifacts
        # has written model.eqx (the harness success signal); now drop the
        # completion.json sentinel and delete the resume_* set. No-op cleanup when
        # checkpoint_every<=0 (no resume files were ever written).
        if resume_enabled:
            _finalize_completion(checkpoint_dir, early_stopped=early_stopped,
                                 epochs_run=epochs_run)
    finally:
        # ALWAYS clear the flusher: on clean return, on early-stop, or on a raised
        # exception. A stale flusher must never survive into the next run.
        if resume_enabled:
            _clear_resume_flusher()
    return metadata


# ---------------------------------------------------------------------------
# run_training -- dispatcher
# ---------------------------------------------------------------------------

def run_training(spec: TrainingSpec, progress_callback=None) -> dict:
    """Train AlecGGAModel end-to-end. Dispatches to strategy-specific loop."""
    spec.validate()
    model = _build_model(spec)
    loss = make_loss(
        spec.loss_name,
        molecules=spec.molecules,
        pbe_anchor_weight=spec.pbe_anchor_weight,
        pbe_anchor_sample=spec.pbe_anchor_sample,
        **spec.loss_kwargs_dict,
    )
    batch = _build_batch(spec, loss)

    # DFS/dpyscf-style per-molecule stochastic updates: one optimizer step per
    # target-group per epoch with fixed channel weights (ignores `balancing`,
    # whose GradNorm rebalancing is a full-batch construct).
    if getattr(spec, "update_scheme", "batched") == "per_molecule":
        return _run_per_molecule_loop(spec, model, batch, loss, progress_callback)

    balancing = spec.balancing

    if balancing is None or type(balancing).__name__ == "BalancingConfig":
        return _run_static_loop(spec, model, batch, loss, progress_callback)

    from xcquinox.alec.balancing import (
        LossNormConfig, TwoPhaseConfig, GradNormConfig,
    )
    if isinstance(balancing, LossNormConfig):
        return _run_lossnorm_loop(spec, model, batch, loss, progress_callback)
    elif isinstance(balancing, TwoPhaseConfig):
        return _run_twophase_loop(spec, model, batch, loss, progress_callback)
    elif isinstance(balancing, GradNormConfig):
        return _run_gradnorm_loop(spec, model, batch, loss, progress_callback)
    else:
        raise TypeError(f"Unknown balancing config type: {type(balancing)}")
