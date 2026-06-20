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


def _save_artifacts(spec, model, losses, aux_log, duration, best_model=None) -> dict:
    """Save model.eqx (final), losses.npy, aux_log, train_metadata.json. If a
    best-loss snapshot is given, ALSO write model_best.eqx side-by-side (the
    final model.eqx is still written) so eval can opt into the pre-instability
    checkpoint."""
    os.makedirs(spec.checkpoint_dir, exist_ok=True)

    model_path = os.path.join(spec.checkpoint_dir, "model.eqx")
    eqx.tree_serialise_leaves(model_path, model)
    if best_model is not None:
        eqx.tree_serialise_leaves(
            os.path.join(spec.checkpoint_dir, "model_best.eqx"), best_model)

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

    md_path = os.path.join(spec.checkpoint_dir, "train_metadata.json")
    with open(md_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


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
    for epoch in range(n_epochs):
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
        if progress_hook is not None:
            progress_hook(epoch + 1, n_epochs, loss_py)

    duration = time.time() - t0
    return _save_artifacts(spec, model, losses_list, aux_log, duration,
                           best_model=tracker.best_model)


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
