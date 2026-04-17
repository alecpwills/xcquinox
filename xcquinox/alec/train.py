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
  _run_gradnorm_loop       -- stub for Task 8
"""
import json
import os
import pickle  # noqa: S403 — saving trusted aux_log data only
import time

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
) -> optax.GradientTransformation:
    """Build canonical optimizer chain for training.

    Chain order: clip_by_global_norm -> adam(lr_schedule).
    LR schedule: optional constant warmup then linear decay.

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
        optax.adam(learning_rate=lr_schedule),
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
    loaded_xnet = eqx.tree_deserialise_leaves(
        os.path.join(spec.pretrain_checkpoint, "xnet.eqx"), xnet_skeleton
    )
    loaded_cnet = eqx.tree_deserialise_leaves(
        os.path.join(spec.pretrain_checkpoint, "cnet.eqx"), cnet_skeleton
    )
    return AlecGGAModel.from_arch(spec.arch, xnet=loaded_xnet, cnet=loaded_cnet)


def _build_batch(spec: TrainingSpec, loss) -> dict:
    """Precompute mol_data and build batch dict."""
    required = set()
    required |= set(loss.required_mol_keys)
    for d in spec.arch.materialize_descriptors():
        required |= set(d.required_mol_keys)

    sc = spec.loss_kwargs_dict.get("solver_config") or spec.solver_config
    if isinstance(sc, SolverConfig) and sc.mode == SolverMode.FULL:
        required.add("eri")

    required_keys = tuple(required)
    mol_data_list = [
        precompute_fixed_density_data(
            m, required_keys=required_keys,
            descriptors=spec.arch.materialize_descriptors(),
        )
        for m in spec.molecules
    ]
    return {
        "mol_data": tuple(mol_data_list),
        "targets": spec.targets_dict,
        "atom_energies": spec.atom_energies_dict,
    }


def _save_artifacts(spec, model, losses, aux_log, duration) -> dict:
    """Save model.eqx, losses.npy, aux_log data, train_metadata.json."""
    os.makedirs(spec.checkpoint_dir, exist_ok=True)

    model_path = os.path.join(spec.checkpoint_dir, "model.eqx")
    eqx.tree_serialise_leaves(model_path, model)

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
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    losses = []
    aux_log = []
    progress_hook = _adapt_progress_callback(
        progress_callback, arch=spec.arch.name, phase="train"
    )

    for step in range(spec.n_steps):
        model, opt_state, loss_value, aux = _train_step(
            model, opt_state, batch, loss, optimizer
        )
        loss_py = float(loss_value)
        losses.append(loss_py)
        aux_log.append({"step": step, "loss": loss_py, "aux": aux})
        if progress_hook is not None:
            progress_hook(step + 1, spec.n_steps, loss_py)

    duration = time.time() - t0
    return _save_artifacts(spec, model, losses, aux_log, duration)


def _run_lossnorm_loop(spec, model, batch, loss, progress_callback):
    """Loss normalization: divide each component by its step-0 magnitude."""
    t0 = time.time()
    optimizer = build_optimizer(
        lr_start=spec.lr_start, lr_end=spec.lr_end,
        n_steps=spec.n_steps, lr_decay_start=spec.lr_decay_start,
        grad_clip=spec.grad_clip,
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
    for step in range(spec.n_steps):
        model, opt_state, loss_value, components = _normed_step(
            model, opt_state, batch)
        loss_py = float(loss_value)
        losses.append(loss_py)
        eff_weights = {k: float(1.0 / norms[k]) for k in component_keys}
        aux_log.append({
            "step": step, "loss": loss_py, "aux": components,
            "balancing_info": {"strategy": "loss_norm", "effective_weights": eff_weights},
        })
        if progress_hook is not None:
            progress_hook(step + 1, spec.n_steps, loss_py)

    duration = time.time() - t0
    return _save_artifacts(spec, model, losses, aux_log, duration)


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

    # Phase 1: energy-only loss
    phase1_kwargs = _filter_loss_kwargs(spec.loss_kwargs_dict, balancing.phase1_loss)
    phase1_loss = make_loss(
        balancing.phase1_loss, molecules=spec.molecules, **phase1_kwargs)
    phase1_optimizer = build_optimizer(
        lr_start=spec.lr_start, lr_end=spec.lr_end,
        n_steps=balancing.phase1_steps,
        lr_decay_start=spec.lr_decay_start, grad_clip=spec.grad_clip,
    )
    opt_state = phase1_optimizer.init(eqx.filter(model, eqx.is_array))

    for step in range(balancing.phase1_steps):
        model, opt_state, loss_value, aux = _train_step(
            model, opt_state, batch, phase1_loss, phase1_optimizer)
        loss_py = float(loss_value)
        losses.append(loss_py)
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
    )
    opt_state = phase2_optimizer.init(eqx.filter(model, eqx.is_array))

    for step in range(phase2_steps):
        global_step = balancing.phase1_steps + step
        model, opt_state, loss_value, aux = _train_step(
            model, opt_state, batch, loss, phase2_optimizer)
        loss_py = float(loss_value)
        losses.append(loss_py)
        aux_log.append({
            "step": global_step, "loss": loss_py, "aux": aux,
            "balancing_info": {"strategy": "two_phase", "phase": 2},
        })
        if progress_hook is not None:
            progress_hook(global_step + 1, spec.n_steps, loss_py)

    duration = time.time() - t0
    return _save_artifacts(spec, model, losses, aux_log, duration)


def _run_gradnorm_loop(spec, model, batch, loss, progress_callback):
    raise NotImplementedError("GradNorm loop -- implemented in Task 8")


# ---------------------------------------------------------------------------
# run_training -- dispatcher
# ---------------------------------------------------------------------------

def run_training(spec: TrainingSpec, progress_callback=None) -> dict:
    """Train AlecGGAModel end-to-end. Dispatches to strategy-specific loop."""
    spec.validate()
    model = _build_model(spec)
    loss = make_loss(spec.loss_name, molecules=spec.molecules, **spec.loss_kwargs_dict)
    batch = _build_batch(spec, loss)
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
