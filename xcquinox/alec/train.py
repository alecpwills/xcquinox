"""xcquinox.alec.train -- custom training loop for AlecGGAModel.

Implements THE SPEC Task 5.2: run_training using eqx.filter_value_and_grad
with has_aux=True (xcTrainer does not support aux dicts).

Public API:
  build_optimizer  -- canonical optimizer chain (shared with pretrain)
  run_training     -- full training loop with checkpoint saving

Internal:
  _adapt_progress_callback -- wraps user callback for 3-arg xcTrainer form
  _train_step              -- JIT-compiled single training step
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
from xcquinox.alec.solver import SolverConfig
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
# run_training
# ---------------------------------------------------------------------------

def run_training(spec: TrainingSpec, progress_callback=None) -> dict:
    """Train AlecGGAModel end-to-end on molecular data.

    Steps:
    1. Validate spec
    2. Build model (from scratch or pretrain checkpoint)
    3. Precompute mol_data with full required-keys union
    4. Build batch dict
    5. Build optimizer
    6. Training loop via _train_step
    7. Save artifacts (model.eqx, losses.npy, aux_log.pkl, train_metadata.json)
    8. Return metadata dict

    Parameters
    ----------
    spec : TrainingSpec
        Full training configuration.
    progress_callback : callable, optional
        Called with a dict payload at each step.

    Returns
    -------
    dict
        Metadata dict (also saved as train_metadata.json).
    """
    t0 = time.time()

    # Step 1: validate
    spec.validate()

    # Step 2: build model
    if spec.pretrain_checkpoint is None:
        model = AlecGGAModel.from_arch(spec.arch, seed=spec.seed)
    else:
        xnet_skeleton, cnet_skeleton = create_network_pair(spec.arch, seed=spec.seed)
        loaded_xnet = eqx.tree_deserialise_leaves(
            os.path.join(spec.pretrain_checkpoint, "xnet.eqx"), xnet_skeleton
        )
        loaded_cnet = eqx.tree_deserialise_leaves(
            os.path.join(spec.pretrain_checkpoint, "cnet.eqx"), cnet_skeleton
        )
        model = AlecGGAModel.from_arch(spec.arch, xnet=loaded_xnet, cnet=loaded_cnet)

    # Step 3: precompute mol_data with full required-keys union
    loss = make_loss(spec.loss_name, molecules=spec.molecules, **spec.loss_kwargs_dict)

    required = set()
    required |= set(loss.required_mol_keys)
    for d in spec.arch.materialize_descriptors():
        required |= set(d.required_mol_keys)
    required_keys = tuple(required)

    mol_data_list = [
        precompute_fixed_density_data(
            m, required_keys=required_keys,
            descriptors=spec.arch.materialize_descriptors(),
        )
        for m in spec.molecules
    ]

    # Step 4: build batch dict
    batch = {
        "mol_data": tuple(mol_data_list),
        "targets": spec.targets_dict,
        "atom_energies": spec.atom_energies_dict,
    }

    # Step 5: build optimizer
    optimizer = build_optimizer(
        lr_start=spec.lr_start,
        lr_end=spec.lr_end,
        n_steps=spec.n_steps,
        lr_decay_start=spec.lr_decay_start,
        grad_clip=spec.grad_clip,
    )

    # Step 6: training loop
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
            progress_hook(step + 1, spec.n_steps, loss_py)  # 1-based

    duration = time.time() - t0

    # Step 7: save artifacts
    os.makedirs(spec.checkpoint_dir, exist_ok=True)

    model_path = os.path.join(spec.checkpoint_dir, "model.eqx")
    eqx.tree_serialise_leaves(model_path, model)

    losses_np = np.array(losses, dtype=np.float64)
    np.save(os.path.join(spec.checkpoint_dir, "losses.npy"), losses_np)

    with open(os.path.join(spec.checkpoint_dir, "aux_log.pkl"), "wb") as f:
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
        "final_loss": float(losses_np[-1]) if len(losses_np) > 0 else float("nan"),
        "min_loss": float(np.min(losses_np)) if len(losses_np) > 0 else float("nan"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "duration_seconds": round(duration, 1),
    }

    md_path = os.path.join(spec.checkpoint_dir, "train_metadata.json")
    with open(md_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Step 8: return metadata
    return metadata
