#!/usr/bin/env python3
"""
Parallel Pre-training Worker for GGA Neural Network Functionals

This script pre-trains a single architecture's exchange and correlation networks.
It's designed to be called from the notebook via subprocess for clean
process isolation.

Usage:
    python parallel_pretrain_worker.py --arch deep --steps 500 \
        --data-dir /path/to/pretrain_data --checkpoint-base /path/to/checkpoints

Output:
    - Saves pre-trained networks to CHECKPOINT_BASE/01_pretrain/arch/
    - Prints JSON result to stdout for the parent process to capture
"""

import os
import sys
import argparse
import pickle
import json
import time
import traceback
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description='Pre-train a single GGA network pair')
    parser.add_argument('--arch', required=True, help='Architecture name')
    parser.add_argument('--steps', type=int, default=500, help='Pre-training steps')
    parser.add_argument('--data-dir', required=True, help='Directory with pre-computed data')
    parser.add_argument('--checkpoint-base', required=True, help='Base checkpoint directory')
    parser.add_argument('--threads', type=int, default=4, help='Threads per worker')
    parser.add_argument('--lr-start', type=float, default=1e-2, help='Initial learning rate')
    parser.add_argument('--lr-end', type=float, default=1e-4, help='Final learning rate')
    parser.add_argument('--lr-decay-start', type=float, default=0.0,
                        help='Fraction of steps before LR decay begins (0.0-1.0). 0=immediate decay')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping norm')
    args = parser.parse_args()

    # Set thread limits BEFORE importing JAX
    # Compile trim only: the eigen token is measured inert on jaxlib 0.7.0, the
    # pool bound is the launcher-side CPU affinity, and the old
    # intra_op_parallelism_threads token was mis-prefixed (no --xla_) so XLA
    # ignored it.
    os.environ['XLA_FLAGS'] = '--xla_llvm_disable_expensive_passes=true'
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.threads)

    start_time = time.time()
    result = {'status': 'failed', 'arch': args.arch}

    try:
        # Import after setting environment
        import jax
        import jax.numpy as jnp
        import numpy as np
        import equinox as eqx
        import optax

        # Match notebook's JAX configuration
        jax.config.update("jax_enable_x64", True)
        jax.config.update('jax_platform_name', 'cpu')

        import xcquinox as xce

        # =====================================================================
        # Load pre-computed data
        # =====================================================================

        print(f"Loading pre-training data from {args.data_dir}...", file=sys.stderr)

        with open(os.path.join(args.data_dir, 'pretrain_data.pkl'), 'rb') as f:
            pretrain_data = pickle.load(f)

        rho_all = jnp.array(pretrain_data['rho_all'])
        sigma_all = jnp.array(pretrain_data['sigma_all'])
        Fx_all = jnp.array(pretrain_data['Fx_all'])
        Fc_all = jnp.array(pretrain_data['Fc_all'])
        cusp_all = jnp.array(pretrain_data['cusp_all'])
        dm_all = jnp.array(pretrain_data['dm_all'])

        # Diagnostic: print data statistics
        print(f"  Data loaded: {len(rho_all)} points", file=sys.stderr)
        print(f"  rho: [{float(rho_all.min()):.2e}, {float(rho_all.max()):.2e}]", file=sys.stderr)
        print(f"  sigma: [{float(sigma_all.min()):.2e}, {float(sigma_all.max()):.2e}]", file=sys.stderr)
        print(f"  Fx-1: [{float(Fx_all.min()):.4f}, {float(Fx_all.max()):.4f}]", file=sys.stderr)
        print(f"  Fc-1: [{float(Fc_all.min()):.4f}, {float(Fc_all.max()):.4f}]", file=sys.stderr)
        print(f"  dtype: {rho_all.dtype}", file=sys.stderr)

        with open(os.path.join(args.data_dir, 'architectures.pkl'), 'rb') as f:
            ARCHITECTURES = pickle.load(f)

        # Use checkpoint_base argument directly instead of pickle to ensure path consistency
        # Convert to absolute path to avoid working directory issues with subprocess
        # The pretrain directory is always at checkpoint_base/01_pretrain/
        checkpoint_base_abs = os.path.abspath(args.checkpoint_base)
        PRETRAIN_DIR = os.path.join(checkpoint_base_abs, '01_pretrain')

        arch_config = ARCHITECTURES[args.arch]
        print(f"Pre-training {args.arch}: {arch_config}", file=sys.stderr)
        print(f"  Checkpoint base: {checkpoint_base_abs}", file=sys.stderr)
        print(f"  Pretrain dir: {PRETRAIN_DIR}", file=sys.stderr)

        # =====================================================================
        # Create network pair
        # =====================================================================

        def create_network_pair(arch_name, seed=42):
            config = ARCHITECTURES[arch_name]
            depth = config['depth']
            nodes = config['nodes']
            use_attn = config['use_self_attention']
            use_cusp = config.get('use_cusp', False)
            use_dm = config.get('use_dm_features', False)
            net_type = config.get('net_type', 'basic')

            if net_type == 'extended' or use_cusp or use_dm:
                # Use non-transform extended networks (better pre-training)
                xnet = xce.net.GGA_FxNet_extended(
                    depth=depth, nodes=nodes, seed=seed, lower_rho_cutoff=1e-6,
                    use_self_attention=use_attn, use_cusp=use_cusp,
                    use_dm_features=use_dm, use_laplacian=False, n_dm_features=3
                )
                cnet = xce.net.GGA_FcNet_extended(
                    depth=depth, nodes=nodes, seed=seed+1, lower_rho_cutoff=1e-6,
                    use_self_attention=use_attn, use_cusp=use_cusp,
                    use_dm_features=use_dm, use_laplacian=False, n_dm_features=3
                )
            else:
                # Use non-transform versions to match step3 (better pre-training)
                xnet = xce.net.GGA_FxNet_sigma(
                    depth=depth, nodes=nodes, seed=seed, lower_rho_cutoff=1e-6,
                    use_self_attention=use_attn
                )
                cnet = xce.net.GGA_FcNet_sigma(
                    depth=depth, nodes=nodes, seed=seed+1, lower_rho_cutoff=1e-6,
                    use_self_attention=use_attn
                )
            return xnet, cnet

        xnet, cnet = create_network_pair(args.arch)

        # =====================================================================
        # PretrainLoss class - STATELESS like step2 notebook
        # =====================================================================

        class PretrainLoss(eqx.Module):
            """
            MSE loss for pre-training enhancement factor networks.

            STATELESS: receives descriptors and targets as arguments to __call__,
            not stored as instance attributes. This is important for proper
            gradient computation with eqx.filter_value_and_grad.
            """
            def __call__(self, model, descriptors, ref_F):
                """
                Compute MSE loss between predicted and reference enhancement factors.

                :param model: GGA network
                :param descriptors: Input descriptors shape (N, F) where F depends on features
                :param ref_F: Reference enhancement factors shape (N,)
                :return: MSE loss
                """
                pred = jax.vmap(model)(descriptors).squeeze()
                pred = pred - 1.0  # Networks output 1 + enhancement, targets are enhancement
                return jnp.mean((pred - ref_F)**2)

        # =====================================================================
        # Training
        # =====================================================================

        # Determine features based on architecture
        use_cusp = arch_config.get('use_cusp', False)
        use_dm = arch_config.get('use_dm_features', False)

        # Build descriptors array based on architecture features
        # Order must match network's __call__ expectation: dm BEFORE cusp
        input_list = [rho_all, sigma_all]
        if use_dm:
            for i in range(dm_all.shape[1]):
                input_list.append(dm_all[:, i])
        if use_cusp:
            input_list.append(cusp_all[:, 0])
            input_list.append(cusp_all[:, 1])

        descriptors = jnp.stack(input_list, axis=1)
        print(f"  Descriptors shape: {descriptors.shape}", file=sys.stderr)

        # Checkpoint directory - use PRETRAIN_DIR constructed from args.checkpoint_base
        arch_dir = os.path.join(PRETRAIN_DIR, args.arch)
        os.makedirs(arch_dir, exist_ok=True)
        print(f"  Saving to: {arch_dir}", file=sys.stderr)

        # Stateless loss function
        loss_fn = PretrainLoss()

        # Progress file for real-time monitoring
        progress_file = os.path.join(arch_dir, 'progress.json')

        def write_progress(phase, step, total, loss):
            """Write progress to file for notebook to poll."""
            progress = {
                'arch': args.arch,
                'phase': phase,
                'step': step,
                'total': total,
                'loss': float(loss),
                'timestamp': time.time()
            }
            try:
                with open(progress_file, 'w') as f:
                    json.dump(progress, f)
            except:
                pass  # Don't fail on progress write errors

        # Train exchange network
        print(f"Training exchange network...", file=sys.stderr)
        initial_loss_x = float(loss_fn(xnet, descriptors, Fx_all))
        print(f"  Initial X loss: {initial_loss_x:.6f}", file=sys.stderr)
        write_progress('X', 0, args.steps, initial_loss_x)

        # Progress callback for X training
        def x_progress_callback(step, total, loss):
            write_progress('X', step, total, loss)

        # Learning rate schedule with optional warmup/constant phase
        decay_start_step = int(args.lr_decay_start * args.steps)
        decay_steps = args.steps - decay_start_step

        if decay_start_step > 0 and decay_steps > 0:
            # Constant LR for first phase, then linear decay
            lr_schedule_x = optax.join_schedules(
                schedules=[
                    optax.constant_schedule(args.lr_start),
                    optax.linear_schedule(
                        init_value=args.lr_start,
                        end_value=args.lr_end,
                        transition_steps=decay_steps
                    )
                ],
                boundaries=[decay_start_step]
            )
        else:
            # Immediate linear decay (original behavior)
            lr_schedule_x = optax.linear_schedule(
                init_value=args.lr_start,
                end_value=args.lr_end,
                transition_steps=args.steps
            )

        optimizer_x = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adam(learning_rate=lr_schedule_x)
        )

        trainer_x = xce.train.xcTrainer(
            model=xnet, optim=optimizer_x,
            loss=loss_fn, steps=args.steps, do_jit=True,
            serialize_every=max(50, args.steps // 10),
            checkpoint_dir=arch_dir,
            progress_callback=x_progress_callback
        )
        xnet_trained, losses_x = trainer_x(1, [descriptors], [Fx_all])

        # Train correlation network
        print(f"Training correlation network...", file=sys.stderr)
        initial_loss_c = float(loss_fn(cnet, descriptors, Fc_all))
        print(f"  Initial C loss: {initial_loss_c:.6f}", file=sys.stderr)
        write_progress('C', 0, args.steps, initial_loss_c)

        # Progress callback for C training
        def c_progress_callback(step, total, loss):
            write_progress('C', step, total, loss)

        # Learning rate schedule with optional warmup/constant phase (same as X)
        if decay_start_step > 0 and decay_steps > 0:
            lr_schedule_c = optax.join_schedules(
                schedules=[
                    optax.constant_schedule(args.lr_start),
                    optax.linear_schedule(
                        init_value=args.lr_start,
                        end_value=args.lr_end,
                        transition_steps=decay_steps
                    )
                ],
                boundaries=[decay_start_step]
            )
        else:
            lr_schedule_c = optax.linear_schedule(
                init_value=args.lr_start,
                end_value=args.lr_end,
                transition_steps=args.steps
            )

        optimizer_c = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adam(learning_rate=lr_schedule_c)
        )

        trainer_c = xce.train.xcTrainer(
            model=cnet, optim=optimizer_c,
            loss=loss_fn, steps=args.steps, do_jit=True,
            serialize_every=max(50, args.steps // 10),
            checkpoint_dir=arch_dir,
            progress_callback=c_progress_callback
        )
        cnet_trained, losses_c = trainer_c(1, [descriptors], [Fc_all])

        # =====================================================================
        # Save results
        # =====================================================================

        eqx.tree_serialise_leaves(os.path.join(arch_dir, 'xnet.eqx'), xnet_trained)
        eqx.tree_serialise_leaves(os.path.join(arch_dir, 'cnet.eqx'), cnet_trained)

        # Save losses with verification
        losses_x_arr = np.array(losses_x)
        losses_c_arr = np.array(losses_c)
        losses_x_path = os.path.join(arch_dir, 'losses_x.npy')
        losses_c_path = os.path.join(arch_dir, 'losses_c.npy')
        np.save(losses_x_path, losses_x_arr)
        np.save(losses_c_path, losses_c_arr)
        print(f"  Saved losses_x: {losses_x_path} (shape={losses_x_arr.shape})", file=sys.stderr)
        print(f"  Saved losses_c: {losses_c_path} (shape={losses_c_arr.shape})", file=sys.stderr)

        metadata = {
            'arch_name': args.arch,
            'pretrain_steps': args.steps,
            'lr_start': args.lr_start,
            'lr_end': args.lr_end,
            'lr_decay_start': args.lr_decay_start,
            'grad_clip': args.grad_clip,
            'final_loss_x': float(losses_x[-1]),
            'final_loss_c': float(losses_c[-1]),
            'min_loss_x': float(min(losses_x)),
            'min_loss_c': float(min(losses_c)),
            'use_cusp': use_cusp,
            'use_dm': use_dm,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time,
        }
        with open(os.path.join(arch_dir, 'pretrain_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved to {arch_dir}", file=sys.stderr)

        result = {
            'status': 'success',
            'arch': args.arch,
            'final_loss_x': float(losses_x[-1]),
            'final_loss_c': float(losses_c[-1]),
            'min_loss_x': float(min(losses_x)),
            'min_loss_c': float(min(losses_c)),
            'duration': time.time() - start_time,
            'checkpoint_dir': arch_dir,
        }

    except Exception as e:
        result = {
            'status': 'failed',
            'arch': args.arch,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'duration': time.time() - start_time,
        }
        print(f"ERROR: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

    # Output result as JSON to stdout
    print(json.dumps(result))
    return 0 if result['status'] == 'success' else 1


if __name__ == '__main__':
    sys.exit(main())
