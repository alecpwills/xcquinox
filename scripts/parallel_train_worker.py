#!/usr/bin/env python3
"""
Parallel Training Worker for GGA Neural Network Functionals

This script trains a single (architecture, approach) combination.
It's designed to be called from the notebook via subprocess for clean
process isolation.

Usage:
    python parallel_train_worker.py --arch deep --approach A_ae_only \
        --steps 50 --data-dir /path/to/parallel_data --checkpoint-base /path/to/checkpoints

Output:
    - Saves model checkpoints to CHECKPOINT_DIRS[approach]/arch/
    - Prints JSON result to stdout for the parent process to capture
"""

import os
import sys
import argparse
import pickle
import json
import time
import gc
import traceback
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Train a single GGA model')
    parser.add_argument('--arch', required=True, help='Architecture name')
    parser.add_argument('--approach', required=True, help='Training approach')
    parser.add_argument('--steps', type=int, default=50, help='Training steps')
    parser.add_argument('--data-dir', required=True, help='Directory with pre-computed data')
    parser.add_argument('--checkpoint-base', required=True, help='Base checkpoint directory')
    parser.add_argument('--threads', type=int, default=4, help='Threads per worker')
    parser.add_argument('--lr-start', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr-end', type=float, default=1e-5, help='Final learning rate')
    parser.add_argument('--lr-decay-start', type=float, default=0.0,
                        help='Fraction of training before LR decay starts (0.0-1.0). '
                             '0.0 = decay from start, 0.5 = constant for first half')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping norm')
    args = parser.parse_args()

    # Set thread limits BEFORE importing JAX
    # Compile trim only: the eigen token is measured inert on jaxlib 0.7.0, the
    # pool bound is the launcher-side CPU affinity, and the old
    # intra_op_parallelism_threads token was mis-prefixed (no --xla_) so XLA
    # ignored it.
    os.environ['XLA_FLAGS'] = '--xla_llvm_disable_expensive_passes=true'

    # CPU bind before the JAX import (see xcquinox/alec/workers/_cpu_bind.py):
    # loaded by file path so nothing of the package -- whose __init__ stands
    # up the JAX thread pool -- runs before the pin.
    import importlib.util as _ilu
    _cb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                            'xcquinox', 'alec', 'workers', '_cpu_bind.py')
    _cb_spec = _ilu.spec_from_file_location('_cpu_bind', _cb_path)
    _cb = _ilu.module_from_spec(_cb_spec)
    _cb_spec.loader.exec_module(_cb)
    _cb.apply()  # no-op unless the caller exports the two bind
    # variables: this superseded flow has no slot-assigning launcher,
    # so it runs as unbounded as it always has unless bound by hand.

    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.threads)

    start_time = time.time()
    result = {'status': 'failed', 'arch': args.arch, 'approach': args.approach}

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
        from xcquinox.xc import RXCModel_GGA, RXCModel_GGA_Extended

        # =====================================================================
        # Load pre-computed data
        # =====================================================================

        print(f"Loading data from {args.data_dir}...", file=sys.stderr)

        with open(os.path.join(args.data_dir, 'fixed_density_data.pkl'), 'rb') as f:
            fixed_density_data_np = pickle.load(f)

        # Convert numpy arrays to JAX arrays
        fixed_density_data = {}
        for name, data in fixed_density_data_np.items():
            fixed_density_data[name] = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    fixed_density_data[name][k] = jnp.array(v)
                else:
                    fixed_density_data[name][k] = v

        with open(os.path.join(args.data_dir, 'ccsd_density_data.pkl'), 'rb') as f:
            ccsd_density_data_np = pickle.load(f)

        ccsd_density_data = {}
        for name, data in ccsd_density_data_np.items():
            ccsd_density_data[name] = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    ccsd_density_data[name][k] = jnp.array(v)
                else:
                    ccsd_density_data[name][k] = v

        with open(os.path.join(args.data_dir, 'refs.pkl'), 'rb') as f:
            refs = pickle.load(f)

        with open(os.path.join(args.data_dir, 'architectures.pkl'), 'rb') as f:
            ARCHITECTURES = pickle.load(f)

        with open(os.path.join(args.data_dir, 'pretrained_paths.pkl'), 'rb') as f:
            pretrained_paths = pickle.load(f)

        with open(os.path.join(args.data_dir, 'checkpoint_dirs.pkl'), 'rb') as f:
            CHECKPOINT_DIRS = pickle.load(f)

        arch_config = ARCHITECTURES[args.arch]
        training_mols = {'H': None, 'O': None, 'H2O': None}  # Names only

        print(f"Data loaded. Training {args.arch} with {args.approach}...", file=sys.stderr)

        # =====================================================================
        # Create network and load pre-trained weights
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
                xnet = xce.net.GGA_FxNet_extended_transform(
                    depth=depth, nodes=nodes, seed=seed, lower_rho_cutoff=1e-6,
                    use_self_attention=use_attn, use_cusp=use_cusp,
                    use_dm_features=use_dm, use_laplacian=False, n_dm_features=3
                )
                cnet = xce.net.GGA_FcNet_extended_transform(
                    depth=depth, nodes=nodes, seed=seed+1, lower_rho_cutoff=1e-6,
                    use_self_attention=use_attn, use_cusp=use_cusp,
                    use_dm_features=use_dm, use_laplacian=False, n_dm_features=3
                )
            else:
                xnet = xce.net.GGA_FxNet_sigma_transform(
                    depth=depth, nodes=nodes, seed=seed, lower_rho_cutoff=1e-6,
                    use_self_attention=use_attn
                )
                cnet = xce.net.GGA_FcNet_sigma_transform(
                    depth=depth, nodes=nodes, seed=seed+1, lower_rho_cutoff=1e-6,
                    use_self_attention=use_attn
                )
            return xnet, cnet

        xnet, cnet = create_network_pair(args.arch)
        xnet = eqx.tree_deserialise_leaves(pretrained_paths[args.arch]['xnet'], xnet)
        cnet = eqx.tree_deserialise_leaves(pretrained_paths[args.arch]['cnet'], cnet)

        # Use extended model for architectures with cusp/dm features
        use_cusp = arch_config.get('use_cusp', False)
        use_dm = arch_config.get('use_dm_features', False)
        if use_cusp or use_dm:
            xcmodel = RXCModel_GGA_Extended(
                xnet=xnet, cnet=cnet,
                use_cusp=use_cusp,
                use_dm_features=use_dm,
                n_dm_features=3
            )
        else:
            xcmodel = RXCModel_GGA(xnet=xnet, cnet=cnet)

        # =====================================================================
        # Helper functions
        # =====================================================================

        def compute_exc_nn(model, rho, sigma, weights, cusp_features=None, dm_features=None):
            """Compute E_xc using neural network."""
            input_list = [rho, sigma]
            if cusp_features is not None:
                input_list.append(cusp_features[:, 0])
                input_list.append(cusp_features[:, 1])
            if dm_features is not None:
                for i in range(dm_features.shape[1]):
                    input_list.append(dm_features[:, i])
            inputs = jnp.stack(input_list, axis=1)
            epsilon = model(inputs)
            return jnp.sum(epsilon * weights)

        def compute_total_energy_nn(model, name):
            """Compute total energy on fixed PBE density."""
            data = fixed_density_data[name]
            cusp_features = data.get('cusp_features') if arch_config.get('use_cusp', False) else None
            dm_features = data.get('dm_features') if arch_config.get('use_dm_features', False) else None
            E_xc_nn = compute_exc_nn(model, data['rho'], data['sigma'], data['weights'],
                                     cusp_features=cusp_features, dm_features=dm_features)
            return data['E_non_xc'] + E_xc_nn

        def compute_vxc_nn(model, rho, sigma, ao, weights, cusp_features=None, dm_features=None):
            """Compute V_xc matrix using forward-mode AD.

            Only includes cusp/dm features if they are provided (not None).
            This ensures the input shape matches what the model expects.
            """
            ngrids = rho.shape[0]
            use_cusp = cusp_features is not None
            use_dm = dm_features is not None

            # Build exc_single function based on which features are enabled
            if use_cusp and use_dm:
                # Full 7-feature case
                def exc_single(rho_val, sigma_val, cusp_vals, dm_vals):
                    inp = jnp.array([rho_val, sigma_val,
                                     cusp_vals[0], cusp_vals[1],
                                     dm_vals[0], dm_vals[1], dm_vals[2]])
                    eps = model(inp)
                    return rho_val * eps

                def get_v_rho(rho_val, sigma_val, cusp_vals, dm_vals):
                    primals = (rho_val, sigma_val, cusp_vals, dm_vals)
                    tangents = (1.0, 0.0, jnp.zeros(2), jnp.zeros(3))
                    _, v_rho = jax.jvp(exc_single, primals, tangents)
                    return v_rho

                v_rho = jax.vmap(get_v_rho)(rho, sigma, cusp_features, dm_features)

            elif use_cusp:
                # 4-feature case: rho, sigma, cusp_0, cusp_1
                def exc_single(rho_val, sigma_val, cusp_vals):
                    inp = jnp.array([rho_val, sigma_val, cusp_vals[0], cusp_vals[1]])
                    eps = model(inp)
                    return rho_val * eps

                def get_v_rho(rho_val, sigma_val, cusp_vals):
                    primals = (rho_val, sigma_val, cusp_vals)
                    tangents = (1.0, 0.0, jnp.zeros(2))
                    _, v_rho = jax.jvp(exc_single, primals, tangents)
                    return v_rho

                v_rho = jax.vmap(get_v_rho)(rho, sigma, cusp_features)

            elif use_dm:
                # 5-feature case: rho, sigma, dm_0, dm_1, dm_2
                def exc_single(rho_val, sigma_val, dm_vals):
                    inp = jnp.array([rho_val, sigma_val,
                                     dm_vals[0], dm_vals[1], dm_vals[2]])
                    eps = model(inp)
                    return rho_val * eps

                def get_v_rho(rho_val, sigma_val, dm_vals):
                    primals = (rho_val, sigma_val, dm_vals)
                    tangents = (1.0, 0.0, jnp.zeros(3))
                    _, v_rho = jax.jvp(exc_single, primals, tangents)
                    return v_rho

                v_rho = jax.vmap(get_v_rho)(rho, sigma, dm_features)

            else:
                # Basic 2-feature case: rho, sigma only
                def exc_single(rho_val, sigma_val):
                    inp = jnp.array([rho_val, sigma_val])
                    eps = model(inp)
                    return rho_val * eps

                def get_v_rho(rho_val, sigma_val):
                    primals = (rho_val, sigma_val)
                    tangents = (1.0, 0.0)
                    _, v_rho = jax.jvp(exc_single, primals, tangents)
                    return v_rho

                v_rho = jax.vmap(get_v_rho)(rho, sigma)

            weighted_v = v_rho * weights
            vxc = jnp.einsum('g,gi,gj->ij', weighted_v, ao, ao)
            return vxc

        def oneshot_dm_prediction(model, name):
            """Fast one-shot DM prediction."""
            data = fixed_density_data[name]

            hcore = data['hcore']
            j_pbe = data['j_pbe']
            overlap = data['overlap']
            rho = data['rho']
            sigma = data['sigma']
            weights = data['weights']
            ao = data['ao']
            is_unrestricted = data.get('is_unrestricted', False)

            cusp_features = data.get('cusp_features') if arch_config.get('use_cusp', False) else None
            dm_features = data.get('dm_features') if arch_config.get('use_dm_features', False) else None

            vxc = compute_vxc_nn(model, rho, sigma, ao, weights, cusp_features, dm_features)

            if is_unrestricted:
                fock = hcore + j_pbe + vxc
            else:
                fock = hcore + j_pbe + vxc

            L = jnp.linalg.cholesky(overlap)
            L_inv = jax.scipy.linalg.solve_triangular(L, jnp.eye(L.shape[0]), lower=True)

            if is_unrestricted:
                fock_orth_a = L_inv @ fock[0] @ L_inv.T
                fock_orth_b = L_inv @ fock[1] @ L_inv.T

                _, mo_coeff_orth_a = jnp.linalg.eigh(fock_orth_a)
                _, mo_coeff_orth_b = jnp.linalg.eigh(fock_orth_b)

                mo_coeff_a = L_inv.T @ mo_coeff_orth_a
                mo_coeff_b = L_inv.T @ mo_coeff_orth_b

                nocc_a = int(data['nocc_a'])
                nocc_b = int(data['nocc_b'])

                dm_a = mo_coeff_a[:, :nocc_a] @ mo_coeff_a[:, :nocc_a].T
                dm_b = mo_coeff_b[:, :nocc_b] @ mo_coeff_b[:, :nocc_b].T
                dm_pred = jnp.stack([dm_a, dm_b])
            else:
                fock_orth = L_inv @ fock @ L_inv.T
                _, mo_coeff_orth = jnp.linalg.eigh(fock_orth)
                mo_coeff = L_inv.T @ mo_coeff_orth
                nocc = int(data['nocc'])
                dm_pred = 2.0 * mo_coeff[:, :nocc] @ mo_coeff[:, :nocc].T

            E_pred = compute_total_energy_nn(model, name)
            return dm_pred, E_pred

        def oneshot_grid_density(model, name):
            """One-shot grid density prediction."""
            dm_pred, E_pred = oneshot_dm_prediction(model, name)
            ao = fixed_density_data[name]['ao']

            if dm_pred.ndim == 3:
                rho_pred = jnp.einsum('ij,gi,gj->g', dm_pred[0] + dm_pred[1], ao, ao)
            else:
                rho_pred = jnp.einsum('ij,gi,gj->g', dm_pred, ao, ao)

            return rho_pred, dm_pred, E_pred

        # =====================================================================
        # Loss functions
        # =====================================================================

        @eqx.filter_value_and_grad
        def loss_A_ae_only(model, mols, refs_arr):
            E_H = compute_total_energy_nn(model, 'H')
            E_O = compute_total_energy_nn(model, 'O')
            E_H2O = compute_total_energy_nn(model, 'H2O')
            AE_pred = E_H2O - 2*E_H - E_O
            AE_target = refs['lit']['H2O_AE']
            ae_loss = (AE_pred - AE_target)**2 / (AE_target**2 + 1e-8)
            H_loss = (E_H - refs['lit']['H_TE'])**2 / (refs['lit']['H_TE']**2 + 1e-8)
            O_loss = (E_O - refs['lit']['O_TE'])**2 / (refs['lit']['O_TE']**2 + 1e-8)
            return ae_loss + 0.01 * (H_loss + O_loss)

        @eqx.filter_value_and_grad
        def loss_D1_delta_e(model, mols, refs_arr):
            E_H = compute_total_energy_nn(model, 'H')
            E_O = compute_total_energy_nn(model, 'O')
            E_H2O = compute_total_energy_nn(model, 'H2O')
            AE_nn = E_H2O - 2*E_H - E_O
            delta_nn = AE_nn - refs['pbe']['AE']
            delta_target = refs['lit']['H2O_AE'] - refs['pbe']['AE']
            delta_loss = (delta_nn - delta_target)**2 / (delta_target**2 + 1e-8)
            H_loss = (E_H - refs['lit']['H_TE'])**2 / (refs['lit']['H_TE']**2 + 1e-8)
            O_loss = (E_O - refs['lit']['O_TE'])**2 / (refs['lit']['O_TE']**2 + 1e-8)
            return delta_loss + 0.01 * (H_loss + O_loss)

        # Systems with degenerate orbitals - skip for DM/grid matching
        # Atoms have degenerate eigenvalues that cause gradient instability
        ATOMIC_SYSTEMS = {'H', 'O', 'C', 'N', 'F', 'S', 'Cl', 'He', 'Ne', 'Ar'}

        @eqx.filter_value_and_grad
        def loss_B_ae_dm(model, mols, refs_arr):
            E_H = compute_total_energy_nn(model, 'H')
            E_O = compute_total_energy_nn(model, 'O')
            E_H2O = compute_total_energy_nn(model, 'H2O')
            AE_pred = E_H2O - 2*E_H - E_O
            AE_target = refs['lit']['H2O_AE']
            ae_loss = (AE_pred - AE_target)**2 / (AE_target**2 + 1e-8)

            # DM matching - skip atoms (degenerate eigenvalues cause gradient issues)
            dm_loss = 0.0
            dm_count = 0
            for name in training_mols.keys():
                if name in ATOMIC_SYSTEMS:
                    continue  # Skip atoms
                dm_pred, _ = oneshot_dm_prediction(model, name)
                dm_ccsd = jnp.array(refs['ccsd']['dms'][name])
                dm_diff = dm_pred - dm_ccsd
                nao = dm_ccsd.shape[-1]
                dm_loss += jnp.sum(dm_diff**2) / (nao * nao)  # Normalized
                dm_count += 1

            dm_loss = dm_loss / max(dm_count, 1)  # Average over molecules
            return ae_loss + 0.1 * dm_loss

        @eqx.filter_value_and_grad
        def loss_C_ae_grid(model, mols, refs_arr):
            E_H = compute_total_energy_nn(model, 'H')
            E_O = compute_total_energy_nn(model, 'O')
            E_H2O = compute_total_energy_nn(model, 'H2O')
            AE_pred = E_H2O - 2*E_H - E_O
            AE_target = refs['lit']['H2O_AE']
            ae_loss = (AE_pred - AE_target)**2 / (AE_target**2 + 1e-8)

            # Grid density matching - skip atoms (uses DM prediction internally)
            grid_loss = 0.0
            grid_count = 0
            for name in training_mols.keys():
                if name in ATOMIC_SYSTEMS:
                    continue  # Skip atoms
                rho_pred, _, _ = oneshot_grid_density(model, name)
                rho_ccsd = jnp.array(ccsd_density_data[name]['rho'])
                weights = fixed_density_data[name]['weights']
                rho_diff = rho_pred - rho_ccsd
                grid_loss += jnp.sum(weights * rho_diff**2)
                grid_count += 1

            grid_loss = grid_loss / max(grid_count, 1)  # Average over molecules
            return ae_loss + 0.1 * grid_loss

        @eqx.filter_value_and_grad
        def loss_D2_delta_dm(model, mols, refs_arr):
            E_H = compute_total_energy_nn(model, 'H')
            E_O = compute_total_energy_nn(model, 'O')
            E_H2O = compute_total_energy_nn(model, 'H2O')
            AE_pred = E_H2O - 2*E_H - E_O
            delta_pred = AE_pred - refs['pbe']['AE']
            delta_target = refs['lit']['H2O_AE'] - refs['pbe']['AE']
            delta_loss = (delta_pred - delta_target)**2 / (delta_target**2 + 1e-8)

            # DM matching - skip atoms (degenerate eigenvalues cause gradient issues)
            dm_loss = 0.0
            dm_count = 0
            for name in training_mols.keys():
                if name in ATOMIC_SYSTEMS:
                    continue  # Skip atoms
                dm_pred, _ = oneshot_dm_prediction(model, name)
                dm_ccsd = jnp.array(refs['ccsd']['dms'][name])
                dm_diff = dm_pred - dm_ccsd
                nao = dm_ccsd.shape[-1]
                dm_loss += jnp.sum(dm_diff**2) / (nao * nao)  # Normalized
                dm_count += 1

            dm_loss = dm_loss / max(dm_count, 1)  # Average over molecules
            return delta_loss + 0.1 * dm_loss

        @eqx.filter_value_and_grad
        def loss_D3_delta_grid(model, mols, refs_arr):
            E_H = compute_total_energy_nn(model, 'H')
            E_O = compute_total_energy_nn(model, 'O')
            E_H2O = compute_total_energy_nn(model, 'H2O')
            AE_pred = E_H2O - 2*E_H - E_O
            delta_pred = AE_pred - refs['pbe']['AE']
            delta_target = refs['lit']['H2O_AE'] - refs['pbe']['AE']
            delta_loss = (delta_pred - delta_target)**2 / (delta_target**2 + 1e-8)

            # Grid density matching - skip atoms (uses DM prediction internally)
            grid_loss = 0.0
            grid_count = 0
            for name in training_mols.keys():
                if name in ATOMIC_SYSTEMS:
                    continue  # Skip atoms
                rho_pred, _, _ = oneshot_grid_density(model, name)
                rho_ccsd = jnp.array(ccsd_density_data[name]['rho'])
                weights = fixed_density_data[name]['weights']
                rho_diff = rho_pred - rho_ccsd
                grid_loss += jnp.sum(weights * rho_diff**2)
                grid_count += 1

            grid_loss = grid_loss / max(grid_count, 1)  # Average over molecules
            return delta_loss + 0.1 * grid_loss

        # Select loss function
        loss_functions = {
            'A_ae_only': loss_A_ae_only,
            'B_ae_dm': loss_B_ae_dm,
            'C_ae_grid': loss_C_ae_grid,
            'D1_delta_e': loss_D1_delta_e,
            'D2_delta_dm': loss_D2_delta_dm,
            'D3_delta_grid': loss_D3_delta_grid,
        }

        loss_fn = loss_functions[args.approach]

        # =====================================================================
        # Training
        # =====================================================================

        # Checkpoint directory for progress file
        arch_dir = os.path.join(CHECKPOINT_DIRS[args.approach], args.arch)
        os.makedirs(arch_dir, exist_ok=True)
        progress_file = os.path.join(arch_dir, 'progress.json')

        def write_progress(step, total, loss):
            """Write progress to file for notebook to poll."""
            progress = {
                'arch': args.arch,
                'approach': args.approach,
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

        # Learning rate schedule with optional constant warmup period
        # lr_decay_start: fraction of training before decay begins (0.0 = immediate, 0.5 = halfway)
        warmup_steps = int(args.steps * args.lr_decay_start)
        decay_steps = args.steps - warmup_steps

        if warmup_steps > 0 and decay_steps > 0:
            # Two-phase schedule: constant warmup, then linear decay
            lr_schedule = optax.join_schedules(
                schedules=[
                    optax.constant_schedule(args.lr_start),
                    optax.linear_schedule(
                        init_value=args.lr_start,
                        end_value=args.lr_end,
                        transition_steps=decay_steps
                    )
                ],
                boundaries=[warmup_steps]
            )
            print(f"  LR schedule: {args.lr_start:.2e} (steps 0-{warmup_steps}) -> "
                  f"{args.lr_end:.2e} (steps {warmup_steps}-{args.steps})", file=sys.stderr)
        else:
            # Simple linear decay from start
            lr_schedule = optax.linear_schedule(
                init_value=args.lr_start,
                end_value=args.lr_end,
                transition_steps=args.steps
            )
            print(f"  LR schedule: {args.lr_start:.2e} -> {args.lr_end:.2e}", file=sys.stderr)

        optim = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adam(learning_rate=lr_schedule)
        )
        print(f"  Grad clip: {args.grad_clip}", file=sys.stderr)

        optimizer = xce.train.Optimizer(
            model=xcmodel,
            optim=optim,
            mols={},
            refs=refs,
            loss=loss_fn,
            steps=args.steps,
            print_every=max(1, args.steps // 10),
            verbose=False,
            progress_callback=write_progress
        )

        print(f"Starting training for {args.steps} steps...", file=sys.stderr)
        print(f"  Progress file: {progress_file}", file=sys.stderr)
        write_progress(0, args.steps, 0.0)  # Initial progress
        xcmodel_trained, losses = optimizer()

        # =====================================================================
        # Save results
        # =====================================================================

        arch_dir = os.path.join(CHECKPOINT_DIRS[args.approach], args.arch)
        os.makedirs(arch_dir, exist_ok=True)

        eqx.tree_serialise_leaves(os.path.join(arch_dir, 'xcmodel.eqx'), xcmodel_trained)
        eqx.tree_serialise_leaves(os.path.join(arch_dir, 'xnet.eqx'), xcmodel_trained.xnet)
        eqx.tree_serialise_leaves(os.path.join(arch_dir, 'cnet.eqx'), xcmodel_trained.cnet)
        np.save(os.path.join(arch_dir, 'losses.npy'), np.array(losses))

        metadata = {
            'arch_name': args.arch,
            'approach_name': args.approach,
            'train_steps': args.steps,
            'lr_start': args.lr_start,
            'lr_end': args.lr_end,
            'lr_decay_start': args.lr_decay_start,
            'grad_clip': args.grad_clip,
            'final_loss': float(losses[-1]),
            'min_loss': float(min(losses)),
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - start_time,
        }
        with open(os.path.join(arch_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved to {arch_dir}", file=sys.stderr)

        result = {
            'status': 'success',
            'arch': args.arch,
            'approach': args.approach,
            'final_loss': float(losses[-1]),
            'min_loss': float(min(losses)),
            'duration': time.time() - start_time,
            'checkpoint_dir': arch_dir,
        }

    except Exception as e:
        result = {
            'status': 'failed',
            'arch': args.arch,
            'approach': args.approach,
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
