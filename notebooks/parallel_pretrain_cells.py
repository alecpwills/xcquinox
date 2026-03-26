# =============================================================================
# NOTEBOOK CELLS FOR PARALLEL PRE-TRAINING
# =============================================================================
# Add these cells after the pre-training data collection cell and before the
# sequential pre-training loop.
# =============================================================================

# -----------------------------------------------------------------------------
# CELL 1: Save pre-training data for parallel workers
# -----------------------------------------------------------------------------
"""
# Save pre-training data for parallel workers
import pickle

PRETRAIN_DATA_DIR = os.path.join(CHECKPOINT_BASE, 'pretrain_data')
os.makedirs(PRETRAIN_DATA_DIR, exist_ok=True)

print('Saving pre-training data for parallel workers...')

# Save all pre-training arrays
pretrain_data = {
    'rho_all': np.array(rho_all),
    'sigma_all': np.array(sigma_all),
    'Fx_all': np.array(Fx_all),
    'Fc_all': np.array(Fc_all),
    'cusp_all': np.array(cusp_all),
    'dm_all': np.array(dm_all),
}
with open(os.path.join(PRETRAIN_DATA_DIR, 'pretrain_data.pkl'), 'wb') as f:
    pickle.dump(pretrain_data, f)
print(f'  Saved pretrain_data: {len(rho_all)} points')

# Save architecture configs
with open(os.path.join(PRETRAIN_DATA_DIR, 'architectures.pkl'), 'wb') as f:
    pickle.dump(ARCHITECTURES, f)
print(f'  Saved ARCHITECTURES: {len(ARCHITECTURES)} architectures')

# Save checkpoint directories
with open(os.path.join(PRETRAIN_DATA_DIR, 'checkpoint_dirs.pkl'), 'wb') as f:
    pickle.dump(CHECKPOINT_DIRS, f)
print('  Saved checkpoint_dirs')

print(f'\\nAll pre-training data saved to: {PRETRAIN_DATA_DIR}')
print('Ready for parallel pre-training.')
"""

# -----------------------------------------------------------------------------
# CELL 2: Parallel pre-training infrastructure
# -----------------------------------------------------------------------------
"""
# =============================================================================
# Parallel Pre-training Infrastructure
# =============================================================================
import subprocess
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Configuration
MAX_PRETRAIN_WORKERS = 4  # Adjust based on your system's memory
PRETRAIN_WORKER_SCRIPT = '/home/awills/Documents/Research/xcquinox/scripts/parallel_pretrain_worker.py'

print(f"Pre-train worker script: {PRETRAIN_WORKER_SCRIPT}")
print(f"Exists: {os.path.exists(PRETRAIN_WORKER_SCRIPT)}")


def run_single_pretrain(arch, steps, data_dir, checkpoint_base, threads=4,
                        lr_start=1e-2, lr_end=1e-4, grad_clip=1.0):
    \"\"\"Run a single pre-training job via subprocess.\"\"\"
    cmd = [
        sys.executable, PRETRAIN_WORKER_SCRIPT,
        '--arch', arch,
        '--steps', str(steps),
        '--data-dir', data_dir,
        '--checkpoint-base', checkpoint_base,
        '--threads', str(threads),
        '--lr-start', str(lr_start),
        '--lr-end', str(lr_end),
        '--grad-clip', str(grad_clip),
    ]

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        # Parse JSON result from stdout
        for line in result.stdout.strip().split('\\n'):
            if line.startswith('{'):
                return json.loads(line)

        return {
            'status': 'failed',
            'arch': arch,
            'error': f"No valid output. stderr: {result.stderr[-500:] if result.stderr else 'none'}",
            'duration': time.time() - start_time,
        }

    except subprocess.TimeoutExpired:
        return {
            'status': 'failed',
            'arch': arch,
            'error': 'Timeout (>1 hour)',
            'duration': time.time() - start_time,
        }
    except Exception as e:
        return {
            'status': 'failed',
            'arch': arch,
            'error': str(e),
            'duration': time.time() - start_time,
        }


def run_parallel_pretrain(architectures, steps, data_dir, checkpoint_base,
                          max_workers=4, threads_per_worker=4):
    \"\"\"Run pre-training jobs in parallel using ThreadPoolExecutor + subprocess.\"\"\"
    results = []
    failed = []
    total = len(architectures)

    print(f"\\n{'='*60}")
    print(f"PARALLEL PRE-TRAINING: {total} architectures with {max_workers} workers")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_arch = {
            executor.submit(
                run_single_pretrain,
                arch, steps, data_dir, checkpoint_base, threads_per_worker
            ): arch
            for arch in architectures
        }

        completed = 0
        for future in as_completed(future_to_arch):
            arch = future_to_arch[future]
            completed += 1

            try:
                result = future.result()
                if result['status'] == 'success':
                    results.append(result)
                    loss_x = result.get('final_loss_x', 'N/A')
                    loss_c = result.get('final_loss_c', 'N/A')
                    duration = result.get('duration', 0)
                    print(f"  [{completed:2d}/{total}] ✓ {arch:<20} | X={loss_x:.6f} C={loss_c:.6f} | time={duration:.1f}s")
                else:
                    failed.append(result)
                    error = result.get('error', 'Unknown error')[:50]
                    print(f"  [{completed:2d}/{total}] ✗ {arch:<20} | ERROR: {error}")
            except Exception as e:
                failed.append({'arch': arch, 'error': str(e)})
                print(f"  [{completed:2d}/{total}] ✗ {arch:<20} | EXCEPTION: {str(e)[:50]}")

    total_time = time.time() - start_time
    print(f"\\nPre-training completed in {total_time/60:.1f} minutes")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {len(failed)}")

    return results, failed


print(f"\\nParallel pre-training infrastructure ready.")
print(f"  MAX_PRETRAIN_WORKERS: {MAX_PRETRAIN_WORKERS}")
"""

# -----------------------------------------------------------------------------
# CELL 3: Run parallel pre-training
# -----------------------------------------------------------------------------
"""
# =============================================================================
# Run Parallel Pre-training
# =============================================================================
PRETRAIN_STEPS = 500
PRETRAIN_DATA_DIR = os.path.join(CHECKPOINT_BASE, 'pretrain_data')

# Verify data exists
if not os.path.exists(os.path.join(PRETRAIN_DATA_DIR, 'pretrain_data.pkl')):
    print("ERROR: Pre-training data not found!")
    print("Please run the 'Save pre-training data' cell first.")
else:
    pretrain_start = time.time()

    print(f"\\n{'#'*60}")
    print(f"# PARALLEL PRE-TRAINING")
    print(f"# Steps per architecture: {PRETRAIN_STEPS}")
    print(f"# Total architectures: {len(ARCHITECTURES)}")
    print(f"# Max workers: {MAX_PRETRAIN_WORKERS}")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    pretrain_results, pretrain_failed = run_parallel_pretrain(
        architectures=list(ARCHITECTURES.keys()),
        steps=PRETRAIN_STEPS,
        data_dir=PRETRAIN_DATA_DIR,
        checkpoint_base=CHECKPOINT_BASE,
        max_workers=MAX_PRETRAIN_WORKERS,
        threads_per_worker=4
    )

    # Save summary
    pretrain_summary = {
        'total_time_seconds': time.time() - pretrain_start,
        'successful': len(pretrain_results),
        'failed': len(pretrain_failed),
        'results': pretrain_results,
        'failed_details': pretrain_failed,
        'timestamp': datetime.now().isoformat(),
    }

    summary_path = os.path.join(CHECKPOINT_DIRS['pretrain'], 'pretrain_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(pretrain_summary, f, indent=2)

    print(f"\\nSaved pre-training summary to: {summary_path}")
"""

# -----------------------------------------------------------------------------
# CELL 4: Load pre-trained networks into memory
# -----------------------------------------------------------------------------
"""
# =============================================================================
# Load Pre-trained Networks
# =============================================================================
# After parallel pre-training, load the networks back into the `networks` dict

print("Loading pre-trained networks...")

for arch in ARCHITECTURES.keys():
    arch_dir = os.path.join(CHECKPOINT_DIRS['pretrain'], arch)
    xnet_path = os.path.join(arch_dir, 'xnet.eqx')
    cnet_path = os.path.join(arch_dir, 'cnet.eqx')

    if os.path.exists(xnet_path) and os.path.exists(cnet_path):
        # Create fresh network pair to get the structure
        xnet, cnet = create_network_pair(arch, SEED)

        # Load trained weights
        xnet = eqx.tree_deserialise_leaves(xnet_path, xnet)
        cnet = eqx.tree_deserialise_leaves(cnet_path, cnet)

        networks[arch]['xnet'] = xnet
        networks[arch]['cnet'] = cnet

        # Load losses if available
        losses_x_path = os.path.join(arch_dir, 'losses_x.npy')
        losses_c_path = os.path.join(arch_dir, 'losses_c.npy')
        if os.path.exists(losses_x_path) and os.path.exists(losses_c_path):
            losses_x = np.load(losses_x_path)
            losses_c = np.load(losses_c_path)
            pretrain_losses[arch] = {'x': losses_x, 'c': losses_c}
            print(f"  ✓ {arch}: X={losses_x[-1]:.6f}, C={losses_c[-1]:.6f}")
        else:
            print(f"  ✓ {arch}: loaded (no loss data)")
    else:
        print(f"  ✗ {arch}: not found at {arch_dir}")

print(f"\\nLoaded {len([a for a in ARCHITECTURES if networks[a]['xnet'] is not None])} pre-trained network pairs")
"""
