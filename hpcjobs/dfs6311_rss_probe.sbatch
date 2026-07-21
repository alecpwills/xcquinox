#!/usr/bin/env bash
#SBATCH --job-name=xcq_rss_probe
#SBATCH --partition=long-96core
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=24
#SBATCH --output=/gpfs/scratch/awills/xcquinox_runs/dfs_step7/dfs6311_grid3_v3/runs/run_20260720T181059Z/logs/rss_probe_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# RSS-growth probe for the dfs6311 production OOM (run_20260720T181059Z).
#
# Reruns spec_0003 (deep_3x16, subset_size 4 -- the fastest-dying cell: OOM at
# epoch 30, MaxRSS 217.4 GB of the node's 251.6 GB) with two instruments:
#   1. per-update RSS/HWM in aux_log (rss_gb/hwm_gb per optimizer update,
#      rss_gb_pre_val/rss_gb_post_val bracketing each validation) -- persisted
#      by the periodic resume flush every 10 epochs, so the curve survives an
#      OOM SIGKILL; the per-epoch step JSON on stdout also carries rss_gb.
#   2. JAX_LOG_COMPILES=1 -- every XLA compile is a WARNING line in this log.
#      Compile records after epoch 1 discriminate a per-step recompile ratchet
#      from a plateaued cache; records at the epoch-25 validation boundary
#      expose the unpadded-validation compile burst.
#
# The worker is invoked DIRECTLY (not via cluster._train_task): the harness
# pipes the worker's merged stdout/stderr into a bounded tail ring, which
# would drop most compile-log lines. Direct invocation streams everything
# here. No eval follows; the probe's product is this log + the aux_log curve.
#
# The spec is re-staged with a FRESH checkpoint_dir before the run: the
# production spec_0003 dir may hold WS5 resume state flushed at epochs
# 10/20/30 before the OOM, and a resumed probe would skip the epoch-0
# compile phase and most of the growth curve. The fresh dir also preserves
# the production failure record untouched.

set -uo pipefail

# --- conda activation ------------------------------------------------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate /gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j070
export PYTHONNOUSERSITE=1

# --- thread caps (identical to production train_eval) ----------------------
CORES="${SLURM_CPUS_ON_NODE:-$(nproc --all)}"
BLAS_THREADS=$(( CORES / 12 )); [ "$BLAS_THREADS" -ge 1 ] || BLAS_THREADS=1
export OMP_NUM_THREADS="$BLAS_THREADS"
export MKL_NUM_THREADS="$BLAS_THREADS"
export OPENBLAS_NUM_THREADS="$BLAS_THREADS"
echo "[rss-probe] CORES=$CORES -> OMP/MKL/OPENBLAS=$BLAS_THREADS"

# --- JAX/XLA (identical to production, plus compile logging) ---------------
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=1 --xla_llvm_disable_expensive_passes=true --xla_backend_optimization_level=1"
export JAX_LOG_COMPILES=1

RUN=/gpfs/scratch/awills/xcquinox_runs/dfs_step7/dfs6311_grid3_v3/runs/run_20260720T181059Z
ulimit -u unlimited 2>/dev/null || true

python - "$RUN" <<'PY'
import dataclasses
import pickle  # noqa: S403 -- spec produced by this codebase on this filesystem
import sys

run = sys.argv[1]
with open(f"{run}/specs/spec_0003.spec", "rb") as f:
    spec = pickle.load(f)  # noqa: S301 -- trusted, same-pipeline artifact
probe = dataclasses.replace(
    spec, checkpoint_dir=f"{run}/checkpoints/rss_probe_spec_0003")
with open(f"{run}/specs/rss_probe_spec_0003.spec", "wb") as f:
    pickle.dump(probe, f)
print(f"[rss-probe] staged fresh-checkpoint spec -> {probe.checkpoint_dir}")
PY

echo "[rss-probe] starting spec_0003 rerun at $(date -Is)"
python -m xcquinox.alec._train_one_spec "$RUN"/specs/rss_probe_spec_0003.spec --device auto
rc=$?
echo "[rss-probe] worker exit code: $rc at $(date -Is)"
exit $rc
