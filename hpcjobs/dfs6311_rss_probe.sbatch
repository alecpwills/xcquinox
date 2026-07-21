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

# RSS-growth probe v2 for the dfs6311 production OOM (run_20260720T181059Z).
#
# Probe v1 (job 2093006) verdict: recompile ratchet REFUTED (939 compile
# records, all in the epoch-1 phase, zero after) and the first validation was
# never reached (SIGKILL at epoch ~17, MaxRSS 221.0 GB of 251.6 GB) -- yet RSS
# climbed 27 -> 184 GB over 100 updates (~1.6 GB/update) on a fully-warm
# compile cache. The growth is retained buffers, not kernels.
#
# v2 reruns the same cell (spec_0003, deep_3x16, subset_size 4) with two
# added levers to discriminate the retention side and test the fix in one run:
#   1. XCQUINOX_MEMDIAG=1 -- every step aux entry gains a live JAX
#      device-array census (live_n, live_gb): Python-side retention makes the
#      census grow with RSS; C-level allocator retention leaves it flat.
#   2. XCQUINOX_GC_EVERY=1 -- one gc.collect per optimizer update. Dead
#      cyclic update graphs (the eager per-update value_and_grad closures)
#      hold GB-scale buffers until a gen-2 pass runs, which rarely triggers
#      on its own in this array-dominated loop. A flat RSS curve here
#      confirms collection lag as the mechanism AND validates the fix.
#   Plus the v1 instruments: per-update rss_gb/hwm_gb (resume-flush-persisted
#   every 10 epochs, so the curve survives an OOM SIGKILL) and
#   JAX_LOG_COMPILES=1 (any late compile record would revive the ratchet).
#
# The worker is invoked DIRECTLY (not via cluster._train_task): the harness
# pipes the worker's merged stdout/stderr into a bounded tail ring, which
# would drop most compile-log lines. Direct invocation streams everything
# here. No eval follows; the probe's product is this log + the aux_log curve.
#
# The spec is re-staged with a FRESH checkpoint_dir before the run
# (rss_probe2_spec_0003): both the production spec_0003 dir and the v1 probe
# dir hold WS5 resume state (v1 flushed through epoch 9), and a resumed run
# would skip the epoch-0 compile phase and most of the growth curve.

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
export XCQUINOX_MEMDIAG=1
export XCQUINOX_GC_EVERY=1

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
    spec, checkpoint_dir=f"{run}/checkpoints/rss_probe2_spec_0003")
with open(f"{run}/specs/rss_probe2_spec_0003.spec", "wb") as f:
    pickle.dump(probe, f)
print(f"[rss-probe] staged fresh-checkpoint spec -> {probe.checkpoint_dir}")
PY

echo "[rss-probe] starting spec_0003 rerun at $(date -Is)"
python -m xcquinox.alec._train_one_spec "$RUN"/specs/rss_probe2_spec_0003.spec --device auto
rc=$?
echo "[rss-probe] worker exit code: $rc at $(date -Is)"
exit $rc
