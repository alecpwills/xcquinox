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

# RSS-growth probe v3 for the dfs6311 production OOM (run_20260720T181059Z).
# Same cell as v1/v2: spec_0003, deep_3x16, subset_size 4, 10 updates/epoch.
#
# Evidence chain:
# - v1 (job 2093006): recompile ratchet REFUTED -- 939 compile records, all
#   in the epoch-1 phase, zero after; yet RSS climbed 27 -> 184 GB over 100
#   updates (~1.6 GB/update) on a warm cache; SIGKILL at epoch ~17 at 221 GB
#   of 251.6 GB. The growth is retained buffers, not kernels.
# - v2 (job 2093139, + gc-per-update + live-array census): the gc cadence
#   ARRESTS the trajectory -- RSS stationary at ~200 GB over the final ~30
#   epochs, no new process HWM (219.4 GB) after epoch 34, lifetime 5.4x v1,
#   overhead ~zero (2.56 vs 2.70 min/epoch). Still SIGKILLed at epoch ~98 on
#   a FLAT curve: a ~200 GB floor plus a >= 19 GB per-update transient plus
#   non-worker node usage leaves no margin on a 251.6 GB node, and the fatal
#   spike falls between the step-boundary HWM reads and sacct's 30 s samples.
#   The census stayed pinned at 23.49 GB across all 900 updates while RSS sat
#   near 200 GB: ~88% of resident memory is C-level (glibc arena caching
#   under the ~96-thread XLA pool + BLAS threads, and/or XLA C++), neither
#   live JAX buffers nor collectable Python garbage. gc is necessary, not
#   sufficient; the floor itself must be cut.
# - v2 side finding (parked, not this probe's target): the unpadded
#   validation path recompiles ~10.3k tiny eager ops at EVERY validation
#   (10328/10322/10322 at the epoch 25/50/75 boundaries), consistent with a
#   bounded dispatch cache cycling; memory-neutral (RSS brackets
#   -4.1/-5.7/+0.2 GB) -- a throughput tax for the val-padding fix.
#
# v3 = v2 + MALLOC_ARENA_MAX=2, the floor-cut treatment. glibc allows up to
# 8 arenas per core (768 on a 96-core node) and parks freed heap on
# per-arena free lists instead of returning it to the OS; the per-update
# transient churn (>= 19 GB high-water over steady RSS) makes arena caching
# the prime candidate for the ~176 GB C-level floor. Two arenas measure the
# cut directly and discriminate arenas vs XLA-internal retention:
#   - u300+ floor well below ~200 GB (production sufficiency wants <~120 GB)
#     -> arenas confirmed; Phase 34 ships gc cadence + arena cap as
#     production defaults.
#   - floor unchanged -> XLA-C++/other C-level retention; escalate to
#     malloc_trim / jemalloc discriminator probes.
# min/epoch guards contention (2 arenas under ~100 runtime threads):
# ~2.6 min/epoch must hold or the cap costs too much. Both v2 levers and
# compile logging stay ON so the census and histogram re-verify under the cap.
#
# The worker is invoked DIRECTLY (not via cluster._train_task): the harness
# pipes the worker's merged stdout/stderr into a bounded tail ring, which
# would drop most compile-log lines. Direct invocation streams everything
# here. No eval follows; the probe's product is this log + the aux_log curve.
#
# The spec is re-staged with a FRESH checkpoint_dir before the run
# (rss_probe3_spec_0003): the production spec_0003 dir and both prior probe
# dirs hold WS5 resume state, and a resumed run would skip the epoch-0
# compile phase and most of the growth curve.

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

# --- glibc allocator: the v3 treatment lever -------------------------------
# Cap malloc arenas at 2 (node default: 8 x 96 cores = 768). Probe-only for
# now; production inherits it in Phase 34 if the floor cut confirms.
export MALLOC_ARENA_MAX=2
echo "[rss-probe] MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX (v3 treatment; v1/v2 ran uncapped)"

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
    spec, checkpoint_dir=f"{run}/checkpoints/rss_probe3_spec_0003")
with open(f"{run}/specs/rss_probe3_spec_0003.spec", "wb") as f:
    pickle.dump(probe, f)
print(f"[rss-probe] staged fresh-checkpoint spec -> {probe.checkpoint_dir}")
PY

echo "[rss-probe] starting spec_0003 rerun at $(date -Is)"
python -m xcquinox.alec._train_one_spec "$RUN"/specs/rss_probe3_spec_0003.spec --device auto
rc=$?
echo "[rss-probe] worker exit code: $rc at $(date -Is)"
exit $rc
