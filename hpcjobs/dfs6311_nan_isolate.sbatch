#!/usr/bin/env bash
#SBATCH --job-name=xcq_nan_isolate
#SBATCH --partition=long-96core
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=24
#SBATCH --output=/gpfs/scratch/awills/dfs6311_nan_isolate_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=END,FAIL
# =============================================================================
# Exact-state replay + decomposition of the bh76:OH+N2_to_H+N2O training NaN
# (job 2091616: step 4, finite loss, all 36 gradient leaves NaN). Replays
# spec_0021 epoch 0 in the padded smoke configuration (pretrained weights,
# seeded shuffle, real optimizer updates; --pad supplies the padding the spec
# file predates, --horizon-epochs 1 mirrors the smoke's schedule), then at the
# first non-finite step decomposes the gradient per molecule (HO/N2/H/N2O) x
# per stage (energy VJP vs rho/vxc channel).
#
# ISOLATE_ARGS overrides the full argument list (default: the spec replay).
# Variants, one per submission (module constants bake at trace time):
#   structural (fresh-init, no pretrained weights needed):
#     sbatch --export=ALL,ISOLATE_ARGS="--local --padded" dfs6311_nan_isolate.sbatch
#   ablations on the replay (a cured epoch exits 0; a persisting NaN decomposes;
#   ISOLATE_ARGS replaces the default list, so --pad --horizon-epochs 1 must be
#   restated or the replay silently runs unpadded):
#     sbatch --export=ALL,ISOLATE_ARGS="--spec <specpath> --pad --horizon-epochs 1 --ablate ec_tail" ...
#     (--ablate one of: sym_break degeneracy_reg both_eigh ec_tail zeta_eps)
# Exit 3 = non-finite found and decomposed (the informative outcome); 0 = ran
# all-finite to completion.
# =============================================================================
set -uo pipefail

# --- conda activation ------------------------------------------------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate /gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j070
export PYTHONNOUSERSITE=1

# --- production compile environment (mirrors the smoke probes) --------------
CORES="${SLURM_CPUS_ON_NODE:-$(nproc --all)}"
BT=$(( CORES / 12 )); [ "$BT" -ge 1 ] || BT=1
export OMP_NUM_THREADS="$BT" MKL_NUM_THREADS="$BT" OPENBLAS_NUM_THREADS="$BT"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=1 --xla_llvm_disable_expensive_passes=true --xla_backend_optimization_level=1"
ulimit -u unlimited 2>/dev/null || true
echo "[isolate] host=$(hostname) CORES=$CORES BLAS_THREADS=$BT max_map_count=$(cat /proc/sys/vm/max_map_count)"

SPEC=/gpfs/scratch/awills/xcquinox_runs/dfs_step7/dfs6311_grid3_v3/runs/run_20260715T030651Z/specs/spec_0021.spec
# ISOLATE_ARGS REPLACES the whole argument list (so "--local ..." variants do
# not collide with the default --spec); unset -> the padded smoke replay.
ARGS="${ISOLATE_ARGS:---spec $SPEC --pad --horizon-epochs 1}"
python -u /gpfs/projects/FernandezGroup/Alec/xcquinox/hpcjobs/dfs6311_nan_isolate.py $ARGS &
PID=$!

# nVMA/RSS sampler (same instrument as the smoke probes): telemetry survives
# even if the run dies by SIGSEGV instead of the fail-loud guard.
printf '%8s %8s %8s %9s %6s\n' t_s RSS_GB VSZ_GB nVMA thr
while kill -0 "$PID" 2>/dev/null; do
  if [ -r "/proc/$PID/status" ]; then
    r=$(awk '/^VmRSS/{printf "%.1f",$2/1048576}'  "/proc/$PID/status" 2>/dev/null)
    v=$(awk '/^VmSize/{printf "%.1f",$2/1048576}' "/proc/$PID/status" 2>/dev/null)
    t=$(awk '/^Threads/{print $2}'                "/proc/$PID/status" 2>/dev/null)
    m=$(wc -l < "/proc/$PID/maps" 2>/dev/null)
    printf '%8s %8s %8s %9s %6s\n' "$SECONDS" "$r" "$v" "$m" "$t"
  fi
  sleep 20
done

wait "$PID"; rc=$?
echo "[isolate] exit code: ${rc}"
exit ${rc}
