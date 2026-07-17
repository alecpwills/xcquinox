#!/usr/bin/env bash
#SBATCH --job-name=xcq_smoke_vma_pad
#SBATCH --partition=long-96core
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=24
#SBATCH --output=/gpfs/scratch/awills/dfs6311_smoke_vma_pad_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=END,FAIL
# =============================================================================
# The "after" harness for the shape-padding fix (pair with dfs6311_smoke_vma.sbatch,
# the unpadded "before"). Same heaviest cell (spec_0021 = deep_attn_3x16, ss26) and
# same production compile environment, but with --pad-group ON: the standalone
# padding pass canonicalizes every molecule to one common shape (with traced
# occupation), so the de-fused per-molecule kernels collapse to one compile per
# spin-type -- two total (RKS + UKS) instead of ~26.
#
# EXPECT: nVMA stays FAR below the 65530 mmap ceiling, RSS well under the node,
#         and the run ends with {"kind": "done", ...}.
# Requires the padding code (padding.py + solver_manual.py seam + the flag +
# --pad-group in _train_one_spec.py) rsync'd to the cluster tree first.
# =============================================================================
set -uo pipefail

# --- conda activation ------------------------------------------------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate /gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j070
export PYTHONNOUSERSITE=1

# --- production compile environment (mirrors train_eval_inline_cpu.sbatch) ---
CORES="${SLURM_CPUS_ON_NODE:-$(nproc --all)}"
BT=$(( CORES / 12 )); [ "$BT" -ge 1 ] || BT=1
export OMP_NUM_THREADS="$BT" MKL_NUM_THREADS="$BT" OPENBLAS_NUM_THREADS="$BT"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=1 --xla_llvm_disable_expensive_passes=true --xla_backend_optimization_level=1"
ulimit -u unlimited 2>/dev/null || true
echo "[vma-pad] host=$(hostname) CORES=$CORES BLAS_THREADS=$BT max_map_count=$(cat /proc/sys/vm/max_map_count)"

# --- compile + one epoch on the heaviest cell, PADDING ON, monitored -------
python -m xcquinox.alec._train_one_spec \
  /gpfs/scratch/awills/xcquinox_runs/dfs_step7/dfs6311_grid3_v3/runs/run_20260715T030651Z/specs/spec_0021.spec \
  --device cpu --smoke --pad-group --no-progress &
PID=$!

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
echo "[vma-pad] _train_one_spec exit code: ${rc}"
exit ${rc}
