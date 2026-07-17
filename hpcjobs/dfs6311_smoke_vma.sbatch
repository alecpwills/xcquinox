#!/usr/bin/env bash
#SBATCH --job-name=xcq_smoke_vma
#SBATCH --partition=long-96core
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=24
#SBATCH --output=/gpfs/scratch/awills/dfs6311_smoke_vma_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=END,FAIL
# =============================================================================
# Instrumented re-run of the dfs6311 compile-smoke probe (spec_0021 =
# deep_attn_3x16, subset_size=26) under the SAME production compile environment
# as a real train node. A background monitor samples the worker's RSS / VSZ /
# mapping-count (VMA) / thread-count every 20 s into this job's .out.
#
# Purpose: characterize the compile-OOM. The de-fuse retains one compiled kernel
# per distinct molecule shape in a never-evicted process-global cache; at
# deep_attn x ~26 distinct shapes x 6-311++G(3df,2pd)/grid3 the accumulated JIT
# mappings are expected to march nVMA toward the default vm.max_map_count=65530
# ceiling (with RSS still well under the node's ~252 GB) and fail with
# "LLVM compilation error: Cannot allocate memory".
#
# Read: the last trace rows before exit.
#   nVMA -> ~65530 at death  => mapping-count ceiling (the diagnosis).
#   RSS climbing toward ~252 => genuinely RAM-bound instead.
# This file is also the before/after harness: after the shape-padding fix,
# a padded re-run must keep nVMA far below 65530 and print {"kind": "done"}.
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
echo "[vma] host=$(hostname) CORES=$CORES BLAS_THREADS=$BT max_map_count=$(cat /proc/sys/vm/max_map_count) swap=$(awk '/SwapTotal/{print $2$3}' /proc/meminfo)"

# --- compile + one epoch on the heaviest cell, monitored -------------------
python -m xcquinox.alec._train_one_spec \
  /gpfs/scratch/awills/xcquinox_runs/dfs_step7/dfs6311_grid3_v3/runs/run_20260715T030651Z/specs/spec_0021.spec \
  --device cpu --smoke --no-progress &
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
echo "[vma] _train_one_spec exit code: ${rc}"
exit ${rc}
