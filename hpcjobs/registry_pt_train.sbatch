#!/usr/bin/env bash
#SBATCH --job-name=xcq_regpt_train
#SBATCH --partition=extended-96core
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=96
#SBATCH --signal=B:TERM@300
#SBATCH --output=/gpfs/scratch/awills/regpt_train_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Training stage of one registry proton-transfer prong notebook
# (notebooks/registry_pt_prong_<P>.ipynb, P = A: per-group updates, twelve
# steps per epoch; P = B: gradient accumulation, one step per epoch). Executes
# the notebook with REGISTRY_PT_STAGE=train through
# tools/run_registry_pt_stage.py: TRAIN_WORKERS workers (default 2, each pinned
# to half the node) claim architectures from the prong's ledger, train each in
# a child, validate it at once in another, record the result, and claim the
# next. A failed child is recorded (stage, exit code, log tail) and the worker
# moves on; the result tables are assembled at the end.
#
# Several jobs of one prong may run together (they share the ledger). On the
# wall-clock TERM every child flushes its per-epoch resume state; the
# interrupted architectures return to pending and the next job resumes them.
#
# Submit (from a milan login node), after the prep job:
#   sbatch --dependency=afterok:<prep_job> hpcjobs/registry_pt_train.sbatch A
#   sbatch --dependency=afterok:<prep_job> hpcjobs/registry_pt_train.sbatch B
# A second argument restricts the job to one training class: "standard" (the
# MLPs, TRAIN_WORKERS at a time) or "kan" (the 2x8 KANs, one at a time, alone
# on the node), e.g. a KAN-only job beside a standard one:
#   sbatch --dependency=afterok:<prep_job> hpcjobs/registry_pt_train.sbatch A kan
# REGISTRY_PT_RETRY_FAILED=1 returns failed architectures to the queue.
#
# Shell discipline: no `set -e` (conda activation can return non-zero on
# success); activation verified by its effect; explicit exit code.
set -uo pipefail

PRONG="${1:-${PRONG:-}}"
case "$PRONG" in
  A|B) : ;;
  *) echo "[regpt-train] FATAL: give the prong, A or B (got: ${PRONG:-none})"; exit 2 ;;
esac
# optional second argument: one training class only ("standard": the MLPs,
# TRAIN_WORKERS at a time; "kan": the KANs, one at a time, alone on the node);
# without it the job runs the standard class, then the KAN class
ARCH_CLASS="${2:-${REGISTRY_PT_ARCH_CLASS:-}}"
case "$ARCH_CLASS" in
  ""|standard|kan) : ;;
  *) echo "[regpt-train] FATAL: class must be standard or kan (got: $ARCH_CLASS)"; exit 2 ;;
esac
if [ -n "$ARCH_CLASS" ]; then export REGISTRY_PT_ARCH_CLASS="$ARCH_CLASS"; else unset REGISTRY_PT_ARCH_CLASS; fi

ENV_PREFIX=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j0102
# --- conda activation (verified by effect, not by return code) --------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[regpt-train] host=$(hostname) python=$PYBIN prong=$PRONG"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[regpt-train] FATAL: parity env python not active (got: ${PYBIN:-none})"
     exit 1 ;;
esac

# --- thread caps (SLURM vars not assumed present) ---------------------------
# PySCF's OpenMP pool and numpy's OpenBLAS pool run at min(allocation, 8)
# threads each (parallel.PYSCF_POOL_THREADS_MAX, workflow-matrix job
# 2134488); each training child's XLA pool is confined to its worker's half of
# the node by the CPU affinity the worker sets.
THREADS="${SLURM_CPUS_PER_TASK:-8}"
case "$THREADS" in ''|*[!0-9]*) THREADS=8 ;; esac
THREADS=$(( 10#$THREADS ))
[ "$THREADS" -ge 1 ] || THREADS=1
[ "$THREADS" -le 8 ] || THREADS=8
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
ulimit -u unlimited 2>/dev/null || true
echo "[regpt-train] allocation=${SLURM_CPUS_PER_TASK:-unset} -> OMP/MKL/OPENBLAS=$THREADS"

REPO=/gpfs/projects/FernandezGroup/Alec/xcquinox
WORK_ROOT="${REGISTRY_PT_WORK_ROOT:-/gpfs/scratch/awills/xcquinox_runs/registry_pt_prongs}"
cd "$REPO" || { echo "[regpt-train] FATAL: no checkout at $REPO"; exit 1; }
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 MALLOC_ARENA_MAX=2
export XCQUINOX_SEED_ALLOW_GENERATE=1
export TRAIN_WORKERS="${TRAIN_WORKERS:-2}"
[ -d "$WORK_ROOT/shared" ] || { echo "[regpt-train] FATAL: no prepared work root at $WORK_ROOT"; exit 1; }
echo "[regpt-train] work root: $WORK_ROOT; commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"

python -c "import nbclient, nbformat, xcquinox.pipeline" \
  || { echo "[regpt-train] FATAL: import probe failed"; exit 1; }

setsid python -u tools/run_registry_pt_stage.py --prong "$PRONG" --stage train \
  --work-root "$WORK_ROOT" &
pid=$!
trap 'echo "[regpt-train] TERM: forwarding to the driver group"; kill -TERM -- -"$pid" 2>/dev/null' TERM
wait "$pid"
rc=$?
while [ "$rc" -ge 128 ] && kill -0 "$pid" 2>/dev/null; do
  wait "$pid"
  rc=$?
done
echo "[regpt-train] driver exit $rc"
exit "$rc"
