#!/usr/bin/env bash
#SBATCH --job-name=xcq_regpt_prep
#SBATCH --partition=extended-96core
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=96
#SBATCH --signal=B:TERM@300
#SBATCH --output=/gpfs/scratch/awills/regpt_prep_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Preparation stage of the registry proton-transfer prong notebooks
# (notebooks/registry_pt_prong_{A,B}.ipynb), shared by both prongs: the
# campaign and run directory, the pretraining data, the 26 clones and their
# certificates, the relaxed CCSD(T) references of the 16 corpus species and
# the 12 proton-transfer path points at 6-311++G(3df,2pd) / grid level 3, and
# the training specs of both prongs. Executes notebook A with
# REGISTRY_PT_STAGE=prep through tools/run_registry_pt_stage.py.
#
# Every artifact is cached under the work root, so a resubmit continues where
# a wall-clock kill or a failure left off; an architecture whose clone or
# certificate cannot be made is recorded in shared/prep_status.json and enters
# the training queues as failed, the others proceed.
#
# Submit (from a milan login node):
#   sbatch hpcjobs/registry_pt_prep.sbatch
# Work root: $REGISTRY_PT_WORK_ROOT, default below. PREP_WORKERS (default 2)
# library pretrainings run at a time.
#
# Shell discipline: no `set -e` (conda activation can return non-zero on
# success); activation verified by its effect; explicit exit code.
set -uo pipefail

ENV_PREFIX=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j0102
# --- conda activation (verified by effect, not by return code) --------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[regpt-prep] host=$(hostname) python=$PYBIN"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[regpt-prep] FATAL: parity env python not active (got: ${PYBIN:-none})"
     exit 1 ;;
esac

# --- thread caps (SLURM vars not assumed present) ---------------------------
# PySCF's OpenMP pool and numpy's OpenBLAS pool run at min(allocation, 8)
# threads each (parallel.PYSCF_POOL_THREADS_MAX): sized to the allocation the
# spin-waiting pools stall the small dense operations of a PySCF SCF loop
# (workflow-matrix job 2134488).
THREADS="${SLURM_CPUS_PER_TASK:-8}"
case "$THREADS" in ''|*[!0-9]*) THREADS=8 ;; esac
THREADS=$(( 10#$THREADS ))
[ "$THREADS" -ge 1 ] || THREADS=1
[ "$THREADS" -le 8 ] || THREADS=8
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
ulimit -u unlimited 2>/dev/null || true
echo "[regpt-prep] allocation=${SLURM_CPUS_PER_TASK:-unset} -> OMP/MKL/OPENBLAS=$THREADS"

REPO=/gpfs/projects/FernandezGroup/Alec/xcquinox
WORK_ROOT="${REGISTRY_PT_WORK_ROOT:-/gpfs/scratch/awills/xcquinox_runs/registry_pt_prongs}"
cd "$REPO" || { echo "[regpt-prep] FATAL: no checkout at $REPO"; exit 1; }
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 MALLOC_ARENA_MAX=2
export XCQUINOX_SEED_ALLOW_GENERATE=1
export PREP_WORKERS="${PREP_WORKERS:-2}"
mkdir -p "$WORK_ROOT" || { echo "[regpt-prep] FATAL: cannot create $WORK_ROOT"; exit 1; }
echo "[regpt-prep] work root: $WORK_ROOT; commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"

python -c "import nbclient, nbformat, xcquinox.pipeline" \
  || { echo "[regpt-prep] FATAL: import probe failed"; exit 1; }

# The driver runs in its own process group so the wall-clock TERM reaches the
# kernel and every child (each flushes its resume state and exits).
setsid python -u tools/run_registry_pt_stage.py --prong A --stage prep \
  --work-root "$WORK_ROOT" &
pid=$!
trap 'echo "[regpt-prep] TERM: forwarding to the driver group"; kill -TERM -- -"$pid" 2>/dev/null' TERM
wait "$pid"
rc=$?
while [ "$rc" -ge 128 ] && kill -0 "$pid" 2>/dev/null; do
  wait "$pid"
  rc=$?
done
echo "[regpt-prep] driver exit $rc"
exit "$rc"
