#!/usr/bin/env bash
#SBATCH --job-name=gf_slim16_prepare
#SBATCH --partition=extended-96core
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/scratch/awills/gf_prepare_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# The Slim16 prepare step of the GGA-final arm, chained afterok on the arm's
# pretrain array by hpcjobs/gga_final_submit.sh (submitting it standalone
# against a finished pretrain run also works). It reads the ledger the driver
# wrote (hpcjobs/gga_final_slim16_ledger.py) and runs slim16_eval.py prepare,
# which deserialises each pretrain checkpoint (a load check in its own right,
# one spec per network) under hpcjobs/slim16_eval.py's manifest contract; the
# run directory prepare prints is recorded for the eval wrapper. No arguments:
# every path is fixed below.
#
# Shell discipline: no `set -e` -- conda activation can return non-zero on
# success; -uo pipefail, activation verified by its effect, an explicit rc gate.
set -uo pipefail

ENV_PREFIX=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j0102
REPO=/gpfs/projects/FernandezGroup/Alec/xcquinox
ARM=/gpfs/scratch/awills/xcquinox_runs/dfs_step8/gga_final
LEDGER="$ARM/slim16_ledger.json"
EVAL_ROOT=/gpfs/scratch/awills/xcquinox_runs/dfs_step8/slim16_eval
RUNDIR_FILE="$ARM/slim16_rundir.txt"

# --- conda activation (verified by effect, not by return code) --------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[gf-prepare] host=$(hostname) python=$PYBIN"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[gf-prepare] FATAL: parity env python not active (got: ${PYBIN:-none})"
     exit 1 ;;
esac

# --- thread caps (SLURM vars not assumed present) ---------------------------
# The pools serve the checkpoint deserialisation and any numerics prepare
# touches; capped at parallel.PYSCF_POOL_THREADS_MAX, the rule the stage
# templates and slim16_eval.sbatch apply.
THREADS="${SLURM_CPUS_PER_TASK:-8}"
case "$THREADS" in ''|*[!0-9]*) THREADS=8 ;; esac
THREADS=$(( 10#$THREADS ))
[ "$THREADS" -ge 1 ] || THREADS=1
[ "$THREADS" -le 8 ] || THREADS=8
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export MALLOC_ARENA_MAX=2

[ -f "$LEDGER" ] || { echo "[gf-prepare] FATAL: no ledger at $LEDGER (run gga_final_submit.sh)"; exit 1; }
cd "$REPO" || { echo "[gf-prepare] FATAL: cannot cd $REPO"; exit 1; }
echo "[gf-prepare] START $(date '+%Y-%m-%d %H:%M:%S %Z') ledger=$LEDGER"
python -u hpcjobs/slim16_eval.py prepare --networks "$LEDGER" --run-root "$EVAL_ROOT"
RC=$?
if [ "$RC" -ne 0 ]; then
  echo "[gf-prepare] FATAL: prepare rc=$RC; the run-dir file is left untouched"
  exit "$RC"
fi
# prepare creates EVAL_ROOT/runs/run_<UTC-stamp> and prints it last; the
# newest run directory is the one just written (a failed prepare exits above)
ls -1dt "$EVAL_ROOT"/runs/run_* | head -1 > "$RUNDIR_FILE"
echo "[gf-prepare] eval run dir recorded:"
cat "$RUNDIR_FILE"
echo "[gf-prepare] END $(date '+%Y-%m-%d %H:%M:%S %Z') rc=0"
exit 0
