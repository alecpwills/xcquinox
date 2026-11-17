#!/usr/bin/env bash
#SBATCH --job-name=xcq_slim16_rerun
#SBATCH --partition=extended-96core
#SBATCH --time=168:00:00
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=96
#SBATCH --output=/gpfs/scratch/awills/slim16_rerun_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# The Slim16 re-evaluation of chosen networks of an existing run (the run
# directory, the only argument), one array task per network:
# slim16_eval.sbatch's array guard requires the array to cover the run's
# networks exactly, so a rerun of a subset goes through this script instead,
# submitted with the subset on the command line, e.g.
#   sbatch --array=1,2,7 hpcjobs/slim16_eval_rerun.sbatch RUN_DIR
# Each task runs hpcjobs/slim16_eval.py evaluate for its index -- the
# evaluation restarts from scratch (a pass consumes only the shard files its
# own invocation writes), and the earlier _shards scratch of a killed pass is
# left in place, unused. Milan (96-core) exclusive so the tasks run
# concurrently with the campaign's other Slim16 evaluations; the run's
# reference tables are not rewritten here: pbe_df.json and functionals_df.json
# come from the run's task 0 or slim16_tables.sbatch, and both write the same
# shard files, so never run this while that job runs on the same run.
#
# Shell discipline: no `set -e` -- conda activation can return non-zero on
# success; -uo pipefail, activation verified by its effect, an explicit rc gate.
set -uo pipefail

ENV_PREFIX=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j0102
REPO=/gpfs/projects/FernandezGroup/Alec/xcquinox
RUN="${1:?usage: sbatch --array=IDX[,IDX...] slim16_eval_rerun.sbatch RUN_DIR}"
IDX="${SLURM_ARRAY_TASK_ID:?submit with --array=IDX[,IDX...] on the command line}"

# --- email digest on every exit path (content, not just SLURM state) --------
LOG="/gpfs/scratch/awills/slim16_rerun_${SLURM_JOB_ID:-manual}.out"
source "$REPO/hpcjobs/job_digest.sh"
job_digest_arm slim16-rerun alec.wills@stonybrook.edu "$LOG" \
  "$RUN/checkpoints/$(printf 'spec_%04d' "$IDX")/eval_holdout_converged/*.json"

# --- conda activation (verified by effect, not by return code) --------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[slim16-rerun] host=$(hostname) python=$PYBIN"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[slim16-rerun] FATAL: parity env python not active (got: ${PYBIN:-none})"
     exit 1 ;;
esac

# --- thread caps (SLURM vars not assumed present) ---------------------------
# The same rule the evaluation array applies (slim16_eval.sbatch): two
# spin-waiting thread pools stall the small dense operations of a PySCF SCF
# loop, so OpenMP/MKL/OpenBLAS run at min(allocation, 8) threads.
THREADS="${SLURM_CPUS_PER_TASK:-8}"
case "$THREADS" in ''|*[!0-9]*) THREADS=8 ;; esac
THREADS=$(( 10#$THREADS ))
[ "$THREADS" -ge 1 ] || THREADS=1
[ "$THREADS" -le 8 ] || THREADS=8
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export MALLOC_ARENA_MAX=2
ulimit -u unlimited 2>/dev/null || true
echo "[slim16-rerun] allocation=${SLURM_CPUS_PER_TASK:-unset} -> OMP/MKL/OPENBLAS=$THREADS  MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX"

[ -d "$RUN" ] || { echo "[slim16-rerun] FATAL: run dir not found: $RUN"; exit 1; }
[ -f "$RUN/manifest.json" ] || { echo "[slim16-rerun] FATAL: $RUN carries no manifest.json"; exit 1; }
N_SPECS="$(grep -o '"n_specs": [0-9]*' "$RUN/manifest.json" | grep -o '[0-9]*$')"
[ -n "$N_SPECS" ] || { echo "[slim16-rerun] FATAL: no n_specs in $RUN/manifest.json"; exit 1; }
[ "$IDX" -lt "$N_SPECS" ] || { echo "[slim16-rerun] FATAL: index $IDX is outside the run's $N_SPECS networks"; exit 1; }

cd "$REPO" || { echo "[slim16-rerun] FATAL: cannot cd $REPO"; exit 1; }
T_START="$(date +%s)"
echo "[slim16-rerun] START $(date '+%Y-%m-%d %H:%M:%S %Z') job=${SLURM_JOB_ID:-none} run=$RUN idx=$IDX"
# -u: unbuffered, so the digest's log tail carries the last real progress line
# even when the job dies by signal.
python -u hpcjobs/slim16_eval.py evaluate "$RUN" "$IDX"
RC=$?
T_END="$(date +%s)"
echo "[slim16-rerun] END $(date '+%Y-%m-%d %H:%M:%S %Z') job=${SLURM_JOB_ID:-none} idx=$IDX rc=$RC elapsed $((T_END - T_START)) s"
exit "$RC"
