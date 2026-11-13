#!/usr/bin/env bash
#SBATCH --job-name=xcq_slim16_tables
#SBATCH --partition=extended-40core
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=40
#SBATCH --output=/gpfs/scratch/awills/slim16_tables_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# The reference tables of a Slim16 evaluation run already evaluated (the run
# directory, the first argument): the PBE-DF table (pbe_df.json; cache hits on the
# run's own cache where the evaluation job wrote it) and the comparator tables of
# r2SCAN, B3LYP and wB97M-V on the same footing (functionals_df.json), both beside
# the run, through hpcjobs/slim16_eval.py pbe-df and comparators-df. The evaluation
# job (slim16_eval.sbatch) runs the same two on its task 0; this job adds them to a
# run evaluated before the comparators existed, or refreshes them, without
# re-evaluating the networks. Not while that task 0 is running on the same run:
# both write the same shard files. The summaries pull carries both files.
#
# Wall: the three comparator sweeps over the 239 species take about five hours at
# 24 workers, wB97M-V's VV10 term being most of it; 24 h is requested.
#
# Shell discipline: no `set -e` -- conda activation can return non-zero on
# success; -uo pipefail, activation verified by its effect, an explicit rc gate.
set -uo pipefail

ENV_PREFIX=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j0102
REPO=/gpfs/projects/FernandezGroup/Alec/xcquinox
RUN="${1:?the run directory of the evaluation (slim16_eval.py prepare printed it)}"

# --- email digest on every exit path (content, not just SLURM state) --------
LOG="/gpfs/scratch/awills/slim16_tables_${SLURM_JOB_ID:-manual}.out"
source "$REPO/hpcjobs/job_digest.sh"
job_digest_arm slim16-tables alec.wills@stonybrook.edu "$LOG" "$RUN/functionals_df.json"

# --- conda activation (verified by effect, not by return code) --------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[slim16-tables] host=$(hostname) python=$PYBIN"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[slim16-tables] FATAL: parity env python not active (got: ${PYBIN:-none})"
     exit 1 ;;
esac

# --- thread caps (SLURM vars not assumed present) ---------------------------
# PySCF's OpenMP pool and numpy's OpenBLAS pool run at min(allocation, 8)
# threads each (parallel.PYSCF_POOL_THREADS_MAX, the rule the stage templates
# and the energy-weight sweep apply): sized to the allocation, the two
# spin-waiting pools stall the small dense operations of a PySCF SCF loop --
# measured at about ten minutes per molecule at 40 threads on a 40-core node
# against 8 s at four (workflow-matrix job 2134488).
THREADS="${SLURM_CPUS_PER_TASK:-8}"
case "$THREADS" in ''|*[!0-9]*) THREADS=8 ;; esac
THREADS=$(( 10#$THREADS ))
[ "$THREADS" -ge 1 ] || THREADS=1
[ "$THREADS" -le 8 ] || THREADS=8
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export MALLOC_ARENA_MAX=2
ulimit -u unlimited 2>/dev/null || true
echo "[slim16-tables] allocation=${SLURM_CPUS_PER_TASK:-unset} -> OMP/MKL/OPENBLAS=$THREADS  MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX"

[ -d "$RUN" ] || { echo "[slim16-tables] FATAL: run dir not found: $RUN"; exit 1; }
[ -f "$RUN/manifest.json" ] || { echo "[slim16-tables] FATAL: $RUN carries no manifest.json; it is not an evaluation run"; exit 1; }

cd "$REPO" || { echo "[slim16-tables] FATAL: cannot cd $REPO"; exit 1; }
T_START="$(date +%s)"
echo "[slim16-tables] START $(date '+%Y-%m-%d %H:%M:%S %Z') job=${SLURM_JOB_ID:-none} run=$RUN"
# -u: unbuffered, so the digest's log tail carries the last real progress line
# even when the job dies by signal. The first failure is the job's status.
python -u hpcjobs/slim16_eval.py pbe-df "$RUN"
RC=$?
python -u hpcjobs/slim16_eval.py comparators-df "$RUN"
RC2=$?
[ "$RC" -ne 0 ] || RC=$RC2
T_END="$(date +%s)"
echo "[slim16-tables] END $(date '+%Y-%m-%d %H:%M:%S %Z') rc=$RC elapsed $((T_END - T_START)) s"
exit "$RC"
