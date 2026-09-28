#!/usr/bin/env bash
#SBATCH --job-name=xcq_slim16_eval
#SBATCH --partition=extended-40core
#SBATCH --time=96:00:00
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=40
#SBATCH --output=/gpfs/scratch/awills/slim16_eval_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# The Slim16 evaluation of the networks of hpcjobs/ledgers/slim16_eval_networks.json,
# one array task per network, on the run directory `slim16_eval.py prepare` wrote
# (the first argument). Each task runs the harness's converged channel over the
# Slim16 set at def2-TZVP / grid level 4 with density fitting (hpcjobs/slim16_eval.py);
# task 0 then writes the PBE-DF table of the set beside the run. The outputs land
# under <run>/checkpoints/spec_NNNN/eval_holdout_converged/ and <run>/pbe_df.json,
# which the summaries pull carries.
#
# Shell discipline: no `set -e` -- conda activation can return non-zero on
# success; -uo pipefail, activation verified by its effect, an explicit rc gate.
set -uo pipefail

ENV_PREFIX=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j0102
REPO=/gpfs/projects/FernandezGroup/Alec/xcquinox
RUN="${1:?the run directory slim16_eval.py prepare printed}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

# --- email digest on every exit path (content, not just SLURM state) --------
LOG="/gpfs/scratch/awills/slim16_eval_${SLURM_JOB_ID:-manual}.out"
source "$REPO/hpcjobs/job_digest.sh"
job_digest_arm slim16-eval alec.wills@stonybrook.edu "$LOG" \
  "$RUN/checkpoints/$(printf 'spec_%04d' "$IDX")/eval_holdout_converged/*.json"

# --- conda activation (verified by effect, not by return code) --------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[slim16-eval] host=$(hostname) python=$PYBIN"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[slim16-eval] FATAL: parity env python not active (got: ${PYBIN:-none})"
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
echo "[slim16-eval] allocation=${SLURM_CPUS_PER_TASK:-unset} -> OMP/MKL/OPENBLAS=$THREADS  MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX"

[ -d "$RUN" ] || { echo "[slim16-eval] FATAL: run dir not found: $RUN"; exit 1; }
[ -f "$RUN/manifest.json" ] || { echo "[slim16-eval] FATAL: $RUN carries no manifest.json; run prepare first"; exit 1; }
# the array must cover the run's networks exactly: a grown ledger would otherwise
# leave networks unevaluated in silence
N_SPECS="$(grep -o '"n_specs": [0-9]*' "$RUN/manifest.json" | grep -o '[0-9]*$')"
[ -n "$N_SPECS" ] || { echo "[slim16-eval] FATAL: no n_specs in $RUN/manifest.json"; exit 1; }
if [ -n "${SLURM_ARRAY_TASK_MAX:-}" ] && [ "$((SLURM_ARRAY_TASK_MAX + 1))" -ne "$N_SPECS" ]; then
  echo "[slim16-eval] FATAL: the array covers tasks 0-$SLURM_ARRAY_TASK_MAX and the run holds $N_SPECS networks; submit with --array=0-$((N_SPECS - 1))"
  exit 1
fi
[ "$IDX" -lt "$N_SPECS" ] || { echo "[slim16-eval] FATAL: task $IDX is outside the run's $N_SPECS networks"; exit 1; }

cd "$REPO" || { echo "[slim16-eval] FATAL: cannot cd $REPO"; exit 1; }
T_START="$(date +%s)"
echo "[slim16-eval] START $(date '+%Y-%m-%d %H:%M:%S %Z') job=${SLURM_JOB_ID:-none} run=$RUN idx=$IDX"
# -u: unbuffered, so the digest's log tail carries the last real progress line
# even when the job dies by signal.
python -u hpcjobs/slim16_eval.py evaluate "$RUN" "$IDX"
RC=$?
if [ "$IDX" -eq 0 ]; then
  python -u hpcjobs/slim16_eval.py pbe-df "$RUN"
  RC2=$?
  [ "$RC" -ne 0 ] || RC=$RC2
fi
T_END="$(date +%s)"
echo "[slim16-eval] END $(date '+%Y-%m-%d %H:%M:%S %Z') rc=$RC elapsed $((T_END - T_START)) s"
exit "$RC"
