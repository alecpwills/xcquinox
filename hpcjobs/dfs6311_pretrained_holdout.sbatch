#!/usr/bin/env bash
#SBATCH --job-name=xcq_pretrained_holdout
# long-40core, NOT long-96core: the 96-core queue is saturated by the production
# training array and this job must not compete with it for nodes. The held-out
# loop shards across the node's CPUs, so 40 cores is ample for ~214 molecules.
#SBATCH --partition=long-40core
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=40
#SBATCH --output=/gpfs/scratch/awills/pretrained_holdout_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

# Held-out eval of the PRETRAINED (zero task training) meta-GGA network.
#
# WHY: the SCAN baseline came back at 4.89 kcal/mol combined over the full
# held-out pool -- better than PBE (11.78) and better than every trained cell in
# the sweep. deep_mgga_3x16, whose entire purpose is to clone SCAN, scores 26.62
# at subset_size 2: 5.4x worse than its own target. The net has diverged from
# what it was pretrained to reproduce; this job localizes WHERE.
#
#   pretrained scores near SCAN  -> the clone is faithful and TASK TRAINING on
#     1-4 molecules destroys it (the large-alpha distortion + the raw-alpha MLP
#     input become the suspects; the fix is training-side).
#   pretrained scores near 26.62 -> the clone was never faithful in DEPLOYMENT
#     despite matching SCAN's F_x to ~0.02 on the alpha<=1 slice, so the problem
#     is the SCF/descriptor path (F_c, the 3-cycle solver, the self-consistently
#     recomputed alpha).
#
# This is a RE-EVAL, not a training run: it loads the pretrain checkpoints the
# sweep already wrote and never touches the trained model.eqx. Output goes to
# <spec>/eval_holdout_pretrained/, a fresh subdir, so nothing the production
# array reads is modified.
#
# The eval protocol is NOT reimplemented -- the driver calls
# _eval_one_spec._run_held_out_eval, the same function the sweep's own eval task
# calls, so basis/grid, the held-out pool, the val/test slicing and the shard
# parallelism are identical. The only difference from the trained cell is which
# weights are in the model, which is what makes the numbers comparable.
#
# Shell discipline (job 2099698 lesson): NO `set -e` -- conda activation can
# return non-zero on success and would kill the job seconds in with an empty
# log. House pattern: -uo pipefail, activation verified by EFFECT (the resolved
# python path), explicit rc gate after the stage.
set -uo pipefail

ENV_PREFIX=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j070
REPO=/gpfs/projects/FernandezGroup/Alec/xcquinox
RUN="${1:-/gpfs/scratch/awills/xcquinox_runs/dfs_step7/dfs6311_grid3_v3/runs/run_20260728T140018Z}"
# spec 0034 = deep_mgga_3x16 at subset_size 2, the cell whose 26.62 this is
# compared against. Its molecules list drives the in-sample-overlap reaction
# filtering, so reusing it scores the pretrained net on the SAME reaction set.
SPEC="${2:-34}"

# --- email digest on every exit path (content, not just SLURM state) --------
LOG="/gpfs/scratch/awills/pretrained_holdout_${SLURM_JOB_ID:-manual}.out"
source "$REPO/hpcjobs/job_digest.sh"
job_digest_arm pre-holdout alec.wills@stonybrook.edu "$LOG" \
  "$RUN/checkpoints/$(printf 'spec_%04d' "$SPEC")/eval_holdout_pretrained/*.json"

T_START="$(date +%s)"
echo "[pre-holdout] ==================================================="
echo "[pre-holdout] START  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[pre-holdout] job=${SLURM_JOB_ID:-none}  run=$RUN  spec=$SPEC"
echo "[pre-holdout] ==================================================="

# --- conda activation (verified by effect, not by return code) --------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[pre-holdout] host=$(hostname) python=$PYBIN"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[pre-holdout] FATAL: parity env python not active (got: ${PYBIN:-none})"
     exit 1 ;;
esac

# --- thread caps (SLURM vars not assumed present) ---------------------------
THREADS="${SLURM_CPUS_PER_TASK:-40}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export MALLOC_ARENA_MAX=2
ulimit -u unlimited 2>/dev/null || true
echo "[pre-holdout] BLAS threads: $THREADS  MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX"

if [ ! -d "$RUN" ]; then
  echo "[pre-holdout] FATAL: run dir not found: $RUN"
  exit 1
fi
if [ ! -f "$RUN/pretrain/deep_mgga_3x16/xnet.eqx" ]; then
  echo "[pre-holdout] FATAL: pretrain checkpoints not found under $RUN/pretrain/"
  exit 1
fi

cd "$REPO" || { echo "[pre-holdout] FATAL: cannot cd $REPO"; exit 1; }
python hpcjobs/dfs6311_pretrained_holdout.py "$RUN" --spec "$SPEC"
RC=$?

T_END="$(date +%s)"
echo "[pre-holdout] ==================================================="
echo "[pre-holdout] rc=$RC  elapsed $((T_END - T_START)) s"
OUT="$RUN/checkpoints/$(printf 'spec_%04d' "$SPEC")/eval_holdout_pretrained"
ls -l "$OUT"/per_reaction.json 2>/dev/null || echo "[pre-holdout] no per_reaction.json written"
if [ "$RC" -ne 0 ]; then
  echo "[pre-holdout] FAILED -- see $OUT/failure.json if present."
fi
echo "[pre-holdout] END  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[pre-holdout] ==================================================="
exit "$RC"
