#!/usr/bin/env bash
#SBATCH --job-name=xcq_nan_verify
# long-40core, NOT long-96core: the production training array saturates the
# 96-core queue and this verification must not compete with it for nodes.
#SBATCH --partition=long-40core
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=40
#SBATCH --output=/gpfs/scratch/awills/dfs6311_nan_verify_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
# =============================================================================
# Corrected-V_xc verification on the meta-GGA path, at production scale.
# Runs dfs6311_nan_verify.py (three legs) with the run's own PRETRAINED
# deep_mgga_3x16 / deep_rung35_mgga_3x16 checkpoints at 6-311++G(3df,2pd):
#
#   leg 1 (assertion)   25-cycle FULL-SCF training gradient finite at every
#                       leaf and nonzero (the alpha-tail regression after the
#                       compute_alpha gradient-freeze removal).
#   leg 2 (measurement) converged-energy shift, corrected potential vs the
#                       prior assembly without the feature-response term.
#   leg 3 (measurement) convergence behaviour (energy vs cycle cap, converged
#                       flags), corrected vs prior.
#
# BLAS IS PINNED TO ONE THREAD, deliberately: at 25 unrolled cycles an
# unconverged diffuse-basis atom's energy and gradient magnitude wander
# between identical multithreaded runs (measured: 60 mHa and eight orders in
# |g|max across four repeats), while single-threaded runs are bit-identical.
# A threaded single run is not evidence; do not raise the thread count here
# for speed.
#
# Shell discipline (job 2099698 lesson): NO `set -e` -- conda activation can
# return non-zero on success and would kill the job seconds in with an empty
# log. House pattern: -uo pipefail, activation verified by EFFECT (the
# resolved python path), explicit rc gate after the stage.
set -uo pipefail

ENV_PREFIX=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j070
REPO=/gpfs/projects/FernandezGroup/Alec/xcquinox
RUN="${1:-/gpfs/scratch/awills/xcquinox_runs/dfs_step7/dfs6311_grid3_v3/runs/run_20260728T140018Z}"
GRID="${2:-3}"
CYCLES="${3:-25}"

# --- email digest on every exit path (content, not just SLURM state): the
#     rc, error-context lines, log tail and report tail are mailed so a
#     failure can be diagnosed from the email alone -------------------------
LOG="/gpfs/scratch/awills/dfs6311_nan_verify_${SLURM_JOB_ID:-manual}.out"
source "$REPO/hpcjobs/job_digest.sh"
job_digest_arm nan-verify alec.wills@stonybrook.edu "$LOG" \
  "$RUN/nan_verify_report_grid*.json"

T_START="$(date +%s)"
echo "[nan-verify] ==================================================="
echo "[nan-verify] START  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[nan-verify] job=${SLURM_JOB_ID:-none} run=$RUN grid=$GRID cycles=$CYCLES"
echo "[nan-verify] ==================================================="

# --- conda activation (verified by effect, not by return code) --------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[nan-verify] host=$(hostname) python=$PYBIN"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[nan-verify] FATAL: parity env python not active (got: ${PYBIN:-none})"
     exit 1 ;;
esac

# --- single-threaded BLAS (see header; a correctness pin, not a performance
#     setting) ----------------------------------------------------------------
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export MALLOC_ARENA_MAX=2
ulimit -u unlimited 2>/dev/null || true
echo "[nan-verify] BLAS threads pinned to 1 (reproducibility requirement)"

if [ ! -d "$RUN" ]; then
  echo "[nan-verify] FATAL: run dir not found: $RUN"
  exit 1
fi
if [ ! -f "$RUN/pretrain/deep_mgga_3x16/xnet.eqx" ]; then
  echo "[nan-verify] FATAL: pretrain checkpoints not found under $RUN/pretrain/"
  exit 1
fi

cd "$REPO" || { echo "[nan-verify] FATAL: cannot cd $REPO"; exit 1; }
# -u: unbuffered, so per-leg progress streams into the SLURM log as it runs.
python -u hpcjobs/dfs6311_nan_verify.py "$RUN" --grid-level "$GRID" --cycles "$CYCLES"
RC=$?

T_END="$(date +%s)"
echo "[nan-verify] ==================================================="
echo "[nan-verify] rc=$RC  elapsed $((T_END - T_START)) s"
ls -l "$RUN"/nan_verify_report_grid"$GRID".json 2>/dev/null \
  || echo "[nan-verify] no report json written"
if [ "$RC" -ne 0 ]; then
  echo "[nan-verify] LEG-1 FINITENESS FAILURE -- the gradient-freeze removal"
  echo "[nan-verify] does not stand at production scale; design the smooth"
  echo "[nan-verify] energy-level damping fallback before re-running cells."
fi
echo "[nan-verify] END  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[nan-verify] ==================================================="
exit "$RC"
