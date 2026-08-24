#!/usr/bin/env bash
#SBATCH --job-name=xcq_ewsweep
# long-40core: the sweep is ONE process on ONE node and needs a wall above the
# 4 h cap of the short-* queues. Override to match the login instance you
# submit from -- the *-96core-shared queues exist only on milan1/milan2 and the
# *-28core queues only on login1/login2:
#   sbatch --partition=long-96core-shared --cpus-per-task=96 ...
#   sbatch --partition=long-28core        --cpus-per-task=28 ...
#SBATCH --partition=long-40core
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/gpfs/scratch/awills/probe_pretrain_energy_weight_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Measure pretrain.energy_term_weight for campaign v6.
#
# WHAT: one pretraining run per (architecture, weight) cell of
# hpcjobs/probe_pretrain_energy_weight.py -- by default the three GGA-rung
# architectures against the PBE parent and the meta-GGA one against SCAN, at
# weights 0, 0.1, 1, 10, 100, 1000 steps each -- reporting per cell the two
# point-wise losses, the RMS per-system XC energy error and the MAXIMUM one.
# The maximum is the Section 3.3 certificate's own quantity (tol_atom = 1.0
# mHa is a max over systems, not an RMS); the RMS is what the objective
# minimizes. The table it writes is the number the v6 YAMLs then carry.
#
# WHY A JOB: the sweep is ~20 pretrainings over ~1e6 grid rows each. It is not
# a workstation task, and its result is a committed campaign parameter, so it
# runs once at a stated identity with the log kept beside the table.
#
# WALL DERIVATION (the 12 h request). One optimizer step of the pretraining
# loss with the energy term active, measured on four threads for
# deep_rung35_3x16, costs 7.22 ms at 20000 rows and 81.89 ms at 200000 rows:
# 0.415 us per row with a fixed cost below the measurement noise. The 37-system
# set at grid level 3 carries the production per-species grids (the grid is
# basis-independent at fixed level: H2O 32136 points, N2 26616, CH4 49408
# measured), so ~1.1e6 exchange rows -- every open shell contributes two spin
# channels on the per-channel footing -- and ~0.95e6 correlation rows. One cell
# is therefore 1000 x (1.1e6 + 0.95e6) x 0.415 us = 850 s, and twenty cells are
# 4.7 h. Generating the two parent-density files costs under an hour at
# def2-svp (the same set at the production 6-311++G(3df,2pd) basis measures
# 0.6 s for Li through 5.2 s for CH4 per species). Total estimate ~6 h; the
# request is twice that, and the four-thread measurement is itself
# conservative for a forty-core node.
#
# EXIT CODES. 0 = a weight cleared the gate on every architecture. 2 = the
# sweep ran to completion and NO weight did; the table is written and the
# finding is the table. Anything else is a failure of the sweep itself.
# SLURM mails FAIL on 2 as on any non-zero code -- read the verdict line in
# the log before treating it as a broken job.
#
# Shell discipline: no `set -e` (conda activation returns non-zero on success
# and would kill the job seconds in with an empty log); activation is verified
# by EFFECT, on the resolved interpreter path.
set -uo pipefail

ENV_PREFIX=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j070
REPO="${XCQ_REPO:-/gpfs/projects/FernandezGroup/Alec/xcquinox}"
RUN_ROOT="${EWSWEEP_RUN_ROOT:-/gpfs/scratch/awills/pretrain_energy_weight}"
DATA_DIR="${EWSWEEP_DATA_DIR:-${RUN_ROOT}/data}"
OUT="${EWSWEEP_OUT:-${RUN_ROOT}/energy_weight_table.json}"
LOG="${RUN_ROOT}/probe_${SLURM_JOB_ID:-manual}.log"

mkdir -p "${RUN_ROOT}" "${DATA_DIR}" || {
    echo "[ewsweep] FATAL: cannot create ${RUN_ROOT}"; exit 1; }

T_START="$(date +%s)"
echo "[ewsweep] ==================================================="
echo "[ewsweep] START  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[ewsweep] job=${SLURM_JOB_ID:-none}  host=$(hostname)"
echo "[ewsweep] repo=${REPO}"
echo "[ewsweep] data=${DATA_DIR}"
echo "[ewsweep] out=${OUT}"
echo "[ewsweep] log=${LOG}"
echo "[ewsweep] ==================================================="

# --- conda activation (verified by effect, not by return code) --------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[ewsweep] python=${PYBIN:-none}"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[ewsweep] FATAL: env python not active (got: ${PYBIN:-none})"
     exit 1 ;;
esac

cd "$REPO" || { echo "[ewsweep] FATAL: cannot cd $REPO"; exit 1; }
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
python -c "import xcquinox.alec.pretrain" \
    || { echo "[ewsweep] FATAL: repo import failed (env/activation)"; exit 1; }

# --- thread caps and JAX platform (SLURM vars not assumed present) ----------
THREADS="${SLURM_CPUS_PER_TASK:-40}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MALLOC_ARENA_MAX=2
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
echo "[ewsweep] BLAS threads=${THREADS}  MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX}"
echo "[ewsweep] JAX_PLATFORMS=${JAX_PLATFORMS} JAX_ENABLE_X64=${JAX_ENABLE_X64}"

# -u: unbuffered, so a job killed by signal still leaves its last real
# progress line in the log rather than discarding a buffered block.
python -u hpcjobs/probe_pretrain_energy_weight.py \
    --data-dir "$DATA_DIR" \
    --out      "$OUT" \
    --basis def2-svp \
    --grid-level 3 \
    --n-steps 1000 \
    2>&1 | tee "$LOG"
RC="${PIPESTATUS[0]}"

T_END="$(date +%s)"
echo "[ewsweep] ==================================================="
echo "[ewsweep] rc=${RC}  elapsed $((T_END - T_START)) s"
if [ -f "$OUT" ]; then
  echo "[ewsweep] table: $OUT"
else
  echo "[ewsweep] NO TABLE WRITTEN -- the sweep did not reach its end."
fi
case "$RC" in
  0) echo "[ewsweep] a swept weight cleared the gate on every architecture." ;;
  2) echo "[ewsweep] COMPLETED, gate NOT cleared by any swept weight."
     echo "[ewsweep] That is the finding; the table is written. Not a crash." ;;
  *) echo "[ewsweep] FAILED -- see the log above." ;;
esac
echo "[ewsweep] END  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[ewsweep] ==================================================="
exit "$RC"
