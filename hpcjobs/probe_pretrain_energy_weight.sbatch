#!/usr/bin/env bash
#SBATCH --job-name=xcq_ewsweep
# long-40core: the sweep is ONE process on ONE node and needs a wall above the
# 4 h cap of the short-* queues. Override to match the login instance you
# submit from -- the *-96core-shared queues exist only on milan1/milan2 and the
# *-28core queues only on login1/login2:
#   sbatch --partition=long-96core-shared --cpus-per-task=96 ...
#   sbatch --partition=long-28core        --cpus-per-task=28 ...
#SBATCH --partition=long-40core
#SBATCH --time=24:00:00
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
# point-wise losses, the RMS per-system XC energy error, and BOTH quantities
# the Section 3.3 certificate gates on: the maximum |dE_xc| over the ATOMS
# against tol_atom = 1.0 mHa, and the maximum |dAE| over the molecules against
# tol_AE = 1.0 kcal/mol. Both are maxima, not means, and neither implies the
# other: an error shared by a molecule and its atoms cancels out of an
# atomization energy. The sweep runs the POLARIZED correlation objective, the
# one every configuration under hpcjobs/configs/ sets and the one the weight
# will be applied to. The table it writes is the number the v6 YAMLs carry.
#
# WHY A JOB: the sweep is ~20 pretrainings over ~1e6 grid rows each. It is not
# a workstation task, and its result is a committed campaign parameter, so it
# runs once at a stated identity with the log kept beside the table.
#
# WALL DERIVATION (the 24 h request). Measured on the path this job actually
# runs -- the probe's own smoke cell, deep_rung35_3x16, energy term active,
# polarized, four threads -- as the SLOPE of the cell wall against the step
# count (10 and 60 steps, so the data load, the descriptor assembly and the
# first compile cancel), at three row counts, twice:
#
#                                       run A       run B
#     16350 rows (sto-3g, level 1)     6.380 ms    9.460 ms  per optimizer step
#     48660 rows (def2-svp, level 3)  21.360 ms   20.820 ms
#    566674 rows (def2-svp, level 9) 255.660 ms  222.760 ms
#
# One step is both networks. Least squares: per step = -0.85 ms + 0.4527 us/row
# (A) and +2.53 ms + 0.3886 us/row (B), residuals under 0.7 ms -- the cost is
# the rows, with no fixed cost per step above the spread between runs. The
# larger slope, 0.4527 us/row, is the one carried below.
#
# Row count of the production sweep. The 37-system set at def2-svp level 3
# carries 1287200 quadrature points (counted, per species, from the grids
# themselves) and 2564592 exchange points, every occupied spin channel
# contributing its own block on the per-channel footing. The generator drops
# points below its density floor; measured at this identity on the smoke pair,
# 85.0 percent of the correlation points and 69.8 percent of the exchange
# points survive, giving ~1.09e6 correlation and ~1.79e6 exchange rows, 2.88e6
# in total. One cell is then 1000 x 2.88e6 x 0.4527 us = 1305 s, and twenty
# cells (4 architectures x 5 weights) are 7.25 h. Generating the two
# parent-density files is allowed 1 h. Estimate 8.25 h, so a 2x margin is
# 16.5 h.
#
# The 24 h request also covers the pre-fix library: before the unvalidated
# pretraining path stopped re-compiling its update step every optimizer step,
# the same cell cost 572.6 ms per step at 16350 rows, i.e. a fixed 566 ms of
# compilation on top of the row term. That puts a production step at 1.87 s,
# a cell at 1871 s, the sweep at 10.4 h and the estimate at 11.4 h -- 22.8 h at
# the same 2x margin. One wall covers both, and the four-thread measurement is
# itself conservative for a forty-core node.
#
# If a queue will not take 24 h, split the sweep instead of shortening it:
# EWSWEEP_ARCHS names one batch of architectures per submission and --resume
# accumulates them into the one table (see below).
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
# Empty (the default) sweeps the probe's own four architectures. A
# comma-separated list sweeps those instead, which is how the sweep is split
# across walls: submit one batch per list, each writing into the SAME $OUT.
# `-` and not `:-`, so an explicitly empty EWSWEEP_ARCHS is not silently the
# whole set.
ARCHS="${EWSWEEP_ARCHS-}"

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

# The architecture selection, as an array so an empty list adds no argument.
# The `+` expansion is what keeps `set -u` from tripping on an empty array
# under bash 4.2/4.3.
ARCH_ARGS=()
if [ -n "$ARCHS" ]; then
    ARCH_ARGS=(--archs "$ARCHS")
    echo "[ewsweep] batch: architectures ${ARCHS} (accumulating into ${OUT})"
else
    echo "[ewsweep] batch: the probe's four default architectures"
fi

# --resume: the table is rewritten after every cell, so a job killed at the
# wall (or a batch submitted after another) carries the finished cells over
# instead of paying for them twice. The identity is checked before anything is
# reused, so a table written at another basis, grid, step count, seed or
# polarization is refused rather than merged.
# -u: unbuffered, so a job killed by signal still leaves its last real
# progress line in the log rather than discarding a buffered block.
python -u hpcjobs/probe_pretrain_energy_weight.py \
    --data-dir "$DATA_DIR" \
    --out      "$OUT" \
    --basis def2-svp \
    --grid-level 3 \
    --n-steps 1000 \
    --polarized \
    --resume \
    ${ARCH_ARGS[@]+"${ARCH_ARGS[@]}"} \
    2>&1 | tee -a "$LOG"
RC="${PIPESTATUS[0]}"

T_END="$(date +%s)"
echo "[ewsweep] ==================================================="
echo "[ewsweep] rc=${RC}  elapsed $((T_END - T_START)) s"
if [ -f "$OUT" ]; then
  echo "[ewsweep] table: $OUT"
else
  echo "[ewsweep] NO TABLE WRITTEN -- not one cell finished."
fi
case "$RC" in
  0) echo "[ewsweep] a swept weight cleared both gates on every architecture." ;;
  2) echo "[ewsweep] COMPLETED, the gates NOT cleared by any swept weight."
     echo "[ewsweep] That is the finding; the table is written. Not a crash." ;;
  *) echo "[ewsweep] FAILED -- see the log above." ;;
esac
# A job killed at the wall still leaves every cell it finished in $OUT.
# Re-submitting this script with the same EWSWEEP_OUT resumes from them.
echo "[ewsweep] resume: sbatch $0   (continues from the cells in $OUT)"
echo "[ewsweep] END  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[ewsweep] ==================================================="
exit "$RC"
