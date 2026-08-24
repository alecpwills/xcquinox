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
# hpcjobs/probe_pretrain_energy_weight.py -- by default the five PBE-parent
# architectures (deep_3x16, deep_cusp_3x16 and the three rung-3.5 networks
# deep_rung35_3x16, deep_rung35_attn_3x16 and deep_rung35ms_3x16; the parent
# is the density, not the rung) and the meta-GGA deep_mgga_3x16 against SCAN,
# at weights 0, 0.1, 1, 10 and 100 (1000 optimizer steps per cell) -- reporting
# per cell the two point-wise losses, the RMS per-system
# XC energy error, and BOTH quantities the Section 3.3 certificate gates on:
# the maximum |dE_xc| over the ATOMS against tol_atom = 1.0 mHa, and the
# maximum |dAE| over the molecules against tol_AE = 1.0 kcal/mol, both of
# them maxima and not means, and neither implying the other: an error shared
# by a molecule and its atoms cancels out of an atomization energy. The sweep
# runs the POLARIZED correlation objective, the one every configuration under
# hpcjobs/configs/ sets and the one the weight will be applied to. The table
# it writes is the number the v6 YAMLs carry.
#
# WHY THESE SIX. They are the six largest pre-certificate atomization-energy
# offsets on record (SPEC_pretrain_fidelity_program.md Section 2, worst of the
# H2O / N2 / CH4 triple in kcal/mol): deep_rung35_attn_3x16 56.1,
# deep_mgga_3x16 55.9, deep_rung35ms_3x16 42.8, deep_rung35_3x16 29.1,
# deep_cusp_3x16 25.7, deep_3x16 4.2. The one architecture that table measured
# and this set leaves out, deep_attn_3x16, is its smallest offset (4.1) and
# carries no descriptor deep_3x16 does not. The set is the worst offenders
# rather than a sample because campaign v6's train array depends afterok on
# the pretrain ARRAY: ONE architecture whose certificate fails blocks all 341
# cells, not its own eleven, so a weight measured only where the offsets were
# small is extrapolated onto exactly the architectures that decide whether the
# campaign runs at all. Both additions are PBE-parent, so the sweep still
# generates two data files and the data allowance below is unchanged.
#
# WHY A JOB: the sweep is ~30 pretrainings over ~2.4e6 rows each. It is not
# a workstation task, and its result is a committed campaign parameter, so it
# runs once at a stated identity with the log kept beside the table.
#
# WALL DERIVATION (the 24 h request). Measured on the path this job actually
# runs -- the probe's own smoke cell, deep_rung35_3x16, energy term active,
# polarized, four threads -- as the SLOPE of the cell wall against the step
# count (100 and 1100 steps, or 50 and 550 on the two largest, so the data
# load, the descriptor assembly and the single compile cancel), at four row
# counts in one session:
#
#     16350 rows (sto-3g,   level 1)     5.916 ms  per optimizer step
#     48660 rows (def2-svp, level 3)    20.734 ms
#    137218 rows (def2-svp, level 5)    56.661 ms
#    275588 rows (def2-svp, level 7)   119.389 ms
#
# One step is both networks, and a row is an exchange-block row or a
# correlation-block row. Least squares: per step = -1.36 ms + 0.4356 us/row,
# residuals under 1.8 ms. The intercept is smaller than those residuals and
# smaller than the run-to-run spread at a fixed row count (5.92 ms here
# against 4.69 ms at 16350 rows in a second session on the same box), so
# there is no fixed per-step cost: the training loop holds ONE filter_jit for
# the whole run and compiles the update step once
# (xcquinox/alec/pretrain.py, _train_pretrain_network). The larger of the two
# sessions' slopes, 0.5218 us/row, is the one carried below.
#
# Row count of the production sweep. The 37-system set at def2-svp level 3
# carries 1287200 Becke-Lebedev points, of which the mean field integrates
# 1149856: pyscf drops the negligible-density points of the initial guess
# before its first get_veff, and that pruned grid is the one the generator
# rebuilds and the rows are drawn from. Twelve of the 37 systems are open
# shell (168344 pruned points, 14.6 percent); the other 25 are closed
# (981512).
#
# ONLY an open shell contributes two exchange blocks, and one open shell in
# this set contributes fewer. _system_columns writes the per-channel exchange
# rows under the spin_channel footing when the system is UKS and falls back to
# the total-density rows otherwise ("x_rows is None for a closed-shell system,
# whose total-density rows already are that footing"), so the exchange block
# spans 2 x open + 1 x closed = 1318200 points, not 2 x 1287200. The exception
# is the HYDROGEN atom: its beta channel is empty, spin_channel_exchange_rows
# writes no rows for it, and H measures rows_x = 9176 = 1x its 9176 pruned
# points rather than 2x. The doubled term below keeps H's grid in it, which
# leaves the exchange row estimate about 8.9e3 rows high -- 0.69 percent, in
# the CONSERVATIVE direction, so the figures are carried as they stand.
#
# The generator then drops the points under its density floor. Measured at
# this identity as rows over pruned points: 0.9775 on closed shells (Be, CH4,
# C2H2), and on open shells 0.9743 for the correlation block and 0.9663 per
# exchange CHANNEL (Li, Cl, O). The He/Li smoke pair is not a proxy for the
# exchange block -- it is 63 percent open shell by grid mass against the
# production set's 15, and the open/closed split is exactly what decides one
# exchange block or two. Weighting the measured fractions by the set's own
# grid mass:
#
#     correlation ~ 0.9775 x 981512 + 0.9743 x 168344         = 1.123e6 rows
#     exchange    ~ 0.9775 x 981512 + 2 x 0.9663 x 168344     = 1.285e6 rows
#     total       ~ 2.41e6 rows
#
# One cell is then 1000 x 2.41e6 x 0.5218 us = 1257 s, and thirty cells
# (6 architectures x 5 weights) are 10.5 h. Generating the two parent-density
# files is allowed 1 h -- the allowance is a property of the PARENT set, not
# of the architecture count, and both architectures added to the default set
# are PBE-parent. Estimate 11.5 h, so a 2x margin is 23.0 h, inside the wall.
#
# The superseded pre-rework path, before the training loop moved in-module,
# re-compiled its update step every optimizer step at a fixed 648 ms measured:
# a production step of 1.91 s, the sweep at 15.9 h and the estimate at 16.9 h.
# Thirty cells on that path still fit the 24 h wall, but at 1.4x rather than
# 2x -- at twenty cells it was 23.2 h and the margin held. So if the loop ever
# regresses to recompiling per step, SPLIT the sweep with EWSWEEP_ARCHS rather
# than trusting the wall. The four-thread measurement is itself conservative
# for a forty-core node.
#
# If a queue will not take 24 h, split the sweep instead of shortening it:
# EWSWEEP_ARCHS names one batch of architectures per submission and --resume
# accumulates them into the one table (see below).
#
# EXIT CODES. 0 = a weight cleared the gate on every architecture MEASURED
# -- this submission's batch plus whatever the resumed table already held,
# named on the recommendation line and not necessarily the six defaults.
# 3 = an exception escaped the sweep; its traceback is in the log and the
# partial table is still written. 1 = a cell failed, or the sweep refused its
# own arguments (an unknown architecture name, a non-positive step count).
#
# 2 IS TWO OUTCOMES, and the epilogue tells them apart rather than reporting
# both as the first. (a) The sweep ran to completion and NO weight cleared;
# the table is written and the finding is the table. (b) ARGPARSE refused the
# command line -- its usage code is also 2, it writes no table, and no cell
# ran. (b) is reachable from here: EWSWEEP_ARCHS is the only user-supplied
# token, and a value beginning with `-` is parsed as an option rather than as
# an architecture list. The epilogue reads THIS submission's output (RUN_OUT,
# separate from the appended $LOG) for argparse's own `usage:` line and says
# which of the two happened. SLURM mails FAIL on 2 as on any non-zero code --
# read that line in the log before treating it as a broken job.
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
# THIS submission's output alone. $LOG is appended across resubmissions, so it
# cannot answer "what did the run that just exited print"; the epilogue needs
# that to tell the sweep's exit 2 (a verdict) from argparse's (a refusal).
RUN_OUT="${RUN_ROOT}/probe_${SLURM_JOB_ID:-manual}.last"
# Empty (the default) sweeps the probe's own six architectures. A
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
echo "[ewsweep] last=${RUN_OUT}"
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
    echo "[ewsweep] batch: the probe's six default architectures"
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
    2>&1 | tee -a "$LOG" | tee "$RUN_OUT"
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
  0) echo "[ewsweep] a swept weight cleared both gates on every architecture"
     echo "[ewsweep] IN THE TABLE. This submission measured:"
     echo "[ewsweep]   ${ARCHS:-the six default architectures}"
     echo "[ewsweep] The recommendation line above names the architectures the"
     echo "[ewsweep] verdict covers: a batch clears only what it measured." ;;
  2) if grep -q "^usage: probe_pretrain_energy_weight" "$RUN_OUT" 2>/dev/null
     then
       echo "[ewsweep] REFUSED: argparse rejected the command line (its usage"
       echo "[ewsweep] code is 2, the same number the sweep uses for its own"
       echo "[ewsweep] finding). NOT a finding -- no cell ran and no table was"
       echo "[ewsweep] written by this submission. EWSWEEP_ARCHS is the only"
       echo "[ewsweep] token this script takes from the caller; a value that"
       echo "[ewsweep] begins with '-' is read as an option. Got:"
       echo "[ewsweep]   EWSWEEP_ARCHS=${ARCHS:-<unset>}"
     else
       echo "[ewsweep] COMPLETED, the gates NOT cleared by any swept weight."
       echo "[ewsweep] That is the finding; the table is written. Not a crash."
     fi ;;
  3) echo "[ewsweep] an exception ESCAPED the sweep -- traceback above."
     echo "[ewsweep] The partial table is written; --resume carries it." ;;
  *) echo "[ewsweep] FAILED -- see the log above." ;;
esac
# A job killed at the wall still leaves every cell it finished in $OUT.
# Re-submitting this script with the same EWSWEEP_OUT resumes from them.
echo "[ewsweep] resume: sbatch $0   (continues from the cells in $OUT)"
echo "[ewsweep] END  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[ewsweep] ==================================================="
exit "$RC"
