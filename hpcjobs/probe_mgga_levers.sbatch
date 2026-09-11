#!/usr/bin/env bash
#SBATCH --job-name=xcq_mgga_levers
# long-96core, submitted from milan1/milan2: ONE process on ONE node for fourteen
# pretrainings in parity with the v7 families meta-GGA run at half its step count, each
# followed by the run's own fidelity certificate. SEAWULF_RUNBOOK.md:133 places the
# *-96core-SHARED queues on that SLURM instance and never names the non-shared
# long-96core, which seven standalone scripts here nevertheless use; every long-* QOS caps
# MaxWall at 48 h (runbook line 170, sacctmgr 2026-08-27). Before the first submission, on
# the login node:
#     sinfo -h -p long-96core -o "%P %l"          (no output: use --partition=long-96core-shared)
#     sacctmgr show qos format=name,maxwall%15 | grep long
# The 96-core queue is also where the production training array runs: a whole-node 48 h
# request may sit pending behind it. `sbatch --partition=long-96core-shared` overrides the
# directive below without editing the file.
#SBATCH --partition=long-96core
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 24, not 96, and NOT a throttle: `--exclusive --mem=0` still takes the whole node, and the
# allocation is what sizes XLA's intra-op pool below. The cloned run's pretrain task was
# rendered at cluster.cpus_per_task 24 (the families YAML), so 24 is what its fits ran at
# and what the cost figures below are calibrated on; asking for 96 here would fit at a pool
# four times the one this study claims parity with, for no measured gain.
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/gpfs/scratch/awills/mgga_levers_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# THE META-GGA LEVER STUDY (2026-09-11). The four learned clones of SCAN in the v7 families
# meta-GGA run (array 2145894) failed the fidelity certificate: atoms 2.1 to 4.1 mHa against
# 1.0, worst atomization 3.3 to 7.3 kcal/mol against 2.0. This job measures, at the run's
# own identity, which lever moves the certificate: the energy-term weight (0.1, the run's;
# 10; 100), the network width and depth (deep_mgga_3x16, the run's; 3x32; 4x16; 4x32), and
# the objective the failed run fitted under (integration, the (r_s, s, alpha) mesh at its
# 0.3 share) against the live rho_w_sampled, on the 3x16 at weights 0.1 and 100. Fourteen
# cells, each a fit of both networks and the 38-system certificate.
#
# PARITY. The probe runs in configuration mode: the identity, the polarization and model
# block (DFS coordinates, no anchor), the pretrain block and the data directory are read
# from the run's own YAML through the harness's resolution (cluster/_pretrain.py and
# cluster/_datagen.py share the code), the data file is the one the run's datagen wrote
# (this script refuses to start without it, and the probe refuses a file at another
# identity rather than rebuilding it), the thread environment is the one the run's own
# pretrain task was rendered with (24 CPUs, the SCF pools capped at 8), and the certificate
# is the run's. The one departure is the step count, 10000 against 20000: the schedule
# fractions scale with it and the early stop (300 validations at every 50 steps) cannot
# fire inside 10000, so a cell is the same protocol at half length. If the probe's
# configuration mode is absent from the checkout, both legs exit 2 with a usage line and
# the epilogue reports REFUSED, rc 1: nothing runs.
#
# COST. The failed task's log (array 2145894, read off the job log and not reproducible
# from this repository) puts a 20000-step fit of both networks at 2 h 20 to 2 h 40 on 24
# threads and the certificate at 8 min. At 10000 steps a fit is 70 to 80 min, a cell with
# its certificate 1 h 18 to 1 h 28, fourteen cells 18.2 to 20.5 h. The 48 h request is 2.3x
# the upper end, carried because the three wider networks are unmeasured against the 3x16
# calibration. The cells run 3x16 first and widen; each table is rewritten after every
# cell, and a resubmission with the same MGGA_LEVERS_OUT resumes at the next cell. Leg 2 is
# skipped rather than started when less than one cell's wall remains.
#
# EXIT. Each leg exits 0 when a cell PASSED the certificate, 2 when the leg completed and
# none did (a finding; the table is written), 1 on a failed cell, 3 on an escaped
# exception. Exit 2 is claimed as that finding ONLY when the leg's own table is on disk and
# its captured output carries no usage line: argparse exits 2 with a usage line, and a
# missing or unreachable probe exits 2 with neither, so both are reported as refusals. The
# job exits 0 when either leg found a passing cell, 2 when both completed without one, and
# 1 or 3 for a refusal, a failed cell or an escaped exception. SLURM mails FAIL on any
# non-zero code: read the epilogue before treating a 2 as a broken job.
#
# Shell discipline: no `set -e` (conda activation returns non-zero on success); activation
# is verified by EFFECT, on the resolved interpreter path. The `:-` fallbacks are
# deliberate: an empty MGGA_LEVERS_* is not a cell list, so it reads as the study's default.
set -uo pipefail

ENV_PREFIX=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j070
REPO="${XCQ_REPO:-/gpfs/projects/FernandezGroup/Alec/xcquinox}"
RUN_ROOT="${MGGA_LEVERS_ROOT:-/gpfs/scratch/awills/mgga_levers}"
CFG="${MGGA_LEVERS_CONFIG:-hpcjobs/configs/dfs_step7.dfs6311_grid3_v7g2_families_mgga.yaml}"
OUT="${MGGA_LEVERS_OUT:-${RUN_ROOT}/mgga_levers_table.json}"
OUT_ARM="${OUT%.json}_integration.json"
STEPS="${MGGA_LEVERS_STEPS:-10000}"
ARCHS="${MGGA_LEVERS_ARCHS:-deep_mgga_3x16,deep_mgga_3x32,deep_mgga_4x16,deep_mgga_4x32}"
WEIGHTS="${MGGA_LEVERS_WEIGHTS:-0.1,10,100}"
ARM_ARCHS="${MGGA_LEVERS_ARM_ARCHS:-deep_mgga_3x16}"
ARM_WEIGHTS="${MGGA_LEVERS_ARM_WEIGHTS:-0.1,100}"
LOG="${RUN_ROOT}/mgga_levers_${SLURM_JOB_ID:-manual}.log"
RUN_OUT1="${RUN_ROOT}/mgga_levers_${SLURM_JOB_ID:-manual}.weights.last"
RUN_OUT2="${RUN_ROOT}/mgga_levers_${SLURM_JOB_ID:-manual}.integration.last"

mkdir -p "${RUN_ROOT}" || { echo "[mgga-levers] FATAL: cannot create ${RUN_ROOT}"; exit 1; }
T_START="$(date +%s)"
echo "[mgga-levers] START $(date '+%Y-%m-%d %H:%M:%S %Z') job=${SLURM_JOB_ID:-none} host=$(hostname)"
echo "[mgga-levers] repo=${REPO} config=${CFG} steps=${STEPS}"
echo "[mgga-levers] weights leg: archs=${ARCHS} weights=${WEIGHTS} out=${OUT}"
echo "[mgga-levers] integration leg: archs=${ARM_ARCHS} weights=${ARM_WEIGHTS} out=${OUT_ARM}"

# --- conda activation (verified by effect, not by return code) --------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[mgga-levers] host=$(hostname) python=$PYBIN"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[mgga-levers] FATAL: parity env python not active (got: ${PYBIN:-none})"
     exit 1 ;;
esac

cd "$REPO" || { echo "[mgga-levers] FATAL: cannot cd $REPO"; exit 1; }
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
# The configuration is judged BEFORE the import: a mistyped path should not cost a full
# numeric-stack start-up first.
[ -f "$CFG" ] || { echo "[mgga-levers] FATAL: no configuration at ${REPO}/${CFG}"; exit 1; }
python -c "import xcquinox.alec.pretrain" \
    || { echo "[mgga-levers] FATAL: repo import failed (env/activation)"; exit 1; }

# --- the run's own pretraining file, or nothing runs ------------------------------
# pretrain.data_dir of the configuration (its one data_dir line) and the polarized SCAN
# file the meta-GGA cells open (pretrain_data_gen.pretrain_data_filename(True, "scan")).
# The study reads the file the run's datagen wrote; generating one here would put an
# unbudgeted production-identity datagen inside the wall and fit against a file the run
# never saw. An unparsed data_dir is refused as what it is -- the two-space indent of the
# shipped configurations is what this pattern reads -- rather than left to compose a path
# that begins at the filesystem root and fail as a missing file.
DATA_DIR="$(sed -n 's/^  data_dir: *//p' "$CFG" | head -1)"
[ -n "$DATA_DIR" ] || { echo "[mgga-levers] FATAL: no '  data_dir:' line parsed from ${CFG}"; exit 1; }
DATA_FILE="${DATA_DIR}/pretrain_data_polarized_scan.npz"
[ -f "$DATA_FILE" ] || { echo "[mgga-levers] FATAL: the run's pretraining file is absent: ${DATA_FILE}"; exit 1; }
echo "[mgga-levers] data=${DATA_FILE}"

# --- thread caps and JAX platform: the SCF pools at min(allocation, 8)
# (parallel.PYSCF_POOL_THREADS_MAX, the rule the ewsweep wrapper and nonempirical_pool.sbatch
# apply), XLA's pool sized to the allocation as the harness's pretrain template does (the
# template's XLA_FLAGS line, with a :-8 default because this script runs under set -u).
THREADS="${SLURM_CPUS_PER_TASK:-8}"
case "$THREADS" in ''|*[!0-9]*) THREADS=8 ;; esac
THREADS=$(( 10#$THREADS ))
[ "$THREADS" -ge 1 ] || THREADS=1
[ "$THREADS" -le 8 ] || THREADS=8
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-8} --xla_force_host_platform_device_count=1"
export MALLOC_ARENA_MAX=2 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
echo "[mgga-levers] allocation=${SLURM_CPUS_PER_TASK:-unset} -> OMP/MKL/OPENBLAS=${THREADS}  MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX}"
echo "[mgga-levers] JAX_PLATFORMS=${JAX_PLATFORMS} JAX_ENABLE_X64=${JAX_ENABLE_X64} XLA_FLAGS=${XLA_FLAGS}"

# --- leg 1: the weights over the four widths, under the run's objective ---------------
python -u hpcjobs/probe_pretrain_energy_weight.py \
    --config "$CFG" --archs "$ARCHS" --weights "$WEIGHTS" --n-steps "$STEPS" \
    --out "$OUT" --resume \
    2>&1 | tee -a "$LOG" | tee "$RUN_OUT1"
RC1="${PIPESTATUS[0]}"
echo "[mgga-levers] weights leg rc=${RC1} ($(( $(date +%s) - T_START )) s elapsed)"

# --- leg 2: the failed run's objective on the 3x16 ---------------------------------
# Started only when a cell's worth of wall remains (5400 s, the upper end of the estimate
# in the COST block). A leg killed mid-cell writes no row, never reaches the epilogue, and
# is mailed as a bare FAIL; skipping it leaves leg 1's tables and a stated reason, and the
# resubmission that resumes leg 1 picks the arm up. SLURM_JOB_END_TIME is absent outside a
# job, where the leg always runs.
CELL_WALL_S="${MGGA_LEVERS_CELL_WALL_S:-5400}"
RC2=0
SKIPPED2=""
LEFT_S=""
if [ -n "${SLURM_JOB_END_TIME:-}" ]; then
  case "$SLURM_JOB_END_TIME" in
    ''|*[!0-9]*) : ;;
    *) LEFT_S=$(( SLURM_JOB_END_TIME - $(date +%s) )) ;;
  esac
fi
if [ -n "$LEFT_S" ] && [ "$LEFT_S" -lt "$CELL_WALL_S" ]; then
  SKIPPED2="yes"
  echo "[mgga-levers] integration leg SKIPPED: ${LEFT_S} s of wall left, under one cell's ${CELL_WALL_S} s"
  echo "[mgga-levers] resubmit to run it: the weights leg resumes from ${OUT} and the arm starts fresh"
else
  python -u hpcjobs/probe_pretrain_energy_weight.py \
      --config "$CFG" --objective-arm integration \
      --archs "$ARM_ARCHS" --weights "$ARM_WEIGHTS" --n-steps "$STEPS" \
      --out "$OUT_ARM" --resume \
      2>&1 | tee -a "$LOG" | tee "$RUN_OUT2"
  RC2="${PIPESTATUS[0]}"
  echo "[mgga-levers] integration leg rc=${RC2} ($(( $(date +%s) - T_START )) s elapsed)"
fi

# --- epilogue: each leg's code read beside its own output AND its own table -----------
# Prints the leg's outcome and returns the code the job carries for it: 0 a passing cell,
# 2 a completed leg without one, 1 a refusal, a lost output, a missing table or a failed
# cell, 3 an escaped exception. Exit 2 is claimed as the study's finding only on POSITIVE
# evidence that the sweep ran: its table on disk and no usage line in its capture. Three
# different things exit 2 -- the sweep's own verdict, argparse's refusal, and an
# interpreter that could not open the script at all (a renamed probe, a wrong XCQ_REPO) --
# and only the first is a result.
leg_code() {
  local rc="$1" run_out="$2" table="$3" name="$4"
  case "$rc" in
    0) echo "[mgga-levers] ${name}: a cell PASSED the fidelity certificate"; return 0 ;;
    2) if [ ! -f "$run_out" ]; then
         echo "[mgga-levers] ${name}: rc 2 with NO captured output (${run_out}); cannot tell a verdict from a refusal"
         return 1
       fi
       if grep -q "^usage: probe_pretrain_energy_weight" "$run_out"; then
         echo "[mgga-levers] ${name}: REFUSED -- argparse rejected the command line; no cell ran"
         return 1
       fi
       if [ ! -f "$table" ]; then
         echo "[mgga-levers] ${name}: rc 2 with NO table at ${table}; the sweep did not run (a missing or unreachable probe exits 2 too) -- NOT a finding"
         return 1
       fi
       echo "[mgga-levers] ${name}: COMPLETED, no cell passed the certificate (the finding; table written)"
       return 2 ;;
    3) echo "[mgga-levers] ${name}: an exception ESCAPED the sweep -- traceback above; partial table written"
       return 3 ;;
    *) echo "[mgga-levers] ${name}: FAILED (rc=${rc}) -- see the log above"; return 1 ;;
  esac
}
leg_code "$RC1" "$RUN_OUT1" "$OUT" "weights leg"; C1=$?
if [ -n "$SKIPPED2" ]; then
  echo "[mgga-levers] integration leg: NOT MEASURED (skipped for wall); the arm is open, resubmit to run it"
  C2=2
else
  leg_code "$RC2" "$RUN_OUT2" "$OUT_ARM" "integration leg"; C2=$?
fi
if   [ "$C1" -eq 1 ] || [ "$C1" -eq 3 ]; then RC="$C1"
elif [ "$C2" -eq 1 ] || [ "$C2" -eq 3 ]; then RC="$C2"
elif [ "$C1" -eq 0 ] || [ "$C2" -eq 0 ]; then RC=0
else RC=2
fi
for table in "$OUT" "$OUT_ARM"; do
  if [ -f "$table" ]; then echo "[mgga-levers] table: ${table}"
  else echo "[mgga-levers] NO TABLE at ${table} -- not one cell of that leg finished"; fi
done
# The literal path, not "$0": under SLURM the script runs from the spool copy, whose path
# is not the one to resubmit.
echo "[mgga-levers] resume: sbatch hpcjobs/probe_mgga_levers.sbatch   (continues from the cells in both tables)"
echo "[mgga-levers] END $(date '+%Y-%m-%d %H:%M:%S %Z') rc=${RC} elapsed $(( $(date +%s) - T_START )) s"
exit "$RC"
