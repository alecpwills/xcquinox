#!/usr/bin/env bash
#SBATCH --job-name=xcq_wfmatrix
# long-40core: the matrix is ONE process on ONE node (it drives every stage as
# a subprocess of itself; nothing here queues anything) and needs a wall above
# the 4 h cap of the short-* queues and the 12 h cap of the medium-* queues.
# Override to match the login instance you submit from -- the *-96core-shared
# queues exist only on milan1/milan2 and the *-28core queues only on
# login1/login2:
#   sbatch --partition=long-96core-shared --cpus-per-task=96 ...
#   sbatch --partition=long-28core        --cpus-per-task=28 ...
#SBATCH --partition=long-40core
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/gpfs/scratch/awills/workflow_matrix_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Per-architecture workflow matrix (SPEC_pretrain_fidelity_program.md 3.4).
#
# WHAT: xcquinox/alec/cluster/workflow_matrix.py drives every registered
# architecture through the harness stage sequence -- submit (dry-run) ->
# datagen -> pretrain -> certificate -> preflight -> train[0] -> train[1] ->
# eval[0] -> eval[1] -> validate_run, then that architecture's slice of the
# spin-scaling oracles -- at the tiny identity of
# cluster/examples/workflow_matrix_template.yaml (def2-svp, grid level 1,
# solver oneshot, two cells at subset sizes 1 and 2, 3 training steps, 50
# pretraining steps on H and O, held-out eval on the six-species slice). It
# writes one row per architecture into a markdown report with a JSON sidecar.
# Nothing it produces is a physics result.
#
# WHY A JOB: 31 architectures x 11 stage subprocesses, every one of them an SCF
# at def2-svp. It is not a workstation task. The matrix itself submits nothing
# and mails nothing -- it runs the stage modules in-process on this one node --
# so this script is the only thing SLURM sees.
#
# EXIT CODES (SLURM mails FAIL on any non-zero code; read the verdict line in
# the log before treating one as a crash):
#   0  every architecture completed every stage with its expected outcome,
#      wrote a certificate, marked its held-out channel as sliced and passed
#      its oracles.
#   1  the matrix RAN TO COMPLETION and at least one architecture did not.
#      The report is written and the finding is the report's Findings block.
#      Not a crash.
#   3  the job could not run the matrix at all (environment, repository or
#      cached inputs). Nothing was measured.
#
# WALL DERIVATION (the 24 h request). Per architecture the matrix runs 11
# timed subprocesses -- the ten stages of workflow_matrix.STAGE_ORDER plus the
# oracle selection -- and kills any one of them at --timeout-s. The request is
# that cap taken as an UPPER BOUND, not an estimate:
#
#     per architecture  11 stages x 1800 s               = 19800 s = 5.5 h
#     per shard         ceil(16 archs / 4 shards) = 4    -> 4 x 5.5 h = 22.0 h
#     staging (74 MB reference copy) + interpreter starts        ~  0.2 h
#     request                                                      24 h
#
# 1800 s is the per-stage wall the matrix's own template renders for these same
# stages (cluster.time / preflight_time / eval_time / pretrain_time =
# "00:30:00"). It is a hang detector, not a schedule: the expectation for the
# WHOLE per-architecture sequence at this identity is 10-15 min, so a single
# stage past 30 min is stuck rather than slow.
#
# 16 architectures per submission, not 31: the registry holds 31, and 31 in one
# job bounds at ceil(31/4) x 5.5 h = 44 h -- inside the 2-day cap of
# long-28core, but with no margin and beyond every shorter queue. MATRIX_BATCHES
# therefore defaults to 2, so the registry is covered by two submissions (16
# and 15 architectures), each bounded by this request. MATRIX_BATCHES=1 puts
# all 31 in one job: raise the request (sbatch --time=46:00:00) and submit to a
# queue that allows it. A single-architecture run (MATRIX_ARCHS=<name>) bounds
# at 11 x 1800 s = 5.5 h, so submit it with --time=06:00:00 rather than paying
# the 24 h request in queue time.
#
# ENVIRONMENT KNOBS (all optional; every one is echoed into the log):
#   MATRIX_ARCHS      'all' (default) or a comma-separated list. A list is
#                     taken verbatim and TURNS BATCHING OFF. An empty value is
#                     refused rather than read as 'all'.
#   MATRIX_BATCHES    number of submissions the registry is split across
#                     (default 2). Applies only when MATRIX_ARCHS is 'all'.
#   MATRIX_BATCH      which batch this job runs, 0-based (default 0).
#   MATRIX_SHARDS     architectures run concurrently, 1..4 (default 4). The
#                     matrix divides the node's cores by this number to set
#                     each stage's BLAS thread cap.
#   MATRIX_TIMEOUT_S  per-stage wall-clock cap in seconds (default 1800; the
#                     wall above is derived from it).
#   MATRIX_NO_ORACLES 1 skips the oracle stage (the bound is then 10 stages).
#   MATRIX_WORK_ROOT  work root (default /gpfs/scratch/awills/workflow_matrix/
#                     <UTC stamp>_<job id>). MUST be outside the repository
#                     (refused here, as `main` refuses it) and MUST NOT be
#                     shared with a concurrent job: the reference copy is
#                     staged into <work root>/_inputs, and two jobs staging
#                     into one directory would remove each other's copy.
#   MATRIX_EXTERNAL_REFS  an already staged copy of the CCSD reference cache,
#                     used instead of copying the repository's. The stages
#                     WRITE run logs into whatever directory this names, so it
#                     must not be a directory another run reads.
#   XCQ_REPO, XCQ_ENV_PREFIX, XCQ_CONDA_PROFILE  repository and environment.
#
# Shell discipline: no `set -e` (conda activation returns non-zero on success
# and would kill the job seconds in with an empty log); activation is verified
# by EFFECT, on the resolved interpreter path.
set -uo pipefail

ENV_PREFIX="${XCQ_ENV_PREFIX:-/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j070}"
CONDA_PROFILE="${XCQ_CONDA_PROFILE:-/gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh}"
REPO="${XCQ_REPO:-/gpfs/projects/FernandezGroup/Alec/xcquinox}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-manual}"
WORK_ROOT="${MATRIX_WORK_ROOT:-/gpfs/scratch/awills/workflow_matrix/${STAMP}}"
REPORT="${WORK_ROOT}/workflow_matrix.md"
LOG="${WORK_ROOT}/workflow_matrix.log"
# `-` and not `:-`: an EXPLICITLY empty MATRIX_ARCHS (an unset variable
# expanded into the submission line, typically) is refused below rather than
# read as the whole registry.
ARCHS="${MATRIX_ARCHS-all}"
SHARDS="${MATRIX_SHARDS:-4}"
TIMEOUT_S="${MATRIX_TIMEOUT_S:-1800}"
BATCHES="${MATRIX_BATCHES:-2}"
BATCH="${MATRIX_BATCH:-0}"
NO_ORACLES="${MATRIX_NO_ORACLES:-0}"
EXTERNAL_REFS="${MATRIX_EXTERNAL_REFS:-}"

# The two cached inputs of the tiny identity, at the relative paths
# workflow_matrix.CACHED_REFS_RELPATH / CACHED_LEDGER_RELPATH name. Both are
# UNTRACKED (.gitignore excludes notebooks/checkpoints_step7/), so a fresh
# rsync of the repository does not carry them.
LEDGER="${REPO}/notebooks/checkpoints_step7/alpha_on/subset_index_log.json"
REPO_REFS="${REPO}/notebooks/checkpoints_step7/external_refs"

fatal() {
    echo "[matrix] FATAL: $*"
    echo "[matrix] nothing was measured; exit 3"
    exit 3
}

is_uint() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

# The criterion workflow_matrix.staged_refs_dir applies to a SUPPLIED
# reference directory: its completion manifest (STAGE_MARKER), or the
# per-species .npz files the staging copies. An empty or wrong directory
# passes every path check and surfaces at the first reference read, deep
# inside a stage.
refs_carry_references() {
    [ -f "$1/_stage_complete" ] && return 0
    for npz in "$1"/*.npz; do
        [ -e "$npz" ] && return 0
    done
    return 1
}

mkdir -p "$WORK_ROOT" || fatal "cannot create work root ${WORK_ROOT}"

T_START="$(date +%s)"
echo "[matrix] ==================================================="
echo "[matrix] START  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[matrix] job=${SLURM_JOB_ID:-none}  host=$(hostname)"
echo "[matrix] repo=${REPO}"
echo "[matrix] work root=${WORK_ROOT}"
echo "[matrix] report=${REPORT}"
echo "[matrix] log=${LOG}"
echo "[matrix] archs=${ARCHS} batches=${BATCHES} batch=${BATCH}"
echo "[matrix] shards=${SHARDS} timeout_s=${TIMEOUT_S} no_oracles=${NO_ORACLES}"
echo "[matrix] external refs=${EXTERNAL_REFS:-<stage from the repository>}"
echo "[matrix] ==================================================="

# --- knob validation (a bad knob must not surface as a python traceback, ----
# --- which would exit 1 and read as a matrix finding) -----------------------
# 1..4 is workflow_matrix.MAX_SHARDS: each shard runs one SCF-heavy stage at a
# time, and the matrix divides the node's cores between them.
is_uint "$SHARDS" && [ "$SHARDS" -ge 1 ] && [ "$SHARDS" -le 4 ] \
    || fatal "MATRIX_SHARDS must be an integer in 1..4 (got '${SHARDS}')"
is_uint "$TIMEOUT_S" && [ "$TIMEOUT_S" -ge 1 ] \
    || fatal "MATRIX_TIMEOUT_S must be a positive integer (got '${TIMEOUT_S}')"
is_uint "$BATCHES" && [ "$BATCHES" -ge 1 ] \
    || fatal "MATRIX_BATCHES must be a positive integer (got '${BATCHES}')"
is_uint "$BATCH" \
    || fatal "MATRIX_BATCH must be a non-negative integer (got '${BATCH}')"
[ "$BATCH" -lt "$BATCHES" ] \
    || fatal "MATRIX_BATCH=${BATCH} is outside MATRIX_BATCHES=${BATCHES} (0-based)"
[ -n "$ARCHS" ] || fatal "MATRIX_ARCHS is empty; unset it for the whole registry"

# --- conda activation (verified by effect, not by return code) --------------
if [ -r "$CONDA_PROFILE" ]; then
    # shellcheck source=/dev/null
    . "$CONDA_PROFILE"
else
    echo "[matrix] WARN: conda profile not readable at ${CONDA_PROFILE}"
fi
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[matrix] python=${PYBIN:-none}"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) fatal "env python not active (got: ${PYBIN:-none})" ;;
esac

cd "$REPO" || fatal "cannot cd ${REPO}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

# `main` refuses a work root inside the repository -- the stages write run
# directories, a copy of the reference cache and pretrain data, and the
# reference copy receives a run log from every preflight. Refused here so it
# reads as a job that could not start rather than as a matrix that found a
# defect.
REPO_ABS="$(pwd -P)"
WORK_ABS="$(cd "$WORK_ROOT" && pwd -P)" || fatal "cannot enter ${WORK_ROOT}"
case "$WORK_ABS" in
  "$REPO_ABS"|"$REPO_ABS"/*)
    fatal "work root ${WORK_ABS} is inside the repository ${REPO_ABS}" ;;
esac

python -c "import xcquinox.alec.cluster.workflow_matrix" \
    || fatal "repo import failed (env/activation)"

# --- thread caps and JAX platform (SLURM vars not assumed present) ----------
# These are this process's caps. Each stage subprocess gets its own from
# workflow_matrix._base_env, at (node cores / shards) threads.
THREADS="${SLURM_CPUS_PER_TASK:-40}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MALLOC_ARENA_MAX=2
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
echo "[matrix] BLAS threads=${THREADS}  nproc=$(nproc 2>/dev/null || echo '?')"
echo "[matrix] JAX_PLATFORMS=${JAX_PLATFORMS} JAX_ENABLE_X64=${JAX_ENABLE_X64}"

# --- the untracked cached inputs -------------------------------------------
# Checked here so a missing input is one line in this log rather than a
# CachedInputsMissing traceback after the node has been allocated.
if [ ! -f "$LEDGER" ]; then
    echo "[matrix] stage it from the workstation with:"
    echo "[matrix]   rsync -av notebooks/checkpoints_step7/alpha_on/subset_index_log.json \\"
    echo "[matrix]         \"\$swpath\":${REPO}/notebooks/checkpoints_step7/alpha_on/"
    fatal "cached subset ledger not found at ${LEDGER} (untracked)"
fi
if [ -n "$EXTERNAL_REFS" ]; then
    [ -d "$EXTERNAL_REFS" ] \
        || fatal "MATRIX_EXTERNAL_REFS=${EXTERNAL_REFS} is not a directory"
    refs_carry_references "$EXTERNAL_REFS" \
        || fatal "MATRIX_EXTERNAL_REFS=${EXTERNAL_REFS} holds neither a staging manifest nor any species .npz, so it carries no references"
else
    if [ ! -d "$REPO_REFS" ] || ! refs_carry_references "$REPO_REFS"; then
        echo "[matrix] stage it from the workstation with:"
        echo "[matrix]   rsync -av notebooks/checkpoints_step7/external_refs/ \\"
        echo "[matrix]         \"\$swpath\":${REPO}/notebooks/checkpoints_step7/external_refs/"
        echo "[matrix] or point MATRIX_EXTERNAL_REFS at a copy already on scratch."
        fatal "cached CCSD references not found at ${REPO_REFS} (untracked)"
    fi
fi

# --- architecture selection ------------------------------------------------
# A named list is taken verbatim: the batch split exists only to keep the
# whole-registry pass inside one queue's wall, and a caller who named the
# architectures has already chosen the work.
if [ "$ARCHS" = "all" ] && [ "$BATCHES" -gt 1 ]; then
    # `|| [ -n "$name" ]` keeps a final line that carries no newline: dropping
    # it would leave one architecture out of the batch and out of the matrix
    # without saying so.
    ALL_ARCHS=()
    while IFS= read -r name || [ -n "$name" ]; do
        [ -n "$name" ] && ALL_ARCHS+=("$name")
    done < <(python -m xcquinox.alec.cluster.workflow_matrix --list)
    N_ARCHS="${#ALL_ARCHS[@]}"
    [ "$N_ARCHS" -gt 0 ] || fatal "--list returned no architecture"
    PER=$(( (N_ARCHS + BATCHES - 1) / BATCHES ))
    START=$(( BATCH * PER ))
    if [ "$START" -ge "$N_ARCHS" ]; then
        fatal "batch ${BATCH} of ${BATCHES} starts at ${START}, past the ${N_ARCHS} registered architectures"
    fi
    END=$(( START + PER ))
    if [ "$END" -gt "$N_ARCHS" ]; then
        END="$N_ARCHS"
    fi
    SELECTED=""
    for (( i = START; i < END; i++ )); do
        SELECTED="${SELECTED}${SELECTED:+,}${ALL_ARCHS[$i]}"
    done
    ARCHS="$SELECTED"
    echo "[matrix] batch ${BATCH} of ${BATCHES}: architectures ${START}..$((END - 1)) of ${N_ARCHS}"
elif [ "$ARCHS" != "all" ]; then
    echo "[matrix] MATRIX_ARCHS names the architectures; batching not applied."
fi
echo "[matrix] running: ${ARCHS}"

EXTRA_ARGS=()
if [ -n "$EXTERNAL_REFS" ]; then
    EXTRA_ARGS+=(--external-refs-dir "$EXTERNAL_REFS")
fi
if [ "$NO_ORACLES" = "1" ]; then
    EXTRA_ARGS+=(--no-oracles)
fi

# -u: unbuffered, so a job killed by signal still leaves its last real progress
# line in the log rather than discarding a buffered block. The exit code read
# is python's (PIPESTATUS[0]), never tee's.
python -u -m xcquinox.alec.cluster.workflow_matrix \
    --archs      "$ARCHS" \
    --work-root  "$WORK_ROOT" \
    --report     "$REPORT" \
    --shards     "$SHARDS" \
    --timeout-s  "$TIMEOUT_S" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    2>&1 | tee "$LOG"
RC="${PIPESTATUS[0]}"

T_END="$(date +%s)"
echo "[matrix] ==================================================="
echo "[matrix] rc=${RC}  elapsed $((T_END - T_START)) s"
if [ -f "$REPORT" ]; then
  echo "[matrix] report:  ${REPORT}"
  echo "[matrix] sidecar: ${REPORT%.md}.json"
  echo "[matrix] per-stage logs: ${WORK_ROOT}/<arch>/logs/<stage>.log"
else
  echo "[matrix] NO REPORT WRITTEN -- the matrix did not reach its end."
fi
case "$RC" in
  0) echo "[matrix] every architecture ran every stage with its expected outcome." ;;
  1) echo "[matrix] COMPLETED; at least one architecture did not."
     echo "[matrix] That is the finding; the report's Findings block names the"
     echo "[matrix] stage, its exit code and its log. Not a crash." ;;
  *) echo "[matrix] FAILED -- see the log above." ;;
esac
echo "[matrix] END  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[matrix] ==================================================="
exit "$RC"
