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
# WHAT THIS SCRIPT NEEDS ON THE CLUSTER: the WHOLE xcquinox package and the
# whole hpcjobs directory, not these two files. The matrix drives the stage
# modules of the same tree (cluster/_datagen, _pretrain, fidelity, _preflight,
# _train_task, _eval_one_spec, validate_run, the rendered template) and the
# spin-scaling oracle module; a checkout carrying an older copy of any of them
# accepts the same command line and then drives contracts that have moved. The
# push is therefore one line from the local repository root:
#
#   rsync -av xcquinox hpcjobs \
#         "$swpath":/gpfs/projects/FernandezGroup/Alec/xcquinox/
#
# and the checkout's own contract is read back below (MATRIX_CONTRACT), so a
# partial sync exits 3 at second zero instead of driving moved stages.
#
# EXIT CODES (SLURM mails FAIL on any non-zero code; read the verdict line in
# the log before treating one as a crash):
#   0  every architecture completed every stage with its expected outcome,
#      wrote a certificate, marked its held-out channel as sliced and passed
#      its oracles.
#   1  the matrix RAN TO COMPLETION and at least one architecture did not.
#      The report is written and the finding is the report's Findings block.
#      Not a crash.
#   3  the job could not run the matrix at all (environment, repository,
#      cached inputs, a knob the matrix would have refused, a wall request
#      below the derived bound, or a command line the matrix refused with its
#      own usage code 2). Nothing was measured.
#
# WALL DERIVATION. The request is derived from the bound, not the bound from
# the request. Per architecture the matrix runs 11 timed subprocesses -- the
# ten stages of workflow_matrix.STAGE_ORDER plus the oracle selection -- and
# kills any one of them at --timeout-s, so the cap taken as an UPPER BOUND is:
#
#     per architecture  11 stages x 1800 s               = 19800 s = 5.5 h
#     per shard         ceil(16 archs / 4 shards) = 4    -> 4 x 5.5 h = 22.0 h
#     bound                                                        22.0 h
#     margin (74 MB reference copy, interpreter starts, slack)  +   2.0 h
#     request           bound + margin                              24 h
#
# The same arithmetic is recomputed AT RUN TIME below from the stage list the
# installed module reports and from the knobs this job was given, and compared
# with the wall actually requested (SLURM_JOB_END_TIME when SLURM set it,
# otherwise the #SBATCH --time directive above): a knob that outruns the
# request is refused with both numbers rather than killed at the wall, where
# the whole pass would be lost -- main writes the report only after the last
# architecture returns.
#
# PARTITION CAP. hpcjobs/SEAWULF_RUNBOOK.md records short-* 4 h, medium-* 12 h,
# long-28core 2 days and long-96core-shared longer still; it states NO cap for
# long-40core. Verify it before the first submission --
#   sinfo -o "%20P %10l" -p long-40core
# -- and if it caps below 24 h, submit to long-28core (--cpus-per-task=28) or
# long-96core-shared (--cpus-per-task=96) from the matching login instance. An
# over-cap request is rejected by sbatch at submission, not after a queue wait.
#
# THE PER-STAGE CAP (1800 s) is half the module's own hang detector,
# DEFAULT_STAGE_TIMEOUT_S = 3600, and equals the per-stage wall the matrix's
# own template renders for these same stages (cluster.time / preflight_time /
# eval_time / pretrain_time = "00:30:00"). It is a hang detector, not a
# schedule: the expectation for the WHOLE per-architecture sequence at this
# identity is 10-15 min. Measured against the cached references' own run logs
# (notebooks/checkpoints_step7/external_refs/_run_log_*.json), the eight
# species the two cells need build in 160.8 s in total at the LARGER step-7
# basis (H 4.0, H2 8.1, O 11.1, Li 12.3, HLi 7.3, H2O 22.4, N2 43.8, N2O
# 51.8), an order of magnitude inside the cap. The tail is what the two caps
# disagree about: the same logs record a FAILED N2O attempt at 2558 s (and
# F2O at 11623 s, not in this pool), which would be killed at 1800 s and
# survive at 3600 s. That is the intended behaviour -- a non-converging OEP
# cascade is the hang this cap detects -- and the signal is rc 124 on
# `preflight` for the first architecture of a work root. If a preflight
# genuinely needs longer, raise MATRIX_TIMEOUT_S and the --time request
# together; the bound below scales with both.
#
# 16 architectures per submission, not 31: the registry holds 31, and 31 in one
# job bounds at ceil(31/4) x 5.5 h = 44 h -- inside the 2-day cap of
# long-28core, but with no margin and beyond every shorter queue. MATRIX_BATCHES
# therefore defaults to 2, so the registry is covered by two submissions (16
# and 15 architectures), each bounded by this request. MATRIX_BATCHES=1 puts
# all 31 in one job and is REFUSED at this request: raise it
# (sbatch --time=46:00:00) on a queue that allows it. A single-architecture run
# (MATRIX_ARCHS=<name>) bounds at 11 x 1800 s = 5.5 h and takes the whole
# allocation (shards drop to 1), so submit it with --time=06:00:00 rather than
# paying the 24 h request in queue time.
#
# ENVIRONMENT KNOBS (all optional; every one is echoed into the log):
#   MATRIX_ARCHS      'all' (default) or a comma-separated list. A list is
#                     checked NAME BY NAME against the installed registry and
#                     TURNS BATCHING OFF. An empty value is refused rather
#                     than read as 'all'.
#   MATRIX_BATCHES    number of submissions the registry is split across
#                     (default 2). Applies only when MATRIX_ARCHS is 'all'.
#   MATRIX_BATCH      which batch this job runs, 0-based (default 0).
#   MATRIX_SHARDS     architectures run concurrently, 1..4 (default 4, or 1
#                     when the selection is a single architecture). The matrix
#                     divides the ALLOCATION (SLURM_CPUS_PER_TASK) by this
#                     number to set each stage's BLAS thread cap.
#   MATRIX_TIMEOUT_S  per-stage wall-clock cap in seconds (default 1800; the
#                     wall bound is derived from it).
#   MATRIX_NO_ORACLES 1 skips the oracle stage (the bound is then 10 stages).
#   MATRIX_WORK_ROOT  work root (default /gpfs/scratch/awills/workflow_matrix/
#                     <UTC stamp>_<job id>). MUST be outside the repository
#                     (refused here, as `main` refuses it) and MUST NOT be
#                     shared with a concurrent job: the reference copy is
#                     staged into <work root>/_inputs, and two jobs staging
#                     into one directory would remove each other's copy.
#   MATRIX_EXTERNAL_REFS  an already staged copy of the CCSD reference cache,
#                     used instead of copying the repository's. Left unset --
#                     the default -- every job stages its OWN copy under its
#                     own work root, which is what keeps two jobs off one
#                     directory. Set it only for a copy this job alone reads:
#                     every preflight writes a run log into whatever directory
#                     it names, and it is refused inside the repository.
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
SCRIPT_PATH="${BASH_SOURCE[0]}"

# The two cached inputs of the tiny identity, at the relative paths
# workflow_matrix.CACHED_REFS_RELPATH / CACHED_LEDGER_RELPATH name. Both are
# UNTRACKED (.gitignore excludes notebooks/checkpoints_step7/), so a fresh
# rsync of the repository does not carry them.
LEDGER="${REPO}/notebooks/checkpoints_step7/alpha_on/subset_index_log.json"
REPO_REFS="${REPO}/notebooks/checkpoints_step7/external_refs"

# The contract this job drives, read back from the checkout that will run it.
# The stage list is what the wall bound counts, MAX_SHARDS is what the knob
# validation admits, and timeout_bound=1 is the parser refusing a non-positive
# --timeout-s with its usage code rather than reporting every stage as killed.
# A checkout that answers anything else is not the tree this script was
# written against.
CONTRACT_PROBE='
import contextlib, io, sys
from xcquinox.alec.cluster import workflow_matrix as wm
code = None
try:
    with contextlib.redirect_stderr(io.StringIO()):
        wm.main(["--work-root", "/nonexistent-contract-probe",
                 "--timeout-s", "0"])
except SystemExit as exc:
    code = exc.code
sys.stdout.write("MATRIX_CONTRACT stages=%s shards=%d timeout_bound=%d\n"
                 % (",".join(wm.STAGE_ORDER), wm.MAX_SHARDS, int(code == 2)))
'
MATRIX_CONTRACT_EXPECTED="MATRIX_CONTRACT stages=submit,datagen,pretrain,certificate,preflight,train[0],train[1],eval[0],eval[1],validate_run shards=4 timeout_bound=1"

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

# Whole hours with one decimal, from integer arithmetic only.
fmt_hours() {
    printf '%d.%d' "$(( $1 / 3600 ))" "$(( ($1 % 3600) * 10 / 3600 ))"
}

# A SLURM walltime (H:MM:SS or D-HH:MM:SS) in seconds. 10# forces base ten:
# an hour field of 08 is not octal here.
walltime_seconds() {
    local spec="$1" days=0 rest h m s
    rest="$spec"
    case "$spec" in
        *-*) days="${spec%%-*}"; rest="${spec#*-}" ;;
    esac
    case "$rest" in
        *:*:*) h="${rest%%:*}"; m="${rest#*:}"; m="${m%%:*}"; s="${rest##*:}" ;;
        *) return 1 ;;
    esac
    is_uint "$days" && is_uint "$h" && is_uint "$m" && is_uint "$s" || return 1
    printf '%d' "$(( 10#$days * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))"
}

# workflow_matrix._stage_is_complete, in shell: the staging manifest is
# complete when it records at least one file and every file it records is
# still there. The manifest's mere presence is NOT the criterion -- an
# interrupted rsync of a warmed copy leaves exactly that -- and
# workflow_matrix.staged_refs_dir refuses it, deep inside a stage, where this
# script refuses it in one line.
# test_the_reference_precheck_is_the_matrix_predicate pins the two against
# each other, shape by shape.
stage_manifest_is_complete() {
    local root="$1" first=1 rel found=0
    [ -f "${root}/_stage_complete" ] || return 1
    while IFS= read -r rel || [ -n "$rel" ]; do
        if [ "$first" = 1 ]; then first=0; continue; fi
        [ -n "$rel" ] || continue
        [ -f "${root}/${rel}" ] || return 1
        found=1
    done < "${root}/_stage_complete"
    [ "$found" = 1 ]
}

# The criterion workflow_matrix.staged_refs_dir applies to a SUPPLIED
# reference directory: a COMPLETE staging manifest, or the per-species .npz
# entries the staging copies. An empty or wrong directory passes every path
# check and surfaces at the first reference read, deep inside a stage.
refs_carry_references() {
    stage_manifest_is_complete "$1" && return 0
    for npz in "$1"/*.npz; do
        [ -e "$npz" ] && return 0
    done
    return 1
}

# The installed registry, one name per line. `|| [ -n "$name" ]` keeps a final
# line that carries no newline: dropping it would leave one architecture out
# of the batch and out of the matrix without saying so.
load_registry() {
    ALL_ARCHS=()
    while IFS= read -r name || [ -n "$name" ]; do
        [ -n "$name" ] && ALL_ARCHS+=("$name")
    done < <(python -m xcquinox.alec.cluster.workflow_matrix --list)
    N_ARCHS="${#ALL_ARCHS[@]}"
    [ "$N_ARCHS" -gt 0 ] || fatal "--list returned no architecture"
}

registry_has() {
    local candidate="$1" name
    for name in "${ALL_ARCHS[@]}"; do
        [ "$name" = "$candidate" ] && return 0
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
# time, and the matrix divides the allocation between them.
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

# --- the checkout's own contract (import probe and partial-sync guard) ------
MATRIX_CONTRACT="$(python -c "$CONTRACT_PROBE")" \
    || fatal "repo import failed (env/activation); the traceback is above"
echo "[matrix] contract=${MATRIX_CONTRACT:-<none reported>}"
if [ "$MATRIX_CONTRACT" != "$MATRIX_CONTRACT_EXPECTED" ]; then
    echo "[matrix] expected: ${MATRIX_CONTRACT_EXPECTED}"
    echo "[matrix] checkout: ${MATRIX_CONTRACT:-<none reported>}"
    echo "[matrix] The matrix drives the stage modules of this same tree, so"
    echo "[matrix] the WHOLE package and hpcjobs are pushed together, from"
    echo "[matrix] the local repository root:"
    echo "[matrix]   rsync -av xcquinox hpcjobs \\"
    echo "[matrix]         \"\$swpath\":${REPO}/"
    fatal "this checkout's workflow_matrix contract is not the one this job drives (partial sync)"
fi

# --- thread caps and JAX platform (SLURM vars not assumed present) ----------
# These are this process's caps. Each stage subprocess gets its own from
# workflow_matrix._base_env, at (SLURM_CPUS_PER_TASK / shards) threads -- the
# allocation, not the machine, so the budget holds on a shared queue and on an
# SMT node.
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
    EXTERNAL_REFS_ABS="$(cd "$EXTERNAL_REFS" && pwd -P)" \
        || fatal "cannot enter MATRIX_EXTERNAL_REFS=${EXTERNAL_REFS}"
    case "$EXTERNAL_REFS_ABS" in
      "$REPO_ABS"|"$REPO_ABS"/*)
        fatal "MATRIX_EXTERNAL_REFS ${EXTERNAL_REFS_ABS} is inside the repository ${REPO_ABS}: every preflight writes a _run_log_<UTC>.json into the reference directory it reads, and nothing the matrix writes belongs in the tree" ;;
    esac
    refs_carry_references "$EXTERNAL_REFS_ABS" \
        || fatal "MATRIX_EXTERNAL_REFS=${EXTERNAL_REFS} holds neither a complete staging manifest nor any species .npz, so it carries no references"
    echo "[matrix] WARN: MATRIX_EXTERNAL_REFS is read AND written by this job --"
    echo "[matrix] every preflight drops a run log into it. It must be this"
    echo "[matrix] job's copy alone; give a second job its own (cp -a) rather"
    echo "[matrix] than this path. Unset, each job stages its own copy."
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
# A named list is checked NAME BY NAME against the installed registry: an
# unregistered name is the one knob the operator types by hand, and the matrix
# refuses it with its usage code after the queue wait has been paid.
if [ "$ARCHS" = "all" ]; then
    load_registry
    if [ "$BATCHES" -gt 1 ]; then
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
        N_SELECTED=$(( END - START ))
        echo "[matrix] batch ${BATCH} of ${BATCHES}: architectures ${START}..$((END - 1)) of ${N_ARCHS}"
    else
        N_SELECTED="$N_ARCHS"
        echo "[matrix] one batch: the whole registry (${N_ARCHS} architectures)"
    fi
else
    load_registry
    NAMED=()
    IFS=',' read -r -a RAW_NAMES <<< "$ARCHS"
    for name in ${RAW_NAMES[@]+"${RAW_NAMES[@]}"}; do
        [ -n "$name" ] || continue
        if ! registry_has "$name"; then
            echo "[matrix] the installed registry (${N_ARCHS} names) is:"
            echo "[matrix]   ${ALL_ARCHS[*]}"
            fatal "MATRIX_ARCHS names '${name}', which is not a registered architecture"
        fi
        for seen in ${NAMED[@]+"${NAMED[@]}"}; do
            if [ "$seen" = "$name" ]; then
                fatal "MATRIX_ARCHS names '${name}' more than once; two rows for one architecture share <work root>/${name} and would report on a single run"
            fi
        done
        NAMED+=("$name")
    done
    N_SELECTED="${#NAMED[@]}"
    [ "$N_SELECTED" -gt 0 ] \
        || fatal "MATRIX_ARCHS names no architecture; unset it for the whole registry"
    echo "[matrix] MATRIX_ARCHS names the architectures; batching not applied."
fi
echo "[matrix] running: ${ARCHS}"

# One architecture is the smoke, and it has the node to itself: the matrix
# divides the allocation by the shard count, so the default 4 would leave the
# single architecture a quarter of it on the very run that gates the matrix.
# An explicit MATRIX_SHARDS still wins.
if [ -z "${MATRIX_SHARDS:-}" ] && [ "$N_SELECTED" -eq 1 ]; then
    SHARDS=1
    echo "[matrix] one architecture selected: shards=1 (the whole allocation)"
fi

# --- the wall bound against the wall requested ------------------------------
# main writes the report only after the LAST architecture returns, so a job
# killed at its wall produces per-stage logs and no table. The bound is the
# per-stage cap taken as an upper bound over the stages the installed module
# reports, and it is refused here rather than at the wall.
STAGE_LIST="${MATRIX_CONTRACT#*stages=}"
STAGE_LIST="${STAGE_LIST%% *}"
IFS=',' read -r -a STAGE_NAMES <<< "$STAGE_LIST"
N_STAGES="${#STAGE_NAMES[@]}"
if [ "$NO_ORACLES" = "1" ]; then
    TIMED_PER_ARCH="$N_STAGES"
else
    TIMED_PER_ARCH=$(( N_STAGES + 1 ))
fi
PER_SHARD=$(( (N_SELECTED + SHARDS - 1) / SHARDS ))
BOUND_S=$(( PER_SHARD * TIMED_PER_ARCH * TIMEOUT_S ))
echo "[matrix] wall bound=${BOUND_S} s ($(fmt_hours "$BOUND_S") h) = ceil(${N_SELECTED} archs / ${SHARDS} shards) x ${TIMED_PER_ARCH} timed stages x ${TIMEOUT_S} s"

REQUEST_S=""
REQUEST_SRC=""
if [ -n "${SLURM_JOB_END_TIME:-}" ] && is_uint "${SLURM_JOB_END_TIME:-}"; then
    REQUEST_S=$(( SLURM_JOB_END_TIME - $(date +%s) ))
    REQUEST_SRC="SLURM_JOB_END_TIME (the wall left on this allocation)"
else
    TIME_LINE="$(grep -m1 '^#SBATCH --time=' "$SCRIPT_PATH" 2>/dev/null || true)"
    TIME_SPEC="${TIME_LINE#*=}"
    if [ -n "$TIME_SPEC" ]; then
        REQUEST_S="$(walltime_seconds "$TIME_SPEC")" || REQUEST_S=""
        REQUEST_SRC="#SBATCH --time=${TIME_SPEC} in ${SCRIPT_PATH}"
    fi
fi
if [ -n "$REQUEST_S" ] && [ "$REQUEST_S" -gt 0 ]; then
    echo "[matrix] wall request=${REQUEST_S} s ($(fmt_hours "$REQUEST_S") h) from ${REQUEST_SRC}"
    if [ "$BOUND_S" -gt "$REQUEST_S" ]; then
        echo "[matrix] raise the request with --time on the submission, or lower"
        echo "[matrix] MATRIX_BATCHES / MATRIX_SHARDS / MATRIX_TIMEOUT_S. A job"
        echo "[matrix] killed at its wall writes no report at all."
        fatal "the derived wall bound ${BOUND_S} s ($(fmt_hours "$BOUND_S") h) exceeds the wall request ${REQUEST_S} s ($(fmt_hours "$REQUEST_S") h)"
    fi
else
    echo "[matrix] WARN: no wall request could be read (${REQUEST_SRC:-no source found}); the bound above is unchecked."
fi

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

# The matrix's own usage code is 2: a flag it refused, which is a job that
# never ran rather than a matrix that found a defect. It is carried into this
# script's class 3 -- everything the preflight above checks is refused before
# the matrix starts, so a 2 reaching here is a command line this script built
# and the log's usage line says which flag.
MATRIX_RC="$RC"
if [ "$RC" = "2" ]; then
    RC=3
fi

T_END="$(date +%s)"
echo "[matrix] ==================================================="
echo "[matrix] rc=${RC}  (matrix rc=${MATRIX_RC})  elapsed $((T_END - T_START)) s"
if [ -f "$REPORT" ]; then
  echo "[matrix] report:  ${REPORT}"
  echo "[matrix] sidecar: ${REPORT%.md}.json"
  echo "[matrix] per-stage logs: ${WORK_ROOT}/<arch>/logs/<stage>.log"
else
  echo "[matrix] NO REPORT WRITTEN -- the matrix did not reach its end."
fi
case "$MATRIX_RC" in
  0) echo "[matrix] every architecture ran every stage with its expected outcome." ;;
  1) echo "[matrix] COMPLETED; at least one architecture did not."
     echo "[matrix] That is the finding; the report's Findings block names the"
     echo "[matrix] stage, its exit code and its log. Not a crash." ;;
  2) echo "[matrix] the matrix refused its command line (usage error, its own"
     echo "[matrix] exit 2, reported here as 3). Nothing was measured and no"
     echo "[matrix] architecture was scored; the usage line above names the flag." ;;
  *) echo "[matrix] FAILED -- see the log above." ;;
esac
echo "[matrix] END  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[matrix] ==================================================="
exit "$RC"
