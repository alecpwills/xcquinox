#!/bin/bash
#SBATCH --job-name=xcq_reeval_spec
# Re-runs the full evaluation (in-sample eval plus every held-out checkpoint
# channel) for ONE completed spec via cluster._eval_one_spec -- the repair
# path when a degraded eval leaves NaN energies or a reference outlier
# behind (the figure layer's cross-spec consistency guard names the spec to
# re-run). Channels are overwritten in place; the shard scheduler only ever
# consumes shard files written by its own invocation, so no stale shard
# output from the degraded attempt can be reused. SBATCH directives mirror
# the arm's train_eval_inline template; the environment that template
# provides (benchmark-refs dir, malloc arena cap, BLAS thread caps) is
# replicated below -- without the refs dir the eval would silently null
# every density column it overwrites.
#SBATCH --partition=long-40core
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=24
#SBATCH --output=reeval_holdout_spec_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -uo pipefail

REPO="${XCQ_REPO:-/gpfs/projects/FernandezGroup/Alec/xcquinox}"
SCRATCH="${XCQ_SCRATCH:-/gpfs/scratch/awills/xcquinox_runs}"
RUN_DIR="${REEVAL_RUN_DIR:-}"
SPEC_IDX="${REEVAL_SPEC_IDX:-}"
# 20-way tier-1 (vs the template's 40) halves the peak memory pressure that
# degraded the original eval; the 6 h budget absorbs the ~2x wall cost.
N_CPUS="${REEVAL_TASKSET_CPUS:-20}"

if [ -z "${RUN_DIR}" ]; then
    RUN_DIR="$(ls -d "${SCRATCH}/dfs_step7/dfs6311_grid3_v4gga/runs"/run_* 2>/dev/null | sort | tail -1)"
    echo "[reeval] WARNING: REEVAL_RUN_DIR not set -- defaulting to the lexically newest run: ${RUN_DIR:-none}"
fi
if [ -z "${RUN_DIR}" ] || [ ! -d "${RUN_DIR}" ]; then
    echo "FATAL: run dir not found (set REEVAL_RUN_DIR)"
    exit 1
fi
# No default index: an accidental bare submit must fail loudly rather than
# silently re-evaluating some spec.
if [ -z "${SPEC_IDX}" ]; then
    echo "FATAL: set REEVAL_SPEC_IDX (the integer spec index to re-evaluate)"
    exit 1
fi

# activation by effect: whatever python resolves must import the repo tree
cd "${REPO}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export MALLOC_ARENA_MAX=2
CORES="${SLURM_CPUS_ON_NODE:-$(nproc --all)}"
BLAS_THREADS=$(( CORES / 12 ))
[ "${BLAS_THREADS}" -ge 1 ] || BLAS_THREADS=1
export OMP_NUM_THREADS="${BLAS_THREADS}"
export MKL_NUM_THREADS="${BLAS_THREADS}"
export OPENBLAS_NUM_THREADS="${BLAS_THREADS}"

python -c "import xcquinox.alec.cluster._eval_one_spec" \
    || { echo "FATAL: repo import failed (env/activation)"; exit 1; }

# The benchmark reference densities resolve ONLY through this env var: the
# rendered arm templates export it and nothing in the eval worker reads it
# from config. Running without it would overwrite every density column of
# the target spec with null, unrecoverably.
if [ -z "${XCQUINOX_BENCH_REFS_DIR:-}" ]; then
    XCQUINOX_BENCH_REFS_DIR="$(sed -n 's/^[[:space:]]*benchmark_refs_dir:[[:space:]]*//p' "${RUN_DIR}/resolved_config.yaml" | head -1 | tr -d "'\"")"
    export XCQUINOX_BENCH_REFS_DIR
fi
if [ -z "${XCQUINOX_BENCH_REFS_DIR:-}" ] || [ ! -d "${XCQUINOX_BENCH_REFS_DIR}" ]; then
    echo "FATAL: benchmark refs dir unresolved (${XCQUINOX_BENCH_REFS_DIR:-unset}) -- refusing to overwrite density columns with nulls"
    exit 1
fi

# Range-check the index against the run's own manifest: a typo must fail
# loudly, not mail a green no-op after fabricating checkpoints/spec_<typo>.
N_SPECS="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_specs'])" "${RUN_DIR}/manifest.json")" \
    || { echo "FATAL: cannot read n_specs from ${RUN_DIR}/manifest.json"; exit 1; }
if ! [ "${SPEC_IDX}" -ge 0 ] 2>/dev/null || [ "${SPEC_IDX}" -ge "${N_SPECS}" ]; then
    echo "FATAL: REEVAL_SPEC_IDX=${SPEC_IDX} outside [0, ${N_SPECS})"
    exit 1
fi

echo "[reeval] run_dir=${RUN_DIR} spec=${SPEC_IDX} refs=${XCQUINOX_BENCH_REFS_DIR} cpus=${N_CPUS} blas=${BLAS_THREADS}"
rc=0
taskset -c 0-$(( N_CPUS - 1 )) \
    python -m xcquinox.alec.cluster._eval_one_spec "${RUN_DIR}" "${SPEC_IDX}" || rc=$?
echo "[reeval] DONE rc=${rc}"
exit "${rc}"
