#!/bin/bash
#SBATCH --job-name=xcq_t1_backfill
# long-40core: the backfill re-runs the CCSD stage (the SCF stage is served
# from its cache) for every benchmark species whose reference file lacks the
# T1 diagnostic -- one coupled-cluster calculation per species at the run's
# basis and grid, ~200 species -- and then writes the run-level
# t1_diagnostics.json the figure layer's outlier-free variant reads. The
# reference files are rewritten atomically with the key added and every
# other array unchanged; a wall-kill leaves complete files complete.
#SBATCH --partition=long-40core
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=t1_backfill_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# The reference dir, basis, grid level, density fitting and auxiliary basis
# are the run's own, read from the first target run's resolved_config.yaml,
# so the backfilled references are the ones the held-out eval consumes.
# Submit from a shell whose python resolves the xcquinox parity environment.

set -uo pipefail

REPO="${XCQ_REPO:-/gpfs/projects/FernandezGroup/Alec/xcquinox}"
SCRATCH="${XCQ_SCRATCH:-/gpfs/scratch/awills/xcquinox_runs}"
RUN_DIRS="${T1_RUN_DIRS:-}"

if [ -z "${RUN_DIRS}" ]; then
    for arm in dfs6311_grid3_v7g1_size dfs6311_grid3_v7g2a_families_core; do
        latest="$(ls -d "${SCRATCH}/dfs_step7/${arm}/runs"/run_* 2>/dev/null | sort | tail -1)"
        [ -n "${latest}" ] && RUN_DIRS="${RUN_DIRS} ${latest}"
    done
fi
if [ -z "${RUN_DIRS}" ]; then
    echo "FATAL: no run dirs found under ${SCRATCH}/dfs_step7/*/runs"
    exit 1
fi
FIRST_RUN="$(echo ${RUN_DIRS} | awk '{print $1}')"
CFG="${FIRST_RUN}/resolved_config.yaml"

cd "${REPO}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu
export MALLOC_ARENA_MAX=2
CORES="${SLURM_CPUS_ON_NODE:-$(nproc --all)}"
export OMP_NUM_THREADS="${CORES}"
export MKL_NUM_THREADS="${CORES}"
export OPENBLAS_NUM_THREADS="${CORES}"

python -c "import xcquinox.alec.benchmark_refs" \
    || { echo "FATAL: repo import failed (env/activation)"; exit 1; }

_cfg() { sed -n "s/^[[:space:]]*$1:[[:space:]]*//p" "${CFG}" | head -1 | tr -d "'\""; }
REFS_DIR="${XCQUINOX_BENCH_REFS_DIR:-$(_cfg benchmark_refs_dir)}"
BASIS="$(_cfg basis)"
GRID="$(_cfg grid_level)"
DF="$(_cfg density_fit)"
AUX="$(_cfg auxbasis)"
if [ -z "${REFS_DIR}" ] || [ ! -d "${REFS_DIR}" ]; then
    echo "FATAL: benchmark refs dir unresolved (${REFS_DIR:-unset})"
    exit 1
fi
DF_ARGS=""
if [ "${DF}" = "true" ] || [ "${DF}" = "True" ]; then
    DF_ARGS="--density-fit"
    [ -n "${AUX}" ] && [ "${AUX}" != "null" ] && DF_ARGS="${DF_ARGS} --auxbasis ${AUX}"
fi
echo "[t1] refs=${REFS_DIR} basis=${BASIS} grid=${GRID} df='${DF_ARGS}' runs:${RUN_DIRS}"

rc=0
# shellcheck disable=SC2086
python -m xcquinox.alec.benchmark_refs --out-dir "${REFS_DIR}" --pool all \
    --basis "${BASIS}" --grid-level "${GRID}" ${DF_ARGS} --t1-backfill || rc=$?
if [ "${rc}" -ne 0 ]; then
    echo "[t1] backfill FAILED rc=${rc} (see the _runlogs ledger); JSON not written"
    exit "${rc}"
fi
for rd in ${RUN_DIRS}; do
    # shellcheck disable=SC2086
    python -m xcquinox.alec.benchmark_refs --out-dir "${REFS_DIR}" --pool all \
        --basis "${BASIS}" --grid-level "${GRID}" ${DF_ARGS} --write-t1-json "${rd}" || rc=$?
done
echo "[t1] DONE rc=${rc}"
exit "${rc}"
