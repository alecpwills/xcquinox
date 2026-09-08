#!/bin/bash
#SBATCH --job-name=xcq_converged
# long-40core, NOT short-40core: the short queues cap at 4 h (see
# hpcjobs/SEAWULF_RUNBOOK.md) and a converged pass runs up to 100 DIIS
# cycles for each of ~200 pool species on 25 completed cells, twice
# (final and val-best checkpoints). Done-skip resumability covers anything
# a wall-kill leaves; resubmit to finish.
#SBATCH --partition=long-40core
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=converged_retro_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Retroactive converged-SCF channel (eval_holdout_converged and its
# _val_best twin) over the COMPLETED specs of the v7 arms: each checkpoint
# re-evaluated under a converged pyscfad SCF (PBE seed, DIIS, 100 cycles at
# 1e-8 Ha, the trained density fitting kept), so the held-out density is
# compared with the converged CCSD reference on the same footing as the
# converged PBE baseline. Resumable (done specs skipped; a spec whose val-best
# pass was killed reads as ready and runs that pass alone), additive (never
# touches existing channels; pending specs simply are not completed yet).
#
# A PREFLIGHT gates the array line: the shard worker evaluates three species
# of spec 0 of the first run dir under the converged override (h2o must
# converge; bn and RKT17 are the audit's tail species and must at least
# record their cycle counts) into a scratch shard that is inspected before
# anything is written into a channel directory.
#
# Submit from a shell whose python resolves the xcquinox parity environment
# (activation by effect, verified by the import probe).

set -uo pipefail

REPO="${XCQ_REPO:-/gpfs/projects/FernandezGroup/Alec/xcquinox}"
SCRATCH="${XCQ_SCRATCH:-/gpfs/scratch/awills/xcquinox_runs}"
RUN_DIRS="${CONVERGED_RUN_DIRS:-}"

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

python -c "import xcquinox.alec.eval_holdout" \
    || { echo "FATAL: repo import failed (env/activation)"; exit 1; }

# The benchmark reference densities resolve ONLY through this env var (the
# rendered arm templates export it; nothing reads it from config at eval
# time). Without it the channel writes null for every density column.
FIRST_RUN="$(echo ${RUN_DIRS} | awk '{print $1}')"
if [ -z "${XCQUINOX_BENCH_REFS_DIR:-}" ]; then
    for rd in ${RUN_DIRS}; do
        cand="$(sed -n 's/^[[:space:]]*benchmark_refs_dir:[[:space:]]*//p' "${rd}/resolved_config.yaml" 2>/dev/null | head -1 | tr -d "'\"")"
        if [ -n "${cand}" ]; then
            XCQUINOX_BENCH_REFS_DIR="${cand}"
            export XCQUINOX_BENCH_REFS_DIR
            break
        fi
    done
fi
if [ -z "${XCQUINOX_BENCH_REFS_DIR:-}" ] || [ ! -d "${XCQUINOX_BENCH_REFS_DIR}" ]; then
    echo "FATAL: benchmark refs dir unresolved (${XCQUINOX_BENCH_REFS_DIR:-unset}) -- refusing to write null density columns"
    exit 1
fi

# --- preflight: three species of spec 0 under the converged override --------
# basis and grid level as the held-out eval resolves them (the run's own
# inputs block, read through a YAML parser rather than a line match).
read -r PF_BASIS PF_GRID < <(python - "${FIRST_RUN}/resolved_config.yaml" <<'PYEOF'
import sys, yaml
inp = (yaml.safe_load(open(sys.argv[1])) or {}).get("inputs") or {}
print(inp.get("basis") or "", inp.get("grid_level", ""))
PYEOF
)
if [ -z "${PF_BASIS}" ] || [ -z "${PF_GRID}" ]; then
    echo "FATAL: basis/grid_level unresolved from ${FIRST_RUN}/resolved_config.yaml"
    exit 1
fi
# scratch OUTSIDE the checkpoint directories: nothing in a channel namespace
PF_DIR="${FIRST_RUN}/_preflight_converged"
mkdir -p "${PF_DIR}"
printf '%s\n' '["h2o", "bn", "RKT17"]' > "${PF_DIR}/names.json"
echo "[converged] preflight: basis=${PF_BASIS} grid=${PF_GRID} species h2o bn RKT17"
python xcquinox/alec/workers/eval_holdout_worker.py \
    --run-dir "${FIRST_RUN}" --spec-idx 0 \
    --names-file "${PF_DIR}/names.json" --out-shard "${PF_DIR}/shard.json" \
    --basis "${PF_BASIS}" --grid-level "${PF_GRID}" --threads "${BLAS_THREADS}" \
    --model-name model.eqx --channel converged \
    || { echo "FATAL: preflight worker failed"; exit 1; }
python - "${PF_DIR}/shard.json" <<'PYEOF' || { echo "FATAL: preflight shard did not pass its checks"; exit 1; }
import json, sys
shard = json.load(open(sys.argv[1]))
recs = {r.get("molecule"): r for r in shard.get("mol_records", [])}
ok = True
for name in ("h2o", "bn", "RKT17"):
    r = recs.get(name)
    if r is None:
        print(f"[converged] preflight: {name} has no record"); ok = False; continue
    print(f"[converged] preflight: {name} converged={r.get('scf_converged')} cycles={r.get('cycles_run')} rho_rmse={r.get('density_rmse')}")
    if r.get("cycles_run") is None:
        ok = False
if recs.get("h2o", {}).get("scf_converged") is not True:
    print("[converged] preflight: h2o did not converge"); ok = False
sys.exit(0 if ok else 1)
PYEOF

# --- the retro array line ----------------------------------------------------
echo "[converged] targets:${RUN_DIRS}"
echo "[converged] refs=${XCQUINOX_BENCH_REFS_DIR} blas=${BLAS_THREADS}"
rc=0
# shellcheck disable=SC2086
python -m xcquinox.alec.cluster.channel_retro ${RUN_DIRS} --channel converged || rc=$?
echo "[converged] DONE rc=${rc}"
exit "${rc}"
