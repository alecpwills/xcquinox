#!/bin/bash
#SBATCH --job-name=xcq_coldstart
# long-40core, NOT short-40core: the short queues cap at 4 h (see
# hpcjobs/SEAWULF_RUNBOOK.md) and this job honestly needs more -- ~26
# completed specs x ~8x a 3-cycle channel pass each. Done-skip
# resumability covers anything a wall-kill leaves.
#SBATCH --partition=long-40core
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=coldstart_holdout_%j.out

# Retroactive cold-start channel (eval_holdout_coldstart) over the COMPLETED
# specs of the v4 arms. Resumable: done specs are skipped, pending specs are
# not completed yet -- safe to run beside the in-flight arrays and to
# resubmit after they drain. Walltime: the channel runs the full 25-cycle
# trajectory for every one of ~200 pool species per spec (conv_tol 1e-12
# never converges early under linear mixing), so budget ~8x a single
# 3-cycle channel pass per spec; resubmits pick up whatever remains.
# Submit from a shell whose python resolves the xcquinox environment
# (activation by effect, verified by the import probe).

set -uo pipefail

REPO="${XCQ_REPO:-/gpfs/projects/FernandezGroup/Alec/xcquinox}"
SCRATCH="${XCQ_SCRATCH:-/gpfs/scratch/awills/xcquinox_runs}"
RUN_DIRS="${COLDSTART_RUN_DIRS:-}"

if [ -z "${RUN_DIRS}" ]; then
    for arm in dfs6311_grid3_v4 dfs6311_grid3_v4gga dfs6311_grid3_v4mgga2; do
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

python -c "import xcquinox.alec.cluster.coldstart_retro" \
    || { echo "FATAL: repo import failed (env/activation)"; exit 1; }

echo "[coldstart] targets:${RUN_DIRS}"
rc=0
# shellcheck disable=SC2086
python -m xcquinox.alec.cluster.coldstart_retro ${RUN_DIRS} || rc=$?
echo "[coldstart] DONE rc=${rc}"
exit "${rc}"
