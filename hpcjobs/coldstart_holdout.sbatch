#!/bin/bash
#SBATCH --job-name=xcq_coldstart
#SBATCH --partition=short-40core
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
# 3-cycle channel pass per spec; 8 h covers the currently-completed specs
# with margin, and a resubmit picks up whatever remains.

set -uo pipefail

REPO=${REPO:-/gpfs/projects/FernandezGroup/Alec/xcquinox}
SCRATCH_ROOT=${SCRATCH_ROOT:-/gpfs/scratch/awills/xcquinox_runs/dfs_step7}
RUN_DIRS=${COLDSTART_RUN_DIRS:-}

CONDA_PROFILE=${CONDA_PROFILE:-/gpfs/software/anaconda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/gpfs/projects/FernandezGroup/Alec/envs/xcq}
if [ -f "${CONDA_PROFILE}" ]; then
    # activation-by-effect: source + activate, then verify by import below
    source "${CONDA_PROFILE}"
    conda activate "${CONDA_ENV}" || true
fi

cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

python -c "import xcquinox.alec.cluster.coldstart_retro" || {
    echo "FATAL: xcquinox import failed -- wrong env?"; exit 1; }

if [ -z "${RUN_DIRS}" ]; then
    # newest run per v4 arm (same discovery idiom as the refinalize job)
    for arm in dfs6311_grid3_v4 dfs6311_grid3_v4gga dfs6311_grid3_v4mgga2; do
        latest=$(ls -1d "${SCRATCH_ROOT}/${arm}/runs"/run_* 2>/dev/null | sort | tail -1)
        [ -n "${latest}" ] && RUN_DIRS="${RUN_DIRS} ${latest}"
    done
fi
if [ -z "${RUN_DIRS}" ]; then
    echo "FATAL: no run dirs found under ${SCRATCH_ROOT} and COLDSTART_RUN_DIRS unset"
    exit 1
fi

echo "[coldstart] run dirs:${RUN_DIRS}"
# shellcheck disable=SC2086 -- word splitting of RUN_DIRS is intended
python -m xcquinox.alec.cluster.coldstart_retro ${RUN_DIRS}
rc=$?
echo "[coldstart] exit ${rc}"
exit ${rc}
