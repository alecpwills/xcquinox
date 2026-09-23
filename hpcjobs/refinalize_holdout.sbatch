#!/bin/bash
#SBATCH --job-name=xcq_refinalize_holdout
#SBATCH --partition=short-96core-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=refinalize_holdout_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alec.wills@stonybrook.edu

# Rewrite the tables of completed held-out evaluations from their EXISTING
# per-species energies -- no SCF -- after a change in the pool definitions or
# in the aggregation. Nothing is excluded: every reaction of every set is
# reported. Idempotent; the first rewrite per channel backs up the old
# artifacts as *.pre_refinalize.*; a channel already current reports
# "unchanged".
# Safe alongside a running train array: only completed specs' directories
# are touched, and the arrays write their own spec dirs.
#
# Override targets with REFINALIZE_RUN_DIRS (space-separated run dirs).

set -uo pipefail
ENV_PREFIX=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j0102
# --- conda activation (verified by effect, not by return code) --------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[${SLURM_JOB_NAME:-job}] host=$(hostname) python=$PYBIN"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[${SLURM_JOB_NAME:-job}] FATAL: parity env python not active (got: ${PYBIN:-none})"
     exit 1 ;;
esac

REPO="${XCQ_REPO:-/gpfs/projects/FernandezGroup/Alec/xcquinox}"
SCRATCH="${XCQ_SCRATCH:-/gpfs/scratch/awills/xcquinox_runs}"
RUN_DIRS="${REFINALIZE_RUN_DIRS:-}"

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

python -c "import xcquinox.pipeline.refinalize_holdout" \
    || { echo "FATAL: repo import failed (env/activation)"; exit 1; }

echo "[refinalize] targets:${RUN_DIRS}"
rc=0
# shellcheck disable=SC2086
python -m xcquinox.pipeline.refinalize_holdout ${RUN_DIRS} || rc=$?
echo "[refinalize] DONE rc=${rc}"
exit "${rc}"
