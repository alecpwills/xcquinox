#!/bin/bash
#SBATCH --job-name=xcq_refinalize_verbatim
#SBATCH --partition=short-40core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=refinalize_verbatim_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alec.wills@stonybrook.edu

# Refinalize completed held-out evals under the verbatim hold-out rule
# (2026-08-13): rewrites per_reaction.json/test_set.csv of specs evaluated
# before the rule deployed, from their EXISTING per-species energies -- no
# SCF. Idempotent; first rewrite per channel backs up the old artifacts as
# *.pre_verbatim.*; specs already on the new rule report "unchanged".
# Safe alongside a running train array: only completed specs' directories
# are touched, and the arrays write their own spec dirs.
#
# Override targets with REFINALIZE_RUN_DIRS (space-separated run dirs).

set -uo pipefail

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

python -c "import xcquinox.alec.refinalize_verbatim" \
    || { echo "FATAL: repo import failed (env/activation)"; exit 1; }

echo "[refinalize] targets:${RUN_DIRS}"
rc=0
# shellcheck disable=SC2086
python -m xcquinox.alec.refinalize_verbatim ${RUN_DIRS} || rc=$?
echo "[refinalize] DONE rc=${rc}"
exit "${rc}"
