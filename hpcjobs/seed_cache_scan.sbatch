#!/bin/bash
#SBATCH --job-name=xcq_seedcache
#SBATCH --partition=short-40core
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=seed_cache_scan_%j.out

# SCAN seed cache for the v5 TRAINING-side species: one converged SCAN
# KS-SCF per species of the full training-point pool (~32: 23 molecules +
# 7 atoms + 2 IP13 cations; enumerated programmatically from the config's
# own pool builder, so a pool change can never silently truncate the
# list), written into inputs.seed_cache_dir/_intermediates/ at the run
# identity INCLUDING the orientation lock. Resumable: cached species are
# skipped by the cache layer; non-converged SCFs escalate and are never
# cached. Eval/val (pool) species are NOT run here -- the scan-pool cache
# already holds all 214 at the production identity; symlink or copy its
# _intermediates/ contents into the seed cache root (see below).
#
# Run BEFORE the v5 submits so the 55 array tasks do not race-generate
# the training seeds.

set -uo pipefail

REPO=${REPO:-/gpfs/projects/FernandezGroup/Alec/xcquinox}
CONFIG=${SEED_CACHE_CONFIG:-${REPO}/hpcjobs/configs/dfs_step7.dfs6311_grid3_v5.yaml}
SCAN_POOL_DIR=${SCAN_POOL_DIR:-/gpfs/scratch/awills/scan_pool_6311ppg3df2pd_g3}

CONDA_PROFILE=${CONDA_PROFILE:-/gpfs/software/anaconda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/gpfs/projects/FernandezGroup/Alec/envs/xcq}
if [ -f "${CONDA_PROFILE}" ]; then
    source "${CONDA_PROFILE}"
    conda activate "${CONDA_ENV}" || true
fi

cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

python -c "import xcquinox.alec.cluster.seed_cache" || {
    echo "FATAL: xcquinox import failed -- wrong env?"; exit 1; }

# Resolve the seed cache dir from the config (single source of truth).
SEED_DIR=$(python - "${CONFIG}" <<'PYEOF'
import sys
from xcquinox.alec.cluster.grid_config import load_grid_config
print(load_grid_config(sys.argv[1]).inputs.seed_cache_dir or "")
PYEOF
)
if [ -z "${SEED_DIR}" ]; then
    echo "FATAL: inputs.seed_cache_dir unset in ${CONFIG}"
    exit 1
fi
mkdir -p "${SEED_DIR}/_intermediates"

# Pool species: link the existing scan-pool intermediates in (idempotent;
# ln -s refuses to overwrite an existing link/file, which is fine).
if [ -d "${SCAN_POOL_DIR}/_intermediates" ]; then
    n_linked=0
    for f in "${SCAN_POOL_DIR}"/_intermediates/*_xcscan_scf.npz; do
        [ -e "${f}" ] || continue
        ln -s "${f}" "${SEED_DIR}/_intermediates/" 2>/dev/null && \
            n_linked=$((n_linked + 1))
    done
    echo "[seed-cache] linked ${n_linked} scan-pool intermediates from ${SCAN_POOL_DIR}"
else
    echo "[seed-cache] WARNING: ${SCAN_POOL_DIR}/_intermediates not found -- pool species not linked"
fi

python -m xcquinox.alec.cluster.seed_cache "${CONFIG}"
rc=$?
echo "[seed-cache] exit ${rc}"
exit ${rc}
