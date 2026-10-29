#!/bin/bash
#SBATCH --job-name=xcq_slim16_refs
#SBATCH --partition=extended-40core
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=slim16_refs_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# The v9 letter arms' inline-readout references: benchmark_refs in pool mode
# writes every slim16 species (the 100 reactions' 239 compounds over 37
# subsets) as a CCSD density npz at the READOUT's footing -- def2-TZVP,
# grid 4, density fitting on the jkfit mesh, unlocked -- into the
# eval_refs_dir the arms' readout consumes. One shared directory for every
# network and arm: the readout is a comparison surface, so every model must
# read the SAME references. Run it from the cluster checkout AFTER the v9
# merge is pulled in and BEFORE (or while) the arms train: the readout's
# guard refuses a pass whose species lack a complete, footing-matching npz
# here. The generator is resumable per species (existing complete npz files
# are skipped), so a wall-kill just needs a resubmit.
#
# The footing is read from the arm configuration itself (the eval_* readout
# block, never a line match), so this job and the readout cannot drift;
# both arms state the block identically.
#
# Shell discipline (house pattern): NO `set -e` -- conda activation can
# return non-zero on success; -uo pipefail, activation verified by its
# EFFECT, explicit rc gate on the generation stage.
set -uo pipefail
ENV_PREFIX=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j0102
# --- conda activation (verified by effect, not by return code) --------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[sl16-refs] host=$(hostname) python=$PYBIN"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[sl16-refs] FATAL: parity env python not active (got: ${PYBIN:-none})"
     exit 1 ;;
esac

REPO="${XCQ_REPO:-/gpfs/projects/FernandezGroup/Alec/xcquinox}"
CONFIG="${XCQ_V9_CONFIG:-hpcjobs/configs/dfs_step8.v9_dfs_letter_parity.yaml}"

cd "${REPO}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu
export MALLOC_ARENA_MAX=2
# PySCF runs in this job's own process, so its pools are capped at the
# measured optimum instead of the allocation (about ten minutes per
# molecule at 40 threads on a 40-core node against 8 s at four,
# workflow-matrix job 2134488).
THREADS="${SLURM_CPUS_PER_TASK:-8}"
case "$THREADS" in ''|*[!0-9]*) THREADS=8 ;; esac
THREADS=$(( 10#$THREADS ))
[ "$THREADS" -ge 1 ] || THREADS=1
[ "$THREADS" -le 8 ] || THREADS=8
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
echo "[sl16-refs] allocation=${SLURM_CPUS_PER_TASK:-unset} -> OMP/MKL/OPENBLAS=$THREADS"

python -c "import xcquinox.pipeline.benchmark_refs" \
    || { echo "[sl16-refs] FATAL: repo import failed (env/activation)"; exit 1; }

# the readout's own footing and pool, from the eval_* block
read -r REFS POOL BASIS GRID DF AUX OL < <(python - "${CONFIG}" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
vals = [cfg.get("eval_refs_dir") or "", cfg.get("eval_pool") or "",
        cfg.get("eval_basis") or "", cfg.get("eval_grid_level", ""),
        str(bool(cfg.get("eval_density_fit", False))).lower(),
        cfg.get("eval_auxbasis") or "-",
        cfg.get("eval_orientation_lock", 0.0)]
print(" ".join(str(v) for v in vals))
PYEOF
)
if [ -z "${REFS}" ] || [ -z "${POOL}" ] || [ -z "${BASIS}" ] || [ -z "${GRID}" ]; then
    echo "[sl16-refs] FATAL: readout block unresolved from ${CONFIG} (refs=${REFS:-unset} pool=${POOL:-unset})"
    exit 1
fi
DF_ARGS=""
if [ "${DF}" = "true" ]; then
    DF_ARGS="--density-fit"
    [ "${AUX}" != "-" ] && DF_ARGS="${DF_ARGS} --auxbasis ${AUX}"
fi
echo "[sl16-refs] out=${REFS} pool=${POOL} basis=${BASIS} grid=${GRID} df='${DF_ARGS}' lock=${OL}"

mkdir -p "${REFS}"
rc=0
# shellcheck disable=SC2086
python -m xcquinox.pipeline.benchmark_refs --out-dir "${REFS}" \
    --pool "${POOL}" --basis "${BASIS}" --grid-level "${GRID}" \
    ${DF_ARGS} --orientation-lock-strength "${OL}" || rc=$?
if [ "${rc}" -ne 0 ]; then
    echo "[sl16-refs] generation FAILED rc=${rc}"
    exit "${rc}"
fi
echo "[sl16-refs] DONE"
exit 0
