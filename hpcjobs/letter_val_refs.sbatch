#!/bin/bash
#SBATCH --job-name=xcq_letter_val_refs
#SBATCH --partition=long-96core-shared
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=letter_val_refs_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# The v9 Letter-parity arms' validation references: benchmark_refs in
# --manifest mode writes every si_letter_validation.json species that
# carries a reference -- the 8 AE compounds, the element atoms and the
# barrier species -- as CCSD density npz files at the arms' own footing,
# into the val_refs_dir the arms read. No held-out pool carries BeH or PF3,
# so pool mode cannot reach these species; this job is the only producer.
# Run it from the cluster checkout AFTER the v9 merge is pulled in and
# BEFORE the v9 chain submissions: the first in-loop validation (epoch 10)
# loads these files, and the loader's stamp assert fails loudly on a
# missing or differently-stamped reference.
#
# The footing is read from the arm configuration itself (the inputs block,
# never a line match), so this job and the training runs cannot drift:
# basis 6-311++G(3df,2pd), grid 3, density fitting, orientation lock 3e-5.
# The generator is resumable per species (existing complete npz files are
# skipped), so a wall-kill just needs a resubmit.
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
echo "[letter-val-refs] host=$(hostname) python=$PYBIN"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[letter-val-refs] FATAL: parity env python not active (got: ${PYBIN:-none})"
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
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
echo "[letter-val-refs] allocation=${SLURM_CPUS_PER_TASK:-unset} -> OMP/MKL/OPENBLAS=$THREADS"

python -c "import xcquinox.pipeline.benchmark_refs" \
    || { echo "[letter-val-refs] FATAL: repo import failed (env/activation)"; exit 1; }

# the arm's own footing and targets, from the inputs block
read -r VAL_REFS MANIFEST BASIS GRID DF AUX OL < <(python - "${CONFIG}" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
inp = cfg.get("inputs") or {}
vals = [inp.get("val_refs_dir") or "", inp.get("validation_manifest") or "",
        inp.get("basis") or "", inp.get("grid_level", ""),
        str(bool(inp.get("density_fit", False))).lower(),
        inp.get("auxbasis") or "-", inp.get("orientation_lock_strength", 0.0)]
print(" ".join(str(v) for v in vals))
PYEOF
)
if [ -z "${VAL_REFS}" ] || [ -z "${MANIFEST}" ] || [ -z "${BASIS}" ] || [ -z "${GRID}" ]; then
    echo "[letter-val-refs] FATAL: inputs unresolved from ${CONFIG} (val_refs=${VAL_REFS:-unset} manifest=${MANIFEST:-unset})"
    exit 1
fi
DF_ARGS=""
if [ "${DF}" = "true" ]; then
    DF_ARGS="--density-fit"
    [ "${AUX}" != "-" ] && DF_ARGS="${DF_ARGS} --auxbasis ${AUX}"
fi
echo "[letter-val-refs] out=${VAL_REFS} manifest=${MANIFEST} basis=${BASIS} grid=${GRID} df='${DF_ARGS}' lock=${OL}"

mkdir -p "${VAL_REFS}"
rc=0
# shellcheck disable=SC2086
python -m xcquinox.pipeline.benchmark_refs --out-dir "${VAL_REFS}" \
    --manifest "${MANIFEST}" --basis "${BASIS}" --grid-level "${GRID}" \
    ${DF_ARGS} --orientation-lock-strength "${OL}" || rc=$?
if [ "${rc}" -ne 0 ]; then
    echo "[letter-val-refs] generation FAILED rc=${rc}"
    exit "${rc}"
fi
echo "[letter-val-refs] DONE"
exit 0
