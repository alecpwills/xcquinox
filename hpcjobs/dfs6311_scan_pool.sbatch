#!/usr/bin/env bash
#SBATCH --job-name=xcq_scan_pool
# long-40core, NOT long-96core: the 96-core queue is saturated by the production
# training array and this job must not queue behind it or compete with it for
# nodes (same reasoning as the CH/NO relock job). One KS-SCF per species is not
# a 96-core problem; --exclusive with --mem=0 still takes the whole node.
#SBATCH --partition=long-40core
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=40
#SBATCH --output=/gpfs/scratch/awills/scan_pool_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=END,FAIL

# SCAN meta-GGA reference over the held-out BH76+W4-11 pool at the production
# identity. 214 unique species, one SCAN KS-SCF each.
#
# WHY: the sweep now trains meta-GGA architectures (deep_mgga_3x16 landed as
# spec 0033; the attn and rung3.5+mgga variants follow). Those nets pretrain to
# SCAN, so the question they exist to answer -- does a net trained on CCSD
# densities beat the meta-GGA it clones? -- cannot be read off a figure whose
# only reference is PBE, a GGA. The figure code has carried SCAN reference lines
# since the rung figures were written; they have never rendered because no SCAN
# cache was ever generated (scan_pool_baseline degrades to all-NaN and the
# figures silently omit the line).
#
# TWO CACHES, ONE PASS. run_scf_with_cache persists the AO-basis density matrix
# beside e_tot, so --with-density scores SCAN's density against the CCSD
# references by contracting that stored dm onto the reference grid -- no second
# SCF. Output: scan_pool_energies_<basis>.json + scan_pool_density_<basis>.json.
#
# ORIENTATION LOCK. --orientation-lock MUST match the lock the references were
# generated with (production: 3e-05). For the degenerate 2-Pi radicals CH/NO/OH
# an unlocked SCAN SCF settles on a different member of the (pi_x, pi_y)
# manifold than the reference, and the resulting density error is a component
# mismatch rather than a functional error -- the artifact diagnosed in
# notebooks/analysis/DENSITY_DIAGNOSIS.md. The density leg refuses to score any
# species whose reference carries no lock stamp, so a pre-stamp reference fails
# loudly here instead of contributing a meaningless number.
#
# READ-ONLY against the live references: SCAN artifacts go to --out-dir, which
# must not be the refs dir. The production array is still reading those
# references while this runs.
#
# RESUMABLE at both layers (per-species SCF npz + atomic JSON after each
# species), so a wall-clock timeout is recovered by resubmitting unchanged.
#
# Shell discipline (job 2099698 lesson): NO `set -e` -- conda activation can
# return non-zero on success and would kill the job seconds in with an empty
# log. House pattern: -uo pipefail, activation verified by EFFECT (the resolved
# python path), explicit rc gate after the stage.
set -uo pipefail

ENV_PREFIX=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j070
REPO=/gpfs/projects/FernandezGroup/Alec/xcquinox
# The BENCHMARK density references (held-out species, rho_ref_grid +
# grid_weights), NOT the training/OEP refs -- the density figures score against
# these. Value copied from the production config's inputs.benchmark_refs_dir.
REFS=/gpfs/scratch/awills/external_refs_bench_6311ppg3df2pd_g3
OUT="${1:-/gpfs/scratch/awills/scan_pool_6311ppg3df2pd_g3}"

# Production identity -- mirrors dfs_step7.dfs6311_grid3_v3.yaml inputs exactly
# (basis, grid_level 3, density_fit true, auxbasis unset -> auto,
# orientation_lock_strength 0.00003).
BASIS='6-311++G(3df,2pd)'
GRID=3
LOCK=0.00003

T_START_EPOCH="$(date +%s)"
echo "[scan] ==================================================="
echo "[scan] START  $(date '+%Y-%m-%d %H:%M:%S %Z')  (epoch $T_START_EPOCH)"
echo "[scan] job=${SLURM_JOB_ID:-none}"
echo "[scan] ==================================================="

# --- conda activation (verified by effect, not by return code) --------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[scan] host=$(hostname) python=$PYBIN"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[scan] FATAL: parity env python not active (got: ${PYBIN:-none})"
     exit 1 ;;
esac

# --- thread caps (SLURM vars not assumed present) ---------------------------
THREADS="${SLURM_CPUS_PER_TASK:-40}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export MALLOC_ARENA_MAX=2
ulimit -u unlimited 2>/dev/null || true
echo "[scan] BLAS threads: $THREADS  MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX"

echo "[scan] density references (READ-ONLY here): $REFS"
echo "[scan] SCAN out-dir:                        $OUT"
echo "[scan] identity: basis=$BASIS grid=$GRID density_fit=on lock=$LOCK"
if [ "$OUT" = "$REFS" ]; then
  echo "[scan] FATAL: --out-dir must NOT be the reference dir; this job writes"
  echo "[scan]        SCAN artifacts and must never touch the live references."
  exit 1
fi
if [ ! -d "$REFS" ]; then
  echo "[scan] FATAL: reference dir not found: $REFS"
  echo "[scan]        (the density leg needs <name>.npz from benchmark_refs)"
  exit 1
fi
mkdir -p "$OUT" || { echo "[scan] FATAL: cannot create out-dir"; exit 1; }

# --- generation --------------------------------------------------------------
cd "$REPO" || { echo "[scan] FATAL: cannot cd $REPO"; exit 1; }
python notebooks/analysis/precompute_scan_pool.py \
    --basis "$BASIS" \
    --grid "$GRID" \
    --density-fit \
    --pool all \
    --orientation-lock "$LOCK" \
    --with-density \
    --refs-dir "$REFS" \
    --out-dir "$OUT"
RC=$?

T_END_EPOCH="$(date +%s)"
echo "[scan] ==================================================="
echo "[scan] rc=$RC  elapsed $((T_END_EPOCH - T_START_EPOCH)) s"
ls -l "$OUT"/scan_pool_*.json 2>/dev/null || echo "[scan] no cache JSON written"
if [ "$RC" -ne 0 ]; then
  echo "[scan] one or more species FAILED -- the caches hold every species that"
  echo "[scan] succeeded and resubmitting this script unchanged resumes the rest."
  echo "[scan] A partial cache is safe to pull: the figures measure SCAN's"
  echo "[scan] coverage against PBE's and withdraw the line below 90%."
fi
echo "[scan] END  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[scan] ==================================================="
exit "$RC"
