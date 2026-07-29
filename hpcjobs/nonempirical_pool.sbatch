#!/usr/bin/env bash
#SBATCH --job-name=xcq_nonemp_pool
#SBATCH --partition=long-96core
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=24
#SBATCH --output=/gpfs/scratch/awills/nonemp_pool_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=END,FAIL

# Nonempirical-functional calibration pool for the DFS Eq. 21 gamma fit:
# converged KS-SCF total energies + Eq. 20 per-electron L1 density errors
# (eps) for PW91/PBE/TPSS/revTPSS/SCAN/PBE0 over the full held-out pool, on
# the dfs6311 axes (6-311++G(3df,2pd), grid_level 3, density fitting),
# against the SAME CCSD benchmark reference densities the NN eval uses.
# The figure code fits gamma as the zero-intercept regression of WTMAD-2 on
# eps across the six functionals -- the Letter's Fig. 3 procedure on our
# axes (make_ablation_arch_figure.nonempirical_gamma).
#
# SUBMIT AFTER the live sweep completes (queue competition only -- this job
# reads the production refs npz read-only and writes exclusively to its own
# out-dir; it never touches a run dir).
#
# Walltime: 48 h requested; the long-96core per-job TimeLimit is not recorded
# in the runbook (long-28core is 2 days) -- check `sinfo -p long-96core -o
# "%P %l"` before submitting and trim if needed; a timeout is harmless (both
# cache layers resume on resubmit).
#
# Out-dir (first sbatch argument, optional; default below) MUST be a fresh /
# job-owned directory, NEVER the production refs dir: pbe-leg SCFs would hit
# the legacy pre-e_tot intermediate caches there and fail loudly on every
# species. Both cache layers (per-(species,xc) JSON entries + per-species
# SCF npz) are resumable -- a resubmit after timeout continues where it
# stopped.
#
# Deliverable: <out-dir>/nonempirical_pool_6-311++G_3df_2pd_.json. Pull it
# next to the run's figures input (the pulled run dir) so the figure suite
# picks up the gamma-fit ED leg automatically.
#
# Shell discipline (job 2099698 lesson): NO `set -e` -- conda activation can
# return non-zero on success. House pattern: -uo pipefail, activation
# verified by its EFFECT (resolved python path), explicit rc gate per stage.
set -uo pipefail

ENV_PREFIX=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j070
# --- conda activation (verified by effect, not by return code) --------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[nonemp-pool] host=$(hostname) python=$PYBIN"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[nonemp-pool] FATAL: parity env python not active (got: ${PYBIN:-none})"
     exit 1 ;;
esac

# --- thread caps (defensive fallback; SLURM vars are not assumed present) ----
THREADS="${SLURM_CPUS_PER_TASK:-24}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
ulimit -u unlimited 2>/dev/null || true
echo "[nonemp-pool] BLAS threads: $THREADS"

REFS=/gpfs/scratch/awills/external_refs_bench_6311ppg3df2pd_g3
OUT="${1:-/gpfs/scratch/awills/nonempirical_pool_6311ppg3df2pd_g3}"
echo "[nonemp-pool] refs: $REFS"
echo "[nonemp-pool] out-dir: $OUT"
[ -d "$REFS" ] || { echo "[nonemp-pool] FATAL: refs dir missing"; exit 1; }
# path-normalized CONTAINMENT guard: the out-dir must be neither the refs
# dir itself nor anywhere inside it (an _intermediates/ tree landing in the
# refs dir would pollute the production cache); trailing slashes / ./
# segments / relative spellings must not evade it
REFS_REAL="$(realpath -m "$REFS")"
OUT_REAL="$(realpath -m "$OUT")"
case "$OUT_REAL/" in
  "$REFS_REAL"/*)
    echo "[nonemp-pool] FATAL: out-dir must NOT be the refs dir or inside it"
    exit 1 ;;
esac
mkdir -p "$OUT" || { echo "[nonemp-pool] FATAL: cannot create out-dir"; exit 1; }

# --- generation (resumable; atomic JSON after every (species, xc)) -----------
# --orientation-lock-strength matches the production refs stamps (3e-05);
# the script verifies every reference's identity stamps and fails loudly on
# a mismatch.
python /gpfs/projects/FernandezGroup/Alec/xcquinox/notebooks/analysis/precompute_nonempirical_pool.py \
  --basis "6-311++G(3df,2pd)" \
  --grid 3 --density-fit \
  --orientation-lock-strength 3e-05 \
  --density-refs "$REFS" \
  --out-dir "$OUT"
rc=$?
[ "$rc" -eq 0 ] || { echo "[nonemp-pool] FATAL: precompute exited rc=$rc (failed entries above; resubmit resumes)"; exit "$rc"; }

# --- verification: cache present, all six functionals at FULL coverage -------
# (rc=0 above means every (species, xc) succeeded, so each functional must
# cover every cached species; anything less indicates a partial/stale cache)
python - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "nonempirical_pool_6-311++G_3df_2pd_.json"
assert p.is_file(), f"missing cache: {p}"
cache = json.loads(p.read_text())
assert cache, "cache is empty"
xcs = ("pw91", "pbe", "tpss", "revtpss", "scan", "pbe0")
per_xc = {xc: sum(1 for v in cache.values() if xc in v) for xc in xcs}
print(f"[nonemp-pool] species cached: {len(cache)}; per-functional coverage: {per_xc}")
short = {xc: n for xc, n in per_xc.items() if n != len(cache)}
assert not short, f"functionals below full coverage ({len(cache)}): {short}"
PYEOF
rc=$?
[ "$rc" -eq 0 ] || { echo "[nonemp-pool] FATAL: cache verification exited rc=$rc"; exit "$rc"; }
echo "[nonemp-pool] DONE"
exit 0
