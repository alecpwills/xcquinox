#!/usr/bin/env bash
#SBATCH --job-name=xcq_c2_ref_regen
#SBATCH --partition=long-96core
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=24
#SBATCH --output=/gpfs/scratch/awills/c2_ref_regen_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=END,FAIL

# Regenerate ONLY the c2 density-only benchmark reference under the
# deterministic-grid semantics (run_scf_with_cache stores the first attempt's
# guess-pruned grid regardless of SCF escalation; HISTORY.md Phase 35). Flags
# mirror the production benchmark_refs job exactly (basis / grid / DF / lock),
# so the regenerated identity stamps satisfy the same completeness predicate.
#
# Output directory (first sbatch argument, optional):
#   - default: <refs>_c2staging -- the SAFE mode while a sweep is running:
#     the production cache is untouched until the verified staged file is
#     swapped in with a single atomic rename.
#   - pass the production refs dir explicitly to regenerate IN PLACE, for the
#     stopped-harness flow (delete c2.npz + both c2_g3_* intermediates first,
#     else the stale intermediate cache is reused verbatim and the drifted
#     grid survives):
#       sbatch dfs6311_c2_ref_regen.sbatch \
#         /gpfs/scratch/awills/external_refs_bench_6311ppg3df2pd_g3
#
# Expected outcome in this job's log either way: the probe block reports c2
# with rho_ref_grid (26568,) and density_fit_used True. Runtime is minutes
# (one small-molecule CCSD); the 4 h limit is slack.
#
# Shell discipline (job 2099698 lesson): NO `set -e` -- conda activation can
# return non-zero on success, which under -e kills the job seconds in with an
# empty log. House pattern (dfs6311_nan_verify / dfs6311_rss_probe): -uo
# pipefail, activation verified by its EFFECT (the resolved python path), and
# an explicit rc gate after every stage so a failure names its own line.
set -uo pipefail

ENV_PREFIX=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j070
# --- conda activation (verified by effect, not by return code) --------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[c2-regen] host=$(hostname) python=$PYBIN"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[c2-regen] FATAL: parity env python not active (got: ${PYBIN:-none})"
     exit 1 ;;
esac

# --- thread caps (SLURM vars not assumed present) ---------------------------
# PySCF's OpenMP pool and numpy's OpenBLAS pool run at min(allocation, 8)
# threads each (parallel.PYSCF_POOL_THREADS_MAX, the rule the stage templates
# and the energy-weight sweep apply): sized to the allocation, the two
# spin-waiting pools stall the small dense operations of a PySCF SCF loop --
# measured at about ten minutes per molecule at 40 threads on a 40-core node
# against 8 s at four (workflow-matrix job 2134488).
THREADS="${SLURM_CPUS_PER_TASK:-8}"
case "$THREADS" in ''|*[!0-9]*) THREADS=8 ;; esac
THREADS=$(( 10#$THREADS ))
[ "$THREADS" -ge 1 ] || THREADS=1
[ "$THREADS" -le 8 ] || THREADS=8
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
ulimit -u unlimited 2>/dev/null || true
echo "[c2-regen] allocation=${SLURM_CPUS_PER_TASK:-unset} -> OMP/MKL/OPENBLAS=$THREADS"

REFS=/gpfs/scratch/awills/external_refs_bench_6311ppg3df2pd_g3
STAGE="${1:-${REFS}_c2staging}"
echo "[c2-regen] out-dir: $STAGE"
mkdir -p "$STAGE" || { echo "[c2-regen] FATAL: cannot create out-dir"; exit 1; }

# --- resolve c2's index in the sorted 'all' pool (fail loud on any drift) ----
IDX="$(python - <<'PYEOF'
from xcquinox.alec.benchmark_refs import load_benchmark_species
names = list(load_benchmark_species(
    "all", basis="6-311++G(3df,2pd)", grid_level=3))
i = names.index("c2")
assert names[i] == "c2"
print(i)
PYEOF
)" || { echo "[c2-regen] FATAL: species index resolution failed"; exit 1; }
[ -n "$IDX" ] || { echo "[c2-regen] FATAL: empty species index"; exit 1; }
echo "[c2-regen] species index $IDX (slice $IDX:$((IDX + 1)))"

# --- generation --------------------------------------------------------------
python -m xcquinox.alec.benchmark_refs \
  --out-dir "$STAGE" \
  --pool all \
  --basis "6-311++G(3df,2pd)" \
  --grid-level 3 --density-fit --orientation-lock-strength 3e-05 \
  --species-slice "$IDX:$((IDX + 1))"
rc=$?
[ "$rc" -eq 0 ] || { echo "[c2-regen] FATAL: benchmark_refs exited rc=$rc"; exit "$rc"; }

# --- verification ------------------------------------------------------------
echo "[c2-regen] result (expect c2 rho_ref_grid (26568,), density_fit_used True):"
python /gpfs/projects/FernandezGroup/Alec/xcquinox/hpcjobs/dfs6311_c2_ref_probe.py "$STAGE"
rc=$?
[ "$rc" -eq 0 ] || { echo "[c2-regen] FATAL: probe exited rc=$rc"; exit "$rc"; }
echo "[c2-regen] DONE"
exit 0
