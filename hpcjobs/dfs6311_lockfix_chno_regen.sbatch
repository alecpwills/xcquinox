#!/usr/bin/env bash
#SBATCH --job-name=xcq_lockfix_chno
# long-40core, NOT long-96core: the 96-core queue is saturated by the production
# training array, and this job must not queue behind it or compete with it for
# nodes. Two small-molecule CCSD+OEP cascades do not need 96 cores; --exclusive
# with --mem=0 still takes the whole node, so the CCSD memory headroom is
# whatever the 40-core node provides rather than a fixed request.
#SBATCH --partition=long-40core
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=40
#SBATCH --output=/gpfs/scratch/awills/lockfix_chno_regen_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=END,FAIL

# STAGE-AND-VERIFY ONLY -- this job never writes into the live training-reference
# directory. The atomic swap is a separate, seconds-long login-node step
# (hpcjobs/dfs6311_lockfix_swap.py), so the staged files can be inspected first
# while the production sweep keeps reading the old ones.
#
# WHY: the training CCSD references for the degenerate 2-Pi radicals CH and NO
# were generated BEFORE the orientation-lock stamp existed (CH 2026-07-03, NO
# 2026-07-04) and were never regenerated after the lock-blind reference-cache
# fix landed on 2026-07-14. HO was (2026-07-15, stamped 3e-05). So the training
# SCF has been running locked at 3e-05 against two UNLOCKED references: the SCF
# and the reference settle on different members of the degenerate (pi_x, pi_y)
# manifold, producing a density error no functional can close. The consumer
# guard in alec/data.py raises on a MISMATCHED stamp but is structurally blind
# to a MISSING one, which is why this survived a full sweep. Diagnosis:
# notebooks/analysis/DENSITY_DIAGNOSIS.md; probe: dfs6311_lock_stamp_probe.py.
#
# Species selection is automatic, not hard-coded: precompute_all's skip
# predicate (external_refs.py, run_oep_cascade) reads a missing stamp as 0.0
# and re-runs only on a lock/basis mismatch, so passing the production lock
# regenerates exactly the species whose cached lock disagrees and skips the
# already-correct ones. The entries are filtered to CH/NO by name so a stray
# unstamped closed-shell reference cannot pull the whole 214-species pool into
# this job.
#
# Flags mirror the production run's inputs block exactly
# (dfs_step7.dfs6311_grid3_v3.yaml: basis 6-311++G(3df,2pd), grid_level 3,
# density_fit true, auxbasis unset -> auto, orientation_lock_strength 3e-05),
# so the regenerated identity stamps satisfy the same completeness predicate
# the training consumer checks.
#
# Shell discipline (job 2099698 lesson): NO `set -e` -- conda activation can
# return non-zero on success and would kill the job seconds in with an empty
# log. House pattern: -uo pipefail, activation verified by EFFECT (the resolved
# python path), explicit rc gate after every stage.
set -uo pipefail

ENV_PREFIX=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j070
REPO=/gpfs/projects/FernandezGroup/Alec/xcquinox
REFS=/gpfs/scratch/awills/external_refs_dfs_6311ppg3df2pd_g3
STAGE="${1:-${REFS}_lockfix_staging}"

# Start/end stamps: this job only STAGES (the live refs change at swap time, in
# the separate script, which records the spec boundary itself). These timestamps
# bound the regeneration window for the record.
T_START_EPOCH="$(date +%s)"
echo "[lockfix] ==================================================="
echo "[lockfix] START  $(date '+%Y-%m-%d %H:%M:%S %Z')  (epoch $T_START_EPOCH)"
echo "[lockfix] job=${SLURM_JOB_ID:-none}"
echo "[lockfix] ==================================================="

# --- conda activation (verified by effect, not by return code) --------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[lockfix] host=$(hostname) python=$PYBIN"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[lockfix] FATAL: parity env python not active (got: ${PYBIN:-none})"
     exit 1 ;;
esac

# --- thread caps (SLURM vars not assumed present) ---------------------------
THREADS="${SLURM_CPUS_PER_TASK:-40}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export MALLOC_ARENA_MAX=2
ulimit -u unlimited 2>/dev/null || true
echo "[lockfix] BLAS threads: $THREADS  MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX"

echo "[lockfix] live refs (READ-ONLY here): $REFS"
echo "[lockfix] staging out-dir:            $STAGE"
if [ "$STAGE" = "$REFS" ]; then
  echo "[lockfix] FATAL: staging dir must NOT be the live refs dir; the swap is"
  echo "[lockfix]        a separate step (dfs6311_lockfix_swap.py)."
  exit 1
fi
mkdir -p "$STAGE" || { echo "[lockfix] FATAL: cannot create staging dir"; exit 1; }

# --- generation --------------------------------------------------------------
cd "$REPO" || { echo "[lockfix] FATAL: cannot cd $REPO"; exit 1; }
LOCKFIX_STAGE="$STAGE" python - <<'PYEOF'
import os
import sys

from xcquinox.alec.external_refs import build_species_union, precompute_all

TARGETS = {"CH", "NO"}
stage = os.environ["LOCKFIX_STAGE"]

entries = [e for e in build_species_union() if e.name in TARGETS]
got = sorted({e.name for e in entries})
print(f"[lockfix] species resolved from the canonical union: {got}", flush=True)
if set(got) != TARGETS:
    print(f"[lockfix] FATAL: expected {sorted(TARGETS)}, resolved {got}")
    sys.exit(1)
for e in entries:
    print(f"[lockfix]   {e.name}: charge={e.charge} spin={e.spin} "
          f"source={e.source}", flush=True)

# Mirrors cluster/inputs.py's DFS-domain call verbatim except for the species
# subset and the staging cache_dir.
precompute_all(
    entries,
    cache_dir=stage,
    basis="6-311++G(3df,2pd)",
    grid_level=3,
    density_fit=True,
    auxbasis=None,
    orientation_lock_strength=3e-05,
)
print("[lockfix] precompute_all returned cleanly", flush=True)
PYEOF
rc=$?
[ "$rc" -eq 0 ] || { echo "[lockfix] FATAL: precompute_all exited rc=$rc"; exit "$rc"; }

# --- verification ------------------------------------------------------------
# --no-verdict: the probe's verdict diagnoses a LIVE cache. Against staging it
# would report "all stamped -> regeneration will not help", which is trivially
# true of a directory we just regenerated and says nothing about the deployed
# state. The authoritative check here is the explicit gate below.
echo "[lockfix] staged result (expect CH and NO: LOCK KEY PRESENT = 3e-05):"
python "$REPO/hpcjobs/dfs6311_lock_stamp_probe.py" --no-verdict "$STAGE"
rc=$?
[ "$rc" -eq 0 ] || { echo "[lockfix] FATAL: probe exited rc=$rc"; exit "$rc"; }

# Machine-checkable gate: both staged files must carry the production lock.
LOCKFIX_STAGE="$STAGE" python - <<'PYEOF'
import os
import sys

import numpy as np

stage = os.environ["LOCKFIX_STAGE"]
bad = []
for name in ("CH", "NO"):
    path = os.path.join(stage, f"{name}.npz")
    if not os.path.isfile(path):
        bad.append(f"{name}: staged file missing")
        continue
    # our own generated reference cache; allow_pickle only for string stamps
    with np.load(path, allow_pickle=True) as z:
        if "orientation_lock_strength" not in z.files:
            bad.append(f"{name}: still has NO lock stamp")
            continue
        got = float(np.asarray(z["orientation_lock_strength"]).item())
        if f"{got:g}" != f"{3e-05:g}":
            bad.append(f"{name}: lock {got:g}, expected 3e-05")
if bad:
    print("[lockfix] FATAL: staged references failed the gate:")
    for b in bad:
        print(f"[lockfix]   {b}")
    sys.exit(1)
print("[lockfix] GATE PASSED: CH and NO staged with orientation_lock_strength=3e-05")
PYEOF
rc=$?
[ "$rc" -eq 0 ] || { echo "[lockfix] FATAL: staged gate failed rc=$rc"; exit "$rc"; }

T_END_EPOCH="$(date +%s)"
echo "[lockfix] ==================================================="
echo "[lockfix] END    $(date '+%Y-%m-%d %H:%M:%S %Z')  (epoch $T_END_EPOCH)"
echo "[lockfix] elapsed $((T_END_EPOCH - T_START_EPOCH)) s"
echo "[lockfix] STAGED ONLY -- the live references are UNCHANGED, so no spec"
echo "[lockfix] boundary exists yet. The boundary is created by the swap, which"
echo "[lockfix] records it automatically."
echo "[lockfix] ==================================================="
echo "[lockfix] NEXT (login node, seconds):"
echo "[lockfix]   python $REPO/hpcjobs/dfs6311_lockfix_swap.py --stage $STAGE"
exit 0
