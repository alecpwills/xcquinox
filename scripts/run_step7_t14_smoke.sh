#!/usr/bin/env bash
# Step-7 T14 -- single-spec end-to-end smoke verification (USER-RUN ONLY).
#
# Per the standing project directive (memory/feedback_expensive_test_handling.md),
# this verification is NOT executed by an agent -- wall-clock is 30-90 minutes on
# first run because Cell 0.5 (CCSD pre-compute) dominates. Cached re-runs are
# under 10 minutes.
#
# Run from the repo root:
#     bash scripts/run_step7_t14_smoke.sh 2>&1 | tee /tmp/step7_t14.log
#
# Exit codes:
#   0   smoke passed (notebook executed, Cell 0.5 wrote >=50 npz files,
#       training-eval produced finite AE-MAE + density-RMSE)
#   1   nbconvert failed
#   2   Cell 0.5 outputs missing
#   3   training-eval AE-MAE or density-RMSE is NaN

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=============================================================================="
echo "  Step-7 T14 smoke -- single-spec end-to-end verification"
echo "  Repo: $REPO_ROOT"
echo "  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================================="

# ---- Step 1: wipe stale smoke artifacts ------------------------------------
echo
echo "[1/4] Wiping stale smoke artifacts"
rm -rf notebooks/checkpoints_step7/l2/bin02
rm -f notebooks/gga_training_example-step7.smoke.ipynb
echo "  done."

# ---- Step 2: run the smoke notebook ----------------------------------------
echo
echo "[2/4] Running smoke notebook (jupyter nbconvert)"
echo "       env: STEP7_SMOKE_ONLY=1 (l2, r=2, no-aug, oneshot, 5 steps)"
echo "       expected wall-clock: 30-90 min first run; <10 min cached re-run"
echo "       Cell 0.5 (CCSD pre-compute) is the long pole."
echo
SMOKE_START=$(date +%s)
JUPYTER_CONFIG_DIR=/tmp STEP7_SMOKE_ONLY=1 jupyter nbconvert \
    --to notebook --execute --ExecutePreprocessor.timeout=3600 \
    --output gga_training_example-step7.smoke.ipynb \
    notebooks/gga_training_example-step7.ipynb
NBCONVERT_RC=$?
SMOKE_END=$(date +%s)
ELAPSED=$((SMOKE_END - SMOKE_START))
echo
echo "  nbconvert exit=$NBCONVERT_RC, elapsed=${ELAPSED}s ($(printf '%dh%dm%ds' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))))"
if [[ $NBCONVERT_RC -ne 0 ]]; then
    echo "  !!! nbconvert FAILED -- inspect the smoke notebook for the failing cell"
    exit 1
fi

# ---- Step 3: verify Cell 0.5 outputs ---------------------------------------
echo
echo "[3/4] Verifying Cell 0.5 outputs"
NPZ_COUNT=$(ls notebooks/checkpoints_step7/external_refs/*.npz 2>/dev/null | wc -l)
LOG_COUNT=$(ls notebooks/checkpoints_step7/external_refs/_run_log_*.json 2>/dev/null | wc -l)
echo "  external_refs/*.npz count : $NPZ_COUNT (expect >= 50)"
echo "  archived run log count    : $LOG_COUNT (expect >= 1)"
if [[ $NPZ_COUNT -lt 50 ]]; then
    echo "  !!! external_refs npz count below threshold"
    exit 2
fi
if [[ $LOG_COUNT -lt 1 ]]; then
    echo "  !!! no archived _run_log_*.json"
    exit 2
fi
LATEST_LOG=$(ls -t notebooks/checkpoints_step7/external_refs/_run_log_*.json | head -1)
echo "  latest run log: $LATEST_LOG"
python -c "
import json, sys
log = json.load(open('$LATEST_LOG'))
results = log.get('results', [])
ok = sum(1 for r in results if r['status'] == 'OK')
skipped = sum(1 for r in results if r['status'] == 'SKIPPED_CACHED')
fail = sum(1 for r in results if r['status'].startswith('FAIL'))
print(f'  RunLog: OK={ok}  SKIPPED_CACHED={skipped}  FAIL={fail}  total={len(results)}')
if fail > 0:
    print('  Failures:')
    for r in results:
        if r['status'].startswith('FAIL'):
            err = (r.get('error_msg') or '').splitlines()[-1] if r.get('error_msg') else ''
            print(f\"    {r['name']}: {r['status']} -- {err}\")
    sys.exit(2)
"
RC=$?
[[ $RC -ne 0 ]] && exit $RC

# ---- Step 4: verify training-eval ------------------------------------------
echo
echo "[4/4] Verifying training-eval AE-MAE + density-RMSE"
AGG="notebooks/checkpoints_step7/l2/bin02/deep_combined_attn/L5_gradnorm_vxc_step7/oneshot/eval/aggregate.json"
if [[ ! -f $AGG ]]; then
    echo "  !!! aggregate.json missing at $AGG"
    exit 3
fi
python -c "
import json, math, sys
agg = json.load(open('$AGG'))
mae = agg['metrics']['atomization_energy']['mae_kcalmol']
rmse = agg['metrics']['density_rmse']['mean_density_rmse']
print(f'  AE-MAE       = {mae:.3f} kcal/mol')
print(f'  density-RMSE = {rmse:.3e} e/bohr^3')
if not (mae == mae) or not math.isfinite(mae):
    print('  !!! AE MAE is NaN/inf'); sys.exit(3)
if not (rmse == rmse) or not math.isfinite(rmse):
    print('  !!! density RMSE is NaN/inf -- external_refs not loaded?'); sys.exit(3)
if not (1e-4 <= rmse <= 1.0):
    print(f'  WARN: density-RMSE {rmse:.3e} is outside expected 1e-3 .. 1e-1 range')
print('  All metrics finite.')
"
RC=$?
[[ $RC -ne 0 ]] && exit $RC

echo
echo "=============================================================================="
echo "  T14 SMOKE PASSED."
echo "  Wall-clock: ${ELAPSED}s ($(printf '%dh%dm%ds' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60))))"
echo "  Next: tag the commit:  git tag step7-ccsd-smoke-passed"
echo "=============================================================================="
