#!/usr/bin/env bash
# =============================================================================
# Submit the dfs6311_grid3_v4 descriptor-ablation sweep (77 cells) plus the
# production V_xc verification job. Run ON A MILAN LOGIN NODE from the cluster
# repo root:
#
#     bash hpcjobs/submit_dfs6311_v4.sh [partition]
#
# partition defaults to long-96core (~256 GB nodes -- REQUIRED: measured peak
# training RSS is 142.8 GB; a 128 GB node OOMs).
#
# What it does, in order:
#   1. Backs up the shared pretrain-data npz + manifest ONCE (no-clobber): the
#      datagen stage regenerates the file in place (it predates the
#      rung35ms_all and *_mesh columns), and the backup preserves v3's exact
#      input bytes for provenance. -n is load-bearing: a re-run of this script
#      after the regeneration must NOT overwrite the original backup with the
#      regenerated file.
#   2. Submits the sweep graph (pretrain -> preflight -> 77-task train array
#      -> eval) with the 48 h wall baked into the YAML.
#   3. Submits the nan_verify job (4 archs, cycle caps 3..25) so the
#      production-scale corrected-V_xc measurement of the two previously
#      unmeasured rung-3.5 archs -- and the 3-cycle point the sweep actually
#      trains at -- rides alongside.
#
# Shell discipline (job 2099698 lesson): NO set -e; explicit rc gates.
# =============================================================================
set -uo pipefail

PARTITION="${1:-long-96core}"
DATA_DIR=/gpfs/scratch/awills/pretrain_data_dfs_6311ppg3df2pd_g3_allelem
NPZ="$DATA_DIR/pretrain_data_polarized.npz"

echo "[submit-v4] partition=$PARTITION"

# --- 1. one-time provenance backup (no-clobber) ------------------------------
for f in "$NPZ" "$NPZ.manifest.json"; do
  if [ -f "$f" ]; then
    if cp -n "$f" "$f.v3bak" 2>/dev/null; then
      echo "[submit-v4] backed up $(basename "$f") -> $(basename "$f").v3bak"
    else
      echo "[submit-v4] backup exists, left untouched: $(basename "$f").v3bak"
    fi
  else
    echo "[submit-v4] NOTE: $f absent (fresh datagen will create it)"
  fi
done

# --- 2. the sweep graph ------------------------------------------------------
python -m xcquinox.alec.cluster submit \
    hpcjobs/configs/dfs_step7.dfs6311_grid3_v4.yaml --submit \
    --partition "$PARTITION" --max-nodes 3
RC=$?
if [ "$RC" -ne 0 ]; then
  echo "[submit-v4] FATAL: sweep submission failed (rc=$RC); nan_verify NOT submitted"
  exit "$RC"
fi

# --- 3. the production verification job --------------------------------------
sbatch hpcjobs/dfs6311_nan_verify.sbatch
RC2=$?
if [ "$RC2" -ne 0 ]; then
  echo "[submit-v4] WARNING: nan_verify submission failed (rc=$RC2); the sweep"
  echo "[submit-v4] is unaffected -- submit it manually when convenient."
fi

echo "[submit-v4] done. After the pretrain stage completes, validate with:"
echo "[submit-v4]   python -m xcquinox.alec.cluster.validate_run <run_dir printed above>"
exit 0
