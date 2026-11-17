#!/usr/bin/env bash
# The GGA-final arm's one-command submission, run from the cluster checkout
# after the arm's merge has been pulled in:
#
#     bash hpcjobs/gga_final_submit.sh
#
# It submits the arm's pretraining suite (hpcjobs/configs/
# dfs_step8.gga_final_pretrain.yaml; --stages pretrain renders datagen ->
# pretrain array 0-3 with the internal afterok), writes the Slim16 ledger of
# the four pretrained networks (hpcjobs/gga_final_slim16_ledger.py), and
# chains the Slim16 readout on it with no idle gap: the prepare wrapper
# (gga_final_slim16_prepare.sbatch) afterok on the whole pretrain array, the
# evaluation array (gga_final_slim16_eval.sbatch, 0-3) afterok on prepare.
# Everything is submitted from the login node in this one pass.
#
# A re-run submits a fresh run directory and rewrites the ledger; the
# previously chained jobs are untouched (SLURM dependencies already bind
# them to their own run directory).
#
# Shell discipline: no `set -e` -- conda activation can return non-zero on
# success; -u and pipefail with explicit rc gates on every step.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

ENV_PREFIX=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j0102
ARM=/gpfs/scratch/awills/xcquinox_runs/dfs_step8/gga_final
CONFIG=hpcjobs/configs/dfs_step8.gga_final_pretrain.yaml
LOG="$ARM/submit.log"

# --- conda activation (verified by effect, not by return code) --------------
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX" || true
export PYTHONNOUSERSITE=1
PYBIN="$(command -v python || true)"
echo "[gf-submit] host=$(hostname) python=$PYBIN"
case "$PYBIN" in
  "$ENV_PREFIX"/*) : ;;
  *) echo "[gf-submit] FATAL: parity env python not active (got: ${PYBIN:-none})"
     exit 1 ;;
esac

mkdir -p "$ARM"

# --- 1) the pretraining suite: datagen -> pretrain array 0-3 ----------------
echo "[gf-submit] submitting $CONFIG (stages=pretrain, partition=extended-96core)"
python -m xcquinox.pipeline.cluster submit "$CONFIG" \
  --stages pretrain --partition extended-96core --submit > "$LOG" 2>&1
RC=$?
grep -E 'created run dir|SUBMITTED|run dir =' "$LOG"
[ "$RC" -eq 0 ] || { echo "[gf-submit] FATAL: submit rc=$RC (log: $LOG)"; exit "$RC"; }
RUN_DIR="$(grep 'run dir = ' "$LOG" | tail -1 | sed 's/.*run dir = //')"
PRETRAIN_ID="$(grep -oE 'pretrain=[0-9]+' "$LOG" | head -1 | cut -d= -f2)"
[ -n "$RUN_DIR" ] || { echo "[gf-submit] FATAL: no run dir in $LOG"; exit 1; }
[ -n "$PRETRAIN_ID" ] || { echo "[gf-submit] FATAL: no pretrain id in $LOG"; exit 1; }
echo "[gf-submit] run dir: $RUN_DIR"
echo "[gf-submit] pretrain array: $PRETRAIN_ID"

# --- 2) the Slim16 ledger of the four networks ------------------------------
python3 hpcjobs/gga_final_slim16_ledger.py "$RUN_DIR" "$ARM/slim16_ledger.json"
[ $? -eq 0 ] || { echo "[gf-submit] FATAL: ledger write failed"; exit 1; }

# --- 3) prepare, afterok on the whole pretrain array ------------------------
# --parsable: sbatch otherwise prints "Submitted batch job <id>", and the
# sentence (not the id) would land in the next call's --dependency
PREP_ID="$(sbatch --parsable --dependency=afterok:"$PRETRAIN_ID" hpcjobs/gga_final_slim16_prepare.sbatch)"
[ -n "$PREP_ID" ] || { echo "[gf-submit] FATAL: prepare sbatch returned nothing"; exit 1; }
echo "[gf-submit] prepare: $PREP_ID (afterok:$PRETRAIN_ID)"

# --- 4) the evaluation array, afterok on prepare ----------------------------
EVAL_ID="$(sbatch --parsable --dependency=afterok:"$PREP_ID" hpcjobs/gga_final_slim16_eval.sbatch)"
[ -n "$EVAL_ID" ] || { echo "[gf-submit] FATAL: eval sbatch returned nothing"; exit 1; }
echo "[gf-submit] eval array: $EVAL_ID (afterok:$PREP_ID)"

echo "[gf-submit] CHAIN pretrain=$PRETRAIN_ID prepare=$PREP_ID eval=$EVAL_ID"
echo "[gf-submit] logs: submit $LOG; prepare /gpfs/scratch/awills/gf_prepare_%j.out; eval /gpfs/scratch/awills/slim16_eval_%j.out"
