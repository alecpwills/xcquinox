#!/usr/bin/env bash
#SBATCH --job-name=xcq_slim16_eval
#SBATCH --partition=extended-40core
#SBATCH --time=96:00:00
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=40
#SBATCH --output=/gpfs/scratch/awills/slim16_eval_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# The Slim16 evaluation array of the GGA-final arm: one task per network of
# the ledger (array 0-3 over the four pretrain entries), chained afterok on
# the prepare wrapper by hpcjobs/gga_final_submit.sh. It carries the same
# resources as hpcjobs/slim16_eval.sbatch and execs that script with the run
# directory prepare recorded, so the evaluation itself -- the converged
# channel over the Slim16 set at def2-TZVP / grid 4 with density fitting, the
# PBE-DF and comparator tables from task 0, the thread caps, the digest mail
# -- is exactly the tracked script's, including its exact-coverage guard
# (array 0-3 over four manifest specs). No arguments: the run directory is
# read from the file the prepare wrapper wrote.
set -uo pipefail

REPO=/gpfs/projects/FernandezGroup/Alec/xcquinox
ARM=/gpfs/scratch/awills/xcquinox_runs/dfs_step8/gga_final
RUNDIR_FILE="$ARM/slim16_rundir.txt"

[ -s "$RUNDIR_FILE" ] || { echo "[gf-eval] FATAL: no run dir recorded at $RUNDIR_FILE (prepare did not run or failed)"; exit 1; }
RUN="$(cat "$RUNDIR_FILE")"
cd "$REPO" || { echo "[gf-eval] FATAL: cannot cd $REPO"; exit 1; }
echo "[gf-eval] exec slim16_eval.sbatch with run dir: $RUN"
exec bash hpcjobs/slim16_eval.sbatch "$RUN"
