#!/bin/bash
#SBATCH --job-name=xcq_c2_ref_regen
#SBATCH --partition=long-96core
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --output=/gpfs/scratch/awills/c2_ref_regen_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=END,FAIL

# Regenerate ONLY the c2 density-only benchmark reference, into a STAGING
# directory, under the deterministic-grid semantics (run_scf_with_cache now
# stores the first attempt's guess-pruned grid regardless of SCF escalation;
# HISTORY.md Phase 35). The production cache is NOT touched by this job: the
# drifted c2.npz (rho_ref_grid (26840,)) stays in place, so in-flight evals
# keep their current behavior until the verified staged file is swapped in
# with a single atomic rename (command block in the deploy notes). Flags
# mirror the production benchmark_refs job exactly (basis / grid / DF / lock),
# so the regenerated identity stamps satisfy the same completeness predicate.
# Expected outcome in this job's log: the probe block reports the staged c2
# with rho_ref_grid (26568,) and density_fit_used True. Runtime is minutes
# (one small-molecule CCSD); the 4 h limit is slack.
set -euo pipefail

source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate /gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j070
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

REFS=/gpfs/scratch/awills/external_refs_bench_6311ppg3df2pd_g3
STAGE="${REFS}_c2staging"
mkdir -p "$STAGE"

# Resolve c2's index in the sorted 'all' pool at runtime and fail loud if the
# slice would select anything else (the pool ordering is the generator's
# shard/slice contract, sorted by species name).
IDX=$(python - <<'PYEOF'
from xcquinox.alec.benchmark_refs import load_benchmark_species
names = list(load_benchmark_species(
    "all", basis="6-311++G(3df,2pd)", grid_level=3))
i = names.index("c2")
assert names[i] == "c2"
print(i)
PYEOF
)
echo "[c2-regen] species index $IDX (slice $IDX:$((IDX + 1)))"

python -m xcquinox.alec.benchmark_refs \
  --out-dir "$STAGE" \
  --pool all \
  --basis "6-311++G(3df,2pd)" \
  --grid-level 3 --density-fit --orientation-lock-strength 3e-05 \
  --species-slice "$IDX:$((IDX + 1))"

echo "[c2-regen] staged result (expect c2 rho_ref_grid (26568,), density_fit_used True):"
python /gpfs/projects/FernandezGroup/Alec/xcquinox/hpcjobs/dfs6311_c2_ref_probe.py "$STAGE"
