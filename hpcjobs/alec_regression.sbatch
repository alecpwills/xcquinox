#!/usr/bin/env bash
#SBATCH --job-name=xcq_alec_regression
#SBATCH --partition=long-96core
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=24
#SBATCH --output=/gpfs/scratch/awills/alec_regression_%j.out
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=END,FAIL
# =============================================================================
# Regression suites for the training-path SCF jit seam.
#
# The de-fused training step differentiates the group loss eagerly. An eager
# lax.scan under reverse-mode AD is rebuilt into fresh jaxprs on every call, so
# it recompiles per step and permanently retains two XLA modules each time;
# the process mapping count then climbs to vm.max_map_count (65530) and the job
# dies with SIGSEGV. Both self-consistent loss channels -- density
# (losses._grid_term) and density matrix (losses._dm_term) -- now evaluate their
# SCF inside a module-scope eqx.filter_jit (oneshot.grid_density_for_loss /
# dm_prediction_for_loss), which is applied to the MANUAL backend only: the
# pyscfad driver requires concrete arrays and raises on tracers.
#
# Run on-cluster because the local box is shared. Batches are independent so a
# hang or failure in one does not mask the others; each batch's exit code is
# reported and the job exits non-zero if any batch failed.
# =============================================================================
set -uo pipefail

REPO=/gpfs/projects/FernandezGroup/Alec/xcquinox

source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
conda activate /gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j070
export PYTHONNOUSERSITE=1

# Fail fast if the env cannot even launch pytest -- otherwise every batch
# "fails" in under a second and the job burns a queue slot reporting nothing.
python -m pytest --version >/dev/null 2>&1 || {
  echo "[regression] FATAL: pytest not importable in the activated env"
  exit 99
}

CORES="${SLURM_CPUS_ON_NODE:-$(nproc --all)}"
BT=$(( CORES / 12 )); [ "$BT" -ge 1 ] || BT=1
export OMP_NUM_THREADS="$BT" MKL_NUM_THREADS="$BT" OPENBLAS_NUM_THREADS="$BT"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=1 --xla_llvm_disable_expensive_passes=true --xla_backend_optimization_level=1"
ulimit -u unlimited 2>/dev/null || true
echo "[regression] host=$(hostname) CORES=$CORES BLAS_THREADS=$BT"
echo "[regression] max_map_count=$(cat /proc/sys/vm/max_map_count)"

cd "$REPO" || exit 1

T=xcquinox/alec/tests
declare -a NAMES=(
  "compile-invariant + de-fuse parity + padding"
  "touched modules (losses, oneshot)"
  "solver / SCF"
  "open-shell + pyscfad backend"
)
declare -a BATCHES=(
  "$T/test_compile_ratchet.py $T/test_defused_grad.py $T/test_shape_padding.py"
  "$T/test_losses_step7.py $T/test_oneshot.py"
  "$T/test_solver.py $T/test_solver_manual.py $T/test_solver_manual_df.py $T/test_solver_manual_checkpoint.py $T/test_solver_config_df.py $T/test_scf_backends.py $T/test_scf_diff.py $T/test_scf_tail.py"
  "$T/test_uks_scf.py $T/test_uks_oneshot.py $T/test_uks_vxc_integration.py $T/test_uks_atom_gradflow.py $T/test_oep_uks.py $T/test_pyscfad_gradflow.py"
)

overall=0
for i in "${!BATCHES[@]}"; do
  echo
  echo "===================================================================="
  echo "BATCH $((i+1))/${#BATCHES[@]}: ${NAMES[$i]}"
  echo "===================================================================="
  # No pipe to tail: a pipeline would report tail's exit code, not pytest's.
  python -u -m pytest ${BATCHES[$i]} -q -p no:randomly
  rc=$?
  echo "[regression] batch $((i+1)) exit code: ${rc}"
  [ "$rc" -eq 0 ] || overall=1
done

echo
echo "===================================================================="
if [ "$overall" -eq 0 ]; then
  echo "[regression] ALL BATCHES PASSED"
else
  echo "[regression] AT LEAST ONE BATCH FAILED -- see per-batch exit codes above"
fi
exit ${overall}
