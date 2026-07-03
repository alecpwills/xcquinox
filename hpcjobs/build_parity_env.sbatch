#!/bin/bash
#SBATCH --job-name=parity_env_build
#SBATCH --partition=long-40core
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.log
# Build the jax-0.7 parity conda env on a COMPUTE node (the login node throttles/kills
# conda's heavy extraction). Package cache goes on node-local scratch ($TMPDIR) so the
# small-file-heavy extraction never touches GPFS. Submit: sbatch hpcjobs/build_parity_env.sbatch
# Watch:  tail -f parity_env_build_<jobid>.log   (look for "== ENV READY ==")
set -eo pipefail

ENV=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j070
REPO=/gpfs/projects/FernandezGroup/Alec/xcquinox

export PYTHONNOUSERSITE=1
export CONDA_PKGS_DIRS="${TMPDIR:-/tmp}/${USER}_conda_pkgs_${SLURM_JOB_ID}"
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
cd "$REPO"

echo "== build node $(hostname); pkgs cache $CONDA_PKGS_DIRS =="
rm -rf "$ENV" "$CONDA_PKGS_DIRS"
conda env create -p "$ENV" -f "$REPO/environment-cluster-parity.yml"
conda run -p "$ENV" pip install -e "$REPO" --no-deps

echo "== VALIDATE =="
conda run -p "$ENV" pip check
conda run -p "$ENV" python -c "
import jax; jax.config.update('jax_enable_x64', True)
import numpy, scipy, jaxlib, ml_dtypes, pyscf, pyscfad, equinox, optax
print('versions: jax', jax.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__, '| pyscf', pyscf.__version__)
import jax.numpy as jnp
from xcquinox.alec.metagga import compute_alpha
print('metagga compute OK; alpha =', float(compute_alpha(jnp.array([1.0]), jnp.array([0.0]), jnp.array([0.3]))[0]))
from xcquinox.alec.config import get_architecture as g
print('archs ok:', g('deep_mgga_3x16').meta_gga, g('deep_rung35_mgga_3x16').meta_gga)
"
rm -rf "$CONDA_PKGS_DIRS"
echo "== ENV READY =="
