#!/bin/bash
#SBATCH --job-name=parity_env_build
#SBATCH --partition=short-96core-shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log
#SBATCH --mail-user=alec.wills@stonybrook.edu
#SBATCH --mail-type=BEGIN,END,FAIL
# Build the parity environment of the current stack on a compute node: conda supplies the
# interpreter (python 3.12), pip installs requirements.txt (the exact releases the
# workstation runs; every one a PyPI wheel on the nodes' glibc 2.34, measured 2026-09-22)
# and the package from this checkout. The node needs network access for conda and pip.
# The package caches go on node-local scratch ($TMPDIR) so the small-file-heavy extraction
# never touches GPFS.
#   Submit: cd $GROUP/Alec/xcquinox && sbatch hpcjobs/build_parity_env.sbatch
#   Watch:  tail -f parity_env_build_<jobid>.log      (ends with "== ENV READY ==")
# The partition is a short milan queue; `sinfo -s -o "%P"` lists what the login node offers.
#
# Shell discipline: no `set -e` (conda activation returns non-zero on success); every step
# is gated on its own exit status, and the activation is verified by its effect.
set -uo pipefail

ENV=/gpfs/projects/FernandezGroup/Alec/conda_envs/xcquinox_j0102
REPO=/gpfs/projects/FernandezGroup/Alec/xcquinox

export PYTHONNOUSERSITE=1
export CONDA_PKGS_DIRS="${TMPDIR:-/tmp}/${USER}_conda_pkgs_${SLURM_JOB_ID:-manual}"
export PIP_CACHE_DIR="${TMPDIR:-/tmp}/${USER}_pip_cache_${SLURM_JOB_ID:-manual}"
source /gpfs/projects/FernandezGroup/Alec/miniconda3/etc/profile.d/conda.sh
cd "$REPO" || exit 1

echo "== build node $(hostname); $(getconf GNU_LIBC_VERSION); caches under ${TMPDIR:-/tmp} =="
rm -rf "$ENV" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
conda create -y -p "$ENV" -c conda-forge --override-channels python=3.12 pip \
  || { echo "[parity-build] FATAL: conda create failed"; exit 1; }

# --- conda activation (verified by effect, not by return code) --------------
conda activate "$ENV" || true
PYBIN="$(command -v python || true)"
echo "[parity-build] host=$(hostname) python=$PYBIN"
case "$PYBIN" in
  "$ENV"/*) : ;;
  *) echo "[parity-build] FATAL: the new environment's python is not active (got: ${PYBIN:-none})"
     exit 1 ;;
esac

echo "== requirements.txt =="
pip install -r "$REPO/requirements.txt" \
  || { echo "[parity-build] FATAL: pip install -r requirements.txt failed"; exit 1; }
pip install -e "$REPO" --no-deps \
  || { echo "[parity-build] FATAL: the package install failed"; exit 1; }

echo "== VALIDATE =="
pip check || { echo "[parity-build] FATAL: pip check failed"; exit 1; }
python -c "
import jax; jax.config.update('jax_enable_x64', True)
import jaxlib, numpy, scipy, ml_dtypes, pyscf, pyscfad, pyscfadlib, equinox, optax, pytest
print('versions: jax', jax.__version__, '| jaxlib', jaxlib.__version__,
      '| numpy', numpy.__version__, '| scipy', scipy.__version__,
      '| pyscf', pyscf.__version__, '| pyscfad', pyscfad.__version__,
      '| pyscfadlib', pyscfadlib.__version__, '| ml_dtypes', ml_dtypes.__version__,
      '| pytest', pytest.__version__)
from pyscfad import gto, dft
mol = gto.Mole(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
mol.build()
mf = dft.RKS(mol)
mf.xc = 'pbe'
print('pyscfad RKS H2/sto-3g PBE:', float(mf.kernel()), 'converged', bool(mf.converged))
import xcquinox
print('xcquinox from', xcquinox.__file__)
" || { echo "[parity-build] FATAL: validation failed"; exit 1; }
rm -rf "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
echo "== ENV READY =="
