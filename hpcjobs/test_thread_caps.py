"""Every standalone job script that drives PySCF in-process caps the two pools
that serve it -- PySCF's OpenMP pool and numpy's OpenBLAS pool -- at
``parallel.PYSCF_POOL_THREADS_MAX`` from the allocation, by the shell rule the
stage templates carry. Sized to the allocation, the spin-waiting pools stall a
PySCF SCF loop (workflow-matrix job 2134488: about ten minutes per molecule at
40 threads on a 40-core node against 8 s at four). The rule is evaluated under
bash for each script, against the module's own function.
"""
import os
import subprocess
from pathlib import Path

import pytest

from xcquinox.alec.parallel import PYSCF_POOL_THREADS_MAX, pyscf_pool_threads

HPCJOBS = Path(__file__).resolve().parent
#: Standalone scripts whose Python runs PySCF in the job's own process.
CAPPED_SCRIPTS = (
    "probe_pretrain_energy_weight.sbatch",
    "dfs6311_scan_pool.sbatch",
    "dfs6311_lockfix_chno_regen.sbatch",
    "dfs6311_pretrained_holdout.sbatch",
    "nonempirical_pool.sbatch",
    "dfs6311_c2_ref_regen.sbatch",
)


def test_no_job_script_sizes_the_pools_from_the_allocation_uncapped():
    """A script that exports the allocation itself to the PySCF-serving
    pools is the regime measured in job 2134488. The one permitted
    allocation-sized THREADS is the workflow matrix driver's, which does no
    numeric work itself and builds every stage's environment at
    parallel.pyscf_pool_threads."""
    import re
    sized = re.compile(r'THREADS="?\$\{?SLURM_CPUS_PER_TASK(?::-(\d+))?\}?"?')
    for path in sorted(HPCJOBS.glob("*.sbatch")):
        text = path.read_text()
        if path.name == "workflow_matrix.sbatch":
            continue
        for m in sized.finditer(text):
            # THREADS="${SLURM_CPUS_PER_TASK:-N}" and the no-default idioms
            # export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" / =${SLURM_CPUS_PER_TASK}
            fallback = int(m.group(1)) if m.group(1) else None
            assert fallback is not None and fallback <= PYSCF_POOL_THREADS_MAX, (
                f"{path.name} sizes a thread pool from the allocation "
                f"({m.group(0)}): cap it like the scripts in CAPPED_SCRIPTS")
            assert path.name in CAPPED_SCRIPTS, path.name


def _cap_block(text):
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines)
                 if l.startswith('THREADS="${SLURM_CPUS_PER_TASK'))
    end = next(i for i, l in enumerate(lines)
               if l.startswith("export ") and "OPENBLAS_NUM_THREADS=" in l)
    return "\n".join(lines[start:end + 1])


def _pools(snippet, env):
    out = subprocess.run(
        ["bash", "-uo", "pipefail", "-c",
         snippet + '\necho "$OMP_NUM_THREADS $MKL_NUM_THREADS $OPENBLAS_NUM_THREADS"'],
        env={"PATH": os.environ.get("PATH", ""), **env},
        capture_output=True, text=True, check=True)
    assert out.stderr == "", out.stderr
    return out.stdout.split()


@pytest.mark.parametrize("script", CAPPED_SCRIPTS)
def test_the_scripts_pools_follow_the_module_rule(script):
    text = (HPCJOBS / script).read_text()
    assert f'THREADS="${{SLURM_CPUS_PER_TASK:-{PYSCF_POOL_THREADS_MAX}}}"' in text
    assert 'THREADS="${SLURM_CPUS_PER_TASK:-40}"' not in text
    snippet = _cap_block(text)
    for n in (0, 1, 4, 8, 9, 28, 40, 96):
        assert _pools(snippet, {"SLURM_CPUS_PER_TASK": str(n)}) == \
            [str(pyscf_pool_threads(n))] * 3, (script, n)
    for text_n, expect in (("04", 4), ("040", 8), ("", 8), ("abc", 8)):
        assert _pools(snippet, {"SLURM_CPUS_PER_TASK": text_n}) == \
            [str(expect)] * 3, (script, text_n)
    assert _pools(snippet, {}) == [str(PYSCF_POOL_THREADS_MAX)] * 3, script


@pytest.mark.parametrize("script", CAPPED_SCRIPTS)
def test_the_scripts_log_line_states_the_allocation_and_the_cap(script):
    """A capped 40-CPU run must be distinguishable from an 8-CPU allocation
    in the job's own log."""
    text = (HPCJOBS / script).read_text()
    assert "allocation=${SLURM_CPUS_PER_TASK:-unset} -> OMP/MKL/OPENBLAS=" in text
