"""Content pins for the single-spec re-evaluation job script.

The job's payload is ``cluster._eval_one_spec``, covered where that module
is; what is pinned here is the script's operational contract: the house
sbatch idiom (``set -uo pipefail``, never errexit), the standing mail
directives, the explicit-index guard plus manifest range check (a bare or
mistyped submit must fail loudly, never mail a green no-op), the
activation-by-effect import probe, and the environment the eval path needs
but does not read from config -- the benchmark-refs dir (without it every
density column is overwritten with null), the malloc arena cap, the BLAS
thread caps, and the reduced-parallelism taskset that avoids re-triggering
the tier degradation being repaired.
"""
from __future__ import annotations

from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent / "reeval_holdout_spec.sbatch"


def _text() -> str:
    return _SCRIPT.read_text()


def test_mail_directives_present():
    t = _text()
    assert "#SBATCH --mail-user=alec.wills@stonybrook.edu" in t
    assert "#SBATCH --mail-type=BEGIN,END,FAIL" in t


def test_house_shell_idiom():
    t = _text()
    assert "set -uo pipefail" in t
    for line in t.splitlines():
        assert not line.strip().startswith("set -e"), line
        assert "errexit" not in line, line


def test_requires_explicit_spec_index_with_range_check():
    t = _text()
    assert 'SPEC_IDX="${REEVAL_SPEC_IDX:-}"' in t
    assert "FATAL: set REEVAL_SPEC_IDX" in t
    assert "outside [0," in t                 # manifest n_specs range check
    assert "n_specs" in t
    # the payload runs the sweep's own eval worker, nothing bespoke
    assert "python -m xcquinox.alec.cluster._eval_one_spec" in t


def test_activation_by_effect_probe():
    t = _text()
    assert 'python -c "import xcquinox.alec.cluster._eval_one_spec"' in t
    assert "FATAL: repo import failed" in t


def test_benchmark_refs_dir_guard():
    # The refs dir resolves only through this env var; running without it
    # nulls every density column of the target spec, in place.
    t = _text()
    assert "XCQUINOX_BENCH_REFS_DIR" in t
    assert "benchmark_refs_dir:" in t         # resolved from the run's config
    assert "refusing to overwrite density columns" in t


def test_memory_and_thread_environment():
    t = _text()
    assert "export MALLOC_ARENA_MAX=2" in t
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert f'export {var}="${{BLAS_THREADS}}"' in t, var
    assert "taskset -c 0-" in t               # reduced tier-1 parallelism
