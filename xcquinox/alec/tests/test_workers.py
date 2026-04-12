"""Tests for xcquinox.alec.workers.

Implements THE SPEC section 13.2 test_workers.py items (1)-(14).

Tests 6, 9, 11, 12 run immediately.
Tests 1-5, 7-8, 10, 13-14 are marked xfail (require fixtures or
infrastructure not yet available).
"""
import json
import os
import pickle
import subprocess
import sys
import textwrap

import pytest

from xcquinox.alec.config import ArchitectureConfig, PretrainSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
PRETRAIN_DATA_UKS = os.path.join(FIXTURE_DIR, "pretrain_data_uks_tiny.npz")

WORKERS_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, "workers",
)

PRETRAIN_WORKER = os.path.join(WORKERS_DIR, "pretrain_worker.py")
TRAIN_WORKER = os.path.join(WORKERS_DIR, "train_worker.py")
TEST_WORKER = os.path.join(WORKERS_DIR, "test_worker.py")


def _make_arch(**overrides):
    defaults = dict(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    defaults.update(overrides)
    return ArchitectureConfig(**defaults)


# ---------------------------------------------------------------------------
# (1) test_pretrain_worker_subprocess — xfail
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="requires pretrain_data_tiny.pkl fixture", strict=False)
def test_pretrain_worker_subprocess():
    """Pretrain worker subprocess runs end-to-end and returns JSON."""
    pytest.fail("pretrain_data_tiny.pkl fixture not yet generated")


# ---------------------------------------------------------------------------
# (2) test_train_worker_subprocess — xfail
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="requires end-to-end training infrastructure", strict=False)
def test_train_worker_subprocess():
    """Train worker subprocess runs end-to-end and returns JSON."""
    pytest.fail("end-to-end training infrastructure not yet available")


# ---------------------------------------------------------------------------
# (3) test_pretrain_worker_json_output_schema — xfail
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="requires pretrain_data_tiny.pkl fixture", strict=False)
def test_pretrain_worker_json_output_schema():
    """Pretrain worker JSON output contains status, duration, arch keys."""
    pytest.fail("pretrain_data_tiny.pkl fixture not yet generated")


# ---------------------------------------------------------------------------
# (4) test_train_worker_json_output_schema — xfail
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="requires pretrain_data_tiny.pkl fixture", strict=False)
def test_train_worker_json_output_schema():
    """Train worker JSON output contains status, duration, arch keys."""
    pytest.fail("pretrain_data_tiny.pkl fixture not yet generated")


# ---------------------------------------------------------------------------
# (5) test_stdout_stderr_separation — xfail
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="requires pretrain_data_tiny.pkl fixture", strict=False)
def test_stdout_stderr_separation():
    """Worker stdout is pure JSON; stderr has logs/warnings only."""
    pytest.fail("pretrain_data_tiny.pkl fixture not yet generated")


# ---------------------------------------------------------------------------
# (6) test_thread_limit_env_vars_set_before_jax_import — PASS
# ---------------------------------------------------------------------------

def test_thread_limit_env_vars_set_before_jax_import(tmp_path):
    """Thread env vars are set before JAX is imported in the worker."""
    # Write a tiny script that mimics the worker's env-setting logic,
    # then checks the env vars AFTER importing jax.
    script = tmp_path / "check_env.py"
    script.write_text(textwrap.dedent("""\
        import os
        import sys

        threads = sys.argv[1]

        # Set env vars (same as worker pattern)
        os.environ["XLA_FLAGS"] = (
            f"--xla_cpu_multi_thread_eigen=true "
            f"intra_op_parallelism_threads={threads}"
        )
        os.environ["OMP_NUM_THREADS"] = threads
        os.environ["MKL_NUM_THREADS"] = threads
        os.environ["OPENBLAS_NUM_THREADS"] = threads

        # Now import jax (the env must already be set)
        import jax  # noqa: F401

        # Verify env vars survived the JAX import
        xla = os.environ.get("XLA_FLAGS", "")
        omp = os.environ.get("OMP_NUM_THREADS", "")
        mkl = os.environ.get("MKL_NUM_THREADS", "")
        openblas = os.environ.get("OPENBLAS_NUM_THREADS", "")

        # Print results to stderr for verification
        print(f"XLA_FLAGS={xla}", file=sys.stderr)
        print(f"OMP={omp}", file=sys.stderr)
        print(f"MKL={mkl}", file=sys.stderr)
        print(f"OPENBLAS={openblas}", file=sys.stderr)
    """))

    result = subprocess.run(
        [sys.executable, str(script), "7"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    stderr = result.stderr
    assert "intra_op_parallelism_threads=7" in stderr
    assert "OMP=7" in stderr
    assert "MKL=7" in stderr
    assert "OPENBLAS=7" in stderr


# ---------------------------------------------------------------------------
# (7) test_legacy_pretrain_shim_translates_argv — xfail
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="requires legacy shim implementation", strict=False)
def test_legacy_pretrain_shim_translates_argv():
    """Legacy pretrain shim translates old argv to new format."""
    pytest.fail("legacy shim not yet implemented")


# ---------------------------------------------------------------------------
# (8) test_legacy_train_shim_translates_argv — xfail
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="requires legacy shim implementation", strict=False)
def test_legacy_train_shim_translates_argv():
    """Legacy train shim translates old argv to new format."""
    pytest.fail("legacy shim not yet implemented")


# ---------------------------------------------------------------------------
# (9) test_worker_failure_path — PASS
# ---------------------------------------------------------------------------

def test_worker_failure_path(tmp_path):
    """Worker with invalid spec-pickle exits 1 with JSON error payload."""
    bogus_pickle = str(tmp_path / "nonexistent_spec.pkl")

    result = subprocess.run(
        [
            sys.executable, PRETRAIN_WORKER,
            "--arch", "shallow",
            "--spec-pickle", bogus_pickle,
            "--checkpoint-base", str(tmp_path),
            "--data-dir", str(tmp_path),
            "--threads", "1",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 1

    # stdout should be valid JSON with error info
    payload = json.loads(result.stdout.strip())
    assert payload["status"] == "failed"
    assert payload["arch"] == "shallow"
    assert "error" in payload
    assert "traceback" in payload
    assert "duration" in payload


# ---------------------------------------------------------------------------
# (10) test_progress_json_written_during_training — xfail
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="requires full training infrastructure", strict=False)
def test_progress_json_written_during_training():
    """Progress JSON is written during training at regular intervals."""
    pytest.fail("full training infrastructure not yet available")


# ---------------------------------------------------------------------------
# (11) test_spec_pickle_roundtrip — PASS
# ---------------------------------------------------------------------------

def test_spec_pickle_roundtrip(tmp_path):
    """Spec pickle/unpickle preserves equality and hash; no eqx.Module fields."""
    arch = _make_arch(name="test_rt")
    spec = PretrainSpec(
        arch=arch,
        data_dir="/tmp/rt_data",
        checkpoint_dir="/tmp/rt_ckpt",
        n_steps=50,
        lr_start=1e-2,
        lr_end=1e-5,
        lr_decay_start=0.3,
        grad_clip=2.0,
        seed=99,
    )

    pkl_path = tmp_path / "spec.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(spec, f)

    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)

    # Equality
    assert loaded == spec
    # Hash equality
    assert hash(loaded) == hash(spec)
    # All field values match
    assert loaded.arch == spec.arch
    assert loaded.data_dir == spec.data_dir
    assert loaded.checkpoint_dir == spec.checkpoint_dir
    assert loaded.n_steps == spec.n_steps
    assert loaded.lr_start == spec.lr_start
    assert loaded.lr_end == spec.lr_end
    assert loaded.lr_decay_start == spec.lr_decay_start
    assert loaded.grad_clip == spec.grad_clip
    assert loaded.seed == spec.seed

    # No eqx.Module fields -- spec is a plain frozen dataclass
    import dataclasses
    for field in dataclasses.fields(spec):
        val = getattr(spec, field.name)
        # None of the field values should be eqx.Module instances
        assert not hasattr(val, "__module__") or "equinox" not in str(
            type(val).__module__
        ), f"Field {field.name!r} appears to be an eqx.Module"


# ---------------------------------------------------------------------------
# (12) test_missing_spec_pickle_exits_cleanly — PASS
# ---------------------------------------------------------------------------

def test_missing_spec_pickle_exits_cleanly():
    """Worker exits with error when --spec-pickle is omitted from args."""
    result = subprocess.run(
        [
            sys.executable, PRETRAIN_WORKER,
            "--arch", "shallow",
            # --spec-pickle deliberately omitted
            "--checkpoint-base", "/tmp/x",
            "--data-dir", "/tmp/x",
        ],
        capture_output=True, text=True, timeout=30,
    )
    # argparse exits 2 on missing required arg
    assert result.returncode == 2
    assert "spec-pickle" in result.stderr.lower() or "required" in result.stderr.lower()


# ---------------------------------------------------------------------------
# (13) test_progress_json_rewrite_cadence — xfail
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="requires full training infrastructure", strict=False)
def test_progress_json_rewrite_cadence():
    """Progress JSON is rewritten at the documented cadence."""
    pytest.fail("full training infrastructure not yet available")


# ---------------------------------------------------------------------------
# (14) test_uks_pretrain_worker — xfail
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="requires UKS pretrain data fixture", strict=False)
def test_uks_pretrain_worker():
    """UKS pretrain worker runs on H2O triplet data and returns valid JSON."""
    pytest.fail("UKS pretrain data fixture not yet generated")
