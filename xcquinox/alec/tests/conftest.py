"""Shared pytest configuration for xcquinox.alec tests.

Enforces the two process-level items of THE SPEC §13.3 principle #2
(Deterministic): `jax_enable_x64` and CPU-only execution. The third
item, fixed seeds, is enforced per-test by hardcoded PRNG key values.
"""
import os
import sys

# These env vars MUST be set before JAX is imported (which conftest module
# load is the earliest hook pytest provides). Setting them in a session
# fixture is too late: jax has already initialized its backend.
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest


def pytest_sessionfinish(session, exitstatus):
    """Record the session status for the hard-exit hook below."""
    session.config._xcq_session_exitstatus = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Exit the process before interpreter teardown.

    JAX's atexit backend cleanup can corrupt the heap during interpreter
    shutdown ("corrupted size vs. prev_size", SIGABRT, exit code 134) AFTER
    the terminal summary has been printed, so a fully green session can still
    hand a non-zero code to whatever launched it -- the cluster regression
    batches read that code and flip to a false FAILED (job 2091615 batch 2:
    "67 passed, 2 xfailed" followed by exit 134). Exiting via os._exit here
    skips interpreter teardown entirely while preserving the real pytest
    status. trylast runs this after every other plugin's unconfigure
    (pytest-cov finishes coverage and writes its reports inside its
    pytest_runtestloop hookwrapper, earlier still); stdio is flushed
    explicitly because os._exit does not flush buffers.
    Set XCQ_NO_HARD_EXIT=1 to restore stock interpreter teardown.
    """
    status = getattr(config, "_xcq_session_exitstatus", None)
    if status is None or os.environ.get("XCQ_NO_HARD_EXIT"):
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


@pytest.fixture(scope="session", autouse=True)
def _configure_jax_for_tests():
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    yield


@pytest.fixture(autouse=True)
def _clear_precompute_cache():
    """Wipe the process-level precompute cache before each test so that
    tests calling ``precompute_fixed_density_data`` are isolated. Without
    this, a test that patches PySCF internals to trigger an error path
    (e.g. ill-conditioned overlap) is short-circuited by a cached result
    from an earlier test on the same MoleculeSpec.
    """
    from xcquinox.alec.data import clear_precompute_cache
    clear_precompute_cache()
    yield
    clear_precompute_cache()
