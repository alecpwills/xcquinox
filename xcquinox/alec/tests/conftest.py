"""Shared pytest configuration for xcquinox.alec tests.

Enforces the two process-level items of THE SPEC §13.3 principle #2
(Deterministic): `jax_enable_x64` and CPU-only execution. The third
item — fixed seeds — is enforced per-test by hardcoded PRNG key values.
"""
import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def _configure_jax_for_tests():
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
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
