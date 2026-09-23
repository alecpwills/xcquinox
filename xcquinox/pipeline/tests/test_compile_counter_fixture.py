"""The kernel-cache clearing helper of the compile-counter fixture.

A test that counts compilations wants the kernels it measures cold. Clearing every cache the
process holds (``jax.clear_caches()``) is not the way to get that in a long session: a
complete suite can abort inside that call, in the weak-reference removal of the
thousands of kernels the earlier tests had compiled. ``clear_kernel_caches`` clears the
jitted kernels defined at the top level of the modules it is given and nothing else, and
refuses a module set that holds none, so a misspelt module cannot leave the count warm.
"""
import types

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from xcquinox.pipeline.tests.fixtures.compile_counter import clear_kernel_caches


def _namespace_of_kernels():
    """A module-like namespace with an equinox jit wrapper and a plain jax.jit function,
    each called once so that its cache holds one entry."""
    ns = types.SimpleNamespace(__name__="synthetic_kernels")
    ns.filtered = eqx.filter_jit(lambda x: x * 2.0)
    ns.plain = jax.jit(lambda x: x + 1.0)
    ns.not_a_kernel = 3
    ns.filtered(jnp.ones(2))
    ns.plain(jnp.ones(2))
    return ns


def _cache_size(kernel) -> int:
    target = getattr(kernel, "_cached", kernel)
    return target._cache_size()


def test_clear_kernel_caches_clears_the_named_kernels_only():
    """Both kinds of kernel in the named namespace are cleared; a kernel in another
    namespace keeps its cache; the count returned is the number of kernels cleared."""
    named = _namespace_of_kernels()
    other = _namespace_of_kernels()
    assert _cache_size(named.filtered) == 1 and _cache_size(named.plain) == 1
    assert clear_kernel_caches(named) == 2
    assert _cache_size(named.filtered) == 0 and _cache_size(named.plain) == 0
    assert _cache_size(other.filtered) == 1 and _cache_size(other.plain) == 1


def test_clear_kernel_caches_refuses_a_set_without_kernels():
    """A module set holding no jitted kernel is a caller error, not a silent no-op."""
    with pytest.raises(ValueError):
        clear_kernel_caches(types.SimpleNamespace(__name__="empty", value=1))
