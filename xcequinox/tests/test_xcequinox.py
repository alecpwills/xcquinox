"""
Unit and regression test for the xcequinox package.
"""

# Import package, test suite, and other packages as needed
import sys
import jax
import jax.numpy as jnp
import equinox

import pytest

import xcequinox as xce


def test_xcequinox_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "xcequinox" in sys.modules


def test_net_lob():
    lob = xce.net.LOB(limit=1.804)
    lim = lob(0.5)
    print(f'test_net_lob lim = {lim}')
    assert lim

def test_net_eX():
    eX = xce.net.eX(n_input = 1,
                    n_hidden = 16,
                    depth = 3,
                    use = [],
                    ueg_limit=False,
                    lob = 1.804,
                    seed = 9001)
    inp = 5*jax.random.normal(key=jax.random.PRNGKey(9001), shape=(1, 1, 1))
    result = eX(inp)
    print(f"text_net_eX result = {result}")
    assert result.sum()


def test_net_eC():
    eC = xce.net.eC(n_input = 1,
                    n_hidden = 16,
                    depth = 3,
                    use = [],
                    ueg_limit=False,
                    lob = 1.804,
                    seed = 9001)
    inp = 5*jax.random.normal(key=jax.random.PRNGKey(9001), shape=(10,1))
    result = eC(inp)
    print(f"text_net_eC result = {result}")
    assert result.sum()