"""
Unit and regression test for the xcquinox package.
"""

# Import package, test suite, and other packages as needed
import sys
import jax
import jax.numpy as jnp

import pytest
from ase import Atoms

import xcquinox as xce


def test_xcquinox_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "xcquinox" in sys.modules


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

def test_net_eX_use_nolob_ueg():
    eX = xce.net.eX(n_input = 1,
                    n_hidden = 16,
                    depth = 3,
                    use = [0],
                    ueg_limit=True,
                    lob = 0,
                    seed = 9001)
    inp = 5*jax.random.normal(key=jax.random.PRNGKey(9001), shape=(1, 1, 1))
    result = eX(inp)
    print(f"text_net_eX result = {result}")
    assert result.sum()


def test_net_eC_use_nolob_ueg():
    eC = xce.net.eC(n_input = 1,
                    n_hidden = 16,
                    depth = 3,
                    use = [0, 1, 2],
                    ueg_limit=True,
                    lob = 0,
                    seed = 9001)
    inp = 5*jax.random.normal(key=jax.random.PRNGKey(9001), shape=(3, 1))
    result = eC(inp)
    print(f"text_net_eX result = {result}")
    assert result.sum()


def test_ase_atoms_to_mol():
    h2 = Atoms('HH', positions=[[ 0.      ,  0.      ,  0.371395],
                                [ 0.      ,  0.      , -0.371395]])
    
    name, mol = xce.utils.ase_atoms_to_mol(h2, basis='def2tzvpd')
    assert name == 'H2'
    assert mol.basis == 'def2tzvpd'
