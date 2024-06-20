"""
Unit and regression test for the xcquinox package.
"""

# Import package, test suite, and other packages as needed
import sys, os
import jax, optax
import jax.numpy as jnp
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import pytest
from ase import Atoms
from pyscfad import dft, scf
from pyscf.dft import UKS as pUKS

import xcquinox as xce

#tests for various utility functions throughout xcquinox
def test_make_net():

    xnet, xp = xce.net.make_net('X', 'GGA', 3, 16, savepath='./testmakenet/x', random_seed = 12345)
    cnet, cp = xce.net.make_net('C', 'GGA', 3, 16, savepath='./testmakenet/c', random_seed = 12345)

    #ensure the networks are created and saved
    assert os.path.exists(os.path.join('./testmakenet/x', 'xc.eqx'))
    assert os.path.exists(os.path.join('./testmakenet/c', 'xc.eqx'))
    #ensure the configuration files are created and saved
    assert os.path.exists(os.path.join('./testmakenet/x', 'network.config'))
    assert os.path.exists(os.path.join('./testmakenet/x', 'network.config'))

def test_get_net():
    xnet, xp = xce.net.make_net('X', 'GGA', 3, 16, random_seed = 12345)

    loadx = xce.net.get_net('X', 'GGA', net_path='./testmakenet/x')

    for idx, layer in enumerate(loadx.net.layers):
        assert jnp.sum(layer.weight - xnet.net.layers[idx].weight) == 0
        assert jnp.sum(layer.bias - xnet.net.layers[idx].bias) == 0

def test_make_xcfunc():

    xc = xce.xc.make_xcfunc(level='GGA',
                            x_net_path = './testmakenet/x',
                            c_net_path = './testmakenet/c',
                            savepath = './testmakenet/xc')
    
    assert os.path.exists('./testmakenet/xc/xc.eqx')
    assert os.path.exists('./testmakenet/xc/xnetwork.config.pkl')
    assert os.path.exists('./testmakenet/xc/cnetwork.config.pkl')

def test_get_xcfunc():

    xc = xce.xc.make_xcfunc(level='GGA',
                            x_net_path = './testmakenet/x',
                            c_net_path = './testmakenet/c')
    
    loadxc = xce.xc.get_xcfunc(level='GGA',
                               xc_net_path='./testmakenet/xc')
    
    for idx, gm in enumerate(loadxc.grid_models):
        for lidx, layer in enumerate(gm.net.layers):
            assert jnp.sum(layer.weight - xc.grid_models[idx].net.layers[lidx].weight) == 0
            assert jnp.sum(layer.bias - xc.grid_models[idx].net.layers[lidx].bias) == 0
