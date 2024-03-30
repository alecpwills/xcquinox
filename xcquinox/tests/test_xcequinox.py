"""
Unit and regression test for the xcquinox package.
"""

# Import package, test suite, and other packages as needed
import sys
import jax
import jax.numpy as jnp

import pytest
from ase import Atoms
from pyscfad import dft, scf
from pyscf.dft import UKS as pUKS

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


def test_xc_lda_x():
    ldax = xce.xc.LDA_X()
    rho = 0.5
    eldax = ldax(rho)
    assert eldax

def test_xc_pw_c():
    pwc = xce.xc.PW_C()
    rs = 0.5
    zeta = 0.5
    eldax = pwc(rs, zeta)
    assert eldax

def test_xc_defaults():
    h2 = Atoms('HH', positions=[[ 0.      ,  0.      ,  0.371395],
                                [ 0.      ,  0.      , -0.371395]])
    
    name, mol = xce.utils.ase_atoms_to_mol(h2, basis='def2tzvpd')
    print('Doing short PBE calculation for inputs...')
    mf = dft.RKS(mol, xc='PBE')
    e_tot = mf.kernel()
    dm = mf.make_rdm1()
    ao_eval = jnp.array(mf._numint.eval_ao(mol, mf.grids.coords, deriv=2))

    eX = xce.net.eX(n_input = 2,
                n_hidden = 16,
                depth = 3,
                use = [1,2],
                ueg_limit=True,
                lob = 1.174,
                seed = 9001)
    eC = xce.net.eC(n_input = 4,
                    n_hidden = 16,
                    depth = 3,
                    use = [2,3],
                    ueg_limit=False,
                    lob = 1.804,
                    seed = 9001)
    xc = xce.xc.eXC(grid_models = [eX, eC], level=3)

    exc = xc(dm, ao_eval, mf.grids.weights)
    assert exc

def test_xc_spinpol_defaults():
    h2 = Atoms('HH', positions=[[ 0.      ,  0.      ,  0.371395],
                                [ 0.      ,  0.      , -0.371395]])
    
    name, mol = xce.utils.ase_atoms_to_mol(h2, basis='def2tzvpd')
    print('Doing short PBE calculation for inputs...')
    mf = scf.UHF(mol)
    e_tot = mf.kernel()
    dm = mf.make_rdm1()
    mfuks = pUKS(mol, xc='PBE')
    e_uks = mfuks.kernel()

    ao_eval = jnp.array(mfuks._numint.eval_ao(mol, mfuks.grids.coords, deriv=2))

    eX = xce.net.eX(n_input = 2,
                n_hidden = 16,
                depth = 3,
                use = [1,2],
                ueg_limit=True,
                lob = 1.174,
                seed = 9001)
    eC = xce.net.eC(n_input = 4,
                    n_hidden = 16,
                    depth = 3,
                    use = [2,3],
                    ueg_limit=False,
                    lob = 1.804,
                    seed = 9001)
    xc = xce.xc.eXC(grid_models = [eX, eC], level=3, debug=True)

    exc = xc(dm, ao_eval, mfuks.grids.weights)
    assert exc

