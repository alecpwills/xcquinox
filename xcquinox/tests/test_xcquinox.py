"""
Unit and regression test for the xcquinox package.
"""

# Import package, test suite, and other packages as needed
import sys, os
import jax
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

from ase import Atoms
from pyscfad import dft, scf
from pyscf.dft import UKS as pUKS

import xcquinox as xce

#for some tests, pre-compute the calculations
g_h2 = Atoms('HH', positions=[[ 0.      ,  0.      ,  0.371395],
                            [ 0.      ,  0.      , -0.371395]])

g_name, g_mol = xce.utils.ase_atoms_to_mol(g_h2, basis='def2tzvpd')
print('Doing short RKS PBE calculation for inputs...')
mf_ad = dft.RKS(g_mol, xc='PBE')
mf_ad_e = mf_ad.kernel()
print('Doing short UHF/UKS PBE calculation for inputs...')
mf_uhf = scf.UHF(g_mol)
e_tot = mf_uhf.kernel()
dm = mf_uhf.make_rdm1()
mf_uks = pUKS(g_mol, xc='PBE')
e_uks = mf_uks.kernel()


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


