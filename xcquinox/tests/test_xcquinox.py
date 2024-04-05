"""
Unit and regression test for the xcquinox package.
"""

# Import package, test suite, and other packages as needed
import sys
import jax, optax
import jax.numpy as jnp

import pytest
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

def test_utils_pad_array():
    #test via array size
    arr1 = jax.random.normal(key=jax.random.PRNGKey(12345),
                             shape=(10,))
    arr2 = jax.random.normal(key=jax.random.PRNGKey(12345),
                             shape=(100,))
    arr3 = xce.utils.pad_array(arr1, arr2)

    #test via shape
    arr4 = xce.utils.pad_array(arr1, arr1, shape=(50,))
    assert arr3.shape == (100,)
    assert arr4.shape == (50,)


def test_utils_pad_array_list():
    arr1 = jax.random.normal(key=jax.random.PRNGKey(12345),
                             shape=(10,))
    arr2 = jax.random.normal(key=jax.random.PRNGKey(12345),
                             shape=(100,))

    newarrs = xce.utils.pad_array_list([arr1, arr2])

    assert newarrs[0].shape == arr2.shape
    assert newarrs[1].shape == arr2.shape

    assert jnp.allclose(jnp.sum(newarrs[0]) ,jnp.sum(arr1))
    assert jnp.allclose(jnp.sum(newarrs[1]), jnp.sum(arr2))

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
    dm = mf_ad.make_rdm1()
    ao_eval = jnp.array(mf_ad._numint.eval_ao(g_mol, mf_ad.grids.coords, deriv=2))

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

    exc = xc(dm, ao_eval, mf_ad.grids.weights)
    assert exc

def test_xc_spinpol_defaults():
    ao_eval = jnp.array(mf_uks._numint.eval_ao(g_mol, mf_uks.grids.coords, deriv=2))

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

    exc = xc(dm, ao_eval, mf_uks.grids.weights)
    assert exc

def test_xc_utils_get_spin():
    h = Atoms('H', positions=[[0,0,0]])
    spin = xce.utils.get_spin(h)
    assert spin == 1
    #try with radical signs, specified spins, and openshell
    h2 = Atoms('HH', positions=[[ 0.      ,  0.      ,  0.371395],
                            [ 0.      ,  0.      , -0.371395]])
    spin = xce.utils.get_spin(h2)
    assert spin == 0
    h2.info['spin'] = 3
    spin = xce.utils.get_spin(h2)
    h2.info['spin'] = None
    assert spin == 3
    h2.info['name'] = 'H2 radical'
    spin = xce.utils.get_spin(h2)
    h2.info['name'] = 'H2'
    assert spin == 1
    h2.info['openshell'] = True
    spin = xce.utils.get_spin(h2)
    assert spin == 2

def test_utils_make_rdm1():
    mrdm = xce.utils.make_rdm1()

    #1D test
    moocc = mf_ad.mo_occ
    mocoe = mf_ad.mo_coeff
    dm = mf_ad.make_rdm1()
    dm2 = mrdm(mocoe, moocc)
    assert jnp.allclose(dm, dm2)

    #2D test
    moocc = mf_uks.mo_occ
    mocoe = mf_uks.mo_coeff
    dm = mf_uks.make_rdm1()
    dm2 = mrdm(mocoe, moocc)
    assert jnp.allclose(dm, dm2)

def test_utils_get_rho():
    gr = xce.utils.get_rho()
    ao_eval = mf_ad._numint.eval_ao(g_mol, mf_ad.grids.coords, deriv=2)
    #1D rho
    dm1d = mf_ad.make_rdm1()
    #choose the first column for just rho, no derivs
    rho1d = gr(dm1d, ao_eval[0])
    nelec1 = jnp.sum(rho1d*mf_ad.grids.weights)
    #assert that the integrate electron numbers are equal to the amount in nelec
    nelec = mf_ad.mol.nelec

    assert jnp.allclose(jnp.array(nelec).sum(), nelec1)
    
    #2D rho
    dm2d = mf_uks.make_rdm1()
    rho2d = gr(dm2d, ao_eval[0])
    nelec = mf_uks.nelec
    nelec1 = jnp.sum(rho2d*mf_ad.grids.weights, axis=1)

    assert jnp.allclose(jnp.array(nelec), nelec1)

def test_utils_energy_tot():

    dm = mf_uks.make_rdm1()
    hcore = mf_uks.get_hcore()
    veff = jnp.array(mf_uks.get_veff())

    ef = xce.utils.energy_tot()
    e_tot = ef(dm, hcore, veff)
    assert e_tot

def test_utils_get_veff():
    eri = g_mol.intor('int2e')
    dm = mf_uks.make_rdm1()
    veff = xce.utils.get_veff()(dm, eri)

    assert (jnp.sum(veff))

def test_utils_get_fock():
    
    hc = mf_uks.get_hcore()
    veff = jnp.array(mf_uks.get_veff())
    fock = xce.utils.get_fock()(hc, veff)

    assert jnp.sum(fock)

def test_utils_get_hcore():

    hcore = mf_uks.get_hcore()
    v = g_mol.intor('int1e_nuc')
    t = g_mol.intor('int1e_kin')
    hcore1 = xce.utils.get_hcore()(v, t)

    assert jnp.allclose(hcore, hcore1)

def test_utils_eig():

    h = mf_uks.get_hcore()
    s_inv = jnp.linalg.inv(jnp.linalg.cholesky(g_mol.intor('int1e_ovlp')))

    e, c = xce.utils.eig()(h, s_inv, h.shape)

    assert(jnp.sum(e))
    assert(jnp.sum(c))

def test_train_e_loss():
    #network to train
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
    #loss function to use
    Eloss = xce.loss.E_loss()

    #trainer to do training
    ao_eval = jnp.array(mf_ad._numint.eval_ao(g_mol, mf_ad.grids.coords, deriv=2))
    first_E = xc(dm, ao_eval, mf_ad.grids.weights)
    xcT = xce.train.xcTrainer(xc, optax.adamw(1e-4), Eloss, steps=10)
    new_model = xcT(1, xc, [dm], [mf_ad_e], [ao_eval], [mf_ad.grids.weights])
    new_E = new_model(dm, ao_eval, mf_ad.grids.weights)
    assert abs(new_E-mf_ad_e) < abs(first_E - mf_ad_e)

