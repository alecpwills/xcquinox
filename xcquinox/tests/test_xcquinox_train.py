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

energies = []
dms = []
ao_evals = []
gws = []
eris = []
mo_occs = []
hcs = []
vs = []
ts = []
ss = []
hologaps = []
ogds = []

#for some tests, pre-compute the calculations
g_h2 = Atoms('HH', positions=[[ 0.      ,  0.      ,  0.371395],
                            [ 0.      ,  0.      , -0.371395]])

name, mol = xce.utils.ase_atoms_to_mol(g_h2, basis='def2tzvpd')
mol.build()
mf = dft.RKS(mol, xc='PBE')
e_tot = mf.kernel()
dm = mf.make_rdm1()
ao_eval = jnp.array(mf._numint.eval_ao(mol, mf.grids.coords, deriv=2))
energies.append(mf.get_veff().exc)
dms.append(dm)
ogds.append(dm.shape)
ao_evals.append(ao_eval)
gws.append(mf.grids.weights)
ts.append(mol.intor('int1e_kin'))
vs.append(mol.intor('int1e_nuc'))
mo_occs.append(mf.mo_occ)
hcs.append(mf.get_hcore())
eris.append(mol.intor('int2e'))
ss.append(jnp.linalg.inv(jnp.linalg.cholesky(mol.intor('int1e_ovlp'))))
hologaps.append(mf.mo_energy[mf.mo_occ == 0][0] - mf.mo_energy[mf.mo_occ > 1][-1])

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


def test_train_e_loss():
    #loss function to use
    Eloss = xce.loss.E_loss()

    #trainer to do training
    first_E = xc(dms[0], ao_evals[0], gws[0])
    xcT = xce.train.xcTrainer(xc, optax.adamw(1e-4), Eloss, steps=10)
    new_model = xcT(1, xc, dms, energies, ao_evals, gws)
    new_E = new_model(dms[0], ao_evals[0], gws[0])
    assert abs(new_E-energies[0]) < abs(first_E - energies[0])

def test_train_dm_loss():
    DMloss = xce.loss.DM_HoLu_loss()
    xcT = xce.train.xcTrainer(xc, optax.adamw(1e-4), DMloss, steps=3)
    init_loss = DMloss(xc, ao_evals[0], gws[0], dms[0], eris[0], mo_occs[0], hcs[0], ss[0], ogds[0], hologaps[0], alpha0=0.7,
                       dmL = 1.0, holuL = 0.0, dm_to_rho=0.0)

    #this test specifically for RMSE DM loss
    new_model = xcT(1, xc, ao_evals, gws, dms, eris, mo_occs, hcs, ss, ogds, hologaps, [0.7], [0.0], [0.0], [1.0])
    new_loss = DMloss(new_model, ao_evals[0], gws[0], dms[0], eris[0], mo_occs[0], hcs[0], ss[0], ogds[0], hologaps[0], alpha0=0.7,
                    dmL = 1.0, holuL = 0.0, dm_to_rho=0.0)

    assert ( abs(new_loss) < abs(init_loss) )

def test_train_holo_loss():
    DMloss = xce.loss.DM_HoLu_loss()
    xcT = xce.train.xcTrainer(xc, optax.adamw(1e-4), DMloss, steps=3)
    init_loss = DMloss(xc, ao_evals[0], gws[0], dms[0], eris[0], mo_occs[0], hcs[0], ss[0], ogds[0], hologaps[0], alpha0=0.7,
                       dmL = 0.0, holuL = 1.0, dm_to_rho=0.0)

    #this test specifically for RMSE DM loss
    new_model = xcT(1, xc, ao_evals, gws, dms, eris, mo_occs, hcs, ss, ogds, hologaps, [0.7], [0.0], [0.0], [1.0])
    new_loss = DMloss(new_model, ao_evals[0], gws[0], dms[0], eris[0], mo_occs[0], hcs[0], ss[0], ogds[0], hologaps[0], alpha0=0.7,
                    dmL = 0.0, holuL = 1.0, dm_to_rho=0.0)

    assert ( abs(new_loss) < abs(init_loss) )

def test_train_rho_loss():
    DMloss = xce.loss.DM_HoLu_loss()
    xcT = xce.train.xcTrainer(xc, optax.adamw(1e-4), DMloss, steps=3)
    init_loss = DMloss(xc, ao_evals[0], gws[0], dms[0], eris[0], mo_occs[0], hcs[0], ss[0], ogds[0], hologaps[0], alpha0=0.7,
                       dmL = 0.0, holuL = 0.0, dm_to_rho=1.0)

    #this test specifically for RMSE DM loss
    new_model = xcT(1, xc, ao_evals, gws, dms, eris, mo_occs, hcs, ss, ogds, hologaps, [0.7], [0.0], [0.0], [1.0])
    new_loss = DMloss(new_model, ao_evals[0], gws[0], dms[0], eris[0], mo_occs[0], hcs[0], ss[0], ogds[0], hologaps[0], alpha0=0.7,
                    dmL = 0.0, holuL = 0.0, dm_to_rho=1.0)

    assert ( abs(new_loss) < abs(init_loss) )
