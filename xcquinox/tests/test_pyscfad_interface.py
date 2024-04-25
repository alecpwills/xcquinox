import sys, os
import jax, optax
import jax.numpy as jnp
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import pytest
from ase import Atoms
from pyscfad import dft, scf
from pyscf.dft import UKS as pUKS
from pyscf.pbc import dft as dftp
from pyscf.pbc import gto as gtop

import xcquinox as xce
from xcquinox.pyscf import generate_network_eval_xc

energies = []
dms = []
init_dms = []
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
e_tots = []

#for some tests, pre-compute the calculations
g_h2 = Atoms('HH', positions=[[ 0.      ,  0.      ,  0.371395],
                            [ 0.      ,  0.      , -0.371395]])

name, mol = xce.utils.ase_atoms_to_mol(g_h2, basis='def2tzvpd')
mol.build()
mf = dft.RKS(mol, xc='PBE')
mf.grids.level = 1
mf.conv_tol = 1e-4
mf.max_cycle = 1
e_tot = mf.kernel()
e_tots.append(e_tot)
dm = mf.make_rdm1()
init_dms.append(mf.get_init_guess())
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
# eX = xce.net.eX(n_input = 14,
#             n_hidden = 16,
#             depth = 3,
#             use = [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
#             ueg_limit=True,
#             lob = 1.174,
#             seed = 9001)
# eC = xce.net.eC(n_input = 16,
#                 n_hidden = 16,
#                 depth = 3,
#                 use = [],
#                 ueg_limit=False,
#                 lob = 1.804,
#                 seed = 9001)
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
                use = [],
                ueg_limit=False,
                lob = 1.804,
                seed = 9001)
xc = xce.xc.eXC(grid_models = [eX, eC], level=3, verbose=True)


def test_pyscfad_e_loss():
    optimizer = optax.adam(learning_rate = 1e-2)
    print('training...')
    trainer = xce.train.xcTrainer(model=xc, optim=optimizer, steps=3, loss = xce.loss.E_PySCFAD_loss(), do_jit=False, logfile='log')
    newm = trainer(1, trainer.model, [mf], init_dms, e_tots)
    print('training done; generating eval_xc functions')
    evxc_init = generate_network_eval_xc(mf=mf, dm=init_dms[0], network=xc)
    evxc_final = generate_network_eval_xc(mf=mf, dm=init_dms[0], network=newm)
    print('testing initial network')
    mf.define_xc_(evxc_init, xctype='MGGA')
    e_pred_init = mf.kernel()
    print('testing updated network')
    mf.define_xc_(evxc_final, xctype='MGGA')
    e_pred_final = mf.kernel()

    assert abs(e_pred_final - e_tots[0]) < abs(e_pred_init - e_tots[0])