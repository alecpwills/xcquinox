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
eX = xce.net.eX(n_input = 18,
            n_hidden = 16,
            depth = 3,
            use = [],
            ueg_limit=True,
            lob = 1.174,
            seed = 9001)
eC = xce.net.eC(n_input = 16,
                n_hidden = 16,
                depth = 3,
                use = [],
                ueg_limit=False,
                lob = 1.804,
                seed = 9001)
xc = xce.xc.eXC(grid_models = [eX, eC], level=4, verbose=True)

def test_nonlocal_nl_e_loss():
    trainer = xce.train.xcTrainer(model=xc, optim=optax.adamw(1e-4), steps=1, loss = xce.loss.NL_E_loss(), do_jit=False, logfile='test_nl_log')
    newm = trainer(1, trainer.model, dms, energies, ao_evals, gws, [mf])

    en1 = xc(dms[0], ao_evals[0], gws[0], mf)
    en2 = newm(dms[0], ao_evals[0], gws[0], mf)

    assert abs(en2 - energies[0]) < abs(en1 - energies[0])
