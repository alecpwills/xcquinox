import sys, os
import jax, optax
import jax.numpy as jnp
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import pytest
from ase import Atoms
from pyscfad import dft, scf
from pyscf.dft import UKS as pUKS
from pyscf.pbc import dft as dftp
from pyscf.pbc import gto as gtop

import xcquinox as xce

mfs = []
mols = []
energies = []
dms = []
ao_evals = []
gws = []
eris = []
init_dms = []
mo_occs = []
hcs = []
vs = []
ts = []
ss = []
hologaps = []
ogds = []

cell = gtop.Cell()
a = 5.43
cell.atom = [['Si', [0,0,0]],
            ['Si', [a/4,a/4,a/4]]]
cell.a = jnp.asarray([[0, a/2, a/2],
                    [a/2, 0, a/2],
                    [a/2, a/2, 0]])
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.exp_to_discard = 0.1
cell.build()
#make the kpts, but this is a gamma point calculation
kpts = cell.make_kpts([2,2,2])
mf = dftp.RKS(cell, xc='pbe0')
init_dms.append(mf.get_init_guess())
e = mf.kernel()
mfs.append(mf)
dm = mf.make_rdm1()
dmj = jnp.array(dm)
dmj.flags = dm.flags
ao_eval = jnp.array(mf._numint.eval_ao(mf.mol, mf.grids.coords, deriv=2))
energies.append(jnp.array(mf.get_veff().exc))
dms.append(dmj)
ogds.append(dm.shape)
ao_evals.append(jnp.array(ao_eval))
gws.append(jnp.array(mf.grids.weights))
ts.append(jnp.array(mf.mol.intor('int1e_kin')))
vs.append(jnp.array(mf.mol.intor('int1e_nuc')))
mo_occs.append(jnp.array(mf.mo_occ))
hcs.append(jnp.array(mf.get_hcore()))
eris.append(jnp.array(mf.mol.intor('int2e')))
ss.append(jnp.linalg.inv(jnp.linalg.cholesky(mf.mol.intor('int1e_ovlp'))))
hologaps.append(jnp.array(mf.mo_energy[mf.mo_occ == 0][0] - mf.mo_energy[mf.mo_occ > 1][-1]))

def test_train_bandgap_si():
    loss = xce.loss.Band_gap_1shot_loss()

    #just do regular MGGA optimization, dont use NL for this test
    xnet = xce.net.eX(n_input = 2, use = [1, 2], ueg_limit=True, lob=1.174)
    cnet = xce.net.eC(n_input = 4, use = [2, 3], ueg_limit=True)
    blankxc = xce.xc.eXC(grid_models = [xnet, cnet], level=3)
    xc = blankxc
    xct = xce.train.xcTrainer(model=xc, optim=optax.adamw(1e-2), steps=3, loss = loss, do_jit=True, logfile='test_bg_log')
    newm = xct(1, xct.model, ao_evals, gws, dms, eris, mo_occs, hcs, ss, ogds, [-1.17], mfs)
    vgf1 = lambda x: xc(x, ao_evals[0], gws[0], mfs[0])
    vgf2 = lambda x: newm(x, ao_evals[0], gws[0], mfs[0])
    _, moe1, _ = xce.utils.get_dm_moe(dms[0], eris[0], vgf1, mo_occs[0], hcs[0], ss[0], ogds[0])
    _, moe2, _ = xce.utils.get_dm_moe(dms[0], eris[0], vgf2, mo_occs[0], hcs[0], ss[0], ogds[0])
    print(moe1 - moe1[mf.mol.nelectron//2-1])
    print(moe2 - moe2[mf.mol.nelectron//2-1])

    assert (abs(moe1[0]-(-1.17))) > (abs(moe2[0]-(-1.17)))

def test_train_dm_gap_si():
    loss = xce.loss.DM_Gap_loss()

    #just do regular MGGA optimization, dont use NL for this test
    xnet = xce.net.eX(n_input = 2, use = [1, 2], ueg_limit=True, lob=1.174)
    cnet = xce.net.eC(n_input = 4, use = [2, 3], ueg_limit=True)
    blankxc = xce.xc.eXC(grid_models = [xnet, cnet], level=3)
    xc = blankxc
    xct = xce.train.xcTrainer(model=xc, optim=optax.adamw(1e-2), steps=3, loss = loss, do_jit=True, logfile='test_dm_bg_log')
    newm = xct(1, xct.model, ao_evals, hcs, eris, ss, gws, init_dms, mo_occs, ogds, dms, hologaps)
    vgf1 = lambda x: xc(x, ao_evals[0], gws[0], mfs[0])
    vgf2 = lambda x: newm(x, ao_evals[0], gws[0], mfs[0])
    _, moe1, _ = xce.utils.get_dm_moe(dms[0], eris[0], vgf1, mo_occs[0], hcs[0], ss[0], ogds[0])
    _, moe2, _ = xce.utils.get_dm_moe(dms[0], eris[0], vgf2, mo_occs[0], hcs[0], ss[0], ogds[0])
    print(moe1 - moe1[mf.mol.nelectron//2-1])
    print(moe2 - moe2[mf.mol.nelectron//2-1])

    assert (abs(moe1[0]-(-1.17))) > (abs(moe2[0]-(-1.17)))
