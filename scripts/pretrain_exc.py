from pyscf import gto,dft,scf
import numpy as np
import jax.numpy as jnp
from ase import Atoms
from ase.io import read
import xcquinox as xce
import equinox as eqx
import os, optax, jax, argparse
import faulthandler

faulthandler.enable()
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

parser = argparse.ArgumentParser(description='Pre-train a network-based xc functional, for further optimization')
parser.add_argument('--pretrain_level', action='store', type=str, choices=['GGA','MGGA','NONLOCAL', 'NL'], help='The level of network to pre-train, i.e. GGA, MGGA, or nonlocal')
parser.add_argument('--pretrain_net_path', action='store', type=str, default='', help='The location of the network to load (if loading one), for get_net')
parser.add_argument('--pretrain_net', action='store', type=str, choices=['x','c'], help='Specify whether to optimize the exchange network or correlation network, via "x" or "c"')
parser.add_argument('--pretrain_lob', action='store', type=float, default=None, help='If specified and net_path is not, creates network using this LOB')
parser.add_argument('--pretrain_ueg', action='store', type=float, default=None, help='If specified and net_path is not, creates network using this flag for HEG scaling')
parser.add_argument('--n_hidden', action='store', type=int, default=16, help='The number of hidden nodes in a given layer for the network')
parser.add_argument('--depth', action='store', type=int, default=3, help='The number of layers in the MLP for the network')
parser.add_argument('--n_input', action='store', type=int, default=2, help='The number of inputs to the network you are generating.')
parser.add_argument('--use', nargs='+', type=int, action='store', default=[], help='Specify the desired indices for the network to actually use, if not the full range of descriptors.')
parser.add_argument('--pretrain_xc', action='store', type=str, help='Specify which XC functional to optimize against, e.g. PBE0')
parser.add_argument('--n_steps', action='store', type=int, default=200, help='The number training epochs to go through.')
parser.add_argument('--do_jit', action='store_true', default=False, help='If flagged, will try to utilize JAX-jitting during training')
parser.add_argument('--g297_data_path', action='store', type=str, default='/home/awills/Documents/Research/xcquinox/scripts/script_data/haunschild_g2/g2_97.traj', help='Location of the data file for use during pre-training')
parser.add_argument('--debug', action='store_true', help='If flagged, only selects first in the target list for quick debugging purposes.')
parser.add_argument('--verbose', action='store_true', help='If flagged, activates verbosity flag in the network.')
parser.add_argument('--spin_scaling', action='store_true', help='If flagged, enforces spin_scaling behavior in the desired network.')
parser.add_argument('--init_lr', action='store', type=float, default=5e-2, help='The initial learning rate for the network.')
parser.add_argument('--lr_decay_start', action='store', type=int, default=50, help='The epoch at which the exponential decay begins.')
parser.add_argument('--lr_decay_rate', action='store', type=float, default=0.9, help='The decay rate for the exponential decay.')



def get_mol(atoms, basis='6-311++G**'):
    pos = atoms.positions
    spec = atoms.get_chemical_symbols()
    mol_input = [[s, p] for s, p in zip(spec, pos)]
    try:
        mol = gto.Mole(atom=mol_input, basis=atoms.info.get('basis',basis),spin=atoms.info.get('spin',0))
    except Exception:
        mol = gto.Mole(atom=mol_input, basis=atoms.info.get('basis','STO-3G'),spin=atoms.info.get('spin',0))
    return mol 

def get_rhos(rho, spin):
    rho0 = rho[0,0]
    drho = rho[0,1:4] + rho[1:4,0]
    tau = 0.5*(rho[1,1] + rho[2,2] + rho[3,3])

    if spin != 0:
        rho0_a = rho0[0]
        rho0_b = rho0[1]
        gamma_a, gamma_b = jnp.einsum('ij,ij->j',drho[:,0],drho[:,0]), jnp.einsum('ij,ij->j',drho[:,1],drho[:,1])              
        gamma_ab = jnp.einsum('ij,ij->j',drho[:,0],drho[:,1])
        tau_a, tau_b = tau
    else:
        rho0_a = rho0_b = rho0*0.5
        gamma_a=gamma_b=gamma_ab= jnp.einsum('ij,ij->j',drho[:],drho[:])*0.25
        tau_a = tau_b = tau*0.5
    return rho0_a, rho0_b, gamma_a, gamma_b, gamma_ab, tau_a, tau_b
    
def get_data(mol, xcmodel, xc_func, localnet=None, xorc=None):
    print('mol: ', mol.atom)
    try:
        mf = scf.UKS(mol)
    except:
        mf = dft.RKS(mol)
    mf.xc = 'PBE'
    mf.grids.level = 1
    mf.kernel()
    ao = mf._numint.eval_ao(mol, mf.grids.coords, deriv=2)
    dm = mf.make_rdm1()
    if len(dm.shape) == 2:
        #artificially spin-polarize
        dm = np.array([0.5*dm, 0.5*dm])
    print('New DM shape: {}'.format(dm.shape))
    print('ao.shape', ao.shape)

    #depending on the x or c type, choose the generation of the exchange or correlation density
    if xorc == 'x':
        print('Exchange contribution only')
        xc_func = xc_func+','
        if xc_func.lower() == 'pbe0,':
            print('PBE0 detected. changing xc_func to be combination of HF and PBE')
            xc_func = '0.25*HF + 0.75*PBE,'
        print(xc_func)
    elif xorc == 'c':
        print('Correlation contribution only')
        xc_func = ','+xc_func
        if xc_func.lower() == ',pbe0':
            print('PBE0 detected. Changing correlation to be just PBE')
            xc_func = ',pbe'
        print(xc_func)
    if localnet.spin_scaling:
        print('spin scaling')
        rho_alpha = mf._numint.eval_rho(mol, ao, dm[0], xctype='metaGGA',hermi=True)
        rho_beta = mf._numint.eval_rho(mol, ao, dm[1], xctype='metaGGA',hermi=True)
        fxc_a =  mf._numint.eval_xc(xc_func,(rho_alpha,rho_alpha*0), spin=1)[0]
        fxc_b =  mf._numint.eval_xc(xc_func,(rho_beta*0,rho_beta), spin=1)[0]
        print('fxc with xc_func = {} = {}'.format(fxc_a, xc_func))
        print(f'rho_a.shape={rho_alpha.shape}, rho_b.shape={rho_beta.shape}')
        print(f'fxc_a.shape={fxc_a.shape}, fxc_b.shape={fxc_b.shape}')

        if mol.spin != 0 and sum(mol.nelec)>1:
            print('mol.spin != 0 and sum(mol.nelec) > 1')
            rho = jnp.concatenate([rho_alpha, rho_beta], axis=-1)
            fxc = jnp.concatenate([fxc_a, fxc_b])
            print(f'rho.shape={rho.shape}, fxc.shape={fxc.shape}')
        else:
            print('NOT (mol.spin != 0 and sum(mol.nelec) > 1)')
            rho = rho_alpha
            fxc = fxc_a
            print(f'rho.shape={rho.shape}, fxc.shape={fxc.shape}')
    else:    
        print('no spin scaling')
        rho_alpha = mf._numint.eval_rho(mol, ao, dm[0], xctype='metaGGA',hermi=True)
        rho_beta = mf._numint.eval_rho(mol, ao, dm[1], xctype='metaGGA',hermi=True)
        exc = mf._numint.eval_xc(xc_func,(rho_alpha,rho_beta), spin=1)[0]
        print('exc with xc_func = {} = {}'.format(exc, xc_func))
        fxc = exc
        rho = jnp.stack([rho_alpha,rho_beta], axis=-1)
    
    dm = jnp.array(mf.make_rdm1())
    print('get_data, dm shape = {}'.format(dm.shape))
    ao_eval = jnp.array(mf._numint.eval_ao(mol, mf.grids.coords, deriv=1))
    print(f'ao_eval.shape={ao_eval.shape}')
    rho = jnp.einsum('xij,yik,...jk->xy...i', ao_eval, ao_eval, dm)        
    rho0 = rho[0,0]

    print('rho shape', rho.shape)
    if dm.ndim == 3:
        rho_filt = (jnp.sum(rho0,axis=0) > 1e-6)
    else:
        rho_filt = (rho0 > 1e-6)
    print('rho_filt shape:', rho_filt.shape)

    
    mf.converged=True
    tdrho = xcmodel.get_descriptors(*get_rhos(rho, spin=1), spin_scaling=localnet.spin_scaling, mf=mf, dm=dm)
    print(f'get descriptors tdrho.shape={tdrho.shape}')
    if localnet.spin_scaling:
        if mol.spin != 0 and sum(mol.nelec) > 1:
            print('mol.spin != 0 and sum(mol.nelec) > 1')
            #tdrho not returned in a spin-polarized form regardless,
            #but the enhancement factors sampled as polarized, so double
            if len(tdrho.shape) == 3:
                print('concatenating spin channels along axis=0')
                tdrho = jnp.concatenate([tdrho[0],tdrho[1]], axis=0)
            else:
                print('concatenating along axis=0')
                tdrho = jnp.concatenate([tdrho, tdrho], axis=0)
            rho_filt = jnp.concatenate([rho_filt]*2)
            print(f'tdrho.shape={tdrho.shape}')
            print(f'rho_filt.shape={rho_filt.shape}')
        else:
            #spin == 0 or hydrogen
            tdrho = tdrho[0]
            
    try:
        tdrho = tdrho[rho_filt]
        tFxc = jnp.array(fxc)[rho_filt]
    except:
        tdrho = tdrho[:, rho_filt, :]
        tFxc = jnp.array(fxc)[rho_filt]
    return tdrho, tFxc
level_dict = {'GGA':2, 'MGGA':3, 'NONLOCAL':4, 'NL':4}

x_lob_level_dict = {'GGA': 1.804, 'MGGA': 1.174, 'NONLOCAL': 1.174, 'NL': 1.174}

class PT_E_Loss(eqx.Module):

    def __call__(self, model, inp, ref):
        if model.spin_scaling and len(inp.shape) == 3:
            #spin scaling shape = (2, N, len(self.use))
            pred = jax.vmap(jax.vmap(model.net), in_axes=1)(inp)[:, 0]
        else:
            pred = jax.vmap(model.net)(inp)[:, 0]

        err = pred-ref

        return jnp.mean(jnp.square(err))

    

if __name__ == '__main__':
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

    pargs = parser.parse_args()

    if pargs.pretrain_net_path:
        localnet, params = xce.net.get_net(xorc = pargs.pretrain_net,
                                   level = pargs.pretrain_level,
                                   net_path = pargs.pretrain_net_path
                                   )
    else:
        localnet, _ = xce.net.make_net(xorc = pargs.pretrain_net,
                                    level = pargs.pretrain_level,
                                    depth = pargs.depth,
                                    nhidden = pargs.n_hidden,
                                    ninput = pargs.n_input,
                                    use = pargs.use,
                                    spin_scaling = pargs.spin_scaling,
                                    lob = pargs.pretrain_lob,
                                    ueg_limit = pargs.pretrain_ueg
        )
        
    ueg = xce.xc.LDA_X()
    xc = xce.xc.eXC(grid_models=[localnet], heg_mult=True,
                    level = level_dict[pargs.pretrain_level], verbose=pargs.verbose)
    

    spins = {
    'Al': 1,
    'B' : 1,
    'Li': 1,
    'Na': 1,
    'Si': 2 ,
    'Be':0,
    'C': 2,
    'Cl': 1,
    'F': 1,
    'H': 1,
    'N': 3,
    'O': 2,
    'P': 3,
    'S': 2
}

    selection = [2, 113, 25, 18, 11, 17, 114, 121, 101, 0, 20, 26, 29, 67, 28, 110, 125, 10, 115, 89, 105, 50]
    atoms = [read(pargs.g297_data_path,':')[s] for s in selection]
    ksr_atoms = atoms
    if pargs.pretrain_level == 'MGGA':
        ksr_atoms = ksr_atoms[2:]
    ksr_atoms = [Atoms('P',info={'spin':3}), Atoms('N', info={'spin':3}), Atoms('H', info={'spin':1}),Atoms('Li', info={'spin':1}), Atoms('O',info={'spin':2}),Atoms('Cl',info={'spin':1}),Atoms('Al',info={'spin':1}), Atoms('S',info={'spin':2})] + ksr_atoms

    mols = [get_mol(atoms) for atoms in ksr_atoms]
    mols = [i for i in mols if len(i.atom) < 8]
    for i in mols:
        print(i, i.atom, len(i.atom))
    if pargs.debug:
        mols = mols[:1]
    ref = pargs.pretrain_xc
    data = [get_data(mol, xcmodel=xc, xc_func=ref, localnet=localnet, xorc=pargs.pretrain_net) for i,mol in enumerate(mols)]
    if localnet.spin_scaling:
        print(f'localnet.spin_scaling: concatenating the data')
        fdshape = data[0][0].shape
        print(f'first data shape = {fdshape}')
        if len(fdshape) == 3:
            tdrho = jnp.concatenate([d[0] for d in data], axis=1)
        else:
            tdrho = jnp.concatenate([d[0] for d in data], axis=0)
        print(f'concatenated: tdrho.shape={tdrho.shape}')
    else:
        tdrho = jnp.concatenate([d[0] for d in data])
    
    tFxc = jnp.concatenate([d[1] for d in data])
    print(f'PRE NAN FILT: tFxc.shape={tFxc.shape}, tdrho.shape={tdrho.shape}')

    nan_filt_rho = ~jnp.any((tdrho != tdrho), axis=-1)
    nan_filt_fxc = ~jnp.isnan(tFxc)
    print(f'nan_filt_rho.shape={nan_filt_rho.shape}')
    print(f'nan_filt_fxc.shape={nan_filt_fxc.shape}')
    if localnet.spin_scaling:
        tFxc = tFxc[nan_filt_fxc]
        tdrho = tdrho[nan_filt_rho, :]
    else:
        tFxc = tFxc[nan_filt_fxc]
        tdrho = tdrho[nan_filt_rho,:]

    print(f'tFxc.shape={tFxc.shape}, tdrho.shape={tdrho.shape}')
    cpus = jax.devices(backend='cpu')

    scheduler = optax.exponential_decay(init_value = pargs.init_lr, transition_begin=pargs.lr_decay_start, transition_steps=pargs.n_steps-pargs.lr_decay_start, decay_rate=pargs.lr_decay_rate)
    optimizer = optax.adam(learning_rate = scheduler)

    trainer = xce.train.xcTrainer(model=localnet, optim=optimizer, steps=pargs.n_steps, loss = PT_E_Loss(), do_jit=pargs.do_jit, logfile='ptlog')

    if params['use']:
        if localnet.spin_scaling:
            if len(tdrho.shape) == 3:
                inp = [tdrho[:, :, params['use']]]
            else:
                inp = [tdrho[:, params['use']]]
        else:
            inp = [tdrho[:, params['use']]]
    else:
        inp = [tdrho]
    print(f'inp[0].shape = {inp[0].shape}')
    with jax.default_device(cpus[0]):
        newm = trainer(1, trainer.model, inp, [tFxc])
