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
parser.add_argument('--pretrain_level', action='store', type=str, choices=['GGA','MGGA','NONLOCAL'], help='The level of network to pre-train, i.e. GGA, MGGA, or nonlocal')
parser.add_argument('--pretrain_net', action='store', type=str, choices=['x','c'], help='Specify whether to optimize the exchange network or correlation network, via "x" or "c"')
parser.add_argument('--n_hidden', action='store', type=int, default=16, help='The number of hidden nodes in a given layer for the network')
parser.add_argument('--depth', action='store', type=int, default=3, help='The number of layers in the MLP for the network')
parser.add_argument('--n_input', action='store', type=int, default=2, help='The number of inputs to the network you are generating.')
parser.add_argument('--use', nargs='+', type=int, action='store', default=[], help='Specify the desired indices for the network to actually use, if not the full range of descriptors.')
parser.add_argument('--pretrain_xc', action='store', type=str, help='Specify which XC functional to optimize against, e.g. PBE0')
parser.add_argument('--n_steps', action='store', type=int, default=200, help='The number training epochs to go through.')
parser.add_argument('--do_jit', action='store_true', default=False, help='If flagged, will try to utilize JAX-jitting during training')
parser.add_argument('--g297_data_path', action='store', type=str, default='/home/awills/Documents/Research/xcquinox/scripts/script_data/haunschild_g2/g2_97.traj', help='Location of the data file for use during pre-training')
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
    
def get_data_synth(xcmodel, xc_func, n=100):
    def get_rho(s, a):
        c0 = 2*(3*np.pi**2)**(1/3)
        c1 = 3/10*(3*np.pi**2)**(2/3)
        gamma = c0*s
        tau = c1*a+c0**2*s**2/8
        rho = np.zeros([len(a),6])
        rho[:, 1] = gamma
        rho[:,-1] = tau
        rho[:, 0] = 1
        return rho
    
    s_grid = jnp.concatenate([[0],jnp.exp(jnp.linspace(-10,4,n))])
    rho = []
    for s in s_grid:
        if 'MGGA' in xc_func:
            a_grid = jnp.concatenate([jnp.exp(jnp.linspace(jnp.log((s/100)+1e-8),8,n))])
        else:
            a_grid = jnp.array([0])
        rho.append(get_rho(s, a_grid))
        
    rho = jnp.concatenate(rho)
    
    fxc =  dft.numint.libxc.eval_xc(xc_func,rho.T, spin=0)[0]/dft.numint.libxc.eval_xc('LDA_X',rho.T, spin=0)[0] -1
 
    rho = jnp.asarray(rho)
    
    tdrho = xcmodel.get_descriptors(rho[:,0]/2,rho[:,0]/2,(rho[:,1]/2)**2,(rho[:,1]/2)**2,(rho[:,1]/2)**2,rho[:,5]/2,rho[:,5]/2, spin_scaling=True, mf=mf, dm=dm)
    


    tFxc = jnp.array(fxc)
    return tdrho[0], tFxc

def get_data(mol, xcmodel, xc_func, localnet=None):
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
        dm = np.array([0.5*dm, 0.5*dm])
    print('New DM shape: {}'.format(dm.shape))
    print('ao.shape', ao.shape)

    if localnet.spin_scaling:
        print('spin scaling, indicates exchange network')
        rho_alpha = mf._numint.eval_rho(mol, ao, dm[0], xctype='metaGGA',hermi=True)
        rho_beta = mf._numint.eval_rho(mol, ao, dm[1], xctype='metaGGA',hermi=True)
        fxc_a =  mf._numint.eval_xc(xc_func,(rho_alpha,rho_alpha*0), spin=1)[0]/mf._numint.eval_xc('LDA_X',(rho_alpha,rho_alpha*0), spin=1)[0] -1
        fxc_b =  mf._numint.eval_xc(xc_func,(rho_beta*0,rho_beta), spin=1)[0]/mf._numint.eval_xc('LDA_X',(rho_beta*0,rho_beta), spin=1)[0] -1
        print('fxc with xc_func = {} = {}'.format(fxc_a, xc_func))

        if mol.spin != 0 and sum(mol.nelec)>1:
            rho = jnp.concatenate([rho_alpha, rho_beta])
            fxc = jnp.concatenate([fxc_a, fxc_b])
        else:
            rho = rho_alpha
            fxc = fxc_a
    else:    
        print('no spin scaling, indicates correlation network')
        rho_alpha = mf._numint.eval_rho(mol, ao, dm[0], xctype='metaGGA',hermi=True)
        rho_beta = mf._numint.eval_rho(mol, ao, dm[1], xctype='metaGGA',hermi=True)
        exc = mf._numint.eval_xc(xc_func,(rho_alpha,rho_beta), spin=1)[0]
        print('exc with xc_func = {} = {}'.format(exc, xc_func))
        fxc = exc/mf._numint.eval_xc('LDA_C_PW',(rho_alpha, rho_beta), spin=1)[0] -1
        rho = jnp.stack([rho_alpha,rho_beta], axis=-1)
    
    dm = jnp.array(mf.make_rdm1())
    print('get_data, dm shape = {}'.format(dm.shape))
    ao_eval = jnp.array(mf._numint.eval_ao(mol, mf.grids.coords, deriv=1))
    rho = jnp.einsum('xij,yik,...jk->xy...i', ao_eval, ao_eval, dm)
    print('rho shape', rho.shape)
    if dm.ndim == 3:
        rho_filt = (jnp.sum(rho[0,0],axis=0) > 1e-6)
    else:
        rho_filt = (rho[0,0] > 1e-6)
    print('rho_filt shape:', rho_filt.shape)

    
    mf.converged=True
    tdrho = xcmodel.get_descriptors(*get_rhos(rho, spin=1), spin_scaling=localnet.spin_scaling, mf=mf, dm=dm)
            
    if localnet.spin_scaling:
        if mol.spin != 0 and sum(mol.nelec) > 1:
            #tdrho not returned in a spin-polarized form regardless,
            #but the enhancement factors sampled as polarized, so double
            tdrho = jnp.concatenate([tdrho,tdrho])
            rho_filt = jnp.concatenate([rho_filt]*2)
        elif sum(mol.nelec) == 1:
            pass

    tdrho = tdrho[rho_filt]
    tFxc = jnp.array(fxc)[rho_filt]
    return tdrho, tFxc

level_dict = {'GGA':2, 'MGGA':3, 'NONLOCAL':4}

x_lob_level_dict = {'GGA': 1.804, 'MGGA': 1.174, 'NONLOCAL': 1.174}

class PT_E_Loss(eqx.Module):

    def __call__(self, model, inp, ref):

        pred = jax.vmap(model.net)(inp)[:, 0]

        err = pred-ref

        return jnp.mean(jnp.square(err))

    

if __name__ == '__main__':
    pargs = parser.parse_args()

    if pargs.pretrain_net == 'x':
        localnet = xce.net.eX(n_input = pargs.n_input,
                              n_hidden = pargs.n_hidden,
                              depth = pargs.depth,
                              use = pargs.use,
                              lob = x_lob_level_dict[pargs.pretrain_level],
                              ueg_limit = True)
    else:
        localnet = xce.net.eC(n_input = pargs.n_input,
                              n_hidden = pargs.n_hidden,
                              depth = pargs.depth,
                              use = pargs.use,
                              ueg_limit = True)
        
    ueg = xce.xc.LDA_X()
    xc = xce.xc.eXC(grid_models=[localnet], heg_mult=True,
                    level = level_dict[pargs.pretrain_level])
    

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

    ref = pargs.pretrain_xc
    data = [get_data(mol, xcmodel=xc, xc_func=ref, localnet=localnet) for i,mol in enumerate(mols)]
    tdrho = jnp.concatenate([d[0] for d in data])
    tFxc = jnp.concatenate([d[1] for d in data])

    nan_filt = ~jnp.any((tdrho != tdrho),axis=-1)

    tFxc = tFxc[nan_filt]
    tdrho = tdrho[nan_filt,:]

    cpus = jax.devices(backend='cpu')

    scheduler = optax.exponential_decay(init_value = 5e-2, transition_begin=50, transition_steps=pargs.n_steps, decay_rate=0.9)
    optimizer = optax.adam(learning_rate = scheduler)

    trainer = xce.train.xcTrainer(model=localnet, optim=optimizer, steps=pargs.n_steps, loss = PT_E_Loss(), do_jit=pargs.do_jit, logfile='ptlog')

    if pargs.use:
        inp = [tdrho[:, pargs.use]]
    else:
        inp = [tdrho]

    with jax.default_device(cpus[0]):
        newm = trainer(1, trainer.model, inp, [tFxc])
