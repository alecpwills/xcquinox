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

def get_rhos2(rho, spin, libxc = False):
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
    print('get_rhos2 stats: mins/maxs')
    print(f'rho0_a = {rho0_a.min(), rho0_a.max()}')
    print(f'rho0_b = {rho0_b.min(), rho0_b.max()}')
    print(f'gamma_a = {gamma_a.min(), gamma_a.max()}')
    print(f'gamma_b = {gamma_b.min(), gamma_b.max()}')
    print(f'gamma_ab = {gamma_ab.min(), gamma_ab.max()}')
    print(f'tau_a = {tau_a.min(), tau_a.max()}')
    print(f'tau_b = {tau_b.min(), tau_b.max()}')
    if libxc:
        return jnp.stack([rho0_a, rho0_b, gamma_a, gamma_b, gamma_ab, jnp.zeros_like(tau_a), jnp.zeros_like(tau_a), tau_a, tau_b], axis=-1)
    else:
        return jnp.stack([rho0_a, rho0_b, gamma_a, gamma_b, gamma_ab, jnp.zeros_like(tau_a), jnp.zeros_like(tau_a), tau_a, tau_b], axis=-1)


def get_data(mol, xcmodel, xc_func, localnet=None, xorc=None, nlnet=False):
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
        print('exc with xc_func = {} = {}'.format(fxc_a, xc_func))
        print(f'rho_a.shape={rho_alpha.shape}, rho_b.shape={rho_beta.shape}')
        print(f'exc_a.shape={fxc_a.shape}, exc_b.shape={fxc_b.shape}')

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
        print(f'rho_a.shape={rho_alpha.shape}, rho_b.shape={rho_beta.shape}')
        print('exc with xc_func = {} = {}'.format(exc, xc_func))
        fxc = exc
        rho = jnp.stack([rho_alpha,rho_beta], axis=-1)

    dm = jnp.array(mf.make_rdm1())
    print('get_data, dm shape = {}'.format(dm.shape))
    ao_eval = jnp.array(mf._numint.eval_ao(mol, mf.grids.coords, deriv=1))
    print(f'ao_eval.shape={ao_eval.shape}')
    rho = jnp.einsum('xij,yik,...jk->xy...i', ao_eval, ao_eval, dm)        

    
    mf.converged=True
    retrho = get_rhos2(rho, spin=1)
    refexc = mf._numint.eval_xc(xc_func,(rho_alpha,rho_beta), spin=1)[0]

    print(f'retrho shape: {retrho.shape}')
    print(f'refexc shape: {refexc.shape}')
    if not nlnet:
        return retrho, refexc
    else:
        return retrho, refexc, mf, dm, ao, mf.grids.weights, mf.grids.coords
level_dict = {'GGA':2, 'MGGA':3, 'NONLOCAL':4, 'NL':4}

x_lob_level_dict = {'GGA': 1.804, 'MGGA': 1.174, 'NONLOCAL': 1.174, 'NL': 1.174}

class PT_E_Loss(eqx.Module):

    def __call__(self, model, inp, ref):
        pred = model.eval_grid_models(inp)[:, 0]
        pred2 = jnp.nan_to_num(pred)
        pred_nans = jnp.isnan(pred)
        print(f'Pred stats: pred_nans = {pred_nans.sum()}')
        print(f'Pred2 stats: shape={pred2.shape} mean={jnp.mean(pred)}')
        print(f'Ref stats: shape={ref.shape} mean={jnp.mean(ref)}')
        err = (pred2-ref)**2

        return jnp.sqrt(jnp.mean(err))

class NL_PT_E_Loss(eqx.Module):

    def __call__(self, model, inp, mf, dm, ao, gw, coor, ref):
        pred = model.eval_grid_models(inp, mf, dm, ao, gw, coor)[:, 0]
        pred2 = jnp.nan_to_num(pred)
        pred_nans = jnp.isnan(pred)
        print(f'Pred stats: pred_nans = {pred_nans.sum()}')
        print(f'Pred2 stats: shape={pred2.shape} mean={jnp.mean(pred)}')
        print(f'Ref stats: shape={ref.shape} mean={jnp.mean(ref)}')
        err = (pred2-ref)**2

        return jnp.sqrt(jnp.mean(err))


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
    ksr_atoms = [Atoms('P',info={'spin':3}), Atoms('N', info={'spin':3}), Atoms('H', info={'spin':1}),Atoms('Li', info={'spin':1}), Atoms('O',info={'spin':2}),Atoms('Cl',info={'spin':1}),Atoms('Al',info={'spin':1}), Atoms('S',info={'spin':2})] + atoms

    mols = [get_mol(atoms) for atoms in ksr_atoms]
    mols = [i for i in mols if len(i.atom) < 8]
    for i in mols:
        print(i, i.atom, len(i.atom))
    if pargs.debug:
        mols = mols[:1]
    ref = pargs.pretrain_xc
    nlnet = True if pargs.pretrain_level in ['NONLOCAL', 'NL'] else False
    data = [get_data(mol, xcmodel=xc, xc_func=ref, localnet=localnet, xorc=pargs.pretrain_net, nlnet=nlnet) for i,mol in enumerate(mols)]


    cpus = jax.devices(backend='cpu')

    scheduler = optax.exponential_decay(init_value = pargs.init_lr, transition_begin=pargs.lr_decay_start, transition_steps=pargs.n_steps-pargs.lr_decay_start, decay_rate=pargs.lr_decay_rate)
    optimizer = optax.adam(learning_rate = scheduler)

    trainer = xce.train.xcTrainer(model=xc, optim=optimizer, steps=pargs.n_steps, loss = PT_E_Loss(), do_jit=pargs.do_jit, logfile='ptlog')
    NLtrainer = xce.train.xcTrainer(model=xc, optim=optimizer, steps=pargs.n_steps, loss = NL_PT_E_Loss(), do_jit=pargs.do_jit, logfile='ptlog')
    #these are (:, 9), so concatenate along 0
    inp = jnp.concatenate([d[-2] for d in data], axis=0)

    #can't have negatives, just clip
    inp = jnp.clip(inp, 0, None)
    #these are (:,) so doesn't matter
    ref = jnp.concatenate([d[-1] for d in data], axis=-1)

    inp_nans = jnp.isnan(inp)
    print(f'inp_nans sum: {inp_nans.sum()}')
    print(f'rho inp.shape = {inp.shape}')
    print(f'ref.shape = {ref.shape}')


    if pargs.pretrain_level.upper() in ['NONLOCAL', 'NL']:
        mfs = [d[2] for d in data]
        dms = [d[3] for d in data]
        aos = [d[4] for d in data]
        gws = [d[5] for d in data]
        coors = [d[6] for d in data]
        nlinp = [ d[0] for d in data]
        nlref = [d[1] for d in data]
    else:
        inp = [inp]
    if not nlnet:
        with jax.default_device(cpus[0]):
            newm = trainer(1, trainer.model, inp, [ref])
    else:
        with jax.default_device(cpus[0]):
            newm = NLtrainer(len(mfs), NLtrainer.model, nlinp, mfs, dms, aos, gws, coors, nlref)
