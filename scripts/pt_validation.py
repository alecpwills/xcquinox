from pyscf import gto,dft,scf
import pickle
import numpy as np
import jax.numpy as jnp
from ase.io import read
import xcquinox as xce
import equinox as eqx
import os, jax, argparse
import faulthandler
import argparse
import pandas as pd
faulthandler.enable()
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

def get_mol(atoms, basis='6-311++G**'):
    '''
    Generates a pyscf.gto.Mole object, given an ASE atoms object

    :param atoms: ase.Atoms object containing the atomic information for the given molecule
    :type atoms: ase.Atoms
    :param basis: Basis with which to generate the various characteristics of the molecule, defaults to '6-311++G**'
    :type basis: str, optional
    :return: The pyscf mole object to use in further characteristic generation (i.e., getting the density)
    :rtype: pyscf.gto.Mole
    '''
    pos = atoms.positions
    spec = atoms.get_chemical_symbols()
    mol_input = [[s, p] for s, p in zip(spec, pos)]
    try:
        mol = gto.Mole(atom=mol_input, basis=atoms.info.get('basis',basis),spin=atoms.info.get('spin',0))
    except Exception:
        mol = gto.Mole(atom=mol_input, basis=atoms.info.get('basis','STO-3G'),spin=atoms.info.get('spin',0))
    return mol 

def get_rhos(rho, spin):
    '''
    Returns a partition of the density array produced by an MGGA PySCF calculation

    :param rho: The multi-dimensional "density" array produced in a PySCF calculation
    :type rho: np.array
    :param spin: The 'spin' of the system, used in polarizing the returned density arrays
    :type spin: int
    :return: A list of the decomposed density arrays
    :rtype: list of np.array
    '''
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

def get_data(mol, xcmodel, xc_func, localnet=None, xorc=None):
    '''
    Gets the exchange or correlation energy density on the grid, given a reference XC functional to use (or a network to evaluate)

    :param mol: The molecule to get the energy density for
    :type mol: pyscf.gto.Mole
    :param xcmodel: the xcquinox network to use in generating the descriptors for network evaluation
    :type xcmodel: xcquinox exchange or correlation network
    :param xc_func: The reference XC functional that was targeted in pre-training, PBE0 or SCAN
    :type xc_func: str
    :param localnet: The MLP portion of the xcmodel network, defaults to None
    :type localnet: , optional
    :param xorc: 'x' for exchange network, 'c' for correlation network, defaults to None
    :type xorc: str, optional
    :return: A tuple of the molecule's network descriptors and the reference functional's energy density on the grid
    :rtype: tuple of arrays
    '''
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
level_dict = {'GGA':2, 'MGGA':3, 'NONLOCAL':4}

x_lob_level_dict = {'GGA': 1.804, 'MGGA': 1.174, 'NONLOCAL': 1.174}

class PT_E_Loss(eqx.Module):
    '''
    A loss module to use in getting the individual network e_xc predictions.
    '''
    def __call__(self, model, inp, ref):
        if model.spin_scaling and len(inp.shape) == 3:
            #spin scaling shape = (2, N, len(self.use))
            pred = jax.vmap(jax.vmap(model.net), in_axes=1)(inp)[:, 0]
        else:
            pred = jax.vmap(model.net)(inp)[:, 0]

        err = pred-ref

        return jnp.mean(jnp.square(err))

def get_model_info(xcdir, model_dir, tlogf = 'ptlog.dat'):
    '''
    Given a top-level directory (which must end with /{target xc functional}), and model directory, returns the values needed to re-create the network

    Model directories MUST have the following format, where 'x' denotes an optional portion of the directory name

    xorc_depth_nodes_'lrspecifier'_'usespecifier'_level,
    where xorc is a string, either 'x' or 'c' to denote the network type
    depth is an integer, indicating the number of hidden layers
    nodes is an integer, indicating the number of nodes per hidden layer
    'lrspecifier', an optional flag, is a string that just categorizes the network as having had a non-default training schedule
    'usespecifer', an option flag, is a string that categorizes the network as having used a different subset of descriptors
    level is a string, i.e. 'gga' or 'mgga' or 'nl' to denote the "level" of the network

    :param xcdir: The top-level directory, ending with the XC functional that was used to train against
    :type xcdir: str
    :param model_dir: A directory whose structure is as delineated above.
    :type model_dir: str
    :param tlogf: The log file used in pre-training, defaults to 'ptlog.dat'
    :type tlogf: str, optional
    :return: A tuple of (refxc, xorc, int(depth), int(nodes), ruse, int(rinp), rlevel.upper(), xcf)
    :rtype: tuple
    '''
    refxc = xcdir.split('/')[-1]
    nd_split = model_dir.split('_')
    
    def_mgga_x_use = [1, 2]
    def_mgga_c_use = []
    def_mgga_x_inp = 2
    def_mgga_c_inp = 4
    def_nl_x_use = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    def_nl_x_inp = len(def_nl_x_use)
    def_nl_c_use = []
    def_nl_c_inp = 16
    
    use2_nl_x_use = [1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    use2_nl_c_use = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    use2_nl_x_inp = len(use2_nl_x_use)
    use2_nl_c_inp = len(use2_nl_c_use)
    
    if len(nd_split) == 4:
        #xorc_depth_nodes_level    
        xorc, depth, nodes, level = nd_split
        lr2 = ''
        use = ''
    elif len(nd_split) == 5:
        #xorc_depth_nodes_level_lr2, just denotes a different learning rate schedule used    
        xorc, depth, nodes, level, lr2 = nd_split
        use = ''
    elif len(nd_split) == 6:
        #xorc_depth_nodes_level_lr2_use2, just denotes a different learning rate schedule used    
        xorc, depth, nodes, level, lr2, use = nd_split

    if xorc == 'x':
        if level == 'mgga':
            rinp = def_mgga_x_inp
            ruse = def_mgga_x_use
        elif level == 'nl':
            if not use:
                rinp = def_nl_x_inp
                ruse = def_nl_x_use
            else:
                rinp = use2_nl_x_inp
                ruse = use2_nl_x_use
    elif xorc == 'c':
        if level == 'mgga':
            rinp = def_mgga_c_inp
            ruse = def_mgga_c_use
        elif level == 'nl':
            if not use:
                rinp = def_nl_c_inp
                ruse = def_nl_c_use
            else:
                rinp = use2_nl_c_inp
                ruse = use2_nl_c_use
    try:
        xcs = sorted([i for i in os.listdir(os.path.join(xcdir,model_dir)) if 'xc.eqx' in i],
                 key = lambda x: int(x.split('.')[-1]))
    except:
        xcs = sorted([i for i in os.listdir(os.path.join(xcdir,model_dir)) if 'xc.eqx' in i])
    if not xcs:
        print('No networks in directory')
        return
    if tlogf:
        try:
            loss = pd.read_csv(os.path.join(xcdir, model_dir, 'ptlog.dat'), delimiter='\t')
            epoch_min = loss[loss['Loss'] == loss['Loss'].min()]['#Epoch'].values[0]
            xcf = [i for i in xcs if int(i.split('.')[-1]) == epoch_min][0]
        except:
            if not xcs:
                print('No networks in directory')
                xcf = ''
            else:
                xcf = xcs[-1]
    else:
        selind = -1
        xcf = xcs[selind]
        
    if level == 'nl':
        rlevel = 'nonlocal'.upper()
    else:
        rlevel = level.upper()
    return (refxc, xorc, int(depth), int(nodes), ruse, int(rinp), rlevel.upper(), xcf)

def gen_network_model(xorc, depth, nodes, ninp, use, level='MGGA', ptpath = None, genverbose=False,
                     ueg_limit = None, spin_scaling = None):
    '''
    Creates the network structure to validate against

    :param xorc: Whether the network validated is and exchange ('x') or correlation ('c') network
    :type xorc: str
    :param depth: The number of hidden layers in the generated network
    :type depth: int
    :param nodes: The number of nodes in a layer
    :type nodes: int
    :param ninp: The number of outputs for the model
    :type ninp: int
    :param use: A list of integers instructing the model which descriptors to use
    :type use: list of ints
    :param level: The "rung" of Jacob's ladder being used, defaults to 'MGGA'
    :type level: str, optional
    :param ptpath: The path to the pre-trained model whose network weights are to be loaded, defaults to None
    :type ptpath: str, optional
    :param genverbose: If flagged, the resulting network will be "verbose" and print a lot of typically extraneous information, defaults to False
    :type genverbose: bool, optional
    :return: The resulting xcquinox network
    :rtype: xcquinox.xc.eXC network
    '''
    level_dict = {'GGA':2, 'MGGA':3, 'NONLOCAL':4, 'NL':4}
    x_lob_level_dict = {'GGA': 1.804, 'MGGA': 1.174, 'NONLOCAL': 1.174, 'NL':1.174}
    ueg_limit = ueg_limit if ueg_limit is not None else True
    if xorc == 'x':
        ss = spin_scaling if spin_scaling is not None else True
        net = xce.net.eX(n_input = ninp,
                         n_hidden = nodes,
                         depth = depth,
                         use = use,
                         ueg_limit = ueg_limit,
                         lob=x_lob_level_dict[level],
                         spin_scaling=ss)
    elif xorc == 'c':
        ss = spin_scaling if spin_scaling is not None else False
        net = xce.net.eC(n_input = ninp,
                         n_hidden = nodes,
                         depth = depth,
                         use = use,
                         ueg_limit = ueg_limit,
                         spin_scaling=ss)

    if ptpath:
        print('Attempting to deserialize network from {}'.format(ptpath))
        net = eqx.tree_deserialise_leaves(ptpath, net)

    xc = xce.xc.eXC(grid_models=[net], heg_mult=True, level=level_dict[level], verbose=genverbose)
    return xc, net

#A dictionary containing ground-state spin values for single atoms,
# as PySCF doesn't even try to guess this and assume 0 if not otherwise specified
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pretrain_traj', type=str, action='store', default = '../scripts/script_data/haunschild_g2/g2_97.traj', help='The location of the ASE trajectory from which the pre-training molecules were selected.')
    parser.add_argument('--pretrain_topdir', type=str, action='store', default = '', help='The location of the directory containing the pre-trained network models.')
    parser.add_argument('--pretrain_xc', type=str, action='store', default = 'PBE', help='The functional that was used in pre-training the networks.')
    parser.add_argument('--pretrain_inds', nargs='+', type=int, action='store', default=[2, 113, 25, 18, 11, 17, 114, 121, 101, 0, 20, 26, 29, 67, 28, 110, 125, 10, 115, 89, 105, 50], help='Specify here the indices of the molecules selected from --pretrain_traj that were used in pre-training.')
    parser.add_argument('--validation_size', type=int, action='store', default=0, help='If non-zero, will select a random subset of this size to use in validation. Otherwise, does the entire trajectory minus those that were used in pre-training.')
    parser.add_argument('--mol_len_limit', type=int, action='store', default=99, help='Will only validate against molecules that have length <= this number.')
    parser.add_argument('--results_savedir', type=str, action='store', default = '', help='The location of the directory where the validation results will be saved.')
    args = parser.parse_args()



    #the location of the g297 trajectory file, within the xcquinox package subdirectories
    #alternatively, the location of the generic pre-train trajectory from which the pre-training molecules were taken (assuming there are more in this trajectory to use in validation)
    g297_path = args.pretrain_traj
    g297 = read(g297_path, ':')
    ng297 = len(g297)
    inds = np.arange(0, ng297)

    #the default selected molecules from the g297 dataset that were used in pre-training
    pt_selection = args.pretrain_inds
    val_selection = [i for i in inds if i not in pt_selection]
    mols = [get_mol(atoms) for atoms in g297]
    #lists of the pre-training molecules and the validation molecules
    pt_atoms = [mols[idx] for idx in pt_selection]
    val_atoms = [mols[idx] for idx in val_selection]

    #select only molecules within the given length limit
    val_mols = [i for i in val_atoms if len(i.atom) <= args.mol_len_limit]

    #set a random seed for validation selection
    np.random.seed(seed=92017)

    SIZE = args.validation_size
    if SIZE:
        #if we specify a certain size for the validation set, the random seed is used in selecting that random subset
        val_selection = np.random.choice(val_mols, size=SIZE, replace=False)
    else:
        #otherwise, just use the molecules not pre-trained on
        val_selection = val_mols

    #print some information about the calculation
    for idx, i in enumerate(val_selection):
        print(idx, i.atom, len(i.atom))



    #the top-level directories under which the pre-trained networks are contained 
    d = args.pretrain_topdir
    #the target directory where results will be saved
    sd = args.results_savedir
    #the reference XC functional used in pre-training
    XCREF = args.pretrain_xc.upper()

    #this generates the list of networks generated, assuming they are the only directories with underscores in them in the subdirectories
    nets = sorted([i for i in os.listdir(d) if '_' in i and os.path.isdir(os.path.join(d, i))])

    #set up empty dictionary
    val_dct = {XCREF: {}}
    refxcps = {XCREF:d}
    refxcps2 = {XCREF:sd}

    #populate the validation dictionaries with the directories of the pre-trained networks as sub-directories, with empty lists for losses
    val_dct[XCREF] = {'x': {'gga': {k : [] for k in nets if 'gga' in k and k[:2] == 'x_' and 'mgga' not in k},
                            'mgga': {k : [] for k in nets if 'mgga' in k and k[:2] == 'x_'},
                            'nl': {k : [] for k in nets if 'nl' in k and k[:2] == 'x_'},
                            },
                    'c': {'gga': {k : [] for k in nets if 'gga' in k and k[:2] == 'c_' and 'mgga' not in k},
                            'mgga': {k : [] for k in nets if 'mgga' in k and k[:2] == 'c_'},
                            'nl': {k : [] for k in nets if 'nl' in k and k[:2] == 'c_'},
                            },
                    }

    #create empty dictionary for the calculated densities on the grid
    calc_dct = {}
    #loop over the reference XC functionals in val_dct
    for krefxc in val_dct.keys():
        #select the sub-directory from val_dct, and create a sub-directory in calc_dct using this reference XC functional
        krefdct = val_dct[krefxc]
        calc_dct[krefxc] = {}
        for kxorc in krefdct.keys():
            #loops over the 'x' or 'c' values in the val_dct sub-dictionary, and again creates an empty dictionary to hold the results
            krefxcdct = krefdct[kxorc]
            calc_dct[krefxc][kxorc] = {}
            for klevel in krefxcdct.keys():
                #loop over the 'mgga' or 'nl' levels in the val_dct subdictionary
                krxcldct = krefxcdct[klevel]
                #the networks included in this reference XC -> x/c -> level
                knets = sorted(list(krxcldct.keys()))
                for knetidx, knet in enumerate(knets): 
                    #loop over the list indices and network directories
                    if ('c0' in knet or 'c1' in knet) and ('nc0' not in knet and 'nc1' not in knet):
                        print('Constrained network...')
                        constr = True
                        ueg_limit = None
                        spin_scaling = None
                        cstr = 'c'
                    elif 'nc0' in knet or 'nc1' in knet:
                        print('Unconstrained network...')
                        constr = False
                        ueg_limit = False
                        spin_scaling = False
                        cstr = 'nc'
                    #if first attempt, print stats and generate the reference data
                    thissave = True
                    if os.path.exists(os.path.join(refxcps2[krefxc], f'{krefxc}_{kxorc}_{klevel}_{cstr}.data.pkl')):
                        print('Data file exists. loading in.')
                        thissave = False
                        with open(os.path.join(refxcps2[krefxc], f'{krefxc}_{kxorc}_{klevel}_{cstr}.data.pkl'), 'rb') as f:
                            data = pickle.load(f)
                    else:
                        print('Data file does not yet exist. Creating...')
                        print(krefxc, kxorc, klevel, knet)
                        try:
                            tup = get_model_info(refxcps[krefxc], knet)
                        except:
                            with open(os.path.join(refxcps[krefxc], knet, 'network.config.pkl'), 'rb') as f:
                                tup = pickle.load(f)
                        print(tup)
                        try:
                            if type(tup) == tuple:
                                refxc, xorc, depth, nodes, ruse, rinp, level, xcf = tup
                            elif type(tup) == dict:
                                refxc = krefxc
                                xorc = knet.split('_')[0]
                                depth = tup['depth']
                                nodes = tup['nhidden']
                                ruse = tup['use']
                                rinp = tup['ninput']
                                level = knet.split('_')[-1].upper()
                                xcf = 'xc.eqx'
                        except Exception as e:
                            print("Exception raise: ", e)
                            print('no networks found')
                            continue
                        #create the network to generate the descriptors for saving
                        xc, net = gen_network_model(xorc, depth, nodes, rinp, ruse, level, 
                                                    ptpath = os.path.join(refxcps[krefxc], knet, xcf), ueg_limit = ueg_limit, spin_scaling=spin_scaling)
                    
                        #DO GET_DATA GENERATION HERE
                        #data generated only depends on get_descriptors, determined by level, not use
                        data = []
                        calcs = []
                        rejects = []
                        for idx, mol in enumerate(val_selection):
                            print('----------------------------')
                            print(f'-----{idx}/{len(val_selection)}-----')
                            print(f'Calculating {mol.atom}')
                            try:
                                data.append(get_data(mol, xcmodel=xc, xc_func=krefxc, localnet=net))
                                calcs.append(mol.atom)
                            except Exception as e:
                                print("Exception raised: ", e)
                                rejects.append(mol.atom)
                                continue
                    #save this combo's descriptors and energy densities to a file for later use
                    if thissave:
                        with open(os.path.join(refxcps2[krefxc], f'{krefxc}_{kxorc}_{klevel}_{cstr}.data.pkl'), 'wb') as f:
                            pickle.dump(data, f)


    #create empty dictionary for the calculated densities on the grid
    calc_dct = {}
    data = []
    calcs = []
    for idx, mol in enumerate(val_selection):
        # print('----------------------------')
        # print(f'-----{idx}/{len(val_selection)}-----')
        # print(f'{mol.atom}')
        calcs.append(mol.atom)

    #loop over the reference XC functionals in val_dct, PBE0 and SCAN
    for krefxc in val_dct.keys():
        #select the sub-directory from val_dct, and create a sub-directory in calc_dct using this reference XC functional
        krefdct = val_dct[krefxc]
        calc_dct[krefxc] = {}
        for kxorc in krefdct.keys():
            #loops over the 'x' or 'c' values in the val_dct sub-dictionary, and again creates an empty dictionary to hold the results
            krefxcdct = krefdct[kxorc]
            calc_dct[krefxc][kxorc] = {}
            for klevel in krefxcdct.keys():
                #loop over the 'mgga' or 'nl' levels in the val_dct subdictionary
                krxcldct = krefxcdct[klevel]
                calc_dct[krefxc][kxorc][klevel] = {}
                #the networks included in this reference XC -> x/c -> level
                knets = sorted(list(krxcldct.keys()))
                for knetidx, knet in enumerate(knets): 
                    #loop over the list indices and network directories
                    if ('c0' in knet or 'c1' in knet) and ('nc0' not in knet and 'nc1' not in knet):
                        print('Constrained network...')
                        constr = True
                        ueg_limit = None
                        spin_scaling = None
                        cstr = 'c'
                    elif 'nc0' in knet or 'nc1' in knet:
                        print('Unconstrained network...')
                        constr = False
                        ueg_limit = False
                        spin_scaling = False
                        cstr = 'nc'
                    #read in the calculated reference dictionary
                    try:
                        with open(os.path.join(refxcps2[krefxc], f'{krefxc}_{kxorc}_{klevel}_{cstr}.data.pkl'), 'rb') as f:
                            data = pickle.load(f)
                    except:
                        raise
                    print(krefxc, kxorc, klevel, knet)
                    try:
                        tup = get_model_info(refxcps[krefxc], knet)
                    except:
                        with open(os.path.join(refxcps[krefxc], knet, 'network.config.pkl'), 'rb') as f:
                            tup = pickle.load(f)
                    print(tup)
                    if type(tup) == tuple:
                        refxc, xorc, depth, nodes, ruse, rinp, level, xcf = tup
                    elif type(tup) == dict:
                        refxc = krefxc
                        xorc = knet.split('_')[0]
                        depth = tup['depth']
                        nodes = tup['nhidden']
                        ruse = tup['use']
                        rinp = tup['ninput']
                        level = knet.split('_')[-1].upper()
                        xcf = 'xc.eqx'
                    xc, net = gen_network_model(xorc, depth, nodes, rinp, ruse, level, 
                                                    ptpath = os.path.join(refxcps[krefxc], knet, xcf), ueg_limit = ueg_limit, spin_scaling=spin_scaling)
                    losses = []
                    tdrhos = [i[0] for i in data]
                    tfxcs = [i[1] for i in data]
                    print(f'{krefxc} -- {knet} Validation beginning...')
                    for idx, dat in enumerate(tdrhos):
                        if idx == 0:
                            print(f'Descriptor shape: {dat.shape}')
                        # print(f'{idx}/{len(tdrhos)}')
                        #reference energy density and the descriptors
                        this_tFxc = tfxcs[idx]
                        this_tdrho = dat
                        if ruse:
                            if idx == 0:
                                print('ruse is specified: {}'.format(ruse))
                            #if there is a specified 'use' for the network, only select those descriptors as inputs
                            if net.spin_scaling:
                                if len(this_tdrho.shape) == 3:
                                    inp = this_tdrho[:, :, ruse]
                                else:
                                    inp = this_tdrho[:, ruse]
                            else:
                                inp = this_tdrho[:, ruse]
                        else:
                            #otherwise, no subsetting necessary
                            inp = this_tdrho
                        loss = PT_E_Loss()(net, inp, this_tFxc)
                        losses.append(loss)
                    calc_dct[krefxc][kxorc][klevel][knet] = {'calcs':calcs, 'calc_losses':losses}
                    val_dct[krefxc][kxorc][klevel][knet] = losses
                    with open(os.path.join(refxcps2[krefxc], f'{krefxc}_{knet}.val_dct.pkl'), 'wb') as f:
                        pickle.dump(val_dct, f)
                    with open(os.path.join(refxcps2[krefxc], f'{krefxc}_{knet}.calc_dct.pkl'), 'wb') as f:
                        pickle.dump(calc_dct, f)
    #loop has completed, now output the complete dictionaries
    with open(os.path.join(refxcps2[krefxc], f'{krefxc}.val_dct.pkl'), 'wb') as f:
        pickle.dump(val_dct, f)
    with open(os.path.join(refxcps2[krefxc], f'{krefxc}.calc_dct.pkl'), 'wb') as f:
        pickle.dump(calc_dct, f)
