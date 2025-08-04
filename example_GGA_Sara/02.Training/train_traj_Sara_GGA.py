from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Hartree
from ase.io import read, write
from time import time
import numpy as np
import traceback
# pure pyscf imports
from pyscf import dft
from pyscf import gto, scf, cc
from pyscf.scf import hf, uhf
from pyscf import pbc
from pyscf.pbc import scf as scfp
from pyscf.pbc import gto as gtop
from pyscf.pbc import dft as dftp
from pyscf.pbc import cc as ccp

# pyscfad imports
from pyscfad import dft as dfta
from pyscfad import gto as gtoa
from pyscfad import scf as scfa
from pyscfad import cc as cca
from pyscfad.scf import hf as hfa
from pyscfad import pbc as pbca
from pyscfad.pbc import scf as scfpa
from pyscfad.pbc import gto as gtopa
from pyscfad.pbc import dft as dftpa

import argparse
import dask
import dask.distributed
from dask.distributed import Client, LocalCluster
import os
from pyscf.tools import cubegen
from opt_einsum import contract
from pyscf import __config__
import pickle

import equinox as eqx
import jax.numpy as jnp
import xcquinox as xce
import os
import optax
import jax
import json


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# %%
# spins for single atoms, since pyscf doesn't guess this correctly.
spins_dict = {
    'Al': 1,
    'B': 1,
    'Li': 1,
    'Na': 1,
    'Si': 2,
    'Be': 0,
    'C': 2,
    'Cl': 1,
    'F': 1,
    'H': 1,
    'N': 3,
    'O': 2,
    'P': 3,
    'S': 2,
    'Ar': 0,  # noble
    'Br': 1,  # one unpaired electron
    'Ne': 0,  # noble
    'Sb': 3,  # same column as N/P
    'Bi': 3,  # same column as N/P/Sb
    'Te': 2,  # same column as O/S
    'I': 1  # one unpaired electron
}


def get_spin(at):
    # if single atom and spin is not specified in at.info dictionary, use spins_dict
    print('======================')
    print("GET SPIN: Atoms Info")
    print(at)
    print(at.info)
    print('======================')
    if ((len(at.positions) == 1) and not ('spin' in at.info)):
        print("Single atom and no spin specified in at.info")
        spin = spins_dict[str(at.symbols)]
    else:
        print("Not a single atom, or spin in at.info")
        if type(at.info.get('spin', None)) == type(0):
            # integer specified in at.info['spin'], so use it
            print('Spin specified in atom info.')
            spin = at.info['spin']
        elif 'radical' in at.info.get('name', ''):
            print('Radical specified in atom.info["name"], assuming spin 1.')
            spin = 1
        elif at.info.get('openshell', None):
            print("Openshell specified in atom info, attempting spin 2.")
            spin = 2
        else:
            print("No specifications in atom info to help, assuming no spin.")
            spin = 0
    return spin


def do_net_calc(epoch, idx, atoms, basis, **kwargs):
    """Run a CCSD(T) (or PBE/SCAN) calculation on an Atoms object, with given basis and kwargs.

    Args:
        idx (int): Index of molecule in its trajectory.
        atoms (ase.Atoms): The Atoms object that will be used to generate the mol object for the pyscf calculations
        basis (str): Basis type to feed to pyscf

    Raises:
        ValueError: Raised if PBC are flagged but no cell length specified.

    Returns:
        result (ase.Atoms): The Atoms object with result.calc.results['energy'] appropriately set for future use.
    """
    result = Atoms(atoms)
    print('======================')
    print("Atoms Info")
    print(atoms.info)
    print('======================')
    print('======================')
    print("kwargs Summary")
    print(kwargs)
    print('======================')

    pos = atoms.positions
    spec = atoms.get_chemical_symbols()
    try:
        PBCALC = np.any(atoms.pbc)
    except:
        PBCALC = None
    if not PBCALC:
        sping = get_spin(atoms)
    else:
        sping = 0
    # As implemented in xcdiff/dpyscf prep_data, this is not used
    # Spin is used instead.
    pol = atoms.info.get('pol', None)
    pol = None
    if kwargs.get('forcepol', False):
        pol = True
    owc = kwargs.get('owcharge', False)
    charge = atoms.info.get('charge', GCHARGE)
    print("===========================")
    print("Option Summary: {} ---> {}".format(atoms.symbols, atoms.get_chemical_formula()))
    print("Spin: {}, Polarized: {}, Charge: {}".format(sping, pol, charge))
    print("===========================")
    if owc:
        if charge != GCHARGE:
            print("OVERWRITING GCHARGE WITH ATOMS.INFO['CHARGE']. {} -> {}".format(GCHARGE, charge))

    mol_input = [[s, p] for s, p in zip(spec, pos)]
    if kwargs.get('rerun', True) and os.path.isfile('{}_{}.traj'.format(idx, atoms.symbols)):
        print("reading in traj file {}_{}.traj".format(idx, atoms.symbols))
        result = read('{}_{}.traj'.format(idx, atoms.symbols))
        return result
    print('Generating mol {} with charge {}'.format(idx, charge))
    if kwargs.get('custom_xc', False):
        print('CUSTOM XC SPECIFIED; WILL OVERWITE eval_xc')
        molgen = False
        scount = 0
        initspin = sping
        while not molgen:
            try:
                mol = gtoa.Mole(atom=mol_input, basis=basis, spin=sping-scount, charge=charge)
                mol.build()
                molgen = True
            except RuntimeError:
                # spin disparity somehow, try with one less until 0
                if initspin > 0:
                    print("RuntimeError. Trying with reduced spin.")
                    scount += 1
                elif initspin == 0:
                    print("RuntimeError. Trying with increased spin.")
                    scount -= 1
                if sping-scount < 0:
                    raise ValueError
        print('S: ', mol.spin)
        print(f'generated pyscfad mol: {type(mol), mol}')
    elif kwargs['PBC'] == False and not PBCALC:
        molgen = False
        scount = 0
        while not molgen:
            try:
                mol = gto.M(atom=mol_input, basis=basis, spin=sping-scount, charge=charge)
                molgen = True
            except RuntimeError:
                # spin disparity somehow, try with one less until 0
                print("RuntimeError. Trying with reduced spin.")
                scount += 1
                if sping-scount < 0:
                    raise ValueError
        print('S: ', mol.spin)
    elif kwargs['PBC'] == True and not PBCALC:
        print('PBC CALCULATION BY FLAG')
        PBCALC = True
        if kwargs['L'] == None:
            raise ValueError('Cannot specify PBC without cell length')
        print('Generating periodic cell of length {}'.format(kwargs['L']))
        cell = np.eye(3)*kwargs['L']
        if kwargs['pseudo'] == None:
            mol = pbc.gto.M(a=cell, atom=mol_input, basis=basis, charge=charge, spin=spin)
        elif kwargs['pseudo'] == 'pbe':
            print('Assigning pseudopotential: GTH-PBE to all atoms')
            mol = pbc.gto.M(a=cell, atom=mol_input, basis=basis, charge=charge, pseudo='gth-pbe')
    elif PBCALC:
        print('PBC CALCULATION BY ATOMS')
        cell = np.array(atoms.cell)
        mol = gtop.Cell(a=cell, rcut=0.1, atom=mol_input, basis=basis,
                        charge=charge, pseudo='gth-pbe', verbose=9, spin=0)
        mol.exp_to_discard = 0.1
        mol.build()
    if kwargs.get('testgen', None):
        print('{} Generated.'.format(atoms.get_chemical_formula()))
        return 0
    if kwargs['XC'].lower() == 'custom_xc':
        print('CUSTOM CALCULATION COMMENCING.....')
        print(type(mol), mol)
        # pyscfad only has spin-restricted DFT right now
        method_gen = False
        while not method_gen:
            try:
                mf = dfta.RKS(mol)
                method_gen = True
            except:
                if mol.spin > 0:
                    mol.spin = mol.spin - 1
        method = dfta.RKS
        print(f'method, mf = {method}, {mf}')
        if kwargs.get('chk', True):
            mf.set(chkfile='{}_{}.chkpt'.format(idx, atoms.symbols))
        if kwargs['restart']:
            print("Restart Flagged -- Setting mf.init_guess to chkfile")
            mf.init_guess = '{}_{}.chkpt'.format(idx, atoms.symbols)
        print('Running get_init_guess()')
        init_dm = mf.get_init_guess()
        # do short calculation to generate necessary ingredients to start
        print('Running short calculation to get ingredients for network run...')
        mf0 = method(mol)
        mf0.max_cycle = -1
        mf0.conv_tol = 1e-5
        mf0.kernel()
        print('Starting kernel calculation complete.')
        # evxc = xce.pyscf.generate_network_eval_xc(mf0, init_dm, kwargs['custom_xc_net'])
        evxc = xce.pyscf.generate_network_eval_xc(mf, init_dm, kwargs['custom_xc_net'])
        mf.grids.level = kwargs.get('gridlevel', 3)
        mf.max_cycle = 5
        mf.max_memory = 64000
        print("Running {} calculation".format(kwargs['XC']))
        mf.define_xc_(evxc, 'GGA')
        xc_start = time()
        try:
            mf.kernel()
            e_tot = mf.e_tot
            dmp = mf.make_rdm1()
            print('Kernel calculation complete: total energy = {}'.format(e_tot))
            print('Kernel calculation complete: density matrix shape = {}'.format(dmp.shape))
            if mf.e_tot >= 0 and kwargs.get('above0mo', False):
                print('non-negative total energy; trying to rewrite with mo_energy of homo')
                print(f'mf.e_tot = {mf.e_tot}')
                print(f'mf.mo_occ = {mf.mo_occ}\nmf.mo_energy={mf.mo_energy}')
                homo_i = jnp.max(jnp.nonzero(mf.mo_occ, size=init_dm.shape[0])[0])
                homo_e = mf.mo_energy[homo_i]
                print(f'homo_e = {homo_e}')
                mf.e_tot = homo_e
            elif mf.e_tot >= 0:
                print(f'NON-NEGATIVE ENERGY DETECTED.\n{str(atoms.symbols), mol, mf}\nENERGY={mf.e_tot}')
                raise
            result.calc = SinglePointCalculator(result)
            result.calc.results = {'energy': mf.e_tot, 'dm': mf.make_rdm1()}
            xc_time = time() - xc_start
            with open('timing', 'a') as tfile:
                tfile.write('{}\t{}\t{}\t{}\t{}\n'.format(epoch, idx, atoms.symbols, mf.xc.upper(), xc_time))
            if kwargs['df'] == True:
                print('Default auxbasis', mf.with_df.auxmol.basis)
            with open('progress', 'a') as progfile:
                progfile.write('{}\t{}\t{}\t{}\n'.format(epoch, idx, atoms.symbols, e_tot))
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print('Kernel calculation failed, perhaps hydrogen is acting up or there is another issue')
            print('Trying with UHF')
            def vgf(x): return kwargs['custom_xc_net'](x, mf._numint.eval_ao(
                mol, mf.grids.coords, deriv=2), mf.grids.weights, mf=mf, coor=mf.grids.coords)
            mf2 = scfa.UHF(mol)
            mf2.max_cycle = 50
            mf2.max_memory = 64000
            print('Setting network and network_eval1')
            mf2.network = kwargs['custom_xc_net']
            mf2.network_eval = vgf
            print('Running UHF calculation')
            mf2.kernel()
            e_tot = mf2.e_tot
            dmp = mf2.make_rdm1()
            result.calc = SinglePointCalculator(result)
            result.calc.results = {'energy': mf2.e_tot, 'dm': mf2.make_rdm1()}
            xc_time = time() - xc_start
            with open('timing', 'a') as tfile:
                tfile.write('{}\t{}\t{}\t{}\t{}\n'.format(epoch, idx, atoms.symbols, 'pyscfad.scf.UHF', xc_time))
            if kwargs['df'] == True:
                print('Default auxbasis', mf.with_df.auxmol.basis)
            with open('progress', 'a') as progfile:
                progfile.write('{}\t{}\t{}\t{}\n'.format(epoch, idx, atoms.symbols, e_tot))

    return (result, e_tot, dmp)


def load_eqx_XC_model(model_xc_path):

    with open(f"{model_xc_path}.json", "r") as f:
        metadata = json.load(f)
    path = model_xc_path.split('/')
    path = '/'.join(path[:-1])
    model_x = xce.net.load_xcquinox_model(f'{path}/{metadata["xnet"]}')
    model_c = xce.net.load_xcquinox_model(f'{path}/{metadata["cnet"]}')
    model = xce.xc.RXCModel_GGA(model_x, model_c)
    return model


def loadnet_from_strucdir(path, ninput, use=[], xc_full=False):
    sp = path.split('/')
    print('PATH SPLIT: {}'.format(sp))
    if '.eqx' in sp[-1]:
        # directing to the specific checkpoint
        f = sp[-1]
        sdir = sp[-2]
        fullpath = True
    elif 'xc' == sp[-1]:
        # directing to checkpoint
        f = sp[-1]
        sdir = sp[-2]
        fullpath = True
    else:
        sdir = sp[-1]
        f = sorted([i for i in os.listdir(path) if '.eqx' in i], key=lambda x: int(x.split('.')[-1]))[-1]
        fullpath = False

    loadnet = path if fullpath else os.path.join(path, f)
    print('loadnet: {}'.format(loadnet))
    levels = {'gga': 2, 'mgga': 3, 'nl': 4}
    print('SDIR SPLIT: ', sdir.split('_'))
    sd_split = sdir.split('_')
    if len(sd_split) == 4:
        net_type, ndepth, nhidden, level = sdir.split('_')
        ss = None
    elif len(sd_split) == 5:
        net_type, ndepth, nhidden, ss, level = sdir.split('_')
    elif len(sd_split) == 6:
        net_type, ndepth, nhidden, ss, _, level = sdir.split('_')
    DEFSSX = True
    DEFSSC = False
    DEFSS = DEFSSX if net_type == 'x' else DEFSSC
    SPINSCALE = True if ss else DEFSS
    if level == 'gga':
        if net_type == 'x':
            use = use if use else [1]
            thisnet = xce.net.eX(n_input=ninput, n_hidden=int(nhidden), use=use,
                                 depth=int(ndepth), lob=1.804, spin_scaling=SPINSCALE)
        elif net_type == 'c':
            use = use if use else [2]
            thisnet = xce.net.eC(n_input=ninput, n_hidden=int(nhidden), use=use,
                                 depth=int(ndepth), ueg_limit=True, spin_scaling=SPINSCALE)
    elif level == 'mgga':
        if net_type == 'x':
            use = use if use else [1, 2]
            thisnet = xce.net.eX(n_input=ninput, n_hidden=int(nhidden), use=use, depth=int(
                ndepth), ueg_limit=True, lob=1.174, spin_scaling=SPINSCALE)
        elif net_type == 'c':
            use = use if use else []
            thisnet = xce.net.eC(n_input=ninput, n_hidden=int(nhidden), use=use,
                                 depth=int(ndepth), ueg_limit=True, spin_scaling=SPINSCALE)
    elif level == 'nl':
        if net_type == 'x':
            use = use if use else []
            thisnet = xce.net.eX(n_input=ninput, n_hidden=int(nhidden), use=use, depth=int(
                ndepth), ueg_limit=True, lob=1.174, spin_scaling=SPINSCALE)
        elif net_type == 'c':
            use = use if use else []
            thisnet = xce.net.eC(n_input=ninput, n_hidden=int(nhidden), use=use,
                                 depth=int(ndepth), ueg_limit=True, spin_scaling=SPINSCALE)
    if not xc_full:
        thisnet = eqx.tree_deserialise_leaves(loadnet, thisnet)
    return thisnet, levels[level], loadnet


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('xyz', action='store', help ='Path to .xyz/.traj file containing list of configurations')
    parser.add_argument('-charge', '-c', action='store', type=int, help='Net charge of the system', default=0)
    parser.add_argument('-fdf', metavar='fdf', type=str, nargs='?', default='')
    parser.add_argument('-basis', metavar='basis', type=str, nargs='?',
                        default='6-311++G(3df,2pd)', help='basis to use. default 6-311++G(3df,2pd)')
    parser.add_argument('-nworkers', metavar='nworkers', type=int, nargs='?', default=1)
    parser.add_argument('-cmargin', '-cm', type=float, default=10.0,
                        help='the margin to define extent of cube files generated')
    parser.add_argument('-xc', '--xc', type=str, default='pbe', help='Type of XC calculation. Either pbe or ccsdt.')
    parser.add_argument('-pbc', '--pbc', type=bool, default=False, help='Whether to do PBC calculation or not.')
    parser.add_argument('-L', '--L', type=float, default=None, help='The box length for the PBC cell')
    parser.add_argument('-df', '--df', type=bool, default=False,
                        help='Choose whether or not the DFT calculation uses density fitting')
    parser.add_argument('-ps', '--pseudo', type=str, default=None,
                        help='Pseudopotential choice. Currently either none or pbe')
    parser.add_argument('-r', '--rerun', type=bool, default=False,
                        help='whether or not to continue and skip previously completed calculations or redo all')
    parser.add_argument('--ghf', default=False, action="store_true",
                        help='whether to have wrapper guess HF to use or do GHF. if flagged, ')
    parser.add_argument('--serial', default=False, action="store_true", help="Run in serial, without DASK.")
    parser.add_argument('--overwrite_gcharge', default=False, action="store_true",
                        help='Whether to try to overwrite specified CHARGE -c if atom.info has charge.')
    parser.add_argument('--restart', default=False, action="store_true",
                        help='If flagged, will use checkfile as init guess for calculations.')
    parser.add_argument('--forcepol', default=False, action='store_true',
                        help='If flagged, all calculations are spin polarized.')
    parser.add_argument('--testgen', default=False, action='store_true',
                        help='If flagged, calculation stops after mol generation.')
    parser.add_argument('--startind', default=-1, type=int, action='store',
                        help='SERIAL MODE ONLY. If specified, will skip indices in trajectory before given value')
    parser.add_argument('--endind', default=999999999999, type=int, action='store',
                        help='SERIAL MODE ONLY. If specified, will only calculate up to this index')
    parser.add_argument('--atomize', default=False, action='store_true',
                        help='If flagged, will generate predictions for the single-atom components present in trajectory molecules.')
    parser.add_argument('--mf_grid_level', type=int, default=3, action='store',
                        help='Grid level for PySCF(AD) calculation')
    # add arguments for pyscfad network driver
    parser.add_argument('--xc_x_net_path', type=str, default='', action='store',
                        help='Path to the trained xcquinox exchange network to use in PySCF(AD) as calculation driver\nParent directory of network assumed to be of form TYPE_MLPDEPTH_NHIDDEN_LEVEL (e.g. x_3_16_mgga)')
    parser.add_argument('--xc_x_ninput', type=int, action='store',
                        help='Number of inputs the exchange network expects')
    parser.add_argument('--xc_x_use', nargs='+', type=int, action='store', default=[],
                        help='Specify the desired indices for the exchange network to actually use, if not the full range of descriptors.')
    parser.add_argument('--xc_c_net_path', type=str, default='', action='store',
                        help='Path to the trained xcquinox correlation network to use in PySCF(AD) as calculation driver\nParent directory of network assumed to be of form TYPE_MLPDEPTH_NHIDDEN_LEVEL (e.g. c_3_16_mgga)')
    parser.add_argument('--xc_c_ninput', type=int, action='store',
                        help='Number of inputs the correlation network expects')
    parser.add_argument('--xc_c_use', nargs='+', type=int, action='store', default=[],
                        help='Specify the desired indices for the correlation network to actually use, if not the full range of descriptors.')
    parser.add_argument('--xc_xc_net_path', type=str, default='', action='store',
                        help='Path to the trained xcquinox exchange-correlation network to use in PySCF(AD) as calculation driver\nParent directory of network assumed to be of form TYPE_MLPDEPTH_NHIDDEN_LEVEL (e.g. xc_3_16_mgga)')
    parser.add_argument('--xc_xc_level', type=str, default='MGGA', action='store',
                        choices=['GGA', 'MGGA', 'NONLOCAL', 'NL'], help='Type of network being loaded (GGA,MGGA,NONLOCAL,NL)')
    parser.add_argument('--xc_xc_ninput', type=int, action='store',
                        help='Number of inputs the exchange-correlation network expects')
    parser.add_argument('--xc_verbose', default=False, action='store_true',
                        help='If flagged, sets verbosity on the network.')
    parser.add_argument('--xc_full', default=False, action='store_true',
                        help='If flagged, will deserialize a saved XC object, as opposed to individual X/C MLP networks.')
    # add arguments for training
    parser.add_argument('--n_steps', action='store', type=int, default=200,
                        help='The number training epochs to go through.')
    parser.add_argument('--singles_start', action='store', type=int, default=4,
                        help='The index at which the single-atom molecules start, assuming they are at the end of a list of trajectories containing molecules and reactions.')
    parser.add_argument('--do_jit', action='store_true', default=False,
                        help='If flagged, will try to utilize JAX-jitting during training')
    parser.add_argument('--train_traj_path', action='store', type=str, default='/home/awills/Documents/Research/xcquinox/scripts/script_data/haunschild_g2/g2_97.traj',
                        help='Location of the data file for use during pre-training')
    parser.add_argument('--train_data_dir', action='store', type=str, default='/home/awills/Documents/Research/xcquinox/scripts/script_data/haunschild_g2/g2_97.traj',
                        help='Location of the data file for use during pre-training')
    parser.add_argument('--debug', action='store_true',
                        help='If flagged, only selects first in the target list for quick debugging purposes.')
    parser.add_argument('--verbose', action='store_true', help='If flagged, activates verbosity flag in the network.')
    parser.add_argument('--init_lr', action='store', type=float, default=5e-2,
                        help='The initial learning rate for the network.')
    parser.add_argument('--lr_decay_start', action='store', type=int, default=50,
                        help='The epoch at which the exponential decay begins.')
    parser.add_argument('--lr_decay_rate', action='store', type=float, default=0.9,
                        help='The decay rate for the exponential decay.')

    args = parser.parse_args()
    setattr(__config__, 'cubegen_box_margin', args.cmargin)
    GCHARGE = args.charge

    atoms = read(args.train_traj_path, ':')
    print("==================")
    print("ARGS SUMMARY")
    print(args)
    print("==================")
    # SNR TODO: cambiar esto
    ATOMIZATION = args.atomize
    if ATOMIZATION:
        print('ATOMIZATION CALCULATION FLAGGED -- GETTING ATOMIC CONSTITUENTS.')
        total_syms = []
        for idx, at in enumerate(atoms):
            total_syms += at.get_chemical_symbols()
        total_syms = sorted(list(set(total_syms)))
        print('SINGLE ATOMS REPRESENTED: {}'.format(total_syms))
        singles = []
        for atm_idx, symbol in enumerate(total_syms):
            print('========================================================')
            print('GROUP {}/{} -- {}'.format(atm_idx, len(total_syms)-1, symbol))
            print('--------------------------------------------------------')

            molsym = symbol
            syms = [symbol]
            pos = [[0, 0, 0]]
            form = symbol
            spin = spins_dict[symbol]

            singles.append(Atoms(form, pos))
        print(f'Singles = {singles}')
        atoms = singles + atoms

    gridmodels = []
    CUSTOM_XC = False
    xcnet = None
    if args.xc_xc_net_path:
        xcnet = load_eqx_XC_model(args.xc_xc_net_path)
        CUSTOM_XC = True
    # S: in our case, it will be always custom_xc, as I want to use our networks.
    input_xc = args.xc if not CUSTOM_XC else 'custom_xc'

    class E_DM_Loss(eqx.Module):
        """ Loss function for the XC model.
        Calculates the loss based on atomization energies and density matrices.

        Useful info:
         - All the dictionaries have the form: 'key' : value
            - key: is a string of the form 'idx_symbol'. Symbol is the chemical symbol of the atom.
            The index depends on which dictionary we have.
            - value: the calculated values.
        """

        def __call__(self, model, refdatatraj, refdatadir, singles_start):
            atoms = read(refdatatraj, ':')
            # args.startind and ia < args.endind - so that we can restart from a specific index
            results = [do_net_calc(0, ia, atoms[ia], basis=args.basis,
                                   margin=args.cmargin,
                                   XC=input_xc,
                                   PBC=args.pbc,
                                   L=args.L,
                                   df=args.df,
                                   pseudo=args.pseudo,
                                   rerun=args.rerun,
                                   owcharge=args.overwrite_gcharge,
                                   restart=args.restart,
                                   forcepol=False,
                                   testgen=args.testgen,
                                   gridlevel=args.mf_grid_level,
                                   atomize=True,
                                   custom_xc=CUSTOM_XC,
                                   custom_xc_net=model) for ia in range(len(atoms)) if ia >= args.startind and ia < args.endind]
            # results is a tuple: (res[0], res[1], res[2])
            # res[0] is the Atoms object with the results - from ASE, so it has .info, .symbols
            # res[1] is the energy in Hartree
            # res[2] is the density matrix in Hartree

            # e_dct: dictionary with gs energies
            # -> keys: 'idx_symbol' (eg, '0_H2', '3_H2O', '4_NaCl'), values: energy in Hartree
            e_dct = {str(idx)+'_'+str(res[0].symbols): res[1] for idx, res in enumerate(results)}

            # dm_dct: dictionary with density matrices
            # -> keys: 'idx_symbol', values: density matrix in Hartree
            dm_dct = {str(idx)+'_'+str(res[0].symbols): res[2] for idx, res in enumerate(results)}

            # single_e_dct: dictionary with single atom energies
            # -> keys: 'symbol', just an atomic species*, values: energy in Hartree
            # *Filters atomic species (e.g., 'H', 'O') that are:
            #  From a certain starting index onward (singles_start) and recognized in spins_dict
            single_e_dct = {k.split('_')[1]: e_dct[k] for k in e_dct.keys() if int(
                k.split('_')[0]) >= singles_start and k.split('_')[1] in spins_dict.keys()}

            # ae_e_dct: dictionary with energies of molecules (that are not obtained via reactions)
            # -> keys: 'idx_symbol', values: energy in Hartree
            # This dictionary is used to calculate atomization energies.
            ae_e_dct = {str(idx)+'_'+str(res[0].symbols): res[1] for idx, res in enumerate(results)
                        if not res[0].info.get('reaction', None) and str(res[0].symbols) not in single_e_dct.keys()}

            # rxn_e_dct: dictionary with energies of compounds that are obtained via reactions
            # -> keys: 'idx_symbol', values: energy in Hartree
            # This dictionary is used to calculate reaction energies.
            rxn_e_dct = {str(idx)+'_'+str(res[0].symbols): res[1] for idx,
                         res in enumerate(results) if type(res[0].info.get('reaction', None)) == int}

            # rxn_part_dct: dictionary with the parts of the reactions
            # -> keys: 'idx_symbol', values: list of strings with the parts of the reaction
            # e.g., '3_H2O' -> ['2_H2', '1_O']
            # This dictionary is used to calculate reaction energies, as we have:
            # for subcomp in rxn_part_dct[k]: rxnloss -= e_dct[subcomp]
            rxn_part_dct = {}
            for k in rxn_e_dct.keys():
                kidx, ksym = k.split('_')
                kidx = int(kidx)
                ncomps = results[kidx][0].info.get('reaction')
                rxncomps = [str(kidx-i)+'_'+str(results[kidx-i][0].symbols) for i in range(1, ncomps+1)]
                rxn_part_dct[k] = rxncomps
            print('------------------------------------------')
            print('DICTIONARY SUMMARIES: ')
            print(f'e_dct = {e_dct}')
            print(f'single_e_dct = {single_e_dct}')
            print(f'ae_e_dct = {ae_e_dct}')
            print(f'rxn_e_dct = {rxn_e_dct}')
            print(f'rxn_part_dct = {rxn_part_dct}')
            print('------------------------------------------')
            ae_losses = []
            dm_losses = []
            ae_losses = jnp.array(ae_losses)
            dm_losses = jnp.array(dm_losses)

            # ATOMIZATION ENERGY (ATE) LOSS CALCULATION
            # ATE = energy_mol - sum(single_atom_energies)
            # Loss: |ATE-REFATE|
            for k, v in ae_e_dct.items():
                kidx, at = k.split('_')
                kidx = int(kidx)

                # REFATE = reference atomization energy
                REFATE = atoms[kidx].info['target_energy']
                # Each atom in the molecule
                syms = atoms[kidx].get_chemical_symbols()
                print(f'{kidx}_{at} energy : {v}')
                ATE = v
                # Substract to the total energy the single atom energies
                # To obtain the real ATE.
                for sym in syms:
                    ATE -= single_e_dct[sym]
                print(f'{kidx}_{at} atomization energy : {ATE}')
                print(f'{kidx}_{at} reference atomization energy : {REFATE}')
                ae_losses = jnp.append(ae_losses, ATE-REFATE)

            # REACTION LOSS (rxn) CALCULATION
            # rxn = energy_compound - sum(energy_parts)
            for k, v in rxn_e_dct.items():
                print(f'Reaction -- {k} : {rxn_part_dct[k]}')
                rxnloss = v
                # rxn_part_dct[k] is a list of strings with the parts of the reaction
                for subcomp in rxn_part_dct[k]:
                    rxnloss -= e_dct[subcomp]
                print(f'Reaction -- {k} : Loss = {rxnloss}')
                ae_losses = jnp.append(ae_losses, rxnloss)

            # DENSITY MATRIX LOSS CALCULATION
            # Loss: |DM-REFDM| for each density matrix
            for k, v in dm_dct.items():
                print('Density matrix losses: {}'.format(k))
                refdm = np.load(os.path.join(refdatadir, k+'.dm.npy'))
                refdm = jnp.array(refdm)
                print(f'{k}: Reference density matrix shape = {refdm.shape}')
                print(f'{k}: Predicted density matrix shape = {v.shape}')
                if refdm.shape != v.shape:
                    print('Reference and predicted density matrix shape mismatch. Reducing spin channels of non-matching.')
                    if len(refdm.shape) == 3:
                        refdm = refdm[0] + refdm[1]
                        print('New reference density matrix shape: {}'.format(refdm.shape))
                    if len(v.shape) == 3:
                        v = v[0] + v[1]
                        print('New predicted density matrix shape: {}'.format(v.shape))
                dm_losses = jnp.append(dm_losses, jnp.sum((refdm-v)**2))
            print('AE Losses: {}'.format(ae_losses))
            print('DM Losses: {}'.format(dm_losses))
            return jnp.sqrt(jnp.mean((ae_losses**2))) + jnp.sqrt(jnp.mean((dm_losses**2)))

    if not args.rerun:
        print('beginning new progress file')
        with open('progress', 'w') as progfile:
            progfile.write('#idx\tatoms.symbols\tetot  (Har)\tehf  (Har)\teccsd  (Har)\teccsdt  (Har)\n')
        print('beginning new nonconverged file')
        with open('unconv', 'w') as ucfile:
            ucfile.write('#idx\tatoms.symbols\tetot  (Har)\tehf  (Har)\teccsd  (Har)\teccsdt  (Har)\n')
        print('beginning new timing file')
        with open('timing', 'w') as tfile:
            tfile.write('#idx\tatoms.symbols\tcalc\ttime (s)\n')
        print("SERIAL CALCULATIONS")

        scheduler = optax.exponential_decay(init_value=args.init_lr, transition_begin=args.lr_decay_start,
                                            transition_steps=args.n_steps-args.lr_decay_start, decay_rate=args.lr_decay_rate)
        optimizer = optax.adam(learning_rate=scheduler)

        trainer = xce.train.xcTrainer(model=xcnet, optim=optimizer, steps=args.n_steps,
                                      loss=E_DM_Loss(), do_jit=args.do_jit, logfile='trlog')

        cpus = jax.devices(backend='cpu')
        with jax.default_device(cpus[0]):
            newm = trainer(1, trainer.model, [args.train_traj_path], [args.train_data_dir], [args.singles_start])
