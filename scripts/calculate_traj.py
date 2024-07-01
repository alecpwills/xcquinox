from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Hartree
from ase.io import read, write
from time import time
import numpy as np

#pure pyscf imports
from pyscf import dft
from pyscf import gto, scf, cc
from pyscf.scf import hf, uhf
from pyscf import pbc
from pyscf.pbc import scf as scfp
from pyscf.pbc import gto as gtop
from pyscf.pbc import dft as dftp
from pyscf.pbc import cc as ccp

#pyscfad imports
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
from dask.distributed  import Client, LocalCluster
import os
from pyscf.tools import cubegen
from opt_einsum import contract
from pyscf import __config__
import pickle

import equinox as eqx
import jax.numpy as jnp
import xcquinox as xce
import os, optax, jax

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

#%%
def write_mycc(idx, atoms, mycc, result, mycc2=None):
    """Write out the information in a PySCF kernel that's been evaluated.
    The assumption is that this comes from a CC calculation, as we pass 'ao_repr=True' to make_rdm1, as the default for CC outputs the DM in the MO basis.

    Outputs will be named '{idx}_{atoms.symbols}.x'

    #TODO: refactor, results is already the atoms object
    Args:
        idx (int): Index of molecule in its trajectory. 
        atoms (ase.Atoms): The Atoms object to generate the symbols
        mycc (cc object): The CC kernel that was used to generate the results in result
        result (ase.Atoms): Another Atoms object, whose calc.results['energy'] has been set appropriately
    """
    #if len(mycc.mo_occ.shape) == 2:
        #if array shape length is 2, it is spin polarized -- (2, --) one channel for each spin
        #transpose in this manner -- first dimension retains spin component, just transpose indiv. mo_coeffs
    #    dm = contract('xij,xj,xjk -> xik', mycc.mo_coeff, mycc.mo_occ, np.transpose(mycc.mo_coeff, (0, 2, 1)))
    #else:
    #    dm = contract('ij,j,jk -> ik', mycc.mo_coeff, mycc.mo_occ, mycc.mo_coeff.T)
    #To be consistent with general mf.make_rdm1(), which from the pyscf code comments makes it in AO rep
    #CC calculations have make_rdm1() create it in the MO basis.
    dm = mycc.make_rdm1(ao_repr=True)
    #TODO: Make cubegen grid density optional
    #cubegen.density(mol,'{}.cube'.format(idx), dm, nx=100, ny=100, nz=100,
    #                margin=kwargs['margin'])
    lumo = np.where(mycc.mo_occ == 0)[0][0]
    homo = lumo - 1
    #TODO: Make cubegen HOMO/LUMO optional
    #cubegen.orbital(mol, '{}_lumo.cube'.format(idx), mycc.mo_coeff[lumo],nx=100, ny=100, nz=100,
    #                margin=kwargs['margin'])
    #cubegen.orbital(mol, '{}_homo.cube'.format(idx), mycc.mo_coeff[homo], nx=100, ny=100, nz=100,
    #                margin=kwargs['margin'])
    write('{}_{}.traj'.format(idx, atoms.symbols), result)
    np.save('{}_{}.dm'.format(idx, atoms.symbols), dm)
    np.save('{}_{}.mo_occ'.format(idx, atoms.symbols), mycc.mo_occ)
    np.save('{}_{}.mo_coeff'.format(idx, atoms.symbols), mycc.mo_coeff)
    try:
        np.save('{}_{}.mo_energy'.format(idx, atoms.symbols), mycc.mo_energy)
    except:
        np.save('{}_{}.mo_energy'.format(idx, atoms.symbols), mycc2.mo_energy)
    write('{}_{}.traj'.format(idx, atoms.symbols), result)

#spins for single atoms, since pyscf doesn't guess this correctly.
spins_dict = {
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
    'S': 2,
    'Ar':0, #noble
    'Br':1, #one unpaired electron
    'Ne':0, #noble
    'Sb':3, #same column as N/P
    'Bi':3, #same column as N/P/Sb
    'Te':2, #same column as O/S
    'I':1 #one unpaired electron
}


def get_spin(at):
    #if single atom and spin is not specified in at.info dictionary, use spins_dict
    print('======================')
    print("GET SPIN: Atoms Info")
    print(at)
    print(at.info)
    print('======================')
    if ( (len(at.positions) == 1) and not ('spin' in at.info) ):
        print("Single atom and no spin specified in at.info")
        spin = spins_dict[str(at.symbols)]
    else:
        print("Not a single atom, or spin in at.info")
        if type(at.info.get('spin', None)) == type(0):
            #integer specified in at.info['spin'], so use it
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



def do_ccsdt(idx,atoms,basis, **kwargs):
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
    #As implemented in xcdiff/dpyscf prep_data, this is not used
    #Spin is used instead.
    pol = atoms.info.get('pol', None)
    pol = None
    if kwargs.get('forcepol', False):
        pol=True
    owc = kwargs.get('owcharge', False)
    charge = atoms.info.get('charge', GCHARGE)
    ATOMGRID = atoms.info.get('grid_level', None)
    SKIPLEN = kwargs.get('skip_length', 0)
    if len(pos) >= SKIPLEN:
        print("===========================")
        print("Option Summary: {} ---> {}".format(atoms.symbols, atoms.get_chemical_formula()))
        print("Spin: {}, Polarized: {}, Charge: {}".format(sping, pol, charge))
        print("Grid: {}".format(ATOMGRID))
        print("LENGTH OF ATOM LIST IN MOLECULE = {}/GREATER OR EQUAL SKIP_LENGTH = {}".format(len(pos), SKIPLEN))
        print("===========================")
        with open('progress','a') as progfile:
            progfile.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(idx, atoms.symbols, 'SKIPPED','SKIPPED', 'SKIPPED', 'SKIPPED'))
        result.calc = SinglePointCalculator(result)
        result.calc.results = {'energy': None,
                                'e_hf': None,
                                'e_ccsd': None,
                                'e_ccsdt':None}
        return result


    print("===========================")
    print("Option Summary: {} ---> {}".format(atoms.symbols, atoms.get_chemical_formula()))
    print("Spin: {}, Polarized: {}, Charge: {}".format(sping, pol, charge))
    print("Grid: {}".format(ATOMGRID))
    print("===========================")
    if owc:
        if charge != GCHARGE:
            print("OVERWRITING GCHARGE WITH ATOMS.INFO['CHARGE']. {} -> {}".format(GCHARGE, charge))

    mol_input = [[s,p] for s,p in zip(spec,pos)]
    if kwargs.get('rerun',True) and os.path.isfile('{}_{}.traj'.format(idx, atoms.symbols)):
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
                molgen=True
            except RuntimeError:
                #spin disparity somehow, try with one less until 0
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
                molgen=True
            except RuntimeError:
                #spin disparity somehow, try with one less until 0
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
        mol = gtop.Cell(a=cell, rcut=0.1, atom=mol_input, basis=basis, charge=charge, pseudo='gth-pbe', verbose=9, spin=0)
        mol.exp_to_discard = 0.1
        mol.build()
    if kwargs.get('testgen', None):
        print('{} Generated.'.format(atoms.get_chemical_formula()))
        return 0
    if kwargs['XC'] == 'ccsdt':
        print('CCSD(T) calculation commencing')
        #If pol specified, it's a bool and takes precedence.
        if type(pol) == bool:
            #polarization specified, UHF
            if pol:
                if not PBCALC:
                    mf = uhf.UHF(mol)
                else:
                    mf = scfp.UHF(mol)
            #specified to not polarize, RHF
            else:
                if not PBCALC:
                    mf = hf.RHF(mol)
                else:
                    mf = scfp.RHF(mol)
        #if pol is not specified in atom info, spin value used instead
        elif pol == None:
            if (mol.spin != 0):
                if not PBCALC:
                    mf = scf.UHF(mol)
                else:
                    mf = scfp.UHF(mol)
            else:
                if not PBCALC:
                    mf = scf.RHF(mol)
                else:
                    mf = scfp.RHF(mol)
        print("METHOD: ", mf)
        if kwargs.get('chk', True):
            mf.set(chkfile='{}_{}.chkpt'.format(idx, atoms.symbols))
        if kwargs['restart']:
            print("Restart Flagged -- Setting mf.init_guess to chkfile")
            mf.init_guess = '{}_{}.chkpt'.format(idx, atoms.symbols)
        print("Running HF calculation")
        hf_start = time()
        mf.max_memory = 128000
        try:
            mf.run()
        except Exception as e:
            print('error raised: ', e)
            return -1
        hf_time = time() - hf_start
        print("Running CCSD calculation from HF")
        with open('timing', 'a') as tfile:
            tfile.write('{}\t{}\t{}\t{}\n'.format(idx, atoms.symbols, 'HF', hf_time))

        if not PBCALC:
            mycc = cc.CCSD(mf)
        else:
            mycc = ccp.CCSD(mf)

        try:
            ccsd_start = time()
            mycc.kernel()
            ccsd_time = time() - ccsd_start
            with open('timing', 'a') as tfile:
                tfile.write('{}\t{}\t{}\t{}\n'.format(idx, atoms.symbols, 'CCSD', ccsd_time))
        except AssertionError:
            print("CCSD Failed. Stopping at HF")
            result.calc = SinglePointCalculator(result)
            ehf = (mf.e_tot) 
            etot = ehf
            eccsd = None
            eccsdt = None
            result.calc.results = {'energy': etot,
                                    'e_hf': ehf,
                                    'e_ccsd': None,
                                    'e_ccsdt':None}
            with open('progress','a') as progfile:
                progfile.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(idx, atoms.symbols, etot, ehf, eccsd, eccsdt))
            write_mycc(idx, atoms, mf, result, mycc2=mf)
            return result

        print('MO Occ shape: ', mycc.mo_occ.shape)
        print("Running CCSD(T) calculation from CCSD")
        try:
            ccsdt_start = time()
            ccsdt = mycc.ccsd_t()
            ccsdt_time = time() - ccsdt_start
            with open('timing', 'a') as tfile:
                tfile.write('{}\t{}\t{}\t{}\n'.format(idx, atoms.symbols, 'CCSD(T)', ccsdt_time))

        except ZeroDivisionError:
            print("CCSD(T) Failed. DIV/0. Stopping at CCSD")
            result.calc = SinglePointCalculator(result)
            ehf = (mf.e_tot) 
            eccsd = (mycc.e_tot) 
            eccsdt = None
            etot = eccsd
            result.calc.results = {'energy': etot,
                                    'e_hf': ehf,
                                    'e_ccsd': eccsd,
                                    'e_ccsdt':eccsdt}
            with open('progress','a') as progfile:
                progfile.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(idx, atoms.symbols, etot, ehf, eccsd, eccsdt))
            write_mycc(idx, atoms, mycc, result, mycc2=mf)
            return result

        result.calc = SinglePointCalculator(result)
        etot = (mycc.e_tot + ccsdt) 
        ehf = (mf.e_tot) 
        eccsd = (mycc.e_tot) 
        eccsdt = (ccsdt) 
        result.calc.results = {'energy': etot,
                                'e_hf': ehf,
                                'e_ccsd': eccsd,
                                'e_ccsdt': eccsdt}
        with open('progress','a') as progfile:
            progfile.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(idx, atoms.symbols, etot, ehf, eccsd, eccsdt))
        with open('timing', 'a') as tfile:
            tfile.write('{}\t{}\t{}\t{}\n'.format(idx, atoms.symbols, 'TOTAL(HF->END)', time() - hf_start))

        write_mycc(idx, atoms, mycc, result, mycc2=mf)

    elif kwargs['XC'].lower() in ['pbe', 'scan']:
        print('{} calculation commencing'.format(kwargs['XC']))
            #If pol specified, it's a bool and takes precedence.
        if type(pol) == bool:
            #polarization specified, UHF
            if pol:
                if not PBCALC:
                    mf = dft.UKS(mol)
                    method = dft.UKS
                else:
                    mf = dftp.UKS(mol)
                    method = dftp.UKS
            #specified to not polarize, RHF
            else:
                if not PBCALC:
                    mf = dft.RKS(mol)
                    method = dft.RKS
                else:
                    mf = dftp.RKS(mol)
                    method = dftp.RKS
        #if pol is not specified in atom info, spin value used instead
        elif pol == None:
            if (mol.spin != 0):
                if not PBCALC:
                    mf = dft.UKS(mol)
                    method = dft.UKS
                else:
                    mf = dftp.UKS(mol)
                    method = dftp.UKS
            else:
                if not PBCALC:
                    mf = dft.RKS(mol)
                    method = dft.RKS
                else:
                    mf = dftp.RKS(mol)
                    method = dftp.RKS

        print("METHOD: ", mf)
        if kwargs.get('chk', True):
            mf.set(chkfile='{}_{}.chkpt'.format(idx, atoms.symbols))
        if kwargs['restart']:
            print("Restart Flagged -- Setting mf.init_guess to chkfile")
            mf.init_guess = '{}_{}.chkpt'.format(idx, atoms.symbols)

        mf.grids.level = ATOMGRID if ATOMGRID else kwargs.get('gridlevel', 3)
        mf.max_cycle = 100
        mf.max_memory = 64000
        mf.grids.build()
        print("Running {} calculation".format(kwargs['XC']))
        #if kwargs['df'] == False:
        #    mf = dft.RKS(mol)
        #elif kwargs['df'] == True:
        #    print('Using density fitting')
        #    mf = dft.RKS(mol).density_fit()
        mf.xc = '{},{}'.format(kwargs['XC'].lower(), kwargs['XC'].lower())
        xc_start = time()
        mf.kernel()
        if not mf.converged:
            print("Calculation did not converge. Trying second order convergence with PBE to feed into calculation.")
            mfp = method(mol, xc='pbe').newton()
            mfp.kernel()
            print("PBE Calculation complete -- feeding into original kernel.")
            mf.kernel(dm0 = mfp.make_rdm1())
            if not mf.converged:
                print("Convergence still failed -- {}".format(atoms.symbols))
                with open('unconv', 'a') as f:
                    f.write('{}\t{}\t{}\n'.format(idx, atoms.symbols, mf.e_tot))
        xc_time = time() - xc_start
        with open('timing', 'a') as tfile:
            tfile.write('{}\t{}\t{}\t{}\n'.format(idx, atoms.symbols, mf.xc.upper(), xc_time))
        if kwargs['df'] == True:
            print('Default auxbasis', mf.with_df.auxmol.basis)
        result.calc = SinglePointCalculator(result)
        result.calc.results = {'energy':mf.e_tot }
        with open('progress','a') as progfile:
            progfile.write('{}\t{}\t{}\n'.format(idx, atoms.symbols, mf.e_tot ))
        #np.savetxt('{}.m_occ'.format(idx), mf.mo_occ, delimiter=' ')
        #np.savetxt('{}.mo_coeff'.format(idx), mf.mo_coeff, delimiter=' ')
        write_mycc(idx, atoms, mf, result)
    elif kwargs['XC'].lower() == 'custom_xc':
        print('CUSTOM CALCULATION COMMENCING.....')
        print(type(mol), mol)
        #pyscfad only has spin-restricted DFT right now
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
        #do short calculation to generate necessary ingredients to start
        print('Running short calculation to get ingredients for potential non-local network run...')
        mf0 = method(mol)
        mf0.max_cycle = -1
        mf0.conv_tol = 1e-5
        mf0.kernel()
        print('Starting kernel calculation complete.')
        # evxc = xce.pyscf.generate_network_eval_xc(mf0, init_dm, kwargs['custom_xc_net'])
        evxc = xce.pyscf.generate_network_eval_xc(mf, init_dm, kwargs['custom_xc_net'])
        mf.grids.level = ATOMGRID if ATOMGRID else kwargs.get('gridlevel', 3)
        mf.max_cycle = 50
        mf.max_memory = 64000
        print("Running {} calculation".format(kwargs['XC']))
        mf.define_xc_(evxc, 'MGGA')
        xc_start = time()
        mf.mo_coeff = mf0.mo_coeff
        try:
            mf.kernel()
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
            result.calc.results = {'energy' : mf.e_tot}
            xc_time = time() - xc_start
            with open('timing', 'a') as tfile:
                tfile.write('{}\t{}\t{}\t{}\n'.format(idx, atoms.symbols, mf.xc.upper(), xc_time))
            if kwargs['df'] == True:
                print('Default auxbasis', mf.with_df.auxmol.basis)
            with open('progress','a') as progfile:
                progfile.write('{}\t{}\t{}\n'.format(idx, atoms.symbols, result.calc.results['energy']))
            write_mycc(idx, atoms, mf, result)
        except Exception as e:
            print(e)
            print('Kernel calculation failed, perhaps hydrogen is acting up or there is another issue')
            print('Trying with UHF')
            vgf = lambda x: kwargs['custom_xc_net'](x, mf._numint.eval_ao(mol, mf.grids.coords, deriv=2), mf.grids.weights, mf=mf, coor=mf.grids.coords)
            mf2 = scfa.UHF(mol)
            mf2.max_cycle = 50
            mf2.max_memory = 64000
            print('Setting network and network_eval1')
            mf2.network = kwargs['custom_xc_net']
            mf2.network_eval = vgf
            print('Running UHF calculation')
            mf2.kernel()
            result.calc = SinglePointCalculator(result)
            result.calc.results = {'energy' : mf2.e_tot}
            xc_time = time() - xc_start
            with open('timing', 'a') as tfile:
                tfile.write('{}\t{}\t{}\t{}\n'.format(idx, atoms.symbols, 'pyscfad.scf.UHF', xc_time))
            if kwargs['df'] == True:
                print('Default auxbasis', mf.with_df.auxmol.basis)
            with open('progress','a') as progfile:
                progfile.write('{}\t{}\t{}\n'.format(idx, atoms.symbols, result.calc.results['energy']))
            write_mycc(idx, atoms, mf2, result)

    return result

def calculate_distributed(atoms, n_workers = -1, basis='6-311++G(3df,2pd)', **kwargs):
    """_summary_

    Args:
        atoms (_type_): _description_
        n_workers (int, optional): _description_. Defaults to -1.
        basis (str, optional): _description_. Defaults to '6-311++G(3df,2pd)'.

    Returns:
        _type_: _description_
    """
    
    print('Calculating {} systems on'.format(len(atoms)))
    cluster = LocalCluster(n_workers = n_workers, threads_per_worker = 1)
    print(cluster)
    client = Client(cluster)

    futures = client.map(do_ccsdt, np.arange(len(atoms)),atoms,[basis]*len(atoms), **kwargs)
    
    return [f.result() for f in futures]

def loadnet_from_strucdir(path, ninput, use=[]):
    sp = path.split('/')
    print('PATH SPLIT: {}'.format(sp))
    if '.eqx' in sp[-1]:
        #directing to the specific checkpoint
        f = sp[-1]
        sdir = sp[-2]
        fullpath = True
    elif 'xc' == sp[-1]:
        #directing to checkpoint
        f = sp[-1]
        sdir = sp[-2]
        fullpath = True
    else:
        sdir = sp[-1]
        f = sorted([i for i in os.listdir(path) if '.eqx' in i], key = lambda x: int(x.split('.')[-1]))[-1]
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
            thisnet = xce.net.eX(n_input=ninput, n_hidden=int(nhidden), use=use, depth=int(ndepth), lob=1.804, spin_scaling=SPINSCALE)
        elif net_type == 'c':
            use = use if use else [2]
            thisnet = xce.net.eC(n_input=ninput, n_hidden=int(nhidden), use=use, depth=int(ndepth), ueg_limit=True, spin_scaling=SPINSCALE)
    elif level == 'mgga':
        if net_type == 'x':
            use = use if use else [1, 2]
            thisnet = xce.net.eX(n_input=ninput, n_hidden=int(nhidden), use=use, depth=int(ndepth), ueg_limit=True, lob=1.174, spin_scaling=SPINSCALE)
        elif net_type == 'c':
            use = use if use else []
            thisnet = xce.net.eC(n_input=ninput, n_hidden=int(nhidden), use=use, depth=int(ndepth), ueg_limit=True, spin_scaling=SPINSCALE)
    elif level == 'nl':
        if net_type == 'x':
            use = use if use else []
            thisnet = xce.net.eX(n_input=ninput, n_hidden=int(nhidden), use=use, depth=int(ndepth), ueg_limit=True, lob=1.174, spin_scaling=SPINSCALE)
        elif net_type == 'c':
            use = use if use else []
            thisnet = xce.net.eC(n_input=ninput, n_hidden=int(nhidden), use=use, depth=int(ndepth), ueg_limit=True, spin_scaling=SPINSCALE)
    
    thisnet = eqx.tree_deserialise_leaves(loadnet, thisnet)
    return thisnet, levels[level]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('xyz', action='store', help ='Path to .xyz/.traj file containing list of configurations')
    parser.add_argument('-charge', '-c', action='store', type=int, help='Net charge of the system', default=0)
    parser.add_argument('-fdf', metavar='fdf', type=str, nargs = '?', default='')
    parser.add_argument('-basis', metavar='basis', type=str, nargs = '?', default='6-311++G(3df,2pd)', help='basis to use. default 6-311++G(3df,2pd)')
    parser.add_argument('-nworkers', metavar='nworkers', type=int, nargs = '?', default=1)
    parser.add_argument('-cmargin', '-cm', type=float, default=10.0, help='the margin to define extent of cube files generated')
    parser.add_argument('-xc', '--xc', type=str, default='pbe', help='Type of XC calculation. Either pbe or ccsdt.')
    parser.add_argument('-pbc', '--pbc', type=bool, default=False, help='Whether to do PBC calculation or not.')
    parser.add_argument('-L', '--L', type=float, default=None, help='The box length for the PBC cell')
    parser.add_argument('-df', '--df', type=bool, default=False, help='Choose whether or not the DFT calculation uses density fitting')
    parser.add_argument('-ps', '--pseudo', type=str, default=None, help='Pseudopotential choice. Currently either none or pbe')
    parser.add_argument('-r', '--rerun', type=bool, default=False, help='whether or not to continue and skip previously completed calculations or redo all')
    parser.add_argument('--ghf', default=False, action="store_true", help='whether to have wrapper guess HF to use or do GHF. if flagged, ')
    parser.add_argument('--serial', default=False, action="store_true", help="Run in serial, without DASK.")
    parser.add_argument('--overwrite_gcharge', default=False, action="store_true", help='Whether to try to overwrite specified CHARGE -c if atom.info has charge.')
    parser.add_argument('--restart', default=False, action="store_true", help='If flagged, will use checkfile as init guess for calculations.')
    parser.add_argument('--forcepol', default=False, action='store_true', help='If flagged, all calculations are spin polarized.')
    parser.add_argument('--testgen', default=False, action='store_true', help='If flagged, calculation stops after mol generation.')
    parser.add_argument('--startind', default=-1, type=int, action='store', help='SERIAL MODE ONLY. If specified, will skip indices in trajectory before given value')
    parser.add_argument('--endind', default=999999999999, type=int, action='store', help='SERIAL MODE ONLY. If specified, will only calculate up to this index')
    parser.add_argument('--atomize', default=False, action='store_true', help='If flagged, will generate predictions for the single-atom components present in trajectory molecules.')
    parser.add_argument('--mf_grid_level', type=int, default=3, action='store', help='Grid level for PySCF(AD) calculation')
    #add arguments for pyscfad network driver
    parser.add_argument('--xc_x_net_path', type=str, default='', action='store', help='Path to the trained xcquinox exchange network to use in PySCF(AD) as calculation driver\nParent directory of network assumed to be of form TYPE_MLPDEPTH_NHIDDEN_LEVEL (e.g. x_3_16_mgga)')
    parser.add_argument('--xc_x_ninput', type=int, action='store', help='Number of inputs the exchange network expects')
    parser.add_argument('--xc_x_use', nargs='+', type=int, action='store', default=[], help='Specify the desired indices for the exchange network to actually use, if not the full range of descriptors.')
    parser.add_argument('--xc_c_net_path', type=str, default='', action='store', help='Path to the trained xcquinox correlation network to use in PySCF(AD) as calculation driver\nParent directory of network assumed to be of form TYPE_MLPDEPTH_NHIDDEN_LEVEL (e.g. c_3_16_mgga)')
    parser.add_argument('--xc_c_ninput', type=int, action='store', help='Number of inputs the correlation network expects')
    parser.add_argument('--xc_c_use', nargs='+', type=int, action='store', default=[], help='Specify the desired indices for the correlation network to actually use, if not the full range of descriptors.')
    parser.add_argument('--xc_xc_net_path', type=str, default='', action='store', help='Path to the trained xcquinox exchange-correlation network to use in PySCF(AD) as calculation driver\nParent directory of network assumed to be of form TYPE_MLPDEPTH_NHIDDEN_LEVEL (e.g. xc_3_16_mgga)')
    parser.add_argument('--xc_xc_level', type=str, action='store', default='MGGA', help='Rung of Jacobs Ladder the loaded XC functional rests on', choices=["GGA","MGGA","NONLOCAL","NL"])
    parser.add_argument('--xc_verbose', default=False, action='store_true', help='If flagged, sets verbosity on the network.')
    parser.add_argument('--skip_length', action='store', type=int, default=0, help='If flagged, will skip calculating molecules of this size or greater due to memory constraints.', default=0)
    args = parser.parse_args()
    setattr(__config__, 'cubegen_box_margin', args.cmargin)
    GCHARGE = args.charge

    atoms = read(args.xyz, ':')
    print("==================")
    print("ARGS SUMMARY")
    print(args)
    print("==================")
    
    ATOMIZATION = args.atomize
    if ATOMIZATION:
        print('ATOMZATION CALCULATION FLAGGED -- GETTING ATOMIC CONSTITUENTS.')
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
            pos = [[0,0,0]]
            form = symbol
            spin = spins_dict[symbol]

            singles.append(Atoms(form, pos))
        print(f'Singles = {singles}')
        atoms = singles + atoms
    
    gridmodels = []
    CUSTOM_XC = False
    xcnet = None
    if args.xc_x_net_path:
        print('xcquinox network exchange path provided, attempting read-in...')
        xnet, xlevel = loadnet_from_strucdir(args.xc_x_net_path, args.xc_x_ninput, args.xc_x_use)
        gridmodels.append(xnet)
        CUSTOM_XC = True
    if args.xc_c_net_path:
        print('xcquinox network correlation path provided, attempting read-in...')
        cnet, clevel = loadnet_from_strucdir(args.xc_c_net_path, args.xc_c_ninput, args.xc_c_use)
        gridmodels.append(cnet)
        CUSTOM_XC = True
    if args.xc_xc_net_path:
        print('xcquinox network exchange-correlation path provided, attempting read-in...')
        xcnet = xce.xc.get_xcfunc(args.xc_xc_level,
                               args.xc_xc_net_path)
        CUSTOM_XC = True
    elif args.xc_x_net_path or args.xc_c_net_path:
        xcnet = xce.xc.eXC(grid_models=gridmodels, heg_mult=True, level=xlevel, verbose=args.xc_verbose)

    input_xc = args.xc if not CUSTOM_XC else 'custom_xc'

    if not args.rerun:
        print('beginning new progress file')
        with open('progress','w') as progfile:
            progfile.write('#idx\tatoms.symbols\tetot  (Har)\tehf  (Har)\teccsd  (Har)\teccsdt  (Har)\n')
        print('beginning new nonconverged file')
        with open('unconv','w') as ucfile:
            ucfile.write('#idx\tatoms.symbols\tetot  (Har)\tehf  (Har)\teccsd  (Har)\teccsdt  (Har)\n')
        print('beginning new timing file')
        with open('timing','w') as tfile:
            tfile.write('#idx\tatoms.symbols\tcalc\ttime (s)\n')
    print("SERIAL CALCULATIONS")
    results = [do_ccsdt(ia, atoms[ia], basis=args.basis,
                                    margin=args.cmargin,
                                    XC=input_xc,
                                    PBC=args.pbc,
                                    L=args.L,
                                    df=args.df,
                                    pseudo=args.pseudo,
                                    rerun=args.rerun,
                                    owcharge=args.overwrite_gcharge,
                                    restart = args.restart,
                                    forcepol = args.forcepol,
                                    testgen = args.testgen,
                                    gridlevel = args.mf_grid_level,
                                    atomize = args.atomize,
                                    custom_xc = CUSTOM_XC,
                                    custom_xc_net = xcnet,
                                    skip_length = args.skip_length) for ia in range(len(atoms)) if ia >= args.startind and ia < args.endind]

    results_path = 'results.traj'
    write(results_path, results)

# %%
