from pyscfad import gto
import equinox as eqx
import jax.numpy as jnp
import jax

def ase_atoms_to_mol(atoms, basis='6-311++G(3df,2pd)', charge=0, spin=None):
    """
    Converts an ASE.Atoms object into a PySCF(ad).gto.Mol object


    :param atoms: ASE.Atoms object of a single molecule/system
    :type atoms: :class:`ASE.Atoms`
    :param basis: Basis set to assign in PySCF(AD), defaults to '6-311++G(3df,2pd)'
    :type basis: str, optional
    :param charge: Global charge of molecule, defaults to 0
    :type charge: int, optional
    :param spin: Specify spin if desired. Defaults to None, which has PySCF guess spin based on electron number/occupation. Use with care.
    :type spin: int, optional
    :return: A string of the molecule's name, and the Mole object for pyscfad to work with.
    :rtype: (str, :class:`pyscfad.gto.Mole`)
    """
    pos = atoms.positions
    spec = atoms.get_chemical_symbols()
    name = atoms.get_chemical_formula()

    mol_input = [[ispec, ipos] for ispec,ipos in zip(spec,pos)]
    c = atoms.info.get('charge', charge)
    s = atoms.info.get('spin', spin)
    mol = gto.Mole(atom=mol_input, basis=basis, spin=s, charge=c)

    return name, mol

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
    """
    Returns single-atom, ground-state spin configurations, since pyscf does not guess this correctly.

    Relies on xcquinox.utils.spins_dict, a hard-coded dictionary populated with atom names and associated spins.

    :param at: :class:`ase.Atoms` object, with an associated :class:`ase.Atoms.info` dictionary optionally containing keys 'spin', 'radical', or 'openshell' and associated integer pair. If specified, returns that value. Otherwise, uses the value in spins_dict for single atoms.
    :type at: :class:`ase.Atoms`
    :return: The guessed spin value
    :rtype: int
    """
    #tries to use information present in the at.info dictionary to guess spin
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
            #radicals are species with a single unpair electron
            print('Radical specified in atom.info["name"], assuming spin 1.')
            spin = 1
        elif at.info.get('openshell', None):
            print("Openshell specified in atom info, attempting spin 2.")
            spin = 2
        else:
            print("No specifications in atom info to help, assuming no spin.")
            spin = 0
    return spin

class make_rdm1(eqx.Module):

    def __init__(self):
        """
        A :class:`equinox.Module` object whose forward pass constructs the 1-electron density matrix, given input molecular coefficients and occupations.

        """
        super().__init__()

    def __call__(self, mo_coeff, mo_occ):
        """
        Forward pass calculating one-particle reduced density matrix.

        .. todo:: Test the spin-polarized case

        :param mo_coeff: Molecular orbital coefficients
        :type mo_coeff: jax.Array
        :param mo_occ: Molecular orbital occupation numbers
        :type mo_occ: jax.Array
        :return: The RDM1
        :rtype: jax.Array
        """
        if mo_coeff.ndim == 3:
            print('Spin-polarized make_rdm1()')
            mocc_a = jnp.where(mo_occ[0] > 0, mo_coeff[0], 0)
            mocc_b = jnp.where(mo_occ[1] > 0, mo_coeff[1], 0)
            
            def true_func(mocc_a, mocc_b, mo_occ, mo_coeff):
                return jnp.stack([jnp.einsum('ij,jk->ik', mocc_a*jnp.where(mo_occ[0] > 0, mo_occ[0], 0), mocc_a.T),
                                    jnp.einsum('ij,jk->ik', mocc_b*jnp.where(mo_occ[1] > 0, mo_occ[1], 0), mocc_b.T)],axis=0)
            def false_func(mocc_a, mocc_b, mo_occ, mo_coeff):
                return jnp.stack([jnp.einsum('ij,jk->ik', mocc_a*jnp.where(mo_occ[0] > 0, mo_occ[0], 0), mocc_a.T),
                                    jnp.zeros_like(mo_coeff)[0]],axis=0)

            retarr = jax.lax.cond(jnp.sum(mo_occ[1]) > 0, true_func, false_func, mocc_a, mocc_b, mo_occ, mo_coeff)
            return retarr

        else:
            print('Spin unpolarized make_rdm1()')
            mocc = jnp.where(mo_occ>0, mo_coeff, 0)
            return jnp.einsum('ij,jk->ik', mocc*jnp.where(mo_occ > 0, mo_occ, 0), mocc.T)
