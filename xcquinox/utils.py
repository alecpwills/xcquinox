from pyscfad import gto

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
