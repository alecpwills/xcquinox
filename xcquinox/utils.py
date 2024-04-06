from pyscfad import gto
import equinox as eqx
import jax.numpy as jnp
import jax

def pad_array(arr, max_arr, shape=None, device=jax.devices('cpu')[0]):
    """
    Utility function designed to pad an array to a given maximum size, for use during jitted forward passes.

    Since JAX jit compilations re-compile everytime a new array shape is passed, padding is a tactic to avoid this memory consumption.

    :param arr: The original array meant to be padded
    :type arr: jax.Array
    :param max_arr: The array whose size is to be emulated, or an empty value if manually specifying `shape`
    :type max_arr: jax.Array
    :param shape: The array shape you want to pad to, if known a priori, defaults to None
    :type shape: tuple of ints, optional
    :param device: Where to place the padded arrays, defaults to jax.devices('cpu')[0]
    :type device: jax.Device, optional
    :return: Padded version of `arr`
    :rtype: jax.Array
    """
    if not shape:
        dims = max_arr.shape
    else:
        dims = shape
    cdims = arr.shape
    print(dims, cdims)
    extensions = [dims[i]-cdims[i] for i in range(len(cdims))]
    after_pad = [(0, e) for e in extensions]
    with jax.default_device(device):
        arr = jnp.pad(arr, after_pad)
    return arr

def pad_array_list(arrlst, device=jax.devices('cpu')[0]):
    """
    Automatically finds maximum size from a list of arrays, and pads all arrays to that size

    :param arrlst: List of jax.Arrays with varying sizes
    :type arrlst: list
    :param device: Where to place the newly padded arrays, defaults to jax.devices('cpu')[0]
    :type device: jax.Device, optional
    :return: List of the padded arrays
    :rtype: list of jax.Array
    """
    ndims = len(arrlst[0].shape)
    maxdims = []
    for i in range(ndims):
        this_dim = [arr.shape[i] for arr in arrlst]
        maxdims.append(max(this_dim))
    newarrlst = []
    for arr in arrlst:
        with jax.default_device(device):
            newarrlst.append(pad_array(arr, arr, shape=maxdims))
    return newarrlst


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

def get_dm_moe(dm, eri, vxc_grad_func, mo_occ, hc, s, ogd, alpha0=0.7):
    """
    Generates the DM, molecular orbital energies, and molecular orbital coefficients for a model

    _extended_summary_

    :param dm: The initial DM starting point for the calculation.
    :type dm: jax.Array
    :param eri: The electron repulsion integrals to use in generating the effective potential
    :type eri: jax.Array
    :param vxc_grad_func: A function F(dm) = E, with which the derivative is taken to generate Vxc
    :type vxc_grad_func: function
    :param mo_occ: The molecular orbital occupation numbers for the molecule
    :type mo_occ: jax.Array
    :param hc: The molecule's core Hamiltonian
    :type hc: jax.Array
    :param s: The molecule's overlap matrix
    :type s: jax.Array
    :param ogd: The molecule's original DM dimensions
    :type ogd: tuple/jax.Array
    :param alpha0: The mixing parameter used for the one-shot DM creation, defaults to 0.7
    :type alpha0: float, optional
    :return: tuple of (density matrix, molecular orbital energies, molecular orbital coefficients)
    :rtype: tuple of jax.Array
    """
    L = jnp.eye(dm.shape[-1])
    scaling = jnp.ones([dm.shape[-1]]*2)
    dm_old = dm
    def true_func(vxc):
        vxc.at[1].set(jnp.zeros_like(vxc[1]))
        return vxc
    def false_func(vxc):
        return vxc
    alpha = jnp.power(alpha0, 0)+0.3
    beta = (1-alpha)
    dm = alpha * dm + beta * dm_old
    dm_old = dm
    veff = get_veff()(dm, eri)
    vxc = jax.grad(vxc_grad_func)(dm)
    if vxc.ndim > 2:
        vxc = jnp.einsum('ij,xjk,kl->xil',L,vxc,L.T)
        vxc = jnp.where(jnp.expand_dims(scaling, 0) > 0 , vxc, jnp.expand_dims(scaling,0))
    else:
        vxc = jnp.matmul(L,jnp.matmul(vxc ,L.T))
        vxc = jnp.where(scaling > 0 , vxc, scaling)
    
    jax.lax.cond(jnp.sum(mo_occ) == 1, true_func, false_func, vxc)
    
    veff += vxc
    f = get_fock()(hc, veff)
    mo_e, mo_c = eig()(f+1e-6*jax.random.uniform(key=jax.random.PRNGKey(92017), shape=f.shape), s, ogd)
    dm = make_rdm1()(mo_c, mo_occ)
    return dm, mo_e, mo_c


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

class get_rho(eqx.Module):
    def __init__(self):
        """
        A :class:`equinox.Module` object whose forward pass constructs the density on the grid, given input density matrix and atomic orbital evaluations.

        """
        super().__init__()


    def __call__(self, dm, ao_eval):
        """
        Forward pass computing the density on the grid, projecting the density matrix onto the atomic orbitals

        :param dm: Density matrix
        :type dm: jax.Array
        :param ao_eval: Atomic orbitals evaluated on the grid, NOT higher order derivatives.
        :type ao_eval: jax.Array
        :return: Density on the grid
        :rtype: jax.Array
        """
        if dm.ndim == 2:
            rho = jnp.einsum('ij,ik,jk->i', ao_eval, ao_eval, dm)
        else:
            rho = jnp.einsum('ij,ik,xjk->xi', ao_eval, ao_eval, dm)
        return rho


class energy_tot(eqx.Module):

    def __init__(self):
        """
        A :class:`equniox.Module` object whose forward pass constructs the total energy: (electron-electron + electron-ion; ion-ion not included)

        """
        super().__init__()

    def __call__(self, dm, hcore, veff):
        """
        Forward pass contraction to find total electron energy (e-e + e-ion)

        :param dm: Density matrix
        :type dm: jax.Array
        :param hcore: Core Hamiltonian
        :type hcore: jax.Array
        :param veff: Effective potential for the electrons
        :type veff: jax.Array
        :return: Total energy
        :rtype: jax.Array/float
        """
        return jnp.expand_dims(jnp.sum((jnp.einsum('...ij,ij', dm, hcore) + .5*jnp.einsum('...ij,...ij', dm, veff))), 0)


class get_veff(eqx.Module):
    def __init__(self):
        """
        A :class:`equniox.Module` object whose forward pass constructs the one-electron effective potential (not including local xc-potential, to be found with the network)

        """
        super().__init__()

    def __call__(self, dm, eri):
        """
        Forward pass, constructing the classical parts of the effective potential

        :param dm: Density matrix
        :type dm: jax.Arary
        :param eri: Electron repulsion integral tensor
        :type eri: jax.Array
        :return: Effective potential
        :rtype: jax.Array
        """
        J = jnp.einsum('...ij,ijkl->...kl',dm, eri)

        if J.ndim == 3:
            return J[0] + J[1]
        else:
            return J        

class get_fock(eqx.Module):
    def __init__(self):
        """
        A :class:`equniox.Module` object whose forward pass constructs the Fock matrix

        """
        super().__init__()

    def __call__(self, hc, veff):
        """
        Returns the Fock matrix, the sum of the core Hamiltonian and the effective potential (including xc-potential)

        :param hc: Core Hamiltonian
        :type hc: jax.Array
        :param veff: Effective Potential (including the xc-contribution)
        :type veff: jax.Array
        :return: Fock matrix
        :rtype: jax.Array
        """
        return hc+veff

class get_hcore(eqx.Module):
    def __init__(self):
        """
        A :class:`equniox.Module` object whose forward pass constructs the core Hamiltonian, given the nuclear potential and kinetic energy matrix

        """
        super().__init__()

    def __call__(self, v, t):
        """
        "Core" Hamiltionian forward pass, includes ion-electron and kinetic contributions

        .. math:: H_{core} = T + V_{nuc-elec}

        :param v: Electron-ion interaction energy
        :type v: jax.Array
        :param t: Kinetic energy
        :type t: jax.Array
        :return: Hcore
        :rtype: jax.Array
        """
        return v+t


class eig(eqx.Module):
    def __init__(self):
        """
        A :class:`equniox.Module` object whose forward pass solves the generalized eigenvalue problem for the Hamiltonian, using Cholesky decomposition

        """
        super().__init__()
        """Solver for generalized eigenvalue problem

        .. todo:: torch.symeig is deprecated for torch.linalg.eigh, replace

        Args:
            h (torch.Tensor): Hamiltionian
            s_chol (torch.Tensor): (Inverse) Cholesky decomp. of overlap matrix S
                                    s_chol = np.linalg.inv(np.linalg.cholesky(S))

        Returns:
            (torch.Tensor, torch.Tensor): Eigenvalues (MO energies), eigenvectors (MO coeffs)
        """

    def __call__(self, h, s_chol, ogdim):
        """
        Solver for generalized eigenvalue problem, finding the molecular energies (eigenvalues) and orbital coefficients (eigenvectors)

        :param h: Hamiltionian
        :type h: jax.Array
        :param s_chol: (Inverse) of the Cholesky decomposition of overlap matrix S
        :type s_chol: jax.Array
        :param ogdim: The original dimensions of the input array, or their current shapes if not padded. JAX jit compilations require static arrays to avoid recompilation with each call, and backpropagating through this eigendecomposition with padded arrays results in NaNs. Arrays are masked to original dimensions to avoid this.
        :type ogdim: tuple, iterable of some original shape
        :return: The eigenvalues (molecular orbital enegies) and eigenvectors (molecular orbital coefficients)
        :rtype: tuple of jax.Arrays
        """
        upper=False
        UPLO = "U" if upper else "L"
        dim = ogdim[0]
        diff = h.shape[0]
        mask_h = h[:dim, :dim]
        mask_s = s_chol[:dim, :dim]
        e, c = jnp.linalg.eigh(jnp.einsum('ij,...jk,kl->...il',mask_s, mask_h, mask_s.T), UPLO=UPLO)
        c = jnp.einsum('ij,...jk ->...ik',mask_s.T, c)
        e = pad_array(e, e, shape=[diff])
        c = pad_array(c, c, shape=(diff,diff))
        return e, c
