from pyscfad import gto
import equinox as eqx
import jax.numpy as jnp
import jax
from pyscf import dft
import numpy as np


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

    mol_input = [[ispec, ipos] for ispec, ipos in zip(spec, pos)]
    c = atoms.info.get('charge', charge)
    s = atoms.info.get('spin', spin)
    mol = gto.Mole(atom=mol_input, basis=basis, spin=s, charge=c)

    return name, mol


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
    """
    Returns single-atom, ground-state spin configurations, since pyscf does not guess this correctly.

    Relies on xcquinox.utils.spins_dict, a hard-coded dictionary populated with atom names and associated spins.

    :param at: :class:`ase.Atoms` object, with an associated :class:`ase.Atoms.info` dictionary optionally containing keys 'spin', 'radical', or 'openshell' and associated integer pair. If specified, returns that value. Otherwise, uses the value in spins_dict for single atoms.
    :type at: :class:`ase.Atoms`
    :return: The guessed spin value
    :rtype: int
    """
    # tries to use information present in the at.info dictionary to guess spin
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
            # radicals are species with a single unpair electron
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
        vxc = jnp.einsum('ij,xjk,kl->xil', L, vxc, L.T)
        vxc = jnp.where(jnp.expand_dims(scaling, 0) > 0, vxc, jnp.expand_dims(scaling, 0))
    else:
        vxc = jnp.matmul(L, jnp.matmul(vxc, L.T))
        vxc = jnp.where(scaling > 0, vxc, scaling)

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
                                  jnp.einsum('ij,jk->ik', mocc_b*jnp.where(mo_occ[1] > 0, mo_occ[1], 0), mocc_b.T)], axis=0)

            def false_func(mocc_a, mocc_b, mo_occ, mo_coeff):
                return jnp.stack([jnp.einsum('ij,jk->ik', mocc_a*jnp.where(mo_occ[0] > 0, mo_occ[0], 0), mocc_a.T),
                                  jnp.zeros_like(mo_coeff)[0]], axis=0)

            retarr = jax.lax.cond(jnp.sum(mo_occ[1]) > 0, true_func, false_func, mocc_a, mocc_b, mo_occ, mo_coeff)
            return retarr

        else:
            print('Spin unpolarized make_rdm1()')
            mocc = jnp.where(mo_occ > 0, mo_coeff, 0)
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
        J = jnp.einsum('...ij,ijkl->...kl', dm, eri)

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
        upper = False
        UPLO = "U" if upper else "L"
        dim = ogdim[0]
        diff = h.shape[0]
        mask_h = h[:dim, :dim]
        mask_s = s_chol[:dim, :dim]
        e, c = jnp.linalg.eigh(jnp.einsum('ij,...jk,kl->...il', mask_s, mask_h, mask_s.T), UPLO=UPLO)
        c = jnp.einsum('ij,...jk ->...ik', mask_s.T, c)
        e = pad_array(e, e, shape=[diff])
        c = pad_array(c, c, shape=(diff, diff))
        return e, c


# Function to calculate statistics
def calculate_stats(true, pred):
    true_flat = true.flatten()
    pred_flat = pred.flatten()

    # R-squared
    ss_res = jnp.sum((true_flat - pred_flat) ** 2)
    ss_tot = jnp.sum((true_flat - jnp.mean(true_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Mean Absolute Error
    mae = jnp.mean(jnp.abs(true_flat - pred_flat))

    # Root Mean Squared Error
    rmse = jnp.sqrt(jnp.mean((true_flat - pred_flat) ** 2))

    # Maximum Absolute Error
    max_error = jnp.max(jnp.abs(true_flat - pred_flat))

    return r2, mae, rmse, max_error


# Generate grid of rho, grad_rho, s values
def gen_grid_s(npts=30000, start_stop_rho=(0.01, 5), start_stop_s=(0.01, 5),
               train_pct=0.8, ranseed=92017, sigma=False):
    '''
    Generates a grid of rho, grad_rho, and reduced_density_gradient values. Will generate ~sqrt(npts) sized rho and s arrays, used to generate a grad_rho array. 

    The indices of these arrays are then randomly sub-sampled to yield (1 - train_pct) percent of validation indices, which are excluded from the training meshgrid.

    :param npts: Number of points to generate in each array, defaults to 30000. The function will get close, but not necessarily exactly the number of points, unless the desired value is a perfect square.
    :type npts: int, optional
    :param start_stop_rho: Bounds for the generated reduced_density_gradient values, defaults to (0.01, 5)
    :type start_stop_rho: tuple, optional
    :param start_stop_s: Bounds for the generated rho values, defaults to (0.01, 5)
    :type start_stop_s: tuple, optional
    :param train_pct: Percentage of generated BASE values to use in training, defaults to 0.8
    :type train_pct: float, optional
    :param ranseed: Seed to set for training/validation index selection, defaults to 92017
    :type ranseed: int, optional
    :param sigma: If flagged, will return array of SIGMA (= grad_rho**2) values instead of "grad_rho" values, defaults to False
    :type sigma: bool, optional
    :return: List of [(train_inds, val_inds), (rho_values, grad_rho_values, s_values),
            (trho_flat, tgrad_rho_flat, ts_flat), (vrho_flat, vgrad_rho_flat, vs_flat)]
    :rtype: list
    '''
    val_pct = 1.0 - train_pct
    START_RHO, STOP_RHO = start_stop_rho
    START_S, STOP_S = start_stop_s
    s_values_low = jnp.linspace(START_S, 0.5, num=int(0.7 * jnp.sqrt(npts)))
    s_values_high = jnp.linspace(0.5, STOP_S, num=int(0.3 * jnp.sqrt(npts)))
    s_values = jnp.concatenate([s_values_low, s_values_high])
    rho_add = len(s_values) - int(jnp.sqrt(npts))
    rho_values = jnp.logspace(jnp.log10(START_RHO), jnp.log10(STOP_RHO), num=int(jnp.sqrt(npts)+rho_add))
    k_F = (3 * jnp.pi**2 * rho_values)**(1/3)
    grad_rho_values = 2 * s_values * k_F * rho_values
    if sigma:
        grad_rho_values = grad_rho_values**2

    ind_sel = np.arange(0, len(s_values))
    np.random.seed(ranseed)
    val_inds = np.random.choice(ind_sel, size=int(val_pct*len(s_values)), replace=False)
    train_inds = np.array([i for i in ind_sel if i not in val_inds])
    # randomize training point order
    np.random.shuffle(train_inds)

    trho_values = rho_values[train_inds]
    vrho_values = rho_values[val_inds]

    ts_values = s_values[train_inds]
    vs_values = s_values[val_inds]

    trho_mesh, ts_mesh = jnp.meshgrid(trho_values, ts_values)
    trho_flat = trho_mesh.flatten()
    ts_flat = ts_mesh.flatten()
    # Calculate grad_rho
    tk_F = (3 * jnp.pi**2 * trho_flat)**(1/3)
    tgrad_rho_flat = 2 * ts_flat * tk_F * trho_flat
    if sigma:
        tgrad_rho_flat = tgrad_rho_flat**2

    vrho_mesh, vs_mesh = jnp.meshgrid(vrho_values, vs_values)
    vrho_flat = vrho_mesh.flatten()
    vs_flat = vs_mesh.flatten()
    # Calculate grad_rho
    vk_F = (3 * jnp.pi**2 * vrho_flat)**(1/3)
    vgrad_rho_flat = 2 * vs_flat * vk_F * vrho_flat
    if sigma:
        vgrad_rho_flat = vgrad_rho_flat**2

    print('shapes- r/gr/s: {}/{}/{}'.format(rho_values.shape, grad_rho_values.shape, s_values.shape))
    return [(train_inds, val_inds), (rho_values, grad_rho_values, s_values),
            (trho_flat, tgrad_rho_flat, ts_flat), (vrho_flat, vgrad_rho_flat, vs_flat)]

# PBE Exchange Enhancement Factor


def PBE_Fx(rho, grad_rho, lower_rho_cutoff=1e-12):
    '''
    Given density and density gradient magnitude values, calculates the PBE Exchange Enhancement Factor as denoted in:

    Equation 14 from PBE paper -- DOI: 10.1103/PhysRevLett.77.3865

    :param rho: the density value on the grid
    :type rho: float, broadcastable array
    :param grad_rho: the magnitude of the density gradient values on the grid
    :type grad_rho: float, broadcastable array
    :param lower_rho_cutoff: a cut-off to bypass potential division by zero in the division by rho, defaults to 1e-12
    :type lower_rho_cutoff: float, optional
    :return: The numerical value(s) of the exchange enhancement factor
    :rtype: float, broadcastable array
    '''
    # Equation 14 from PBE paper -- DOI: 10.1103/PhysRevLett.77.3865
    rho = jnp.maximum(lower_rho_cutoff, rho)  # Prevents division by 0
    k_F = (3 * jnp.pi**2 * rho)**(1/3)
    s = grad_rho / (2 * k_F * rho)
    kappa, mu = 0.804, 0.21951

    Fx = 1 + kappa - kappa / (1 + mu * s**2 / kappa)  # exchange enhancement factor

    return Fx

# NON-POLARIZED PBE Correlation Enhancement Factor
# i.e., zeta == 0


def PBE_Fc(rho, grad_rho,  lower_rho_cutoff=1e-12):
    '''
    Given density and density gradient magnitude values, calculates the PBE Correlation Enhancement Factor as denoted in:

    *CRITICALLY*, this function is only valid for the non-spin-polarized case where zeta = 0

    Equation 3 from PBE paper -- DOI: 10.1103/PhysRevLett.77.3865

    :param rho: the density value on the grid
    :type rho: float, broadcastable array
    :param grad_rho: the magnitude of the density gradient values on the grid
    :type grad_rho: float, broadcastable array
    :param lower_rho_cutoff: a cut-off to bypass potential division by zero in the division by rho, defaults to 1e-12
    :type lower_rho_cutoff: float, optional
    :return: The numerical value(s) of the correlation enhancement factor
    :rtype: float, broadcastable array
    '''
    # Equation 3 from PBE paper -- DOI: 10.1103/PhysRevLett.77.3865
    # Ec = Integral[ rho * (e_C^HEG + H) ]
    # H from equation 7
    # A from equation 8
    rho = jnp.maximum(lower_rho_cutoff, rho)  # Prevents division by 0
    pi = jnp.pi
    k_F = (3 * pi**2 * rho)**(1/3)
    s = grad_rho / (2 * k_F * rho)
    k_s = jnp.sqrt((4 * k_F) / pi)
    t = jnp.abs(grad_rho) / (2 * k_s * rho)
    beta = 0.066725
    gamma = (1 - jnp.log(2)) / (pi**2)

    # Calculate e_heg_c (heterogeneous electron gas correlation energy)
    N = rho.size
    # Initialize the rho_array with the correct shape. Only the first element (rho) matters, so the rest are populated with 0.
    rho_array = jnp.zeros((6, N))
    rho_array = rho_array.at[0, :].set(rho)  # Populate first array value with rho
    e_heg_c = dft.libxc.eval_xc(',LDA_C_PW', rho_array, spin=0, deriv=1)[0]

    A = (beta / gamma) / (jnp.exp(-e_heg_c / (gamma)) - 1)

    H = gamma * jnp.log(1 + (beta / gamma) * t**2 * ((1 + A * t**2) / (1 + A * t**2 + A**2 * t**4)))

    Fc = 1 + (H / e_heg_c)  # correlation enhancement factor

    return Fc


def pw91_correlation_energy_density(rho):
    """
    Calculate the correlation energy density according to the PW91 functional.

    Parameters:
    - rho: electron density (numpy array or scalar)
    - grad_rho: gradient of the electron density (numpy array or scalar)

    Returns:
    - correlation energy density (numpy array or scalar)
    """
    A = 0.0311
    B = 0.116
    C = 0.145

    # Compute the correlation energy density
    rho_1_3 = jnp.cbrt(rho)  # rho^(1/3)
    term1 = 1 / jnp.sqrt(1 + C / rho_1_3)
    epsilon_c = -A / rho * (1 + B * (term1 - 1))

    return epsilon_c


def lda_c_pw(rho):
    params_a_pp = [1,  1,  1]
    params_a_a = [0.031091, 0.015545, 0.016887]
    params_a_alpha1 = [0.21370,  0.20548,  0.11125]
    params_a_beta1 = [7.5957, 14.1189, 10.357]
    params_a_beta2 = [3.5876, 6.1977, 3.6231]
    params_a_beta3 = [1.6382, 3.3662,  0.88026]
    params_a_beta4 = [0.49294, 0.62517, 0.49671]
    params_a_fz20 = 1.709921

    zeta = (rho - rho)/(rho + rho + 1e-8)
    rs = (4*np.pi*(rho + 1e-8)/3)**(-1/3)

    def g_aux(k, rs):
        return params_a_beta1[k]*jnp.sqrt(rs) + params_a_beta2[k]*rs\
            + params_a_beta3[k]*rs**1.5 + params_a_beta4[k]*rs**(params_a_pp[k] + 1)

    def g(k, rs):
        return -2*params_a_a[k]*(1 + params_a_alpha1[k]*rs)\
            * jnp.log(1 + 1/(2*params_a_a[k]*g_aux(k, rs)))

    def f_zeta(zeta):
        return ((1+zeta)**(4/3) + (1-zeta)**(4/3) - 2)/(2**(4/3)-2)

    def f_pw(rs, zeta):
        return g(0, rs) + zeta**4*f_zeta(zeta)*(g(1, rs) - g(0, rs) + g(2, rs)/params_a_fz20)\
            - f_zeta(zeta)*g(2, rs)/params_a_fz20

    return f_pw(rs, zeta)


def lda_x(rho):
    '''
    The HEG exchange energy density.

    :param rho: Total electron density array on a grid.
    :type rho: jax.numpy array
    :return: Exchange energy density array.
    :rtype: jax.numpy array
    '''
    return -3/4*(3/jnp.pi)**(1/3)*rho**(1/3)


def pw92c(rho):
    '''
    Implements the Perdew-Wang '92 local correlation (beyond RPA) for the unpolarized case.
    Reference: J.P.Perdew & Y.Wang, PRB, 45, 13244 (1992)

    :param rho: Total electron density array on a grid.
    :type rho: jax.numpy array
    :return: Correlation energy density and potential arrays.
    :rtype: tuple of (jax.numpy array, jax.numpy array)
    '''
    # Constants
    nspin = 1
    DENMIN = 1.e-12
    ONE = 1 + 1.e-12
    PI = jnp.pi
    # Parameters
    P = jnp.array([1.00, 1.00, 1.00])
    A = jnp.array([0.031091, 0.015545, 0.016887])
    ALPHA1 = jnp.array([0.21370, 0.20548, 0.11125])
    # MVFS test the transpose
    X = jnp.array([[7.5957, 14.1189, 10.357],
                   [3.5876, 6.1977, 3.6231],
                   [1.6382, 3.3662, 0.88026],
                   [0.49294, 0.62517, 0.49671]])
    BETA = jnp.transpose(X)
    # Calculate rs and zeta
    if nspin == 1:
        DTOT = jnp.maximum(DENMIN, rho[0])
        ZETA = 0
        RS = (3 / (4 * PI * DTOT)) ** (1 / 3)
        DRSDD = -RS / DTOT / 3
        DZDD = jnp.array([0.0])
    else:
        DTOT = jnp.maximum(DENMIN, rho[0] + rho[1])
        ZETA = (rho[0] - rho[1]) / DTOT
        RS = (3 / (4 * PI * DTOT)) ** (1 / 3)
        DRSDD = -RS / DTOT / 3
        DZDD = jnp.array([(ONE - ZETA) / DTOT, - (ONE + ZETA) / DTOT])
    # Compute G and its derivatives
    G = jnp.zeros(3)
    DGDRS = jnp.zeros(3)
    for IG in range(3):
        B = BETA[IG, 0] * RS ** 0.5 + \
            BETA[IG, 1] * RS + \
            BETA[IG, 2] * RS ** 1.5 + \
            BETA[IG, 3] * RS ** (P[IG] + 1)
        DBDRS = BETA[IG, 0] * 0.5 / RS ** 0.5 + \
            BETA[IG, 1] + \
            BETA[IG, 2] * 1.5 * RS ** 0.5 + \
            BETA[IG, 3] * (P[IG] + 1) * RS ** P[IG]
        C = 1 + 1 / (2 * A[IG] * B)
        DCDRS = - ((C - 1) * DBDRS / B)
        Gtmp = (-2) * A[IG] * (1 + ALPHA1[IG] * RS) * jnp.log(C)
        DGDRStmp = (-2) * A[IG] * (ALPHA1[IG] * jnp.log(C) +
                                   (1 + ALPHA1[IG] * RS) * DCDRS / C)
        G = G.at[IG].set(Gtmp)
        DGDRS = DGDRS.at[IG].set(DGDRStmp)
    # Find f’’(0) and f(zeta)
    C = 1 / (2 ** (4 / 3) - 2)
    FPP0 = 8 * C / 9
    F = ((ONE + ZETA) ** (4 / 3) + (ONE - ZETA) ** (4 / 3) - 2) * C
    DFDZ = (4 / 3) * ((ONE + ZETA) ** (1 / 3) - (ONE - ZETA) ** (1 / 3)) * C
    # Compute EC and VC
    EC = G[0] - G[2] * F / FPP0 * (ONE - ZETA ** 4) + \
        (G[1] - G[0]) * F * ZETA ** 4
    DECDRS = DGDRS[0] - DGDRS[2] * F / FPP0 * (ONE - ZETA ** 4) + \
        (DGDRS[1] - DGDRS[0]) * F * ZETA ** 4
    DECDZ = (-G[2]) / FPP0 * (DFDZ * (ONE - ZETA ** 4) - F * 4 * ZETA ** 3) + \
            (G[1] - G[0]) * (DFDZ * ZETA ** 4 + F * 4 * ZETA ** 3)
    # Calculate correlation potential
    if nspin == 1:
        DECDD = DECDRS * DRSDD
        VC = jnp.array([EC + DTOT * DECDD])
    else:
        DECDD = jnp.array([DECDRS * DRSDD + DECDZ * DZDD[0],
                           DECDRS * DRSDD + DECDZ * DZDD[1]])
        VC = jnp.array([EC + DTOT * DECDD[0],
                        EC + DTOT * DECDD[1]])
    return EC, VC


def pw92c_unpolarized(rho):
    '''
    Implements the Perdew-Wang '92 local correlation (beyond RPA) for the unpolarized case.
    Reference: J.P.Perdew & Y.Wang, PRB, 45, 13244 (1992)

    :param rho: Total electron density array on a grid.
    :type rho: jax.numpy array
    :return: Correlation energy density array.
    :rtype: jax.numpy array
    '''
    # Ensure rho is a jax.numpy array
    rho = jnp.asarray(rho)

    # Parameters from Table I of Perdew & Wang, PRB, 45, 13244 (92)
    A = jnp.array([0.031091, 0.015545, 0.016887])
    ALPHA1 = jnp.array([0.21370, 0.20548, 0.11125])
    BETA1 = jnp.array([7.5957, 14.1189, 10.357])
    BETA2 = jnp.array([3.5876, 6.1977, 3.6231])
    BETA3 = jnp.array([1.6382, 3.3662, 0.88026])
    BETA4 = jnp.array([0.49294, 0.62517, 0.49671])

    # Compute rs (Wigner-Seitz radius) for each grid point
    rs = (3 / (4 * jnp.pi * rho))**(1/3)

    # Compute G for unpolarized case (zeta = 0) across all grid points
    def compute_g(rs):
        try:
            G = jnp.zeros((len(rs), 3))
        except:
            G = jnp.zeros((1, 3))
        for k in range(3):
            B = (BETA1[k] * jnp.sqrt(rs) +
                 BETA2[k] * rs +
                 BETA3[k] * rs**1.5 +
                 BETA4[k] * rs**2)
            C = 1 + 1 / (2 * A[k] * B)
            G = G.at[:, k].set(-2 * A[k] * (1 + ALPHA1[k] * rs) * jnp.log(C))
        return G

    # Apply compute_g to each grid point
    G = compute_g(rs)

    # For unpolarized case, correlation energy density is G[0]
    EC = G[:, 0]

    return EC
