import equinox as eqx
import jax
import jax.numpy as jnp
from xcquinox.utils import get_dm_moe, pad_array, pad_array_list
from xcquinox.pyscf import generate_network_eval_xc


@eqx.filter_value_and_grad
def compute_loss_mae(model, inputs, ref):
    '''
    Computes the mean-absolute-error loss of the model's prediction using the given inputs against the provided reference.

    :param model: The model which is given to `jax.vmap` to generate predictions using the inputs given.
    :type model: eqx.Module
    :param inputs: The input points that are given to the network. Shape will be dependent on your network architecture.
    :type inputs: array
    :param ref: The reference values to be used in generating prediction error.
    :type ref: array
    :return: The MAE that will be used in backpropagation. 
    :rtype: float
    '''
    pred = jax.vmap(model)(inputs)
    loss = jnp.mean(jnp.abs(pred - ref))
    return loss

# =====================================================================
# =====================================================================
# DEPRECATED CLASSES -- TO BE REMOVED
# =====================================================================
# =====================================================================


class E_loss(eqx.Module):
    def __init__(self):
        '''
        The standard energy loss module, RMSE loss of predicted vs. reference energies.
        '''
        super().__init__()

    def __call__(self, model, inp_dm, ref_en, ao_eval, grid_weights):
        '''
        Computes the energy loss for a given model and associated input density matrix, atomic orbitals on the grid, and grid weights

        Loss is the RMSE energy, so predicted energy can potentially be a jax.Array of SCF guesses.

        :param model: The XC object whose forward pass predicts the XC energy based on the inputs here.
        :type model: xcquinox.xc.eXC
        :param inp_dm: The density matrix to pass into the network for density creation on the grid.
        :type inp_dm: jax.Array
        :param ref_en: The reference energy to take the loss with respect to.
        :type ref_en: jax.Array
        :param ao_eval: Atomic orbitals evaluated on the grid
        :type ao_eval: jax.Array
        :param grid_weights: pyscfad's grid weights for the reference calculation
        :type grid_weights: jax.Array
        :return: The RMSE error.
        :rtype: jax.Array
        '''
        e_pred = model(inp_dm, ao_eval, grid_weights)
        eL = jnp.sqrt(jnp.mean((e_pred-ref_en)**2))
        return eL


class NL_E_loss(eqx.Module):
    def __init__(self):
        '''
        The standard energy loss module for a non-local descriptor training, RMSE loss of predicted vs. reference energies.
        '''
        super().__init__()

    def __call__(self, model, inp_dm, ref_en, ao_eval, grid_weights, mf):
        '''
        Computes the energy loss for a given model and associated input density matrix, atomic orbitals on the grid, and grid weights

        Loss is the RMSE energy, so predicted energy can potentially be a jax.Array of SCF guesses.

        :param model: The XC object whose forward pass predicts the XC energy based on the inputs here.
        :type model: xcquinox.xc.eXC
        :param inp_dm: The density matrix to pass into the network for density creation on the grid.
        :type inp_dm: jax.Array
        :param ref_en: The reference energy to take the loss with respect to.
        :type ref_en: jax.Array
        :param ao_eval: Atomic orbitals evaluated on the grid
        :type ao_eval: jax.Array
        :param grid_weights: pyscfad's grid weights for the reference calculation
        :type grid_weights: jax.Array
        :param mf: A pyscf(ad) converged calculation kernel if self.level > 3, used for building the CIDER nonlocal descriptors, defaults to None
        :type mf: pyscfad.dft.RKS kernel
        :return: The RMSE error.
        :rtype: jax.Array
        '''
        e_pred = model(inp_dm, ao_eval, grid_weights, mf)
        eL = jnp.sqrt(jnp.mean((e_pred-ref_en)**2))
        return eL


class DM_HoLu_loss(eqx.Module):
    def __init__(self):
        """
        Creates DM_HoLu_loss object for use in training.

        Options to compute the RMSE loss with respect to the density matrix, the RMSE homo-lumo gap loss, and the root-integrated-squared loss for the density on the grid.

        """
        super().__init__()

    def __call__(self, model, ao_eval, gw, dm, eri, mo_occ, hc, s, ogd, holu=None, alpha0=0.7,
                 dmL=1.0, holuL=1.0, dm_to_rho=0.0):
        """
        Forward pass to compute the total loss based on the given inputs.

        If more than one loss flag evaluates to True, total loss returned is the rooted sum-of-squares for the individual losses.
        i.e. total_loss = jnp.sqrt( (dmL*dmLv)**2 + (holuL*holuLv)**2 + (dm_to_rho*rhoLv)**2)

        :param model: The model for use in generating the Vxc during the DM generation
        :type model: xcquinox.xc.eXC
        :param ao_eval: The atomic orbitals evaluated on the grid for the given molecule
        :type ao_eval: jax.Array
        :param gw: The grid weights associated to the current molecule's grids
        :type gw: jax.Array
        :param dm: Input reference density matrix for use during the one-shot forward pass to generate the new DM
        :type dm: jax.Array
        :param eri: Electron repulsion integrals associated with this molecule
        :type eri: jax.Array
        :param mo_occ: The molecule's molecular orbital occupation numbers
        :type mo_occ: jax.Array
        :param hc: The molecule's core Hamiltonian
        :type hc: jax.Array
        :param s: The molecule's overlap matrix
        :type s: jax.Array
        :param ogd: The original dimensions of this molecule's density matrix, used if padded to constrict the eigendecomposition to a relevant shape
        :type ogd: jax.Array
        :param holu: The reference HOMO-LUMO bandgap, if doing the corresponding loss, defaults to None
        :type holu: jax.Array, optional
        :param alpha0: The mixing parameter for the one-shot density matrix generation, defaults to 0.7
        :type alpha0: float, optional
        :param dmL: Float to evaluate whether or not to include RMSE DM loss, used as the loss weight, defaults to 1.0
        :type dmL: float, optional
        :param holuL: Float to evaluate whether or not to include RMSE HOMO-LUMO gap loss, used as the loss weight, defaults to 1.0
        :type holuL: float, optional
        :param dm_to_rho: Float to evaluate whether or not to include integrated rho-on-grid loss, used as the loss weight, defaults to 0.0
        :type dm_to_rho: float, optional
        :return: The root-sum of squares loss
        :rtype: jax.Array
        """
        # create the function for calculating E to take derivative of for vxc
        def vgf(x): return model(x, ao_eval, gw)
        # predict the network-based DM, mo_e, and mo_coeff
        dmp, moep, mocp = get_dm_moe(dm, eri, vgf, mo_occ, hc, s, ogd, alpha0)

        # density matrix loss, RMSE
        dmLv = jnp.sqrt(jnp.mean((dmp - dm)**2)) if dmL else 0

        # holu loss if needed
        if holuL:
            homo_i = jnp.max(jnp.nonzero(mo_occ, size=dm.shape[0])[0])
            holup = moep[homo_i+1] - moep[homo_i]
            holuLv = jnp.sqrt(jnp.mean((holu - holup)**2))
        else:
            holuLv = 0

        # rho on grid loss if needed
        if dm_to_rho:
            rho = jnp.einsum('xij,yik,...jk->xy...i', ao_eval, ao_eval, dm)+1e-10
            rhop = jnp.einsum('xij,yik,...jk->xy...i', ao_eval, ao_eval, dmp)+1e-10
            # integrate squared density difference on grid, sqrt then weight by number of electrons
            rhoLv = jnp.sqrt(jnp.sum((rho-rhop)**2 * gw))/jnp.sum(mo_occ)
        else:
            rhoLv = 0

        return jnp.sqrt((dmL*dmLv)**2 + (holuL*holuLv)**2 + (dm_to_rho*rhoLv)**2)


class Band_gap_1shot_loss(eqx.Module):
    def __init__(self):
        """
        Initializer for the loss module, which attempts to find loss band gaps w.r.t. reference

        .. todo: Make more robust for non-local descriptors
        """
        super().__init__()

    def __call__(self, model, ao_eval, gw, dm, eri, mo_occ, hc, s, ogd, refgap, mf, alpha0=0.7):
        """
        Forward pass for loss object

        NOTE: This differs from HoLu loss in that it selects the deepest minimum w.r.t. the LUMO (Fermi energy)

        :param model: The model that will be used in generating the molecular orbital energies ('band' energies)
        :type model: xcquinox.xc.eXC
        :param ao_eval: The atomic orbitals evaluated on the grid for the given molecule
        :type ao_eval: jax.Array
        :param gw: The grid weights associated to the current molecule's grids
        :type gw: jax.Array
        :param dm: Input reference density matrix for use during the one-shot forward pass to generate the new DM
        :type dm: jax.Array
        :param eri: Electron repulsion integrals associated with this molecule
        :type eri: jax.Array
        :param mo_occ: The molecule's molecular orbital occupation numbers
        :type mo_occ: jax.Array
        :param hc: The molecule's core Hamiltonian
        :type hc: jax.Array
        :param s: The molecule's overlap matrix
        :type s: jax.Array
        :param ogd: The original dimensions of this molecule's density matrix, used if padded to constrict the eigendecomposition to a relevant shape
        :type ogd: jax.Array
        :param refgap: The reference gap to optimzie against
        :type refgap: jax.Array
        :param mf: A pyscf(ad) converged calculation kernel if self.level > 3, used for building the CIDER nonlocal descriptors, defaults to None
        :type mf: pyscfad.dft.RKS kernel
        :param alpha0: The mixing parameter for the one-shot density matrix generation, defaults to 0.7
        :type alpha0: float, optional
        :return: Root-squared error between predicted gap (minimum of molecular energies) and the reference
        :rtype: jax.Array
        """
        def vgf(x): return model(x, ao_eval, gw, mf)
        dmp, moep, mocp = get_dm_moe(dm, eri, vgf, mo_occ, hc, s, ogd, alpha0)

        efermi = moep[mf.mol.nelectron//2-1]
        moep -= efermi
        # print(moep)
        moep_gap = jnp.min(moep)
        # print(moep_gap)
        loss = jnp.sqrt((moep_gap - refgap)**2)
        # print(loss)
        return jnp.sqrt((moep_gap - refgap)**2)


class DM_Gap_loss(eqx.Module):
    def __init__(self):
        '''
        Initializer for the DM_Gap_loss, a semi-local loss calculator to optimize the one-shot density matrices and band gaps for 
        Gamma-Gamma direct transitions for PBC structures.

        Band gap optimized is the HOMO-LUMO gap here, hence why specifically Gamma-Gamma
        '''
        super().__init__()

    def __call__(self, model, ao, hc, eri, s, gw, inp_dm, mo_occ, ogd, refDM, refGap):
        '''
        Forward pass to calculate the DM (sum of squared error) and gap (squared error) loss for a given molecule.

        Individual molecule loss is jnp.sqrt( dmL + gapL ),
        where dmL = jnp.sum( (dm_pred-dm_ref)**2 )
        and gapL = (gap_pred - gap_ref)**2

        :param model: The model to use in the predictions, here to generate DM and molecular energies
        :type model: xcquinox.xc.eXC
        :param ao: Atomic orbitals evaluated on a grid
        :type ao: jax.Array
        :param hc: Core Hamiltonian
        :type hc: jax.Array
        :param eri: Electron repulsion integrals
        :type eri: jax.Array
        :param s: Overlap matrices
        :type s: jax.Array
        :param gw: Weights for the grid being used
        :type gw: jax.Array
        :param inp_dm: Initial density matrix guesses, from mf.get_init_guess(), to be used in the one-shot DM generation to produce a mixed DM to optimize against reference
        :type inp_dm: jax.Array
        :param mo_occ: Molecular orbital occupations
        :type mo_occ: jax.Array
        :param ogd: The original dimensions of the density matrix
        :type ogd: tuple
        :param refDM: Reference density matricex from high-accuracy method (e.g., CCSD(T)).
        :type refDM: jax.Array
        :param refGap: Reference band gap (e.g. from the Borlido 2019 dataset).
        :type refGap: jax.Array
        :return: The molecule's loss
        :rtype: jax.Array/scalar
        '''
        homo_i = jnp.max(jnp.nonzero(mo_occ, size=inp_dm.shape[0])[0])
        # vxc function for gradient
        def vgf(x): return model(x, ao, gw)
        dmp, moep, mocoep = get_dm_moe(inp_dm, eri, vgf, mo_occ, hc, s, ogd)

        dmp = pad_array(dmp, inp_dm)
        moep = pad_array(moep, moep,  shape=(dmp.shape[0],))
        gap_pred = moep[homo_i+1]-moep[homo_i]

        dm_L = jnp.sum((dmp-refDM)**2)

        gap_L = (gap_pred-refGap)**2

        return jnp.sqrt(dm_L+gap_L)


class DM_Gap_Loop_loss(eqx.Module):
    def __init__(self):
        '''
        Initializer for the DM_Gap_Loop_loss, a semi-local loss calculator to optimize the one-shot density matrices and band gaps for 
        Gamma-Gamma direct transitions for PBC structures.

        Band gap optimized is the HOMO-LUMO gap here, hence why specifically Gamma-Gamma

        This is a BATCH-CUMULATIVE LOSS class -- it will loop over the inputs and accumulate each DM_Gap loss before the loss is returned and used for optimization
        '''
        super().__init__()

    def __call__(self, model, aos, hcs, eris, ss, gws, inp_dms, mo_occs, ogds, refDMs, refGaps):
        '''
        Forward pass to calculate the DM (sum of squared error) and gap (squared error) loss across a given dataset.

        Individual molecule loss is jnp.sqrt( dmL + gapL ),
        where dmL = jnp.sum( (dm_pred-dm_ref)**2 )
        and gapL = (gap_pred - gap_ref)**2

        :param model: The model to use in the predictions, here to generate DM and molecular energies
        :type model: xcquinox.xc.eXC
        :param aos: Atomic orbitals evaluated on a grid
        :type aos: list of jax.Arrays
        :param hcs: Core Hamiltonians
        :type hcs: list of jax.Arrays
        :param eris: Electron repulsion integrals
        :type eris: list of jax.Arrays
        :param ss: Overlap matrices
        :type ss: list of jax.Arrays
        :param gws: Weights for the grids being used
        :type gws: list of jax.Arrays
        :param inp_dms: Initial density matrix guesses, from mf.get_init_guess(), to be used in the one-shot DM generation to produce a mixed DM to optimize against reference
        :type inp_dms: list of jax.Arrays
        :param mo_occs: Molecular orbital occupations
        :type mo_occs: list of jax.Arrays
        :param ogds: The original dimensions of the density matrices
        :type ogds: list of tuples
        :param refDMs: List of reference density matrices from high-accuracy method (e.g., CCSD(T)).
        :type refDMs: list of jax.Arrays
        :param refGaps: List of reference band gaps (e.g. from the Borlido 2019 dataset).
        :type refGaps: list of jax.Arrays
        :return: The cumulative loss across the dataset
        :rtype: jax.Array/scalar
        '''
        total_loss = 0
        for idx in range(len(aos)):
            # subselect the individual loss data
            mo_occ = mo_occs[idx]
            inp_dm = inp_dms[idx]
            ao = aos[idx]
            gw = gws[idx]
            eri = eris[idx]
            hc = hcs[idx]
            s = ss[idx]
            ogd = ogds[idx]
            refGap = refGaps[idx]
            refDM = refDMs[idx]

            homo_i = jnp.max(jnp.nonzero(mo_occ, size=inp_dm.shape[0])[0])
            # vxc function for gradient
            def vgf(x): return model(x, ao, gw)
            dmp, moep, mocoep = get_dm_moe(inp_dm, eri, vgf, mo_occ, hc, s, ogd)

            dmp = pad_array(dmp, inp_dm)
            moep = pad_array(moep, moep,  shape=(dmp.shape[0],))
            gap_pred = moep[homo_i+1]-moep[homo_i]

            dm_L = jnp.sum((dmp-refDM)**2)

            gap_L = (gap_pred-refGap)**2

            total_loss += jnp.sqrt(dm_L+gap_L)
        return total_loss


class E_PySCFAD_loss(eqx.Module):
    def __init__(self):
        '''
        The standard energy loss module, RMSE loss of predicted vs. reference energies.
        '''
        super().__init__()

    def __call__(self, model, mf, inp_dm, ref_en):
        '''
        Computes the energy loss for a given model and associated input density matrix, atomic orbitals on the grid, and grid weights

        Loss is the RMSE energy, so predicted energy can potentially be a jax.Array of SCF guesses.

        :param model: The XC object whose forward pass predicts the XC energy based on the inputs here.
        :type model: xcquinox.xc.eXC
        :param mf: A pyscf(ad) converged calculation kernel, whose eval_xc is overwritten to use the model calculation
        :type mf: pyscfad.dft.RKS kernel
        :param inp_dm: The density matrix to pass into the network for density creation on the grid.
        :type inp_dm: jax.Array
        :param ref_en: The reference energy to take the loss with respect to.
        :type ref_en: jax.Array
        :return: The RMSE error.
        :rtype: jax.Array
        '''
        print('generating eval_xc function to overwrite')
        evxc = generate_network_eval_xc(mf=mf, dm=inp_dm, network=model)
        mf.define_xc_(evxc, xctype='MGGA')
        print('predicting energy...')
        e_pred = mf.kernel()
        print('energy predicted')
        eL = jnp.sqrt(jnp.mean((e_pred-ref_en)**2))
        return eL
