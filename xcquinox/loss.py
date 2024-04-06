import equinox as eqx
import jax.numpy as jnp
from xcquinox.utils import get_dm_moe

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
        eL = jnp.sqrt( jnp.mean((e_pred-ref_en)**2))
        return eL

class DM_HoLu_loss(eqx.Module):
    def __init__(self):
        """
        Creates DM_HoLu_loss object for use in training.

        Options to compute the RMSE loss with respect to the density matrix, the RMSE homo-lumo gap loss, and the root-integrated-squared loss for the density on the grid.

        """
        super().__init__()

    def __call__(self, model, ao_eval, gw, dm, eri, mo_occ, hc, s, ogd, holu=None, alpha0=0.7,
                dmL = 1.0, holuL = 1.0, dm_to_rho = 0.0):
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
        #create the function for calculating E to take derivative of for vxc
        vgf = lambda x: model(x, ao_eval, gw)
        #predict the network-based DM, mo_e, and mo_coeff
        dmp, moep, mocp = get_dm_moe(dm, eri, vgf, mo_occ, hc, s, ogd, alpha0)

        #density matrix loss, RMSE
        dmLv = jnp.sqrt(jnp.mean( (dmp - dm)**2)) if dmL else 0

        #holu loss if needed
        if holuL:
            homo_i = jnp.max(jnp.nonzero(mo_occ, size=dm.shape[0])[0])
            holup = moep[homo_i+1] - moep[homo_i]
            holuLv = jnp.sqrt(jnp.mean((holu - holup)**2))
        else:
            holuLv = 0

        #rho on grid loss if needed
        if dm_to_rho:
            rho = jnp.einsum('xij,yik,...jk->xy...i', ao_eval, ao_eval, dm)+1e-10
            rhop = jnp.einsum('xij,yik,...jk->xy...i', ao_eval, ao_eval, dmp)+1e-10
            #integrate squared density difference on grid, sqrt then weight by number of electrons
            rhoLv = jnp.sqrt(jnp.sum( (rho-rhop)**2 * gw))/jnp.sum(mo_occ)
        else:
            rhoLv = 0


        return jnp.sqrt( (dmL*dmLv)**2 + (holuL*holuLv)**2 + (dm_to_rho*rhoLv)**2)
