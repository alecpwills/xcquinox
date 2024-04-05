import equinox as eqx
import jax.numpy as jnp

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
