import equinox as eqx
import jax.numpy as jnp

class E_loss(eqx.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, model, inp_dm, ref_en, ao_eval, grid_weights):

        e_pred = model(inp_dm, ao_eval, grid_weights)
        eL = jnp.sqrt( jnp.mean((e_pred-ref_en)**2))
        return eL
