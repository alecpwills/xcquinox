import jax
import os
import pickle
import jax.numpy as jnp
import equinox as eqx
from warnings import warn
import json
from typing import Union
import numpy as np

# =====================================================================
# =====================================================================
# Lieb-Oxford Bound Enforcer
# =====================================================================
# =====================================================================


class LOB(eqx.Module):
    limit: float

    def __init__(self, limit: float):
        '''
        Utility function to squash output to [-1, limit-1] inteval.

        :param limit: The Lieb-Oxford bound value to impose, defaults to 1.804
        :type limit: float, optional
        '''
        super().__init__()
        self.limit = limit

    def __call__(self, x):
        '''
        Method calling the actual mapping of the input to the desired bounded region.

        :param x: Energy value to map back into the bounded region.
        :type x: float 
        :return: Energy value mapped into bounded region.
        :rtype: float
        '''
        return self.limit * jax.nn.sigmoid(x-jnp.log(self.limit - 1))-1


# =====================================================================
# =====================================================================
# GGA LEVEL NETWORKS
# =====================================================================
# =====================================================================

# Base Fx/Fc networks:
# Define the neural network module for Fx
class GGA_FxNet_s(eqx.Module):
    """S: Exchange enhancementt factor for GGA.

    The input to the network is the reduced density gradient, s, and the output is the enhancement factor, Fx.
    """
    name: str
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=1.804):
        '''
        Constructor for the exchange enhancement factor object, for the GGA case.

        In a GGA XC functional, the relevant quantities are (rho, grad_rho). Here, the network's input size is
        hard-coded to 1 -- just the gradient information is passed to the network, to guarantee that the energy
        yielded from this multiplicative factor behaves correctly under uniform scaling of the electron density
        and obeys the spin-scaling relation.

        :param depth: Depth of the neural network
        :type depth: int
        :param nodes: Number of nodes in each layer
        :type nodes: int
        :param seed: The random seed to initiate baseline weight values for the network
        :type seed: int
        :param lob_lim: The Lieb-Oxford bound to respect, defaults to 1.804
        :type lob_lim: float, optional
        '''
        self.name = 'GGA_FxNet_s'
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.lob_lim = lob_lim
        # to constrain this, we require only gradient inputs
        self.net = eqx.nn.MLP(in_size=1,  # Input is ONLY gradient_descriptor
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        '''
        The network's forward pass, resulting in the enhancement factor associated to the input gradient descriptor.

        *NOTE*: This forward pass is explicitly NOT vectorized -- it expects one grid point worth of data, the (rho, gradient_descriptor) values at that point. This structure expects the :jax.vmap: call to be coded OUTSIDE of the network class.

        *NOTE*: Here, the gradient_descriptor is assumed to be the reduced density gradient, :s:, and the call is structured in such a way to respect the UEG limits for when gradients vanish. Namely, when s = 0, Fx = 1, so the resulting e = Fx*e_heg = e_heg.

        :param inputs: _description_
        :type inputs: tuple, list, array of size 2 in order (rho, gradient_descriptor)
        :return: The enhancement factor value
        :rtype: float
        '''
        # takes forever if inputs[1] tanh input has extended shape , i.e. (1,1) as opposed to scalar shape (1,)
        return 1+self.lobf((jnp.tanh(inputs[1])**2)*self.net(inputs[1, jnp.newaxis]).squeeze())


# Define the neural network module for Fc
class GGA_FcNet_s(eqx.Module):
    """S: Correlation enhancement factor for GGA.

    The input to the network is the reduced density gradient, s, and the output is the enhancement factor, Fc.
    """
    name: str
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=2.0):
        '''
        Constructor for the correlation enhancement factor object, for the GGA case.

        In a GGA XC functional, the relevant quantities are (rho, grad_rho). Here, the network's input size is hard-coded to 2 -- both the density and gradient information is passed to the network.

        The default Lieb-Oxford bound the outputs are wrapped here is set to 2.0, to enforce the non-negativity of the correlation energy.

        :param depth: Depth of the neural network
        :type depth: int
        :param nodes: Number of nodes in each layer
        :type nodes: int
        :param seed: The random seed to initiate baseline weight values for the network
        :type seed: int
        :param lob_lim: The Lieb-Oxford bound to respect, defaults to 2
        :type lob_lim: float, optional
        '''
        self.name = 'GGA_FcNet_s'
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.lob_lim = lob_lim
        self.net = eqx.nn.MLP(in_size=2,  # Input is rho, gradient_descriptor
                              out_size=1,  # Output is Fc
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        '''
        The network's forward pass, resulting in the enhancement factor associated to the input gradient descriptor.

        *NOTE*: This forward pass is explicitly NOT vectorized -- it expects one grid point worth of data, the (rho, gradient_descriptor) values at that point. This structure expects the :jax.vmap: call to be coded OUTSIDE of the network class.

        *NOTE*: Here, the gradient_descriptor is assumed to be the reduced density gradient, :s:, and the call is structured in such a way to respect the UEG limits for when gradients vanish. Namely, when s = 0, Fx = 1, so the resulting e = Fx*e_heg = e_heg.

        :param inputs: _description_
        :type inputs: tuple, list, array of size 2 in order (rho, gradient_descriptor)
        :return: The enhancement factor value
        :rtype: float
        '''
        # takes forever if inputs[1] tanh input has extended shape , i.e. (1,1) as opposed to scalar shape (1,)
        return 1+self.lobf((jnp.tanh(inputs[1])**2)*self.net(inputs).squeeze())


class GGA_FxNet_G(eqx.Module):
    """S: Exchange enhancementt factor for GGA.

    It takes rho and grad_rho as input and outputs the exchange enhancement factor, Fx.
    It transforms the input to the reduced density gradient, s, and then passes it through the network.
    """
    name: str
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=1.804):
        self.name = 'GGA_FxNet_G'
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.lob_lim = lob_lim
        # to constrain this, we require only gradient inputs
        self.net = eqx.nn.MLP(in_size=1,  # Input is ONLY s
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        # rho = jnp.maximum(1e-12, inputs[0])  # Prevents division by 0
        # rho = rho.flatten()
        # print('WITHOUT RHO MAXIMUM')
        rho = inputs[0].flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = inputs[1].flatten() / (2 * k_F * rho)
        s = s.flatten()
        tanhterm = jnp.tanh(s)**2
        netterm = self.net(s)
        lobterm = self.lobf(tanhterm*netterm)
        return 1+lobterm.squeeze()


class GGA_FcNet_G(eqx.Module):
    name: str
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=2.0):
        """
        Constructor for the correlation enhancement factor object, for the GGA case.

        In a GGA XC functional, the relevant quantities are (rho, grad_rho). Here, the network's input size is hard-coded to 2 -- both the density and gradient information is passed to the network.

        The default Lieb-Oxford bound the outputs are wrapped here is set to 2.0, to enforce the non-negativity of the correlation energy.

        :param depth: Depth of the neural network
        :type depth: int
        :param nodes: Number of nodes in each layer
        :type nodes: int
        :param seed: The random seed to initiate baseline weight values for the network
        :type seed: int
        :param lob_lim: The Lieb-Oxford bound to respect, defaults to 2
        :type lob_lim: float, optional
        """
        self.name = 'GGA_FcNet_G'
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.lob_lim = lob_lim
        self.net = eqx.nn.MLP(in_size=2,  # Input is rho, s
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        rho = inputs[0].flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = inputs[1].flatten() / (2 * k_F * rho)
        s = s.flatten()
        netinp = jnp.stack([rho, s], axis=0).flatten()
        tanhterm = jnp.tanh(s)**2
        netterm = self.net(netinp)
        lobterm = self.lobf(tanhterm*netterm)
        return 1+lobterm.squeeze()


# Define the neural network module for Fx
class GGA_FxNet_sigma(eqx.Module):
    """S: Exchange enhancementt factor for GGA.

    It takes rho and sigma (grad_rho²) as input and outputs the exchange enhancement factor, Fx.
    It transforms the input to the reduced density gradient, s, and then passes it through the network.
    """
    name: str
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    lower_rho_cutoff: float
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=1.804, lower_rho_cutoff=1e-12):
        '''
        Constructor for the exchange enhancement factor object, for the GGA case.

        In a GGA XC functional, the relevant quantities are (rho, grad_rho). Here, the network's input size is hard-coded to 1 -- just the gradient information is passed to the network, to guarantee that the energy yielded from this multiplicative factor behaves correctly under uniform scaling of the electron density and obeys the spin-scaling relation.

        :param depth: Depth of the neural network
        :type depth: int
        :param nodes: Number of nodes in each layer
        :type nodes: int
        :param seed: The random seed to initiate baseline weight values for the network
        :type seed: int
        :param lob_lim: The Lieb-Oxford bound to respect, defaults to 1.804
        :type lob_lim: float, optional
        :param lower_rho_cutoff: a cut-off to bypass potential division by zero in the division by rho, defaults to 1e-12
        :type lower_rho_cutoff: float, optional
        '''
        self.name = 'GGA_FxNet_sigma'
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.lob_lim = lob_lim
        self.lower_rho_cutoff = lower_rho_cutoff
        # to constrain this, we require only gradient inputs
        self.net = eqx.nn.MLP(in_size=1,  # Input is ONLY gradient_descriptor
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        '''
        The network's forward pass, resulting in the enhancement factor associated to the input gradient descriptor.

        *NOTE*: This forward pass is explicitly NOT vectorized -- it expects one grid point worth of data, the (rho, gradient_descriptor) values at that point. This structure expects the :jax.vmap: call to be coded OUTSIDE of the network class.

        *NOTE*: Here, the gradient_descriptor is assumed to be Libxc's/PySCF's internal variable for the density gradient -- sigma (gradient squared in non-spin-polarized, gradient contracted with itself in spin-polarized). This is so that we have easy access to automatic derivatives with respect to sigma, thus can generate v_sigma and use in convergence testing. However, within the call sigma is translated to the reduced density gradient, :s:, which the network is still assumed to be parameterized by, and the call is structured in such a way to respect the UEG limits for when gradients vanish. Namely, when s = 0, Fx = 1, so the resulting e = Fx*e_heg = e_heg.

        :param inputs: _description_
        :type inputs: tuple, list, array of size 2 in order (rho, gradient_descriptor)
        :return: The enhancement factor value
        :rtype: float
        '''
        # here, assume the inputs is [rho, sigma] and select the appropriate input
        # takes forever if inputs[1] tanh input has extended shape , i.e. (1,1) as opposed to scalar shape (1,)
        # rho = jnp.maximum(self.lower_rho_cutoff, inputs[0]) #Prevents division by 0
        # rho = rho.flatten()
        # sigma = jnp.maximum(self.lower_rho_cutoff, inputs[1]) #Prevents division by 0
        # sigma = sigma.flatten()
        rho = inputs[0]
        sigma = inputs[1]
        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = jnp.sqrt(sigma) / (2 * k_F * rho)
        s = s.flatten()
        tanhterm = jnp.tanh(s)**2
        netterm = self.net(s)
        lobterm = self.lobf(tanhterm*netterm)
        return 1+lobterm.squeeze()


# Define the neural network module for Fc
class GGA_FcNet_sigma(eqx.Module):
    """S: Correlation enhancement factor for GGA.

    It takes rho and sigma (grad_rho²) as input and outputs the correlation enhancement factor, Fc.
    It transforms the input to the reduced density gradient, s, and then passes it through the network.
    """
    name: str
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    lower_rho_cutoff: float
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=2.0, lower_rho_cutoff=1e-12):
        '''
        Constructor for the correlation enhancement factor object, for the GGA case.

        In a GGA XC functional, the relevant quantities are (rho, grad_rho). Here, the network's input size is hard-coded to 2 -- both the density and gradient information is passed to the network.

        The default Lieb-Oxford bound the outputs are wrapped here is set to 2.0, to enforce the non-negativity of the correlation energy.

        :param depth: Depth of the neural network
        :type depth: int
        :param nodes: Number of nodes in each layer
        :type nodes: int
        :param seed: The random seed to initiate baseline weight values for the network
        :type seed: int
        :param lob_lim: The Lieb-Oxford bound to respect, defaults to 2
        :type lob_lim: float, optional
        :param lower_rho_cutoff: a cut-off to bypass potential division by zero in the division by rho, defaults to 1e-12
        :type lower_rho_cutoff: float, optional
        '''
        self.name = 'GGA_FcNet_sigma'
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.lob_lim = lob_lim
        self.lower_rho_cutoff = lower_rho_cutoff
        self.net = eqx.nn.MLP(in_size=2,  # Input is rho, gradient_descriptor
                              out_size=1,  # Output is Fc
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        '''
        The network's forward pass, resulting in the enhancement factor associated to the input gradient descriptor.

        *NOTE*: This forward pass is explicitly NOT vectorized -- it expects one grid point worth of data, the (rho, gradient_descriptor) values at that point. This structure expects the :jax.vmap: call to be coded OUTSIDE of the network class.

        *NOTE*: Here, the gradient_descriptor is assumed to be Libxc's/PySCF's internal variable for the density gradient -- sigma (gradient squared in non-spin-polarized, gradient contracted with itself in spin-polarized). This is so that we have easy access to automatic derivatives with respect to sigma, thus can generate v_sigma and use in convergence testing. However, within the call sigma is translated to the reduced density gradient, :s:, which the network is still assumed to be parameterized by, and the call is structured in such a way to respect the UEG limits for when gradients vanish. Namely, when s = 0, Fx = 1, so the resulting e = Fx*e_heg = e_heg.

        :param inputs: _description_
        :type inputs: tuple, list, array of size 2 in order (rho, gradient_descriptor)
        :return: The enhancement factor value
        :rtype: float
        '''
        # here, assume the inputs is [rho, sigma] and select the appropriate input
        # takes forever if inputs[1] tanh input has extended shape , i.e. (1,1) as opposed to scalar shape (1,)
        rho = jnp.maximum(self.lower_rho_cutoff, inputs[0])  # Prevents division by 0
        rho = rho.flatten()
        sigma = jnp.maximum(self.lower_rho_cutoff, inputs[1])  # Prevents division by 0
        sigma = sigma.flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = jnp.sqrt(sigma) / (2 * k_F * rho)
        s = s.flatten()
        netinp = jnp.stack([rho, s], axis=0).flatten()
        tanhterm = jnp.tanh(s)**2
        netterm = self.net(netinp)
        lobterm = self.lobf(tanhterm*netterm)
        return 1+lobterm.squeeze()

# Saving models


def save_xcquinox_model(model, path: str = '', fixing: Union[str, None] = None,
                        tail_info: Union[str, None] = None, loss: Union[list[float], None] = None):
    """Save our NN model to a file.

    :param model: The model to save.
    :type model: eqx.Module.
    :param path: The path to save the model to, defaults to .
    :type path: str, optional.
    :param fixing: A string to append to the model name, defaults to None. Useful to determine the type of fixing used in the model.
    :type fixing: Union[str, None], optional.
    :param tail_info: A string to append to the model name, defaults to None. Useful to determine any additional information about the model.
    :type tail_info: Union[str, None], optional.
    :param loss: A list of loss values to save, defaults to None. Useful to determine the loss values during training. Will be saved in a separate file.
    :type loss: Union[list[float], None], optional.
    """
    if fixing is None:
        fixing = ''
    else:
        fixing = f'_{fixing}'
    if tail_info is None:
        tail_info = ''
    else:
        tail_info = f'_{tail_info}'
    save_name = f'{model.name}_d{model.depth}_n{model.nodes}_s{model.seed}\
{fixing}{tail_info}'

    needen_info = {'depth': model.depth, 'nodes': model.nodes,
                   'seed': model.seed, 'name': model.name}
    eqx.tree_serialise_leaves(f'{path}/{save_name}.eqx', model)
    with open(f"{path}/{save_name}.json", "w") as f:
        json.dump(needen_info, f)
    print(f'Saved {path}/{save_name}.eqx')

    if loss is not None:
        with open(f"{path}/{save_name}_loss.txt", "w") as f:
            np.savetxt(f, loss)
            print(f'Saved the loss values in {path}/{save_name}_loss.txt')


def load_xcquinox_model(path: str):
    """Load a model from a file.

    Note that we must give the path where the model is stored, without the extension.
    I.e, in path, we should have the files path.eqx and path.json.
    """
    jax.config.update("jax_enable_x64", True)  # Ensure 64-bit is enabled first

    with open(f"{path}.json", "r") as f:
        metadata = json.load(f)

    # Model selection
    name = metadata['name']
    Model_Object = {
        'GGA_FxNet_s': GGA_FxNet_s,
        'GGA_FcNet_s': GGA_FcNet_s,
        'GGA_FxNet_G': GGA_FxNet_G,
        'GGA_FcNet_G': GGA_FcNet_G,
        'GGA_FxNet_sigma': GGA_FxNet_sigma,
        'GGA_FcNet_sigma': GGA_FcNet_sigma,
        'MGGA_FxNet_sigma': MGGA_FxNet_sigma,
        'MGGA_FcNet_sigma': MGGA_FcNet_sigma,
        'MGGA_FxNet_sigma_tranform': MGGA_FxNet_sigma_transform,
        'MGGA_FcNet_sigma_tranform': MGGA_FcNet_sigma_transform
    }.get(name)

    dummy_model = Model_Object(depth=metadata["depth"],
                               nodes=metadata["nodes"],
                               seed=metadata["seed"])

    # Load the saved model into the dummy structure
    model = eqx.tree_deserialise_leaves(f"{path}.eqx", like=dummy_model)
    print(f'Loaded {path}.eqx')
    return model

# unconstrained networks, for testing purposes


class GGA_FxNet_sigma_UNC(eqx.Module):
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    lower_rho_cutoff: float
    net: eqx.nn.MLP

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=1.804, lower_rho_cutoff=1e-12):
        '''
        Constructor for the exchange enhancement factor object, for the GGA case.

        In a GGA XC functional, the relevant quantities are (rho, grad_rho). Here, the network's input size is hard-coded to 1 -- just the gradient information is passed to the network, to guarantee that the energy yielded from this multiplicative factor behaves correctly under uniform scaling of the electron density and obeys the spin-scaling relation.

        :param depth: Depth of the neural network
        :type depth: int
        :param nodes: Number of nodes in each layer
        :type nodes: int
        :param seed: The random seed to initiate baseline weight values for the network
        :type seed: int
        :param lob_lim: The Lieb-Oxford bound to respect, defaults to 1.804
        :type lob_lim: float, optional
        :param lower_rho_cutoff: a cut-off to bypass potential division by zero in the division by rho, defaults to 1e-12
        :type lower_rho_cutoff: float, optional
        '''
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.lob_lim = lob_lim
        self.lower_rho_cutoff = lower_rho_cutoff
        self.net = eqx.nn.MLP(in_size=2,  # Input is rho, gradient_descriptor
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))

    def __call__(self, inputs):
        '''
        The network's forward pass, resulting in the enhancement factor associated to the input gradient descriptor.

        *NOTE*: This forward pass is explicitly NOT vectorized -- it expects one grid point worth of data, the (rho, gradient_descriptor) values at that point. This structure expects the :jax.vmap: call to be coded OUTSIDE of the network class.

        *NOTE*: Here, the gradient_descriptor is assumed to be Libxc's/PySCF's internal variable for the density gradient -- sigma (gradient squared in non-spin-polarized, gradient contracted with itself in spin-polarized). This is so that we have easy access to automatic derivatives with respect to sigma, thus can generate v_sigma and use in convergence testing. However, within the call sigma is translated to the reduced density gradient, :s:, which the network is still assumed to be parameterized by, and the call is structured in such a way to respect the UEG limits for when gradients vanish. Namely, when s = 0, Fx = 1, so the resulting e = Fx*e_heg = e_heg.

        :param inputs: _description_
        :type inputs: tuple, list, array of size 2 in order (rho, gradient_descriptor)
        :return: The enhancement factor value
        :rtype: float
        '''
        # here, assume the inputs is [rho, sigma] and select the appropriate input
        # takes forever if inputs[1] tanh input has extended shape , i.e. (1,1) as opposed to scalar shape (1,)
        rho = jnp.maximum(self.lower_rho_cutoff, inputs[0])  # Prevents division by 0
        rho = rho.flatten()
        sigma = jnp.maximum(self.lower_rho_cutoff, inputs[1])  # Prevents division by 0
        sigma = sigma.flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = jnp.sqrt(sigma) / (2 * k_F * rho)
        s = s.flatten()
        netinp = jnp.stack([rho, s], axis=0).flatten()
        netterm = self.net(netinp)
        return netterm.squeeze()

# Define the neural network module for Fc


class GGA_FcNet_sigma_UNC(eqx.Module):
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    lower_rho_cutoff: float
    net: eqx.nn.MLP

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=2.0, lower_rho_cutoff=1e-12):
        '''
        Constructor for the correlation enhancement factor object, for the GGA case.

        In a GGA XC functional, the relevant quantities are (rho, grad_rho). Here, the network's input size is hard-coded to 2 -- both the density and gradient information is passed to the network.

        The default Lieb-Oxford bound the outputs are wrapped here is set to 2.0, to enforce the non-negativity of the correlation energy.

        :param depth: Depth of the neural network
        :type depth: int
        :param nodes: Number of nodes in each layer
        :type nodes: int
        :param seed: The random seed to initiate baseline weight values for the network
        :type seed: int
        :param lob_lim: The Lieb-Oxford bound to respect, defaults to 2
        :type lob_lim: float, optional
        :param lower_rho_cutoff: a cut-off to bypass potential division by zero in the division by rho, defaults to 1e-12
        :type lower_rho_cutoff: float, optional
        '''
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.lob_lim = lob_lim
        self.lower_rho_cutoff = lower_rho_cutoff
        self.net = eqx.nn.MLP(in_size=2,  # Input is rho, gradient_descriptor
                              out_size=1,  # Output is Fc
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))

    def __call__(self, inputs):
        '''
        The network's forward pass, resulting in the enhancement factor associated to the input gradient descriptor.

        *NOTE*: This forward pass is explicitly NOT vectorized -- it expects one grid point worth of data, the (rho, gradient_descriptor) values at that point. This structure expects the :jax.vmap: call to be coded OUTSIDE of the network class.

        *NOTE*: Here, the gradient_descriptor is assumed to be Libxc's/PySCF's internal variable for the density gradient -- sigma (gradient squared in non-spin-polarized, gradient contracted with itself in spin-polarized). This is so that we have easy access to automatic derivatives with respect to sigma, thus can generate v_sigma and use in convergence testing. However, within the call sigma is translated to the reduced density gradient, :s:, which the network is still assumed to be parameterized by, and the call is structured in such a way to respect the UEG limits for when gradients vanish. Namely, when s = 0, Fx = 1, so the resulting e = Fx*e_heg = e_heg.

        :param inputs: _description_
        :type inputs: tuple, list, array of size 2 in order (rho, gradient_descriptor)
        :return: The enhancement factor value
        :rtype: float
        '''
        # here, assume the inputs is [rho, sigma] and select the appropriate input
        # takes forever if inputs[1] tanh input has extended shape , i.e. (1,1) as opposed to scalar shape (1,)
        rho = jnp.maximum(self.lower_rho_cutoff, inputs[0])  # Prevents division by 0
        rho = rho.flatten()
        sigma = jnp.maximum(self.lower_rho_cutoff, inputs[1])  # Prevents division by 0
        sigma = sigma.flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = jnp.sqrt(sigma) / (2 * k_F * rho)
        s = s.flatten()
        netinp = jnp.stack([rho, s], axis=0).flatten()
        netterm = self.net(netinp)
        return netterm.squeeze()

# =====================================================================
# =====================================================================
# Meta-GGA LEVEL NETWORKS
# =====================================================================
# =====================================================================


class MGGA_FxNet_sigma(eqx.Module):
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    lower_rho_cutoff: float
    net: eqx.nn.MLP
    lobf: eqx.Module
    name: str

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=1.174, lower_rho_cutoff=1e-12):
        '''
        Constructor for the exchange enhancement factor object, for the MGGA case.

        In a MGGA XC functional, the relevant quantities are (rho, grad_rho, laplacian_rho, tau=kinetic energy density). Here, 
        the network's input size is hard-coded to 2 -- just the gradient and alpha (related to tau) information 
        is passed to the network, to guarantee that the energy yielded from this multiplicative 
        factor behaves correctly under uniform scaling of the electron density and obeys the 
        spin-scaling relation.

        :param depth: Depth of the neural network
        :type depth: int
        :param nodes: Number of nodes in each layer
        :type nodes: int
        :param seed: The random seed to initiate baseline weight values for the network
        :type seed: int
        :param lob_lim: The Lieb-Oxford bound to respect, defaults to 1.804
        :type lob_lim: float, optional
        :param lower_rho_cutoff: a cut-off to bypass potential division by zero in the division by rho, defaults to 1e-12
        :type lower_rho_cutoff: float, optional
        '''
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.lob_lim = lob_lim
        self.lower_rho_cutoff = lower_rho_cutoff
        # to constrain this, we require only gradient inputs
        self.net = eqx.nn.MLP(in_size=2,  # Input is gradient_descriptor, tau_descriptor
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)
        self.name = 'MGGA_FxNet_sigma'

    def __call__(self, inputs):
        '''
        The network's forward pass, resulting in the enhancement factor associated to the input gradient descriptor.

        *NOTE*: This forward pass is explicitly NOT vectorized -- it expects one grid point worth of data, the (rho, gradient_descriptor) values at that point. This structure expects the :jax.vmap: call to be coded OUTSIDE of the network class.

        *NOTE*: Here, the gradient_descriptor is assumed to be Libxc's/PySCF's internal variable for the density gradient -- sigma (gradient squared in non-spin-polarized, gradient contracted with itself in spin-polarized). This is so that we have easy access to automatic derivatives with respect to sigma, thus can generate v_sigma and use in convergence testing. However, within the call sigma is translated to the reduced density gradient, :s:, which the network is still assumed to be parameterized by, and the call is structured in such a way to respect the UEG limits for when gradients vanish. Namely, when s = 0, Fx = 1, so the resulting e = Fx*e_heg = e_heg.

        :param inputs: A one-dimensional list/array of inputs [rho, sigma, laplacian_rho, alpha]
        :type inputs: tuple, list, one-dimensional array of size 4 in order [rho, sigma, laplacian_rho, alpha]
        :return: The enhancement factor value
        :rtype: float
        '''
        # here, assume the inputs is [rho, sigma, laplacian, tau] and select the appropriate input
        # takes forever if inputs[1] tanh input has extended shape , i.e. (1,1) as opposed to scalar shape (1,)
        # rho = jnp.maximum(self.lower_rho_cutoff, inputs[0]) #Prevents division by 0
        # rho = rho.flatten()
        # sigma = jnp.maximum(self.lower_rho_cutoff, inputs[1]) #Prevents division by 0
        # sigma = sigma.flatten()
        rho = inputs[0]
        sigma = inputs[1]
        tau = inputs[3]
        tau_w = sigma/(8*rho)
        tau_unif = (3/10)*(3*jnp.pi**2)**(2/3)*rho**(5/3)
        alpha = ((tau - tau_w)/tau_unif).flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = jnp.sqrt(sigma) / (2 * k_F * rho)
        s = s.flatten()
        tanhterm = jnp.tanh(s)**2 + jnp.tanh(alpha-1)**2
        netterm = self.net(jnp.array([s, alpha]).flatten())
        lobterm = self.lobf(tanhterm*netterm)
        return 1+lobterm.squeeze()


class MGGA_FxNet_sigma_transform(eqx.Module):
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    lower_rho_cutoff: float
    net: eqx.nn.MLP
    lobf: eqx.Module
    name: str

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=1.174, lower_rho_cutoff=1e-12):
        '''
        Constructor for the exchange enhancement factor object, for the MGGA case.

        In a MGGA XC functional, the relevant quantities are (rho, grad_rho, laplacian_rho, tau=kinetic energy density). Here, 
        the network's input size is hard-coded to 2 -- just the gradient and alpha (related to tau) information 
        is passed to the network, to guarantee that the energy yielded from this multiplicative 
        factor behaves correctly under uniform scaling of the electron density and obeys the 
        spin-scaling relation.

        This network transforms function inputs [rho, sigma, lapl, tau] to the below inputs for the network:
        rho -> log(rho**1/3 + 1e-5) [not sent to network itself, here, due to contraints]
        sigma -> (1-exp(-s**2))*log(s+1)
        tau -> log((alpha+1)/2)

        :param depth: Depth of the neural network
        :type depth: int
        :param nodes: Number of nodes in each layer
        :type nodes: int
        :param seed: The random seed to initiate baseline weight values for the network
        :type seed: int
        :param lob_lim: The Lieb-Oxford bound to respect, defaults to 1.804
        :type lob_lim: float, optional
        :param lower_rho_cutoff: a cut-off to bypass potential division by zero in the division by rho, defaults to 1e-12
        :type lower_rho_cutoff: float, optional
        '''
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.lob_lim = lob_lim
        self.lower_rho_cutoff = lower_rho_cutoff
        # to constrain this, we require only gradient inputs
        self.net = eqx.nn.MLP(in_size=2,  # Input is gradient_descriptor, tau_descriptor
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)
        self.name = 'MGGA_FxNet_sigma_transform'

    def __call__(self, inputs):
        '''
        The network's forward pass, resulting in the enhancement factor associated to the input gradient descriptor.

        *NOTE*: This forward pass is explicitly NOT vectorized -- it expects one grid point worth of data, the (rho, gradient_descriptor) values at that point. This structure expects the :jax.vmap: call to be coded OUTSIDE of the network class.

        *NOTE*: Here, the gradient_descriptor is assumed to be Libxc's/PySCF's internal variable for the density gradient -- sigma (gradient squared in non-spin-polarized, gradient contracted with itself in spin-polarized). This is so that we have easy access to automatic derivatives with respect to sigma, thus can generate v_sigma and use in convergence testing. However, within the call sigma is translated to the reduced density gradient, :s:, which the network is still assumed to be parameterized by, and the call is structured in such a way to respect the UEG limits for when gradients vanish. Namely, when s = 0, Fx = 1, so the resulting e = Fx*e_heg = e_heg.

        :param inputs: A one-dimensional list/array of inputs [rho, sigma, laplacian_rho, alpha]
        :type inputs: tuple, list, one-dimensional array of size 4 in order [rho, sigma, laplacian_rho, alpha]
        :return: The enhancement factor value
        :rtype: float
        '''
        # here, assume the inputs is [rho, sigma, laplacian, tau] and select the appropriate input
        # takes forever if inputs[1] tanh input has extended shape , i.e. (1,1) as opposed to scalar shape (1,)
        # rho = jnp.maximum(self.lower_rho_cutoff, inputs[0]) #Prevents division by 0
        # rho = rho.flatten()
        # sigma = jnp.maximum(self.lower_rho_cutoff, inputs[1]) #Prevents division by 0
        # sigma = sigma.flatten()
        rho = inputs[0]
        sigma = inputs[1]
        tau = inputs[3]
        tau_w = sigma/(8*rho)
        tau_unif = (3/10)*(3*jnp.pi**2)**(2/3)*rho**(5/3)
        alpha = ((tau - tau_w)/tau_unif).flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = jnp.sqrt(sigma) / (2 * k_F * rho)
        s = s.flatten()
        # here we log-transform our descriptors to see if it improves convergence
        x0 = jnp.log(rho**1/3+1e-5)
        x1 = (1-jnp.exp(-s**2))*jnp.log(s+1)
        x2 = jnp.log((alpha+1)/2)
        # the tanh term here to match xcdiff paper
        tanhterm = x1 + jnp.tanh(x2)**2
        netterm = self.net(jnp.array([x1, x2]).flatten())
        lobterm = self.lobf(tanhterm*netterm)
        return 1+lobterm.squeeze()


class MGGA_FcNet_sigma(eqx.Module):
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    lower_rho_cutoff: float
    net: eqx.nn.MLP
    lobf: eqx.Module
    name: str

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=1.174, lower_rho_cutoff=1e-12):
        '''
        Constructor for the correlation enhancement factor object, for the MGGA case.

        In a MGGA XC functional, the relevant quantities are (rho, grad_rho, laplacian_rho, tau=kinetic energy density). Here, 
        the network's input size is hard-coded to 3 -- just the density, gradient, and alpha (related to tau) information 
        is passed to the network.

        :param depth: Depth of the neural network
        :type depth: int
        :param nodes: Number of nodes in each layer
        :type nodes: int
        :param seed: The random seed to initiate baseline weight values for the network
        :type seed: int
        :param lob_lim: The Lieb-Oxford bound to respect, defaults to 1.804
        :type lob_lim: float, optional
        :param lower_rho_cutoff: a cut-off to bypass potential division by zero in the division by rho, defaults to 1e-12
        :type lower_rho_cutoff: float, optional
        '''
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.lob_lim = lob_lim
        self.lower_rho_cutoff = lower_rho_cutoff
        # to constrain this, we require only gradient inputs
        self.net = eqx.nn.MLP(in_size=3,  # Input is all rho, gradient, tau descriptors
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)
        self.name = 'MGGA_FcNet_sigma'

    def __call__(self, inputs):
        '''
        The network's forward pass, resulting in the enhancement factor associated to the input gradient descriptor.

        *NOTE*: This forward pass is explicitly NOT vectorized -- it expects one grid point worth of data, the (rho, gradient_descriptor) values at that point. This structure expects the :jax.vmap: call to be coded OUTSIDE of the network class.

        *NOTE*: Here, the gradient_descriptor is assumed to be Libxc's/PySCF's internal variable for the density gradient -- sigma (gradient squared in non-spin-polarized, gradient contracted with itself in spin-polarized). This is so that we have easy access to automatic derivatives with respect to sigma, thus can generate v_sigma and use in convergence testing. However, within the call sigma is translated to the reduced density gradient, :s:, which the network is still assumed to be parameterized by, and the call is structured in such a way to respect the UEG limits for when gradients vanish. Namely, when s = 0, Fx = 1, so the resulting e = Fx*e_heg = e_heg.

        :param inputs: A one-dimensional list/array of inputs [rho, sigma, laplacian_rho, alpha]
        :type inputs: tuple, list, one-dimensional array of size 4 in order [rho, sigma, laplacian_rho, alpha]
        :return: The enhancement factor value
        :rtype: float
        '''
        # here, assume the inputs is [rho, sigma, laplacian, tau] and select the appropriate input
        # takes forever if inputs[1] tanh input has extended shape , i.e. (1,1) as opposed to scalar shape (1,)
        # rho = jnp.maximum(self.lower_rho_cutoff, inputs[0]) #Prevents division by 0
        # rho = rho.flatten()
        # sigma = jnp.maximum(self.lower_rho_cutoff, inputs[1]) #Prevents division by 0
        # sigma = sigma.flatten()
        rho = inputs[0].flatten()
        sigma = inputs[1]
        tau = inputs[3]
        tau_w = sigma/(8*rho)
        tau_unif = (3/10)*(3*jnp.pi**2)**(2/3)*rho**(5/3)
        alpha = ((tau - tau_w)/tau_unif).flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = jnp.sqrt(sigma) / (2 * k_F * rho)
        s = s.flatten()
        tanhterm = jnp.tanh(s)**2 + jnp.tanh(alpha-1)**2
        netterm = self.net(jnp.array([rho, s, alpha]).flatten())
        lobterm = self.lobf(tanhterm*netterm)
        return 1+lobterm.squeeze()


class MGGA_FcNet_sigma_transform(eqx.Module):
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    lower_rho_cutoff: float
    net: eqx.nn.MLP
    lobf: eqx.Module
    name: str

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=1.174, lower_rho_cutoff=1e-12):
        '''
        Constructor for the correlation enhancement factor object, for the MGGA case.

        In a MGGA XC functional, the relevant quantities are (rho, grad_rho, laplacian_rho, tau=kinetic energy density). Here, 
        the network's input size is hard-coded to 3 -- just the density, gradient, and alpha (related to tau) information 
        is passed to the network.

        This network transforms function inputs [rho, sigma, lapl, tau] to the below inputs for the network:
        rho -> log(rho**1/3 + 1e-5)
        sigma -> (1-exp(-s**2))*log(s+1)
        tau -> log((alpha+1)/2)

        :param depth: Depth of the neural network
        :type depth: int
        :param nodes: Number of nodes in each layer
        :type nodes: int
        :param seed: The random seed to initiate baseline weight values for the network
        :type seed: int
        :param lob_lim: The Lieb-Oxford bound to respect, defaults to 1.804
        :type lob_lim: float, optional
        :param lower_rho_cutoff: a cut-off to bypass potential division by zero in the division by rho, defaults to 1e-12
        :type lower_rho_cutoff: float, optional
        '''
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.lob_lim = lob_lim
        self.lower_rho_cutoff = lower_rho_cutoff
        # to constrain this, we require only gradient inputs
        self.net = eqx.nn.MLP(in_size=3,  # Input is all rho, gradient, tau descriptors
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)
        self.name = 'MGGA_FcNet_sigma_transform'

    def __call__(self, inputs):
        '''
        The network's forward pass, resulting in the enhancement factor associated to the input gradient descriptor.

        *NOTE*: This forward pass is explicitly NOT vectorized -- it expects one grid point worth of data, the (rho, gradient_descriptor) values at that point. This structure expects the :jax.vmap: call to be coded OUTSIDE of the network class.

        *NOTE*: Here, the gradient_descriptor is assumed to be Libxc's/PySCF's internal variable for the density gradient -- sigma (gradient squared in non-spin-polarized, gradient contracted with itself in spin-polarized). This is so that we have easy access to automatic derivatives with respect to sigma, thus can generate v_sigma and use in convergence testing. However, within the call sigma is translated to the reduced density gradient, :s:, which the network is still assumed to be parameterized by, and the call is structured in such a way to respect the UEG limits for when gradients vanish. Namely, when s = 0, Fx = 1, so the resulting e = Fx*e_heg = e_heg.

        :param inputs: A one-dimensional list/array of inputs [rho, sigma, laplacian_rho, alpha]
        :type inputs: tuple, list, one-dimensional array of size 4 in order [rho, sigma, laplacian_rho, alpha]
        :return: The enhancement factor value
        :rtype: float
        '''
        # here, assume the inputs is [rho, sigma, laplacian, tau] and select the appropriate input
        # takes forever if inputs[1] tanh input has extended shape , i.e. (1,1) as opposed to scalar shape (1,)
        # rho = jnp.maximum(self.lower_rho_cutoff, inputs[0]) #Prevents division by 0
        # rho = rho.flatten()
        # sigma = jnp.maximum(self.lower_rho_cutoff, inputs[1]) #Prevents division by 0
        # sigma = sigma.flatten()
        rho = inputs[0].flatten()
        sigma = inputs[1]
        tau = inputs[3]
        tau_w = sigma/(8*rho)
        tau_unif = (3/10)*(3*jnp.pi**2)**(2/3)*rho**(5/3)
        alpha = ((tau - tau_w)/tau_unif).flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = jnp.sqrt(sigma) / (2 * k_F * rho)
        s = s.flatten()
        # here we log-transform our descriptors to see if it improves convergence
        x0 = jnp.log(rho**1/3+1e-5)
        x1 = (1-jnp.exp(-s**2))*jnp.log(s+1)
        x2 = jnp.log((alpha+1)/2)
        # the tanh term here to match xcdiff paper
        tanhterm = x1 + jnp.tanh(x2)**2
        netterm = self.net(jnp.array([x0, x1, x2]).flatten())
        lobterm = self.lobf(tanhterm*netterm)
        return 1+lobterm.squeeze()


# =====================================================================
# =====================================================================
# DEPRECATED CLASSES -- TO BE REMOVED
# =====================================================================
# =====================================================================
class eX(eqx.Module):
    n_input: int
    n_hidden: int
    ueg_limit: jax.Array
    spin_scaling: bool
    lob: jax.Array
    use: list
    net: eqx.Module
    tanh: jax.named_call
    lobf: jax.named_call
    sig: jax.named_call
    shift: jax.Array
    lobf: eqx.Module
    seed: int
    depth: int

    def __init__(self, n_input, n_hidden=16, depth=3, use=[], ueg_limit=False, lob=1.804, seed=92017, spin_scaling=True):
        """
        __init__ Local exchange model based on MLP.

        Receives density descriptors in this order : [rho, s, alpha, nl], where the input may be truncated depending on XC-level of approximation.

        The MLP generated is hard-coded to have one output value -- the predicted exchange energy given a specific input from the grid.

        :param n_input: Input dimensions (LDA: 1, GGA: 2, meta-GGA: 3, ...)
        :type n_input: int
        :param n_hidden: Number of hidden nodes (three hidden layers used by default), defaults to 16
        :type n_hidden: int, optional
        :param depth: Depth of the MLP, defaults to 3
        :type depth: int, optional
        :param use: Only these indices are used as input to the model (can be used to omit density as input to enforce uniform density scaling). These indices are also used to enforce UEG where the assumed order is [s, alpha, ...], defaults to []
        :type use: list, optional
        :param ueg_limit: Flag to determine whether or not to enforce uniform homoegeneous electron gas limit, defaults to False
        :type ueg_limit: bool, optional
        :param lob: Enforce this value as local Lieb-Oxford bound (don't enforce if set to 0), defaults to 1.804
        :type lob: float, optional
        :param seed: Random seed used to generate initial weights and biases for the MLP, defaults to 92017
        :type seed: int, optional
        """
        super().__init__()
        self.ueg_limit = ueg_limit
        self.spin_scaling = spin_scaling
        self.lob = lob
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.seed = seed
        self.depth = depth

        if not use:
            self.use = jnp.arange(n_input)
        else:
            self.use = use
        self.net = eqx.nn.MLP(in_size=self.n_input,
                              out_size=1,
                              width_size=self.n_hidden,
                              depth=self.depth,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))

        self.tanh = jnp.tanh
        self.lobf = LOB(limit=self.lob)
        self.sig = jax.nn.sigmoid
        self.shift = 1/(1+jnp.exp(-1e-3))

        warn("WARNING - DEPRECATED. This class is not the currently working class and will be removed in the future.")

    def __call__(self, rho, **kwargs):
        """
        __call__ Forward pass for the exchange network.

        Uses :jax.vmap: to vectorize evaluation of the MLP on the descriptors, assuming a shape [batch, *, n_input]

        .. todo: Make sure the :vmap: call can work with specific :use: values beyond the defaults assumed in the previous implementation.

        :param rho: The descriptors to the MLP -- transformed densities and gradients appropriate to the XC-level. This network will only use the dimensions specified in self.use.
        :type rho: jax.Array
        :return: The exchange energy on the grid
        :rtype: jax.Array
        """
        print(f"eX.__call__, rho shape: {rho.shape}")
        print(f"eX.__call__, rho nans: {jnp.isnan(rho).sum()}")
        if self.spin_scaling:
            squeezed = jnp.squeeze(jax.vmap(jax.vmap(self.net), in_axes=1)(rho[..., self.use])).T
        else:
            squeezed = jnp.squeeze(jax.vmap(self.net)(rho[..., self.use]))

        if self.ueg_limit:
            ueg_lim = rho[..., self.use[0]]
            if len(self.use) > 1:
                ueg_lim_a = jnp.power(self.tanh(rho[..., self.use[1]]), 2)
            else:
                ueg_lim_a = 0
            if len(self.use) > 2:
                ueg_lim_nl = jnp.sum(rho[..., self.use[2:]], axis=-1)
            else:
                ueg_lim_nl = 0
        else:
            ueg_lim = 1
            ueg_lim_a = 0
            ueg_lim_nl = 0

        if self.lob:
            result = self.lobf(squeezed*(ueg_lim + ueg_lim_a + ueg_lim_nl))
        else:
            result = squeezed*(ueg_lim + ueg_lim_a + ueg_lim_nl)

        return result


# DEPRECATED
class eC(eqx.Module):
    n_input: int
    n_hidden: int
    ueg_limit: jax.Array
    spin_scaling: bool
    lob: jax.Array
    use: list
    net: eqx.Module
    tanh: jax.named_call
    lobf: jax.named_call
    sig: jax.named_call
    lobf: eqx.Module
    seed: int
    depth: int

    def __init__(self, n_input=2, n_hidden=16, depth=3, use=[], ueg_limit=False, lob=2.0, seed=92017, spin_scaling=False):
        """
        __init__ Local correlation model based on MLP.

        Receives density descriptors in this order : [rho, spinscale, s, alpha, nl], where the input may be truncated depending on XC-level of approximation

        .. todo: Make sure the :vmap: call can work with specific :use: values beyond the defaults assumed in the previous implementation.

        :param n_input: Input dimensions (LDA: 2, GGA: 3 , meta-GGA: 4), defaults to 2.
        :type n_input: int
        :param n_hidden: Number of hidden nodes (three hidden layers used by default), defaults to 16
        :type n_hidden: int, optional
        :param depth: Depth of the MLP, defaults to 3
        :type depth: int, optional
        :param use: Only these indices are used as input to the model. These indices are also used to enforce UEG where the assumed order is [s, alpha, ...], defaults to []
        :type use: list, optional
        :param ueg_limit: Flag to determine whether or not to enforce uniform homoegeneous electron gas limit, defaults to False
        :type ueg_limit: bool, optional
        :param lob: Enforce this value as local Lieb-Oxford bound (don't enforce if set to 0), defaults to 2.0
        :type lob: float, optional
        :param seed: Random seed used to generate initial weights and biases for the MLP, defaults to 92017
        :type seed: int, optional
        """
        super().__init__()
        self.spin_scaling = spin_scaling
        self.lob = False
        self.ueg_limit = ueg_limit
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.seed = seed
        self.depth = depth

        if not use:
            self.use = jnp.arange(n_input)
        else:
            self.use = use
        self.net = eqx.nn.MLP(in_size=self.n_input,
                              out_size=1,
                              width_size=self.n_hidden,
                              depth=self.depth,
                              activation=jax.nn.gelu,
                              final_activation=jax.nn.softplus,
                              key=jax.random.PRNGKey(self.seed))
        self.sig = jax.nn.sigmoid
        self.tanh = jnp.tanh
        self.lob = lob
        if self.lob:
            self.lobf = LOB(self.lob)
        else:
            self.lob = 1000.0
            self.lobf = LOB(self.lob)
        warn("WARNING - DEPRECATED. This class is not the currently working class and will be removed in the future.")

    def __call__(self, rho, **kwargs):
        """
        __call__ Forward pass for the correlation network.

        Uses :jax.vmap: to vectorize evaluation of the MLP on the descriptors, assuming a shape [*, n_input]

        :param rho: The descriptors to the MLP -- transformed densities and gradients appropriate to the XC-level. This network will only use the dimensions specified in self.use in determining the UEG limits.
        :type rho: jax.Array
        :return: The exchange energy on the grid
        :rtype: jax.Array
        """
        print(f"eC.__call__, rho shape: {rho.shape}")
        print(f"eC.__call__, rho nans: {jnp.isnan(rho).sum()}")
        if self.spin_scaling:
            squeezed = -jnp.squeeze(jax.vmap(jax.vmap(self.net), in_axes=1)(rho[..., self.use])).T
        else:
            squeezed = -jnp.squeeze(jax.vmap(self.net)(rho[..., self.use]))

        if self.ueg_limit:
            ueg_lim = self.tanh(rho[..., self.use[0]])
            if len(self.use) > 1:
                ueg_lim_a = jnp.pow(self.tanh(rho[..., self.use[1]]), 2)
            else:
                ueg_lim_a = 0
            if len(self.use) > 2:
                ueg_lim_nl = jnp.sum(self.tanh(rho[..., self.use[2:]])**2, axis=-1)
            else:
                ueg_lim_nl = 0

            ueg_factor = ueg_lim + ueg_lim_a + ueg_lim_nl
        else:
            ueg_factor = 1
        if self.lob:
            return self.lobf(squeezed*ueg_factor)
        else:
            return squeezed*ueg_factor


def make_net(xorc, level, depth, nhidden, ninput=None, use=None, spin_scaling=None, lob=None, ueg_limit=None,
             random_seed=None, savepath=None, configfile='network.config'):
    '''
    make_net is a utility function designed to easily create new, individual exchange or correlation networks with ease. If no extra arguments are specified, the network will be generated with a default structure that respects the various constraints implemented within xcquinox

    :param xorc: 'X' or 'C' -- the type of network to generate, exchange or correlation
    :type xorc: str
    :param level: one of ['GGA', 'MGGA', 'NONLOCAL', 'NL'], indicating the desired rung of Jacob's Ladder. NONLOCAL = NL
    :type level: str
    :param depth: The number of hidden layers in the generated network.
    :type depth: int
    :param nhidden: The number of nodes in a hidden layer
    :type nhidden: int
    :param ninput: The number of inputs the network will expect, defaults to None for automatic selection based on level
    :type ninput: int, optional
    :param use: The indices of the descriptors to evaluate the network on, defaults to None
    :type use: list of ints, optional
    :param spin_scaling: Whether or not to enforce the spin-scaling contraint in the generated network, defaults to None
    :type spin_scaling: bool, optional
    :param lob: Lieb-Oxford bound: If non-zero (i.e., truthy), the output values of e_x or e_c will be squashed between [-1, lob-1], defaults to None
    :type lob: float, optional
    :param ueg_limit: Whether or not to enforce the UEG scaling constraint, defaults to None
    :type ueg_limit: bool, optional
    :param random_seed: The random seed to use in generating initial network weights, defaults to None
    :type random_seed: int, optional
    :param savepath: Location to save the generated network and associated config file, defaults to None
    :type savepath: str, optional
    :param configfile: Name for the configuration file, needed when reading in the network to re-generate the same structure, defaults to 'network.config'
    :type configfile: str, optional
    :return: The resulting exchange or correlation network.
    :rtype: :xcquinox.net.eX: or :xcquinox.net.eC:
    '''
    defaults_dct = {'GGA': {'X': {'ninput': 1, 'depth': 3, 'nhidden': 16, 'use': [1], 'spin_scaling': True, 'lob': 1.804, 'ueg_limit': True},
                            'C': {'ninput': 1, 'depth': 3, 'nhidden': 16, 'use': [2], 'spin_scaling': False, 'lob': 2.0, 'ueg_limit': True}
                            },
                    'MGGA': {'X': {'ninput': 2, 'depth': 3, 'nhidden': 16, 'use': [1, 2], 'spin_scaling': True, 'lob': 1.174, 'ueg_limit': True},
                             'C': {'ninput': 2, 'depth': 3, 'nhidden': 16, 'use': [2, 3], 'spin_scaling': False, 'lob': 2.0, 'ueg_limit': True}
                             },
                    'NONLOCAL': {'X': {'ninput': 15, 'depth': 3, 'nhidden': 16, 'use': None, 'spin_scaling': True, 'lob': 1.174, 'ueg_limit': True},
                                 'C': {'ninput': 16, 'depth': 3, 'nhidden': 16, 'use': None, 'spin_scaling': False, 'lob': 2.0, 'ueg_limit': True}
                                 },
                    'NL': {'X': {'ninput': 15, 'depth': 3, 'nhidden': 16, 'use': None, 'spin_scaling': True, 'lob': 1.174, 'ueg_limit': True},
                           'C': {'ninput': 16, 'depth': 3, 'nhidden': 16, 'use': None, 'spin_scaling': False, 'lob': 2.0, 'ueg_limit': True}
                           }
                    }
    assert level.upper() in ['GGA', 'MGGA', 'NONLOCAL', 'NL']

    ninput = ninput if ninput is not None else defaults_dct[level.upper()][xorc.upper()]['ninput']
    depth = depth if depth is not None else defaults_dct[level.upper()][xorc.upper()]['depth']
    nhidden = nhidden if nhidden is not None else defaults_dct[level.upper()][xorc.upper()]['nhidden']
    use = use if use is not None else defaults_dct[level.upper()][xorc.upper()]['use']
    spin_scaling = spin_scaling if spin_scaling is not None else defaults_dct[level.upper(
    )][xorc.upper()]['spin_scaling']
    ueg_limit = ueg_limit if ueg_limit is not None else defaults_dct[level.upper()][xorc.upper()]['ueg_limit']
    lob = lob if lob is not None else defaults_dct[level.upper()][xorc.upper()]['lob']
    random_seed = random_seed if random_seed is not None else 92017
    config = {'ninput': ninput,
              'depth': depth,
              'nhidden': nhidden,
              'use': use,
              'spin_scaling': spin_scaling,
              'ueg_limit': ueg_limit,
              'lob': lob,
              'random_seed': random_seed}
    if xorc.upper() == 'X':
        net = eX(n_input=ninput, use=use, depth=depth, n_hidden=nhidden,
                 spin_scaling=spin_scaling, lob=lob, seed=random_seed)
    elif xorc.upper() == 'C':
        net = eC(n_input=ninput, use=use, depth=depth, n_hidden=nhidden,
                 spin_scaling=spin_scaling, lob=lob, seed=random_seed)

    if savepath:
        try:
            os.makedirs(savepath)
        except Exception as e:
            print(e)
            print(f'Exception raised in creating {savepath}.')
        with open(os.path.join(savepath, configfile), 'w') as f:
            for k, v in config.items():
                f.write(f'{k}\t{v}\n')
        with open(os.path.join(savepath, configfile+'.pkl'), 'wb') as f:
            pickle.dump(config, f)
        eqx.tree_serialise_leaves(os.path.join(savepath, 'xc.eqx'), net)

    return net, config


def get_net(xorc, level, net_path, configfile='network.config', netfile='xc.eqx'):
    '''
    A utility function to easily load in a previously generated network. Functionally creates a random network of the same architecture, then overwrites the weights with those of the saved network.

    :param xorc: 'X' or 'C' -- the type of network to generate, exchange or correlation
    :type xorc: str
    :param level: one of ['GGA', 'MGGA', 'NONLOCAL', 'NL'], indicating the desired rung of Jacob's Ladder. NONLOCAL = NL
    :type level: str
    :param net_path: Location of the saved network. Must have a {configfile}.pkl parameter file within.
    :type net_path: str
    :param configfile: Name for the configuration file, needed when reading in the network to re-generate the same structure, defaults to 'network.config'
    :type configfile: str, optional
    :param netfile: Name for the network file, needed when reading in the network overwrite generated random weights, defaults to 'xc.eqx'. If Falsy, just generates random network based on config file.
    :type netfile: str, optional
    :return: The requested exchange or correlation network.
    :rtype: :xcquinox.net.eX: or :xcquinox.net.eC:
    '''
    with open(os.path.join(net_path, configfile+'.pkl'), 'rb') as f:
        params = pickle.load(f)
    # network parameters
    depth = params['depth']
    nodes = params['nhidden']
    use = params['use']
    inp = params['ninput']
    ss = params['spin_scaling']
    lob = params['lob']
    ueg = params['ueg_limit']
    seed = params['random_seed']

    net, _ = make_net(xorc=xorc, level=level, depth=depth, nhidden=nodes, ninput=inp, use=use,
                      spin_scaling=ss, lob=lob, ueg_limit=ueg, random_seed=seed, configfile=configfile)
    if netfile:
        # make sure the netfile is actually there
        netfs = [i for i in os.listdir(net_path) if netfile in i]
        # if multiple returned, there was training to take place -- sort and select last checkpoint
        if len(netfs) == 1:
            print('SINGLE NETFILE MATCH FOUND. DESERIALIZING...')
            net = eqx.tree_deserialise_leaves(os.path.join(net_path, netfs[0]), net)
        elif len(netfs) > 1:
            print('NETFILE MATCHES FOUND -- MULTIPLE. SELECTING LAST ONE.')
            netf = sorted(netfs, key=lambda x: int(x.split('.')[-1]))[-1]
            print('ATTEMPTING TO DESERIALIZE {}'.format(netf))
            net = eqx.tree_deserialise_leaves(os.path.join(net_path, netf), net)
        else:
            print('NETFILE SPECIFIED BUT NO MATCHING FILE FOUND.')

    return net, params
