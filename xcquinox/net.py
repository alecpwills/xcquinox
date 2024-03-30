import jax
import jax.numpy as jnp
import equinox as eqx


class LOB(eqx.Module):
    limit: jax.Array
    sig: jax.named_call
    
    
    def __init__(self, limit=1.804):
        """
        __init__ Utility function to squash output to [-1, limit-1] inteval.

        Can be used to enforce non-negativity, Lieb-Oxford bounds, etc.
        Initializes method :self.sig: -- the :jax.nn.sigmoid: function, as well.

        :param limit: The Lieb-Oxford bound value to impose, defaults to 1.804
        :type limit: float, optional
        """
        super().__init__()
        self.sig = jax.nn.sigmoid
        self.limit = limit

    def __call__(self, x):
        """
        __call__ Method calling the actual mapping of the input to the desired bounded region.


        :param x: Energy value to map back into the bounded region.
        :type x: float
        :return: Energy value mapped into bounded region.
        :rtype: float
        """
        return self.limit*self.sig(x-jnp.log(self.limit-1))-1


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

    def __init__(self, n_input, n_hidden=16, depth=3, use=[], ueg_limit=False, lob=1.804, seed=92017):
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
        self.spin_scaling = True
        self.lob = lob
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.seed = seed
        self.depth = depth

        if not use:
            self.use = jnp.arange(n_input)
        else:
            self.use = use
        self.net =  eqx.nn.MLP(in_size = self.n_input,
                               out_size = 1,
                               width_size = self.n_hidden,
                               depth = self.depth,
                               activation = jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        
        self.tanh = jnp.tanh
        self.lobf = LOB(limit=self.lob)
        self.sig = jax.nn.sigmoid
        self.shift = 1/(1+jnp.exp(-1e-3))

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
        squeezed = jnp.squeeze(jax.vmap(jax.vmap(self.net), in_axes=1)(rho[...,self.use])).T
        if self.ueg_limit:
            ueg_lim = rho[...,self.use[0]]
            if len(self.use) > 1:
                ueg_lim_a = jnp.power(self.tanh(rho[...,self.use[1]]),2)
            else:
                ueg_lim_a = 0
            if len(self.use) > 2:
                ueg_lim_nl = jnp.sum(rho[...,self.use[2:]],dim=-1)
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

    def __init__(self, n_input=2,n_hidden=16, depth=3, use = [], ueg_limit=False, lob=2.0, seed=92017):
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
        self.spin_scaling = False
        self.lob = False
        self.ueg_limit = ueg_limit
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.seed = seed
        self.depth = depth

        if not use:
            self.use = jnp.arange(n_input)
        else:
            self.use = use
        self.net =  eqx.nn.MLP(in_size = self.n_input,
                               out_size = 1,
                               width_size = self.n_hidden,
                               depth = self.depth,
                               activation = jax.nn.gelu,
                               final_activation = jax.nn.softplus,
                               key=jax.random.PRNGKey(self.seed))
        self.sig = jax.nn.sigmoid
        self.tanh = jnp.tanh
        self.lob = lob
        if self.lob:
            self.lobf = LOB(self.lob)
        else:
            self.lob =  1000.0
            self.lobf = LOB(self.lob)


    def __call__(self, rho, **kwargs):
        """
        __call__ Forward pass for the correlation network.

        Uses :jax.vmap: to vectorize evaluation of the MLP on the descriptors, assuming a shape [*, n_input]

        :param rho: The descriptors to the MLP -- transformed densities and gradients appropriate to the XC-level. This network will only use the dimensions specified in self.use in determining the UEG limits.
        :type rho: jax.Array
        :return: The exchange energy on the grid
        :rtype: jax.Array
        """
        squeezed = jnp.squeeze(-jax.vmap(self.net)(rho))

        if self.ueg_limit:
            ueg_lim = self.tanh(rho[...,self.use[0]])
            if len(self.use) > 1:
                ueg_lim_a = jnp.pow(self.tanh(rho[...,self.use[1]]),2)
            else:
                ueg_lim_a = 0
            if len(self.use) > 2:
                ueg_lim_nl = jnp.sum(self.tanh(rho[...,self.use[2:]])**2,axis=-1)
            else:
                ueg_lim_nl = 0

            ueg_factor = ueg_lim + ueg_lim_a + ueg_lim_nl
        else:
            ueg_factor = 1
        if self.lob:
            return self.lobf(squeezed*ueg_factor)
        else:
            return squeezed*ueg_factor