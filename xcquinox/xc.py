import numpy as np
import equinox as eqx
import jax, os, sys
import jax.numpy as jnp

#check if cider is in the environment
from mldftdat.density import get_exchange_descriptors2
from mldftdat.analyzers import RKSAnalyzer

class LDA_X(eqx.Module):
    def __init__(self):
        """
        __init__ Constructs an object whose forward pass computes the LDA exchange energy based on a given density input.

        """
        super().__init__()

    def __call__(self, rho):
        """
        __call__ Computes the LDA exchange energy for a given value of the density.

        .. math:: E_x = -\\frac{3}{4} \\Big(\\frac{3\\rho}{\\pi} \\Big)^{1/3}

        :param rho: The value of the density
        :type rho: float, broadcastable
        :return: The LDA exchange energy for the input value(s).
        :rtype: float, broadcastable
        """
        return -3/4*(3/np.pi)**(1/3)*rho**(1/3)
        
params_a_pp     = [1,  1,  1]
params_a_alpha1 = [0.21370,  0.20548,  0.11125]
params_a_a      = [0.031091, 0.015545, 0.016887]
params_a_beta1  = [7.5957, 14.1189, 10.357]
params_a_beta2  = [3.5876, 6.1977, 3.6231]
params_a_beta3  = [1.6382, 3.3662,  0.88026]
params_a_beta4  = [0.49294, 0.62517, 0.49671]
params_a_fz20   = 1.709921
       
class PW_C(eqx.Module):
    
    def __init__(self):
        """
        __init__ Constructs an object whose forward pass computes the UEG correlation energy, as parameterized by Perdew & Wang

        DOI: `10.1103/PhysRevB.45.13244`_.

        .. _10.1103/PhysRevB.45.13244: https://doi.org/10.1103/PhysRevB.45.13244

        """
        super().__init__()
        
    def __call__(self, rs, zeta):
        """
        __call__ Forward pass, computing the correlation energy per electron for the given input values, rs and zeta, where
        
        .. math:: r_s = \\Big[\\frac{3}{4\\pi (\\rho_\\uparrow+\\rho_\\downarrow)} \\Big]^{1/3}

        .. math:: \\zeta = \\frac{\\rho_\\uparrow-\\rho_\\downarrow}{\\rho_\\uparrow+\\rho_\\downarrow}

        Atomic units are assumed.


        :param rs: The Wigner-Seiz radius corresponding to the given density value
        :type rs: float, broadcastable
        :param zeta: The spin-polarization
        :type zeta: float, broadcastable
        """
        def g_aux(k, rs):
            return params_a_beta1[k]*jnp.sqrt(rs) + params_a_beta2[k]*rs\
          + params_a_beta3[k]*rs**1.5 + params_a_beta4[k]*rs**(params_a_pp[k] + 1)

        def g(k, rs):
            return -2*params_a_a[k]*(1 + params_a_alpha1[k]*rs)\
          * jnp.log(1 +  1/(2*params_a_a[k]*g_aux(k, rs)))

        def f_zeta(zeta):
            return ((1+zeta)**(4/3) + (1-zeta)**(4/3) - 2)/(2**(4/3)-2)

        def f_pw(rs, zeta):
            return g(0, rs) + zeta**4*f_zeta(zeta)*(g(1, rs) - g(0, rs) + g(2, rs)/params_a_fz20)\
          - f_zeta(zeta)*g(2, rs)/params_a_fz20

        return f_pw(rs, zeta)

class eXC(eqx.Module):
    grid_models: list
    heg_mult: bool
    pw_mult: bool
    level: int
    exx_a: jax.Array
    epsilon: jax.Array
    loge: jax.Array
    s_gam: jax.Array
    heg_model: eqx.Module
    pw_model: eqx.Module
    model_mult: list
    debug: bool
    nlstart_i: int
    nlend_i: int
    verbose: bool
    
    def __init__(self, grid_models=[], heg_mult=True, pw_mult=True,
                    level = 1, exx_a=None, epsilon=1e-8, debug=False,
                nlstart_i = 3, nlend_i = 12, verbose=False):
        """
        __init__ Defines the XC functional

        Constructed with two MLPs -- one for the local exchange energy on the grid, the other for the local correlation energy.

        :param grid_models: list of eX (local exchange) or eC (local correlation). Defines the xc-models/enhancement factors, defaults to []
        :type grid_models: list, optional
        :param heg_mult: Use homoegeneous electron gas exchange (multiplicative if grid_models is not empty), defaults to True
        :type heg_mult: bool, optional
        :param pw_mult: Use homoegeneous electron gas correlation (Perdew & Wang), defaults to True
        :type pw_mult: bool, optional
        :param level: Controls the number of density "descriptors" generated. 1: LDA, 2: GGA, 3:meta-GGA, 4: meta-GGA + electrostatic (nonlocal), defaults to 1
        :type level: int, optional
        :param exx_a: Exact exchange mixing parameter, defaults to None
        :type exx_a: float, optional
        :param epsilon: Offset to avoid div/0 in calculations, defaults to 1e-8
        :type epsilon: float, optional
        :param debug: Controls printing of various stats throughout, defaults to False
        :type debug: bool, optional
        :param nlstart_i: If level > 3, this controls the number of CIDER Nonlocal parameters are selected, defaults to 3 as the first nonlocal CIDER descriptor
        :type nlstart_i: int, optional
        :param nlend_i: If level > 3, this controls the number of CIDER Nonlocal parameters are selected, defaults to 12 as the last nonlocal CIDER descriptor
        :type nlend_i: int, optional
        :param verbose: Flag to determine printing of a lot more extra information to the terminal, useful for debuggin, defaults to False
        :type verbose: bool, optional

        """
        super().__init__()
        self.heg_mult = heg_mult
        self.pw_mult = pw_mult
        self.level = level
        self.grid_models = grid_models
        self.epsilon = epsilon
        if level > 3:
            print('WARNING: External module "mldftdat" required for non-local descriptor use.')
        self.loge = 1e-5
        self.s_gam = 1
        self.debug = debug
        self.nlstart_i = nlstart_i
        self.nlend_i = nlend_i
        self.verbose = verbose

        if heg_mult:
            self.heg_model = LDA_X()
        if pw_mult:
            self.pw_model = PW_C()
        self.model_mult = [1 for m in self.grid_models]
        if not exx_a:
            self.exx_a = 0
        else:
            self.exx_a = exx_a

    def __call__(self, dm, ao_eval, grid_weights, mf = None, coor=None):
        """
        __call__ Forward call for the XC network to get the grid point e_xc

        Generates the density-on-grid from the density matrix, atomic orbital evaluation, and the grid weights from a :pyscfad: calculation.


        :param dm: Density matrix
        :type dm: jax.Array
        :param ao_eval: Atomic orbitals evaluated on the grid
        :type ao_eval: jax.Array
        :param grid_weights: Grid weights associated to the grid on which the atomic orbitals are evaluated
        :type grid_weights: jax.Array
        :param mf: A pyscf(ad) converged calculation kernel if self.level > 3, used for building the CIDER nonlocal descriptors, defaults to None
        :type mf: pyscfad.dft.RKS kernel
        :param coor: Grid coordinates associated to the grid on which the atomic orbitals are evaluated, passed into eval_grid_models, defaults to None
        :type coor: jax.Array
        :return: Exc, exchange-correlation energy from integrating the network calls across the grid
        :rtype: float
        """
        Exc = 0
        if self.level > 3:
            assert mf is not None
        if self.grid_models or self.heg_mult:
            if ao_eval.ndim==2:
                ao_eval = jnp.expand_dims(ao_eval,0)
            else:
                ao_eval = ao_eval

            # Create density (and gradients) from atomic orbitals evaluated on grid
            # and density matrix
            # rho[ijsp]: del_i phi del_j phi dm (s: spin, p: grid point index)
            rho = jnp.einsum('xij,yik,...jk->xy...i', ao_eval, ao_eval, dm)+1e-10
            
            rho0 = rho[0,0]
            drho = rho[0,1:4] + rho[1:4,0]
            tau = 0.5*(rho[1,1] + rho[2,2] + rho[3,3])

            non_loc = jnp.zeros_like(tau)

            if dm.ndim == 3: # If unrestricted (open-shell) calculation

                # Density
                rho0_a = rho0[0]
                rho0_b = rho0[1]

                # jnp.einsumed density gradient
                gamma_a, gamma_b = jnp.einsum('ij,ij->j',drho[:,0],drho[:,0]), jnp.einsum('ij,ij->j',drho[:,1],drho[:,1])
                gamma_ab = jnp.einsum('ij,ij->j',drho[:,0],drho[:,1])

                # Kinetic energy density
                tau_a, tau_b = tau

                # E.-static
                non_loc_a, non_loc_b = non_loc
            else:
                rho0_a = rho0_b = rho0*0.5
                gamma_a=gamma_b=gamma_ab= jnp.einsum('ij,ij->j',drho[:],drho[:])*0.25
                tau_a = tau_b = tau*0.5
                non_loc_a=non_loc_b = non_loc*0.5

            # xc-energy per unit particle
            exc = self.eval_grid_models(jnp.concatenate([jnp.expand_dims(rho0_a,-1),
                                                    jnp.expand_dims(rho0_b,-1),
                                                    jnp.expand_dims(gamma_a,-1),
                                                    jnp.expand_dims(gamma_ab,-1),
                                                    jnp.expand_dims(gamma_b,-1),
                                                    jnp.expand_dims(jnp.zeros_like(rho0_a),-1), #Dummy for laplacian
                                                    jnp.expand_dims(jnp.zeros_like(rho0_a),-1), #Dummy for laplacian
                                                    jnp.expand_dims(tau_a,-1),
                                                    jnp.expand_dims(tau_b,-1),
                                                    jnp.expand_dims(non_loc_a,-1),
                                                    jnp.expand_dims(non_loc_b,-1)],axis=-1),
                                       mf = mf, dm = dm, ao=ao_eval, gw=grid_weights, coor=coor)
            excnans = jnp.sum(jnp.isnan(exc[:, 0]))
            self.vprint(f'NaNs in exc from self.eval_grid_models = {excnans}')
            Exc += jnp.sum(((rho0_a + rho0_b)*exc[:,0])*grid_weights)
        return Exc
    
    def vprint(self, str):
        """
        Prints if verbose tag is set on the object

        :param str: The string to be printed
        :type str: str
        """
        if self.verbose:
            print(str)
            
    # Density (rho)
    def l_1(self, rho):
        """
        l_1 Level 1 (LDA-level) Descriptor -- Creates dimensionless quantity from rho.

        Equation 3 from the `base paper`_.

        .. _base paper: https://link.aps.org/doi/10.1103/PhysRevB.104.L161109

        .. math:: x_0 = \\rho^{1/3}


        :param rho: density
        :type rho: jax.Array
        :return: Scaled density
        :rtype: jax.Array
        """
        self.vprint('l_1, descr shape: {}'.format(rho.shape))
        descrnans = jnp.sum(jnp.isnan(rho))
        self.vprint(f'NaNs in descr from self.l_1 = {descrnans}')

        return rho**(1/3)

    # Reduced density gradient s
    def l_2(self, rho, gamma):
        """
        l_2 Level 2 (GGA-level) Descriptor -- Reduced gradient density

        Equation 5 from the `base paper`_.

        .. _base paper: https://link.aps.org/doi/10.1103/PhysRevB.104.L161109

        .. math:: x_2=s=\\frac{1}{2(3\\pi^2)^{1/3}} \\frac{|\\nabla \\rho|}{\\rho^{4/3}}

        
        :param rho: density
        :type rho: jax.Array
        :param gamma: squared density gradient
        :type gamma: jax.Array
        :return: reduced density gradient s
        :rtype: jax.Array
        """
        l2 = jnp.sqrt(gamma)/(2*(3*np.pi**2)**(1/3)*rho**(4/3)+self.epsilon)
        self.vprint('l_2, descr shape: {}'.format(l2.shape))
        descrnans = jnp.sum(jnp.isnan(l2))
        self.vprint(f'NaNs in descr from self.l_2 = {descrnans}')
        return l2

    # Reduced kinetic energy density alpha
    def l_3(self, rho, gamma, tau):
        """
        l_3 Level 3 (MGGA-level) Descriptor -- Reduced kinetic energy density

        Equation 6 from the `base paper`_.

        .. _base paper: https://link.aps.org/doi/10.1103/PhysRevB.104.L161109

        .. math:: \\tau^W = \\frac{|\\nabla \\rho|^2}{8\\rho}, \\tau^{unif} = \\frac{3}{10} (3\\pi^2)^{2/3}\\rho^{5/3}.


        :param rho: density
        :type rho: jax.Array
        :param gamma: squared density gradient
        :type gamma: jax.Array
        :param tau: kinetic energy density
        :type tau: jax.Array
        :return: reduced kinetic energy density
        :rtype: jax.Array
        """
        uniform_factor = (3/10)*(3*np.pi**2)**(2/3)
        tw = gamma/(8*rho+self.epsilon)
        l3 = (tau - tw)/(uniform_factor*rho**(5/3)+self.epsilon)
        self.vprint('l_3, descr shape: {}'.format(l3.shape))
        descrnans = jnp.sum(jnp.isnan(l3))
        self.vprint(f'NaNs in descr from self.l_3 = {descrnans}')
        return l3

    # Unit-less electrostatic potential
    def l_4(self, rho, nl):
        """
        l_4 Level 4 (Non-local level) Descriptor -- Unitless electrostatic potential

        .. todo:: document/implement in a more descriptive manner

        :param rho: density
        :type rho: jax.Array
        :param nl: non-local values arising from density contractions
        :type nl: jax.Array
        :return: the non-local descriptors
        :rtype: jax.Array
        """
        u = nl[:,:1]/((jnp.expand_dims(rho, -1)**(1/3))*self.nl_ueg[:,:1] + self.epsilon)
        wu = nl[:,1:]/((jnp.expand_dims(rho, -1))*self.nl_ueg[:,1:] + self.epsilon)
        return jax.nn.relu(jnp.concatenate([u,wu],axis=-1))

    def nl_4(self, mf, dm, ao=None, gw=None, coor=None):
        """
        Level 4 (non-local level) descriptor generator -- generates the CIDER nonlocal descriptors, given a density matrix and a converged kernel with associated mol and grids

        Optional parameters not provided will use the values already associated with `mf.grids`; provide custom values for things like grid subsets.
        
        :param mf: The converged calculation kernel, passed to RKSAnalyzer for descriptor generation
        :type mf: pyscf(ad).dft.RKS kernel
        :param dm: Density matrix to use in nonlocal desciptor generation
        :type dm: jax.Array
        :param ao: Atomic orbital values to use in nonlocal desciptor generation, defaults to None
        :type ao: jax.Array
        :param gw: Grid weights to use in nonlocal desciptor generation, defaults to None
        :type gw: jax.Array
        :param coor: Grid coordinates to use in nonlocal desciptor generation, defaults to None
        :type coor: jax.Array
        :return: The non-local CIDER descriptors, from self.nlstart_i to self.nlend_i
        :rtype: jax.Array
        """
        an = RKSAnalyzer(mf, idm=True, dm=dm,
                         coor=coor, weight=gw)
        
        if len(dm.shape) == 2:
            restric = True
        elif len(dm.shape) == 3:
            restric = False
        descr5_func = lambda x: jax.lax.stop_gradient(jnp.asarray(get_exchange_descriptors2(an, restricted=restric, version='c', auxbasis=mf.mol.basis,
                                   rdm1=True, dm=x, inmol=True, mol=mf.mol, ingrid=True, grid=mf.grids, weights=gw, coords=coor)
        ))
        descr5 = descr5_func(dm)
        self.vprint('nl_4, descr5 shape: {}'.format(descr5.shape))
        descrnans = jnp.sum(jnp.isnan(descr5))
        self.vprint(f'NaNs in descr from self.l_1 = {descrnans}')

        return descr5
        
    # @eqx.filter_jit
    def get_descriptors(self, rho0_a, rho0_b, gamma_a, gamma_b, gamma_ab, tau_a, tau_b, spin_scaling = False,
                       mf = None, dm = None, ao=None, gw=None, coor=None):
        """
        get_descriptors Creates 'ML-compatible' descriptors from the electron density and its gradients, a & b correspond to spin channels

        :param rho0_a: :math:`\\rho` in spin-channel a
        :type rho0_a: jax.Array
        :param rho0_b: :math:`\\rho` in spin-channel b
        :type rho0_b: jax.Array
        :param gamma_a: :math:`|\\nabla \\rho|^2` in spin-channel b
        :type gamma_a: jax.Array
        :param gamma_b: :math:`|\\nabla \\rho|^2` in spin-channel b
        :type gamma_b: jax.Array
        :param gamma_ab: :math:`|\\nabla \\rho|^2`, contracted from both spin channels
        :type gamma_ab: jax.Array
        :param tau_a: KE density in spin-channel a
        :type tau_a: jax.Array
        :param tau_b: KE density in spin-channel b
        :type tau_b: jax.Array
        :param spin_scaling: Flag for spin-scaling, defaults to False
        :type spin_scaling: bool, optional
        :param mf: The converged calculation kernel, passed to RKSAnalyzer for descriptor generation if self.level > 3, defaults to None
        :type mf: pyscf(ad).dft.RKS kernel
        :param dm: Density matrix to use in nonlocal desciptor generation if self.level > 3, defaults to None
        :type dm: jax.Array
        :param ao: Atomic orbital values to use in nonlocal desciptor generation if self.level > 3, defaults to None
        :type ao: jax.Array
        :param gw: Grid weights to use in nonlocal desciptor generation if self.level > 3, defaults to None
        :type gw: jax.Array
        :param coor: Grid coordinates to use in nonlocal desciptor generation if self.level > 3, defaults to None
        :type coor: jax.Array

        :return: Array of the machine-learning descriptors on the grid
        :rtype: jax.Array
        """
        if not spin_scaling:
            #If no spin-scaling, calculate polarization and use for X1
            zeta = (rho0_a - rho0_b)/(rho0_a + rho0_b + self.epsilon)
            spinscale = 0.5*((1+zeta)**(4/3) + (1-zeta)**(4/3)) # zeta

        if self.level > 0:  #  LDA
            if spin_scaling:
                descr1 = jnp.log(self.l_1(2*rho0_a) + self.loge)
                descr2 = jnp.log(self.l_1(2*rho0_b) + self.loge)
            else:
                descr1 = jnp.log(self.l_1(rho0_a + rho0_b) + self.loge)# rho
                descr2 = jnp.log(spinscale) # zeta
            self.vprint(f'self.level > 0; descr1 Nans = {jnp.sum(jnp.isnan(descr1))}')
            self.vprint(f'self.level > 0; descr2 Nans = {jnp.sum(jnp.isnan(descr2))}')
            descr = jnp.concatenate([jnp.expand_dims(descr1, -1), jnp.expand_dims(descr2, -1)],axis=-1)
            self.vprint(f'get_descriptors -> self.level > 0\ndescr1.shape={descr1.shape}, descr2.shape={descr2.shape}, descr.shape={descr.shape}')
        if self.level > 1: # GGA
            if spin_scaling:
                descr3a = self.l_2(2*rho0_a, 4*gamma_a) # s
                descr3b = self.l_2(2*rho0_b, 4*gamma_b) # s
                descr3 = jnp.concatenate([jnp.expand_dims(descr3a,-1), jnp.expand_dims(descr3b,-1)],axis=-1)
                descr3 = (1-jnp.exp(-descr3**2/self.s_gam))*jnp.log(descr3 + 1)
                self.vprint(f'self.level > 1; descr3a Nans = {jnp.sum(jnp.isnan(descr3a))}')
                self.vprint(f'self.level > 1; descr3b Nans = {jnp.sum(jnp.isnan(descr3b))}')
                self.vprint(f'get_descriptors -> self.level > 1 and spin_scaling\ndescr3a.shape={descr3a.shape}, descr3b.shape={descr3b.shape}')
            else:
                descr3 = self.l_2(rho0_a + rho0_b, gamma_a + gamma_b + 2*gamma_ab) # s
                descr3 = descr3/((1+zeta)**(2/3) + (1-zeta)**2/3)
                descr3 = jnp.expand_dims(descr3,-1)
                descr3 = (1-jnp.exp(-descr3**2/self.s_gam))*jnp.log(descr3 + 1)
            self.vprint(f'self.level > 1; descr3 Nans = {jnp.sum(jnp.isnan(descr3))}')
            descr = jnp.concatenate([descr, descr3],axis=-1)
            self.vprint(f'get_descriptors -> self.level > 1\ndescr3.shape={descr3.shape}, descr.shape={descr.shape}')
        if self.level > 2: # meta-GGA
            if spin_scaling:
                descr4a = self.l_3(2*rho0_a, 4*gamma_a, 2*tau_a)
                descr4b = self.l_3(2*rho0_b, 4*gamma_b, 2*tau_b)
                descr4 = jnp.concatenate([jnp.expand_dims(descr4a,-1), jnp.expand_dims(descr4b,-1)],axis=-1)
                descr4 = descr4**3/(descr4**2+self.epsilon)
                self.vprint(f'self.level > 2; descr4a Nans = {jnp.sum(jnp.isnan(descr4a))}')
                self.vprint(f'self.level > 2; descr4b Nans = {jnp.sum(jnp.isnan(descr4b))}')
                self.vprint(f'get_descriptors -> self.level > 2 and spin_scaling\ndescr3a.shape={descr4a.shape}, descr3b.shape={descr4b.shape}')
            else:
                descr4 = self.l_3(rho0_a + rho0_b, gamma_a + gamma_b + 2*gamma_ab, tau_a + tau_b)
                descr4 = 2*descr4/((1+zeta)**(5/3) + (1-zeta)**(5/3))
                descr4 = descr4**3/(descr4**2+self.epsilon)
                descr4 = jnp.expand_dims(descr4,-1)
            self.vprint(f'self.level > 2; pre-log descr4 Nans = {jnp.sum(jnp.isnan(descr4))}')
            self.vprint(f'descr4.min/max: {jnp.min(descr4)}, {jnp.max(descr4)}')
            descr4 = jnp.log((descr4 + 1)/2)
            self.vprint(f'self.level > 2; descr4 Nans = {jnp.sum(jnp.isnan(descr4))}')
            descr = jnp.concatenate([descr, descr4],axis=-1)
            self.vprint(f'get_descriptors -> self.level > 2\ndescr4.shape={descr4.shape}, descr.shape={descr.shape}')
        if self.level > 3: # meta-GGA + V_estat
            # if spin_scaling:
            #     descr5a = self.l_4(2*rho0_a, 2*nl_a)
            #     descr5b = self.l_4(2*rho0_b, 2*nl_b)
            #     descr5 = jnp.log(jnp.stack([descr5a, descr5b],axis=-1) + self.loge)
            #     descr5 = descr5.view(descr5.size()[0],-1)
            # else:
            #     descr5= jnp.log(self.l_4(rho0_a + rho0_b, nl_a + nl_b) + self.loge)

            # descr = jnp.concatenate([descr, descr5],axis=-1)
            def convert_dm(dm):
                res_shape = jax.ShapeDtypeStruct(dm.shape, dm.dtype)
                return jax.lax.stop_gradient(jax.pure_callback(np.asarray, res_shape, dm))
            dmnp = np.asarray(jax.lax.stop_gradient(dm))

            descr5 = self.nl_4(mf, dmnp, ao=ao, gw=gw, coor=coor).T
            self.vprint(f'get_descriptors -> self.level > 3 -> returned descr5 shape={descr5.shape}')
            if len(descr5.shape) == 3:
                #descr5 returned (spin, 12 descriptors, Ngrid)
                #transpose makes (Ngrid, 12 descriptors, spin)
                #want (spin, Ngrid, 12 descriptors)
                descr5 = jnp.transpose(descr5, (2, 0, 1))
                self.vprint(f'reshaped descr5.shape={descr5.shape}')

            if spin_scaling:
                self.vprint(f'spin_scaling and self.level > 3, descr5.shape={descr5.shape}')
                if len(descr5.shape) == 2:
                    self.vprint('decomposing descriptors into spin channels, half each')
                    descr5 = jnp.reshape(jnp.concatenate([0.5*descr5, 0.5*descr5]), (2, descr5.shape[0], -1))
                    self.vprint(f'new descr5.shape={descr5.shape}')
            else:
                if len(descr5.shape) == 3:
                    self.vprint(f'not spin_scaling and self.level > 3 but spin polarized NL descriptors, averaging descr5 spin channels')
                    descr5 = 0.5*(descr5[0] + descr5[1])
        if spin_scaling:
            descr = jnp.transpose(jnp.reshape(descr,(jnp.shape(descr)[0],-1,2)), (2,0,1)) 
            if self.level > 3:
                descr = jnp.concatenate([descr, descr5], axis=-1)
                self.vprint(f'get_descriptors -> self.level > 3, descr5.shape={descr5.shape}')
            self.vprint(f'spin_scaling, get_descriptors -> reshaping -> descr.shape={descr.shape}')
        else:
            if self.level > 3:
                descr = jnp.concatenate([descr, descr5], axis=-1)
                self.vprint(f'get_descriptors not_spin_scaling -> self.level > 3 -> descr5.shape={descr5.shape}')
            self.vprint(f'get_descriptors, not spin_scaling -> descr.shape={descr.shape}')
        return descr

    def eval_grid_models(self, rho, mf=None, dm=None, ao=None,
                          gw=None, coor=None):
        
        """
        eval_grid_models Evaluates all models stored in self.grid_models along with HEG exchange and correlation

        :param rho: List/array with [rho0_a,rho0_b,gamma_a,gamma_ab,gamma_b, dummy for laplacian, dummy for laplacian, tau_a, tau_b, non_loc_a, non_loc_b]
                    Shape assumes, for instance, that rho0_a = rho[:, 0], etc.
        :type rho: jax.Array
        :param mf: The converged calculation kernel, passed to RKSAnalyzer for nonlocal descriptor generation
        :type mf: pyscf(ad).dft.RKS kernel
        :param dm: Density matrix to use in nonlocal desciptor generation
        :type dm: jax.Array
        :param ao: Atomic orbitals to use in non-local descriptor generation, defaults to None
        :type ao: jax.Array
        :param gw: Grid weights to use in non-local descr generation, defaults to None
        :type gw: jax.Array
        :param coor: Grid coordinates to use in non-local descr generation, defaults to None
        :type coor: jax.Array
        :return: The exchange-correlation energy density (on the grid)
        :rtype: jax.Array
        """
        Exc = 0
        rho0_a = rho[:, 0]
        rho0_b = rho[:, 1]
        gamma_a = rho[:, 2]
        gamma_ab = rho[:, 3]
        gamma_b = rho[:, 4]
        tau_a = rho[:, 7]
        tau_b = rho[:, 8]
        nl = rho[:,9:]
        nl_size = jnp.size(nl, -1)//2
        nl_a = nl[:,:nl_size]
        nl_b = nl[:,nl_size:]

        C_F= 3/10*(3*np.pi**2)**(2/3)
        rho0_a_ueg = rho0_a
        rho0_b_ueg = rho0_b

        if gw is not None:
            try:
                self.vprint(f'custom gw and coor present in eval_grid_models; shapes: gw={gw.shape}, coor={coor.shape}')
            except:
                self.vprint(f'custom gw and coor present in eval_grid_models but shape print error; shapes: gw={gw}, coor={coor}')
        zeta = (rho0_a_ueg - rho0_b_ueg)/(rho0_a_ueg + rho0_b_ueg + 1e-8)
        rs = (4*np.pi/3*(rho0_a_ueg+rho0_b_ueg + 1e-8))**(-1/3)
        rs_a = (4*np.pi/3*(rho0_a_ueg + 1e-8))**(-1/3)
        rs_b = (4*np.pi/3*(rho0_b_ueg + 1e-8))**(-1/3)

        #initialize zero values for the ex/ec/grid values
        exc_a = jnp.zeros_like(rho0_a)
        exc_b = jnp.zeros_like(rho0_a)
        exc_ab = jnp.zeros_like(rho0_a)

        self.vprint('eval_grid_models initial nan summary:')
        self.vprint('zeta, rs, rs_a, rs_b, exc_a, exc_b, exc_ab')
        self.vprint('{}, {}, {}, {}, {}, {}, {}'.format(
            jnp.sum((jnp.isnan(zeta))),
            jnp.sum((jnp.isnan(rs))),
            jnp.sum((jnp.isnan(rs_a))),
            jnp.sum((jnp.isnan(rs_b))),
            jnp.sum((jnp.isnan(exc_a))),
            jnp.sum((jnp.isnan(exc_b))),
            jnp.sum((jnp.isnan(exc_ab))),                
            ))

        descr_dict = {}
        rho_tot = rho0_a + rho0_b

        #spin scaling false descriptors; no NL inputs, as not used in get_descriptors
        descr_dict[False] = self.get_descriptors(rho0_a, rho0_b, gamma_a, gamma_b,
                                                         gamma_ab, tau_a, tau_b, spin_scaling = False,
                                                mf = mf, dm = dm, ao=ao, gw=gw, coor=coor)
        #spin scaling true descriptors; no NL inputs, as not used in get_descriptors
        descr_dict[True] = self.get_descriptors(rho0_a, rho0_b, gamma_a, gamma_b,
                                                         gamma_ab, tau_a, tau_b, spin_scaling = True,
                                                mf = mf, dm = dm, ao=ao, gw=gw, coor=coor)
        self.vprint(f'NaNs in descr_dict[False] = {jnp.sum(jnp.isnan(descr_dict[False]))}')
        self.vprint(f'NaNs in descr_dict[True] = {jnp.sum(jnp.isnan(descr_dict[True]))}')

        def else_test_fun(exc, exc_b):
            if self.heg_mult:
                exc_b += (1 + exc[1])*self.heg_model(2*rho0_b_ueg)*(1-self.exx_a)
            else:
                exc_b += exc[1]*(1-self.exx_a)
            return exc_b
            
        def if_not_test_fun(exc, exc_b):
            exc_b += exc[0]*0
            return exc_b

        def gm_eval_func(grid_model, exc_a, exc_b, exc_ab):
            if not grid_model.spin_scaling:
                descr = descr_dict[False]
                # print(f"spin_scaling = False; input descr to exc shape: {descr.shape}")
                exc = grid_model(descr)
                excnans = jnp.sum(jnp.isnan(exc))
                self.vprint(f'exc.shape, descr with spin_scaling=False -> {exc.shape}')
                self.vprint(f'NaNs in exc from gm_eval_func, spin_scaling=False -> = {excnans}')

                if jnp.ndim(exc) == 2: #If using spin decomposition
                    pw_alpha = self.pw_model(rs_a, jnp.ones_like(rs_a))
                    pw_beta = self.pw_model(rs_b, jnp.ones_like(rs_b))
                    pw = self.pw_model(rs, zeta)
                    ec_alpha = (1 + exc[:,0])*pw_alpha*rho0_a/(rho_tot+1e-8)
                    ec_beta =  (1 + exc[:,1])*pw_beta*rho0_b/(rho_tot+1e-8)
                    ec_mixed = (1 + exc[:,2])*(pw*rho_tot - pw_alpha*rho0_a - pw_beta*rho0_b)/(rho_tot+1e-8)
                    exc_ab += ec_alpha + ec_beta + ec_mixed
                else:
                    if self.pw_mult:
                        exc_ab += (1 + exc)*self.pw_model(rs, zeta)
                    else:
                        exc_ab += exc
            else:
                descr = descr_dict[True]
                # print(f"spin_scaling = True; input descr to exc shape: {descr.shape}")
                exc = grid_model(descr)
                excnans = jnp.sum(jnp.isnan(exc))
                self.vprint(f'exc.shape, descr with spin_scaling=True -> {exc.shape}')
                self.vprint(f'NaNs in exc from gm_eval_func, spin_scaling=True -> = {excnans}')


                if self.heg_mult:
                    exc_a += (1 + exc[0])*self.heg_model(2*rho0_a_ueg)*(1-self.exx_a)
                else:
                    exc_a += exc[0]*(1-self.exx_a)
                test = jnp.sum(jnp.abs(rho0_b))
                exc_b = jax.lax.cond(test, else_test_fun, if_not_test_fun, exc, exc_b)

            return (exc_a, exc_b, exc_ab)

        if self.grid_models:
            self.vprint('Grid models present; looping over separate networks to construct exc')
            exc_a, exc_b, exc_ab = gm_eval_func(self.grid_models[0], exc_a, exc_b, exc_ab)
            self.vprint('eval_grid_models gm_eval_func [0] nan summary:')
            self.vprint('exc_a, exc_b, exc_ab')
            self.vprint('{}, {}, {}'.format(
                jnp.sum((jnp.isnan(exc_a))),
                jnp.sum((jnp.isnan(exc_b))),
                jnp.sum((jnp.isnan(exc_ab))),                
                ))
            exc_a, exc_b, exc_ab = gm_eval_func(self.grid_models[1], exc_a, exc_b, exc_ab) 
            self.vprint('eval_grid_models gm_eval_func [1] nan summary:')
            self.vprint('exc_a, exc_b, exc_ab')
            self.vprint('{}, {}, {}'.format(
                jnp.sum((jnp.isnan(exc_a))),
                jnp.sum((jnp.isnan(exc_b))),
                jnp.sum((jnp.isnan(exc_ab))),                
                ))
                   
        else:
            if self.heg_mult:
                exc_a = self.heg_model(2*rho0_a_ueg)
                exc_b = self.heg_model(2*rho0_b_ueg)
            if self.pw_mult:
                exc_ab = self.pw_model(rs, zeta)


        exc = exc_a * (rho0_a_ueg/ (rho_tot + self.epsilon)) + exc_b*(rho0_b_ueg / (rho_tot + self.epsilon)) + exc_ab
        self.vprint('eval_grid_models final nan summary:')
        self.vprint('zeta, rs, rs_a, rs_b, exc_a, exc_b, exc_ab')
        self.vprint('{}, {}, {}, {}, {}, {}, {}'.format(
            jnp.sum((jnp.isnan(zeta))),
            jnp.sum((jnp.isnan(rs))),
            jnp.sum((jnp.isnan(rs_a))),
            jnp.sum((jnp.isnan(rs_b))),
            jnp.sum((jnp.isnan(exc_a))),
            jnp.sum((jnp.isnan(exc_b))),
            jnp.sum((jnp.isnan(exc_ab))),                
            ))
        self.vprint(f'returning exc shape, pre-expanded dims: {exc.shape}')
        return jnp.expand_dims(exc, -1)