import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp

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

        DOI: `10.1103/PhysRevB.45.13244`_

        .. 10.1103/PhysRevB.45.13244_: https://doi.org/10.1103/PhysRevB.45.13244

        """
        super().__init__()
        
    def __call__(self, rs, zeta):
        """
        __call__ Forward pass, computing the correlation energy per electron for the given input values, rs and zeta, where

        Atomic units are assumed.
        
        .. math:: r_s = \\Big[\\frac{3}{4\\pi (\\rho_\\uparrow+\\rho_\\downarrow)} \\Big]^{1/3}

        .. math:: \\zeta = \\frac{\\rho_\\uparrow-\\rho_\\downarrow}{\\rho_\\uparrow+\\rho_\\downarrow}

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
    

    def __init__(self, grid_models=[], heg_mult=True, pw_mult=True,
                    level = 1, exx_a=None, epsilon=1e-8):
        """Defines the XC functional on a grid

        Args:
            grid_models (list, optional): list of eX (local exchange) or eC (local correlation). Defines the xc-models/enhancement factors. Defaults to None.
            heg_mult (bool, optional): Use homoegeneous electron gas exchange (multiplicative if grid_models is not empty). Defaults to True.
            pw_mult (bool, optional): Use homoegeneous electron gas correlation (Perdew & Wang). Defaults to True.
            level (int, optional): Controls the number of density "descriptors" generated. 1: LDA, 2: GGA, 3:meta-GGA, 4: meta-GGA + electrostatic (nonlocal). Defaults to 1.
            exx_a (_type_, optional): Exact exchange mixing parameter. Defaults to None.
            epsilon (float, optional): Offset to avoid div/0 in calculations. Defaults to 1e-8.
        """

        super().__init__()
        self.heg_mult = heg_mult
        self.pw_mult = pw_mult
        self.level = level
        self.grid_models = grid_models
        self.epsilon = epsilon
        if level > 3:
            print('WARNING: Non-local models highly experimental and likely will not work ')
        self.loge = 1e-5
        self.s_gam = 1

        if heg_mult:
            self.heg_model = LDA_X()
        if pw_mult:
            self.pw_model = PW_C()
        self.model_mult = [1 for m in self.grid_models]
        if not exx_a:
            self.exx_a = 0
        else:
            self.exx_a = exx_a
            
    # Density (rho)
    def l_1(self, rho):
        """Level 1 Descriptor -- Creates dimensionless quantity from rho.
        Eq. 3 in `base paper <https://link.aps.org/doi/10.1103/PhysRevB.104.L161109>`_

        .. math:: x_0 = \\rho^{1/3}

        Args:
            rho (jax.Array): density

        Returns:
            jax.Array: Scaled density
        """
        return rho**(1/3)

    # Reduced density gradient s
    def l_2(self, rho, gamma):
        """Level 2 Descriptor -- Reduced gradient density
        Eq. 5 in `base paper <https://link.aps.org/doi/10.1103/PhysRevB.104.L161109>`_

        .. math:: x_2=s=\\frac{1}{2(3\\pi^2)^{1/3}} \\frac{|\\nabla \\rho|}{\\rho^{4/3}}

        Args:
            rho (jax.Array): density
            gamma (jax.Array): squared density gradient

        Returns:
            jax.Array: reduced density gradient s
        """
        return jnp.sqrt(gamma)/(2*(3*np.pi**2)**(1/3)*rho**(4/3)+self.epsilon)

    # Reduced kinetic energy density alpha
    def l_3(self, rho, gamma, tau):
        """Level 3 Descriptor -- Reduced kinetic energy density
        Eq. 6 in `base paper <https://link.aps.org/doi/10.1103/PhysRevB.104.L161109>`_

        .. math:: x_3 = \\alpha = \\frac{\\tau-\\tau^W}{\\tau^{unif}},

        where

        .. math:: \\tau^W = \\frac{|\\nabla \\rho|^2}{8\\rho}, \\tau^{unif} = \\frac{3}{10} (3\\pi^2)^{2/3}\\rho^{5/3}.

        Args:
            rho (jax.Array): density
            gamma (jax.Array): squared density gradient
            tau (jax.Array): kinetic energy density

        Returns:
            jax.Array: reduced kinetic energy density
        """
        uniform_factor = (3/10)*(3*np.pi**2)**(2/3)
        tw = gamma/(8*(rho+self.epsilon))
        return (tau - gamma/(8*(rho+self.epsilon)))/(uniform_factor*rho**(5/3)+self.epsilon)

    # Unit-less electrostatic potential
    def l_4(self, rho, nl):
        """Level 4 Descriptor -- Unitless electrostatic potential

        .. todo:: implement in a useful manner

        Args:
            rho (jax.Array): density
            nl (jax.Array): some non-local descriptor

        Returns:
            jax.Array: the non-local descriptors
        """
        u = nl[:,:1]/((jnp.expand_dims(rho, -1)**(1/3))*self.nl_ueg[:,:1] + self.epsilon)
        wu = nl[:,1:]/((jnp.expand_dims(rho, -1))*self.nl_ueg[:,1:] + self.epsilon)
        return jax.nn.relu(jnp.concatenate([u,wu],axis=-1))
    # @eqx.filter_jit
    def get_descriptors(self, rho0_a, rho0_b, gamma_a, gamma_b, gamma_ab,nl_a,nl_b, tau_a, tau_b, spin_scaling = False):
        """Creates 'ML-compatible' descriptors from the electron density and its gradients, a & b correspond to spin channels

        Args:
            rho0_a (jax.Array): :math:`\\rho` in spin-channel a
            rho0_b (jax.Array): :math:`\\rho` in spin-channel b
            gamma_a (jax.Array): :math:`|\\nabla \\rho|^2` in spin-channel a 
            gamma_b (jax.Array): :math:`|\\nabla \\rho|^2` in spin-channel b
            gamma_ab (jax.Array): _description_
            nl_a (jax.Array): Non-local descriptors in spin-channel a, not currently used.
            nl_b (jax.Array): Non-local descriptors in spin-channel b, not currently used.
            tau_a (jax.Array): KE density in spin-channel a
            tau_b (jax.Array): KE density in spin-channel b
            spin_scaling (bool, optional): Flag for spin-scaling. Defaults to False.

        .. todo: implement non-local descriptors.

        Returns:
            _type_: _description_
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
            descr = jnp.concatenate([jnp.expand_dims(descr1, -1), jnp.expand_dims(descr2, -1)],axis=-1)
        if self.level > 1: # GGA
            if spin_scaling:
                descr3a = self.l_2(2*rho0_a, 4*gamma_a) # s
                descr3b = self.l_2(2*rho0_b, 4*gamma_b) # s
                descr3 = jnp.concatenate([jnp.expand_dims(descr3a,-1), jnp.expand_dims(descr3b,-1)],axis=-1)
                descr3 = (1-jnp.exp(-descr3**2/self.s_gam))*jnp.log(descr3 + 1)
            else:
                descr3 = self.l_2(rho0_a + rho0_b, gamma_a + gamma_b + 2*gamma_ab) # s
                descr3 = descr3/((1+zeta)**(2/3) + (1-zeta)**2/3)
                descr3 = jnp.expand_dims(descr3,-1)
                descr3 = (1-jnp.exp(-descr3**2/self.s_gam))*jnp.log(descr3 + 1)
            descr = jnp.concatenate([descr, descr3],axis=-1)
        if self.level > 2: # meta-GGA
            if spin_scaling:
                descr4a = self.l_3(2*rho0_a, 4*gamma_a, 2*tau_a)
                descr4b = self.l_3(2*rho0_b, 4*gamma_b, 2*tau_b)
                descr4 = jnp.concatenate([jnp.expand_dims(descr4a,-1), jnp.expand_dims(descr4b,-1)],axis=-1)
                descr4 = descr4**3/(descr4**2+self.epsilon)
            else:
                descr4 = self.l_3(rho0_a + rho0_b, gamma_a + gamma_b + 2*gamma_ab, tau_a + tau_b)
                descr4 = 2*descr4/((1+zeta)**(5/3) + (1-zeta)**(5/3))
                descr4 = descr4**3/(descr4**2+self.epsilon)

                descr4 = jnp.expand_dims(descr4,-1)
            descr4 = jnp.log((descr4 + 1)/2)
            descr = jnp.concatenate([descr, descr4],axis=-1)
        if self.level > 3: # meta-GGA + V_estat
            if spin_scaling:
                descr5a = self.l_4(2*rho0_a, 2*nl_a)
                descr5b = self.l_4(2*rho0_b, 2*nl_b)
                descr5 = jnp.log(jnp.stack([descr5a, descr5b],axis=-1) + self.loge)
                descr5 = descr5.view(descr5.size()[0],-1)
            else:
                descr5= jnp.log(self.l_4(rho0_a + rho0_b, nl_a + nl_b) + self.loge)

            descr = jnp.concatenate([descr, descr5],axis=-1)
        if spin_scaling:
            descr = jnp.transpose(jnp.reshape(descr,(jnp.shape(descr)[0],-1,2)), (2,0,1)) 
        return descr


    def __call__(self, dm, ao_eval, grid_weights):
        """
        __call__ Forward call for the XC network to get the grid point e_xc

        Generates the density-on-grid from the density matrix, atomic orbital evaluation, and the grid weights from a :pyscfad: calculation.


        :param dm: Density matrix
        :type dm: jax.Array
        :param ao_eval: Atomic orbitals evaluated on the grid
        :type ao_eval: jax.Array
        :param grid_weights: Grid weights associated to the grid on which the atomic orbitals are evaluated
        :type grid_weights: jax.Array
        :return: Exc, exchange-correlation energy from integrating the network calls across the grid
        :rtype: float
        """
        Exc = 0
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
                                                    jnp.expand_dims(non_loc_b,-1)],axis=-1))
            Exc += jnp.sum(((rho0_a + rho0_b)*exc[:,0])*grid_weights)
        return Exc

    def eval_grid_models(self, rho, debug=False):
        """
        eval_grid_models Evaluates all models stored in self.grid_models along with HEG exchange and correlation

        _extended_summary_

        :param rho: List/array with [rho0_a,rho0_b,gamma_a,gamma_ab,gamma_b, dummy for laplacian, dummy for laplacian, tau_a, tau_b, non_loc_a, non_loc_b]
                    Shape assumes, for instance, that rho0_a = rho[:, 0], etc.
        :type rho: jax.Array
        :param debug: If flagged, will print various statistics during the call, defaults to False
        :type debug: bool, optional
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

        zeta = (rho0_a_ueg - rho0_b_ueg)/(rho0_a_ueg + rho0_b_ueg + 1e-8)
        rs = (4*np.pi/3*(rho0_a_ueg+rho0_b_ueg + 1e-8))**(-1/3)
        rs_a = (4*np.pi/3*(rho0_a_ueg + 1e-8))**(-1/3)
        rs_b = (4*np.pi/3*(rho0_b_ueg + 1e-8))**(-1/3)

        #initialize zero values for the ex/ec/grid values
        exc_a = jnp.zeros_like(rho0_a)
        exc_b = jnp.zeros_like(rho0_a)
        exc_ab = jnp.zeros_like(rho0_a)

        if debug:
            print('eval_grid_models nan summary:')
            print('zeta, rs, rs_a, rs_b, exc_a, exc_b, exc_ab')
            print('{}, {}, {}, {}, {}, {}, {}'.format(
                jnp.sum(jnp.any(jnp.isnan(zeta))),
                jnp.sum(jnp.any(jnp.isnan(rs))),
                jnp.sum(jnp.any(jnp.isnan(rs_a))),
                jnp.sum(jnp.any(jnp.isnan(rs_b))),
                jnp.sum(jnp.any(jnp.isnan(exc_a))),
                jnp.sum(jnp.any(jnp.isnan(exc_b))),
                jnp.sum(jnp.any(jnp.isnan(exc_ab))),                
            ))

        descr_dict = {}
        rho_tot = rho0_a + rho0_b
        #spin scaling false descriptors
        descr_dict[False] = self.get_descriptors(rho0_a, rho0_b, gamma_a, gamma_b,
                                                         gamma_ab, nl_a, nl_b, tau_a, tau_b, spin_scaling = False)
        #spin scaling true descriptors
        descr_dict[True] = self.get_descriptors(rho0_a, rho0_b, gamma_a, gamma_b,
                                                         gamma_ab, nl_a, nl_b, tau_a, tau_b, spin_scaling = True)

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
                exc = grid_model(descr)
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

                exc = grid_model(descr)


                if self.heg_mult:
                    exc_a += (1 + exc[0])*self.heg_model(2*rho0_a_ueg)*(1-self.exx_a)
                else:
                    exc_a += exc[0]*(1-self.exx_a)
                test = jnp.sum(jnp.abs(rho0_b))
                exc_b = jax.lax.cond(test, else_test_fun, if_not_test_fun, exc, exc_b)

            return (exc_a, exc_b, exc_ab)

        if self.grid_models:
            exc_a, exc_b, exc_ab = gm_eval_func(self.grid_models[0], exc_a, exc_b, exc_ab)
            exc_a, exc_b, exc_ab = gm_eval_func(self.grid_models[1], exc_a, exc_b, exc_ab)                    
        else:
            if self.heg_mult:
                exc_a = self.heg_model(2*rho0_a_ueg)
                exc_b = self.heg_model(2*rho0_b_ueg)
            if self.pw_mult:
                exc_ab = self.pw_model(rs, zeta)


        exc = exc_a * (rho0_a_ueg/ (rho_tot + self.epsilon)) + exc_b*(rho0_b_ueg / (rho_tot + self.epsilon)) + exc_ab
        if debug:
            print('eval_grid_models nan summary:')
            print('zeta, rs, rs_a, rs_b, exc_a, exc_b, exc_ab')
            print('{}, {}, {}, {}, {}, {}, {}'.format(
                jnp.sum(jnp.any(jnp.isnan(zeta))),
                jnp.sum(jnp.any(jnp.isnan(rs))),
                jnp.sum(jnp.any(jnp.isnan(rs_a))),
                jnp.sum(jnp.any(jnp.isnan(rs_b))),
                jnp.sum(jnp.any(jnp.isnan(exc_a))),
                jnp.sum(jnp.any(jnp.isnan(exc_b))),
                jnp.sum(jnp.any(jnp.isnan(exc_ab))),                
            ))

        return jnp.expand_dims(exc, -1)
