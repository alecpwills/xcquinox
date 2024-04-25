import jax
import jax.numpy as jnp

def generate_network_eval_xc(mf, dm, network):
    '''
    Generates a function to overwrite eval_xc with on the mf object, for use in training with pyscfad's SCF cycle

    :param mf: Pyscfad calculation kernel object
    :type mf: Pyscfad calculation kernel object
    :param dm: Initial density matrix to use in the cycle
    :type dm: jax.Array
    :param network: The network to use in evaluating the SCF cycle
    :type network: xcquinox.xc.eXC
    :return: A function `eval_xc` that uses an xcquinox network as the pyscfad kernel calculation driver.
    :rtype: function

    The returned function:

    eval_xc(xc_code, rho, ao, gw, coords, spin=0, relativity=0, deriv=1, omega=None, verbose=None)
    The function to use as driver for a pyscf(ad) calculation, using an xcquinox network.

    This overwrites mf.eval_xc with a custom function, evaluating:

    Exc_exc, vs = jax.value_and_grad(EXC_exc_vs, has_aux=True)(jnp.concatenate([jnp.expand_dims(rho0_a,-1),
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


        :param xc_code: The XC functional code string in libxc format, but it is ignored as the network is the calculation driver
        :type xc_code: str
        :param rho: The [..., *, N] arrays (... for spin polarized), N is the number of grid points.
                    rho (*,N) ordered as (rho, grad_x, grad_y, grad_z, laplacian, tau)
                    rho (2,*,N) is [(rho_up, grad_x_up, grad_y_up, grad_z_up, laplacian_up, tau_up),
                                    (rho_down, grad_x_down, grad_y_down, grad_z_down, laplacian_down, tau_down)]
                    PySCFAD doesn't do spin-polarized grid calculations yet, so this will be unpolarized.
        :type rho: jax.Array
        :param ao: The atomic orbitals on the grid to use in the network calculation. Explcitly specified as the block loops break down the grid if memory is too low
        :type ao: jax.Array
        :param ao: The grid weights to use in the network calculation. Explcitly specified as the block loops break down the grid if memory is too low
        :type ao: jax.Array
        :param ao: The grid coordinates to use in the network calculation. Explcitly specified as the block loops break down the grid if memory is too low
        :type ao: jax.Array
        :param spin: The spin of the calculation, integer valued, polarized if non-zero, defaults to zero
        :type spin: int
        :param relativity: Integer, unused right now, defaults to zero
        :type relativity: int
        :param deriv: Unused here, defaults to 1
        :type deriv: int
        :param omega: Hybrid mixing term, unused here, defaults to None
        :type omega: float
        :param verbose: Unused here, defaults to None
        :type verbose: int
        :return: ex, vxc, fxc, kxc
                 where: ex -> exc, XC energy density on the grid
                        vxc -> (vrho, vsigma, vlapl, vtau), gradients of the exc w.r.t. the quantities given.
                        Only vrho and vtau are used, vsigma=vlapl=fxc=kxc=None.
                        vrho = vs[:, 0]+vs[:, 1]
                        vtau = vs[:, 7]+vs[:, 8]
        
        :rtype: tuple
    '''
    def eval_xc(xc_code, rho, ao, gw, coords, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
        '''
        The function to use as driver for a pyscf(ad) calculation, using an xcquinox network.

        This overwrites mf.eval_xc with a custom function, evaluating:

        Exc_exc, vs = jax.value_and_grad(EXC_exc_vs, has_aux=True)(jnp.concatenate([jnp.expand_dims(rho0_a,-1),
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

        :param xc_code: The XC functional code string in libxc format, but it is ignored as the network is the calculation driver
        :type xc_code: str
        :param rho: The [..., *, N] arrays (... for spin polarized), N is the number of grid points.
                    rho (*,N) ordered as (rho, grad_x, grad_y, grad_z, laplacian, tau)
                    rho (2,*,N) is [(rho_up, grad_x_up, grad_y_up, grad_z_up, laplacian_up, tau_up),
                                    (rho_down, grad_x_down, grad_y_down, grad_z_down, laplacian_down, tau_down)]
                    PySCFAD doesn't do spin-polarized grid calculations yet, so this will be unpolarized.
        :type rho: jax.Array
        :param ao: The atomic orbitals on the grid to use in the network calculation. Explcitly specified as the block loops break down the grid if memory is too low
        :type ao: jax.Array
        :param ao: The grid weights to use in the network calculation. Explcitly specified as the block loops break down the grid if memory is too low
        :type ao: jax.Array
        :param ao: The grid coordinates to use in the network calculation. Explcitly specified as the block loops break down the grid if memory is too low
        :type ao: jax.Array
        :param spin: The spin of the calculation, integer valued, polarized if non-zero, defaults to zero
        :type spin: int
        :param relativity: Integer, unused right now, defaults to zero
        :type relativity: int
        :param deriv: Unused here, defaults to 1
        :type deriv: int
        :param omega: Hybrid mixing term, unused here, defaults to None
        :type omega: float
        :param verbose: Unused here, defaults to None
        :type verbose: int
        :return: ex, vxc, fxc, kxc
                 where: ex -> exc, XC energy density on the grid
                        vxc -> (vrho, vsigma, vlapl, vtau), gradients of the exc w.r.t. the quantities given.
                        Only vrho and vtau are used, vsigma=vlapl=fxc=kxc=None.
                        vrho = vs[:, 0]+vs[:, 1]
                        vtau = vs[:, 7]+vs[:, 8]
        
        :rtype: tuple
        '''
        # print('custom eval_xc; input rho shape: ', rho.shape)
        if len(rho.shape) == 2:
            rho0 = rho[0] #density
            drho = rho[1:4] #grad_x, grad_y, grad_z
            #laplacian next
            # tau = 0.5*(rho[1] + rho[2] + rho[3])
            tau = rho[-1] # tau
            
            non_loc = jnp.zeros_like(tau)
        if network.verbose:
            print(f'decomposed shapes:\nrho0={rho0.shape}\ndrho={drho.shape}\ntau={tau.shape}\nnon_loc={non_loc.shape}')
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
        # print(f'EVALUATING GRID MODELS; OPTIONAL PARAMETERS:')
        # try:
        #     print(f'gw.shape={gw.shape}, coor.shape={coor.shape}')
        # except:
        #     print('no externally supplied gw or coor')
        # print('eval_xc eval_grid_models call')
        
        def EXC_exc_vs(x):
            exc = network.eval_grid_models(x, mf=mf, dm=dm, ao=ao, gw=gw, coor=coords)
            Exc = jnp.sum(((rho0_a + rho0_b)*exc[:,0])*gw)
            return Exc, exc
        if network.verbose:
            print(f'eval_xc -> Exc_exc and potentials on grid via autodiff')
        Exc_exc, vs = jax.value_and_grad(EXC_exc_vs, has_aux=True)(jnp.concatenate([jnp.expand_dims(rho0_a,-1),
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
        Exc, exc = Exc_exc
        
        if jnp.sum(jnp.isnan(exc[:, 0])):
            print('NaNs detected in exc. Number of NaNs: {}'.format(jnp.sum(jnp.isnan(exc[:, 0]))))
            raise
        else:
            exc = exc[:, 0]
            
        # print('ao shape: ', ao.shape)
        # print('exc from network evaluation on grid models shape: ', exc.shape)
        # print('vs from network evaluation on grid models shape: ', vs.shape)
        # print('Exc from network evaluation on grid models shape: ', Exc)

        vgf = lambda x: network(x, ao, gw, mf=mf, coor=coords)
        mf.network = network
        mf.network_eval = vgf

        #vrho; d Exc/d rho, separate spin channels
        vrho = vs[:, 0]+vs[:, 1]
        #vtau; d Exc/d tau, separate spin channels
        vtau = vs[:, 7]+vs[:, 8]
        
        vgamma = jnp.zeros_like(vrho)
        
        vlapl = None
        
        fxc = None #second order functional derivative
        kxc = None #third order functional derivative
        if network.verbose:
            print(f'shapes: vrho={vrho.shape}, vgamma={vgamma.shape}')
        return exc, (vrho, vgamma, vlapl, vtau), fxc, kxc
    return eval_xc