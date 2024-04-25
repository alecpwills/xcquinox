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

    '''
    def eval_xc(xc_code, rho, ao, gw, coords, spin=0, relativity=0, deriv=1, omega=None, verbose=None):
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