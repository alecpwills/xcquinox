import jax
import equinox as eqx
import numpy as np
import jax.numpy as jnp
from xcquinox.utils import lda_x, pw92c_unpolarized
from functools import partial


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
            # not spin-polarized
            rho0 = rho[0]  # density
            drho = rho[1:4]  # grad_x, grad_y, grad_z
            # laplacian next
            # tau = 0.5*(rho[1] + rho[2] + rho[3])
            tau = rho[-1]  # tau

            non_loc = jnp.zeros_like(tau)
            # decompose into spin channels
            rho0_a = rho0_b = rho0*0.5
            gamma_a = gamma_b = gamma_ab = jnp.einsum('ij,ij->j', drho[:], drho[:])*0.25
            tau_a = tau_b = tau*0.5
            non_loc_a = non_loc_b = non_loc*0.5
            if network.verbose:
                print(
                    f'decomposed shapes:\nrho0={rho0.shape}\ndrho={drho.shape}\ntau={tau.shape}\nnon_loc={non_loc.shape}')
                print(
                    f'decomposed shapes:\ngamma_a={gamma_a.shape}\ngamma_b={gamma_b.shape}\ngamma_ab={gamma_ab.shape}')
        else:
            # spin-polarized density
            rho0_a = rho[0, 0]
            rho0_b = rho[1, 0]

            drho_a = rho[0, 1:4]
            drho_b = rho[1, 1:4]
            # jnp.einsumed density gradient
            gamma_a, gamma_b = jnp.einsum('ij,ij->j', drho_a, drho_a), jnp.einsum('ij,ij->j', drho_b, drho_b)
            gamma_ab = jnp.einsum('ij,ij->j', drho_a, drho_b)
            # Kinetic energy density
            tau_a = rho[0, -1]
            tau_b = rho[1, -1]

            non_loc_a, non_loc_b = jnp.zeros_like(tau_a), jnp.zeros_like(tau_b)
            if network.verbose:
                print(
                    f'decomposed shapes:\nrho0(a,b)={rho0_a.shape},{rho0_b.shape}\ndrho(a,b)={drho_a.shape},{drho_b.shape}\ntau(a,b)={tau_a.shape},{tau_b.shape}\nnon_loc(a,b)={non_loc_a.shape},{non_loc_b.shape}')
                print(
                    f'decomposed shapes:\ngamma_a={gamma_a.shape}\ngamma_b={gamma_b.shape}\ngamma_ab={gamma_ab.shape}')

        # xc-energy per unit particle
        # print(f'EVALUATING GRID MODELS; OPTIONAL PARAMETERS:')
        # try:
        #     print(f'gw.shape={gw.shape}, coor.shape={coor.shape}')
        # except:
        #     print('no externally supplied gw or coor')
        # print('eval_xc eval_grid_models call')

        def EXC_exc_vs(x):
            exc = network.eval_grid_models(x, mf=mf, dm=dm, ao=ao, gw=gw, coor=coords)
            Exc = jnp.sum(((rho0_a + rho0_b)*exc[:, 0])*gw)
            return Exc, exc
        if network.verbose:
            print(f'eval_xc -> Exc_exc and potentials on grid via autodiff')
        v_and_g_inp = jnp.concatenate([jnp.expand_dims(rho0_a, -1),
                                       jnp.expand_dims(rho0_b, -1),
                                       jnp.expand_dims(gamma_a, -1),
                                       jnp.expand_dims(gamma_ab, -1),
                                       jnp.expand_dims(gamma_b, -1),
                                       jnp.expand_dims(jnp.zeros_like(rho0_a), -1),  # Dummy for laplacian
                                       jnp.expand_dims(jnp.zeros_like(rho0_a), -1),  # Dummy for laplacian
                                       jnp.expand_dims(tau_a, -1),
                                       jnp.expand_dims(tau_b, -1),
                                       jnp.expand_dims(non_loc_a, -1),
                                       jnp.expand_dims(non_loc_b, -1)], axis=-1)
        print(f'v_and_g_inp.shape={v_and_g_inp.shape}')
        Exc_exc, vs = jax.value_and_grad(EXC_exc_vs, has_aux=True)(v_and_g_inp)
        print(f'Exc_exc and vs returned: Exc = {Exc_exc[0]}, exc.shape={Exc_exc[1].shape}, vs.shape={vs.shape}')
        Exc, exc = Exc_exc
        print(f'eval_xc Exc = {Exc}')
        if jnp.sum(jnp.isnan(exc[:, 0])):
            print('NaNs detected in exc. Number of NaNs: {}'.format(jnp.sum(jnp.isnan(exc[:, 0]))))
            raise
        else:
            exc = exc[:, 0]

        # print('ao shape: ', ao.shape)
        # print('exc from network evaluation on grid models shape: ', exc.shape)
        # print('vs from network evaluation on grid models shape: ', vs.shape)
        # print('Exc from network evaluation on grid models shape: ', Exc)

        def vgf(x): return network(x, ao, gw, mf=mf, coor=coords)
        mf.converged = True
        mf.network = network
        mf.network_eval = vgf

        # vrho; d Exc/d rho, separate spin channels
        vrho = vs[:, 0]+vs[:, 1]
        # vtau; d Exc/d tau, separate spin channels
        vtau = vs[:, 7]+vs[:, 8]

        vgamma = jnp.zeros_like(vrho)

        vlapl = None

        fxc = None  # second order functional derivative
        kxc = None  # third order functional derivative
        if network.verbose:
            print(f'shapes: vrho={vrho.shape}, vgamma={vgamma.shape}')
        return exc, (vrho, vgamma, vlapl, vtau), fxc, kxc
    return eval_xc


# updated versions of this
# GGA
def custom_pbe_Fx(rho, sigma, XNET=None):
    # this will be a call to the Fx neural network we want
    # print('DEBUG custom_pbe_Fx, rho/sigma shapes: ', rho.shape, sigma.shape)
    # print('DEBUG custom_pbe_Fx: rho: ', rho)
    # print('DEBUG custom_pbe_Fx: sigma: ', sigma)

    Fx = XNET([rho, sigma])
    return Fx


def custom_pbe_Fc(rho, sigma, CNET=None):  # Assumes zeta = 0
    # this will be a call to the Fc neural network we want
    Fc = CNET([rho, sigma])
    return Fc


def custom_pbe_e(rho, sigma, XNET=None, CNET=None):
    Fx = custom_pbe_Fx(rho, sigma, XNET=XNET)
    Fc = custom_pbe_Fc(rho, sigma, CNET=CNET)

    exc = lda_x(rho)*Fx + pw92c_unpolarized(rho)*Fc

    return exc


def custom_pbe_epsilon(rho, sigma, XNET=None, CNET=None):

    return rho*custom_pbe_e(rho, sigma, XNET=XNET, CNET=CNET)


def derivable_custom_pbe_e(rhosigma, XNET=None, CNET=None):
    rho, sigma = rhosigma
    # print('DEBUG derivable_custom_pbe_e: rhosigma len/shapes: ', len(rhosigma), rhosigma)
    # print('DEBUG derivable_custom_pbe_e: rho/sigma shapes: ', rho.shape, sigma.shape)
    # print('DEBUG derivable_custom_pbe_e: rho: ', rho)
    # print('DEBUG derivable_custom_pbe_e: sigma: ', sigma)
    return custom_pbe_e(rho, sigma, XNET=XNET, CNET=CNET)


def derivable_custom_pbe_epsilon(rhosigma, XNET=None, CNET=None):
    rho = rhosigma[0]
    sigma = rhosigma[1]
    result = custom_pbe_epsilon(rho, sigma, XNET=XNET, CNET=CNET)
    return result[0]


def eval_xc_gga_j(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None,
                  XNET=None, CNET=None):
    # we only expect there to be a rho0 array, but I unpack it as (rho, deriv) here to be in line with the
    # pyscf example -- the size of the 'rho' array depends on the xc type (LDA, GGA, etc.)
    # so since LDA calculation, check for size first.
    rho0, dx, dy, dz = rho[:4]
    rho0 = jnp.array(rho0)
    sigma = jnp.array(dx**2+dy**2+dz**2)
    # print('DEBUG eval_xc_gga_j: rho0/sigma shapes: ', rho0.shape, sigma.shape)
    rhosig = (rho0, sigma)
    # calculate the "custom" energy with rho -- THIS IS e
    # cast back to np.array since that's what pyscf works with
    # pass as tuple -- (rho, sigma)
    derivable_net_e = partial(derivable_custom_pbe_e, XNET=XNET, CNET=CNET)
    derivable_net_epsilon = partial(derivable_custom_pbe_epsilon, XNET=XNET, CNET=CNET)
    exc = np.array(jax.vmap(derivable_net_e)(rhosig))

    # first order derivatives w.r.t. rho and sigma
    vrho_f = eqx.filter_grad(derivable_net_epsilon)
    vrhosigma = np.array(jax.vmap(vrho_f)(rhosig))
    # print('vrhosigma shape:', vrhosigma.shape)
    vxc = (vrhosigma[0], vrhosigma[1], None, None)

    # v2_f = eqx.filter_hessian(derivable_custom_pbe_epsilon)
    v2_f = jax.hessian(derivable_net_epsilon)
    # v2_f = jax.hessian(custom_pbe_epsilon, argnums=[0, 1])
    v2 = np.array(jax.vmap(v2_f)(rhosig))
    # print('v2 shape', v2.shape)
    v2rho2 = v2[0][0]
    v2rhosigma = v2[0][1]
    v2sigma2 = v2[1][1]
    v2lapl2 = None
    vtau2 = None
    v2rholapl = None
    v2rhotau = None
    v2lapltau = None
    v2sigmalapl = None
    v2sigmatau = None
    # 2nd order functional derivative
    fxc = (v2rho2, v2rhosigma, v2sigma2, v2lapl2, vtau2, v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)
    # 3rd order
    kxc = None

    return exc, vxc, fxc, kxc


def eval_xc_gga_j2(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None,
                   xcmodel=None):
    # we only expect there to be a rho0 array, but I unpack it as (rho, deriv) here to be in line with the
    # pyscf example -- the size of the 'rho' array depends on the xc type (LDA, GGA, etc.)
    # so since LDA calculation, check for size first.
    try:
        rho0, dx, dy, dz = rho[:4]
        sigma = jnp.array(dx**2+dy**2+dz**2)
    except:
        rho0, drho = rho[:4]
        sigma = jnp.array(drho**2)
    rho0 = jnp.array(rho0)
    # sigma = jnp.array(dx**2+dy**2+dz**2)
    # print('DEBUG eval_xc_gga_j: rho0/sigma shapes: ', rho0.shape, sigma.shape)
    # rhosig = (rho0, sigma)
    rhosig = jnp.stack([rho0, sigma], axis=1)
    print(rhosig.shape)
    # calculate the "custom" energy with rho -- THIS IS e
    # cast back to np.array since that's what pyscf works with
    # pass as tuple -- (rho, sigma)
    exc = jax.vmap(xcmodel)(rhosig)
    exc = jnp.array(exc)/rho0
    # exc = jnp.array(jax.vmap(xcmodel)( rhosig ) )/rho0
    # print('exc shape = {}'.format(exc.shape))
    # first order derivatives w.r.t. rho and sigma
    vrho_f = eqx.filter_grad(xcmodel)
    vrhosigma = jnp.array(jax.vmap(vrho_f)(rhosig))
    # print('vrhosigma shape:', vrhosigma.shape)
    vxc = (vrhosigma[:, 0], vrhosigma[:, 1], None, None)

    # v2_f = eqx.filter_hessian(derivable_custom_pbe_epsilon)
    v2_f = jax.hessian(xcmodel)
    # v2_f = jax.hessian(custom_pbe_epsilon, argnums=[0, 1])
    v2 = jnp.array(jax.vmap(v2_f)(rhosig))
    print('v2 shape', v2.shape)
    v2rho2 = v2[:, 0, 0]
    v2rhosigma = v2[:, 0, 1]
    v2sigma2 = v2[:, 1, 1]
    v2lapl2 = None
    vtau2 = None
    v2rholapl = None
    v2rhotau = None
    v2lapltau = None
    v2sigmalapl = None
    v2sigmatau = None
    # 2nd order functional derivative
    fxc = (v2rho2, v2rhosigma, v2sigma2, v2lapl2, vtau2, v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)
    # 3rd order
    kxc = None

    return exc, vxc, fxc, kxc

def eval_xc_gga_grho(xc_code, rho, spin=0, relativity=0, deriv=1,
                        omega=None, verbose=None,
                        xcmodel=None):
    """ With networks that expect rho and grad rho - the derivatives here must still be wrt sigma"""
    rho0, dx, dy, dz = rho[:4]
    sigma = (dx ** 2 + dy ** 2 + dz ** 2)
    if xcmodel is None:
        raise ValueError("xcmodel must be provided")
    def xcmodel_sigma(rho0, sigma):
        grad_rho = jnp.sqrt(sigma)
        if rho0.ndim == 0:
            rhogradrho = jnp.array([rho0, grad_rho])
        else:
            rhogradrho = jnp.stack([rho0, grad_rho], axis=-1)
        return xcmodel(rhogradrho)


    # Calculate the "custom" energy with rho -- THIS IS e
    exc = jax.vmap(xcmodel_sigma)(rho0, sigma)
    exc = jnp.array(exc) / rho0

    vrho, vsigma = jax.vmap(
        jax.grad(xcmodel_sigma, argnums=(0, 1)))(rho0, sigma)

    vxc = (vrho, vsigma, None, None)
    fxc = None
    kxc = None

    return exc, vxc, fxc, kxc


def eval_xc_gga_pol(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None,
                    xcmodel=None):
    # we only expect there to be a rho0 array, but I unpack it as (rho, deriv) here to be in line with the
    # pyscf example -- the size of the 'rho' array depends on the xc type (LDA, GGA, etc.)
    # so since LDA calculation, check for size first.
    try:
        rhoshape = len(rho.shape)
        pol = 3
    except:
        rhoshape = len(rho)
        pol = 2
    # if len of shape == 3, spin polarized so compress to unpolarized for calculation
    if rhoshape != pol:
        # SPIN-UNPOLARIZED, ALL ARRAYS PASSED AS IS TO LIBXC
        try:
            # print("unpacking rho[:4] into rho0, dx, dy, dz")
            rho0, dx, dy, dz = rho[:4]
            sigma = jnp.array(dx**2+dy**2+dz**2)
        except:
            print("Unpacking failed...")
            rho0, drho = rho[:4]
            sigma = jnp.array(drho**2)
        rho0 = jnp.array(rho0)
        rhosig = jnp.stack([rho0, sigma], axis=1)
        # print('rho/sig/rhosig shapes: ', rho0.shape, sigma.shape, rhosig.shape)
        # calculate the "custom" energy with rho -- THIS IS e
        # cast back to np.array since that's what pyscf works with
        # pass as tuple -- (rho, sigma)
        exc = jax.vmap(xcmodel)(rhosig)
        exc = jnp.array(exc)/rho0
        vrho_f = eqx.filter_grad(xcmodel)
        vrhosigma = jnp.array(jax.vmap(vrho_f)(rhosig))
        # vxc = vrho and vsigma, unpolarized, followed by nothing higher order in GGA
        vxc = (vrhosigma[:, 0], vrhosigma[:, 1], None, None)

        v2_f = jax.hessian(xcmodel)
        v2 = jnp.array(jax.vmap(v2_f)(rhosig))
        # print('v2 shape', v2.shape)
        v2rho2 = v2[:, 0, 0]
        v2rhosigma = v2[:, 0, 1]
        v2sigma2 = v2[:, 1, 1]
        v2lapl2 = None
        vtau2 = None
        v2rholapl = None
        v2rhotau = None
        v2lapltau = None
        v2sigmalapl = None
        v2sigmatau = None
        # 2nd order functional derivative
        fxc = (v2rho2, v2rhosigma, v2sigma2, v2lapl2, vtau2, v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)
        # 3rd order
        kxc = None

    else:
        # SPIN POLARIZED; RESULT ARRAYS MUST BE RETURNED SPIN POLARIZED
        # THIS IS HACKY -- THE NETWORK IS NOT ARCHITECTED TO ACCEPT ALL THE POLARIZED PARAMETERS, SO THE GRADIENTS ARE JUST DUPLICATED IN THE RETURN;
        # GENERATE A FUNCTION THAT COMBINES THEN CALLS
        def make_epsilon_function(model):
            # importantly, do not place the vmap here
            def get_epsilon(arr):
                rhou, rhod, sigma1, sigma2, sigma3 = arr
                rho0 = jnp.array(rhou+rhod)
                # sum the sigma contributions
                sumsigma = sigma1+sigma2+sigma3

                rhosig = jnp.stack([rho0, sumsigma])
                # calculate the "custom" energy with rho -- THIS IS e
                # cast back to np.array since that's what pyscf works with
                # pass as tuple -- (rho, sigma)
                exc = model(rhosig)
                return exc
            return get_epsilon

        # model_epsilon = partial(get_epsilon, model=xcmodel)
        model_epsilon = make_epsilon_function(model=xcmodel)
        rho_u, rho_d = rho
        # print('rho_u, rho_d shapes:', rho_u.shape, rho_d.shape)
        rho0u, dxu, dyu, dzu = rho_u[:4]
        rho0d, dxd, dyd, dzd = rho_d[:4]
        # up-up
        dxu2 = dxu*dxu
        dyu2 = dyu*dyu
        dzu2 = dzu*dzu
        # up-down
        dxud = dxu*dxd
        dyud = dyu*dyd
        dzud = dzu*dzd
        # down-down
        dxd2 = dxd*dxd
        dyd2 = dyd*dyd
        dzd2 = dzd*dzd
        sigma1 = dxu2+dyu2+dzu2
        sigma2 = dxud+dyud+dzud
        sigma3 = dxd2+dyd2+dzd2

        rho0 = jnp.array(rho0u+rho0d)
        # print('rho0 shape', rho0.shape)
        # print('sigma1/2/3 shapes', sigma1.shape, sigma2.shape, sigma3.shape)
        sumsigma = sigma1+sigma2+sigma3
        # print('sumsigma shape', sumsigma.shape)
        # sum the sigma contributions
        rhosig = jnp.stack([rho0, sigma1+sigma2+sigma3], axis=1)
        # calculate the "custom" energy with rho -- THIS IS e
        # cast back to np.array since that's what pyscf works with
        # pass as tuple -- (rho, sigma)
        # epsilon here
        input_arr = jnp.stack([rho0u, rho0d, sigma1, sigma2, sigma3], axis=1)
        exc = jax.vmap(model_epsilon)(input_arr)
        # print('epsilon shape', exc.shape)
        # e here
        exc = jnp.array(exc)/rho0
        # exc = exc[jnp.newaxis, :]
        # print('exc shape', exc.shape)
        v1_f = jax.grad(model_epsilon)
        v1 = jax.vmap(v1_f)(input_arr)
        # vrho = vrho_up, vrho_down
        vrho = jnp.vstack((v1[:, 0], v1[:, 1]))
        # vsigma = vsigma1, vsigma2, vsigma3
        vsigma = jnp.vstack((v1[:, 2], v1[:, 3], v1[:, 4]))
        vxc = (vrho, vsigma)
        # print('vrho shape', vrho.shape)
        # print('vsigma shape', vsigma.shape)
        v2_f = jax.hessian(model_epsilon)
        v2 = jax.vmap(v2_f)(input_arr)
        # print('v2 shape', v2.shape)
        # v2rho2 = (v2rhou2, v2rhoud, v2rhod2)
        v2rho2 = jnp.vstack((v2[:, 0, 0], v2[:, 0, 1], v2[:, 1, 1]))
        # v2rhosigma is six-part = (u,1),(u,2),(u,3),(d,1),(d,2),(d,3)
        v2rhosigma = jnp.vstack((v2[:, 0, 2], v2[:, 0, 3], v2[:, 0, 4], v2[:, 1, 2], v2[:, 1, 3], v2[:, 1, 4]))
        # v2sigma2 is also six-part
        v2sigma2 = jnp.vstack((v2[:, 2, 2], v2[:, 2, 3], v2[:, 2, 4], v2[:, 3, 3], v2[:, 3, 4], v2[:, 4, 4]))
        # print('v2rho2 shape', v2rho2.shape)
        # print('v2rhosigma shape', v2rhosigma.shape)
        # print('v2sigma2 shape', v2sigma2.shape)
        v2lapl2 = None
        vtau2 = None
        v2rholapl = None
        v2rhotau = None
        v2lapltau = None
        v2sigmalapl = None
        v2sigmatau = None
        # 2nd order functional derivative
        fxc = (v2rho2, v2rhosigma, v2sigma2, v2lapl2, vtau2, v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau)
        # 3rd order
        kxc = None
        TRANSPOSE = True
        if TRANSPOSE:
            vxc = [i.T for i in vxc]
            fxc = [i.T for i in fxc if type(i) == type(jnp.array([1]))]

    return exc, vxc, fxc, kxc
