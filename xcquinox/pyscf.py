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
            nan_count = jnp.sum(jnp.isnan(exc[:, 0]))
            print('NaNs detected in exc. Number of NaNs: {}'.format(nan_count))
            raise ValueError(f'NaNs detected in exchange-correlation energy density. Count: {nan_count}')
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
    '''
    Compute the exchange enhancement factor using a neural network.

    This function wraps a neural network call to compute the exchange
    enhancement factor Fx given density and sigma (squared gradient) inputs.

    :param rho: Electron density value(s) on the grid
    :type rho: jax.Array or float
    :param sigma: Squared density gradient (|nabla rho|^2) on the grid
    :type sigma: jax.Array or float
    :param XNET: Neural network model for exchange enhancement factor, defaults to None
    :type XNET: eqx.Module, optional
    :return: Exchange enhancement factor Fx
    :rtype: jax.Array or float
    '''
    # print('DEBUG custom_pbe_Fx, rho/sigma shapes: ', rho.shape, sigma.shape)
    # print('DEBUG custom_pbe_Fx: rho: ', rho)
    # print('DEBUG custom_pbe_Fx: sigma: ', sigma)

    Fx = XNET([rho, sigma])
    return Fx


def custom_pbe_Fc(rho, sigma, CNET=None):
    '''
    Compute the correlation enhancement factor using a neural network.

    This function wraps a neural network call to compute the correlation
    enhancement factor Fc given density and sigma inputs. Assumes unpolarized
    case (zeta = 0).

    :param rho: Electron density value(s) on the grid
    :type rho: jax.Array or float
    :param sigma: Squared density gradient (|nabla rho|^2) on the grid
    :type sigma: jax.Array or float
    :param CNET: Neural network model for correlation enhancement factor, defaults to None
    :type CNET: eqx.Module, optional
    :return: Correlation enhancement factor Fc
    :rtype: jax.Array or float
    '''
    Fc = CNET([rho, sigma])
    return Fc


def custom_pbe_e(rho, sigma, XNET=None, CNET=None):
    '''
    Compute the exchange-correlation energy density using neural network enhancement factors.

    Calculates exc = lda_x(rho) * Fx + pw92c(rho) * Fc, where Fx and Fc are
    obtained from neural networks.

    :param rho: Electron density value(s) on the grid
    :type rho: jax.Array or float
    :param sigma: Squared density gradient (|nabla rho|^2) on the grid
    :type sigma: jax.Array or float
    :param XNET: Neural network for exchange enhancement factor, defaults to None
    :type XNET: eqx.Module, optional
    :param CNET: Neural network for correlation enhancement factor, defaults to None
    :type CNET: eqx.Module, optional
    :return: Exchange-correlation energy density (exc)
    :rtype: jax.Array or float
    '''
    Fx = custom_pbe_Fx(rho, sigma, XNET=XNET)
    Fc = custom_pbe_Fc(rho, sigma, CNET=CNET)

    exc = lda_x(rho)*Fx + pw92c_unpolarized(rho)*Fc

    return exc


def custom_pbe_epsilon(rho, sigma, XNET=None, CNET=None):
    '''
    Compute epsilon (rho * exc) using neural network enhancement factors.

    This is the quantity that libxc expects derivatives of: epsilon = rho * exc.

    :param rho: Electron density value(s) on the grid
    :type rho: jax.Array or float
    :param sigma: Squared density gradient (|nabla rho|^2) on the grid
    :type sigma: jax.Array or float
    :param XNET: Neural network for exchange enhancement factor, defaults to None
    :type XNET: eqx.Module, optional
    :param CNET: Neural network for correlation enhancement factor, defaults to None
    :type CNET: eqx.Module, optional
    :return: Epsilon value (rho * exc)
    :rtype: jax.Array or float
    '''
    return rho*custom_pbe_e(rho, sigma, XNET=XNET, CNET=CNET)


def derivable_custom_pbe_e(rhosigma, XNET=None, CNET=None):
    '''
    Wrapper for custom_pbe_e that accepts a tuple input for JAX differentiation.

    This function unpacks (rho, sigma) from a tuple to enable use with jax.grad
    and similar transformation functions.

    :param rhosigma: Tuple of (rho, sigma) values
    :type rhosigma: tuple
    :param XNET: Neural network for exchange enhancement factor, defaults to None
    :type XNET: eqx.Module, optional
    :param CNET: Neural network for correlation enhancement factor, defaults to None
    :type CNET: eqx.Module, optional
    :return: Exchange-correlation energy density
    :rtype: jax.Array or float
    '''
    rho, sigma = rhosigma
    # print('DEBUG derivable_custom_pbe_e: rhosigma len/shapes: ', len(rhosigma), rhosigma)
    # print('DEBUG derivable_custom_pbe_e: rho/sigma shapes: ', rho.shape, sigma.shape)
    # print('DEBUG derivable_custom_pbe_e: rho: ', rho)
    # print('DEBUG derivable_custom_pbe_e: sigma: ', sigma)
    return custom_pbe_e(rho, sigma, XNET=XNET, CNET=CNET)


def derivable_custom_pbe_epsilon(rhosigma, XNET=None, CNET=None):
    '''
    Wrapper for custom_pbe_epsilon that accepts a tuple input for JAX differentiation.

    This function unpacks (rho, sigma) from a tuple and returns the first element
    of epsilon for use with jax.grad and similar transformation functions.

    :param rhosigma: Tuple of (rho, sigma) values
    :type rhosigma: tuple
    :param XNET: Neural network for exchange enhancement factor, defaults to None
    :type XNET: eqx.Module, optional
    :param CNET: Neural network for correlation enhancement factor, defaults to None
    :type CNET: eqx.Module, optional
    :return: Scalar epsilon value
    :rtype: float
    '''
    rho = rhosigma[0]
    sigma = rhosigma[1]
    result = custom_pbe_epsilon(rho, sigma, XNET=XNET, CNET=CNET)
    return result[0]


def eval_xc_gga_j(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None,
                  XNET=None, CNET=None):
    '''
    Evaluate GGA exchange-correlation functional using neural networks.

    This function serves as a custom eval_xc replacement for PySCF, computing
    the exchange-correlation energy density, first derivatives (vxc), and
    second derivatives (fxc) using neural network enhancement factors.

    :param xc_code: XC functional code string (ignored, networks are used instead)
    :type xc_code: str
    :param rho: Density and gradient arrays with shape (4, N) containing
        [rho, grad_x, grad_y, grad_z] for N grid points
    :type rho: numpy.ndarray or jax.Array
    :param spin: Spin polarization flag (0 for unpolarized), defaults to 0
    :type spin: int, optional
    :param relativity: Relativity flag (unused), defaults to 0
    :type relativity: int, optional
    :param deriv: Derivative order to compute, defaults to 1
    :type deriv: int, optional
    :param omega: Range-separation parameter (unused), defaults to None
    :type omega: float, optional
    :param verbose: Verbosity level (unused), defaults to None
    :type verbose: int, optional
    :param XNET: Neural network for exchange enhancement factor
    :type XNET: eqx.Module
    :param CNET: Neural network for correlation enhancement factor
    :type CNET: eqx.Module
    :return: Tuple of (exc, vxc, fxc, kxc) where:
        - exc: Exchange-correlation energy density
        - vxc: First derivatives (vrho, vsigma, None, None)
        - fxc: Second derivatives tuple
        - kxc: Third derivatives (None)
    :rtype: tuple
    '''
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


def _eval_xc_gga_j2_unpol(xcmodel, rhosig):
    """Helper for unpolarized eval_xc_gga_j2. No JIT - let outer loss handle tracing."""
    # Get the single-point evaluation function from the model
    eval_point = xcmodel._eval_single_point

    # Compute epsilon
    epsilon = jax.vmap(eval_point)(rhosig)

    # Compute gradient per point: d epsilon / d [rho, sigma]
    grad_fn = jax.grad(eval_point)
    v1 = jax.vmap(grad_fn)(rhosig)  # Shape (N, 2)

    # Compute Hessian per point
    hess_fn = jax.hessian(eval_point)
    v2 = jax.vmap(hess_fn)(rhosig)  # Shape (N, 2, 2)

    return epsilon, v1, v2


def _eval_xc_gga_j2_pol(xcmodel, rhosig_pol):
    """Helper for polarized eval_xc_gga_j2. No JIT - let outer loss handle tracing."""
    from xcquinox.utils import lda_x
    from xcquinox.xc import lda_c_pw

    # For polarized, we need a wrapper that handles the 5-input format
    def eval_point_pol(point):
        rho_a, rho_b = point[0], point[1]
        sigma_aa, sigma_ab, sigma_bb = point[2], point[3], point[4]

        rho = rho_a + rho_b
        sigma = sigma_aa + 2*sigma_ab + sigma_bb

        ex_lda = lda_x(rho)
        ec_pw92 = lda_c_pw(rho_a, rho_b)

        net_input = jnp.array([rho, sigma])
        # Force scalar output using sum (works during JAX tracing)
        Fx = jnp.sum(xcmodel.xnet(net_input))
        Fc = jnp.sum(xcmodel.cnet(net_input))

        rho_safe = jnp.maximum(rho, 1e-18)
        epsilon = rho_safe * (ex_lda * Fx + ec_pw92 * Fc)
        return epsilon

    # Compute epsilon
    epsilon = jax.vmap(eval_point_pol)(rhosig_pol)

    # Compute gradient per point
    grad_fn = jax.grad(eval_point_pol)
    v1 = jax.vmap(grad_fn)(rhosig_pol)  # Shape (N, 5)

    # Compute Hessian per point
    hess_fn = jax.hessian(eval_point_pol)
    v2 = jax.vmap(hess_fn)(rhosig_pol)  # Shape (N, 5, 5)

    return epsilon, v1, v2


def eval_xc_gga_j2(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None,
                   xcmodel=None):
    '''
    Evaluate GGA exchange-correlation functional using a combined XC model.

    Handles both spin-polarized and unpolarized cases.
    Uses vmap internally for per-grid-point evaluation.

    :param xc_code: XC functional code (ignored)
    :param rho: Density and gradients from PySCF. For unpolarized: (4, N) array.
                For polarized: tuple of (rho_a, rho_b) each (4, N).
    :param spin: Spin polarization flag
    :param xcmodel: Combined XC model (RXCModel_GGA)
    :return: (exc, vxc, fxc, kxc)
    '''
    # Detect if spin-polarized by checking if rho is a tuple/list
    try:
        # Try unpolarized first
        rho0, dx, dy, dz = rho[:4]
        sigma = jnp.array(dx**2 + dy**2 + dz**2)
        rho0 = jnp.array(rho0)
        is_polarized = False
    except (ValueError, TypeError):
        # Spin-polarized: rho = [rho_a, rho_b]
        rho_a, rho_b = rho
        rho0a, dxa, dya, dza = rho_a[:4]
        rho0b, dxb, dyb, dzb = rho_b[:4]

        rho0 = rho0a + rho0b
        sigma_aa = dxa**2 + dya**2 + dza**2
        sigma_ab = dxa*dxb + dya*dyb + dza*dzb
        sigma_bb = dxb**2 + dyb**2 + dzb**2
        is_polarized = True

    if not is_polarized:
        # ============ UNPOLARIZED CASE ============
        rhosig = jnp.stack([rho0, sigma], axis=1)  # Shape (N, 2)

        epsilon, v1, v2 = _eval_xc_gga_j2_unpol(xcmodel, rhosig)
        exc = epsilon / (rho0 + 1e-18)

        vrho = v1[:, 0]
        vsigma = v1[:, 1]
        vxc = (vrho, vsigma, None, None)

        v2rho2 = v2[:, 0, 0]
        v2rhosigma = v2[:, 0, 1]
        v2sigma2 = v2[:, 1, 1]

        fxc = (v2rho2, v2rhosigma, v2sigma2,
               None, None, None, None, None, None, None)
        kxc = None

    else:
        # ============ POLARIZED CASE ============
        rhosig_pol = jnp.stack([rho0a, rho0b, sigma_aa, sigma_ab, sigma_bb], axis=1)

        epsilon, v1, v2 = _eval_xc_gga_j2_pol(xcmodel, rhosig_pol)
        exc = epsilon / (rho0 + 1e-18)

        # vrho = [vrho_a, vrho_b]
        vrho = jnp.stack([v1[:, 0], v1[:, 1]], axis=1)
        # vsigma = [vsigma_aa, vsigma_ab, vsigma_bb]
        vsigma = jnp.stack([v1[:, 2], v1[:, 3], v1[:, 4]], axis=1)
        vxc = (vrho, vsigma, None, None)

        # v2rho2 = [aa, ab, bb]
        v2rho2 = jnp.stack([v2[:, 0, 0], v2[:, 0, 1], v2[:, 1, 1]], axis=1)

        # v2rhosigma = [a-aa, a-ab, a-bb, b-aa, b-ab, b-bb]
        v2rhosigma = jnp.stack([
            v2[:, 0, 2], v2[:, 0, 3], v2[:, 0, 4],
            v2[:, 1, 2], v2[:, 1, 3], v2[:, 1, 4]
        ], axis=1)

        # v2sigma2 = [aa-aa, aa-ab, aa-bb, ab-ab, ab-bb, bb-bb]
        v2sigma2 = jnp.stack([
            v2[:, 2, 2], v2[:, 2, 3], v2[:, 2, 4],
            v2[:, 3, 3], v2[:, 3, 4], v2[:, 4, 4]
        ], axis=1)

        fxc = (v2rho2, v2rhosigma, v2sigma2,
               None, None, None, None, None, None, None)
        kxc = None

    return exc, vxc, fxc, kxc


def eval_xc_gga_pol(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None,
                    xcmodel=None):
    '''
    Evaluate GGA exchange-correlation functional with spin polarization support.

    This function handles both spin-polarized and spin-unpolarized cases,
    returning appropriately shaped output arrays for PySCF compatibility.

    For spin-polarized calculations, the network receives combined density
    and the gradients are duplicated across spin channels (hacky workaround
    for networks not architected for full polarized parameters).

    :param xc_code: XC functional code string (ignored, xcmodel is used instead)
    :type xc_code: str
    :param rho: Density and gradient arrays. Shape (4, N) for unpolarized,
        shape (2, 4, N) for polarized with [up, down] spin channels
    :type rho: numpy.ndarray or jax.Array
    :param spin: Spin polarization flag (0 for unpolarized), defaults to 0
    :type spin: int, optional
    :param relativity: Relativity flag (unused), defaults to 0
    :type relativity: int, optional
    :param deriv: Derivative order to compute, defaults to 1
    :type deriv: int, optional
    :param omega: Range-separation parameter (unused), defaults to None
    :type omega: float, optional
    :param verbose: Verbosity level (unused), defaults to None
    :type verbose: int, optional
    :param xcmodel: Combined XC model that computes epsilon(rho, sigma)
    :type xcmodel: eqx.Module
    :return: Tuple of (exc, vxc, fxc, kxc) with shapes appropriate
        for spin-polarized or unpolarized calculations
    :rtype: tuple
    '''
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


# =============================================================================
# Extended eval_xc functions with additional feature support
# =============================================================================

def eval_xc_nn_gga(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None,
                   xcmodel=None,
                   dm=None, overlap=None,
                   grid_coords=None, nuclear_coords=None, nuclear_charges=None,
                   use_laplacian=False, use_dm_features=False, use_cusp=False):
    """
    Evaluate GGA exchange-correlation using a neural network model with extended features.

    This is the primary eval_xc function for use with extended GGA neural network
    functionals. It supports both spin-polarized and unpolarized calculations,
    and can optionally include:

    - Reduced Laplacian (q) descriptor
    - Density matrix features (correlation indicators)
    - Cusp proximity features (nuclear position information)

    The function computes exc (XC energy density per particle), vxc (first derivatives),
    and fxc (second derivatives) suitable for use with PySCF's DFT framework.

    :param xc_code: XC functional code string (ignored, xcmodel is used instead)
    :type xc_code: str
    :param rho: Density and gradient arrays. For GGA:
        - Unpolarized: shape (4, N) or (5, N) with laplacian = [rho, dx, dy, dz, (lapl)]
        - Polarized: shape (2, 4, N) or (2, 5, N) = [[rho_a, dx_a, ...], [rho_b, dx_b, ...]]
    :type rho: jnp.ndarray
    :param spin: Spin polarization flag (0 for unpolarized), defaults to 0
    :type spin: int, optional
    :param relativity: Relativity flag (unused), defaults to 0
    :type relativity: int, optional
    :param deriv: Derivative order to compute, defaults to 1
    :type deriv: int, optional
    :param omega: Range-separation parameter (unused), defaults to None
    :type omega: float, optional
    :param verbose: Verbosity level, defaults to None
    :type verbose: int, optional
    :param xcmodel: Combined XC model (e.g., RXCModel_GGA) that computes epsilon(inputs)
    :type xcmodel: eqx.Module
    :param dm: Density matrix for DM feature computation, shape (nao, nao)
    :type dm: jnp.ndarray, optional
    :param overlap: Overlap matrix S for DM feature computation, shape (nao, nao)
    :type overlap: jnp.ndarray, optional
    :param grid_coords: Grid point coordinates for cusp features, shape (N, 3)
    :type grid_coords: jnp.ndarray, optional
    :param nuclear_coords: Nuclear positions for cusp features, shape (M, 3)
    :type nuclear_coords: jnp.ndarray, optional
    :param nuclear_charges: Nuclear charges for cusp features, shape (M,)
    :type nuclear_charges: jnp.ndarray, optional
    :param use_laplacian: Whether model expects Laplacian input, defaults to False
    :type use_laplacian: bool, optional
    :param use_dm_features: Whether model expects DM features, defaults to False
    :type use_dm_features: bool, optional
    :param use_cusp: Whether model expects cusp features, defaults to False
    :type use_cusp: bool, optional
    :return: Tuple of (exc, vxc, fxc, kxc) where:
        - exc: XC energy density per particle, shape (N,)
        - vxc: First derivatives (vrho, vsigma, vlapl, vtau)
        - fxc: Second derivatives tuple
        - kxc: Third derivatives (None)
    :rtype: tuple

    Example usage::

        from functools import partial
        from xcquinox.pyscf import eval_xc_nn_gga

        # Create custom eval_xc with your model
        custom_eval_xc = partial(
            eval_xc_nn_gga,
            xcmodel=my_xcmodel,
            use_laplacian=True,
            use_dm_features=True,
            dm=dm_matrix,
            overlap=S_matrix
        )

        # Use with PySCF
        mf = dft.RKS(mol)
        mf.define_xc_(custom_eval_xc, 'GGA')
        mf.kernel()
    """
    # Import features module for DM and cusp computations
    from xcquinox.features import compute_dm_features_array, compute_cusp_descriptor

    # Determine if spin-polarized based on rho shape
    rho_shape = rho.shape
    is_polarized = len(rho_shape) == 3

    if not is_polarized:
        # =====================================================================
        # UNPOLARIZED CASE
        # =====================================================================
        rho0 = jnp.array(rho[0])
        dx, dy, dz = rho[1], rho[2], rho[3]
        sigma = jnp.array(dx**2 + dy**2 + dz**2)

        # Extract Laplacian if available and requested
        laplacian = None
        if use_laplacian:
            if rho.shape[0] >= 5:
                laplacian = jnp.array(rho[4])
            else:
                # Approximate Laplacian as zero if not provided
                laplacian = jnp.zeros_like(rho0)
                if verbose:
                    print("Warning: use_laplacian=True but Laplacian not in rho array, using zeros")

        # Compute DM features if requested
        dm_features = None
        if use_dm_features:
            if dm is not None and overlap is not None:
                dm_feat_array = compute_dm_features_array(dm, overlap)
                # Broadcast to all grid points (same global features)
                n_grid = rho0.shape[0]
                dm_features = jnp.tile(dm_feat_array, (n_grid, 1))  # (N, 3)
            else:
                if verbose:
                    print("Warning: use_dm_features=True but dm/overlap not provided")
                dm_features = jnp.zeros((rho0.shape[0], 3))

        # Compute cusp features if requested
        cusp_features = None
        if use_cusp:
            if grid_coords is not None and nuclear_coords is not None and nuclear_charges is not None:
                cusp_features = compute_cusp_descriptor(grid_coords, nuclear_coords, nuclear_charges)
            else:
                if verbose:
                    print("Warning: use_cusp=True but coordinates not provided")
                cusp_features = jnp.zeros((rho0.shape[0], 2))

        # Build input array for the model
        # Base: [rho, sigma] per grid point
        input_list = [rho0, sigma]

        if use_laplacian:
            input_list.append(laplacian)
        if use_dm_features:
            for i in range(dm_features.shape[1]):
                input_list.append(dm_features[:, i])
        if use_cusp:
            for i in range(cusp_features.shape[1]):
                input_list.append(cusp_features[:, i])

        # Stack into (N, n_features) array
        inputs = jnp.stack(input_list, axis=1)

        # Compute epsilon = rho * exc using vmap
        epsilon = jax.vmap(xcmodel)(inputs)
        exc = epsilon / (rho0 + 1e-18)

        # Compute first derivatives via autodiff
        grad_fn = eqx.filter_grad(xcmodel)
        grads = jax.vmap(grad_fn)(inputs)

        # Extract vrho and vsigma (derivatives w.r.t. rho and sigma)
        vrho = grads[:, 0]
        vsigma = grads[:, 1]
        vxc = (vrho, vsigma, None, None)

        # Compute second derivatives (Hessian)
        hess_fn = jax.hessian(xcmodel)
        hess = jax.vmap(hess_fn)(inputs)

        v2rho2 = hess[:, 0, 0]
        v2rhosigma = hess[:, 0, 1]
        v2sigma2 = hess[:, 1, 1]

        fxc = (v2rho2, v2rhosigma, v2sigma2, None, None, None, None, None, None, None)
        kxc = None

    else:
        # =====================================================================
        # SPIN-POLARIZED CASE
        # =====================================================================
        rho_up, rho_dn = rho[0], rho[1]
        rho0_up = jnp.array(rho_up[0])
        rho0_dn = jnp.array(rho_dn[0])
        rho0 = rho0_up + rho0_dn

        # Gradients
        dx_up, dy_up, dz_up = rho_up[1], rho_up[2], rho_up[3]
        dx_dn, dy_dn, dz_dn = rho_dn[1], rho_dn[2], rho_dn[3]

        # Sigma components: sigma_uu, sigma_ud, sigma_dd
        sigma_uu = dx_up**2 + dy_up**2 + dz_up**2
        sigma_ud = dx_up*dx_dn + dy_up*dy_dn + dz_up*dz_dn
        sigma_dd = dx_dn**2 + dy_dn**2 + dz_dn**2
        sigma_total = sigma_uu + 2*sigma_ud + sigma_dd

        # For networks not designed for spin, use total density and sigma
        # This is a simplification - proper spin handling requires spin-dependent networks

        # Extract Laplacian if available
        laplacian = None
        if use_laplacian:
            if rho_up.shape[0] >= 5 and rho_dn.shape[0] >= 5:
                laplacian = jnp.array(rho_up[4] + rho_dn[4])
            else:
                laplacian = jnp.zeros_like(rho0)

        # DM features (same for all grid points)
        dm_features = None
        if use_dm_features:
            if dm is not None and overlap is not None:
                dm_feat_array = compute_dm_features_array(dm, overlap)
                n_grid = rho0.shape[0]
                dm_features = jnp.tile(dm_feat_array, (n_grid, 1))
            else:
                dm_features = jnp.zeros((rho0.shape[0], 3))

        # Cusp features
        cusp_features = None
        if use_cusp:
            if grid_coords is not None and nuclear_coords is not None and nuclear_charges is not None:
                cusp_features = compute_cusp_descriptor(grid_coords, nuclear_coords, nuclear_charges)
            else:
                cusp_features = jnp.zeros((rho0.shape[0], 2))

        # Build model that takes spin-summed inputs
        def make_spin_summed_model(model):
            def wrapped(arr):
                # arr contains: [rho_up, rho_dn, sigma_uu, sigma_ud, sigma_dd, ...]
                rho_u, rho_d = arr[0], arr[1]
                sig_uu, sig_ud, sig_dd = arr[2], arr[3], arr[4]

                rho_total = rho_u + rho_d
                sigma_total = sig_uu + 2*sig_ud + sig_dd

                # Build input for base model
                model_input = [rho_total, sigma_total]
                idx = 5

                # Optional features are appended after spin quantities
                n_extra = arr.shape[0] - 5
                if n_extra > 0:
                    for i in range(n_extra):
                        model_input.append(arr[5 + i])

                model_input = jnp.array(model_input)
                return model(model_input)
            return wrapped

        wrapped_model = make_spin_summed_model(xcmodel)

        # Build input array
        input_list = [rho0_up, rho0_dn, sigma_uu, sigma_ud, sigma_dd]

        if use_laplacian:
            input_list.append(laplacian)
        if use_dm_features:
            for i in range(dm_features.shape[1]):
                input_list.append(dm_features[:, i])
        if use_cusp:
            for i in range(cusp_features.shape[1]):
                input_list.append(cusp_features[:, i])

        inputs = jnp.stack(input_list, axis=1)

        # Compute epsilon and exc
        epsilon = jax.vmap(wrapped_model)(inputs)
        exc = epsilon / (rho0 + 1e-18)

        # First derivatives
        grad_fn = jax.grad(wrapped_model)
        grads = jax.vmap(grad_fn)(inputs)

        # vrho = [vrho_up, vrho_dn]
        vrho = jnp.stack([grads[:, 0], grads[:, 1]], axis=0)
        # vsigma = [vsigma_uu, vsigma_ud, vsigma_dd]
        vsigma = jnp.stack([grads[:, 2], grads[:, 3], grads[:, 4]], axis=0)

        vxc = (vrho.T, vsigma.T)

        # Second derivatives
        hess_fn = jax.hessian(wrapped_model)
        hess = jax.vmap(hess_fn)(inputs)

        # v2rho2 = [v2rho_uu, v2rho_ud, v2rho_dd]
        v2rho2 = jnp.stack([hess[:, 0, 0], hess[:, 0, 1], hess[:, 1, 1]], axis=0)

        # v2rhosigma has 6 components
        v2rhosigma = jnp.stack([
            hess[:, 0, 2], hess[:, 0, 3], hess[:, 0, 4],
            hess[:, 1, 2], hess[:, 1, 3], hess[:, 1, 4]
        ], axis=0)

        # v2sigma2 has 6 components
        v2sigma2 = jnp.stack([
            hess[:, 2, 2], hess[:, 2, 3], hess[:, 2, 4],
            hess[:, 3, 3], hess[:, 3, 4], hess[:, 4, 4]
        ], axis=0)

        fxc = (v2rho2.T, v2rhosigma.T, v2sigma2.T, None, None, None, None, None, None, None)
        kxc = None

    return exc, vxc, fxc, kxc


def make_eval_xc_nn_gga(xcmodel,
                        use_laplacian=False,
                        use_dm_features=False,
                        use_cusp=False,
                        dm=None,
                        overlap=None,
                        grid_coords=None,
                        nuclear_coords=None,
                        nuclear_charges=None):
    """
    Factory function to create a configured eval_xc function for PySCF.

    This is a convenience wrapper that returns a partial function with all
    the model and feature settings pre-configured, ready to be passed to
    mf.define_xc_().

    :param xcmodel: Combined XC model (e.g., RXCModel_GGA)
    :type xcmodel: eqx.Module
    :param use_laplacian: Whether model expects Laplacian input
    :type use_laplacian: bool
    :param use_dm_features: Whether model expects DM features
    :type use_dm_features: bool
    :param use_cusp: Whether model expects cusp features
    :type use_cusp: bool
    :param dm: Density matrix for DM features (can be updated later)
    :type dm: jnp.ndarray, optional
    :param overlap: Overlap matrix for DM features
    :type overlap: jnp.ndarray, optional
    :param grid_coords: Grid coordinates for cusp features
    :type grid_coords: jnp.ndarray, optional
    :param nuclear_coords: Nuclear positions for cusp features
    :type nuclear_coords: jnp.ndarray, optional
    :param nuclear_charges: Nuclear charges for cusp features
    :type nuclear_charges: jnp.ndarray, optional
    :return: Configured eval_xc function
    :rtype: callable

    Example::

        eval_xc_custom = make_eval_xc_nn_gga(
            xcmodel=my_model,
            use_laplacian=True,
            use_cusp=True,
            nuclear_coords=mol.atom_coords(),
            nuclear_charges=mol.atom_charges()
        )

        mf = dft.RKS(mol)
        mf.define_xc_(eval_xc_custom, 'GGA')
    """
    return partial(
        eval_xc_nn_gga,
        xcmodel=xcmodel,
        use_laplacian=use_laplacian,
        use_dm_features=use_dm_features,
        use_cusp=use_cusp,
        dm=dm,
        overlap=overlap,
        grid_coords=grid_coords,
        nuclear_coords=nuclear_coords,
        nuclear_charges=nuclear_charges
    )
