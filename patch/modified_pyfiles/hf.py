from functools import (
    partial,
    wraps,
)
import numpy

from pyscf.data import nist
from pyscf.lib import alias, module_method
from pyscf.scf import hf as pyscf_hf
from pyscf.scf.hf import TIGHT_GRAD_CONV_TOL

from pyscfad import config
from pyscfad import numpy as np
from pyscfad import pytree
from pyscfad import util
from pyscfad import ops
from pyscfad import lib
from pyscfad.lib import logger
from pyscfad.df import df_jk
from pyscfad.implicit_diff import make_implicit_diff
from pyscfad.scf import _vhf
from pyscfad.scf import chkfile
from pyscfad.scf.diis import SCF_DIIS
from pyscfad.scipy.linalg import eigh
from pyscfad.tools.linear_solver import gen_gmres

Traced_Attributes = ['mol', '_eri', 'mo_coeff', 'mo_energy']


def _scf_fixed_point(dm, mf, s1e, h1e):
    vhf = mf.get_veff(mf.mol, dm)
    fock = mf.get_fock(h1e, s1e, vhf, dm)
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    del mo_energy, mo_occ
    return dm


def _scf(dm, mf, s1e, h1e, *,
         conv_tol, conv_tol_grad, diis=None,
         dump_chk=False, callback=None, log,
         e_tot, vhf, cput1, **kwargs):
    scf_conv = False
    fock_last = None
    mol = mf.mol

    # ALEC ADDS
    dm_last = dm
    # ALEC ENDS

    for cycle in range(mf.max_cycle):
        # ALEC ADDS
        alpha = kwargs.get('linear_mixing_alpha', None)
        if alpha:
            print("LINEAR MIXING SPECIFIED, ALPHA = ", alpha)
            alpha = alpha**(cycle)+0.3
            beta = 1-alpha
            dm = alpha * dm + beta * dm_last
        # ALEC ENDS
        dm_last = dm
        last_hf_e = e_tot

        dm_last = dm
        last_hf_e = e_tot

        # ALEC ADDS
        if alpha:
            # FOLLOW SAME ORDER AS DPYSCF:
            # - Mix DM = alpha * dm + beta * dm_last
            # - dm_last = dm
            # - Hcore, then veff calculated using dm
            # - exc/vxc calculated, but this is wrapped into veff in pyscf
            # - fock matrix then calculated
            # - then mo_e, mo_coeff
            # - then dm from these
            # - DPYSCF uses dm_last in generating e_tot; doing so here
            # - causes some kind of artificially short convergence after two steps only,
            # - which is wrong.
            print("LINEAR MIXING SPECIFIED -- Using alternate order for matrix calculations")
            vhf = mf.get_veff(mol, dm, vhf)
            fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, diis, fock_last=fock_last)
            mo_energy, mo_coeff = mf.eig(fock, s1e)
            mo_occ = mf.get_occ(mo_energy, mo_coeff)
            dm = mf.make_rdm1(mo_coeff, mo_occ)
            e_tot = mf.energy_tot(dm, h1e, vhf)
        else:
            # ALEC ENDS -- BELOW IS ORIGINAL CODE, OUTSIDE OF IF/ELSE
            fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, diis, fock_last=fock_last)
            mo_energy, mo_coeff = mf.eig(fock, s1e)
            mo_occ = mf.get_occ(mo_energy, mo_coeff)
            dm = mf.make_rdm1(mo_coeff, mo_occ)
            vhf = mf.get_veff(mol, dm, dm_last, vhf)
            e_tot = mf.energy_tot(dm, h1e, vhf)

        fock_last = fock
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / np.sqrt(norm_gorb.size)
        norm_ddm = np.linalg.norm(dm-dm_last)
        log.info('cycle= %d E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                 cycle+1, e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = log.timer(f'cycle = {cycle+1:d}', *cput1)

        if scf_conv:
            break

    return dm, scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


@wraps(pyscf_hf.kernel)
def kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    log = logger.new_logger(mf)
    cput0 = log.get_t0()
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        log.info('Set gradient conv threshold to %g', conv_tol_grad)

    mol = mf.mol
    s1e = mf.get_ovlp(mol)

    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess, s1e=s1e)
    else:
        dm = dm0

    # ALEC ADDS
    dm_last = dm
    # ALEC ENDS

    h1e = mf.get_hcore(mol)
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf)
    log.info('init E= %.15g', e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        # hack for ROHF
        mo_energy = getattr(mo_energy, 'mo_energy', mo_energy)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
        mf_diis.damp = mf.diis_damp

        # We get the used orthonormalized AO basis from any old eigendecomposition.
        # Since the ingredients for the Fock matrix has already been built, we can
        # just go ahead and use it to determine the orthonormal basis vectors.
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    cput1 = log.timer('initialize scf', *cput0)
    # wrapped function for SCF iteration that is implicitly differentiable
    _scf_wrapped = make_implicit_diff(
        _scf,
        config.scf_implicit_diff,
        optimality_cond=_scf_fixed_point,
        solver=gen_gmres(),
        has_aux=True,
    )
    if config.scf_implicit_diff:
        vhf = ops.stop_grad(vhf)
    # NOTE if use implicit differentiation, only dm will have gradient.
    dm, scf_conv, e_tot, mo_energy, mo_coeff, mo_occ = _scf_wrapped(
        dm, mf, s1e, h1e,
        conv_tol=conv_tol, conv_tol_grad=conv_tol_grad,
        diis=mf_diis, dump_chk=dump_chk, callback=callback,
        log=log, e_tot=e_tot, vhf=vhf, cput1=cput1,
        linear_mixing_alpha=kwargs.get('linear_mixing_alpha', 0)
    )

    _extra_cycle = False
    if config.scf_implicit_diff and (not conv_check or not scf_conv):
        log.warn('\tAn extra scf cycle is going to be run\n'
                 '\tin order to restore the mo_energy derivatives\n'
                 '\tmissing in implicit differentiation.')
        _extra_cycle = True

    if (scf_conv and conv_check) or _extra_cycle:
        vhf = mf.get_veff(mf.mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf), e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)
        norm_gorb = np.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb = norm_gorb / np.sqrt(norm_gorb.size)
        norm_ddm = np.linalg.norm(dm - dm_last)

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol or norm_gorb < conv_tol_grad:
            scf_conv = True
        log.info('Extra cycle  E= %.15g  delta_E= %4.3g  |g|= %4.3g  |ddm|= %4.3g',
                 e_tot, e_tot-last_hf_e, norm_gorb, norm_ddm)
        if dump_chk:
            mf.dump_chk(locals())

    # hack for ROHF
    mo_energy = getattr(mo_energy, 'mo_energy', mo_energy)

    log.timer('scf_cycle', *cput0)
    del log
    # A post-processing hook before return
    mf.post_kernel(locals())
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


@partial(ops.jit, static_argnums=(2, 3))
def _dot_eri_dm_s1(eri, dm, with_j, with_k):
    nao = dm.shape[-1]
    eri = eri.reshape((nao,)*4)
    dms = dm.reshape(-1, nao, nao)
    vj = vk = None
    if with_j:
        vj = np.einsum('ijkl,xji->xkl', eri, dms)
        vj = vj.reshape(dm.shape)
    if with_k:
        vk = np.einsum('ijkl,xjk->xil', eri, dms)
        vk = vk.reshape(dm.shape)
    return vj, vk


def dot_eri_dm(eri, dm, hermi=0, with_j=True, with_k=True):
    dm = np.asarray(dm)
    nao = dm.shape[-1]
    if np.iscomplexobj(eri) or eri.size == nao**4:
        vj, vk = _dot_eri_dm_s1(eri, dm, with_j, with_k)
    else:
        if np.iscomplexobj(eri):
            raise NotImplementedError
        vj, vk = _vhf.incore(eri, dm, hermi, with_j, with_k)
    return vj, vk


@wraps(pyscf_hf.energy_elec)
def energy_elec(mf, dm=None, h1e=None, vhf=None):
    if dm is None:
        dm = mf.make_rdm1()
    if h1e is None:
        h1e = mf.get_hcore()
    if vhf is None:
        vhf = mf.get_veff(mf.mol, dm)
    e1 = np.einsum('ij,ji->', h1e, dm).real
    e_coul = np.einsum('ij,ji->', vhf, dm).real * .5
    mf.scf_summary['e1'] = e1
    mf.scf_summary['e2'] = e_coul
    logger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
    return e1+e_coul, e_coul


@wraps(pyscf_hf.make_rdm1)
def make_rdm1(mo_coeff, mo_occ, **kwargs):
    mocc = mo_coeff[:, mo_occ > 0]
    dm = (mocc*mo_occ[mo_occ > 0]) @ mocc.conj().T
    return dm


@wraps(pyscf_hf.level_shift)
def level_shift(s, d, f, factor):
    dm_vir = s - s @ d @ s
    return f + dm_vir * factor


@wraps(pyscf_hf.dip_moment)
def dip_moment(mol, dm, unit='Debye', verbose=logger.NOTE, **kwargs):
    log = logger.new_logger(mol, verbose)

    if 'unit_symbol' in kwargs:
        log.warn('Kwarg "unit_symbol" was deprecated. It was replaced by kwarg '
                 'unit since PySCF-1.5.')
        unit = kwargs['unit_symbol']

    if getattr(dm, 'ndim', None) != 2:
        # UHF denisty matrices
        dm = dm[0] + dm[1]

    with mol.with_common_orig((0, 0, 0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    el_dip = np.einsum('xij,ji->x', np.asarray(ao_dip), dm).real

    charges = np.asarray(mol.atom_charges(), dtype=float)
    coords = np.asarray(mol.atom_coords())
    nucl_dip = np.einsum('i,ix->x', charges, coords)
    mol_dip = nucl_dip - el_dip

    if unit.upper() == 'DEBYE':
        mol_dip *= nist.AU2DEBYE
        log.note('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)
    else:
        log.note('Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mol_dip)
    del log
    return mol_dip


@wraps(pyscf_hf.get_fock)
def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
             fock_last=None):
    if h1e is None:
        h1e = mf.get_hcore()
    if vhf is None:
        vhf = mf.get_veff(mf.mol, dm)

    # hack for DFT
    vhf = getattr(vhf, 'vxc', vhf)

    f = h1e + vhf
    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None:
        s1e = mf.get_ovlp()
    if dm is None:
        dm = mf.make_rdm1()

    if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4 and fock_last is not None:
        f = pyscf_hf.damping(f, fock_last, damp_factor)
    if diis is not None and cycle >= diis_start_cycle:
        f = diis.update(s1e, dm, f, mf, h1e, vhf, f_prev=fock_last)
    if abs(level_shift_factor) > 1e-4:
        f = level_shift(s1e, dm*.5, f, level_shift_factor)
    return f


class SCF(pytree.PytreeNode, pyscf_hf.SCF):
    """Subclass of :class:`pyscf.scf.hf.SCF` with traceable attributes.

    Attributes
    ----------
    mol : :class:`pyscfad.gto.Mole`
        :class:`pyscfad.gto.Mole` instance.
    mo_coeff : array
        MO coefficients.
    mo_energy : array
        MO energies.
    _eri : array
        Two-electron repulsion integrals.
    """
    DIIS = SCF_DIIS
    _dynamic_attr = {'mol', '_eri', 'mo_coeff', 'mo_energy'}

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()

        aosym = 's4' if config.moleintor_opt else 's1'
        if self._eri is None:
            self._eri = mol.intor('int2e', aosym=aosym)
        if omega:
            with mol.with_range_coulomb(omega):
                _eri = mol.intor('int2e', aosym=aosym)
        else:
            _eri = self._eri

        vj, vk = dot_eri_dm(_eri, dm, hermi, with_j, with_k)
        return vj, vk

    def get_init_guess(self, mol=None, key='minao', **kwargs):
        if mol is None:
            mol = self.mol
        dm0 = pyscf_hf.SCF.get_init_guess(self, mol.to_pyscf(), key, **kwargs)
        dm0 = np.asarray(dm0)  # remove tags
        return dm0

    def scf(self, dm0=None, **kwargs):
        self.dump_flags()
        self.build(self.mol)

        if self.max_cycle > 0 or self.mo_coeff is None:
            self.converged, self.e_tot, \
                self.mo_energy, self.mo_coeff, self.mo_occ = \
                kernel(self, self.conv_tol, self.conv_tol_grad,
                       dm0=dm0, callback=self.callback,
                       conv_check=self.conv_check, **kwargs)
        else:
            self.e_tot = kernel(self, self.conv_tol, self.conv_tol_grad,
                                dm0=dm0, callback=self.callback,
                                conv_check=self.conv_check, **kwargs)[1]

        self._finalize()
        return self.e_tot

    kernel = alias(scf, alias_name='kernel')

    def _eigh(self, h, s):
        return eigh(h, s)

    def eig(self, h, s):
        return self._eigh(h, s)

    def energy_grad(self, dm0=None, mode='rev'):
        """Computing energy gradients w.r.t AO parameters.

        In principle, MO response is not needed, and it is sufficient to
        compute the gradient of the eigen decomposition with the converged
        density matrix. But this function is implemented as to trace the SCF iterations
        to show the difference between unrolling for loops and implicit differentiation.

        Parameters
        ----------
        dm0 : array, optional
            Input density matrix.
        mode : string, default='rev'
            Differentiating using the ``forward`` or ``reverse`` mode.

        Returns
        -------
        mol : :class:`pyscfad.gto.Mole`
            :class:`Mole` object that contains the gradients.

        Notes
        -----
        The attributes of the :class:`SCF` instance will not be modified.
        This function only works with the JAX backend.

        .. deprecated:: 0.2.0
            This function will be deprecated in PySCFAD 0.2.0.
        """
        import jax
        if dm0 is None:
            try:
                dm0 = self.make_rdm1()
            except TypeError:
                pass

        def hf_energy(self, dm0=None):
            self.reset()
            e_tot = self.kernel(dm0=dm0)
            return e_tot

        if mode.lower().startswith('rev'):
            jac = jax.grad(hf_energy)(self, dm0=dm0)
        else:
            if config.scf_implicit_diff:
                msg = """Forward mode differentiation is not available
                         when applying the implicit function differentiation."""
                raise KeyError(msg)
            jac = jax.jacfwd(hf_energy)(self, dm0=dm0)
        if hasattr(jac, 'cell'):
            return jac.cell
        else:
            return jac.mol

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        return df_jk.density_fit(self, auxbasis, with_df, only_dfj)

    @wraps(pyscf_hf.SCF.get_veff)
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if self.direct_scf:
            ddm = np.asarray(dm) - dm_last
            vj, vk = self.get_jk(mol, ddm, hermi=hermi)
            return vhf_last + vj - vk * .5
        else:
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            return vj - vk * .5

    @wraps(pyscf_hf.SCF.dip_moment)
    def dip_moment(self, mol=None, dm=None, unit='Debye', verbose=logger.NOTE,
                   **kwargs):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        return dip_moment(mol, dm, unit, verbose=verbose, **kwargs)

    def dump_chk(self, envs):
        if self.chkfile:
            chkfile.dump_scf(self.mol, self.chkfile,
                             envs['e_tot'], envs['mo_energy'],
                             envs['mo_coeff'], envs['mo_occ'],
                             overwrite_mol=False)
        return self

    def energy_nuc(self):
        # recompute nuclear energy to trace it
        return self.mol.energy_nuc()

    def check_sanity(self):
        return pyscf_hf.SCF.check_sanity(ops.stop_grad(self))

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None:
            mo_energy = self.mo_energy
        return pyscf_hf.SCF.get_occ(self, ops.to_numpy(mo_energy))

    make_rdm1 = module_method(make_rdm1, absences=['mo_coeff', 'mo_occ'])
    energy_elec = energy_elec
    get_fock = get_fock
    to_pyscf = util.to_pyscf


class RHF(SCF, pyscf_hf.RHF):
    @wraps(pyscf_hf.RHF.check_sanity)
    def check_sanity(self):
        mol = self.mol
        if mol.nelectron != 1 and mol.spin != 0:
            logger.warn(self, 'Invalid number of electrons %d for RHF method.',
                        mol.nelectron)
        return SCF.check_sanity(self)

    @wraps(pyscf_hf.RHF.get_veff)
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        if self._eri is not None or not self.direct_scf:
            vj, vk = self.get_jk(mol, dm, hermi)
            vhf = vj - vk * .5
        else:
            ddm = np.asarray(dm) - np.asarray(dm_last)
            vj, vk = self.get_jk(mol, ddm, hermi)
            vhf = vj - vk * .5
            vhf += np.asarray(vhf_last)
        return vhf
