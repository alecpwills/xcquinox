"""xcquinox.alec.oep: Wu-Yang OEP inversion for reference V_xc generation.

Offline utility: generates V_xc^ref matrices from high-level density
matrices (e.g., CCSD). Not part of the training loop, produces .npz
files consumed by ``MoleculeSpec.external_data_path``.

Implements the displacement-form Wu-Yang OEP of Wu & Yang
J. Chem. Phys. 118, 2498 (2003) §II.B:

    V_xc(r) = V_xc^baseline(r) + Σ_t  b_t · g_t(r)

where ``V_xc^baseline`` is computed from a user-chosen XC functional
(LDA / PBE / BLYP / SCAN / B3LYP / ...; any pyscf-compatible string) and
``g_t`` are auxiliary basis functions. At ``b = 0`` the KS equations
already produce a physically reasonable density (close to the
``baseline_xc`` answer) and the optimizer fits only the correction
δV_xc that bridges the baseline to the reference target. This is the
standard Wu-Yang formulation; the alternative ``V_xc = Σ b_t g_t`` form
(no baseline) starts from Hartree-only KS and is numerically much
harder.

Regularization is in V-space via the overlap-metric penalty
``0.5 * lambda * b^T S_aux b`` (``S_aux`` = auxiliary-basis overlap
matrix). The CONCEPT of regularizing in the space of the potential
(rather than the basis-dependent coefficient-norm ``|b|^2``) follows
Heaton-Burgess, Bulat & Yang Phys. Rev. Lett. 98, 256401 (2007),
who introduce a λ-regularized OEP energy functional to tame the ill-
posed finite-basis OEP problem. Note on the implemented form: the
penalty used here is an *overlap-metric (S_aux) Tikhonov / amplitude*
penalty on ``b``, NOT the kinetic-energy smoothness norm of that
paper. Heaton-Burgess Eq. (1) regularizes with the smoothness measure
``‖∇v_b‖^2 = b^T T b`` (``T`` = kinetic-energy integral matrix in the
potential basis); the present code instead penalizes the V_xc
amplitude through the overlap metric ``S_aux``. Both are basis-aware
penalties in V-space, switching ``def2-svp-jkfit`` <->
``def2-tzvp-jkfit`` no longer silently changes the meaning of
``regularization`` as a bare ``|b|^2`` penalty would, but they are
mathematically distinct regularizers (amplitude vs. gradient/curvature
smoothness).
"""
from __future__ import annotations

import os
import warnings
from typing import Any, NamedTuple

import numpy as np
from scipy.optimize import minimize

from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.pyscf_determinism import pin_reference_scf


class OEPResult(NamedTuple):
    """Wu-Yang OEP inversion result.

    Fields
    ------
    n_iter
        L-BFGS-B iteration count clipped at ``max_iter``:
        ``min(scipy.result.nit, max_iter)``. Some scipy builds
        report ``nit == max_iter + 1`` when the optimizer exits at the
        iteration cap; the clip ensures ``n_iter`` matches the user-
        requested cap exactly.
    lbfgs_status
        Scipy's L-BFGS-B termination message. When the
        post-optimization final SCF fails (the SCF whose DM determines
        ``density_error``), ``" + final_scf_failed"`` is appended so a
        consumer can distinguish scipy failure from final-SCF failure
        without inspecting ``converged`` alone.
    terminated_by
        How the optimization terminated: ``"conv_tol"`` (early-stop
        sentinel fired on density-L2 below conv_tol), ``"plateau"``
        (plateau sentinel fired - both density_error and F_val flat
        for plateau_window iterations), or ``"max_iter"`` (neither
        sentinel fired; L-BFGS-B exhausted max_iter). In-memory only;
        not persisted to ``<name>.npz`` by ``save_vxc_ref``. Default
        ``"max_iter"`` for NamedTuple back-compat with existing
        constructions.
    dm_final
        Post-finalization inner-SCF DM corresponding to ``vxc_matrix``.
        Shape ``(n_ao, n_ao)`` for RKS, ``(2, n_ao, n_ao)`` for UKS.
        Set to ``None`` if final-SCF failed. Default ``None`` for
        back-compat. Consumers wanting expectation values must
        spin-sum: ``dm.sum(axis=0) if dm.ndim == 3 else dm``.
        In-memory only; not persisted by ``save_vxc_ref``.
    stop_reason
        A convergence-semantics diagnostic, distinct
        from ``terminated_by`` (which records WHICH sentinel fired). This
        records WHETHER the returned V_xc is a verified converged result
        and, if it stopped early, why:
          * ``"converged"``: the SCF-verified ``density_error``
            (recomputed on the post-optimization SCF density) is below
            ``conv_tol`` AND the optimization either reached a stationary
            point (scipy L-BFGS-B success) or early-stopped on the
            conv_tol sentinel. ``converged`` is True.
          * ``"plateau"``: the optimization stopped on a plateau (the
            density-error / F_val history flattened) rather than at a
            stationary point. ``converged`` is True ONLY if the
            SCF-verified ``density_error`` is below ``conv_tol``; it does
            NOT use the (possibly biased) plateau-median error. Even when
            ``converged`` is True, ``stop_reason`` stays ``"plateau"`` so
            consumers can tell a plateau stop from a true stationary
            convergence and avoid feeding a non-variational V_xc into
            training targets unaware.
          * ``"max_iter"``: L-BFGS-B exhausted ``max_iter`` without a
            sentinel firing; ``converged`` reflects the SCF-verified
            density error vs ``conv_tol``.
        In-memory only; not persisted by ``save_vxc_ref``. Default
        ``"max_iter"`` for NamedTuple back-compat.
    """
    vxc_matrix: np.ndarray
    converged: bool
    n_iter: int
    density_error: float
    # Provenance for downstream loaders/audits.
    baseline_xc: str | None
    aux_basis: str
    regularization: float
    n_electrons: float          # Tr(S . D_target) sanity check
    lbfgs_status: str           # success / message from scipy result
    terminated_by: str = "max_iter"
    dm_final: np.ndarray | None = None
    stop_reason: str = "max_iter"


class _OEPEarlyStop(Exception):
    """Sentinel raised by the L-BFGS-B iter callback inside
    ``run_oep_inversion`` when ``density_error_l2`` at the accepted
    iterate has dropped below ``conv_tol``. Caught immediately after
    ``scipy.optimize.minimize(...)`` returns; the most-recent accepted
    parameter vector is carried out via the ``b`` attribute so the OEP
    can finalize using that iterate rather than running L-BFGS-B all the
    way to its own ftol/gtol/maxiter (which would burn hundreds of extra
    iterations of basin oscillation at the noise floor for UKS Π-state
    inversions).
    """

    def __init__(self, b: np.ndarray) -> None:
        super().__init__("OEP early stop: density_error < conv_tol")
        self.b = b


class _OEPPlateau(Exception):
    """Sentinel raised by the L-BFGS-B iter callback inside
    ``run_oep_inversion`` when the accepted-iterate
    ``density_error_l2`` AND ``F_val`` (unregularized Lagrangian) have
    BOTH plateaued for ``plateau_window`` consecutive accepted iterates
    (after at least ``plateau_min_iter`` iterations have completed).
    Caught immediately after ``scipy.optimize.minimize(...)`` returns;
    the most-recent accepted parameter vector is carried out via the
    ``b`` attribute (mirroring ``_OEPEarlyStop``) so the OEP can
    finalize using that iterate. The plateau density-error value is
    carried separately so consumers can record it as the achievable
    floor for that setting (used by the per-species harness verifier
    in scripts/oep_per_species_emit_overrides.py). The detector watches
    F_val (the unregularized Lagrangian), not the regularized objective.
    """

    def __init__(self, b: np.ndarray, plateau_density_error: float) -> None:
        super().__init__(
            f"OEP plateau: density_error_l2 ~ {float(plateau_density_error):.3e}"
        )
        self.b = b
        self.plateau_density_error = float(plateau_density_error)


def _detect_plateau(
    d_e: list[float] | np.ndarray,
    F_val: list[float] | np.ndarray,
    *,
    plateau_window: int,
    plateau_rtol: float,
) -> tuple[bool, float]:
    """Pure plateau-detection rule.

    Returns ``(fired, plateau_density_error)``. Used by the inline
    detector in `_scipy_iter_callback` and the unit-test suite.

    Fires iff ALL of:
    - ``plateau_window > 0`` AND ``plateau_rtol > 0.0`` (gate kwargs).
    - ``len(d_e) == plateau_window`` AND ``len(F_val) == plateau_window``.
    - ``(max(d_e) - min(d_e)) / max(|median(d_e)|, 1e-30) < plateau_rtol``.
    - ``(max(F_val) - min(F_val)) / max(|median(F_val)|, 1e-30) < plateau_rtol``.
    - ``last-half median(d_e) >= first-half median(d_e) - plateau_rtol * |median|``
      (sign-of-trend with slack).

    Returns ``(True, median(d_e))`` when fired; ``(False, 0.0)`` otherwise.
    """
    if plateau_window <= 0 or plateau_rtol <= 0.0:
        return False, 0.0
    d_e_arr = np.asarray(d_e)
    F_arr = np.asarray(F_val)
    if len(d_e_arr) != plateau_window or len(F_arr) != plateau_window:
        return False, 0.0
    d_e_med = float(np.median(d_e_arr))
    F_med = float(np.median(F_arr))
    d_e_rel = (
        (float(np.max(d_e_arr)) - float(np.min(d_e_arr)))
        / max(abs(d_e_med), 1e-30)
    )
    F_rel = (
        (float(np.max(F_arr)) - float(np.min(F_arr)))
        / max(abs(F_med), 1e-30)
    )
    half = plateau_window // 2
    first_half_med = float(np.median(d_e_arr[:half]))
    last_half_med = float(np.median(d_e_arr[half:]))
    slack = plateau_rtol * abs(d_e_med)
    non_descending = last_half_med >= first_half_med - slack
    if d_e_rel < plateau_rtol and F_rel < plateau_rtol and non_descending:
        return True, d_e_med
    return False, 0.0


def _build_mol_and_mf(mol_spec: MoleculeSpec, basis: str | None = None,
                      baseline_xc: str | None = "pbe"):
    """Build PySCF molecule and run baseline-XC SCF. Returns ``(mol, mf)``.

    ``baseline_xc`` is forwarded as ``mf.xc``. Any pyscf-compatible XC
    string works (case-insensitive): ``"lda"``, ``"pbe"``, ``"blyp"``,
    ``"scan"``, ``"b3lyp"``, ``"hf"``, etc. Pass ``None`` to set
    ``mf.xc = ""`` (Hartree-only baseline) -- not recommended for
    Wu-Yang; only included so callers can opt out explicitly.

    Honors ``mol_spec.grid_level``: when non-None, sets
    ``mf.grids.level = mol_spec.grid_level`` and calls
    ``mf.grids.build()`` BEFORE ``mf.kernel()`` to guarantee the SCF
    (and any downstream consumer of ``mf.grids.coords``) uses the
    requested mesh, avoiding a two-grid mismatch with cached SCF/CCSD
    intermediates built at the same grid_level. When ``None``, PySCF's
    own default (level 3) applies.
    """
    from pyscf import dft, gto
    mol = gto.M(
        atom=mol_spec.atom,
        basis=basis or mol_spec.basis,
        charge=mol_spec.charge,
        spin=mol_spec.spin,
        verbose=0,
    )
    if mol_spec.spin != 0:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    if baseline_xc is None:
        mf.xc = ""
    else:
        mf.xc = str(baseline_xc)
        # A HYBRID baseline carries non-local HF exchange (K). The inner
        # SCF here freezes that K at the baseline DM, so the recovered local
        # V_xc^ref bakes in the frozen non-local piece and is NOT a pure
        # local-multiplier target, using it as a local-V_xc training reference
        # is inconsistent. Warn loudly; prefer a semilocal baseline (e.g. 'pbe').
        try:
            _hyb = float(dft.libxc.hybrid_coeff(mf.xc, spin=mol.spin))
        except (AttributeError, KeyError, ValueError, TypeError):
            _hyb = 0.0
        if abs(_hyb) > 0.0:
            warnings.warn(
                f"OEP baseline_xc={baseline_xc!r} is a hybrid (HF-exchange "
                f"fraction {_hyb:.3g}); the recovered vxc_ref bakes in the "
                f"non-local HF-K frozen at the baseline DM and is NOT a pure "
                f"local-potential target. Prefer a semilocal baseline "
                f"(e.g. 'pbe') for a consistent local-V_xc reference.",
                RuntimeWarning, stacklevel=2,
            )
    if mol_spec.grid_level is not None:
        mf.grids.level = int(mol_spec.grid_level)
        mf.grids.build()
    # Fixed quadrature blocking and integral path (pyscf_determinism): pyscf
    # sizes both from the memory the process has left, which moves the
    # baseline density and every potential derived from it at the 1e-13
    # level with process history.
    pin_reference_scf(mf)
    mf.kernel()
    return mol, mf


def _dm_to_rho_on_grid(mol, mf, dm, *, per_spin: bool = False):
    """Evaluate density on the DFT grid from a density matrix.

    If ``dm`` is 3D (UKS) and ``per_spin`` is True, returns
    ``(rho_a, rho_b)``; otherwise returns the total density (sum over
    spins for UKS).
    """
    coords = mf.grids.coords
    ao = mf._numint.eval_ao(mol, coords)
    dm_arr = np.asarray(dm)
    if dm_arr.ndim == 2:
        rho = np.einsum("pi,ij,pj->p", ao, dm_arr, ao)
        if per_spin:
            return 0.5 * rho, 0.5 * rho
        return rho
    rho_a = np.einsum("pi,ij,pj->p", ao, dm_arr[0], ao)
    rho_b = np.einsum("pi,ij,pj->p", ao, dm_arr[1], ao)
    if per_spin:
        return rho_a, rho_b
    return rho_a + rho_b


def _baseline_vxc_matrix(mol, mf, dm) -> np.ndarray:
    """V_xc^baseline matrix in AO basis from the baseline-XC functional.

    Returns the XC-only matrix (not J): ``V_xc = veff - h_core - J``.
    For UKS, returns shape ``(2, nao, nao)``.
    """
    veff = mf.get_veff(mol, dm)
    # PySCF convention: get_veff returns J + V_xc (no h_core).
    j = mf.get_j(mol, dm)
    veff_arr = np.asarray(veff)
    j_arr = np.asarray(j)
    if veff_arr.ndim == 3:
        # UKS: veff per spin contains J_total + V_xc_s
        j_total = j_arr.sum(axis=0) if j_arr.ndim == 3 else j_arr
        return np.stack([veff_arr[0] - j_total, veff_arr[1] - j_total], axis=0)
    return veff_arr - j_arr


def _build_aux_basis_matrices(mol, mf, aux_basis: str):
    """Build auxiliary-basis matrices for V_xc expansion.

    Returns ``(aux_mol, three_center, aux_on_grid, S_aux)`` where
      * ``three_center[t, i, j] = ∫ g_t(r) φ_i(r) φ_j(r) dr`` (weighted),
        the AO-basis matrix increment per coefficient: V_xc += b_t · 3c[t].
      * ``aux_on_grid[g, t] = g_t(r_g) · w_g`` for grid-side projections.
      * ``S_aux[t, t'] = ∫ g_t(r) g_{t'}(r) dr``: the auxiliary-basis
        overlap matrix used for the V-space (overlap-metric) amplitude
        regularization. V-space regularization follows the
        CONCEPT of Heaton-Burgess PRL 98, 256401 (2007); the implemented
        penalty here is an overlap-metric (S_aux) amplitude/Tikhonov term,
        NOT that paper's kinetic-energy smoothness norm ``b^T T b``. A
        coefficient-space ‖b‖² regularization would be basis-dependent and
        silently change meaning when aux_basis is swapped.

    Uses ``GTOval_sph`` if ``mol.cart`` is False; ``GTOval_cart``
    otherwise (a hardcoded ``GTOval_sph`` would be inconsistent
    with cartesian-basis molecules).
    """
    from pyscf import gto as gto_mod
    aux_mol = gto_mod.M(
        atom=mol.atom, basis=aux_basis, charge=mol.charge,
        spin=mol.spin, verbose=0,
    )
    coords = mf.grids.coords
    weights = mf.grids.weights
    gto_val = "GTOval_cart" if getattr(mol, "cart", False) else "GTOval_sph"
    ao_aux = aux_mol.eval_gto(gto_val, coords)
    ao_orb = mf._numint.eval_ao(mol, coords)
    # Fused 4-tensor einsum, ~10-100x faster than Python loop.
    three_center = np.einsum(
        "gt,gi,gj,g->tij", ao_aux, ao_orb, ao_orb, weights, optimize=True,
    )
    # Symmetrize against AO-quadrature noise so V_xc = V_xc^baseline +
    # Σ b_t · three_center[t] is exactly Hermitian in (i, j), analytically
    # three_center[t,i,j] == three_center[t,j,i] from φ_iφ_j = φ_jφ_i, but
    # finite-grid quadrature breaks symmetry at ε_machine·N_grid.
    three_center = 0.5 * (three_center + three_center.transpose(0, 2, 1))
    aux_on_grid = ao_aux * weights[:, None]
    # Auxiliary-basis overlap matrix from grid quadrature.
    S_aux = np.einsum("gt,gu,g->tu", ao_aux, ao_aux, weights, optimize=True)
    # Symmetrize against quadrature noise.
    S_aux = 0.5 * (S_aux + S_aux.T)
    return aux_mol, three_center, aux_on_grid, S_aux


def _ks_from_vxc_matrix(mol, mf, vxc_matrix, *, dm0=None, level_shift=0.0,
                         damp: float = 0.1, diis_start_cycle: int = 5):
    """Dispatch RHF or UHF inner SCF based on vxc_matrix dimensionality
    OR mol.spin (a 2-D vxc with mol.spin>0 still routes to UHF).

    ``level_shift`` is forwarded to the inner ``mf_fixed.level_shift``
    attribute (PySCF SCF stabilization). ``damp`` and
    ``diis_start_cycle`` are forwarded to the inner ``mf_fixed.damp`` /
    ``mf_fixed.diis_start_cycle`` (defaults 0.1 and 5; PySCF ship
    defaults are 0.0 and 1, respectively).
    """
    v = np.asarray(vxc_matrix)
    if v.ndim == 3 or mol.spin != 0:
        return _ks_from_vxc_matrix_uhf(
            mol, mf, vxc_matrix, dm0=dm0, level_shift=level_shift,
            damp=damp, diis_start_cycle=diis_start_cycle,
        )
    return _ks_from_vxc_matrix_rhf(
        mol, mf, vxc_matrix, dm0=dm0, level_shift=level_shift,
        damp=damp, diis_start_cycle=diis_start_cycle,
    )


def _ks_from_vxc_matrix_rhf(mol, mf, vxc_matrix, *, dm0=None, level_shift=0.0,
                             damp: float = 0.1, diis_start_cycle: int = 5):
    """RHF inner SCF with damped + DIIS-stabilized convergence.

    Returns ``(dm, ts, j_matrix, success)``. ``success=False`` indicates
    SCF blew up (LinAlgError, ValueError, or non-finite output); the
    outer Wu-Yang objective should treat this as +inf so L-BFGS-B backs
    off. Silently returning ``dm0`` and a finite ``ts`` instead would
    leave the objective and gradient inconsistent on failed line-search
    probes.

    ``level_shift`` (default 0.0) is forwarded to ``mf_fixed.level_shift``;
    closed-shell RKS rarely needs this, but the kwarg exists for symmetry
    with the UHF path.
    """
    from pyscf import scf
    from numpy.linalg import LinAlgError as _NpLinAlgError
    from scipy.linalg import LinAlgError as _ScLinAlgError

    mf_fixed = scf.RHF(mol)
    # No quadrature here (get_veff is replaced below by J plus the fixed
    # matrix); the incore/direct choice of J is pinned to the system size so
    # the inner SCF does not follow process memory (pyscf_determinism).
    pin_reference_scf(mf_fixed)
    mf_fixed.verbose = 0
    mf_fixed.max_cycle = 200
    mf_fixed.conv_tol = 1e-10
    # DIIS + damping for robustness against ill-conditioned line-search
    # probes from L-BFGS-B (Pulay DIIS, Pulay CPL 73, 393 (1980); the
    # diis_start_cycle delays activation until damped iterations settle
    # the wavefunction).
    mf_fixed.diis_start_cycle = diis_start_cycle
    mf_fixed.diis_space = 4
    mf_fixed.damp = damp
    if level_shift != 0.0:
        mf_fixed.level_shift = level_shift

    def get_veff_fixed(mol_, dm_, *args, **kwargs):
        j_mat = mf_fixed.get_j(mol_, dm_)
        return j_mat + vxc_matrix

    mf_fixed.get_veff = get_veff_fixed
    if dm0 is None:
        dm0 = mf.make_rdm1()
    dm0 = np.asarray(dm0)
    if dm0.ndim == 3:
        dm0 = dm0.sum(axis=0)

    success = True
    try:
        mf_fixed.kernel(dm0=dm0)
        if not np.all(np.isfinite(mf_fixed.mo_coeff)):
            raise _NpLinAlgError("inner SCF produced non-finite MO coefficients")
        dm_final = mf_fixed.make_rdm1()
        if not np.all(np.isfinite(dm_final)):
            raise _NpLinAlgError("inner SCF produced non-finite DM")
    except (_NpLinAlgError, _ScLinAlgError, ValueError):
        dm_final = np.asarray(dm0)
        success = False

    j_matrix = mf_fixed.get_j(mol, dm_final)
    t_matrix = mol.intor("int1e_kin")
    ts = float(np.einsum("ij,ij->", t_matrix, dm_final))
    return dm_final, ts, j_matrix, success


def _ks_from_vxc_matrix_uhf(mol, mf, vxc_matrix, *, dm0=None, level_shift=0.0,
                             damp: float = 0.1, diis_start_cycle: int = 5):
    """UHF inner SCF; returns ``(dm, ts, j_matrix, success)``.

    ``level_shift`` (default 0.0) forwards to ``mf_fixed.level_shift`` and
    is critical for OEP on UKS species with orbital degeneracy
    (X²Π radicals like HO, CN, NO; X²A1 like NO2). Without level-shifting,
    the inner SCF flips between symmetry-equivalent basins (e.g. π_x vs
    π_y singly-occupied) under L-BFGS-B perturbations of the OEP
    coefficients ``b``, breaking F(b) smoothness. Recommended: 0.5 Ha
    for UKS species, 0.0 for closed-shell. See
    ``xcquinox/alec/tests/test_oep_uks.py`` module docstring for
    background on the basin-hopping failure mode.
    """
    from pyscf import scf

    v = np.asarray(vxc_matrix)
    if v.ndim != 3 or v.shape[0] != 2:
        raise ValueError(
            f"_ks_from_vxc_matrix_uhf expects vxc_matrix shape (2, nao, nao), "
            f"got {v.shape}"
        )

    from numpy.linalg import LinAlgError as _NpLinAlgError
    from scipy.linalg import LinAlgError as _ScLinAlgError

    mf_fixed = scf.UHF(mol)
    # As in the RHF inner SCF: no quadrature, integral path pinned.
    pin_reference_scf(mf_fixed)
    mf_fixed.verbose = 0
    mf_fixed.max_cycle = 200
    mf_fixed.conv_tol = 1e-10
    mf_fixed.diis_start_cycle = diis_start_cycle
    mf_fixed.diis_space = 4
    mf_fixed.damp = damp
    if level_shift != 0.0:
        mf_fixed.level_shift = level_shift

    def get_veff_fixed(mol_, dm_, *args, **kwargs):
        dm_arr = np.asarray(dm_)
        if dm_arr.ndim == 2:
            dm_arr = np.stack([0.5 * dm_arr, 0.5 * dm_arr], axis=0)
        j = mf_fixed.get_j(mol_, dm_arr)
        j_total = j[0] + j[1]
        return np.stack(
            [j_total + vxc_matrix[0], j_total + vxc_matrix[1]], axis=0,
        )

    mf_fixed.get_veff = get_veff_fixed
    if dm0 is None:
        dm0 = mf.make_rdm1()
    dm0 = np.asarray(dm0)
    if dm0.ndim == 2:
        dm0 = np.stack([0.5 * dm0, 0.5 * dm0], axis=0)

    success = True
    try:
        mf_fixed.kernel(dm0=dm0)
        if not np.all(np.isfinite(mf_fixed.mo_coeff)):
            raise _NpLinAlgError("inner UHF SCF produced non-finite MO coefficients")
        dm_final = mf_fixed.make_rdm1()
        if not np.all(np.isfinite(dm_final)):
            raise _NpLinAlgError("inner UHF SCF produced non-finite DM")
    except (_NpLinAlgError, _ScLinAlgError, ValueError):
        dm_final = np.asarray(dm0)
        success = False

    j_matrix = mf_fixed.get_j(mol, dm_final)
    # PySCF version-defensive normalization. UHF + 3-D DM in
    # PySCF ≥ 2.0 returns spin-resolved (2, nao, nao); older versions
    # may return spin-summed (nao, nao). Downstream callers index
    # j_matrix[0]/j_matrix[1] unconditionally, guarantee 3-D here.
    j_matrix = np.asarray(j_matrix)
    if j_matrix.ndim == 2:
        j_matrix = np.stack([0.5 * j_matrix, 0.5 * j_matrix], axis=0)
    t_matrix = mol.intor("int1e_kin")
    ts = float(
        np.einsum("ij,ij->", t_matrix, dm_final[0])
        + np.einsum("ij,ij->", t_matrix, dm_final[1])
    )
    return dm_final, ts, j_matrix, success


def run_oep_inversion(
    mol_spec: MoleculeSpec,
    dm_target: np.ndarray,
    *,
    basis: str | None = None,
    baseline_xc: str | None = "pbe",
    aux_basis: str = "def2-svp-jkfit",
    max_iter: int = 200,
    conv_tol: float = 1e-6,
    regularization: float = 1e-4,
    level_shift: float = 0.0,
    progress_callback=None,
    plateau_window: int = 20,
    plateau_rtol: float = 0.02,
    plateau_min_iter: int = 30,
    inner_damp: float = 0.1,
    inner_diis_start_cycle: int = 5,
) -> OEPResult:
    """Wu-Yang displacement-form OEP inversion.

    Finds a coefficient vector ``b`` such that the KS potential

        V_xc(r) = V_xc^baseline(r) + Σ_t b_t · g_t(r)

    yields a Kohn-Sham density that matches ``dm_target`` (e.g. a
    CCSD(T) reference DM). The displacement form (Wu & Yang JCP 118,
    2498 (2003) §II.B) lets the auxiliary basis fit only the
    correction between the baseline and reference; ``b = 0`` already
    gives a physical KS density rather than the Hartree-only pathology
    of the bare ``V_xc = Σ b_t g_t`` ansatz.

    Parameters
    ----------
    mol_spec : MoleculeSpec
        Molecule + basis specification.
    dm_target : np.ndarray
        Target density matrix in AO basis. RHF: ``(nao, nao)``;
        UKS: ``(2, nao, nao)``.
    basis : str | None
        Override ``mol_spec.basis``.
    baseline_xc : str | None
        XC functional string for the displacement baseline. Any
        pyscf-compatible string: ``"lda"``, ``"pbe"`` (default),
        ``"blyp"``, ``"scan"``, ``"hf"``, etc. Pass ``None`` for
        Hartree-only baseline (NOT recommended; see Wu & Yang §II.B).

        Hybrid functionals (e.g. ``"b3lyp"``): the baseline V_xc
        matrix captures the local exchange-correlation portion at the
        baseline-XC SCF DM, but the non-local exact-exchange (HF-K) piece
        is frozen at the baseline DM rather than recomputed from the
        evolving inner-SCF DM. This means at b=0 the inner SCF does
        NOT exactly reproduce the hybrid baseline DM (the K piece
        becomes inconsistent with the new D). Hybrid baselines work in
        practice but do not enjoy the "b=0 = baseline answer" property
        that pure-DFT baselines do; expect slightly more L-BFGS-B
        iterations to compensate.
    aux_basis : str
        Auxiliary basis for V_xc expansion. Default
        ``"def2-svp-jkfit"`` (matches step6 notebook). Larger bases
        (``def2-tzvp-jkfit``) give finer V_xc resolution at higher
        cost; with V-space regularization the meaning of
        ``regularization`` is basis-independent.
    max_iter : int
        L-BFGS-B max iterations.
    conv_tol : float
        Density-error L2 tolerance for the ``converged`` flag.
    regularization : float
        Overlap-metric (S_aux) Tikhonov / amplitude penalty in V-space:
        ``+0.5 * lambda * b^T S_aux b``. The CONCEPT of V-space
        regularization follows Heaton-Burgess et al. PRL 98, 256401
        (2007); the implemented penalty is an amplitude term in the
        overlap metric, NOT that paper's kinetic-energy smoothness
        norm ``b^T T b``. Penalizes the V_xc(r) magnitude in a way that
        is basis-independent in meaning regardless of which auxiliary
        basis is chosen. A plain ``0.5 * lambda * |b|^2`` penalty would
        instead silently change meaning when ``aux_basis`` was swapped.
    level_shift : float
        Energy shift (Ha) applied to virtual orbitals during the inner
        SCF (``mf_fixed.level_shift``). Default 0.0.
        Use ``level_shift=0.5`` for UKS species with orbital
        degeneracy (X²Π radicals like HO, CN, NO; near-degenerate cases
        like NO2's X²A1) to suppress basin-hopping during DIIS, the
        inner SCF would otherwise flip between symmetry-equivalent
        broken-symmetry solutions under L-BFGS-B perturbations of ``b``,
        making F(b) non-smooth and preventing convergence to
        ``conv_tol``. Without level-shifting, HO at def2-svp/grid_level=1
        plateaus at ``density_error≈0.17``; with ``level_shift=0.5``,
        it converges to ``density_error<2e-3``. Closed-shell RKS doesn't
        need this; default 0.0 is correct for those.
    progress_callback : callable or None
        Optional ``fn(iter_int, density_error_float)`` invoked once per
        L-BFGS outer iteration.

    Returns
    -------
    OEPResult with provenance fields.
    """
    mol, mf = _build_mol_and_mf(mol_spec, basis, baseline_xc=baseline_xc)
    _, three_center, aux_on_grid, S_aux = _build_aux_basis_matrices(
        mol, mf, aux_basis,
    )
    n_aux = three_center.shape[0]
    weights = mf.grids.weights
    h_core = mf.get_hcore()

    # A spin=0 mol with 3-D dm_target is incoherent.
    # RKS mol means mf=RKS and vxc_baseline is 2-D; coercing the path to
    # UKS (because dm_target.ndim==3) silently broadcasts 1-D rows through
    # the 3-D vxc construction. Reject up-front with a clear error.
    dm_target_arr_check = np.asarray(dm_target)
    if mol.spin == 0 and dm_target_arr_check.ndim == 3:
        raise ValueError(
            f"OEP target DM has shape {dm_target_arr_check.shape} "
            f"(spin-resolved UKS) but mol_spec.spin = 0 (closed-shell). "
            f"Either pass a 2-D RKS dm_target (sum of alpha + beta) or "
            f"set mol_spec.spin > 0 for an open-shell inversion."
        )
    is_uks = (mol.spin != 0) or (dm_target_arr_check.ndim == 3)

    # Shape + nelectron sanity check on the target DM. If
    # Tr(S * D_target) is far from the actual electron count OR the
    # shape mismatches the AO basis, the target was built in a different
    # basis than mol_spec.basis and the inversion would silently produce
    # garbage. Catch both pathologies here with a clear "different basis"
    # message so callers know what's wrong.
    s1e = mol.intor("int1e_ovlp")
    dm_target_arr = np.asarray(dm_target)
    expected_shape_2d = (s1e.shape[0], s1e.shape[1])
    expected_shape_3d = (2, s1e.shape[0], s1e.shape[1])
    if dm_target_arr.ndim == 2:
        if dm_target_arr.shape != expected_shape_2d:
            raise ValueError(
                f"OEP target DM shape {dm_target_arr.shape} does not match "
                f"AO-basis shape {expected_shape_2d} for mol_spec.basis "
                f"({mol_spec.basis!r}). The dm_target is in a different "
                f"basis than mol_spec.basis."
            )
        n_elec_target = float(np.einsum("ij,ij->", s1e, dm_target_arr))
    elif dm_target_arr.ndim == 3:
        if dm_target_arr.shape != expected_shape_3d:
            raise ValueError(
                f"OEP target DM shape {dm_target_arr.shape} does not match "
                f"AO-basis UKS shape {expected_shape_3d} for "
                f"mol_spec.basis ({mol_spec.basis!r}). The dm_target is "
                f"in a different basis than mol_spec.basis."
            )
        n_elec_target = float(
            np.einsum("ij,ij->", s1e, dm_target_arr[0])
            + np.einsum("ij,ij->", s1e, dm_target_arr[1])
        )
    else:
        raise ValueError(
            f"OEP target DM must be 2-D (RKS) or 3-D (UKS); got "
            f"shape {dm_target_arr.shape}"
        )
    n_elec_expected = float(mol.nelectron)
    if abs(n_elec_target - n_elec_expected) > 1e-3:
        raise ValueError(
            f"OEP target DM has Tr(S*D) = {n_elec_target:.6f} but mol expects "
            f"{n_elec_expected:.0f} electrons. The dm_target is likely in a "
            f"different basis than mol_spec.basis ({mol_spec.basis!r})."
        )

    # Baseline V_xc^baseline matrix(es). For RKS this is shape (nao, nao);
    # for UKS, (2, nao, nao). Computed at the BASELINE-XC SCF DM so the
    # zero-coefficient state of the optimizer corresponds to the baseline.
    vxc_baseline = _baseline_vxc_matrix(mol, mf, mf.make_rdm1())

    if is_uks:
        rho_target_a, rho_target_b = _dm_to_rho_on_grid(
            mol, mf, dm_target, per_spin=True,
        )
        rho_target_total = rho_target_a + rho_target_b
        rhotarget_a_integrals = np.einsum("gp,g->p", aux_on_grid, rho_target_a)
        rhotarget_b_integrals = np.einsum("gp,g->p", aux_on_grid, rho_target_b)
    else:
        rho_target_total = _dm_to_rho_on_grid(mol, mf, dm_target)
        rhotarget_integrals = np.einsum("gp,g->p", aux_on_grid, rho_target_total)

    def _vxc_from_b(b):
        """Build V_xc = V_xc^baseline + Σ_t b_t · g_t (displacement form)."""
        if is_uks:
            b_a = b[:n_aux]
            b_b = b[n_aux:]
            delta_a = np.einsum("t,tij->ij", b_a, three_center)
            delta_b = np.einsum("t,tij->ij", b_b, three_center)
            return np.stack(
                [vxc_baseline[0] + delta_a, vxc_baseline[1] + delta_b],
                axis=0,
            )
        return vxc_baseline + np.einsum("t,tij->ij", b, three_center)

    # Keep the most recent ACCEPTED iterate's DM, not the
    # last objective-evaluation's DM (which may correspond to a rejected
    # line-search trial).
    scf_state: dict[str, Any] = {
        "dm0_accepted": None,
        "dm0_last_eval": None,
        "density_error_l2_accepted": float("inf"),
        "density_error_l2_last_eval": float("inf"),
        "F_val_accepted": float("-inf"),
        "F_val_last_eval": float("-inf"),
    }
    _progress_state = {"iter": 0, "density_error_l2": float("inf")}

    # Plateau detector setup. Two deques tracking the
    # last `plateau_window` accepted-iterate snapshots of
    # density_error_l2 and F_val. Plateau fires when both are flat
    # within `plateau_rtol` AND density_error_l2 is non-descending
    # (last-half median >= first-half median, with rtol-scaled slack).
    # Pass plateau_window=0 / plateau_rtol=0 / plateau_min_iter > max_iter
    # to disable.
    from collections import deque
    _plateau_density_error_deque: deque = deque(
        maxlen=plateau_window if plateau_window > 0 else 1
    )
    _plateau_F_val_deque: deque = deque(
        maxlen=plateau_window if plateau_window > 0 else 1
    )

    def objective_and_grad(b):
        vxc_matrix = _vxc_from_b(b)
        dm_scf, _, j_matrix, scf_success = _ks_from_vxc_matrix(
            mol, mf, vxc_matrix, dm0=scf_state["dm0_last_eval"],
            level_shift=level_shift,
            damp=inner_damp,
            diis_start_cycle=inner_diis_start_cycle,
        )
        scf_state["dm0_last_eval"] = dm_scf

        # On inner-SCF failure, return a large objective
        # and zero-magnitude (but non-NaN) gradient so L-BFGS-B's Wolfe
        # line search backs off rather than treating the failure as a
        # successful evaluation at a bogus point.
        if not scf_success:
            obj = 1e20
            grad = np.zeros_like(b)
            # On inner-SCF failure, write
            # +inf for density_error and -inf for F_val. The plateau
            # detector reads these as descending sentinels (won't
            # contribute to a flat-window judgement).
            scf_state["density_error_l2_last_eval"] = float("inf")
            scf_state["F_val_last_eval"] = float("-inf")
            return obj, grad

        if is_uks:
            rho_scf_a, rho_scf_b = _dm_to_rho_on_grid(
                mol, mf, dm_scf, per_spin=True,
            )
            delta_a = rho_scf_a - rho_target_a
            delta_b = rho_scf_b - rho_target_b
            j_total = j_matrix[0] + j_matrix[1]
            dm_total = dm_scf[0] + dm_scf[1]
            e_ks = (
                float(np.einsum("ij,ij->", dm_total, h_core))
                + 0.5 * float(np.einsum("ij,ij->", dm_total, j_total))
                + float(np.einsum("ij,ij->", dm_scf[0], vxc_matrix[0]))
                + float(np.einsum("ij,ij->", dm_scf[1], vxc_matrix[1]))
            )
            b_a = b[:n_aux]
            b_b = b[n_aux:]
            F_val = (
                e_ks
                - float(np.dot(b_a, rhotarget_a_integrals))
                - float(np.dot(b_b, rhotarget_b_integrals))
            )
            # V-space regularization. UKS uses a
            # JOINT (b_a + b_b) regularization rather than per-spin so
            # spin-symmetric solutions aren't artificially broken when
            # the target is closed-shell.
            b_sum = b_a + b_b
            reg_term = 0.5 * regularization * float(b_sum @ S_aux @ b_sum)
            obj = -F_val + reg_term
            # Gradient of reg_term wrt b_a or b_b: regularization * S_aux @ b_sum
            reg_grad = regularization * (S_aux @ b_sum)
            grad_a = -np.einsum("gp,g->p", aux_on_grid, delta_a) + reg_grad
            grad_b = -np.einsum("gp,g->p", aux_on_grid, delta_b) + reg_grad
            grad = np.concatenate([grad_a, grad_b])
            _delta_tot = (rho_scf_a + rho_scf_b) - rho_target_total
            _progress_state["density_error_l2"] = float(
                np.sqrt(np.sum(weights * _delta_tot ** 2))
            )
            # Snapshot density_error_l2 and F_val (the
            # unregularized Lagrangian, NOT obj=-F_val+reg_term)
            # for the plateau detector. F_val computed above.
            scf_state["density_error_l2_last_eval"] = (
                _progress_state["density_error_l2"]
            )
            scf_state["F_val_last_eval"] = float(F_val)
            return obj, grad

        rho_scf = _dm_to_rho_on_grid(mol, mf, dm_scf)
        delta_rho = rho_scf - rho_target_total
        e_ks = (
            float(np.einsum("ij,ij->", dm_scf, h_core))
            + 0.5 * float(np.einsum("ij,ij->", dm_scf, j_matrix))
            + float(np.einsum("ij,ij->", dm_scf, vxc_matrix))
        )
        F_val = e_ks - float(np.dot(b, rhotarget_integrals))
        # V-space regularization.
        reg_term = 0.5 * regularization * float(b @ S_aux @ b)
        obj = -F_val + reg_term
        reg_grad = regularization * (S_aux @ b)
        grad = -np.einsum("gp,g->p", aux_on_grid, delta_rho) + reg_grad
        _progress_state["density_error_l2"] = float(
            np.sqrt(np.sum(weights * delta_rho ** 2))
        )
        # Snapshot density_error_l2 and F_val (the
        # unregularized Lagrangian, computed above).
        scf_state["density_error_l2_last_eval"] = (
            _progress_state["density_error_l2"]
        )
        scf_state["F_val_last_eval"] = float(F_val)
        return obj, grad

    def _scipy_iter_callback(_xk):
        # Snapshot the most-recent
        # ACCEPTED objective_and_grad's outputs so the plateau detector
        # reads only accepted iterates (not rejected line-search probes,
        # which can leave stale +inf / 1e20 values from SCF failures).
        scf_state["dm0_accepted"] = scf_state["dm0_last_eval"]
        scf_state["density_error_l2_accepted"] = scf_state["density_error_l2_last_eval"]
        scf_state["F_val_accepted"] = scf_state["F_val_last_eval"]
        _progress_state["iter"] += 1
        if progress_callback is not None:
            # NOTE: deliberately reads _progress_state["density_error_l2"]
            # (last-eval) rather than scf_state["density_error_l2_accepted"]
            # progress reporting reflects the most recent evaluation for
            # liveness, while the early-stop and plateau checks below read
            # the accepted-iterate snapshot for correctness.
            progress_callback(
                _progress_state["iter"],
                _progress_state["density_error_l2"],
            )

        # Plateau detector. Append accepted-iterate
        # snapshots to both deques; check plateau BEFORE early-stop
        # so a plateau-below-conv_tol convergence outranks an
        # early-stop on the same iterate.
        if (plateau_window > 0
                and plateau_rtol > 0.0
                and _progress_state["iter"] >= plateau_min_iter):
            _plateau_density_error_deque.append(
                scf_state["density_error_l2_accepted"]
            )
            _plateau_F_val_deque.append(scf_state["F_val_accepted"])
            if (len(_plateau_density_error_deque) == plateau_window
                    and len(_plateau_F_val_deque) == plateau_window):
                _fired, _d_e_med = _detect_plateau(
                    d_e=list(_plateau_density_error_deque),
                    F_val=list(_plateau_F_val_deque),
                    plateau_window=plateau_window,
                    plateau_rtol=plateau_rtol,
                )
                if _fired:
                    raise _OEPPlateau(
                        b=np.asarray(_xk).copy(),
                        plateau_density_error=_d_e_med,
                    )

        # Early-stop: when the density-L2 at the accepted iterate
        # satisfies the user's conv_tol, abort minimize() rather than
        # running the full max_iter. L-BFGS-B's own stopping criteria
        # (ftol, gtol) don't react to density_error directly, so without
        # this check UKS Π-state cases plateau at the noise floor for
        # hundreds of extra iterations even after density_error has
        # dropped below conv_tol. The sentinel exception is caught
        # immediately after minimize() returns; the most-recent accepted
        # iterate (_xk) is carried out via the sentinel payload.
        if scf_state["density_error_l2_accepted"] < conv_tol:
            raise _OEPEarlyStop(np.asarray(_xk).copy())

    b0 = np.zeros(2 * n_aux if is_uks else n_aux)

    early_stopped_b = None
    plateau_b = None
    plateau_density_error = None
    try:
        result = minimize(
            objective_and_grad,
            b0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-12},
            callback=_scipy_iter_callback,
        )
    except _OEPEarlyStop as _es:
        early_stopped_b = _es.b
        result = None  # not used in early-stop path
    except _OEPPlateau as _pl:
        # Plateau handler, parallel to early-stop.
        # The b is used for vxc_final reconstruction; the
        # plateau_density_error is carried separately for the
        # OEPResult.density_error override below.
        plateau_b = _pl.b
        plateau_density_error = _pl.plateau_density_error
        result = None

    if early_stopped_b is not None:
        b_final = early_stopped_b
    elif plateau_b is not None:
        b_final = plateau_b
    else:
        b_final = result.x
    vxc_final = _vxc_from_b(b_final)
    # Run the final SCF from the most recently ACCEPTED warm-start,
    # not from a possibly-rejected trial DM.
    final_warm = (
        scf_state["dm0_accepted"]
        if scf_state["dm0_accepted"] is not None
        else scf_state["dm0_last_eval"]
    )
    dm_final, _, _, final_success = _ks_from_vxc_matrix(
        mol, mf, vxc_final, dm0=final_warm, level_shift=level_shift,
        damp=inner_damp,
        diis_start_cycle=inner_diis_start_cycle,
    )
    if is_uks:
        rho_final_a, rho_final_b = _dm_to_rho_on_grid(
            mol, mf, dm_final, per_spin=True,
        )
        rho_final = rho_final_a + rho_final_b
    else:
        rho_final = _dm_to_rho_on_grid(mol, mf, dm_final)
    final_error = float(
        np.sqrt(np.sum(weights * (rho_target_total - rho_final) ** 2))
    )
    # Never return a potential worse than not optimizing. The b = 0
    # baseline is re-solved through the same inner-SCF machinery and kept
    # when the optimized iterate regressed past it (measured on H2/6-31g
    # at module defaults with a CCSD target: b=0 density error 3.97e-3 vs
    # 4.9e-1 for the returned iterate -- a ~100x regression that scipy's
    # own ftol stop accepted silently; the finite-basis Wu-Yang pathology
    # class). The regression is recorded in stop_reason/lbfgs_status.
    regressed_below_baseline = False
    vxc_b0 = _vxc_from_b(np.zeros_like(np.asarray(b_final)))
    dm_b0, _, _, b0_success = _ks_from_vxc_matrix(
        mol, mf, vxc_b0, dm0=final_warm, level_shift=level_shift,
        damp=inner_damp,
        diis_start_cycle=inner_diis_start_cycle,
    )
    if b0_success:
        if is_uks:
            rho_b0_a, rho_b0_b = _dm_to_rho_on_grid(
                mol, mf, dm_b0, per_spin=True,
            )
            rho_b0 = rho_b0_a + rho_b0_b
        else:
            rho_b0 = _dm_to_rho_on_grid(mol, mf, dm_b0)
        b0_error = float(
            np.sqrt(np.sum(weights * (rho_target_total - rho_b0) ** 2))
        )
        if np.isfinite(b0_error) and (
                not np.isfinite(final_error) or b0_error < final_error):
            regressed_below_baseline = True
            optimized_error = final_error
            vxc_final, dm_final = vxc_b0, dm_b0
            final_error, final_success = b0_error, b0_success
    # Clip scipy's reported nit at our requested max_iter so
    # n_iter never exceeds what the user asked for; documented in the
    # OEPResult.n_iter docstring above.
    if early_stopped_b is not None or plateau_b is not None:
        n_iter = min(_progress_state["iter"], max_iter)
    else:
        n_iter = min(int(result.nit), max_iter)
    # Convergence semantics. The user's contract is: "the V_xc that
    # ``run_oep_inversion`` returns produces a KS density that matches
    # ``dm_target`` to within ``conv_tol``." That depends on:
    #   1. final_success, the post-optimization SCF actually solved
    #   2. final_error < conv_tol, the density matches to tolerance
    #   3. final_error is finite (rules out NaN from a blown-up SCF)
    #
    # We deliberately do NOT also require ``result.success``: that flag is
    # False whenever scipy exits at ``max_iter`` even if density_error
    # already passed conv_tol, which would conflate "L-BFGS-B optimizer
    # converged" with "OEP inversion converged". Hitting max_iter
    # at a tight density_error is still a successful inversion: the V_xc at
    # iteration N is mathematically a valid Wu-Yang displacement
    # (V_xc^baseline + Σ b_t^(N) g_t) regardless of whether the gradient
    # had reached scipy's pgtol/factr threshold.
    converged = bool(
        final_success
        and np.isfinite(final_error)
        and (final_error < conv_tol)
    )
    if early_stopped_b is not None:
        lbfgs_status = (
            f"early_stopped (density_error<{conv_tol:.2e} "
            f"at iter {n_iter}/{max_iter})"
        )
        plateau_terminated = False
    elif plateau_b is not None:
        lbfgs_status = (
            f"plateau (density_error~{plateau_density_error:.2e} "
            f"at iter {n_iter}/{max_iter})"
        )
        plateau_terminated = True
    else:
        lbfgs_status = str(getattr(result, "message", "no message"))
        plateau_terminated = False
    # Surface final-SCF failure in lbfgs_status so a consumer
    # reading converged=False can distinguish "scipy failed" (its message)
    # from "scipy succeeded but the post-optimization SCF blew up".
    if not final_success:
        lbfgs_status = lbfgs_status + " + final_scf_failed"

    # Determine terminated_by and the appropriate density_error to report.
    #
    # ALWAYS report the SCF-verified ``final_error``
    # (recomputed above on the post-optimization SCF density), never the
    # plateau MEDIAN. The plateau median is a flatness statistic of the
    # density-error/F_val history, not the residual of the iterate we
    # actually return; reporting it (and worse, marking ``converged`` from
    # it) could feed a biased, non-variational V_xc into training targets
    # while labeling it converged. The plateau median is still surfaced
    # for diagnostics via ``lbfgs_status`` (built above).
    if early_stopped_b is not None:
        terminated_by = "conv_tol"
    elif plateau_terminated:
        terminated_by = "plateau"
    else:
        terminated_by = "max_iter"
    density_error_reported = final_error

    # ``converged`` was already computed (above) from the SCF-verified
    # condition: final_success AND isfinite(final_error) AND
    # final_error < conv_tol. That is the genuine-stationarity / matches-
    # to-tolerance contract and applies uniformly to every stop path,
    # including plateau. We deliberately do NOT re-derive ``converged``
    # from the plateau median here.
    #
    # ``stop_reason`` records WHETHER this is a verified convergence and,
    # if it stopped early, why, distinct from ``terminated_by`` (which
    # sentinel fired). A plateau stop keeps stop_reason="plateau" even
    # when converged is True, so downstream can distinguish it from a
    # true stationary-point convergence.
    if regressed_below_baseline:
        stop_reason = "regressed_below_baseline"
        lbfgs_status = (
            lbfgs_status
            + f" + regressed_below_baseline(optimized_error="
              f"{optimized_error:.3e}, baseline kept)"
        )
    elif plateau_terminated:
        stop_reason = "plateau"
    elif converged:
        stop_reason = "converged"
    else:
        stop_reason = terminated_by
    # dm_final = post-finalization SCF DM. On final-SCF
    # failure, set None so the harness's bias check can skip safely.
    dm_final_returned = dm_final if final_success else None

    return OEPResult(
        vxc_matrix=vxc_final,
        converged=converged,
        n_iter=n_iter,
        density_error=density_error_reported,
        baseline_xc=baseline_xc,
        aux_basis=aux_basis,
        regularization=regularization,
        n_electrons=n_elec_target,
        lbfgs_status=lbfgs_status,
        terminated_by=terminated_by,
        dm_final=dm_final_returned,
        stop_reason=stop_reason,
    )


def save_vxc_ref(
    oep_result: OEPResult,
    output_path: str,
    *,
    dm_target: np.ndarray | None = None,
    method: str = "CCSD",
) -> None:
    """Save OEP result as .npz compatible with ``_load_external_data``.

    Provenance fields (``oep_baseline_xc``,
    ``oep_aux_basis``, ``oep_regularization``, ``oep_density_error``,
    ``oep_converged``, ``oep_lbfgs_status``, ``oep_n_electrons``) are
    written so downstream loaders can validate consistency; writing only
    ``vxc_ref`` and ``ref_density_method`` would allow a
    wrong-basis or wrong-baseline V_xc to load silently.

    If the file already exists, merges new keys with existing ones.
    """
    payload: dict[str, np.ndarray | str] = {"vxc_ref": oep_result.vxc_matrix}
    if dm_target is not None:
        payload["dm_target"] = dm_target
    if method:
        payload["ref_density_method"] = np.array(method)
    # Provenance, written even when None / empty so loaders can detect.
    payload["oep_baseline_xc"] = np.array(
        "" if oep_result.baseline_xc is None else oep_result.baseline_xc
    )
    payload["oep_aux_basis"] = np.array(oep_result.aux_basis)
    payload["oep_regularization"] = np.array(oep_result.regularization)
    payload["oep_density_error"] = np.array(oep_result.density_error)
    payload["oep_converged"] = np.array(bool(oep_result.converged))
    payload["oep_lbfgs_status"] = np.array(oep_result.lbfgs_status)
    payload["oep_n_electrons"] = np.array(oep_result.n_electrons)
    # Structured stop provenance (previously free text inside lbfgs_status
    # only): a consumer of oep_converged=True cannot otherwise tell a true
    # stationary convergence from a plateau/early stop, nor interpret the
    # error without the tolerance it was accepted against.
    payload["oep_stop_reason"] = np.array(oep_result.stop_reason or "")
    payload["oep_terminated_by"] = np.array(oep_result.terminated_by or "")

    if os.path.isfile(output_path):
        with np.load(output_path) as existing:
            for key in existing.files:
                if key not in payload:
                    payload[key] = existing[key]

    # Atomic write: tempfile + os.replace so an interrupted save_vxc_ref
    # cannot leave a half-written or empty .npz that future runs would
    # mis-load. Mirrors the run_scf_with_cache / run_ccsd_with_cache
    # pattern at external_refs.py:264-274.
    import tempfile
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    fd, tmp_name = tempfile.mkstemp(dir=out_dir, suffix=".npz")
    try:
        os.close(fd)
        np.savez(tmp_name, **payload)
        os.replace(tmp_name, output_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
