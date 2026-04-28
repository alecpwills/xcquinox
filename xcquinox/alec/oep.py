"""xcquinox.alec.oep — Wu-Yang OEP inversion for reference V_xc generation.

Offline utility: generates V_xc^ref matrices from high-level density
matrices (e.g., CCSD). Not part of the training loop — produces .npz
files consumed by ``MoleculeSpec.external_data_path``.

Implements the **displacement-form Wu-Yang OEP** of Wu & Yang
*J. Chem. Phys.* **118**, 2498 (2003) §II.B:

    V_xc(r) = V_xc^baseline(r) + Σ_t  b_t · g_t(r)

where ``V_xc^baseline`` is computed from a user-chosen XC functional
(LDA / PBE / BLYP / SCAN / B3LYP / …; any pyscf-compatible string) and
``g_t`` are auxiliary basis functions. At ``b = 0`` the KS equations
already produce a physically reasonable density (close to the
``baseline_xc`` answer) and the optimizer fits only the *correction*
δV_xc that bridges the baseline to the reference target. This is the
standard Wu-Yang formulation; the alternative ``V_xc = Σ b_t g_t`` form
(no baseline) starts from Hartree-only KS and is numerically much
harder.

Regularization is in **V-space** via ``b^T S_aux b`` (Heaton-Burgess et
al. *Phys. Rev. Lett.* **98**, 256401 (2007)), making the regularization
strength basis-independent — switching ``def2-svp-jkfit`` ↔
``def2-tzvp-jkfit`` no longer silently changes the meaning of
``regularization``.
"""
from __future__ import annotations

import os
from typing import Any, NamedTuple

import numpy as np
from scipy.optimize import minimize

from xcquinox.alec.config import MoleculeSpec


class OEPResult(NamedTuple):
    """Wu-Yang OEP inversion result.

    Fields
    ------
    n_iter
        L-BFGS-B iteration count clipped at ``max_iter`` (R3-D L3 audit
        note): ``min(scipy.result.nit, max_iter)``. Some scipy builds
        report ``nit == max_iter + 1`` when the optimizer exits at the
        iteration cap; the clip ensures ``n_iter`` matches the user-
        requested cap exactly.
    lbfgs_status
        Scipy's L-BFGS-B termination message. R3-D L6 audit: when the
        post-optimization final SCF fails (the SCF whose DM determines
        ``density_error``), ``" + final_scf_failed"`` is appended so a
        consumer can distinguish scipy failure from final-SCF failure
        without inspecting ``converged`` alone.
    """
    vxc_matrix: np.ndarray
    converged: bool
    n_iter: int
    density_error: float
    # New (D7 audit fix): provenance for downstream loaders/audits.
    baseline_xc: str | None
    aux_basis: str
    regularization: float
    n_electrons: float          # Tr(S · D_target) sanity check (D10 audit)
    lbfgs_status: str           # "success" / message from scipy result (D5)


def _build_mol_and_mf(mol_spec: MoleculeSpec, basis: str | None = None,
                      baseline_xc: str | None = "pbe"):
    """Build PySCF molecule and run baseline-XC SCF. Returns ``(mol, mf)``.

    ``baseline_xc`` is forwarded as ``mf.xc``. Any pyscf-compatible XC
    string works (case-insensitive): ``"lda"``, ``"pbe"``, ``"blyp"``,
    ``"scan"``, ``"b3lyp"``, ``"hf"``, etc. Pass ``None`` to set
    ``mf.xc = ""`` (Hartree-only baseline) — not recommended for
    Wu-Yang; only included so callers can opt out explicitly.
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
        # Hartree-only baseline; pyscf accepts empty xc string.
        mf.xc = ""
    else:
        mf.xc = str(baseline_xc)
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
      * ``S_aux[t, t'] = ∫ g_t(r) g_{t'}(r) dr`` — the auxiliary-basis
        overlap matrix used for V-space regularization (D2 audit fix:
        Heaton-Burgess PRL 98, 256401 (2007); the prior coefficient-
        space ‖b‖² regularization is basis-dependent and silently
        changes meaning when aux_basis is swapped).

    Uses ``GTOval_sph`` if ``mol.cart`` is False; ``GTOval_cart``
    otherwise (D12 audit fix: hardcoded ``GTOval_sph`` was inconsistent
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
    # Fused 4-tensor einsum — ~10-100x faster than Python loop (D8 fix).
    three_center = np.einsum(
        "gt,gi,gj,g->tij", ao_aux, ao_orb, ao_orb, weights, optimize=True,
    )
    # Symmetrize against AO-quadrature noise so V_xc = V_xc^baseline +
    # Σ b_t · three_center[t] is exactly Hermitian in (i, j) — analytically
    # three_center[t,i,j] == three_center[t,j,i] from φ_iφ_j = φ_jφ_i, but
    # finite-grid quadrature breaks symmetry at ε_machine·N_grid (R3-D L5).
    three_center = 0.5 * (three_center + three_center.transpose(0, 2, 1))
    aux_on_grid = ao_aux * weights[:, None]
    # Auxiliary-basis overlap matrix from grid quadrature.
    S_aux = np.einsum("gt,gu,g->tu", ao_aux, ao_aux, weights, optimize=True)
    # Symmetrize against quadrature noise.
    S_aux = 0.5 * (S_aux + S_aux.T)
    return aux_mol, three_center, aux_on_grid, S_aux


def _ks_from_vxc_matrix(mol, mf, vxc_matrix, *, dm0=None):
    """Run a KS-SCF with a fixed V_xc matrix replacing the XC potential.

    Dispatches to RHF or UHF based on ``vxc_matrix`` shape and
    ``mol.spin``. Returns ``(dm, kinetic, j_matrix, success)`` where
    ``success`` is False if the inner SCF raised; callers should
    increase the outer Wu-Yang objective on failure rather than
    silently using the input DM (D4 audit fix).
    """
    v = np.asarray(vxc_matrix)
    if v.ndim == 3 or mol.spin != 0:
        return _ks_from_vxc_matrix_uhf(mol, mf, vxc_matrix, dm0=dm0)
    return _ks_from_vxc_matrix_rhf(mol, mf, vxc_matrix, dm0=dm0)


def _ks_from_vxc_matrix_rhf(mol, mf, vxc_matrix, *, dm0=None):
    """RHF inner SCF with damped + DIIS-stabilized convergence.

    Returns ``(dm, ts, j_matrix, success)``. ``success=False`` indicates
    SCF blew up (LinAlgError, ValueError, or non-finite output); the
    outer Wu-Yang objective should treat this as +inf so L-BFGS-B backs
    off (D4 audit fix). Pre-fix code silently returned ``dm0`` and a
    finite ``ts``, leaving the objective and gradient inconsistent on
    failed line-search probes.
    """
    from pyscf import scf
    from numpy.linalg import LinAlgError as _NpLinAlgError
    from scipy.linalg import LinAlgError as _ScLinAlgError

    mf_fixed = scf.RHF(mol)
    mf_fixed.verbose = 0
    mf_fixed.max_cycle = 200
    mf_fixed.conv_tol = 1e-10
    # DIIS + damping for robustness against ill-conditioned line-search
    # probes from L-BFGS-B (Pulay DIIS — Pulay CPL 73, 393 (1980); the
    # diis_start_cycle delays activation until damped iterations settle
    # the wavefunction). D3 audit fix: prior comments contradicted code
    # by claiming "disable DIIS" while leaving DIIS on; comments now
    # match behavior.
    mf_fixed.diis_start_cycle = 5
    mf_fixed.diis_space = 4
    mf_fixed.damp = 0.1

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


def _ks_from_vxc_matrix_uhf(mol, mf, vxc_matrix, *, dm0=None):
    """UHF inner SCF; returns ``(dm, ts, j_matrix, success)``."""
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
    mf_fixed.verbose = 0
    mf_fixed.max_cycle = 200
    mf_fixed.conv_tol = 1e-10
    mf_fixed.diis_start_cycle = 5
    mf_fixed.diis_space = 4
    mf_fixed.damp = 0.1

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
    # R3-D L7: PySCF version-defensive normalization. UHF + 3-D DM in
    # PySCF ≥ 2.0 returns spin-resolved (2, nao, nao); older versions
    # may return spin-summed (nao, nao). Downstream callers index
    # j_matrix[0]/j_matrix[1] unconditionally — guarantee 3-D here.
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
    progress_callback=None,
) -> OEPResult:
    """Wu-Yang displacement-form OEP inversion.

    Finds a coefficient vector ``b`` such that the KS potential

        V_xc(r) = V_xc^baseline(r) + Σ_t b_t · g_t(r)

    yields a Kohn-Sham density that matches ``dm_target`` (e.g. a
    CCSD(T) reference DM). The displacement form (Wu & Yang JCP 118,
    2498 (2003) §II.B) lets the auxiliary basis fit only the
    *correction* between the baseline and reference; ``b = 0`` already
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

        **Hybrid functionals (e.g. ``"b3lyp"``):** the baseline V_xc
        matrix captures the *local* exchange-correlation portion at the
        baseline-XC SCF DM, but the non-local exact-exchange (HF-K) piece
        is frozen at the baseline DM rather than recomputed from the
        evolving inner-SCF DM. This means at b=0 the inner SCF does
        NOT exactly reproduce the hybrid baseline DM (the K piece
        becomes inconsistent with the new D). Hybrid baselines work in
        practice but do not enjoy the "b=0 = baseline answer" property
        that pure-DFT baselines do; expect slightly more L-BFGS-B
        iterations to compensate. R2-D NEW-M1 audit note.
    aux_basis : str
        Auxiliary basis for V_xc expansion. Default
        ``"def2-svp-jkfit"`` (matches step6 notebook). Larger bases
        (``def2-tzvp-jkfit``) give finer V_xc resolution at higher
        cost; with V-space regularization the meaning of
        ``regularization`` is now basis-independent (D2 audit fix).
    max_iter : int
        L-BFGS-B max iterations.
    conv_tol : float
        Density-error L2 tolerance for the ``converged`` flag.
    regularization : float
        Tikhonov regularization in V-space: ``+0.5 * lambda *
        b^T S_aux b``. Heaton-Burgess et al. PRL 98, 256401 (2007).
        Smooth in V_xc(r) magnitude regardless of which auxiliary
        basis is chosen. Pre-fix code used ``0.5 * lambda * |b|^2``
        which silently changed meaning when ``aux_basis`` was swapped.
    progress_callback : callable or None
        Optional ``fn(iter_int, density_error_float)`` invoked once per
        L-BFGS outer iteration.

    Returns
    -------
    OEPResult with provenance fields (D7 audit fix).
    """
    mol, mf = _build_mol_and_mf(mol_spec, basis, baseline_xc=baseline_xc)
    _, three_center, aux_on_grid, S_aux = _build_aux_basis_matrices(
        mol, mf, aux_basis,
    )
    n_aux = three_center.shape[0]
    weights = mf.grids.weights
    h_core = mf.get_hcore()

    # R2-D NEW-M2 audit fix: spin=0 mol with 3-D dm_target is incoherent.
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

    # D10 audit: shape + nelectron sanity check on the target DM. If
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

    # D11 audit fix: keep the most recent ACCEPTED iterate's DM, not the
    # last objective-evaluation's DM (which may correspond to a rejected
    # line-search trial).
    scf_state: dict[str, Any] = {"dm0_accepted": None, "dm0_last_eval": None}
    _progress_state = {"iter": 0, "density_error_l2": float("inf")}

    def objective_and_grad(b):
        vxc_matrix = _vxc_from_b(b)
        dm_scf, _, j_matrix, scf_success = _ks_from_vxc_matrix(
            mol, mf, vxc_matrix, dm0=scf_state["dm0_last_eval"],
        )
        scf_state["dm0_last_eval"] = dm_scf

        # D4 audit fix: on inner-SCF failure, return a large objective
        # and zero-magnitude (but non-NaN) gradient so L-BFGS-B's Wolfe
        # line search backs off rather than treating the failure as a
        # successful evaluation at a bogus point.
        if not scf_success:
            obj = 1e20
            grad = np.zeros_like(b)
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
            # D2 + D14 audit fix: V-space regularization. UKS uses a
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
            return obj, grad

        rho_scf = _dm_to_rho_on_grid(mol, mf, dm_scf)
        delta_rho = rho_scf - rho_target_total
        e_ks = (
            float(np.einsum("ij,ij->", dm_scf, h_core))
            + 0.5 * float(np.einsum("ij,ij->", dm_scf, j_matrix))
            + float(np.einsum("ij,ij->", dm_scf, vxc_matrix))
        )
        F_val = e_ks - float(np.dot(b, rhotarget_integrals))
        # D2 audit fix: V-space regularization.
        reg_term = 0.5 * regularization * float(b @ S_aux @ b)
        obj = -F_val + reg_term
        reg_grad = regularization * (S_aux @ b)
        grad = -np.einsum("gp,g->p", aux_on_grid, delta_rho) + reg_grad
        _progress_state["density_error_l2"] = float(
            np.sqrt(np.sum(weights * delta_rho ** 2))
        )
        return obj, grad

    def _scipy_iter_callback(_xk):
        # D11 audit fix: SCF DM at the accepted iterate is the one cached
        # by the most-recent ACCEPTED objective_and_grad call. Update
        # accepted-DM cache here (after L-BFGS-B accepts the step).
        scf_state["dm0_accepted"] = scf_state["dm0_last_eval"]
        _progress_state["iter"] += 1
        if progress_callback is not None:
            progress_callback(
                _progress_state["iter"],
                _progress_state["density_error_l2"],
            )

    b0 = np.zeros(2 * n_aux if is_uks else n_aux)

    result = minimize(
        objective_and_grad,
        b0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-12},
        callback=_scipy_iter_callback,
    )

    b_final = result.x
    vxc_final = _vxc_from_b(b_final)
    # Run the final SCF from the most recently ACCEPTED warm-start (D11),
    # not from a possibly-rejected trial DM.
    final_warm = (
        scf_state["dm0_accepted"]
        if scf_state["dm0_accepted"] is not None
        else scf_state["dm0_last_eval"]
    )
    dm_final, _, _, final_success = _ks_from_vxc_matrix(
        mol, mf, vxc_final, dm0=final_warm,
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
    # R3-D L3: clip scipy's reported nit at our requested max_iter so
    # n_iter never exceeds what the user asked for; documented in the
    # OEPResult.n_iter docstring above.
    n_iter = min(int(result.nit), max_iter)
    # D5 audit fix: converged requires BOTH density error AND L-BFGS-B
    # success status (not just the density-error tolerance).
    converged = bool(
        final_success
        and (final_error < conv_tol)
        and getattr(result, "success", False)
    )
    lbfgs_status = str(getattr(result, "message", "no message"))
    # R3-D L6: surface final-SCF failure in lbfgs_status so a consumer
    # reading converged=False can distinguish "scipy failed" (its message)
    # from "scipy succeeded but the post-optimization SCF blew up".
    if not final_success:
        lbfgs_status = lbfgs_status + " + final_scf_failed"

    return OEPResult(
        vxc_matrix=vxc_final,
        converged=converged,
        n_iter=n_iter,
        density_error=final_error,
        baseline_xc=baseline_xc,
        aux_basis=aux_basis,
        regularization=regularization,
        n_electrons=n_elec_target,
        lbfgs_status=lbfgs_status,
    )


def save_vxc_ref(
    oep_result: OEPResult,
    output_path: str,
    *,
    dm_target: np.ndarray | None = None,
    method: str = "CCSD",
) -> None:
    """Save OEP result as .npz compatible with ``_load_external_data``.

    D7 audit fix: provenance fields (``oep_baseline_xc``,
    ``oep_aux_basis``, ``oep_regularization``, ``oep_density_error``,
    ``oep_converged``, ``oep_lbfgs_status``, ``oep_n_electrons``) are
    written so downstream loaders can validate consistency. Pre-fix
    code wrote only ``vxc_ref`` and ``ref_density_method``, allowing a
    wrong-basis or wrong-baseline V_xc to load silently.

    If the file already exists, merges new keys with existing ones.
    """
    payload: dict[str, np.ndarray | str] = {"vxc_ref": oep_result.vxc_matrix}
    if dm_target is not None:
        payload["dm_target"] = dm_target
    if method:
        payload["ref_density_method"] = np.array(method)
    # Provenance — written even when None / empty so loaders can detect.
    payload["oep_baseline_xc"] = np.array(
        "" if oep_result.baseline_xc is None else oep_result.baseline_xc
    )
    payload["oep_aux_basis"] = np.array(oep_result.aux_basis)
    payload["oep_regularization"] = np.array(oep_result.regularization)
    payload["oep_density_error"] = np.array(oep_result.density_error)
    payload["oep_converged"] = np.array(bool(oep_result.converged))
    payload["oep_lbfgs_status"] = np.array(oep_result.lbfgs_status)
    payload["oep_n_electrons"] = np.array(oep_result.n_electrons)

    if os.path.isfile(output_path):
        with np.load(output_path) as existing:
            for key in existing.files:
                if key not in payload:
                    payload[key] = existing[key]

    np.savez(output_path, **payload)
