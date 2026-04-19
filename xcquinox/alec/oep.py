"""xcquinox.alec.oep — Wu-Yang OEP inversion for reference V_xc generation.

Offline utility: generates V_xc^ref matrices from high-level density matrices
(e.g., CCSD). Not part of the training loop — produces .npz files consumed by
MoleculeSpec.external_data_path.
"""
import os
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize

from xcquinox.alec.config import MoleculeSpec


class OEPResult(NamedTuple):
    vxc_matrix: np.ndarray
    converged: bool
    n_iter: int
    density_error: float


def _build_mol_and_mf(mol_spec: MoleculeSpec, basis: str | None = None):
    """Build PySCF molecule and run PBE SCF. Returns (mol, mf)."""
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
    mf.xc = "pbe"
    mf.kernel()
    return mol, mf


def _dm_to_rho_on_grid(mol, mf, dm, *, per_spin: bool = False):
    """Evaluate density on the DFT grid from a density matrix.

    If ``dm`` is 3D (UKS) and ``per_spin`` is True, returns (rho_a, rho_b);
    otherwise returns the total density (sum over spins for UKS).
    """
    coords = mf.grids.coords
    ao = mf._numint.eval_ao(mol, coords)
    dm_arr = np.asarray(dm)
    if dm_arr.ndim == 2:
        rho = np.einsum("pi,ij,pj->p", ao, dm_arr, ao)
        if per_spin:
            # For an RHF DM, split evenly between spin channels
            return 0.5 * rho, 0.5 * rho
        return rho
    rho_a = np.einsum("pi,ij,pj->p", ao, dm_arr[0], ao)
    rho_b = np.einsum("pi,ij,pj->p", ao, dm_arr[1], ao)
    if per_spin:
        return rho_a, rho_b
    return rho_a + rho_b


def _build_aux_basis_matrices(mol, mf, aux_basis: str):
    """Build auxiliary-basis overlap integrals for V_xc expansion.

    Returns (aux_mol, three_center, aux_on_grid) where three_center[t, i, j]
    is the 3-center integral <i|g_t|j> weighted by grid weights, and the V_xc
    matrix contribution from coefficient b_t is:
      V_xc += b_t * <i|g_t|j>.
    """
    from pyscf import gto as gto_mod
    aux_mol = gto_mod.M(
        atom=mol.atom, basis=aux_basis, charge=mol.charge,
        spin=mol.spin, verbose=0,
    )
    coords = mf.grids.coords
    weights = mf.grids.weights
    ao_aux = aux_mol.eval_gto("GTOval_sph", coords)
    ao_orb = mf._numint.eval_ao(mol, coords)
    n_aux = ao_aux.shape[1]
    nao = ao_orb.shape[1]
    three_center = np.zeros((n_aux, nao, nao))
    for t in range(n_aux):
        three_center[t] = np.einsum(
            "g,gi,gj,g->ij", ao_aux[:, t], ao_orb, ao_orb, weights,
        )
    aux_on_grid = ao_aux * weights[:, None]
    return aux_mol, three_center, aux_on_grid


def _ks_from_vxc_matrix(mol, mf, vxc_matrix, *, dm0=None):
    """Run a KS-SCF with a fixed V_xc matrix replacing the XC potential.

    Dispatches to an RHF or UHF SCF driver based on the shape of vxc_matrix
    (2D = RHF, 3D = UKS) and on mol.spin.

    RHF path returns (dm (nao, nao), kinetic, j_matrix (nao, nao)).
    UKS path returns (dm (2, nao, nao), kinetic, j_matrix (2, nao, nao)).
    Uses PySCF's robust SCF driver (DIIS + damping) by overriding get_veff
    to return the fixed V_xc + J. This is essential: plain fixed-point
    iteration diverges for many molecules (e.g. H2O) without damping/DIIS.

    Pass ``dm0`` to warm-start the SCF; this is essential for open-shell
    Wu-Yang inversion where nearby V_xc perturbations can land on different
    near-degenerate UHF solutions unless seeded consistently.
    """
    v = np.asarray(vxc_matrix)
    if v.ndim == 3 or mol.spin != 0:
        return _ks_from_vxc_matrix_uhf(mol, mf, vxc_matrix, dm0=dm0)
    return _ks_from_vxc_matrix_rhf(mol, mf, vxc_matrix, dm0=dm0)


def _ks_from_vxc_matrix_rhf(mol, mf, vxc_matrix, *, dm0=None):
    """RHF path: closed-shell, single-DM, J built on RHF object."""
    from pyscf import scf

    mf_fixed = scf.RHF(mol)
    mf_fixed.verbose = 0
    mf_fixed.max_cycle = 200
    mf_fixed.conv_tol = 1e-12

    def get_veff_fixed(mol_, dm_, *args, **kwargs):
        j_mat = mf_fixed.get_j(mol_, dm_)
        # Total effective potential (beyond h_core) = J + V_xc_matrix
        return j_mat + vxc_matrix

    mf_fixed.get_veff = get_veff_fixed
    if dm0 is None:
        dm0 = mf.make_rdm1()
    dm0 = np.asarray(dm0)
    # If caller passed a UKS-shaped DM into the RHF path, sum spins.
    if dm0.ndim == 3:
        dm0 = dm0.sum(axis=0)
    mf_fixed.kernel(dm0=dm0)

    dm_final = mf_fixed.make_rdm1()
    j_matrix = mf_fixed.get_j(mol, dm_final)
    t_matrix = mol.intor("int1e_kin")
    ts = float(np.einsum("ij,ij->", t_matrix, dm_final))
    return dm_final, ts, j_matrix


def _ks_from_vxc_matrix_uhf(mol, mf, vxc_matrix, *, dm0=None):
    """UKS path: spin-resolved Fock, (2, nao, nao) DM and J.

    vxc_matrix has shape (2, nao, nao). get_veff returns
    veff[s] = J_total + V_xc[s], because the Hartree potential couples
    both spins to the total density.
    """
    from pyscf import scf

    v = np.asarray(vxc_matrix)
    if v.ndim != 3 or v.shape[0] != 2:
        raise ValueError(
            f"_ks_from_vxc_matrix_uhf expects vxc_matrix shape (2, nao, nao), "
            f"got {v.shape}"
        )

    mf_fixed = scf.UHF(mol)
    mf_fixed.verbose = 0
    mf_fixed.max_cycle = 200
    mf_fixed.conv_tol = 1e-12

    def get_veff_fixed(mol_, dm_, *args, **kwargs):
        dm_arr = np.asarray(dm_)
        if dm_arr.ndim == 2:
            # Shouldn't happen for UHF, but guard it: split evenly.
            dm_arr = np.stack([0.5 * dm_arr, 0.5 * dm_arr], axis=0)
        j = mf_fixed.get_j(mol_, dm_arr)  # per-spin J
        j_total = j[0] + j[1]
        return np.stack(
            [j_total + vxc_matrix[0], j_total + vxc_matrix[1]], axis=0,
        )

    mf_fixed.get_veff = get_veff_fixed
    if dm0 is None:
        dm0 = mf.make_rdm1()
    dm0 = np.asarray(dm0)
    if dm0.ndim == 2:
        # Reference was closed-shell; split into alpha/beta
        dm0 = np.stack([0.5 * dm0, 0.5 * dm0], axis=0)
    mf_fixed.kernel(dm0=dm0)

    dm_final = mf_fixed.make_rdm1()  # (2, nao, nao)
    j_matrix = mf_fixed.get_j(mol, dm_final)  # (2, nao, nao)
    t_matrix = mol.intor("int1e_kin")
    ts = float(
        np.einsum("ij,ij->", t_matrix, dm_final[0])
        + np.einsum("ij,ij->", t_matrix, dm_final[1])
    )
    return dm_final, ts, j_matrix


def run_oep_inversion(
    mol_spec: MoleculeSpec,
    dm_target: np.ndarray,
    *,
    basis: str | None = None,
    aux_basis: str = "def2-svp-jkfit",
    max_iter: int = 200,
    conv_tol: float = 1e-6,
    regularization: float = 1e-4,
) -> OEPResult:
    """Wu-Yang OEP inversion: find V_xc such that KS(V_xc) reproduces dm_target.

    Minimizes the Wu-Yang functional W[v] via L-BFGS. The V_xc potential is
    expanded in the auxiliary basis: V_xc = sum_t b_t <i|g_t|j>.
    """
    mol, mf = _build_mol_and_mf(mol_spec, basis)
    _, three_center, aux_on_grid = _build_aux_basis_matrices(mol, mf, aux_basis)
    n_aux = three_center.shape[0]
    weights = mf.grids.weights
    h_core = mf.get_hcore()

    is_uks = (mol.spin != 0) or (np.asarray(dm_target).ndim == 3)

    if is_uks:
        # Target density per spin channel
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
        """Build V_xc matrix (RHF: (nao, nao); UKS: (2, nao, nao))."""
        if is_uks:
            b_a = b[:n_aux]
            b_b = b[n_aux:]
            vxc_a = np.einsum("t,tij->ij", b_a, three_center)
            vxc_b = np.einsum("t,tij->ij", b_b, three_center)
            return np.stack([vxc_a, vxc_b], axis=0)
        return np.einsum("t,tij->ij", b, three_center)

    # Warm-start cache: seed each SCF with the previous converged DM. Without
    # this, small b perturbations may land on different spin-broken minima
    # (UHF) or different near-degenerate SCF solutions, breaking the smooth
    # dependence of F(b) on b that the Hellmann-Feynman gradient relies on.
    scf_state = {"dm0": None}

    def objective_and_grad(b):
        """Wu-Yang functional in minimization form with consistent obj/grad.

        Wu-Yang variational principle (RHF): define

            F(b) = E_KS[v(b)] - int v(b) * rho_target dr

        where E_KS is the total KS energy at self-consistent D[v(b)]:

            E_KS[v] = Tr(D[v] * h_core) + 0.5 * Tr(D[v] * J[D[v]]) + Tr(D[v] * v)

        By the Hellmann-Feynman theorem (valid because D[v] minimizes E_KS at
        fixed v), dF/db_t = int g_t * (rho[v] - rho_target) dr = int g_t * Delta_rho.

        For UKS, the functional generalizes to

            F(b_a, b_b) = E_KS[v_a, v_b] - sum_s int v_s * rho_target_s dr
            E_KS = Tr(D_tot * h_core) + 0.5 Tr(D_tot * J_tot) + sum_s Tr(D_s * v_s)

        with per-spin gradient dF/db_s_t = int g_t * (rho_s[v] - rho_target_s).

        F is concave in b (second derivative involves the non-interacting
        density response chi, which is negative-semidefinite), so we MAXIMIZE F.
        Equivalently, L-BFGS-B minimizes G = -F + 0.5 * reg * |b|^2, whose
        gradient is:

            dG/db_t = -int g_t * Delta_rho dr + reg * b_t.

        This obj and grad are consistent derivatives of the same function, so
        L-BFGS-B line search (Wolfe conditions) works correctly.
        """
        vxc_matrix = _vxc_from_b(b)
        dm_scf, _, j_matrix = _ks_from_vxc_matrix(
            mol, mf, vxc_matrix, dm0=scf_state["dm0"],
        )
        scf_state["dm0"] = dm_scf

        if is_uks:
            rho_scf_a, rho_scf_b = _dm_to_rho_on_grid(
                mol, mf, dm_scf, per_spin=True,
            )
            delta_a = rho_scf_a - rho_target_a
            delta_b = rho_scf_b - rho_target_b
            # j_matrix is (2, nao, nao); J_total = j[0] + j[1]
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
            obj = -F_val + 0.5 * regularization * float(np.sum(b ** 2))
            grad_a = -np.einsum("gp,g->p", aux_on_grid, delta_a) + regularization * b_a
            grad_b = -np.einsum("gp,g->p", aux_on_grid, delta_b) + regularization * b_b
            grad = np.concatenate([grad_a, grad_b])
            return obj, grad

        rho_scf = _dm_to_rho_on_grid(mol, mf, dm_scf)
        delta_rho = rho_scf - rho_target_total
        # E_KS[v] = Tr(D h_core) + 0.5 Tr(D J) + Tr(D V_xc_matrix)
        e_ks = (
            float(np.einsum("ij,ij->", dm_scf, h_core))
            + 0.5 * float(np.einsum("ij,ij->", dm_scf, j_matrix))
            + float(np.einsum("ij,ij->", dm_scf, vxc_matrix))
        )
        F_val = e_ks - float(np.dot(b, rhotarget_integrals))
        obj = -F_val + 0.5 * regularization * float(np.sum(b ** 2))
        grad = -np.einsum("gp,g->p", aux_on_grid, delta_rho) + regularization * b
        return obj, grad

    b0 = np.zeros(2 * n_aux if is_uks else n_aux)

    result = minimize(
        objective_and_grad,
        b0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-12},
    )

    b_final = result.x
    vxc_final = _vxc_from_b(b_final)
    dm_final, _, _ = _ks_from_vxc_matrix(
        mol, mf, vxc_final, dm0=scf_state["dm0"],
    )
    rho_final = _dm_to_rho_on_grid(mol, mf, dm_final)
    final_error = float(
        np.sqrt(np.sum(weights * (rho_target_total - rho_final) ** 2))
    )
    n_iter = min(result.nit, max_iter)
    converged = final_error < conv_tol

    return OEPResult(
        vxc_matrix=vxc_final,
        converged=converged,
        n_iter=n_iter,
        density_error=final_error,
    )


def save_vxc_ref(
    oep_result: OEPResult,
    output_path: str,
    *,
    dm_target: np.ndarray | None = None,
    method: str = "CCSD",
) -> None:
    """Save OEP result as .npz compatible with _load_external_data.

    If the file already exists, merges new keys with existing ones.
    """
    payload = {"vxc_ref": oep_result.vxc_matrix}
    if dm_target is not None:
        payload["dm_target"] = dm_target
    if method:
        payload["ref_density_method"] = np.array(method)

    if os.path.isfile(output_path):
        with np.load(output_path) as existing:
            for key in existing.files:
                if key not in payload:
                    payload[key] = existing[key]

    np.savez(output_path, **payload)
