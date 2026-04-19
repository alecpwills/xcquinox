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


def _dm_to_rho_on_grid(mol, mf, dm):
    """Evaluate density on the DFT grid from a density matrix."""
    coords = mf.grids.coords
    ao = mf._numint.eval_ao(mol, coords)
    if dm.ndim == 2:
        rho = np.einsum("pi,ij,pj->p", ao, dm, ao)
    else:
        rho_a = np.einsum("pi,ij,pj->p", ao, dm[0], ao)
        rho_b = np.einsum("pi,ij,pj->p", ao, dm[1], ao)
        rho = rho_a + rho_b
    return rho


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


def _ks_from_vxc_matrix(mol, mf, vxc_matrix):
    """Run a KS-SCF with a fixed V_xc matrix replacing the XC potential.

    Uses PySCF's robust SCF driver (DIIS + damping) by overriding get_veff
    to return the fixed V_xc + J. This is essential: plain fixed-point
    iteration diverges for many molecules (e.g. H2O) without damping/DIIS.

    Returns (dm, kinetic_energy, j_matrix) where kinetic_energy = Tr(D T_kin)
    and j_matrix is the Coulomb matrix at the final density.
    """
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
    mf_fixed.kernel(dm0=mf.make_rdm1())

    dm_final = mf_fixed.make_rdm1()
    j_matrix = mf_fixed.get_j(mol, dm_final)
    t_matrix = mol.intor("int1e_kin")
    ts = np.einsum("ij,ij->", t_matrix, dm_final)
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
    rho_target = _dm_to_rho_on_grid(mol, mf, dm_target)

    # Precompute target-density integrals against the aux basis and h_core.
    rhotarget_integrals = np.einsum("gp,g->p", aux_on_grid, rho_target)
    h_core = mf.get_hcore()

    def objective_and_grad(b):
        """Wu-Yang functional in minimization form with consistent obj/grad.

        Wu-Yang variational principle: define

            F(b) = E_KS[v(b)] - int v(b) * rho_target dr

        where E_KS is the total KS energy at self-consistent D[v(b)]:

            E_KS[v] = Tr(D[v] * h_core) + 0.5 * Tr(D[v] * J[D[v]]) + Tr(D[v] * v)

        By the Hellmann-Feynman theorem (valid because D[v] minimizes E_KS at
        fixed v), dF/db_t = int g_t * (rho[v] - rho_target) dr = int g_t * Delta_rho.

        F is concave in b (second derivative involves the non-interacting
        density response chi, which is negative-semidefinite), so we MAXIMIZE F.
        Equivalently, L-BFGS-B minimizes G = -F + 0.5 * reg * |b|^2, whose
        gradient is:

            dG/db_t = -int g_t * Delta_rho dr + reg * b_t.

        This obj and grad are consistent derivatives of the same function, so
        L-BFGS-B line search (Wolfe conditions) works correctly.
        """
        vxc_matrix = np.einsum("t,tij->ij", b, three_center)
        dm_scf, _, j_matrix = _ks_from_vxc_matrix(mol, mf, vxc_matrix)
        rho_scf = _dm_to_rho_on_grid(mol, mf, dm_scf)
        delta_rho = rho_scf - rho_target

        # E_KS[v] = Tr(D h_core) + 0.5 Tr(D J) + Tr(D V_xc_matrix)
        e_ks = (
            float(np.einsum("ij,ij->", dm_scf, h_core))
            + 0.5 * float(np.einsum("ij,ij->", dm_scf, j_matrix))
            + float(np.einsum("ij,ij->", dm_scf, vxc_matrix))
        )
        # F(b) = E_KS - sum_t b_t * I_t^target  (to be maximized)
        F_val = e_ks - float(np.dot(b, rhotarget_integrals))
        # G = -F + 0.5 * reg * |b|^2   (to be minimized)
        obj = -F_val + 0.5 * regularization * float(np.sum(b ** 2))
        grad = -np.einsum("gp,g->p", aux_on_grid, delta_rho) + regularization * b
        return obj, grad

    b0 = np.zeros(n_aux)

    result = minimize(
        objective_and_grad,
        b0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-12},
    )

    b_final = result.x
    vxc_final = np.einsum("t,tij->ij", b_final, three_center)
    dm_final, _, _ = _ks_from_vxc_matrix(mol, mf, vxc_final)
    rho_final = _dm_to_rho_on_grid(mol, mf, dm_final)
    final_error = float(np.sqrt(np.sum(weights * (rho_target - rho_final) ** 2)))
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
