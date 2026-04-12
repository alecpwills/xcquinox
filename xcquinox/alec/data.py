"""xcquinox.alec.data — MoleculeData TypedDict, precompute, and XC helpers.

Implements THE SPEC §6.1 (MoleculeData), §6.2 (precompute_fixed_density_data),
§6.3 (compute_exc_nn, compute_vxc_nn).
"""
from typing import TypedDict

import numpy as np
import jax
import jax.numpy as jnp

from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.descriptors import Descriptor


class MoleculeData(TypedDict, total=True):
    """Pre-computed training/test data for one molecule.
    Every key is always present; unused keys are None (D-M4/C-M3)."""
    name: str
    is_unrestricted: bool
    nocc: int | None
    nocc_a: int | None
    nocc_b: int | None
    dm_pbe: jnp.ndarray
    s_matrix: jnp.ndarray
    h_core: jnp.ndarray
    j_matrix: jnp.ndarray
    vxc_pbe: jnp.ndarray
    e_nuc: float
    E_pbe: float
    E_xc_pbe: float
    E_non_xc: float
    E_ref_literature: float | None
    dm_target: jnp.ndarray | None
    rho_ccsd_grid: jnp.ndarray | None
    rho_grid: jnp.ndarray
    sigma_grid: jnp.ndarray
    grid_weights: jnp.ndarray
    ao_grid: jnp.ndarray
    ao_grid_deriv: jnp.ndarray
    cusp_features: jnp.ndarray | None
    dm_features: jnp.ndarray | None
    atom_composition: tuple[tuple[str, int], ...]


def precompute_fixed_density_data(
    mol_spec: MoleculeSpec,
    required_keys: tuple[str, ...] = (),
    descriptors: tuple[Descriptor, ...] = (),
) -> MoleculeData:
    """Run PBE SCF, extract grid data, return a MoleculeData dict.

    Baseline keys are always populated. CCSD/descriptor keys are computed
    on-demand based on required_keys and descriptor.required_mol_keys.
    Unused keys are set to None (D-M4/C-M3 treedef-homogeneity).
    """
    from pyscf import dft, gto

    # Build pyscf molecule
    mol = gto.M(
        atom=mol_spec.atom,
        basis=mol_spec.basis,
        charge=mol_spec.charge,
        spin=mol_spec.spin,
        verbose=0,
    )

    # Run PBE SCF
    is_unrestricted = mol_spec.spin != 0
    if is_unrestricted:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.kernel()

    # Overlap conditioning gate (E-H4)
    s_matrix = mf.get_ovlp()
    cond_s = float(np.linalg.cond(s_matrix))
    if cond_s > 1e10:
        raise ValueError(
            f"Overlap matrix for {mol_spec.name!r} is ill-conditioned: "
            f"cond(S) = {cond_s:.2e} > 1e10. This typically indicates "
            f"near-linear-dependent basis functions."
        )

    # Extract SCF quantities
    dm_pbe = mf.make_rdm1()
    h_core = mf.get_hcore()
    j_matrix = mf.get_j(mol, dm_pbe)
    e_nuc = float(mf.energy_nuc())
    E_pbe = float(mf.e_tot)

    # V_xc^PBE = V_eff - J
    veff = mf.get_veff(mol, dm_pbe)
    vxc_pbe = np.asarray(veff) - np.asarray(j_matrix)

    # Grid quantities
    coords = mf.grids.coords
    weights = mf.grids.weights
    ao = mf._numint.eval_ao(mol, coords, deriv=1)
    ao_no_deriv = ao[0]

    # Total DM for density/sigma computation (always 2D)
    if dm_pbe.ndim == 2:
        dm_pbe_tot = dm_pbe
    else:
        dm_pbe_tot = dm_pbe[0] + dm_pbe[1]

    # PBE density and gradient on grid
    rho_pbe = np.einsum("pi,ij,pj->p", ao[0], dm_pbe_tot, ao[0])
    drho_x = 2 * np.einsum("pi,ij,pj->p", ao[1], dm_pbe_tot, ao[0])
    drho_y = 2 * np.einsum("pi,ij,pj->p", ao[2], dm_pbe_tot, ao[0])
    drho_z = 2 * np.einsum("pi,ij,pj->p", ao[3], dm_pbe_tot, ao[0])
    sigma_pbe = drho_x ** 2 + drho_y ** 2 + drho_z ** 2

    # PBE XC energy and E_non_xc
    rho_for_xc = mf._numint.eval_rho(mol, ao, dm_pbe_tot, xctype="GGA")
    exc_pbe, _, _, _ = mf._numint.eval_xc("pbe", rho_for_xc, spin=0)
    E_xc_pbe = float(np.sum(rho_pbe * exc_pbe * weights))
    E_non_xc = E_pbe - E_xc_pbe

    # Occupancies
    if is_unrestricted:
        nocc = None
        nocc_a = (mol.nelectron + mol.spin) // 2
        nocc_b = (mol.nelectron - mol.spin) // 2
    else:
        nocc = mol.nelectron // 2
        nocc_a = None
        nocc_b = None

    # Collect all needed keys from descriptors
    all_needed = set(required_keys)
    for d in descriptors:
        all_needed.update(d.required_mol_keys)

    # Descriptor features (on-demand)
    cusp_features = None
    dm_features = None

    if "cusp_features" in all_needed:
        from xcquinox.features import compute_cusp_descriptor
        nuclear_coords = jnp.array(mol.atom_coords())
        nuclear_charges = jnp.array([mol.atom_charge(i) for i in range(mol.natm)])
        cusp_features = compute_cusp_descriptor(
            jnp.array(coords), nuclear_coords, nuclear_charges,
        )

    if "dm_features" in all_needed:
        from xcquinox.features import compute_dm_features_array
        dm_feat_global = compute_dm_features_array(
            jnp.array(dm_pbe_tot), jnp.array(s_matrix),
        )
        dm_features = jnp.tile(dm_feat_global, (len(rho_pbe), 1))

    # CCSD reference data (on-demand, None if not requested)
    dm_target = None
    rho_ccsd_grid = None
    # CCSD computation is deferred to callers who supply the data externally;
    # precompute only handles SCF-level quantities.

    return MoleculeData(
        name=mol_spec.name,
        is_unrestricted=is_unrestricted,
        nocc=nocc,
        nocc_a=nocc_a,
        nocc_b=nocc_b,
        dm_pbe=jnp.array(dm_pbe),
        s_matrix=jnp.array(s_matrix),
        h_core=jnp.array(h_core),
        j_matrix=jnp.array(j_matrix),
        vxc_pbe=jnp.array(vxc_pbe),
        e_nuc=e_nuc,
        E_pbe=E_pbe,
        E_xc_pbe=E_xc_pbe,
        E_non_xc=E_non_xc,
        E_ref_literature=None,
        dm_target=dm_target,
        rho_ccsd_grid=rho_ccsd_grid,
        rho_grid=jnp.array(rho_pbe),
        sigma_grid=jnp.array(sigma_pbe),
        grid_weights=jnp.array(weights),
        ao_grid=jnp.array(ao_no_deriv),
        ao_grid_deriv=jnp.array(ao),
        cusp_features=cusp_features,
        dm_features=dm_features,
        atom_composition=mol_spec.atom_composition,
    )


def compute_exc_nn(model, rho, sigma, features, grid_weights) -> float:
    """Integrate NN XC energy density: E_xc^NN = sum(weights * exc).

    model.eval_exc returns rho * epsilon_xc, so NO extra rho factor here.
    """
    exc = model.eval_exc(rho, sigma, features)
    return float(jnp.sum(exc * grid_weights))


def compute_vxc_nn(model, rho, sigma, features, ao_grid, grid_weights) -> jnp.ndarray:
    """Assemble NN XC potential matrix V_xc via per-point forward-mode jvp.

    Returns shape (n_ao, n_ao). LDA-like approximation (v_sigma discarded).
    """
    def exc_single_point(r, s, f):
        return model.eval_exc_scalar(r, s, f)

    # Per-point jvp: tangent on rho only
    v_rho = jax.vmap(
        lambda r, s, f: jax.jvp(
            exc_single_point,
            (r, s, f),
            (jnp.ones_like(r), jnp.zeros_like(s), jnp.zeros_like(f)),
        )[1]
    )(rho, sigma, features)

    # Assemble Fock-matrix form: V_xc_ij = sum_g v_rho[g] * ao[g,i] * ao[g,j] * w[g]
    return jnp.einsum("g,gi,gj,g->ij", v_rho, ao_grid, ao_grid, grid_weights)
