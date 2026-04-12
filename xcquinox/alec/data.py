"""xcquinox.alec.data — MoleculeData TypedDict and precompute.

Implements THE SPEC §6.1 (MoleculeData), §6.2 (precompute_fixed_density_data).
"""
import os
from typing import TypedDict

import numpy as np
import jax.numpy as jnp

from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.descriptors import Descriptor


# Keys allowed in MoleculeSpec.external_data_path .npz files. Kept as a
# module-level constant so tests and documentation can share it.
_ALLOWED_EXTERNAL_KEYS = frozenset({
    "dm_target",
    "rho_ccsd_grid",
    "E_ref_literature",
})


def _load_external_data(
    path: str,
    *,
    dm_pbe_shape: tuple[int, ...],
    rho_pbe_shape: tuple[int, ...],
    mol_name: str,
) -> tuple[jnp.ndarray | None, jnp.ndarray | None, float | None]:
    """Load and validate a MoleculeSpec.external_data_path .npz.

    The .npz may contain any subset of ``dm_target``, ``rho_ccsd_grid``,
    ``E_ref_literature``; unknown keys trigger ``ValueError``. Shape
    validation matches freshly computed PBE quantities so callers cannot
    silently mismatch densities/DMs against the PBE grid or basis.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"MoleculeSpec.external_data_path does not exist for "
            f"{mol_name!r}: {path}"
        )

    with np.load(path) as npz:
        present = set(npz.files)
        unknown = present - _ALLOWED_EXTERNAL_KEYS
        if unknown:
            raise ValueError(
                f"external_data .npz for {mol_name!r} contains unknown "
                f"keys {sorted(unknown)}; allowed keys: "
                f"{sorted(_ALLOWED_EXTERNAL_KEYS)}"
            )

        dm_target = None
        if "dm_target" in present:
            dm_arr = np.asarray(npz["dm_target"])
            if tuple(dm_arr.shape) != tuple(dm_pbe_shape):
                raise ValueError(
                    f"external dm_target shape {tuple(dm_arr.shape)} does "
                    f"not match dm_pbe shape {tuple(dm_pbe_shape)} for "
                    f"{mol_name!r}"
                )
            dm_target = jnp.array(dm_arr)

        rho_ccsd_grid = None
        if "rho_ccsd_grid" in present:
            rho_arr = np.asarray(npz["rho_ccsd_grid"])
            if tuple(rho_arr.shape) != tuple(rho_pbe_shape):
                raise ValueError(
                    f"external rho_ccsd_grid shape {tuple(rho_arr.shape)} "
                    f"does not match rho_grid shape {tuple(rho_pbe_shape)} "
                    f"for {mol_name!r}"
                )
            rho_ccsd_grid = jnp.array(rho_arr)

        E_ref_literature = None
        if "E_ref_literature" in present:
            val = np.asarray(npz["E_ref_literature"])
            if val.ndim == 0 or (val.ndim == 1 and val.size == 1):
                E_ref_literature = float(val.reshape(()).item())
            else:
                raise ValueError(
                    f"external E_ref_literature for {mol_name!r} must be "
                    f"scalar, got shape {tuple(val.shape)}"
                )

    return dm_target, rho_ccsd_grid, E_ref_literature


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
    # Pin grid level when the spec requires it (e.g., external rho_ccsd_grid
    # was generated on a non-default grid). Setting .level must happen before
    # the first kernel call so .build() picks it up.
    if mol_spec.grid_level is not None:
        mf.grids.level = mol_spec.grid_level
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

    # External reference data (dm_target / rho_ccsd_grid / E_ref_literature)
    # come from an optional .npz pointed to by mol_spec.external_data_path.
    # precompute only handles SCF-level quantities; CCSD/HF post-SCF
    # computations are the caller's responsibility and are injected through
    # this path so run_training / run_test pick them up automatically.
    dm_target = None
    rho_ccsd_grid = None
    E_ref_literature = None
    if mol_spec.external_data_path is not None:
        dm_target, rho_ccsd_grid, E_ref_literature = _load_external_data(
            mol_spec.external_data_path,
            dm_pbe_shape=tuple(np.asarray(dm_pbe).shape),
            rho_pbe_shape=tuple(np.asarray(rho_pbe).shape),
            mol_name=mol_spec.name,
        )

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
        E_ref_literature=E_ref_literature,
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
