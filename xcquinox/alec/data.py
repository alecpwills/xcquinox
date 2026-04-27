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
    "rho_ref_grid",
    "ref_density_method",
    "E_ref_literature",
    "vxc_ref",
})


def _load_external_data(
    path: str,
    *,
    dm_pbe_shape: tuple[int, ...],
    rho_pbe_shape: tuple[int, ...],
    vxc_pbe_shape: tuple[int, ...],
    mol_name: str,
) -> tuple[jnp.ndarray | None, jnp.ndarray | None, str | None, float | None, jnp.ndarray | None]:
    """Load and validate a MoleculeSpec.external_data_path .npz.

    The .npz may contain any subset of ``dm_target``, ``rho_ref_grid``,
    ``ref_density_method``, ``E_ref_literature``, ``vxc_ref``; unknown
    keys trigger ``ValueError``. Shape validation matches freshly computed
    PBE quantities so callers cannot silently mismatch densities/DMs/V_xc
    against the PBE grid or basis.
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

        rho_ref_grid = None
        ref_density_method = None
        if "rho_ref_grid" in present:
            rho_arr = np.asarray(npz["rho_ref_grid"])
            if tuple(rho_arr.shape) != tuple(rho_pbe_shape):
                raise ValueError(
                    f"external rho_ref_grid shape {tuple(rho_arr.shape)} "
                    f"does not match rho_grid shape {tuple(rho_pbe_shape)} "
                    f"for {mol_name!r}"
                )
            rho_ref_grid = jnp.array(rho_arr)
        if "ref_density_method" in present:
            method_arr = np.asarray(npz["ref_density_method"])
            ref_density_method = str(method_arr.item())

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

        vxc_ref = None
        if "vxc_ref" in present:
            vxc_arr = np.asarray(npz["vxc_ref"])
            if tuple(vxc_arr.shape) != tuple(vxc_pbe_shape):
                raise ValueError(
                    f"external vxc_ref shape {tuple(vxc_arr.shape)} does not "
                    f"match vxc_pbe shape {tuple(vxc_pbe_shape)} for "
                    f"{mol_name!r}"
                )
            vxc_ref = jnp.array(vxc_arr)

    return dm_target, rho_ref_grid, ref_density_method, E_ref_literature, vxc_ref


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
    rho_ref_grid: jnp.ndarray | None
    ref_density_method: str | None
    vxc_ref: jnp.ndarray | None
    rho_grid: jnp.ndarray
    sigma_grid: jnp.ndarray
    nabla_rho_grid: jnp.ndarray
    grid_weights: jnp.ndarray
    ao_grid: jnp.ndarray
    ao_grid_deriv: jnp.ndarray
    cusp_features: jnp.ndarray | None
    dm_features: jnp.ndarray | None
    eri: jnp.ndarray | None
    atom_composition: tuple[tuple[str, int], ...]
    mol_metadata: dict
    # Cached pyscfad.gto.Mole built once at precompute time so that hot-path
    # training (pyscfad backend, filter_jit'd) does not call Mole.build()
    # inside the traced region — Mole.build() invokes numpy.__array__ and
    # raises TracerArrayConversionError under jit. Always present; may be
    # None if pyscfad is unavailable or Mole construction failed.
    _pyscfad_mol: object | None


_PRECOMPUTE_CACHE: dict = {}
_PRECOMPUTE_CACHE_ENABLED: bool = True


def _precompute_cache_key(
    mol_spec: MoleculeSpec,
    required_keys: tuple[str, ...],
    descriptors: tuple[Descriptor, ...],
) -> tuple:
    # MoleculeSpec is a frozen dataclass and hashes by structural identity.
    # required_keys are sorted to canonicalize set-equivalence.
    # Descriptors are keyed by class name + n_features so different
    # parameterizations of the same descriptor type don't collide.
    desc_key = tuple(
        (type(d).__name__, getattr(d, "n_features", None)) for d in descriptors
    )
    return (mol_spec, tuple(sorted(required_keys)), desc_key)


def clear_precompute_cache() -> None:
    """Wipe the in-memory precompute cache. Tests use this to isolate runs."""
    _PRECOMPUTE_CACHE.clear()


def set_precompute_cache_enabled(enabled: bool) -> None:
    """Toggle the in-memory precompute cache (default: enabled).

    Disable when calling precompute on streaming / changing inputs where the
    same MoleculeSpec object is reused with mutated external_data on disk.
    """
    global _PRECOMPUTE_CACHE_ENABLED
    _PRECOMPUTE_CACHE_ENABLED = bool(enabled)


def precompute_fixed_density_data(
    mol_spec: MoleculeSpec,
    required_keys: tuple[str, ...] = (),
    descriptors: tuple[Descriptor, ...] = (),
) -> MoleculeData:
    """Run PBE SCF, extract grid data, return a MoleculeData dict.

    Baseline keys are always populated. Reference/descriptor keys are computed
    on-demand based on required_keys and descriptor.required_mol_keys.
    Unused keys are set to None (D-M4/C-M3 treedef-homogeneity).

    Results are memoized in a process-level dict keyed on
    ``(mol_spec, sorted(required_keys), descriptor_classes)``. The
    precompute is pure (PBE SCF on a frozen geometry), so caching is
    correctness-preserving and gives O(N_specs) speedup when the notebook
    sweep evaluates the same molecule under many trained models.
    Disable via :func:`set_precompute_cache_enabled` if external_data on
    disk changes between calls.
    """
    cache_key = None
    if _PRECOMPUTE_CACHE_ENABLED:
        try:
            cache_key = _precompute_cache_key(mol_spec, required_keys, descriptors)
        except TypeError:
            cache_key = None  # mol_spec or descriptors not hashable
        if cache_key is not None and cache_key in _PRECOMPUTE_CACHE:
            return _PRECOMPUTE_CACHE[cache_key]

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
    # Pin grid level when the spec requires it (e.g., external rho_ref_grid
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
    # For UKS, mf.get_j returns per-spin J[dm_s]; veff[s] = V_xc[s] + J_total.
    # For RKS, mf.get_j returns J[dm_total].
    veff = mf.get_veff(mol, dm_pbe)
    if np.asarray(j_matrix).ndim == 3:  # UKS
        j_total = np.asarray(j_matrix).sum(axis=0)  # (nao, nao)
        vxc_pbe = np.asarray(veff) - j_total[np.newaxis, ...]  # (2, nao, nao)
    else:  # RKS
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
    # Store nabla_rho as (n_grid, 3) so compute_vxc_nn can assemble the GGA
    # v_sigma term V_xc_ij += 2 * integral v_sigma nabla_rho . nabla(phi_i phi_j) dr.
    nabla_rho_pbe = np.stack([drho_x, drho_y, drho_z], axis=-1)

    # PBE XC energy and E_non_xc
    if dm_pbe.ndim == 3:  # UKS
        # Use pyscf's veff.exc which already has correct spin-resolved PBE evaluation.
        # The `veff` object was computed above (mf.get_veff(mol, dm_pbe)); reuse its .exc.
        E_xc_pbe = float(veff.exc)
    else:  # RKS
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

    eri = None
    if "eri" in all_needed:
        eri = jnp.array(mol.intor("int2e", aosym="s1"))

    # External reference data (dm_target / rho_ref_grid / E_ref_literature)
    # come from an optional .npz pointed to by mol_spec.external_data_path.
    # precompute only handles SCF-level quantities; CCSD/HF post-SCF
    # computations are the caller's responsibility and are injected through
    # this path so run_training / run_test pick them up automatically.
    dm_target = None
    rho_ref_grid = None
    ref_density_method = None
    E_ref_literature = None
    vxc_ref = None
    if mol_spec.external_data_path is not None:
        dm_target, rho_ref_grid, ref_density_method, E_ref_literature, vxc_ref = _load_external_data(
            mol_spec.external_data_path,
            dm_pbe_shape=tuple(np.asarray(dm_pbe).shape),
            rho_pbe_shape=tuple(np.asarray(rho_pbe).shape),
            vxc_pbe_shape=tuple(np.asarray(vxc_pbe).shape),
            mol_name=mol_spec.name,
        )

    # Cache pyscfad Mole for hot-path training (avoids Mole.build() inside
    # jit; see MoleculeData._pyscfad_mol docstring). pyscfad is optional,
    # so swallow any import/build failure and leave the slot as None — the
    # pyscfad backend's _build_pyscfad_mf will fall back to rebuilding.
    pyscfad_mol: object | None = None
    try:
        import pyscfad.gto as pyscfad_gto
        mol_ad = pyscfad_gto.Mole()
        mol_ad.atom = mol_spec.atom
        mol_ad.basis = mol_spec.basis
        mol_ad.charge = mol_spec.charge
        mol_ad.spin = mol_spec.spin
        mol_ad.verbose = 0
        mol_ad.build()
        pyscfad_mol = mol_ad
    except Exception:
        pyscfad_mol = None

    result = MoleculeData(
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
        rho_ref_grid=rho_ref_grid,
        ref_density_method=ref_density_method,
        vxc_ref=vxc_ref,
        rho_grid=jnp.array(rho_pbe),
        sigma_grid=jnp.array(sigma_pbe),
        nabla_rho_grid=jnp.array(nabla_rho_pbe),
        grid_weights=jnp.array(weights),
        ao_grid=jnp.array(ao_no_deriv),
        ao_grid_deriv=jnp.array(ao),
        cusp_features=cusp_features,
        dm_features=dm_features,
        eri=eri,
        atom_composition=mol_spec.atom_composition,
        mol_metadata={
            "atom": mol_spec.atom,
            "basis": mol_spec.basis,
            "charge": mol_spec.charge,
            "spin": mol_spec.spin,
            "grid_level": mol_spec.grid_level,
        },
        _pyscfad_mol=pyscfad_mol,
    )
    if cache_key is not None and _PRECOMPUTE_CACHE_ENABLED:
        _PRECOMPUTE_CACHE[cache_key] = result
    return result
