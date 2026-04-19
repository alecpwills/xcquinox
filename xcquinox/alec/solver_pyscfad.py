"""xcquinox.alec.solver_pyscfad — pyscfad-based SCF backend.

Wraps pyscfad's dft.RKS (or dft.UKS for spin>0) with an alec-specific
eval_xc callback built from AlecGGAModel.eval_exc_scalar. pyscfad is
imported lazily inside run_pyscfad_scf so that users of the manual
backend don't pay the import cost.
"""
import warnings

import jax
import jax.numpy as jnp

from xcquinox.alec.solver import (
    SolverConfig,
    SolverMode,
    FeaturePolicy,
    SCFResult,
    _oneshot_result,
    _contract_dm_to_grid,
    _reassemble_features,
)


def _rebuild_mol_from_mol_data(mol_data: dict):
    """Rebuild a pyscfad gto.Mole from the metadata stashed by precompute.

    Must use pyscfad.gto.Mole (not pyscf.gto.Mole) because
    pyscfad.dft.RKS / UKS require a pyscfad-wrapped molecule object.
    """
    import pyscfad.gto
    md = mol_data["mol_metadata"]
    mol = pyscfad.gto.Mole()
    mol.atom = md["atom"]
    mol.basis = md["basis"]
    mol.charge = md["charge"]
    mol.spin = md["spin"]
    mol.verbose = 0
    mol.build()
    return mol


def _build_pyscfad_mf(mol, mol_data: dict):
    """Instantiate the appropriate pyscfad mean-field object.

    UKS for spin-polarized molecules (is_unrestricted or mol.spin != 0),
    RKS otherwise. Callers should subsequently call mf.define_xc_(...) and
    set mf.max_cycle / mf.conv_tol. The grid level is pinned from
    mol_metadata when present so that pyscfad uses the same grid as the
    precompute step — otherwise features_frozen (assembled on the
    precompute grid) would mismatch pyscfad's per-point rho.

    For UKS, pyscfad's built-in ``initialize_grids`` skips the density-based
    grid pruning (``prune_small_rho_grids_``) because it gates on
    ``dm.ndim == 2``. That means pyscfad's UKS grid retains more points
    than pyscf's post-SCF grid, and the precomputed FROZEN features (which
    were assembled on pyscf's post-prune grid) would mismatch pyscfad's
    rho at eval_xc time. To fix this, for UKS we pre-build and prune
    pyscfad's grid using the total DM stashed in ``mol_data`` before
    ``mf.kernel`` runs. The RKS path is left untouched so pyscfad's usual
    initialization flow (including ``non0tab`` bookkeeping inside
    ``initialize_grids``) runs exactly as before.
    """
    import pyscfad.dft
    is_uks = bool(mol_data.get("is_unrestricted", False)) or int(getattr(mol, "spin", 0)) != 0
    if is_uks:
        mf = pyscfad.dft.UKS(mol)
    else:
        mf = pyscfad.dft.RKS(mol)
    md = mol_data.get("mol_metadata") or {}
    grid_level = md.get("grid_level")
    if grid_level is not None:
        mf.grids.level = int(grid_level)
    if is_uks:
        from pyscfad.dft.rks import prune_small_rho_grids_
        mf.grids.build(with_non0tab=True)
        if mf.small_rho_cutoff > 1e-20:
            dm_pbe = mol_data.get("dm_pbe")
            if dm_pbe is not None:
                import numpy as np
                dm_np = np.asarray(dm_pbe)
                dm_total = dm_np[0] + dm_np[1] if dm_np.ndim == 3 else dm_np
                mf.grids = prune_small_rho_grids_(mf, mol, dm_total, mf.grids)
    return mf


def _make_alec_eval_xc(model, descriptors, mol_data, policy):
    """Return a libxc-compatible eval_xc callback that uses alec's XC NN.

    Only FROZEN policy is supported in this task; REASSEMBLE support with
    a _current_dm_holder closure is added in Task 6.3. For FROZEN,
    features are captured at construction time and reused every cycle.

    The callback handles both RKS (spin=0) and UKS (spin=1) pyscf/pyscfad
    numint conventions:
      - spin=0, GGA: rho shape (4, n_grid); returns vrho (n_grid,), vsigma (n_grid,)
      - spin=1, GGA: rho shape (2, 4, n_grid); returns vrho (n_grid, 2) and
        vsigma (n_grid, 3) in (uu, ud, dd) ordering.

    UKS uses the spin-scaled RKS approximation (see alec.oneshot
    `_uks_spin_resolved_vxc`):
        E_xc^UKS[rho_a, rho_b]
            ~= 0.5 * (E_xc^RKS[2 rho_a, 4 sigma_aa] + E_xc^RKS[2 rho_b, 4 sigma_bb])
    so vrho^s = v_rho^RKS(2 rho_s, 4 sigma_ss) and vsigma_ss = 2 *
    v_sigma^RKS(2 rho_s, 4 sigma_ss); the ud cross-term is zero under
    this approximation.
    """
    from xcquinox.alec.descriptors import assemble_descriptor_features

    if policy != FeaturePolicy.FROZEN:
        raise NotImplementedError(
            "REASSEMBLE policy in pyscfad backend is added in Task 6.3"
        )

    features_frozen = assemble_descriptor_features(descriptors, mol_data)

    def eval_single(r, s, f):
        return model.eval_exc_scalar(r, s, f)

    def _eval_rks(rho0, sigma, features):
        """Return (exc_density, vrho, vsigma) where exc_density = rho * eps (NN output).

        Callers divide ``exc_density`` by rho + reg to obtain libxc's
        per-particle ``exc``; the UKS branch of the callback spin-scales
        before dividing.
        """
        exc_density = jax.vmap(eval_single)(rho0, sigma, features)
        drho_fn = lambda r, s, f: jax.grad(eval_single, argnums=0)(r, s, f)
        dsigma_fn = lambda r, s, f: jax.grad(eval_single, argnums=1)(r, s, f)
        vrho = jax.vmap(drho_fn)(rho0, sigma, features)
        vsigma = jax.vmap(dsigma_fn)(rho0, sigma, features)
        return exc_density, vrho, vsigma

    def eval_xc_alec_gga(xc_code, rho, spin=0, relativity=0, deriv=1, verbose=None):
        # Detect spin-polarized input: pyscf/libxc convention sends
        # rho as a 2-tuple or a (2, 4, n_grid) array for spin=1 GGA and a
        # (4, n_grid) array for spin=0. Use ``spin`` (passed by pyscfad's
        # numint ``nr_rks`` / ``nr_uks``) as the primary signal.
        if spin == 1 or (isinstance(rho, tuple) and len(rho) == 2):
            import numpy as _np
            rho_arr = jnp.asarray(_np.asarray(rho))
            # UKS: rho[0] = (den_a, dxa, dya, dza), rho[1] = beta.
            rho_a = rho_arr[0, 0]
            rho_b = rho_arr[1, 0]
            dxa, dya, dza = rho_arr[0, 1], rho_arr[0, 2], rho_arr[0, 3]
            dxb, dyb, dzb = rho_arr[1, 1], rho_arr[1, 2], rho_arr[1, 3]
            sigma_aa = dxa * dxa + dya * dya + dza * dza
            sigma_bb = dxb * dxb + dyb * dyb + dzb * dzb

            # Spin-scaled RKS evaluation for each channel.
            exc_a_density, vrho_a, vsigma_a = _eval_rks(
                2.0 * rho_a, 4.0 * sigma_aa, features_frozen,
            )
            exc_b_density, vrho_b, vsigma_b = _eval_rks(
                2.0 * rho_b, 4.0 * sigma_bb, features_frozen,
            )

            # libxc convention: E_xc = integral (rho_a + rho_b) * eps_uks(r) dr,
            # so eps_uks is the per-particle energy density returned here.
            # Our approximation: E_xc ≈ 0.5 * ∫ (exc_a_density + exc_b_density) dr
            # where exc_s_density ≡ model.eval_exc(2*rho_s, 4*sigma_ss, ...)
            # (NN output at the scaled inputs). Dividing by (rho_a + rho_b)
            # yields eps_uks.
            rho_tot = rho_a + rho_b
            exc = 0.5 * (exc_a_density + exc_b_density) / (rho_tot + 1e-18)

            # vrho: (n_grid, 2) in (u, d) order.
            vrho_stack = jnp.stack([vrho_a, vrho_b], axis=-1)
            # vsigma: (n_grid, 3) in (uu, ud, dd) order. The spin-scaled
            # approximation has zero ud cross-term.
            vsigma_ud = jnp.zeros_like(vsigma_a)
            vsigma_stack = jnp.stack(
                [2.0 * vsigma_a, vsigma_ud, 2.0 * vsigma_b], axis=-1,
            )
            vxc = (vrho_stack, vsigma_stack, None, None)
            return exc, vxc, None, None

        # RKS path (spin=0, GGA). rho shape: (4, n_grid).
        rho0 = jnp.asarray(rho[0])
        dx, dy, dz = jnp.asarray(rho[1]), jnp.asarray(rho[2]), jnp.asarray(rho[3])
        sigma = dx * dx + dy * dy + dz * dz
        exc_density, vrho, vsigma = _eval_rks(rho0, sigma, features_frozen)
        exc = exc_density / (rho0 + 1e-18)
        vxc = (vrho, vsigma, None, None)
        return exc, vxc, None, None

    return eval_xc_alec_gga


def run_pyscfad_scf(config: SolverConfig, model, mol_data: dict) -> SCFResult:
    from xcquinox.alec.descriptors import assemble_descriptor_features

    if config.mode == SolverMode.ONESHOT:
        return _oneshot_result(model, mol_data)

    import pyscfad.dft  # noqa: F401 — lazy import

    policy = config.effective_feature_policy
    descriptors = model.descriptors

    if policy == FeaturePolicy.REASSEMBLE:
        warnings.warn(
            "REASSEMBLE policy on pyscfad backend is not yet implemented; "
            "falling back to FROZEN features.",
            RuntimeWarning,
            stacklevel=2,
        )
        policy = FeaturePolicy.FROZEN

    eval_xc_callback = _make_alec_eval_xc(
        model=model,
        descriptors=descriptors,
        mol_data=mol_data,
        policy=policy,
    )

    mol = _rebuild_mol_from_mol_data(mol_data)
    mf = _build_pyscfad_mf(mol, mol_data)
    mf.define_xc_(eval_xc_callback, "GGA")
    mf.max_cycle = int(config.max_cycles)
    mf.conv_tol = float(config.conv_tol)

    if config.mode == SolverMode.FIXED_J:
        # pyscfad UKS get_veff calls ks.get_j(mol, dm_total_2d, hermi) — the
        # spin DMs are summed before the Coulomb build (Coulomb is spin-blind),
        # so get_j returns a 2D matrix. For RKS, j_matrix is already 2D. For
        # UKS, the precompute stored j_matrix as (2, nao, nao) (per-spin J),
        # and the total J that enters the Fock build is the sum over spins.
        j_pinned_raw = mol_data["j_matrix"]
        j_pinned_arr = jnp.asarray(j_pinned_raw)
        if j_pinned_arr.ndim == 3 and j_pinned_arr.shape[0] == 2:
            J_pinned = j_pinned_arr[0] + j_pinned_arr[1]
        else:
            J_pinned = j_pinned_arr

        def fixed_get_j(mol_=None, dm=None, hermi=1, **kwargs):
            return J_pinned

        mf.get_j = fixed_get_j

    mf.kernel(dm0=mol_data["dm_pbe"])

    D_final = jnp.asarray(mf.make_rdm1())
    E_final = jnp.asarray(mf.e_tot)
    cycles_run = jnp.int32(getattr(mf, "cycles", config.max_cycles))
    converged = jnp.bool_(bool(mf.converged))
    features_used = assemble_descriptor_features(descriptors, mol_data)

    return SCFResult(
        density_matrix=D_final,
        total_energy=E_final,
        cycles_run=cycles_run,
        converged=converged,
        features_used=features_used,
    )
