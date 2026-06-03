"""xcquinox.alec.solver_pyscfad: pyscfad-based SCF backend.

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
    precompute step, otherwise features_frozen (assembled on the
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

    When ``mol_data`` carries a cached ``_pyscfad_mol`` (built once at
    precompute time), that pre-built Mole is used in preference to the
    ``mol`` argument, this avoids ``Mole.build()`` inside any jit-traced
    hot path (it invokes ``numpy.__array__`` and raises
    ``TracerArrayConversionError`` under ``filter_jit``).
    """
    import pyscfad.dft
    cached = mol_data.get("_pyscfad_mol")
    if cached is not None:
        mol = cached
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


def _reassemble_features_on_grid(
    descriptors: tuple,
    dm: "jnp.ndarray",
    s_matrix: "jnp.ndarray",
    grid_coords: "jnp.ndarray",
    mol,
) -> "jnp.ndarray":
    """Compute descriptor features from (dm, S) on a specific grid.

    Used by the REASSEMBLE policy in the pyscfad backend: pyscfad's grid
    may not match the precompute grid (pyscfad applies its own
    small_rho_cutoff pruning), so cusp features must be recomputed on
    pyscfad's actual grid coords. DM-statistics features are grid-
    agnostic (tiled), so they are recomputed at ``len(grid_coords)``.

    For UKS, ``dm`` may have shape ``(2, nao, nao)``. We pass the
    spin-resolved DM unchanged to ``DMStatisticsDescriptor.compute_from_dm``
    so the underlying ``compute_dm_features`` picks its UKS branch
    (Pople-Nesbet 1954: D_sigma S D_sigma = D_sigma per spin). Summing
    alpha+beta into a 2-D total DM and routing UKS through the RKS
    idempotency branch would instead produce a non-zero,
    physically-meaningless idempotency_error.
    """
    from xcquinox.alec.descriptors import CuspDescriptor, DMStatisticsDescriptor
    from xcquinox.features import compute_cusp_descriptor

    dm_arr = jnp.asarray(dm)
    # Keep dm_arr's ndim intact; compute_dm_features dispatches on it.

    n_grid = int(grid_coords.shape[0])
    if not descriptors:
        return jnp.zeros((n_grid, 0))

    nuclear_coords = jnp.asarray(mol.atom_coords())
    nuclear_charges = jnp.asarray(mol.atom_charges())

    cols = []
    for d in descriptors:
        if isinstance(d, CuspDescriptor):
            # Honor the descriptor's log_transform so the pyscfad-backend eval
            # cusp matches what training (data.py) and the cusp-using archs
            # expect; the default raw form saturates near nuclei (feature skew).
            cols.append(compute_cusp_descriptor(
                grid_coords, nuclear_coords, nuclear_charges,
                log_transform=bool(getattr(d, "log_transform", False)),
            ))
        elif isinstance(d, DMStatisticsDescriptor):
            cols.append(d.compute_from_dm(
                dm=dm_arr, s_matrix=s_matrix, n_grid=n_grid,
            ))
        else:
            raise NotImplementedError(
                f"_reassemble_features_on_grid does not yet know how to "
                f"recompute {type(d).__name__}"
            )
    return jnp.concatenate(cols, axis=1)


def _make_alec_eval_xc(model, descriptors, mol_data, policy,
                       feature_holder=None):
    """Return a libxc-compatible eval_xc callback that uses alec's XC NN.

    For FROZEN policy, features are captured at construction time and
    reused every cycle.

    For REASSEMBLE policy, features are recomputed from the current DM
    each SCF cycle. The caller must pass a mutable ``feature_holder``
    dict that the outer SCF loop updates before each ``get_veff`` call.
    The holder carries:
      - ``"features_full"``: jnp.ndarray of shape (n_grid, n_features),
        the descriptor features on pyscfad's actual grid assembled from
        the current DM.
      - ``"offset"``: int, running offset into the full-grid feature
        array. Reset to 0 at the start of each ``get_veff`` call and
        advanced by ``block_size`` each time ``eval_xc`` is invoked.

    The callback handles both RKS (spin=0) and UKS (spin=1) pyscf/pyscfad
    numint conventions:
      - spin=0, GGA: rho shape (4, n_grid); returns vrho (n_grid,), vsigma (n_grid,)
      - spin=1, GGA: rho shape (2, 4, n_grid); returns vrho (n_grid, 2) and
        vsigma (n_grid, 3) in (uu, ud, dd) ordering.

    UKS uses the SOLV-01 split (see alec.oneshot `_uks_spin_resolved_vxc`):
        E_xc^UKS = 0.5 (E_x[2 rho_a, 4 sigma_aa] + E_x[2 rho_b, 4 sigma_bb])
                 +      E_c[rho_tot, sigma_tot]
    EXCHANGE obeys the exact spin-scaling relation (Oliver & Perdew, Phys.
    Rev. A 20, 397 (1979)); CORRELATION does NOT and is evaluated once on the
    TOTAL density (zeta=0), because the baseline ``pw92c_unpolarized_scalar``
    is spin-unpolarized (von Barth & Hedin, J. Phys. C 5, 1629 (1972); PW92,
    Phys. Rev. B 45, 13244 (1992)). The per-spin libxc derivatives are then
        vrho_s   = v_rho^x(2 rho_s, 4 sigma_ss) + v_rho^c(rho_tot, sigma_tot)
        vsigma_uu = 2 v_sigma^x(2 rho_a, 4 sigma_aa) + v_sigma^c(rho_tot)
        vsigma_dd = 2 v_sigma^x(2 rho_b, 4 sigma_bb) + v_sigma^c(rho_tot)
        vsigma_ud = 2 v_sigma^c(rho_tot)
    The non-zero ``ud`` term comes entirely from the total-density
    correlation (sigma_tot = sigma_uu + 2 sigma_ud + sigma_dd).

    When ``cnet.use_spin_polarization`` is set, correlation uses the
    zeta-dependent PW92 baseline and ``vrho_c`` becomes PER-SPIN
    (``vrho_c_a != vrho_c_b``); ``vsigma_c`` stays shared because zeta has no
    sigma dependence (Dick & Fernandez-Serra, PRB 104 L161109 (2021)). Flag
    False keeps the zeta=0 shared-correlation path byte-identical.
    """
    from xcquinox.alec.descriptors import assemble_descriptor_features

    features_frozen = assemble_descriptor_features(descriptors, mol_data)
    n_features = features_frozen.shape[1]

    def _features_for_block(block_size: int) -> jnp.ndarray:
        """Return features sized for a single block of ``block_size`` points.

        pyscfad's ``block_loop`` splits the grid into chunks (default
        ~224 points) under ``jax.grad`` / traced execution, and may keep
        the whole grid as one block under eager execution. The block
        loop iterates sequentially through the grid.

        When a ``feature_holder`` is supplied (descriptor-ful
        architecture on pyscfad's actual grid), we slice
        ``features_full`` at the current offset and advance the counter.
        The holder is refreshed per-cycle under REASSEMBLE and stays at
        the dm_pbe features under FROZEN.

        When no holder is supplied (empty descriptors or legacy
        FROZEN-on-precompute-grid), we fall back to the precompute
        features.
        """
        if n_features == 0:
            return jnp.zeros((block_size, 0), dtype=features_frozen.dtype)

        if feature_holder is not None:
            features_full = feature_holder["features_full"]
            offset = int(feature_holder["offset"])
            features_slice = features_full[offset:offset + block_size]
            # pyscfad's block_loop may emit non-uniform
            # block sizes, the last block of an unpadded grid is smaller
            # than NBLK, and `non0tab` pruning can skip blocks entirely
            # while still advancing block_loop's internal cursor. Both
            # cases produce ``features_slice.shape[0] != block_size``.
            #
            # When the slice is shorter than the requested block (last-
            # block tail / pruned overshoot), zero-pad the trailing rows.
            # Those grid points have zero weight in pyscfad's downstream
            # numint summation (the corresponding ``rho``/``weights``
            # arrays are zero), so padded feature rows contribute nothing
            # to the energy/Fock, the value of the padding is irrelevant
            # so long as the shape contract is satisfied.
            slice_n = features_slice.shape[0]
            if slice_n < block_size:
                pad_n = block_size - slice_n
                pad = jnp.zeros(
                    (pad_n, features_slice.shape[1]),
                    dtype=features_slice.dtype,
                )
                features_slice = jnp.concatenate([features_slice, pad], axis=0)
            elif slice_n > block_size:
                # Cannot happen from a Python slice; defensive only.
                raise ValueError(
                    "Feature slice oversized: offset="
                    f"{offset}, block_size={block_size}, slice="
                    f"{slice_n}, full grid={features_full.shape[0]}. "
                    "This indicates a bug in the slicing logic."
                )
            feature_holder["offset"] = offset + block_size
            return features_slice

        # Legacy path: use precompute features directly. Works only when
        # the block loop returns the whole grid as one block AND
        # pyscfad's grid matches the precompute grid.
        if block_size == features_frozen.shape[0]:
            return features_frozen
        raise ValueError(
            "pyscfad backend with FROZEN features requires block_loop to "
            "return the full grid as one block, but got block_size="
            f"{block_size} != full grid {features_frozen.shape[0]}. This "
            "happens under jax.grad/jit tracing with descriptor-ful "
            "architectures; use REASSEMBLE policy to resolve."
        )

    def eval_single(r, s, f):
        return model.eval_exc_scalar(r, s, f)

    # SOLV-01 split scalar energy densities for the UKS callback.
    def eval_single_x(r, s, f):
        return model.eval_ex_scalar(r, s, f)

    def eval_single_c(r, s, f):
        return model.eval_ec_scalar(r, s, f)

    def _eval_part(fn, rho0, sigma, features):
        """Return (e_density, de/drho, de/dsigma) for scalar energy-density
        ``fn`` evaluated batched over the grid. Used for RKS (fn=eval_single)
        and for the SOLV-01 split exchange / correlation pieces."""
        e_density = jax.vmap(fn)(rho0, sigma, features)
        drho_fn = lambda r, s, f: jax.grad(fn, argnums=0)(r, s, f)
        dsigma_fn = lambda r, s, f: jax.grad(fn, argnums=1)(r, s, f)
        vrho = jax.vmap(drho_fn)(rho0, sigma, features)
        vsigma = jax.vmap(dsigma_fn)(rho0, sigma, features)
        return e_density, vrho, vsigma

    def _eval_rks(rho0, sigma, features):
        """Return (exc_density, vrho, vsigma) where exc_density = rho * eps (NN output).

        Callers divide ``exc_density`` by rho + reg to obtain libxc's
        per-particle ``exc``; the UKS branch of the callback spin-scales
        before dividing.
        """
        return _eval_part(eval_single, rho0, sigma, features)

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
            sigma_ab = dxa * dxb + dya * dyb + dza * dzb
            # total-density gradient invariant for the correlation
            # piece. sigma_tot = |nabla rho_tot|^2 = sigma_aa + 2 sigma_ab + sigma_bb.
            sigma_tot = sigma_aa + 2.0 * sigma_ab + sigma_bb

            # SOLV-01 split. EXCHANGE obeys the exact spin-scaling relation
            # (Oliver & Perdew, Phys. Rev. A 20, 397 (1979)):
            #   E_x = 0.5 (E_x[2 rho_a, 4 sigma_aa] + E_x[2 rho_b, 4 sigma_bb]),
            # evaluated per spin. CORRELATION does NOT, it is evaluated ONCE
            # on the TOTAL density (zeta=0) on the default fast path, because
            # the baseline pw92c_unpolarized_scalar is spin-unpolarized (von
            # Barth & Hedin, J. Phys. C 5, 1629 (1972); PW92, Phys. Rev. B 45,
            # 13244 (1992)). When cnet.use_spin_polarization is set,
            # correlation instead uses the zeta-dependent PW92 baseline and a
            # per-spin vrho_c (Dick & Fernandez-Serra, PRB 104 L161109 (2021)).
            #
            # Use a block-sized features slice since pyscfad chunks the grid
            # under jax.grad / jit tracing.
            features_blk = _features_for_block(int(rho_a.shape[0]))
            rho_tot = rho_a + rho_b
            # Exchange: per-spin, at the spin-scaled (2 rho_s, 4 sigma_ss).
            ex_a_density, vrho_x_a, vsigma_x_a = _eval_part(
                eval_single_x, 2.0 * rho_a, 4.0 * sigma_aa, features_blk,
            )
            ex_b_density, vrho_x_b, vsigma_x_b = _eval_part(
                eval_single_x, 2.0 * rho_b, 4.0 * sigma_bb, features_blk,
            )
            # Correlation. When the cnet is spin-polarization-aware,
            # eps_c depends on rho_a/rho_b through BOTH rho_tot AND
            # zeta = (rho_a-rho_b)/rho_tot (Dick & Fernandez-Serra, PRB 104
            # L161109 (2021)), so vrho_c is PER-SPIN. zeta has no sigma
            # dependence, so vsigma_c stays the single total-density
            # derivative. Flag False keeps the shared zeta=0 fast path.
            if getattr(model.cnet, "use_spin_polarization", False):
                def ec_spin_scalar(ra, rb, s, f):
                    rt = ra + rb
                    z = jnp.clip((ra - rb) / jnp.maximum(rt, 1e-300),
                                 -1.0, 1.0)
                    return model.eval_ec_scalar(rt, s, f, zeta=z)
                ec_density = jax.vmap(ec_spin_scalar)(
                    rho_a, rho_b, sigma_tot, features_blk)
                vrho_c_a = jax.vmap(
                    lambda ra, rb, s, f: jax.grad(ec_spin_scalar, 0)(ra, rb, s, f)
                )(rho_a, rho_b, sigma_tot, features_blk)
                vrho_c_b = jax.vmap(
                    lambda ra, rb, s, f: jax.grad(ec_spin_scalar, 1)(ra, rb, s, f)
                )(rho_a, rho_b, sigma_tot, features_blk)
                vsigma_c = jax.vmap(
                    lambda ra, rb, s, f: jax.grad(ec_spin_scalar, 2)(ra, rb, s, f)
                )(rho_a, rho_b, sigma_tot, features_blk)
            else:
                # zeta=0 fast path: correlation once on the total density,
                # the SAME vrho_c for both spins.
                ec_density, vrho_c, vsigma_c = _eval_part(
                    eval_single_c, rho_tot, sigma_tot, features_blk,
                )
                vrho_c_a = vrho_c
                vrho_c_b = vrho_c

            # libxc convention: E_xc = integral (rho_a + rho_b) * eps_uks(r) dr,
            # so eps_uks is the per-particle energy density returned here.
            # SOLV-01 split energy density:
            #   E_density = 0.5 (ex_a_density + ex_b_density) + ec_density.
            xc_density = 0.5 * (ex_a_density + ex_b_density) + ec_density
            # 1/(rho_tot + 1e-18) gives O(1/eps^2) JVP at
            # tail points (rho ≈ 0). Use jnp.where with a higher floor
            # that masks tail contributions to 0 instead of letting the
            # autodiff propagate amplified noise.
            _RHO_EPS = 1e-12
            rho_safe = jnp.maximum(rho_tot, _RHO_EPS)
            exc = jnp.where(
                rho_tot > _RHO_EPS,
                xc_density / rho_safe,
                0.0,
            )

            # vrho_s = d E_density / d rho_s.
            #   Exchange: d/drho_a [0.5 ex(2 rho_a)] = 0.5 * 2 * vrho_x_a = vrho_x_a.
            #   Correlation: vrho_c_a/vrho_c_b = d ec/d rho_{a,b}, IDENTICAL
            #     for both spins on the zeta=0 fast path, PER-SPIN when the
            #     polarized (zeta-dependent) correlation is active.
            vrho_a = vrho_x_a + vrho_c_a
            vrho_b = vrho_x_b + vrho_c_b
            # vrho: (n_grid, 2) in (u, d) order.
            vrho_stack = jnp.stack([vrho_a, vrho_b], axis=-1)

            # vsigma in (uu, ud, dd) order.
            #   Exchange: d/dsigma_aa [0.5 ex(4 sigma_aa)] = 0.5 * 4 * vsigma_x_a
            #     = 2 vsigma_x_a (uu); zero exchange ud cross-term.
            #   Correlation: ec depends on sigma_tot = sigma_aa + 2 sigma_ab
            #     + sigma_bb, so d ec/d sigma_uu = vsigma_c, d ec/d sigma_ud =
            #     2 vsigma_c, d ec/d sigma_dd = vsigma_c.
            vsigma_uu = 2.0 * vsigma_x_a + vsigma_c
            vsigma_ud = 2.0 * vsigma_c
            vsigma_dd = 2.0 * vsigma_x_b + vsigma_c
            vsigma_stack = jnp.stack([vsigma_uu, vsigma_ud, vsigma_dd], axis=-1)
            vxc = (vrho_stack, vsigma_stack, None, None)
            return exc, vxc, None, None

        # RKS path (spin=0, GGA). rho shape: (4, n_grid).
        rho0 = jnp.asarray(rho[0])
        dx, dy, dz = jnp.asarray(rho[1]), jnp.asarray(rho[2]), jnp.asarray(rho[3])
        sigma = dx * dx + dy * dy + dz * dz
        # Pyscfad splits the grid into blocks under jax.grad / jit
        # tracing, so size the features slice to the current block.
        features_blk = _features_for_block(int(rho0.shape[0]))
        exc_density, vrho, vsigma = _eval_rks(rho0, sigma, features_blk)
        # Same low-rho JVP guard as the UKS path above.
        _RHO_EPS = 1e-12
        rho_safe = jnp.maximum(rho0, _RHO_EPS)
        exc = jnp.where(
            rho0 > _RHO_EPS,
            exc_density / rho_safe,
            0.0,
        )
        vxc = (vrho, vsigma, None, None)
        return exc, vxc, None, None

    return eval_xc_alec_gga


def _cpu_device_context():
    """Return a ``jax.default_device`` context manager pinned to the first
    CPU device, or a no-op context when no CPU device is available.

    Motivation: pyscfad's ``eigh_gen_p`` custom primitive has no CUDA
    kernel (as of pyscfad 0.x), so ``jax.grad`` through pyscfad on GPU
    raises ``UNIMPLEMENTED: No registered implementation for custom
    call to cusolver_sygvd_ffi``. Wrapping the pyscfad subgraph in a
    CPU default-device context routes all XLA lowerings to CPU, which
    has a working kernel. Forward-only pyscfad calls that don't invoke
    ``eigh_gen_p`` are unaffected but get pinned to CPU for consistency.
    """
    import contextlib

    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        return contextlib.nullcontext()
    return jax.default_device(cpu_devices[0])


def run_pyscfad_scf(config: SolverConfig, model, mol_data: dict) -> SCFResult:
    # pyscfad's SCF driver depends on CONCRETE numpy arrays
    # for h_core / S / J construction (it goes through libcint via pyscf
    # backends that do not accept JAX tracers). Calling this from inside
    # @jit / @eqx.filter_jit produces a confusing TracerArrayConversion
    # error deep in pyscfad. Detect tracers early and raise a clear
    # message explaining the constraint.
    import jax
    # Scan ALL likely-traced keys, not just the first
    # present one: stopping after the first hit would miss
    # tracers in dm_pbe/j_matrix when rho_grid (concrete) preceded them.
    candidate_keys = (
        "dm_pbe", "j_matrix", "rho_grid", "ao_grid", "s_matrix", "h_core",
    )
    if any(
        key in mol_data and isinstance(mol_data[key], jax.core.Tracer)
        for key in candidate_keys
    ):
        raise RuntimeError(
            "run_pyscfad_scf cannot be called from inside @jit / "
            "@eqx.filter_jit: pyscfad's SCF driver requires concrete "
            "numpy/jnp arrays for libcint integral construction. Wrap "
            "the caller without JIT, or use SolverMode.ONESHOT (which is "
            "fully traceable)."
        )
    # ONESHOT doesn't enter pyscfad at all, skip the CPU pin.
    if config.mode == SolverMode.ONESHOT:
        return _oneshot_result(model, mol_data)
    # Pin the whole pyscfad subgraph (including eval_xc_callback
    # construction, which captures jnp arrays) to CPU so that jax.grad
    # through the pyscfad-specific eigh_gen primitive works. See
    # `_cpu_device_context` for the rationale.
    with _cpu_device_context():
        return _run_pyscfad_scf_impl(config, model, mol_data)


def _run_pyscfad_scf_impl(config: SolverConfig, model, mol_data: dict) -> SCFResult:
    from xcquinox.alec.descriptors import assemble_descriptor_features

    import pyscfad.dft  # noqa: F401, lazy import

    policy = config.effective_feature_policy
    descriptors = model.descriptors

    # Prefer the Mole cached in mol_data by precompute (avoids Mole.build()
    # inside any jit-traced hot path). Fall back to rebuilding from metadata
    # when the cache is absent (e.g., older mol_data dicts).
    cached_mol = mol_data.get("_pyscfad_mol")
    if cached_mol is not None:
        mol = cached_mol
    else:
        warnings.warn(
            "mol_data does not carry a cached _pyscfad_mol; rebuilding "
            "pyscfad.gto.Mole inside run_pyscfad_scf. This will fail under "
            "@eqx.filter_jit. Re-run precompute_fixed_density_data to cache "
            "the Mole.",
            RuntimeWarning,
            stacklevel=2,
        )
        mol = _rebuild_mol_from_mol_data(mol_data)
    mf = _build_pyscfad_mf(mol, mol_data)

    # When descriptors are present, pyscfad's grid may differ from the
    # precompute grid (pyscfad applies its own small_rho_cutoff pruning),
    # so descriptor features must live on pyscfad's actual grid. We
    # install a ``feature_holder`` closure shared with the eval_xc
    # callback. For UKS, _build_pyscfad_mf already built + pruned
    # mf.grids. For RKS we eagerly call initialize_grids here so
    # mf.grids.coords is populated before we wrap get_veff.
    #
    # FROZEN: features are computed once from dm_pbe on pyscfad's grid
    #         and never updated.
    # REASSEMBLE: get_veff wrapper (below) refreshes features from the
    #         current DM on every cycle.
    feature_holder = None
    if descriptors:
        is_uks = bool(mol_data.get("is_unrestricted", False)) or int(getattr(mol, "spin", 0)) != 0
        if not is_uks:
            mf.initialize_grids(mol, mol_data["dm_pbe"])
        feature_holder = {
            "features_full": _reassemble_features_on_grid(
                descriptors=descriptors,
                dm=mol_data["dm_pbe"],
                s_matrix=jnp.asarray(mol_data["s_matrix"]),
                grid_coords=jnp.asarray(mf.grids.coords),
                mol=mol,
            ),
            "offset": 0,
        }

    eval_xc_callback = _make_alec_eval_xc(
        model=model,
        descriptors=descriptors,
        mol_data=mol_data,
        policy=policy,
        feature_holder=feature_holder,
    )

    mf.define_xc_(eval_xc_callback, "GGA")
    mf.max_cycle = int(config.max_cycles)
    mf.conv_tol = float(config.conv_tol)

    # Wrap get_veff so the block offset is reset to 0 before pyscfad's
    # numint enters block_loop (each get_veff call == one full pass over
    # the grid). For REASSEMBLE, additionally refresh features_full from
    # the current DM. For FROZEN with a holder, only the offset reset
    # runs; features_full stays at its initial (dm_pbe) value.
    if feature_holder is not None:
        original_get_veff = mf.get_veff
        _s_matrix = jnp.asarray(mol_data["s_matrix"])
        _grid_coords = jnp.asarray(mf.grids.coords)

        def _holder_get_veff(mol_=None, dm=None, *args, **kwargs):
            # Pass the caller-supplied ``mol_`` through
            # to ``_reassemble_features_on_grid`` and the original
            # ``get_veff`` rather than substituting the closed-over
            # ``mol``. Pyscfad's SCF driver passes the live ``mol``
            # explicitly; using the closure variable would silently
            # ignore any geometry/basis change pyscfad introduces (e.g.
            # mol updates inside its scan). Fall back to the closed-over
            # ``mol`` only when the caller passes ``None``.
            mol_eff = mol_ if mol_ is not None else mol
            if policy == FeaturePolicy.REASSEMBLE and dm is not None:
                feature_holder["features_full"] = _reassemble_features_on_grid(
                    descriptors=descriptors,
                    dm=dm,
                    s_matrix=_s_matrix,
                    grid_coords=_grid_coords,
                    mol=mol_eff,
                )
            feature_holder["offset"] = 0
            return original_get_veff(mol_eff, dm, *args, **kwargs)

        mf.get_veff = _holder_get_veff

    if config.mode == SolverMode.FIXED_J:
        # pyscfad UKS get_veff calls ks.get_j(mol, dm_total_2d, hermi), the
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

    # pyscfad's SCF kernel does not persist the actual iteration count on
    # the mean-field object (``mf.cycles`` is the input parameter that
    # pyscfad reads as an upper bound, not a tracker, it stays at its
    # initial value 0 after kernel()). We install a callback into pyscfad's
    # inner _scf loop to count iterations directly. The callback runs once
    # per cycle and sees the loop-local ``cycle`` index in its ``envs`` dict.
    cycle_counter = [0]
    energy_history: list[float] = []

    def _count_cycles_cb(envs):
        # ``cycle`` in pyscfad's _scf loop is 0-based; record the 1-based
        # count so that a successful single-iteration convergence reports 1.
        cycle_counter[0] = int(envs.get("cycle", cycle_counter[0] - 1)) + 1
        # Capture per-cycle total energy when pyscfad exposes it. Different
        # pyscfad/pyscf versions key this as ``e_tot`` or ``etot``; we accept
        # either and silently skip if neither is present (the metric falls
        # back to a degenerate trace in that case).
        e_step = envs.get("e_tot", envs.get("etot"))
        if e_step is None:
            mf_local = envs.get("mf")
            if mf_local is not None:
                e_step = getattr(mf_local, "e_tot", None)
        if e_step is not None:
            try:
                energy_history.append(float(e_step))
            except (TypeError, ValueError):
                pass

    mf.callback = _count_cycles_cb
    mf.kernel(dm0=mol_data["dm_pbe"])

    D_final = jnp.asarray(mf.make_rdm1())
    E_final = jnp.asarray(mf.e_tot)
    cycles_run = jnp.int32(cycle_counter[0])
    converged = jnp.bool_(bool(mf.converged))
    features_used = assemble_descriptor_features(descriptors, mol_data)
    # Pad the history to ``max_cycles`` with NaNs so callers can stack
    # traces from different SCFs into a fixed-shape array. The length of
    # ``energy_history`` corresponds to the actual cycles executed; trailing
    # NaNs mark cycles that never ran (early convergence).
    # SolverConfig always defines ``max_cycles``; a
    # ``getattr(config, "max_cycles", ...)`` default branch would be
    # unreachable. Use the field directly so a missing attribute fails
    # loudly instead of silently substituting ``len(energy_history)``.
    max_cyc = int(config.max_cycles)
    pad_len = max(max_cyc, len(energy_history))
    if energy_history:
        # No ``dtype=jnp.float64`` pin here. Under
        # the suite's ``jax_enable_x64=True`` default a pin changes
        # nothing, but under x32 it would force a silent dtype
        # promotion that breaks downstream metric reductions.
        trace_arr = jnp.full((pad_len,), jnp.nan)
        trace_arr = trace_arr.at[: len(energy_history)].set(jnp.asarray(energy_history))
    else:
        trace_arr = None

    return SCFResult(
        density_matrix=D_final,
        total_energy=E_final,
        cycles_run=cycles_run,
        converged=converged,
        features_used=features_used,
        energy_trace=trace_arr,
    )
