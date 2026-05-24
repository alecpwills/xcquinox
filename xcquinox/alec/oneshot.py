"""xcquinox.alec.oneshot — fast pure-JAX one-shot prediction.

Implements THE SPEC §6.3:
  - fixed_density_total_energy (A/D1 losses, evaluation metrics)
  - oneshot_dm_prediction_fast (B/C/D2/D3 losses)
  - oneshot_grid_density (C/D3 grid losses, DensityRMSEMetric)
  - oneshot_total_energy (Harris diagnostic — research only)
  - compute_exc_nn, compute_vxc_nn (internal helpers)
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from xcquinox.alec.descriptors import assemble_descriptor_features

# R2-C N4 audit fix: shared numerical-regularization constants live in
# ``solver`` (single source of truth across all SCF backends). Pre-fix
# duplicated copies in oneshot.py and solver_manual.py could silently
# diverge. Re-exported here for backwards compatibility with
# ``from xcquinox.alec.oneshot import DEGENERACY_REG`` callers.
from xcquinox.alec.solver import (
    DEGENERACY_REG,
    SYM_BREAK_SHIFT,
    _sym_break_diag,
)


@eqx.filter_jit
def compute_exc_nn(model, rho, sigma, features, grid_weights):
    """Integrate NN XC energy density: E_xc^NN = sum(weights * exc).

    model.eval_exc returns rho * epsilon_xc, so NO extra rho factor here.
    Returns a JAX scalar (jit/grad-safe).

    JIT-cached: keyed on (model architecture pytree structure, input shapes).
    Calling this with a different model instance of the same architecture
    skips re-tracing — critical for the eval sweep that loads 72
    checkpoints sharing two architectures.
    """
    exc = model.eval_exc(rho, sigma, features)
    return jnp.sum(exc * grid_weights)


def _exc_scalar_for_part(model, part):
    """Return the scalar energy-density callable for the requested ``part``.

    SOLV-01: the UKS V_xc must be built from the SPLIT energy density —
    exchange spin-scaled per Oliver & Perdew (PRA 20, 397 (1979)), but
    correlation evaluated on the TOTAL density (zeta=0; von Barth & Hedin,
    J. Phys. C 5, 1629 (1972); PW92, PRB 45, 13244 (1992)). Selecting which
    scalar to JVP lets the same V_xc assembler produce the exchange-only,
    correlation-only, or combined potential.

    ``part="xc"`` (default) reproduces the pre-SOLV-01 combined behavior so
    RKS callers are byte-identical.
    """
    if part == "xc":
        return lambda r, s, f: model.eval_exc_scalar(r, s, f)
    if part == "x":
        return lambda r, s, f: model.eval_ex_scalar(r, s, f)
    if part == "c":
        return lambda r, s, f: model.eval_ec_scalar(r, s, f)
    raise ValueError(
        f"compute_vxc_nn: part must be 'xc', 'x', or 'c'; got {part!r}."
    )


def compute_vxc_nn(
    model,
    rho,
    sigma,
    features,
    ao_grid,
    grid_weights,
    nabla_rho=None,
    ao_grad=None,
    lda_only=False,
    part="xc",
) -> jnp.ndarray:
    """Assemble the NN XC potential matrix V_xc, dispatching to the
    JIT-compiled core.

    ``part`` selects which scalar energy density is JVP'd (SOLV-01):
    ``"xc"`` (default, combined — byte-identical to pre-SOLV-01 and used by
    RKS), ``"x"`` (exchange-only ``eval_ex_scalar``), or ``"c"``
    (correlation-only ``eval_ec_scalar``). The UKS path uses "x" per spin
    (spin-scaled) and "c" once on the total density.

    ``AlecGGAModel`` is a GGA functional: its XC energy depends on
    ``sigma = |nabla rho|^2``, so a physically correct V_xc *must* include
    the GGA ``v_sigma`` term. PRE-07 audit fix: rather than silently
    dropping ``v_sigma`` (returning LDA-only V_xc, which is physically
    wrong for a GGA model), this function now *refuses* to do so unless the
    caller explicitly opts in.

    Contract
    --------
    * ``lda_only=False`` (default) + both ``nabla_rho`` and ``ao_grad``
      provided -> full GGA V_xc (V_rho + V_sigma).
    * ``lda_only=False`` + either GGA input missing -> ``ValueError``
      (the silent-LDA footgun is gone).
    * ``lda_only=True`` -> explicit, genuinely-LDA path: only ``V_rho`` is
      assembled and the GGA inputs are ignored. Use this only when you
      truly want the LDA-like ``v_rho`` contribution in isolation.
    """
    if lda_only:
        return _compute_vxc_nn_lda(
            model, rho, sigma, features, ao_grid, grid_weights, part)
    if nabla_rho is None or ao_grad is None:
        raise ValueError(
            "compute_vxc_nn: AlecGGAModel is a GGA functional, so a correct "
            "V_xc requires the GGA inputs nabla_rho and ao_grad. Both were "
            "not supplied (nabla_rho is "
            f"{'set' if nabla_rho is not None else 'None'}, ao_grad is "
            f"{'set' if ao_grad is not None else 'None'}). Refusing to "
            "silently return LDA-only V_xc (the v_sigma term would be "
            "dropped, which is physically wrong for a GGA model). Pass both "
            "GGA inputs, or set lda_only=True to explicitly request the "
            "LDA-only v_rho contribution."
        )
    return _compute_vxc_nn_gga(
        model, rho, sigma, features, ao_grid, grid_weights, nabla_rho, ao_grad,
        part,
    )


@eqx.filter_jit
def _compute_vxc_nn_lda(model, rho, sigma, features, ao_grid, grid_weights,
                        part="xc"):
    return _compute_vxc_nn_core(
        model, rho, sigma, features, ao_grid, grid_weights,
        nabla_rho=None, ao_grad=None, part=part,
    )


@eqx.filter_jit
def _compute_vxc_nn_gga(model, rho, sigma, features, ao_grid, grid_weights,
                        nabla_rho, ao_grad, part="xc"):
    return _compute_vxc_nn_core(
        model, rho, sigma, features, ao_grid, grid_weights,
        nabla_rho=nabla_rho, ao_grad=ao_grad, part=part,
    )


def _compute_vxc_nn_core(
    model,
    rho,
    sigma,
    features,
    ao_grid,
    grid_weights,
    nabla_rho=None,
    ao_grad=None,
    part="xc",
) -> jnp.ndarray:
    """Assemble NN XC potential matrix V_xc via per-point forward-mode jvp.

    For a GGA E_xc[rho, sigma = |nabla rho|^2], the V_xc matrix element is
        V_xc_ij = integral phi_i v_rho phi_j dr
                + 2 * integral v_sigma nabla_rho . nabla(phi_i phi_j) dr
    The factor of 2 comes from dE_xc/d(nabla_rho) = 2 v_sigma * nabla_rho
    (because sigma = |nabla rho|^2). Expanding nabla(phi_i phi_j) =
    phi_j nabla(phi_i) + phi_i nabla(phi_j) and defining
        A_ij = sum_g w_g v_sigma(g) [nabla_rho(g) . nabla(phi_i)(g)] phi_j(g)
    gives the symmetric form
        V_sigma = 2 * (A + A.T).

    Parameters
    ----------
    rho : (n_grid,)
    sigma : (n_grid,)
    features : (n_grid, n_features) or (n_grid, 0)
    ao_grid : (n_grid, n_ao)
    grid_weights : (n_grid,)
    nabla_rho : (n_grid, 3), optional. If ``None`` the v_sigma term is
        omitted (LDA-only path). The public ``compute_vxc_nn`` wrapper only
        reaches this path when the caller passes ``lda_only=True``.
    ao_grad : (3, n_grid, n_ao) or (4, n_grid, n_ao), optional. If 4 leading
        dims, interpreted as ``eval_ao(..., deriv=1)`` and ``ao_grad[1:4]``
        is used. If ``None`` the v_sigma term is omitted (LDA-only path).

    Returns
    -------
    V_xc : (n_ao, n_ao), symmetric.
    """
    # SOLV-01: select exchange-only / correlation-only / combined scalar.
    exc_single_point = _exc_scalar_for_part(model, part)

    # Sanitize JVP inputs at low-density / vanishing-gradient points.
    #
    # CODE-02 audit fix: the networks' reduced-gradient transform uses
    # sqrt(sigma), whose derivative d/dsigma sqrt(sigma) = 1/(2 sqrt(sigma))
    # diverges as sigma -> 0, and the downstream tanh(s)^2 JVP then evaluates
    # 0 * inf = NaN. The PRE-existing sanitizer masked only on rho > thr, so a
    # grid point with rho > thr AND sigma == 0 exactly (reachable on
    # symmetric / high-symmetry systems, and identically on a zero-occupation
    # spin channel) slipped through: its NaN v_sigma was NOT masked and spread
    # through V_sigma -> Fock -> energy/grad.
    #
    # Fix: the sanitize predicate (and the v_sigma mask) now ALSO require
    # sigma > _V_SIGMA_THRESHOLD. We use the standard "safe value under where,
    # then mask the output with where" double-where trick so that BOTH the
    # forward value and the reverse-mode (VJP/JVP) gradient are NaN-free: the
    # masked-out points feed safe (rho=1, sigma=1) inputs into the JVP, and
    # the JVP output is then forced to exactly 0 — contributing nothing to
    # V_sigma while keeping 1/(2 sqrt(sigma)) out of the tape entirely.
    #
    # _V_SIGMA_THRESHOLD is DENORMAL-LEVEL (1e-30), NOT 1e-10. This is critical
    # for energy<->potential consistency (verified 2026-05-23). v_sigma is NOT
    # singular as sigma->0: with the tanh(s)^2 gate the enhancement obeys
    # F-1 ~ s^2 so F'(s) ~ s ~ sqrt(sigma), which exactly CANCELS the
    # 1/(2 sqrt(sigma)) from d sqrt(sigma)/d sigma, leaving a FINITE v_sigma
    # limit. The NaN is purely the 0*inf artefact at sigma == 0 EXACTLY (and at
    # denormal underflow). An earlier 1e-10 threshold masked v_sigma over the
    # whole sigma <= 1e-10 RANGE, zeroing a finite, energy-significant
    # contribution: on an open-shell Li channel ~49% of points fall in that
    # range and the masked V_xc captured only ~52% of the true energy
    # derivative (FD energy<->potential residual 0.92 vs 2.3e-7 at 1e-30). At
    # 1e-30 only the genuinely-singular sigma==0 / denormal points are masked
    # (1/(2 sqrt(1e-30)) = 5e14 is finite; underflow risk is below ~1e-300), so
    # v_sigma stays finite everywhere AND the analytic V_xc remains the true
    # functional derivative of E_xc.
    _V_RHO_THRESHOLD = 1e-10
    _V_SIGMA_THRESHOLD = 1e-30
    rho_ok = rho > _V_RHO_THRESHOLD
    # v_rho only needs the rho guard; v_sigma needs BOTH guards because the
    # sqrt(sigma)-derivative divergence is what makes the sigma-tangent JVP
    # blow up.
    sigma_ok = sigma > _V_SIGMA_THRESHOLD
    safe_mask = rho_ok & sigma_ok
    safe_rho = jnp.where(safe_mask, rho, jnp.ones_like(rho))
    safe_sigma = jnp.where(safe_mask, sigma, jnp.ones_like(sigma))

    # Per-point JVPs: tangent on rho and then on sigma
    v_rho, v_sigma = jax.vmap(
        lambda r, s, f: (
            jax.jvp(
                exc_single_point,
                (r, s, f),
                (jnp.ones_like(r), jnp.zeros_like(s), jnp.zeros_like(f)),
            )[1],
            jax.jvp(
                exc_single_point,
                (r, s, f),
                (jnp.zeros_like(r), jnp.ones_like(s), jnp.zeros_like(f)),
            )[1],
        )
    )(safe_rho, safe_sigma, features)

    # Mask JVP outputs to zero at the masked-out points (physically negligible
    # contribution AND keeps gradients finite at rho/sigma = 0). v_rho is
    # masked on the rho guard alone (its tangent does not pass through the
    # sqrt(sigma) singularity); v_sigma is masked on BOTH guards.
    v_rho = jnp.where(rho_ok, v_rho, 0.0)
    v_sigma = jnp.where(safe_mask, v_sigma, 0.0)

    # LDA-like contribution: V_rho_ij = sum_g w_g v_rho(g) phi_i(g) phi_j(g).
    V_rho = jnp.einsum("g,gi,gj->ij", grid_weights * v_rho, ao_grid, ao_grid)

    if nabla_rho is None or ao_grad is None:
        # Explicit LDA-only path (reached only via lda_only=True at the
        # public wrapper). The GGA v_sigma term is intentionally omitted.
        return V_rho

    # Accept either the (4, n_grid, n_ao) eval_ao(deriv=1) layout or the
    # (3, n_grid, n_ao) "derivatives-only" layout. This keeps callers
    # flexible: run_manual_scf/_vxc_term can pass mol_data["ao_grid_deriv"]
    # directly without an extra slice step.
    ao_grad_xyz = ao_grad[1:4] if ao_grad.shape[0] == 4 else ao_grad

    # (nabla_rho . nabla phi_i)(g) contracting cartesian axis d.
    # nabla_rho: (n_grid, 3); ao_grad_xyz: (3, n_grid, n_ao) -> (n_grid, n_ao).
    nabla_rho_dot_ao_grad = jnp.einsum("gd,dgi->gi", nabla_rho, ao_grad_xyz)

    # A_ij = sum_g w_g v_sigma(g) [nabla_rho . nabla phi_i](g) phi_j(g).
    # v_sigma already masked above at rho < threshold, so tail contribution
    # is exactly zero.
    A_matrix = jnp.einsum(
        "g,gi,gj->ij",
        grid_weights * v_sigma,
        nabla_rho_dot_ao_grad,
        ao_grid,
    )
    # V_sigma from d/dD_ij integral v_sigma |nabla rho|^2 dr = 2 v_sigma nabla_rho . nabla(phi_i phi_j).
    V_sigma = 2.0 * (A_matrix + A_matrix.T)
    return V_rho + V_sigma


def split_exc_energy_uks(model, rho_a, rho_b, sigma_aa, sigma_bb,
                         sigma_tot, features, grid_weights):
    """Integrated UKS XC energy using the SOLV-01 split (exchange spin-scaled,
    correlation on the total density).

        E_xc = 1/2 sum_g w_g [eps_x(2 rho_a, 4 sigma_aa)
                              + eps_x(2 rho_b, 4 sigma_bb)]
             +     sum_g w_g  eps_c(rho_tot, sigma_tot)

    where eps_x = model.eval_ex, eps_c = model.eval_ec (the exact split of
    eval_exc with identical tail masking). Exchange spin-scaling: Oliver &
    Perdew, Phys. Rev. A 20, 397 (1979). Correlation on the TOTAL density
    (zeta=0): von Barth & Hedin, J. Phys. C 5, 1629 (1972); PW92, Phys. Rev.
    B 45, 13244 (1992). This is the energy whose functional derivative
    is the split V_xc built by ``_uks_spin_resolved_vxc`` / the manual solver
    (the FD-consistency test C guards this).

    LIMITATION (descriptor features) — P2-02: the EXACT exchange spin-scaling
    relation holds for an F_x that depends only on (rho, sigma). When descriptor
    features (cusp, DM-statistics) are active, the SAME molecular ``features``
    are passed into BOTH the (2*rho_a) and (2*rho_b) exchange evaluations below
    -- those features encode molecular/structural context that has no
    doubled-spin-density transform -- so for OPEN-SHELL systems the relation is
    an APPROXIMATION (the features are evaluated at the physical density, not at
    the fictitious doubled-spin density). The closed-shell reduction to RKS
    remains EXACT because rho_a = rho_b gives identical features in both terms.

    P2-03: when the cnet is spin-polarization-aware
    (``cnet.use_spin_polarization``), correlation is evaluated with the real
    zeta = (rho_a-rho_b)/rho_tot and the zeta-dependent PW92 baseline (Dick &
    Fernandez-Serra, PRB 104 L161109 (2021)); this is the energy whose per-spin
    functional derivative ``compute_vc_polarized_per_spin`` builds. Flag False
    keeps the zeta=0 total-density correlation. ``rho_tot = rho_a + rho_b`` is
    implied by ``sigma_tot``.
    """
    rho_tot = rho_a + rho_b
    ex_a = model.eval_ex(2.0 * rho_a, 4.0 * sigma_aa, features)
    ex_b = model.eval_ex(2.0 * rho_b, 4.0 * sigma_bb, features)
    if getattr(model.cnet, "use_spin_polarization", False):
        zeta = jnp.clip((rho_a - rho_b) / jnp.maximum(rho_tot, 1e-300),
                        -1.0, 1.0)
        ec = model.eval_ec(rho_tot, sigma_tot, features, zeta=zeta)
    else:
        ec = model.eval_ec(rho_tot, sigma_tot, features)
    E_x = 0.5 * jnp.sum(grid_weights * (ex_a + ex_b))
    E_c = jnp.sum(grid_weights * ec)
    return E_x + E_c


def fixed_density_total_energy(model, mol_data) -> float:
    """Total energy with NN XC on frozen PBE density. No Roothaan step.

    E_total = E_non_xc + E_xc^NN[rho_PBE]
    Used by A, D1 losses and all energy-based evaluation metrics.

    SOLV-01: the UKS branch uses the SPLIT XC energy (exchange spin-scaled
    per Oliver & Perdew PRA 20, 397 (1979); correlation on the total density
    at zeta=0 per von Barth & Hedin 1972 / PW92 1992) so that this energy is
    consistent with the split V_xc used by the SCF solvers. RKS is unchanged
    (combined eval_exc on the total density).
    """
    features = assemble_descriptor_features(model.descriptors, mol_data)
    if mol_data["is_unrestricted"]:
        dm_pbe = mol_data["dm_pbe"]  # (2, nao, nao)
        ao_grid = mol_data["ao_grid"]
        ao_xyz = mol_data["ao_grid_deriv"][1:4]
        grid_weights = mol_data["grid_weights"]
        rho_a = jnp.einsum("ij,gi,gj->g", dm_pbe[0], ao_grid, ao_grid)
        rho_b = jnp.einsum("ij,gi,gj->g", dm_pbe[1], ao_grid, ao_grid)
        nabla_rho_a = 2.0 * jnp.einsum("ij,dgi,gj->gd", dm_pbe[0], ao_xyz, ao_grid)
        nabla_rho_b = 2.0 * jnp.einsum("ij,dgi,gj->gd", dm_pbe[1], ao_xyz, ao_grid)
        sigma_aa = jnp.sum(nabla_rho_a * nabla_rho_a, axis=1)
        sigma_bb = jnp.sum(nabla_rho_b * nabla_rho_b, axis=1)
        nabla_rho_tot = nabla_rho_a + nabla_rho_b
        sigma_tot = jnp.sum(nabla_rho_tot * nabla_rho_tot, axis=1)
        exc_integrated = split_exc_energy_uks(
            model, rho_a, rho_b, sigma_aa, sigma_bb, sigma_tot,
            features, grid_weights,
        )
        return mol_data["E_non_xc"] + exc_integrated
    exc_integrated = compute_exc_nn(
        model,
        mol_data["rho_grid"],
        mol_data["sigma_grid"],
        features,
        mol_data["grid_weights"],
    )
    return mol_data["E_non_xc"] + exc_integrated


def compute_vc_polarized_per_spin(model, rho_a, rho_b, sigma_tot, features,
                                  ao_grid, grid_weights, nabla_rho_tot, ao_grad):
    """Per-spin correlation potential V_c^a, V_c^b for a spin-polarization-aware
    cnet (P2-03). E_c depends on rho_a/rho_b through BOTH rho_tot = rho_a+rho_b
    AND zeta = (rho_a-rho_b)/rho_tot, so V_c^s = dE_c/drho_s is NOT shared.

    The per-spin rho COEFFICIENTS are obtained EXACTLY by JVP'ing the
    correlation energy density through a helper that forms rho_tot and zeta
    INTERNALLY w.r.t. rho_a / rho_b — so jax performs the full (clip-aware)
    zeta chain rule and the result is byte-consistent with autodiff of the
    energy (verified to ~1e-10), avoiding a hand-coded d zeta/d rho.

    The SIGMA term is SHARED: sigma_tot = |nabla rho_a + nabla rho_b|^2 gives
    d sigma_tot/d(nabla rho_a) = d sigma_tot/d(nabla rho_b) = 2 nabla rho_tot,
    and zeta has no gradient dependence. Only the sigma tangent hits the
    sqrt(sigma)-derivative singularity, so it alone uses the denormal sigma
    guard (the rho_a/rho_b tangents are finite at the real sigma).
    """
    # eps_c density as a function of the SPIN densities (rho_tot + zeta formed
    # internally with the SAME clip/floor the UKS energy uses).
    def ec_spin(ra, rb, s, f):
        rt = ra + rb
        z = jnp.clip((ra - rb) / jnp.maximum(rt, 1e-300), -1.0, 1.0)
        return model.eval_ec_scalar(rt, s, f, zeta=z)

    _V_SIGMA_THRESHOLD = 1e-30
    sigma_ok = sigma_tot > _V_SIGMA_THRESHOLD
    safe_sigma = jnp.where(sigma_ok, sigma_tot, jnp.ones_like(sigma_tot))

    # Per-spin rho coefficients = d eps_c / d rho_{a,b} (real sigma; the rho
    # tangents do not hit the sqrt(sigma) singularity).
    coeff_a, coeff_b = jax.vmap(
        lambda ra, rb, s, f: (
            jax.jvp(ec_spin, (ra, rb, s, f),
                    (jnp.ones_like(ra), jnp.zeros_like(rb),
                     jnp.zeros_like(s), jnp.zeros_like(f)))[1],
            jax.jvp(ec_spin, (ra, rb, s, f),
                    (jnp.zeros_like(ra), jnp.ones_like(rb),
                     jnp.zeros_like(s), jnp.zeros_like(f)))[1],
        )
    )(rho_a, rho_b, sigma_tot, features)

    # Shared sigma coefficient = d eps_c / d sigma_tot (safe sigma guard).
    v_sigma = jax.vmap(
        lambda ra, rb, s, f: jax.jvp(
            ec_spin, (ra, rb, s, f),
            (jnp.zeros_like(ra), jnp.zeros_like(rb),
             jnp.ones_like(s), jnp.zeros_like(f)))[1]
    )(rho_a, rho_b, safe_sigma, features)
    v_sigma = jnp.where(sigma_ok, v_sigma, 0.0)

    V_rho_a = jnp.einsum("g,gi,gj->ij", grid_weights * coeff_a, ao_grid, ao_grid)
    V_rho_b = jnp.einsum("g,gi,gj->ij", grid_weights * coeff_b, ao_grid, ao_grid)

    ao_grad_xyz = ao_grad[1:4] if ao_grad.shape[0] == 4 else ao_grad
    ndphi = jnp.einsum("gd,dgi->gi", nabla_rho_tot, ao_grad_xyz)
    A = jnp.einsum("g,gi,gj->ij", grid_weights * v_sigma, ndphi, ao_grid)
    V_sigma = 2.0 * (A + A.T)   # shared by both spins (zeta has no grad dep.)
    return V_rho_a + V_sigma, V_rho_b + V_sigma


def _uks_spin_resolved_vxc(model, mol_data, features):
    """Build spin-resolved V_xc^NN_a, V_xc^NN_b for the SOLV-01 split energy.

    SOLV-01 physics. The XC energy is split into exchange + correlation:

      * EXCHANGE obeys the exact spin-scaling relation (Oliver & Perdew,
        Phys. Rev. A 20, 397 (1979)):
            E_x[n_a, n_b] = 1/2 (E_x[2 n_a] + E_x[2 n_b]).
        Its functional derivative w.r.t. the alpha DM is exactly what
        ``compute_vxc_nn(..., part="x")`` produces when called with
        (2 rho_a, 4 sigma_aa, nabla = 2 nabla_rho_a): the v_sigma factor of 2
        absorbs the 2*nabla_rho_a scaling and the remaining factor of 2 from
        4*sigma_aa. Beta is symmetric.

      * CORRELATION does NOT obey the exchange spin-scaling relation; it is
        spin-interpolated (von Barth & Hedin, J. Phys. C 5, 1629 (1972);
        PW92, Phys. Rev. B 45, 13244 (1992)). Two paths exist, gated on the
        cnet's ``use_spin_polarization`` flag:

        - flag FALSE (default / RKS-era checkpoints): the correlation baseline
          ``pw92c_unpolarized_scalar`` is zeta-independent, so correlation is
          evaluated ONCE on the TOTAL density (the zeta=0 approximation). Since
          E_c depends only on rho_tot, delta rho_tot/delta rho_a =
          delta rho_tot/delta rho_b = 1, so the SAME matrix V_c[rho_tot,
          sigma_tot] enters BOTH spin channels (the fast path).

        - flag TRUE (P2-03, Dick & Fernandez-Serra PRB 104 L161109 (2021)):
          correlation uses the zeta-dependent PW92 baseline
          ``pw92c_polarized_scalar`` plus a spin-polarization input feature, so
          E_c depends on rho_a/rho_b through BOTH rho_tot AND zeta. Then
          delta zeta/delta rho_a != delta zeta/delta rho_b and V_c is PER-SPIN;
          ``compute_vc_polarized_per_spin`` builds V_c^a, V_c^b exactly.

    Therefore (flag FALSE) V_xc^a = vx[2 rho_a, 4 sigma_aa; 2 nabla_rho_a] +
    vc[rho_tot] and V_xc^b = vx[2 rho_b, 4 sigma_bb; 2 nabla_rho_b] + vc[rho_tot],
    with vc computed exactly ONCE; (flag TRUE) vc is replaced by the per-spin
    vc_a, vc_b.

    LIMITATION (descriptor features) — P2-02: the exchange spin-scaling is EXACT
    only for a feature-free (rho, sigma) F_x. With descriptor features active the
    same molecular ``features`` feed both doubled-spin exchange terms, so the
    open-shell relation is an approximation (closed-shell -> RKS stays exact).
    See ``split_exc_energy_uks`` for the full discussion; the V_xc here is the
    exact functional derivative of that (approximate-for-open-shell) energy.

    """
    dm_pbe = mol_data["dm_pbe"]  # (2, nao, nao)
    ao_grid = mol_data["ao_grid"]
    ao_grid_deriv = mol_data["ao_grid_deriv"]
    grid_weights = mol_data["grid_weights"]

    # Per-spin rho and nabla_rho from spin-resolved PBE DM.
    ao_xyz = ao_grid_deriv[1:4]  # (3, n_grid, n_ao)
    rho_a = jnp.einsum("ij,gi,gj->g", dm_pbe[0], ao_grid, ao_grid)
    rho_b = jnp.einsum("ij,gi,gj->g", dm_pbe[1], ao_grid, ao_grid)
    nabla_rho_a = 2.0 * jnp.einsum("ij,dgi,gj->gd", dm_pbe[0], ao_xyz, ao_grid)
    nabla_rho_b = 2.0 * jnp.einsum("ij,dgi,gj->gd", dm_pbe[1], ao_xyz, ao_grid)
    sigma_aa = jnp.sum(nabla_rho_a * nabla_rho_a, axis=1)
    sigma_bb = jnp.sum(nabla_rho_b * nabla_rho_b, axis=1)

    # Total density for the correlation piece (zeta=0 treatment).
    rho_tot = rho_a + rho_b
    nabla_rho_tot = nabla_rho_a + nabla_rho_b
    sigma_tot = jnp.sum(nabla_rho_tot * nabla_rho_tot, axis=1)

    # Exchange: per-spin, spin-scaled (part="x").
    vx_a = compute_vxc_nn(
        model, 2.0 * rho_a, 4.0 * sigma_aa, features, ao_grid, grid_weights,
        nabla_rho=2.0 * nabla_rho_a, ao_grad=ao_grid_deriv, part="x",
    )
    vx_b = compute_vxc_nn(
        model, 2.0 * rho_b, 4.0 * sigma_bb, features, ao_grid, grid_weights,
        nabla_rho=2.0 * nabla_rho_b, ao_grad=ao_grid_deriv, part="x",
    )
    # Correlation. P2-03: a spin-polarization-aware cnet makes V_c PER-SPIN
    # (zeta = (rho_a-rho_b)/rho_tot couples the spins); otherwise V_c is the
    # zeta=0 total-density potential, shared by both spins (the fast path).
    if getattr(model.cnet, "use_spin_polarization", False):
        vc_a, vc_b = compute_vc_polarized_per_spin(
            model, rho_a, rho_b, sigma_tot, features, ao_grid, grid_weights,
            nabla_rho_tot, ao_grid_deriv,
        )
        return vx_a + vc_a, vx_b + vc_b
    vc = compute_vxc_nn(
        model, rho_tot, sigma_tot, features, ao_grid, grid_weights,
        nabla_rho=nabla_rho_tot, ao_grad=ao_grid_deriv, part="c",
    )
    return vx_a + vc, vx_b + vc


def oneshot_dm_prediction_fast(model, mol_data, solver_config=None) -> jnp.ndarray:
    """Fixed-J Roothaan one-shot DM prediction.

    Builds F_NN = h_core + J[D_PBE] + V_xc^NN(rho_PBE), solves the
    generalized eigenvalue problem F C = S C eps via Cholesky transform,
    and returns the predicted density matrix.

    Returns:
      RKS: shape (n_ao, n_ao)
      UKS: shape (2, n_ao, n_ao)
    """
    if solver_config is not None:
        from xcquinox.alec.solver import run_scf
        return run_scf(solver_config, model, mol_data).density_matrix
    features = assemble_descriptor_features(model.descriptors, mol_data)

    h_core = mol_data["h_core"]
    j_pbe = mol_data["j_matrix"]
    s_matrix = mol_data["s_matrix"]
    nao = s_matrix.shape[0]

    # Cholesky decomposition of regularized overlap
    overlap_reg = s_matrix + DEGENERACY_REG * jnp.eye(nao)
    L = jnp.linalg.cholesky(overlap_reg)
    L_inv = jnp.linalg.inv(L)

    if mol_data["is_unrestricted"]:
        nocc_a = mol_data["nocc_a"]
        nocc_b = mol_data["nocc_b"]

        # Spin-resolved V_xc^NN (spin-scaled approximation; see helper docstring).
        vxc_nn_a, vxc_nn_b = _uks_spin_resolved_vxc(model, mol_data, features)

        # UKS: j_pbe has shape (2, n_ao, n_ao); J_total = J[dm_a] + J[dm_b]
        # enters both spin Fock matrices identically (Coulomb is spin-blind).
        j_total = j_pbe[0] + j_pbe[1]
        fock_a = h_core + j_total + vxc_nn_a
        fock_b = h_core + j_total + vxc_nn_b

        # Transform to orthogonal basis. The uniform DEGENERACY_REG * I
        # shift alone does NOT resolve eigenvalue degeneracies (every
        # eigenvalue moves by the same amount), which breaks the eigh VJP
        # on linear-symmetry molecules. Adding a small non-uniform diag
        # (_sym_break_diag) resolves exact degeneracies so 1/(λ_i - λ_j)
        # in the reverse-mode derivative stays finite. See SYM_BREAK_SHIFT
        # block comment for full rationale.
        # R2-C M4 audit fix: dropped dead `+ DEGENERACY_REG * jnp.eye(nao)`
        # uniform shift (commutes through eigh; doesn't break degeneracies).
        # Only the non-uniform _sym_break_diag does work.
        _sb = jnp.diag(_sym_break_diag(nao, fock_a.dtype))
        fock_orth_a = L_inv @ fock_a @ L_inv.T + _sb
        fock_orth_b = L_inv @ fock_b @ L_inv.T + _sb

        # Eigendecomposition (JAX-native: preserves grad flow through the solver).
        _, mo_coeff_orth_a = jnp.linalg.eigh(fock_orth_a)
        _, mo_coeff_orth_b = jnp.linalg.eigh(fock_orth_b)

        # Back-transform
        mo_coeff_a = L_inv.T @ mo_coeff_orth_a
        mo_coeff_b = L_inv.T @ mo_coeff_orth_b

        # Density matrices (no factor of 2 for UKS).
        # R2-C N1 audit fix: occupation-mask form (C * occ) @ C.T matches
        # the RKS path's gradient stability under multi-cycle eigh on
        # degenerate-eigenvalue Fock matrices (e.g. linear-symmetry mols
        # C2H2 / HCN / C2H4). Pre-fix slice form C[:, :nocc] @ C[:, :nocc].T
        # produced 0*NaN=NaN through reverse-mode at exact p-orbital
        # degeneracies.
        occ_a = (jnp.arange(nao) < nocc_a).astype(mo_coeff_a.dtype)
        occ_b = (jnp.arange(nao) < nocc_b).astype(mo_coeff_b.dtype)
        dm_a = (mo_coeff_a * occ_a) @ mo_coeff_a.T
        dm_b = (mo_coeff_b * occ_b) @ mo_coeff_b.T
        dm_pred = jnp.stack([dm_a, dm_b])
    else:
        nocc = mol_data["nocc"]

        # RKS path: single spin-blind V_xc^NN evaluated on the total density.
        vxc_nn = compute_vxc_nn(
            model,
            mol_data["rho_grid"],
            mol_data["sigma_grid"],
            features,
            mol_data["ao_grid"],
            mol_data["grid_weights"],
            nabla_rho=mol_data.get("nabla_rho_grid"),
            ao_grad=mol_data.get("ao_grid_deriv"),
        )

        # RKS: j_pbe has shape (n_ao, n_ao)
        fock = h_core + j_pbe + vxc_nn

        # Transform to orthogonal basis. Only the non-uniform
        # _sym_break_diag perturbation does work — a uniform
        # DEGENERACY_REG * I shift commutes through eigh and leaves
        # eigenvalue gaps unchanged (M4 audit fix: dropped uniform
        # term as dead weight). See SYM_BREAK_SHIFT block comment for
        # full rationale on the non-uniform shift.
        fock_orth = (L_inv @ fock @ L_inv.T
                     + jnp.diag(_sym_break_diag(nao, fock.dtype)))

        # Eigendecomposition
        _, mo_coeff_orth = jnp.linalg.eigh(fock_orth)

        # Back-transform
        mo_coeff = L_inv.T @ mo_coeff_orth

        # Density matrix (factor of 2 for RKS double occupation).
        # Use occupation-mask form to match the UKS path's gradient
        # stability under multi-cycle eigh on degenerate-eigenvalue
        # Fock matrices (H2 audit fix).
        occ = (jnp.arange(nao) < nocc).astype(mo_coeff.dtype)
        dm_pred = 2.0 * (mo_coeff * occ) @ mo_coeff.T

    return dm_pred


def oneshot_grid_density(model, mol_data, solver_config=None) -> jnp.ndarray:
    """Run oneshot DM prediction, then compute grid density.

    Returns spin-summed density of shape (n_points,) for both RKS and UKS.
    """
    if solver_config is not None:
        from xcquinox.alec.solver import run_scf
        D_total = run_scf(solver_config, model, mol_data).density_matrix
        if mol_data["is_unrestricted"]:
            D_total = D_total[0] + D_total[1]
        ao = mol_data["ao_grid"]
        return jnp.einsum("ij,gi,gj->g", D_total, ao, ao)
    D_NN = oneshot_dm_prediction_fast(model, mol_data)
    ao = mol_data["ao_grid"]

    if mol_data["is_unrestricted"]:
        D_total = D_NN[0] + D_NN[1]
    else:
        D_total = D_NN

    return jnp.einsum("ij,gi,gj->g", D_total, ao, ao)


def oneshot_total_energy(model, mol_data) -> float:
    """Harris functional energy at the NN-predicted density.

    E_harris = Tr[D_NN * (h_core + J[D_PBE] + V_xc^PBE)]
             - 0.5 * Tr[D_PBE * J[D_PBE]]
             - (Tr[D_PBE * V_xc^PBE] - E_xc^PBE)
             + e_nuc

    Research diagnostic only — NOT used by any loss or metric.
    """
    D_NN = oneshot_dm_prediction_fast(model, mol_data)

    h_core = mol_data["h_core"]
    j_pbe = mol_data["j_matrix"]
    vxc_pbe = mol_data["vxc_pbe"]
    dm_pbe = mol_data["dm_pbe"]
    E_xc_pbe = mol_data["E_xc_pbe"]
    e_nuc = mol_data["e_nuc"]

    if mol_data["is_unrestricted"]:
        # UKS: sum traces over both spin channels
        # j_pbe: (2, n_ao, n_ao), dm_pbe: (2, n_ao, n_ao), vxc_pbe: (2, n_ao, n_ao)
        # D_NN: (2, n_ao, n_ao)
        j_total = j_pbe[0] + j_pbe[1]

        # Tr[D_NN * (h_core + J_total + vxc_pbe)] summed over spins
        F_pbe_a = h_core + j_total + vxc_pbe[0]
        F_pbe_b = h_core + j_total + vxc_pbe[1]
        term1 = jnp.trace(D_NN[0] @ F_pbe_a) + jnp.trace(D_NN[1] @ F_pbe_b)

        # Double-counting: 0.5 * Tr[D_PBE * J[D_PBE]]
        term2 = 0.5 * (jnp.trace(dm_pbe[0] @ j_total) + jnp.trace(dm_pbe[1] @ j_total))

        # XC double-counting: Tr[D_PBE * V_xc^PBE] - E_xc^PBE
        term3 = (jnp.trace(dm_pbe[0] @ vxc_pbe[0]) + jnp.trace(dm_pbe[1] @ vxc_pbe[1])) - E_xc_pbe
    else:
        # RKS
        F_pbe = h_core + j_pbe + vxc_pbe
        term1 = jnp.trace(D_NN @ F_pbe)
        term2 = 0.5 * jnp.trace(dm_pbe @ j_pbe)
        term3 = jnp.trace(dm_pbe @ vxc_pbe) - E_xc_pbe

    return term1 - term2 - term3 + e_nuc


def _nn_fx_local_uks(model, rho_alpha: jnp.ndarray,
                    rho_beta: jnp.ndarray,
                    s: jnp.ndarray) -> jnp.ndarray:
    """Evaluate model.xnet as UKS F_x on synthetic (rho_alpha, rho_beta, s) points.

    Step-6 PBE-anchor helper. AlecGGA_XNet.__call__ is a single-grid-point
    evaluator taking a 1-D input tensor ``[rho, sigma, *extras]`` and
    returning a scalar F_x. We therefore vmap over N sample points.

    Spin-scaled UKS approximation (matches ``_uks_spin_resolved_vxc`` at
    SCF time):

        F_x_UKS(ra, rb, s) = 0.5 * (F_x_RKS(2*ra, sigma_aa_eff)
                                   + F_x_RKS(2*rb, sigma_bb_eff))

    where ``sigma_sigma_eff = (1 +/- zeta)**2 * sigma_tot``,
    ``zeta = (ra-rb)/(ra+rb)``, and ``sigma_tot = (2*kF(rho_tot)*s*rho_tot^(4/3))^2``.
    This is the SAME per-spin effective sigma that
    ``_uks_spin_resolved_vxc`` feeds into ``compute_vxc_nn`` during SCF:
    nabla_rho_sigma = (1 +/- zeta)/2 * nabla_rho_tot spatially, so
    ``4 * sigma_sigma_sigma = (1 +/- zeta)**2 * sigma_tot`` — exactly
    ``sigma_sigma_eff`` above.

    Uses zero extras (no descriptor features). The anchor probes the bare
    functional form at synthetic (rho, s) points — no molecular grid
    visits them, so there is no physical descriptor value to feed in.
    """
    n_extra = model.xnet.n_extra_features
    kF_tot = (3.0 * jnp.pi ** 2) ** (1.0 / 3.0)

    rho_tot = rho_alpha + rho_beta
    sigma_tot = (
        2.0 * kF_tot * s
        * jnp.clip(rho_tot, 1e-30, None) ** (4.0 / 3.0)
    ) ** 2
    zeta = jnp.where(
        rho_tot > 0,
        (rho_alpha - rho_beta) / jnp.clip(rho_tot, 1e-30, None),
        0.0,
    )
    sigma_aa_eff = (1.0 + zeta) ** 2 * sigma_tot
    sigma_bb_eff = (1.0 - zeta) ** 2 * sigma_tot

    def _fx_one(rho_spin_doubled, sigma_spin_eff):
        extras = jnp.zeros(n_extra, dtype=rho_spin_doubled.dtype)
        inputs = jnp.concatenate([
            jnp.atleast_1d(rho_spin_doubled),
            jnp.atleast_1d(sigma_spin_eff),
            extras,
        ])
        return model.xnet(inputs)

    fx_a = jax.vmap(_fx_one)(2.0 * rho_alpha, sigma_aa_eff)
    fx_b = jax.vmap(_fx_one)(2.0 * rho_beta, sigma_bb_eff)
    return 0.5 * (fx_a + fx_b)
