"""Tests for xcquinox.alec.metagga: the self-consistent meta-GGA tau / SCAN-alpha.

tau(r) = 1/2 sum_munu P_munu grad chi_mu . grad chi_nu is a LINEAR contraction of
the live DM against the AO gradients already computed on the grid (deriv=1) -- the
same self-consistent, differentiable pattern as the rung-3.5 occupancy. The
iso-orbital indicator alpha = (tau - tau_W)/tau_unif (SCAN 2015 / DFS Eq. 6) then
feeds the meta-GGA network. These tests pin tau against PySCF's own MGGA eval_rho,
alpha against the repo's existing (numpy) formula, and differentiability in the DM.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from pyscf import gto, dft

from xcquinox.alec.metagga import compute_tau_from_dm, compute_alpha


def _scf(atom, spin):
    mol = gto.M(atom=atom, basis="def2-svp", spin=spin, verbose=0)
    mf = dft.UKS(mol) if spin else dft.RKS(mol)
    mf.xc = "pbe"; mf.grids.level = 1
    mf.kernel()
    ao = mf._numint.eval_ao(mol, mf.grids.coords, deriv=2)  # (10, N, nao)
    return mol, mf, ao, mf.make_rdm1()


def test_tau_from_dm_matches_pyscf_rks():
    mol, mf, ao, dm = _scf("O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469", 0)
    tau_ref = mf._numint.eval_rho(mol, ao, dm, xctype="MGGA")[5]
    tau = np.asarray(compute_tau_from_dm(jnp.asarray(ao[1:4]), jnp.asarray(dm)))
    assert np.allclose(tau, tau_ref, atol=1e-9)


def test_tau_from_dm_matches_pyscf_uks_total():
    # OH doublet. compute_tau_from_dm sums the two spin slots of a 3-D density
    # matrix, so on the PHYSICAL matrix it returns the total kinetic-energy
    # density -- the iso-orbital ingredient of the total density. The same
    # summation on the symmetric doubled matrix diag(P_sigma, P_sigma) returns
    # 2 tau_sigma, the ingredient of the channel the exact exchange spin
    # scaling evaluates (test below).
    mol, mf, ao, dm = _scf("O 0 0 0; H 0 0 0.97", 1)
    tau_ref = mf._numint.eval_rho(mol, ao, dm[0] + dm[1], xctype="MGGA")[5]
    tau = np.asarray(compute_tau_from_dm(jnp.asarray(ao[1:4]), jnp.asarray(dm)))
    assert np.allclose(tau, tau_ref, atol=1e-9)


def test_tau_from_doubled_spin_dm_is_twice_the_channel_tau():
    """tau(diag(P_sigma, P_sigma)) = 2 tau_sigma -- the meta-GGA ingredient of
    the spin-unpolarized system the Oliver-Perdew relation refers to (Phys. Rev.
    A 20, 397 (1979))."""
    from xcquinox.alec.descriptors import doubled_spin_dm
    mol, mf, ao, dm = _scf("O 0 0 0; H 0 0 0.97", 1)
    for s in (0, 1):
        tau_ref = mf._numint.eval_rho(mol, ao, dm[s], xctype="MGGA")[5]
        # A purely relative criterion demands exact equality wherever the
        # reference vanishes, so the absence of a hard zero is asserted rather
        # than assumed: the smallest reference value 2 tau_sigma measured on
        # this grid is 2.6e-13 (alpha) and 4.3e-13 (beta). A grid or basis
        # change that produces an exact zero fails here instead of silently
        # tightening the comparison to bit equality.
        assert np.all(tau_ref > 0.0)
        tau_doubled = np.asarray(compute_tau_from_dm(
            jnp.asarray(ao[1:4]), doubled_spin_dm(jnp.asarray(dm), s)))
        # Purely relative: tau spans 2.6e-13 to 8.1e+03 Ha/bohr^3 on this grid,
        # so an absolute floor would leave the tail untested and a default
        # rtol of 1e-5 would admit ~8e-2 at the peak. The two evaluations
        # differ only by summation order and agree to 1.0e-15 relative.
        np.testing.assert_allclose(tau_doubled, 2.0 * tau_ref,
                                   rtol=1e-12, atol=0.0)


def test_alpha_matches_repo_scan_formula():
    """Against the numpy twin (hard clip): the two agree to the smoothing's
    own footprint ``width^2 / (4 |alpha_raw|)`` wherever the raw indicator is
    away from zero, and to ``width / 2`` at most anywhere."""
    from xcquinox.alec.subset_selection import compute_descriptor_triple
    from xcquinox.alec.metagga import _ALPHA_SMOOTHING_WIDTH as c
    rng = np.random.default_rng(0)
    rho = np.abs(rng.normal(1.0, 0.5, 64)) + 1e-3
    sigma = np.abs(rng.normal(0.5, 0.3, 64))
    tau = np.abs(rng.normal(0.4, 0.2, 64))
    ref = compute_descriptor_triple(rho, sigma, tau)["alpha"]
    got = np.asarray(compute_alpha(jnp.asarray(rho), jnp.asarray(sigma),
                                   jnp.asarray(tau)))
    raw = tau - sigma / (8 * rho)
    raw = raw / (0.3 * (3 * np.pi ** 2) ** (2 / 3) * rho ** (5 / 3))
    assert np.all(np.abs(got - ref) <= c ** 2 / (4 * np.abs(raw)) * (1 + 1e-6)
                  + 1e-15)
    assert np.all(np.abs(got - ref) <= 0.5 * c)
    # This draw stays away from alpha = 0 (smallest |raw| measured 7.7e-4), so
    # the agreement is 3.3e-8 or better and the comparison is not vacuous.
    assert np.min(np.abs(raw)) > 5e-4
    assert np.allclose(got, ref, atol=1e-7)


def test_alpha_uniform_gas_limit_is_one():
    # For the uniform electron gas sigma=0 (tau_W=0) and tau=tau_unif -> alpha=1
    # up to the smoothing's footprint width^2/4 = 2.5e-11 at alpha = 1.
    rho = jnp.array([1.0, 2.0, 0.5])
    tau_unif = (3.0 / 10.0) * (3.0 * jnp.pi**2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
    alpha = np.asarray(compute_alpha(rho, jnp.zeros_like(rho), tau_unif))
    assert np.allclose(alpha, 1.0, atol=1e-10)


def test_alpha_one_orbital_floor_is_half_the_width():
    """A one-orbital density has tau = tau_W exactly, so the raw indicator is
    zero and the smoothed one sits at width / 2 -- the same value at every
    density, since the width is a multiple of tau_unif (scale invariance)."""
    from xcquinox.alec.metagga import _ALPHA_SMOOTHING_WIDTH as c
    rho = jnp.array([1e-8, 1e-4, 0.1, 1.0, 10.0])
    sigma = jnp.array([1e-17, 1e-9, 0.05, 0.3, 40.0])
    tau_w = sigma / (8.0 * rho)
    alpha = np.asarray(compute_alpha(rho, sigma, tau_w))
    np.testing.assert_allclose(alpha, 0.5 * c, rtol=1e-12, atol=0.0)


def test_smooth_positive_part_properties():
    """The construction behind the lower bound: strictly positive, C-infinity,
    max(x, 0) up to width^2/(4|x|) away from zero, slope 1/2 at zero, exact
    odd part ``p(x) - p(-x) = x`` (so a central difference across zero
    reproduces the derivative), an exact inverse, and degree-one homogeneity
    in (x, width), which is what makes the indicator's smoothing invariant
    under the uniform density scaling alpha is invariant under."""
    from xcquinox.alec.metagga import (
        invert_smooth_positive_part, smooth_positive_part)
    c = 1e-5
    x = jnp.array([-1e3, -1.0, -1e-3, -3e-5, -1e-5, -1e-7, 0.0, 1e-7, 1e-5,
                   3e-5, 1e-3, 1.0, 1e3, 1e7])
    p = np.asarray(smooth_positive_part(x, c))
    assert np.all(p > 0.0)
    assert float(smooth_positive_part(jnp.array(0.0), c)) == 0.5 * c
    assert float(jax.grad(lambda t: smooth_positive_part(t, c))(0.0)) == 0.5
    footprint = np.abs(p - np.maximum(np.asarray(x), 0.0))
    # The footprint is resolvable in double precision only while it is large
    # against the rounding of x itself (width^2/(4|x|) >> 1e-16 |x|, i.e.
    # |x| << 2.5e-3 / 1e-8 ~ 1e2): checked on 3e-5 <= |x| <= 1.
    away = (np.abs(np.asarray(x)) >= 3e-5) & (np.abs(np.asarray(x)) <= 1.0)
    np.testing.assert_allclose(footprint[away],
                               c ** 2 / (4 * np.abs(np.asarray(x)[away])),
                               rtol=0.3, atol=1e-30)
    assert np.all(footprint <= 0.5 * c + 1e-30)
    odd = np.asarray(smooth_positive_part(x, c) - smooth_positive_part(-x, c))
    np.testing.assert_allclose(odd, np.asarray(x), rtol=0.0, atol=1e-16 * 1e7)
    # derivative continuous and monotone in (0, 1): finite differences of the
    # autodiff derivative across zero at the width scale
    grid = jnp.linspace(-5e-5, 5e-5, 2001)
    dp = np.asarray(jax.vmap(jax.grad(lambda t: smooth_positive_part(t, c)))(grid))
    assert np.all(dp > 0.0) and np.all(dp < 1.0)
    assert np.all(np.diff(dp) > 0.0)
    # max |p''| = 1/(2 width) = 5e4, so adjacent grid points (step 5e-8)
    # differ by at most 2.5e-3 (measured exactly that); a jump would read 0.5.
    assert np.max(np.abs(np.diff(dp))) < 3e-3, "derivative jumps at the width scale"
    # Exact inverse, on the domain where the value determines x at double
    # precision: for x < 0 the value is the footprint c^2/(4|x|), whose
    # relative rounding eps x^2/c^2 exceeds 1e-9 beyond |x| ~ 2e-2 (at
    # x = -1e3 the sqrt's argument x^2 + c^2 rounds to x^2 and the value to
    # 5.7e-14, carrying no x at all); every stored indicator column is
    # positive-side, where the round-trip holds to 1e-9 at any magnitude.
    dom = np.asarray(x) >= -2e-2
    back = np.asarray(invert_smooth_positive_part(smooth_positive_part(x, c), c))
    np.testing.assert_allclose(back[dom], np.asarray(x)[dom],
                               rtol=1e-9, atol=1e-13)
    # Homogeneity p(s x; s c) = s p(x; c), exact in exact arithmetic; in
    # floating point the NEGATIVE branch's value is the footprint
    # width^2/(4|x|), realized through the cancellation x + sqrt(x^2 + c^2)
    # whose relative rounding eps x^2/c^2 is not scale-invariant, so the
    # bitwise-tight comparison is made where that rounding is below 1e-14
    # (x >= -3e-5 for this width; the positive branch is exact at every
    # magnitude and is fully covered).
    s = 3.7e4
    dom = np.asarray(x) >= -3e-5
    np.testing.assert_allclose(
        np.asarray(smooth_positive_part(s * x, s * c))[dom], (s * p)[dom],
        rtol=1e-14, atol=0.0)


def test_compute_alpha_is_invariant_under_uniform_density_scaling():
    """alpha(n_lambda) = alpha(n) for n_lambda(r) = lambda^3 n(lambda r): rho,
    sigma and tau scale as lambda^3, lambda^8 and lambda^5, and the smoothing
    width, a multiple of tau_unif ~ lambda^5, scales with them. Checked on
    points at, near and away from the one-orbital limit."""
    rho = jnp.array([0.3, 0.3, 0.3, 0.3, 2.0])
    sigma = jnp.array([0.2, 0.2, 0.2, 0.0, 1.0])
    tau_w = sigma / (8.0 * rho)
    tau_unif = 0.3 * (3.0 * jnp.pi ** 2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
    tau = tau_w + tau_unif * jnp.array([0.0, 2e-6, 0.4, 1.0, 3.0])
    base = np.asarray(compute_alpha(rho, sigma, tau))
    for lam in (1e-3, 0.5, 7.0, 1e2):
        scaled = np.asarray(compute_alpha(lam ** 3 * rho, lam ** 8 * sigma,
                                          lam ** 5 * tau))
        np.testing.assert_allclose(scaled, base, rtol=1e-11, atol=0.0)


def test_compute_alpha_derivative_is_continuous_across_the_one_orbital_limit():
    """The reason for the smoothing: autodiff through compute_alpha at
    tau = tau_W +- epsilon must vary continuously with epsilon (the hard clip
    returned 0 on one side and the full response on the other). Central
    differences of the energy-like scalar sum(alpha) agree with autodiff on
    both sides of the limit and AT it, where the clip's derivative was a
    rounding-selected 0/0."""
    from xcquinox.alec.metagga import _ALPHA_SMOOTHING_WIDTH as c
    rho = jnp.array([0.7]); sigma = jnp.array([0.9])
    tau_w = sigma / (8.0 * rho)
    tau_unif = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
    f = lambda t: compute_alpha(rho, sigma, t).sum()
    slopes = []
    for eps_alpha in (-10 * c, -c, -0.1 * c, 0.0, 0.1 * c, c, 10 * c):
        t = tau_w + eps_alpha * tau_unif
        ad = float(jax.grad(f)(t)[0])
        h = 1e-9 * float(tau_unif[0])
        fd = (float(f(t + h)) - float(f(t - h))) / (2 * h)
        assert abs(ad - fd) < 1e-6 * abs(ad), (eps_alpha, ad, fd)
        slopes.append(ad * float(tau_unif[0]))
    # d alpha / d alpha_raw runs from ~0 to ~1 through exactly 1/2 at the limit
    assert slopes[3] == 0.5
    assert np.all(np.diff(slopes) > 0.0)
    assert slopes[0] < 3e-3 and slopes[-1] > 1.0 - 3e-3


def test_tau_is_differentiable_in_dm():
    mol, mf, ao, dm = _scf("O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469", 0)
    ao_grad = jnp.asarray(ao[1:4])
    g = jax.grad(lambda d: jnp.sum(compute_tau_from_dm(ao_grad, d)))(jnp.asarray(dm))
    g = np.asarray(g)
    assert np.all(np.isfinite(g)) and np.any(g != 0.0)


def test_precompute_and_self_consistent_reassembly():
    """precompute populates metagga_features (one-shot alpha from the PBE DM), and
    the FULL/REASSEMBLE path reproduces it exactly at the PBE DM -- proving the
    descriptor is self-consistent (recomputed from the live DM + AO gradients)."""
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.data import (
        precompute_fixed_density_data, clear_precompute_cache)
    from xcquinox.alec.descriptors import MetaGGAAlphaDescriptor
    from xcquinox.alec.solver import _reassemble_features

    spec = MoleculeSpec(
        name="H2O_mgga",
        atom="O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469",
        basis="def2-svp", grid_level=1)
    desc = MetaGGAAlphaDescriptor()
    clear_precompute_cache()
    md = precompute_fixed_density_data(spec, descriptors=(desc,))

    feat = np.asarray(md["metagga_features"])
    assert feat.shape[1] == 1 and np.all(np.isfinite(feat))

    reas = _reassemble_features(
        descriptors=(desc,), dm=md["dm_pbe"], s_matrix=md["s_matrix"],
        n_grid=feat.shape[0],
        ao_grad=md["ao_grid_deriv"][1:4], rho=md["rho_grid"], sigma=md["sigma_grid"])
    assert np.allclose(np.asarray(reas), feat, atol=1e-8)


# ---------------------------------------------------------------------------
# M2: the DFS-faithful meta-GGA network extension (meta_gga flag)
# ---------------------------------------------------------------------------
def test_meta_gga_xnet_ueg_recovery_and_alpha_sensitivity():
    from xcquinox.alec.networks import AlecGGA_XNet
    # non-zero MLP so the (x2 + tanh^2(x3)) gate actually matters
    net = AlecGGA_XNet(n_extra_features=1, depth=3, nodes=16, lob_lim=1.174,
                       meta_gga=True, metagga_alpha_index=0,
                       descriptor_log_transform=True, zero_init_final_layer=False)
    # UEG: s=0 (sigma=0) AND alpha=1 -> gate=0 -> F_x = 1 exactly
    fx_ueg = float(net(jnp.array([1.0, 0.0, 1.0])))
    assert abs(fx_ueg - 1.0) < 1e-6
    # away from the iso-orbital UEG (alpha != 1) -> gate != 0 -> F_x != 1
    fx_off = float(net(jnp.array([1.0, 0.0, 3.0])))
    assert abs(fx_off - 1.0) > 1e-6


def test_meta_gga_off_path_byte_identical():
    from xcquinox.alec.networks import AlecGGA_XNet
    a = AlecGGA_XNet(n_extra_features=1, depth=3, nodes=16, seed=7)
    b = AlecGGA_XNet(n_extra_features=1, depth=3, nodes=16, seed=7, meta_gga=False)
    inp = jnp.array([0.5, 0.1, 0.3])
    assert float(a(inp)) == float(b(inp))


def test_from_spec_meta_gga_requires_metagga_descriptor():
    from xcquinox.alec.config import ArchitectureConfig
    with pytest.raises(ValueError, match="requires a 'metagga' descriptor"):
        ArchitectureConfig.from_spec("bad_mgga", 3, 16, descriptors=[], meta_gga=True)
    arch = ArchitectureConfig.from_spec("ok_mgga", 3, 16, descriptors=["metagga"],
                                        meta_gga=True)
    assert arch.meta_gga is True


def test_create_network_pair_meta_gga_wiring():
    from xcquinox.alec.config import ArchitectureConfig
    from xcquinox.alec.networks import create_network_pair
    # combined arch: [cusp, rung35, metagga] -> alpha is the LAST column
    arch = ArchitectureConfig.from_spec(
        "combo_mgga", 3, 16, descriptors=["cusp", "rung35", "metagga"],
        meta_gga=True, descriptor_log_transform=True, zero_init_final_layer=True)
    xnet, cnet = create_network_pair(arch)
    n_before = sum(d.n_features for d in arch.materialize_descriptors()
                   if type(d).__name__ != "MetaGGAAlphaDescriptor")
    assert xnet.meta_gga and xnet.metagga_alpha_index == n_before
    assert cnet.meta_gga and cnet.metagga_alpha_index == n_before
    assert xnet.lob_lim == 1.174 and cnet.lob_lim == 2.0


# ---------------------------------------------------------------------------
# M1+M2 integration: a meta_gga arch through a FULL differentiable SCF, with the
# iso-orbital alpha reassembled self-consistently each cycle and gradients
# flowing through the DFS (x2 + tanh^2(x3)) gate. The definitive "it works".
# ---------------------------------------------------------------------------
def test_meta_gga_full_scf_is_differentiable_and_alpha_flows():
    import equinox as eqx
    from xcquinox.alec.config import ArchitectureConfig
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.descriptors import MetaGGAAlphaDescriptor
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.solver import (
        SolverConfig, SolverBackend, SolverMode, FeaturePolicy, run_scf)
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule

    arch = ArchitectureConfig.from_spec(
        "t_mgga_scf", 3, 16, descriptors=["metagga"], meta_gga=True,
        descriptor_log_transform=True, use_polarized_correlation=True,
        zero_init_final_layer=False)  # non-zero MLP so the alpha gate matters
    model = AlecGGAModel.from_arch(arch, seed=0)
    data = precompute_fixed_density_data(
        h2_molecule(), descriptors=(MetaGGAAlphaDescriptor(),),
        required_keys=("eri",))
    cfg = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL, max_cycles=3,
        conv_tol=1e-12, feature_policy=FeaturePolicy.REASSEMBLE)

    def total_energy_fn(m):
        return run_scf(cfg, m, data).total_energy

    val, grads = eqx.filter_value_and_grad(total_energy_fn)(model)
    assert jnp.isfinite(val), "meta-GGA full-SCF energy is not finite"
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert len(leaves) > 0
    assert all(bool(jnp.all(jnp.isfinite(l))) for l in leaves), "non-finite grad"
    # alpha actually flows through the gate -> at least one non-zero gradient
    assert any(bool(jnp.any(l != 0.0)) for l in leaves), "no gradient reached the net"


def test_meta_gga_full_scf_manual_runs_no_nan():
    """FULL mode: a meta_gga arch reassembles the iso-orbital alpha each cycle
    (treating alpha as a descriptor feature rather than native MGGA) and stays
    finite.

    Backend note (2026-08-06): this ran on PYSCFAD until that backend began
    refusing DM-dependent descriptors under REASSEMBLE -- its per-point eval_xc
    callback cannot carry the de/dfeatures . dfeatures/dP term, so it would
    return a V_xc that is not the derivative of E_xc. The refusal is pinned in
    test_scf_backends.py; the finiteness property this test exists for is
    exercised on MANUAL, which assembles that term exactly and is the backend
    the production sweep runs.
    """
    from xcquinox.alec.config import ARCHITECTURES
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.data import (precompute_fixed_density_data,
                                    clear_precompute_cache)
    from xcquinox.alec.solver import (SolverConfig, SolverBackend, SolverMode,
                                      FeaturePolicy, run_scf)
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule
    clear_precompute_cache()
    arch = ARCHITECTURES["deep_mgga_3x16"]
    model = AlecGGAModel.from_arch(arch, seed=0)
    data = precompute_fixed_density_data(
        h2_molecule(), descriptors=arch.materialize_descriptors(),
        # "eri" is required by the MANUAL backend, which builds the Coulomb
        # matrix itself; the pyscfad backend this test previously used got J
        # from pyscf and did not need it in mol_data.
        required_keys=("metagga_features", "eri"))
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                       feature_policy=FeaturePolicy.REASSEMBLE, max_cycles=10)
    result = run_scf(cfg, model, data)
    assert np.isfinite(float(result.total_energy)), "non-finite energy"
    assert np.all(np.isfinite(np.asarray(result.density_matrix))), "non-finite DM"
    assert np.all(np.isfinite(np.asarray(result.features_used))), "non-finite features"


# ---------------------------------------------------------------------------
# HISTORY Phase 17 (singularity 1 of 2): the iso-orbital-alpha low-density-tail
# blowup. alpha = (tau - tau_W)/tau_unif divides by n / n^{5/3}; on a low-density
# tail its VALUE reaches ~1e4-1e7 (even at normal bases -- alpha is NOT physically
# O(1) here) and its GRADIENT ~1e28, which the unrolled full-SCF backprop compounds
# toward a NaN meta-GGA training gradient. Fixed by the clip [0, _ALPHA_MAX].
# (The OTHER bh76:HLi singularity -- the polarized-correlation potential jvp NaN on
# the negative-density tail -- lives in test_solv01_split_xc.py.)
#
# 2026-08-06: the companion tail-gradient FREEZE was removed. This test previously
# asserted `gs == 0.0 and gr == 0.0` below _RHO_GRAD_CUTOFF, i.e. it pinned the
# very stop_gradient that made autodiff disagree with the function it
# differentiates -- and, because alpha is an energy ingredient, made V_xc stop
# being dE_xc/dP for every meta-GGA architecture. The property worth pinning is
# FINITENESS across the whole tail, which the clip delivers on its own; a frozen
# gradient is not a correctness property, it was a symptom mask.
# ---------------------------------------------------------------------------
def test_compute_alpha_bounded_and_tail_gradient_finite():
    from xcquinox.alec.metagga import compute_alpha, _ALPHA_MAX
    for rho_v in (1e-20, 1e-12, 1e-8, 1e-6, 1e-3, 1e-2, 0.1, 1.0):
        rho = jnp.array([rho_v])
        sigma = jnp.array([max(rho_v ** 2 * 1e-2, 1e-40)])
        tau = jnp.array([1e-3])
        a = float(compute_alpha(rho, sigma, tau)[0])
        assert 0.0 <= a <= _ALPHA_MAX + 1e-9, f"alpha={a} out of [0,{_ALPHA_MAX}] at rho={rho_v}"
        gs = float(jax.grad(lambda s: compute_alpha(rho, s, tau).sum())(sigma)[0])
        gr = float(jax.grad(lambda r: compute_alpha(r, sigma, tau).sum())(rho)[0])
        assert np.isfinite(gs) and np.isfinite(gr), f"non-finite alpha grad at rho={rho_v}"


def test_compute_alpha_has_no_stop_gradient_on_the_energy_path():
    """alpha feeds E_xc, so a frozen gradient anywhere in it breaks V_xc = dE_xc/dP.

    Pinned two ways: the source carries no ``stop_gradient``, and autodiff agrees
    with a finite difference in the tail regime where the freeze used to be live.
    """
    import inspect
    from xcquinox.alec import metagga as MG

    # Strip the docstring and every comment before scanning: the body carries a
    # comment quoting the removed line verbatim, so a naive substring search
    # matches the documentation of the fix rather than a regression.
    src = inspect.getsource(MG.compute_alpha).split('"""')[-1]
    code_only = "\n".join(
        line.split("#", 1)[0] for line in src.splitlines()
    )
    assert "stop_gradient" not in code_only, (
        "compute_alpha reintroduced a stop_gradient on the energy path")

    # The probe must sit below the old _RHO_GRAD_CUTOFF (so the freeze WOULD have
    # been live) AND below _ALPHA_MAX (so the clip is not legitimately zeroing the
    # derivative on its own -- at rho = 1e-7 with an arbitrary tau, alpha pins at
    # the ceiling and AD = FD = 0 for a reason that has nothing to do with the
    # freeze). Choosing tau = tau_W + tau_unif puts alpha at exactly 1.
    rho_v, sigma_v = 1e-7, 1e-16
    tau_unif = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0) * rho_v ** (5.0 / 3.0)
    tau_v = sigma_v / (8.0 * rho_v) + tau_unif
    rho = jnp.array([rho_v])
    sigma = jnp.array([sigma_v])
    tau = jnp.array([tau_v])
    assert rho_v < MG._RHO_GRAD_CUTOFF, "probe no longer in the formerly-frozen regime"
    assert float(MG.compute_alpha(rho, sigma, tau)[0]) < MG._ALPHA_MAX - 1.0, (
        "probe drifted onto the alpha ceiling, where the clip zeroes the "
        "derivative for an unrelated reason")

    ad = float(jax.grad(lambda t: MG.compute_alpha(rho, sigma, t).sum())(tau)[0])
    eps = 1e-15
    fd = (float(MG.compute_alpha(rho, sigma, tau + eps).sum())
          - float(MG.compute_alpha(rho, sigma, tau - eps).sum())) / (2 * eps)
    assert np.isfinite(ad) and np.isfinite(fd), "non-finite alpha derivative"
    assert ad != 0.0, "alpha gradient is still frozen in the tail"
    rel = abs(fd - ad) / max(abs(fd), abs(ad), 1e-30)
    assert rel < 1e-5, (
        f"autodiff disagrees with finite difference in the tail: "
        f"FD={fd:.6e} AD={ad:.6e} rel={rel:.3e}")


def test_compute_alpha_ceiling_is_noop_below_it_and_smoothing_is_its_footprint():
    """Below the ceiling the only difference from the bare formula is the
    smooth positive part's footprint width^2/(4 alpha_raw), which at these
    resolved points (alpha_raw >= 0.044) is at most 5.7e-10; the ceiling itself
    is a no-op there. (Real molecular grids DO have alpha > _ALPHA_MAX at
    rho > 1e-8 -- those points ARE clipped; the ceiling is energy-faithful via
    gate saturation, not via leaving alpha untouched.)"""
    from xcquinox.alec.metagga import compute_alpha, _ALPHA_SMOOTHING_WIDTH as c
    def bare(rho, sigma, tau):
        rs = jnp.maximum(rho, 1e-30)
        tw = sigma / (8.0 * rs)
        tu = (3.0 / 10.0) * (3.0 * jnp.pi ** 2) ** (2.0 / 3.0) * rs ** (5.0 / 3.0)
        return (tau - tw) / jnp.maximum(tu, 1e-30)
    rho = jnp.array([0.05, 0.1, 0.3, 1.0, 3.0])
    sigma = jnp.array([0.01, 0.02, 0.05, 0.1, 0.2])
    tau = jnp.array([0.03, 0.05, 0.1, 0.3, 0.8])
    raw = np.asarray(bare(rho, sigma, tau))
    assert np.min(raw) > 0.04, raw
    gap = np.asarray(compute_alpha(rho, sigma, tau)) - raw
    np.testing.assert_allclose(gap, c ** 2 / (4.0 * raw), rtol=1e-6, atol=0.0)


def test_subset_selection_alpha_clip_matches_metagga():
    """The numpy twin clips to the same ceiling so precomputed and live alpha
    agree at the top; at the bottom it keeps the hard clip (a selection
    heuristic, never differentiated) and differs from the live indicator by
    at most width / 2 = 5e-6 there."""
    from xcquinox.alec.subset_selection import compute_descriptor_triple
    from xcquinox.alec.metagga import _ALPHA_MAX
    out = compute_descriptor_triple(  # a low-density point that blows up unclipped
        np.array([1e-12]), np.array([1e-20]), np.array([1e-3]))
    assert out["alpha"][0] <= _ALPHA_MAX + 1e-9


def test_meta_gga_full_scf_gradient_finite_on_diffuse_tail():
    """Regression (HISTORY 17): a meta_gga arch's FULL-SCF training gradient on a
    fully-polarized atom at the diffuse DFS-parity basis (H rho_min ~ 6e-10) was NaN
    before the alpha clip -- the notebook `bh76:HLi` step-0 failure. Now finite."""
    import equinox as eqx
    from xcquinox.alec.config import ArchitectureConfig, MoleculeSpec
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.descriptors import MetaGGAAlphaDescriptor
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.solver import (SolverConfig, SolverBackend, SolverMode,
                                      FeaturePolicy, run_scf)
    arch = ArchitectureConfig.from_spec(
        "t_mgga_tail", 3, 16, descriptors=["metagga"], meta_gga=True,
        descriptor_log_transform=True, use_polarized_correlation=True,
        zero_init_final_layer=False)
    model = AlecGGAModel.from_arch(arch, seed=0)
    spec = MoleculeSpec(name="H", atom="H 0 0 0", basis="6-311++G(3df,2pd)",
                        charge=0, spin=1, atom_composition=(("H", 1),), grid_level=2)
    data = precompute_fixed_density_data(
        spec, descriptors=(MetaGGAAlphaDescriptor(),), required_keys=("eri",))
    assert float(np.min(np.asarray(data["rho_grid"]))) < 1e-8, "lost the low-density trigger"
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                       max_cycles=3, conv_tol=1e-12, feature_policy=FeaturePolicy.REASSEMBLE)
    val, grads = eqx.filter_value_and_grad(
        lambda m: run_scf(cfg, m, data).total_energy)(model)
    assert bool(jnp.isfinite(val)), "meta-GGA full-SCF energy non-finite on the diffuse tail"
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert all(bool(jnp.all(jnp.isfinite(l))) for l in leaves), \
        "NON-FINITE meta-GGA training gradient (the bh76:HLi NaN regression)"
    assert any(bool(jnp.any(l != 0.0)) for l in leaves), "no gradient reached the net"
