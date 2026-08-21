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
        tau_doubled = np.asarray(compute_tau_from_dm(
            jnp.asarray(ao[1:4]), doubled_spin_dm(jnp.asarray(dm), s)))
        # Purely relative: tau spans 2.6e-13 to 8.1e+03 Ha/bohr^3 on this grid,
        # so an absolute floor would leave the tail untested and a default
        # rtol of 1e-5 would admit ~8e-2 at the peak. The two evaluations
        # differ only by summation order and agree to 1.0e-15 relative.
        np.testing.assert_allclose(tau_doubled, 2.0 * tau_ref,
                                   rtol=1e-12, atol=0.0)


def test_alpha_matches_repo_scan_formula():
    from xcquinox.alec.subset_selection import compute_descriptor_triple
    rng = np.random.default_rng(0)
    rho = np.abs(rng.normal(1.0, 0.5, 64)) + 1e-3
    sigma = np.abs(rng.normal(0.5, 0.3, 64))
    tau = np.abs(rng.normal(0.4, 0.2, 64))
    ref = compute_descriptor_triple(rho, sigma, tau)["alpha"]
    got = np.asarray(compute_alpha(jnp.asarray(rho), jnp.asarray(sigma),
                                   jnp.asarray(tau)))
    assert np.allclose(got, ref, atol=1e-10)


def test_alpha_uniform_gas_limit_is_one():
    # For the uniform electron gas sigma=0 (tau_W=0) and tau=tau_unif -> alpha=1.
    rho = jnp.array([1.0, 2.0, 0.5])
    tau_unif = (3.0 / 10.0) * (3.0 * jnp.pi**2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
    alpha = np.asarray(compute_alpha(rho, jnp.zeros_like(rho), tau_unif))
    assert np.allclose(alpha, 1.0, atol=1e-10)


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


def test_compute_alpha_clip_is_noop_below_ceiling():
    """The clip is a no-op wherever raw alpha < _ALPHA_MAX and rho > cutoff:
    such points are byte-identical to the pre-fix formula. (Note: real molecular
    grids DO have alpha > _ALPHA_MAX at rho > 1e-8 -- those points ARE clipped;
    the fix is energy-faithful via gate saturation, not via leaving alpha
    untouched. This test only pins the no-op-below-ceiling property.)"""
    from xcquinox.alec.metagga import compute_alpha
    def old(rho, sigma, tau):
        rs = jnp.maximum(rho, 1e-30)
        tw = sigma / (8.0 * rs)
        tu = (3.0 / 10.0) * (3.0 * jnp.pi ** 2) ** (2.0 / 3.0) * rs ** (5.0 / 3.0)
        return jnp.maximum((tau - tw) / jnp.maximum(tu, 1e-30), 0.0)
    rho = jnp.array([0.05, 0.1, 0.3, 1.0, 3.0])
    sigma = jnp.array([0.01, 0.02, 0.05, 0.1, 0.2])
    tau = jnp.array([0.02, 0.05, 0.1, 0.3, 0.8])
    assert float(jnp.max(jnp.abs(
        compute_alpha(rho, sigma, tau) - old(rho, sigma, tau)))) == 0.0


def test_subset_selection_alpha_clip_matches_metagga():
    """The numpy twin clips to the same ceiling so precomputed and live alpha agree."""
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
