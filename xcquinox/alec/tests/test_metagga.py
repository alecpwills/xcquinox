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
    # OH doublet: the meta-GGA iso-orbital tau is the TOTAL kinetic energy density.
    mol, mf, ao, dm = _scf("O 0 0 0; H 0 0 0.97", 1)
    tau_ref = mf._numint.eval_rho(mol, ao, dm[0] + dm[1], xctype="MGGA")[5]
    tau = np.asarray(compute_tau_from_dm(jnp.asarray(ao[1:4]), jnp.asarray(dm)))
    assert np.allclose(tau, tau_ref, atol=1e-9)


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


def test_meta_gga_full_scf_pyscfad_runs_no_nan():
    """pyscfad backend, FULL mode: a meta_gga arch reassembles the iso-orbital
    alpha on pyscfad's pruned grid each cycle (mirroring rung-3.5, treating alpha as
    a descriptor feature -- NOT native pyscfad MGGA -- for consistency with the
    manual backend). Finite + no NaN == pyscfad reaches meta-GGA parity."""
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
        required_keys=("metagga_features",))
    cfg = SolverConfig(backend=SolverBackend.PYSCFAD, mode=SolverMode.FULL,
                       feature_policy=FeaturePolicy.REASSEMBLE, max_cycles=10)
    result = run_scf(cfg, model, data)
    assert np.isfinite(float(result.total_energy)), "non-finite energy"
    assert np.all(np.isfinite(np.asarray(result.density_matrix))), "non-finite DM"
    assert np.all(np.isfinite(np.asarray(result.features_used))), "non-finite features"
