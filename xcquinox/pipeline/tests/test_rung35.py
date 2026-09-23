"""Phase-1 tests for the rung-3.5 projected-AO occupancy machinery.

The descriptor is the bounded local occupancy
    n_sigma(r) = A(r)^T P^sigma A(r)  in [0, 1]
(Janesko, arXiv:2206.07118 Eq. 12-13; M11plus, Verma et al. JCTC 15, 4804 (2019)),
where
    A_mu(r) = <chi_mu | phi^G_r>,   phi^G = (2 alpha/pi)^{3/4} exp(-alpha |r - r_m|^2)
is the overlap of basis function chi_mu with an L2-normalized Gaussian projector at
the grid point r_m.

Key property exploited throughout: A_mu(r) depends only on the basis, the grid, and
alpha -- NOT on the density matrix or the density -- so it is a precomputed CONSTANT
(a plain PySCF overlap), never differentiated. The occupancy n_sigma = A^T P A is then
a trivial einsum, linear and differentiable in the live DM, and bounded [0, 1] by
Bessel's inequality (P^sigma is PSD => >= 0; {psi_i} L2-orthonormal + ||phi^G||=1 => <= 1).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest


ALPHA = 0.2  # projector width (a0^-2), grounded at the M11plus kernel scale d^2=5 a0^2


def _h2():
    """Small real closed-shell molecule: H2 / def2-svp PBE."""
    from pyscf import dft, gto
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="def2-svp", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.kernel()
    return (mol, np.asarray(mf.make_rdm1()),
            np.asarray(mf.grids.coords), np.asarray(mf.grids.weights))


def test_projected_ao_analytic_s_s_overlap():
    """Exact closed-form check pinning the normalization, independent of the
    intor_cross implementation. For a single s-primitive chi (exponent beta) and
    the normalized s-Gaussian projector phi^G (exponent alpha) at r_m:
        A = N_beta N_alpha (pi/(alpha+beta))^{3/2} exp(-alpha beta/(alpha+beta) |R-r_m|^2),
        N_x = (2x/pi)^{3/4}.
    """
    from pyscf import gto
    from xcquinox.pipeline.rung35 import compute_projected_ao
    beta = 0.8
    mol = gto.M(atom="H 0 0 0", basis={"H": [[0, [beta, 1.0]]]}, spin=1, verbose=0)
    rm = np.array([[0.5, 0.1, -0.2]])
    A = float(np.asarray(compute_projected_ao(mol, rm, ALPHA))[0, 0])
    R2 = float(np.sum(rm[0] ** 2))
    Nb = (2 * beta / np.pi) ** 0.75
    Na = (2 * ALPHA / np.pi) ** 0.75
    ref = Nb * Na * (np.pi / (ALPHA + beta)) ** 1.5 * \
        np.exp(-ALPHA * beta / (ALPHA + beta) * R2)
    np.testing.assert_allclose(A, ref, rtol=1e-9, atol=1e-12)


def test_projected_ao_matches_numerical_quadrature():
    """A_mu(r_m) = integral chi_mu(r) phi^G(r-r_m) dr matches a direct grid
    quadrature sum_g w_g chi_mu(r_g) phi^G(r_g - r_m) -- an oracle independent of
    the intor_cross implementation, valid for general angular momentum."""
    from xcquinox.pipeline.rung35 import compute_projected_ao
    mol, _dm, coords, weights = _h2()
    test_pts = coords[:: max(1, len(coords) // 6)][:5]
    A = np.asarray(compute_projected_ao(mol, test_pts, ALPHA))
    ao = mol.eval_gto("GTOval", coords)               # (Ngrid, nao)
    norm = (2 * ALPHA / np.pi) ** 0.75
    for p, rm in enumerate(test_pts):
        g = norm * np.exp(-ALPHA * np.sum((coords - rm) ** 2, axis=1))
        ref = np.einsum("g,gm->m", weights * g, ao)
        np.testing.assert_allclose(A[p], ref, rtol=2e-2, atol=2e-3,
                                   err_msg=f"projected-AO row {p} vs quadrature")


def test_occupancy_bounded_0_1_real_dm():
    """n_sigma(r) = A(r)^T P^sigma A(r) in [0, 1] for the real PBE
    single-determinant DM (PSD P^sigma => >= 0; Bessel + normalized projector => <= 1)."""
    from xcquinox.pipeline.rung35 import (compute_projected_ao,
                                       compute_rung35_occupancy)
    mol, dm, coords, _w = _h2()
    A = compute_projected_ao(mol, coords, ALPHA)
    n = np.asarray(compute_rung35_occupancy(jnp.asarray(A), jnp.asarray(dm)))
    assert n.shape == (len(coords), 2), n.shape
    assert np.all(np.isfinite(n)), "occupancy has non-finite entries"
    assert n.min() >= -1e-9, f"occupancy < 0: min={n.min()}"
    assert n.max() <= 1.0 + 1e-6, f"occupancy > 1: max={n.max()}"


def test_occupancy_linear_and_differentiable_in_dm():
    """n_sigma is linear in P^sigma (A constant) => finite gradient wrt the live
    DM -- the property the self-consistent SCF relies on."""
    from xcquinox.pipeline.rung35 import compute_rung35_occupancy
    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.standard_normal((20, 6)))
    f = lambda d: jnp.sum(compute_rung35_occupancy(A, d + d.T))
    dm = jnp.asarray(rng.standard_normal((6, 6)))
    grad = jax.grad(f)(dm)
    assert np.all(np.isfinite(np.asarray(grad))), "non-finite gradient wrt DM"


# ===========================================================================
# Phase 2: the DMRung35Descriptor (registration + delegation to rung35.py).
# ===========================================================================


def test_rung35_descriptor_compute_from_dm_reassembles_occupancy():
    """The reassemble kernel recomputes the occupancy from the LIVE DM + the
    constant projected-AO matrix A (the self-consistent SCF path)."""
    from xcquinox.pipeline.descriptors import make_descriptor
    from xcquinox.pipeline.rung35 import compute_rung35_occupancy
    d = make_descriptor("rung35")
    rng = np.random.default_rng(1)
    A = jnp.asarray(rng.standard_normal((15, 5)))
    dm = jnp.asarray(rng.standard_normal((5, 5)))
    dm = dm + dm.T
    got = d.compute_from_dm(proj_ao=A, dm=dm)
    assert got.shape == (15, 2)
    assert jnp.allclose(got, compute_rung35_occupancy(A, dm))


# ===========================================================================
# Phase 3: gated precompute + self-consistent SCF reassemble.
# ===========================================================================

def _h2_spec():
    from xcquinox.pipeline.config import MoleculeSpec
    return MoleculeSpec.from_dict(name="H2", atom="H 0 0 0; H 0 0 0.74",
                                  basis="def2-svp", charge=0, spin=0,
                                  atom_composition={"H": 2})


def test_precompute_populates_rung35_features_when_descriptor_present():
    """The gated precompute computes + stores both the constant projected-AO
    matrix A (rung35_proj_ao) and the one-shot occupancy (rung35_features),
    correctly shaped, finite, and bounded [0, 1]."""
    from xcquinox.pipeline.data import (precompute_fixed_density_data,
                                    clear_precompute_cache)
    from xcquinox.pipeline.descriptors import make_descriptor
    clear_precompute_cache()
    data = precompute_fixed_density_data(
        _h2_spec(), descriptors=(make_descriptor("rung35"),),
        required_keys=("rung35_features",))
    A = data.get("rung35_proj_ao")
    feat = data.get("rung35_features")
    N = data["rho_grid"].shape[0]
    nao = data["s_matrix"].shape[0]
    assert A is not None and tuple(A.shape) == (N, nao), None if A is None else A.shape
    assert feat is not None and tuple(feat.shape) == (N, 2)
    assert jnp.all(jnp.isfinite(feat))
    assert float(jnp.min(feat)) >= -1e-9 and float(jnp.max(feat)) <= 1 + 1e-6


def test_reassemble_on_grid_rung35_matches_occupancy():
    """The pyscfad-backend reassemble produces the rung-3.5 occupancy on its own
    grid. With a cached proj_ao A it uses it; without one it recomputes A on the
    grid (fallback) -- both give compute_rung35_occupancy(A, dm)."""
    from pyscf import gto
    from xcquinox.pipeline.solver_pyscfad import _reassemble_features_on_grid
    from xcquinox.pipeline.descriptors import make_descriptor
    from xcquinox.pipeline.rung35 import (compute_projected_ao,
                                       compute_rung35_occupancy)
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="def2-svp", verbose=0)
    rng = np.random.default_rng(4)
    coords = jnp.asarray(rng.standard_normal((10, 3)))
    d = make_descriptor("rung35")
    A = jnp.asarray(compute_projected_ao(mol, np.asarray(coords), float(d.alpha)))
    nao = mol.nao_nr()
    dm = jnp.asarray(rng.standard_normal((nao, nao))); dm = dm + dm.T
    S = jnp.asarray(mol.intor("int1e_ovlp"))
    ref = compute_rung35_occupancy(A, dm)
    f_cached = _reassemble_features_on_grid(
        (d,), dm=dm, s_matrix=S, grid_coords=coords, mol=mol, rung35_proj_ao=A)
    f_fallback = _reassemble_features_on_grid(
        (d,), dm=dm, s_matrix=S, grid_coords=coords, mol=mol)
    assert f_cached.shape == (10, 2)
    assert jnp.allclose(f_cached, ref)
    assert jnp.allclose(f_fallback, ref)  # fallback recomputes the same A


# ===========================================================================
# Phase 4: additive deep_rung35 archs + X/C parity.
# ===========================================================================


# ===========================================================================
# Phase 5: NaN robustness. The occupancy A^T P A is bounded [0,1] by
# construction (no division by rho / k_F anywhere), so it is NaN-safe; these
# pin that across extreme widths and the low-density (far-field) limit.
# ===========================================================================


def test_occupancy_far_field_vanishes_smoothly():
    """Low-density / far-from-molecule limit: A -> 0, so the occupancy -> 0
    smoothly (no NaN, no blow-up -- the rho->0 limit is benign by construction)."""
    from xcquinox.pipeline.rung35 import (compute_projected_ao,
                                       compute_rung35_occupancy)
    mol, dm, _c, _w = _h2()
    far = np.array([[0.0, 0.0, 50.0], [50.0, 50.0, 50.0]])
    A = compute_projected_ao(mol, far, 0.2)
    n = np.asarray(compute_rung35_occupancy(jnp.asarray(A), jnp.asarray(dm)))
    assert np.all(np.isfinite(n))
    assert np.allclose(n, 0.0, atol=1e-8), f"occupancy not ~0 far away: max={n.max()}"


def test_occupancy_leak_free_size_consistent():
    """DEFINITIVE leak-freeness check (the whole motivation): the local occupancy
    near a fragment is UNCHANGED by a distant fragment (size-consistency). This is
    exactly what the global dm_statistics descriptor FAILED -- dm_entropy ~ ln(Nocc)
    grows with system size and leaks molecule identity. Compute n_sigma at points
    near fragment A in molecule (A) vs molecule (A + far-away B); assert equal."""
    from pyscf import dft, gto
    from xcquinox.pipeline.rung35 import (compute_projected_ao,
                                       compute_rung35_occupancy)
    alpha = 0.2
    # All in BOHR (PySCF's internal/grid unit; the probe points below are Bohr).
    # Fragment A = H2 (bond 1.4 a0); B = a second H2 20 a0 away. A's geometry is
    # IDENTICAL in both molecules so any occupancy change is a leak, not geometry.
    mA = gto.M(atom="H 0 0 0; H 0 0 1.4", basis="def2-svp", unit="Bohr", verbose=0)
    mfA = dft.RKS(mA); mfA.xc = "pbe"; mfA.kernel()
    mAB = gto.M(atom="H 0 0 0; H 0 0 1.4; H 0 0 20.0; H 0 0 21.4",
                basis="def2-svp", unit="Bohr", verbose=0)
    mfAB = dft.RKS(mAB); mfAB.xc = "pbe"; mfAB.kernel()
    pts = np.array([[0., 0., 0.0], [0., 0., 0.7], [0., 0., 1.4], [0.5, 0., 0.7]])
    nA = np.asarray(compute_rung35_occupancy(
        jnp.asarray(compute_projected_ao(mA, pts, alpha)),
        jnp.asarray(mfA.make_rdm1())))
    nAB = np.asarray(compute_rung35_occupancy(
        jnp.asarray(compute_projected_ao(mAB, pts, alpha)),
        jnp.asarray(mfAB.make_rdm1())))
    np.testing.assert_allclose(
        nA, nAB, atol=1e-4, rtol=1e-3,
        err_msg="occupancy near A changed when a distant fragment B was added "
                "-> NOT leak-free / size-consistent (the dm_statistics failure mode)")


# ===========================================================================
# Gap-closing tests from the 2026-06-29 review: d-functions, and the multi-cycle
# SCF loop + model-grad + additivity. The H-only fixtures above never exercised d-shells or the lax.scan
# SCF body; these close that.
# ===========================================================================

def _h2o_mol(spin=0):
    """Real molecule WITH a d-shell (O in def2-svp), RKS (spin=0) or its OH/UKS
    radical (spin=1). Geometry in Bohr."""
    from pyscf import gto
    atom = ("O 0 0 0; H 0 0 1.81; H 1.75 0 -0.45" if spin == 0
            else "O 0 0 0; H 0 0 1.83")
    return gto.M(atom=atom, basis="def2-svp", spin=spin, unit="Bohr", verbose=0)


def test_projected_ao_d_functions_vs_ghost_atom_oracle():
    """A_mu(r) for the O d-shell (H2O/def2-svp) matches an INDEPENDENT ghost-atom
    gto.M overlap (a different construction than the module's fakemol_for_charges).
    Closes the gap that the H-only fixtures left d-functions empirically untested."""
    from pyscf import gto
    from xcquinox.pipeline.rung35 import compute_projected_ao
    mol = _h2o_mol()
    assert any(mol.bas_angular(b) >= 2 for b in range(mol.nbas)), "no d shell in fixture"
    pts = np.array([[0., 0., 0.1], [0.3, 0.2, -0.1], [0., 0., 1.81],
                    [1.0, 0., -0.3], [0.5, 0.5, 0.5]])
    A_mod = np.asarray(compute_projected_ao(mol, pts, ALPHA))
    A_ghost = []
    for rm in pts:
        g = gto.M(atom=[["H", (float(rm[0]), float(rm[1]), float(rm[2]))]],
                  basis={"H": [[0, [ALPHA, 1.0]]]}, spin=1, charge=0,
                  unit="Bohr", verbose=0)
        A_ghost.append(np.asarray(gto.intor_cross("int1e_ovlp", mol, g))[:, 0])
    np.testing.assert_allclose(A_mod, np.array(A_ghost), rtol=1e-9, atol=1e-11,
                               err_msg="projected-AO A wrong for d-functions")


def _rung35_model_data(eri=True):
    from xcquinox.pipeline.config import ARCHITECTURES
    from xcquinox.pipeline.models import AlecGGAModel
    from xcquinox.pipeline.data import (precompute_fixed_density_data,
                                    clear_precompute_cache)
    clear_precompute_cache()
    arch = ARCHITECTURES["deep_rung35_3x16"]
    model = AlecGGAModel.from_arch(arch, seed=0)
    keys = ("cusp_features", "rung35_features") + (("eri",) if eri else ())
    data = precompute_fixed_density_data(
        _h2_spec(), descriptors=arch.materialize_descriptors(), required_keys=keys)
    return model, data


@pytest.mark.slow
def test_rung35_self_consistent_through_full_scf_loop():
    """The FULL-SCF LOOP (not just the reassemble kernel) recomputes the occupancy
    from the EVOLVING DM each cycle: after a manual FULL SCF the features actually
    used differ from the frozen PBE one-shot (same grid -> a fair comparison) and
    equal compute_rung35_occupancy(A, final_DM). Proves the reassemble fires inside
    the lax.scan body and tracks the live DM."""
    from xcquinox.pipeline.solver import (SolverConfig, SolverBackend, SolverMode,
                                      FeaturePolicy, run_scf)
    from xcquinox.pipeline.rung35 import compute_rung35_occupancy
    model, data = _rung35_model_data(eri=True)
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                       feature_policy=FeaturePolicy.REASSEMBLE, max_cycles=8)
    result = run_scf(cfg, model, data)
    f_used = np.asarray(result.features_used)            # (N, 4): cusp 0-1, rung35 2-3
    pbe_occ = np.asarray(data["rung35_features"])        # (N, 2) PBE one-shot
    assert np.all(np.isfinite(f_used))
    assert not np.allclose(f_used[:, 2:], pbe_occ, atol=1e-7), \
        "rung-3.5 occupancy frozen at the PBE value -> reassemble did NOT fire in the SCF loop"
    A = jnp.asarray(data["rung35_proj_ao"])
    final_occ = np.asarray(compute_rung35_occupancy(A, jnp.asarray(result.density_matrix)))
    np.testing.assert_allclose(f_used[:, 2:], final_occ, atol=1e-6,
                               err_msg="features_used != occupancy of the final DM")


@pytest.mark.slow
def test_rung35_training_gradient_flows_through_the_occupancy_path(monkeypatch):
    """The training gradient flows through the multi-cycle SCF AND specifically the
    rung-3.5 occupancy: jax.grad of a FULL-SCF energy loss wrt the model is finite
    and non-zero, and DETACHING the occupancy (stop_gradient) CHANGES that gradient
    -- so A's precompute did not sever the graph."""
    import equinox as eqx
    import xcquinox.pipeline.rung35 as r35
    from xcquinox.pipeline.config import ArchitectureConfig
    from xcquinox.pipeline.models import AlecGGAModel
    from xcquinox.pipeline.solver import (SolverConfig, SolverBackend, SolverMode,
                                      FeaturePolicy, run_scf)
    _, data = _rung35_model_data(eri=True)  # reuse the precompute (descriptors fixed)
    # NON-zero-init so the NN enhancement is actually sensitive to its inputs: with
    # zero_init_final_layer=True, F=1+0 is constant and dF/d(occupancy)=0, so the
    # occupancy would (correctly) carry no gradient -- masking the real path. A
    # sensitive functional is what training uses anyway.
    arch = ArchitectureConfig.from_spec("rung35_grad", 3, 16,
                                        descriptors=["cusp", "rung35"],
                                        zero_init_final_layer=False)
    model = AlecGGAModel.from_arch(arch, seed=0)
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                       feature_policy=FeaturePolicy.REASSEMBLE, max_cycles=4)

    def loss(m):
        return run_scf(cfg, m, data).total_energy ** 2

    def _flat(g):
        leaves = [np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(g)
                  if hasattr(x, "shape") and x.dtype.kind == "f"]
        return np.concatenate(leaves)

    g_live = _flat(eqx.filter_grad(loss)(model))
    assert np.all(np.isfinite(g_live)), "non-finite training gradient"
    assert np.any(np.abs(g_live) > 1e-12), "training gradient is identically zero"

    orig = r35.compute_rung35_occupancy
    monkeypatch.setattr(r35, "compute_rung35_occupancy",
                        lambda proj_ao, dm: jax.lax.stop_gradient(orig(proj_ao, dm)))
    g_det = _flat(eqx.filter_grad(loss)(model))
    assert not np.allclose(g_live, g_det, atol=1e-9), \
        "detaching the rung-3.5 occupancy did NOT change the gradient -> the path carries none"


# ---------------------------------------------------------------------------
# Multi-width rung-3.5 (DMRung35MultishellDescriptor), added 2026-08-06.
#
# The test set is shaped by the mutants that defeated an earlier draft of it:
# a "multishell" that IGNORES its widths, an off-by-one atom slice, a factor of
# two, a transposed spin axis, and a swapped column order all passed a battery
# built only from bounds / finiteness / size-consistency checks. Distinctness
# per width and an explicit pairing convention are therefore pinned directly.
# ---------------------------------------------------------------------------
def test_multishell_reduces_bitwise_to_single_width():
    """alphas=(ALPHA,) must reproduce the shipped single-width occupancy EXACTLY.

    This is what makes the generalization safe: the shipped descriptor is the
    len(alphas) == 1 member, not an approximation of it.
    """
    from xcquinox.pipeline.rung35 import (
        compute_projected_ao, compute_rung35_occupancy,
        compute_projected_ao_multishell, compute_rung35_multishell_occupancy)
    mol, dm, coords, _w = _h2()
    ref = np.asarray(compute_rung35_occupancy(
        jnp.asarray(compute_projected_ao(mol, coords, ALPHA)), jnp.asarray(dm)))
    new = np.asarray(compute_rung35_multishell_occupancy(
        jnp.asarray(compute_projected_ao_multishell(mol, coords, (ALPHA,))),
        jnp.asarray(dm)))
    assert new.shape == ref.shape, (new.shape, ref.shape)
    assert np.array_equal(ref, new), (
        f"single-width multishell is not bitwise identical to the shipped "
        f"occupancy: max|d| = {np.max(np.abs(ref - new)):.3e}")


def test_multishell_column_order_is_alpha_major_then_spin():
    """Pin the pairing convention explicitly.

    A spin-major implementation has the same shape, the same bounds and the
    same per-column statistics up to permutation, so nothing else in this file
    can distinguish it. Checked against both candidate orderings built from the
    single-width primitive.
    """
    from xcquinox.pipeline.rung35 import (
        compute_projected_ao, compute_rung35_occupancy,
        compute_projected_ao_multishell, compute_rung35_multishell_occupancy)
    alphas = (0.05, 0.2, 0.8)
    mol, dm, coords, _w = _h2()
    new = np.asarray(compute_rung35_multishell_occupancy(
        jnp.asarray(compute_projected_ao_multishell(mol, coords, alphas)),
        jnp.asarray(dm)))
    per = [np.asarray(compute_rung35_occupancy(
        jnp.asarray(compute_projected_ao(mol, coords, a)), jnp.asarray(dm)))
        for a in alphas]
    alpha_major = np.concatenate(per, axis=1)
    spin_major = np.concatenate(
        [np.stack([p[:, s] for p in per], axis=1) for s in (0, 1)], axis=1)
    assert np.allclose(new, alpha_major, atol=1e-14), "not alpha-major"
    assert not np.allclose(new, spin_major, atol=1e-8), (
        "alpha-major and spin-major are indistinguishable on this system, so "
        "this test has no power -- pick widths that separate them")


    # No ordering is asserted: the mean occupancy is NOT monotonic in the
    # projector width. Measured on H2/def2-svp it runs [0.254, 0.420, 0.253]
    # for alpha = 0.05 / 0.2 / 0.8, peaking at the intermediate width -- a broad
    # projector integrates a large volume of low density, a narrow one a tiny
    # volume, and the product is largest in between.


