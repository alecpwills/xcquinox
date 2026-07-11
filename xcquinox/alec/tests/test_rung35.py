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
    from xcquinox.alec.rung35 import compute_projected_ao
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
    from xcquinox.alec.rung35 import compute_projected_ao
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
    from xcquinox.alec.rung35 import (compute_projected_ao,
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
    from xcquinox.alec.rung35 import compute_rung35_occupancy
    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.standard_normal((20, 6)))
    f = lambda d: jnp.sum(compute_rung35_occupancy(A, d + d.T))
    dm = jnp.asarray(rng.standard_normal((6, 6)))
    grad = jax.grad(f)(dm)
    assert np.all(np.isfinite(np.asarray(grad))), "non-finite gradient wrt DM"


# ===========================================================================
# Phase 2: the DMRung35Descriptor (registration + delegation to rung35.py).
# ===========================================================================

def test_rung35_descriptor_registration():
    """Registered under "rung35" (matches the arch descriptor list), 2 features,
    gated on the precomputed "rung35_features" key, default alpha from rung35.py."""
    from xcquinox.alec.descriptors import DESCRIPTOR_REGISTRY, make_descriptor
    from xcquinox.alec.rung35 import DEFAULT_RUNG35_ALPHA
    assert "rung35" in DESCRIPTOR_REGISTRY
    d = make_descriptor("rung35")
    assert d.n_features == 2, d.n_features
    assert d.required_mol_keys == ("rung35_features",)
    assert float(d.alpha) == DEFAULT_RUNG35_ALPHA


def test_rung35_descriptor_compute_reads_precomputed_feature():
    """compute(mol_data) returns the precomputed occupancy (one-shot path)."""
    from xcquinox.alec.descriptors import make_descriptor
    d = make_descriptor("rung35")
    feat = jnp.ones((12, 2)) * 0.4
    assert jnp.array_equal(d.compute({"rung35_features": feat}), feat)


def test_rung35_descriptor_compute_from_dm_reassembles_occupancy():
    """The reassemble kernel recomputes the occupancy from the LIVE DM + the
    constant projected-AO matrix A (the self-consistent SCF path)."""
    from xcquinox.alec.descriptors import make_descriptor
    from xcquinox.alec.rung35 import compute_rung35_occupancy
    d = make_descriptor("rung35")
    rng = np.random.default_rng(1)
    A = jnp.asarray(rng.standard_normal((15, 5)))
    dm = jnp.asarray(rng.standard_normal((5, 5)))
    dm = dm + dm.T
    got = d.compute_from_dm(proj_ao=A, dm=dm)
    assert got.shape == (15, 2)
    assert jnp.allclose(got, compute_rung35_occupancy(A, dm))


def test_rung35_descriptor_alpha_configurable_and_static():
    """alpha is a static (hyperparameter) field, configurable per instance."""
    from xcquinox.alec.descriptors import make_descriptor
    d = make_descriptor("rung35", alpha=0.5)
    assert float(d.alpha) == 0.5
    assert d.n_features == 2


# ===========================================================================
# Phase 3: gated precompute + self-consistent SCF reassemble.
# ===========================================================================

def _h2_spec():
    from xcquinox.alec.config import MoleculeSpec
    return MoleculeSpec.from_dict(name="H2", atom="H 0 0 0; H 0 0 0.74",
                                  basis="def2-svp", charge=0, spin=0,
                                  atom_composition={"H": 2})


def test_precompute_populates_rung35_features_when_descriptor_present():
    """The gated precompute computes + stores both the constant projected-AO
    matrix A (rung35_proj_ao) and the one-shot occupancy (rung35_features),
    correctly shaped, finite, and bounded [0, 1]."""
    from xcquinox.alec.data import (precompute_fixed_density_data,
                                    clear_precompute_cache)
    from xcquinox.alec.descriptors import make_descriptor
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


def test_precompute_rung35_keys_none_without_descriptor():
    """Gated by required_mol_keys: with no rung-3.5 descriptor present the
    rung35 keys stay None, so the existing-arch precompute is byte-identical
    (in-flight safety)."""
    from xcquinox.alec.data import (precompute_fixed_density_data,
                                    clear_precompute_cache)
    from xcquinox.alec.descriptors import make_descriptor
    clear_precompute_cache()
    data = precompute_fixed_density_data(
        _h2_spec(), descriptors=(make_descriptor("cusp"),),
        required_keys=("cusp_features",))
    assert data.get("rung35_proj_ao") is None
    assert data.get("rung35_features") is None


def test_reassemble_manual_rung35_tracks_live_dm():
    """The manual-backend REASSEMBLE recomputes the occupancy from the LIVE DM
    (self-consistency, not frozen at the PBE value): a different DM gives a
    different occupancy, matching compute_rung35_occupancy(A, dm)."""
    from xcquinox.alec.solver import _reassemble_features
    from xcquinox.alec.descriptors import make_descriptor
    from xcquinox.alec.rung35 import compute_rung35_occupancy
    rng = np.random.default_rng(3)
    nao, N = 5, 12
    A = jnp.asarray(rng.standard_normal((N, nao)))
    S = jnp.eye(nao)
    d = make_descriptor("rung35")
    dm1 = jnp.asarray(rng.standard_normal((nao, nao))); dm1 = dm1 + dm1.T
    dm2 = jnp.asarray(rng.standard_normal((nao, nao))); dm2 = dm2 + dm2.T
    f1 = _reassemble_features((d,), dm=dm1, s_matrix=S, n_grid=N, rung35_proj_ao=A)
    f2 = _reassemble_features((d,), dm=dm2, s_matrix=S, n_grid=N, rung35_proj_ao=A)
    assert f1.shape == (N, 2) and f2.shape == (N, 2)
    assert jnp.allclose(f1, compute_rung35_occupancy(A, dm1))
    assert not jnp.allclose(f1, f2)  # genuinely tracks the live DM


def test_reassemble_on_grid_rung35_matches_occupancy():
    """The pyscfad-backend reassemble produces the rung-3.5 occupancy on its own
    grid. With a cached proj_ao A it uses it; without one it recomputes A on the
    grid (fallback) -- both give compute_rung35_occupancy(A, dm)."""
    from pyscf import gto
    from xcquinox.alec.solver_pyscfad import _reassemble_features_on_grid
    from xcquinox.alec.descriptors import make_descriptor
    from xcquinox.alec.rung35 import (compute_projected_ao,
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

def test_deep_rung35_archs_registered():
    """The new archs exist: deep_rung35_3x16 (cusp+rung35) replacing the leaky
    deep_combined; deep_rung35only_3x16 (rung35 alone) replacing deep_dm; + attn."""
    from xcquinox.alec.config import ARCHITECTURES
    for name in ("deep_rung35_3x16", "deep_rung35_attn_3x16", "deep_rung35only_3x16"):
        assert name in ARCHITECTURES, name
    a = ARCHITECTURES["deep_rung35_3x16"]
    assert [s.name for s in a.descriptors] == ["cusp", "rung35"]
    assert sum(d.n_features for d in a.materialize_descriptors()) == 4  # cusp 2 + rung35 2
    assert [type(d).__name__ for d in a.materialize_descriptors()] == \
        ["CuspDescriptor", "DMRung35Descriptor"]
    only = ARCHITECTURES["deep_rung35only_3x16"]
    assert [s.name for s in only.descriptors] == ["rung35"]
    assert sum(d.n_features for d in only.materialize_descriptors()) == 2
    assert ARCHITECTURES["deep_rung35_attn_3x16"].attention is True


def test_existing_leaky_archs_untouched_in_flight_safe():
    """In-flight safety: the deep_combined / deep_dm registry entries are
    byte-identical, so a pending in-flight array task still resolves them."""
    from xcquinox.alec.config import ARCHITECTURES
    assert [s.name for s in ARCHITECTURES["deep_combined_3x16"].descriptors] == \
        ["dm_statistics", "cusp"]
    assert [s.name for s in ARCHITECTURES["deep_combined_attn_3x16"].descriptors] == \
        ["dm_statistics", "cusp"]
    assert [s.name for s in ARCHITECTURES["deep_dm_3x16"].descriptors] == ["dm_statistics"]


def test_rung35_feeds_both_nets_xc_parity():
    """X/C PARITY (user hard requirement): the rung-3.5 channel feeds BOTH the
    exchange and correlation networks. With a non-zero-init model, perturbing the
    rung35 feature columns must change BOTH Fx and Fc (no X-only path)."""
    from xcquinox.alec.config import ArchitectureConfig
    from xcquinox.alec.models import AlecGGAModel
    arch = ArchitectureConfig.from_spec(
        "test_rung35_parity", 3, 16, descriptors=["cusp", "rung35"],
        zero_init_final_layer=False)
    model = AlecGGAModel.from_arch(arch, seed=0)
    assert model.xnet.n_extra_features == 4
    assert model.cnet.n_extra_features == 4
    rho = jnp.array([0.5, 0.3])
    sigma = jnp.array([0.1, 0.2])
    feat0 = jnp.zeros((2, 4))
    feat1 = feat0.at[:, 2:].set(0.7)  # perturb ONLY the rung35 columns (2, 3)
    assert not jnp.allclose(model.eval_Fx(rho, sigma, feat0),
                            model.eval_Fx(rho, sigma, feat1)), \
        "rung-3.5 feature does not affect the exchange net (X/C parity broken)"
    assert not jnp.allclose(model.eval_Fc(rho, sigma, feat0),
                            model.eval_Fc(rho, sigma, feat1)), \
        "rung-3.5 feature does not affect the correlation net (X/C parity broken)"


# ===========================================================================
# Phase 5: NaN robustness. The occupancy A^T P A is bounded [0,1] by
# construction (no division by rho / k_F anywhere), so it is NaN-safe; these
# pin that across extreme widths and the low-density (far-field) limit.
# ===========================================================================

def test_occupancy_alpha_extremes_finite_and_bounded():
    """At extreme projector widths (very tight 1e4, very diffuse 1e-3) the
    occupancy stays finite and in [0, 1]."""
    from xcquinox.alec.rung35 import (compute_projected_ao,
                                       compute_rung35_occupancy)
    mol, dm, coords, _w = _h2()
    for alpha in (1e4, 1e-3, 5.0):
        A = compute_projected_ao(mol, coords, alpha)
        n = np.asarray(compute_rung35_occupancy(jnp.asarray(A), jnp.asarray(dm)))
        assert np.all(np.isfinite(n)), f"alpha={alpha}: non-finite occupancy"
        assert n.min() >= -1e-9 and n.max() <= 1 + 1e-6, \
            f"alpha={alpha}: occupancy out of [0,1] ({n.min()}, {n.max()})"


def test_occupancy_far_field_vanishes_smoothly():
    """Low-density / far-from-molecule limit: A -> 0, so the occupancy -> 0
    smoothly (no NaN, no blow-up -- the rho->0 limit is benign by construction)."""
    from xcquinox.alec.rung35 import (compute_projected_ao,
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
    from xcquinox.alec.rung35 import (compute_projected_ao,
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


def test_v3_yamls_swept_to_rung35_not_combined():
    """In-flight-safe sweep swap: the v3 + full25 dfs_step7 YAMLs now train the
    rung-3.5 archs (NOT the leaky deep_combined/deep_dm), keep deep_cusp_3x16 as
    the control, and every swept arch resolves in the registry. The leaky entries
    remain in the registry (so the running array still resolves them) but are no
    longer in the sweep, so a fresh submit auto-trains rung-3.5."""
    import pathlib
    import yaml
    from xcquinox.alec.config import ARCHITECTURES
    root = pathlib.Path(__file__).resolve().parents[3]
    cfg_dir = root / "hpcjobs" / "configs"
    for fn in ("dfs_step7.svp_grid2_v3.yaml", "dfs_step7.svp_grid2_v3_full25.yaml"):
        cfg_path = cfg_dir / fn
        if not cfg_path.exists():
            # The cluster-config tree is optional in a source-only checkout;
            # skip rather than couple this physics test to its presence.
            pytest.skip(f"cluster config {fn} not present in this checkout")
        cfg = yaml.safe_load(cfg_path.read_text())
        archs = cfg["sweep"]["arch"]
        assert "deep_rung35_3x16" in archs, fn
        assert "deep_rung35_attn_3x16" in archs, fn
        assert "deep_rung35only_3x16" in archs, fn
        assert "deep_cusp_3x16" in archs, f"{fn}: control deep_cusp_3x16 dropped"
        assert "deep_combined_3x16" not in archs, f"{fn}: leaky deep_combined still swept"
        assert "deep_combined_attn_3x16" not in archs, fn
        assert "deep_dm_3x16" not in archs, f"{fn}: leaky deep_dm still swept"
        for a in archs:
            assert a in ARCHITECTURES, f"{fn}: swept arch {a!r} not in registry"
    # registry still resolves the leaky archs for the in-flight array.
    assert "deep_combined_3x16" in ARCHITECTURES and "deep_dm_3x16" in ARCHITECTURES


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
    from xcquinox.alec.rung35 import compute_projected_ao
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


@pytest.mark.slow
def test_occupancy_bounded_d_functions_rks_and_uks():
    """The [0,1] bound holds with d-shell basis functions, closed-shell (H2O) AND
    open-shell (OH/UKS, spin-resolved DM)."""
    from pyscf import dft
    from xcquinox.alec.rung35 import (compute_projected_ao,
                                       compute_rung35_occupancy)
    for spin in (0, 1):
        mol = _h2o_mol(spin)
        mf = (dft.UKS if spin else dft.RKS)(mol); mf.xc = "pbe"; mf.kernel()
        A = compute_projected_ao(mol, mf.grids.coords, ALPHA)
        n = np.asarray(compute_rung35_occupancy(jnp.asarray(A),
                                                jnp.asarray(mf.make_rdm1())))
        assert np.all(np.isfinite(n)), f"spin={spin} non-finite"
        assert n.min() >= -1e-9 and n.max() <= 1 + 1e-6, (spin, n.min(), n.max())


def _rung35_model_data(eri=True):
    from xcquinox.alec.config import ARCHITECTURES
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.data import (precompute_fixed_density_data,
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
    from xcquinox.alec.solver import (SolverConfig, SolverBackend, SolverMode,
                                      FeaturePolicy, run_scf)
    from xcquinox.alec.rung35 import compute_rung35_occupancy
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
    import xcquinox.alec.rung35 as r35
    from xcquinox.alec.config import ArchitectureConfig
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.solver import (SolverConfig, SolverBackend, SolverMode,
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


@pytest.mark.slow
def test_rung35_code_does_not_perturb_an_existing_arch():
    """Additivity (in-flight safety) at the SCF level: a non-rung35 arch
    (deep_cusp_3x16) precomputes NO rung35 keys and runs a FULL SCF to a finite
    energy with the rung-3.5 code present -- the rung-3.5 branch is never taken."""
    from xcquinox.alec.config import ARCHITECTURES
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.data import (precompute_fixed_density_data,
                                    clear_precompute_cache)
    from xcquinox.alec.solver import (SolverConfig, SolverBackend, SolverMode,
                                      FeaturePolicy, run_scf)
    clear_precompute_cache()
    arch = ARCHITECTURES["deep_cusp_3x16"]
    model = AlecGGAModel.from_arch(arch, seed=0)
    data = precompute_fixed_density_data(
        _h2_spec(), descriptors=arch.materialize_descriptors(),
        required_keys=("cusp_features", "eri"))
    assert data.get("rung35_proj_ao") is None and data.get("rung35_features") is None
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                       feature_policy=FeaturePolicy.REASSEMBLE, max_cycles=5)
    result = run_scf(cfg, model, data)
    assert np.isfinite(float(result.total_energy))
    assert np.asarray(result.features_used).shape[1] == 2  # cusp only, no rung35 cols


@pytest.mark.slow
def test_rung35_full_scf_pyscfad_runs_no_nan():
    """End-to-end smoke validating the Phase-3 threading: a deep_rung35
    (cusp+rung35) model runs a FULL pyscfad SCF under REASSEMBLE -- A is computed
    once on pyscfad's (pruned) grid and the occupancy is recomputed from the live
    DM each cycle. Asserts finite energy + DM + features (no NaN through the
    self-consistent loop)."""
    from xcquinox.alec.config import ARCHITECTURES
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.data import (precompute_fixed_density_data,
                                    clear_precompute_cache)
    from xcquinox.alec.solver import (SolverConfig, SolverBackend, SolverMode,
                                      FeaturePolicy, run_scf)
    clear_precompute_cache()
    arch = ARCHITECTURES["deep_rung35_3x16"]
    model = AlecGGAModel.from_arch(arch, seed=0)
    data = precompute_fixed_density_data(
        _h2_spec(), descriptors=arch.materialize_descriptors(),
        required_keys=("cusp_features", "rung35_features"))
    cfg = SolverConfig(backend=SolverBackend.PYSCFAD, mode=SolverMode.FULL,
                       feature_policy=FeaturePolicy.REASSEMBLE, max_cycles=10)
    result = run_scf(cfg, model, data)
    assert np.isfinite(float(result.total_energy)), "non-finite energy"
    assert np.all(np.isfinite(np.asarray(result.density_matrix))), "non-finite DM"
    assert np.all(np.isfinite(np.asarray(result.features_used))), "non-finite features"


@pytest.mark.slow
def test_rung35_full_scf_manual_runs_no_nan():
    """End-to-end smoke for the MANUAL backend: deep_rung35 FULL SCF under
    REASSEMBLE uses the precomputed A (precompute grid). Finite + no NaN."""
    from xcquinox.alec.config import ARCHITECTURES
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.data import (precompute_fixed_density_data,
                                    clear_precompute_cache)
    from xcquinox.alec.solver import (SolverConfig, SolverBackend, SolverMode,
                                      FeaturePolicy, run_scf)
    clear_precompute_cache()
    arch = ARCHITECTURES["deep_rung35_3x16"]
    model = AlecGGAModel.from_arch(arch, seed=0)
    data = precompute_fixed_density_data(
        _h2_spec(), descriptors=arch.materialize_descriptors(),
        required_keys=("cusp_features", "rung35_features", "eri"))  # manual FULL needs the ERI
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                       feature_policy=FeaturePolicy.REASSEMBLE, max_cycles=10)
    result = run_scf(cfg, model, data)
    assert np.isfinite(float(result.total_energy)), "non-finite energy"
    assert np.all(np.isfinite(np.asarray(result.density_matrix))), "non-finite DM"
