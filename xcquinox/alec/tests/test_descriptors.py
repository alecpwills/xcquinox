import numpy as np
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import ClassVar


# §13.2 item (1): DESCRIPTOR_REGISTRY contains both built-ins
def test_descriptor_registry_has_two_builtins():
    from xcquinox.alec.descriptors import DESCRIPTOR_REGISTRY
    assert "cusp" in DESCRIPTOR_REGISTRY
    assert "dm_statistics" in DESCRIPTOR_REGISTRY


# §13.2 item (2): make_descriptor registry roundtrip
def test_make_descriptor_roundtrip():
    from xcquinox.alec.descriptors import make_descriptor, CuspDescriptor, DMStatisticsDescriptor
    d_cusp = make_descriptor("cusp")
    d_dm = make_descriptor("dm_statistics")
    assert isinstance(d_cusp, CuspDescriptor)
    assert isinstance(d_dm, DMStatisticsDescriptor)


# §13.2 item (3): CuspDescriptor.n_features == 2
def test_cusp_n_features_value():
    from xcquinox.alec.descriptors import CuspDescriptor
    assert CuspDescriptor().n_features == 2


# §13.2 item (4): DMStatisticsDescriptor.n_features == 2
def test_dm_statistics_n_features_value():
    from xcquinox.alec.descriptors import DMStatisticsDescriptor
    # 3 -> 2 on 2026-08-06: dm_entropy removed (no usable gradient at any
    # converged density). This count sets the network input width.
    assert DMStatisticsDescriptor().n_features == 2


# §13.2 item (5): assemble_descriptor_features with empty tuple returns (N, 0)
def test_assemble_empty_returns_zero_width():
    from xcquinox.alec.descriptors import assemble_descriptor_features
    mol_data = {"rho_grid": jnp.ones((7,))}
    out = assemble_descriptor_features((), mol_data)
    assert out.shape == (7, 0)


# §13.2 item (6): assemble_descriptor_features with single descriptor returns correct shape
def test_assemble_single_descriptor_shape():
    from xcquinox.alec.descriptors import assemble_descriptor_features, make_descriptor
    d = make_descriptor("cusp")
    mol_data = {"cusp_features": jnp.ones((7, 2))}
    out = assemble_descriptor_features((d,), mol_data)
    assert out.shape == (7, 2)


# §13.2 item (7): assemble two descriptors concatenates left-to-right
def test_assemble_two_descriptors_concatenates():
    from xcquinox.alec.descriptors import assemble_descriptor_features, make_descriptor
    descriptors = (make_descriptor("dm_statistics"), make_descriptor("cusp"))
    mol_data = {
        "dm_features": jnp.ones((5, 3)),
        "cusp_features": jnp.ones((5, 2)) * 2.0,
    }
    out = assemble_descriptor_features(descriptors, mol_data)
    assert out.shape == (5, 5)
    assert jnp.allclose(out[:, :3], 1.0)
    assert jnp.allclose(out[:, 3:], 2.0)


# §13.2 item (8): dm BEFORE cusp invariant in ARCHITECTURES
def test_dm_before_cusp_invariant():
    from xcquinox.alec.config import ARCHITECTURES
    for name, cfg in ARCHITECTURES.items():
        descr_names = [d.name for d in cfg.descriptors]
        if "dm_statistics" in descr_names and "cusp" in descr_names:
            assert descr_names.index("dm_statistics") < descr_names.index("cusp"), (
                f"Architecture {name!r} lists cusp before dm_statistics"
            )


# §13.2 item (9): make_descriptor raises KeyError on unknown name
def test_make_descriptor_unknown():
    from xcquinox.alec.descriptors import make_descriptor
    with pytest.raises(KeyError):
        make_descriptor("not-a-descriptor")


# §13.2 item (10): list_descriptors returns sorted list
def test_list_descriptors_sorted():
    from xcquinox.alec.descriptors import list_descriptors
    names = list_descriptors()
    assert names == sorted(names)


# §13.2 item (11): CuspDescriptor.compute is differentiable
def test_cusp_compute_is_differentiable():
    from xcquinox.alec.descriptors import make_descriptor
    d = make_descriptor("cusp")

    def scalar(x):
        return jnp.sum(d.compute({"cusp_features": x}))

    grad_fn = jax.grad(scalar)
    g = grad_fn(jnp.ones((5, 2)))
    assert g.shape == (5, 2)
    assert jnp.all(jnp.isfinite(g))


# §13.2 item (12): DMStatisticsDescriptor.compute is differentiable
def test_dm_statistics_compute_is_differentiable():
    from xcquinox.alec.descriptors import make_descriptor
    d = make_descriptor("dm_statistics")

    def scalar(x):
        return jnp.sum(d.compute({"dm_features": x}))

    grad_fn = jax.grad(scalar)
    g = grad_fn(jnp.ones((5, 3)))
    assert g.shape == (5, 3)
    assert jnp.all(jnp.isfinite(g))


# §13.2 item (13): D-H1 __post_init__ rejects jax.Array on float field
def test_descriptor_post_init_rejects_jax_array_on_float_field():
    from xcquinox.alec.descriptors import (
        DESCRIPTOR_REGISTRY, Descriptor, register_descriptor, make_descriptor,
    )

    try:
        @register_descriptor("_fake_float_kwarg")
        class _FakeFloatKwargDescriptor(Descriptor):
            n_features: int = eqx.field(default=2, static=True)
            scale: float = eqx.field(default=1.0, static=True)
            required_mol_keys: ClassVar[tuple[str, ...]] = ()

            def compute(self, mol_data):
                return jnp.zeros((1, 2))

        with pytest.raises(TypeError, match="scale"):
            make_descriptor("_fake_float_kwarg", scale=jnp.array(1.0))
    finally:
        DESCRIPTOR_REGISTRY.pop("_fake_float_kwarg", None)


# §13.2 item (14): D-H3 rejects non-static field at registration time
def test_register_descriptor_rejects_non_static_field():
    from xcquinox.alec.descriptors import (
        DESCRIPTOR_REGISTRY, Descriptor, register_descriptor,
    )

    try:
        with pytest.raises(TypeError, match="static"):
            @register_descriptor("bad")
            class BadDescriptor(Descriptor):
                n_features: int = eqx.field(default=2, static=True)
                required_mol_keys: ClassVar[tuple[str, ...]] = ()
                trainable: jnp.ndarray = eqx.field(default_factory=lambda: jnp.zeros(3))

                def compute(self, mol_data):
                    return jnp.zeros((1, 2))
    finally:
        DESCRIPTOR_REGISTRY.pop("bad", None)


def test_dm_statistics_compute_from_dm_matches_precomputed():
    """compute_from_dm should produce the same tiled features as the
    precompute path for identical (dm, S) inputs."""
    from xcquinox.alec.descriptors import DMStatisticsDescriptor
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule

    desc = DMStatisticsDescriptor()
    data = precompute_fixed_density_data(
        h2_molecule(), descriptors=(desc,),
    )
    n_grid = data["rho_grid"].shape[0]

    features_kernel = desc.compute_from_dm(
        dm=data["dm_pbe"], s_matrix=data["s_matrix"], n_grid=n_grid,
    )
    features_precomp = data["dm_features"]
    assert features_kernel.shape == features_precomp.shape
    np.testing.assert_allclose(
        np.asarray(features_kernel),
        np.asarray(features_precomp),
        atol=1e-12, rtol=0.0,
    )


def test_dm_statistics_compute_from_dm_output_shape():
    """compute_from_dm returns (n_grid, 2) tiled features."""
    from xcquinox.alec.descriptors import DMStatisticsDescriptor

    desc = DMStatisticsDescriptor()
    dm = jnp.eye(2) * 0.5
    s = jnp.eye(2)
    features = desc.compute_from_dm(dm=dm, s_matrix=s, n_grid=17)
    assert features.shape == (17, 2)


# ---------------------------------------------------------------------------
# Per-spin-channel feature blocks: the symmetric doubled density
# diag(P_sigma, P_sigma) (Oliver and Perdew, Phys. Rev. A 20, 397 (1979)).
# ---------------------------------------------------------------------------

def test_doubled_spin_dm_places_the_channel_in_both_slots():
    from xcquinox.alec.descriptors import doubled_spin_dm
    rng = np.random.default_rng(20260821)
    p = jnp.asarray(rng.standard_normal((2, 4, 4)))
    for s in (0, 1):
        d = doubled_spin_dm(p, s)
        assert d.shape == (2, 4, 4)
        assert bool(jnp.all(d[0] == p[s]))
        assert bool(jnp.all(d[1] == p[s]))


def test_doubled_spin_dm_refuses_a_total_density_matrix():
    from xcquinox.alec.descriptors import doubled_spin_dm
    with pytest.raises(ValueError, match="spin-resolved"):
        doubled_spin_dm(jnp.zeros((4, 4)), 0)


def test_doubled_spin_dm_refuses_an_out_of_range_channel():
    from xcquinox.alec.descriptors import doubled_spin_dm
    with pytest.raises(ValueError, match="spin_channel"):
        doubled_spin_dm(jnp.zeros((2, 4, 4)), 2)


def test_cusp_per_channel_block_equals_the_shared_block():
    from xcquinox.alec.descriptors import CuspDescriptor
    d = CuspDescriptor()
    mol_data = {"cusp_features": jnp.arange(6.0).reshape(3, 2),
                "rho_grid": jnp.ones(3)}
    for s in (0, 1):
        got = d.compute_for_spin_channel(mol_data, s)
        assert bool(jnp.all(got == mol_data["cusp_features"]))


def test_rung35_per_channel_block_reads_its_own_spin_key():
    from xcquinox.alec.descriptors import DMRung35Descriptor
    d = DMRung35Descriptor()
    mol_data = {"rung35_features": jnp.zeros((3, 2)),
                "rung35_features_a": jnp.full((3, 2), 0.25),
                "rung35_features_b": jnp.full((3, 2), 0.75),
                "rho_grid": jnp.ones(3)}
    assert float(d.compute_for_spin_channel(mol_data, 0)[0, 0]) == 0.25
    assert float(d.compute_for_spin_channel(mol_data, 1)[0, 0]) == 0.75


def test_metagga_and_dm_statistics_declare_their_spin_keys():
    from xcquinox.alec.descriptors import (
        DMStatisticsDescriptor, DMRung35MultishellDescriptor,
        MetaGGAAlphaDescriptor, CuspDescriptor)
    assert DMStatisticsDescriptor.spin_mol_keys == (
        "dm_features_a", "dm_features_b")
    assert DMRung35MultishellDescriptor.spin_mol_keys == (
        "rung35ms_features_a", "rung35ms_features_b")
    assert MetaGGAAlphaDescriptor.spin_mol_keys == (
        "metagga_features_a", "metagga_features_b")
    assert CuspDescriptor.spin_mol_keys == ()


def test_per_channel_block_refuses_an_absent_spin_key():
    from xcquinox.alec.descriptors import DMRung35Descriptor
    d = DMRung35Descriptor()
    with pytest.raises(KeyError, match="rung35_features_a"):
        d.compute_for_spin_channel(
            {"rung35_features": jnp.zeros((3, 2)), "rung35_features_a": None}, 0)


def test_assemble_descriptor_features_spin_channel_preserves_column_order():
    from xcquinox.alec.descriptors import (
        assemble_descriptor_features, CuspDescriptor, DMRung35Descriptor)
    descriptors = (CuspDescriptor(), DMRung35Descriptor())
    mol_data = {
        "rho_grid": jnp.ones(3),
        "cusp_features": jnp.full((3, 2), 7.0),
        "rung35_features": jnp.zeros((3, 2)),
        "rung35_features_a": jnp.full((3, 2), 0.25),
        "rung35_features_b": jnp.full((3, 2), 0.75),
    }
    out = assemble_descriptor_features(descriptors, mol_data, spin_channel=0)
    assert out.shape == (3, 4)
    assert bool(jnp.all(out[:, :2] == 7.0))
    assert bool(jnp.all(out[:, 2:] == 0.25))


def test_assemble_descriptor_features_defaults_to_the_total_block():
    from xcquinox.alec.descriptors import (
        assemble_descriptor_features, DMRung35Descriptor)
    mol_data = {
        "rho_grid": jnp.ones(3),
        "rung35_features": jnp.full((3, 2), 0.5),
        "rung35_features_a": jnp.full((3, 2), 0.25),
        "rung35_features_b": jnp.full((3, 2), 0.75),
    }
    out = assemble_descriptor_features((DMRung35Descriptor(),), mol_data)
    assert bool(jnp.all(out == 0.5))


def test_assemble_descriptor_features_empty_descriptors_ignores_spin_channel():
    from xcquinox.alec.descriptors import assemble_descriptor_features
    mol_data = {"rho_grid": jnp.ones(5)}
    assert assemble_descriptor_features((), mol_data, spin_channel=1).shape == (5, 0)


def test_doubled_spin_dm_refuses_a_non_integral_channel():
    """Only an integral scalar selects a spin channel.

    Each value below satisfies ``spin_channel in (0, 1)`` without being an
    integral channel, and each was admitted before the guard: a Python ``True``
    indexed the array as a MASK, giving shape (2, 1, 2, 2, 2) on a (2, 2, 2)
    input and (2, 0, 2, 2, 2) for ``False``; ``np.bool_(True)`` returned the
    beta block and ``np.bool_(False)`` the alpha block; ``1.0`` and
    ``np.float64(1.0)`` were accepted outright.
    """
    from xcquinox.alec.descriptors import doubled_spin_dm, DMRung35Descriptor
    d = DMRung35Descriptor()
    mol_data = {"rung35_features": jnp.zeros((3, 2)),
                "rung35_features_a": jnp.full((3, 2), 0.25),
                "rung35_features_b": jnp.full((3, 2), 0.75)}
    for bad in (True, False, np.bool_(True), np.bool_(False),
                1.0, 0.0, np.float64(1.0), "1", None):
        with pytest.raises(ValueError, match="spin_channel"):
            doubled_spin_dm(jnp.zeros((2, 2, 2)), bad)
        with pytest.raises(ValueError, match="spin_channel"):
            d.compute_for_spin_channel(mol_data, bad)


def test_doubled_spin_dm_accepts_a_numpy_integer_channel():
    """The guard rejects non-integral scalars without rejecting numpy integers."""
    from xcquinox.alec.descriptors import doubled_spin_dm, DMRung35Descriptor
    p = jnp.asarray(np.arange(8.0).reshape(2, 2, 2))
    out = doubled_spin_dm(p, np.int64(1))
    assert out.shape == (2, 2, 2)
    assert bool(jnp.all(out[0] == p[1])) and bool(jnp.all(out[1] == p[1]))
    got = DMRung35Descriptor().compute_for_spin_channel(
        {"rung35_features_a": jnp.full((3, 2), 0.25),
         "rung35_features_b": jnp.full((3, 2), 0.75)}, np.int64(1))
    assert float(got[0, 0]) == 0.75


# ---------------------------------------------------------------------------
# Density-matrix dependence is declared, not inferred from the key tuple.
# ---------------------------------------------------------------------------

def test_density_matrix_dependence_flags_match_the_descriptor_family():
    from xcquinox.alec.descriptors import (
        CuspDescriptor, DMStatisticsDescriptor, DMRung35Descriptor,
        DMRung35MultishellDescriptor, MetaGGAAlphaDescriptor)
    assert CuspDescriptor.density_matrix_dependent is False
    for cls in (DMStatisticsDescriptor, DMRung35Descriptor,
                DMRung35MultishellDescriptor, MetaGGAAlphaDescriptor):
        assert cls.density_matrix_dependent is True, cls.__name__
        assert len(cls.spin_mol_keys) == 2, cls.__name__


def test_dm_dependent_descriptor_without_spin_keys_is_refused_at_definition():
    """A density-matrix descriptor that declares no per-spin keys would silently
    fall back to the shared block, which is the defect the doubled density
    removes; the class must not be definable."""
    from xcquinox.alec.descriptors import Descriptor
    with pytest.raises(TypeError, match="_UndeclaredSpinKeys"):
        class _UndeclaredSpinKeys(Descriptor):
            density_matrix_dependent: ClassVar[bool] = True
            n_features: int = eqx.field(default=1, static=True)

            def compute(self, mol_data):
                return jnp.zeros((1, 1))


def test_spin_mol_keys_of_the_wrong_width_are_refused_at_definition():
    """One name per spin channel. Under an emptiness-only check a one-name tuple
    defined without complaint and raised ``IndexError: tuple index out of range``
    when the beta channel was requested (measured), so the width belongs to the
    declaration rather than to the first call that trips over it."""
    from xcquinox.alec.descriptors import Descriptor
    for keys in (("rung35_features_a",),
                 ("a_key", "b_key", "extra_key")):
        with pytest.raises(TypeError, match="_WrongWidthSpinKeys") as excinfo:
            class _WrongWidthSpinKeys(Descriptor):
                density_matrix_dependent: ClassVar[bool] = True
                spin_mol_keys: ClassVar[tuple[str, ...]] = keys
                n_features: int = eqx.field(default=1, static=True)

                def compute(self, mol_data):
                    return jnp.zeros((1, 1))

        assert "spin_mol_keys" in str(excinfo.value)


def test_dm_dependent_descriptor_with_cleared_spin_keys_raises_at_use():
    """The same condition reached by post-definition mutation raises at use
    rather than returning the shared block."""
    from xcquinox.alec.descriptors import DMRung35Descriptor
    d = DMRung35Descriptor()
    original = DMRung35Descriptor.spin_mol_keys
    DMRung35Descriptor.spin_mol_keys = ()
    try:
        with pytest.raises(TypeError, match="spin_mol_keys"):
            d.compute_for_spin_channel({"rung35_features": jnp.zeros((3, 2))}, 0)
    finally:
        DMRung35Descriptor.spin_mol_keys = original
    assert DMRung35Descriptor.spin_mol_keys == ("rung35_features_a",
                                                "rung35_features_b")


def test_geometry_only_descriptor_is_definable_without_spin_keys():
    from xcquinox.alec.descriptors import Descriptor

    class _GeometryOnly(Descriptor):
        n_features: int = eqx.field(default=1, static=True)

        def compute(self, mol_data):
            return mol_data["geom_features"]

    assert _GeometryOnly.density_matrix_dependent is False
    block = _GeometryOnly().compute_for_spin_channel(
        {"geom_features": jnp.full((3, 1), 4.0)}, 1)
    assert bool(jnp.all(block == 4.0))


# ---------------------------------------------------------------------------
# What the descriptor kernels return when handed diag(P_sigma, P_sigma).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def o_atom_mol():
    """O atom (sto-3g) from the shared fixture spec, with no SCF."""
    from pyscf import gto
    from xcquinox.alec.tests.fixtures.molecules import o_atom
    spec = o_atom()
    return gto.M(atom=spec.atom, basis=spec.basis, charge=spec.charge,
                 spin=spec.spin, verbose=0)


@pytest.fixture(scope="module")
def o_atom_uks(o_atom_mol):
    """O atom (sto-3g, UKS/PBE, grid level 1): a spin-resolved DM with
    P_alpha != P_beta, so a per-channel claim cannot pass by symmetry."""
    from pyscf import dft
    mf = dft.UKS(o_atom_mol)
    mf.xc = "pbe"
    mf.grids.level = 1
    mf.kernel()
    dm = jnp.asarray(mf.make_rdm1())
    assert float(jnp.abs(dm[0] - dm[1]).max()) > 1e-3
    return o_atom_mol, mf.grids.coords, dm


def test_doubled_dm_rung35_occupancy_carries_one_channel_in_both_slots(o_atom_uks):
    """n(diag(P_s, P_s)) = [n_s, n_s], the rung-3.5 ingredient of the
    spin-unpolarized system the Oliver-Perdew relation refers to."""
    from xcquinox.alec.descriptors import doubled_spin_dm
    from xcquinox.alec.rung35 import (compute_projected_ao,
                                      compute_rung35_occupancy)
    mol, coords, dm = o_atom_uks
    proj = compute_projected_ao(mol, coords)
    occ_phys = np.asarray(compute_rung35_occupancy(proj, dm))
    for s in (0, 1):
        occ_d = np.asarray(compute_rung35_occupancy(proj, doubled_spin_dm(dm, s)))
        assert occ_d.shape == occ_phys.shape
        # measured deviation 2.22e-16 in both channels, on 4328 grid points
        np.testing.assert_allclose(occ_d[:, 0], occ_phys[:, s], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(occ_d[:, 1], occ_phys[:, s], rtol=0.0, atol=1e-12)
        # Bessel bound preserved. At this projector width the occupancy is
        # dominated by the 1s core and the measured ranges reproduced to the
        # digits shown across four initial guesses (minao, 1e, atom, huckel):
        # [3.96e-04, 7.49e-01] alpha and [1.28e-04, 7.49e-01] beta.
        assert occ_d.min() >= 0.0 and occ_d.max() <= 1.0


def test_doubled_dm_multishell_occupancy_keeps_the_alpha_major_layout(o_atom_uks):
    """Per width w, the doubled block holds [n_s(w), n_s(w)] in the
    alpha-major-then-spin column order of the physical block."""
    from xcquinox.alec.descriptors import doubled_spin_dm
    from xcquinox.alec.rung35 import (compute_projected_ao_multishell,
                                      compute_rung35_multishell_occupancy,
                                      DEFAULT_RUNG35_MULTISHELL_ALPHAS)
    mol, coords, dm = o_atom_uks
    n_w = len(DEFAULT_RUNG35_MULTISHELL_ALPHAS)
    proj = compute_projected_ao_multishell(mol, coords)
    ms_phys = np.asarray(compute_rung35_multishell_occupancy(proj, dm))
    assert ms_phys.shape[1] == 2 * n_w
    for s in (0, 1):
        ms_d = np.asarray(compute_rung35_multishell_occupancy(
            proj, doubled_spin_dm(dm, s)))
        assert ms_d.shape == ms_phys.shape
        for w in range(n_w):
            for slot in (0, 1):
                # measured deviation 2.22e-16 over every width and slot
                np.testing.assert_allclose(ms_d[:, 2 * w + slot],
                                           ms_phys[:, 2 * w + s],
                                           rtol=0.0, atol=1e-12)
        # Bessel bound preserved. The alpha range reproduced as
        # [2.45e-07, 9.54e-01] across four initial guesses, and the beta minimum
        # as 3.23e-08; the beta MAXIMUM does not reproduce -- the singly occupied
        # beta 2p of the O atom is orientation-degenerate, so the converged
        # solution selects an arbitrary member of that set and the narrowest
        # projector reports a peak anywhere in 0.949-0.954. Only the bound is
        # asserted, which is why the spread does not weaken the test.
        assert ms_d.min() >= 0.0 and ms_d.max() <= 1.0


def _fractional_occupation_dm(mol):
    """A NON-idempotent spin-resolved DM with natural occupations strictly
    inside (0, 1), built as ``P_sigma = C diag(f_sigma) C^T`` on orbitals
    orthonormal in the AO metric (``C = S^{-1/2} Q``, Q orthogonal). ``P`` is
    invariant under column sign flips of ``Q``, so it is reproducible across
    LAPACK sign conventions."""
    s_matrix = np.asarray(mol.intor("int1e_ovlp"))
    w, u = np.linalg.eigh(s_matrix)
    x = u @ np.diag(w ** -0.5) @ u.T
    q, _ = np.linalg.qr(np.random.default_rng(20260821).standard_normal(
        (s_matrix.shape[0], s_matrix.shape[0])))
    c = x @ q
    occ_a = np.array([0.95, 0.85, 0.60, 0.45, 0.30])
    occ_b = np.array([0.90, 0.70, 0.50, 0.35, 0.15])
    dm = np.stack([c @ np.diag(occ_a) @ c.T, c @ np.diag(occ_b) @ c.T])
    return dm, s_matrix, occ_a, occ_b


def test_doubled_dm_statistics_are_the_channel_statistics(o_atom_mol):
    """The dm_statistics block of diag(P_s, P_s) is the pair (per-spin
    idempotency error of P_s, off-diagonal norm of P_s); the factor 2 in the
    aggregated total cancels between the off-diagonal norm and its trace."""
    from xcquinox.alec.descriptors import doubled_spin_dm
    from xcquinox.features import compute_dm_features_array
    dm, s_matrix, occ_a, occ_b = _fractional_occupation_dm(o_atom_mol)
    w, u = np.linalg.eigh(s_matrix)
    root_s = u @ np.diag(np.sqrt(w)) @ u.T
    for s, occ in ((0, occ_a), (1, occ_b)):
        nat = np.linalg.eigvalsh(root_s @ dm[s] @ root_s)
        np.testing.assert_allclose(nat, np.sort(occ), rtol=1e-12, atol=0.0)
        assert nat.min() > 0.0 and nat.max() < 1.0

    # measured on this construction: the doubled-DM block against the per-spin
    # oracle agrees to 6.0e-16 / 2.0e-16 relative (idempotency) and
    # 1.5e-13 / 1.8e-13 relative (off-diagonal norm)
    expected = {0: (5.7828803759233306e-02, 1.7604057570571247e-01),
                1: (7.0508386631186415e-02, 2.2526539629454820e-01)}
    for s in (0, 1):
        block = np.asarray(compute_dm_features_array(
            doubled_spin_dm(jnp.asarray(dm), s), jnp.asarray(s_matrix)))
        p_s = dm[s]
        residual = p_s @ s_matrix @ p_s - p_s
        idempotency = float((residual * residual).sum()
                            / (np.trace(p_s @ s_matrix) + 1e-12))
        off = p_s - np.diag(np.diag(p_s))
        off_diag_norm = float(np.sqrt((off * off).sum()) / np.trace(p_s))
        assert idempotency > 1e-3, "the reference DM must not be idempotent"
        np.testing.assert_allclose(block, [idempotency, off_diag_norm],
                                   rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(block, expected[s], rtol=1e-12, atol=0.0)
