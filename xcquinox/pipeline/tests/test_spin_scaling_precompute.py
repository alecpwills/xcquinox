"""Per-channel descriptor blocks on precomputed molecule data.

Every UKS exchange evaluation is posed on the symmetric doubled density
diag(P_sigma, P_sigma) (Oliver and Perdew, Phys. Rev. A 20, 397 (1979)). These
tests pin what precompute stores for that system: the channel occupancy in both
rung-3.5 spin slots, the iso-orbital indicator at (2 rho_sigma, 4 sigma_sigma
sigma, 2 tau_sigma), the density-matrix statistics of diag(P_sigma, P_sigma),
and the per-spin kinetic-energy density itself.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from xcquinox.pipeline.config import MoleculeSpec
from xcquinox.pipeline.data import MoleculeData, precompute_fixed_density_data
from xcquinox.pipeline.descriptors import (
    DMRung35Descriptor, DMRung35MultishellDescriptor, DMStatisticsDescriptor,
    MetaGGAAlphaDescriptor, assemble_descriptor_features)
from xcquinox.pipeline.pyscf_determinism import pin_small_rho_cutoff


_ALL_DM_DESCRIPTORS = (DMStatisticsDescriptor(), DMRung35Descriptor(),
                       DMRung35MultishellDescriptor(), MetaGGAAlphaDescriptor())


def _precompute(name, atom, spin, composition, descriptors, basis="def2-svp",
                grid_level=1):
    keys = tuple(sorted({k for d in descriptors for k in d.required_mol_keys}))
    return precompute_fixed_density_data(
        MoleculeSpec(name=name, atom=atom, basis=basis, charge=0, spin=spin,
                     atom_composition=composition, grid_level=grid_level),
        required_keys=keys, descriptors=descriptors)


def _spin_grid(md, s):
    """(rho_sigma, sigma_sigma_sigma) for one spin channel of an open shell."""
    ao = np.asarray(md["ao_grid_deriv"])
    d = np.asarray(md["dm_pbe"])[s]
    rho = np.einsum("pi,ij,pj->p", ao[0], d, ao[0])
    gx = 2 * np.einsum("pi,ij,pj->p", ao[1], d, ao[0])
    gy = 2 * np.einsum("pi,ij,pj->p", ao[2], d, ao[0])
    gz = 2 * np.einsum("pi,ij,pj->p", ao[3], d, ao[0])
    return rho, gx ** 2 + gy ** 2 + gz ** 2


def test_open_shell_precompute_populates_every_per_channel_block():
    md = _precompute("Li", "Li 0 0 0", 1, (("Li", 1),), _ALL_DM_DESCRIPTORS)
    n = int(np.asarray(md["grid_weights"]).shape[0])
    for key, width in (("dm_features", 2), ("rung35_features", 2),
                       ("rung35ms_features", 6), ("metagga_features", 1)):
        for suffix in ("_a", "_b"):
            block = md[key + suffix]
            assert block is not None, key + suffix
            assert np.asarray(block).shape == (n, width), key + suffix
            assert np.all(np.isfinite(np.asarray(block))), key + suffix
    for key in ("tau_spin_a", "tau_spin_b"):
        assert np.asarray(md[key]).shape == (n,), key


def test_closed_shell_precompute_leaves_every_per_channel_block_none():
    md = _precompute("H2", "H 0 0 0; H 0 0 0.74", 0, (("H", 2),),
                     _ALL_DM_DESCRIPTORS)
    for key in ("dm_features_a", "dm_features_b", "rung35_features_a",
                "rung35_features_b", "rung35ms_features_a",
                "rung35ms_features_b", "metagga_features_a",
                "metagga_features_b", "tau_spin_a", "tau_spin_b"):
        assert md[key] is None, key


def test_rung35_per_channel_block_is_the_channel_occupancy_in_both_slots():
    md = _precompute("Li", "Li 0 0 0", 1, (("Li", 1),), (DMRung35Descriptor(),))
    tot = np.asarray(md["rung35_features"])
    for s, suffix in ((0, "_a"), (1, "_b")):
        block = np.asarray(md["rung35_features" + suffix])
        np.testing.assert_allclose(block[:, 0], block[:, 1], rtol=0, atol=1e-14)
        np.testing.assert_allclose(block[:, 0], tot[:, s], rtol=0, atol=1e-14)
        assert float(np.min(block)) > -1e-12
        assert float(np.max(block)) < 1.0 + 1e-12


def test_metagga_per_channel_alpha_uses_the_doubled_ingredients():
    from pyscf import gto, dft
    from xcquinox.pipeline.metagga import compute_alpha
    md = _precompute("Li", "Li 0 0 0", 1, (("Li", 1),),
                     (MetaGGAAlphaDescriptor(),))
    mol = gto.M(atom="Li 0 0 0", basis="def2-svp", spin=1, verbose=0)
    mf = dft.UKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 1
    # The oracle prunes its grid at the precompute's density cutoff (pyscf
    # 2.14's class default keeps every point; the precompute pins 1e-7).
    pin_small_rho_cutoff(mf)
    mf.kernel()
    # Same molecule, basis, spin, functional, grid level and density cutoff as
    # the precompute, so the two grids are the same set of points; assert it
    # rather than assume it, since a shape mismatch would otherwise surface as
    # a broadcast error.
    assert mf.grids.coords.shape[0] == np.asarray(md["grid_weights"]).shape[0]
    ao2 = mf._numint.eval_ao(mol, mf.grids.coords, deriv=2)
    dm = np.asarray(md["dm_pbe"])
    for s, suffix in ((0, "_a"), (1, "_b")):
        tau_ref = mf._numint.eval_rho(mol, ao2, dm[s], xctype="MGGA")[5]
        tau_spin = np.asarray(md["tau_spin" + suffix])
        # The stored per-spin tau against an independent contraction (pyscf's
        # MGGA rho slot on a deriv=2 AO evaluation): 1.4e-14 Ha/bohr^3 on
        # |tau| <= 24.7, i.e. the two summation orders round differently and
        # nothing else.
        np.testing.assert_allclose(tau_spin, tau_ref, rtol=0, atol=1e-9)
        rho_s, sigma_ss = _spin_grid(md, s)
        got = np.asarray(md["metagga_features" + suffix])[:, 0]
        # The block is the indicator AT THE DOUBLED INGREDIENTS, composed from
        # the stored per-spin tau. Exact: the doubled density matrix scales
        # every contracted term by 2, which is exact in binary, so both sides
        # evaluate compute_alpha on bit-identical arguments.
        np.testing.assert_allclose(
            got,
            np.asarray(compute_alpha(jnp.asarray(2.0 * rho_s),
                                     jnp.asarray(4.0 * sigma_ss),
                                     jnp.asarray(2.0 * tau_spin))),
            rtol=0, atol=0)
        # Rebuilt from pyscf's tau the same comparison is ill-conditioned in the
        # exponential tail: alpha subtracts tau_W from tau, and at
        # rho_sigma = 2.6e-14 those agree to |tau - tau_W|/tau = 1.7e-16 (total
        # cancellation), so dividing the rounding residue by
        # tau_unif = 2.0e-22 leaves an O(1e-8) difference that also moves
        # between SCF runs. Where density remains (rho_sigma > 1e-8, 4524 of
        # 4864 points for alpha and 3456 for beta) the two tau paths give alpha
        # to 1.5e-11, against alpha values reaching 6.24.
        expect = np.asarray(compute_alpha(jnp.asarray(2.0 * rho_s),
                                          jnp.asarray(4.0 * sigma_ss),
                                          jnp.asarray(2.0 * tau_ref)))
        keep = rho_s > 1e-8
        np.testing.assert_allclose(got[keep], expect[keep], rtol=0, atol=1e-9)
    # Li's beta channel holds one orbital, so its doubled system is a
    # single-orbital (iso-orbital) density: tau = tau_W and the raw indicator
    # vanishes identically; what is stored is the smooth positive part
    # (width 1e-5, floor 5e-6) of the cancellation residue of tau - tau_W
    # divided by tau_unif in the density tail. Measured max residue between
    # 5.8e-08 and 2.05e-07 over multi-threaded processes (1.07e-07,
    # reproducible to the bit, under single-thread BLAS), so the stored
    # column is at most 5e-6 + 1.1e-7 and the 6e-6 bound sits ~5x above the
    # residue's contribution, while the alpha channel reaches 6.24; the bound
    # therefore also refuses a beta block built from the physical total
    # density.
    from xcquinox.pipeline.metagga import _ALPHA_SMOOTHING_WIDTH
    assert float(np.max(np.asarray(md["metagga_features_b"]))) < (
        0.5 * _ALPHA_SMOOTHING_WIDTH + 1e-6)
    assert float(np.min(np.asarray(md["metagga_features_b"]))) > 0.0
    assert float(np.max(np.asarray(md["metagga_features_a"]))) > 1.0


def test_assemble_descriptor_features_reads_the_precomputed_blocks():
    md = _precompute("Li", "Li 0 0 0", 1, (("Li", 1),), _ALL_DM_DESCRIPTORS)
    n = int(np.asarray(md["grid_weights"]).shape[0])
    width = sum(d.n_features for d in _ALL_DM_DESCRIPTORS)
    for spin in (0, 1):
        block = assemble_descriptor_features(_ALL_DM_DESCRIPTORS, md,
                                             spin_channel=spin)
        assert block.shape == (n, width)
    total = assemble_descriptor_features(_ALL_DM_DESCRIPTORS, md)
    a = assemble_descriptor_features(_ALL_DM_DESCRIPTORS, md, spin_channel=0)
    assert float(np.max(np.abs(np.asarray(total) - np.asarray(a)))) > 1e-6


def test_every_per_spin_grid_key_is_declared_and_padded():
    from xcquinox.pipeline.padding import _PAD_GRID_EDGE
    per_spin = {k for k in MoleculeData.__annotations__
                if k.endswith(("_features_a", "_features_b"))
                or k in ("tau_spin_a", "tau_spin_b")}
    assert len(per_spin) == 10, sorted(per_spin)
    missing = sorted(per_spin - set(_PAD_GRID_EDGE))
    assert not missing, f"padding._PAD_GRID_EDGE is missing {missing}"


# ---------------------------------------------------------------------------
# The live per-channel closures: the single place the doubled-density
# convention is implemented for a density matrix that is not the precompute's.
# ---------------------------------------------------------------------------

def test_live_uks_feature_closures_reproduce_the_precomputed_blocks():
    """The live map P -> f_sigma(P) evaluated at the PBE density matrix must
    return what precompute stored, or the potential belongs to a different
    functional than the energy.

    Agreement is BITWISE for every column linear in the density matrix -- the
    density-matrix statistics and both rung-3.5 occupancies, 0 differing
    elements of 4864 x 10 in each of the three blocks. The iso-orbital
    indicator is not linear: precompute contracts rho and sigma with numpy and
    the live closure with JAX, two summation orders of the same contraction
    that differ by at most 4.7e-16 relative (2 ulp), and
    alpha = (tau - sigma/(8 rho))/tau_unif divides that perturbation by
    tau_unif, amplifying it by tau/tau_unif -- 3.8e6 (alpha channel), 9.0e7
    (beta) and 6.0e6 (total) at the outermost grid points, where the density
    has decayed to rho_sigma ~ 1e-14 and the numerator cancels completely. The
    indicator column is therefore checked where the density is resolved, and
    everywhere against that amplification.
    """
    from xcquinox.pipeline.descriptors import doubled_spin_dm
    from xcquinox.pipeline.metagga import compute_tau_from_dm
    from xcquinox.pipeline.solver import make_uks_feature_fns
    md = _precompute("Li", "Li 0 0 0", 1, (("Li", 1),), _ALL_DM_DESCRIPTORS)
    n = int(np.asarray(md["grid_weights"]).shape[0])
    fa, fb, ft = make_uks_feature_fns(
        descriptors=_ALL_DM_DESCRIPTORS,
        ao_deriv=jnp.asarray(md["ao_grid_deriv"]),
        s_matrix=jnp.asarray(md["s_matrix"]),
        n_grid=n,
        cusp_features=md.get("cusp_features"),
        rung35_proj_ao=md.get("rung35_proj_ao"),
        rung35ms_proj_ao=md.get("rung35ms_proj_ao"),
    )
    P0 = jnp.asarray(md["dm_pbe"])
    # Column layout: the iso-orbital indicator is the single trailing column,
    # everything before it is linear in the density matrix.
    assert isinstance(_ALL_DM_DESCRIPTORS[-1], MetaGGAAlphaDescriptor)
    assert _ALL_DM_DESCRIPTORS[-1].n_features == 1
    n_linear = sum(d.n_features for d in _ALL_DM_DESCRIPTORS[:-1])
    ao_grad = jnp.asarray(md["ao_grid_deriv"])[1:4]
    cases = (
        ("alpha", np.asarray(fa(P0)),
         np.asarray(assemble_descriptor_features(_ALL_DM_DESCRIPTORS, md,
                                                 spin_channel=0)),
         doubled_spin_dm(P0, 0), 2.0 * _spin_grid(md, 0)[0]),
        ("beta", np.asarray(fb(P0)),
         np.asarray(assemble_descriptor_features(_ALL_DM_DESCRIPTORS, md,
                                                 spin_channel=1)),
         doubled_spin_dm(P0, 1), 2.0 * _spin_grid(md, 1)[0]),
        ("total", np.asarray(ft(P0)),
         np.asarray(assemble_descriptor_features(_ALL_DM_DESCRIPTORS, md)),
         P0, np.asarray(md["rho_grid"])),
    )
    for label, live, ref, dm_doubled, rho_total in cases:
        assert live.shape == (n, n_linear + 1), label
        np.testing.assert_allclose(live[:, :n_linear], ref[:, :n_linear],
                                   rtol=0, atol=0, err_msg=label)
        gap = np.abs(live[:, -1] - ref[:, -1])
        # Resolved density (rho_total > 2e-8, i.e. rho_sigma > 1e-8 for a
        # channel -- the cut the stored blocks are already pinned against):
        # 2.4e-11 to 7.8e-11 measured worst over eight processes (the PBE
        # reference density moves between them), against indicator values
        # reaching 9.05, so the bound sits at least 12.9x above the floor. It
        # refuses a block taken from the wrong channel or from the physical
        # density, which differ here by 6.2 and 9.0.
        resolved = rho_total > 2e-8
        assert int(resolved.sum()) > n // 2, (label, int(resolved.sum()))
        assert float(gap[resolved].max()) < 1e-9, (label,
                                                   float(gap[resolved].max()))
        # Everywhere, including the exponential tail: the residual is the
        # ingredient rounding times the amplification. tau/tau_unif is the
        # amplification factor of alpha = (tau - tau_W)/tau_unif (SCAN,
        # PRL 115, 036402 (2015), Eq. 2); floored at 1 so a well-conditioned
        # point is held to 1e-12 absolute. Measured worst ratio 1.1e-14 to
        # 3.9e-14 over eight processes, at least 26x under the bound.
        tau = np.asarray(compute_tau_from_dm(ao_grad, dm_doubled))
        tau_unif = (0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
                    * np.maximum(rho_total, 1e-30) ** (5.0 / 3.0))
        amplification = tau / np.maximum(tau_unif, 1e-30)
        ratio = float((gap / np.maximum(amplification, 1.0)).max())
        assert ratio < 1e-12, (label, ratio)


def test_live_uks_feature_closures_collapse_at_a_closed_shell_density():
    """rho_a = rho_b makes the three blocks identical -- the structural reason
    every closed-shell number is unchanged by the exact spin scaling. Bitwise
    in eager evaluation; under jax.jit to the stated rounding bound."""
    from xcquinox.pipeline.solver import make_uks_feature_fns
    md = _precompute("H2O", "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469",
                     0, (("O", 1), ("H", 2)), _ALL_DM_DESCRIPTORS)
    fa, fb, ft = make_uks_feature_fns(
        descriptors=_ALL_DM_DESCRIPTORS,
        ao_deriv=jnp.asarray(md["ao_grid_deriv"]),
        s_matrix=jnp.asarray(md["s_matrix"]),
        n_grid=int(np.asarray(md["grid_weights"]).shape[0]),
        cusp_features=md.get("cusp_features"),
        rung35_proj_ao=md.get("rung35_proj_ao"),
        rung35ms_proj_ao=md.get("rung35ms_proj_ao"),
    )
    from xcquinox.pipeline.metagga import compute_tau_from_dm
    D = jnp.asarray(md["dm_pbe"])
    half = 0.5 * D
    P0 = jnp.stack([half, half], axis=0)
    # Eager evaluation: exact by construction. 0.5 * D and half + half are
    # exact binary scalings, so diag(P_sigma, P_sigma) reproduces the matrix and
    # 2 rho_a / 4 sigma_aa reproduce rho_tot / sigma_tot bit for bit.
    a, b, t = np.asarray(fa(P0)), np.asarray(fb(P0)), np.asarray(ft(P0))
    np.testing.assert_allclose(a, b, rtol=0, atol=0)
    np.testing.assert_allclose(a, t, rtol=0, atol=0)
    # Under jax.jit the three closures are three different programs, and XLA
    # may fuse and order their reductions differently, so bitwise agreement is
    # not guaranteed there, and what is measured depends on the process
    # configuration. On this molecule the three jitted blocks agree bitwise
    # with and without JAX_PLATFORMS set; jit against eager, the
    # density-matrix-linear columns agree to 8.8e-47 (idempotency error) here
    # and to one ulp, 5.6e-17 in off_diag_norm, on the O atom, and the
    # indicator is bitwise under JAX_PLATFORMS=cpu but differs by 1.4e-11 to
    # 2.0e-11 where the density is resolved and 5.4e-10 to 6.0e-10 in the tail
    # without it, i.e. 9.5e-16 after dividing by its amplification
    # tau/tau_unif -- the same ingredient rounding, and the same bounds, as
    # the precompute comparison above.
    assert isinstance(_ALL_DM_DESCRIPTORS[-1], MetaGGAAlphaDescriptor)
    n_linear = sum(d.n_features for d in _ALL_DM_DESCRIPTORS[:-1])
    rho = np.asarray(md["rho_grid"])
    tau = np.asarray(compute_tau_from_dm(jnp.asarray(md["ao_grid_deriv"])[1:4], D))
    tau_unif = (0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
                * np.maximum(rho, 1e-30) ** (5.0 / 3.0))
    amplification = tau / np.maximum(tau_unif, 1e-30)
    resolved = rho > 2e-8
    assert int(resolved.sum()) > rho.shape[0] // 2
    ja, jb, jt = (np.asarray(jax.jit(f)(P0)) for f in (fa, fb, ft))
    for label, x, y in (("jit a vs b", ja, jb), ("jit a vs t", ja, jt),
                        ("jit a vs eager a", ja, a), ("jit b vs eager b", jb, b),
                        ("jit t vs eager t", jt, t)):
        np.testing.assert_allclose(x[:, :n_linear], y[:, :n_linear],
                                   rtol=0, atol=1e-15, err_msg=label)
        gap = np.abs(x[:, -1] - y[:, -1])
        assert float(gap[resolved].max()) < 1e-9, (label, float(gap[resolved].max()))
        ratio = float((gap / np.maximum(amplification, 1.0)).max())
        assert ratio < 1e-12, (label, ratio)


