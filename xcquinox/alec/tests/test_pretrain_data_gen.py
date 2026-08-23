"""Tests for the canonical pretrain-data generator (xcquinox.alec.pretrain_data_gen).

Uses a tiny 2-atom set (He closed-shell, H open-shell) on a coarse grid so the
PBE SCFs are fast, while still exercising both the spin=0 and spin=1 branches and
the zeta column.
"""
import types

import jax.numpy as jnp
import numpy as np
import pytest

from xcquinox.alec.pretrain_data_gen import generate_pretrain_data_npz

_TINY_ATOMS = (("He", 0), ("H", 1))


def test_polarized_npz_has_zeta_and_consistent_columns(tmp_path):
    path = generate_pretrain_data_npz(
        str(tmp_path), atoms=_TINY_ATOMS, polarized=True, descriptors=True)
    assert path.endswith("pretrain_data_polarized.npz")
    d = dict(np.load(path))
    # Required columns + zeta + descriptors all present.
    for key in ("rho_all", "sigma_all", "Fx_all", "Fc_all", "weights_all",
                "zeta_all", "cusp_all", "dm_all", "rung35_all"):
        assert key in d, f"missing {key}"
    n = d["rho_all"].shape[0]
    assert n > 0
    # All per-point columns are aligned and finite.
    for key in ("sigma_all", "Fx_all", "Fc_all", "weights_all", "zeta_all"):
        assert d[key].shape[0] == n
        assert np.all(np.isfinite(d[key]))
    assert d["cusp_all"].shape[0] == n and d["dm_all"].shape[0] == n
    # rung-3.5 occupancy column: aligned with rho, two per-spin channels, in [0, 1].
    assert d["rung35_all"].shape == (n, 2)
    assert np.all(d["rung35_all"] >= -1e-6) and np.all(d["rung35_all"] <= 1.0 + 1e-6)
    # multi-width rung-3.5 column: 3 widths x 2 spins, alpha-major, in [0, 1].
    assert d["rung35ms_all"].shape == (n, 6)
    assert np.all(d["rung35ms_all"] >= -1e-6) and np.all(d["rung35ms_all"] <= 1.0 + 1e-6)
    # the alphas=(0.2,) middle width must reproduce the single-width column
    # exactly (columns 2:4 are the DEFAULT_RUNG35_ALPHA pair, alpha-major).
    np.testing.assert_array_equal(d["rung35ms_all"][:, 2:4], d["rung35_all"])
    # zeta in [-1, 1]; ~0 on the closed-shell He points, ~+1 on the fully spin-
    # polarized H points -> the column must span from near 0 up toward 1.
    z = d["zeta_all"]
    assert np.all(z <= 1.0 + 1e-6) and np.all(z >= -1.0 - 1e-6)
    assert z.min() < 0.05      # He contributes ~zero polarization
    assert z.max() > 0.5       # H is strongly polarized


def test_unpolarized_npz_omits_zeta_and_default_name(tmp_path):
    path = generate_pretrain_data_npz(
        str(tmp_path), atoms=_TINY_ATOMS, polarized=False, descriptors=False)
    assert path.endswith("pretrain_data.npz")
    assert not path.endswith("pretrain_data_polarized.npz")
    d = dict(np.load(path))
    assert "zeta_all" not in d
    assert "cusp_all" not in d and "dm_all" not in d
    for key in ("rho_all", "sigma_all", "Fx_all", "Fc_all", "weights_all"):
        assert key in d


def test_stale_pretrain_column_width_is_rejected():
    """A pretrain .npz written before a descriptor's width changed must FAIL
    loudly at assembly, not silently widen the network input.

    Before this gate, a 3-column dm_all against the now-2-feature
    dm_statistics produced a 6-wide input where n_input_features was 5, and
    training proceeded against a mismatched layout without complaint.
    """
    import xcquinox.alec as alec
    from xcquinox.alec.pretrain import _assemble_pretrain_descriptors

    arch = alec.get_architecture("deep_dm_3x16")
    n = 5
    fresh = dict(rho_all=jnp.ones(n), sigma_all=jnp.ones(n),
                 dm_all=jnp.ones((n, 2)))
    assert _assemble_pretrain_descriptors(arch, fresh).shape == (n, 4)
    stale = dict(rho_all=jnp.ones(n), sigma_all=jnp.ones(n),
                 dm_all=jnp.ones((n, 3)))
    with pytest.raises(ValueError, match="predates a change"):
        _assemble_pretrain_descriptors(arch, stale)


# ---------------------------------------------------------------------------
# Open-shell exchange row footing: the inputs the production UKS exchange
# actually evaluates, (2 rho_sigma, 4 sigma_sigma_sigma, features of
# diag(P_sigma, P_sigma)), with the parent's SPIN-UNPOLARIZED enhancement factor
# at those inputs as the target.
# ---------------------------------------------------------------------------

def _open_shell_scf(symbol="O", spin=2, basis="def2-svp", grid_level=1):
    from pyscf import dft, gto
    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, charge=0, spin=spin,
                verbose=0)
    mf = dft.UKS(mol)
    mf.xc = "pbe"
    mf.grids.level = grid_level
    mf.kernel()
    ao = mf._numint.eval_ao(mol, mf.grids.coords, deriv=1)
    return mol, mf, ao, mf.make_rdm1()


def test_spin_channel_rows_reproduce_the_parent_open_shell_exchange_energy():
    """The rows are an exact quadrature of the parent's open-shell exchange:
    summing w_row * rho_row * eps_x^LDA(rho_row) * (1 + Fx_row) reproduces
    libxc's spin-polarized PBE exchange, because 1/2 (E_x[2 rho_a] +
    E_x[2 rho_b]) IS that energy (Oliver and Perdew, Phys. Rev. A 20, 397
    (1979))."""
    from pyscf import dft
    from xcquinox.alec.pretrain_data_gen import spin_channel_exchange_rows
    mol, mf, ao, dm_ab = _open_shell_scf()
    rows = spin_channel_exchange_rows(mol, mf, ao, dm_ab, descriptors=False)
    c_lda = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)
    ex_lda = c_lda * np.cbrt(np.clip(rows["rho"], 1e-300, None))
    got = float(np.sum(rows["weights"] * rows["rho"] * ex_lda
                       * (1.0 + rows["Fx"])))
    rho_a_gga = mf._numint.eval_rho(mol, ao, dm_ab[0], xctype="GGA", hermi=True)
    rho_b_gga = mf._numint.eval_rho(mol, ao, dm_ab[1], xctype="GGA", hermi=True)
    eps = np.asarray(mf._numint.eval_xc(
        "PBE,", np.stack([rho_a_gga, rho_b_gga]), spin=1)[0])
    ref = float(np.sum(np.asarray(mf.grids.weights)
                       * (rho_a_gga[0] + rho_b_gga[0]) * eps))
    # The residual is the rho floor that drops the deep tail from the row set
    # plus the +-5 clip on the stored enhancement factor; both carry negligible
    # exchange mass at this basis and grid. Measured over six O draws (two with
    # single-threaded BLAS, four at four threads): 2.56e-12 to 3.34e-12 Ha. The
    # gate below sits ~300x above that floor, tight enough to reject a 1e-9
    # relative error in the row weights, which moves the sum by 8.2e-09 Ha.
    assert abs(got - ref) < 1e-9, (got, ref)


def test_spin_channel_rows_match_the_dfs_zeroed_channel_recipe():
    """The DFS protocol (spec Section 6) targets e_x^ref(rho_sigma, 0) with libxc
    spin=1 and the other channel zeroed. For exchange E_x[n_sigma, 0] =
    E_x[2 n_sigma] / 2 (Oliver and Perdew, Phys. Rev. A 20, 397 (1979)) and
    libxc's spin=1 per-electron output is normalized by the total density, so
    that recipe returns the unpolarized enhancement at the doubled inputs -- the
    number this row builder computes through the spin=0 call."""
    from xcquinox.alec.pretrain_data_gen import spin_channel_exchange_rows
    mol, mf, ao, dm_ab = _open_shell_scf()
    rows = spin_channel_exchange_rows(mol, mf, ao, dm_ab, descriptors=False)
    rho_a_gga = mf._numint.eval_rho(mol, ao, dm_ab[0], xctype="GGA", hermi=True)
    zeroed = np.zeros_like(rho_a_gga)
    ex_ref = np.asarray(mf._numint.eval_xc(
        "PBE,", np.stack([rho_a_gga, zeroed]), spin=1)[0])
    ex_lda_ref = np.asarray(mf._numint.eval_xc(
        "LDA_X,", (rho_a_gga[0], zeroed[0]), spin=1)[0])
    safe = np.where(np.abs(ex_lda_ref) > 1e-12, ex_lda_ref, 1e-12)
    fx_dfs = np.clip(ex_ref / safe - 1.0, -5.0, 5.0)
    keep = 2.0 * rho_a_gga[0] > 1e-10
    n_a = int(keep.sum())
    # The two evaluations are the same identity taken through different libxc
    # calls, so the deviation is round-off on an O(1) enhancement factor:
    # 1.11e-15 over four four-thread O draws, 1.55e-15 single-threaded. The
    # gate sits ~600x above that, tight enough to reject a 1e-11 shift of the
    # stored column.
    np.testing.assert_allclose(rows["Fx"][:n_a], fx_dfs[keep],
                               rtol=0, atol=1e-12)


def test_spin_channel_rows_skip_the_empty_channel_of_h():
    """H carries no beta electron, so the beta channel contributes no rows and
    the alpha rows alone are the whole exchange, closing on libxc's
    spin-polarized PBE exchange to 7.0e-14 Ha at def2-svp / grid level 1.

    The empty channel is the one place the doubled-density construction meets a
    density that is identically zero, where the enhancement factor would be a
    zero divided by a zeroed LDA denominator, so its handling is pinned here.
    """
    from xcquinox.alec.pretrain_data_gen import spin_channel_exchange_rows
    mol, mf, ao, dm_ab = _open_shell_scf(symbol="H", spin=1)
    rows = spin_channel_exchange_rows(mol, mf, ao, dm_ab, descriptors=True)
    rho_a_gga = mf._numint.eval_rho(mol, ao, dm_ab[0], xctype="GGA", hermi=True)
    rho_b_gga = mf._numint.eval_rho(mol, ao, dm_ab[1], xctype="GGA", hermi=True)
    assert float(np.max(np.abs(rho_b_gga[0]))) == 0.0
    n_a = int(np.sum(2.0 * rho_a_gga[0] > 1e-10))
    assert rows["rho"].shape[0] == n_a
    for column in rows.values():
        assert np.all(np.isfinite(np.asarray(column)))
    c_lda = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)
    ex_lda = c_lda * np.cbrt(np.clip(rows["rho"], 1e-300, None))
    got = float(np.sum(rows["weights"] * rows["rho"] * ex_lda
                       * (1.0 + rows["Fx"])))
    eps = np.asarray(mf._numint.eval_xc(
        "PBE,", np.stack([rho_a_gga, rho_b_gga]), spin=1)[0])
    ref = float(np.sum(np.asarray(mf.grids.weights)
                       * (rho_a_gga[0] + rho_b_gga[0]) * eps))
    assert abs(got - ref) < 1e-10, (got, ref)


def test_spin_channel_rows_carry_the_doubled_ingredients():
    from xcquinox.alec.descriptors import doubled_spin_dm
    from xcquinox.alec.metagga import compute_alpha, compute_tau_from_dm
    from xcquinox.alec.pretrain_data_gen import spin_channel_exchange_rows
    mol, mf, ao, dm_ab = _open_shell_scf()
    rows = spin_channel_exchange_rows(mol, mf, ao, dm_ab, descriptors=False)
    rho_a_gga = mf._numint.eval_rho(mol, ao, dm_ab[0], xctype="GGA", hermi=True)
    n_a = int(np.sum(2.0 * rho_a_gga[0] > 1e-10))
    # Rows are emitted alpha channel first, so the leading block is the alpha
    # channel's doubled density.
    np.testing.assert_allclose(rows["rho"][:n_a],
                               2.0 * rho_a_gga[0][2.0 * rho_a_gga[0] > 1e-10],
                               rtol=0, atol=1e-12)
    tau_doubled = np.asarray(compute_tau_from_dm(
        jnp.asarray(ao[1:4]), doubled_spin_dm(jnp.asarray(dm_ab), 0)))
    sigma_doubled = 4.0 * (rho_a_gga[1:4] ** 2).sum(axis=0)
    expect = np.asarray(compute_alpha(jnp.asarray(2.0 * rho_a_gga[0]),
                                      jnp.asarray(sigma_doubled),
                                      jnp.asarray(tau_doubled)))
    keep = 2.0 * rho_a_gga[0] > 1e-10
    np.testing.assert_allclose(rows["metagga"][:n_a, 0], expect[keep],
                               rtol=0, atol=1e-12)


def test_spin_channel_rows_rung35_block_is_the_channel_in_both_slots():
    from xcquinox.alec.pretrain_data_gen import spin_channel_exchange_rows
    mol, mf, ao, dm_ab = _open_shell_scf()
    rows = spin_channel_exchange_rows(mol, mf, ao, dm_ab, descriptors=True)
    r = rows["rung35"]
    np.testing.assert_allclose(r[:, 0], r[:, 1], rtol=0, atol=1e-14)
    assert float(np.max(r)) < 1.0 + 1e-12
    ms = rows["rung35ms"]
    assert ms.shape[1] == 6
    for w in range(3):
        np.testing.assert_allclose(ms[:, 2 * w], ms[:, 2 * w + 1],
                                   rtol=0, atol=1e-14)


def test_spin_channel_rows_sigma_is_the_doubled_gradient_invariant():
    """``sigma`` is the gradient invariant of the DOUBLED density,
    ``|grad(2 rho_sigma)|^2 = 4 |grad rho_sigma|^2``, not the physical channel's
    ``|grad rho_sigma|^2``: the factor of four is what the exact spin-scaling
    relation hands the enhancement factor alongside ``2 rho_sigma`` (Oliver and
    Perdew, Phys. Rev. A 20, 397 (1979)).

    Oracle: PySCF's ``eval_rho`` on the doubled density matrix ``2 P_sigma``,
    whose density is that of ``diag(P_sigma, P_sigma)``. Scaling by two is exact
    in binary, so the oracle, the closed form ``4 |grad rho_sigma|^2`` and the
    stored column are the same floating-point number -- measured deviation 0.0
    on both channels over six O draws. The comparison is held at 1e-15 relative,
    a few ulp, against a halved column that would sit 5e-01 away.
    """
    from xcquinox.alec.pretrain_data_gen import spin_channel_exchange_rows
    mol, mf, ao, dm_ab = _open_shell_scf()
    rows = spin_channel_exchange_rows(mol, mf, ao, dm_ab, descriptors=False)
    start = 0
    for s in (0, 1):
        rho_s = mf._numint.eval_rho(mol, ao, dm_ab[s], xctype="GGA", hermi=True)
        rho_doubled = mf._numint.eval_rho(mol, ao, 2.0 * np.asarray(dm_ab[s]),
                                          xctype="GGA", hermi=True)
        sigma_ref = (rho_doubled[1:4] ** 2).sum(axis=0)
        np.testing.assert_allclose(sigma_ref,
                                   4.0 * (rho_s[1:4] ** 2).sum(axis=0),
                                   rtol=1e-15, atol=0)
        keep = 2.0 * rho_s[0] > 1e-10
        n_s = int(keep.sum())
        np.testing.assert_allclose(rows["sigma"][start:start + n_s],
                                   sigma_ref[keep], rtol=1e-15, atol=0)
        start += n_s
    assert start == rows["sigma"].shape[0]


def test_spin_channel_rows_fx_scan_is_scan_at_the_doubled_inputs():
    """``Fx_scan`` -- the meta-GGA pretraining target -- is SCAN's
    spin-unpolarized enhancement at the DOUBLED inputs
    ``(2 rho_sigma, 4 sigma_sigma_sigma, 2 tau_sigma)`` over
    ``eps_x^LDA(2 rho_sigma)``.

    Oracle: libxc reached by a second, independent path. The meta-GGA row comes
    from PySCF's ``eval_rho`` on the doubled density matrix, so ``tau`` does NOT
    come from the ``metagga.compute_tau_from_dm`` the row builder uses, and the
    denominator comes from libxc's own ``LDA_X`` rather than the analytic
    coefficient (the two agree exactly here, 0.0 relative). Worst measured
    deviation 3.4e-13 over six O draws, set by the 2.7e-12 to 4.5e-12 spread
    between the two tau paths; the gate keeps ~300x headroom. Evaluating SCAN
    at the physical channel density instead moves this column by 0.276 at worst
    and 0.221 on average, which is the error the pin exists to catch.
    """
    from xcquinox.alec.pretrain_data_gen import spin_channel_exchange_rows
    mol, mf, ao, dm_ab = _open_shell_scf()
    rows = spin_channel_exchange_rows(mol, mf, ao, dm_ab, descriptors=False)
    start = 0
    for s in (0, 1):
        rho_d = mf._numint.eval_rho(mol, ao, 2.0 * np.asarray(dm_ab[s]),
                                    xctype="MGGA", hermi=True, with_lapl=False)
        mgga_row = np.vstack([rho_d[:4], np.zeros_like(rho_d[0]), rho_d[4]])
        ex_scan = np.asarray(mf._numint.eval_xc("SCAN,", mgga_row, spin=0)[0])
        ex_lda = np.asarray(mf._numint.eval_xc("LDA_X,", rho_d[0], spin=0)[0])
        keep = rho_d[0] > 1e-10
        n_s = int(keep.sum())
        np.testing.assert_allclose(rows["Fx_scan"][start:start + n_s],
                                   (ex_scan / ex_lda - 1.0)[keep],
                                   rtol=0, atol=1e-10)
        start += n_s
    assert start == rows["Fx_scan"].shape[0]
    # The comparison above is against the UNCLIPPED ratio, which is legitimate
    # only because no row reaches the +-5 clip: max |Fx_scan| = 0.909 on O.
    assert float(np.max(np.abs(rows["Fx_scan"]))) < 5.0


def test_spin_channel_rows_cusp_block_is_the_geometry_only_total_density_block():
    """``cusp`` is a function of the grid coordinates and the nuclei alone, so
    doubling a spin channel cannot change it: each channel's block must be the
    total-density path's block at the points that channel keeps, in the
    production log-compressed convention (``_atom_columns`` builds its cusp
    column from the same ``features.compute_cusp_descriptor`` call with
    ``log_transform=cusp_log_transform``, default True).

    Oracle: that descriptor on the FULL grid, sliced by each channel's own
    retention mask. Measured deviation 0.0 over six O draws; the 1e-15 gate is
    a few ulp on features bounded in [-1, 1], while running the descriptor with
    the flag flipped moves them by 0.534.
    """
    from xcquinox.alec.pretrain_data_gen import spin_channel_exchange_rows
    from xcquinox.features import compute_cusp_descriptor
    mol, mf, ao, dm_ab = _open_shell_scf()
    rows = spin_channel_exchange_rows(mol, mf, ao, dm_ab, descriptors=True)
    cusp_grid = np.asarray(compute_cusp_descriptor(
        jnp.asarray(mf.grids.coords), jnp.asarray(mol.atom_coords()),
        jnp.asarray(mol.atom_charges()), log_transform=True))
    spans, start = [], 0
    for s in (0, 1):
        rho_s = mf._numint.eval_rho(mol, ao, dm_ab[s], xctype="GGA", hermi=True)
        keep = 2.0 * rho_s[0] > 1e-10
        n_s = int(keep.sum())
        np.testing.assert_allclose(rows["cusp"][start:start + n_s],
                                   cusp_grid[keep], rtol=0, atol=1e-15)
        spans.append((keep, start, n_s))
        start += n_s
    assert start == rows["cusp"].shape[0]
    # ... and, being geometry-only, the two channels carry identical values at
    # every point both of them keep.
    (keep_a, start_a, n_a), (keep_b, start_b, n_b) = spans
    both = keep_a & keep_b
    assert int(both.sum()) > 0
    np.testing.assert_array_equal(
        rows["cusp"][start_a:start_a + n_a][both[keep_a]],
        rows["cusp"][start_b:start_b + n_b][both[keep_b]])


def test_spin_channel_rows_dm_block_is_the_doubled_density_matrix():
    """``dm`` is the density-matrix feature block of ``diag(P_sigma, P_sigma)``
    tiled down the channel's rows -- not the block of the physical spin density
    matrix, and not the block of the total one.

    Oracle: ``features.compute_dm_features_array`` on
    ``descriptors.doubled_spin_dm(P, sigma)``; measured deviation 0.0 over six O
    draws. The pin discriminates because the three candidate matrices give
    measurably different blocks on O: the doubled matrix takes the per-spin
    idempotency branch and returns 5.7e-31 there, while the physical spin matrix
    is read as a restricted one and returns 7.7e-02; the total matrix agrees on
    idempotency but differs in the off-diagonal norm (0.2523 against 0.2454 for
    the alpha channel). The smallest of those separations, 6.9e-03, sets the
    1e-3 floor asserted below; the alpha and beta blocks themselves differ by
    8.0e-02.
    """
    from xcquinox.alec.descriptors import doubled_spin_dm
    from xcquinox.alec.pretrain_data_gen import spin_channel_exchange_rows
    from xcquinox.features import compute_dm_features_array
    mol, mf, ao, dm_ab = _open_shell_scf()
    rows = spin_channel_exchange_rows(mol, mf, ao, dm_ab, descriptors=True)
    s_matrix = jnp.asarray(mol.intor("int1e_ovlp"))
    blocks, start = [], 0
    for s in (0, 1):
        rho_s = mf._numint.eval_rho(mol, ao, dm_ab[s], xctype="GGA", hermi=True)
        n_s = int(np.sum(2.0 * rho_s[0] > 1e-10))
        ref = np.asarray(compute_dm_features_array(
            doubled_spin_dm(jnp.asarray(dm_ab), s), s_matrix))
        got = rows["dm"][start:start + n_s]
        # one global feature vector per channel, tiled over that channel's rows
        np.testing.assert_array_equal(got, np.tile(got[0], (n_s, 1)))
        np.testing.assert_allclose(got[0], ref, rtol=1e-15, atol=0)
        # the two matrices the block could otherwise have come from are not it
        for other in (compute_dm_features_array(jnp.asarray(dm_ab[s]),
                                                s_matrix),
                      compute_dm_features_array(jnp.asarray(dm_ab), s_matrix)):
            assert float(np.max(np.abs(np.asarray(other) - ref))) > 1e-3
        blocks.append(ref)
        start += n_s
    assert start == rows["dm"].shape[0]
    assert float(np.max(np.abs(blocks[0] - blocks[1]))) > 1e-3


def test_spin_channel_rows_refuse_a_restricted_density_matrix():
    """A restricted ``(nao, nao)`` density matrix carries no spin resolution, so
    there is no channel to double: it is refused (the guard lives in
    ``descriptors.doubled_spin_dm``) rather than read as though its two leading
    rows were the two spin channels, or silently halved into both.
    """
    from xcquinox.alec.pretrain_data_gen import spin_channel_exchange_rows
    mol, mf, ao, dm_ab = _open_shell_scf(symbol="H", spin=1)
    dm_restricted = np.asarray(dm_ab[0] + dm_ab[1])
    assert dm_restricted.ndim == 2
    with pytest.raises(ValueError, match="spin-resolved"):
        spin_channel_exchange_rows(mol, mf, ao, dm_restricted,
                                   descriptors=False)


# The O-atom column builds below run at grid level 3, the production level:
# the parent density now carries the training orientation lock
# (pretrain_data_gen.PRETRAIN_ORIENTATION_LOCK_STRENGTH), and on the coarse
# level-1 atomic grid the lock's 2p splitting competes with the grid's own
# angular anisotropy, so the locked PBE SCF of O stalls there under pyscf's
# defaults (2 of 10 draws converge at level 1, 5 of 10 at level 2, 10 of 10 at
# level 3 and at the production 6-311++G(3df,2pd) / level 3 identity).
_O_GRID_LEVEL = 3


def test_atom_columns_default_footing_is_unchanged():
    from xcquinox.alec.pretrain_data_gen import _atom_columns
    cols = _atom_columns("O", 2, "def2-svp", _O_GRID_LEVEL, polarized=True,
                         descriptors=True)
    assert "x_rows" not in cols


def _shared_scf_dft_module(monkeypatch, module):
    """Point ``module`` at a pyscf.dft stand-in that hands every call the SAME
    converged UKS object, so two column builds share one density.

    Two independent SCFs of the same atom under identical settings are not
    bit-reproducible for ANY of the pretraining atoms once BLAS runs
    multi-threaded, since the reduction order of a threaded dot product varies
    between calls and the converged density with it. At four threads the
    difference stays at rounding level in the density for H, He and N (max
    ``|dP|`` of order 1e-19 to 1e-15) but amplifies through the ``tau - tau_W``
    cancellation in alpha to order 1e-10 in the ``metagga`` column, and to order
    1e-09 in ``sigma`` on N. On O it is not a rounding effect at all: the 2p
    hole is orbitally degenerate and rounding noise -- not the energy -- picks
    its orientation within the p subspace, so two runs land on the same energy
    solution (order 1e-07 Ha apart) with density matrices order 1e-01 apart,
    which propagates to order 1e-01 in rho and 1e+03 in sigma at individual grid
    points. Every figure here is order-of-magnitude and moves from draw to draw;
    the same comparison with BLAS pinned to one thread reproduces exactly. The
    scatter is a property of the atoms and of threaded linear algebra, not of
    the row footing, and it swamps an exact column comparison. With one SCF
    behind both builds the comparison is exact.
    """
    from pyscf import dft as pyscf_dft
    converged = {}

    def _shared_uks(mol):
        mf = converged.get("mf")
        if mf is None:
            mf = converged["mf"] = pyscf_dft.UKS(mol)
            return mf
        # Assigning ``grids.level`` clears the built quadrature, so hold on to
        # the arrays the first build produced and put them back in place of the
        # SCF: the orbitals of this object are already converged, and reusing
        # the same arrays (rather than rebuilding) keeps the two column sets on
        # one quadrature by construction.
        grid = converged.setdefault("grid", (mf.grids.coords, mf.grids.weights))

        def _restore_grid_and_skip_scf(*args, **kwargs):
            mf.grids.coords, mf.grids.weights = grid
            return mf.e_tot

        mf.kernel = _restore_grid_and_skip_scf
        return mf

    monkeypatch.setattr(module, "dft", types.SimpleNamespace(
        UKS=_shared_uks, RKS=pyscf_dft.RKS))


def test_atom_columns_spin_channel_footing_only_adds_x_rows(monkeypatch):
    from xcquinox.alec import pretrain_data_gen as pdg
    _shared_scf_dft_module(monkeypatch, pdg)
    base = pdg._atom_columns("O", 2, "def2-svp", _O_GRID_LEVEL, polarized=True,
                             descriptors=True)
    extended = pdg._atom_columns("O", 2, "def2-svp", _O_GRID_LEVEL,
                                 polarized=True, descriptors=True,
                                 exchange_footing="spin_channel")
    assert set(extended) - set(base) == {"x_rows"}
    for key in base:
        np.testing.assert_array_equal(np.asarray(base[key]),
                                      np.asarray(extended[key]))
    assert extended["x_rows"] is not None
    assert extended["x_rows"]["rho"].ndim == 1


def test_atom_columns_spin_channel_footing_is_none_for_a_closed_shell_atom():
    from xcquinox.alec.pretrain_data_gen import _atom_columns
    cols = _atom_columns("He", 0, "def2-svp", 1, polarized=True,
                         descriptors=True,
                         exchange_footing="spin_channel")
    assert cols["x_rows"] is None


def test_atom_columns_rejects_an_unknown_footing():
    from xcquinox.alec.pretrain_data_gen import _atom_columns
    with pytest.raises(ValueError, match="exchange_footing"):
        _atom_columns("He", 0, "def2-svp", 1, polarized=True,
                      descriptors=True, exchange_footing="per_orbital")
