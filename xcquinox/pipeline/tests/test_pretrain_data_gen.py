"""Tests for the canonical pretrain-data generator (xcquinox.pipeline.pretrain_data_gen).

Uses a tiny 2-atom set (He closed-shell, H open-shell) on a coarse grid so the
PBE SCFs are fast, while still exercising both the spin=0 and spin=1 branches and
the zeta column.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from xcquinox.pipeline.pretrain_data_gen import generate_pretrain_data_npz

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


def test_stale_pretrain_column_width_is_rejected():
    """A pretrain .npz written before a descriptor's width changed must FAIL
    loudly at assembly, not silently widen the network input.

    Before this gate, a 3-column dm_all against the now-2-feature
    dm_statistics produced a 6-wide input where n_input_features was 5, and
    training proceeded against a mismatched layout without complaint.
    """
    import xcquinox.pipeline as pipeline
    from xcquinox.pipeline.pretrain import _assemble_pretrain_descriptors

    arch = pipeline.get_architecture("deep_dm_3x16")
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
    from xcquinox.pipeline.pretrain_data_gen import spin_channel_exchange_rows
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
    from xcquinox.pipeline.pretrain_data_gen import spin_channel_exchange_rows
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
    from xcquinox.pipeline.pretrain_data_gen import spin_channel_exchange_rows
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


def test_spin_channel_rows_refuse_a_restricted_density_matrix():
    """A restricted ``(nao, nao)`` density matrix carries no spin resolution, so
    there is no channel to double: it is refused (the guard lives in
    ``descriptors.doubled_spin_dm``) rather than read as though its two leading
    rows were the two spin channels, or silently halved into both.
    """
    from xcquinox.pipeline.pretrain_data_gen import spin_channel_exchange_rows
    mol, mf, ao, dm_ab = _open_shell_scf(symbol="H", spin=1)
    dm_restricted = np.asarray(dm_ab[0] + dm_ab[1])
    assert dm_restricted.ndim == 2
    with pytest.raises(ValueError, match="spin-resolved"):
        spin_channel_exchange_rows(mol, mf, ao, dm_restricted,
                                   descriptors=False)


# The O-atom column builds below run at grid level 3, the production level and
# the level at which the program will write these rows at all: the generator
# REFUSES a spatially degenerate free atom below
# pretrain_data_gen.COARSE_DEGENERATE_MIN_GRID_LEVEL unless the run states the
# irreproducible-degenerate waiver, because that quadrature does not resolve
# the P term and independent processes then write different rows under one
# manifest identity. These tests call ``_atom_columns`` directly, below that
# gate, so nothing stops them building coarse -- but the columns they compare
# would be an identity no run produces, and the comparison would carry the
# process-to-process scatter that is the reason it does not.
#
# The measurement that first set this level still holds beside it: the parent
# density carries the training orientation lock
# (pretrain_data_gen.PRETRAIN_ORIENTATION_LOCK_STRENGTH), and on the coarse
# level-1 atomic grid the lock's 2p splitting competes with the grid's own
# angular anisotropy, so the locked PBE SCF of O stalls there under pyscf's
# defaults (2 of 10 draws converge at level 1, 5 of 10 at level 2, 10 of 10 at
# level 3 and at the production 6-311++G(3df,2pd) / level 3 identity).
_O_GRID_LEVEL = 3


def test_atom_columns_rejects_an_unknown_footing():
    from xcquinox.pipeline.pretrain_data_gen import _atom_columns
    with pytest.raises(ValueError, match="exchange_footing"):
        _atom_columns("He", 0, "def2-svp", 1, polarized=True,
                      descriptors=True, exchange_footing="per_orbital")


def test_pretraining_grid_rebuild_pins_the_cutoff(monkeypatch):
    """The rebuilt grid of the generator is the record's on any release.

    ``_system_columns`` replays the precompute's grid from the same initial
    guess and refuses to continue when the two differ. The record's grid was
    pruned at the reference density cutoff, while the mean-field the replay
    builds takes whatever the release leaves on the Kohn-Sham class: pyscf
    2.14.0 moved that default from 1e-7 to 0, which keeps the small-density
    tail rows and makes the replayed grid a different quadrature from the
    record's. The class default is set to 0 for the second build below, which
    poses that release's behaviour on any release; the record the second
    build reads is the first build's, held by the process-level precompute
    memo (keyed on the molecule, the basis, the grid level, the parent
    functional and the lock, none of which change between the two calls), so
    the two builds differ in the density cutoff and in nothing else.
    """
    from pyscf import dft, gto
    from pyscf.dft.rks import KohnShamDFT
    from xcquinox.pipeline import pretrain_data_gen as pdg
    from xcquinox.pipeline.data import clear_precompute_cache

    system = pdg.PretrainSystem(name="H2", atom="H 0 0 0; H 0 0 0.74",
                                charge=0, spin=0)
    columns = dict(reference_xc="pbe", polarized=False, descriptors=False)

    def grid_points(cutoff):
        mol = gto.M(atom=system.atom, basis="sto-3g", verbose=0)
        mf = dft.RKS(mol)
        mf.grids.level = 0
        mf.small_rho_cutoff = cutoff
        mf.initialize_grids(mol, mf.get_init_guess(mol, mf.init_guess,
                                                   s1e=mf.get_ovlp(mol)))
        return int(mf.grids.weights.size)

    # The cutoff is a real difference on this system (1224 points against
    # 1240 here), so the comparison below cannot pass for want of a pruned
    # row; the counts themselves are not pinned, only their order.
    pruned, unpruned = grid_points(1e-7), grid_points(0.0)
    assert 0 < pruned < unpruned

    clear_precompute_cache()
    monkeypatch.setattr(KohnShamDFT, "small_rho_cutoff", 1e-7)
    pinned = pdg._system_columns(system, "sto-3g", 0, **columns)
    monkeypatch.setattr(KohnShamDFT, "small_rho_cutoff", 0.0)
    rebuilt = pdg._system_columns(system, "sto-3g", 0, **columns)
    assert int(np.asarray(pinned["weights"]).size) == pruned
    for key in ("rho", "sigma", "weights", "Fx", "Fc", "e_lda_x", "e_lda_c"):
        np.testing.assert_array_equal(np.asarray(pinned[key]),
                                      np.asarray(rebuilt[key]))
