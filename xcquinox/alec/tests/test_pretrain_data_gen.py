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
    # exchange mass at this basis and grid.
    assert abs(got - ref) < 1e-6, (got, ref)


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
    np.testing.assert_allclose(rows["Fx"][:n_a], fx_dfs[keep],
                               rtol=0, atol=1e-9)


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


def test_atom_columns_default_footing_is_unchanged():
    from xcquinox.alec.pretrain_data_gen import _atom_columns
    cols = _atom_columns("O", 2, "def2-svp", 1, polarized=True,
                         descriptors=True)
    assert "x_rows" not in cols


def _shared_scf_dft_module(monkeypatch, module):
    """Point ``module`` at a pyscf.dft stand-in that hands every call the SAME
    converged UKS object, so two column builds share one density.

    The O-atom UKS solution has a degenerate 2p hole whose orientation within
    the p subspace is fixed by rounding noise rather than by the energy: two
    independent SCFs of the same atom under the same settings land on
    ``e_tot`` values 5.9e-08 Ha apart but on density matrices differing by
    3.5e-01, which propagates to 3.4e-01 in rho and 5.5e+03 in sigma at
    individual grid points. That scatter is a property of the atom, not of the
    row footing, and it swamps an exact column comparison. With one SCF behind
    both builds the comparison is exact.
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
    base = pdg._atom_columns("O", 2, "def2-svp", 1, polarized=True,
                             descriptors=True)
    extended = pdg._atom_columns("O", 2, "def2-svp", 1, polarized=True,
                                 descriptors=True,
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
