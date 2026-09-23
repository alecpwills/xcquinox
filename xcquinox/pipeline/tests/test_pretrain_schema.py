"""The pretrain-data .npz schema after the pretraining-protocol change.

Two row blocks, because per-channel exchange rows and total-density correlation
rows are no longer the same rows (spec Section 3.2): the historical ``*_all``
block is the correlation / total-density block, and a ``*_x`` block appears
under the ``spin_channel`` footing. A per-row ``system_*`` index and a
per-system energy table carry the energy term of Section 6 deviation 3. The
writer refuses a column it has no slot for instead of dropping it, and the
reader refuses a file with a missing required block or an unknown key, so the
schema is closed in both directions.

Tolerances are anchored to measured floors, quoted at each constant.
"""
import os

import numpy as np
import pytest

import xcquinox.pipeline.pretrain_data_gen as pdg


_TINY = (("He", 0), ("H", 1))
_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                        "pretrain_data_default_reference.npz")

#: The new keys the default configuration gains. Everything else the default
#: file carries is pinned bit-for-bit against the recorded fixture.
_NEW_DEFAULT_KEYS = sorted([
    "e_c_parent_scan_sys", "e_c_parent_sys", "e_lda_c_all", "e_lda_x_all",
    "e_x_parent_scan_sys", "e_x_parent_sys", "mesh_weight_fraction",
    "system_all", "system_natoms",
])


def _gen(tmp_path, **kw):
    kw.setdefault("atoms", _TINY)
    kw.setdefault("basis", "sto-3g")
    kw.setdefault("grid_level", 0)
    kw.setdefault("polarized", True)
    kw.setdefault("descriptors", True)
    path = pdg.generate_pretrain_data_npz(str(tmp_path), **kw)
    with np.load(path) as z:
        return path, {k: np.array(z[k]) for k in z.files}


def _fake_columns(n=3, *, dtype=np.float64, polarized=True, descriptors=True,
                  x_rows=None):
    """A complete column set in the builder's contract, for tests of the
    writer's bookkeeping that need no SCF."""
    cols = {k: np.ones(n, dtype=dtype) for k in (
        "rho", "sigma", "Fx", "Fc", "Fx_scan", "Fc_scan", "weights",
        "e_lda_x", "e_lda_c")}
    cols["metagga"] = np.ones((n, 1), dtype=dtype)
    if polarized:
        cols["zeta"] = np.zeros(n, dtype=dtype)
    if descriptors:
        cols["cusp"] = np.ones((n, 2), dtype=dtype)
        cols["dm"] = np.ones((n, 2), dtype=dtype)
        cols["rung35"] = np.ones((n, 2), dtype=dtype)
        cols["rung35ms"] = np.ones((n, 6), dtype=dtype)
    if x_rows is not None:
        cols["x_rows"] = x_rows
    return cols


def _install_fake(monkeypatch, factory):
    calls = []

    def _fake(system, basis, grid_level, **kw):
        calls.append(system)
        return factory(system, **kw)

    monkeypatch.setattr(pdg, "_system_columns", _fake)
    return calls


# ---------------------------------------------------------------------------
# The regression pin: the default configuration is unchanged
# ---------------------------------------------------------------------------

def test_default_output_matches_the_recorded_reference(tmp_path):
    """Every column the generator writes at the default configuration is
    bit-identical to the recorded fixture, so a YAML already in flight trains
    on the same numbers. New keys may appear; old ones may not move. (The
    .npz CONTAINER is a zip whose headers carry write timestamps, so the pin
    is on array contents, not on the file's bytes.) The recording predates
    the orientation lock; both atoms carry one s function, on which the
    traceless-quadrupole bias vanishes identically, so the locked default
    reproduces it.

    The fixture was re-recorded when the iso-orbital indicator's lower bound
    became a smooth positive part (``metagga.compute_alpha``, width 1e-5;
    docs/open_items.md entry 27). Against the previous recording exactly two
    keys moved: ``metagga_all`` on 1200 of 1200 rows, from the hard clip's
    0.0 (largest raw residue 1.4e-10) to the smoothing's floor 5.0e-6 (both
    atoms are one orbital in this basis), and ``metagga_mesh`` on 560 of 560
    rows by at most 5.0e-6 (the alpha = 0 nodes by the floor, the others by
    ``width^2 / (4 alpha)`` <= 2.5e-10); every other key -- rho, sigma, the
    PBE and SCAN targets, cusp, dm, rung35, rung35ms, weights, zeta, on the
    atomic rows and on the mesh -- is bit-identical.

    The fixture was re-recorded once more on the pyscf 2.14.0 / jax 0.10.2
    stack (2026-09-21), with the reference SCF's density cutoff pinned at
    1e-7 (``pyscf_determinism.REFERENCE_SMALL_RHO_CUTOFF``) so the grid is
    the one the previous recording used. Against the jax 0.7.0 recording
    five keys moved, every one a column jax computes and none that pyscf
    computes: ``metagga_all`` on 156 of 1200 rows by at most 1.0e-11 (XLA
    0.10 contracts the kinetic-energy density's einsum in another order,
    one ulp of tau on 116 of the 576 He points, which the same contraction
    in numpy does not show; both atoms are one orbital, so tau - tau_W is a
    rounding residue that the smooth floor maps to 5.0e-6 and the ulp to
    1e-11 there), through the indicator ``Fx_scan_all`` on 32 rows by at
    most 4.5e-13 and ``Fc_scan_all`` on 48 rows by at most 1.2e-15 (libxc's
    own values are bit-identical between the two releases on fixed inputs)
    and ``e_c_parent_scan_sys`` of the H atom by 1.3e-19, and ``cusp_all``
    on 1010 of 2400 entries by at most 1.1e-16, one ulp of a value below
    one (the column is jax's exponential and hyperbolic tangent of the same
    geometry); rho, sigma, the weights, the PBE targets, dm, rung35,
    rung35ms, zeta and the mesh are bit-identical."""
    ref = dict(np.load(_FIXTURE))
    _path, got = _gen(tmp_path)
    missing = sorted(set(ref) - set(got))
    assert not missing, f"the default output lost {missing}"
    for key in sorted(ref):
        assert got[key].dtype == ref[key].dtype, key
        assert got[key].shape == ref[key].shape, key
        np.testing.assert_array_equal(got[key], ref[key], err_msg=key)


def _legacy_view(ref):
    """The recorded fixture without the keys the protocol change added: the
    pre-protocol file format (the fixture itself was re-recorded after the
    change and carries them all)."""
    return {k: v for k, v in ref.items() if k not in _NEW_DEFAULT_KEYS}


def test_default_output_writes_no_exchange_block(tmp_path):
    _path, got = _gen(tmp_path)
    assert not [k for k in got if k.endswith("_x")]


# ---------------------------------------------------------------------------
# The system index and the energy table
# ---------------------------------------------------------------------------

def test_system_index_partitions_the_rows_in_declaration_order(tmp_path):
    _path, got = _gen(tmp_path)
    seg = got["system_all"]
    assert seg.dtype == np.int32
    assert seg.shape == got["rho_all"].shape
    assert sorted(set(seg.tolist())) == [0, 1]
    # Rows are emitted system by system, so the index is non-decreasing.
    assert np.all(np.diff(seg) >= 0)
    assert got["system_natoms"].dtype == np.int32
    assert got["system_natoms"].tolist() == [1, 1]


def test_energy_table_is_the_per_system_row_quadrature(tmp_path):
    _path, got = _gen(tmp_path)
    for key in ("e_x_parent_sys", "e_c_parent_sys", "e_x_parent_scan_sys",
                "e_c_parent_scan_sys"):
        assert got[key].dtype == np.float64 and got[key].shape == (2,), key
    for s in (0, 1):
        rows = got["system_all"] == s
        w = got["weights_all"][rows]
        expect = {
            "e_x_parent_sys": w * got["e_lda_x_all"][rows]
            * (1.0 + got["Fx_all"][rows]),
            "e_c_parent_sys": w * got["e_lda_c_all"][rows]
            * (1.0 + got["Fc_all"][rows]),
            "e_x_parent_scan_sys": w * got["e_lda_x_all"][rows]
            * (1.0 + got["Fx_scan_all"][rows]),
            "e_c_parent_scan_sys": w * got["e_lda_c_all"][rows]
            * (1.0 + got["Fc_scan_all"][rows]),
        }
        for key, contrib in expect.items():
            assert got[key][s] == pytest.approx(float(np.sum(contrib)), rel=0,
                                                abs=1e-12), (key, s)
        assert got["e_x_parent_sys"][s] < 0.0
        assert got["e_c_parent_sys"][s] < 0.0


#: Point-wise gap between the stored correlation baseline (libxc ``LDA_C_PW``
#: at spin=1) and the production baseline ``utils.pw92c_polarized_scalar``, on
#: rows with rho >= 1e-6. libxc's LDA_C_PW carries the rounded spin-stiffness
#: constant fz20 = 1.709921 (3.85e-8 from the exact 8 / [9 (2^(4/3) - 2)] the
#: production formula uses), which bounds the gap on the alpha_c term of the
#: interpolation; measured: 2.8e-10 (H, STO-3G), 7.4e-9 (O) and 7.0e-9 (N) at
#: def2-SVP / grid level 3, zero at zeta = 0. The PW92 variant with re-rounded
#: parameters (``LDA_C_PW_MOD``) sits 4.4e-6 away, 150x outside the gate.
_PW92_POINTWISE_RTOL = 3e-8
#: Below rho ~ 1e-7 on a fully polarized row libxc floors the empty spin
#: channel, so its zeta sits below one and the point-wise gap grows toward
#: 2.4e-5 at rho = 1e-10; those rows carry no energy, so the whole-file check
#: is the weighted one: sum |w rho (eps_ours - eps_libxc)|, measured 3.5e-14 Ha
#: on the H atom (STO-3G / level 0) and 1.1e-9 Ha on N at def2-SVP / level 3,
#: five orders below the certificate's tol_atom.
_PW92_INTEGRATED_GAP = 1e-8


# ---------------------------------------------------------------------------
# The exchange block
# ---------------------------------------------------------------------------

def test_spin_channel_footing_writes_an_exchange_block(tmp_path):
    _path, got = _gen(tmp_path, exchange_footing="spin_channel")
    n_x = got["rho_x"].shape[0]
    for key in ("sigma_x", "Fx_x", "Fx_scan_x", "weights_x", "e_lda_x_x"):
        assert got[key].shape == (n_x,), key
        assert got[key].dtype == np.float64, key
    assert got["system_x"].shape == (n_x,)
    assert got["system_x"].dtype == np.int32
    assert got["metagga_x"].shape == (n_x, 1)
    assert got["cusp_x"].shape == (n_x, 2)
    assert got["dm_x"].shape[0] == n_x
    assert got["rung35_x"].shape == (n_x, 2)
    assert got["rung35ms_x"].shape == (n_x, 6)
    assert sorted(set(got["system_x"].tolist())) == [0, 1]
    assert np.all(np.diff(got["system_x"]) >= 0)
    # He is closed-shell: its exchange rows ARE its total-density rows. H is a
    # one-electron open shell: only the alpha channel survives the floor, so
    # its exchange block is the alpha channel alone.
    assert int(np.sum(got["system_x"] == 0)) == int(
        np.sum(got["system_all"] == 0))
    # The correlation block is untouched by the footing.
    assert "Fc_x" not in got and "e_lda_c_x" not in got and "zeta_x" not in got


def test_one_electron_exchange_block_is_the_doubled_alpha_channel(tmp_path):
    """On H the exact spin scaling is an identity the file must satisfy: the
    alpha channel doubled is 2 rho, and PBE's spin-unpolarized enhancement at
    (2 rho, 4 sigma) equals its spin-polarized enhancement at (rho, 0), so the
    exchange block's target at a kept point is the total-density block's
    target there. The two are one identity through two libxc calls; measured
    deviation 1.1e-15 on the O channels (1e-12 gate, as in the row-builder
    tests)."""
    _path, got = _gen(tmp_path, exchange_footing="spin_channel")
    h_x = got["system_x"] == 1
    h_a = got["system_all"] == 1
    rho_x, rho_a = got["rho_x"][h_x], got["rho_all"][h_a]
    # The doubled density keeps every point the total density keeps (and may
    # keep more: 2 rho clears the floor where rho alone does not).
    assert rho_x.shape[0] >= rho_a.shape[0]
    assert np.all(np.isin(2.0 * rho_a, rho_x))
    common = np.isin(rho_x, 2.0 * rho_a)
    np.testing.assert_array_equal(rho_x[common], 2.0 * rho_a)
    np.testing.assert_allclose(got["Fx_x"][h_x][common], got["Fx_all"][h_a],
                               rtol=0, atol=1e-12)
    np.testing.assert_array_equal(got["weights_x"][h_x][common],
                                  0.5 * got["weights_all"][h_a])
    # The stored LDA column of the block is the analytic unpolarized LDA at
    # the doubled density, the denominator the block's ratio was formed with.
    np.testing.assert_array_equal(
        got["e_lda_x_x"][h_x], rho_x * (pdg._LDA_X_C * np.cbrt(rho_x)))


# ---------------------------------------------------------------------------
# Filename, reference density, mesh fraction, composition
# ---------------------------------------------------------------------------

def test_scan_reference_writes_its_own_file(tmp_path):
    path, got = _gen(tmp_path, reference_xc="scan", grid_level=1)
    assert os.path.basename(path) == "pretrain_data_polarized_scan.npz"
    assert got["rho_all"].shape[0] > 0
    assert pdg.read_pretrain_manifest(path)["reference_xc"] == "scan"


def test_mesh_fraction_is_stored_and_scales_the_mesh_weights(tmp_path):
    _path, base = _gen(tmp_path)
    assert base["mesh_weight_fraction"].shape == ()
    assert base["mesh_weight_fraction"].dtype == np.float64
    assert float(base["mesh_weight_fraction"]) == pdg.MESH_WEIGHT_FRACTION
    other = tmp_path / "half"
    other.mkdir()
    _p2, got = _gen(other, mesh_fraction=0.5)
    assert float(got["mesh_weight_fraction"]) == 0.5
    share = float(got["weights_mesh"].sum()
                  / (got["weights_mesh"].sum() + got["weights_all"].sum()))
    assert share == pytest.approx(0.5, rel=1e-12)


# ---------------------------------------------------------------------------
# The writer is closed: no column is dropped, none is invented
# ---------------------------------------------------------------------------

def test_writer_refuses_an_unknown_column_rather_than_dropping_it(
        monkeypatch, tmp_path):
    def _with_extra(system, **kw):
        cols = _fake_columns()
        cols["tau"] = np.ones(3)
        return cols

    _install_fake(monkeypatch, _with_extra)
    with pytest.raises(ValueError, match="tau"):
        pdg.generate_pretrain_data_npz(str(tmp_path), atoms=_TINY,
                                       basis="sto-3g", grid_level=0)
    assert not os.listdir(tmp_path)


def test_writer_refuses_a_single_precision_column(monkeypatch, tmp_path):
    """A float32 column is a column computed in single precision; casting it
    up would not recover the lost digits, so the file is not written."""
    def _f32(system, **kw):
        cols = _fake_columns()
        cols["metagga"] = cols["metagga"].astype(np.float32)
        return cols

    _install_fake(monkeypatch, _f32)
    with pytest.raises(ValueError, match="float64"):
        pdg.generate_pretrain_data_npz(str(tmp_path), atoms=_TINY,
                                       basis="sto-3g", grid_level=0)
    assert not os.listdir(tmp_path)


# ---------------------------------------------------------------------------
# The reader: bit-for-bit, and closed in both directions
# ---------------------------------------------------------------------------

def test_loader_round_trips_every_column_bit_for_bit(tmp_path):
    path, raw = _gen(tmp_path, exchange_footing="spin_channel")
    got = pdg.load_pretrain_data_npz(path)
    assert set(got) == set(raw)
    for key in raw:
        assert got[key].dtype == raw[key].dtype, key
        assert got[key].shape == raw[key].shape, key
        np.testing.assert_array_equal(got[key], raw[key], err_msg=key)
        if key.startswith("system"):
            assert got[key].dtype == np.int32, key
        else:
            assert got[key].dtype == np.float64, key


def test_loader_accepts_a_legacy_file(tmp_path):
    """A file written before the protocol (the recorded fixture stripped of
    the keys the protocol added, ``_legacy_view``) carries the total-density
    block and the mesh but no system table; it loads, and its layout says so,
    because an existing production file is still valid data for the
    point-wise loss."""
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, **_legacy_view(dict(np.load(_FIXTURE))))
    got = pdg.load_pretrain_data_npz(str(legacy))
    assert "system_all" not in got
    assert pdg.pretrain_npz_layout(set(got)) == {
        "polarized": True, "descriptors": True, "exchange_footing": "total",
        "system_table": False, "mesh": True}


# ---------------------------------------------------------------------------
# Data identity: what forces a regeneration
# ---------------------------------------------------------------------------


def test_manifest_records_the_new_identity(tmp_path):
    path, _got = _gen(tmp_path, exchange_footing="spin_channel",
                      mesh_fraction=0.4)
    meta = pdg.read_pretrain_manifest(path)
    assert meta["basis"] == "sto-3g" and meta["grid_level"] == 0
    assert meta["reference_xc"] == "pbe"
    assert meta["exchange_footing"] == "spin_channel"
    assert meta["mesh"]["weight_fraction"] == 0.4
    assert [row[0] for row in meta["systems"]] == ["He", "H"]
    assert meta["systems"][1] == ["H", "H 0 0 0", 0, 1]
    # The legacy projection stays, so a manifest reader written before the set
    # became a system list still sees an atom list.
    assert meta["atoms"] == [["He", 0], ["H", 1]]
    assert meta["orientation_lock_strength"] == \
        pdg.PRETRAIN_ORIENTATION_LOCK_STRENGTH == 3e-5
    assert meta["x64"] is True
    from xcquinox.pipeline.metagga import ALPHA_DEFINITION
    assert meta["alpha_definition"] == ALPHA_DEFINITION \
        == "smooth_positive_part:width=1e-05"


def test_currency_check_keys_on_every_integration_and_lock_ingredient(
        tmp_path):
    path, _got = _gen(tmp_path)
    sysm = pdg.resolve_pretrain_systems(atoms=_TINY)
    base = dict(basis="sto-3g", grid_level=0, systems=sysm)
    assert pdg.pretrain_data_is_current(path, **base) is True
    for what, change in (("basis", dict(basis="def2-svp")),
                         ("grid", dict(grid_level=1)),
                         ("auxbasis", dict(auxbasis="def2-universal-jkfit")),
                         ("lock", dict(orientation_lock_strength=0.0)),
                         ("lock", dict(orientation_lock_strength=1e-4)),
                         ("x64", dict(x64=False))):
        assert pdg.pretrain_data_is_current(
            path, **{**base, **change}) is False, what


def test_ensure_is_idempotent_at_the_new_identity(tmp_path):
    p1 = pdg.ensure_pretrain_data(str(tmp_path), atoms=_TINY, basis="sto-3g",
                                  grid_level=0, polarized=True,
                                  descriptors=True,
                                  exchange_footing="spin_channel")
    mtime = os.path.getmtime(p1)
    p2 = pdg.ensure_pretrain_data(str(tmp_path), atoms=_TINY, basis="sto-3g",
                                  grid_level=0, polarized=True,
                                  descriptors=True,
                                  exchange_footing="spin_channel")
    assert p1 == p2
    assert os.path.getmtime(p2) == mtime
    layout = pdg.pretrain_npz_layout(set(pdg.load_pretrain_data_npz(p2)))
    assert layout["exchange_footing"] == "spin_channel"


# ---------------------------------------------------------------------------
# The irreproducible-degenerate refusal: a coarse grid OR an unlocked SCF
# ---------------------------------------------------------------------------


def test_generator_refuses_an_unlocked_degenerate_atom_at_a_fine_grid(tmp_path):
    """A fine grid is not sufficient. With the lock OFF the SCF may land on
    any orientation of the 2p hole, so independent draws of the O atom at grid
    level 3 keep different numbers of rows and disagree at the 3e-7 Ha level
    in the total energy -- a different file at one manifest identity. The
    refusal covers the lock as well as the grid."""
    with pytest.raises(ValueError, match="orientation lock") as excinfo:
        pdg.generate_pretrain_data_npz(str(tmp_path), atoms=(("O", 2),),
                                       basis="sto-3g", grid_level=3,
                                       orientation_lock_strength=0.0)
    message = str(excinfo.value)
    assert "O" in message
    assert "grid level 3" in message
    assert "allow_irreproducible_degenerate" in message
    # Row COUNTS differ; the energy spread is stated to one order (2.9e-7 Ha
    # over three draws here against 2.6e-7 Ha when the guard was written).
    assert "row counts" in message
    assert "3e-7 Ha" in message
    assert "11682" not in message
    assert not os.listdir(tmp_path)


# ---------------------------------------------------------------------------
# The three layout keys name themselves when they go missing
# ---------------------------------------------------------------------------
#
# ``zeta_all``, ``cusp_all`` and ``rho_x`` are not ordinary columns: their
# PRESENCE is what declares the polarization, the descriptors and the exchange
# footing. A file that lost one reads as a file written without it, and the
# refusal then names the columns that go with the missing key -- every one of
# them present and correct -- while the key itself is never mentioned.


# ---------------------------------------------------------------------------
# The exchange footing is a property of the FILE, not of the manifest alone
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The precision field, end to end
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The waiver cannot reach a caller that granted none
# ---------------------------------------------------------------------------


