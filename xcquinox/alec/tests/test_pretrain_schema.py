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
import json
import os

import numpy as np
import pytest

import xcquinox.alec.pretrain_data_gen as pdg


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
    DEFERRED_WORK.md entry 27). Against the previous recording exactly two
    keys moved: ``metagga_all`` on 1200 of 1200 rows, from the hard clip's
    0.0 (largest raw residue 1.4e-10) to the smoothing's floor 5.0e-6 (both
    atoms are one orbital in this basis), and ``metagga_mesh`` on 560 of 560
    rows by at most 5.0e-6 (the alpha = 0 nodes by the floor, the others by
    ``width^2 / (4 alpha)`` <= 2.5e-10); every other key -- rho, sigma, the
    PBE and SCAN targets, cusp, dm, rung35, rung35ms, weights, zeta, on the
    atomic rows and on the mesh -- is bit-identical."""
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


def test_default_output_adds_only_the_documented_new_keys(tmp_path):
    ref = dict(np.load(_FIXTURE))
    _path, got = _gen(tmp_path)
    assert set(got) == set(ref)
    assert sorted(set(got) - set(_legacy_view(ref))) == _NEW_DEFAULT_KEYS


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


def test_polarized_correlation_baseline_matches_the_model(tmp_path):
    """e_lda_c_all / rho_all is the libxc PW92 baseline the Fc ratio divided by.
    The production correlation path multiplies the network's F_c by
    utils.pw92c_polarized_scalar at the same zeta, so the two must be the same
    function or the pretraining energy target is not the production energy."""
    import jax.numpy as jnp
    from xcquinox.utils import pw92c_polarized_scalar
    _path, got = _gen(tmp_path)
    rho = got["rho_all"]
    zeta = got["zeta_all"]
    half = 0.5 * (1.0 + zeta)
    ours = np.asarray(pw92c_polarized_scalar(jnp.asarray(rho * half),
                                             jnp.asarray(rho * (1.0 - half))))
    stored = got["e_lda_c_all"] / rho
    dense = rho >= 1e-6
    assert int(dense.sum()) > 100
    np.testing.assert_allclose(stored[dense], ours[dense],
                               rtol=_PW92_POINTWISE_RTOL, atol=0)
    gap = float(np.sum(np.abs(got["weights_all"] * rho * (ours - stored))))
    assert gap < _PW92_INTEGRATED_GAP, gap


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


def test_closed_shell_exchange_rows_are_the_total_density_rows(tmp_path):
    _path, got = _gen(tmp_path, exchange_footing="spin_channel")
    he_x = got["system_x"] == 0
    he_a = got["system_all"] == 0
    for stem in ("rho", "sigma", "Fx", "Fx_scan", "metagga", "weights",
                 "e_lda_x", "cusp", "dm", "rung35", "rung35ms"):
        np.testing.assert_array_equal(got[f"{stem}_x"][he_x],
                                      got[f"{stem}_all"][he_a], err_msg=stem)


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


def test_exchange_energy_table_uses_the_exchange_block(tmp_path):
    _path, got = _gen(tmp_path, exchange_footing="spin_channel")
    for s in (0, 1):
        rows = got["system_x"] == s
        expect = float(np.sum(got["weights_x"][rows] * got["e_lda_x_x"][rows]
                              * (1.0 + got["Fx_x"][rows])))
        assert got["e_x_parent_sys"][s] == pytest.approx(expect, rel=0,
                                                         abs=1e-12)
        expect_scan = float(np.sum(got["weights_x"][rows]
                                   * got["e_lda_x_x"][rows]
                                   * (1.0 + got["Fx_scan_x"][rows])))
        assert got["e_x_parent_scan_sys"][s] == pytest.approx(
            expect_scan, rel=0, abs=1e-12)
    # ... and the same systems' total-density quadrature agrees on H to the
    # spin-scaling identity (exact on a one-electron system up to the rows the
    # floor treats differently, measured 3.2e-12 Ha on O).
    rows = got["system_all"] == 1
    total = float(np.sum(got["weights_all"][rows] * got["e_lda_x_all"][rows]
                         * (1.0 + got["Fx_all"][rows])))
    assert abs(got["e_x_parent_sys"][1] - total) < 1e-9


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


def test_systems_argument_overrides_the_composition_flags(tmp_path):
    """ensure_pretrain_data resolves the set once and hands the SAME tuple to
    the currency check and to the generator, so the two can never disagree."""
    sysm = (pdg.PretrainSystem("He", "He 0 0 0", 0, 0),)
    _path, got = _gen(tmp_path, systems=sysm, atoms=(("H", 1),))
    assert got["system_natoms"].tolist() == [1]
    assert sorted(set(got["system_all"].tolist())) == [0]


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


def test_writer_refuses_an_unknown_exchange_column(monkeypatch, tmp_path):
    def _with_extra(system, **kw):
        x = {k: v for k, v in _fake_columns(4).items()
             if k in ("rho", "sigma", "Fx", "Fx_scan", "metagga", "weights",
                      "cusp", "dm", "rung35", "rung35ms")}
        x["Fc"] = np.ones(4)
        return _fake_columns(x_rows=x)

    _install_fake(monkeypatch, _with_extra)
    with pytest.raises(ValueError, match="Fc_x"):
        pdg.generate_pretrain_data_npz(str(tmp_path), atoms=_TINY,
                                       basis="sto-3g", grid_level=0,
                                       exchange_footing="spin_channel")


def test_writer_refuses_a_missing_column(monkeypatch, tmp_path):
    def _without(system, **kw):
        cols = _fake_columns()
        cols.pop("e_lda_c")
        return cols

    _install_fake(monkeypatch, _without)
    with pytest.raises(ValueError, match="e_lda_c_all"):
        pdg.generate_pretrain_data_npz(str(tmp_path), atoms=_TINY,
                                       basis="sto-3g", grid_level=0)


def test_writer_refuses_systems_with_different_column_sets(monkeypatch,
                                                           tmp_path):
    def _uneven(system, **kw):
        return _fake_columns(polarized=(system.name == "He"))

    _install_fake(monkeypatch, _uneven)
    with pytest.raises(ValueError, match="zeta"):
        pdg.generate_pretrain_data_npz(str(tmp_path), atoms=_TINY,
                                       basis="sto-3g", grid_level=0)


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


def test_writer_refuses_misaligned_columns(monkeypatch, tmp_path):
    def _short(system, **kw):
        cols = _fake_columns()
        cols["Fc"] = np.ones(2)
        return cols

    _install_fake(monkeypatch, _short)
    with pytest.raises(ValueError, match="Fc_all"):
        pdg.generate_pretrain_data_npz(str(tmp_path), atoms=_TINY,
                                       basis="sto-3g", grid_level=0)


@pytest.mark.parametrize("polarized", [False, True])
@pytest.mark.parametrize("descriptors", [False, True])
@pytest.mark.parametrize("footing", ["total", "spin_channel"])
def test_written_keys_are_exactly_the_declared_schema(monkeypatch, tmp_path,
                                                      polarized, descriptors,
                                                      footing):
    def _cols(system, **kw):
        x = None
        if footing == "spin_channel" and system.name == "H":
            keep = ("rho", "sigma", "Fx", "Fx_scan", "metagga", "weights",
                    "cusp", "dm", "rung35", "rung35ms")
            x = {k: v for k, v in _fake_columns(
                5, polarized=False, descriptors=descriptors).items()
                 if k in keep}
        return _fake_columns(polarized=polarized, descriptors=descriptors,
                             x_rows=x)

    _install_fake(monkeypatch, _cols)
    path = pdg.generate_pretrain_data_npz(
        str(tmp_path), atoms=_TINY, basis="sto-3g", grid_level=0,
        polarized=polarized, descriptors=descriptors, exchange_footing=footing)
    with np.load(path) as z:
        keys = set(z.files)
    assert keys == pdg.pretrain_npz_keys(polarized=polarized,
                                         descriptors=descriptors,
                                         exchange_footing=footing)
    layout = pdg.pretrain_npz_layout(keys)
    assert layout == {"polarized": polarized, "descriptors": descriptors,
                      "exchange_footing": footing, "system_table": True,
                      "mesh": True}


def test_generator_validates_its_arguments_before_any_scf(monkeypatch,
                                                          tmp_path):
    calls = _install_fake(monkeypatch, lambda system, **kw: _fake_columns())
    for bad in (dict(reference_xc="blyp"), dict(exchange_footing="per_orbital"),
                dict(mesh_fraction=0.0), dict(mesh_fraction=1.0),
                dict(mesh_fraction=-0.2)):
        with pytest.raises(ValueError):
            pdg.generate_pretrain_data_npz(str(tmp_path), atoms=_TINY,
                                           basis="sto-3g", grid_level=0, **bad)
    assert calls == []
    with pytest.raises(ValueError, match="empty"):
        pdg.generate_pretrain_data_npz(str(tmp_path), atoms=(),
                                       basis="sto-3g", grid_level=0)


def test_generator_releases_the_precompute_cache(tmp_path):
    """Each system's MoleculeData holds the (4, n_grid, n_ao) AO table; the
    generator drops it once the columns are extracted so a long set does not
    accumulate them."""
    import xcquinox.alec.data as data_mod
    _gen(tmp_path)
    assert data_mod._PRECOMPUTE_CACHE == {}


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


def _rewrite(path, arrays):
    np.savez(path, **arrays)


def test_loader_refuses_a_file_missing_a_required_block(tmp_path):
    path, raw = _gen(tmp_path, exchange_footing="spin_channel")
    for key in ("weights_x", "system_natoms", "rho_all", "Fc_scan_mesh",
                "e_lda_x_x"):
        arrays = dict(raw)
        arrays.pop(key)
        _rewrite(path, arrays)
        with pytest.raises(ValueError, match=key):
            pdg.load_pretrain_data_npz(path)


def test_loader_refuses_an_unknown_key(tmp_path):
    path, raw = _gen(tmp_path)
    arrays = dict(raw)
    arrays["tau_all"] = np.ones_like(raw["rho_all"])
    _rewrite(path, arrays)
    with pytest.raises(ValueError, match="tau_all"):
        pdg.load_pretrain_data_npz(path)


def test_loader_refuses_a_single_precision_or_misaligned_column(tmp_path):
    path, raw = _gen(tmp_path)
    arrays = dict(raw)
    arrays["rung35_all"] = raw["rung35_all"].astype(np.float32)
    _rewrite(path, arrays)
    with pytest.raises(ValueError, match="rung35_all"):
        pdg.load_pretrain_data_npz(path)
    arrays = dict(raw)
    arrays["e_x_parent_sys"] = raw["e_x_parent_sys"][:1]
    _rewrite(path, arrays)
    with pytest.raises(ValueError, match="e_x_parent_sys"):
        pdg.load_pretrain_data_npz(path)
    arrays = dict(raw)
    arrays["system_all"] = raw["system_all"] + 5
    _rewrite(path, arrays)
    with pytest.raises(ValueError, match="system_all"):
        pdg.load_pretrain_data_npz(path)


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

def _legacy_stub(tmp_path, atoms=(("H", 1),), **manifest_extra):
    """A file and manifest in the pre-protocol format: the historical columns
    and the mesh, a manifest carrying basis / grid / DF / auxbasis / atoms."""
    p = tmp_path / "pretrain_data.npz"
    np.savez(p, rho_all=np.ones(3), sigma_all=np.ones(3), Fx_all=np.ones(3),
             Fc_all=np.ones(3), weights_all=np.ones(3),
             metagga_all=np.ones((3, 1)), Fx_scan_all=np.ones(3),
             Fc_scan_all=np.ones(3), rho_mesh=np.ones(4),
             sigma_mesh=np.ones(4), Fx_scan_mesh=np.ones(4),
             Fc_scan_mesh=np.ones(4), metagga_mesh=np.ones((4, 1)),
             weights_mesh=np.ones(4))
    meta = {"basis": "def2-svp", "grid_level": 1, "density_fit": False,
            "auxbasis": None, "atoms": [[s, sp] for s, sp in atoms]}
    meta.update(manifest_extra)
    with open(str(p) + ".manifest.json", "w") as f:
        json.dump(meta, f)
    return p


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
    from xcquinox.alec.metagga import ALPHA_DEFINITION
    assert meta["alpha_definition"] == ALPHA_DEFINITION \
        == "smooth_positive_part:width=1e-05"


def test_a_manifest_from_before_the_indicator_smoothing_is_not_current(
        tmp_path):
    """The iso-orbital indicator is a stored column, and its definition moved
    when its lower bound became a smooth positive part
    (``metagga.compute_alpha``): the hard clip wrote 0.0 on every one-orbital
    row where the smoothing writes width / 2 = 5e-6 (1200 of 1200 rows of
    the default set's H atom, the mesh's alpha = 0 nodes), with no other key
    changing. A manifest without ``alpha_definition`` (every file written
    before the key existed) reads as the hard-clipped definition and is
    stale at every identity; one naming another definition is stale too;
    the live one is current. The key is not a request parameter -- the
    generator can only write the live definition -- so there is no identity
    at which a pre-smoothing file is served."""
    from xcquinox.alec.metagga import ALPHA_DEFINITION
    path, _got = _gen(tmp_path)
    sysm = pdg.resolve_pretrain_systems(atoms=_TINY)
    base = dict(basis="sto-3g", grid_level=0, systems=sysm)
    assert pdg.pretrain_data_is_current(path, **base) is True
    mpath = str(path) + ".manifest.json"
    meta = pdg.read_pretrain_manifest(path)
    assert meta["alpha_definition"] == ALPHA_DEFINITION
    for stale in ({k: v for k, v in meta.items() if k != "alpha_definition"},
                  {**meta, "alpha_definition": "hard_clip"},
                  {**meta, "alpha_definition":
                   "smooth_positive_part:width=1e-06"}):
        with open(mpath, "w") as f:
            json.dump(stale, f)
        assert pdg.pretrain_data_is_current(path, **base) is False, stale.get(
            "alpha_definition", "<absent>")
    with open(mpath, "w") as f:
        json.dump(meta, f)
    assert pdg.pretrain_data_is_current(path, **base) is True


def test_manifest_x64_flag_is_the_live_jax_configuration(tmp_path):
    """The flag records what JAX was computing in when the file was written;
    a file stamped single-precision is stale for a double-precision request.
    (Only the manifest writer is exercised with the flag off: the generator
    itself refuses to write a single-precision column.)"""
    import jax
    p = tmp_path / "pretrain_data.npz"
    np.savez(p, x=np.zeros(3))
    jax.config.update("jax_enable_x64", False)
    try:
        pdg._write_pretrain_manifest(p, basis="def2-svp", grid_level=1,
                                     density_fit=False)
    finally:
        jax.config.update("jax_enable_x64", True)
    assert pdg.read_pretrain_manifest(p)["x64"] is False
    assert pdg.pretrain_data_is_current(p, basis="def2-svp",
                                        grid_level=1) is False
    assert pdg.pretrain_data_is_current(p, basis="def2-svp", grid_level=1,
                                        x64=False) is True


def test_currency_check_legacy_manifest_is_current_at_the_legacy_identity(
        tmp_path):
    """A file written before the protocol change carries no reference_xc /
    footing / systems / lock / x64 keys. They read as the values the
    historical generator used -- PBE, the total footing, the atom list, NO
    orientation lock, double precision -- so, with its indicator definition
    stated, the file is current for a request at that identity and stale for
    the production one, whose lock the historical rows of a degenerate atom
    were not computed at. Without the ``alpha_definition`` key the file
    predates the smoothing of the indicator's lower bound and its alpha rows
    are the hard-clipped ones, so it is stale at every identity (the
    definition is not a request parameter; see
    ``test_a_manifest_from_before_the_indicator_smoothing_is_not_current``)."""
    from xcquinox.alec.metagga import ALPHA_DEFINITION
    p = _legacy_stub(tmp_path)
    assert pdg.pretrain_data_is_current(
        p, basis="def2-svp", grid_level=1, atoms=[("H", 1)],
        orientation_lock_strength=0.0) is False
    p = _legacy_stub(tmp_path, alpha_definition=ALPHA_DEFINITION)
    assert pdg.pretrain_data_is_current(
        p, basis="def2-svp", grid_level=1, atoms=[("H", 1)],
        orientation_lock_strength=0.0) is True
    assert pdg.pretrain_data_is_current(
        p, basis="def2-svp", grid_level=1, atoms=[("H", 1)]) is False
    assert pdg.pretrain_data_is_current(
        p, basis="def2-svp", grid_level=1,
        systems=pdg.resolve_pretrain_systems(atoms=[("H", 1)]),
        orientation_lock_strength=0.0) is True


def test_currency_check_footing_is_part_of_the_identity(tmp_path):
    path, _got = _gen(tmp_path)
    sysm = pdg.resolve_pretrain_systems(atoms=_TINY)
    assert pdg.pretrain_data_is_current(
        path, basis="sto-3g", grid_level=0, systems=sysm) is True
    assert pdg.pretrain_data_is_current(
        path, basis="sto-3g", grid_level=0, systems=sysm,
        exchange_footing="spin_channel") is False


def test_currency_check_reference_density_is_part_of_the_identity(tmp_path):
    path, _got = _gen(tmp_path)
    sysm = pdg.resolve_pretrain_systems(atoms=_TINY)
    assert pdg.pretrain_data_is_current(
        path, basis="sto-3g", grid_level=0, systems=sysm,
        reference_xc="scan") is False


def test_currency_check_mesh_fraction_is_part_of_the_identity(tmp_path):
    path, _got = _gen(tmp_path)
    sysm = pdg.resolve_pretrain_systems(atoms=_TINY)
    assert pdg.pretrain_data_is_current(
        path, basis="sto-3g", grid_level=0, systems=sysm,
        mesh_fraction=0.4) is False


def test_currency_check_system_list_is_part_of_the_identity(tmp_path):
    """The set is keyed on every field of the system: a geometry moved by
    1e-3 Angstrom, another spin, another charge or another membership is
    another file."""
    path, _got = _gen(tmp_path)
    same = pdg.resolve_pretrain_systems(atoms=_TINY)
    assert pdg.pretrain_data_is_current(
        path, basis="sto-3g", grid_level=0, systems=same) is True
    he, h = same
    variants = {
        "membership": (he,),
        "order": (h, he),
        "geometry": (he, pdg.PretrainSystem("H", "H 0 0 0.001", 0, 1)),
        "spin": (he, pdg.PretrainSystem("H", "H 0 0 0", 0, 3)),
        "charge": (he, pdg.PretrainSystem("H-", "H 0 0 0", -1, 0)),
    }
    for what, other in variants.items():
        assert pdg.pretrain_data_is_current(
            path, basis="sto-3g", grid_level=0, systems=other) is False, what


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


def test_manifest_without_a_system_list_is_accepted_through_its_atoms(
        tmp_path):
    """A manifest written before the set became a system list identifies its
    rows through ``atoms``: the legacy generator could only write neutral free
    atoms at the origin, in that order, so a request for exactly those systems
    is shown to be held; a molecule, an ion, a displaced atom or another order
    cannot be, and regenerates."""
    path, _got = _gen(tmp_path)
    meta = pdg.read_pretrain_manifest(path)
    meta.pop("systems")
    with open(str(path) + ".manifest.json", "w") as f:
        json.dump(meta, f)
    he, h = pdg.resolve_pretrain_systems(atoms=_TINY)
    assert pdg.pretrain_data_is_current(
        path, basis="sto-3g", grid_level=0, systems=(he, h)) is True
    for other in ((h, he),
                  (he, pdg.PretrainSystem("H", "H 0 0 0.001", 0, 1)),
                  (he, pdg.PretrainSystem("H-", "H 0 0 0", -1, 0)),
                  (he, pdg.PretrainSystem("h2", "H 0 0 0; H 0 0 0.74", 0, 0)),
                  (he,)):
        assert pdg.pretrain_data_is_current(
            path, basis="sto-3g", grid_level=0, systems=other) is False


def test_ensure_resolves_the_set_once(monkeypatch, tmp_path):
    """The currency check and the generation must see the SAME resolved tuple:
    resolving twice would let a non-deterministic inventory silently regenerate
    on every call."""
    seen = []

    def _fake_generate(out_dir, **kw):
        seen.append(kw["systems"])
        p = os.path.join(out_dir, pdg.pretrain_data_filename(
            kw["polarized"], kw["reference_xc"]))
        np.savez(p, rho_all=np.ones(1))
        return p

    monkeypatch.setattr(pdg, "generate_pretrain_data_npz", _fake_generate)
    # The pool set carries eight spatially degenerate atoms and this test runs
    # at grid level 0 deliberately (the generator is faked; no SCF is paid),
    # so the irreproducible-degenerate refusal is waived here.
    pdg.ensure_pretrain_data(str(tmp_path), basis="sto-3g", grid_level=0,
                             pool_atoms=True,
                             allow_irreproducible_degenerate=True)
    assert len(seen) == 1
    assert len(seen[0]) == 14


def test_ensure_uses_the_reference_specific_filename(monkeypatch, tmp_path):
    paths = []

    def _fake_generate(out_dir, **kw):
        p = os.path.join(out_dir, pdg.pretrain_data_filename(
            kw["polarized"], kw["reference_xc"]))
        paths.append(p)
        np.savez(p, rho_all=np.ones(1))
        return p

    monkeypatch.setattr(pdg, "generate_pretrain_data_npz", _fake_generate)
    # Default set (O is degenerate) at grid level 0 deliberately: the test is
    # about the FILENAME, and the generator is faked.
    pdg.ensure_pretrain_data(str(tmp_path), basis="sto-3g", grid_level=0,
                             reference_xc="scan", polarized=True,
                             allow_irreproducible_degenerate=True)
    assert os.path.basename(paths[0]) == "pretrain_data_polarized_scan.npz"


def test_ensure_hands_the_full_identity_to_the_generator(monkeypatch,
                                                         tmp_path):
    seen = {}

    def _fake_generate(out_dir, **kw):
        seen.update(kw)
        p = os.path.join(out_dir, pdg.pretrain_data_filename(
            kw["polarized"], kw["reference_xc"]))
        np.savez(p, rho_all=np.ones(1))
        return p

    monkeypatch.setattr(pdg, "generate_pretrain_data_npz", _fake_generate)
    pdg.ensure_pretrain_data(
        str(tmp_path), atoms=_TINY, basis="sto-3g", grid_level=0,
        polarized=False, descriptors=False, reference_xc="scan",
        exchange_footing="spin_channel", mesh_fraction=0.25,
        orientation_lock_strength=1e-4, cusp_log_transform=False,
        progress=True)
    assert seen["systems"] == pdg.resolve_pretrain_systems(atoms=_TINY)
    for key, value in (("basis", "sto-3g"), ("grid_level", 0),
                       ("polarized", False), ("descriptors", False),
                       ("reference_xc", "scan"),
                       ("exchange_footing", "spin_channel"),
                       ("mesh_fraction", 0.25),
                       ("orientation_lock_strength", 1e-4),
                       ("cusp_log_transform", False), ("progress", True),
                       ("density_fit", False), ("auxbasis", None)):
        assert seen[key] == value, key


def test_ensure_regenerates_a_legacy_directory_once(monkeypatch, tmp_path):
    """An existing data directory (atom list, no lock) is regenerated on the
    next ensure at the production identity, and only once."""
    calls = []

    def _fake_generate(out_dir, **kw):
        calls.append(kw["systems"])
        p = os.path.join(out_dir, pdg.pretrain_data_filename(
            kw["polarized"], kw["reference_xc"]))
        np.savez(p, rho_all=np.ones(1))
        pdg._write_pretrain_manifest(
            p, basis=kw["basis"], grid_level=kw["grid_level"],
            density_fit=kw["density_fit"], auxbasis=kw["auxbasis"],
            systems=kw["systems"],
            atoms=tuple((s.name, s.spin) for s in kw["systems"]),
            reference_xc=kw["reference_xc"],
            exchange_footing=kw["exchange_footing"],
            mesh_fraction=kw["mesh_fraction"],
            orientation_lock_strength=kw["orientation_lock_strength"])
        return p

    monkeypatch.setattr(pdg, "generate_pretrain_data_npz", _fake_generate)
    _legacy_stub(tmp_path)
    pdg.ensure_pretrain_data(str(tmp_path), atoms=(("H", 1),),
                             basis="def2-svp", grid_level=1, polarized=False)
    pdg.ensure_pretrain_data(str(tmp_path), atoms=(("H", 1),),
                             basis="def2-svp", grid_level=1, polarized=False)
    assert len(calls) == 1


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

def _fake_all_columns(monkeypatch):
    """Every system yields a complete column set, so these tests exercise the
    guard without paying an SCF."""
    return _install_fake(monkeypatch, lambda system, **kw: _fake_columns())


def test_generator_refuses_a_degenerate_atom_below_grid_level_3(tmp_path):
    """At the generator's own default grid level the locked O rows are NOT
    reproducible between processes -- across separate sets of draws rho
    spreads at the 1e-3..1e-1 level, the iso-orbital indicator by of order
    unity and the stored E_x at the 1e-6 Ha level, against 3e-11 relative at
    grid level 3 -- while the manifest records an identity the file therefore
    does not have. The generation is refused rather than written.

    The spreads are quoted as ORDERS OF MAGNITUDE because they are samples of
    a process-to-process scatter rather than bounds: two independent sets of
    draws measured 3e-3 / 0.64 / 3.7e-6 Ha and 5.7e-2 / 12.4 / 1.3e-6 Ha, so a
    single figure would be read as a reproducible quantity and is not one. The
    indicator's own spread ran 0.55 to 2.46 over six draw pairs, three of them
    below unity, so the message says "of order unity" rather than naming a
    floor no draw pair establishes."""
    with pytest.raises(ValueError, match="grid level") as excinfo:
        pdg.generate_pretrain_data_npz(str(tmp_path), atoms=(("O", 2),),
                                       basis="sto-3g", grid_level=1)
    message = str(excinfo.value)
    assert "O" in message
    assert "grid level 1" in message
    assert "allow_irreproducible_degenerate" in message
    # The remedy an operator reading a datagen log actually has: the YAML key.
    assert "inputs.allow_irreproducible_degenerate" in message
    assert "inputs.irreproducible_degenerate_reason" in message
    # The order-of-magnitude form spans both sets of draws.
    assert "1e-3..1e-1" in message
    assert "of order unity" in message
    assert "1e-6 Ha" in message
    assert "0.64" not in message and "3e-3" not in message
    assert not os.listdir(tmp_path)


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


def test_ensure_refuses_the_irreproducible_degenerate_identity(tmp_path):
    """The identity is refused, not only the generation: a file already on
    disk at that identity would otherwise be served to a caller the generator
    itself would have refused. Both conditions are checked there too."""
    with pytest.raises(ValueError, match="grid level"):
        pdg.ensure_pretrain_data(str(tmp_path), atoms=(("O", 2),),
                                 basis="sto-3g", grid_level=1)
    with pytest.raises(ValueError, match="orientation lock"):
        pdg.ensure_pretrain_data(str(tmp_path), atoms=(("O", 2),),
                                 basis="sto-3g", grid_level=3,
                                 orientation_lock_strength=0.0)


def test_generator_accepts_a_degenerate_atom_at_the_production_identity(
        monkeypatch, tmp_path):
    """Grid level 3 AND the lock on: the rows reproduce to 3e-11 relative, so
    nothing is refused and nothing is recorded as waived."""
    _fake_all_columns(monkeypatch)
    path = pdg.generate_pretrain_data_npz(str(tmp_path), atoms=(("O", 2),),
                                          basis="sto-3g", grid_level=3)
    meta = json.loads(open(pdg._pretrain_manifest_path(path)).read())
    assert meta["allow_irreproducible_degenerate"] is False


@pytest.mark.parametrize("grid_level,lock", [
    (1, pdg.PRETRAIN_ORIENTATION_LOCK_STRENGTH),   # coarse grid, locked
    (3, 0.0),                                      # fine grid, unlocked
    (1, 0.0),                                      # both
])
def test_the_flag_waives_either_condition(monkeypatch, tmp_path, grid_level,
                                          lock):
    """The escape hatch is explicit and recorded: a file built through it
    carries the flag, so a reader can see that its degenerate-atom rows are
    one arbitrary member of the manifold. One flag covers both conditions --
    the defect is the same one either way, a manifest identity the file does
    not have."""
    _fake_all_columns(monkeypatch)
    path = pdg.generate_pretrain_data_npz(
        str(tmp_path), atoms=(("O", 2),), basis="sto-3g",
        grid_level=grid_level, orientation_lock_strength=lock,
        allow_irreproducible_degenerate=True)
    meta = json.loads(open(pdg._pretrain_manifest_path(path)).read())
    assert meta["allow_irreproducible_degenerate"] is True


@pytest.mark.parametrize("grid_level,lock", [(1, 0.0), (3, 0.0), (1, 3e-5)])
def test_a_spherical_atom_is_unaffected_by_either_condition(monkeypatch,
                                                            tmp_path,
                                                            grid_level, lock):
    """N is a half-filled p shell, spherically symmetric, so its rows depend
    neither on an orientation the SCF happened to reach nor on a bias that
    selects one."""
    _fake_all_columns(monkeypatch)
    path = pdg.generate_pretrain_data_npz(str(tmp_path), atoms=(("N", 3),),
                                          basis="sto-3g",
                                          grid_level=grid_level,
                                          orientation_lock_strength=lock)
    meta = json.loads(open(pdg._pretrain_manifest_path(path)).read())
    assert meta["allow_irreproducible_degenerate"] is False


# ---------------------------------------------------------------------------
# The three layout keys name themselves when they go missing
# ---------------------------------------------------------------------------
#
# ``zeta_all``, ``cusp_all`` and ``rho_x`` are not ordinary columns: their
# PRESENCE is what declares the polarization, the descriptors and the exchange
# footing. A file that lost one reads as a file written without it, and the
# refusal then names the columns that go with the missing key -- every one of
# them present and correct -- while the key itself is never mentioned.

@pytest.mark.parametrize("sentinel,companion", [
    ("zeta_all", "zeta_mesh"),
    ("cusp_all", "dm_all"),
    ("rho_x", "sigma_x"),
])
def test_a_deleted_layout_key_is_named_by_the_reader(sentinel, companion):
    keys = pdg.pretrain_npz_keys(polarized=True, descriptors=True,
                                 exchange_footing="spin_channel")
    with pytest.raises(ValueError) as excinfo:
        pdg.pretrain_npz_layout(keys - {sentinel})
    message = str(excinfo.value)
    assert sentinel in message, message
    # the companions are still reported, as the evidence rather than as the
    # defect
    assert companion in message, message


def test_a_configuration_without_a_block_is_not_a_missing_layout_key():
    """The other direction: a file written unpolarized, descriptor-free or on
    the total footing carries neither the layout key nor its companions, and
    is a configuration rather than a torn file."""
    for polarized in (False, True):
        for descriptors in (False, True):
            for footing in ("total", "spin_channel"):
                keys = pdg.pretrain_npz_keys(polarized=polarized,
                                             descriptors=descriptors,
                                             exchange_footing=footing)
                layout = pdg.pretrain_npz_layout(keys)
                assert layout["polarized"] is polarized
                assert layout["descriptors"] is descriptors
                assert layout["exchange_footing"] == footing


def test_a_legacy_file_keeps_its_layout_reading():
    """A pre-protocol file (no system table) carries the historical columns
    and, when it was written with them, the descriptors; the layout keys read
    the same way there."""
    legacy = {f"{s}_all" for s in ("rho", "sigma", "Fx", "Fc", "weights",
                                   "Fx_scan", "Fc_scan", "metagga", "zeta",
                                   "cusp", "dm", "rung35", "rung35ms")}
    layout = pdg.pretrain_npz_layout(legacy)
    assert layout == {"polarized": True, "descriptors": True,
                      "exchange_footing": "total", "system_table": False,
                      "mesh": False}
    with pytest.raises(ValueError, match="cusp_all"):
        pdg.pretrain_npz_layout(legacy - {"cusp_all"})


def test_the_reader_refuses_a_torn_file_by_its_missing_key(monkeypatch,
                                                           tmp_path):
    """End to end through the reader: a written file with one layout key
    stripped is refused on the way back in, naming the key."""
    def _cols(system, **kw):
        keep = ("rho", "sigma", "Fx", "Fx_scan", "metagga", "weights",
                "cusp", "dm", "rung35", "rung35ms")
        x = {k: v for k, v in _fake_columns(5, polarized=False).items()
             if k in keep}
        return _fake_columns(x_rows=x)

    _install_fake(monkeypatch, _cols)
    path = pdg.generate_pretrain_data_npz(
        str(tmp_path), atoms=_TINY, basis="sto-3g", grid_level=0,
        exchange_footing="spin_channel")
    with np.load(path) as z:
        arrays = {k: np.array(z[k]) for k in z.files if k != "rho_x"}
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="rho_x"):
        pdg.load_pretrain_data_npz(path)


# ---------------------------------------------------------------------------
# The exchange footing is a property of the FILE, not of the manifest alone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("built,declared", [("total", "spin_channel"),
                                            ("spin_channel", "total")])
def test_a_manifest_declaring_the_other_footing_is_not_current(monkeypatch,
                                                               tmp_path,
                                                               built,
                                                               declared):
    """The ``*_x`` block IS the spin_channel footing and its absence IS the
    total one, so a manifest that names one beside a file carrying the other
    describes rows the file does not hold. Every other identity key matches,
    so without this the file would be served and the pretraining objective
    would read exchange rows on a footing the run did not ask for."""
    def _cols(system, **kw):
        x = None
        if built == "spin_channel":
            keep = ("rho", "sigma", "Fx", "Fx_scan", "metagga", "weights",
                    "cusp", "dm", "rung35", "rung35ms")
            x = {k: v for k, v in _fake_columns(5, polarized=False).items()
                 if k in keep}
        return _fake_columns(x_rows=x)

    _install_fake(monkeypatch, _cols)
    path = pdg.generate_pretrain_data_npz(
        str(tmp_path), atoms=_TINY, basis="sto-3g", grid_level=0,
        exchange_footing=built)
    systems = pdg.resolve_pretrain_systems(atoms=_TINY)
    assert pdg.pretrain_data_is_current(
        path, basis="sto-3g", grid_level=0, systems=systems,
        exchange_footing=built) is True
    # the manifest is rewritten to declare the other footing; nothing else
    # about the file changes
    mpath = pdg._pretrain_manifest_path(path)
    meta = json.loads(open(mpath).read())
    meta["exchange_footing"] = declared
    with open(mpath, "w") as f:
        json.dump(meta, f)
    assert pdg.pretrain_data_is_current(
        path, basis="sto-3g", grid_level=0, systems=systems,
        exchange_footing=declared) is False


def test_the_footing_check_leaves_a_manifest_only_stub_alone(tmp_path):
    """The block comparison is gated on a real data file (``Fx_all``): a stub
    written by a test or a partial writer carries neither footing and is
    judged by the manifest keys alone, as before."""
    p = tmp_path / "pretrain_data.npz"
    np.savez(p, rho_all=np.ones(1))
    pdg._write_pretrain_manifest(p, basis="sto-3g", grid_level=0,
                                 density_fit=False, atoms=(("H", 1),),
                                 systems=pdg.resolve_pretrain_systems(
                                     atoms=(("H", 1),)),
                                 exchange_footing="spin_channel")
    assert pdg.pretrain_data_is_current(
        p, basis="sto-3g", grid_level=0,
        systems=pdg.resolve_pretrain_systems(atoms=(("H", 1),)),
        exchange_footing="spin_channel") is True


# ---------------------------------------------------------------------------
# The precision field, end to end
# ---------------------------------------------------------------------------

def test_generation_in_single_precision_is_refused_and_writes_nothing(
        tmp_path):
    """With ``jax_enable_x64`` off the generator does not produce a file.

    The manifest's ``x64`` field says a file was written in double precision;
    what makes that a fact rather than a claim is that the generator cannot
    write anything else. Two guards refuse it, in this order:

    - the grid-identity check in ``_system_columns`` fires first -- the AO
      table the precompute stored is single precision, and the replayed
      libcint evaluation of the same grid differs from it by far more than the
      1e-10 that check allows, so the run stops before a column is assembled
      (``RuntimeError``: the rebuilt integration grid ... is not the one
      ``precompute_fixed_density_data`` used);
    - with that check disabled the schema refuses the column itself
      (``ValueError``: pretrain data column 'cusp_all' is float32, not
      float64), which is the guarantee the manifest field records.

    With BOTH disabled a file is written, so this test measures the pair
    rather than either one. The same configuration with x64 on writes the file
    normally, which ties the refusal to the precision and not to the system.
    """
    import jax
    from xcquinox.alec.data import clear_precompute_cache
    kw = dict(atoms=(("He", 0),), basis="sto-3g", grid_level=0)
    control = tmp_path / "with_x64"
    assert pdg.generate_pretrain_data_npz(str(control), **kw)
    single = tmp_path / "without_x64"
    single.mkdir()
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises((RuntimeError, ValueError)) as excinfo:
            pdg.generate_pretrain_data_npz(str(single), **kw)
    finally:
        jax.config.update("jax_enable_x64", True)
        # the failed build left its single-precision MoleculeData in the
        # process-level precompute cache; a later double-precision test of the
        # same species would otherwise be handed it
        clear_precompute_cache()
    assert "quadrature" in str(excinfo.value) or "float64" in str(excinfo.value)
    assert os.listdir(single) == []


def test_a_single_precision_manifest_is_stale_for_a_double_precision_run(
        tmp_path):
    """The recorded precision is part of the identity end to end: a file whose
    manifest says it was computed in single precision is regenerated for a run
    in double, and current only for a request that asks for single."""
    path = pdg.generate_pretrain_data_npz(str(tmp_path), atoms=(("He", 0),),
                                          basis="sto-3g", grid_level=0)
    systems = pdg.resolve_pretrain_systems(atoms=(("He", 0),))
    ident = dict(basis="sto-3g", grid_level=0, systems=systems)
    assert pdg.pretrain_data_is_current(path, **ident) is True
    mpath = pdg._pretrain_manifest_path(path)
    meta = json.loads(open(mpath).read())
    assert meta["x64"] is True
    meta["x64"] = False
    with open(mpath, "w") as f:
        json.dump(meta, f)
    assert pdg.pretrain_data_is_current(path, **ident) is False
    assert pdg.pretrain_data_is_current(path, x64=False, **ident) is True


# ---------------------------------------------------------------------------
# The waiver cannot reach a caller that granted none
# ---------------------------------------------------------------------------

def test_a_waived_file_is_never_served_to_a_non_waiving_caller(monkeypatch,
                                                               tmp_path):
    """``pretrain_data_is_current`` does not compare the waiver, and does not
    need to.

    The generator's refusal is a function of the systems, the basis, the grid
    level and the lock -- all four members of the identity the currency check
    compares -- and ``ensure_pretrain_data`` applies it to the REQUESTED
    identity before asking whether the file is current. So a caller that
    matches a waived file necessarily reproduces that file's own refusal and
    is turned away first; at an identity that needs no waiver the flag was
    never exercised and the manifest records False. Both halves are checked
    here against a file that IS on disk and current.
    """
    seen = []

    def _fake_generate(out_dir, **kw):
        seen.append(kw)
        p = os.path.join(out_dir, pdg.pretrain_data_filename(
            kw["polarized"], kw["reference_xc"]))
        np.savez(p, rho_all=np.ones(1))
        pdg._write_pretrain_manifest(
            p, basis=kw["basis"], grid_level=kw["grid_level"],
            density_fit=kw["density_fit"], auxbasis=kw["auxbasis"],
            systems=kw["systems"],
            atoms=tuple((s.name, s.spin) for s in kw["systems"]),
            reference_xc=kw["reference_xc"],
            exchange_footing=kw["exchange_footing"],
            mesh_fraction=kw["mesh_fraction"],
            orientation_lock_strength=kw["orientation_lock_strength"],
            allow_irreproducible_degenerate=True)
        return p

    monkeypatch.setattr(pdg, "generate_pretrain_data_npz", _fake_generate)
    # An identity that NEEDS the waiver (grid level 1, degenerate O): the
    # waived build writes a file whose manifest records the waiver ...
    waived = dict(atoms=(("O", 2),), basis="sto-3g", grid_level=1)
    path = pdg.ensure_pretrain_data(str(tmp_path),
                                    allow_irreproducible_degenerate=True,
                                    **waived)
    meta = json.loads(open(pdg._pretrain_manifest_path(path)).read())
    assert meta["allow_irreproducible_degenerate"] is True
    # ... the file is current at that identity ...
    assert pdg.pretrain_data_is_current(
        path, basis="sto-3g", grid_level=1,
        systems=pdg.resolve_pretrain_systems(atoms=(("O", 2),))) is True
    # ... and a caller that grants no waiver is refused BEFORE that check,
    # so the waived file is unreachable rather than merely uncompared.
    with pytest.raises(ValueError, match="grid level"):
        pdg.ensure_pretrain_data(str(tmp_path), **waived)
    assert len(seen) == 1


def test_at_an_identity_needing_no_waiver_the_flag_was_never_exercised(
        monkeypatch, tmp_path):
    """The other half: at grid level 3 with the lock on, a caller granting the
    waiver and one refusing it produce the same file, and the manifest records
    the permission as unexercised, so there is nothing for the currency check
    to compare."""
    _fake_all_columns(monkeypatch)
    production = dict(atoms=(("O", 2),), basis="sto-3g", grid_level=3)
    p1 = pdg.ensure_pretrain_data(str(tmp_path / "granted"),
                                  allow_irreproducible_degenerate=True,
                                  **production)
    p2 = pdg.ensure_pretrain_data(str(tmp_path / "refused"), **production)
    for p in (p1, p2):
        meta = json.loads(open(pdg._pretrain_manifest_path(p)).read())
        assert meta["allow_irreproducible_degenerate"] is False
    assert pdg.pretrain_data_is_current(
        p1, basis="sto-3g", grid_level=3,
        systems=pdg.resolve_pretrain_systems(atoms=(("O", 2),))) is True


def test_ensure_hands_the_generator_one_statement_of_the_set(monkeypatch,
                                                             tmp_path):
    """The resolved system list is the ONLY statement of the set that reaches
    the generator.

    ``ensure_pretrain_data`` resolves ``atoms`` / ``dfs_set`` / ``pool_atoms``
    into a system tuple, checks the file against it and builds from it. It
    used to pass the raw ``atoms`` alongside that tuple; the generator ignores
    ``atoms`` whenever ``systems`` is given, so the second argument said the
    same thing again and invited the two to disagree -- a future generator
    that preferred ``atoms`` would build a set the currency check never
    examined."""
    seen = {}

    def _fake_generate(out_dir, **kw):
        seen.update(kw)
        p = os.path.join(out_dir, pdg.pretrain_data_filename(
            kw["polarized"], kw["reference_xc"]))
        np.savez(p, rho_all=np.ones(1))
        return p

    monkeypatch.setattr(pdg, "generate_pretrain_data_npz", _fake_generate)
    pdg.ensure_pretrain_data(str(tmp_path), atoms=_TINY, basis="sto-3g",
                             grid_level=0)
    assert "atoms" not in seen, sorted(seen)
    assert seen["systems"] == pdg.resolve_pretrain_systems(atoms=_TINY)
