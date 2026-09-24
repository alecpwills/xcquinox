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
#: The per-element relative tolerance the recorded fixture is held to, in ulp.
#: The columns jax and libxc compute carry the rounding residue of the
#: machine's own BLAS and instruction order, so the recording is reproduced to
#: a few ulp of each element rather than bit for bit. Measured: one element of
#: ``Fc_scan_mesh`` moved by one ulp on a hosted runner. Four covers a
#: rounding-order difference of a few operations on any element, whatever its
#: magnitude, and sits many orders below the relative change a protocol edit
#: makes to a column.
_ULP_TOLERANCE = 4

#: The stems the generator stores as an enhancement factor minus one, as its
#: module docstring states them (``Fx = F_x^PBE - 1``). The subtraction keeps
#: the difference to the last bits of the FACTOR, so such a column's absolute
#: resolution is the factor's and not its own: a hosted runner moved five
#: elements of ``Fx_scan_mesh`` of magnitude 9.05e-03 to 8.10e-02 by a half,
#: one and two ulp of one, which the per-element relative band reads as up to
#: 4.91e-14. These columns therefore carry an absolute band beside it, at
#: ``_FACTOR_ULP`` ulp of the factor's own largest magnitude, and every other
#: column keeps the relative band alone -- the recorded columns run from
#: ``zeta_mesh``, identically zero, and ``dm_all`` at 4.93e-32 up to
#: ``sigma_mesh`` at 2.10e+09, so a band scaled to one would admit any change
#: at all to the smallest of them. Scaling by the factor rather than by one
#: is what the columns differ in: on the recorded file the band works out at
#: 2.765 ulp of one on ``Fc_scan_all``, whose factors never exceed 0.346, and
#: at 14.432 on ``Fx_all``, whose reach 1.804.
_FACTOR_STEMS = ("Fx", "Fc", "Fx_scan", "Fc_scan")
_FACTOR_ULP = 8

#: The stem of the iso-orbital indicator, ``p_delta(tau - tau_W)/tau_unif``
#: (``metagga.compute_alpha``), which the mesh realizes as
#: ``tau = alpha tau_unif + tau_W`` and the SCF grid takes from the density
#: matrix. The subtraction cancels the von Weizsacker term, so an element's
#: absolute resolution is that of the operand ``tau/tau_unif = |alpha| +
#: tau_W/tau_unif`` and not of the indicator: a hosted runner moved two
#: elements of ``metagga_mesh``, 0.25 and 1.5, by 0.80 and 0.61 ulp of that
#: operand, which the per-element relative band read as 22.0 and 6.7 ulp of
#: the value, the ratio being 6.67 and 15.0 there. Each indicator element
#: therefore carries ``_ALPHA_ULP`` ulp of its own operand, formed from the
#: density and gradient recorded beside it. The constant is measured rather
#: than derived: the mesh rebuilt from its literal grids reproduces the
#: recording bit for bit, a second float64 spelling of the same
#: kinetic-energy density put through the same helper moves 12 of the 560
#: elements by more than one ulp of the operand and none by more than 1.48,
#: and the density column's own four-ulp band, carried through the whole
#: mesh chain rebuilt from the moved density, moves the indicator by at most
#: 1.48 ulp of the operand (6.67, four times five thirds, if the
#: kinetic-energy density were held fixed instead). Eight clears the sum of
#: the two, 2.95, by 2.7x, and the tests hold it to a window: at least five
#: thirds of ``_ULP_TOLERANCE`` below, and under a hundredfold of the runner's
#: movement above. On the SCF grid the ratio reaches 6.8e+05 in the density
#: tails, where the indicator sits at its smoothing floor and the recorded
#: value is the rounding residue of the subtraction; the band there is at
#: most 2.4e-04 of the element.
_ALPHA_STEM = "metagga"
_ALPHA_ULP = 8


def _alpha_band(key, ref):
    """The per-element band of an indicator column, ``_ALPHA_ULP`` ulp of
    ``|alpha| + tau_W/tau_unif`` in the column's shape, the ratio formed from
    the density and gradient columns of the key's own block
    (``rho_<suffix>``, ``sigma_<suffix>``), which every recorded block
    carries."""
    suffix = key.rpartition("_")[2]
    for sibling in (f"rho_{suffix}", f"sigma_{suffix}"):
        assert sibling in ref, (
            f"{key}: the indicator's band is formed from {sibling}, which the "
            "mapping does not carry")
    alpha = np.asarray(ref[key], dtype=float)
    rho = np.asarray(ref[f"rho_{suffix}"], dtype=float)
    sigma = np.asarray(ref[f"sigma_{suffix}"], dtype=float)
    tau_w = sigma / (8.0 * rho)
    tau_unif = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
    ratio = (tau_w / tau_unif).reshape(alpha.shape)
    eps = float(np.finfo(np.float64).eps)
    return _ALPHA_ULP * eps * (np.abs(alpha) + ratio)


def _column_band(key, ref):
    """The absolute band of one recorded column: an array for an indicator
    column, from :func:`_alpha_band`; otherwise :func:`_factor_band` on the
    column alone, which is zero for every column that cancels nothing."""
    stem, _, suffix = key.rpartition("_")
    if stem == _ALPHA_STEM and suffix:
        return _alpha_band(key, ref)
    return _factor_band(key, ref[key])


def _assert_column_close(key, got, ref, rtol, band):
    """``|got - ref| <= rtol |ref| + band`` for a per-element band, which
    numpy's ``assert_allclose`` takes only as a scalar. Written in the
    positive form, every element inside its allowance, so a non-finite
    element is refused as ``assert_allclose`` refuses it."""
    got = np.asarray(got, dtype=float)
    ref = np.asarray(ref, dtype=float)
    allowance = rtol * np.abs(ref) + np.asarray(band, dtype=float)
    difference = np.abs(got - ref)
    inside = difference <= allowance
    if not bool(np.all(inside)):
        outside = np.argwhere(~inside)
        # A non-finite difference is the worst excess there is.
        excess = np.where(inside, -np.inf,
                          np.nan_to_num(difference - allowance, nan=np.inf))
        worst = tuple(int(i) for i in np.unravel_index(
            int(np.argmax(excess)), ref.shape))
        raise AssertionError(
            f"{key}: {len(outside)} of {ref.size} elements outside the band; "
            f"worst at {worst}: {got[worst]!r} against {ref[worst]!r}, "
            f"difference {difference[worst]:.6e} against an allowance of "
            f"{allowance[worst]:.6e}")


def _factor_band(key, ref):
    """The absolute band one recorded column is held to beside the relative
    one: zero unless the generator stores the column as ``F - 1``, where it is
    ``_FACTOR_ULP`` ulp of ``max|1 + ref|``, the largest factor the column's
    subtraction was taken from."""
    if key.rsplit("_", 1)[0] not in _FACTOR_STEMS:
        return 0.0
    ref = np.asarray(ref)
    if not ref.size or not np.issubdtype(ref.dtype, np.floating):
        return 0.0
    eps = float(np.finfo(ref.dtype).eps)
    return _FACTOR_ULP * eps * float(np.max(np.abs(1.0 + ref)))


def _assert_columns_match(got, ref):
    """Hold every recorded key: dtype and shape exactly, floating values
    within ``_ULP_TOLERANCE`` ulp of each element (a relative tolerance)
    beside :func:`_column_band`, which is zero for every column but those
    stored as an enhancement factor minus one and the iso-orbital indicator
    columns, and integer values exactly. Returns the keys that moved with
    their largest absolute difference, for the report."""
    moved = []
    for key in sorted(ref):
        assert got[key].dtype == ref[key].dtype, key
        assert got[key].shape == ref[key].shape, key
        if not np.issubdtype(ref[key].dtype, np.floating):
            np.testing.assert_array_equal(got[key], ref[key], err_msg=key)
            continue
        rtol = _ULP_TOLERANCE * float(np.finfo(ref[key].dtype).eps)
        band = _column_band(key, ref)
        if isinstance(band, np.ndarray):
            _assert_column_close(key, got[key], ref[key], rtol, band)
        else:
            np.testing.assert_allclose(got[key], ref[key], rtol=rtol,
                                       atol=band, err_msg=key)
        if not np.array_equal(got[key], ref[key]):
            moved.append((key, float(np.max(np.abs(got[key] - ref[key])))))
    return moved

#: The new keys the default configuration gains. Everything else the default
#: file carries is held to the tolerance above against the recorded fixture.
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
    """Every column the generator writes at the default configuration is held
    against the recorded fixture, within ``_ULP_TOLERANCE`` ulp of each
    element, where the columns jax and libxc compute carry the rounding
    residue of the machine's own BLAS and instruction order (a hosted runner
    moved one element of ``Fc_scan_mesh`` by one ulp). New keys may appear;
    recorded ones may not move beyond that, so a YAML already in flight trains
    on the same numbers. The .npz container is a zip whose headers carry write
    timestamps, so the pin is on array contents, not on the file's bytes. Both
    atoms carry one s function, on which the traceless-quadrupole bias of the
    orientation lock vanishes identically, so the locked default reproduces
    the recording, which is at the pinned reference density cutoff
    (``pyscf_determinism.REFERENCE_SMALL_RHO_CUTOFF``)."""
    ref = dict(np.load(_FIXTURE))
    _path, got = _gen(tmp_path)
    missing = sorted(set(ref) - set(got))
    assert not missing, f"the default output lost {missing}"
    _assert_columns_match(got, ref)


def test_the_default_output_pin_is_a_few_ulp_per_element():
    """A one-ulp movement of an element passes the pin, a hundred-ulp movement
    fails it, and the same absolute movement on a small element of the same
    column fails it, since the tolerance is per element and not per column; an
    integer column is exact; and the refusal names the offending key and no
    other.

    The large element is a power of two, so one ulp of it is exactly
    ``eps`` times its value and the movement is representable."""
    big = 2.0 ** 31
    ref = {"alpha": np.array([1.0, big]), "count": np.array([1, 2])}
    one_ulp = float(np.finfo(np.float64).eps) * big
    moved_one = {"alpha": ref["alpha"] + np.array([0.0, one_ulp]),
                 "count": ref["count"].copy()}
    moved_far = {"alpha": ref["alpha"] + np.array([0.0, 100.0 * one_ulp]),
                 "count": ref["count"].copy()}
    moved_small = {"alpha": ref["alpha"] + np.array([one_ulp, 0.0]),
                   "count": ref["count"].copy()}

    def _refused(got):
        with pytest.raises(AssertionError) as exc:
            _assert_columns_match(got, ref)
        return str(exc.value)

    moved = _assert_columns_match(moved_one, ref)
    assert moved == [("alpha", one_ulp)]
    assert "alpha" in _refused(moved_far)
    assert "alpha" in _refused(moved_small)
    message = _refused({"alpha": ref["alpha"], "count": ref["count"] + 1})
    assert "count" in message and "alpha" not in message


#: The five ``Fx_scan_mesh`` elements a hosted runner computed differently from
#: the recording (GitHub Actions run 35918397461, both interpreters): index ->
#: the value the runner produced, the recorded column carrying the other side
#: of each pair. The movements are 1.1102e-16 (three of them), 2.2204e-16 and
#: 4.4409e-16 -- a half, one and two ulp of ONE, the column being stored as
#: ``F - 1`` and recovered by a subtraction -- on elements of magnitude
#: 9.0534e-03 to 8.1003e-02; the worst is a relative 4.9052e-14, 55x the
#: four-ulp relative band 8.8818e-16.
_RUNNER_FX_SCAN_MESH = {
    166: -0.04015583959777602,
    176: -0.03271906057807161,
    177: -0.08100326601677632,
    197: -0.0299856934434638,
    206: 0.009053440597357687,
}


def test_the_factor_band_is_attached_to_the_factor_columns_alone():
    """The absolute band is nonzero for the columns the generator stores as
    ``F - 1`` and for no other column it can write.

    Oracle: the recorded fixture's own float keys and, since the recording is
    on the total-density footing, the generator's whole key universe
    (``pretrain_data_gen._KNOWN_KEYS``) beside it, so the two exchange-block
    factor columns of the ``spin_channel`` footing are classified as well. The
    stems the band keys on are held to the generator's own declarations, every
    stem tuple it writes a column from: the only enhancement-factor stems among
    them are the four the band names, so a factor column added to the schema
    without a band fails here wherever its stem is declared.

    The scale is the factor's own largest magnitude, ``max|1 + ref|``, which a
    band independent of the column and a band scaled by the column's own range
    both fail to be; a column whose factors span 0.09 to 0.5 separates the
    three, and the band it receives is asserted exactly.
    """
    declared = (set(pdg._ALL_CORE) | set(pdg._X_CORE) | set(pdg._MESH_CORE)
                | set(pdg._ALL_PROTOCOL) | set(pdg._DESCRIPTOR_STEMS)
                | set(pdg._SYSTEM_TABLE) | set(pdg._SCALARS))
    assert set(_FACTOR_STEMS) == {s for s in declared if s.startswith("F")}

    eps = float(np.finfo(np.float64).eps)
    spanning = np.array([-0.91, -0.5])
    factor_scale = float(np.max(np.abs(1.0 + spanning)))
    column_scale = 1.0 + float(np.max(np.abs(spanning)))
    assert factor_scale == 0.5
    assert len({factor_scale, column_scale, 1.0}) == 3
    band = _factor_band("Fc_all", spanning)
    assert band == _FACTOR_ULP * eps * 0.5
    assert band != _FACTOR_ULP * eps * column_scale
    assert band != _FACTOR_ULP * eps
    # A column with no elements has no factor to scale by, and is reported as
    # unbanded rather than raising.
    assert _factor_band("Fx_all", np.array([])) == 0.0

    ref = dict(np.load(_FIXTURE))
    banded = {k for k, v in ref.items()
              if np.issubdtype(v.dtype, np.floating)
              and _factor_band(k, v) != 0.0}
    assert banded == {"Fc_all", "Fc_scan_all", "Fc_scan_mesh", "Fx_all",
                      "Fx_scan_all", "Fx_scan_mesh"}

    probe = np.array([0.25, -0.5, 0.75])
    universe = {k for k in pdg._KNOWN_KEYS if _factor_band(k, probe) != 0.0}
    assert universe == banded | {"Fx_x", "Fx_scan_x"}


def test_the_factor_band_admits_the_runner_movement_and_no_more():
    """The movement a hosted runner recorded on ``Fx_scan_mesh`` passes and a
    hundred times it does not.

    Oracle: the recorded column with the runner's five elements substituted,
    which is the failure itself -- refused by the relative tolerance alone,
    admitted by it beside the band -- and the same column displaced by a
    hundred times the worst of those movements, which is refused. The
    comparison helper is exercised on the recorded column directly, so no SCF
    is run.
    """
    eps = float(np.finfo(np.float64).eps)
    recorded = np.array(dict(np.load(_FIXTURE))["Fx_scan_mesh"])
    ref = {"Fx_scan_mesh": recorded}
    got = {"Fx_scan_mesh": recorded.copy()}
    for index, value in _RUNNER_FX_SCAN_MESH.items():
        assert got["Fx_scan_mesh"][index] != value, index
        got["Fx_scan_mesh"][index] = value
    assert int(np.sum(got["Fx_scan_mesh"] != recorded)) == 5
    worst = float(np.max(np.abs(got["Fx_scan_mesh"] - recorded)))
    assert worst == 2.0 * eps
    assert worst / abs(float(recorded[206])) > _ULP_TOLERANCE * eps

    with pytest.raises(AssertionError, match="Fx_scan_mesh"):
        np.testing.assert_allclose(got["Fx_scan_mesh"], recorded,
                                   rtol=_ULP_TOLERANCE * eps, atol=0.0,
                                   err_msg="Fx_scan_mesh")

    assert _assert_columns_match(got, ref) == [("Fx_scan_mesh", worst)]

    far = {"Fx_scan_mesh": recorded + 100.0 * worst}
    with pytest.raises(AssertionError, match="Fx_scan_mesh"):
        _assert_columns_match(far, ref)

    # The band's own floor and ceiling on this column, neither of which the two
    # comparisons above reach: it carries a margin of at least four over the
    # movement recorded, so a band merely equal to that movement is refused,
    # and stays under the displacement above, which the relative tolerance
    # would otherwise be left to refuse alone.
    band = _factor_band("Fx_scan_mesh", recorded)
    assert band >= 4.0 * worst, band / worst
    assert band < 100.0 * worst, band / worst


def test_no_column_the_band_would_swallow_carries_one():
    """No column whose whole range sits below a band scaled to one carries
    such a band.

    Oracle: the recorded fixture's float columns. ``zeta_mesh`` is identically
    zero and ``dm_all`` reaches 4.9304e-32, against ``sigma_mesh`` at
    2.0992e+09, so a blanket absolute floor of a few ulp of one would admit any
    change at all to the first two. Both are refused: every element of
    ``dm_all`` doubled, and ``zeta_mesh`` displaced by eight ulp of one.
    """
    eps = float(np.finfo(np.float64).eps)
    ref = dict(np.load(_FIXTURE))
    scale = {k: float(np.max(np.abs(v))) for k, v in ref.items()
             if np.issubdtype(v.dtype, np.floating)}
    assert min(scale, key=scale.get) == "zeta_mesh"
    assert scale["zeta_mesh"] == 0.0
    nonzero = {k: s for k, s in scale.items() if s > 0.0}
    assert min(nonzero, key=nonzero.get) == "dm_all"
    assert max(scale, key=scale.get) == "sigma_mesh"
    # Doubling the smallest nonzero column moves it by under one ulp of one.
    assert 2.0 * nonzero["dm_all"] < eps

    with pytest.raises(AssertionError, match="dm_all"):
        _assert_columns_match({"dm_all": 2.0 * ref["dm_all"]},
                              {"dm_all": ref["dm_all"]})
    with pytest.raises(AssertionError, match="zeta_mesh"):
        _assert_columns_match({"zeta_mesh": ref["zeta_mesh"] + 8.0 * eps},
                              {"zeta_mesh": ref["zeta_mesh"]})


#: The two ``metagga_mesh`` elements a hosted runner computed differently from
#: the recording: ``(row, column)`` -> the value the runner produced, the
#: recorded column carrying the other side of each pair. The movements are
#: 1.2212e-15 and 2.2204e-15 -- 5.5 and 10 ulp of ONE -- on elements of
#: magnitude 2.5000e-01 and 1.5000e+00, which the per-element relative band
#: reads as 4.8850e-15 and 1.4803e-15, 5.5x and 1.7x the four-ulp band
#: 8.8818e-16. The column is the iso-orbital indicator
#: ``p_delta(tau - tau_W)/tau_unif`` and the mesh realizes each node as
#: ``tau = alpha tau_unif + tau_W``, so the subtraction cancels the von
#: Weizsacker term and an element's absolute resolution is that of the operand
#: ``tau/tau_unif = |alpha| + tau_W/tau_unif``, 6.9167 and 16.5000 at these two
#: elements; the movements are 0.80 and 0.61 ulp of THAT.
_RUNNER_METAGGA_MESH = {
    (212, 0): 0.25000000009999934,
    (226, 0): 1.5000000000166653,
}


def test_the_alpha_band_admits_the_runner_movement_and_no_more():
    """The movement a hosted runner recorded on ``metagga_mesh`` passes and a
    hundred times it does not.

    Oracle: the recorded column with the runner's two elements substituted,
    which is the failure itself -- refused by the relative tolerance alone,
    admitted by it beside the band -- carried beside the recorded density and
    gradient columns of its own block, out of which the band is formed. The
    same column displaced by a hundred times the worst of those movements, and
    by a relative 1e-6, are both refused. The comparison helper is exercised on
    the recorded columns directly, so no SCF is run.
    """
    eps = float(np.finfo(np.float64).eps)
    recorded = dict(np.load(_FIXTURE))
    ref = {k: np.array(recorded[k])
           for k in ("metagga_mesh", "rho_mesh", "sigma_mesh")}
    column = ref["metagga_mesh"]
    got = {k: v.copy() for k, v in ref.items()}
    for index, value in _RUNNER_METAGGA_MESH.items():
        assert got["metagga_mesh"][index] != value, index
        got["metagga_mesh"][index] = value
    assert int(np.sum(got["metagga_mesh"] != column)) == 2
    worst = float(np.max(np.abs(got["metagga_mesh"] - column)))
    assert worst == 10.0 * eps
    assert worst / abs(float(column[226, 0])) > _ULP_TOLERANCE * eps

    with pytest.raises(AssertionError, match="metagga_mesh"):
        np.testing.assert_allclose(got["metagga_mesh"], column,
                                   rtol=_ULP_TOLERANCE * eps, atol=0.0,
                                   err_msg="metagga_mesh")

    assert _assert_columns_match(got, ref) == [("metagga_mesh", worst)]

    far = dict(ref, metagga_mesh=column + 100.0 * worst)
    with pytest.raises(AssertionError, match="metagga_mesh"):
        _assert_columns_match(far, ref)
    scaled = dict(ref, metagga_mesh=column * (1.0 + 1e-6))
    with pytest.raises(AssertionError, match="metagga_mesh"):
        _assert_columns_match(scaled, ref)

    # A per-element band cannot be handed to ``assert_allclose`` on this
    # stack, so an array band needs an explicit comparison; that comparison
    # has to be the positive one -- every element within its allowance -- and
    # not its negation, since ``NaN > allowance`` is False and a column of NaN
    # would pass the negation, which ``assert_allclose`` refuses.
    blank = dict(ref, metagga_mesh=np.full_like(column, np.nan))
    with pytest.raises(AssertionError, match="metagga_mesh"):
        _assert_columns_match(blank, ref)
    # An infinite column is refused as well, and only because the band and
    # the relative term are read off the recorded side: read off the generated
    # side both would be infinite, and ``inf <= inf`` would admit the column.
    for sign in (np.inf, -np.inf):
        endless = dict(ref, metagga_mesh=np.full_like(column, sign))
        with pytest.raises(AssertionError, match="metagga_mesh"):
            _assert_columns_match(endless, ref)
    # The band is formed from the density and gradient of the indicator's own
    # block; a mapping without them is refused naming the missing column
    # rather than compared without a band.
    with pytest.raises(AssertionError, match="rho_mesh"):
        _assert_columns_match({"metagga_mesh": column},
                              {"metagga_mesh": column})

    # The band's own floor and ceiling on this column, neither of which the
    # comparisons above reach: at each of the two elements it carries a margin
    # of at least eight over the movement recorded there (10.06 and 13.20), so
    # a band merely equal to those movements is refused, and its largest value
    # over the whole column, 8.2897e-14, stays under the displacement above,
    # which the relative tolerance would otherwise be left to refuse alone.
    band = _column_band("metagga_mesh", ref)
    for index in _RUNNER_METAGGA_MESH:
        movement = abs(_RUNNER_METAGGA_MESH[index] - float(column[index]))
        assert band[index] >= 8.0 * movement, (index, band[index] / movement)
    assert float(np.max(band)) < 100.0 * worst


def test_the_alpha_band_is_the_stated_ulp_of_the_operand_per_element():
    """The indicator columns carry a per-element band of ``_ALPHA_ULP`` ulp of
    the operand their subtraction is taken from, and every other column keeps
    the band it had. The constant is stated once, on the module, and held
    here to its window: the floor by consistency with the density column's
    own band, the ceiling by the hundredfold refusal of the test above.

    Oracle: the recorded fixture's own density and gradient columns, from which
    ``tau_W/tau_unif = [sigma/(8 rho)] / [0.3 (3 pi^2)^(2/3) rho^(5/3)]`` is
    formed here rather than imported, so a band that reads the wrong block's
    siblings or the wrong exponent is refused. On the mesh that ratio is
    6.666667 at row 212 and 15.000000 at row 226 -- the two elements a hosted
    runner moved -- and reaches 4.166667e+01; on the SCF grid it reaches
    6.801432e+05 in the density tails, where the indicator sits at the
    smoothing floor and the recorded value is the rounding residue of
    ``tau - tau_W``. The band that follows is at most 1.480298e-08 of its
    element on the mesh and 2.416388e-04 on the SCF grid, both below a
    thousandth, so neither column is banded to the point where its recorded
    values stop constraining the protocol.

    The classification is asserted over every float key the recording carries:
    an array for the two indicator columns, the factor band unchanged for the
    six columns stored as ``F - 1``, and zero for the rest. The band is
    compared with its expression at a relative 1e-12 rather than bit for bit,
    since a re-association of the same expression moves the last bit and
    moves nothing the band decides.
    """
    eps = float(np.finfo(np.float64).eps)
    ref = dict(np.load(_FIXTURE))
    for key, ratio_max, relative_max in (("metagga_mesh", 4.166667e+01,
                                          1.480298e-08),
                                         ("metagga_all", 6.801432e+05,
                                          2.416388e-04)):
        band = _column_band(key, ref)
        assert isinstance(band, np.ndarray), key
        assert band.shape == ref[key].shape, key
        suffix = key.rsplit("_", 1)[1]
        rho = np.asarray(ref[f"rho_{suffix}"], dtype=float)
        sigma = np.asarray(ref[f"sigma_{suffix}"], dtype=float)
        alpha = np.asarray(ref[key], dtype=float)
        tau_w = sigma / (8.0 * rho)
        tau_unif = 0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
        ratio = (tau_w / tau_unif).reshape(alpha.shape)
        np.testing.assert_allclose(band,
                                   _ALPHA_ULP * eps * (np.abs(alpha) + ratio),
                                   rtol=1e-12, atol=0.0, err_msg=key)
        assert float(np.max(ratio)) == pytest.approx(ratio_max, rel=1e-4), key
        relative = float(np.max(band / np.abs(alpha)))
        assert relative == pytest.approx(relative_max, rel=1e-4), key
        assert relative < 1e-3, key
    # The floor on the constant: the density column beside the indicator is
    # admitted ``_ULP_TOLERANCE`` ulp, and with the kinetic-energy density held
    # fixed that carries five thirds as many ulp of the operand into the
    # indicator through ``tau_unif ~ rho^(5/3)``, which the band must clear;
    # the ceiling is the hundredfold displacement the test above refuses.
    assert _ALPHA_ULP >= _ULP_TOLERANCE * 5.0 / 3.0

    mesh_ratio = ((np.asarray(ref["sigma_mesh"], dtype=float)
                   / (8.0 * np.asarray(ref["rho_mesh"], dtype=float)))
                  / (0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
                     * np.asarray(ref["rho_mesh"], dtype=float) ** (5.0 / 3.0)))
    assert float(mesh_ratio[212]) == pytest.approx(20.0 / 3.0, rel=1e-4)
    assert float(mesh_ratio[226]) == pytest.approx(15.0, rel=1e-4)

    floats = {k for k, v in ref.items()
              if np.issubdtype(v.dtype, np.floating)}
    arrays = {k for k in floats
              if isinstance(_column_band(k, ref), np.ndarray)}
    assert arrays == {"metagga_all", "metagga_mesh"}
    scalars = {k: _column_band(k, ref) for k in floats - arrays}
    factors = {"Fc_all", "Fc_scan_all", "Fc_scan_mesh", "Fx_all",
               "Fx_scan_all", "Fx_scan_mesh"}
    assert {k for k, b in scalars.items() if b != 0.0} == factors
    assert all(scalars[k] > 0.0 for k in factors)
    assert all(scalars[k] == _factor_band(k, ref[k]) for k in scalars)


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


