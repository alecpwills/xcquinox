"""Acceptance tests for ``report_equation_figures.py``.

The script is executed once into a temporary directory; every assertion below
reads the CSV the corresponding figure wrote, so a figure whose curve moved
fails here rather than silently shipping a wrong graph. Each figure carries at
least one load-bearing pin -- a value the reports quote, or an identity the
construction guarantees.

Three of the script's internal oracles are additionally shown to FIRE under the
weakest mutation that restores the defect they guard against: the pre-image
clamp check (the map's clamp moved away from the bind-threshold formula), the
spin-interpolation check (a perturbed normalization of ``f(zeta)``) and the C2
basin-occupancy check (a log whose own summary disagrees with its trajectory).
"""
from __future__ import annotations

import csv
import importlib.util
import math
import struct
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE / "report_equation_figures.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "report_equation_figures", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ref = _load_module()

STEMS = tuple(stem for stem, _ in ref.FIGURES)


# --------------------------------------------------------------------------- #
# One run of the script for the whole module
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def outdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("figures_report_pretraining")
    ref.main(outdir=d, dpi=200)
    return d


def _rows(outdir, stem):
    with open(Path(outdir) / f"{stem}.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def _select(rows, **conditions):
    out = rows
    for key, value in conditions.items():
        out = [r for r in out if r[key] == value]
    return out


def _one(rows, x, tol=0.0, **conditions):
    """The single row of ``conditions`` whose ``x`` matches (exactly, or within
    ``tol``)."""
    sel = _select(rows, **conditions)
    assert sel, f"no rows for {conditions}"
    if tol == 0.0:
        hits = [r for r in sel if float(r["x"]) == x]
    else:
        hits = [r for r in sel if abs(float(r["x"]) - x) <= tol]
    assert len(hits) == 1, (
        f"expected exactly one row at x={x} for {conditions}, got {len(hits)}")
    return hits[0]


def _png_dpi(path):
    """Resolution recorded in the PNG ``pHYs`` chunk, in dots per inch."""
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    i = 8
    while i + 8 <= len(data):
        length = struct.unpack(">I", data[i:i + 4])[0]
        kind = data[i + 4:i + 8]
        if kind == b"pHYs":
            x_ppm, _y_ppm, unit = struct.unpack(">IIB", data[i + 8:i + 17])
            assert unit == 1, "pHYs unit is not metres"
            return x_ppm * 0.0254
        i += 12 + length
    raise AssertionError(f"{path} carries no pHYs resolution chunk")


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

def test_every_figure_and_csv_is_written(outdir):
    assert len(STEMS) == 9
    for stem in STEMS:
        png = Path(outdir) / f"{stem}.png"
        csv_path = Path(outdir) / f"{stem}.csv"
        assert png.is_file(), f"{stem}.png missing"
        assert csv_path.is_file(), f"{stem}.csv missing"
        assert png.stat().st_size > 20_000, f"{stem}.png looks empty"
        assert len(_rows(outdir, stem)) > 10, f"{stem}.csv looks empty"


def test_every_png_is_at_least_150_dpi(outdir):
    for stem in STEMS:
        dpi = _png_dpi(Path(outdir) / f"{stem}.png")
        assert dpi >= 150.0, f"{stem}.png at {dpi:.1f} dpi"


def test_every_csv_row_is_finite_and_self_describing(outdir):
    for stem in STEMS:
        rows = _rows(outdir, stem)
        assert set(rows[0]).issuperset(
            {"panel", "series", "x_name", "x", "y_name", "y"})
        for r in rows:
            assert r["x_name"] and r["y_name"], f"{stem}: unnamed axis"
            for key in ("x", "y"):
                assert math.isfinite(float(r[key])), f"{stem}: non-finite {key}"


# --------------------------------------------------------------------------- #
# Figure 1 -- the bounded map
# --------------------------------------------------------------------------- #

def test_bounded_map_returns_the_parent_at_zero_and_binds_where_documented(
        outdir):
    rows = _rows(outdir, "bounded_map")
    eps = 2.220446049250313e-16
    for limit in (1.174, 1.804, 2.0):
        # F(0) = 1 for every limit: the anchored network's initialization.
        # The map evaluates Lambda / (1 + exp(ln(Lambda - 1))) and the
        # exponential of the logarithm is not exact, so the identity holds to
        # one unit in the last place, not bitwise: measured, exactly 1.0 at
        # 1.804 and 2.0 and half an ulp low (-1.11e-16) at 1.174.
        row = _one(rows, 0.0, panel="a_map", series=f"F_limit_{limit}")
        assert abs(float(row["y"]) - 1.0) <= eps

    # The clamp's bind thresholds, as parents.lob_preimage documents them.
    expected = {1.174: (8.68e-19, 2.87e-17),
                1.804: (6.16e-18, 9.53e-18),
                2.0: (8.50e-18, 8.50e-18)}
    for limit, (upper, lower) in expected.items():
        sel = _select(rows, panel="c_bind", series=f"bind_limit_{limit}")
        assert len(sel) == 2
        got_upper, got_lower = (float(sel[0]["y"]), float(sel[1]["y"]))
        assert got_upper == pytest.approx(upper, rel=2e-3)
        assert got_lower == pytest.approx(lower, rel=2e-3)


def test_bind_threshold_oracle_fires_when_the_clamp_moves(tmp_path,
                                                          monkeypatch):
    """Weakest mutation: the map's clamp is at 39 while the bind-threshold
    formula still uses 40. The oracle must refuse."""
    original = ref.parents.lob_preimage
    monkeypatch.setattr(
        ref.parents, "lob_preimage",
        lambda f, limit, z_max=39.0: original(f, limit, z_max=z_max))
    with pytest.raises(AssertionError, match="bind thresholds"):
        ref.make_bounded_map(tmp_path, 150)


# --------------------------------------------------------------------------- #
# Figure 2 -- pre-image sensitivity
# --------------------------------------------------------------------------- #

def test_preimage_sensitivity_pins_the_recorded_pbe_suppression(outdir):
    rows = _rows(outdir, "preimage_sensitivity")
    at_zero = _one(rows, 0.0, panel="a_exchange", series="pbe_fx")
    # REPORT_pretraining_evolution.md Section 6.2: L' = 0.446 at s = 0.
    assert float(at_zero["y"]) == pytest.approx(0.4457, abs=1e-3)
    assert float(at_zero["F_parent"]) == 1.0
    assert float(at_zero["z_parent"]) == 0.0

    at_twenty = _one(rows, 20.0, panel="a_exchange", series="pbe_fx")
    # ... falling to 0.0073 by s = 20.
    assert float(at_twenty["y"]) == pytest.approx(0.0073, abs=1e-4)

    # The SCAN alpha = 0 parent sits AT the ceiling at s = 0, so the pre-image
    # clamps and the sensitivity is exactly zero.
    scan0 = _one(rows, 0.0, panel="a_exchange", series="scan_fx_alpha0")
    assert float(scan0["y"]) == 0.0
    assert float(scan0["z_parent"]) == 40.0

    # The correlation mirror: 0.500 at s = 0, 0.0015 at s = 6.
    c0 = _one(rows, 0.0, panel="b_correlation", series="pbe_fc_slope")
    c6 = _one(rows, 6.0, panel="b_correlation", series="pbe_fc_slope",
              tol=1e-9)
    assert float(c0["y"]) == pytest.approx(0.500, abs=1e-3)
    assert float(c6["y"]) == pytest.approx(0.0015, abs=1e-4)


# --------------------------------------------------------------------------- #
# Figure 3 -- the smooth positive part
# --------------------------------------------------------------------------- #

def test_smooth_positive_part_floor_and_exact_inversion(outdir):
    rows = _rows(outdir, "smooth_positive_part")
    at_zero = _one(rows, 0.0, panel="a_value", series="smooth_positive_part")
    # p(0) = w / 2 exactly, w = 1e-5 (metagga._ALPHA_SMOOTHING_WIDTH).
    assert float(at_zero["y"]) == 5e-06
    hard = _one(rows, 0.0, panel="a_value", series="hard_positive_part")
    assert float(hard["y"]) == 0.0

    errors = [float(r["y"])
              for r in _select(rows, panel="b_roundtrip",
                               series="inversion_error")]
    assert errors, "no round-trip series"
    # The inversion is exact to round-off over the whole window.
    assert max(errors) < 1e-18


# --------------------------------------------------------------------------- #
# Figure 4 -- the indicator ceiling
# --------------------------------------------------------------------------- #

def test_alpha_ceiling_saturation_matches_the_recorded_residual(outdir):
    rows = _rows(outdir, "alpha_ceiling")
    at_cap = _one(rows, 100.0, panel="a_residual", series="s0")
    assert float(at_cap["y"]) == 0.0, "the residual must vanish at the cap"

    s0 = sorted(_select(rows, panel="a_residual", series="s0"),
                key=lambda r: float(r["x"]))
    saturation = float(s0[-1]["y"])
    # HISTORY 2026-08-31 (erratum): the ceiling residual saturates at
    # 1.74e-3 at s = 0.
    assert saturation == pytest.approx(1.74e-3, rel=0.02)
    # It is a monotone approach, so the last point is the largest.
    assert saturation == max(float(r["y"]) for r in s0)


# --------------------------------------------------------------------------- #
# Figure 5 -- the parent enhancement factors
# --------------------------------------------------------------------------- #

def test_parent_enhancement_uniform_gas_limits(outdir):
    rows = _rows(outdir, "parent_enhancement")
    # PBE: F_x(s = 0) = 1 exactly.
    assert float(_one(rows, 0.0, panel="a_exchange",
                      series="pbe_fx")["y"]) == 1.0
    # SCAN at alpha = 0: F_x(0) = h_x^0 = 1.174 exactly (the ceiling).
    assert float(_one(rows, 0.0, panel="a_exchange",
                      series="scan_fx_alpha0")["y"]) == 1.174
    # SCAN at alpha = 1 recovers the uniform gas.
    assert float(_one(rows, 0.0, panel="a_exchange",
                      series="scan_fx_alpha1")["y"]) == 1.0
    # PBE correlation at s = 0 is unity up to the two PW92 parameter sets
    # (parents.pbe_fc docstring: 1 + O(4e-6)).
    fc0 = float(_one(rows, 0.0, panel="b_correlation",
                     series="pbe_fc_rs2.0")["y"])
    assert fc0 == pytest.approx(1.0, abs=1e-5)
    assert fc0 != 1.0


# --------------------------------------------------------------------------- #
# Figure 6 -- the PW92 spin interpolation
# --------------------------------------------------------------------------- #

def test_zeta_pole_curvature_agrees_between_analytic_and_difference(outdir):
    rows = _rows(outdir, "zeta_pole")
    analytic = float(_one(rows, 0.5, panel="b_curvature",
                          series="f_second_analytic")["y"])
    difference = float(_one(rows, 0.5, panel="b_curvature",
                            series="f_second_finite_difference")["y"])
    assert abs(difference - analytic) / analytic < 1e-4
    # f''(0) is the constant libxc's LDA_C_PW_MOD carries.
    at_zero = float(_one(rows, 0.0, panel="b_curvature",
                         series="f_second_analytic")["y"])
    assert at_zero == ref.parents._PW_MOD_FZ20
    # The interpolation itself: f(0) = 0, f(+-1) = 1 at the production clip.
    assert float(_one(rows, 0.0, panel="a_interpolation",
                      series="f_zeta")["y"]) == 0.0
    edge = float(_one(rows, 1.0 - ref._ZETA_BOUNDARY_EPS,
                      panel="a_interpolation", series="f_zeta")["y"])
    assert edge == pytest.approx(1.0, abs=1e-5)


def test_spin_interpolation_oracle_fires_on_a_perturbed_normalization(
        tmp_path, monkeypatch):
    """Weakest mutation: ``f(zeta)`` normalized 1e-6 relative off. The forward
    check against ``parents._pw92_mod_eps`` must refuse."""
    original = ref.f_spin
    monkeypatch.setattr(ref, "f_spin",
                        lambda z: original(z) * (1.0 + 1e-6))
    with pytest.raises(AssertionError, match="_pw92_mod_eps"):
        ref.make_zeta_pole(tmp_path, 150)


# --------------------------------------------------------------------------- #
# Figure 7 -- the pretraining mesh
# --------------------------------------------------------------------------- #

def test_dfs_mesh_node_count_and_alpha_coverage(outdir):
    rows = _rows(outdir, "dfs_mesh")
    nodes = _select(rows, panel="a_projection", series="mesh_node")
    assert len(nodes) == 560
    assert len(nodes) == (len(ref.MESH_RS) * len(ref.MESH_S)
                          * len(ref.MESH_ALPHA))
    # Every (r_s, s, alpha) triple occurs exactly once.
    triples = {(r["r_s"], r["x"], r["y"]) for r in nodes}
    assert len(triples) == 560
    # The alpha axis stops well below the stored column's ceiling.
    alpha_nodes = [float(r["x"])
                   for r in _select(rows, panel="b_alpha_axis",
                                    series="mesh_alpha_node")]
    ceiling = float(_one(rows, ref.metagga._ALPHA_MAX, panel="b_alpha_axis",
                         series="alpha_max")["x"])
    assert max(alpha_nodes) == 5.0
    assert ceiling == 100.0
    assert ref.MESH_WEIGHT_FRACTION == 0.3


# --------------------------------------------------------------------------- #
# Figure 8 -- the C2 DIIS trajectory
# --------------------------------------------------------------------------- #

def test_c2_trajectory_row_count_and_extrema(outdir):
    rows = _rows(outdir, "c2_diis_trajectory")
    traj = _select(rows, panel="a_trajectory", series="energy")
    assert len(traj) == 100
    assert [int(float(r["x"])) for r in traj] == list(range(100))

    energies = [float(r["y"]) for r in traj]
    grads = [float(r["grad_norm"]) for r in traj]
    assert energies.index(min(energies)) == 12
    assert grads.index(min(grads)) == 25

    marker = _select(rows, panel="c_markers", series="lowest_energy_cycle")
    assert len(marker) == 1
    assert int(float(marker[0]["x"])) == 12
    assert float(marker[0]["y"]) == pytest.approx(-75.8167361296, abs=1e-9)

    solutions = sorted(float(r["x"])
                       for r in _select(rows, panel="b_solutions",
                                        series="converged_soscf"))
    assert solutions[0] == pytest.approx(-75.8167407121, abs=1e-9)
    assert solutions[1] == pytest.approx(-75.7368945310, abs=1e-9)
    gap_kcal = (solutions[1] - solutions[0]) * ref.HARTREE_PER_KCAL
    assert gap_kcal == pytest.approx(50.10, abs=0.01)


def test_c2_basin_occupancy_oracle_fires_on_an_inconsistent_log(tmp_path):
    """Weakest mutation: the log's own basin-occupancy summary is off by one.
    The figure must refuse rather than draw a midpoint it cannot justify."""
    text = Path(ref.DEFAULT_C2_LOG).read_text()
    assert "midpoint -75.7768: 73 of 100" in text
    doctored = tmp_path / "doctored.log"
    doctored.write_text(text.replace("midpoint -75.7768: 73 of 100",
                                     "midpoint -75.7768: 72 of 100"))
    with pytest.raises(AssertionError, match="basin occupancy"):
        ref.make_c2_diis_trajectory(tmp_path, 150, log_path=doctored)


def test_c2_parser_refuses_a_log_without_a_trajectory(tmp_path):
    empty = tmp_path / "empty.log"
    empty.write_text("identity: C2\ndone\n")
    with pytest.raises(ValueError, match="no per-cycle DIIS trajectory"):
        ref.parse_c2_log(empty)


# --------------------------------------------------------------------------- #
# Figure 9 -- the iso-orbital indicator
# --------------------------------------------------------------------------- #

def test_alpha_indicator_floor_uniform_gas_and_ceiling(outdir):
    rows = _rows(outdir, "alpha_indicator")
    w = ref.metagga._ALPHA_SMOOTHING_WIDTH
    linear = _select(rows, panel="a_linear", series="compute_alpha")
    by_raw = {float(r["raw_indicator"]): float(r["y"]) for r in linear}
    # tau = tau_W: the smooth floor, w / 2.
    assert by_raw[0.0] == 5e-06
    # tau = tau_W + tau_unif: the uniform electron gas, displaced by the
    # smoothing's own w^2 / 4 = 2.5e-11 (p(1) - 1 = (sqrt(1 + w^2) - 1)/2).
    uniform = by_raw[1.0]
    assert uniform == pytest.approx(1.0 + w * w / 4.0, rel=1e-15)
    assert uniform - 1.0 == pytest.approx(2.5e-11, rel=1e-6)

    ceiling = _select(rows, panel="b_ceiling", series="compute_alpha")
    capped = [float(r["y"]) for r in ceiling
              if float(r["raw_indicator"]) >= 100.0]
    assert capped, "the scan does not reach the ceiling"
    assert all(v == 100.0 for v in capped)
    assert max(float(r["raw_indicator"]) for r in ceiling) > 100.0


# --------------------------------------------------------------------------- #
# House rules
# --------------------------------------------------------------------------- #

#: Marker for the two lines that must spell the tokens out -- the list itself
#: and the probe that proves the sweep fires. The exemption is per LINE, so a
#: marked line does not license the rest of the file; without it the sweep
#: would match itself and then pass vacuously everywhere else.
_SWEEP_EXEMPT = "sweep-exempt-line"

_FORBIDDEN = ("agent", "subagent", "adversarial", "opus", "sonnet", "claude", "anthropic", "co-authored", "generated with", "audit")  # sweep-exempt-line

_SWEEP_PROBE = "# reviewed by a subagent, then by an opus auditor"  # sweep-exempt-line


def _metadata_hits(text):
    """Forbidden tokens in ``text``, line by line, skipping exempt lines."""
    hits = []
    for number, line in enumerate(text.splitlines(), start=1):
        if _SWEEP_EXEMPT in line:
            continue
        lowered = line.lower()
        hits += [(number, token) for token in _FORBIDDEN if token in lowered]
    return hits


@pytest.mark.parametrize("path", [_SCRIPT, Path(__file__)])
def test_sources_are_ascii_and_carry_no_process_metadata(path):
    raw = path.read_bytes()
    text = raw.decode("ascii")     # raises on any non-ASCII byte
    assert not _metadata_hits(text), f"{path.name}: {_metadata_hits(text)}"


def test_metadata_sweep_fires():
    """The sweep is not vacuous: a line naming the machinery is caught, and the
    exemption is line-scoped rather than file-scoped."""
    caught = [token for _, token in _metadata_hits(_SWEEP_PROBE)]
    # The probe names four distinct entries of the list; a clean line names
    # none. (The tokens are not restated here: doing so would put them back in
    # this file and make the sweep match itself again.)
    assert len(caught) == 4 and len(set(caught)) == 4, caught
    assert not _metadata_hits(
        "# the enhancement factors were checked against libxc\n")
    # The same line, marked, is skipped ...
    assert not _metadata_hits(_SWEEP_PROBE + "  " + _SWEEP_EXEMPT)
    # ... and marking one line does not license the next.
    mixed = f"# a marked line  {_SWEEP_EXEMPT}\n" + _SWEEP_PROBE + "\n"
    assert [token for _, token in _metadata_hits(mixed)] == caught


def test_figures_carry_no_process_metadata_in_their_text(outdir):
    """Nothing drawn on a figure names the machinery that produced it: the CSV
    carries the series names that appear in the legends and annotations."""
    for stem in STEMS:
        for row in _rows(outdir, stem):
            blob = (row["panel"] + row["series"] + row["x_name"]
                    + row["y_name"]).lower()
            assert not [t for t in _FORBIDDEN if t in blob], (
                f"{stem}: {row['series']!r}")
