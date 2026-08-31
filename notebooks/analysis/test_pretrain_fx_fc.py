#!/usr/bin/env python
"""Tests for pretrain_fx_fc.py -- pretrained enhancement factors vs parent.

The load-bearing pin is the anchored identity: a freshly-built parent-anchored
network with a zero-initialized final layer IS the parent (F = F_parent +
T(g), g = 0 at init), so the module's curves must reproduce the parent
baselines to round-off -- PBE for the GGA rung, SCAN at the exact iso-orbital
slices alpha in {0, 1} for the meta-GGA rung. An unanchored build fails the
same assertion by more than 1e-2, which is what makes each pin discriminating
rather than vacuous, and the routing test holds the two rungs to their OWN
parents through one dispatch path.
"""
import csv
import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pretrain_fx_fc as P  # noqa: E402


def _fresh_pair(arch_name: str, *, parent_anchor: bool):
    """(arch, model) freshly built as the anchored runs build them.

    ``create_network_pair`` resolves the parent from the arch's own rung
    (PBE for GGA, SCAN for meta-GGA) and forces the zero-initialized final
    layer whenever ``parent_anchor`` is set, so the anchored build IS its
    parent at initialization. ``deep_3x16`` and ``deep_mgga_3x16`` both carry
    ``zero_init_final_layer=True`` in the registry, so the UNANCHORED control
    is exactly ``F = 1`` (the LDA/PW92 limit), far from either parent.
    """
    from xcquinox.alec.config import get_architecture
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.networks import create_network_pair
    arch = get_architecture(arch_name)
    arch = dataclasses.replace(
        arch, parent_anchor=parent_anchor,
        descriptor_coordinates="dfs" if parent_anchor else
        arch.descriptor_coordinates,
        use_polarized_correlation=True)
    xnet, cnet = create_network_pair(arch, seed=0)
    return arch, AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def _fresh_model(arch_name: str, *, parent_anchor: bool):
    return _fresh_pair(arch_name, parent_anchor=parent_anchor)[1]


def test_anchored_init_curves_reproduce_the_parent():
    """deep_3x16 (zero_init_final_layer=True): anchored fresh init == PBE."""
    model = _fresh_model("deep_3x16", parent_anchor=True)
    curves = P.compute_curves(model)
    dfx = np.max(np.abs(curves["fx_model"] - curves["fx_parent"]))
    assert dfx < 1e-10, dfx
    for rs in P.RS_VALUES:
        pair = curves["fc"][rs]
        assert pair["parent"] is not None
        dfc = np.max(np.abs(pair["model"] - pair["parent"]))
        assert dfc < 1e-8, (rs, dfc)


def test_unanchored_init_differs_from_the_parent():
    """The identity pin above is discriminating: without the anchor the same
    fresh build sits far from PBE, so a module that plotted the wrong model
    class (or the parent against itself) could not pass both tests."""
    model = _fresh_model("deep_3x16", parent_anchor=False)
    curves = P.compute_curves(model)
    dfx = np.max(np.abs(curves["fx_model"] - curves["fx_parent"]))
    assert dfx > 1e-2, dfx


def test_render_and_csv_seam(tmp_path):
    """Figures and the CSV land with the schema the docstring states."""
    model = _fresh_model("deep_3x16", parent_anchor=True)
    curves = P.compute_curves(model)
    out1 = P.render_arch_figure("deep_3x16", curves, tmp_path, "footer")
    out2 = P.render_delta_figure({"deep_3x16": curves}, tmp_path, "footer")
    out3 = P.write_curves_csv({"deep_3x16": curves}, tmp_path)
    for out in (out1, out2, out3):
        assert out.is_file() and out.stat().st_size > 0, out
    with open(out3) as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0].keys() == {"arch", "channel", "rs", "s", "f_model",
                              "f_parent"}
    n_expected = len(P.S_GRID) * (1 + len(P.RS_VALUES))
    assert len(rows) == n_expected
    fx_rows = [r for r in rows if r["channel"] == "fx"]
    worst = max(abs(float(r["f_model"]) - float(r["f_parent"]))
                for r in fx_rows)
    assert worst < 1e-10, worst


def test_discover_archs_requires_both_networks(tmp_path):
    (tmp_path / "pretrain" / "a").mkdir(parents=True)
    (tmp_path / "pretrain" / "b").mkdir(parents=True)
    for name in ("xnet.eqx", "cnet.eqx"):
        (tmp_path / "pretrain" / "a" / name).write_bytes(b"x")
    (tmp_path / "pretrain" / "b" / "xnet.eqx").write_bytes(b"x")
    assert P.discover_archs(tmp_path) == ["a"]
    assert P.discover_archs(tmp_path / "nowhere") == []


def test_meta_gga_load_is_routed_not_refused(monkeypatch, tmp_path):
    """A meta-GGA arch loads through the same certified builder as a GGA one
    -- the historical by-name refusal is retired -- and the arch handed back
    resolves to the SCAN parent through the real rung predicate
    (``parents.parent_for_arch`` reads the descriptor list), which is what
    ``compute_curves_for_arch`` routes on."""
    from xcquinox.alec import parents
    from xcquinox.alec.config import get_architecture
    real_arch = get_architecture("deep_mgga_3x16")
    sentinel = object()
    monkeypatch.setattr(
        "xcquinox.alec.cluster.fidelity.build_certified_model",
        lambda cfg, run_dir, name: (real_arch, sentinel))
    monkeypatch.setattr(
        "xcquinox.alec.cluster.grid_config.load_grid_config",
        lambda path: object())
    arch, model = P.load_pretrained_model(tmp_path, "deep_mgga_3x16")
    assert model is sentinel
    assert parents.parent_for_arch(arch) == "scan"


# ---------------------------------------------------------------------------
# Meta-GGA (SCAN parent) slices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch_name",
                         ["deep_mgga_3x16", "deep_cusp_mgga_3x16"])
def test_anchored_mgga_init_reproduces_the_scan_parent(arch_name):
    """An anchored fresh meta-GGA init == SCAN at BOTH alpha slices.

    The slices are F_x(s) and F_c(s; r_s) at the EXACT raw iso-orbital
    indicator alpha in {0, 1} (the F_x(s)-at-fixed-alpha convention of Sun,
    Ruzsinszky and Perdew, PRL 115, 036402 (2015), Fig. 1). The <1e-10 bound
    also pins the module's alpha-column encoding: the networks invert the
    stored smooth-positive-part column (``networks._raw_indicator``), and a
    raw 0.0 placed in the column is read back as alpha ~ 1 through the
    inverse's positivity guard, which would put the alpha=0 slice ~0.17 off
    SCAN; the tight F_c pin therefore sits at an alpha=0 site, because at
    alpha=1 the same defect is invisible (SCAN is stationary in alpha there:
    a raw column of 1.0 lands 9.5e-14 off). ``deep_cusp_mgga_3x16`` holds
    the alpha column at index 2, so a curve helper that hardcodes index 0
    fails on it while passing on ``deep_mgga_3x16`` (index 0). Measured at
    the encoded column: 2.2e-16 in F_x at both alphas and <= 2.2e-16 in F_c
    on every (r_s, alpha) pair, both architectures.
    """
    model = _fresh_model(arch_name, parent_anchor=True)
    curves = P.compute_curves_scan(model)
    for alpha in P.ALPHA_VALUES:
        pair = curves["fx_alpha"][alpha]
        dfx = np.max(np.abs(pair["model"] - pair["parent"]))
        assert dfx < 1e-10, (arch_name, alpha, dfx)
    pin = curves["fc_alpha"][0.0][2.0]
    dfc_pin = np.max(np.abs(pin["model"] - pin["parent"]))
    assert dfc_pin < 1e-10, dfc_pin
    for alpha in P.ALPHA_VALUES:
        for rs in P.RS_VALUES:
            pair = curves["fc_alpha"][alpha][rs]
            dfc = np.max(np.abs(pair["model"] - pair["parent"]))
            assert dfc < 1e-8, (arch_name, alpha, rs, dfc)


def test_alpha_column_value_caps_at_the_indicator_ceiling():
    """The stored column is ``min(p(alpha), _ALPHA_MAX)``: a raw indicator
    past the ceiling encodes AT the ceiling (``metagga.compute_alpha``'s
    clip, where SCAN's switching function has saturated and
    ``networks._raw_indicator`` returns the column unchanged). Without the
    clamp the encoding would hand the network a column value the storage
    convention can never produce."""
    from xcquinox.alec.metagga import _ALPHA_MAX
    assert P.alpha_column_value(200.0) == _ALPHA_MAX
    # Below the ceiling the encoding is the smooth positive part itself:
    # p(0) = width/2 = 5e-6.
    assert P.alpha_column_value(0.0) == pytest.approx(5e-6, rel=1e-9)


def test_unanchored_mgga_differs_from_the_scan_parent():
    """The SCAN identity pin is discriminating: the same fresh build without
    the anchor sits at F = 1 (zero-initialized final layer), 0.174 under
    SCAN's alpha=0 ceiling at s=0, so a module drawing the wrong model class
    or the parent against itself could not pass both tests."""
    model = _fresh_model("deep_mgga_3x16", parent_anchor=False)
    curves = P.compute_curves_scan(model)
    pair = curves["fx_alpha"][0.0]
    dfx = np.max(np.abs(pair["model"] - pair["parent"]))
    assert dfx > 1e-2, dfx


def test_parent_routing_resolves_per_arch_in_one_path():
    """One dispatch, two rungs: the GGA arch draws against PBE and the
    meta-GGA arch against SCAN, with no cross-parent draw."""
    g_arch, g_model = _fresh_pair("deep_3x16", parent_anchor=True)
    m_arch, m_model = _fresh_pair("deep_mgga_3x16", parent_anchor=True)
    g_parent, g_curves = P.compute_curves_for_arch(g_arch, g_model)
    m_parent, m_curves = P.compute_curves_for_arch(m_arch, m_model)
    assert (g_parent, m_parent) == ("pbe", "scan")
    assert "fx_model" in g_curves and "fx_alpha" not in g_curves
    assert "fx_alpha" in m_curves and "fx_model" not in m_curves
    # The GGA baseline is the PBE parent curve, byte for byte ...
    assert np.array_equal(g_curves["fx_parent"], P.parent_fx_curve(P.S_GRID))
    # ... the meta-GGA baseline is SCAN at each exact slice alpha ...
    for alpha in P.ALPHA_VALUES:
        assert np.array_equal(m_curves["fx_alpha"][alpha]["parent"],
                              P.parent_fx_curve_scan(P.S_GRID, alpha))
    # ... and the two baselines are far apart, so a cross-parent draw could
    # not have produced either equality above.
    assert np.max(np.abs(P.parent_fx_curve_scan(P.S_GRID, 0.0)
                         - P.parent_fx_curve(P.S_GRID))) > 1e-2


def test_render_and_csv_seam_scan(tmp_path):
    """Scan-mode figures render and the mixed-run CSV carries the alpha
    column, with GGA rows keeping an empty alpha as fx rows keep an empty
    rs."""
    m_model = _fresh_model("deep_mgga_3x16", parent_anchor=True)
    m_curves = P.compute_curves_scan(m_model)
    g_model = _fresh_model("deep_3x16", parent_anchor=True)
    g_curves = P.compute_curves(g_model)
    out1 = P.render_arch_figure("deep_mgga_3x16", m_curves, tmp_path, "footer")
    mixed = {"deep_3x16": g_curves, "deep_mgga_3x16": m_curves}
    out2 = P.render_delta_figure(mixed, tmp_path, "footer")
    out3 = P.write_curves_csv(mixed, tmp_path)
    for out in (out1, out2, out3):
        assert out.is_file() and out.stat().st_size > 0, out
    with open(out3) as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0].keys() == {"arch", "channel", "rs", "alpha", "s",
                              "f_model", "f_parent"}
    n_gga = len(P.S_GRID) * (1 + len(P.RS_VALUES))
    n_scan = len(P.S_GRID) * len(P.ALPHA_VALUES) * (1 + len(P.RS_VALUES))
    assert len(rows) == n_gga + n_scan
    gga_rows = [r for r in rows if r["arch"] == "deep_3x16"]
    assert len(gga_rows) == n_gga
    assert all(r["alpha"] == "" for r in gga_rows)
    scan_fx_rows = [r for r in rows
                    if r["arch"] == "deep_mgga_3x16" and r["channel"] == "fx"]
    assert sorted({r["alpha"] for r in scan_fx_rows}) == ["0", "1"]
    worst = max(abs(float(r["f_model"]) - float(r["f_parent"]))
                for r in scan_fx_rows)
    assert worst < 1e-10, worst
