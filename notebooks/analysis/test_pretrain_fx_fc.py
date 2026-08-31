#!/usr/bin/env python
"""Tests for pretrain_fx_fc.py -- pretrained enhancement factors vs parent.

The load-bearing pin is the anchored identity: a freshly-built parent-anchored
network with a zero-initialized final layer IS the parent (F = F_parent +
T(g), g = 0 at init), so the module's curves must reproduce the PBE baselines
to round-off. An unanchored build fails the same assertion by more than 1e-2,
which is what makes the pin discriminating rather than vacuous.
"""
import csv
import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pretrain_fx_fc as P  # noqa: E402


def _fresh_model(arch_name: str, *, parent_anchor: bool):
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
    return AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


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


def test_meta_gga_is_refused_by_name(monkeypatch, tmp_path):
    class FakeArch:
        meta_gga = True

    monkeypatch.setattr(
        "xcquinox.alec.cluster.fidelity.build_certified_model",
        lambda cfg, run_dir, name: (FakeArch(), object()))
    monkeypatch.setattr(
        "xcquinox.alec.cluster.grid_config.load_grid_config",
        lambda path: object())
    with pytest.raises(ValueError, match="meta-GGA"):
        P.load_pretrained_model(tmp_path, "deep_mgga_3x16")
