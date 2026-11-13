"""The dissociation and fractional-charge figure, read off the matplotlib artists.

``make_diagnostics_figure.main`` on a run holding a hand ``diagnostics_curves.json``:
per panel (H2 against R, the H atom against N) one line per curve whose y
data are the curve's deviation from the panel's reference in kcal/mol, the
networks solid in their architecture's colour, the comparators dashed in the
Slim16 figure's greys (PBE in PBE-DF's), a zero line in each panel, the
legend naming every curve; the main copy carries three title lines and the
plain copy none; the printed manifest names every curve, its largest
deviation and the output paths. The network selection of the module is held
to a hand manifest.

Oracle: the hand deviations of ``_DEV``, from which the payload's energies
are built (the reference plus the deviation in Hartree), read back from the
figure as each file is written (``Figure.savefig`` spied), never from the
manifest the tool prints; the colours of ``arch_style`` and
``make_slim16_figure``. Bound: 1e-9 on the read-back deviations (the energies
round-trip through Hartree at 7e-14 kcal/mol).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.figure
import numpy as np
import pytest

matplotlib.use("Agg")

from xcquinox.pipeline import diagnostics  # noqa: E402

_HERE = Path(__file__).resolve().parent
_KCAL = 627.5094740631
_COMPARATORS = (("pbe", "PBE"), ("r2scan", "r2SCAN"), ("b3lyp", "B3LYP"),
                ("wb97m-v", "wB97M-V"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[f"{name}_under_test"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _hex(color):
    return mcolors.to_hex(color, keep_alpha=False)


def _snapshot(fig):
    """What ``fig`` shows: the figure-level title lines (each text of the
    figure once; the suptitle is one of them), the legend texts and per
    axes its lines."""
    title_lines = [line for t in fig.texts for line in t.get_text().split("\n")
                   if line.strip()]
    legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes
                                   if ax.get_legend() is not None]
    legend_texts = [t.get_text() for leg in legends for t in leg.get_texts()]
    axes = []
    for ax in fig.axes:
        lines = [{"x": np.asarray(line.get_xdata(), dtype=float),
                  "y": np.asarray(line.get_ydata(), dtype=float),
                  "color": _hex(line.get_color()), "linestyle": line.get_linestyle()}
                 for line in ax.get_lines()]
        axes.append({"lines": lines, "texts": [t.get_text() for t in ax.texts],
                     "titles": [ax.get_title(loc) for loc in ("left", "center", "right")
                                if ax.get_title(loc)]})
    return {"title_lines": title_lines, "legend": legend_texts, "axes": axes}


def _all_texts(snap):
    out = list(snap["title_lines"]) + list(snap["legend"])
    for ax in snap["axes"]:
        out += ax["titles"] + ax["texts"]
    return out


@pytest.fixture
def saves(monkeypatch):
    """The figures the tool writes, ``[(path, snapshot)]``, each read off the
    figure right after its file is written."""
    out = []
    original = matplotlib.figure.Figure.savefig

    def spy(self, fname, *args, **kwargs):
        result = original(self, fname, *args, **kwargs)
        out.append((str(fname), _snapshot(self)))
        return result

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)
    yield out
    import matplotlib.pyplot as plt
    plt.close("all")


def _printed_json(text):
    start = text.index("{")
    document, _end = json.JSONDecoder().raw_decode(text[start:])
    return document


def _scalars(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _scalars(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _scalars(value)
    else:
        yield obj


_R = [1.0, 1.4, 3.0, 6.0]
_R_REF = [-1.124, -1.168, -1.057, -1.0002]
_N = [0.0, 0.5, 1.0, 1.5, 2.0]
_E1, _E2 = -0.4998, -0.5154
_N_REF = [0.0, 0.5 * _E1, _E1, 0.5 * (_E1 + _E2), _E2]
_NETS = {"S_deep_3x16": "deep_3x16", "S_deep_geom_3x16": "deep_geom_3x16"}
#: hand deviations (kcal/mol) per curve: (H2 at _R, the H atom at _N)
_DEV = {
    "S_deep_3x16": ([0.9, 1.1, 14.0, 40.5], [0.0, -30.0, 0.3, -18.0, 9.0]),
    "S_deep_geom_3x16": ([0.7, 1.3, 16.0, 43.0], [0.0, -28.0, 0.5, -17.0, 8.5]),
    "PBE": ([1.2, 1.5, 15.0, 41.2], [0.0, -33.4, 0.12, -19.0, 8.1]),
    "r2SCAN": ([0.4, 0.8, 12.5, 38.0], [0.0, -27.0, 0.05, -15.0, 6.0]),
    "B3LYP": ([-0.6, -0.3, 18.0, 52.0], [0.0, -22.0, 0.2, -12.0, 5.0]),
    "wB97M-V": ([0.2, -0.4, 20.0, 60.0], [0.0, -14.0, 0.1, -8.0, 3.0]),
}


def _curve(reference, deviations_kcal):
    energies = [e + d / _KCAL for e, d in zip(reference, deviations_kcal)]
    return {"E": energies, "converged": [True] * len(energies), "cycles": [7] * len(energies)}


def _payload():
    return {
        "identity": {"basis": "def2-TZVP", "grid_level": 4, "r_bohr": _R, "n": _N},
        "networks": {"S_deep_3x16": {"index": 0, "arch_name": "deep_3x16"},
                     "S_deep_geom_3x16": {"index": 2, "arch_name": "deep_geom_3x16"}},
        "h2": {"r_bohr": _R, "reference_ccsd": _R_REF,
               "curves": {label: _curve(_R_REF, dev[0]) for label, dev in _DEV.items()}},
        "h": {"n": _N, "reference_linear": _N_REF,
              "integer_energies": {"0": 0.0, "1": _E1, "2": _E2},
              "curves": {label: _curve(_N_REF, dev[1]) for label, dev in _DEV.items()}},
    }


def _expected_style(slim16, arch_style, label):
    if label in _NETS:
        return _hex(arch_style.arch_color(_NETS[label])), "-"
    key = {display: key for key, display in _COMPARATORS}[label]
    return _hex(slim16.PBE_COLOR if key == "pbe" else slim16.COMPARATOR_COLORS[key]), "--"


def _panel(snap, x_grid):
    found = [ax for ax in snap["axes"]
             if any(line["x"].shape == (len(x_grid),)
                    and np.allclose(line["x"], x_grid, rtol=0, atol=1e-12)
                    for line in ax["lines"])]
    assert len(found) == 1
    return found[0]


def _check_curves(snap, slim16, arch_style):
    for x_grid, which in ((_R, 0), (_N, 1)):
        ax = _panel(snap, x_grid)
        for label, dev in _DEV.items():
            y = np.array(dev[which])
            hits = [line for line in ax["lines"]
                    if line["x"].shape == y.shape
                    and np.allclose(line["x"], x_grid, rtol=0, atol=1e-12)
                    and np.allclose(line["y"], y, rtol=1e-9, atol=1e-9)]
            assert len(hits) == 1, (label, which, len(hits))
            assert (hits[0]["color"], hits[0]["linestyle"]) == _expected_style(
                slim16, arch_style, label), label
        assert any(line["y"].size and np.all(line["y"] == 0.0) for line in ax["lines"])
    for label in _DEV:
        assert any(label in text for text in snap["legend"]), label


def test_the_figure_draws_the_deviations_from_the_references(tmp_path, saves, capsys):
    tool = _load("make_diagnostics_figure")
    slim16 = _load("make_slim16_figure")
    sys.path.insert(0, str(_HERE))
    import arch_style
    run = tmp_path / "run"
    run.mkdir()
    (run / "diagnostics_curves.json").write_text(json.dumps(_payload()))
    assert tool.main([str(run), "--plain"]) == 0
    printed = capsys.readouterr().out
    main_png, plain_png = run / "diagnostics.png", run / "diagnostics_plain.png"
    assert main_png.is_file() and plain_png.is_file()
    by_name = {Path(path).name: snap for path, snap in saves}
    assert set(by_name) == {"diagnostics.png", "diagnostics_plain.png"}
    for snap in by_name.values():
        _check_curves(snap, slim16, arch_style)
    title = by_name["diagnostics.png"]["title_lines"]
    assert len(title) == 3, title
    plain = by_name["diagnostics_plain.png"]
    assert plain["title_lines"] == []
    assert not any(line in text for line in title for text in _all_texts(plain))

    manifest = _printed_json(printed)
    strings = [v for v in _scalars(manifest) if isinstance(v, str)]
    numbers = [float(v) for v in _scalars(manifest)
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for label, dev in _DEV.items():
        assert label in strings
        for values in dev:
            largest = float(np.max(np.abs(np.array(values))))
            assert any(math.isclose(v, largest, rel_tol=1e-9, abs_tol=1e-9) for v in numbers)
    for path in (main_png, plain_png):
        assert any(s.endswith(".png") and Path(s).resolve() == path.resolve() for s in strings)

    elsewhere = tmp_path / "out" / "figure.png"
    elsewhere.parent.mkdir()
    assert tool.main([str(run), "--out", str(elsewhere), "--plain"]) == 0
    assert elsewhere.is_file() and (elsewhere.parent / "figure_plain.png").is_file()
    second = tmp_path / "run2"
    second.mkdir()
    (second / "diagnostics_curves.json").write_text(json.dumps(_payload()))
    assert tool.main([str(second)]) == 0
    assert (second / "diagnostics.png").is_file()
    assert not (second / "diagnostics_plain.png").exists()
    capsys.readouterr()


def _network(label, index, arch_name):
    return {"label": label, "index": index, "arch_name": arch_name, "kind": "pretrain"}


def test_one_network_per_architecture_is_the_first_in_manifest_order():
    """``select_networks``: the first of each architecture in manifest order,
    keyed by label; with labels named, those; an unknown label refused."""
    manifest = {"networks": [
        _network("S_deep_3x16", 0, "deep_3x16"),
        _network("slim05_deep_geom_3x16", 1, "deep_geom_3x16"),
        _network("slim05_deep_3x16", 2, "deep_3x16"),
        _network("S_deep_geom_3x16", 3, "deep_geom_3x16"),
        _network("v7_3x16_ss7", 4, "medium"),
    ]}
    picked = diagnostics.select_networks(manifest)
    assert list(picked) == ["S_deep_3x16", "slim05_deep_geom_3x16", "v7_3x16_ss7"]
    assert picked["slim05_deep_geom_3x16"] == {"index": 1, "arch_name": "deep_geom_3x16"}
    named = diagnostics.select_networks(manifest, labels=["S_deep_geom_3x16"])
    assert named == {"S_deep_geom_3x16": {"index": 3, "arch_name": "deep_geom_3x16"}}
    with pytest.raises((KeyError, ValueError)):
        diagnostics.select_networks(manifest, labels=["no_such_network"])


@pytest.fixture
def open_figures(monkeypatch):
    """The figures written, ``[(path, figure)]``, each held open."""
    out = []
    original = matplotlib.figure.Figure.savefig

    def spy(self, fname, *args, **kwargs):
        result = original(self, fname, *args, **kwargs)
        out.append((str(fname), self))
        return result

    import matplotlib.pyplot as plt
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    yield out
    plt.close("all")


def _small_payload(curves_h2, curves_h, networks):
    r = [1.0, 2.0, 3.0]
    n = [0.0, 0.5, 1.0]
    return {"identity": {"r_bohr": r, "n": n}, "networks": networks,
            "h2": {"r_bohr": r, "reference_ccsd": [-1.1, -1.05, -1.0], "curves": curves_h2},
            "h": {"n": n, "reference_linear": [0.0, -0.25, -0.5],
                  "integer_energies": {"0": 0.0, "1": -0.5, "2": -0.5}, "curves": curves_h}}


def _dashes(line):
    """The unscaled dash pattern of a line, the artist's own record of the
    tuple it was given (matplotlib reports every tuple as dashed)."""
    pattern = getattr(line, "_unscaled_dash_pattern", None) or getattr(line, "_dash_pattern", None)
    return tuple(pattern[1]) if pattern and pattern[1] is not None else None


def test_an_unconverged_point_is_marked_and_a_missing_one_leaves_a_gap(tmp_path, open_figures):
    """A missing energy (None) and a non-finite one leave gaps in the line
    (NaN in its y data, the manifest naming the x values); an unconverged
    point carries an open circle in the curve's colour at its position; the
    manifest holds no NaN once cleaned."""
    tool = _load("make_diagnostics_figure")
    h2 = {"S_deep_3x16": {"E": [-1.1 + 1.0 / _KCAL, None, -1.0 + 3.0 / _KCAL],
                          "converged": [True, True, False], "cycles": [3, 3, 100]}}
    h = {"S_deep_3x16": {"E": [0.0, float("nan"), -0.5 + 2.0 / _KCAL],
                         "converged": [True, True, True], "cycles": [0, 3, 3]}}
    networks = {"S_deep_3x16": {"index": 0, "arch_name": "deep_3x16"}}
    manifest = tool.plot_diagnostics(_small_payload(h2, h, networks), tmp_path / "fig.png")
    (_path, fig), = open_figures
    ax_r, ax_n = fig.axes[:2]
    line = [ln for ln in ax_r.get_lines() if ln.get_label() == "S_deep_3x16"][0]
    y = np.asarray(line.get_ydata(), dtype=float)
    assert y[0] == pytest.approx(1.0) and math.isnan(y[1]) and y[2] == pytest.approx(3.0)
    markers = [ln for ln in ax_r.get_lines() if ln.get_marker() == "o"
               and ln.get_linestyle() in ("None", "none", " ", "")]
    assert len(markers) == 1 and list(markers[0].get_xdata()) == [3.0]
    assert _hex(markers[0].get_markeredgecolor()) == _hex(line.get_color())
    assert mcolors.to_rgba(markers[0].get_markerfacecolor())[3] == 0.0
    line_n = [ln for ln in ax_n.get_lines() if ln.get_label() == "S_deep_3x16"][0]
    assert math.isnan(np.asarray(line_n.get_ydata(), dtype=float)[1])
    entries = {(c["panel"], c["label"]): c for c in manifest["curves"]}
    assert entries[("h2", "S_deep_3x16")]["missing_x"] == [2.0]
    assert entries[("h2", "S_deep_3x16")]["unconverged_x"] == [3.0]
    assert entries[("h", "S_deep_3x16")]["missing_x"] == [0.5]
    assert "NaN" not in json.dumps(diagnostics.json_safe(manifest))


def test_networks_of_one_architecture_read_apart_by_rank_and_comparators_by_dashes(
        tmp_path, open_figures):
    """Two networks of one architecture share the colour but not the line
    style (the first solid, the second, by its rank, dashed with a
    pattern of its own); the four comparators carry four distinct dash
    patterns; the zero line is lighter than any curve; more networks of
    one architecture than line styles are refused by name."""
    tool = _load("make_diagnostics_figure")

    def curve(values):
        return {"E": values, "converged": [True] * len(values), "cycles": [3] * len(values)}

    labels = ("S_deep_3x16", "slim05_deep_3x16", "PBE", "r2SCAN", "B3LYP", "wB97M-V")
    h2 = {label: curve([-1.1 + 1.0 / _KCAL, -1.05 + 1.0 / _KCAL, -1.0 + 1.0 / _KCAL])
          for label in labels}
    h = {label: curve([1.0 / _KCAL, -0.25 + 1.0 / _KCAL, -0.5 + 1.0 / _KCAL]) for label in labels}
    networks = {"S_deep_3x16": {"index": 0, "arch_name": "deep_3x16"},
                "slim05_deep_3x16": {"index": 1, "arch_name": "deep_3x16"}}
    manifest = tool.plot_diagnostics(_small_payload(h2, h, networks), tmp_path / "fig.png")
    (_path, fig), = open_figures
    ax = fig.axes[0]
    by_label = {ln.get_label(): ln for ln in ax.get_lines() if ln.get_label() in h2}
    assert _hex(by_label["S_deep_3x16"].get_color()) == _hex(by_label["slim05_deep_3x16"].get_color())
    assert by_label["S_deep_3x16"].get_linestyle() == "-"
    assert by_label["slim05_deep_3x16"].get_linestyle() == "--"
    patterns = {label: _dashes(by_label[label]) for label in ("PBE", "r2SCAN", "B3LYP", "wB97M-V")}
    assert all(p is not None for p in patterns.values()) and len(set(patterns.values())) == 4
    assert _dashes(by_label["slim05_deep_3x16"]) not in set(patterns.values())
    zero = [ln for ln in ax.get_lines() if ln.get_ydata().size
            and np.all(np.asarray(ln.get_ydata(), dtype=float) == 0.0)][0]
    r, g, b = mcolors.to_rgb(zero.get_color())
    assert 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.6
    assert {c["label"]: c["rank"] for c in manifest["curves"] if c["kind"] == "network"} == {
        "S_deep_3x16": 0, "slim05_deep_3x16": 1}
    many = {f"S_deep_3x16_s{i}": {"index": i, "arch_name": "deep_3x16"}
            for i in range(len(tool.NETWORK_DASHES) + 1)}
    with pytest.raises(ValueError, match="deep_3x16"):
        tool.plot_diagnostics(
            _small_payload({k: curve([-1.0, -1.0, -1.0]) for k in many},
                           {k: curve([0.0, -0.25, -0.5]) for k in many}, many),
            tmp_path / "many.png")


def test_a_plain_run_names_a_payload_computed_for_other_networks_or_grids(tmp_path, capsys):
    """A payload beside the run computed with --networks or another grid is
    redrawn by a plain run with a notice on stderr and in the manifest
    naming what differs from the defaults; a payload with the defaults
    draws without a notice."""
    tool = _load("make_diagnostics_figure")
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "width": 4, "networks": [{"label": "S_deep_3x16", "index": 0, "arch_name": "deep_3x16"},
                                 {"label": "slim05_deep_3x16", "index": 1, "arch_name": "deep_3x16"}]}))
    r = diagnostics.grid(*diagnostics.R_BOHR_DEFAULT)
    n = diagnostics.grid(*diagnostics.N_DEFAULT)

    def payload(labels, r_values, n_values):
        def curve(values):
            return {"E": values, "converged": [True] * len(values), "cycles": [1] * len(values)}
        return {"identity": {"r_bohr": r_values, "n": n_values},
                "networks": {label: {"index": i, "arch_name": "deep_3x16"}
                             for i, label in enumerate(labels)},
                "h2": {"r_bohr": r_values, "reference_ccsd": [-1.0] * len(r_values),
                       "curves": {label: curve([-1.0] * len(r_values)) for label in labels}},
                "h": {"n": n_values, "reference_linear": [0.0] * len(n_values),
                      "integer_energies": {"0": 0.0, "1": -0.5, "2": -0.5},
                      "curves": {label: curve([0.0] * len(n_values)) for label in labels}}}

    path = run / diagnostics.CURVES_FILE
    path.write_text(json.dumps(payload(["slim05_deep_3x16"], r, n)))
    assert tool.main([str(run)]) == 0
    captured = capsys.readouterr()
    assert "notice" in captured.err and "slim05_deep_3x16" in captured.err
    assert "slim05_deep_3x16" in _printed_json(captured.out)["notice"]
    path.write_text(json.dumps(payload(["S_deep_3x16"], [1.0, 2.0], n)))
    assert tool.main([str(run)]) == 0
    assert "bond lengths" in capsys.readouterr().err
    path.write_text(json.dumps(payload(["S_deep_3x16"], r, n)))
    assert tool.main([str(run)]) == 0
    captured = capsys.readouterr()
    assert "notice" not in captured.err and _printed_json(captured.out)["notice"] is None


def _fci_h2(r_bohr: float) -> float:
    """Full CI of H2 at ``r_bohr`` on the module's own molecule (def2-TZVP,
    the core-potential keyword), exact in the basis."""
    from pyscf import fci, scf
    mf = scf.RHF(diagnostics._pyscf_mol(diagnostics.h2_spec(r_bohr))).run(conv_tol=1e-12)
    return float(fci.FCI(mf).kernel()[0])


@pytest.mark.slow
def test_compute_diagnostics_on_a_clone_writes_the_payload_the_figure_reads(tmp_path, capsys):
    """The whole assembly on a run directory with a hand manifest and a
    seed-0 deep_3x16 clone handed in, at one bond length and two electron
    numbers: one network (the clone's label), the H2 point through the
    pyscfad configuration (converged), the CCSD reference at the full-CI
    energy of the basis, the four comparators by label on both curves, the
    fractional curve at N = 0 zero without a run and at N = 1 the plain
    atom, the reference line through the UHF energy; the payload written
    beside the run, read back by the figure tool with a notice naming its
    grids, and drawn."""
    import dataclasses
    import types

    import xcquinox.pipeline as pipeline
    from pyscf import scf
    tool = _load("make_diagnostics_figure")
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "kind": "slim16_eval", "width": 4, "n_specs": 2,
        "networks": [{"label": "S_deep_3x16", "index": 0, "arch_name": "deep_3x16"},
                     {"label": "slim05_deep_3x16", "index": 1, "arch_name": "deep_3x16"}]}))
    arch = dataclasses.replace(pipeline.get_architecture("deep_3x16"),
                               use_polarized_correlation=True, zero_init_final_layer=False)
    xnet, cnet = pipeline.create_network_pair(arch, seed=0)
    model = pipeline.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
    models = {"S_deep_3x16": (model, types.SimpleNamespace(arch=arch))}
    payload = diagnostics.compute_diagnostics(run, r_values=[1.4], n_values=[0.0, 1.0],
                                              models=models)
    assert payload["networks"] == {"S_deep_3x16": {"index": 0, "arch_name": "deep_3x16"}}
    assert set(payload["h2"]["curves"]) == {"S_deep_3x16", "PBE", "r2SCAN", "B3LYP", "wB97M-V"}
    assert set(payload["h"]["curves"]) == set(payload["h2"]["curves"])
    h2 = payload["h2"]["curves"]["S_deep_3x16"]
    assert h2["converged"] == [True] and math.isfinite(h2["E"][0])
    assert payload["h2"]["reference_ccsd"][0] == pytest.approx(_fci_h2(1.4), abs=1e-8)
    assert payload["identity"]["solver"]["h2"]["backend"] == "pyscfad"
    assert payload["identity"]["solver"]["h"]["backend"] == "manual"
    h = payload["h"]["curves"]["S_deep_3x16"]
    assert h["E"][0] == 0.0 and h["cycles"][0] == 0 and h["converged"][0] is True
    e_uhf = float(scf.UHF(diagnostics._pyscf_mol(diagnostics.h_atom_spec())).run().e_tot)
    assert payload["h"]["integer_energies"]["1"] == pytest.approx(e_uhf, abs=1e-10)
    assert payload["h"]["reference_linear"] == pytest.approx([0.0, e_uhf], abs=1e-10)
    assert payload["h"]["curves"]["PBE"]["E"][0] == 0.0

    path = run / diagnostics.CURVES_FILE
    diagnostics.write_payload(path, payload)
    read = json.loads(path.read_text(encoding="utf-8"))
    assert read["h2"]["curves"]["S_deep_3x16"]["E"] == h2["E"]
    reread, notice = tool.load_or_compute(run, None, None, None, False)
    assert reread == read
    assert "bond lengths" in notice and "electron numbers" in notice
    manifest = tool.plot_diagnostics(read, run / "fig.png")
    assert {c["label"] for c in manifest["curves"]} == set(payload["h2"]["curves"])
    assert (run / "fig.png").is_file()


def test_the_rank_and_the_legend_follow_the_manifest_index_not_the_payload_order(
        tmp_path, open_figures):
    """A payload read back from disk holds its keys sorted: the network with
    the lower manifest index is solid and listed first whatever the key
    order, and the legend lists the networks by index, then the comparators
    in the module's order; the order survives the writer's sorting."""
    tool = _load("make_diagnostics_figure")

    def curve(values):
        return {"E": values, "converged": [True] * len(values), "cycles": [3] * len(values)}

    labels = ["B3LYP", "PBE", "S_deep_3x16", "r2SCAN", "slim05_deep_3x16", "wB97M-V"]
    h2 = {label: curve([-1.1 + 1.0 / _KCAL, -1.05 + 1.0 / _KCAL, -1.0 + 1.0 / _KCAL])
          for label in labels}
    h = {label: curve([1.0 / _KCAL, -0.25 + 1.0 / _KCAL, -0.5 + 1.0 / _KCAL]) for label in labels}
    networks = {"slim05_deep_3x16": {"index": 1, "arch_name": "deep_3x16"},
                "S_deep_3x16": {"index": 0, "arch_name": "deep_3x16"}}
    payload = _small_payload(h2, h, networks)
    expected = ["S_deep_3x16", "slim05_deep_3x16", "PBE", "r2SCAN", "B3LYP", "wB97M-V"]
    assert tool._curve_order(payload) == expected
    manifest = tool.plot_diagnostics(payload, tmp_path / "fig.png")
    ranks = {c["label"]: c["rank"] for c in manifest["curves"] if c["kind"] == "network"}
    assert ranks == {"S_deep_3x16": 0, "slim05_deep_3x16": 1}
    (_path, fig), = open_figures
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert legend[:len(expected)] == expected and len(legend) == len(expected) + 1
    by_label = {ln.get_label(): ln for ln in fig.axes[0].get_lines() if ln.get_label() in h2}
    assert by_label["S_deep_3x16"].get_linestyle() == "-"
    assert by_label["slim05_deep_3x16"].get_linestyle() == "--"
    path = tmp_path / diagnostics.CURVES_FILE
    diagnostics.write_payload(path, payload)
    read = json.loads(path.read_text(encoding="utf-8"))
    assert list(read["networks"]) != list(networks)
    assert tool._curve_order(read) == expected
    assert tool._ranks(read["networks"]) == tool._ranks(networks)
