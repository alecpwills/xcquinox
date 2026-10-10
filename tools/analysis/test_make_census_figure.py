"""The cold-start census figure, read off the matplotlib artists.

``make_census_figure.main`` on a hand census whose cells are out of order:
one column per cell at its architecture's tick, the ticks in ``arch_style``
order; per column the converged species as filled markers in the
architecture's colour at their cycle count (x jittered inside the column),
the unconverged as open markers at theirs, ``converged/species`` written at
the column; an errored cell draws no marker and carries its error. Two
title lines in the main copy, none in the plain copy. Two census files with
labels draw every row of both. The cycles are read off the energy trace at
a tolerance when asked, or when a cell records that its solver ran its whole
budget without freezing on convergence.

Oracle: the hand census and hand traces (closed forms); ``arch_style``'s
order and colours; the figures read back as each file is written
(``Figure.savefig`` spied), never from the printed manifest.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.figure
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path as MplPath
from matplotlib.text import Annotation

matplotlib.use("Agg")

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[f"{name}_under_test"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _hex(color):
    return mcolors.to_hex(color, keep_alpha=False)


def _no_face(color):
    if color is None:
        return True
    if isinstance(color, str) and color.lower() == "none":
        return True
    return mcolors.to_rgba(color)[3] == 0.0


def _snapshot(fig):
    title_lines = [line for t in fig.texts for line in t.get_text().split("\n")
                   if line.strip()]
    legends = list(fig.legends) + [ax.get_legend() for ax in fig.axes
                                   if ax.get_legend() is not None]
    legend_texts = [t.get_text() for leg in legends for t in leg.get_texts()]
    axes = []
    for ax in fig.axes:
        lines = [{"x": np.asarray(line.get_xdata(), dtype=float),
                  "y": np.asarray(line.get_ydata(), dtype=float),
                  "color": _hex(line.get_color()), "marker": line.get_marker(),
                  "mfc": line.get_markerfacecolor(), "mec": line.get_markeredgecolor()}
                 for line in ax.get_lines()]
        collections = []
        for coll in ax.collections:
            if not isinstance(coll, PathCollection):
                continue
            offsets = np.asarray(coll.get_offsets(), dtype=float)
            if offsets.ndim != 2 or offsets.shape[0] == 0:
                continue
            closed = any(path.codes is not None and MplPath.CLOSEPOLY in path.codes
                         for path in coll.get_paths())
            collections.append({"offsets": offsets,
                                "facecolors": np.asarray(coll.get_facecolors()),
                                "edgecolors": np.asarray(coll.get_edgecolors()),
                                "fillable": closed})
        texts = []
        for t in ax.texts:
            x = float(t.xy[0]) if isinstance(t, Annotation) else float(t.get_position()[0])
            texts.append({"text": t.get_text(), "x": x})
        ticks = list(zip([float(v) for v in ax.get_xticks()],
                         [t.get_text() for t in ax.get_xticklabels()]))
        axes.append({"lines": lines, "collections": collections, "texts": texts,
                     "titles": [ax.get_title(loc) for loc in ("left", "center", "right")
                                if ax.get_title(loc)], "ticks": ticks})
    return {"title_lines": title_lines, "legend": legend_texts, "axes": axes}


def _all_texts(snap):
    out = list(snap["title_lines"]) + list(snap["legend"])
    for ax in snap["axes"]:
        out += ax["titles"] + [t["text"] for t in ax["texts"]]
        out += [label for _x, label in ax["ticks"]]
    return out


@pytest.fixture
def saves(monkeypatch):
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


def _row(name, converged, cycles):
    trace = [-1.0 - 1e-3 / (k + 1) for k in range(25)]
    return {"name": name, "converged": converged, "cycles_run": cycles,
            "total_energy": trace[-1], "finite": True,
            "last_delta_e": abs(trace[-1] - trace[-2]), "energy_trace": trace, "wall_s": 1.5}


def _cell(arch, rows, error=None, n_species=None):
    return {"arch": arch, "spec_path": f"/runs/specs/{arch}.spec",
            "n_species": len(rows) if n_species is None else n_species,
            "n_converged": sum(1 for r in rows if r["converged"]),
            "error": error, "rows": rows}


_CENSUS = {"cells": [
    _cell("deep_geom_3x16", [_row("H", True, 7), _row("Li", True, 9), _row("N2", False, 25)]),
    _cell("deep_3x16", [_row("H", True, 12), _row("Li", False, 25), _row("N2", False, 25)]),
    _cell("deep_attn_3x16", [], error="ValueError: refused", n_species=3),
]}
_SECOND = {"cells": [_cell("deep_3x16", [_row("H", True, 5), _row("F", False, 25)])]}


def _marker_positions(ax, color):
    """``{(x, y): filled}`` of the markers drawn in ``color``."""
    found = {}

    def put(x, y, filled):
        key = (round(float(x), 9), float(y))
        found[key] = found.get(key, False) or filled

    for line in ax["lines"]:
        marker = line["marker"]
        if marker in (None, "", " ", "None", "none"):
            continue
        filled = MarkerStyle(marker).is_filled() and not _no_face(line["mfc"])
        drawn = _hex(line["mfc"]) if filled else _hex(line["mec"])
        if drawn != color:
            continue
        for x, y in zip(line["x"], line["y"]):
            put(x, y, filled)
    for coll in ax["collections"]:
        faces, edges = coll["facecolors"], coll["edgecolors"]
        for i, (x, y) in enumerate(coll["offsets"]):
            face = faces[i % len(faces)] if len(faces) else None
            edge = edges[i % len(edges)] if len(edges) else face
            filled = coll["fillable"] and face is not None and face[3] > 0
            drawn = _hex(face[:3]) if filled else (_hex(edge[:3]) if edge is not None else None)
            if drawn != color:
                continue
            put(x, y, filled)
    return found


def _texts_near(ax, x, half_width=0.45):
    return [t["text"] for t in ax["texts"] if abs(t["x"] - x) < half_width]


def test_the_figure_draws_one_column_per_cell_in_arch_style_order(tmp_path, saves, capsys):
    tool = _load("make_census_figure")
    sys.path.insert(0, str(_HERE))
    import arch_style
    census = tmp_path / "coldstart_census.json"
    census.write_text(json.dumps(_CENSUS))
    out = tmp_path / "census.png"
    assert tool.main([str(census), "--out", str(out), "--plain"]) == 0
    printed = capsys.readouterr().out
    assert out.is_file() and (tmp_path / "census_plain.png").is_file()
    by_name = {Path(path).name: snap for path, snap in saves}
    assert set(by_name) == {"census.png", "census_plain.png",
                            "census_steps.png", "census_steps_plain.png"}
    by_name = {name: snap for name, snap in by_name.items() if "steps" not in name}

    cells = {cell["arch"]: cell for cell in _CENSUS["cells"]}
    order = arch_style.order_present([cell["arch"] for cell in _CENSUS["cells"]])
    assert order == ["deep_3x16", "deep_attn_3x16", "deep_geom_3x16"]
    for snap in by_name.values():
        drawn = [ax for ax in snap["axes"] if ax["ticks"] and any(
            arch_style.as_shown(order[0]) in label for _x, label in ax["ticks"])]
        assert len(drawn) == 1
        ax = drawn[0]
        ticks = sorted((x, label) for x, label in ax["ticks"] if label.strip())
        assert len(ticks) == len(order)
        for (x, label), arch in zip(ticks, order):
            assert arch_style.as_shown(arch) in label
            cell = cells[arch]
            color = _hex(arch_style.arch_color(arch))
            near = {k: v for k, v in _marker_positions(ax, color).items() if abs(k[0] - x) < 0.45}
            if cell["error"]:
                assert near == {}
                assert any(cell["error"] in t for t in _texts_near(ax, x))
                continue
            filled = sorted(y for (_x, y), f in near.items() if f)
            open_ = sorted(y for (_x, y), f in near.items() if not f)
            assert filled == sorted(r["cycles_run"] for r in cell["rows"] if r["converged"])
            assert open_ == sorted(r["cycles_run"] for r in cell["rows"] if not r["converged"])
            assert any(f"{cell['n_converged']}/{cell['n_species']}" in t
                       for t in _texts_near(ax, x))
    title = by_name["census.png"]["title_lines"]
    assert len(title) == 2
    plain = by_name["census_plain.png"]
    assert plain["title_lines"] == []
    assert not any(line in text for line in title for text in _all_texts(plain))

    manifest = _printed_json(printed)
    strings = [v for v in _scalars(manifest) if isinstance(v, str)]
    for cell in _CENSUS["cells"]:
        assert cell["arch"] in strings
        for r in cell["rows"]:
            if not r["converged"]:
                assert r["name"] in strings
    assert "2/3" in strings and "1/3" in strings
    assert any("ValueError: refused" in s for s in strings)

    second = tmp_path / "second.json"
    second.write_text(json.dumps(_SECOND))
    saves.clear()
    two = tmp_path / "two.png"
    assert tool.main([str(census), str(second), "--labels", "first,second", "--out", str(two)]) == 0
    capsys.readouterr()
    assert sorted(Path(path).name for path, _snap in saves) == ["two.png", "two_steps.png"]
    (snap,) = [snap for path, snap in saves if Path(path).name == "two.png"]
    texts = _all_texts(snap)
    assert any("first" in t for t in texts) and any("second" in t for t in texts)
    n_points = sum(len(_marker_positions(ax, _hex(arch_style.arch_color(arch))))
                   for ax in snap["axes"] for arch in order)
    assert n_points == sum(len(c["rows"]) for c in _CENSUS["cells"] + _SECOND["cells"])


def test_the_cycles_are_read_off_the_trace_when_asked(tmp_path):
    """``cycles_to_convergence`` is the first cycle whose energy step falls
    below the tolerance, counted from one (on the hand trace the steps are
    0.1, 0.05, 5e-7 and 1e-7: 4 at 1e-6, none at 1e-9, none through a missing
    value); ``columns`` with a tolerance reads every row that way, and by
    itself a cell that records freeze_on_convergence false with its conv_tol,
    while a cell that froze keeps the census's fields."""
    tool = _load("make_census_figure")
    trace = [-1.0, -1.1, -1.15, -1.1500005, -1.1500006]
    row = {"name": "H", "converged": False, "cycles_run": 5, "energy_trace": trace}
    assert tool.cycles_to_convergence(row, 1e-6) == 4
    assert tool.cycles_to_convergence(row, 1e-9) is None
    assert tool.cycles_to_convergence({**row, "energy_trace": [-1.0, None, -1.1]}, 1e-6) is None

    def cell(arch, **extra):
        return {"arch": arch, "n_species": 2, "n_converged": 0, "error": None,
                "rows": [row, {**row, "name": "Li", "energy_trace": trace[:3]}], **extra}

    path = tmp_path / "coldstart_census.json"
    path.write_text(json.dumps({"cells": [
        cell("deep_3x16", conv_tol=1e-6, freeze_on_convergence=False),
        cell("deep_geom_3x16", conv_tol=1e-6, freeze_on_convergence=True)]}))
    cols = {c["arch"]: c for c in tool.columns([str(path)])}
    assert cols["deep_3x16"]["cycles_read"] == "trace below 1e-06"
    assert cols["deep_3x16"]["converged_cycles"] == [4]
    assert [n for n, _c in cols["deep_3x16"]["unconverged"]] == ["Li"]
    assert cols["deep_geom_3x16"]["cycles_read"] == "census"
    assert cols["deep_geom_3x16"]["converged_cycles"] == []
    forced = {c["arch"]: c for c in tool.columns([str(path)], tol=1e-9)}
    assert forced["deep_3x16"]["converged_cycles"] == []


def _trace_row(name, converged, cycles, trace):
    return {"name": name, "converged": converged, "cycles_run": cycles,
            "total_energy": trace[-1], "finite": True,
            "last_delta_e": abs(trace[-1] - trace[-2]), "energy_trace": trace, "wall_s": 1.0}


def test_cells_sharing_an_architecture_all_draw_and_the_steps_figure_shows_the_depth(
        tmp_path, saves, capsys):
    """Two cells of one architecture in one census draw two columns named by
    their specs; the companion steps figure draws one marker per species at
    its last energy step on a logarithmic axis, filled for a converged
    species and open for an unconverged one, with the cell's tolerance as a
    short line where recorded; its plain copy carries no title lines."""
    tool = _load("make_census_figure")
    sys.path.insert(0, str(_HERE))
    import arch_style
    cell_a = {"arch": "deep_3x16", "spec_path": "/runs/specs/spec_0000.spec", "n_species": 2,
              "n_converged": 1, "error": None, "conv_tol": 1e-6,
              "rows": [_trace_row("H", True, 7, [-1.0, -1.1, -1.1000001]),
                       _trace_row("Li", False, 25, [-7.0, -7.4, -7.401])]}
    cell_b = {"arch": "deep_3x16", "spec_path": "/runs/specs/spec_0001.spec", "n_species": 1,
              "n_converged": 1, "error": None,
              "rows": [_trace_row("H", True, 9, [-1.0, -1.1, -1.10000005])]}
    path = tmp_path / "coldstart_census.json"
    path.write_text(json.dumps({"cells": [cell_a, cell_b]}))
    out = tmp_path / "census.png"
    assert tool.main([str(path), "--out", str(out), "--plain"]) == 0
    capsys.readouterr()
    by_name = {Path(p).name: snap for p, snap in saves}
    assert set(by_name) == {"census.png", "census_plain.png", "census_steps.png",
                            "census_steps_plain.png"}
    ticks = [label for _x, label in by_name["census.png"]["axes"][0]["ticks"] if label.strip()]
    assert len(ticks) == 2 and all(arch_style.as_shown("deep_3x16") in t for t in ticks)
    assert "spec_0000" in ticks[0] and "spec_0001" in ticks[1]
    steps = by_name["census_steps.png"]["axes"][0]
    points = []
    for coll in steps["collections"]:
        for i, (x, y) in enumerate(coll["offsets"]):
            face = coll["facecolors"][i % len(coll["facecolors"])] if len(coll["facecolors"]) else None
            points.append((round(float(x)), float(y), coll["fillable"] and face is not None
                           and face[3] > 0))
    expected = sorted([(0, 1e-7, True), (0, 1e-3, False), (1, 5e-8, True)])
    assert len(points) == 3
    for (x, y, filled), (ex, ey, ef) in zip(sorted(points), expected):
        assert x == ex and filled == ef and y == pytest.approx(ey, rel=1e-6)
    tol = [ln for ln in steps["lines"] if ln["y"].size and np.allclose(ln["y"], 1e-6)]
    assert len(tol) == 1 and np.allclose(tol[0]["x"], [-0.3, 0.3])
    assert len(by_name["census_steps.png"]["title_lines"]) == 2
    assert by_name["census_steps_plain.png"]["title_lines"] == []


def test_the_trace_reading_keeps_a_census_flagged_convergence(tmp_path):
    """Under the trace reading a row the census flags converged keeps its
    flag at its cycle count when the trace shows no crossing; a row flagged
    unconverged with no crossing stays unconverged."""
    tool = _load("make_census_figure")
    flat = [-1.0, -1.001, -1.0015, -1.0018]
    cell = {"arch": "deep_3x16", "n_species": 2, "n_converged": 1, "error": None,
            "conv_tol": 1e-6, "freeze_on_convergence": False,
            "rows": [_trace_row("H", True, 4, flat), _trace_row("Li", False, 4, flat)]}
    path = tmp_path / "coldstart_census.json"
    path.write_text(json.dumps({"cells": [cell]}))
    (col,) = tool.columns([str(path)])
    assert col["cycles_read"] == "trace below 1e-06"
    assert col["converged_cycles"] == [4]
    assert [name for name, _c in col["unconverged"]] == ["Li"] and col["n_converged"] == 1


def test_the_steps_figure_reads_a_frozen_solver_at_its_convergence_cycle(tmp_path):
    """A row of a solver that froze on convergence (its trace repeating the
    converged energy after cycles_run) is read at the convergence cycle,
    |E_c - E_c-1| with c = cycles_run, not at the zero step of the repeated
    tail; a row that ran its whole budget is read at its last step; a row
    converged within one cycle has no step; a row without a trace is read
    on the census's own step."""
    tool = _load("make_census_figure")
    frozen = [-1.0, -1.1, -1.1000005, -1.1000005, -1.1000005]
    assert tool.last_step({"cycles_run": 3, "energy_trace": frozen,
                           "last_delta_e": 0.0}) == pytest.approx(5e-7)
    assert tool.last_step({"cycles_run": 5, "energy_trace": frozen, "last_delta_e": 0.0}) == 0.0
    assert tool.last_step({"cycles_run": 1, "energy_trace": frozen}) is None
    assert tool.last_step({"cycles_run": 9, "energy_trace": [],
                           "last_delta_e": 3e-4}) == pytest.approx(3e-4)
    cell = {"arch": "deep_3x16", "n_species": 1, "n_converged": 1, "error": None,
            "conv_tol": 1e-6, "freeze_on_convergence": True,
            "rows": [_trace_row("H", True, 3, frozen)]}
    path = tmp_path / "coldstart_census.json"
    path.write_text(json.dumps({"cells": [cell]}))
    (col,) = tool.columns([str(path)])
    (name, value, converged), = col["steps"]
    assert (name, converged) == ("H", True) and value == pytest.approx(5e-7)


def test_the_trace_tolerance_draws_the_line_on_a_census_without_conv_tol(
        tmp_path, saves, capsys):
    """Under --cycles-from-trace TOL the steps figure draws the given
    tolerance as every cell's line (the tolerance the marks are read at),
    whether or not the cell records a conv_tol; without the flag a cell
    without conv_tol draws no line and one with it draws its own."""
    tool = _load("make_census_figure")
    trace = [-1.0, -1.1, -1.10001, -1.100011]
    cells = [{"arch": "deep_3x16", "n_species": 1, "n_converged": 0, "error": None,
              "rows": [_trace_row("H", False, 4, trace)]},
             {"arch": "deep_geom_3x16", "n_species": 1, "n_converged": 0, "error": None,
              "conv_tol": 1e-8, "rows": [_trace_row("H", False, 4, trace)]}]
    path = tmp_path / "coldstart_census.json"
    path.write_text(json.dumps({"cells": cells}))
    out = tmp_path / "census.png"
    assert tool.main([str(path), "--out", str(out), "--cycles-from-trace", "1e-5"]) == 0
    capsys.readouterr()
    by_name = {Path(p).name: snap for p, snap in saves}
    steps = by_name["census_steps.png"]["axes"][0]
    tol_lines = sorted((round(float(np.mean(ln["x"])), 3), float(ln["y"][0]))
                       for ln in steps["lines"] if ln["x"].size == 2)
    assert tol_lines == [(0.0, 1e-5), (1.0, 1e-5)]
    assert [c["step_tol"] for c in tool.columns([str(path)], tol=1e-5)] == [1e-5, 1e-5]
    assert [c["step_tol"] for c in tool.columns([str(path)])] == [None, 1e-8]


def test_the_companion_counts_the_species_without_a_step(tmp_path, saves, capsys):
    """A species converged within one cycle (no step in its trace) or with
    no finite trace draws no marker in the companion: the column counts
    such species above itself and the manifest names them."""
    tool = _load("make_census_figure")
    cells = [{"arch": "deep_3x16", "n_species": 3, "n_converged": 2, "error": None,
              "conv_tol": 1e-6, "freeze_on_convergence": True,
              "rows": [_trace_row("H", True, 1, [-1.0, -1.0, -1.0]),
                       _trace_row("Li", True, 3, [-7.0, -7.4, -7.4000004, -7.4000004]),
                       {"name": "Be", "converged": False, "cycles_run": 4, "total_energy": None,
                        "finite": False, "last_delta_e": None,
                        "energy_trace": [-14.0, None, None, None], "wall_s": 1.0}]}]
    path = tmp_path / "coldstart_census.json"
    path.write_text(json.dumps({"cells": cells}))
    out = tmp_path / "census.png"
    assert tool.main([str(path), "--out", str(out)]) == 0
    manifest = _printed_json(capsys.readouterr().out)
    (col,) = manifest["steps"]["columns"]
    assert col["without_step"] == ["H", "Be"]
    assert col["annotation"] == "2 without a step"
    assert [name for name, _v, _c in col["steps"]] == ["Li"]
    by_name = {Path(p).name: snap for p, snap in saves}
    steps = by_name["census_steps.png"]["axes"][0]
    assert sum(len(c["offsets"]) for c in steps["collections"]) == 1
    assert [t["text"] for t in steps["texts"]] == ["2 without a step"]
