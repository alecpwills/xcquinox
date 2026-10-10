"""The cold-start convergence census as a figure.

The preflight's census (``xcquinox.pipeline.cluster.coldstart_census``) runs
the training solver of every cell from the atomic guess on each training
species and records, per species, whether it converged, the cycles it ran
and its energy trace. This figure draws one column per architecture of each
census file given: the converged species as filled markers at their cycle
count, the unconverged as open markers crossed out at theirs, and the count
``converged/species`` above the column; a cell the census could not measure
is annotated with its refusal. The columns follow the architecture order and
colours of ``arch_style``; with several census files each file's label
(``--labels``, or the file's run directory name) prefixes the column.

The cycle count is the census's own (``converged``, ``cycles_run``) unless
the solver ran its whole budget without freezing on convergence, when every
row reports the budget: then the count is read off the energy trace as the
first cycle whose energy step falls below the tolerance, a row whose trace
never does being drawn as unconverged. That reading is used with
``--cycles-from-trace TOL`` and, by itself, for a cell that records
``freeze_on_convergence`` false with its ``conv_tol``. A companion figure
``<stem>_steps.png`` draws the energy step of every species' last cycle run
on a logarithmic axis, read at the row's cycle count (a solver that froze on
convergence is read at the convergence cycle, not on the repeated tail of
its trace), filled for a species that converged and open for one that did
not, with the tolerance the marks are read at as a short line (the one
given, else the cell's where the census records it). Two title lines
explain each figure's marks; ``--plain`` writes second files without them.

Usage::

    python tools/analysis/make_census_figure.py <census.json> [<census.json> ...]
        [--labels a,b] [--cycles-from-trace TOL] [--out PATH] [--plain]

The PNG is written as ``coldstart_census.png`` beside the first census unless
``--out`` says otherwise, the companion as ``<stem>_steps.png``; the render
manifest (one entry per column with its label, architecture, colour, the
converged cycle counts, the unconverged species, the annotation, how the
cycles were read and, under ``steps``, the companion's columns) is printed.
A cell the census could not measure carries its whole refusal in the
manifest and its first thirty characters in the figure.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arch_style import arch_color, as_shown, order_present  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_NAME = "coldstart_census.png"
UNKNOWN_ARCH = "unknown"
TITLE_LINES = (
    "Cold-start convergence census: the training solver of each cell from the atomic "
    "guess on every training species, one column per architecture",
    "a filled mark is a species that converged, at the cycles it took; an open crossed "
    "mark one that did not, at the cycles it ran; the count above the column is "
    "converged/species",
)
STEPS_TITLE_LINES = (
    "Cold-start convergence census: the last energy step of every species at the end of "
    "its run, one column per architecture, on a logarithmic axis",
    "a filled mark is a species that converged, an open mark one that did not, each at the "
    "energy step of its last cycle run; the short line is the tolerance the marks are read "
    "at; a species without a step (converged within one cycle, or no finite trace) is "
    "counted above its column",
)
#: the figure width per column; with the side margins in inches the column
#: pitch tends to COLUMN_WIDTH_IN from above as the count grows (1.63 in at
#: six columns, 4.2 in at one)
COLUMN_WIDTH_IN = 1.6
#: the side margins in inches, the companion's left one wider for its
#: logarithmic tick labels, so that the y label stays inside the figure at
#: the minimum width of 6 in
LEFT_MARGIN_IN = 0.6
STEPS_LEFT_MARGIN_IN = 0.85
RIGHT_MARGIN_IN = 0.15
#: the characters of a refusal shown under ``not measured`` in a column's
#: annotation (the whole text is in the manifest), in a monospace face at
#: 7 pt (0.602 em per character, 1.29 in for 22), so that two annotations
#: of neighbouring columns stay clear of each other at every column count
ANNOTATION_CHARS = 22
#: the top fraction of the companion's axis kept clear of marks, where its
#: annotations (a refusal, the count without a step) sit above the column's
#: largest step
ANNOTATION_BAND = 0.16
JITTER = 0.16


def _figsize(n_columns: int) -> tuple:
    return (max(6.0, COLUMN_WIDTH_IN * n_columns + 1.5), 4.8)


def cycles_to_convergence(row: dict, tol: float):
    """The first cycle at which the row's energy step falls below ``tol``
    (the cycles run until then, one-based), or None when its trace never
    does or holds a non-finite value."""
    trace = row.get("energy_trace") or []
    for k in range(1, len(trace)):
        a, b = trace[k - 1], trace[k]
        if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
            return None
        if abs(b - a) < tol:
            return k + 1
    return None


def read_census(path) -> list:
    """The cells of one census file, each with the file's path."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [{**cell, "path": str(path)} for cell in payload.get("cells", [])]


def _marks(cell: dict, tol) -> tuple:
    """``(converged_cycles, unconverged, mode)`` of a cell: the census's own
    fields, or the trace reading at ``tol`` (given, or the cell's own
    tolerance when it ran without freezing on convergence); under the trace
    reading a row the census flags converged keeps its flag, at the trace's
    crossing or at its cycle count when the trace shows none (the census's
    criterion is its own)."""
    rows = cell.get("rows") or []
    if tol is None and cell.get("freeze_on_convergence") is False \
            and cell.get("conv_tol") is not None:
        tol = float(cell["conv_tol"])
    if tol is None:
        converged = [int(r["cycles_run"]) for r in rows if r.get("converged")]
        unconverged = [(r["name"], int(r["cycles_run"])) for r in rows
                       if not r.get("converged")]
        return converged, unconverged, "census"
    converged, unconverged = [], []
    for r in rows:
        cycles = cycles_to_convergence(r, tol)
        if cycles is None and r.get("converged"):
            cycles = int(r["cycles_run"])
        if cycles is None:
            unconverged.append((r["name"], int(r["cycles_run"])))
        else:
            converged.append(cycles)
    return converged, unconverged, f"trace below {tol:g}"


def last_step(row: dict):
    """The energy step of a row's last cycle run, ``|E_c - E_c-1|`` at its
    cycle count ``c`` (``cycles_run``) from its trace, so a solver that
    froze on convergence is read at the convergence cycle and not on the
    repeated tail of its trace; the census's own ``last_delta_e`` when the
    trace is shorter than the cycle count (a result without a trace); None
    below two cycles run or when neither energy is finite."""
    trace = row.get("energy_trace") or []
    cycles = row.get("cycles_run")
    if isinstance(cycles, int) and 0 <= cycles <= len(trace):
        if cycles < 2:
            return None
        trace = trace[:cycles]
    if len(trace) >= 2 and all(isinstance(v, (int, float)) and math.isfinite(v)
                               for v in trace[-2:]):
        return abs(float(trace[-1]) - float(trace[-2]))
    value = row.get("last_delta_e")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def columns(census_paths, labels=None, tol=None) -> list:
    """The columns in drawing order: per file (in the order given) the
    architectures in ``arch_style`` order, every cell of an architecture in
    file order (a repeated architecture's columns named by their specs);
    each column carries the file's label, the architecture, the converged
    cycle counts, the unconverged species with their cycles, the counts,
    how the cycles were read, the last energy step of every species with
    its convergence, the cell's tolerance and the error if any."""
    paths = [str(p) for p in census_paths]
    if labels is not None and len(labels) != len(paths):
        raise ValueError(f"{len(labels)} labels for {len(paths)} census files")
    out = []
    for i, path in enumerate(paths):
        label = labels[i] if labels else Path(path).resolve().parent.name
        by_arch: dict = {}
        for k, cell in enumerate(read_census(path)):
            by_arch.setdefault(cell.get("arch") or UNKNOWN_ARCH, []).append((k, cell))
        known = order_present([a for a in by_arch if a != UNKNOWN_ARCH])
        for arch in known + ([UNKNOWN_ARCH] if UNKNOWN_ARCH in by_arch else []):
            repeated = len(by_arch[arch]) > 1
            for k, cell in by_arch[arch]:
                converged, unconverged, mode = _marks(cell, tol)
                rows = cell.get("rows") or []
                converged_names = {r["name"] for r in rows} - {n for n, _c in unconverged}
                out.append({
                    "label": label, "arch": arch,
                    "cell": (Path(cell.get("spec_path") or f"cell{k}").stem
                             if repeated else ""),
                    "color": arch_color(arch) if arch != UNKNOWN_ARCH else "#000000",
                    "converged_cycles": converged, "unconverged": unconverged,
                    "n_converged": len(converged) if mode != "census"
                    else int(cell.get("n_converged", len(converged))),
                    "n_species": int(cell.get("n_species", len(rows))),
                    "cycles_read": mode,
                    "steps": [(r["name"], last_step(r), r["name"] in converged_names)
                              for r in rows],
                    "conv_tol": cell.get("conv_tol"),
                    # the tolerance the marks are read at: the one given,
                    # else the cell's own where the census records it
                    "step_tol": float(tol) if tol is not None else cell.get("conv_tol"),
                    "error": cell.get("error")})
    return out


def _jitter(n: int) -> list:
    if n <= 1:
        return [0.0] * n
    return [-JITTER + 2.0 * JITTER * k / (n - 1) for k in range(n)]


def _refusal_text(error) -> str:
    """The two-line annotation of a cell the census could not measure: the
    first ``ANNOTATION_CHARS`` characters of its refusal under ``not
    measured``."""
    short = " ".join(str(error).split())
    if len(short) > ANNOTATION_CHARS:
        short = short[:ANNOTATION_CHARS - 3] + "..."
    return "not measured\n" + short


def plot_census(cols: list, out_path, plain: bool = False, several: bool = False) -> dict:
    """Draw the columns and return the render manifest."""
    fig, ax = plt.subplots(figsize=_figsize(len(cols)))
    top = max([c for col in cols for c in col["converged_cycles"]]
              + [c for col in cols for _n, c in col["unconverged"]] + [1])
    manifest = []
    for x, col in enumerate(cols):
        text = (_refusal_text(col["error"]) if col["error"]
                else f"{col['n_converged']}/{col['n_species']}")
        if not col["error"]:
            jitter = _jitter(len(col["converged_cycles"]))
            if col["converged_cycles"]:
                ax.scatter([x + j for j in jitter], col["converged_cycles"], s=18,
                           color=col["color"], edgecolor="white", linewidth=0.3, zorder=3)
            jitter = _jitter(len(col["unconverged"]))
            if col["unconverged"]:
                ys = [c for _n, c in col["unconverged"]]
                xs = [x + j for j in jitter]
                ax.scatter(xs, ys, s=30, facecolors="none", edgecolors=col["color"],
                           linewidth=0.9, zorder=3)
                ax.scatter(xs, ys, s=30, marker="x", color=col["color"], linewidth=0.9,
                           zorder=4)
        ax.annotate(text, (x, top * 1.04), ha="center", va="bottom", fontsize=7,
                    family="monospace", color="black")
        manifest.append({**col, "x": x, "annotation": text})
    _ticks(ax, cols, several)
    ax.set_ylim(0, top * 1.14)
    ax.set_ylabel("SCF cycles from the atomic guess")
    _finish(fig, ax, plain, TITLE_LINES, LEFT_MARGIN_IN)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return {"out": str(out_path), "plain": bool(plain), "columns": manifest,
            "title_lines": [] if plain else list(TITLE_LINES)}


def _ticks(ax, cols: list, several: bool) -> None:
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([(f"{col['label']} " if several else "")
                        + (as_shown(col["arch"]) if col["arch"] != UNKNOWN_ARCH
                           else UNKNOWN_ARCH)
                        + (f" {col['cell']}" if col.get("cell") else "")
                        for col in cols], rotation=20, ha="right", fontsize=8)
    ax.set_xlim(-0.6, len(cols) - 0.4)


def _finish(fig, ax, plain: bool, title_lines, left_in: float) -> None:
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    width = fig.get_figwidth()
    left, right = left_in / width, 1.0 - RIGHT_MARGIN_IN / width
    if not plain:
        fig.suptitle("\n".join(title_lines), fontsize=8, y=0.995, va="top", wrap=True)
        fig.subplots_adjust(top=0.82, bottom=0.22, left=left, right=right)
    else:
        fig.subplots_adjust(top=0.95, bottom=0.22, left=left, right=right)


def plot_census_steps(cols: list, out_path, plain: bool = False,
                      several: bool = False) -> dict:
    """The companion figure: the energy step of every species' last cycle
    run on a logarithmic axis, one column per cell, filled for a species
    that converged and open for one that did not, with the tolerance the
    marks are read at as a short line (the one given to
    ``--cycles-from-trace``, else the cell's where the census records it);
    a species without a step (converged within one cycle, or no finite
    trace) is counted above its column, a refused cell annotated as in the
    cycles figure. A census whose solver ran its whole budget on every
    species (every cycle count the budget) shows its depth of convergence
    here."""
    fig, ax = plt.subplots(figsize=_figsize(len(cols)))
    manifest = []
    floor = 1e-12
    for x, col in enumerate(cols):
        without = [name for name, value, _c in col.get("steps", []) if value is None]
        steps = [(name, value, converged) for name, value, converged in col.get("steps", [])
                 if value is not None]
        jitter = _jitter(len(steps))
        for (name, value, converged), j in zip(steps, jitter):
            value = max(float(value), floor)
            if converged:
                ax.scatter([x + j], [value], s=18, color=col["color"], edgecolor="white",
                           linewidth=0.3, zorder=3)
            else:
                ax.scatter([x + j], [value], s=30, facecolors="none",
                           edgecolors=col["color"], linewidth=0.9, zorder=3)
        if col.get("step_tol") is not None and not col["error"]:
            ax.plot([x - 0.3, x + 0.3], [float(col["step_tol"])] * 2, color="black",
                    linewidth=0.9, zorder=2)
        text = (_refusal_text(col["error"]) if col["error"]
                else f"{len(without)} without a step" if without else "")
        if text:
            ax.annotate(text, (x, 0.98), xycoords=("data", "axes fraction"), ha="center",
                        va="top", fontsize=7, family="monospace", color="black")
        manifest.append({"label": col["label"], "arch": col["arch"], "cell": col.get("cell", ""),
                         "x": x, "steps": steps, "without_step": without,
                         "step_tol": col.get("step_tol"), "annotation": text,
                         "error": col["error"]})
    ax.set_yscale("log")
    drawn = [max(float(v), floor) for col in manifest for _n, v, _c in col["steps"]]
    drawn += [float(col["step_tol"]) for col in manifest
              if col["step_tol"] is not None and not col["error"]]
    if drawn:
        # the marks within the lower part of the axis, the band above them
        # for the annotations; the range a decade at least, centred on the
        # marks, so that the axis carries major tick labels only (below a
        # decade the minor ticks are labelled, and the wider labels push
        # the y label off the figure)
        lo, hi = math.log10(min(drawn)), math.log10(max(drawn))
        if hi - lo < 1.0:
            lo, hi = 0.5 * (lo + hi) - 0.5, 0.5 * (lo + hi) + 0.5
        extent = hi - lo
        bottom, top = lo - 0.06 * extent, hi + 0.06 * extent
        top += (top - bottom) * ANNOTATION_BAND / (1.0 - ANNOTATION_BAND)
        ax.set_ylim(10.0 ** bottom, 10.0 ** top)
    _ticks(ax, cols, several)
    ax.set_ylabel("last energy step |E_n - E_n-1| (Ha)")
    _finish(fig, ax, plain, STEPS_TITLE_LINES, STEPS_LEFT_MARGIN_IN)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return {"out": str(out_path), "plain": bool(plain), "columns": manifest,
            "title_lines": [] if plain else list(STEPS_TITLE_LINES)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("census", nargs="+")
    parser.add_argument("--labels", default=None,
                        help="comma-separated labels, one per census file")
    parser.add_argument("--cycles-from-trace", type=float, default=None, metavar="TOL",
                        help="read the cycles to convergence off each row's energy trace "
                             "at this tolerance (a census run without freezing on "
                             "convergence)")
    parser.add_argument("--out", default=None,
                        help="output PNG (default: %s beside the first census)" % OUT_NAME)
    parser.add_argument("--plain", action="store_true",
                        help="write a second copy without the title lines")
    args = parser.parse_args(argv)
    labels = ([s.strip() for s in args.labels.split(",")] if args.labels else None)
    cols = columns(args.census, labels, args.cycles_from_trace)
    out = Path(args.out) if args.out else Path(args.census[0]).resolve().parent / OUT_NAME
    several = len(args.census) > 1
    manifest = plot_census(cols, out, several=several)
    steps_out = out.with_name(out.stem + "_steps" + out.suffix)
    manifest["steps"] = plot_census_steps(cols, steps_out, several=several)
    if args.plain:
        plain_out = out.with_name(out.stem + "_plain" + out.suffix)
        manifest["out_plain"] = plot_census(cols, plain_out, plain=True, several=several)["out"]
        steps_plain = out.with_name(out.stem + "_steps_plain" + out.suffix)
        manifest["steps"]["out_plain"] = plot_census_steps(
            cols, steps_plain, plain=True, several=several)["out"]
    print(json.dumps(manifest, indent=2))
    print(f"written {out} and {steps_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
