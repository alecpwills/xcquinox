"""The Slim16 figure: each held-out subset's error per network beside PBE-DF.

The subsets of the pool stand on the x axis, in name order, wrapped into
rows of at most ``SETS_PER_ROW``. Per subset there is one bar per evaluated
network, in manifest order, and then one PBE-DF bar; a bar's height is the
subset's mean absolute error of the reaction energies against the GMTKN55
references over every reaction the network reports, converged or not. Each
network's WTMAD-2 is the cloning paper's form with the FULL set's weights,
computed twice (``slim16_table.wtmad2_full_weights``): over all reactions as
reported, and over the converged reactions alone, where a subset with no
converged reaction is dropped from the sum and its weight is not given to the
others; a network with no converged reaction has no converged total, shown
as ``n/a``. The legend reads ``label (all / converged)``; PBE-DF, whose table
carries no convergence, has one number. The scale of the number is the full
set's own mean absolute reference, not the GMTKN55 constant. Where a subset
holds both converged and unconverged reactions, the converged-only error is
marked on the bar; the title names each network's unconverged reactions, the
subsets dropped from its converged total, the reactions it reports without an
energy (kept in the weights, in no mean) and the subsets whose references
average to zero (no weight in either total). Bars are coloured by
architecture (``arch_style.arch_color``) and hatched by network group, the
group being the label's leading token (``S``, ``slim05``, ``v7``); two
networks of one group and one architecture draw alike. A network that draws
no bar is named in the title with the reason: no evaluated channel, or a
channel with no reaction that carries a reference and a reported energy. The
row assembly is ``slim16_table``'s own, so the figure and the metrics CSV
state one set of numbers.

Usage::

    python tools/analysis/make_slim16_figure.py <local run dir> [--out PATH]

The PNG is written as ``slim16_wtmad2.png`` beside the run unless ``--out``
says otherwise (``--log`` draws the y axis logarithmically and adds ``_log``
to the name), and a render manifest (the bars and markers drawn, the legend
entries, the reaction count behind the weights, the absent networks, the
unconverged and unscored reactions, the dropped and unweighted subsets) is
printed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import slim16_table  # noqa: E402
from arch_style import arch_color  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT_NAME = "slim16_wtmad2.png"
#: subsets per row of axes; the pool's 37 subsets wrap into three rows
SETS_PER_ROW = 13
PBE_LABEL = "PBE-DF"
PBE_COLOR = "#7f7f7f"
#: one hatch per network group, in order of first appearance; the first
#: group is unhatched
HATCHES = ("", "////", "xxxx", "....", "\\\\\\\\", "++++")
#: inches of figure width per bar; the hatch stays legible at this width
INCH_PER_BAR = 0.15


def _finite(value) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def collect_set_errors(run_dir) -> tuple:
    """``(order, records, pbe)`` over a run: the network labels in manifest
    order; per label ``mae`` (``{subset: MAD}`` over every scored reaction),
    ``mae_converged`` (the same over the converged reactions, None for a
    subset with none), ``wtmad2_all`` (None when the network draws no bar),
    ``wtmad2_converged`` (None when no subset contributes), ``mixed`` (the
    subsets holding converged and unconverged scored reactions both),
    ``unconverged`` and ``unscored`` (reaction names), ``dropped`` (subsets
    out of the converged total), ``unweighted`` (subsets whose references
    average to zero), ``arch_name``, ``group`` (the label's leading token)
    and ``reason`` (None, ``no evaluated channel``, or ``no reaction with a
    reference and a reported energy``); and ``pbe``, PBE-DF's ``mae``,
    ``wtmad2``, ``n_reactions`` and ``unscored`` over every reaction its
    table covers. The networks that draw bars must carry one list of
    referenced reactions, so that the weights in the legend are one set;
    the collection refuses a run where they differ, naming the labels."""
    run_dir = Path(run_dir)
    manifest, width, pbe_df, subsets, _n_not_df = slim16_table.run_context(run_dir)
    order, records = [], {}
    pbe = {"mae": {}, "wtmad2": None, "n_reactions": 0, "unscored": []}
    referenced: dict = {}
    for network in manifest["networks"]:
        label = network.get("label")
        order.append(label)
        entry = {"mae": {}, "mae_converged": {}, "wtmad2_all": None,
                 "wtmad2_converged": None, "mixed": [], "unconverged": [],
                 "unscored": [], "dropped": [], "unweighted": [],
                 "n_reactions": 0, "arch_name": network.get("arch_name"),
                 "group": str(label).split("_", 1)[0], "reason": None}
        records[label] = entry
        found = slim16_table.channel_records(run_dir, width, int(network["index"]))
        if found is None:
            entry["reason"] = "no evaluated channel"
            continue
        molecules, reactions = found
        rows_nn, rows_pbe = slim16_table.reference_rows(
            molecules, reactions, pbe_df, subsets)
        scored = slim16_table.wtmad2_full_weights(rows_nn)
        if not _finite(scored["all"]):
            entry["reason"] = "no reaction with a reference and a reported energy"
            continue
        referenced[label] = sorted(row["name"] for row in rows_nn
                                   if slim16_table._finite(row["de_ref"]))
        entry["mae"] = {name: e["MAD_all"] for name, e in scored["per_subset"].items()
                        if e["MAD_all"] is not None}
        entry["mae_converged"] = {name: e["MAD_converged"]
                                  for name, e in scored["per_subset"].items()}
        entry["wtmad2_all"] = scored["all"]
        entry["wtmad2_converged"] = (scored["converged"]
                                     if _finite(scored["converged"]) else None)
        entry["mixed"] = [name for name, e in scored["per_subset"].items()
                          if 0 < e["n_converged"] < e["n_scored"]]
        entry["unconverged"] = scored["unconverged"]
        entry["unscored"] = scored["unscored"]
        entry["dropped"] = scored["dropped"]
        entry["unweighted"] = scored["unweighted"]
        entry["n_reactions"] = scored["n_reactions"]
        if pbe["wtmad2"] is None:
            scored_pbe = slim16_table.wtmad2_full_weights(rows_pbe)
            if _finite(scored_pbe["all"]):
                pbe = {"mae": {name: e["MAD_all"]
                               for name, e in scored_pbe["per_subset"].items()
                               if e["MAD_all"] is not None},
                       "wtmad2": scored_pbe["all"],
                       "n_reactions": scored_pbe["n_reactions"],
                       "unscored": scored_pbe["unscored"]}
    lists = list(referenced.items())
    differing = [label for label, names in lists[1:] if names != lists[0][1]]
    if differing:
        raise ValueError(
            "the evaluated channels do not carry one list of referenced reactions: "
            f"{', '.join(differing)} differ from {lists[0][0]}")
    return order, records, pbe


def _header_lines(evaluated: list, records: dict, absent: dict) -> list:
    """The title lines: what a bar and a mark are, then per network the
    unconverged reactions with the subsets dropped from its converged total,
    the reactions reported without an energy, the subsets without weight,
    and the networks that draw no bar with the reason."""
    lines = ["Slim16: mean absolute error per held-out set over every reported "
             "reaction; a black mark is the converged-only error of a set "
             "holding converged and unconverged reactions both"]
    unconverged = {label: records[label]["unconverged"] for label in evaluated
                   if records[label]["unconverged"]}
    dropped = {label: records[label]["dropped"] for label in evaluated
               if records[label]["dropped"]}
    unscored = {label: records[label]["unscored"] for label in evaluated
                if records[label]["unscored"]}
    unweighted = {label: records[label]["unweighted"] for label in evaluated
                  if records[label]["unweighted"]}
    if unconverged:
        lines.append("unconverged: " + "; ".join(
            f"{label}: {', '.join(names)}"
            + (f" (dropped from its converged total: {', '.join(dropped[label])})"
               if label in dropped else "")
            for label, names in unconverged.items()))
    if unscored:
        lines.append("reported without an energy (kept in the weights, in no "
                     "mean): " + "; ".join(f"{label}: {', '.join(names)}"
                                          for label, names in unscored.items()))
    if unweighted:
        lines.append("references averaging to zero, no weight in either total: "
                     + "; ".join(f"{label}: {', '.join(names)}"
                                 for label, names in unweighted.items()))
    if absent:
        lines.append("no bar for " + ", ".join(
            f"{label} ({reason})" for label, reason in absent.items()))
    return lines


def plot_slim16_set_errors(order: list, records: dict, pbe: dict,
                           out_path, log: bool = False) -> dict:
    """Draw the grouped bars and return the render manifest: the output
    path, the network order, the subsets on the axis, one entry per bar
    (label, subset, value, color, hatch; PBE-DF's under ``PBE-DF``), the
    markers (label, subset, the converged-only value), the legend texts, the
    reaction count the weights come from, the absent labels with their
    reasons, the unconverged and unscored reactions, the dropped and
    unweighted subsets per network, the y scale and the y cap. With ``log``
    the y axis is logarithmic from half the smallest drawn value, so the
    small bars read; a zero bar then draws nothing."""
    evaluated = [label for label in order if records[label]["wtmad2_all"] is not None]
    absent = {label: records[label]["reason"] for label in order
              if records[label]["wtmad2_all"] is None}
    unconverged = {label: records[label]["unconverged"] for label in evaluated
                   if records[label]["unconverged"]}
    unscored = {label: records[label]["unscored"] for label in evaluated
                if records[label]["unscored"]}
    dropped = {label: records[label]["dropped"] for label in evaluated
               if records[label]["dropped"]}
    unweighted = {label: records[label]["unweighted"] for label in evaluated
                  if records[label]["unweighted"]}
    subsets = sorted(set(pbe["mae"])
                     | {name for label in evaluated for name in records[label]["mae"]})
    groups: list = []
    for label in evaluated:
        if records[label]["group"] not in groups:
            groups.append(records[label]["group"])
    hatch_of = {name: HATCHES[i % len(HATCHES)] for i, name in enumerate(groups)}
    values = [v for label in evaluated for v in records[label]["mae"].values()]
    values += [records[label]["mae_converged"][name] for label in evaluated
               for name in records[label]["mixed"]]
    values += list(pbe["mae"].values())
    y_cap = 1.1 * max(values, default=1.0)
    positive = [v for v in values if v > 0]
    y_floor = 0.5 * min(positive) if log and positive else 0.0
    # the weights of the legend's numbers come from the networks' one list
    # of referenced reactions, PBE-DF's table when no network draws a bar
    n_reactions = (records[evaluated[0]]["n_reactions"] if evaluated
                   else pbe["n_reactions"])

    n_bars = len(evaluated) + 1
    group_w = 0.8
    bar_w = group_w / n_bars
    per_row = min(SETS_PER_ROW, max(len(subsets), 1))
    n_rows = max(1, math.ceil(len(subsets) / SETS_PER_ROW))
    fig_w = max(10.0, per_row * n_bars * INCH_PER_BAR + 0.8)
    n_handles = len(evaluated) + (1 if pbe["wtmad2"] is not None else 0)
    n_legend_rows = math.ceil(n_handles / 4) if n_handles else 0
    lines = _header_lines(evaluated, records, absent)
    # the header's height in inches: the legend's title and rows, and the
    # title lines as they wrap at about 16 characters per inch of width
    chars_per_line = max(40, int(fig_w * 16))
    n_lines = sum(math.ceil(len(line) / chars_per_line) for line in lines)
    header_h = 0.3 + 0.22 * (n_legend_rows + (1 if n_handles else 0)) + 0.17 * n_lines
    fig_h = 3.2 * n_rows + header_h + 0.5
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h), squeeze=False)
    bars, markers = [], []
    for row, ax in enumerate(axes[:, 0]):
        names = subsets[row * SETS_PER_ROW:(row + 1) * SETS_PER_ROW]
        for i, name in enumerate(names):
            for slot, label in enumerate(evaluated):
                record = records[label]
                value = record["mae"].get(name)
                if value is None:
                    continue
                x = i - group_w / 2 + (slot + 0.5) * bar_w
                color = arch_color(record["arch_name"])
                hatch = hatch_of[record["group"]]
                ax.bar(x, value, width=0.92 * bar_w, color=color,
                       hatch=hatch or None, edgecolor="white", linewidth=0.4)
                bars.append({"label": label, "subset": name, "value": value,
                             "color": color, "hatch": hatch})
                # a subset holding converged and unconverged reactions both:
                # the converged-only error is marked on the bar
                if name in record["mixed"]:
                    converged = record["mae_converged"][name]
                    ax.plot([x], [converged], marker="_", markersize=7,
                            markeredgewidth=1.2, color="black", linestyle="none",
                            zorder=4)
                    markers.append({"label": label, "subset": name,
                                    "value": converged})
            value = pbe["mae"].get(name)
            if value is not None:
                x = i - group_w / 2 + (n_bars - 0.5) * bar_w
                ax.bar(x, value, width=0.92 * bar_w, color=PBE_COLOR,
                       edgecolor="white", linewidth=0.4)
                bars.append({"label": PBE_LABEL, "subset": name, "value": value,
                             "color": PBE_COLOR, "hatch": ""})
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_xlim(-0.6, per_row - 0.4)
        if log:
            ax.set_yscale("log")
        ax.set_ylim(y_floor, y_cap)
        ax.set_ylabel("MAE vs reference (kcal/mol)", fontsize=9)
        ax.yaxis.grid(True, color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    def _converged_text(label):
        value = records[label]["wtmad2_converged"]
        return "n/a" if value is None else f"{value:.2f}"

    legend = [f"{label} ({records[label]['wtmad2_all']:.2f} / {_converged_text(label)})"
              for label in evaluated]
    handles = [Patch(facecolor=arch_color(records[label]["arch_name"]),
                     hatch=hatch_of[records[label]["group"]] or None,
                     edgecolor="white", label=text)
               for label, text in zip(evaluated, legend)]
    if pbe["wtmad2"] is not None:
        legend.append(f"{PBE_LABEL} ({pbe['wtmad2']:.2f})")
        handles.append(Patch(facecolor=PBE_COLOR, label=legend[-1]))
    if handles:
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.0 - 0.1 / fig_h), ncol=min(4, len(handles)),
                   frameon=False, fontsize=8,
                   title=("WTMAD-2 against the references with the full set's "
                          f"weights ({n_reactions} reactions), the paper's "
                          "form: all reactions as reported / converged reactions "
                          "only (kcal/mol)"),
                   title_fontsize=8)
    legend_h = 0.1 + 0.22 * (n_legend_rows + (1 if n_handles else 0))
    fig.suptitle("\n".join(lines), fontsize=8, y=1.0 - (legend_h + 0.08) / fig_h,
                 va="top", wrap=True)
    # the axes start under the header; a rotated tick row needs 0.65 in
    fig.subplots_adjust(top=1.0 - (header_h + 0.15) / fig_h, bottom=0.65 / fig_h,
                        left=0.75 / fig_w, right=1.0 - 0.15 / fig_w,
                        hspace=0.65 / (3.2 - 0.65))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return {"out": str(out_path), "order": list(order), "subsets": subsets,
            "bars": bars, "markers": markers, "legend": legend,
            "n_reactions": n_reactions, "absent": absent,
            "unconverged": unconverged, "unscored": unscored, "dropped": dropped,
            "unweighted": unweighted, "yscale": "log" if log else "linear",
            "y_floor": y_floor, "y_cap": y_cap}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--out", default=None,
                        help="output PNG (default: <run_dir>/%s)" % OUT_NAME)
    parser.add_argument("--log", action="store_true",
                        help="logarithmic y axis; the PNG name gains _log")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    out = Path(args.out) if args.out else run_dir / OUT_NAME
    if args.log:
        out = out.with_name(out.stem + "_log" + out.suffix)
    order, records, pbe = collect_set_errors(run_dir)
    manifest = plot_slim16_set_errors(order, records, pbe, out, log=args.log)
    print(json.dumps(manifest, indent=2))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
