"""The Slim16 figure: the paper's WTMAD-2 per network per subset.

One horizontal bar per network (manifest order, first at the top), its width
the network's total paper WTMAD-2 against PBE-DF and its segments that total
split by subset -- each subset's share of the sum of per-subset contributions,
so the segments add up to the number printed at the bar's end. A network that
draws no bar is listed in order with the reason in grey: no evaluated
channel, or a WTMAD-2 that is not finite on the run's PBE-DF footing (the
slice runs, whose PBE-DF table covers fewer species than their reactions
name). The row assembly (the reference leg PBE-DF, the converged-species
filter) is ``slim16_table``'s own, so the figure and the metrics CSV state
one set of numbers.

Usage::

    python tools/analysis/make_slim16_figure.py <local run dir> [--out PATH]

The PNG is written as ``slim16_wtmad2.png`` beside the run unless ``--out``
says otherwise, and a render manifest (the bars drawn and their colors) is
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
from xcquinox.pipeline.eval_holdout import paper_wtmad2  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT_NAME = "slim16_wtmad2.png"
MAX_TAB20 = 20  # tab20 holds 20 colors; the pool's 37 subsets cycle it


def collect_subset_wtmad2(run_dir) -> tuple:
    """``(order, records)`` over a run: the network labels in manifest order
    and, per label, the total converged-filter paper WTMAD-2 (kcal/mol, None
    when the network has no evaluated channel or the total is not finite) and
    each subset's share of the sum of per-subset contributions. A network
    that draws no bar carries the reason: ``no evaluated channel``, or
    ``WTMAD-2 not finite on this run's PBE-DF footing`` -- the slice runs,
    whose PBE-DF table covers fewer species than their reactions name."""
    run_dir = Path(run_dir)
    manifest, width, pbe_df, subsets, _n_not_df = slim16_table.run_context(run_dir)
    order, records = [], {}
    for network in manifest["networks"]:
        label = network.get("label")
        order.append(label)
        entry = {"total": None, "shares": {}, "kind": network.get("kind"),
                 "reason": None}
        records[label] = entry
        found = slim16_table.channel_records(run_dir, width, int(network["index"]))
        if found is None:
            entry["reason"] = "no evaluated channel"
            continue
        molecules, reactions = found
        _rows_all, rows_converged = slim16_table.reaction_rows(
            molecules, reactions, pbe_df, subsets)
        total, per_subset = paper_wtmad2(rows_converged)
        if not (isinstance(total, float) and math.isfinite(total)):
            entry["reason"] = "WTMAD-2 not finite on this run's PBE-DF footing"
            continue
        weight = sum(v["contribution"] for v in per_subset.values())
        entry["total"] = total
        if weight > 0:
            entry["shares"] = {name: v["contribution"] / weight
                               for name, v in per_subset.items()}
    return order, records


def plot_slim16_wtmad2(order: list, records: dict, out_path) -> dict:
    """Draw the stacked bars and return the render manifest: the output path,
    the network order, the sorted subset list, one entry per bar segment
    (label, subset, share, width, color), the totals printed, and the labels
    that drew no bar."""
    subsets = sorted({name for record in records.values()
                      for name in record["shares"]})
    colors = {name: i % MAX_TAB20 for i, name in enumerate(subsets)}
    cmap = matplotlib.colormaps["tab20"]
    bars, totals, absent = [], {}, []
    fig, ax = plt.subplots(figsize=(8.0, 0.55 * len(order) + 2.4))
    for row, label in enumerate(order):
        record = records[label]
        y = len(order) - 1 - row
        if record["total"] is None:
            absent.append(label)
            ax.text(0.0, y, record["reason"], color="0.55",
                    va="center", ha="left", fontsize=9)
            continue
        totals[label] = record["total"]
        left = 0.0
        for name in subsets:
            share = record["shares"].get(name)
            if not share:
                continue
            width = share * record["total"]
            color = cmap(colors[name])
            ax.barh(y, width, left=left, height=0.62, color=color,
                    edgecolor="white", linewidth=0.4)
            bars.append({"label": label, "subset": name, "share": share,
                         "width": width, "color": colors[name]})
            left += width
        ax.text(left, y, f" {record['total']:.2f}", va="center", ha="left",
                fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(list(reversed(order)))
    ax.set_xlabel("WTMAD-2 vs PBE-DF (kcal/mol)")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Patch(color=cmap(colors[name]), label=name) for name in subsets]
    if handles:
        ncol = min(6, len(handles))
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                  ncol=ncol, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return {"out": str(out_path), "order": list(order), "subsets": subsets,
            "bars": bars, "totals": totals, "absent": absent}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--out", default=None,
                        help="output PNG (default: <run_dir>/%s)" % OUT_NAME)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    out = Path(args.out) if args.out else run_dir / OUT_NAME
    order, records = collect_subset_wtmad2(run_dir)
    manifest = plot_slim16_wtmad2(order, records, out)
    print(json.dumps(manifest, indent=2))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
