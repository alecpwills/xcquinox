#!/usr/bin/env python
"""Certificate summary across architectures and campaign generations.

Reads every ``<run_dir>/pretrain/<arch>/fidelity_certificate.json`` under the
given run directories and draws, per (label, arch), the mean |dAE| against the
parent functional as a bar with the per-species max |dAE| as a marker above
it, so a reader sees the set-level cloning fidelity and the worst species on
one axis, against the certificate gates. The gate lines are read from the
certificates' own recorded tolerances, never hard-coded: the ``tol_AE`` line
is labeled with the recorded aggregate (``mae`` gates the MEAN at that value;
``max`` gates every species), and a ``tol_AE_max_backstop`` line is drawn
when any certificate records one. FAIL verdicts hatch their bar; species
above 1.0 kcal/mol are printed above it.

Statistics are recomputed here from ``per_atomization`` (rows with a null
``dAE_kcalmol`` skipped), so certificates written before the summary carried
``mean_dAE_kcalmol`` / ``rmse_dAE_kcalmol`` / ``species_over_1_kcalmol``
plot identically to regated ones; a ``regate`` provenance block, when
present, is ignored beyond not being an error. A CSV with the same numbers
is written beside the PNG (same basename).

The y axis is linear and capped a little above the tallest gate so the gate
region stays readable next to multi-kcal/mol legacy outliers; a value beyond
the cap is drawn as an up-pointing marker at the axis edge with the number
printed beside it.

Usage:
    python notebooks/analysis/plot_certificate_summary.py \\
        --runs v7=<run dir> [--runs v7=<second run dir> ...] \\
        --runs legacy=<run dir> --out <figure.png>

A label may repeat (two groups of one campaign merge under one label); a
duplicate (label, arch) pair is refused, as is a run directory holding no
certificate at all.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

# Categorical palette: slot 1 blue, slot 2 orange, then aqua/yellow for
# further labels. Color follows the LABEL (the campaign generation); the FAIL
# state is carried by hatching and the verdict text, never by color alone.
# Separation was checked by execution with an OKLab-based colorblind
# validator (Delta E x100 in OKLab, light surface #fcfcfb): worst adjacent
# pair 24.7 protan / 33.6 normal for the first two slots. Note the METRIC:
# under CIEDE2000 with the Vienot 1999 protan model the same pair measures
# ~48.5 normal / ~57.3 protan -- different formulations, both comfortably
# above their guidelines; any quoted number must name its metric.
_LABEL_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

# The per-species flag threshold the certificates record (kcal/mol; the
# original per-species gate). Stated here only as a fallback for certificates
# written before the summary carried the species list.
_SPECIES_FLAG_KCALMOL = 1.0


def collect_certificates(runs):
    """``[(label, arch, record)]`` for every certificate under ``runs``.

    ``runs`` is a list of ``(label, run_dir)`` pairs. Each record carries the
    recomputed statistics plus the recorded verdict and tolerances. A
    duplicate (label, arch) pair and a run directory with no certificate are
    both refused: the first silently averages two campaigns into one bar, the
    second draws an empty axis that reads as a clean sweep.
    """
    out = []
    seen = set()
    for label, run_dir in runs:
        paths = sorted(glob.glob(
            os.path.join(run_dir, "pretrain", "*", "fidelity_certificate.json")))
        if not paths:
            raise ValueError(
                f"no pretrain/*/fidelity_certificate.json under {run_dir}")
        for path in paths:
            with open(path) as f:
                cert = json.load(f)
            dir_arch = os.path.basename(os.path.dirname(path))
            arch = str(cert.get("arch") or dir_arch)
            if cert.get("arch") and str(cert["arch"]) != dir_arch:
                raise ValueError(
                    f"certificate at {path} names arch {cert['arch']!r} but "
                    f"sits in directory {dir_arch!r}; a mislabeled "
                    "certificate must not be plotted under either name")
            key = (label, arch)
            if key in seen:
                raise ValueError(
                    f"duplicate certificate for label={label!r} arch={arch!r}"
                    f" (second copy at {path}); merge distinct groups under "
                    "one label only when their architecture sets are disjoint")
            seen.add(key)
            devs = [abs(float(r["dAE_kcalmol"]))
                    for r in cert.get("per_atomization", [])
                    if isinstance(r, dict)
                    and r.get("dAE_kcalmol") is not None]
            names = [(str(r.get("name")), abs(float(r["dAE_kcalmol"])))
                     for r in cert.get("per_atomization", [])
                     if isinstance(r, dict)
                     and r.get("dAE_kcalmol") is not None]
            tol = cert.get("tolerances") or {}
            summary = cert.get("summary") or {}
            species = summary.get("species_over_1_kcalmol")
            if species is None:
                species = [n for n, v in names if v > _SPECIES_FLAG_KCALMOL]
            out.append((label, arch, {
                "verdict": str(cert.get("verdict")),
                "n": len(devs),
                "mean": (sum(devs) / len(devs)) if devs else None,
                "rmse": (math.sqrt(sum(v * v for v in devs) / len(devs))
                         if devs else None),
                "max": max(devs) if devs else None,
                "species_over": list(species),
                "tol_AE": tol.get("tol_AE"),
                "aggregate": tol.get("tol_AE_aggregate", "max"),
                "backstop": tol.get("tol_AE_max_backstop"),
            }))
    return out


def write_csv(records, path):
    """The figure's numbers as one row per (label, arch)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "arch", "verdict", "n_atomizations",
                    "mean_abs_dAE_kcalmol", "rmse_dAE_kcalmol",
                    "max_abs_dAE_kcalmol", "species_over_1_kcalmol",
                    "tol_AE", "tol_AE_aggregate", "tol_AE_max_backstop"])
        for label, arch, r in records:
            w.writerow([label, arch, r["verdict"], r["n"],
                        r["mean"], r["rmse"], r["max"],
                        ";".join(r["species_over"]),
                        r["tol_AE"], r["aggregate"], r["backstop"]])


def plot_certificate_summary(records, out_path):
    """Render the grouped mean-bar / max-marker figure to ``out_path``.

    Returns the render manifest -- what was actually drawn (gate lines with
    their rule text, hatched FAIL bars, clipped max markers with their
    values, note texts, per-label colors, the y cap), built at the draw
    sites so tests pin drawn behaviour without parsing pixels.
    """
    labels = []
    for label, _arch, _r in records:
        if label not in labels:
            labels.append(label)
    archs = sorted({arch for _l, arch, _r in records})
    by_key = {(label, arch): r for label, arch, r in records}

    # One line per DISTINCT (kind, value): certificates recording different
    # aggregates at the same tol_AE (a partially regated pull) merge into a
    # single caption, instead of two annotations overprinting at one anchor.
    tol_lines: dict = {}
    for _l, _a, r in records:
        if r["tol_AE"] is not None:
            key = ("tol_AE", float(r["tol_AE"]))
            tol_lines.setdefault(key, set()).add(str(r["aggregate"]))
        if r["backstop"] is not None:
            tol_lines.setdefault(("backstop", float(r["backstop"])), set())
    gate_values = [v for _kind, v in tol_lines]
    finite_max = [r["max"] for _l, _a, r in records if r["max"] is not None]
    finite_mean = [r["mean"] for _l, _a, r in records if r["mean"] is not None]
    # The cap always covers every BAR (a clipped bar misstates its mean); only
    # the max MARKERS clip, drawn at the edge with their number. Sized a
    # little above the tallest gate so the gate region stays readable beside
    # multi-kcal/mol legacy outliers.
    y_cap = 2.5 * max(gate_values + [1.0])
    if finite_mean:
        y_cap = max(y_cap, 1.15 * max(finite_mean))
    if finite_max and max(finite_max) <= 1.6 * y_cap:
        y_cap = max(y_cap, 1.05 * max(finite_max))

    n_labels = max(len(labels), 1)
    group_w = 0.8
    bar_w = group_w / n_labels
    fig_w = max(7.5, 1.05 * len(archs) * n_labels + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))

    # The render manifest is built AT the draw sites and returned, so tests
    # can pin every drawn behaviour (gate lines and their rule text, FAIL
    # hatching, clipped markers with their values, note text, per-label
    # colors, the bar-covering cap) without parsing pixels.
    manifest = {"out_path": out_path, "y_cap": y_cap, "gate_lines": [],
                "hatched": [], "clipped": [], "colors": {}, "notes": {}}

    for li, label in enumerate(labels):
        color = _LABEL_COLORS[li % len(_LABEL_COLORS)]
        manifest["colors"][label] = color
        for ai, arch in enumerate(archs):
            r = by_key.get((label, arch))
            if r is None:
                continue
            if r["mean"] is None:
                # A certificate with no usable atomization rows still shows:
                # an unmarked gap would read as a clean absence.
                x = ai - group_w / 2 + (li + 0.5) * bar_w
                text = f"{r['verdict']}\nno atomization data"
                ax.annotate(text, (x, 0.0), xytext=(0, 6),
                            textcoords="offset points", ha="center",
                            fontsize=7, color="#444444", zorder=5)
                manifest["notes"][(label, arch)] = text
                continue
            x = ai - group_w / 2 + (li + 0.5) * bar_w
            hatch = "///" if r["verdict"] != "PASS" else None
            if hatch:
                manifest["hatched"].append((label, arch))
            ax.bar(x, r["mean"], width=0.92 * bar_w, color=color,
                   hatch=hatch, edgecolor="white", linewidth=0.5,
                   zorder=3)
            if r["max"] is not None:
                if r["max"] <= y_cap:
                    ax.plot([x, x], [r["mean"], r["max"]], color=color,
                            linewidth=1.0, alpha=0.55, zorder=3)
                    ax.plot([x], [r["max"]], marker="D", markersize=5,
                            color=color, markeredgecolor="white",
                            markeredgewidth=0.5, zorder=4)
                else:
                    ax.plot([x, x], [r["mean"], y_cap * 0.985], color=color,
                            linewidth=1.0, alpha=0.55, zorder=3)
                    ax.plot([x], [y_cap * 0.985], marker="^", markersize=7,
                            color=color, markeredgecolor="white",
                            markeredgewidth=0.5, zorder=4)
                    ax.annotate(f"{r['max']:.1f}", (x, y_cap * 0.985),
                                xytext=(0, -11), textcoords="offset points",
                                ha="center", fontsize=7, color="#444444",
                                zorder=5)
                    manifest["clipped"].append((label, arch, r["max"]))
            note = []
            if r["verdict"] != "PASS":
                note.append(r["verdict"])
            if r["species_over"]:
                # At most three species inline; the full list is in the CSV.
                shown = r["species_over"][:3]
                more = len(r["species_over"]) - len(shown)
                note.append(",".join(shown) + (f" +{more}" if more else ""))
            if note:
                y_note = min(r["max"] if r["max"] is not None else r["mean"],
                             y_cap * 0.985)
                # Near the top edge the text goes BELOW its anchor so it
                # cannot collide with the title band or a clipped-value
                # number.
                below = y_note > 0.86 * y_cap
                text = "\n".join(note)
                ax.annotate(text, (x, y_note),
                            xytext=(0, -22 if below else 6),
                            textcoords="offset points",
                            va="top" if below else "bottom",
                            ha="center", fontsize=7, color="#444444",
                            zorder=5)
                manifest["notes"][(label, arch)] = text

    for (kind, value), aggregates in sorted(tol_lines.items()):
        if kind == "tol_AE":
            parts = []
            if "mae" in aggregates:
                parts.append("mae: gates the set mean")
            if "max" in aggregates:
                parts.append("max: gates every species")
            text = f"tol_AE = {value:g} ({'; '.join(parts)})"
            style = dict(color="#555555", linestyle="--", linewidth=1.2)
        else:
            text = f"tol_AE_max_backstop = {value:g} (per-species ceiling)"
            style = dict(color="#555555", linestyle=":", linewidth=1.2)
        ax.axhline(value, zorder=2, **style)
        ax.annotate(text, (len(archs) - 0.52, value),
                    xytext=(0, 3), textcoords="offset points",
                    ha="right", fontsize=8, color="#555555")
        manifest["gate_lines"].append((value, text))

    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels(archs, rotation=20, ha="right", fontsize=9)
    ax.set_xlim(-0.6, len(archs) - 0.4)
    ax.set_ylim(0.0, y_cap)
    ax.set_ylabel("|dAE| vs parent (kcal/mol)")
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # Explicit Patch proxies: an empty ax.bar() call does not reliably carry
    # its facecolor into the legend handle.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=_LABEL_COLORS[i % len(_LABEL_COLORS)],
                     label=label) for i, label in enumerate(labels)]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9,
              title="pretraining round")
    ax.set_title("Cloning-fidelity certificates: mean |dAE| (bar) and "
                 "per-species max (marker) by architecture", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return manifest


def _parse_runs(values):
    runs = []
    for value in values:
        label, sep, run_dir = value.partition("=")
        if not sep or not label or not run_dir:
            raise ValueError(
                f"--runs takes LABEL=RUN_DIR, got {value!r}")
        runs.append((label, run_dir))
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", action="append", required=True,
                    metavar="LABEL=RUN_DIR",
                    help="a labeled run directory; repeat to add more (a "
                         "repeated label merges disjoint architecture sets)")
    ap.add_argument("--out", required=True, help="output PNG path")
    args = ap.parse_args(argv)

    records = collect_certificates(_parse_runs(args.runs))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    write_csv(records, csv_path)
    plot_certificate_summary(records, args.out)
    print(f"wrote {args.out}  ({len(records)} certificates)")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
