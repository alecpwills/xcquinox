#!/usr/bin/env python
"""Assemble the 3x3 constraint/pretraining comparison figure.

Rows are three runs of ``constraint_pretrain_gmtkn55_demo.py``:
  row 1 = 150-step pretrain, unpolarized correlation baseline
  row 2 = 1000-step pretrain, unpolarized
  row 3 = 1000-step pretrain, spin-polarized PW92c baseline
Columns are the demo's three metrics (BH76 reaction-energy MAE, per-species
|E_nn - E_pbe| MAE, GMTKN55 W4-11 atomization-energy MAE).

This does NO recomputation — it parses the three demo run-logs' printed metric
tables, so it's cheap and reproducible from the captured runs. Each table block
in a demo log looks like::

    ============== ...
    BH76 reaction-energy MAE vs GMTKN55-BH76RC (kcal/mol)  (lower is better)
    ============== ...
    level               random mean   random max  random std   pretrained
    -------------- ...
    PBE baseline               8.08                                 (n/a)
    unconstrained             21.37        30.79        5.84        (n/a)
    +LO(x)                    20.94        25.18        2.63        10.55
    ...
    ============== ...

Usage::

    python notebooks/analysis/make_constraint_3x3.py LOG_150 LOG_1000 LOG_1000_POLC [OUT.png]
"""
from __future__ import annotations

import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# (substring identifying the table title, results key, panel ylabel, has PBE baseline)
METRICS = [
    ("BH76 reaction-energy", "bh76",
     "BH76 reaction-energy MAE\nvs GMTKN55-BH76RC (kcal/mol)", True),
    ("per-species", "pbe_dev",
     "per-species |E_nn - E_pbe| MAE\n(kcal/mol; deviation from PBE)", False),
    ("atomization-energy", "w411",
     "atomization-energy MAE\nvs GMTKN55 W4-11 (kcal/mol)", True),
]
_METRIC_KEYS = [m[1] for m in METRICS]


def parse_demo_log(path):
    """Parse a demo run-log into ``{pbe, levels, rand, pre}``.

    - ``pbe[metric]`` -> float (PBE baseline, where present)
    - ``levels`` -> ordered list of level labels (e.g. 'unconstrained', '+LO(x)')
    - ``rand[level][metric]`` -> {mean, max, std}
    - ``pre[level][metric]``  -> float or None  (None for the '(n/a)' baseline level)
    """
    res = {"pbe": {}, "levels": [], "rand": {}, "pre": {}}
    cur = None        # current metric key
    in_table = False  # past the 'level ...' header, reading data rows
    with open(path) as f:
        lines = f.read().splitlines()
    for line in lines:
        if "(lower is better)" in line:
            cur = next((k for sub, k, _, _ in METRICS if sub in line), None)
            in_table = False
            continue
        if cur and line.strip().startswith("level"):
            in_table = True
            continue
        if not in_table:
            continue
        s = line.strip()
        if not s:
            continue
        if set(s) <= set("-"):     # the '---' rule under the header
            continue
        if set(s) <= set("="):     # closing rule ends the table
            in_table = False
            cur = None
            continue
        parts = re.split(r"\s{2,}", s)
        if len(parts) < 2:
            continue
        label, vals = parts[0], parts[1:]
        if label == "PBE baseline":
            res["pbe"][cur] = float(vals[0])
            continue
        if label not in res["levels"]:
            res["levels"].append(label)
        res["rand"].setdefault(label, {})[cur] = {
            "mean": float(vals[0]), "max": float(vals[1]), "std": float(vals[2]),
        }
        pre = vals[3] if len(vals) > 3 else "(n/a)"
        res["pre"].setdefault(label, {})[cur] = None if pre == "(n/a)" else float(pre)
    # sanity: every level must have all three metrics
    for lvl in res["levels"]:
        missing = set(_METRIC_KEYS) - set(res["rand"].get(lvl, {}))
        if missing:
            raise ValueError(f"{path}: level {lvl!r} missing metrics {sorted(missing)}")
    return res


_STYLE = {
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
    "axes.axisbelow": True, "figure.dpi": 120, "savefig.dpi": 150,
    "savefig.bbox": "tight",
}
_RANDOM = "#c0504d"      # random-init bars
_WORST = "#9a9a9a"       # worst-seed whisker (faint)
_STD = "#5a1714"         # +/- std whisker (bold, dark)
_PRETRAINED = "#4f81bd"  # pretrained bars
_BARW = 0.38


def _draw_random(ax, x, means, maxes, stds, *, first):
    """Random bars with TWO whiskers: a faint upper cap at the worst seed and a
    bold symmetric +/- std error bar."""
    ax.bar(x, means, _BARW, color=_RANDOM, zorder=2,
           label="random init (mean)" if first else None)
    # worst-seed reach: faint, upper-only whisker from mean -> max.
    ax.errorbar(x, means, yerr=[np.zeros_like(means), np.maximum(maxes - means, 0)],
                fmt="none", ecolor=_WORST, elinewidth=1.0, capsize=6, capthick=1.0,
                zorder=4, label="worst seed" if first else None)
    # +/- std across seeds: bold, symmetric.
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=_STD, elinewidth=1.8,
                capsize=3, capthick=1.6, zorder=5,
                label="± std (seeds)" if first else None)


def plot_3x3(configs, out_path):
    """``configs`` = list of (row_label, parsed_results). Renders rows x metrics."""
    n = len(configs)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(n, 3, figsize=(16, 4.2 * n), squeeze=False)
        for r, (row_label, res) in enumerate(configs):
            labels = res["levels"]
            x = np.arange(len(labels))
            first_panel = (r == 0)
            for c, (_, key, ylab, has_pbe) in enumerate(METRICS):
                ax = axes[r][c]
                means = np.array([res["rand"][l][key]["mean"] for l in labels])
                maxes = np.array([res["rand"][l][key]["max"] for l in labels])
                stds = np.array([res["rand"][l][key]["std"] for l in labels])
                first = first_panel and c == 0
                _draw_random(ax, x - _BARW / 2, means, maxes, stds, first=first)
                pre = [res["pre"][l][key] for l in labels]
                xp = [xi for xi, v in zip(x, pre) if v is not None]
                yp = [v for v in pre if v is not None]
                ax.bar(np.array(xp) + _BARW / 2, yp, _BARW, color=_PRETRAINED,
                       zorder=3, label="pretrained" if first else None)
                if has_pbe and key in res["pbe"]:
                    ax.axhline(res["pbe"][key], ls="--", color="k", lw=1.0,
                               label=f"PBE ({res['pbe'][key]:.1f})" if first else None)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=18, ha="right")
                ax.grid(axis="y", alpha=0.3)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                ax.margins(y=0.08)
                if r == 0:
                    ax.set_title(ylab)
                if c == 0:
                    ax.set_ylabel(f"{row_label}\n\nMAE (kcal/mol)")
                if first:
                    ax.legend(loc="upper left", framealpha=0.9)
        fig.suptitle(
            "Physical constraints + pretraining vs GMTKN55 — "
            "150-step → 1000-step → 1000-step + spin-polarized PW92c baseline",
            fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_path)
        plt.close(fig)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    log_150, log_1000, log_polc = argv[0], argv[1], argv[2]
    out = argv[3] if len(argv) > 3 else \
        "notebooks/analysis/constraint_pretrain_gmtkn55_demo_3x3.png"
    configs = [
        ("150-step pretrain\n(unpolarized)", parse_demo_log(log_150)),
        ("1000-step pretrain\n(unpolarized)", parse_demo_log(log_1000)),
        ("1000-step pretrain\n(spin-polarized PW92c)", parse_demo_log(log_polc)),
    ]
    plot_3x3(configs, out)
    print(f"3x3 figure -> {out}")


if __name__ == "__main__":
    main()
