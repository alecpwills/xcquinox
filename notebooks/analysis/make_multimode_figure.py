#!/usr/bin/env python
"""Figures for the multi-mode (self-consistency ladder) constraint/pretraining run.

Reads the JSON written by ``multimode_constraint_eval.py`` and renders:

  1. A 3x3 grid (rows = eval mode fixed-rho / one-shot / 3-step; cols = metric
     BH76 reaction energy / per-species |E-E_PBE| / W4-11 atomization). Each panel
     shows, per constraint level: the random-init bar (mean, with a faint worst-of-
     seeds cap and a bold +/-std whisker) plus the two pretrained bars (unweighted
     and integration). PBE baseline as a dashed line where defined.
  2. A convergence figure: steps-to-converge (xnet & cnet) per constraint level for
     each pretraining weighting.

Usage::
    python notebooks/analysis/make_multimode_figure.py [RESULTS.json] [OUT_PREFIX]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_JSON = os.path.join(_HERE, "demo_logs", "multimode_polarized.json")

_MODES = ("fixed_rho", "one_shot", "3step")
_MODE_LABEL = {"fixed_rho": "fixed-ρ", "one_shot": "one-shot (1 step)",
               "3step": "3-step SCF"}
_METRICS = (
    ("bh76", "BH76 reaction-energy MAE\nvs GMTKN55-BH76RC (kcal/mol)", True),
    ("pbe_dev", "per-species |E_nn-E_PBE| MAE\n(kcal/mol; deviation from PBE)", False),
    ("w411_ae", "atomization-energy MAE\nvs GMTKN55 W4-11 (kcal/mol)", True),
)
_RANDOM = "#c0504d"
_WORST = "#9a9a9a"
_STD = "#5a1714"
_PRE = {"unweighted": "#4f81bd", "integration": "#4bacc6"}
_STYLE = {
    "figure.dpi": 120, "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
    "legend.fontsize": 7, "axes.grid": True, "grid.alpha": 0.3,
    "savefig.bbox": "tight",
}


def short_level(label: str) -> str:
    return label.replace("(x)", "").replace("(c)", "")


def cell_random(results: dict, mode: str, level: str, metric: str) -> dict:
    """Pure accessor: the random mean/worst/std for one (mode, level, metric)."""
    return results["cells"][mode][level]["random"][metric]


def plot_multimode(results: dict, out_path: str) -> str:
    levels = list(next(iter(results["cells"].values())).keys())
    weightings = results["weightings"]
    modes = [m for m in _MODES if m in results["cells"]]
    x = np.arange(len(levels))
    nbar = 1 + len(weightings)
    bw = 0.8 / nbar
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(len(modes), 3, figsize=(16, 4.2 * len(modes)),
                                 squeeze=False)
        for r, mode in enumerate(modes):
            for c, (key, ylab, has_pbe) in enumerate(_METRICS):
                ax = axes[r][c]
                first = (r == 0 and c == 0)
                means = np.array([cell_random(results, mode, lv, key)["mean"]
                                  for lv in levels])
                worst = np.array([cell_random(results, mode, lv, key)["worst"]
                                  for lv in levels])
                stds = np.array([cell_random(results, mode, lv, key)["std"]
                                 for lv in levels])
                x0 = x - 0.4 + bw / 2
                ax.bar(x0, means, bw, color=_RANDOM, zorder=2,
                       label="random init (mean)" if first else None)
                ax.errorbar(x0, means,
                            yerr=[np.zeros_like(means), np.maximum(worst - means, 0)],
                            fmt="none", ecolor=_WORST, elinewidth=1.0, capsize=5,
                            zorder=4, label="worst of seeds" if first else None)
                ax.errorbar(x0, means, yerr=stds, fmt="none", ecolor=_STD,
                            elinewidth=1.6, capsize=3, zorder=5,
                            label="± std (seeds)" if first else None)
                for j, w in enumerate(weightings, start=1):
                    yv = [results["cells"][mode][lv]["pretrained"][w][key]
                          for lv in levels]
                    ax.bar(x - 0.4 + bw * (j + 0.5), yv, bw, color=_PRE.get(w, None),
                           zorder=3, label=f"pretrained [{w}]" if first else None)
                if has_pbe and key in results.get("pbe_baseline", {}):
                    pbe = results["pbe_baseline"][key]
                    ax.axhline(pbe, ls="--", color="k", lw=1.0,
                               label=f"PBE ({pbe:.1f})" if first else None)
                ax.set_xticks(x)
                ax.set_xticklabels([short_level(lv) for lv in levels],
                                   rotation=18, ha="right")
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                if r == 0:
                    ax.set_title(ylab)
                if c == 0:
                    ax.set_ylabel(f"{_MODE_LABEL.get(mode, mode)}\n\nMAE (kcal/mol)")
                if first:
                    ax.set_ylim(top=float(np.max(worst)) * 1.5)
                    ax.legend(loc="upper left", framealpha=0.9)
        fig.suptitle(
            "Self-consistency ladder × constraints × pretraining vs GMTKN55 "
            f"({results.get('config','?')}, {results.get('seeds','?')} seeds)",
            fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def plot_convergence(results: dict, out_path: str) -> str:
    conv = results.get("convergence", {})
    if not conv:
        return ""
    weightings = list(conv.keys())
    levels = list(next(iter(conv.values())).keys())
    x = np.arange(len(levels))
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, len(weightings),
                                 figsize=(7 * len(weightings), 4), squeeze=False)
        for c, w in enumerate(weightings):
            ax = axes[0][c]
            xs = [conv[w][lv].get("steps_to_converge_x") for lv in levels]
            cs = [conv[w][lv].get("steps_to_converge_c") for lv in levels]
            ax.bar(x - 0.2, xs, 0.4, color="#c0504d", label="xnet")
            ax.bar(x + 0.2, cs, 0.4, color="#4f81bd", label="cnet")
            ax.set_xticks(x)
            ax.set_xticklabels([short_level(lv) for lv in levels], rotation=18,
                               ha="right")
            ax.set_title(f"pretraining steps-to-converge [{w}]")
            ax.set_ylabel("steps to reach 1.05× min loss")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.legend()
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    json_path = argv[0] if argv else _DEFAULT_JSON
    prefix = argv[1] if len(argv) > 1 else os.path.join(
        _HERE, "multimode_" + str(json.load(open(json_path)).get("config", "run")))
    with open(json_path) as f:
        results = json.load(f)
    p1 = plot_multimode(results, prefix + "_3x3.png")
    p2 = plot_convergence(results, prefix + "_convergence.png")
    print(f"figures -> {p1}" + (f" , {p2}" if p2 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
