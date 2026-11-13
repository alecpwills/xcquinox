"""The dissociation and fractional-charge figure of a Slim16 evaluation run.

Two panels from the curves ``xcquinox.pipeline.diagnostics`` computes for one
trained network per architecture, the libxc comparators and the exact
references (kept beside the run as ``diagnostics_curves.json``): on the left
the restricted H2 energy against CCSD, exact for two electrons, as a function
of the bond length, where a positive gap at large separation is the
static-correlation error; on the right the H atom's energy against the
straight lines between the exact energies at the integers as a function of
the electron number, where a negative bow is the delocalization error (in
def2-TZVP the anion is unbound, so between one and two electrons a
functional's deviation carries its anion error with its curvature). Both
deviations in kcal/mol. A network draws a line in its architecture's colour
(``arch_style.arch_color``) under its manifest label, solid for the first
network of an architecture and with a dash pattern of its own for each
further one, so that networks sharing a colour read apart; a missing energy
leaves a gap in a line and an unconverged point carries an open circle. A
comparator draws a dashed line in the grey of the Slim16 figure
(``make_slim16_figure``) with a dash pattern of its own, and the reference
is the light zero line. Three title lines explain the panels; ``--plain``
writes a second file without them.

Usage::

    python tools/analysis/make_diagnostics_figure.py <local run dir>
        [--networks a,b] [--r-step 0.1] [--n-step 0.1] [--out PATH] [--plain] [--recompute]

The curves beside the run are reused, a notice on stderr and in the manifest
naming what they hold beyond a plain run's defaults; they are computed
through the module when absent, with ``--recompute``, or when
``--networks``, ``--r-step`` or ``--n-step`` is given. The PNG is written as
``diagnostics.png`` beside the
run unless ``--out`` says otherwise, the plain copy as ``<stem>_plain.png``;
the render manifest (one entry per line with its panel, colour, style,
points and largest deviation, the references, the title lines, the output
paths) is printed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arch_style import arch_color  # noqa: E402
from make_slim16_figure import COMPARATOR_COLORS, PBE_COLOR  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from xcquinox.pipeline import diagnostics  # noqa: E402
from xcquinox.pipeline.eval_holdout import KCAL_PER_HA  # noqa: E402

OUT_NAME = "diagnostics.png"
#: one dash pattern per comparator label, so two greys nine CIEDE2000 units
#: apart still read apart as thin lines
COMPARATOR_DASHES = {"PBE": (0, (4, 2)), "r2SCAN": (0, (1, 1.5)),
                     "B3LYP": (0, (6, 2, 1, 2)), "wB97M-V": (0, (2, 1))}
#: one line style per network of an architecture, by its rank among the
#: payload's networks of that architecture (the first solid), so networks
#: sharing a colour read apart; an architecture with more networks than
#: styles is refused
NETWORK_DASHES = ("solid", (0, (6, 3)), (0, (5, 2, 1, 2)), (0, (3, 1)), (0, (8, 2, 2, 2)),
                  (0, (1, 3)))
REFERENCE_COLOR = "#bbbbbb"
TITLE_LINES = (
    "H2 dissociation and the fractional-charge curve of the H atom at def2-TZVP, grid "
    "level 4: the networks through the converged channel's solver with full integrals "
    "(the Slim16 evaluation's density fitting moves H2 by 0.02 to 0.06 kcal/mol), the "
    "comparators through pyscf",
    "left: the restricted H2 energy against CCSD, exact for two electrons; a positive gap "
    "at large separation is the static-correlation error",
    "right: the H atom's energy against the straight lines between E(0) = 0, E(1) (UHF, "
    "exact) and E(2) (CCSD, exact; the anion is unbound in this basis); a negative bow "
    "is the delocalization error",
)


def comparator_color(label: str) -> str:
    """The grey of a comparator label: PBE-DF's for PBE, the Slim16 figure's
    for the three others; a label that is neither a network of the payload
    nor a comparator is refused by name."""
    by_label = {text: xc for xc, text in diagnostics.COMPARATORS}
    if label not in by_label:
        raise ValueError(f"not a network of the payload and not a comparator: {label!r}")
    key = by_label[label]
    return PBE_COLOR if key == "pbe" else COMPARATOR_COLORS[key]


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and \
        math.isfinite(value)


def deviations(payload: dict) -> list:
    """One entry per curve and panel: the label, its kind (``network`` or
    ``comparator``), the panel (``h2`` or ``h``), the x values, the
    deviation from the panel's reference in kcal/mol (None where the energy
    or the reference is missing or not finite) and the convergence flag per
    point."""
    out = []
    networks = set(payload["networks"])
    for panel, x_key, ref_key in (("h2", "r_bohr", "reference_ccsd"),
                                  ("h", "n", "reference_linear")):
        block = payload[panel]
        x = list(block[x_key])
        reference = list(block[ref_key])
        for label in _curve_order(payload):
            if label not in block["curves"]:
                continue
            curve = block["curves"][label]
            kind = "network" if label in networks else "comparator"
            y = [(e - r) * KCAL_PER_HA if _finite(e) and _finite(r) else None
                 for e, r in zip(curve["E"], reference)]
            converged = list(curve.get("converged") or [True] * len(x))
            out.append({"label": label, "kind": kind, "panel": panel, "x": x, "y_kcal": y,
                        "converged": [bool(c) for c in converged]})
    return out


def _curve_order(payload: dict) -> list:
    """The labels in drawing (and legend) order, independent of the order
    the payload holds them in (its writer sorts keys): the networks by
    their manifest index, then the comparators in the module's order, then
    any other label alphabetically."""
    networks = payload["networks"]
    labels = list(dict.fromkeys(list(payload["h2"]["curves"]) + list(payload["h"]["curves"])))
    comparators = [label for _xc, label in diagnostics.COMPARATORS]
    by_index = sorted([label for label in labels if label in networks],
                      key=lambda label: (int(networks[label].get("index", 0)), label))
    ordered = by_index + [label for label in comparators if label in labels]
    ordered += sorted(label for label in labels
                      if label not in networks and label not in comparators)
    return ordered


def _ranks(networks: dict) -> dict:
    """Each network's rank among the payload's networks of its architecture
    by manifest index (the line style's index), independent of the order
    the payload holds them in; an architecture with more networks than
    there are styles is refused by name."""
    count: dict = {}
    ranks = {}
    ordered = sorted(networks.items(), key=lambda item: (int(item[1].get("index", 0)), item[0]))
    for label, entry in ordered:
        ranks[label] = count.get(entry["arch_name"], 0)
        count[entry["arch_name"]] = ranks[label] + 1
    crowded = sorted(arch for arch, n in count.items() if n > len(NETWORK_DASHES))
    if crowded:
        raise ValueError(f"more networks of {crowded} than line styles "
                         f"({max(count.values())} > {len(NETWORK_DASHES)}): pass fewer "
                         "--networks")
    return ranks


def plot_diagnostics(payload: dict, out_path, plain: bool = False) -> dict:
    """Draw the two panels and return the render manifest. A missing or
    non-finite energy leaves a gap in its line; an unconverged point carries
    an open circle."""
    networks = payload["networks"]
    ranks = _ranks(networks)
    fig, (ax_r, ax_n) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    axes = {"h2": ax_r, "h": ax_n}
    curves = []
    for entry in deviations(payload):
        ax = axes[entry["panel"]]
        if entry["kind"] == "network":
            color = arch_color(networks[entry["label"]]["arch_name"])
            rank = ranks[entry["label"]]
            linestyle = NETWORK_DASHES[rank]
        else:
            color = comparator_color(entry["label"])
            rank = None
            linestyle = COMPARATOR_DASHES.get(entry["label"], (0, (4, 2)))
        y = [float("nan") if v is None else v for v in entry["y_kcal"]]
        ax.plot(entry["x"], y, color=color, linestyle=linestyle, linewidth=1.4,
                label=entry["label"])
        unconverged = [(xi, yi) for xi, yi, c in zip(entry["x"], y, entry["converged"])
                       if not c and math.isfinite(yi)]
        if unconverged:
            ax.plot([p[0] for p in unconverged], [p[1] for p in unconverged],
                    linestyle="none", marker="o", markersize=5, markerfacecolor="none",
                    markeredgecolor=color, markeredgewidth=0.9)
        finite = [v for v in y if math.isfinite(v)]
        curves.append({**entry, "color": color, "rank": rank,
                       "linestyle": "solid" if linestyle == "solid" else "dashed",
                       "dashes": [] if linestyle == "solid" else list(linestyle[1]),
                       "unconverged_x": [p[0] for p in unconverged],
                       "missing_x": [xi for xi, yi in zip(entry["x"], y)
                                     if not math.isfinite(yi)],
                       "max_abs_deviation": max((abs(v) for v in finite), default=None)})
    x_r = payload["h2"]["r_bohr"]
    x_n = payload["h"]["n"]
    ax_r.plot([x_r[0], x_r[-1]], [0.0, 0.0], color=REFERENCE_COLOR, linewidth=1.0,
              label="CCSD (exact)", zorder=1)
    ax_n.plot([x_n[0], x_n[-1]], [0.0, 0.0], color=REFERENCE_COLOR, linewidth=1.0,
              label="linear between the integers (exact)", zorder=1)
    ax_r.set_xlabel("H-H distance (bohr)")
    ax_r.set_ylabel("E - E_CCSD (kcal/mol)")
    ax_n.set_xlabel("electron number N of the H atom")
    ax_n.set_ylabel("E - E_linear (kcal/mol)")
    for ax in (ax_r, ax_n):
        ax.yaxis.grid(True, color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    if not plain:
        fig.suptitle("\n".join(TITLE_LINES), fontsize=8, y=0.995, va="top", wrap=True)
        fig.subplots_adjust(top=0.78, bottom=0.14, left=0.08, right=0.98, wspace=0.28)
    else:
        fig.subplots_adjust(top=0.95, bottom=0.14, left=0.08, right=0.98, wspace=0.28)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return {"out": str(out_path), "plain": bool(plain), "curves": curves,
            "references": {"h2": "CCSD", "h": "linear between the integers",
                           "integer_energies": payload["h"]["integer_energies"]},
            "title_lines": [] if plain else list(TITLE_LINES),
            "r_bohr": list(x_r), "n": list(x_n)}


def payload_mismatch(payload: dict, run_dir: Path):
    """What a payload beside the run holds that differs from a plain run's
    defaults (the default grids, one network per architecture), or None;
    a plain run redraws such a payload and says so."""
    lo_r, hi_r, step_r = diagnostics.R_BOHR_DEFAULT
    lo_n, hi_n, step_n = diagnostics.N_DEFAULT
    identity = payload.get("identity", {})
    found = []
    if [float(v) for v in identity.get("r_bohr", [])] != diagnostics.grid(lo_r, hi_r, step_r):
        found.append(f"bond lengths {identity.get('r_bohr')}")
    if [float(v) for v in identity.get("n", [])] != diagnostics.grid(lo_n, hi_n, step_n):
        found.append(f"electron numbers {identity.get('n')}")
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        try:
            default = list(diagnostics.select_networks(manifest))
        except ValueError as exc:
            # a manifest the default selection refuses (a network without an
            # architecture name) still has its payload drawn, the refusal named
            found.append(f"a manifest whose default selection cannot be read ({exc})")
            default = None
        if default is not None and list(payload.get("networks", {})) != default:
            found.append(f"networks {list(payload.get('networks', {}))} (the default "
                         f"selection is {default})")
    return "; ".join(found) if found else None


def load_or_compute(run_dir: Path, networks, r_step, n_step, recompute: bool) -> tuple:
    """``(payload, notice)``: the curves beside the run, computed through the
    module when absent, when asked for, or when the networks or a step are
    given, and written back. A plain run redraws the payload beside the run
    whatever it was computed for, and the notice names what differs from
    the defaults (the networks, the grids) and how to replace it; None when
    nothing differs or the payload was computed now."""
    path = run_dir / diagnostics.CURVES_FILE
    chosen = networks is not None or r_step is not None or n_step is not None
    if path.is_file() and not recompute and not chosen:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mismatch = payload_mismatch(payload, run_dir)
        notice = (f"{path} holds curves computed for {mismatch}; drawn as they are "
                  "(--recompute replaces them with the defaults)" if mismatch else None)
        return payload, notice
    # the style capacity is checked before the computation, not after it
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    _ranks(diagnostics.select_networks(manifest, networks))
    lo_r, hi_r, step_r = diagnostics.R_BOHR_DEFAULT
    lo_n, hi_n, step_n = diagnostics.N_DEFAULT
    payload = diagnostics.compute_diagnostics(
        run_dir, labels=networks,
        r_values=diagnostics.grid(lo_r, hi_r, step_r if r_step is None else r_step),
        n_values=diagnostics.grid(lo_n, hi_n, step_n if n_step is None else n_step))
    diagnostics.write_payload(path, payload)
    return payload, None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--networks", default=None,
                        help="comma-separated manifest labels (default: the first "
                             "network of each architecture); implies a recomputation")
    parser.add_argument("--r-step", type=float, default=None,
                        help="the bond-length step in bohr (default %g); implies a "
                             "recomputation" % diagnostics.R_BOHR_DEFAULT[2])
    parser.add_argument("--n-step", type=float, default=None,
                        help="the electron-number step (default %g); implies a "
                             "recomputation" % diagnostics.N_DEFAULT[2])
    parser.add_argument("--out", default=None, help="output PNG (default: <run_dir>/%s)" % OUT_NAME)
    parser.add_argument("--plain", action="store_true",
                        help="write a second copy without the title lines")
    parser.add_argument("--recompute", action="store_true",
                        help="recompute the curves beside the run")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    networks = ([s.strip() for s in args.networks.split(",") if s.strip()]
                if args.networks else None)
    payload, notice = load_or_compute(run_dir, networks, args.r_step, args.n_step,
                                      args.recompute)
    if notice:
        print(f"notice: {notice}", file=sys.stderr)
    out = Path(args.out) if args.out else run_dir / OUT_NAME
    manifest = plot_diagnostics(payload, out)
    manifest["notice"] = notice
    if args.plain:
        plain_out = out.with_name(out.stem + "_plain" + out.suffix)
        manifest["out_plain"] = plot_diagnostics(payload, plain_out, plain=True)["out"]
    print(json.dumps(diagnostics.json_safe(manifest), indent=2))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
