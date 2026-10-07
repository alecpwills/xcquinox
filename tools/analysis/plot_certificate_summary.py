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

A second panel under the first draws the certificate's other gate on the same
architecture axis: the largest |dE_xc| over the free atoms, in mHa, against
each certificate's recorded ``tol_atom``. A certificate can fail on its free
atoms alone, with its atomization statistics inside both atomization gates,
and the first panel then shows a hatched bar with no reason in sight. The
hatch is the certificate's verdict in both panels, whichever gate it failed,
so a certificate that fails on its atomization energies alone has a hatched
bar under the atom gate. A certificate with no finite free-atom value draws
no bar there and is noted with its verdict.

A third panel draws the potential record written beside the certificate
(``fidelity_vxc.json``, ``cluster.fidelity.evaluate_potential``): the largest
over the oracle systems of the action of the clone's potential minus the
parent's on an occupied orbital of the reference density, in mHa, at the same
x position and in the same color as the certificate's bars, hatched by the
energy verdict (the record gates nothing). The panel is drawn when at least
one certificate of the figure carries a record; a figure of certificates
from before the record keeps the two energy panels. Where the panel is
drawn, a certificate with no potential record beside it is noted ``no
potential record``; a record whose checkpoint digests differ from the
certificate's belongs to another fit and is noted ``potential record of
another checkpoint``. The CSV carries the record's largest potential
difference, largest occupied-eigenvalue shift and largest one-step drift,
empty where no record exists.

Statistics are recomputed here from ``per_atomization`` (rows with a null
``dAE_kcalmol`` skipped), so certificates written before the summary carried
``mean_dAE_kcalmol`` / ``rmse_dAE_kcalmol`` / ``species_over_1_kcalmol``
plot identically to regated ones; a ``regate`` provenance block, when
present, is ignored beyond not being an error. A CSV with the same numbers
is written beside the PNG (same basename).

The upper panel's y axis is linear and capped a little above the tallest gate
so the gate region stays readable next to multi-kcal/mol legacy outliers; a
per-species max beyond the cap is drawn as an up-pointing marker at the axis
edge with the number printed beside it. The lower panel holds bars only, and
its range covers every bar and every gate line.

Usage:
    python tools/analysis/plot_certificate_summary.py \\
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

# the certificate directory is the STORED registry key; the axis and the CSV
# show the derived name (medium -> medium_3x16, deep -> deep_4x32; a key that
# states its size, deep_3x16, is its own shown name)
from xcquinox.pipeline.arch_names import display_name  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_style import ARCH_ORDER  # noqa: E402


def _display_order(stored_keys):
    """The stored keys in the display order of their shown names (ARCH_ORDER;
    a name outside it last, in sorted order)."""
    def _key(a):
        shown = display_name(a)
        try:
            return (ARCH_ORDER.index(shown), shown)
        except ValueError:
            return (len(ARCH_ORDER), shown)
    return sorted(stored_keys, key=_key)

# Categorical palette: slot 1 blue, slot 2 orange, then aqua/yellow for
# further labels. Color follows the LABEL (the campaign generation); the FAIL
# state is carried by hatching and the verdict text, never by color alone.
# Separation was checked by execution with an OKLab-based colorblind
# validator (Delta E x100 in OKLab, light surface #fcfcfb): worst adjacent
# pair 24.7 protan / 33.6 normal for the first two slots. Note the METRIC:
# under CIEDE2000 with the Vienot 1999 protan model the same pair measures
# ~48.5 normal / ~57.3 protan -- different formulations, both comfortably
# above their guidelines; any quoted number must name its metric. The second
# four (brown, pink, grey, dark grey) extend the palette to eight labels, a
# figure of several pretraining rounds beside their predecessors; they were
# chosen from a candidate list by the smallest pairwise CIEDE2000 (Sharma, Wu
# and Dalal 2005) over every pair under normal vision and under the Vienot
# 1999 protan and deutan simulations: the eight colors' closest pair
# measures 18.8 under normal vision (the two greys), 9.7 under protan (blue
# against pink) and 9.4 under deutan (the first four's orange against
# yellow, the palette's floor before and after the extension). More labels
# than colors are refused; color follows the label and is never reused.
_LABEL_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#8c564b", "#e377c2", "#7f7f7f", "#4d4d4d")

# The per-species flag threshold the certificates record (kcal/mol; the
# original per-species gate). Stated here only as a fallback for certificates
# written before the summary carried the species list.
_SPECIES_FLAG_KCALMOL = 1.0


def _certificate_paths(run_dir):
    """The certificate files ``run_dir`` plots, through the one resolution
    rule: an architecture the run's resolved config warm-starts from a donor
    reads the DONOR's certificate, every other architecture the run-local
    ``pretrain/<arch>`` product (``grid_config.pretrain_checkpoint_for``,
    the rule every consumer shares). Without a loadable config the run-local
    glob stands alone, as it always did.
    """
    run_local = sorted(glob.glob(
        os.path.join(run_dir, "pretrain", "*", "fidelity_certificate.json")))
    try:
        from xcquinox.pipeline.cluster.grid_config import (
            load_resolved_run_config, pretrain_checkpoint_for)
        cfg = load_resolved_run_config(run_dir)
        donors = getattr(cfg.pretrain, "donor_checkpoints", None) or {}
    except Exception:  # noqa: BLE001 -- no readable config: run-local glob
        return run_local
    # The arch universe: what the run-local glob found plus the donated
    # names, each then resolved through the rule -- so a donated arch plots
    # its donor's certificate even when the run wrote no pretrain tree, and
    # a leftover run-local file under a donated name is not plotted beside
    # the donor's as a second word on the same arch.
    by_arch = {os.path.basename(os.path.dirname(p)): p
               for p in run_local}
    paths = []
    for arch in sorted(set(by_arch) | set(donors)):
        p = os.path.join(pretrain_checkpoint_for(cfg, run_dir, arch),
                         "fidelity_certificate.json")
        if os.path.isfile(p):
            paths.append(p)
    return sorted(paths)


def _max_atom_mha(cert):
    """The largest |dE_xc| over the certificate's free atoms, in mHa: the
    value its summary records (the one the verdict was formed from), else
    the largest over the ``per_system`` rows marked ``is_atom`` (a row
    without a value skipped). ``None`` when the certificate carries neither,
    and when a value read is not finite: a NaN or an infinity is a failed
    measurement, with no height to draw and no largest magnitude to state
    (``max`` over a NaN depends on the order of the rows)."""
    recorded = (cert.get("summary") or {}).get("max_atom_mHa")
    if recorded is not None:
        values = [float(recorded)]
    else:
        values = [float(r["dE_xc_mHa"])
                  for r in cert.get("per_system") or []
                  if isinstance(r, dict) and r.get("is_atom")
                  and r.get("dE_xc_mHa") is not None]
    if not values or not all(math.isfinite(v) for v in values):
        return None
    return max(abs(v) for v in values)


#: The keys of the checkpoint block a record is matched to its certificate
#: by: the digests of the two networks, never the directory string, which
#: another spelling of the same run directory (a symbolic link, a moved
#: run) changes.
_CHECKPOINT_DIGESTS = ("xnet_sha256", "cnet_sha256")


def _potential_summary(certificate_path, certificate):
    """``(values, note, present)`` of the potential record beside
    ``certificate_path``: the three summary values the tool reads, each a
    finite float or ``None``; a note for the panel where no measured value
    of this checkpoint exists (the record absent or unreadable, a record
    whose checkpoint digests differ from the certificate's, a record whose
    every system failed), the values empty then; and whether a record of
    this checkpoint exists at all, which decides whether the panel is drawn."""
    path = os.path.join(os.path.dirname(certificate_path), "fidelity_vxc.json")
    try:
        with open(path) as f:
            record = json.load(f)
    except (OSError, ValueError):
        return {}, "no potential record", False
    if not isinstance(record, dict):
        return {}, "no potential record", False
    recorded = record.get("checkpoint") or {}
    stated = certificate.get("checkpoint") or {}
    if any(recorded.get(key) != stated.get(key) for key in _CHECKPOINT_DIGESTS):
        return {}, "potential record of another checkpoint", True
    summary = record.get("summary") or {}
    out = {}
    for key in ("max_dV_occ_mHa", "max_deps_occ_mHa",
                "max_dE_one_step_kcalmol"):
        value = summary.get(key)
        out[key] = (float(value) if isinstance(value, (int, float))
                    and math.isfinite(float(value)) else None)
    if all(value is None for value in out.values()):
        n_failed = summary.get("n_failed")
        n_systems = len(record.get("per_system") or [])
        return {}, (f"potential record: {n_failed} of {n_systems} systems "
                    "failed, none measured"), True
    return out, None, True


def collect_certificates(runs):
    """``[(label, arch, record)]`` for every certificate under ``runs``.

    ``runs`` is a list of ``(label, run_dir)`` pairs; each run's
    certificates resolve through the donor rule (:func:`_certificate_paths`),
    so a donor-backed architecture plots its donor's numbers. Each record
    carries the recomputed statistics plus the recorded verdict and
    tolerances. A duplicate (label, arch) pair and a run directory with no
    certificate are both refused: the first silently averages two campaigns
    into one bar, the second draws an empty axis that reads as a clean sweep.
    """
    out = []
    seen = set()
    for label, run_dir in runs:
        paths = _certificate_paths(run_dir)
        if not paths:
            raise ValueError(
                f"no fidelity_certificate.json under {run_dir} (run-local "
                "pretrain/*/ or a config-stated donor)")
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
            potential, potential_note, potential_present = _potential_summary(
                path, cert)
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
                # the free-atom gate: the gated value (None without a finite
                # one) and the certificate's own tolerance (None where it
                # records none)
                "max_atom": _max_atom_mha(cert),
                "tol_atom": tol.get("tol_atom"),
                # the potential record beside the certificate: None without
                # a measured value, and the panel's note then; whether a
                # record of this checkpoint exists at all
                "max_dV": potential.get("max_dV_occ_mHa"),
                "max_deps": potential.get("max_deps_occ_mHa"),
                "max_drift": potential.get("max_dE_one_step_kcalmol"),
                "potential_note": potential_note,
                "potential_present": potential_present,
            }))
    return out


def write_csv(records, path):
    """The figure's numbers as one row per (label, arch); ``arch`` is the
    shown name and ``arch_stored`` the certificate directory."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "arch", "arch_stored", "verdict",
                    "n_atomizations",
                    "mean_abs_dAE_kcalmol", "rmse_dAE_kcalmol",
                    "max_abs_dAE_kcalmol", "species_over_1_kcalmol",
                    "tol_AE", "tol_AE_aggregate", "tol_AE_max_backstop",
                    "max_atom_mHa", "tol_atom",
                    "max_dV_occ_mHa", "max_deps_occ_mHa",
                    "max_dE_one_step_kcalmol"])
        for label, arch, r in records:
            w.writerow([label, display_name(arch), arch, r["verdict"], r["n"],
                        r["mean"], r["rmse"], r["max"],
                        ";".join(r["species_over"]),
                        r["tol_AE"], r["aggregate"], r["backstop"],
                        r["max_atom"], r["tol_atom"],
                        r["max_dV"], r["max_deps"], r["max_drift"]])


def plot_certificate_summary(records, out_path):
    """Render the grouped mean-bar / max-marker figure to ``out_path``, the
    free-atom panel under it.

    Returns the render manifest: a record, written beside each draw call, of
    the gate lines with their rule text, the hatched FAIL bars, the clipped
    max markers with their values, the note texts, the per-label colors and
    the y cap. The lower panel's entries carry the ``atom_`` prefix: its
    bars by (label, arch), the hatched ones, one gate line per distinct
    recorded ``tol_atom``, the records noted as carrying no free-atom value,
    and its y cap. The third panel's entries carry the ``vxc_`` prefix: its
    bars (the record's largest potential difference, mHa) and its markers
    (the largest occupied-level shift, mHa) by (label, arch), the notes of
    the certificates drawn without a bar, and its y cap. ``vxc_panel``
    states whether the panel was drawn at all: it is drawn when a potential
    record exists beside at least one certificate of the figure; without
    one the bars, markers and notes are empty and the cap is ``None``.
    """
    labels = []
    for label, _arch, _r in records:
        if label not in labels:
            labels.append(label)
    # Colour follows the label: more labels than the palette holds would
    # reuse a colour, so such a figure is refused rather than drawn.
    if len(labels) > len(_LABEL_COLORS):
        raise ValueError(
            f"{len(labels)} labels for a palette of {len(_LABEL_COLORS)} "
            "colours; colour follows the label, so a figure with more labels "
            "than colours is not drawn")
    # the axis runs in the figures' display order of the SHOWN names (the
    # ticks are labelled with them), not in the sorted order of the
    # directory keys
    archs = _display_order({arch for _l, arch, _r in records})
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

    # The free-atom panel: one gate line per distinct recorded tol_atom, and a
    # range that covers every bar and every gate line (a clipped bar would
    # misstate the one number the atom gate judges).
    atom_gates = sorted({float(r["tol_atom"]) for _l, _a, r in records
                         if r["tol_atom"] is not None})
    atom_values = [r["max_atom"] for _l, _a, r in records
                   if r["max_atom"] is not None]
    atom_cap = 1.25 * max(atom_values + atom_gates, default=1.0)
    # The potential panel: no gate line (the record gates nothing); a range
    # that covers every bar and every marker. It is drawn only where a record
    # exists beside some certificate of the figure: a figure of certificates
    # from before the record keeps the two energy panels instead of a panel
    # of notes.
    vxc_values = [v for _l, _a, r in records
                  for v in (r["max_dV"], r["max_deps"]) if v is not None]
    has_potential = any(r.get("potential_present") for _l, _a, r in records)
    vxc_cap = (1.25 * max(vxc_values, default=1.0)) if has_potential else None

    n_labels = max(len(labels), 1)
    group_w = 0.8
    bar_w = group_w / n_labels
    fig_w = max(7.5, 1.05 * len(archs) * n_labels + 2.5)
    if has_potential:
        fig, (ax, ax_atom, ax_vxc) = plt.subplots(
            3, 1, figsize=(fig_w, 10.6), sharex=True,
            gridspec_kw={"height_ratios": [3, 2, 2]})
    else:
        fig, (ax, ax_atom) = plt.subplots(
            2, 1, figsize=(fig_w, 8.2), sharex=True,
            gridspec_kw={"height_ratios": [3, 2]})
        ax_vxc = None

    # The render manifest is written beside each draw call and returned: the
    # gate lines and their rule text, FAIL hatching, clipped markers with
    # their values, note text, per-label colors and the bar-covering caps,
    # so a caller reads what the figure states without parsing pixels.
    manifest = {"out_path": out_path, "y_cap": y_cap, "gate_lines": [],
                "hatched": [], "clipped": [], "colors": {}, "notes": {},
                "atom_bars": {}, "atom_hatched": [], "atom_gate_lines": [],
                "atom_notes": {}, "atom_y_cap": atom_cap,
                "vxc_panel": has_potential, "vxc_bars": {}, "vxc_markers": {},
                "vxc_notes": {}, "vxc_y_cap": vxc_cap}

    for li, label in enumerate(labels):
        color = _LABEL_COLORS[li % len(_LABEL_COLORS)]
        manifest["colors"][label] = color
        for ai, arch in enumerate(archs):
            r = by_key.get((label, arch))
            if r is None:
                continue
            x = ai - group_w / 2 + (li + 0.5) * bar_w
            hatch = "///" if r["verdict"] != "PASS" else None
            # The lower panel first: it is drawn for every record, whether or
            # not the record has atomization rows for the upper one. A record
            # without a free-atom value has no bar to hatch, so its note
            # carries the verdict, as the upper panel's does; the reason is
            # wrapped so that the notes of neighbouring records stay apart.
            if r["max_atom"] is None:
                text = f"{r['verdict']}\nno free-atom\nvalue"
                ax_atom.annotate(text, (x, 0.0), xytext=(0, 6),
                                 textcoords="offset points", ha="center",
                                 fontsize=7, color="#444444", zorder=5)
                manifest["atom_notes"][(label, arch)] = text
            else:
                if hatch:
                    manifest["atom_hatched"].append((label, arch))
                ax_atom.bar(x, r["max_atom"], width=0.92 * bar_w, color=color,
                            hatch=hatch, edgecolor="white", linewidth=0.5,
                            zorder=3)
                manifest["atom_bars"][(label, arch)] = r["max_atom"]
            # The potential panel, where drawn: the record's largest potential
            # difference as the bar, hatched by the energy verdict, and its
            # largest occupied-level shift as the marker above it (a
            # difference acting on the virtual space alone shows there and
            # not in the bar); a note where no measured record of this
            # checkpoint exists.
            if ax_vxc is None:
                pass
            elif r["max_dV"] is None:
                text = r.get("potential_note") or "no potential record"
                ax_vxc.annotate(text, (x, 0.0), xytext=(0, 6),
                                textcoords="offset points", ha="center",
                                fontsize=7, color="#444444", zorder=5)
                manifest["vxc_notes"][(label, arch)] = text
            else:
                ax_vxc.bar(x, r["max_dV"], width=0.92 * bar_w, color=color,
                           hatch=hatch, edgecolor="white", linewidth=0.5,
                           zorder=3)
                manifest["vxc_bars"][(label, arch)] = r["max_dV"]
                if r["max_deps"] is not None:
                    ax_vxc.plot([x, x], [r["max_dV"], r["max_deps"]],
                                color=color, linewidth=1.0, alpha=0.55,
                                zorder=3)
                    ax_vxc.plot([x], [r["max_deps"]], marker="D",
                                markersize=5, color=color,
                                markeredgecolor="white", markeredgewidth=0.5,
                                zorder=4)
                    manifest["vxc_markers"][(label, arch)] = r["max_deps"]
            if r["mean"] is None:
                # A certificate with no usable atomization rows still shows:
                # an unmarked gap would read as a clean absence.
                text = f"{r['verdict']}\nno atomization data"
                ax.annotate(text, (x, 0.0), xytext=(0, 6),
                            textcoords="offset points", ha="center",
                            fontsize=7, color="#444444", zorder=5)
                manifest["notes"][(label, arch)] = text
                continue
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

    for value in atom_gates:
        text = f"tol_atom = {value:g} (gates every free atom)"
        ax_atom.axhline(value, zorder=2, color="#555555", linestyle="--",
                        linewidth=1.2)
        ax_atom.annotate(text, (len(archs) - 0.52, value),
                         xytext=(0, 3), textcoords="offset points",
                         ha="right", fontsize=8, color="#555555")
        manifest["atom_gate_lines"].append((value, text))

    # The panels share the architecture axis; the tick labels sit on the
    # lowest one.
    lowest = ax_atom if ax_vxc is None else ax_vxc
    lowest.set_xticks(range(len(archs)))
    lowest.set_xticklabels([display_name(a) for a in archs], rotation=20,
                           ha="right", fontsize=9)
    ax.set_xlim(-0.6, len(archs) - 0.4)
    ax.set_ylim(0.0, y_cap)
    ax.set_ylabel("|dAE| vs parent (kcal/mol)")
    ax_atom.set_ylim(0.0, atom_cap)
    ax_atom.set_ylabel("max |dE_xc| over free atoms (mHa)")
    if ax_vxc is not None:
        ax_vxc.set_ylim(0.0, vxc_cap)
        ax_vxc.set_ylabel("max |dV_xc| on occupied orbitals (bar),\n"
                          "max occupied-level shift (marker) (mHa)")
    for axis in ((ax, ax_atom) if ax_vxc is None else (ax, ax_atom, ax_vxc)):
        axis.yaxis.grid(True, color="#dddddd", linewidth=0.7, zorder=0)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
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
